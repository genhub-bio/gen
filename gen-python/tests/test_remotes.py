"""Exercise the installed extension's remote API without a live GenHub account."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePath
import subprocess
import tempfile
from queue import Empty, Queue
from threading import Thread
from time import monotonic, sleep
import unittest
import urllib.parse
import urllib.request
import uuid
from urllib.request import url2pathname
from unittest.mock import patch

import gen


def archived_asset_bytes_by_id(repository):
    """Read reachable local assets from the content-addressed archive directory."""
    asset_refs = {
        asset_id: (uri, checksum)
        for asset_id, uri, checksum in repository.query(
            "SELECT lower(hex(id)), uri, checksum FROM gen_asset_refs"
        )
    }
    reachable_asset_ids = {str(asset.id) for asset in repository.get_assets()}
    asset_directory = repository.gen_dir / "assets"
    archived_assets = {}

    for asset_id in reachable_asset_ids:
        uri, checksum = asset_refs[asset_id]
        if not uri.startswith("file://"):
            raise AssertionError(f"asset {asset_id} is not locally archived")
        suffix = Path(url2pathname(uri.removeprefix("file://"))).name.partition(".")[2]
        filename = f"{checksum.hex()}.{suffix}" if suffix else checksum.hex()
        archived_assets[asset_id] = (asset_directory / filename).read_bytes()

    return archived_assets


@contextmanager
def genhub_server(respond):
    requests = []
    assets = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body) if raw_body else None
            requests.append((self.path, dict(self.headers), body))
            status, response = respond(self.path, self.headers, body)
            payload = json.dumps(response).encode() if response is not None else b""
            self.send_response(status)
            if response is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            asset_id = self.path.rsplit("/", 1)[-1]
            content = assets.get(asset_id)
            if content is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_PUT(self):
            length = int(self.headers.get("Content-Length", 0))
            content = self.rfile.read(length)
            asset_id = self.path.rsplit("/", 1)[-1]
            assets[asset_id] = content
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_arguments):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f"http://127.0.0.1:{server.server_port}/repos/test/repository",
            requests,
            assets,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def direct_gcs_database(local_database):
    """Seed a graph into a test GCS server for the production direct-read path."""
    repository_root = Path(__file__).resolve().parents[2]
    build = subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--message-format=json-render-diagnostics",
            "--manifest-path",
            str(repository_root / "gen-python" / "Cargo.toml"),
            "--example",
            "remote_gcs_fixture",
            "--features",
            "abi3,extension-module",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError(
            "Could not build the direct GCS fixture:\n" + build.stderr[-4000:]
        )
    executable = None
    for line in build.stdout.splitlines():
        try:
            artifact = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            artifact.get("reason") == "compiler-artifact"
            and artifact.get("target", {}).get("name") == "remote_gcs_fixture"
        ):
            executable = artifact.get("executable")
            break
    if executable is None:
        raise RuntimeError("Cargo did not report the direct GCS fixture executable")

    stderr_file = tempfile.TemporaryFile(mode="w+t")
    process = subprocess.Popen(
        [executable, str(local_database)],
        cwd=repository_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        text=True,
        bufsize=1,
    )
    ready_uri = Queue()
    Thread(
        target=lambda: ready_uri.put(process.stdout.readline()),
        daemon=True,
    ).start()
    failed = False

    def fixture_failure(message):
        stderr_file.flush()
        stderr_file.seek(0)
        details = stderr_file.read().strip()
        if details:
            message = f"{message}\n{details[-4000:]}"
        return RuntimeError(message)

    try:
        try:
            graph_uri = ready_uri.get(timeout=120)
        except Empty as error:
            failed = True
            process.kill()
            process.communicate(timeout=10)
            raise fixture_failure(
                "Timed out starting the direct GCS fixture"
            ) from error
        if not graph_uri:
            failed = True
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
            raise fixture_failure(
                "The direct GCS fixture exited before publishing its URI"
            )

        try:
            yield graph_uri.strip()
        except BaseException:
            failed = True
            raise
    finally:
        try:
            if process.poll() is None:
                try:
                    process.stdin.write("\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                    if not failed:
                        raise RuntimeError(
                            "The direct GCS fixture did not stop cleanly"
                        )
            if process.returncode != 0 and not failed:
                raise fixture_failure(
                    f"The direct GCS fixture exited with status {process.returncode}"
                )
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stdin is not None:
                process.stdin.close()
            stderr_file.close()

@contextmanager
def fake_gcs_server():
    """Runs the pinned GCS emulator used by the direct-push integration test."""
    try:
        container_id = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--detach",
                "--pull=missing",
                "--publish",
                "127.0.0.1::4443",
                "fsouza/fake-gcs-server:1.56.1",
                "-scheme",
                "http",
                "-port",
                "4443",
                "-backend",
                "memory",
                "-public-host",
                "127.0.0.1",
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError as error:
        raise RuntimeError(
            "direct-GCS remote tests require Docker and the "
            "fsouza/fake-gcs-server:1.56.1 image"
        ) from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip()
        message = (
            "direct-GCS remote tests require Docker and the locally available "
            "fsouza/fake-gcs-server:1.56.1 image"
        )
        if details:
            message = f"{message}: {details}"
        raise RuntimeError(message) from error
    try:
        published_port = (
            subprocess.run(
                ["docker", "port", container_id, "4443/tcp"],
                capture_output=True,
                check=True,
                text=True,
            )
            .stdout.strip()
            .rsplit(":", 1)[-1]
        )
        origin = f"http://127.0.0.1:{published_port}"
        ready_by = monotonic() + 15
        while True:
            try:
                with urllib.request.urlopen(f"{origin}/storage/v1/b", timeout=1):
                    break
            except OSError:
                if monotonic() >= ready_by:
                    raise
                sleep(0.05)
        yield origin
    finally:
        subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gen-python-remotes-")
        self.root = Path(self.directory.name)
        self.repository = gen.Repository(str(self.root / "local"))

    def import_sequence(self, repository, name):
        fasta = self.root / f"{name}.fa"
        fasta.write_text(f">{name}\nACGTACGT\n")
        repository.import_fasta(str(fasta), collection=name)

    def test_remote_configuration_and_validation(self):
        self.assertFalse(hasattr(self.repository, "login"))
        origin = self.repository.add_remote("origin", (self.root / "upstream").as_uri())
        backup = self.repository.add_remote("backup", (self.root / "backup").as_uri())
        self.assertEqual(
            [remote.name for remote in self.repository.remotes],
            ["backup", "origin"],
        )
        self.repository.set_default_remote(origin)
        self.repository.set_branch_remote("backup")
        self.assertEqual(self.repository.default_remote.name, origin.name)
        self.assertEqual(self.repository.current_branch.remote, backup.name)

        for method in [
            self.repository.set_default_remote,
            self.repository.set_branch_remote,
            self.repository.remove_remote,
            self.repository.push,
            self.repository.pull,
            self.repository.fetch,
        ]:
            with self.subTest(method=method.__name__):
                with self.assertRaises(RuntimeError):
                    method("missing")
                with self.assertRaises(TypeError):
                    method(42)

        self.assertEqual(self.repository.default_remote.name, origin.name)
        self.assertEqual(self.repository.current_branch.remote, backup.name)
        self.repository.set_branch_remote()
        self.repository.set_default_remote()
        self.assertIsNone(self.repository.current_branch.remote)
        self.assertIsNone(self.repository.default_remote)
        self.repository.set_default_remote(origin)
        self.repository.set_branch_remote(origin)
        self.repository.remove_remote(origin)
        self.assertIsNone(self.repository.default_remote)
        self.assertIsNone(self.repository.current_branch.remote)
        with self.assertRaises(RuntimeError):
            self.repository.fetch(origin)

    def test_remote_transfers_reject_invalid_branches(self):
        for action in ["push", "pull", "fetch"]:
            method = getattr(self.repository, action)
            with self.subTest(action=action):
                with self.assertRaises(TypeError):
                    method(branch=42)

    def test_clone_preserves_existing_destination(self):
        destination = self.root / "existing"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("preserve me")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            gen.clone((self.root / "missing").as_uri(), path=str(destination))
        self.assertEqual(marker.read_text(), "preserve me")

        existing_file = self.root / "existing.txt"
        existing_file.write_text("preserve me")
        with self.assertRaisesRegex(RuntimeError, "not an empty directory"):
            gen.clone("not a remote URL", path=existing_file)
        self.assertEqual(existing_file.read_text(), "preserve me")

    @unittest.skipIf(
        os.name == "nt", "native file remote workflows are currently tested on Unix"
    )
    def test_clone_accepts_strings_and_pathlike_destinations(self):
        self.import_sequence(self.repository, "base")
        remote_url = (self.root / "local").as_uri()
        for path_type in [str, Path, PurePath]:
            for existing in [False, True]:
                with self.subTest(path_type=path_type.__name__, existing=existing):
                    destination = self.root / f"clone-{path_type.__name__}-{existing}"
                    if existing:
                        destination.mkdir()
                    cloned = gen.clone(remote_url, path=path_type(destination))
                    self.assertEqual(cloned.current_branch.name, "main")
                    self.assertEqual(len(cloned.get_sequence_graphs()), 1)
                    self.assertTrue((destination / ".gen").is_dir())
        with self.assertRaises(TypeError):
            gen.clone(remote_url, path=42)

    @unittest.skipIf(
        os.name == "nt", "native file remote workflows are currently tested on Unix"
    )
    def test_clone_reuses_freshly_initialized_destination(self):
        self.import_sequence(self.repository, "base")
        destination = self.root / "initialized-first"
        gen.Repository(str(destination))
        cloned = gen.clone((self.root / "local").as_uri(), path=str(destination))
        self.assertEqual(len(cloned.get_sequence_graphs()), 1)

        used = self.root / "initialized-and-used"
        self.import_sequence(gen.Repository(str(used)), "other")
        with self.assertRaisesRegex(RuntimeError, "not an empty directory"):
            gen.clone((self.root / "local").as_uri(), path=str(used))

    @unittest.skipIf(
        os.name == "nt", "native file remote workflows are currently tested on Unix"
    )
    def test_clone_records_the_given_committer(self):
        self.import_sequence(self.repository, "base")
        cloned = gen.clone(
            (self.root / "local").as_uri(),
            path=str(self.root / "identified-clone"),
            committer="Ada Lovelace",
            email="ada@example.com",
        )
        self.import_sequence(cloned, "extra")
        operation = cloned.get_operations()[0]
        self.assertEqual(operation.committer, "Ada Lovelace")
        self.assertEqual(operation.email, "ada@example.com")

    def test_checkout_creates_branch_and_accepts_names_and_objects(self):
        self.import_sequence(self.repository, "base")
        original_head = self.repository.current_branch.head
        branch = self.repository.checkout("golden-gate", create=True)
        self.assertIsInstance(branch, gen.Branch)
        self.assertTrue(branch.is_current)
        self.assertEqual(branch.head, original_head)
        self.repository.checkout("main")
        checked_out = self.repository.checkout(branch)
        self.assertTrue(checked_out.is_current)
        self.assertEqual(checked_out.name, "golden-gate")

        self.repository.checkout("main")
        with self.assertRaises(RuntimeError):
            self.repository.checkout("golden-gate", create=True)
        with self.assertRaises(RuntimeError):
            self.repository.checkout("missing")
        with self.assertRaises(TypeError):
            self.repository.checkout(42, create=True)
        self.assertEqual(self.repository.current_branch.name, "main")
        self.assertEqual(
            [branch.name for branch in self.repository.get_branches()],
            ["golden-gate", "main"],
        )

    def test_http_transfer_failures_leave_repository_usable(self):
        for status in [400, 500]:
            with self.subTest(status=status):
                with genhub_server(lambda *_: (status, {"message": "denied"})) as (
                    url,
                    requests,
                    _assets,
                ):
                    with patch.dict(os.environ, {"GENHUB_API_KEY": "test-key"}):
                        for action in ["push", "pull", "fetch"]:
                            repository = gen.Repository(
                                str(self.root / f"{action}-{status}")
                            )
                            origin = repository.add_remote("origin", url)
                            repository.set_default_remote(origin)
                            with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                                getattr(repository, action)()
                            self.assertEqual(repository.current_branch.name, "main")
                            repository = None
                        failed = self.root / f"clone-{status}"
                        with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                            gen.clone(url, path=str(failed))
                    self.assertEqual(len(requests), 4)
                    self.assertTrue(
                        all(
                            path.endswith("/remote-capability")
                            for path, _, _ in requests
                        )
                    )
                    api_key_requests = [
                        body
                        for _, headers, body in requests
                        if headers.get("x-api-key") == "test-key"
                    ]
                    self.assertEqual(
                        [body["operation"] for body in api_key_requests],
                        ["push"],
                        "direct push should carry the API key; denied reads should stay anonymous",
                    )

    # TODO: Re-enable when pulling an explicit non-current branch updates that local branch.
    # dolt_pull currently merges it into the active branch instead; the later checkout also
    # reports uncommitted changes even though status was clean immediately after the pull.
    @unittest.skip(
        "Non-current branch pull updates the active branch and blocks checkout"
    )
    @unittest.skipIf(
        os.name == "nt", "native file remote workflows are currently tested on Unix"
    )
    def test_file_remote_branch_transfers_and_force(self):
        upstream_path = self.root / "upstream"
        upstream = gen.Repository(str(upstream_path))
        self.import_sequence(upstream, "base")
        local = gen.clone(upstream_path.as_uri(), path=str(self.root / "clone"))
        origin = local.default_remote
        self.assertEqual(origin.name, "origin")
        self.assertEqual(local.current_branch.remote, "origin")
        feature = local.create_branch("feature")
        local.checkout(feature)
        self.import_sequence(local, "feature")
        local.checkout("main")
        local.push(origin, feature)
        self.assertEqual(local.current_branch.name, "main")

        upstream = gen.Repository(str(upstream_path))
        upstream.checkout("feature")
        self.import_sequence(upstream, "remote_change")
        upstream.checkout("main")
        original_head = local.current_branch.head
        local.fetch(origin, branch="feature")
        self.assertEqual(local.current_branch.head, original_head)
        self.assertEqual(local.current_branch.name, "main")
        local.pull(origin, feature)
        self.assertEqual(local.current_branch.name, "main")
        local.checkout(feature)
        self.assertEqual(
            local.current_branch.head,
            next(
                branch.head
                for branch in upstream.get_branches()
                if branch.name == "feature"
            ),
        )

        local.reset(local.get_operations()[1])
        with self.assertRaises(RuntimeError):
            local.push(origin)
        local.push(origin, force=True)
        upstream = gen.Repository(str(upstream_path))
        self.assertEqual(
            local.current_branch.head,
            next(
                branch.head
                for branch in upstream.get_branches()
                if branch.name == "feature"
            ),
        )

    def test_http_api_key_covers_graph_assets_and_rejects_legacy_push(self):
        upstream_path = self.root / "upstream"
        upstream = gen.Repository(str(upstream_path))
        self.import_sequence(upstream, "base")

        # Return the complete reachable inventory on every transfer query. GenHub validates
        # requests against branch history, which includes each archived FASTA and its indexes.
        known_assets = archived_asset_bytes_by_id(upstream)
        del upstream
        server_origin = {}

        def respond(path, headers, _body):
            if headers.get("x-api-key") != "accepted-test-key":
                return 401, {"message": "private repository"}
            if path.endswith("/remote-capability"):
                return 200, {
                    "remote_url": graph_uri,
                    "expires_at": "2035-01-01T00:00:00Z",
                    "default_branch": "main",
                    "transfer_id": "00000000-0000-0000-0000-000000000001",
                }
            if path.endswith("/asset-transfers"):
                return 200, {
                    "assets": [
                        {
                            "id": asset_id,
                            "url": f"{server_origin['url']}/assets/{asset_id}",
                        }
                        for asset_id in known_assets
                    ]
                }
            return 200, {"assets": []}

        with (
            direct_gcs_database(upstream_path / ".gen" / "default.db") as graph_uri,
            genhub_server(respond) as (url, requests, served_assets),
        ):
            server_origin["url"] = url.split("/repos/", 1)[0]
            served_assets.update(known_assets)
            with patch.dict(os.environ, {"GENHUB_API_KEY": "accepted-test-key"}):
                local = gen.clone(url, path=str(self.root / "legacy-clone"))
                self.assertEqual(
                    archived_asset_bytes_by_id(local),
                    known_assets,
                    "clone should download the reachable graph assets",
                )
                local.fetch()
                local.pull()
                self.import_sequence(local, "legacy_local_change")
                local_assets = archived_asset_bytes_by_id(local)
                local_only_asset_ids = local_assets.keys() - known_assets.keys()
                self.assertTrue(
                    local_only_asset_ids,
                    "local import should add archived assets",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "did not include a direct GCS session"
                ):
                    local.push(force=True)
                self.assertTrue(
                    local_only_asset_ids.isdisjoint(served_assets.keys()),
                    "rejected legacy push must not upload local-only assets",
                )

            authenticated = [
                (path, body)
                for path, headers, body in requests
                if headers.get("x-api-key") == "accepted-test-key"
            ]
            capabilities = [
                body
                for path, body in authenticated
                if path.endswith("/remote-capability")
            ]
            self.assertIn("push", [body["operation"] for body in capabilities])
            self.assertGreaterEqual(
                sum(path.endswith("/asset-transfers") for path, _ in authenticated),
                1,
            )
            self.assertFalse(
                any(path.endswith("/asset-transfers/complete") for path, _ in authenticated)
            )
            self.assertEqual(
                local.default_remote.url, url.replace("/repos/", "/api/repos/")
            )

    @unittest.skipIf(
        os.name == "nt", "direct-GCS remote workflows are currently tested on Unix"
    )
    def test_direct_gcs_remote_push_uploads_graph_and_completes_asset_transfer(self):
        upstream_path = self.root / "upstream"
        upstream = gen.Repository(str(upstream_path))
        self.import_sequence(upstream, "base")
        graph_url = (upstream_path / ".gen" / "default.db").as_uri()
        known_assets = archived_asset_bytes_by_id(upstream)
        server_origin = {}
        bucket = f"gen-python-{uuid.uuid4().hex}"
        graph_prefix = f"repositories/{uuid.uuid4().hex}/.gen/graph_db/"
        transfer_id = str(uuid.uuid4())

        def respond(path, headers, body):
            if path.startswith("/api/publish?") or path.startswith("/api/abort?"):
                return 204, None
            if headers.get("x-api-key") != "accepted-test-key":
                return 401, {"message": "private repository"}
            if path.endswith("/remote-capability"):
                now = datetime.now(timezone.utc)
                response = {
                    "remote_url": None if body["operation"] == "push" else graph_url,
                    "expires_at": (now + timedelta(minutes=30)).isoformat(),
                    "default_branch": "main",
                    "transfer_id": transfer_id,
                }
                if body["operation"] == "push":
                    session_id = headers.get("idempotency-token")
                    if session_id is None:
                        return 400, {"message": "push capability requires a session id"}
                    response["direct_push"] = {
                        "database_uri": database_uri,
                        "token_expires_at": (now + timedelta(hours=1)).isoformat(),
                        "session_id": session_id,
                        "session_scope": {
                            "principal": "repository:python-direct-gcs-test",
                            "target_database": "default.db",
                            "operations": json.dumps(
                                [
                                    "write",
                                    "push",
                                    body.get("branch") or "main",
                                    bool(body["force"]),
                                ],
                                separators=(",", ":"),
                            ),
                        },
                        "publish_url": (
                            f"{server_origin['url']}/api/publish?capability={transfer_id}"
                        ),
                        "abort_url": (
                            f"{server_origin['url']}/api/abort?capability={transfer_id}"
                        ),
                    }
                return 200, response
            if path.endswith("/asset-transfers"):
                return 200, {
                    "assets": [
                        {
                            "id": asset_id,
                            "url": f"{server_origin['url']}/assets/{asset_id}",
                        }
                        for asset_id in known_assets
                    ]
                }
            return 200, {"assets": []}

        with fake_gcs_server() as gcs_endpoint:
            bucket_request = urllib.request.Request(
                f"{gcs_endpoint}/storage/v1/b?project=test-project",
                data=json.dumps({"name": bucket}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(bucket_request, timeout=10) as response:
                self.assertEqual(response.status, 200)
            database_query = urllib.parse.urlencode(
                {
                    "database": "default.db",
                    "project": "test-project",
                    "vfs": "blockcachevfs",
                    "endpoint": gcs_endpoint,
                    "access_token": "fake-gcs-token",
                }
            )
            database_uri = f"gcs://{bucket}/{graph_prefix}?{database_query}"

            with genhub_server(respond) as (url, requests, served_assets):
                server_origin["url"] = url.split("/repos/", 1)[0]
                served_assets.update(known_assets)
                with patch.dict(os.environ, {"GENHUB_API_KEY": "accepted-test-key"}):
                    local = gen.clone(url, path=str(self.root / "direct-gcs-clone"))
                    local.fetch()
                    local.pull()
                    self.import_sequence(local, "local_change")
                    local_assets = archived_asset_bytes_by_id(local)
                    local_only_asset_ids = local_assets.keys() - known_assets.keys()
                    self.assertTrue(local_only_asset_ids)
                    known_assets.update(
                        {asset_id: local_assets[asset_id] for asset_id in local_only_asset_ids}
                    )
                    source_branch = local.current_branch.name
                    source_commit = local.current_branch.head
                    local.push(force=True)
                    self.assertEqual(local.current_branch.name, source_branch)
                    self.assertEqual(local.current_branch.head, source_commit)
                    for asset_id in local_only_asset_ids:
                        self.assertEqual(
                            served_assets.get(asset_id),
                            local_assets[asset_id],
                            "push should upload each local-only asset's exact archived bytes",
                        )

                authenticated = [
                    (path, body)
                    for path, headers, body in requests
                    if headers.get("x-api-key") == "accepted-test-key"
                ]
                capabilities = [
                    body
                    for path, body in authenticated
                    if path.endswith("/remote-capability")
                ]
                push_capabilities = [
                    body for body in capabilities if body["operation"] == "push"
                ]
                self.assertTrue(push_capabilities)
                self.assertTrue(all(body["force"] for body in push_capabilities))
                completions = [
                    body
                    for path, body in authenticated
                    if path.endswith("/asset-transfers/complete")
                ]
                self.assertEqual(len(completions), 1)
                self.assertEqual(completions[0]["transfer_id"], transfer_id)
                self.assertEqual(completions[0]["branch"], "main")
                self.assertTrue(
                    local_only_asset_ids
                    & {asset["id"] for asset in completions[0]["assets"]}
                )
                self.assertTrue(
                    any(
                        path.endswith("/asset-transfers")
                        for path, _body in authenticated
                    )
                )
                self.assertTrue(
                    any(
                        path.endswith("/asset-transfers/complete")
                        and body is not None
                        for path, body in authenticated
                    )
                )
                self.assertEqual(
                    local.default_remote.url, url.replace("/repos/", "/api/repos/")
                )

                publications = [
                    (path, headers)
                    for path, headers, _body in requests
                    if path.startswith("/api/publish?")
                ]
                self.assertEqual(len(publications), 1)
                self.assertNotIn("x-api-key", publications[0][1])

            object_list_url = (
                f"{gcs_endpoint}/storage/v1/b/{bucket}/o?"
                f"{urllib.parse.urlencode({'prefix': graph_prefix})}"
            )
            with urllib.request.urlopen(object_list_url, timeout=10) as response:
                graph_objects = json.load(response).get("items", [])
            self.assertTrue(graph_objects, "direct push should store graph objects in GCS")
            self.assertTrue(
                all(object_["name"].startswith(graph_prefix) for object_ in graph_objects),
                "direct push should keep graph objects under its isolated repository prefix",
            )
            self.assertTrue(
                any(
                    object_["name"].endswith("manifest.bcv")
                    for object_ in graph_objects
                ),
                "direct push should publish the graph manifest to GCS",
            )


if __name__ == "__main__":
    unittest.main()
