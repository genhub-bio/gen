"""Exercise the installed extension's remote API without a live GenHub account."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePath
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

import gen


def asset_ids_by_name(repository):
    """Maps each asset's tracked file name to its content-addressed id.

    Every file import (even non-shallow ones) is tracked as a provenance asset, so any test that
    imports a fixture over an HTTP remote must be prepared to serve that asset's bytes back to the
    client during clone or pull, and to accept them during push.
    """
    return {asset.name: asset.id for asset in repository.get_assets()}


@contextmanager
def genhub_server(respond):
    requests = []
    assets = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            status, response = respond(self.path, self.headers, body)
            payload = json.dumps(response).encode()
            self.send_response(status)
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


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gen-python-remotes-")
        self.addCleanup(self.directory.cleanup)
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
            [remote.name for remote in self.repository.get_remotes()],
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

    def test_clone_preserves_existing_destination_and_cleans_failure(self):
        destination = self.root / "existing"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("preserve me")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            gen.clone((self.root / "missing").as_uri(), path=str(destination))
        self.assertEqual(marker.read_text(), "preserve me")
        failed = self.root / "failed"
        with self.assertRaises(RuntimeError):
            gen.clone("not a remote URL", path=str(failed))
        self.assertFalse(failed.exists())

        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(RuntimeError):
            gen.clone("not a remote URL", path=empty)
        self.assertTrue(empty.is_dir())
        self.assertEqual(list(empty.iterdir()), [])

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
                        origin = self.repository.add_remote("origin", url)
                        self.repository.set_default_remote(origin)
                        for action in ["push", "pull", "fetch"]:
                            with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                                getattr(self.repository, action)()
                            self.assertEqual(
                                self.repository.current_branch.name, "main"
                            )
                        failed = self.root / f"clone-{status}"
                        with self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                            gen.clone(url, path=str(failed))
                        self.assertFalse(failed.exists())
                        self.repository.remove_remote(origin)
                    self.assertEqual(len(requests), 4)
                    self.assertTrue(
                        all(
                            path.endswith("/remote-capability")
                            for path, _, _ in requests
                        )
                    )
                    self.assertEqual(
                        sum(
                            headers.get("x-api-key") == "test-key"
                            for _, headers, _ in requests
                        ),
                        1,
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

    @unittest.skipIf(
        os.name == "nt", "mock capabilities use native file remote transport"
    )
    def test_http_api_key_covers_graph_assets_and_push_completion(self):
        upstream_path = self.root / "upstream"
        upstream = gen.Repository(str(upstream_path))
        self.import_sequence(upstream, "base")
        graph_url = (upstream_path / ".gen" / "default.db").as_uri()

        # Every fasta import is tracked as a provenance asset (see `asset_ids_by_name`), so a
        # faithful mock must hand back real asset ids and serve or accept their bytes, not an empty
        # list. `known_asset_ids` grows as new assets are created locally; returning every known id
        # on each `/asset-transfers` call is always a safe superset (GenHub validates requested
        # transfers against the branch's full asset history, not just the ones actually needed).
        known_asset_ids = dict(asset_ids_by_name(upstream))
        server_origin = {}

        def respond(path, headers, _body):
            if headers.get("x-api-key") != "accepted-test-key":
                return 401, {"message": "private repository"}
            if path.endswith("/remote-capability"):
                return 200, {
                    "remote_url": graph_url,
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
                        for asset_id in known_asset_ids.values()
                    ]
                }
            return 200, {"assets": []}

        with genhub_server(respond) as (url, requests, served_assets):
            server_origin["url"] = url.split("/repos/", 1)[0]
            for name, asset_id in known_asset_ids.items():
                served_assets[asset_id] = (self.root / name).read_bytes()
            with patch.dict(os.environ, {"GENHUB_API_KEY": "accepted-test-key"}):
                local = gen.clone(url, path=str(self.root / "clone"))
                self.import_sequence(local, "local_change")
                local_asset_ids = asset_ids_by_name(local)
                known_asset_ids["local_change.fa"] = local_asset_ids["local_change.fa"]
                local.push(force=True)
                self.assertEqual(
                    served_assets.get(known_asset_ids["local_change.fa"]),
                    (self.root / "local_change.fa").read_bytes(),
                    "push should upload the local-only asset's exact bytes",
                )
                local.fetch()
                local.pull()
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
            self.assertEqual(
                [body["operation"] for body in capabilities],
                ["clone", "push", "pull", "pull", "pull"],
            )
            self.assertTrue(capabilities[1]["force"])
            self.assertEqual(
                sum(path.endswith("/asset-transfers") for path, _ in authenticated), 4
            )
            self.assertEqual(
                sum(
                    path.endswith("/asset-transfers/complete")
                    for path, _ in authenticated
                ),
                1,
            )
            self.assertEqual(
                local.default_remote.url, url.replace("/repos/", "/api/repos/")
            )


if __name__ == "__main__":
    unittest.main()
