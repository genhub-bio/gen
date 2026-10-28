"""Exercise browser notebook sessions with a native Gen repository."""

import asyncio
import base64
import hashlib
import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest
import zlib
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gen
from gen.browser import NotebookSessionError, _capture_active_notebook
from gen.browser_files import BrowserFileError

REPOSITORY_URL = "https://genhub.example/api/repos/test/repository"
NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "print('saved notebook')",
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


class FakeBranch:
    def __init__(self, name, head, *, is_current=False, remote=None):
        self.name = name
        self.head = head
        self.is_current = is_current
        self.remote = remote


class FakeRemoteRepository:
    def __init__(self, branch="main"):
        self.calls = []
        self.branches = {branch: "a" * 40}
        self.branch = branch
        self.asset_views = {branch: []}
        self.remote_heads = {}

    @property
    def current_branch(self):
        return FakeBranch(self.branch, self.branches[self.branch], is_current=True)

    def get_branches(self):
        branches = [
            FakeBranch(
                name,
                head,
                is_current=name == self.branch,
                remote="origin" if name.startswith("origin/") else None,
            )
            for name, head in self.branches.items()
        ]
        return branches

    def get_assets(self, branch=None):
        return list(self.asset_views.get(branch or self.branch, []))

    def fetch(self, remote=None, branch=None, *, include_assets=True):
        self.calls.append(("fetch", remote, branch, include_assets))

    def create_branch(self, name, start):
        self.calls.append(("create_branch", name, start))
        self.branches[name] = "b" * 40

    def checkout(
        self, branch, *, create=False, exist_ok=False, materialize_assets=True
    ):
        self.calls.append(("checkout", branch, create, exist_ok, materialize_assets))
        self.branch = branch

    def set_branch_remote(self, remote):
        self.calls.append(("set_branch_remote", remote))

    def pull(self, remote=None, branch=None, *, include_assets=True):
        self.calls.append(("pull", remote, branch, include_assets))
        tracking = self.remote_heads.get(self.branch)
        if tracking is not None:
            head, assets = tracking
            self.branches[self.branch] = head
            self.branches[f"origin/{self.branch}"] = head
            self.asset_views[self.branch] = list(assets)
            self.asset_views[f"origin/{self.branch}"] = list(assets)


class RecordingRepository:
    def __init__(self, repository):
        self.repository = repository
        self.calls = []
        self.push_error = None

    def __getattr__(self, name):
        return getattr(self.repository, name)

    def fetch(self, remote=None, branch=None, *, include_assets=True):
        self.calls.append(("fetch", remote, branch, include_assets))

    def pull(self, remote=None, branch=None, *, include_assets=True):
        self.calls.append(("pull", remote, branch, include_assets))

    def push(self, remote=None, branch=None, force=False):
        self.calls.append(("push", remote, branch, force))
        if self.push_error is not None:
            raise self.push_error

    def checkout(
        self,
        branch,
        *,
        create=False,
        exist_ok=False,
        materialize_assets=True,
    ):
        self.calls.append(("checkout", branch, create, exist_ok, materialize_assets))
        return self.repository.checkout(
            branch,
            create=create,
            exist_ok=exist_ok,
            materialize_assets=materialize_assets,
        )

    def create_branch(self, name, start=None):
        self.calls.append(("create_branch", name, start))
        return self.repository.create_branch(name, start)

    def set_branch_remote(self, remote=None):
        self.calls.append(("set_branch_remote", remote))
        return self.repository.set_branch_remote(remote)

    def add_file(self, filename, message=None, *, allow_no_changes=False):
        self.calls.append(("add_file", str(filename), allow_no_changes))
        return self.repository.add_file(
            filename, message, allow_no_changes=allow_no_changes
        )


class FakeWidget:
    def __init__(self, *, options=None, value=None, **kwargs):
        self.options = options
        self._observers = []
        self._click_handler = None
        self.layout = SimpleNamespace(display="")
        self.disabled = kwargs.get("disabled", False)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.options is not None and value not in [
            option[1] for option in self.options
        ]:
            raise ValueError("value must be present in options")
        old_value = getattr(self, "_value", None)
        self._value = value
        if old_value != value:
            for callback in self._observers:
                callback({"new": value})

    def observe(self, callback, names):
        self._observers.append(callback)

    def on_click(self, callback):
        self._click_handler = callback

    def click(self):
        self._click_handler(self)


class FakeWidgetContainer:
    def __init__(self, children):
        self.children = children


class FakeComm:
    def __init__(self, target_name, data):
        self.target_name = target_name
        self.data = data
        self.sent = []
        self.closed = False
        self._message_callback = None
        self._close_callback = None

    def on_msg(self, callback):
        self._message_callback = callback

    def on_close(self, callback):
        self._close_callback = callback

    def send(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True
        if self._close_callback is not None:
            self._close_callback({})

    def receive(self, data):
        self._message_callback({"content": {"data": data}})


class FakeAsset:
    def __init__(
        self,
        path,
        *,
        name=None,
        asset_id=None,
        archive_needs_decode=False,
    ):
        self.logical_path = path
        self.name = name or path.rsplit("/", 1)[-1]
        self.id = (
            asset_id
            if asset_id is not None
            else hashlib.sha256(f"asset:{path}".encode()).hexdigest()
        )
        self.archive_needs_decode = archive_needs_decode
        self._stored_path = None

    @property
    def path(self):
        if self._stored_path is None:
            raise FileNotFoundError("remote-only fixture")
        return self._stored_path


class FakeReadableReader:
    def __init__(self, contents, *, chunk_size=64 * 1024):
        self._contents = contents
        self._chunk_size = chunk_size
        self._offset = 0
        self.bytes_returned = 0
        self.read_calls = 0
        self.cancelled = False
        self.released = False

    async def read(self):
        self.read_calls += 1
        if self._offset >= len(self._contents):
            return SimpleNamespace(done=True, value=None)
        end = min(self._offset + self._chunk_size, len(self._contents))
        value = self._contents[self._offset : end]
        self._offset = end
        self.bytes_returned += len(value)
        return SimpleNamespace(done=False, value=value)

    async def cancel(self):
        self.cancelled = True

    def releaseLock(self):
        self.released = True


class FakeReadableBody:
    def __init__(self, reader):
        self._reader = reader

    def getReader(self):
        return self._reader

    async def cancel(self):
        await self._reader.cancel()


class FakeAssetResponse:
    def __init__(self, reader, status=200):
        self.status = status
        self.ok = status == 200
        self.body = FakeReadableBody(reader)


class CaptureWidget:
    def __init__(self, document=None, error="", respond=True):
        self._capture_request_id = ""
        self.capture_response_id = ""
        self.capture_notebook_json = ""
        self.capture_error = ""
        self._document = document
        self._error = error
        self._respond = respond
        self._observer = None

    @property
    def capture_request_id(self):
        return self._capture_request_id

    @capture_request_id.setter
    def capture_request_id(self, request_id):
        self._capture_request_id = request_id
        if request_id:
            self.request(request_id)

    def observe(self, callback, names):
        self._observer = callback

    def unobserve(self, callback, names):
        if self._observer is callback:
            self._observer = None

    def request(self, request_id):
        if not request_id or not self._respond:
            return
        self.capture_notebook_json = self._document or ""
        self.capture_error = self._error
        self.capture_response_id = request_id
        self._observer({"new": request_id})


class NotebookSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gen-python-notebook-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source_repository = gen.Repository(str(self.root / "source"))
        self.repository_class = gen.Repository
        self.addCleanup(self._release_source_repository)

        fasta = self.root / "source.fa"
        fasta.write_text(">chr1\nACGTACGTACGT\n")
        self.source_repository.import_fasta(str(fasta))
        self.branch = self.source_repository.current_branch.name
        self.base_commit = str(self.source_repository.current_branch.head)
        self.remote_open_calls = []
        self.opened_repositories = []
        self.asset_metadata_calls = []
        self.asset_body_requests = []
        self.asset_sizes = {"source.fa": fasta.stat().st_size}
        self.remote_asset_contents = {}
        self.asset_status_overrides = {}
        self._asset_urls = {}
        self._asset_response_readers = []
        self._asset_url_counter = 0
        self._metadata_patcher = patch(
            "gen.browser.fetch_repository_files",
            side_effect=self._fetch_repository_files,
        )
        self._asset_patcher = patch(
            "gen.browser.fetch_asset_response",
            side_effect=self._fetch_asset_response,
        )
        self._metadata_patcher.start()
        self._asset_patcher.start()
        self.addCleanup(self._metadata_patcher.stop)
        self.addCleanup(self._asset_patcher.stop)

    def _release_source_repository(self):
        self.source_repository = None

    def _add_workspace_asset(self, session, relative_path, contents):
        path = Path(session._temporary_directory.name) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        session.repository.add_file(path)
        self.asset_sizes[relative_path] = len(contents)
        return path

    def _repository_for_snapshot(self, snapshot_hash):
        for repository in reversed(self.opened_repositories):
            try:
                if str(repository.current_branch.head) == snapshot_hash:
                    return repository
            except (AttributeError, RuntimeError):
                continue
        return None

    async def _fetch_repository_files(self, repository_url, branch, snapshot_hash):
        self.asset_metadata_calls.append((repository_url, branch, snapshot_hash))
        repository = self._repository_for_snapshot(snapshot_hash)
        if repository is None:
            return []

        files = []
        for asset in repository.get_assets():
            path = asset.logical_path or asset.name
            if not isinstance(path, str) or not path:
                continue
            self._asset_url_counter += 1
            signature = hashlib.sha256(
                f"{self._asset_url_counter}:{path}".encode()
            ).hexdigest()
            signed_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            url = (
                "https://storage.googleapis.com/repos/test/repository/.gen/assets/"
                f"{signature}.none.bgz?X-Goog-Date={signed_date}"
                f"&X-Goog-Expires=900&X-Goog-Signature={signature}"
            )
            self._asset_urls[url] = (repository, asset, path)
            files.append(
                {
                    "name": asset.name or path.rsplit("/", 1)[-1],
                    "path": path,
                    "size": self.asset_sizes.get(path),
                    "last_modified": "2026-10-28T00:00:00.000Z",
                    "file_type": "file",
                    "url": url,
                }
            )
        return files

    async def _fetch_asset_response(self, url):
        entry = self._asset_urls[url]
        _repository, asset, path = entry
        self.asset_body_requests.append(path)
        statuses = self.asset_status_overrides.get(path, [])
        status = statuses.pop(0) if statuses else 200
        if path in self.remote_asset_contents:
            contents = self.remote_asset_contents[path]
        else:
            try:
                contents = Path(asset.path).read_bytes()
            except (AttributeError, FileNotFoundError, OSError):
                contents = b""
        reader = FakeReadableReader(contents)
        self._asset_response_readers.append(reader)
        return FakeAssetResponse(reader, status)

    async def _send_comm(self, session, comm, request):
        comm.receive(request)
        tasks = tuple(session._contents_tasks)
        if tasks:
            await asyncio.gather(*tasks)

    def _open_local_repository(
        self, repository_url, *, path, branch=None, committer=None, email=None
    ):
        self.remote_open_calls.append((repository_url, path, branch, committer, email))
        destination = Path(path)
        shutil.copytree(Path(self.source_repository.gen_dir), destination / ".gen")
        repository = self.repository_class(
            str(destination), committer=committer, email=email
        )
        repository.add_remote("origin", repository_url)
        repository.set_default_remote("origin")
        if branch is not None and branch != repository.current_branch.name:
            repository.checkout(branch, materialize_assets=False)
        repository.set_branch_remote("origin")
        recording_repository = RecordingRepository(repository)
        self.opened_repositories.append(recording_repository)
        return recording_repository

    async def _open_local_graph(self, **kwargs):
        repository_api = SimpleNamespace(open_remote_graph=self._open_local_repository)
        with patch("gen.Repository", repository_api):
            return await gen.NotebookSession.open(REPOSITORY_URL, **kwargs)

    async def test_open_uses_direct_graph_for_same_origin_repository(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph(
                committer="Notebook User",
                email="notebook@example.test",
            )
        self.addCleanup(session.close)

        self.assertEqual(len(self.remote_open_calls), 1)
        repository_url, workspace, branch, committer, email = self.remote_open_calls[0]
        self.assertEqual(repository_url, REPOSITORY_URL)
        self.assertIsNone(branch)
        self.assertEqual(committer, "Notebook User")
        self.assertEqual(email, "notebook@example.test")
        self.assertEqual(Path(workspace), Path(session._temporary_directory.name))
        self.assertEqual(session.repository.current_branch.name, self.branch)
        self.assertEqual(session.base_commit, self.base_commit)
        self.assertEqual(session.repository_url, REPOSITORY_URL)
        self.assertEqual(session.repository.calls, [])

    async def test_open_selects_a_nondefault_branch_without_fetch_or_pull(self):
        self.source_repository.create_branch("feature", self.base_commit)
        self.source_repository.checkout(self.branch)
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph(branch="feature")
        self.addCleanup(session.close)

        self.assertEqual(session.repository.current_branch.name, "feature")
        self.assertEqual(session.repository.calls, [])
        self.assertEqual(self.remote_open_calls[-1][2], "feature")

    async def test_open_rejects_missing_selected_branch(self):
        repository = FakeRemoteRepository()
        repository_api = SimpleNamespace(
            open_remote_graph=lambda *_args, **_kwargs: repository
        )
        with (
            patch("gen.browser._browser_origin", return_value="https://genhub.example"),
            patch("gen.Repository", repository_api),
        ):
            with self.assertRaisesRegex(
                NotebookSessionError, "could not check out branch"
            ):
                await gen.NotebookSession.open(REPOSITORY_URL, branch="feature")
        self.assertEqual(repository.calls, [])

    async def test_refresh_pins_files_to_verified_origin_branch_head(self):
        repository = FakeRemoteRepository()
        path = "remote/new.txt"
        asset = FakeAsset(path)
        remote_head = "c" * 40
        repository.remote_heads["main"] = (remote_head, [asset])
        with (
            patch("gen.browser._browser_origin", return_value="https://genhub.example"),
            patch(
                "gen.Repository",
                SimpleNamespace(open_remote_graph=lambda *_args, **_kwargs: repository),
            ),
        ):
            session = await gen.NotebookSession.open(REPOSITORY_URL)
        self.addCleanup(session.close)
        self.opened_repositories.append(repository)
        old_file_map = list(session.files)
        self.assertEqual(old_file_map, [])

        self.assertEqual(session.refresh(), remote_head)
        self.assertIn(path, session.files)
        self.assertEqual(repository.calls, [("pull", None, None, False)])
        await session.files[path].url()
        self.assertEqual(
            self.asset_metadata_calls[-1],
            (REPOSITORY_URL, "main", remote_head),
        )

    async def test_refresh_uses_graph_pull_without_asset_hydration(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)

        self.assertEqual(session.refresh(), self.base_commit)
        self.assertEqual(session.repository.calls, [("pull", None, None, False)])

    async def test_contents_lists_and_reads_local_assets_on_demand(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)

        self._add_workspace_asset(session, "docs/README.txt", b"repository notes")
        self._add_workspace_asset(session, "docs/50%GC.fa", b">chr1\nACGT\n")
        self._add_workspace_asset(session, "data/opaque.bin", b"\x00\x01\xff")
        self._add_workspace_asset(
            session,
            "images/icon.svg",
            b'<svg xmlns="http://www.w3.org/2000/svg"></svg>',
        )
        self._add_workspace_asset(
            session,
            "notebooks/example.ipynb",
            json.dumps(NOTEBOOK).encode("utf-8"),
        )
        untracked = Path(session._temporary_directory.name) / "docs/untracked.txt"
        untracked.write_text("not a tracked asset")

        with (
            patch.object(
                Path,
                "open",
                side_effect=AssertionError("directory listing read file bytes"),
            ),
            patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("directory listing read file bytes"),
            ),
        ):
            root = await session._contents_model("", content=True)
        self.assertEqual(root["type"], "directory")
        self.assertFalse(root["writable"])
        self.assertIsInstance(root["created"], str)
        self.assertIsInstance(root["last_modified"], str)
        self.assertIn("docs", {entry["name"] for entry in root["content"]})
        self.assertNotIn("untracked.txt", {entry["name"] for entry in root["content"]})

        docs = await session._contents_model("docs", content=True)
        self.assertEqual(
            [entry["name"] for entry in docs["content"]],
            ["50%GC.fa", "README.txt"],
        )
        text_file = await session._contents_model("docs/README.txt", content=True)
        self.assertEqual(text_file["content"], "repository notes")
        self.assertEqual(text_file["format"], "text")
        self.assertFalse(text_file["writable"])
        self.assertIsInstance(text_file["created"], str)
        self.assertIsInstance(text_file["last_modified"], str)
        percent_name = await session._contents_model("docs/50%GC.fa", content=True)
        self.assertEqual(percent_name["content"], ">chr1\nACGT\n")

        binary = await session._contents_model("data/opaque.bin", content=True)
        self.assertEqual(binary["format"], "base64")
        self.assertEqual(base64.b64decode(binary["content"]), b"\x00\x01\xff")
        svg = await session._contents_model(
            "images/icon.svg", content=True, format="base64"
        )
        self.assertEqual(svg["format"], "base64")
        self.assertEqual(svg["mimetype"], "image/svg+xml")
        self.assertEqual(
            base64.b64decode(svg["content"]),
            b'<svg xmlns="http://www.w3.org/2000/svg"></svg>',
        )

        notebook_listing = await session._contents_model("notebooks", content=True)
        notebook_entry = next(
            entry
            for entry in notebook_listing["content"]
            if entry["name"] == "example.ipynb"
        )
        self.assertEqual(notebook_entry["type"], "notebook")
        self.assertIsNone(notebook_entry["content"])

        notebook = await session._contents_model(
            "notebooks/example.ipynb", content=True
        )
        self.assertEqual(notebook["type"], "notebook")
        self.assertEqual(notebook["format"], "json")
        self.assertEqual(notebook["content"], NOTEBOOK)
        self.assertEqual(self.asset_body_requests, [])

    async def test_remote_files_are_streamed_only_on_request_and_urls_refresh_in_scope(
        self,
    ):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)

        path = "reads/nested/remote.fa"
        asset = FakeAsset(path)
        payload = bytes(range(256)) * 1024
        session.repository.get_assets = lambda: [asset]
        self.remote_asset_contents[path] = payload
        self.asset_sizes[path] = len(payload)
        session._record_published_ref()

        self.assertEqual(list(session.files), [path])
        listing = await session._contents_model("reads", content=True)
        self.assertEqual(listing["content"][0]["type"], "directory")
        self.assertEqual(self.asset_metadata_calls, [])
        self.assertEqual(self.asset_body_requests, [])

        file = session.files[path]
        signed_url = await file.url()
        self.assertEqual(len(self.asset_metadata_calls), 1)
        self.assertEqual(
            self.asset_metadata_calls[0],
            (REPOSITORY_URL, self.branch, self.base_commit),
        )
        self.assertNotIn("X-Goog-Signature", repr(signed_url))
        self.assertIn("X-Goog-Signature", str(signed_url))
        files_repr = repr(session.files)
        self.assertIn(path, files_repr)
        self.assertNotIn("X-Goog-Signature", files_repr)
        self.assertNotIn("file://", files_repr)
        self.assertEqual(self.asset_body_requests, [])

        self.asset_status_overrides[path] = [403, 200]
        async with file.open() as stream:
            first = await stream.read(1024)
            second = await stream.read(1024)
        self.assertEqual(first + second, payload[:2048])
        self.assertEqual(self.asset_body_requests, [path, path])
        self.assertEqual(len(self.asset_metadata_calls), 2)
        self.assertEqual(
            {call[1:] for call in self.asset_metadata_calls},
            {(self.branch, self.base_commit)},
        )
        self.assertEqual(self._asset_response_readers[-1].bytes_returned, 64 * 1024)
        self.assertTrue(self._asset_response_readers[-1].cancelled)
        self.assertTrue(self._asset_response_readers[-1].released)
        self.assertTrue(self._asset_response_readers[-2].cancelled)

        open_stream = file.open()
        await open_stream.__aenter__()
        active_reader = self._asset_response_readers[-1]
        session.close()
        await asyncio.sleep(0)
        with self.assertRaisesRegex(BrowserFileError, "session is closed"):
            await open_stream.read(1)
        self.assertTrue(active_reader.cancelled)

    async def test_local_edits_keep_matching_published_assets_readable(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)

        path = "reads/remote.fa"
        asset = FakeAsset(path)
        payload = b">remote\nACGTACGT\n"
        self.remote_asset_contents[path] = payload
        repository_get_assets = session.repository.get_assets
        current_asset = asset

        def get_assets(branch=None):
            return [current_asset, *repository_get_assets(branch=branch)]

        session.repository.get_assets = get_assets
        session._record_published_ref()
        original_commit = session.base_commit
        old_file = session.files[path]
        await old_file.url()

        graph = session.repository.get_sequence_graphs()[0]
        session.repository.import_sequence(
            "TTTTACGTACGT",
            name=graph.name,
            sample="notebook-local-edit",
            collection=graph.collection,
        )
        current_file = session.files[path]
        self.assertNotEqual(current_file._snapshot_hash, original_commit)
        self.assertEqual(current_file._published_snapshot_hash, original_commit)
        self.assertEqual(session.base_commit, original_commit)
        with self.assertRaisesRegex(NotebookSessionError, "published remote head"):
            session.refresh()
        self.assertEqual(session.base_commit, original_commit)
        with self.assertRaisesRegex(NotebookSessionError, "older graph commit"):
            await old_file.read(16)
        self.assertEqual(await current_file.read(len(payload)), payload)
        self.assertEqual(self.asset_metadata_calls[-1][2], original_commit)

        local_path = "notes/local-only.txt"
        self._add_workspace_asset(session, local_path, b"local content")
        local_file = session.files[local_path]
        self.assertEqual(await local_file.read(64), b"local content")
        with self.assertRaisesRegex(NotebookSessionError, "not been published"):
            await local_file.url()
        self.assertEqual(session.base_commit, original_commit)
        self.assertNotEqual(
            str(session.repository.current_branch.head), original_commit
        )
        self.assertEqual(len(self.asset_metadata_calls), 1)

        current_asset = FakeAsset(path, asset_id="replacement-content-hash")
        graph = session.repository.get_sequence_graphs()[0]
        session.repository.import_sequence(
            "GGGGACGTACGT",
            name=graph.name,
            sample="notebook-replacement-edit",
            collection=graph.collection,
        )
        replacement_file = session.files[path]
        self.assertEqual(replacement_file._asset_id, "replacement-content-hash")
        with self.assertRaisesRegex(NotebookSessionError, "not been published"):
            await replacement_file.url()

    async def test_asset_mapping_requires_content_identity(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)

        session.repository.get_assets = lambda: [
            SimpleNamespace(
                logical_path="assets/no-identity.fa",
                name="no-identity.fa",
                archive_needs_decode=False,
            )
        ]
        session._file_refs_hash = None
        with self.assertRaisesRegex(
            NotebookSessionError, "missing its content identity"
        ):
            list(session.files)

    async def test_gzip_asset_is_not_decoded_a_second_time(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)

        path = "reads/input.fa.gz"
        stored_gzip = gzip.compress(b">chr1\nACGTACGT\n")
        asset = FakeAsset(path, archive_needs_decode=False)
        session.repository.get_assets = lambda: [asset]
        self.remote_asset_contents[path] = stored_gzip
        session._record_published_ref()

        contents = await session.files[path].read(len(stored_gzip) + 1)
        self.assertEqual(contents, stored_gzip)
        self.assertEqual(gzip.decompress(contents), b">chr1\nACGTACGT\n")
        self.assertEqual(self.asset_body_requests, [path])

    async def test_concatenated_bgzf_members_stream_as_one_logical_file(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)

        path = "reads/multi-member.txt"
        payload = (b"ACGT" * 100_000) + b"end"
        self._add_workspace_asset(session, path, payload)
        source_asset = next(
            item
            for item in session.repository.get_assets()
            if item.logical_path == path
        )
        self.assertTrue(source_asset.archive_needs_decode)
        stored_bgzf = Path(source_asset.path).read_bytes()
        first_member = zlib.decompressobj(31)
        first_member.decompress(stored_bgzf)
        self.assertTrue(first_member.unused_data)

        remote_asset = FakeAsset(path, archive_needs_decode=True)
        session.repository.get_assets = lambda: [remote_asset]
        self.remote_asset_contents[path] = stored_bgzf
        session._record_published_ref()

        contents = await session.files[path].read(len(payload))
        self.assertEqual(contents, payload)
        self.assertEqual(self.asset_body_requests, [path])

    async def test_contents_follow_the_checked_out_branch_asset_view(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        main_branch = session.repository.current_branch.name
        session.repository.create_branch("contents-feature")
        session.repository.checkout("contents-feature")
        self._add_workspace_asset(session, "feature-only.txt", b"feature")
        feature_file = session.files["feature-only.txt"]

        session.repository.checkout(main_branch)
        with self.assertRaises(NotebookSessionError):
            await session._contents_model("feature-only.txt", content=True)
        with self.assertRaisesRegex(NotebookSessionError, "older graph commit"):
            await feature_file.read(32)

        session.repository.checkout("contents-feature")
        model = await session._contents_model("feature-only.txt", content=True)
        self.assertEqual(model["content"], "feature")

    async def test_contents_hide_database_names_and_reject_aliased_paths(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        database_path = Path(session.repository.gen_dir) / "default.db"
        assets_path = Path(session.repository.gen_dir) / "assets"
        assets_path.mkdir(exist_ok=True)
        hardlink_to_database = assets_path / "database-alias.bgz"
        os.link(database_path, hardlink_to_database)
        workspace_database_path = Path(session._temporary_directory.name) / "default.db"
        if not workspace_database_path.exists():
            workspace_database_path.write_bytes(b"workspace database fixture")
        workspace_database_alias = assets_path / "workspace-database-alias.bgz"
        os.link(workspace_database_path, workspace_database_alias)

        database_alias = FakeAsset("records/database-alias.fa")
        database_alias._stored_path = hardlink_to_database
        workspace_database_alias_asset = FakeAsset(
            "records/workspace-database-alias.fa"
        )
        workspace_database_alias_asset._stored_path = workspace_database_alias
        outside_alias = FakeAsset("records/outside-alias.fa")
        outside_file = Path(self.directory.name) / "outside.txt"
        outside_file.write_text("outside the repository")
        symlink_outside = assets_path / "outside-alias.bgz"
        os.symlink(outside_file, symlink_outside)
        outside_alias._stored_path = symlink_outside
        imported_asset_path = ".gen/outside_root/simple.fa"
        assets = [
            FakeAsset("default.db"),
            FakeAsset("records/default.db-wal"),
            FakeAsset(".gen/default.db"),
            FakeAsset(".gen/assets/archive.bgz"),
            FakeAsset(".gen/other/hidden.fa"),
            FakeAsset(".gen/outside_root/default.db"),
            FakeAsset(".gen/outside_root/nested/.gen/hidden.fa"),
            FakeAsset(".gen/outside_root/nested/.git/config"),
            FakeAsset(".git/config"),
            database_alias,
            workspace_database_alias_asset,
            outside_alias,
            FakeAsset(imported_asset_path, name="simple.fa"),
            FakeAsset("large.fa"),
        ]
        session.repository.get_assets = lambda: assets
        self.asset_sizes["large.fa"] = 8 * 1024 * 1024 + 1
        self.remote_asset_contents["large.fa"] = b"x" * (8 * 1024 * 1024 + 1)
        self.remote_asset_contents[imported_asset_path] = b">simple\nACGT\n"
        session._record_published_ref()

        root = await session._contents_model("", content=True)
        self.assertEqual(
            [entry["name"] for entry in root["content"]],
            [".gen", "large.fa"],
        )
        gen_directory = await session._contents_model(".gen", content=True)
        self.assertEqual(
            [entry["name"] for entry in gen_directory["content"]], ["outside_root"]
        )
        imported_directory = await session._contents_model(
            ".gen/outside_root", content=True
        )
        self.assertEqual(
            [entry["name"] for entry in imported_directory["content"]], ["simple.fa"]
        )
        self.assertIsNone(imported_directory["content"][0]["size"])
        self.assertEqual(self.asset_body_requests, [])
        imported_file = await session._contents_model(imported_asset_path, content=True)
        self.assertEqual(imported_file["content"], ">simple\nACGT\n")
        self.assertEqual(self.asset_body_requests, [imported_asset_path])
        self.assertNotIn("records/database-alias.fa", session.files)
        self.assertNotIn("records/workspace-database-alias.fa", session.files)
        self.assertNotIn("records/outside-alias.fa", session.files)
        self.assertIn(imported_asset_path, session.files)
        self.assertNotIn(".gen/assets/archive.bgz", session.files)
        self.assertNotIn(".gen/other/hidden.fa", session.files)
        with self.assertRaisesRegex(NotebookSessionError, "8 MiB browser read limit"):
            await session._contents_model("large.fa", content=True)
        self.assertEqual(self.asset_body_requests, [imported_asset_path, "large.fa"])
        self.assertLessEqual(
            self._asset_response_readers[-1].bytes_returned,
            8 * 1024 * 1024 + 64 * 1024,
        )

        for path in (
            "records/database-alias.fa",
            "records/workspace-database-alias.fa",
            "records/outside-alias.fa",
        ):
            with self.subTest(path=path), self.assertRaises(NotebookSessionError):
                await session._contents_model(path, content=True)
        for path in (
            "default.db",
            "records/default.db-wal",
            ".gen/default.db",
            ".gen/assets/archive.bgz",
            ".gen/other/hidden.fa",
            ".gen/outside_root/default.db",
            ".gen/outside_root/nested/.gen/hidden.fa",
            ".gen/outside_root/nested/.git/config",
            ".git/config",
            "../outside.txt",
            "/etc/passwd",
            "aliases\\outside-link",
            "%2e%2e/outside.txt",
            "docs%2fREADME.txt",
            "%252e%252e/outside.txt",
        ):
            with self.subTest(path=path), self.assertRaises(NotebookSessionError):
                await session._contents_model(path, content=True)

    async def test_contents_comm_is_session_scoped_refreshes_and_detaches(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            first = await self._open_local_graph()
            second = await self._open_local_graph()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self._add_workspace_asset(first, "shared.txt", b"first repository")
        self._add_workspace_asset(second, "shared.txt", b"second repository")

        opened_comms = []

        def create_comm(*, target_name, data):
            comm = FakeComm(target_name, data)
            opened_comms.append(comm)
            return comm

        comm_module = type(sys)("comm")
        comm_module.create_comm = create_comm
        with patch.dict(sys.modules, {"comm": comm_module}):
            first_comm = first.expose_contents()
            second_comm = second.expose_contents()

        self.assertEqual(first_comm.data, {"type": "ready", "protocol": 1})
        self.assertEqual(first_comm.target_name, "genhub.repository.contents")
        self.assertIs(first.expose_contents(), first_comm)
        await self._send_comm(
            first,
            first_comm,
            {"id": "first", "method": "get", "path": "shared.txt", "content": True},
        )
        await self._send_comm(
            second,
            second_comm,
            {"id": "second", "method": "get", "path": "shared.txt", "content": True},
        )
        self.assertEqual(first_comm.sent[-1]["model"]["content"], "first repository")
        self.assertEqual(second_comm.sent[-1]["model"]["content"], "second repository")
        await self._send_comm(
            second,
            second_comm,
            {
                "id": "binary-format",
                "method": "get",
                "path": "shared.txt",
                "content": True,
                "format": "base64",
            },
        )
        self.assertEqual(second_comm.sent[-1]["model"]["format"], "base64")
        self.assertEqual(
            base64.b64decode(second_comm.sent[-1]["model"]["content"]),
            b"second repository",
        )
        await self._send_comm(
            second,
            second_comm,
            {"id": "write", "method": "put", "path": "shared.txt", "content": True},
        )
        self.assertIn("error", second_comm.sent[-1])

        self._add_workspace_asset(second, "after-refresh.txt", b"new ref")
        await second.commit(second.source_branch)
        self.assertEqual(second_comm.sent[-1], {"type": "changed", "path": ""})
        await self._send_comm(
            second,
            second_comm,
            {
                "id": "refresh",
                "method": "get",
                "path": "after-refresh.txt",
                "content": True,
            },
        )
        self.assertEqual(second_comm.sent[-1]["model"]["content"], "new ref")

        sent_before_close = len(first_comm.sent)
        first_file = first.files["shared.txt"]
        first.close()
        self.assertTrue(first_comm.closed)
        first_comm.receive(
            {"id": "closed", "method": "get", "path": "shared.txt", "content": True}
        )
        self.assertEqual(len(first_comm.sent), sent_before_close)
        self.assertEqual(len(opened_comms), 2)
        with self.assertRaisesRegex(BrowserFileError, "session is closed"):
            await first_file.read(16)

    async def test_close_cancels_pending_remote_contents_reads(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        asset = FakeAsset("remote/pending.txt")
        session.repository.get_assets = lambda: [asset]
        session._record_published_ref()

        opened = asyncio.Event()
        pending = asyncio.Event()

        async def blocked_manifest(_repository_url, _branch, _snapshot_hash):
            opened.set()
            await pending.wait()
            return []

        opened_comms = []

        def create_comm(*, target_name, data):
            comm = FakeComm(target_name, data)
            opened_comms.append(comm)
            return comm

        comm_module = type(sys)("comm")
        comm_module.create_comm = create_comm
        with (
            patch.dict(sys.modules, {"comm": comm_module}),
            patch("gen.browser.fetch_repository_files", side_effect=blocked_manifest),
        ):
            comm = session.expose_contents()
            comm.receive(
                {
                    "id": "pending",
                    "method": "get",
                    "path": "remote/pending.txt",
                    "content": True,
                }
            )
            await opened.wait()
            self.assertTrue(session._contents_tasks)
            session.close()
            await asyncio.sleep(0)

        self.assertTrue(comm.closed)
        self.assertFalse(session._contents_tasks)
        self.assertFalse(any(message.get("id") == "pending" for message in comm.sent))

    async def test_notebook_is_added_as_an_asset_and_pushed_only_on_commit(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        self.assertEqual(session.repository.calls, [])

        relative_path = "notebooks/review.ipynb"
        result = await session.commit(
            self.branch,
            notebook=NOTEBOOK,
            notebook_path=relative_path,
        )

        self.assertEqual(result["branch"], self.branch)
        self.assertEqual(
            result["commit_hash"], str(session.repository.current_branch.head)
        )
        self.assertEqual(session.repository.calls[0][0], "add_file")
        self.assertTrue(session.repository.calls[0][2])
        self.assertEqual(
            session.repository.calls[-1],
            ("push", None, self.branch, False),
        )
        self.assertEqual(session.notebook_path, relative_path)
        self.assertIn(relative_path, session.files)
        self.assertEqual(
            self.asset_metadata_calls[-1],
            (REPOSITORY_URL, self.branch, result["commit_hash"]),
        )
        self.assertIn("X-Goog-Signature", str(await session.files[relative_path].url()))

        notebook_asset = next(
            asset
            for asset in session.repository.get_assets()
            if asset.name == "review.ipynb"
        )
        self.assertEqual(
            gzip.decompress(notebook_asset.path.read_bytes()),
            json.dumps(
                NOTEBOOK,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )
        self.assertFalse(
            any(
                asset.name == "review.ipynb"
                for asset in self.source_repository.get_assets()
            )
        )

    async def test_new_branch_starts_at_edited_head_and_pushes_without_force(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        graph = session.repository.get_sequence_graphs()[0]
        session.repository.import_sequence(
            "TTTTACGTACGT",
            name=graph.name,
            sample="notebook-edited",
            collection=graph.collection,
        )
        edited_head = str(session.repository.current_branch.head)

        result = await session.commit("notebook-review", create_branch=True)

        self.assertEqual(result["branch"], "notebook-review")
        self.assertEqual(session.repository.current_branch.name, "notebook-review")
        self.assertEqual(str(session.repository.current_branch.head), edited_head)
        self.assertIn(
            ("checkout", "notebook-review", True, False, False),
            session.repository.calls,
        )
        self.assertIn(("set_branch_remote", "origin"), session.repository.calls)
        self.assertEqual(
            session.repository.calls[-1],
            ("push", None, "notebook-review", False),
        )

    async def test_push_failure_keeps_saved_notebook_on_local_branch(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        session.repository.push_error = RuntimeError("login required")

        with self.assertRaisesRegex(NotebookSessionError, "login required"):
            await session.commit(
                "notebook-draft",
                create_branch=True,
                notebook=NOTEBOOK,
                notebook_path="notebooks/draft.ipynb",
            )

        self.assertEqual(session.repository.current_branch.name, "notebook-draft")
        self.assertEqual(session.source_branch, "notebook-draft")
        self.assertEqual(session.notebook_path, "notebooks/draft.ipynb")
        self.assertEqual(
            (
                Path(session._temporary_directory.name) / "notebooks/draft.ipynb"
            ).read_bytes(),
            json.dumps(
                NOTEBOOK,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )
        self.assertTrue(
            any(
                asset.name == "draft.ipynb" for asset in session.repository.get_assets()
            )
        )

    async def test_different_existing_branch_fails_before_saving_or_switching(self):
        self.source_repository.create_branch("other", self.base_commit)
        self.source_repository.checkout(self.branch)
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        destination = (
            Path(session._temporary_directory.name) / "notebooks" / "review.ipynb"
        )

        with self.assertRaisesRegex(
            NotebookSessionError, "Open that branch before editing"
        ):
            await session.commit(
                "other",
                notebook=NOTEBOOK,
                notebook_path="notebooks/review.ipynb",
            )

        self.assertFalse(destination.exists())
        self.assertEqual(session.repository.current_branch.name, self.branch)
        self.assertFalse(
            any(
                call[0] in {"add_file", "push", "checkout"}
                for call in session.repository.calls
            )
        )

    async def test_invalid_repository_urls_are_rejected_before_graph_open(self):
        opener = Mock()
        repository_api = SimpleNamespace(open_remote_graph=opener)
        with (
            patch("gen.browser._browser_origin", return_value="https://genhub.example"),
            patch("gen.Repository", repository_api),
        ):
            for repository_url, error in [
                ("https://attacker.example/api/repos/test/repository", "page origin"),
                ("/api/repos/test/repository?branch=main", "query or fragment"),
                ("/api/repos/test", "one GenHub repository"),
                ("https://[invalid]/api/repos/test/repository", "URL is invalid"),
            ]:
                with (
                    self.subTest(repository_url=repository_url),
                    self.assertRaisesRegex(NotebookSessionError, error),
                ):
                    await gen.NotebookSession.open(repository_url)
        opener.assert_not_called()

    async def test_invalid_notebook_inputs_do_not_change_local_repository(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_graph()
        self.addCleanup(session.close)
        starting_head = session.repository.current_branch.head

        with self.assertRaises(ValueError):
            await session.commit(self.branch, notebook=NOTEBOOK)
        with self.assertRaisesRegex(NotebookSessionError, "reserved path component"):
            await session.commit(
                self.branch,
                notebook=NOTEBOOK,
                notebook_path="../escape.ipynb",
            )
        with self.assertRaisesRegex(NotebookSessionError, "valid nbformat 4"):
            await session.commit(
                self.branch,
                notebook={"cells": [], "metadata": {}, "nbformat": 3},
                notebook_path="notebooks/invalid.ipynb",
            )
        invalid_json = dict(NOTEBOOK, unsupported_number=float("nan"))
        with self.assertRaisesRegex(NotebookSessionError, "cannot be encoded as JSON"):
            await session.commit(
                self.branch,
                notebook=invalid_json,
                notebook_path="notebooks/invalid-json.ipynb",
            )

        self.assertEqual(session.repository.current_branch.head, starting_head)
        self.assertFalse(
            any(call[0] in {"add_file", "push"} for call in session.repository.calls)
        )


class NotebookCommitWidgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_branch_success_updates_dropdown_after_options(self):
        class WidgetRepository:
            current_branch = FakeBranch("main", "a" * 40, is_current=True)

        class WidgetSession:
            def __init__(self):
                self.repository = WidgetRepository()
                self.notebook_path = None

            async def commit(self, branch, *, create_branch=False, **kwargs):
                self.asserted_branch = (branch, create_branch)
                self.repository.current_branch = FakeBranch(
                    branch, "b" * 40, is_current=True
                )
                return {"branch": branch, "commit_hash": "b" * 40}

        widget_module = SimpleNamespace(
            Dropdown=FakeWidget,
            Checkbox=FakeWidget,
            Text=FakeWidget,
            HTML=FakeWidget,
            Button=FakeWidget,
            VBox=FakeWidgetContainer,
        )
        session = WidgetSession()
        loop = asyncio.get_running_loop()
        tasks = []
        schedule_task = loop.create_task

        capture = SimpleNamespace(
            branch="",
            commit_hash="",
            hold_sync=lambda: nullcontext(),
        )
        with (
            patch.dict(sys.modules, {"ipywidgets": widget_module}),
            patch("gen.browser._new_notebook_capture_widget", return_value=capture),
            patch(
                "gen.browser.asyncio.create_task",
                side_effect=lambda coroutine: tasks.append(schedule_task(coroutine)),
            ),
        ):
            controls = gen.NotebookSession.commit_widget(session)
            target, create, new_branch, _, _, publish, message, _ = controls.children
            create.value = True
            new_branch.value = "notebook-review"
            publish.click()
            await tasks[0]

        self.assertEqual(session.asserted_branch, ("notebook-review", True))
        self.assertEqual(target.options, [("notebook-review", "notebook-review")])
        self.assertEqual(target.value, "notebook-review")
        self.assertEqual(capture.branch, "notebook-review")
        self.assertEqual(capture.commit_hash, "b" * 40)
        self.assertFalse(create.value)
        self.assertFalse(publish.disabled)
        self.assertIn("Published graph edits", message.value)


class NotebookCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_clears_synced_payload_and_observer(self):
        widget = CaptureWidget(json.dumps(NOTEBOOK))

        document = await _capture_active_notebook(widget)

        self.assertEqual(document, NOTEBOOK)
        self.assertEqual(widget.capture_request_id, "")
        self.assertEqual(widget.capture_response_id, "")
        self.assertEqual(widget.capture_notebook_json, "")
        self.assertEqual(widget.capture_error, "")
        self.assertIsNone(widget._observer)

    async def test_capture_error_cleans_up_observer_and_traits(self):
        widget = CaptureWidget(error="No active notebook")

        with self.assertRaisesRegex(NotebookSessionError, "No active notebook"):
            await _capture_active_notebook(widget)

        self.assertEqual(widget.capture_request_id, "")
        self.assertEqual(widget.capture_response_id, "")
        self.assertEqual(widget.capture_notebook_json, "")
        self.assertEqual(widget.capture_error, "")
        self.assertIsNone(widget._observer)

    async def test_capture_timeout_cleans_up_observer(self):
        widget = CaptureWidget(respond=False)
        with (
            patch("gen.browser._NOTEBOOK_CAPTURE_TIMEOUT", 0.001),
            self.assertRaisesRegex(NotebookSessionError, "Timed out"),
        ):
            await _capture_active_notebook(widget)

        self.assertEqual(widget.capture_request_id, "")
        self.assertIsNone(widget._observer)


if __name__ == "__main__":
    unittest.main()
