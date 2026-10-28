"""Exercise browser notebook sessions with a native Gen repository."""

import asyncio
import base64
import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gen
from gen.browser import NotebookSessionError, _capture_active_notebook

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
    def __init__(self, name, head, *, is_current=False):
        self.name = name
        self.head = head
        self.is_current = is_current


class FakeRemoteRepository:
    def __init__(self):
        self.calls = []
        self.branches = {"main": "a" * 40}
        self.branch = "main"

    @property
    def current_branch(self):
        return FakeBranch(self.branch, self.branches[self.branch], is_current=True)

    def get_branches(self):
        return [FakeBranch(name, head) for name, head in self.branches.items()]

    def fetch(self, remote, branch):
        self.calls.append(("fetch", remote, branch))

    def create_branch(self, name, start):
        self.calls.append(("create_branch", name, start))
        self.branches[name] = "b" * 40

    def checkout(self, branch):
        self.calls.append(("checkout", branch))
        self.branch = branch

    def set_branch_remote(self, remote):
        self.calls.append(("set_branch_remote", remote))

    def pull(self):
        self.calls.append(("pull",))


class RecordingRepository:
    def __init__(self, repository):
        self.repository = repository
        self.calls = []
        self.push_error = None

    def __getattr__(self, name):
        return getattr(self.repository, name)

    def fetch(self, remote=None, branch=None):
        self.calls.append(("fetch", remote, branch))

    def pull(self, remote=None, branch=None):
        self.calls.append(("pull", remote, branch))

    def push(self, remote=None, branch=None, force=False):
        self.calls.append(("push", remote, branch, force))
        if self.push_error is not None:
            raise self.push_error

    def checkout(self, branch, *, create=False, exist_ok=False):
        self.calls.append(("checkout", branch, create, exist_ok))
        return self.repository.checkout(branch, create=create, exist_ok=exist_ok)

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
        self.addCleanup(self._release_source_repository)

        fasta = self.root / "source.fa"
        fasta.write_text(">chr1\nACGTACGTACGT\n")
        self.source_repository.import_fasta(str(fasta))
        self.branch = self.source_repository.current_branch.name
        self.base_commit = str(self.source_repository.current_branch.head)
        self.clone_calls = []
        self.cloned_repositories = []

    def _release_source_repository(self):
        self.source_repository = None

    def _add_workspace_asset(self, session, relative_path, contents):
        path = Path(session._temporary_directory.name) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        session.repository.add_file(path)
        return path

    def _clone_local_repository(
        self, repository_url, *, path, committer=None, email=None
    ):
        self.clone_calls.append((repository_url, path, committer, email))
        destination = Path(path)
        shutil.copytree(Path(self.source_repository.gen_dir), destination / ".gen")
        repository = gen.Repository(str(destination), committer=committer, email=email)
        repository.add_remote("origin", repository_url)
        repository.set_default_remote("origin")
        repository.set_branch_remote("origin")
        recording_repository = RecordingRepository(repository)
        self.cloned_repositories.append(recording_repository)
        return recording_repository

    async def _open_local_clone(self, **kwargs):
        with patch("gen.clone", side_effect=self._clone_local_repository):
            return await gen.NotebookSession.open(REPOSITORY_URL, **kwargs)

    async def test_open_uses_gen_clone_for_same_origin_repository(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone(
                committer="Notebook User",
                email="notebook@example.test",
            )
        self.addCleanup(session.close)

        self.assertEqual(len(self.clone_calls), 1)
        repository_url, workspace, committer, email = self.clone_calls[0]
        self.assertEqual(repository_url, REPOSITORY_URL)
        self.assertEqual(committer, "Notebook User")
        self.assertEqual(email, "notebook@example.test")
        self.assertEqual(Path(workspace), Path(session._temporary_directory.name))
        self.assertEqual(session.repository.current_branch.name, self.branch)
        self.assertEqual(session.base_commit, self.base_commit)
        self.assertEqual(session.repository_url, REPOSITORY_URL)
        self.assertEqual(session.repository.calls, [])

    async def test_open_fetches_and_pulls_a_selected_nondefault_branch(self):
        self.source_repository.create_branch("feature", self.base_commit)
        self.source_repository.checkout(self.branch)
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone(branch="feature")
        self.addCleanup(session.close)

        self.assertEqual(session.repository.current_branch.name, "feature")
        self.assertEqual(
            session.repository.calls,
            [
                ("fetch", "origin", "feature"),
                ("checkout", "feature", False, False),
                ("set_branch_remote", "origin"),
                ("pull", None, None),
            ],
        )

    async def test_open_creates_missing_local_branch_from_remote_tracking_ref(self):
        repository = FakeRemoteRepository()
        with (
            patch("gen.browser._browser_origin", return_value="https://genhub.example"),
            patch("gen.clone", return_value=repository) as clone,
        ):
            session = await gen.NotebookSession.open(
                REPOSITORY_URL,
                branch="feature",
            )
        self.addCleanup(session.close)

        self.assertEqual(
            clone.call_args.kwargs["path"], session._temporary_directory.name
        )
        self.assertEqual(
            repository.calls,
            [
                ("fetch", "origin", "feature"),
                ("create_branch", "feature", "origin/feature"),
                ("checkout", "feature"),
                ("set_branch_remote", "origin"),
                ("pull",),
            ],
        )
        self.assertEqual(session.source_branch, "feature")

    async def test_refresh_uses_normal_pull(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone()
        self.addCleanup(session.close)

        self.assertEqual(session.refresh(), self.base_commit)
        self.assertEqual(session.repository.calls, [("pull", None, None)])

    async def test_contents_lists_and_reads_materialized_assets_on_demand(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone()
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
            root = session._contents_model("", content=True)
        self.assertEqual(root["type"], "directory")
        self.assertFalse(root["writable"])
        self.assertIsInstance(root["created"], str)
        self.assertIsInstance(root["last_modified"], str)
        self.assertIn("docs", {entry["name"] for entry in root["content"]})
        self.assertNotIn("untracked.txt", {entry["name"] for entry in root["content"]})

        docs = session._contents_model("docs", content=True)
        self.assertEqual(
            [entry["name"] for entry in docs["content"]],
            ["50%GC.fa", "README.txt"],
        )
        text_file = session._contents_model("docs/README.txt", content=True)
        self.assertEqual(text_file["content"], "repository notes")
        self.assertEqual(text_file["format"], "text")
        self.assertFalse(text_file["writable"])
        self.assertIsInstance(text_file["created"], str)
        self.assertIsInstance(text_file["last_modified"], str)
        percent_name = session._contents_model("docs/50%GC.fa", content=True)
        self.assertEqual(percent_name["content"], ">chr1\nACGT\n")

        binary = session._contents_model("data/opaque.bin", content=True)
        self.assertEqual(binary["format"], "base64")
        self.assertEqual(base64.b64decode(binary["content"]), b"\x00\x01\xff")
        svg = session._contents_model("images/icon.svg", content=True, format="base64")
        self.assertEqual(svg["format"], "base64")
        self.assertEqual(svg["mimetype"], "image/svg+xml")
        self.assertEqual(
            base64.b64decode(svg["content"]),
            b'<svg xmlns="http://www.w3.org/2000/svg"></svg>',
        )

        notebook_listing = session._contents_model("notebooks", content=True)
        notebook_entry = next(
            entry
            for entry in notebook_listing["content"]
            if entry["name"] == "example.ipynb"
        )
        self.assertEqual(notebook_entry["type"], "notebook")
        self.assertIsNone(notebook_entry["content"])

        notebook = session._contents_model("notebooks/example.ipynb", content=True)
        self.assertEqual(notebook["type"], "notebook")
        self.assertEqual(notebook["format"], "json")
        self.assertEqual(notebook["content"], NOTEBOOK)

    async def test_contents_follow_the_checked_out_branch_asset_view(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone()
        self.addCleanup(session.close)
        main_branch = session.repository.current_branch.name
        session.repository.create_branch("contents-feature")
        session.repository.checkout("contents-feature")
        self._add_workspace_asset(session, "feature-only.txt", b"feature")

        session.repository.checkout(main_branch)
        with self.assertRaises(NotebookSessionError):
            session._contents_model("feature-only.txt", content=True)

        session.repository.checkout("contents-feature")
        model = session._contents_model("feature-only.txt", content=True)
        self.assertEqual(model["content"], "feature")

    async def test_contents_hide_database_names_and_reject_aliased_paths(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone()
        self.addCleanup(session.close)
        workspace = Path(session._temporary_directory.name)
        database_path = Path(session.repository.gen_dir) / "default.db"

        self._add_workspace_asset(session, "default.db", b"tracked database name")
        self._add_workspace_asset(session, "records/default.db-wal", b"sidecar")
        tracked_paths = {
            asset.logical_path for asset in session.repository.get_assets()
        }
        self.assertIn("default.db", tracked_paths)
        self.assertIn("records/default.db-wal", tracked_paths)

        alias_directory = workspace / "aliases"
        alias_directory.mkdir()
        symlink_to_database = alias_directory / "database-link"
        os.symlink(database_path, symlink_to_database)
        hardlink_to_database = alias_directory / "database-copy"
        os.link(database_path, hardlink_to_database)
        outside_file = Path(self.directory.name) / "outside.txt"
        outside_file.write_text("outside the repository")
        symlink_outside = alias_directory / "outside-link"
        os.symlink(outside_file, symlink_outside)

        assets = [
            type("Asset", (), {"logical_path": path})()
            for path in (
                "default.db",
                "records/default.db-wal",
                "aliases/database-link",
                "aliases/database-copy",
                "aliases/outside-link",
                ".gen/default.db",
                "large.fa",
            )
        ]
        large_file = workspace / "large.fa"
        with large_file.open("wb") as file:
            file.truncate(8 * 1024 * 1024 + 1)
        session.repository.get_assets = lambda: assets

        root = session._contents_model("", content=True)
        self.assertEqual([entry["name"] for entry in root["content"]], ["large.fa"])
        self.assertEqual(root["content"][0]["size"], 8 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(NotebookSessionError, "8 MiB browser read limit"):
            session._contents_model("large.fa", content=True)
        for path in (
            "default.db",
            "records/default.db-wal",
            ".gen/default.db",
            ".git/config",
            "../outside.txt",
            "/etc/passwd",
            "aliases\\outside-link",
            "%2e%2e/outside.txt",
            "docs%2fREADME.txt",
            "%252e%252e/outside.txt",
        ):
            with self.subTest(path=path), self.assertRaises(NotebookSessionError):
                session._contents_model(path, content=True)
        for path in (
            "aliases/database-link",
            "aliases/database-copy",
            "aliases/outside-link",
        ):
            with self.subTest(path=path), self.assertRaises(NotebookSessionError):
                session._contents_model(path, content=True)

    async def test_contents_comm_is_session_scoped_refreshes_and_detaches(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            first = await self._open_local_clone()
            second = await self._open_local_clone()
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
        first_comm.receive(
            {"id": "first", "method": "get", "path": "shared.txt", "content": True}
        )
        second_comm.receive(
            {"id": "second", "method": "get", "path": "shared.txt", "content": True}
        )
        self.assertEqual(first_comm.sent[-1]["model"]["content"], "first repository")
        self.assertEqual(second_comm.sent[-1]["model"]["content"], "second repository")
        second_comm.receive(
            {
                "id": "binary-format",
                "method": "get",
                "path": "shared.txt",
                "content": True,
                "format": "base64",
            }
        )
        self.assertEqual(second_comm.sent[-1]["model"]["format"], "base64")
        self.assertEqual(
            base64.b64decode(second_comm.sent[-1]["model"]["content"]),
            b"second repository",
        )
        second_comm.receive(
            {"id": "write", "method": "put", "path": "shared.txt", "content": True}
        )
        self.assertIn("error", second_comm.sent[-1])

        self._add_workspace_asset(second, "after-refresh.txt", b"new ref")
        second.refresh()
        self.assertEqual(second_comm.sent[-1], {"type": "changed", "path": ""})
        second_comm.receive(
            {
                "id": "refresh",
                "method": "get",
                "path": "after-refresh.txt",
                "content": True,
            }
        )
        self.assertEqual(second_comm.sent[-1]["model"]["content"], "new ref")

        sent_before_close = len(first_comm.sent)
        first.close()
        self.assertTrue(first_comm.closed)
        first_comm.receive(
            {"id": "closed", "method": "get", "path": "shared.txt", "content": True}
        )
        self.assertEqual(len(first_comm.sent), sent_before_close)
        self.assertEqual(len(opened_comms), 2)

    async def test_notebook_is_added_as_an_asset_and_pushed_only_on_commit(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone()
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
            session = await self._open_local_clone()
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
            ("checkout", "notebook-review", True, False),
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
            session = await self._open_local_clone()
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
            session = await self._open_local_clone()
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

    async def test_invalid_repository_urls_are_rejected_before_clone(self):
        with (
            patch("gen.browser._browser_origin", return_value="https://genhub.example"),
            patch("gen.clone") as clone,
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
        clone.assert_not_called()

    async def test_invalid_notebook_inputs_do_not_change_local_repository(self):
        with patch(
            "gen.browser._browser_origin", return_value="https://genhub.example"
        ):
            session = await self._open_local_clone()
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
