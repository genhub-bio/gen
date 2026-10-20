"""Exercise notebook downloads and explicit publication with a native Gen database."""

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlsplit
from unittest.mock import patch

import gen
from gen.browser import NotebookSessionError, _capture_active_notebook


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


def hash_text(value):
    return value.hex() if isinstance(value, bytes) else str(value)


class FakeResponse:
    def __init__(
        self, url, *, status=200, headers=None, data=b"", payload=None, text=""
    ):
        self.url = url
        self.status = status
        self.ok = status < 400
        self.headers = headers or {}
        self._data = data
        self._payload = payload
        self._text = text

    async def bytes(self):
        return self._data

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class NotebookServer:
    def __init__(self, snapshot_url, database, branch, base_commit, assets):
        self.snapshot_url = snapshot_url
        self.database = database
        self.branch = branch
        self.base_commit = base_commit
        self.assets = assets
        self.asset = assets[0]
        self.requests = []
        self.manifest_override = None
        self.snapshot_status = 200

    async def __call__(self, url, **kwargs):
        self.requests.append((url, kwargs))
        request = urlsplit(url)
        snapshot = urlsplit(self.snapshot_url)
        if kwargs.get("method") == "POST":
            return FakeResponse(
                url,
                payload={
                    "branch": kwargs["body"].fields["target_branch"],
                    "commit_hash": "d" * 40,
                },
            )
        if request.path == snapshot.path:
            return FakeResponse(
                url,
                status=self.snapshot_status,
                headers={
                    "X-Gen-Notebook-Branch": self.branch,
                    "X-Gen-Notebook-Base-Commit": self.base_commit,
                },
                data=self.database,
                text="snapshot denied",
            )
        if request.path.endswith("/assets"):
            return FakeResponse(
                url,
                payload=self.manifest_override
                or {
                    "assets": [
                        {
                            "id": asset["id"],
                            "filename": asset["filename"],
                            "checksum": asset["checksum"],
                            "url": (
                                "/api/repos/test/repository/notebook/assets/"
                                f"{asset['id']}?branch={self.branch}"
                                f"&commit={self.base_commit}"
                            ),
                        }
                        for asset in self.assets
                    ]
                },
            )
        asset = next(
            (
                asset
                for asset in self.assets
                if request.path.endswith(f"/assets/{asset['id']}")
            ),
            None,
        )
        if asset is not None:
            return FakeResponse(url, data=asset["contents"])
        raise AssertionError(f"Unexpected notebook request: {url}")


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
        self.repository = gen.Repository(str(self.root / "source"))
        self.addCleanup(self._release_repository)

        fasta = self.root / "source.fa"
        fasta.write_text(">chr1\nACGTACGTACGT\n")
        self.repository.import_fasta(str(fasta))
        self.base_commit = str(self.repository.current_branch.head)
        self.branch = self.repository.current_branch.name
        asset_rows = self.repository.query("SELECT id, checksum FROM gen_asset_refs")
        asset_files = list((self.repository.gen_dir / "assets").iterdir())
        files_by_checksum = {
            hashlib.sha256(path.read_bytes()).hexdigest(): path for path in asset_files
        }
        self.assets = []
        seen_asset_ids = set()
        for row in asset_rows:
            asset_id = hash_text(row[0])
            checksum = hash_text(row[1]).lower()
            asset_path = files_by_checksum.get(checksum)
            if asset_path is None or asset_id in seen_asset_ids:
                continue
            seen_asset_ids.add(asset_id)
            self.assets.append(
                {
                    "id": asset_id,
                    "filename": asset_path.name,
                    "checksum": checksum,
                    "contents": asset_path.read_bytes(),
                }
            )
        self.assertTrue(
            self.assets,
            "fixture asset should be content-addressed in the native database",
        )
        self.asset = self.assets[0]
        self.snapshot_url = (
            "https://genhub.example/api/repos/test/repository/notebook/snapshot"
            "?branch=main"
        )
        self.server = NotebookServer(
            self.snapshot_url,
            self.repository.snapshot_bytes(),
            self.branch,
            self.base_commit,
            self.assets,
        )

    def _release_repository(self):
        self.repository = None

    async def test_snapshot_loads_native_gen_and_publishes_only_on_commit(self):
        with patch("gen.browser._pyfetch", self.server):
            session = await gen.NotebookSession.open(self.snapshot_url)
        self.addCleanup(session.close)

        self.assertEqual(session.source_branch, self.branch)
        self.assertEqual(session.repository.current_branch.name, self.branch)
        self.assertEqual(len(session.repository.get_sequence_graphs()), 1)
        local_asset = (
            Path(session.repository.gen_dir) / "assets" / self.asset["filename"]
        )
        self.assertEqual(local_asset.read_bytes(), self.asset["contents"])
        self.assertFalse(
            any(kwargs.get("method") == "POST" for _, kwargs in self.server.requests)
        )

        graph = session.repository.get_sequence_graphs()[0]
        session.repository.import_sequence(
            "TTTTACGTACGT",
            name=graph.name,
            sample="notebook-edited",
            collection=graph.collection,
        )
        self.assertFalse(
            any(kwargs.get("method") == "POST" for _, kwargs in self.server.requests)
        )

        result = await session.commit("notebook-review", create_branch=True)
        self.assertEqual(result["branch"], "notebook-review")
        commit_request = next(
            kwargs
            for _, kwargs in self.server.requests
            if kwargs.get("method") == "POST"
        )
        form = commit_request["body"]
        self.assertEqual(
            form.fields,
            {
                "source_branch": self.branch,
                "base_commit": self.base_commit,
                "target_branch": "notebook-review",
                "create_branch": "true",
            },
        )
        self.assertTrue(form.database)
        self.assertEqual(
            urlsplit(
                next(
                    url
                    for url, kwargs in self.server.requests
                    if kwargs.get("method") == "POST"
                )
            ).path,
            "/api/repos/test/repository/notebook/commit",
        )

        await session.commit(self.branch)
        self.assertEqual(session.base_commit, "d" * 40)
        existing_target_form = next(
            kwargs["body"]
            for _, kwargs in self.server.requests
            if kwargs.get("method") == "POST"
            and kwargs["body"].fields["target_branch"] == self.branch
        )
        self.assertEqual(existing_target_form.fields["create_branch"], "false")

        uploaded = self.root / "uploaded"
        (uploaded / ".gen").mkdir(parents=True)
        (uploaded / ".gen" / "default.db").write_bytes(form.database)
        uploaded_repository = gen.Repository(str(uploaded))
        self.addCleanup(lambda: setattr(self, "uploaded_repository", None))
        self.uploaded_repository = uploaded_repository
        self.assertIn(
            "notebook-edited",
            {sample.name for sample in uploaded_repository.samples},
        )

    async def test_asset_checksum_failure_prevents_repository_open(self):
        self.server.asset["checksum"] = "0" * 64
        with patch("gen.browser._pyfetch", self.server):
            with self.assertRaisesRegex(
                gen.NotebookSessionError, "Checksum verification failed"
            ):
                await gen.NotebookSession.open(self.snapshot_url)
        self.assertFalse(
            any(kwargs.get("method") == "POST" for _, kwargs in self.server.requests)
        )

    async def test_snapshot_http_error_is_reported_before_assets_are_requested(self):
        self.server.snapshot_status = 403
        with patch("gen.browser._pyfetch", self.server):
            with self.assertRaisesRegex(
                gen.NotebookSessionError, "HTTP 403: snapshot denied"
            ):
                await gen.NotebookSession.open(self.snapshot_url)
        self.assertEqual(len(self.server.requests), 1)

    async def test_cross_origin_asset_url_is_rejected_before_fetch(self):
        self.server.manifest_override = {
            "assets": [
                {
                    "id": self.asset["id"],
                    "filename": self.asset["filename"],
                    "checksum": self.asset["checksum"],
                    "url": f"https://attacker.example/assets/{self.asset['id']}"
                    f"?branch={self.branch}&commit={self.base_commit}",
                }
            ]
        }
        with patch("gen.browser._pyfetch", self.server):
            with self.assertRaisesRegex(gen.NotebookSessionError, "invalid asset URL"):
                await gen.NotebookSession.open(self.snapshot_url)
        self.assertEqual(len(self.server.requests), 2)

    async def test_notebook_file_is_tracked_and_sent_with_the_graph_snapshot(self):
        relative_path = "notebooks/review.ipynb"
        initial_path = "notebooks/existing.ipynb"
        with patch("gen.browser._pyfetch", self.server):
            session = await gen.NotebookSession.open(
                self.snapshot_url,
                notebook_path=initial_path,
            )
        self.addCleanup(session.close)

        self.assertEqual(session.notebook_path, initial_path)
        result = await session.commit(
            self.branch,
            notebook=NOTEBOOK,
            notebook_path=relative_path,
        )
        commit_request = next(
            kwargs
            for _, kwargs in self.server.requests
            if kwargs.get("method") == "POST"
        )
        form = commit_request["body"]
        notebook_bytes = form.notebook
        self.assertEqual(form.notebook_filename, "review.ipynb")
        self.assertEqual(form.fields["notebook_path"], relative_path)
        self.assertEqual(json.loads(notebook_bytes), NOTEBOOK)
        self.assertEqual(result["branch"], self.branch)
        self.assertEqual(session.base_commit, "d" * 40)
        self.assertEqual(session.notebook_path, relative_path)

        uploaded = self.root / "notebook-upload"
        (uploaded / ".gen").mkdir(parents=True)
        (uploaded / ".gen" / "default.db").write_bytes(form.database)
        uploaded_repository = gen.Repository(str(uploaded))
        self.addCleanup(lambda: setattr(self, "uploaded_repository", None))
        self.uploaded_repository = uploaded_repository
        notebook_asset = next(
            row
            for row in uploaded_repository.query(
                "SELECT logical_path, name, checksum, materialized_checksum "
                "FROM gen_asset_refs"
            )
            if row[0] == relative_path
        )
        self.assertEqual(notebook_asset[1], "review.ipynb")
        self.assertEqual(
            notebook_asset[3].hex(), hashlib.sha256(notebook_bytes).hexdigest()
        )
        stored_notebook = next(
            path
            for path in (Path(session.repository.gen_dir) / "assets").iterdir()
            if hashlib.sha256(path.read_bytes()).hexdigest() == notebook_asset[2].hex()
        )
        self.assertEqual(gzip.decompress(stored_notebook.read_bytes()), notebook_bytes)

    async def test_invalid_notebook_inputs_do_not_change_the_local_repository(self):
        with patch("gen.browser._pyfetch", self.server):
            session = await gen.NotebookSession.open(self.snapshot_url)
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
            any(kwargs.get("method") == "POST" for _, kwargs in self.server.requests)
        )

    async def test_open_rejects_invalid_notebook_path_before_network_requests(self):
        with patch("gen.browser._pyfetch", self.server):
            with self.assertRaisesRegex(NotebookSessionError, "safe relative"):
                await gen.NotebookSession.open(
                    self.snapshot_url,
                    notebook_path="/absolute.ipynb",
                )
        self.assertEqual(self.server.requests, [])


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
        with patch("gen.browser._NOTEBOOK_CAPTURE_TIMEOUT", 0.001):
            with self.assertRaisesRegex(NotebookSessionError, "Timed out"):
                await _capture_active_notebook(widget)

        self.assertEqual(widget.capture_request_id, "")
        self.assertIsNone(widget._observer)


if __name__ == "__main__":
    unittest.main()
