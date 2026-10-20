"""Verify that arbitrary repository files use Gen's tracked asset operations."""

from hashlib import sha256
import gzip
from pathlib import Path
import tempfile
import unittest

import gen


class RepositoryFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gen-python-files-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "repository"
        self.repository = gen.Repository(str(self.root))
        self.addCleanup(self._release_repository)

    def _release_repository(self):
        self.repository = None

    def test_add_file_tracks_notebook_asset(self):
        relative_path = "notebooks/review.ipynb"
        notebook = self.root / relative_path
        notebook.parent.mkdir(parents=True)
        contents = b'{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}'
        notebook.write_bytes(contents)

        asset = self.repository.add_file(notebook)

        self.assertEqual(asset.name, "review.ipynb")
        rows = self.repository.query(
            "SELECT file_type, logical_path, name, checksum, materialized_checksum "
            "FROM gen_asset_refs"
        )
        self.assertEqual(len(rows), 1)
        file_type, logical_path, name, checksum, materialized_checksum = rows[0]
        self.assertEqual(file_type, "none")
        self.assertEqual(logical_path, relative_path)
        self.assertEqual(name, "review.ipynb")
        self.assertNotEqual(checksum.hex(), sha256(contents).hexdigest())
        self.assertEqual(materialized_checksum.hex(), sha256(contents).hexdigest())

        asset_files = list((Path(self.repository.gen_dir) / "assets").iterdir())
        self.assertEqual(len(asset_files), 1)
        archived_contents = asset_files[0].read_bytes()
        self.assertEqual(checksum.hex(), sha256(archived_contents).hexdigest())
        self.assertEqual(gzip.decompress(archived_contents), contents)
        operation_count = self.repository.query(
            "SELECT COUNT(*) FROM gen_operation_log"
        )[0][0]
        with self.assertRaisesRegex(RuntimeError, "already in the repository"):
            self.repository.add_file(notebook)
        repeated_asset = self.repository.add_file(notebook, allow_no_changes=True)
        self.assertEqual(repeated_asset.id, asset.id)
        self.assertEqual(
            self.repository.query("SELECT COUNT(*) FROM gen_operation_log")[0][0],
            operation_count,
        )

    def test_add_file_does_not_hide_file_read_errors(self):
        with self.assertRaises(FileNotFoundError):
            self.repository.add_file("notebooks/missing.ipynb")
        with self.assertRaises(FileNotFoundError):
            self.repository.add_file("notebooks/missing.ipynb", allow_no_changes=True)


if __name__ == "__main__":
    unittest.main()
