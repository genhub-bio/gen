"""Ensure Python file paths are resolved relative to the process working directory."""

import os
from pathlib import Path
import tempfile
import unittest

import gen


class RelativePathTests(unittest.TestCase):
    def test_import_fasta_uses_current_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_directory = directory_path / "inputs"
            input_directory.mkdir()
            (input_directory / "input.fa").write_text(">sequence\nACGT\n")
            repository = gen.Repository(str(directory_path / "repository"))
            original_directory = Path.cwd()
            try:
                os.chdir(input_directory)
                sample = repository.import_fasta("input.fa", sample="reference")
            finally:
                os.chdir(original_directory)

            self.assertEqual(len(sample), 1)
