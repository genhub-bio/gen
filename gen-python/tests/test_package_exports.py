"""Check browser-facing exports from the Python package initializer."""

import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import gen


class PackageExportTests(unittest.TestCase):
    def tearDown(self):
        importlib.reload(gen)

    def test_remote_and_clone_are_exported_on_emscripten(self):
        with patch.object(sys, "platform", "emscripten"):
            package = importlib.reload(gen)

        self.assertIs(package.Remote, package.gen.Remote)
        self.assertIs(package.clone, package.gen.clone)
        self.assertIn("Remote", package.__all__)
        self.assertIn("clone", package.__all__)

    def test_package_import_allows_missing_optional_clone_binding(self):
        engine_module = importlib.import_module("gen.gen")
        engine_without_clone = ModuleType("gen.gen")
        for name in dir(engine_module):
            if not name.startswith("__"):
                setattr(engine_without_clone, name, getattr(engine_module, name))
        del engine_without_clone.clone
        if hasattr(gen, "clone"):
            del gen.clone

        with patch.dict(sys.modules, {"gen.gen": engine_without_clone}):
            with patch.object(sys, "platform", "emscripten"):
                package = importlib.reload(gen)

            self.assertIs(package.Remote, engine_without_clone.Remote)
            self.assertFalse(hasattr(package, "clone"))
            self.assertIn("Remote", package.__all__)
            self.assertNotIn("clone", package.__all__)


if __name__ == "__main__":
    unittest.main()
