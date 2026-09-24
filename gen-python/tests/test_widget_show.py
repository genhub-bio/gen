"""Check which annotation a plotted graph connects across sequence blocks after show()."""

from pathlib import Path
import tempfile
import unittest

import gen

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "puc19.gb"


class ShowFocusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gen-python-show-")
        self.addCleanup(self.directory.cleanup)
        self.repository = gen.Repository(str(Path(self.directory.name) / ".gen"))
        # Release the SQLite handle before the temp directory is removed: Windows
        # refuses to delete a file that a live handle still has open.
        self.addCleanup(self._release_repository)
        self.graph = self.repository.import_genbank(str(FIXTURE))[0]

    def _release_repository(self):
        self.graph = None
        self.repository = None

    def test_show_focuses_the_shown_annotation(self):
        widget = self.graph.plot(rows=24)
        controller = widget._controller
        self.assertIsNone(controller.focused_annotation)

        first, second = self.graph.list_annotations()[:2]
        widget.show(first)
        self.assertEqual(controller.focused_annotation, first.id)

        widget.show(second)
        self.assertEqual(controller.focused_annotation, second.id)

        widget.show(self.graph.search("CCWGG", "dna")[0])
        self.assertIsNone(controller.focused_annotation)

        widget.show(first)
        controller.clear_highlights()
        self.assertIsNone(controller.focused_annotation)


if __name__ == "__main__":
    unittest.main()
