"""Widget navigation to alternatives created by sequence edits."""

import gen

from test_api import FIXTURES, RepositoryTestCase


class WidgetEditingTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        [self.locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.widget = self.graph.plot()

    def test_go_to_accepts_a_superposition(self):
        alternative = self.graph.replace(self.locus, "TTTT", stack=True)
        arm_ends = gen.SuperPosition(self.locus.end(), alternative.end())

        self.widget.go_to(arm_ends)
        self.widget.go_to(arm_ends, center=True)

    def test_show_accepts_a_superposition(self):
        alternative = self.graph.replace(self.locus, "TTTT", stack=True)
        positions = gen.SuperPosition(alternative.end(), self.locus.start())
        self.widget = self.graph.plot(rows=16, cols=90, detail="full").hide_cursor()

        self.widget.show(positions, color="cyan", center=True)
        highlighted = [
            cell
            for cell in self.widget.frame["cells"]
            if cell.get("bg") == "#00cccc" and cell["text"] in "ACGT"
        ]
        self.assertEqual(len(highlighted), 2)
        self.assertEqual({cell["text"] for cell in highlighted}, {"G", "T"})
        leftmost_column = min(cell["x"] for cell in highlighted)

        self.widget.clear_highlights()
        self.widget.show(self.locus.start(), color="cyan", center=True)
        [start_cell] = [
            cell
            for cell in self.widget.frame["cells"]
            if cell.get("bg") == "#00cccc" and cell["text"] == "G"
        ]
        self.assertEqual(leftmost_column, start_cell["x"])
