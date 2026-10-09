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
        arm_ends = gen.SuperPosition(self.locus.end(), alternative.end())

        self.widget.show(arm_ends)
        self.widget.show(arm_ends, center=True)
