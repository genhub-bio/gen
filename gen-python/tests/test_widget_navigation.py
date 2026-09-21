"""Exercise GraphWidget.go_to()/show() with Position and SuperPosition targets."""

import gen

from test_api import RepositoryTestCase, FIXTURES


class WidgetNavigationTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        [self.locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.widget = self.graph.plot()

    def test_go_to_accepts_a_position(self):
        self.widget.go_to(self.locus.start())
        self.widget.go_to(self.locus.start(), center=True)

    def test_show_accepts_a_position(self):
        self.widget.show(self.locus.start())
        self.widget.show(self.locus.start(), center=True)

    def test_go_to_accepts_a_superposition(self):
        arm_ends = gen.SuperPosition(self.locus.start(), self.locus.end())

        self.widget.go_to(arm_ends)
        self.widget.go_to(arm_ends, center=True)

    def test_show_accepts_a_superposition(self):
        arm_ends = gen.SuperPosition(self.locus.start(), self.locus.end())

        self.widget.show(arm_ends)
        self.widget.show(arm_ends, center=True)
