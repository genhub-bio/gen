"""Loading minigraph GAF reads as one Locus per read."""

import re

import gen

from test_api import RepositoryTestCase

LEFT = "ACGTTGCAAC"
RIGHT = "GGATCCTTAG"
OTHER = "TTTTCCCCAAAAGGGG"
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


class GafReadsTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        # LEFT then RIGHT, plus an edge from the end of RIGHT back into the reverse strand of
        # LEFT, so a read can cross an inversion hop.
        self.graph = self.repository.import_library(
            "reads",
            [[gen.Sequence("left", LEFT)], [gen.Sequence("right", RIGHT)]],
            sample="sample",
        )
        left = self.graph.locus(0, len(LEFT))
        right = self.graph.locus(len(LEFT), len(LEFT) + len(RIGHT))
        self.graph.connect(
            after=right.end(), before=left.start().reverse_complement(), stack=True
        )
        self.segments = self.segment_ids(self.graph)
        self.other_graph = self.repository.import_library(
            "elsewhere", [[gen.Sequence("other", OTHER)]], sample="other_sample"
        )
        [self.other_segment] = self.segment_ids(self.other_graph)

    def segment_ids(self, graph):
        """Segment ids along the graph's path, in the order a forward read visits them."""
        gfa = self.root / f"{graph.name}.gfa"
        graph.export_gfa(str(gfa))
        [path_line] = [line for line in gfa.read_text().splitlines() if line[0] == "P"]
        return re.findall(r"([0-9a-f]+\.\d+\.\d+)\+", path_line.split("\t")[2])

    def gaf_row(self, name, path, start, end, tags="tp:A:P\tcg:Z:*"):
        return "\t".join(
            [name, "30", "0", "30", "+", path, "30", str(start), str(end), "30", "30", "60", tags]
        )

    def load(self, *rows, **kwargs):
        gaf = self.root / "reads.gaf"
        gaf.write_text("\n".join(rows) + "\n")
        return self.graph.load_gaf(str(gaf), **kwargs)

    def forward_path(self):
        return "".join(f">{segment}" for segment in self.segments)

    def test_forward_read_is_one_locus_equal_to_the_search_hit(self):
        reads = self.load(self.gaf_row("forward", self.forward_path(), 0, 20))
        self.assertEqual(list(reads), ["forward"])
        locus = reads["forward"]
        self.assertEqual(locus.sequence, LEFT + RIGHT)
        self.assertEqual(len(locus), 20)
        self.assertEqual(locus.strand, "+")
        hits = self.graph.search(LEFT + RIGHT)
        self.assertIn(locus, hits)

    def test_read_crossing_the_inversion_hop_is_mixed_strand(self):
        path = self.forward_path() + f"<{self.segments[0]}"
        reads = self.load(self.gaf_row("inverted", path, 0, 21))
        locus = reads["inverted"]
        self.assertEqual(locus.sequence, LEFT + RIGHT + "T")
        self.assertEqual(locus.strand, "mixed")
        self.assertEqual(locus.end().strand, "-")

    def test_reverse_read_uses_the_reverse_complement_twin_edges(self):
        path = "".join(f"<{segment}" for segment in reversed(self.segments))
        reads = self.load(self.gaf_row("backward", path, 0, 20))
        self.assertEqual(reads["backward"].sequence, reverse_complement(LEFT + RIGHT))
        self.assertEqual(reads["backward"].strand, "-")

    def test_clipping_trims_reverse_first_and_last_visits(self):
        path = "".join(f"<{segment}" for segment in reversed(self.segments))
        reads = self.load(self.gaf_row("clipped", path, 2, 17))
        expected = reverse_complement(LEFT + RIGHT)[2:17]
        self.assertEqual(reads["clipped"].sequence, expected)
        self.assertEqual(len(reads["clipped"]), 15)

    def test_unjoined_steps_are_untraceable_and_not_loaded(self):
        path = f">{self.segments[-1]}>{self.segments[0]}"
        reads = self.load(
            self.gaf_row("good", self.forward_path(), 0, 20),
            self.gaf_row("bad", path, 0, 11),
        )
        self.assertEqual(list(reads), ["good"])
        self.assertEqual(list(reads.untraceable), ["bad"])
        self.assertIn(self.segments[0], reads.untraceable["bad"])
        self.assertEqual(reads.skipped, 0)

    def test_reads_outside_the_graph_are_skipped_and_counted(self):
        reads = self.load(
            self.gaf_row("here", self.forward_path(), 0, 20),
            self.gaf_row("elsewhere", f">{self.other_segment}", 0, 16),
        )
        self.assertEqual(list(reads), ["here"])
        self.assertEqual(reads.skipped, 1)
        self.assertEqual(reads.untraceable, {})

    def test_primary_row_wins_unless_all_rows_are_allowed(self):
        short = self.gaf_row("read", f">{self.segments[1]}", 0, 9)
        # The secondary row has the longer aligned block, so it wins only when allowed.
        long = self.gaf_row("read", self.forward_path(), 0, 20, tags="tp:A:S").split("\t")
        long[10] = "99"
        long = "\t".join(long)
        primary_only = self.load(short, long)
        self.assertEqual(primary_only["read"].sequence, LEFT[1:])
        everything = self.load(short, long, primary_only=False)
        self.assertEqual(everything["read"].sequence, LEFT + RIGHT)

    def test_unmapped_rows_are_counted_and_cigarless_rows_load(self):
        unmapped = "\t".join(["gone", "30", "0", "30", "*", "*", "0", "0", "0", "0", "0", "0"])
        reads = self.load(
            unmapped,
            self.gaf_row("plain", self.forward_path(), 0, 20, tags="tp:A:P"),
        )
        self.assertEqual(reads.unmapped, 1)
        self.assertEqual(list(reads), ["plain"])
        self.assertIn("plain", reads)
        self.assertEqual(len(reads), 1)

    def test_malformed_segment_names_raise(self):
        with self.assertRaises(ValueError):
            self.load(self.gaf_row("bad", ">notasegment", 0, 5))
        with self.assertRaises(ValueError):
            self.load("too\tshort")
