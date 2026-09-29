"""Locus addressing remains stable across sequence edits."""

import gen

from test_api import FIXTURES, RepositoryTestCase


class LocusEditingTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_fasta(str(FIXTURES / "simple.fa"))[0]
        [self.locus] = self.graph.search("GGAACACA", sequence_kind="exact")

    def crossing_locus(self):
        self.graph.replace("m123:20-22", "TT")
        [locus] = self.graph.search("CGTTAAC", sequence_kind="exact")
        self.assertEqual(len(locus.slices), 3)
        return locus

    def assert_same_graph_coordinate(self, left, right):
        self.assertEqual(left.node.id, right.node.id)
        self.assertEqual(
            left.node.sequence_start + left.offset,
            right.node.sequence_start + right.offset,
        )

    def assert_reverse_correspondence(self, forward):
        reverse = forward.reverse_complement()
        self.assertEqual(forward[0].node.id, reverse[-1].node.id)
        self.assertEqual(forward[0].offset, reverse[-1].offset)
        self.assertEqual(forward[0].strand, "+")
        self.assertEqual(reverse[-1].strand, "-")
        self.assertNotEqual(forward[0], reverse[-1])
        for index in range(len(forward)):
            self.assert_same_graph_coordinate(forward[index], reverse[-index - 1])
            self.assertNotEqual(forward[index], reverse[-index - 1])
        self.assertEqual(reverse[0], reverse.start())
        self.assertEqual(reverse[-1], reverse.end())

    def test_integer_indexing_across_slices(self):
        locus = self.crossing_locus()
        self.assertEqual([part.end - part.start for part in locus.slices], [2, 2, 3])
        for index, coordinate in enumerate([18, 19, 0, 1, 22, 23, 24]):
            position = locus[index]
            part = locus.slices[0 if index < 2 else 1 if index < 4 else 2]
            self.assertEqual(position.node.id, part.node.id)
            self.assertEqual(position.node.sequence_start + position.offset, coordinate)
            self.assertEqual(position, locus.slice(index, index + 1).start())
        self.assertGreater(locus[4].node.sequence_start, 0)

    def test_slice_across_nodes(self):
        locus = self.crossing_locus()
        result = locus[1:6]
        self.assertEqual(len(result.slices), 3)
        self.assertEqual([part.end - part.start for part in result.slices], [1, 2, 2])
        self.assertEqual(result, locus.slice(1, 6))
        for index in range(len(result)):
            self.assertEqual(result[index], locus[index + 1])

    def test_reverse_indexing_and_slicing_across_nodes(self):
        forward = self.crossing_locus()
        self.assert_reverse_correspondence(forward)
        reverse = forward.reverse_complement()
        result = reverse[1:6]
        self.assertEqual(len(result.slices), 3)
        self.assertEqual([part.end - part.start for part in result.slices], [2, 2, 1])
        self.assertEqual(result, reverse.slice(1, 6))
        self.assertEqual(result, forward[1:6].reverse_complement())
        for index in range(len(result)):
            self.assertEqual(result[index], reverse[index + 1])

    def test_locus_equality_and_positions_survive_node_splitting(self):
        original = self.locus[1:6]
        self.graph.replace("m123:3-5", "TT")
        [current] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.assertEqual(self.locus, current)
        self.assertEqual(hash(self.locus), hash(current))
        self.assertEqual({self.locus: "saved"}[current], "saved")
        self.assertEqual(len({self.locus, current}), 1)
        self.assertEqual(original, current[1:6])
        for index in range(len(self.locus)):
            self.assertEqual(self.locus[index], current[index])
            self.assert_same_graph_coordinate(self.locus[index], current[index])

    def test_annotation_roundtrip_uses_locus_identity(self):
        saved = gen.Annotation(self.locus, "saved")
        self.graph.replace("m123:3-5", "TT")
        [current] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.assertEqual(saved.locus, current)
        self.assertEqual(hash(saved.locus), hash(current))
        self.assertEqual(gen.Annotation(current, "current").segments, saved.segments)
        self.assertNotEqual(current, current.reverse_complement())
        self.graph.replace(saved.locus[1:3], "TT")
        self.assertTrue(self.contains(self.graph, "GTTACACA"))

    def test_insert_before_and_after_indexed_positions(self):
        self.graph.insert("TT", after=self.locus[3], before=self.locus[4])
        self.assertTrue(self.contains(self.graph, "GGAATTCACA"))
        self.graph.insert("CC", before=self.locus[0])
        self.assertTrue(self.contains(self.graph, "CCGGAATTCACA"))
        self.graph.insert("AA", after=self.locus[-1])
        self.assertTrue(self.contains(self.graph, "CACAAACAGAGA"))

    def test_indexed_position_orientation_controls_insert_direction(self):
        forward = self.locus
        reverse = forward.reverse_complement()
        self.graph.insert("TT", after=forward[0])
        self.assertTrue(self.contains(self.graph, "GTTGAACACA"))
        self.graph.insert("CC", after=reverse[-1])
        self.assertTrue(self.contains(self.graph, "GATCGGGGTTGAACACA"))

    def test_replace_and_delete_sliced_loci(self):
        locus = self.crossing_locus()
        inserted = self.graph.replace(locus[1:6], "GG")
        self.assertEqual(len(inserted), 2)
        self.assertTrue(self.contains(self.graph, "ATCGATCGGCACACAGAGA"))
        self.graph.delete(inserted[:])
        self.assertTrue(self.contains(self.graph, "ATCGATCCACACAGAGA"))

    def test_replace_and_delete_reverse_slices(self):
        reverse = self.crossing_locus().reverse_complement()
        inserted = self.graph.replace(reverse[1:6], "CC")
        self.assertEqual(inserted.strand, "-")
        self.assertTrue(self.contains(self.graph, "ATCGATCGGCACACAGAGA"))
        self.graph.delete(inserted[:])
        self.assertTrue(self.contains(self.graph, "ATCGATCCACACAGAGA"))
