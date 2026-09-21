"""Reading-order indexing through the installed graph-editing API."""

import gen

from test_api import RepositoryTestCase, FIXTURES


class IndexKey:
    def __index__(self):
        return 2


class LocusIndexingTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_fasta(str(FIXTURES / "simple.fa"))[0]
        [self.locus] = self.graph.search("GGAACACA", sequence_kind="exact")

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

    def test_integer_indexing_single_forward_slice(self):
        self.assertEqual(len(self.locus.slices), 1)
        for index in range(len(self.locus)):
            position = self.locus[index]
            self.assertIsInstance(position, gen.Position)
            self.assertEqual(position.offset, 20 + index)
            self.assertEqual(position.node.id, self.locus.start().node.id)
            self.assertEqual(position.strand, "+")
        self.assertEqual(self.locus[0], self.locus.start())
        self.assertEqual(self.locus[len(self.locus) - 1], self.locus.end())

    def test_negative_indices_and_boolean_keys(self):
        for index in range(1, len(self.locus) + 1):
            self.assertEqual(self.locus[-index], self.locus[len(self.locus) - index])
        self.assertEqual(self.locus[False], self.locus[0])
        self.assertEqual(self.locus[True], self.locus[1])
        self.assertEqual(self.locus[IndexKey()], self.locus[2])

    def test_forward_slice_within_node(self):
        result = self.locus[2:6]
        self.assertIsInstance(result, gen.Locus)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, self.locus.slice(2, 6))
        self.assertEqual((result.slices[0].start, result.slices[0].end), (22, 26))
        self.assertEqual(result[0], self.locus[2])
        self.assertEqual(result[-1], self.locus[5])

    def test_reverse_indexing_single_node(self):
        self.assert_reverse_correspondence(self.locus)

    def test_omitted_negative_and_clipped_slice_bounds(self):
        for locus in (self.locus, self.locus.reverse_complement()):
            self.assertEqual(locus[:], locus)
            self.assertEqual(locus[::1], locus)
            self.assertEqual(locus[:3], locus.slice(0, 3))
            self.assertEqual(locus[3:], locus.slice(3, 8))
            self.assertEqual(locus[-5:-1], locus.slice(3, 7))
            self.assertEqual(locus[-100:100], locus)
            self.assertEqual(locus[-(10**100) : 10**100], locus)
            self.assertEqual(locus[IndexKey() :], locus[2:])
        with self.assertRaises(IndexError):
            self.locus.slice(0, 100)

    def test_out_of_range_indices(self):
        for index in (8, -9, 10**100, -(10**100)):
            with self.subTest(index=index), self.assertRaises(IndexError):
                self.locus[index]

    def test_empty_and_inverted_slices(self):
        for key in (slice(0, 0), slice(5, 2), slice(8, None), slice(None, -100)):
            with self.subTest(key=key), self.assertRaises(IndexError):
                self.locus[key]

    def test_invalid_keys_and_steps(self):
        for key in (None, "1", 1.0, [], {}, (1, 2), object(), slice(1.5, 3)):
            with self.subTest(key=key), self.assertRaises(TypeError):
                self.locus[key]
        for step in (0, 2, -1):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.locus[::step]
        with self.assertRaises(TypeError):
            gen.Locus()
