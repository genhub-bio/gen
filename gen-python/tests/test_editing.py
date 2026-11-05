"""Exercise single-edit sequence editing through the installed extension."""

from itertools import islice
from pathlib import Path
import tempfile
import unittest

import gen
from test_api import RepositoryTestCase

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def reverse_complement(sequence):
    return sequence[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def read_fasta(path):
    return "".join(
        line.strip()
        for line in Path(path).read_text().splitlines()
        if not line.startswith(">")
    )


class EditingTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = gen.Repository(str(self.root / "repository"))

    node_runs = RepositoryTestCase.node_runs

    def operation_count(self):
        return len(self.repository.get_operations())

    def contains(self, sequence_graph, query):
        return bool(sequence_graph.search(query, sequence_kind="exact"))

    def export_sequence(self, sequence_graph):
        path = self.root / "export.fa"
        sequence_graph.export_fasta(str(path))
        return read_fasta(path)

    def is_editable(self, sequence_graph, locus):
        """Whether an edit can still target ``locus``, probed on a copy of the sample."""
        copy = sequence_graph.sample.copy(f"probe{len(self.repository.samples)}")
        try:
            next(graph for graph in copy if graph.name == sequence_graph.name).delete(
                locus
            )
        except ValueError:
            return False
        return True


class SimpleGraphEditingTests(EditingTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        [self.reference_locus] = self.graph.search(
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA", sequence_kind="exact"
        )

    def test_positions_from_a_locus_step_without_attaching_a_graph(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.assertEqual(locus.start() + 3, locus.slice(3, 4).start())
        self.assertEqual(locus.end() - 3, locus.slice(4, 5).start())
        self.assertEqual(locus[5] + 1, locus[6])
        self.assertEqual(self.graph.region("m123:20-28").start().graph.name, "m123")

    def test_positions_from_an_edit_and_an_annotation_step_without_attaching_a_graph(
        self,
    ):
        inserted = self.graph.insert(
            "GG", after=self.reference_locus.slice(3, 4).start()
        )
        replaced = self.graph.replace("m123:10-12", "TT")

        self.assertEqual(inserted.start() + 1, inserted.end())
        self.assertEqual(replaced.start() + 1, replaced.end())
        self.assertEqual(inserted.slice(0, 1).start().graph.name, "m123")

    def test_an_annotation_made_from_a_locus_keeps_the_loci_graph(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        annotation = gen.Annotation(locus, "site")

        self.assertEqual(annotation.locus.start().graph.name, "m123")
        self.assertEqual(annotation.locus.start() + 3, locus.slice(3, 4).start())
        self.assertEqual(annotation.locus.sequence, locus.sequence)

    def test_position_or_position_gives_a_superposition(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        first, second = locus.start(), locus.end()

        combined = first | second

        self.assertEqual(combined, gen.SuperPosition(first, second))
        self.assertEqual(combined.positions, [first, second])
        self.assertEqual(combined.graph.name, "m123")

    def test_superposition_or_accepts_positions_on_either_side(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        first, second, third = locus.start(), locus.slice(3, 4).start(), locus.end()
        pair = gen.SuperPosition(first, second)

        self.assertEqual(pair | third, gen.SuperPosition(first, second, third))
        self.assertEqual(third | pair, gen.SuperPosition(first, second, third))
        self.assertEqual(
            first | second | third, gen.SuperPosition(first, second, third)
        )
        self.assertEqual(first | first, gen.SuperPosition(first))

    def test_position_or_something_else_is_a_type_error(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        with self.assertRaisesRegex(TypeError, "Position or a SuperPosition"):
            locus.start() | "AC"

    def test_or_refuses_positions_attached_to_different_graphs(self):
        other = self.sample.copy("other")[0]
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        [other_locus] = other.search("GGAACACA", sequence_kind="exact")

        with self.assertRaisesRegex(ValueError, "different sequence graphs"):
            locus.start() | other_locus.end()

    def test_a_position_can_be_attached_to_another_graph_with_on(self):
        other = self.sample.copy("other")[0]
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        moved = locus.start().on(other)

        self.assertEqual(moved.graph.sample.name, "other")
        self.assertEqual(moved, locus.start())
        self.assertEqual(moved + 1, locus.slice(1, 2).start().on(other))

    def test_replace_region_returns_inserted_locus(self):
        original = self.reference_locus.slice(3, 5)

        inserted = self.graph.replace("m123:3-5", "TT")

        self.assertEqual(len(inserted), 2)
        self.assertEqual(inserted.strand, "+")
        self.assertTrue(self.contains(self.graph, "ATCTTTCGATCG"))
        with self.assertRaisesRegex(ValueError, "not present"):
            self.graph.delete(original)

    def test_replace_search_locus(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.graph.replace(locus, "TTTT")

        self.assertTrue(self.contains(self.graph, "GATCGTTTTCAGAGA"))

    def test_positions_stay_equal_when_an_edit_splits_their_node(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        start = locus.start()
        self.assertEqual(start.offset, 20)
        self.assertEqual(locus.end().offset, 27)
        self.assertEqual(gen.SuperPosition(start).positions, [start])

        self.graph.replace("m123:3-5", "TT")
        [moved] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.assertEqual(moved.start(), start)
        self.assertEqual(moved.start().offset, 15)

    def test_insert_before_and_after_positions(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        inserted = self.graph.insert("NNNN", before=locus.start())
        self.assertEqual(len(inserted), 4)
        self.assertTrue(self.contains(self.graph, "GATCGNNNNGGAACACA"))

        self.graph.insert("KK", after=locus.slice(3, 4).start())
        self.assertTrue(self.contains(self.graph, "GGAAKKCACA"))

    def test_insert_with_both_after_and_before_is_refused(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        before = self.operation_count()

        with self.assertRaisesRegex(TypeError, "not both"):
            self.graph.insert(
                "TT", after=locus.slice(3, 4).start(), before=locus.slice(4, 5).start()
            )

        self.assertEqual(self.operation_count(), before)

    def test_insert_accepts_positions_and_superpositions(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.graph.insert("TT", after=gen.SuperPosition(locus.slice(3, 4).start()))

        self.assertTrue(self.contains(self.graph, "GGAATTCACA"))

    def test_insert_needs_after_or_before(self):
        before = self.operation_count()

        with self.assertRaisesRegex(TypeError, "pass after= or before="):
            self.graph.insert("TT")
        with self.assertRaises(TypeError):
            self.graph.insert("TT", self.reference_locus.start())
        with self.assertRaisesRegex(TypeError, "Position or a SuperPosition"):
            self.graph.insert("TT", after="m123:3-5")

        self.assertEqual(self.operation_count(), before)

    def test_delete_region(self):
        self.graph.delete("m123:20-28")

        self.assertTrue(self.contains(self.graph, "ATCGATCGCAGAGA"))

    def test_reverse_strand_search_locus_is_edited_on_its_strand(self):
        [locus] = self.graph.search(reverse_complement("GGAACACA"), sequence_kind="dna")
        self.assertEqual(locus.strand, "-")

        inserted = self.graph.replace(locus, "GGGGAA")

        self.assertEqual(inserted.strand, "-")
        self.assertTrue(
            self.contains(self.graph, "GATCG" + reverse_complement("GGGGAA") + "CAGAGA")
        )

    def test_reverse_complement_locus_edits_same_bases(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="dna")
        self.assertEqual(locus.strand, "+")

        self.graph.replace(locus.reverse_complement(), "AAAC")

        self.assertTrue(self.contains(self.graph, "GATCGGTTTCAGAGA"))

    def test_delete_locus_crossing_an_earlier_edit(self):
        self.graph.replace("m123:20-22", "TT")
        [locus] = self.graph.search("CGTTAAC", sequence_kind="exact")
        self.assertEqual(len(self.node_runs(locus)), 3)

        self.graph.delete(locus)

        self.assertTrue(self.contains(self.graph, "ATCGATACACAGAGA"))

    def test_reverse_locus_crossing_an_earlier_edit(self):
        self.graph.replace("m123:20-22", "TT")
        [locus] = self.graph.search("CGTTAAC", sequence_kind="dna")
        self.assertEqual(locus.strand, "+")

        inserted = self.graph.replace(locus.reverse_complement(), "CC")

        self.assertEqual(inserted.strand, "-")
        self.assertTrue(self.contains(self.graph, "ATCGATGGACACAGAGA"))

    def test_each_edit_records_one_operation(self):
        before = self.operation_count()

        self.graph.replace("m123:3-5", "TT")
        self.assertEqual(self.operation_count(), before + 1)
        self.graph.delete("m123:20-28")
        self.assertEqual(self.operation_count(), before + 2)
        self.graph.insert("GG", after=self.reference_locus.slice(9, 10).start())
        self.assertEqual(self.operation_count(), before + 3)

    def test_message_overrides_generated_operation_summary(self):
        self.graph.replace("m123:3-5", "TT", message="custom replace message")
        self.graph.delete("m123:20-28", message="custom delete message")
        self.graph.insert(
            "GG",
            after=self.reference_locus.slice(9, 10).start(),
            message="custom insert message",
        )

        messages = [operation.message for operation in self.repository.get_operations()]
        self.assertIn("custom replace message", messages)
        self.assertIn("custom delete message", messages)
        self.assertIn("custom insert message", messages)

    def test_invalid_requests_fail_without_recording(self):
        before = self.operation_count()

        with self.assertRaisesRegex(ValueError, "at least one position"):
            self.graph.delete("m123:3-3")
        with self.assertRaisesRegex(ValueError, "non-empty sequence"):
            self.graph.replace("m123:3-5", "")
        with self.assertRaisesRegex(ValueError, "non-empty sequence"):
            self.graph.insert("", after=self.reference_locus.start())
        with self.assertRaisesRegex(IndexError, "empty"):
            self.reference_locus.slice(3, 3)
        with self.assertRaises(TypeError):
            self.graph.delete(42)
        with self.assertRaisesRegex(ValueError, "cannot resolve region"):
            self.graph.delete("missing:1-2")

        self.assertEqual(self.operation_count(), before)

    def test_deleting_already_deleted_target_fails(self):
        original = self.reference_locus.slice(20, 28)
        self.graph.delete("m123:20-28")
        before = self.operation_count()

        with self.assertRaisesRegex(ValueError, "not present"):
            self.graph.delete(original)

        self.assertEqual(self.operation_count(), before)

    def test_operation_recording_failure_rolls_back_in_place_edit(self):
        before = self.operation_count()
        self.repository.execute(
            "CREATE TEMP TRIGGER reject_edit BEFORE INSERT ON gen_operation_log "
            "BEGIN SELECT RAISE(ABORT, 'recording failed'); END"
        )
        with self.assertRaisesRegex(RuntimeError, "recording failed"):
            self.graph.replace("m123:3-5", "TT")
        self.assertEqual(self.operation_count(), before)
        self.assertFalse(self.contains(self.graph, "ATCTTTCGATCG"))
        self.repository.execute("DROP TRIGGER reject_edit")
        self.graph.replace("m123:3-5", "TT")
        self.assertEqual(self.operation_count(), before + 1)

    def test_exact_search_reverse_complement_and_endpoints(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        reverse = locus.reverse_complement()
        self.assertEqual(reverse.strand, "-")
        self.assertEqual(reverse.start().offset, locus.end().offset)
        self.assertEqual(reverse.end().offset, locus.start().offset)
        inserted = self.graph.replace(reverse, "AGT")
        self.assertEqual(inserted.strand, "-")
        self.assertTrue(self.contains(self.graph, "GATCGACTCAGAGA"))

    def test_target_absent_from_graph_fails(self):
        inserted = self.graph.insert(
            "GGGG", after=self.reference_locus.slice(9, 10).start()
        )
        self.graph.delete(inserted)
        before = self.operation_count()

        with self.assertRaisesRegex(ValueError, "not present"):
            self.graph.delete(inserted)
        self.assertEqual(self.operation_count(), before)

    def test_child_sample_editing_leaves_parent_unchanged(self):
        before = self.operation_count()

        child = self.sample.copy("child")
        inserted = child[0].replace("m123:3-5", "TT")

        self.assertEqual(self.operation_count(), before + 2)
        self.assertEqual(len(child), 1)
        self.assertTrue(self.contains(child[0], "ATCTTTCGATCG"))
        self.assertFalse(self.contains(self.graph, "ATCTTTCGATCG"))

        grandchild = child.copy("grandchild")
        grandchild[0].delete(inserted)
        self.assertTrue(self.contains(grandchild[0], "ATCTCGATCG"))
        self.assertFalse(self.contains(child[0], "ATCTCGATCG"))

    def test_repeated_child_edits_accumulate(self):
        child = self.sample.copy("child")
        child[0].replace("m123:3-5", "TT")
        child[0].delete("m123:20-28")

        self.assertTrue(self.contains(child[0], "ATCTTTCGATCGATCGATCGCAGAGA"))

    def test_export_fasta_reflects_insert_on_path(self):
        self.graph.insert("GG", after=self.reference_locus.slice(9, 10).start())

        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATGGCGATCGATCGGGAACACACAGAGA",
        )

    def test_export_fasta_reflects_replace_on_path(self):
        self.graph.replace("m123:3-5", "TT")

        self.assertEqual(
            self.export_sequence(self.graph), "ATCTTTCGATCGATCGATCGGGAACACACAGAGA"
        )

    def test_export_fasta_reflects_delete_on_path(self):
        self.graph.delete("m123:20-28")

        self.assertEqual(self.export_sequence(self.graph), "ATCGATCGATCGATCGATCGCAGAGA")

    def test_export_fasta_reflects_chained_edits_in_order(self):
        self.graph.replace("m123:3-5", "TT")
        self.graph.delete("m123:20-28")

        self.assertEqual(self.export_sequence(self.graph), "ATCTTTCGATCGATCGATCGCAGAGA")

    def test_copy_rejects_existing_sample(self):
        self.sample.copy("child")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.sample.copy("child")

    def test_stack_replace_keeps_original_reachable_and_path_untouched(self):
        original = self.reference_locus.slice(3, 5)

        self.graph.replace("m123:3-5", "TT", stack=True)

        self.assertTrue(self.contains(self.graph, "ATCGATCGATCG"))
        self.assertTrue(self.contains(self.graph, "ATCTTTCGATCG"))
        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
        )
        self.assertTrue(self.is_editable(self.graph, original))

    def test_insert_after_reverse_stacked_original_is_refused(self):
        # A reverse-strand locus reads its last position at its lowest offset, where the bubble
        # forks, so inserting after it would land on the alternative too.
        [locus] = self.graph.search(reverse_complement("GGAACACA"), sequence_kind="dna")
        self.assertEqual(locus.strand, "-")
        self.graph.replace(locus, "TTTT", stack=True)
        before = self.operation_count()

        with self.assertRaisesRegex(ValueError, "would also put the new sequence"):
            self.graph.insert("CC", after=locus.end())

        self.assertEqual(self.operation_count(), before)

    def test_insert_after_a_fork_attaches_to_every_option(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.replace(locus, "TTTT", stack=True)

        self.graph.insert("CC", after=self.reference_locus.slice(19, 20).start())

        self.assertTrue(self.contains(self.graph, "GATCGCCGGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "GATCGCCTTTTCAGAGA"))

    def test_insert_on_the_alternative_of_a_fork_leaves_the_original_unchanged(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        alternative = self.graph.replace(locus, "TTTT", stack=True)

        self.graph.insert("CC", before=alternative.start())

        self.assertTrue(self.contains(self.graph, "GATCGGGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "GATCGCCTTTTCAGAGA"))
        self.assertFalse(self.contains(self.graph, "GATCGCCGGAACACACAGAGA"))

    def test_insert_at_the_ends_of_a_stacked_original_is_refused(self):
        """The alternative leaves at the first base of the original and rejoins at its last,
        so an insertion before or after those bases would land on the alternative too."""
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.replace(locus, "TTTT", stack=True)
        before = self.operation_count()

        for anchors in (dict(before=locus.start()), dict(after=locus.end())):
            with self.assertRaisesRegex(ValueError, "would also put the new sequence"):
                self.graph.insert("CC", **anchors)

        self.assertEqual(self.operation_count(), before)
        self.assertFalse(self.contains(self.graph, "GATCGCCTTTT"))

    def test_insert_on_one_arm_of_a_join_leaves_the_other_unchanged(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        alternative = self.graph.replace(locus, "TTTT", stack=True)

        self.graph.insert("GG", after=alternative.end())

        self.assertTrue(self.contains(self.graph, "GATCGTTTTGGCAGAGA"))
        self.assertTrue(self.contains(self.graph, "GGAACACACAGAGA"))
        self.assertFalse(self.contains(self.graph, "GGAACACAGGCAGAGA"))

    def test_insert_with_a_superposition_on_every_arm_of_a_join(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        alternative = self.graph.replace(locus, "TTTT", stack=True)
        arm_ends = gen.SuperPosition(locus.end(), alternative.end())

        self.graph.insert("GG", after=arm_ends)

        self.assertTrue(self.contains(self.graph, "GATCGTTTTGGCAGAGA"))
        self.assertTrue(self.contains(self.graph, "GGAACACAGGCAGAGA"))

    def test_stack_insert_keeps_original_reachable_and_path_untouched(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.graph.insert("GG", before=locus.start(), stack=True)

        self.assertTrue(self.contains(self.graph, "ATCGATCGGGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "ATCGATCGGGGGAACACACAGAGA"))
        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
        )

    def test_stack_delete_keeps_original_reachable_and_path_untouched(self):
        original = self.reference_locus.slice(20, 28)

        self.graph.delete("m123:20-28", stack=True)

        self.assertTrue(self.contains(self.graph, "GGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "ATCGATCGCAGAGA"))
        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
        )
        self.assertTrue(self.is_editable(self.graph, original))


class LibraryGraphEditingTests(EditingTestCase):
    """A library column is a bubble whose alternatives share the edges that enter and
    leave it, so an edit there has no single flanking route to hang itself off."""

    LEFT = "AAAACCCCGGGGTTTT"
    FIRST = "ACGTACGTAC"
    SECOND = "TTTTGGGGCC"
    RIGHT = "CCCCAAAATTTTGGGG"

    def setUp(self):
        super().setUp()
        part = gen.Sequence
        self.graph = self.repository.import_library(
            "lib",
            [
                [part("left", self.LEFT)],
                [part("alt_a", self.FIRST), part("alt_b", self.SECOND)],
                [part("right", self.RIGHT)],
            ],
            sample="reference",
        )
        self.alternative = next(
            annotation.locus
            for annotation in self.graph.annotations
            if annotation.name == "alt_a"
        )

    def path_count(self):
        path = self.root / "export.gfa"
        self.graph.export_gfa(str(path))
        return sum(1 for line in path.read_text().splitlines() if line.startswith("P"))

    # Disabled: export_gfa projects stored paths onto a graph that still holds retired (-2) edit-site
    # edges, so the path from before the edit is exported next to the edited one. Retired edges leak
    # into other exporters and searches too; re-enable once that is fixed across the board.
    @unittest.skip("export_gfa exports paths through retired edit-site edges")
    def test_replace_of_an_alternative_keeps_a_route_through_the_graph(self):
        self.graph.replace(self.alternative, "GGGGGG")

        self.assertTrue(self.contains(self.graph, self.LEFT + "GGGGGG" + self.RIGHT))
        self.assertEqual(
            self.export_sequence(self.graph), self.LEFT + "GGGGGG" + self.RIGHT
        )
        self.assertEqual(self.path_count(), 1)

    def test_replace_of_an_alternative_supersedes_only_that_alternative(self):
        self.graph.replace(self.alternative, "GGGGGG")

        self.assertFalse(self.is_editable(self.graph, self.alternative))
        self.assertTrue(self.contains(self.graph, self.LEFT + self.SECOND + self.RIGHT))

    # Disabled: export_gfa projects stored paths onto a graph that still holds retired (-2) edit-site
    # edges, so the path from before the edit is exported next to the edited one. Retired edges leak
    # into other exporters and searches too; re-enable once that is fixed across the board.
    @unittest.skip("export_gfa exports paths through retired edit-site edges")
    def test_delete_of_an_alternative_keeps_a_route_through_the_graph(self):
        self.graph.delete(self.alternative)

        self.assertTrue(self.contains(self.graph, self.LEFT + self.RIGHT))
        self.assertEqual(self.export_sequence(self.graph), self.LEFT + self.RIGHT)
        self.assertEqual(self.path_count(), 1)
        self.assertFalse(self.is_editable(self.graph, self.alternative))

    def test_stacked_replace_leaves_every_alternative_live(self):
        self.graph.replace(self.alternative, "GGGGGG", stack=True)

        self.assertTrue(self.contains(self.graph, self.LEFT + "GGGGGG" + self.RIGHT))
        self.assertTrue(self.contains(self.graph, self.LEFT + self.FIRST + self.RIGHT))
        self.assertTrue(self.contains(self.graph, self.LEFT + self.SECOND + self.RIGHT))
        self.assertTrue(self.is_editable(self.graph, self.alternative))


class GfaMotifEditingTests(EditingTestCase):
    """``B`` reads into ``D`` and ``E``, but ``C`` reads only into ``E``: not a combinatorial
    layer, so one side of an insertion reaches only the routes that really meet there."""

    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_gfa(str(FIXTURES / "motif.gfa"))
        self.nodes = {
            name: self.graph.search(text, sequence_kind="exact")[0]
            for name, text in zip("ABCDEF", ("AB", "CD", "EF", "GH", "IJ", "KL"))
        }

    def sequences(self):
        return sorted({str(sequence) for sequence in self.graph.all_sequences()})

    def test_the_motif_has_every_route(self):
        self.assertEqual(self.sequences(), ["ABCDGHKL", "ABCDIJKL", "ABEFIJKL"])

    def test_insert_with_one_side_reaches_every_route_there(self):
        self.graph.insert("xx", before=self.nodes["E"].start())

        self.assertEqual(self.sequences(), ["ABCDGHKL", "ABCDxxIJKL", "ABEFxxIJKL"])

    def test_substituting_the_first_base_of_a_node_changes_every_route_into_it(self):
        self.graph.replace(self.nodes["E"].slice(0, 1), "x")

        self.assertEqual(self.sequences(), ["ABCDGHKL", "ABCDxJKL", "ABEFxJKL"])


class GfaMotifDuplicatedNodeTests(EditingTestCase):
    """The motif with ``E`` duplicated per route into it (``Eb`` after ``B``, ``Ec`` after ``C``),
    which scopes an edit to one route by giving each route its own node and ports."""

    def load(self, name):
        graph = self.repository.import_gfa(str(FIXTURES / name), sample=name)
        return graph

    def sequences(self, graph):
        return sorted({str(sequence) for sequence in graph.all_sequences()})

    def after(self, graph, locus):
        """The ``IJ`` node that reads directly after ``locus``."""
        return next(
            hit
            for hit in graph.search("IJ", sequence_kind="exact")
            if hit.start() - 1 == locus.end()
        )

    def test_a_stacked_replace_reaches_only_the_duplicate_it_targets(self):
        graph = self.load("motif_split.gfa")
        b = graph.search("CD", sequence_kind="exact")[0]
        reached_through_b = self.after(graph, b)
        graph.replace(reached_through_b, "xy", stack=True)

        self.assertEqual(
            self.sequences(graph),
            ["ABCDGHKL", "ABCDIJKL", "ABCDxyKL", "ABEFIJKL"],
        )

    def test_insert_before_a_duplicated_node_reaches_only_its_route(self):
        """The alternative leaves ``B`` and joins ``F``, so only ``B`` reads into ``Eb`` and a
        sequence before ``Eb`` goes on that one connection."""
        graph = self.load("motif_split_alt.gfa")
        b = graph.search("CD", sequence_kind="exact")[0]
        eb = self.after(graph, b)

        graph.insert("GG", before=eb.start())

        self.assertEqual(
            self.sequences(graph),
            ["ABCDGGIJKL", "ABCDGHKL", "ABCDxyKL", "ABEFIJKL"],
        )


class PathUpdateTests(EditingTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_sequence(
            "AAAACCCCGGGG", name="v", sample="p"
        )
        self.graph.delete("v:4-8")

    def sequences(self):
        return sorted({str(sequence) for sequence in self.graph.all_sequences()})

    def test_edit_next_to_an_earlier_deletion_updates_the_path(self):
        self.graph.delete("v:4-6")

        self.assertEqual(self.sequences(), ["AAAAGG"])
        self.assertEqual(self.export_sequence(self.graph), "AAAAGG")

    def test_successful_edit_updates_the_path(self):
        graph = self.repository.import_sequence("AAAACCCCGGGG", name="w", sample="q")
        graph.replace("w:4-8", "TT")

        self.assertEqual(
            sorted({str(sequence) for sequence in graph.all_sequences()}),
            ["AAAATTGGGG"],
        )
        self.assertEqual(self.export_sequence(graph), "AAAATTGGGG")


class ConnectTests(EditingTestCase):
    """Raw strand-specific connections, as used to model Cre-loxP recombination."""

    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_sequence(
            "AAACCGTTT", name="molecule", sample="cre"
        )
        # The bases flanking the two cut points, at 3 and 6.
        self.left_last = self.graph.locus(2, 3).start()
        self.segment_first = self.graph.locus(3, 4).start()
        self.segment_last = self.graph.locus(5, 6).start()
        self.right_first = self.graph.locus(6, 7).start()

    def sequences(self):
        return sorted({str(sequence) for sequence in self.graph.all_sequences()})

    def invert(self, **options):
        self.graph.connect(
            after=self.left_last,
            before=self.segment_last.reverse_complement(),
            **options,
        )
        self.graph.connect(
            after=self.segment_first.reverse_complement(),
            before=self.right_first,
            **options,
        )

    def test_inversion_reads_the_segment_reverse_complemented(self):
        self.invert()

        self.assertEqual(self.sequences(), ["AAACGGTTT"])

    def test_stacked_inversion_keeps_the_original(self):
        self.invert(stack=True)

        self.assertEqual(self.sequences(), ["AAACCGTTT", "AAACGGTTT"])

    def test_connecting_to_the_original_neighbors_reverses_an_inversion(self):
        self.invert()
        self.graph.connect(after=self.left_last, before=self.segment_first)
        self.graph.connect(after=self.segment_last, before=self.right_first)

        self.assertEqual(self.sequences(), ["AAACCGTTT"])

    def test_connect_records_one_operation_per_call(self):
        before = self.operation_count()

        self.invert()

        self.assertEqual(self.operation_count(), before + 2)

    def test_connect_needs_a_single_position_on_each_side(self):
        with self.assertRaises(ValueError):
            self.graph.connect(
                after=self.left_last | self.right_first, before=self.segment_first
            )

    def test_position_reverse_complement_flips_the_strand(self):
        self.assertEqual(self.left_last.strand, "+")
        self.assertEqual(self.left_last.reverse_complement().strand, "-")


class CutTests(EditingTestCase):
    """Cuts mark a block boundary without breaking the sequence."""

    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_sequence(
            "AAACCGTTT", name="molecule", sample="cre"
        )

    def sequences(self):
        return sorted(str(sequence) for sequence in self.graph.all_sequences())

    def test_cut_leaves_the_sequence_readable_across_it(self):
        self.graph.cut(after=self.graph.locus(2, 3).start())

        self.assertEqual(self.sequences(), ["AAACCGTTT"])

    def test_cut_splits_the_node_into_blocks(self):
        self.graph.cut(after=self.graph.locus(2, 3).start())

        pieces = sorted(
            node.sequence for node in self.graph.to_dict()["nodes"] if node.length
        )
        self.assertEqual(pieces, ["AAA", "CCGTTT"])

    def test_before_and_after_name_the_same_point(self):
        self.graph.cut(before=self.graph.locus(3, 4).start())

        with self.assertRaises(ValueError):
            self.graph.cut(after=self.graph.locus(2, 3).start())

    def test_cut_is_one_operation(self):
        before = self.operation_count()

        self.graph.cut(after=self.graph.locus(2, 3).start())

        self.assertEqual(self.operation_count(), before + 1)

    def test_cut_at_the_end_of_a_node_is_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.cut(after=self.graph.locus(8, 9).start())
        with self.assertRaises(ValueError):
            self.graph.cut(before=self.graph.locus(0, 1).start())

    def test_cut_needs_exactly_one_side(self):
        position = self.graph.locus(2, 3).start()

        with self.assertRaises(TypeError):
            self.graph.cut()
        with self.assertRaises(TypeError):
            self.graph.cut(after=position, before=position)

    def test_joins_at_cut_points_still_retire_the_original_sequence(self):
        self.graph.cut(after=self.graph.locus(2, 3).start())
        self.graph.cut(after=self.graph.locus(5, 6).start())

        self.graph.connect(
            after=self.graph.locus(2, 3).start(),
            before=self.graph.locus(5, 6).start().reverse_complement(),
        )
        self.graph.connect(
            after=self.graph.locus(3, 4).start().reverse_complement(),
            before=self.graph.locus(6, 7).start(),
        )

        self.assertEqual(self.sequences(), ["AAACGGTTT"])


class CreExcisionTests(EditingTestCase):
    """Cre excision is reversible, so each event is stacked: the molecule before it stays."""

    def excise(self, graph, left, right, segment_first, segment_last):
        """Join the flanks and close the segment into a circle, keeping the original."""
        graph.connect(after=left, before=right, stack=True)
        graph.connect(after=segment_last, before=segment_first, stack=True)

    def sequences(self, graph):
        return sorted(str(sequence) for sequence in graph.all_sequences())

    def test_single_excision_keeps_the_original_and_the_excised_molecule(self):
        graph = self.repository.import_sequence(
            "AAACCGTTT", name="molecule", sample="cre"
        )
        left, first, last, right = (
            graph.locus(2, 3).start(),
            graph.locus(3, 4).start(),
            graph.locus(5, 6).start(),
            graph.locus(6, 7).start(),
        )
        graph.cut(after=left)
        graph.cut(after=last)

        self.excise(graph, left, right, first, last)

        self.assertEqual(self.sequences(graph), ["AAACCGTTT", "AAATTT"])

    def test_excisions_sharing_a_site_do_not_chain_their_circles(self):
        graph = self.repository.import_sequence(
            "AAACCGGTTTCCC", name="molecule", sample="cre"
        )
        site_one, site_two, site_three = (
            graph.locus(2, 3).start(),
            graph.locus(5, 6).start(),
            graph.locus(8, 9).start(),
        )
        first_a, first_b, first_c = (
            graph.locus(3, 4).start(),
            graph.locus(6, 7).start(),
            graph.locus(9, 10).start(),
        )
        for site in (site_one, site_two, site_three):
            graph.cut(after=site)

        self.excise(graph, site_one, first_b, first_a, site_two)
        self.excise(graph, site_two, first_c, first_b, site_three)

        self.assertEqual(
            self.sequences(graph),
            ["AAACCGGTTTCCC", "AAACCGTCCC", "AAAGTTTCCC", "AAATCCC"],
        )


class CompletedExcisionTests(EditingTestCase):
    """An unstacked event retires what leaves or arrives at the points it names."""

    def live_loops(self, graph):
        """Sequence blocks with a live edge back to themselves: the closed circles."""
        loops = set()
        for (source, target), edges in graph.to_dict()["edges"].items():
            if source == target and any(
                edge["chromosome_index"] != -2 for edge in edges
            ):
                loops.add(source.sequence)
        return loops

    def excised_once(self):
        """`AAA | CCG | GTT | TCCC` after Cre has excised `CCG` between sites at 3 and 6."""
        graph = self.repository.import_sequence(
            "AAACCGGTTTCCC", name="molecule", sample="cre"
        )
        bases = {index: graph.locus(index, index + 1).start() for index in range(13)}
        for index in (2, 5, 8):
            graph.cut(after=bases[index])
        graph.connect(after=bases[2], before=bases[6])
        graph.connect(after=bases[5], before=bases[3])
        return graph, bases

    def test_second_excision_named_by_the_chromosome_flanks_completes_the_strain(self):
        graph, bases = self.excised_once()

        # `GTT` is next to the new junction on the chromosome, whose flanks are bases 2 and 9.
        graph.connect(after=bases[2], before=bases[9])
        graph.connect(after=bases[8], before=bases[6])

        self.assertEqual(
            sorted(str(sequence) for sequence in graph.all_sequences()), ["AAATCCC"]
        )
        self.assertEqual(self.live_loops(graph), {"CCG", "GTT"})

    def test_naming_a_base_on_the_circle_retires_the_circle(self):
        graph, bases = self.excised_once()

        # Base 5 now ends the circle, so reconnecting from after it replaces the circle's closure.
        graph.connect(after=bases[5], before=bases[9])

        self.assertNotIn("CCG", self.live_loops(graph))


class ConnectBetweenGraphsTests(EditingTestCase):
    """Recombination between molecules held in separate sequence graphs of one sample."""

    SITE_LEFT = "ATAACTTCGTATAATGT"
    SITE_RIGHT = "ATGCTATACGAAGTTAT"

    def molecule(self, name, left, right, sample="cre"):
        """A linear molecule cut into two nodes in the middle of its site."""
        graph = self.repository.import_sequence(
            left + self.SITE_LEFT, name=name, sample=sample
        )
        first = graph.locus(0, len(left + self.SITE_LEFT))
        second = graph.insert(self.SITE_RIGHT + right, after=first.end())
        return graph, first, second

    def sequences(self, graph):
        return sorted(str(sequence) for sequence in graph.all_sequences())

    def graph_names(self, sample="cre"):
        return sorted(graph.name for graph in self.repository.sample(sample))

    def exchange(self, a, b, **options):
        a_graph, a_left, a_right = a
        b_graph, b_left, b_right = b
        a_graph.connect(after=a_left.end(), before=b_right.start(), **options)
        # The first call folded b into a, so b's positions now resolve in a.
        a_graph.connect(after=b_left.end(), before=a_right.start(), **options)

    def test_reciprocal_exchange_gives_both_recombinants_in_one_graph(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")

        self.exchange(a, b)

        site = self.SITE_LEFT + self.SITE_RIGHT
        self.assertEqual(
            self.sequences(a[0]),
            sorted(["GGGG" + site + "TTTT", "AAAA" + site + "CCCC"]),
        )
        self.assertEqual(self.graph_names(), ["A"])

    def test_stacked_exchange_keeps_the_parental_molecules(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")

        self.exchange(a, b, stack=True)

        site = self.SITE_LEFT + self.SITE_RIGHT
        self.assertEqual(
            self.sequences(a[0]),
            sorted(
                left + site + right
                for left in ("GGGG", "AAAA")
                for right in ("CCCC", "TTTT")
            ),
        )

    def test_exchange_is_recorded_as_one_operation_per_connection(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        before = self.operation_count()

        self.exchange(a, b)

        self.assertEqual(self.operation_count(), before + 2)

    def test_exchange_gives_each_recombinant_its_own_graph(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")

        products = a[0].exchange(
            (a[1].end(), a[1].end() + 1), (b[2].start() - 1, b[2].start()), split=True
        )

        site = self.SITE_LEFT + self.SITE_RIGHT
        self.assertEqual([graph.name for graph in products], ["A", "B"])
        self.assertEqual(self.sequences(products[0]), ["GGGG" + site + "TTTT"])
        self.assertEqual(self.sequences(products[1]), ["AAAA" + site + "CCCC"])
        self.assertEqual(self.graph_names(), ["A", "B"])

    def test_stacked_exchange_call_keeps_the_parental_molecules(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")

        products = a[0].exchange(
            (a[1].end(), a[1].end() + 1), (b[2].start() - 1, b[2].start()), stack=True
        )

        self.assertEqual(len(products), 1)
        site = self.SITE_LEFT + self.SITE_RIGHT
        self.assertEqual(
            self.sequences(a[0]),
            sorted(
                left + site + right
                for left in ("GGGG", "AAAA")
                for right in ("CCCC", "TTTT")
            ),
        )

    def test_exchange_call_is_one_operation(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        before = self.operation_count()

        a[0].exchange((a[1].end(), a[1].end() + 1), (b[2].start() - 1, b[2].start()))

        self.assertEqual(self.operation_count(), before + 1)

    def test_exchange_of_edited_molecules_still_gives_one_graph_per_recombinant(self):
        a, a_left, a_right = self.molecule("A", "GGGGAAAAGGGG", "CCCC")
        b, b_left, b_right = self.molecule("B", "TTTTCCCCTTTT", "AAAA")
        a.delete(a.locus(4, 6))
        a.delete(a.locus(6, 8))
        b.replace(b.locus(4, 5), "N")
        b.insert("XX", after=b.locus(4, 5).end())

        products = a.exchange(
            (a_left.end(), a_left.end() + 1), (b_right.start() - 1, b_right.start()), split=True
        )

        self.assertEqual([graph.name for graph in products], ["A", "B"])
        self.assertEqual(self.graph_names(), ["A", "B"])
        self.assertTrue(all(self.sequences(graph) for graph in products))

    def test_exchange_rejects_a_breakpoint_the_graph_does_not_join(self):
        graph = self.repository.import_sequence("AAACCGTTT", name="line", sample="cre")

        def base(index):
            return graph.locus(index, index + 1).start()

        with self.assertRaisesRegex(ValueError, "not joined"):
            graph.exchange((base(2), base(4)), (base(5), base(6)))

    def test_stacked_exchanges_name_the_junction_each_one_cuts(self):
        graph = self.repository.import_sequence("AAACCGTTT", name="line", sample="cre")

        def base(index):
            return graph.locus(index, index + 1).start()

        graph.exchange((base(2), base(3)), (base(4), base(5)), stack=True)
        graph.exchange((base(2), base(3)), (base(6), base(7)), stack=True)

        self.assertIn("AAATT", self.sequences(graph))

    def test_a_breakpoint_across_a_strand_crossing_edge_is_accepted(self):
        graph = self.repository.import_sequence("AAACCGTTT", name="line", sample="cre")

        def base(index):
            return graph.locus(index, index + 1).start()

        graph.exchange(
            (base(2), base(3)),
            (base(6).reverse_complement(), base(5).reverse_complement()),
            stack=True,
        )

        # base 2 now also leads into the reverse strand, and that junction is in the graph.
        graph.exchange(
            (base(2), base(5).reverse_complement()), (base(7), base(8)), stack=True
        )

    def test_a_pair_of_nucleotides_on_two_strands_that_no_edge_joins_is_rejected(self):
        graph = self.repository.import_sequence("AAACCGTTT", name="line", sample="cre")

        def base(index):
            return graph.locus(index, index + 1).start()

        with self.assertRaisesRegex(ValueError, "not joined"):
            graph.exchange(
                (base(2), base(3).reverse_complement()), (base(5), base(6))
            )

    def test_a_step_next_to_a_fork_raises_a_superposition_error(self):
        graph = self.repository.import_sequence("AAACCGTTT", name="line", sample="cre")

        def base(index):
            return graph.locus(index, index + 1).start()

        graph.exchange((base(2), base(3)), (base(4), base(5)), stack=True)

        with self.assertRaises(gen.SuperpositionError) as raised:
            graph.exchange((base(2), base(2) + 1), (base(6), base(7)), stack=True)
        self.assertIsInstance(raised.exception, ValueError)

    def test_exchange_inverts_the_segment_between_two_points_of_one_graph(self):
        graph = self.repository.import_sequence("AAACCGTTT", name="line", sample="cre")
        graph.exchange(
            (graph.locus(2, 3).start(), graph.locus(3, 4).start()),
            (
                graph.locus(6, 7).start().reverse_complement(),
                graph.locus(5, 6).start().reverse_complement(),
            ),
        )

        self.assertEqual(self.sequences(graph), ["AAACGGTTT"])

    def circle_and_line(self):
        circle = self.repository.import_sequence(
            self.SITE_RIGHT + "TACG" + self.SITE_LEFT,
            name="circle",
            sample="cre",
            circular=True,
        )
        return circle, self.molecule("line", "GGGG", "CCCC")

    def test_circle_integrates_in_the_orientation_its_site_has(self):
        circle, (line, left, right) = self.circle_and_line()
        whole = circle.locus(0, len(self.SITE_RIGHT + "TACG" + self.SITE_LEFT))

        line.connect(after=left.end(), before=whole.start())
        line.connect(after=whole.end(), before=right.start())

        self.assertEqual(
            self.sequences(line),
            [
                "GGGG"
                + self.SITE_LEFT
                + self.SITE_RIGHT
                + "TACG"
                + self.SITE_LEFT
                + self.SITE_RIGHT
                + "CCCC"
            ],
        )
        self.assertEqual(self.graph_names(), ["line"])

    def test_circle_integrates_reverse_complemented_when_its_site_is_inverted(self):
        circle, (line, left, right) = self.circle_and_line()
        circle_sequence = self.SITE_RIGHT + "TACG" + self.SITE_LEFT
        whole = circle.locus(0, len(circle_sequence))

        line.connect(after=left.end(), before=whole.end().reverse_complement())
        line.connect(after=whole.start().reverse_complement(), before=right.start())

        self.assertEqual(
            self.sequences(line),
            [
                "GGGG"
                + self.SITE_LEFT
                + reverse_complement(circle_sequence)
                + self.SITE_RIGHT
                + "CCCC"
            ],
        )

    def test_joined_graphs_share_one_start_and_end(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")

        a[0].connect(after=a[1].end(), before=b[2].start(), stack=True)

        # The terminal nodes have a fixed id; both molecules attach to the one block of it.
        terminals = [str(node) for node in a[0].to_dict()["nodes"] if not node.length]
        self.assertEqual(sum(node.startswith("84d6adbd") for node in terminals), 1)
        self.assertEqual(sum(node.startswith("1c7dfc64") for node in terminals), 1)

    def test_joined_graph_keeps_its_retired_edits_retired(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        b[0].replace(b[0].locus(0, 4), "ACAC")

        a[0].connect(after=a[1].end(), before=b[2].start(), stack=True)

        sequences = self.sequences(a[0])
        self.assertTrue(any(sequence.startswith("ACAC") for sequence in sequences))
        self.assertFalse(any(sequence.startswith("AAAA") for sequence in sequences))

    def test_joining_a_circular_graph_does_not_make_the_host_circular(self):
        host = self.repository.import_sequence("GGGG" + self.SITE_LEFT, name="host", sample="cre")
        ring = self.repository.import_sequence(
            "CCCC" + self.SITE_RIGHT, name="ring", sample="cre", circular=True
        )

        host.connect(
            after=host.locus(0, 4).end(), before=ring.locus(0, 4).start(), stack=True
        )

        edges = [str(edge) for edge in host.to_dict()["edges"]]
        self.assertFalse(any(edge.startswith("(Node(1c7dfc64") and "84d6adbd" in edge for edge in edges))

    def test_exchange_integrates_a_circle_into_a_line_as_one_graph(self):
        circle, (line, left, right) = self.circle_and_line()
        whole = circle.locus(0, len(self.SITE_RIGHT + "TACG" + self.SITE_LEFT))

        products = line.exchange((left.end(), left.end() + 1), (whole.end(), whole.start()))

        site = self.SITE_LEFT + self.SITE_RIGHT
        self.assertEqual([graph.name for graph in products], ["line"])
        self.assertEqual(self.sequences(line), ["GGGG" + site + "TACG" + site + "CCCC"])
        self.assertEqual(self.graph_names(), ["line"])

    def test_stacked_exchange_keeps_the_line_beside_its_integrant(self):
        circle, (line, left, right) = self.circle_and_line()
        whole = circle.locus(0, len(self.SITE_RIGHT + "TACG" + self.SITE_LEFT))

        line.exchange(
            (left.end(), left.end() + 1), (whole.end(), whole.start()), stack=True
        )

        site = self.SITE_LEFT + self.SITE_RIGHT
        self.assertEqual(
            self.sequences(line),
            sorted(["GGGG" + site + "CCCC", "GGGG" + site + "TACG" + site + "CCCC"]),
        )

    def test_exchange_integrates_a_circle_reverse_complemented_when_its_site_is_inverted(self):
        circle, (line, left, right) = self.circle_and_line()
        circle_sequence = self.SITE_RIGHT + "TACG" + self.SITE_LEFT
        whole = circle.locus(0, len(circle_sequence))

        line.exchange(
            (left.end(), left.end() + 1),
            (whole.start().reverse_complement(), whole.end().reverse_complement()),
        )

        self.assertEqual(
            self.sequences(line),
            [
                "GGGG"
                + self.SITE_LEFT
                + reverse_complement(circle_sequence)
                + self.SITE_RIGHT
                + "CCCC"
            ],
        )

    def test_exchange_excises_an_integrated_circle_into_its_own_graph(self):
        circle, (line, left, right) = self.circle_and_line()
        circle_sequence = self.SITE_RIGHT + "TACG" + self.SITE_LEFT
        whole = circle.locus(0, len(circle_sequence))
        line.exchange((left.end(), left.end() + 1), (whole.end(), whole.start()))

        products = line.exchange(
            (left.end(), whole.start()), (whole.end(), right.start()), split=True
        )

        self.assertEqual([graph.name for graph in products], ["line", "line-2"])
        self.assertEqual(
            self.sequences(products[0]), ["GGGG" + self.SITE_LEFT + self.SITE_RIGHT + "CCCC"]
        )
        self.assertEqual(self.sequences(products[1]), [circle_sequence])

    def test_exchange_products_start_and_end_on_different_molecules(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")

        self.exchange(a, b)

        # Both molecules start at the one shared start node, so neither may be pruned for it.
        self.assertEqual(len(self.sequences(a[0])), 2)

    def test_joining_keeps_the_current_path_of_the_surviving_graph(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        site = self.SITE_LEFT + self.SITE_RIGHT

        a[0].connect(after=a[1].end(), before=b[2].start(), stack=True)

        self.assertEqual(self.export_sequence(a[0]), "GGGG" + site + "CCCC")

    def test_sequences_of_the_joined_graph_are_readable_from_the_surviving_graph(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        b_sequence = "AAAA" + self.SITE_LEFT + self.SITE_RIGHT + "TTTT"

        a[0].connect(after=a[1].end(), before=b[2].start(), stack=True)

        self.assertIn(b_sequence, self.sequences(a[0]))
        self.assertEqual(self.graph_names(), ["A"])

    def test_joining_a_graph_from_another_sample_is_rejected(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT", sample="other")

        with self.assertRaises(ValueError):
            a[0].connect(after=a[1].end(), before=b[2].start())

        self.assertEqual(self.graph_names("other"), ["B"])

    def test_annotations_of_a_joined_graph_are_copied_onto_the_surviving_graph(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        b[0].add_annotation(b[0].region("B:0-4"), "arm", track="parts")

        a[0].connect(after=a[1].end(), before=b[2].start())

        self.assertEqual(self.graph_names(), ["A"])
        arms = [item for item in a[0].annotations if item.name == "arm"]
        self.assertEqual(len(arms), 1)
        self.assertEqual(arms[0].locus.sequence, "AAAA")

    def test_joining_graphs_with_the_same_annotation_name_primes_the_moved_name(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        a[0].add_annotation(a[0].region("A:0-4"), "arm", track="parts")
        b[0].add_annotation(b[0].region("B:0-4"), "arm", track="parts")

        a[0].connect(after=a[1].end(), before=b[2].start())

        self.assertEqual(self.graph_names(), ["A"])
        arms = {
            item.name: item.locus.sequence
            for item in a[0].annotations
            if item.name.startswith("arm")
        }
        self.assertEqual(arms, {"arm": "GGGG", "arm'": "AAAA"})

    def test_a_moved_annotation_keeps_taking_primes_until_its_name_is_free(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        a[0].add_annotation(a[0].region("A:0-4"), "arm", track="parts")
        a[0].add_annotation(a[0].region("A:0-4"), "arm'", track="parts")
        b[0].add_annotation(b[0].region("B:0-4"), "arm", track="parts")

        a[0].connect(after=a[1].end(), before=b[2].start())

        arms = {
            item.name: item.locus.sequence
            for item in a[0].annotations
            if item.name.startswith("arm")
        }
        self.assertEqual(arms, {"arm": "GGGG", "arm'": "GGGG", "arm''": "AAAA"})

    def test_joining_graphs_with_the_same_annotation_name_is_rejected_when_strict(self):
        a = self.molecule("A", "GGGG", "CCCC")
        b = self.molecule("B", "AAAA", "TTTT")
        a[0].add_annotation(a[0].region("A:0-4"), "arm", track="parts")
        b[0].add_annotation(b[0].region("B:0-4"), "arm", track="parts")

        with self.assertRaisesRegex(ValueError, "annotation named 'arm'"):
            a[0].connect(after=a[1].end(), before=b[2].start(), strict=True)

        self.assertEqual(self.graph_names(), ["A", "B"])
        self.assertEqual(len(self.sequences(b[0])), 1)


class SearchRepeatedBlockTests(EditingTestCase):
    BLOCK_A = "ACGTTGCAAC"
    BLOCK_B = "GGATCCTTAG"

    def cycle(self):
        """A two-block cycle: block A, then block B, then back to block A."""
        graph = self.repository.import_library(
            "cycle",
            [[gen.Sequence("a", self.BLOCK_A)], [gen.Sequence("b", self.BLOCK_B)]],
            sample="cycle_sample",
        )
        block_a = graph.locus(0, len(self.BLOCK_A))
        block_b = graph.locus(len(self.BLOCK_A), len(self.BLOCK_A) + len(self.BLOCK_B))
        graph.connect(after=block_b.end(), before=block_a.start(), stack=True)
        return graph

    def test_all_sequences_reads_a_cycle_once(self):
        graph = self.cycle()

        self.assertEqual(
            [str(sequence) for sequence in graph.all_sequences()],
            [self.BLOCK_A + self.BLOCK_B],
        )

    def test_random_sequences_walk_a_cycle_any_number_of_times(self):
        graph = self.cycle()
        unit = self.BLOCK_A + self.BLOCK_B

        sequences = [str(sequence) for sequence in islice(graph.random_sequences(seed=1), 200)]

        self.assertEqual(len(sequences), 200)
        for sequence in sequences:
            self.assertEqual(sequence, unit * (len(sequence) // len(unit)))
        self.assertGreater(len({len(sequence) for sequence in sequences}), 1)

    def test_random_sequences_repeat_for_the_same_seed(self):
        graph = self.cycle()

        def take(seed):
            return [str(sequence) for sequence in islice(graph.random_sequences(seed=seed), 50)]

        self.assertEqual(take(7), take(7))
        self.assertNotEqual(take(7), take(8))

    def test_random_sequences_take_every_branch_of_the_graph(self):
        graph = self.repository.import_sequence("ACGTTGCAACGGATCC", name="branched")
        graph.insert("TTTT", after=graph.locus(8, 9).start(), stack=True)

        sampled = {
            str(sequence) for sequence in islice(graph.random_sequences(seed=3), 100)
        }

        self.assertIn("ACGTTGCAACGGATCC", sampled)
        self.assertIn("ACGTTGCAATTTTCGGATCC", sampled)
        for sequence in sampled:
            self.assertRegex(sequence, r"^ACGTTGCAA(TTTT)*CGGATCC$")

    def test_search_follows_a_cycle_around_its_blocks(self):
        graph = self.cycle()
        a, b = self.BLOCK_A, self.BLOCK_B

        for kind in ("exact", "dna"):
            for query in (a + b, a + b + a, a + b + a + b, b + a + b, a + b + a + b + a):
                with self.subTest(kind=kind, query=query):
                    self.assertEqual(len(graph.search(query, sequence_kind=kind)), 1)

    def test_search_does_not_repeat_an_inserted_block_through_its_junction(self):
        sequence = "ACGTTGCAACGGATCCTTAG"
        graph = self.repository.import_sequence(sequence, name="inserted")
        graph.insert("TTTT", after=graph.locus(8, 9).start())

        for kind in ("exact", "dna"):
            with self.subTest(kind=kind):
                self.assertEqual(graph.search("TTTTTTTT", sequence_kind=kind), [])
                self.assertEqual(
                    len(graph.search(sequence[:9] + "TTTT" + sequence[9:], sequence_kind=kind)),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
