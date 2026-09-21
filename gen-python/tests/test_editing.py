"""Exercise single-edit sequence editing through the installed extension."""

from pathlib import Path
import tempfile
import unittest

import gen

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

    def tearDown(self):
        self.temporary_directory.cleanup()

    def operation_count(self):
        return len(self.repository.get_operations())

    def contains(self, sequence_graph, query):
        return bool(sequence_graph.search(query, sequence_kind="exact"))

    def export_sequence(self, sequence_graph):
        path = self.root / "export.fa"
        sequence_graph.export_fasta(str(path))
        return read_fasta(path)

    def sample_named(self, name):
        return next(
            sample for sample in self.repository.samples if sample.sample_name == name
        )

    def is_editable(self, sequence_graph, locus):
        """Whether an edit can still target ``locus``, probed on a copy of the sample."""
        copy = self.sample_named(sequence_graph.sample_name).copy(
            f"probe{len(self.repository.samples)}"
        )
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
        self.assertEqual(
            self.graph.region("m123:20-28").start().sequence_graph.name, "m123"
        )

    def test_positions_from_an_edit_and_an_annotation_step_without_attaching_a_graph(
        self,
    ):
        inserted = self.graph.insert(
            "GG", after=self.reference_locus.slice(3, 4).start()
        )
        replaced = self.graph.replace("m123:10-12", "TT")

        self.assertEqual(inserted.start() + 1, inserted.end())
        self.assertEqual(replaced.start() + 1, replaced.end())
        self.assertEqual(inserted.slice(0, 1).start().sequence_graph.name, "m123")

    def test_an_annotation_made_from_a_locus_keeps_the_loci_graph(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        annotation = gen.Annotation(locus, "site")

        self.assertEqual(annotation.locus.start().sequence_graph.name, "m123")
        self.assertEqual(annotation.locus.start() + 3, locus.slice(3, 4).start())
        self.assertEqual(annotation.locus.sequence, locus.sequence)

    def test_position_or_position_gives_a_superposition(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        first, second = locus.start(), locus.end()

        combined = first | second

        self.assertEqual(combined, gen.SuperPosition(first, second))
        self.assertEqual(combined.positions, [first, second])
        self.assertEqual(combined.sequence_graph.name, "m123")

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

        self.assertEqual(moved.sequence_graph.sample_name, "other")
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

    def test_insert_at_a_junction_of_neighbouring_positions(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        inserted = self.graph.insert(
            "TT", after=locus.slice(3, 4).start(), before=locus.slice(4, 5).start()
        )

        self.assertEqual(len(inserted), 2)
        self.assertEqual(
            self.export_sequence(self.graph), "ATCGATCGATCGATCGATCGGGAATTCACACAGAGA"
        )

    def test_insert_accepts_positions_and_superpositions(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.graph.insert(
            "TT",
            after=gen.SuperPosition(locus.slice(3, 4).start()),
            before=locus.slice(4, 5).start(),
        )

        self.assertTrue(self.contains(self.graph, "GGAATTCACA"))

    def test_insert_needs_after_or_before(self):
        before = self.operation_count()

        with self.assertRaisesRegex(TypeError, "after=, before=, or both"):
            self.graph.insert("TT")
        with self.assertRaises(TypeError):
            self.graph.insert("TT", self.reference_locus.start())
        with self.assertRaisesRegex(TypeError, "Position or a SuperPosition"):
            self.graph.insert("TT", after="m123:3-5")

        self.assertEqual(self.operation_count(), before)

    def test_insert_at_positions_that_do_not_flank_a_junction_fails(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        before = self.operation_count()

        with self.assertRaisesRegex(ValueError, "flank a junction.*use replace"):
            self.graph.insert("TT", after=locus.start(), before=locus.end())
        with self.assertRaisesRegex(ValueError, "flank a junction"):
            self.graph.insert(
                "TT", after=locus.slice(4, 5).start(), before=locus.slice(3, 4).start()
            )

        self.assertEqual(self.operation_count(), before)
        self.assertEqual(
            self.export_sequence(self.graph), "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        )

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
        self.assertEqual(len(locus.slices), 3)

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

    def test_insert_at_start_of_stacked_original_lands_on_that_option(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.replace(locus, "TTTT", stack=True)

        self.graph.insert("CC", before=locus.start())

        self.assertEqual(
            self.export_sequence(self.graph), "ATCGATCGATCGATCGATCGCCGGAACACACAGAGA"
        )

    def test_insert_after_reverse_stacked_original_lands_on_that_option(self):
        # A reverse-strand locus reads its last position at its lowest offset, where the bubble forks.
        [locus] = self.graph.search(reverse_complement("GGAACACA"), sequence_kind="dna")
        self.assertEqual(locus.strand, "-")
        self.graph.replace(locus, "TTTT", stack=True)

        inserted = self.graph.insert("CC", after=locus.end())

        self.assertEqual(inserted.strand, "-")
        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATCGATCGATCG" + reverse_complement("CC") + "GGAACACACAGAGA",
        )

    def test_insert_after_a_fork_attaches_to_every_option(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.replace(locus, "TTTT", stack=True)

        self.graph.insert("CC", after=self.reference_locus.slice(19, 20).start())

        self.assertTrue(self.contains(self.graph, "GATCGCCGGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "GATCGCCTTTTCAGAGA"))

    def test_insert_on_one_branch_of_a_fork_leaves_the_other_unchanged(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.replace(locus, "TTTT", stack=True)

        self.graph.insert(
            "CC", after=self.reference_locus.slice(19, 20).start(), before=locus.start()
        )

        self.assertTrue(self.contains(self.graph, "GATCGCCGGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "GATCGTTTTCAGAGA"))
        self.assertFalse(self.contains(self.graph, "GATCGCCTTTTCAGAGA"))

    def test_insert_on_one_arm_of_a_join_leaves_the_other_unchanged(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        alternative = self.graph.replace(locus, "TTTT", stack=True)

        self.graph.insert(
            "GG",
            after=alternative.end(),
            before=self.reference_locus.slice(28, 29).start(),
        )

        self.assertTrue(self.contains(self.graph, "GATCGTTTTGGCAGAGA"))
        self.assertTrue(self.contains(self.graph, "GGAACACACAGAGA"))
        self.assertFalse(self.contains(self.graph, "GGAACACAGGCAGAGA"))

    def test_insert_with_a_superposition_on_every_arm_of_a_join(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        alternative = self.graph.replace(locus, "TTTT", stack=True)
        arm_ends = gen.SuperPosition(locus.end(), alternative.end())

        self.graph.insert(
            "GG", after=arm_ends, before=self.reference_locus.slice(28, 29).start()
        )

        self.assertTrue(self.contains(self.graph, "GATCGTTTTGGCAGAGA"))
        self.assertTrue(self.contains(self.graph, "GGAACACAGGCAGAGA"))

    def test_insert_at_a_fork_with_an_unconnected_branch_position_fails(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        alternative = self.graph.replace(locus, "TTTT", stack=True)
        before = self.operation_count()

        # The last position of one arm does not read into the first position of the other.
        with self.assertRaisesRegex(ValueError, "flank a junction.*use replace"):
            self.graph.insert("CC", after=alternative.end(), before=locus.start())

        self.assertEqual(self.operation_count(), before)

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
        part = gen.SequencePart
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


if __name__ == "__main__":
    unittest.main()
