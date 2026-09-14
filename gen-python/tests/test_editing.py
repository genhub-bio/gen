"""Exercise single-edit sequence editing through the installed extension."""

from pathlib import Path
import tempfile
import unittest

import gen

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def reverse_complement(sequence):
    return sequence[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def node_range(node_slice):
    """Node-absolute bases of a slice; ``str(Node)`` is ``hash:start-end``."""
    block_start = int(str(node_slice.node).rsplit(":", 1)[1].split("-")[0])
    return (block_start + node_slice.start, block_start + node_slice.end)


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
            sample
            for sample in self.repository.get_samples()
            if sample.sample_name == name
        )


class SimpleGraphEditingTests(EditingTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        [self.reference_locus] = self.graph.search(
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA", sequence_kind="exact"
        )

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

    def test_insert_at_position_and_region_point(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        inserted = self.graph.insert(locus.start(), "NNNN")
        self.assertEqual(len(inserted), 4)
        self.assertTrue(self.contains(self.graph, "GATCGNNNNGGAACACA"))

        self.graph.insert("m123:2-2", "YYY")
        self.assertTrue(self.contains(self.graph, "ATYYYCGATCG"))

    def test_insert_at_zero_length_locus_slice(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.graph.insert(locus.slice(4, 4), "KK")

        self.assertTrue(self.contains(self.graph, "GGAAKKCACA"))

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
        self.graph.insert("m123:10-10", "GG")
        self.assertEqual(self.operation_count(), before + 3)

    def test_message_overrides_generated_operation_summary(self):
        self.graph.replace("m123:3-5", "TT", message="custom replace message")
        self.graph.delete("m123:20-28", message="custom delete message")
        self.graph.insert("m123:10-10", "GG", message="custom insert message")

        messages = [operation.message for operation in self.repository.get_operations()]
        self.assertIn("custom replace message", messages)
        self.assertIn("custom delete message", messages)
        self.assertIn("custom insert message", messages)

    def test_invalid_requests_fail_without_recording(self):
        before = self.operation_count()

        with self.assertRaisesRegex(ValueError, "zero-length target"):
            self.graph.insert("m123:3-5", "TT")
        with self.assertRaisesRegex(ValueError, "at least one base"):
            self.graph.delete("m123:3-3")
        with self.assertRaisesRegex(ValueError, "non-empty sequence"):
            self.graph.replace("m123:3-5", "")
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
        inserted = self.graph.insert("m123:10-10", "GGGG")
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
        self.graph.insert("m123:10-10", "GG")

        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATGGCGATCGATCGGGAACACACAGAGA",
        )

    def test_export_fasta_reflects_replace_on_path(self):
        self.graph.replace("m123:3-5", "TT")

        self.assertEqual(
            self.export_sequence(self.graph), "ATCTTTCGATCGATCGATCGGGAACACACAGAGA"
        )
        self.assertTrue(self.contains(self.graph, "ATCGATCGA"))

    def test_export_fasta_reflects_delete_on_path(self):
        self.graph.delete("m123:20-28")

        self.assertEqual(self.export_sequence(self.graph), "ATCGATCGATCGATCGATCGCAGAGA")

    def test_export_fasta_reflects_chained_edits_in_order(self):
        # Matches test_repeated_child_edits_accumulate's edits, but checked through
        # export_fasta (a materialized Path) rather than search (the live graph),
        # since a Path is a snapshot that earlier edits had never refreshed.
        self.graph.replace("m123:3-5", "TT")
        self.graph.delete("m123:20-28")

        self.assertEqual(self.export_sequence(self.graph), "ATCTTTCGATCGATCGATCGCAGAGA")

    def test_copy_rejects_existing_sample(self):
        self.sample.copy("child")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.sample.copy("child")

    def test_stack_replace_keeps_original_reachable_and_path_untouched(self):
        self.graph.replace("m123:3-5", "TT", stack=True)

        self.assertTrue(self.contains(self.graph, "ATCGATCGATCG"))
        self.assertTrue(self.contains(self.graph, "ATCTTTCGATCG"))
        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
        )

    def test_stack_insert_keeps_original_reachable_and_path_untouched(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")

        self.graph.insert(locus.start(), "GG", stack=True)

        self.assertTrue(self.contains(self.graph, "ATCGATCGGGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "ATCGATCGGGGGAACACACAGAGA"))
        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
        )

    def test_stack_delete_keeps_original_reachable_and_path_untouched(self):
        self.graph.delete("m123:20-28", stack=True)

        self.assertTrue(self.contains(self.graph, "GGAACACACAGAGA"))
        self.assertTrue(self.contains(self.graph, "ATCGATCGCAGAGA"))
        self.assertEqual(
            self.export_sequence(self.graph),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
        )


if __name__ == "__main__":
    unittest.main()
