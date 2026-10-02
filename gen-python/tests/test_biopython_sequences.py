"""Exercise import_sequences with Biopython inputs; skipped when Biopython isn't installed."""

from pathlib import Path
import tempfile
import unittest

import gen

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except ImportError:
    SeqIO = None


@unittest.skipIf(SeqIO is None, "Biopython is not installed")
class BiopythonSequencesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.repository = gen.Repository(str(self.directory / ".gen"))

    def graph_names(self, sample):
        return [graph.name for graph in sample]

    def sample_named(self, name):
        return next(s for s in self.repository.get_samples() if s.sample_name == name)

    def test_seq_value_needs_a_name(self):
        graph = self.repository.import_sequence(Seq("ACGT"), "a", sample="seq")
        self.assertEqual(graph.name, "a")
        with self.assertRaises(ValueError):
            self.repository.import_sequence(Seq("ACGT"))

    def test_seq_record_supplies_its_name_unless_overridden(self):
        record = SeqRecord(Seq("ACGT"), id="r1")
        graph = self.repository.import_sequence(record, sample="record")
        self.assertEqual(graph.name, "r1")
        graph = self.repository.import_sequence(record, "renamed", sample="override")
        self.assertEqual(graph.name, "renamed")

    def test_seqio_loop_matches_fasta_import(self):
        fasta = self.directory / "input.fa"
        fasta.write_text(">chr1\nACGTACGT\n>chr2\nTTTTGGGG\n")
        from_file = self.repository.import_fasta(str(fasta), sample="file")
        with fasta.open() as handle:
            for record in SeqIO.parse(handle, "fasta"):
                self.repository.import_sequence(record, sample="records")
        self.assertEqual(
            sorted(self.graph_names(self.sample_named("records"))),
            sorted(self.graph_names(from_file)),
        )

    def test_non_sequence_value_is_rejected(self):
        with self.assertRaises(ValueError):
            self.repository.import_sequence(42, "a")


if __name__ == "__main__":
    unittest.main()
