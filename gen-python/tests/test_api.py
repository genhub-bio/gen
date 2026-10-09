"""Repository properties and widget display controls without sequence editing."""

from pathlib import Path
import tempfile
import unittest

import gen

try:
    from gen.jupyter_widget import GraphWidget
except ImportError:
    GraphWidget = None

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class RepositoryTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = gen.Repository(str(self.root / "repository"))

    def contains(self, sequence_graph, query):
        return bool(sequence_graph.search(query, sequence_kind="exact"))


class GeneralApiTests(RepositoryTestCase):
    def test_repository_records_the_given_committer(self):
        repository = gen.Repository(
            str(self.root / "identified"),
            committer="Ada Lovelace",
            email="ada@example.com",
        )
        repository.import_fasta(str(FIXTURES / "simple.fa"))
        operation = repository.get_operations()[0]
        self.assertEqual(operation.committer, "Ada Lovelace")
        self.assertEqual(operation.email, "ada@example.com")

    def test_repository_committer_defaults_without_arguments(self):
        self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        operation = self.repository.get_operations()[0]
        self.assertEqual(operation.committer, "gen")
        self.assertEqual(operation.email, "gen@genhub.bio")

    def test_import_annotations_detects_format_and_index(self):
        self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        annotation = self.root / "genes.gff"
        annotation.write_bytes((FIXTURES / "simple.gff").read_bytes())
        index = self.root / "genes.gff.tbi"
        index.write_bytes(b"index fixture")
        commit_hash = self.repository.import_annotations(
            str(annotation), name="genes", message="Add genes"
        )
        self.assertEqual(self.repository.get_operations()[0].id, commit_hash)
        self.assertEqual(self.repository.get_operations()[0].message, "Add genes")
        asset_names = {asset.name for asset in self.repository.get_assets()}
        self.assertIn("genes", asset_names)
        self.assertIn(index.name, asset_names)

    def test_import_annotations_accepts_explicit_format(self):
        annotation = self.root / "genes"
        annotation.write_bytes((FIXTURES / "simple.gff").read_bytes())
        commit_hash = self.repository.import_annotations(str(annotation), format="gff3")
        self.assertEqual(self.repository.get_operations()[0].id, commit_hash)

    def test_repository_rejects_empty_committer(self):
        with self.assertRaises(ValueError):
            gen.Repository(str(self.root / "empty-name"), committer="")
        with self.assertRaises(ValueError):
            gen.Repository(str(self.root / "empty-email"), email="")

    def test_sample_properties_keep_indexing_and_iteration(self):
        sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.assertEqual(
            [item.sample_name for item in self.repository.samples], [sample.sample_name]
        )
        self.assertEqual(
            [graph.name for graph in sample.sequence_graphs],
            [graph.name for graph in sample],
        )
        self.assertEqual(sample[-1].name, sample.sequence_graphs[-1].name)

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_annotation_tracks_and_temporary_highlight_controls(self):
        graph = self.repository.import_genbank(str(FIXTURES / "puc19.gb"))[0]
        annotation = graph.annotations[0]
        widget = GraphWidget(graph.plot()._controller)
        names = widget.tracks
        self.assertTrue(names)
        widget.hide_all_tracks()
        self.assertEqual(widget.tracks, names)
        widget.show_track(names[0])
        widget.show(annotation)
        widget.show_path()
        widget.clear_highlights()
        widget.hide_path()
        widget.hide_track(names[0])
        self.assertEqual(widget.tracks, names)
