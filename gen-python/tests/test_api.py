"""Repository properties and widget display controls without sequence editing."""

import gzip
import tempfile
import unittest
from pathlib import Path

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

    def node_runs(self, locus):
        """The `(node, length)` stretches of a locus in reading order."""
        runs = []
        for index in range(len(locus)):
            node = locus[index].node
            if runs and runs[-1][0] == node:
                runs[-1][1] += 1
            else:
                runs.append([node, 1])
        return runs


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

    def test_import_annotations_detects_format_and_rebuilds_csi_index(self):
        self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        annotation = self.root / "genes.gff"
        fixture_lines = (FIXTURES / "simple.gff").read_text().splitlines()
        fixture_records = [line for line in fixture_lines if not line.startswith("#")]
        expected_positions = sorted(int(line.split("\t")[3]) for line in fixture_records)
        fixture_headers = [line for line in fixture_lines if line.startswith("#")]
        annotation.write_text(
            "\n".join([*fixture_headers, *reversed(fixture_records)]) + "\n"
        )
        stale_index = self.root / "genes.gff.tbi"
        stale_index.write_bytes(b"stale index for the original source")
        commit_hash = self.repository.import_annotations(
            str(annotation), name="genes", message="Add genes"
        )
        self.assertEqual(self.repository.get_operations()[0].id, commit_hash)
        self.assertEqual(self.repository.get_operations()[0].message, "Add genes")
        assets = self.repository.get_assets()
        asset_names = {asset.name for asset in assets}
        self.assertIn("genes", asset_names)
        self.assertIn("genes.gff.csi", asset_names)
        self.assertNotIn(stale_index.name, asset_names)

        coordinate_index = next(
            asset for asset in assets if asset.name == "genes.gff.csi"
        )
        saved_index = coordinate_index.save_as(self.root / "generated-index.csi")
        self.assertTrue(
            gzip.decompress(saved_index.read_bytes()).startswith(b"CSI\x01"),
            "local annotation imports should retain a real CSI index",
        )

        annotation_asset = next(asset for asset in assets if asset.name == "genes")
        saved_annotation = annotation_asset.save_as(self.root / "retained.gff")
        indexed_positions = [
            int(line.split("\t")[3])
            for line in saved_annotation.read_text().splitlines()
            if not line.startswith("#")
        ]
        self.assertEqual(indexed_positions, expected_positions)

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
            [item.name for item in self.repository.samples], [sample.name]
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
