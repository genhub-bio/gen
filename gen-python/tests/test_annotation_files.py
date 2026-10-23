"""Annotation files recorded with import_annotations read like database annotations."""

import json
import gzip
import unittest

from test_api import FIXTURES, RepositoryTestCase

try:
    from gen.jupyter_widget import GraphWidget
except ImportError:
    GraphWidget = None

LABELLED_FEATURE = "foobar"
WHOLE_GRAPH_FEATURE = "m123_region"


def frame_text(widget):
    frame = json.loads(widget._controller.render_frame(200, 50))
    return "".join(cell["text"] for cell in frame["cells"])


class GffAnnotationTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        self.repository.import_annotations(
            str(FIXTURES / "simple.gff"), name="genes", sample=self.sample
        )

    def plot_widget(self):
        return GraphWidget(self.graph.plot()._controller)

    def test_annotations_list_imported_gff_features_by_name(self):
        by_name = {annotation.name: annotation for annotation in self.graph.annotations}

        self.assertIn(LABELLED_FEATURE, by_name)
        self.assertEqual(by_name[LABELLED_FEATURE].track, "genes")

    def test_imported_gff_annotations_can_be_used_as_database_annotations(self):
        [annotation] = [
            annotation
            for annotation in self.graph.annotations
            if annotation.name == LABELLED_FEATURE
        ]

        self.assertTrue(len(annotation.locus.sequence) > 0)

    def test_annotations_keep_database_ones_beside_imported_gff_features(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.add_annotation(locus, "motif", track="motifs")

        names = {annotation.name for annotation in self.graph.annotations}

        self.assertLessEqual({"motif", LABELLED_FEATURE}, names)

    def test_imported_gff_annotations_inherit_to_descendant_samples(self):
        child = self.sample.copy("edited")

        features = {annotation.name: annotation for annotation in child[0].annotations}
        self.assertIn(LABELLED_FEATURE, features)
        self.assertEqual(features[LABELLED_FEATURE].track, "genes")

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_widget_lists_imported_gff_tracks_automatically(self):
        widget = self.plot_widget()

        self.assertIn("genes", widget.tracks)
        self.assertNotIn(WHOLE_GRAPH_FEATURE, frame_text(widget))

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_hide_and_show_work_on_imported_gff_tracks(self):
        widget = self.plot_widget()

        widget.hide_track("genes")
        frame_text(widget)

        widget.show_track("genes")
        frame_text(widget)
        self.assertIn("genes", widget.tracks)

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_hide_all_tracks_keeps_gff_tracks_hidden_across_renders(self):
        widget = self.plot_widget()

        widget.hide_all_tracks()

        self.assertNotIn(WHOLE_GRAPH_FEATURE, frame_text(widget))
        self.assertNotIn(WHOLE_GRAPH_FEATURE, frame_text(widget))

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_show_track_rejects_unknown_names(self):
        widget = self.plot_widget()

        with self.assertRaises(RuntimeError):
            widget.show_track("no-such-track")


class CompressedGffAnnotationTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        compressed = self.root / "genes.gff.gz"
        compressed.write_bytes(gzip.compress((FIXTURES / "simple.gff").read_bytes()))
        self.repository.import_annotations(
            str(compressed), name="genes", sample=self.sample
        )

    def test_annotations_read_the_whole_compressed_gff(self):
        names = {annotation.name for annotation in self.graph.annotations}

        self.assertIn(LABELLED_FEATURE, names)

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_widget_loads_compressed_gff_annotations(self):
        widget = GraphWidget(self.graph.plot()._controller)

        self.assertIn("genes", widget.tracks)
        self.assertNotIn(WHOLE_GRAPH_FEATURE, frame_text(widget))

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_hidden_compressed_gff_track_stays_hidden(self):
        widget = GraphWidget(self.graph.plot()._controller)

        widget.hide_track("genes")

        self.assertNotIn(WHOLE_GRAPH_FEATURE, frame_text(widget))
