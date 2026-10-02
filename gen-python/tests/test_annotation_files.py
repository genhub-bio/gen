"""Annotation files recorded with import_annotations read like database annotations."""

import json
import shutil
import subprocess
import unittest

from test_api import FIXTURES, RepositoryTestCase

try:
    from gen.jupyter_widget import GraphWidget
except ImportError:
    GraphWidget = None

LABELLED_FEATURE = "gene-a0001"
# Fills its whole node, so the widget labels it at the default detail level.
DRAWN_FEATURE = "m123_region"


def frame_text(widget):
    frame = json.loads(widget._controller.render_frame(200, 50))
    return "".join(cell["text"] for cell in frame["cells"])


class AnnotationFileTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        self.repository.import_annotations(str(FIXTURES / "simple.gff"), name="genes")

    def plot_widget(self):
        return GraphWidget(self.graph.plot()._controller)

    def test_annotations_list_file_features_by_name(self):
        by_name = {annotation.name: annotation for annotation in self.graph.annotations}

        self.assertIn(LABELLED_FEATURE, by_name)
        self.assertEqual(by_name[LABELLED_FEATURE].track, "genes")

    def test_file_annotations_can_be_edited_like_database_ones(self):
        [annotation] = [
            annotation
            for annotation in self.graph.annotations
            if annotation.name == LABELLED_FEATURE
        ]

        self.assertTrue(len(annotation.locus.sequence) > 0)

    def test_annotations_keep_database_ones_beside_file_features(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.add_annotation(locus, "motif", track="motifs")

        names = {annotation.name for annotation in self.graph.annotations}

        self.assertLessEqual({"motif", LABELLED_FEATURE}, names)

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_widget_lists_and_draws_file_tracks_without_being_asked(self):
        widget = self.plot_widget()

        self.assertIn("genes", widget.tracks)
        self.assertIn(DRAWN_FEATURE, frame_text(widget))

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_hide_and_show_track_work_on_file_tracks(self):
        widget = self.plot_widget()

        widget.hide_track("genes")
        self.assertNotIn(DRAWN_FEATURE, frame_text(widget))

        widget.show_track("genes")
        self.assertIn(DRAWN_FEATURE, frame_text(widget))

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_hide_all_tracks_keeps_file_tracks_hidden_across_renders(self):
        widget = self.plot_widget()

        widget.hide_all_tracks()

        self.assertNotIn(DRAWN_FEATURE, frame_text(widget))
        self.assertNotIn(DRAWN_FEATURE, frame_text(widget))

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_show_track_rejects_unknown_names(self):
        widget = self.plot_widget()

        with self.assertRaises(RuntimeError):
            widget.show_track("no-such-track")


@unittest.skipIf(
    shutil.which("bgzip") is None or shutil.which("tabix") is None,
    "requires bgzip and tabix",
)
class IndexedAnnotationFileTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]
        compressed = self.root / "genes.gff.gz"
        compressed.write_bytes(
            subprocess.run(
                ["bgzip", "-c", str(FIXTURES / "simple.gff")],
                check=True,
                capture_output=True,
            ).stdout
        )
        subprocess.run(["tabix", "-p", "gff", str(compressed)], check=True)
        self.repository.import_annotations(str(compressed), name="genes")

    def test_annotations_read_the_whole_indexed_file(self):
        names = {annotation.name for annotation in self.graph.annotations}

        self.assertIn(LABELLED_FEATURE, names)

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_widget_loads_indexed_file_for_the_viewport(self):
        widget = GraphWidget(self.graph.plot()._controller)

        self.assertIn(DRAWN_FEATURE, frame_text(widget))

    @unittest.skipIf(GraphWidget is None, "requires gen[jupyter]")
    def test_hidden_indexed_track_is_not_reloaded_by_rendering(self):
        widget = GraphWidget(self.graph.plot()._controller)

        widget.hide_track("genes")

        self.assertNotIn(DRAWN_FEATURE, frame_text(widget))
