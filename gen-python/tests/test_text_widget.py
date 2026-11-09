"""Test the plain-text graph viewer when Jupyter extras are unavailable."""

import json
import os
import pathlib
import struct
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import gen
from gen import text_widget
from gen.png_image import PngImage


class FakeController:
    """Supply the minimal native-controller API required by TextGraphWidget."""

    annotations_loaded = True
    page_count = 1
    page_index = 0

    def render_png(self, cols, rows, scale=1):
        return b"\x89PNG" + f"{cols}x{rows}{self._scale_suffix(scale)}".encode()

    def render_png_viewport(self, cols, rows, scale=1):
        return (
            b"\x89PNG viewport " + f"{cols}x{rows}{self._scale_suffix(scale)}".encode()
        )

    @staticmethod
    def _scale_suffix(scale):
        return "" if scale == 1 else f"@{scale}"

    def png_cell_size(self):
        return (12, 22)

    def png_background_color(self):
        return "#1e1e2e"

    def render_hit_regions(self, cols, rows):
        return json.dumps([{"x": 1, "y": 1, "kind": "node"}])

    def render_frame(self, cols, rows):
        return json.dumps(
            {
                "cols": cols,
                "rows": rows,
                "cells": [{"x": 0, "y": 0, "text": "A"}],
            }
        )


class TextGraphWidgetTests(unittest.TestCase):
    def test_color_mapping_reads_sequence_graph_annotations_for_each_page(self):
        from gen import jupyter_widget

        class ColorController(FakeController):
            annotations_loaded = False
            page_count = 2

            def __init__(self):
                self.page_index = 0
                self.color_maps = []

            def load_annotation_groups_with_colors(self, color_map):
                self.color_maps.append(color_map)
                self.annotations_loaded = True

            def next_page(self):
                self.page_index = 1
                self.annotations_loaded = False

        graphs = [
            SimpleNamespace(annotations=[SimpleNamespace(id="first", name="one")]),
            SimpleNamespace(annotations=[SimpleNamespace(id="second", name="two")]),
        ]
        for widget_type in (text_widget.TextGraphWidget, jupyter_widget.GraphWidget):
            with self.subTest(widget_type=widget_type.__name__):
                controller = ColorController()
                widget = widget_type(
                    controller,
                    colors={"one": "#ff0000", "two": "#0000ff"},
                    sequence_graphs=graphs,
                )
                try:
                    widget.next_page()
                    self.assertEqual(
                        controller.color_maps,
                        [{"first": "#ff0000"}, {"second": "#0000ff"}],
                    )
                finally:
                    if isinstance(widget, jupyter_widget.GraphWidget):
                        widget.close()

    def test_save_png_writes_controller_bytes_and_returns_none(self):
        widget = text_widget.TextGraphWidget(FakeController(), cols=7, rows=3, scale=1)
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "plot.png"
            self.assertIsNone(widget.save_png(target))
            self.assertEqual(target.read_bytes(), b"\x89PNG7x3")
            widget.save_png(target, rows=4)
            self.assertEqual(target.read_bytes(), b"\x89PNG7x4")
            widget.save_png(target, cols=9)
            self.assertEqual(target.read_bytes(), b"\x89PNG9x3")
            widget.save_png(target, cols=9, rows=4)
            self.assertEqual(target.read_bytes(), b"\x89PNG9x4")

    def test_jupyter_save_png_uses_widget_size_and_independent_overrides(self):
        from gen import jupyter_widget

        widget = jupyter_widget.GraphWidget(FakeController(), cols=7, rows=3, scale=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                target = pathlib.Path(directory) / "plot.png"
                self.assertIsNone(widget.save_png(target))
                self.assertEqual(target.read_bytes(), b"\x89PNG7x3")
                widget.save_png(target, rows=4)
                self.assertEqual(target.read_bytes(), b"\x89PNG7x4")
                widget.save_png(target, cols=9)
                self.assertEqual(target.read_bytes(), b"\x89PNG9x3")
        finally:
            widget.close()

    def test_save_png_scale_defaults_to_the_widget_scale_and_can_be_overridden(self):
        from gen import jupyter_widget

        for widget_type in (text_widget.TextGraphWidget, jupyter_widget.GraphWidget):
            with self.subTest(widget_type=widget_type.__name__):
                widget = widget_type(FakeController(), cols=7, rows=3, scale=2)
                try:
                    with tempfile.TemporaryDirectory() as directory:
                        target = pathlib.Path(directory) / "plot.png"
                        widget.save_png(target)
                        self.assertEqual(target.read_bytes(), b"\x89PNG7x3@2")
                        widget.save_png(target, scale=3)
                        self.assertEqual(target.read_bytes(), b"\x89PNG7x3@3")
                        widget.save_png(target, scale=1)
                        self.assertEqual(target.read_bytes(), b"\x89PNG7x3")
                finally:
                    if isinstance(widget, jupyter_widget.GraphWidget):
                        widget.close()

    def test_widgets_default_to_a_scale_of_two(self):
        from gen import jupyter_widget

        text = text_widget.TextGraphWidget(FakeController(), cols=7, rows=3)
        self.assertEqual(text.scale, 2)
        widget = jupyter_widget.GraphWidget(FakeController(), cols=7, rows=3)
        try:
            self.assertEqual(widget.scale, 2)
            self.assertEqual(widget.viewport_png, b"\x89PNG viewport 7x3@2")
            self.assertEqual((widget.cell_width, widget.cell_height), (24, 44))
        finally:
            widget.close()

    def test_jupyter_widget_scale_syncs_physical_cell_size_and_rerenders(self):
        from gen import jupyter_widget

        widget = jupyter_widget.GraphWidget(FakeController(), cols=7, rows=3, scale=2)
        try:
            self.assertEqual(widget.scale, 2)
            self.assertEqual(widget.viewport_png, b"\x89PNG viewport 7x3@2")
            self.assertEqual((widget.cell_width, widget.cell_height), (24, 44))
            widget.scale = 1
            self.assertEqual(widget.viewport_png, b"\x89PNG viewport 7x3")
            self.assertEqual((widget.cell_width, widget.cell_height), (12, 22))
        finally:
            widget.close()

    def test_png_image_reports_the_logical_size_of_a_scaled_png(self):
        header = (
            b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 88, 44)
        )
        self.assertEqual(PngImage(header)._repr_png_(), header)
        self.assertEqual(
            PngImage(header, scale=2)._repr_png_(),
            (header, {"width": 44, "height": 22}),
        )

    def test_static_plot_scale_doubles_the_pixels_and_validates_the_range(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = gen.Repository(directory)
            graph = repository.import_sequence("ACGT" * 40, name="chain", sample="s")

            def size(image):
                return struct.unpack(">II", image.data[16:24])

            single = graph.plot(interactive=False, rows=8, cols=40, scale=1)
            double = graph.plot(interactive=False, rows=8, cols=40, scale=2)
            self.assertEqual(graph.plot(interactive=False, rows=8, cols=40).scale, 2)
            self.assertEqual(double.scale, 2)
            self.assertEqual(size(double), tuple(2 * value for value in size(single)))
            image, metadata = double._repr_png_()
            self.assertEqual(image, double.data)
            self.assertEqual((metadata["width"], metadata["height"]), size(single))
            for scale in (0, 9):
                with self.assertRaises(ValueError):
                    graph.plot(interactive=False, scale=scale)

    def test_jupyter_widget_syncs_native_png_as_bytes_with_rust_cell_geometry(self):
        from gen import jupyter_widget

        widget = jupyter_widget.GraphWidget(FakeController(), cols=7, rows=3, scale=1)
        try:
            self.assertEqual(widget.viewport_png, b"\x89PNG viewport 7x3")
            self.assertIsInstance(widget.viewport_png, bytes)
            self.assertEqual((widget.cell_width, widget.cell_height), (12, 22))
            self.assertEqual(widget.background_color, "#1e1e2e")
            self.assertEqual(widget.hit_regions, [{"x": 1, "y": 1, "kind": "node"}])
            self.assertEqual((widget.frame["cols"], widget.frame["rows"]), (7, 3))
        finally:
            widget.close()

    def test_static_plot_returns_displayable_png(self):
        from IPython.core.formatters import PNGFormatter

        with tempfile.TemporaryDirectory() as directory:
            repository = gen.Repository(directory)
            graph = repository.import_sequence("ACGT" * 40, name="chain", sample="s")
            image = graph.plot(interactive=False, rows=8, cols=40, scale=1)
            self.assertIsInstance(image, PngImage)
            self.assertTrue(image.data.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(image._repr_png_(), image.data)
            self.assertEqual(PNGFormatter()(image), image.data)
            sample_image = repository.samples[0].plot(interactive=False)
            self.assertIsInstance(sample_image, PngImage)
            self.assertTrue(sample_image.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_plot_go_to_matches_widget_navigation_for_all_target_types(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = gen.Repository(directory)
            graph = repository.import_sequence("ACGT" * 400, name="chain", sample="s")
            locus = graph.region("chain:900-920")
            annotation = gen.Annotation(locus, "target")
            targets = [locus, locus.start(), locus.start() | locus.end(), annotation]
            for plot_source in [graph, repository.samples[0]]:
                for target in targets:
                    with self.subTest(source=type(plot_source), target=type(target)):
                        expected = plot_source.plot(rows=12, cols=60).go_to(target)
                        widget = plot_source.plot(rows=12, cols=60, go_to=target)
                        self.assertEqual(widget.frame, expected.frame)
                        image = plot_source.plot(
                            rows=12, cols=60, interactive=False, go_to=target
                        )
                        self.assertEqual(image.data, expected._render_png().data)

    def setUp(self):
        self.original_hint_shown = text_widget._text_fallback_hint_shown
        text_widget._text_fallback_hint_shown = False

    def tearDown(self):
        text_widget._text_fallback_hint_shown = self.original_hint_shown

    def test_text_fallback_hint_is_shown_once_per_process(self):
        widget = text_widget.TextGraphWidget(FakeController())

        first_frame = repr(widget)
        second_frame = repr(widget)

        self.assertIn("Gen SequenceGraph textual output", first_frame)
        self.assertIn("a", first_frame)
        self.assertNotIn("Gen SequenceGraph textual output", second_frame)
        self.assertEqual(second_frame.splitlines()[0], "a")

    def test_frozen_notebooks_get_no_hint_in_their_text_frames(self):
        widget = text_widget.TextGraphWidget(FakeController())

        with mock.patch.dict(os.environ, {"GEN_NOTEBOOK_TEXT_OUTPUT": "1"}):
            frame = repr(widget)

        self.assertEqual(frame.splitlines()[0], "a")

    def test_non_jupyter_sessions_use_text_graph_widget(self):
        self.assertIs(gen.GraphWidget, text_widget.TextGraphWidget)

    def test_both_widgets_expose_the_same_graph_api(self):
        from gen import jupyter_widget

        for name in _GRAPH_WIDGET_API:
            with self.subTest(name=name):
                self.assertTrue(hasattr(jupyter_widget.GraphWidget, name))
                self.assertTrue(hasattr(text_widget.TextGraphWidget, name))

    def test_both_widgets_expose_the_same_state_attributes(self):
        from gen import jupyter_widget

        widget = text_widget.TextGraphWidget(FakeController())
        for name in ("cols", "rows", "page_count", "page_index", "frame"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(widget, name))
                self.assertTrue(hasattr(jupyter_widget.GraphWidget, name))

    def test_widgets_are_not_part_of_the_public_namespace(self):
        self.assertNotIn("GraphWidget", dir(gen))
        self.assertNotIn("TextGraphWidget", dir(gen))

    def test_navigation_methods_return_the_widget_for_chaining(self):
        repository = gen.Repository(tempfile.mkdtemp())
        graph = repository.import_sequence("ACGT" * 40, name="chain", sample="s")
        widget = graph.plot(rows=8, cols=40)
        locus = graph.region("chain:10-20")
        chained = (
            widget.zoom_in()
            .zoom_out()
            .show_labels()
            .hide_labels()
            .scroll_right()
            .scroll_left()
            .scroll_down()
            .scroll_up()
            .next_page()
            .prev_page()
            .go_to(locus.start())
            .show(locus)
            .clear_highlights()
            .refresh()
        )
        self.assertIs(chained, widget)
        self.assertIsInstance(repr(widget), str)


_GRAPH_WIDGET_API = {
    "handle_click",
    "zoom_in",
    "zoom_out",
    "show_labels",
    "hide_labels",
    "scroll_left",
    "scroll_right",
    "scroll_up",
    "scroll_down",
    "next_page",
    "prev_page",
    "go_to",
    "show",
    "refresh",
    "clear_highlights",
    "show_path",
    "hide_path",
    "show_track",
    "hide_track",
    "tracks",
    "hide_all_tracks",
    "save_png",
}
