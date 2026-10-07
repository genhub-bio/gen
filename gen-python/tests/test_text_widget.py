"""Test the plain-text graph viewer when Jupyter extras are unavailable."""

import json
import tempfile
import unittest

import gen
from gen import text_widget


class FakeController:
    """Supply the minimal native-controller API required by TextGraphWidget."""

    annotations_loaded = True
    page_count = 1
    page_index = 0

    def render_frame(self, cols, rows):
        return json.dumps(
            {
                "cols": cols,
                "rows": rows,
                "cells": [{"x": 0, "y": 0, "text": "A"}],
            }
        )


class TextGraphWidgetTests(unittest.TestCase):
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
}
