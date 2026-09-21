"""Test the plain-text graph viewer when Jupyter extras are unavailable."""

import json
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

        self.assertIn("# Gen graph textual output", first_frame)
        self.assertIn("a", first_frame)
        self.assertNotIn("# Gen graph textual output", second_frame)
        self.assertEqual(second_frame.splitlines()[0], "a")

    def test_non_jupyter_sessions_use_text_graph_widget(self):
        self.assertIs(gen.GraphWidget, text_widget.TextGraphWidget)
