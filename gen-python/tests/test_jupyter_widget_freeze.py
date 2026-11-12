"""Tests for static HTML emitted when interactive graph widgets are frozen."""

import importlib.util
import unittest
from unittest.mock import Mock


_HAS_JUPYTER_WIDGET_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("anywidget", "ipywidgets", "traitlets")
)


@unittest.skipUnless(_HAS_JUPYTER_WIDGET_DEPS, "requires the optional jupyter extra")
class FrozenJupyterWidgetTests(unittest.TestCase):
    def test_frozen_plot_html_scales_down_to_fit_notebook_cell(self):
        from gen.jupyter_widget import GraphWidget

        class FakeController:
            annotations_loaded = True
            page_count = 1
            page_index = 0

            def render_frame(self, cols, rows):
                return '{"cols":60,"rows":12,"cells":[]}'

        widget = GraphWidget(FakeController())
        display_handle = Mock()
        widget._display_handle = display_handle

        widget._on_frontend_msg(
            widget,
            {
                "type": "freeze",
                "data": "data:image/png;base64,ZmFrZQ==",
                "width": 640,
                "height": 240,
            },
            [],
        )

        rendered = display_handle.update.call_args.args[0]
        html = rendered["text/html"]
        self.assertIn("width:640px", html)
        self.assertIn("max-width:100%", html)
        self.assertIn("height:auto", html)
        self.assertNotIn("height:240px", html)
