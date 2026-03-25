"""Jupyter widget for interactive gen graph visualization.

Uses anywidget to embed the gen-tui WASM renderer in a notebook cell.
The WASM bundle (gen-python/python/gen/static/graph_widget.js) is built separately
via `python gen-python/build_bundle.py` after running wasm-pack.

Typical usage — install gen-python, then in a notebook::

    repo = gen.Repository()
    bg = repo.get_block_groups()[0]
    bg  # renders the interactive graph widget automatically

The _ipython_display_ hook on PyBlockGroup is attached at the bottom of
gen/__init__.py when both gen_graph_widget and IPython are available.
"""

import json
import os
import pathlib

import anywidget
import traitlets

# Catppuccin Mocha (dark) and Latte (light) — hex strings without '#'.
# Keys match the PaletteHex struct in jupyter-widget/src/lib.rs.
_MOCHA = {
    "canvas_bg": "181825",
    "node_bg":   "45475a",
    "node_fg":   "cdd6f4",
    "panel_bg":  "1e1e2e",
    "text_muted": "585b70",
    "highlight": "b4befe",
}

_LATTE = {
    "canvas_bg": "e6e9ef",
    "node_bg":   "bcc0cc",
    "node_fg":   "4c4f69",
    "panel_bg":  "eff1f5",
    "text_muted": "acb0be",
    "highlight": "7287fd",
}

def _default_palette() -> dict:
    return _LATTE if os.environ.get("GEN_THEME") == "light" else _MOCHA

_STATIC = pathlib.Path(__file__).parent / "static"
_WIDGET_JS = _STATIC / "graph_widget.js"

if not _WIDGET_JS.is_file():
    raise ImportError(
        "gen widget bundle not found. Build it with:\n"
        "  cd gen-python/jupyter-widget && wasm-pack build --target web --out-dir pkg\n"
        "  python gen-python/build_bundle.py"
    )


class GenGraphWidget(anywidget.AnyWidget):
    # anywidget reads this file and inlines it as a blob URL.
    # The file is a self-contained ESM module with the WASM bytes embedded as
    # base64 — no separate server or extension needed.
    _esm = _STATIC / "graph_widget.js"
    _css = ""

    # All three traitlets are raw JSON strings so Rust serialization is used
    # end-to-end with no Python parse/re-serialize round-trips.
    topology = traitlets.Unicode("{}").tag(sync=True)
    palette = traitlets.Unicode("{}").tag(sync=True)
    path_nodes = traitlets.Unicode("[]").tag(sync=True)

    def __init__(self, block_group, repository, **kwargs):
        topology = block_group.to_widget_json()

        palette = kwargs.pop("palette", _default_palette())
        palette_json = json.dumps(palette)

        path_nodes = kwargs.pop("path_nodes", None)
        if path_nodes is None:
            path_nodes_json = block_group.path_nodes_json()
        else:
            path_nodes_json = json.dumps(path_nodes)

        super().__init__(
            topology=topology,
            palette=palette_json,
            path_nodes=path_nodes_json,
            **kwargs,
        )
        self._block_group = block_group
        self.on_msg(self._on_message)

    def _on_message(self, widget, content, buffers):
        if content.get("type") != "get_sequences":
            return
        data_json = self._block_group.get_sequences_json(content.get("nodes", []))
        self.send({"type": "sequences_response", "data_json": data_json})
