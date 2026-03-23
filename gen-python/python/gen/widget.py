"""Jupyter widget for interactive gen graph visualization.

Uses anywidget to embed the gen-tui WASM renderer in a notebook cell.
The WASM bundle (gen-python/python/gen/static/widget.js) is built separately
via `python gen-python/build_bundle.py` after running wasm-pack.

Typical usage — install gen-python, then in a notebook::

    repo = gen.Repository()
    bg = repo.get_block_groups()[0]
    bg  # renders the interactive graph widget automatically

The _ipython_display_ hook on PyBlockGroup is attached at the bottom of
gen/__init__.py when both gen_widget and IPython are available.
"""

import json
import pathlib

import anywidget
import traitlets

_STATIC = pathlib.Path(__file__).parent / "static"
_WIDGET_JS = _STATIC / "widget.js"

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
    _esm = _STATIC / "widget.js"
    _css = ""

    # Topology JSON sent to the frontend at creation time.
    topology = traitlets.Dict({}).tag(sync=True)

    def __init__(self, block_group, repository, **kwargs):
        topology, self._spec_map = _serialize_topology(block_group, repository)
        super().__init__(topology=topology, **kwargs)
        self._repository = repository
        self.on_msg(self._on_message)

    def _on_message(self, widget, content, buffers):
        if content.get("type") != "get_sequences":
            return
        result = {}
        for spec in content.get("nodes", []):
            node_key = self._spec_map.get(spec)
            if node_key is not None:
                result[spec] = self._repository.get_block_sequence(node_key)
        self.send({"type": "sequences_response", "data": result})


def _serialize_topology(block_group, repository):
    """Convert a PyBlockGroup to TopologyResponse JSON and a spec→PyNodeKey map.

    The TopologyResponse schema matches what mount_app() in lib.rs expects:
      {"nodes": [...GraphNode...], "edges": [[src, dst], ...]}

    The spec map is used in _on_message to look up PyNodeKey objects by their
    spec string ("node_id_hex:start-end") so we can call get_block_sequence.
    """
    graph_dict = repository.block_group_to_dict(block_group)
    spec_map = {}
    nodes = []

    for node_key, node_data in graph_dict["nodes"].items():
        node_id_hex = str(node_key.node_id)
        seq_start = node_key.sequence_start
        seq_end = node_key.sequence_end
        spec = f"{node_id_hex}:{seq_start}-{seq_end}"
        spec_map[spec] = node_key
        nodes.append(
            {
                "block_id": node_data["block_id"],
                "node_id": node_id_hex,
                "sequence_start": seq_start,
                "sequence_end": seq_end,
            }
        )

    edges = []
    for src_key, dst_key in graph_dict["edges"]:
        src_data = graph_dict["nodes"].get(src_key, {})
        dst_data = graph_dict["nodes"].get(dst_key, {})
        edges.append(
            [
                {
                    "block_id": src_data.get("block_id", 0),
                    "node_id": str(src_key.node_id),
                    "sequence_start": src_key.sequence_start,
                    "sequence_end": src_key.sequence_end,
                },
                {
                    "block_id": dst_data.get("block_id", 0),
                    "node_id": str(dst_key.node_id),
                    "sequence_start": dst_key.sequence_start,
                    "sequence_end": dst_key.sequence_end,
                },
            ]
        )

    return {"nodes": nodes, "edges": edges}, spec_map
