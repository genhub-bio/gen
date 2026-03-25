# gen-python

Python bindings for the `gen` version control system for biological sequences.

## Structure

```
gen-python/
  src/                 ← PyO3 Rust source → built by maturin → gen.cpython-*.so
  jupyter-widget/      ← WASM Rust source → built by wasm-pack → python/gen/static/graph_widget.js
  python/gen/          ← the actual Python package (ships in the wheel)
    __init__.py
    widget.py
    gen.cpython-*.so   ← compiled native extension (maturin output)
    static/
      graph_widget_src.js    ← hand-written anywidget glue (source, not loaded directly)
      graph_widget.js        ← self-contained WASM bundle (build_bundle.py output)
  build_bundle.py      ← combines wasm-pack output into a single inlined JS bundle
  pyproject.toml       ← package metadata; maturin is the build backend
```

Neither `src/` nor `jupyter-widget/` can be compiled with a normal `cargo build`, they each have 
their own build target and toolchain. This is why `gen-python` is excluded from the root gen 
Cargo workspace (`exclude = ["gen-python"]` in `gen/Cargo.toml`), and
`jupyter-widget` declares its own `[workspace]` so Cargo stops searching
upward and never tries to pull it into a workspace that can't build it.

## Building

### Python extension (normal install)

```bash
cd gen-python
maturin develop          # or: maturin build --release
```

### Jupyter widget (requires wasm-pack)

```bash
cd gen-python/jupyter-widget
wasm-pack build --target web --out-dir pkg

cd ..
python build_bundle.py
# → writes python/gen/static/graph_widget.js
```

Run `maturin develop` (or reinstall the package) after building the bundle so
the updated `graph_widget.js` is picked up.

## Widget data flow

The widget avoids unnecessary serialization by keeping data as raw JSON strings
across every boundary.  All three anywidget traitlets (`topology`, `palette`,
`path_nodes`) are `Unicode` strings; `graph_widget_src.js` passes them straight to
`mount_app` without any `JSON.stringify` / `JSON.parse` calls.

### Initial render

```
PyBlockGroup.to_widget_json()     → JSON string (serde, native Rust)
    ↓  Unicode traitlet (no parse)
graph_widget_src.js: model.get("topology")
    ↓  passed directly
mount_app(..., topology_json, ...)  → serde_json::from_str (WASM Rust)
```

`palette` is a small Python dict that Python serialises once with `json.dumps`.
`path_nodes` is produced by `PyBlockGroup.path_nodes_json()` — the path query,
graph projection, and JSON serialisation all happen in native Rust.

### On-demand sequence fetching

Graph nodes store only coordinates (`node_id`, `sequence_start`, `sequence_end`);
actual DNA sequences are fetched lazily as the user zooms in:

```
WASM renderer needs sequence for a node
    ↓  sequence_callback(["node_id_hex:start-end", ...])
graph_widget_src.js: model.send({ type: "get_sequences", nodes })
    ↓  Jupyter comm message
widget.py _on_message: PyBlockGroup.get_sequences_json(specs)
    → single batched Node::get_sequences_by_node_ids DB query (native Rust)
    → serde_json::to_string (native Rust)
    ↓  model.send({ type: "sequences_response", data_json: "..." })
graph_widget_src.js: app.deliver_sequences(msg.data_json)
    ↓  passed directly
WASM AppHandle::deliver_sequences → serde_json::from_str → renderer cache
```

The `get_sequences_json` method on `PyBlockGroup` batches all requested node IDs
into a single database query, so one zoom event results in one round-trip
regardless of how many nodes are visible.

## Usage in a notebook

```python
import gen

repo = gen.Repository()
bg = repo.get_block_groups()[0]
bg  # renders the interactive gen-tui graph widget
```

The `_ipython_display_` hook on `PyBlockGroup` is attached automatically when
the package is imported. It degrades gracefully to `repr` if the widget bundle
has not been built or if `anywidget` is not installed.
