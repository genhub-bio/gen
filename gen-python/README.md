# gen-python

Python bindings for the `gen` version control system for biological sequences.

## Structure

```
gen-python/
  src/                 ← PyO3 Rust source → built by maturin → gen.cpython-*.so
  jupyter-widget/      ← WASM Rust source → built by wasm-pack → python/gen/static/widget.js
  python/gen/          ← the actual Python package (ships in the wheel)
    __init__.py
    widget.py
    gen.cpython-*.so   ← compiled native extension (maturin output)
    static/
      widget.js        ← self-contained WASM bundle (build_bundle.py output)
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
# → writes python/gen/static/widget.js
```

Run `maturin develop` (or reinstall the package) after building the bundle so
the updated `widget.js` is picked up.

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
