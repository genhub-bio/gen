# gen-python

Python bindings to the Gen version control system for genetic sequences.

The bindings expose the full Gen data model — repositories, sequence graphs,
import/export pipelines — from Python and Jupyter notebooks. An optional Jupyter
widget provides interactive graph visualization.
## Architecture

The package is built from three layers:

### Rust (`src/python_api/`)

The core of the package. [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs)
compile the Gen engine into a native extension module (`gen.so`). This layer owns:

- **`Repository`** — opens a Gen workspace, exposes block-group and node queries,
  and drives all import/export operations (FASTA, GenBank, GFA, VCF, GAF, …).
- **`PyBlockGroup`, `PyNodeKey`, `PyHashId`** — typed wrappers around internal
  objects so Python code can work with them safely.
- **`PyGraphController`** — wraps the GraphController and owns the ratatui render loop for the Jupyter widget. On each
  frame request it renders the graph into a ratatui `Buffer` and serialises the
  result to a JSON structure that the frontend can paint.

### Python (`python/gen/`)

A thin layer on top of the compiled extension. `__init__.py` re-exports everything
from the native module at the package level. `jupyter_widget.py` contains `GenGraphWidget`,
an [anywidget](https://anywidget.dev) subclass that:

- holds an internal Rust graph controller and requests rendered frames from it,
- syncs frames to the browser frontend via the `frame` traitlet,
- forwards mouse and drag events from the frontend to Rust,
- exposes `zoom_in()`, `zoom_out()`, `move_by()`, and `refresh()` for programmatic control.

The Python layer does no rendering or layout logic itself; it is a bridge.

### JavaScript (`python/gen/static/jupyter_widget.js`)

Loaded by anywidget directly in the browser. Responsible for:

- painting each received JSON frame onto an HTML `<canvas>`,
- capturing mouse events and posting them back as custom widget messages.

## Building

```sh
make          # from the project root — builds the native extension via maturin
```

## Testing

`gen-python/Makefile` contains a single `pyenv-test` target that runs `cargo test`
with the pyenv-managed Python interpreter set as the PyO3 Python — necessary because
PyO3 must link against the same Python that will load the extension. Use it when
working on the Rust layer:

```sh
cd gen-python && make pyenv-test
```
