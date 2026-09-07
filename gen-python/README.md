# gen-python

Python bindings to the Gen version control system for genetic sequences.

The package installs the `gen` command-line client and exposes the full Gen data
model — repositories, sequence graphs, import/export pipelines — from Python and
Jupyter notebooks. An optional Jupyter widget provides interactive graph
visualization.

## Quick start

`Repository` import/update/query methods return live `Sample` or `SequenceGraph`
objects directly — never bare ids or names — so you can chain calls instead of
looking things up afterward:

```python
import gen

repo = gen.Repository("path/to/.gen")

sample = repo.import_fasta("path/to.fa")     # -> Sample
sg = sample[0]                               # -> SequenceGraph
samples = repo.get_samples()                 # -> list[Sample]
graphs = repo.get_sequence_graphs()          # -> list[SequenceGraph]

sample.plot()  # or sg.plot()
```

Clone and version-control repositories with the same object-oriented workflow:

```python
repo = gen.clone("https://www.genhub.bio/api/repos/owner/repository")
feature = repo.checkout("experiment", create=True)

# Import or update sequences on the feature branch.
repo.checkout("main")
repo.merge(feature)

operations = repo.get_operations()
repo.reset(operations[1])
```

`gen.clone(url, path=None)` accepts a string or an `os.PathLike` destination,
including `pathlib.Path`. The destination must be new or an empty directory.
Omitting `path` creates a directory named after the remote beneath the current
directory.

`checkout(branch)` accepts a branch name or a `Branch` object. Use
`checkout("experiment", create=True)` to create a branch at the current HEAD and
switch to it in one call; it raises an error if that branch already exists.
To create a branch without switching, use `create_branch(name, start=None)`.

Push, pull, and fetch use the same remote selection, authentication, retries,
asset transfer, and conflict behavior as the command-line client:

```python
# Arguments may be names or the corresponding Remote/Branch objects.
origin = repo.get_remotes()[0]
main = repo.current_branch

repo.push(remote=origin, branch=main)
repo.pull()                    # tracked remote and current branch
repo.fetch(branch="feature")  # updates origin/feature without checkout
```

Repositories opened with `gen.Repository(...)` can configure remotes directly:

```python
origin = repo.add_remote("origin", "file:///path/to/another/repository")
repo.set_default_remote(origin)  # available as repo.default_remote
repo.set_branch_remote(origin)  # omit the argument to clear branch tracking
remotes = repo.get_remotes()
repo.remove_remote(origin)
```

`push(remote=None, branch=None, force=False)`,
`pull(remote=None, branch=None)`, and `fetch(remote=None, branch=None)` allow you
to optionally override the remote or use a different branch than is currently
checked out. For operations or repositories that require authentication, an API
key can be set through the environment variable `GENHUB_API_KEY`. If this variable
is not set, the extension falls back to the same login process the CLI uses. See 
the [branches, remotes, and authentication notebook](examples/branches_and_remotes.ipynb)
for a complete walkthrough.

## Architecture

The package is built from three layers:

### Client (`src/main.rs`)

The existing Rust command-line client is compiled separately and staged in
maturin's wheel data `scripts` directory. Package installers place that executable
on `PATH` as `gen` on macOS and Linux or `gen.exe` on Windows.

### Rust (`src/python_api/`)

The core of the package. [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs)
compile the Gen engine into a native extension module (`gen.so`). Release wheels
use CPython's stable ABI with Python 3.11 as the minimum supported version. This
layer owns:

- **`Repository`** — opens a Gen workspace, drives all import/export operations
  (FASTA, GenBank, GFA, VCF, GAF, …), exposes version-control and remote
  workflows, and provides node/sample/sequence-graph queries. These methods
  return live `Sample` (`PySample`) and `SequenceGraph` (`PySequenceGraph`)
  objects.
- **`Branch`, `Operation`, `Remote`** — typed version-control values accepted
  directly by the corresponding `Repository` methods.
- **`Node`, `NodeSlice`, `HashId`, `Annotation`, `SequencePart`** — typed
  wrappers around internal objects so Python code can work with them safely.
- **`PyGraphController`** — wraps the GraphController and owns the ratatui render loop for the Jupyter widget. On each
  frame request it renders the graph into a ratatui `Buffer` and serialises the
  result to a JSON structure that the frontend can paint.

### Python (`python/gen/`)

A thin layer on top of the compiled extension. `__init__.py` re-exports everything
from the native module at the package level. `jupyter_widget.py` contains `GraphWidget`,
an [anywidget](https://anywidget.dev) subclass that:

- holds an internal Rust graph controller and requests rendered frames from it,
- syncs frames to the browser frontend via the `frame` traitlet (plus `page_count`/
  `page_index` for the pager indicator),
- forwards mouse and drag events from the frontend to Rust,
- exposes `zoom_in()`/`zoom_out()`, `scroll_left()`/`scroll_right()`/`scroll_up()`/
  `scroll_down()`, `next_page()`/`prev_page()`, and `refresh()` for programmatic
  control, plus higher-level helpers like `go_to()`, `show()`, `highlight_match()`,
  and annotation-track management.

The Python layer does no rendering or layout logic itself; it is a bridge.

### JavaScript (`python/gen/static/jupyter_widget.js`)

Loaded by anywidget directly in the browser. Responsible for:

- painting each received JSON frame onto an HTML `<canvas>`,
- capturing mouse events and posting them back as custom widget messages.

## Building

```sh
make          # from the project root — builds the native extension via maturin
make python-wheel  # builds a wheel containing the extension and client
make jupyter  # also builds the JS widget bundle and installs the `jupyter` extras
```

## Testing

`gen-python/Makefile` has four targets:

- `pyenv-test` — runs `cargo test` with the pyenv-managed Python interpreter set
  as the PyO3 Python — necessary because PyO3 must link against the same Python
  that will load the extension. Use it when working on the Rust layer.
- `installed-test` — rebuilds the extension into the project-root `.venv` and runs
  `unittest discover` over `tests/`, exercising the installed extension's public
  API (including remote clone/push/pull/fetch against mock HTTP and file remotes).
- `notebook-test` — rebuilds the extension into the project-root `.venv` and runs
  `pytest --nbmake` over `examples/`, executing every example notebook end to end.
- `test` — runs all of the above.

```sh
cd gen-python && make pyenv-test      # Rust-layer tests
cd gen-python && make installed-test  # installed-extension tests
cd gen-python && make notebook-test   # example notebooks
cd gen-python && make test            # all three
```

## For AI agents

`Sample.plot()` / `SequenceGraph.plot()` return a `GraphWidget` you can drive and
inspect from plain Python — no browser or JS required to verify behavior; use
`repr(widget)` to see the current state as ASCII.

- A widget from `sample.plot()` pages through every sequence graph in the
  sample; one from `sg.plot()` shows just that one graph (one page).
- `widget.next_page()` / `widget.prev_page()`: switch to a different sequence
  graph (paging wraps around). No-op on a single-graph widget.
- `widget.scroll_left()` / `scroll_right()` / `scroll_up()` / `scroll_down()`:
  pan the viewport by one screenful within the current graph. Distinct from
  `next_page`/`prev_page` — don't confuse "page" (sample pagination) with
  "scroll" (viewport panning).
- `widget.zoom_in()` / `widget.zoom_out()`: step the detail/zoom level.
- All of the above mutate the widget in place and re-render; print
  `repr(widget)` afterward to see the effect.

```python
import gen

repo = gen.Repository("path/to/.gen")
sample = repo.import_fasta("path/to.fa")  # or repo.get_samples()[0], etc.

widget = sample.plot()       # GraphWidget; pages through the sample's sequence graphs
print(repr(widget))          # plain-text fallback, e.g. "[1/20] <name> ..."

widget.next_page()           # switch graphs: next_page() / prev_page()
widget.zoom_in()             # zoom_in() / zoom_out()
widget.scroll_right()        # pan within current graph: scroll_left/right/up/down()
print(repr(widget))          # check the result via repr(), no browser needed
```
