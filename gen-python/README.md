# gen-python

Python bindings to the Gen version control system for genetic sequences.

The package installs the `gen` command-line client and exposes the full Gen data
model — repositories, sequence graphs, import/export pipelines — from Python and
Jupyter notebooks. An optional Jupyter widget provides interactive graph
visualization. Plotting falls back to a readable textual rendering without the
extra or outside a live Jupyter kernel, including terminal and AI REPL sessions.

## Quick start

`Repository` import/update/query methods return live `Sample` or `SequenceGraph`
objects directly — never bare ids or names — so you can chain calls instead of
looking things up afterward:

```python
import gen

repo = gen.Repository("path/to/.gen")

sample = repo.import_fasta("path/to.fa")     # -> Sample
sg = sample[0]                               # -> SequenceGraph
samples = repo.samples                       # -> list[Sample]
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

## Sequence editing

`SequenceGraph.replace(target, sequence)` and `delete(target)` edit the sequence
a target covers. Targets accept region strings, search-result `Locus` objects,
and `Annotation` objects. Replacement and deletion remove the target from the
active path; later edits using those removed positions raise `ValueError`.

Insertions are addressed by a `Position`, such as `locus.start()` or
`locus.end()`, the first and last position of a locus in reading order.

Index or slice a search-result locus in sequence-reading order, across node boundaries:

```python
locus[100]       # Position
locus[100:150]   # Locus, half-open interval [100, 150)
locus[-1]        # final Position, equivalent to locus.end()
locus[:]        # a Locus covering the same span
```

Use Python slice syntax `locus[100:150]`; `locus[100-150]` evaluates to `locus[-50]`.
Offsets are relative to the locus, not to a graph node. `locus[0]` equals
`locus.start()`. Negative indices count from the end; omitted and negative slice
bounds follow Python conventions, and slice bounds are clipped to the locus.
Out-of-range indices (including all indices on an empty locus) and empty or
inverted slices raise `IndexError`. Keys must be integers (including objects
implementing `__index__`) or slices; other keys raise `TypeError`. As with sample
indexing, `False` and `True` act as `0` and `1`. Only slice step `1` is supported;
other steps raise `ValueError`. The explicit `.slice(start, end)` method retains
its strict bounds checking.

For `bar = foo.reverse_complement()`, `bar[i]` addresses the same physical base
as `foo[len(foo) - 1 - i]`, with the opposite strand. Thus `foo[0] != bar[-1]`,
even though they refer to the same base. A Position's `offset` is relative to its
displayed node. Later edits can split nodes without changing the location a saved
Position refers to. Strand is part of Position identity and controls the reading
direction of `insert(after=...)` and `insert(before=...)`.

Loci compare equal and have the same hash when they cover the same positions in
the same reading order and strands, even if intervening edits split nodes.
You can use them directly as dictionary keys or set members; no manual normalization
is needed. Saved loci also work with `Annotation(locus, name)`, widget `show()`
and `go_to()`, and annotation tracks after unrelated edits.

Indexed positions can be passed to `sg.insert("ACGT", after=locus[100])` or
`sg.insert("ACGT", before=locus[100])`. Sliced loci can be passed to
`sg.replace(locus[100:150], "ACGT")` and `sg.delete(locus[100:150])`.

`insert(sequence, after=position)` adds the sequence right after that position
and `insert(sequence, before=position)` right before it. Given only one side,
the insertion attaches to whatever reads next to it, so inserting after the last
position before a fork puts the new sequence on every branch. Pass both,
`insert(sequence, after=left, before=right)`, to insert on the one junction
where `left` reads directly into `right` and leave the other branches as they
were. The two positions must flank a junction; `insert` never removes sequence,
so use `replace` to swap out what lies between two positions. A position keeps
naming the same point of its node when later edits split that node.

To insert at several places at once, combine positions with `|`:
`position_a | position_b` gives a `SuperPosition`, and `|` also joins a
superposition with a position or with another superposition. `after` and
`before` each accept a `Position` or a `SuperPosition`, and a superposition
attaches the inserted sequence to every position it covers.

Positions and superpositions know the sequence graph their locus came from, so
they can walk it: `position + 1` is the next position on its strand, and
`position - 1` the one before. The result stays a `Position` while the step lands
on a single point and becomes a `SuperPosition` once it lands at a fork, one
position per branch. A superposition that covers several positions does not
step. Use `position.on(sg)` to attach a position to a different sequence graph
that shares its nodes, such as a copy of the sample.

Both sides matter most in a combinatorial layer, where every part of one column
reads into every part of the next. With parts `a1`, `a2`, `a3` each reading into
`b1`, `b2`, `b3`, `after=a1.end()` puts the new sequence between `a1` and all of
`b1`, `b2`, `b3`, and adding `before=b1.start()` narrows it to `a1` into `b1`.
`after=SuperPosition(a1.end(), a2.end())` routes `a1` and `a2` through the new
sequence and leaves `a3` reading straight into the next column. One call inserts
one piece of sequence connected to everything its sides name, so to keep
separate routes apart, insert once per route:

```python
# a reads into b, and separately c reads into d. A single insert after
# SuperPosition(a.end(), c.end()) would also let a read on into d and c into b.
for left in (a.end(), c.end()):
    sg.insert("ACGT", after=left)
```

Each call runs one transaction and records one operation. Failures roll back
both the edit and its operation record. Every editing method accepts an
optional `message` used as the operation's commit message; when omitted, a
description of the edit is generated instead. `stack=True` adds the edit as an
alternative next to what is already there and leaves the active path unchanged.

At a join or fork, `delete` and `replace` reconnect every route at the target's
boundary. This also applies immediately downstream of a heterozygous call;
shared edits use alternative routes rather than a single chromosome copy.
See [Delete and replace near junctions](examples/editing_junctions.ipynb) for
examples with follow-up insertions and stacking.

```python
for annotation in sg.annotations:
    sg.delete(annotation.locus)

inserted = sg.replace("chr1:100-110", "ACGT")
sg.insert("TT", after=inserted.end())
sg.delete(inserted.slice(1, 3))
```

`replace` and `insert` return the `Locus` of the new sequence.
Saved loci remain valid when unrelated edits shift or carve the
graph. Use `Sample.copy()` to create a complete child sample before editing its sequence graphs. The
destination name must be new; copying an existing sample raises an error.

```python
child = sample.copy("edited")
sequence = child[0]
inserted = sequence.replace(annotation, "ACGT")
```

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
  control, plus higher-level helpers like `go_to()`, `show()`, `clear_highlights()`,
  and track management (`load_track()`, `show_track()`, `hide_track()`, `tracks`,
  `hide_all_tracks()`).

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

`gen-python/Makefile` has four testing targets:

- `bindings-test` — runs `cargo test` with the pyenv-managed Python interpreter set
  as the PyO3 Python — necessary because PyO3 must link against the same Python
  that will load the extension. Use it when working on the Rust layer.
- `api-test` — rebuilds the extension into the project-root `.venv` and runs
  `unittest discover` over `tests/`, exercising the installed extension's public
  API (including remote clone/push/pull/fetch against mock HTTP and file remotes).
- `notebook-test` — rebuilds the extension into the project-root `.venv` and runs
  `pytest --nbmake` over `examples/`, executing every example notebook end to end.
- `test` — runs all of the above.

```sh
cd gen-python && make bindings-test  # Rust-layer tests
cd gen-python && make api-test       # installed public API tests
cd gen-python && make notebook-test  # example notebooks
cd gen-python && make test           # all three
```

## For AI agents

`Sample.plot()` / `SequenceGraph.plot()` return a `TextGraphWidget` in plain
Python, including AI REPLs, even when `gen[jupyter]` is installed. Drive and
inspect it without a browser or JS; use `repr(widget)` to see the current state
as ASCII.

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
sample = repo.import_fasta("path/to.fa")  # or repo.samples[0], etc.

widget = sample.plot()       # GraphWidget; pages through the sample's sequence graphs
print(repr(widget))          # plain-text fallback, e.g. "[1/20] <name> ..."

widget.next_page()           # switch graphs: next_page() / prev_page()
widget.zoom_in()             # zoom_in() / zoom_out()
widget.scroll_right()        # pan within current graph: scroll_left/right/up/down()
print(repr(widget))          # check the result via repr(), no browser needed
```
