# Gen Python API workflows

Use this reference for the current repository bindings. Signatures were checked
against `gen-python/src/python_api/` and the installed development extension.
Check `help()` in the actual interpreter when using a different release. The generated
stub `gen-python/python/gen/gen/__init__.pyi` has every signature and docstring.

A sequence graph is not one sequence. Read strings with `locus.sequence` (a region),
`graph.all_sequences()` (every path, as `Sequence` objects) and `graph.export_fasta(file, all_sequences=True)`.

Contents: workspace and history; remotes; imports and return types; direct edits;
loci and positions; annotations; file-driven updates; search and sequence reads;
partitioning and libraries; exports; widgets.

## Workspace and history

```python
import gen

repo = gen.Repository("path/to/workspace", committer="Name", email="name@example.org")
repo.gen_dir
repo.samples
repo.get_sequence_graphs(name=None, sample=None, collection=None)   # sample: name or Sample
repo.get_sequence_graph(graph.id)

repo.current_branch
repo.get_branches()
repo.create_branch(name, start=None)
repo.checkout(branch, create=False)
repo.delete_branch(branch)
repo.get_operations(branch=None, limit=None)
repo.get_assets(branch=None)
repo.merge(branch)
repo.apply(operation)
repo.reset(operation)
```

`Repository(path=None, committer=None, email=None)` opens or initializes a workspace;
`path` can be the workspace or its `.gen` directory. Omitted path discovers the
workspace from the current directory. File APIs below take filename strings.

`repo.samples` is a list of `Sample` objects. A sample is iterable and indexable
(including negative indices), with `.name`, `.collection`, and
`.sequence_graphs`. Graphs expose `.id` (`HashId`), `.name`, `.sample` (the owning
`Sample`), and `.collection`. Pass the typed id, or its string, to `get_sequence_graph()`.

`HashId` is hashable and compares by value; it identifies graphs, annotations, assets,
operations and branch heads. `str(hash_id)` is the hex digest.

Mutating import/update/edit/copy APIs record operations automatically. The public
API has no transaction context manager. Raw SQL (`execute`/`query`) is low-level and hidden from the stubs; use the
operation-aware APIs instead.

Branch arguments accept names or `Branch` objects. `create_branch()` creates
without switching and accepts an `Operation`, its `HashId` or a string commit ref for `start`.
`checkout(branch, *, create=False)` switches branches, not arbitrary operation
hashes. `get_operations()` returns newest-first `Operation` objects.
`merge()`, `apply()`, and `reset()` return the resulting HEAD `Operation`;
apply/reset accept an `Operation`, its `HashId` or a string ref. Reset hard-resets the current
branch. History actions require a clean working set; do not discard dirty state
to bypass an error.

`Branch` fields: `name`, `head` (`HashId`), `remote`, `is_current`, `dirty`.
`Operation` fields: `id` (`HashId`), `parent_id`, `committer`, `email`, `date`, `message`,
`is_head`.

Assets are files kept in the repository: imported files, annotation files, and anything added
with `repo.add_file(path, message=None)`, which stores a file (a README, a protocol) without
importing it as sequence and returns its `Asset`. `repo.get_assets()` lists them. `Asset` has
`id` (`HashId`), `name` (original file name), `path` (the stored copy under `.gen`, which has a
hashed filename; raises `FileNotFoundError` if not downloaded yet) and
`save_as(destination, overwrite=False)`, which copies it to a path of your choice (or into a
directory under `name`).

```python
original_branch = repo.current_branch
repo.checkout("design-experiment", create=True)
graph = repo.import_sequence("AAAACCCC", name="vector", sample="design")
graph.replace("vector:4-8", "GGGG", message="change downstream motif")
assert repo.get_operations(limit=1)[0].message == "change downstream motif"
repo.checkout(original_branch)
# Merge only when the requested workflow calls for adopting the experiment.
# repo.merge("design-experiment")
```

## Remotes

```python
cloned = gen.clone(url, path=None, committer=None, email=None)
repo.remotes
repo.default_remote
repo.add_remote(name, url)
repo.remove_remote(remote)
repo.set_default_remote(remote=None)
repo.set_branch_remote(remote=None)
repo.fetch(remote=None, branch=None)
repo.pull(remote=None, branch=None)
repo.push(remote=None, branch=None, force=False)
```

`clone()` returns an open `Repository`. Pass a destination that does not exist, is empty, or
only holds a freshly initialized workspace; do not copy files into `.gen` by hand. The clone
is already checked out on the remote's default branch (`checkout("main")` works afterwards).
`add_remote()` + `pull()` is for a repository that already has history. `add_remote()` returns a `Remote` with
`.name` and `.url`. Remote arguments accept a name or `Remote`; branch arguments
accept a name or `Branch`. Omitted remote/branch follows repository defaults and
branch tracking. Passing no remote to the setters clears that configuration.
`set_branch_remote()` applies to the current branch.

Fetch updates a remote-tracking ref without changing checkout; pull merges remote
state and transfers assets; push updates the remote. These methods return `None`
and refresh the repository connection. Re-query graphs and samples after sync or
history changes instead of relying on previously held handles from another state.

## Imports and return types

```python
repo.import_fasta(filename, sample=None, collection=None)
repo.import_reference_fasta(filename, reference, collection=None)
repo.import_genbank(filename, sample=None, collection=None)
repo.import_gfa(filename, sample=None, collection=None)
repo.import_sequence(sequence, name=None, sample=None, circular=False, collection=None)
repo.import_reference_sequence(sequence, reference, name=None, circular=False, collection=None)
repo.import_library(library_name, parts_list, sample=None, collection=None)
repo.import_library_files(library_name, parts, library, sample=None, collection=None)
```

| Import | Return |
|---|---|
| FASTA, reference FASTA, GenBank | `Sample` |
| GFA, library, library files | `SequenceGraph` |
| sequence, reference sequence | `SequenceGraph` |

There is no Python `shallow=` argument on FASTA imports. GenBank imports load
features automatically, accessible through `graph.annotations`.

`import_sequence()` accepts a string, Biopython `Seq`, or `SeqRecord`. Strings and
`Seq` values need `name`; a `SeqRecord` supplies its id unless overridden.
It imports sequence, not the record's feature annotations. Its `sample` argument
(and `import_reference_sequence()`'s `reference`) accepts a name or a `Sample`;
a sample object supplies its collection unless explicitly overridden. Do not
assume file imports or updates also accept sample objects: their sample arguments
are strings. Repeated sequence imports add graphs to a sample; duplicate graph
names fail. Each call records an operation; use FASTA for large file imports.

```python
graph = repo.import_sequence("AAAACCCCGGGGTTTT", name="vector", sample="parent")
# Biopython interoperability, when installed:
# for record in SeqIO.parse("input.fa", "fasta"):
#     repo.import_sequence(record, sample="records")
```

## Copy a sample and edit directly

```python
sample.copy(new_name, message=None)
graph.replace(target, sequence, message=None, stack=False)
graph.delete(target, message=None, stack=False)
graph.insert(sequence, before=None, after=None, message=None, stack=False)
```

`copy()` returns a new `Sample` in the same collection, retaining the parent's
graphs and annotations for independent editing; an existing sample name fails.
Replace/delete targets are region strings, loci, or annotations. These methods
edit the graph's existing sample, not a new sample. Replace and insert return the
new sequence's `Locus`; delete returns `None`. Each edit records an operation,
using `message` when supplied. Replacement/insertion require nonempty sequence;
use delete to remove sequence.

```python
source = next(sample for sample in repo.samples if sample.name == "parent")
design = source.copy("design")
graph = design[0]
replacement = graph.replace("vector:4-8", "ACAC", message="replace motif")
assert replacement.sequence == "ACAC"
inserted = graph.insert("GG", after=replacement.end())
assert inserted.sequence == "GG"
graph.delete(inserted)
assert graph.region("vector:0-16").sequence == "AAAAACACGGGGTTTT"
assert source[0].region("vector:0-16").sequence == "AAAACCCCGGGGTTTT"
```

`before` and `after` are keyword-only and each accepts `Position` or
`SuperPosition`; pass exactly one (both raise `TypeError`). The insertion
connects to all current neighbors on the other side. For example, at a fork,
insertion after the shared upstream base affects all outgoing alternatives,
while insertion before one downstream option affects only that option. Combine
positions with `|` to choose several routes at once. One call adds one shared
piece; use separate calls to keep independent routes separate.

Every route that reaches a node coordinate reads every route that leaves it. An
insertion that would split such a shared coordinate, such as before the first or
after the last base of a stacked edit's original sequence, raises `ValueError`
instead of landing on the alternative too. Replace
and delete targets need only be present and connected, including across
zero-width routing blocks; they need not lie on one path.

Edits read sequence on the target strand, so replacement on a reverse locus uses
the reverse-oriented sequence. Insertion after a reverse position follows that
strand's reading direction. Targets must remain present in the edited graph;
canonical locus identity survives node splitting but does not make deleted bases
valid targets. Re-query current sequence when necessary.

`stack=True` retains original routes and adds the edit as an alternative. It
leaves the current named path unchanged. Default edits supersede targeted routes
and update affected current paths. Use all-path reads/exports to inspect stacked
alternatives; default FASTA alone will still show the original path.

## Loci and positions

```python
locus = graph.region("vector:4-8")
graph.locus(4, 8)             # same span from path coordinates, no region string
len(locus)
locus.sequence
locus.strand
locus.start()
locus.end()
locus[0]
locus[-1]
locus[1:3]
locus.slice(1, 3)
locus.reverse_complement()
position = locus.start()
position.node
position.offset
position.strand
position.graph
position + 1
position - 1
position.on(other_graph)
combined = position | locus.end()
gen.SuperPosition(position, locus.end())
combined.positions
combined.graph
combined.on(other_graph)
combined + 1
```

Region coordinates are 0-based and half-open along a named graph/path or
annotation. `graph.region()` rejects regions resolved to another graph, ambiguous
routes, empty intervals, and out-of-range spans. Graph-name regions use the current
path when available. Locus offsets count bases in reading order, including across
nodes and on the reverse strand. `start()`/`end()` are the first/last **included
base**, not interval boundary coordinates.

`graph.locus(start, end)` takes 0-based half-open path coordinates.

Integer indexing returns a `Position`; unit-step slicing returns a `Locus`.
Python slicing supports omitted/negative/clipped bounds; explicit `slice(start,
end)` requires an in-range nonempty span. Empty slices raise `IndexError` and
steps other than one fail. Reverse complement changes strand and reading order;
use `reverse_complement()`, not `[::-1]`.

A `Node` is a stretch of stored sequence: `.sequence` (its bases) and `.length`.
`Position.offset` is relative to its node, not a path coordinate. Use typed endpoints
rather than reconstructing targets from offsets.

Positions from search, regions, edits, and annotation loci carry graph context.
Arithmetic walks the current graph in reading order; a fork can turn a
`Position` into `SuperPosition`. Superpositions retain alternative endpoints;
stepping beyond a terminal fails. `.on(graph)` attaches the same address to
another graph (for example a sample copy); it does not translate path coordinates.
Unions reject attached positions from different graphs.

## Annotations

```python
graph.annotations
annotation = graph.add_annotation(locus, "motif", track="motifs")
repo.import_annotations(filename, format=None, index=None, name=None, message=None)
ad_hoc = gen.Annotation(locus, "temporary label")
```

`graph.annotations` returns database and attached-file annotations together.
`add_annotation()` requires a nonempty `Locus` and returns a persisted
`Annotation`, recording an operation. `gen.Annotation(locus, name)` constructs
an ad-hoc annotation, not a persisted feature.

`import_annotations()` records an annotation file as an asset and returns the
`HashId` of the operation that recorded it. Format is inferred unless supplied; a neighboring tabix
index is discovered unless explicitly specified. File reference names must match
the graph context to appear in `.annotations`. Use `format="gff3"`, `"bed"`, or
`"genbank"` as appropriate. `name` controls the display/track name.

Annotation fields: `.name`, `.id` (`HashId`), `.locus`, `.track`, `.metadata`.
Annotations hash and compare by id and location. Use `.locus.sequence` to inspect feature sequence and the annotation
or its locus as a direct edit target.

## File-driven updates

```python
repo.update_with_fasta(filename, sample, new_sample, region_name, collection=None)
repo.update_with_genbank(filename, sample, create_missing=False, collection=None)
repo.update_with_gfa(filename, sample, new_sample, collection=None)
repo.update_with_vcf(filename, reference=None, genotype=None, sample=None,
                     in_place=False, collection=None)
repo.update_with_gaf(filename, csv, sample, parent_sample=None, collection=None)
repo.update_with_library(sample, new_sample_name, path_name, parts_list, collection=None)
repo.update_with_library_files(sample, new_sample, path_name, library, parts, collection=None)
```

VCF returns `list[Sample]`; GAF and other updates return a `Sample`. FASTA,
GFA, and library updates create the named output sample. To replace a region with a literal
sequence, `sample.copy()` it and call `graph.replace()` on the copy. GenBank updates the
specified sample; `create_missing=True` allows missing graphs. VCF exposes
`in_place`; do not assume every update creates a new sample.

```python
edited = repo.update_with_fasta(
    "parts.fa", sample="parent", new_sample="file-style-design",
    region_name="vector:4-8",
)
# VCF reference names the parent/reference; sample selects the VCF sample column.
# variants = repo.update_with_vcf("variants.vcf", reference="ref", sample="sequenced")
# variant_graph = variants[0][0]
```

## Search and sequence reads

```python
graph.search(query, sequence_kind="dna")
repo.search(query, bgs=None, sequence_kind="dna")
graph.build_index(sequence_kind="dna", k=16)
repo.build_index(sequence_kind="dna", k=16)         # every sequence graph
graph.clear_index()
repo.clear_index(bgs=None)
graph.all_sequences()
graph.to_dict()
graph.to_networkx()
graph.to_rustworkx()
```

Graph search returns `list[Locus]`; repository search returns
`list[(SequenceGraph, list[Locus])]`. `bgs` is a list of graph objects.
Kinds: `"exact"`, `"dna"`, `"ssdna"`, `"protein"`. Exact is case-sensitive raw-byte
matching without IUPAC expansion or reverse-complement search; DNA matches both
strands with IUPAC support. Check result count and strand before choosing a target.
Build an index for repeated large-graph searches.

`locus.sequence` is a string in reading order; `str(locus)` also reads sequence.
`all_sequences()` is an iterator of `Sequence` objects for the current graph alternatives
(`str(sequence)` is the bases; they also compare, hash, slice and sort like that string);
combinatorial enumeration can be exponential. `node.sequence` reads a `Node`'s bases. Repository and graph handles cannot
cross threads; open a repository in the worker and retrieve the graph by its typed
id instead. NetworkX/rustworkx conversions need
their optional packages and return `DiGraph`/`PyDiGraph`, respectively.

## Partitioning, translation, and combinatorial libraries

```python
graph.subgraph(new_sample, locus)            # or (new_sample, start_position, end_position)
graph.subgraph(new_sample, start, end)       # integer path coordinates
graph.chunks(new_sample, breakpoints=None, chunk_size=None)
repo.stitch(parts, new_sample, new_region)
graph.translate_annotation(region=None, output_collection=None, name=None,
                           strand=None, frame=0, codon_table=1, start=None)
```

`graph.subgraph()` takes a `Locus` (only its first and last positions are used, so a locus from
another graph works if both are still in this one), two `Position`s, or two integer path
coordinates, keeps every variant route between the ends, and returns a `SequenceGraph` in
`new_sample` (with a current path when this graph's path runs from start to end); `graph.chunks()` returns
`list[SequenceGraph]`. Breakpoints are a list of integer coordinates.

`repo.stitch(parts, new_sample, new_region)` joins `SequenceGraph` and `Locus` parts end to end
into a new `SequenceGraph`, e.g. `repo.stitch([a.region("a:0-100"), b], "asm", "construct")`.
A `Locus` is linear, but the subgraph of **every variant** between its first and last positions
is stitched, so no subgraph needs deriving first. Parts must share a collection, be forward-strand
and not overlap. A `SequenceGraph` part must have a current path;
a subgraph taken between positions off the current path may not, so stitch a `Locus` of it
instead (a `Locus` part needs no path).

Translation returns a protein `SequenceGraph` in the source sample. A string
region resolves first as a path, then an annotation in this graph; passing an
`Annotation` disambiguates by id. Translation reads from `start` (or the feature's
entry point) until the first in-frame stop; an end coordinate does not bound it.

Library `parts_list` is a list of columns, each containing alternative
named `gen.Sequence(name, sequence)` objects. Single-option columns are fixed flanks.

```python
parts_list = [
    [gen.Sequence("left", "AAAA")],
    [gen.Sequence("option-a", "CCCC"), gen.Sequence("option-b", "GGGG")],
    [gen.Sequence("right", "TTTT")],
]
library = repo.import_library("cassette", parts_list, sample="library")
assert {str(s) for s in library.all_sequences()} == {"AAAACCCCTTTT", "AAAAGGGGTTTT"}
```

For file libraries, `parts` is a named-parts FASTA and `library` is a headerless
CSV with one slot per column and alternatives in rows. To replace an existing
region with a library use `update_with_library()`/`update_with_library_files()`.

## Exports

```python
graph.export_fasta(filename, all_sequences=False)
graph.export_genbank(filename)
graph.export_gfa(filename, node_max=None)
```

All return `None`. Graph-level exports cover the **entire sample**.
FASTA defaults to current named paths; `all_sequences=True` enumerates alternatives
with `"{graph_name}.{index}"` headers (1-based). Use it to inspect stacked edits or
libraries, mindful of the number of combinations. GenBank preserves annotations;
GFA preserves graph structure.

## Widgets in agents and Jupyter

```python
widget = graph.plot(rows=20, cols=100, show_history=False)
# sample.plot() pages through the sample's graphs.
widget.show(locus, color="cyan", center=False)    # each returns the widget
widget.go_to(locus.start(), center=False)
widget.refresh()
widget.next_page(); widget.prev_page()
widget.zoom_in(); widget.zoom_out()
widget.scroll_left(); widget.scroll_right(); widget.scroll_up(); widget.scroll_down()
print(widget)   # text rendering
```

Terminals, scripts, and agent REPLs get `TextGraphWidget` (no Jupyter needed). A live
Jupyter kernel with optional dependencies gets `GraphWidget`; otherwise it falls back to text.
Both have the same methods, and every navigation or display method returns the widget, so
calls chain: `print(widget.show(locus).zoom_in().scroll_right())`. In notebooks use the
interactive widget freely. `show()` navigates and highlights loci/annotations;
positions/superpositions navigate without highlighting. `go_to()` navigates only.
Refresh after graph edits; paging (`next_page`/`prev_page`) switches graphs while scrolling
moves within a graph.

Graph plot supports `detail`, `colors`, and `show_history`; sample plot
supports `colors` and `show_history`. History display includes retired/pruned
edges dimmed; it is not an operation diff.

Also on both widgets: `clear_highlights()`, `show_path()`, `hide_path()`,
`show_track(name)`, `hide_track(name)`, `.tracks`, `hide_all_tracks()`, `handle_click(col, row)`.
Persist annotations through the graph API (`graph.add_annotation`).

Compact graph rendering uses one period (`.`) per skipped sequence run and `N.N`
for unannotated long nodes. Annotation tails have brackets, directed heads use
`<` or `>`, and undirected features have brackets at both ends.
