# Python API overview

This guide follows the usual workflow: open a repository, import genetic sequence data, inspect
and edit graphs, then preserve or share the result. The generated type signatures are in
[`python/gen/gen/__init__.pyi`](python/gen/gen/__init__.pyi); runnable workflows are in
[`examples/`](examples/).

## 1. Open or clone a repository

`gen.Repository(path=None, committer=None, email=None)` opens or initializes a workspace. `path`
can name either the workspace or its `.gen` directory; if omitted, Gen discovers the workspace from
the current directory. `committer` and `email` set the identity on operations created through that
repository. `repo.gen_dir` returns the `.gen` directory.

Use `gen.clone(url, path=None, committer=None, email=None)` to clone a remote and return an open
`Repository`. A path may be a string or a `pathlib.Path`.

The repository is the entry point for imports, graph lookup, search, history, remotes, and assets.
`repo.samples` lists current non-empty sample views. `repo.sample(name, collection=None)` returns a
view even when it currently has no graphs; `repo.get_sample(name, collection=None)` requires at
least one matching graph. `Sample` is a live, repository-bound view over graphs sharing a sample
name and collection. It queries current membership whenever accessed, and each iteration starts
from the graphs present at that point. `repo.get_sequence_graphs(name=None, sample=None,
collection=None)` lists graphs, with optional filters; a sample argument can be either a name or a
`Sample`. `repo.get_sequence_graph(id)` resolves a graph by its `HashId` or hash string. A graph
object retains its repository connection, so use its `id` and look it up through a newly opened
repository when crossing thread boundaries.

## 2. Import sequences and build samples

Gen stores genetic sequences as **sequence graphs** so it can represent multiple possible sequences
in one structure. Those alternatives may represent the two haplotypes of a diploid genome, designs in an
experimental library, or variation found across a natural population. A graph's **paths** are
walks through its nodes and edges, each spelling one possible sequence. Most graphs also have a
single designated reference path. That path supplies a familiar linear coordinate frame for
regions and coordinates, while the graph retains other paths and alternatives around it.

A **sample** groups sequence graphs that belong to one biological sample or design. For example,
the records of a multi-contig FASTA become multiple sequence graphs in one sample. Graphs can also
be assigned to a **collection**, which groups related data sets within the repository; if omitted,
the repository's default collection is used. A sample can be iterated or indexed like a list to
access its current graphs, and `sample.sequence_graphs` gives the current graphs as a Python list.
Create an empty view with `repo.sample("design")`; the view itself does not create repository data.
Use `repo.get_sample("design")` when the view must already contain at least one graph.

Imports return the resulting `Sample` or `SequenceGraph`, which can be used directly:

```python
import gen

repo = gen.Repository("work")
sample = repo.import_fasta("reference.fa", sample="reference")
graph = sample[0]
```

Common import methods are:

- `repo.import_fasta(filename, sample=None, collection=None, fai=None, gzi=None)` imports FASTA
  records as graphs and returns their `Sample`. Pass `.fai` and `.gzi` for a remote indexed BGZF
  FASTA.
- `repo.import_reference_fasta(filename, reference, collection=None, fai=None, gzi=None)` imports
  FASTA records into a named reference sample.
- `repo.import_sequence(sequence, name=None, sample=None, circular=False, collection=None,
  *, exist_ok=False)` imports one in-memory string, Biopython `Seq`, or `SeqRecord`. A record can
  provide its own name. `sample` accepts a name or `Sample`; repeated calls can build one sample.
  `exist_ok=True` reuses a same-named graph only if its contents match; it does not overwrite that
  graph with different sequence.
- `repo.import_reference_sequence(sequence, reference, name=None, circular=False,
  collection=None, *, exist_ok=False)` is the in-memory equivalent for reference samples.
- `repo.import_gfa(filename, sample=None, collection=None)` imports a graph while preserving its
  nodes and edges. `repo.import_genbank(filename, sample=None, collection=None)` imports sequence
  and features.
- `repo.import_library(library_name, parts_list, sample=None, collection=None)` builds a
  combinatorial graph. `parts_list` is a list of columns, each a list of named `gen.Sequence`
  alternatives. `repo.import_library_files(...)` reads the same design from a parts FASTA and a
  headerless CSV.

Update methods create or update samples from files: `update_with_fasta`, `update_with_gfa`,
`update_with_gaf`, `update_with_vcf`, `update_with_genbank`, `update_with_library`, and
`update_with_library_files`. Their parameters select the source sample, result sample, target
region, collection, and format-specific options; they return the resulting `Sample` (VCF returns a
list of samples). VCF updates can create one sample per VCF sample column, select a single sample,
or edit the reference in place. Use `add_reference_alias()` when file contig names differ from the
reference graph name.

`Sample` can be indexed and iterated like a list of graphs: use `len(sample)`, `sample[index]`,
iterate over it, or read `sample.sequence_graphs`. Each access fetches current membership from the
repository; an active iterator uses the graphs it found when iteration began. `sample.name` and
`sample.collection` identify its scope. `sample.copy(new_name, message=None)` creates a child sample
with matching graphs, ready for edits, and errors if the destination name already exists.
`sample.add_metadata({...})` stores string, signed 64-bit integer, finite float, and boolean values.
`sample.metadata` returns a fresh dictionary, or `{}` when no metadata is stored. Adding metadata
updates matching keys while preserving others; the update is recorded in operation history and
follows branch checkout and reset. Metadata keys must be strings, and nested or non-finite values
are rejected.

## 3. Read, search, and inspect graphs

A `SequenceGraph` exposes `name`, `id`, `sample`, `collection`, and `annotations`.

- `graph.search(query, sequence_kind="dna")` returns matching `Locus` objects. Kinds include
  `"exact"`, `"dna"`, `"ssdna"`, and `"protein"`. `repo.search(query, sgs=None,
  sequence_kind="dna")` searches selected graphs or all graphs and returns `(graph, loci)` pairs.
- `repo.build_index(sequence_kind="dna", k=16)` indexes all graphs; `graph.build_index(...)`
  indexes one. Search loads an available index automatically. `repo.clear_index(sgs=None)` clears
  all or selected indexes; `graph.clear_index()` clears one.
- `graph.region("chr1:100-110")` resolves a region string to a locus without editing. A region
  string names a sequence graph and a 0-based, half-open interval on its linear reference path:
  `"graph_name:start-end"`. `graph.locus(start, end)` makes the equivalent span without building
  the string.
- `graph.all_sequences()` returns every possible walk through the graph as `Sequence` values.
  Different walks that spell the same bases are represented once. `str(sequence)` gives its bases.
  `Sequence(name, sequence)` also represents named library
  parts; sequence values compare and hash by bases, and support containment, indexing, and unit-step
  slicing.
- `graph.to_dict()` returns nodes and edges. `graph.to_networkx()` and `graph.to_rustworkx()`
  provide interoperability with the popular NetworkX and rustworkx graph libraries.

`Position` identifies one base by its node, offset, and strand. `position + n` and `position - n`
walk forward and backward through the graph along that strand. If a step crosses a junction and
reaches multiple positions, the result is a `SuperPosition` representing those positions at once.
You can also make a `SuperPosition` by combining positions with `|` or by calling
`SuperPosition(pos_a, pos_b, ...)`. SuperPositions allow you to edit multiple sites in the graph at
once. `position.on(graph)` and `superposition.on(graph)` use those coordinates on another graph
that shares and can reach the named nodes.

`Locus` represents an ordered graph span. Its `sequence`, `strand`, and `len(locus)` describe the
covered sequence. `locus.start()` and `.end()` return the first and last `Position` in reading
order. Integer indexing returns a position, and slicing returns a locus; negative indexes and
Python-style slice bounds are supported. Slice steps other than one and empty slices are rejected.
`locus.reverse_complement()` returns the same physical span read on the opposite strand. Loci can
be compared and used as dictionary keys.

## 4. Edit sequence graphs

Edits are methods on `SequenceGraph` and are recorded as repository operations. Each accepts an
optional `message`; the default operation message describes the edit. `stack=True` preserves the
existing route as an alternative and leaves the current path unchanged.

```python
hit = graph.search("ATG")[0]
replacement = graph.replace(hit, "GTG")
graph.insert("AA", after=replacement.end())
graph.delete("chr1:100-110")
```

- `graph.replace(target, sequence, message=None, stack=False)` replaces the target and returns the
  inserted `Locus`.
- `graph.delete(target, message=None, stack=False)` removes the target. It returns `None`.
- `graph.insert(sequence, *, before=None, after=None, message=None, stack=False)` inserts beside a
  position. Supply exactly one of `before` and `after`; each accepts a `Position` or
  `SuperPosition`. It returns the inserted `Locus`.

Replacement and deletion targets may be region strings, `Locus` values, or `Annotation` values.
Use a returned locus for follow-up edits or annotations. Saved loci and positions continue to name
their node coordinates when unrelated edits split display nodes. Edits resolve targets against the
graph and support spans crossing graph junctions. At forks, an insertion after a shared position
attaches to all routes leaving it; insertion before one branch position narrows the insertion to
that branch. A `SuperPosition` selects several sites at once. One insertion call creates one shared
piece connected to the routes selected by its positions; use separate calls when routes must remain
separate.

After a replacement or deletion, the original target is no longer accessible on the active graph.
`stack=True` keeps the original route and adds the edited genotype as another option in the graph;
the current reference path stays on the original route. A `Sample.copy()` is a convenient way to
preserve the original sample before making ordinary in-place edits to the copy.

## 5. Annotations and translation

`repo.import_annotations(filename, format=None, index=None, name=None, message=None)` records an
annotation file. The format can be inferred from the filename; supported sources include GFF,
BED, and GenBank, with a neighboring tabix index detected when available. Imported features appear
in `graph.annotations` and widget tracks. `repo.import_genbank()` also brings in GenBank features.

`graph.add_annotation(target, name, track="default")` persists an annotation over a region string,
locus, or existing annotation. `Annotation` exposes `id`, `name`, `track`, `locus`, and optional
metadata. Construct `gen.Annotation(locus, name)` for an in-memory annotation, for example to pass
to a widget; use `graph.add_annotation()` to persist it.

`graph.translate_annotation(region=None, output_collection=None, name=None, strand=None, frame=0,
codon_table=1, start=None)` translates the whole graph or a named path/annotation into a protein
graph in the same sample. The region can be a name or an `Annotation`; strand, reading frame,
codon table, output collection, and start coordinate control translation.

## 6. Derive and assemble graphs

- `graph.subgraph(new_sample, start, end)` derives every route between two points into a new
  sample. The points can be the start and end of a `Locus` (pass `locus.start()` and `locus.end()`),
  two individual `Position` objects, or two integer coordinates along the linear reference frame.
  Internal variant routes are retained.
- `graph.chunks(new_sample, breakpoints=None, chunk_size=None)` divides a graph into ordered
  subgraphs using breakpoints or equal chunk size, with coordinates measured along the linear
  reference frame.
- `repo.stitch(parts, new_sample, new_region)` joins `SequenceGraph` or `Locus` parts end to end
  into a new graph. A locus contributes all variant routes inside its bounds. Parts must be from
  one collection, forward strand, and non-overlapping; graph parts need a current path.

## 7. History, branches, and remotes

Every change is recorded as an operation in the repository database. This history lets you inspect
what happened, reset to a specific point in time, or check out a new branch from that point.
`repo.get_operations(branch=None, limit=None)` lists operations from newest to oldest. History
methods operate on the current checkout unless a branch is specified. `repo.get_branches()`
lists branches; `repo.current_branch` returns the current `Branch`. `create_branch(name, start=None)`
creates a branch at HEAD or at a specified operation. `checkout(branch, *, create=False,
exist_ok=False)` switches branches and can create a branch at HEAD. `delete_branch(branch)` removes
one. Branch arguments can be names or `Branch` objects.

`merge(branch)` merges another branch. `apply(operation)` applies an operation and `reset(operation)` moves the
current branch to one; each accepts an operation, `HashId`, or hash string. `Branch` exposes its
name, tracked remote, current/dirty flags, and head hash. `Operation` exposes committer, email, date,
message, head status, id, and parent id.

A remote is another repository that you can fetch changes from or push changes to. Configure
remotes with `add_remote(name, url)`, `remove_remote(remote)`, `set_default_remote(remote=None)`,
and `set_branch_remote(remote=None)`. `repo.remotes` and `repo.default_remote` inspect the setup.
`push(remote=None, branch=None, force=False)`, `pull(remote=None, branch=None)`, and
`fetch(remote=None, branch=None)` accept remote and branch names or objects. `fetch()` updates the
remote-tracking reference without switching the checkout. Push, pull, and fetch use the CLI's
authentication and asset-transfer behavior.

## 8. Files, assets, and exports

`repo.add_file(filename, message=None)` stores a file as a repository asset without importing it
as sequence data. `repo.get_assets(branch=None)` lists assets reachable from a branch. `Asset`
exposes a content `id`, original `name`, and stored `path`; `asset.save_as(destination,
overwrite=False)` writes a named readable copy.

Export methods are called on a `SequenceGraph`, but they export the sequence graphs in that graph's
sample and collection. `graph.export_fasta(filename, all_sequences=False)` writes current paths by
default; `all_sequences=True` writes every possible graph walk. `graph.export_gfa(filename,
node_max=None)` writes the sample's graph structure, optionally splitting long nodes.
`graph.export_genbank(filename)` writes the sample's sequences and annotations. Repository-level
export helpers also support selecting a sample and collection, but they are internal and not part
of the public Python API.

## 9. Plot and navigate

`graph.plot(...)` and `sample.plot(...)` generate a visual representation of the graph and return a
widget that you can interact with through methods such as `go_to()`, `show()`, zoom, scrolling,
highlight controls, and annotation-track visibility. Install the `jupyter` extra with
`pip install gen[jupyter]` to use the interactive notebook widget. In other Python environments,
plotting returns a text widget with navigation controls. Plot options include viewport size,
annotation colors, graph detail, and whether to show edit history.

Widgets support `go_to(target, center=False)` and `show(target, color=None, center=False)` for
positions, loci, annotations, or supported region targets. They also expose zoom, scrolling,
page navigation, refresh, highlight clearing, path visibility, and annotation-track controls such
as `show_track(name)`, `hide_track(name)`, `hide_all_tracks()`, and `tracks`.

## 10. Common value types

- `HashId` identifies stored objects; it supports equality, hashing, string conversion, and
  `to_bytes()`.
- `Node` exposes its sequence and length.
- `Remote` exposes its name and URL.
- `Asset`, `Annotation`, `Branch`, `Operation`, `Sample`, and `SequenceGraph` are returned by the
  repository or graph methods described above.

