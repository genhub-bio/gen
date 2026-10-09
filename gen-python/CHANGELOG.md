# Python API changelog

Changes on `restack/py-agent-editor` relative to `main`.

## Breaking changes

### Removed or renamed classes

- `NodeSlice` and `SequencePart` are removed. Use `Sequence` (a named piece of sequence) for library parts and `Locus` for graph spans.
- `Position` and `SuperPosition` replace the old graph position types. `GraphController.go_to_pos` now takes a `Position`.
- `Locus.slices` is gone. A `Locus` is now a linear span with `sequence`, `strand`, `start`, `end`, `len()` and slicing.
- `HashId` can no longer be built directly with `HashId(...)`.

### Repository

- `get_samples()` is now the `samples` property.
- `get_remotes()` is now the `remotes` property.
- Removed: `get_node_sequence`, `get_sequence_graph_by_id`, `get_sequence_graphs_by_collection`, `make_stitch`, `update_with_sequence`, `Repository.plot`.
  - Use `get_sequence_graph(id)` and `get_sequence_graphs(name=, sample=, collection=)` for lookups.
  - Read node text from the new `GraphNode.sequence` or `Node.sequence`.
  - `make_stitch` (which took sample and region strings) has no direct replacement; use `stitch` with `SequenceGraph` or `Locus` parts.
  - `update_with_sequence` is replaced by `Sample.copy(new_name)` followed by `SequenceGraph.replace(region, sequence)` on the copy. These are two operations rather than one. `no_reference_path_update=True` corresponds roughly to `stack=True`.
- `derive_chunks`, `derive_subgraph`, `export_fasta`, `export_genbank` and `export_gfa` are no longer public on `Repository`. Call `chunks`, `subgraph` and the `export_*` methods on `SequenceGraph`.
- `bgs=` is renamed to `sgs=` in `search` and `clear_index`. `build_index` drops its `bgs` argument.
- `stitch` is rewritten. It takes a list of `SequenceGraph | Locus` parts and returns the new `SequenceGraph`.
- `import_library` and `update_with_library` take `Sequence` parts instead of `SequencePart`.
- Methods that took a branch, remote, operation or sample now accept either a string or the object. This covers `checkout`, `merge`, `apply`, `reset`, `push`, `pull`, `fetch`, `remove_remote`, `set_default_remote` and `get_assets`.
- `update_with_vcf(reference=)` accepts a string, a list of strings or `None`.

### SequenceGraph

- Removed: `get_node_sequence`, `list_annotations`. Read node text from `GraphNode.sequence`. Use the `annotations` property for annotations.
- `chunks` drops its `backbone` argument.
- `subgraph` takes `Position | int` for its bounds.
- `translate_annotation(region=)` accepts a string, an `Annotation` or `None`.

### Annotation, GraphLocus and GraphController

- `Annotation.group` and `Annotation.segments` are removed. Use `Annotation.locus`.
- On the graph controller, these are removed: `add_annotation`, `add_track_annotations`, `add_track_file`, `clear_path`, `get_annotation_names`, `get_track_names`, `list_annotations`, `remove_annotation`.
  - `hide_path` replaces `clear_path`.
  - The `track_names` and `annotations` properties replace the getters.

### Widgets

- `GraphWidget` and `TextGraphWidget` are no longer exported from `gen`. Widgets are only returned by `plot()`.
- `dir(gen)` lists only the public names.
- Every navigation method returns the widget so calls can be chained: `zoom_in`, `zoom_out`, `scroll_*`, `next_page`, `prev_page`, `go_to`, `show`, `show_path`, `refresh` and the clear/hide methods. They previously returned `None`.
- Annotation track methods are renamed:
  - `add_annotation_track` is replaced by `show_track(name)`.
  - `remove_annotation_track` is replaced by `hide_track(name)`.
  - `clear_all_annotations` is replaced by `hide_all_tracks()`.
  - `annotation_tracks` is replaced by `tracks`.
  - `clear_path` is replaced by `hide_path`.
  - `highlight_match` is removed.
  - The widget `add_annotation`, `annotations`, `list_annotations` and `remove_annotation` methods are removed.
- `TextGraphWidget` has the full `GraphWidget` API. Its orientation header is shorter.

## New features

### Editing

- `SequenceGraph.insert(sequence, *, before=, after=, message=, stack=)` inserts sequence next to a `Position` or `SuperPosition`. It returns the `Locus` of the new sequence.
- `SequenceGraph.replace(target, sequence, message=, stack=)` replaces a region, `Locus` or `Annotation` and returns the `Locus` of the replacement.
- `SequenceGraph.delete(target, message=, stack=)` deletes the sequence under a region, `Locus` or `Annotation`.
- `stack=True` on any of the three adds the edit as a sibling option and leaves the current path unchanged.
- Each edit is recorded as its own operation, with an optional commit message.
- `Sample.copy(new_name, message=)` copies a sample into a new one so it can be edited in place. It is recorded as one operation.
- `Sample` is a live view. `sequence_graphs` is a property and the sample supports `len()`, indexing and iteration (`SampleIterator`).
- `Repository.sample(name, collection=)` returns a sample handle even before any graphs exist. `get_sample` still requires existing graphs.

### Positions and loci

- `Position` has `node`, `offset`, `strand` and `graph`, plus `on(graph)`, `+`, `-`, `|`, equality and hashing.
- `SuperPosition` is built with `SuperPosition(*positions)` or `a | b`. It has `positions`, `graph`, `on()`, `+`, `-`, `|` and `len()`.
- `SequenceGraph.locus(start, end)` and `SequenceGraph.region("chr1:100-200")` return a `Locus`.
- `Locus` has `sequence`, `strand`, `start`, `end`, `slice`, `reverse_complement`, `len()`, indexing and equality. `Locus.sequence` is new; `strand` stays a property.
- `Sequence(sequence, name=)` is a new public class with `str`, `len`, hash, comparison, `in` and slicing. It accepts the same inputs as `import_sequence`.

### Imports and annotations

- `Repository.import_sequence` and `import_reference_sequence` add an in-memory string, Biopython `Seq` or `SeqRecord` as a sequence graph. A `SeqRecord` supplies its own name.
- `exist_ok=True` on `import_sequence`, `copy` and `checkout` returns the existing result instead of raising when it already matches.
- `Repository.import_annotations` persists annotation files.
- `Repository.add_file` and `Repository.get_assets` manage tracked assets. `Asset` gains `id`, `name`, `path`, `save_as()`, equality and hashing.
- `SequenceGraph.add_annotation` persists an annotation from Python.
- `Repository.add_reference_alias` declares equivalent sequence names (RefSeq, GenBank, Ensembl, UCSC and so on) so VCF and annotation files in another naming scheme still match.
- `SequenceGraph.all_sequences()` lists every path sequence as `Sequence` objects.
- `SequenceGraph.export_fasta(filename, all_sequences=True)` writes every path.

### Other additions

- `Repository(path, committer=, email=)` and `clone()` set the committer identity.
- `gen.clone` reuses an untouched initialized workspace.
- `Branch.head`, `Operation.id` and `Operation.parent_id` are new properties.
- `Operation` and `Annotation` gain `==` and hashing.
- `GraphNode` gains `id`, `sequence`, `py_sequence_start` and `py_sequence_end`.
- Relative file paths passed to the bindings resolve against the current directory.
- Search reports each hit once and lists chunks in numeric order.
- `Asset.save_as` restores the original bytes now that text assets are archived as BGZF.

## Tooling and docs

- Type stubs are generated with `pyo3-stub-gen` into `python/gen/gen/__init__.pyi`, and `py.typed` is added.
- `make stubs` and `make stubs-check` are new Makefile targets.
- pyo3 is upgraded to 0.27.
- `API.md` documents the surface, and the README is rewritten.
- New example notebooks: `editing_primitives` and `editing_targets_and_junctions`. All other example notebooks are updated for the API changes.
- The agent skill is replaced by a `gen-sequences` router with per-task references.
