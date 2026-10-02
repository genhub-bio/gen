# Gen Python API reference

Signatures below were captured from the installed `gen` module (`help(gen.<Class>)`)
against the current build. If Gen has been rebuilt since this file was written and a
call's arguments don't match, re-run `help()` rather than guessing — do not silently
fall back to the CLI.

## `gen.Repository(path=...)`

The workspace handle. Opens (or creates) a `.gen` directory at `path` (or the current
working directory if omitted).

```python
repo = gen.Repository("path/to/.gen")
```

### Import

```python
repo.import_fasta(filename, sample=None, shallow=False, collection=None)
repo.import_reference_fasta(filename, reference, shallow=False, collection=None)
repo.import_genbank(filename, sample=None, collection=None)
repo.import_gfa(filename, sample=None, collection=None)
repo.import_library(library_name, parts_list, sample=None, collection=None)
repo.import_library_files(library_name, parts, library, sample=None, collection=None)
```

- `import_fasta`/`import_reference_fasta` differ only in whether the sample is marked
  as a reference in the database (`reference=` vs `sample=`).
- `import_library_files`: `parts` is a FASTA of named parts, `library` is a CSV
  describing the column layout (see `import_library` for the equivalent in-memory
  form using `gen.SequencePart`).
- All import methods return a `Sample`.

```python
sample = repo.import_genbank("puc19.gbk", sample="wt")
sg = sample[0]                      # Sample is list-like: index/iterate/len()
print(sg.name, sg.sample_name)
```

### Update (branch-friendly edits)

Every `update_with_*` takes a source `sample` and a `new_sample` (or
`new_sample_name`) and returns the new `Sample` (or `list[Sample]` for
`update_with_vcf`/`update_with_gaf`, one per resulting branch). The new sample shares
the whole graph with the parent except at the edited region.

```python
repo.update_with_sequence(sequence, sample, new_sample, region_name,
                           no_reference_path_update=False, collection=None)
repo.update_with_fasta(filename, sample, new_sample, region_name, collection=None)
repo.update_with_genbank(filename, sample, create_missing=False, collection=None)
repo.update_with_gfa(filename, sample, new_sample, collection=None)
repo.update_with_vcf(filename, reference=None, genotype=None, sample=None,
                      in_place=False, collection=None)
repo.update_with_gaf(filename, csv, sample, parent_sample=None, collection=None)
repo.update_with_library(sample, new_sample_name, path_name, parts_list, collection=None)
repo.update_with_library_files(sample, new_sample, path_name, library, parts,
                                collection=None)
```

Recipes (from `gen-python/examples/introduction.ipynb` and
`gen-python/examples/repository_api.ipynb`):

```python
# Replace an explicit region with a literal sequence
sample_gg = repo.update_with_sequence(
    "GAATTCGGTCTCAAATGCATCATCATCATGCATTGAGACCAAGCTT",
    sample="wt",
    new_sample="gg_design",
    region_name="pUC19:395-452",   # 0-based, half-open
)

# Apply variants from a VCF against a reference sample
result_samples = repo.update_with_vcf(
    "sequenced.vcf",
    reference="gg_design",
    sample="sequenced",
)   # -> list[Sample], one per sample column / branch
sg_seq = result_samples[0][0]

# reference= names the parent sample; genotype= (e.g. "1/1") fixes a specific GT;
# sample= selects a VCF sample column for genotype info.

# Merge a GFA walk into an existing sample
merged = repo.update_with_gfa("walk.gfa", sample="ref", new_sample="gfa_merged")

# GAF: csv describes insertion flanks, parent_sample is the branch point
gaf_sample = repo.update_with_gaf(
    "reads.gaf", csv="insert_flanks.csv",
    sample="gaf_sample", parent_sample="ref",
)

# Swap a region for a combinatorial library
lib_updated = repo.update_with_library(
    sample="ref",
    new_sample_name="lib_updated",
    path_name="m123:5-20",
    parts_list=[
        [gen.SequencePart("tag_a", "ATGATGATG"), gen.SequencePart("tag_b", "TGATGATGA")],
    ],
)
```

### Query / navigate

```python
repo.get_samples()                              # list[Sample]
repo.get_sequence_graphs()                       # list[SequenceGraph]
repo.get_sequence_graphs_by_collection(name)      # list[SequenceGraph]
repo.get_sequence_graph_by_id(id)                 # SequenceGraph
repo.get_node_sequence(node_key)
repo.search(query, bgs=None, sequence_kind="dna") # exact search across graphs
repo.execute(query) / repo.query(query)
```

`search()` returns `list[(SequenceGraph, list[Locus])]` — one entry per graph with at
least one hit. `sequence_kind` is one of `"exact"`, `"dna"`, `"ssdna"`, `"protein"`;
`"exact"` is case-sensitive raw-byte matching with no IUPAC expansion or
reverse-complement search. `bgs=None` searches all sequence graphs. Build a seed index
first with `repo.build_index(sequence_kind="dna", k=16, bgs=None)` to speed up repeat
searches on large graphs; `clear_index(bgs=None)` removes it.

```python
hits = repo.search("GGTCTC", bgs=[sg_gg], sequence_kind="dna")
loci = hits[0][1]          # [(SequenceGraph, [Locus]), ...]
print(f"BsaI sites: {len(loci)}")
```

### Graph partitioning

```python
repo.derive_subgraph(sample, new_sample, region, backbone=None, collection=None)
repo.derive_chunks(sample, new_sample, region, backbone=None, breakpoints=None,
                    chunk_size=None, collection=None)
repo.stitch(bgs, new_sample, new_region)          # concatenate SequenceGraphs, same collection+sample
repo.make_stitch(sample, new_sample, regions, new_region, collection=None)
```

`region`/`regions` use the same `"<name>:<start>-<end>"` syntax as `region_name=`.
`stitch()` connects end nodes of each preceding block group to the start nodes of the
next, in list order.

### Export

```python
repo.export_fasta(filename, sample=None, collection=None)
repo.export_genbank(filename, sample=None, collection=None)
repo.export_gfa(filename, sample=None, node_max=None, collection=None)
```

### Transactions

```python
with repo.transaction():
    repo.import_fasta("reference.fasta")
    repo.import_gfa("graph.gfa")
```

## `SequenceGraph`

One graph (a "block group") within a sample. Cannot cross threads — capture `.id` and
reopen a fresh `Repository` in a worker thread if needed
(`repo.get_sequence_graph_by_id(sg_id)`).

```python
sg.name / sg.id / sg.sample_name / sg.collection_name
sg.get_node_sequence(node)
sg.list_annotations()                              # list[Annotation]
sg.search(query, sequence_kind="dna")               # list[Locus], scoped to this graph
sg.translate_annotation(region=None, output_collection=None, name=None,
                         strand=None, frame=0, codon_table=1, start=None)
sg.subgraph(new_sample, start, end, backbone=None)  # -> SequenceGraph, shorthand for derive_subgraph
sg.chunks(new_sample, breakpoints=None, chunk_size=None, backbone=None)  # -> list[SequenceGraph]
sg.export_fasta(filename) / .export_genbank(filename) / .export_gfa(filename, node_max=None)
sg.to_dict() / .to_networkx() / .to_rustworkx()
sg.build_index(sequence_kind="dna", k=16) / .clear_index()
sg.plot(rows=None, cols=None, detail=None, colors=None)
```

`export_*` on a `SequenceGraph` exports the whole sample it belongs to, not just this
one graph.

`translate_annotation`: `region` resolves first as a path name, then an annotation
name, scoped to this graph only; pass an `Annotation` object directly to disambiguate
by id. Reads forward from `start` (or the annotation's own entry point) to the first
in-frame stop codon — not bounded by an end coordinate. Returns a protein
`SequenceGraph` in this graph's sample.

```python
chunks = sg_gg.chunks("gg_500bp_chunks", chunk_size=500)
reassembled = repo.stitch(bgs=chunks, new_sample="gg_reassembled",
                           new_region="pUC19.reassembled")

protein_sg = sg.translate_annotation(region="lacZ", frame=0, codon_table=1)
```

## `Sample`

List-like container of the `SequenceGraph`s produced by one call.

```python
len(sample) / sample[0] / for sg in sample: ...
sample.sample_name / sample.collection_name / sample.block_groups
sample.plot(rows=None, cols=None, colors=None)   # pages through every graph in the sample
```

## `Annotation`

```python
annotation.name / .id / .locus / .group / .track / .metadata / .segments
```

Auto-loaded from GenBank imports; also constructible directly for ad-hoc widget
overlays: `gen.Annotation(locus, name)`.

## `Locus` / `Position` / `NodeSlice`

```python
locus.start() / locus.end()   # -> Position
locus.slices                  # -> list[NodeSlice]
locus.strand                  # "+" / "-"

position.node / position.offset
node_slice.node / .start / .end / .strand
```

Pass a `Locus`, `Position`, or `Annotation` directly to `widget.go_to(...)`.

## `SequencePart`

```python
gen.SequencePart(name, sequence)
```

Building block for `import_library`/`update_with_library`'s `parts_list`: a list of
columns, each a list of alternative `SequencePart`s for that position.

## `GraphWidget` (from `.plot()`)

Interactive Jupyter widget; also the headless verification surface — drive it and
`repr()` it from plain Python, no browser required.

```python
widget = sg.plot()
repr(widget)                       # e.g. "[1/20] <name> ..."
widget.go_to(locus_or_position_or_annotation)
widget.next_page() / .prev_page()  # switch graphs (Sample-backed widgets only; no-op on single-graph widgets)
widget.zoom_in() / .zoom_out()
widget.scroll_left() / .scroll_right() / .scroll_up() / .scroll_down()
widget.highlight_match(locus, color="cyan")
widget.clear_highlights()
widget.add_annotation(gen.Annotation(locus, "label"))
widget.list_annotations()
widget.clear_all_annotations()
widget.show_path()                 # highlight the currently-diverging edited path
```

All mutators act in place and re-render; re-`repr()` to see the effect. This is the
preferred way for an agent to confirm a plot's contents without a browser.

## Region string syntax (all `region`/`region_name`/`path_name` args)

`"<path or annotation name>:<start>-<end>"` — 0-based, half-open, resolved first
against a named path in the target graph, then an annotation name in that graph's
lineage.

## Graph export interop

```python
import networkx as nx
nx_graph = sg.to_networkx()          # DiGraph

import rustworkx as rx
rx_graph = sg.to_rustworkx()         # PyDiGraph
```

Both require the corresponding package installed; catch `ImportError` if optional.
