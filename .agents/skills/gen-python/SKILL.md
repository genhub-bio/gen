---
name: gen-genetic-engineering-python
description: Use the Gen Python bindings (import gen) to edit sequences, query and visualize sequence graphs, import/export files, run combinatorial libraries, manage branches, and clone or sync repositories. Use Gen's own editing methods instead of sed, Biopython or string slicing.
---

# Gen (Python)

Gen stores sequences as **graphs**, not strings. Edits are recorded operations on the graph,
so edit through Gen and read the result back as strings. Do not export to FASTA, edit the text
with `sed`/Biopython/string slicing and re-import: that loses history, annotations and
alternatives.

A **sequence graph can hold many sequences**: after a library import, `stack=True` edit or
VCF, one graph has many paths. `graph.all_sequences()` yields every path as a `Sequence` (`str(sequence)` is its bases).
`locus.sequence` is the string for one region. A `Sample` holds several graphs.

## Cheat sheet

Run `help(gen.Repository)` or read `gen-python/python/gen/gen/__init__.pyi` (full typed
signatures and docstrings). Longer recipes: [references/gen-python-workflows.md](references/gen-python-workflows.md).

**Open / clone**
```python
import gen
repo = gen.Repository("workspace")                 # open or create
repo = gen.clone(url, "workspace")                 # clone; the path must not hold data yet
repo.checkout("branch", create=True)  # exist_ok=True to reuse; also get_branches(); repo.get_operations(limit=5)
repo.pull(); repo.push()                           # remotes: add_remote(name, url), fetch()
```

**Load sequence** (all return a `Sample` or `SequenceGraph`)
```python
graph = repo.import_sequence("ACGT...", name="vector", sample="parent")   # string/Seq/SeqRecord
sample = repo.import_fasta("in.fa", sample="parent")   # also import_genbank, import_gfa
```

**Edit** (mutates the graph's sample; copy first to keep the original)
```python
design = sample.copy("design"); graph = design[0]
locus = graph.replace("vector:4-8", "ACAC", message="swap motif")  # 0-based, half-open
graph.insert("GG", after=locus.end());  graph.delete(locus)
graph.replace(annotation, "ACAC")           # annotations and Loci are valid targets too
```
Other edits: `repo.update_with_vcf/fasta/genbank/library`, `repo.import_library` (combinatorial).

**Read back as strings**
```python
graph.region("vector:0-16").sequence        # one region, str
[str(s) for s in graph.all_sequences()]   # every path through the graph, strs
graph.search("GAATTC")                      # list[Locus]; sequence_kind="exact" for literal
graph.export_fasta("out.fa", all_sequences=True)   # whole sample; also export_genbank/gfa
```

**Pieces of a graph** (loci, subgraphs, stitching, files)
```python
locus = graph.locus(100, 200)                       # path coordinates; same as graph.region("name:100-200")
sub = graph.subgraph("piece", locus)                # keeps every variant between its ends (also two Positions)
joined = repo.stitch([sub, other_graph.region("o:0-50")], "assembly", "construct")  # graphs or loci
asset = repo.add_file("README.md")                  # keep a file without importing it as sequence
asset.path;  asset.save_as("copy.md")               # stored copy has a hashed name; save_as renames it
```

**Annotate**
```python
graph.annotations;  graph.add_annotation(locus, "motif", track="motifs")
repo.import_annotations("features.gff3")
```

**See it** (works in plain scripts and agent REPLs: no Jupyter needed)
```python
widget = graph.plot()                       # text rendering in terminals/agents
print(widget.show(locus).zoom_in().scroll_right())   # every method returns the widget
```
`TextGraphWidget` (scripts, terminals, agents) and the Jupyter `GraphWidget` (live notebook
kernel) have the same methods: `show`, `go_to`, `zoom_in/out`, `scroll_*`, `next_page/prev_page`,
`refresh`, `show_track/hide_track/tracks`, `show_path/hide_path`, `clear_highlights`. In notebooks
use the interactive widget freely. `print(widget)` redraws the text view; call `widget.refresh()`
after graph edits.

**Showing a plot to a user** (`TextWidget` is `TextGraphWidget`): when asked to show, plot, or
compare graphs outside a live Jupyter kernel, `print(widget)` and paste the output in a fenced
code block. The first frame printed in a process starts with a `#` orientation header written for
you; drop it and show only the frame. Later frames have no header. Print one widget per graph
when comparing, and label each block with the sample or graph name.

## Rules that prevent mistakes

- Coordinates are `"<name>:<start>-<end>"`, 0-based, half-open. `locus.end()` is the last
  *included* base, not the exclusive bound. Use `Locus`/`Position` objects as targets after edits.
- Insert takes keyword `before=` or `after=` positions, not both. Choose routes by combining
  positions into a `SuperPosition` with `|`. Replace a span between positions instead of inserting.
- `stack=True` keeps original routes as alternatives; default edits supersede them. An edit
  applies to every route through its coordinates. Inserting before the first or after the last
  base of a stacked edit's original sequence raises `ValueError`, since the alternative shares
  those points; insert inside it or at the alternative's ends.
- Graph-level exports cover the **whole sample**, not only that graph.
- Mutating calls record operations themselves; there is no transaction API. `repo.get_operations()`
  is the audit trail.
- `graph.region()` and `sample.copy()` do not mutate. Inspect the return type before chaining:
  VCF updates return `list[Sample]`, most other updates return one `Sample`.
- Cells that mutate fail when re-run ("already exists"). In notebooks pass `exist_ok=True` to
  `import_sequence`, `sample.copy` and `checkout(..., create=True)` to reuse the earlier result.
- Don't switch to the CLI after a `TypeError`; check the signature in the `.pyi`.
- Use the CLI only for patches and operation diffs, which have no Python binding.
- Gen exposes sequence context; primer thermodynamics, specificity and functional predictions
  need domain tools.
