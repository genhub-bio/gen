# Designing sequences: edit, locate, cut and join, annotate, libraries

Signatures were checked against `gen-python/src/python_api/` and the stub
`gen-python/python/gen/gen/__init__.pyi`, which has every signature and docstring. Check
`help()` in the actual interpreter when using a different release.

Contents: workflow; copy and edit; loci and positions; search and reading sequence;
annotations; cutting out and joining pieces; translation; combinatorial libraries; mistakes
to avoid.

## Workflow

1. Import the starting sequence (`files.md`) and, on a branch if the change is an experiment
   (`version-control.md`).
2. `sample.copy("design")` so the parent stays intact, then take the graph: `design[0]`.
3. Find the target: `graph.region("vector:4-8")`, `graph.search("GAATTC")`, or an annotation.
4. Edit with `replace`, `insert`, `delete`. Keep the returned `Locus` as the target for the
   next edit; do not recompute coordinates by hand.
5. Read back as strings (`locus.sequence`, `graph.region(...).sequence`,
   `graph.all_sequences()`) and check the result before exporting.

A single base change, a promoter swap and a multi-site design all use the same three edit
methods; only the number of calls differs. Never leave Gen to edit a string.

## Copy and edit

```python
sample.copy(new_name, message=None)
graph.replace(target, sequence, message=None, stack=False)
graph.delete(target, message=None, stack=False)
graph.insert(sequence, before=None, after=None, message=None, stack=False)
```

`copy()` returns a new `Sample` in the same collection, retaining the parent's
graphs and annotations for independent editing; an existing sample name fails. A `Sample` is a
live view: its list accessors query current membership, and a new iteration sees graphs added since
the handle was created. Replace/delete targets are region strings, loci, or annotations. These methods
edit the graph's existing sample, not a new sample, so copy first when the original matters.
Replace and insert return the new sequence's `Locus`; delete returns `None`. Each edit
records an operation, using `message` when supplied. Replacement/insertion require nonempty
sequence; use delete to remove sequence.

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

Which method for which intent:

| Intent | Call |
|---|---|
| change bases in a span | `graph.replace(region_or_locus, "NEW")` |
| remove a span | `graph.delete(region_or_locus)` |
| add bases between two bases | `graph.insert("NEW", after=position)` or `before=position` |
| replace a named feature | `graph.replace(annotation, "NEW")` |
| keep the original and add an option | any of the above with `stack=True` |
| apply many changes from a file | `update_with_*` in `files.md` |

`before` and `after` are keyword-only and each accepts `Position` or `SuperPosition`; pass
exactly one (both raise `TypeError`). The insertion connects to all current neighbors on the
other side. For example, at a fork, insertion after the shared upstream base affects all
outgoing alternatives, while insertion before one downstream option affects only that option.
Combine positions with `|` to choose several routes at once. One call adds one shared piece;
use separate calls to keep independent routes separate.

Every route that reaches a node coordinate reads every route that leaves it. An insertion
that would split such a shared coordinate, such as before the first or after the last base of
a stacked edit's original sequence, raises `ValueError` instead of landing on the alternative
too. Replace and delete targets need only be present and connected, including across
zero-width routing blocks; they need not lie on one path.

Edits read sequence on the target strand, so replacement on a reverse locus uses the
reverse-oriented sequence. Insertion after a reverse position follows that strand's reading
direction. Targets must remain present in the edited graph; canonical locus identity survives
node splitting but does not make deleted bases valid targets. Re-query current sequence when
necessary.

`stack=True` retains original routes and adds the edit as an alternative. It leaves the
current named path unchanged. Default edits supersede targeted routes and update affected
current paths. Use all-path reads/exports to inspect stacked alternatives; default FASTA
alone will still show the original path. Use `stack=True` when the user wants to compare
variants or pool options in one graph, and the default when they want the edit to become the
sequence.

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

Region coordinates are 0-based and half-open along a named graph/path or annotation.
`graph.region()` rejects regions resolved to another graph, ambiguous routes, empty
intervals, and out-of-range spans. Graph-name regions use the current path when available.
Locus offsets count bases in reading order, including across nodes and on the reverse
strand. `start()`/`end()` are the first/last **included base**, not interval boundary
coordinates. `graph.locus(start, end)` takes 0-based half-open path coordinates.

Integer indexing returns a `Position`; unit-step slicing returns a `Locus`. Python slicing
supports omitted/negative/clipped bounds; explicit `slice(start, end)` requires an in-range
nonempty span. Empty slices raise `IndexError` and steps other than one fail. Reverse
complement changes strand and reading order; use `reverse_complement()`, not `[::-1]`.

A `Node` is a stretch of stored sequence: `.sequence` (its bases) and `.length`.
`Position.offset` is relative to its node, not a path coordinate. Use typed endpoints rather
than reconstructing targets from offsets.

Positions from search, regions, edits, and annotation loci carry graph context. Arithmetic
walks the current graph in reading order; a fork can turn a `Position` into `SuperPosition`.
Superpositions retain alternative endpoints; stepping beyond a terminal fails. `.on(graph)`
attaches the same address to another graph (for example a sample copy); it does not
translate path coordinates. Unions reject attached positions from different graphs.

## Search and reading sequence

```python
graph.search(query, sequence_kind="dna")
repo.search(query, sgs=None, sequence_kind="dna")
graph.build_index(sequence_kind="dna", k=16)
repo.build_index(sequence_kind="dna", k=16)         # every sequence graph
graph.clear_index()
repo.clear_index(sgs=None)
graph.all_sequences()
graph.to_dict()
graph.to_networkx()
graph.to_rustworkx()
```

Graph search returns `list[Locus]`; repository search returns
`list[(SequenceGraph, list[Locus])]`. `sgs` is a list of graph objects to restrict the search to. Kinds: `"exact"`,
`"dna"`, `"ssdna"`, `"protein"`. Exact is case-sensitive raw-byte matching without IUPAC
expansion or reverse-complement search; DNA matches both strands with IUPAC support. Check
result count and strand before choosing a target; a restriction site such as `GAATTC` can
match more than once. Build an index for repeated large-graph searches.

`locus.sequence` is a string in reading order; `str(locus)` also reads sequence.
`all_sequences()` is a list of distinct `Sequence` objects, one per different spelling of the
graph's paths, sorted by bases (`str(sequence)` is the bases; they also compare, hash and slice
like that string). It is built in full, and combinatorial enumeration can be exponential. `node.sequence` reads a `Node`'s bases.
Repository and graph handles cannot cross threads; open a repository in the worker and
retrieve the graph by its typed id instead. NetworkX/rustworkx conversions need their
optional packages and return `DiGraph`/`PyDiGraph`, respectively.

## Annotations

```python
graph.annotations
annotation = graph.add_annotation(locus, "motif", track="motifs")
ad_hoc = gen.Annotation(locus, "temporary label")
```

`graph.annotations` returns database and attached-file annotations together.
`add_annotation()` requires a nonempty `Locus` and returns a persisted `Annotation`,
recording an operation. `gen.Annotation(locus, name)` constructs an ad-hoc annotation, not a
persisted feature; it is useful as a temporary edit target or highlight. Annotations from
GFF, BED or GenBank files are loaded with `repo.import_annotations()` (`files.md`).

Annotation fields: `.name`, `.id` (`HashId`), `.locus`, `.track`, `.metadata`. Annotations
hash and compare by id and location. Use `.locus.sequence` to inspect feature sequence, and
the annotation or its locus as a direct edit target:

```python
promoter = next(a for a in graph.annotations if a.name == "promoter")
print(promoter.locus.sequence)
graph.replace(promoter, "TTGACA")        # edit the feature by name, not by coordinates
```

## Cutting out and joining pieces

```python
graph.subgraph(new_sample, locus.start(), locus.end())   # two Positions
graph.subgraph(new_sample, start, end)                    # integer path coordinates
graph.chunks(new_sample, breakpoints=None, chunk_size=None)
repo.stitch(parts, new_sample, new_region)
```

`graph.subgraph()` takes two forward-strand `Position`s (the first and last base to include,
so for a `Locus` pass `locus.start()` and `locus.end()`; the locus itself is rejected) or two
integer path coordinates (0-based, end exclusive), keeps every variant route between the ends, and returns a `SequenceGraph` in
`new_sample` (with a current path when this graph's path runs from start to end);
`graph.chunks()` returns `list[SequenceGraph]`. Breakpoints are a list of integer
coordinates. Use `subgraph` to extract a part (an insert, a cassette) and `chunks` to split
a long sequence into pieces of fixed size or at chosen breakpoints.

`repo.stitch(parts, new_sample, new_region)` joins `SequenceGraph` and `Locus` parts end to
end into a new `SequenceGraph`, e.g. `repo.stitch([a.region("a:0-100"), b], "asm",
"construct")`. A `Locus` is linear, but the subgraph of **every variant** between its first
and last positions is stitched, so no subgraph needs deriving first. Parts must share a
collection, be forward-strand (reverse loci are rejected) and not overlap, since that would
make the result cyclic. A `SequenceGraph` part must have a current path; a subgraph taken
between positions off the current path may not, so stitch a `Locus` of it instead (a `Locus`
part needs no path).

```python
construct = repo.stitch(
    [vector.region("vector:0-100"), insert, vector.region("vector:150-400")],
    new_sample="assembly", new_region="construct",
)
```

## Translation

```python
graph.translate_annotation(region=None, output_collection=None, name=None,
                           strand=None, frame=0, codon_table=1, start=None)
```

Translation returns a protein `SequenceGraph` in the source sample. A string region resolves
first as a path, then an annotation in this graph; passing an `Annotation` disambiguates by
id. Translation reads from `start` (or the feature's entry point) until the first in-frame
stop; an end coordinate does not bound it.

## Combinatorial libraries

```python
repo.import_library(library_name, parts_list, sample=None, collection=None)
repo.import_library_files(library_name, parts, library, sample=None, collection=None)
repo.update_with_library(sample, new_sample_name, path_name, parts_list, collection=None)
repo.update_with_library_files(sample, new_sample, path_name, library, parts, collection=None)
```

Library `parts_list` is a list of columns, each containing alternative named
`gen.Sequence(name, sequence)` objects. Single-option columns are fixed flanks. Every path
through the resulting graph is one assembled design.

```python
parts_list = [
    [gen.Sequence("left", "AAAA")],
    [gen.Sequence("option-a", "CCCC"), gen.Sequence("option-b", "GGGG")],
    [gen.Sequence("right", "TTTT")],
]
library = repo.import_library("cassette", parts_list, sample="library")
assert {str(s) for s in library.all_sequences()} == {"AAAACCCCTTTT", "AAAAGGGGTTTT"}
```

For file libraries, `parts` is a named-parts FASTA and `library` is a headerless CSV with one
slot per column and alternatives in rows. To replace an existing region with a library use
`update_with_library()`/`update_with_library_files()`; `import_library*` starts a new graph,
`update_with_library*` splices into an existing sample's region and stores the result as a new
sample. The number of designs is the product of the column sizes, so check it before
enumerating `all_sequences()`.

## Mistakes to avoid

- Editing a string copy of the sequence and importing it back as a new sample. Edit the graph.
- Recomputing coordinates after an edit by hand. Use the `Locus` the edit returned, or
  re-query with `graph.region()`/`graph.search()`.
- Passing both `before=` and `after=`, or inserting at the end points of a stacked edit
  (`ValueError`); replace the span between positions instead.
- Forgetting `sample.copy()` and editing the parent sample in place.
- Treating `locus.end()` as an exclusive coordinate. It is the last included base.
- Enumerating `all_sequences()` on a large library without checking the combination count.
