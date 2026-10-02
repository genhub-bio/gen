---
name: gen-genetic-engineering-python
description: Use the Gen Python bindings (import gen) for sequence engineering, graph queries, annotations, direct edits, combinatorial libraries, repository history and remote sync, and verification in scripts or agent REPLs.
---

# Gen Genetic Engineering (Python)

## Approach and source of truth

Use `import gen` and compose operations on returned `Sample`, `SequenceGraph`,
`Locus`, and `Position` objects. Prefer Python for agent workflows, including
branching and remote sync. Use the CLI when requested or for functionality not
bound in Python, such as patches and operation diffs.

Read [references/gen-python-workflows.md](references/gen-python-workflows.md) for
signatures, return types, and recipes. These describe the current repository API;
an installed release may differ. Check `help(gen.Repository)`,
`help(gen.SequenceGraph)`, and `help(gen.Sample)` in the interpreter actually used.
Inside the repository, verify against `gen-python/src/python_api/`,
`gen-python/python/gen/`, and the focused examples and tests in `gen-python/`.
Do not guess arguments or silently switch to the CLI after a `TypeError`.

## Choose the workflow

Establish the workspace, collection, source sample, input artifacts, and intended
change before writing. Open an existing workspace or create one with:

```python
import gen

repo = gen.Repository("path/to/workspace")
samples = repo.samples
```

- Import FASTA or GenBank to get a `Sample`. Import a string, Biopython `Seq` or
  `SeqRecord` with `repo.import_sequence()` to get one `SequenceGraph`. GFA and
  library imports also return a `SequenceGraph` directly.
- For a series of direct edits, copy the source with `sample.copy("design")`,
  choose a graph from the copy, and use `graph.replace()`, `graph.delete()`, and
  `graph.insert()`. These mutate that graph's sample; they do not create a sample.
- Use `repo.update_with_*()` for file-driven updates, variant application, and
  region/library workflows. Inspect the return type: VCF/GAF return lists of
  samples, while most other updates return one sample.
- Use `repo.checkout("design", create=True)` to isolate repository history.
  A Gen branch and a copied biological sample serve different purposes; choose
  either or both according to the requested design workflow.
- Query `graph.annotations`, persist a feature with `graph.add_annotation()`, or
  attach an annotation file with `repo.import_annotations()`.
- Verify sequence with `locus.sequence`, `graph.region()`, search, and exports.
  Use `graph.all_sequences()` when all graph alternatives matter. Plotting gives
  a text widget in terminals/agents and an interactive widget in a live Jupyter
  kernel with the optional dependencies.

Mutating APIs record operations themselves. There is no `repo.transaction()`
context manager. `repo.get_operations()` provides the audit trail.

## Coordinates and editing invariants

Region strings are `"<name>:<start>-<end>"`, with 0-based, half-open coordinates
along a named graph/path or annotation. `graph.region()` resolves a read-only
`Locus`; graph-name coordinates follow the current path when one exists.

A `Locus` reads in strand order: `locus[0]` and `locus[-1]` are positions,
`locus[2:5]` is a locus, and `locus.sequence` is its sequence on that strand.
`start()` and `end()` identify the first and **last included base**; `end()` is
not the exclusive interval boundary. Slice offsets count bases along the locus,
not stored node or graph coordinates. Empty slices and non-unit steps fail.

`Node.sequence_start` / `sequence_end` slice the stored sequence represented by
that graph node. `Position.offset` is relative to that node slice. Neither is a
coordinate along the graph's path. Keep loci/positions as targets instead of
reconstructing targets from display offsets after edits split nodes.

Replacement and deletion accept a region string, `Locus`, or `Annotation`.
Replacement sequence is read on the target strand. Insertion needs keyword
`before=` or `after=` positions, not both. Use replacement to change a span between
nonadjacent positions.

Position arithmetic can return `SuperPosition` at forks. Combine endpoints with
`position_a | position_b` or `gen.SuperPosition(...)` only when one shared insert
should connect all specified alternatives. Separate insert calls preserve
separate routes. `stack=True` adds an alternative while retaining the original
routes and current path; ordinary edits supersede the targeted routes. An edit applies
to every route through its coordinates, so inserting before the first or after the last
base of a stacked edit's original sequence raises `ValueError`; insert inside it or at
the alternative's ends.

## Verify and export

For exact assertions use `sequence_kind="exact"`; default `"dna"` search supports
IUPAC matching and reverse complements. Confirm hit count and strand before editing.

```python
parent = repo.import_sequence("AAAACCCCGGGGTTTT", name="vector", sample="parent")
source = next(sample for sample in repo.samples if sample.sample_name == "parent")
design = source.copy("design")
graph = design[0]
replacement = graph.replace("vector:4-8", "ACAC", message="replace motif")
assert replacement.sequence == "ACAC"
assert parent.region("vector:4-8").sequence == "CCCC"
widget = graph.plot()
widget.show(replacement)
print(repr(widget))
graph.export_fasta("design.fa")
```

Default FASTA export writes current paths; use `all_sequences=True` to export all
alternatives. Graph-level exports cover the **whole sample**, not just that graph.
Enumeration of combinatorial paths can be large; consume iterators selectively.
Call `widget.refresh()` after edits before relying on an existing plot.

For ORFs, frames, and peptide changes, use `graph.translate_annotation()` and
inspect the returned protein graph. Inspect annotations and junction sequences
after edits. Gen manages and exposes sequence context; functional predictions and
final primer thermodynamics, specificity, and assembly constraints need the
appropriate domain tools.
