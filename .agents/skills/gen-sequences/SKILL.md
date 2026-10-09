---
name: gen-sequences
description: Use for any task that reads, edits, designs, compares or tracks DNA, RNA or protein sequences in Python - plasmids, genomes, constructs, variants (VCF), annotations (GenBank, GFF, BED), FASTA/GenBank/GFA files, combinatorial or pooled libraries, branches, history, cloning and syncing - or whenever a `.gen` directory exists. Gen (the `gen` Python package) stores sequences as version-controlled graphs, so import, edit, branch, compare and export through Gen. Do NOT export to FASTA and edit with Biopython, sed or string slicing; that loses history, annotations and alternatives. If `import gen` fails, run `python -m pip install gen`.
---

# Gen sequences

## Before anything else

```bash
python -c "import gen" || python -m pip install gen     # same interpreter you will run scripts with
```

Use `python -m pip`, not bare `pip`, so the package lands in the interpreter that runs your
code. The `gen[jupyter]` extra is only for notebooks. If Gen cannot be installed (no network
or no permission), tell the user. Do not reimplement its operations with Biopython, `sed` or
string slicing.

## The model

Gen stores sequences as **graphs**, not strings. Edits are recorded operations on the graph,
so edit through Gen and read the result back as strings. Do not export to FASTA, edit the text
with `sed`/Biopython/string slicing and re-import: that loses history, annotations and
alternatives. Biopython is fine for parsing records you then hand to `repo.import_sequence()`.

A **sequence graph can hold many sequences**: after a library import, `stack=True` edit or
VCF, one graph has many paths. `graph.all_sequences()` returns every distinct path as a `Sequence`
(`str(sequence)` is its bases). It builds the whole list, so avoid it on very large libraries. `locus.sequence` is the string for one region. A `Sample`
is a repository-bound, live view of graphs sharing a sample name and collection. Its list accessors
query current membership; each iteration uses the graphs present when it starts. A `Repository`
holds graph data, collections, branches and history.

Before using a method you have not used, check its real signature. Do not guess method
names, argument names or return types.

1. One method: `help(gen.Repository.update_with_vcf)`.
2. The whole API at a glance: read the installed type stub.
   ```bash
   find "$(python -c 'import gen, pathlib; print(pathlib.Path(gen.__file__).parent)')" -name '*.pyi'
   ```

## Read the reference for your step before you do it

| You are about to | Read first |
|---|---|
| load or save sequence files (FASTA, GenBank, GFA, VCF, GFF/BED, library CSV), or keep a non-sequence file | [references/files.md](references/files.md) |
| edit sequence, apply variants, search, cut out or join pieces, translate, annotate, build a combinatorial library | [references/design.md](references/design.md) |
| branch, merge, undo, inspect history, clone, push or pull | [references/version-control.md](references/version-control.md) |
| show or inspect a graph, or give the user a picture of one | [references/visualize.md](references/visualize.md) |
| run `gen` shell commands, or do something only the CLI offers | [references/cli.md](references/cli.md) |

A typical job crosses several rows: import (files), branch (version control), edit (design),
check (visualize), export (files). Read each reference when you reach that step; they are
short. Do not skip `design.md` because an edit looks simple: coordinate and routing rules
there prevent silent mistakes.

Minimal end-to-end shape:

```python
import gen

repo = gen.Repository("workspace")                          # open or create
sample = repo.import_fasta("plasmid.fa", sample="parent")   # files.md
repo.checkout("swap-promoter", create=True, exist_ok=True)  # version-control.md
design = sample.copy("design")                              # design.md
graph = design[0]
locus = graph.replace("vector:4-8", "ACAC", message="swap motif")
print(graph.plot())                                         # visualize.md
graph.export_fasta("design.fa")                             # files.md (whole sample)
```

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
- `graph.region()` does not mutate. `sample.copy()` records an operation but leaves the original sample untouched. Inspect the return type before chaining:
  VCF updates return `list[Sample]`, most other updates return one `Sample`.
- Cells that mutate fail when re-run ("already exists"). `import_sequence(..., exist_ok=True)` can
  reuse an existing graph only when its sequence matches. `sample.copy()` errors if its destination
  name already exists; `checkout(..., create=True, exist_ok=True)` switches to the existing branch.
- Don't switch to the CLI after a `TypeError`; check the signature with `help()` or the stub.
- Python is the default. Use the CLI only for what Python lacks (`gen diff`, `gen view-diff`,
  patches, `gen transform`, `gen propagate-annotations`), or when the user wants shell
  commands or Python is unavailable. `references/cli.md` maps every CLI command to its Python
  call and has the command recipes. Both work on the same `.gen` workspace.
- Gen exposes sequence context; primer thermodynamics, specificity and functional predictions
  need domain tools.
