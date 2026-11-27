---
name: gen-genetic-engineering-python
description: Help users apply the Gen Python bindings (`import gen`) to genetic engineering workflows, including importing FASTA/GenBank/GFA genomes and plasmids, editing sequences via VCF/GAF/library/explicit-string updates, designing combinatorial libraries, deriving subgraphs and chunks, searching and navigating graphs, translating coding regions to protein, exporting synthesis or cloning artifacts, and verifying results programmatically without a browser.
---

# Gen Genetic Engineering (Python)

## Core Approach

Write Python against the `gen` module directly. `Repository` methods return live
`Sample` / `SequenceGraph` objects — never bare ids or strings — so chain calls and
inspect objects instead of parsing text output. This is the primary way to drive Gen
as an agent: it lets you compose multi-step workflows, hold state as objects, branch
on results, and verify outcomes by asserting on returned objects instead of scraping
stdout.

The `gen` CLI exists too (`gen import fasta`, `gen update vcf`, ...) and is a thin
wrapper over the same engine for everything the Python bindings also cover. Reach for
it only when the user explicitly wants shell commands, or for the operations below
that the Python bindings don't expose yet. Otherwise, write Python.

### CLI-only operations

The Python bindings cover import/update/export/query/search/derive/translate, but not
version control or remote sync. These have no Python equivalent — use the CLI:

- **History & branching**: `gen branch [--create|--checkout|--delete|--list|--merge] <name>`,
  `gen merge <branch>`, `gen checkout <hash>`, `gen reset <hash>`, `gen operations`,
  `gen apply <hash>`.
- **Remote sync**: `gen clone <url>`, `gen push [--remote <name>]`,
  `gen pull [--remote <name>]`, `gen remote add|list|remove|set-default <name> [<url>]`.
- **Patches**: `gen patch-create -n <name> <operation-range>`, `gen patch-apply <file>`,
  `gen patch-view <file>`, `gen diff --sample1 <a> --sample2 <b> --gfa <out>`,
  `gen view-diff <from> [<to>]`.
- **Config**: `gen defaults --database <db> --collection <name>`.
- **Annotation attachment without importing intervals**: `gen add-annotation -n <name> -s <sample> <region>`,
  `gen add-annotation-file <path> [--format gff3|bed|genbank]`,
  `gen propagate-annotations --from-sample <a> --to-sample <b> --gff <in> --output-gff <out>`.
- **Misc**: `gen add-file <files...>` (attach files as an operation),
  `gen transform --format-csv-for-gaf <file>`, `gen translate --bed|--gff <file> --sample <s>`
  (coordinate translation into graph space), `gen add-reference-aliases --reference-name <n> ...`,
  `gen view [<graph>] [--full]` (interactive TUI explorer — `sg.plot()` covers the
  scriptable/headless case, but not the full-screen sidebar explorer).

Everything else in this skill (import, update, export, search, derive, translate,
combinatorial libraries) should be done through Python, not these CLI commands.

The method signatures below were captured directly from the installed `gen` module
(`help(gen.Repository)`, etc.) — they reflect the actual bound API, not the CLI help
text. If a call raises `TypeError` about arguments, re-check with `help(gen.<Class>)`
rather than guessing.

Start by identifying:

- The workspace: an existing `.gen` directory to open, or a fresh path to open (Gen
  initializes it automatically) — see [Setup](#setup).
- The input artifacts: FASTA, GenBank, GFA, VCF, GAF, a parts list / library CSV, or
  raw sequence strings.
- The desired biological change: replacement, insertion, deletion, variant
  application, combinatorial library expansion, subgraph extraction, or
  impact/translation analysis.
- The coordinate system: region strings are `"<path or annotation name>:<start>-<end>"`,
  0-based and half-open along that named path — not GraphNode `sequence_start`/
  `sequence_end`, which are slice offsets into a stored `Node`'s sequence, not graph
  coordinates.
- The risk level: `gen` produces and inspects sequences, but primer thermodynamics,
  vendor constraints, off-target analysis, assembly overhang rules, and
  regulatory/safety review need domain tools or user confirmation — see
  [Guardrails](#genetic-engineering-guardrails).

## Setup

```python
import gen

repo = gen.Repository("path/to/.gen")   # opens (or creates) a workspace
```

Batch multiple operations into one transaction with `repo.transaction()`:

```python
with repo.transaction():
    repo.import_fasta("reference.fasta")
    repo.import_gfa("graph.gfa")
```

For detailed method signatures and worked recipes for every import/update/export
path, read `references/gen-python-workflows.md`.

## The Object Model

| Type | What it is | Key members |
|---|---|---|
| `Repository` | Workspace handle; owns import/update/export/query methods | see reference doc |
| `Sample` | All sequence graphs produced by one import/update/derive call | list-like: index, iterate, `len()`; `.sample_name`, `.block_groups`, `.plot()` |
| `SequenceGraph` | One graph within a sample (a "block group") | `.name`, `.id`, `.sample_name`, `.collection_name`, `.plot()`, `.search()`, `.list_annotations()`, `.to_dict()`/`.to_networkx()`/`.to_rustworkx()` |
| `Annotation` | A stored or ad-hoc feature (gene, promoter, MCS, ...) | `.name`, `.id`, `.locus`, `.group`, `.track`, `.metadata`, `.segments` |
| `Locus` | A region within a graph (search hit, annotation span) | `.start()`/`.end()` → `Position`, `.slices` → `list[NodeSlice]`, `.strand` |
| `Position` | A single point in graph space | `.node`, `.offset` |
| `NodeSlice` | Part of a stored `Node` used by a graph | `.node`, `.start`, `.end`, `.strand` |
| `SequencePart` | A named sequence used to build a combinatorial library column | `.name`, `.sequence` |
| `GraphWidget` | Interactive plot returned by `.plot()`; also your headless verification tool | see [Verifying without a browser](#verifying-without-a-browser) |

`Repository` import/update methods generally return a `Sample`; `SequenceGraph`-level
convenience methods (`subgraph`, `chunks`) return `SequenceGraph` / `list[SequenceGraph]`
directly. Read the return type in `references/gen-python-workflows.md` before writing
code that indexes into the result.

## Workflow

1. **Open the workspace** — `gen.Repository(path)`; use `repo.transaction()` to batch
   related writes.

2. **Import biological context**:
   - `repo.import_fasta(path, sample=...)` for raw sequence references or samples.
   - `repo.import_genbank(path, sample=...)` when features/annotations matter — they
     load automatically and are queryable via `sg.list_annotations()`.
   - `repo.import_gfa(path, sample=...)` when the graph itself is the source artifact.
   - `repo.import_reference_fasta(path, reference=...)` to mark a sample as a
     reference (distinct from a regular `sample=`).
   - `repo.import_library(name, parts_list, sample=...)` or
     `repo.import_library_files(name, parts, library, sample=...)` for combinatorial
     designs — see [Combinatorial libraries](#combinatorial-libraries).

3. **Apply edits, branch-friendly** — every update takes the source `sample` and a
   `new_sample` (or `new_sample_name`); the new sample shares the whole graph with the
   parent except where the path diverges through the edit:
   - `repo.update_with_sequence(seq, sample=, new_sample=, region_name=...)` — replace
     a region with an explicit string.
   - `repo.update_with_fasta(path, sample=, new_sample=, region_name=...)`.
   - `repo.update_with_genbank(path, sample=, create_missing=False)`.
   - `repo.update_with_vcf(path, reference=, sample=, genotype=None, in_place=False)`.
   - `repo.update_with_gaf(path, csv=, sample=, parent_sample=...)`.
   - `repo.update_with_gfa(path, sample=, new_sample=...)`.
   - `repo.update_with_library(sample=, new_sample_name=, path_name=, parts_list=...)`
     or `update_with_library_files(..., library=, parts=)` — replace a region with a
     combinatorial set of variants.

4. **Inspect and verify** — prefer asserting on returned objects over printing text:
   - `repo.get_samples()`, `repo.get_sequence_graphs()`,
     `repo.get_sequence_graphs_by_collection(name)`.
   - `sg.list_annotations()`, `sg.get_node_sequence(node)`, `sg.to_dict()`.
   - `repo.search(query, bgs=[...], sequence_kind="dna")` /
     `sg.search(query, sequence_kind="dna")` — returns loci with `.start()`, `.end()`,
     `.slices`, `.strand`; use these to confirm restriction sites landed, junctions
     are clean, etc.
   - `sg.plot()` / `sample.plot()` for a `GraphWidget` you can drive and `repr()`
     headlessly — see below.

5. **Export for downstream tools**:
   - `repo.export_fasta(path, sample=...)` for synthesis, primer design, alignment.
   - `repo.export_genbank(path, sample=...)` to preserve annotations for editors or
     vendors.
   - `repo.export_gfa(path, sample=..., node_max=...)` for graph-aware tools.

## Region / path syntax

Region strings used by `region_name=`, `region=`, `path_name=` arguments are
`"<path or annotation name>:<start>-<end>"`, 0-based and half-open along that named
path (e.g. `"pUC19:395-452"`, `"m123:5-15"`). The name resolves first against a named
path in the graph, then against an annotation name. This is unrelated to
`NodeSlice`/`GraphNode` `sequence_start`/`sequence_end`, which slice a stored `Node`'s
sequence and are not graph coordinates.

## Combinatorial libraries

`import_library()` / `update_with_library()` take a `parts_list`: a list of columns,
each column a list of `gen.SequencePart(name, sequence)` alternatives. The resulting
graph has one path per combination:

```python
parts_list = [
    [gen.SequencePart("upstream", "AATTCGGATCCAAGCTT")],
    [
        gen.SequencePart("pTrc", "TTGACAATTAATCATCCGGCTCGTATAATGTGTGG"),
        gen.SequencePart("pLac", "AATTGTGAGCGGATAACAATT"),
    ],
    [
        gen.SequencePart("gfp", "ATGAGTAAAGGAGAAGAACTTTTCACTGG"),
        gen.SequencePart("rfp", "ATGGCTTCCTCCGAAGACGTTATCAAAGAG"),
    ],
]
cassette_sg = repo.import_library("expression-cassette", parts_list)
```

Single-option columns act as fixed flanking sequence. To swap a region of an
*existing* sample with a library instead of building a fresh graph, use
`update_with_library(sample=, new_sample_name=, path_name=<region>, parts_list=...)`.

## Verifying without a browser

`sample.plot()` / `sg.plot()` return a `GraphWidget` you can drive and inspect from
plain Python — no browser or JS required:

```python
widget = sg.plot()
print(repr(widget))     # ASCII snapshot of current state

widget.go_to(locus)               # jump to a Locus/Position/Annotation
widget.next_page() / .prev_page() # switch sequence graphs within a Sample-backed widget
widget.zoom_in() / .zoom_out()
widget.scroll_left()/.scroll_right()/.scroll_up()/.scroll_down()
print(repr(widget))     # re-check after mutating
```

`next_page`/`prev_page` switch *which graph* is shown (only meaningful on a
`sample.plot()` widget); `scroll_*` pans the viewport within the current graph. Don't
confuse the two. All of the above mutate the widget in place.

## Guardrails

Do not pretend `gen` alone predicts functional impact. For impact predictions,
combine `gen` outputs with explicit biological checks:

- Use `sg.translate_annotation(region=..., frame=..., codon_table=...)` when ORFs,
  frames, start/stop codons, or peptide changes matter — it returns a protein
  `SequenceGraph`, so inspect its sequence rather than eyeballing codons by hand.
- Inspect annotations after GenBank imports/updates (`sg.list_annotations()`) and
  mention whether feature coordinates may need propagation or manual review.
- Check junction sequences for cloning scars, restriction sites, homology arms,
  overhangs, or unwanted motifs — use `repo.search()`/`sg.search()` rather than
  visual inspection.
- For primer ordering, derive candidate binding regions from exported or extracted
  sequence, then state plainly that final primer Tm, secondary structure, dimers,
  off-targets, vendor limits, and assembly chemistry must be checked with appropriate
  primer-design tools — Gen does not do this.

Avoid giving operational assistance for unsafe or disallowed biological engineering.
If a request involves pathogenicity, toxin expression, evading detection, or harmful
organism engineering, refuse that portion and offer benign sequence-management help
instead.
