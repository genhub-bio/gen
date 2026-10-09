# Gen CLI (fallback)

Use this reference when the task is a command someone will type in a terminal, or when the
agent cannot or will not use Python. Python is the default for scripted and multi-step work
(see `SKILL.md`): it keeps loci and positions between steps, while the CLI makes you pass
coordinates by hand. Both operate on the same `.gen` workspace.

## What Python already covers

| CLI | Python |
|---|---|
| `gen init`, `gen clone` | `gen.Repository(path)`, `gen.clone()` |
| `gen import ...` | `repo.import_fasta/genbank/gfa/sequence/library*` |
| `gen update ...` | `repo.update_with_*`, `graph.replace/insert/delete` |
| `gen export ...` | `graph.export_fasta/genbank/gfa` |
| `gen derive subgraph` / `chunks` | `graph.subgraph()` / `graph.chunks()` |
| `gen make-stitch` | `repo.stitch()` |
| `gen add-annotation` / `add-annotation-file` | `graph.add_annotation()` / `repo.import_annotations()` |
| `gen add-reference-aliases` | `repo.add_reference_alias()` |
| `gen search`, `build-index`, `clear-index` | `graph.search()`, `repo.build_index()`, `clear_index()` |
| `gen view` | `graph.plot()` |
| `gen branch`, `checkout`, `merge`, `reset`, `apply`, `operations` | `repo.create_branch/checkout/merge/reset/apply`, `get_operations()` |
| `gen remote`, `push`, `pull`, `fetch` | `repo.add_remote/push/pull/fetch` |
| `gen translate --bed/--gff` (coordinates) | annotation import resolves automatically; `position.on(graph)` for explicit moves |

## CLI-only operations

| Operation | Command |
|---|---|
| compare two revisions or samples | `gen diff`, `gen view-diff` |
| create, apply or inspect a patch | `gen patch-create`, `gen patch-apply`, `gen patch-view` |
| turn a GAF replacement CSV into FASTA | `gen transform --format-csv-for-gaf` |
| write an annotation file in another sample's coordinates | `gen propagate-annotations` |

Python resolves relative file paths against the current working directory. The CLI's handling
of relative paths may differ, so use absolute paths when mixing the two.

Run `gen <command> --help` before presenting a command sequence.

## Core approach

Use this reference to turn a biological engineering intent into concrete `gen` commands, validation steps, and caveats.

Start by identifying:

- The workspace: whether `gen init` has run, which database and collection are active, and which sample is the reference or parent.
- The input artifacts: FASTA, GenBank, GFA, VCF, GAF, CSV library design, parts FASTA, annotations, or raw sequence strings.
- The desired biological change: replacement, insertion, deletion, variant application, library slot expansion, clone-ready export, or impact analysis.
- The coordinate system: Gen update regions use path/accession/annotation-style region names such as `chr1:2-5`; do not confuse GraphNode `sequence_start`/`sequence_end` with graph coordinates.
- The risk level: for wet-lab ordering, be explicit that `gen` can produce and inspect sequences, but primer thermodynamics, vendor constraints, off-target analysis, assembly overhang rules, and regulatory/safety review need domain tools or user confirmation.

## Source of truth

When working inside the Gen repository, inspect these files before giving precise syntax:

- `src/commands/mod.rs` for top-level commands.
- `src/commands/import/*.rs`, `src/commands/update/*.rs`, and `src/commands/export/*.rs` for argument names.
- `src/lib.rs` for the public Rust facade and reexports.
- `docs/commands.md`, `examples/yeast_editing/`, `examples/externally_edited_files/`, and `examples/combinatorial_plasmid_design/` for workflows.

If `gen` is installed in the user's environment, verify command syntax with `gen --help` and `gen <subcommand> --help` before presenting a final command sequence. The source currently uses subcommand syntax such as `gen import fasta ...`, even if older examples show option-style forms.

## Workflow

1. Establish repository defaults:
   - Use `gen init` when there is no `.gen` directory.
   - Use `gen defaults --collection <collection>` to avoid repeating `--name`.
   - Use `gen operations` before and after meaningful edits so the user can audit changes.

2. Import biological context:
   - Use FASTA for raw sequence references or samples.
   - Use GenBank when features and annotations matter.
   - Use GFA when the graph itself is the source artifact.
   - Use `--reference <name>` for a reference sample or `--sample <name>` for a regular sample.

3. Apply edits in a branch-friendly way:
   - Create a branch for exploratory edits with `gen branch --create <name>` then `gen branch --checkout <name>`.
   - Use `gen update sequence`, `gen update fasta`, `gen update genbank`, `gen update vcf`, `gen update gaf`, `gen update gfa`, or `gen update library` depending on the artifact.
   - Always provide a meaningful `--new-sample` for non-in-place designed outcomes where the command supports it.

4. Inspect and summarize:
   - Use `gen list-samples`, `gen list-graphs`, `gen get-sequence`, `gen search`, `gen view`, `gen diff`, `gen view-diff`, and `gen operations`.
   - Summarize edits biologically: changed coordinates, inserted/deleted/replaced sequence, affected annotations/features, sample lineage, and exported artifacts.

5. Export for downstream tools:
   - Export FASTA for synthesis, primer design, alignment, or simple validation.
   - Export GenBank when preserving annotations for editors or vendors.
   - Export GFA for graph-aware mapping or visualization.

## Biological checks

`gen` does not predict functional impact by itself. For impact questions, combine `gen` outputs with explicit biological checks:

- Translate coding sequences when ORFs, frames, start/stop codons, or peptide changes matter.
- Inspect annotations after GenBank imports/updates and mention whether feature coordinates may need propagation or manual review.
- Check junction sequences for cloning scars, restriction sites, homology arms, overhangs, or unwanted motifs.
- For primer ordering, derive candidate binding regions from exported or extracted sequence, then state that final primer Tm, secondary structure, dimers, off-targets, vendor limits, and assembly chemistry must be checked with appropriate primer-design tools.


## Current CLI Shape

Top-level pattern:

```sh
gen <command> ...
```

There is no global `--db` flag; the database location is resolved from the
`.gen` workspace directory discovered by walking up from the current
directory (created by `gen init`).

Common setup:

```sh
gen init
gen defaults --collection plasmids
gen operations
```

Source files to verify syntax in-repo:

- Top-level commands: `src/commands/mod.rs`
- Imports: `src/commands/import/*.rs`
- Updates: `src/commands/update/*.rs`
- Exports: `src/commands/export/*.rs`

## Importing Genomes, Plasmids, And Libraries

Use one of `--sample` or `--reference` on imports. A reference sample is appropriate for a starting genome, strain, chromosome set, or vector backbone that downstream samples derive from.

```sh
gen import fasta reference.fa --reference ref
gen import fasta construct.fa --sample design-a
gen import genbank plasmid.gb --reference pbackbone
gen import genbank annotated.gb --sample clone-1 --annotation-group clone-1-features
gen import genbank annotated.gb --sample clone-1 --no-annotations
gen import gfa library.gfa --sample pooled-library
gen import library promoter-rbs-library parts.fa design.csv --sample library-design
```

Useful flags:

- `-n, --name <collection>`: override the default collection.
- `--shallow` on FASTA import: store filename instead of sequence.
- `--index <path>` on FASTA import (requires `--shallow`, repeatable): attach an index file for the shallow-referenced FASTA asset.
- `--annotation-group <name>` on GenBank import: control imported annotation group name.

## Listing, Viewing, Searching, And Extracting

```sh
gen list-samples
gen list-graphs --sample ref
gen view <graph-name> --sample ref
gen view <graph-name> --sample ref --full
gen get-sequence --sample ref --graph chr1 --start 100 --end 160
gen get-sequence --sample ref --region chr1:100-160
gen search ATGCGTACGTAG --sample ref
gen build-index --sample ref --kmer-size 16
gen clear-index --sample ref
```

Use `get-sequence` before primer or junction design. Use `search` to check whether a proposed primer binding sequence or inserted part is present in the current sample.

## Updating Sequences

Prefer branch-per-design for exploratory work:

```sh
gen branch --create design-a
gen branch --checkout design-a
```

Explicit sequence replacement or insertion:

```sh
gen update sequence ATCGATCG --sample ref --new-sample edited --region-name chr1:3-5
gen update fasta insert.fa --sample ref --new-sample edited --region-name chr1:3-5
```

For pure insertion, use an empty or zero-length interval only if the underlying region parser and command help confirm support. Otherwise state the intended replacement interval explicitly.

GenBank round trip from an external editor:

```sh
gen update genbank edited.gb --sample ref
gen update genbank edited.gb --sample ref --create-missing
```

Variant application:

```sh
gen update vcf variants.vcf --sample sample-a --genotype 0/1
gen update vcf variants.vcf --parent-samples ref --sample sample-a
gen update vcf variants.vcf --parent-samples ref,sample-a --inplace
```

Graph/alignment updates:

```sh
gen update gfa edited.gfa --sample ref --new-sample edited
gen update gaf alignments.gaf --csv edits.csv --sample edited --parent-sample ref
gen transform --format-csv-for-gaf edits.csv > edits.fa
```

## Combinatorial Design

Use `gen update library` when replacing a locus in a backbone with all combinations of parts.

Parts FASTA:

```fa
>promoter_A
TTGACA...
>rbs_B
AGGAGG...
>payload
ATG...
```

Library CSV has no header. Each column is a slot; each non-empty cell is an option for that slot. Empty cells still need commas.

```csv
promoter_A,rbs_A,payload
promoter_B,rbs_B,
promoter_C,,
```

Apply to a region:

```sh
gen update library \
  --sample backbone \
  --new-sample library \
  --region-name vector:106-539 \
  --library design.csv \
  --parts parts.fa
```

Export to GFA for graph-aware mapping or visualization:

```sh
gen export gfa library.gfa --sample library
```

## Summarizing Changes

Useful audit commands:

```sh
gen operations
gen view-diff <from-ref> <to-ref>
gen diff --sample1 ref --sample2 edited --gfa diff.gfa
gen patch-create --name design-a.patch HEAD~1..HEAD
gen patch-view design-a.patch
```

When summarizing, include:

- Database, collection, branch, parent sample, and new sample.
- Operation hashes or refs used for comparison.
- Regions changed and whether they were replaced, inserted, deleted, or imported.
- Feature or annotation implications if GenBank data is present.
- Files exported for downstream design or ordering.

## Exporting For Synthesis, Cloning, Or Editors

```sh
gen export fasta edited.fa --sample edited
gen export genbank edited.gb --sample edited
gen export gfa edited.gfa --sample edited
gen export gfa edited.gfa --sample edited --node-max 5000
```

For vendor or cloning workflows, prefer GenBank when annotations communicate part boundaries, resistance markers, origins, CDSs, or homology arms. Prefer FASTA for raw synthesis sequences or tools that do not need annotations.

## Primer And Synthesis Support

Gen does not design primers by itself. Use it to produce exact template, insert, junction, and variant context:

```sh
gen get-sequence --sample edited --region vector:80-140
gen get-sequence --sample edited --region vector:520-580
gen search CANDIDATE_PRIMER_SEQUENCE --sample edited
gen export fasta edited.fa --sample edited
gen export genbank edited.gb --sample edited
```

For primer-ordering help:

1. Ask for cloning method, vendor/ordering constraints, desired overlaps or overhangs, and validation target.
2. Extract 40-120 bp around each junction or edit.
3. Draft primer intent, not final guaranteed primers, unless an external primer-design tool is also used.
4. Check and report required follow-up validation: Tm, GC percent, secondary structure, primer dimers, off-targets, restriction sites, overhang compatibility, synthesis length limits, and sequence safety/compliance.

## Rust Interface Notes

The root crate reexports the major internal crates from `src/lib.rs`:

- `gen::commands` for CLI command structs and execution paths.
- `gen::imports`, `gen::updates`, and `gen::exports` for programmatic workflows.
- `gen::annotations`, `gen::core`, `gen::graph`, `gen::models`, and optional `gen::diff` for lower-level automation.
- `gen::get_connection`, `gen::get_operation_connection`, and `gen::track_database` for database setup.

Use the CLI for ordinary user workflows. Use Rust APIs for scripts, tests, integrations, or when a user asks to build a new capability on top of Gen.
