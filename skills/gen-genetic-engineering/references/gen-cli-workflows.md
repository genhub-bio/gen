# Gen CLI Workflows For Genetic Engineering

Use this reference after `SKILL.md` triggers and the user needs concrete `gen` usage.

## Current CLI Shape

Top-level pattern:

```sh
gen [--db <db-path>] <command> ...
```

Common setup:

```sh
gen init
gen defaults --database project.db --collection plasmids
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
