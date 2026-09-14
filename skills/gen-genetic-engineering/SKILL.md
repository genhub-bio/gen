---
name: gen-genetic-engineering
description: Help users apply the Gen `gen` CLI and Rust library to genetic engineering workflows, including importing FASTA/GenBank/GFA genomes and plasmids, viewing regions and graphs, summarizing operation history and diffs, predicting likely sequence/annotation impacts of edits, updating sequences from explicit strings/files/VCF/GAF/GenBank, designing combinatorial libraries, exporting synthesis or cloning artifacts, and reasoning about primer ordering for synthesis, cloning, validation, or sequencing.
---

# Gen Genetic Engineering

## Core Approach

Use this skill to turn a biological engineering intent into concrete `gen` commands, validation steps, and caveats.

Start by identifying:

- The workspace: whether `gen init` has run, which database and collection are active, and which sample is the reference or parent.
- The input artifacts: FASTA, GenBank, GFA, VCF, GAF, CSV library design, parts FASTA, annotations, or raw sequence strings.
- The desired biological change: replacement, insertion, deletion, variant application, library slot expansion, clone-ready export, or impact analysis.
- The coordinate system: Gen update regions use path/accession/annotation-style region names such as `chr1:2-5`; do not confuse GraphNode `sequence_start`/`sequence_end` with graph coordinates.
- The risk level: for wet-lab ordering, be explicit that `gen` can produce and inspect sequences, but primer thermodynamics, vendor constraints, off-target analysis, assembly overhang rules, and regulatory/safety review need domain tools or user confirmation.

Prefer CLI recipes for user-facing help. Use the Rust API only when the user asks to build automation or when CLI behavior is ambiguous.

## Source Of Truth

When working inside the Gen repository, inspect these files before giving precise syntax:

- `src/commands/mod.rs` for top-level commands.
- `src/commands/import/*.rs`, `src/commands/update/*.rs`, and `src/commands/export/*.rs` for argument names.
- `src/lib.rs` for the public Rust facade and reexports.
- `docs/commands.md`, `examples/yeast_editing/`, `examples/externally_edited_files/`, and `examples/combinatorial_plasmid_design/` for workflows.

If `gen` is installed in the user's environment, verify command syntax with `gen --help` and `gen <subcommand> --help` before presenting a final command sequence. The source currently uses subcommand syntax such as `gen import fasta ...`, even if older examples show option-style forms.

For detailed command recipes and workflow patterns, read `references/gen-cli-workflows.md`.

## Workflow

1. Establish repository defaults:
   - Use `gen init` when there is no `.gen` directory.
   - Use `gen defaults --database <db>.db --collection <collection>` to avoid repeating `--db` and `--name`.
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

## Genetic Engineering Guardrails

Do not pretend `gen` alone predicts functional impact. For impact predictions, combine `gen` outputs with explicit biological checks:

- Translate coding sequences when ORFs, frames, start/stop codons, or peptide changes matter.
- Inspect annotations after GenBank imports/updates and mention whether feature coordinates may need propagation or manual review.
- Check junction sequences for cloning scars, restriction sites, homology arms, overhangs, or unwanted motifs.
- For primer ordering, derive candidate binding regions from exported or extracted sequence, then state that final primer Tm, secondary structure, dimers, off-targets, vendor limits, and assembly chemistry must be checked with appropriate primer-design tools.

Avoid giving operational assistance for unsafe or disallowed biological engineering. If a request involves pathogenicity, toxin expression, evading detection, or harmful organism engineering, refuse that portion and offer benign sequence-management help instead.
