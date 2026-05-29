# gen-r

R bindings for the Gen sequence graph and version control system.

## Quick start

```r
library(genr)

repo <- Repository()          # open workspace in current directory
repo$import_fasta("seq.fa", sample = "ref")
bgs  <- repo$get_block_groups()
repo$plot(bgs[[1]])           # opens the graph viewer
```

`Repository()` is the primary entry point. It discovers the `.gen/` workspace,
opens the database, and exposes all import, export, update, search, and plot
operations as methods on the returned environment.

## Repository methods

**Import**

| Method | Description |
|--------|-------------|
| `import_fasta(filename, sample, shallow, collection)` | Import a FASTA file |
| `import_gfa(filename, sample, collection)` | Import a GFA file |
| `import_genbank(filename, sample, collection)` | Import a GenBank file (plain or gzipped) |
| `import_library(library_name, parts_list, seq_containers, sample, collection)` | Import a combinatorial sequence library |
| `import_library_files(library_name, parts, library, sample, collection)` | Import a library from parts/library CSV files |

See also the standalone `import_bioconductor()` and `import_granges()` helpers for
Bioconductor `DNAStringSet` / `GRanges` objects.

**Export**

| Method | Description |
|--------|-------------|
| `export_fasta(filename, sample, collection)` | Export to FASTA |
| `export_gfa(filename, sample, node_max, collection)` | Export to GFA |
| `export_genbank(filename, sample, collection)` | Export to GenBank |

**Update**

| Method | Description |
|--------|-------------|
| `update_with_fasta(...)` | Apply a FASTA update to an existing block group |
| `update_with_gfa(...)` | Apply a GFA update |
| `update_with_gaf(...)` | Apply a GAF update |
| `update_with_vcf(...)` | Apply a VCF update |
| `update_with_genbank(...)` | Apply a GenBank update |
| `update_with_sequence(...)` | Apply a raw sequence update |
| `update_with_library(...)` | Apply a library update |
| `update_with_library_files(...)` | Apply a library-files update |

**Derive / transform**

| Method | Description |
|--------|-------------|
| `derive_subgraph(sample, new_sample, region, ...)` | Extract a coordinate-bounded subgraph |
| `derive_chunks(sample, new_sample, region, ...)` | Split a block group at breakpoints or a fixed chunk size |
| `stitch(bgs, new_sample, new_region)` | Concatenate block groups end-to-end |

**Query**

| Method | Description |
|--------|-------------|
| `get_block_groups()` | All block groups in the database |
| `get_block_groups_by_collection(collection)` | Block groups in one collection |
| `get_block_group_by_id(id)` | Block group by `HashId` |
| `get_node_sequence(node)` | Sequence string for a graph node (`node$key` from graph dict) |
| `search(query, bgs, sequence_kind)` | K-mer search across block groups |
| `build_index(bgs, sequence_kind, k)` | Build search index |
| `clear_index(bgs)` | Remove cached index files |

**Visualize**

```r
repo$plot(bg)                 # returns a gen_plot; displays in RStudio Viewer / Shiny
```

## Graph viewer (GenPlot)

`repo$plot(bg)` returns a `gen_plot` object. Printing it in RStudio or Shiny
renders an interactive canvas widget via `anyhtmlwidget`.

Navigate the view programmatically:

```r
p <- repo$plot(bg)
p$zoom_in()
p$zoom_out()
p$move_by(dx = 5, dy = 0)   # pan right by 5 columns
p                            # re-render
```

Track overlays and search highlights:

```r
p$add_track_file("annotations.gff3", name = "genes", sample = "ref")
results <- repo$search("ATCGATCG")
p$goto_match(results[[1]]$matches[[1]])
p$highlight_match(results[[1]]$matches[[1]], color = "yellow")
p
```

## Bioconductor integration

```r
library(Biostrings)
seqs <- DNAStringSet(c(chr1 = "ACGTACGT", chr2 = "TTGCTTGC"))
import_bioconductor(seqs, sample = "hg38", repo = repo)

library(GenomicRanges)
gr <- GRanges(seqnames = c("chr1", "chr1"),
              ranges   = IRanges(start = c(1, 5), end = c(4, 8)))
names(gr) <- c("gene_a", "gene_b")
import_granges(gr, seqs, sample = "hg38", repo = repo)
```

See `inst/examples/combinatorial_expression_cassette.R` and
`inst/examples/library_from_granges.R` for full worked examples.

## Layout

The package follows the standard `extendr` structure:

- `src/rust/` — Rust crate linking against the Gen workspace crates.
- `src/entrypoint.c` — forwards R's package loader entrypoint to the Rust registration function.
- `R/extendr-wrappers.R` — auto-generated R-callable wrappers for exported Rust functions (do not edit by hand).
- `R/high_level.R` — `Repository`, `GenPlot`, `import_bioconductor`, `import_granges`, and supporting helpers.

## Installation

Install a prebuilt package from a tagged GitHub release. This is the supported
end-user path on macOS and Windows and does not require Rust, cargo,
`libclang`, or `capnp`.

On macOS and Windows, install the published R package binary:

```r
install.packages("remotes")
version <- "0.1.31"
sysname <- Sys.info()[["sysname"]]
arch <- R.version$arch

asset <- switch(
  sysname,
  "Windows" = sprintf("genr_%s-windows-x86_64.zip", version),
  "Darwin" = if (grepl("aarch64|arm64", arch)) {
    sprintf("genr_%s-macos-arm64.tgz", version)
  } else {
    sprintf("genr_%s-macos-x86_64.tgz", version)
  },
  stop("Prebuilt genr packages are currently published for macOS and Windows.")
)

remotes::install_url(sprintf(
  "https://github.com/genhub-bio/gen/releases/download/v%s/%s",
  version,
  asset
))
```

Linux installs currently require a source build. For development or Linux,
install Rust/cargo, `libclang`, and `capnp`, then use:

```r
remotes::install_github("genhub-bio/gen", subdir = "gen-r", ref = "v0.1.31")
```

## Development

### Docker install test

Build from the repository root so the nested Rust crate can see the workspace:

```sh
docker build -f gen-r/Dockerfile -t gen-r-install-test .
```

The image runs `R CMD INSTALL gen-r` during build and then executes a minimal
`library(genr); Repository()` smoke test.
