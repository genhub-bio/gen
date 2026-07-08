# gen-r

R bindings for the Gen sequence graph and version control system.

## Quick start

```r
library(genr)

repo <- Repository()          # open workspace in current directory
sample <- repo$import_fasta("seq.fa", sample = "ref")
plot(sample$block_groups[[1]])  # opens the graph viewer
```

`Repository()` is the primary entry point. It discovers the `.gen/` workspace,
opens the database, and exposes all import, export, update, search, and plot
operations as methods on the returned environment.

## Repository methods

**Import**

| Method | Description |
|--------|-------------|
| `import_fasta(filename, sample, shallow, collection)` | Import a FASTA file, returns a `gen_sample` |
| `import_gfa(filename, sample, collection)` | Import a GFA file, returns a `SequenceGraph` |
| `import_genbank(filename, sample, collection)` | Import a GenBank file (plain or gzipped), returns a `gen_sample` |
| `import_library(library_name, parts_list, seq_containers, sample, collection)` | Import a combinatorial sequence library, returns a `SequenceGraph` |
| `import_library_files(library_name, parts, library, sample, collection)` | Import a library from parts/library CSV files, returns a `SequenceGraph` |

Every import call returns the sequence graph(s) it just created directly, so
there's no need to follow up with `get_sequence_graphs()`. A `gen_sample` is a
list with `collection_name`, `sample_name`, and `block_groups` (a list of
`SequenceGraph`); index it with `sample$block_groups[[1]]` or `length(sample)`.

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
| `update_with_fasta(...)` | Apply a FASTA update to an existing block group, returns a `gen_sample` |
| `update_with_gfa(...)` | Apply a GFA update, returns a `gen_sample` |
| `update_with_gaf(...)` | Apply a GAF update, returns a `gen_sample` |
| `update_with_vcf(...)` | Apply a VCF update, returns a list of `gen_sample` (one per output sample) |
| `update_with_genbank(...)` | Apply a GenBank update, returns a `gen_sample` |
| `update_with_sequence(...)` | Apply a raw sequence update, returns a `gen_sample` |
| `update_with_library(...)` | Apply a library update, returns a `gen_sample` |
| `update_with_library_files(...)` | Apply a library-files update, returns a `gen_sample` |

**Derive / transform**

| Method | Description |
|--------|-------------|
| `derive_subgraph(sample, new_sample, region, ...)` | Extract a coordinate-bounded subgraph, returns a `SequenceGraph` |
| `derive_chunks(sample, new_sample, region, ...)` | Split a sequence graph at breakpoints or a fixed chunk size, returns a `gen_sample` |
| `stitch(sgs, new_sample, new_region)` | Concatenate sequence graphs end-to-end |

**Query**

| Method | Description |
|--------|-------------|
| `get_sequence_graphs()` | All sequence graphs in the database |
| `get_sequence_graphs_by_collection(collection)` | Sequence graphs in one collection |
| `get_sequence_graph_by_id(id)` | Sequence graph by `HashId` |
| `get_node_sequence(node)` | Sequence string for a graph node (`node$key` from graph dict) |
| `search(query, sgs, sequence_kind)` | K-mer search across sequence graphs |
| `build_index(sgs, sequence_kind, k)` | Build search index |
| `clear_index(sgs)` | Remove cached index files |

**Visualize**

```r
plot(bg)                 # returns a gen_plot; displays in RStudio Viewer / Shiny
```

## Graph viewer (GenPlot)

`plot(bg)` returns a `gen_plot` object. Printing it in RStudio or Shiny
renders an interactive canvas widget via `anyhtmlwidget`.

Navigate the view programmatically:

```r
p <- plot(bg)
p$zoom_in()
p$zoom_out()
p$move_by(dx = 5, dy = 0)   # pan right by 5 columns
p                            # re-render
```

Track overlays and search highlights:

```r
p$add_track_file("annotations.gff3", name = "genes", sample = "ref")
results <- repo$search("ATCGATCG")
p$go_to_match(results[[1]]$matches[[1]])
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

## Vignettes

- [Introduction](https://htmlpreview.github.io/?https://github.com/genhub-bio/gen/blob/main/docs/vignettes/introduction.html) — pUC19 import, editing, and export walkthrough.
- [Yeast expression library](https://htmlpreview.github.io/?https://github.com/genhub-bio/gen/blob/main/docs/vignettes/yeast_expression_library.html) — combinatorial library design for a carotenoid pathway.

The rendered HTML is checked into `docs/vignettes/`. After changing a vignette,
rebuild it with `Rscript -e "tools::buildVignettes(dir='gen-r', tangle=FALSE)"`
and copy the output from `gen-r/vignettes/*.html` into `docs/vignettes/`.

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
  "Windows" = sprintf("genr-windows-%s.zip", version),
  "Darwin" = if (grepl("aarch64|arm64", arch)) {
    sprintf("genr_%s-macos-arm64.tgz", version)
  } else {
    sprintf("genr_%s-macos-x86_64.tgz", version)
  },
  stop("Prebuilt genr packages are currently published for macOS and Windows.")
)

remotes::install_url(
  sprintf(
    "https://github.com/genhub-bio/gen/releases/download/v%s/%s",
    version,
    asset
  ),
  dependencies = c("Depends", "Imports")
)
```

`dependencies` is pinned to skip `Suggests` (`knitr`, `rmarkdown`, `testthat`,
and the `webshot2`/`htmlwidgets` pair used only for rendering PDF vignettes),
which `remotes` would otherwise install by default. Drop that argument (or
set it to `NA`) if you want those too, e.g. to build the vignettes locally.

Linux installs currently require a source build. For development or Linux,
install Rust/cargo, `libclang`, and `capnp`, then use:

```r
remotes::install_github(
  "genhub-bio/gen",
  subdir = "gen-r",
  ref = "v0.1.31",
  dependencies = c("Depends", "Imports")
)
```

## Development

### Building PDF vignettes

Vignettes render as HTML by default. To build a PDF instead (e.g.
`introduction.Rmd`), install the optional PDF-only dependencies and a LaTeX
distribution, then render with `output_format` set explicitly:

```r
install.packages(c("webshot2", "htmlwidgets"))
if (!tinytex::is_tinytex()) tinytex::install_tinytex()

rmarkdown::render(
  "gen-r/vignettes/introduction.Rmd",
  output_format = "rmarkdown::pdf_document"
)
```

Same call works for `yeast_expression_library.Rmd`. `webshot2` rasterizes
each `gen_plot` widget to a PNG for the LaTeX output (see
`knit_print.gen_plot` in `R/high_level.R`); `xelatex` is required for the
unicode characters (α, β, ×, →) used in the vignette text.

### Docker install test

Build from the repository root so the nested Rust crate can see the workspace:

```sh
docker build -f gen-r/Dockerfile -t gen-r-install-test .
```

The image runs `R CMD INSTALL gen-r` during build and then executes a minimal
`library(genr); Repository()` smoke test.
