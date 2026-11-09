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
| `import_fasta(filename, sample, collection, fai, gzi)` | Import a FASTA file, returns a `gen_sample` |
| `import_reference_fasta(filename, reference, collection, fai, gzi)` | Import a FASTA file into a reference sample, returns a `gen_sample` |
| `import_gfa(filename, sample, collection)` | Import a GFA file, returns a `SequenceGraph` |
| `import_genbank(filename, sample, collection)` | Import a GenBank file (plain or gzipped), returns a `gen_sample` |
| `import_library(library_name, parts_list, seq_containers, sample, collection)` | Import a combinatorial sequence library, returns a `SequenceGraph` |
| `import_library_files(library_name, parts, library, sample, collection)` | Import a library from parts/library CSV files, returns a `SequenceGraph` |

For a remote indexed BGZF FASTA, pass its FASTA and BGZF index files explicitly:

```r
sample <- repo$import_fasta(
  "https://example.org/reference.fa.gz",
  sample = "reference",
  fai = "https://example.org/reference.fa.gz.fai",
  gzi = "https://example.org/reference.fa.gz.gzi"
)
```

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
plot(bg)  # opens the interactive viewer
```

## Graph viewer (GenPlot)

`plot(bg)` returns a `gen_plot` controller and displays it interactively when
evaluated at the console. Assign it to a variable to configure it before
display:

```r
p <- plot(bg)
p$zoom_in()
p$zoom_out()
p$move_by(dx = 5, dy = 0)   # pan right by 5 columns
p                            # open the interactive viewer
```

Track overlays and search highlights:

```r
p$add_track_file("annotations.gff3", name = "genes", sample = "ref")
results <- repo$search("ATCGATCG")
p$go_to_match(results[[1]]$matches[[1]])
p$highlight_match(results[[1]]$matches[[1]], color = "yellow")
p  # open the configured viewer
```

Use `view()` when selecting the display destination explicitly. It accepts
either a sequence graph directly or a configured `gen_plot`:

```r
p <- plot(bg, rows = 24, cols = 80)
view(p)                         # RStudio pane when available, otherwise browser
view(p, viewer = "pane")        # RStudio Viewer
view(p, viewer = "browser")     # web browser

view(bg, rows = 24, cols = 80)  # construct and display in one step
```

The viewer supports zoom controls, drag-to-pan behavior, and live updates
from the Rust-backed graph controller. The graph is drawn in Rust and sent to
the browser as a PNG, so the browser only forwards clicks and drags. Pass
`scale = 1` to `plot()` or `view()` for smaller images; the default of 2 draws every
pixel twice over, which is crisp on high-density displays at the same logical size. PNGs
are labeled 96 dpi times the scale, so scale 2 is 192 dpi and scale 3 is 288 dpi, close to a
300 dpi print.

When knitting a document, a `gen_plot` becomes a static base64 `<img>` with no
JavaScript. Write the same image to disk with `p$save_png("graph.png")`.

The viewer is also available as a conventional Shiny module for inclusion in a
larger application:

```r
ui <- shiny::fillPage(gen_shiny_ui("graph"))
server <- function(input, output, session) {
  gen_shiny_server("graph", p)
}
shiny::shinyApp(ui, server)
```

`gen_shiny_server()` clones `p` for each Shiny session. Viewport changes made
by one user therefore do not affect other users when the app is deployed.
`gen_app(p)` returns the standalone `shiny.appobj` without running it, which
is useful when configuring the host, port, launch behavior, or deployment
separately with `shiny::runApp()`.

Set `options(genr.viewer = FALSE)` to keep `print.gen_plot()` from launching
Shiny. This is useful in scripts and documentation builds that need the static
widget output.

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

- [Introduction](https://genhub-bio.github.io/gen/vignettes/introduction.html) — pUC19 import, editing, and export walkthrough.
- [Yeast expression library](https://genhub-bio.github.io/gen/vignettes/yeast_expression_library.html) — combinatorial library design for a carotenoid pathway.

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

Both installation commands install the Shiny viewer and its R dependencies.

## Development

### Docker install test

Build from the repository root so the nested Rust crate can see the workspace:

```sh
docker build -f gen-r/Dockerfile -t gen-r-install-test .
```

The image runs `R CMD INSTALL gen-r` during build and then executes a minimal
`library(genr); Repository()` smoke test.
