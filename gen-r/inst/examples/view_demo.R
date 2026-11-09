#!/usr/bin/env Rscript
# Try the genr graph viewer end to end.
#
# Usage (from a shell):
#   Rscript view_demo.R                  # build a demo graph and open the Shiny viewer in a browser
#   Rscript view_demo.R --scale 1        # draw at the logical pixel size instead of 2x
#   Rscript view_demo.R --port 4321      # serve on a fixed port
#   Rscript view_demo.R --check          # build, render and smoke-test the server without a browser
#   Rscript view_demo.R --repo ~/work    # view the first graph of an existing Gen workspace instead
#
# From an R session: source("view_demo.R"), or copy the pieces you want.

suppressPackageStartupMessages(library(genr))

args <- commandArgs(trailingOnly = TRUE)
option <- function(name, default = NULL) {
  at <- match(paste0("--", name), args)
  if (is.na(at) || at == length(args)) default else args[[at + 1L]]
}
check_only <- "--check" %in% args
scale <- as.integer(option("scale", 2L))
port <- option("port")
repo_path <- option("repo")

# Open an existing workspace, or build a small one: pUC19 plus a sequenced variant sample.
if (!is.null(repo_path)) {
  repo <- Repository(repo_path)
  graphs <- repo$get_sequence_graphs()
  if (length(graphs) == 0L) stop("no sequence graphs in ", repo_path, call. = FALSE)
  graph <- graphs[[1L]]
} else {
  workspace <- file.path(tempdir(), "genr_view_demo")
  dir.create(workspace, showWarnings = FALSE)
  setwd(workspace)
  repo <- Repository(NULL)
  gbk <- system.file("extdata", "puc19.gbk", package = "genr")
  vcf <- system.file("extdata", "puc19_sequenced.vcf", package = "genr")
  stopifnot(nzchar(gbk), nzchar(vcf))
  repo$import_genbank(gbk, sample = "wt")
  repo$update_with_vcf(vcf, reference = "wt")
  graphs <- repo$get_sequence_graphs()
  cat("Graphs in the workspace:\n")
  for (candidate in graphs) cat("  ", candidate$sample_name(), "/", candidate$name(), "\n")
  # Prefer the graph with variants over the reference, when there is one.
  graph <- graphs[[length(graphs)]]
}

# A plot is a controller you can configure before viewing it.
p <- plot(graph, scale = scale)
cat(sprintf("Plot: %d x %d cells, scale %d\n", p$cols, p$rows, p$scale))

# Search the plotted graph for a restriction site and highlight where it falls. Results come
# back per graph, each with a list of matches.
results <- repo$search("GAATTC", ids = graph$id())
matches <- if (length(results) > 0L) results[[1L]]$matches else list()
cat(sprintf("EcoRI sites (GAATTC) found: %d\n", length(matches)))
if (length(matches) > 0L) {
  p$highlight_match(matches[[1L]], color = "yellow")
  p$go_to_match(matches[[1L]], center = TRUE)
}

# Render the same view straight to PNG files at both scales and report their sizes.
png_dimensions <- function(bytes) {
  # PNG header: 8 byte signature, then the IHDR chunk with width and height as big-endian ints.
  as.integer(c(
    sum(as.integer(bytes[17:20]) * 256^(3:0)),
    sum(as.integer(bytes[21:24]) * 256^(3:0))
  ))
}
for (s in c(1L, 2L)) {
  path <- file.path(tempdir(), sprintf("genr_view_demo_scale%d.png", s))
  p$save_png(path, scale = s)
  dims <- png_dimensions(readBin(path, "raw", n = 32L))
  cat(sprintf("  scale %d -> %s (%d x %d px)\n", s, path, dims[[1L]], dims[[2L]]))
}

app <- gen_app(p, title = "genr viewer demo")

if (check_only) {
  # Apply the operations the Shiny server applies for browser events and confirm each one
  # changes the rendered frame.
  frame <- function() p$render_png_viewport(p$cols, p$rows, p$scale)
  start <- frame()
  p$zoom_in()
  zoomed <- frame()
  stopifnot(!identical(start, zoomed))
  p$move_by(6L, 0L)
  panned <- frame()
  stopifnot(!identical(zoomed, panned))
  p$zoom_out()
  stopifnot(inherits(app, "shiny.appobj"))
  cat("Check complete: zoom and pan change the frame, and the Shiny app builds.\n")
  quit(status = 0L)
}

cat("Opening the viewer. Drag to pan, use the magnifiers to zoom, close the tab and press Ctrl+C to stop.\n")
if (is.null(port)) {
  shiny::runApp(app, launch.browser = TRUE)
} else {
  shiny::runApp(app, port = as.integer(port), launch.browser = TRUE)
}
