#!/usr/bin/env Rscript
# Regenerate R/extendr-wrappers.R from the compiled Rust library.
#
# extendr's proc macros compile a hidden function
# `wrap__make_genr_wrappers` into the Rust library.  This script
# calls that function and writes its output to
# R/extendr-wrappers.R.
#
# Usage:
#   Rscript tools/regenerate-wrappers.R
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - R package dependencies (jsonlite, anyhtmlwidget, methods)

# NOTE: The auto-generated file will use the exact Rust parameter names from
# lib.rs (e.g., `name` rather than `collection_name`).  If you want different
# R-side naming, patch the R file after regeneration.

stopifnot(nzchar(Sys.which("cargo")))
stopifnot(require("rprojroot", quietly = TRUE))

pkg_root <- rprojroot::find_package_root_file(path = ".")

# 1. Build the Rust library
cat("Building Rust library...\n")
rust_dir <- file.path(pkg_root, "src", "rust")
system2(
  "cargo",
  args = c("build", "--release", "--lib",
           paste0("--manifest-path=", shQuote(file.path(rust_dir, "Cargo.toml"))),
           paste0("--target-dir=", shQuote(file.path(rust_dir, "target")))),
  stdout = if (interactive()) "" else FALSE,
  stderr = if (interactive()) "" else FALSE
)

# 2. Build & install the R package (so the shared library exists and
#    wrap__make_genr_wrappers is callable)
cat("Installing R package...\n")
system2(
  "R",
  args = c("CMD", "INSTALL", "--preclean",
           shQuote(pkg_root)),
  stdout = if (interactive()) "" else FALSE,
  stderr = if (interactive()) "" else FALSE
)

# 3. Load the package and call the generated wrapper function
cat("Generating R wrappers...\n")
library(genr)
wrapper_code <- .Call("wrap__make_genr_wrappers", FALSE, "genr", PACKAGE = "genr")

# 4. Write to file
outfile <- file.path(pkg_root, "R", "extendr-wrappers.R")
writeLines(wrapper_code, outfile)
cat("Wrote", outfile, "\n")
