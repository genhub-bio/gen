# gen-r

R bindings for the Gen sequence graph and version control system.

## Layout

The package follows the standard `extendr` structure:

- `src/rust/` contains the Rust crate that links against the Gen workspace crates.
- `src/entrypoint.c` forwards R's package loader entrypoint to the Rust-generated
  registration function.
- `R/extendr-wrappers.R` contains the R-callable wrappers for exported Rust
  functions.

The current exported surface is intentionally small:

- `init()` initializes a `.gen/` workspace in the current directory.
- `get_gen_dir()` returns the resolved `.gen/` path for the active workspace.
- `db_context()` validates that Gen can resolve the workspace and target database,
  then returns the resolved paths as an R list.
- `import_fasta()` imports sequence data into a collection.
- `export_fasta()` exports a collection or sample back to FASTA.

## Installation

Install a prebuilt package from a tagged GitHub release. This is the supported
end-user path and does not require Rust, cargo, `libclang`, or `capnp`.

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

If you need to build from source for development, install Rust/cargo,
`libclang`, and `capnp`, then use:

```r
remotes::install_github("genhub-bio/gen", subdir = "gen-r", ref = "v0.1.31")
```

## Development

## Docker Install Test

Build from the repository root so the nested Rust crate can see the workspace:

```sh
docker build -f gen-r/Dockerfile -t gen-r-install-test .
```

The image runs `R CMD INSTALL gen-r` during build and then executes a minimal
`library(genr); init()` smoke test.
