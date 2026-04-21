# Generated bindings for the Gen Rust module.
# Update with rextendr::document() when the Rust export surface changes.

#' Initialise a Gen workspace in the current directory.
#' @export
init <- function() {
  .Call(wrap__init)
}

#' Return the current workspace `.gen` directory path.
#' @export
get_gen_dir <- function() {
  .Call(wrap__get_gen_dir)
}

#' Open a Gen database context.
#'
#' @param workspace_path Optional path to the workspace root.
#' @param db_path Optional path to a specific SQLite database.
#' @return A list containing the resolved workspace and database paths.
#' @export
db_context <- function(workspace_path = NULL, db_path = NULL) {
  .Call(wrap__db_context, workspace_path, db_path)
}

#' Import a FASTA file into a Gen collection.
#'
#' @param workspace_path Optional path to the workspace root.
#' @param db_path Optional path to a specific SQLite database.
#' @param filename Path to the FASTA file to import.
#' @param name Collection name. Required if no default collection is set.
#' @param sample Sample name to import into.
#' @param shallow Whether to store sequence data by reference instead of inline.
#' @export
import_fasta <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample = "sample", shallow = FALSE) {
  .Call(wrap__import_fasta, workspace_path, db_path, filename, name, sample, shallow)
}

#' Export a Gen collection to FASTA.
#'
#' @param workspace_path Optional path to the workspace root.
#' @param db_path Optional path to a specific SQLite database.
#' @param filename Output FASTA path.
#' @param name Collection name. Required if no default collection is set.
#' @param sample Optional sample name to export.
#' @export
export_fasta <- function(workspace_path = NULL, db_path = NULL, filename, name = NULL, sample = NULL) {
  .Call(wrap__export_fasta, workspace_path, db_path, filename, name, sample)
}
