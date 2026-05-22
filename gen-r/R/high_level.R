#' @useDynLib genr, .registration = TRUE
NULL

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Construct a HashId
#'
#' Wraps a raw hash string as a typed \code{gen_hash_id} object used to
#' identify block groups and nodes within the Gen database.
#'
#' @param hash_id Character. The hex hash string.
#' @return A \code{gen_hash_id} list.
#' @export
HashId <- function(hash_id) {
  structure(list(hash_id = as.character(hash_id)), class = "gen_hash_id")
}

#' Construct a NodeKey
#'
#' A node key identifies a contiguous byte range within a graph node.
#' Pass the result to \code{repo$get_block_sequence()} to retrieve the
#' underlying sequence.
#'
#' @param node_id Character or \code{gen_hash_id}. Node identifier.
#' @param sequence_start Integer. Start byte offset (inclusive).
#' @param sequence_end Integer. End byte offset (exclusive).
#' @return A \code{gen_node_key} list.
#' @export
NodeKey <- function(node_id, sequence_start, sequence_end) {
  structure(
    list(
      node_id = if (inherits(node_id, "gen_hash_id")) node_id else HashId(node_id),
      sequence_start = as.integer(sequence_start),
      sequence_end = as.integer(sequence_end)
    ),
    class = "gen_node_key"
  )
}

#' Construct a SequencePart
#'
#' Represents a named sequence element used when importing combinatorial
#' libraries via \code{repo$import_library()}.
#'
#' @param name Character. Display name for the part.
#' @param sequence Character. Nucleotide or amino-acid sequence string.
#' @return A \code{gen_sequence_part} list.
#' @export
SequencePart <- function(name, sequence) {
  structure(
    list(
      name = as.character(name),
      sequence = as.character(sequence),
      sequence_length = nchar(as.character(sequence), type = "chars")
    ),
    class = "gen_sequence_part"
  )
}

#' Construct a DbContext
#'
#' Opens the Gen graph database and operations database, returning a context
#' object used by low-level Rust entry points.  Prefer \code{Repository()}
#' for day-to-day use.
#'
#' @param workspace_path Character or \code{NULL}. Path to the workspace root.
#'   Defaults to the current working directory.
#' @param db_path Character or \code{NULL}. Explicit path to the graph SQLite
#'   database.  When \code{NULL} the path is derived from the workspace.
#' @return A \code{gen_db_context} object.
#' @export
DbContext <- function(workspace_path = NULL, db_path = NULL) {
  structure(db_context(workspace_path = workspace_path, db_path = db_path), class = "gen_db_context")
}

.as_block_group <- function(x, gen_dir = NULL) {
  bg <- new.env(parent = emptyenv())
  bg$id <- HashId(x$id)
  bg$collection_name <- x$collection_name
  bg$sample_name <- x$sample_name
  bg$name <- x$name
  bg$db_path <- x$db_path %||% NULL
  bg$gen_dir <- gen_dir

  bg$plot <- function(rows = NULL, cols = NULL, detail = "normal") {
    if (is.null(bg$db_path)) {
      stop("plot() requires a db_path; obtain BlockGroup via Repository()", call. = FALSE)
    }
    GenPlot(bg$db_path, bg$id$hash_id, detail = detail, rows = rows, cols = cols)
  }

  bg$export_fasta <- function(filename) {
    if (is.null(bg$db_path)) stop("export_fasta() requires a db_path", call. = FALSE)
    repo_bg_export_fasta(bg$db_path, bg$collection_name, bg$sample_name, filename)
  }

  bg$export_gfa <- function(filename, node_max = NULL) {
    if (is.null(bg$db_path)) stop("export_gfa() requires a db_path", call. = FALSE)
    repo_bg_export_gfa(bg$db_path, bg$collection_name, bg$sample_name, filename, node_max)
  }

  bg$export_genbank <- function(filename) {
    if (is.null(bg$db_path)) stop("export_genbank() requires a db_path", call. = FALSE)
    repo_bg_export_genbank(bg$db_path, bg$collection_name, bg$sample_name, filename)
  }

  bg$build_index <- function(sequence_kind = "dna", k = 16L) {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("build_index() requires a Repository context", call. = FALSE)
    repo_build_index(bg$db_path, bg$gen_dir, c(bg$id$hash_id), sequence_kind, as.integer(k))
  }

  bg$search <- function(query, sequence_kind = "dna") {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("search() requires a Repository context", call. = FALSE)
    result <- repo_search(bg$db_path, bg$gen_dir, query, c(bg$id$hash_id), sequence_kind)
    if (length(result) == 0L) list() else result[[1L]]$matches
  }

  bg$clear_index <- function() {
    if (is.null(bg$gen_dir)) stop("clear_index() requires a Repository context", call. = FALSE)
    repo_clear_index(bg$gen_dir, c(bg$id$hash_id))
  }

  bg$subgraph <- function(new_sample, start, end, backbone = NULL) {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("subgraph() requires a Repository context", call. = FALSE)
    workspace_path <- dirname(bg$gen_dir)
    result <- repo_bg_subgraph(
      workspace_path, bg$db_path, bg$collection_name, bg$sample_name,
      bg$name, new_sample, as.integer(start), as.integer(end), backbone
    )
    .as_block_group(result, gen_dir = bg$gen_dir)
  }

  bg$chunks <- function(new_sample, breakpoints = NULL, chunk_size = NULL, backbone = NULL) {
    if (is.null(bg$db_path) || is.null(bg$gen_dir))
      stop("chunks() requires a Repository context", call. = FALSE)
    workspace_path <- dirname(bg$gen_dir)
    results <- repo_bg_chunks(
      workspace_path, bg$db_path, bg$collection_name, bg$sample_name,
      bg$name, new_sample, breakpoints,
      if (is.null(chunk_size)) NULL else as.integer(chunk_size),
      backbone
    )
    lapply(results, .as_block_group, gen_dir = bg$gen_dir)
  }

  class(bg) <- "gen_block_group"
  bg
}

#' Create a graph plot controller
#'
#' Returns a \code{gen_plot} environment that renders the sequence graph for a
#' block group.  Printing or displaying the object in RStudio / Shiny shows an
#' interactive htmlwidget.
#'
#' @param db_path Character. Path to the Gen graph SQLite database.
#' @param block_group_id Character or \code{gen_hash_id}. Block group to render.
#' @param detail Character. Level of node detail: \code{"normal"} (default) or
#'   \code{"compressed"}.
#' @param rows Integer or \code{NULL}. Canvas height in terminal rows (default 24).
#' @param cols Integer or \code{NULL}. Canvas width in terminal columns (default 80).
#' @return A \code{gen_plot} environment with methods:
#'   \describe{
#'     \item{\code{zoom_in()}}{Step one zoom level in. Returns self invisibly.}
#'     \item{\code{zoom_out()}}{Step one zoom level out. Returns self invisibly.}
#'     \item{\code{move_by(dx, dy)}}{Pan the viewport by \code{dx} columns and \code{dy} rows. Returns self invisibly.}
#'     \item{\code{handle_click(col, row)}}{Send a mouse click; returns \code{TRUE} if a node was hit.}
#'     \item{\code{set_detail(detail)}}{Change node detail level. Returns self invisibly.}
#'     \item{\code{render_frame(cols, rows)}}{Render to JSON string (used internally by the widget).}
#'   }
#' @export
GenPlot <- function(db_path, block_group_id, detail = "normal", rows = NULL, cols = NULL) {
  ctrl <- new.env(parent = emptyenv())
  ctrl$db_path <- db_path
  ctrl$block_group_id <- if (inherits(block_group_id, "gen_hash_id")) block_group_id$hash_id else as.character(block_group_id)
  ctrl$detail <- detail
  ctrl$ops <- character()
  ctrl$track_specs <- list()
  ctrl$rows <- rows %||% 24L
  ctrl$cols <- cols %||% 80L

  ctrl$set_detail <- function(detail) {
    ctrl$detail <- detail
    invisible(ctrl)
  }

  ctrl$render_frame <- function(cols = ctrl$cols, rows = ctrl$rows) {
    ctrl$cols <- cols
    ctrl$rows <- rows
    graph_render_frame(ctrl$db_path, ctrl$block_group_id, ctrl$detail, as.integer(cols), as.integer(rows), paste(ctrl$ops, collapse = ";"), jsonlite::toJSON(ctrl$track_specs, auto_unbox = TRUE))
  }

  ctrl$add_track_group <- function(group) {
    ctrl$track_specs <- c(ctrl$track_specs, list(list(type = "group", name = group)))
    invisible(ctrl)
  }

  ctrl$add_track_file <- function(path, name = NULL, sample = NULL) {
    spec <- list(type = "file", path = path, name = name, sample = sample)
    ctrl$track_specs <- c(ctrl$track_specs, list(spec))
    invisible(ctrl)
  }

  ctrl$clear_tracks <- function() {
    ctrl$track_specs <- list()
    invisible(ctrl)
  }

  ctrl$zoom_in <- function() {
    ctrl$ops <- c(ctrl$ops, "zi")
    invisible(ctrl)
  }

  ctrl$zoom_out <- function() {
    ctrl$ops <- c(ctrl$ops, "zo")
    invisible(ctrl)
  }

  ctrl$handle_click <- function(col, row) {
    clicked <- graph_handle_click(
      ctrl$db_path,
      ctrl$block_group_id,
      ctrl$detail,
      paste(ctrl$ops, collapse = ";"),
      as.integer(col),
      as.integer(row)
    )
    if (isTRUE(clicked)) {
      ctrl$ops <- c(ctrl$ops, sprintf("c,%d,%d", as.integer(col), as.integer(row)))
    }
    clicked
  }

  ctrl$move_by <- function(dx, dy) {
    ctrl$ops <- c(ctrl$ops, sprintf("m,%d,%d", as.integer(dx), as.integer(dy)))
    invisible(ctrl)
  }

  ctrl$goto_match <- function(match_locus) {
    ctrl$detail <- "full"
    block <- match_locus$start$block
    offset <- match_locus$start$offset
    node_len <- block$sequence_end - block$sequence_start
    frac_x <- if (node_len > 1L) offset / (node_len - 1L) else 0.5
    frac_x <- max(0.0, min(1.0, frac_x))
    ctrl$ops <- c(ctrl$ops, sprintf("goto,%s,%d,%d,%.6f",
      block$node_id,
      as.integer(block$sequence_start),
      as.integer(block$sequence_end),
      frac_x))
    invisible(ctrl)
  }

  ctrl$highlight_match <- function(match_locus, color = "yellow") {
    blocks <- match_locus$blocks
    n <- length(blocks)
    start_offset <- match_locus$start$offset
    end_offset <- match_locus$end$offset
    strand_code <- switch(match_locus$strand, forward = "f", reverse = "r", "u")
    block_parts <- paste(
      sapply(blocks, function(b) {
        sprintf("%s,%d,%d", b$node_id, as.integer(b$sequence_start), as.integer(b$sequence_end))
      }),
      collapse = ","
    )
    ctrl$ops <- c(ctrl$ops, sprintf("hl,%s,%d,%d,%s,%d,%s",
      color,
      as.integer(start_offset),
      as.integer(end_offset),
      strand_code,
      as.integer(n),
      block_parts))
    invisible(ctrl)
  }

  ctrl$clear_highlights <- function() {
    ctrl$ops <- c(ctrl$ops, "clrhl")
    invisible(ctrl)
  }

  class(ctrl) <- "gen_plot"
  ctrl
}

#' @export
print.gen_plot <- function(x, ...) {
  json <- x$render_frame(x$cols, x$rows)
  w <- .genplot_widget(json)
  print(w)
  invisible(x)
}

methods::setOldClass("gen_plot")
methods::setMethod("show", "gen_plot", function(object) print(object))

#' Open a Gen repository
#'
#' The main entry point for working with a Gen sequence graph database.
#' Discovers the \code{.gen/} directory from \code{path} (or the current
#' working directory) and opens both the graph and operations databases.
#'
#' @param path Character or \code{NULL}. Path to the workspace root.  When
#'   \code{NULL} the current working directory is used.
#' @return A \code{gen_repository} environment with the following methods:
#'   \describe{
#'     \item{\code{import_fasta(filename, sample, shallow, collection_name=NULL)}}{Import a FASTA file as a new block group.}
#'     \item{\code{import_gfa(filename, sample, collection_name=NULL)}}{Import a GFA file.}
#'     \item{\code{import_genbank(filename, sample, collection_name=NULL)}}{Import a GenBank file (plain or gzipped).}
#'     \item{\code{import_library(library_name, parts_list, sample=NULL, collection_name=NULL)}}{Import a combinatorial sequence library.}
#'     \item{\code{import_library_files(library_name, parts, library, sample, collection_name=NULL)}}{Import a library from parts/library CSV files.}
#'     \item{\code{export_fasta(block_group, filename)}}{Export a block group to FASTA.}
#'     \item{\code{export_gfa(block_group, filename, node_max)}}{Export a block group to GFA.}
#'     \item{\code{export_genbank(block_group, filename)}}{Export a block group to GenBank.}
#'     \item{\code{update_with_fasta(filename, sample, new_sample, region_name, start, end, collection_name=NULL)}}{Apply a FASTA update to an existing block group.}
#'     \item{\code{update_with_gfa(filename, sample, new_sample, collection_name=NULL)}}{Apply a GFA update.}
#'     \item{\code{update_with_vcf(filename, genotype=NULL, sample=NULL, parent_samples, in_place, collection_name=NULL)}}{Apply a VCF update.}
#'     \item{\code{update_with_genbank(filename, sample, create_missing, collection_name=NULL)}}{Apply a GenBank update.}
#'     \item{\code{update_with_sequence(sequence, sample, new_sample, region_name, start, end, no_reference_path_update, collection_name=NULL)}}{Apply a raw sequence update.}
#'     \item{\code{update_with_library(sample=NULL, new_sample_name, path_name, start, end, parts_list, collection_name=NULL)}}{Apply a library update.}
#'     \item{\code{get_block_groups()}}{Return a list of all block groups.}
#'     \item{\code{get_block_group_by_id(id)}}{Return a block group by its \code{HashId}.}
#'     \item{\code{get_block_groups_by_collection(collection_name)}}{Return block groups in a collection.}
#'     \item{\code{get_block_sequence(node_key)}}{Return the sequence string for a \code{NodeKey}.}
#'     \item{\code{plot(block_group, rows, cols, detail)}}{Return a \code{gen_plot} for a block group.}
#'     \item{\code{stitch(bgs, new_sample, new_region)}}{Concatenate block groups end-to-end into a new block group.}
#'     \item{\code{derive_subgraph(new_sample, start, end, backbone)}}{Derive a subgraph block group.}
#'     \item{\code{derive_chunks(new_sample, breakpoints, chunk_size, backbone)}}{Split a block group into chunks.}
#'     \item{\code{build_index(bgs, sequence_kind, k)}}{Build a k-mer seed index to accelerate \code{search()}.}
#'     \item{\code{search(query, bgs, sequence_kind)}}{Search for exact sequence occurrences across block groups.}
#'     \item{\code{clear_index(bgs)}}{Remove cached search index files.}
#'     \item{\code{execute(query)}}{Run a raw SQL statement against the graph database.}
#'     \item{\code{query(query)}}{Run a raw SQL query and return results as a list of rows.}
#'   }
#' @export
Repository <- function(path = NULL) {
  repo <- new.env(parent = emptyenv())
  repo$gen_dir <- repo_get_gen_dir(path)
  repo$db_path <- repo_get_db_path(path)

  repo$execute <- function(query) {
    repo_execute(repo$db_path, query)
  }

  repo$query <- function(query) {
    repo_query(repo$db_path, query)
  }

  repo$get_block_group_by_id <- function(id) {
    id_str <- if (inherits(id, "gen_hash_id")) id$hash_id else as.character(id)
    .as_block_group(repo_get_block_group_by_id(repo$db_path, id_str), gen_dir = repo$gen_dir)
  }

  repo$get_block_groups <- function() {
    lapply(repo_get_block_groups(repo$db_path), .as_block_group, gen_dir = repo$gen_dir)
  }

  repo$get_block_groups_by_collection <- function(collection_name) {
    lapply(
      repo_get_block_groups_by_collection(repo$db_path, collection_name),
      .as_block_group,
      gen_dir = repo$gen_dir
    )
  }

  repo$block_group_to_dict <- function(block_group) {
    repo_block_group_to_dict(repo$db_path, block_group$id$hash_id)
  }

  repo$block_group_to_rustworkx <- function(block_group) {
    repo$block_group_to_dict(block_group)
  }

  repo$block_group_to_networkx <- function(block_group) {
    repo$block_group_to_dict(block_group)
  }

  repo$plot <- function(block_group, rows = NULL, cols = NULL, detail = "normal") {
    GenPlot(repo$db_path, block_group$id$hash_id, detail = detail, rows = rows, cols = cols)
  }

  repo$get_block_sequence <- function(node_key) {
    repo_get_block_sequence(
      repo$db_path,
      node_key$node_id$hash_id,
      as.integer(node_key$sequence_start),
      as.integer(node_key$sequence_end)
    )
  }

  repo$stitch <- function(bgs, new_sample, new_region) {
    if (length(bgs) == 0L) stop("stitch() requires at least one block group", call. = FALSE)
    collection_name <- bgs[[1L]]$collection_name
    sample_name <- bgs[[1L]]$sample_name
    for (bg in bgs[-1L]) {
      if (bg$collection_name != collection_name)
        stop(sprintf("All block groups must be in the same collection ('%s' vs '%s')",
                     collection_name, bg$collection_name), call. = FALSE)
      if (bg$sample_name != sample_name)
        stop(sprintf("All block groups must be in the same sample ('%s' vs '%s')",
                     sample_name, bg$sample_name), call. = FALSE)
    }
    regions <- paste(sapply(bgs, function(bg) bg$name), collapse = ",")
    workspace_path <- dirname(repo$gen_dir)
    result <- repo_stitch(
      workspace_path, repo$db_path, collection_name, sample_name,
      new_sample, new_region, regions
    )
    .as_block_group(result, gen_dir = repo$gen_dir)
  }

  repo$build_index <- function(bgs = NULL, sequence_kind = "dna", k = 16L) {
    ids <- if (is.null(bgs)) character() else sapply(bgs, function(bg) bg$id$hash_id)
    repo_build_index(repo$db_path, repo$gen_dir, ids, sequence_kind, as.integer(k))
  }

  repo$search <- function(query, bgs = NULL, sequence_kind = "dna") {
    ids <- if (is.null(bgs)) character() else sapply(bgs, function(bg) bg$id$hash_id)
    results <- repo_search(repo$db_path, repo$gen_dir, query, ids, sequence_kind)
    lapply(results, function(r) {
      list(
        block_group = .as_block_group(r$block_group, gen_dir = repo$gen_dir),
        matches = r$matches
      )
    })
  }

  repo$clear_index <- function(bgs = NULL) {
    ids <- if (is.null(bgs)) character() else sapply(bgs, function(bg) bg$id$hash_id)
    repo_clear_index(repo$gen_dir, ids)
  }

  class(repo) <- "gen_repository"
  repo
}
