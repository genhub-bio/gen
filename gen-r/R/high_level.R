`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

HashId <- function(hash_id) {
  structure(list(hash_id = as.character(hash_id)), class = "gen_hash_id")
}

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

DbContext <- function(workspace_path = NULL, db_path = NULL) {
  structure(db_context(workspace_path = workspace_path, db_path = db_path), class = "gen_db_context")
}

.as_block_group <- function(x) {
  bg <- new.env(parent = emptyenv())
  bg$id <- HashId(x$id)
  bg$collection_name <- x$collection_name
  bg$sample_name <- x$sample_name
  bg$name <- x$name
  bg$db_path <- x$db_path %||% NULL
  bg$plot <- function(rows = NULL, cols = NULL, detail = "normal") {
    if (is.null(bg$db_path)) {
      stop("plot() requires a db_path; obtain BlockGroup via Repository()", call. = FALSE)
    }
    GraphController(bg$db_path, bg$id$hash_id, detail = detail, rows = rows, cols = cols)
  }
  class(bg) <- "gen_block_group"
  bg
}

GraphController <- function(db_path, block_group_id, detail = "normal", rows = NULL, cols = NULL) {
  ctrl <- new.env(parent = emptyenv())
  ctrl$db_path <- db_path
  ctrl$block_group_id <- if (inherits(block_group_id, "gen_hash_id")) block_group_id$hash_id else as.character(block_group_id)
  ctrl$detail <- detail
  ctrl$ops <- character()
  ctrl$rows <- rows %||% 24L
  ctrl$cols <- cols %||% 80L

  ctrl$set_detail <- function(detail) {
    ctrl$detail <- detail
    invisible(ctrl)
  }

  ctrl$render_frame <- function(cols = ctrl$cols, rows = ctrl$rows) {
    ctrl$cols <- cols
    ctrl$rows <- rows
    graph_render_frame(ctrl$db_path, ctrl$block_group_id, ctrl$detail, as.integer(cols), as.integer(rows), paste(ctrl$ops, collapse = ";"))
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

  class(ctrl) <- "gen_graph_controller"
  ctrl
}

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
    .as_block_group(repo_get_block_group_by_id(repo$db_path, if (inherits(id, "gen_hash_id")) id$hash_id else as.character(id)))
  }

  repo$get_block_groups <- function() {
    lapply(repo_get_block_groups(repo$db_path), .as_block_group)
  }

  repo$get_block_groups_by_collection <- function(collection_name) {
    lapply(repo_get_block_groups_by_collection(repo$db_path, collection_name), .as_block_group)
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

  repo$create_block_group <- function(name, collection_name, sample_name) {
    .as_block_group(repo_create_block_group(repo$db_path, name, collection_name, sample_name))
  }

  repo$plot <- function(block_group, rows = NULL, cols = NULL, detail = "normal") {
    GraphController(repo$db_path, block_group$id$hash_id, detail = detail, rows = rows, cols = cols)
  }

  repo$get_block_sequence <- function(node_key) {
    repo_get_block_sequence(
      repo$db_path,
      node_key$node_id$hash_id,
      as.integer(node_key$sequence_start),
      as.integer(node_key$sequence_end)
    )
  }

  class(repo) <- "gen_repository"
  repo
}
