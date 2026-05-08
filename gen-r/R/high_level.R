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
    GraphController(bg$db_path, bg$id$hash_id, detail = detail, rows = rows, cols = cols)
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
