library(genr)

local_fixture_root <- function() {
  candidates <- c(
    tryCatch(testthat::test_path("..", "fixtures"), error = function(...) NA_character_),
    file.path(Sys.getenv("GITHUB_WORKSPACE", unset = ""), "gen-r", "tests", "fixtures"),
    file.path(Sys.getenv("GITHUB_WORKSPACE", unset = ""), "tests", "fixtures"),
    file.path("gen-r", "tests", "fixtures"),
    file.path("tests", "fixtures"),
    file.path("fixtures")
  )
  for (candidate in candidates) {
    if (!is.na(candidate) && nzchar(candidate) && dir.exists(candidate)) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
  }
  stop("Could not locate test fixtures for genr.", call. = FALSE)
}

fixture_root <- local_fixture_root()

fixture_path <- function(...) {
  normalizePath(file.path(fixture_root, ...), mustWork = TRUE)
}

setup_repository <- function() {
  path <- tempfile("genr-test-")
  dir.create(path, recursive = TRUE)
  Repository(path)
}

is_png <- function(bytes) {
  is.raw(bytes) && identical(
    as.integer(bytes[1:8]),
    c(0x89L, 0x50L, 0x4EL, 0x47L, 0x0DL, 0x0AL, 0x1AL, 0x0AL)
  )
}

png_size <- function(bytes) {
  unname(genr:::.png_dimensions(bytes))
}

expect_binding_result <- function(result) {
  expect_false(inherits(result, "try-error"), info = paste(result, collapse = "\n"))
  expect_true(
    is.character(result) ||
      is.list(result) ||
      is.logical(result) ||
      is.null(result) ||
      inherits(result, "gen_sample") ||
      inherits(result, "SequenceGraph")
  )
}

simple_parts_list <- function() {
  list(
    c(`part-a` = "AAAA", `part-b` = "TAAT", `part-c` = "CAAC"),
    c(`part-d` = "ATGA", `part-e` = "TGTT", `part-f` = "TGCT")
  )
}

first_sequence_node <- function(graph_dict) {
  sequence_node_indexes <- which(vapply(
    graph_dict$nodes,
    function(node) node$sequence_end > node$sequence_start,
    logical(1)
  ))
  expect_gt(length(sequence_node_indexes), 0)
  graph_dict$nodes[[sequence_node_indexes[[1]]]]
}

test_that("Repository initializes workspace and returns correct paths", {
  repo <- setup_repository()
  expect_s3_class(repo, "gen_repository")
  expect_true(dir.exists(repo$gen_dir))
  expect_match(repo$db_path, "default\\.db$")
})

test_that("HashId constructor works", {
  hash_id <- HashId("abc123")
  expect_s3_class(hash_id, "gen_hash_id")
})

test_that("FASTA import and export work", {
  repo <- setup_repository()
  output_fasta <- tempfile(fileext = ".fa")

  imported <- repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")
  exported_path <- repo$export_fasta(output_fasta, sample = "sample-a")

  expect_s3_class(imported, "gen_sample")
  expect_length(imported, 1)
  expect_s3_class(imported[[1]], "SequenceGraph")
  expect_equal(normalizePath(exported_path, mustWork = FALSE), normalizePath(output_fasta, mustWork = FALSE))
  expect_equal(readLines(output_fasta)[[2]], "ATCGATCGATCGATCGATCGGGAACACACAGAGA")
})

test_that("successful R imports create clean Dolt history commits", {
  repo <- setup_repository()
  history_before <- repo$query(
    "SELECT commit_hash FROM dolt_log ORDER BY date DESC"
  )

  repo$import_fasta(
    fixture_path("simple.fa"),
    sample = "sample-a",
    collection = "test-collection"
  )

  history_after <- repo$query(
    "SELECT commit_hash FROM dolt_log ORDER BY date DESC"
  )
  status_after <- repo$query(
    "SELECT table_name, staged, status FROM dolt_status"
  )
  expect_length(history_after, length(history_before) + 1)
  expect_length(status_after, 0)
})

test_that("GFA and GenBank import/export work", {
  repo <- setup_repository()
  gfa_out <- tempfile(fileext = ".gfa")
  gb_out  <- tempfile(fileext = ".gb")

  expect_s3_class(
    repo$import_gfa(fixture_path("simple.gfa"), sample = "sample-a"),
    "SequenceGraph"
  )
  expect_binding_result(try(repo$export_gfa(gfa_out, sample = "sample-a"), silent = TRUE))

  expect_binding_result(try(
    repo$import_genbank(fixture_path("geneious_genbank", "insertion.gb"), sample = "sample-a"),
    silent = TRUE
  ))
  expect_binding_result(try(repo$export_genbank(gb_out, sample = "sample-a"), silent = TRUE))
})

test_that("library import works from files and in-memory parts", {
  repo <- setup_repository()

  expect_binding_result(try(repo$import_library_files(
    library_name = "library-from-files",
    parts   = fixture_path("affix_parts.fa"),
    library = fixture_path("affix_layout.csv"),
    sample  = "sample-a"
  ), silent = TRUE))

  expect_binding_result(try(repo$import_library(
    library_name = "library-from-memory",
    parts_list   = simple_parts_list(),
    sample       = "sample-a"
  ), silent = TRUE))
})

test_that("sequence update bindings work", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")

  expect_binding_result(try(repo$update_with_fasta(
    fixture_path("aa.fa"),
    sample = "sample-a", new_sample = "from-fasta",
    region_name = "m123:3-5"
  ), silent = TRUE))

  expect_binding_result(try(repo$update_with_sequence(
    sequence = "AAAAAAAA",
    sample = "sample-a", new_sample = "from-sequence",
    region_name = "m123:2-5"
  ), silent = TRUE))
})

test_that("graph-derived update bindings work", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")

  expect_binding_result(try(repo$update_with_gfa(
    fixture_path("path-diff.gfa"),
    sample = "sample-a", new_sample = "from-gfa"
  ), silent = TRUE))

  expect_binding_result(try(repo$update_with_vcf(
    fixture_path("simple.vcf"),
    reference = "sample-a"
  ), silent = TRUE))

  repo2 <- setup_repository()
  expect_binding_result(try(
    repo2$import_gfa(fixture_path("chr22_het.gfa"), sample = ""),
    silent = TRUE
  ))
  expect_binding_result(try(repo2$update_with_gaf(
    fixture_path("chr22_het.gaf"),
    csv    = fixture_path("chr22_insert.csv"),
    sample = "child"
  ), silent = TRUE))
})

test_that("GenBank and library update bindings work", {
  repo <- setup_repository()

  expect_binding_result(try(
    repo$import_genbank(fixture_path("geneious_genbank", "insertion.gb"), sample = "sample-a"),
    silent = TRUE
  ))
  expect_binding_result(try(repo$update_with_genbank(
    fixture_path("geneious_genbank", "multiple_insertions_deletions.gb"),
    sample = "sample-a", create_missing = TRUE
  ), silent = TRUE))

  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")

  expect_binding_result(try(repo$update_with_library_files(
    sample = "sample-a", new_sample = "library-files-child",
    path_name = "m123:7-20",
    library = fixture_path("combinatorial_design.csv"),
    parts   = fixture_path("parts.fa")
  ), silent = TRUE))

  expect_binding_result(try(update_with_library(
    repo,
    sample = "sample-a",
    new_sample_name = "library-memory-child",
    path_name = "m123:7-20",
    parts_list = simple_parts_list()
  ), silent = TRUE))
})

test_that("graph operation bindings work", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")

  chunked <- repo$derive_chunks(
    collection = NULL, sample = "sample-a", new_sample = "chunked",
    region = "m123", backbone = NULL,
    breakpoints = integer(0), chunk_size = 10L
  )
  expect_s3_class(chunked, "gen_sample")
  expect_true(length(chunked) >= 1)

  expect_binding_result(try(repo$derive_subgraph(
    collection = NULL, sample = "sample-a", new_sample = "subgraph",
    region = "m123:3-12", backbone = NULL
  ), silent = TRUE))

  expect_binding_result(try(
    stitch(repo, bgs = chunked$block_groups, new_sample = "stitched", new_region = "m123.stitched"),
    silent = TRUE
  ))
})

test_that("repository inspection and graph controller work", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")

  groups <- repo$get_sequence_graphs()
  expect_length(groups, 1)
  expect_s3_class(groups[[1]], "SequenceGraph")

  group_by_id <- repo$get_sequence_graph_by_id(groups[[1]]$id())
  expect_equal(group_by_id$name(), groups[[1]]$name())

  rows <- repo$query("SELECT name FROM block_groups ORDER BY name")
  expect_equal(rows[[1]][[1]], "m123")

  repo$execute("CREATE TABLE IF NOT EXISTS r_binding_smoke (id INTEGER)")
  repo$execute("INSERT INTO r_binding_smoke (id) VALUES (1)")
  expect_equal(repo$query("SELECT id FROM r_binding_smoke")[[1]][[1]], 1)

  graph_dict <- groups[[1]]$to_dict()
  expect_true(length(graph_dict$nodes) >= 1)
  expect_true(length(graph_dict$edges) >= 1)

  node <- first_sequence_node(graph_dict)
  expect_true(nzchar(get_node_sequence(repo, node)))

  controller <- GenPlot(groups[[1]]$db_path(), groups[[1]]$id(), rows = 12, cols = 40)
  expect_s3_class(controller, "gen_plot")
  controller$set_detail("minimal")
  viewport <- controller$render_png_viewport(40, 12, 1)
  expect_true(is_png(viewport))
  cell <- controller$png_cell_size(1)
  expect_equal(png_size(viewport), c(40 * cell[[1]], 12 * cell[[2]]))
  expect_type(controller$handle_click(2, 2), "logical")
  controller$zoom_in()
  controller$zoom_out()
  before_pan <- controller$render_png_viewport(40, 12, 1)
  controller$move_by(1, 0)
  after_pan <- controller$render_png_viewport(40, 12, 1)
  expect_true(is_png(after_pan))
  expect_false(identical(before_pan, after_pan))

  controller_copy <- controller$clone()
  expect_identical(
    controller_copy$render_png_viewport(40, 12, 1),
    controller$render_png_viewport(40, 12, 1)
  )
  copy_pre_move <- controller_copy$render_png_viewport(40, 12, 1)
  controller_copy$move_by(1, 0)
  expect_identical(controller$render_png_viewport(40, 12, 1), after_pan)
  expect_false(identical(controller_copy$render_png_viewport(40, 12, 1), copy_pre_move))

  via_plot <- plot(groups[[1]], rows = 10, cols = 30)
  expect_s3_class(via_plot, "gen_plot")
  expect_equal(
    png_size(via_plot$render_png_viewport(30, 10, 1)),
    c(30 * cell[[1]], 10 * cell[[2]])
  )

  default_plot <- plot(groups[[1]])
  expect_equal(default_plot$rows, 24L)
  expect_equal(default_plot$cols, 42L)
  expect_equal(default_plot$scale, 2L)
})

test_that("scaled PNGs keep the cell grid and double the pixels", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")
  sequence_graph <- repo$get_sequence_graphs()[[1]]
  plot <- GenPlot(sequence_graph$db_path(), sequence_graph$id(), rows = 12, cols = 40)

  one <- plot$render_png_viewport(40, 12, 1)
  two <- plot$render_png_viewport(40, 12, 2)
  expect_true(is_png(one))
  expect_true(is_png(two))
  expect_equal(png_size(two), 2 * png_size(one))
  expect_equal(plot$png_cell_size(2), 2 * plot$png_cell_size(1))

  cropped <- plot$render_png(40, 12, 1)
  expect_true(is_png(cropped))
  expect_lte(png_size(cropped)[[1]], png_size(one)[[1]])

  expect_match(plot$background_color(), "^#[0-9a-f]{6}$")
  regions <- plot$hit_regions(40, 12)
  expect_type(regions, "list")
  for (region in regions) {
    expect_true(all(c("x", "y", "kind") %in% names(region)))
    expect_true(region$kind %in% c("node", "wormhole"))
  }

  expect_error(plot$render_png(40, 12, 0), "scale")
  expect_error(plot$render_png(40, 12, 9), "scale")
  expect_error(GenPlot(sequence_graph$db_path(), sequence_graph$id(), scale = 2.5), "scale")
})

test_that("plots are written to disk and knit as a base64 image", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")
  sequence_graph <- repo$get_sequence_graphs()[[1]]
  plot <- plot(sequence_graph, rows = 12, cols = 40, scale = 2)
  expect_equal(plot$scale, 2L)

  path <- tempfile(fileext = ".png")
  expect_identical(plot$save_png(path), path)
  written <- readBin(path, "raw", n = file.size(path))
  expect_true(is_png(written))
  expect_identical(written, plot$render_png(40, 12, 2))

  html <- as.character(genr:::knit_print.gen_plot(plot))
  expect_match(html, "<img src=\"data:image/png;base64,", fixed = TRUE)
  expect_match(html, "alt=\"Gen sequence graph plot", fixed = TRUE)
  expect_false(grepl("<script", html, fixed = TRUE))
  size <- png_size(written) / 2
  expect_match(html, sprintf("width=\"%d\" height=\"%d\"", size[[1]], size[[2]]), fixed = TRUE)
  base64 <- sub('.*base64,([^"]+)".*', "\\1", html)
  expect_identical(jsonlite::base64_dec(base64), written)

  expect_output(print(plot), "gen_plot 40 x 12 cells, scale 2")
})

test_that("Shiny browser operations round trip through the Rust controller", {
  skip_if_not_installed("anyhtmlwidget")
  skip_if_not_installed("shiny")

  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")
  sequence_graph <- repo$get_sequence_graphs()[[1]]
  controller <- GenPlot(
    sequence_graph$db_path(),
    sequence_graph$id(),
    rows = 12,
    cols = 40
  )
  initial_png <- controller$render_png_viewport()
  initial_uri <- genr:::.png_data_uri(initial_png)

  app <- shiny::shinyApp(
    shiny::fluidPage(gen_shiny_ui("graph")),
    function(input, output, session) {
      gen_shiny_server("graph", controller)
    }
  )
  sent <- new.env(parent = emptyenv())
  sent$messages <- list()
  test_session <- shiny::MockShinySession$new()
  test_session$sendCustomMessage <- function(type, message) {
    sent$messages <- c(sent$messages, list(list(type = type, message = message)))
  }

  shiny::testServer(app, session = test_session, {
    session$flushReact()
    sent$messages <- list()
    session$setInputs(
      `graph-anyhtmlwidget_on_save_changes` = list(
        op = list(type = "pan", dx = 1L, dy = 0L),
        seq = 1L
      )
    )
    session$flushReact()
  })

  view_messages <- Filter(
    function(message) identical(message$type, "graph-genplot_view"),
    sent$messages
  )
  expect_length(view_messages, 1)
  view <- view_messages[[1]]$message
  expect_match(view$viewport_png, "^data:image/png;base64,")
  expect_false(identical(view$viewport_png, initial_uri))
  expect_equal(view$ack, 1L)
  expect_type(view$hit_regions, "list")
  expect_equal(view$cell_width, controller$png_cell_size(controller$scale)[[1]])
  expect_equal(view$cell_height, controller$png_cell_size(controller$scale)[[2]])
  expect_match(view$background_color, "^#[0-9a-f]{6}$")

  # `gen_shiny_server` operates on a clone (`session_plot`), so the Shiny
  # interaction above must not have mutated the original controller.
  expect_identical(controller$render_png_viewport(), initial_png)
})

test_that("raw SQL execute/query and sequence graph object API work", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")

  repo$execute("CREATE TABLE IF NOT EXISTS low_level_smoke (id INTEGER)")
  repo$execute("INSERT INTO low_level_smoke (id) VALUES (7)")
  expect_equal(repo$query("SELECT id FROM low_level_smoke")[[1]][[1]], 7)

  groups <- repo$get_sequence_graphs()
  expect_length(groups, 1)

  sg <- groups[[1]]
  expect_equal(sg$name(), "m123")

  graph_dict <- sg$to_dict()
  expect_true(length(graph_dict$nodes) >= 1)

  node <- first_sequence_node(graph_dict)
  expect_true(nzchar(repo$get_node_sequence(node$node_id, node$sequence_start, node$sequence_end)))

  controller <- repo$create_graph_controller(sg$id(), "normal")
  png <- controller$render_png_viewport(40L, 12L, 1L)
  expect_true(is_png(png))
  expect_type(controller$handle_click(1L, 1L), "logical")
})

test_that("variant routing positions remain usable by the R graph widget", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")
  repo$update_with_vcf(
    fixture_path("simple.vcf"),
    reference = "sample-a"
  )

  variant_graphs <- Filter(
    function(graph) graph$sample_name() == "unknown",
    repo$get_sequence_graphs()
  )
  expect_length(variant_graphs, 1)
  graph <- variant_graphs[[1]]
  graph_dict <- graph$to_dict()
  expect_true(any(vapply(
    graph_dict$nodes,
    function(node) {
      node$sequence_start > 0 && node$sequence_start == node$sequence_end
    },
    logical(1)
  )))
  expect_true(any(vapply(
    graph_dict$edges,
    function(edge) {
      (edge$source$sequence_start > 0 &&
        edge$source$sequence_start == edge$source$sequence_end) ||
        (edge$target$sequence_start > 0 &&
          edge$target$sequence_start == edge$target$sequence_end)
    },
    logical(1)
  )))

  controller <- repo$create_graph_controller(graph$id(), "full")
  png <- controller$render_png_viewport(100L, 48L, 1L)
  expect_true(is_png(png))
  expect_equal(png_size(png), c(100, 48) * controller$png_cell_size(1L))
})
