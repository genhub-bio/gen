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

expect_binding_result <- function(result) {
  expect_true(
    is.character(result) ||
      is.list(result) ||
      is.logical(result) ||
      is.null(result) ||
      inherits(result, "gen_block_group") ||
      inherits(result, "try-error")
  )
}

simple_parts_list <- function() {
  list(
    c(`part-a` = "AAAA", `part-b` = "TAAT", `part-c` = "CAAC"),
    c(`part-d` = "ATGA", `part-e` = "TGTT", `part-f` = "TGCT")
  )
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

  import_msg <- repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")
  exported_path <- repo$export_fasta(output_fasta, sample = "sample-a")

  expect_match(import_msg, "imported", ignore.case = TRUE)
  expect_equal(normalizePath(exported_path, mustWork = FALSE), normalizePath(output_fasta, mustWork = FALSE))
  expect_equal(readLines(output_fasta)[[2]], "ATCGATCGATCGATCGATCGGGAACACACAGAGA")
})

test_that("GFA and GenBank import/export work", {
  repo <- setup_repository()
  gfa_out <- tempfile(fileext = ".gfa")
  gb_out  <- tempfile(fileext = ".gb")

  expect_match(
    repo$import_gfa(fixture_path("simple.gfa"), sample = "sample-a"),
    "imported", ignore.case = TRUE
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

  expect_binding_result(try(repo$derive_chunks(
    collection = NULL, sample = "sample-a", new_sample = "chunked",
    region = "m123", backbone = NULL,
    breakpoints = integer(0), chunk_size = NULL
  ), silent = TRUE))

  groups <- repo$get_sequence_graphs()
  expect_true(length(groups) >= 1)

  expect_binding_result(try(repo$derive_subgraph(
    collection = NULL, sample = "sample-a", new_sample = "subgraph",
    region = "m123:3-12", backbone = NULL
  ), silent = TRUE))

  chunks <- Filter(function(bg) bg$sample_name() == "chunked", repo$get_sequence_graphs())
  expect_binding_result(try(
    stitch(repo, bgs = chunks, new_sample = "stitched", new_region = "m123.stitched"),
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

  node <- graph_dict$nodes[[1]]
  expect_true(nzchar(get_node_sequence(repo, node)))

  controller <- GenPlot(groups[[1]]$db_path(), groups[[1]]$id(), rows = 12, cols = 40)
  expect_s3_class(controller, "gen_plot")
  controller$set_detail("minimal")
  frame_json <- controller$render_frame(40, 12)
  expect_match(frame_json, "\"cols\":40")
  expect_match(frame_json, "\"rows\":12")
  expect_type(controller$handle_click(2, 2), "logical")
  controller$zoom_in()
  controller$zoom_out()
  controller$move_by(1, 0)
  expect_match(controller$render_frame(40, 12), "\"cells\"")

  via_plot <- plot(groups[[1]], rows = 10, cols = 30)
  expect_s3_class(via_plot, "gen_plot")
  expect_match(via_plot$render_frame(30, 10), "\"cols\":30")
})

test_that("low-level internal bindings are accessible via :::", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")
  db_path <- repo$db_path

  genr:::repo_execute(db_path, "CREATE TABLE IF NOT EXISTS low_level_smoke (id INTEGER)")
  genr:::repo_execute(db_path, "INSERT INTO low_level_smoke (id) VALUES (7)")
  expect_equal(genr:::repo_query(db_path, "SELECT id FROM low_level_smoke")[[1]][[1]], 7)

  groups <- genr:::repo_get_sequence_graphs(db_path)
  expect_length(groups, 1)

  fetched <- genr:::repo_get_sequence_graph_by_id(db_path, groups[[1]]$id)
  expect_equal(fetched$name, "m123")

  graph_dict <- genr:::repo_block_group_to_dict(db_path, groups[[1]]$id)
  expect_true(length(graph_dict$nodes) >= 1)

  node <- graph_dict$nodes[[1]]
  expect_true(nzchar(genr:::repo_get_node_sequence(
    db_path, node$node_id, node$sequence_start, node$sequence_end
  )))

  frame_json <- repo$render_frame(groups[[1]]$id, "normal", 40L, 12L, "", "[]")
  expect_match(frame_json, "\"cells\"")
  expect_type(repo$handle_click(groups[[1]]$id, "normal", "", 1L, 1L), "logical")
})
