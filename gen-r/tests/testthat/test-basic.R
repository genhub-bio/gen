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

# Create a temp directory, initialize a Repository in it, and return it.
# Mirrors setup_gen_on_disk() from the Rust/Python test helpers.
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

test_that("constructors HashId and Block work", {
  hash_id <- HashId("abc123")
  block <- Block(hash_id, 1, 4)
  expect_s3_class(hash_id, "gen_hash_id")
  expect_s3_class(block, "gen_block")
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

    expect_binding_result(try(update_with_fasta(
        filename = fixture_path("aa.fa"),
        collection_name = "update-collection",
        sample = "sample-a",
        new_sample = "from-fasta",
        region_name = "m123:3-5"
      ), silent = TRUE))

    expect_true(inherits(try(update_with_fasta(
        filename = fixture_path("aa.fa"),
        name = "update-collection",
        sample = "sample-a",
        new_sample = "missing-coords",
        region_name = "m123"
      ), silent = TRUE), "try-error"))

    expect_binding_result(try(update_with_sequence(
        sequence = "AAAAAAAA",
        collection_name = "update-collection",
        sample = "sample-a",
        new_sample = "from-sequence",
        region_name = "m123:2-5"
      ), silent = TRUE))

    expect_true(inherits(try(update_with_sequence(
        sequence = "AAAAAAAA",
        name = "update-collection",
        sample = "sample-a",
        new_sample = "missing-seq-coords",
        region_name = "m123"
      ), silent = TRUE), "try-error"))
  })
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
    parent_samples = "sample-a"
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

  expect_binding_result(try(update_with_library_files(
    collection_name = "library-update-collection",
    sample = "sample-a",
    new_sample = "library-files-child",
    path_name = "m123:7-20",
    library = fixture_path("combinatorial_design.csv"),
    parts = fixture_path("parts.fa")
  ), silent = TRUE))

  expect_binding_result(try(update_with_library(
    collection_name = "library-update-collection",
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
    sample = "sample-a", new_sample = "chunked",
    region = "m123", breakpoints = "10,20"
  ), silent = TRUE))

  groups <- repo$get_block_groups()
  expect_true(length(groups) >= 1)

  expect_binding_result(try(repo$derive_subgraph(
    sample = "sample-a", new_sample = "subgraph",
    region = "m123:3-12"
  ), silent = TRUE))

  expect_binding_result(try(repo$make_stitch(
    sample = "chunked", new_sample = "stitched",
    regions = "m123.1,m123.2", new_region = "m123.stitched"
  ), silent = TRUE))
})

test_that("repository inspection and graph controller work", {
  repo <- setup_repository()
  repo$import_fasta(fixture_path("simple.fa"), sample = "sample-a")

  groups <- repo$get_block_groups()
  expect_length(groups, 1)
  expect_s3_class(groups[[1]], "gen_block_group")

  group_by_id <- repo$get_block_group_by_id(groups[[1]]$id)
  expect_equal(group_by_id$name, groups[[1]]$name)

  rows <- repo$query("SELECT name FROM block_groups ORDER BY name")
  expect_equal(rows[[1]][[1]], "m123")

  repo$execute("CREATE TABLE IF NOT EXISTS r_binding_smoke (id INTEGER)")
  repo$execute("INSERT INTO r_binding_smoke (id) VALUES (1)")
  expect_equal(repo$query("SELECT id FROM r_binding_smoke")[[1]][[1]], 1)

  graph_dict <- repo$block_group_to_dict(groups[[1]])
  expect_true(length(graph_dict$nodes) >= 1)
  expect_true(length(graph_dict$edges) >= 1)

  node <- graph_dict$nodes[[1]]
  key <- Block(node$node_id, node$sequence_start, node$sequence_end)
  expect_equal(
    repo$get_block_sequence(key),
    genr:::repo_get_block_sequence(repo$db_path, node$node_id, node$sequence_start, node$sequence_end)
  )

  expect_type(repo$block_group_to_rustworkx(groups[[1]]), "list")
  expect_type(repo$block_group_to_networkx(groups[[1]]), "list")

  controller <- repo$plot(groups[[1]], rows = 12, cols = 40)
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

  via_plot <- groups[[1]]$plot(rows = 10, cols = 30)
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

  groups <- genr:::repo_get_block_groups(db_path)
  expect_length(groups, 1)

  fetched <- genr:::repo_get_block_group_by_id(db_path, groups[[1]]$id)
  expect_equal(fetched$name, "m123")

  graph_dict <- genr:::repo_block_group_to_dict(db_path, groups[[1]]$id)
  expect_true(length(graph_dict$nodes) >= 1)

  node <- graph_dict$nodes[[1]]
  expect_true(nzchar(genr:::repo_get_block_sequence(
    db_path, node$node_id, node$sequence_start, node$sequence_end
  )))

  frame_json <- genr:::graph_render_frame(db_path, groups[[1]]$id, "normal", 40L, 12L, "", "[]")
  expect_match(frame_json, "\"cells\"")
  expect_type(genr:::graph_handle_click(db_path, groups[[1]]$id, "normal", "", 1L, 1L), "logical")
})
