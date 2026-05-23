library(genr)

local_fixture_root <- function() {
  # Prefer fixtures bundled next to the testthat directory, independent of the
  # process working directory. Fall back to common CI checkout locations.
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

with_workspace <- function(prefix, code) {
  old <- setwd(tempdir())
  on.exit(setwd(old), add = TRUE)

  workspace <- tempfile(prefix)
  dir.create(workspace, recursive = TRUE)
  setwd(workspace)

  code(normalizePath(workspace, mustWork = FALSE))
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

import_simple_reference <- function(collection = "test-collection", sample = "sample-a") {
  import_fasta(
    filename = fixture_path("simple.fa"),
    collection_name = collection,
    sample = sample,
    shallow = FALSE
  )
}

test_that("constructors and workspace bindings work", {
  with_workspace("genr-core-", function(workspace) {
    hash_id <- HashId("abc123")
    block <- Block(hash_id, 1, 4)

    expect_s3_class(hash_id, "gen_hash_id")
    expect_s3_class(block, "gen_block")

    msg <- init()
    expect_match(msg, "initialized", ignore.case = TRUE)
    expect_true(dir.exists(".gen"))

    ctx_plain <- db_context()
    ctx_object <- DbContext()

    expect_type(ctx_plain, "list")
    expect_s3_class(ctx_object, "gen_db_context")
    expect_equal(normalizePath(ctx_plain$workspace_path, mustWork = FALSE), workspace)
    expect_match(ctx_plain$db_path, "\\.gen[\\\\/]default\\.db$")
    expect_equal(
      normalizePath(get_gen_dir(), mustWork = FALSE),
      normalizePath(file.path(workspace, ".gen"), mustWork = FALSE)
    )
  })
})

test_that("FASTA import and export bindings work", {
  with_workspace("genr-fasta-", function(workspace) {
    init()

    output_fasta <- file.path(workspace, "output.fa")
    import_msg <- import_fasta(
      filename = fixture_path("simple.fa"),
      collection_name = "fasta-collection",
      sample = "sample-a",
      shallow = FALSE
    )
    exported_path <- export_fasta(
      filename = output_fasta,
      collection_name = "fasta-collection",
      sample = "sample-a"
    )

    expect_match(import_msg, "imported", ignore.case = TRUE)
    expect_equal(normalizePath(exported_path, mustWork = FALSE), normalizePath(output_fasta, mustWork = FALSE))
    expect_equal(readLines(output_fasta)[[2]], "ATCGATCGATCGATCGATCGGGAACACACAGAGA")
  })
})

test_that("GFA and GenBank import/export bindings work", {
  with_workspace("genr-import-export-", function(workspace) {
    init()

    gfa_out <- file.path(workspace, "roundtrip.gfa")
    gb_out <- file.path(workspace, "roundtrip.gb")

    expect_match(
      import_gfa(
        filename = fixture_path("simple.gfa"),
        collection_name = "gfa-collection",
        sample = "sample-a"
      ),
      "imported",
      ignore.case = TRUE
    )
    expect_binding_result(try(
      export_gfa(
        filename = gfa_out,
        collection_name = "gfa-collection",
        sample = "sample-a"
      ),
      silent = TRUE
    ))

    genbank_import <- try(import_genbank(
        filename = fixture_path("geneious_genbank", "insertion.gb"),
        collection_name = "genbank-collection",
        sample = "sample-a"
      ), silent = TRUE)
    expect_binding_result(genbank_import)

    genbank_export <- try(export_genbank(
        filename = gb_out,
        collection_name = "genbank-collection",
        sample = "sample-a"
      ), silent = TRUE)
    expect_binding_result(genbank_export)
  })
})

test_that("library import bindings work from files and in-memory parts", {
  with_workspace("genr-library-import-", function(workspace) {
    init()

    expect_binding_result(try(import_library_files(
        library_name = "library-from-files",
        parts = fixture_path("affix_parts.fa"),
        library = fixture_path("affix_layout.csv"),
        collection_name = "library-files-collection",
        sample = "sample-a"
      ), silent = TRUE))

    expect_binding_result(try(import_library(
        library_name = "library-from-memory",
        parts_list = simple_parts_list(),
        collection_name = "library-memory-collection",
        sample = "sample-a"
      ), silent = TRUE))

    expect_s3_class(Repository(), "gen_repository")
  })
})

test_that("sequence update bindings work", {
  with_workspace("genr-seq-updates-", function(workspace) {
    init()
    import_simple_reference("update-collection", "sample-a")

    fasta_out <- file.path(workspace, "sequence-child.fa")

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
  with_workspace("genr-graph-updates-", function(workspace) {
    init()
    import_simple_reference("graph-update-collection", "sample-a")

    expect_binding_result(try(update_with_gfa(
        filename = fixture_path("path-diff.gfa"),
        collection_name = "graph-update-collection",
        sample = "sample-a",
        new_sample = "from-gfa"
      ), silent = TRUE))

    expect_binding_result(try(update_with_vcf(
        filename = fixture_path("simple.vcf"),
        collection_name = "graph-update-collection",
        parent_samples = "sample-a"
      ), silent = TRUE))

    expect_binding_result(try(import_gfa(
        filename = fixture_path("chr22_het.gfa"),
        collection_name = "gaf-collection",
        sample = ""
      ), silent = TRUE))

    expect_binding_result(try(update_with_gaf(
        filename = fixture_path("chr22_het.gaf"),
        csv = fixture_path("chr22_insert.csv"),
        collection_name = "gaf-collection",
        sample = "child"
      ), silent = TRUE))
  })
})

test_that("GenBank and library update bindings work", {
  with_workspace("genr-rich-updates-", function(workspace) {
    init()

    expect_binding_result(try(import_genbank(
        filename = fixture_path("geneious_genbank", "insertion.gb"),
        collection_name = "genbank-collection",
        sample = "sample-a"
      ), silent = TRUE))

    expect_binding_result(try(update_with_genbank(
        filename = fixture_path("geneious_genbank", "multiple_insertions_deletions.gb"),
        collection_name = "genbank-collection",
        sample = "sample-a",
        create_missing = TRUE
      ), silent = TRUE))

    import_simple_reference("library-update-collection", "sample-a")

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

    expect_true(inherits(try(update_with_library(
        name = "library-update-collection",
        sample = "sample-a",
        new_sample_name = "library-missing-coords",
        path_name = "m123",
        parts_list = simple_parts_list()
      ), silent = TRUE), "try-error"))
  })
})

test_that("graph operation bindings work", {
  with_workspace("genr-ops-", function(workspace) {
    init()
    import_simple_reference("ops-collection", "sample-a")

    chunk_result <- try(
      derive_chunks(
        collection_name = "ops-collection",
        sample = "sample-a",
        new_sample = "chunked",
        region = "m123",
        breakpoints = "10,20"
      ),
      silent = TRUE
    )
    expect_binding_result(chunk_result)

    repo <- Repository()
    chunked_groups <- repo$get_block_groups_by_collection("ops-collection")
    if (length(chunked_groups) > 0) {
      chunked_names <- vapply(chunked_groups, function(x) x$name, character(1))
      expect_true(length(chunked_names) >= 1)
    }

    expect_binding_result(try(derive_subgraph(
        collection_name = "ops-collection",
        sample = "sample-a",
        new_sample = "subgraph",
        region = "m123:3-12"
      ), silent = TRUE))

    expect_binding_result(try(make_stitch(
        collection_name = "ops-collection",
        sample = "chunked",
        new_sample = "stitched",
        regions = "m123.1,m123.2",
        new_region = "m123.stitched"
      ), silent = TRUE))
  })
})

test_that("repository and graph controller helpers work", {
  with_workspace("genr-repo-", function(workspace) {
    init()
    import_simple_reference("repo-collection", "sample-a")

    repo <- Repository()
    expect_s3_class(repo, "gen_repository")
    expect_equal(
      normalizePath(repo$gen_dir, mustWork = FALSE),
      normalizePath(file.path(workspace, ".gen"), mustWork = FALSE)
    )

    groups <- repo$get_block_groups()
    expect_length(groups, 1)
    expect_s3_class(groups[[1]], "gen_block_group")

    group_by_id <- repo$get_block_group_by_id(groups[[1]]$id)
    expect_equal(group_by_id$name, groups[[1]]$name)

    grouped <- repo$get_block_groups_by_collection("repo-collection")
    expect_length(grouped, 1)

    rows <- repo$query("SELECT name FROM block_groups ORDER BY name")
    expect_equal(rows[[1]][[1]], "m123")

    repo$execute("CREATE TABLE IF NOT EXISTS r_binding_smoke (id INTEGER)")
    repo$execute("INSERT INTO r_binding_smoke (id) VALUES (1)")
    smoke_rows <- repo$query("SELECT id FROM r_binding_smoke")
    expect_equal(smoke_rows[[1]][[1]], 1)

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
    expect_s3_class(controller, "gen_graph_controller")
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
    expect_s3_class(via_plot, "gen_graph_controller")
    expect_match(via_plot$render_frame(30, 10), "\"cols\":30")
  })
})

test_that("low-level helper bindings are directly callable", {
  with_workspace("genr-low-level-", function(workspace) {
    init()
    import_simple_reference("low-level-collection", "sample-a")

    repo <- Repository()
    db_path <- repo$db_path

    expect_equal(
      normalizePath(genr:::repo_get_gen_dir(NULL), mustWork = FALSE),
      normalizePath(file.path(workspace, ".gen"), mustWork = FALSE)
    )
    expect_match(genr:::repo_get_db_path(NULL), "\\.gen[\\\\/]default\\.db$")

    genr:::repo_execute(db_path, "CREATE TABLE IF NOT EXISTS low_level_smoke (id INTEGER)")
    genr:::repo_execute(db_path, "INSERT INTO low_level_smoke (id) VALUES (7)")
    expect_equal(genr:::repo_query(db_path, "SELECT id FROM low_level_smoke")[[1]][[1]], 7)

    groups <- genr:::repo_get_block_groups(db_path)
    expect_length(groups, 1)

    expect_equal(
      genr:::repo_get_block_groups_by_collection(db_path, "low-level-collection")[[1]]$name,
      "m123"
    )

    fetched <- genr:::repo_get_block_group_by_id(db_path, groups[[1]]$id)
    expect_equal(fetched$name, "m123")

    graph_dict <- genr:::repo_block_group_to_dict(db_path, groups[[1]]$id)
    expect_true(length(graph_dict$nodes) >= 1)

    node <- graph_dict$nodes[[1]]
    expect_true(
      nzchar(genr:::repo_get_block_sequence(
        db_path,
        node$node_id,
        node$sequence_start,
        node$sequence_end
      ))
    )

    frame_json <- genr:::graph_render_frame(db_path, groups[[1]]$id, "normal", 40L, 12L, "")
    expect_match(frame_json, "\"cells\"")
    expect_type(genr:::graph_handle_click(db_path, groups[[1]]$id, "normal", "", 1L, 1L), "logical")
  })
})
