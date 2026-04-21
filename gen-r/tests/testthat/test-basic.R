library(genr)

test_that("init creates a Gen workspace", {
  old <- setwd(tempdir())
  on.exit(setwd(old), add = TRUE)

  workspace <- tempfile("genr-workspace-")
  dir.create(workspace, recursive = TRUE)
  setwd(workspace)

  msg <- init()

  expect_match(msg, "initialized", ignore.case = TRUE)
  expect_true(dir.exists(".gen"))
  expect_true(file.exists(file.path(".gen", "changesets")))
})

test_that("db_context resolves default database paths", {
  old <- setwd(tempdir())
  on.exit(setwd(old), add = TRUE)

  workspace <- tempfile("genr-dbctx-")
  dir.create(workspace, recursive = TRUE)
  setwd(workspace)

  init()
  ctx <- db_context()

  expect_type(ctx, "list")
  expect_equal(normalizePath(ctx$workspace_path, mustWork = FALSE), normalizePath(workspace, mustWork = FALSE))
  expect_match(ctx$db_path, "\\.gen/default\\.db$")
})

test_that("get_gen_dir returns the active .gen directory", {
  old <- setwd(tempdir())
  on.exit(setwd(old), add = TRUE)

  workspace <- tempfile("genr-gendir-")
  dir.create(workspace, recursive = TRUE)
  setwd(workspace)

  init()

  gen_dir <- get_gen_dir()

  expect_equal(
    normalizePath(gen_dir, mustWork = FALSE),
    normalizePath(file.path(workspace, ".gen"), mustWork = FALSE)
  )
})

test_that("import_fasta and export_fasta round trip sequence data", {
  old <- setwd(tempdir())
  on.exit(setwd(old), add = TRUE)

  workspace <- tempfile("genr-fasta-")
  dir.create(workspace, recursive = TRUE)
  setwd(workspace)

  init()

  input_fasta <- file.path(workspace, "input.fa")
  output_fasta <- file.path(workspace, "output.fa")
  writeLines(
    c(">demo-seq", "ATCGATCGATCGATCGATCGGGAACACACAGAGA"),
    input_fasta
  )

  import_msg <- import_fasta(
    filename = input_fasta,
    name = "test-collection",
    sample = "sample-a",
    shallow = FALSE
  )
  exported_path <- export_fasta(
    filename = output_fasta,
    name = "test-collection",
    sample = "sample-a"
  )

  expect_match(import_msg, "imported", ignore.case = TRUE)
  expect_equal(normalizePath(exported_path, mustWork = FALSE), normalizePath(output_fasta, mustWork = FALSE))
  expect_true(file.exists(output_fasta))

  exported_lines <- readLines(output_fasta)
  expect_equal(exported_lines[[1]], ">demo-seq")
  expect_equal(exported_lines[[2]], "ATCGATCGATCGATCGATCGGGAACACACAGAGA")
})
