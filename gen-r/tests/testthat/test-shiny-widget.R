test_that("Shiny widget operations update the plot controller", {
  calls <- new.env(parent = emptyenv())
  calls$values <- list()
  plot <- new.env(parent = emptyenv())
  class(plot) <- "gen_plot"
  plot$zoom_in <- function() calls$values <- c(calls$values, "zoom_in")
  plot$zoom_out <- function() calls$values <- c(calls$values, "zoom_out")
  plot$move_by <- function(dx, dy) calls$values <- c(calls$values, list(c(dx, dy)))
  plot$handle_click <- function(col, row) {
    calls$values <- c(calls$values, list(c(col, row)))
    TRUE
  }

  expect_true(genr:::.apply_genplot_operation(plot, list(type = "zoom", direction = "in")))
  expect_true(genr:::.apply_genplot_operation(plot, list(type = "zoom", direction = "out")))
  expect_true(genr:::.apply_genplot_operation(plot, list(type = "pan", dx = 3L, dy = -2L)))
  expect_true(genr:::.apply_genplot_operation(plot, list(type = "mouse_click", col = 4L, row = 5L)))
  expect_false(genr:::.apply_genplot_operation(plot, list(type = "snapshot")))

  expect_equal(calls$values, list("zoom_in", "zoom_out", c(3L, -2L), c(4L, 5L)))
})

test_that("Shiny widget operations default missing coordinates to zero", {
  plot <- new.env(parent = emptyenv())
  class(plot) <- "gen_plot"
  plot$move_by <- function(dx, dy) expect_equal(c(dx, dy), c(0L, 0L))

  expect_true(genr:::.apply_genplot_operation(plot, list(type = "pan")))
})

test_that("gen_app creates a standalone Shiny application", {
  plot <- new.env(parent = emptyenv())
  class(plot) <- "gen_plot"

  app <- gen_app(plot, title = "Test graph")

  expect_s3_class(app, "shiny.appobj")
})

test_that("Shiny viewer bundle draws PNG frames and has no canvas painter", {
  bundle_path <- system.file("widget", "genplot.esm.js", package = "genr")
  expect_true(nzchar(bundle_path))
  bundle <- paste(readLines(bundle_path, warn = FALSE), collapse = "\n")

  expect_true(grepl("top: 8px; left: 8px", bundle, fixed = TRUE))
  expect_true(grepl("viewport_png", bundle, fixed = TRUE))
  expect_false(grepl("fillText", bundle, fixed = TRUE))
  expect_false(file.exists(system.file("widget", "genplot-shiny.esm.js", package = "genr")))
})

test_that("PNG helpers read dimensions and encode data URIs", {
  png <- as.raw(c(
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x01, 0x2C, 0x00, 0x00, 0x00, 0x64
  ))

  expect_equal(unname(genr:::.png_dimensions(png)), c(300L, 100L))
  expect_match(genr:::.png_data_uri(png), "^data:image/png;base64,iVBORw0KGgo")
  expect_equal(genr:::.check_png_scale(2), 2L)
  expect_error(genr:::.check_png_scale(0), "from 1 to 8")
})

test_that("view rejects unsupported objects before launching Shiny", {
  expect_error(view(list()), "SequenceGraph or gen_plot")
})
