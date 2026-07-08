.onLoad <- function(libname, pkgname) {
  if (requireNamespace("knitr", quietly = TRUE)) {
    registerS3method("knit_print", "gen_plot", knit_print.gen_plot,
                     envir = asNamespace("knitr"))
  }
}

.genplot_widget <- function(frame_json) {
  esm <- paste(
    readLines(system.file("widget/genplot.esm.js", package = "genr"), warn = FALSE),
    collapse = "\n"
  )
  anyhtmlwidget::AnyHtmlWidget$new(
    .esm = esm,
    .mode = "static",
    frame = jsonlite::fromJSON(frame_json, simplifyDataFrame = FALSE)
  )
}

# Rasterize a gen_plot widget to a PNG for output formats (e.g. PDF) that
# can't embed the interactive JS canvas widget directly.
.genplot_snapshot <- function(w) {
  if (!requireNamespace("webshot2", quietly = TRUE) || !requireNamespace("htmlwidgets", quietly = TRUE)) {
    stop(
      "Rendering gen_plot objects to non-HTML output (e.g. PDF) requires the ",
      "'webshot2' and 'htmlwidgets' packages. Install them with ",
      "install.packages(c('webshot2', 'htmlwidgets')).",
      call. = FALSE
    )
  }
  html_path <- tempfile(fileext = ".html")
  png_path <- tempfile(fileext = ".png")
  htmlwidgets::saveWidget(w$.get_htmlwidget(), html_path, selfcontained = TRUE)
  invisible(capture.output(
    suppressMessages(webshot2::webshot(html_path, png_path, selector = "canvas", zoom = 2))
  ))
  knitr::include_graphics(png_path)
}
