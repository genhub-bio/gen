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
