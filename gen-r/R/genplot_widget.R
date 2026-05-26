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
