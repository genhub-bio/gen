.onLoad <- function(libname, pkgname) {
  if (requireNamespace("knitr", quietly = TRUE)) {
    registerS3method("knit_print", "gen_plot", knit_print.gen_plot,
                     envir = asNamespace("knitr"))
  }
}

.genplot_widget <- function(frame_json) {
  # This optional install lets us test R functions without having to install the jupyter widget
  if (!requireNamespace("anyhtmlwidget", quietly = TRUE)) {
    stop(
      "Interactive widget rendering requires the optional 'anyhtmlwidget' package. ",
      "Install it before printing or knitting a gen_plot.",
      call. = FALSE
    )
  }
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
