.onLoad <- function(libname, pkgname) {
  if (requireNamespace("knitr", quietly = TRUE)) {
    registerS3method("knit_print", "gen_plot", knit_print.gen_plot,
                     envir = asNamespace("knitr"))
  }
}

# What the browser needs to draw one frame of the viewer: the viewport PNG as a base64 data URI
# (the shared widget script decodes it exactly like the bytes Jupyter sends), the clickable
# cells, and the pixel size of one cell. Built after every operation and sent as one message.
.genplot_view_state <- function(plot) {
  png <- plot$render_png_viewport(plot$cols, plot$rows, plot$scale)
  cell_size <- plot$png_cell_size(plot$scale)
  list(
    viewport_png = .png_data_uri(png),
    hit_regions = plot$hit_regions(plot$cols, plot$rows),
    cell_width = cell_size[[1]],
    cell_height = cell_size[[2]],
    background_color = plot$background_color()
  )
}

.genplot_shiny_widget <- function(plot, view_message) {
  esm <- paste(
    readLines(system.file("widget/genplot.esm.js", package = "genr"), warn = FALSE),
    collapse = "\n"
  )
  view <- .genplot_view_state(plot)
  anyhtmlwidget::AnyHtmlWidget$new(
    .esm = esm,
    .mode = "static",
    viewport_png = view$viewport_png,
    hit_regions = view$hit_regions,
    cell_width = view$cell_width,
    cell_height = view$cell_height,
    background_color = view$background_color,
    view_message = view_message,
    scale = plot$scale,
    cols = plot$cols,
    rows = plot$rows,
    op = NULL,
    seq = 0L,
    ack = 0L,
    page_count = 1L,
    page_index = 0L
  )
}

#' Shiny UI for an interactive Gen graph
#'
#' @param id Shiny module identifier.
#' @param width,height CSS dimensions for the widget.
#' @return A Shiny output element.
#' @export
gen_shiny_ui <- function(id, width = "100%", height = "100%") {
  anyhtmlwidget::widgetUI(id, width = width, height = height)
}

.apply_genplot_operation <- function(plot, operation) {
  switch(operation$type,
    zoom = {
      if (identical(operation$direction, "in")) plot$zoom_in() else plot$zoom_out()
      TRUE
    },
    pan = {
      plot$move_by(operation$dx %||% 0L, operation$dy %||% 0L)
      TRUE
    },
    mouse_click = {
      plot$handle_click(operation$col %||% 0L, operation$row %||% 0L)
      TRUE
    },
    FALSE
  )
}

# The browser sends one operation at a time as the widget values `op` and `seq`. The server
# answers with a single custom message holding the whole next frame, which the widget script
# applies to its model. It travels under its own message name rather than the widget's
# `anyhtmlwidget_on_change` channel, because that channel would upload every value it delivers
# back to R with the next operation, and a frame carries a whole PNG. `ack` goes last so the
# script can release its one-request-in-flight pan lock even when the image did not change.
.send_genplot_view <- function(session, message_name, view) {
  session$sendCustomMessage(message_name, view)
}

#' Shiny server for an interactive Gen graph
#'
#' @param id Shiny module identifier used by [genr::gen_shiny_ui()].
#' @param plot A `gen_plot` returned by [GenPlot()] or `plot()`. The server
#'   clones this controller so each Shiny session has independent viewport
#'   state.
#' @return The widget's reactive values, invisibly.
#' @export
gen_shiny_server <- function(id, plot) {
  if (!inherits(plot, "gen_plot")) {
    stop("plot must be a gen_plot object", call. = FALSE)
  }

  session_plot <- plot$clone()
  view_message <- shiny::NS(id)("genplot_view")
  widget <- .genplot_shiny_widget(session_plot, view_message)
  state <- anyhtmlwidget::widgetServer(id, widget)
  session <- shiny::getDefaultReactiveDomain()

  shiny::observeEvent(state$seq, {
    operation <- state$op
    shiny::req(operation$type)

    changed <- .apply_genplot_operation(session_plot, operation)

    if (isTRUE(changed)) {
      view <- .genplot_view_state(session_plot)
      view$ack <- state$seq
      for (key in names(view)) {
        widget$.set_value(key, view[[key]], emit_change = FALSE)
      }
      .send_genplot_view(session, view_message, view)
    }
  }, ignoreInit = TRUE)

  invisible(state)
}

#' Create a standalone Shiny app for a Gen graph
#'
#' @param plot A `gen_plot` returned by [GenPlot()] or `plot()`.
#' @param title Character. Application title.
#' @param width,height CSS dimensions for the graph viewer.
#' @return A `shiny.appobj` suitable for [shiny::runApp()].
#' @export
gen_app <- function(
  plot,
  title = "Gen graph viewer",
  width = "100%",
  height = "calc(100vh - 52px)"
) {
  if (!inherits(plot, "gen_plot")) {
    stop("plot must be a gen_plot object", call. = FALSE)
  }

  ui <- shiny::fluidPage(
    shiny::tags$head(
      shiny::tags$style(shiny::HTML(
        "body { overflow: hidden; }
        .genplot-header { margin: 12px 0 8px; }
        .genplot-header h2 { display: inline; font-size: 20px; }
        .genplot-help { float: right; color: #666; padding-top: 4px; }"
      ))
    ),
    shiny::div(
      class = "genplot-header",
      shiny::tags$h2(title),
      shiny::span(
        class = "genplot-help",
        "Drag to pan · use the magnifiers to zoom"
      )
    ),
    gen_shiny_ui("genplot", width = width, height = height)
  )
  server <- function(input, output, session) {
    gen_shiny_server("genplot", plot)
  }
  shiny::shinyApp(ui, server)
}

#' View a Gen graph interactively
#'
#' Accepts either a `SequenceGraph` or an existing `gen_plot`. In RStudio,
#' `viewer = "auto"` opens the Viewer pane; elsewhere it opens a browser.
#'
#' @param x A `SequenceGraph` or `gen_plot`.
#' @param viewer Where to display the app: `"auto"`, `"pane"`, or `"browser"`.
#' @param rows,cols Optional graph dimensions when `x` is a `SequenceGraph`.
#' @param detail Graph detail level when `x` is a `SequenceGraph`.
#' @param colors Annotation colors when `x` is a `SequenceGraph`.
#' @param scale Image pixels per logical pixel when `x` is a `SequenceGraph`; see [GenPlot()].
#' @inheritParams gen_app
#' @param ... Additional arguments passed to [shiny::runApp()] or
#'   [shiny::runGadget()].
#' @return The value returned by Shiny when the application exits.
#' @export
view <- function(
  x,
  viewer = c("auto", "pane", "browser"),
  rows = NULL,
  cols = NULL,
  detail = "normal",
  colors = NULL,
  scale = 2L,
  title = "Gen graph viewer",
  width = "100%",
  height = "calc(100vh - 52px)",
  ...
) {
  viewer <- match.arg(viewer)
  controller <- if (inherits(x, "gen_plot")) {
    x
  } else if (inherits(x, "SequenceGraph")) {
    graphics::plot(x, rows = rows, cols = cols, detail = detail, colors = colors, scale = scale)
  } else {
    stop("x must be a SequenceGraph or gen_plot object", call. = FALSE)
  }
  app <- gen_app(controller, title = title, width = width, height = height)

  if (identical(viewer, "auto")) {
    viewer <- if (identical(Sys.getenv("RSTUDIO"), "1")) "pane" else "browser"
  }
  if (identical(viewer, "pane")) {
    shiny::runGadget(app, viewer = shiny::paneViewer(), ...)
  } else {
    shiny::runApp(app, launch.browser = TRUE, ...)
  }
}
