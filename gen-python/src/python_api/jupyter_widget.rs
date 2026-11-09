use std::collections::HashMap;

pub(crate) use r#gen::views::graph_session::PlotOptions;
use r#gen::views::{
    annotation_track::{AnnotationSpan, annotation_span_from_graph_locus},
    graph_database::GraphDatabase,
    graph_session::{GraphPage, GraphSession, GraphSessionError, PageRef, validated_png_scale},
};
use gen_annotations::projection::AnnotationSegment;
use gen_core::{HashId, Workspace, range::Range};
use gen_models::db::GraphConnection;
use gen_tui::theme::current_theme;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};
use ratatui::{
    buffer::Buffer,
    style::{Color, Modifier},
};
use serde::Serialize;

use crate::python_api::{
    annotation::PyAnnotation, block_group::PySequenceGraph, graph_search::PyGraphLocus,
    position::PyPosition,
};

fn workspace_for_connection(conn: &GraphConnection) -> PyResult<Workspace> {
    r#gen::views::graph_session::workspace_for_connection(conn).map_err(session_error)
}

/// Convert a ratatui `Color` to a CSS hex string.
fn color_to_hex(color: Option<ratatui::style::Color>, default_hex: &str) -> String {
    match color {
        None | Some(Color::Reset) => default_hex.to_string(),
        Some(Color::Rgb(r, g, b)) => format!("#{r:02x}{g:02x}{b:02x}"),
        Some(Color::Black) => "#000000".to_string(),
        Some(Color::Red) => "#cc0000".to_string(),
        Some(Color::Green) => "#00cc00".to_string(),
        Some(Color::Yellow) => "#cccc00".to_string(),
        Some(Color::Blue) => "#0000cc".to_string(),
        Some(Color::Magenta) => "#cc00cc".to_string(),
        Some(Color::Cyan) => "#00cccc".to_string(),
        Some(Color::Gray) => "#888888".to_string(),
        Some(Color::DarkGray) => "#444444".to_string(),
        Some(Color::LightRed) => "#ff5555".to_string(),
        Some(Color::LightGreen) => "#55ff55".to_string(),
        Some(Color::LightYellow) => "#ffff55".to_string(),
        Some(Color::LightBlue) => "#5555ff".to_string(),
        Some(Color::LightMagenta) => "#ff55ff".to_string(),
        Some(Color::LightCyan) => "#55ffff".to_string(),
        Some(Color::White) => "#ffffff".to_string(),
        Some(Color::Indexed(i)) => indexed_to_hex(i),
    }
}

/// Map an ANSI 256-colour index to a hex string.
fn indexed_to_hex(i: u8) -> String {
    // Indices 0-15: standard + bright ANSI colours.
    const STANDARD: [&str; 16] = [
        "#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#c0c0c0",
        "#808080", "#ff0000", "#00ff00", "#ffff00", "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
    ];
    if (i as usize) < STANDARD.len() {
        return STANDARD[i as usize].to_string();
    }
    // Indices 232-255: grayscale ramp (8, 18, 28, … 238).
    if i >= 232 {
        let v = 8u8.saturating_add((i - 232) * 10);
        return format!("#{v:02x}{v:02x}{v:02x}");
    }
    // Indices 16-231: 6×6×6 colour cube.
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let n = i - 16;
    let r = LEVELS[(n / 36) as usize];
    let g = LEVELS[((n / 6) % 6) as usize];
    let b = LEVELS[(n % 6) as usize];
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn is_false(b: &bool) -> bool {
    !b
}

/// Format by which the buffer is to be serialized.
///
/// Only non-empty or non-neutral cells are emitted; `fg`/`bg` are omitted when
/// equal to the frame-level neutral colours; `bold`/`italic`/`underline` are
/// omitted when false.
#[derive(Serialize)]
struct RenderedCell {
    x: u16,
    y: u16,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bg: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    bold: bool,
    #[serde(skip_serializing_if = "is_false")]
    italic: bool,
    #[serde(skip_serializing_if = "is_false")]
    underline: bool,
}

#[derive(Serialize)]
struct RenderedFrame {
    cols: u16,
    rows: u16,
    /// CSS hex colour for the canvas background and neutral edge/text colour
    /// (theme slot 0x00 and 0x05).  Sent per-frame so the frontend always
    /// reflects the active theme without requiring a page reload.
    neutral_fg: String,
    neutral_bg: String,
    cells: Vec<RenderedCell>,
}

fn serialize_buffer(buf: &Buffer, cols: u16, rows: u16) -> RenderedFrame {
    let theme = current_theme();
    // slot 0x00 = canvas bg / edge bg; slot 0x05 = main text / edge fg.
    let neutral_bg = color_to_hex(Some(theme[0x00]), "#000000");
    let neutral_fg = color_to_hex(Some(theme[0x05]), "#ffffff");

    let mut cells = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let cell = buf.cell((col, row)).expect("cell index in bounds");
            let text = cell.symbol().to_string();
            let style = cell.style();
            let fg_str = color_to_hex(style.fg, &neutral_fg);
            let bg_str = color_to_hex(style.bg, &neutral_bg);
            let bold = style.add_modifier.contains(Modifier::BOLD);
            let italic = style.add_modifier.contains(Modifier::ITALIC);
            let underline = style.add_modifier.contains(Modifier::UNDERLINED);

            let is_empty = text == " " || text.is_empty();
            let is_neutral = fg_str == neutral_fg && bg_str == neutral_bg;

            // Skip blank cells that carry no styling information.
            if is_empty && is_neutral {
                continue;
            }

            cells.push(RenderedCell {
                x: col,
                y: row,
                text,
                fg: if fg_str == neutral_fg {
                    None
                } else {
                    Some(fg_str)
                },
                bg: if bg_str == neutral_bg {
                    None
                } else {
                    Some(bg_str)
                },
                bold,
                italic,
                underline,
            });
        }
    }
    RenderedFrame {
        cols,
        rows,
        neutral_fg,
        neutral_bg,
        cells,
    }
}

fn annotation_to_span(annotation: &PyAnnotation) -> AnnotationSpan {
    AnnotationSpan {
        id: annotation.inner.id,
        name: annotation.inner.name.clone(),
        segments: annotation.ann_segments.clone(),
    }
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Report a session failure the way the Python API always has: a bad argument is a
/// `ValueError`, anything else a `RuntimeError`.
fn session_error(error: GraphSessionError) -> PyErr {
    match error {
        GraphSessionError::InvalidArgument(message) => PyValueError::new_err(message),
        GraphSessionError::Operation(message) => PyRuntimeError::new_err(message),
    }
}

/// The page operations that take Python objects. The pages themselves live in the shared
/// graph session, which knows nothing of Python.
pub(crate) trait PyPageExt {
    /// Fails with a `ValueError` naming why a position cannot be shown: the block group lacks it,
    /// or edits cut it off from the start of the pruned view.
    fn go_to_pos(&mut self, pos: &PyPosition, center: bool) -> PyResult<()>;

    /// Highlight each named base and navigate to the leftmost one in the current layout.
    fn show_positions(
        &mut self,
        positions: &[PyPosition],
        color: Option<&str>,
        center: bool,
    ) -> PyResult<()>;

    /// Highlight the path of nodes covered by `locus` in the given colour.
    ///
    /// `color` must be a CSS hex string like `"#ffff00"` or one of the named
    /// ratatui colours (`"yellow"`, `"cyan"`, `"red"`, …).  When omitted the
    /// theme's highlight accent is used.
    fn highlight_match(&mut self, locus: &PyGraphLocus, color: Option<&str>) -> PyResult<()>;

    /// Navigate to an `Annotation` object.
    fn go_to_annotation_obj(&mut self, annotation: &PyAnnotation, center: bool) -> PyResult<()>;

    /// Highlight an `Annotation` on the graph without a label. The highlight gets its own id,
    /// so it neither inherits the track's cached color nor repaints the track when pinned.
    fn highlight_annotation_obj(
        &mut self,
        annotation: &PyAnnotation,
        color: Option<&str>,
    ) -> PyResult<()>;

    /// Navigate to a `GraphLocus`.
    fn go_to_locus(&mut self, locus: &PyGraphLocus, center: bool) -> PyResult<()>;
}

impl PyPageExt for GraphPage {
    fn go_to_pos(&mut self, pos: &PyPosition, center: bool) -> PyResult<()> {
        self.go_to_coordinate(pos.position.node_id, pos.position.coordinate, center)
            .map_err(|error| PyValueError::new_err(format!("cannot go to position: {error}")))
    }

    fn show_positions(
        &mut self,
        positions: &[PyPosition],
        color: Option<&str>,
        center: bool,
    ) -> PyResult<()> {
        let view_state = self.controller.view_state();
        let levels = self.controller.zoom_levels();
        let renderer = &levels[view_state.zoom_index.min(levels.len() - 1)].1;
        let leftmost = positions.iter().min_by_key(|position| {
            let block = position
                .position
                .block(self.controller.engine().graph())
                .unwrap_or(position.block);
            view_state
                .frame
                .rect_of(block)
                .map_or((1, 0, position.position.coordinate), |rect| {
                    (
                        0,
                        rect.min.x
                            + renderer.map_column(
                                &block,
                                position.position.coordinate - block.sequence_start,
                            ),
                        position.position.coordinate,
                    )
                })
        });
        // A hit the pruned view omits has no block to show, but the other hits still highlight.
        if let Some(position) = leftmost {
            let _ = self.go_to_pos(position, center);
        }
        let span = AnnotationSpan {
            id: HashId::random_str(),
            name: String::new(),
            segments: positions
                .iter()
                .map(|position| AnnotationSegment {
                    node_id: position.position.node_id,
                    range: Range {
                        start: position.position.coordinate,
                        end: position.position.coordinate + 1,
                    },
                    strand: position.position.strand,
                })
                .collect(),
        };
        self.push_adhoc_highlight(span, color)
            .map_err(session_error)
    }

    fn highlight_match(&mut self, locus: &PyGraphLocus, color: Option<&str>) -> PyResult<()> {
        self.push_adhoc_highlight(
            annotation_span_from_graph_locus(&locus.graph_locus(), ""),
            color,
        )
        .map_err(session_error)
    }

    fn go_to_annotation_obj(&mut self, annotation: &PyAnnotation, center: bool) -> PyResult<()> {
        // The locus carries absolute node coordinates, so the controller locates the target
        // itself and nodes outside the loaded window are reachable.
        let locus = PyGraphLocus::with_context(annotation.graph_locus(), None);
        self.go_to_locus(&locus, center)
    }

    fn highlight_annotation_obj(
        &mut self,
        annotation: &PyAnnotation,
        color: Option<&str>,
    ) -> PyResult<()> {
        self.push_adhoc_highlight(annotation_to_span(annotation), color)
            .map_err(session_error)
    }

    fn go_to_locus(&mut self, locus: &PyGraphLocus, center: bool) -> PyResult<()> {
        let Some(position) = locus.target_position(center) else {
            return Ok(());
        };
        self.go_to_pos(&position, center)
    }
}

/// The database handle a `PySequenceGraph`'s widget reads through, pinned to the branch it
/// is plotted from.
fn database_for_sequence_graph(sg: &PySequenceGraph) -> PyResult<GraphDatabase> {
    let context = sg.context.clone().ok_or_else(|| {
        PyRuntimeError::new_err(
            "plot() requires a Repository context; obtain SequenceGraphs via Repository by query or id.",
        )
    })?;
    let graph_conn = context.graph().conn();
    let workspace = workspace_for_connection(graph_conn)?;
    GraphDatabase::for_connection(graph_conn, &workspace).map_err(runtime_error)
}

/// Build a lazily-loaded `GraphPage` for a `PySequenceGraph` - see `GraphPage::new`.
fn loaded_page_for_sequence_graph(
    sg: &PySequenceGraph,
    options: PlotOptions,
) -> PyResult<GraphPage> {
    GraphPage::new(
        sg.name.clone(),
        database_for_sequence_graph(sg)?,
        sg.id,
        options,
    )
    .map_err(session_error)
}

/// Capture the information needed to lazily build a page for `sg` later.
fn page_ref_for_sequence_graph(sg: &PySequenceGraph, options: PlotOptions) -> PyResult<PageRef> {
    Ok(PageRef::new(
        sg.name.clone(),
        database_for_sequence_graph(sg)?,
        sg.id,
        options,
    ))
}

/// Controller backing a `GraphWidget`.
///
/// Not intended for direct use from Python — users should call
/// `repo.plot(sg)`, `sg.plot()`, or `sample.plot()`, all of which return a
/// `GraphWidget`.
///
/// Pages through one or more sequence graphs; a plain single-graph widget is
/// just the common case of one page, with the page index hidden from the
/// header and the frontend's pager arrows hidden (see `page_count`). Pages
/// beyond the first are built lazily on first visit, since a `Sample` may
/// hold many sequence graphs that most viewing sessions never page through.
#[pyclass(name = "GraphWidget", from_py_object)]
#[derive(Clone)]
pub struct PyGraphController {
    session: GraphSession,
}

impl PyGraphController {
    /// Wrap a single block group as a one-page controller, its graph loaded lazily -
    /// see `GraphPage::new`. Only used by this module's own tests.
    #[cfg(test)]
    fn new(conn: &GraphConnection, block_group_id: HashId) -> PyResult<Self> {
        let workspace = workspace_for_connection(conn)?;
        let database = GraphDatabase::for_connection(conn, &workspace).map_err(runtime_error)?;
        let page = GraphPage::new(
            String::new(),
            database,
            block_group_id,
            PlotOptions {
                show_history: true,
                ..PlotOptions::default()
            },
        )
        .map_err(session_error)?;
        Ok(Self {
            session: GraphSession::single(page),
        })
    }

    /// Build a single-page controller for `sg`, loading its graph lazily.
    pub(crate) fn for_sequence_graph(sg: &PySequenceGraph, options: PlotOptions) -> PyResult<Self> {
        Ok(Self {
            session: GraphSession::single(loaded_page_for_sequence_graph(sg, options)?),
        })
    }

    /// Build a multi-page controller paging through every sequence graph in
    /// `block_groups`. Each page's graph is loaded lazily on first visit.
    pub(crate) fn for_sample(
        block_groups: &[PySequenceGraph],
        options: PlotOptions,
    ) -> PyResult<Self> {
        let page_refs = block_groups
            .iter()
            .map(|sg| page_ref_for_sequence_graph(sg, options))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            session: GraphSession::paged(page_refs).map_err(session_error)?,
        })
    }

    fn active(&mut self) -> PyResult<&mut GraphPage> {
        self.session.active().map_err(session_error)
    }
}

#[pymethods]
impl PyGraphController {
    /// Deep-clone this controller (all loaded pages' graph state + view state).
    fn clone_controller(&self) -> Self {
        self.clone()
    }

    /// Whether annotation groups have already been loaded into the active page.
    ///
    /// Returns ``True`` after either :meth:`trigger_auto_load` or
    /// :meth:`load_annotation_groups_with_colors` has been called. Cloned
    /// controllers inherit this flag, preventing double-loads on cell re-display.
    #[getter]
    fn annotations_loaded(&mut self) -> PyResult<bool> {
        Ok(self.active()?.annotations_loaded())
    }

    /// Load all annotation groups using the automatic theme-colour palette.
    ///
    /// Called by ``GraphWidget`` when no ``colors`` mapping is provided to
    /// ``plot()``. No-op if annotations have already been loaded.
    fn trigger_auto_load(&mut self) -> PyResult<()> {
        self.active()?.trigger_auto_load().map_err(session_error)
    }

    /// Load annotation groups applying per-annotation colours from a Python-resolved map.
    ///
    /// Parameters
    /// color_map : dict[str, str | None]
    ///     Maps annotation ID hex strings to a CSS hex colour string (e.g.
    ///     ``"#ff4444"``) or ``None`` to hide that annotation entirely.
    ///     Annotations absent from the map fall back to the auto theme palette.
    ///
    /// Called by ``GraphWidget`` after it evaluates the ``colors`` callable/dict/list
    /// provided to ``plot()``. No-op if annotations have already been loaded.
    fn load_annotation_groups_with_colors(
        &mut self,
        color_map: HashMap<String, Option<String>>,
    ) -> PyResult<()> {
        self.active()?
            .load_annotation_groups_with_colors(&color_map)
            .map_err(session_error)
    }

    /// Set the level of node detail.
    ///
    /// Parameters
    /// detail : {"minimal", "normal", "labels", "full"}
    ///     ``"minimal"`` shows the smallest representation; ``"normal"`` shows truncated
    ///     nodes with annotation names hidden; ``"labels"`` shows truncated nodes with
    ///     annotation names; ``"full"`` shows complete sequences.
    pub fn set_detail(&mut self, detail: &str) -> PyResult<()> {
        self.active()?.set_detail(detail).map_err(session_error)
    }

    pub fn truncate_sequences(&mut self) -> PyResult<()> {
        self.active()?.truncate_sequences();
        Ok(())
    }

    pub fn full_sequences(&mut self) -> PyResult<()> {
        self.active()?.full_sequences();
        Ok(())
    }

    pub fn minimize_sequences(&mut self) -> PyResult<()> {
        self.active()?.minimize_sequences();
        Ok(())
    }

    fn render_frame(&mut self, cols: u16, rows: u16) -> PyResult<String> {
        let buf = self
            .session
            .render_frame_buffer(cols, rows)
            .map_err(session_error)?;
        let frame = serialize_buffer(&buf, cols, rows);
        serde_json::to_string(&frame).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Rasterize the current view to PNG bytes, one terminal cell per 11x21 pixels times
    /// ``scale``. The PNG is labeled 96 dpi times the scale (2 is 192 dpi, 3 is 288 dpi,
    /// close to a 300 dpi print).
    ///
    /// Works without any Jupyter dependency, so scripts and agents can save plots.
    #[pyo3(signature = (cols, rows, scale=2))]
    fn render_png<'py>(
        &mut self,
        py: Python<'py>,
        cols: u16,
        rows: u16,
        scale: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.render_png_bytes(py, (cols, rows), false, scale)
    }

    /// Rasterize the full interactive viewport to PNG bytes, preserving the header and cell
    /// positions, with ``scale`` image pixels per logical pixel.
    #[pyo3(signature = (cols, rows, scale=2))]
    fn render_png_viewport<'py>(
        &mut self,
        py: Python<'py>,
        cols: u16,
        rows: u16,
        scale: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.render_png_bytes(py, (cols, rows), true, scale)
    }

    /// Return Rust hit-test results for clickable cells in the last rendered viewport.
    fn render_hit_regions(&mut self, cols: u16, rows: u16) -> PyResult<String> {
        self.session
            .hit_regions_json(cols, rows)
            .map_err(session_error)
    }

    /// Return the logical pixel dimensions of a PNG terminal cell. A PNG rendered at ``scale``
    /// has cells ``scale`` times this size in image pixels.
    fn png_cell_size(&self) -> (usize, usize) {
        GraphSession::png_cell_size()
    }

    /// Return the exact canvas color used by the native PNG renderer.
    fn png_background_color(&self) -> String {
        GraphSession::png_background_color()
    }

    /// Keep the cursor out of every later render (`True`), or let navigation bring it back
    /// (`False`).
    fn set_cursor_hidden(&mut self, hidden: bool) -> PyResult<()> {
        self.session
            .set_cursor_hidden(hidden)
            .map_err(session_error)
    }

    fn zoom_in(&mut self) -> PyResult<()> {
        self.active()?.zoom_in();
        Ok(())
    }

    /// Show or hide annotation names at normal zoom, where long names would stretch the
    /// nodes. Full detail always shows them.
    fn set_labels_shown(&mut self, shown: bool) -> PyResult<()> {
        self.active()?.set_labels_shown(shown);
        Ok(())
    }

    /// Whether annotation names are shown at normal zoom.
    #[getter]
    fn labels_shown(&mut self) -> PyResult<bool> {
        Ok(self.active()?.labels_shown())
    }

    fn zoom_out(&mut self) -> PyResult<()> {
        self.active()?.zoom_out();
        Ok(())
    }

    fn handle_click(&mut self, col: u16, row: u16) -> PyResult<bool> {
        Ok(self.active()?.handle_click(col, row))
    }

    fn move_by(&mut self, dx: i16, dy: i16) -> PyResult<()> {
        self.active()?.move_by(dx, dy);
        Ok(())
    }

    #[pyo3(signature = (pos, center=false))]
    fn go_to_pos(&mut self, pos: &PyPosition, center: bool) -> PyResult<()> {
        self.active()?.go_to_pos(pos, center)
    }

    #[pyo3(signature = (positions, color=None, center=false))]
    fn show_positions(
        &mut self,
        positions: Vec<PyPosition>,
        color: Option<&str>,
        center: bool,
    ) -> PyResult<()> {
        self.active()?.show_positions(&positions, color, center)
    }

    /// Highlight the path of nodes covered by `match_obj` in the given colour.
    ///
    /// `color` must be a CSS hex string like `"#ffff00"` or one of the named
    /// ratatui colours (`"yellow"`, `"cyan"`, `"red"`, …).  When omitted the
    /// theme's highlight accent is used.
    fn highlight_match(&mut self, locus: &PyGraphLocus, color: Option<&str>) -> PyResult<()> {
        self.active()?.highlight_match(locus, color)
    }

    /// Remove ephemeral highlights added via `show()`, leaving persistent
    /// tracks and the path highlight from `show_path` untouched.
    fn clear_highlights(&mut self) -> PyResult<()> {
        self.active()?.clear_highlights();
        Ok(())
    }

    /// Highlight the most recent path associated with this sequence graph.
    ///
    /// Parameters
    /// color : str, optional
    ///     Colour for the highlight.  Accepts named colours
    ///     (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
    ///     (``"#ff4444"``).  When omitted the next unused theme accent
    ///     colour is chosen automatically.
    ///
    /// Raises
    /// RuntimeError
    ///     If no sequence graph is associated with this widget, or if no path
    ///     exists for the sequence graph.
    /// ValueError
    ///     If ``color`` is not a recognised colour name or CSS hex string.
    #[pyo3(signature = (color=None))]
    pub fn show_path(&mut self, color: Option<&str>) -> PyResult<()> {
        self.active()?.show_path(color).map_err(session_error)
    }

    /// Clear path highlighting previously applied by `show_path`.
    pub fn hide_path(&mut self) -> PyResult<()> {
        self.active()?.clear_path();
        Ok(())
    }

    /// Load annotations from the database by group name and add them as a
    /// horizontal track panel below the graph.
    pub fn add_track_group(&mut self, group: &str) -> PyResult<()> {
        self.active()?.add_track_group(group).map_err(session_error)
    }

    /// Navigate to an `Annotation` object.
    #[pyo3(signature = (annotation, center=false))]
    pub fn go_to_annotation_obj(
        &mut self,
        annotation: &PyAnnotation,
        center: bool,
    ) -> PyResult<()> {
        self.active()?.go_to_annotation_obj(annotation, center)
    }

    /// Highlight an `Annotation` on the graph as a nameless inline annotation,
    /// so the locus is coloured without duplicating the track label.
    pub fn highlight_annotation_obj(
        &mut self,
        annotation: &PyAnnotation,
        color: Option<&str>,
    ) -> PyResult<()> {
        self.active()?.highlight_annotation_obj(annotation, color)
    }

    /// Navigate to a `GraphLocus`.
    #[pyo3(signature = (locus, center=false))]
    pub fn go_to_locus(&mut self, locus: &PyGraphLocus, center: bool) -> PyResult<()> {
        self.active()?.go_to_locus(locus, center)
    }

    /// Every annotation-group name visible from this sequence graph — the full menu
    /// `add_track_group` accepts, independent of which tracks are currently displayed.
    #[getter]
    pub fn track_names(&mut self) -> PyResult<Vec<String>> {
        self.session.track_names().map_err(session_error)
    }

    /// Remove a track-panel annotation by name.
    pub fn remove_track(&mut self, name: &str) -> PyResult<()> {
        self.active()?.remove_track(name);
        Ok(())
    }

    pub fn remove_annotation(&mut self, name: &str) -> PyResult<()> {
        self.active()?.remove_annotation(name);
        Ok(())
    }

    /// Clear all track-panel annotations.
    pub fn clear_all_annotations(&mut self) -> PyResult<()> {
        self.active()?.clear_all_annotations();
        Ok(())
    }

    /// Switch to the next sequence graph in the sample, wrapping around.
    /// A no-op when there is only one page.
    fn next_page(&mut self) {
        self.session.next_page();
    }

    /// Switch to the previous sequence graph in the sample, wrapping around.
    /// A no-op when there is only one page.
    fn prev_page(&mut self) {
        self.session.previous_page();
    }

    /// Number of pages available. The frontend only shows pager arrows when
    /// this is greater than 1 (plain single-graph widgets have exactly one page).
    #[getter]
    fn page_count(&self) -> usize {
        self.session.page_count()
    }

    /// Index of the currently active page, for the frontend's `<index/count>`
    /// indicator.
    #[getter]
    fn page_index(&self) -> usize {
        self.session.page_index()
    }
}

impl PyGraphController {
    fn render_png_bytes<'py>(
        &mut self,
        py: Python<'py>,
        size: (u16, u16),
        full_viewport: bool,
        scale: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .session
            .render_png(size, full_viewport, scale)
            .map_err(session_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Static PNGs render on a wider canvas than the interactive viewport because the
/// image is cropped to its content afterwards: the extra cells capture more of the
/// graph without adding blank margins.
const STATIC_COLS: u32 = 400;
const STATIC_ROWS: u32 = 24;

/// Resolve the canvas for a static PNG, keeping any dimension the caller chose.
fn static_viewport(rows: Option<u32>, cols: Option<u32>) -> (u32, u32) {
    (rows.unwrap_or(STATIC_ROWS), cols.unwrap_or(STATIC_COLS))
}

/// Presentation options shared by graph and sample plots.
pub(crate) struct WidgetOptions {
    /// Viewport height in cells; absent values use the presentation's default.
    pub rows: Option<u32>,
    /// Viewport width in cells; absent values use the presentation's default.
    pub cols: Option<u32>,
    /// Optional annotation color mapping or callback.
    pub colors: Option<Py<PyAny>>,
    /// Return a navigable widget instead of a detached PNG.
    pub interactive: bool,
    /// Initial navigation target, dispatched through the widget's shared API.
    pub go_to: Option<Py<PyAny>>,
    /// Image pixels per logical pixel in the widget's PNGs.
    pub scale: usize,
}

/// Instantiate a widget or render a static PNG from a controller and optional viewport.
/// Shared by `PySequenceGraph::plot` and `PySample::plot`.
pub(crate) fn build_widget(
    py: Python<'_>,
    ctrl: Py<PyGraphController>,
    options: WidgetOptions,
    sequence_graphs: &[PySequenceGraph],
) -> PyResult<Py<PyAny>> {
    let WidgetOptions {
        rows,
        cols,
        colors,
        interactive,
        go_to,
        scale,
    } = options;
    let scale = validated_png_scale(scale).map_err(session_error)?;
    let gen_module = py.import("gen")?;
    let widget_class = if interactive {
        gen_module.getattr("GraphWidget")?
    } else {
        gen_module.getattr("TextGraphWidget")?
    };
    // Interactive defaults stay on the Python widget classes, which the terminal
    // text fallback shares. Only the static path overrides them.
    let (rows, cols) = if interactive {
        (rows, cols)
    } else {
        let (rows, cols) = static_viewport(rows, cols);
        (Some(rows), Some(cols))
    };
    let kwargs = PyDict::new(py);
    if let Some(r) = rows {
        kwargs.set_item("rows", r)?;
    }
    if let Some(c) = cols {
        kwargs.set_item("cols", c)?;
    }
    if let Some(c) = colors {
        kwargs.set_item("colors", c)?;
    }
    kwargs.set_item("sequence_graphs", sequence_graphs.to_vec())?;
    kwargs.set_item("scale", scale)?;
    let widget = widget_class.call((ctrl,), Some(&kwargs))?;
    if let Some(target) = go_to {
        widget.call_method1("go_to", (target,))?;
    }
    if interactive {
        Ok(widget.into())
    } else {
        Ok(widget.call_method0("_render_png")?.into())
    }
}

#[cfg(test)]
mod tests {
    use r#gen::{
        test_helpers::{TestContext, setup_block_group, setup_gen_on_disk},
        views::{
            annotation_track::AnnotationSpan,
            graph_overlay::{GraphOverlay, OverlayContent, OverlaySource},
            graph_session::frame_areas,
            png_export::MAX_SCALE,
        },
    };
    use gen_annotations::projection::AnnotationSegment;
    use gen_core::{
        BranchName, HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, is_end_node,
        is_start_node, range::Range,
    };
    use gen_models::{
        ForeignKeyCache,
        annotations::Annotation,
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        history::{
            HistoryStore as _,
            dolt::{DoltHistoryStore, active_branch},
        },
        locus::{GraphLocus, GraphNodeSlice},
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use gen_tui::highlight_style::PathStyle;
    use pyo3::{exceptions::PyValueError, prelude::*};
    use ratatui::{layout::Rect, style::Color};
    use serde_json::Value;

    use super::{PlotOptions, PyGraphController, PyPageExt, current_theme, static_viewport};
    use crate::python_api::{
        annotation::PyAnnotation, block_group::PySequenceGraph, graph_search::PyGraphLocus,
    };

    #[test]
    fn test_widget_loci_and_annotation_entry_points_use_absolute_spans() {
        Python::initialize();
        Python::attach(|python| {
            let context = setup_gen_on_disk();
            let mut controller = make_controller(&context, None).expect("should create the widget");
            // The graph is only seeded until a render drives the crawl.
            controller
                .render_frame(80, 24)
                .expect("should render the first batch");
            let page = controller.active().expect("should load the page");
            let node = page
                .controller
                .engine()
                .graph()
                .nodes()
                .find(|node| node.length() > 4)
                .expect("should have a sequence node");
            let locus = Py::new(
                python,
                PyGraphLocus::from_locus(GraphLocus {
                    slices: vec![GraphNodeSlice {
                        block: node,
                        start: 1,
                        end: 4,
                        strand: Strand::Reverse,
                    }],
                }),
            )
            .expect("should wrap the locus");
            let ephemeral = PyAnnotation::from_locus(locus.borrow(python), "saved");
            // Database annotations take the segments path instead of a cached locus.
            let mut stored = ephemeral.clone();
            stored.locus = None;
            let stored = Py::new(python, stored).expect("should wrap the stored annotation");
            page.highlight_match(&locus.borrow(python), None)
                .expect("should highlight a locus");
            page.go_to_locus(&locus.borrow(python), true)
                .expect("should go to the locus");
            page.push_track(
                "stored",
                vec![super::annotation_to_span(&stored.borrow(python))],
            );
            page.highlight_annotation_obj(&stored.borrow(python), None)
                .expect("should highlight a stored annotation");
            page.go_to_annotation_obj(&stored.borrow(python), true)
                .expect("should go to the stored annotation");
            assert_eq!(page.controller.overlays().len(), 3);
            for overlay in page.controller.overlays() {
                let OverlayContent::Span(span) = &overlay.content else {
                    panic!("should store annotation spans");
                };
                assert_eq!(span.segments.len(), 1);
                let segment = &span.segments[0];
                assert_eq!(segment.node_id, node.node_id);
                assert_eq!(
                    (segment.range.start, segment.range.end),
                    (node.sequence_start + 1, node.sequence_start + 4)
                );
                assert_eq!(segment.strand, Strand::Reverse);
            }
            controller
                .render_frame(80, 24)
                .expect("should render absolute annotation spans");
        });
    }

    #[test]
    fn test_widget_keeps_branch_for_annotations_and_lazy_pages() {
        Python::initialize();
        let context = setup_gen_on_disk();
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let branch = BranchName("design".to_string());
        history_store
            .create_branch(&branch, None)
            .expect("should create design branch");
        history_store
            .checkout_branch(&branch)
            .expect("should checkout design branch");
        let (block_group_id, _) = setup_block_group(context.graph().conn());
        history_store
            .commit_all("design graph")
            .expect("should commit graph on design branch");
        let block_group = BlockGroup::get_by_id(context.graph().conn(), &block_group_id, None)
            .expect("should find design graph");
        let sequence_graph = PySequenceGraph {
            id: block_group_id,
            collection_name: block_group.collection_name,
            sample_name: block_group.sample_name,
            name: block_group.name,
            context: Some(context.db_context().clone()),
        };
        let history_options = PlotOptions {
            show_history: true,
            ..PlotOptions::default()
        };
        let mut graph_controller =
            PyGraphController::for_sequence_graph(&sequence_graph, history_options)
                .expect("should create graph widget on design branch");
        let mut sample_controller =
            PyGraphController::for_sample(&[sequence_graph], history_options)
                .expect("should capture lazy sample page on design branch");
        history_store
            .checkout_branch(&BranchName("main".to_string()))
            .expect("should return to main");

        graph_controller
            .track_names()
            .expect("should find the plotted graph on its original branch");
        sample_controller
            .track_names()
            .expect("should load a pending page from its original branch");
        graph_controller
            .render_frame(100, 30)
            .expect("should render the graph from its original branch");
        let page = graph_controller
            .active()
            .expect("should have an active page");
        assert!(
            page.controller
                .engine()
                .graph()
                .nodes()
                .any(|node| !is_start_node(node.node_id) && !is_end_node(node.node_id)),
            "the crawl should load the plotted graph's nodes from its original branch"
        );
        assert_eq!(
            active_branch(context.graph().conn()).expect("should read repository branch"),
            "main",
            "viewing a design should preserve the repository checkout"
        );
    }

    fn make_controller(ctx: &TestContext, detail: Option<&str>) -> PyResult<PyGraphController> {
        let graph_handle = ctx.graph();
        let (bg_id, _) = setup_block_group(graph_handle.conn());
        let mut ctrl = PyGraphController::new(graph_handle.conn(), bg_id)?;
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        Ok(ctrl)
    }

    #[test]
    fn test_detail_invalid_raises_value_error() {
        Python::initialize();
        let context = setup_gen_on_disk();
        let result = make_controller(&context, Some("bad"));
        Python::attach(|py| match result {
            Ok(_) => panic!("expected a PyValueError for invalid detail value"),
            Err(e) => assert!(e.is_instance_of::<PyValueError>(py)),
        });
    }

    fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
        let read = |range: core::ops::Range<usize>| {
            u32::from_be_bytes(bytes[range].try_into().expect("should read four bytes"))
        };
        (read(16..20), read(20..24))
    }

    #[test]
    fn test_render_png_at_scale_two_doubles_the_pixels_and_rejects_bad_scales() {
        Python::initialize();
        let context = setup_gen_on_disk();
        let mut controller = make_controller(&context, None).expect("should create a controller");
        Python::attach(|python| {
            let single = controller
                .render_png(python, 60, 12, 1)
                .expect("should render at scale 1");
            let (width, height) = png_dimensions(single.as_bytes());
            let doubled = controller
                .render_png(python, 60, 12, 2)
                .expect("should render at scale 2");
            assert_eq!(png_dimensions(doubled.as_bytes()), (width * 2, height * 2));
            let viewport = controller
                .render_png_viewport(python, 60, 12, 2)
                .expect("should render the viewport at scale 2");
            assert_eq!(
                png_dimensions(viewport.as_bytes()),
                (60 * 11 * 2, 12 * 21 * 2)
            );
            for scale in [0, MAX_SCALE + 1] {
                let error = controller
                    .render_png(python, 60, 12, scale)
                    .expect_err("should reject the scale");
                assert!(error.is_instance_of::<PyValueError>(python));
            }
            assert_eq!(controller.png_cell_size(), (11, 21));
        });
    }

    #[test]
    fn test_detail_all_valid_values_accepted() {
        for detail in [
            None,
            Some("minimal"),
            Some("normal"),
            Some("labels"),
            Some("full"),
        ] {
            let context = setup_gen_on_disk();
            let result = make_controller(&context, detail);
            assert!(result.is_ok(), "detail={detail:?} should be accepted");
        }
    }

    #[test]
    fn test_static_viewport_uses_wide_defaults_and_keeps_explicit_sizes() {
        assert_eq!(
            static_viewport(None, None),
            (super::STATIC_ROWS, super::STATIC_COLS)
        );
        assert_eq!(static_viewport(Some(10), None), (10, super::STATIC_COLS));
        assert_eq!(static_viewport(None, Some(80)), (super::STATIC_ROWS, 80));
    }

    #[test]
    fn test_interactive_png_uses_render_frame_header_and_graph_areas() {
        let (header_area, graph_area) = frame_areas(80, 24);
        assert_eq!(header_area, Rect::new(0, 0, 80, 1));
        assert_eq!(graph_area, Rect::new(0, 1, 80, 23));
        let (short_header_area, short_graph_area) = frame_areas(80, 0);
        assert_eq!(short_header_area, Rect::new(0, 0, 80, 0));
        assert_eq!(short_graph_area, Rect::new(0, 1, 80, 0));
    }

    #[test]
    fn test_render_frame_returns_valid_json() {
        Python::initialize();
        Python::attach(|_py| {
            let context = setup_gen_on_disk();
            let mut ctrl = make_controller(&context, None).unwrap();
            let json_str = ctrl
                .render_frame(80, 24)
                .expect("render_frame should succeed");
            let v: Value = serde_json::from_str(&json_str).expect("output must be valid JSON");
            assert_eq!(v["cols"], 80);
            assert_eq!(v["rows"], 24);
            // Neutral colours come from the active theme (slots 0x00 / 0x05).
            let theme = current_theme();
            let expected_fg = super::color_to_hex(Some(theme[0x05]), "#ffffff");
            let expected_bg = super::color_to_hex(Some(theme[0x00]), "#000000");
            assert_eq!(v["neutral_fg"], expected_fg);
            assert_eq!(v["neutral_bg"], expected_bg);
            // Sparse: only non-empty / non-neutral cells are emitted.
            let cells = v["cells"].as_array().expect("cells must be an array");
            assert!(cells.len() < 80 * 24, "sparse frame must omit blank cells");
            assert!(!cells.is_empty(), "a graph must produce at least one cell");
            // Every cell must have x/y coordinates within bounds.
            for cell in cells {
                let x = cell["x"].as_u64().expect("x must be present");
                let y = cell["y"].as_u64().expect("y must be present");
                assert!(x < 80 && y < 24, "cell coordinates out of bounds");
            }
        });
    }

    #[test]
    fn test_annotation_bars_follow_detail_and_removal() {
        let context = setup_gen_on_disk();
        let mut controller =
            make_controller(&context, Some("full")).expect("should create a controller");
        // The graph is only seeded, not fully loaded, until a render drives the crawl - see
        // `GraphPage::new`.
        controller
            .render_frame(100, 30)
            .expect("should render the graph");
        let page = controller.active().expect("should have an active page");
        let (source, target, _) = page
            .controller
            .engine()
            .graph()
            .all_edges()
            .find(|(source, target, _)| {
                !is_start_node(source.node_id) && !is_end_node(target.node_id)
            })
            .expect("should have an edge between sequence nodes");
        page.controller.overlays_mut().push(GraphOverlay {
            content: OverlayContent::Span(AnnotationSpan {
                id: HashId::convert_str("gene"),
                name: "gene".to_string(),
                segments: [source, target]
                    .iter()
                    .map(|node| AnnotationSegment {
                        node_id: node.node_id,
                        range: Range {
                            start: node.sequence_start,
                            end: node.sequence_end,
                        },
                        strand: Strand::Forward,
                    })
                    .collect(),
            }),
            source: OverlaySource::Track("gene".to_string()),
            style: PathStyle::new(Color::Red),
        });
        for detail in ["full", "labels", "normal", "minimal", "full"] {
            controller.set_detail(detail).expect("should change detail");
            let text = rendered_text(&mut controller);
            let annotations_visible = detail != "minimal";
            assert_eq!(
                text.contains('═'),
                annotations_visible,
                "annotation bars at {detail} detail"
            );
            assert_eq!(
                text.contains('>'),
                annotations_visible,
                "direction cap at {detail} detail"
            );
            // Full detail names every piece of the span, one per node it covers; the normal
            // view hides names and the labels view shows them.
            let labels = match detail {
                "full" | "labels" => 2,
                _ => 0,
            };
            assert_eq!(
                text.matches("gene").count(),
                labels,
                "should label the span {labels} times at {detail} detail"
            );
        }

        controller
            .set_detail("normal")
            .expect("should change detail");
        controller
            .set_labels_shown(true)
            .expect("should show the labels");
        assert_eq!(
            rendered_text(&mut controller).matches("gene").count(),
            2,
            "should name the span on each node it covers at normal detail"
        );

        controller
            .remove_annotation("gene")
            .expect("should remove annotation");
        let text = rendered_text(&mut controller);
        assert!(!text.contains('═'), "should remove annotation bars");
        assert!(!text.contains("gene"), "should remove the annotation label");
    }

    /// A `start -> n0 -> ... -> n79 -> end` chain of 5-base nodes, long enough that the first
    /// render crawls only the head of it.
    fn setup_long_chain(conn: &GraphConnection) -> HashId {
        Collection::get_or_create(conn, "test").expect("should create the collection");
        Sample::get_or_create(
            conn,
            NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .expect("should create the sample");
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "chr1",
                ..Default::default()
            },
        )
        .expect("should create the block group");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAA")
            .save(conn)
            .expect("should save the sequence");
        let node_ids: Vec<HashId> = (0..80)
            .map(|index| {
                Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!("n{index}")),
                )
                .expect("should create a chain node")
            })
            .collect();
        let mut links = vec![(PATH_START_NODE_ID, 0, node_ids[0])];
        links.extend(node_ids.windows(2).map(|pair| (pair[0], 5, pair[1])));
        links.push((node_ids[79], 5, PATH_END_NODE_ID));
        let edges: Vec<BlockGroupEdgeData> = links
            .into_iter()
            .map(|(source, source_coordinate, target)| {
                let edge = Edge::create(
                    conn,
                    source,
                    source_coordinate,
                    Strand::Forward,
                    target,
                    0,
                    Strand::Forward,
                )
                .expect("should create a chain edge");
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                }
            })
            .collect();
        BlockGroupEdge::bulk_create(conn, &edges);
        block_group.id
    }

    #[test]
    fn test_go_to_annotation_reaches_a_node_the_crawl_has_not() {
        Python::initialize();
        let context = setup_gen_on_disk();
        let block_group_id = setup_long_chain(context.graph().conn());
        let mut controller = PyGraphController::new(context.graph().conn(), block_group_id)
            .expect("should create a controller");
        controller
            .render_frame(40, 12)
            .expect("should render the first batch");
        let far_node_id = HashId::convert_str("n75");
        let loaded_before = controller
            .active()
            .expect("should have an active page")
            .controller
            .engine()
            .graph()
            .nodes()
            .any(|node| node.node_id == far_node_id);
        assert!(!loaded_before, "the first batch should not reach n75");
        let annotation = PyAnnotation {
            inner: Annotation {
                id: HashId::convert_str("far"),
                name: "far".to_string(),
                group: String::new(),
                accession_id: HashId([0u8; 16]),
                cached_accession: ForeignKeyCache::default(),
                cached_interval_tree: None,
                history_ref: None,
                extra: None,
            },
            context: None,
            ann_segments: vec![AnnotationSegment {
                node_id: far_node_id,
                range: Range { start: 1, end: 4 },
                strand: Strand::Forward,
            }],
            source_block_group_id: None,
            locus: None,
            sequence_graph: None,
        };

        controller
            .go_to_annotation_obj(&annotation, true)
            .expect("should navigate to the annotation");
        controller
            .render_frame(40, 12)
            .expect("should render the target batch");

        let page = controller.active().expect("should have an active page");
        let in_active_batch = page
            .controller
            .engine()
            .active_world()
            .expect("should have an active batch")
            .members()
            .any(|node| node.node_id == far_node_id);
        assert!(in_active_batch, "should open the batch holding n75");
    }

    #[test]
    fn test_go_to_annotation_off_the_block_group_raises_and_keeps_the_view() {
        Python::initialize();
        let context = setup_gen_on_disk();
        let block_group_id = setup_long_chain(context.graph().conn());
        let mut controller = PyGraphController::new(context.graph().conn(), block_group_id)
            .expect("should create a controller");
        let before = rendered_text(&mut controller);
        let annotation = PyAnnotation {
            inner: Annotation {
                id: HashId::convert_str("missing"),
                name: "missing".to_string(),
                group: String::new(),
                accession_id: HashId([0u8; 16]),
                cached_accession: ForeignKeyCache::default(),
                cached_interval_tree: None,
                history_ref: None,
                extra: None,
            },
            context: None,
            ann_segments: vec![AnnotationSegment {
                node_id: HashId::convert_str("not in this block group"),
                range: Range { start: 1, end: 4 },
                strand: Strand::Forward,
            }],
            source_block_group_id: None,
            locus: None,
            sequence_graph: None,
        };

        let error = controller
            .go_to_annotation_obj(&annotation, true)
            .expect_err("should refuse a position that is not in the block group");

        assert!(
            error.to_string().contains("cannot go to position"),
            "{error}"
        );
        assert_eq!(rendered_text(&mut controller), before);
    }

    fn rendered_text(controller: &mut PyGraphController) -> String {
        let frame = controller
            .render_frame(100, 30)
            .expect("should render the graph");
        let value: Value = serde_json::from_str(&frame).expect("should serialize a valid frame");
        value["cells"]
            .as_array()
            .expect("should include rendered cells")
            .iter()
            .map(|cell| cell["text"].as_str().expect("should include cell text"))
            .collect()
    }
}
