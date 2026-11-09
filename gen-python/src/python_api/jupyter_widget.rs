use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use r#gen::views::{
    annotation_files::{AnnotationFileEntry, load_annotation_file_entries},
    annotation_track::{AnnotationSpan, annotation_span_from_graph_locus},
    annotations::{AnnotationFileTrackRequest, load_annotation_file_track},
    gen_graph_controller::{ClickOutcome, GenGraphController},
    graph_database::GraphDatabase,
    graph_overlay::{
        GraphOverlay, OverlayContent, OverlaySource, PathMembership, remove_path_overlay,
        set_path_overlay,
    },
    path_scale::ScalePlacement,
    png_export::{MAX_SCALE, write_cropped_png, write_uncropped_png},
};
use gen_annotations::projection::AnnotationSegment;
use gen_core::{HashId, Workspace, is_terminal, range::Range};
use gen_models::{block_group::BlockGroup, db::GraphConnection};
use gen_tui::{
    LayoutMode, LineStyle, highlight_style::PathStyle, layout::VisualDetail,
    node_renderer::NodeRenderer, theme::current_theme,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Clear, Widget},
};

/// The theme slot (yellow in the default theme) ad hoc `show()` highlights use when no color
/// is given. The accent cursor uses a different slot so it stays visible on top of one.
const HIGHLIGHT_ACCENT_SLOT: usize = 0x0A;
const FRAME_HEADER_HEIGHT: u16 = 1;

fn frame_areas(cols: u16, rows: u16) -> (Rect, Rect) {
    (
        Rect::new(0, 0, cols, FRAME_HEADER_HEIGHT.min(rows)),
        Rect::new(
            0,
            FRAME_HEADER_HEIGHT,
            cols,
            rows.saturating_sub(FRAME_HEADER_HEIGHT),
        ),
    )
}

fn workspace_for_connection(conn: &GraphConnection) -> PyResult<Workspace> {
    let database_path = conn
        .path()
        .map(PathBuf::from)
        .ok_or_else(|| PyRuntimeError::new_err("graph DB has no file path"))?;
    let base_dir = database_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| PyRuntimeError::new_err("graph DB path has no workspace parent"))?;
    Ok(Workspace::new(base_dir))
}
use serde::Serialize;

use crate::python_api::{
    annotation::PyAnnotation, block_group::PySequenceGraph, graph_search::PyGraphLocus,
    position::PyPosition,
};

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

/// Parse a CSS hex colour string like `"#rrggbb"` into a ratatui `Color`.
fn parse_hex_color(hex: &str) -> PyResult<ratatui::style::Color> {
    use ratatui::style::Color;
    if hex.starts_with('#') && hex.len() == 7 {
        let r = u8::from_str_radix(&hex[1..3], 16)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("bad color"))?;
        let g = u8::from_str_radix(&hex[3..5], 16)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("bad color"))?;
        let b = u8::from_str_radix(&hex[5..7], 16)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("bad color"))?;
        Ok(Color::Rgb(r, g, b))
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid colour {hex:?}; expected a CSS hex string like \"#ff4444\""
        )))
    }
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

#[derive(Serialize)]
struct RenderedHitRegion {
    x: u16,
    y: u16,
    kind: &'static str,
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

/// Sort key ordering annotation spans longest-first, so shorter (inner) spans paint on top.
fn sort_key_longest_first(span: &AnnotationSpan) -> i64 {
    -span
        .segments
        .iter()
        .map(|segment| segment.range.end - segment.range.start)
        .sum::<i64>()
}

/// One sequence graph "page" of a `PyGraphController`. A plain `GraphWidget` has exactly one
/// page; a `Sample`-backed widget pages through several.
///
/// The graph, its view, dimming, overlays and annotation groups all live in a
/// [`GenGraphController`], the same one the terminal viewers draw, so a notebook widget loads
/// lazily, steps through doors and reloads annotation groups per batch exactly as they do. The
/// page adds only what is particular to notebooks: the Python-side colors, file tracks and
/// annotation listing.
///
/// # Thread safety
///
/// ipykernel 6+ runs cell code in a thread-pool executor while anywidget comm/observe callbacks
/// fire on the asyncio ioloop, so a page is created on one thread and used from another. The
/// controller owns its database handle rather than borrowing a connection, which keeps it
/// `Send + Sync`.
#[derive(Clone)]
struct GraphPage {
    name: String,
    controller: GenGraphController,
    /// Set once Python has set up the annotation groups (with or without colors). Survives
    /// cloning so that cell-display clones do not set them up again.
    annotation_groups_loaded: bool,
    annotation_files: Vec<FileTrack>,
}

#[derive(Clone)]
struct FileTrack {
    entry: AnnotationFileEntry,
    shown: bool,
    index_available: bool,
    loaded_window: Option<(i64, i64)>,
}

/// How `plot()` loads and lays out a page.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlotOptions {
    /// Keep pruned/retired edit-site edges in the graph, dimmed, instead of removing them.
    pub show_history: bool,
    /// Place the block group's current path on one straight row.
    pub align_path: bool,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            show_history: false,
            align_path: true,
        }
    }
}

/// The information needed to lazily build a `GraphPage` on first visit.
#[derive(Clone)]
struct PageRef {
    name: String,
    database: GraphDatabase,
    block_group_id: HashId,
    /// Mirrors `plot(show_history=..., align_path=...)` - see `GraphPage::new`.
    options: PlotOptions,
}

/// One page of a `PyGraphController`: either already loaded, or pending lazy
/// construction the first time it becomes the active page.
#[derive(Clone)]
enum Page {
    Loaded(Box<GraphPage>),
    Pending(Box<PageRef>),
}

impl Page {
    fn name(&self) -> &str {
        match self {
            Page::Loaded(page) => &page.name,
            Page::Pending(page_ref) => &page_ref.name,
        }
    }
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

impl GraphPage {
    /// Open `block_group_id` in a controller, its graph seeded and grown lazily as renders
    /// crawl it. `show_history=False` leaves pruned/retired edit-site edges out of the crawl
    /// entirely; `show_history=True` loads them and dims them instead. `align_path=True`
    /// aligns the current path to y = 0, with variants placed around that axis.
    fn new(
        name: String,
        database: GraphDatabase,
        block_group_id: HashId,
        options: PlotOptions,
    ) -> PyResult<Self> {
        let mut controller = GenGraphController::new(database, None)
            .with_pruned_history(!options.show_history)
            .with_aligned_current_path(options.align_path);
        controller
            .open_block_group(&block_group_id)
            .map_err(runtime_error)?;
        Ok(Self {
            name,
            controller,
            annotation_groups_loaded: false,
            annotation_files: Vec::new(),
        })
    }

    fn block_group_id(&self) -> Option<HashId> {
        self.controller
            .block_group()
            .map(|block_group| block_group.id)
    }

    /// Draw the graph, its annotation flags into `graph_area`.
    fn render_into(&mut self, buf: &mut Buffer, graph_area: Rect) {
        self.draw_graph(buf, graph_area);
        // Annotation files are read for the nodes already loaded, so the first load has to
        // follow a first draw; a file loaded against an empty graph would find nothing.
        let mut redraw = false;
        if self.annotation_files.is_empty() {
            let mut database = self.controller.database_mut().clone();
            if let Ok(conn) = database.connection() {
                self.show_all_annotation_files(conn);
            }
            redraw = self.annotation_files.iter().any(|file| file.shown);
        }
        redraw |= self.refresh_annotation_files_for_viewport();
        if redraw {
            Clear.render(graph_area, buf);
            self.draw_graph(buf, graph_area);
        }
    }

    fn draw_graph(&mut self, buf: &mut Buffer, graph_area: Rect) {
        self.controller
            .render_settled(buf, graph_area, Style::default());
    }

    fn resolve_color(&mut self, color: Option<&str>) -> PyResult<Color> {
        match color {
            None => Ok(self.controller.view_state_mut().next_accent_color()),
            Some(s) => match s {
                "red" => Ok(Color::Red),
                "green" => Ok(Color::Green),
                "yellow" => Ok(Color::Yellow),
                "blue" => Ok(Color::Blue),
                "magenta" => Ok(Color::Magenta),
                "cyan" => Ok(Color::Cyan),
                "white" => Ok(Color::White),
                hex if hex.starts_with('#') => parse_hex_color(hex),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown color {other:?}"
                ))),
            },
        }
    }

    /// Add `span` as a nameless ad hoc highlight. It gets its own id so an explicit `color`
    /// pins only this highlight. Without one it takes the theme's highlight accent, the same
    /// on every call: ad hoc highlights mark what the reader just asked to see, so they
    /// aren't colored apart from each other or from the annotation tracks.
    fn push_adhoc_highlight(
        &mut self,
        mut span: AnnotationSpan,
        color: Option<&str>,
    ) -> PyResult<()> {
        span.id = HashId::random_str();
        let color = match color {
            Some(color) => self.resolve_color(Some(color))?,
            None => current_theme()[HIGHLIGHT_ACCENT_SLOT],
        };
        self.controller.pin_annotation_color(span.id, color);
        self.controller.overlays_mut().push(GraphOverlay {
            content: OverlayContent::Span(span),
            source: OverlaySource::Adhoc,
            style: annotation_style(color),
        });
        Ok(())
    }

    /// Add `spans` as the file or ad hoc track `name`; `reapply_overlays` colors them.
    fn push_track(&mut self, name: &str, mut spans: Vec<AnnotationSpan>) {
        spans.sort_by_key(sort_key_longest_first);
        let overlays = self.controller.overlays_mut();
        overlays.extend(spans.into_iter().map(|span| GraphOverlay {
            content: OverlayContent::Span(span),
            source: OverlaySource::Track(name.to_string()),
            style: annotation_style(Color::Reset),
        }));
    }

    /// The annotation group named `name`, as the entry id its overlays are keyed by.
    fn annotation_group_id(&self, name: &str) -> Option<String> {
        self.controller
            .annotation_group_entries()
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.id.clone())
    }
}

fn expand_query_window(window: (i64, i64)) -> (i64, i64) {
    let span = (window.1 - window.0).max(1);
    (window.0.saturating_sub(span), window.1.saturating_add(span))
}

fn annotation_style(color: Color) -> PathStyle {
    PathStyle::new(color)
        .with_line_style(LineStyle::Bold)
        .with_merge_glyphs(true)
}

impl GraphPage {
    /// Whether Python has already set up this page's annotation groups.
    ///
    /// Returns ``true`` after either [`Self::trigger_auto_load`] or
    /// [`Self::load_annotation_groups_with_colors`] has been called. Cloned
    /// pages inherit this flag, preventing double set-up on cell re-display.
    fn annotations_loaded(&self) -> bool {
        self.annotation_groups_loaded
    }

    /// Show every annotation group in the automatic theme palette. The groups load for each
    /// batch as renders reach it.
    fn trigger_auto_load(&mut self) -> PyResult<()> {
        self.annotation_groups_loaded = true;
        Ok(())
    }

    /// Show every annotation group, coloring annotations from a Python-resolved map.
    ///
    /// `color_map` maps annotation ID hex strings to a CSS hex colour string
    /// (e.g. ``"#ff4444"``) or ``None`` to hide that annotation entirely.
    /// Annotations absent from the map fall back to the auto theme palette. No-op if
    /// annotations have already been set up.
    fn load_annotation_groups_with_colors(
        &mut self,
        color_map: &HashMap<String, Option<String>>,
    ) -> PyResult<()> {
        if self.annotation_groups_loaded {
            return Ok(());
        }
        let overrides = color_map
            .iter()
            .filter_map(|(id, color)| {
                let id = HashId::try_from(id.as_str()).ok()?;
                match color {
                    None => Some((id, None)),
                    // An unreadable color falls back to the automatic palette.
                    Some(hex) => parse_hex_color(hex).ok().map(|color| (id, Some(color))),
                }
            })
            .collect();
        self.controller.set_annotation_color_overrides(overrides);
        self.annotation_groups_loaded = true;
        Ok(())
    }

    /// Set the level of node detail, matching the four zoom steps of the CLI viewer.
    ///
    /// Parameters
    /// detail : {"minimal", "normal", "labels", "full"}
    ///     ``"minimal"`` shows the smallest representation; ``"normal"`` shows truncated
    ///     nodes with annotation names hidden; ``"labels"`` shows truncated nodes with
    ///     annotation names; ``"full"`` shows complete sequences.
    pub fn set_detail(&mut self, detail: &str) -> PyResult<()> {
        let (level, labels) = match detail {
            "minimal" => (VisualDetail::Minimal, None),
            "normal" => (VisualDetail::Truncated, Some(false)),
            "labels" => (VisualDetail::Truncated, Some(true)),
            "full" => (VisualDetail::Full, None),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "detail must be \"minimal\", \"normal\", \"labels\", or \"full\"; got {other:?}"
                )));
            }
        };
        self.controller.set_detail_level(level);
        if let Some(shown) = labels {
            self.controller.show_annotation_names(shown);
        }
        Ok(())
    }

    pub fn truncate_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Truncated);
    }

    pub fn full_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Full);
    }

    pub fn minimize_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Minimal);
    }

    fn zoom_in(&mut self) {
        self.controller.zoom_in();
    }

    fn set_labels_shown(&mut self, shown: bool) {
        self.controller.show_annotation_names(shown);
    }

    fn labels_shown(&self) -> bool {
        self.controller.annotation_names_shown()
    }

    fn zoom_out(&mut self) {
        self.controller.zoom_out();
    }

    /// Apply a click; a door takes the view into the batch behind it. Returns whether
    /// anything changed.
    fn handle_click(&mut self, col: u16, row: u16) -> bool {
        self.controller.click(col, row) != ClickOutcome::Missed
    }

    fn move_by(&mut self, dx: i16, dy: i16) {
        self.controller.pan(dx, dy);
    }

    /// Fails with a `ValueError` naming why a position cannot be shown: the block group lacks it,
    /// or edits cut it off from the start of the pruned view.
    fn go_to_pos(&mut self, pos: &PyPosition, center: bool) -> PyResult<()> {
        self.controller
            .go_to_coordinate(pos.position.node_id, pos.position.coordinate, center)
            .map_err(|error| PyValueError::new_err(format!("cannot go to position: {error}")))
    }

    /// Highlight each named base and navigate to the leftmost one in the current layout.
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
    }

    /// Highlight the path of nodes covered by `match_obj` in the given colour.
    ///
    /// `color` must be a CSS hex string like `"#ffff00"` or one of the named
    /// ratatui colours (`"yellow"`, `"cyan"`, `"red"`, …).  When omitted the
    /// theme's highlight accent is used.
    fn highlight_match(&mut self, locus: &PyGraphLocus, color: Option<&str>) -> PyResult<()> {
        self.push_adhoc_highlight(
            annotation_span_from_graph_locus(&locus.graph_locus(), ""),
            color,
        )
    }

    /// Remove ephemeral highlights while keeping named tracks and the selected path.
    fn clear_highlights(&mut self) {
        self.controller.overlays_mut().retain(|overlay| {
            matches!(
                overlay.source,
                OverlaySource::Track(_) | OverlaySource::Path
            )
        });
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
    pub fn show_path(&mut self, color: Option<&str>) -> PyResult<()> {
        let block_group_id = self.block_group_id().ok_or_else(|| {
            PyRuntimeError::new_err(
                "show_path() requires a sequence graph; obtain the widget via SequenceGraph.plot()",
            )
        })?;
        let highlight_color = self.resolve_color(color)?;
        let conn = self
            .controller
            .database_mut()
            .connection()
            .map_err(runtime_error)?;
        let path =
            BlockGroup::get_current_path(conn, &block_group_id, None).map_err(runtime_error)?;
        // Only the path's edge ids are fetched; nothing is added to the crawled graph. Each
        // reapply highlights whichever of those edges are loaded, so batches crawled later
        // extend the highlight without touching the database again.
        let membership = PathMembership::load(conn, &path.id, None);
        if membership.is_empty() {
            return Err(PyRuntimeError::new_err("Path has no edges"));
        }
        set_path_overlay(
            self.controller.overlays_mut(),
            annotation_style(highlight_color),
            membership,
        );
        Ok(())
    }

    /// Clear path highlighting previously applied by `show_path`.
    pub fn clear_path(&mut self) {
        remove_path_overlay(self.controller.overlays_mut());
    }

    /// Show the annotation group `group` again after `remove_track` or
    /// `clear_all_annotations` hid it.
    pub fn add_track_group(&mut self, group: &str) -> PyResult<()> {
        if self.annotation_group_id(group).is_none() {
            let mut database = self.controller.database_mut().clone();
            let conn = database.connection().map_err(runtime_error)?;
            if self.show_annotation_file(conn, group) {
                return Ok(());
            }
        }
        let group_id = self.annotation_group_id(group).ok_or_else(|| {
            PyRuntimeError::new_err(format!("no annotation group named {group:?}"))
        })?;
        self.controller
            .set_annotation_group_enabled(&group_id, true);
        Ok(())
    }

    /// Navigate to an `Annotation` object.
    pub fn go_to_annotation_obj(
        &mut self,
        annotation: &PyAnnotation,
        center: bool,
    ) -> PyResult<()> {
        // The locus carries absolute node coordinates, so the controller locates the target
        // itself and nodes outside the loaded window are reachable.
        let locus = PyGraphLocus::with_context(annotation.graph_locus(), None);
        self.go_to_locus(&locus, center)
    }

    /// Highlight an `Annotation` on the graph without a label. The highlight gets its own id,
    /// so it neither inherits the track's cached color nor repaints the track when pinned.
    pub fn highlight_annotation_obj(
        &mut self,
        annotation: &PyAnnotation,
        color: Option<&str>,
    ) -> PyResult<()> {
        self.push_adhoc_highlight(annotation_to_span(annotation), color)
    }

    /// Navigate to a `GraphLocus`.
    pub fn go_to_locus(&mut self, locus: &PyGraphLocus, center: bool) -> PyResult<()> {
        let Some(position) = locus.target_position(center) else {
            return Ok(());
        };
        self.go_to_pos(&position, center)
    }

    /// Remove the track `name`: an annotation group stays hidden for later batches too, until
    /// `add_track_group` shows it again.
    pub fn remove_track(&mut self, name: &str) {
        for file in &mut self.annotation_files {
            if file.entry.display_name == name {
                file.shown = false;
            }
        }
        if let Some(group_id) = self.annotation_group_id(name) {
            self.controller
                .set_annotation_group_enabled(&group_id, false);
        }
        self.controller
            .overlays_mut()
            .retain(|overlay| !matches!(&overlay.source, OverlaySource::Track(key) if key == name));
    }

    /// Clear all annotations from the graph, hiding every annotation group.
    pub fn clear_all_annotations(&mut self) {
        for file in &mut self.annotation_files {
            file.shown = false;
        }
        let group_ids: Vec<String> = self
            .controller
            .annotation_group_entries()
            .iter()
            .map(|entry| entry.id.clone())
            .collect();
        for group_id in &group_ids {
            self.controller
                .set_annotation_group_enabled(group_id, false);
        }
        // Keep ad hoc highlights (e.g. search matches) and the path.
        self.controller
            .overlays_mut()
            .retain(|overlay| matches!(overlay.source, OverlaySource::Adhoc | OverlaySource::Path));
    }

    /// Remove all overlays whose annotation name matches `name`, regardless of
    /// which track (if any) they belong to. If the same name was added more
    /// than once, every copy is removed.
    pub fn remove_annotation(&mut self, name: &str) {
        self.controller
            .overlays_mut()
            .retain(|overlay| overlay.span().is_none_or(|span| span.name != name));
    }
}

impl GraphPage {
    fn visible_coordinate_window(&self) -> Option<(i64, i64)> {
        let world = self.controller.engine().active_world()?;
        let mut start = i64::MAX;
        let mut end = i64::MIN;
        for node in world.members() {
            if !is_terminal(node.node_id) {
                start = start.min(node.sequence_start);
                end = end.max(node.sequence_end);
            }
        }
        (start <= end).then_some((start, end))
    }

    /// Read the repository's annotation files, keeping the display state of ones already known.
    fn sync_annotation_files(&mut self, conn: &GraphConnection) {
        let entries = load_annotation_file_entries(conn, None);
        let mut previous = std::mem::take(&mut self.annotation_files);
        self.annotation_files = entries
            .into_iter()
            .map(|entry| {
                let known = previous
                    .iter()
                    .position(|file| file.entry.file_addition.id == entry.file_addition.id);
                match known {
                    Some(index) => FileTrack {
                        entry,
                        ..previous.swap_remove(index)
                    },
                    None => FileTrack {
                        entry,
                        shown: false,
                        index_available: false,
                        loaded_window: None,
                    },
                }
            })
            .collect();
    }

    /// Load one file for `window`, replacing whatever it showed before.
    fn load_annotation_file(
        &mut self,
        conn: &GraphConnection,
        index: usize,
        window: Option<(i64, i64)>,
    ) -> bool {
        let Some(block_group_id) = self.block_group_id() else {
            return false;
        };
        let Ok(block_group) = BlockGroup::get_by_id(conn, &block_group_id, None) else {
            return false;
        };
        let node_filter = self.controller.loaded_node_ids();
        let entry = self.annotation_files[index].entry.clone();
        let Ok(workspace) = workspace_for_connection(conn) else {
            return false;
        };
        let loaded = load_annotation_file_track(&AnnotationFileTrackRequest {
            conn,
            history_ref: None,
            workspace: &workspace,
            collection_name: &block_group.collection_name,
            sample_name: &block_group.sample_name,
            block_group_name: Some(&block_group.name),
            query_window: window,
            node_filter: &node_filter,
            entry: &entry,
        });
        let Ok(loaded) = loaded else {
            return false;
        };
        let name = entry.display_name.clone();
        self.controller.overlays_mut().retain(
            |overlay| !matches!(&overlay.source, OverlaySource::Track(track) if *track == name),
        );
        self.push_track(&loaded.track.name, loaded.track.annotations);
        let file = &mut self.annotation_files[index];
        file.shown = true;
        file.index_available = loaded.index_available;
        file.loaded_window = loaded.loaded_window;
        true
    }

    /// The window of sequence coordinates to read an indexed file for: the viewport and as much
    /// again on each side, so small camera moves don't reload.
    fn annotation_query_window(&self) -> Option<(i64, i64)> {
        self.visible_coordinate_window().map(expand_query_window)
    }

    /// Show every annotation file, as database groups are all shown when a graph is plotted.
    /// A file that can't be read is skipped so it doesn't hide the others.
    fn show_all_annotation_files(&mut self, conn: &GraphConnection) {
        self.sync_annotation_files(conn);
        let window = self.annotation_query_window();
        for index in 0..self.annotation_files.len() {
            self.load_annotation_file(conn, index, window);
        }
    }

    /// Show the file recorded under `name`, returning whether there is one.
    fn show_annotation_file(&mut self, conn: &GraphConnection, name: &str) -> bool {
        self.sync_annotation_files(conn);
        let Some(index) = self
            .annotation_files
            .iter()
            .position(|file| file.entry.display_name == name)
        else {
            return false;
        };
        let window = self.annotation_query_window();
        self.load_annotation_file(conn, index, window)
    }

    /// Reload shown, indexed files whose loaded window no longer covers the viewport.
    fn refresh_annotation_files_for_viewport(&mut self) -> bool {
        let Some(visible) = self.visible_coordinate_window() else {
            return false;
        };
        let window = expand_query_window(visible);
        let stale: Vec<usize> = self
            .annotation_files
            .iter()
            .enumerate()
            .filter(|(_, file)| file.shown && file.index_available)
            .filter(|(_, file)| match file.loaded_window {
                Some((start, end)) => visible.0 < start || visible.1 > end,
                None => true,
            })
            .map(|(index, _)| index)
            .collect();
        if stale.is_empty() {
            return false;
        }
        let mut database = self.controller.database_mut().clone();
        let Ok(conn) = database.connection() else {
            return false;
        };
        let mut reloaded = false;
        for index in stale {
            reloaded |= self.load_annotation_file(conn, index, Some(window));
        }
        reloaded
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
}

/// Capture the information needed to lazily build a page for `sg` later.
fn page_ref_for_sequence_graph(sg: &PySequenceGraph, options: PlotOptions) -> PyResult<PageRef> {
    Ok(PageRef {
        name: sg.name.clone(),
        database: database_for_sequence_graph(sg)?,
        block_group_id: sg.id,
        options,
    })
}

/// Draw a one-line header into `area`: the sequence graph name, centred. The
/// page index/count are not drawn into the grid; they are exposed as widget
/// metadata instead, so the frontend can render its own `<index/count>`
/// pager indicator outside the canvas.
fn draw_header(buf: &mut Buffer, area: Rect, name: &str) {
    let theme = current_theme();
    let style = Style::default().fg(theme[0x07]);
    let name_x = area.x + area.width.saturating_sub(name.len() as u16) / 2;
    buf.set_string(name_x, area.y, name, style);
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
    pages: Vec<Page>,
    current_index: usize,
    /// Set by `hide_cursor()`: keeps the cursor out of every render, including after a
    /// later `go_to`/`show`/click would otherwise bring it back.
    cursor_hidden: bool,
}

impl PyGraphController {
    /// Wrap a single block group as a one-page controller, its graph loaded lazily -
    /// see `GraphPage::new`. Only used by this module's own tests.
    #[cfg(test)]
    fn new(conn: &GraphConnection, block_group_id: HashId) -> PyResult<Self> {
        let workspace = workspace_for_connection(conn)?;
        let database = GraphDatabase::for_connection(conn, &workspace).map_err(runtime_error)?;
        Ok(Self {
            pages: vec![Page::Loaded(Box::new(GraphPage::new(
                String::new(),
                database,
                block_group_id,
                PlotOptions {
                    show_history: true,
                    ..PlotOptions::default()
                },
            )?))],
            current_index: 0,
            cursor_hidden: false,
        })
    }

    /// Build a single-page controller for `sg`, loading its graph lazily.
    pub(crate) fn for_sequence_graph(sg: &PySequenceGraph, options: PlotOptions) -> PyResult<Self> {
        Ok(Self {
            pages: vec![Page::Loaded(Box::new(loaded_page_for_sequence_graph(
                sg, options,
            )?))],
            current_index: 0,
            cursor_hidden: false,
        })
    }

    /// Build a multi-page controller paging through every sequence graph in
    /// `block_groups`. Each page's graph is loaded lazily on first visit.
    pub(crate) fn for_sample(
        block_groups: &[PySequenceGraph],
        options: PlotOptions,
    ) -> PyResult<Self> {
        if block_groups.is_empty() {
            return Err(PyRuntimeError::new_err(
                "Sample has no sequence graphs to plot",
            ));
        }
        let pages = block_groups
            .iter()
            .map(|sg| {
                page_ref_for_sequence_graph(sg, options)
                    .map(|page_ref| Page::Pending(Box::new(page_ref)))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            pages,
            current_index: 0,
            cursor_hidden: false,
        })
    }

    /// Draw just the active page's graph into a fresh `cols` x `rows` buffer, without the header.
    fn render_graph_buffer(&mut self, cols: u16, rows: u16) -> PyResult<Buffer> {
        let mut buf = Buffer::empty(Rect::new(0, 0, cols, rows));
        self.render_active(&mut buf, Rect::new(0, 0, cols, rows))?;
        Ok(buf)
    }

    /// Draw the active page into `graph_area`, honouring `hide_cursor()`.
    fn render_active(&mut self, buf: &mut Buffer, graph_area: Rect) -> PyResult<()> {
        let cursor_hidden = self.cursor_hidden;
        let page = self.active()?;
        if cursor_hidden {
            page.controller.view_state_mut().hide_cursor();
        }
        page.render_into(buf, graph_area);
        Ok(())
    }

    fn active(&mut self) -> PyResult<&mut GraphPage> {
        let page = &mut self.pages[self.current_index];
        if let Page::Pending(page_ref) = page {
            let loaded = GraphPage::new(
                page_ref.name.clone(),
                page_ref.database.clone(),
                page_ref.block_group_id,
                page_ref.options,
            )?;
            *page = Page::Loaded(Box::new(loaded));
        }
        match page {
            Page::Loaded(page) => Ok(page.as_mut()),
            Page::Pending(_) => unreachable!("just loaded above"),
        }
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
        self.active()?.trigger_auto_load()
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
    }

    /// Set the level of node detail.
    ///
    /// Parameters
    /// detail : {"minimal", "normal", "labels", "full"}
    ///     ``"minimal"`` shows the smallest representation; ``"normal"`` shows truncated
    ///     nodes with annotation names hidden; ``"labels"`` shows truncated nodes with
    ///     annotation names; ``"full"`` shows complete sequences.
    pub fn set_detail(&mut self, detail: &str) -> PyResult<()> {
        self.active()?.set_detail(detail)
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
        let total_area = Rect::new(0, 0, cols, rows);
        let mut buf = Buffer::empty(total_area);
        let (header_area, graph_area) = frame_areas(cols, rows);
        draw_header(&mut buf, header_area, self.pages[self.current_index].name());
        self.render_active(&mut buf, graph_area)?;
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
        render_png_bytes(self, py, (cols, rows), false, scale)
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
        render_png_bytes(self, py, (cols, rows), true, scale)
    }

    /// Return Rust hit-test results for clickable cells in the last rendered viewport.
    fn render_hit_regions(&mut self, cols: u16, rows: u16) -> PyResult<String> {
        let view_state = self.active()?.controller.view_state();
        let mut hit_regions = Vec::new();
        for y in 0..rows {
            for x in 0..cols {
                let kind = if view_state.wormhole_hit(x, y).is_some() {
                    Some("wormhole")
                } else if view_state.node_hit(x, y).is_some() {
                    Some("node")
                } else {
                    None
                };
                if let Some(kind) = kind {
                    hit_regions.push(RenderedHitRegion { x, y, kind });
                }
            }
        }
        serde_json::to_string(&hit_regions)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Return the logical pixel dimensions of a PNG terminal cell. A PNG rendered at ``scale``
    /// has cells ``scale`` times this size in image pixels.
    fn png_cell_size(&self) -> (usize, usize) {
        gen_tui::RasterConfig::default().logical_cell_size()
    }

    /// Return the exact canvas color used by the native PNG renderer.
    fn png_background_color(&self) -> String {
        let [red, green, blue] = gen_tui::RasterConfig::default().background;
        format!("#{red:02x}{green:02x}{blue:02x}")
    }

    /// Keep the cursor out of every later render (`True`), or let navigation bring it back
    /// (`False`).
    fn set_cursor_hidden(&mut self, hidden: bool) -> PyResult<()> {
        self.cursor_hidden = hidden;
        if !hidden {
            self.active()?.controller.view_state_mut().show_cursor();
        }
        Ok(())
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
        self.active()?.show_path(color)
    }

    /// Clear path highlighting previously applied by `show_path`.
    pub fn hide_path(&mut self) -> PyResult<()> {
        self.active()?.clear_path();
        Ok(())
    }

    /// Load annotations from the database by group name and add them as a
    /// horizontal track panel below the graph.
    pub fn add_track_group(&mut self, group: &str) -> PyResult<()> {
        self.active()?.add_track_group(group)
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
        let page = self.active()?;
        let mut database = page.controller.database_mut().clone();
        let conn = database.connection().map_err(runtime_error)?;
        page.sync_annotation_files(conn);
        Ok(page
            .controller
            .annotation_group_entries()
            .iter()
            .map(|entry| entry.name.clone())
            .chain(
                page.annotation_files
                    .iter()
                    .map(|file| file.entry.display_name.clone()),
            )
            .collect())
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
        self.current_index = (self.current_index + 1) % self.pages.len();
    }

    /// Switch to the previous sequence graph in the sample, wrapping around.
    /// A no-op when there is only one page.
    fn prev_page(&mut self) {
        self.current_index = (self.current_index + self.pages.len() - 1) % self.pages.len();
    }

    /// Number of pages available. The frontend only shows pager arrows when
    /// this is greater than 1 (plain single-graph widgets have exactly one page).
    #[getter]
    fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Index of the currently active page, for the frontend's `<index/count>`
    /// indicator.
    #[getter]
    fn page_index(&self) -> usize {
        self.current_index
    }
}

/// Reject a pixel scale the exporter should not be asked for.
fn validated_png_scale(scale: usize) -> PyResult<usize> {
    if (1..=MAX_SCALE).contains(&scale) {
        Ok(scale)
    } else {
        Err(PyValueError::new_err(format!(
            "scale must be between 1 and {}, got {scale}",
            MAX_SCALE
        )))
    }
}

fn render_png_bytes<'py>(
    widget: &mut PyGraphController,
    py: Python<'py>,
    (cols, rows): (u16, u16),
    full_viewport: bool,
    scale: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let scale = validated_png_scale(scale)?;
    let controller = &mut widget.active()?.controller;
    let previous_layout_mode = controller.view_state().layout_mode;
    let previous_placement = controller.scale_placement();
    controller.view_state_mut().layout_mode = LayoutMode::Curved;
    controller.set_scale_placement(if full_viewport {
        ScalePlacement::Bottom
    } else {
        ScalePlacement::BelowContent
    });
    let buffer_result = if full_viewport {
        let mut buffer = Buffer::empty(Rect::new(0, 0, cols, rows));
        let (header_area, graph_area) = frame_areas(cols, rows);
        draw_header(
            &mut buffer,
            header_area,
            widget.pages[widget.current_index].name(),
        );
        widget
            .render_active(&mut buffer, graph_area)
            .map(|()| buffer)
    } else {
        widget.render_graph_buffer(cols, rows)
    };
    let controller = &mut widget.active()?.controller;
    controller.view_state_mut().layout_mode = previous_layout_mode;
    controller.set_scale_placement(previous_placement);
    let buffer = buffer_result?;
    let mut bytes = Vec::new();
    let write_result = if full_viewport {
        write_uncropped_png(&buffer, controller.overlay_primitives(), scale, &mut bytes)
    } else {
        write_cropped_png(&buffer, controller.overlay_primitives(), scale, &mut bytes)
    };
    write_result.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &bytes))
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
    let scale = validated_png_scale(scale)?;
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

    use super::{
        MAX_SCALE, PlotOptions, PyGraphController, current_theme, frame_areas, static_viewport,
    };
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
