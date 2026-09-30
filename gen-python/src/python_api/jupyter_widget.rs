use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, Cursor},
    path::{Path, PathBuf},
};

use r#gen::views::{
    annotation_track::{
        AnnotationSpan, AnnotationTrack, LoadedNodeSlices, annotation_span_from_graph_locus,
        graph_locus_from_annotation_span,
    },
    annotations::{parse_translated_bed, parse_translated_gff},
    gen_graph_controller::{AnnotationDisplay, ClickOutcome, GenGraphController},
    graph_database::GraphDatabase,
    graph_overlay::{
        GraphOverlay, OverlayContent, OverlaySource, PathMembership, remove_path_overlay,
        set_path_overlay,
    },
};
use gen_annotations::{
    projection::annotation_segments,
    translate::{bed::translate_bed, gff::translate_gff},
};
use gen_core::{HashId, Workspace};
use gen_models::{
    annotations::Annotation, block_group::BlockGroup, db::GraphConnection, sample::Sample,
};
use gen_tui::{LineStyle, layout::VisualDetail, plotter::PathStyle, theme::current_theme};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
};

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
    use r#gen::views::annotation_track::AnnotationSegment as ViewSegment;
    AnnotationSpan {
        id: annotation.inner.id,
        name: annotation.inner.name.clone(),
        segments: annotation
            .ann_segments
            .iter()
            .map(|s| ViewSegment {
                node_id: s.node_id,
                start: s.range.start,
                end: s.range.end,
                strand: s.strand,
            })
            .collect(),
    }
}

/// Sort key ordering annotation spans longest-first, so shorter (inner) spans paint on top.
fn sort_key_longest_first(span: &AnnotationSpan) -> i64 {
    -span
        .segments
        .iter()
        .map(|segment| segment.end - segment.start)
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
}

/// How `plot()` loads and lays out a page.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PlotOptions {
    /// Keep pruned/retired edit-site edges in the graph, dimmed, instead of removing them.
    pub show_history: bool,
    /// Place the block group's current path on one straight row.
    pub center_reference: bool,
}

/// The information needed to lazily build a `GraphPage` on first visit.
#[derive(Clone)]
struct PageRef {
    name: String,
    database: GraphDatabase,
    block_group_id: HashId,
    /// Mirrors `plot(show_history=..., center_reference=...)` - see `GraphPage::new`.
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
    /// entirely; `show_history=True` loads them and dims them instead. `center_reference=True`
    /// lays the block group's current path out as one straight row.
    fn new(
        name: String,
        database: GraphDatabase,
        block_group_id: HashId,
        options: PlotOptions,
    ) -> PyResult<Self> {
        let mut controller = GenGraphController::new(database, None)
            .with_pruned_history(!options.show_history)
            .with_centered_current_path(options.center_reference);
        controller
            .open_block_group(&block_group_id)
            .map_err(runtime_error)?;
        Ok(Self {
            name,
            controller,
            annotation_groups_loaded: false,
        })
    }

    fn block_group_id(&self) -> Option<HashId> {
        self.controller
            .block_group()
            .map(|block_group| block_group.id)
    }

    fn navigate_to_span(&mut self, span: &AnnotationSpan, center: bool) {
        let loaded = LoadedNodeSlices::new(self.controller.engine().graph());
        let Some(locus) = graph_locus_from_annotation_span(span, &loaded) else {
            return;
        };
        let Some(position) = PyGraphLocus::from_locus(locus).target_position(center) else {
            return;
        };
        self.go_to_pos(&position, center);
    }

    /// Draw the graph, its annotation flags and floating labels into `graph_area`.
    fn render_into(&mut self, buf: &mut Buffer, graph_area: Rect) {
        self.controller.render_settled(
            buf,
            graph_area,
            AnnotationDisplay::FlagsUnderNodes,
            Style::default(),
        );
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

    /// Add `span` as a nameless ad hoc highlight. It gets its own id so automatic colors
    /// rotate across calls instead of collapsing onto one cache entry, and so an explicit
    /// `color` pins only this highlight.
    fn push_adhoc_highlight(
        &mut self,
        mut span: AnnotationSpan,
        color: Option<&str>,
    ) -> PyResult<()> {
        span.id = HashId::random_str();
        let color = match color {
            Some(color) => {
                let color = self.resolve_color(Some(color))?;
                self.controller.pin_annotation_color(span.id, color);
                color
            }
            None => Color::Reset,
        };
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

    /// Set the level of node detail.
    ///
    /// Parameters
    /// detail : {"normal", "full", "minimal"}
    ///     ``"normal"`` shows truncated labels; ``"full"`` shows
    ///     complete labels; ``"minimal"`` shows the smallest representation.
    pub fn set_detail(&mut self, detail: &str) -> PyResult<()> {
        let level = match detail {
            "normal" => VisualDetail::Truncated,
            "full" => VisualDetail::Full,
            "minimal" => VisualDetail::Minimal,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "detail must be \"normal\", \"full\", or \"minimal\"; got {other:?}"
                )));
            }
        };
        self.controller.set_detail_level(level);
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

    fn go_to_pos(&mut self, pos: &PyPosition, center: bool) {
        self.controller
            .go_to_coordinate(pos.position.node_id, pos.position.coordinate, center);
    }

    /// Highlight the path of nodes covered by `match_obj` in the given colour.
    ///
    /// `color` must be a CSS hex string like `"#ffff00"` or one of the named
    /// ratatui colours (`"yellow"`, `"cyan"`, `"red"`, …).  When omitted the
    /// next unused theme accent colour (slots 0x08–0x0F) is chosen automatically.
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
        self.controller.set_focused_annotation(None);
    }

    /// Make `annotation` the one whose pieces are joined by connectors, or clear the focus.
    fn focus_annotation(&mut self, annotation: Option<&PyAnnotation>) {
        self.controller
            .set_focused_annotation(annotation.map(|annotation| annotation.inner.id));
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
        let group_id = self.annotation_group_id(group).ok_or_else(|| {
            PyRuntimeError::new_err(format!("no annotation group named {group:?}"))
        })?;
        self.controller
            .set_annotation_group_enabled(&group_id, true);
        Ok(())
    }

    /// Add a list of `Annotation` objects as inline graph overlays grouped under `name`.
    pub fn add_track_annotations(&mut self, annotations: Vec<PyRef<PyAnnotation>>, name: &str) {
        let spans = annotations
            .iter()
            .map(|annotation| annotation_to_span(annotation))
            .collect();
        self.push_track(name, spans);
    }

    /// Load annotations from a GFF3 or BED file and render them as
    /// inline graph highlights with floating labels.
    ///
    /// Accepts both standard files (chromosome/contig names as reference) and
    /// pre-translated files (node hash-IDs as reference).  Standard files are
    /// translated in-memory against `from_sample` before parsing.  If
    /// translation produces no output the file is parsed as-is, so
    /// pre-translated files work without specifying `from_sample`.
    pub fn add_track_file(
        &mut self,
        file_path: &str,
        display_name: Option<&str>,
        from_sample: Option<&str>,
    ) -> PyResult<()> {
        let name = display_name.unwrap_or(file_path);
        let node_ids = self.controller.loaded_node_ids();
        let sample = from_sample.unwrap_or(Sample::DEFAULT_NAME);

        let spans = match self.controller.block_group().cloned() {
            Some(block_group) => {
                let workspace = self.controller.database_mut().workspace().clone();
                let conn = self
                    .controller
                    .database_mut()
                    .connection()
                    .map_err(runtime_error)?;
                let extension = Path::new(file_path)
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                let mut buffer: Vec<u8> = Vec::new();
                let translate_result: Result<(), String> = match extension.as_str() {
                    "gff" | "gff3" => {
                        let reader = BufReader::new(File::open(file_path).map_err(runtime_error)?);
                        translate_gff(
                            conn,
                            &workspace,
                            &block_group.collection_name,
                            sample,
                            None,
                            reader,
                            &mut buffer,
                        )
                    }
                    .map_err(|error| error.to_string()),
                    "bed" => {
                        let reader = File::open(file_path).map_err(runtime_error)?;
                        translate_bed(
                            conn,
                            &workspace,
                            &block_group.collection_name,
                            sample,
                            None,
                            reader,
                            &mut buffer,
                        )
                    }
                    .map_err(|error| error.to_string()),
                    other => {
                        return Err(PyRuntimeError::new_err(format!(
                            "unsupported annotation file type: {other:?}; expected .gff, .gff3, or .bed"
                        )));
                    }
                };
                translate_result.map_err(PyRuntimeError::new_err)?;

                if buffer.is_empty() {
                    // Translation found no matching sequences, so the file may already be in
                    // translated (hash-ID) format.
                    load_track_from_file(file_path, name, &node_ids)
                        .map_err(runtime_error)?
                        .annotations
                } else if matches!(extension.as_str(), "gff" | "gff3") {
                    parse_translated_gff(Cursor::new(buffer), &node_ids, name, HashMap::new())
                } else {
                    parse_translated_bed(Cursor::new(buffer), &node_ids, name, HashMap::new())
                }
            }
            None => {
                load_track_from_file(file_path, name, &node_ids)
                    .map_err(runtime_error)?
                    .annotations
            }
        };
        self.push_track(name, spans);
        Ok(())
    }

    /// Navigate to an `Annotation` object.
    pub fn go_to_annotation_obj(&mut self, annotation: &PyAnnotation, center: bool) {
        let span = annotation_to_span(annotation);
        self.navigate_to_span(&span, center);
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
    pub fn go_to_locus(&mut self, locus: &PyGraphLocus, center: bool) {
        let Some(position) = locus.target_position(center) else {
            return;
        };
        self.go_to_pos(&position, center);
    }

    /// Return all gene annotations for this sequence graph.
    ///
    /// Delegates to the database, returning every annotation stored for this
    /// sequence graph — independent of which tracks are currently loaded in the
    /// widget.
    ///
    /// The returned ``Annotation`` objects carry no repository context. To
    /// translate one, pass it to
    /// ``SequenceGraph.translate_annotation(region=ann)``, which resolves the
    /// annotation through its own context.
    pub fn list_annotations(&mut self) -> PyResult<Vec<PyAnnotation>> {
        let block_group = self.controller.block_group().cloned().ok_or_else(|| {
            PyRuntimeError::new_err(
                "annotations requires a sequence graph; \
                 create the widget via SequenceGraph.plot()",
            )
        })?;
        let conn = self
            .controller
            .database_mut()
            .connection()
            .map_err(runtime_error)?;
        let annotations = Annotation::query_with_lineage(
            conn,
            &block_group.collection_name,
            &block_group.sample_name,
            &block_group.name,
        )
        .map_err(runtime_error)?;
        Ok(annotations
            .into_iter()
            .map(|annotation| PyAnnotation {
                ann_segments: annotation_segments(conn, &annotation, None),
                inner: annotation,
                context: None,
                source_block_group_id: Some(block_group.id),
                locus: None,
                sequence_graph: None,
            })
            .collect())
    }

    /// Return a JSON list of the tracks currently drawn: annotation groups by name, then
    /// tracks from `add_track_annotations` and `add_track_file`.
    pub fn get_track_names(&self) -> PyResult<String> {
        let entries = self.controller.annotation_group_entries();
        let mut seen = HashSet::new();
        let names: Vec<&str> = self
            .controller
            .overlays()
            .iter()
            .filter_map(|overlay| match &overlay.source {
                OverlaySource::Track(key) => Some(
                    key.strip_prefix("group:")
                        .and_then(|group_id| entries.iter().find(|entry| entry.id == group_id))
                        .map_or(key.as_str(), |entry| entry.name.as_str()),
                ),
                OverlaySource::Adhoc | OverlaySource::Path | OverlaySource::Search => None,
            })
            .filter(|name| seen.insert(*name))
            .collect();
        serde_json::to_string(&names).map_err(runtime_error)
    }

    /// Remove the track `name`: an annotation group stays hidden for later batches too, until
    /// `add_track_group` shows it again.
    pub fn remove_track(&mut self, name: &str) {
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

    /// Add annotations rendered directly on the graph canvas.
    /// Annotations are tinted with an accent colour and labelled below their span.
    pub fn add_annotation(
        &mut self,
        annotations: Vec<PyRef<PyAnnotation>>,
        track_name: Option<String>,
    ) {
        let existing_color = track_name.as_deref().and_then(|name| {
            self.controller
                .overlays()
                .iter()
                .find_map(|overlay| match &overlay.source {
                    OverlaySource::Track(existing) if existing == name => Some(overlay.style.color),
                    _ => None,
                })
        });
        let color =
            existing_color.unwrap_or_else(|| self.controller.view_state_mut().next_accent_color());
        let source = match &track_name {
            Some(name) => OverlaySource::Track(name.clone()),
            None => OverlaySource::Adhoc,
        };
        let overlays = self.controller.overlays_mut();
        overlays.extend(annotations.iter().map(|annotation| GraphOverlay {
            content: OverlayContent::Span(annotation_to_span(annotation)),
            source: source.clone(),
            style: annotation_style(color),
        }));
    }

    /// Return a JSON list of annotation names currently loaded (from
    /// `add_annotation`; annotations loaded as part of a track keep their own
    /// name here too, separately from the track's name).
    pub fn get_annotation_names(&self) -> PyResult<String> {
        let mut seen = HashSet::new();
        let names: Vec<&str> = self
            .controller
            .overlays()
            .iter()
            .filter_map(|overlay| overlay.span().map(|span| span.name.as_str()))
            .filter(|name| !name.is_empty() && seen.insert(*name))
            .collect();
        serde_json::to_string(&names).map_err(runtime_error)
    }

    /// Remove all overlays whose annotation name matches `name`, regardless of
    /// which track (if any) they belong to. If the same name was added more
    /// than once, every copy is removed.
    pub fn remove_annotation(&mut self, name: &str) {
        self.controller
            .overlays_mut()
            .retain(|overlay| overlay.span().is_none_or(|span| span.name != name));
        self.controller.set_focused_annotation(None);
    }
}

impl GraphPage {
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
        let Some(block_group_id) = self.block_group_id else {
            return false;
        };
        let Ok(block_group) = BlockGroup::get_by_id(conn, &block_group_id, None) else {
            return false;
        };
        let node_filter = self.all_node_ids();
        let entry = self.annotation_files[index].entry.clone();
        let loaded = load_annotation_file_track(&AnnotationFileTrackRequest {
            conn,
            history_ref: None,
            workspace: &self.workspace,
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
        self.overlays.retain(
            |overlay| !matches!(&overlay.source, OverlaySource::Track(track) if *track == name),
        );
        self.push_track_as_overlays(loaded.track);
        let file = &mut self.annotation_files[index];
        file.shown = true;
        file.index_available = loaded.index_available;
        file.loaded_window = loaded.loaded_window;
        true
    }

    /// The window of sequence coordinates to read an indexed file for: the viewport and as much
    /// again on each side, so small camera moves don't reload.
    fn annotation_query_window(&self) -> Option<(i64, i64)> {
        current_view_coordinate_window(&self.controller).map(expand_query_window)
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
        let Some(visible) = current_view_coordinate_window(&self.controller) else {
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
        let Ok(conn) = self.open_conn() else {
            return false;
        };
        let mut reloaded = false;
        for index in stale {
            reloaded |= self.load_annotation_file(&conn, index, Some(window));
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
#[pyclass(name = "GraphWidget")]
#[derive(Clone)]
pub struct PyGraphController {
    pages: Vec<Page>,
    current_index: usize,
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
        })
    }

    /// Build a single-page controller for `sg`, loading its graph lazily.
    pub(crate) fn for_sequence_graph(sg: &PySequenceGraph, options: PlotOptions) -> PyResult<Self> {
        Ok(Self {
            pages: vec![Page::Loaded(Box::new(loaded_page_for_sequence_graph(
                sg, options,
            )?))],
            current_index: 0,
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
        })
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
    /// detail : {"normal", "full", "minimal"}
    ///     ``"normal"`` shows truncated labels; ``"full"`` shows
    ///     complete labels; ``"minimal"`` shows the smallest representation.
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
        const HEADER_HEIGHT: u16 = 1;
        let total_area = Rect::new(0, 0, cols, rows);
        let mut buf = Buffer::empty(total_area);
        draw_header(
            &mut buf,
            Rect::new(0, 0, cols, HEADER_HEIGHT.min(rows)),
            self.pages[self.current_index].name(),
        );
        let graph_area = Rect::new(0, HEADER_HEIGHT, cols, rows.saturating_sub(HEADER_HEIGHT));
        self.active()?.render_into(&mut buf, graph_area);
        let frame = serialize_buffer(&buf, cols, rows);
        serde_json::to_string(&frame).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn zoom_in(&mut self) -> PyResult<()> {
        self.active()?.zoom_in();
        Ok(())
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
        self.active()?.go_to_pos(pos, center);
        Ok(())
    }

    /// Highlight the path of nodes covered by `match_obj` in the given colour.
    ///
    /// `color` must be a CSS hex string like `"#ffff00"` or one of the named
    /// ratatui colours (`"yellow"`, `"cyan"`, `"red"`, …).  When omitted the
    /// next unused theme accent colour (slots 0x08–0x0F) is chosen automatically.
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
        self.active()?.go_to_annotation_obj(annotation, center);
        Ok(())
    }

    /// Connect the pieces of `annotation` across nodes at full detail, replacing any
    /// previously focused annotation; ``None`` clears the focus. ``show()`` calls this, so
    /// the most recently shown annotation is the connected one.
    #[pyo3(signature = (annotation=None))]
    pub fn focus_annotation(&mut self, annotation: Option<PyRef<PyAnnotation>>) -> PyResult<()> {
        self.active()?.focus_annotation(annotation.as_deref());
        Ok(())
    }

    /// Hash ID of the annotation whose pieces are connected, or ``None``.
    #[getter]
    fn focused_annotation(&mut self) -> PyResult<Option<String>> {
        Ok(self
            .active()?
            .controller
            .focused_annotation()
            .map(|id| id.to_string()))
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
        self.active()?.go_to_locus(locus, center);
        Ok(())
    }

    /// Return all gene annotations for this sequence graph.
    ///
    /// Delegates to the database, returning every annotation stored for this
    /// sequence graph — independent of which tracks are currently loaded in the
    /// widget.
    ///
    /// The returned ``Annotation`` objects carry no repository context. To
    /// translate one, pass it to
    /// ``SequenceGraph.translate_annotation(region=ann)``, which resolves the
    /// annotation through its own context.
    #[getter]
    pub fn annotations(&mut self) -> PyResult<Vec<PyAnnotation>> {
        self.active()?.list_annotations()
    }

    /// Every annotation-group name visible from this sequence graph — the full menu
    /// `add_track_group` accepts, independent of which tracks are currently displayed.
    #[getter]
    pub fn track_names(&mut self) -> PyResult<Vec<String>> {
        Ok(self
            .active()?
            .controller
            .annotation_group_entries()
            .iter()
            .map(|entry| entry.name.clone())
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

/// Instantiate a `GraphWidget` from a controller and optional viewport.
/// Shared by `PySequenceGraph::plot`, `PyRepository::plot`, and `PySample::plot`.
pub fn build_widget(
    py: Python<'_>,
    ctrl: Py<PyGraphController>,
    rows: Option<u32>,
    cols: Option<u32>,
    colors: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let gen_module = py.import("gen")?;
    let widget_cls = gen_module.getattr("GraphWidget")?;
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
    let widget = widget_cls.call((ctrl,), Some(&kwargs))?;
    Ok(widget.into())
}

#[cfg(test)]
mod tests {
    use r#gen::{
        test_helpers::{setup_block_group, setup_gen_on_disk},
        views::{
            annotation_track::{AnnotationSegment, AnnotationSpan},
            graph_overlay::{GraphOverlay, OverlayContent, OverlaySource},
        },
    };
    use gen_core::{BranchName, HashId, Strand, is_end_node, is_start_node};
    use gen_models::{
        block_group::BlockGroup,
        history::{
            HistoryStore as _,
            dolt::{DoltHistoryStore, active_branch},
        },
    };
    use gen_tui::plotter::PathStyle;
    use pyo3::{exceptions::PyValueError, prelude::*};
    use ratatui::style::Color;
    use serde_json::Value;

    use super::{PlotOptions, PyGraphController, current_theme};
    use crate::python_api::block_group::PySequenceGraph;

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
            context: Some(context.clone()),
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
            .annotations()
            .expect("should find the plotted graph on its original branch");
        sample_controller
            .annotations()
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

    fn make_controller(detail: Option<&str>) -> PyResult<PyGraphController> {
        let ctx = setup_gen_on_disk();
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
        let result = make_controller(Some("bad"));
        Python::attach(|py| match result {
            Ok(_) => panic!("expected a PyValueError for invalid detail value"),
            Err(e) => assert!(e.is_instance_of::<PyValueError>(py)),
        });
    }

    #[test]
    fn test_detail_all_valid_values_accepted() {
        for detail in [None, Some("normal"), Some("full"), Some("minimal")] {
            let result = make_controller(detail);
            assert!(result.is_ok(), "detail={detail:?} should be accepted");
        }
    }

    #[test]
    fn test_render_frame_returns_valid_json() {
        Python::initialize();
        Python::attach(|_py| {
            let mut ctrl = make_controller(None).unwrap();
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
        let mut controller = make_controller(Some("full")).expect("should create a controller");
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
                        start: node.sequence_start,
                        end: node.sequence_end,
                        strand: Strand::Forward,
                    })
                    .collect(),
            }),
            source: OverlaySource::Track("gene".to_string()),
            style: PathStyle::new(Color::Red),
        });
        let unfocused = rendered_text(&mut controller);
        assert!(
            !unfocused
                .chars()
                .any(|glyph| ('\u{2801}'..='\u{28ff}').contains(&glyph)),
            "should draw no connectors without a focused annotation"
        );
        controller
            .active()
            .expect("should have an active page")
            .controller
            .set_focused_annotation(Some(HashId::convert_str("gene")));

        for detail in ["full", "normal", "minimal", "full"] {
            controller.set_detail(detail).expect("should change detail");
            let text = rendered_text(&mut controller);
            let full = detail == "full";
            assert_eq!(
                text.contains('═'),
                full,
                "annotation bars at {detail} detail"
            );
            assert_eq!(text.contains('▶'), full, "direction cap at {detail} detail");
            assert_eq!(
                text.chars()
                    .any(|glyph| ('\u{2801}'..='\u{28ff}').contains(&glyph)),
                full,
                "annotation connector at {detail} detail"
            );
            if full {
                assert_eq!(
                    text.matches("gene").count(),
                    1,
                    "should label the span once"
                );
            }
        }

        controller
            .remove_annotation("gene")
            .expect("should remove annotation");
        let text = rendered_text(&mut controller);
        assert!(!text.contains('═'), "should remove annotation bars");
        assert!(!text.contains("gene"), "should remove the annotation label");
        assert!(
            !text
                .chars()
                .any(|glyph| ('\u{2801}'..='\u{28ff}').contains(&glyph)),
            "should remove annotation connectors"
        );
        assert_eq!(
            controller
                .focused_annotation()
                .expect("should read the focused annotation"),
            None,
            "should drop the focus on a removed annotation"
        );
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
