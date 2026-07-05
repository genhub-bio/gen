use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    path::PathBuf,
};

use r#gen::{
    get_connection,
    views::{
        annotation_groups::{annotation_group_names, load_annotation_group_entries},
        annotation_track::{
            AnnotationSpan, AnnotationTrack, annotation_span_from_graph_locus,
            graph_locus_from_annotation_span, span_covered_by_later, span_is_single_node,
            span_label_text,
        },
        annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
        gen_graph_widget::{
            GenGraphNodeRenderer, GenGraphNodeSizer, highlight_locus, locus_label_bounds,
            viewport_pos_map,
        },
        graph_overlay::{GraphOverlay, OverlaySource},
        inline_label_placement::{draw_annotation_overflow_note, draw_label_near_pos},
    },
};
use gen_annotations::projection::annotation_segments;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice, project_path};
use gen_models::{
    annotations::{Annotation, AnnotationError},
    block_group::BlockGroup,
    db::GraphConnection,
    locus::GraphLocus,
};
use gen_tui::{
    LineStyle, geometry::WorldPos, graph_controller::GraphController, graph_widget::GraphWidget,
    layout::VisualDetail, plotter::PathStyle, theme::current_theme,
};
use petgraph::{graph::NodeIndex, visit::NodeIndexable};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::StatefulWidget,
};
use serde::Serialize;

use crate::python_api::{
    annotation::PyAnnotation,
    block_group::PySequenceGraph,
    graph_search::{PyGraphLocus, PyGraphPos},
    utils::block_group_err_to_pyerr,
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

/// Build a `GraphLocus` from an `AnnotationSpan` using only nodes visible in
/// the current viewport (`pos_map`).  Segments whose node is not in the map
/// are silently dropped; `locus_label_bounds` handles partial coverage fine.
fn locus_from_span_and_pos_map(
    span: &AnnotationSpan,
    pos_map: &HashMap<GraphNode, (WorldPos, (u64, u64))>,
) -> GraphLocus {
    let node_by_id: HashMap<HashId, GraphNode> = pos_map.keys().map(|n| (n.node_id, *n)).collect();
    let slices = span
        .segments
        .iter()
        .filter_map(|seg| {
            let node = *node_by_id.get(&seg.node_id)?;
            let start = (seg.start - node.sequence_start).max(0) as usize;
            let end = (seg.end - node.sequence_start).max(0) as usize;
            Some(GraphNodeSlice {
                block: node,
                start,
                end,
                strand: seg.strand,
            })
        })
        .collect();
    GraphLocus { slices }
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

/// View state and renderer for a single sequence graph "page" of a
/// `PyGraphController`. A plain `GraphWidget` has exactly one page; a
/// `Sample`-backed widget pages through several.
///
/// # Thread safety
///
/// ipykernel 6+ runs cell code in a thread-pool executor (e.g. thread 12) while
/// anywidget comm/observe callbacks fire on the asyncio ioloop (thread 1).
/// `GraphPage` is therefore created on one thread and accessed from another,
/// so it must be `Send`.  `GraphHandle` contains `Rc<GraphConnection>` which is
/// `!Send`, so we store only the DB path and open a fresh connection per operation
/// instead of holding a live handle.
#[derive(Clone)]
struct GraphPage {
    name: String,
    db_path: PathBuf,
    pub(crate) block_group_id: Option<HashId>,
    controller: GraphController<GenGraph, GenGraphNodeSizer>,
    overlays: Vec<GraphOverlay>,
    /// Set by `show_path`, cleared by `clear_path`/`clear_highlights`.
    /// Restored by `reapply_highlights` so it survives zoom/detail changes
    /// the same way `overlays` do.
    path_highlight: Option<(PathStyle, Vec<GraphNode>)>,
    /// Set to `true` once annotation groups have been loaded (auto or with colors).
    /// Survives cloning so that cell-display clones do not double-load.
    annotation_groups_loaded: bool,
}

/// The information needed to lazily build a `GraphPage` on first visit,
/// without holding a live (non-`Send`) database handle in the meantime.
#[derive(Clone)]
struct PageRef {
    name: String,
    db_path: PathBuf,
    block_group_id: HashId,
}

/// One page of a `PyGraphController`: either already loaded, or pending lazy
/// construction the first time it becomes the active page.
#[derive(Clone)]
enum Page {
    Loaded(GraphPage),
    Pending(PageRef),
}

impl Page {
    fn name(&self) -> &str {
        match self {
            Page::Loaded(page) => &page.name,
            Page::Pending(page_ref) => &page_ref.name,
        }
    }
}

impl GraphPage {
    fn new(name: String, db_path: PathBuf, graph: GenGraph) -> Self {
        let mut controller = GraphController::new(graph, GenGraphNodeSizer);
        controller.set_detail_level(VisualDetail::Truncated);
        controller.hide_cursor();
        Self {
            name,
            db_path,
            block_group_id: None,
            controller,
            overlays: Vec::new(),
            path_highlight: None,
            annotation_groups_loaded: false,
        }
    }

    fn open_conn(&self) -> PyResult<GraphConnection> {
        get_connection(&self.db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn all_node_ids(&self) -> HashSet<HashId> {
        self.controller
            .graph()
            .nodes()
            .filter(|n| !is_start_node(n.node_id) && !is_end_node(n.node_id))
            .map(|n| n.node_id)
            .collect()
    }

    fn load_group_as_track(
        &self,
        conn: &GraphConnection,
        group: &str,
    ) -> Result<AnnotationTrack, AnnotationError> {
        let block_group_id = self.block_group_id.ok_or(AnnotationError::DatabaseError(
            rusqlite::Error::QueryReturnedNoRows,
        ))?;
        let current_block_group = BlockGroup::get_by_id(conn, &block_group_id)
            .map_err(|_| AnnotationError::DatabaseError(rusqlite::Error::QueryReturnedNoRows))?;
        let node_ids = self.all_node_ids();
        let entry = load_annotation_group_entries(conn, &current_block_group)
            .into_iter()
            .find(|entry| entry.name == group)
            .ok_or_else(|| AnnotationError::DatabaseError(rusqlite::Error::QueryReturnedNoRows))?;
        let spans = load_annotations_for_group(&AnnotationGroupTrackRequest {
            conn,
            current_block_group: &current_block_group,
            entry: &entry,
            node_ids: &node_ids,
        })?;
        Ok(AnnotationTrack::new(group.to_string(), spans))
    }

    pub(crate) fn auto_load_annotation_groups(&mut self, conn: &GraphConnection) {
        let Some(bg_id) = self.block_group_id else {
            return;
        };
        let Ok(current_block_group) = BlockGroup::get_by_id(conn, &bg_id) else {
            return;
        };
        for name in annotation_group_names(conn, &current_block_group) {
            if let Ok(track) = self.load_group_as_track(conn, &name) {
                self.push_track_as_overlays(track);
            }
        }
        self.annotation_groups_loaded = true;
    }

    /// Load annotation groups applying per-annotation colors supplied by the Python layer.
    ///
    /// `color_map` maps annotation ID hex strings to an optional CSS hex color.
    /// `Some(color)` → paint with that color; `None` → skip the annotation entirely.
    /// Annotations whose ID is absent from the map fall back to the auto theme palette.
    fn load_annotation_groups_with_colors_inner(
        &mut self,
        conn: &GraphConnection,
        color_map: &HashMap<String, Option<String>>,
    ) {
        let Some(bg_id) = self.block_group_id else {
            return;
        };
        let Ok(current_block_group) = BlockGroup::get_by_id(conn, &bg_id) else {
            return;
        };
        for name in annotation_group_names(conn, &current_block_group) {
            if let Ok(track) = self.load_group_as_track(conn, &name) {
                self.push_track_as_overlays_with_colors(track, color_map);
            }
        }
        self.annotation_groups_loaded = true;
    }

    fn push_track_as_overlays_with_colors(
        &mut self,
        track: AnnotationTrack,
        color_map: &HashMap<String, Option<String>>,
    ) {
        let track_name = track.name;
        let mut spans = track.annotations;
        spans.sort_by_key(|s| {
            -(s.segments
                .iter()
                .map(|seg| seg.end - seg.start)
                .sum::<i64>())
        });
        let theme = current_theme();
        let accent_base = self
            .overlays
            .iter()
            .filter(|o| !matches!(o.source, OverlaySource::Adhoc))
            .count();
        let mut accent_offset = 0usize;
        let spans_with_styles: Vec<(AnnotationSpan, PathStyle)> = spans
            .into_iter()
            .filter_map(|span| {
                let id_hex = span.id.to_string();
                match color_map.get(&id_hex) {
                    // None map value means the caller requested this annotation be hidden.
                    Some(None) => None,
                    Some(Some(hex)) => {
                        let color = parse_hex_color(hex)
                            .unwrap_or_else(|_| theme[0x08 + ((accent_base + accent_offset) % 8)]);
                        accent_offset += 1;
                        Some((span, PathStyle::new(color)))
                    }
                    None => {
                        // not in map → auto theme color
                        let color = theme[0x08 + ((accent_base + accent_offset) % 8)];
                        accent_offset += 1;
                        Some((span, PathStyle::new(color)))
                    }
                }
            })
            .collect();
        let loci: Vec<Option<GraphLocus>> = spans_with_styles
            .iter()
            .map(|(span, _)| graph_locus_from_annotation_span(span, self.controller.graph()))
            .collect();
        for (locus, (_, style)) in loci.iter().zip(spans_with_styles.iter()) {
            if let Some(l) = locus {
                highlight_locus(&mut self.controller, l, *style);
            }
        }
        for (span, style) in spans_with_styles {
            self.overlays.push(GraphOverlay {
                span,
                source: OverlaySource::Track(track_name.clone()),
                style,
            });
        }
    }

    fn push_track_as_overlays(&mut self, track: AnnotationTrack) {
        let track_name = track.name;
        let mut spans = track.annotations;
        // Sort longest-first so shorter (inner) annotations paint on top.
        spans.sort_by_key(|s| {
            -(s.segments
                .iter()
                .map(|seg| seg.end - seg.start)
                .sum::<i64>())
        });
        let theme = current_theme();
        let accent_base = self
            .overlays
            .iter()
            .filter(|o| !matches!(o.source, OverlaySource::Adhoc))
            .count();
        let spans_with_styles: Vec<(AnnotationSpan, PathStyle)> = spans
            .into_iter()
            .enumerate()
            .map(|(i, span)| (span, PathStyle::new(theme[0x08 + ((accent_base + i) % 8)])))
            .collect();
        // Resolve loci with immutable graph borrow.
        let loci: Vec<Option<GraphLocus>> = spans_with_styles
            .iter()
            .map(|(span, _)| graph_locus_from_annotation_span(span, self.controller.graph()))
            .collect();
        // Apply highlights with mutable controller borrow.
        for (locus, (_, style)) in loci.iter().zip(spans_with_styles.iter()) {
            if let Some(l) = locus {
                highlight_locus(&mut self.controller, l, *style);
            }
        }
        for (span, style) in spans_with_styles {
            self.overlays.push(GraphOverlay {
                span,
                source: OverlaySource::Track(track_name.clone()),
                style,
            });
        }
    }

    fn navigate_to_span(&mut self, span: &AnnotationSpan) {
        let Some(locus) = graph_locus_from_annotation_span(span, self.controller.graph()) else {
            return;
        };
        let slice = &locus.slices[0];
        self.go_to_pos(&PyGraphPos::new(slice.block, slice.start));
    }

    /// Render the graph, overlay labels, and annotation tracks into `buf` within `graph_area`.
    ///
    /// Shared by the standalone `render_frame` pymethod and `PySampleController`,
    /// which renders a header row above the graph and offsets `graph_area` accordingly.
    fn render_into(&mut self, buf: &mut Buffer, graph_area: Rect) -> PyResult<()> {
        {
            let conn = self.open_conn()?;
            let renderer = GenGraphNodeRenderer::new(&conn);
            GraphWidget::with_renderer(renderer).render(graph_area, buf, &mut self.controller);
        }

        // Draw overlay labels. Midpoints are recomputed each render because the viewport may have changed.
        let mut labeled_overlays: Vec<_> = self
            .overlays
            .iter()
            .filter(|o| !o.span.name.is_empty())
            .collect();
        if !labeled_overlays.is_empty() {
            // Sort longest-first so the covered-by-later check matches highlight order.
            labeled_overlays
                .sort_by_key(|o| -(o.span.segments.iter().map(|s| s.end - s.start).sum::<i64>()));
            let span_refs: Vec<&AnnotationSpan> =
                labeled_overlays.iter().map(|o| &o.span).collect();
            let theme = current_theme();
            let pos_map = viewport_pos_map(&self.controller);
            let detail_level = self.controller.get_detail_level();
            let mut hidden_count: usize = 0;
            for (idx, overlay) in labeled_overlays.iter().enumerate() {
                let color = match overlay.style.color {
                    ratatui::style::Color::Reset => theme[0x06],
                    c => c,
                };
                if span_covered_by_later(&overlay.span, idx, &span_refs) {
                    hidden_count += 1;
                    continue;
                }
                if detail_level == gen_tui::layout::VisualDetail::Truncated
                    && span_is_single_node(&overlay.span)
                {
                    hidden_count += 1;
                    continue;
                }
                let locus = locus_from_span_and_pos_map(&overlay.span, &pos_map);
                let Some((left_pos, right_pos)) = locus_label_bounds(
                    &locus,
                    &pos_map,
                    detail_level,
                    &self.controller.viewport_state,
                ) else {
                    continue;
                };
                let max_distance = if detail_level == gen_tui::layout::VisualDetail::Minimal {
                    10
                } else {
                    5
                };
                let label = span_label_text(&overlay.span);
                if draw_label_near_pos(
                    buf,
                    graph_area,
                    (left_pos, right_pos),
                    &label,
                    color,
                    &self.controller.viewport_state,
                    max_distance,
                )
                .is_none()
                {
                    hidden_count += 1;
                }
            }
            let note_style = Style::default().fg(theme[0x09]).bg(theme[0x00]);
            draw_annotation_overflow_note(
                buf,
                graph_area,
                hidden_count,
                note_style,
                detail_level != gen_tui::layout::VisualDetail::Full,
            );
        }

        Ok(())
    }

    fn resolve_color(&mut self, color: Option<&str>) -> PyResult<Color> {
        match color {
            None => Ok(self.controller.next_accent_color()),
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
}

impl GraphPage {
    /// Whether annotation groups have already been loaded into this page.
    ///
    /// Returns ``true`` after either [`Self::trigger_auto_load`] or
    /// [`Self::load_annotation_groups_with_colors`] has been called. Cloned
    /// pages inherit this flag, preventing double-loads on cell re-display.
    fn annotations_loaded(&self) -> bool {
        self.annotation_groups_loaded
    }

    /// Load all annotation groups using the automatic theme-colour palette.
    ///
    /// Called by ``GraphWidget`` when no ``colors`` mapping is provided to
    /// ``plot()``. No-op if annotations have already been loaded.
    fn trigger_auto_load(&mut self) -> PyResult<()> {
        if self.annotation_groups_loaded {
            return Ok(());
        }
        let conn = self.open_conn()?;
        self.auto_load_annotation_groups(&conn);
        Ok(())
    }

    /// Load annotation groups applying per-annotation colours from a Python-resolved map.
    ///
    /// `color_map` maps annotation ID hex strings to a CSS hex colour string
    /// (e.g. ``"#ff4444"``) or ``None`` to hide that annotation entirely.
    /// Annotations absent from the map fall back to the auto theme palette.
    ///
    /// Called by ``GraphWidget`` after it evaluates the ``colors`` callable/dict/list
    /// provided to ``plot()``. No-op if annotations have already been loaded.
    fn load_annotation_groups_with_colors(
        &mut self,
        color_map: &HashMap<String, Option<String>>,
    ) -> PyResult<()> {
        if self.annotation_groups_loaded {
            return Ok(());
        }
        let conn = self.open_conn()?;
        self.load_annotation_groups_with_colors_inner(&conn, color_map);
        Ok(())
    }

    /// Set the level of node detail.
    ///
    /// Parameters
    /// detail : {"normal", "full", "minimal"}
    ///     ``"normal"`` shows truncated labels (default); ``"full"`` shows
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
        self.reapply_highlights();
        Ok(())
    }

    pub fn truncate_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Truncated);
        self.reapply_highlights();
    }

    pub fn full_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Full);
        self.reapply_highlights();
    }

    pub fn minimize_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Minimal);
        self.reapply_highlights();
    }

    /// Re-register all overlays (and the path highlight, if any) using the current
    /// detail level.
    ///
    /// Highlight column offsets are clamped at registration time, so they go stale
    /// when the detail level changes. Call this after any zoom or detail change.
    fn reapply_highlights(&mut self) {
        self.controller.clear_all_highlights();
        // Collect loci with immutable graph borrow, then apply with mutable controller borrow.
        let loci_with_styles: Vec<(GraphLocus, PathStyle)> = self
            .overlays
            .iter()
            .filter_map(|o| {
                graph_locus_from_annotation_span(&o.span, self.controller.graph())
                    .map(|l| (l, o.style))
            })
            .collect();
        for (locus, style) in &loci_with_styles {
            highlight_locus(&mut self.controller, locus, *style);
        }
        if let Some((style, nodes)) = self.path_highlight.clone() {
            self.controller.set_path_highlight(style, nodes);
        }
    }

    fn zoom_in(&mut self) {
        self.controller.zoom_in();
        self.reapply_highlights();
    }

    fn zoom_out(&mut self) {
        self.controller.zoom_out();
        self.reapply_highlights();
    }

    fn handle_click(&mut self, col: u16, row: u16) -> bool {
        self.controller.handle_click(col, row)
    }

    fn move_by(&mut self, dx: i16, dy: i16) {
        self.controller.move_by_terminal(dx, dy);
        self.controller.sync_cursor_to_closest_node();
    }

    fn go_to_pos(&mut self, pos: &PyGraphPos) {
        let block = pos.inner.block;
        self.controller.set_detail_level(VisualDetail::Full);

        // Find the partition this block is on, load it and anchor it
        let Ok((partition_idx, _)) = self
            .controller
            .partition_controller
            .partition_table
            .find_node(&block)
        else {
            return;
        };
        let _ = self.controller.ensure_partition_loaded(partition_idx);
        let _ = self.controller.set_anchor_partition(partition_idx);

        // Go to node works with an index, not a raw GraphNode
        let domain_idx = NodeIndex::new(NodeIndexable::to_index(self.controller.graph(), block));

        // The cursor is positioned using normalized coordinates between 0 and 1,
        // relative to the area in which the node (block) is represented in the plot.
        let block_len = block.length();
        let frac_x = if block_len > 1 {
            pos.inner.offset as f64 / (block_len - 1) as f64
        } else {
            0.0
        };

        self.controller.go_to_node(domain_idx, (frac_x, 0.5));
        self.controller.queue_snap_left();
        self.controller.hide_cursor();
        self.reapply_highlights();
    }

    /// Highlight the path of nodes covered by `match_obj` in the given colour.
    ///
    /// `color` must be a CSS hex string like `"#ffff00"` or one of the named
    /// ratatui colours (`"yellow"`, `"cyan"`, `"red"`, …).  When omitted the
    /// next unused theme accent colour (slots 0x08–0x0F) is chosen automatically.
    fn highlight_match(&mut self, locus: &PyGraphLocus, color: Option<&str>) -> PyResult<()> {
        let c = self.resolve_color(color)?;
        let style = PathStyle::new(c)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        highlight_locus(&mut self.controller, &locus.inner, style);
        self.overlays.push(GraphOverlay {
            span: annotation_span_from_graph_locus(&locus.inner, ""),
            source: OverlaySource::Adhoc,
            style,
        });
        Ok(())
    }

    /// Remove all highlights from the graph, including any path shown by `show_path`.
    fn clear_highlights(&mut self) {
        self.overlays.clear();
        self.path_highlight = None;
        self.controller.clear_all_highlights();
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
        let block_group_id = self.block_group_id.ok_or_else(|| {
            PyRuntimeError::new_err(
                "show_path() requires a sequence graph; obtain the widget via SequenceGraph.plot()",
            )
        })?;

        let highlight_color = self.resolve_color(color)?;
        let conn = self.open_conn()?;

        let path = BlockGroup::get_current_path(&conn, &block_group_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let path_blocks = path.blocks(&conn).unwrap_or_default();
        let projected_path = project_path(self.controller.graph(), &path_blocks);

        let path_nodes: Vec<GraphNode> = projected_path
            .iter()
            .filter_map(|(node, _)| {
                if node.node_id != PATH_START_NODE_ID && node.node_id != PATH_END_NODE_ID {
                    Some(*node)
                } else {
                    None
                }
            })
            .collect();

        if path_nodes.is_empty() {
            return Err(PyRuntimeError::new_err(
                "Path nodes not found in current graph state",
            ));
        }

        let style = PathStyle::new(highlight_color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);

        self.controller
            .set_path_highlight(style, path_nodes.clone());
        self.path_highlight = Some((style, path_nodes));
        Ok(())
    }

    /// Clear path highlighting previously applied by `show_path`.
    pub fn clear_path(&mut self) {
        self.path_highlight = None;
        self.controller.clear_all_highlights();
        self.reapply_highlights(); // restore overlays cleared by clear_all_highlights
    }

    /// Load annotations from the database by group name and add them as inline graph overlays.
    pub fn add_track_group(&mut self, group: &str) -> PyResult<()> {
        let conn = self.open_conn()?;
        let track = self
            .load_group_as_track(&conn, group)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.push_track_as_overlays(track);
        Ok(())
    }

    /// Add a list of `Annotation` objects as inline graph overlays grouped under `name`.
    pub fn add_track_annotations(&mut self, annotations: Vec<PyRef<PyAnnotation>>, name: &str) {
        let spans: Vec<AnnotationSpan> =
            annotations.iter().map(|a| annotation_to_span(a)).collect();
        self.push_track_as_overlays(AnnotationTrack::new(name, spans));
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
        use std::io::Cursor;

        use r#gen::views::annotations::{parse_translated_bed, parse_translated_gff};
        use gen_annotations::translate::{bed::translate_bed, gff::translate_gff};
        use gen_models::sample::Sample;

        let name = display_name.unwrap_or(file_path);
        let node_ids = self.all_node_ids();
        let sample = from_sample.unwrap_or(Sample::DEFAULT_NAME);

        let track = if let Some(bg_id) = self.block_group_id {
            let conn = self.open_conn()?;
            let bg = BlockGroup::get_by_id(&conn, &bg_id)
                .map_err(|e| PyRuntimeError::new_err(format!("Block group not found: {e}")))?;

            let path = std::path::Path::new(file_path);
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            let mut buffer: Vec<u8> = Vec::new();
            let translate_result: Result<(), String> = match ext.as_str() {
                "gff" | "gff3" => translate_gff(
                    &conn,
                    &bg.collection_name,
                    sample,
                    BufReader::new(
                        File::open(file_path)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                    ),
                    &mut buffer,
                )
                .map_err(|e| e.to_string()),
                "bed" => translate_bed(
                    &conn,
                    &bg.collection_name,
                    sample,
                    File::open(file_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                    &mut buffer,
                )
                .map_err(|e| e.to_string()),
                other => {
                    return Err(PyRuntimeError::new_err(format!(
                        "unsupported annotation file type: {other:?}; expected .gff, .gff3, or .bed"
                    )));
                }
            };

            if let Err(e) = translate_result {
                return Err(PyRuntimeError::new_err(e.to_string()));
            }

            let spans = if !buffer.is_empty() {
                match ext.as_str() {
                    "gff" | "gff3" => {
                        parse_translated_gff(Cursor::new(buffer), &node_ids, name, HashMap::new())
                    }
                    _ => parse_translated_bed(Cursor::new(buffer), &node_ids, name, HashMap::new()),
                }
            } else {
                // Buffer empty means translation found no matching sequences —
                // file may already be in translated (hash-ID) format.
                load_track_from_file(file_path, name, &node_ids)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .annotations
            };

            AnnotationTrack::new(name.to_string(), spans)
        } else {
            load_track_from_file(file_path, name, &node_ids)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };

        self.push_track_as_overlays(track);
        Ok(())
    }

    /// Navigate to an `Annotation` object.
    pub fn go_to_annotation_obj(&mut self, annotation: &PyAnnotation) {
        let span = annotation_to_span(annotation);
        self.navigate_to_span(&span);
    }

    /// Highlight an `Annotation` on the graph without a label.
    pub fn highlight_annotation_obj(
        &mut self,
        annotation: &PyAnnotation,
        color: Option<&str>,
    ) -> PyResult<()> {
        let c = self.resolve_color(color)?;
        let style = PathStyle::new(c)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        let span = annotation_to_span(annotation);
        let locus = annotation
            .locus
            .clone()
            .or_else(|| graph_locus_from_annotation_span(&span, self.controller.graph()));
        if let Some(locus) = locus {
            highlight_locus(&mut self.controller, &locus, style);
        }
        self.overlays.push(GraphOverlay {
            span,
            source: OverlaySource::Adhoc,
            style,
        });
        Ok(())
    }

    /// Navigate to a `GraphLocus`.
    pub fn go_to_locus(&mut self, locus: &PyGraphLocus) {
        let slice = &locus.inner.slices[0];
        self.go_to_pos(&PyGraphPos::new(slice.block, slice.start));
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
    pub fn list_annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        let bg_id = self.block_group_id.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "list_annotations() requires a sequence graph; \
                 create the widget via SequenceGraph.plot()",
            )
        })?;
        let conn = self.open_conn()?;
        let block_group = BlockGroup::get_by_id(&conn, bg_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let annotations = Annotation::query_with_lineage(
            &conn,
            &block_group.collection_name,
            &block_group.sample_name,
            &block_group.name,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(annotations
            .into_iter()
            .map(|a| PyAnnotation {
                ann_segments: annotation_segments(&conn, &a),
                inner: a,
                context: None,
                source_block_group_id: Some(*bg_id),
                locus: None,
            })
            .collect())
    }

    /// Return a JSON list of track names currently loaded (from `add_track_group`,
    /// `add_track_file`, or auto-loaded annotation groups).
    pub fn get_track_names(&self) -> PyResult<String> {
        let mut seen = std::collections::HashSet::new();
        let names: Vec<&str> = self
            .overlays
            .iter()
            .filter_map(|o| match &o.source {
                OverlaySource::Track(name) => Some(name.as_str()),
                OverlaySource::Annotation(_) | OverlaySource::Adhoc => None,
            })
            .filter(|n| seen.insert(*n))
            .collect();
        serde_json::to_string(&names).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove all overlays belonging to the track `name` (loaded via
    /// `add_track_group` / `add_track_file` / auto-loaded annotation groups).
    pub fn remove_track(&mut self, name: &str) {
        self.overlays
            .retain(|o| !matches!(&o.source, OverlaySource::Track(n) if n == name));
        self.reapply_highlights();
    }

    /// Clear all annotations from the graph.
    pub fn clear_all_annotations(&mut self) {
        // Keep only ad hoc highlights (e.g. search matches); drop everything
        // added via a track or `add_annotation`, then repaint what remains.
        self.overlays
            .retain(|o| matches!(o.source, OverlaySource::Adhoc));
        self.reapply_highlights();
    }

    /// Add annotations rendered directly on the graph canvas.
    /// Annotations are tinted with an accent colour and labelled below their span.
    pub fn add_annotation(
        &mut self,
        annotations: Vec<PyRef<PyAnnotation>>,
        track_name: Option<String>,
    ) {
        let existing_color = track_name.as_deref().and_then(|name| {
            self.overlays.iter().find_map(|o| match &o.source {
                OverlaySource::Annotation(n) if n == name => Some(o.style.color),
                _ => None,
            })
        });
        let color = existing_color.unwrap_or_else(|| self.controller.next_accent_color());
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        let source = match &track_name {
            Some(name) => OverlaySource::Annotation(name.clone()),
            None => OverlaySource::Adhoc,
        };
        // Collect spans and loci with immutable graph borrow, then apply with mutable controller borrow.
        let spans_and_loci: Vec<(AnnotationSpan, Option<GraphLocus>)> = annotations
            .iter()
            .map(|ann| {
                let span = annotation_to_span(ann);
                let locus = ann
                    .locus
                    .clone()
                    .or_else(|| graph_locus_from_annotation_span(&span, self.controller.graph()));
                (span, locus)
            })
            .collect();
        for (_, locus) in &spans_and_loci {
            if let Some(locus) = locus {
                highlight_locus(&mut self.controller, locus, style);
            }
        }
        for (span, _) in spans_and_loci {
            self.overlays.push(GraphOverlay {
                span,
                source: source.clone(),
                style,
            });
        }
    }

    /// Return a JSON list of annotation names currently loaded (from
    /// `add_annotation`; annotations loaded as part of a track keep their own
    /// name here too, separately from the track's name).
    pub fn get_annotation_names(&self) -> PyResult<String> {
        let mut seen = std::collections::HashSet::new();
        let names: Vec<&str> = self
            .overlays
            .iter()
            .map(|o| o.span.name.as_str())
            .filter(|n| !n.is_empty() && seen.insert(*n))
            .collect();
        serde_json::to_string(&names).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove all overlays whose annotation name matches `name`, regardless of
    /// which track (if any) they belong to. If the same name was added more
    /// than once, every copy is removed.
    pub fn remove_annotation(&mut self, name: &str) {
        self.overlays.retain(|o| o.span.name != name);
        self.reapply_highlights();
    }
}

// File loading helper

/// Parse a translated GFF3 or BED file into an `AnnotationTrack`.
///
/// The file must use node hash-ID strings as reference names (i.e. the
/// "translated" format produced by gen's GFF/BED translation step).
fn load_track_from_file(
    file_path: &str,
    display_name: &str,
    node_filter: &HashSet<HashId>,
) -> Result<AnnotationTrack, Box<dyn std::error::Error>> {
    use r#gen::views::annotations::{parse_translated_bed_file, parse_translated_gff_file};
    let path = std::path::Path::new(file_path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let spans = match ext.as_str() {
        "gff" | "gff3" => parse_translated_gff_file(path, node_filter, display_name)?,
        "bed" => parse_translated_bed_file(path, node_filter, display_name)?,
        other => {
            return Err(format!(
                "unsupported annotation file type: {other:?}; expected .gff, .gff3, or .bed"
            )
            .into());
        }
    };
    Ok(AnnotationTrack::new(display_name.to_string(), spans))
}

/// Build an eagerly-loaded `GraphPage` for a `PySequenceGraph`, loading its
/// graph and auto-loading any stored annotation groups.
fn loaded_page_for_sequence_graph(sg: &PySequenceGraph) -> PyResult<GraphPage> {
    let context = sg.context.clone().ok_or_else(|| {
        PyRuntimeError::new_err(
            "plot() requires a Repository context; obtain SequenceGraphs via Repository by query or id.",
        )
    })?;
    let graph_conn = context.graph().conn();
    let db_path = graph_conn
        .path()
        .map(PathBuf::from)
        .ok_or_else(|| PyRuntimeError::new_err("graph DB has no file path"))?;
    let graph = BlockGroup::get_graph(graph_conn, &sg.id).map_err(block_group_err_to_pyerr)?;
    let mut page = GraphPage::new(sg.name.clone(), db_path, graph);
    page.block_group_id = Some(sg.id);
    Ok(page)
}

/// Capture the information needed to lazily build a page for `sg` later,
/// without holding a live (non-`Send`) database handle in the meantime.
fn page_ref_for_sequence_graph(sg: &PySequenceGraph) -> PyResult<PageRef> {
    let context = sg.context.clone().ok_or_else(|| {
        PyRuntimeError::new_err(
            "plot() requires a Repository context; obtain SequenceGraphs via Repository by query or id.",
        )
    })?;
    let db_path = context
        .graph()
        .conn()
        .path()
        .map(PathBuf::from)
        .ok_or_else(|| PyRuntimeError::new_err("graph DB has no file path"))?;
    Ok(PageRef {
        name: sg.name.clone(),
        db_path,
        block_group_id: sg.id,
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
    /// Wrap a single, already-loaded graph as a one-page controller.
    pub fn new(db_path: PathBuf, graph: GenGraph) -> Self {
        Self {
            pages: vec![Page::Loaded(GraphPage::new(String::new(), db_path, graph))],
            current_index: 0,
        }
    }

    /// Build a single-page controller for `sg`, loading its graph eagerly.
    pub(crate) fn for_sequence_graph(sg: &PySequenceGraph) -> PyResult<Self> {
        Ok(Self {
            pages: vec![Page::Loaded(loaded_page_for_sequence_graph(sg)?)],
            current_index: 0,
        })
    }

    /// Build a multi-page controller paging through every sequence graph in
    /// `block_groups`. Each page's graph is loaded lazily on first visit.
    pub(crate) fn for_sample(block_groups: &[PySequenceGraph]) -> PyResult<Self> {
        if block_groups.is_empty() {
            return Err(PyRuntimeError::new_err(
                "Sample has no sequence graphs to plot",
            ));
        }
        let pages = block_groups
            .iter()
            .map(|sg| page_ref_for_sequence_graph(sg).map(Page::Pending))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            pages,
            current_index: 0,
        })
    }

    fn active(&mut self) -> PyResult<&mut GraphPage> {
        let page = &mut self.pages[self.current_index];
        if let Page::Pending(page_ref) = page {
            let conn = get_connection(&page_ref.db_path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let graph = BlockGroup::get_graph(&conn, &page_ref.block_group_id)
                .map_err(block_group_err_to_pyerr)?;
            let mut loaded = GraphPage::new(page_ref.name.clone(), page_ref.db_path.clone(), graph);
            loaded.block_group_id = Some(page_ref.block_group_id);
            *page = Page::Loaded(loaded);
        }
        match page {
            Page::Loaded(page) => Ok(page),
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
    ///     ``"normal"`` shows truncated labels (default); ``"full"`` shows
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
        self.active()?.render_into(&mut buf, graph_area)?;
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

    fn go_to_pos(&mut self, pos: &PyGraphPos) -> PyResult<()> {
        self.active()?.go_to_pos(pos);
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

    /// Remove all highlights from the graph, including any path shown by `show_path`.
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
    pub fn clear_path(&mut self) -> PyResult<()> {
        self.active()?.clear_path();
        Ok(())
    }

    /// Load annotations from the database by group name and add them as a
    /// horizontal track panel below the graph.
    pub fn add_track_group(&mut self, group: &str) -> PyResult<()> {
        self.active()?.add_track_group(group)
    }

    /// Build a track panel from a list of `Annotation` objects.
    /// Each `Annotation` becomes one span; all are grouped under `name`.
    pub fn add_track_annotations(
        &mut self,
        annotations: Vec<PyRef<PyAnnotation>>,
        name: &str,
    ) -> PyResult<()> {
        self.active()?.add_track_annotations(annotations, name);
        Ok(())
    }

    /// Load annotations from a GFF3 or BED file and add them as a
    /// horizontal track panel below the graph.
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
        self.active()?
            .add_track_file(file_path, display_name, from_sample)
    }

    /// Navigate to an `Annotation` object.
    pub fn go_to_annotation_obj(&mut self, annotation: &PyAnnotation) -> PyResult<()> {
        self.active()?.go_to_annotation_obj(annotation);
        Ok(())
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
    pub fn go_to_locus(&mut self, locus: &PyGraphLocus) -> PyResult<()> {
        self.active()?.go_to_locus(locus);
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
    pub fn list_annotations(&mut self) -> PyResult<Vec<PyAnnotation>> {
        self.active()?.list_annotations()
    }

    /// Return a JSON list of track-panel annotation names currently loaded.
    pub fn get_track_names(&mut self) -> PyResult<String> {
        self.active()?.get_track_names()
    }

    /// Remove a track-panel annotation by name.
    pub fn remove_track(&mut self, name: &str) -> PyResult<()> {
        self.active()?.remove_track(name);
        Ok(())
    }

    /// Clear all track-panel annotations.
    pub fn clear_all_annotations(&mut self) -> PyResult<()> {
        self.active()?.clear_all_annotations();
        Ok(())
    }

    /// Add annotations rendered directly on the graph canvas.
    /// Annotations are tinted with an accent colour and labelled below their span.
    pub fn add_annotation(
        &mut self,
        annotations: Vec<PyRef<PyAnnotation>>,
        track_name: Option<String>,
    ) -> PyResult<()> {
        self.active()?.add_annotation(annotations, track_name);
        Ok(())
    }

    /// Return a JSON list of annotation names currently loaded.
    pub fn get_annotation_names(&mut self) -> PyResult<String> {
        self.active()?.get_annotation_names()
    }

    /// Remove all annotations whose track name matches `name`.
    /// If the same name was added more than once, all copies are removed.
    pub fn remove_annotation(&mut self, name: &str) -> PyResult<()> {
        self.active()?.remove_annotation(name);
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

#[cfg(test)]
mod tests {
    use r#gen::test_helpers::{setup_block_group, setup_gen_on_disk};
    use gen_models::block_group::BlockGroup;
    use pyo3::{exceptions::PyValueError, prelude::*};
    use serde_json::Value;

    use super::{PyGraphController, current_theme};

    fn make_controller(detail: Option<&str>) -> PyResult<PyGraphController> {
        let ctx = setup_gen_on_disk();
        let graph_handle = ctx.graph();
        let db_path = graph_handle
            .conn()
            .path()
            .map(std::path::PathBuf::from)
            .expect("test DB must be file-backed");
        let (bg_id, _) = setup_block_group(graph_handle.conn());
        let graph = BlockGroup::get_graph(graph_handle.conn(), &bg_id)
            .map_err(crate::python_api::utils::block_group_err_to_pyerr)?;
        let mut ctrl = PyGraphController::new(db_path, graph);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        Ok(ctrl)
    }

    #[test]
    fn test_detail_invalid_raises_value_error() {
        pyo3::prepare_freethreaded_python();
        let result = make_controller(Some("bad"));
        Python::with_gil(|py| match result {
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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
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
}

/// Instantiate a `GraphWidget` from a controller and optional viewport.
/// Shared by `PySequenceGraph::plot`, `PyRepository::plot`, and `PySample::plot`.
pub fn build_widget(
    py: Python<'_>,
    ctrl: Py<PyGraphController>,
    rows: Option<u32>,
    cols: Option<u32>,
    colors: Option<PyObject>,
) -> PyResult<PyObject> {
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
