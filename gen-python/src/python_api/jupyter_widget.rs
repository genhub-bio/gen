use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    path::PathBuf,
};

use r#gen::{
    get_connection,
    views::{
        annotation_groups::load_annotation_group_entries,
        annotation_search::{AnnotationSearchState, match_style, selected_match_style},
        annotation_track::{
            AnnotationSpan, AnnotationTrack, annotation_span_from_graph_locus,
            graph_locus_from_annotation_span,
        },
        annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
        gen_graph_widget::{
            GenGraphNodeRenderer, GenGraphNodeSizer, center_on_node_offset, highlight_match_range,
            locus_label_bounds, viewport_pos_map,
        },
        inline_label_placement::draw_label_near_pos,
    },
};
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode, project_path};
use gen_models::{
    annotations::AnnotationError,
    block_group::BlockGroup,
    db::GraphConnection,
    locus::{BlockSlice, GraphLocus},
};
use gen_tui::{
    LineStyle, geometry::WorldPos, graph_controller::GraphController, graph_widget::GraphWidget,
    layout::VisualDetail, plotter::PathStyle, theme::current_theme,
};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyType},
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier},
    widgets::StatefulWidget,
};
use serde::Serialize;

use crate::python_api::{
    block_group::PyBlockGroup,
    graph_search::{PyAnnotation, PyGraphLocus, PyGraphPos},
    repository::PyRepository,
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
            Some(BlockSlice { node, start, end })
        })
        .collect();
    let strand = span
        .segments
        .first()
        .map(|s| s.strand)
        .unwrap_or(Strand::Unknown);
    GraphLocus { slices, strand }
}

/// A single annotation entry that may render as an inline canvas highlight,
/// a track panel row, or both.
///
/// `span.name` doubles as the inline label (empty = no label drawn).
/// `track` names a horizontal panel below the graph.
/// `style` drives the inline canvas tinting.
struct AnnotationDisplay {
    span: AnnotationSpan,
    track: Option<String>,
    style: Option<PathStyle>,
}

/// Internal graph controller for the Jupyter notebook widget.
///
/// Not intended for direct use from Python — users should call
/// `repo.plot(bg)` or `bg.plot()` which return a `GenGraphWidget`.
///
/// # Thread safety
///
/// ipykernel 6+ runs cell code in a thread-pool executor (e.g. thread 12) while
/// anywidget comm/observe callbacks fire on the asyncio ioloop (thread 1).
/// `PyGraphController` is therefore created on one thread and accessed from another,
/// so it must be `Send`.  `GraphHandle` contains `Rc<GraphConnection>` which is
/// `!Send`, so we store only the DB path and open a fresh connection per operation
/// instead of holding a live handle.
#[pyclass]
pub struct PyGraphController {
    db_path: PathBuf,
    pub(crate) block_group_id: Option<HashId>,
    controller: GraphController<GenGraph, GenGraphNodeSizer>,
    annotations: Vec<AnnotationDisplay>,
    search_state: AnnotationSearchState,
}

impl PyGraphController {
    pub fn new(db_path: PathBuf, graph: GenGraph) -> Self {
        let mut controller = GraphController::new(graph, GenGraphNodeSizer);
        controller.set_detail_level(VisualDetail::Truncated);
        controller.hide_cursor();
        Self {
            db_path,
            block_group_id: None,
            controller,
            annotations: Vec::new(),
            search_state: AnnotationSearchState::new(),
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

    fn all_node_ranges(&self) -> HashMap<HashId, Vec<(i64, i64)>> {
        self.controller
            .graph()
            .nodes()
            .filter(|n| !is_start_node(n.node_id) && !is_end_node(n.node_id))
            .map(|n| (n.node_id, vec![(n.sequence_start, n.sequence_end)]))
            .collect()
    }

    fn load_group_as_track(
        &self,
        conn: &GraphConnection,
        group: &str,
    ) -> Result<AnnotationTrack, AnnotationError> {
        let block_group_id = self
            .block_group_id
            .expect("block group id should be set for track loading");
        let current_block_group =
            BlockGroup::get_by_id(conn, &block_group_id).expect("current block group should exist");
        let entry = load_annotation_group_entries(conn, &current_block_group)
            .into_iter()
            .find(|entry| entry.name == group)
            .ok_or_else(|| AnnotationError::DatabaseError(rusqlite::Error::QueryReturnedNoRows))?;
        let ranges = self.all_node_ranges();
        let spans = load_annotations_for_group(&AnnotationGroupTrackRequest {
            conn,
            current_block_group: &current_block_group,
            entry: &entry,
            visible_ranges_by_node: &ranges,
        })?;
        Ok(AnnotationTrack::new(group.to_string(), spans))
    }

    fn navigate_to_span(&mut self, span: &AnnotationSpan) {
        let Some(locus) = graph_locus_from_annotation_span(span, self.controller.graph()) else {
            return;
        };
        let slice = &locus.slices[0];
        let node_len = slice.node.length();
        let frac_x = if node_len > 1 {
            slice.start as f64 / (node_len - 1) as f64
        } else {
            0.0
        };
        self.controller.set_detail_level(VisualDetail::Full);
        center_on_node_offset(&mut self.controller, slice.node, (frac_x, 0.5));
        self.controller.queue_snap_left();
        self.controller.hide_cursor();
    }

    fn apply_search_highlights(&mut self) {
        self.controller.clear_highlight(&match_style());
        self.controller.clear_highlight(&selected_match_style());
        let loci_with_rank: Vec<(usize, gen_models::locus::GraphLocus)> = self
            .search_state
            .results
            .iter()
            .enumerate()
            .filter_map(|(rank, &ann_idx)| {
                graph_locus_from_annotation_span(
                    &self.annotations[ann_idx].span,
                    self.controller.graph(),
                )
                .map(|locus| (rank, locus))
            })
            .collect();
        for (rank, locus) in loci_with_rank {
            let style = if rank == self.search_state.cursor {
                selected_match_style()
            } else {
                match_style()
            };
            highlight_match_range(&mut self.controller, &locus, style);
        }
    }
}

#[pymethods]
impl PyGraphController {
    /// Create a `GraphController` from a `Repository` and a `PyBlockGroup`.
    #[classmethod]
    fn from_block_group(
        _cls: &Bound<'_, PyType>,
        repo: &PyRepository,
        block_group: &PyBlockGroup,
    ) -> PyResult<Self> {
        let bg_id = block_group.id;
        let conn = repo.context.graph().conn();
        let db_path = conn
            .path()
            .map(PathBuf::from)
            .ok_or_else(|| PyRuntimeError::new_err("graph DB has no file path"))?;
        let graph = BlockGroup::get_graph(conn, &bg_id);
        let mut ctrl = Self::new(db_path, graph);
        ctrl.block_group_id = Some(bg_id);
        Ok(ctrl)
    }

    /// Set the level of node detail.
    ///
    /// Parameters
    /// ----------
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
        self.reapply_inline_highlight_ranges();
        Ok(())
    }

    pub fn truncate_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Truncated);
        self.reapply_inline_highlight_ranges();
    }

    pub fn full_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Full);
        self.reapply_inline_highlight_ranges();
    }

    pub fn minimize_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Minimal);
        self.reapply_inline_highlight_ranges();
    }

    fn render_frame(&mut self, cols: u16, rows: u16) -> PyResult<String> {
        let graph_area = Rect::new(0, 0, cols, rows);
        let total_area = Rect::new(0, 0, cols, rows);
        let mut buf = Buffer::empty(total_area);

        {
            let conn = self.open_conn()?;
            let renderer = GenGraphNodeRenderer::new(&conn);
            GraphWidget::with_renderer(renderer).render(graph_area, &mut buf, &mut self.controller);
        }

        // Draw inline annotation labels (tinting was applied at add time via highlight system).
        // Midpoints are recomputed each render because the viewport may have changed.
        let has_inline = self.annotations.iter().any(|d| d.style.is_some());
        if has_inline {
            let theme = current_theme();
            let pos_map = viewport_pos_map(&self.controller);
            let detail_level = self.controller.get_detail_level();
            for d in self.annotations.iter().filter(|d| d.style.is_some()) {
                let style = d.style.unwrap();
                let color = match style.color {
                    ratatui::style::Color::Reset => theme[0x06],
                    c => c,
                };
                // Anonymous highlights (from highlight_match) have an empty name and are skipped.
                if !d.span.name.is_empty() {
                    let locus = locus_from_span_and_pos_map(&d.span, &pos_map);
                    let Some((left_pos, right_pos)) =
                        locus_label_bounds(&locus, &pos_map, detail_level)
                    else {
                        continue;
                    };
                    let max_distance = if detail_level == gen_tui::layout::VisualDetail::Minimal {
                        10
                    } else {
                        5
                    };
                    draw_label_near_pos(
                        &mut buf,
                        graph_area,
                        (left_pos, right_pos),
                        &d.span.name,
                        color,
                        &self.controller.viewport_state,
                        max_distance,
                    );
                }
            }
        }

        // Overlay annotation tracks at the bottom of the canvas area.
        // Tracks are rebuilt at render time by grouping displays by track name.
        let mut remaining = Rect::new(0, 0, cols, rows);
        let mut track_names: Vec<String> = Vec::new();
        let mut track_spans: HashMap<String, Vec<AnnotationSpan>> = HashMap::new();
        for d in &self.annotations {
            if let Some(ref name) = d.track {
                if !track_spans.contains_key(name) {
                    track_names.push(name.clone());
                }
                track_spans
                    .entry(name.clone())
                    .or_default()
                    .push(d.span.clone());
            }
        }
        for name in track_names.iter().rev() {
            let spans = track_spans.remove(name).unwrap_or_default();
            let track = AnnotationTrack::new(name.clone(), spans);
            let height = track.draw(&mut buf, remaining, &self.controller);
            if height == 0 {
                break;
            }
            remaining.height = remaining.height.saturating_sub(height);
        }

        let frame = serialize_buffer(&buf, cols, rows);
        serde_json::to_string(&frame).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Re-register all inline annotation highlights using the current detail level.
    ///
    /// Highlight column offsets are clamped at registration time, so they go stale
    /// when the detail level changes. Call this after any zoom to re-clamp everything.
    fn reapply_inline_highlight_ranges(&mut self) {
        let styles: Vec<PathStyle> = self.annotations.iter().filter_map(|d| d.style).collect();
        for style in &styles {
            self.controller.clear_highlight(style);
        }
        // Collect loci with immutable graph borrow, then apply with mutable controller borrow.
        let loci_with_styles: Vec<(GraphLocus, PathStyle)> = self
            .annotations
            .iter()
            .filter(|d| d.style.is_some())
            .filter_map(|d| {
                graph_locus_from_annotation_span(&d.span, self.controller.graph())
                    .map(|l| (l, d.style.unwrap()))
            })
            .collect();
        for (locus, style) in &loci_with_styles {
            highlight_match_range(&mut self.controller, locus, *style);
        }
        if !self.search_state.is_empty() {
            self.apply_search_highlights();
        }
    }

    fn zoom_in(&mut self) {
        self.controller.zoom_in();
        self.reapply_inline_highlight_ranges();
    }

    fn zoom_out(&mut self) {
        self.controller.zoom_out();
        self.reapply_inline_highlight_ranges();
    }

    fn handle_click(&mut self, col: u16, row: u16) -> bool {
        self.controller.handle_click(col, row)
    }

    fn move_by(&mut self, dx: i16, dy: i16) {
        self.controller.move_by_terminal(dx, dy);
        self.controller.sync_cursor_to_closest_node();
    }

    /// Center the view on the position described by `pos`.
    ///
    /// Forces full detail level and makes the cursor visible so the user can
    /// see the exact position within the node.  The fractional x-offset is
    /// computed from `pos.offset / node.length()` so the camera lands on the
    /// exact byte; y is always centered (0.5).
    /// This is the Rust-side counterpart to `GenGraphWidget.go_to()`.
    fn go_to_pos(&mut self, pos: &PyGraphPos) {
        self.controller.set_detail_level(VisualDetail::Full);
        let node = pos.inner.block;
        let node_len = node.length();
        let frac_x = if node_len > 1 {
            pos.inner.offset as f64 / (node_len - 1) as f64
        } else {
            0.5
        };
        center_on_node_offset(&mut self.controller, node, (frac_x, 0.5));
        self.controller.hide_cursor();
    }

    /// Highlight the path of nodes covered by `match_obj` in the given colour.
    ///
    /// `color` must be a CSS hex string like `"#ffff00"` or one of the named
    /// ratatui colours (`"yellow"`, `"cyan"`, `"red"`, …).  When omitted the
    /// next unused theme accent colour (slots 0x08–0x0F) is chosen automatically.
    fn highlight_match(&mut self, locus: &PyGraphLocus, color: Option<&str>) -> PyResult<()> {
        use ratatui::style::Color;
        let c = match color {
            None => self.controller.next_accent_color(),
            Some(s) => match s {
                "red" => Color::Red,
                "green" => Color::Green,
                "yellow" => Color::Yellow,
                "blue" => Color::Blue,
                "magenta" => Color::Magenta,
                "cyan" => Color::Cyan,
                "white" => Color::White,
                hex if hex.starts_with('#') => parse_hex_color(hex)?,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown color {other:?}"
                    )));
                }
            },
        };
        let style = PathStyle::new(c)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        highlight_match_range(&mut self.controller, &locus.inner, style);
        let span = annotation_span_from_graph_locus(&locus.inner, "");
        self.annotations.push(AnnotationDisplay {
            span,
            track: None,
            style: Some(style),
        });
        Ok(())
    }

    /// Remove all highlights from the graph.
    fn clear_highlights(&mut self) {
        self.controller.clear_all_highlights();
    }

    /// Highlight the most recent path associated with this block group.
    ///
    /// Parameters
    /// ----------
    /// color : str, optional
    ///     Colour for the highlight.  Accepts named colours
    ///     (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
    ///     (``"#ff4444"``).  When omitted the next unused theme accent
    ///     colour is chosen automatically.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If no block group is associated with this widget, or if no path
    ///     exists for the block group.
    /// ValueError
    ///     If ``color`` is not a recognised colour name or CSS hex string.
    #[pyo3(signature = (color=None))]
    pub fn show_path(&mut self, color: Option<&str>) -> PyResult<()> {
        let block_group_id = self.block_group_id.ok_or_else(|| {
            PyRuntimeError::new_err(
                "show_path() requires a block group; obtain the widget via BlockGroup.plot()",
            )
        })?;

        use ratatui::style::Color;
        let highlight_color = match color {
            None => self.controller.next_accent_color(),
            Some(s) => match s {
                "red" => Color::Red,
                "green" => Color::Green,
                "yellow" => Color::Yellow,
                "blue" => Color::Blue,
                "magenta" => Color::Magenta,
                "cyan" => Color::Cyan,
                "white" => Color::White,
                hex if hex.starts_with('#') => parse_hex_color(hex)?,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown color {other:?}"
                    )));
                }
            },
        };

        let conn = self.open_conn()?;

        let path = BlockGroup::get_current_path(&conn, &block_group_id);
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

        self.controller.set_path_highlight(style, path_nodes);
        Ok(())
    }

    /// Clear path highlighting previously applied by `show_path`.
    pub fn clear_path(&mut self) {
        self.controller.clear_all_highlights();
    }

    /// Load annotations from the database by group name and add them as a
    /// horizontal track panel below the graph.
    pub fn add_track_group(&mut self, group: &str) -> PyResult<()> {
        let conn = self.open_conn()?;
        let track = self
            .load_group_as_track(&conn, group)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        for span in track.annotations {
            self.annotations.push(AnnotationDisplay {
                span,
                track: Some(group.to_string()),
                style: None,
            });
        }
        Ok(())
    }

    /// Build a track panel from a list of `Annotation` objects.
    /// Each `Annotation` becomes one span; all are grouped under `name`.
    pub fn add_track_annotations(&mut self, annotations: Vec<PyRef<PyAnnotation>>, name: &str) {
        for ann in &annotations {
            self.annotations.push(AnnotationDisplay {
                span: ann.inner.clone(),
                track: Some(name.to_string()),
                style: None,
            });
        }
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

        for span in track.annotations {
            self.annotations.push(AnnotationDisplay {
                span,
                track: Some(name.to_string()),
                style: None,
            });
        }
        Ok(())
    }

    /// Navigate to an `Annotation` object.
    pub fn go_to_annotation_obj(&mut self, annotation: &PyAnnotation) {
        self.navigate_to_span(&annotation.inner);
    }

    /// Navigate to a `GraphLocus`.
    pub fn go_to_locus(&mut self, locus: &PyGraphLocus) {
        let span = annotation_span_from_graph_locus(&locus.inner, "");
        self.navigate_to_span(&span);
    }

    /// Navigate the camera to the first annotation span whose name matches
    /// `annotation_name` across all loaded track panels.
    ///
    /// Returns the `GraphPos` the camera moved to.  Raises `KeyError` if no
    /// annotation with that name is found in any track.
    pub fn go_to_annotation_by_name(&mut self, annotation_name: &str) -> PyResult<PyGraphPos> {
        let segment = self
            .annotations
            .iter()
            .filter(|d| d.track.is_some())
            .map(|d| &d.span)
            .find(|span| span.name == annotation_name)
            .and_then(|span| span.segments.first())
            .cloned()
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "no annotation named {annotation_name:?} found in loaded tracks"
                ))
            })?;

        let node = self
            .controller
            .graph()
            .nodes()
            .find(|n| n.node_id == segment.node_id)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "annotation {annotation_name:?} references a node not present in the current graph"
                ))
            })?;

        let node_len = node.length();
        let frac_x = if node_len > 1 {
            (segment.start as f64) / (node_len - 1) as f64
        } else {
            0.0
        };
        self.controller.set_detail_level(VisualDetail::Full);
        center_on_node_offset(&mut self.controller, node, (frac_x, 0.5));
        self.controller.queue_snap_left();
        self.controller.hide_cursor();

        Ok(PyGraphPos::new(node, segment.start as usize))
    }

    /// Return a JSON list of track-panel annotation names currently loaded.
    pub fn get_track_names(&self) -> PyResult<String> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut names: Vec<&str> = Vec::new();
        for d in &self.annotations {
            if let Some(ref name) = d.track {
                if seen.insert(name.as_str()) {
                    names.push(name.as_str());
                }
            }
        }
        serde_json::to_string(&names).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove a track-panel annotation by name.
    pub fn remove_track(&mut self, name: &str) {
        self.annotations
            .retain(|d| d.track.as_deref() != Some(name));
    }

    /// Clear all track-panel annotations.
    pub fn clear_all_annotations(&mut self) {
        self.annotations.retain(|d| d.track.is_none());
    }

    /// Add a single inline annotation rendered directly on the graph canvas.
    /// The annotation is tinted with an accent colour and labelled below its span.
    pub fn add_inline_annotation(&mut self, annotations: Vec<PyRef<PyAnnotation>>, _name: &str) {
        let theme = current_theme();
        const ACCENT: [usize; 8] = [0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F];
        let inline_count = self
            .annotations
            .iter()
            .filter(|d| d.style.is_some())
            .count();
        let color = theme[ACCENT[inline_count % ACCENT.len()]];
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        // Collect loci with immutable graph borrow, then apply with mutable controller borrow.
        let loci: Vec<Option<GraphLocus>> = annotations
            .iter()
            .map(|ann| graph_locus_from_annotation_span(&ann.inner, self.controller.graph()))
            .collect();
        for locus in loci.into_iter().flatten() {
            highlight_match_range(&mut self.controller, &locus, style);
        }
        for ann in &annotations {
            self.annotations.push(AnnotationDisplay {
                span: ann.inner.clone(),
                track: None,
                style: Some(style),
            });
        }
    }

    /// Return a JSON list of inline annotation names currently loaded.
    pub fn get_inline_annotation_names(&self) -> PyResult<String> {
        let names: Vec<&str> = self
            .annotations
            .iter()
            .filter(|d| d.style.is_some())
            .map(|d| d.span.name.as_str())
            .collect();
        serde_json::to_string(&names).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove all inline annotations whose span name matches `name`.
    /// If the same name was added more than once, all copies are removed.
    pub fn remove_inline_annotation(&mut self, name: &str) {
        let mut remaining = Vec::with_capacity(self.annotations.len());
        for d in self.annotations.drain(..) {
            if d.style.is_some() && d.span.name == name {
                self.controller.clear_highlight(&d.style.unwrap());
            } else {
                remaining.push(d);
            }
        }
        self.annotations = remaining;
    }

    /// Clear all inline annotations.
    pub fn clear_all_inline_annotations(&mut self) {
        let styles: Vec<PathStyle> = self.annotations.iter().filter_map(|d| d.style).collect();
        for style in &styles {
            self.controller.clear_highlight(style);
        }
        self.annotations.retain(|d| d.style.is_none());
    }

    /// Search annotation names (track + inline) for a case-insensitive substring.
    /// Navigates to the first match and highlights all matches. Returns match count.
    pub fn search_annotations(&mut self, query: &str) -> usize {
        let count = self
            .search_state
            .search(query, self.annotations.iter().map(|d| d.span.name.as_str()));
        self.apply_search_highlights();
        if let Some(idx) = self.search_state.current_annotation_idx() {
            let span = self.annotations[idx].span.clone();
            self.navigate_to_span(&span);
        }
        count
    }

    /// Advance to the next search result. Returns false if no results are loaded.
    pub fn search_next(&mut self) -> bool {
        if !self.search_state.advance() {
            return false;
        }
        self.apply_search_highlights();
        if let Some(idx) = self.search_state.current_annotation_idx() {
            let span = self.annotations[idx].span.clone();
            self.navigate_to_span(&span);
        }
        true
    }

    /// Go to the previous search result. Returns false if no results are loaded.
    pub fn search_prev(&mut self) -> bool {
        if !self.search_state.retreat() {
            return false;
        }
        self.apply_search_highlights();
        if let Some(idx) = self.search_state.current_annotation_idx() {
            let span = self.annotations[idx].span.clone();
            self.navigate_to_span(&span);
        }
        true
    }

    /// Clear search results and remove all search highlights.
    pub fn clear_search(&mut self) {
        self.controller.clear_highlight(&match_style());
        self.controller.clear_highlight(&selected_match_style());
        self.search_state.clear();
    }
}

// --------------------------------------------------------------------------
// File loading helper
// --------------------------------------------------------------------------

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

// --------------------------------------------------------------------------

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
        let graph = BlockGroup::get_graph(graph_handle.conn(), &bg_id);
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

/// Instantiate a `GenGraphWidget` from a controller and optional viewport
/// Shared by `PyBlockGroup::plot` and `PyRepository::plot`.
pub fn build_widget(
    py: Python<'_>,
    ctrl: Py<PyGraphController>,
    rows: Option<u32>,
    cols: Option<u32>,
) -> PyResult<PyObject> {
    let gen_module = py.import("gen")?;
    let widget_cls = gen_module.getattr("GenGraphWidget")?;
    let kwargs = PyDict::new(py);
    if let Some(r) = rows {
        kwargs.set_item("rows", r)?;
    }
    if let Some(c) = cols {
        kwargs.set_item("cols", c)?;
    }
    let widget = widget_cls.call((ctrl,), Some(&kwargs))?;
    Ok(widget.into())
}
