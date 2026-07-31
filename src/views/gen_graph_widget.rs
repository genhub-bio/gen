use std::collections::{HashMap, HashSet, VecDeque};

use gen_core::{
    INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
    is_end_node, is_start_node,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode, GraphNodeSlice};
use gen_models::{db::GraphConnection, locus::GraphLocus, node::Node, sequence::SequenceError};
use gen_tui::{
    ViewportState,
    geometry::{WorldPos, WorldRect},
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::{GraphWidget, NODE_GLYPH},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer, PathStyle},
    theme::current_theme,
};
use petgraph::visit::NodeIndexable;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

use crate::views::{
    annotation_track::{
        AnnotationSpan, graph_locus_from_annotation_span, span_covered_by_later, span_label_text,
        span_should_hide_in_truncated,
    },
    graph_overlay::{AnnotationColorCache, GraphOverlay, OverlaySource},
    inline_label_placement::draw_label_near_pos,
};

/// Labels for special start/end nodes
pub mod label {
    pub const START: &str = "╟";
    pub const END: &str = "╢";
}

const ZERO_WIDTH_GLYPH: char = '─';

fn is_zero_width_junction(node: &GraphNode) -> bool {
    !is_start_node(node.node_id)
        && !is_end_node(node.node_id)
        && node.sequence_start == node.sequence_end
}

/// Domain-specific node sizer for GenGraph that calculates visual dimensions
/// based on genomic sequence length.
#[derive(Clone)]
pub struct GenGraphNodeSizer;

impl NodeSizer<GenGraph> for GenGraphNodeSizer {
    /// Calculate how much screen space a GenGraph node needs based on sequence length and level of detail
    fn get_node_size(&self, node: &GraphNode, detail_level: VisualDetail) -> (u64, u64) {
        // Handle special start/end nodes with fixed label sizes (always show full label)
        if is_start_node(node.node_id) {
            return (label::START.chars().count() as u64, 1u64);
        }
        if is_end_node(node.node_id) {
            return (label::END.chars().count() as u64, 1u64);
        }
        if is_zero_width_junction(node) {
            return (1, 1);
        }

        let sequence_length = (node.sequence_end - node.sequence_start) as u64;
        match detail_level {
            VisualDetail::Minimal => (1u64, 1u64), // Just a glyph
            VisualDetail::Truncated => (sequence_length.min(13), 1u64), // 13 = 5 border + 3 mid + 5 border
            VisualDetail::Full => (sequence_length, 1u64),              // Full sequence length
        }
    }

    fn is_selectable(&self, node: &GraphNode) -> bool {
        !is_zero_width_junction(node)
    }

    /// Map a raw sequence column to the visual cell it occupies for `detail_level`.
    ///
    /// In `Truncated` mode a node longer than 13 bases is drawn as `AAAAA...BBBBB`, so
    /// interior columns collapse onto the central `...` cell. `Minimal` mode draws a
    /// single glyph, so every column maps to column 0.
    fn map_column(&self, node: &GraphNode, raw_col: i64, detail_level: VisualDetail) -> i64 {
        match detail_level {
            VisualDetail::Minimal => 0,
            VisualDetail::Truncated if node.length() > 13 => {
                map_truncated_col(raw_col, node.length())
            }
            _ => raw_col,
        }
    }
}

/// Domain-specific node renderer for GenGraph that handles database sequence fetching
/// and genomic sequence visualization with caching.
pub struct GenGraphNodeRenderer<'a> {
    conn: &'a GraphConnection,
    cache: HashMap<GraphNode, String>,
}

impl<'a> GenGraphNodeRenderer<'a> {
    /// Create a new GenGraph node renderer with database connection
    pub fn new(conn: &'a GraphConnection) -> Self {
        Self {
            conn,
            cache: HashMap::new(),
        }
    }

    /// Get the database connection (for accessing from other code)
    pub fn connection(&self) -> &'a GraphConnection {
        self.conn
    }

    /// Fetch sequence for a GenGraph node with caching
    pub fn get_sequence(&mut self, node_key: &GraphNode) -> Result<String, SequenceError> {
        // Check cache first
        if let Some(cached) = self.cache.get(node_key) {
            return Ok(cached.clone());
        }

        // Cache miss - query database
        let (db_node_id, start, end) = (
            node_key.node_id,
            node_key.sequence_start,
            node_key.sequence_end,
        );
        let sequences = Node::get_sequences_by_node_ids(self.conn, &[db_node_id], None);
        let sequence = match sequences.get(&db_node_id) {
            Some(seq) => seq.get_sequence(start, end)?,
            None => "?".repeat((end - start).max(0) as usize),
        };

        self.cache.insert(*node_key, sequence.clone());
        Ok(sequence)
    }
}

impl NodeRenderer<GenGraph> for GenGraphNodeRenderer<'_> {
    /// Render a GenGraph node with genomic sequence data and theme styling
    /// The rendering changes based on level of detail:
    /// - Minimal: Simple glyph representation
    /// - Truncated: Sequence with inner truncation if needed
    /// - Full: Complete sequence display
    fn render_node(
        &mut self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node_id: &GraphNode,
        detail_level: VisualDetail,
    ) {
        let theme = current_theme();
        if is_zero_width_junction(node_id) {
            let edge_style = Style::default().bg(theme[0x00]).fg(theme[0x05]);
            buffer.set_char_styled(area.left_center(), ZERO_WIDTH_GLYPH, edge_style);
            return;
        }

        let background_style = Style::default().bg(theme[0x05]);
        let text_style = Style::default().bg(theme[0x05]).fg(theme[0x00]);

        buffer.fill_rect(area, ' ');
        buffer.set_char_styled(area.left_center(), ' ', background_style);

        // Handle special start/end nodes (always show full label)
        if is_start_node(node_id.node_id) {
            let edge_style = Style::default().bg(theme[0x00]).fg(theme[0x05]);
            buffer.set_string_styled(area.left_center(), label::START, edge_style);
            return;
        }
        if is_end_node(node_id.node_id) {
            let edge_style = Style::default().bg(theme[0x00]).fg(theme[0x05]);
            buffer.set_string_styled(area.left_center(), label::END, edge_style);
            return;
        }

        match detail_level {
            VisualDetail::Minimal => {
                // Base scale: Just show a simple glyph
                let text_style = Style::default().fg(theme[0x05]).bg(theme[0x00]);
                buffer.set_string_styled(area.left_center(), &NODE_GLYPH.to_string(), text_style);
            }
            VisualDetail::Truncated => {
                // 13 cells: 5 left bases + "..." + 5 right bases.
                let sequence = self
                    .get_sequence(node_id)
                    .unwrap_or_else(|_| "Unknown Sequence".to_string());
                let max_width = 13u32;
                let truncated = inner_truncation(&sequence, max_width);
                buffer.set_string_styled(area.left_center(), &truncated, text_style);
            }
            VisualDetail::Full => {
                // Full scale: Show complete sequence (truncated only by area width)
                let sequence = self
                    .get_sequence(node_id)
                    .unwrap_or_else(|_| "Unknown Sequence".to_string());
                buffer.set_string_styled(area.left_center(), &sequence, text_style);
            }
        }
    }
}

/// Truncate a genomic sequence from the inside, keeping the beginning and end.
///
/// # Arguments
/// * `s` - The sequence string to truncate
/// * `target_length` - Maximum length for the output string
///
/// # Returns
/// A string showing beginning...end if truncation needed, or original if short enough.
pub fn inner_truncation(s: &str, target_length: u32) -> String {
    if s.len() <= target_length as usize {
        return s.to_string();
    } else if target_length < 5 {
        return NODE_GLYPH.to_string(); // ⏺ is U+23FA
    }
    // length - 3 because we need space for the ellipsis
    let left_len = (target_length - 3) / 2 + ((target_length - 3) % 2);
    let right_len = (target_length - 3) / 2;

    let left = &s[..left_len as usize];
    let right = &s[(s.len() - right_len as usize)..];

    format!("{}...{}", left, right)
}

/// Convenience function to create a GenGraph widget with all domain-specific components
/// configured. This provides a simple interface for creating GenGraph visualizations.
///
/// # Arguments
/// * `conn` - Database connection for sequence fetching
///
/// # Returns
/// A configured GraphWidget ready to visualize GenGraph data
pub fn create_gen_graph_widget(
    conn: &GraphConnection,
) -> GraphWidget<'_, GenGraph, GenGraphNodeSizer, GenGraphNodeRenderer<'_>> {
    let renderer = GenGraphNodeRenderer::new(conn);
    GraphWidget::with_renderer(renderer)
}

/// Compute which edges would be removed by `BlockGroup::prune_graph`.
///
/// Mirrors the per-source-node, per-chromosome_index deduplication logic: for each
/// chromosome_index appearing on outgoing edges of a node, the edge with the highest
/// `created_on` is kept; all others are dimmed. Edges with
/// `PRESERVE_EDIT_SITE_CHROMOSOME_INDEX` are always dimmed; edges with
/// `NO_CHROMOSOME_INDEX` or `INDETERMINATE_CHROMOSOME_INDEX` are never dimmed.
fn compute_pruned_edges(graph: &GenGraph) -> HashSet<(GraphNode, GraphNode)> {
    let mut pruned: HashSet<(GraphNode, GraphNode)> = HashSet::new();

    for node in graph.nodes() {
        // chromosome_index -> (source, target, best_created_on)
        let mut edges_by_ci: HashMap<i64, (GraphNode, GraphNode, i64)> = HashMap::new();

        for (source_node, target_node, edge_weights) in graph.edges(node) {
            for edge_weight in edge_weights {
                let GraphEdge {
                    chromosome_index,
                    created_on,
                    ..
                } = *edge_weight;

                if chromosome_index == NO_CHROMOSOME_INDEX
                    || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
                {
                    continue;
                }
                if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
                    pruned.insert((source_node, target_node));
                    continue;
                }
                edges_by_ci
                    .entry(chromosome_index)
                    .and_modify(|(best_src, best_tgt, best_ts)| {
                        if created_on > *best_ts {
                            pruned.insert((*best_src, *best_tgt));
                            *best_src = source_node;
                            *best_tgt = target_node;
                            *best_ts = created_on;
                        } else {
                            pruned.insert((source_node, target_node));
                        }
                    })
                    .or_insert((source_node, target_node, created_on));
            }
        }
    }

    pruned
}

/// Find nodes that become inaccessible when all pruned edges are removed.
///
/// BFS from all start nodes following only non-pruned edges. Any node not reached
/// is only reachable through pruned (lowlighted) edges and should be dimmed.
fn compute_inaccessible_nodes(
    graph: &GenGraph,
    pruned: &HashSet<(GraphNode, GraphNode)>,
) -> Vec<GraphNode> {
    let mut reachable: HashSet<GraphNode> = HashSet::new();
    let mut queue: VecDeque<GraphNode> =
        graph.nodes().filter(|n| is_start_node(n.node_id)).collect();
    for &node in &queue {
        reachable.insert(node);
    }

    while let Some(node) = queue.pop_front() {
        for (src, tgt, _) in graph.edges(node) {
            if !pruned.contains(&(src, tgt)) && reachable.insert(tgt) {
                queue.push_back(tgt);
            }
        }
    }

    graph.nodes().filter(|n| !reachable.contains(n)).collect()
}

/// Create a configured GraphController for a GenGraph with the standard theme and settings.
///
/// This is the standard way to initialize a graph controller for GenGraph visualization.
/// It applies the application's theme colors, sets the detail level to Truncated, and
/// Starts in free-camera mode (cursor hidden until the user clicks a node or uses keyboard nav).
///
/// # Arguments
/// * `graph` - The GenGraph to visualize
///
/// # Returns
/// A configured GraphController ready for use with `create_gen_graph_widget`
pub fn create_gen_graph_controller(
    graph: GenGraph,
) -> GraphController<GenGraph, GenGraphNodeSizer> {
    let pruned = compute_pruned_edges(&graph);
    let inaccessible = compute_inaccessible_nodes(&graph, &pruned);
    let node_sizer = GenGraphNodeSizer;
    let mut controller = GraphController::new(graph, node_sizer);
    for edge in pruned {
        controller.dim_edge(edge);
    }
    for node in inaccessible {
        controller.dim_node(node);
    }
    controller.set_detail_level(VisualDetail::Truncated);
    controller.hide_cursor();
    controller
}

/// Navigate to an exact byte offset within a node, snapping the camera left.
///
/// Build a GraphNode → (WorldPos, node_size) lookup from the current viewport graph.
pub fn viewport_pos_map<S: NodeSizer<GenGraph>>(
    controller: &GraphController<GenGraph, S>,
) -> HashMap<GraphNode, (WorldPos, (u64, u64))> {
    let graph = controller.graph();
    controller
        .get_viewport_graph()
        .data_nodes()
        .map(|(world_pos, domain_idx, layout_node)| {
            let node = <&GenGraph as NodeIndexable>::from_index(&graph, domain_idx.index());
            (node, (world_pos, layout_node.size))
        })
        .collect()
}

/// Compute the world-space bounding corners of the matched region in a `GraphLocus`.
///
/// Returns `(left_pos, right_pos)` where:
/// - `left_pos` is the world position of the first matched column in `blocks[0]`,
///   with the minimum y across all blocks
/// - `right_pos` is the world position of the last matched column in `blocks[last]`,
///   with the maximum y across all blocks
///
/// Column offsets are mapped with `NodeSizer::map_column` so they land correctly in
/// every detail level (e.g. truncated nodes collapse interior columns to the `...` cell).
/// The x bounds are clipped to the visible camera rect so viewport-spanning annotations
/// place their label near the visible portion rather than off-screen.
///
/// Returns `None` if no block in the locus is present in `pos_map` (all off-screen),
/// or if the annotation's x span is entirely outside the camera rect.
pub fn locus_label_bounds(
    locus: &GraphLocus,
    pos_map: &HashMap<GraphNode, (WorldPos, (u64, u64))>,
    detail_level: VisualDetail,
    viewport_state: &ViewportState,
) -> Option<(WorldPos, WorldPos)> {
    let block_world_pos = |block: GraphNode, col_raw: i64| -> Option<WorldPos> {
        let &(center, size) = pos_map.get(&block)?;
        let rect = WorldRect::from_center_and_size(center, size);
        let col = GenGraphNodeSizer.map_column(&block, col_raw, detail_level);
        Some(WorldPos::new(rect.min.x + col, center.y))
    };

    if locus.slices.is_empty() {
        return None;
    }
    let last = locus.slices.len() - 1;

    let left_pos = locus.slices.iter().enumerate().find_map(|(i, s)| {
        let col_raw = if i == 0 { s.start as i64 } else { 0 };
        block_world_pos(s.block, col_raw)
    })?;

    let right_pos = locus.slices.iter().enumerate().rev().find_map(|(i, s)| {
        let col_raw = if i == last {
            s.end.saturating_sub(1) as i64
        } else {
            s.block.length() - 1
        };
        block_world_pos(s.block, col_raw)
    })?;

    let mut y_min = i64::MAX;
    let mut y_max = i64::MIN;
    for s in &locus.slices {
        let Some(&(center, _)) = pos_map.get(&s.block) else {
            continue;
        };
        y_min = y_min.min(center.y);
        y_max = y_max.max(center.y);
    }

    if y_min == i64::MAX {
        return None;
    }

    let cam = viewport_state.camera_rect();
    let clipped_left_x = left_pos.x.max(cam.min.x);
    let clipped_right_x = right_pos.x.min(cam.max.x);
    if clipped_right_x < clipped_left_x {
        return None;
    }

    let left_pos = WorldPos::new(clipped_left_x, y_max);
    let right_pos = WorldPos::new(clipped_right_x, y_min);

    Some((left_pos, right_pos))
}

/// Map a sequence offset to a visual cell column inside a 13-cell truncated node.
///
/// Display layout: `AAAAA...BBBBB` — first 5 bases (cells 0-4), `...` (cells 5-7),
/// last 5 bases (cells 8-12). Interior positions that fall in the `...` region map
/// to cell 6 (the centre dot).
fn map_truncated_col(value: i64, block_seq_len: i64) -> i64 {
    if value < 5 {
        value
    } else if block_seq_len - value <= 5 {
        13 - (block_seq_len - value)
    } else {
        6
    }
}

/// Highlight only the matched bytes of a `GraphLocus`, using sub-rect tinting.
///
/// - Start node: tinted from `start_offset` to its right edge.
/// - Middle nodes: fully tinted.
/// - End node: tinted from its left edge to `end_offset` (exclusive).
///
/// The stored columns are raw sequence offsets; the controller maps them to visual
/// cells for the current detail level on every rebuild via `NodeSizer::map_column`,
/// so the highlight self-heals across detail and zoom changes.
pub fn highlight_locus<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    m: &GraphLocus,
    style: PathStyle,
) {
    for s in &m.slices {
        let col_start = s.start as i64;
        let col_end = s.end.saturating_sub(1) as i64;
        controller.set_cell_highlight(s.block, (col_start, 0), (col_end, 0), style);
    }

    for (s, t) in m.slices.iter().zip(m.slices.iter().skip(1)) {
        controller.set_edge_highlight((s.block, t.block), style);
    }
}

/// A mapped cell rectangle: the node it's on, and its top-left/bottom-right columns.
type CellRegion = (GraphNode, (i64, i64), (i64, i64));

/// The mapped column range a `GraphNodeSlice` occupies once rendered, using the exact
/// same `NodeSizer::map_column` math `highlight_locus` uses to paint it. Computing
/// conflicts against this (rather than against raw, unmapped sequence coordinates) is
/// what catches collisions that only exist after mapping — e.g. two annotations that
/// don't overlap at `Full` detail can still both collapse onto the same cell once a node
/// is small enough to be `Truncated`.
fn slice_region<S: NodeSizer<GenGraph>>(
    node_sizer: &S,
    detail_level: VisualDetail,
    slice: &GraphNodeSlice,
) -> CellRegion {
    let col_start = slice.start as i64;
    let col_end = slice.end.saturating_sub(1) as i64;
    let tl = (
        node_sizer.map_column(&slice.block, col_start, detail_level),
        0,
    );
    let br = (
        node_sizer.map_column(&slice.block, col_end, detail_level),
        0,
    );
    (slice.block, tl, br)
}

/// Whether two mapped cell regions occupy any of the same cells.
fn regions_overlap(a: &CellRegion, b: &CellRegion) -> bool {
    a.0 == b.0 && a.1.0 <= b.2.0 && b.1.0 <= a.2.0
}

/// The 8 theme accent slots annotation colors and `next_accent_color` are drawn from.
fn accent_colors() -> [Color; 8] {
    let theme = current_theme();
    [
        theme[0x08],
        theme[0x09],
        theme[0x0A],
        theme[0x0B],
        theme[0x0C],
        theme[0x0D],
        theme[0x0E],
        theme[0x0F],
    ]
}

/// Re-register every overlay highlight on `controller`, replacing whatever highlights
/// were previously set.
///
/// Span overlays are processed longest-first so shorter (inner) spans paint on top; any
/// path overlay is applied last so the route paints over the span tints. Each span's color
/// is chosen greedily: its color from a previous pass (`color_cache`) if it's still
/// conflict-free, or else the next color in rotation, so spans that never conflict with
/// anything still get spread across distinct colors instead of colors reshuffling across
/// frames or collapsing onto one repeated color. Only hunts for a different free accent
/// slot when the preferred color is actually taken by a previously-processed span whose
/// mapped cell range overlaps this one; only gives up and accepts a collision if every
/// slot is already taken by a genuine neighbor — with only 8 slots, a dense pile of
/// mutually-overlapping annotations can still collide, but this makes collisions the
/// exception rather than the default.
///
/// `overlays` is written back with the colors actually used, so `draw_annotation_labels`
/// (which reads `overlay.style` separately, after this runs) labels each span in the same
/// color that got painted. Callers run this after any change that invalidates mapped
/// highlight columns (zoom, detail change) or, in the live TUI viewers, every frame
/// because the overlay set changes with scrolling.
pub fn reapply_overlays<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    overlays: &mut [GraphOverlay],
    color_cache: &mut AnnotationColorCache,
) {
    let detail_level = controller.get_detail_level();
    let node_sizer = &controller.partition_controller.node_sizer;
    let graph = controller.graph();

    // DB-loaded tracks are too busy to paint at minimal detail; a span confined to a
    // partial slice of a single node is also dropped at truncated detail, mirroring the
    // label suppression below.
    let mut span_indices: Vec<usize> = overlays
        .iter()
        .enumerate()
        .filter_map(|(idx, overlay)| {
            overlay
                .span()
                .filter(|_| {
                    !matches!(
                        (detail_level, &overlay.source),
                        (VisualDetail::Minimal, OverlaySource::Track(_))
                    )
                })
                .filter(|span| {
                    detail_level != VisualDetail::Truncated
                        || !span_should_hide_in_truncated(span, graph)
                })
                .map(|_| idx)
        })
        .collect();
    span_indices.sort_by_key(|&idx| {
        let span = overlays[idx]
            .span()
            .expect("filtered to span overlays above");
        -(span
            .segments
            .iter()
            .map(|segment| segment.end - segment.start)
            .sum::<i64>())
    });

    // Decide every span's locus and color first (only needs `&controller`); painting
    // (`&mut controller`) happens in a second pass once every color is settled.
    let accents = accent_colors();
    let mut occupied: Vec<(CellRegion, Color)> = Vec::new();
    let mut decisions: Vec<(usize, GraphLocus, Color)> = Vec::new();
    for idx in span_indices {
        let span = overlays[idx]
            .span()
            .expect("filtered to span overlays above");
        let Some(locus) = graph_locus_from_annotation_span(span, graph) else {
            continue;
        };
        let regions: Vec<CellRegion> = locus
            .slices
            .iter()
            .map(|slice| slice_region(node_sizer, detail_level, slice))
            .collect();
        let used: Vec<Color> = occupied
            .iter()
            .filter(|(placed, _)| regions.iter().any(|region| regions_overlap(placed, region)))
            .map(|(_, color)| *color)
            .collect();

        // Prefer this span's previous color, or, the first time it's seen, the next color
        // in rotation, so spans that never conflict with anything still get spread across
        // distinct colors instead of repeatedly landing on the same one. Only hunt for a
        // different free accent slot when the preferred color is actually taken by
        // something this span overlaps; only give up and accept a collision if every slot
        // is taken.
        let preferred = color_cache
            .get(&span.id)
            .unwrap_or_else(|| color_cache.next_color(&accents));
        let color = if used.contains(&preferred) {
            accents
                .into_iter()
                .find(|c| !used.contains(c))
                .unwrap_or(preferred)
        } else {
            preferred
        };
        color_cache.set(span.id, color);

        for region in regions {
            occupied.push((region, color));
        }
        decisions.push((idx, locus, color));
    }

    controller.clear_all_highlights();
    for (idx, locus, color) in &decisions {
        overlays[*idx].style.color = *color;
        highlight_locus(controller, locus, overlays[*idx].style);
    }
    for overlay in overlays.iter() {
        if let Some(nodes) = overlay.path_nodes() {
            controller.set_path_highlight(overlay.style, nodes.to_vec());
        }
    }
}

/// Draw floating labels for `overlays` after the graph has been rendered into `buf`.
///
/// Overlays are labelled longest-first so the covered-by-later check matches highlight
/// paint order. A label is suppressed when its span is fully covered by a shorter overlay
/// on top, when it collapses into a truncated node, or when no free cell is found near its
/// span. Returns `true` if any labelled overlay was suppressed, so the caller can show a
/// single "some annotations hidden" hint.
pub fn draw_annotation_labels<S: NodeSizer<GenGraph>>(
    buf: &mut Buffer,
    area: Rect,
    controller: &GraphController<GenGraph, S>,
    overlays: &[GraphOverlay],
) -> bool {
    let detail_level = controller.get_detail_level();
    let mut labeled: Vec<(&AnnotationSpan, PathStyle)> = overlays
        .iter()
        .filter_map(|overlay| {
            overlay
                .span()
                .filter(|span| !span.name.is_empty())
                .filter(|_| {
                    !matches!(
                        (detail_level, &overlay.source),
                        (VisualDetail::Minimal, OverlaySource::Track(_))
                    )
                })
                .map(|span| (span, overlay.style))
        })
        .collect();
    if labeled.is_empty() {
        return false;
    }
    labeled.sort_by_key(|(span, _)| {
        -(span
            .segments
            .iter()
            .map(|segment| segment.end - segment.start)
            .sum::<i64>())
    });

    let span_refs: Vec<&AnnotationSpan> = labeled.iter().map(|(span, _)| *span).collect();
    let pos_map = viewport_pos_map(controller);
    let theme = current_theme();
    let max_distance = if detail_level == VisualDetail::Minimal {
        10
    } else {
        5
    };

    let mut any_hidden = false;
    for (idx, (span, style)) in labeled.iter().enumerate() {
        let Some(locus) = graph_locus_from_annotation_span(span, controller.graph()) else {
            continue;
        };
        if span_covered_by_later(span, idx, &span_refs) {
            any_hidden = true;
            continue;
        }
        if detail_level == VisualDetail::Truncated
            && span_should_hide_in_truncated(span, controller.graph())
        {
            any_hidden = true;
            continue;
        }
        let Some(bounds) =
            locus_label_bounds(&locus, &pos_map, detail_level, &controller.viewport_state)
        else {
            continue;
        };
        let color = match style.color {
            Color::Reset => theme[0x06],
            other => other,
        };
        let label = span_label_text(span);
        if draw_label_near_pos(
            buf,
            area,
            bounds,
            &label,
            color,
            &controller.viewport_state,
            max_distance,
        )
        .is_none()
        {
            any_hidden = true;
        }
    }
    any_hidden
}

/// The slice and local offset (0-based, within that slice's block) at the
/// midpoint of a locus's total sequence length. `None` for an empty locus.
pub fn locus_midpoint(locus: &GraphLocus) -> Option<(GraphNodeSlice, usize)> {
    let total: usize = locus
        .slices
        .iter()
        .map(|slice| slice.end - slice.start)
        .sum();
    if total == 0 {
        return None;
    }
    let half = total / 2;
    let mut consumed = 0;
    for (index, slice) in locus.slices.iter().enumerate() {
        let len = slice.end - slice.start;
        let is_last = index == locus.slices.len() - 1;
        if consumed + len > half || is_last {
            return Some((*slice, slice.start + (half - consumed)));
        }
        consumed += len;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_core::{HashId, PATH_START_NODE_ID, Strand};
    use gen_models::sample::Sample;
    use gen_tui::{
        geometry::WorldPos,
        graph_controller::HighlightKind,
        testing::create_test_terminal,
        viewport_state::{ViewportState, WorldBuffer},
    };
    use ratatui::{
        backend::TestBackend, buffer::Buffer, layout::Rect, widgets::StatefulWidget as _,
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::setup_gen_on_disk,
        updates::vcf::update_with_vcf,
        views::{annotation_track::AnnotationSegment, graph_overlay::OverlayContent},
    };

    /// Test coordinate handling for very large genomic sequences
    ///
    /// Genomic sequences can span hundreds of thousands of base pairs, creating
    /// world coordinates that exceed u16::MAX (65,535) when rendered. This test
    /// verifies that the coordinate conversion system handles such large values
    /// correctly without integer overflow or wraparound artifacts.
    #[test]
    fn test_coordinate_overflow_with_large_genomic_sequences() {
        // Set up a viewport for rendering genomic data
        let mut viewport_state = ViewportState::new();
        viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 80, 20);

        // Position camera to simulate viewing a region of a large genome
        // where sequence coordinates naturally reach high values
        let camera_center = WorldPos::new(40000, 0);
        viewport_state.camera_current = camera_center;
        viewport_state.camera_target = camera_center;

        let backend = TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let mut buffer = terminal.current_buffer_mut().clone();
        let world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);

        // Test a coordinate representing the end of a 100K base pair genomic sequence
        // Such large sequences are common in genomics (genes, regulatory regions, etc.)
        let large_genomic_pos = WorldPos::new(camera_center.x + 70000, 0); // ~110K coordinate

        // The coordinate conversion should handle large values gracefully:
        // - Return None if outside viewport (correct behavior)
        // - Never wrap around due to u16 overflow (incorrect behavior)
        let result = world_buffer.world_to_viewport(large_genomic_pos);

        assert!(
            result.is_none(),
            "Large genomic coordinates outside viewport should return None, not wrap around"
        );

        // Verify that normal-sized coordinates still work correctly
        let normal_pos = WorldPos::new(camera_center.x, camera_center.y);
        assert!(
            world_buffer.world_to_viewport(normal_pos).is_some(),
            "Coordinates within normal range should convert successfully"
        );
    }

    #[test]
    fn test_inner_truncation_no_truncation_needed() {
        let s = "hello";
        let truncated = inner_truncation(s, 10);
        assert_eq!(truncated, "hello");
    }

    #[test]
    fn test_inner_truncation_truncate_to_odd_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 7);
        assert_eq!(truncated, "he...ld");
    }

    #[test]
    fn test_inner_truncation_truncate_to_even_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 8);
        assert_eq!(truncated, "hel...ld");
    }

    #[test]
    fn test_inner_truncation_empty_string() {
        let s = "";
        let truncated = inner_truncation(s, 5);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_inner_truncation_short_target() {
        let s = "hello world";
        let truncated = inner_truncation(s, 3);
        assert_eq!(truncated, NODE_GLYPH.to_string());
    }

    #[test]
    fn test_zero_width_junction_is_line_sized_and_not_selectable() {
        let junction = GraphNode {
            node_id: HashId::convert_str("zero-width-junction"),
            sequence_start: 3,
            sequence_end: 3,
        };
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };

        for detail_level in [
            VisualDetail::Minimal,
            VisualDetail::Truncated,
            VisualDetail::Full,
        ] {
            assert_eq!(
                GenGraphNodeSizer.get_node_size(&junction, detail_level),
                (1, 1)
            );
        }
        assert!(!GenGraphNodeSizer.is_selectable(&junction));
        assert!(GenGraphNodeSizer.is_selectable(&start));
    }

    #[test]
    fn test_zero_width_junction_renders_as_line() {
        let context = setup_gen_on_disk();
        let junction = GraphNode {
            node_id: HashId::convert_str("zero-width-junction"),
            sequence_start: 3,
            sequence_end: 3,
        };
        let mut viewport_state = ViewportState::new();
        viewport_state.viewport_bounds = Rect::new(0, 0, 5, 5);
        let mut buffer = Buffer::empty(viewport_state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);
        let mut renderer = GenGraphNodeRenderer::new(context.graph().conn());

        renderer.render_node(
            &mut world_buffer,
            WorldRect::from_center_and_size(WorldPos::ZERO, (1, 1)),
            &junction,
            VisualDetail::Full,
        );

        assert_eq!(
            world_buffer.get_char(WorldPos::ZERO),
            Some(ZERO_WIDTH_GLYPH)
        );
    }

    #[test]
    fn test_snapshot_zygosity_pruned_edges() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple.fa")
            .to_str()
            .unwrap()
            .to_string();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_zygosity.vcf")
            .to_str()
            .unwrap()
            .to_string();

        import_fasta(
            &context,
            &fasta_path,
            collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        update_with_vcf(
            &context,
            &vcf_path,
            collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        let gen_graph = Sample::get_graph(conn, collection, "SAMPLE1", None).unwrap();
        let mut controller = create_gen_graph_controller(gen_graph);

        let mut terminal = create_test_terminal(120, 30);
        terminal
            .draw(|f| {
                let area = f.area();
                controller.viewport_state.viewport_bounds = area;
                controller.ensure_camera_coverage().unwrap();
                controller.rebuild_viewport_graph().unwrap();
                GraphWidget::with_renderer(GenGraphNodeRenderer::new(conn)).render(
                    area,
                    f.buffer_mut(),
                    &mut controller,
                );
            })
            .unwrap();

        insta::assert_snapshot!("zygosity_pruned_edges", terminal.backend().to_string());
    }

    /// A cell highlight set in `Full` detail must re-map onto the truncated cells
    /// when the detail level changes, without any external `reapply_overlays` call.
    ///
    /// The highlight stores raw content columns; the controller maps them through
    /// `GenGraphNodeSizer::map_column` on every rebuild. For a 34-base node drawn
    /// as `AAAAA...BBBBB` (13 cells), raw column 2 stays at cell 2 while raw column
    /// 31 (three from the end) lands at cell 10.
    #[test]
    fn test_cell_highlight_remaps_on_detail_change() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple.fa")
            .to_str()
            .unwrap()
            .to_string();
        import_fasta(
            &context,
            &fasta_path,
            collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let gen_graph = Sample::get_graph(conn, collection, Sample::DEFAULT_NAME, None).unwrap();
        let mut controller = create_gen_graph_controller(gen_graph);

        let block = controller
            .graph()
            .nodes()
            .find(|node| !is_start_node(node.node_id) && !is_end_node(node.node_id))
            .expect("should have a data node");
        assert_eq!(block.length(), 34);

        // Highlight raw columns 2..=31 while showing the full sequence.
        controller.set_detail_level(VisualDetail::Full);
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice {
                block,
                start: 2,
                end: 32,
                strand: Strand::Forward,
            }],
        };
        highlight_locus(&mut controller, &locus, PathStyle::new(Color::Yellow));

        // Switch to the truncated view; the stored raw columns must re-map.
        controller.set_detail_level(VisualDetail::Truncated);
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 120, 30);
        controller.ensure_camera_coverage().unwrap();
        controller.rebuild_viewport_graph().unwrap();

        let highlights = controller.get_cell_highlights();
        assert_eq!(highlights.len(), 1, "expected exactly one cell highlight");
        let (_, top_left, bottom_right, _) = highlights[0];
        assert_eq!(top_left.0, 2, "raw column 2 stays at cell 2");
        assert_eq!(
            bottom_right.0, 10,
            "raw column 31 maps to cell 10 in the truncated node"
        );
    }

    /// An annotation spanning multiple blocks must get a highlight attempt on each
    /// covered block, and overlay application order must be by total annotation
    /// length, not by the length of any individual per-block segment.
    #[test]
    fn test_reapply_overlays_covers_each_block_and_orders_by_total_length() {
        let node_a = GraphNode {
            node_id: HashId::convert_str("block-a"),
            sequence_start: 0,
            sequence_end: 10,
        };
        let node_b = GraphNode {
            node_id: HashId::convert_str("block-b"),
            sequence_start: 0,
            sequence_end: 10,
        };
        let node_c = GraphNode {
            node_id: HashId::convert_str("block-c"),
            sequence_start: 0,
            sequence_end: 10,
        };

        let mut graph = GenGraph::new();
        graph.add_node(node_a);
        graph.add_node(node_b);
        graph.add_node(node_c);
        let edge = vec![GraphEdge {
            edge_id: HashId::convert_str("edge"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 0,
            phased: 0,
            created_on: 0,
        }];
        graph.add_edge(node_a, node_b, edge.clone());
        graph.add_edge(node_b, node_c, edge);

        let mut controller = create_gen_graph_controller(graph);

        // Two 6-base segments, one per block: total length 12.
        let multi_block_span = AnnotationSpan {
            id: HashId::convert_str("multi"),
            name: "multi".to_string(),
            segments: vec![
                AnnotationSegment {
                    node_id: node_a.node_id,
                    start: 0,
                    end: 6,
                    strand: Strand::Forward,
                },
                AnnotationSegment {
                    node_id: node_b.node_id,
                    start: 0,
                    end: 6,
                    strand: Strand::Forward,
                },
            ],
        };
        // One 10-base segment: shorter total length than the multi-block span above,
        // but its single segment is individually longer than either of that span's
        // per-block segments.
        let single_block_span = AnnotationSpan {
            id: HashId::convert_str("single"),
            name: "single".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node_c.node_id,
                start: 0,
                end: 10,
                strand: Strand::Forward,
            }],
        };

        let mut overlays = vec![
            GraphOverlay {
                content: OverlayContent::Span(single_block_span),
                source: OverlaySource::Track("single".to_string()),
                style: PathStyle::new(Color::Cyan),
            },
            GraphOverlay {
                content: OverlayContent::Span(multi_block_span),
                source: OverlaySource::Track("multi".to_string()),
                style: PathStyle::new(Color::Yellow),
            },
        ];

        let mut color_cache = AnnotationColorCache::new();
        reapply_overlays(&mut controller, &mut overlays, &mut color_cache);

        let cell_highlights: Vec<GraphNode> = controller
            .highlights
            .iter()
            .filter_map(|(kind, _)| match kind {
                HighlightKind::Cells { node, .. } => Some(*node),
                _ => None,
            })
            .collect();

        assert_eq!(
            cell_highlights.len(),
            3,
            "expected one highlight attempt per block covered by the two spans"
        );
        assert_eq!(
            cell_highlights[0], node_a,
            "multi-block span (total length 12) applied before the single-block span (length 10)"
        );
        assert_eq!(
            cell_highlights[1], node_b,
            "second block covered by the multi-block span is also highlighted"
        );
        assert_eq!(
            cell_highlights[2], node_c,
            "single-block span applied last despite its lone segment being longer than \
             either per-block segment of the multi-block span"
        );
    }

    /// At minimal detail, DB-loaded track overlays are hidden but ad-hoc annotation
    /// overlays still paint.
    #[test]
    fn test_reapply_overlays_hides_track_overlays_at_minimal_detail() {
        let node = GraphNode {
            node_id: HashId::convert_str("block"),
            sequence_start: 0,
            sequence_end: 10,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let mut controller = create_gen_graph_controller(graph);
        controller.set_detail_level(VisualDetail::Minimal);

        // Both spans cover the node's full width, so the truncated-level rule (hide only
        // partial-node spans) never suppresses them once zoomed past minimal detail.
        let track_span = AnnotationSpan {
            id: HashId::convert_str("track-span"),
            name: "track".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 0,
                end: 10,
                strand: Strand::Forward,
            }],
        };
        let adhoc_span = AnnotationSpan {
            id: HashId::convert_str("adhoc-span"),
            name: "adhoc".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 0,
                end: 10,
                strand: Strand::Forward,
            }],
        };

        let mut overlays = vec![
            GraphOverlay {
                content: OverlayContent::Span(track_span),
                source: OverlaySource::Track("file:1".to_string()),
                style: PathStyle::new(Color::Cyan),
            },
            GraphOverlay {
                content: OverlayContent::Span(adhoc_span),
                source: OverlaySource::Adhoc,
                style: PathStyle::new(Color::Yellow),
            },
        ];

        let mut color_cache = AnnotationColorCache::new();
        reapply_overlays(&mut controller, &mut overlays, &mut color_cache);

        let cell_highlights: Vec<GraphNode> = controller
            .highlights
            .iter()
            .filter_map(|(kind, _)| match kind {
                HighlightKind::Cells { node, .. } => Some(*node),
                _ => None,
            })
            .collect();

        assert_eq!(
            cell_highlights.len(),
            1,
            "only the ad-hoc overlay should paint at minimal detail"
        );

        controller.set_detail_level(VisualDetail::Truncated);
        reapply_overlays(&mut controller, &mut overlays, &mut color_cache);
        let cell_highlights_zoomed_in = controller
            .highlights
            .iter()
            .filter(|(kind, _)| matches!(kind, HighlightKind::Cells { .. }))
            .count();
        assert_eq!(
            cell_highlights_zoomed_in, 2,
            "both overlays should paint once zoomed in past minimal detail"
        );
    }

    /// Two annotations whose ranges overlap on the same node (e.g. a GenBank `source`
    /// feature and a shorter `lacZalpha` feature nested inside it) must not be painted
    /// with the same accent color. A third, unrelated span elsewhere doesn't conflict with
    /// either, so nothing forces it away from whatever color the rotation gives it next —
    /// collision-avoidance should only kick in for annotations that actually touch.
    #[test]
    fn test_reapply_overlays_avoids_color_collisions_between_overlapping_spans() {
        let shared_node = GraphNode {
            node_id: HashId::convert_str("shared-block"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let other_node = GraphNode {
            node_id: HashId::convert_str("unrelated-block"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let mut graph = GenGraph::new();
        graph.add_node(shared_node);
        graph.add_node(other_node);
        let mut controller = create_gen_graph_controller(graph);

        let long_span = AnnotationSpan {
            id: HashId::convert_str("source"),
            name: "source".to_string(),
            segments: vec![AnnotationSegment {
                node_id: shared_node.node_id,
                start: 0,
                end: 20,
                strand: Strand::Forward,
            }],
        };
        let nested_span = AnnotationSpan {
            id: HashId::convert_str("lacZalpha"),
            name: "lacZalpha".to_string(),
            segments: vec![AnnotationSegment {
                node_id: shared_node.node_id,
                start: 5,
                end: 10,
                strand: Strand::Reverse,
            }],
        };
        let unrelated_span = AnnotationSpan {
            id: HashId::convert_str("unrelated"),
            name: "unrelated".to_string(),
            segments: vec![AnnotationSegment {
                node_id: other_node.node_id,
                start: 0,
                end: 20,
                strand: Strand::Forward,
            }],
        };

        let mut overlays = vec![
            GraphOverlay {
                content: OverlayContent::Span(long_span),
                source: OverlaySource::Track("t".to_string()),
                style: PathStyle::new(Color::Reset),
            },
            GraphOverlay {
                content: OverlayContent::Span(nested_span),
                source: OverlaySource::Track("t".to_string()),
                style: PathStyle::new(Color::Reset),
            },
            GraphOverlay {
                content: OverlayContent::Span(unrelated_span),
                source: OverlaySource::Track("t".to_string()),
                style: PathStyle::new(Color::Reset),
            },
        ];

        let mut color_cache = AnnotationColorCache::new();
        reapply_overlays(&mut controller, &mut overlays, &mut color_cache);

        assert_ne!(
            overlays[0].style.color, overlays[1].style.color,
            "source and the nested lacZalpha-like span overlap and must get different colors"
        );
        assert_ne!(
            overlays[0].style.color, overlays[2].style.color,
            "the unrelated span gets its own turn in the color rotation"
        );
        assert_ne!(
            overlays[1].style.color, overlays[2].style.color,
            "the unrelated span gets its own turn in the color rotation"
        );
    }

    /// Never-before-seen annotations that don't conflict with each other should still get
    /// spread across different colors — cycling through the accent slots in turn, rather
    /// than deriving a color from each annotation's id, so two arbitrary unrelated
    /// annotations don't coincidentally land on the same one.
    #[test]
    fn test_reapply_overlays_cycles_colors_for_new_non_conflicting_spans() {
        let mut graph = GenGraph::new();
        let nodes: Vec<GraphNode> = (0..3)
            .map(|i| {
                let node = GraphNode {
                    node_id: HashId::convert_str(&format!("block-{i}")),
                    sequence_start: 0,
                    sequence_end: 10,
                };
                graph.add_node(node);
                node
            })
            .collect();
        let mut controller = create_gen_graph_controller(graph);

        let mut overlays: Vec<GraphOverlay> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| GraphOverlay {
                content: OverlayContent::Span(AnnotationSpan {
                    id: HashId::convert_str(&format!("span-{i}")),
                    name: format!("span-{i}"),
                    segments: vec![AnnotationSegment {
                        node_id: node.node_id,
                        start: 0,
                        end: 10,
                        strand: Strand::Forward,
                    }],
                }),
                source: OverlaySource::Track("t".to_string()),
                style: PathStyle::new(Color::Reset),
            })
            .collect();

        let mut color_cache = AnnotationColorCache::new();
        reapply_overlays(&mut controller, &mut overlays, &mut color_cache);

        let colors: HashSet<Color> = overlays.iter().map(|o| o.style.color).collect();
        assert_eq!(
            colors.len(),
            overlays.len(),
            "three unrelated, non-conflicting spans should each get a distinct color \
             from the rotation, not repeat one by chance"
        );
    }

    /// The color chosen for an annotation must stay the same across repeated
    /// `reapply_overlays` calls (the live TUI viewers call this every frame), even when
    /// the overlays happen to be given in a different order, so annotations don't flicker
    /// between colors as the user scrolls.
    #[test]
    fn test_reapply_overlays_keeps_colors_stable_across_repeated_calls() {
        let node = GraphNode {
            node_id: HashId::convert_str("block"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let mut controller = create_gen_graph_controller(graph);

        let make_overlays = |reversed: bool| {
            let span_a = AnnotationSpan {
                id: HashId::convert_str("a"),
                name: "a".to_string(),
                segments: vec![AnnotationSegment {
                    node_id: node.node_id,
                    start: 0,
                    end: 5,
                    strand: Strand::Forward,
                }],
            };
            let span_b = AnnotationSpan {
                id: HashId::convert_str("b"),
                name: "b".to_string(),
                segments: vec![AnnotationSegment {
                    node_id: node.node_id,
                    start: 10,
                    end: 15,
                    strand: Strand::Forward,
                }],
            };
            let mut overlays = vec![
                GraphOverlay {
                    content: OverlayContent::Span(span_a),
                    source: OverlaySource::Track("t".to_string()),
                    style: PathStyle::new(Color::Reset),
                },
                GraphOverlay {
                    content: OverlayContent::Span(span_b),
                    source: OverlaySource::Track("t".to_string()),
                    style: PathStyle::new(Color::Reset),
                },
            ];
            if reversed {
                overlays.reverse();
            }
            overlays
        };

        let mut color_cache = AnnotationColorCache::new();
        let mut first = make_overlays(false);
        reapply_overlays(&mut controller, &mut first, &mut color_cache);
        let color_a = first
            .iter()
            .find(|o| o.span().unwrap().name == "a")
            .unwrap()
            .style
            .color;
        let color_b = first
            .iter()
            .find(|o| o.span().unwrap().name == "b")
            .unwrap()
            .style
            .color;

        // Reload with the two spans in the opposite order, as a fresh track reload might.
        let mut second = make_overlays(true);
        reapply_overlays(&mut controller, &mut second, &mut color_cache);
        let color_a_again = second
            .iter()
            .find(|o| o.span().unwrap().name == "a")
            .unwrap()
            .style
            .color;
        let color_b_again = second
            .iter()
            .find(|o| o.span().unwrap().name == "b")
            .unwrap()
            .style
            .color;

        assert_eq!(
            color_a, color_a_again,
            "span a's color should survive a reload"
        );
        assert_eq!(
            color_b, color_b_again,
            "span b's color should survive a reload"
        );
    }

    fn midpoint_node(id: u8, start: i64, end: i64) -> GraphNode {
        GraphNode {
            node_id: HashId([id; 16]),
            sequence_start: start,
            sequence_end: end,
        }
    }

    fn midpoint_slice(block: GraphNode, start: usize, end: usize) -> GraphNodeSlice {
        GraphNodeSlice {
            block,
            start,
            end,
            strand: Strand::Forward,
        }
    }

    #[test]
    fn test_midpoint_single_slice_is_centered_within_it() {
        let block = midpoint_node(1, 0, 10);
        let locus = GraphLocus {
            slices: vec![midpoint_slice(block, 2, 8)],
        };
        // length 6, half = 3, so local offset = start(2) + 3 = 5
        assert_eq!(
            locus_midpoint(&locus),
            Some((midpoint_slice(block, 2, 8), 5))
        );
    }

    #[test]
    fn test_midpoint_spans_into_second_slice() {
        let first = midpoint_node(1, 0, 10);
        let second = midpoint_node(2, 0, 10);
        let locus = GraphLocus {
            slices: vec![midpoint_slice(first, 8, 10), midpoint_slice(second, 0, 10)],
        };
        // total length 12, half = 6; first slice covers 2 bases (consumed=2),
        // midpoint falls 4 bases into the second slice.
        assert_eq!(
            locus_midpoint(&locus),
            Some((midpoint_slice(second, 0, 10), 4))
        );
    }

    #[test]
    fn test_midpoint_of_empty_locus_is_none() {
        let locus = GraphLocus { slices: vec![] };
        assert_eq!(locus_midpoint(&locus), None);
    }
}
