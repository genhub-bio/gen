use std::collections::{HashMap, HashSet, VecDeque, hash_map::Entry};

use gen_core::{Workspace, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::{
    block_group::BlockGroup, db::GraphConnection, locus::GraphLocus, node::Node,
    sequence::SequenceError,
};
use gen_tui::{
    ViewportState,
    geometry::{WorldPos, WorldRect},
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::{GraphWidget, NODE_GLYPH},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer, PathStyle},
    theme::current_theme,
};
use petgraph::{Direction, visit::NodeIndexable};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

use crate::views::{
    annotation_track::{
        AnnotationSpan, CoordinatePrefixMaximumIndex, GraphNodeFragmentLookup,
        SegmentContainmentIndex, locus_should_show_in_truncated, span_label_text,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenGraphNodeSizer {
    bases_per_cell: u64,
}

impl Default for GenGraphNodeSizer {
    fn default() -> Self {
        Self { bases_per_cell: 1 }
    }
}

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
            VisualDetail::Full => {
                let scaled_width = sequence_length.div_ceil(self.bases_per_cell.max(1));
                (scaled_width.max(sequence_length.min(13)), 1u64)
            }
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
            VisualDetail::Full if self.bases_per_cell > 1 && node.length() > 13 => map_scaled_col(
                raw_col,
                node.length(),
                self.get_node_size(node, detail_level).0,
            ),
            _ => raw_col,
        }
    }

    fn zoom_scale(&self) -> u64 {
        self.bases_per_cell
    }

    fn zoom_out_scale(&mut self) -> bool {
        let Some(next_scale) = self.bases_per_cell.checked_mul(2) else {
            return false;
        };
        self.bases_per_cell = next_scale;
        true
    }

    fn zoom_in_scale(&mut self) -> bool {
        if self.bases_per_cell <= 1 {
            return false;
        }
        self.bases_per_cell /= 2;
        true
    }

    fn reset_zoom_scale(&mut self) -> bool {
        if self.bases_per_cell == 1 {
            return false;
        }
        self.bases_per_cell = 1;
        true
    }
}

fn map_scaled_col(raw_col: i64, sequence_length: i64, visual_width: u64) -> i64 {
    if sequence_length <= 1 || visual_width <= 1 {
        return 0;
    }
    if raw_col < 0 {
        return raw_col;
    }
    if raw_col >= sequence_length {
        return i64::try_from(visual_width).unwrap_or(i64::MAX)
            + raw_col.saturating_sub(sequence_length);
    }

    let source_offset = i128::from(raw_col);
    let source_span = i128::from(sequence_length - 1);
    let visual_span = i128::from(visual_width - 1);
    let rounded = (source_offset * visual_span * 2 + source_span) / (source_span * 2);
    i64::try_from(rounded).unwrap_or(i64::MAX)
}

/// Domain-specific node renderer for GenGraph that handles database sequence fetching
/// and genomic sequence visualization with caching.
pub struct GenGraphNodeRenderer<'a> {
    conn: &'a GraphConnection,
    workspace: &'a Workspace,
    cache: HashMap<(GraphNode, i64, i64), String>,
    zoom_scale: u64,
}

impl<'a> GenGraphNodeRenderer<'a> {
    /// Create a new GenGraph node renderer with database connection
    pub fn new(conn: &'a GraphConnection, workspace: &'a Workspace) -> Self {
        Self {
            conn,
            workspace,
            cache: HashMap::new(),
            zoom_scale: 1,
        }
    }

    /// Get the database connection (for accessing from other code)
    pub fn connection(&self) -> &'a GraphConnection {
        self.conn
    }

    /// Fetch sequence for a GenGraph node with caching
    pub fn get_sequence(&mut self, node_key: &GraphNode) -> Result<String, SequenceError> {
        self.get_sequence_range_ref(node_key, node_key.sequence_start, node_key.sequence_end)
            .map(str::to_owned)
    }

    fn get_sequence_ref(&mut self, node_key: &GraphNode) -> Result<&str, SequenceError> {
        self.get_sequence_range_ref(node_key, node_key.sequence_start, node_key.sequence_end)
    }

    fn get_sequence_range_ref(
        &mut self,
        node_key: &GraphNode,
        start: i64,
        end: i64,
    ) -> Result<&str, SequenceError> {
        let cache_key = (*node_key, start, end);
        if let Entry::Vacant(entry) = self.cache.entry(cache_key) {
            let (db_node_id, start, end) = (node_key.node_id, start, end);
            let sequences =
                Node::get_sequences_by_node_ids(self.conn, self.workspace, &[db_node_id], None);
            let sequence = match sequences.get(&db_node_id) {
                Some(sequence) => sequence.get_sequence(start, end)?,
                None => "?".repeat((end - start).max(0) as usize),
            };
            entry.insert(sequence);
        }

        Ok(self
            .cache
            .get(&cache_key)
            .expect("should cache a sequence after loading it"))
    }
}

impl NodeRenderer<GenGraph> for GenGraphNodeRenderer<'_> {
    fn set_zoom_scale(&mut self, zoom_scale: u64) {
        self.zoom_scale = zoom_scale.max(1);
    }

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
        let Some(visible_area) = buffer.calculate_visible_area(area) else {
            return;
        };

        if is_zero_width_junction(node_id) {
            let position = area.left_center();
            let (glyph, color) = buffer
                .get_char_styled(position)
                .map(|(glyph, style)| {
                    let glyph = if glyph == ' ' {
                        ZERO_WIDTH_GLYPH
                    } else {
                        glyph
                    };
                    (glyph, style.fg.unwrap_or(theme[0x05]))
                })
                .unwrap_or((ZERO_WIDTH_GLYPH, theme[0x05]));
            let edge_style = Style::default().bg(theme[0x00]).fg(color);
            buffer.set_char_styled(position, glyph, edge_style);
            return;
        }

        let background_style = Style::default().bg(theme[0x05]);
        let text_style = Style::default().bg(theme[0x05]).fg(theme[0x00]);

        buffer.fill_rect(visible_area, ' ');
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
                let sequence = self.get_sequence_ref(node_id).unwrap_or("Unknown Sequence");
                let max_width = 13u32;
                let truncated = inner_truncation(sequence, max_width);
                buffer.set_string_styled(area.left_center(), &truncated, text_style);
            }
            VisualDetail::Full => {
                // WorldRect corners are inclusive, while width() is the corner difference.
                let visible_cell_count = area.width().saturating_add(1).max(0) as u64;
                if self.zoom_scale > 1 && node_id.length().max(0) as u64 > visible_cell_count {
                    let aggregate_glyphs = "·".repeat(visible_area.width() as usize + 1);
                    buffer.set_string_styled(
                        WorldPos::new(visible_area.min.x, area.left_center().y),
                        &aggregate_glyphs,
                        text_style,
                    );
                    return;
                }

                let node_length = usize::try_from(node_id.length().max(0)).unwrap_or(usize::MAX);
                let visible_start = usize::try_from(visible_area.min.x.saturating_sub(area.min.x))
                    .unwrap_or(usize::MAX)
                    .min(node_length);
                let visible_end = usize::try_from(
                    visible_area
                        .max
                        .x
                        .saturating_sub(area.min.x)
                        .saturating_add(1),
                )
                .unwrap_or(usize::MAX)
                .min(node_length)
                .max(visible_start);
                let sequence_start = node_id
                    .sequence_start
                    .saturating_add(i64::try_from(visible_start).unwrap_or(i64::MAX));
                let sequence_end = node_id
                    .sequence_start
                    .saturating_add(i64::try_from(visible_end).unwrap_or(i64::MAX));
                let visible_sequence =
                    match self.get_sequence_range_ref(node_id, sequence_start, sequence_end) {
                        Ok(sequence) => sequence,
                        Err(_) => {
                            let fallback = "Unknown Sequence";
                            let fallback_start = visible_start.min(fallback.len());
                            let fallback_end = visible_end.min(fallback.len()).max(fallback_start);
                            fallback
                                .as_bytes()
                                .get(fallback_start..fallback_end)
                                .and_then(|bases| std::str::from_utf8(bases).ok())
                                .unwrap_or_default()
                        }
                    };
                buffer.set_string_styled(
                    WorldPos::new(visible_area.min.x, area.left_center().y),
                    visible_sequence,
                    text_style,
                );
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
pub fn create_gen_graph_widget<'a>(
    conn: &'a GraphConnection,
    workspace: &'a Workspace,
) -> GraphWidget<'a, GenGraph, GenGraphNodeSizer, GenGraphNodeRenderer<'a>> {
    let renderer = GenGraphNodeRenderer::new(conn, workspace);
    GraphWidget::with_renderer(renderer)
}

/// Contract zero-width routing positions in a display copy while retaining real cycles
/// and edge metadata in the source graph used by sequence operations.
fn graph_for_display(graph: &GenGraph) -> GenGraph {
    let mut contracted_graph = graph.clone();
    let junctions = graph
        .nodes()
        .filter(is_zero_width_junction)
        .collect::<Vec<_>>();

    for junction in junctions {
        let incoming = contracted_graph
            .neighbors_directed(junction, Direction::Incoming)
            .filter(|source| *source != junction)
            .filter_map(|source| {
                contracted_graph
                    .edge_weight(source, junction)
                    .map(|edges| (source, edges.clone()))
            })
            .collect::<Vec<_>>();
        let outgoing = contracted_graph
            .neighbors_directed(junction, Direction::Outgoing)
            .filter(|target| *target != junction)
            .filter_map(|target| {
                contracted_graph
                    .edge_weight(junction, target)
                    .map(|edges| (target, edges.clone()))
            })
            .collect::<Vec<_>>();

        for (source, incoming_edges) in &incoming {
            for (target, outgoing_edges) in &outgoing {
                if source == target {
                    continue;
                }

                let mut projected_edges = incoming_edges.clone();
                for edge in outgoing_edges {
                    if !projected_edges.contains(edge) {
                        projected_edges.push(*edge);
                    }
                }
                if let Some(existing_edges) = contracted_graph.edge_weight_mut(*source, *target) {
                    for edge in projected_edges {
                        if !existing_edges.contains(&edge) {
                            existing_edges.push(edge);
                        }
                    }
                } else {
                    contracted_graph.add_edge(*source, *target, projected_edges);
                }
            }
        }

        contracted_graph.remove_node(junction);
    }

    let retained_nodes = graph
        .nodes()
        .filter(|node| !is_zero_width_junction(node))
        .collect::<Vec<_>>();
    let node_order = retained_nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (*node, index))
        .collect::<HashMap<_, _>>();
    let mut display_graph = GenGraph::new();
    for node in &retained_nodes {
        display_graph.add_node(*node);
    }
    for source in &retained_nodes {
        let mut targets = contracted_graph
            .neighbors_directed(*source, Direction::Outgoing)
            .collect::<Vec<_>>();
        targets.sort_unstable_by_key(|target| node_order[target]);
        for target in targets {
            let edge_bundle = contracted_graph
                .edge_weight(*source, target)
                .expect("should retain contracted edge bundle")
                .clone();
            display_graph.add_edge(*source, target, edge_bundle);
        }
    }

    display_graph
}

/// Project the source graph and its actual pruning result before comparing dimmed routes.
fn display_graph_with_lowlights(
    graph: &GenGraph,
) -> (
    GenGraph,
    HashSet<(GraphNode, GraphNode)>,
    HashSet<GraphNode>,
) {
    let mut pruned_graph = graph.clone();
    BlockGroup::prune_graph(&mut pruned_graph);

    let display_graph = graph_for_display(graph);
    let pruned_display_graph = graph_for_display(&pruned_graph);
    let dimmed_edges = display_graph
        .all_edges()
        .filter_map(|(source_node, target_node, _)| {
            pruned_display_graph
                .edge_weight(source_node, target_node)
                .is_none()
                .then_some((source_node, target_node))
        })
        .collect();
    let dimmed_nodes = display_graph
        .nodes()
        .filter(|node| !pruned_display_graph.contains_node(*node))
        .collect();

    (display_graph, dimmed_edges, dimmed_nodes)
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
    let (display_graph, dimmed_edges, dimmed_nodes) = display_graph_with_lowlights(&graph);
    create_gen_graph_controller_with_lowlights(display_graph, dimmed_edges, dimmed_nodes)
}

/// Create a configured controller for graphs whose edges must remain fully visible.
///
/// Diff views already remove persistence-only edit-site marker edges and retain
/// historical context, so applying current-path pruning would obscure changes.
pub fn create_gen_graph_controller_without_dimming(
    graph: GenGraph,
) -> GraphController<GenGraph, GenGraphNodeSizer> {
    create_gen_graph_controller_with_lowlights(
        graph_for_display(&graph),
        HashSet::new(),
        HashSet::new(),
    )
}

fn create_gen_graph_controller_with_lowlights(
    graph: GenGraph,
    dimmed_edges: HashSet<(GraphNode, GraphNode)>,
    dimmed_nodes: HashSet<GraphNode>,
) -> GraphController<GenGraph, GenGraphNodeSizer> {
    let node_sizer = GenGraphNodeSizer::default();
    let mut controller = GraphController::new(graph, node_sizer);
    for edge in dimmed_edges {
        controller.dim_edge(edge);
    }
    for node in dimmed_nodes {
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

/// Compute unclipped world-space bounds for a matched region in a graph locus.
fn locus_label_world_bounds<S: NodeSizer<GenGraph>>(
    locus: &GraphLocus,
    pos_map: &ViewportNodePositionMap,
    node_sizer: &S,
    detail_level: VisualDetail,
) -> Option<(WorldPos, WorldPos)> {
    let block_world_pos = |block: GraphNode, col_raw: i64| -> Option<WorldPos> {
        let &(center, size) = pos_map.get(&block)?;
        let rect = WorldRect::from_center_and_size(center, size);
        let col = node_sizer.map_column(&block, col_raw, detail_level);
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

    Some((
        WorldPos::new(left_pos.x, y_max),
        WorldPos::new(right_pos.x, y_min),
    ))
}

fn clip_label_bounds(
    (left_pos, right_pos): (WorldPos, WorldPos),
    viewport_state: &ViewportState,
) -> Option<(WorldPos, WorldPos)> {
    let cam = viewport_state.camera_rect();
    let clipped_left_x = left_pos.x.max(cam.min.x);
    let clipped_right_x = right_pos.x.min(cam.max.x);
    if clipped_right_x < clipped_left_x {
        return None;
    }

    let left_pos = WorldPos::new(clipped_left_x, left_pos.y);
    let right_pos = WorldPos::new(clipped_right_x, right_pos.y);

    Some((left_pos, right_pos))
}

/// Compute world-space bounds for a matched region and clip them to the visible camera.
///
/// Column offsets map through the current detail level, and viewport-spanning annotations
/// place their labels near the visible portion. Returns `None` if the locus has no visible
/// block or its x span falls outside the camera rect.
pub fn locus_label_bounds<S: NodeSizer<GenGraph>>(
    locus: &GraphLocus,
    pos_map: &ViewportNodePositionMap,
    node_sizer: &S,
    detail_level: VisualDetail,
    viewport_state: &ViewportState,
) -> Option<(WorldPos, WorldPos)> {
    clip_label_bounds(
        locus_label_world_bounds(locus, pos_map, node_sizer, detail_level)?,
        viewport_state,
    )
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
/// - Consecutive annotated nodes: their real edge route is tinted, including any
///   zero-width junctions introduced at deletion boundaries.
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

    for (source, target) in m.slices.iter().zip(m.slices.iter().skip(1)) {
        let Some(path) =
            edge_path_across_zero_width_junctions(controller.graph(), source.block, target.block)
        else {
            continue;
        };
        for edge in path.windows(2) {
            controller.set_edge_highlight((edge[0], edge[1]), style);
        }
    }
}

/// Find the edge route between two annotation blocks when every intermediate block is a
/// zero-width junction. Junctions are rendering topology rather than annotation content,
/// so they do not appear in the locus itself but their edges still need the span's color.
fn edge_path_across_zero_width_junctions(
    graph: &GenGraph,
    source: GraphNode,
    target: GraphNode,
) -> Option<Vec<GraphNode>> {
    let mut queue = VecDeque::from([source]);
    let mut visited = HashSet::from([source]);
    let mut predecessor = HashMap::new();

    while let Some(current) = queue.pop_front() {
        let neighbors = graph
            .neighbors_directed(current, Direction::Outgoing)
            .chain(graph.neighbors_directed(current, Direction::Incoming));
        for neighbor in neighbors {
            if !visited.insert(neighbor) {
                continue;
            }
            predecessor.insert(neighbor, current);
            if neighbor == target {
                let mut path = vec![target];
                let mut cursor = target;
                while cursor != source {
                    cursor = *predecessor.get(&cursor)?;
                    path.push(cursor);
                }
                path.reverse();
                return Some(path);
            }
            if is_zero_width_junction(&neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    None
}

/// A mapped cell rectangle: the node it's on, and its top-left/bottom-right columns.
type CellRegion = (GraphNode, (i64, i64), (i64, i64));
type ViewportNodePositionMap = HashMap<GraphNode, (WorldPos, (u64, u64))>;

struct SpanOverlayCandidate<'a> {
    overlay_index: usize,
    span: &'a AnnotationSpan,
    total_length: i64,
    locus: GraphLocus,
    regions: Vec<CellRegion>,
}

struct AnnotationLabelCandidate<'a> {
    span: &'a AnnotationSpan,
    style: PathStyle,
    bounds: (WorldPos, WorldPos),
}

struct PreparedAnnotationLabel {
    overlay_index: usize,
    locus: GraphLocus,
    world_bounds: Option<(WorldPos, WorldPos)>,
}

// The full viewer keeps exact overlay snapshots and viewport positions so unchanged spans
// can reuse their graph projections and world bounds. Callers invalidate this on graph
// replacement; camera clipping and label placement remain current on every draw.
#[derive(Default)]
pub(crate) struct OverlayRenderCache {
    detail_level: Option<VisualDetail>,
    zoom_scale: Option<u64>,
    overlays: Option<Vec<GraphOverlay>>,
    labels: Vec<PreparedAnnotationLabel>,
    hidden_truncated_labels: bool,
    viewport_positions: Option<ViewportNodePositionMap>,
}

impl OverlayRenderCache {
    pub(crate) fn invalidate(&mut self) {
        self.detail_level = None;
        self.zoom_scale = None;
        self.overlays = None;
        self.labels.clear();
        self.hidden_truncated_labels = false;
        self.viewport_positions = None;
    }

    fn matches(
        &self,
        detail_level: VisualDetail,
        zoom_scale: u64,
        overlays: &[GraphOverlay],
    ) -> bool {
        self.detail_level == Some(detail_level)
            && self.zoom_scale == Some(zoom_scale)
            && self.overlays.as_deref() == Some(overlays)
    }

    fn prepare_labels<S: NodeSizer<GenGraph>>(
        &mut self,
        controller: &GraphController<GenGraph, S>,
        overlays: &[GraphOverlay],
    ) {
        let detail_level = controller.get_detail_level();
        let mut ordered_labels = overlays
            .iter()
            .enumerate()
            .filter_map(|(overlay_index, overlay)| {
                let span = overlay.span()?;
                if span.name.is_empty()
                    || matches!(
                        (detail_level, &overlay.source),
                        (VisualDetail::Minimal, OverlaySource::Track(_))
                    )
                {
                    return None;
                }
                let total_length = span
                    .segments
                    .iter()
                    .map(|segment| segment.end - segment.start)
                    .sum::<i64>();
                Some((overlay_index, total_length))
            })
            .collect::<Vec<_>>();
        ordered_labels.sort_by_key(|(_, total_length)| -*total_length);

        let fragment_lookup = GraphNodeFragmentLookup::new(controller.graph());
        self.labels.clear();
        self.hidden_truncated_labels = false;
        for (overlay_index, _) in ordered_labels {
            let Some(locus) = overlays[overlay_index]
                .span()
                .and_then(|span| fragment_lookup.locus_for_span(span))
            else {
                continue;
            };
            if detail_level == VisualDetail::Truncated && !locus_should_show_in_truncated(&locus) {
                self.hidden_truncated_labels = true;
                continue;
            }
            self.labels.push(PreparedAnnotationLabel {
                overlay_index,
                locus,
                world_bounds: None,
            });
        }
        self.viewport_positions = None;
        self.detail_level = Some(detail_level);
        self.zoom_scale = Some(controller.zoom_scale());
        self.overlays = Some(overlays.to_vec());
    }
}

pub(crate) fn reapply_overlays_cached<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    overlays: &mut [GraphOverlay],
    color_cache: &mut AnnotationColorCache,
    render_cache: &mut OverlayRenderCache,
) {
    if render_cache.matches(
        controller.get_detail_level(),
        controller.zoom_scale(),
        overlays,
    ) {
        return;
    }

    reapply_overlays(controller, overlays, color_cache);
    render_cache.prepare_labels(controller, overlays);
}

struct NodeColorOverlapIndex {
    starts: Vec<i64>,
    by_color: HashMap<Color, CoordinatePrefixMaximumIndex>,
}

struct ColorOverlapIndex {
    by_node: HashMap<GraphNode, NodeColorOverlapIndex>,
}

impl ColorOverlapIndex {
    fn new(regions: &[CellRegion]) -> Self {
        let mut starts_by_node: HashMap<GraphNode, Vec<i64>> = HashMap::new();
        for region in regions {
            starts_by_node.entry(region.0).or_default().push(region.1.0);
        }
        let by_node = starts_by_node
            .into_iter()
            .map(|(node, mut starts)| {
                starts.sort_unstable();
                starts.dedup();
                (
                    node,
                    NodeColorOverlapIndex {
                        starts,
                        by_color: HashMap::new(),
                    },
                )
            })
            .collect();
        Self { by_node }
    }

    fn colors_overlapping(&self, region: &CellRegion) -> Vec<Color> {
        self.by_node
            .get(&region.0)
            .into_iter()
            .flat_map(|node| node.by_color.iter())
            .filter_map(|(color, index)| {
                (index.maximum_end_for_start_at_most(region.2.0) >= region.1.0).then_some(*color)
            })
            .collect()
    }

    fn insert_region(&mut self, region: &CellRegion, color: Color) {
        let Some(node) = self.by_node.get_mut(&region.0) else {
            return;
        };
        if let Entry::Vacant(entry) = node.by_color.entry(color) {
            entry.insert(CoordinatePrefixMaximumIndex::new(node.starts.clone()));
        }
        if let Some(index) = node.by_color.get_mut(&color) {
            index.insert(region.1.0, region.2.0);
        }
    }
}

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
/// highlight columns (overlay or detail changes). The full viewer uses
/// `reapply_overlays_cached` to skip unchanged passes.
pub fn reapply_overlays<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    overlays: &mut [GraphOverlay],
    color_cache: &mut AnnotationColorCache,
) {
    let detail_level = controller.get_detail_level();
    let node_sizer = &controller.partition_controller.node_sizer;
    let graph = controller.graph();
    let fragment_lookup = GraphNodeFragmentLookup::new(graph);

    // DB-loaded tracks are too busy to paint at minimal detail. At truncated detail, a span
    // confined to a partial slice of a single node remains suppressed for ordinary annotations,
    // while explicit search/ad-hoc highlights stay visible as user targets.
    let mut span_candidates = Vec::new();
    for (overlay_index, overlay) in overlays.iter().enumerate() {
        let Some(span) = overlay.span() else {
            continue;
        };
        if matches!(
            (detail_level, &overlay.source),
            (VisualDetail::Minimal, OverlaySource::Track(_))
        ) {
            continue;
        }
        let Some(locus) = fragment_lookup.locus_for_span(span) else {
            continue;
        };
        if detail_level == VisualDetail::Truncated
            && !matches!(
                &overlay.source,
                OverlaySource::Adhoc | OverlaySource::Search
            )
            && !locus_should_show_in_truncated(&locus)
        {
            continue;
        }
        let total_length = span
            .segments
            .iter()
            .map(|segment| segment.end - segment.start)
            .sum();
        let regions = locus
            .slices
            .iter()
            .map(|slice| slice_region(node_sizer, detail_level, slice))
            .collect();
        span_candidates.push(SpanOverlayCandidate {
            overlay_index,
            span,
            total_length,
            locus,
            regions,
        });
    }
    span_candidates.sort_by_key(|candidate| -candidate.total_length);

    let all_regions = span_candidates
        .iter()
        .flat_map(|candidate| candidate.regions.iter().copied())
        .collect::<Vec<_>>();
    let mut color_overlap_index = ColorOverlapIndex::new(&all_regions);

    // Decide every span's locus and color first (only needs `&controller`); painting
    // (`&mut controller`) happens in a second pass once every color is settled.
    let accents = accent_colors();
    let mut decisions: Vec<(usize, GraphLocus, Color)> = Vec::new();
    for candidate in span_candidates {
        let span = candidate.span;
        let mut used = Vec::new();
        for region in &candidate.regions {
            for color in color_overlap_index.colors_overlapping(region) {
                if !used.contains(&color) {
                    used.push(color);
                }
            }
        }

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

        for region in &candidate.regions {
            color_overlap_index.insert_region(region, color);
        }
        decisions.push((candidate.overlay_index, candidate.locus, color));
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
    let mut labeled: Vec<(&AnnotationSpan, PathStyle, i64)> = overlays
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
                .map(|span| {
                    let total_length = span
                        .segments
                        .iter()
                        .map(|segment| segment.end - segment.start)
                        .sum();
                    (span, overlay.style, total_length)
                })
        })
        .collect();
    if labeled.is_empty() {
        return false;
    }
    labeled.sort_by_key(|(_, _, total_length)| -*total_length);

    let fragment_lookup = GraphNodeFragmentLookup::new(controller.graph());
    let pos_map = viewport_pos_map(controller);
    let max_distance = if detail_level == VisualDetail::Minimal {
        10
    } else {
        5
    };

    let mut any_hidden = false;
    let mut candidates = Vec::with_capacity(labeled.len());
    for (span, style, _total_length) in labeled {
        let Some(locus) = fragment_lookup.locus_for_span(span) else {
            continue;
        };
        if detail_level == VisualDetail::Truncated && !locus_should_show_in_truncated(&locus) {
            any_hidden = true;
            continue;
        }
        let Some(bounds) = locus_label_bounds(
            &locus,
            &pos_map,
            &controller.partition_controller.node_sizer,
            detail_level,
            &controller.viewport_state,
        ) else {
            continue;
        };
        candidates.push(AnnotationLabelCandidate {
            span,
            style,
            bounds,
        });
    }

    draw_annotation_label_candidates(buf, area, controller, candidates, any_hidden, max_distance)
}

pub(crate) fn draw_annotation_labels_cached<S: NodeSizer<GenGraph>>(
    buf: &mut Buffer,
    area: Rect,
    controller: &GraphController<GenGraph, S>,
    overlays: &[GraphOverlay],
    render_cache: &mut OverlayRenderCache,
) -> bool {
    let detail_level = controller.get_detail_level();
    if !render_cache.matches(detail_level, controller.zoom_scale(), overlays) {
        return draw_annotation_labels(buf, area, controller, overlays);
    }

    let pos_map = viewport_pos_map(controller);
    if render_cache.viewport_positions.as_ref() != Some(&pos_map) {
        for prepared in &mut render_cache.labels {
            prepared.world_bounds = locus_label_world_bounds(
                &prepared.locus,
                &pos_map,
                &controller.partition_controller.node_sizer,
                detail_level,
            );
        }
        render_cache.viewport_positions = Some(pos_map);
    }

    let max_distance = if detail_level == VisualDetail::Minimal {
        10
    } else {
        5
    };
    let mut candidates = Vec::with_capacity(render_cache.labels.len());
    for prepared in &render_cache.labels {
        let Some(overlay) = overlays.get(prepared.overlay_index) else {
            continue;
        };
        let Some(span) = overlay.span() else {
            continue;
        };
        let Some(bounds) = prepared
            .world_bounds
            .and_then(|world_bounds| clip_label_bounds(world_bounds, &controller.viewport_state))
        else {
            continue;
        };
        candidates.push(AnnotationLabelCandidate {
            span,
            style: overlay.style,
            bounds,
        });
    }

    if candidates.is_empty() {
        return render_cache.hidden_truncated_labels;
    }

    draw_annotation_label_candidates(
        buf,
        area,
        controller,
        candidates,
        render_cache.hidden_truncated_labels,
        max_distance,
    )
}

fn draw_annotation_label_candidates<S: NodeSizer<GenGraph>>(
    buf: &mut Buffer,
    area: Rect,
    controller: &GraphController<GenGraph, S>,
    candidates: Vec<AnnotationLabelCandidate<'_>>,
    mut any_hidden: bool,
    max_distance: u16,
) -> bool {
    let theme = current_theme();
    let span_refs: Vec<&AnnotationSpan> =
        candidates.iter().map(|candidate| candidate.span).collect();
    let mut containment_index = SegmentContainmentIndex::for_spans(&span_refs);
    let mut covered = vec![false; candidates.len()];
    for index in (0..candidates.len()).rev() {
        let candidate = &candidates[index];
        covered[index] = containment_index.covers_every_segment(candidate.span);
        containment_index.insert_span(candidate.span);
    }

    for (index, candidate) in candidates.iter().enumerate() {
        if covered[index] {
            any_hidden = true;
            continue;
        }
        let color = match candidate.style.color {
            Color::Reset => theme[0x06],
            other => other,
        };
        let label = span_label_text(candidate.span);
        if draw_label_near_pos(
            buf,
            area,
            candidate.bounds,
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

    use gen_core::{
        HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID,
        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand,
    };
    use gen_graph::GraphEdge;
    use gen_models::{
        assets::{AssetRef, AssetRole},
        sample::Sample,
        sequence::Sequence,
    };
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

    fn prepare_label_test_controller(
        graph: GenGraph,
        detail_level: VisualDetail,
        area: Rect,
    ) -> GraphController<GenGraph, GenGraphNodeSizer> {
        let mut controller = create_gen_graph_controller(graph);
        controller.set_detail_level(detail_level);
        controller.viewport_state.viewport_bounds = area;
        controller.ensure_camera_coverage().unwrap();
        controller.rebuild_viewport_graph().unwrap();
        controller
    }

    fn span_overlay(span: AnnotationSpan) -> GraphOverlay {
        GraphOverlay {
            content: OverlayContent::Span(span),
            source: OverlaySource::Track("labels".to_string()),
            style: PathStyle::new(Color::Cyan),
        }
    }

    #[test]
    fn test_overlay_render_cache_invalidates_on_exact_overlay_and_graph_changes() {
        let node = GraphNode {
            node_id: HashId::convert_str("cached-overlay-node"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let area = Rect::new(0, 0, 40, 5);
        let mut controller = prepare_label_test_controller(graph, VisualDetail::Full, area);
        let mut overlays = vec![span_overlay(AnnotationSpan {
            id: HashId::convert_str("cached-overlay-span"),
            name: "cached".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 0,
                end: 8,
                strand: Strand::Forward,
            }],
        })];
        let mut color_cache = AnnotationColorCache::new();
        let mut render_cache = OverlayRenderCache::default();

        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 1, &overlays));
        let first_highlights = controller.highlights.clone();
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert_eq!(controller.highlights, first_highlights);

        let span = match &mut overlays[0].content {
            OverlayContent::Span(span) => span,
            OverlayContent::Path(_) => unreachable!(),
        };
        span.name = "renamed".to_string();
        assert!(!render_cache.matches(VisualDetail::Full, 1, &overlays));
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 1, &overlays));

        let span = match &mut overlays[0].content {
            OverlayContent::Span(span) => span,
            OverlayContent::Path(_) => unreachable!(),
        };
        span.segments[0].start = 2;
        assert!(!render_cache.matches(VisualDetail::Full, 1, &overlays));
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 1, &overlays));

        overlays[0].style.color = Color::Yellow;
        assert!(!render_cache.matches(VisualDetail::Full, 1, &overlays));
        overlays[0].source = OverlaySource::Search;
        assert!(!render_cache.matches(VisualDetail::Full, 1, &overlays));
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 1, &overlays));

        controller.set_detail_level(VisualDetail::Truncated);
        assert!(!render_cache.matches(VisualDetail::Truncated, 1, &overlays));
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Truncated, 1, &overlays));

        overlays.clear();
        assert!(!render_cache.matches(VisualDetail::Truncated, 1, &overlays));
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(controller.highlights.is_empty());

        overlays.push(span_overlay(AnnotationSpan {
            id: HashId::convert_str("added-cached-overlay-span"),
            name: "added".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 0,
                end: 20,
                strand: Strand::Forward,
            }],
        }));
        assert!(!render_cache.matches(VisualDetail::Truncated, 1, &overlays));
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Truncated, 1, &overlays));
        assert_eq!(controller.highlights.len(), 1);

        assert!(render_cache.matches(VisualDetail::Truncated, 1, &overlays));
        let replacement_node = GraphNode {
            node_id: node.node_id,
            sequence_start: 10,
            sequence_end: 20,
        };
        let mut replacement_graph = GenGraph::new();
        replacement_graph.add_node(replacement_node);
        let mut replacement_controller =
            prepare_label_test_controller(replacement_graph.clone(), VisualDetail::Truncated, area);
        let mut uncached_replacement_controller =
            prepare_label_test_controller(replacement_graph, VisualDetail::Truncated, area);
        let mut replacement_overlays = overlays.clone();
        let mut uncached_replacement_overlays = overlays.clone();
        let mut uncached_replacement_color_cache = color_cache.clone();
        render_cache.invalidate();
        assert!(!render_cache.matches(VisualDetail::Truncated, 1, &replacement_overlays));
        reapply_overlays_cached(
            &mut replacement_controller,
            &mut replacement_overlays,
            &mut color_cache,
            &mut render_cache,
        );
        reapply_overlays(
            &mut uncached_replacement_controller,
            &mut uncached_replacement_overlays,
            &mut uncached_replacement_color_cache,
        );
        assert_eq!(
            replacement_controller.highlights, uncached_replacement_controller.highlights,
            "should rebuild projections against the replacement graph after invalidation"
        );
    }

    #[test]
    fn test_overlay_cache_and_label_bounds_follow_proportional_zoom() {
        let node = GraphNode {
            node_id: HashId::convert_str("scaled-label-node"),
            sequence_start: 0,
            sequence_end: 100,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let area = Rect::new(0, 0, 40, 5);
        let mut controller = prepare_label_test_controller(graph, VisualDetail::Full, area);
        let mut overlays = vec![span_overlay(AnnotationSpan {
            id: HashId::convert_str("scaled-label-span"),
            name: "scaled".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 3,
                end: 29,
                strand: Strand::Forward,
            }],
        })];
        let mut color_cache = AnnotationColorCache::new();
        let mut render_cache = OverlayRenderCache::default();
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 1, &overlays));

        controller.zoom_out();
        controller.rebuild_viewport_graph().unwrap();
        assert_eq!(controller.zoom_scale(), 2);
        assert!(!render_cache.matches(VisualDetail::Full, 2, &overlays));
        reapply_overlays_cached(
            &mut controller,
            &mut overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 2, &overlays));

        let node_sizer = &controller.partition_controller.node_sizer;
        let (actual_cell_position, actual_top_left, actual_bottom_right, _) = controller
            .get_cell_highlights()
            .first()
            .expect("should project the annotation into visible cells");
        let (node_center, _) = viewport_pos_map(&controller)[&node];
        assert_eq!(*actual_cell_position, node_center);
        assert_eq!(
            (*actual_top_left, *actual_bottom_right),
            (
                (node_sizer.map_column(&node, 3, VisualDetail::Full), 0),
                (node_sizer.map_column(&node, 28, VisualDetail::Full), 0),
            )
        );

        let span = overlays[0]
            .span()
            .expect("should retain the annotation span");
        let locus = GraphNodeFragmentLookup::new(controller.graph())
            .locus_for_span(span)
            .expect("should map the annotation into the graph");
        let pos_map = viewport_pos_map(&controller);
        let bounds = locus_label_world_bounds(
            &locus,
            &pos_map,
            &controller.partition_controller.node_sizer,
            VisualDetail::Full,
        )
        .expect("should compute scaled label bounds");
        let (center, size) = pos_map[&node];
        let rect = WorldRect::from_center_and_size(center, size);
        let node_sizer = &controller.partition_controller.node_sizer;
        assert_eq!(
            bounds,
            (
                WorldPos::new(
                    rect.min.x + node_sizer.map_column(&node, 3, VisualDetail::Full),
                    center.y,
                ),
                WorldPos::new(
                    rect.min.x + node_sizer.map_column(&node, 28, VisualDetail::Full),
                    center.y,
                ),
            )
        );
    }

    #[test]
    fn test_scaled_disjoint_annotations_get_distinct_colors_after_cell_mapping() {
        let node = GraphNode {
            node_id: HashId::convert_str("scaled-collision-node"),
            sequence_start: 0,
            sequence_end: 100,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let area = Rect::new(0, 0, 120, 5);
        let mut controller = prepare_label_test_controller(graph, VisualDetail::Full, area);
        controller.zoom_out();
        controller.rebuild_viewport_graph().unwrap();
        assert_eq!(controller.zoom_scale(), 2);

        let mut overlays = vec![
            span_overlay(AnnotationSpan {
                id: HashId::convert_str("scaled-collision-first"),
                name: "first-base".to_string(),
                segments: vec![AnnotationSegment {
                    node_id: node.node_id,
                    start: 2,
                    end: 3,
                    strand: Strand::Forward,
                }],
            }),
            span_overlay(AnnotationSpan {
                id: HashId::convert_str("scaled-collision-second"),
                name: "second-base".to_string(),
                segments: vec![AnnotationSegment {
                    node_id: node.node_id,
                    start: 3,
                    end: 4,
                    strand: Strand::Forward,
                }],
            }),
        ];
        reapply_overlays(
            &mut controller,
            &mut overlays,
            &mut AnnotationColorCache::new(),
        );

        assert_ne!(overlays[0].style.color, overlays[1].style.color);
        let highlights = controller.get_cell_highlights();
        assert_eq!(highlights.len(), 2);
        for (_, top_left, bottom_right, _) in highlights {
            assert_eq!((*top_left, *bottom_right), ((1, 0), (1, 0)));
        }
    }

    #[test]
    fn test_cached_annotation_labels_match_after_camera_and_detail_changes() {
        let node = GraphNode {
            node_id: HashId::convert_str("cached-label-node"),
            sequence_start: 0,
            sequence_end: 30,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let area = Rect::new(0, 0, 80, 5);
        let mut cached_controller =
            prepare_label_test_controller(graph.clone(), VisualDetail::Full, area);
        let mut public_controller = prepare_label_test_controller(graph, VisualDetail::Full, area);
        let make_overlays = || {
            vec![span_overlay(AnnotationSpan {
                id: HashId::convert_str("cached-label-span"),
                name: "ADA2".to_string(),
                segments: vec![AnnotationSegment {
                    node_id: node.node_id,
                    start: 4,
                    end: 14,
                    strand: Strand::Forward,
                }],
            })]
        };
        let mut cached_overlays = make_overlays();
        let mut public_overlays = make_overlays();
        let mut color_cache = AnnotationColorCache::new();
        let mut render_cache = OverlayRenderCache::default();
        reapply_overlays_cached(
            &mut cached_controller,
            &mut cached_overlays,
            &mut color_cache,
            &mut render_cache,
        );
        reapply_overlays(
            &mut public_controller,
            &mut public_overlays,
            &mut AnnotationColorCache::new(),
        );

        let assert_labels_match =
            |cached_controller: &GraphController<GenGraph, GenGraphNodeSizer>,
             public_controller: &GraphController<GenGraph, GenGraphNodeSizer>,
             cached_overlays: &[GraphOverlay],
             public_overlays: &[GraphOverlay],
             render_cache: &mut OverlayRenderCache| {
                let mut cached_buffer = Buffer::empty(area);
                let mut public_buffer = Buffer::empty(area);
                let cached_hidden = draw_annotation_labels_cached(
                    &mut cached_buffer,
                    area,
                    cached_controller,
                    cached_overlays,
                    render_cache,
                );
                let public_hidden = draw_annotation_labels(
                    &mut public_buffer,
                    area,
                    public_controller,
                    public_overlays,
                );
                assert_eq!(cached_buffer, public_buffer);
                assert_eq!(cached_hidden, public_hidden);
            };

        assert_labels_match(
            &cached_controller,
            &public_controller,
            &cached_overlays,
            &public_overlays,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 1, &cached_overlays));

        for controller in [&mut cached_controller, &mut public_controller] {
            controller.zoom_out();
            controller.rebuild_viewport_graph().unwrap();
            assert_eq!(controller.zoom_scale(), 2);
        }
        reapply_overlays_cached(
            &mut cached_controller,
            &mut cached_overlays,
            &mut color_cache,
            &mut render_cache,
        );
        reapply_overlays(
            &mut public_controller,
            &mut public_overlays,
            &mut AnnotationColorCache::new(),
        );
        assert_eq!(
            cached_controller.get_cell_highlights(),
            public_controller.get_cell_highlights()
        );
        let expected_start = cached_controller
            .partition_controller
            .node_sizer
            .map_column(&node, 4, VisualDetail::Full);
        let expected_end = cached_controller
            .partition_controller
            .node_sizer
            .map_column(&node, 13, VisualDetail::Full);
        assert_eq!(
            (
                cached_controller.get_cell_highlights()[0].1,
                cached_controller.get_cell_highlights()[0].2,
            ),
            ((expected_start, 0), (expected_end, 0))
        );
        assert_labels_match(
            &cached_controller,
            &public_controller,
            &cached_overlays,
            &public_overlays,
            &mut render_cache,
        );
        assert!(render_cache.matches(VisualDetail::Full, 2, &cached_overlays));

        for controller in [&mut cached_controller, &mut public_controller] {
            controller.viewport_state.camera_current.x += 4;
            controller.viewport_state.camera_target = controller.viewport_state.camera_current;
        }
        assert_labels_match(
            &cached_controller,
            &public_controller,
            &cached_overlays,
            &public_overlays,
            &mut render_cache,
        );

        for controller in [&mut cached_controller, &mut public_controller] {
            controller.trigger_rebuild();
            controller.ensure_camera_coverage().unwrap();
            controller.rebuild_viewport_graph().unwrap();
        }
        let highlights_before_reuse = cached_controller.highlights.clone();
        reapply_overlays_cached(
            &mut cached_controller,
            &mut cached_overlays,
            &mut color_cache,
            &mut render_cache,
        );
        assert_eq!(cached_controller.highlights, highlights_before_reuse);
        assert_eq!(cached_controller.get_cell_highlights().len(), 1);
        assert_labels_match(
            &cached_controller,
            &public_controller,
            &cached_overlays,
            &public_overlays,
            &mut render_cache,
        );

        cached_controller.set_detail_level(VisualDetail::Truncated);
        public_controller.set_detail_level(VisualDetail::Truncated);
        reapply_overlays_cached(
            &mut cached_controller,
            &mut cached_overlays,
            &mut color_cache,
            &mut render_cache,
        );
        reapply_overlays(
            &mut public_controller,
            &mut public_overlays,
            &mut AnnotationColorCache::new(),
        );
        assert_labels_match(
            &cached_controller,
            &public_controller,
            &cached_overlays,
            &public_overlays,
            &mut render_cache,
        );
    }

    fn regions_overlap(a: &CellRegion, b: &CellRegion) -> bool {
        a.0 == b.0 && a.1.0 <= b.2.0 && b.1.0 <= a.2.0
    }

    #[test]
    fn test_draw_annotation_labels_does_not_let_hidden_truncated_spans_cover_visible_spans() {
        let node_a = GraphNode {
            node_id: HashId::convert_str("label-node-a"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let node_b = GraphNode {
            node_id: HashId::convert_str("label-node-b"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let mut graph = GenGraph::new();
        graph.add_edge(node_a, node_b, Vec::new());

        let visible_span = AnnotationSpan {
            id: HashId::convert_str("visible-cross-node"),
            name: "VISIBLE".to_string(),
            segments: vec![
                AnnotationSegment {
                    node_id: node_a.node_id,
                    start: 0,
                    end: 10,
                    strand: Strand::Forward,
                },
                AnnotationSegment {
                    node_id: node_b.node_id,
                    start: 0,
                    end: 10,
                    strand: Strand::Forward,
                },
            ],
        };
        let mut overlays = vec![span_overlay(visible_span)];
        overlays.extend((0..50_000).map(|index| {
            let node = if index % 2 == 0 { node_a } else { node_b };
            span_overlay(AnnotationSpan {
                id: HashId::convert_str(&format!("hidden-{index}")),
                name: format!("hidden-{index}"),
                segments: vec![AnnotationSegment {
                    node_id: node.node_id,
                    start: 0,
                    end: 10,
                    strand: Strand::Reverse,
                }],
            })
        }));

        let area = Rect::new(0, 0, 80, 5);
        let controller = prepare_label_test_controller(graph, VisualDetail::Truncated, area);
        let mut buffer = Buffer::empty(area);
        let any_hidden = draw_annotation_labels(&mut buffer, area, &controller, &overlays);

        assert!(
            any_hidden,
            "partial-node labels should still set the hidden hint"
        );
        assert!(
            buffer.content().iter().any(|cell| cell.symbol() == "V"),
            "the visible cross-node label must not be suppressed by hidden single-node spans"
        );
    }

    #[test]
    fn test_draw_annotation_labels_handles_50k_disjoint_spans_at_full_detail() {
        let node = GraphNode {
            node_id: HashId::convert_str("many-label-node"),
            sequence_start: 0,
            sequence_end: 50_000,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let mut overlays = vec![span_overlay(AnnotationSpan {
            id: HashId::convert_str("visible-label"),
            name: "VISIBLE".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 0,
                end: 1,
                strand: Strand::Forward,
            }],
        })];
        overlays.extend((1..50_000).map(|index| {
            span_overlay(AnnotationSpan {
                id: HashId::convert_str(&format!("disjoint-label-{index}")),
                name: format!("span-{index}"),
                segments: vec![AnnotationSegment {
                    node_id: node.node_id,
                    start: index,
                    end: index + 1,
                    strand: Strand::Forward,
                }],
            })
        }));

        let area = Rect::new(0, 0, 80, 5);
        let controller = prepare_label_test_controller(graph, VisualDetail::Full, area);
        let mut buffer = Buffer::empty(area);
        let _ = draw_annotation_labels(&mut buffer, area, &controller, &overlays);
        assert!(buffer.content().iter().any(|cell| cell.symbol() == "V"));
    }

    #[test]
    fn test_draw_annotation_labels_skips_fully_offscreen_candidates() {
        let node = GraphNode {
            node_id: HashId::convert_str("offscreen-label-node"),
            sequence_start: 0,
            sequence_end: 50_000,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let span = AnnotationSpan {
            id: HashId::convert_str("offscreen"),
            name: "OFFSCREEN".to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 45_000,
                end: 45_010,
                strand: Strand::Forward,
            }],
        };
        let overlays = vec![span_overlay(span)];
        let area = Rect::new(0, 0, 80, 5);
        let controller = prepare_label_test_controller(graph, VisualDetail::Full, area);
        let mut buffer = Buffer::empty(area);

        assert!(!draw_annotation_labels(
            &mut buffer,
            area,
            &controller,
            &overlays
        ));
        assert!(buffer.content().iter().all(|cell| cell.symbol() == " "));
    }

    #[test]
    fn test_color_overlap_index_matches_inclusive_reference() {
        let node_a = GraphNode {
            node_id: HashId::convert_str("color-node-a"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let node_b = GraphNode {
            node_id: HashId::convert_str("color-node-b"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let regions = vec![
            (node_a, (0, 0), (4, 0)),
            (node_a, (4, 0), (8, 0)),
            (node_a, (9, 0), (12, 0)),
            (node_b, (0, 0), (20, 0)),
        ];
        let colors = [Color::Red, Color::Blue, Color::Red, Color::Green];
        let mut index = ColorOverlapIndex::new(&regions);
        let mut occupied = Vec::new();

        for (region, color) in regions.iter().zip(colors) {
            let expected = occupied
                .iter()
                .filter(|(placed, _)| regions_overlap(placed, region))
                .map(|(_, placed_color)| *placed_color)
                .collect::<HashSet<_>>();
            let indexed = index
                .colors_overlapping(region)
                .into_iter()
                .collect::<HashSet<_>>();
            assert_eq!(indexed, expected);
            index.insert_region(region, color);
            occupied.push((*region, color));
        }
    }

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
    fn test_pruned_edit_site_dims_inaccessible_nodes_and_edges() {
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let upstream = GraphNode {
            node_id: HashId::convert_str("upstream"),
            sequence_start: 0,
            sequence_end: 1,
        };
        let inaccessible = GraphNode {
            node_id: HashId::convert_str("inaccessible"),
            sequence_start: 0,
            sequence_end: 1,
        };
        let downstream = GraphNode {
            node_id: HashId::convert_str("downstream"),
            sequence_start: 0,
            sequence_end: 1,
        };
        let end = GraphNode {
            node_id: PATH_END_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let edge = |chromosome_index| {
            vec![GraphEdge {
                edge_id: HashId::convert_str(&format!("edge-{chromosome_index}")),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index,
                phased: 0,
                created_on: 0,
            }]
        };
        let mut graph = GenGraph::new();
        graph.add_edge(start, upstream, edge(NO_CHROMOSOME_INDEX));
        graph.add_edge(
            upstream,
            inaccessible,
            edge(PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
        );
        graph.add_edge(inaccessible, downstream, edge(NO_CHROMOSOME_INDEX));
        graph.add_edge(downstream, end, edge(NO_CHROMOSOME_INDEX));

        let controller = create_gen_graph_controller(graph);
        let dimmed_edges: HashSet<(GraphNode, GraphNode)> =
            controller.get_lowlights().iter().copied().collect();
        let dimmed_nodes: HashSet<GraphNode> =
            controller.get_node_lowlights().iter().copied().collect();

        assert_eq!(
            dimmed_edges,
            HashSet::from([
                (upstream, inaccessible),
                (inaccessible, downstream),
                (downstream, end),
            ]),
            "edges that require the -2 edge from the start should be dimmed"
        );
        assert_eq!(
            dimmed_nodes,
            HashSet::from([inaccessible, downstream, end]),
            "nodes that require the -2 edge from the start should be dimmed"
        );
    }

    #[test]
    fn test_undimmed_controller_retains_pruned_graph_elements() {
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let target = GraphNode {
            node_id: HashId::convert_str("diff-target"),
            sequence_start: 0,
            sequence_end: 1,
        };
        let edge = vec![GraphEdge {
            edge_id: HashId::convert_str("diff-edit-site-edge"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
            phased: 0,
            created_on: 0,
        }];
        let mut graph = GenGraph::new();
        graph.add_edge(start, target, edge);

        let controller = create_gen_graph_controller_without_dimming(graph);

        assert!(
            controller.get_lowlights().is_empty(),
            "undimmed controllers should retain diff context"
        );
        assert!(
            controller.get_node_lowlights().is_empty(),
            "undimmed controllers should not dim diff nodes"
        );
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
                GenGraphNodeSizer::default().get_node_size(&junction, detail_level),
                (1, 1)
            );
        }
        assert!(!GenGraphNodeSizer::default().is_selectable(&junction));
        assert!(GenGraphNodeSizer::default().is_selectable(&start));
    }

    #[test]
    fn test_proportional_zoom_uses_partial_terminal_cell_and_maps_endpoints() {
        let node = GraphNode {
            node_id: HashId::convert_str("scaled-node"),
            sequence_start: 0,
            sequence_end: 27,
        };
        let mut node_sizer = GenGraphNodeSizer::default();

        assert_eq!(node_sizer.get_node_size(&node, VisualDetail::Full), (27, 1));
        assert!(node_sizer.zoom_out_scale());
        assert_eq!(node_sizer.zoom_scale(), 2);
        assert_eq!(node_sizer.get_node_size(&node, VisualDetail::Full), (14, 1));
        assert_eq!(node_sizer.map_column(&node, 0, VisualDetail::Full), 0);
        assert_eq!(node_sizer.map_column(&node, 26, VisualDetail::Full), 13);
        assert_eq!(node_sizer.map_column(&node, 27, VisualDetail::Full), 14);
        assert_eq!(
            node_sizer.get_node_size(&node, VisualDetail::Truncated),
            (13, 1)
        );
        assert!(node_sizer.zoom_in_scale());
        assert_eq!(node_sizer.zoom_scale(), 1);
        assert_eq!(node_sizer.get_node_size(&node, VisualDetail::Full), (27, 1));
    }

    #[test]
    fn test_full_detail_zoom_scales_proportionally_and_round_trips_cursor_anchor() {
        let node = GraphNode {
            node_id: HashId::convert_str("proportional-zoom-node"),
            sequence_start: 0,
            sequence_end: 100,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let mut controller = create_gen_graph_controller(graph);
        controller.set_detail_level(VisualDetail::Full);
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 40, 12);
        controller.ensure_camera_coverage().unwrap();
        controller.rebuild_viewport_graph().unwrap();
        let node_index = petgraph::graph::NodeIndex::new(<GenGraph as NodeIndexable>::to_index(
            controller.graph(),
            node,
        ));
        controller.go_to_node(node_index, (0.63, 0.5));
        controller.rebuild_viewport_graph().unwrap();
        let cursor_viewport = controller.cursor.viewport_pos();
        let cursor_fraction = controller.cursor.fractional_pos();

        controller.zoom_out();
        controller.rebuild_viewport_graph().unwrap();
        assert_eq!(controller.get_detail_level(), VisualDetail::Full);
        assert_eq!(controller.zoom_scale(), 2);
        assert_eq!(
            controller
                .partition_controller
                .node_sizer
                .get_node_size(&node, VisualDetail::Full)
                .0,
            50
        );
        assert_eq!(controller.cursor.viewport_pos(), cursor_viewport);
        assert_eq!(controller.cursor.fractional_pos(), cursor_fraction);

        for (expected_scale, expected_width) in [(4, 25), (8, 13)] {
            controller.zoom_out();
            controller.rebuild_viewport_graph().unwrap();
            assert_eq!(controller.zoom_scale(), expected_scale);
            assert_eq!(
                controller
                    .partition_controller
                    .node_sizer
                    .get_node_size(&node, VisualDetail::Full)
                    .0,
                expected_width
            );
            assert_eq!(controller.cursor.viewport_pos(), cursor_viewport);
            assert_eq!(controller.cursor.fractional_pos(), cursor_fraction);
        }

        controller.zoom_out();
        assert_eq!(controller.get_detail_level(), VisualDetail::Truncated);
        assert_eq!(controller.zoom_scale(), 8);
        assert_eq!(
            controller
                .partition_controller
                .node_sizer
                .get_node_size(&node, VisualDetail::Truncated)
                .0,
            13
        );

        controller.zoom_in();
        assert_eq!(controller.get_detail_level(), VisualDetail::Full);
        assert_eq!(controller.zoom_scale(), 8);
        controller.zoom_in();
        controller.rebuild_viewport_graph().unwrap();
        assert_eq!(controller.zoom_scale(), 4);
        assert_eq!(
            controller
                .partition_controller
                .node_sizer
                .get_node_size(&node, VisualDetail::Full)
                .0,
            25
        );
        controller.zoom_in();
        controller.zoom_in();
        controller.rebuild_viewport_graph().unwrap();
        assert_eq!(controller.zoom_scale(), 1);
        assert_eq!(
            controller
                .partition_controller
                .node_sizer
                .get_node_size(&node, VisualDetail::Full)
                .0,
            100
        );
        assert_eq!(controller.cursor.viewport_pos(), cursor_viewport);
        assert_eq!(controller.cursor.fractional_pos(), cursor_fraction);
    }

    #[test]
    fn test_scaled_full_renderer_draws_aggregate_cells_without_sequence_lookup() {
        let context = setup_gen_on_disk();
        let node = GraphNode {
            node_id: HashId::convert_str("aggregate-renderer-node"),
            sequence_start: 0,
            sequence_end: 27,
        };
        let mut viewport_state = ViewportState::new();
        viewport_state.viewport_bounds = Rect::new(0, 0, 3, 1);
        viewport_state.camera_current = WorldPos::new(1, 0);
        viewport_state.camera_target = viewport_state.camera_current;
        let mut buffer = Buffer::empty(viewport_state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);
        let mut renderer = GenGraphNodeRenderer::new(context.graph().conn(), context.workspace());
        renderer.set_zoom_scale(2);

        renderer.render_node(
            &mut world_buffer,
            WorldRect::from_corners(WorldPos::new(0, 0), WorldPos::new(13, 0)),
            &node,
            VisualDetail::Full,
        );

        assert_eq!(world_buffer.get_char(WorldPos::new(0, 0)), Some('·'));
        assert_eq!(world_buffer.get_char(WorldPos::new(1, 0)), Some('·'));
        assert_eq!(world_buffer.get_char(WorldPos::new(2, 0)), Some('·'));
        assert!(
            renderer.cache.is_empty(),
            "should not fetch hidden sequence bases"
        );
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
        let mut renderer = GenGraphNodeRenderer::new(context.graph().conn(), context.workspace());

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
    fn test_zero_width_junction_preserves_edge_highlight_glyph_and_color() {
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
        let position = WorldPos::ZERO;
        world_buffer.set_char_styled(position, '━', Style::default().fg(Color::Yellow));
        let mut renderer = GenGraphNodeRenderer::new(context.graph().conn(), context.workspace());

        renderer.render_node(
            &mut world_buffer,
            WorldRect::from_center_and_size(position, (1, 1)),
            &junction,
            VisualDetail::Full,
        );

        let (character, style) = world_buffer
            .get_char_styled(position)
            .expect("should render the zero-width junction");
        assert_eq!(character, '━');
        assert_eq!(style.fg, Some(Color::Yellow));
    }

    #[test]
    fn test_full_renderer_clips_large_sequence_to_visible_node_slice() {
        const NODE_START: i64 = 50;
        const SEQUENCE_LENGTH: usize = 1_000_000;
        const VIEWPORT_WIDTH: u16 = 5;

        let context = setup_gen_on_disk();
        let sequence_text = "ACGT".repeat(SEQUENCE_LENGTH / 4);
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(&sequence_text)
            .save(context.graph().conn())
            .unwrap();
        let node_id = Node::create(
            context.graph().conn(),
            &sequence.hash,
            &HashId::convert_str(&format!("large-renderer-node.{}", sequence.hash)),
        )
        .unwrap();
        let graph_node = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: SEQUENCE_LENGTH as i64,
        };
        let node_area = WorldRect::from_corners(
            WorldPos::new(NODE_START, 0),
            WorldPos::new(NODE_START + SEQUENCE_LENGTH as i64 - 1, 0),
        );
        let mut renderer = GenGraphNodeRenderer::new(context.graph().conn(), context.workspace());

        let visible_offset = 123_457i64;
        let visible_start = NODE_START + visible_offset;
        let mut viewport_state = ViewportState::new();
        viewport_state.viewport_bounds = Rect::new(0, 0, VIEWPORT_WIDTH, 1);
        viewport_state.camera_current =
            WorldPos::new(visible_start + i64::from(VIEWPORT_WIDTH / 2), 0);
        viewport_state.camera_target = viewport_state.camera_current;
        let mut buffer = Buffer::empty(viewport_state.viewport_bounds);
        {
            let mut world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);
            renderer.render_node(
                &mut world_buffer,
                node_area,
                &graph_node,
                VisualDetail::Full,
            );

            for offset in 0..i64::from(VIEWPORT_WIDTH) {
                let expected = sequence_text.as_bytes()[(visible_offset + offset) as usize] as char;
                assert_eq!(
                    world_buffer.get_char(WorldPos::new(visible_start + offset, 0)),
                    Some(expected),
                    "should render the visible world base at its matching sequence coordinate"
                );
            }
        }

        let mut left_boundary_state = ViewportState::new();
        left_boundary_state.viewport_bounds = Rect::new(0, 0, VIEWPORT_WIDTH, 1);
        left_boundary_state.camera_current = WorldPos::new(NODE_START, 0);
        left_boundary_state.camera_target = left_boundary_state.camera_current;
        let mut left_boundary_buffer = Buffer::empty(left_boundary_state.viewport_bounds);
        {
            let mut world_buffer =
                WorldBuffer::new(&mut left_boundary_buffer, &left_boundary_state);
            renderer.render_node(
                &mut world_buffer,
                node_area,
                &graph_node,
                VisualDetail::Full,
            );

            assert_eq!(
                world_buffer.get_char(WorldPos::new(NODE_START - 2, 0)),
                Some(' ')
            );
            assert_eq!(
                world_buffer.get_char(WorldPos::new(NODE_START - 1, 0)),
                Some(' ')
            );
            for offset in 0..3i64 {
                assert_eq!(
                    world_buffer.get_char(WorldPos::new(NODE_START + offset, 0)),
                    Some(sequence_text.as_bytes()[offset as usize] as char)
                );
            }
        }

        let node_end = NODE_START + SEQUENCE_LENGTH as i64 - 1;
        let mut right_boundary_state = ViewportState::new();
        right_boundary_state.viewport_bounds = Rect::new(0, 0, VIEWPORT_WIDTH, 1);
        right_boundary_state.camera_current = WorldPos::new(node_end, 0);
        right_boundary_state.camera_target = right_boundary_state.camera_current;
        let mut right_boundary_buffer = Buffer::empty(right_boundary_state.viewport_bounds);
        {
            let mut world_buffer =
                WorldBuffer::new(&mut right_boundary_buffer, &right_boundary_state);
            renderer.render_node(
                &mut world_buffer,
                node_area,
                &graph_node,
                VisualDetail::Full,
            );

            let last_visible_start = node_end - 2;
            for offset in 0..3i64 {
                let sequence_offset = SEQUENCE_LENGTH as i64 - 3 + offset;
                assert_eq!(
                    world_buffer.get_char(WorldPos::new(last_visible_start + offset, 0)),
                    Some(sequence_text.as_bytes()[sequence_offset as usize] as char)
                );
            }
            assert_eq!(
                world_buffer.get_char(WorldPos::new(node_end + 1, 0)),
                Some(' ')
            );
            assert_eq!(
                world_buffer.get_char(WorldPos::new(node_end + 2, 0)),
                Some(' ')
            );
        }

        let mut offscreen_renderer =
            GenGraphNodeRenderer::new(context.graph().conn(), context.workspace());
        let mut offscreen_state = ViewportState::new();
        offscreen_state.viewport_bounds = Rect::new(0, 0, VIEWPORT_WIDTH, 1);
        offscreen_state.camera_current = WorldPos::new(node_end + 10, 0);
        offscreen_state.camera_target = offscreen_state.camera_current;
        let mut offscreen_buffer = Buffer::empty(offscreen_state.viewport_bounds);
        {
            let mut world_buffer = WorldBuffer::new(&mut offscreen_buffer, &offscreen_state);
            offscreen_renderer.render_node(
                &mut world_buffer,
                node_area,
                &graph_node,
                VisualDetail::Full,
            );
            assert!(offscreen_renderer.cache.is_empty());
            let first_visible_x = node_end + 8;
            for offset in 0..i64::from(VIEWPORT_WIDTH) {
                assert_eq!(
                    world_buffer.get_char(WorldPos::new(first_visible_x + offset, 0)),
                    Some(' ')
                );
            }
        }
    }

    #[test]
    fn test_full_renderer_reads_visible_range_from_indexed_external_sequence() {
        const NODE_WORLD_START: i64 = 100;
        const NODE_SEQUENCE_START: i64 = 5;
        const NODE_SEQUENCE_END: i64 = 29;
        const VISIBLE_OFFSET: i64 = 7;
        const VIEWPORT_WIDTH: u16 = 5;
        const SEQUENCE_TEXT: &str = "ATCGATCGATCGATCGATCGGGAACACACAGAGA";

        let context = setup_gen_on_disk();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple.fa")
            .to_str()
            .expect("should have a UTF-8 fixture path")
            .to_string();
        import_fasta(
            &context,
            &fasta_path,
            "renderer-range",
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .expect("should import the indexed external FASTA fixture");

        let graph = Sample::get_graph(
            context.graph().conn(),
            context.workspace(),
            "renderer-range",
            Sample::DEFAULT_NAME,
            None,
        )
        .expect("should load the fixture graph");
        let graph_node = graph
            .nodes()
            .find(|node| !is_start_node(node.node_id) && !is_end_node(node.node_id))
            .expect("should contain a sequence node");
        let graph_node = GraphNode {
            sequence_start: NODE_SEQUENCE_START,
            sequence_end: NODE_SEQUENCE_END,
            ..graph_node
        };
        let sequence = Node::get_sequences_by_node_ids(
            context.graph().conn(),
            context.workspace(),
            &[graph_node.node_id],
            None,
        )
        .remove(&graph_node.node_id)
        .expect("should load the external sequence for the graph node");
        assert!(sequence.external_sequence);
        assert!(sequence.asset_ref_id.is_some());

        let node_area = WorldRect::from_corners(
            WorldPos::new(NODE_WORLD_START, 0),
            WorldPos::new(NODE_WORLD_START + graph_node.length() - 1, 0),
        );
        let visible_start = NODE_WORLD_START + VISIBLE_OFFSET;
        let mut viewport_state = ViewportState::new();
        viewport_state.viewport_bounds = Rect::new(0, 0, VIEWPORT_WIDTH, 1);
        viewport_state.camera_current =
            WorldPos::new(visible_start + i64::from(VIEWPORT_WIDTH / 2), 0);
        viewport_state.camera_target = viewport_state.camera_current;
        let mut buffer = Buffer::empty(viewport_state.viewport_bounds);
        let mut renderer = GenGraphNodeRenderer::new(context.graph().conn(), context.workspace());
        {
            let mut world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);
            renderer.render_node(
                &mut world_buffer,
                node_area,
                &graph_node,
                VisualDetail::Full,
            );

            let expected_start = usize::try_from(NODE_SEQUENCE_START + VISIBLE_OFFSET)
                .expect("should have a positive sequence offset");
            for offset in 0..i64::from(VIEWPORT_WIDTH) {
                let sequence_offset = expected_start + usize::try_from(offset).unwrap();
                assert_eq!(
                    world_buffer.get_char(WorldPos::new(visible_start + offset, 0)),
                    Some(SEQUENCE_TEXT.as_bytes()[sequence_offset] as char),
                    "should render the visible external sequence at its genomic coordinate"
                );
            }
        }

        assert_eq!(renderer.cache.len(), 1);
        let ((cached_node, cached_start, cached_end), cached_sequence) = renderer
            .cache
            .iter()
            .next()
            .expect("should cache the visible range");
        assert_eq!(*cached_node, graph_node);
        assert_eq!(*cached_start, NODE_SEQUENCE_START + VISIBLE_OFFSET);
        assert_eq!(
            *cached_end,
            NODE_SEQUENCE_START + VISIBLE_OFFSET + i64::from(VIEWPORT_WIDTH)
        );
        assert_eq!(cached_sequence.len(), usize::from(VIEWPORT_WIDTH));

        assert_eq!(
            renderer
                .get_sequence(&graph_node)
                .expect("should preserve the public full-node sequence API"),
            &SEQUENCE_TEXT[NODE_SEQUENCE_START as usize..NODE_SEQUENCE_END as usize]
        );
    }

    #[test]
    fn test_full_renderer_clamps_unknown_sequence_to_visible_fallback_bases() {
        const NODE_WORLD_START: i64 = 100;
        const NODE_LENGTH: i64 = 40;
        const VIEWPORT_WIDTH: u16 = 5;

        let context = setup_gen_on_disk();
        let missing_asset_id = HashId::convert_str("missing-renderer-sequence-asset");
        AssetRef::create(
            context.graph().conn(),
            &AssetRef {
                id: missing_asset_id,
                uri: "missing-renderer-sequence.fa".to_string(),
                file_type: "fasta".to_string(),
                checksum: None,
                size: None,
                role: AssetRole::Input,
                logical_path: Some("missing-renderer-sequence.fa".to_string()),
                name: None,
                created_on: 0,
                upstream_asset_ref_id: None,
                materialized_checksum: None,
            },
        )
        .expect("should create the intentionally unresolved sequence asset");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .name("missing-sequence")
            .asset_ref_id(Some(&missing_asset_id))
            .length(NODE_LENGTH)
            .save(context.graph().conn())
            .expect("should save an external sequence with a missing asset reference");
        let node_id = Node::create(
            context.graph().conn(),
            &sequence.hash,
            &HashId::convert_str("unknown-renderer-sequence-node"),
        )
        .expect("should create a graph node for the missing sequence");
        let graph_node = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: NODE_LENGTH,
        };
        let node_area = WorldRect::from_corners(
            WorldPos::new(NODE_WORLD_START, 0),
            WorldPos::new(NODE_WORLD_START + NODE_LENGTH - 1, 0),
        );
        let mut renderer = GenGraphNodeRenderer::new(context.graph().conn(), context.workspace());

        for (visible_offset, expected) in [(10i64, "quenc"), (12, "ence "), (16, "     ")] {
            let visible_start = NODE_WORLD_START + visible_offset;
            let mut viewport_state = ViewportState::new();
            viewport_state.viewport_bounds = Rect::new(0, 0, VIEWPORT_WIDTH, 1);
            viewport_state.camera_current =
                WorldPos::new(visible_start + i64::from(VIEWPORT_WIDTH / 2), 0);
            viewport_state.camera_target = viewport_state.camera_current;
            let mut buffer = Buffer::empty(viewport_state.viewport_bounds);
            let mut world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);
            renderer.render_node(
                &mut world_buffer,
                node_area,
                &graph_node,
                VisualDetail::Full,
            );

            for (offset, expected_base) in expected.chars().enumerate() {
                assert_eq!(
                    world_buffer.get_char(WorldPos::new(
                        visible_start + i64::try_from(offset).unwrap(),
                        0,
                    )),
                    Some(expected_base),
                    "should preserve available fallback bases at the correct visible offsets"
                );
            }
        }
    }

    #[test]
    fn test_highlight_locus_colors_edges_across_zero_width_junctions() {
        let node_a = GraphNode {
            node_id: HashId::convert_str("a"),
            sequence_start: 0,
            sequence_end: 1,
        };
        let first_junction = GraphNode {
            node_id: HashId::convert_str("first-junction"),
            sequence_start: 1,
            sequence_end: 1,
        };
        let node_t = GraphNode {
            node_id: HashId::convert_str("t"),
            sequence_start: 0,
            sequence_end: 1,
        };
        let second_junction = GraphNode {
            node_id: HashId::convert_str("second-junction"),
            sequence_start: 1,
            sequence_end: 1,
        };
        let node_gataa = GraphNode {
            node_id: HashId::convert_str("gataa"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let mut graph = GenGraph::new();
        graph.add_edge(node_a, first_junction, Vec::new());
        graph.add_edge(first_junction, node_t, Vec::new());
        graph.add_edge(node_t, second_junction, Vec::new());
        graph.add_edge(second_junction, node_gataa, Vec::new());
        let mut controller = create_gen_graph_controller(graph);
        let locus = GraphLocus {
            slices: vec![
                GraphNodeSlice::full(node_a, Strand::Forward),
                GraphNodeSlice::full(node_t, Strand::Forward),
                GraphNodeSlice::full(node_gataa, Strand::Forward),
            ],
        };

        highlight_locus(&mut controller, &locus, PathStyle::new(Color::Yellow));

        let highlighted_edges: Vec<(GraphNode, GraphNode)> = controller
            .highlights
            .iter()
            .filter_map(|(kind, _)| match kind {
                HighlightKind::Edge(source, target) => Some((*source, *target)),
                _ => None,
            })
            .collect();
        assert_eq!(
            highlighted_edges,
            vec![(node_a, node_t), (node_t, node_gataa),]
        );
    }

    #[test]
    fn test_display_projection_contracts_positions_and_preserves_pruned_dimming() {
        let terminal = |node_id| GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 0,
        };
        let start = terminal(PATH_START_NODE_ID);
        let upstream = midpoint_node(1, 0, 1);
        let junction = midpoint_node(2, 1, 1);
        let downstream = midpoint_node(3, 0, 1);
        let end = terminal(PATH_END_NODE_ID);
        let edge = |edge_id: &str, chromosome_index| {
            vec![GraphEdge {
                edge_id: HashId::convert_str(edge_id),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index,
                phased: 0,
                created_on: 0,
            }]
        };
        let mut graph = GenGraph::new();
        graph.add_edge(
            start,
            upstream,
            edge("projection-start", NO_CHROMOSOME_INDEX),
        );
        graph.add_edge(
            upstream,
            junction,
            edge("projection-marker", PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
        );
        graph.add_edge(junction, downstream, Vec::new());
        graph.add_edge(downstream, end, edge("projection-end", NO_CHROMOSOME_INDEX));

        let (display_graph, dimmed_edges, dimmed_nodes) = display_graph_with_lowlights(&graph);

        assert!(graph.contains_node(junction));
        assert!(!display_graph.contains_node(junction));
        assert!(display_graph.edge_weight(upstream, downstream).is_some());
        assert!(dimmed_edges.contains(&(upstream, downstream)));
        assert!(dimmed_nodes.contains(&downstream));
        assert!(dimmed_nodes.contains(&end));
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
            None,
            None,
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

        let gen_graph =
            Sample::get_graph(conn, context.workspace(), collection, "SAMPLE1", None).unwrap();
        let mut controller = create_gen_graph_controller(gen_graph);

        let mut terminal = create_test_terminal(120, 30);
        terminal
            .draw(|f| {
                let area = f.area();
                controller.viewport_state.viewport_bounds = area;
                controller.ensure_camera_coverage().unwrap();
                controller.rebuild_viewport_graph().unwrap();
                GraphWidget::with_renderer(GenGraphNodeRenderer::new(conn, context.workspace()))
                    .render(area, f.buffer_mut(), &mut controller);
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
            None,
            None,
        )
        .unwrap();

        let gen_graph = Sample::get_graph(
            conn,
            context.workspace(),
            collection,
            Sample::DEFAULT_NAME,
            None,
        )
        .unwrap();
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

    #[test]
    fn test_reapply_overlays_keeps_partial_adhoc_spans_at_truncated_detail() {
        let node = GraphNode {
            node_id: HashId::convert_str("block"),
            sequence_start: 0,
            sequence_end: 20,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let mut controller = create_gen_graph_controller(graph);
        controller.set_detail_level(VisualDetail::Truncated);

        let partial_span = |name: &str| AnnotationSpan {
            id: HashId::convert_str(name),
            name: name.to_string(),
            segments: vec![AnnotationSegment {
                node_id: node.node_id,
                start: 2,
                end: 5,
                strand: Strand::Forward,
            }],
        };
        let mut overlays = vec![
            GraphOverlay {
                content: OverlayContent::Span(partial_span("track")),
                source: OverlaySource::Track("file:1".to_string()),
                style: PathStyle::new(Color::Cyan),
            },
            GraphOverlay {
                content: OverlayContent::Span(partial_span("adhoc")),
                source: OverlaySource::Adhoc,
                style: PathStyle::new(Color::Yellow),
            },
        ];

        let mut color_cache = AnnotationColorCache::new();
        reapply_overlays(&mut controller, &mut overlays, &mut color_cache);

        let cell_highlights = controller
            .highlights
            .iter()
            .filter(|(kind, _)| matches!(kind, HighlightKind::Cells { .. }))
            .count();
        assert_eq!(
            cell_highlights, 1,
            "partial ad-hoc search spans should remain visible at truncated detail"
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

    #[test]
    fn test_reapply_overlays_handles_50k_non_overlapping_spans() {
        let node = GraphNode {
            node_id: HashId::convert_str("many-overlay-node"),
            sequence_start: 0,
            sequence_end: 50_000,
        };
        let mut graph = GenGraph::new();
        graph.add_node(node);
        let mut controller = create_gen_graph_controller(graph);
        controller.set_detail_level(VisualDetail::Full);
        let mut overlays = (0..50_000)
            .map(|index| {
                span_overlay(AnnotationSpan {
                    id: HashId::convert_str(&format!("non-overlap-{index}")),
                    name: format!("span-{index}"),
                    segments: vec![AnnotationSegment {
                        node_id: node.node_id,
                        start: index,
                        end: index + 1,
                        strand: Strand::Forward,
                    }],
                })
            })
            .collect::<Vec<_>>();
        let mut color_cache = AnnotationColorCache::new();

        reapply_overlays(&mut controller, &mut overlays, &mut color_cache);

        assert_eq!(
            controller
                .highlights
                .iter()
                .filter(|(kind, _)| matches!(kind, HighlightKind::Cells { .. }))
                .count(),
            overlays.len()
        );
    }
}
