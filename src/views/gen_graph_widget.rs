use std::collections::HashMap;

use gen_core::{is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{db::GraphConnection, node::Node, sequence::SequenceError};
use gen_tui::{
    geometry::{WorldPos, WorldRect},
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::{GraphWidget, NODE_GLYPH},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer, PathStyle},
    theme::current_theme,
};
use petgraph::{graph::NodeIndex, visit::NodeIndexable};
use ratatui::style::Style;

use crate::graphs::graph_search::GraphLocus;

/// Labels for special start/end nodes
pub mod label {
    pub const START: &str = "╟";
    pub const END: &str = "╢";
}

/// Domain-specific node sizer for GenGraph that calculates visual dimensions
/// based on genomic sequence length.
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

        let sequence_length = (node.sequence_end - node.sequence_start) as u64;
        match detail_level {
            VisualDetail::Minimal => (1u64, 1u64), // Just a glyph
            VisualDetail::Truncated => (sequence_length.min(13), 1u64), // 13 = 5 border + 3 mid + 5 border
            VisualDetail::Full => (sequence_length, 1u64),              // Full sequence length
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
        let sequences = Node::get_sequences_by_node_ids(self.conn, &[db_node_id]);
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
    let node_sizer = GenGraphNodeSizer;
    let mut controller = GraphController::new(graph, node_sizer);
    controller.set_detail_level(VisualDetail::Truncated);
    controller.hide_cursor();
    controller
}

/// Position the cursor on a specific graph node at a given fractional offset.
///
/// The caller must supply the full `GraphNode` key (node_id + sequence_start +
/// sequence_end), because multiple graph nodes can share the same underlying
/// `node_id` when they represent different sub-ranges of the same sequence.
/// The offset `(0.5, 0.5)` centers on the middle of the node.
///
/// Camera positioning is handled by the cursor-anchored rebuild system: the
/// caller is responsible for setting the cursor's viewport position to the
/// desired screen location before the next render (e.g. screen center to
/// center on the node), then showing the cursor so camera-following engages.
///
/// # Arguments
/// * `controller` — the graph controller to position
/// * `node`       — the exact `GraphNode` key to center on
/// * `offset`     — fractional `(x, y)` position within the node (0.0–1.0)
pub fn center_on_node_offset<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    node: GraphNode,
    offset: (f64, f64),
) {
    // Find which partition owns this node.
    let (partition_idx, _) = match controller
        .partition_controller
        .partition_table
        .find_node(&node)
    {
        Ok(p) => p,
        Err(_) => return,
    };

    // Ensure the partition is loaded and set it as the anchor.
    let _ = controller.ensure_partition_loaded(partition_idx);
    let _ = controller.set_anchor_partition(partition_idx);

    // Resolve the domain NodeIndex from the node key.
    let domain_idx = NodeIndex::new(NodeIndexable::to_index(controller.graph(), node));

    // Delegate to the controller's go_to method, which sets the cursor,
    // switches to fine mode, shows it, and queues the one-shot centering.
    controller.go_to_node(domain_idx, offset);
}

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
/// Column offsets are clamped with `clamp_col` so they map correctly in every
/// detail level (e.g. truncated nodes collapse interior columns to the `...` cell).
///
/// Returns `None` if no block in the locus is present in `pos_map` (all off-screen).
pub fn locus_label_bounds(
    locus: &GraphLocus,
    pos_map: &HashMap<GraphNode, (WorldPos, (u64, u64))>,
    detail_level: VisualDetail,
) -> Option<(WorldPos, WorldPos)> {
    let block_world_pos = |block: GraphNode, col_raw: i64| -> Option<WorldPos> {
        let &(center, size) = pos_map.get(&block)?;
        let rect = WorldRect::from_center_and_size(center, size);
        let col = clamp_col(col_raw, block.length(), detail_level);
        Some(WorldPos::new(rect.min.x + col, center.y))
    };

    let last = locus.blocks.len() - 1;

    let left_pos = locus.blocks.iter().enumerate().find_map(|(i, &block)| {
        let col_raw = if i == 0 { locus.start_offset as i64 } else { 0 };
        block_world_pos(block, col_raw)
    })?;

    let right_pos = locus
        .blocks
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, &block)| {
            let col_raw = if i == last {
                locus.end_offset.saturating_sub(1) as i64
            } else {
                block.length() - 1
            };
            block_world_pos(block, col_raw)
        })?;

    let mut y_min = i64::MAX;
    let mut y_max = i64::MIN;
    for &block in &locus.blocks {
        let Some(&(center, _)) = pos_map.get(&block) else {
            continue;
        };
        y_min = y_min.min(center.y);
        y_max = y_max.max(center.y);
    }

    if y_min == i64::MAX {
        return None;
    }

    let left_pos = WorldPos::new(left_pos.x, y_max);
    let right_pos = WorldPos::new(right_pos.x, y_min);

    Some((left_pos, right_pos))
}

/// Apply detail-level clamping to a raw column offset.
///
/// In `Truncated` mode, interior columns of long nodes map to the `...` region.
fn clamp_col(col_raw: i64, block_seq_len: i64, detail_level: VisualDetail) -> i64 {
    match detail_level {
        VisualDetail::Minimal => 0,
        VisualDetail::Truncated if block_seq_len > 13 => {
            clamp_truncated_col(col_raw, block_seq_len)
        }
        _ => col_raw,
    }
}

/// Map a sequence offset to a visual cell column inside a 13-cell truncated node.
///
/// Display layout: `AAAAA...BBBBB` — first 5 bases (cells 0-4), `...` (cells 5-7),
/// last 5 bases (cells 8-12). Interior positions that fall in the `...` region clamp
/// to cell 6 (the centre dot).
fn clamp_truncated_col(value: i64, block_seq_len: i64) -> i64 {
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
/// In `Truncated` detail level the column offsets are clamped so that interior
/// positions map to the `...` cell rather than a precise (wrong) location.
pub fn highlight_match_range<S: NodeSizer<GenGraph>>(
    controller: &mut GraphController<GenGraph, S>,
    m: &GraphLocus,
    style: PathStyle,
) {
    let detail_level = controller.get_detail_level();

    let path_len = m.blocks.len();
    for (i, &node) in m.blocks.iter().enumerate() {
        let block_seq_len = node.length();
        let col_start_raw = if i == 0 { m.start_offset as i64 } else { 0 };
        let col_end_raw = if i == path_len - 1 {
            m.end_offset.saturating_sub(1) as i64
        } else {
            block_seq_len - 1
        };
        let (col_start, col_end) = (
            clamp_col(col_start_raw, block_seq_len, detail_level),
            clamp_col(col_end_raw, block_seq_len, detail_level),
        );
        controller.set_cell_highlight(node, (col_start, 0), (col_end, 0), style);
    }

    for (&src, &dst) in m.blocks.iter().zip(m.blocks.iter().skip(1)) {
        controller.set_edge_highlight((src, dst), style);
    }
}

#[cfg(test)]
mod tests {
    use gen_tui::{
        geometry::WorldPos,
        viewport_state::{ViewportState, WorldBuffer},
    };
    use ratatui::backend::TestBackend;

    use super::*;

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
}
