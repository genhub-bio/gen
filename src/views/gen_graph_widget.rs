use std::collections::HashMap;

use gen_core::{is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{db::GraphConnection, node::Node};
use gen_tui::{
    geometry::WorldRect,
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::{GraphWidget, NODE_GLYPH},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer},
    theme::current_theme,
};
use ratatui::style::Style;

/// Labels for special start/end nodes
pub mod label {
    pub const START: &str = "Start >";
    pub const END: &str = "> End";
}

/// Domain-specific node sizer for GenGraph that calculates visual dimensions
/// based on genomic sequence length.
pub struct GenGraphNodeSizer;

impl NodeSizer<GenGraph> for GenGraphNodeSizer {
    /// Calculate how much screen space a GenGraph node needs based on sequence length and level of detail
    fn get_node_size(&self, node: &GraphNode, detail_level: VisualDetail) -> (u64, u64) {
        // Handle special start/end nodes with fixed label sizes (always show full label)
        if is_start_node(node.node_id) {
            return (label::START.len() as u64, 1u64);
        }
        if is_end_node(node.node_id) {
            return (label::END.len() as u64, 1u64);
        }

        let sequence_length = (node.sequence_end - node.sequence_start) as u64;
        match detail_level {
            VisualDetail::Minimal => (1u64, 1u64), // Just a glyph
            VisualDetail::Truncated => (sequence_length.min(12), 1u64), // Truncated to max 12 chars
            VisualDetail::Full => (sequence_length, 1u64), // Full sequence length
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
    pub fn get_sequence(&mut self, node_key: &GraphNode) -> String {
        // Check cache first
        if let Some(cached) = self.cache.get(node_key) {
            return cached.clone();
        }

        // Cache miss - query database
        let (db_node_id, start, end) = (
            node_key.node_id,
            node_key.sequence_start,
            node_key.sequence_end,
        );
        let sequences = Node::get_sequences_by_node_ids(self.conn, &[db_node_id]);
        let sequence = match sequences.get(&db_node_id) {
            Some(seq) => seq.get_sequence(start, end),
            None => "?".repeat((end - start).max(0) as usize),
        };

        self.cache.insert(*node_key, sequence.clone());
        sequence
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
                let text_style = Style::default().fg(theme[0x00]).bg(theme[0x05]);
                buffer.set_string_styled(area.left_center(), &NODE_GLYPH.to_string(), text_style);
            }
            VisualDetail::Truncated => {
                // Truncated scale: Show sequence with truncation to max 12 chars
                let sequence = self.get_sequence(node_id);
                let max_width = 12u32;
                let truncated = inner_truncation(&sequence, max_width);
                buffer.set_string_styled(area.left_center(), &truncated, text_style);
            }
            VisualDetail::Full => {
                // Full scale: Show complete sequence (truncated only by area width)
                let sequence = self.get_sequence(node_id);
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
