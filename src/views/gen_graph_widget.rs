use std::collections::HashMap;

use gen_graph::{GenGraph, GraphNode};
use gen_models::{db::GraphConnection, node::Node};
use gen_widget::{
    geometry::WorldRect,
    graph_controller::WorldBuffer,
    graph_widget::{GraphWidget, NODE_GLYPH},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer},
    theme::get_theme_color,
};
use ratatui::style::Style;

/// Domain-specific node sizer for GenGraph that calculates visual dimensions
/// based on genomic sequence length.
pub struct GenGraphNodeSizer;

impl NodeSizer<&GenGraph> for GenGraphNodeSizer {
    /// Calculate how much screen space a GenGraph node needs based on sequence length and level of detail
    fn get_node_size(&self, node: &GraphNode, detail_level: VisualDetail) -> (u64, u64) {
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

impl NodeRenderer<&GenGraph> for GenGraphNodeRenderer<'_> {
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
        let background_style = Style::default().bg(get_theme_color("node").unwrap_or_default());
        let text_style = Style::default()
            .bg(get_theme_color("node").unwrap_or(ratatui::style::Color::Blue))
            .fg(get_theme_color("text").unwrap_or(ratatui::style::Color::White));

        buffer.fill_rect(area, ' ');
        buffer.set_char_styled(area.left_center(), ' ', background_style);

        match detail_level {
            VisualDetail::Minimal => {
                // Base scale: Just show a simple glyph
                let text_style = Style::default()
                    .fg(get_theme_color("text").unwrap_or(ratatui::style::Color::White))
                    .bg(get_theme_color("canvas").unwrap_or(ratatui::style::Color::Blue));
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
) -> GraphWidget<'_, &GenGraph, GenGraphNodeSizer, GenGraphNodeRenderer<'_>> {
    let renderer = GenGraphNodeRenderer::new(conn);
    GraphWidget::with_renderer(renderer)
}

#[cfg(test)]
mod tests {
    use super::*;

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
