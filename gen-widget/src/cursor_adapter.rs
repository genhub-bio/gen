use log::trace;
use petgraph::graph::NodeIndex;

use crate::{
    cursor_v2::ViewportCursor,
    geometry::{ViewportPos, WorldPos},
    graph_controller::GraphController,
    layout::{LayoutNode, NodeRole},
    plotter::NodeSizer,
};

/// Cursor-related methods for GraphController that adapt the new ViewportCursor system
impl<G, S> GraphController<G, S>
where
    G: petgraph::visit::GraphBase
        + Clone
        + petgraph::visit::EdgeIndexable
        + petgraph::visit::NodeIndexable
        + petgraph::visit::NodeCount
        + petgraph::visit::Visitable
        + petgraph::visit::IntoNodeIdentifiers
        + petgraph::visit::IntoEdgeReferences
        + petgraph::visit::IntoNeighborsDirected,
    G::NodeId: Copy + Eq + std::hash::Hash + Ord,
    G::EdgeId: Clone,
    for<'b> &'b G: petgraph::visit::IntoNodeIdentifiers
        + petgraph::visit::IntoEdgeReferences
        + petgraph::visit::IntoNeighborsDirected,
    for<'b> &'b G::NodeId: std::hash::Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
    S: NodeSizer<G>,
{
    /// Enable cursor functionality and initialize cursor position to the first data node
    pub fn enable_cursor(&mut self) {
        self.cursor.set_enabled(true);
        self.initialize_cursor();
    }

    /// Initialize cursor position to the first data node in the viewport graph
    pub fn initialize_cursor(&mut self) {
        // Find the first data node and extract needed data
        let cursor_data =
            self.find_first_data_node_in_viewport()
                .and_then(|(world_pos, node_data)| {
                    if let NodeRole::Data(domain_idx) = &node_data.role {
                        self.viewport_state
                            .world_to_viewport(world_pos)
                            .map(|viewport_pos| (*domain_idx, viewport_pos))
                    } else {
                        None
                    }
                });

        // Now mutate cursor with extracted data
        if let Some((domain_idx, viewport_pos)) = cursor_data {
            // Position cursor at left edge of node (fractional x=0.0, y=0.5 for center)
            self.cursor.set_node(domain_idx, (0.0, 0.5));
            self.cursor.set_viewport_pos(viewport_pos);

            trace!(
                "Initialized cursor at node {:?}, viewport ({}, {})",
                domain_idx, viewport_pos.x, viewport_pos.y
            );
        }
    }

    /// Move cursor horizontally (between layers/ranks)
    pub fn move_cursor_horizontal(&mut self, direction: i64) {
        if !self.cursor.is_enabled() || self.cursor.get_node_idx().is_none() {
            return;
        }

        // Use ViewportCursor's move_horizontal method
        match self.cursor.move_horizontal(direction, &self.viewport_graph) {
            Ok(()) => {
                trace!("Cursor moved horizontally by {}", direction);
                self.ensure_cursor_visible();
            }
            Err(e) => {
                if std::env::var("RUST_LOG").is_ok() {
                    eprintln!("Horizontal cursor movement failed: {}", e);
                }
            }
        }
    }

    /// Move cursor vertically (within same layer/rank)
    pub fn move_cursor_vertical(&mut self, direction: i64) {
        if !self.cursor.is_enabled() || self.cursor.get_node_idx().is_none() {
            return;
        }

        // Use ViewportCursor's move_vertical method
        match self.cursor.move_vertical(direction, &self.viewport_graph) {
            Ok(()) => {
                trace!("Cursor moved vertically by {}", direction);
                self.ensure_cursor_visible();
            }
            Err(e) => {
                if std::env::var("RUST_LOG").is_ok() {
                    eprintln!("Vertical cursor movement failed: {}", e);
                }
            }
        }
    }

    /// Helper to ensure the cursor is visible in the viewport by adjusting camera if needed
    fn ensure_cursor_visible(&mut self) {
        if !self.cursor.is_enabled() {
            return;
        }

        // Get cursor's viewport position
        let cursor_viewport = self.cursor.get_viewport_pos();

        // Check if it's within bounds
        let vp_bounds = self.viewport_state.viewport_bounds;
        if cursor_viewport.x >= vp_bounds.width || cursor_viewport.y >= vp_bounds.height {
            // Cursor is outside viewport - need to adjust camera
            // Compute the world position and center camera on it
            if let Some(world_pos) = self.cursor.to_world_pos(&self.viewport_graph) {
                self.viewport_state.camera_current = world_pos;
                self.viewport_state.camera_target = world_pos;
                self.viewport_state.camera_anim = None;

                trace!(
                    "Adjusted camera to ({}, {}) to keep cursor visible",
                    world_pos.x, world_pos.y
                );
            }
        }
    }

    /// Find the first data node in the viewport graph (leftmost, then topmost)
    fn find_first_data_node_in_viewport(&self) -> Option<(WorldPos, &LayoutNode)> {
        let mut best_candidate: Option<(WorldPos, &LayoutNode)> = None;

        for (world_pos, node_data) in &self.viewport_graph.nodes {
            if let NodeRole::Data(_) = &node_data.role {
                if let Some((best_pos, _)) = best_candidate {
                    // Prefer nodes further left, then higher up
                    if world_pos.x < best_pos.x
                        || (world_pos.x == best_pos.x && world_pos.y > best_pos.y)
                    {
                        best_candidate = Some((*world_pos, node_data));
                    }
                } else {
                    best_candidate = Some((*world_pos, node_data));
                }
            }
        }

        best_candidate
    }

    /// Get cursor's current viewport position for rendering
    pub fn get_cursor_viewport_pos(&self) -> Option<ViewportPos> {
        if self.cursor.is_enabled() {
            Some(self.cursor.get_viewport_pos())
        } else {
            None
        }
    }

    /// Get cursor's glyph character
    pub fn get_cursor_glyph(&self) -> char {
        self.cursor.get_glyph()
    }

    /// Set cursor glyph character
    pub fn set_cursor_glyph(&mut self, glyph: char) {
        self.cursor.set_glyph(glyph);
    }

    /// Check if cursor is enabled
    pub fn is_cursor_enabled(&self) -> bool {
        self.cursor.is_enabled()
    }
}
