use log::trace;

use crate::{
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
    /// Enable cursor rendering and initialize cursor position to the first data node
    /// TODO: This will be replaced by automatic initialization in the render loop
    pub fn enable_cursor(&mut self) {
        self.cursor.set_show_cursor(true);
        self.initialize_cursor_and_camera();
    }

    /// Initialize both cursor and camera positions based on the first data node in the anchor partition.
    ///
    /// This method couples camera and cursor initialization:
    /// 1. Finds a data node in the anchor partition's layout
    /// 2. Positions the camera so the node's left edge appears at a comfortable viewport position (1/4 from left, centered vertically)
    /// 3. Sets the cursor to track that node at that viewport position
    ///
    /// This ensures the cursor is visible and well-positioned when first enabled.
    pub fn initialize_cursor_and_camera(&mut self) {
        let detail_level = self.get_detail_level();
        let anchor_partition = self.partition_controller.get_anchor_partition();

        // Get the anchor partition's layout and find the first data node
        let node_info = self
            .partition_controller
            .partition_table
            .get_layout(anchor_partition, detail_level)
            .and_then(|layout| {
                layout.graph.node_indices().find_map(|layout_node_idx| {
                    layout.graph.node_weight(layout_node_idx).and_then(|node| {
                        if let NodeRole::Data(domain_idx) = &node.role {
                            // Convert local position to world position
                            let world_pos = self
                                .partition_controller
                                .partition_table
                                .local_to_world(node.pos, detail_level);
                            Some((*domain_idx, world_pos))
                        } else {
                            None
                        }
                    })
                })
            });

        if let Some((domain_idx, node_world_pos)) = node_info {
            // Calculate desired viewport position for the cursor
            // Position it 1/4 from the left edge and vertically centered
            let desired_viewport_x = self.viewport_state.viewport_bounds.width / 4;
            let desired_viewport_y = self.viewport_state.viewport_bounds.height / 2;
            let desired_viewport_pos = ViewportPos::new(desired_viewport_x, desired_viewport_y);

            // Update camera so the node's left edge appears at the desired viewport position
            self.update_camera(node_world_pos, desired_viewport_pos);

            // Set cursor to track this node at its left edge (fractional x=0.0, y=0.5 for center)
            self.cursor.set_node(domain_idx, (0.0, 0.5));

            // Set cursor viewport position to where we positioned it via the camera
            self.cursor.set_viewport_pos(desired_viewport_pos);

            trace!(
                "Initialized cursor and camera: node {:?}, world ({}, {}), viewport ({}, {})",
                domain_idx,
                node_world_pos.x,
                node_world_pos.y,
                desired_viewport_pos.x,
                desired_viewport_pos.y
            );
        } else {
            trace!(
                "No data nodes found in anchor partition layout for cursor/camera initialization"
            );
        }
    }

    /// Move cursor horizontally (between layers/ranks)
    pub fn move_cursor_horizontal(&mut self, direction: i64) {
        if self.cursor.get_node_idx().is_none() {
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
        if self.cursor.get_node_idx().is_none() {
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

    /// Get cursor's current viewport position for rendering (if visible)
    pub fn get_cursor_viewport_pos(&self) -> Option<ViewportPos> {
        if self.cursor.show_cursor() {
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

    /// Check if cursor should be shown (visible)
    pub fn is_cursor_shown(&self) -> bool {
        self.cursor.show_cursor()
    }

    /// Check if cursor is enabled (legacy - cursor is always functionally enabled)
    pub fn is_cursor_enabled(&self) -> bool {
        true
    }
}
