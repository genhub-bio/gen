use log::trace;

use crate::{
    geometry::ViewportPos, graph_controller::GraphController, layout::NodeRole, plotter::NodeSizer,
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
    /// Enable cursor rendering
    pub fn show_cursor(&mut self) {
        self.cursor.set_visibility(true);
    }

    /// Initialize both cursor and camera positions based on the first data node in the anchor partition.
    ///
    /// This method couples camera and cursor initialization:
    /// 1. Finds a data node in the anchor partition's layout
    /// 2. Positions the camera so the node's left edge appears at the left edge of the deadzone (centered vertically)
    /// 3. Sets the cursor to track that node at that viewport position
    ///
    /// This ensures the cursor is visible and well-positioned when first enabled,
    /// with maximum room to move right before triggering camera following.
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
            // Calculate desired viewport position for the cursor at the left edge of the deadzone
            // Deadzone center is at viewport center, deadzone radius is (viewport_size * dead_zone_fraction * 0.5)
            let viewport_center_x = self.viewport_state.viewport_bounds.width / 2;
            let viewport_center_y = self.viewport_state.viewport_bounds.height / 2;

            // Calculate dead zone radius (fraction is already halved in animation.rs, so we apply the same here)
            let dead_zone_fraction_x =
                (self.viewport_state.dead_zone_fraction.0 * 0.5).clamp(0.0, 0.5);
            let dead_zone_radius_x =
                (self.viewport_state.viewport_bounds.width as f32 * dead_zone_fraction_x) as u16;

            // Position cursor at left edge of dead zone (horizontally) and centered vertically
            let desired_viewport_x = viewport_center_x.saturating_sub(dead_zone_radius_x);
            let desired_viewport_y = viewport_center_y;
            let desired_viewport_pos = ViewportPos::new(desired_viewport_x, desired_viewport_y);

            // Update camera so the node's left edge appears at the desired viewport position
            self.update_camera(node_world_pos, desired_viewport_pos);

            // Set cursor to track this node at its left edge (fractional x=0.0, y=0.5 for center)
            self.cursor.set_node(domain_idx, (0.7, 0.5));

            // Set cursor viewport position to where we positioned it via the camera
            self.cursor.set_viewport_pos(desired_viewport_pos);

            trace!(
                "Initialized cursor and camera: node {:?}, world ({}, {}), viewport ({}, {}) at left edge of deadzone",
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
}
