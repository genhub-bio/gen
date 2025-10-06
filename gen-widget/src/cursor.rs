use log::trace;
use petgraph::graph::NodeIndex;

use crate::{
    animation::Animation,
    geometry::{BigRect, Point, ViewportPos, WorldPos, WorldRect},
    graph_controller::GraphController,
    layout::{LayoutNode, NodeRole},
};

/// Cursor-related state grouped together for clarity
pub struct CursorState {
    /// Whether cursor functionality is enabled
    pub enabled: bool,

    /// Cursor's current and target world positions.
    pub current: WorldPos,
    pub target: WorldPos,
    pub anim: Option<Animation>,

    /// Current node position in the ViewportGraph (world coordinates)
    pub node_world_pos: Option<WorldPos>,
    /// Current node that cursor is positioned on (domain graph NodeIndex)
    pub node_domain_idx: Option<NodeIndex>,
    /// Cursor position relative to node center (0,0 = exactly on node center)
    pub node_offset: Point<i64>,

    /// Stored viewport coordinates for cursor restoration after layout changes
    /// This preserves the exact viewport position where the cursor appeared before layout change
    pub stored_viewport_pos: Option<ViewportPos>,

    /// Character to use for cursor rendering
    pub glyph: char,
}

impl Default for CursorState {
    fn default() -> Self {
        Self {
            enabled: false,
            current: WorldPos::ZERO,
            target: WorldPos::ZERO,
            anim: None,
            node_world_pos: None,
            node_domain_idx: None,
            node_offset: Point::new(0, 0),
            stored_viewport_pos: None,
            glyph: '█',
        }
    }
}

/// Cursor-related methods for GraphController
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
    S: crate::plotter::NodeSizer<G>,
{
    /// Enable cursor functionality and initialize cursor position to the first data node
    pub fn enable_cursor(&mut self) {
        self.viewport_state.cursor.enabled = true;
        self.initialize_cursor();
    }

    /// Initialize cursor position to the first data node in the viewport graph
    pub fn initialize_cursor(&mut self) {
        // Find the first data node in the viewport graph - collect the needed data first
        let first_node_data =
            self.find_first_data_node_in_viewport()
                .map(|(world_pos, node_data)| {
                    let node_size = node_data.size;
                    let domain_idx = if let NodeRole::Data(idx) = &node_data.role {
                        *idx
                    } else {
                        // This should never happen since find_first_data_node_in_viewport only returns Data nodes
                        panic!("Expected Data node but got {:?}", node_data.role);
                    };
                    (world_pos, node_size, domain_idx)
                });

        if let Some((world_pos, node_size, domain_idx)) = first_node_data {
            // Position cursor at left edge of first node
            let node_rect = BigRect::from_center_and_size(world_pos, node_size);
            let cursor_pos = WorldPos::new(node_rect.min.x, world_pos.y);

            self.viewport_state.cursor.current = cursor_pos;
            self.viewport_state.cursor.target = cursor_pos;
            self.viewport_state.cursor.node_world_pos = Some(world_pos);
            self.viewport_state.cursor.node_domain_idx = Some(domain_idx);

            // Calculate offset from center
            self.viewport_state.cursor.node_offset =
                Point::new(cursor_pos.x - world_pos.x, cursor_pos.y - world_pos.y);
        }
    }

    /// Find the exact same node at cursor position, returning both the rect and the domain node index
    pub fn find_domain_node_at_cursor(&self) -> Option<(WorldRect, NodeIndex)> {
        let cursor_pos = self.viewport_state.cursor.current;
        let viewport_graph = &self.viewport_graph;

        // Search through all nodes in the viewport graph
        for (world_pos, node_data) in &viewport_graph.nodes {
            let rect = BigRect::from_center_and_size(*world_pos, node_data.size);

            if rect.contains(cursor_pos) {
                if let NodeRole::Data(domain_node_idx) = &node_data.role {
                    return Some((rect, *domain_node_idx));
                }
            }
        }

        None
    }

    /// Move cursor horizontally to connected node (between ranks in DAG)
    pub fn move_cursor_horizontal(&mut self, direction: i64) {
        // Check if cursor has been properly initialized
        if self.viewport_state.cursor.node_domain_idx.is_none() {
            return;
        }

        // Step 1: Try intra-node movement
        let current_pos = self.viewport_state.cursor.current;
        let new_pos = WorldPos::new(current_pos.x + direction, current_pos.y);

        if let Some(world_pos) = self.viewport_state.cursor.node_world_pos {
            if let Some(node_data) = self.viewport_graph.nodes.get(&world_pos) {
                let rect = BigRect::from_center_and_size(world_pos, node_data.size);
                if rect.contains(new_pos) {
                    // Update position within node
                    self.viewport_state.cursor.current = new_pos;
                    self.viewport_state.cursor.target = new_pos;
                    self.viewport_state.cursor.node_offset =
                        Point::new(new_pos.x - world_pos.x, new_pos.y - world_pos.y);
                    self.log_cursor_position("horizontal movement within node");
                    self.ensure_cursor_visible();
                    return;
                }
            }
        }

        // Step 2: At boundary - jump to connected node
        if let Some((target_world_pos, target_node)) =
            self.find_connected_node_in_direction(direction)
        {
            let target_node_clone = target_node.clone();
            self.jump_cursor_to_node_viewport(target_world_pos, &target_node_clone);
        } else if std::env::var("RUST_LOG").is_ok() {
            eprintln!("No connected node found in direction {}", direction);
        }
    }

    /// Move cursor vertically to node in same rank (column)
    pub fn move_cursor_vertical(&mut self, direction: i64) {
        // Check if cursor has been properly initialized
        if self.viewport_state.cursor.node_domain_idx.is_none() {
            return;
        }

        // Step 1: Try intra-node movement
        let current_pos = self.viewport_state.cursor.current;
        let new_pos = WorldPos::new(current_pos.x, current_pos.y + direction);

        if let Some(world_pos) = self.viewport_state.cursor.node_world_pos {
            if let Some(node_data) = self.viewport_graph.nodes.get(&world_pos) {
                let rect = BigRect::from_center_and_size(world_pos, node_data.size);
                if rect.contains(new_pos) {
                    // Update position within node
                    self.viewport_state.cursor.current = new_pos;
                    self.viewport_state.cursor.target = new_pos;
                    self.viewport_state.cursor.node_offset =
                        Point::new(new_pos.x - world_pos.x, new_pos.y - world_pos.y);
                    self.log_cursor_position("vertical movement within node");
                    self.ensure_cursor_visible();
                    return;
                }
            }
        }

        // Step 2: At boundary - jump to node in same column
        if let Some((target_world_pos, target_node)) = self.find_node_in_column(direction) {
            let target_node_clone = target_node.clone();
            self.jump_cursor_to_node_viewport(target_world_pos, &target_node_clone);
        }
    }

    /// Unified jump function that handles cursor positioning based on ViewportGraph nodes
    fn jump_cursor_to_node_viewport(
        &mut self,
        target_world_pos: WorldPos,
        target_node: &LayoutNode,
    ) {
        let current_world_pos = self.viewport_state.cursor.node_world_pos;

        // Determine movement type by comparing current and target positions
        if let Some(current_pos) = current_world_pos {
            // Calculate direction vector
            let dx = target_world_pos.x - current_pos.x;
            let dy = target_world_pos.y - current_pos.y;

            // Determine primary movement direction
            let direction = if dx.abs() > dy.abs() { dx.signum() } else { 0 };

            if direction != 0 {
                // Horizontal movement - position at appropriate edge
                self.jump_cursor_between_ranks_viewport(target_world_pos, target_node, direction);
            } else {
                // Vertical movement - maintain x position if possible
                let current_x = self.viewport_state.cursor.current.x;
                self.jump_cursor_within_rank_viewport(target_world_pos, target_node, current_x);
            }
        } else {
            // Fallback: no current node - use within rank behavior
            let current_x = self.viewport_state.cursor.current.x;
            self.jump_cursor_within_rank_viewport(target_world_pos, target_node, current_x);
        }
    }

    /// Jump cursor within rank (vertical movement), maintaining x position if possible
    fn jump_cursor_within_rank_viewport(
        &mut self,
        target_world_pos: WorldPos,
        target_node: &LayoutNode,
        preferred_x: i64,
    ) {
        // Calculate target rectangle and find closest position
        let target_rect = BigRect::from_center_and_size(target_world_pos, target_node.size);
        let desired_pos = WorldPos::new(preferred_x, target_world_pos.y);

        // Clamp to node bounds
        let target_pos = WorldPos::new(
            desired_pos.x.clamp(target_rect.min.x, target_rect.max.x),
            desired_pos.y.clamp(target_rect.min.y, target_rect.max.y),
        );

        // Update cursor state directly
        self.viewport_state.cursor.current = target_pos;
        self.viewport_state.cursor.target = target_pos;
        self.viewport_state.cursor.node_world_pos = Some(target_world_pos);

        // Update domain index - cursor can only be on Data nodes
        let NodeRole::Data(domain_idx) = &target_node.role else {
            panic!(
                "Expected Data node in cursor jump, got {:?}",
                target_node.role
            );
        };
        self.viewport_state.cursor.node_domain_idx = Some(*domain_idx);

        // Update offset from center
        self.viewport_state.cursor.node_offset = Point::new(
            target_pos.x - target_world_pos.x,
            target_pos.y - target_world_pos.y,
        );

        // Log the jump
        self.log_cursor_position("jump within rank");

        // Ensure node is visible in viewport
        self.ensure_cursor_visible();
    }

    /// Jump cursor between ranks (horizontal movement), positioning at appropriate edge
    fn jump_cursor_between_ranks_viewport(
        &mut self,
        target_world_pos: WorldPos,
        target_node: &LayoutNode,
        direction: i64,
    ) {
        // Calculate target position based on direction
        let target_rect = BigRect::from_center_and_size(target_world_pos, target_node.size);
        let target_pos = if direction > 0 {
            // Moving right - position at left edge
            WorldPos::new(target_rect.min.x, target_world_pos.y)
        } else {
            // Moving left - position at right edge
            WorldPos::new(target_rect.max.x, target_world_pos.y)
        };

        // Update cursor state directly
        self.viewport_state.cursor.current = target_pos;
        self.viewport_state.cursor.target = target_pos;
        self.viewport_state.cursor.node_world_pos = Some(target_world_pos);

        // Update domain index - cursor can only be on Data nodes
        let NodeRole::Data(domain_idx) = &target_node.role else {
            panic!(
                "Expected Data node in cursor jump, got {:?}",
                target_node.role
            );
        };
        self.viewport_state.cursor.node_domain_idx = Some(*domain_idx);

        // Update offset from center
        self.viewport_state.cursor.node_offset = Point::new(
            target_pos.x - target_world_pos.x,
            target_pos.y - target_world_pos.y,
        );

        // Log the jump
        self.log_cursor_position("jump between ranks");

        // Ensure node is visible in viewport
        self.ensure_cursor_visible();
    }

    /// Restore cursor to a tracked node after viewport update
    /// This is a simplified version that just finds the node and sets the cursor position.
    /// It assumes the camera has already been adjusted to the correct position.
    pub fn restore_cursor_to_tracked_node(
        &mut self,
        target_node: NodeIndex,
        node_offset: Point<i64>,
    ) -> Result<(), String> {
        // Search viewport graph for the target node
        for (world_pos, node_data) in &self.viewport_graph.nodes {
            if let NodeRole::Data(domain_node_idx) = &node_data.role {
                if *domain_node_idx == target_node {
                    // Found the target node! Calculate new cursor position
                    let new_cursor_pos =
                        WorldPos::new(world_pos.x + node_offset.x, world_pos.y + node_offset.y);

                    // Ensure cursor stays within node bounds
                    let rect = BigRect::from_center_and_size(*world_pos, node_data.size);
                    let clamped_cursor = WorldPos::new(
                        new_cursor_pos.x.clamp(rect.min.x, rect.max.x),
                        new_cursor_pos.y.clamp(rect.min.y, rect.max.y),
                    );

                    // Update cursor position and tracking info
                    self.viewport_state.cursor.current = clamped_cursor;
                    self.viewport_state.cursor.target = clamped_cursor;
                    self.viewport_state.cursor.node_domain_idx = Some(target_node);
                    self.viewport_state.cursor.node_world_pos = Some(*world_pos);
                    self.viewport_state.cursor.node_offset = Point::new(
                        clamped_cursor.x - world_pos.x,
                        clamped_cursor.y - world_pos.y,
                    );

                    return Ok(());
                }
            }
        }

        Err(format!(
            "Could not find node {:?} in viewport graph after update",
            target_node
        ))
    }

    /// Restore cursor position after viewport update using node tracking information
    pub fn restore_cursor_after_viewport_update(
        &mut self,
        target_node: NodeIndex,
        node_offset: Point<i64>,
        _viewport_pos: ViewportPos,
    ) -> Result<(), String> {
        // Search viewport graph for the target node
        for (world_pos, node_data) in &self.viewport_graph.nodes {
            if let NodeRole::Data(domain_node_idx) = &node_data.role {
                if *domain_node_idx == target_node {
                    // Found the target node! Calculate new cursor position
                    let new_cursor_pos =
                        WorldPos::new(world_pos.x + node_offset.x, world_pos.y + node_offset.y);

                    // Ensure cursor stays within node bounds
                    let rect = BigRect::from_center_and_size(*world_pos, node_data.size);
                    let clamped_cursor = WorldPos::new(
                        new_cursor_pos.x.clamp(rect.min.x, rect.max.x),
                        new_cursor_pos.y.clamp(rect.min.y, rect.max.y),
                    );

                    // Update cursor position and tracking info
                    self.viewport_state.cursor.current = clamped_cursor;
                    self.viewport_state.cursor.target = clamped_cursor;
                    self.viewport_state.cursor.node_domain_idx = Some(target_node);
                    self.viewport_state.cursor.node_world_pos = Some(*world_pos);
                    self.viewport_state.cursor.node_offset = Point::new(
                        clamped_cursor.x - world_pos.x,
                        clamped_cursor.y - world_pos.y,
                    );

                    // NOTE: We do NOT adjust camera here anymore!
                    // The camera should already be positioned correctly by the caller.
                    // This function just restores the cursor to the tracked node.

                    return Ok(());
                }
            }
        }

        Err(format!(
            "Could not find node {:?} in viewport graph after update",
            target_node
        ))
    }

    /// Helper to check if a cursor position is still within the bounds of a given node
    #[allow(dead_code)]
    fn is_cursor_within_node(&self, pos: WorldPos, world_pos: WorldPos) -> bool {
        if let Some(node_data) = self.viewport_graph.nodes.get(&world_pos) {
            let rect = BigRect::from_center_and_size(world_pos, node_data.size);
            return rect.contains(pos);
        }
        false
    }

    /// Helper to ensure the cursor is visible in the viewport by adjusting camera if needed
    fn ensure_cursor_visible(&mut self) {
        let cursor_pos = self.viewport_state.cursor.current;
        if !self.viewport_state.is_visible(cursor_pos) {
            // Move camera to center on cursor
            self.viewport_state.camera_current = cursor_pos;
            self.viewport_state.camera_target = cursor_pos;
        }
    }

    /// Find the first data node in the viewport graph
    fn find_first_data_node_in_viewport(&self) -> Option<(WorldPos, &LayoutNode)> {
        // Find the leftmost, topmost data node
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

    /// Find connected node in the given horizontal direction using layer-based traversal
    fn find_connected_node_in_direction(&self, direction: i64) -> Option<(WorldPos, &LayoutNode)> {
        let current_node = self
            .viewport_state
            .cursor
            .node_domain_idx
            .expect("Cursor should be associated with a node");
        let current_pos = self.viewport_state.cursor.current;
        let current_layer = self
            .viewport_graph
            .find_domain_node_layer(current_node)
            .expect("Post layout a node should have an associated layer");

        // Determine target layer based on direction
        let target_layer = if direction > 0 {
            // Moving right: next layer
            if current_layer + 1 < self.viewport_graph.layer_count() {
                current_layer + 1
            } else {
                return None; // No next layer
            }
        } else {
            // Moving left: previous layer
            if current_layer == 0 {
                return None; // No previous layer
            } else {
                current_layer.saturating_sub(1)
            }
        };

        // Get domain nodes in the target layer
        let target_layer_nodes = self.viewport_graph.get_layer(target_layer)?;

        // Filter by target layer and choose the best candidate
        self.viewport_graph
            .find_adjacent_data_nodes(current_node)
            .into_iter()
            .filter_map(|(pos, node, _path)| {
                // Since cursor can only be on Data nodes, we can assume node.role is Data
                let NodeRole::Data(domain_idx) = node.role else {
                    panic!(
                        "Expected Data node in cursor navigation, got {:?}",
                        node.role
                    );
                };
                if target_layer_nodes.contains(&domain_idx) {
                    Some((pos, node))
                } else {
                    None
                }
            })
            .min_by_key(|(pos, _)| {
                // Sort by vertical distance first (prefer same Y level), then by -y (highest y wins)
                let dy = (pos.y - current_pos.y).abs();
                (dy, -pos.y)
            })
    }

    /// Find node in the same layer (rank) in the given vertical direction
    fn find_node_in_column(&self, direction: i64) -> Option<(WorldPos, &LayoutNode)> {
        let current_pos = self.viewport_state.cursor.current;
        let current_world_pos = self.viewport_state.cursor.node_world_pos?;

        // Get the current node's layer using the unified layer structure
        let current_layer = self.viewport_graph.find_node_layer(&current_world_pos)?;
        let layer_domain_nodes = self.viewport_graph.get_layer(current_layer)?;

        let viewport_graph = &self.viewport_graph;
        let mut candidates = Vec::new();

        // Search for Data nodes in the same layer by their domain indices
        for &domain_idx in layer_domain_nodes {
            // Find the world position for this domain node
            if let Some(&world_pos) = viewport_graph.domain_to_world.get(&domain_idx) {
                if let Some(node_data) = viewport_graph.nodes.get(&world_pos) {
                    let dy = world_pos.y - current_pos.y;

                    // Check vertical direction
                    if (direction > 0 && dy > 0) || (direction < 0 && dy < 0) {
                        candidates.push((world_pos, node_data, dy.abs()));
                    }
                }
            }
        }

        // Choose the closest candidate in the vertical direction
        candidates
            .into_iter()
            .min_by_key(|(_, _, dy)| *dy)
            .map(|(pos, node, _)| (pos, node))
    }

    /// Log comprehensive cursor position information in all coordinate formats
    pub fn log_cursor_position(&self, context: &str) {
        if !log::log_enabled!(log::Level::Trace) {
            return;
        }

        let cursor_world_pos = self.viewport_state.cursor.current;

        // Calculate viewport position
        let viewport_pos = self.viewport_state.world_to_viewport(cursor_world_pos);

        // Calculate terminal position
        let terminal_pos = self.viewport_state.world_to_terminal(cursor_world_pos);

        // Get local position relative to node if available
        let local_pos = self
            .viewport_state
            .cursor
            .node_world_pos
            .map(|node_world_pos| {
                Point::new(
                    cursor_world_pos.x - node_world_pos.x,
                    cursor_world_pos.y - node_world_pos.y,
                )
            });

        // Camera information
        let camera_world_pos = self.viewport_state.camera_current;
        let camera_origin = self.viewport_state.camera_origin_world();

        trace!("=== Cursor Position Debug ({}) ===", context);
        trace!(
            "  World Position: ({}, {})",
            cursor_world_pos.x, cursor_world_pos.y
        );

        if let Some(vp_pos) = viewport_pos {
            trace!("  Viewport Position: ({}, {})", vp_pos.x, vp_pos.y);
        } else {
            trace!("  Viewport Position: OUT_OF_BOUNDS");
        }

        if let Some((term_x, term_y)) = terminal_pos {
            trace!("  Terminal Position: ({}, {})", term_x, term_y);
        } else {
            trace!("  Terminal Position: OUT_OF_BOUNDS");
        }

        if let Some(local) = local_pos {
            trace!("  Local Position (node offset): ({}, {})", local.x, local.y);
            if let Some(node_world_pos) = self.viewport_state.cursor.node_world_pos {
                trace!(
                    "  Node World Position: ({}, {})",
                    node_world_pos.x, node_world_pos.y
                );
            }
        } else {
            trace!("  Local Position: NO_NODE_TRACKED");
        }

        if let Some(domain_idx) = self.viewport_state.cursor.node_domain_idx {
            trace!("  Domain Node Index: {}", domain_idx.index());
        } else {
            trace!("  Domain Node Index: NONE");
        }

        trace!(
            "  Camera Current: ({}, {})",
            camera_world_pos.x, camera_world_pos.y
        );
        trace!(
            "  Camera Origin: ({}, {})",
            camera_origin.x, camera_origin.y
        );

        if let Some((term_x, term_y)) = terminal_pos {
            trace!(
                "  -> With camera at origin, cursor would appear at terminal ({}, {})",
                term_x, term_y
            );
        }

        trace!("=== End Cursor Position Debug ===");
    }

    /// Log cursor position tracking during layout changes with detailed coordinate information
    pub fn log_cursor_restoration(&self, context: &str, stored_viewport_pos: Option<ViewportPos>) {
        if !log::log_enabled!(log::Level::Trace) {
            return;
        }

        trace!("=== Cursor Restoration Debug ({}) ===", context);

        if let Some(stored_vp) = stored_viewport_pos {
            trace!(
                "  Stored Viewport Position: ({}, {})",
                stored_vp.x, stored_vp.y
            );

            // Show where cursor would appear with current camera
            if let Some(actual_vp) = self
                .viewport_state
                .world_to_viewport(self.viewport_state.cursor.current)
            {
                trace!(
                    "  Current Viewport Position: ({}, {})",
                    actual_vp.x, actual_vp.y
                );
                trace!(
                    "  Viewport Delta: ({}, {})",
                    actual_vp.x as i32 - stored_vp.x as i32,
                    actual_vp.y as i32 - stored_vp.y as i32
                );
            } else {
                trace!("  Current Viewport Position: OUT_OF_BOUNDS");
            }
        } else {
            trace!("  No stored viewport position");
        }

        // Log current positions in all formats
        self.log_cursor_position(&format!("during {}", context));

        trace!("=== End Cursor Restoration Debug ===");
    }
}
