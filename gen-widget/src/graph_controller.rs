use std::{cmp, collections::HashSet, hash::Hash};

use crossterm::event::{KeyCode, KeyEvent};
use log::trace;
use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{
        EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeCount, NodeIndexable, Visitable,
    },
};
#[cfg(test)]
use ratatui::style::Color;

// Re-export the core module types
pub use crate::cursor::CursorState;
pub use crate::viewport_state::{ViewportState, WorldBuffer};
use crate::{
    geometry::{BigRect, Point, ViewportPos, WorldPos},
    layout::{LayoutEdge, LayoutNode, NodeRole, PartitionLayout, VisualDetail},
    partition_controller::{ControllerConfig, PartitionController},
    partition_table::PartitionConfig,
    plotter::NodeSizer,
    standalone_sugiyama::{self, VERTEX_SPACING_DEFAULT},
    viewport_graph::ViewportGraph,
};

/// Combined configuration for the entire graph widget system
#[derive(Debug, Clone, Default)]
pub struct GraphConfig {
    /// Partition behavior configuration  
    pub partition: PartitionConfig,
    /// Controller memory management configuration
    pub controller: ControllerConfig,
    /// Layout algorithm configuration
    pub layout: standalone_sugiyama::Config,
}

/// The controller is designed to be initialized outside the event loop for
/// graph loading and layout computation, then used by widgets during rendering.
pub struct GraphController<G, S>
where
    G: GraphBase + Clone,
    S: NodeSizer<G>,
{
    /// The original graph used for node lookups and rendering
    pub graph: G,

    /// Viewport state managing camera, cursor, animations, and viewport bounds
    pub viewport_state: ViewportState,

    /// Current detail level for visualization
    detail_level: VisualDetail,

    /// Partition controller for managing graph partitioning and layout (without its own viewport state)
    pub partition_controller: PartitionController<G, S>,
    /// Set of partitions currently loaded in the unified layout
    loaded_partitions: std::collections::HashSet<usize>,
    /// Viewport-based graph containing only visible nodes and edges
    pub viewport_graph: ViewportGraph,
    /// Camera position when viewport graph was last rebuilt
    /// (for hysteresis to control rebuild frequency)
    last_rebuild_camera_center: WorldPos,
    /// Flag indicating that the viewport graph needs to be rebuilt
    pub rebuild_needed: bool,
}

impl<G, S> GraphController<G, S>
where
    G: GraphBase
        + Clone
        + EdgeIndexable
        + NodeIndexable
        + NodeCount
        + Visitable
        + IntoNodeIdentifiers
        + IntoEdgeReferences
        + IntoNeighborsDirected,
    G::NodeId: Copy + Eq + Hash + Ord,
    G::EdgeId: Clone,
    for<'b> &'b G: IntoNodeIdentifiers + IntoEdgeReferences + IntoNeighborsDirected,
    for<'b> &'b G::NodeId: Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
    S: NodeSizer<G>,
{
    /// Create a new GraphController with a graph and node sizer
    ///
    /// # Parameters
    /// - graph: The graph to partition and manage
    /// - node_sizer: Function object to determine node sizes at different levels of detail
    pub fn new(graph: G, node_sizer: S) -> Self
    where
        <G as petgraph::visit::GraphBase>::NodeId: std::fmt::Debug,
    {
        Self::new_with_config(graph, node_sizer, GraphConfig::default())
    }

    /// Create a new GraphController with a graph, node sizer, and custom configuration
    ///
    /// # Parameters
    /// - graph: The graph to partition and manage
    /// - node_sizer: Function object to determine node sizes at different levels of detail
    /// - config: Configuration for partitioning, memory management, and layout
    pub fn new_with_config(graph: G, node_sizer: S, config: GraphConfig) -> Self
    where
        <G as petgraph::visit::GraphBase>::NodeId: std::fmt::Debug,
    {
        let partition_controller = PartitionController::new_with_config(
            graph,
            node_sizer,
            config.partition,
            config.controller,
        );

        let mut controller = Self {
            graph,
            viewport_state: ViewportState::new(),
            detail_level: VisualDetail::Truncated, // Default detail level
            partition_controller,
            loaded_partitions: HashSet::new(),
            viewport_graph: ViewportGraph::empty(),
            last_rebuild_camera_center: WorldPos::ZERO,
            rebuild_needed: true, // Start with a rebuild required
        };

        // Initialize by loading the reference partition (partition 0)
        // This ensures at least one partition is available for viewport expansion
        if let Err(e) = controller.partition_controller.set_anchor_partition(0) {
            eprintln!("Warning: Failed to initialize reference partition: {}", e);
        }

        controller
    }

    pub fn get_layout(&self, partition_idx: usize) -> Option<&PartitionLayout> {
        let detail_level = self.get_detail_level();
        self.partition_controller
            .partition_table
            .get_layout(partition_idx, detail_level)
    }

    /// Legacy: Get the unified layout graph that spans all partitions
    /// This is kept temporarily for debugging but should not be used
    #[allow(dead_code)]
    fn get_unified_layout(&self) -> &StableGraph<LayoutNode, LayoutEdge, Undirected, u32> {
        let detail_level = self.get_detail_level();
        self.partition_controller
            .partition_table
            .get_unified_layout(detail_level)
    }

    /// Export the viewport graph to DOT format for visualization
    pub fn export_to_dot(&self, filename: &str) -> Result<(), std::io::Error> {
        crate::dot_export::export_to_dot(&self.viewport_graph, filename)
    }

    /// Set the reference partition and reset coordinate system  
    pub fn set_anchor_partition(&mut self, partition_idx: usize) -> Result<(), String> {
        self.partition_controller
            .set_anchor_partition(partition_idx)
    }

    /// Ensure a partition is loaded for rendering
    pub fn ensure_partition_loaded(&mut self, partition_idx: usize) -> Result<(), String> {
        self.partition_controller
            .ensure_partition_loaded(partition_idx)
    }

    /// Detect camera motion and set rebuild flag if viewport graph needs updating
    /// This checks camera movement and viewport bounds changes to determine if a rebuild is needed
    pub fn detect_motion(&mut self) {
        // Always rebuild if the viewport graph is empty (handles initialization)
        if self.viewport_graph.graph.node_count() == 0 {
            self.rebuild_needed = true;
            return;
        }

        let threshold_x = self.viewport_state.viewport_bounds.width as i64 / 2;
        let threshold_y = self.viewport_state.viewport_bounds.height as i64 / 2;

        let current_camera = self.viewport_state.camera_current;
        let movement_x = (current_camera.x - self.last_rebuild_camera_center.x).abs();
        let movement_y = (current_camera.y - self.last_rebuild_camera_center.y).abs();

        if movement_x > threshold_x || movement_y > threshold_y {
            self.rebuild_needed = true;
        }
    }

    /// Change the current level of detail and sync with partition controller
    pub fn set_detail_level(&mut self, detail_level: VisualDetail) {
        // Only proceed if detail level is actually changing
        if detail_level != self.detail_level {
            // Only try to preserve cursor position if we have a tracked node
            if let Some(node_idx) = self.viewport_state.cursor.node_domain_idx {
                // Log comprehensive cursor position before detail level change
                self.log_cursor_position("before detail level change");

                // Get old node size and calculate proportional offset BEFORE changing detail level
                let old_detail = self.detail_level;
                let node_id = <G as NodeIndexable>::from_index(&self.graph, node_idx.index());
                let (old_width, old_height) = self
                    .partition_controller
                    .node_sizer
                    .get_node_size(&node_id, old_detail);

                // Calculate proportional offset (0.0 to 1.0) within the node
                let proportional_offset_x = if old_width > 0 {
                    (self.viewport_state.cursor.node_offset.x as f64) / (old_width as f64)
                } else {
                    0.0
                };
                let proportional_offset_y = if old_height > 0 {
                    (self.viewport_state.cursor.node_offset.y as f64) / (old_height as f64)
                } else {
                    0.0
                };

                trace!(
                    "Old node size: ({}, {}), offset: {:?}, proportional: ({:.2}, {:.2})",
                    old_width,
                    old_height,
                    self.viewport_state.cursor.node_offset,
                    proportional_offset_x,
                    proportional_offset_y
                );

                // Store current viewport position for restoration after detail level change
                if let Some(viewport_pos) = self
                    .viewport_state
                    .world_to_viewport(self.viewport_state.cursor.current)
                {
                    trace!(
                        "Storing cursor viewport position {:?} for restoration after detail level change",
                        viewport_pos
                    );
                    self.viewport_state.cursor.stored_viewport_pos = Some(viewport_pos);
                } else {
                    trace!(
                        "Could not get viewport position for cursor at world {:?}",
                        self.viewport_state.cursor.current
                    );
                    self.viewport_state.cursor.stored_viewport_pos = None;
                }

                // Now change the detail level
                self.detail_level = detail_level;
                self.partition_controller.set_detail_level(detail_level);

                // Get new node size and scale the offset proportionally
                let (new_width, new_height) = self
                    .partition_controller
                    .node_sizer
                    .get_node_size(&node_id, detail_level);

                let new_offset_x = (proportional_offset_x * new_width as f64).round() as i64;
                let new_offset_y = (proportional_offset_y * new_height as f64).round() as i64;

                self.viewport_state.cursor.node_offset = Point::new(new_offset_x, new_offset_y);

                trace!(
                    "New node size: ({}, {}), scaled offset: {:?}",
                    new_width, new_height, self.viewport_state.cursor.node_offset
                );

                // Reset camera to origin for new coordinate system
                self.viewport_state.camera_current = WorldPos::ZERO;
                self.viewport_state.camera_target = WorldPos::ZERO;
                self.viewport_state.camera_anim = None;

                // Force viewport graph to rebuild with new detail level
                self.viewport_graph = ViewportGraph::empty();

                // The cursor will be restored in update_viewport_graph() which will:
                // 1. Find the node's new world position
                // 2. Adjust camera BEFORE building viewport graph
                // 3. Build viewport graph with adjusted camera
                // 4. Restore cursor position
            } else {
                // No tracked node, just change the detail level
                self.detail_level = detail_level;
                self.partition_controller.set_detail_level(detail_level);

                // Reset camera to origin for new coordinate system
                self.viewport_state.camera_current = WorldPos::ZERO;
                self.viewport_state.camera_target = WorldPos::ZERO;
                self.viewport_state.camera_anim = None;

                // Force viewport graph to rebuild with new detail level
                self.viewport_graph = ViewportGraph::empty();
            }
        }
    }

    /// Get the current level of detail
    pub fn get_detail_level(&self) -> VisualDetail {
        self.detail_level
    }

    /// Calculate total bounds needed to display all partitions
    #[allow(clippy::type_complexity)]
    pub fn calculate_total_bounds(&mut self) -> Result<BigRect<i64>, String> {
        self.partition_controller.calculate_total_bounds()
    }

    /// Get the current reference partition index
    pub fn get_anchor_partition(&self) -> usize {
        self.partition_controller.get_anchor_partition()
    }

    /// Get the number of currently loaded partitions
    pub fn loaded_partition_count(&self) -> usize {
        self.partition_controller.loaded_partition_count()
    }

    /// Get information about all loaded partitions (idx, start_x, width, height)
    pub fn get_loaded_partitions_info(&self) -> Vec<(usize, i64, i64, i64)> {
        self.partition_controller.get_loaded_partitions_info()
    }

    /// Convert local coordinates to world coordinates
    pub fn local_to_world(&self, local_pos: crate::geometry::LocalPos) -> WorldPos {
        self.partition_controller.local_to_world(local_pos)
    }

    /// Convert world coordinates to local coordinates  
    pub fn world_to_local(
        &self,
        world_pos: WorldPos,
    ) -> Result<crate::geometry::LocalPos, crate::partition_table::PartitionIdxError> {
        self.partition_controller.world_to_local(world_pos)
    }

    /// Load partitions to cover the camera's current view
    /// Returns a sorted list of partition indices that cover the camera view
    pub fn ensure_camera_coverage(&mut self) -> Result<Vec<usize>, String> {
        log::trace!("ensure_camera_coverage: called");
        let buffer_factor = 2.0;
        let camera_rect = self.viewport_state.camera_rect().resize(buffer_factor);
        log::trace!(
            "ensure_camera_coverage: loading partitions for rect coordinates: x: {}-{}, y: {}-{}",
            camera_rect.left(),
            camera_rect.right(),
            camera_rect.bottom(),
            camera_rect.top()
        );
        self.partition_controller
            .load_partitions_for_rect(camera_rect)
    }

    /// Find a domain node's world position in the new layout system after layout changes
    /// This searches the partition layouts directly to find the node's current world position
    pub fn find_domain_node_world_position(&self, domain_idx: NodeIndex) -> Option<WorldPos> {
        // Search through all loaded partitions for this domain node
        let loaded_partitions = self.partition_controller.get_loaded_partitions_info();
        let detail_level = self.get_detail_level();

        for (partition_idx, _, _, _) in loaded_partitions {
            // Get the layout for this partition
            if let Some(layout) = self
                .partition_controller
                .partition_table
                .get_layout(partition_idx, detail_level)
            {
                // Search for the domain node in this partition's layout
                for layout_node_idx in layout.graph.node_indices() {
                    if let Some(layout_node) = layout.graph.node_weight(layout_node_idx) {
                        if let NodeRole::Data(layout_domain_idx) = &layout_node.role {
                            if *layout_domain_idx == domain_idx {
                                // Found the node! Convert its local position to world coordinates
                                return Some(
                                    self.partition_controller
                                        .partition_table
                                        .local_to_world(layout_node.pos, detail_level),
                                );
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Zoom in: increase detail level or call disperse
    pub fn zoom_in(&mut self) {
        match self.detail_level {
            VisualDetail::Minimal => {
                self.set_detail_level(VisualDetail::Truncated);
            }
            VisualDetail::Truncated => {
                self.set_detail_level(VisualDetail::Full);
            }
            VisualDetail::Full => {
                self.disperse();
            }
        }
    }

    /// Zoom out: decrease detail level or call contract
    pub fn zoom_out(&mut self) {
        match self.detail_level {
            VisualDetail::Full => {
                // Check if we can contract first
                if self.get_current_vertex_spacing() > VERTEX_SPACING_DEFAULT {
                    self.contract();
                } else {
                    self.set_detail_level(VisualDetail::Truncated);
                }
            }
            VisualDetail::Truncated => {
                self.set_detail_level(VisualDetail::Minimal);
            }
            VisualDetail::Minimal => {
                // Already at minimum zoom
            }
        }
    }

    /// Increase vertex spacing and refresh layouts
    pub fn disperse(&mut self) {
        // Store cursor position before layout change
        let cursor_node = self.viewport_state.cursor.node_domain_idx;
        let cursor_node_offset = self.viewport_state.cursor.node_offset;

        // Log comprehensive cursor position before disperse
        if cursor_node.is_some() {
            self.log_cursor_position("before disperse");
        }

        // IMPORTANT: Store viewport position BEFORE changing layouts
        let stored_viewport_pos = if cursor_node.is_some() {
            let viewport_pos = self
                .viewport_state
                .world_to_viewport(self.viewport_state.cursor.current);
            if let Some(pos) = viewport_pos {
                trace!(
                    "Storing cursor viewport position {:?} for restoration after disperse",
                    pos
                );
            }
            // Store in cursor state for update_viewport_graph() to use
            self.viewport_state.cursor.stored_viewport_pos = viewport_pos;
            viewport_pos
        } else {
            None
        };

        // Look up which partition contains the tracked node BEFORE clearing layouts
        let tracked_node_partition = if let Some(node_idx) = cursor_node {
            let node_id = <G as NodeIndexable>::from_index(&self.graph, node_idx.index());
            self.partition_controller
                .partition_table
                .node_map
                .get(&node_id)
                .map(|(partition_idx, _)| *partition_idx)
        } else {
            None
        };

        // Increase vertex spacing
        self.partition_controller.increment_vertex_spacing(2.0);
        trace!(
            "vertex spacing now {}",
            self.partition_controller.get_vertex_spacing()
        );

        // Clear all layouts to force complete recalculation with new spacing
        self.partition_controller.clear_all_layouts();

        // Clear loaded partitions tracking since layouts are gone
        self.loaded_partitions.clear();

        // Set the tracked node's partition as anchor
        // This makes the node's world position = its local position within the partition
        // We only load what we need, not everything from partition 0 to here
        let anchor_partition = tracked_node_partition.unwrap_or(0);
        if let Err(e) = self
            .partition_controller
            .set_anchor_partition(anchor_partition)
        {
            eprintln!(
                "Warning: Failed to set anchor partition during disperse: {}",
                e
            );
        }

        // Now try to adjust camera to preserve cursor position if we have a tracked node
        if let (Some(node_idx), Some(viewport_pos)) = (cursor_node, stored_viewport_pos) {
            // Find the tracked node's position in the new layout
            // The partition is now loaded because set_anchor_partition() loaded it
            if let Some(node_world_pos) = self.find_domain_node_world_position(node_idx) {
                trace!(
                    "Found tracked node {:?} at new world position {:?}",
                    node_idx, node_world_pos
                );

                // Calculate the cursor world position with offset
                let cursor_world_pos = WorldPos::new(
                    node_world_pos.x + cursor_node_offset.x,
                    node_world_pos.y + cursor_node_offset.y,
                );

                // Calculate the camera position needed to keep cursor at stored viewport position
                //
                // IMPORTANT: camera_current represents the CENTER of the viewport in world coordinates,
                // not the top-left corner. The top-left corner is at: camera - (width/2, height/2)
                //
                // The camera origin (top-left) is: camera_origin = camera - (width/2, height/2)
                // Viewport position relative to origin: viewport_pos = world_pos - camera_origin
                // Substituting: viewport_pos = world_pos - camera + (width/2, height/2)
                // Solving for camera: camera = world_pos - viewport_pos + (width/2, height/2)
                let half_width = self.viewport_state.viewport_bounds.width as i64 / 2;
                let half_height = self.viewport_state.viewport_bounds.height as i64 / 2;

                let camera_x = cursor_world_pos.x - viewport_pos.x as i64 + half_width;
                let camera_y = cursor_world_pos.y - viewport_pos.y as i64 + half_height;

                self.viewport_state.camera_current = WorldPos::new(camera_x, camera_y);
                self.viewport_state.camera_target = WorldPos::new(camera_x, camera_y);
                self.viewport_state.camera_anim = None;

                trace!(
                    "Adjusted camera to {:?} to preserve cursor at viewport ({}, {})",
                    self.viewport_state.camera_current, viewport_pos.x, viewport_pos.y
                );
            } else {
                trace!(
                    "Could not find tracked node {:?} in new layout, resetting camera to origin",
                    node_idx
                );
                self.viewport_state.camera_current = WorldPos::ZERO;
                self.viewport_state.camera_target = WorldPos::ZERO;
                self.viewport_state.camera_anim = None;
            }
        } else {
            // No tracked node or no stored viewport position, reset camera to origin
            self.viewport_state.camera_current = WorldPos::ZERO;
            self.viewport_state.camera_target = WorldPos::ZERO;
            self.viewport_state.camera_anim = None;
        }

        // Load partitions needed for current camera view
        if let Err(e) = self.ensure_camera_coverage() {
            eprintln!("Error ensuring camera coverage during disperse: {}", e);
        }

        // Force viewport graph rebuild - let update_viewport_graph() handle cursor restoration
        self.viewport_graph = ViewportGraph::empty();
        self.rebuild_needed = true;
    }

    /// Decrease vertex spacing and refresh layouts
    pub fn contract(&mut self) {
        let current_spacing = self.get_current_vertex_spacing();
        if current_spacing > VERTEX_SPACING_DEFAULT {
            // Store cursor position before layout change
            let cursor_node = self.viewport_state.cursor.node_domain_idx;
            let cursor_node_offset = self.viewport_state.cursor.node_offset;

            // Log comprehensive cursor position before contract
            if cursor_node.is_some() {
                self.log_cursor_position("before contract");
            }

            // IMPORTANT: Store viewport position BEFORE resetting camera
            let stored_viewport_pos = if cursor_node.is_some() {
                let viewport_pos = self
                    .viewport_state
                    .world_to_viewport(self.viewport_state.cursor.current);
                if let Some(pos) = viewport_pos {
                    trace!(
                        "Storing cursor viewport position {:?} for restoration after contract",
                        pos
                    );
                }
                // Store in cursor state for update_viewport_graph() to use
                self.viewport_state.cursor.stored_viewport_pos = viewport_pos;
                viewport_pos
            } else {
                None
            };

            // Look up which partition contains the tracked node BEFORE clearing layouts
            let tracked_node_partition = if let Some(node_idx) = cursor_node {
                let node_id = <G as NodeIndexable>::from_index(&self.graph, node_idx.index());
                self.partition_controller
                    .partition_table
                    .node_map
                    .get(&node_id)
                    .map(|(partition_idx, _)| *partition_idx)
            } else {
                None
            };

            // Decrease vertex spacing
            let new_spacing = (current_spacing - 2.0).max(VERTEX_SPACING_DEFAULT);
            self.partition_controller.set_vertex_spacing(new_spacing);

            // Clear all layouts to force complete recalculation with new spacing
            self.partition_controller.clear_all_layouts();

            // Clear loaded partitions tracking since layouts are gone
            self.loaded_partitions.clear();

            // Set the tracked node's partition as anchor
            // This makes the node's world position = its local position within the partition
            // We only load what we need, not everything from partition 0 to here
            let anchor_partition = tracked_node_partition.unwrap_or(0);
            if let Err(e) = self
                .partition_controller
                .set_anchor_partition(anchor_partition)
            {
                eprintln!(
                    "Warning: Failed to set anchor partition during contract: {}",
                    e
                );
            }

            // Now try to adjust camera to preserve cursor position if we have a tracked node
            if let (Some(node_idx), Some(viewport_pos)) = (cursor_node, stored_viewport_pos) {
                // Find the tracked node's position in the new layout
                // The partition is now loaded because set_anchor_partition() loaded it
                if let Some(node_world_pos) = self.find_domain_node_world_position(node_idx) {
                    trace!(
                        "Found tracked node {:?} at new world position {:?}",
                        node_idx, node_world_pos
                    );

                    // Calculate the cursor world position with offset
                    let cursor_world_pos = WorldPos::new(
                        node_world_pos.x + cursor_node_offset.x,
                        node_world_pos.y + cursor_node_offset.y,
                    );

                    // Calculate the camera position needed to keep cursor at stored viewport position
                    //
                    // IMPORTANT: camera_current represents the CENTER of the viewport in world coordinates,
                    // not the top-left corner. The top-left corner is at: camera - (width/2, height/2)
                    //
                    // The camera origin (top-left) is: camera_origin = camera - (width/2, height/2)
                    // Viewport position relative to origin: viewport_pos = world_pos - camera_origin
                    // Substituting: viewport_pos = world_pos - camera + (width/2, height/2)
                    // Solving for camera: camera = world_pos - viewport_pos + (width/2, height/2)
                    let half_width = self.viewport_state.viewport_bounds.width as i64 / 2;
                    let half_height = self.viewport_state.viewport_bounds.height as i64 / 2;

                    let camera_x = cursor_world_pos.x - viewport_pos.x as i64 + half_width;
                    let camera_y = cursor_world_pos.y - viewport_pos.y as i64 + half_height;

                    self.viewport_state.camera_current = WorldPos::new(camera_x, camera_y);
                    self.viewport_state.camera_target = WorldPos::new(camera_x, camera_y);
                    self.viewport_state.camera_anim = None;

                    trace!(
                        "Adjusted camera to {:?} to preserve cursor at viewport ({}, {})",
                        self.viewport_state.camera_current, viewport_pos.x, viewport_pos.y
                    );
                } else {
                    trace!(
                        "Could not find tracked node {:?} in new layout, resetting camera to origin",
                        node_idx
                    );
                    self.viewport_state.camera_current = WorldPos::ZERO;
                    self.viewport_state.camera_target = WorldPos::ZERO;
                    self.viewport_state.camera_anim = None;
                }
            } else {
                // No tracked node or no stored viewport position, reset camera to origin
                self.viewport_state.camera_current = WorldPos::ZERO;
                self.viewport_state.camera_target = WorldPos::ZERO;
                self.viewport_state.camera_anim = None;
            }

            // Load partitions needed for current camera view
            if let Err(e) = self.ensure_camera_coverage() {
                eprintln!("Error ensuring camera coverage during contract: {}", e);
            }

            // Force viewport graph rebuild - let update_viewport_graph() handle cursor restoration
            self.viewport_graph = ViewportGraph::empty();
            self.rebuild_needed = true;
        }
    }

    /// Get current vertex spacing from partition controller
    fn get_current_vertex_spacing(&self) -> f64 {
        self.partition_controller.get_vertex_spacing()
    }

    /// Handle keyboard events for graph navigation and control
    ///
    /// Returns Some(true) for normal exit, Some(false) for abort, None to continue
    pub fn handle_key_event(&mut self, key: KeyEvent) -> Option<bool> {
        match key.code {
            KeyCode::Esc => Some(false),                       // abort
            KeyCode::Enter | KeyCode::Char('q') => Some(true), // normal exit

            // Graph navigation controls - move cursor with node awareness
            KeyCode::Left | KeyCode::Char('h') => {
                self.move_cursor_horizontal(-1);
                None
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.move_cursor_horizontal(1);
                None
            }
            // Note: In world coordinates, Y increases upward
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_cursor_vertical(1); // Move up = positive Y
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_cursor_vertical(-1); // Move down = negative Y
                None
            }

            // Zoom/Scale controls
            KeyCode::Char('+') | KeyCode::Char('=') => {
                if key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::SHIFT)
                {
                    // Shift+'+': Direct disperse
                    self.disperse();
                } else {
                    // Regular zoom in
                    self.zoom_in();
                }
                None
            }
            KeyCode::Char('-') => {
                if key
                    .modifiers
                    .contains(crossterm::event::KeyModifiers::SHIFT)
                {
                    // Shift+'-': Direct contract
                    self.contract();
                } else {
                    // Regular zoom out
                    self.zoom_out();
                }
                None
            }

            // Export to DOT
            KeyCode::Char('d') | KeyCode::Char('D') => {
                // Export unified layout graph to DOT format
                // This will be handled by the caller who has access to file system
                // Return a special signal that indicates DOT export was requested
                // We use None here but the caller should check for this key specifically
                None
            }

            _ => None,
        }
    }

    // TODO: rename this function and retain only the node positioning logic
    // (viewport graphs are made on demand instead of receiving updates)
    /// Update the viewport graph with nodes and edges visible in the given viewport
    pub fn update_viewport_graph(
        &mut self,
        viewport: crate::geometry::BigRect<i64>,
        detail_level: VisualDetail,
    ) -> Result<(), String> {
        // Sanity check - log if detail levels don't match
        if self.detail_level != detail_level {
            log::debug!(
                "Detail level mismatch: controller has {:?}, requested {:?}",
                self.detail_level,
                detail_level
            );
        }

        if self.detail_level != detail_level {
            self.detail_level = detail_level;
            self.partition_controller.set_detail_level(detail_level);
        }

        // If rebuild is needed (e.g., after disperse/contract), use camera's actual viewport
        // This handles coordinate system changes when anchor partition shifts
        let viewport = if self.rebuild_needed {
            self.viewport_state.visible_world_rect()
        } else {
            viewport
        };

        if self.viewport_state.cursor.enabled
            && self.viewport_state.cursor.node_domain_idx.is_some()
            && self.viewport_state.cursor.stored_viewport_pos.is_some()
        {
            let tracked_node = self.viewport_state.cursor.node_domain_idx.unwrap();
            let viewport_pos = self.viewport_state.cursor.stored_viewport_pos.unwrap();

            // Find the tracked node's position in the new layout system
            let node_world_pos = match self.find_domain_node_world_position(tracked_node) {
                Some(pos) => pos,
                None => {
                    log::debug!(
                        "Cannot find tracked node {:?} in new layout during zoom/detail change. \
                         Stored viewport position: ({}, {}), Camera: {:?}. \
                         Clearing cursor tracking.",
                        tracked_node,
                        viewport_pos.x,
                        viewport_pos.y,
                        self.viewport_state.camera_current
                    );
                    // Clear cursor tracking and continue without restoration
                    self.viewport_state.cursor.stored_viewport_pos = None;
                    self.viewport_state.cursor.node_domain_idx = None;
                    self.last_rebuild_camera_center = self.viewport_state.camera_current;
                    self.rebuild_needed = false;
                    return Ok(());
                }
            };

            {
                trace!(
                    "Found tracked node {:?} at world position {:?}",
                    tracked_node, node_world_pos
                );

                // Add the cursor offset to get the actual cursor position
                let cursor_offset = self.viewport_state.cursor.node_offset;
                let new_cursor_world_pos = WorldPos::new(
                    node_world_pos.x + cursor_offset.x,
                    node_world_pos.y + cursor_offset.y,
                );
                trace!(
                    "Cursor world position with offset {:?}: {:?}",
                    cursor_offset, new_cursor_world_pos
                );

                // Verify the adjustment worked - cursor should now be at exact viewport position
                if let Some(actual_vp) = self.viewport_state.world_to_viewport(new_cursor_world_pos)
                {
                    trace!(
                        "After adjustment, cursor appears at viewport ({}, {})",
                        actual_vp.x, actual_vp.y
                    );

                    // CRITICAL: Always verify camera adjustment worked in all builds
                    if actual_vp != viewport_pos {
                        trace!(
                            "BUG: Camera adjustment failed during disperse/contract! \
                             Cursor at viewport ({}, {}) instead of expected ({}, {}). \
                             Off by: ({}, {}). \
                             This will cause cursor to jump position. \
                             Camera: {:?}, Cursor world: {:?}",
                            actual_vp.x,
                            actual_vp.y,
                            viewport_pos.x,
                            viewport_pos.y,
                            actual_vp.x as i32 - viewport_pos.x as i32,
                            actual_vp.y as i32 - viewport_pos.y as i32,
                            self.viewport_state.camera_current,
                            new_cursor_world_pos
                        );
                    }
                } else {
                    trace!(
                        "BUG: Cursor not visible in viewport after camera adjustment! \
                         This should never happen if camera adjustment worked correctly. \
                         Camera: {:?}, Cursor world: {:?}, Expected viewport: ({}, {})",
                        self.viewport_state.camera_current,
                        new_cursor_world_pos,
                        viewport_pos.x,
                        viewport_pos.y
                    );
                }
            }
        }

        // TODO: handle the partition controller outside of the ViewportGraph constructor (and
        // rename it to layoutgraph or something like that).
        // Or alternatively, put a constructor in graph_controller that takes care of the
        // partition and viewport coverage. That way we don't have to lie about the size of the
        // viewport we're trying to render. It would require moving all the cursor code out as
        // well.
        // Build larger than we need right now, because we don't trigger updates on every scroll
        let covered_rect = viewport.resize(2.0);
        self.viewport_graph =
            ViewportGraph::new(covered_rect, &mut self.partition_controller, detail_level)?;

        // ASSERTION: If we have a tracked node with stored position, it MUST be in viewport graph
        if self.viewport_state.cursor.enabled
            && self.viewport_state.cursor.node_domain_idx.is_some()
            && self.viewport_state.cursor.stored_viewport_pos.is_some()
        {
            let tracked_node = self.viewport_state.cursor.node_domain_idx.unwrap();
            let mut found_in_viewport = false;

            for node_data in self.viewport_graph.nodes.values() {
                if let crate::layout::NodeRole::Data(domain_idx) = &node_data.role {
                    if *domain_idx == tracked_node {
                        found_in_viewport = true;
                        break;
                    }
                }
            }

            if !found_in_viewport {
                trace!(
                    "BUG: Tracked node {:?} not found in viewport graph after camera adjustment! \
                     This means cursor restoration will fail. \
                     Stored viewport pos: {:?}, Camera: {:?}, Covered rect: {:?}",
                    tracked_node,
                    self.viewport_state.cursor.stored_viewport_pos,
                    self.viewport_state.camera_current,
                    covered_rect
                );
            }
        }

        // Handle cursor restoration/initialization
        if self.viewport_state.cursor.enabled {
            if let Some(tracked_node) = self.viewport_state.cursor.node_domain_idx {
                // Try to restore cursor position using stored state
                let node_offset = self.viewport_state.cursor.node_offset;

                if self.viewport_state.cursor.stored_viewport_pos.is_some() {
                    // The camera has already been adjusted to position the node correctly.
                    // Now we just need to restore the cursor to the node.

                    let stored_viewport_pos = self.viewport_state.cursor.stored_viewport_pos;

                    // CRITICAL: If we have a stored viewport position, restoration MUST succeed
                    // Failing to restore means the camera adjustment or viewport graph is broken
                    if let Err(e) = self.restore_cursor_to_tracked_node(tracked_node, node_offset) {
                        trace!(
                            "Cursor restoration failed during zoom/detail change: {}. \
                             This should never happen - stored viewport pos: {:?}, \
                             tracked node: {:?}, camera: {:?}",
                            e,
                            stored_viewport_pos,
                            tracked_node,
                            self.viewport_state.camera_current
                        );
                    }

                    // Log cursor restoration results
                    self.log_cursor_restoration("cursor restoration", stored_viewport_pos);

                    // Verify cursor is at exact viewport position after restoration
                    if let Some(expected_vp) = stored_viewport_pos {
                        if let Some(actual_vp) = self
                            .viewport_state
                            .world_to_viewport(self.viewport_state.cursor.current)
                        {
                            if actual_vp != expected_vp {
                                log::debug!(
                                    "Cursor viewport position restoration mismatch during zoom/detail change: \
                                     expected viewport ({}, {}), got ({}, {}). \
                                     Cursor world pos: {:?}, Camera: {:?}",
                                    expected_vp.x,
                                    expected_vp.y,
                                    actual_vp.x,
                                    actual_vp.y,
                                    self.viewport_state.cursor.current,
                                    self.viewport_state.camera_current
                                );
                            }
                        } else {
                            log::debug!(
                                "Cursor viewport position restoration failed during zoom/detail change: \
                                 cursor at world {:?} is not visible in viewport. \
                                 Expected viewport position: ({}, {}), Camera: {:?}",
                                self.viewport_state.cursor.current,
                                expected_vp.x,
                                expected_vp.y,
                                self.viewport_state.camera_current
                            );
                        }
                    }

                    // Clear the stored position regardless of success/failure
                    self.viewport_state.cursor.stored_viewport_pos = None;
                } else {
                    // No stored viewport position, just try to restore cursor position on the same node
                    // This handles cases where viewport graph was rebuilt but not due to layout changes
                    let current_viewport_offset =
                        self.viewport_state.cursor.current - self.viewport_state.camera_current;
                    let viewport_pos = ViewportPos::new(
                        cmp::max(current_viewport_offset.x, 0) as u16,
                        cmp::max(current_viewport_offset.y, 0) as u16,
                    );
                    if self
                        .restore_cursor_after_viewport_update(
                            tracked_node,
                            node_offset,
                            viewport_pos,
                        )
                        .is_err()
                    {
                        // If restoration fails, just re-initialize
                        self.initialize_cursor();
                    }
                }
            } else {
                // No tracked node, initialize cursor for the first time
                self.initialize_cursor();
            }
        }

        // Update rebuild tracking
        self.last_rebuild_camera_center = self.viewport_state.camera_current;
        self.rebuild_needed = false; // Clear the rebuild flag after successful rebuild

        Ok(())
    }

    /// Get a reference to the currently available viewport graph
    pub fn get_viewport_graph(&self) -> &ViewportGraph {
        &self.viewport_graph
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ratatui::{buffer::Buffer, layout::Rect, style::Style};

    use super::*;
    use crate::geometry::WorldRect;

    #[test]
    fn test_coordinate_conversions() {
        let mut state = ViewportState::new();
        state.camera_current = WorldPos::new(100, 100);
        state.viewport_bounds = Rect::new(0, 0, 20, 10);

        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);
        let writer = WorldBuffer::new(&mut buffer, &state);

        // Test camera origin calculation
        let origin = state.camera_origin_world();
        assert_eq!(origin, WorldPos::new(90, 95)); // 100 - 10, 100 - 5

        // Test world to viewport conversion
        let world_pos = WorldPos::new(95, 98);
        let viewport_pos = writer.world_to_viewport(world_pos);
        assert_eq!(viewport_pos, Some(ViewportPos::new(5, 3)));

        // Test out of bounds
        let out_of_bounds = WorldPos::new(80, 80);
        assert_eq!(writer.world_to_viewport(out_of_bounds), None);

        // Test viewport to world conversion
        let vp = ViewportPos::new(5, 3);
        let world = writer.viewport_to_world(vp);
        assert_eq!(world, WorldPos::new(95, 98));
    }

    #[test]
    fn test_focus_management() {
        let mut state = ViewportState::new();
        assert!(state.has_focus); // Focus is enabled by default for keyboard input

        state.focus();
        assert!(state.has_focus);

        // Test scroll with focus
        state.handle_mouse_scroll(5, 5, Duration::from_millis(100));
        assert!(state.camera_anim.is_some());

        state.blur();
        assert!(!state.has_focus);

        // Reset animation
        state.camera_anim = None;

        // Test scroll without focus (should be ignored)
        state.handle_mouse_scroll(5, 5, Duration::from_millis(100));
        assert!(state.camera_anim.is_none());
    }

    #[test]
    fn test_world_bounds_clamping() {
        let mut state = ViewportState::new();
        state.world_bounds = Some(WorldRect::from_corners(
            WorldPos::new(-10, -10),
            WorldPos::new(10, 10),
        ));

        // Try to set cursor outside bounds
        state.set_cursor_target(WorldPos::new(20, 20), Duration::from_millis(100));

        // Should be clamped to (10, 10)
        assert_eq!(state.cursor.target, WorldPos::new(10, 10));
    }

    #[test]
    fn test_panning_behavior() {
        let mut state = ViewportState::new();
        // Set zone fractions to enable multizone logic
        state.dead_zone_fraction = (0.2, 0.2);
        state.soft_zone_fraction = (0.4, 0.4);
        state.focus(); // Enable focus to allow scrolling
        let viewport_size = (10_u16, 10_u16);

        // Initial state: not panning
        assert!(!state.panning);

        // Scroll should enable panning mode
        state.handle_mouse_scroll(5, 5, Duration::from_millis(100));
        assert!(state.panning);
        assert!(state.camera_anim.is_some());

        // While panning, multizone logic should be disabled
        // Place cursor in what would normally be the soft zone
        state.cursor.current = WorldPos::new(3, 0);
        let _camera_before = state.camera_current;

        // Update should not trigger multizone camera movement because panning is true
        state.update(Duration::from_millis(16), viewport_size);

        // Camera should only move due to the scroll animation, not multizone logic
        // (The camera animation from scrolling should be active)
        assert!(state.camera_anim.is_some()); // Still animating from scroll

        // Clicking should disable panning
        state.stop_panning();
        assert!(!state.panning);

        // Now multizone logic should work again when camera animation finishes
        state.camera_anim = None; // Simulate animation completion
        let camera_before_multizone = state.camera_current;
        state.update(Duration::from_millis(16), viewport_size);

        // Since cursor is positioned outside dead zone, camera should move
        // (This tests that multizone logic is reactivated after panning ends)
        assert!(state.camera_anim.is_some() || state.camera_current != camera_before_multizone);
    }

    #[test]
    fn test_world_buffer_writer_coordinate_conversion() {
        let mut state = ViewportState::new();
        state.camera_current = WorldPos::new(10, 10);

        let area = Rect::new(0, 0, 10, 5);
        let mut buffer = Buffer::empty(area);

        state.viewport_bounds = area;

        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Test visibility check - use the origin which should be at center
        let world_pos = WorldPos::new(10, 10); // Should be at center of viewport
        assert!(state.is_visible(world_pos));

        // Test world coordinates that should be out of viewport
        let out_of_bounds = WorldPos::new(100, 100);
        assert!(!state.is_visible(out_of_bounds));

        // Test writing a character
        let result = writer.set_char(world_pos, 'X');
        assert!(result, "Character should be written successfully");
    }

    #[test]
    fn test_world_buffer_writer_string_operations() {
        let mut state = ViewportState::new();
        let area = Rect::new(0, 0, 20, 10);
        let mut buffer = Buffer::empty(area);

        // Set up the state similar to the working coordinate test
        state.viewport_bounds = area;
        state.camera_current = WorldPos::new(10, 5); // Half the viewport size

        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Test horizontal string writing - world (5,5) should map to viewport (0,0)
        let test_string = "Hello";
        let world_pos = WorldPos::new(5, 5); // Should be visible
        let written = writer.set_string(world_pos, test_string);
        assert_eq!(written, 5, "All characters should be written");

        let read_string = writer.get_string(world_pos, 10);
        assert!(
            read_string.starts_with(test_string),
            "Should get back the written string"
        );

        // Write a vertical string and read it back
        let vertical_string = "VERT";
        let vertical_pos = WorldPos::new(15, 6); // Should be visible
        let written_vertical = writer.set_string_vertical(vertical_pos, vertical_string);
        assert_eq!(
            written_vertical, 4,
            "All vertical characters should be written"
        );

        let read_vertical = writer.get_string_vertical(vertical_pos, 10);
        assert!(
            read_vertical.starts_with(vertical_string),
            "Should read back the written vertical string"
        );

        // Test reading beyond written content - make sure we're not reading beyond what was written
        let partial_read = writer.get_string(WorldPos::new(7, 5), 5); // Read from position (7,5) with max 5 chars
        assert!(
            partial_read.len() <= 5,
            "Should not read more than requested length"
        );
    }

    #[test]
    fn test_world_buffer_writer_clipping() {
        let mut state = ViewportState::new();
        let area = Rect::new(0, 0, 5, 5);
        let mut buffer = Buffer::empty(area);

        // Set the viewport bounds in the state to match our test area
        state.viewport_bounds = area;

        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Try to write outside viewport bounds
        let result = writer.set_char(WorldPos::new(100, 100), 'X');
        assert!(!result, "Character outside bounds should not be written");

        // Try to write a string that goes beyond viewport
        let written = writer.set_string(WorldPos::ZERO, "Hello World");
        assert!(
            written < 11,
            "String should be clipped at viewport boundary"
        );
    }

    #[test]
    fn test_world_buffer_writer_read_write_consistency() {
        let mut state = ViewportState::new();
        let area = Rect::new(0, 0, 8, 8);
        let mut buffer = Buffer::empty(area);

        // Set up the state similar to the working coordinate test
        state.viewport_bounds = area;
        state.camera_current = WorldPos::new(4, 4); // Half the viewport size

        let mut writer = WorldBuffer::new(&mut buffer, &state);

        // Fill a small rectangle and read it back - use positions that should be visible
        let rect = WorldRect::from_corners(WorldPos::new(1, 1), WorldPos::new(3, 3));
        let fill_char = '#';
        let filled = writer.fill_rect_styled(rect, fill_char, Style::default().fg(Color::Blue));
        assert_eq!(filled, 9, "Should fill 3x3 rectangle (9 characters)");

        // Read back all the filled positions
        for y in rect.min.y..=rect.max.y {
            for x in rect.min.x..=rect.max.x {
                let world_pos = WorldPos::new(x, y);
                let (read_char, read_style) = writer.get_char_styled(world_pos).unwrap();
                assert_eq!(read_char, fill_char, "Should read back the fill character");
                assert_eq!(
                    read_style.fg,
                    Some(Color::Blue),
                    "Should read back the fill style"
                );
            }
        }

        // Check that positions outside the rectangle are not filled
        let outside_pos = WorldPos::new(4, 4); // Just outside the rectangle
        let outside_char = writer.get_char(outside_pos).unwrap();
        assert_eq!(
            outside_char, ' ',
            "Position outside rectangle should remain empty"
        );
    }

    #[test]
    fn test_disperse_functionality() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use petgraph::graph::NodeIndex;
        use ratatui::layout::Rect;

        use crate::{layout::VisualDetail, plotter::NodeSizer, testing::mocks::MockDomainGraph};

        // Create a simple NodeSizer for our domain graph
        #[derive(Clone)]
        struct TestNodeSizer;

        impl NodeSizer<&MockDomainGraph> for TestNodeSizer {
            fn get_node_size(&self, _node: &NodeIndex, _scale: VisualDetail) -> (u64, u64) {
                (1, 1)
            }

            fn get_dummy_size(&self) -> (u64, u64) {
                (1, 1)
            }
        }

        // Create a simple domain graph for testing
        let mut domain_graph = MockDomainGraph::new();
        let _a = domain_graph.add_node(());
        let _b = domain_graph.add_node(());
        let _c = domain_graph.add_node(());

        let node_sizer = TestNodeSizer;

        // Create GraphController with the test graph (use reference)
        let mut controller = GraphController::new(&domain_graph, node_sizer);

        // Set up viewport bounds
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 20, 10);

        // Initialize the controller by loading layouts
        let _ = controller.ensure_camera_coverage();

        println!(
            "Initial vertex spacing: {}",
            controller.get_current_vertex_spacing()
        );

        // Simulate hitting '+' key (zoom in / disperse)
        let plus_key = KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE);

        // Try multiple times to replicate "hitting + a few times"
        for i in 1..=3 {
            println!("Attempt {} to disperse", i);

            match controller.handle_key_event(plus_key) {
                None => {
                    println!("Key handled successfully on attempt {}", i);
                    println!(
                        "New vertex spacing: {}",
                        controller.get_current_vertex_spacing()
                    );
                }
                Some(_) => {
                    println!("Key event returned exit signal on attempt {}", i);
                    break;
                }
            }
        }

        // If we get here without panicking, the test passes
        println!("Test completed successfully");
    }
}
