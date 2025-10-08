use std::{collections::HashSet, hash::Hash};

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
pub use crate::viewport_state::{ViewportState, WorldBuffer};
use crate::{
    cursor_v2::ViewportCursor,
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

    /// Viewport state managing camera, animations, and viewport bounds
    pub viewport_state: ViewportState,

    /// Cursor state managing cursor position and tracking
    pub cursor: ViewportCursor,

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
            cursor: ViewportCursor::default(),
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
        if detail_level == self.detail_level {
            return;
        }

        trace!(
            "set_detail_level: changing from {:?} to {:?}",
            self.detail_level, detail_level
        );

        // Get OLD cursor world position before layout change
        let w1 = if self.cursor.is_enabled() {
            self.cursor.to_world_pos(&self.viewport_graph)
        } else {
            None
        };

        // Change detail level
        self.detail_level = detail_level;
        self.partition_controller.set_detail_level(detail_level);

        // Reset camera to origin for new coordinate system
        self.viewport_state.camera_current = WorldPos::ZERO;
        self.viewport_state.camera_target = WorldPos::ZERO;
        self.viewport_state.camera_anim = None;

        // Clear all layouts to force recalculation with new detail level
        self.partition_controller.clear_all_layouts();
        self.loaded_partitions.clear();

        // Reload anchor partition (usually 0) for new coordinate system
        if let Err(e) = self.partition_controller.set_anchor_partition(0) {
            eprintln!(
                "Warning: Failed to set anchor partition during detail level change: {}",
                e
            );
        }

        // Load partitions for current camera view
        if let Err(e) = self.ensure_camera_coverage() {
            eprintln!(
                "Error ensuring camera coverage during detail level change: {}",
                e
            );
        }

        // Rebuild viewport graph with new detail level
        let camera_rect = self.viewport_state.camera_rect();
        if let Err(e) = self.rebuild_viewport_graph(camera_rect, detail_level) {
            eprintln!("Error rebuilding viewport graph: {}", e);
        }

        // Compute NEW cursor world position from updated layout
        if let Some(old_world) = w1 {
            if let Some(new_world) = self.cursor.to_world_pos(&self.viewport_graph) {
                // Calculate camera delta and adjust
                let delta = new_world - old_world;
                self.viewport_state.camera_current = self.viewport_state.camera_current + delta;
                self.viewport_state.camera_target = self.viewport_state.camera_target + delta;

                trace!(
                    "set_detail_level: adjusted camera by delta ({}, {}) to maintain cursor position",
                    delta.x, delta.y
                );
            }
        }

        trace!("set_detail_level: complete");
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
        trace!("disperse: starting");

        // Get OLD cursor world position before layout change
        let w1 = if self.cursor.is_enabled() {
            self.cursor.to_world_pos(&self.viewport_graph)
        } else {
            None
        };

        // Look up which partition contains the tracked node BEFORE clearing layouts
        let tracked_node_partition = if let Some(node_idx) = self.cursor.get_node_idx() {
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
            "disperse: vertex spacing now {}",
            self.partition_controller.get_vertex_spacing()
        );

        // Clear all layouts to force complete recalculation with new spacing
        self.partition_controller.clear_all_layouts();
        self.loaded_partitions.clear();

        // Set the tracked node's partition as anchor
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

        // Reset camera to origin for new coordinate system
        self.viewport_state.camera_current = WorldPos::ZERO;
        self.viewport_state.camera_target = WorldPos::ZERO;
        self.viewport_state.camera_anim = None;

        // Load partitions needed for current camera view
        if let Err(e) = self.ensure_camera_coverage() {
            eprintln!("Error ensuring camera coverage during disperse: {}", e);
        }

        // Rebuild viewport graph
        let camera_rect = self.viewport_state.camera_rect();
        if let Err(e) = self.rebuild_viewport_graph(camera_rect, self.detail_level) {
            eprintln!("Error rebuilding viewport graph: {}", e);
        }

        // Compute NEW cursor world position and adjust camera by delta
        if let Some(old_world) = w1 {
            if let Some(new_world) = self.cursor.to_world_pos(&self.viewport_graph) {
                let delta = new_world - old_world;
                self.viewport_state.camera_current = self.viewport_state.camera_current + delta;
                self.viewport_state.camera_target = self.viewport_state.camera_target + delta;

                trace!(
                    "disperse: adjusted camera by delta ({}, {}) to maintain cursor position",
                    delta.x, delta.y
                );
            }
        }

        trace!("disperse: complete");
    }

    /// Decrease vertex spacing and refresh layouts
    pub fn contract(&mut self) {
        let current_spacing = self.get_current_vertex_spacing();
        if current_spacing <= VERTEX_SPACING_DEFAULT {
            return; // Already at minimum spacing
        }

        trace!("contract: starting");

        // Get OLD cursor world position before layout change
        let w1 = if self.cursor.is_enabled() {
            self.cursor.to_world_pos(&self.viewport_graph)
        } else {
            None
        };

        // Look up which partition contains the tracked node BEFORE clearing layouts
        let tracked_node_partition = if let Some(node_idx) = self.cursor.get_node_idx() {
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
        trace!("contract: vertex spacing now {}", new_spacing);

        // Clear all layouts to force complete recalculation with new spacing
        self.partition_controller.clear_all_layouts();
        self.loaded_partitions.clear();

        // Set the tracked node's partition as anchor
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

        // Reset camera to origin for new coordinate system
        self.viewport_state.camera_current = WorldPos::ZERO;
        self.viewport_state.camera_target = WorldPos::ZERO;
        self.viewport_state.camera_anim = None;

        // Load partitions needed for current camera view
        if let Err(e) = self.ensure_camera_coverage() {
            eprintln!("Error ensuring camera coverage during contract: {}", e);
        }

        // Rebuild viewport graph
        let camera_rect = self.viewport_state.camera_rect();
        if let Err(e) = self.rebuild_viewport_graph(camera_rect, self.detail_level) {
            eprintln!("Error rebuilding viewport graph: {}", e);
        }

        // Compute NEW cursor world position and adjust camera by delta
        if let Some(old_world) = w1 {
            if let Some(new_world) = self.cursor.to_world_pos(&self.viewport_graph) {
                let delta = new_world - old_world;
                self.viewport_state.camera_current = self.viewport_state.camera_current + delta;
                self.viewport_state.camera_target = self.viewport_state.camera_target + delta;

                trace!(
                    "contract: adjusted camera by delta ({}, {}) to maintain cursor position",
                    delta.x, delta.y
                );
            }
        }

        trace!("contract: complete");
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

    /// Rebuild the viewport graph to show nodes and edges visible in the current camera view.
    /// This is a simple graph builder - all camera and cursor positioning must be done beforehand.
    ///
    /// # Design
    /// This function has a single responsibility: build the viewport graph from the current
    /// camera view. It does NOT handle:
    /// - Camera positioning (use update_camera() before calling)
    /// - Cursor restoration (use preserve_cursor_during_layout_change() before calling)
    /// - Detail level changes (caller should handle)
    ///
    /// # Parameters
    /// - viewport: The world-space rectangle to build the graph for
    /// - detail_level: The visual detail level to use
    pub fn rebuild_viewport_graph(
        &mut self,
        viewport: crate::geometry::BigRect<i64>,
        detail_level: VisualDetail,
    ) -> Result<(), String> {
        trace!(
            "rebuild_viewport_graph: viewport rect x:{}-{}, y:{}-{}, detail:{:?}",
            viewport.min.x, viewport.max.x, viewport.min.y, viewport.max.y, detail_level
        );

        // Sync detail level if needed
        if self.detail_level != detail_level {
            self.detail_level = detail_level;
            self.partition_controller.set_detail_level(detail_level);
        }

        // Build viewport graph with a buffer around the requested rect
        // Buffer factor of 2.0 reduces rebuild frequency during small camera movements
        let covered_rect = viewport.resize(2.0);

        trace!(
            "rebuild_viewport_graph: building with covered rect x:{}-{}, y:{}-{}",
            covered_rect.min.x, covered_rect.max.x, covered_rect.min.y, covered_rect.max.y
        );

        self.viewport_graph =
            ViewportGraph::new(covered_rect, &mut self.partition_controller, detail_level)?;

        // Update rebuild tracking
        self.last_rebuild_camera_center = self.viewport_state.camera_current;
        self.rebuild_needed = false;

        trace!(
            "rebuild_viewport_graph: complete - {} nodes, {} edges",
            self.viewport_graph.node_count(),
            self.viewport_graph.edge_count()
        );

        Ok(())
    }

    /// Get a reference to the currently available viewport graph
    pub fn get_viewport_graph(&self) -> &ViewportGraph {
        &self.viewport_graph
    }

    /// Update camera position to keep a focal point (in world coordinates) at a specific viewport position.
    /// This is the key transformation for maintaining visual stability during layout changes.
    ///
    /// # Coordinate System
    /// - Camera represents the CENTER of the viewport in world coordinates
    /// - Viewport origin (top-left) = camera - (width/2, height/2)
    /// - Transformation: viewport_pos = world_pos - camera + (width/2, height/2)
    /// - Inverse: camera = world_pos - viewport_pos + (width/2, height/2)
    ///
    /// # Parameters
    /// - focal_world: World position of the point we want to keep stable
    /// - focal_viewport: Viewport position where we want the focal point to appear
    pub fn update_camera(&mut self, focal_world: WorldPos, focal_viewport: ViewportPos) {
        let half_width = self.viewport_state.viewport_bounds.width as i64 / 2;
        let half_height = self.viewport_state.viewport_bounds.height as i64 / 2;

        let camera_x = focal_world.x - focal_viewport.x as i64 + half_width;
        let camera_y = focal_world.y - focal_viewport.y as i64 + half_height;

        self.viewport_state.camera_current = WorldPos::new(camera_x, camera_y);
        self.viewport_state.camera_target = WorldPos::new(camera_x, camera_y);
        self.viewport_state.camera_anim = None;

        trace!(
            "update_camera: focal_world={:?}, focal_viewport=({}, {}), camera={:?}",
            focal_world, focal_viewport.x, focal_viewport.y, self.viewport_state.camera_current
        );
    }

    // Legacy cursor helper methods removed - now handled by ViewportCursor + camera delta adjustment
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
