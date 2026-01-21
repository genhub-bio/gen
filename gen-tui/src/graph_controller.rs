use std::hash::Hash;

use crossterm::event::{KeyCode, KeyEvent};
use gen_sugiyama::{self, VERTEX_SPACING_DEFAULT};
use log::trace;
use petgraph::{
    graph::NodeIndex,
    graphmap::DiGraphMap,
    visit::{
        EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeCount, NodeIndexable, Visitable,
    },
};
use ratatui::style::Color;

// Re-export the core module types
pub use crate::viewport_state::{ViewportState, WorldBuffer};
use crate::{
    cursor::Cursor,
    geometry::{BigRect, ViewportPos, WorldPos, WorldRect},
    layout::{NodeRole, PartitionLayout, VisualDetail},
    partition_controller::{ControllerConfig, PartitionController},
    partition_table::PartitionConfig,
    plotter::NodeSizer,
    viewport_graph::CroppedGraph,
};

/// Combined configuration for the entire graph widget system
#[derive(Debug, Clone, Default)]
pub struct GraphConfig {
    /// Partition behavior configuration  
    pub partition: PartitionConfig,
    /// Controller memory management configuration
    pub controller: ControllerConfig,
    /// Layout algorithm configuration
    pub layout: gen_sugiyama::Config,
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
    pub cursor: Cursor,

    /// Current detail level for visualization
    detail_level: VisualDetail,

    /// Partition controller for managing graph partitioning and layout (without its own viewport state)
    pub partition_controller: PartitionController<G, S>,
    /// Cropped graph containing only visible nodes and edges
    pub viewport_graph: CroppedGraph,
    /// Camera position when viewport graph was last rebuilt
    /// (for hysteresis to control rebuild frequency)
    last_rebuild_camera_center: WorldPos,
    /// Flag indicating that the viewport graph needs to be rebuilt
    rebuild_needed: bool,

    /// Path highlighting: list of subgraphs with their associated colors
    path_highlights: Vec<(DiGraphMap<NodeIndex, ()>, Color)>,
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
            cursor: Cursor::default(),
            detail_level: VisualDetail::Truncated, // Default detail level
            partition_controller,
            viewport_graph: CroppedGraph::empty(),
            last_rebuild_camera_center: WorldPos::ZERO,
            rebuild_needed: true, // Start with a rebuild required
            path_highlights: Vec::new(),
        };

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

    /// Export the viewport graph to DOT format for visualization
    pub fn export_to_dot(&self, filename: &str) -> Result<(), std::io::Error> {
        crate::dot_export::export_to_dot(&self.viewport_graph, filename)
    }

    /// Ensure a partition is loaded for rendering
    pub fn ensure_partition_loaded(&mut self, partition_idx: usize) -> Result<(), String> {
        self.partition_controller
            .ensure_partition_loaded(partition_idx)
    }

    /// Check camera movement and viewport bounds changes to determine if a rebuild is needed
    pub fn detect_motion(&self) -> bool {
        let threshold_x = self.viewport_state.viewport_bounds.width as i64 / 2;
        let threshold_y = self.viewport_state.viewport_bounds.height as i64 / 2;

        let current_camera = self.viewport_state.camera_current;
        let movement_x = (current_camera.x - self.last_rebuild_camera_center.x).abs();
        let movement_y = (current_camera.y - self.last_rebuild_camera_center.y).abs();

        movement_x > threshold_x || movement_y > threshold_y
    }

    /// Force the viewport graph to be rebuilt on the next update.
    /// This is useful when external changes require a full rebuild.
    pub fn trigger_rebuild(&mut self) {
        self.rebuild_needed = true;
    }

    /// Check if a rebuild is currently needed
    pub fn needs_rebuild(&self) -> bool {
        self.rebuild_needed
    }

    // TODO: maintain one cache keyed on partition idx and detail level (renamed to layoutkey
    // for example
    pub fn set_detail_level(&mut self, detail_level: VisualDetail) {
        // Only proceed if detail level is actually changing
        if detail_level == self.detail_level {
            return;
        }

        trace!(
            "set_detail_level: changing from {:?} to {:?}",
            self.detail_level, detail_level
        );

        // Change detail level
        self.detail_level = detail_level;
        self.partition_controller.set_detail_level(detail_level);
        self.rebuild_needed = true;
        trace!("set_detail_level: complete");
    }

    /// Get the current level of detail
    pub fn get_detail_level(&self) -> VisualDetail {
        self.detail_level
    }

    /// Get a reference to the path highlights
    pub fn get_path_highlights(&self) -> &[(DiGraphMap<NodeIndex, ()>, Color)] {
        &self.path_highlights
    }

    /// Set a path highlight with a specific color
    ///
    /// # Parameters
    /// - color: Color for highlighting the path
    /// - path_nodes: Sequence of nodes that form the path
    pub fn set_path_highlight(&mut self, color: Color, path_nodes: Vec<G::NodeId>) {
        let mut path_graph = DiGraphMap::<NodeIndex, ()>::new();

        // Convert G::NodeId to NodeIndex and add all nodes to the path graph
        let node_indices: Vec<NodeIndex> = path_nodes
            .iter()
            .map(|node_id| NodeIndex::new(<G as NodeIndexable>::to_index(&self.graph, *node_id)))
            .collect();

        for node_idx in &node_indices {
            path_graph.add_node(*node_idx);
        }

        // Add edges between consecutive nodes in the path
        for window in node_indices.windows(2) {
            if let [src, tgt] = window {
                path_graph.add_edge(*src, *tgt, ());
            }
        }

        self.path_highlights.push((path_graph, color));
    }

    /// Check if a specific color has path highlighting
    pub fn has_path_highlight(&self, color: &Color) -> bool {
        self.path_highlights.iter().any(|(_, c)| c == color)
    }

    /// Clear path highlighting for a specific color
    pub fn clear_path_highlight(&mut self, color: &Color) {
        self.path_highlights.retain(|(_, c)| c != color);
        self.trigger_rebuild();
    }

    /// Clear all path highlights
    pub fn clear_all_path_highlights(&mut self) {
        self.path_highlights.clear();
        self.trigger_rebuild();
    }

    /// Calculate total bounds needed to display all partitions
    #[allow(clippy::type_complexity)]
    pub fn calculate_total_bounds(&mut self) -> Result<BigRect<i64>, String> {
        self.partition_controller.calculate_total_bounds()
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
            "ensure_camera_coverage: loading partitions for rect of {}x{} (coord x: {} - {}, y: {} - {})",
            camera_rect.width(),
            camera_rect.height(),
            camera_rect.left(),
            camera_rect.right(),
            camera_rect.bottom(),
            camera_rect.top(),
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
                    if let Some(layout_node) = layout.graph.node_weight(layout_node_idx)
                        && let NodeRole::Data(layout_domain_idx) = &layout_node.role
                        && *layout_domain_idx == domain_idx
                    {
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
        // Increase vertex spacing
        self.partition_controller.adjust_vertex_spacing(2.0);
        trace!(
            "disperse: vertex spacing now {}",
            self.partition_controller.get_vertex_spacing()
        );

        // Clear all layouts to force complete recalculation with new spacing
        self.partition_controller.clear_all_layouts();
        self.rebuild_needed = true;
    }

    /// Decrease vertex spacing and refresh layouts
    pub fn contract(&mut self) {
        let current_spacing = self.get_current_vertex_spacing();
        if current_spacing <= VERTEX_SPACING_DEFAULT {
            return; // Already at minimum spacing
        }

        // Decrease vertex spacing
        self.partition_controller.adjust_vertex_spacing(-2.0);
        trace!(
            "contract: vertex spacing now {}",
            self.partition_controller.get_vertex_spacing()
        );

        // Clear all layouts to force complete recalculation with new spacing
        self.partition_controller.clear_all_layouts();
        self.rebuild_needed = true;
    }

    /// Get current vertex spacing from partition controller
    fn get_current_vertex_spacing(&self) -> f64 {
        self.partition_controller.get_vertex_spacing()
    }

    /// Initialize cursor at soft_zone + 1 from left edge, associated with the node closest to origin in anchor partition
    /// This should be called once when the controller is first created or when cursor needs reset
    pub fn initialize_cursor(&mut self) {
        let viewport_center_y = self.viewport_state.viewport_bounds.height as i64 / 2;
        let desired_viewport_x = self.viewport_state.soft_zone + 1;
        let desired_viewport_y = viewport_center_y as u16;
        let desired_viewport_pos = ViewportPos::new(desired_viewport_x, desired_viewport_y);
        self.cursor.set_viewport_pos(desired_viewport_pos);

        // Find the node closest to partition origin (0, 0) and associate cursor with it
        if let Some(node_idx) = self.find_node_closest_to_origin() {
            // Set cursor to track this node at fractional (0.0, 0.5) = left edge, vertical middle
            self.cursor.set_node(node_idx, (0.0, 0.5));
            trace!(
                "Cursor initialized: viewport={:?}, node={:?}, fractional=(0.0, 0.5)",
                desired_viewport_pos, node_idx
            );
        } else {
            trace!("Warning: No data nodes found in anchor partition for cursor initialization");
        }
    }

    /// Find the data node closest to the origin (0, 0) in local coordinates of the anchor partition
    /// Since anchor partition has localpos = worldpos, this finds the node closest to world origin
    /// Returns node_idx if found
    fn find_node_closest_to_origin(&self) -> Option<NodeIndex> {
        let detail_level = self.get_detail_level();
        let anchor_partition = self.partition_controller.get_anchor_partition();

        // Get the anchor partition's layout
        let layout = self
            .partition_controller
            .partition_table
            .get_layout(anchor_partition, detail_level)?;

        // Find all data nodes and their distances from local origin (0, 0)
        let mut candidates: Vec<(NodeIndex, i64)> = layout
            .graph
            .node_indices()
            .filter_map(|layout_node_idx| {
                let node = layout.graph.node_weight(layout_node_idx)?;
                if let NodeRole::Data(domain_idx) = &node.role {
                    // Calculate distance from local origin (0, 0) using Manhattan distance
                    let distance = node.pos.x.abs() + node.pos.y.abs();
                    Some((*domain_idx, distance))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance (closest first)
        candidates.sort_by_key(|(_, dist)| *dist);

        // Return the closest node
        candidates.first().map(|(idx, _)| *idx)
    }

    /// Handle keyboard events for graph navigation and control
    ///
    /// Returns Some(true) for normal exit, Some(false) for abort, None to continue
    pub fn handle_key_event(&mut self, key: KeyEvent) -> Result<(), String> {
        match key.code {
            KeyCode::Char('r') => {
                self.trigger_rebuild();
            }

            // Graph navigation controls - move cursor with node awareness
            KeyCode::Left | KeyCode::Char('h') => {
                self.cursor.move_horizontal(-1, &self.viewport_graph)?;
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.cursor.move_horizontal(1, &self.viewport_graph)?;
            }
            // Note: In world coordinates, Y increases upward
            KeyCode::Up | KeyCode::Char('k') => {
                self.cursor.move_vertical(1, &self.viewport_graph)?;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.cursor.move_vertical(-1, &self.viewport_graph)?; // Move down = negative Y
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
            }

            _ => (),
        }
        Ok(())
    }

    /// Rebuild the viewport graph to show nodes and edges visible in the current camera view.
    ///
    /// This method implements cursor-anchored rebuilding, which positions the camera based on
    /// the cursor's position to maintain visual stability across rebuilds. The algorithm:
    ///
    /// 1. Find which partition contains the cursor's node
    /// 2. Set that partition as the anchor (ensures layout available at current detail_level)
    /// 3. Compute cursor world position from its node association and fractional offset
    /// 4. Calculate camera position using the formula:
    ///    `camera_center = (viewport_center - cursor_viewport_pos) + cursor_world_pos`
    /// 5. Load partitions covering the camera rect (with 2x buffer)
    /// 6. Build viewport graph with loaded partitions
    ///
    /// This ensures the cursor stays at its viewport position while the world coordinates
    /// align correctly, preventing coordinate drift during rebuilds.
    pub fn rebuild_viewport_graph(&mut self) -> Result<(), String> {
        let detail_level = self.detail_level;

        // Capture viewport bounds at start to ensure consistency throughout rebuild
        let viewport_bounds_snapshot = self.viewport_state.viewport_bounds;

        trace!("rebuild_viewport_graph: starting cursor-anchored rebuild");

        // Step 1: Initialize cursor if it has no node association
        if self.cursor.node_idx().is_none() {
            trace!("rebuild_viewport_graph: cursor has no node, initializing");
            self.initialize_cursor();
        }

        // Step 2: Find which partition the cursor's node belongs to
        let cursor_partition = if let Some(node_idx) = self.cursor.node_idx() {
            let node_id = <G as NodeIndexable>::from_index(&self.graph, node_idx.index());
            self.partition_controller
                .partition_table
                .node_map
                .get(&node_id)
                .map(|(partition_idx, _)| *partition_idx)
                .unwrap_or_else(|| {
                    trace!("rebuild_viewport_graph: cursor node not in node_map, using anchor");
                    self.partition_controller.get_anchor_partition()
                })
        } else {
            trace!("rebuild_viewport_graph: no cursor node, using anchor");
            self.partition_controller.get_anchor_partition()
        };

        trace!(
            "rebuild_viewport_graph: cursor partition = {}",
            cursor_partition
        );

        // Step 3: Inactivate any running animations and set cursor partition as anchor
        self.viewport_state.camera_anim = None;

        if let Err(e) = self
            .partition_controller
            .set_anchor_partition(cursor_partition)
        {
            return Err(format!("Failed to set anchor partition: {}", e));
        }

        // Step 4: Compute cursor world position directly from anchor partition layout
        // Since anchor partition has localpos = worldpos, we can compute directly
        let cursor_world = if let Some(node_idx) = self.cursor.node_idx() {
            // Get the anchor partition's layout
            let layout = self
                .partition_controller
                .partition_table
                .get_layout(cursor_partition, detail_level)
                .ok_or("Anchor partition layout not available")?;

            // Find the node in the layout
            let layout_node = layout
                .graph
                .node_indices()
                .find_map(|layout_idx| {
                    let node = layout.graph.node_weight(layout_idx)?;
                    if let NodeRole::Data(domain_idx) = &node.role
                        && *domain_idx == node_idx
                    {
                        Some(node)
                    } else {
                        None
                    }
                })
                .ok_or("Cursor node not found in anchor partition layout")?;

            // Get node center and size
            let node_center = layout_node.pos;
            let (width, height) = layout_node.size;

            // Calculate world position from fractional offset (same logic as cursor.to_world_pos())
            let half_width = (width as i64 - 1) / 2;
            let half_height = (height as i64 - 1) / 2;
            let wo_x = node_center.x - half_width;
            let wo_y = node_center.y - half_height;

            let span_x = width.saturating_sub(1);
            let span_y = height.saturating_sub(1);

            let (frac_x, frac_y) = self.cursor.fractional_pos();
            let cursor_x = wo_x + (frac_x * span_x as f64).round() as i64;
            let cursor_y = wo_y + (frac_y * span_y as f64).round() as i64;

            WorldPos::new(cursor_x, cursor_y)
        } else {
            return Err("Cursor has no node association".to_string());
        };

        trace!(
            "rebuild_viewport_graph: cursor_world = ({}, {})",
            cursor_world.x, cursor_world.y
        );

        // Step 5: Get cursor viewport position and viewport center
        let cursor_viewport = self.cursor.viewport_pos();
        let half_width = (viewport_bounds_snapshot.width as i64 - 1) / 2;
        let half_height = (viewport_bounds_snapshot.height as i64 - 1) / 2;
        let viewport_center = ViewportPos::new(half_width as u16, half_height as u16);

        trace!(
            "rebuild_viewport_graph: viewport dimensions width={}, height={}, half_width={}, half_height={}",
            viewport_bounds_snapshot.width,
            viewport_bounds_snapshot.height,
            half_width,
            half_height
        );

        trace!(
            "rebuild_viewport_graph: cursor_viewport = ({}, {}), viewport_center = ({}, {})",
            cursor_viewport.x, cursor_viewport.y, viewport_center.x, viewport_center.y
        );

        // Step 6: Camera positioning formula
        let camera = WorldPos::new(
            cursor_world.x - cursor_viewport.x as i64 + half_width,
            cursor_world.y - cursor_viewport.y as i64 + half_height,
        );

        trace!(
            "rebuild_viewport_graph: camera formula: cursor_world.y={} - cursor_viewport.y={} + half_height={} = camera.y={}",
            cursor_world.y, cursor_viewport.y, half_height, camera.y
        );

        self.viewport_state.camera_current = camera;
        self.viewport_state.camera_target = camera;

        let camera_rect = WorldRect::from_center_and_size(
            camera,
            (
                viewport_bounds_snapshot.width as u64,
                viewport_bounds_snapshot.height as u64,
            ),
        );
        trace!(
            "rebuild_viewport_graph: camera_rect from_center_and_size: center.y={}, height={}, rect.min.y={}, rect.max.y={}",
            camera.y, viewport_bounds_snapshot.height, camera_rect.min.y, camera_rect.max.y
        );

        trace!(
            "rebuild_viewport_graph: camera positioned at ({}, {})",
            camera.x, camera.y
        );

        // Step 7: Compute camera rect with buffer (2x) and load partitions
        let camera_rect = self.viewport_state.camera_rect();
        let covered_rect = camera_rect.resize(2.0);

        trace!(
            "rebuild_viewport_graph: loading partitions for covered rect x:{}-{}, y:{}-{}",
            covered_rect.min.x, covered_rect.max.x, covered_rect.min.y, covered_rect.max.y
        );

        let active_partitions = self
            .partition_controller
            .load_partitions_for_rect(covered_rect)?;

        trace!(
            "rebuild_viewport_graph: active partitions = {:?}",
            active_partitions
        );

        // Step 8: Build viewport graph with those partitions
        self.viewport_graph = CroppedGraph::new(
            covered_rect,
            &self.partition_controller.partition_table,
            &active_partitions,
            detail_level,
        );

        // Update rebuild tracking
        self.last_rebuild_camera_center = self.viewport_state.camera_current;
        self.rebuild_needed = false;

        // Note: We don't update cursor viewport position here because cursor-anchored rebuilding
        // relies on preserving the cursor's viewport position to calculate the camera
        // self.cursor.update(&self.viewport_graph, camera_rect)?;

        trace!(
            "rebuild_viewport_graph: complete - {} nodes, {} edges, cursor viewport ({}, {})",
            self.viewport_graph.node_count(),
            self.viewport_graph.edge_count(),
            self.cursor.viewport_pos().x,
            self.cursor.viewport_pos().y
        );

        Ok(())
    }

    /// Get a reference to the currently available viewport graph
    pub fn get_viewport_graph(&self) -> &CroppedGraph {
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
        let half_width = (self.viewport_state.viewport_bounds.width as i64 - 1) / 2;
        let half_height = (self.viewport_state.viewport_bounds.height as i64 - 1) / 2;

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

    /// Update animations (camera and cursor) for the given frame delta.
    /// This should be called once per frame to advance smooth animations.
    ///
    /// # Parameters
    /// - delta: Time elapsed since last frame
    ///
    /// # Note
    /// viewport_bounds must be set on viewport_state BEFORE calling this method,
    /// as it's needed for coordinate calculations during animation updates.
    pub fn update_animations(&mut self, delta: std::time::Duration) {
        self.viewport_state
            .update(delta, &mut self.cursor, &self.viewport_graph);
    }

    /// Enable cursor rendering
    pub fn show_cursor(&mut self) {
        self.cursor.set_visibility(true);
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

        // Test camera origin calculation using correct formula: camera - ((width-1)/2, (height-1)/2)
        let origin = state.camera_rect().min;
        assert_eq!(origin, WorldPos::new(91, 96)); // 100 - (20-1)/2, 100 - (10-1)/2 = 100 - 9, 100 - 4

        // Test world to viewport conversion
        let world_pos = WorldPos::new(95, 98);
        let viewport_pos = writer.world_to_viewport(world_pos);
        // world_pos - origin = (95, 98) - (91, 96) = (4, 2)
        assert_eq!(viewport_pos, Some(ViewportPos::new(4, 2)));

        // Test out of bounds
        let out_of_bounds = WorldPos::new(80, 80);
        assert_eq!(writer.world_to_viewport(out_of_bounds), None);

        // Test viewport to world conversion (round-trip)
        let vp = ViewportPos::new(4, 2);
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
        writer.set_char(world_pos, 'X');
        assert_eq!(
            writer
                .get_char(world_pos)
                .expect("Position should be accessible"),
            'X',
            "Character should be written successfully"
        );
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
        writer.set_string(world_pos, test_string);

        let read_string = writer.get_string(world_pos, 10);
        assert!(
            read_string.starts_with(test_string),
            "Should get back the written string"
        );

        // Write a vertical string and read it back
        let vertical_string = "VERT";
        let vertical_pos = WorldPos::new(15, 6); // Should be visible
        writer.set_string_vertical(vertical_pos, vertical_string);

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

        // Try to write outside viewport bounds, it shouldn't panic
        writer.set_char(WorldPos::new(100, 100), 'X');

        // Try to write a string that goes beyond viewport
        writer.set_string(WorldPos::ZERO, "Hello World");
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
        writer.fill_rect_styled(rect, fill_char, Style::default().fg(Color::Blue));

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
            let _ = controller.handle_key_event(plus_key);
        }

        // If we get here without panicking, the test passes
        println!("Test completed successfully");
    }
}
