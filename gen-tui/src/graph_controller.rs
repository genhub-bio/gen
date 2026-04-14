use std::hash::Hash;

#[cfg(feature = "crossterm")]
use crossterm::event::{KeyCode, KeyEvent};
use gen_sugiyama::{self, VERTEX_SPACING_DEFAULT};
use log::trace;
use petgraph::{
    graph::NodeIndex,
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
    geometry::{BigRect, ViewportPos, WorldPos},
    layout::{NodeRole, PartitionLayout, VisualDetail},
    partition_controller::{ControllerConfig, PartitionController},
    partition_table::PartitionConfig,
    plotter::{NodeSizer, PathStyle},
    theme::current_theme,
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
    G: GraphBase,
    S: NodeSizer<G>,
{
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
    /// Flag indicating that the layout changed (zoom, spacing) and the camera
    /// should be repositioned to keep the cursor at its current viewport position.
    layout_changed: bool,

    /// Persistent requested highlights in domain terms
    pub highlights: Vec<(HighlightKind<G::NodeId>, PathStyle)>,
}

/// Type of element to highlight in the graph
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HighlightKind<N> {
    /// A single node
    Node(N),
    /// An edge between two nodes (source, target)
    Edge(N, N),
    /// A path consisting of a sequence of nodes
    Path(Vec<N>),
}

impl<G, S> GraphController<G, S>
where
    G: GraphBase + EdgeIndexable + NodeIndexable + NodeCount + Visitable,
    G::NodeId: Copy + Eq + Hash + Ord,
    G::EdgeId: Clone,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNodeIdentifiers<NodeId = G::NodeId>
        + IntoEdgeReferences<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNeighborsDirected<NodeId = G::NodeId>,
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
        let mut partition_controller = PartitionController::new_with_config(
            graph,
            node_sizer,
            config.partition,
            config.controller,
        );

        if let Err(e) = partition_controller.set_anchor_partition(0) {
            eprintln!("Warning: Failed to initialize reference partition: {}", e);
        }

        Self {
            viewport_state: ViewportState::new(),
            cursor: Cursor::default(),
            detail_level: VisualDetail::Truncated, // Default detail level
            partition_controller,
            viewport_graph: CroppedGraph::empty(),
            last_rebuild_camera_center: WorldPos::ZERO,
            rebuild_needed: true,
            layout_changed: true, // Treat initial build as a layout change to place the camera
            highlights: Vec::new(),
        }
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

    /// Get a reference to the original domain graph
    pub fn graph(&self) -> &G {
        &self.partition_controller.graph
    }

    /// Set the anchor partition and reset coordinate system
    pub fn set_anchor_partition(&mut self, partition_idx: usize) -> Result<(), String> {
        self.partition_controller
            .set_anchor_partition(partition_idx)
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
    // for example)
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
        self.layout_changed = true;
    }

    /// Get the current level of detail
    pub fn get_detail_level(&self) -> VisualDetail {
        self.detail_level
    }

    /// Get a reference to the node highlights in the current viewport
    pub fn get_node_highlights(&self) -> &[(WorldPos, PathStyle)] {
        &self.viewport_graph.node_highlights
    }

    /// Get a reference to the edge highlights in the current viewport
    pub fn get_edge_highlights(&self) -> &[((WorldPos, WorldPos), PathStyle)] {
        &self.viewport_graph.edge_highlights
    }

    /// Internal helper to apply a node highlight to the viewport graph
    fn apply_node_highlight(
        viewport_graph: &mut CroppedGraph,
        graph: &G,
        node_id: G::NodeId,
        style: PathStyle,
    ) {
        let node_idx = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
        if let Some(pos) = viewport_graph.node_positions.get(&node_idx) {
            viewport_graph.node_highlights.push((*pos, style));
        }
    }

    /// Internal helper to apply an edge highlight to the viewport graph
    fn apply_edge_highlight(
        viewport_graph: &mut CroppedGraph,
        graph: &G,
        src_id: G::NodeId,
        tgt_id: G::NodeId,
        style: PathStyle,
    ) {
        let u = NodeIndex::new(<G as NodeIndexable>::to_index(graph, src_id));
        let v = NodeIndex::new(<G as NodeIndexable>::to_index(graph, tgt_id));

        // Find all visual edges that contain this domain edge in their bundle
        // Collect first to avoid mutable borrow of viewport_graph while iterating
        let edges_to_highlight: Vec<(WorldPos, WorldPos)> = viewport_graph
            .edges()
            .filter(|(_, _, bundle)| bundle.contains(&(u, v)) || bundle.contains(&(v, u)))
            .map(|(s, t, _)| (s, t))
            .collect();

        for (source_pos, target_pos) in edges_to_highlight {
            viewport_graph
                .edge_highlights
                .push(((source_pos, target_pos), style));
        }
    }

    /// Internal helper to apply a path highlight to the viewport graph
    fn apply_path_highlight(
        viewport_graph: &mut CroppedGraph,
        graph: &G,
        nodes: &[G::NodeId],
        style: PathStyle,
    ) {
        // Highlight all nodes
        for &node_id in nodes {
            Self::apply_node_highlight(viewport_graph, graph, node_id, style);
        }

        // Highlight all consecutive edges
        for window in nodes.windows(2) {
            if let [src, tgt] = window {
                Self::apply_edge_highlight(viewport_graph, graph, *src, *tgt, style);
            }
        }
    }

    /// Pick the next unused accent color from the theme (slots 0x08–0x0F).
    ///
    /// Scans `self.highlights` for colors already in use and returns the first
    /// accent slot that hasn't been claimed yet.  Wraps around to 0x08 once all
    /// eight accent slots are occupied.
    pub fn next_accent_color(&self) -> Color {
        use std::collections::HashSet;
        let theme = current_theme();
        let used: HashSet<Color> = self
            .highlights
            .iter()
            .filter(|(_, s)| s.color != Color::Reset)
            .map(|(_, s)| s.color)
            .collect();
        (0x08..=0x0F)
            .map(|i| theme[i])
            .find(|c| !used.contains(c))
            .unwrap_or(theme[0x08])
    }

    /// Highlight a node using the next available theme accent color.
    /// Returns the color that was chosen.
    pub fn add_node_highlight(&mut self, node_id: G::NodeId) -> Color {
        let color = self.next_accent_color();
        self.set_node_highlight(node_id, PathStyle::new(color));
        color
    }

    /// Highlight an edge using the next available theme accent color.
    /// Returns the color that was chosen.
    pub fn add_edge_highlight(&mut self, edge: (G::NodeId, G::NodeId)) -> Color {
        let color = self.next_accent_color();
        self.set_edge_highlight(edge, PathStyle::new(color));
        color
    }

    /// Highlight a path using the next available theme accent color.
    /// Returns the color that was chosen.
    pub fn add_path_highlight(&mut self, path_nodes: Vec<G::NodeId>) -> Color {
        let color = self.next_accent_color();
        self.set_path_highlight(PathStyle::new(color), path_nodes);
        color
    }

    /// Set a node highlight
    pub fn set_node_highlight(&mut self, node_id: G::NodeId, style: PathStyle) {
        let graph = &self.partition_controller.graph;
        Self::apply_node_highlight(&mut self.viewport_graph, graph, node_id, style);
        let kind = HighlightKind::Node(node_id);
        self.highlights.push((kind, style));
    }

    /// Set an edge highlight
    pub fn set_edge_highlight(&mut self, edge: (G::NodeId, G::NodeId), style: PathStyle) {
        let graph = &self.partition_controller.graph;
        Self::apply_edge_highlight(&mut self.viewport_graph, graph, edge.0, edge.1, style);
        let kind = HighlightKind::Edge(edge.0, edge.1);
        self.highlights.push((kind, style));
    }

    /// Set a path highlight with a specific style
    ///
    /// # Parameters
    /// - style: PathStyle for highlighting the path
    /// - path_nodes: Sequence of nodes that form the path
    pub fn set_path_highlight(&mut self, style: PathStyle, path_nodes: Vec<G::NodeId>) {
        let graph = &self.partition_controller.graph;
        Self::apply_path_highlight(&mut self.viewport_graph, graph, &path_nodes, style);
        let kind = HighlightKind::Path(path_nodes);
        self.highlights.push((kind, style));
    }

    /// Set a path highlight with a specific color (convenience method)
    ///
    /// # Parameters
    /// - color: Color for highlighting the path
    /// - path_nodes: Sequence of nodes that form the path
    pub fn set_path_highlight_color(&mut self, color: Color, path_nodes: Vec<G::NodeId>) {
        self.set_path_highlight(PathStyle::new(color), path_nodes);
    }

    /// Check if a specific style has any highlighting
    pub fn has_highlight(&self, style: &PathStyle) -> bool {
        self.highlights.iter().any(|(_, s)| s == style)
    }

    /// Clear highlighting for a specific style
    pub fn clear_highlight(&mut self, style: &PathStyle) {
        self.highlights.retain(|(_, s)| s != style);
        // Also clear from viewport graph
        self.viewport_graph
            .node_highlights
            .retain(|(_, s)| s != style);
        self.viewport_graph
            .edge_highlights
            .retain(|(_, s)| s != style);
        self.trigger_rebuild();
    }

    /// Clear all highlights
    pub fn clear_all_highlights(&mut self) {
        self.highlights.clear();
        self.viewport_graph.node_highlights.clear();
        self.viewport_graph.edge_highlights.clear();
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
    /// (used only as a helper in tests currently)
    pub fn ensure_camera_coverage(&mut self) -> Result<Vec<usize>, String> {
        let buffer_factor = 2.0;
        let camera_rect = self.viewport_state.camera_rect().resize(buffer_factor);
        self.partition_controller
            .load_partitions_for_rect(camera_rect)
    }

    /// Find a domain node's world position in the new layout system after layout changes
    /// This searches the partition layouts directly to find the node's current world position
    pub fn find_domain_node_world_position(&self, domain_idx: NodeIndex) -> Option<WorldPos> {
        let loaded_partitions = self.partition_controller.get_loaded_partitions_info();
        let detail_level = self.get_detail_level();

        for (partition_idx, _, _, _) in loaded_partitions {
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
        // Cap at ~5 zoom levels above default to prevent runaway layout recomputes
        const MAX_VERTEX_SPACING: f64 = VERTEX_SPACING_DEFAULT + 10.0;
        if self.partition_controller.get_vertex_spacing() >= MAX_VERTEX_SPACING {
            return;
        }
        // Increase vertex spacing
        self.partition_controller.adjust_vertex_spacing(2.0);
        trace!(
            "disperse: vertex spacing now {}",
            self.partition_controller.get_vertex_spacing()
        );

        // Clear all layouts to force complete recalculation with new spacing
        self.partition_controller.clear_all_layouts();
        self.rebuild_needed = true;
        self.layout_changed = true;
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
        self.layout_changed = true;
    }

    /// Get current vertex spacing from partition controller
    fn get_current_vertex_spacing(&self) -> f64 {
        self.partition_controller.get_vertex_spacing()
    }

    /// Place cursor near left edge, vertically centered.
    /// This positions the cursor in a comfortable viewing position within the viewport.
    /// Should be called when viewport bounds are first established (transition from 0x0).
    pub fn place_cursor(&mut self) {
        let viewport_center_y = self.viewport_state.viewport_bounds.height as i64 / 2;
        let desired_viewport_x = self.viewport_state.soft_zone + 1;
        let desired_viewport_y = viewport_center_y as u16;
        let desired_viewport_pos = ViewportPos::new(desired_viewport_x, desired_viewport_y);
        self.cursor.set_viewport_pos(desired_viewport_pos);

        trace!("Cursor placed: viewport={:?}", desired_viewport_pos);
    }

    /// Associate cursor with node closest to origin.
    /// Call this only if no node has been explicitly set.
    pub fn associate_cursor(&mut self) {
        // Find the node closest to partition origin (0, 0) and associate cursor with it
        if let Some(node_idx) = self.find_node_closest_to_origin() {
            // Set cursor to track this node at fractional (0.0, 0.5) = left edge, vertical middle
            self.cursor.set_node(node_idx, (0.0, 0.5));
            trace!(
                "Cursor associated: node={:?}, fractional=(0.0, 0.5)",
                node_idx
            );
        } else {
            trace!("Warning: No data nodes found in anchor partition for cursor association");
        }
    }

    /// Initialize cursor completely: viewport position + node association.
    /// This is a convenience method for the common case where both need initialization.
    pub fn initialize_cursor(&mut self) {
        self.place_cursor();
        self.associate_cursor();
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

    /// Move the cursor horizontally by `delta` world units.
    pub fn navigate_horizontal(&mut self, delta: i64) -> Result<(), String> {
        self.cursor.move_horizontal(delta, &self.viewport_graph)
    }

    /// Move the cursor vertically by `delta` world units.
    pub fn navigate_vertical(&mut self, delta: i64) -> Result<(), String> {
        self.cursor.move_vertical(delta, &self.viewport_graph)
    }

    /// Handle keyboard events for graph navigation and control
    ///
    /// Returns Some(true) for normal exit, Some(false) for abort, None to continue
    #[cfg(feature = "crossterm")]
    pub fn handle_key_event(&mut self, key: KeyEvent) -> Result<(), String> {
        match key.code {
            KeyCode::Char('r') => {
                self.trigger_rebuild();
            }

            // Graph navigation controls - move cursor with node awareness
            KeyCode::Left | KeyCode::Char('h') => {
                let vp_w = self.viewport_state.viewport_bounds.width as i64;
                let delta = if self.cursor.is_coarse_mode() {
                    -vp_w
                } else {
                    -1
                };
                self.cursor.move_horizontal(delta, &self.viewport_graph)?;
                self.trigger_rebuild();
            }
            KeyCode::Right | KeyCode::Char('l') => {
                let vp_w = self.viewport_state.viewport_bounds.width as i64;
                let delta = if self.cursor.is_coarse_mode() {
                    vp_w
                } else {
                    1
                };
                self.cursor.move_horizontal(delta, &self.viewport_graph)?;
                self.trigger_rebuild();
            }
            // Note: In world coordinates, Y increases upward
            KeyCode::Up | KeyCode::Char('k') => {
                let vp_h = self.viewport_state.viewport_bounds.height as i64;
                let delta = if self.cursor.is_coarse_mode() {
                    vp_h
                } else {
                    1
                };
                self.cursor.move_vertical(delta, &self.viewport_graph)?;
                self.trigger_rebuild();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let vp_h = self.viewport_state.viewport_bounds.height as i64;
                let delta = if self.cursor.is_coarse_mode() {
                    -vp_h
                } else {
                    -1
                };
                self.cursor.move_vertical(delta, &self.viewport_graph)?; // Move down = negative Y
                self.trigger_rebuild();
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

        // Step 1: Handle first-time viewport initialization
        // When transitioning from 0x0 viewport to real bounds, cursor viewport position needs setup
        let viewport_was_uninitialized =
            viewport_bounds_snapshot.width == 0 || viewport_bounds_snapshot.height == 0;

        if viewport_was_uninitialized {
            self.place_cursor();

            // If cursor also lacks a node (wasn't set before first render), find one
            if self.cursor.node_idx().is_none() {
                trace!("rebuild_viewport_graph: cursor also has no node, associating with default");
                self.associate_cursor();
            }
        } else if self.cursor.node_idx().is_none() {
            // Viewport was already valid but cursor lost its node somehow - reinitialize completely
            trace!("rebuild_viewport_graph: cursor has no node (unusual), reinitializing");
            self.initialize_cursor();
        }

        // Step 2: Find which partition the cursor's node belongs to
        let cursor_partition = if let Some(node_idx) = self.cursor.node_idx() {
            let node_id = <G as NodeIndexable>::from_index(
                &self.partition_controller.graph,
                node_idx.index(),
            );
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

        // Step 3: Inactivate any running animations and set cursor partition as anchor.
        // Track whether the anchor changes: world coordinates are defined relative to the anchor
        // (local_pos + partition_origin - anchor_origin), so a new anchor shifts the entire
        // coordinate system. camera_current is stored in world coords, so it becomes invalid
        // the moment the anchor changes and must be recomputed.
        self.viewport_state.camera_anim = None;

        let old_anchor = self.partition_controller.get_anchor_partition();
        if let Err(e) = self
            .partition_controller
            .set_anchor_partition(cursor_partition)
        {
            return Err(format!("Failed to set anchor partition: {}", e));
        }
        let anchor_changed = self.partition_controller.get_anchor_partition() != old_anchor;

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

        // Step 5: Get cursor and camera positions in viewport coordinates
        let cursor_viewport = self.cursor.viewport_pos();
        let half_width = (viewport_bounds_snapshot.width as i64 - 1) / 2;
        let half_height = (viewport_bounds_snapshot.height as i64 - 1) / 2;

        // Step 6: Position the camera.
        //
        // # Coordinate System
        // - Camera represents the CENTER of the viewport in world coordinates
        // - Viewport origin (top-left) = camera - (width/2, height/2)
        // - Transformation: viewport_pos = world_pos - camera + (width/2, height/2)
        // - Inverse: camera = world_pos - viewport_pos + (width/2, height/2)
        //
        // Reposition the camera when the layout changed (zoom, spacing) OR when the anchor
        // partition changed. An anchor change invalidates camera_current because world coords
        // are defined relative to the anchor; the old camera position is in the wrong coordinate
        // system and must be recomputed from cursor_world (which was derived from the new anchor).
        let layout_changed = self.layout_changed;
        self.layout_changed = false;
        if layout_changed || anchor_changed {
            let cam = WorldPos::new(
                cursor_world.x - cursor_viewport.x as i64 + half_width,
                cursor_world.y - cursor_viewport.y as i64 + half_height,
            );
            self.viewport_state.camera_current = cam;
            self.viewport_state.camera_target = cam;
        }

        // Step 7: Compute camera rect with buffer (2x) and load partitions
        let camera_rect = self.viewport_state.camera_rect();
        let covered_rect = camera_rect.resize(2.0);

        let active_partitions = self
            .partition_controller
            .load_partitions_for_rect(covered_rect)?;

        // Step 8: Build a cropped graph with those partitions
        self.viewport_graph = CroppedGraph::new(
            covered_rect,
            &self.partition_controller.partition_table,
            &active_partitions,
            detail_level,
        );

        // Apply persistent highlights to the new viewport graph
        for (kind, style) in &self.highlights {
            match kind {
                HighlightKind::Node(node_id) => {
                    Self::apply_node_highlight(
                        &mut self.viewport_graph,
                        &self.partition_controller.graph,
                        *node_id,
                        *style,
                    );
                }
                HighlightKind::Edge(src, tgt) => {
                    Self::apply_edge_highlight(
                        &mut self.viewport_graph,
                        &self.partition_controller.graph,
                        *src,
                        *tgt,
                        *style,
                    );
                }
                HighlightKind::Path(nodes) => {
                    Self::apply_path_highlight(
                        &mut self.viewport_graph,
                        &self.partition_controller.graph,
                        nodes,
                        *style,
                    );
                }
            }
        }

        // Update rebuild tracking
        self.last_rebuild_camera_center = self.viewport_state.camera_current;
        self.rebuild_needed = false;

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

    /// Returns true if any animation is currently in flight (camera easing).
    /// Callers can use this to decide whether to redraw at a fixed rate or block
    /// indefinitely waiting for the next input event.
    pub fn is_animating(&self) -> bool {
        self.viewport_state.camera_anim.is_some()
    }

    // ==================== Mouse / Panning Mode ====================

    /// Returns true when the cursor is visible (keyboard-driven mode).
    pub fn is_cursor_visible(&self) -> bool {
        self.cursor.is_visible()
    }

    /// Hide the cursor and let the camera move freely.
    /// The cursor still tracks the closest node so it is ready when switching back.
    pub fn hide_cursor(&mut self) {
        self.cursor.set_visibility(false);
    }

    /// Show the cursor and enable camera-following.
    pub fn show_cursor(&mut self) {
        self.cursor.set_visibility(true);
    }

    /// Pan the camera by a drag delta expressed in terminal coordinates.
    ///
    /// The Y-axis flip (world Y+ is up, terminal Y+ is down) is handled here so
    /// callers can pass raw terminal deltas directly.
    ///
    /// Panning is applied immediately with no animation so the canvas tracks the
    /// pointer without lag.
    pub fn move_by_terminal(&mut self, terminal_dx: i16, terminal_dy: i16) {
        // X: drag right → camera moves left (negate)
        // Y: drag down (+terminal_dy) → camera moves down → camera_y increases because world Y+ is up
        let world_dx = -(terminal_dx as i64);
        let world_dy = terminal_dy as i64;

        self.hide_cursor();
        let new = WorldPos::new(
            self.viewport_state.camera_target.x + world_dx,
            self.viewport_state.camera_target.y + world_dy,
        );
        self.viewport_state.camera_current = new;
        self.viewport_state.camera_target = new;
        self.viewport_state.camera_anim = None;
    }

    /// Handle a click at the given terminal coordinates.
    ///
    /// - If a data node occupies the clicked cell: places the cursor on that node,
    ///   switches to cursor-anchored mode, and returns `true`.
    /// - Otherwise: switches to free-camera mode and returns `false`.
    pub fn handle_click(&mut self, terminal_x: u16, terminal_y: u16) -> bool {
        let Some(click_world) = self
            .viewport_state
            .terminal_to_world(terminal_x, terminal_y)
        else {
            self.hide_cursor();
            return false;
        };

        // Collect the hit result before any mutable borrow to satisfy the borrow checker.
        let hit =
            self.viewport_graph
                .data_nodes()
                .find_map(|(node_center, domain_idx, layout_node)| {
                    let rect = BigRect::from_center_and_size(node_center, layout_node.size);
                    if rect.contains(click_world) {
                        let frac_x = ((click_world.x - rect.left()) as f64
                            / layout_node.size.0.max(1) as f64)
                            .clamp(0.0, 1.0);
                        let frac_y = ((click_world.y - rect.bottom()) as f64
                            / layout_node.size.1.max(1) as f64)
                            .clamp(0.0, 1.0);
                        Some((domain_idx, (frac_x, frac_y)))
                    } else {
                        None
                    }
                });

        if let Some((domain_idx, frac)) = hit {
            self.cursor.set_node(domain_idx, frac);
            self.show_cursor();
            true
        } else {
            self.hide_cursor();
            false
        }
    }

    /// In free-camera mode, keep the cursor on the cell of the visible node closest to
    /// the camera center, and update the cursor's viewport position to match.
    ///
    /// Both are updated on every drag move so that when the user zooms (layout changes),
    /// `rebuild_viewport_graph` honours the contract: the anchor cell will be at the
    /// exact same viewport position before and after the zoom.
    pub fn sync_cursor_to_closest_node(&mut self) {
        let center = self.viewport_state.camera_current;

        // Find the node whose bounding rect is closest to the camera center.
        let best = self
            .viewport_graph
            .data_nodes()
            .min_by_key(|(pos, _, layout_node)| {
                let rect = BigRect::from_center_and_size(*pos, layout_node.size);
                let closest_x = center.x.clamp(rect.left(), rect.right());
                let closest_y = center.y.clamp(rect.bottom(), rect.top());
                let dx = closest_x - center.x;
                let dy = closest_y - center.y;
                dx * dx + dy * dy
            });

        if let Some((node_center, domain_idx, layout_node)) = best {
            let rect = BigRect::from_center_and_size(node_center, layout_node.size);

            // Closest cell within the node rect to the camera center.
            let closest_x = center.x.clamp(rect.left(), rect.right());
            let closest_y = center.y.clamp(rect.bottom(), rect.top());

            let (width, height) = layout_node.size;
            let frac_x = ((closest_x - rect.left()) as f64 / width.max(1) as f64).clamp(0.0, 1.0);
            let frac_y =
                ((closest_y - rect.bottom()) as f64 / height.max(1) as f64).clamp(0.0, 1.0);

            self.cursor.set_node(domain_idx, (frac_x, frac_y));

            // Keep the cursor's viewport position in sync with where that cell currently
            // appears on screen.  The zoom formula uses this:
            //   camera = cursor_world - cursor_viewport + half_viewport
            // so cursor_viewport must be the *current* screen position of the anchor cell.
            let closest_cell = WorldPos::new(closest_x, closest_y);
            if let Some(vp) = self.viewport_state.world_to_viewport(closest_cell) {
                self.cursor.set_viewport_pos(vp);
            }
        }
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
        state.camera_anim.is_some();

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

        impl NodeSizer<MockDomainGraph> for TestNodeSizer {
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

        let mut controller = GraphController::new(domain_graph.clone(), node_sizer);

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
