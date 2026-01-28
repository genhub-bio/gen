use std::collections::{HashMap, HashSet};

use log::{debug, trace};
use petgraph::{graphmap::UnGraphMap, prelude::NodeIndex};

use crate::{
    geometry::{BigRect, PartitionIndex, WorldPos},
    layout::{LayoutNode, NodeRole, VisualDetail},
    partition_table::PartitionTable,
};

/// A graph containing only the nodes and edges visible in the current viewport.
/// Uses world coordinates as keys for natural deduplication at partition boundaries.
#[derive(Clone)]
pub struct CroppedGraph {
    /// Graph keyed by world coordinates, edges store domain node pairs
    pub graph: UnGraphMap<WorldPos, Vec<(NodeIndex, NodeIndex)>>,

    /// Node data at each world position (because GraphMaps don't store node labels)
    pub node_data_by_pos: HashMap<WorldPos, LayoutNode>,

    /// Map from domain NodeIndex to world position for quick lookups
    pub node_positions: HashMap<NodeIndex, WorldPos>,

    /// Layers of the graph - each layer contains domain node indices in that layer
    /// This provides a unified layer view across all partitions
    pub layers: Vec<Vec<NodeIndex>>,

    /// Track which nodes we've added from which partitions
    pub included_nodes: HashSet<(PartitionIndex, NodeIndex<u32>)>,

    /// Node highlights: list of world positions with their associated styles
    pub node_highlights: Vec<(WorldPos, crate::plotter::PathStyle)>,
    /// Edge highlights: list of world position pairs with their associated styles
    pub edge_highlights: Vec<((WorldPos, WorldPos), crate::plotter::PathStyle)>,
}

impl CroppedGraph {
    pub fn empty() -> Self {
        debug!(
            "[DEBUG_TRACE_LAYOUT] empty: -> result=ViewportGraph{{graph_nodes=0, nodes_count=0, domain_to_world_count=0, layers_count=0, included_nodes_count=0}}"
        );
        Self {
            graph: UnGraphMap::new(),
            node_data_by_pos: HashMap::new(),
            node_positions: HashMap::new(),
            layers: Vec::new(),
            included_nodes: HashSet::new(),
            node_highlights: Vec::new(),
            edge_highlights: Vec::new(),
        }
    }

    pub fn new<G>(
        viewport: BigRect<i64>,
        partition_table: &PartitionTable<G>,
        active_partitions: &[usize],
        detail_level: VisualDetail,
    ) -> Self
    where
        G: petgraph::visit::GraphBase
            + Clone
            + petgraph::visit::EdgeIndexable
            + petgraph::visit::NodeIndexable
            + petgraph::visit::NodeCount
            + petgraph::visit::Visitable
            + petgraph::visit::IntoEdgeReferences
            + petgraph::visit::IntoNeighborsDirected
            + petgraph::visit::IntoNodeIdentifiers,
        G::NodeId: Copy + Eq + std::hash::Hash + Ord,
        G::EdgeId: Clone,
    {
        debug!(
            "[DEBUG_TRACE_LAYOUT] new: viewport=BigRect{{min={{x={}, y={}}}, max={{x={}, y={}}}}}, detail_level={:?}",
            viewport.min.x, viewport.min.y, viewport.max.x, viewport.max.y, detail_level
        );

        debug!(
            "[DEBUG_TRACE_LAYOUT] new: coverage=ViewportCoverage{{covered_partitions_count={}, covered_partitions={:?}}}",
            active_partitions.len(),
            active_partitions
        );

        // Start off with an empty instance (can't call self in the constructor)
        let mut this = Self::empty();

        for &partition_idx in active_partitions {
            if let Some(partition_layout) =
                &partition_table.partitions[partition_idx].layouts[detail_level.as_index()]
            {
                let clamped_viewport = partition_table
                    .clamp_rect_to_partition(viewport, partition_idx, detail_level)
                    .unwrap();

                // Convert the corners of the clamped viewport to LocalPos coordinates
                let local_corner_min = partition_table
                    .world_to_local(clamped_viewport.min, detail_level)
                    .unwrap();
                let local_corner_max = partition_table
                    .world_to_local(clamped_viewport.max, detail_level)
                    .unwrap();

                let envelope = rstar::AABB::from_corners(
                    [local_corner_min.x, local_corner_min.y],
                    [local_corner_max.x, local_corner_max.y],
                );

                let visible_objects: Vec<_> = partition_layout
                    .spatial_index
                    .locate_in_envelope_intersecting(&envelope)
                    .cloned()
                    .collect();

                trace!(
                    "[DEBUG_TRACE_LAYOUT] new: partition_idx={}, x = {} - {}, visible_objects_count={:?}",
                    partition_idx,
                    local_corner_min.x,
                    local_corner_max.x,
                    visible_objects.len()
                );
                // First pass: collect nodes and track layers
                for obj in &visible_objects {
                    if obj.is_node() {
                        let node_idx = obj.primary_node.layout;
                        if let Some(node_data) = partition_layout.graph.node_weight(node_idx) {
                            let world_pos =
                                partition_table.local_to_world(node_data.pos, detail_level);

                            // Debug output for data nodes
                            if let NodeRole::Data(domain_idx) = &node_data.role {
                                trace!("  Data Node:");
                                trace!("    Domain Index: {:?}", domain_idx);
                                trace!("    LocalPos: {:?}", node_data.pos);
                                trace!("    World Pos: {:?}", world_pos);
                                trace!("    Size: {:?}", node_data.size);
                                trace!("    Layer: {:?}", node_data.layer);
                            }

                            this.merge_node(world_pos, node_data.clone()).unwrap();
                            this.included_nodes.insert((partition_idx, node_idx));
                        }
                    }
                }

                // Second pass: edges
                for obj in &visible_objects {
                    if obj.is_edge() {
                        let source_idx = obj.primary_node.layout;
                        let target_idx = obj
                            .secondary_node
                            .ok_or_else(|| "Edge missing target node".to_string())
                            .unwrap()
                            .layout;

                        if let Some(edge_idx) =
                            partition_layout.graph.find_edge(source_idx, target_idx)
                        {
                            let edge_data = partition_layout
                                .graph
                                .edge_weight(edge_idx)
                                .ok_or_else(|| "Edge data not found".to_string())
                                .unwrap();
                            let source_data = partition_layout
                                .graph
                                .node_weight(source_idx)
                                .ok_or_else(|| "Source node not found".to_string())
                                .unwrap();
                            let target_data = partition_layout
                                .graph
                                .node_weight(target_idx)
                                .ok_or_else(|| "Target node not found".to_string())
                                .unwrap();

                            // Convert local positions to world positions
                            let source_world =
                                partition_table.local_to_world(source_data.pos, detail_level);
                            let target_world =
                                partition_table.local_to_world(target_data.pos, detail_level);

                            this.merge_node(source_world, source_data.clone()).unwrap();
                            this.merge_node(target_world, target_data.clone()).unwrap();

                            // Only add edge if both endpoints are non-stitch nodes
                            // Stitch nodes are partition boundary artifacts that shouldn't appear in ViewportGraph
                            if !matches!(source_data.role, NodeRole::Stitch(_))
                                && !matches!(target_data.role, NodeRole::Stitch(_))
                            {
                                this.merge_edge(
                                    source_world,
                                    target_world,
                                    edge_data.bundle.clone(),
                                )
                            }
                        }
                    }
                }
            } else {
                trace!("Couldn't find a layout for partition {}", partition_idx);
            }
        }

        // Divide the graph up in logical layers by grouping Data nodes by x-coordinate
        this.build_layers_from_coordinates();

        debug!(
            "[DEBUG_TRACE_LAYOUT] new: -> result=ViewportGraph{{graph_nodes={}, nodes_count={}, domain_to_world_count={}, layers_count={}, included_nodes_count={}}}",
            this.graph.node_count(),
            this.node_data_by_pos.len(),
            this.node_positions.len(),
            this.layers.len(),
            this.included_nodes.len()
        );

        this
    }

    /// Build layers by grouping Data nodes by x-coordinate.
    /// All nodes with the same x-coordinate belong to the same layer.
    /// Nodes are sorted first by x, then by y within each layer.
    fn build_layers_from_coordinates(&mut self) {
        debug!(
            "[DEBUG_TRACE_LAYOUT] build_layers_from_coordinates: starting with {} nodes",
            self.node_data_by_pos.len()
        );

        // Collect all Data nodes with their world positions
        let mut data_nodes: Vec<(WorldPos, NodeIndex)> = Vec::new();

        for (world_pos, layout_node) in &self.node_data_by_pos {
            if let crate::layout::NodeRole::Data(domain_idx) = &layout_node.role {
                data_nodes.push((*world_pos, *domain_idx));
            }
        }

        // Sort nodes first by x-coordinate, then by y-coordinate
        data_nodes.sort_by(|a, b| {
            let x_cmp = a.0.x.cmp(&b.0.x);
            if x_cmp == std::cmp::Ordering::Equal {
                a.0.y.cmp(&b.0.y)
            } else {
                x_cmp
            }
        });

        // Group nodes by x-coordinate into layers
        let mut layers: Vec<Vec<NodeIndex>> = Vec::new();
        let mut current_layer: Vec<NodeIndex> = Vec::new();
        let mut current_x: Option<i64> = None;

        for (world_pos, domain_idx) in data_nodes {
            match current_x {
                None => {
                    // First node
                    current_x = Some(world_pos.x);
                    current_layer.push(domain_idx);
                }
                Some(x) if x == world_pos.x => {
                    // Same x-coordinate, add to current layer
                    current_layer.push(domain_idx);
                }
                Some(_) => {
                    // Different x-coordinate, start new layer
                    if !current_layer.is_empty() {
                        layers.push(current_layer);
                    }
                    current_layer = vec![domain_idx];
                    current_x = Some(world_pos.x);
                }
            }
        }

        // Add the last layer
        if !current_layer.is_empty() {
            layers.push(current_layer);
        }

        self.layers = layers;
    }

    /// Add node to the viewport graph.
    /// HashMap and UnGraphMap handle duplicate insertions gracefully (replace/ignore).
    fn merge_node(&mut self, world_pos: WorldPos, layout_node: LayoutNode) -> Result<(), String> {
        // Skip stitch nodes entirely - they should not appear in the final ViewportGraph
        if matches!(layout_node.role, NodeRole::Stitch(_)) {
            return Ok(());
        }

        // Update domain to world mapping for data nodes
        if let NodeRole::Data(domain_idx) = layout_node.role {
            self.node_positions.insert(domain_idx, world_pos);
        }

        self.node_data_by_pos.insert(world_pos, layout_node);
        self.graph.add_node(world_pos);
        Ok(())
    }

    /// Merge edge into the viewport graph.
    /// Adds edge pairs to existing bundle with deduplication, or creates new edge.
    fn merge_edge(
        &mut self,
        source: WorldPos,
        target: WorldPos,
        bundle: Vec<(NodeIndex, NodeIndex)>,
    ) {
        // Get existing bundle or create empty vec
        let mut combined_bundle = self
            .graph
            .edge_weight(source, target)
            .cloned()
            .unwrap_or_default();

        // Append new pairs with deduplication
        for pair in bundle {
            if !combined_bundle.contains(&pair) {
                combined_bundle.push(pair);
            }
        }

        // Update the edge (replaces if exists, creates if not)
        self.graph.add_edge(source, target, combined_bundle);
    }

    /// Get all nodes in the viewport graph
    pub fn nodes(&self) -> impl Iterator<Item = (&WorldPos, &LayoutNode)> {
        self.node_data_by_pos.iter()
    }

    /// Get all edges in the viewport graph
    pub fn edges(
        &self,
    ) -> impl Iterator<Item = (WorldPos, WorldPos, &Vec<(NodeIndex, NodeIndex)>)> + '_ {
        self.graph.all_edges()
    }

    /// Get node at a specific world position
    pub fn get_node(&self, pos: &WorldPos) -> Option<&LayoutNode> {
        let result = self.node_data_by_pos.get(pos);
        debug!(
            "[DEBUG_TRACE_LAYOUT] get_node: pos={{x={}, y={}}} -> result={:?}",
            pos.x,
            pos.y,
            result.map(|n| format!("Some(LayoutNode{{role={:?}, layer={:?}}})", n.role, n.layer))
        );
        result
    }

    /// Get neighbors of a node at a world position
    pub fn neighbors(&self, pos: WorldPos) -> impl Iterator<Item = WorldPos> + '_ {
        self.graph.neighbors(pos)
    }

    /// Check if graph contains a node at world position
    pub fn contains_node(&self, pos: &WorldPos) -> bool {
        self.node_data_by_pos.contains_key(pos)
    }

    /// Get the number of nodes in the viewport graph
    pub fn node_count(&self) -> usize {
        self.node_data_by_pos.len()
    }

    /// Get the number of edges in the viewport graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get nodes at a specific layer
    pub fn get_layer(&self, layer: usize) -> Option<&Vec<NodeIndex>> {
        let result = self.layers.get(layer);
        debug!(
            "[DEBUG_TRACE_LAYOUT] get_layer: layer={} -> result={:?}",
            layer,
            result.map(|nodes| format!(
                "Some(Vec<NodeIndex> with {} nodes: {:?})",
                nodes.len(),
                nodes
            ))
        );
        result
    }

    /// Get the number of layers in the graph
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Find which layer a domain node belongs to
    pub fn find_domain_node_layer(&self, domain_idx: NodeIndex) -> Option<usize> {
        self.layers
            .iter()
            .position(|layer| layer.contains(&domain_idx))
    }

    #[allow(dead_code)]
    /// Find all adjacent Data nodes by traversing the graph through Routing nodes.
    /// Returns each adjacent Data node along with the path taken to reach it.
    ///
    /// # Parameters
    /// - `domain_node`: Starting domain node index
    ///
    /// # Returns
    /// Vector of tuples containing:
    /// - The adjacent Data node's world position
    /// - The LayoutNode reference
    /// - The path taken (including source and target positions)
    pub fn find_adjacent_data_nodes(
        &self,
        domain_node: NodeIndex,
    ) -> Vec<(WorldPos, &LayoutNode, Vec<WorldPos>)> {
        let mut results = Vec::new();

        // Find the world position for this domain node, or return early
        let Some(&from_pos) = self.node_positions.get(&domain_node) else {
            return results;
        };

        // Make sure we have the LayoutNode too
        if !self.node_data_by_pos.contains_key(&from_pos) {
            return results;
        }

        // Use BFS to find all adjacent Data nodes, tracking paths
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        // Queue entries: (current_pos, path_to_current)
        visited.insert(from_pos);

        // Start with immediate neighbors, each gets the source in its path
        for neighbor_pos in self.graph.neighbors(from_pos) {
            if !visited.contains(&neighbor_pos) {
                queue.push_back((neighbor_pos, vec![from_pos]));
                visited.insert(neighbor_pos);
            }
        }

        debug!(
            "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: starting BFS traversal, queue_size={}",
            queue.len()
        );

        // Traverse the graph to find all adjacent Data nodes
        while let Some((current_pos, mut path)) = queue.pop_front() {
            // Add current position to path
            path.push(current_pos);

            debug!(
                "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: visiting current_pos={{x={}, y={}}}, path_length={}, path={:?}",
                current_pos.x,
                current_pos.y,
                path.len(),
                path
            );

            if let Some(node) = self.node_data_by_pos.get(&current_pos) {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: found node={{role={:?}, layer={:?}}}",
                    node.role, node.layer
                );

                // Check if this is a Data node
                if let NodeRole::Data(domain_idx) = &node.role {
                    debug!(
                        "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: found adjacent Data node domain_idx={:?}, path={:?}",
                        domain_idx, path
                    );
                    // Found an adjacent Data node - record it with its path
                    results.push((current_pos, node, path));
                    // Don't traverse beyond Data nodes
                    continue;
                }

                // If it's a Routing node, continue traversing
                if matches!(node.role, NodeRole::Routing) {
                    let neighbors: Vec<_> = self.graph.neighbors(current_pos).collect();
                    debug!(
                        "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: routing node, neighbors_count={}, neighbors={:?}",
                        neighbors.len(),
                        neighbors
                    );

                    for next_pos in neighbors {
                        if !visited.contains(&next_pos) {
                            debug!(
                                "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: adding unvisited neighbor {{x={}, y={}}} to queue",
                                next_pos.x, next_pos.y
                            );
                            visited.insert(next_pos);
                            // Clone the path for this branch of exploration
                            queue.push_back((next_pos, path.clone()));
                        }
                    }
                }
            } else {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: no node found at current_pos={{x={}, y={}}}",
                    current_pos.x, current_pos.y
                );
            }
        }

        debug!(
            "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: -> result=Vec with {} adjacent data nodes",
            results.len()
        );
        for (i, (pos, node, path)) in results.iter().enumerate() {
            if let NodeRole::Data(domain_idx) = &node.role {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes:   result[{}]: pos={{x={}, y={}}}, domain_idx={:?}, path_length={}, path={:?}",
                    i,
                    pos.x,
                    pos.y,
                    domain_idx,
                    path.len(),
                    path
                );
            }
        }

        results
    }

    /// Create a new CroppedGraph from a list of visual edges.
    /// This is useful for creating temporary graphs for highlighting.
    pub fn from_visual_edges(edges: &[((WorldPos, WorldPos), crate::plotter::PathStyle)]) -> Self {
        let mut new_graph = Self::empty();
        for ((source, target), _) in edges {
            new_graph.graph.add_edge(*source, *target, Vec::new());
        }
        new_graph
    }

    /// Invert edge bundle information to map domain edges to their visual segments.
    /// Returns a HashMap where keys are domain edge pairs (NodeIndex, NodeIndex)
    /// and values are vectors of visual edge segments (WorldPos, WorldPos) that
    /// represent that domain edge.
    pub fn invert_edge_bundles(
        &self,
    ) -> HashMap<(NodeIndex, NodeIndex), Vec<(WorldPos, WorldPos)>> {
        debug!(
            "[DEBUG_TRACE_LAYOUT] invert_edge_bundles: starting with {} edges",
            self.edge_count()
        );
        let mut domain_to_visual = HashMap::new();

        // Iterate through all edges in the viewport graph
        for (source_pos, target_pos, bundle) in self.edges() {
            debug!(
                "[DEBUG_TRACE_LAYOUT] invert_edge_bundles: processing edge source={{x={}, y={}}} -> target={{x={}, y={}}}, bundle_count={}",
                source_pos.x,
                source_pos.y,
                target_pos.x,
                target_pos.y,
                bundle.len()
            );

            // For each domain edge in this visual segment's bundle
            for &domain_edge in bundle {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] invert_edge_bundles: mapping domain_edge={:?} -> visual_segment={{source={{x={}, y={}}}, target={{x={}, y={}}}}}",
                    domain_edge, source_pos.x, source_pos.y, target_pos.x, target_pos.y
                );
                domain_to_visual
                    .entry(domain_edge)
                    .or_insert_with(Vec::new)
                    .push((source_pos, target_pos));
            }
        }

        debug!(
            "[DEBUG_TRACE_LAYOUT] invert_edge_bundles: -> result=HashMap with {} domain edges",
            domain_to_visual.len()
        );
        for (domain_edge, visual_segments) in &domain_to_visual {
            debug!(
                "[DEBUG_TRACE_LAYOUT] invert_edge_bundles:   domain_edge={:?} -> {} visual segments",
                domain_edge,
                visual_segments.len()
            );
        }

        domain_to_visual
    }

    /// Create a subgraph containing only the edges (and their connected nodes)
    /// that satisfy the given predicate.
    pub fn subgraph<F>(&self, edge_predicate: F) -> Self
    where
        F: Fn(&Vec<(NodeIndex, NodeIndex)>) -> bool,
    {
        let mut new_graph = Self::empty();

        // Iterate through all edges in the current graph
        for (source, target, bundle) in self.edges() {
            if edge_predicate(bundle) {
                // Add source node if not already present
                if let std::collections::hash_map::Entry::Vacant(e) =
                    new_graph.node_data_by_pos.entry(source)
                    && let Some(node) = self.node_data_by_pos.get(&source)
                {
                    e.insert(node.clone());
                    new_graph.graph.add_node(source);

                    // If it's a Data node, also update the node_positions map
                    if let crate::layout::NodeRole::Data(domain_idx) = node.role {
                        new_graph.node_positions.insert(domain_idx, source);
                    }
                }

                // Add target node if not already present
                if let std::collections::hash_map::Entry::Vacant(e) =
                    new_graph.node_data_by_pos.entry(target)
                    && let Some(node) = self.node_data_by_pos.get(&target)
                {
                    e.insert(node.clone());
                    new_graph.graph.add_node(target);

                    // If it's a Data node, also update the node_positions map
                    if let crate::layout::NodeRole::Data(domain_idx) = node.role {
                        new_graph.node_positions.insert(domain_idx, target);
                    }
                }

                // Add the edge with its bundle
                new_graph.graph.add_edge(source, target, bundle.clone());
            }
        }

        new_graph
    }

    /// Find the Data node closest to the centroid of all Data nodes in the viewport.
    /// Returns the world position and domain index of the center node.
    /// Returns None if there are no Data nodes in the viewport.
    pub fn find_center(&self) -> Option<(WorldPos, NodeIndex)> {
        let mut positions = Vec::new();
        for layer in &self.layers {
            for &domain_idx in layer {
                if let Some(&world_pos) = self.node_positions.get(&domain_idx) {
                    positions.push((world_pos, domain_idx));
                }
            }
        }

        if positions.is_empty() {
            return None;
        }

        let sum_x: i64 = positions.iter().map(|(pos, _)| pos.x).sum();
        let sum_y: i64 = positions.iter().map(|(pos, _)| pos.y).sum();
        let count = positions.len() as i64;
        let centroid_x = sum_x / count;
        let centroid_y = sum_y / count;

        positions.into_iter().min_by_key(|(pos, _)| {
            let dx = pos.x - centroid_x;
            let dy = pos.y - centroid_y;
            dx * dx + dy * dy
        })
    }
}

impl Default for CroppedGraph {
    fn default() -> Self {
        Self::empty()
    }
}
