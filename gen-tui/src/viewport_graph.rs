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
    /// Sub-rect highlights keyed by world position (node centre).
    /// tl/br are node-local (col, row) offsets from the node's top-left corner,
    /// both inclusive.
    #[allow(clippy::type_complexity)]
    pub cell_highlights: Vec<(WorldPos, (i64, i64), (i64, i64), crate::plotter::PathStyle)>,

    /// Visual edge segments that are dimmed: stored as (source, target) world positions.
    /// Populated by apply_lowlights from domain-level edge_lowlights on each rebuild.
    pub edge_lowlights: Vec<(WorldPos, WorldPos)>,

    /// Visual node positions that are dimmed: stored as world positions.
    /// Populated by apply_node_lowlights from domain-level node_lowlights on each rebuild.
    pub node_lowlights: Vec<WorldPos>,

    /// Domain edges rewired onto pin nodes (see `PartitionTable::backward_edges`), copied
    /// through so the plotter can mark the rendered bypass edge with direction arrows.
    pub backward_edges: Vec<(NodeIndex, NodeIndex)>,
    /// `(source_partition_idx, target_partition_idx)` for each entry in `backward_edges`,
    /// aligned by index (see `PartitionTable::backward_edge_partitions`).
    pub backward_edge_partitions: Vec<(PartitionIndex, PartitionIndex)>,
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
            cell_highlights: Vec::new(),
            edge_lowlights: Vec::new(),
            node_lowlights: Vec::new(),
            backward_edges: Vec::new(),
            backward_edge_partitions: Vec::new(),
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
            + petgraph::visit::EdgeIndexable
            + petgraph::visit::NodeIndexable
            + petgraph::visit::NodeCount
            + petgraph::visit::Visitable,
        G::NodeId: Copy + Eq + std::hash::Hash + Ord,
        G::EdgeId: Clone,
        for<'b> &'b G: petgraph::visit::GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
            + petgraph::visit::IntoEdgeReferences<NodeId = G::NodeId, EdgeId = G::EdgeId>
            + petgraph::visit::IntoNeighborsDirected<NodeId = G::NodeId>
            + petgraph::visit::IntoNodeIdentifiers<NodeId = G::NodeId>,
    {
        debug!(
            "[DEBUG_TRACE_LAYOUT] new CroppedGraph: viewport=BigRect{{min={{x={}, y={}}}, max={{x={}, y={}}}}}, detail_level={:?}, covered_partitions={:?}",
            viewport.min.x,
            viewport.min.y,
            viewport.max.x,
            viewport.max.y,
            detail_level,
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

                // Routing nodes at the partition's right edge have bounding rects starting
                // at local_corner_max.x + 1, so extend the query envelope by 1 unit to the
                // right to avoid excluding the rightmost column of routing nodes (and their
                // connecting edges) from the viewport graph.
                let envelope = rstar::AABB::from_corners(
                    [local_corner_min.x, local_corner_min.y],
                    [local_corner_max.x + 1, local_corner_max.y],
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

                            let source_world =
                                partition_table.local_to_world(source_data.pos, detail_level);
                            let target_world =
                                partition_table.local_to_world(target_data.pos, detail_level);

                            this.merge_node(source_world, source_data.clone()).unwrap();
                            this.merge_node(target_world, target_data.clone()).unwrap();

                            // Don't add edges to stitch nodes
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
        this.backward_edges = partition_table.backward_edges.clone();
        this.backward_edge_partitions = partition_table.backward_edge_partitions.clone();

        this
    }

    /// Build layers by grouping Data nodes by their Sugiyama rank within each partition.
    ///
    /// Nodes in the same (partition_idx, layer) belong to the same logical rank and should
    /// navigate together vertically — even when redistribution has given them different
    /// visual x-coordinates. Groups are ordered left-to-right by their minimum world x,
    /// giving a globally consistent layer index across partition boundaries.
    fn build_layers_from_coordinates(&mut self) {
        // (partition_idx, sugiyama_layer) -> Vec<(world_y, domain_idx)>
        let mut groups: HashMap<(PartitionIndex, i32), Vec<(i64, NodeIndex)>> = HashMap::new();
        // Minimum world x seen for each group — used for left-to-right sort order.
        let mut group_rep_x: HashMap<(PartitionIndex, i32), i64> = HashMap::new();

        for (world_pos, layout_node) in &self.node_data_by_pos {
            if let crate::layout::NodeRole::Data(domain_idx) = &layout_node.role
                && let Some(layer) = layout_node.layer
            {
                let key = (layout_node.pos.partition_idx, layer);
                groups
                    .entry(key)
                    .or_default()
                    .push((world_pos.y, *domain_idx));
                group_rep_x
                    .entry(key)
                    .and_modify(|x| *x = (*x).min(world_pos.x))
                    .or_insert(world_pos.x);
            }
        }

        // Sort groups left-to-right by representative x for a globally consistent ordering.
        let mut sorted_groups: Vec<_> = groups.into_iter().collect();
        sorted_groups.sort_by_key(|(key, _)| group_rep_x[key]);

        // Within each group sort by y (top-to-bottom), then collect domain indices.
        self.layers = sorted_groups
            .into_iter()
            .map(|(_, mut nodes)| {
                nodes.sort_by_key(|(y, _)| *y);
                nodes.into_iter().map(|(_, idx)| idx).collect()
            })
            .collect();
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

    /// Get all Data nodes in the viewport graph with their world position, domain index, and layout info.
    /// Filters out Routing and Stitch nodes, returning only nodes that represent original graph data.
    pub fn data_nodes(&self) -> impl Iterator<Item = (WorldPos, NodeIndex, &LayoutNode)> + '_ {
        self.node_data_by_pos.iter().filter_map(|(pos, node)| {
            if let NodeRole::Data(idx) = &node.role {
                Some((*pos, *idx, node))
            } else {
                None
            }
        })
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
        self.layers.get(layer)
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

    /// Create a new CroppedGraph from a list of visual (rectilinear) edges.
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
        let mut domain_to_visual = HashMap::new();

        // Iterate through all edges in the viewport graph
        for (source_pos, target_pos, bundle) in self.edges() {
            // For each domain edge in this visual segment's bundle
            for &domain_edge in bundle {
                domain_to_visual
                    .entry(domain_edge)
                    .or_insert_with(Vec::new)
                    .push((source_pos, target_pos));
            }
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
