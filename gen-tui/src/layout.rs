use std::{collections::HashMap, hash::Hash};

use gen_sugiyama::{Config, Edge, Vertex, assign_coordinates, run_sugiyama_algorithm};
use itertools::Itertools;
use log::warn;
use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::{StableDiGraph, StableGraph},
    visit::{EdgeRef, IntoEdgeReferences, NodeIndexable},
};
use rstar::{AABB, RTree};
use serde::{Deserialize, Serialize};

use crate::{
    edge_router::route_graph::make_rectilinear,
    geometry::{BigRect, LayoutObject, LayoutPos, LocalPos, PartitionIndex},
    partition::{PartitionEdge, PartitionNode, StitchSide},
    plotter::NodeSizer,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisualDetail {
    Minimal,
    Full,
    Truncated,
}

impl VisualDetail {
    /// Convert VisualDetail enum to array index for layout storage
    pub fn as_index(self) -> usize {
        match self {
            VisualDetail::Minimal => 0,
            VisualDetail::Full => 1,
            VisualDetail::Truncated => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JunctionSymbol {
    pub index: u8,
}

impl JunctionSymbol {
    /// Dashed box-drawing characters for routing nodes
    /// Index: 4-bit value where bits represent [North, East, South, West]
    const DASHED_ROUTING_GLYPHS: [char; 16] = [
        '?', // 0000 ____
        '╴', // 0001 ___W
        '╷', // 0010 __S_
        '╮', // 0011 __SW
        '╶', // 0100 _E__
        '┄', // 0101 _E_W (light triple dash horizontal)
        '╭', // 0110 _ES_
        '┬', // 0111 _ESW
        '╵', // 1000 N___
        '╯', // 1001 N__W
        '┆', // 1010 N_S_ (light triple dash vertical)
        '┤', // 1011 N_SW
        '╰', // 1100 NE__
        '┴', // 1101 NE_W
        '├', // 1110 NES_
        '┼', // 1111 NESW
    ];
    /// Heavy box-drawing characters for highlighted routing nodes
    /// Index: 4-bit value where bits represent [North, East, South, West]
    const HEAVY_ROUTING_GLYPHS: [char; 16] = [
        '?', // 0000 ____
        '╸', // 0001 ___W
        '╻', // 0010 __S_
        '┓', // 0011 __SW
        '╺', // 0100 _E__
        '━', // 0105 _E_W
        '┏', // 0110 _ES_
        '┳', // 0111 _ESW
        '╹', // 1000 N___
        '┛', // 1001 N__W
        '┃', // 1010 N_S_
        '┫', // 1011 N_SW
        '┗', // 1100 NE__
        '┻', // 1101 NE_W
        '┣', // 1110 NES_
        '╋', // 1111 NESW
    ];
    /// Box-drawing characters for routing nodes based on connection directions
    /// Index: 4-bit value where bits represent [North, East, South, West]
    const ROUTING_GLYPHS: [char; 16] = [
        '?', // 0000 ____
        '╴', // 0001 ___W
        '╷', // 0010 __S_
        '╮', // 0011 __SW
        '╶', // 0100 _E__
        '─', // 0101 _E_W
        '╭', // 0110 _ES_
        '┬', // 0111 _ESW
        '╵', // 1000 N___
        '╯', // 1001 N__W
        '│', // 1010 N_S_
        '┤', // 1011 N_SW
        '╰', // 1100 NE__
        '┴', // 1101 NE_W
        '├', // 1110 NES_
        '┼', // 1111 NESW
    ];

    /// Get the routing glyph character for this index
    pub fn glyph(&self) -> char {
        Self::ROUTING_GLYPHS
            .get(self.index as usize)
            .copied()
            .unwrap_or('?')
    }

    /// Get the heavy routing glyph character for this index
    pub fn heavy_glyph(&self) -> char {
        Self::HEAVY_ROUTING_GLYPHS
            .get(self.index as usize)
            .copied()
            .unwrap_or('?')
    }

    /// Get the dashed routing glyph character for this index
    pub fn dashed_glyph(&self) -> char {
        Self::DASHED_ROUTING_GLYPHS
            .get(self.index as usize)
            .copied()
            .unwrap_or('?')
    }

    /// Create a new GlyphIndex from a u8 value
    pub fn new(value: u8) -> Self {
        Self { index: value }
    }
}

/// NodeRole represents the distinction between nodes that were in the input graph (Data)  
/// and dummy nodes that were added to route the edges during layout (Routing).
/// The Data variant stores the original domain graph NodeIndex.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeRole {
    Data(NodeIndex),
    Routing, // No stored data - glyph computed on-the-fly from connectivity
    Stitch(StitchSide),
}

/// Layout graphs contain two types of nodes: nodes that represent the input nodes,
/// and new nodes that were added to route the edges during layout. The role field
/// with NodeRole enum indicate which type the node is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutNode {
    pub role: NodeRole,
    pub pos: LocalPos,
    pub size: (u64, u64),
    /// Layer information from Sugiyama algorithm. Only valid for Data nodes.
    pub layer: Option<i32>,
}

impl LayoutNode {
    pub fn new(role: NodeRole, pos: LocalPos, size: (u64, u64), layer: Option<i32>) -> Self {
        Self {
            role,
            pos,
            size,
            layer,
        }
    }

    pub fn data(
        domain_node_idx: NodeIndex,
        pos: LocalPos,
        size: (u64, u64),
        layer: Option<i32>,
    ) -> Self {
        Self::new(NodeRole::Data(domain_node_idx), pos, size, layer)
    }

    pub fn routing(pos: LocalPos, size: (u64, u64)) -> Self {
        Self::new(NodeRole::Routing, pos, size, None)
    }

    pub fn stitch(side: StitchSide, pos: LocalPos, size: (u64, u64)) -> Self {
        Self::new(NodeRole::Stitch(side), pos, size, None)
    }

    /// Get the partition index for this node
    pub fn partition_idx(&self) -> PartitionIndex {
        self.pos.partition_idx
    }
}

/// LayoutEdge represents a bundle of edges as a vector of node index pairs.
/// During layout, multiple edges may be bundled together for simplified visualization
/// and routing purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayoutEdge {
    pub bundle: Vec<(NodeIndex, NodeIndex)>,
}

impl LayoutEdge {
    pub fn new(source: NodeIndex, target: NodeIndex) -> Self {
        Self {
            bundle: vec![(source, target)],
        }
    }

    pub fn empty() -> Self {
        Self { bundle: Vec::new() }
    }

    pub fn append(&mut self, source: NodeIndex, target: NodeIndex) {
        let pair = (source, target);
        if !self.bundle.contains(&pair) {
            self.bundle.push(pair);
        }
    }
}

impl Default for LayoutEdge {
    fn default() -> Self {
        Self::empty()
    }
}

/// A single coordinate layout for a specific zoom level,
/// has its own graph (undirected to allow for bidirectional edge routing),
/// coordinates mapped to nodes, and a spatial index for all objects.
#[derive(Clone, Serialize)]
pub struct PartitionLayout {
    pub graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    /// Single spatial index that holds both domain and layout indices
    #[serde(skip)]
    pub spatial_index: RTree<LayoutObject>,
    pub width: i64,
    pub height: i64,
}

impl PartitionLayout {
    /// Create a PartitionLayout for a regular partition (section)
    pub fn for_section(
        mut layout_graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
        vertex_spacing: f64,
    ) -> Self {
        log::trace!(
            "for_section: input layout_graph has {} nodes, {} edges",
            layout_graph.node_count(),
            layout_graph.edge_count()
        );

        if let Err(e) = make_rectilinear(&mut layout_graph, vertex_spacing) {
            log::warn!("Edge routing failed: {:?}", e);
        }

        let (dx, dy) = align_partition_to_origin(&mut layout_graph);

        let spatial_index = Self::build_spatial_index(&layout_graph);

        log::trace!(
            "for_section: final layout_graph has {} nodes, {} edges, width: {}, height: {}",
            layout_graph.node_count(),
            layout_graph.edge_count(),
            dx,
            dy
        );

        Self {
            graph: layout_graph,
            spatial_index,
            width: dx,
            height: dy,
        }
    }

    /// Create a PartitionLayout for an inter-partition space (bridge)
    pub fn for_bridge(
        mut layout_graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
        vertex_spacing: f64,
    ) -> Self {
        log::trace!("for_bridge: entering function");
        log::trace!(
            "for_bridge: input layout_graph has {} nodes, {} edges",
            layout_graph.node_count(),
            layout_graph.edge_count()
        );

        if let Err(e) = make_rectilinear(&mut layout_graph, vertex_spacing) {
            log::warn!("Edge routing failed: {:?}", e);
        }

        let spatial_index = Self::build_spatial_index(&layout_graph);

        let (min_x, max_x) = layout_graph
            .node_weights()
            .map(|node| node.pos.x)
            .minmax()
            .into_option()
            .unwrap_or((0, 0));

        log::trace!(
            "for_bridge: final layout_graph has {} nodes, {} edges, width: {}, height: {}",
            layout_graph.node_count(),
            layout_graph.edge_count(),
            max_x - min_x,
            0
        );

        Self {
            graph: layout_graph,
            spatial_index,
            width: max_x - min_x,
            height: 0,
        }
    }

    pub fn build_spatial_index(
        graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    ) -> RTree<LayoutObject> {
        log::debug!(
            "build_spatial_index: layout_graph has {} nodes",
            graph.node_count()
        );

        // Debug: Check for duplicate domain nodes in the layout graph
        let mut domain_node_counts = std::collections::HashMap::new();
        for node_idx in graph.node_indices() {
            if let Some(node_data) = graph.node_weight(node_idx)
                && let NodeRole::Data(domain_idx) = &node_data.role
            {
                *domain_node_counts.entry(*domain_idx).or_insert(0) += 1;
                log::debug!(
                    "build_spatial_index: found domain node {:?} at layout_pos({}, {}) layout_idx={:?}",
                    domain_idx,
                    node_data.pos.x,
                    node_data.pos.y,
                    node_idx
                );
            }
        }

        // Report duplicates
        for (domain_idx, count) in &domain_node_counts {
            if *count > 1 {
                log::debug!(
                    "build_spatial_index: WARNING - domain node {:?} appears {} times in layout graph!",
                    domain_idx,
                    count
                );
            }
        }
        let node_index_mapping: HashMap<NodeIndex<u32>, NodeIndex<u32>> = graph
            .node_indices()
            .map(|node_idx| (node_idx, node_idx))
            .collect();

        let mut objects = graph
            .node_indices()
            .map(|local_node_idx| {
                let node_data = graph
                    .node_weight(local_node_idx)
                    .expect("Node data not found");
                let new_idx = node_index_mapping
                    .get(&local_node_idx)
                    .expect("Provided mapping not complete");
                match &node_data.role {
                    NodeRole::Data(original_node_idx) => LayoutObject::node(
                        LayoutPos::new(node_data.pos.x, node_data.pos.y),
                        node_data.size,
                        *new_idx,
                        *original_node_idx,
                    ),
                    NodeRole::Routing => LayoutObject::routing_node(
                        LayoutPos::new(node_data.pos.x, node_data.pos.y),
                        *new_idx,
                    ),
                    NodeRole::Stitch(stitch_side) => LayoutObject::stitch_node(
                        LayoutPos::new(node_data.pos.x, node_data.pos.y),
                        *new_idx,
                        *stitch_side,
                    ),
                }
            })
            .collect::<Vec<_>>();

        objects.extend(graph.edge_references().map(|edge_ref| {
            let source_id = edge_ref.source();
            let source_node = graph.node_weight(source_id).expect("Source node not found");
            let target_id = edge_ref.target();
            let target_node = graph.node_weight(target_id).expect("Target node not found");

            LayoutObject::line(
                LayoutPos::new(source_node.pos.x, source_node.pos.y),
                LayoutPos::new(target_node.pos.x, target_node.pos.y),
                source_id,
                target_id,
            )
        }));

        RTree::bulk_load(objects)
    }

    /// Find all nodes that overlap a Rect given in local coordinates
    pub fn find_nodes_in_rect(&self, rect: BigRect<i64>) -> Vec<LayoutObject> {
        let envelope = AABB::from_corners(rect.min.into(), rect.max.into());
        self.spatial_index
            .locate_in_envelope_intersecting(&envelope)
            .filter(|rect| rect.is_node())
            .cloned()
            .collect()
    }

    /// Find all edges overlapping a viewport rectangle
    pub fn find_edges_in_rect(&self, viewport: BigRect<i64>) -> Vec<LayoutObject> {
        let envelope = AABB::from_corners(viewport.min.into(), viewport.max.into());

        let all_objects: Vec<_> = self
            .spatial_index
            .locate_in_envelope_intersecting(&envelope)
            .cloned()
            .collect();

        let edge_objects: Vec<_> = all_objects
            .into_iter()
            .filter(|rect| rect.is_edge())
            .collect();

        edge_objects
    }

    /// Check if the layout and/or flower is empty (no nodes)
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Check if the layout contains a specific node
    pub fn contains_key(&self, node_idx: &NodeIndex<u32>) -> bool {
        self.graph.node_weight(*node_idx).is_some()
    }

    /// Get the position of a node in the layout's coordinate system
    pub fn get_node_position(&self, node_idx: NodeIndex<u32>) -> Option<LocalPos> {
        self.graph.node_weight(node_idx).map(|node| node.pos)
    }

    /// Set the position of a node and update the spatial index
    pub fn set_node_position(&mut self, node_idx: NodeIndex<u32>, pos: LocalPos) -> bool {
        if let Some(node) = self.graph.node_weight_mut(node_idx) {
            node.pos = pos;

            // Update the spatial index by removing and re-adding the object
            // First, find and remove the old object
            let objects_to_remove: Vec<_> = self
                .spatial_index
                .iter()
                .filter(|obj| obj.primary_node.layout == node_idx)
                .cloned()
                .collect();

            for old_obj in objects_to_remove {
                self.spatial_index.remove(&old_obj);

                // Create new object with updated position
                let new_obj = match old_obj.object_type {
                    crate::geometry::SpatialObjectType::DataNode(original_idx) => {
                        crate::geometry::LayoutObject::node(
                            pos.pos(),
                            node.size,
                            node_idx,
                            original_idx,
                        )
                    }
                    crate::geometry::SpatialObjectType::RoutingNode(node_idx) => {
                        crate::geometry::LayoutObject::routing_node(pos.pos(), node_idx)
                    }
                    crate::geometry::SpatialObjectType::StitchNode(side) => {
                        crate::geometry::LayoutObject::stitch_node(pos.pos(), node_idx, side)
                    }
                    _ => continue, // Skip edges
                };

                self.spatial_index.insert(new_obj);
            }
            true
        } else {
            false
        }
    }

    /// Get the size of a node
    pub fn get_node_size(&self, node_idx: NodeIndex<u32>) -> Option<(u64, u64)> {
        self.graph.node_weight(node_idx).map(|node| node.size)
    }

    /// Get all node positions as an iterator
    pub fn get_all_positions(&self) -> impl Iterator<Item = (NodeIndex<u32>, LocalPos)> + '_ {
        self.graph
            .node_indices()
            .filter_map(move |idx| self.graph.node_weight(idx).map(|node| (idx, node.pos)))
    }

    /// Find the left and right stitch nodes in the layout, if they exist.
    /// Returns a tuple of (left_stitch, right_stitch) where each element is an Option<NodeIndex>.
    pub fn find_stitch_nodes(&self) -> (Option<NodeIndex>, Option<NodeIndex>) {
        let left_stitch = self.graph.node_indices().find(|&idx| {
            self.graph
                .node_weight(idx)
                .is_some_and(|node| matches!(node.role, NodeRole::Stitch(StitchSide::Left)))
        });

        let right_stitch = self.graph.node_indices().find(|&idx| {
            self.graph
                .node_weight(idx)
                .is_some_and(|node| matches!(node.role, NodeRole::Stitch(StitchSide::Right)))
        });
        (left_stitch, right_stitch)
    }
}

impl std::fmt::Debug for PartitionLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("PartitionLayout");

        debug_struct
            .field("width", &self.width)
            .field("height", &self.height)
            .field("node_count", &self.graph.node_count())
            .field("edge_count", &self.graph.edge_count());

        // Add positions for all nodes
        let mut positions = Vec::new();
        for node_idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(node_idx) {
                positions.push((node_idx, node.pos, &node.role));
            }
        }
        debug_struct.field("node_positions", &positions);

        // Add edge bundle information
        let mut edge_bundles = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge) = self.graph.edge_weight(edge_idx)
                && let Some((source, target)) = self.graph.edge_endpoints(edge_idx)
            {
                edge_bundles.push((source, target, &edge.bundle));
            }
        }
        debug_struct.field("edge_bundles", &edge_bundles);

        debug_struct.finish()
    }
}

/// A builder for creating a PartitionLayout from a StableDiGraph<PartitionNode, PartitionEdge, u32>.
/// Generate multiple PartitionLayouts (e.g. zoom levels) on a common layer framework,
/// by reusing intermediate data from the Sugiyama algorithm. This ensures
/// that the relative orientation of nodes now and later is preserved.
#[derive(Debug)]
pub struct LayoutEngine<'a> {
    partition_graph: &'a StableDiGraph<PartitionNode, PartitionEdge, u32>,
    partition_idx: PartitionIndex,
    vertex_graph: StableDiGraph<Vertex, Edge, u32>,
    vertex_layers: Option<Vec<Vec<NodeIndex<u32>>>>,
    config: Config,
}

impl<'a> LayoutEngine<'a> {
    /// Create a new LayoutEngine for the given partition graph.
    pub fn new(
        partition_graph: &'a StableDiGraph<PartitionNode, PartitionEdge, u32>,
        partition_idx: PartitionIndex,
    ) -> Self {
        // Make a vertex graph for the sugiyama algorithm
        let mut vertex_graph = StableDiGraph::<Vertex, Edge, u32>::with_capacity(
            partition_graph.node_count(),
            partition_graph.edge_count(),
        );

        // Add all nodes and store mapping from partition indices to vertex indices
        let node_map: HashMap<NodeIndex<u32>, NodeIndex<u32>> = partition_graph
            .node_indices()
            .map(|partition_idx| {
                let mut vertex = Vertex::new(partition_idx);
                let sort_bias = match &partition_graph[partition_idx] {
                    PartitionNode::Data(domain_idx) => {
                        let d: i32 = i32::try_from(domain_idx.index()).unwrap_or(i32::MAX);
                        i32::MAX.saturating_sub(d)
                    }
                    PartitionNode::Stitch(_) => 0,
                };
                vertex.set_sort_bias(sort_bias);
                let new_vertex_idx = vertex_graph.add_node(vertex);
                (partition_idx, new_vertex_idx)
            })
            .collect();

        // Add all edges using the map
        for edge_idx in partition_graph.edge_indices() {
            if let Some((src_partition_idx, dst_partition_idx)) =
                partition_graph.edge_endpoints(edge_idx)
            {
                let src_vertex_idx = node_map[&src_partition_idx];
                let dst_vertex_idx = node_map[&dst_partition_idx];

                // Get the edge weight to determine if this edge should have a label
                let partition_edge = partition_graph
                    .edge_weight(edge_idx)
                    .expect("Edge weight not found for edge");

                // Use the edge weight to determine Edge construction
                let edge = match partition_edge {
                    Some((src_domain_idx, dst_domain_idx)) => {
                        // Edge has domain node information, use it as label
                        Edge::default().with_label((*src_domain_idx, *dst_domain_idx))
                    }
                    None => {
                        // Edge has no domain information (e.g., the initial stitch edges at the
                        // beginning and end of the graph (TODO: think about omitting those
                        // entirely)
                        Edge::default()
                    }
                };

                vertex_graph.add_edge(src_vertex_idx, dst_vertex_idx, edge);
            }
        }

        Self {
            partition_graph,
            partition_idx,
            vertex_graph,
            vertex_layers: None,
            config: Config::default(),
        }
    }

    pub fn set_sugiyama_config(&mut self, config: Config) {
        self.config = config;
    }

    /// Set the vertex spacing for this layout engine
    pub fn set_vertex_spacing(&mut self, spacing: f64) {
        self.config.vertex_spacing = spacing;
    }

    /// Get the current vertex spacing
    pub fn get_vertex_spacing(&self) -> f64 {
        self.config.vertex_spacing
    }

    pub fn compute_layout<S>(
        &mut self,
        node_sizer: &S,
        detail_level: VisualDetail,
    ) -> Result<PartitionLayout, String>
    where
        S: for<'g> NodeSizer<&'g StableDiGraph<PartitionNode, PartitionEdge, u32>>,
    {
        // Check for empty graph
        if self.partition_graph.node_count() == 0 {
            return Err("Cannot compute layout for empty graph".to_string());
        }

        // Special case: Single node graph
        // Bypass Sugiyama algorithm and edge routing entirely
        if self.partition_graph.node_count() == 1 {
            let node_idx = self.partition_graph.node_indices().next().unwrap();
            let partition_node = &self.partition_graph[node_idx];

            // Calculate size
            let size = if let PartitionNode::Data(_domain_idx) = partition_node {
                // For Data nodes, use the node sizer
                node_sizer.get_node_size(&node_idx, detail_level)
            } else {
                // For Stitch nodes, use dummy size
                node_sizer.get_dummy_size()
            };

            let (width, height) = (
                (size.0 as f64 + self.config.vertex_spacing).round() as u64,
                (size.1 as f64 + self.config.vertex_spacing).round() as u64,
            );

            // Create LayoutNode at (0,0)
            let pos = LocalPos::new(self.partition_idx, LayoutPos::ZERO);
            let role = match partition_node {
                PartitionNode::Data(d) => NodeRole::Data(*d),
                PartitionNode::Stitch(s) => NodeRole::Stitch(*s),
            };
            let layout_node = LayoutNode::new(role, pos, (width, height), Some(0));

            let mut layout_graph = StableGraph::default();
            layout_graph.add_node(layout_node);

            let spatial_index = PartitionLayout::build_spatial_index(&layout_graph);

            return Ok(PartitionLayout {
                graph: layout_graph,
                spatial_index,
                width: width as i64,
                height: height as i64,
            });
        }

        // Check for disconnected graph (multiple components)
        // A graph with multiple nodes must be connected to be laid out
        // The sugiyama algorithm (phase 1: ranking) will panic if the graph is disconnected
        // because it expects to be able to build a spanning tree.
        let components = count_connected_components(&self.vertex_graph);
        if components > 1 {
            return Err(format!(
                "Graph is disconnected ({} components). Layout requires a connected graph.",
                components
            ));
        }

        // Phases 1 & 2 of the Sugiyama algorithm organize the nodes into layers
        // and add dummy nodes to the graph. We do this only once and store the result
        // in the LayoutEngine.
        if self.vertex_layers.is_none() {
            self.vertex_layers = Some(run_sugiyama_algorithm(&mut self.vertex_graph, &self.config));
        }

        // Make a fresh copy for this specific layout computation
        // (if memory ends up being an issue, we can probably figure out
        // which variables to reset instead)
        let mut working_layers = self.vertex_layers.clone().expect("Could not derive layers");
        let mut working_graph = self.vertex_graph.clone();

        for vertex_idx in working_graph.node_indices().collect::<Vec<_>>() {
            let size = if let Some(partition_idx) = working_graph[vertex_idx].input_node_idx {
                // Use partition index to get node size from partition graph
                let partition_node_id = self.partition_graph.from_index(partition_idx.index());
                node_sizer.get_node_size(&partition_node_id, detail_level)
            } else {
                node_sizer.get_dummy_size()
            };
            working_graph[vertex_idx].set_size(size, self.config.vertex_spacing);
        }

        // Phase 3 of the algorithm assigns provisional coordinates to each node,
        // based on how much space each label will take up once rendered.
        // If no dimensions are provided, 1x1 dimensions are assumed and a standard marker
        // is used as label instead.
        let vertex_coords_map = assign_coordinates(&mut working_layers, &mut working_graph)
            .into_iter()
            .collect::<HashMap<NodeIndex<u32>, (i64, i64)>>();

        let layout_graph = self.build_layout_graph(&working_graph, vertex_coords_map);

        // Save the layout and associated data using the section constructor
        // todo: move more of this logic into partitionlayout constructor
        let layout = PartitionLayout::for_section(layout_graph, self.config.vertex_spacing);

        Ok(layout)
    }

    /// Converts output of Sugiyama algorithm into a layout graph that holds:
    /// - layoutnodes which refer to the original input node or dummy nodes
    ///   to route edges around other nodes and each other.
    /// - node positions in LocalPos format (referenced to the partition)
    /// - node sizes
    ///
    ///   Layout graphs are undirected because in rectilinear routing one vertical edge
    ///   may carry signals in both directions.
    pub fn build_layout_graph(
        &mut self,
        sugiyama_graph: &StableDiGraph<Vertex, Edge, u32>,
        vertex_coords_map: HashMap<NodeIndex<u32>, (i64, i64)>,
    ) -> StableGraph<LayoutNode, LayoutEdge, Undirected, u32> {
        let mut layout_graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::with_capacity(sugiyama_graph.node_count(), sugiyama_graph.edge_count());

        log::debug!(
            "build_layout_graph: starting with working_graph {} nodes, {} edges",
            sugiyama_graph.node_count(),
            sugiyama_graph.edge_count()
        );

        // Map node indices from working graph to layout graph.
        let mut node_map = HashMap::new();

        for node_idx in sugiyama_graph.node_indices() {
            if let Some(node) = sugiyama_graph.node_weight(node_idx) {
                let layout_pos = vertex_coords_map
                    .get(&node_idx)
                    .map(|&(x, y)| LayoutPos::new(x, y))
                    .unwrap_or_else(|| {
                        warn!(
                            "Could not find coordinates for vertex {:?}, using (0,0)",
                            node_idx
                        );
                        LayoutPos::ZERO
                    });
                let pos = LocalPos::new(self.partition_idx, layout_pos);
                let (width, height) = node.get_size(self.config.vertex_spacing);
                let layer = Some(node.get_rank());

                // If the Vertex was labeled with the input node index,
                // we obtain its role from the partition graph. If it wasn't
                // labeled, we know it's a routing node added by the sugiyama algorithm.
                let role = if let Some(pidx) = node.input_node_idx {
                    match &self.partition_graph[pidx] {
                        PartitionNode::Data(domain_idx) => NodeRole::Data(*domain_idx),
                        PartitionNode::Stitch(side) => NodeRole::Stitch(*side),
                    }
                } else {
                    NodeRole::Routing
                };

                let layout_node = LayoutNode::new(role, pos, (width, height), layer);
                let new_idx = layout_graph.add_node(layout_node);
                node_map.insert(node_idx, new_idx);
            }
        }

        // Collect and aggregate edges by their endpoints to combine bundles
        #[allow(clippy::type_complexity)]
        let mut edge_bundles: HashMap<
            (NodeIndex<u32>, NodeIndex<u32>),
            Vec<(NodeIndex, NodeIndex)>,
        > = HashMap::new();

        for edge_ref in sugiyama_graph.edge_references() {
            let vertex_source_idx = edge_ref.source();
            let vertex_target_idx = edge_ref.target();
            let layout_source_idx = node_map[&vertex_source_idx];
            let layout_target_idx = node_map[&vertex_target_idx];

            if let Some(edge_bundle) = edge_ref.weight().input_node_idx_pair {
                edge_bundles
                    .entry((layout_source_idx, layout_target_idx))
                    .or_default()
                    .push(edge_bundle);
            } else {
                // Ensure edges without bundles still get created
                edge_bundles
                    .entry((layout_source_idx, layout_target_idx))
                    .or_default();
            }
        }

        // Create layout edges with aggregated bundles
        for ((layout_source_idx, layout_target_idx), bundles) in edge_bundles {
            let layout_edge = LayoutEdge { bundle: bundles };
            layout_graph.add_edge(layout_source_idx, layout_target_idx, layout_edge);
        }

        layout_graph
    }
}

/// Helper to count connected components in an undirected sense for a StableGraph
fn count_connected_components<N, E>(graph: &StableDiGraph<N, E, u32>) -> usize {
    let mut visited = std::collections::HashSet::new();
    let mut components = 0;

    for node in graph.node_indices() {
        if !visited.contains(&node) {
            components += 1;
            // BFS traversal
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(node);
            visited.insert(node);

            while let Some(current) = queue.pop_front() {
                for neighbor in graph.neighbors_undirected(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    components
}

fn mean_y_for_x(
    layout_graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    x: i64,
) -> i64 {
    let layer_ys: Vec<i64> = layout_graph
        .node_weights()
        .filter(|node| node.pos.x == x)
        .map(|node| node.pos.y)
        .collect::<Vec<i64>>();

    (layer_ys.iter().sum::<i64>() as f64 / layer_ys.len() as f64).round() as i64
}

/// Take the contents of a layout graph and translate the node coordinates
/// so that the centers of all data nodes in the first layer are aligned to
/// each other and the origin (they all share x=0). In the Y-direction the
/// nodes are shifted so that the mean y=0.
/// - Returns run and rise: the horizontal and vertical distance
///   from the origin to the mean of the rightmost nodes
fn align_partition_to_origin(
    layout_graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
) -> (i64, i64) {
    let (min_x, max_x) = layout_graph
        .node_weights()
        .filter(|node| !matches!(node.role, NodeRole::Stitch(_)))
        .map(|node| node.pos.x)
        .minmax()
        .into_option()
        .unwrap_or((0, 0));

    let mean_y_left = mean_y_for_x(layout_graph, min_x);
    let mean_y_right = mean_y_for_x(layout_graph, max_x);

    // Apply normalization offsets
    for node in layout_graph.node_weights_mut() {
        node.pos.x -= min_x;
        node.pos.y -= mean_y_left;
    }

    let run = max_x - min_x;
    let rise = mean_y_right - mean_y_left;

    (run, rise)
}

#[cfg(test)]
mod tests {
    use petgraph::graph::NodeIndex;

    use super::*;

    #[test]
    fn test_interpartition_alignment_and_width() {
        // Create an inter-partition space graph (no stitch nodes)
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::with_capacity(3, 0);

        // Add data nodes
        let _n1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos::new_xy(0, 10, 5),
            (4, 2),
            Some(0),
        ));
        let _n2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos::new_xy(0, 10, -5),
            (6, 2),
            Some(0),
        ));
        let _n3 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos::new_xy(0, 30, 0),
            (8, 3),
            Some(1),
        ));

        // Apply alignment for inter-partition space
        align_partition_to_origin(&mut graph);

        // After alignment, the leftmost nodes (at x=10) should be at x=0
        // n1 and n2 originally have center x=10, n3 has center x=30
        // After normalizing by subtracting min_x (10):
        // n1 and n2: x = 10 - 10 = 0
        // n3: x = 30 - 10 = 20

        let n1 = &graph[NodeIndex::new(0)];
        let n2 = &graph[NodeIndex::new(1)];
        let n3 = &graph[NodeIndex::new(2)];

        assert_eq!(n1.pos.x, 0, "Node 1 should be at x=0 after alignment");
        assert_eq!(n2.pos.x, 0, "Node 2 should be at x=0 after alignment");
        assert_eq!(n3.pos.x, 20, "Node 3 should be at x=20 after alignment");

        // Y coordinates should be unchanged
        assert_eq!(n1.pos.y, 5, "Node 1 y should be unchanged");
        assert_eq!(n2.pos.y, -5, "Node 2 y should be unchanged");
    }
}
