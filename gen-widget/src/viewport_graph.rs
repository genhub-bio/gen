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
pub struct ViewportGraph {
    /// Graph keyed by world coordinates, edges store domain node pairs
    pub graph: UnGraphMap<WorldPos, Vec<(NodeIndex, NodeIndex)>>,

    /// Node data at each world position
    pub nodes: HashMap<WorldPos, LayoutNode>,

    /// Map from domain NodeIndex to world position for quick lookups
    pub domain_to_world: HashMap<NodeIndex, WorldPos>,

    /// Layers of the graph - each layer contains domain node indices in that layer
    /// This provides a unified layer view across all partitions
    pub layers: Vec<Vec<NodeIndex>>,

    /// Track which nodes we've added from which partitions
    pub included_nodes: HashSet<(PartitionIndex, NodeIndex<u32>)>,
}

impl ViewportGraph {
    pub fn empty() -> Self {
        debug!(
            "[DEBUG_TRACE_LAYOUT] empty: -> result=ViewportGraph{{graph_nodes=0, nodes_count=0, domain_to_world_count=0, layers_count=0, included_nodes_count=0}}"
        );
        Self {
            graph: UnGraphMap::new(),
            nodes: HashMap::new(),
            domain_to_world: HashMap::new(),
            layers: Vec::new(),
            included_nodes: HashSet::new(),
        }
    }

    pub fn new<G, S>(
        viewport: BigRect<i64>,
        partition_controller: &mut crate::partition_controller::PartitionController<G, S>,
        detail_level: VisualDetail,
    ) -> Result<Self, String>
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
        S: crate::plotter::NodeSizer<G>,
    {
        debug!(
            "[DEBUG_TRACE_LAYOUT] new: viewport=BigRect{{min={{x={}, y={}}}, max={{x={}, y={}}}}}, detail_level={:?}",
            viewport.min.x, viewport.min.y, viewport.max.x, viewport.max.y, detail_level
        );

        // Find partitions that intersect with the viewport
        let visible_partitions = partition_controller
            .load_partitions_for_rect(viewport.resize(2.0))
            .unwrap();

        assert!(
            visible_partitions
                .iter()
                .all(|&idx| idx < partition_controller.partition_table.partitions.len()),
            "Invalid partition indices: {:?}, max allowed: {}",
            visible_partitions
                .iter()
                .filter(|&&idx| idx >= partition_controller.partition_table.partitions.len())
                .collect::<Vec<_>>(),
            partition_controller.partition_table.partitions.len()
        );
        debug!(
            "[DEBUG_TRACE_LAYOUT] new: coverage=ViewportCoverage{{covered_partitions_count={}, covered_partitions={:?}}}",
            visible_partitions.len(),
            visible_partitions
        );

        let mut viewport_graph = Self::empty();

        let vp = viewport_graph.build_from_partitions(
            &partition_controller.partition_table,
            &visible_partitions,
            viewport,
            detail_level,
        );

        if vp.is_err() {
            //TODO: handle error and make this a proper constructor
            panic!("Could not create ViewportGraph: {}", vp.unwrap_err());
        }

        debug!(
            "[DEBUG_TRACE_LAYOUT] new: -> result=ViewportGraph{{graph_nodes={}, nodes_count={}, domain_to_world_count={}, layers_count={}, included_nodes_count={}}}",
            viewport_graph.graph.node_count(),
            viewport_graph.nodes.len(),
            viewport_graph.domain_to_world.len(),
            viewport_graph.layers.len(),
            viewport_graph.included_nodes.len()
        );

        Ok(viewport_graph)
    }

    //TODO:make this the main constructor
    fn build_from_partitions<G>(
        &mut self,
        partition_table: &PartitionTable<G>,
        visible_partitions: &[usize],
        viewport: BigRect<i64>,
        detail_level: VisualDetail,
    ) -> Result<(), String>
    where
        G: petgraph::visit::GraphBase
            + Clone
            + petgraph::visit::EdgeIndexable
            + petgraph::visit::NodeIndexable
            + petgraph::visit::NodeCount
            + petgraph::visit::Visitable
            + petgraph::visit::IntoEdgeReferences
            + petgraph::visit::IntoNodeIdentifiers
            + petgraph::visit::IntoNeighborsDirected,
        G::NodeId: Copy + Eq + std::hash::Hash + Ord,
        G::EdgeId: Clone,
    {
        debug!(
            "[DEBUG_TRACE_LAYOUT] build_from_partitions: visible_partitions_count={}, visible_partitions={:?}, viewport=BigRect{{min={{x={}, y={}}}, max={{x={}, y={}}}}}, detail_level={:?}",
            visible_partitions.len(),
            visible_partitions,
            viewport.min.x,
            viewport.min.y,
            viewport.max.x,
            viewport.max.y,
            detail_level
        );

        // Process partitions in order (they should already be sorted by index)
        for &partition_idx in visible_partitions {
            debug!(
                "[DEBUG_TRACE_LAYOUT] build_from_partitions: processing partition_idx={}",
                partition_idx
            );
            if let Some(partition_layout) =
                &partition_table.partitions[partition_idx].layouts[detail_level.as_index()]
            {
                let clamped_viewport = partition_table
                    .clamp_rect_to_partition(viewport, partition_idx, detail_level)
                    .unwrap();

                debug!(
                    "[DEBUG_TRACE_LAYOUT] build_from_partitions: clamped_viewport=BigRect{{min={{x={}, y={}}}, max={{x={}, y={}}}}}",
                    clamped_viewport.min.x,
                    clamped_viewport.min.y,
                    clamped_viewport.max.x,
                    clamped_viewport.max.y
                );
                // assert!(
                //     clamped_viewport.max.x > clamped_viewport.min.x
                //         && clamped_viewport.max.y > clamped_viewport.min.y
                // );

                // Convert the corners of the clamped viewport to LocalPos coordinates
                let local_corner_min = partition_table
                    .world_to_local(clamped_viewport.min, detail_level)
                    .unwrap();
                let local_corner_max = partition_table
                    .world_to_local(clamped_viewport.max, detail_level)
                    .unwrap();

                debug!(
                    "[DEBUG_TRACE_LAYOUT] build_from_partitions: local_corner_min={{x={}, y={}}}, local_corner_max={{x={}, y={}}}",
                    local_corner_min.x, local_corner_min.y, local_corner_max.x, local_corner_max.y
                );

                let envelope = rstar::AABB::from_corners(
                    [local_corner_min.x, local_corner_min.y],
                    [local_corner_max.x, local_corner_max.y],
                );

                let visible_objects: Vec<_> = partition_layout
                    .spatial_index
                    .locate_in_envelope_intersecting(&envelope)
                    .cloned()
                    .collect();

                debug!(
                    "[DEBUG_TRACE_LAYOUT] build_from_partitions: visible_objects_count={}",
                    visible_objects.len()
                );

                // Debug: Print partition info and Fenwick tree state
                trace!("\n=== PARTITION {} DEBUG ===", partition_idx);
                trace!(
                    "Partition type: {}",
                    if partition_idx % 2 == 0 {
                        "Section"
                    } else {
                        "Bridge"
                    }
                );

                let measurements = partition_table.get_scale_data(detail_level);

                trace!("  Fenwick tree state:");
                for i in 0..=partition_idx {
                    // prefix_sum(i, 0) gives cumulative sum up to but NOT including i
                    // So to get cumulative INCLUDING i, we need prefix_sum(i+1, 0)
                    trace!(
                        "    Partition {} cumulative width up to (not including): {}",
                        i,
                        measurements.widths.prefix_sum(i, 0)
                    );
                    trace!(
                        "    Partition {} cumulative width including: {}",
                        i,
                        measurements.widths.prefix_sum(i + 1, 0)
                    );
                }
                trace!(
                    "    Anchor partition: {}",
                    measurements.anchor_partition_idx
                );
                trace!(
                    "Partition graph: {} nodes, {} edges",
                    measurements.layout_graph.node_count(),
                    measurements.layout_graph.edge_count(),
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

                            debug!(
                                "[DEBUG_TRACE_LAYOUT] merge_node: NODE_PROCESSING_PHASE adding node"
                            );
                            self.merge_node(world_pos, node_data.clone()).unwrap();
                            self.included_nodes.insert((partition_idx, node_idx));
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

                            debug!(
                                "[DEBUG_TRACE_LAYOUT] merge_node: EDGE_PROCESSING_PHASE adding source node"
                            );
                            self.merge_node(source_world, source_data.clone()).unwrap();
                            debug!(
                                "[DEBUG_TRACE_LAYOUT] merge_node: EDGE_PROCESSING_PHASE adding target node"
                            );
                            self.merge_node(target_world, target_data.clone()).unwrap();

                            // Only add edge if both endpoints are non-stitch nodes
                            // Stitch nodes are partition boundary artifacts that shouldn't appear in ViewportGraph
                            if !matches!(source_data.role, NodeRole::Stitch(_))
                                && !matches!(target_data.role, NodeRole::Stitch(_))
                            {
                                self.merge_edge_bundle(
                                    source_world,
                                    target_world,
                                    edge_data.bundle.clone(),
                                )
                                .unwrap();
                            }
                        }
                    }
                }
            }
        }

        // Build layers by grouping Data nodes by x-coordinate
        self.build_layers_from_coordinates();

        debug!(
            "[DEBUG_TRACE_LAYOUT] build_from_partitions: -> result=Ok(graph_nodes={}, nodes_count={}, domain_to_world_count={}, layers_count={}, included_nodes_count={})",
            self.graph.node_count(),
            self.nodes.len(),
            self.domain_to_world.len(),
            self.layers.len(),
            self.included_nodes.len()
        );

        let coordinates_in_use = self.nodes.keys().copied().collect::<Vec<_>>();
        debug!("VPGraph nodes: {:?}", coordinates_in_use);
        Ok(())
    }

    /// Build layers by grouping Data nodes by x-coordinate.
    /// All nodes with the same x-coordinate belong to the same layer.
    /// Nodes are sorted first by x, then by y within each layer.
    fn build_layers_from_coordinates(&mut self) {
        debug!(
            "[DEBUG_TRACE_LAYOUT] build_layers_from_coordinates: starting with {} nodes",
            self.nodes.len()
        );

        // Collect all Data nodes with their world positions
        let mut data_nodes: Vec<(WorldPos, NodeIndex)> = Vec::new();

        for (world_pos, layout_node) in &self.nodes {
            if let crate::layout::NodeRole::Data(domain_idx) = &layout_node.role {
                data_nodes.push((*world_pos, *domain_idx));
            }
        }

        debug!(
            "[DEBUG_TRACE_LAYOUT] build_layers_from_coordinates: found {} Data nodes",
            data_nodes.len()
        );

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
                        debug!(
                            "[DEBUG_TRACE_LAYOUT] build_layers_from_coordinates: created layer {} with {} nodes at x={}: {:?}",
                            layers.len(),
                            current_layer.len(),
                            current_x.unwrap(),
                            current_layer
                        );
                        layers.push(current_layer);
                    }
                    current_layer = vec![domain_idx];
                    current_x = Some(world_pos.x);
                }
            }
        }

        // Add the last layer
        if !current_layer.is_empty() {
            debug!(
                "[DEBUG_TRACE_LAYOUT] build_layers_from_coordinates: created final layer {} with {} nodes at x={}: {:?}",
                layers.len(),
                current_layer.len(),
                current_x.unwrap(),
                current_layer
            );
            layers.push(current_layer);
        }

        self.layers = layers;

        debug!(
            "[DEBUG_TRACE_LAYOUT] build_layers_from_coordinates: created {} layers",
            self.layers.len()
        );
    }

    fn merge_node(&mut self, world_pos: WorldPos, layout_node: LayoutNode) -> Result<(), String> {
        use crate::layout::NodeRole;

        debug!(
            "[DEBUG_TRACE_LAYOUT] merge_node: world_pos={{x={}, y={}}}, layout_node={{role={:?}, layer={:?}, size={{width={}, height={}}}, local_pos={{partition_idx={}, x={}, y={}}}}}",
            world_pos.x,
            world_pos.y,
            layout_node.role,
            layout_node.layer,
            layout_node.size.0,
            layout_node.size.1,
            layout_node.pos.partition_idx,
            layout_node.pos.x,
            layout_node.pos.y
        );

        // Skip stitch nodes entirely - they should not appear in the final ViewportGraph
        if matches!(layout_node.role, NodeRole::Stitch(_)) {
            debug!("[DEBUG_TRACE_LAYOUT] merge_node: skipping stitch node -> result=Ok(())");
            return Ok(());
        }

        if let Some(existing) = self.nodes.get(&world_pos).cloned() {
            debug!(
                "[DEBUG_TRACE_LAYOUT] merge_node: node collision detected, existing={{role={:?}, layer={:?}}}, resolving collision",
                existing.role, existing.layer
            );
            let result = self.resolve_node_collision(world_pos, &existing, &layout_node);
            debug!(
                "[DEBUG_TRACE_LAYOUT] merge_node: collision resolution -> result={:?}",
                result
            );
            result
        } else {
            // Update domain to world mapping if this is a data node
            if let NodeRole::Data(domain_idx) = &layout_node.role {
                // Check for domain node duplication
                if let Some(existing_world_pos) = self.domain_to_world.get(domain_idx) {
                    if *existing_world_pos != world_pos {
                        debug!(
                            "[DEBUG_TRACE_LAYOUT] merge_node: WARNING - DUPLICATE domain node! domain_idx={:?} already exists at {{x={}, y={}}} but trying to add at {{x={}, y={}}} from local_pos={{partition_idx={}, x={}, y={}}}",
                            domain_idx,
                            existing_world_pos.x,
                            existing_world_pos.y,
                            world_pos.x,
                            world_pos.y,
                            layout_node.pos.partition_idx,
                            layout_node.pos.x,
                            layout_node.pos.y
                        );
                    }
                }
                debug!(
                    "[DEBUG_TRACE_LAYOUT] merge_node: adding data node mapping domain_idx={:?} -> world_pos={{x={}, y={}}} from local_pos={{partition_idx={}, x={}, y={}}}",
                    domain_idx,
                    world_pos.x,
                    world_pos.y,
                    layout_node.pos.partition_idx,
                    layout_node.pos.x,
                    layout_node.pos.y
                );
                self.domain_to_world.insert(*domain_idx, world_pos);
            }
            self.nodes.insert(world_pos, layout_node);
            self.graph.add_node(world_pos);
            debug!("[DEBUG_TRACE_LAYOUT] merge_node: new node added -> result=Ok(())");
            Ok(())
        }
    }

    // TODO: we don't need this anymore (stitch nodes do not have to overlap anymore, remove
    fn resolve_node_collision(
        &mut self,
        world_pos: WorldPos,
        existing: &LayoutNode,
        new: &LayoutNode,
    ) -> Result<(), String> {
        debug!(
            "[DEBUG_TRACE_LAYOUT] resolve_node_collision: world_pos={{x={}, y={}}}, existing={{role={:?}, layer={:?}}}, new={{role={:?}, layer={:?}}}",
            world_pos.x, world_pos.y, existing.role, existing.layer, new.role, new.layer
        );

        let result = match (&existing.role, &new.role) {
            (NodeRole::Data(e), NodeRole::Data(n)) if e == n => {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] resolve_node_collision: identical data nodes, keeping existing"
                );
                Ok(())
            }

            (NodeRole::Data(e), NodeRole::Data(n)) => {
                let error = format!(
                    "Conflicting data nodes at {:?}: {:?} vs {:?}",
                    world_pos, e, n
                );
                debug!(
                    "[DEBUG_TRACE_LAYOUT] resolve_node_collision: data node conflict - {}",
                    error
                );
                Err(error)
            }

            (NodeRole::Stitch(_), NodeRole::Data(_)) | (NodeRole::Stitch(_), NodeRole::Routing) => {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] resolve_node_collision: replacing stitch with data/routing node"
                );
                // Update domain to world mapping if this is a data node
                if let NodeRole::Data(domain_idx) = &new.role {
                    debug!(
                        "[DEBUG_TRACE_LAYOUT] resolve_node_collision: updating domain mapping domain_idx={:?} -> world_pos={{x={}, y={}}}",
                        domain_idx, world_pos.x, world_pos.y
                    );
                    self.domain_to_world.insert(*domain_idx, world_pos);
                }
                self.nodes.insert(world_pos, new.clone());
                Ok(())
            }

            (NodeRole::Data(_), NodeRole::Stitch(_)) | (NodeRole::Routing, NodeRole::Stitch(_)) => {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] resolve_node_collision: keeping existing data/routing node over stitch"
                );
                Ok(())
            }

            (NodeRole::Routing, NodeRole::Routing) => {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] resolve_node_collision: identical routing nodes, keeping existing"
                );
                Ok(())
            }

            (NodeRole::Data(_), NodeRole::Routing) | (NodeRole::Routing, NodeRole::Data(_)) => {
                let error = format!("Data/Routing conflict at {:?}", world_pos);
                debug!(
                    "[DEBUG_TRACE_LAYOUT] resolve_node_collision: data/routing conflict - {}",
                    error
                );
                Err(error)
            }

            (NodeRole::Stitch(_), NodeRole::Stitch(_)) => {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] resolve_node_collision: identical stitch nodes, keeping existing"
                );
                Ok(())
            }
        };

        debug!(
            "[DEBUG_TRACE_LAYOUT] resolve_node_collision: -> result={:?}",
            result
        );
        result
    }

    fn merge_edge_bundle(
        &mut self,
        source: WorldPos,
        target: WorldPos,
        bundle: Vec<(NodeIndex, NodeIndex)>,
    ) -> Result<(), String> {
        debug!(
            "[DEBUG_TRACE_LAYOUT] merge_edge_bundle: source={{x={}, y={}}}, target={{x={}, y={}}}, bundle_count={}, bundle={:?}",
            source.x,
            source.y,
            target.x,
            target.y,
            bundle.len(),
            bundle
        );

        match self.graph.edge_weight_mut(source, target) {
            Some(existing_bundle) => {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] merge_edge_bundle: existing edge found, existing_bundle_count={}",
                    existing_bundle.len()
                );
                let mut added_count = 0;
                for pair in bundle {
                    if !existing_bundle.contains(&pair) {
                        existing_bundle.push(pair);
                        added_count += 1;
                    }
                }
                debug!(
                    "[DEBUG_TRACE_LAYOUT] merge_edge_bundle: merged edge bundle, added={}, new_total={}",
                    added_count,
                    existing_bundle.len()
                );
            }
            None => {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] merge_edge_bundle: creating new edge with bundle_count={}",
                    bundle.len()
                );
                self.graph.add_edge(source, target, bundle);
            }
        }
        debug!("[DEBUG_TRACE_LAYOUT] merge_edge_bundle: -> result=Ok(())");
        Ok(())
    }

    /// Get all nodes in the viewport graph
    pub fn nodes(&self) -> impl Iterator<Item = (&WorldPos, &LayoutNode)> {
        self.nodes.iter()
    }

    /// Get all edges in the viewport graph
    pub fn edges(
        &self,
    ) -> impl Iterator<Item = (WorldPos, WorldPos, &Vec<(NodeIndex, NodeIndex)>)> + '_ {
        self.graph.all_edges()
    }

    /// Get node at a specific world position
    pub fn get_node(&self, pos: &WorldPos) -> Option<&LayoutNode> {
        let result = self.nodes.get(pos);
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
        self.nodes.contains_key(pos)
    }

    /// Get the number of nodes in the viewport graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
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

    /// Find which layer a node belongs to (by world position)
    pub fn find_node_layer(&self, pos: &WorldPos) -> Option<usize> {
        debug!(
            "[DEBUG_TRACE_LAYOUT] find_node_layer: pos={{x={}, y={}}}",
            pos.x, pos.y
        );

        // First, find the domain node at this position
        if let Some(node) = self.nodes.get(pos) {
            debug!(
                "[DEBUG_TRACE_LAYOUT] find_node_layer: found node={{role={:?}, layer={:?}}}",
                node.role, node.layer
            );
            if let NodeRole::Data(domain_idx) = node.role {
                // Then find which layer contains this domain node
                let result = self
                    .layers
                    .iter()
                    .position(|layer| layer.contains(&domain_idx));
                debug!(
                    "[DEBUG_TRACE_LAYOUT] find_node_layer: domain_idx={:?} -> result={:?}",
                    domain_idx, result
                );
                return result;
            } else {
                debug!(
                    "[DEBUG_TRACE_LAYOUT] find_node_layer: node is not a Data node -> result=None"
                );
            }
        } else {
            debug!(
                "[DEBUG_TRACE_LAYOUT] find_node_layer: no node found at position -> result=None"
            );
        }
        None
    }

    /// Find which layer a domain node belongs to
    pub fn find_domain_node_layer(&self, domain_idx: NodeIndex) -> Option<usize> {
        debug!(
            "[DEBUG_TRACE_LAYOUT] find_domain_node_layer: domain_idx={:?}",
            domain_idx
        );
        let result = self
            .layers
            .iter()
            .position(|layer| layer.contains(&domain_idx));
        debug!(
            "[DEBUG_TRACE_LAYOUT] find_domain_node_layer: -> result={:?}",
            result
        );
        result
    }

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
        debug!(
            "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: domain_node={:?}",
            domain_node
        );
        let mut results = Vec::new();

        // Find the world position for this domain node
        let Some(&from_pos) = self.domain_to_world.get(&domain_node) else {
            debug!(
                "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: domain node not in viewport -> result=Vec::new()"
            );
            return results; // Domain node not in viewport
        };

        debug!(
            "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: from_pos={{x={}, y={}}}",
            from_pos.x, from_pos.y
        );

        // Verify starting node exists
        if !self.nodes.contains_key(&from_pos) {
            debug!(
                "[DEBUG_TRACE_LAYOUT] find_adjacent_data_nodes: starting node does not exist -> result=Vec::new()"
            );
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

            if let Some(node) = self.nodes.get(&current_pos) {
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

    /// Verify that partitions have been joined correctly in the viewport graph.
    /// This function performs comprehensive validation of partition merging integrity.
    pub fn verify_partition_joining(&self) -> PartitionJoinVerification {
        debug!(
            "[DEBUG_TRACE_LAYOUT] verify_partition_joining: starting verification with {} nodes, {} edges, {} layers",
            self.nodes.len(),
            self.graph.edge_count(),
            self.layers.len()
        );

        let mut verification = PartitionJoinVerification::new();

        // 1. Verify graph contiguity - ensure no isolated components
        verification.contiguity_result = self.verify_graph_contiguity();

        // 2. Verify domain node consistency - check for duplicates or missing mappings
        verification.domain_consistency = self.verify_domain_node_consistency();

        // 3. Verify layer integrity - ensure layers are properly unified across partitions
        verification.layer_integrity = self.verify_layer_integrity();

        // 4. Verify edge bundle coherence - check that edge bundles are properly merged
        verification.edge_coherence = self.verify_edge_bundle_coherence();

        // 5. Verify partition boundary integrity
        verification.boundary_integrity = self.verify_partition_boundaries();

        verification.overall_valid = verification.all_checks_passed();

        debug!(
            "[DEBUG_TRACE_LAYOUT] verify_partition_joining: -> result=PartitionJoinVerification{{overall_valid={}, issues_count={}}}",
            verification.overall_valid,
            verification.total_issues()
        );

        verification
    }

    /// Check if the graph is contiguous (no isolated components that should be connected)
    fn verify_graph_contiguity(&self) -> ContiguityResult {
        use std::collections::{HashSet, VecDeque};

        if self.graph.node_count() == 0 {
            return ContiguityResult::valid("Empty graph is trivially contiguous");
        }

        // Find all connected components using BFS
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for start_node in self.graph.nodes() {
            if !visited.contains(&start_node) {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(start_node);
                visited.insert(start_node);

                while let Some(current) = queue.pop_front() {
                    component.push(current);

                    for neighbor in self.graph.neighbors(current) {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
                components.push(component);
            }
        }

        if components.len() == 1 {
            ContiguityResult::valid("Graph is fully connected")
        } else {
            let mut isolated_components = Vec::new();
            for component in components.iter() {
                if component.len() == 1 {
                    isolated_components.push(component[0]);
                }
            }

            ContiguityResult::invalid(
                format!("Graph has {} connected components", components.len()),
                isolated_components,
                components,
            )
        }
    }

    /// Verify that domain nodes have consistent mappings and no duplicates
    fn verify_domain_node_consistency(&self) -> DomainConsistencyResult {
        let mut issues = Vec::new();
        let mut position_to_domains: HashMap<WorldPos, Vec<NodeIndex>> = HashMap::new();

        // Group domain nodes by their world positions
        for (&domain_idx, &world_pos) in &self.domain_to_world {
            position_to_domains
                .entry(world_pos)
                .or_default()
                .push(domain_idx);
        }

        // Check for multiple domain nodes at the same position (potential duplication issue)
        for (world_pos, domain_nodes) in &position_to_domains {
            if domain_nodes.len() > 1 {
                issues.push(format!(
                    "Multiple domain nodes at position {:?}: {:?}",
                    world_pos, domain_nodes
                ));
            }
        }

        // Check for domain nodes in domain_to_world that don't have corresponding entries in nodes
        for (&domain_idx, &world_pos) in &self.domain_to_world {
            if !self.nodes.contains_key(&world_pos) {
                issues.push(format!(
                    "Domain node {:?} maps to position {:?} but no node exists there",
                    domain_idx, world_pos
                ));
            }
        }

        // Check for data nodes in the graph that aren't in domain_to_world
        for (world_pos, layout_node) in &self.nodes {
            if let crate::layout::NodeRole::Data(domain_idx) = layout_node.role {
                if let Some(&mapped_pos) = self.domain_to_world.get(&domain_idx) {
                    if mapped_pos != *world_pos {
                        issues.push(format!(
                            "Domain node {:?} position mismatch: mapped to {:?} but found at {:?}",
                            domain_idx, mapped_pos, world_pos
                        ));
                    }
                } else {
                    issues.push(format!(
                        "Data node {:?} at position {:?} not in domain_to_world mapping",
                        domain_idx, world_pos
                    ));
                }
            }
        }

        DomainConsistencyResult {
            valid: issues.is_empty(),
            issues,
            total_domain_nodes: self.domain_to_world.len(),
            total_positions: position_to_domains.len(),
        }
    }

    /// Verify that layers are properly unified across partitions
    fn verify_layer_integrity(&self) -> LayerIntegrityResult {
        let mut issues = Vec::new();
        let mut nodes_in_layers = HashSet::new();

        // Check that all nodes in layers exist in domain_to_world
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for &domain_node in layer {
                nodes_in_layers.insert(domain_node);

                if !self.domain_to_world.contains_key(&domain_node) {
                    issues.push(format!(
                        "Layer {} contains domain node {:?} that doesn't exist in domain_to_world",
                        layer_idx, domain_node
                    ));
                }
            }
        }

        // Check that all domain nodes are represented in layers
        for &domain_node in self.domain_to_world.keys() {
            if !nodes_in_layers.contains(&domain_node) {
                issues.push(format!(
                    "Domain node {:?} exists in domain_to_world but not in any layer",
                    domain_node
                ));
            }
        }

        // Check for duplicate nodes across layers
        let total_nodes_in_layers = nodes_in_layers.len();
        let sum_layer_sizes: usize = self.layers.iter().map(|layer| layer.len()).sum();

        if sum_layer_sizes != total_nodes_in_layers {
            issues.push(format!(
                "Node duplication detected: {} total unique nodes but {} layer entries",
                total_nodes_in_layers, sum_layer_sizes
            ));
        }

        LayerIntegrityResult {
            valid: issues.is_empty(),
            issues,
            layer_count: self.layers.len(),
            total_nodes_in_layers,
        }
    }

    /// Verify that edge bundles are coherent and represent valid domain paths
    fn verify_edge_bundle_coherence(&self) -> EdgeCoherenceResult {
        let mut issues = Vec::new();
        let mut total_bundle_size = 0;

        // First, basic structural checks
        for (source_pos, target_pos, bundle) in self.graph.all_edges() {
            total_bundle_size += bundle.len();

            // Check that edge endpoints exist as nodes
            if !self.nodes.contains_key(&source_pos) {
                issues.push(format!(
                    "Edge source position {:?} has no corresponding node",
                    source_pos
                ));
            }
            if !self.nodes.contains_key(&target_pos) {
                issues.push(format!(
                    "Edge target position {:?} has no corresponding node",
                    target_pos
                ));
            }

            // Check for empty bundles (should not happen)
            if bundle.is_empty() {
                issues.push(format!(
                    "Edge from {:?} to {:?} has empty bundle",
                    source_pos, target_pos
                ));
            }
        }

        // Build domain connectivity graph to verify path reachability
        let domain_reachability = self.build_domain_reachability_graph();

        // Verify that each bundle represents valid domain-level connections
        for (source_pos, target_pos, bundle) in self.graph.all_edges() {
            for &(source_domain, target_domain) in bundle {
                // Check that both domain nodes exist (somewhere in the graph)
                if !self.domain_to_world.contains_key(&source_domain) {
                    issues.push(format!(
                        "Bundle contains unknown source domain node {:?} in edge {:?} -> {:?}",
                        source_domain, source_pos, target_pos
                    ));
                    continue;
                }
                if !self.domain_to_world.contains_key(&target_domain) {
                    issues.push(format!(
                        "Bundle contains unknown target domain node {:?} in edge {:?} -> {:?}",
                        target_domain, source_pos, target_pos
                    ));
                    continue;
                }

                // Verify that this domain connection has a valid path in the connectivity graph
                if !domain_reachability.has_path(source_domain, target_domain) {
                    issues.push(format!(
                        "Bundle claims connection {:?} -> {:?}, but no valid path exists in domain connectivity",
                        source_domain, target_domain
                    ));
                }
            }
        }

        // Verify path coverage: ensure all domain connections are properly bundled
        let bundle_coverage_issues = self.verify_bundle_coverage(&domain_reachability);
        issues.extend(bundle_coverage_issues);

        EdgeCoherenceResult {
            valid: issues.is_empty(),
            issues,
            edge_count: self.graph.edge_count(),
            total_bundle_size,
        }
    }

    /// Build a domain-level reachability graph from the edge bundles
    fn build_domain_reachability_graph(&self) -> DomainReachabilityGraph {
        let mut reachability = DomainReachabilityGraph::new();

        // Add all domain nodes
        for &domain_node in self.domain_to_world.keys() {
            reachability.add_node(domain_node);
        }

        // Add direct connections from bundles
        for (_source_pos, _target_pos, bundle) in self.graph.all_edges() {
            for &(source_domain, target_domain) in bundle {
                reachability.add_edge(source_domain, target_domain);
            }
        }

        reachability
    }

    /// Verify that all expected domain connections are properly represented in bundles
    fn verify_bundle_coverage(&self, domain_graph: &DomainReachabilityGraph) -> Vec<String> {
        let mut issues = Vec::new();

        // Create inverse mapping: domain edge -> visual segments
        let domain_to_visual = self.invert_edge_bundles();

        // Check that each domain connection has at least one visual representation
        for &(source_domain, target_domain) in domain_graph.edges() {
            if !domain_to_visual.contains_key(&(source_domain, target_domain)) {
                issues.push(format!(
                    "Domain connection {:?} -> {:?} exists in reachability but has no visual representation",
                    source_domain, target_domain
                ));
            }
        }

        // Conversely, check that bundled connections exist in domain graph
        for &(source_domain, target_domain) in domain_to_visual.keys() {
            if !domain_graph.has_edge(source_domain, target_domain) {
                issues.push(format!(
                    "Visual bundle claims domain connection {:?} -> {:?} but it's not in domain reachability",
                    source_domain, target_domain
                ));
            }
        }

        issues
    }

    /// Verify partition boundary integrity
    fn verify_partition_boundaries(&self) -> BoundaryIntegrityResult {
        let mut issues = Vec::new();

        // Group nodes by their source partitions
        let mut partition_groups: HashMap<PartitionIndex, Vec<WorldPos>> = HashMap::new();

        for &(partition_idx, _) in &self.included_nodes {
            partition_groups.entry(partition_idx).or_default();
        }

        // Check for potential boundary issues by examining coordinate patterns
        let mut boundary_coords = Vec::new();
        for positions in partition_groups.values() {
            if !positions.is_empty() {
                let x_coords: Vec<i64> = self
                    .nodes
                    .keys()
                    .filter(|pos| positions.contains(pos))
                    .map(|pos| pos.x)
                    .collect();
                if let (Some(&min_x), Some(&max_x)) = (x_coords.iter().min(), x_coords.iter().max())
                {
                    boundary_coords.push((min_x, max_x));
                }
            }
        }

        // Look for overlapping boundaries (which is expected and good)
        let mut overlap_count = 0;
        for i in 0..boundary_coords.len() {
            for j in (i + 1)..boundary_coords.len() {
                let (min1, max1) = boundary_coords[i];
                let (min2, max2) = boundary_coords[j];

                // Check for overlap
                if min1 <= max2 && min2 <= max1 {
                    overlap_count += 1;
                }
            }
        }

        // If we have adjacent partitions but no overlaps, that might be an issue
        if partition_groups.len() > 1 && overlap_count == 0 {
            issues.push(
                "No coordinate overlaps detected between partitions - potential gap in coverage"
                    .to_string(),
            );
        }

        BoundaryIntegrityResult {
            valid: issues.is_empty(),
            issues,
            partition_count: partition_groups.len(),
            boundary_overlaps: overlap_count,
        }
    }
}

/// Comprehensive result of partition joining verification
#[derive(Debug, Clone)]
pub struct PartitionJoinVerification {
    pub overall_valid: bool,
    pub contiguity_result: ContiguityResult,
    pub domain_consistency: DomainConsistencyResult,
    pub layer_integrity: LayerIntegrityResult,
    pub edge_coherence: EdgeCoherenceResult,
    pub coordinate_alignment: CoordinateAlignmentResult,
    pub boundary_integrity: BoundaryIntegrityResult,
}

impl PartitionJoinVerification {
    fn new() -> Self {
        Self {
            overall_valid: false,
            contiguity_result: ContiguityResult::valid("Not yet checked"),
            domain_consistency: DomainConsistencyResult::valid(),
            layer_integrity: LayerIntegrityResult::valid(),
            edge_coherence: EdgeCoherenceResult::valid(),
            coordinate_alignment: CoordinateAlignmentResult::valid(),
            boundary_integrity: BoundaryIntegrityResult::valid(),
        }
    }

    fn all_checks_passed(&self) -> bool {
        self.contiguity_result.valid
            && self.domain_consistency.valid
            && self.layer_integrity.valid
            && self.edge_coherence.valid
            && self.coordinate_alignment.valid
            && self.boundary_integrity.valid
    }

    fn total_issues(&self) -> usize {
        let mut count = 0;
        if !self.contiguity_result.valid {
            count += 1;
        }
        count += self.domain_consistency.issues.len();
        count += self.layer_integrity.issues.len();
        count += self.edge_coherence.issues.len();
        count += self.coordinate_alignment.issues.len();
        count += self.boundary_integrity.issues.len();
        count
    }

    /// Generate a human-readable report of all verification results
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Partition Join Verification Report ===\n");
        report.push_str(&format!(
            "Overall Status: {}\n\n",
            if self.overall_valid {
                "✓ VALID"
            } else {
                "✗ ISSUES DETECTED"
            }
        ));

        // Contiguity
        report.push_str(&format!(
            "Graph Contiguity: {}\n",
            if self.contiguity_result.valid {
                "✓"
            } else {
                "✗"
            }
        ));
        if !self.contiguity_result.valid {
            report.push_str(&format!("  Issue: {}\n", self.contiguity_result.message));
        }

        // Domain consistency
        report.push_str(&format!(
            "Domain Consistency: {} ({} issues)\n",
            if self.domain_consistency.valid {
                "✓"
            } else {
                "✗"
            },
            self.domain_consistency.issues.len()
        ));
        for issue in &self.domain_consistency.issues {
            report.push_str(&format!("  - {}\n", issue));
        }

        // Layer integrity
        report.push_str(&format!(
            "Layer Integrity: {} ({} layers, {} issues)\n",
            if self.layer_integrity.valid {
                "✓"
            } else {
                "✗"
            },
            self.layer_integrity.layer_count,
            self.layer_integrity.issues.len()
        ));
        for issue in &self.layer_integrity.issues {
            report.push_str(&format!("  - {}\n", issue));
        }

        // Edge coherence
        report.push_str(&format!(
            "Edge Coherence: {} ({} edges, {} issues)\n",
            if self.edge_coherence.valid {
                "✓"
            } else {
                "✗"
            },
            self.edge_coherence.edge_count,
            self.edge_coherence.issues.len()
        ));
        for issue in &self.edge_coherence.issues {
            report.push_str(&format!("  - {}\n", issue));
        }

        // Coordinate alignment
        report.push_str(&format!(
            "Coordinate Alignment: {} (range: {:?}, {} potential viewer issues)\n",
            if self.coordinate_alignment.valid {
                "✓"
            } else {
                "✗"
            },
            self.coordinate_alignment.coordinate_range,
            self.coordinate_alignment.potential_viewer_differences.len()
        ));
        for issue in &self.coordinate_alignment.issues {
            report.push_str(&format!("  - {}\n", issue));
        }
        for diff in &self.coordinate_alignment.potential_viewer_differences {
            report.push_str(&format!("  ⚠ {}\n", diff));
        }

        // Boundary integrity
        report.push_str(&format!(
            "Boundary Integrity: {} ({} partitions, {} overlaps)\n",
            if self.boundary_integrity.valid {
                "✓"
            } else {
                "✗"
            },
            self.boundary_integrity.partition_count,
            self.boundary_integrity.boundary_overlaps
        ));
        for issue in &self.boundary_integrity.issues {
            report.push_str(&format!("  - {}\n", issue));
        }

        report
    }
}

// Individual verification result types
#[derive(Debug, Clone)]
pub struct ContiguityResult {
    pub valid: bool,
    pub message: String,
    pub isolated_nodes: Vec<WorldPos>,
    pub components: Vec<Vec<WorldPos>>,
}

impl ContiguityResult {
    fn valid(message: &str) -> Self {
        Self {
            valid: true,
            message: message.to_string(),
            isolated_nodes: Vec::new(),
            components: Vec::new(),
        }
    }

    fn invalid(
        message: String,
        isolated_nodes: Vec<WorldPos>,
        components: Vec<Vec<WorldPos>>,
    ) -> Self {
        Self {
            valid: false,
            message,
            isolated_nodes,
            components,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DomainConsistencyResult {
    pub valid: bool,
    pub issues: Vec<String>,
    pub total_domain_nodes: usize,
    pub total_positions: usize,
}

impl DomainConsistencyResult {
    fn valid() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            total_domain_nodes: 0,
            total_positions: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerIntegrityResult {
    pub valid: bool,
    pub issues: Vec<String>,
    pub layer_count: usize,
    pub total_nodes_in_layers: usize,
}

impl LayerIntegrityResult {
    fn valid() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            layer_count: 0,
            total_nodes_in_layers: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdgeCoherenceResult {
    pub valid: bool,
    pub issues: Vec<String>,
    pub edge_count: usize,
    pub total_bundle_size: usize,
}

impl EdgeCoherenceResult {
    fn valid() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            edge_count: 0,
            total_bundle_size: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoordinateAlignmentResult {
    pub valid: bool,
    pub issues: Vec<String>,
    pub potential_viewer_differences: Vec<String>,
    pub coordinate_range: (i64, i64), // (x_range, y_range)
    pub node_count: usize,
}

impl CoordinateAlignmentResult {
    fn valid() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            potential_viewer_differences: Vec::new(),
            coordinate_range: (0, 0),
            node_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoundaryIntegrityResult {
    pub valid: bool,
    pub issues: Vec<String>,
    pub partition_count: usize,
    pub boundary_overlaps: usize,
}

impl BoundaryIntegrityResult {
    fn valid() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            partition_count: 0,
            boundary_overlaps: 0,
        }
    }
}

/// Helper structure for domain-level reachability analysis
struct DomainReachabilityGraph {
    nodes: HashSet<NodeIndex>,
    edges: HashSet<(NodeIndex, NodeIndex)>,
    adjacency: HashMap<NodeIndex, HashSet<NodeIndex>>,
}

impl DomainReachabilityGraph {
    fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashSet::new(),
            adjacency: HashMap::new(),
        }
    }

    fn add_node(&mut self, node: NodeIndex) {
        self.nodes.insert(node);
        self.adjacency.entry(node).or_default();
    }

    fn add_edge(&mut self, source: NodeIndex, target: NodeIndex) {
        self.edges.insert((source, target));
        self.adjacency.entry(source).or_default().insert(target);
        self.adjacency.entry(target).or_default(); // Ensure target exists
    }

    fn has_edge(&self, source: NodeIndex, target: NodeIndex) -> bool {
        self.edges.contains(&(source, target))
    }

    fn has_path(&self, source: NodeIndex, target: NodeIndex) -> bool {
        if source == target {
            return true;
        }

        use std::collections::VecDeque;
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(source);
        visited.insert(source);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&current) {
                for &neighbor in neighbors {
                    if neighbor == target {
                        return true;
                    }
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        false
    }

    fn edges(&self) -> impl Iterator<Item = &(NodeIndex, NodeIndex)> {
        self.edges.iter()
    }
}

impl Default for ViewportGraph {
    fn default() -> Self {
        Self::empty()
    }
}
