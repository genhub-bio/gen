use std::collections::{HashMap, HashSet};

use log::debug;
use petgraph::{graphmap::UnGraphMap, prelude::NodeIndex};

use crate::{
    geometry::WorldPos,
    layout::{LayoutNode, NodeRole},
};

/// A graph containing only the nodes and edges visible in the current viewport.
/// Uses world coordinates as keys for natural deduplication.
#[derive(Clone)]
pub struct ViewportGraph {
    /// Graph keyed by world coordinates, edges store domain node pairs
    pub graph: UnGraphMap<WorldPos, Vec<(NodeIndex, NodeIndex)>>,

    /// Node data at each world position (because GraphMaps don't store node labels)
    pub node_data_by_pos: HashMap<WorldPos, LayoutNode>,

    /// Map from domain NodeIndex to world position for quick lookups
    pub node_positions: HashMap<NodeIndex, WorldPos>,

    /// Layers of the graph - each layer contains domain node indices in that layer
    pub layers: Vec<Vec<NodeIndex>>,

    /// Domain edges rewired onto pin nodes (see `crawl::build_window_graph`), copied through
    /// so the plotter can mark the rendered bypass edge with direction arrows.
    pub backward_edges: Vec<(NodeIndex, NodeIndex)>,

    /// World position pairs whose edge is a backward edge's `left_pin -> right_pin` main
    /// span (`LayoutEdge::is_backward_span`), keyed by both `(source, target)` orderings so a
    /// lookup doesn't need to know which ordering `graph.all_edges()` happened to store.
    /// Used by `plotter::draw_arrows`, which only marks direction on the main span - the
    /// loop's two short excursions to/from its real endpoints are left unmarked.
    pub backward_span_edges: HashSet<(WorldPos, WorldPos)>,
}

impl ViewportGraph {
    pub fn empty() -> Self {
        debug!(
            "[DEBUG_TRACE_LAYOUT] empty: -> result=ViewportGraph{{graph_nodes=0, nodes_count=0, domain_to_world_count=0, layers_count=0}}"
        );
        Self {
            graph: UnGraphMap::new(),
            node_data_by_pos: HashMap::new(),
            node_positions: HashMap::new(),
            layers: Vec::new(),
            backward_edges: Vec::new(),
            backward_span_edges: HashSet::new(),
        }
    }

    /// Build a `ViewportGraph` from an already-assembled window geometry.
    ///
    /// The window is a single coordinate space produced by
    /// [`crate::assembly::assemble_window`] + [`crate::layout::WindowGeometry::new`], so
    /// layout positions *are* world positions (identity mapping): there is no per-window
    /// origin to add and no spatial query to run. Every node and edge in the geometry is inside
    /// the window by construction, so this iterates the whole layout graph.
    pub fn from_window_geometry(
        geometry: &crate::layout::WindowGeometry,
        backward_edges: &[(NodeIndex, NodeIndex)],
    ) -> Self {
        let mut this = Self::empty();

        for node_idx in geometry.graph.node_indices() {
            if let Some(node_data) = geometry.graph.node_weight(node_idx) {
                let world_pos = WorldPos::new(node_data.pos.x, node_data.pos.y);
                this.merge_node(world_pos, node_data.clone()).unwrap();
            }
        }

        for edge_idx in geometry.graph.edge_indices() {
            let Some((source_idx, target_idx)) = geometry.graph.edge_endpoints(edge_idx) else {
                continue;
            };
            let (Some(source_data), Some(target_data)) = (
                geometry.graph.node_weight(source_idx),
                geometry.graph.node_weight(target_idx),
            ) else {
                continue;
            };

            let source_world = WorldPos::new(source_data.pos.x, source_data.pos.y);
            let target_world = WorldPos::new(target_data.pos.x, target_data.pos.y);

            if let Some(edge_data) = geometry.graph.edge_weight(edge_idx) {
                this.merge_edge(source_world, target_world, edge_data.bundle.clone());
                if edge_data.is_backward_span {
                    this.backward_span_edges
                        .insert((source_world, target_world));
                    this.backward_span_edges
                        .insert((target_world, source_world));
                }
            }
        }

        this.build_layers_from_coordinates();
        this.backward_edges = backward_edges.to_vec();

        this
    }

    /// Build layers by grouping Data nodes by their Sugiyama rank.
    ///
    /// Nodes in the same layer belong to the same logical rank and should navigate together
    /// vertically — even when redistribution has given them different visual x-coordinates.
    /// Groups are ordered left-to-right by their minimum world x, giving a globally
    /// consistent layer index.
    fn build_layers_from_coordinates(&mut self) {
        // sugiyama_layer -> Vec<(world_y, domain_idx)>
        let mut groups: HashMap<i32, Vec<(i64, NodeIndex)>> = HashMap::new();
        // Minimum world x seen for each group — used for left-to-right sort order.
        let mut group_rep_x: HashMap<i32, i64> = HashMap::new();

        for (world_pos, layout_node) in &self.node_data_by_pos {
            if let crate::layout::NodeRole::Data(domain_idx) = &layout_node.role
                && let Some(layer) = layout_node.layer
            {
                let key = layer;
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
    /// Filters out Routing nodes, returning only nodes that represent original graph data.
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

    /// Get neighbors of a node at a world position
    pub fn neighbors(&self, pos: WorldPos) -> impl Iterator<Item = WorldPos> + '_ {
        self.graph.neighbors(pos)
    }

    /// Get nodes at a specific layer
    #[cfg(test)]
    pub(crate) fn get_layer(&self, layer: usize) -> Option<&Vec<NodeIndex>> {
        self.layers.get(layer)
    }

    /// Get the number of layers in the graph
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Create a new ViewportGraph from a list of visual (rectilinear) edges.
    /// This is useful for creating temporary graphs for highlighting.
    pub fn from_visual_edges(edges: &[((WorldPos, WorldPos), crate::plotter::PathStyle)]) -> Self {
        let mut new_graph = Self::empty();
        for ((source, target), _) in edges {
            new_graph.graph.add_edge(*source, *target, Vec::new());
        }
        new_graph
    }
}

impl Default for ViewportGraph {
    fn default() -> Self {
        Self::empty()
    }
}
