use std::collections::{HashMap, HashSet, hash_map};

use petgraph::stable_graph::StableGraph;

use super::{EdgeData, LayoutError, NodeData};

#[derive(Clone, Debug)]
pub struct TempGraph {
    node_indices: HashSet<u64>,
    nodes_by_index: HashMap<u64, NodeData>,
    edge_indices: HashSet<u64>,
    edges_by_index: HashMap<u64, EdgeData>,
    adjacencies_by_index: HashMap<u64, (u64, u64)>,
    neighbors: HashMap<u64, HashSet<u64>>,
    edge_ids_by_node_pair: HashMap<(u64, u64), u64>,
}

impl Default for TempGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TempGraph {
    pub fn new() -> Self {
        Self {
            node_indices: HashSet::new(),
            nodes_by_index: HashMap::new(),
            edge_indices: HashSet::new(),
            edges_by_index: HashMap::new(),
            adjacencies_by_index: HashMap::new(),
            neighbors: HashMap::new(),
            edge_ids_by_node_pair: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node_id: u64, node: NodeData) {
        self.node_indices.insert(node_id);
        self.nodes_by_index.insert(node_id, node);
    }

    pub fn update_node(&mut self, node_id: u64, node: NodeData) -> Result<(), LayoutError> {
        if !(self.node_indices.contains(&node_id)) {
            return Err(LayoutError::NodeNotFound(node_id));
        }
        self.nodes_by_index.insert(node_id, node);

        Ok(())
    }

    pub fn add_edge(
        &mut self,
        edge_id: u64,
        source: u64,
        target: u64,
        edge: EdgeData,
    ) -> Result<(), LayoutError> {
        if !(self.node_indices.contains(&source)) {
            return Err(LayoutError::NodeNotFound(source));
        }
        if !(self.node_indices.contains(&target)) {
            return Err(LayoutError::NodeNotFound(target));
        }
        self.edge_indices.insert(edge_id);
        self.edges_by_index.insert(edge_id, edge.clone());
        self.neighbors.entry(source).or_default().insert(target);
        self.adjacencies_by_index.insert(edge_id, (source, target));
        self.edge_ids_by_node_pair.insert((source, target), edge_id);

        Ok(())
    }

    pub fn update_edge(&mut self, edge_id: u64, edge: EdgeData) -> Result<(), LayoutError> {
        if !(self.edge_indices.contains(&edge_id)) {
            return Err(LayoutError::EdgeNotFound(edge_id));
        }
        self.edges_by_index.insert(edge_id, edge);

        Ok(())
    }

    pub fn node_indices(&self) -> hash_map::IntoKeys<u64, NodeData> {
        self.nodes_by_index.clone().into_keys()
    }

    pub fn nodes(&self) -> hash_map::IntoValues<u64, NodeData> {
        self.nodes_by_index.clone().into_values()
    }

    pub fn get_node(&self, node_id: u64) -> Option<NodeData> {
        self.nodes_by_index.get(&node_id).cloned()
    }

    pub fn get_edge(&self, edge_id: u64) -> Option<EdgeData> {
        self.edges_by_index.get(&edge_id).cloned()
    }

    pub fn contains_edge(&self, source: u64, target: u64) -> bool {
        if let Some(neighbors) = self.neighbors.get(&source) {
            neighbors.contains(&target)
        } else {
            false
        }
    }

    pub fn find_edge(&self, source: u64, target: u64) -> Option<u64> {
        self.edge_ids_by_node_pair.get(&(source, target)).cloned()
    }

    pub fn node_count(&self) -> usize {
        self.node_indices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edge_indices.len()
    }

    pub fn neighbors(&self, node_id: u64) -> HashSet<u64> {
        if self.neighbors.contains_key(&node_id) {
            self.neighbors.get(&node_id).unwrap().clone()
        } else {
            HashSet::new()
        }
    }

    pub fn edge_indices(&self) -> hash_map::IntoKeys<u64, EdgeData> {
        self.edges_by_index.clone().into_keys()
    }

    pub fn to_stable_graph(&self) -> StableGraph<NodeData, EdgeData> {
        let mut graph = StableGraph::new();
        let mut node_id_map = HashMap::new();
        for node in self.nodes() {
            let new_node_id = graph.add_node(node.clone());
            node_id_map.insert(node.node_id, new_node_id);
        }

        for node in self.nodes() {
            for neighbor_id in self.neighbors(node.node_id) {
                let node_id = node_id_map.get(&node.node_id).unwrap();
                let new_neighbor_id = node_id_map.get(&neighbor_id).unwrap();
                let edge_id = self.find_edge(node.node_id, neighbor_id).unwrap();
                let edge = self.get_edge(edge_id).unwrap();
                graph.add_edge(*node_id, *new_neighbor_id, edge);
            }
        }

        graph
    }

    pub fn edge_adjacencies(&self, edge_id: u64) -> Option<(u64, u64)> {
        self.adjacencies_by_index.get(&edge_id).copied()
    }

    pub fn remove_node(&mut self, node_id: u64) {
        self.nodes_by_index.remove(&node_id);
        for remaining_node_id in &self.node_indices {
            self.neighbors
                .entry(*remaining_node_id)
                .or_default()
                .remove(&node_id);
        }
    }
}
