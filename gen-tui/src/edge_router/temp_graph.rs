use std::collections::{HashMap, HashSet, hash_map};

use super::{LayoutError, NodeData};

#[derive(Clone, Debug)]
pub struct TempGraph {
    nodes_by_index: HashMap<u64, NodeData>,
    neighbors: HashMap<u64, HashSet<u64>>,
}

impl Default for TempGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TempGraph {
    pub fn new() -> Self {
        Self {
            nodes_by_index: HashMap::new(),
            neighbors: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node_id: u64, node: NodeData) {
        self.nodes_by_index.insert(node_id, node);
    }

    pub fn update_node(&mut self, node_id: u64, node: NodeData) -> Result<(), LayoutError> {
        if !self.nodes_by_index.contains_key(&node_id) {
            return Err(LayoutError::NodeNotFound(node_id));
        }
        self.nodes_by_index.insert(node_id, node);

        Ok(())
    }

    pub fn add_edge(&mut self, source: u64, target: u64) -> Result<(), LayoutError> {
        if !self.nodes_by_index.contains_key(&source) {
            return Err(LayoutError::NodeNotFound(source));
        }
        if !self.nodes_by_index.contains_key(&target) {
            return Err(LayoutError::NodeNotFound(target));
        }
        self.neighbors.entry(source).or_default().insert(target);

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

    pub fn node_count(&self) -> usize {
        self.nodes_by_index.len()
    }

    pub fn neighbors(&self, node_id: u64) -> HashSet<u64> {
        self.neighbors.get(&node_id).cloned().unwrap_or_default()
    }
}
