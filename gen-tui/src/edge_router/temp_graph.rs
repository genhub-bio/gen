use std::{
    collections::{HashMap, HashSet, hash_map},
    fmt::Write,
};

use petgraph::stable_graph::StableGraph;

use super::{LayoutError, NodeData};
use crate::layout::LayoutEdge;

#[derive(Clone, Debug)]
pub struct TempGraph {
    node_indices: HashSet<u64>,
    nodes_by_index: HashMap<u64, NodeData>,
    edge_indices: HashSet<u64>,
    edges_by_index: HashMap<u64, LayoutEdge>,
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
        edge: LayoutEdge,
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

    pub fn update_edge(&mut self, edge_id: u64, edge: LayoutEdge) -> Result<(), LayoutError> {
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

    pub fn get_edge(&self, edge_id: u64) -> Option<LayoutEdge> {
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

    pub fn edge_indices(&self) -> hash_map::IntoKeys<u64, LayoutEdge> {
        self.edges_by_index.clone().into_keys()
    }

    pub fn to_stable_graph(&self) -> StableGraph<NodeData, LayoutEdge> {
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

    /// Export this graph to DOT format for debugging
    #[allow(dead_code)]
    pub fn export_to_dot(&self, base_filename: &str, title: &str) {
        let test_timestamp = 0u64;

        let mut dot = String::new();
        writeln!(&mut dot, "// {}", title).unwrap();
        writeln!(&mut dot, "graph {{").unwrap();
        writeln!(&mut dot, "    layout=neato;").unwrap();
        writeln!(&mut dot, "    overlap=false;").unwrap();
        writeln!(&mut dot, "    label=\"{}\";", title).unwrap();
        writeln!(&mut dot, "    labelloc=t;").unwrap();

        // Check for duplicate coordinates
        let mut position_to_nodes = HashMap::new();
        for node_data in self.nodes() {
            let pos = node_data.position;
            position_to_nodes
                .entry(pos)
                .or_insert_with(Vec::new)
                .push(node_data.node_id);
        }

        let duplicate_positions: HashSet<(i64, i64)> = position_to_nodes
            .iter()
            .filter(|(_, nodes)| nodes.len() > 1)
            .map(|(&pos, _)| pos)
            .collect();

        // Write nodes with positions
        for node_data in self.nodes() {
            let x = node_data.position.0;
            let y = node_data.position.1;

            // Create label based on node type, using original_node_id for data nodes
            let node_title = if let Some(ref node_type) = node_data.node_type {
                if node_type == "Routing" {
                    format!("R{}", node_data.node_id)
                } else if node_type == "Graph" {
                    if let Some(original_id) = node_data.original_node_id {
                        format!("N{}=D{}", node_data.node_id, original_id)
                    } else {
                        format!("N{} (D?)", node_data.node_id) // Fallback
                    }
                } else {
                    format!("N{}", node_data.node_id)
                }
            } else {
                // Default to data node if no type specified
                if let Some(original_id) = node_data.original_node_id {
                    format!("maybe D{}", original_id)
                } else {
                    format!("maybe N{}", node_data.node_id)
                }
            };

            // Add coordinates below the title
            let label = format!("{}\\n({},{})", node_title, x, y);

            // Shape based on node type (matching layout.rs to_dot function)
            let shape = if let Some(ref node_type) = node_data.node_type {
                if node_type == "Routing" {
                    "circle"
                } else if node_type == "Stitching" {
                    "diamond"
                } else {
                    "box" // Data nodes
                }
            } else {
                "box" // Default for nodes without type
            };

            // Color nodes with duplicate coordinates red
            let color_attr = if duplicate_positions.contains(&(x, y)) {
                ", color=\"red\", fillcolor=\"pink\", style=\"filled\""
            } else {
                ""
            };

            writeln!(
                &mut dot,
                "    n{} [label=\"{}\", pos=\"{},{}!\", shape=\"{}\"{}];",
                node_data.node_id, label, x, y, shape, color_attr
            )
            .unwrap();
        }

        // Write edges with color coding for non-rectilinear edges
        let mut edges_written = HashSet::new();
        for node_data in self.nodes() {
            let node_id = node_data.node_id;
            let node_pos = node_data.position;

            for neighbor_id in self.neighbors(node_id) {
                // Avoid duplicate edges in undirected graph
                let edge = if node_id < neighbor_id {
                    (node_id, neighbor_id)
                } else {
                    (neighbor_id, node_id)
                };

                if !edges_written.contains(&edge) {
                    // Get neighbor position to check if edge is rectilinear
                    let neighbor_pos = self
                        .get_node(neighbor_id)
                        .map(|n| n.position)
                        .unwrap_or((0, 0));

                    // Check if edge is perfectly horizontal or vertical
                    let is_rectilinear =
                        node_pos.0 == neighbor_pos.0 || node_pos.1 == neighbor_pos.1;

                    if is_rectilinear {
                        writeln!(&mut dot, "    n{} -- n{};", edge.0, edge.1).unwrap();
                    } else {
                        // Color non-rectilinear edges red
                        writeln!(
                            &mut dot,
                            "    n{} -- n{} [color=\"red\", penwidth=3];",
                            edge.0, edge.1
                        )
                        .unwrap();
                    }

                    edges_written.insert(edge);
                }
            }
        }

        writeln!(&mut dot, "}}").unwrap();

        // Extract layer information from node positions
        // Include all nodes to determine layer boundaries
        let mut x_positions = HashSet::new();
        for node_data in self.nodes() {
            x_positions.insert(node_data.position.0);
        }

        let mut sorted_positions: Vec<_> = x_positions.into_iter().collect();
        sorted_positions.sort();

        // Convert x coordinates to layer indices
        let layer_str = if sorted_positions.len() >= 2 {
            // Layers are 0-indexed based on position in sorted x coordinates
            let min_layer = 0;
            let max_layer = sorted_positions.len() - 1;
            format!("layers_{}_{}", min_layer, max_layer)
        } else if sorted_positions.len() == 1 {
            "layer_0".to_string()
        } else {
            "no_layers".to_string()
        };

        // Write to file with timestamp and layer info
        let filename = format!("{}_{}_{}.dot", test_timestamp, base_filename, layer_str);
        std::fs::write(&filename, dot).expect("Unable to write DOT file");
        println!("Exported graph to {}", filename);
    }
}
