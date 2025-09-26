use std::collections::HashMap;

use gen_graph::GraphNode;
use gen_models::block_group::BlockGroup;
use pyo3::{prelude::*, types::PyDict};
use rusqlite::Connection;

use super::node_key::PyNodeKey;

// Private factory struct for BlockGroup transformations
// Not exposed to Python, only used internally by the Repository
#[derive(Default)]
pub struct Factory {}

impl Factory {
    pub fn new() -> Self {
        Self::default()
    }

    // Convert a BlockGroup to a dictionary representation
    pub fn to_dict(&self, conn: &Connection, block_group_id: &str) -> PyResult<PyObject> {
        let graph = BlockGroup::get_graph(conn, block_group_id);

        // Convert the graph to a Python dictionary
        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            // Add nodes to the dictionary
            let nodes = PyDict::new(py);
            for node in graph.nodes() {
                let node_dict = PyDict::new(py);
                node_dict.set_item("block_id", node.block_id)?;
                node_dict.set_item("node_id", node.node_id)?;
                node_dict.set_item("sequence_start", node.sequence_start)?;
                node_dict.set_item("sequence_end", node.sequence_end)?;

                let node_key = PyNodeKey::new(node.node_id, node.sequence_start, node.sequence_end);

                nodes.set_item(node_key, node_dict)?;
            }
            dict.set_item("nodes", nodes)?;

            // Add edges to the dictionary
            let edges = PyDict::new(py);
            for (src, dst, edge_weights) in graph.all_edges() {
                let mut weights: Vec<_> = vec![];
                for weight in edge_weights {
                    let weight_dict = PyDict::new(py);
                    weight_dict.set_item("edge_id", weight.edge_id)?;
                    weight_dict.set_item("source_strand", weight.source_strand.to_string())?;
                    weight_dict.set_item("target_strand", weight.target_strand.to_string())?;
                    weight_dict.set_item("chromosome_index", weight.chromosome_index)?;
                    weight_dict.set_item("phased", weight.phased)?;
                    weights.push(weight_dict);
                }

                // Use PyNodeKey without block_id - #[pyclass] objects are automatically converted
                let src_key = PyNodeKey::new(src.node_id, src.sequence_start, src.sequence_end);

                let dst_key = PyNodeKey::new(dst.node_id, dst.sequence_start, dst.sequence_end);

                let edge_key = (src_key, dst_key);
                edges.set_item(edge_key, weights)?;
            }
            dict.set_item("edges", edges)?;

            // Convert the final dictionary to a PyObject
            Ok(dict.into_pyobject(py)?.into_any().unbind())
        })
    }

    // Convert a BlockGroup to a rustworkx graph representation
    pub fn to_rustworkx(&self, conn: &Connection, block_group_id: &str) -> PyResult<PyObject> {
        let graph = BlockGroup::get_graph(conn, block_group_id);

        Python::with_gil(|py| {
            // Import rustworkx module
            let rustworkx = PyModule::import(py, "rustworkx")?;

            // Create a new PyDiGraph
            let py_digraph = rustworkx.getattr("PyDiGraph")?.call0()?;

            // Create a mapping from our GraphNode to rustworkx node indices
            let mut node_map: HashMap<GraphNode, usize> = HashMap::new();

            // Add nodes to the rustworkx graph
            for node in graph.nodes() {
                // Create a Python dictionary to store node data
                let node_data = PyDict::new(py);
                node_data.set_item("block_id", node.block_id)?;
                node_data.set_item("node_id", node.node_id)?;
                node_data.set_item("sequence_start", node.sequence_start)?;
                node_data.set_item("sequence_end", node.sequence_end)?;

                // Add PyNodeKey to node data for easier reference
                let node_key = PyNodeKey::new(node.node_id, node.sequence_start, node.sequence_end);
                node_data.set_item("key", node_key)?;

                // Add the node to the rustworkx graph and store its index
                let index: usize = py_digraph
                    .call_method1("add_node", (node_data,))?
                    .extract()?;
                node_map.insert(node, index);
            }

            // Add edges to the rustworkx graph
            for (src, dst, edge_weights) in graph.all_edges() {
                // Get the rustworkx node indices
                let src_idx = *node_map.get(&src).unwrap();
                let dst_idx = *node_map.get(&dst).unwrap();

                // Create a Python dictionary to store edge data
                let mut weights: Vec<_> = vec![];
                for weight in edge_weights {
                    let weight_dict = PyDict::new(py);
                    weight_dict.set_item("edge_id", weight.edge_id)?;
                    weight_dict.set_item("source_strand", weight.source_strand.to_string())?;
                    weight_dict.set_item("target_strand", weight.target_strand.to_string())?;
                    weight_dict.set_item("chromosome_index", weight.chromosome_index)?;
                    weight_dict.set_item("phased", weight.phased)?;
                    weights.push(weight_dict);
                }

                // Add the edge to the rustworkx graph
                py_digraph.call_method1("add_edge", (src_idx, dst_idx, weights))?;
            }

            // Convert the final graph to a PyObject
            Ok(py_digraph.into_pyobject(py)?.into_any().unbind())
        })
    }

    // Convert a BlockGroup to a NetworkX graph representation
    pub fn to_networkx(&self, conn: &Connection, block_group_id: &str) -> PyResult<PyObject> {
        let graph = BlockGroup::get_graph(conn, block_group_id);

        Python::with_gil(|py| {
            // Import networkx module
            let networkx = PyModule::import(py, "networkx")?;

            // Create a new DiGraph
            let nx_digraph = networkx.getattr("DiGraph")?.call0()?;

            // Add nodes to the networkx graph
            for node in graph.nodes() {
                // Create a Python dictionary to store node data
                let node_data = PyDict::new(py);
                node_data.set_item("block_id", node.block_id)?;
                node_data.set_item("node_id", node.node_id)?;
                node_data.set_item("sequence_start", node.sequence_start)?;
                node_data.set_item("sequence_end", node.sequence_end)?;

                // Create a PyNodeKey for the node (without block_id)
                let node_key = PyNodeKey::new(node.node_id, node.sequence_start, node.sequence_end);

                // Add the node to the NetworkX graph with its attributes
                let kwargs = PyDict::new(py);
                kwargs.set_item("attr_dict", node_data)?;
                nx_digraph.call_method("add_node", (node_key,), Some(&kwargs))?;
            }

            // Add edges to the networkx graph
            for (src, dst, edge_weights) in graph.all_edges() {
                let src_key = PyNodeKey::new(src.node_id, src.sequence_start, src.sequence_end);

                let dst_key = PyNodeKey::new(dst.node_id, dst.sequence_start, dst.sequence_end);

                let mut weights: Vec<_> = vec![];
                for weight in edge_weights {
                    let weight_dict = PyDict::new(py);
                    weight_dict.set_item("edge_id", weight.edge_id)?;
                    weight_dict.set_item("source_strand", weight.source_strand.to_string())?;
                    weight_dict.set_item("target_strand", weight.target_strand.to_string())?;
                    weight_dict.set_item("chromosome_index", weight.chromosome_index)?;
                    weight_dict.set_item("phased", weight.phased)?;
                    weights.push(weight_dict);
                }

                let kwargs = PyDict::new(py);
                kwargs.set_item("attr_dict", weights)?;
                nx_digraph.call_method("add_edge", (src_key, dst_key), Some(&kwargs))?;
            }

            // Convert the final graph to a PyObject
            Ok(nx_digraph.into_pyobject(py)?.into_any().unbind())
        })
    }
}
