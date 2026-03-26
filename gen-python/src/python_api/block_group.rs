use std::collections::HashMap;

use r#gen::core::HashId;
use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::{GraphNode, project_path};
use gen_models::{block_group::BlockGroup, node::Node, path::Path, traits::Query};
use pyo3::prelude::*;
use serde_json::to_string;

use super::{hash_id::PyHashId, repository::PyRepository};

/// Exposes a BlockGroup to Python.
#[pyclass]
pub struct PyBlockGroup {
    pub id: HashId,
    #[pyo3(get)]
    pub collection_name: String,
    #[pyo3(get)]
    pub sample_name: Option<String>,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub repository: Option<Py<PyRepository>>,
}

impl Clone for PyBlockGroup {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyBlockGroup {
            id: self.id,
            collection_name: self.collection_name.clone(),
            sample_name: self.sample_name.clone(),
            name: self.name.clone(),
            repository: self.repository.as_ref().map(|r| r.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyBlockGroup {
    #[new]
    pub fn new(
        id: HashId,
        collection_name: String,
        name: String,
        sample_name: Option<String>,
    ) -> Self {
        PyBlockGroup {
            id,
            collection_name,
            sample_name,
            name,
            repository: None,
        }
    }

    #[getter]
    fn id(&self) -> PyHashId {
        PyHashId::new(self.id)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "BlockGroup({}, {}, {:?}, {})",
            self.id, self.collection_name, self.sample_name, self.name
        ))
    }

    fn __hash__(&self) -> PyResult<isize> {
        // Combine all fields for a more comprehensive hash
        let mut hash = self.id.0.len() as isize;
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.collection_name.len() as isize);
        if let Some(ref sample_name) = self.sample_name {
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(sample_name.len() as isize);
        }
        hash = hash.wrapping_mul(31).wrapping_add(self.name.len() as isize);
        Ok(hash)
    }

    fn __eq__(&self, py: Python<'_>, other: PyObject) -> PyResult<bool> {
        // Try to extract PyBlockGroup from the PyObject
        if let Ok(other_bg) = other.extract::<PyRef<PyBlockGroup>>(py) {
            Ok(self.id == other_bg.id
                && self.collection_name == other_bg.collection_name
                && self.sample_name == other_bg.sample_name
                && self.name == other_bg.name)
        } else {
            // If other is not a PyBlockGroup, they're not equal
            Ok(false)
        }
    }

    /// Serializes the block group topology to a JSON string for the WASM widget.
    pub fn to_widget_json(&self, py: Python<'_>) -> PyResult<String> {
        let repo = self.repository.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "PyBlockGroup has no associated repository",
            )
        })?;
        let bg_id = self.id;
        repo.borrow(py).with_connection(|conn| {
            let graph = BlockGroup::get_graph(conn, &bg_id);
            to_string(&graph)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Serializes the most recent path's nodes to a JSON string matching the
    /// `Vec<GraphNode>` schema expected by the WASM widget's `path_nodes_json`
    /// parameter.  Returns `"[]"` when no path exists.
    pub fn path_nodes_json(&self, py: Python<'_>) -> PyResult<String> {
        let repo = self.repository.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "PyBlockGroup has no associated repository",
            )
        })?;
        let bg_id = self.id;
        repo.borrow(py).with_connection(|conn| {
            let path = Path::get(
                conn,
                "SELECT * FROM paths WHERE block_group_id = ?1 ORDER BY created_on DESC LIMIT 1",
                rusqlite::params![bg_id],
            );
            let path = match path {
                Ok(p) => p,
                Err(_) => return Ok("[]".to_string()),
            };

            let path_blocks = path.blocks(conn);
            let graph = BlockGroup::get_graph(conn, &bg_id);
            let projected = project_path(&graph, &path_blocks);

            let nodes: Vec<&GraphNode> = projected
                .iter()
                .filter_map(|(node, _strand)| {
                    if node.node_id == PATH_START_NODE_ID || node.node_id == PATH_END_NODE_ID {
                        None
                    } else {
                        Some(node)
                    }
                })
                .collect();

            to_string(&nodes)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Fetches sequences for multiple spec strings in a single database query
    /// and returns the result as a JSON object string mapping spec → sequence.
    ///
    /// This is more efficient than calling `get_sequence_by_spec` in a loop
    /// because it batches the node lookup into one `Node::get_sequences_by_node_ids`
    /// call, and the result is serialized directly to JSON without Python dict
    /// construction.
    pub fn get_sequences_json(&self, py: Python<'_>, specs: Vec<String>) -> PyResult<String> {
        let repo = self.repository.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "PyBlockGroup has no associated repository",
            )
        })?;

        // Parse all specs up-front so we can batch the DB lookup.
        struct ParsedSpec {
            spec: String,
            node_id: HashId,
            sequence_start: i64,
            sequence_end: i64,
        }

        let mut parsed: Vec<ParsedSpec> = Vec::with_capacity(specs.len());
        for spec in specs {
            let (node_id_hex, range) = spec.split_once(':').ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid spec (missing ':'): {spec}"
                ))
            })?;
            let (start_str, end_str) = range.split_once('-').ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid spec range (missing '-'): {range}"
                ))
            })?;
            let node_id = HashId::try_from(node_id_hex.to_string()).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid node id: {e}"))
            })?;
            let sequence_start: i64 = start_str.parse().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid start: {e}"))
            })?;
            let sequence_end: i64 = end_str.parse().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid end: {e}"))
            })?;
            parsed.push(ParsedSpec { spec, node_id, sequence_start, sequence_end });
        }

        repo.borrow(py).with_connection(|conn| {
            let node_ids: Vec<HashId> = parsed.iter().map(|p| p.node_id).collect();
            let sequences = Node::get_sequences_by_node_ids(conn, &node_ids);

            let mut result: HashMap<&str, String> = HashMap::with_capacity(parsed.len());
            for p in &parsed {
                if let Some(seq) = sequences.get(&p.node_id) {
                    result.insert(&p.spec, seq.get_sequence(p.sequence_start, p.sequence_end));
                }
            }

            to_string(&result)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Returns the DNA sequence for a node identified by its spec string.
    ///
    /// The spec format is `node_id_hex:sequence_start-sequence_end`, which is
    /// the same key used by the WASM renderer when requesting sequences.
    pub fn get_sequence_by_spec(&self, py: Python<'_>, spec: &str) -> PyResult<String> {
        let repo = self.repository.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "PyBlockGroup has no associated repository",
            )
        })?;
        let (node_id_hex, range) = spec.split_once(':').ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid spec (missing ':'): {spec}"))
        })?;
        let (start_str, end_str) = range.split_once('-').ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid spec range (missing '-'): {range}"
            ))
        })?;
        let node_id = HashId::try_from(node_id_hex.to_string())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid node id: {e}")))?;
        let sequence_start: i64 = start_str
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid start: {e}")))?;
        let sequence_end: i64 = end_str
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid end: {e}")))?;

        repo.borrow(py).with_connection(|conn| {
            let sequences = Node::get_sequences_by_node_ids(conn, &[node_id]);
            let seq = sequences.get(&node_id).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "node {node_id_hex} not found"
                ))
            })?;
            Ok(seq.get_sequence(sequence_start, sequence_end))
        })
    }
}
