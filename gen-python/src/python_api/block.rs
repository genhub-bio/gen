use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use r#gen::core::HashId;
use pyo3::prelude::*;

use super::hash_id::PyHashId;

/// A block in the graph: a node with a sequence window and a stable identity.
///
/// Construct with `Block(sequence)` — a UUID-7 node ID is generated automatically.
/// Optional keyword args: `sequence_start`, `sequence_end`, `node_id`.
///
/// Use `Block` as the node key when building a NetworkX graph for
/// `Repository.create_block_group_from_graph()`:
///
///     b = Block("ACGT")
///     G.add_node(b)
#[pyclass(name = "Block", subclass)]
#[derive(Clone)]
pub struct PyBlock {
    pub node_id: HashId,
    pub node_sequence: String,
    #[pyo3(get)]
    pub sequence_start: i64,
    #[pyo3(get)]
    pub sequence_end: i64,
}

#[pymethods]
impl PyBlock {
    #[new]
    #[pyo3(signature = (sequence, sequence_start=None, sequence_end=None, node_id=None))]
    pub fn new(
        sequence: String,
        sequence_start: Option<i64>,
        sequence_end: Option<i64>,
        node_id: Option<PyHashId>,
    ) -> Self {
        let seq_len = sequence.len() as i64;
        PyBlock {
            node_id: node_id.map(|h| h.hash_id).unwrap_or_else(HashId::uuid7),
            sequence_start: sequence_start.unwrap_or(0),
            sequence_end: sequence_end.unwrap_or(seq_len),
            node_sequence: sequence,
        }
    }

    #[getter]
    fn node_id(&self) -> PyHashId {
        PyHashId::new(self.node_id)
    }

    #[getter]
    fn _node_sequence(&self) -> &str {
        &self.node_sequence
    }

    fn __repr__(&self) -> String {
        let preview = &self.node_sequence[..self.node_sequence.len().min(20)];
        let ellipsis = if self.node_sequence.len() > 20 { "…" } else { "" };
        format!(
            "Block({:?}{}, {}..{})",
            preview, ellipsis, self.sequence_start, self.sequence_end
        )
    }

    fn __hash__(&self) -> isize {
        let mut hash: isize = 0;
        for &b in &self.node_id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        hash = hash.wrapping_mul(31).wrapping_add(self.sequence_start as isize);
        hash = hash.wrapping_mul(31).wrapping_add(self.sequence_end as isize);
        hash
    }

    fn __eq__(&self, py: Python<'_>, other: PyObject) -> bool {
        other
            .extract::<PyRef<PyBlock>>(py)
            .map(|o| {
                self.node_id == o.node_id
                    && self.sequence_start == o.sequence_start
                    && self.sequence_end == o.sequence_end
            })
            .unwrap_or(false)
    }
}

/// Sentinel representing the upstream boundary of a path.
#[pyclass(name = "StartBlock", extends = PyBlock)]
pub struct PyStartBlock;

#[pymethods]
impl PyStartBlock {
    #[new]
    pub fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyBlock {
            node_id: PATH_START_NODE_ID,
            node_sequence: String::new(),
            sequence_start: 0,
            sequence_end: 0,
        })
        .add_subclass(PyStartBlock)
    }

    fn __repr__(&self) -> &str {
        "StartBlock()"
    }
}

/// Sentinel representing the downstream boundary of a path.
#[pyclass(name = "EndBlock", extends = PyBlock)]
pub struct PyEndBlock;

#[pymethods]
impl PyEndBlock {
    #[new]
    pub fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyBlock {
            node_id: PATH_END_NODE_ID,
            node_sequence: String::new(),
            sequence_start: 0,
            sequence_end: 0,
        })
        .add_subclass(PyEndBlock)
    }

    fn __repr__(&self) -> &str {
        "EndBlock()"
    }
}
