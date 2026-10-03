use r#gen::core::HashId;
use gen_models::{db::DbContext, node::Node};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::hash_id::PyHashId;

/// A stretch of stored sequence in a sequence graph, usable as a dict key.
///
/// Obtain it from `position.node` or the keys of `SequenceGraph.to_dict()`. Read its bases with
/// `.sequence`.
#[gen_stub_pyclass]
#[pyclass(name = "Node", unsendable)] // pyclass includes  #[derive(IntoPyObject)]
#[derive(Clone)]
pub struct PyGraphNode {
    pub node_id: HashId,
    pub sequence_start: i64,
    pub sequence_end: i64,
    // Database connection `.sequence` reads through; `None` for nodes built without a Repository.
    context: Option<DbContext>,
}

impl PyGraphNode {
    pub fn new(node_id: HashId, sequence_start: i64, sequence_end: i64) -> Self {
        PyGraphNode {
            node_id,
            sequence_start,
            sequence_end,
            context: None,
        }
    }

    /// This node reading its sequence through `context`.
    pub fn with_context(mut self, context: Option<DbContext>) -> Self {
        self.context = context;
        self
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGraphNode {
    /// Inclusive start of this block in the underlying node's sequence.
    #[getter(_sequence_start)]
    #[gen_stub(skip)]
    fn py_sequence_start(&self) -> i64 {
        self.sequence_start
    }

    /// Exclusive end of this block in the underlying node's sequence.
    #[getter(_sequence_end)]
    #[gen_stub(skip)]
    fn py_sequence_end(&self) -> i64 {
        self.sequence_end
    }

    /// Stable ID of the underlying sequence node, independent of block boundaries.
    #[getter(_id)]
    #[gen_stub(skip)]
    fn id(&self) -> PyHashId {
        PyHashId::new(self.node_id)
    }

    /// The bases this node holds.
    #[getter]
    fn sequence(&self) -> PyResult<String> {
        let context = self.context.as_ref().ok_or_else(|| {
            PyValueError::new_err("node has no database connection to read its sequence from")
        })?;
        let sequences = Node::get_sequences_by_node_ids(
            context.graph().conn(),
            context.workspace(),
            &[self.node_id],
            None,
        );
        let sequence = sequences.get(&self.node_id).ok_or_else(|| {
            PyValueError::new_err(format!("Node with id {:?} not found", self.node_id))
        })?;
        sequence
            .get_sequence(self.sequence_start, self.sequence_end)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let h = format!("{}", self.node_id);
        let hash8 = &h[..8.min(h.len())];
        Ok(format!(
            "Node({}[{}:{}])",
            hash8, self.sequence_start, self.sequence_end
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        let h = format!("{}", self.node_id);
        let hash8 = &h[..8.min(h.len())];
        Ok(format!(
            "{}:{}-{}",
            hash8, self.sequence_start, self.sequence_end
        ))
    }

    fn __hash__(&self) -> PyResult<isize> {
        // Combine all fields for a consistent hash value
        let mut hash: isize = 0;
        for &b in &self.node_id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.sequence_start as isize);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.sequence_end as isize);
        Ok(hash)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_key) = other.extract::<PyRef<PyGraphNode>>() {
            Ok(self.node_id == other_key.node_id
                && self.sequence_start == other_key.sequence_start
                && self.sequence_end == other_key.sequence_end)
        } else {
            Ok(false)
        }
    }

    /// Length of this node's sequence in bytes.
    #[getter]
    fn length(&self) -> i64 {
        self.sequence_end - self.sequence_start
    }
}
