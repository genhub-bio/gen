use r#gen::core::HashId;
use gen_core::Strand;
use gen_graph::GraphNodeSlice;
use pyo3::prelude::*;

/// An opaque handle to a graph node, usable as a dict key in Python.
/// Used to ensure consistent hashing when used as dictionary keys in Python.
#[pyclass(name = "Node")] // pyclass includes  #[derive(IntoPyObject)]
#[derive(Clone, Copy)]
pub struct PyGraphNode {
    pub node_id: HashId,
    pub sequence_start: i64,
    pub sequence_end: i64,
}

impl PyGraphNode {
    pub fn new(node_id: HashId, sequence_start: i64, sequence_end: i64) -> Self {
        PyGraphNode {
            node_id,
            sequence_start,
            sequence_end,
        }
    }
}

#[pymethods]
impl PyGraphNode {
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
}

/// A slice of a single graph block with local byte offsets and strand.
#[pyclass(name = "NodeSlice")]
#[derive(Clone, Copy)]
pub struct PyGraphNodeSlice {
    pub inner: GraphNodeSlice,
}

impl PyGraphNodeSlice {
    pub fn from_slice(s: GraphNodeSlice) -> Self {
        Self { inner: s }
    }
}

#[pymethods]
impl PyGraphNodeSlice {
    #[getter]
    fn block(&self) -> PyGraphNode {
        PyGraphNode::new(
            self.inner.block.node_id,
            self.inner.block.sequence_start,
            self.inner.block.sequence_end,
        )
    }

    #[getter]
    fn start(&self) -> usize {
        self.inner.start
    }

    #[getter]
    fn end(&self) -> usize {
        self.inner.end
    }

    #[getter]
    fn strand(&self) -> &str {
        match self.inner.strand {
            Strand::Forward => "forward",
            Strand::Reverse => "reverse",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeSlice({}[{}..{}], {}..{}, strand={})",
            self.inner.block.node_id,
            self.inner.block.sequence_start,
            self.inner.block.sequence_end,
            self.inner.start,
            self.inner.end,
            self.strand(),
        )
    }
}
