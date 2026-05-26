use r#gen::core::HashId;
use gen_core::Strand;
use gen_graph::BlockSlice;
use pyo3::prelude::*;

use super::hash_id::PyHashId;

/// A Python-friendly representation of a block (node id, sequence start, sequence end).
/// Used to ensure consistent hashing when used as dictionary keys in Python.
#[pyclass(name = "Block")] // pyclass includes  #[derive(IntoPyObject)]
#[derive(Clone, Copy)]
pub struct PyBlock {
    pub node_id: HashId,
    #[pyo3(get)]
    pub sequence_start: i64,
    #[pyo3(get)]
    pub sequence_end: i64,
}

#[pymethods]
impl PyBlock {
    #[new]
    pub fn new(node_id: HashId, sequence_start: i64, sequence_end: i64) -> Self {
        PyBlock {
            node_id,
            sequence_start,
            sequence_end,
        }
    }

    #[getter]
    fn node_id(&self) -> PyHashId {
        PyHashId::new(self.node_id)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Block({}, {}, {})",
            self.node_id, self.sequence_start, self.sequence_end
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
        if let Ok(other_key) = other.extract::<PyRef<PyBlock>>() {
            Ok(self.node_id == other_key.node_id
                && self.sequence_start == other_key.sequence_start
                && self.sequence_end == other_key.sequence_end)
        } else {
            Ok(false)
        }
    }
}

/// A slice of a single graph block with local byte offsets and strand.
#[pyclass(name = "BlockSlice")]
#[derive(Clone, Copy)]
pub struct PyBlockSlice {
    pub inner: BlockSlice,
}

impl PyBlockSlice {
    pub fn from_slice(s: BlockSlice) -> Self {
        Self { inner: s }
    }
}

#[pymethods]
impl PyBlockSlice {
    #[getter]
    fn block(&self) -> PyBlock {
        PyBlock::new(
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
            "BlockSlice({}[{}..{}], {}..{}, strand={})",
            self.inner.block.node_id,
            self.inner.block.sequence_start,
            self.inner.block.sequence_end,
            self.inner.start,
            self.inner.end,
            self.strand(),
        )
    }
}
