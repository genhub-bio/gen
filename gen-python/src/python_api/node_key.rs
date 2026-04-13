use r#gen::core::HashId;
use pyo3::prelude::*;

use super::hash_id::PyHashId;

/// A Python-friendly representation of a block (node id, sequence start, sequence end).
/// Used to ensure consistent hashing when used as dictionary keys in Python.
#[pyclass] // pyclass includes  #[derive(IntoPyObject)]
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

    fn __eq__(&self, py: Python<'_>, other: PyObject) -> PyResult<bool> {
        if let Ok(other_key) = other.extract::<PyRef<PyBlock>>(py) {
            Ok(self.node_id == other_key.node_id
                && self.sequence_start == other_key.sequence_start
                && self.sequence_end == other_key.sequence_end)
        } else {
            Ok(false)
        }
    }
}
