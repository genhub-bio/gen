use r#gen::core::HashId;
use pyo3::{prelude::*, types::PyBytes};

/// Exposes a HashId to Python.
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyHashId {
    #[pyo3(get)]
    pub hash_id: HashId,
}

#[pymethods]
impl PyHashId {
    #[new]
    pub fn new(hash_id: HashId) -> Self {
        PyHashId { hash_id }
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.hash_id.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("HashId(\"{}\")", self.hash_id))
    }

    fn __hash__(&self) -> PyResult<isize> {
        // Combine the bytes of the hash until it fits
        let mut hash: isize = 0;
        for &b in &self.hash_id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        Ok(hash)
    }

    fn __eq__(&self, py: Python<'_>, other: PyObject) -> PyResult<bool> {
        // Try to extract PyHashId from the PyObject
        if let Ok(other_hash_id) = other.extract::<PyRef<PyHashId>>(py) {
            Ok(self.hash_id == other_hash_id.hash_id)
        } else {
            // If other is not a PyHashId, they're not equal
            Ok(false)
        }
    }

    /// Returns the HashId as a 32-byte bytes object.
    fn to_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.hash_id.0).into()
    }
}
