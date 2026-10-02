use r#gen::core::HashId;
use pyo3::{prelude::*, types::PyBytes};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// Exposes a HashId to Python.
#[gen_stub_pyclass]
#[pyclass(name = "HashId")]
#[derive(Clone, Copy)]
pub struct PyHashId {
    pub hash_id: HashId,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHashId {
    #[new]
    #[gen_stub(skip)]
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

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Try to extract PyHashId from the Py<PyAny>
        if let Ok(other_hash_id) = other.extract::<PyRef<PyHashId>>() {
            Ok(self.hash_id == other_hash_id.hash_id)
        } else {
            // If other is not a PyHashId, they're not equal
            Ok(false)
        }
    }

    /// Returns the HashId as a 16-byte bytes object.
    #[expect(
        clippy::wrong_self_convention,
        reason = "exposed to Python as to_bytes(); pyo3 pyclass methods require &self"
    )]
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.hash_id.0)
    }
}
