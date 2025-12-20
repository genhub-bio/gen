use r#gen::core::HashId;
use pyo3::prelude::*;

/// Exposes a BlockGroup to Python.
#[pyclass]
#[derive(Clone)]
pub struct PyBlockGroup {
    #[pyo3(get)]
    pub id: HashId,
    #[pyo3(get)]
    pub collection_name: String,
    #[pyo3(get)]
    pub sample_name: Option<String>,
    #[pyo3(get)]
    pub name: String,
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
        }
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
}
