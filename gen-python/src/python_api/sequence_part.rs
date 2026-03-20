use pyo3::pyclass;

/// Exposes a SequencePart to Python.
#[pyclass]
#[derive(Clone)]
pub struct PySequencePart {
    pub name: String,
    pub sequence: String,
    pub sequence_length: i64,
}
