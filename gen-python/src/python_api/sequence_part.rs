use pyo3::{prelude::*, pyclass};

/// Exposes a SequencePart to Python.
#[pyclass]
#[derive(Clone)]
pub struct PySequencePart {
    pub name: String,
    pub sequence: String,
    pub sequence_length: i64,
}

#[pymethods]
impl PySequencePart {
    #[new]
    fn new(name: String, sequence: String, sequence_length: i64) -> Self {
        PySequencePart {
            name,
            sequence,
            sequence_length,
        }
    }
}
