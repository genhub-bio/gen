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
    fn new(name: String, sequence: String) -> Self {
        PySequencePart {
            name,
            sequence: sequence.clone(),
            sequence_length: sequence.len() as i64,
        }
    }
}
