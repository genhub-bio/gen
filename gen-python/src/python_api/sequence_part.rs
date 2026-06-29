use pyo3::{prelude::*, pyclass};

#[pyclass(name = "SequencePart")]
#[derive(Clone)]
pub struct PySequencePart {
    pub name: String,
    pub sequence: String,
    pub sequence_length: i64,
}

#[pymethods]
impl PySequencePart {
    #[new]
    #[pyo3(signature = (name, sequence))]
    fn new(name: String, sequence: String) -> Self {
        PySequencePart {
            sequence_length: sequence.len() as i64,
            name,
            sequence,
        }
    }
}
