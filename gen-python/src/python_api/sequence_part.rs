use pyo3::{prelude::*, pyclass};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(name = "SequencePart")]
#[derive(Clone)]
pub struct PySequencePart {
    pub name: String,
    pub sequence: String,
    pub sequence_length: i64,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySequencePart {
    /// Name of this part.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Sequence of this part.
    #[getter]
    fn sequence(&self) -> &str {
        &self.sequence
    }

    /// A named sequence option for one column of a combinatorial library.
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
