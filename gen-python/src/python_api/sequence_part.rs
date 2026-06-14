use pyo3::{prelude::*, pyclass};

#[pyclass(name = "SequencePart")]
#[derive(Clone)]
pub struct PySequencePart {
    pub name: String,
    pub sequence: String,
    pub sequence_length: i64,
    pub metadata: Option<String>,
    pub annotation_start: Option<i64>,
    pub annotation_end: Option<i64>,
}

#[pymethods]
impl PySequencePart {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn sequence(&self) -> &str {
        &self.sequence
    }

    #[new]
    #[pyo3(signature = (name, sequence, metadata=None, annotation_start=None, annotation_end=None))]
    fn new(
        py: Python<'_>,
        name: String,
        sequence: String,
        metadata: Option<Bound<'_, PyAny>>,
        annotation_start: Option<i64>,
        annotation_end: Option<i64>,
    ) -> PyResult<Self> {
        let metadata_str = if let Some(m) = metadata {
            let json = py.import("json")?;
            Some(json.call_method1("dumps", (m,))?.extract::<String>()?)
        } else {
            None
        };
        Ok(PySequencePart {
            sequence_length: sequence.len() as i64,
            name,
            sequence,
            metadata: metadata_str,
            annotation_start,
            annotation_end,
        })
    }
}
