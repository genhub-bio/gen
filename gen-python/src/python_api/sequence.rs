use r#gen::graphs::combinatorial_library::SequencePart;
use pyo3::{
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    prelude::*,
    pyclass,
    pyclass::CompareOp,
    types::{PySlice, PyString},
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A DNA, RNA or protein sequence, with an optional name.
///
/// `SequenceGraph.all_sequences()` yields these, and combinatorial libraries are built from named
/// ones. `str(sequence)` is just the bases, and it compares, hashes, slices and measures like that
/// string.
#[gen_stub_pyclass]
#[pyclass(name = "Sequence")]
#[derive(Clone)]
pub struct PySequence {
    pub name: Option<String>,
    pub sequence: String,
}

impl PySequence {
    pub fn unnamed(sequence: String) -> Self {
        PySequence {
            name: None,
            sequence,
        }
    }

    /// The library part this sequence stands for, which needs a name to appear in the graph.
    fn to_part(&self) -> PyResult<SequencePart> {
        let name = self.name.clone().ok_or_else(|| {
            PyValueError::new_err(
                "library parts need a name: build them as Sequence(name, sequence)",
            )
        })?;
        Ok(SequencePart {
            name,
            sequence: self.sequence.clone(),
            sequence_length: self.sequence.len() as i64,
        })
    }
}

/// The columns of a combinatorial library, as the library builders take them.
pub fn library_parts(columns: &[Vec<PySequence>]) -> PyResult<Vec<Vec<SequencePart>>> {
    columns
        .iter()
        .map(|column| column.iter().map(PySequence::to_part).collect())
        .collect()
}

#[gen_stub_pymethods]
#[pymethods]
impl PySequence {
    /// A named sequence, such as one option for a column of a combinatorial library.
    #[new]
    #[pyo3(signature = (name, sequence))]
    fn new(name: Option<String>, sequence: String) -> Self {
        PySequence { name, sequence }
    }

    /// Name of this sequence, or `None` when it has none.
    #[getter]
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The bases of this sequence as a string.
    #[getter]
    fn sequence(&self) -> &str {
        &self.sequence
    }

    fn __str__(&self) -> &str {
        &self.sequence
    }

    fn __repr__(&self) -> String {
        match &self.name {
            Some(name) => format!("Sequence({name:?}, {:?})", self.sequence),
            None => format!("Sequence({:?})", self.sequence),
        }
    }

    fn __len__(&self) -> usize {
        self.sequence.len()
    }

    fn __hash__(&self, python: Python<'_>) -> PyResult<isize> {
        // Hash like the equal string so a Sequence and its str are one dict key.
        PyString::new(python, &self.sequence).hash()
    }

    /// Compares by bases with another `Sequence` or a `str`, so sequences sort and match like the
    /// strings they read as; the name is not compared.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, operator: CompareOp) -> PyResult<bool> {
        let other_bases = if let Ok(other) = other.extract::<PyRef<PySequence>>() {
            other.sequence.clone()
        } else if let Ok(other) = other.extract::<String>() {
            other
        } else {
            return match operator {
                CompareOp::Eq => Ok(false),
                CompareOp::Ne => Ok(true),
                _ => Err(PyTypeError::new_err(
                    "Sequence can only be ordered against a Sequence or str",
                )),
            };
        };
        Ok(operator.matches(self.sequence.cmp(&other_bases)))
    }

    fn __contains__(&self, needle: &str) -> bool {
        self.sequence.contains(needle)
    }

    /// Index a single base as a string, or slice a sub-`Sequence` with a step of 1.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let python = key.py();
        let length = self.sequence.len() as isize;
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(length)?;
            if indices.step != 1 {
                return Err(PyValueError::new_err("Sequence slices require a step of 1"));
            }
            let start = indices.start as usize;
            let stop = (indices.stop as usize).max(start);
            let part = self
                .sequence
                .get(start..stop)
                .ok_or_else(|| PyValueError::new_err("Sequence is not plain ASCII text"))?;
            return Py::new(python, PySequence::unnamed(part.to_string())).map(Py::into_any);
        }
        let index = key.extract::<isize>()?;
        let offset = if index < 0 { length + index } else { index };
        if offset < 0 || offset >= length {
            return Err(PyIndexError::new_err("Sequence index out of range"));
        }
        let base = self
            .sequence
            .get(offset as usize..offset as usize + 1)
            .ok_or_else(|| PyValueError::new_err("Sequence is not plain ASCII text"))?;
        Ok(base.into_pyobject(python)?.into_any().unbind())
    }
}
