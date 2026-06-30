use pyo3::{exceptions::PyIndexError, prelude::*};

use crate::python_api::{
    block_group::PySequenceGraph,
    jupyter_widget::{PyGraphController, build_widget},
};

/// The sequence graphs produced by a single import/update/derive call, all
/// within one sample.
///
/// Acts like a read-only list of ``SequenceGraph``: index it, iterate it, or
/// call ``len()`` on it. Indexing out of range raises ``IndexError``.
#[pyclass(name = "Sample", unsendable)]
#[derive(Clone)]
pub struct PySample {
    #[pyo3(get)]
    pub collection_name: String,
    #[pyo3(get)]
    pub sample_name: String,
    #[pyo3(get)]
    pub block_groups: Vec<PySequenceGraph>,
}

impl PySample {
    pub fn new(
        collection_name: String,
        sample_name: String,
        block_groups: Vec<PySequenceGraph>,
    ) -> Self {
        PySample {
            collection_name,
            sample_name,
            block_groups,
        }
    }
}

#[pymethods]
impl PySample {
    fn __len__(&self) -> usize {
        self.block_groups.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<PySequenceGraph> {
        let len = self.block_groups.len() as isize;
        let i = if index < 0 { index + len } else { index };
        if i < 0 || i >= len {
            return Err(PyIndexError::new_err("Sample index out of range"));
        }
        Ok(self.block_groups[i as usize].clone())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PySampleIter>> {
        Py::new(
            slf.py(),
            PySampleIter {
                block_groups: slf.block_groups.clone(),
                index: 0,
            },
        )
    }

    /// Plot this sample as an interactive Jupyter widget that pages through
    /// each of its sequence graphs.
    ///
    /// Displays the widget immediately and returns it for further use.
    /// Outside of an IPython/Jupyter environment the display call is silently
    /// skipped and only the widget is returned.
    ///
    /// Parameters
    /// ----------
    /// rows : int, optional
    ///     Initial viewport height in terminal rows.
    /// cols : int, optional
    ///     Initial viewport width in terminal columns.
    /// colors : callable | dict | list, optional
    ///     Controls how annotation group entries are coloured when they are
    ///     auto-loaded from the repository. See ``Repository.plot`` for details.
    #[pyo3(signature = (rows=None, cols=None, colors=None))]
    fn plot(
        slf: &Bound<'_, PySample>,
        rows: Option<u32>,
        cols: Option<u32>,
        colors: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        let ctrl = PyGraphController::for_sample(&slf.borrow().block_groups)?;
        let ctrl = Py::new(py, ctrl)?;
        build_widget(py, ctrl, rows, cols, colors)
    }

    /// IPython display hook — called when a cell ends with a Sample.
    fn _ipython_display_(slf: &Bound<'_, PySample>) -> PyResult<()> {
        let py = slf.py();
        let widget = slf.call_method0("plot")?;
        PyModule::import(py, "IPython.display")?.call_method1("display", (widget,))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let mut lines = vec![format!(
            "Sample({:?}, collection={:?}, {} sequence graph{}):",
            self.sample_name,
            self.collection_name,
            self.block_groups.len(),
            if self.block_groups.len() == 1 {
                ""
            } else {
                "s"
            }
        )];
        for (i, bg) in self.block_groups.iter().enumerate() {
            lines.push(format!("  {}: {}", i, bg.name));
        }
        lines.join("\n")
    }
}

#[pyclass(unsendable)]
pub struct PySampleIter {
    block_groups: Vec<PySequenceGraph>,
    index: usize,
}

#[pymethods]
impl PySampleIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PySequenceGraph> {
        let bg = slf.block_groups.get(slf.index).cloned();
        slf.index += 1;
        bg
    }
}
