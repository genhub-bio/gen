use std::collections::HashSet;

use gen_models::{
    block_group::BlockGroup,
    operations::{OperationInfo, OperationSummary},
    sample::{NewSample, Sample, SampleError},
    sample_lineage::SampleLineage,
};
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
    prelude::*,
};

use crate::python_api::{
    block_group::PySequenceGraph,
    jupyter_widget::{PyGraphController, build_widget},
    repository::run_context_operation_write,
    utils::block_group_err_to_pyerr,
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

    /// Copy this sample into a new sample with the same sequence graphs.
    ///
    /// The destination name must not already exist. The returned sample is
    /// ready for explicit in-place edits on its sequence graphs. The copy is
    /// recorded as its own operation, using ``message`` as the operation's
    /// commit message when given, or a generated description otherwise.
    #[pyo3(signature = (new_name, message=None))]
    fn copy(&self, new_name: String, message: Option<&str>) -> PyResult<PySample> {
        let context = self
            .block_groups
            .first()
            .and_then(|sequence_graph| sequence_graph.context.as_ref())
            .ok_or_else(|| {
                PyRuntimeError::new_err("copy() requires a sample with sequence graphs")
            })?;
        if new_name.is_empty() || new_name == self.sample_name {
            return Err(PyValueError::new_err(
                "copy() requires a different, non-empty sample name",
            ));
        }

        run_context_operation_write(
            context,
            |context| {
                let conn = context.graph().conn();
                let created_sample = Sample::create(
                    conn,
                    NewSample {
                        name: &new_name,
                        is_reference: false,
                    },
                )
                .map_err(|error| match error {
                    SampleError::Duplicate(_) => {
                        PyValueError::new_err(format!("sample '{new_name}' already exists"))
                    }
                    other => PyRuntimeError::new_err(format!("cannot copy sample: {other}")),
                })?;
                let group_names = self
                    .block_groups
                    .iter()
                    .map(|sequence_graph| sequence_graph.name.clone())
                    .collect::<HashSet<_>>();
                for group_name in group_names {
                    BlockGroup::get_or_create_sample_block_groups(
                        conn,
                        &self.collection_name,
                        &new_name,
                        &group_name,
                        vec![self.sample_name.clone()],
                    )
                    .map_err(block_group_err_to_pyerr)?;
                }
                SampleLineage::create(conn, &self.sample_name, &created_sample.name).map_err(
                    |error| PyRuntimeError::new_err(format!("cannot copy sample: {error}")),
                )?;

                let copied_block_groups =
                    Sample::get_block_groups(conn, &self.collection_name, &new_name, None)
                        .into_iter()
                        .map(|block_group| PySequenceGraph {
                            id: block_group.id,
                            collection_name: block_group.collection_name,
                            sample_name: block_group.sample_name,
                            name: block_group.name,
                            context: Some(context.clone()),
                        })
                        .collect();
                let copied_sample = PySample::new(
                    self.collection_name.clone(),
                    new_name.clone(),
                    copied_block_groups,
                );
                let summary = OperationSummary::new(
                    OperationInfo {
                        files: vec![],
                        description: "sample_copy".to_string(),
                    },
                    message.map_or_else(
                        || format!("copied sample '{}' to '{}'", self.sample_name, new_name),
                        str::to_string,
                    ),
                );
                Ok((copied_sample, summary))
            },
            |error| PyRuntimeError::new_err(format!("failed to copy sample: {error}")),
        )
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
