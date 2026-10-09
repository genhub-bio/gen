use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    operations::{OperationInfo, OperationSummary},
    sample::{NewSample, Sample, SampleError},
    sample_lineage::SampleLineage,
};
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::python_api::{
    block_group::PySequenceGraph,
    jupyter_widget::{PyGraphController, build_widget},
    repository::run_context_operation_write,
    utils::block_group_err_to_pyerr,
};

mod metadata;

/// A repository-bound view of sequence graphs sharing a sample name and collection.
///
/// Create a handle with ``Repository.sample`` or retrieve an existing one with
/// ``Repository.get_sample``. A handle can exist before any matching sequence graph does; it does
/// not create repository data by itself. It queries the repository whenever its graphs are
/// accessed, so the membership stays current. Acts like a read-only list of ``SequenceGraph``:
/// index it, iterate it, or call ``len()`` on it. Each iteration uses the graphs present when that
/// iteration starts. Indexing out of range raises ``IndexError``.
#[gen_stub_pyclass]
#[pyclass(name = "Sample", unsendable)]
#[derive(Clone)]
pub struct PySample {
    /// Collection this sample belongs to.
    #[pyo3(get, name = "collection")]
    pub collection_name: String,
    /// Name of the sample.
    #[pyo3(get, name = "name")]
    pub sample_name: String,
    pub(crate) context: DbContext,
}

impl PySample {
    pub fn new(collection_name: String, sample_name: String, context: DbContext) -> Self {
        PySample {
            collection_name,
            sample_name,
            context,
        }
    }

    fn load_sequence_graphs(&self) -> Vec<PySequenceGraph> {
        Sample::get_block_groups(
            self.context.graph().conn(),
            &self.collection_name,
            &self.sample_name,
            None,
        )
        .into_iter()
        .map(|block_group| PySequenceGraph {
            id: block_group.id,
            collection_name: block_group.collection_name,
            sample_name: block_group.sample_name,
            name: block_group.name,
            context: Some(self.context.clone()),
        })
        .collect()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySample {
    /// All sequence graphs held by this sample.
    #[getter]
    fn sequence_graphs(&self) -> Vec<PySequenceGraph> {
        self.load_sequence_graphs()
    }

    fn __len__(&self) -> usize {
        self.load_sequence_graphs().len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<PySequenceGraph> {
        let sequence_graphs = self.load_sequence_graphs();
        let len = sequence_graphs.len() as isize;
        let i = if index < 0 { index + len } else { index };
        if i < 0 || i >= len {
            return Err(PyIndexError::new_err("Sample index out of range"));
        }
        Ok(sequence_graphs[i as usize].clone())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PySampleIter>> {
        let sequence_graphs = slf.load_sequence_graphs();
        Py::new(
            slf.py(),
            PySampleIter {
                block_groups: sequence_graphs,
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
    /// show_history : bool, optional
    ///     Keep retired edit-site and pruned edges in the graph, dimmed,
    ///     instead of removing them along with the nodes only they reach.
    ///     Defaults to ``False``.
    #[pyo3(signature = (rows=None, cols=None, colors=None, show_history=false))]
    fn plot(
        slf: &Bound<'_, PySample>,
        rows: Option<u32>,
        cols: Option<u32>,
        colors: Option<Py<PyAny>>,
        show_history: bool,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let sequence_graphs = slf.borrow().load_sequence_graphs();
        let ctrl = PyGraphController::for_sample(&sequence_graphs, show_history)?;
        let ctrl = Py::new(py, ctrl)?;
        build_widget(py, ctrl, rows, cols, colors)
    }

    /// IPython display hook — called when a cell ends with a Sample.
    #[gen_stub(skip)]
    fn _ipython_display_(slf: &Bound<'_, PySample>) -> PyResult<()> {
        let py = slf.py();
        let widget = slf.call_method0("plot")?;
        PyModule::import(py, "IPython.display")?.call_method1("display", (widget,))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let sequence_graphs = self.load_sequence_graphs();
        let mut lines = vec![format!(
            "Sample({:?}, collection={:?}, {} sequence graph{}):",
            self.sample_name,
            self.collection_name,
            sequence_graphs.len(),
            if sequence_graphs.len() == 1 { "" } else { "s" }
        )];
        for (i, sequence_graph) in sequence_graphs.iter().enumerate() {
            lines.push(format!("  {}: {}", i, sequence_graph.name));
        }
        lines.join("\n")
    }

    /// Copy this sample into a new sample with the same sequence graphs.
    ///
    /// The destination name must not already exist. The returned sample is
    /// ready for explicit in-place edits on its sequence graphs. `new_name` is the name of the
    /// copy. The copy is
    /// recorded as its own operation, using ``message`` as the operation's
    /// commit message when given, or a generated description otherwise.
    #[pyo3(signature = (new_name, message=None))]
    fn copy(&self, new_name: String, message: Option<&str>) -> PyResult<PySample> {
        let context = &self.context;
        let sequence_graphs = self.load_sequence_graphs();
        if sequence_graphs.is_empty() {
            return Err(PyRuntimeError::new_err(
                "copy() requires a sample with sequence graphs",
            ));
        }
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
                        PyValueError::new_err(format!(
                            "sample '{new_name}' already exists; fetch it with repo.get_sample() or choose another name"
                        ))
                    }
                    other => PyRuntimeError::new_err(format!("cannot copy sample: {other}")),
                })?;
                for sequence_graph in &sequence_graphs {
                    BlockGroup::get_or_create_sample_block_groups(
                        conn,
                        &self.collection_name,
                        &new_name,
                        &sequence_graph.name,
                        vec![self.sample_name.clone()],
                    )
                    .map_err(block_group_err_to_pyerr)?;
                }
                SampleLineage::create(conn, &self.sample_name, &created_sample.name).map_err(
                    |error| PyRuntimeError::new_err(format!("cannot copy sample: {error}")),
                )?;

                let copied_sample = PySample::new(
                    self.collection_name.clone(),
                    new_name.clone(),
                    self.context.clone(),
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

#[gen_stub_pyclass]
#[pyclass(name = "SampleIterator", unsendable)]
pub struct PySampleIter {
    block_groups: Vec<PySequenceGraph>,
    index: usize,
}

#[gen_stub_pymethods]
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
