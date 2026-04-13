use std::{fs, path::PathBuf};

use r#gen::{
    core::HashId,
    get_connection,
    graphs::graph_search::{GenGraphMatcher, SeedIndex},
};
use gen_models::block_group::BlockGroup;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::{
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_and_display_widget},
};

/// Exposes a BlockGroup to Python.
#[pyclass]
#[derive(Clone)]
pub struct PyBlockGroup {
    pub id: HashId,
    #[pyo3(get)]
    pub collection_name: String,
    #[pyo3(get)]
    pub sample_name: String,
    #[pyo3(get)]
    pub name: String,
    /// Path to the SQLite database for the Repository that created this block group.
    pub db_path: Option<PathBuf>,
}

#[pymethods]
impl PyBlockGroup {
    #[new]
    pub fn new(id: HashId, collection_name: String, name: String, sample_name: String) -> Self {
        PyBlockGroup {
            id,
            collection_name,
            sample_name,
            name,
            db_path: None,
        }
    }

    #[getter]
    fn id(&self) -> PyHashId {
        PyHashId::new(self.id)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "BlockGroup({}, {}, {}, {})",
            self.id, self.collection_name, self.sample_name, self.name
        ))
    }

    fn __hash__(&self) -> PyResult<isize> {
        // Fold the raw bytes of the HashId into a single isize using a
        // polynomial rolling hash (same approach as PyHashId).
        let mut hash: isize = 0;
        for &b in &self.id.0 {
            hash = hash.wrapping_mul(31).wrapping_add(b as isize);
        }
        Ok(hash)
    }

    fn __eq__(&self, py: Python<'_>, other: PyObject) -> PyResult<bool> {
        // Try to extract PyBlockGroup from the PyObject
        if let Ok(other_bg) = other.extract::<PyRef<PyBlockGroup>>(py) {
            Ok(self.id == other_bg.id
                && self.collection_name == other_bg.collection_name
                && self.sample_name == other_bg.sample_name
                && self.name == other_bg.name)
        } else {
            // If other is not a PyBlockGroup, they're not equal
            Ok(false)
        }
    }

    /// Plot this block group's graph as an interactive Jupyter widget.
    ///
    /// Displays the widget immediately and returns it for further use.
    /// Outside of an IPython/Jupyter environment the display call is silently
    /// skipped and only the widget is returned.
    ///
    /// Raises ``RuntimeError`` if this block group was not created via a
    /// ``Repository`` (i.e. ``db_path`` is unset).
    ///
    /// Parameters
    /// ----------
    /// rows : int, optional
    ///     Initial viewport height in terminal rows.
    /// cols : int, optional
    ///     Initial viewport width in terminal columns.
    /// detail : {"normal", "full", "minimal"}, optional
    ///     Initial level of node detail.  ``"normal"`` (default) shows
    ///     truncated labels; ``"full"`` shows complete labels; ``"minimal"``
    ///     shows the smallest representation.
    #[pyo3(signature = (rows=None, cols=None, detail=None))]
    fn plot(
        slf: &Bound<'_, PyBlockGroup>,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
    ) -> PyResult<PyObject> {
        let py = slf.py();

        let (db_path, bg_id) = {
            let bg = slf.borrow();
            match bg.db_path.clone() {
                Some(p) => (p, bg.id),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "plot() requires a db_path; obtain BlockGroup via Repository",
                    ));
                }
            }
        };

        let conn = get_connection(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let graph = BlockGroup::get_graph(&conn, &bg_id);
        let mut ctrl = PyGraphController::new(db_path, graph);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_and_display_widget(py, ctrl, rows, cols)
    }

    /// IPython display hook — called by `display(block_group)` in Jupyter.
    fn _ipython_display_(slf: &Bound<'_, PyBlockGroup>) -> PyResult<()> {
        // plot() already calls IPython.display.display() internally; just
        // delegate and ignore errors (e.g. db_path unset, anywidget missing).
        let _ = slf.call_method0("plot");
        Ok(())
    }

    /// Build a junction-aware k-mer seed index for this block group.
    ///
    /// Saves the index to `.gen/search_index/{id}.bin` so that subsequent
    /// calls to `Repository.search()` load it automatically.
    ///
    /// Raises ``RuntimeError`` if this block group was not created via a
    /// ``Repository`` (i.e. ``db_path`` is unset).
    ///
    /// Parameters
    /// ----------
    /// k : int, optional
    ///     k-mer size. Defaults to 16.
    #[pyo3(signature = (k=16))]
    pub fn build_index(&self, k: usize) -> PyResult<()> {
        let db_path = self.db_path.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "build_index() requires a db_path; obtain BlockGroup via Repository",
            )
        })?;
        let gen_dir = db_path.parent().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot determine .gen directory from db_path")
        })?;
        let index_dir = gen_dir.join("search_index");
        fs::create_dir_all(&index_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index dir: {e}")))?;
        let conn = get_connection(db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let graph = BlockGroup::get_graph(&conn, &self.id);
        let matcher = GenGraphMatcher::new(&conn, graph);
        let index = SeedIndex::build(&matcher, k);
        let path = index_dir.join(format!("{}.bin", self.id));
        let bytes = postcard::to_allocvec(&index)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize index: {e}")))?;
        fs::write(&path, bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write index: {e}")))?;
        Ok(())
    }

    /// Clear the search index for this block group.
    ///
    /// Removes `.gen/search_index/{id}.bin` if it exists.
    ///
    /// Raises ``RuntimeError`` if this block group was not created via a
    /// ``Repository`` (i.e. ``db_path`` is unset).
    pub fn clear_index(&self) -> PyResult<()> {
        let db_path = self.db_path.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "clear_index() requires a db_path; obtain BlockGroup via Repository",
            )
        })?;
        let gen_dir = db_path.parent().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot determine .gen directory from db_path")
        })?;
        let path = gen_dir
            .join("search_index")
            .join(format!("{}.bin", self.id));
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete index: {e}")))?;
        }
        Ok(())
    }
}
