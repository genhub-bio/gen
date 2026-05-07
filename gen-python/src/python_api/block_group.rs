use r#gen::core::HashId;
use gen_models::{block_group::BlockGroup, db::DbContext};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::{
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_and_display_widget},
};

/// A block group returned by a ``Repository``.
///
/// ``BlockGroup`` objects cannot be shared across threads because they hold a
/// reference to the database connection of the ``Repository`` that created them.
///
/// To use a block group's identity in another thread, capture its ``id``,
/// open a fresh ``Repository`` in that thread, and look it up by id::
///
///     bg_id = bg.id
///     path  = str(repo.db_path)
///
///     def worker():
///         r = gen.Repository(path)
///         bg = r.get_block_group_by_id(bg_id)
///         ...
// unsendable because DbContext contains Rc (rusqlite::Connection is !Sync)
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct PyBlockGroup {
    pub id: HashId,
    #[pyo3(get)]
    pub collection_name: String,
    #[pyo3(get)]
    pub sample_name: String,
    #[pyo3(get)]
    pub name: String,
    pub context: Option<DbContext>,
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
            context: None,
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
    /// ``Repository``.
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

        let (context, bg_id) = {
            let bg = slf.borrow();
            match bg.context.clone() {
                Some(ctx) => (ctx, bg.id),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "plot() requires a Repository context; obtain BlockGroups via Repository by query or id.",
                    ));
                }
            }
        };

        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(conn, &bg_id);
        let db_path = context
            .gen_db_path()
            .map(|p| p.with_file_name("default.db"))
            .unwrap_or_default();
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
}
