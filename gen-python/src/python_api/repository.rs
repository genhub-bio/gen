use r#gen::{get_connection, get_operation_connection};
use gen_core::config::Workspace;
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    db::DbContext,
    node::Node,
    traits::Query,
};
use pyo3::{prelude::*, types::PyModule};

use super::{
    block_group::PyBlockGroup,
    factory::Factory,
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_and_display_widget},
    node_key::PyNodeKey,
    utils::{block_group_err_to_pyerr, path_to_py_path, py_query, sqlite_err_to_pyerr},
};

/// The main entry point for the gen Python module.
///
/// This class manages the database connection and provides methods for
/// querying and manipulating the database.
// unsendable because DbContext contains Rc (rusqlite::Connection is !Sync)
#[pyclass(name = "Repository", unsendable)]
pub struct PyRepository {
    pub context: DbContext,
    pub factory: Factory,
}

impl PyRepository {
    /// Converts a model [`BlockGroup`] into a [`PyBlockGroup`], cloning the
    /// repository's [`DbContext`] into it so the block group can issue its own
    /// queries without holding a reference back to the repository.
    fn into_py_block_group(&self, bg: BlockGroup) -> PyBlockGroup {
        PyBlockGroup {
            id: bg.id,
            collection_name: bg.collection_name,
            sample_name: bg.sample_name,
            name: bg.name,
            context: Some(self.context.clone()),
        }
    }
}

#[pymethods]
impl PyRepository {
    #[new]
    #[pyo3(signature = (path = Option::<String>::None))]
    fn new(path: Option<String>) -> PyResult<Self> {
        let workspace = match path {
            Some(p) => Workspace::new(p),
            None => Workspace::from_current_dir(),
        };
        let gen_dir = workspace.ensure_gen_dir();
        let db_path = gen_dir.join("default.db");
        let ops_path = gen_dir.join("gen.db");

        let graph_conn = get_connection(db_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let ops_conn = get_operation_connection(Some(ops_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyRepository {
            context: DbContext::new(workspace, graph_conn, ops_conn),
            factory: Factory::new(),
        })
    }

    #[getter]
    fn get_gen_dir(&self, py: Python) -> PyResult<PyObject> {
        path_to_py_path(py, &self.context.workspace().ensure_gen_dir())
    }

    #[getter]
    fn get_db_path(&self, py: Python) -> PyResult<PyObject> {
        path_to_py_path(
            py,
            &self.context.workspace().ensure_gen_dir().join("default.db"),
        )
    }

    fn execute(&self, query: &str) -> PyResult<()> {
        self.context
            .graph()
            .conn()
            .execute(query, [])
            .map_err(sqlite_err_to_pyerr)?;
        Ok(())
    }

    fn query(&self, query: &str) -> PyResult<Vec<Vec<PyObject>>> {
        py_query(self.context.graph().conn(), query)
    }

    /// Retrieves a BlockGroup by its ID.
    ///
    /// Args:
    ///     id: The ID of the BlockGroup to retrieve
    ///
    /// Returns:
    ///     A PyBlockGroup instance representing the requested BlockGroup
    fn get_block_group(
        &self,
        name: &str,
        sample_name: &str,
        collection_name: &str,
    ) -> PyResult<PyBlockGroup> {
        let id = BlockGroup::get_id(collection_name, sample_name, name, None);
        let conn = self.context.graph().conn();
        let block_group = BlockGroup::get_by_id(conn, &id).map_err(block_group_err_to_pyerr)?;
        Ok(self.into_py_block_group(block_group))
    }

    fn get_block_group_by_id(&self, id: &PyHashId) -> PyResult<PyBlockGroup> {
        let conn = self.context.graph().conn();
        let block_group = match BlockGroup::get_by_id(conn, &id.hash_id) {
            Ok(bg) => bg,
            Err(r#gen::models::block_group::BlockGroupError::QueryError(_)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "BlockGroup with id {} not found",
                    id.hash_id
                )));
            }
            Err(e) => return Err(block_group_err_to_pyerr(e)),
        };
        Ok(self.into_py_block_group(block_group))
    }

    /// Retrieves all BlockGroups.
    ///
    /// Returns:
    ///     A vector of PyBlockGroup instances
    fn get_block_groups(&self) -> PyResult<Vec<PyBlockGroup>> {
        let conn = self.context.graph().conn();
        Ok(BlockGroup::all(conn)
            .into_iter()
            .map(|bg| self.into_py_block_group(bg))
            .collect())
    }

    /// Retrieves all BlockGroups belonging to a specific collection.
    ///
    /// Args:
    ///     collection_name: The name of the collection to retrieve BlockGroups from
    ///
    /// Returns:
    ///     A vector of PyBlockGroup instances
    fn get_block_groups_by_collection(&self, collection_name: &str) -> PyResult<Vec<PyBlockGroup>> {
        let conn = self.context.graph().conn();
        Ok(BlockGroup::query(
            conn,
            "SELECT * FROM block_groups WHERE collection_name = ?1",
            rusqlite::params![collection_name],
        )
        .into_iter()
        .map(|bg| self.into_py_block_group(bg))
        .collect())
    }

    // Factory methods:
    // BlockGroup objects themselves don't hold all their data, so we need to
    // query the database again to transform into to a different representation.
    // These methods use an embedded Factory to handle the transformation while
    // the Repository manages the database connection.

    /// Converts a BlockGroup to a rustworkx graph representation
    ///
    /// Args:
    ///     block_group: The BlockGroup instance to convert
    ///
    /// Returns:
    ///     A rustworkx PyDiGraph representing the BlockGroup
    ///
    /// Raises:
    ///     PyModuleNotFoundError: If rustworkx is not installed
    fn block_group_to_rustworkx(&self, block_group: &PyBlockGroup) -> PyResult<PyObject> {
        Python::with_gil(|py| match PyModule::import(py, "rustworkx") {
            Ok(_) => self
                .factory
                .to_rustworkx(self.context.graph().conn(), &block_group.id),
            Err(_) => Err(pyo3::exceptions::PyModuleNotFoundError::new_err(
                "The 'rustworkx' module is not installed. Please install it using 'pip install rustworkx' to use this functionality.",
            )),
        })
    }

    /// Converts a BlockGroup to a dictionary representation
    ///
    /// Args:
    ///     block_group: The BlockGroup instance to convert
    ///
    /// Returns:
    ///     A Python dictionary containing the graph representation
    fn block_group_to_dict(&self, block_group: &PyBlockGroup) -> PyResult<PyObject> {
        self.factory
            .to_dict(self.context.graph().conn(), &block_group.id)
    }

    /// Converts a BlockGroup to a NetworkX graph representation
    ///
    /// Args:
    ///     block_group: The BlockGroup instance to convert
    ///
    /// Returns:
    ///     A NetworkX DiGraph representing the BlockGroup
    ///
    /// Raises:
    ///     PyModuleNotFoundError: If networkx is not installed
    fn block_group_to_networkx(&self, block_group: &PyBlockGroup) -> PyResult<PyObject> {
        Python::with_gil(|py| match PyModule::import(py, "networkx") {
            Ok(_) => self
                .factory
                .to_networkx(self.context.graph().conn(), &block_group.id),
            Err(_) => Err(pyo3::exceptions::PyModuleNotFoundError::new_err(
                "The 'networkx' module is not installed. Please install it using 'pip install networkx' to use this functionality.",
            )),
        })
    }

    // TODO: implement operation tracking and wire into CLI simple sequence entry code
    /// Creates a new BlockGroup.
    ///
    /// Args:
    ///     name: The name of the BlockGroup
    ///     collection_name: The name of the collection
    ///     sample_name: Name of the sample
    ///
    /// Returns:
    ///     A PyBlockGroup instance
    fn create_block_group(
        &self,
        name: String,
        collection_name: String,
        sample_name: String,
    ) -> PyResult<PyBlockGroup> {
        let conn = self.context.graph().conn();
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &collection_name,
                sample_name: &sample_name,
                name: &name,
                ..Default::default()
            },
        )
        .map_err(block_group_err_to_pyerr)?;
        Ok(self.into_py_block_group(block_group))
    }

    /// Plot a BlockGroup's graph as an interactive Jupyter widget.
    ///
    /// Displays the widget immediately and returns it for further use.
    /// Outside of an IPython/Jupyter environment the display call is silently
    /// skipped and only the widget is returned.
    ///
    /// # Example
    /// ```python
    /// repo = gen.Repository()
    /// bg   = repo.get_block_groups()[0]
    /// w    = repo.plot(bg)   # widget appears immediately
    /// w.zoom_in()            # further interaction via the handle
    /// ```
    ///
    /// Parameters
    /// ----------
    /// block_group : PyBlockGroup
    ///     The block group to visualise.
    /// rows : int, optional
    ///     Initial viewport height in terminal rows.
    /// cols : int, optional
    ///     Initial viewport width in terminal columns.
    /// detail : {"normal", "full", "minimal"}, optional
    ///     Initial level of node detail.  ``"normal"`` (default) shows
    ///     truncated labels; ``"full"`` shows complete labels; ``"minimal"``
    ///     shows the smallest representation.
    #[pyo3(signature = (block_group, rows=None, cols=None, detail=None))]
    fn plot(
        &self,
        py: Python<'_>,
        block_group: &PyBlockGroup,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
    ) -> PyResult<PyObject> {
        let conn = self.context.graph().conn();
        let graph = BlockGroup::get_graph(conn, &block_group.id);
        let db_path = self.context.workspace().ensure_gen_dir().join("default.db");
        let mut ctrl = PyGraphController::new(db_path, graph);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_and_display_widget(py, ctrl, rows, cols)
    }

    /// Gets the sequence for a block specified by a NodeKey
    ///
    /// Args:
    ///     node_key: The NodeKey containing node_id, sequence_start, and sequence_end
    ///
    /// Returns:
    ///     A string containing the sequence for the specified block
    fn get_block_sequence(&self, node_key: &PyNodeKey) -> PyResult<String> {
        let conn = self.context.graph().conn();
        let sequences_by_node_id = Node::get_sequences_by_node_ids(conn, &[node_key.node_id]);
        let sequence = sequences_by_node_id.get(&node_key.node_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node with id {:?} not found",
                node_key.node_id
            ))
        })?;
        Ok(sequence
            .get_sequence(node_key.sequence_start, node_key.sequence_end)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?)
    }
}

#[cfg(test)]
mod python_tests {
    use r#gen::{
        core::HashId,
        test_helpers::{create_bg, setup_gen, setup_gen_on_disk},
    };
    use pyo3::{prelude::*, py_run};

    use crate::python_api::{factory::Factory, hash_id::PyHashId, repository::PyRepository};

    fn make_repo_with_data(py: Python) -> Py<PyRepository> {
        use gen_models::collection::Collection;
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        Collection::create(conn, "col-a").unwrap();
        Collection::create(conn, "col-b").unwrap();
        create_bg(conn, "col-a", "s1", "bg1");
        create_bg(conn, "col-b", "s1", "bg2");
        Py::new(
            py,
            PyRepository {
                context,
                factory: Factory::new(),
            },
        )
        .unwrap()
    }

    #[test]
    fn test_repository_creation() {
        setup_gen();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let repository = py.get_type::<PyRepository>();
            py_run!(
                py,
                repository,
                r#"
                repo = repository()
                assert hasattr(repo, "gen_dir")
                assert hasattr(repo, "db_path")
            "#
            );
        });
    }

    #[test]
    fn test_get_block_groups() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let repo = make_repo_with_data(py);
            py_run!(
                py,
                repo,
                r#"
                bgs = repo.get_block_groups()
                assert len(bgs) == 2
                names = {bg.name for bg in bgs}
                assert names == {"bg1", "bg2"}
            "#
            );
        });
    }

    #[test]
    fn test_get_block_group_by_id_not_found() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let repo = make_repo_with_data(py);
            let bad_id = Py::new(py, PyHashId::new(HashId([0u8; 32]))).unwrap();
            py_run!(
                py,
                repo bad_id,
                r#"
                try:
                    repo.get_block_group_by_id(bad_id)
                    assert False, "expected ValueError"
                except ValueError:
                    pass
            "#
            );
        });
    }
}
