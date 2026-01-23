use std::{path::PathBuf, sync::Mutex};

use r#gen::{core::HashId, get_connection, views::block_layout::BaseLayout};
use gen_core::config::Workspace;
use gen_models::{block_group::BlockGroup, db::GraphConnection, node::Node, traits::Query};
use pyo3::{prelude::*, types::PyModule};

use super::{
    block_group::PyBlockGroup,
    factory::Factory,
    layouts::PyBaseLayout,
    node_key::PyNodeKey,
    utils::{path_to_py_path, py_query, sqlite_err_to_pyerr},
};

/// The main entry point for the gen Python module.
///
/// This class manages the database connection and provides methods for
/// querying and manipulating the database.
#[pyclass(name = "Repository")]
pub struct PyRepository {
    // We use custom getters, hence no #[pyo3(get)]
    pub gen_dir: PathBuf,
    pub db_path: PathBuf,
    pub conn: Mutex<Option<GraphConnection>>, // Private to Rust, not exposed to Python
    pub factory: Factory,                     // Embedded factory for BlockGroup transformations
}

// Regular Rust implementation outside of PyO3 exposure
impl PyRepository {
    // Private helper method that provides a connection to a closure
    // This pattern avoids exposing Rust-specific types like MutexGuard to Python
    // while still ensuring proper connection management
    pub fn with_connection<F, T>(&self, op: F) -> T
    where
        F: FnOnce(&GraphConnection) -> T,
    {
        let mut conn_guard = self.conn.lock().unwrap();
        if conn_guard.is_none() {
            *conn_guard = Some(get_connection(self.db_path.to_str().unwrap()).unwrap());
        }

        op(conn_guard.as_ref().unwrap())
    }
}

#[pymethods]
impl PyRepository {
    #[new]
    #[pyo3(signature = (path = Option::<String>::None))]
    fn new(_py: Python, path: Option<String>) -> PyResult<Self> {
        // PathBuf instead of Path to avoid borrowing issues
        let gen_dir: PathBuf = match path {
            Some(path_str) => PathBuf::from(path_str),
            None => Workspace::from_current_dir().ensure_gen_dir(),
        };

        let db_path = gen_dir.join("default.db");

        // Initialize with no connection - it will be created lazily
        // We do need to use a Mutex for memory safety
        Ok(PyRepository {
            gen_dir,
            db_path,
            conn: Mutex::new(None),
            factory: Factory::new(), // Initialize the factory
        })
    }

    #[getter]
    fn get_gen_dir(&self, py: Python) -> PyResult<PyObject> {
        path_to_py_path(py, &self.gen_dir)
    }

    #[getter]
    fn get_db_path(&self, py: Python) -> PyResult<PyObject> {
        path_to_py_path(py, &self.db_path)
    }

    // Database operations directly on PyRepository
    fn execute(&self, query: &str) -> PyResult<()> {
        self.with_connection(|conn| {
            conn.execute(query, []).map_err(sqlite_err_to_pyerr)?;
            Ok(())
        })
    }

    fn query(&self, query: &str) -> PyResult<Vec<Vec<PyObject>>> {
        self.with_connection(|conn| py_query(conn, query))
    }

    /// Retrieves a BlockGroup by its ID.
    ///
    /// Args:
    ///     id: The ID of the BlockGroup to retrieve
    ///
    /// Returns:
    ///     A PyBlockGroup instance representing the requested BlockGroup
    fn get_block_group_by_id(&self, id: &HashId) -> PyResult<PyBlockGroup> {
        self.with_connection(|conn| {
            let block_group = BlockGroup::get_by_id(conn, id);

            Ok(PyBlockGroup {
                id: block_group.id,
                collection_name: block_group.collection_name,
                sample_name: block_group.sample_name,
                name: block_group.name,
            })
        })
    }

    /// Retrieves all BlockGroups.
    ///
    /// Returns:
    ///     A vector of PyBlockGroup instances
    fn get_block_groups(&self) -> PyResult<Vec<PyBlockGroup>> {
        self.with_connection(|conn| {
            let block_groups = BlockGroup::all(conn);

            let result = block_groups
                .into_iter()
                .map(|bg| PyBlockGroup {
                    id: bg.id,
                    collection_name: bg.collection_name,
                    sample_name: bg.sample_name,
                    name: bg.name,
                })
                .collect();

            Ok(result)
        })
    }

    /// Retrieves all BlockGroups belonging to a specific collection.
    ///
    /// Args:
    ///     collection_name: The name of the collection to retrieve BlockGroups from
    ///
    /// Returns:
    ///     A vector of PyBlockGroup instances
    fn get_block_groups_by_collection(&self, collection_name: &str) -> PyResult<Vec<PyBlockGroup>> {
        self.with_connection(|conn| {
            let block_groups = BlockGroup::query(
                conn,
                "SELECT * FROM block_groups WHERE collection_name = ?1",
                rusqlite::params![collection_name],
            );

            let result = block_groups
                .into_iter()
                .map(|bg| PyBlockGroup {
                    id: bg.id,
                    collection_name: bg.collection_name,
                    sample_name: bg.sample_name,
                    name: bg.name,
                })
                .collect();

            Ok(result)
        })
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
        Python::with_gil(|py| {
            // Check if rustworkx is installed
            match PyModule::import(py, "rustworkx") {
                Ok(_) => {
                    // rustworkx is available, proceed with the conversion
                    self.with_connection(|conn| self.factory.to_rustworkx(conn, &block_group.id))
                }
                Err(_) => {
                    // rustworkx is not available, return a helpful error message
                    Err(pyo3::exceptions::PyModuleNotFoundError::new_err(
                        "The 'rustworkx' module is not installed. Please install it using 'pip install rustworkx' to use this functionality.",
                    ))
                }
            }
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
        self.with_connection(|conn| self.factory.to_dict(conn, &block_group.id))
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
        Python::with_gil(|py| {
            // Check if networkx is installed
            match PyModule::import(py, "networkx") {
                Ok(_) => {
                    // networkx is available, proceed with the conversion
                    self.with_connection(|conn| self.factory.to_networkx(conn, &block_group.id))
                }
                Err(_) => {
                    // networkx is not available, return a helpful error message
                    Err(pyo3::exceptions::PyModuleNotFoundError::new_err(
                        "The 'networkx' module is not installed. Please install it using 'pip install networkx' to use this functionality.",
                    ))
                }
            }
        })
    }

    // TODO: implement operation tracking and wire into CLI simple sequence entry code
    /// Creates a new BlockGroup.
    ///
    /// Args:
    ///     name: The name of the BlockGroup
    ///     collection_name: The name of the collection
    ///     sample_name: Optional name of the sample
    ///
    /// Returns:
    ///     A PyBlockGroup instance
    fn create_block_group(
        &self,
        name: String,
        collection_name: String,
        sample_name: Option<String>,
    ) -> PyResult<PyBlockGroup> {
        self.with_connection(|conn| {
            let block_group = BlockGroup::create(
                conn,
                &collection_name,
                sample_name.as_deref(), // Option<String> to Option<&str>
                &name,
            );

            Ok(PyBlockGroup {
                id: block_group.id,
                collection_name: block_group.collection_name,
                sample_name: block_group.sample_name,
                name: block_group.name,
            })
        })
    }

    /// Creates a BaseLayout from a BlockGroup
    ///
    /// Args:
    ///     block_group: The BlockGroup to create a layout for
    ///
    /// Returns:
    ///     A PyBaseLayout instance
    fn create_base_layout(&self, block_group: &PyBlockGroup) -> PyResult<PyBaseLayout> {
        self.with_connection(|conn| {
            let graph = BlockGroup::get_graph(conn, &block_group.id);
            let block_layout = BaseLayout::new(&graph);

            Ok(PyBaseLayout {
                layout: block_layout,
            })
        })
    }

    /// Gets the sequence for a block specified by a NodeKey
    ///
    /// Args:
    ///     node_key: The NodeKey containing node_id, sequence_start, and sequence_end
    ///
    /// Returns:
    ///     A string containing the sequence for the specified block
    fn get_block_sequence(&self, node_key: &PyNodeKey) -> PyResult<String> {
        self.with_connection(|conn| {
            let sequences_by_node_id = Node::get_sequences_by_node_ids(conn, &[node_key.node_id]);
            let sequence = sequences_by_node_id.get(&node_key.node_id).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Node with id {:?} not found",
                    node_key.node_id
                ))
            })?;
            Ok(sequence.get_sequence(node_key.sequence_start, node_key.sequence_end))
        })
    }

    /// Creates a context manager for grouping operations into a single transaction
    ///
    /// Args:
    ///     title: The title/summary for the operation group
    ///
    /// Returns:
    ///     An OperationContext that can be used as a context manager
    ///
    /// Example:
    ///     ```python
    ///     repo = Repository()
    ///     with repo.commit('Import multiple FASTA files') as ctx:
    ///         ctx.import_fasta('file1.fasta')
    ///         ctx.import_fasta('file2.fasta')
    ///         ctx.append_message('Additional notes about the import')
    ///     ```
    fn commit(&self, title: String) -> PyResult<PyOperationContext> {
        Ok(PyOperationContext::new(self, title))
    }

    /// Imports a FASTA file with automatic transaction management
    ///
    /// Args:
    ///     filename: Path to the FASTA file
    ///     name: Optional collection name
    ///     sample: Optional sample name
    ///     shallow: Whether to do a shallow import
    ///
    /// Returns:
    ///     Success message
    fn import_fasta(
        &self,
        py: Python,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
        shallow: bool,
    ) -> PyResult<String> {
        // Get or create DbContext from repository
        let workspace = gen_core::config::Workspace::from_current_dir();
        let gen_dir = workspace.ensure_gen_dir();
        let operations_path = gen_dir.join("gen.db");
        let operations_conn =
            r#gen::get_operation_connection(Some(operations_path)).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to open operations database: {err}"
                ))
            })?;

        let db_path = gen_dir.join("default.db");
        let graph_conn = r#gen::get_connection(db_path.to_str().unwrap()).map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to open database: {err}"))
        })?;

        let db_context = gen_models::db::DbContext::new(workspace, graph_conn, operations_conn);
        let py_db_context = crate::PyDbContext(db_context);

        // Call the existing import function
        // We need to create a PyRef from the PyDbContext
        let py_db_obj = Py::new(py, py_db_context)?;
        let py_db_bound = py_db_obj.bind(py);
        let py_db_ref = py_db_bound.borrow();
        crate::imports::import_fasta(py_db_ref, filename, name, sample, shallow)
    }

    /// Updates with a FASTA file with automatic transaction management
    ///
    /// Args:
    ///     filename: Path to the FASTA file
    ///     name: Optional collection name
    ///     sample: Optional sample name
    ///     new_sample: Name for the new sample
    ///     region_name: Name of the region
    ///     start: Start coordinate
    ///     end: End coordinate
    ///
    /// Returns:
    ///     Success message
    fn update_with_fasta(
        &self,
        py: Python,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
        new_sample: String,
        region_name: String,
        start: i64,
        end: i64,
    ) -> PyResult<String> {
        // Similar to import_fasta - create DbContext and call update function
        let workspace = gen_core::config::Workspace::from_current_dir();
        let gen_dir = workspace.ensure_gen_dir();
        let operations_path = gen_dir.join("gen.db");
        let operations_conn =
            r#gen::get_operation_connection(Some(operations_path)).map_err(|err| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to open operations database: {err}"
                ))
            })?;

        let db_path = gen_dir.join("default.db");
        let graph_conn = r#gen::get_connection(db_path.to_str().unwrap()).map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to open database: {err}"))
        })?;

        let db_context = gen_models::db::DbContext::new(workspace, graph_conn, operations_conn);
        let py_db_context = crate::PyDbContext(db_context);

        // Call the existing update function
        let py_db_obj = Py::new(py, py_db_context)?;
        let py_db_bound = py_db_obj.bind(py);
        let py_db_ref = py_db_bound.borrow();
        crate::updates::update_with_fasta(
            py_db_ref,
            filename,
            name,
            sample,
            new_sample,
            region_name,
            start,
            end,
        )
    }
}

/// Context manager for grouping multiple operations into a single transaction
#[pyclass(name = "OperationContext")]
pub struct PyOperationContext {
    gen_dir: PathBuf,
    db_path: PathBuf,
    title: String,
    messages: Vec<String>,
    session_started: bool,
}

#[pymethods]
impl PyOperationContext {
    #[new]
    fn new(repository: &PyRepository, title: String) -> Self {
        Self {
            gen_dir: repository.gen_dir.clone(),
            db_path: repository.db_path.clone(),
            title,
            messages: Vec::new(),
            session_started: false,
        }
    }

    fn __enter__(&mut self, py: Python) -> PyResult<PyOperationProxy> {
        // Start the operation session
        self.session_started = true;
        // Create and return the proxy object
        let context_py = Py::new(
            py,
            PyOperationContext {
                gen_dir: self.gen_dir.clone(),
                db_path: self.db_path.clone(),
                title: self.title.clone(),
                messages: self.messages.clone(),
                session_started: self.session_started,
            },
        )?;
        Ok(PyOperationProxy::new(context_py))
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> PyResult<bool> {
        // If session was started, end the operation
        if self.session_started {
            // End operation with combined message
            let summary = if self.messages.is_empty() {
                self.title.clone()
            } else {
                format!("{}\n{}", self.title, self.messages.join("\n"))
            };

            // For now, we'll implement this as a placeholder
            // The actual implementation will need to call end_operation
            println!("Ending operation with summary: {}", summary);
        }

        // Return false to propagate exceptions
        Ok(false)
    }

    fn append_message(&mut self, message: String) {
        self.messages.push(message);
    }

    fn get_title(&self) -> String {
        self.title.clone()
    }

    fn get_messages(&self) -> Vec<String> {
        self.messages.clone()
    }
}

/// Proxy object returned by OperationContext.__enter__ that wraps method calls
#[pyclass(name = "OperationProxy")]
pub struct PyOperationProxy {
    context: Py<PyOperationContext>,
}

#[pymethods]
impl PyOperationProxy {
    #[new]
    fn new(context: Py<PyOperationContext>) -> Self {
        Self { context }
    }

    fn append_message(&self, py: Python, message: String) -> PyResult<()> {
        self.context.borrow_mut(py).append_message(message);
        Ok(())
    }

    fn get_title(&self, py: Python) -> PyResult<String> {
        Ok(self.context.borrow(py).get_title())
    }

    fn get_messages(&self, py: Python) -> PyResult<Vec<String>> {
        Ok(self.context.borrow(py).get_messages())
    }

    /// Imports a FASTA file within the operation context
    fn import_fasta(
        &self,
        py: Python,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
        shallow: bool,
    ) -> PyResult<String> {
        let context = self.context.borrow(py);
        let mut repo = PyRepository::new(py, Some(context.gen_dir.to_string_lossy().to_string()))?;
        repo.db_path = context.db_path.clone();
        repo.import_fasta(py, filename, name, sample, shallow)
    }

    /// Updates with a FASTA file within the operation context
    fn update_with_fasta(
        &self,
        py: Python,
        filename: String,
        name: Option<String>,
        sample: Option<String>,
        new_sample: String,
        region_name: String,
        start: i64,
        end: i64,
    ) -> PyResult<String> {
        let context = self.context.borrow(py);
        let mut repo = PyRepository::new(py, Some(context.gen_dir.to_string_lossy().to_string()))?;
        repo.db_path = context.db_path.clone();
        repo.update_with_fasta(
            py,
            filename,
            name,
            sample,
            new_sample,
            region_name,
            start,
            end,
        )
    }
}

/// Creates a context manager for grouping operations into a single transaction
///
/// Args:
///     title: The title/summary for the operation group
///
/// Returns:
///     An OperationContext that can be used as a context manager
///
/// Example:
///     ```python
///     repo = Repository()
///     with repo.commit('Import multiple FASTA files') as ctx:
///         ctx.import_fasta('file1.fasta')
///         ctx.import_fasta('file2.fasta')
///         ctx.append_message('Additional notes about the import')
///     ```

#[cfg(test)]
mod python_tests {
    use r#gen::test_helpers::setup_gen;
    use pyo3::{prelude::*, py_run};

    use crate::python_api::repository::PyRepository;

    #[test]
    fn test_repository_creation() {
        setup_gen();
        // Run python code to confirm that the repository class is available, with the repository class passed in from Rust
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let repository = py.get_type::<PyRepository>();
            py_run!(
                py,
                repository,
                r#"
                # Create a repository in the present working directory
                repo = repository()

                # Test that the repository was created successfully
                assert hasattr(repo, "gen_dir")
                assert hasattr(repo, "db_path")
            "#
            );
        });
    }
}
