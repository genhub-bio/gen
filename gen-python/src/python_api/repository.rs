use std::{path::PathBuf, sync::Mutex};

use r#gen::{core::HashId, get_connection};
use gen_core::config::Workspace;
use gen_models::{block_group::BlockGroup, db::GraphConnection, node::Node, traits::Query};
use pyo3::{prelude::*, types::PyModule};

use super::{
    block_group::PyBlockGroup,
    factory::Factory,
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
}

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
