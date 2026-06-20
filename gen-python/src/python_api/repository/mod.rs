use std::path::PathBuf;

use r#gen::{get_connection, get_operation_connection, track_database};
use gen_core::config::Workspace;
use gen_models::{
    block_group::BlockGroup, collection::Collection, db::DbContext, node::Node,
    operations::Defaults, traits::Query,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::{
    block_group::PySequenceGraph,
    graph_node::PyGraphNode,
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_widget},
    utils::{block_group_err_to_pyerr, path_to_py_path, py_query, sqlite_err_to_pyerr},
};

pub mod exports;
pub mod graph_ops;
pub mod imports;
pub mod search;
pub mod updates;

fn tx_begin(context: &DbContext) -> PyResult<()> {
    let conn = context.graph().conn();
    let op_conn = context.operations().conn();
    track_database(conn, op_conn)
        .map_err(|e| PyRuntimeError::new_err(format!("Error tracking database: {e}")))?;
    conn.execute("BEGIN TRANSACTION", [])
        .map_err(sqlite_err_to_pyerr)?;
    op_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(sqlite_err_to_pyerr)?;
    Ok(())
}

fn tx_commit(context: &DbContext) -> PyResult<()> {
    context
        .graph()
        .conn()
        .execute("END TRANSACTION", [])
        .map_err(sqlite_err_to_pyerr)?;
    context
        .operations()
        .conn()
        .execute("END TRANSACTION", [])
        .map_err(sqlite_err_to_pyerr)?;
    Ok(())
}

fn tx_rollback(context: &DbContext) {
    context.graph().conn().execute("ROLLBACK", []).ok();
    context.operations().conn().execute("ROLLBACK", []).ok();
}

pub(crate) fn run_write<F, T>(context: &DbContext, managed: bool, op: F) -> PyResult<T>
where
    F: FnOnce(&DbContext) -> PyResult<T>,
{
    if managed {
        tx_begin(context)?;
    }
    match op(context) {
        Ok(val) => {
            if managed {
                tx_commit(context)?;
            }
            Ok(val)
        }
        Err(err) => {
            if managed {
                tx_rollback(context);
            }
            Err(err)
        }
    }
}

/// The main entry point for the gen Python module.
///
/// This class manages the database connection and provides methods for
/// querying and manipulating the database.
#[pyclass(name = "Repository", unsendable)]
pub struct PyRepository {
    pub context: DbContext,
    pub in_transaction: bool,
}

impl PyRepository {
    pub(crate) fn get_default_collection(&self) -> String {
        Defaults::get(self.context.operations().conn())
            .and_then(|d| d.collection_name)
            .unwrap_or_else(|| "default".to_string())
    }

    pub(crate) fn to_py_block_group(&self, bg: BlockGroup) -> PySequenceGraph {
        PySequenceGraph {
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
            Some(path_str) => Workspace::new(path_str),
            None => Workspace::from_current_dir(),
        };

        let gen_dir = workspace.ensure_gen_dir();
        let ops_path = gen_dir.join("gen.db");
        let ops_conn = get_operation_connection(Some(ops_path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let db_path = Defaults::get(&ops_conn)
            .and_then(|d| d.db_name)
            .map(PathBuf::from)
            .unwrap_or_else(|| gen_dir.join("default.db"));

        let graph_conn = get_connection(db_path.clone()).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "Failed to open database '{}': {err}",
                db_path.display()
            ))
        })?;

        Ok(PyRepository {
            context: DbContext::new(workspace, graph_conn, ops_conn),
            in_transaction: false,
        })
    }

    #[getter]
    fn get_gen_dir(&self, py: Python<'_>) -> PyResult<PyObject> {
        path_to_py_path(py, &self.context.workspace().ensure_gen_dir())
    }

    #[getter]
    fn get_db_path(&self, py: Python<'_>) -> PyResult<PyObject> {
        let defaults = Defaults::get(self.context.operations().conn());
        let path = defaults
            .and_then(|d| d.db_name)
            .map(PathBuf::from)
            .unwrap_or_else(|| self.context.workspace().ensure_gen_dir().join("default.db"));
        path_to_py_path(py, &path)
    }

    // -------------------------------------------------------------------------
    // Transaction context manager
    // -------------------------------------------------------------------------

    /// Returns self so that Python's `with` statement calls `__enter__`/`__exit__`
    /// on this repository, batching multiple operations into one transaction.
    ///
    /// Example:
    ///     with repo.transaction():
    ///         repo.import_fasta("reference.fasta")
    ///         repo.import_gfa("graph.gfa")
    fn transaction(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<()> {
        tx_begin(&slf.context)?;
        slf.in_transaction = true;
        Ok(())
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        slf.in_transaction = false;
        if exc_type.is_some() {
            tx_rollback(&slf.context);
        } else {
            tx_commit(&slf.context)?;
        }
        Ok(false)
    }

    // -------------------------------------------------------------------------
    // Raw database access
    // -------------------------------------------------------------------------

    fn execute(&self, query: &str) -> PyResult<()> {
        self.context
            .graph()
            .conn()
            .execute(query, [])
            .map_err(sqlite_err_to_pyerr)?;
        Ok(())
    }

    fn query(&self, py: Python<'_>, query: &str) -> PyResult<Vec<Vec<PyObject>>> {
        py_query(py, self.context.graph().conn(), query)
    }

    // -------------------------------------------------------------------------
    // SequenceGraph queries
    // -------------------------------------------------------------------------

    fn get_sequence_graph_by_id(&self, id: &PyHashId) -> PyResult<PySequenceGraph> {
        let conn = self.context.graph().conn();
        let block_group =
            BlockGroup::get_by_id(conn, &id.hash_id).map_err(block_group_err_to_pyerr)?;
        Ok(self.to_py_block_group(block_group))
    }

    fn get_sequence_graphs(&self) -> PyResult<Vec<PySequenceGraph>> {
        let conn = self.context.graph().conn();
        Ok(BlockGroup::all(conn)
            .into_iter()
            .map(|bg| self.to_py_block_group(bg))
            .collect())
    }

    fn get_sequence_graphs_by_collection(
        &self,
        collection_name: &str,
    ) -> PyResult<Vec<PySequenceGraph>> {
        let conn = self.context.graph().conn();
        Ok(Collection::get_block_groups(conn, collection_name)
            .into_iter()
            .map(|bg| self.to_py_block_group(bg))
            .collect())
    }

    // -------------------------------------------------------------------------
    // Plot
    // -------------------------------------------------------------------------

    #[pyo3(signature = (sequence_graph, rows=None, cols=None, detail=None))]
    fn plot(
        &self,
        py: Python<'_>,
        sequence_graph: &PySequenceGraph,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
    ) -> PyResult<PyObject> {
        let graph_conn = self.context.graph().conn();
        let db_path = graph_conn
            .path()
            .map(std::path::PathBuf::from)
            .ok_or_else(|| PyRuntimeError::new_err("graph DB has no file path"))?;
        let graph = BlockGroup::get_graph(graph_conn, &sequence_graph.id)
            .map_err(block_group_err_to_pyerr)?;
        let mut ctrl = PyGraphController::new(db_path, graph);
        ctrl.block_group_id = Some(sequence_graph.id);
        ctrl.auto_load_annotation_groups(graph_conn);
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_widget(py, ctrl, rows, cols)
    }

    fn get_node_sequence(&self, node_key: &PyGraphNode) -> PyResult<String> {
        let sequences_by_node_id =
            Node::get_sequences_by_node_ids(self.context.graph().conn(), &[node_key.node_id]);
        let sequence = sequences_by_node_id.get(&node_key.node_id).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Node with id {:?} not found",
                node_key.node_id
            ))
        })?;
        sequence
            .get_sequence(node_key.sequence_start, node_key.sequence_end)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
    }
}

#[cfg(test)]
mod python_tests {
    use std::fs;

    use r#gen::test_helpers::setup_gen_on_disk;
    use pyo3::{PyTypeInfo, prelude::*, py_run};
    use tempfile::tempdir;

    use crate::python_api::repository::PyRepository;

    fn make_repo(py: Python<'_>) -> Py<PyRepository> {
        let ctx = setup_gen_on_disk();
        Py::new(
            py,
            PyRepository {
                context: ctx,
                in_transaction: false,
            },
        )
        .unwrap()
    }

    fn write_fasta(
        dir: &tempfile::TempDir,
        name: &str,
        seq_name: &str,
        sequence: &str,
    ) -> std::path::PathBuf {
        let path = dir.path().join(name);
        fs::write(&path, format!(">{seq_name}\n{sequence}\n")).unwrap();
        path
    }

    #[test]
    fn test_repository_creation() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let tmp_dir = tempdir().unwrap();
            let path = tmp_dir.path().to_str().unwrap().to_string();
            let repository = PyRepository::type_object(py);
            py_run!(
                py,
                repository,
                &format!(
                    r#"
                    repo = repository("{path}")
                    assert hasattr(repo, "gen_dir")
                    assert hasattr(repo, "db_path")
                    "#
                )
            );
        });
    }

    #[test]
    fn test_import_fasta_creates_block_group() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_sequence_graphs().unwrap();
            assert_eq!(block_groups.len(), 1);
            assert_eq!(block_groups[0].name, "chr1");
        });
    }

    #[test]
    fn test_import_fasta_duplicate_gives_specific_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGT");
            let path = fasta.to_str().unwrap().to_string();

            py_repo
                .borrow(py)
                .import_fasta(path.clone(), Some("test".to_string()), false, None)
                .unwrap();

            let err = py_repo
                .borrow(py)
                .import_fasta(path, Some("test".to_string()), false, None)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("already exist"),
                "Expected 'already exist' in error: {err}"
            );
        });
    }

    #[test]
    fn test_transaction_commits_both_imports() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta1 = write_fasta(&dir, "one.fa", "chr1", "ACGTACGT");
            let fasta2 = write_fasta(&dir, "two.fa", "chr2", "TTTTGGGG");

            PyRepository::__enter__(py_repo.borrow_mut(py)).unwrap();

            {
                let borrow = py_repo.borrow(py);
                borrow
                    .import_fasta(
                        fasta1.to_str().unwrap().to_string(),
                        Some("test".to_string()),
                        false,
                        None,
                    )
                    .unwrap();
                borrow
                    .import_fasta(
                        fasta2.to_str().unwrap().to_string(),
                        Some("test".to_string()),
                        false,
                        None,
                    )
                    .unwrap();
            }

            PyRepository::__exit__(py_repo.borrow_mut(py), None, None, None).unwrap();

            let block_groups = py_repo.borrow(py).get_sequence_graphs().unwrap();
            assert_eq!(
                block_groups.len(),
                2,
                "Both imports should have been committed"
            );
        });
    }

    #[test]
    fn test_search_finds_exact_match() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            let hits = py_repo.borrow(py).search("ACGT", None, "dna").unwrap();
            assert!(!hits.is_empty(), "Expected at least one match for 'ACGT'");
            assert_eq!(hits.len(), 1);
            let (_, loci) = &hits[0];
            assert!(!loci.is_empty(), "Expected at least one locus for 'ACGT'");
        });
    }

    #[test]
    fn test_search_no_match() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            let hits = py_repo.borrow(py).search("ZZZZ", None, "dna").unwrap();
            assert!(hits.is_empty(), "Expected no matches for 'ZZZZ'");
        });
    }

    #[test]
    fn test_build_index_creates_file() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_sequence_graphs().unwrap();
            let bg = &block_groups[0];

            py_repo.borrow(py).build_index("dna", 4, None).unwrap();

            let index_dir = py_repo
                .borrow(py)
                .context
                .workspace()
                .ensure_gen_dir()
                .join("search_index");
            let index_file = index_dir.join(format!("{}.bin", bg.id));
            assert!(
                index_file.exists(),
                "Index file should exist after build_index"
            );
        });
    }

    #[test]
    fn test_search_with_index_finds_match() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            py_repo.borrow(py).build_index("dna", 4, None).unwrap();
            let hits = py_repo.borrow(py).search("ACGT", None, "dna").unwrap();
            assert!(!hits.is_empty(), "Expected match when searching with index");
        });
    }

    #[test]
    fn test_clear_index_removes_file() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_sequence_graphs().unwrap();
            let bg = &block_groups[0];

            py_repo.borrow(py).build_index("dna", 4, None).unwrap();
            let index_dir = py_repo
                .borrow(py)
                .context
                .workspace()
                .ensure_gen_dir()
                .join("search_index");
            let index_file = index_dir.join(format!("{}.bin", bg.id));
            assert!(index_file.exists(), "Index should exist before clear");

            py_repo.borrow(py).clear_index(None).unwrap();
            assert!(
                !index_file.exists(),
                "Index should be gone after clear_index"
            );
        });
    }

    #[test]
    fn test_blockgroup_build_and_clear_index() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGTACGT");

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_sequence_graphs().unwrap();
            let bg = &block_groups[0];
            let bg_id = bg.id;

            let index_dir = py_repo
                .borrow(py)
                .context
                .workspace()
                .ensure_gen_dir()
                .join("search_index");
            let index_file = index_dir.join(format!("{}.bin", bg_id));

            bg.build_index("protein", 4).unwrap();
            assert!(
                index_file.exists(),
                "Index should exist after PySequenceGraph::build_index"
            );

            bg.clear_index().unwrap();
            assert!(
                !index_file.exists(),
                "Index should be gone after PySequenceGraph::clear_index"
            );
        });
    }

    #[test]
    fn test_transaction_rolls_back_on_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGT");

            PyRepository::__enter__(py_repo.borrow_mut(py)).unwrap();

            py_repo
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    false,
                    None,
                )
                .unwrap();

            // Simulate an exception reaching __exit__ — passes a non-None exc_type
            let fake_exc = py.None().into_bound(py);
            PyRepository::__exit__(py_repo.borrow_mut(py), Some(&fake_exc), None, None).unwrap();

            let block_groups = py_repo.borrow(py).get_sequence_graphs().unwrap();
            assert!(
                block_groups.is_empty(),
                "Import should have been rolled back, but found {} sequence graph(s)",
                block_groups.len()
            );
        });
    }
}
