use std::cell::RefCell;

use r#gen::{get_config_connection, get_connection};
use gen_core::config::Workspace;
use gen_models::{
    block_group::BlockGroup,
    collection::Collection,
    db::DbContext,
    errors::OperationError,
    node::Node,
    operations::{Defaults, OperationInfo, OperationSummary, commit_operation_summary},
    sample::Sample,
    traits::Query,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use super::{
    block_group::PySequenceGraph,
    graph_node::PyGraphNode,
    hash_id::PyHashId,
    jupyter_widget::{PyGraphController, build_widget},
    sample::PySample,
    utils::{block_group_err_to_pyerr, path_to_py_path, py_query, sqlite_err_to_pyerr},
};

pub mod exports;
pub mod graph_ops;
pub mod imports;
pub mod search;
pub mod updates;

fn tx_begin(context: &DbContext) -> PyResult<()> {
    let conn = context.graph().conn();
    conn.execute("BEGIN TRANSACTION", [])
        .map_err(sqlite_err_to_pyerr)?;
    Ok(())
}

fn tx_commit(context: &DbContext) -> PyResult<()> {
    context
        .graph()
        .conn()
        .execute("END TRANSACTION", [])
        .map_err(sqlite_err_to_pyerr)?;
    Ok(())
}

fn tx_rollback(context: &DbContext) {
    context.graph().conn().execute("ROLLBACK", []).ok();
}

pub(crate) fn run_operation_write<F, T, M>(
    repository: &PyRepository,
    op: F,
    map_operation_error: M,
) -> PyResult<T>
where
    F: FnOnce(&DbContext) -> PyResult<(T, OperationSummary)>,
    M: FnOnce(OperationError) -> PyErr,
{
    let managed = !repository.in_transaction;
    if managed {
        tx_begin(&repository.context)?;
    }

    let (value, operation_summary) = match op(&repository.context) {
        Ok(value) => value,
        Err(err) => {
            if managed {
                tx_rollback(&repository.context);
            }
            return Err(err);
        }
    };

    if managed {
        if let Err(err) = tx_commit(&repository.context) {
            tx_rollback(&repository.context);
            return Err(err);
        }
        commit_operation_summary(&repository.context, &operation_summary)
            .map_err(map_operation_error)?;
    } else {
        repository
            .pending_operation_summaries
            .borrow_mut()
            .push(operation_summary);
    }

    Ok(value)
}

fn combine_operation_summaries(
    mut operation_summaries: Vec<OperationSummary>,
) -> Option<OperationSummary> {
    match operation_summaries.len() {
        0 => None,
        1 => operation_summaries.pop(),
        _ => {
            let mut files = Vec::new();
            let mut summaries = Vec::with_capacity(operation_summaries.len());
            for operation_summary in operation_summaries {
                files.extend(operation_summary.operation_info.files);
                summaries.push(operation_summary.summary);
            }
            Some(OperationSummary::new(
                OperationInfo {
                    files,
                    description: "python_transaction".to_string(),
                },
                summaries.join("\n"),
            ))
        }
    }
}

fn operation_err_to_pyerr(err: OperationError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// The main entry point for the gen Python module.
///
/// This class manages the database connection and provides methods for
/// querying and manipulating the database.
#[pyclass(name = "Repository", unsendable)]
pub struct PyRepository {
    pub context: DbContext,
    pub in_transaction: bool,
    pending_operation_summaries: RefCell<Vec<OperationSummary>>,
}

impl PyRepository {
    pub(crate) fn get_default_collection(&self) -> String {
        Defaults::get(self.context.config().conn())
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

    /// All block groups currently in `(collection, sample)`.
    pub(crate) fn block_groups_in_sample(
        &self,
        collection_name: &str,
        sample_name: &str,
    ) -> PySample {
        let block_groups = Sample::get_block_groups(
            self.context.graph().conn(),
            collection_name,
            sample_name,
            None,
        )
        .into_iter()
        .map(|bg| self.to_py_block_group(bg))
        .collect();
        PySample::new(
            collection_name.to_string(),
            sample_name.to_string(),
            block_groups,
        )
    }

    /// Look up a single block group by its deterministic (collection, sample, name) id.
    pub(crate) fn get_block_group(
        &self,
        collection_name: &str,
        sample_name: &str,
        name: &str,
    ) -> PyResult<PySequenceGraph> {
        Sample::get_block_groups(
            self.context.graph().conn(),
            collection_name,
            sample_name,
            None,
        )
        .into_iter()
        .find(|bg| bg.name == name)
        .map(|bg| self.to_py_block_group(bg))
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Block group '{}' not found in sample '{}'",
                name, sample_name
            ))
        })
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
        let config_path = gen_dir.join("gen.db");
        let config_conn = get_config_connection(Some(config_path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let db_path = gen_dir.join("default.db");

        let graph_conn = get_connection(db_path.clone()).map_err(|err| {
            PyRuntimeError::new_err(format!(
                "Failed to open database '{}': {err}",
                db_path.display()
            ))
        })?;

        Ok(PyRepository {
            context: DbContext::new(workspace, graph_conn, config_conn)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
            in_transaction: false,
            pending_operation_summaries: RefCell::new(Vec::new()),
        })
    }

    #[getter]
    fn get_gen_dir(&self, py: Python<'_>) -> PyResult<PyObject> {
        path_to_py_path(py, &self.context.workspace().ensure_gen_dir())
    }

    #[getter]
    fn get_db_path(&self, py: Python<'_>) -> PyResult<PyObject> {
        let path = self
            .context
            .workspace()
            .graph_db_path()
            .unwrap_or_else(|_| self.context.workspace().ensure_gen_dir().join("default.db"));
        path_to_py_path(py, &path)
    }

    // Transaction context manager

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
        if slf.in_transaction {
            return Err(PyRuntimeError::new_err("transaction already active"));
        }
        tx_begin(&slf.context)?;
        slf.pending_operation_summaries.borrow_mut().clear();
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
            slf.pending_operation_summaries.borrow_mut().clear();
            tx_rollback(&slf.context);
            return Ok(false);
        }

        let operation_summaries = slf
            .pending_operation_summaries
            .borrow_mut()
            .drain(..)
            .collect::<Vec<_>>();
        if let Err(err) = tx_commit(&slf.context) {
            tx_rollback(&slf.context);
            return Err(err);
        }

        if let Some(operation_summary) = combine_operation_summaries(operation_summaries) {
            commit_operation_summary(&slf.context, &operation_summary)
                .map_err(operation_err_to_pyerr)?;
        }
        Ok(false)
    }

    // Raw database access

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

    // SequenceGraph queries

    fn get_sequence_graph_by_id(&self, id: &PyHashId) -> PyResult<PySequenceGraph> {
        let conn = self.context.graph().conn();
        let block_group =
            BlockGroup::get_by_id(conn, &id.hash_id, None).map_err(block_group_err_to_pyerr)?;
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
        Ok(Collection::get_block_groups(conn, collection_name, None)
            .into_iter()
            .map(|bg| self.to_py_block_group(bg))
            .collect())
    }

    /// All samples in the repository, each holding its sequence graphs.
    fn get_samples(&self) -> PyResult<Vec<PySample>> {
        let conn = self.context.graph().conn();
        let mut samples: Vec<PySample> = Vec::new();
        for bg in BlockGroup::all(conn) {
            let py_bg = self.to_py_block_group(bg);
            match samples.iter_mut().find(|sample| {
                sample.collection_name == py_bg.collection_name
                    && sample.sample_name == py_bg.sample_name
            }) {
                Some(sample) => sample.block_groups.push(py_bg),
                None => samples.push(PySample::new(
                    py_bg.collection_name.clone(),
                    py_bg.sample_name.clone(),
                    vec![py_bg],
                )),
            }
        }
        Ok(samples)
    }

    // Plot

    #[pyo3(signature = (sequence_graph, rows=None, cols=None, detail=None, colors=None))]
    fn plot(
        &self,
        py: Python<'_>,
        sequence_graph: &PySequenceGraph,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
        colors: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let mut ctrl = PyGraphController::for_sequence_graph(sequence_graph)?;
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_widget(py, ctrl, rows, cols, colors)
    }

    fn get_node_sequence(&self, node_key: &PyGraphNode) -> PyResult<String> {
        let sequences_by_node_id =
            Node::get_sequences_by_node_ids(self.context.graph().conn(), &[node_key.node_id], None);
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
    use std::{cell::RefCell, fs};

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
                pending_operation_summaries: RefCell::new(Vec::new()),
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
            // Escape backslashes so a Windows path (e.g. `C:\Users\...`) survives
            // interpolation into a double-quoted Python string literal; otherwise
            // Python reads `\U...` as a truncated unicode escape.
            let path = tmp_dir.path().to_str().unwrap().replace('\\', "\\\\");
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

            let err =
                match py_repo
                    .borrow(py)
                    .import_fasta(path, Some("test".to_string()), false, None)
                {
                    Err(e) => e.to_string(),
                    Ok(_) => panic!("expected duplicate import to fail"),
                };
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
