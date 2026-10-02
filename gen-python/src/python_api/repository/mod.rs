use std::{fs, path::PathBuf};

use r#gen::{get_config_connection, get_connection_for_branch};
use gen_core::config::Workspace;
use gen_models::{
    block_group::BlockGroup,
    collection::Collection,
    db::DbContext,
    errors::OperationError,
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, set_commit_author_email, set_commit_author_name},
    },
    node::Node,
    operations::{Defaults, OperationSummary, commit_operation_summary},
    sample::Sample,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

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
pub mod history;
pub mod imports;
pub mod remote;
pub mod search;
pub mod updates;

/// Clones a remote Gen repository and opens it.
///
/// When `path` is omitted, the remote repository name is used beneath the
/// current directory. When supplied, `path` is the exact destination and accepts
/// strings or Python path-like objects. The destination may be an empty directory.
/// `committer` and `email`, when given, become the committer identity recorded on operations
/// made in this repository from now on. A destination that only holds a freshly initialized,
/// still-empty `.gen` workspace (for example from an earlier `Repository(path)`) is reused.
#[gen_stub_pyfunction]
#[pyfunction(name = "clone")]
#[pyo3(signature = (url, path=None, committer=None, email=None))]
pub fn clone_repository(
    python: Python<'_>,
    url: &str,
    path: Option<PathBuf>,
    committer: Option<&str>,
    email: Option<&str>,
) -> PyResult<PyRepository> {
    let workspace = match path {
        Some(path) => Workspace::new(path),
        None => {
            let parent = Workspace::from_current_dir();
            let destination =
                r#gen::commands::remote::operations::clone_destination_path(&parent, url)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Workspace::new(destination)
        }
    };
    discard_untouched_workspace(&workspace)?;
    python.detach(|| {
        r#gen::commands::clone::clone_to_workspace(url, &workspace)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    })?;
    let repository = PyRepository::open_workspace(workspace)?;
    repository.set_committer(committer, email)?;
    Ok(repository)
}

const INITIALIZATION_OPERATION_COUNT: usize = 2;

/// Removes a destination's `.gen` directory when it is the only entry and holds no graph data or
/// operations, so agents that opened `Repository(path)` before cloning do not hit a spurious
/// "not an empty directory" error. Anything with content is left for the clone to reject.
fn discard_untouched_workspace(workspace: &Workspace) -> PyResult<()> {
    let destination = workspace.base_dir();
    let Ok(mut entries) = fs::read_dir(destination) else {
        return Ok(());
    };
    let only_gen_dir = match (entries.next(), entries.next()) {
        (Some(Ok(entry)), None) => entry.file_name() == ".gen",
        _ => false,
    };
    if !only_gen_dir {
        return Ok(());
    }
    let untouched = {
        let repository = PyRepository::open_workspace(Workspace::new(destination))?;
        let conn = repository.context.graph().conn();
        let has_graphs = !BlockGroup::select(conn)
            .load()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .is_empty();
        // Initialization itself records the schema-migration and repository-init operations.
        let has_user_operations = DoltHistoryStore::new(conn)
            .log(Some(INITIALIZATION_OPERATION_COUNT + 1))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .len()
            > INITIALIZATION_OPERATION_COUNT;
        !has_graphs && !has_user_operations
    };
    if untouched {
        fs::remove_dir_all(destination.join(".gen"))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    }
    Ok(())
}

/// Runs `op` in one graph transaction and records it as one operation.
///
pub(crate) fn run_context_operation_write<F, T, M>(
    context: &DbContext,
    op: F,
    map_operation_error: M,
) -> PyResult<T>
where
    F: FnOnce(&DbContext) -> PyResult<(T, OperationSummary)>,
    M: FnOnce(OperationError) -> PyErr,
{
    // dolt_commit seals the SQL transaction after recording the operation. Keep
    // the guard alive until then so failures also roll back the graph mutations.
    let _transaction = context
        .graph()
        .conn()
        .unchecked_transaction()
        .map_err(sqlite_err_to_pyerr)?;
    let (value, operation_summary) = op(context)?;
    commit_operation_summary(context, &operation_summary).map_err(map_operation_error)?;

    Ok(value)
}

/// The main entry point for the gen Python module.
///
/// This class manages the database connection and provides methods for
/// querying and manipulating the database.
#[gen_stub_pyclass]
#[pyclass(name = "Repository", unsendable)]
pub struct PyRepository {
    pub context: DbContext,
}

impl PyRepository {
    fn open_workspace(workspace: Workspace) -> PyResult<Self> {
        let gen_dir = workspace.ensure_gen_dir();
        let config_path = gen_dir.join("gen.db");
        let config_conn = get_config_connection(Some(config_path))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let intended_branch = Defaults::get_current_branch(&config_conn);
        let db_path = gen_dir.join("default.db");
        let graph_conn = get_connection_for_branch(db_path.clone(), intended_branch.as_deref())
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "Failed to open database '{}': {error}",
                    db_path.display()
                ))
            })?;

        Ok(Self {
            context: DbContext::new(workspace, graph_conn, config_conn)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
        })
    }

    /// Reopens the graph connection after orchestration performed work through another connection.
    pub(crate) fn refresh_graph_connection(&mut self) -> PyResult<()> {
        let graph_path = self
            .context
            .workspace()
            .graph_db_path()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let intended_branch = Defaults::get_current_branch(self.context.config().conn());
        let graph_connection =
            get_connection_for_branch(graph_path.clone(), intended_branch.as_deref()).map_err(
                |error| {
                    PyRuntimeError::new_err(format!(
                        "Failed to reopen database '{}': {error}",
                        graph_path.display()
                    ))
                },
            )?;
        self.context.set_graph(graph_connection);
        Ok(())
    }

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
            self.context.clone(),
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

    /// Sets the Dolt commit identity for operations recorded from now on. It applies to this
    /// repository only, not to the `gen defaults` config, so it is set once at construction or
    /// clone.
    fn set_committer(&self, committer: Option<&str>, email: Option<&str>) -> PyResult<()> {
        if let Some(committer) = committer {
            if committer.is_empty() {
                return Err(PyValueError::new_err("committer must not be empty"));
            }
            set_commit_author_name(self.context.graph().conn(), committer)
                .map_err(sqlite_err_to_pyerr)?;
        }
        if let Some(email) = email {
            if email.is_empty() {
                return Err(PyValueError::new_err("email must not be empty"));
            }
            set_commit_author_email(self.context.graph().conn(), email)
                .map_err(sqlite_err_to_pyerr)?;
        }
        Ok(())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRepository {
    /// Open the workspace at `path`, creating it if it does not exist. `path` may be the workspace
    /// or its `.gen` directory; when omitted the workspace is discovered from the current
    /// directory. `committer` and `email` become the identity recorded on operations made through
    /// this object.
    #[new]
    #[pyo3(signature = (path = Option::<String>::None, committer = None, email = None))]
    fn new(path: Option<String>, committer: Option<&str>, email: Option<&str>) -> PyResult<Self> {
        let workspace = match path {
            Some(path_str) => Workspace::new(path_str),
            None => Workspace::from_current_dir(),
        };

        let repository = Self::open_workspace(workspace)?;
        repository.set_committer(committer, email)?;
        Ok(repository)
    }

    /// Path of the `.gen` directory holding this repository's databases and assets.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "pathlib.Path", imports = ("pathlib")))]
    fn get_gen_dir(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        path_to_py_path(py, &self.context.workspace().ensure_gen_dir())
    }

    /// Path of the graph database file.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "pathlib.Path", imports = ("pathlib")))]
    fn get_db_path(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let path = self
            .context
            .workspace()
            .graph_db_path()
            .unwrap_or_else(|_| self.context.workspace().ensure_gen_dir().join("default.db"));
        path_to_py_path(py, &path)
    }

    // Raw database access

    /// Run raw SQL against the graph database. Low-level: prefer the editing and import APIs, which
    /// record operations.
    #[gen_stub(skip)]
    fn execute(&self, query: &str) -> PyResult<()> {
        self.context
            .graph()
            .conn()
            .execute(query, [])
            .map_err(sqlite_err_to_pyerr)?;
        Ok(())
    }

    /// Run raw SQL against the graph database and return rows. Low-level: prefer `graph.search()`,
    /// `graph.region()` and the typed getters.
    #[gen_stub(skip)]
    fn query(&self, py: Python<'_>, query: &str) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        py_query(py, self.context.graph().conn(), query)
    }

    // SequenceGraph queries

    /// Return the sequence graph with this `HashId` (see `SequenceGraph.id`). Use it to rebuild a
    /// graph handle in another Repository object, for example in a worker thread.
    fn get_sequence_graph_by_id(&self, id: &PyHashId) -> PyResult<PySequenceGraph> {
        let conn = self.context.graph().conn();
        let block_group =
            BlockGroup::get_by_id(conn, &id.hash_id, None).map_err(block_group_err_to_pyerr)?;
        Ok(self.to_py_block_group(block_group))
    }

    /// Return every sequence graph in the repository, across all samples and collections. Group
    /// them by sample with `repo.samples` instead.
    fn get_sequence_graphs(&self) -> PyResult<Vec<PySequenceGraph>> {
        let conn = self.context.graph().conn();
        Ok(BlockGroup::select(conn)
            .load()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_iter()
            .map(|bg| self.to_py_block_group(bg))
            .collect())
    }

    /// Return the sequence graphs belonging to one collection.
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
    #[getter]
    fn samples(&self) -> PyResult<Vec<PySample>> {
        let conn = self.context.graph().conn();
        let mut samples: Vec<PySample> = Vec::new();
        for bg in BlockGroup::select(conn)
            .load()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        {
            let py_bg = self.to_py_block_group(bg);
            match samples.iter_mut().find(|sample| {
                sample.collection_name == py_bg.collection_name
                    && sample.sample_name == py_bg.sample_name
            }) {
                Some(sample) => sample.sequence_graphs.push(py_bg),
                None => samples.push(PySample::new(
                    py_bg.collection_name.clone(),
                    py_bg.sample_name.clone(),
                    vec![py_bg],
                    self.context.clone(),
                )),
            }
        }
        Ok(samples)
    }

    // Plot

    /// show_history : bool, optional
    ///     Keep retired edit-site and pruned edges in the graph, dimmed,
    ///     instead of removing them along with the nodes only they reach.
    ///     Defaults to ``False``.
    #[pyo3(signature = (sequence_graph, rows=None, cols=None, detail=None, colors=None, show_history=false))]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors plot()'s Python signature"
    )]
    fn plot(
        &self,
        py: Python<'_>,
        sequence_graph: &PySequenceGraph,
        rows: Option<u32>,
        cols: Option<u32>,
        detail: Option<&str>,
        colors: Option<Py<PyAny>>,
        show_history: bool,
    ) -> PyResult<Py<PyAny>> {
        let mut ctrl = PyGraphController::for_sequence_graph(sequence_graph, show_history)?;
        if let Some(node_detail) = detail {
            ctrl.set_detail(node_detail)?;
        }
        let ctrl = Py::new(py, ctrl)?;
        build_widget(py, ctrl, rows, cols, colors)
    }

    /// Return the stored sequence slice of a node. Prefer `SequenceGraph.get_node_sequence()`,
    /// `locus.sequence` or `graph.all_sequences()`.
    #[gen_stub(skip)]
    fn get_node_sequence(&self, node_key: &PyGraphNode) -> PyResult<String> {
        let sequences_by_node_id = Node::get_sequences_by_node_ids(
            self.context.graph().conn(),
            self.context.workspace(),
            &[node_key.node_id],
            None,
        );
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
    #[cfg(unix)]
    use crate::python_api::repository::clone_repository;

    fn make_repo(py: Python<'_>) -> Py<PyRepository> {
        let ctx = setup_gen_on_disk();
        Py::new(py, PyRepository { context: ctx }).unwrap()
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

    #[cfg(unix)]
    #[test]
    fn test_clone_returns_open_repository() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let source = make_repo(py);
            let fasta_dir = tempdir().unwrap();
            let fasta = write_fasta(&fasta_dir, "test.fa", "chr1", "ACGTACGT");
            source
                .borrow(py)
                .import_fasta(
                    fasta.to_str().unwrap().to_string(),
                    Some("test".to_string()),
                    None,
                    None,
                    None,
                )
                .unwrap();
            let source_root = source.borrow(py).context.workspace().repo_root().unwrap();
            drop(source);

            let destination_parent = tempdir().unwrap();
            let destination = destination_parent.path().join("clone");
            let remote_url = format!("file://{}", source_root.display());
            let cloned = clone_repository(py, &remote_url, Some(destination), None, None)
                .expect("should clone and open repository");

            let block_groups = cloned.get_sequence_graphs().unwrap();
            assert_eq!(
                block_groups.len(),
                1,
                "clone should contain the source sequence graph"
            );
            assert_eq!(
                block_groups[0].name, "chr1",
                "clone should preserve the source sequence graph name"
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
                    None,
                    None,
                    None,
                )
                .unwrap();

            let block_groups = py_repo.borrow(py).get_sequence_graphs().unwrap();
            assert_eq!(block_groups.len(), 1);
            assert_eq!(block_groups[0].name, "chr1");
        });
    }

    #[test]
    fn test_import_fasta_accepts_explicit_fai_and_gzi_keywords() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            let dir = tempdir().unwrap();
            let fasta = write_fasta(&dir, "test.fa", "chr1", "ACGTACGT");
            let fasta = fasta.to_str().unwrap().replace('\\', "\\\\");
            let fai = dir
                .path()
                .join("missing.fai")
                .to_str()
                .unwrap()
                .replace('\\', "\\\\");
            let gzi = dir
                .path()
                .join("missing.gzi")
                .to_str()
                .unwrap()
                .replace('\\', "\\\\");

            py_run!(
                py,
                py_repo,
                &format!(
                    r#"
                    try:
                        py_repo.import_fasta("{fasta}", sample="fai", fai="{fai}")
                    except RuntimeError as error:
                        assert "could not read supplied FAI index" in str(error), str(error)
                    else:
                        raise AssertionError("explicit FAI path should be read")

                    try:
                        py_repo.import_fasta("{fasta}", sample="gzi", gzi="{gzi}")
                    except RuntimeError as error:
                        assert "GZI index can only be used" in str(error), str(error)
                    else:
                        raise AssertionError("explicit GZI path should be validated")
                    "#
                )
            );
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
                .import_fasta(path.clone(), Some("test".to_string()), None, None, None)
                .unwrap();

            let err = match py_repo.borrow(py).import_fasta(
                path,
                Some("test".to_string()),
                None,
                None,
                None,
            ) {
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
    fn test_import_sequence_returns_graph_and_builds_sample_in_a_loop() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            py_run!(
                py,
                py_repo,
                r#"
                first = py_repo.import_sequence("ACGTACGT", "chr1", sample="wt")
                assert first.name == "chr1" and first.sample_name == "wt"
                wt = next(sample for sample in py_repo.samples if sample.sample_name == "wt")
                second = py_repo.import_sequence("TTTT", "chr2", sample=wt)
                assert second.sample_name == "wt"
                third = py_repo.import_sequence("GGGG", "chr3", sample="wt")
                samples = {sample.sample_name: sample for sample in py_repo.samples}
                assert sorted(graph.name for graph in samples["wt"]) == ["chr1", "chr2", "chr3"]
                other = py_repo.import_sequence("GGGG", "chrA", sample="other")
                assert len(py_repo.get_sequence_graphs()) == 4
                for not_a_sample in (42, first):
                    try:
                        py_repo.import_sequence("ACGT", "x", sample=not_a_sample)
                    except TypeError as error:
                        assert "sample must be" in str(error), str(error)
                    else:
                        raise AssertionError(f"expected TypeError for {not_a_sample!r}")
                "#
            );
        });
    }

    #[test]
    fn test_import_sequence_rejects_bad_input() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            py_run!(
                py,
                py_repo,
                r#"
                for args, message in [
                    (("ACGT",), "name is required"),
                    (("ACGT", ""), "must not be empty"),
                    (("", "chr1"), "is empty"),
                    ((42, "chr1"), "must be a string"),
                ]:
                    try:
                        py_repo.import_sequence(*args)
                    except ValueError as error:
                        assert message in str(error), str(error)
                    else:
                        raise AssertionError(f"expected ValueError for {args!r}")
                assert len(py_repo.get_sequence_graphs()) == 0
                "#
            );
        });
    }

    #[test]
    fn test_import_sequence_duplicate_gives_specific_error() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            py_run!(
                py,
                py_repo,
                r#"
                py_repo.import_sequence("ACGT", "chr1", sample="a")
                try:
                    py_repo.import_sequence("ACGT", "chr1", sample="a")
                except RuntimeError as error:
                    assert "already exist" in str(error), str(error)
                else:
                    raise AssertionError("expected duplicate import to fail")
                "#
            );
        });
    }

    #[test]
    fn test_import_sequence_reusing_a_name_with_new_contents_fails() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            py_run!(
                py,
                py_repo,
                r#"
                py_repo.import_sequence("ACGT", "chr1", sample="a")
                try:
                    py_repo.import_sequence("TTTT", "chr1", sample="a")
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("expected reusing a name in a sample to fail")
                assert len(py_repo.get_sequence_graphs()) == 1
                "#
            );
        });
    }

    #[test]
    fn test_import_sequence_circular_and_reference() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let py_repo = make_repo(py);
            py_run!(
                py,
                py_repo,
                r#"
                graph = py_repo.import_sequence("ACGTAC", "plasmid", circular=True)
                assert graph.name == "plasmid"
                reference = py_repo.import_reference_sequence("GGCC", "ref", name="ring", circular=True)
                assert reference.name == "ring" and reference.sample_name == "ref"
                try:
                    py_repo.import_reference_sequence("TTTT", "ref", name="ring")
                except RuntimeError as error:
                    assert "already exists" in str(error), str(error)
                else:
                    raise AssertionError("expected reusing a name in a reference sample to fail")
                "#
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
                    None,
                    None,
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
                    None,
                    None,
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
                    None,
                    None,
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
                    None,
                    None,
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
                    None,
                    None,
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
                    None,
                    None,
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
}
