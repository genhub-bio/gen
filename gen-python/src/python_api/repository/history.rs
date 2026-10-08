use std::{
    fs::{self, File},
    io::copy,
    path::PathBuf,
};

use flate2::read::MultiGzDecoder;
use gen_core::{BranchName, CommitRef, DoltHashId};
use gen_models::{
    assets::{AssetRef, Assets},
    db::DbContext,
    errors::{AddFilesOperationError, FileAdditionError, OperationError},
    history::{
        HistoryEntry, HistoryStore,
        dolt::{DoltBranchRow, DoltHistoryStore, branch_rows},
    },
    operations::{OperationFile, RemoteBranch, add_files_operation},
};
use pyo3::{
    exceptions::{PyFileExistsError, PyFileNotFoundError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyAny,
};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::PyRepository;
use crate::python_api::{
    hash_id::PyHashId,
    utils::{absolute_path, path_to_py_path},
};

fn history_err_to_pyerr(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A repository branch and its current head operation.
#[gen_stub_pyclass]
#[pyclass(name = "Branch")]
#[derive(Clone, Debug)]
pub struct PyBranch {
    /// Branch name.
    #[pyo3(get)]
    pub name: String,
    pub head: DoltHashId,
    /// Name of the tracked remote, or `None`.
    #[pyo3(get)]
    pub remote: Option<String>,
    /// Whether this is the checked-out branch.
    #[pyo3(get)]
    pub is_current: bool,
    /// Whether the branch has uncommitted working-set changes.
    #[pyo3(get)]
    pub dirty: bool,
}

impl PyBranch {
    fn from_row(
        row: DoltBranchRow,
        current_branch: Option<&BranchName>,
        remote: Option<String>,
    ) -> Self {
        Self {
            is_current: current_branch.is_some_and(|branch| branch.0 == row.name),
            name: row.name,
            head: row.hash,
            remote,
            dirty: row.dirty,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBranch {
    /// Hash of the operation at the branch head.
    #[getter]
    fn head(&self) -> PyHashId {
        PyHashId::from_dolt(self.head)
    }

    fn __str__(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!(
            "Branch(name={:?}, head={:?}, current={})",
            self.name,
            self.head.to_string(),
            self.is_current
        )
    }
}

/// A committed Gen operation in repository history.
#[gen_stub_pyclass]
#[pyclass(name = "Operation")]
#[derive(Clone)]
pub struct PyOperation {
    pub id: DoltHashId,
    pub parent_id: Option<DoltHashId>,
    /// Committer name recorded on the operation.
    #[pyo3(get)]
    pub committer: String,
    /// Committer email recorded on the operation.
    #[pyo3(get)]
    pub email: String,
    /// Commit timestamp.
    #[pyo3(get)]
    pub date: String,
    /// Operation message, such as the `message=` given to an edit.
    #[pyo3(get)]
    pub message: String,
    /// Whether this is the head operation of the branch.
    #[pyo3(get)]
    pub is_head: bool,
}

impl From<HistoryEntry> for PyOperation {
    fn from(entry: HistoryEntry) -> Self {
        Self {
            id: entry.commit_hash,
            parent_id: entry.parent_hash,
            committer: entry.committer,
            email: entry.email,
            date: entry.date,
            message: entry.message,
            is_head: entry.is_head,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOperation {
    /// Operation hash.
    #[getter]
    fn id(&self) -> PyHashId {
        PyHashId::from_dolt(self.id)
    }

    /// Hash of the previous operation, or `None` for the first.
    #[getter]
    fn parent_id(&self) -> Option<PyHashId> {
        self.parent_id.map(PyHashId::from_dolt)
    }

    fn __str__(&self) -> String {
        self.id.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Operation(id={:?}, message={:?})",
            self.id.to_string(),
            self.message
        )
    }

    fn __hash__(&self) -> isize {
        PyHashId::from_dolt(self.id).__hash__()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<PyOperation>>()
            .is_ok_and(|other| other.id == self.id)
    }
}

/// A file kept in the repository without being imported as sequence, such as a README, a FASTA or
/// GenBank file that was imported, or an annotation file.
///
/// Add one with `Repository.add_file()` and list them with `Repository.get_assets()`. The
/// content is stored once under a hashed filename inside `.gen`, and text files such as FASTA or
/// README notes are kept compressed. `.name` is the file's original name, `.path` locates the
/// stored copy, which may be compressed, and `.save_as()` writes the original, readable file under
/// any name you choose. Treat the stored copy as read-only.
#[gen_stub_pyclass]
#[pyclass(name = "Asset", unsendable)]
#[derive(Clone)]
pub struct PyAsset {
    asset: AssetRef,
    context: Option<DbContext>,
}

impl PyAsset {
    fn new(asset: AssetRef, context: &DbContext) -> Self {
        Self {
            asset,
            context: Some(context.clone()),
        }
    }

    fn stored_path(&self) -> PyResult<PathBuf> {
        let context = self.context.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("asset has no repository to locate its file in")
        })?;
        self.asset
            .versioned_store_path(context.workspace())
            .map_err(|error| match error {
                FileAdditionError::FileNotFound(path) => PyFileNotFoundError::new_err(format!(
                    "the stored copy of this asset is not on disk ({path}); fetch or pull to download it"
                )),
                other => PyRuntimeError::new_err(other.to_string()),
            })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAsset {
    /// Content-addressed asset id.
    #[getter]
    fn id(&self) -> PyHashId {
        PyHashId::new(self.asset.id)
    }

    /// The file's original name, if known.
    #[getter]
    fn name(&self) -> Option<String> {
        self.asset.name.clone()
    }

    /// Where the stored copy lives, as a `pathlib.Path` inside `.gen`. The filename is a hash and
    /// the bytes may be compressed (BGZF), so use `save_as()` for a plain copy you can name, read
    /// and share. Raises `FileNotFoundError` when the
    /// content has not been downloaded to this repository yet.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "pathlib.Path", imports = ("pathlib")))]
    fn path(&self, python: Python<'_>) -> PyResult<Py<PyAny>> {
        path_to_py_path(python, &self.stored_path()?)
    }

    /// Write a copy of the file to `destination` and return its path.
    ///
    /// If `destination` is a directory the copy is named after `.name` inside it. An existing
    /// file is only replaced with `overwrite=True`. A relative `destination` is resolved against
    /// the current working directory.
    #[pyo3(signature = (destination, *, overwrite=false))]
    #[gen_stub(override_return_type(type_repr = "pathlib.Path", imports = ("pathlib")))]
    fn save_as(
        &self,
        python: Python<'_>,
        #[gen_stub(override_type(type_repr = "str | os.PathLike[str]", imports = ("os")))]
        destination: PathBuf,
        overwrite: bool,
    ) -> PyResult<Py<PyAny>> {
        let source = self.stored_path()?;
        let target = if destination.is_dir() {
            let name = self.asset.name.as_deref().ok_or_else(|| {
                PyValueError::new_err("this asset has no name; give save_as() a file path")
            })?;
            destination.join(name)
        } else {
            destination
        };
        if target.exists() && !overwrite {
            return Err(PyFileExistsError::new_err(format!(
                "{} already exists; pass overwrite=True to replace it",
                target.display()
            )));
        }
        // Text inputs are archived as BGZF; a recorded materialized checksum means the stored bytes
        // differ from the original file, so restore the original bytes for the user.
        let saved = if self.asset.materialized_checksum.is_some() {
            File::open(&source)
                .and_then(|file| copy(&mut MultiGzDecoder::new(file), &mut File::create(&target)?))
        } else {
            fs::copy(&source, &target)
        };
        saved.map_err(|error| PyRuntimeError::new_err(format!("Failed to save asset: {error}")))?;
        path_to_py_path(python, &target)
    }

    fn __str__(&self) -> String {
        self.asset.id.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Asset(id={:?}, name={:?})",
            self.asset.id.to_string(),
            self.asset.name
        )
    }

    fn __hash__(&self) -> isize {
        PyHashId::new(self.asset.id).__hash__()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<PyAsset>>()
            .is_ok_and(|other| other.asset.id == self.asset.id)
    }
}

pub(super) fn branch_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(branch) = value.extract::<PyRef<'_, PyBranch>>() {
        Ok(branch.name.clone())
    } else {
        value.extract::<String>()
    }
}

fn operation_ref(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(operation) = value.extract::<PyRef<'_, PyOperation>>() {
        Ok(operation.id.to_string())
    } else if let Ok(hash_id) = value.extract::<PyRef<'_, PyHashId>>() {
        Ok(hash_id.__str__())
    } else {
        value.extract::<String>()
    }
}

impl PyRepository {
    fn branch_from_row(&self, row: DoltBranchRow, current_branch: Option<&BranchName>) -> PyBranch {
        let remote = if row.remote.is_empty() {
            RemoteBranch::get_remote(self.context.config().conn(), &row.name)
        } else {
            Some(row.remote.clone())
        };
        PyBranch::from_row(row, current_branch, remote)
    }

    fn find_branch(&self, name: &str) -> PyResult<PyBranch> {
        self.get_branches()?
            .into_iter()
            .find(|branch| branch.name == name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("branch '{name}' not found")))
    }

    fn head_operation(&self) -> PyResult<PyOperation> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        history_store
            .log(Some(1))
            .map_err(history_err_to_pyerr)?
            .into_iter()
            .next()
            .map(PyOperation::from)
            .ok_or_else(|| PyRuntimeError::new_err("repository has no operations"))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRepository {
    /// Returns every branch, ordered by name.
    fn get_branches(&self) -> PyResult<Vec<PyBranch>> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        let current_branch = history_store
            .current_branch()
            .map_err(history_err_to_pyerr)?;
        branch_rows(self.context.graph().conn())
            .map_err(history_err_to_pyerr)
            .map(|rows| {
                rows.into_iter()
                    .map(|row| self.branch_from_row(row, current_branch.as_ref()))
                    .collect()
            })
    }

    /// The currently checked out branch.
    #[getter]
    fn current_branch(&self) -> PyResult<Option<PyBranch>> {
        self.get_branches()
            .map(|branches| branches.into_iter().find(|branch| branch.is_current))
    }

    /// Creates a branch named `name` at HEAD, or at `start` (an operation or its hash) when
    /// supplied. It does not switch to the new branch; use `checkout()` for that.
    #[pyo3(signature = (name, start=None))]
    fn create_branch(
        &self,
        name: &str,
        #[gen_stub(override_type(type_repr = "str | HashId | Operation | None", imports = ()))]
        start: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBranch> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        let start_ref = start.map(operation_ref).transpose()?.map(CommitRef);
        history_store
            .create_branch(&BranchName(name.to_string()), start_ref.as_ref())
            .map_err(history_err_to_pyerr)?;
        self.find_branch(name)
    }

    /// Deletes a branch.
    fn delete_branch(
        &self,
        #[gen_stub(override_type(type_repr = "str | Branch", imports = ()))] branch: &Bound<
            '_,
            PyAny,
        >,
    ) -> PyResult<()> {
        let name = branch_name(branch)?;
        DoltHistoryStore::new(self.context.graph().conn())
            .delete_branch(&BranchName(name))
            .map_err(history_err_to_pyerr)
    }

    /// Checks out a branch by name or Branch object and returns its updated metadata.
    ///
    /// Set `create=True` to create a new branch at HEAD before checking it out. Creating an
    /// existing branch is an error unless `exist_ok=True`, which switches to it instead; use that
    /// in notebook cells that may be run more than once.
    #[pyo3(signature = (branch, *, create=false, exist_ok=false))]
    fn checkout(
        &self,
        #[gen_stub(override_type(type_repr = "str | Branch", imports = ()))] branch: &Bound<
            '_,
            PyAny,
        >,
        create: bool,
        exist_ok: bool,
    ) -> PyResult<PyBranch> {
        let name = branch_name(branch)?;
        let already_exists = self.find_branch(&name).is_ok();
        if create && already_exists && !exist_ok {
            return Err(PyRuntimeError::new_err(format!(
                "branch '{name}' already exists; use checkout('{name}') to switch to it, or pass exist_ok=True"
            )));
        }
        if create && !already_exists {
            let history_store = DoltHistoryStore::new(self.context.graph().conn());
            r#gen::history::ensure_clean_working_set(&history_store, "checkout")
                .map_err(history_err_to_pyerr)?;
            self.create_branch(&name, None)?;
        }
        r#gen::commands::checkout::execute(
            self.context.graph().conn(),
            self.context.config().conn(),
            self.context.workspace(),
            None,
            Some(&name),
        )
        .map_err(history_err_to_pyerr)?;
        self.find_branch(&name)
    }

    /// Returns operations for the current branch or a named branch, newest first. `limit` caps how
    /// many are returned.
    #[pyo3(signature = (branch=None, limit=None))]
    fn get_operations(
        &self,
        #[gen_stub(override_type(type_repr = "str | Branch | None", imports = ()))] branch: Option<
            &Bound<'_, PyAny>,
        >,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyOperation>> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        let entries = match branch {
            Some(branch) => history_store
                .log_for_ref(&CommitRef(branch_name(branch)?), limit)
                .map_err(history_err_to_pyerr)?,
            None => history_store.log(limit).map_err(history_err_to_pyerr)?,
        };
        Ok(entries.into_iter().map(PyOperation::from).collect())
    }

    /// Returns the assets reachable from the current branch or a named branch.
    #[pyo3(signature = (branch=None))]
    fn get_assets(
        &self,
        #[gen_stub(override_type(type_repr = "str | Branch | None", imports = ()))] branch: Option<
            &Bound<'_, PyAny>,
        >,
    ) -> PyResult<Vec<PyAsset>> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        let name = match branch {
            Some(branch) => branch_name(branch)?,
            None => {
                history_store
                    .current_branch()
                    .map_err(history_err_to_pyerr)?
                    .ok_or_else(|| PyRuntimeError::new_err("repository has no current branch"))?
                    .0
            }
        };
        Assets::get_branch_assets(self.context.graph().conn(), &name)
            .map_err(history_err_to_pyerr)
            .map(|assets| {
                assets
                    .into_values()
                    .map(|asset| PyAsset::new(asset, &self.context))
                    .collect()
            })
    }

    /// Keep a file in the repository without importing it as sequence, for example a README or a
    /// protocol, and return its `Asset`.
    ///
    /// The content is stored once, inside `.gen`, and recorded as its own operation (with
    /// `message` when given). Adding the same file again fails. Use `asset.path` to locate the
    /// stored copy or `asset.save_as(name)` to write a copy under a name you choose. `filename` is
    /// the path of the file to store; a relative path is resolved against the current working
    /// directory.
    #[pyo3(signature = (filename, message=None))]
    fn add_file(
        &self,
        #[gen_stub(override_type(type_repr = "str | os.PathLike[str]", imports = ("os")))]
        filename: PathBuf,
        message: Option<&str>,
    ) -> PyResult<PyAsset> {
        let filename = absolute_path(&filename)?;
        let path = filename
            .to_str()
            .ok_or_else(|| PyValueError::new_err("filename must be valid UTF-8"))?;
        if !filename.is_file() {
            return Err(PyFileNotFoundError::new_err(format!("no file at '{path}'")));
        }
        let file = OperationFile::new(path);
        let asset_name = file.filename.clone();
        add_files_operation(&self.context, &[file], message).map_err(|error| match error {
            AddFilesOperationError::OperationError(OperationError::NoChanges) => {
                PyRuntimeError::new_err(format!("'{path}': this file is already in the repository"))
            }
            AddFilesOperationError::FileAdditionError(FileAdditionError::FileNotFound(missing)) => {
                PyFileNotFoundError::new_err(missing)
            }
            other => PyRuntimeError::new_err(format!("Failed to add '{path}': {other}")),
        })?;
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        let branch = history_store
            .current_branch()
            .map_err(history_err_to_pyerr)?
            .ok_or_else(|| PyRuntimeError::new_err("repository has no current branch"))?;
        Assets::get_branch_assets(self.context.graph().conn(), &branch.0)
            .map_err(history_err_to_pyerr)?
            .into_values()
            .filter(|asset| asset.name.as_deref() == Some(asset_name.as_str()))
            .max_by_key(|asset| asset.created_on)
            .map(|asset| PyAsset::new(asset, &self.context))
            .ok_or_else(|| PyRuntimeError::new_err("the added file was not recorded as an asset"))
    }

    /// Merges a branch into the current branch and returns the new HEAD operation.
    fn merge(
        &self,
        #[gen_stub(override_type(type_repr = "str | Branch", imports = ()))] branch: &Bound<
            '_,
            PyAny,
        >,
    ) -> PyResult<PyOperation> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        r#gen::history::ensure_clean_working_set(&history_store, "merge")
            .map_err(history_err_to_pyerr)?;
        let name = branch_name(branch)?;
        self.context
            .graph()
            .conn()
            .with_transaction(|| history_store.merge(&CommitRef(name)))
            .map_err(|error| r#gen::history::history_action_error("Merge", &error))
            .map_err(history_err_to_pyerr)?;
        self.head_operation()
    }

    /// Applies one operation to the current branch and returns the new HEAD operation.
    fn apply(
        &self,
        #[gen_stub(override_type(type_repr = "str | HashId | Operation", imports = ()))] operation: &Bound<'_, PyAny>,
    ) -> PyResult<PyOperation> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        r#gen::history::ensure_clean_working_set(&history_store, "apply")
            .map_err(history_err_to_pyerr)?;
        let reference = operation_ref(operation)?;
        let commit_hash = history_store
            .resolve_operation_hash(&CommitRef(reference))
            .map_err(history_err_to_pyerr)?;
        history_store
            .cherry_pick(&commit_hash)
            .map_err(|error| r#gen::history::history_action_error("Apply", &error))
            .map_err(history_err_to_pyerr)?;
        self.head_operation()
    }

    /// Hard-resets the current branch to an operation and returns the resulting HEAD.
    fn reset(
        &self,
        #[gen_stub(override_type(type_repr = "str | HashId | Operation", imports = ()))] operation: &Bound<'_, PyAny>,
    ) -> PyResult<PyOperation> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        r#gen::history::ensure_clean_working_set(&history_store, "reset")
            .map_err(history_err_to_pyerr)?;
        let reference = operation_ref(operation)?;
        history_store
            .reset_hard(&CommitRef(reference))
            .map_err(|error| history_err_to_pyerr(format!("Operation reset failed: {error}")))?;
        self.head_operation()
    }
}

#[cfg(test)]
mod tests {
    use gen_models::{
        collection::Collection,
        history::{HistoryStore, dolt::DoltHistoryStore},
    };
    use pyo3::{IntoPyObject as _, Py, Python};

    use super::PyRepository;

    fn make_repository() -> PyRepository {
        PyRepository {
            context: r#gen::test_helpers::setup_gen_on_disk(),
        }
    }

    #[test]
    fn test_branch_merge_and_reset_workflow() {
        Python::initialize();
        Python::attach(|python| {
            let repository = make_repository();
            Collection::create(repository.context.graph().conn(), "base")
                .expect("should create base collection");
            DoltHistoryStore::new(repository.context.graph().conn())
                .commit_all("base operation")
                .expect("should commit base operation");
            let base_operation = repository
                .get_operations(None, None)
                .expect("should list base operation")
                .remove(0);

            let feature = repository
                .create_branch("feature", None)
                .expect("should create feature branch");
            assert!(!feature.is_current, "new branch should not be current");
            let feature_object = Py::new(python, feature).expect("should create Python branch");
            let checked_out = repository
                .checkout(feature_object.bind(python).as_any(), false, false)
                .expect("should checkout Branch object");
            assert!(
                checked_out.is_current,
                "checked-out branch should be current"
            );

            Collection::create(repository.context.graph().conn(), "feature")
                .expect("should create feature collection");
            DoltHistoryStore::new(repository.context.graph().conn())
                .commit_all("feature operation")
                .expect("should commit feature operation");
            let feature_operations = repository
                .get_operations(Some(feature_object.bind(python).as_any()), None)
                .expect("should list operations for Branch object");
            assert_eq!(
                feature_operations[0].message, "feature operation",
                "feature history should start with its operation"
            );

            let main = "main"
                .into_pyobject(python)
                .expect("should create Python branch name");
            repository
                .checkout(main.as_any(), false, false)
                .expect("should checkout branch name");
            assert!(
                Collection::all(repository.context.graph().conn())
                    .expect("should list collections")
                    .iter()
                    .all(|collection| collection.name != "feature"),
                "main should not contain feature state before merge"
            );

            let merged = repository
                .merge(feature_object.bind(python).as_any())
                .expect("should merge Branch object");
            assert!(merged.is_head, "merge result should describe HEAD");
            assert!(
                Collection::all(repository.context.graph().conn())
                    .expect("should list collections")
                    .iter()
                    .any(|collection| collection.name == "feature"),
                "merge should bring feature state into main"
            );

            let base_operation_object =
                Py::new(python, base_operation).expect("should create Python operation");
            let reset = repository
                .reset(base_operation_object.bind(python).as_any())
                .expect("should reset to Operation object");
            assert_eq!(
                reset.message, "base operation",
                "reset result should describe the selected operation"
            );
            assert!(
                Collection::all(repository.context.graph().conn())
                    .expect("should list collections")
                    .iter()
                    .all(|collection| collection.name != "feature"),
                "reset should restore graph state from the selected operation"
            );
        });
    }
}
