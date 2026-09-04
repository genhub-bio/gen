use gen_core::{BranchName, CommitRef};
use gen_models::{
    history::{
        HistoryEntry, HistoryStore,
        dolt::{DoltBranchRow, DoltHistoryStore, branch_rows},
    },
    operations::RemoteBranch,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyAny};

use super::PyRepository;

fn history_err_to_pyerr(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A repository branch and its current head operation.
#[pyclass(name = "Branch")]
#[derive(Clone, Debug)]
pub struct PyBranch {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub head: String,
    #[pyo3(get)]
    pub remote: Option<String>,
    #[pyo3(get)]
    pub is_current: bool,
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
            head: row.hash.to_string(),
            remote,
            dirty: row.dirty,
        }
    }
}

#[pymethods]
impl PyBranch {
    fn __str__(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!(
            "Branch(name={:?}, head={:?}, current={})",
            self.name, self.head, self.is_current
        )
    }
}

/// A committed Gen operation in repository history.
#[pyclass(name = "Operation")]
#[derive(Clone)]
pub struct PyOperation {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub parent_id: Option<String>,
    #[pyo3(get)]
    pub committer: String,
    #[pyo3(get)]
    pub email: String,
    #[pyo3(get)]
    pub date: String,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub is_head: bool,
}

impl From<HistoryEntry> for PyOperation {
    fn from(entry: HistoryEntry) -> Self {
        Self {
            id: entry.commit_hash.to_string(),
            parent_id: entry.parent_hash.map(|hash| hash.to_string()),
            committer: entry.committer,
            email: entry.email,
            date: entry.date,
            message: entry.message,
            is_head: entry.is_head,
        }
    }
}

#[pymethods]
impl PyOperation {
    fn __str__(&self) -> &str {
        &self.id
    }

    fn __repr__(&self) -> String {
        format!("Operation(id={:?}, message={:?})", self.id, self.message)
    }
}

fn branch_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(branch) = value.extract::<PyRef<'_, PyBranch>>() {
        Ok(branch.name.clone())
    } else {
        value.extract::<String>()
    }
}

fn operation_ref(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(operation) = value.extract::<PyRef<'_, PyOperation>>() {
        Ok(operation.id.clone())
    } else {
        value.extract::<String>()
    }
}

impl PyRepository {
    fn ensure_no_history_transaction(&self, action: &str) -> PyResult<()> {
        if self.in_transaction {
            return Err(PyRuntimeError::new_err(format!(
                "cannot {action} while a repository transaction is active"
            )));
        }
        Ok(())
    }

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

    /// Creates a branch at HEAD, or at `start` when supplied.
    #[pyo3(signature = (name, start=None))]
    fn create_branch(&self, name: &str, start: Option<&str>) -> PyResult<PyBranch> {
        self.ensure_no_history_transaction("create a branch")?;
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        let start_ref = start.map(|reference| CommitRef(reference.to_string()));
        history_store
            .create_branch(&BranchName(name.to_string()), start_ref.as_ref())
            .map_err(history_err_to_pyerr)?;
        self.find_branch(name)
    }

    /// Deletes a branch.
    fn delete_branch(&self, branch: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_no_history_transaction("delete a branch")?;
        let name = branch_name(branch)?;
        DoltHistoryStore::new(self.context.graph().conn())
            .delete_branch(&BranchName(name))
            .map_err(history_err_to_pyerr)
    }

    /// Checks out an existing branch and returns its updated metadata.
    fn checkout(&self, branch: &Bound<'_, PyAny>) -> PyResult<PyBranch> {
        self.ensure_no_history_transaction("checkout a branch")?;
        let name = branch_name(branch)?;
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

    /// Returns operations for the current branch or a named branch.
    #[pyo3(signature = (branch=None, limit=None))]
    fn get_operations(
        &self,
        branch: Option<&Bound<'_, PyAny>>,
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

    /// Merges a branch into the current branch and returns the new HEAD operation.
    fn merge(&self, branch: &Bound<'_, PyAny>) -> PyResult<PyOperation> {
        self.ensure_no_history_transaction("merge")?;
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        r#gen::history::ensure_clean_working_set(&history_store, "merge")
            .map_err(history_err_to_pyerr)?;
        let name = branch_name(branch)?;
        history_store
            .merge(&CommitRef(name))
            .map_err(|error| r#gen::history::history_action_error("Merge", &error))
            .map_err(history_err_to_pyerr)?;
        self.head_operation()
    }

    /// Applies one operation to the current branch and returns the new HEAD operation.
    fn apply(&self, operation: &Bound<'_, PyAny>) -> PyResult<PyOperation> {
        self.ensure_no_history_transaction("apply an operation")?;
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
    fn reset(&self, operation: &Bound<'_, PyAny>) -> PyResult<PyOperation> {
        self.ensure_no_history_transaction("reset")?;
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
    use std::cell::RefCell;

    use gen_models::{
        collection::Collection,
        history::{HistoryStore, dolt::DoltHistoryStore},
    };
    use pyo3::{IntoPyObject as _, Py, Python};

    use super::{PyBranch, PyRepository};

    fn make_repository() -> PyRepository {
        PyRepository {
            context: r#gen::test_helpers::setup_gen_on_disk(),
            in_transaction: false,
            pending_operation_summaries: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn test_branch_merge_and_reset_workflow() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|python| {
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
                .checkout(feature_object.bind(python).as_any())
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
                .checkout(main.as_any())
                .expect("should checkout branch name");
            assert!(
                Collection::all(repository.context.graph().conn(), None)
                    .iter()
                    .all(|collection| collection.name != "feature"),
                "main should not contain feature state before merge"
            );

            let merged = repository
                .merge(feature_object.bind(python).as_any())
                .expect("should merge Branch object");
            assert!(merged.is_head, "merge result should describe HEAD");
            assert!(
                Collection::all(repository.context.graph().conn(), None)
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
                Collection::all(repository.context.graph().conn(), None)
                    .iter()
                    .all(|collection| collection.name != "feature"),
                "reset should restore graph state from the selected operation"
            );
        });
    }

    #[test]
    fn test_history_actions_reject_repository_transaction() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|python| {
            let mut repository = make_repository();
            repository.in_transaction = true;
            let branch = Py::new(
                python,
                PyBranch {
                    name: "main".to_string(),
                    head: String::new(),
                    remote: None,
                    is_current: true,
                    dirty: false,
                },
            )
            .expect("should create Python branch");

            let error = repository
                .checkout(branch.bind(python).as_any())
                .expect_err("should reject checkout during transaction");
            assert!(
                error.to_string().contains("transaction is active"),
                "checkout error should explain the active transaction"
            );
        });
    }
}
