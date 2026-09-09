use core::fmt::Display;

use gen_models::{
    history::{HistoryStore as _, dolt::DoltHistoryStore},
    operations::{Defaults, Remote as ModelRemote, RemoteBranch},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyAny};

use super::{PyRepository, history::branch_name};

/// A configured Gen repository remote.
#[pyclass(name = "Remote")]
#[derive(Clone, Debug)]
pub struct PyRemote {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub url: String,
}

impl From<ModelRemote> for PyRemote {
    fn from(remote: ModelRemote) -> Self {
        Self {
            name: remote.name,
            url: remote.url,
        }
    }
}

#[pymethods]
impl PyRemote {
    fn __str__(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("Remote(name={:?}, url={:?})", self.name, self.url)
    }
}

fn remote_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(remote) = value.extract::<PyRef<'_, PyRemote>>() {
        Ok(remote.name.clone())
    } else {
        value.extract::<String>()
    }
}

fn optional_remote_name(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    value.map(remote_name).transpose()
}

fn optional_branch_name(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    value.map(branch_name).transpose()
}

impl PyRepository {
    /// Refreshes the live graph connection even when a partially completed remote workflow fails.
    fn finish_remote_operation(
        &mut self,
        action: &str,
        operation_result: Result<(), impl Display>,
    ) -> PyResult<()> {
        let refresh_result = self.refresh_graph_connection();
        match (operation_result, refresh_result) {
            (Ok(()), refresh_result) => refresh_result,
            (Err(error), Ok(())) => Err(PyRuntimeError::new_err(error.to_string())),
            (Err(error), Err(refresh_error)) => Err(PyRuntimeError::new_err(format!(
                "{action} failed: {error}; the repository connection also could not be refreshed: {refresh_error}"
            ))),
        }
    }
}

#[pymethods]
impl PyRepository {
    /// Returns every configured remote, ordered by name.
    fn get_remotes(&self) -> Vec<PyRemote> {
        ModelRemote::list_all(self.context.config().conn())
            .into_iter()
            .map(PyRemote::from)
            .collect()
    }

    /// The repository's default remote, if one is configured.
    #[getter]
    fn default_remote(&self) -> PyResult<Option<PyRemote>> {
        Defaults::get_default_remote(self.context.config().conn())
            .map(|name| ModelRemote::get_by_name(self.context.config().conn(), &name))
            .transpose()
            .map(|remote| remote.map(PyRemote::from))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Adds a named repository remote.
    fn add_remote(&self, name: &str, url: &str) -> PyResult<PyRemote> {
        ModelRemote::create(self.context.config().conn(), name, url)
            .map(PyRemote::from)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Removes a configured remote and clears references to it.
    fn remove_remote(&self, remote: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = remote_name(remote)?;
        r#gen::commands::remote::remove_remote(self.context.config().conn(), &name)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Sets the repository default remote, or clears it when omitted.
    #[pyo3(signature = (remote=None))]
    fn set_default_remote(&self, remote: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let remote = optional_remote_name(remote)?;
        Defaults::set_default_remote(self.context.config().conn(), remote.as_deref())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Associates a remote with the current branch, or clears it when omitted.
    #[pyo3(signature = (remote=None))]
    fn set_branch_remote(&self, remote: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let history_store = DoltHistoryStore::new(self.context.graph().conn());
        let branch = history_store
            .current_branch()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err("no current branch is checked out"))?;
        let remote = optional_remote_name(remote)?;
        RemoteBranch::set_remote_validated(
            self.context.config().conn(),
            &branch.0,
            remote.as_deref(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Pushes a local branch using the same remote workflow as the Gen CLI.
    #[pyo3(signature = (remote=None, branch=None, force=false))]
    fn push(
        &mut self,
        python: Python<'_>,
        remote: Option<&Bound<'_, PyAny>>,
        branch: Option<&Bound<'_, PyAny>>,
        force: bool,
    ) -> PyResult<()> {
        let remote = optional_remote_name(remote)?;
        let branch = optional_branch_name(branch)?;
        let workspace = self.context.workspace().clone();
        let result = python.allow_threads(|| {
            r#gen::commands::remote::operations::execute_push(
                &workspace,
                remote.as_deref(),
                branch.as_deref(),
                force,
            )
            .map_err(|error| error.to_string())
        });
        self.finish_remote_operation("push", result)
    }

    /// Pulls a remote branch using the same merge and asset workflow as the Gen CLI.
    #[pyo3(signature = (remote=None, branch=None))]
    fn pull(
        &mut self,
        python: Python<'_>,
        remote: Option<&Bound<'_, PyAny>>,
        branch: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let remote = optional_remote_name(remote)?;
        let branch = optional_branch_name(branch)?;
        let workspace = self.context.workspace().clone();
        let result = python.allow_threads(|| {
            r#gen::commands::remote::operations::execute_pull(
                &workspace,
                remote.as_deref(),
                branch.as_deref(),
            )
            .map_err(|error| error.to_string())
        });
        self.finish_remote_operation("pull", result)
    }

    /// Fetches a branch into its remote-tracking ref without changing the checkout.
    #[pyo3(signature = (remote=None, branch=None))]
    fn fetch(
        &mut self,
        python: Python<'_>,
        remote: Option<&Bound<'_, PyAny>>,
        branch: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let remote = optional_remote_name(remote)?;
        let branch = optional_branch_name(branch)?;
        let workspace = self.context.workspace().clone();
        let result = python.allow_threads(|| {
            r#gen::commands::remote::operations::execute_fetch(
                &workspace,
                remote.as_deref(),
                branch.as_deref(),
            )
            .map_err(|error| error.to_string())
        });
        self.finish_remote_operation("fetch", result)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use gen_core::{BranchName, config::Workspace};
    use gen_models::{
        collection::Collection,
        history::{
            HistoryStore as _,
            dolt::{DoltHistoryStore, active_branch, hash_of},
        },
        operations::RemoteBranch,
    };
    use pyo3::{Py, Python};
    use tempfile::tempdir;

    use super::PyRepository;
    use crate::python_api::repository::{clone_repository, history::PyBranch};

    fn create_repository(path: &Path) -> PyRepository {
        PyRepository::open_workspace(Workspace::new(path))
            .expect("should create and open repository")
    }

    fn commit_collection(repository: &PyRepository, name: &str) {
        Collection::create(repository.context.graph().conn(), name)
            .expect("should create collection");
        DoltHistoryStore::new(repository.context.graph().conn())
            .commit_all(&format!("add {name}"))
            .expect("should commit collection");
    }

    fn has_collection(repository: &PyRepository, name: &str) -> bool {
        Collection::all(repository.context.graph().conn())
            .expect("should list collections")
            .iter()
            .any(|collection| collection.name == name)
    }

    #[cfg(unix)]
    #[test]
    fn test_remote_configuration_accepts_remote_objects() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|python| {
            let repository_dir = tempdir().expect("should create repository directory");
            let repository = create_repository(repository_dir.path());
            let remote = repository
                .add_remote("origin", "file:///tmp/example")
                .expect("should add remote");
            assert_eq!(
                repository.get_remotes()[0].url,
                "file:///tmp/example",
                "listed remote should preserve its configured URL"
            );

            let remote_object = Py::new(python, remote).expect("should create Python remote");
            repository
                .set_default_remote(Some(remote_object.bind(python).as_any()))
                .expect("should set default from Remote object");
            let default_remote = repository
                .default_remote()
                .expect("should read default remote")
                .expect("should have default remote");
            assert_eq!(
                default_remote.name, "origin",
                "default remote should return its typed configuration"
            );
            repository
                .set_default_remote(None)
                .expect("should clear default remote");
            assert!(
                repository
                    .default_remote()
                    .expect("should read cleared default remote")
                    .is_none(),
                "cleared default remote should be None"
            );
            repository
                .set_default_remote(Some(remote_object.bind(python).as_any()))
                .expect("should restore default remote");
            repository
                .set_branch_remote(Some(remote_object.bind(python).as_any()))
                .expect("should track Remote object");
            assert_eq!(
                RemoteBranch::get_remote(repository.context.config().conn(), "main"),
                Some("origin".to_string()),
                "current branch should track the selected remote"
            );

            repository
                .remove_remote(remote_object.bind(python).as_any())
                .expect("should remove Remote object");
            assert!(
                repository.get_remotes().is_empty(),
                "removed remote should no longer be listed"
            );
            assert_eq!(
                RemoteBranch::get_remote(repository.context.config().conn(), "main"),
                None,
                "removing a remote should clear branch tracking"
            );
            assert!(
                repository
                    .default_remote()
                    .expect("should read default after removal")
                    .is_none(),
                "removing the default remote should clear the default"
            );
        });
    }

    #[cfg(unix)]
    #[test]
    fn test_file_remote_push_pull_and_fetch_refresh_repository_connection() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|python| {
            let remote_dir = tempdir().expect("should create remote directory");
            let mut remote_repository = create_repository(remote_dir.path());
            commit_collection(&remote_repository, "base");

            let local_parent = tempdir().expect("should create local parent directory");
            let local_path = local_parent.path().join("local");
            let remote_url = format!("file://{}", remote_dir.path().display());
            let mut local_repository = clone_repository(python, &remote_url, Some(local_path))
                .expect("should clone remote repository");

            commit_collection(&local_repository, "pushed");
            let remote = local_repository
                .get_remotes()
                .into_iter()
                .next()
                .expect("clone should configure origin");
            let branch = PyBranch {
                name: "main".to_string(),
                head: String::new(),
                remote: Some("origin".to_string()),
                is_current: true,
                dirty: false,
            };
            let remote_object = Py::new(python, remote).expect("should create Python remote");
            let branch_object = Py::new(python, branch).expect("should create Python branch");
            local_repository
                .push(
                    python,
                    Some(remote_object.bind(python).as_any()),
                    Some(branch_object.bind(python).as_any()),
                    false,
                )
                .expect("should push with Remote and Branch objects");
            remote_repository
                .refresh_graph_connection()
                .expect("should refresh remote repository");
            assert!(
                has_collection(&remote_repository, "pushed"),
                "push should update the remote branch"
            );

            commit_collection(&remote_repository, "pulled");
            local_repository
                .pull(python, None, None)
                .expect("should pull current tracked branch");
            assert!(
                has_collection(&local_repository, "pulled"),
                "the live repository connection should observe pulled graph state"
            );

            let remote_history = DoltHistoryStore::new(remote_repository.context.graph().conn());
            remote_history
                .create_branch(&BranchName("feature".to_string()), None)
                .expect("should create remote feature branch");
            remote_history
                .checkout_branch(&BranchName("feature".to_string()))
                .expect("should checkout remote feature branch");
            commit_collection(&remote_repository, "fetched");
            remote_history
                .checkout_branch(&BranchName("main".to_string()))
                .expect("should restore remote main branch");

            let feature = PyBranch {
                name: "feature".to_string(),
                head: String::new(),
                remote: Some("origin".to_string()),
                is_current: false,
                dirty: false,
            };
            let feature_object = Py::new(python, feature).expect("should create Python branch");
            local_repository
                .fetch(python, None, Some(feature_object.bind(python).as_any()))
                .expect("should fetch Branch object");
            assert_eq!(
                active_branch(local_repository.context.graph().conn())
                    .expect("should read local active branch"),
                "main",
                "fetch should not change the live checkout"
            );
            assert_eq!(
                hash_of(local_repository.context.graph().conn(), "origin/feature",)
                    .expect("should resolve fetched tracking ref"),
                hash_of(remote_repository.context.graph().conn(), "feature")
                    .expect("should resolve remote feature ref"),
                "the refreshed connection should observe the fetched tracking ref"
            );
        });
    }
}
