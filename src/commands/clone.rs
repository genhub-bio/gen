use std::{
    fs,
    io::{self, ErrorKind},
    path::Path,
};

use gen_core::config::Workspace;
use gen_models::{
    db::{DbContext, GraphConnection},
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, clone_remote},
    },
    operations::{Defaults, Remote},
};
use rusqlite::Connection;
use thiserror::Error;

use crate::{
    commands::remote::{discover_dolt_remote_url, login_remote, validate_dolt_remote_url},
    get_config_connection,
};

const ORIGIN: &str = "origin";

#[derive(Debug, Error)]
enum RemoteOperationError {
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "constructed in focused auth/URL retry tests")
    )]
    #[error("Remote url is not valid: {0}")]
    InvalidRemoteUrl(String),
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "constructed in focused auth/URL retry tests")
    )]
    #[error("Auth Error: {0}")]
    AuthError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
}

fn dolt_remote_url(remote_url: &str) -> String {
    let Ok(parsed_url) = url::Url::parse(remote_url) else {
        return remote_url.to_string();
    };
    if parsed_url.scheme() != "file" {
        return remote_url.to_string();
    }

    let Ok(remote_path) = parsed_url.to_file_path() else {
        return remote_url.to_string();
    };
    if remote_path.extension().and_then(|value| value.to_str()) == Some("db") {
        return remote_url.to_string();
    }

    let graph_db_path = remote_path.join(".gen").join("default.db");
    if !graph_db_path.exists() {
        return remote_url.to_string();
    }

    url::Url::from_file_path(graph_db_path)
        .map(|url| url.to_string())
        .unwrap_or_else(|()| remote_url.to_string())
}

pub fn execute(url: &str, workspace: &Workspace) -> Result<(), Box<dyn std::error::Error>> {
    let repo_name = infer_repo_name(url)?;
    let repo_path = workspace.base_dir().join(&repo_name);

    create_clone_directory(&repo_path)?;

    let workspace = Workspace::new(&repo_path);
    workspace.ensure_gen_dir();
    println!("Gen repository initialized.");

    let operation_conn = get_config_connection(Some(workspace.gen_db_path()?))?;
    Remote::create(&operation_conn, ORIGIN, url)?;
    println!("Remote '{ORIGIN}' added successfully");

    Defaults::set_default_remote(&operation_conn, Some(ORIGIN))?;
    println!("Default remote set to '{ORIGIN}'");

    let graph_conn = open_clone_graph_connection(&workspace)?;
    let context = DbContext::new(workspace, graph_conn, operation_conn)?;
    let dolt_remote_url = discover_dolt_remote_url(url)?.unwrap_or_else(|| dolt_remote_url(url));
    validate_dolt_remote_url(&dolt_remote_url)?;
    pull_with_login_on_auth_error(
        || {
            clone_remote(context.graph().conn(), &dolt_remote_url)
                .map_err(RemoteOperationError::from)
        },
        || login_remote(context.config().conn(), Some(ORIGIN)),
    )?;
    let _ = DoltHistoryStore::new(context.graph().conn()).current_branch()?;

    Ok(())
}

fn open_clone_graph_connection(
    workspace: &Workspace,
) -> Result<GraphConnection, Box<dyn std::error::Error>> {
    let graph_db_path = workspace.graph_db_path()?;
    let connection = Connection::open(graph_db_path)?;
    rusqlite::vtab::array::load_module(&connection)?;
    Ok(GraphConnection(connection))
}

fn create_clone_directory(repo_path: &Path) -> io::Result<()> {
    match fs::create_dir(repo_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::AlreadyExists => {
            eprintln!(
                "Directory '{}' already exists.  Will not clone into an existing repository path.",
                repo_path.display()
            );
            Err(err)
        }
        Err(err) => Err(err),
    }
}

fn pull_with_login_on_auth_error<P, L>(
    mut pull_repo: P,
    login: L,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: FnMut() -> Result<(), RemoteOperationError>,
    L: FnOnce() -> Result<(), Box<dyn std::error::Error>>,
{
    match pull_repo() {
        Ok(()) => Ok(()),
        Err(RemoteOperationError::AuthError(_)) => {
            login()?;
            pull_repo()?;
            Ok(())
        }
        Err(err) => Err(Box::new(err)),
    }
}

fn infer_repo_name(repo_url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let parsed = url::Url::parse(repo_url)?;
    let name = parsed
        .path_segments()
        .and_then(|segments| segments.rev().find(|segment| !segment.is_empty()))
        .ok_or("Unable to infer repository name from URL.")?;

    if name == "." || name == ".." || name.contains(std::path::MAIN_SEPARATOR) {
        return Err(format!("Invalid repository name inferred from URL: {name}").into());
    }

    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, io};

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_infer_repo_name_from_genhub_url() {
        assert_eq!(
            infer_repo_name("https://www.genhub.bio/api/repos/foo/bar").unwrap(),
            "bar"
        );
    }

    #[test]
    fn test_infer_repo_name_ignores_trailing_slash() {
        assert_eq!(
            infer_repo_name("https://www.genhub.bio/api/repos/foo/bar/").unwrap(),
            "bar"
        );
    }

    #[test]
    fn test_create_clone_directory_errors_when_directory_exists() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path().join("existing-repo");
        fs::create_dir(&repo_path).unwrap();

        let err = create_clone_directory(&repo_path).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::AlreadyExists);
    }

    #[test]
    fn test_create_clone_directory_creates_missing_directory() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path().join("new-repo");

        create_clone_directory(&repo_path).unwrap();

        assert!(repo_path.is_dir());
    }

    #[test]
    fn test_pull_without_login_when_initial_pull_succeeds() {
        let pull_count = Cell::new(0);
        let login_count = Cell::new(0);

        pull_with_login_on_auth_error(
            || {
                pull_count.set(pull_count.get() + 1);
                Ok(())
            },
            || {
                login_count.set(login_count.get() + 1);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(pull_count.get(), 1);
        assert_eq!(login_count.get(), 0);
    }

    #[test]
    fn test_login_and_retry_pull_after_auth_error() {
        let pull_count = Cell::new(0);
        let login_count = Cell::new(0);

        pull_with_login_on_auth_error(
            || {
                pull_count.set(pull_count.get() + 1);
                if pull_count.get() == 1 {
                    Err(RemoteOperationError::AuthError("missing token".to_string()))
                } else {
                    Ok(())
                }
            },
            || {
                login_count.set(login_count.get() + 1);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(pull_count.get(), 2);
        assert_eq!(login_count.get(), 1);
    }

    #[test]
    fn test_non_auth_pull_errors_do_not_login() {
        let pull_count = Cell::new(0);
        let login_count = Cell::new(0);

        let result = pull_with_login_on_auth_error(
            || {
                pull_count.set(pull_count.get() + 1);
                Err(RemoteOperationError::InvalidRemoteUrl(
                    "not-a-url".to_string(),
                ))
            },
            || {
                login_count.set(login_count.get() + 1);
                Ok(())
            },
        );

        assert!(result.is_err());
        assert_eq!(pull_count.get(), 1);
        assert_eq!(login_count.get(), 0);
    }

    #[test]
    fn test_login_errors_stop_before_retrying_pull() {
        let pull_count = Cell::new(0);
        let login_count = Cell::new(0);

        let result = pull_with_login_on_auth_error(
            || {
                pull_count.set(pull_count.get() + 1);
                Err(RemoteOperationError::AuthError("missing token".to_string()))
            },
            || {
                login_count.set(login_count.get() + 1);
                Err(Box::new(io::Error::other("login failed")))
            },
        );

        assert!(result.is_err());
        assert_eq!(pull_count.get(), 1);
        assert_eq!(login_count.get(), 1);
    }
}
