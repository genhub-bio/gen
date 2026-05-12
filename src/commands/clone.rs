use std::{fs, path::PathBuf};

use gen_core::config::Workspace;
use gen_models::{
    db::DbContext,
    operations::{Defaults, Remote},
};

use crate::{
    commands::remote::login_remote,
    get_connection, get_operation_connection,
    operation_management::{RemoteOperationError, pull},
    track_database,
};

const ORIGIN: &str = "origin";

pub fn execute(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let repo_name = infer_repo_name(url)?;
    let repo_path = PathBuf::from(&repo_name);

    fs::create_dir(&repo_path)?;

    let workspace = Workspace::new(&repo_path);
    workspace.ensure_gen_dir();
    println!("Gen repository initialized.");

    let operation_conn = get_operation_connection(Some(workspace.gen_db_path()?))?;
    Remote::create(&operation_conn, ORIGIN, url)?;
    println!("Remote '{ORIGIN}' added successfully");

    Defaults::set_default_remote(&operation_conn, Some(ORIGIN))?;
    println!("Default remote set to '{ORIGIN}'");

    let graph_conn = get_connection(workspace.ensure_gen_dir().join("default.db"))?;
    let context = DbContext::new(workspace, graph_conn, operation_conn);
    track_database(context.graph().conn(), context.operations().conn())?;
    pull_with_login_on_auth_error(
        || pull(&context, None),
        || login_remote(context.operations().conn(), Some(ORIGIN)),
    )?;

    Ok(())
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

    use super::*;

    #[test]
    fn infer_repo_name_from_genhub_url() {
        assert_eq!(
            infer_repo_name("https://www.genhub.bio/api/repos/foo/bar").unwrap(),
            "bar"
        );
    }

    #[test]
    fn infer_repo_name_ignores_trailing_slash() {
        assert_eq!(
            infer_repo_name("https://www.genhub.bio/api/repos/foo/bar/").unwrap(),
            "bar"
        );
    }

    #[test]
    fn pull_without_login_when_initial_pull_succeeds() {
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
    fn login_and_retry_pull_after_auth_error() {
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
    fn non_auth_pull_errors_do_not_login() {
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
    fn login_errors_stop_before_retrying_pull() {
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
