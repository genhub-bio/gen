use std::{
    fs,
    io::{self, ErrorKind},
    path::Path,
};

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
    execute_in_parent(url, Path::new("."))
}

fn execute_in_parent(url: &str, parent_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let repo_name = infer_repo_name(url)?;
    let repo_path = parent_path.join(&repo_name);

    create_clone_directory(&repo_path)?;

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

    use gen_core::HashId;
    use gen_models::{
        file_types::FileTypes,
        manifest::ManifestGenerator,
        operations::{Branch, Operation},
        traits::Query,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::test_helpers::{create_operation, setup_gen_on_disk};

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
    fn create_clone_directory_errors_when_directory_exists() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path().join("existing-repo");
        fs::create_dir(&repo_path).unwrap();

        let err = create_clone_directory(&repo_path).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::AlreadyExists);
    }

    #[test]
    fn create_clone_directory_creates_missing_directory() {
        let temp_dir = tempdir().unwrap();
        let repo_path = temp_dir.path().join("new-repo");

        create_clone_directory(&repo_path).unwrap();

        assert!(repo_path.is_dir());
    }

    #[test]
    fn clone_from_file_remote_materializes_asset_files() {
        let remote_context = setup_gen_on_disk();
        let remote_conn = remote_context.graph().conn();
        let remote_op_conn = remote_context.operations().conn();
        track_database(remote_conn, remote_op_conn).unwrap();

        let remote_operation = create_operation(
            &remote_context,
            "fastas/clone_file.fa",
            FileTypes::Fasta,
            "remote operation",
            HashId::random_str(),
        );
        let remote_branch = Branch::get_by_name(remote_op_conn, "main").unwrap();
        let remote_manifest = ManifestGenerator::new(remote_op_conn)
            .generate_manifest("main", remote_branch.current_operation_hash.as_ref())
            .unwrap();
        let manifest_operation = remote_manifest
            .operations
            .iter()
            .find(|op| op.operation.hash == remote_operation.hash)
            .unwrap();
        let remote_asset_path = remote_context.workspace().repo_root().unwrap().join(
            manifest_operation.file_additions[0]
                .file_addition
                .file_path(),
        );
        fs::remove_file(
            remote_context
                .workspace()
                .repo_root()
                .unwrap()
                .join("fastas/clone_file.fa"),
        )
        .unwrap();

        let clone_parent = tempdir().unwrap();
        let remote_url = format!(
            "file://{}",
            remote_context.workspace().base_dir().to_string_lossy()
        );
        execute_in_parent(&remote_url, clone_parent.path()).unwrap();

        let cloned_repo_path = clone_parent
            .path()
            .join(remote_context.workspace().base_dir().file_name().unwrap());
        let cloned_file_path = cloned_repo_path.join("fastas/clone_file.fa");
        assert!(cloned_file_path.exists());
        assert_eq!(
            fs::read(cloned_file_path).unwrap(),
            fs::read(remote_asset_path).unwrap()
        );

        let operation_conn =
            get_operation_connection(Some(cloned_repo_path.join(".gen/gen.db"))).unwrap();
        let cloned_ops = Operation::all(&operation_conn);
        assert!(cloned_ops.iter().any(|op| op.hash == remote_operation.hash));
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
