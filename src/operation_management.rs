use std::{
    collections::HashMap,
    fs,
    io::copy,
    path::{Path as FilePath, PathBuf},
    str,
};

use gen_core::{
    HashId,
    config::{get_gen_dir, get_repo_root_path},
    errors::{ConfigError, ConnectionError},
    traits::Capnp,
};
use gen_models::{
    changesets::{apply_changeset, revert_changeset},
    errors::{ChangesetError, FileAdditionError, OperationError, RemoteError},
    file_types::FileTypes,
    manifest::{
        ManifestComparer, ManifestDiff, ManifestDiffError, ManifestError, ManifestGenerator,
        ManifestOperation,
    },
    metadata::get_db_uuid,
    operations::{
        Branch, Defaults, FileAddition, Operation, OperationFile, OperationInfo, OperationState,
        Remote,
    },
    session_operations::{end_operation, start_operation},
    traits::*,
};
use itertools::Itertools;
use petgraph::Direction;
use reqwest::blocking::{Client, multipart};
use rusqlite::{self, Connection, Error as SQLError};
use serde::Deserialize;
use thiserror::Error;
use url_parse::core::Parser;

use crate::{
    commands::remote::utils::load_tokens, get_connection, get_operation_connection, track_database,
};

/* General information

Changesets from sqlite will be created in the order that operations are applied in the database,
so given our foreign key setup, we would not expect out of order table/row creation. i.e. block
groups will always appear before block group edges, etc.

 */

#[derive(Debug, PartialEq, Error)]
pub enum CheckoutError {
    #[error("Branch Error: {0}")]
    BranchError(String),
    #[error("Move Error: {0}")]
    MoveError(#[from] MoveError),
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
}

#[derive(Debug, PartialEq, Error)]
pub enum MergeError {
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Invalid Branch: {0}")]
    InvalidBranch(String),
}

#[derive(Debug, PartialEq, Error)]
pub enum MoveError {
    #[error("Changeset Error: {0}")]
    ChangesetError(#[from] ChangesetError),
    #[error("Connection Error: {0}")]
    ConnectionError(#[from] ConnectionError),
    #[error("{0}")]
    NoPath(String),
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
}

#[derive(Debug, PartialEq, Error)]
pub enum ResetError {
    #[error("Branch Error: {0}")]
    BranchError(String),
    #[error("Changeset Error: {0}")]
    ChangesetError(#[from] ChangesetError),
    #[error("Move Error: {0}")]
    MoveError(#[from] MoveError),
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
}

#[derive(Debug, Error)]
pub enum RemoteOperationError {
    #[error("Failed to transfer {0} from {1} to {2}")]
    FileTransferError(String, String, String),
    #[error("Remote url is not valid: {0}")]
    InvalidRemoteUrl(String),
    #[error("Gen does not support the scheme {0} in the remote url {1}")]
    UnsupportedRemoteScheme(String, String),
    #[error("Remote url not set, please set it using set-remote before pushing or pulling")]
    RemoteUrlNotSet,
    #[error(
        "Remote repo has changes that are not in the local repo, please pull them before pushing."
    )]
    RemoteBranchAhead,
    #[error("Remote Error: {0}")]
    RemoteError(#[from] RemoteError),
    #[error("Auth Error: {0}")]
    AuthError(String),
    #[error("IO Error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Manifest Error: {0}")]
    ManifestError(#[from] ManifestError),
    #[error("Manifest Diff Error: {0}")]
    ManifestDiffError(#[from] ManifestDiffError),
    #[error("Connection Error: {0}")]
    ConnectionError(#[from] ConnectionError),
    #[error("Config Error: {0}")]
    ConfigError(#[from] ConfigError),
    #[error("Reqwest Error: {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),
    #[error("Remote {0} does not exist.")]
    DoesNotExist(String),
    #[error("No operations present in current branch")]
    NoOperations,
    #[error("File Addition Error: {0}")]
    FileAdditionError(#[from] FileAdditionError),
    #[error("Branch Error: {0}")]
    BranchError(String),
}

pub enum FileMode {
    Read,
    Write,
}

pub fn get_file(path: &PathBuf, mode: FileMode) -> fs::File {
    let file;
    match mode {
        FileMode::Read => {
            if fs::metadata(path).is_ok() {
                file = fs::File::open(path);
            } else {
                file = fs::File::create_new(path);
            }
        }
        FileMode::Write => {
            file = fs::File::create(path);
        }
    }

    file.unwrap()
}

pub fn reset(
    conn: Option<&Connection>,
    operation_conn: &Connection,
    op_hash: &HashId,
) -> Result<(), ResetError> {
    let dest_operation = Operation::get_by_id(operation_conn, op_hash)
        .ok_or(OperationError::NoOperation(format!("{op_hash}")))?;
    move_to(conn, operation_conn, &dest_operation)?;
    Ok(())
}

pub fn apply(
    connection: Option<&Connection>,
    operation_conn: &Connection,
    op_hash: &HashId,
    force_hash: impl Into<Option<HashId>>,
) -> Result<Operation, OperationError> {
    let operation = Operation::get_by_id(operation_conn, op_hash)
        .ok_or(OperationError::NoOperation(format!("{op_hash}")))?;
    let changeset = operation.get_changeset();
    let conn = if let Some(c) = connection {
        c
    } else {
        &get_connection(changeset.db_path)?
    };

    let dependencies = operation.get_changeset_dependencies();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let mut session = start_operation(conn);
    match apply_changeset(conn, &changeset.changes, &dependencies) {
        Ok(_) => {}
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(OperationError::ChangesetError(e));
        }
    }
    let full_op_hash = operation.hash;
    match end_operation(
        conn,
        operation_conn,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: format!("{full_op_hash}/changeset"),
                file_type: FileTypes::Changeset,
            }],
            description: "changeset_application".to_string(),
        },
        &format!("Applied changeset {full_op_hash}."),
        force_hash,
    ) {
        Ok(op) => {
            operation_conn.execute("END TRANSACTION", [])?;
            conn.execute("END TRANSACTION", [])?;
            Ok(op)
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            Err(e)
        }
    }
}

pub fn merge<'a>(
    conn: Option<&Connection>,
    operation_conn: &Connection,
    source_branch: i64,
    other_branch: i64,
    force_hash: impl Into<Option<&'a str>>,
) -> Result<Vec<Operation>, MergeError> {
    let mut new_operations: Vec<Operation> = vec![];
    let hash_prefix = force_hash.into();
    let current_branch =
        OperationState::get_current_branch(operation_conn).expect("No current branch.");
    if source_branch != current_branch {
        return Err(MergeError::InvalidBranch("Source branch and current branch must match. Checkout the branch you wish to merge into.".to_string()));
    }
    if other_branch == current_branch {
        return Err(MergeError::InvalidBranch(
            "Target branch to merge is the currently checked out branch.".to_string(),
        ));
    }
    let current_operations = Branch::get_operations(operation_conn, source_branch);
    let other_operations = Branch::get_operations(operation_conn, other_branch);
    let first_different_op = other_operations
        .iter()
        .position(|op| !current_operations.contains(op))
        .expect("No common operations between two branches.");
    if first_different_op < other_operations.len() {
        for (index, operation) in other_operations[first_different_op..].iter().enumerate() {
            println!("Applying operation {op_id}", op_id = operation.hash);
            // Apply sets operation state via end_operation so we don't need to do it here
            let new_op = if let Some(hash) = hash_prefix {
                apply(
                    conn,
                    operation_conn,
                    &operation.hash,
                    HashId::convert_str(format!("{hash}-{index}").as_str()),
                )?
            } else {
                apply(conn, operation_conn, &operation.hash, None)?
            };
            new_operations.push(new_op);
        }
    }
    Ok(new_operations)
}

pub fn move_to(
    connection: Option<&Connection>,
    operation_conn: &Connection,
    operation: &Operation,
) -> Result<(), MoveError> {
    let current_op_hash = OperationState::get_operation(operation_conn)
        .ok_or(OperationError::NoOperation("No operation set".to_string()))?;
    let op_hash = operation.hash;
    if current_op_hash == op_hash {
        return Ok(());
    }
    let path = Operation::get_path_between(operation_conn, current_op_hash, op_hash);
    if path.is_empty() {
        return Err(MoveError::NoPath(format!(
            "No path exists from {current_op_hash} to {op_hash}."
        )));
    }
    for (operation_hash, direction, next_op) in path.iter() {
        match direction {
            Direction::Incoming => {
                println!("Reverting operation {operation_hash}");
                let op_to_apply = Operation::get_by_id(operation_conn, operation_hash)
                    .ok_or(OperationError::NoOperation(format!("{operation_hash}")))?;
                let changeset = op_to_apply.get_changeset();
                let conn = if let Some(c) = connection {
                    c
                } else {
                    &get_connection(changeset.db_path)?
                };

                conn.execute("BEGIN TRANSACTION", []).unwrap();

                match revert_changeset(conn, &changeset.changes) {
                    Ok(_) => {
                        conn.execute("END TRANSACTION", [])?;
                    }
                    Err(e) => {
                        conn.execute("ROLLBACK TRANSACTION;", [])?;
                        return Err(MoveError::ChangesetError(e));
                    }
                }
            }
            Direction::Outgoing => {
                println!("Applying operation {next_op}");
                let op_to_apply = Operation::get_by_id(operation_conn, next_op)
                    .ok_or(OperationError::NoOperation(format!("{operation_hash}")))?;
                let changeset = op_to_apply.get_changeset();
                let dependencies = op_to_apply.get_changeset_dependencies();
                let conn = if let Some(c) = connection {
                    c
                } else {
                    &get_connection(changeset.db_path)?
                };
                conn.execute("BEGIN TRANSACTION", [])?;
                match apply_changeset(conn, &changeset.changes, &dependencies) {
                    Ok(_) => {
                        conn.execute("END TRANSACTION", [])?;
                    }
                    Err(e) => {
                        conn.execute("ROLLBACK TRANSACTION;", [])?;
                        return Err(MoveError::ChangesetError(e));
                    }
                }
            }
        }
        OperationState::set_operation(operation_conn, next_op);
    }
    Ok(())
}

pub fn checkout(
    conn: Option<&Connection>,
    operation_conn: &Connection,
    branch_name: &Option<String>,
    operation_hash: Option<HashId>,
) -> Result<(), CheckoutError> {
    let mut dest_op_hash = None;
    if let Some(name) = branch_name {
        let current_branch = OperationState::get_current_branch(operation_conn).ok_or(
            CheckoutError::BranchError("No current branch set".to_string()),
        )?;
        let branch = Branch::get_by_name(operation_conn, name).ok_or(
            CheckoutError::BranchError(format!("No branch named {name}")),
        )?;
        if operation_hash.is_none() {
            dest_op_hash = Some(branch.current_operation_hash.ok_or(
                CheckoutError::BranchError(
                    "Destination Branch has no current operation set.".to_string(),
                ),
            )?);
        }
        let branch_operations = Branch::get_operations(operation_conn, branch.id);
        // TODO: Make a recursive sql to do this instead of having to load all the operations each time
        let operation_in_branch = branch_operations
            .iter()
            .any(|operation| Some(operation.hash) == dest_op_hash);
        if !operation_in_branch {
            return Err(CheckoutError::OperationError(OperationError::NoOperation(
                format!("Operation {dest_op_hash:?} not found in branch {name}"),
            )));
        }
        if current_branch != branch.id {
            OperationState::set_branch(operation_conn, name);
        }
    }
    if let Some(hash) = dest_op_hash {
        move_to(
            conn,
            operation_conn,
            &Operation::get_by_id(operation_conn, &hash)
                .ok_or(OperationError::NoOperation(format!("{hash}")))?,
        )?;
        Ok(())
    } else {
        Err(CheckoutError::OperationError(OperationError::NoOperation(
            "No operation found to checkout".to_string(),
        )))
    }
}

pub fn parse_patch_operations(
    branch_operations: &[Operation],
    head_hash: &HashId,
    operations: &str,
) -> Vec<HashId> {
    let mut results = vec![];
    let (head_pos, _) = branch_operations
        .iter()
        .find_position(|op| op.hash == *head_hash)
        .expect("Current head position is not in branch.");
    for operation in operations.split(",") {
        if operation.contains("..") {
            let mut it = operation.split("..");
            let start = it.next().unwrap().parse::<String>().unwrap();
            let end = it.next().unwrap().parse::<String>().unwrap();

            let start_hash = if start.starts_with("HEAD") {
                if start.contains("~") {
                    let mut it = start.rsplit("~");
                    let count = it.next().unwrap().parse::<usize>().unwrap();
                    format!("{}", branch_operations[head_pos - count].hash)
                } else {
                    format!("{}", branch_operations[head_pos].hash)
                }
            } else {
                start
            };

            let end_hash = if end.starts_with("HEAD") {
                if end.contains("~") {
                    let mut it = end.rsplit("~");
                    let count = it.next().unwrap().parse::<usize>().unwrap();
                    format!("{}", branch_operations[head_pos - count].hash)
                } else {
                    format!("{}", branch_operations[head_pos].hash)
                }
            } else {
                end
            };
            let mut start_iter = branch_operations
                .iter()
                .positions(|op| op.hash.starts_with(start_hash.as_str()));
            let start_pos = start_iter
                .next()
                .unwrap_or_else(|| panic!("Unable to find starting hash {start_hash:?}"));
            let mut end_iter = branch_operations
                .iter()
                .positions(|op| op.hash.starts_with(end_hash.as_str()));
            let end_pos = end_iter
                .next()
                .unwrap_or_else(|| panic!("Unable to find end hash {end_hash:?}"));
            if start_iter.next().is_some() {
                panic!("Start hash {start_hash} is ambiguous.");
            }
            if end_iter.next().is_some() {
                panic!("Ending hash {end_hash} is ambiguous.");
            }
            results.extend(
                branch_operations[start_pos..end_pos + 1]
                    .iter()
                    .map(|op| op.hash),
            );
        } else {
            let hash = if operation.starts_with("HEAD") {
                if operation.contains("~") {
                    let mut it = operation.rsplit("~");
                    let count = it.next().unwrap().parse::<usize>().unwrap();
                    branch_operations[head_pos - count].hash
                } else {
                    branch_operations[head_pos].hash
                }
            } else {
                let mut iter = branch_operations
                    .iter()
                    .positions(|op| op.hash.starts_with(operation));
                let pos = iter
                    .next()
                    .unwrap_or_else(|| panic!("Unable to find starting hash {operation:?}"));
                if iter.next().is_some() {
                    panic!("Hash {operation:?} is ambiguous.");
                }
                branch_operations[pos].hash
            };
            results.push(hash);
        }
    }
    results
}

// The url-parse crate doesn't know about file-based urls, so we need to provide it with a
// custom set of port mappings
fn port_mappings() -> HashMap<&'static str, (u32, &'static str)> {
    HashMap::from([
        ("file", (0, "file")),
        ("https", (443, "Hypertext Transfer Protocol Secure")),
        ("http", (80, "Hypertext Transfer Protocol")),
        ("s3", (443, "Amazon S3 File Transfer Protocol")),
    ])
}

fn connect_file_remote(remote_url: &str) -> Result<(PathBuf, Connection), RemoteOperationError> {
    let parsed_url = Parser::new(Some(port_mappings()))
        .parse(remote_url)
        .map_err(|_| RemoteOperationError::InvalidRemoteUrl(remote_url.to_string()))?;

    let scheme = parsed_url
        .scheme
        .ok_or_else(|| RemoteOperationError::InvalidRemoteUrl(remote_url.to_string()))?;

    if scheme != "file" {
        return Err(RemoteOperationError::UnsupportedRemoteScheme(
            scheme,
            remote_url.to_string(),
        ));
    }

    let remote_path = PathBuf::from(remote_url.strip_prefix("file://").unwrap());

    let op_db_path = remote_path.join(".gen").join("gen.db");
    let remote_op_conn = if op_db_path.exists() {
        get_operation_connection(Some(op_db_path))
            .map_err(|e| RemoteOperationError::IOError(std::io::Error::other(e.to_string())))?
    } else {
        return Err(RemoteOperationError::DoesNotExist(remote_url.to_string()));
    };

    Ok((remote_path, remote_op_conn))
}

fn apply_operations_to_remote(
    remote_op_conn: &Connection,
    operations: &[ManifestOperation],
    remote_path: &FilePath,
) -> Result<(), RemoteOperationError> {
    let gen_dir = get_gen_dir().ok_or_else(|| {
        RemoteOperationError::IOError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Gen directory not found",
        ))
    })?;
    let gen_dir = PathBuf::from(gen_dir);

    for manifest_op in operations {
        let operation = &manifest_op.operation;
        let op_hash = &operation.hash;

        let changeset_src = operation.get_changeset_path();

        let changeset_dst = remote_path
            .join(".gen")
            .join("changeset")
            .join(op_hash.to_string());
        fs::create_dir_all(&changeset_dst)?;

        fs::copy(&changeset_src, changeset_dst.join("changeset")).map_err(|_| {
            RemoteOperationError::FileTransferError(
                "changeset".to_string(),
                changeset_src.to_string_lossy().to_string(),
                changeset_dst.to_string_lossy().to_string(),
            )
        })?;

        let dependencies_src = operation.get_changeset_dependencies_path();
        fs::copy(&dependencies_src, changeset_dst.join("dependencies")).map_err(|_| {
            RemoteOperationError::FileTransferError(
                "dependencies".to_string(),
                dependencies_src.to_string_lossy().to_string(),
                changeset_dst.to_string_lossy().to_string(),
            )
        })?;

        for file_addition in &manifest_op.file_additions {
            let src_path = FilePath::new(&gen_dir)
                .parent()
                .unwrap()
                .join(&file_addition.file_path);
            let dst_path = remote_path.join(&file_addition.file_path);

            // we do a conditional transfer because users may be making tmp files to just add nodes/etc. and don't actually
            // care about keeping those files around
            if src_path.exists() {
                // Create parent directories if needed on the remote
                if let Some(parent) = dst_path.parent() {
                    fs::create_dir_all(parent)?;
                }

                fs::copy(&src_path, &dst_path).map_err(|_| {
                    RemoteOperationError::FileTransferError(
                        file_addition.file_path.clone(),
                        src_path.to_string_lossy().to_string(),
                        dst_path.to_string_lossy().to_string(),
                    )
                })?;
            }
        }

        let changeset = operation.get_changeset();
        let dependencies = operation.get_changeset_dependencies();

        let remote_data_db = remote_path.join(changeset.db_path);
        let new_db = !remote_data_db.exists();
        let remote_data_conn = &get_connection(&remote_data_db)?;
        if new_db {
            track_database(remote_data_conn, remote_op_conn)?;
        };
        let remote_db_uuid = get_db_uuid(remote_data_conn);
        remote_data_conn.execute("BEGIN TRANSACTION", [])?;
        match apply_changeset(remote_data_conn, &changeset.changes, &dependencies) {
            Ok(_) => {
                remote_data_conn.execute("COMMIT TRANSACTION", [])?;
            }
            Err(e) => {
                remote_data_conn.execute("ROLLBACK TRANSACTION", [])?;
                return Err(RemoteOperationError::IOError(std::io::Error::other(
                    format!("Failed to apply changeset for operation {}: {}", op_hash, e),
                )));
            }
        }

        remote_op_conn.execute("BEGIN TRANSACTION", [])?;
        match Operation::create_without_tracking(
            remote_op_conn,
            &operation.hash,
            &operation.change_type,
            operation.parent_hash,
            Some(operation.created_on),
        ) {
            Ok(_) => {
                // Add file associations for this operation, these aren't tracked in changesets atm
                for file_addition in &manifest_op.file_additions {
                    let remote_file_addition = FileAddition::get_or_create(
                        remote_op_conn,
                        &file_addition.file_path,
                        file_addition.file_type,
                        None,
                    )?;
                    Operation::add_file(remote_op_conn, &operation.hash, &remote_file_addition.id)?;
                }
                Operation::add_database(remote_op_conn, &operation.hash, &remote_db_uuid)?;

                remote_op_conn.execute("COMMIT TRANSACTION", [])?;
            }
            Err(e) => {
                remote_op_conn.execute("ROLLBACK TRANSACTION", [])?;
                return Err(RemoteOperationError::IOError(std::io::Error::other(
                    format!(
                        "Failed to save operation {} to remote database: {}",
                        op_hash, e
                    ),
                )));
            }
        }
    }

    Ok(())
}

fn push_to_file_remote(
    local_op_conn: &Connection,
    remote_url: &str,
    branch_name: &str,
) -> Result<(), RemoteOperationError> {
    let generator = ManifestGenerator::new(local_op_conn);
    let current_branch = Branch::get_by_name(local_op_conn, branch_name).ok_or_else(|| {
        RemoteOperationError::IOError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Current branch not found",
        ))
    })?;

    let current_hash = current_branch
        .current_operation_hash
        .ok_or(RemoteOperationError::NoOperations)?;

    let local_manifest = generator.generate_manifest(&current_branch.name, Some(&current_hash))?;

    let (remote_path, ref remote_op_conn) = connect_file_remote(remote_url)?;

    let remote_branch = Branch::get_or_create(remote_op_conn, branch_name);
    let remote_generator = ManifestGenerator::new(remote_op_conn);
    let remote_manifest = if let Some(hash) = remote_branch.current_operation_hash {
        Some(remote_generator.generate_manifest(branch_name, Some(&hash))?)
    } else {
        None
    };

    let diff = if let Some(remote_manifest) = remote_manifest {
        ManifestComparer::diff_manifests(&local_manifest, &remote_manifest)?
    } else {
        // Empty remote - all local operations are missing
        ManifestDiff {
            missing_in_manifest2: local_manifest.operations.clone(),
            missing_in_manifest1: vec![],
        }
    };

    if !diff.missing_in_manifest1.is_empty() {
        return Err(RemoteOperationError::RemoteBranchAhead);
    }

    if !diff.missing_in_manifest2.is_empty() {
        apply_operations_to_remote(remote_op_conn, &diff.missing_in_manifest2, &remote_path)?;

        // Update remote branch to point to the latest operation
        let latest_op_hash = diff
            .missing_in_manifest2
            .last()
            .map(|op| op.operation.hash)
            .unwrap_or(current_hash);

        Branch::set_current_operation(remote_op_conn, remote_branch.id, &latest_op_hash);
        let current_state = OperationState::get_current_branch(remote_op_conn);
        if let Some(current_branch) = current_state
            && current_branch == remote_branch.id
        {
            OperationState::set_operation(remote_op_conn, &latest_op_hash);
        }
    }

    Ok(())
}

// Pushes the current state of the local repo and branch to the corresponding remote repo and branch
pub fn push(operation_conn: &Connection, remote: Option<&str>) -> Result<(), RemoteOperationError> {
    let remote_name = &remote
        .map(str::to_owned)
        .or_else(|| Defaults::get_default_remote(operation_conn))
        .ok_or(RemoteOperationError::RemoteUrlNotSet)?;
    let remote = Remote::get_by_name(operation_conn, remote_name)?;
    let remote_url = remote.url;

    let parsed_url = Parser::new(Some(port_mappings())).parse(&remote_url);

    match parsed_url {
        Ok(result) => {
            if let Some(scheme) = result.scheme {
                if scheme == "file" {
                    let current_branch_id = OperationState::get_current_branch(operation_conn)
                        .ok_or_else(|| {
                            RemoteOperationError::IOError(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                "No current branch set",
                            ))
                        })?;
                    let branch = Branch::get_by_id(operation_conn, current_branch_id).unwrap();

                    push_to_file_remote(operation_conn, &remote_url, &branch.name)
                } else {
                    let generator = ManifestGenerator::new(operation_conn);
                    let current_branch_id =
                        OperationState::get_current_branch(operation_conn).unwrap();
                    let current_branch =
                        Branch::get_by_id(operation_conn, current_branch_id).unwrap();
                    let current_hash = if let Some(h) = current_branch.current_operation_hash {
                        h
                    } else {
                        Err(RemoteOperationError::NoOperations)?
                    };
                    let manifest =
                        generator.generate_manifest(&current_branch.name, Some(&current_hash))?;
                    let diff = send_manifest_to_remote(remote_name, &remote_url, &manifest)?;

                    let auth_tokens = load_tokens(remote_name).map_err(|e| {
                        RemoteOperationError::AuthError(format!(
                            "Unable to load tokens: {e}. Did you login?"
                        ))
                    })?;
                    let manifest_url = {
                        let mut url = remote_url.trim_end_matches('/').to_string();
                        url.push_str("/manifest/operation");
                        url
                    };
                    for manifest_operation in diff.missing_in_manifest2.iter() {
                        let client = Client::new();
                        let op = Operation::get_by_id(
                            operation_conn,
                            &manifest_operation.operation.hash,
                        )
                        .unwrap();
                        let cs_path = op.get_changeset_path();
                        let dep_path = op.get_changeset_dependencies_path();

                        let mut builder = capnp::message::Builder::new_default();
                        let mut manifest_op_capnp = builder.init_root::<gen_models::gen_models_capnp::manifest_operation::Builder>();
                        manifest_operation.write_capnp(&mut manifest_op_capnp);

                        let mut encoded = Vec::new();
                        capnp::serialize_packed::write_message(&mut encoded, &builder).unwrap();

                        let part =
                            multipart::Part::bytes(encoded).mime_str("application/octet-stream")?;

                        let mut form = multipart::Form::new()
                            .part("manifest_operation", part)
                            .file("files", cs_path)
                            .unwrap()
                            .file("files", dep_path)
                            .unwrap();

                        let operation_files =
                            FileAddition::get_files_for_operation(operation_conn, &op.hash);
                        for op_file in operation_files {
                            form = form
                                .file(
                                    "assets",
                                    FilePath::new(".gen")
                                        .join("assets")
                                        .join(op_file.asset_filename()),
                                )
                                .unwrap();
                        }

                        form = form.text("branch", current_branch.name.clone());

                        let response = client
                            .post(&manifest_url)
                            .bearer_auth(auth_tokens.jwt.clone())
                            .multipart(form)
                            .send()?;
                        println!("response: {}", response.text()?);
                    }
                    Ok(())
                }
            } else {
                Err(RemoteOperationError::InvalidRemoteUrl(
                    remote_url.to_string(),
                ))
            }
        }
        Err(_) => Err(RemoteOperationError::InvalidRemoteUrl(
            remote_url.to_string(),
        )),
    }
}

pub fn pull(operation_conn: &Connection, remote: Option<&str>) -> Result<(), RemoteOperationError> {
    let remote_name = &remote
        .map(str::to_owned)
        .or_else(|| Defaults::get_default_remote(operation_conn))
        .ok_or(RemoteOperationError::RemoteUrlNotSet)?;
    let remote = Remote::get_by_name(operation_conn, remote_name)?;
    let remote_url = remote.url;

    let current_branch_id = OperationState::get_current_branch(operation_conn)
        .ok_or_else(|| RemoteOperationError::BranchError("No current branch set".to_string()))?;
    let branch = Branch::get_by_id(operation_conn, current_branch_id).ok_or_else(|| {
        RemoteOperationError::DoesNotExist(format!(
            "Branch {current_branch_id} not found in database."
        ))
    })?;

    let parsed_url = Parser::new(Some(port_mappings())).parse(&remote_url);
    match parsed_url {
        Ok(result) => {
            if let Some(scheme) = result.scheme {
                if scheme == "file" {
                    pull_from_file_remote(operation_conn, &remote_url, &branch)
                } else {
                    pull_from_remote_server(operation_conn, remote_name, &remote_url, &branch)
                }
            } else {
                Err(RemoteOperationError::InvalidRemoteUrl(
                    remote_url.to_string(),
                ))
            }
        }
        Err(_) => Err(RemoteOperationError::InvalidRemoteUrl(
            remote_url.to_string(),
        )),
    }
}

fn pull_from_file_remote(
    operation_conn: &Connection,
    remote_url: &str,
    current_branch: &Branch,
) -> Result<(), RemoteOperationError> {
    let generator = ManifestGenerator::new(operation_conn);
    let local_manifest = generator.generate_manifest(
        &current_branch.name,
        current_branch.current_operation_hash.as_ref(),
    )?;

    let (remote_path, ref remote_op_conn) = connect_file_remote(remote_url)?;
    let remote_branch =
        Branch::get_by_name(remote_op_conn, &current_branch.name).ok_or_else(|| {
            RemoteOperationError::DoesNotExist(format!(
                "Branch {} not found on remote",
                current_branch.name
            ))
        })?;

    let diff = if let Some(remote_hash) = remote_branch.current_operation_hash {
        let remote_manifest = ManifestGenerator::new(remote_op_conn)
            .generate_manifest(&current_branch.name, Some(&remote_hash))?;
        ManifestComparer::diff_manifests(&local_manifest, &remote_manifest)?
    } else {
        // There's nothing in the remote, so just make it empty since we have nothing to pull.
        ManifestDiff {
            missing_in_manifest2: vec![],
            missing_in_manifest1: vec![],
        }
    };

    if diff.missing_in_manifest1.is_empty() {
        return Ok(());
    }

    let repo_root = get_repo_root_path()?;
    for manifest_operation in diff.missing_in_manifest1.iter() {
        copy_operation_from_remote_fs(
            manifest_operation,
            remote_path.as_path(),
            repo_root.as_path(),
        )?;
        ingest_manifest_operation(operation_conn, manifest_operation, repo_root.as_path())?;
        OperationState::set_operation(operation_conn, &manifest_operation.operation.hash);
        Branch::set_current_operation(
            operation_conn,
            current_branch.id,
            &manifest_operation.operation.hash,
        );
    }

    Ok(())
}

fn pull_from_remote_server(
    operation_conn: &Connection,
    remote_name: &str,
    remote_url: &str,
    current_branch: &Branch,
) -> Result<(), RemoteOperationError> {
    let generator = ManifestGenerator::new(operation_conn);
    let manifest = generator.generate_manifest(
        &current_branch.name,
        current_branch.current_operation_hash.as_ref(),
    )?;
    let diff = send_manifest_to_remote(remote_name, remote_url, &manifest)?;

    if diff.missing_in_manifest1.is_empty() {
        return Ok(());
    }

    let auth_tokens = load_tokens(remote_name).map_err(|e| {
        RemoteOperationError::AuthError(format!("Unable to load tokens: {e}. Did you login?"))
    })?;
    let manifest_url = {
        let mut url = remote_url.trim_end_matches('/').to_string();
        url.push_str("/manifest/operation");
        url
    };
    let client = Client::new();
    let repo_root = get_repo_root_path()?;

    for manifest_operation in diff.missing_in_manifest1.iter() {
        download_remote_operation_assets(
            &client,
            &auth_tokens.jwt,
            &manifest_url,
            manifest_operation,
            repo_root.as_path(),
        )?;
        ingest_manifest_operation(operation_conn, manifest_operation, repo_root.as_path())?;
        OperationState::set_operation(operation_conn, &manifest_operation.operation.hash);
        Branch::set_current_operation(
            operation_conn,
            current_branch.id,
            &manifest_operation.operation.hash,
        );
    }

    Ok(())
}

fn ingest_manifest_operation(
    operation_conn: &Connection,
    manifest_operation: &ManifestOperation,
    repo_root: &FilePath,
) -> Result<(), RemoteOperationError> {
    let operation = &manifest_operation.operation;
    let changeset = operation.get_changeset();
    let dependencies = operation.get_changeset_dependencies();

    let data_db_path = repo_root.join(&changeset.db_path);
    let new_db = !data_db_path.exists();
    let data_conn = &get_connection(&data_db_path)?;
    if new_db {
        track_database(data_conn, operation_conn)?;
    }
    let db_uuid = get_db_uuid(data_conn);

    data_conn.execute("BEGIN TRANSACTION", [])?;
    match apply_changeset(data_conn, &changeset.changes, &dependencies) {
        Ok(_) => {
            data_conn.execute("COMMIT TRANSACTION", [])?;
        }
        Err(e) => {
            data_conn.execute("ROLLBACK TRANSACTION", [])?;
            return Err(RemoteOperationError::IOError(std::io::Error::other(
                format!(
                    "Failed to apply changeset for operation {}: {}",
                    operation.hash, e
                ),
            )));
        }
    }

    operation_conn.execute("BEGIN TRANSACTION", [])?;
    match Operation::create_without_tracking(
        operation_conn,
        &operation.hash,
        &operation.change_type,
        operation.parent_hash,
        Some(operation.created_on),
    ) {
        Ok(_) => {
            for file_addition in &manifest_operation.file_additions {
                let local_file_addition = FileAddition::get_or_create(
                    operation_conn,
                    &file_addition.file_path,
                    file_addition.file_type,
                    None,
                )?;
                Operation::add_file(operation_conn, &operation.hash, &local_file_addition.id)?;
            }
            Operation::add_database(operation_conn, &operation.hash, &db_uuid)?;
            operation_conn.execute("COMMIT TRANSACTION", [])?;
        }
        Err(e) => {
            operation_conn.execute("ROLLBACK TRANSACTION", [])?;
            return Err(RemoteOperationError::IOError(std::io::Error::other(
                format!(
                    "Failed to record operation {} locally: {}",
                    operation.hash, e
                ),
            )));
        }
    }

    Ok(())
}

fn copy_operation_from_remote_fs(
    manifest_operation: &ManifestOperation,
    remote_path: &FilePath,
    repo_root: &FilePath,
) -> Result<(), RemoteOperationError> {
    let op_hash = format!("{}", manifest_operation.operation.hash);
    let remote_changeset_dir = remote_path.join(".gen").join("changeset").join(&op_hash);
    let remote_changeset_src = remote_changeset_dir.join("changeset");
    let remote_dependencies_src = remote_changeset_dir.join("dependencies");

    let local_changeset_dst = manifest_operation.operation.get_changeset_path();
    if !remote_changeset_src.exists() {
        return Err(RemoteOperationError::FileTransferError(
            "changeset".to_string(),
            remote_changeset_src.to_string_lossy().to_string(),
            local_changeset_dst.to_string_lossy().to_string(),
        ));
    }
    fs::copy(&remote_changeset_src, &local_changeset_dst).map_err(|_| {
        RemoteOperationError::FileTransferError(
            "changeset".to_string(),
            remote_changeset_src.to_string_lossy().to_string(),
            local_changeset_dst.to_string_lossy().to_string(),
        )
    })?;

    let local_dependencies_dst = manifest_operation
        .operation
        .get_changeset_dependencies_path();
    if !remote_dependencies_src.exists() {
        return Err(RemoteOperationError::FileTransferError(
            "dependencies".to_string(),
            remote_dependencies_src.to_string_lossy().to_string(),
            local_dependencies_dst.to_string_lossy().to_string(),
        ));
    }
    fs::copy(&remote_dependencies_src, &local_dependencies_dst).map_err(|_| {
        RemoteOperationError::FileTransferError(
            "dependencies".to_string(),
            remote_dependencies_src.to_string_lossy().to_string(),
            local_dependencies_dst.to_string_lossy().to_string(),
        )
    })?;

    for file_addition in &manifest_operation.file_additions {
        let src_path = remote_path.join(&file_addition.file_path);
        let dst_path = repo_root.join(&file_addition.file_path);
        if src_path.exists() {
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&src_path, &dst_path).map_err(|_| {
                RemoteOperationError::FileTransferError(
                    file_addition.file_path.clone(),
                    src_path.to_string_lossy().to_string(),
                    dst_path.to_string_lossy().to_string(),
                )
            })?;
        }
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct RemoteOperationAssetResponse {
    changeset: String,
    dependencies: String,
    #[serde(default)]
    files: Vec<RemoteFileAsset>,
}

#[derive(Debug, Deserialize)]
struct RemoteFileAsset {
    asset_path: String,
    file_path: String,
    url: String,
}

fn download_remote_operation_assets(
    client: &Client,
    auth_token: &str,
    endpoint: &str,
    manifest_operation: &ManifestOperation,
    repo_root: &FilePath,
) -> Result<(), RemoteOperationError> {
    let url = format!("{endpoint}/{}", manifest_operation.operation.hash);
    let response = client.get(url).bearer_auth(auth_token).send()?;
    let status = response.status();
    if !status.is_success() {
        return Err(RemoteOperationError::FileTransferError(
            "manifest_operation".to_string(),
            endpoint.to_string(),
            format!("HTTP {status} {r:?}", r = response.bytes().unwrap()),
        ));
    }

    let asset_response: RemoteOperationAssetResponse = response.json()?;
    let changeset_dst = manifest_operation.operation.get_changeset_path();
    download_binary(
        client,
        &asset_response.changeset,
        changeset_dst.as_path(),
        Some(auth_token),
        "changeset",
    )?;

    let dependencies_dst = manifest_operation
        .operation
        .get_changeset_dependencies_path();
    download_binary(
        client,
        &asset_response.dependencies,
        dependencies_dst.as_path(),
        Some(auth_token),
        "dependencies",
    )?;

    let gen_dir = get_gen_dir().unwrap();
    let gen_path = FilePath::new(&gen_dir);
    for file in asset_response.files {
        let destination = gen_path.join("assets").join(&file.asset_path);
        let user_destination = repo_root.join(&file.file_path);
        if !destination.exists() {
            download_binary(
                client,
                &file.url,
                destination.as_path(),
                Some(auth_token),
                &file.file_path,
            )?;
        }
        if !user_destination.exists() {
            std::fs::copy(destination.as_path(), user_destination.as_path())?;
        }
    }

    Ok(())
}

fn download_binary(
    client: &Client,
    url: &str,
    dest: &FilePath,
    bearer_token: Option<&str>,
    resource_name: &str,
) -> Result<(), RemoteOperationError> {
    let mut request = client.get(url);
    if let Some(token) = bearer_token {
        request = request.bearer_auth(token);
    }

    let mut response = request.send()?;
    let status = response.status();
    if !status.is_success() {
        return Err(RemoteOperationError::FileTransferError(
            resource_name.to_string(),
            url.to_string(),
            format!("{} (HTTP {status})", dest.to_string_lossy()),
        ));
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(dest)?;
    copy(&mut response, &mut file)?;
    Ok(())
}

fn send_manifest_to_remote(
    remote_name: &str,
    remote_url: &str,
    manifest: &gen_models::manifest::Manifest,
) -> Result<ManifestDiff, RemoteOperationError> {
    let auth_tokens = load_tokens(remote_name).map_err(|e| {
        RemoteOperationError::AuthError(format!("Unable to load auth token: {e}. Did you login?"))
    })?;

    let client = Client::new();
    let manifest_url = {
        let mut url = remote_url.trim_end_matches('/').to_string();
        url.push_str("/manifest");
        url
    };

    let mut builder = capnp::message::Builder::new_default();
    let mut manifest_capnp = builder.init_root::<gen_models::gen_models_capnp::manifest::Builder>();
    manifest.write_capnp(&mut manifest_capnp);

    let mut buf = Vec::new();
    capnp::serialize_packed::write_message(&mut buf, &builder).unwrap();

    let response = client
        .post(manifest_url)
        .bearer_auth(auth_tokens.jwt)
        .header("Content-Type", "application/octet-stream")
        .body(buf)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        return Err(RemoteOperationError::FileTransferError(
            "manifest".to_string(),
            "local".to_string(),
            format!("{remote_url} - {status}"),
        ));
    }

    println!("Manifest sent successfully to {remote_url}");
    Ok(response.json()?)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        env,
        path::{Path, PathBuf},
    };

    use gen_core::config::{get_or_create_gen_dir, set_base_dir};
    use gen_models::{
        block_group::BlockGroup,
        block_group_edge::BlockGroupEdge,
        edge::Edge,
        file_types::FileTypes,
        node::Node,
        operations::{Branch, Operation, OperationState},
        sample::Sample,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{create_operation, get_connection, get_operation_connection, setup_gen_dir},
        track_database,
        updates::vcf::update_with_vcf,
    };

    #[cfg(test)]
    mod merge {
        use super::*;
        use crate::{operation_management::checkout, track_database};

        #[test]
        fn test_merges() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();

            track_database(conn, op_conn).unwrap();

            let op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-1"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-2"),
            );

            let branch_1 = Branch::get_or_create(op_conn, "branch-1");
            let branch_2 = Branch::get_or_create(op_conn, "branch-2");
            OperationState::set_branch(op_conn, "branch-1");
            let op_3 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "vcf_addition",
                HashId::convert_str("op-3"),
            );
            let op_4 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "vcf_addition",
                HashId::convert_str("op-4"),
            );
            checkout(Some(conn), op_conn, &Some("branch-2".to_string()), None).unwrap();
            let op_5 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "vcf_addition",
                HashId::convert_str("op-5"),
            );
            let op_6 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "vcf_addition",
                HashId::convert_str("op-6"),
            );

            checkout(Some(conn), op_conn, &Some("branch-1".to_string()), None).unwrap();
            let new_operations = merge(Some(conn), op_conn, branch_1.id, branch_2.id, "merge-test")
                .unwrap()
                .iter()
                .map(|op: &Operation| op.hash)
                .collect::<Vec<_>>();

            let b1_ops = Branch::get_operations(op_conn, branch_1.id)
                .iter()
                .map(|f| f.hash)
                .collect::<Vec<_>>();

            let b2_ops = Branch::get_operations(op_conn, branch_2.id)
                .iter()
                .map(|f| f.hash)
                .collect::<Vec<_>>();

            assert_eq!(
                b1_ops,
                vec![op_1.hash, op_2.hash, op_3.hash, op_4.hash]
                    .into_iter()
                    .chain(new_operations.into_iter())
                    .collect::<Vec<_>>()
            );
            assert_eq!(b2_ops, vec![op_1.hash, op_2.hash, op_5.hash, op_6.hash]);
        }
    }

    #[cfg(test)]
    mod parse_patch_operations {
        use super::*;
        use crate::{operation_management::parse_patch_operations, track_database};

        #[test]
        fn test_head_shorthand() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();

            track_database(conn, op_conn).unwrap();

            let _op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-1"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-2"),
            );
            let op_3 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "vcf_addition",
                HashId::convert_str("op-3"),
            );

            let branch = Branch::get_by_name(op_conn, "main").unwrap();
            let ops = Branch::get_operations(op_conn, branch.id);
            assert_eq!(
                parse_patch_operations(
                    &ops,
                    &branch.current_operation_hash.unwrap(),
                    "HEAD~1..HEAD"
                ),
                vec![op_2.hash, op_3.hash]
            );
        }

        #[test]
        fn test_hash_shorthand() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();

            track_database(conn, op_conn).unwrap();

            let _op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-1-abc-123"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-2-abc-123"),
            );
            let op_3 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "vcf_addition",
                HashId::convert_str("op-3-abc-13"),
            );

            let branch = Branch::get_by_name(op_conn, "main").unwrap();
            let ops = Branch::get_operations(op_conn, branch.id);
            let head_hash = branch.current_operation_hash.unwrap();
            assert_eq!(
                parse_patch_operations(
                    &ops,
                    &head_hash,
                    &format!(
                        "{op_2}..{op_3}",
                        op_2 = &format!("{}", op_2.hash)[..6],
                        op_3 = &format!("{}", op_3.hash)[..6]
                    )
                ),
                vec![op_2.hash, op_3.hash]
            );

            assert_eq!(
                parse_patch_operations(&ops, &head_hash, &format!("{}", op_2.hash)[..6]),
                vec![op_2.hash]
            );
        }

        #[test]
        #[should_panic(expected = "Start hash 587 is ambiguous.")]
        fn test_error_on_ambiguous_hash_shorthand() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();

            track_database(conn, op_conn).unwrap();

            let _op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-1-abc-123"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "fasta_addition",
                HashId::convert_str("op-2-abc-123"),
            );
            let op_3 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::None,
                "vcf_addition",
                // some random string i found to collide with prefix of above
                HashId::convert_str("AXf5SuLvAM"),
            );

            let branch = Branch::get_by_name(op_conn, "main").unwrap();
            let ops = Branch::get_operations(op_conn, branch.id);
            assert_eq!(
                parse_patch_operations(
                    &ops,
                    &branch.current_operation_hash.unwrap(),
                    &format!(
                        "{op_2}..{op_3}",
                        op_2 = &format!("{}", op_2.hash)[..3],
                        op_3 = &format!("{}", op_3.hash)[..3]
                    )
                ),
                vec![op_2.hash, op_3.hash]
            );
        }
    }

    #[test]
    fn test_round_trip() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();
        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();
        let block_group_count =
            BlockGroup::query(conn, "select * from block_groups", rusqlite::params!()).len();
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(block_group_count, 1);
        assert_eq!(edge_count, 2);
        assert_eq!(block_group_edge_count, 2);
        assert_eq!(node_count, 3);
        assert_eq!(sample_count, 0);
        assert_eq!(op_count, 1);
        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();
        let block_group_count =
            BlockGroup::query(conn, "select * from block_groups", rusqlite::params!()).len();
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        // NOTE: 3 block groups get created with the update from vcf, corresponding to the unknown, G1, and foo samples
        assert_eq!(block_group_count, 4);
        // NOTE: The edge count is 5 because of the following:
        // * 1 edge from the source node to the node created by the fasta import
        // * 1 edge from the node created by the fasta import to the sink node
        // * 1 edge representing the deletion
        // * 2 edges to and from the node representing the second alt sequence
        assert_eq!(edge_count, 5);
        // NOTE: The block group edge count is 16 because of the following:
        // * 4 edges (one per block group) from the virtual source node
        // * 4 edges (one per block group) to the virtual sink node
        // * 2 block group edges for the G1 sample (1 edge for the first alt sequence, a deletion, with both the 0 and 1 chromosome indices for those edges, 1 * 2 = 2)
        // * 6 block group edges for the foo sample (1 edge to and from the node representing the
        // first alt sequence, with both the 0 and 1 chromosome indices for those edges, 1 * 2 = 2;
        // 2 edges to and from the node representing the second alt sequence, with both the 0 and 1
        // chromosome indices for those edges, 2 * 2 = 4)
        // 4 + 4 + 2 + 6 = 16
        assert_eq!(block_group_edge_count, 16);
        // NOTE: The node count is 5:
        // * 2 source and sink nodes
        // * 1 node created by the initial fasta import
        // * 2 nodes created by the VCF update.  See above explanation of edge count for more details.
        assert_eq!(node_count, 5);
        assert_eq!(sample_count, 3);
        assert_eq!(op_count, 2);

        // revert back to state 1 where vcf samples and blockpaths do not exist

        let current_op = Operation::get_by_id(
            operation_conn,
            &OperationState::get_operation(operation_conn).unwrap(),
        )
        .expect("Hash does not exist.");
        let changeset = current_op.get_changeset();
        revert_changeset(conn, &changeset.changes).unwrap();

        let block_group_count =
            BlockGroup::query(conn, "select * from block_groups", rusqlite::params!()).len();
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(block_group_count, 1);
        assert_eq!(edge_count, 2);
        assert_eq!(block_group_edge_count, 2);
        assert_eq!(node_count, 3);
        assert_eq!(sample_count, 0);
        assert_eq!(op_count, 2);

        let op = Operation::get_by_id(
            operation_conn,
            &OperationState::get_operation(operation_conn).unwrap(),
        )
        .unwrap();
        let changeset = op.get_changeset();
        let dependencies = op.get_changeset_dependencies();

        apply_changeset(conn, &changeset.changes, &dependencies).unwrap();
        let block_group_count =
            BlockGroup::query(conn, "select * from block_groups", rusqlite::params!()).len();
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(block_group_count, 4);
        assert_eq!(edge_count, 5);
        assert_eq!(block_group_edge_count, 16);
        assert_eq!(node_count, 5);
        assert_eq!(sample_count, 3);
        assert_eq!(op_count, 2);
    }

    #[test]
    fn test_cross_branch_patch() {
        setup_gen_dir();
        let fasta_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let vcf2_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple2.vcf");
        let conn = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();

        let _op_1 = import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();

        Branch::get_or_create(operation_conn, "branch-1");
        Branch::get_or_create(operation_conn, "branch-2");
        checkout(
            Some(conn),
            operation_conn,
            &Some("branch-1".to_string()),
            None,
        )
        .unwrap();

        let op_2 = update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();

        let foo_bg_id = BlockGroup::get_id(&collection, Some("foo"), "m123");
        let patch_1_seqs =
            HashSet::from_iter(vec!["ATCATCGATCGATCGATCGGGAACACACAGAGA".to_string()]);

        assert_eq!(
            BlockGroup::get_all_sequences(conn, &foo_bg_id, false),
            patch_1_seqs
        );
        assert_eq!(
            BlockGroup::query(conn, "select * from block_groups;", rusqlite::params!())
                .iter()
                .map(|v| v.sample_name.clone().unwrap_or("".to_string()))
                .collect::<Vec<String>>(),
            vec![
                "".to_string(),
                "unknown".to_string(),
                "G1".to_string(),
                "foo".to_string()
            ]
        );

        checkout(
            Some(conn),
            operation_conn,
            &Some("branch-2".to_string()),
            None,
        )
        .unwrap();
        let _op_3 = update_with_vcf(
            &vcf2_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        );

        let foo_bg_id = BlockGroup::get_id(&collection, Some("foo"), "m123");
        let patch_2_seqs = HashSet::from_iter(vec!["ATCGATCGATCGAGATCGGGAACACACAGAGA".to_string()]);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &foo_bg_id, false),
            patch_2_seqs
        );
        assert_ne!(patch_1_seqs, patch_2_seqs);
        assert_eq!(
            BlockGroup::query(conn, "select * from block_groups;", rusqlite::params!())
                .iter()
                .map(|v| v.sample_name.clone().unwrap_or("".to_string()))
                .collect::<Vec<String>>(),
            vec!["".to_string(), "foo".to_string()]
        );

        // apply changes from branch-1, it will be operation id 2
        apply(Some(conn), operation_conn, &op_2.hash, None).unwrap();

        let foo_bg_id = BlockGroup::get_id(&collection, Some("foo"), "m123");
        let patch_2_seqs = HashSet::from_iter(vec!["ATCATCGATCGAGATCGGGAACACACAGAGA".to_string()]);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &foo_bg_id, false),
            patch_2_seqs
        );
        assert_eq!(
            BlockGroup::query(conn, "select * from block_groups;", rusqlite::params!())
                .iter()
                .map(|v| v.sample_name.clone().unwrap_or("".to_string()))
                .collect::<HashSet<String>>(),
            HashSet::from_iter([
                "".to_string(),
                "foo".to_string(),
                "unknown".to_string(),
                "G1".to_string()
            ])
        );

        let unknown_bg_id = BlockGroup::get_id(&collection, Some("unknown"), "m123");
        let unknown_seqs =
            HashSet::from_iter(vec!["ATCATCGATAGACGATCGATCGGGAACACACAGAGA".to_string()]);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &unknown_bg_id, false),
            unknown_seqs
        );
        assert_ne!(unknown_seqs, patch_2_seqs);
    }

    #[test]
    fn test_branch_movement() {
        setup_gen_dir();
        let fasta_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let vcf2_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple2.vcf");
        let conn = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();
        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(edge_count, 2);
        assert_eq!(block_group_edge_count, 2);
        assert_eq!(node_count, 3);
        assert_eq!(sample_count, 0);
        assert_eq!(op_count, 1);

        let branch_1 = Branch::get_or_create(operation_conn, "branch_1");

        let branch_2 = Branch::get_or_create(operation_conn, "branch_2");

        OperationState::set_branch(operation_conn, "branch_1");
        assert_eq!(
            OperationState::get_current_branch(operation_conn).unwrap(),
            branch_1.id
        );

        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(edge_count, 5);
        assert_eq!(block_group_edge_count, 16);
        assert_eq!(node_count, 5);
        assert_eq!(sample_count, 3);
        assert_eq!(op_count, 2);

        // checkout branch 2
        checkout(
            Some(conn),
            operation_conn,
            &Some("branch_2".to_string()),
            None,
        )
        .unwrap();

        assert_eq!(
            OperationState::get_current_branch(operation_conn).unwrap(),
            branch_2.id
        );

        // ensure branch 1 operations have been undone
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(edge_count, 2);
        assert_eq!(block_group_edge_count, 2);
        assert_eq!(node_count, 3);
        assert_eq!(sample_count, 0);
        assert_eq!(op_count, 2);

        // apply vcf2
        update_with_vcf(
            &vcf2_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();
        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(edge_count, 3);
        assert_eq!(block_group_edge_count, 6);
        assert_eq!(node_count, 4);
        assert_eq!(sample_count, 1);
        assert_eq!(op_count, 3);

        // migrate to branch 1 again
        checkout(
            Some(conn),
            operation_conn,
            &Some("branch_1".to_string()),
            None,
        )
        .unwrap();
        assert_eq!(
            OperationState::get_current_branch(operation_conn).unwrap(),
            branch_1.id
        );

        let edge_count = Edge::query(conn, "select * from edges", rusqlite::params!()).len();
        let block_group_edge_count =
            BlockGroupEdge::query(conn, "select * from block_group_edges", rusqlite::params!())
                .len();
        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        let sample_count = Sample::query(conn, "select * from samples", rusqlite::params!()).len();
        let op_count = Operation::query(
            operation_conn,
            "select * from operations",
            rusqlite::params!(),
        )
        .len();
        assert_eq!(edge_count, 5);
        assert_eq!(block_group_edge_count, 16);
        assert_eq!(node_count, 5);
        assert_eq!(sample_count, 3);
        assert_eq!(op_count, 3);
    }

    #[test]
    fn test_reset_with_branches() {
        // Our setup is like this:
        //          -> 3 -> 4 -> 5 -> 10  branch a
        //        /                \
        //   1-> 2 -> 6 -> 7 -> 8    -> 9 branch b
        //
        // We want to make sure if we reset branch a to 3 that branch b will still show its operations
        setup_gen_dir();
        let conn = &mut get_connection(None).unwrap();
        let operation_conn = &get_operation_connection(None).unwrap();

        track_database(conn, operation_conn).unwrap();

        let main_branch = Branch::get_by_name(operation_conn, "main").unwrap();

        let op_1 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-1"),
        );
        let op_2 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-2"),
        );

        let branch_a = Branch::get_or_create(operation_conn, "branch-a");
        OperationState::set_branch(operation_conn, "branch-a");
        let op_3 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-3"),
        );
        let op_4 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-4"),
        );
        let op_5 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-5"),
        );
        OperationState::set_branch(operation_conn, "main");
        OperationState::set_operation(operation_conn, &HashId::convert_str("op-2"));
        let op_6 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-6"),
        );
        let op_7 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-7"),
        );
        let op_8 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-8"),
        );
        OperationState::set_branch(operation_conn, "branch-a");
        OperationState::set_operation(operation_conn, &HashId::convert_str("op-5"));
        let branch_b = Branch::get_or_create(operation_conn, "branch-b");
        OperationState::set_branch(operation_conn, "branch-b");
        let op_9 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-9"),
        );
        OperationState::set_branch(operation_conn, "branch-a");
        OperationState::set_operation(operation_conn, &HashId::convert_str("op-5"));
        let op_10 = create_operation(
            conn,
            operation_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-10"),
        );

        assert_eq!(
            Branch::get_operations(operation_conn, main_branch.id)
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
            vec![op_1.hash, op_2.hash, op_6.hash, op_7.hash, op_8.hash]
        );
        assert_eq!(
            Branch::get_operations(operation_conn, branch_a.id)
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
            vec![
                op_1.hash, op_2.hash, op_3.hash, op_4.hash, op_5.hash, op_10.hash
            ]
        );
        assert_eq!(
            Branch::get_operations(operation_conn, branch_b.id)
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
            vec![
                op_1.hash, op_2.hash, op_3.hash, op_4.hash, op_5.hash, op_9.hash
            ]
        );
        reset(Some(conn), operation_conn, &HashId::convert_str("op-2")).unwrap();
        assert_eq!(
            Branch::get_operations(operation_conn, main_branch.id)
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
            vec![op_1.hash, op_2.hash, op_6.hash, op_7.hash, op_8.hash]
        );
        assert_eq!(
            Branch::get_operations(operation_conn, branch_a.id)
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
            vec![op_1.hash, op_2.hash]
        );
        assert_eq!(
            Branch::get_operations(operation_conn, branch_b.id)
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
            vec![
                op_1.hash, op_2.hash, op_3.hash, op_4.hash, op_5.hash, op_9.hash
            ]
        );
    }

    #[test]
    fn test_bifurcation_allowed_on_reset() {
        // We make a simple branch from 1 -> 2 -> 3 -> 4 and ensure we can reset to operation 2
        // and create a new operation from that point on the same branch because we reset.

        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let op_1 = create_operation(
            conn,
            op_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-1"),
        );
        let op_2 = create_operation(
            conn,
            op_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-2"),
        );
        let _op_3 = create_operation(
            conn,
            op_conn,
            "foo",
            FileTypes::None,
            "foo",
            HashId::convert_str("op-3"),
        );
        let _op_4 = create_operation(
            conn,
            op_conn,
            "fixtures/aa.fa",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-4"),
        );

        reset(Some(conn), op_conn, &HashId::convert_str("op-2")).unwrap();
        let op_5 = create_operation(
            conn,
            op_conn,
            "fixtures/aa.fa",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-5"),
        );
        assert_eq!(
            Branch::get_operations(
                op_conn,
                OperationState::get_current_branch(op_conn).unwrap()
            )
            .iter()
            .map(|op| op.hash)
            .collect::<Vec<_>>(),
            vec![op_1.hash, op_2.hash, op_5.hash]
        );
    }

    #[cfg(test)]
    mod connect_file_remote {
        use super::*;

        #[test]
        fn test_with_invalid_url() {
            let result = connect_file_remote("invalid-url");
            assert!(matches!(
                result,
                Err(RemoteOperationError::InvalidRemoteUrl(_))
            ));
        }

        #[test]
        fn test_with_unsupported_scheme() {
            let result = connect_file_remote("http://example.com/repo");
            assert!(matches!(
                result,
                Err(RemoteOperationError::UnsupportedRemoteScheme(_, _))
            ));
        }

        #[test]
        fn test_with_nonexistent_remote() {
            let temp_dir = tempdir().unwrap();
            let nonexistent_path = temp_dir.path().join("nonexistent");
            let remote_url = format!("file://{}", nonexistent_path.to_str().unwrap());

            let result = connect_file_remote(&remote_url);
            assert!(matches!(result, Err(RemoteOperationError::DoesNotExist(_))));
        }

        #[test]
        fn test_with_existing_remote() {
            let temp_dir = tempdir().unwrap().keep();
            let remote_path = &temp_dir;

            // Create .gen directory and operation database
            let gen_dir = remote_path.join(".gen");
            fs::create_dir_all(&gen_dir).unwrap();

            let op_db_path = gen_dir.join("gen.db");
            get_operation_connection(op_db_path.to_str()).unwrap();

            let remote_url = format!("file://{}", remote_path.to_str().unwrap());
            let result = connect_file_remote(&remote_url);
            assert!(result.is_ok());

            let (parsed_remote_path, _remote_op_conn) = result.unwrap();
            assert_eq!(parsed_remote_path, *remote_path);
        }
    }

    #[cfg(test)]
    mod apply_operations_to_remote {
        use gen_models::{
            collection::Collection,
            operations::{OperationFile, OperationInfo},
            session_operations::{end_operation, start_operation},
        };
        use tempfile::tempdir;

        use super::*;

        #[test]
        fn test_apply_operations_to_remote() {
            let local_gen_dir = setup_gen_dir();
            let local_dir = local_gen_dir.parent().unwrap();
            let local_conn = &get_connection(None).unwrap();
            let local_op_conn = &get_operation_connection(None).unwrap();
            track_database(local_conn, local_op_conn).unwrap();

            // Create a test collection and operation
            let _collection = Collection::create(local_conn, "test_collection");
            for i in 0..3 {
                let mut session = start_operation(local_conn);

                // Make some actual changes to trigger changeset creation
                Collection::create(local_conn, &format!("test_collection_{i}"));

                let file_path = format!("test_file_{i}.fa");
                let op_info = OperationInfo {
                    files: vec![OperationFile {
                        file_path: file_path.clone(),
                        file_type: FileTypes::Fasta,
                    }],
                    description: format!("Test operation {i}"),
                };

                // Create the file addition we're transferring
                fs::write(
                    FilePath::new(local_dir).join(file_path),
                    "test file content",
                )
                .unwrap();

                let previous_dir = env::current_dir().unwrap();
                env::set_current_dir(local_dir).unwrap();
                end_operation(
                    local_conn,
                    local_op_conn,
                    &mut session,
                    &op_info,
                    &format!("Test operation {i}"),
                    None,
                )
                .unwrap();
                env::set_current_dir(previous_dir).unwrap();
            }

            let local_main = Branch::get_by_name(local_op_conn, "main").unwrap();

            // Create remote directory structure
            let remote_dir = tempdir().unwrap();
            let remote_path = remote_dir.path();
            let remote_gen_dir = remote_path.join(".gen");
            fs::create_dir_all(&remote_gen_dir).unwrap();

            // Create remote operation database
            let remote_op_db_path = remote_gen_dir.join("gen.db");
            let remote_op_conn = &get_operation_connection(remote_op_db_path.to_str()).unwrap();

            // Create manifest operation
            let local_manifest = ManifestGenerator::new(local_op_conn)
                .generate_manifest("main", local_main.current_operation_hash.as_ref())
                .unwrap();

            let result =
                apply_operations_to_remote(remote_op_conn, &local_manifest.operations, remote_path);

            assert!(
                result.is_ok(),
                "apply_operations_to_remote failed: {:?}",
                result.err()
            );

            // Verify files were transferred
            for (index, m_op) in local_manifest.operations.iter().enumerate() {
                let operation = m_op.operation.clone();
                let remote_op_dir = remote_gen_dir
                    .join("changeset")
                    .join(operation.hash.to_string());
                assert!(remote_op_dir.join("changeset").exists());
                assert!(remote_op_dir.join("dependencies").exists());
                assert!(remote_path.join(format!("test_file_{index}.fa")).exists());

                // Verify operation was saved to remote database
                let remote_operation = Operation::get_by_id(remote_op_conn, &operation.hash);
                assert!(remote_operation.is_some());
                assert_eq!(remote_operation.unwrap().hash, operation.hash);
            }
        }
    }

    #[cfg(test)]
    mod pull_from_file_remote_tests {
        use tempfile::tempdir;

        use super::*;

        #[test]
        fn test_pull_from_file_remote_transfers_operations() {
            let local_gen_dir = setup_gen_dir();
            let local_repo_root = local_gen_dir.parent().unwrap().to_path_buf();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();
            track_database(conn, op_conn).unwrap();

            let remote_dir = setup_gen_dir();
            let remote_repo_root = remote_dir.parent().unwrap();

            set_base_dir(remote_repo_root);
            let remote_conn = &get_connection(remote_dir.join("remote.db").to_str()).unwrap();
            let remote_op_db_path = remote_dir.join("gen.db");
            let remote_op_conn = &get_operation_connection(remote_op_db_path.to_str()).unwrap();
            track_database(remote_conn, remote_op_conn).unwrap();

            let remote_operation = create_operation(
                remote_conn,
                remote_op_conn,
                "fixtures/empty.fa",
                FileTypes::Fasta,
                "remote operation",
                HashId::random_str(),
            );

            set_base_dir(local_repo_root.as_path());

            let remote_url = format!("file://{}", remote_repo_root.to_string_lossy());
            let branch = Branch::get_by_name(op_conn, "main").unwrap();
            pull_from_file_remote(op_conn, &remote_url, &branch).unwrap();

            let updated_branch = Branch::get_by_name(op_conn, "main").unwrap();
            assert_eq!(
                updated_branch.current_operation_hash,
                Some(remote_operation.hash)
            );

            let changeset_dir = local_repo_root
                .join(".gen")
                .join("changeset")
                .join(remote_operation.hash.to_string());
            assert!(changeset_dir.join("changeset").exists());
            assert!(changeset_dir.join("dependencies").exists());

            let local_ops = Operation::all(op_conn);
            let remote_ops = Operation::all(remote_op_conn);
            assert_eq!(local_ops, remote_ops);
        }

        #[test]
        fn test_pull_from_file_remote_missing_branch_errors() {
            let local_gen_dir = setup_gen_dir();
            let local_repo_root = local_gen_dir.parent().unwrap().to_path_buf();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();
            track_database(conn, op_conn).unwrap();

            let remote_dir = tempdir().unwrap();
            let remote_repo_root = remote_dir.path().to_path_buf();
            set_base_dir(remote_repo_root.as_path());
            let remote_gen_dir = get_or_create_gen_dir();
            let remote_conn = &get_connection(None).unwrap();
            let remote_op_db_path = remote_gen_dir.join("gen.db");
            let remote_op_conn = &get_operation_connection(remote_op_db_path.to_str()).unwrap();
            track_database(remote_conn, remote_op_conn).unwrap();

            set_base_dir(local_repo_root.as_path());

            let remote_url = format!("file://{}", remote_repo_root.to_string_lossy());
            let feature_branch = Branch::create_with_remote(op_conn, "feature", None).unwrap();
            let result = pull_from_file_remote(op_conn, &remote_url, &feature_branch);
            assert!(matches!(
                result,
                Err(RemoteOperationError::DoesNotExist(branch_name))
                    if branch_name.contains("feature")
            ));
        }
    }

    #[cfg(test)]
    mod push_to_file_remote {
        use super::*;

        #[test]
        fn test_push_to_uninitialized_remote_is_error() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            let remote_dir = tempdir().unwrap();
            let remote_url = format!("file://{}", remote_dir.path().to_string_lossy());
            let result = push_to_file_remote(op_conn, &remote_url, "main");
            assert!(result.is_err());
        }

        #[test]
        fn test_push_to_remote() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();

            track_database(conn, op_conn).unwrap();

            // Create first operation with actual changes
            let mut session = start_operation(conn);
            gen_models::sequence::Sequence::new()
                .sequence("ATCG")
                .sequence_type("DNA")
                .save(conn);
            let op_info = OperationInfo {
                files: vec![],
                description: "first operation".to_string(),
            };
            let op1 = end_operation(conn, op_conn, &mut session, &op_info, "test1", None).unwrap();

            let mut session = start_operation(conn);
            gen_models::sequence::Sequence::new()
                .sequence("GCTA")
                .sequence_type("DNA")
                .save(conn);
            let op_info = OperationInfo {
                files: vec![],
                description: "second operation".to_string(),
            };
            let op2 = end_operation(conn, op_conn, &mut session, &op_info, "test2", None).unwrap();

            let remote_dir = tempdir().unwrap();
            let remote_url = format!("file://{}", remote_dir.path().to_string_lossy());

            let remote_gen_path = remote_dir.path().join(".gen");
            fs::create_dir_all(&remote_gen_path).unwrap();
            get_operation_connection(remote_gen_path.join("gen.db").to_str()).unwrap();

            let result = push_to_file_remote(op_conn, &remote_url, "main");
            assert!(result.is_ok());

            // Verify both operations exist in remote
            let remote_op1_dir = remote_gen_path.join("changeset").join(op1.hash.to_string());
            let remote_op2_dir = remote_gen_path.join("changeset").join(op2.hash.to_string());
            assert!(remote_op1_dir.exists());
            assert!(remote_op2_dir.exists());
        }

        #[test]
        fn test_push_when_remote_ahead() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();

            track_database(conn, op_conn).unwrap();

            create_operation(
                conn,
                op_conn,
                "fixtures/simple.fa",
                FileTypes::Fasta,
                "local",
                HashId::random_str(),
            );

            let remote_dir = tempdir().unwrap();
            let remote_url = format!("file://{}", remote_dir.path().to_string_lossy());

            let remote_gen_path = remote_dir.path().join(".gen");
            fs::create_dir_all(&remote_gen_path).unwrap();
            let remote_conn = &get_connection(None).unwrap();
            let remote_op_conn =
                &get_operation_connection(remote_gen_path.join("gen.db").to_str()).unwrap();
            track_database(remote_conn, remote_op_conn).unwrap();

            create_operation(
                remote_conn,
                remote_op_conn,
                "fixtures/aa.fa",
                FileTypes::Fasta,
                "remote",
                HashId::random_str(),
            );

            let result = push_to_file_remote(op_conn, &remote_url, "main");
            assert!(matches!(
                result,
                Err(RemoteOperationError::RemoteBranchAhead)
            ));
        }

        #[test]
        fn test_push_with_no_operations() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();

            track_database(conn, op_conn).unwrap();

            let remote_dir = tempdir().unwrap();
            let remote_url = format!("file://{}", remote_dir.path().to_string_lossy());

            let result = push_to_file_remote(op_conn, &remote_url, "main");
            assert!(matches!(result, Err(RemoteOperationError::NoOperations)));
        }
    }
}
