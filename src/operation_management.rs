use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path as FilePath, PathBuf},
    str,
};

use gen_core::{HashId, config::get_gen_dir, errors::ConnectionError, traits::Capnp};
use gen_models::{
    changesets::{apply_changeset, revert_changeset},
    errors::{ChangesetError, OperationError, RemoteError},
    file_types::FileTypes,
    manifest::{
        ManifestComparer, ManifestDiff, ManifestDiffError, ManifestError, ManifestGenerator,
        ManifestOperation,
    },
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
use thiserror::Error;
use url_parse::core::Parser;

use crate::{commands::remote::utils::load_tokens, get_connection, get_operation_connection};

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
    #[error("Reqwest Error: {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] SQLError),

    #[error("No operations present in current branch")]
    NoOperations,
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

fn setup_remote_repository(
    remote_url: &str,
) -> Result<(PathBuf, Option<Connection>), RemoteOperationError> {
    // Parse the file:// URL to get the filesystem path
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

    let path_parts = parsed_url
        .path
        .ok_or_else(|| RemoteOperationError::InvalidRemoteUrl(remote_url.to_string()))?;

    // Construct the remote filesystem path
    let mut remote_path = PathBuf::from("/");
    for part in path_parts {
        remote_path.push(part);
    }

    // Check if remote operation database exists and open connection
    let remote_op_conn =
        {
            let op_db_path = remote_path.join(".gen").join("gen.db");
            if op_db_path.exists() {
                Some(get_operation_connection(Some(op_db_path)).map_err(|e| {
                    RemoteOperationError::IOError(std::io::Error::other(e.to_string()))
                })?)
            } else {
                None
            }
        };

    Ok((remote_path, remote_op_conn))
}

/// Apply operations to remote repository by transferring files and applying changesets
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

    // Process operations in dependency order (they should already be ordered correctly)
    for manifest_op in operations {
        let operation = &manifest_op.operation;
        let op_hash = &operation.hash;

        let changeset_src = operation.get_changeset_path();

        // Create remote operation directory
        let changeset_dst = remote_path.join("changeset").join(op_hash.to_string());
        fs::create_dir_all(&changeset_dst)?;

        fs::copy(&changeset_src, changeset_dst.join("changeset")).map_err(|_| {
            RemoteOperationError::FileTransferError(
                "changeset".to_string(),
                changeset_src.to_string_lossy().to_string(),
                changeset_dst.to_string_lossy().to_string(),
            )
        })?;

        // Transfer dependencies file
        let dependencies_src = operation.get_changeset_dependencies_path();
        fs::copy(&dependencies_src, changeset_dst.join("dependencies")).map_err(|_| {
            RemoteOperationError::FileTransferError(
                "dependencies".to_string(),
                dependencies_src.to_string_lossy().to_string(),
                changeset_dst.to_string_lossy().to_string(),
            )
        })?;

        // Transfer file additions
        for file_addition in &manifest_op.file_additions {
            let src_path = gen_dir.join(&file_addition.file_path);
            let dst_path = remote_path.join(&file_addition.file_path);

            // Create parent directories if needed
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }

            if src_path.exists() {
                fs::copy(&src_path, &dst_path).map_err(|_| {
                    RemoteOperationError::FileTransferError(
                        file_addition.file_path.clone(),
                        src_path.to_string_lossy().to_string(),
                        dst_path.to_string_lossy().to_string(),
                    )
                })?;
            }
        }

        // Apply changeset to remote data database if it exists
        let changeset = operation.get_changeset();
        let dependencies = operation.get_changeset_dependencies();

        // Apply changeset within a transaction
        remote_op_conn.execute("BEGIN TRANSACTION", [])?;
        match apply_changeset(remote_op_conn, &changeset.changes, &dependencies) {
            Ok(_) => {
                remote_op_conn.execute("COMMIT TRANSACTION", [])?;
            }
            Err(e) => {
                remote_op_conn.execute("ROLLBACK TRANSACTION", [])?;
                return Err(RemoteOperationError::IOError(std::io::Error::other(
                    format!("Failed to apply changeset for operation {}: {}", op_hash, e),
                )));
            }
        }

        // Insert operation into remote operation database
        remote_op_conn.execute("BEGIN TRANSACTION", [])?;
        match Operation::create_without_tracking(
            remote_op_conn,
            &operation.hash,
            &operation.change_type,
            operation.parent_hash,
        ) {
            Ok(_) => {
                // Add file associations for this operation
                for file_addition in &manifest_op.file_additions {
                    // Create file addition record in remote database
                    let remote_file_addition = FileAddition::create(
                        remote_op_conn,
                        &file_addition.file_path,
                        file_addition.file_type,
                    );

                    // Link operation to file addition
                    Operation::add_file(remote_op_conn, &operation.hash, remote_file_addition.id)?;
                }

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

/// Push to file scheme remote using manifest comparison
fn push_to_file_remote(
    local_op_conn: &Connection,
    remote_url: &str,
    current_branch_id: i64,
) -> Result<(), RemoteOperationError> {
    // Generate local manifest
    let generator = ManifestGenerator::new(local_op_conn);
    let current_branch = Branch::get_by_id(local_op_conn, current_branch_id).ok_or_else(|| {
        RemoteOperationError::IOError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Current branch not found",
        ))
    })?;

    let current_hash = current_branch
        .current_operation_hash
        .ok_or(RemoteOperationError::NoOperations)?;

    let local_manifest = generator.generate_manifest(&current_branch.name, &current_hash)?;

    // Setup remote repository connections
    let (remote_path, remote_op_conn) = setup_remote_repository(remote_url)?;

    // Generate remote manifest if remote repository exists and has operations
    let remote_manifest = if let Some(ref remote_op_conn) = remote_op_conn {
        let remote_branch = Branch::get_by_id(remote_op_conn, current_branch_id);
        if let Some(branch) = remote_branch {
            if let Some(hash) = branch.current_operation_hash {
                Some(generator.generate_manifest(&branch.name, &hash)?)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Compare manifests to determine missing operations
    let diff = if let Some(remote_manifest) = remote_manifest {
        ManifestComparer::diff_manifests(&local_manifest, &remote_manifest)?
    } else {
        // Empty remote - all local operations are missing
        ManifestDiff {
            missing_in_manifest2: local_manifest.operations.clone(),
            missing_in_manifest1: vec![],
        }
    };

    // Check for RemoteBranchAhead condition
    if !diff.missing_in_manifest1.is_empty() {
        return Err(RemoteOperationError::RemoteBranchAhead);
    }

    // Apply missing operations to remote if any
    if !diff.missing_in_manifest2.is_empty() {
        // Create remote .gen directory if it doesn't exist
        let remote_gen_dir = remote_path.join(".gen");
        fs::create_dir_all(&remote_gen_dir)?;

        let remote_op_conn = remote_op_conn.ok_or(RemoteOperationError::InvalidRemoteUrl(
            "Remote repository not initialized.".to_string(),
        ))?;
        Branch::get_or_create(&remote_op_conn, &current_branch.name);

        // Apply operations to remote
        apply_operations_to_remote(&remote_op_conn, &diff.missing_in_manifest2, &remote_path)?;

        // Update remote branch to point to the latest operation
        let latest_op_hash = diff
            .missing_in_manifest2
            .last()
            .map(|op| op.operation.hash)
            .unwrap_or(current_hash);

        Branch::set_current_operation(&remote_op_conn, current_branch_id, &latest_op_hash);
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

                    push_to_file_remote(operation_conn, &remote_url, current_branch_id)
                } else {
                    // New manifest-based push logic
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
                        generator.generate_manifest(&current_branch.name, &current_hash)?;
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

                        let form = multipart::Form::new()
                            .part("manifest_operation", part)
                            .file("files", cs_path)
                            .unwrap()
                            .file("files", dep_path)
                            .unwrap()
                            .text("branch", current_branch.name.clone());
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
            format!("{} - {}", remote_url, status),
        ));
    }

    println!("Manifest sent successfully to {}", remote_url);
    Ok(response.json()?)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use gen_models::{
        block_group::BlockGroup,
        block_group_edge::BlockGroupEdge,
        edge::Edge,
        file_types::FileTypes,
        node::Node,
        operations::{Branch, Operation, OperationState, setup_db},
        sample::Sample,
    };
    use rusqlite::params;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        get_connection as get_real_connection,
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
            setup_db(op_conn);
            track_database(conn, op_conn).unwrap();

            let op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "fasta_addition",
                HashId::convert_str("op-1"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
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
                FileTypes::Fasta,
                "vcf_addition",
                HashId::convert_str("op-3"),
            );
            let op_4 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "vcf_addition",
                HashId::convert_str("op-4"),
            );
            checkout(Some(conn), op_conn, &Some("branch-2".to_string()), None).unwrap();
            let op_5 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "vcf_addition",
                HashId::convert_str("op-5"),
            );
            let op_6 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
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
            setup_db(op_conn);
            track_database(conn, op_conn).unwrap();

            let _op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "fasta_addition",
                HashId::convert_str("op-1"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "fasta_addition",
                HashId::convert_str("op-2"),
            );
            let op_3 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
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
            setup_db(op_conn);
            track_database(conn, op_conn).unwrap();

            let _op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "fasta_addition",
                HashId::convert_str("op-1-abc-123"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "fasta_addition",
                HashId::convert_str("op-2-abc-123"),
            );
            let op_3 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
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
            setup_db(op_conn);
            track_database(conn, op_conn).unwrap();

            let _op_1 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "fasta_addition",
                HashId::convert_str("op-1-abc-123"),
            );
            let op_2 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
                "fasta_addition",
                HashId::convert_str("op-2-abc-123"),
            );
            let op_3 = create_operation(
                conn,
                op_conn,
                "foo",
                FileTypes::Fasta,
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
        setup_db(operation_conn);
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
            "select * from operation",
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
            "select * from operation",
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
            "select * from operation",
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
            "select * from operation",
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
        setup_db(operation_conn);
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
        setup_db(operation_conn);
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
            "select * from operation",
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
            "select * from operation",
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
            "select * from operation",
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
            "select * from operation",
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
            "select * from operation",
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
        setup_db(operation_conn);
        track_database(conn, operation_conn).unwrap();

        let main_branch = Branch::get_by_name(operation_conn, "main").unwrap();

        let op_1 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-1"),
        );
        let op_2 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-2"),
        );

        let branch_a = Branch::get_or_create(operation_conn, "branch-a");
        OperationState::set_branch(operation_conn, "branch-a");
        let op_3 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-3"),
        );
        let op_4 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-4"),
        );
        let op_5 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-5"),
        );
        OperationState::set_branch(operation_conn, "main");
        OperationState::set_operation(operation_conn, &HashId::convert_str("op-2"));
        let op_6 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-6"),
        );
        let op_7 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-7"),
        );
        let op_8 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
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
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-9"),
        );
        OperationState::set_branch(operation_conn, "branch-a");
        OperationState::set_operation(operation_conn, &HashId::convert_str("op-5"));
        let op_10 = create_operation(
            conn,
            operation_conn,
            "test.fasta",
            FileTypes::Fasta,
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
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let op_1 = create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-1"),
        );
        let op_2 = create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-2"),
        );
        let _op_3 = create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-3"),
        );
        let _op_4 = create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-4"),
        );

        reset(Some(conn), op_conn, &HashId::convert_str("op-2")).unwrap();
        let op_5 = create_operation(
            conn,
            op_conn,
            "test.fasta",
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

    #[test]
    fn test_push_to_empty_repo() {
        let gen_path = setup_gen_dir();
        let mut db_path = gen_path.clone();
        db_path.push("default.db");
        let mut op_db_path = gen_path.clone();
        op_db_path.push("gen.db");
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");

        let conn = &mut get_connection(db_path.to_str()).unwrap();
        let operation_conn = &get_operation_connection(op_db_path.to_str()).unwrap();
        setup_db(operation_conn);
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

        let binding = BlockGroup::query(
            conn,
            "SELECT * FROM block_groups ORDER BY created_on DESC;",
            rusqlite::params!(),
        );
        let block_group = binding.first().unwrap();

        let remote_path = tempdir().unwrap().keep();
        let remote_dir = remote_path.to_str().unwrap().to_string();
        let formatted_remote_dir = format!("file://{remote_dir}");

        // Create a remote and set it as default using the new remote management system
        Remote::create(operation_conn, "origin", &formatted_remote_dir).unwrap();
        Defaults::set_default_remote(operation_conn, Some("origin")).unwrap();

        // Force sync everything in the db to disk before pushing the repo.  TRUNCATE is the most
        // aggressive option (full sync, then truncate the WAL file that had any unsynced changes,
        // to indicate nothing is left to sync).
        operation_conn
            .pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();

        let result = push(operation_conn, None);
        assert!(result.is_ok());

        let mut remote_db_path = remote_path.clone();
        remote_db_path.push(".gen");
        remote_db_path.push("default.db");

        // Need to use get_real_connection here because get_connection is a test method and resets
        // the database if it exists
        let remote_conn = &mut get_real_connection(remote_db_path.to_str().unwrap()).unwrap();

        let all_local_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        let binding = BlockGroup::query(
            remote_conn,
            "SELECT * FROM block_groups ORDER BY created_on DESC;",
            rusqlite::params!(),
        );
        let block_group2 = binding.first().unwrap();
        assert_eq!(block_group2.id, block_group.id);
        let all_remote_sequences =
            BlockGroup::get_all_sequences(remote_conn, &block_group.id, false);
        assert_eq!(all_remote_sequences, all_local_sequences);

        let mut remote_op_db_path = remote_path.clone();
        remote_op_db_path.push(".gen");
        remote_op_db_path.push("gen.db");

        // Need to use get_real_connection here because get_operation_connection is a test method
        // and resets the database if it exists
        let remote_operation_conn =
            &mut get_real_connection(remote_op_db_path.to_str().unwrap()).unwrap();

        let local_operation_hashes: HashSet<HashId> = HashSet::from_iter(
            Operation::query(operation_conn, "SELECT * FROM operation;", params![])
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
        );

        let remote_operation_hashes = HashSet::from_iter(
            Operation::query(remote_operation_conn, "SELECT * FROM operation;", params![])
                .iter()
                .map(|op| op.hash)
                .collect::<Vec<_>>(),
        );

        assert_eq!(remote_operation_hashes, local_operation_hashes);
    }

    #[test]
    fn test_push_with_operation_files() {
        let gen_path = setup_gen_dir();
        let mut db_path = gen_path.clone();
        db_path.push("default.db");
        let mut op_db_path = gen_path.clone();
        op_db_path.push("gen.db");

        let main_repo_path = gen_path.parent().unwrap().to_path_buf();
        let fixtures_path = main_repo_path.clone().join("fixtures");
        fs::create_dir(fixtures_path.clone()).unwrap();

        let mut original_vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let vcf_relative_path = "fixtures/simple.vcf";
        original_vcf_path.push(vcf_relative_path);
        let vcf_path = fixtures_path.clone().join("simple.vcf");
        fs::copy(original_vcf_path, &vcf_path).unwrap();

        let mut original_fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fasta_relative_path = "fixtures/simple.fa";
        original_fasta_path.push(fasta_relative_path);
        let fasta_path = fixtures_path.clone().join("simple.fa");
        fs::copy(original_fasta_path, &fasta_path).unwrap();

        let conn = &mut get_connection(db_path.to_str()).unwrap();
        let operation_conn = &get_operation_connection(op_db_path.to_str()).unwrap();
        setup_db(operation_conn);
        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &fasta_relative_path.to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();

        update_with_vcf(
            &vcf_relative_path.to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            operation_conn,
            None,
        )
        .unwrap();

        let binding = BlockGroup::query(
            conn,
            "SELECT * FROM block_groups ORDER BY created_on DESC;",
            rusqlite::params!(),
        );
        let block_group = binding.first().unwrap();

        let remote_path = tempdir().unwrap().keep();
        let remote_dir = remote_path.to_str().unwrap().to_string();
        let formatted_remote_dir = format!("file://{remote_dir}");

        // Create a remote and set it as default using the new remote management system
        Remote::create(operation_conn, "origin", &formatted_remote_dir).unwrap();
        Defaults::set_default_remote(operation_conn, Some("origin")).unwrap();

        // Force sync everything in the db to disk before pushing the repo.  TRUNCATE is the most
        // aggressive option (full sync, then truncate the WAL file that had any unsynced changes,
        // to indicate nothing is left to sync).
        operation_conn
            .pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();

        let result = push(operation_conn, None);
        assert!(result.is_ok());

        let mut remote_db_path = remote_path.clone();
        remote_db_path.push(".gen");
        remote_db_path.push("default.db");

        // Need to use get_real_connection here because get_connection is a test method and resets
        // the database if it exists
        let remote_conn = &mut get_real_connection(remote_db_path.to_str().unwrap()).unwrap();

        let all_local_sequences = BlockGroup::get_all_sequences(conn, &block_group.id, false);
        let binding = BlockGroup::query(
            remote_conn,
            "SELECT * FROM block_groups ORDER BY created_on DESC;",
            rusqlite::params!(),
        );
        let block_group2 = binding.first().unwrap();
        assert_eq!(block_group2.id, block_group.id);
        let all_remote_sequences =
            BlockGroup::get_all_sequences(remote_conn, &block_group.id, false);
        assert_eq!(all_remote_sequences, all_local_sequences);

        let remote_path = remote_path.clone().join("fixtures");
        let remote_fasta_path = remote_path
            .clone()
            .join("simple.fa")
            .to_str()
            .unwrap()
            .to_string();
        let remote_vcf_path = remote_path
            .clone()
            .join("simple.vcf")
            .to_str()
            .unwrap()
            .to_string();
        assert!(fs::exists(remote_fasta_path).unwrap());
        assert!(fs::exists(remote_vcf_path).unwrap());
    }

    #[test]
    fn test_push_with_remote_repo_ahead() {
        let gen_path = setup_gen_dir();
        let mut db_path = gen_path.clone();
        db_path.push("default.db");
        let mut op_db_path = gen_path.clone();
        op_db_path.push("gen.db");

        let main_repo_path = gen_path.parent().unwrap().to_path_buf();
        let fixtures_path = main_repo_path.clone().join("fixtures");
        fs::create_dir(fixtures_path.clone()).unwrap();

        let mut original_fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fasta_relative_path = "fixtures/simple.fa";
        original_fasta_path.push(fasta_relative_path);
        let fasta_path = fixtures_path.clone().join("simple.fa");
        fs::copy(original_fasta_path, &fasta_path).unwrap();

        let conn = &mut get_connection(db_path.to_str()).unwrap();
        let operation_conn = &get_operation_connection(op_db_path.to_str()).unwrap();
        setup_db(operation_conn);
        track_database(conn, operation_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &fasta_relative_path.to_string(),
            &collection,
            None,
            false,
            conn,
            operation_conn,
        )
        .unwrap();

        let remote_path = tempdir().unwrap().keep();
        let remote_dir = remote_path.to_str().unwrap().to_string();
        let formatted_remote_dir = format!("file://{remote_dir}");

        // Create a remote and set it as default using the new remote management system
        Remote::create(operation_conn, "origin", &formatted_remote_dir).unwrap();
        Defaults::set_default_remote(operation_conn, Some("origin")).unwrap();

        // Force sync everything in the db to disk before pushing the repo.  TRUNCATE is the most
        // aggressive option (full sync, then truncate the WAL file that had any unsynced changes,
        // to indicate nothing is left to sync).
        operation_conn
            .pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();

        let result = push(operation_conn, None);
        assert!(result.is_ok());

        let mut remote_db_path = remote_path.clone();
        remote_db_path.push(".gen");
        remote_db_path.push("default.db");

        // Need to use get_real_connection here because get_connection is a test method and resets
        // the database if it exists
        let remote_conn = &mut get_real_connection(remote_db_path.to_str().unwrap()).unwrap();

        let mut remote_op_db_path = remote_path.clone();
        remote_op_db_path.push(".gen");
        remote_op_db_path.push("gen.db");

        // Need to use get_real_connection here because get_operation_connection is a test method
        // and resets the database if it exists
        let remote_operation_conn =
            &mut get_real_connection(remote_op_db_path.to_str().unwrap()).unwrap();

        let mut original_vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let vcf_relative_path = "fixtures/simple.vcf";
        original_vcf_path.push(vcf_relative_path);
        let vcf_path = remote_path.clone().join("fixtures/simple.vcf");
        fs::copy(original_vcf_path, &vcf_path).unwrap();

        update_with_vcf(
            &vcf_relative_path.to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            remote_conn,
            remote_operation_conn,
            None,
        )
        .unwrap();

        // Force sync everything in the db to disk before pushing the repo.  TRUNCATE is the most
        // aggressive option (full sync, then truncate the WAL file that had any unsynced changes,
        // to indicate nothing is left to sync).
        remote_operation_conn
            .pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();
        remote_conn
            .pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();

        let result = push(operation_conn, None).unwrap_err();
        assert!(matches!(result, RemoteOperationError::RemoteBranchAhead));
    }

    #[test]
    fn test_push_with_named_remote() {
        setup_gen_dir();
        let gen_path = setup_gen_dir();
        let mut db_path = gen_path.clone();
        db_path.push("default.db");
        let mut op_db_path = gen_path.clone();
        op_db_path.push("gen.db");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");

        let conn = &mut get_connection(db_path.to_str()).unwrap();
        let operation_conn = &get_operation_connection(op_db_path.to_str()).unwrap();
        setup_db(operation_conn);
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

        let remote_path = tempdir().unwrap().keep();
        let remote_dir = remote_path.to_str().unwrap().to_string();
        let formatted_remote_dir = format!("file://{remote_dir}");

        // Create a remote using the new remote management system
        Remote::create(operation_conn, "test-remote", &formatted_remote_dir).unwrap();
        Defaults::set_default_remote(operation_conn, Some("test-remote")).unwrap();

        // Verify that the remote name resolves to the correct URL
        let resolved_url = Defaults::get_default_remote_url(operation_conn);
        assert_eq!(resolved_url, Some(formatted_remote_dir.clone()));

        // Force sync everything in the db to disk before pushing the repo
        operation_conn
            .pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();

        // Test push with named remote
        let result = push(operation_conn, None);
        assert!(result.is_ok());

        // Verify that files were pushed to the remote location
        let mut remote_gen_path = remote_path.clone();
        remote_gen_path.push(".gen");
        assert!(remote_gen_path.exists());

        let mut remote_db_path = remote_gen_path.clone();
        remote_db_path.push("gen.db");
        assert!(remote_db_path.exists());
    }

    #[test]
    fn test_setup_remote_repository_with_invalid_url() {
        let result = setup_remote_repository("invalid-url");
        assert!(matches!(
            result,
            Err(RemoteOperationError::InvalidRemoteUrl(_))
        ));
    }

    #[test]
    fn test_setup_remote_repository_with_unsupported_scheme() {
        let result = setup_remote_repository("http://example.com/repo");
        assert!(matches!(
            result,
            Err(RemoteOperationError::UnsupportedRemoteScheme(_, _))
        ));
    }

    #[test]
    fn test_setup_remote_repository_with_nonexistent_remote() {
        let temp_dir = tempdir().unwrap();
        let nonexistent_path = temp_dir.path().join("nonexistent");
        let remote_url = format!("file://{}", nonexistent_path.to_str().unwrap());

        let result = setup_remote_repository(&remote_url);
        assert!(result.is_ok());

        let (remote_path, remote_op_conn) = result.unwrap();
        assert_eq!(remote_path, nonexistent_path);
        assert!(remote_op_conn.is_none());
    }

    #[test]
    fn test_setup_remote_repository_with_existing_remote() {
        let temp_dir = tempdir().unwrap().keep();
        let remote_path = &temp_dir;

        // Create .gen directory and operation database
        let gen_dir = remote_path.join(".gen");
        fs::create_dir_all(&gen_dir).unwrap();

        let op_db_path = gen_dir.join("gen.db");
        let op_conn = get_operation_connection(op_db_path.to_str()).unwrap();
        setup_db(&op_conn);

        // Set a database name in defaults
        op_conn
            .execute("update defaults set db_name=?1 where id = 1", ("test_db",))
            .unwrap();

        // Create the data database
        let data_db_path = gen_dir.join("test_db.db");
        let _data_conn = get_connection(data_db_path.to_str()).unwrap();

        // Close connections to ensure files are written
        drop(op_conn);
        drop(_data_conn);

        let remote_url = format!("file://{}", remote_path.to_str().unwrap());
        let result = setup_remote_repository(&remote_url);
        assert!(result.is_ok());

        let (parsed_remote_path, remote_op_conn) = result.unwrap();
        assert_eq!(parsed_remote_path, *remote_path);
        assert!(remote_op_conn.is_some());
    }

    #[test]
    fn test_setup_remote_repository_with_operation_db_only() {
        let temp_dir = tempdir().unwrap().keep();
        let remote_path = &temp_dir;

        // Create .gen directory and operation database only
        let gen_dir = remote_path.join(".gen");
        fs::create_dir_all(&gen_dir).unwrap();

        let op_db_path = gen_dir.join("gen.db");
        let op_conn = get_operation_connection(op_db_path.to_str()).unwrap();
        setup_db(&op_conn);

        // Don't create the data database
        drop(op_conn);

        let remote_url = format!("file://{}", remote_path.to_str().unwrap());
        let result = setup_remote_repository(&remote_url);
        assert!(result.is_ok());

        let (parsed_remote_path, remote_op_conn) = result.unwrap();
        assert_eq!(parsed_remote_path, *remote_path);
        assert!(remote_op_conn.is_some());
    }

    #[cfg(test)]
    mod apply_operations_to_remote {
        use gen_models::{
            collection::Collection,
            operations::{FileAddition, OperationFile, OperationInfo},
            session_operations::{end_operation, start_operation},
        };
        use tempfile::tempdir;

        use super::*;

        #[test]
        fn test_apply_operations_to_remote_basic() {
            setup_gen_dir();
            let local_conn = &get_connection(None).unwrap();
            let local_op_conn = &get_operation_connection(None).unwrap();
            setup_db(local_op_conn);
            track_database(local_conn, local_op_conn).unwrap();

            // Create a test collection and operation
            let _collection = Collection::create(local_conn, "test_collection");
            let mut session = start_operation(local_conn);

            // Make some actual changes to trigger changeset creation
            Collection::create(local_conn, "test_collection_2");

            let op_info = OperationInfo {
                files: vec![OperationFile {
                    file_path: "test_file.txt".to_string(),
                    file_type: FileTypes::None,
                }],
                description: "Test operation".to_string(),
            };
            let operation = end_operation(
                local_conn,
                local_op_conn,
                &mut session,
                &op_info,
                "Test operation",
                None,
            )
            .unwrap();

            // Create file addition
            let file_addition =
                FileAddition::create(local_op_conn, "test_file.txt", FileTypes::None);
            Operation::add_file(local_op_conn, &operation.hash, file_addition.id).unwrap();

            // Create manifest operation
            let manifest_op = ManifestOperation {
                operation: operation.clone(),
                changeset_hash: "test_changeset_hash".to_string(),
                dependencies_hash: "test_dependencies_hash".to_string(),
                file_additions: vec![file_addition],
                operation_summary: None,
            };

            // Create remote directory structure
            let remote_dir = tempdir().unwrap();
            let remote_path = remote_dir.path();
            let remote_gen_dir = remote_path.join(".gen");
            fs::create_dir_all(&remote_gen_dir).unwrap();

            // Create remote operation database
            let remote_op_db_path = remote_gen_dir.join("gen.db");
            let remote_op_conn = get_operation_connection(remote_op_db_path.to_str()).unwrap();
            setup_db(&remote_op_conn);

            // Create remote data database
            let remote_data_db_path = remote_gen_dir.join("test.db");
            let remote_data_conn = get_connection(remote_data_db_path.to_str()).unwrap();

            // Create test files in local gen directory
            let local_gen_dir = PathBuf::from(get_gen_dir().unwrap());
            let op_dir = local_gen_dir.join(operation.hash.to_string());
            fs::create_dir_all(&op_dir).unwrap();
            fs::write(op_dir.join("changeset"), "test changeset content").unwrap();
            fs::write(op_dir.join("dependencies"), "test dependencies content").unwrap();
            fs::write(local_gen_dir.join("test_file.txt"), "test file content").unwrap();

            // Test the function
            let result = apply_operations_to_remote(&remote_op_conn, &[manifest_op], remote_path);

            assert!(
                result.is_ok(),
                "apply_operations_to_remote failed: {:?}",
                result.err()
            );

            // Verify files were transferred
            let remote_op_dir = remote_path.join(operation.hash.to_string());
            assert!(remote_op_dir.join("changeset").exists());
            assert!(remote_op_dir.join("dependencies").exists());
            assert!(remote_path.join("test_file.txt").exists());

            // Verify operation was saved to remote database
            let remote_operation = Operation::get_by_id(&remote_op_conn, &operation.hash);
            assert!(remote_operation.is_some());
            assert_eq!(remote_operation.unwrap().hash, operation.hash);
        }

        #[test]
        fn test_apply_operations_to_remote_no_data_conn() {
            setup_gen_dir();
            let local_conn = &get_connection(None).unwrap();
            let local_op_conn = &get_operation_connection(None).unwrap();
            setup_db(local_op_conn);
            track_database(local_conn, local_op_conn).unwrap();

            // Create a simple operation
            let mut session = start_operation(local_conn);

            // Make some actual changes to trigger changeset creation
            Collection::create(local_conn, "test_collection");

            let op_info = OperationInfo {
                files: vec![],
                description: "Test operation".to_string(),
            };
            let operation = end_operation(
                local_conn,
                local_op_conn,
                &mut session,
                &op_info,
                "Test operation",
                None,
            )
            .unwrap();

            let manifest_op = ManifestOperation {
                operation: operation.clone(),
                changeset_hash: "test_changeset_hash".to_string(),
                dependencies_hash: "test_dependencies_hash".to_string(),
                file_additions: vec![],
                operation_summary: None,
            };

            // Create remote directory structure
            let remote_dir = tempdir().unwrap();
            let remote_path = remote_dir.path();
            let remote_gen_dir = remote_path.join(".gen");
            fs::create_dir_all(&remote_gen_dir).unwrap();

            // Create remote operation database
            let remote_op_db_path = remote_gen_dir.join("gen.db");
            let remote_op_conn = get_operation_connection(remote_op_db_path.to_str()).unwrap();
            setup_db(&remote_op_conn);

            // Create test files in local gen directory
            let local_gen_dir = PathBuf::from(get_gen_dir().unwrap());
            let op_dir = local_gen_dir.join(operation.hash.to_string());
            fs::create_dir_all(&op_dir).unwrap();
            fs::write(op_dir.join("changeset"), "test changeset content").unwrap();
            fs::write(op_dir.join("dependencies"), "test dependencies content").unwrap();

            // Test the function without remote data connection
            let result = apply_operations_to_remote(&remote_op_conn, &[manifest_op], remote_path);

            assert!(
                result.is_ok(),
                "apply_operations_to_remote failed: {:?}",
                result.err()
            );

            // Verify files were transferred
            let remote_op_dir = remote_path.join(operation.hash.to_string());
            assert!(remote_op_dir.join("changeset").exists());
            assert!(remote_op_dir.join("dependencies").exists());

            // Verify operation was saved to remote database
            let remote_operation = Operation::get_by_id(&remote_op_conn, &operation.hash);
            assert!(remote_operation.is_some());
        }

        #[test]
        fn test_apply_operations_to_remote_multiple_operations() {
            setup_gen_dir();
            let local_conn = &get_connection(None).unwrap();
            let local_op_conn = &get_operation_connection(None).unwrap();
            setup_db(local_op_conn);
            track_database(local_conn, local_op_conn).unwrap();

            // Create multiple operations
            let mut operations = Vec::new();
            let mut manifest_ops = Vec::new();

            for i in 0..3 {
                let mut session = start_operation(local_conn);

                // Make some actual changes to trigger changeset creation
                Collection::create(local_conn, &format!("test_collection_{}", i));

                let op_info = OperationInfo {
                    files: vec![OperationFile {
                        file_path: format!("test_file_{}.txt", i),
                        file_type: FileTypes::None,
                    }],
                    description: format!("Test operation {}", i),
                };
                let operation = end_operation(
                    local_conn,
                    local_op_conn,
                    &mut session,
                    &op_info,
                    &format!("Test operation {}", i),
                    None,
                )
                .unwrap();

                let file_addition = FileAddition::create(
                    local_op_conn,
                    &format!("test_file_{}.txt", i),
                    FileTypes::None,
                );
                Operation::add_file(local_op_conn, &operation.hash, file_addition.id).unwrap();

                let manifest_op = ManifestOperation {
                    operation: operation.clone(),
                    changeset_hash: format!("test_changeset_hash_{}", i),
                    dependencies_hash: format!("test_dependencies_hash_{}", i),
                    file_additions: vec![file_addition],
                    operation_summary: None,
                };

                operations.push(operation);
                manifest_ops.push(manifest_op);
            }

            // Create remote directory structure
            let remote_dir = tempdir().unwrap();
            let remote_path = remote_dir.path();
            let remote_gen_dir = remote_path.join(".gen");
            fs::create_dir_all(&remote_gen_dir).unwrap();

            // Create remote operation database
            let remote_op_db_path = remote_gen_dir.join("gen.db");
            let remote_op_conn = get_operation_connection(remote_op_db_path.to_str()).unwrap();
            setup_db(&remote_op_conn);

            // Create test files in local gen directory
            let local_gen_dir = PathBuf::from(get_gen_dir().unwrap());
            for (i, operation) in operations.iter().enumerate() {
                let op_dir = local_gen_dir.join(operation.hash.to_string());
                fs::create_dir_all(&op_dir).unwrap();
                fs::write(
                    op_dir.join("changeset"),
                    format!("test changeset content {}", i),
                )
                .unwrap();
                fs::write(
                    op_dir.join("dependencies"),
                    format!("test dependencies content {}", i),
                )
                .unwrap();
                fs::write(
                    local_gen_dir.join(format!("test_file_{}.txt", i)),
                    format!("test file content {}", i),
                )
                .unwrap();
            }

            // Test the function with multiple operations
            let result = apply_operations_to_remote(&remote_op_conn, &manifest_ops, remote_path);

            assert!(
                result.is_ok(),
                "apply_operations_to_remote failed: {:?}",
                result.err()
            );

            // Verify all files were transferred and operations saved
            for (i, operation) in operations.iter().enumerate() {
                let remote_op_dir = remote_path.join(operation.hash.to_string());
                assert!(remote_op_dir.join("changeset").exists());
                assert!(remote_op_dir.join("dependencies").exists());
                assert!(remote_path.join(format!("test_file_{}.txt", i)).exists());

                let remote_operation = Operation::get_by_id(&remote_op_conn, &operation.hash);
                assert!(remote_operation.is_some());
            }
        }
    }

    #[cfg(test)]
    mod push_to_file_remote {
        use super::*;

        #[test]
        fn test_push_to_empty_remote() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();
            setup_db(op_conn);
            track_database(conn, op_conn).unwrap();

            // Create a test operation with actual changes
            let mut session = start_operation(conn);
            // Add some data to create actual changes
            gen_models::sequence::Sequence::new()
                .sequence("ATCG")
                .sequence_type("DNA")
                .save(conn);
            let op_info = OperationInfo {
                files: vec![],
                description: "test operation".to_string(),
            };
            let operation =
                end_operation(conn, op_conn, &mut session, &op_info, "test", None).unwrap();

            // Create a temporary remote directory
            let remote_dir = tempdir().unwrap();
            let remote_url = format!("file://{}", remote_dir.path().to_string_lossy());

            let current_branch_id = OperationState::get_current_branch(op_conn).unwrap();

            // Test push to empty remote
            let result = push_to_file_remote(op_conn, &remote_url, current_branch_id);
            assert!(result.is_ok());

            // Verify remote directory structure was created
            let remote_gen_dir = remote_dir.path().join(".gen");
            assert!(remote_gen_dir.exists());
            assert!(remote_gen_dir.join("gen.db").exists());

            // Verify operation was transferred
            let remote_op_dir = remote_dir.path().join(operation.hash.to_string());
            assert!(remote_op_dir.exists());
        }

        #[test]
        fn test_push_to_existing_remote() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();
            setup_db(op_conn);
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

            // Create a temporary remote directory and initialize it
            let remote_dir = tempdir().unwrap();
            let remote_url = format!("file://{}", remote_dir.path().to_string_lossy());

            let current_branch_id = OperationState::get_current_branch(op_conn).unwrap();

            // First push to initialize remote
            let result = push_to_file_remote(op_conn, &remote_url, current_branch_id);
            assert!(result.is_ok());

            // Create second operation with actual changes
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

            // Second push should only transfer the new operation
            let result = push_to_file_remote(op_conn, &remote_url, current_branch_id);
            assert!(result.is_ok());

            // Verify both operations exist in remote
            let remote_op1_dir = remote_dir.path().join(op1.hash.to_string());
            let remote_op2_dir = remote_dir.path().join(op2.hash.to_string());
            assert!(remote_op1_dir.exists());
            assert!(remote_op2_dir.exists());
        }

        #[test]
        fn test_push_when_remote_ahead() {
            // This test verifies that the RemoteBranchAhead logic works
            // by directly testing the manifest comparison logic
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();
            setup_db(op_conn);
            track_database(conn, op_conn).unwrap();

            // Create a local operation
            let mut session = start_operation(conn);
            gen_models::sequence::Sequence::new()
                .sequence("ATCG")
                .sequence_type("DNA")
                .save(conn);
            let op_info = OperationInfo {
                files: vec![],
                description: "local operation".to_string(),
            };
            let local_op =
                end_operation(conn, op_conn, &mut session, &op_info, "local", None).unwrap();

            // Create a mock remote manifest with a different operation
            let remote_manifest_op = ManifestOperation {
                operation: Operation {
                    hash: gen_core::HashId::convert_str("remote_operation_hash"),
                    parent_hash: None,
                    change_type: "test".to_string(),
                },
                changeset_hash: "remote_changeset".to_string(),
                dependencies_hash: "remote_deps".to_string(),
                file_additions: vec![],
                operation_summary: None,
            };

            let local_manifest = gen_models::manifest::Manifest {
                manifest_version: "1.0".to_string(),
                branch_name: "main".to_string(),
                end_hash: local_op.hash,
                operations: vec![ManifestOperation {
                    operation: local_op,
                    changeset_hash: "local_changeset".to_string(),
                    dependencies_hash: "local_deps".to_string(),
                    file_additions: vec![],
                    operation_summary: None,
                }],
            };

            let remote_manifest = gen_models::manifest::Manifest {
                manifest_version: "1.0".to_string(),
                branch_name: "main".to_string(),
                end_hash: gen_core::HashId::convert_str("remote_operation_hash"),
                operations: vec![remote_manifest_op],
            };

            // Test that ManifestComparer correctly identifies the conflict
            let diff = ManifestComparer::diff_manifests(&local_manifest, &remote_manifest).unwrap();

            // Both manifests should have operations missing in the other
            assert!(
                !diff.missing_in_manifest1.is_empty(),
                "Remote should have operations not in local"
            );
            assert!(
                !diff.missing_in_manifest2.is_empty(),
                "Local should have operations not in remote"
            );

            // This simulates the RemoteBranchAhead condition
            // In the actual push_to_file_remote function, this would trigger the error
        }

        #[test]
        fn test_push_with_no_operations() {
            setup_gen_dir();
            let conn = &get_connection(None).unwrap();
            let op_conn = &get_operation_connection(None).unwrap();
            setup_db(op_conn);
            track_database(conn, op_conn).unwrap();

            let remote_dir = tempdir().unwrap();
            let remote_url = format!("file://{}", remote_dir.path().to_string_lossy());
            let current_branch_id = OperationState::get_current_branch(op_conn).unwrap();

            // Push with no operations should fail
            let result = push_to_file_remote(op_conn, &remote_url, current_branch_id);
            assert!(matches!(result, Err(RemoteOperationError::NoOperations)));
        }
    }
}
