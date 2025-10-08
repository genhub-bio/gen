use std::{
    collections::HashMap,
    convert::TryInto,
    io::{self, BufReader},
    path::{Path, PathBuf},
    rc::Rc,
    string::ToString,
};

use gen_core::{HashId, calculate_hash, config::get_changeset_path, traits::Capnp};
use gen_graph::{OperationGraph, all_simple_paths};
use petgraph::{Direction, graphmap::UnGraphMap};
use rusqlite::{Connection, Result as SQLResult, Row, params, types::Value};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    changesets::{
        DatabaseChangeset, get_changeset_dependencies_from_path, get_changeset_from_path,
    },
    errors::{BranchError, FileAdditionError, RemoteError},
    file_types::FileTypes,
    gen_models_capnp::operation,
    session_operations::DependencyModels,
    traits::*,
};

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Operation {
    pub hash: HashId,
    pub parent_hash: Option<HashId>,
    pub change_type: String,
    pub created_on: i64,
}

impl<'a> Capnp<'a> for Operation {
    type Builder = operation::Builder<'a>;
    type Reader = operation::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_hash(&self.hash.0).unwrap();
        match &self.parent_hash {
            None => {
                builder.reborrow().get_parent_hash().set_none(());
            }
            Some(n) => {
                builder.reborrow().get_parent_hash().set_some(&n.0).unwrap();
            }
        }
        builder.set_change_type(&self.change_type);
        builder.set_created_on(self.created_on);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let hash = reader
            .get_hash()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let parent_hash = match reader.get_parent_hash().which().unwrap() {
            operation::parent_hash::None(()) => None,
            operation::parent_hash::Some(n) => {
                Some(n.unwrap().as_slice().unwrap().try_into().unwrap())
            }
        };
        let change_type = reader.get_change_type().unwrap().to_string().unwrap();
        let created_on = reader.get_created_on();

        Operation {
            hash,
            parent_hash,
            change_type,
            created_on,
        }
    }
}

impl Operation {
    pub fn create(conn: &Connection, change_type: &str, hash: &HashId) -> SQLResult<Operation> {
        let current_op = OperationState::get_operation(conn);
        let current_branch_id =
            OperationState::get_current_branch(conn).expect("No branch is checked out.");

        let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap();
        let query = "INSERT INTO operations (hash, change_type, parent_hash, created_on) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![hash, change_type, current_op, timestamp])?;
        let operation = Operation {
            hash: *hash,
            parent_hash: current_op,
            change_type: change_type.to_string(),
            created_on: timestamp,
        };
        // TODO: error condition here where we can write to disk but transaction fails
        OperationState::set_operation(conn, &operation.hash);
        Branch::set_current_operation(conn, current_branch_id, &operation.hash);
        Ok(operation)
    }

    pub fn create_without_tracking(
        conn: &Connection,
        hash: &HashId,
        change_type: &str,
        parent_hash: Option<HashId>,
        created_on: Option<i64>,
    ) -> SQLResult<Operation> {
        let timestamp = created_on.unwrap_or(chrono::Utc::now().timestamp_nanos_opt().unwrap());
        let query = "INSERT INTO operations (hash, change_type, parent_hash, created_on) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![hash, change_type, parent_hash, timestamp])?;
        let operation = Operation {
            hash: *hash,
            parent_hash,
            change_type: change_type.to_string(),
            created_on: timestamp,
        };
        Ok(operation)
    }

    pub fn add_file(
        conn: &Connection,
        operation_hash: &HashId,
        file_addition_id: &HashId,
    ) -> SQLResult<()> {
        let query =
            "INSERT INTO operation_files (operation_hash, file_addition_id) VALUES (?1, ?2)";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![operation_hash, file_addition_id])?;
        Ok(())
    }

    pub fn add_database(
        conn: &Connection,
        operation_hash: &HashId,
        db_uuid: &str,
    ) -> SQLResult<()> {
        let query =
            "INSERT INTO operation_databases (operation_hash, database_uuid) VALUES (?1, ?2)";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![operation_hash, db_uuid])?;
        Ok(())
    }

    pub fn get_upstream(conn: &Connection, operation_hash: &HashId) -> Vec<HashId> {
        let query = "WITH RECURSIVE r_operations(operation_hash, depth) AS ( \
        select ?1, 0 UNION \
        select parent_hash, depth + 1 from r_operations join operations ON hash=operation_hash \
        ) SELECT operation_hash, depth from r_operations where operation_hash is not null order by depth desc;";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.query_map([operation_hash], |row| row.get(0))
            .unwrap()
            .map(|id| id.unwrap())
            .collect::<Vec<HashId>>()
    }

    pub fn get_operation_graph(conn: &Connection) -> OperationGraph {
        let mut graph = OperationGraph::new();
        let operations = Operation::query(conn, "select * from operations;", rusqlite::params![]);
        for op in operations.iter() {
            graph.add_node(op.hash);
            if let Some(v) = op.parent_hash {
                graph.add_node(v);
                graph.add_edge(v, op.hash, ());
            }
        }
        graph
    }

    pub fn get_path_between(
        conn: &Connection,
        source_node: HashId,
        target_node: HashId,
    ) -> Vec<(HashId, Direction, HashId)> {
        let directed_graph = Operation::get_operation_graph(conn);
        let mut undirected_graph: UnGraphMap<HashId, ()> = Default::default();

        for node in directed_graph.nodes() {
            undirected_graph.add_node(node);
        }
        for (source, target, _weight) in directed_graph.all_edges() {
            undirected_graph.add_edge(source, target, ());
        }
        let mut patch_path: Vec<(HashId, Direction, HashId)> = vec![];
        for path in all_simple_paths(&undirected_graph, source_node, target_node) {
            let mut last_node = source_node;
            for node in &path[1..] {
                if *node != source_node {
                    for (_edge_src, edge_target, _edge_weight) in
                        directed_graph.edges_directed(last_node, Direction::Outgoing)
                    {
                        if edge_target == *node {
                            patch_path.push((last_node, Direction::Outgoing, *node));
                            break;
                        }
                    }
                    for (edge_src, _edge_target, _edge_weight) in
                        directed_graph.edges_directed(last_node, Direction::Incoming)
                    {
                        if edge_src == *node {
                            patch_path.push((last_node, Direction::Incoming, *node));
                            break;
                        }
                    }
                }
                last_node = *node;
            }
        }
        patch_path
    }

    pub fn search_hash(conn: &Connection, op_hash: &str) -> SQLResult<Operation> {
        Operation::get(
            conn,
            "select * from operations where hash LIKE ?1",
            params![op_hash],
        )
    }

    pub fn get_changeset_path(&self) -> PathBuf {
        get_changeset_path(&self.hash).join("changeset")
    }

    pub fn get_changeset_dependencies_path(&self) -> PathBuf {
        get_changeset_path(&self.hash).join("dependencies")
    }

    pub fn get_changeset(&self) -> DatabaseChangeset {
        let path = get_changeset_path(&self.hash).join("changeset");
        get_changeset_from_path(path)
    }

    pub fn get_changeset_dependencies(&self) -> DependencyModels {
        let path = get_changeset_path(&self.hash).join("dependencies");
        get_changeset_dependencies_from_path(path)
    }
}

impl Query for Operation {
    type Model = Operation;

    const PRIMARY_KEY: &'static str = "hash";
    const TABLE_NAME: &'static str = "operations";

    fn process_row(row: &Row) -> Self::Model {
        Operation {
            hash: row.get(0).unwrap(),
            parent_hash: row.get(1).unwrap(),
            change_type: row.get(2).unwrap(),
            created_on: row.get(3).unwrap(),
        }
    }
}

pub struct OperationFile {
    pub file_path: String,
    pub file_type: FileTypes,
}

pub struct OperationInfo {
    pub files: Vec<OperationFile>,
    pub description: String,
}

pub fn calculate_file_checksum<P: AsRef<Path>>(file_path: P) -> Result<HashId, std::io::Error> {
    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);
    let hash_bytes = calculate_stream_hash(reader)?;
    Ok(HashId(hash_bytes))
}

fn calculate_stream_hash<R: std::io::Read>(mut reader: R) -> Result<[u8; 32], std::io::Error> {
    let mut hasher = Sha256::new();
    io::copy(&mut reader, &mut hasher)?;
    let result = hasher.finalize();
    let mut hash_array = [0u8; 32];
    hash_array.copy_from_slice(&result);
    Ok(hash_array)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct FileAddition {
    pub id: HashId,
    pub file_path: String,
    pub file_type: FileTypes,
    pub checksum: HashId,
}

impl Query for FileAddition {
    type Model = FileAddition;

    const TABLE_NAME: &'static str = "file_additions";

    fn process_row(row: &Row) -> Self::Model {
        Self::Model {
            id: row.get(0).unwrap(),
            file_path: row.get(1).unwrap(),
            file_type: row.get(2).unwrap(),
            checksum: row.get(3).unwrap(),
        }
    }
}

impl FileAddition {
    pub fn generate_file_addition_id(checksum: &HashId, file_path: &str) -> HashId {
        let combined = format!("{checksum};{file_path}");
        HashId(calculate_hash(&combined))
    }

    pub fn get_or_create(
        conn: &Connection,
        file_path: &str,
        file_type: FileTypes,
    ) -> Result<FileAddition, FileAdditionError> {
        let checksum = if file_path.is_empty() {
            HashId::convert_str("empty")
        } else {
            match calculate_file_checksum(file_path) {
                Ok(checksum) => checksum,
                Err(e) => match e.kind() {
                    std::io::ErrorKind::NotFound => HashId::convert_str("non-existent"),
                    std::io::ErrorKind::PermissionDenied => {
                        return Err(FileAdditionError::FilePermissionDenied(
                            file_path.to_string(),
                        ));
                    }
                    _ => {
                        return Err(FileAdditionError::FileReadError(e));
                    }
                },
            }
        };

        let id = FileAddition::generate_file_addition_id(&checksum, file_path);

        let query = "INSERT INTO file_additions (id, file_path, file_type, checksum) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query).unwrap();

        let addition = FileAddition {
            id,
            file_path: file_path.to_string(),
            file_type,
            checksum,
        };

        match stmt.execute((&id, file_path, file_type, &checksum)) {
            Ok(_) => Ok(addition),
            Err(err) => match &err {
                rusqlite::Error::SqliteFailure(suberr, _details) => {
                    if suberr.code == rusqlite::ErrorCode::ConstraintViolation {
                        Ok(addition)
                    } else {
                        Err(FileAdditionError::DatabaseError(err))
                    }
                }
                _ => Err(FileAdditionError::DatabaseError(err)),
            },
        }
    }

    pub fn get_files_for_operation(
        conn: &Connection,
        operation_hash: &HashId,
    ) -> Vec<FileAddition> {
        let query = "select fa.* from file_additions fa left join operation_files of on (fa.id = of.file_addition_id) where of.operation_hash = ?1";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(params![operation_hash], |row| {
                Ok(FileAddition::process_row(row))
            })
            .unwrap();
        rows.map(|row| row.unwrap()).collect()
    }

    pub fn query_by_operations(
        conn: &Connection,
        operations: &[HashId],
    ) -> Result<HashMap<HashId, Vec<FileAddition>>, FileAdditionError> {
        let query = "select fa.*, of.operation_hash from file_additions fa left join operation_files of on (fa.id = of.file_addition_id) where of.operation_hash in rarray(?1)";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(
                params![Rc::new(
                    operations
                        .iter()
                        .map(|h| Value::from(*h))
                        .collect::<Vec<Value>>()
                )],
                |row| Ok((FileAddition::process_row(row), row.get::<_, HashId>(4)?)),
            )
            .unwrap();
        rows.into_iter()
            .try_fold(HashMap::new(), |mut acc: HashMap<_, Vec<_>>, row| {
                let (item, hash) = row?;
                acc.entry(hash).or_default().push(item);
                Ok(acc)
            })
    }
}

#[derive(Debug, Error)]
pub enum OperationSummaryError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct OperationSummary {
    pub id: i64,
    pub operation_hash: HashId,
    pub summary: String,
}

impl Query for OperationSummary {
    type Model = OperationSummary;

    const TABLE_NAME: &'static str = "operation_summaries";

    fn process_row(row: &Row) -> Self::Model {
        Self::Model {
            id: row.get(0).unwrap(),
            operation_hash: row.get(1).unwrap(),
            summary: row.get(2).unwrap(),
        }
    }
}

impl OperationSummary {
    pub fn create(conn: &Connection, operation_hash: &HashId, summary: &str) -> OperationSummary {
        let query = "INSERT INTO operation_summaries (operation_hash, summary) VALUES (?1, ?2) RETURNING (id)";
        let mut stmt = conn.prepare(query).unwrap();
        let mut rows = stmt
            .query_map(params![operation_hash, summary], |row| {
                Ok(OperationSummary {
                    id: row.get(0)?,
                    operation_hash: *operation_hash,
                    summary: summary.to_string(),
                })
            })
            .unwrap();
        rows.next().unwrap().unwrap()
    }

    pub fn set_message(conn: &Connection, id: i64, message: &str) -> SQLResult<()> {
        let query = "UPDATE operation_summaries SET summary = ?2 where id = ?1";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![id, message])?;
        Ok(())
    }

    pub fn query_by_operations(
        conn: &Connection,
        operations: &[HashId],
    ) -> Result<HashMap<HashId, Vec<Self>>, OperationSummaryError> {
        let query = "select * from operation_summaries where operation_hash in rarray(?1)";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(
                params![Rc::new(
                    operations
                        .iter()
                        .map(|h| Value::from(*h))
                        .collect::<Vec<Value>>()
                )],
                |row| Ok(Self::process_row(row)),
            )
            .unwrap();
        rows.into_iter()
            .try_fold(HashMap::new(), |mut acc: HashMap<_, Vec<_>>, row| {
                let item = row?;
                acc.entry(item.operation_hash).or_default().push(item);
                Ok(acc)
            })
    }
}

impl<'a> Capnp<'a> for FileAddition {
    type Builder = crate::gen_models_capnp::file_addition::Builder<'a>;
    type Reader = crate::gen_models_capnp::file_addition::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_file_path(&self.file_path);
        builder.set_file_type(self.file_type.into());
        builder.set_checksum(&self.checksum.0).unwrap();
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        Self {
            id: reader
                .get_id()
                .unwrap()
                .as_slice()
                .unwrap()
                .try_into()
                .unwrap(),
            file_path: reader.get_file_path().unwrap().to_string().unwrap(),
            file_type: reader.get_file_type().unwrap().into(),
            checksum: reader
                .get_checksum()
                .unwrap()
                .as_slice()
                .unwrap()
                .try_into()
                .unwrap(),
        }
    }
}

impl<'a> Capnp<'a> for OperationSummary {
    type Builder = crate::gen_models_capnp::operation_summary::Builder<'a>;
    type Reader = crate::gen_models_capnp::operation_summary::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(self.id);
        builder.set_operation_hash(&self.operation_hash.0).unwrap();
        builder.set_summary(&self.summary);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        Self {
            id: reader.get_id(),
            operation_hash: reader
                .get_operation_hash()
                .unwrap()
                .as_slice()
                .unwrap()
                .try_into()
                .unwrap(),
            summary: reader.get_summary().unwrap().to_string().unwrap(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Remote {
    pub name: String,
    pub url: String,
}

impl Query for Remote {
    type Model = Remote;

    const TABLE_NAME: &'static str = "remotes";

    fn process_row(row: &Row) -> Self::Model {
        Remote {
            name: row.get(0).unwrap(),
            url: row.get(1).unwrap(),
        }
    }
}

impl Remote {
    /// Validate remote name - no spaces or special characters except hyphens and underscores
    pub fn validate_name(name: &str) -> Result<(), RemoteError> {
        if name.is_empty() {
            return Err(RemoteError::EmptyName);
        }

        if name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '-' && c != '_')
        {
            return Err(RemoteError::InvalidNameCharacters);
        }

        Ok(())
    }

    /// Validate URL format
    pub fn validate_url(url: &str) -> Result<(), RemoteError> {
        if url.is_empty() {
            return Err(RemoteError::EmptyUrl);
        }

        // Check if it looks like a URL with a scheme
        if url.contains("://") {
            match url::Url::parse(url) {
                Ok(parsed_url) => {
                    // Only allow http, https, and ssh schemes
                    match parsed_url.scheme() {
                        "http" | "https" | "ssh" | "file" => Ok(()),
                        _ => Err(RemoteError::UnsupportedUrlScheme),
                    }
                }
                Err(_) => Err(RemoteError::InvalidUrl("Invalid URL format".to_string())),
            }
        } else if url.starts_with('/') || url.contains(':') {
            // Assume it's a file path or SSH-style path (like user@host:path)
            Ok(())
        } else {
            Err(RemoteError::UnsupportedUrlScheme)
        }
    }

    /// Create a new remote with the given name and URL
    /// Validates input and handles constraint violations gracefully
    pub fn create(conn: &Connection, name: &str, url: &str) -> Result<Remote, RemoteError> {
        // Validate input
        Self::validate_name(name)?;
        Self::validate_url(url)?;

        let query = "INSERT INTO remotes (name, url) VALUES (?1, ?2)";
        let mut stmt = conn.prepare(query)?;

        match stmt.execute(params![name, url]) {
            Ok(_) => Ok(Remote {
                name: name.to_string(),
                url: url.to_string(),
            }),
            Err(rusqlite::Error::SqliteFailure(err, _))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(RemoteError::RemoteAlreadyExists(name.to_string()))
            }
            Err(e) => Err(RemoteError::DatabaseError(e)),
        }
    }

    /// Get a remote by name
    pub fn get_by_name(conn: &Connection, name: &str) -> Result<Remote, RemoteError> {
        let query = "SELECT name, url FROM remotes WHERE name = ?1";
        match Remote::get(conn, query, params![name]) {
            Ok(remote) => Ok(remote),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(RemoteError::RemoteNotFound(name.to_string()))
            }
            Err(e) => Err(RemoteError::DatabaseError(e)),
        }
    }

    /// Get a remote by name, returning None if not found (for backward compatibility)
    pub fn get_by_name_optional(conn: &Connection, name: &str) -> Option<Remote> {
        Self::get_by_name(conn, name).ok()
    }

    /// List all remotes
    pub fn list_all(conn: &Connection) -> Vec<Remote> {
        Remote::query(
            conn,
            "SELECT name, url FROM remotes ORDER BY name",
            params![],
        )
    }

    /// Delete a remote by name
    pub fn delete(conn: &Connection, name: &str) -> Result<(), RemoteError> {
        // Check if remote exists first
        Self::get_by_name(conn, name)?;

        let query = "DELETE FROM remotes WHERE name = ?1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![name])?;
        Ok(())
    }

    /// Check if a remote exists
    pub fn exists(conn: &Connection, name: &str) -> bool {
        Self::get_by_name_optional(conn, name).is_some()
    }
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub id: i64,
    pub name: String,
    pub current_operation_hash: Option<HashId>,
    pub remote_name: Option<String>,
}

impl Query for Branch {
    type Model = Branch;

    const TABLE_NAME: &'static str = "branches";

    fn process_row(row: &Row) -> Self::Model {
        Branch {
            id: row.get(0).unwrap(),
            name: row.get(1).unwrap(),
            current_operation_hash: row.get(2).unwrap(),
            remote_name: row.get(3).unwrap(),
        }
    }
}

impl Branch {
    pub fn get_or_create(conn: &Connection, branch_name: &str) -> Branch {
        match Branch::create_with_remote(conn, branch_name, None) {
            Ok(res) => res,
            Err(rusqlite::Error::SqliteFailure(err, details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    Branch::get_by_name(conn, branch_name)
                        .unwrap_or_else(|| panic!("No branch named {branch_name}."))
                } else {
                    panic!("something bad happened querying the database {err:?} {details:?}");
                }
            }
            Err(_) => {
                panic!("something bad happened querying the database");
            }
        }
    }

    pub fn create_with_remote(
        conn: &Connection,
        branch_name: &str,
        remote_name: Option<&str>,
    ) -> SQLResult<Branch> {
        let current_operation_hash = OperationState::get_operation(conn);
        let mut stmt = conn.prepare_cached("insert into branch (name, current_operation_hash, remote_name) values (?1, ?2, ?3) returning (id);").unwrap();

        let mut rows = stmt
            .query_map((branch_name, current_operation_hash, remote_name), |row| {
                Ok(Branch {
                    id: row.get(0)?,
                    name: branch_name.to_string(),
                    current_operation_hash,
                    remote_name: remote_name.map(|s| s.to_string()),
                })
            })
            .unwrap();
        rows.next().unwrap()
    }

    pub fn delete(conn: &Connection, branch_id: i64) -> Result<(), BranchError> {
        if let Some(current_branch) = OperationState::get_current_branch(conn)
            && current_branch == branch_id
        {
            return Err(BranchError::CannotDelete(
                "Unable to delete the branch that is currently active.".to_string(),
            ));
        }
        conn.execute("delete from branch where id = ?1", (branch_id,))?;
        Ok(())
    }

    pub fn all(conn: &Connection) -> Vec<Branch> {
        Branch::query(conn, "select * from branch;", params![])
    }

    pub fn get_by_name(conn: &Connection, branch_name: &str) -> Option<Branch> {
        let mut branch: Option<Branch> = None;
        let results = Branch::query(
            conn,
            "select * from branch where name = ?1",
            params![branch_name],
        );
        for result in results.iter() {
            branch = Some(result.clone());
        }
        branch
    }

    pub fn get_by_id(conn: &Connection, branch_id: i64) -> Option<Branch> {
        let mut branch: Option<Branch> = None;
        for result in Branch::query(
            conn,
            "select * from branch where id = ?1",
            params![Value::from(branch_id)],
        )
        .iter()
        {
            branch = Some(result.clone());
        }
        branch
    }

    pub fn set_current_operation(conn: &Connection, branch_id: i64, operation_hash: &HashId) {
        conn.execute(
            "UPDATE branch set current_operation_hash = ?2 where id = ?1",
            params![branch_id, operation_hash],
        )
        .unwrap();
    }

    pub fn get_operations(conn: &Connection, branch_id: i64) -> Vec<Operation> {
        let branch = Branch::get_by_id(conn, branch_id)
            .unwrap_or_else(|| panic!("No branch with id {branch_id}."));
        let hashes = Operation::get_upstream(conn, &branch.current_operation_hash.unwrap());
        hashes
            .iter()
            .map(|hash| Operation::get_by_id(conn, hash).unwrap())
            .collect::<Vec<Operation>>()
    }

    /// Associate a branch with a remote
    pub fn set_remote(
        conn: &Connection,
        branch_id: i64,
        remote_name: Option<&str>,
    ) -> SQLResult<()> {
        let query = "UPDATE branch SET remote_name = ?1 WHERE id = ?2";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![remote_name, branch_id])?;
        Ok(())
    }

    /// Associate a branch with a remote with validation
    pub fn set_remote_validated(
        conn: &Connection,
        branch_id: i64,
        remote_name: Option<&str>,
    ) -> Result<(), RemoteError> {
        // If setting a remote name, validate that it exists
        if let Some(name) = remote_name {
            Remote::get_by_name(conn, name)?;
        }

        let query = "UPDATE branch SET remote_name = ?1 WHERE id = ?2";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![remote_name, branch_id])?;
        Ok(())
    }

    /// Get the remote associated with a branch
    pub fn get_remote(conn: &Connection, branch_id: i64) -> Option<String> {
        let query = "SELECT remote_name FROM branch WHERE id = ?1";
        let mut stmt = conn.prepare(query).ok()?;
        let mut rows = stmt
            .query_map(params![branch_id], |row| row.get::<_, Option<String>>(0))
            .ok()?;

        if let Some(Ok(remote_name)) = rows.next() {
            remote_name
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Defaults {
    pub id: i64,
    pub db_name: Option<String>,
    pub collection_name: Option<String>,
    pub remote_name: Option<String>,
}

impl Query for Defaults {
    type Model = Defaults;

    const TABLE_NAME: &'static str = "defaults";

    fn process_row(row: &Row) -> Self::Model {
        Defaults {
            id: row.get(0).unwrap(),
            db_name: row.get(1).unwrap(),
            collection_name: row.get(2).unwrap(),
            remote_name: row.get(3).unwrap(),
        }
    }
}

impl Defaults {
    /// Set the default remote by name
    pub fn set_default_remote(
        conn: &Connection,
        remote_name: Option<&str>,
    ) -> Result<(), RemoteError> {
        // If setting a remote name, validate that it exists
        if let Some(name) = remote_name {
            Remote::get_by_name(conn, name)?;
        }

        let query = "UPDATE defaults SET remote_name = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![remote_name])?;
        Ok(())
    }

    pub fn set_default_remote_compat(
        conn: &Connection,
        remote_name: Option<&str>,
    ) -> SQLResult<()> {
        let query = "UPDATE defaults SET remote_name = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![remote_name])?;
        Ok(())
    }

    /// Get the default remote name
    pub fn get_default_remote(conn: &Connection) -> Option<String> {
        let query = "SELECT remote_name FROM defaults WHERE id = 1";
        let mut stmt = conn.prepare(query).ok()?;
        let mut rows = stmt
            .query_map(params![], |row| row.get::<_, Option<String>>(0))
            .ok()?;

        if let Some(Ok(remote_name)) = rows.next() {
            remote_name
        } else {
            None
        }
    }

    /// Helper method to get the default remote URL by resolving the remote name
    pub fn get_default_remote_url(conn: &Connection) -> Option<String> {
        if let Some(remote_name) = Self::get_default_remote(conn) {
            if let Some(remote) = Remote::get_by_name_optional(conn, &remote_name) {
                Some(remote.url)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get the defaults record
    pub fn get(conn: &Connection) -> Option<Defaults> {
        let query = "SELECT id, db_name, collection_name, remote_name FROM defaults WHERE id = 1";
        Self::get_single(conn, query, params![]).ok()
    }

    /// Helper method to get a single defaults record using the Query trait
    fn get_single(
        conn: &Connection,
        query: &str,
        params: &[&dyn rusqlite::ToSql],
    ) -> SQLResult<Defaults> {
        let mut stmt = conn.prepare(query)?;
        let mut rows = stmt.query_map(params, |row| Ok(Self::process_row(row)))?;

        if let Some(row) = rows.next() {
            row
        } else {
            Err(rusqlite::Error::QueryReturnedNoRows)
        }
    }
}

pub struct OperationState {}

impl OperationState {
    pub fn set_operation(conn: &Connection, op_hash: &HashId) {
        let mut stmt = conn
            .prepare(
                "INSERT INTO operation_state (id, operation_hash)
          VALUES (1, ?1)
          ON CONFLICT (id) DO
          UPDATE SET operation_hash=excluded.operation_hash;",
            )
            .unwrap();
        stmt.execute([op_hash]).unwrap();
        let branch_id = OperationState::get_current_branch(conn).expect("No current branch set.");
        Branch::set_current_operation(conn, branch_id, op_hash);
    }

    pub fn get_operation(conn: &Connection) -> Option<HashId> {
        let mut hash: Option<HashId> = None;
        let mut stmt = conn
            .prepare("SELECT operation_hash from operation_state where id = 1;")
            .unwrap();
        let rows = stmt.query_map((), |row| row.get(0)).unwrap();
        for row in rows {
            hash = row.unwrap();
        }
        hash
    }

    pub fn set_branch(conn: &Connection, branch_name: &str) {
        let branch = Branch::get_by_name(conn, branch_name)
            .unwrap_or_else(|| panic!("No branch named {branch_name}."));
        let mut stmt = conn
            .prepare(
                "INSERT INTO operation_state (id, branch_id)
          VALUES (1, ?1)
          ON CONFLICT (id) DO
          UPDATE SET branch_id=excluded.branch_id;",
            )
            .unwrap();
        stmt.execute(params![branch.id]).unwrap();
        if let Some(current_branch_id) = OperationState::get_current_branch(conn) {
            if current_branch_id != branch.id {
                panic!("Failed to set branch to {branch_name}");
            }
        } else {
            panic!("Failed to set branch.");
        }
    }

    pub fn get_current_branch(conn: &Connection) -> Option<i64> {
        let mut id: Option<i64> = None;
        let mut stmt = conn
            .prepare("SELECT branch_id from operation_state where id = 1;")
            .unwrap();
        let rows = stmt.query_map((), |row| row.get(0)).unwrap();
        for row in rows {
            id = row.unwrap();
        }
        id
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        io::{Cursor, Write},
    };

    use tempfile::NamedTempFile;

    use super::*;
    use crate::{
        files::GenDatabase,
        test_helpers::{create_operation, get_connection, get_operation_connection, setup_gen_dir},
    };

    #[cfg(test)]
    mod defaults {
        use super::*;

        #[test]
        fn test_writes_operation_hash() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            let operation =
                Operation::create(op_conn, "test", &HashId::convert_str("some-hash")).unwrap();
            OperationState::set_operation(op_conn, &operation.hash);
            assert_eq!(
                OperationState::get_operation(op_conn).unwrap(),
                operation.hash
            );
        }

        #[test]
        fn test_default_remote_functionality() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            // Create test remotes
            Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();
            Remote::create(op_conn, "upstream", "https://upstream.com/repo.gen").unwrap();

            // Test getting default remote when none is set
            assert_eq!(Defaults::get_default_remote(op_conn), None);
            assert_eq!(Defaults::get_default_remote_url(op_conn), None);

            // Test setting default remote
            Defaults::set_default_remote(op_conn, Some("origin")).unwrap();
            assert_eq!(
                Defaults::get_default_remote(op_conn),
                Some("origin".to_string())
            );
            assert_eq!(
                Defaults::get_default_remote_url(op_conn),
                Some("https://example.com/repo.gen".to_string())
            );

            // Test changing default remote
            Defaults::set_default_remote(op_conn, Some("upstream")).unwrap();
            assert_eq!(
                Defaults::get_default_remote(op_conn),
                Some("upstream".to_string())
            );
            assert_eq!(
                Defaults::get_default_remote_url(op_conn),
                Some("https://upstream.com/repo.gen".to_string())
            );

            // Test clearing default remote
            Defaults::set_default_remote(op_conn, None).unwrap();
            assert_eq!(Defaults::get_default_remote(op_conn), None);
            assert_eq!(Defaults::get_default_remote_url(op_conn), None);

            // Test getting URL for non-existent remote (using the compat method to bypass validation)
            Defaults::set_default_remote_compat(op_conn, Some("nonexistent")).unwrap();
            assert_eq!(
                Defaults::get_default_remote(op_conn),
                Some("nonexistent".to_string())
            );
            assert_eq!(Defaults::get_default_remote_url(op_conn), None);
        }

        #[test]
        fn test_defaults_get() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            // Test getting defaults record
            let defaults = Defaults::get(op_conn).unwrap();
            assert_eq!(defaults.id, 1);
            assert_eq!(defaults.db_name, None);
            assert_eq!(defaults.collection_name, None);
            assert_eq!(defaults.remote_name, None);

            // Set a default remote and test again (using compat method to bypass validation)
            Defaults::set_default_remote_compat(op_conn, Some("test-remote")).unwrap();
            let defaults = Defaults::get(op_conn).unwrap();
            assert_eq!(defaults.remote_name, Some("test-remote".to_string()));
        }
    }

    #[cfg(test)]
    mod remote {
        use super::*;

        #[test]
        fn test_validate_remote_name() {
            // Valid names
            assert!(Remote::validate_name("origin").is_ok());
            assert!(Remote::validate_name("my-remote").is_ok());
            assert!(Remote::validate_name("remote_1").is_ok());
            assert!(Remote::validate_name("test123").is_ok());

            // Invalid names
            assert!(Remote::validate_name("").is_err());
            assert!(Remote::validate_name("remote with spaces").is_err());
            assert!(Remote::validate_name("remote@special").is_err());
            assert!(Remote::validate_name("remote.dot").is_err());
        }

        #[test]
        fn test_validate_url() {
            // Valid URLs
            assert!(Remote::validate_url("https://genhub.bio/user/repo.gen").is_ok());
            assert!(Remote::validate_url("http://example.com/repo").is_ok());
            assert!(Remote::validate_url("ssh://git@genhub.bio/user/repo.gen").is_ok());
            assert!(Remote::validate_url("/path/to/local/repo").is_ok());
            assert!(Remote::validate_url("user@host:path/to/repo").is_ok());

            // Invalid URLs
            assert!(Remote::validate_url("").is_err());
            assert!(Remote::validate_url("not-a-url").is_err());

            assert!(Remote::validate_url("ftp://invalid-protocol.com").is_err());
        }
    }

    mod branch {
        use super::*;

        #[test]
        fn test_branch_set_remote_valid() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            // Get database UUID and setup database

            // Create test remotes
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Create test branch
            let branch = Branch::get_or_create(op_conn, "test_branch");

            // Initially, branch should have no remote
            assert_eq!(Branch::get_remote(op_conn, branch.id), None);

            // Set remote
            let result = Branch::set_remote(op_conn, branch.id, Some("origin"));
            assert!(result.is_ok());

            // Verify remote was set
            assert_eq!(
                Branch::get_remote(op_conn, branch.id),
                Some("origin".to_string())
            );
        }

        #[test]
        fn test_branch_set_remote_nonexistent() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            // Get database UUID and setup database

            // Create test branch
            let branch = Branch::get_or_create(op_conn, "test_branch");

            // Try to set a remote that doesn't exist - should fail due to foreign key constraint
            let result = Branch::set_remote(op_conn, branch.id, Some("nonexistent"));
            assert!(result.is_err());

            // Verify branch still has no remote
            assert_eq!(Branch::get_remote(op_conn, branch.id), None);
        }

        #[test]
        fn test_branch_clear_remote() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            // Get database UUID and setup database

            // Create test remotes
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Create test branch
            let branch = Branch::get_or_create(op_conn, "test_branch");

            // Set a remote first
            Branch::set_remote(op_conn, branch.id, Some("origin")).unwrap();
            assert_eq!(
                Branch::get_remote(op_conn, branch.id),
                Some("origin".to_string())
            );

            // Clear the remote
            Branch::set_remote(op_conn, branch.id, None).unwrap();
            assert_eq!(Branch::get_remote(op_conn, branch.id), None);
        }

        #[test]
        fn test_branch_remote_cascade_on_remote_delete() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

            // Get database UUID and setup database

            // Create test remotes
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Create a branch and associate it with a remote
            let branch = Branch::get_or_create(op_conn, "test_branch_cascade");
            Branch::set_remote(op_conn, branch.id, Some("origin")).unwrap();

            // Verify the association
            assert_eq!(
                Branch::get_remote(op_conn, branch.id),
                Some("origin".to_string())
            );

            // Delete the remote
            Remote::delete(op_conn, "origin").unwrap();

            // Verify the branch remote association was set to null
            assert_eq!(Branch::get_remote(op_conn, branch.id), None);

            // Verify the branch still exists
            let branch_from_db = Branch::get_by_id(op_conn, branch.id);
            assert!(branch_from_db.is_some());
            assert_eq!(branch_from_db.unwrap().remote_name, None);
        }
    }

    #[test]
    fn test_create_operation_adds_database() {
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        let db_uuid = crate::metadata::get_db_uuid(conn);
        let gen_db = GenDatabase::create(op_conn, &db_uuid, "foo.db", "/foo.db").unwrap();

        let op = create_operation(
            conn,
            op_conn,
            "something.fa",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-1"),
        );

        let databases = GenDatabase::query_by_operations(op_conn, &[op.hash]).unwrap();
        assert_eq!(databases[&op.hash], vec![gen_db]);
    }

    #[test]
    fn test_gets_operations_of_branch() {
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-1"),
        );
        // operations will be made in ascending order.
        // The branch topology is as follows. () indicate where a branch starts
        //
        //                     -> 4 -> 5
        //                   /
        //         -> 2 -> 3 (branch-1-sub-1)
        //        /
        //      branch-1
        //    /
        //   1 (main, branch-1, branch-2)
        //    \
        //    branch-2
        //       \
        //        -> 6 -> 7 (branch-2-midpoint-1) -> 8 (branch-2-sub-1)
        //                 \                           \
        //                   -> 12 -> 13                9 -> 10 -> 11
        //
        //
        //
        //
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-2"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-3"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-4"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-5"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-1"));
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-6"),
        );
        let branch_2_midpoint = create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-7"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-8"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-9"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-10"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-11"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-7"));
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-12"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-13"),
        );

        OperationState::set_operation(op_conn, &HashId::convert_str("op-3"));
        let branch_1 = Branch::get_or_create(op_conn, "branch-1");
        OperationState::set_operation(op_conn, &HashId::convert_str("op-8"));
        let branch_2 = Branch::get_or_create(op_conn, "branch-2");
        OperationState::set_operation(op_conn, &HashId::convert_str("op-5"));
        let branch_1_sub_1 = Branch::get_or_create(op_conn, "branch-1-sub-1");
        OperationState::set_operation(op_conn, &HashId::convert_str("op-11"));
        let branch_2_sub_1 = Branch::get_or_create(op_conn, "branch-2-sub-1");
        OperationState::set_operation(op_conn, &HashId::convert_str("op-13"));
        let branch_2_midpoint_1 = Branch::get_or_create(op_conn, "branch-2-midpoint-1");

        let ops = Branch::get_operations(op_conn, branch_2_midpoint_1.id)
            .iter()
            .map(|f| f.hash)
            .collect::<Vec<_>>();
        assert_eq!(
            ops,
            vec![
                HashId::convert_str("op-1"),
                HashId::convert_str("op-6"),
                HashId::convert_str("op-7"),
                HashId::convert_str("op-12"),
                HashId::convert_str("op-13")
            ]
        );

        let ops = Branch::get_operations(op_conn, branch_1.id)
            .iter()
            .map(|f| f.hash)
            .collect::<Vec<_>>();
        assert_eq!(
            ops,
            vec![
                HashId::convert_str("op-1"),
                HashId::convert_str("op-2"),
                HashId::convert_str("op-3")
            ]
        );

        let ops = Branch::get_operations(op_conn, branch_2.id)
            .iter()
            .map(|f| f.hash)
            .collect::<Vec<_>>();
        assert_eq!(
            ops,
            vec![
                HashId::convert_str("op-1"),
                HashId::convert_str("op-6"),
                HashId::convert_str("op-7"),
                HashId::convert_str("op-8")
            ]
        );

        let ops = Branch::get_operations(op_conn, branch_1_sub_1.id)
            .iter()
            .map(|f| f.hash)
            .collect::<Vec<_>>();
        assert_eq!(
            ops,
            vec![
                HashId::convert_str("op-1"),
                HashId::convert_str("op-2"),
                HashId::convert_str("op-3"),
                HashId::convert_str("op-4"),
                HashId::convert_str("op-5")
            ]
        );

        let ops = Branch::get_operations(op_conn, branch_2_sub_1.id)
            .iter()
            .map(|f: &Operation| f.hash)
            .collect::<Vec<_>>();
        assert_eq!(
            ops,
            vec![
                HashId::convert_str("op-1"),
                HashId::convert_str("op-6"),
                HashId::convert_str("op-7"),
                HashId::convert_str("op-8"),
                HashId::convert_str("op-9"),
                HashId::convert_str("op-10"),
                HashId::convert_str("op-11")
            ]
        );
    }

    #[test]
    fn test_graph_representation() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // operations will be made in ascending order.
        // The branch topology is as follows. () indicate where a branch starts
        //
        //
        //
        //    branch-3   /-> 7
        //    main      1 -> 2 -> 3
        //    branch-1             \-> 4 -> 5
        //    branch-2                  \-> 6

        let mut expected_graph = OperationGraph::new();
        expected_graph.add_edge(HashId::convert_str("op-1"), HashId::convert_str("op-2"), ());
        expected_graph.add_edge(HashId::convert_str("op-2"), HashId::convert_str("op-3"), ());
        expected_graph.add_edge(HashId::convert_str("op-3"), HashId::convert_str("op-4"), ());
        expected_graph.add_edge(HashId::convert_str("op-4"), HashId::convert_str("op-5"), ());
        expected_graph.add_edge(HashId::convert_str("op-4"), HashId::convert_str("op-6"), ());
        expected_graph.add_edge(HashId::convert_str("op-1"), HashId::convert_str("op-7"), ());

        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-1")).unwrap();
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-2")).unwrap();
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-3")).unwrap();
        Branch::get_or_create(op_conn, "branch-1");
        OperationState::set_branch(op_conn, "branch-1");
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-4")).unwrap();
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-5")).unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-4"));
        Branch::get_or_create(op_conn, "branch-2");
        OperationState::set_branch(op_conn, "branch-2");
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-6")).unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-1"));
        Branch::get_or_create(op_conn, "branch-3");
        OperationState::set_branch(op_conn, "branch-3");
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-7")).unwrap();
        let graph = Operation::get_operation_graph(op_conn);

        assert_eq!(
            graph.nodes().collect::<HashSet<_>>(),
            expected_graph.nodes().collect::<HashSet<_>>()
        );
        assert_eq!(
            graph.all_edges().collect::<HashSet<_>>(),
            expected_graph.all_edges().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn test_path_between() {
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        // operations will be made in ascending order.
        // The branch topology is as follows. () indicate where a branch starts
        //
        //
        //
        //    branch-3   /-> 7
        //    main      1 -> 2 -> 3
        //    branch-1             \-> 4 -> 5
        //    branch-2                  \-> 6

        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-1"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-2"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-3"),
        );
        Branch::get_or_create(op_conn, "branch-1");
        OperationState::set_branch(op_conn, "branch-1");
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-4"),
        );
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-5"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-4"));
        Branch::get_or_create(op_conn, "branch-2");
        OperationState::set_branch(op_conn, "branch-2");
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-6"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-1"));
        Branch::get_or_create(op_conn, "branch-3");
        OperationState::set_branch(op_conn, "branch-3");
        create_operation(
            conn,
            op_conn,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-7"),
        );
        assert_eq!(
            Operation::get_path_between(
                op_conn,
                HashId::convert_str("op-1"),
                HashId::convert_str("op-6")
            ),
            vec![
                (
                    HashId::convert_str("op-1"),
                    Direction::Outgoing,
                    HashId::convert_str("op-2")
                ),
                (
                    HashId::convert_str("op-2"),
                    Direction::Outgoing,
                    HashId::convert_str("op-3")
                ),
                (
                    HashId::convert_str("op-3"),
                    Direction::Outgoing,
                    HashId::convert_str("op-4")
                ),
                (
                    HashId::convert_str("op-4"),
                    Direction::Outgoing,
                    HashId::convert_str("op-6")
                ),
            ]
        );

        assert_eq!(
            Operation::get_path_between(
                op_conn,
                HashId::convert_str("op-7"),
                HashId::convert_str("op-1")
            ),
            vec![(
                HashId::convert_str("op-7"),
                Direction::Incoming,
                HashId::convert_str("op-1")
            ),]
        );

        assert_eq!(
            Operation::get_path_between(
                op_conn,
                HashId::convert_str("op-3"),
                HashId::convert_str("op-7")
            ),
            vec![
                (
                    HashId::convert_str("op-3"),
                    Direction::Incoming,
                    HashId::convert_str("op-2")
                ),
                (
                    HashId::convert_str("op-2"),
                    Direction::Incoming,
                    HashId::convert_str("op-1")
                ),
                (
                    HashId::convert_str("op-1"),
                    Direction::Outgoing,
                    HashId::convert_str("op-7")
                ),
            ]
        );
    }

    #[test]
    fn test_remote_create() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Test successful remote creation
        let remote = Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();
        assert_eq!(remote.name, "origin");
        assert_eq!(remote.url, "https://example.com/repo.gen");

        // Test duplicate name constraint violation
        let result = Remote::create(op_conn, "origin", "https://different.com/repo.gen");
        assert!(result.is_err());
    }

    #[test]
    fn test_remote_get_by_name() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Test getting non-existent remote
        let result = Remote::get_by_name_optional(op_conn, "nonexistent");
        assert!(result.is_none());

        // Create a remote and test retrieval
        Remote::create(op_conn, "upstream", "https://upstream.com/repo.gen").unwrap();
        let result = Remote::get_by_name_optional(op_conn, "upstream");
        assert!(result.is_some());
        let remote = result.unwrap();
        assert_eq!(remote.name, "upstream");
        assert_eq!(remote.url, "https://upstream.com/repo.gen");
    }

    #[test]
    fn test_remote_list_all() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Test empty list
        let remotes = Remote::list_all(op_conn);
        assert!(remotes.is_empty());

        // Create multiple remotes
        Remote::create(op_conn, "origin", "https://origin.com/repo.gen").unwrap();
        Remote::create(op_conn, "upstream", "https://upstream.com/repo.gen").unwrap();
        Remote::create(op_conn, "fork", "https://fork.com/repo.gen").unwrap();

        // Test list returns all remotes in alphabetical order
        let remotes = Remote::list_all(op_conn);
        assert_eq!(remotes.len(), 3);
        assert_eq!(remotes[0].name, "fork");
        assert_eq!(remotes[1].name, "origin");
        assert_eq!(remotes[2].name, "upstream");
    }

    #[test]
    fn test_remote_delete() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Create a remote
        Remote::create(op_conn, "temp", "https://temp.com/repo.gen").unwrap();

        // Verify it exists
        let remote = Remote::get_by_name_optional(op_conn, "temp");
        assert!(remote.is_some());

        // Delete the remote
        let result = Remote::delete(op_conn, "temp");
        assert!(result.is_ok());

        // Verify it's gone
        let remote = Remote::get_by_name_optional(op_conn, "temp");
        assert!(remote.is_none());

        // Test deleting non-existent remote (should return error)
        let result = Remote::delete(op_conn, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_remote_delete_with_branch_associations() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Create a remote
        Remote::create(op_conn, "test_remote", "https://test.com/repo.gen").unwrap();

        // Create a branch and associate it with the remote
        let branch = Branch::get_or_create(op_conn, "test_branch");

        // Set the remote association (this would be done by the Branch::set_remote method when implemented)
        op_conn
            .execute(
                "UPDATE branch SET remote_name = ?1 WHERE id = ?2",
                params!["test_remote", branch.id],
            )
            .unwrap();

        // Verify the association exists
        let remote_name: Option<String> = op_conn
            .query_row(
                "SELECT remote_name FROM branch WHERE id = ?1",
                params![branch.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remote_name, Some("test_remote".to_string()));

        // Delete the remote - this should succeed and automatically set branch remote_name to NULL
        let result = Remote::delete(op_conn, "test_remote");
        assert!(result.is_ok());

        // Verify the branch association was automatically cleared by the foreign key constraint
        let remote_name_after_delete: Option<String> = op_conn
            .query_row(
                "SELECT remote_name FROM branch WHERE id = ?1",
                params![branch.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remote_name_after_delete, None);

        // Verify the remote was actually deleted
        let remote = Remote::get_by_name_optional(op_conn, "test_remote");
        assert!(remote.is_none());
    }

    #[test]
    fn test_branch_set_remote() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Create a remote
        Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();

        // Create a branch
        let branch = Branch::get_or_create(op_conn, "test_branch");

        // Initially, branch should have no remote
        let remote = Branch::get_remote(op_conn, branch.id);
        assert_eq!(remote, None);

        // Set the remote association
        Branch::set_remote(op_conn, branch.id, Some("origin")).unwrap();

        // Verify the association was set
        let remote = Branch::get_remote(op_conn, branch.id);
        assert_eq!(remote, Some("origin".to_string()));

        // Clear the remote association
        Branch::set_remote(op_conn, branch.id, None).unwrap();

        // Verify the association was cleared
        let remote = Branch::get_remote(op_conn, branch.id);
        assert_eq!(remote, None);
    }

    #[test]
    fn test_branch_get_remote() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Create remotes
        Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();
        Remote::create(op_conn, "upstream", "https://upstream.com/repo.gen").unwrap();

        // Create branches
        let branch1 = Branch::get_or_create(op_conn, "branch1");
        let branch2 = Branch::get_or_create(op_conn, "branch2");

        // Set different remotes for each branch
        Branch::set_remote(op_conn, branch1.id, Some("origin")).unwrap();
        Branch::set_remote(op_conn, branch2.id, Some("upstream")).unwrap();

        // Verify each branch has the correct remote
        assert_eq!(
            Branch::get_remote(op_conn, branch1.id),
            Some("origin".to_string())
        );
        assert_eq!(
            Branch::get_remote(op_conn, branch2.id),
            Some("upstream".to_string())
        );

        // Test non-existent branch
        assert_eq!(Branch::get_remote(op_conn, 99999), None);
    }

    #[test]
    fn test_branch_create_with_remote() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Create a remote
        Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();

        // Create a branch with remote association
        let branch = Branch::create_with_remote(op_conn, "test_branch", Some("origin")).unwrap();

        // Verify the branch was created with the remote association
        assert_eq!(branch.remote_name, Some("origin".to_string()));
        assert_eq!(
            Branch::get_remote(op_conn, branch.id),
            Some("origin".to_string())
        );

        // Create a branch without remote association
        let branch2 = Branch::create_with_remote(op_conn, "test_branch2", None).unwrap();
        assert_eq!(branch2.remote_name, None);
        assert_eq!(Branch::get_remote(op_conn, branch2.id), None);
    }

    #[test]
    fn test_branch_process_row_with_remote() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Create a remote
        Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();

        // Create a branch with remote
        let branch = Branch::create_with_remote(op_conn, "test_branch", Some("origin")).unwrap();

        // Query the branch back to test process_row
        let branches = Branch::query(
            op_conn,
            "SELECT * FROM branch WHERE id = ?1",
            params![branch.id],
        );
        assert_eq!(branches.len(), 1);

        let queried_branch = &branches[0];
        assert_eq!(queried_branch.id, branch.id);
        assert_eq!(queried_branch.name, "test_branch");
        assert_eq!(queried_branch.remote_name, Some("origin".to_string()));
    }

    #[test]
    fn test_branch_set_remote_foreign_key_constraint() {
        setup_gen_dir();
        let op_conn = &get_operation_connection(None).unwrap();

        // Create a branch
        let branch = Branch::get_or_create(op_conn, "test_branch");

        // Try to set a remote that doesn't exist - this should fail due to foreign key constraint
        let result = Branch::set_remote(op_conn, branch.id, Some("nonexistent_remote"));
        assert!(result.is_err());

        // Verify the branch still has no remote
        let remote = Branch::get_remote(op_conn, branch.id);
        assert_eq!(remote, None);
    }

    #[test]
    fn operation_capnp_serialization() {
        use capnp::message::TypedBuilder;

        let model = Operation {
            hash: HashId::convert_str("test"),
            parent_hash: Some(HashId::convert_str("parent")),
            change_type: "foo".to_string(),
            created_on: 0,
        };

        let mut message = TypedBuilder::<operation::Owned>::new_default();
        let mut root = message.init_root();
        model.write_capnp(&mut root);

        let deserialized = Operation::read_capnp(root.into_reader());
        assert_eq!(model, deserialized);
    }

    #[test]
    fn operation_capnp_serialization_no_parent() {
        use capnp::message::TypedBuilder;

        let model = Operation {
            hash: HashId::convert_str("test"),
            parent_hash: None,
            change_type: "foo".to_string(),
            created_on: 1,
        };

        let mut message = TypedBuilder::<operation::Owned>::new_default();
        let mut root = message.init_root();
        model.write_capnp(&mut root);

        let deserialized = Operation::read_capnp(root.into_reader());
        assert_eq!(model, deserialized);
    }

    #[test]
    fn file_addition_capnp_serialization() {
        use capnp::message::TypedBuilder;

        let file_addition = FileAddition {
            id: HashId([42u8; 32]),
            file_path: "test/path.fasta".to_string(),
            file_type: FileTypes::Fasta,
            checksum: HashId([24u8; 32]),
        };

        let mut message =
            TypedBuilder::<crate::gen_models_capnp::file_addition::Owned>::new_default();
        let mut root = message.init_root();
        file_addition.write_capnp(&mut root);

        let deserialized = FileAddition::read_capnp(root.into_reader());
        assert_eq!(file_addition, deserialized);
    }

    #[test]
    fn operation_summary_capnp_serialization() {
        use capnp::message::TypedBuilder;

        let operation_summary = OperationSummary {
            id: 123,
            operation_hash: HashId::convert_str("op-hash-123"),
            summary: "Added new sequences from FASTA file".to_string(),
        };

        let mut message =
            TypedBuilder::<crate::gen_models_capnp::operation_summary::Owned>::new_default();
        let mut root = message.init_root();
        operation_summary.write_capnp(&mut root);

        let deserialized = OperationSummary::read_capnp(root.into_reader());
        assert_eq!(operation_summary, deserialized);
    }

    #[test]
    fn test_calculate_stream_hash() {
        let content = b"Hello, World!";
        let cursor = Cursor::new(content);
        let hash = calculate_stream_hash(cursor).unwrap();

        assert_eq!(hash.len(), 32);

        // Test consistency - same content should produce same hash
        let cursor2 = Cursor::new(content);
        let hash2 = calculate_stream_hash(cursor2).unwrap();
        assert_eq!(hash, hash2);

        // Test different content produces different hash
        let different_content = b"Hello, World!!";
        let cursor3 = Cursor::new(different_content);
        let hash3 = calculate_stream_hash(cursor3).unwrap();
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_calculate_file_checksum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = b"Test file content for checksum calculation";
        temp_file.write_all(content).unwrap();
        temp_file.flush().unwrap();

        let checksum = calculate_file_checksum(temp_file.path()).unwrap();

        assert_eq!(checksum.0.len(), 32);

        // Test consistency - same file should produce same checksum
        let checksum2 = calculate_file_checksum(temp_file.path()).unwrap();
        assert_eq!(checksum, checksum2);

        // Test with different file content
        let mut temp_file2 = NamedTempFile::new().unwrap();
        let different_content = b"Different test file content";
        temp_file2.write_all(different_content).unwrap();
        temp_file2.flush().unwrap();

        let checksum3 = calculate_file_checksum(temp_file2.path()).unwrap();
        assert_ne!(checksum, checksum3);
    }

    #[test]
    fn test_calculate_file_checksum_nonexistent_file() {
        let result = calculate_file_checksum("/nonexistent/file/path");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        ));
    }

    #[test]
    fn test_generate_file_addition_id_consistency() {
        let checksum = HashId([1u8; 32]);
        let file_path = "/path/to/file.txt";

        let id1 = FileAddition::generate_file_addition_id(&checksum, file_path);
        let id2 = FileAddition::generate_file_addition_id(&checksum, file_path);

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_paths() {
        let checksum = HashId([1u8; 32]);
        let file_path1 = "/path/to/file1.txt";
        let file_path2 = "/path/to/file2.txt";

        let id1 = FileAddition::generate_file_addition_id(&checksum, file_path1);
        let id2 = FileAddition::generate_file_addition_id(&checksum, file_path2);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_checksums() {
        let checksum1 = HashId([1u8; 32]);
        let checksum2 = HashId([2u8; 32]);
        let file_path = "/path/to/file.txt";

        let id1 = FileAddition::generate_file_addition_id(&checksum1, file_path);
        let id2 = FileAddition::generate_file_addition_id(&checksum2, file_path);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_file_addition_get_or_create() {
        setup_gen_dir();
        let conn = &get_operation_connection(None).unwrap();

        let mut temp_file = NamedTempFile::new().unwrap();
        let content = b"Test file content";
        temp_file.write_all(content).unwrap();
        temp_file.flush().unwrap();

        let test_file_path = temp_file.path().to_str().unwrap().to_string();

        let fa1 = FileAddition::get_or_create(conn, &test_file_path, FileTypes::Fasta)
            .expect("Failed to create FileAddition");

        assert_eq!(
            fa1.id,
            FileAddition::generate_file_addition_id(
                &calculate_file_checksum(temp_file.path()).unwrap(),
                &test_file_path
            )
        );

        // Second call with same file should return the same FileAddition
        let fa2 = FileAddition::get_or_create(conn, &test_file_path, FileTypes::Fasta)
            .expect("Failed to get existing FileAddition");

        assert_eq!(fa1, fa2);

        let mut temp_file2 = NamedTempFile::new().unwrap();
        let content = b"Test file content";
        temp_file2.write_all(content).unwrap();
        temp_file2.flush().unwrap();

        let test_file_path2 = temp_file2.path().to_str().unwrap();

        let fa3 = FileAddition::get_or_create(conn, test_file_path2, FileTypes::Fasta)
            .expect("Failed to create different FileAddition");

        assert_ne!(fa1.id, fa3.id);

        temp_file.write_all(b"new content").unwrap();
        temp_file.flush().unwrap();
        let fa1_new = FileAddition::get_or_create(conn, &test_file_path, FileTypes::Fasta)
            .expect("Failed to create FileAddition");

        assert_ne!(fa1.id, fa1_new.id);
    }
}
