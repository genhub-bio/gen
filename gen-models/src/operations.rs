use std::{
    collections::HashMap,
    convert::TryInto,
    io::{self, BufReader},
    path::{Path, PathBuf},
    rc::Rc,
    string::ToString,
};

use gen_core::{HashId, Workspace, calculate_hash, traits::Capnp};
use gen_graph::{OperationGraph, all_simple_paths};
use itertools::Itertools;
use petgraph::{Direction, graphmap::UnGraphMap};
use rusqlite::{Result as SQLResult, Row, params, types::Value};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    assets::{AssetUri, FileAssetUri},
    changesets::{
        ChangesetModels, DatabaseChangeset, get_changeset_dependencies_from_path,
        get_changeset_from_path, write_changeset,
    },
    db::{DbContext, OperationsConnection},
    errors::{
        AddFilesOperationError, FileAdditionError, FileStoreError, OperationError, RemoteError,
    },
    file_types::FileTypes,
    files::GenDatabase,
    gen_models_capnp::operation,
    metadata,
    session_operations::DependencyModels,
    traits::*,
};

#[derive(Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HashRange {
    pub from: Option<HashId>,
    pub to: Option<HashId>,
}

#[derive(Debug, Error)]
pub enum HashParseError {
    #[error("No current branch is checked out.")]
    NoCurrentBranch,
    #[error("Branch '{0}' not found.")]
    BranchNotFound(String),
    #[error("Branch '{0}' has no operations.")]
    EmptyBranch(String),
    #[error("Reference '{0}' is not a valid HEAD shorthand.")]
    InvalidHead(String),
    #[error("HEAD offset {0} is out of range for the current branch.")]
    HeadOffsetOutOfRange(usize),
    #[error("Reference '{0}' did not match any operation.")]
    OperationNotFound(String),
    #[error("Reference '{0}' matches multiple operations.")]
    OperationAmbiguous(String),
}

impl Operation {
    pub fn create(
        conn: &OperationsConnection,
        change_type: &str,
        hash: &HashId,
    ) -> SQLResult<Operation> {
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
        conn: &OperationsConnection,
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
        conn: &OperationsConnection,
        operation_hash: &HashId,
        file_addition_id: &HashId,
        filename: &str,
    ) -> SQLResult<()> {
        let query = "INSERT INTO operation_files (operation_hash, file_addition_id, filename) VALUES (?1, ?2, ?3)";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![operation_hash, file_addition_id, filename])?;
        Ok(())
    }

    pub fn add_database(
        conn: &OperationsConnection,
        operation_hash: &HashId,
        db_uuid: &str,
    ) -> SQLResult<()> {
        let query =
            "INSERT INTO operation_databases (operation_hash, database_uuid) VALUES (?1, ?2)";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![operation_hash, db_uuid])?;
        Ok(())
    }

    pub fn get_upstream(conn: &OperationsConnection, operation_hash: &HashId) -> Vec<HashId> {
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

    pub fn get_operation_graph(conn: &OperationsConnection) -> OperationGraph {
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
        conn: &OperationsConnection,
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

    pub fn search_hash(
        conn: &OperationsConnection,
        op_hash: &str,
    ) -> Result<Operation, HashParseError> {
        let matches = Operation::search_hashes(conn, op_hash);
        match matches.len() {
            0 => Err(HashParseError::OperationNotFound(op_hash.to_string())),
            1 => Ok(matches[0].clone()),
            _ => Err(HashParseError::OperationAmbiguous(op_hash.to_string())),
        }
    }

    pub fn search_hashes(conn: &OperationsConnection, op_hash: &str) -> Vec<Operation> {
        Operation::query(
            conn,
            "select * from operations where hex(hash) LIKE ?1",
            params![format!("{op_hash}%")],
        )
    }

    pub fn get_changeset_path(&self, workspace: &Workspace) -> PathBuf {
        workspace.changeset_path(&self.hash).join("changeset")
    }

    pub fn get_changeset_dependencies_path(&self, workspace: &Workspace) -> PathBuf {
        workspace.changeset_path(&self.hash).join("dependencies")
    }

    pub fn get_changeset(&self, workspace: &Workspace) -> DatabaseChangeset {
        let path = self.get_changeset_path(workspace);
        get_changeset_from_path(path)
    }

    pub fn get_changeset_dependencies(&self, workspace: &Workspace) -> DependencyModels {
        let path = self.get_changeset_dependencies_path(workspace);
        get_changeset_dependencies_from_path(path)
    }
}

pub fn parse_hash(conn: &OperationsConnection, input: &str) -> Result<HashRange, HashParseError> {
    if input.contains("..") {
        let mut it = input.split("..");
        let from_ref = it.next().unwrap_or_default();
        let to_ref = it.next().unwrap_or_default();
        return Ok(HashRange {
            from: Some(resolve_reference(conn, from_ref)?),
            to: Some(resolve_reference(conn, to_ref)?),
        });
    }

    Ok(HashRange {
        from: None,
        to: Some(resolve_reference(conn, input)?),
    })
}

fn resolve_reference(
    conn: &OperationsConnection,
    reference: &str,
) -> Result<HashId, HashParseError> {
    if reference.starts_with("HEAD") {
        return resolve_head(conn, reference);
    }

    if let Some(branch) = Branch::get_by_name(conn, reference) {
        if let Some(hash) = branch.current_operation_hash {
            return Ok(hash);
        }
        return Err(HashParseError::EmptyBranch(branch.name));
    }

    let operation = Operation::search_hash(conn, reference)?;
    Ok(operation.hash)
}

fn resolve_head(conn: &OperationsConnection, reference: &str) -> Result<HashId, HashParseError> {
    let branch_id =
        OperationState::get_current_branch(conn).ok_or(HashParseError::NoCurrentBranch)?;
    let branch = Branch::get_by_id(conn, branch_id)
        .ok_or_else(|| HashParseError::BranchNotFound(branch_id.to_string()))?;
    let operations = Branch::get_operations(conn, branch.id);
    if operations.is_empty() {
        return Err(HashParseError::EmptyBranch(branch.name));
    }
    if reference == "HEAD" {
        return Ok(operations.last().unwrap().hash);
    }
    if let Some(offset) = reference.strip_prefix("HEAD~") {
        let offset: usize = offset
            .parse()
            .map_err(|_| HashParseError::InvalidHead(reference.to_string()))?;
        let head_index = operations.len() - 1;
        let target_index = head_index
            .checked_sub(offset)
            .ok_or(HashParseError::HeadOffsetOutOfRange(offset))?;
        return Ok(operations[target_index].hash);
    }

    Err(HashParseError::InvalidHead(reference.to_string()))
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
    pub filename: String,
    pub file_path: String,
    pub file_type: FileTypes,
    pub checksum_override: Option<HashId>,
}

impl OperationFile {
    pub fn new(file_path: impl Into<String>) -> Self {
        let file_path = file_path.into();
        let file_type = FileTypes::infer_from_path(&file_path);
        let filename = Path::new(&file_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&file_path)
            .to_string();
        Self {
            filename,
            file_path,
            file_type,
            checksum_override: None,
        }
    }

    pub fn set_file_type(mut self, file_type: FileTypes) -> Self {
        self.file_type = file_type;
        self
    }
}

pub struct OperationInfo {
    pub files: Vec<OperationFile>,
    pub description: String,
}

pub fn add_files_operation(
    context: &DbContext,
    files: &[String],
    message: Option<&str>,
) -> Result<Operation, AddFilesOperationError> {
    let workspace = context.workspace();
    let operation_conn = context.operations().conn();
    let graph_conn = context.graph().conn();
    let db_uuid = metadata::get_db_uuid(graph_conn);

    let file_additions = files
        .iter()
        .map(|path| {
            let file_type = FileTypes::infer_from_path(path);
            let file_addition =
                FileAddition::get_or_create(workspace, operation_conn, path, file_type, None)?;
            Ok::<(FileAddition, String), FileAdditionError>((
                file_addition,
                OperationFile::new(path.clone()).filename,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let unique_file_additions = file_additions
        .into_iter()
        .unique_by(|(file_addition, filename)| (file_addition.id, filename.clone()))
        .collect::<Vec<_>>();

    let operation_hash = HashId(calculate_hash(
        &unique_file_additions
            .iter()
            .map(|(file_addition, filename)| format!("{}:{filename}", file_addition.id))
            .sorted()
            .join(":"),
    ));

    let operation = match Operation::create(operation_conn, "add-file", &operation_hash) {
        Ok(operation) => operation,
        Err(rusqlite::Error::SqliteFailure(err, _details))
            if err.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            return Err(AddFilesOperationError::OperationError(
                OperationError::NoChanges,
            ));
        }
        Err(err) => return Err(err.into()),
    };

    for (file_addition, filename) in &unique_file_additions {
        Operation::add_file(operation_conn, &operation.hash, &file_addition.id, filename)?;
    }

    Operation::add_database(operation_conn, &operation.hash, &db_uuid)?;

    let summary = message.map(str::to_string).unwrap_or_else(|| {
        if files.len() == 1 {
            format!("Add file {}", files[0])
        } else {
            format!("Add {} files", files.len())
        }
    });
    OperationSummary::create(operation_conn, &operation.hash, &summary);

    let gen_db = GenDatabase::get_by_uuid(operation_conn, &db_uuid)?;
    write_changeset(
        workspace,
        &operation,
        DatabaseChangeset {
            db_path: gen_db.path,
            changes: ChangesetModels::default(),
        },
        &DependencyModels::default(),
    );

    for file_addition in unique_file_additions
        .into_iter()
        .map(|(file_addition, _filename)| file_addition)
        .unique_by(|file_addition| file_addition.id)
    {
        file_addition.store_file(workspace)?;
    }

    Ok(operation)
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
    pub asset_uri: String,
    pub file_type: FileTypes,
    pub checksum: HashId,
}

impl Query for FileAddition {
    type Model = FileAddition;

    const TABLE_NAME: &'static str = "file_additions";

    fn process_row(row: &Row) -> Self::Model {
        Self::Model {
            id: row.get(0).unwrap(),
            asset_uri: row.get(1).unwrap(),
            file_type: row.get(2).unwrap(),
            checksum: row.get(3).unwrap(),
        }
    }
}

impl FileAddition {
    pub fn file_path(&self) -> &str {
        self.asset_uri
            .strip_prefix(FileAssetUri::SCHEME)
            .unwrap_or(&self.asset_uri)
    }

    pub fn get_or_create(
        workspace: &Workspace,
        conn: &OperationsConnection,
        file_path: &str,
        file_type: FileTypes,
        checksum_override: Option<HashId>,
    ) -> Result<FileAddition, FileAdditionError> {
        let asset_uri = <dyn AssetUri>::new(file_path);
        let checksum = asset_uri.checksum(workspace, checksum_override)?;
        let stored_asset_uri = asset_uri.stored_asset_uri(workspace, &checksum, file_type)?;
        asset_uri.ensure_asset(workspace, &checksum, file_type)?;

        let id = FileAssetUri::generate_file_addition_id(&checksum, &stored_asset_uri);

        let query = "INSERT INTO file_additions (id, asset_uri, file_type, checksum) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query).unwrap();

        let addition = FileAddition {
            id,
            asset_uri: stored_asset_uri.clone(),
            file_type,
            checksum,
        };

        match stmt.execute((&id, &stored_asset_uri, file_type, &checksum)) {
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
        conn: &OperationsConnection,
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
        conn: &OperationsConnection,
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

    pub fn store_file(&self, workspace: &Workspace) -> Result<(), FileStoreError> {
        let asset_uri = <dyn AssetUri>::new(&self.asset_uri);
        asset_uri.store_file(self, workspace)
    }

    pub fn hashed_filename(self) -> String {
        FileAssetUri::asset_filename(&self.checksum, self.file_type)
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
    pub fn create(
        conn: &OperationsConnection,
        operation_hash: &HashId,
        summary: &str,
    ) -> OperationSummary {
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

    pub fn set_message(conn: &OperationsConnection, id: i64, message: &str) -> SQLResult<()> {
        let query = "UPDATE operation_summaries SET summary = ?2 where id = ?1";
        let mut stmt = conn.prepare(query).unwrap();
        stmt.execute(params![id, message])?;
        Ok(())
    }

    pub fn query_by_operations(
        conn: &OperationsConnection,
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
        builder.set_asset_uri(&self.asset_uri);
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
            asset_uri: reader.get_asset_uri().unwrap().to_string().unwrap(),
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
    pub fn create(
        conn: &OperationsConnection,
        name: &str,
        url: &str,
    ) -> Result<Remote, RemoteError> {
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
    pub fn get_by_name(conn: &OperationsConnection, name: &str) -> Result<Remote, RemoteError> {
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
    pub fn get_by_name_optional(conn: &OperationsConnection, name: &str) -> Option<Remote> {
        Self::get_by_name(conn, name).ok()
    }

    /// List all remotes
    pub fn list_all(conn: &OperationsConnection) -> Vec<Remote> {
        Remote::query(
            conn,
            "SELECT name, url FROM remotes ORDER BY name",
            params![],
        )
    }

    /// Delete a remote by name
    pub fn delete(conn: &OperationsConnection, name: &str) -> Result<(), RemoteError> {
        // Check if remote exists first
        Self::get_by_name(conn, name)?;

        let query = "DELETE FROM remotes WHERE name = ?1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![name])?;
        Ok(())
    }

    /// Check if a remote exists
    pub fn exists(conn: &OperationsConnection, name: &str) -> bool {
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

#[derive(Debug, PartialEq, Error)]
pub enum BranchError {
    #[error("Cannot delete branch: {0}")]
    CannotDelete(String),
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
    #[error("Duplicate entry: {0}")]
    Duplicate(String),
    #[error("Couldn't create branch: {0}")]
    CannotCreate(String),
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
    pub fn get_or_create(
        conn: &OperationsConnection,
        branch_name: &str,
    ) -> Result<Branch, BranchError> {
        match Branch::create_with_remote(conn, branch_name, None) {
            Ok(res) => Ok(res),
            Err(BranchError::Duplicate(_)) => match Branch::get_by_name(conn, branch_name) {
                Some(branch) => Ok(branch),
                None => Err(BranchError::CannotCreate(
                    "Failed to create branch".to_string(),
                )),
            },
            Err(e) => Err(e),
        }
    }

    pub fn create_with_remote(
        conn: &OperationsConnection,
        branch_name: &str,
        remote_name: Option<&str>,
    ) -> Result<Branch, BranchError> {
        let current_operation_hash = OperationState::get_operation(conn);
        let mut stmt = match conn.prepare_cached("insert into branch (name, current_operation_hash, remote_name) values (?1, ?2, ?3) returning (id);") {
	    Ok(stmt) => stmt,
	    Err(e) => return Err(BranchError::SqliteError(e)),
	};

        let results: Result<_, _> =
            stmt.query_map((branch_name, current_operation_hash, remote_name), |row| {
                Ok(Branch {
                    id: row.get(0)?,
                    name: branch_name.to_string(),
                    current_operation_hash,
                    remote_name: remote_name.map(|s| s.to_string()),
                })
            });
        match results {
            Ok(mut results) => match results.next() {
                Some(Ok(branch)) => Ok(branch),
                Some(Err(rusqlite::Error::SqliteFailure(err, _)))
                    if err.code == rusqlite::ErrorCode::ConstraintViolation =>
                {
                    Err(BranchError::Duplicate(branch_name.to_string()))
                }
                Some(Err(e)) => Err(BranchError::SqliteError(e)),
                None => Err(BranchError::CannotCreate(
                    "Failed to create branch".to_string(),
                )),
            },
            Err(e) => Err(BranchError::SqliteError(e)),
        }
    }

    pub fn delete(conn: &OperationsConnection, branch_id: i64) -> Result<(), BranchError> {
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

    pub fn all(conn: &OperationsConnection) -> Vec<Branch> {
        Branch::query(conn, "select * from branch;", params![])
    }

    pub fn get_by_name(conn: &OperationsConnection, branch_name: &str) -> Option<Branch> {
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

    pub fn get_by_id(conn: &OperationsConnection, branch_id: i64) -> Option<Branch> {
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

    pub fn set_current_operation(
        conn: &OperationsConnection,
        branch_id: i64,
        operation_hash: &HashId,
    ) {
        conn.execute(
            "UPDATE branch set current_operation_hash = ?2 where id = ?1",
            params![branch_id, operation_hash],
        )
        .unwrap();
    }

    pub fn get_operations(conn: &OperationsConnection, branch_id: i64) -> Vec<Operation> {
        let branch = Branch::get_by_id(conn, branch_id)
            .unwrap_or_else(|| panic!("No branch with id {branch_id}."));
        if let Some(hash) = branch.current_operation_hash {
            let hashes = Operation::get_upstream(conn, &hash);
            hashes
                .iter()
                .map(|hash| Operation::get_by_id(conn, hash).unwrap())
                .collect::<Vec<Operation>>()
        } else {
            vec![]
        }
    }

    /// Associate a branch with a remote
    pub fn set_remote(
        conn: &OperationsConnection,
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
        conn: &OperationsConnection,
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
    pub fn get_remote(conn: &OperationsConnection, branch_id: i64) -> Option<String> {
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
        conn: &OperationsConnection,
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
        conn: &OperationsConnection,
        remote_name: Option<&str>,
    ) -> SQLResult<()> {
        let query = "UPDATE defaults SET remote_name = ?1 WHERE id = 1";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![remote_name])?;
        Ok(())
    }

    /// Get the default remote name
    pub fn get_default_remote(conn: &OperationsConnection) -> Option<String> {
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
    pub fn get_default_remote_url(conn: &OperationsConnection) -> Option<String> {
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
    pub fn get(conn: &OperationsConnection) -> Option<Defaults> {
        let query = "SELECT id, db_name, collection_name, remote_name FROM defaults WHERE id = 1";
        Self::get_single(conn, query, params![]).ok()
    }

    /// Helper method to get a single defaults record using the Query trait
    fn get_single(
        conn: &OperationsConnection,
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
    pub fn set_operation(conn: &OperationsConnection, op_hash: &HashId) {
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

    pub fn get_operation(conn: &OperationsConnection) -> Option<HashId> {
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

    pub fn set_branch(conn: &OperationsConnection, branch_name: &str) {
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

    pub fn get_current_branch(conn: &OperationsConnection) -> Option<i64> {
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
        fs,
        io::{Cursor, Write},
    };

    use tempfile::NamedTempFile;

    use super::*;
    use crate::{
        files::GenDatabase,
        test_helpers::{create_operation, setup_gen},
    };

    #[cfg(test)]
    mod defaults {
        use super::*;

        #[test]
        fn test_writes_operation_hash() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

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
            let context = setup_gen();
            let op_conn = context.operations().conn();

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
            let context = setup_gen();
            let op_conn = context.operations().conn();

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
            let context = setup_gen();
            let op_conn = context.operations().conn();

            // Get database UUID and setup database

            // Create test remotes
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Create test branch
            let branch = Branch::get_or_create(op_conn, "test_branch").unwrap();

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
            let context = setup_gen();
            let op_conn = context.operations().conn();

            // Get database UUID and setup database

            // Create test branch
            let branch = Branch::get_or_create(op_conn, "test_branch").unwrap();

            // Try to set a remote that doesn't exist - should fail due to foreign key constraint
            let result = Branch::set_remote(op_conn, branch.id, Some("nonexistent"));
            assert!(result.is_err());

            // Verify branch still has no remote
            assert_eq!(Branch::get_remote(op_conn, branch.id), None);
        }

        #[test]
        fn test_branch_clear_remote() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            // Get database UUID and setup database

            // Create test remotes
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Create test branch
            let branch = Branch::get_or_create(op_conn, "test_branch").unwrap();

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
            let context = setup_gen();
            let op_conn = context.operations().conn();

            // Get database UUID and setup database

            // Create test remotes
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Create a branch and associate it with a remote
            let branch = Branch::get_or_create(op_conn, "test_branch_cascade").unwrap();
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

    mod parse_hash {
        use super::*;

        #[test]
        fn test_parse_hash_head_and_range() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let op_1 =
                Operation::create(op_conn, "add", &HashId::convert_str("op-1-abc-123")).unwrap();
            let op_2 =
                Operation::create(op_conn, "add", &HashId::convert_str("op-2-abc-123")).unwrap();

            let head = parse_hash(op_conn, "HEAD").unwrap();
            assert_eq!(
                head,
                HashRange {
                    from: None,
                    to: Some(op_2.hash),
                }
            );

            let range = parse_hash(op_conn, "HEAD~1..HEAD").unwrap();
            assert_eq!(
                range,
                HashRange {
                    from: Some(op_1.hash),
                    to: Some(op_2.hash),
                }
            );
        }

        #[test]
        fn test_parse_hash_branch_and_partial() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let op_1 =
                Operation::create(op_conn, "add", &HashId::convert_str("op-1-xyz-123")).unwrap();
            let op_2 =
                Operation::create(op_conn, "add", &HashId::convert_str("op-2-xyz-123")).unwrap();

            let branch_ref = parse_hash(op_conn, "main").unwrap();
            assert_eq!(
                branch_ref,
                HashRange {
                    from: None,
                    to: Some(op_2.hash),
                }
            );

            let partial = format!("{}", op_1.hash);
            let prefix = &partial[..6];
            let resolved = parse_hash(op_conn, prefix).unwrap();
            assert_eq!(
                resolved,
                HashRange {
                    from: None,
                    to: Some(op_1.hash),
                }
            );
        }

        #[test]
        fn test_parse_hash_head_offset_out_of_range() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let _op = Operation::create(op_conn, "add", &HashId::convert_str("op-1")).unwrap();
            let result = parse_hash(op_conn, "HEAD~1");
            assert!(matches!(
                result,
                Err(HashParseError::HeadOffsetOutOfRange(1))
            ));
        }
    }

    mod search_hash {
        use super::*;

        #[test]
        fn test_search_hashes_returns_matches() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let _op_1 = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "abc0000000000000000000000000000000000000000000000000000000000001",
                ),
            )
            .unwrap();
            let _op_2 = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "abc0000000000000000000000000000000000000000000000000000000000002",
                ),
            )
            .unwrap();
            let _op_3 = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "def0000000000000000000000000000000000000000000000000000000000003",
                ),
            )
            .unwrap();

            let matches = Operation::search_hashes(op_conn, "abc");
            assert_eq!(matches.len(), 2);
        }

        #[test]
        fn test_search_hash_resolves_and_errors() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let op_unique = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "def0000000000000000000000000000000000000000000000000000000000001",
                ),
            )
            .unwrap();
            let _op_ambiguous = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "abc0000000000000000000000000000000000000000000000000000000000001",
                ),
            )
            .unwrap();
            let _op_ambiguous_2 = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "abc0000000000000000000000000000000000000000000000000000000000002",
                ),
            )
            .unwrap();

            let resolved = Operation::search_hash(op_conn, "def").unwrap();
            assert_eq!(resolved.hash, op_unique.hash);

            let ambiguous = Operation::search_hash(op_conn, "abc");
            assert!(matches!(
                ambiguous,
                Err(HashParseError::OperationAmbiguous(_))
            ));
        }
    }

    mod resolve_reference {
        use super::*;

        #[test]
        fn test_resolve_reference_branch_and_hash() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let op_unique = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "def0000000000000000000000000000000000000000000000000000000000001",
                ),
            )
            .unwrap();

            let branch_hash = resolve_reference(op_conn, "main").unwrap();
            assert_eq!(branch_hash, op_unique.hash);

            let hash_ref = resolve_reference(op_conn, "def").unwrap();
            assert_eq!(hash_ref, op_unique.hash);
        }

        #[test]
        fn test_resolve_reference_ambiguous() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let _op_1 = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "abc0000000000000000000000000000000000000000000000000000000000001",
                ),
            )
            .unwrap();
            let _op_2 = Operation::create(
                op_conn,
                "add",
                &HashId::pad_str(
                    "abc0000000000000000000000000000000000000000000000000000000000002",
                ),
            )
            .unwrap();

            let result = resolve_reference(op_conn, "abc");
            assert!(matches!(result, Err(HashParseError::OperationAmbiguous(_))));
        }
    }

    mod resolve_head {
        use super::*;

        #[test]
        fn test_resolve_head_variants() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let op_1 = Operation::create(op_conn, "add", &HashId::convert_str("op-1")).unwrap();
            let op_2 = Operation::create(op_conn, "add", &HashId::convert_str("op-2")).unwrap();

            let head = resolve_head(op_conn, "HEAD").unwrap();
            assert_eq!(head, op_2.hash);

            let head_prev = resolve_head(op_conn, "HEAD~1").unwrap();
            assert_eq!(head_prev, op_1.hash);
        }

        #[test]
        fn test_resolve_head_invalid() {
            let context = setup_gen();
            let op_conn = context.operations().conn();

            let branch = Branch::get_or_create(op_conn, "main").unwrap();
            OperationState::set_branch(op_conn, &branch.name);

            let _op = Operation::create(op_conn, "add", &HashId::convert_str("op-1")).unwrap();
            let result = resolve_head(op_conn, "HEAD~2");
            assert!(matches!(
                result,
                Err(HashParseError::HeadOffsetOutOfRange(2))
            ));
        }
    }

    #[test]
    fn test_create_operation_adds_database() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        let db_uuid = crate::metadata::get_db_uuid(conn);
        let gen_db = GenDatabase::create(op_conn, &db_uuid, "foo.db", "/foo.db").unwrap();

        let op = create_operation(
            &context,
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
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        create_operation(
            &context,
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
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-2"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-3"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-4"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-5"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-1"));
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-6"),
        );
        let _branch_2_midpoint = create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-7"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-8"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-9"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-10"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-11"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-7"));
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-12"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-13"),
        );

        OperationState::set_operation(op_conn, &HashId::convert_str("op-3"));
        let branch_1 = Branch::get_or_create(op_conn, "branch-1").unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-8"));
        let branch_2 = Branch::get_or_create(op_conn, "branch-2").unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-5"));
        let branch_1_sub_1 = Branch::get_or_create(op_conn, "branch-1-sub-1").unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-11"));
        let branch_2_sub_1 = Branch::get_or_create(op_conn, "branch-2-sub-1").unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-13"));
        let branch_2_midpoint_1 = Branch::get_or_create(op_conn, "branch-2-midpoint-1").unwrap();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

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
        Branch::get_or_create(op_conn, "branch-1").unwrap();
        OperationState::set_branch(op_conn, "branch-1");
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-4")).unwrap();
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-5")).unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-4"));
        Branch::get_or_create(op_conn, "branch-2").unwrap();
        OperationState::set_branch(op_conn, "branch-2");
        let _ = Operation::create(op_conn, "vcf_addition", &HashId::convert_str("op-6")).unwrap();
        OperationState::set_operation(op_conn, &HashId::convert_str("op-1"));
        Branch::get_or_create(op_conn, "branch-3").unwrap();
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
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

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
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-1"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-2"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-3"),
        );
        Branch::get_or_create(op_conn, "branch-1").unwrap();
        OperationState::set_branch(op_conn, "branch-1");
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-4"),
        );
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-5"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-4"));
        Branch::get_or_create(op_conn, "branch-2").unwrap();
        OperationState::set_branch(op_conn, "branch-2");
        create_operation(
            &context,
            "test.fasta",
            FileTypes::Fasta,
            "foo",
            HashId::convert_str("op-6"),
        );
        OperationState::set_operation(op_conn, &HashId::convert_str("op-1"));
        Branch::get_or_create(op_conn, "branch-3").unwrap();
        OperationState::set_branch(op_conn, "branch-3");
        create_operation(
            &context,
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
        let context = setup_gen();
        let op_conn = context.operations().conn();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

        // Create a remote
        Remote::create(op_conn, "test_remote", "https://test.com/repo.gen").unwrap();

        // Create a branch and associate it with the remote
        let branch = Branch::get_or_create(op_conn, "test_branch").unwrap();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

        // Create a remote
        Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();

        // Create a branch
        let branch = Branch::get_or_create(op_conn, "test_branch").unwrap();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

        // Create remotes
        Remote::create(op_conn, "origin", "https://example.com/repo.gen").unwrap();
        Remote::create(op_conn, "upstream", "https://upstream.com/repo.gen").unwrap();

        // Create branches
        let branch1 = Branch::get_or_create(op_conn, "branch1").unwrap();
        let branch2 = Branch::get_or_create(op_conn, "branch2").unwrap();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

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
        let context = setup_gen();
        let op_conn = context.operations().conn();

        // Create a branch
        let branch = Branch::get_or_create(op_conn, "test_branch").unwrap();

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
            asset_uri: "file://test/path.fasta".to_string(),
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
    fn test_operation_file_new_infers_file_type_from_path() {
        let operation_file = OperationFile::new("fixtures/sample.gb");

        assert_eq!(operation_file.filename, "sample.gb");
        assert_eq!(operation_file.file_path, "fixtures/sample.gb");
        assert_eq!(operation_file.file_type, FileTypes::GenBank);
        assert_eq!(operation_file.checksum_override, None);
    }

    #[test]
    fn test_file_addition_get_or_create() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let repo_root = context.workspace().repo_root().unwrap();

        let file1_path = repo_root.join("test_file.txt");
        fs::write(&file1_path, b"Test file content").unwrap();
        let file1_path_str = file1_path.to_string_lossy().to_string();
        let relative1 = file1_path
            .strip_prefix(&repo_root)
            .unwrap()
            .to_string_lossy()
            .to_string();

        let fa1 = FileAddition::get_or_create(
            context.workspace(),
            op_conn,
            &file1_path_str,
            FileTypes::Fasta,
            None,
        )
        .expect("Failed to create FileAddition");

        assert_eq!(fa1.asset_uri, FileAssetUri::asset_uri(&relative1));
        assert_eq!(fa1.file_path(), relative1);

        let checksum = calculate_file_checksum(&file1_path_str).unwrap();
        let relative1_id = FileAssetUri::generate_file_addition_id(
            &checksum,
            &FileAssetUri::asset_uri(&relative1),
        );

        assert_eq!(fa1.id, relative1_id);
        assert!(
            context
                .workspace()
                .asset_dir()
                .unwrap()
                .join(fa1.clone().hashed_filename())
                .exists()
        );

        // Second call with same file should return the same FileAddition
        let fa2 = FileAddition::get_or_create(
            context.workspace(),
            op_conn,
            &file1_path_str,
            FileTypes::Fasta,
            None,
        )
        .expect("Failed to get existing FileAddition");

        assert_eq!(fa1, fa2);

        let file2_path = repo_root.join("nested").join("file2.txt");
        fs::create_dir_all(file2_path.parent().unwrap()).unwrap();
        fs::write(&file2_path, b"Test file content").unwrap();
        let file2_path_str = file2_path.to_string_lossy().to_string();

        let fa3 = FileAddition::get_or_create(
            context.workspace(),
            op_conn,
            &file2_path_str,
            FileTypes::Fasta,
            None,
        )
        .expect("Failed to create different FileAddition");

        assert_ne!(fa1.id, fa3.id);

        fs::write(&file1_path, b"new content").unwrap();
        let fa1_new = FileAddition::get_or_create(
            context.workspace(),
            op_conn,
            &file1_path_str,
            FileTypes::Fasta,
            None,
        )
        .expect("Failed to create FileAddition");

        assert_ne!(fa1.id, fa1_new.id);

        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"Outside repo file").unwrap();
        outside_file.flush().unwrap();
        let outside_path = outside_file.path().to_string_lossy().to_string();
        let outside_checksum = calculate_file_checksum(outside_file.path()).unwrap();
        let outside_asset_path = format!(
            ".gen/assets/{outside_checksum}.{}",
            FileTypes::suffix(FileTypes::Fasta)
        );

        let outside = FileAddition::get_or_create(
            context.workspace(),
            op_conn,
            &outside_path,
            FileTypes::Fasta,
            None,
        )
        .expect("Failed to create external FileAddition");

        assert_eq!(
            outside.asset_uri,
            FileAssetUri::asset_uri(&outside_asset_path)
        );
        assert_eq!(outside.file_path(), outside_asset_path);
        assert!(
            context
                .workspace()
                .asset_dir()
                .unwrap()
                .join(outside.clone().hashed_filename())
                .exists()
        );
    }

    #[test]
    fn test_file_addition_get_or_create_http_uri() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let asset_uri = "https://example.com/assets/reference.fa";

        let addition = FileAddition::get_or_create(
            context.workspace(),
            op_conn,
            asset_uri,
            FileTypes::Fasta,
            None,
        )
        .expect("Failed to create HTTP FileAddition");

        assert_eq!(addition.asset_uri, asset_uri);
        assert_eq!(addition.file_path(), asset_uri);
        assert!(
            !context
                .workspace()
                .asset_dir()
                .unwrap()
                .join(addition.clone().hashed_filename())
                .exists()
        );
    }

    #[test]
    fn test_file_addition_get_or_create_remote_uri() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let asset_uri = "s3://bucket/reference.fa";

        let addition = FileAddition::get_or_create(
            context.workspace(),
            op_conn,
            asset_uri,
            FileTypes::Fasta,
            None,
        )
        .expect("Failed to create remote FileAddition");

        assert_eq!(addition.asset_uri, asset_uri);
        assert_eq!(addition.file_path(), asset_uri);
    }

    #[test]
    fn test_add_files_operation_stores_shared_unmodified_asset_once_across_operations() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let operation_conn = context.operations().conn();
        let workspace = context.workspace();

        let db_uuid = metadata::get_db_uuid(graph_conn);
        GenDatabase::create(operation_conn, &db_uuid, "default", "default.db").unwrap();
        Branch::get_or_create(operation_conn, "main").unwrap();
        OperationState::set_branch(operation_conn, "main");

        let repo_root = workspace.repo_root().unwrap();
        fs::write(repo_root.join("shared.txt"), "shared contents").unwrap();
        fs::write(repo_root.join("unique.txt"), "unique contents").unwrap();

        let operation_1 =
            add_files_operation(&context, &["shared.txt".to_string()], Some("first")).unwrap();
        let operation_2 = add_files_operation(
            &context,
            &["shared.txt".to_string(), "unique.txt".to_string()],
            Some("second"),
        )
        .unwrap();

        let operation_1_files =
            FileAddition::get_files_for_operation(operation_conn, &operation_1.hash);
        let operation_2_files =
            FileAddition::get_files_for_operation(operation_conn, &operation_2.hash);

        let shared_file_1 = operation_1_files
            .iter()
            .find(|file| file.file_path() == "shared.txt")
            .unwrap();
        let shared_file_2 = operation_2_files
            .iter()
            .find(|file| file.file_path() == "shared.txt")
            .unwrap();

        assert_eq!(shared_file_1.id, shared_file_2.id);

        let assets_dir = workspace.asset_dir().unwrap();
        let asset_names = fs::read_dir(&assets_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(asset_names.len(), 2);
        assert!(asset_names.contains(&shared_file_1.clone().hashed_filename()));
        assert!(
            asset_names.contains(
                &operation_2_files
                    .iter()
                    .find(|file| file.file_path() == "unique.txt")
                    .unwrap()
                    .clone()
                    .hashed_filename()
            )
        );
    }

    #[test]
    fn test_add_files_operation_tracks_distinct_filenames_for_same_asset() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let operation_conn = context.operations().conn();

        let db_uuid = metadata::get_db_uuid(graph_conn);
        GenDatabase::create(operation_conn, &db_uuid, "default", "default.db").unwrap();
        Branch::get_or_create(operation_conn, "main").unwrap();
        OperationState::set_branch(operation_conn, "main");

        let outside_dir = tempfile::tempdir().unwrap();
        let alpha_path = outside_dir.path().join("alpha.fa");
        let beta_path = outside_dir.path().join("beta.fa");
        fs::write(&alpha_path, "shared contents").unwrap();
        fs::write(&beta_path, "shared contents").unwrap();

        let operation = add_files_operation(
            &context,
            &[
                alpha_path.to_string_lossy().to_string(),
                beta_path.to_string_lossy().to_string(),
            ],
            Some("same asset, different filenames"),
        )
        .unwrap();

        let mut stmt = operation_conn
            .prepare(
                "select filename from operation_files where operation_hash = ?1 order by filename",
            )
            .unwrap();
        let filenames = stmt
            .query_map([operation.hash], |row| row.get::<_, String>(0))
            .unwrap()
            .map(|row| row.unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            filenames,
            vec!["alpha.fa".to_string(), "beta.fa".to_string()]
        );
    }
}
