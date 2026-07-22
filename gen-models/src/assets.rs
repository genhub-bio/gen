#[cfg(not(target_os = "emscripten"))]
use std::sync::LazyLock;
use std::{
    collections::HashMap,
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    string::ToString,
    sync::{Arc, Mutex},
};

use gen_core::{HashId, Sha256Hash, Workspace, calculate_hash};
#[cfg(not(target_os = "emscripten"))]
use opendal::{blocking, services};
use rusqlite::{
    Row, ToSql, named_params, params,
    types::{FromSql, FromSqlResult, ToSqlOutput, ValueRef},
};
use sha2::{Digest, Sha256};
#[cfg(not(target_os = "emscripten"))]
use url::Position;
use url::Url;

use crate::{
    db::GraphConnection,
    errors::{FileAdditionError, FileStoreError, QueryError},
    operations::{FileAddition, calculate_reader_checksum},
    traits::Query,
};

#[cfg(not(target_os = "emscripten"))]
static OPENDAL_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build OpenDAL runtime")
});

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum OperationKind {
    AddFile,
    AnnotationFile,
    HistoryCommit,
    Other(String),
}

impl OperationKind {
    pub fn as_str(&self) -> &str {
        match self {
            OperationKind::AddFile => "add-file",
            OperationKind::AnnotationFile => "annotation-file",
            OperationKind::HistoryCommit => "history-commit",
            OperationKind::Other(kind) => kind.as_str(),
        }
    }
}

impl From<&str> for OperationKind {
    fn from(kind: &str) -> Self {
        match kind {
            "add-file" => OperationKind::AddFile,
            "annotation-file" => OperationKind::AnnotationFile,
            "history-commit" => OperationKind::HistoryCommit,
            other => OperationKind::Other(other.to_string()),
        }
    }
}

impl From<String> for OperationKind {
    fn from(kind: String) -> Self {
        match kind.as_str() {
            "add-file" => OperationKind::AddFile,
            "annotation-file" => OperationKind::AnnotationFile,
            "history-commit" => OperationKind::HistoryCommit,
            _ => OperationKind::Other(kind),
        }
    }
}

impl core::fmt::Display for OperationKind {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ToSql for OperationKind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for OperationKind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        value.as_str().map(OperationKind::from)
    }
}

#[cfg(not(target_os = "emscripten"))]
fn with_opendal_runtime<T>(f: impl FnOnce() -> T) -> T {
    let _guard = OPENDAL_RUNTIME.enter();
    f()
}

#[cfg(not(target_os = "emscripten"))]
fn opendal_file_addition_error(err: opendal::Error) -> FileAdditionError {
    FileAdditionError::FileReadError(io::Error::other(err))
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum AssetRole {
    Input,
    Annotation,
    AnnotationIndex,
    Other(String),
}

impl AssetRole {
    pub fn as_str(&self) -> &str {
        match self {
            AssetRole::Input => "input",
            AssetRole::Annotation => "annotation",
            AssetRole::AnnotationIndex => "annotation-index",
            AssetRole::Other(role) => role.as_str(),
        }
    }
}

impl From<&str> for AssetRole {
    fn from(role: &str) -> Self {
        match role {
            "input" => AssetRole::Input,
            "annotation" => AssetRole::Annotation,
            "annotation-index" => AssetRole::AnnotationIndex,
            other => AssetRole::Other(other.to_string()),
        }
    }
}

impl From<String> for AssetRole {
    fn from(role: String) -> Self {
        match role.as_str() {
            "input" => AssetRole::Input,
            "annotation" => AssetRole::Annotation,
            "annotation-index" => AssetRole::AnnotationIndex,
            _ => AssetRole::Other(role),
        }
    }
}

impl core::fmt::Display for AssetRole {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl ToSql for AssetRole {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_str().into())
    }
}

impl FromSql for AssetRole {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        value.as_str().map(AssetRole::from)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetRef {
    pub id: HashId,
    pub uri: String,
    pub file_type: String,
    pub checksum: Option<Sha256Hash>,
    pub size: Option<i64>,
    pub role: AssetRole,
    pub logical_path: Option<String>,
    pub name: Option<String>,
    pub created_on: i64,
}

pub struct Assets;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotationFileAssets {
    pub log_id: HashId,
    pub annotation: AssetRef,
    pub index: Option<AssetRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationLog {
    pub id: HashId,
    pub operation_kind: OperationKind,
    pub command: String,
    pub created_on: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationAsset {
    pub log_id: HashId,
    pub asset_ref_id: HashId,
    pub role: AssetRole,
}

impl Query for AssetRef {
    type Model = AssetRef;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "gen_asset_refs";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get("id").unwrap(),
            uri: row.get("uri").unwrap(),
            file_type: row.get("file_type").unwrap(),
            checksum: row.get("checksum").unwrap(),
            size: row.get("size").unwrap(),
            role: row.get("role").unwrap(),
            logical_path: row.get("logical_path").unwrap(),
            name: row.get("name").unwrap(),
            created_on: row.get("created_on").unwrap(),
        }
    }
}

impl Query for OperationLog {
    type Model = OperationLog;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "gen_operation_log";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get("id").unwrap(),
            operation_kind: row.get("operation_kind").unwrap(),
            command: row.get("command").unwrap(),
            created_on: row.get("created_on").unwrap(),
        }
    }
}

impl Query for OperationAsset {
    type Model = OperationAsset;

    const PRIMARY_KEY: &'static str = "log_id";
    const TABLE_NAME: &'static str = "gen_operation_assets";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            log_id: row.get("log_id").unwrap(),
            asset_ref_id: row.get("asset_ref_id").unwrap(),
            role: row.get("role").unwrap(),
        }
    }
}

impl AssetRef {
    pub fn id_hash(
        uri: &str,
        file_type: &str,
        checksum: Option<&Sha256Hash>,
        role: &AssetRole,
        logical_path: Option<&str>,
        name: Option<&str>,
    ) -> HashId {
        let checksum = checksum
            .map(|checksum| checksum.to_string())
            .unwrap_or_default();
        HashId(calculate_hash(&format!(
            "{uri}:{file_type}:{checksum}:{role}:{logical_path}:{name}",
            role = role.as_str(),
            logical_path = logical_path.unwrap_or_default(),
            name = name.unwrap_or_default(),
        )))
    }

    pub fn create(conn: &GraphConnection, asset_ref: &AssetRef) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_asset_refs \
             (id, uri, file_type, checksum, size, role, logical_path, name, created_on) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                asset_ref.id,
                asset_ref.uri,
                asset_ref.file_type,
                asset_ref.checksum,
                asset_ref.size,
                &asset_ref.role,
                asset_ref.logical_path,
                asset_ref.name,
                asset_ref.created_on
            ],
        )?;
        Ok(())
    }
}

impl Assets {
    pub fn get_branch_assets(
        conn: &GraphConnection,
        branch: &str,
    ) -> Result<HashMap<HashId, AssetRef>, QueryError> {
        let query = format!(
            "SELECT * FROM {} ORDER BY id",
            AssetRef::table_name_with_history_ref(Some(branch))
        );
        let assets = AssetRef::try_query(conn, &query, named_params! { ":history_ref": branch })?;
        Ok(assets
            .into_iter()
            .filter(|asset| LocalAssetUri::is_file_uri(&asset.uri))
            .map(|asset| (asset.id, asset))
            .collect())
    }

    pub fn get_annotation_files(
        conn: &GraphConnection,
        history_ref: Option<&str>,
    ) -> Result<Vec<AnnotationFileAssets>, QueryError> {
        let operation_logs_table = OperationLog::table_name_with_history_ref(history_ref);
        let operation_assets_table = OperationAsset::table_name_with_history_ref(history_ref);
        let asset_refs_table = AssetRef::table_name_with_history_ref(history_ref);
        let query = format!(
            "SELECT operation_logs.id AS log_id, \
                    annotation_assets.id AS annotation_id, \
                    annotation_assets.uri AS annotation_uri, \
                    annotation_assets.file_type AS annotation_file_type, \
                    annotation_assets.checksum AS annotation_checksum, \
                    annotation_assets.size AS annotation_size, \
                    annotation_assets.role AS annotation_role, \
                    annotation_assets.logical_path AS annotation_logical_path, \
                    annotation_assets.name AS annotation_name, \
                    annotation_assets.created_on AS annotation_created_on, \
                    index_assets.id AS index_id, \
                    index_assets.uri AS index_uri, \
                    index_assets.file_type AS index_file_type, \
                    index_assets.checksum AS index_checksum, \
                    index_assets.size AS index_size, \
                    index_assets.role AS index_role, \
                    index_assets.logical_path AS index_logical_path, \
                    index_assets.name AS index_name, \
                    index_assets.created_on AS index_created_on \
             FROM {operation_logs_table} operation_logs \
             JOIN {operation_assets_table} annotation_operation_assets \
               ON annotation_operation_assets.log_id = operation_logs.id \
              AND annotation_operation_assets.role = :annotation_role \
             JOIN {asset_refs_table} annotation_assets \
               ON annotation_assets.id = annotation_operation_assets.asset_ref_id \
             LEFT JOIN {operation_assets_table} index_operation_assets \
               ON index_operation_assets.log_id = operation_logs.id \
              AND index_operation_assets.role = :index_role \
             LEFT JOIN {asset_refs_table} index_assets \
               ON index_assets.id = index_operation_assets.asset_ref_id \
             WHERE operation_logs.operation_kind = :operation_kind \
             ORDER BY operation_logs.created_on, annotation_assets.created_on, \
                      annotation_assets.name"
        );
        let operation_kind = OperationKind::AnnotationFile;
        let annotation_role = AssetRole::Annotation;
        let index_role = AssetRole::AnnotationIndex;
        let mut query_params: Vec<(&str, &dyn ToSql)> = vec![
            (":operation_kind", &operation_kind),
            (":annotation_role", &annotation_role),
            (":index_role", &index_role),
        ];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let mut statement = conn.prepare(&query)?;
        let rows = statement.query_map(&query_params[..], |row| {
            let index = match row.get::<_, Option<HashId>>("index_id")? {
                Some(id) => Some(AssetRef {
                    id,
                    uri: row.get("index_uri")?,
                    file_type: row.get("index_file_type")?,
                    checksum: row.get("index_checksum")?,
                    size: row.get("index_size")?,
                    role: row.get("index_role")?,
                    logical_path: row.get("index_logical_path")?,
                    name: row.get("index_name")?,
                    created_on: row.get("index_created_on")?,
                }),
                None => None,
            };
            Ok(AnnotationFileAssets {
                log_id: row.get("log_id")?,
                annotation: AssetRef {
                    id: row.get("annotation_id")?,
                    uri: row.get("annotation_uri")?,
                    file_type: row.get("annotation_file_type")?,
                    checksum: row.get("annotation_checksum")?,
                    size: row.get("annotation_size")?,
                    role: row.get("annotation_role")?,
                    logical_path: row.get("annotation_logical_path")?,
                    name: row.get("annotation_name")?,
                    created_on: row.get("annotation_created_on")?,
                },
                index,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(QueryError::from)
    }
}

impl OperationLog {
    pub fn id_hash(operation_kind: &OperationKind, command: &str, created_on: i64) -> HashId {
        HashId(calculate_hash(&format!(
            "{operation_kind}:{command}:{created_on}"
        )))
    }

    pub fn create(conn: &GraphConnection, operation_log: &OperationLog) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_operation_log (id, operation_kind, command, created_on) \
             VALUES (?1, ?2, ?3, ?4)",
            params![
                operation_log.id,
                &operation_log.operation_kind,
                operation_log.command,
                operation_log.created_on
            ],
        )?;
        Ok(())
    }
}

impl OperationAsset {
    pub fn create(
        conn: &GraphConnection,
        operation_asset: &OperationAsset,
    ) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT OR IGNORE INTO gen_operation_assets (log_id, asset_ref_id, role) \
             VALUES (?1, ?2, ?3)",
            params![
                operation_asset.log_id,
                operation_asset.asset_ref_id,
                &operation_asset.role
            ],
        )?;
        Ok(())
    }

    pub fn by_log_id(conn: &GraphConnection, log_id: &HashId) -> Vec<Self> {
        Self::query(
            conn,
            "SELECT * FROM gen_operation_assets WHERE log_id = ?1",
            params![log_id],
        )
    }
}

#[doc(hidden)]
#[cfg(not(target_os = "emscripten"))]
pub struct OpenDalLocation {
    operator: blocking::Operator,
    path: String,
}

struct ChecksumState {
    hasher: Sha256,
    checksum: Option<Sha256Hash>,
    complete: bool,
}

#[derive(Clone)]
pub struct ChecksumHandle {
    state: Arc<Mutex<ChecksumState>>,
}

impl ChecksumHandle {
    pub fn checksum(&self) -> Option<Sha256Hash> {
        let state = self.state.lock().unwrap();
        if state.complete { state.checksum } else { None }
    }

    pub fn finalized_checksum(&self) -> Option<Sha256Hash> {
        self.state.lock().unwrap().checksum
    }
}

pub struct ChecksummedReader {
    inner: Box<dyn Read>,
    state: Arc<Mutex<ChecksumState>>,
}

impl ChecksummedReader {
    fn new(inner: Box<dyn Read>) -> Self {
        Self {
            inner,
            state: Arc::new(Mutex::new(ChecksumState {
                hasher: Sha256::new(),
                checksum: None,
                complete: false,
            })),
        }
    }

    pub fn checksum_handle(&self) -> ChecksumHandle {
        ChecksumHandle {
            state: Arc::clone(&self.state),
        }
    }
}

impl Read for ChecksummedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.inner.read(buf)?;
        let mut state = self.state.lock().unwrap();
        if bytes_read == 0 {
            if state.checksum.is_none() {
                let finalized = state.hasher.clone().finalize();
                state.checksum = Some(Sha256Hash(finalized.into()));
            }
            state.complete = true;
        } else {
            state.hasher.update(&buf[..bytes_read]);
        }
        Ok(bytes_read)
    }
}

impl Drop for ChecksummedReader {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        if state.checksum.is_none() {
            let finalized = state.hasher.clone().finalize();
            state.checksum = Some(Sha256Hash(finalized.into()));
        }
    }
}

#[cfg(not(target_os = "emscripten"))]
impl OpenDalLocation {
    fn new_fs(root: &Path, path: &Path) -> Result<Self, opendal::Error> {
        let path = path
            .strip_prefix(Path::new("/"))
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let operator = with_opendal_runtime(|| {
            let builder = services::Fs::default().root(&root.to_string_lossy());
            let op = opendal::Operator::new(builder)?.finish();
            blocking::Operator::new(op)
        })?;

        Ok(Self { operator, path })
    }

    fn from_workspace_path(workspace: &Workspace, path: &Path) -> Result<Self, FileAdditionError> {
        let repo_root = workspace.repo_root()?;

        if path.is_absolute() {
            Self::new_fs(Path::new("/"), path).map_err(opendal_file_addition_error)
        } else {
            Self::new_fs(&repo_root, path).map_err(opendal_file_addition_error)
        }
    }

    fn from_absolute_path(path: &Path) -> Result<Self, FileAdditionError> {
        if !path.is_absolute() {
            return Err(FileAdditionError::FileReadError(io::Error::new(
                io::ErrorKind::InvalidInput,
                "local path must be absolute",
            )));
        }

        Self::new_fs(Path::new("/"), path).map_err(opendal_file_addition_error)
    }

    fn writer_handle(&self) -> io::Result<WriteHandle> {
        self.operator
            .writer(&self.path)
            .map_err(io::Error::other)
            .map(|writer| writer.into_std_write())
    }

    fn from_remote_uri(asset_uri: &str) -> Result<Self, FileAdditionError> {
        let url = Url::parse(asset_uri)
            .map_err(|err| FileAdditionError::FileReadError(io::Error::other(err)))?;

        if url.scheme().eq_ignore_ascii_case("s3") {
            let bucket = url.host_str().ok_or_else(|| {
                FileAdditionError::FileReadError(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("s3 uri is missing bucket: {asset_uri}"),
                ))
            })?;

            let operator = with_opendal_runtime(|| {
                let builder = services::S3::default().bucket(bucket).allow_anonymous();
                let op = opendal::Operator::new(builder)?.finish();
                blocking::Operator::new(op)
            })
            .map_err(opendal_file_addition_error)?;

            return Ok(Self {
                operator,
                path: url.path().trim_start_matches('/').to_string(),
            });
        }

        let operator_uri = url[..Position::BeforePath].to_string();
        opendal::init_default_registry();
        let operator = with_opendal_runtime(|| blocking::Operator::from_uri(operator_uri.as_str()))
            .map_err(opendal_file_addition_error)?;

        Ok(Self {
            operator,
            path: url.path().trim_start_matches('/').to_string(),
        })
    }

    fn reader(self) -> Result<ChecksummedReader, FileAdditionError> {
        let reader = self
            .operator
            .reader(&self.path)
            .map_err(opendal_file_addition_error)?
            .into_std_read(..)
            .map_err(opendal_file_addition_error)?;
        Ok(ChecksummedReader::new(Box::new(reader)))
    }

    fn checksum(&self, display_path: &str) -> Result<Sha256Hash, FileAdditionError> {
        let reader = match self.operator.reader(&self.path) {
            Ok(reader) => reader,
            Err(err) => {
                return match err.kind() {
                    opendal::ErrorKind::NotFound => Ok(Sha256Hash::convert_str("non-existent")),
                    opendal::ErrorKind::PermissionDenied => Err(
                        FileAdditionError::FilePermissionDenied(display_path.to_string()),
                    ),
                    _ => Err(opendal_file_addition_error(err)),
                };
            }
        }
        .into_std_read(..)
        .map_err(opendal_file_addition_error)?;

        match calculate_reader_checksum(reader) {
            Ok(checksum) => Ok(checksum),
            Err(err) => match err.kind() {
                io::ErrorKind::NotFound => Ok(Sha256Hash::convert_str("non-existent")),
                io::ErrorKind::PermissionDenied => Err(FileAdditionError::FilePermissionDenied(
                    display_path.to_string(),
                )),
                _ => Err(FileAdditionError::FileReadError(err)),
            },
        }
    }

    fn copy_to_local_path(
        &self,
        workspace: &Workspace,
        destination_path: &Path,
    ) -> io::Result<u64> {
        let asset_location =
            Self::from_workspace_path(workspace, destination_path).map_err(io::Error::other)?;
        let mut reader = self
            .operator
            .reader(&self.path)
            .map_err(io::Error::other)?
            .into_std_read(..)
            .map_err(io::Error::other)?;
        let mut writer = asset_location
            .operator
            .writer(&asset_location.path)
            .map_err(io::Error::other)?
            .into_std_write();
        let bytes_copied = io::copy(&mut reader, &mut writer)?;
        writer.close()?;
        Ok(bytes_copied)
    }
}

// `LocalAssetUri`'s local (file://) file access is target-agnostic: `OpenDalLocation` backs it
// with opendal's fs service natively, and `LocalFsLocation` backs it with plain std::fs on
// emscripten, where opendal's tokio/mio dependency chain can't build at all. Remote (http/s3/...)
// asset access stays opendal-only and unavailable on emscripten; see `RemoteAssetUri::reader`.
#[cfg(not(target_os = "emscripten"))]
type Location = OpenDalLocation;
#[cfg(target_os = "emscripten")]
type Location = LocalFsLocation;

#[cfg(not(target_os = "emscripten"))]
type WriteHandle = opendal::blocking::StdWriter;
#[cfg(target_os = "emscripten")]
type WriteHandle = fs::File;

#[cfg(target_os = "emscripten")]
struct LocalFsLocation {
    path: PathBuf,
}

#[cfg(target_os = "emscripten")]
impl LocalFsLocation {
    fn from_workspace_path(workspace: &Workspace, path: &Path) -> Result<Self, FileAdditionError> {
        let repo_root = workspace.repo_root()?;
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            repo_root.join(path)
        };
        Ok(Self { path })
    }

    fn from_absolute_path(path: &Path) -> Result<Self, FileAdditionError> {
        if !path.is_absolute() {
            return Err(FileAdditionError::FileReadError(io::Error::new(
                io::ErrorKind::InvalidInput,
                "local path must be absolute",
            )));
        }

        Ok(Self {
            path: path.to_path_buf(),
        })
    }

    fn reader(self) -> Result<ChecksummedReader, FileAdditionError> {
        let file = fs::File::open(&self.path).map_err(FileAdditionError::FileReadError)?;
        Ok(ChecksummedReader::new(Box::new(file)))
    }

    fn checksum(&self, display_path: &str) -> Result<HashId, FileAdditionError> {
        let file = match fs::File::open(&self.path) {
            Ok(file) => file,
            Err(err) => {
                return match err.kind() {
                    io::ErrorKind::NotFound => Ok(HashId::convert_str("non-existent")),
                    io::ErrorKind::PermissionDenied => Err(
                        FileAdditionError::FilePermissionDenied(display_path.to_string()),
                    ),
                    _ => Err(FileAdditionError::FileReadError(err)),
                };
            }
        };

        match calculate_reader_checksum(file) {
            Ok(checksum) => Ok(checksum),
            Err(err) => match err.kind() {
                io::ErrorKind::NotFound => Ok(HashId::convert_str("non-existent")),
                io::ErrorKind::PermissionDenied => Err(FileAdditionError::FilePermissionDenied(
                    display_path.to_string(),
                )),
                _ => Err(FileAdditionError::FileReadError(err)),
            },
        }
    }

    fn copy_to_local_path(
        &self,
        _workspace: &Workspace,
        destination_path: &Path,
    ) -> io::Result<u64> {
        if let Some(parent) = destination_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&self.path, destination_path)
    }

    fn writer_handle(&self) -> io::Result<WriteHandle> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::File::create(&self.path)
    }
}

pub trait AssetUri {
    fn uri(&self) -> &str;

    fn reader(&self, workspace: &Workspace) -> Result<ChecksummedReader, FileAdditionError>;

    fn checksum(
        &self,
        workspace: &Workspace,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<Sha256Hash, FileAdditionError>;

    fn stored_asset_uri(
        &self,
        workspace: &Workspace,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError>;

    fn ensure_asset(
        &self,
        workspace: &Workspace,
        checksum: &Sha256Hash,
    ) -> Result<(), FileAdditionError>;

    fn store_file(
        &self,
        file_addition: &FileAddition,
        workspace: &Workspace,
    ) -> Result<(), FileStoreError>;

    fn hashed_filename(&self, checksum: &Sha256Hash) -> String {
        self.asset_filename(checksum)
    }

    fn suffix(&self) -> Option<String> {
        let path = if LocalAssetUri::is_file_uri(self.uri()) {
            LocalAssetUri::path_from_uri(self.uri())?
        } else if LocalAssetUri::has_uri_scheme(self.uri()) {
            Url::parse(self.uri()).ok()?.path().to_string()
        } else {
            self.uri().to_string()
        };

        Path::new(&path)
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|file_name| {
                file_name
                    .split_once('.')
                    .map(|(_, suffix)| suffix.to_string())
            })
    }

    fn generate_file_addition_id(checksum: &Sha256Hash, asset_uri: &str) -> HashId
    where
        Self: Sized,
    {
        let combined = format!("{checksum};{asset_uri}");
        HashId(calculate_hash(&combined))
    }

    fn asset_filename(&self, checksum: &Sha256Hash) -> String {
        let suffix = self.suffix().unwrap_or_default();
        let suffix = suffix.strip_prefix('.').unwrap_or(&suffix);
        if suffix.is_empty() {
            return checksum.to_string();
        }
        format!("{checksum}.{suffix}")
    }

    fn asset_relative_path(
        workspace: &Workspace,
        asset_uri: &Self,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError>
    where
        Self: Sized,
    {
        let repo_root = workspace.repo_root()?;
        let asset_path = workspace
            .asset_dir()?
            .join(asset_uri.asset_filename(checksum));

        Ok(asset_path
            .strip_prefix(&repo_root)
            .map_err(|_| FileAdditionError::PathOutsideRepo {
                path: asset_path.clone(),
                repo_root: repo_root.clone(),
            })?
            .to_string_lossy()
            .to_string())
    }
}

impl dyn AssetUri {
    pub fn new(workspace: &Workspace, uri: &str) -> Box<dyn AssetUri> {
        let (scheme, _) = uri.split_once("://").unwrap_or(("file", uri));
        match scheme.to_ascii_lowercase().as_str() {
            "file" => Box::new(
                LocalAssetUri::new_for_workspace(workspace, uri)
                    .expect("failed to construct local asset uri"),
            ),
            _ => Box::new(RemoteAssetUri::new(uri)),
        }
    }

    pub fn from_uri(uri: &str) -> Box<dyn AssetUri> {
        let (scheme, _) = uri.split_once("://").unwrap_or(("file", uri));
        match scheme.to_ascii_lowercase().as_str() {
            "file" => Box::new(LocalAssetUri::new(uri)),
            _ => Box::new(RemoteAssetUri::new(uri)),
        }
    }
}

pub fn materialization_destination_path(
    workspace: &Workspace,
    asset_uri: &str,
    checksum: Option<&Sha256Hash>,
    logical_path: Option<&str>,
) -> Result<PathBuf, FileAdditionError> {
    if let Some(logical_path) = logical_path.filter(|path| !path.is_empty()) {
        return LocalAssetUri::repo_relative_destination_path(workspace, logical_path);
    }

    let checksum = checksum.ok_or_else(|| {
        FileAdditionError::FileReadError(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("cannot materialize asset without checksum: {asset_uri}"),
        ))
    })?;
    Ok(workspace
        .asset_dir()?
        .join(<dyn AssetUri>::from_uri(asset_uri).hashed_filename(checksum)))
}

pub struct LocalAssetUri {
    asset_uri: String,
    source_path: Option<PathBuf>,
    workspace_root: Option<PathBuf>,
    read_file: Option<ChecksummedReader>,
    write_file: Option<WriteHandle>,
}

impl AssetUri for LocalAssetUri {
    fn uri(&self) -> &str {
        &self.asset_uri
    }

    fn reader(&self, workspace: &Workspace) -> Result<ChecksummedReader, FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        Location::from_workspace_path(workspace, &source_file_path)?.reader()
    }

    fn checksum(
        &self,
        workspace: &Workspace,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<Sha256Hash, FileAdditionError> {
        if let Some(checksum_override) = checksum_override {
            return Ok(checksum_override);
        }

        let source_file_path = self.resolved_source_file_path(workspace)?;
        let checksum_path = if source_file_path.is_file() {
            source_file_path
        } else {
            PathBuf::from(self.file_path())
        };
        Location::from_workspace_path(workspace, &checksum_path)?
            .checksum(&checksum_path.to_string_lossy())
    }

    fn stored_asset_uri(
        &self,
        workspace: &Workspace,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        if !source_file_path.starts_with(&repo_root) {
            return Ok(self.asset_uri.clone());
        }

        let source_asset_uri = Self::new(&source_file_path.to_string_lossy());
        let relative_file_path = Self::asset_relative_path(workspace, &source_asset_uri, checksum)?;
        Ok(Self::asset_uri(&relative_file_path))
    }

    fn ensure_asset(
        &self,
        workspace: &Workspace,
        checksum: &Sha256Hash,
    ) -> Result<(), FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        if source_file_path.is_file() {
            Self::ensure_asset_copy(workspace, &source_file_path, checksum)?;
        }
        Ok(())
    }

    fn store_file(
        &self,
        file_addition: &FileAddition,
        workspace: &Workspace,
    ) -> Result<(), FileStoreError> {
        Self::store_file(file_addition, workspace)
    }
}

impl LocalAssetUri {
    pub const OUTSIDE_ROOT_DIRECTORY: &'static str = ".gen/outside_root";
    pub const SCHEME: &'static str = "file://";

    pub fn new(path_or_uri: &str) -> Self {
        Self {
            asset_uri: Self::asset_uri(path_or_uri),
            source_path: None,
            workspace_root: None,
            read_file: None,
            write_file: None,
        }
    }

    pub fn new_for_workspace(
        workspace: &Workspace,
        path_or_uri: &str,
    ) -> Result<Self, FileAdditionError> {
        let source_path = if Self::is_file_uri(path_or_uri) {
            Self::resolve_source_path(workspace, path_or_uri)?
        } else {
            Self::resolve_input_source_path(workspace, path_or_uri)?
        };
        let asset_uri = Self::asset_uri(&Self::logical_file_path(workspace, &source_path)?);
        Ok(Self {
            asset_uri,
            source_path: Some(source_path),
            workspace_root: Some(workspace.repo_root()?),
            read_file: None,
            write_file: None,
        })
    }

    pub fn is_file_uri(asset_uri: &str) -> bool {
        asset_uri.starts_with(Self::SCHEME)
    }

    pub fn has_uri_scheme(path_or_uri: &str) -> bool {
        path_or_uri.contains("://")
    }

    pub fn is_local_path_or_file_uri(path_or_uri: &str) -> bool {
        Self::is_file_uri(path_or_uri) || !Self::has_uri_scheme(path_or_uri)
    }

    pub fn asset_uri(path: &str) -> String {
        let path = path.strip_prefix('/').unwrap_or(path);
        if Self::is_file_uri(path) {
            path.to_string()
        } else {
            format!("{}{path}", Self::SCHEME)
        }
    }

    pub fn path_from_uri(asset_uri: &str) -> Option<String> {
        asset_uri
            .strip_prefix(Self::SCHEME)
            .map(ToString::to_string)
    }

    pub fn operation_file_path(
        workspace: &Workspace,
        path_or_uri: &str,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        let source_path = if Self::is_file_uri(path_or_uri) {
            Self::resolve_source_path(workspace, path_or_uri)?
        } else {
            Self::resolve_input_source_path(workspace, path_or_uri)?
        };
        Self::stored_relative_path(workspace, &source_path, checksum)
    }

    /// Checks whether a relative path is in the .gen/assets directory of a workspace.
    pub fn is_asset_relative_path(
        workspace: &Workspace,
        relative_path: &str,
    ) -> Result<bool, FileAdditionError> {
        let repo_root = workspace.repo_root()?;
        let asset_dir = workspace.asset_dir()?;
        let asset_relative_dir =
            asset_dir
                .strip_prefix(&repo_root)
                .map_err(|_| FileAdditionError::PathOutsideRepo {
                    path: asset_dir.clone(),
                    repo_root: repo_root.clone(),
                })?;

        let candidate = if relative_path.is_empty() {
            PathBuf::new()
        } else {
            Self::sanitize_relative_path(Path::new(relative_path))?
        };

        Ok(candidate.starts_with(asset_relative_dir))
    }

    /// Given a workspace and a relative path, try to safely give a destination path to copy
    /// the asset into
    pub fn repo_relative_destination_path(
        workspace: &Workspace,
        relative_path: &str,
    ) -> Result<PathBuf, FileAdditionError> {
        let sanitized = if relative_path.is_empty() {
            PathBuf::new()
        } else {
            Self::sanitize_relative_path(Path::new(relative_path))?
        };

        Ok(workspace.repo_root()?.join(sanitized))
    }

    fn file_path(&self) -> &str {
        self.asset_uri
            .strip_prefix(Self::SCHEME)
            .unwrap_or(&self.asset_uri)
    }

    fn io_path(&self) -> PathBuf {
        self.source_path.clone().unwrap_or_else(|| {
            let file_path = PathBuf::from(self.file_path());
            if file_path.is_absolute() {
                file_path
            } else if let Some(workspace_root) = &self.workspace_root {
                workspace_root.join(file_path)
            } else {
                file_path
            }
        })
    }

    pub fn close_write(&mut self) -> io::Result<()> {
        if let Some(mut file) = self.write_file.take() {
            #[cfg(not(target_os = "emscripten"))]
            file.close()?;
            #[cfg(target_os = "emscripten")]
            file.flush()?;
        }
        Ok(())
    }

    fn resolved_source_file_path(
        &self,
        workspace: &Workspace,
    ) -> Result<PathBuf, FileAdditionError> {
        self.source_path
            .clone()
            .map(Ok)
            .unwrap_or_else(|| Self::resolve_source_path(workspace, &self.asset_uri))
    }

    fn resolve_input_source_path(
        workspace: &Workspace,
        file_path_or_uri: &str,
    ) -> Result<PathBuf, FileAdditionError> {
        if !Self::is_local_path_or_file_uri(file_path_or_uri) {
            return Err(FileAdditionError::FileReadError(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported non-local uri: {file_path_or_uri}"),
            )));
        }

        let file_path =
            Self::path_from_uri(file_path_or_uri).unwrap_or_else(|| file_path_or_uri.to_string());
        if file_path.is_empty() {
            return Ok(PathBuf::new());
        }

        let provided_path = Path::new(&file_path);
        if provided_path.is_absolute() {
            Ok(Self::canonicalize_or_normalize(provided_path))
        } else {
            Ok(Self::canonicalize_or_normalize(
                &workspace.repo_root()?.join(provided_path),
            ))
        }
    }

    fn resolve_source_path(
        workspace: &Workspace,
        asset_uri: &str,
    ) -> Result<PathBuf, FileAdditionError> {
        let file_path = Self::path_from_uri(asset_uri).unwrap_or_else(|| asset_uri.to_string());
        if file_path.is_empty() {
            return Ok(PathBuf::new());
        }

        let provided_path = Path::new(&file_path);
        if provided_path.is_absolute() {
            return Err(FileAdditionError::PathOutsideRepo {
                path: provided_path.to_path_buf(),
                repo_root: workspace.repo_root()?,
            });
        }

        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        let joined_path = workspace
            .repo_root()?
            .join(Self::sanitize_relative_path(provided_path)?);
        let normalized_path = Self::canonicalize_or_normalize(&joined_path);
        if !normalized_path.starts_with(&repo_root) {
            return Err(FileAdditionError::PathOutsideRepo {
                path: normalized_path,
                repo_root,
            });
        }
        Ok(normalized_path)
    }

    fn logical_file_path(
        workspace: &Workspace,
        source_path: &Path,
    ) -> Result<String, FileAdditionError> {
        if source_path.as_os_str().is_empty() {
            return Ok(String::new());
        }

        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        let source_path = Self::canonicalize_or_normalize(source_path);
        if let Ok(relative_path) = source_path.strip_prefix(&repo_root) {
            return Ok(relative_path.to_string_lossy().to_string());
        }

        source_path
            .file_name()
            .map(|file_name| {
                format!(
                    "{}/{}",
                    Self::OUTSIDE_ROOT_DIRECTORY,
                    file_name.to_string_lossy()
                )
            })
            .ok_or_else(|| {
                FileAdditionError::FileReadError(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("asset path has no filename: {}", source_path.display()),
                ))
            })
    }

    fn stored_relative_path(
        workspace: &Workspace,
        source_path: &Path,
        checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        if source_path.as_os_str().is_empty() {
            return Ok(String::new());
        }
        let repo_root = Self::canonicalize_or_normalize(&workspace.repo_root()?);
        let source_path = Self::canonicalize_or_normalize(source_path);

        if source_path.starts_with(&repo_root) {
            return Ok(source_path
                .strip_prefix(&repo_root)
                .map_err(|_| FileAdditionError::PathOutsideRepo {
                    path: source_path.to_path_buf(),
                    repo_root: repo_root.clone(),
                })?
                .to_string_lossy()
                .to_string());
        }

        if source_path.is_absolute() {
            let source_asset_uri = Self::new(&source_path.to_string_lossy());
            return Self::asset_relative_path(workspace, &source_asset_uri, checksum);
        }

        Err(FileAdditionError::FileReadError(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unable to resolve local asset path: {}",
                source_path.display()
            ),
        )))
    }

    fn sanitize_relative_path(path: &Path) -> Result<PathBuf, FileAdditionError> {
        let mut sanitized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Normal(part) => sanitized.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(FileAdditionError::FileReadError(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("path escapes workspace root: {}", path.display()),
                    )));
                }
            }
        }
        Ok(sanitized)
    }

    fn canonicalize_or_normalize(path: &Path) -> PathBuf {
        let normalized_path = fs::canonicalize(path).unwrap_or_else(|_| Self::normalize_path(path));
        // Unfortunately, the path we get back from Mac OS X sometimes starts
        // with /private even if the input path doesn't, which causes problems
        // for our path comparisons. Strip it out if it's there.
        let cleaned_path = normalized_path
            .as_path()
            .strip_prefix("/private")
            .unwrap_or(&normalized_path);
        Path::new("/").join(cleaned_path)
    }

    fn normalize_path(path: &Path) -> PathBuf {
        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
                Component::RootDir => normalized.push(Path::new("/")),
                Component::CurDir => {}
                Component::ParentDir => {
                    normalized.pop();
                }
                Component::Normal(part) => normalized.push(part),
            }
        }
        normalized
    }

    /// This exists along with store_file because store_file uses an existing FileAddition
    /// object to determine where to where to store the file, while this method uses the
    /// provided source_path.
    pub fn ensure_asset_copy(
        workspace: &Workspace,
        source_path: &Path,
        checksum: &Sha256Hash,
    ) -> Result<(), FileAdditionError> {
        let asset_uri = Self::new(&source_path.to_string_lossy());
        let asset_path = workspace
            .asset_dir()?
            .join(asset_uri.asset_filename(checksum));
        if asset_path.exists() {
            return Ok(());
        }

        Location::from_workspace_path(workspace, source_path)?
            .copy_to_local_path(workspace, &asset_path)
            .map_err(FileAdditionError::FileReadError)?;
        Ok(())
    }

    pub fn store_file(
        file_addition: &FileAddition,
        workspace: &Workspace,
    ) -> Result<(), FileStoreError> {
        let asset_path = workspace
            .asset_dir()?
            .join(file_addition.clone().hashed_filename());
        if asset_path.exists() {
            return Ok(());
        }

        let source_path = Self::resolve_source_path(workspace, &file_addition.asset_uri).map_err(
            |err| match err {
                FileAdditionError::FileReadError(err) => FileStoreError::IoError(err),
                other => FileStoreError::IoError(io::Error::other(other)),
            },
        )?;
        if source_path == asset_path {
            return Ok(());
        }
        Location::from_workspace_path(workspace, &source_path)
            .map_err(|err| match err {
                FileAdditionError::FileReadError(err) => FileStoreError::IoError(err),
                other => FileStoreError::IoError(io::Error::other(other)),
            })?
            .copy_to_local_path(workspace, &asset_path)
            .map_err(FileStoreError::IoError)?;
        Ok(())
    }
}

impl Read for LocalAssetUri {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.read_file.is_none() {
            let io_path = self.io_path();
            self.read_file = Some(
                Location::from_absolute_path(&io_path)
                    .map_err(io::Error::other)?
                    .reader()
                    .map_err(io::Error::other)?,
            );
        }
        self.read_file.as_mut().unwrap().read(buf)
    }
}

impl Write for LocalAssetUri {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.write_file.is_none() {
            let io_path = self.io_path();
            let location = Location::from_absolute_path(&io_path).map_err(io::Error::other)?;
            self.write_file = Some(location.writer_handle()?);
        }
        self.write_file.as_mut().unwrap().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = self.write_file.as_mut() {
            file.flush()
        } else {
            Ok(())
        }
    }
}

impl Drop for LocalAssetUri {
    fn drop(&mut self) {
        let _ = self.close_write();
    }
}

pub struct RemoteAssetUri {
    asset_uri: String,
}

impl AssetUri for RemoteAssetUri {
    fn uri(&self) -> &str {
        &self.asset_uri
    }

    #[cfg(not(target_os = "emscripten"))]
    fn reader(&self, _workspace: &Workspace) -> Result<ChecksummedReader, FileAdditionError> {
        OpenDalLocation::from_remote_uri(&self.asset_uri)?.reader()
    }

    #[cfg(target_os = "emscripten")]
    fn reader(&self, _workspace: &Workspace) -> Result<ChecksummedReader, FileAdditionError> {
        Err(FileAdditionError::FileReadError(io::Error::new(
            io::ErrorKind::Unsupported,
            "remote asset URIs are not supported in this environment",
        )))
    }

    fn checksum(
        &self,
        _workspace: &Workspace,
        checksum_override: Option<Sha256Hash>,
    ) -> Result<Sha256Hash, FileAdditionError> {
        Ok(Self::checksum_for_uri(&self.asset_uri, checksum_override))
    }

    fn stored_asset_uri(
        &self,
        _workspace: &Workspace,
        _checksum: &Sha256Hash,
    ) -> Result<String, FileAdditionError> {
        Ok(self.asset_uri.clone())
    }

    fn ensure_asset(
        &self,
        _workspace: &Workspace,
        _checksum: &Sha256Hash,
    ) -> Result<(), FileAdditionError> {
        Ok(())
    }

    fn store_file(
        &self,
        _file_addition: &FileAddition,
        _workspace: &Workspace,
    ) -> Result<(), FileStoreError> {
        Ok(())
    }
}

impl RemoteAssetUri {
    pub fn new(asset_uri: &str) -> Self {
        Self {
            asset_uri: asset_uri.to_string(),
        }
    }

    fn checksum_for_uri(asset_uri: &str, checksum_override: Option<Sha256Hash>) -> Sha256Hash {
        checksum_override.unwrap_or_else(|| Sha256Hash::convert_str(asset_uri))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Read, Write},
        net::TcpListener,
        path::PathBuf,
        thread,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::{
        history::dolt::commit_all, operations::calculate_file_checksum, test_helpers::setup_gen,
        traits::Query,
    };

    #[test]
    fn test_asset_reference_tables_round_trip() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let asset_ref = AssetRef {
            id: HashId::convert_str("asset-ref"),
            uri: "s3://bucket/reference.fa".to_string(),
            file_type: "fasta".to_string(),
            checksum: Some(Sha256Hash::convert_str("checksum")),
            size: Some(1024),
            role: AssetRole::Input,
            logical_path: Some("refs/reference.fa".to_string()),
            name: Some("reference.fa".to_string()),
            created_on: 1,
        };
        let operation_log = OperationLog {
            id: HashId::convert_str("log"),
            operation_kind: OperationKind::Other("import".to_string()),
            command: "gen import fasta".to_string(),
            created_on: 1,
        };
        let operation_asset = OperationAsset {
            log_id: operation_log.id,
            asset_ref_id: asset_ref.id,
            role: AssetRole::Input,
        };

        OperationLog::create(conn, &operation_log).expect("should insert operation log");
        AssetRef::create(conn, &asset_ref).expect("should insert asset ref");
        OperationAsset::create(conn, &operation_asset).expect("should insert operation asset");

        let asset_refs = AssetRef::all(conn);
        let operation_logs = OperationLog::all(conn);
        let operation_assets = OperationAsset::all(conn);

        assert_eq!(asset_refs, vec![asset_ref]);
        assert_eq!(operation_logs, vec![operation_log]);
        assert_eq!(operation_assets, vec![operation_asset]);
    }

    #[test]
    fn test_get_annotation_files_pairs_annotation_and_index_in_one_query() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let log = OperationLog {
            id: HashId::convert_str("annotation-log"),
            operation_kind: OperationKind::AnnotationFile,
            command: "add annotation".to_string(),
            created_on: 1,
        };
        let annotation = AssetRef {
            id: HashId::convert_str("annotation-asset"),
            uri: "file://annotation.gff".to_string(),
            file_type: "gff3".to_string(),
            checksum: Some(Sha256Hash::convert_str("annotation-checksum")),
            size: None,
            role: AssetRole::Annotation,
            logical_path: Some("annotation.gff".to_string()),
            name: Some("genes".to_string()),
            created_on: 1,
        };
        let index = AssetRef {
            id: HashId::convert_str("annotation-index-asset"),
            uri: "file://annotation.gff.tbi".to_string(),
            file_type: "tabix".to_string(),
            checksum: Some(Sha256Hash::convert_str("annotation-index-checksum")),
            size: None,
            role: AssetRole::AnnotationIndex,
            logical_path: Some("annotation.gff.tbi".to_string()),
            name: Some("genes".to_string()),
            created_on: 1,
        };
        OperationLog::create(conn, &log).expect("should insert annotation log");
        AssetRef::create(conn, &annotation).expect("should insert annotation asset");
        AssetRef::create(conn, &index).expect("should insert annotation index asset");
        OperationAsset::create(
            conn,
            &OperationAsset {
                log_id: log.id,
                asset_ref_id: annotation.id,
                role: AssetRole::Annotation,
            },
        )
        .expect("should associate annotation asset");
        OperationAsset::create(
            conn,
            &OperationAsset {
                log_id: log.id,
                asset_ref_id: index.id,
                role: AssetRole::AnnotationIndex,
            },
        )
        .expect("should associate annotation index asset");
        let commit = commit_all(conn, "add annotation assets").expect("should commit assets");

        let expected = vec![AnnotationFileAssets {
            log_id: log.id,
            annotation,
            index: Some(index),
        }];
        assert_eq!(
            Assets::get_annotation_files(conn, None).expect("should query current annotations"),
            expected
        );
        assert_eq!(
            Assets::get_annotation_files(conn, Some(&commit.to_string()))
                .expect("should query historical annotations"),
            expected
        );
    }

    #[test]
    fn test_get_branch_assets_returns_local_assets_from_the_requested_commit() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let local_asset = AssetRef {
            id: HashId::convert_str("local-asset"),
            uri: "file://inputs/reference.fa".to_string(),
            file_type: "fasta".to_string(),
            checksum: Some(Sha256Hash::convert_str("local-checksum")),
            size: Some(4),
            role: AssetRole::Input,
            logical_path: Some("inputs/reference.fa".to_string()),
            name: Some("reference.fa".to_string()),
            created_on: 1,
        };
        let remote_asset = AssetRef {
            id: HashId::convert_str("remote-asset"),
            uri: "https://example.com/reference.fa".to_string(),
            ..local_asset.clone()
        };
        AssetRef::create(conn, &local_asset).expect("should insert local asset");
        AssetRef::create(conn, &remote_asset).expect("should insert remote asset");
        let commit = commit_all(conn, "add assets").expect("should commit assets");

        let assets = Assets::get_branch_assets(conn, &commit.to_string())
            .expect("should read assets at commit");

        assert_eq!(assets, HashMap::from([(local_asset.id, local_asset)]));
    }

    #[test]
    fn test_generate_file_addition_id_consistency() {
        let checksum = Sha256Hash([1u8; 32]);
        let file_path = "/path/to/file.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(&checksum, file_path);
        let id2 = LocalAssetUri::generate_file_addition_id(&checksum, file_path);

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_paths() {
        let checksum = Sha256Hash([1u8; 32]);
        let file_path1 = "/path/to/file1.txt";
        let file_path2 = "/path/to/file2.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(&checksum, file_path1);
        let id2 = LocalAssetUri::generate_file_addition_id(&checksum, file_path2);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_checksums() {
        let checksum1 = Sha256Hash([1u8; 32]);
        let checksum2 = Sha256Hash([2u8; 32]);
        let file_path = "/path/to/file.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(&checksum1, file_path);
        let id2 = LocalAssetUri::generate_file_addition_id(&checksum2, file_path);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_resolve_source_path_rejects_absolute_path_from_file_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.base_dir();

        let absolute_path = repo_root.join("inputs").join("absolute.txt");
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"absolute").unwrap();
        let absolute_string = absolute_path.to_string_lossy().to_string();

        let err =
            LocalAssetUri::resolve_source_path(workspace, absolute_string.as_str()).unwrap_err();
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_stored_relative_path_returns_relative_path_from_file_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.base_dir();

        let absolute_path = repo_root.join("inputs").join("absolute.txt");
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"absolute").unwrap();
        let relative_string = absolute_path
            .strip_prefix(repo_root)
            .unwrap()
            .to_string_lossy()
            .to_string();

        let relative = LocalAssetUri::stored_relative_path(
            workspace,
            &absolute_path,
            &calculate_file_checksum(&absolute_path).unwrap(),
        )
        .unwrap();

        assert_eq!(relative, relative_string);
    }

    #[test]
    fn test_resolve_source_path_returns_absolute_path_from_relative_path_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();

        let relative_path = PathBuf::from("relative/path/file.txt");
        let absolute_path = repo_root.join(&relative_path);
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"relative").unwrap();
        let relative_string = relative_path.to_string_lossy().to_string();
        let absolute_string = absolute_path.to_string_lossy().to_string();

        let absolute =
            LocalAssetUri::resolve_source_path(workspace, relative_string.as_str()).unwrap();

        assert_eq!(absolute, PathBuf::from(absolute_string));
    }

    #[test]
    fn test_stored_relative_path_returns_relative_path_from_relative_path_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();

        let relative_path = PathBuf::from("relative/path/file.txt");
        let absolute_path = repo_root.join(&relative_path);
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"relative").unwrap();
        let relative_string = relative_path.to_string_lossy().to_string();

        let relative = LocalAssetUri::stored_relative_path(
            workspace,
            &absolute_path,
            &calculate_file_checksum(&absolute_path).unwrap(),
        )
        .unwrap();

        assert_eq!(relative, relative_string);
    }

    #[test]
    fn test_resolve_source_path_rejects_absolute_path_outside_repo() {
        let context = setup_gen();
        let workspace = context.workspace();

        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();
        let outside_string = outside_file.path().to_string_lossy().to_string();

        let err =
            LocalAssetUri::resolve_source_path(workspace, outside_string.as_str()).unwrap_err();
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_stored_relative_path_returns_asset_path_on_file_outside_repo() {
        let context = setup_gen();
        let workspace = context.workspace();

        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();
        let checksum = calculate_file_checksum(outside_file.path()).unwrap();
        let outside_suffix = LocalAssetUri::new(&outside_file.path().to_string_lossy())
            .suffix()
            .unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, outside_file.path(), &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.{outside_suffix}"));
    }

    #[test]
    fn test_new_for_workspace_namespaces_path_outside_workspace() {
        let context = setup_gen();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("simple.fa");
        fs::write(&outside_path, b">seq\nACGT\n").unwrap();

        let asset_uri =
            LocalAssetUri::new_for_workspace(context.workspace(), &outside_path.to_string_lossy())
                .unwrap();

        assert_eq!(asset_uri.uri(), "file://.gen/outside_root/simple.fa");
        assert_eq!(asset_uri.suffix().as_deref(), Some("fa"));
    }

    #[test]
    fn test_new_for_workspace_distinguishes_external_file_from_workspace_root_file() {
        let context = setup_gen();
        let workspace_path = context.workspace().repo_root().unwrap().join("simple.fa");
        fs::write(&workspace_path, b"workspace file").unwrap();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("simple.fa");
        fs::write(&outside_path, b"external file").unwrap();

        let workspace_asset = LocalAssetUri::new_for_workspace(
            context.workspace(),
            &workspace_path.to_string_lossy(),
        )
        .unwrap();
        let outside_asset =
            LocalAssetUri::new_for_workspace(context.workspace(), &outside_path.to_string_lossy())
                .unwrap();

        assert_eq!(workspace_asset.uri(), "file://simple.fa");
        assert_eq!(outside_asset.uri(), "file://.gen/outside_root/simple.fa");
        assert_ne!(workspace_asset.uri(), outside_asset.uri());
    }

    #[test]
    fn stored_relative_path_keeps_compression_suffix_for_asset_path() {
        let context = setup_gen();
        let workspace = context.workspace();
        let fixture_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/simple.fa.bgz");
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("simple.fa.bgz");
        fs::copy(&fixture_path, &outside_path).unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.fa.bgz"));
    }

    #[test]
    fn stored_relative_path_keeps_unknown_file_type_suffix_for_asset_path() {
        let context = setup_gen();
        let workspace = context.workspace();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("sample.custom.gz");
        fs::write(&outside_path, b"custom contents").unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.custom.gz"));
    }

    #[test]
    fn stored_relative_path_keeps_full_source_suffix_for_asset_path() {
        let context = setup_gen();
        let workspace = context.workspace();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("sample.fa.unhandled");
        fs::write(&outside_path, b"custom contents").unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}.fa.unhandled"));
    }

    #[test]
    fn stored_relative_path_omits_suffix_for_source_without_suffix() {
        let context = setup_gen();
        let workspace = context.workspace();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside_path = outside_dir.path().join("sample");
        fs::write(&outside_path, b"custom contents").unwrap();
        let checksum = calculate_file_checksum(&outside_path).unwrap();

        let relative =
            LocalAssetUri::stored_relative_path(workspace, &outside_path, &checksum).unwrap();

        assert_eq!(relative, format!(".gen/assets/{checksum}"));
    }

    #[test]
    fn hashed_filename_uses_http_uri_path_suffix() {
        let checksum = Sha256Hash::convert_str("remote");
        let asset_uri = RemoteAssetUri::new("http://example.com/assets/fasta.fa.gz?download=1");

        let filename = asset_uri.hashed_filename(&checksum);

        assert_eq!(filename, format!("{checksum}.fa.gz"));
    }

    #[test]
    fn test_resolve_source_path_anchors_relative_paths_to_workspace() {
        let context = setup_gen();
        let workspace = context.workspace();
        let expected = workspace.repo_root().unwrap().join("detached/file.txt");

        let absolute = LocalAssetUri::resolve_source_path(workspace, "detached/file.txt").unwrap();
        assert_eq!(absolute, expected);
    }

    #[test]
    fn test_resolve_source_path_rejects_parent_traversal() {
        let context = setup_gen();
        let workspace = context.workspace();

        let err = LocalAssetUri::resolve_source_path(workspace, "../detached/file.txt")
            .expect_err("expected traversal to be rejected");
        assert!(matches!(err, FileAdditionError::FileReadError(_)));
    }

    mod test_sanitize_relative_path {
        use super::*;

        #[test]
        fn keeps_normal_relative_path() {
            let sanitized = LocalAssetUri::sanitize_relative_path(Path::new("nested/file.txt"))
                .expect("expected relative path to be allowed");
            assert_eq!(sanitized, PathBuf::from("nested/file.txt"));
        }

        #[test]
        fn rejects_parent_dir() {
            let err = LocalAssetUri::sanitize_relative_path(Path::new("../nested/file.txt"))
                .expect_err("expected parent traversal to be rejected");
            assert!(matches!(err, FileAdditionError::FileReadError(_)));
        }
    }

    mod test_normalize_path {
        use super::*;

        #[test]
        fn collapses_dot_and_parent_components() {
            let normalized =
                LocalAssetUri::normalize_path(Path::new("nested/./deeper/../file.txt"));
            assert_eq!(normalized, PathBuf::from("nested/file.txt"));
        }
    }

    mod test_canonicalize_or_normalize {
        use super::*;

        #[test]
        fn normalizes_missing_path() {
            let normalized = LocalAssetUri::canonicalize_or_normalize(Path::new(
                "/tmp/gen-models-assets-tests/one/../two/file.txt",
            ));
            assert_eq!(
                normalized,
                PathBuf::from("/tmp/gen-models-assets-tests/two/file.txt")
            );
        }

        #[test]
        fn canonicalizes_existing_path() {
            let temp_dir = tempfile::tempdir().unwrap();
            let target_dir = temp_dir.path().join("target");
            fs::create_dir_all(&target_dir).unwrap();
            let file_path = target_dir.join("file.txt");
            fs::write(&file_path, b"data").unwrap();

            let canonicalized = LocalAssetUri::canonicalize_or_normalize(
                &temp_dir.path().join("target/../target/file.txt"),
            );

            // Unfortunately, the path we get back from Mac OS X sometimes starts
            // with /private even if the input path doesn't, which causes problems
            // for our path comparisons. Strip it out if it's there.
            let expected_canonical = fs::canonicalize(&file_path).unwrap();
            let cleaned_path = expected_canonical
                .strip_prefix("/private")
                .unwrap_or(&expected_canonical);
            let expected_path = Path::new("/").join(cleaned_path);
            assert_eq!(canonicalized, expected_path);
        }
    }

    #[test]
    fn test_new_for_workspace_rejects_parent_traversal_file_uri() {
        let context = setup_gen();
        let workspace = context.workspace();

        let err = LocalAssetUri::new_for_workspace(workspace, "file://../detached/file.txt")
            .err()
            .expect("expected traversal file uri to be rejected");
        assert!(matches!(err, FileAdditionError::FileReadError(_)));
    }

    #[test]
    fn test_resolve_source_path_rejects_absolute_asset_uri() {
        let context = setup_gen();
        let workspace = context.workspace();

        let absolute_path = workspace.repo_root().unwrap().join("detached/file.txt");
        let err = LocalAssetUri::resolve_source_path(
            workspace,
            &format!("file://{}", absolute_path.display()),
        )
        .expect_err("expected absolute asset uri to be rejected");
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_new_for_workspace_rejects_non_local_uri() {
        let context = setup_gen();
        let workspace = context.workspace();

        let err = LocalAssetUri::new_for_workspace(workspace, "https://example.com/reference.fa")
            .err()
            .expect("expected non-local uri to be rejected");
        assert!(matches!(err, FileAdditionError::FileReadError(_)));
        assert!(
            err.to_string()
                .contains("unsupported non-local uri: https://example.com/reference.fa")
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_resolve_source_path_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();
        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();

        let symlink_path = repo_root.join("escape-link.txt");
        if symlink_path.exists() || symlink_path.symlink_metadata().is_ok() {
            fs::remove_file(&symlink_path).unwrap();
        }
        symlink(outside_file.path(), &symlink_path).unwrap();

        let err = LocalAssetUri::resolve_source_path(workspace, "escape-link.txt")
            .expect_err("expected symlink escape to be rejected");
        assert!(matches!(err, FileAdditionError::PathOutsideRepo { .. }));
    }

    #[test]
    fn test_stored_relative_path_keeps_workspace_relative_paths() {
        let context = setup_gen();
        let workspace = context.workspace();

        let relative = LocalAssetUri::stored_relative_path(
            workspace,
            &workspace.repo_root().unwrap().join("detached/file.txt"),
            &Sha256Hash::convert_str("detached"),
        )
        .unwrap();
        assert_eq!(relative, "detached/file.txt");
    }

    #[test]
    fn test_resolve_source_path_empty() {
        let context = setup_gen();
        let workspace = context.workspace();

        let absolute_empty = LocalAssetUri::resolve_source_path(workspace, "").unwrap();
        assert_eq!(absolute_empty, PathBuf::new());
    }

    #[test]
    fn test_stored_relative_path_empty() {
        let context = setup_gen();
        let workspace = context.workspace();

        let relative_empty = LocalAssetUri::stored_relative_path(
            workspace,
            Path::new(""),
            &Sha256Hash::convert_str("empty"),
        )
        .unwrap();
        assert_eq!(relative_empty, "");
    }

    #[test]
    fn test_local_asset_uri_read_write() {
        let context = setup_gen();
        let repo_root = context.workspace().repo_root().unwrap();
        let path = repo_root.join("asset-uri-io.txt");
        fs::write(&path, "initial").unwrap();

        let mut reader =
            LocalAssetUri::new_for_workspace(context.workspace(), "asset-uri-io.txt").unwrap();
        let mut contents = String::new();
        reader.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "initial");

        let mut writer =
            LocalAssetUri::new_for_workspace(context.workspace(), "asset-uri-io.txt").unwrap();
        writer.write_all(b"updated").unwrap();
        writer.close_write().unwrap();

        assert_eq!(fs::read_to_string(path).unwrap(), "updated");
    }

    #[test]
    fn test_remote_asset_uri_reads_http_uri() {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u16;
        let base_port = 20_000 + (seed % 20_000);
        let listener = (0..32)
            .find_map(|offset| {
                let port = base_port.saturating_add(offset);
                TcpListener::bind(("127.0.0.1", port)).ok()
            })
            .expect("failed to bind test HTTP listener");
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let started = Instant::now();
            let mut served_get = false;
            while !served_get && started.elapsed() < Duration::from_secs(5) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                let mut request = [0; 1024];
                let len = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..len]);
                served_get = request.starts_with("GET ");
                if served_get {
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\ninitial",
                        )
                        .unwrap();
                } else {
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\n",
                        )
                        .unwrap();
                }
            }
            assert!(served_get);
        });

        let context = setup_gen();
        let uri = format!("http://{addr}/asset.fa");
        let mut reader = RemoteAssetUri::new(&uri)
            .reader(context.workspace())
            .unwrap();
        let mut contents = String::new();
        reader.read_to_string(&mut contents).unwrap();

        handle.join().unwrap();
        assert_eq!(contents, "initial");
    }
}
