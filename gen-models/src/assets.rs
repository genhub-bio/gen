use std::{
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    string::ToString,
    sync::{Arc, LazyLock, Mutex},
};

use gen_core::{HashId, Workspace, calculate_hash};
use opendal::{blocking, services};
use sha2::{Digest, Sha256};
use url::{Position, Url};

use crate::{
    errors::{FileAdditionError, FileStoreError},
    file_types::FileTypes,
    operations::{FileAddition, calculate_reader_checksum},
};

static OPENDAL_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build OpenDAL runtime")
});

fn with_opendal_runtime<T>(f: impl FnOnce() -> T) -> T {
    let _guard = OPENDAL_RUNTIME.enter();
    f()
}

fn opendal_file_addition_error(err: opendal::Error) -> FileAdditionError {
    FileAdditionError::FileReadError(io::Error::other(err))
}

fn opendal_file_store_error(err: opendal::Error) -> FileStoreError {
    FileStoreError::IoError(io::Error::other(err))
}

#[doc(hidden)]
pub struct OpenDalLocation {
    operator: blocking::Operator,
    path: String,
}

struct ChecksumState {
    hasher: Sha256,
    checksum: Option<HashId>,
    complete: bool,
}

#[derive(Clone)]
pub struct ChecksumHandle {
    state: Arc<Mutex<ChecksumState>>,
}

impl ChecksumHandle {
    pub fn checksum(&self) -> Option<HashId> {
        let state = self.state.lock().unwrap();
        if state.complete { state.checksum } else { None }
    }

    pub fn finalized_checksum(&self) -> Option<HashId> {
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
                state.checksum = Some(HashId(finalized.into()));
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
            state.checksum = Some(HashId(finalized.into()));
        }
    }
}

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

    fn from_workspace_path(workspace: &Workspace, path: &Path) -> Result<Self, opendal::Error> {
        let repo_root = workspace.repo_root().map_err(|err| {
            opendal::Error::new(opendal::ErrorKind::Unexpected, "repo_root failed").set_source(err)
        })?;

        if path.is_absolute() {
            Self::new_fs(Path::new("/"), path)
        } else {
            Self::new_fs(&repo_root, path)
        }
    }

    fn from_absolute_path(path: &Path) -> Result<Self, opendal::Error> {
        if !path.is_absolute() {
            return Err(opendal::Error::new(
                opendal::ErrorKind::Unexpected,
                "local path must be absolute",
            ));
        }

        Self::new_fs(Path::new("/"), path)
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

    fn checksum(&self, display_path: &str) -> Result<HashId, FileAdditionError> {
        let reader = match self.operator.reader(&self.path) {
            Ok(reader) => reader,
            Err(err) => {
                return match err.kind() {
                    opendal::ErrorKind::NotFound => Ok(HashId::convert_str("non-existent")),
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

pub trait AssetUri {
    fn uri(&self) -> &str;

    fn reader(&self, workspace: &Workspace) -> Result<ChecksummedReader, FileAdditionError>;

    fn checksum(
        &self,
        workspace: &Workspace,
        checksum_override: Option<HashId>,
    ) -> Result<HashId, FileAdditionError>;

    fn stored_asset_uri(
        &self,
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<String, FileAdditionError>;

    fn ensure_asset(
        &self,
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<(), FileAdditionError>;

    fn store_file(
        &self,
        file_addition: &FileAddition,
        workspace: &Workspace,
    ) -> Result<(), FileStoreError>;

    fn generate_file_addition_id(checksum: &HashId, asset_uri: &str) -> HashId
    where
        Self: Sized,
    {
        let combined = format!("{checksum};{asset_uri}");
        HashId(calculate_hash(&combined))
    }

    fn asset_filename(checksum: &HashId, file_type: FileTypes) -> String
    where
        Self: Sized,
    {
        format!("{checksum}.{}", FileTypes::suffix(file_type))
    }

    fn asset_path(
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<PathBuf, FileAdditionError>
    where
        Self: Sized,
    {
        Ok(workspace
            .asset_dir()?
            .join(Self::asset_filename(checksum, file_type)))
    }

    fn asset_relative_path(
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<String, FileAdditionError>
    where
        Self: Sized,
    {
        let repo_root = workspace.repo_root()?;
        let asset_path = Self::asset_path(workspace, checksum, file_type)?;

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
}

pub struct LocalAssetUri {
    asset_uri: String,
    source_path: Option<PathBuf>,
    workspace_root: Option<PathBuf>,
    read_file: Option<ChecksummedReader>,
    write_file: Option<opendal::blocking::StdWriter>,
}

impl AssetUri for LocalAssetUri {
    fn uri(&self) -> &str {
        &self.asset_uri
    }

    fn reader(&self, workspace: &Workspace) -> Result<ChecksummedReader, FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        OpenDalLocation::from_workspace_path(workspace, &source_file_path)
            .map_err(opendal_file_addition_error)?
            .reader()
    }

    fn checksum(
        &self,
        workspace: &Workspace,
        checksum_override: Option<HashId>,
    ) -> Result<HashId, FileAdditionError> {
        if let Some(checksum_override) = checksum_override {
            return Ok(checksum_override);
        }

        let source_file_path = self.resolved_source_file_path(workspace)?;
        let checksum_path = if source_file_path.is_file() {
            source_file_path
        } else {
            PathBuf::from(self.file_path())
        };
        OpenDalLocation::from_workspace_path(workspace, &checksum_path)
            .map_err(opendal_file_addition_error)?
            .checksum(&checksum_path.to_string_lossy())
    }

    fn stored_asset_uri(
        &self,
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<String, FileAdditionError> {
        let relative_file_path = Self::stored_relative_path(
            workspace,
            &self.resolved_source_file_path(workspace)?,
            checksum,
            file_type,
        )?;
        Ok(Self::asset_uri(&relative_file_path))
    }

    fn ensure_asset(
        &self,
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<(), FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        if source_file_path.is_file() {
            Self::ensure_asset_copy(workspace, &source_file_path, checksum, file_type)?;
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
            file.close()?;
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
        path_or_uri: &str,
    ) -> Result<PathBuf, FileAdditionError> {
        let file_path = Self::path_from_uri(path_or_uri).unwrap_or_else(|| path_or_uri.to_string());
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

        Ok(source_path
            .strip_prefix(Path::new("/"))
            .unwrap_or(&source_path)
            .to_string_lossy()
            .to_string())
    }

    fn stored_relative_path(
        workspace: &Workspace,
        source_path: &Path,
        checksum: &HashId,
        file_type: FileTypes,
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
            return Self::asset_relative_path(workspace, checksum, file_type);
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
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<(), FileAdditionError> {
        let asset_path = Self::asset_path(workspace, checksum, file_type)?;
        if asset_path.exists() {
            return Ok(());
        }

        OpenDalLocation::from_workspace_path(workspace, source_path)
            .map_err(opendal_file_addition_error)?
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
        OpenDalLocation::from_workspace_path(workspace, &source_path)
            .map_err(opendal_file_store_error)?
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
                OpenDalLocation::from_absolute_path(&io_path)
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
            let location =
                OpenDalLocation::from_absolute_path(&io_path).map_err(io::Error::other)?;
            self.write_file = Some(
                location
                    .operator
                    .writer(&location.path)
                    .map_err(io::Error::other)?
                    .into_std_write(),
            );
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

    fn reader(&self, _workspace: &Workspace) -> Result<ChecksummedReader, FileAdditionError> {
        OpenDalLocation::from_remote_uri(&self.asset_uri)?.reader()
    }

    fn checksum(
        &self,
        _workspace: &Workspace,
        checksum_override: Option<HashId>,
    ) -> Result<HashId, FileAdditionError> {
        Ok(Self::checksum_for_uri(&self.asset_uri, checksum_override))
    }

    fn stored_asset_uri(
        &self,
        _workspace: &Workspace,
        _checksum: &HashId,
        _file_type: FileTypes,
    ) -> Result<String, FileAdditionError> {
        Ok(self.asset_uri.clone())
    }

    fn ensure_asset(
        &self,
        _workspace: &Workspace,
        _checksum: &HashId,
        _file_type: FileTypes,
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

    fn checksum_for_uri(asset_uri: &str, checksum_override: Option<HashId>) -> HashId {
        checksum_override.unwrap_or_else(|| HashId(calculate_hash(asset_uri)))
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
    use crate::{operations::calculate_file_checksum, test_helpers::setup_gen};

    #[test]
    fn test_generate_file_addition_id_consistency() {
        let checksum = HashId([1u8; 32]);
        let file_path = "/path/to/file.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(&checksum, file_path);
        let id2 = LocalAssetUri::generate_file_addition_id(&checksum, file_path);

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_paths() {
        let checksum = HashId([1u8; 32]);
        let file_path1 = "/path/to/file1.txt";
        let file_path2 = "/path/to/file2.txt";

        let id1 = LocalAssetUri::generate_file_addition_id(&checksum, file_path1);
        let id2 = LocalAssetUri::generate_file_addition_id(&checksum, file_path2);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_generate_file_addition_id_uniqueness_different_checksums() {
        let checksum1 = HashId([1u8; 32]);
        let checksum2 = HashId([2u8; 32]);
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
            FileTypes::Fasta,
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
            FileTypes::Fasta,
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

        let relative = LocalAssetUri::stored_relative_path(
            workspace,
            outside_file.path(),
            &checksum,
            FileTypes::Fasta,
        )
        .unwrap();

        assert_eq!(
            relative,
            format!(
                ".gen/assets/{checksum}.{}",
                FileTypes::suffix(FileTypes::Fasta)
            )
        );
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
            &HashId::convert_str("detached"),
            FileTypes::Fasta,
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
            &HashId::convert_str("empty"),
            FileTypes::Fasta,
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
