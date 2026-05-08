use std::{
    io::{self, Read, Write},
    path::{Path, PathBuf},
    string::ToString,
    sync::LazyLock,
};

use gen_core::{HashId, Workspace, calculate_hash};
use opendal::{blocking, services};
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

impl OpenDalLocation {
    fn from_local_path(path: &Path) -> Result<Self, opendal::Error> {
        let absolute_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|err| {
                    opendal::Error::new(opendal::ErrorKind::Unexpected, "current_dir failed")
                        .set_source(err)
                })?
                .join(path)
        };
        let op_path = absolute_path
            .strip_prefix(Path::new("/"))
            .unwrap_or(&absolute_path)
            .to_string_lossy()
            .to_string();
        let operator = with_opendal_runtime(|| {
            let builder = services::Fs::default().root("/");
            let op = opendal::Operator::new(builder)?.finish();
            blocking::Operator::new(op)
        })?;

        Ok(Self {
            operator,
            path: op_path,
        })
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

    fn reader(self) -> Result<opendal::blocking::StdReader, FileAdditionError> {
        self.operator
            .reader(&self.path)
            .map_err(opendal_file_addition_error)?
            .into_std_read(..)
            .map_err(opendal_file_addition_error)
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

    fn copy_to_local_path(&self, destination_path: &Path) -> io::Result<u64> {
        let asset_location = Self::from_local_path(destination_path).map_err(io::Error::other)?;
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

    fn reader(&self, workspace: &Workspace) -> Result<Box<dyn Read>, FileAdditionError>;

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
    pub fn new(uri: &str) -> Box<dyn AssetUri> {
        let (scheme, _) = uri.split_once("://").unwrap_or(("file", uri));
        match scheme.to_ascii_lowercase().as_str() {
            "file" => Box::new(LocalAssetUri::new(uri)),
            _ => Box::new(RemoteAssetUri::new(uri)),
        }
    }
}

pub struct LocalAssetUri {
    asset_uri: String,
    source_path: Option<PathBuf>,
    read_file: Option<opendal::blocking::StdReader>,
    write_file: Option<opendal::blocking::StdWriter>,
}

impl AssetUri for LocalAssetUri {
    fn uri(&self) -> &str {
        &self.asset_uri
    }

    fn reader(&self, workspace: &Workspace) -> Result<Box<dyn Read>, FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        Ok(Box::new(
            OpenDalLocation::from_local_path(Path::new(&source_file_path))
                .map_err(opendal_file_addition_error)?
                .reader()?,
        ))
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
        let checksum_path = if Path::new(&source_file_path).is_file() {
            source_file_path.as_str()
        } else {
            self.file_path()
        };
        OpenDalLocation::from_local_path(Path::new(checksum_path))
            .map_err(opendal_file_addition_error)?
            .checksum(checksum_path)
    }

    fn stored_asset_uri(
        &self,
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<String, FileAdditionError> {
        let relative_file_path =
            Self::stored_file_path(workspace, &self.asset_uri, checksum, file_type)?;
        Ok(Self::asset_uri(&relative_file_path))
    }

    fn ensure_asset(
        &self,
        workspace: &Workspace,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<(), FileAdditionError> {
        let source_file_path = self.resolved_source_file_path(workspace)?;
        if Path::new(&source_file_path).is_file() {
            Self::ensure_asset_copy(workspace, Path::new(&source_file_path), checksum, file_type)?;
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
            read_file: None,
            write_file: None,
        }
    }

    pub fn new_for_workspace(
        workspace: &Workspace,
        path_or_uri: &str,
    ) -> Result<Self, FileAdditionError> {
        let asset_uri = Self::asset_uri(path_or_uri);
        let source_path = Self::source_file_path(workspace, &asset_uri)?;
        Ok(Self {
            asset_uri,
            source_path: Some(PathBuf::from(source_path)),
            read_file: None,
            write_file: None,
        })
    }

    pub fn is_file_uri(asset_uri: &str) -> bool {
        asset_uri.starts_with(Self::SCHEME)
    }

    pub fn asset_uri(path: &str) -> String {
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
        self.source_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(self.file_path()))
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
    ) -> Result<String, FileAdditionError> {
        Self::source_file_path(workspace, &self.asset_uri)
    }

    /// Returns a usable path for system file access. It prefers returning an absolute path
    /// when one can be resolved, and otherwise falls back to the original input.
    pub fn source_file_path(
        workspace: &Workspace,
        asset_uri: &str,
    ) -> Result<String, FileAdditionError> {
        let file_path = Self::path_from_uri(asset_uri).unwrap_or_else(|| asset_uri.to_string());
        if file_path.is_empty() {
            return Ok(String::new());
        }
        let repo_root = workspace.repo_root()?;

        let provided_path = Path::new(&file_path);

        if provided_path.is_absolute() {
            if provided_path.starts_with(&repo_root) {
                return Ok(provided_path.to_string_lossy().to_string());
            }
        } else {
            let absolute = repo_root.join(provided_path);
            if absolute.exists() {
                return Ok(absolute.to_string_lossy().to_string());
            }
        }

        Ok(file_path)
    }

    /// Returns the asset URI as it should be stored. If a file is within the gen
    /// directory, it stores a file URI for the path relative to the repo root.
    /// Otherwise, if the path is absolute, it stores a file URI for the copied
    /// asset. If it is not absolute and not within the repo, it is stored as-is
    /// with a file scheme.
    pub fn stored_file_path(
        workspace: &Workspace,
        asset_uri: &str,
        checksum: &HashId,
        file_type: FileTypes,
    ) -> Result<String, FileAdditionError> {
        let file_path = Self::path_from_uri(asset_uri).unwrap_or_else(|| asset_uri.to_string());
        if file_path.is_empty() {
            return Ok(String::new());
        }
        let repo_root = workspace.repo_root()?;

        let provided_path = Path::new(&file_path);

        if provided_path.is_absolute() {
            if provided_path.starts_with(&repo_root) {
                return Ok(provided_path
                    .strip_prefix(&repo_root)
                    .map_err(|_| FileAdditionError::PathOutsideRepo {
                        path: provided_path.to_path_buf(),
                        repo_root: repo_root.clone(),
                    })?
                    .to_string_lossy()
                    .to_string());
            }
        } else {
            let absolute = repo_root.join(provided_path);
            if absolute.exists() {
                return Ok(absolute
                    .strip_prefix(&repo_root)
                    .map_err(|_| FileAdditionError::PathOutsideRepo {
                        path: absolute.clone(),
                        repo_root: repo_root.clone(),
                    })?
                    .to_string_lossy()
                    .to_string());
            }
        }

        if provided_path.is_absolute() {
            return Self::asset_relative_path(workspace, checksum, file_type);
        }

        Ok(file_path)
    }

    /// This exists along with store_file because one uses the model Self attributes to
    /// identify where to store it, while this works without having an initialized model.
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

        OpenDalLocation::from_local_path(source_path)
            .map_err(opendal_file_addition_error)?
            .copy_to_local_path(&asset_path)
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

        let file_path = Self::path_from_uri(&file_addition.asset_uri)
            .unwrap_or_else(|| file_addition.asset_uri.clone());
        let source_path = if Path::new(&file_path).is_absolute() {
            PathBuf::from(&file_path)
        } else {
            workspace.repo_root()?.join(&file_path)
        };
        if source_path == asset_path {
            return Ok(());
        }
        OpenDalLocation::from_local_path(&source_path)
            .map_err(opendal_file_store_error)?
            .copy_to_local_path(&asset_path)
            .map_err(FileStoreError::IoError)?;
        Ok(())
    }
}

impl Read for LocalAssetUri {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.read_file.is_none() {
            self.read_file = Some(
                OpenDalLocation::from_local_path(&self.io_path())
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
            let location =
                OpenDalLocation::from_local_path(&self.io_path()).map_err(io::Error::other)?;
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

    fn reader(&self, _workspace: &Workspace) -> Result<Box<dyn Read>, FileAdditionError> {
        Ok(Box::new(
            OpenDalLocation::from_remote_uri(&self.asset_uri)?.reader()?,
        ))
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
        time::{Duration, Instant},
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
    fn test_source_file_path_returns_absolute_path_from_file_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.base_dir();

        let absolute_path = repo_root.join("inputs").join("absolute.txt");
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"absolute").unwrap();
        let absolute_string = absolute_path.to_string_lossy().to_string();

        let absolute =
            LocalAssetUri::source_file_path(workspace, absolute_string.as_str()).unwrap();

        assert_eq!(absolute, absolute_string);
    }

    #[test]
    fn test_stored_file_path_returns_relative_path_from_file_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.base_dir();

        let absolute_path = repo_root.join("inputs").join("absolute.txt");
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"absolute").unwrap();
        let absolute_string = absolute_path.to_string_lossy().to_string();
        let relative_string = absolute_path
            .strip_prefix(repo_root)
            .unwrap()
            .to_string_lossy()
            .to_string();

        let relative = LocalAssetUri::stored_file_path(
            workspace,
            absolute_string.as_str(),
            &calculate_file_checksum(&absolute_path).unwrap(),
            FileTypes::Fasta,
        )
        .unwrap();

        assert_eq!(relative, relative_string);
    }

    #[test]
    fn test_source_file_path_returns_absolute_path_from_relative_path_in_repo() {
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
            LocalAssetUri::source_file_path(workspace, relative_string.as_str()).unwrap();

        assert_eq!(absolute, absolute_string);
    }

    #[test]
    fn test_stored_file_path_returns_relative_path_from_relative_path_in_repo() {
        let context = setup_gen();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();

        let relative_path = PathBuf::from("relative/path/file.txt");
        let absolute_path = repo_root.join(&relative_path);
        fs::create_dir_all(absolute_path.parent().unwrap()).unwrap();
        fs::write(&absolute_path, b"relative").unwrap();
        let relative_string = relative_path.to_string_lossy().to_string();

        let relative = LocalAssetUri::stored_file_path(
            workspace,
            relative_string.as_str(),
            &calculate_file_checksum(&absolute_path).unwrap(),
            FileTypes::Fasta,
        )
        .unwrap();

        assert_eq!(relative, relative_string);
    }

    #[test]
    fn test_source_file_path_fallback_to_initial_path() {
        let context = setup_gen();
        let workspace = context.workspace();

        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();
        let outside_string = outside_file.path().to_string_lossy().to_string();

        let absolute = LocalAssetUri::source_file_path(workspace, outside_string.as_str()).unwrap();

        assert_eq!(absolute, outside_string);
    }

    #[test]
    fn test_stored_file_path_returns_asset_path_on_file_outside_repo() {
        let context = setup_gen();
        let workspace = context.workspace();

        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        outside_file.write_all(b"outside repo").unwrap();
        outside_file.flush().unwrap();
        let outside_string = outside_file.path().to_string_lossy().to_string();
        let checksum = calculate_file_checksum(outside_file.path()).unwrap();

        let relative = LocalAssetUri::stored_file_path(
            workspace,
            outside_string.as_str(),
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
    fn test_source_file_path_gives_initial_path_if_no_resolution() {
        let context = setup_gen();
        let workspace = context.workspace();

        let absolute = LocalAssetUri::source_file_path(workspace, "detached/file.txt").unwrap();
        assert_eq!(absolute, "detached/file.txt");
    }

    #[test]
    fn test_stored_file_path_gives_initial_path_if_no_resolution() {
        let context = setup_gen();
        let workspace = context.workspace();

        let relative = LocalAssetUri::stored_file_path(
            workspace,
            "detached/file.txt",
            &HashId::convert_str("detached"),
            FileTypes::Fasta,
        )
        .unwrap();
        assert_eq!(relative, "detached/file.txt");
    }

    #[test]
    fn test_source_file_path_empty() {
        let context = setup_gen();
        let workspace = context.workspace();

        let absolute_empty = LocalAssetUri::source_file_path(workspace, "").unwrap();
        assert_eq!(absolute_empty, "");
    }

    #[test]
    fn test_stored_file_path_empty() {
        let context = setup_gen();
        let workspace = context.workspace();

        let relative_empty = LocalAssetUri::stored_file_path(
            workspace,
            "",
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
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
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
