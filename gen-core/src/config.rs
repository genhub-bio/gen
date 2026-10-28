use std::{
    env, fmt, fs,
    io::{self, Read, Seek},
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{DoltHashId, HashId, Sha256Hash, errors::ConfigError};

pub const ASSETS_DIR_NAME: &str = "assets";
pub const CACHE_DIR_NAME: &str = "cache";
pub const DEFAULT_GRAPH_DB_NAME: &str = "default.db";

/// A reader whose cursor can seek within a content-addressed asset.
pub trait AssetReadSeek: Read + Seek + Send {}

impl<T> AssetReadSeek for T where T: Read + Seek + Send {}

/// Identity required to open one archived repository asset from an external read source.
pub struct AssetReadRequest<'asset> {
    /// Stable Gen asset reference identity.
    pub id: HashId,
    /// SHA-256 of the exact archived bytes in object storage.
    pub checksum: Sha256Hash,
    /// Recorded URI, used only as metadata to derive the content-addressed archive name.
    pub uri: &'asset str,
}

/// Opens immutable assets for a workspace whose content is not present in its local store.
///
/// A source is attached only to workspaces that explicitly opt into remote asset reads. The
/// source controls which published branch snapshot its IDs and URLs may resolve against.
pub trait AssetReaderSource: Send + Sync {
    /// Pins future asset reads to a verified published branch and commit hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the ref is invalid or the source has been closed.
    fn set_ref(&self, branch: &str, snapshot_hash: &str) -> io::Result<()>;

    /// Closes the source so later readers cannot obtain or refresh URLs.
    fn close(&self);

    /// Returns the verified source commit when it belongs to the selected repository remote.
    ///
    /// Push orchestration may use this as an initial asset baseline for a new branch, after
    /// independently proving that the commit is on the branch's first-parent history.
    fn published_base_hash_for_remote(&self, remote_url: &str) -> Option<DoltHashId> {
        let _ = remote_url;
        None
    }

    /// Opens the archived bytes for one asset using a seekable, bounded reader.
    ///
    /// # Errors
    ///
    /// Returns an error if the source is closed, the asset is outside the pinned ref, or its
    /// content cannot be read.
    fn open(&self, request: &AssetReadRequest<'_>) -> io::Result<Box<dyn AssetReadSeek>>;
}

#[derive(Clone)]
pub struct Workspace {
    base_dir: PathBuf,
    graph_db_path_override: Option<PathBuf>,
    asset_reader_source: Option<Arc<dyn AssetReaderSource>>,
}

impl fmt::Debug for Workspace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The source owns signed URLs in memory, so show only whether it is configured.
        formatter
            .debug_struct("Workspace")
            .field("base_dir", &self.base_dir)
            .field("graph_db_path_override", &self.graph_db_path_override)
            .field(
                "has_asset_reader_source",
                &self.asset_reader_source.is_some(),
            )
            .finish()
    }
}

impl PartialEq for Workspace {
    fn eq(&self, other: &Self) -> bool {
        self.base_dir == other.base_dir
            && self.graph_db_path_override == other.graph_db_path_override
            && match (&self.asset_reader_source, &other.asset_reader_source) {
                (Some(left), Some(right)) => Arc::ptr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for Workspace {}

impl Workspace {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            graph_db_path_override: None,
            asset_reader_source: None,
        }
    }

    /// Uses an explicit graph database path, including a credential-free SQLite URI for a
    /// database exposed through a registered VFS.
    ///
    /// The override belongs to this workspace handle only; GenHub credentials remain owned by
    /// the connection that registered the VFS and are never stored in workspace configuration.
    pub fn with_graph_db_path_override(mut self, path: impl Into<PathBuf>) -> Self {
        self.graph_db_path_override = Some(path.into());
        self
    }

    /// Uses an explicit reader for immutable assets missing from this direct graph workspace.
    ///
    /// Local content-addressed files remain the first choice. This capability is intended for a
    /// direct graph view that opens referenced asset bytes on demand.
    pub fn with_asset_reader_source(mut self, source: Arc<dyn AssetReaderSource>) -> Self {
        self.asset_reader_source = Some(source);
        self
    }

    /// Returns the explicitly configured reader for missing immutable assets, if any.
    pub fn asset_reader_source(&self) -> Option<&Arc<dyn AssetReaderSource>> {
        self.asset_reader_source.as_ref()
    }

    /// Returns whether the graph database path comes from an explicit override.
    pub fn has_graph_db_path_override(&self) -> bool {
        self.graph_db_path_override.is_some()
    }

    pub fn from_current_dir() -> Self {
        Self::new(env::current_dir().unwrap())
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn ensure_gen_dir(&self) -> PathBuf {
        let gen_path = self.base_dir.join(".gen");
        ensure_dir(&gen_path);
        let assets = gen_path.join(ASSETS_DIR_NAME);
        ensure_dir(&assets);
        gen_path
    }

    pub fn find_gen_dir(&self) -> Option<PathBuf> {
        let mut cur_dir = self.base_dir.as_path();
        let mut gen_path = cur_dir.join(".gen");
        while !gen_path.is_dir() {
            cur_dir = cur_dir.parent()?;
            gen_path = cur_dir.join(".gen");
        }
        Some(gen_path)
    }

    pub fn repo_root(&self) -> Result<PathBuf, ConfigError> {
        let gen_dir = self
            .find_gen_dir()
            .ok_or(ConfigError::GenDirectoryNotFound)?;

        gen_dir
            .parent()
            .map(Path::to_path_buf)
            .ok_or(ConfigError::RepoRootNotFound)
    }

    pub fn gen_db_path(&self) -> Result<PathBuf, ConfigError> {
        self.find_gen_dir()
            .map(|dir| dir.join("gen.db"))
            .ok_or(ConfigError::GenDirectoryNotFound)
    }

    pub fn graph_db_path(&self) -> Result<PathBuf, ConfigError> {
        if let Some(path) = &self.graph_db_path_override {
            return Ok(path.clone());
        }

        self.find_gen_dir()
            .map(|dir| dir.join(DEFAULT_GRAPH_DB_NAME))
            .ok_or(ConfigError::GenDirectoryNotFound)
    }

    pub fn asset_dir(&self) -> Result<PathBuf, ConfigError> {
        Ok(self
            .find_gen_dir()
            .ok_or(ConfigError::GenDirectoryNotFound)?
            .join(ASSETS_DIR_NAME))
    }

    /// Returns the workspace cache location without creating it.
    ///
    /// Cache inspection and clearing use this path so those operations never create an empty
    /// `.gen/cache` as a side effect.
    pub fn find_cache_dir(&self) -> Option<PathBuf> {
        self.find_gen_dir().map(|dir| dir.join(CACHE_DIR_NAME))
    }

    /// Creates and returns the workspace cache used for downloaded remote data.
    ///
    /// The directory is created lazily so repositories that only use local assets do not acquire
    /// cache state.
    pub fn ensure_cache_dir(&self) -> Result<PathBuf, ConfigError> {
        let dir = self
            .find_cache_dir()
            .ok_or(ConfigError::GenDirectoryNotFound)?;
        ensure_dir(&dir);
        Ok(dir)
    }

    pub fn find_search_index(&self) -> Option<PathBuf> {
        self.find_gen_dir().map(|d| d.join("search_index"))
    }

    pub fn ensure_search_index(&self) -> Result<PathBuf, ConfigError> {
        let dir = self
            .find_gen_dir()
            .ok_or(ConfigError::GenDirectoryNotFound)?
            .join("search_index");
        ensure_dir(&dir);
        Ok(dir)
    }
}

fn ensure_dir(path: &Path) {
    if !path.is_dir() {
        fs::create_dir_all(path).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn ensure_gen_dir_creates_directory() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let workspace = Workspace::new(&tmp_dir_path);

        let gen_dir = workspace.ensure_gen_dir();

        assert_eq!(gen_dir, tmp_dir_path.join(".gen"));
        assert!(gen_dir.is_dir());
    }

    #[test]
    fn find_gen_dir_walks_up_tree() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let root_workspace = Workspace::new(&tmp_dir_path);
        let gen_dir = root_workspace.ensure_gen_dir();

        let nested_dir = tmp_dir_path.join("nested").join("deep");
        fs::create_dir_all(&nested_dir).unwrap();
        let nested_workspace = Workspace::new(&nested_dir);

        assert_eq!(nested_workspace.find_gen_dir(), Some(gen_dir));
    }

    #[test]
    fn repo_root_returns_parent_of_gen_dir() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let workspace = Workspace::new(&tmp_dir_path);
        workspace.ensure_gen_dir();

        assert_eq!(workspace.repo_root().unwrap(), tmp_dir_path);
    }

    #[test]
    fn repo_root_errors_when_missing_gen_dir() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let workspace = Workspace::new(&tmp_dir_path);

        assert_eq!(
            Err(ConfigError::GenDirectoryNotFound),
            workspace.repo_root()
        );
    }

    #[test]
    fn gen_db_path_resolves_inside_gen_dir() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let workspace = Workspace::new(&tmp_dir_path);
        let gen_dir = workspace.ensure_gen_dir();

        assert_eq!(workspace.gen_db_path().unwrap(), gen_dir.join("gen.db"));
    }

    #[test]
    fn test_graph_db_path_resolves_inside_gen_dir() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let workspace = Workspace::new(&tmp_dir_path);
        let gen_dir = workspace.ensure_gen_dir();

        assert_eq!(
            workspace.graph_db_path().unwrap(),
            gen_dir.join(DEFAULT_GRAPH_DB_NAME)
        );
    }

    #[test]
    fn test_graph_db_path_uses_explicit_override() {
        let temp = tempdir().unwrap();
        let workspace = Workspace::new(temp.path())
            .with_graph_db_path_override("file:///owned/default.db?vfs=cloud_session");

        assert_eq!(
            workspace.graph_db_path().unwrap(),
            PathBuf::from("file:///owned/default.db?vfs=cloud_session")
        );
        assert!(workspace.has_graph_db_path_override());
    }

    #[test]
    fn asset_dir_creates_assets_directory() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let workspace = Workspace::new(&tmp_dir_path);
        workspace.ensure_gen_dir();

        let asset_dir = workspace.asset_dir().unwrap();

        assert_eq!(asset_dir, tmp_dir_path.join(".gen").join(ASSETS_DIR_NAME));
        assert!(asset_dir.is_dir());
    }

    #[test]
    fn test_ensure_cache_dir_creates_cache_directory_lazily() {
        let tmp_dir = tempdir().unwrap();
        let workspace = Workspace::new(tmp_dir.path());
        let gen_dir = workspace.ensure_gen_dir();
        let cache_dir = gen_dir.join(CACHE_DIR_NAME);
        assert!(!cache_dir.exists());

        assert_eq!(workspace.ensure_cache_dir().unwrap(), cache_dir);
        assert!(cache_dir.is_dir());
    }
}
