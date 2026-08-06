use std::{
    env, fs,
    path::{Path, PathBuf},
};

use crate::errors::ConfigError;

pub const ASSETS_DIR_NAME: &str = "assets";
pub const CACHE_DIR_NAME: &str = "cache";
pub const DEFAULT_GRAPH_DB_NAME: &str = "default.db";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Workspace {
    base_dir: PathBuf,
}

impl Workspace {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
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
