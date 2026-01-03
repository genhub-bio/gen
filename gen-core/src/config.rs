use std::{
    env, fs,
    path::{Path, PathBuf},
};

use crate::{HashId, errors::ConfigError};

pub const CHANGESET_DIR_NAME: &str = "changesets";

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
        let changesets = gen_path.join(CHANGESET_DIR_NAME);
        ensure_dir(&changesets);
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

    pub fn changeset_path(&self, hash: &HashId) -> PathBuf {
        let path = self
            .ensure_gen_dir()
            .join(CHANGESET_DIR_NAME)
            .join(format!("{hash}"));
        ensure_dir(&path);
        path
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
    fn changeset_path_creates_directory_for_hash() {
        let tmp_dir = tempdir().unwrap();
        let tmp_dir_path = tmp_dir.path().to_path_buf();
        let workspace = Workspace::new(&tmp_dir_path);
        let hash = HashId([0; 32]);

        let path = workspace.changeset_path(&hash);

        assert_eq!(
            path,
            tmp_dir_path
                .join(".gen")
                .join(CHANGESET_DIR_NAME)
                .join(format!("{hash}"))
        );
        assert!(path.is_dir());
    }
}
