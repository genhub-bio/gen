use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, RwLock},
};

use rusqlite::Connection;

use crate::{HashId, errors::ConfigError};

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
            .join("changeset")
            .join(format!("{hash}"));
        ensure_dir(&path);
        path
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RepoKind {
    Operations,
    Graph,
}

pub struct RepoHandle {
    workspace: Arc<Workspace>,
    kind: RepoKind,
    conn: Connection,
}

impl RepoHandle {
    pub fn new(kind: RepoKind, workspace: Arc<Workspace>, conn: Connection) -> Self {
        Self {
            workspace,
            kind,
            conn,
        }
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn conn_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    pub fn workspace(&self) -> &Workspace {
        self.workspace.as_ref()
    }

    pub fn kind(&self) -> RepoKind {
        self.kind
    }

    pub fn path(&self) -> Option<PathBuf> {
        self.conn.path().map(PathBuf::from)
    }
}

pub struct DbContext {
    workspace: Arc<Workspace>,
    operations: RepoHandle,
    graph: RepoHandle,
}

impl DbContext {
    pub fn new(workspace: Workspace, graph_conn: Connection, operations_conn: Connection) -> Self {
        let workspace = Arc::new(workspace);
        let operations = RepoHandle::new(RepoKind::Operations, workspace.clone(), operations_conn);
        let graph = RepoHandle::new(RepoKind::Graph, workspace.clone(), graph_conn);
        Self {
            workspace,
            operations,
            graph,
        }
    }

    pub fn workspace(&self) -> &Workspace {
        self.workspace.as_ref()
    }

    pub fn operations(&self) -> &RepoHandle {
        &self.operations
    }

    pub fn graph(&self) -> &RepoHandle {
        &self.graph
    }
}

thread_local! {
pub static WORKSPACE: LazyLock<RwLock<Workspace>> =
    LazyLock::new(|| RwLock::new(Workspace::from_current_dir()));
}

fn ensure_dir(path: &Path) {
    if !path.is_dir() {
        fs::create_dir_all(path).unwrap();
    }
}

pub fn set_base_dir(d: &Path) {
    WORKSPACE
        .try_with(|v| {
            let mut w = v.write().unwrap();
            *w = Workspace::new(d)
        })
        .unwrap();
}

pub fn get_base_dir() -> PathBuf {
    WORKSPACE.with(|v| v.read().unwrap().base_dir.clone())
}

pub fn get_or_create_gen_dir() -> PathBuf {
    WORKSPACE.with(|v| v.read().unwrap().ensure_gen_dir())
}

pub fn get_gen_dir() -> Option<String> {
    WORKSPACE
        .with(|v| v.read().unwrap().find_gen_dir())
        .and_then(|p| p.to_str().map(|p| p.to_string()))
}

pub fn get_repo_root_path() -> Result<PathBuf, ConfigError> {
    WORKSPACE.with(|v| v.read().unwrap().repo_root())
}

pub fn get_gen_db_path() -> Result<PathBuf, ConfigError> {
    WORKSPACE.with(|v| v.read().unwrap().gen_db_path())
}

pub fn get_changeset_path(hash: &HashId) -> PathBuf {
    WORKSPACE.with(|v| v.read().unwrap().changeset_path(hash))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_finds_gen_dir() {
        let tmp_dir = tempdir().unwrap().keep();
        set_base_dir(&tmp_dir);
        get_or_create_gen_dir();
        assert!(get_gen_dir().is_some());
    }

    #[test]
    fn test_set_base_dir() {
        let old_dir = get_base_dir();
        let tmp_dir = tempdir().unwrap().keep();
        set_base_dir(&tmp_dir);
        assert_eq!(tmp_dir, get_base_dir());
        assert_ne!(get_base_dir(), old_dir);
    }

    #[test]
    fn test_get_repo_root_path() {
        let gen_dir = setup_gen_environment();
        let expected_root = gen_dir.parent().unwrap().to_path_buf();
        assert_eq!(get_repo_root_path().unwrap(), expected_root);
    }

    #[test]
    fn test_get_repo_root_path_missing_gen_dir() {
        let tmp_dir = tempdir().unwrap().keep();
        set_base_dir(&tmp_dir);
        assert_eq!(Err(ConfigError::GenDirectoryNotFound), get_repo_root_path());
    }

    fn setup_gen_environment() -> PathBuf {
        let tmp_dir = tempdir().unwrap().keep();
        set_base_dir(&tmp_dir);
        get_or_create_gen_dir()
    }
}
