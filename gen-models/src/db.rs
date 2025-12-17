use std::{ops::Deref, path::Path, rc::Rc, sync::Arc};

use gen_core::{config::Workspace, errors::ConfigError};
use rusqlite::Connection;

#[derive(Debug)]
pub struct GraphConnection(pub Connection);

impl Deref for GraphConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GraphConnection {
    pub fn graph_conn(&self) -> &Self {
        self
    }
}

#[derive(Debug)]
pub struct OperationsConnection(pub Connection);

impl Deref for OperationsConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl OperationsConnection {
    pub fn operations_conn(&self) -> &Self {
        self
    }
}

#[derive(Clone)]
pub struct DbHandle<C> {
    workspace: Arc<Workspace>,
    conn: Rc<C>,
}

impl<C> DbHandle<C> {
    pub fn new(workspace: Arc<Workspace>, conn: Rc<C>) -> Self {
        Self { workspace, conn }
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    pub fn conn(&self) -> &C {
        self.conn.as_ref()
    }

    pub fn conn_rc(&self) -> Rc<C> {
        self.conn.clone()
    }

    pub fn path(&self) -> Option<&Path>
    where
        C: Deref<Target = Connection>,
    {
        self.conn.path().map(Path::new)
    }

    pub fn ensure_gen_dir(&self) -> std::path::PathBuf {
        self.workspace.ensure_gen_dir()
    }
}

pub type GraphHandle = DbHandle<GraphConnection>;
pub type OperationsHandle = DbHandle<OperationsConnection>;

pub struct DbContext {
    workspace: Arc<Workspace>,
    graph: GraphHandle,
    operations: OperationsHandle,
}

impl DbContext {
    pub fn new(
        workspace: Workspace,
        graph_conn: GraphConnection,
        operations_conn: OperationsConnection,
    ) -> Self {
        let workspace = Arc::new(workspace);
        let graph = DbHandle::new(workspace.clone(), graph_conn.into());
        let operations = DbHandle::new(workspace.clone(), operations_conn.into());
        Self {
            workspace,
            graph,
            operations,
        }
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    pub fn graph(&self) -> &GraphHandle {
        &self.graph
    }

    pub fn operations(&self) -> &OperationsHandle {
        &self.operations
    }

    pub fn repo_root(&self) -> Result<std::path::PathBuf, ConfigError> {
        self.workspace.repo_root()
    }

    pub fn gen_db_path(&self) -> Result<std::path::PathBuf, ConfigError> {
        self.workspace.gen_db_path()
    }
}
