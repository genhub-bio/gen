use std::{ops::Deref, rc::Rc, sync::Arc};

use gen_core::{config::Workspace, errors::ConfigError};
use rusqlite::Connection;

#[derive(Debug)]
pub struct GraphConnection(pub Connection);

/// The Deref lets us use GraphConnection any place a &Connection is expected, such as the generic traits for query
impl Deref for GraphConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        &self.0
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

pub struct DbHandle<C> {
    workspace: Arc<Workspace>,
    conn: Rc<C>,
}

impl<C> Clone for DbHandle<C> {
    fn clone(&self) -> Self {
        Self {
            workspace: self.workspace.clone(),
            conn: self.conn.clone(),
        }
    }
}

impl<C> DbHandle<C> {
    pub fn new(workspace: Arc<Workspace>, conn: Rc<C>) -> Self {
        Self { workspace, conn }
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    pub fn conn(&self) -> &C {
        // We don't use &self.conn here to get rid of the Rc
        self.conn.as_ref()
    }
}

pub type GraphHandle = DbHandle<GraphConnection>;
pub type OperationsHandle = DbHandle<OperationsConnection>;

#[derive(Clone)]
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

    pub fn set_graph(&mut self, graph_conn: GraphConnection) {
        self.graph = DbHandle::new(self.workspace.clone(), graph_conn.into());
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
