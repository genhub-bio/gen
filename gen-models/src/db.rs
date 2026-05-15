use std::{
    ops::Deref,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, mpsc},
    thread,
};

use gen_core::{
    config::Workspace,
    errors::{ConfigError, ConnectionError},
};
use rusqlite::Connection;
use thiserror::Error;
use tokio::sync::oneshot;

use crate::migrations::{run_migrations, run_operation_migrations};

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

    pub fn from_paths(
        workspace: Workspace,
        graph_db_path: impl AsRef<Path>,
        operations_db_path: impl AsRef<Path>,
    ) -> Result<Self, ConnectionError> {
        let graph_conn = open_graph_connection(graph_db_path)?;
        let operations_conn = open_operations_connection(operations_db_path)?;

        Ok(Self::new(workspace, graph_conn, operations_conn))
    }

    pub fn from_workspace(
        workspace: Workspace,
        graph_db_path: impl AsRef<Path>,
    ) -> Result<Self, ConnectionError> {
        let operations_db_path = workspace.gen_db_path()?;
        Self::from_paths(workspace, graph_db_path, operations_db_path)
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

pub fn open_graph_connection(
    db_path: impl AsRef<Path>,
) -> Result<GraphConnection, ConnectionError> {
    let mut conn = Connection::open(db_path.as_ref()).map_err(ConnectionError::OpenFailed)?;
    rusqlite::vtab::array::load_module(&conn).map_err(ConnectionError::OpenFailed)?;
    run_migrations(&mut conn);
    Ok(GraphConnection(conn))
}

pub fn open_operations_connection(
    db_path: impl AsRef<Path>,
) -> Result<OperationsConnection, ConnectionError> {
    let mut conn = Connection::open(db_path.as_ref()).map_err(ConnectionError::OpenFailed)?;
    rusqlite::vtab::array::load_module(&conn).map_err(ConnectionError::OpenFailed)?;
    run_operation_migrations(&mut conn);
    Ok(OperationsConnection(conn))
}

type DbJob = Box<dyn FnOnce(&DbContext) + Send + 'static>;

enum DbMessage {
    Run(DbJob),
}

#[derive(Clone)]
pub struct DbWorker {
    sender: mpsc::Sender<DbMessage>,
}

#[derive(Debug, Error)]
pub enum DbWorkerError {
    #[error("database worker is not running")]
    Stopped,
    #[error("database worker dropped the response")]
    ResponseDropped,
}

impl DbWorker {
    pub fn spawn_from_paths(
        workspace: Workspace,
        graph_db_path: impl Into<PathBuf>,
        operations_db_path: impl Into<PathBuf>,
    ) -> Result<Self, ConnectionError> {
        let graph_db_path = graph_db_path.into();
        let operations_db_path = operations_db_path.into();
        let (sender, receiver) = mpsc::channel::<DbMessage>();
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);

        thread::Builder::new()
            .name("gen-db-worker".to_string())
            .spawn(move || {
                let context =
                    match DbContext::from_paths(workspace, graph_db_path, operations_db_path) {
                        Ok(context) => context,
                        Err(err) => {
                            let _ = ready_sender.send(Err(err));
                            return;
                        }
                    };

                let _ = ready_sender.send(Ok(()));

                while let Ok(DbMessage::Run(job)) = receiver.recv() {
                    job(&context);
                }
            })
            .map_err(|err| {
                ConnectionError::DatabaseTracking(format!("failed to spawn database worker: {err}"))
            })?;

        match ready_receiver.recv() {
            Ok(Ok(())) => Ok(Self { sender }),
            Ok(Err(err)) => Err(err),
            Err(err) => Err(ConnectionError::DatabaseTracking(format!(
                "database worker failed to initialize: {err}"
            ))),
        }
    }

    pub fn spawn_from_workspace(
        workspace: Workspace,
        graph_db_path: impl Into<PathBuf>,
    ) -> Result<Self, ConnectionError> {
        let operations_db_path = workspace.gen_db_path()?;
        Self::spawn_from_paths(workspace, graph_db_path, operations_db_path)
    }

    pub async fn run<R, F>(&self, f: F) -> Result<R, DbWorkerError>
    where
        R: Send + 'static,
        F: FnOnce(&DbContext) -> R + Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        self.sender
            .send(DbMessage::Run(Box::new(move |context| {
                let _ = sender.send(f(context));
            })))
            .map_err(|_| DbWorkerError::Stopped)?;

        receiver.await.map_err(|_| DbWorkerError::ResponseDropped)
    }
}
