use std::{ops::Deref, path::Path, rc::Rc, sync::Arc};

use gen_core::{config::Workspace, errors::ConfigError};
use rusqlite::Connection;

use crate::{
    history::dolt::{active_branch, checkout, connect_branch},
    migrations::{run_migrations, run_operation_migrations},
    operations::Defaults,
};

#[derive(Debug)]
pub struct GraphConnection(pub Connection);

/// The Deref lets us use GraphConnection any place a &Connection is expected, such as the generic traits for query
impl Deref for GraphConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn get_connection(path: impl AsRef<Path>) -> Result<GraphConnection, rusqlite::Error> {
    let mut conn = Connection::open(path)?;
    rusqlite::vtab::array::load_module(&conn)?;
    run_migrations(&mut conn);
    Ok(GraphConnection(conn))
}

#[derive(Debug)]
pub struct ConfigConnection(pub Connection);

impl Deref for ConfigConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn get_config_connection(path: impl AsRef<Path>) -> Result<ConfigConnection, rusqlite::Error> {
    let mut conn = Connection::open(path)?;
    rusqlite::vtab::array::load_module(&conn)?;
    run_operation_migrations(&mut conn);
    Ok(ConfigConnection(conn))
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
pub type ConfigHandle = DbHandle<ConfigConnection>;

#[derive(Clone)]
pub struct DbContext {
    workspace: Arc<Workspace>,
    graph: GraphHandle,
    config: ConfigHandle,
}

impl DbContext {
    fn build(
        workspace: Workspace,
        graph_conn: GraphConnection,
        config_conn: ConfigConnection,
    ) -> Self {
        let workspace = Arc::new(workspace);
        let graph = DbHandle::new(workspace.clone(), graph_conn.into());
        let config = DbHandle::new(workspace.clone(), config_conn.into());
        Self {
            workspace,
            graph,
            config,
        }
    }

    pub fn new(
        workspace: Workspace,
        graph_conn: GraphConnection,
        config_conn: ConfigConnection,
    ) -> Result<Self, rusqlite::Error> {
        let context = Self::build(workspace, graph_conn, config_conn);
        if let Some(intended_branch) = Defaults::get_current_branch(context.config().conn()) {
            let branch_name = active_branch(context.graph().conn())?;
            if branch_name != intended_branch {
                connect_branch(context.graph().conn(), &intended_branch)?;
            }
        }
        Ok(context)
    }

    pub fn new_with_ref(
        workspace: Workspace,
        graph_conn: GraphConnection,
        config_conn: ConfigConnection,
        history_ref: &str,
    ) -> Result<Self, rusqlite::Error> {
        let context = Self::build(workspace, graph_conn, config_conn);
        let branch_exists = context.graph().conn().query_row(
            "SELECT EXISTS(SELECT 1 FROM dolt_branches WHERE name = ?1)",
            [history_ref],
            |row| row.get::<_, bool>(0),
        )?;
        if branch_exists {
            let branch_name = active_branch(context.graph().conn())?;
            if branch_name != history_ref {
                connect_branch(context.graph().conn(), history_ref)?;
            }
        } else {
            checkout(context.graph().conn(), history_ref)?;
        }
        Ok(context)
    }

    pub fn new_raw(
        workspace: Workspace,
        graph_conn: GraphConnection,
        config_conn: ConfigConnection,
    ) -> Self {
        Self::build(workspace, graph_conn, config_conn)
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

    pub fn config(&self) -> &ConfigHandle {
        &self.config
    }

    pub fn repo_root(&self) -> Result<std::path::PathBuf, ConfigError> {
        self.workspace.repo_root()
    }

    pub fn gen_db_path(&self) -> Result<std::path::PathBuf, ConfigError> {
        self.workspace.gen_db_path()
    }
}

#[cfg(test)]
mod tests {
    use gen_core::config::Workspace;
    use tempfile::tempdir;

    use super::{DbContext, get_config_connection, get_connection};
    use crate::{
        collection::Collection,
        history::dolt::{active_branch, commit_all, connect_branch, create_branch},
        operations::Defaults,
        sample::{NewSample, Sample},
    };

    fn branch_has_sample(context: &DbContext, sample_name: &str) -> bool {
        context
            .graph()
            .conn()
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM samples WHERE name = ?1)",
                [sample_name],
                |row| row.get::<_, bool>(0),
            )
            .expect("should query sample existence")
    }

    fn setup_branch_restore_repo() -> Workspace {
        let temp_dir = tempdir().expect("should create temp repo");
        let workspace = Workspace::new(temp_dir.keep());
        workspace.ensure_gen_dir();
        let graph_path = workspace
            .graph_db_path()
            .expect("should resolve graph database path");
        let config_path = workspace
            .gen_db_path()
            .expect("should resolve config database path");

        let graph_conn = get_connection(&graph_path).expect("should open graph connection");
        let config_conn =
            get_config_connection(&config_path).expect("should open config connection");

        Collection::create(&graph_conn, "main-collection").expect("should create main collection");
        commit_all(&graph_conn, "initial commit").expect("should commit initial graph state");
        create_branch(&graph_conn, "feature").expect("should create feature branch");
        connect_branch(&graph_conn, "feature").expect("should connect feature branch");
        Sample::create(
            &graph_conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should create feature sample");
        commit_all(&graph_conn, "feature commit").expect("should commit feature branch change");
        Defaults::set_current_branch(&config_conn, Some("feature"))
            .expect("should persist current branch intent");

        workspace
    }

    #[test]
    fn test_new_restores_saved_current_branch() {
        let workspace = setup_branch_restore_repo();
        let graph_path = workspace
            .graph_db_path()
            .expect("should resolve graph database path");
        let config_path = workspace
            .gen_db_path()
            .expect("should resolve config database path");
        let graph_conn = get_connection(&graph_path).expect("should reopen graph connection");
        let config_conn =
            get_config_connection(&config_path).expect("should reopen config connection");

        let context = DbContext::new(workspace, graph_conn, config_conn)
            .expect("should restore current branch when opening context");

        assert_eq!(
            active_branch(context.graph().conn()).expect("should resolve active branch"),
            "feature"
        );
        assert!(
            branch_has_sample(&context, "feature-sample"),
            "default DbContext construction should restore the saved branch contents"
        );
    }

    #[test]
    fn test_new_with_ref_overrides_saved_current_branch() {
        let workspace = setup_branch_restore_repo();
        let graph_path = workspace
            .graph_db_path()
            .expect("should resolve graph database path");
        let config_path = workspace
            .gen_db_path()
            .expect("should resolve config database path");
        let graph_conn = get_connection(&graph_path).expect("should reopen graph connection");
        let config_conn =
            get_config_connection(&config_path).expect("should reopen config connection");

        let context = DbContext::new_with_ref(workspace, graph_conn, config_conn, "main")
            .expect("should open context on explicit ref");

        assert_eq!(
            active_branch(context.graph().conn()).expect("should resolve active branch"),
            "main"
        );
        assert!(
            !branch_has_sample(&context, "feature-sample"),
            "explicit ref checkout should override the saved branch intent"
        );
    }
}
