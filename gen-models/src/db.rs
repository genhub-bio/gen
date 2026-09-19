use std::{ops::Deref, path::Path, rc::Rc, sync::Arc};

use gen_core::{config::Workspace, errors::ConfigError};
use rusqlite::{Connection, limits::Limit};

use crate::{
    history::dolt::{active_branch, checkout, connect_branch},
    migrations::{run_config_migrations, run_migrations},
    operations::Defaults,
};

/// Returns the SQLite variable parameter limit for the provided connection.
pub fn sqlite_parameter_limit(conn: &Connection) -> usize {
    let limit = conn
        .limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER)
        .expect("SQLite parameter limit should be readable");
    usize::try_from(limit).expect("SQLite parameter limit should be positive")
}

/// Computes how many rows can be inserted per batch given a parameter count.
pub fn max_rows_per_batch(conn: &Connection, params_per_row: usize) -> usize {
    let params_per_row = params_per_row.max(1);
    let max_params = sqlite_parameter_limit(conn);
    (max_params / params_per_row).max(1)
}

#[derive(Debug)]
pub struct GraphConnection(pub Connection);

/// The Deref lets us use GraphConnection any place a &Connection is expected, such as the generic traits for query
impl Deref for GraphConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GraphConnection {
    /// Starts a plain SQL transaction unless one is already active.
    ///
    /// The returned flag is true when the caller already owned the transaction. Pass it to
    /// [`Self::end_transaction`] so only the transaction started by this method is finalized.
    pub fn start_transaction(&self) -> rusqlite::Result<bool> {
        let in_transaction = !self.is_autocommit();
        if !in_transaction {
            self.execute("BEGIN;", [])?;
        }
        Ok(in_transaction)
    }

    /// Finishes a transaction started by [`Self::start_transaction`].
    ///
    /// An existing transaction remains owned by its caller. Dolt version-control statements can
    /// seal a transaction opened by this helper themselves, so commit and rollback are
    /// conditional on the connection still being in a transaction after the operation returns.
    pub fn end_transaction<T>(
        &self,
        in_transaction: bool,
        result: rusqlite::Result<T>,
    ) -> rusqlite::Result<T> {
        if in_transaction || self.is_autocommit() {
            return result;
        }

        match result {
            Ok(value) => {
                self.execute("COMMIT;", [])?;
                Ok(value)
            }
            Err(error) => {
                self.execute("ROLLBACK;", [])?;
                Err(error)
            }
        }
    }

    /// Runs an operation in a plain SQL transaction owned by this helper when needed.
    ///
    /// An existing transaction remains owned by its caller. Dolt version-control statements can
    /// seal a transaction opened by this helper themselves, so commit and rollback are
    /// conditional on the connection still being in a transaction after the operation returns.
    pub fn with_transaction<T>(
        &self,
        operation: impl FnOnce() -> rusqlite::Result<T>,
    ) -> rusqlite::Result<T> {
        let in_transaction = self.start_transaction()?;
        let result = operation();
        self.end_transaction(in_transaction, result)
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
    run_config_migrations(&mut conn);
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
        test_helpers::get_connection as test_graph_connection,
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

    #[test]
    fn test_with_transaction_commits_successful_operation() {
        let conn = test_graph_connection(None).expect("should create graph database");
        conn.execute("CREATE TABLE transaction_rows (value TEXT NOT NULL)", [])
            .expect("should create transaction fixture table");

        conn.with_transaction(|| {
            conn.execute(
                "INSERT INTO transaction_rows (value) VALUES ('committed')",
                [],
            )?;
            Ok(())
        })
        .expect("should commit successful transaction operation");

        assert_eq!(
            conn.query_row("SELECT value FROM transaction_rows", [], |row| row
                .get::<_, String>(0))
                .expect("should query committed transaction row"),
            "committed",
            "successful transaction should commit its inserted row"
        );
    }

    #[test]
    fn test_with_transaction_rolls_back_sql_error() {
        let conn = test_graph_connection(None).expect("should create graph database");
        conn.execute("CREATE TABLE transaction_rows (value TEXT NOT NULL)", [])
            .expect("should create transaction fixture table");

        let error = conn.with_transaction(|| {
            conn.execute(
                "INSERT INTO transaction_rows (value) VALUES ('rolled back')",
                [],
            )?;
            conn.execute("INSERT INTO transaction_rows (value) VALUES (NULL)", [])?;
            Ok(())
        });
        let sqlite_error = error.expect_err("constraint violation should fail transaction");
        assert!(
            matches!(
                sqlite_error,
                rusqlite::Error::SqliteFailure(error, _)
                    if error.code == rusqlite::ErrorCode::ConstraintViolation
            ),
            "transaction should return the SQL constraint violation"
        );
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM transaction_rows", [], |row| row
                .get::<_, i64>(0))
                .expect("should query rolled-back transaction rows"),
            0,
            "SQL-error transaction should roll back its valid inserted row"
        );
    }

    #[test]
    fn test_start_and_end_transaction_commits_successful_operation() {
        let conn = test_graph_connection(None).expect("should create graph database");
        conn.execute("CREATE TABLE transaction_rows (value TEXT NOT NULL)", [])
            .expect("should create transaction fixture table");

        let in_transaction = conn
            .start_transaction()
            .expect("should start transaction explicitly");
        assert!(
            !in_transaction,
            "start_transaction should own a newly opened transaction"
        );
        let result = conn
            .execute(
                "INSERT INTO transaction_rows (value) VALUES ('explicit commit')",
                [],
            )
            .map(|_| ());
        conn.end_transaction(in_transaction, result)
            .expect("should commit explicit transaction");

        assert_eq!(
            conn.query_row("SELECT value FROM transaction_rows", [], |row| row
                .get::<_, String>(0))
                .expect("should query explicitly committed row"),
            "explicit commit",
            "end_transaction should commit a successful explicit transaction"
        );
    }

    #[test]
    fn test_end_transaction_rolls_back_explicit_error() {
        let conn = test_graph_connection(None).expect("should create graph database");
        conn.execute("CREATE TABLE transaction_rows (value TEXT NOT NULL)", [])
            .expect("should create transaction fixture table");

        let in_transaction = conn
            .start_transaction()
            .expect("should start transaction explicitly");
        let result: rusqlite::Result<()> = (|| {
            conn.execute(
                "INSERT INTO transaction_rows (value) VALUES ('explicit rollback')",
                [],
            )?;
            Err(rusqlite::Error::InvalidParameterName(
                "expected explicit transaction failure".to_string(),
            ))
        })();
        let error = conn.end_transaction(in_transaction, result);
        assert!(error.is_err(), "explicit failure should return its error");
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM transaction_rows", [], |row| row
                .get::<_, i64>(0))
                .expect("should query explicitly rolled-back rows"),
            0,
            "end_transaction should roll back an explicit transaction error"
        );
    }

    #[test]
    fn test_start_transaction_preserves_outer_transaction() {
        let conn = test_graph_connection(None).expect("should create graph database");
        conn.execute("CREATE TABLE transaction_rows (value TEXT NOT NULL)", [])
            .expect("should create transaction fixture table");
        conn.execute("BEGIN;", [])
            .expect("should begin outer transaction");

        let in_transaction = conn
            .start_transaction()
            .expect("should inspect outer transaction");
        assert!(
            in_transaction,
            "start_transaction should report an existing outer transaction"
        );
        let result = conn
            .execute("INSERT INTO transaction_rows (value) VALUES ('outer')", [])
            .map(|_| ());
        conn.end_transaction(in_transaction, result)
            .expect("should leave outer transaction active");

        assert!(
            !conn.is_autocommit(),
            "end_transaction should leave an outer transaction active"
        );
        conn.execute("ROLLBACK;", [])
            .expect("should roll back outer transaction");
        let row_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM transaction_rows", [], |row| {
                row.get(0)
            })
            .expect("should count transaction rows");
        assert_eq!(
            row_count, 0,
            "rolling back the outer transaction should discard its inserted row"
        );
    }

    #[test]
    fn test_end_transaction_accepts_dolt_commit_sealing_transaction() {
        let conn = test_graph_connection(None).expect("should create graph database");
        Collection::create(&conn, "transaction-commit")
            .expect("should create Dolt transaction fixture row");

        let in_transaction = conn
            .start_transaction()
            .expect("should start transaction explicitly");
        let result = commit_all(&conn, "transaction commit");
        let commit_hash = conn
            .end_transaction(in_transaction, result)
            .expect("should preserve a Dolt commit that seals its transaction");

        assert_eq!(
            conn.query_row("SELECT dolt_hashof('HEAD')", [], |row| row
                .get::<_, gen_core::DoltHashId>(
                0
            ))
            .expect("should query Dolt head after transaction commit"),
            commit_hash,
            "end_transaction should preserve a Dolt commit that sealed its transaction"
        );
    }
}
