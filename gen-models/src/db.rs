use std::{ops::Deref, path::Path, rc::Rc, sync::Arc};

use gen_core::{config::Workspace, errors::ConfigError};
use rusqlite::{Connection, TransactionBehavior, limits::Limit};

use crate::{
    history::dolt::{active_branch, checkout, connect_branch, is_current_branch_dirty},
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

/// Selects write transaction behavior for a graph workspace.
///
/// Direct graph views can have live widget connections reading the same VFS, so writes reserve
/// their transaction before the operation reads graph state. Ordinary disk-backed workspaces keep
/// SQLite's existing deferred transaction behavior.
pub fn graph_write_transaction_behavior(workspace: &Workspace) -> TransactionBehavior {
    if workspace.has_graph_db_path_override() {
        TransactionBehavior::Immediate
    } else {
        TransactionBehavior::Deferred
    }
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
        self.start_transaction_with_behavior(TransactionBehavior::Deferred)
    }

    /// Starts a transaction with the requested behavior unless a caller already owns one.
    ///
    /// # Errors
    ///
    /// Returns a SQLite error if the transaction cannot be started.
    pub fn start_transaction_with_behavior(
        &self,
        behavior: TransactionBehavior,
    ) -> rusqlite::Result<bool> {
        let in_transaction = !self.is_autocommit();
        if !in_transaction {
            let statement = match behavior {
                TransactionBehavior::Deferred => "BEGIN DEFERRED;",
                TransactionBehavior::Immediate => "BEGIN IMMEDIATE;",
                TransactionBehavior::Exclusive => "BEGIN EXCLUSIVE;",
                _ => {
                    return Err(rusqlite::Error::InvalidParameterName(
                        "unsupported graph transaction behavior".to_string(),
                    ));
                }
            };
            self.execute_batch(statement)?;
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
        self.with_transaction_behavior(TransactionBehavior::Deferred, operation)
    }

    /// Runs an operation under the requested transaction behavior unless an outer transaction exists.
    ///
    /// Dolt commit statements may seal the SQL transaction themselves, so this also preserves the
    /// existing conditional commit and rollback behavior used by [`Self::with_transaction`].
    ///
    /// # Errors
    ///
    /// Returns the operation error, a SQLite error starting the transaction, or a commit error.
    pub fn with_transaction_behavior<T, E>(
        &self,
        behavior: TransactionBehavior,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<rusqlite::Error>,
    {
        let in_transaction = self
            .start_transaction_with_behavior(behavior)
            .map_err(E::from)?;
        let result = operation();
        if in_transaction || self.is_autocommit() {
            return result;
        }

        match result {
            Ok(value) => {
                self.execute_batch("COMMIT;").map_err(E::from)?;
                Ok(value)
            }
            Err(error) => {
                self.execute_batch("ROLLBACK;").map_err(E::from)?;
                Err(error)
            }
        }
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
/// A shared owner for a connection that registers a VFS used by secondary graph handles.
///
/// Callers retain this handle only for its lifetime; they must query through a separate graph
/// connection. The mutex makes the owner safe to carry with browser widgets that cross threads.
pub type GraphVfsAnchor = Arc<std::sync::Mutex<Connection>>;

const SAVEPOINT_NAME: &str = "gen_savepoint";
const MISSING_SAVEPOINT: &str = "no such savepoint: gen_savepoint";
const TRANSACTION_SAVEPOINT: &str = "gen_transaction";

fn is_missing_savepoint(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(code, Some(message))
            if code.code == rusqlite::ErrorCode::Unknown
                && message == MISSING_SAVEPOINT
    )
}

/// Provides shared access to a repository's graph and configuration connections.
///
/// [`begin_transaction`](Self::begin_transaction) starts a SQLite transaction when none is open
/// and otherwise creates a nested transaction savepoint. It returns True if the transaction is the
/// outermose transaction in the stack, which is useful for passing to
/// [`commit_transaction`](Self::commit_transaction) or
/// [`rollback_transaction`](Self::rollback_transaction) for simple commits/rollbacks. Nested scopes
/// close in reverse order: an inner commit releases its savepoint into its parent and returns `None`;
/// an inner rollback removes only that scope's changes. The outermost commit returns `None` for a clean
/// working set. For a dirty set, it calls the supplied callback and returns `Some(value)`. That callback
/// typically creates a Dolt history entry; SQLite's `COMMIT` closes the SQL transaction if the
/// callback did not already do so. Errors roll back any still-open outer transaction.
///
/// [`begin_savepoint`](Self::begin_savepoint) can be used inside a transaction or by itself; SQLite
/// treats a standalone savepoint as a transaction. Each call pushes a savepoint with the same name
/// onto SQLite's connection-wide stack; it does not replace an earlier one. Both
/// [`rollback_savepoint`](Self::rollback_savepoint) and [`undo_savepoint`](Self::undo_savepoint)
/// run `ROLLBACK TO` and `RELEASE` on the newest matching savepoint, so repeated calls unwind older
/// checkpoints in order. Undo returns `false` when no matching savepoint remains. Cloned contexts
/// share both graph and configuration connections. The graph and configuration connections point
/// to separate databases, and transaction/savepoint state belongs to the shared graph connection.
#[derive(Clone)]
pub struct DbContext {
    workspace: Arc<Workspace>,
    graph: GraphHandle,
    config: ConfigHandle,
    // The connection that registers a credentialed BlockCacheVFS must outlive every secondary
    // SQLite handle opened through its VFS name. Cloned model contexts share this owner.
    graph_anchor: Option<GraphVfsAnchor>,
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
            graph_anchor: None,
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

    /// Replaces the graph connection after its current SQLite transaction has ended.
    pub fn set_graph(&mut self, graph_conn: GraphConnection) -> rusqlite::Result<()> {
        if !self.graph.conn().is_autocommit() {
            return Err(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_MISUSE),
                Some(
                    "cannot replace graph connection while an SQLite transaction is active".into(),
                ),
            ));
        }
        self.graph = DbHandle::new(self.workspace.clone(), graph_conn.into());
        Ok(())
    }

    /// Keeps the owner of a registered graph VFS alive while secondary graph handles use it.
    ///
    /// Browser CAB connections own the VFS registration and credential refresh callback. A
    /// separate graph connection uses the registered VFS by name, so retaining this anchor across
    /// context clones and graph refreshes prevents the VFS from being torn down mid-session.
    pub fn set_graph_anchor(&mut self, anchor: GraphVfsAnchor) {
        self.graph_anchor = Some(anchor);
    }

    /// Returns a clone of the lifetime guard for a registered graph VFS, if present.
    pub fn graph_anchor(&self) -> Option<GraphVfsAnchor> {
        self.graph_anchor.as_ref().map(Arc::clone)
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

    /// Starts a SQLite transaction or opens a nested transaction savepoint.
    pub fn begin_transaction(&self) -> rusqlite::Result<bool> {
        let graph = self.graph.conn();
        let is_outermost = graph.is_autocommit();
        if is_outermost {
            graph.start_transaction_with_behavior(graph_write_transaction_behavior(
                &self.workspace,
            ))?;
        } else {
            graph.execute_batch(&format!("SAVEPOINT {TRANSACTION_SAVEPOINT}"))?;
        }
        Ok(is_outermost)
    }

    /// Creates a SQLite savepoint.
    pub fn begin_savepoint(&self) -> rusqlite::Result<()> {
        self.graph
            .conn()
            .execute_batch(&format!("SAVEPOINT {SAVEPOINT_NAME}"))
    }

    /// Rolls back and releases the newest matching savepoint.
    pub fn rollback_savepoint(&self) -> rusqlite::Result<()> {
        let graph = self.graph.conn();
        graph.execute_batch(&format!("ROLLBACK TO SAVEPOINT {SAVEPOINT_NAME}"))?;
        graph.execute_batch(&format!("RELEASE SAVEPOINT {SAVEPOINT_NAME}"))?;
        Ok(())
    }

    /// Undoes the newest savepoint on the SQLite connection, if one exists.
    pub fn undo_savepoint(&self) -> rusqlite::Result<bool> {
        let graph = self.graph.conn();
        if let Err(error) = graph.execute_batch(&format!("ROLLBACK TO SAVEPOINT {SAVEPOINT_NAME}"))
        {
            if is_missing_savepoint(&error) {
                return Ok(false);
            }
            return Err(error);
        }
        graph.execute_batch(&format!("RELEASE SAVEPOINT {SAVEPOINT_NAME}"))?;
        Ok(true)
    }

    /// Commits the outermost transaction or releases an inner transaction into its parent.
    pub fn commit_transaction<T>(
        &self,
        is_outermost: bool,
        commit: impl FnOnce() -> rusqlite::Result<T>,
    ) -> rusqlite::Result<Option<T>> {
        let graph = self.graph.conn();
        if !is_outermost {
            graph.execute_batch(&format!("RELEASE SAVEPOINT {TRANSACTION_SAVEPOINT}"))?;
            return Ok(None);
        }
        let is_dirty = match is_current_branch_dirty(graph) {
            Ok(is_dirty) => is_dirty,
            Err(error) => {
                if !graph.is_autocommit() {
                    let _ = graph.execute_batch("ROLLBACK");
                }
                return Err(error);
            }
        };
        if !is_dirty {
            graph.execute_batch("ROLLBACK")?;
            return Ok(None);
        }

        match commit() {
            Ok(value) => {
                if !graph.is_autocommit()
                    && let Err(error) = graph.execute_batch("COMMIT")
                {
                    let _ = graph.execute_batch("ROLLBACK");
                    return Err(error);
                }
                Ok(Some(value))
            }
            Err(error) => {
                if !graph.is_autocommit() {
                    let _ = graph.execute_batch("ROLLBACK");
                }
                Err(error)
            }
        }
    }

    /// Rolls back the outer transaction or the newest nested transaction savepoint.
    pub fn rollback_transaction(&self, is_outermost: bool) -> rusqlite::Result<()> {
        let graph = self.graph.conn();
        if is_outermost {
            graph.execute_batch("ROLLBACK")?;
        } else {
            graph.execute_batch(&format!("ROLLBACK TO SAVEPOINT {TRANSACTION_SAVEPOINT}"))?;
            graph.execute_batch(&format!("RELEASE SAVEPOINT {TRANSACTION_SAVEPOINT}"))?;
        }
        Ok(())
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
    use std::{
        ffi::{c_char, c_int, c_void},
        sync::{Arc, Mutex},
    };

    use gen_core::config::Workspace;
    use rusqlite::Connection;
    use tempfile::tempdir;

    use super::{DbContext, get_config_connection, get_connection};
    use crate::{
        collection::Collection,
        history::dolt::{active_branch, commit_all, connect_branch, create_branch},
        operations::Defaults,
        sample::{NewSample, Sample},
        test_helpers::get_connection as test_graph_connection,
    };

    // This is here to manufacture some errors for testing conditions
    unsafe extern "C" fn deny_savepoint_actions(
        _user_data: *mut c_void,
        action: c_int,
        _first_argument: *const c_char,
        _second_argument: *const c_char,
        _database: *const c_char,
        _trigger: *const c_char,
    ) -> c_int {
        if action == rusqlite::ffi::SQLITE_SAVEPOINT {
            rusqlite::ffi::SQLITE_DENY
        } else {
            rusqlite::ffi::SQLITE_OK
        }
    }

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

    fn setup_branch_restore_repo() -> (Workspace, tempfile::TempDir) {
        let temp_dir = tempdir().expect("should create temp repo");
        let workspace = Workspace::new(temp_dir.path());
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

        (workspace, temp_dir)
    }

    #[test]
    fn test_graph_write_transaction_behavior_preserves_workspace_defaults() {
        assert!(matches!(
            super::graph_write_transaction_behavior(&Workspace::new("local-repository")),
            rusqlite::TransactionBehavior::Deferred
        ));

        let direct_workspace = Workspace::new("direct-repository")
            .with_graph_db_path_override("file:///owned/default.db?vfs=cloud_graph");
        assert!(matches!(
            super::graph_write_transaction_behavior(&direct_workspace),
            rusqlite::TransactionBehavior::Immediate
        ));
    }

    #[test]
    fn test_new_restores_saved_current_branch() {
        let (workspace, temp_dir) = setup_branch_restore_repo();
        let base_dir = workspace.base_dir().to_path_buf();
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
        drop(context);
        drop(temp_dir);
        assert!(
            !base_dir.exists(),
            "workspace should be removed after context drops"
        );
    }

    #[test]
    fn test_new_with_ref_overrides_saved_current_branch() {
        let (workspace, temp_dir) = setup_branch_restore_repo();
        let base_dir = workspace.base_dir().to_path_buf();
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
        drop(context);
        drop(temp_dir);
        assert!(
            !base_dir.exists(),
            "workspace should be removed after context drops"
        );
    }

    #[test]
    fn test_graph_anchor_lives_until_the_last_context_clone_drops() {
        let (workspace, _temp_dir) = setup_branch_restore_repo();
        let graph_path = workspace
            .graph_db_path()
            .expect("should resolve graph database path");
        let config_path = workspace
            .gen_db_path()
            .expect("should resolve config database path");
        let graph_conn = get_connection(&graph_path).expect("should open graph connection");
        let config_conn =
            get_config_connection(&config_path).expect("should open config connection");
        let mut context = DbContext::new(workspace, graph_conn, config_conn)
            .expect("should open context on the saved branch");
        let anchor = Arc::new(Mutex::new(
            Connection::open_in_memory().expect("should create anchor connection"),
        ));
        let weak_anchor = Arc::downgrade(&anchor);
        context.set_graph_anchor(Arc::clone(&anchor));
        let context_clone = context.clone();

        drop(anchor);
        drop(context);
        assert!(
            weak_anchor.upgrade().is_some(),
            "a cloned context should keep the VFS owner alive after its source drops"
        );

        drop(context_clone);
        assert!(
            weak_anchor.upgrade().is_none(),
            "the VFS owner should release after the last context drops"
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
    fn test_set_graph_rejects_replacement_during_transaction() {
        let temp_dir = tempdir().expect("should create temporary workspace");
        let workspace = Workspace::new(temp_dir.path());
        let gen_dir = workspace.ensure_gen_dir();
        let graph_path = workspace
            .graph_db_path()
            .expect("should resolve graph database path");
        let config_path = workspace
            .gen_db_path()
            .expect("should resolve config database path");
        let graph_conn = get_connection(&graph_path).expect("should open graph database");
        let config_conn = get_config_connection(&config_path).expect("should open config database");
        let mut context = DbContext::new_raw(workspace, graph_conn, config_conn);

        let replacement_path = gen_dir.join("replacement.db");
        let replacement =
            get_connection(&replacement_path).expect("should open replacement graph database");
        Collection::create(&replacement, "replacement-collection")
            .expect("should create replacement database collection");
        context
            .set_graph(replacement)
            .expect("should replace graph connection outside a transaction");
        assert!(
            Collection::all(context.graph().conn())
                .expect("should query replacement graph")
                .iter()
                .any(|collection| collection.name == "replacement-collection"),
            "inactive graph connection should be replaced"
        );

        assert!(
            context
                .begin_transaction()
                .expect("should begin transaction"),
            "transaction should be outermost"
        );
        Collection::create(context.graph().conn(), "pending-collection")
            .expect("should create pending collection");
        let blocked_replacement = get_connection(gen_dir.join("blocked.db"))
            .expect("should open blocked replacement database");
        let error = context
            .set_graph(blocked_replacement)
            .expect_err("should refuse to detach an active transaction");
        assert!(
            matches!(
                &error,
                rusqlite::Error::SqliteFailure(code, Some(message))
                    if code.code == rusqlite::ErrorCode::ApiMisuse
                        && message == "cannot replace graph connection while an SQLite transaction is active"
            ),
            "connection replacement should return a clear SQLite misuse error: {error}"
        );
        assert!(
            Collection::all(context.graph().conn())
                .expect("should query active graph connection")
                .iter()
                .any(|collection| collection.name == "pending-collection"),
            "rejected replacement should leave pending transaction data on its connection"
        );
        context
            .rollback_transaction(true)
            .expect("should roll back original transaction");
        assert!(
            !Collection::all(context.graph().conn())
                .expect("should query graph after rollback")
                .iter()
                .any(|collection| collection.name == "pending-collection"),
            "rollback should still remove the pending transaction data"
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

    #[test]
    fn test_undo_savepoint_uses_nested_sqlite_savepoints() {
        let (workspace, _temp_dir) = setup_branch_restore_repo();
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
            .expect("should create repository context");
        let graph_conn = context.graph().conn();
        let is_outermost = context
            .begin_transaction()
            .expect("should begin transaction");
        assert!(is_outermost, "the test transaction should be outermost");

        assert!(
            !context
                .undo_savepoint()
                .expect("should report an empty undo stack from SQLite"),
            "an absent savepoint should report nothing to undo"
        );

        context
            .begin_savepoint()
            .expect("should begin first edit savepoint");
        Collection::create(graph_conn, "first-pending-collection")
            .expect("should write first pending collection");
        context
            .begin_savepoint()
            .expect("should begin a nested edit savepoint with the same name");
        Collection::create(graph_conn, "second-pending-collection")
            .expect("should write second pending collection");

        assert!(
            context
                .undo_savepoint()
                .expect("should undo the newest nested savepoint")
        );
        let collections = Collection::all(graph_conn).expect("should read collections");
        assert!(
            collections
                .iter()
                .any(|collection| collection.name == "first-pending-collection"),
            "undo should retain the earlier edit"
        );
        assert!(
            collections
                .iter()
                .all(|collection| collection.name != "second-pending-collection"),
            "undo should remove only the newest edit"
        );

        assert!(
            context
                .undo_savepoint()
                .expect("should undo the remaining savepoint")
        );
        assert!(
            !context
                .undo_savepoint()
                .expect("should report no remaining SQLite savepoint"),
            "undo should stop when SQLite reports the fixed savepoint is absent"
        );
        let collections =
            Collection::all(graph_conn).expect("should read collections after second undo");
        assert!(
            collections
                .iter()
                .all(|collection| collection.name != "first-pending-collection"),
            "the second undo should remove the first collection before transaction rollback"
        );
        context
            .rollback_transaction(is_outermost)
            .expect("should roll back transaction");
    }

    #[test]
    fn test_undo_savepoint_propagates_unrelated_sqlite_errors() {
        let (workspace, _temp_dir) = setup_branch_restore_repo();
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
            .expect("should create repository context");
        let graph_conn = context.graph().conn();
        let is_outermost = context
            .begin_transaction()
            .expect("should begin transaction");
        assert!(is_outermost, "the test transaction should be outermost");
        context
            .begin_savepoint()
            .expect("should begin edit savepoint");

        let authorizer_result = unsafe {
            rusqlite::ffi::sqlite3_set_authorizer(
                graph_conn.handle(),
                Some(deny_savepoint_actions),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            authorizer_result,
            rusqlite::ffi::SQLITE_OK,
            "the authorizer should be installed"
        );
        let error = context
            .undo_savepoint()
            .expect_err("should propagate an unrelated SQLite authorization failure");
        let authorizer_result = unsafe {
            rusqlite::ffi::sqlite3_set_authorizer(graph_conn.handle(), None, std::ptr::null_mut())
        };
        assert_eq!(
            authorizer_result,
            rusqlite::ffi::SQLITE_OK,
            "the authorizer should be removed"
        );
        assert!(
            error.to_string().contains("not authorized"),
            "undo should return the unrelated SQLite error: {error}"
        );

        context
            .rollback_transaction(is_outermost)
            .expect("should roll back transaction");
    }

    #[test]
    fn test_failed_transaction_commit_rolls_back_and_closes_transaction() {
        let (workspace, _temp_dir) = setup_branch_restore_repo();
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
            .expect("should create repository context");
        let graph_conn = context.graph().conn();
        let head_before = graph_conn
            .query_row("SELECT dolt_hashof('HEAD')", [], |row| {
                row.get::<_, gen_core::DoltHashId>(0)
            })
            .expect("should read current HEAD");
        let is_outermost = context
            .begin_transaction()
            .expect("should begin transaction");
        assert!(is_outermost, "the test transaction should be outermost");
        context
            .begin_savepoint()
            .expect("should begin edit savepoint");
        Collection::create(graph_conn, "pending-collection")
            .expect("should write pending graph data");

        let error = context
            .commit_transaction(is_outermost, || {
                Err::<(), _>(rusqlite::Error::InvalidParameterName(
                    "expected commit callback failure".to_string(),
                ))
            })
            .expect_err("should return a commit callback failure");
        assert!(
            error
                .to_string()
                .contains("expected commit callback failure"),
            "the original commit error should be returned: {error}"
        );
        assert!(
            graph_conn.is_autocommit(),
            "failed finalization should leave no open transaction"
        );
        assert!(
            Collection::all(graph_conn)
                .expect("should read rolled-back collections")
                .iter()
                .all(|collection| collection.name != "pending-collection"),
            "failed finalization should roll back staged graph data"
        );
        assert_eq!(
            graph_conn
                .query_row("SELECT dolt_hashof('HEAD')", [], |row| {
                    row.get::<_, gen_core::DoltHashId>(0)
                })
                .expect("should read HEAD after failed finalization"),
            head_before,
            "failed finalization should preserve history HEAD"
        );
        let fresh_is_outermost = context
            .begin_transaction()
            .expect("should start a fresh transaction after failed commit cleanup");
        context
            .rollback_transaction(fresh_is_outermost)
            .expect("should roll back the fresh transaction");
    }
}
