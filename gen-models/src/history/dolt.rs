//! Dolt-backed history access for Gen's graph database.
//!
//! Gen stores its biological graph tables in a DoltLite-backed `GraphConnection`, so commits,
//! branches, status, merges, and remotes are database operations rather than a separate metadata
//! store. This module is the central adapter between the rest of the codebase and Dolt's SQL
//! functions and system tables. It exposes the raw Dolt row shapes needed by branch and remote
//! workflows and implements the backend-neutral [`HistoryStore`] interface used by CLI commands,
//! diff construction, patch handling, imports, and updates.
//!
//! # Why operations have two call styles
//!
//! The public free functions are connection-oriented building blocks. Code that already has a
//! `GraphConnection`—especially model/database setup and remote-transfer code—can call a specific
//! Dolt operation directly, including operations that do not belong in the common [`HistoryStore`]
//! interface.
//!
//! [`DoltHistoryStore`] is the bound form. It borrows the graph connection, optionally borrows Gen's
//! config database for commit-author defaults, and implements [`HistoryStore`] so higher-level code
//! can work through one cohesive history API. Where both forms expose the same operation, the bound
//! method delegates to the free function; the two forms are entry points for different callers, not
//! separate implementations of Dolt behavior.

use std::rc::Rc;

use gen_core::{BranchName, CommitRef, DoltHashId};
use rusqlite::{OptionalExtension, Result as SqlResult, params, types::Value};

use crate::{
    db::{ConfigConnection, GraphConnection},
    history::{
        HistoryConflict, HistoryEntry, HistoryError, HistoryResult, HistoryStatus, HistoryStore,
    },
    operations::{DEFAULT_COMMITTER_EMAIL, Defaults, GEN_DEFAULT_COMMITTER_NAME},
};

/// Raw commit metadata read from `dolt_log` before conversion to [`HistoryEntry`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltLogEntry {
    pub commit_hash: DoltHashId,
    pub parent_hash: Option<DoltHashId>,
    pub committer: String,
    pub email: String,
    pub date: String,
    pub message: String,
}

/// Raw table status returned to connection-oriented callers and converted to [`HistoryStatus`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltStatusRow {
    pub table_name: String,
    pub staged: bool,
    pub status: String,
}

/// Branch metadata used by branch selection, checkout, and remote-tracking workflows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltBranchRow {
    pub name: String,
    pub hash: DoltHashId,
    pub latest_committer: String,
    pub latest_committer_email: String,
    pub latest_commit_date: String,
    pub latest_commit_message: String,
    pub remote: String,
    pub branch: String,
    pub dirty: bool,
}

/// Remote configuration read from `dolt_remotes` for remote-transfer workflows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltRemoteRow {
    pub name: String,
    pub url: String,
    pub fetch_specs: String,
    pub params: String,
}

/// Executes a Dolt SQL function whose scalar result only signals whether the command succeeded.
fn run_history_statement(
    conn: &GraphConnection,
    query: &str,
    params: &[&dyn rusqlite::ToSql],
) -> SqlResult<()> {
    conn.query_row(query, params, |row| row.get_ref(0).map(|_| ()))?;
    Ok(())
}

/// A borrowed, backend-neutral history facade over a Dolt-backed graph connection.
///
/// The optional config connection is used only when an operation needs Gen-level defaults, such as
/// choosing the commit author. Callers that only need graph history can construct the store without
/// it.
pub struct DoltHistoryStore<'connection> {
    graph: &'connection GraphConnection,
    config: Option<&'connection ConfigConnection>,
}

impl<'connection> DoltHistoryStore<'connection> {
    /// Binds history operations to a graph connection without Gen configuration defaults.
    pub fn new(graph: &'connection GraphConnection) -> Self {
        Self {
            graph,
            config: None,
        }
    }

    /// Binds history operations to both the graph and Gen config connections.
    ///
    /// Use this form for commits that should inherit the configured default committer identity.
    pub fn new_with_config(
        graph: &'connection GraphConnection,
        config: &'connection ConfigConnection,
    ) -> Self {
        Self {
            graph,
            config: Some(config),
        }
    }
}

// These connection-oriented functions are also used by `DoltHistoryStore` below. Keeping the SQL
// operations here lets direct database callers and `HistoryStore` callers share the same behavior.
pub fn commit_author_name(conn: &GraphConnection) -> SqlResult<Option<String>> {
    conn.query_row("SELECT dolt_config('user.name')", [], |row| row.get(0))
}

pub fn commit_author_email(conn: &GraphConnection) -> SqlResult<Option<String>> {
    conn.query_row("SELECT dolt_config('user.email')", [], |row| row.get(0))
}

pub fn set_commit_author_name(conn: &GraphConnection, name: &str) -> SqlResult<()> {
    conn.query_row("SELECT dolt_config('user.name', ?1)", [name], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(())
}

pub fn set_commit_author_email(conn: &GraphConnection, email: &str) -> SqlResult<()> {
    conn.query_row("SELECT dolt_config('user.email', ?1)", [email], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(())
}

fn default_committer_name(config_conn: Option<&ConfigConnection>) -> String {
    config_conn
        .map(Defaults::get_default_committer_name)
        .unwrap_or_else(|| GEN_DEFAULT_COMMITTER_NAME.to_string())
}

fn default_committer_email(config_conn: Option<&ConfigConnection>) -> String {
    config_conn
        .map(Defaults::get_default_committer_email)
        .unwrap_or_else(|| DEFAULT_COMMITTER_EMAIL.to_string())
}

fn should_set_committer_name(name: Option<&str>, email: Option<&str>) -> bool {
    name.is_none()
        || name == Some("")
        || name == Some(GEN_DEFAULT_COMMITTER_NAME)
        || email.is_none()
        || email == Some("")
}

fn should_set_committer_email(email: Option<&str>) -> bool {
    email.is_none() || email == Some("")
}

pub fn set_default_commit_author(
    conn: &GraphConnection,
    config_conn: Option<&ConfigConnection>,
) -> SqlResult<()> {
    let committer_name = commit_author_name(conn)?;
    let committer_email = commit_author_email(conn)?;
    if should_set_committer_name(committer_name.as_deref(), committer_email.as_deref()) {
        set_commit_author_name(conn, &default_committer_name(config_conn))?;
    }
    if should_set_committer_email(committer_email.as_deref()) {
        set_commit_author_email(conn, &default_committer_email(config_conn))?;
    }
    Ok(())
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, message)))]
pub fn commit_all(conn: &GraphConnection, message: &str) -> SqlResult<DoltHashId> {
    commit_all_with_config(conn, None, message)
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip(conn, config_conn, message))
)]
pub fn commit_all_with_config(
    conn: &GraphConnection,
    config_conn: Option<&ConfigConnection>,
    message: &str,
) -> SqlResult<DoltHashId> {
    set_default_commit_author(conn, config_conn)?;
    commit_staged_all(conn, message)
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, message)))]
pub fn commit_staged_all(conn: &GraphConnection, message: &str) -> SqlResult<DoltHashId> {
    conn.query_row("SELECT dolt_commit('-A', '-m', ?1)", [message], |row| {
        row.get(0)
    })
}

pub fn active_branch(conn: &GraphConnection) -> SqlResult<String> {
    conn.query_row("SELECT active_branch()", [], |row| row.get(0))
}

pub fn log_entries(conn: &GraphConnection) -> SqlResult<Vec<DoltLogEntry>> {
    let mut statement = conn.prepare(
        "SELECT commits.commit_hash, ancestors.parent_hash, commits.committer, \
                commits.email, commits.date, commits.message \
         FROM dolt_log AS commits \
         LEFT JOIN dolt_commit_ancestors AS ancestors \
           ON ancestors.commit_hash = commits.commit_hash AND ancestors.parent_index = 0",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(DoltLogEntry {
            commit_hash: row.get(0)?,
            parent_hash: row.get(1)?,
            committer: row.get(2)?,
            email: row.get(3)?,
            date: row.get(4)?,
            message: row.get(5)?,
        })
    })?;
    rows.collect()
}

/// Loads metadata for existing commit hashes in the requested order.
///
/// Dolt omits hashes that do not identify commits. Callers can compare the returned hashes with
/// their request to report the missing commit while still allowing commits from unrelated branches.
///
/// # Arguments
///
/// * `conn` - The graph database containing the Dolt repository.
/// * `commit_hashes` - The commit hashes whose metadata should be loaded.
///
/// # Errors
///
/// Returns an error if Dolt commit metadata cannot be queried.
pub fn log_entries_for_hashes(
    conn: &GraphConnection,
    commit_hashes: &[DoltHashId],
) -> SqlResult<Vec<DoltLogEntry>> {
    if commit_hashes.is_empty() {
        return Ok(Vec::new());
    }
    let commit_hashes = Rc::new(
        commit_hashes
            .iter()
            .map(|commit_hash| Value::Text(commit_hash.to_string()))
            .collect::<Vec<_>>(),
    );
    let mut statement = conn.prepare(
        "WITH requested AS ( \
             SELECT rowid AS position, value AS commit_hash FROM rarray(?1) \
         ) \
         SELECT commits.commit_hash, ancestors.parent_hash, commits.committer, \
                commits.email, commits.date, commits.message \
         FROM requested \
         JOIN dolt_log AS commits ON commits.commit_hash = requested.commit_hash \
         LEFT JOIN dolt_commit_ancestors AS ancestors \
           ON ancestors.commit_hash = requested.commit_hash AND ancestors.parent_index = 0 \
         ORDER BY requested.position",
    )?;
    let rows = statement.query_map([commit_hashes], |row| {
        Ok(DoltLogEntry {
            commit_hash: row.get(0)?,
            parent_hash: row.get(1)?,
            committer: row.get(2)?,
            email: row.get(3)?,
            date: row.get(4)?,
            message: row.get(5)?,
        })
    })?;
    rows.collect()
}

/// Loads the commit log selected by a Dolt revision or revision range.
///
/// # Arguments
///
/// * `conn` - The graph database containing the Dolt repository.
/// * `revision` - A Dolt ref or range such as `feature`, `main..feature`, or `main...feature`.
///
/// # Errors
///
/// Returns an error if Dolt cannot resolve or query the revision.
pub fn log_entries_for_revision(
    conn: &GraphConnection,
    revision: &str,
) -> SqlResult<Vec<DoltLogEntry>> {
    let mut statement = conn.prepare(
        "SELECT commits.commit_hash, first_parents.parent_hash, commits.committer, \
                commits.email, commits.date, commits.message \
         FROM dolt_log(?1) AS commits \
         LEFT JOIN dolt_commit_ancestors AS first_parents \
           ON first_parents.commit_hash = commits.commit_hash \
          AND first_parents.parent_index = 0",
    )?;
    let rows = statement.query_map([revision], |row| {
        Ok(DoltLogEntry {
            commit_hash: row.get(0)?,
            parent_hash: row.get(1)?,
            committer: row.get(2)?,
            email: row.get(3)?,
            date: row.get(4)?,
            message: row.get(5)?,
        })
    })?;
    rows.collect()
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, head_hash)))]
fn head_log_entries(
    conn: &GraphConnection,
    head_hash: &DoltHashId,
    limit: Option<usize>,
) -> SqlResult<Vec<DoltLogEntry>> {
    let commit_hashes: Vec<DoltHashId> = match limit {
        Some(limit) => {
            let max_depth = i64::try_from(limit).map_err(|_| {
                rusqlite::Error::ToSqlConversionFailure(
                    format!("history limit {limit} exceeds i64").into(),
                )
            })?;
            let mut statement = conn.prepare(
                "WITH RECURSIVE ancestry(commit_hash, depth) AS ( \
                     SELECT ?1, 0 \
                     UNION ALL \
                     SELECT ancestors.parent_hash, ancestry.depth + 1 \
                     FROM ancestry \
                     JOIN dolt_commit_ancestors AS ancestors ON ancestors.commit_hash = ancestry.commit_hash \
                     WHERE ancestors.parent_hash IS NOT NULL AND ancestry.depth + 1 < ?2 \
                 ) \
                 SELECT commit_hash \
                 FROM ancestry \
                 GROUP BY commit_hash \
                 ORDER BY MIN(depth) \
                 LIMIT ?2",
            )?;
            let rows = statement.query_map(params![head_hash, max_depth], |row| row.get(0))?;
            rows.collect::<SqlResult<Vec<_>>>()
        }
        None => {
            let mut statement = conn.prepare(
                "WITH RECURSIVE ancestry(commit_hash, depth) AS ( \
                     SELECT ?1, 0 \
                     UNION ALL \
                     SELECT ancestors.parent_hash, ancestry.depth + 1 \
                     FROM ancestry \
                     JOIN dolt_commit_ancestors AS ancestors ON ancestors.commit_hash = ancestry.commit_hash \
                     WHERE ancestors.parent_hash IS NOT NULL \
                 ) \
                 SELECT commit_hash \
                 FROM ancestry \
                 GROUP BY commit_hash \
                 ORDER BY MIN(depth)",
            )?;
            let rows = statement.query_map([head_hash], |row| row.get(0))?;
            rows.collect::<SqlResult<Vec<_>>>()
        }
    }?;

    log_entries_for_hashes(conn, &commit_hashes)
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, head_hash)))]
fn nth_ancestor_hash(
    conn: &GraphConnection,
    head_hash: &DoltHashId,
    offset: usize,
) -> SqlResult<Option<DoltHashId>> {
    let offset = i64::try_from(offset).map_err(|_| {
        rusqlite::Error::ToSqlConversionFailure(
            format!("ancestor offset {offset} exceeds i64").into(),
        )
    })?;
    conn.query_row(
        "WITH RECURSIVE ancestry(commit_hash, depth) AS ( \
             SELECT ?1, 0 \
             UNION ALL \
             SELECT ancestors.parent_hash, ancestry.depth + 1 \
             FROM ancestry \
             JOIN dolt_commit_ancestors AS ancestors ON ancestors.commit_hash = ancestry.commit_hash \
             WHERE ancestors.parent_hash IS NOT NULL \
               AND ancestors.parent_index = 0 \
               AND ancestry.depth < ?2 \
         ) \
         SELECT commit_hash \
         FROM ancestry \
         WHERE depth = ?2 \
         LIMIT 1",
        params![head_hash, offset],
        |row| row.get(0),
    )
    .optional()
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn)))]
pub fn status_rows(conn: &GraphConnection) -> SqlResult<Vec<DoltStatusRow>> {
    let mut statement = conn.prepare(
        "SELECT table_name, staged, status FROM dolt_status ORDER BY table_name, staged DESC",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(DoltStatusRow {
            table_name: row.get(0)?,
            staged: row.get(1)?,
            status: row.get(2)?,
        })
    })?;
    rows.collect()
}

pub fn create_branch(conn: &GraphConnection, branch_name: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_branch(?1)", &[&branch_name])
}

pub fn checkout(conn: &GraphConnection, target: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_checkout(?1)", &[&target])
}

pub fn connect_branch(conn: &GraphConnection, branch_name: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_connect_branch(?1)", &[&branch_name])
}

pub fn merge(conn: &GraphConnection, branch_name: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_merge(?1)", &[&branch_name])
}

pub fn reset_hard(conn: &GraphConnection, target: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_reset('--hard', ?1)", &[&target])
}

pub fn add_remote(conn: &GraphConnection, remote_name: &str, remote_url: &str) -> SqlResult<()> {
    run_history_statement(
        conn,
        "SELECT dolt_remote('add', ?1, ?2)",
        &[&remote_name, &remote_url],
    )
}

pub fn remove_remote(conn: &GraphConnection, remote_name: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_remote('remove', ?1)", &[&remote_name])
}

/// This command is specific to the libdoltlite-sys library. It allows for an in-place update of remote urls
/// for baking in short lived credentials and allowing use with the standard dolt_push/pull/clone commands.
pub fn set_remote_url(
    conn: &GraphConnection,
    remote_name: &str,
    remote_url: &str,
) -> SqlResult<()> {
    run_history_statement(
        conn,
        "SELECT dolt_remote('set-url', ?1, ?2)",
        &[&remote_name, &remote_url],
    )
}

pub fn clone_remote(conn: &GraphConnection, remote_url: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_clone(?1)", &[&remote_url])
}

pub fn branch_rows(conn: &GraphConnection) -> SqlResult<Vec<DoltBranchRow>> {
    // DoltLite exposes an all-zero hash for an uninitialized branch, but its
    // latest_commit_* columns currently try to load that hash as a commit.
    // CASE is lazy in DoltLite and keeps those placeholder rows readable.
    let mut statement = conn.prepare(
        "SELECT name, hash, \
         CASE WHEN hash = '0000000000000000000000000000000000000000' \
              THEN '' ELSE latest_committer END, \
         CASE WHEN hash = '0000000000000000000000000000000000000000' \
              THEN '' ELSE latest_committer_email END, \
         CASE WHEN hash = '0000000000000000000000000000000000000000' \
              THEN '' ELSE latest_commit_date END, \
         CASE WHEN hash = '0000000000000000000000000000000000000000' \
              THEN '' ELSE latest_commit_message END, \
         remote, branch, dirty \
         FROM dolt_branches ORDER BY name",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(DoltBranchRow {
            name: row.get(0)?,
            hash: row.get(1)?,
            latest_committer: row.get(2)?,
            latest_committer_email: row.get(3)?,
            latest_commit_date: row.get(4)?,
            latest_commit_message: row.get(5)?,
            remote: row.get(6)?,
            branch: row.get(7)?,
            dirty: row.get(8)?,
        })
    })?;
    rows.collect()
}

/// Searches materialized branch names, ranking exact and prefix matches first.
pub fn search_branch_names(conn: &GraphConnection, search_term: &str) -> SqlResult<Vec<String>> {
    let zero_hash = "0000000000000000000000000000000000000000";
    let mut statement = conn.prepare(
        "SELECT name FROM dolt_branches \
         WHERE hash <> ?1 AND LOWER(name) LIKE '%' || LOWER(?2) || '%' \
         ORDER BY CASE \
           WHEN LOWER(name) = LOWER(?2) THEN 0 \
           WHEN LOWER(name) LIKE LOWER(?2) || '%' THEN 1 \
           ELSE 2 \
         END, LOWER(name), name",
    )?;
    let rows = statement.query_map([zero_hash, search_term], |row| row.get(0))?;
    rows.collect()
}

pub fn branch_exists(conn: &GraphConnection, branch_name: &str) -> SqlResult<bool> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM dolt_branches WHERE name = ?1)",
        [branch_name],
        |row| row.get(0),
    )
}

pub fn branch_hash(conn: &GraphConnection, branch_name: &str) -> SqlResult<Option<DoltHashId>> {
    conn.query_row(
        "SELECT hash FROM dolt_branches WHERE name = ?1",
        [branch_name],
        |row| row.get(0),
    )
    .optional()
}

pub fn hash_of(conn: &GraphConnection, commit_ref: &str) -> SqlResult<DoltHashId> {
    conn.query_row("SELECT dolt_hashof(?1)", [commit_ref], |row| row.get(0))
}

pub fn commit_exists(conn: &GraphConnection, commit_hash: &DoltHashId) -> SqlResult<bool> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM dolt_log WHERE commit_hash = ?1)",
        [commit_hash],
        |row| row.get(0),
    )
}

pub fn merge_base(
    conn: &GraphConnection,
    source_ref: &str,
    target_ref: &str,
) -> SqlResult<DoltHashId> {
    conn.query_row(
        "SELECT dolt_merge_base(?1, ?2)",
        [source_ref, target_ref],
        |row| row.get(0),
    )
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn)))]
pub fn is_current_branch_dirty(conn: &GraphConnection) -> SqlResult<bool> {
    conn.query_row(
        "SELECT dirty FROM dolt_branches WHERE name = active_branch() LIMIT 1",
        [],
        |row| row.get::<_, bool>(0),
    )
}

pub fn delete_branch(conn: &GraphConnection, branch_name: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_branch('-d', ?1)", &[&branch_name])
}

pub fn delete_branch_force(conn: &GraphConnection, branch_name: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_branch('-D', ?1)", &[&branch_name])
}

pub fn cherry_pick(conn: &GraphConnection, commit_hash: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_cherry_pick(?1)", &[&commit_hash])
}

pub fn push(conn: &GraphConnection, remote_name: &str, branch_name: &str) -> SqlResult<()> {
    run_history_statement(
        conn,
        "SELECT dolt_push(?1, ?2)",
        &[&remote_name, &branch_name],
    )
}

pub fn push_force(conn: &GraphConnection, remote_name: &str, branch_name: &str) -> SqlResult<()> {
    run_history_statement(
        conn,
        "SELECT dolt_push(?1, ?2, '--force')",
        &[&remote_name, &branch_name],
    )
}

pub fn pull(conn: &GraphConnection, remote_name: &str, branch_name: &str) -> SqlResult<()> {
    run_history_statement(
        conn,
        "SELECT dolt_pull(?1, ?2)",
        &[&remote_name, &branch_name],
    )
}

pub fn fetch(
    conn: &GraphConnection,
    remote_name: &str,
    branch_name: Option<&str>,
) -> SqlResult<()> {
    match branch_name {
        Some(branch_name) => run_history_statement(
            conn,
            "SELECT dolt_fetch(?1, ?2)",
            &[&remote_name, &branch_name],
        ),
        None => run_history_statement(conn, "SELECT dolt_fetch(?1)", &[&remote_name]),
    }
}

pub fn remote_rows(conn: &GraphConnection) -> SqlResult<Vec<DoltRemoteRow>> {
    let mut statement =
        conn.prepare("SELECT name, url, fetch_specs, params FROM dolt_remotes ORDER BY name")?;
    let rows = statement.query_map([], |row| {
        Ok(DoltRemoteRow {
            name: row.get(0)?,
            url: row.get(1)?,
            fetch_specs: row.get(2)?,
            params: row.get(3)?,
        })
    })?;
    rows.collect()
}

pub fn diff_row_count(
    conn: &GraphConnection,
    table_name: &str,
    from_ref: &str,
    to_ref: &str,
) -> SqlResult<i64> {
    if !table_name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return Err(rusqlite::Error::InvalidParameterName(
            table_name.to_string(),
        ));
    }

    let query = format!("SELECT COUNT(*) FROM dolt_diff_{table_name}(?1, ?2)");
    conn.query_row(&query, params![from_ref, to_ref], |row| row.get(0))
}

pub fn conflict_rows(conn: &GraphConnection) -> SqlResult<Vec<HistoryConflict>> {
    let mut statement =
        conn.prepare("SELECT \"table\", num_conflicts FROM dolt_conflicts ORDER BY \"table\"")?;
    let rows = statement.query_map([], |row| {
        Ok(HistoryConflict {
            table_name: row.get(0)?,
            num_conflicts: row.get(1)?,
        })
    })?;
    rows.collect()
}

// This is the self-bound entry point used by code written against `HistoryStore`. Methods delegate
// to the connection-oriented functions above when an equivalent function exists.
impl HistoryStore for DoltHistoryStore<'_> {
    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self)))]
    fn current_head(&self) -> SqlResult<Option<DoltHashId>> {
        self.graph
            .query_row(
                "SELECT hash FROM dolt_branches WHERE name = active_branch()",
                [],
                |row| row.get(0),
            )
            .optional()
    }

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self)))]
    fn current_branch(&self) -> SqlResult<Option<BranchName>> {
        self.graph
            .query_row("SELECT active_branch()", [], |row| row.get::<_, String>(0))
            .optional()
            .map(|branch_name| branch_name.map(BranchName))
    }

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self, reference)))]
    fn resolve_ref(&self, reference: &CommitRef) -> HistoryResult<Option<DoltHashId>> {
        if reference.0 == "HEAD" {
            return self.current_head().map_err(HistoryError::from);
        }

        if let Some((base_reference, offset)) = reference.0.rsplit_once('~') {
            let offset = match offset.parse::<usize>() {
                Ok(offset) => offset,
                Err(_) => return Ok(None),
            };
            let Some(head_hash) = self.resolve_ref(&CommitRef(base_reference.to_string()))? else {
                return Ok(None);
            };
            return nth_ancestor_hash(self.graph, &head_hash, offset).map_err(HistoryError::from);
        }

        let branch_hash = self
            .graph
            .query_row(
                "SELECT hash FROM dolt_branches WHERE name = ?1",
                [&reference.0],
                |row| row.get(0),
            )
            .optional()
            .map_err(HistoryError::from)?;
        if branch_hash.is_some() {
            return Ok(branch_hash);
        }

        let prefix_matches = log_entries(self.graph)
            .map_err(HistoryError::from)?
            .into_iter()
            .filter(|entry| entry.commit_hash.to_string().starts_with(&reference.0))
            .map(|entry| entry.commit_hash)
            .collect::<Vec<_>>();
        if prefix_matches.len() == 1 {
            return Ok(prefix_matches.into_iter().next());
        }
        if prefix_matches.len() > 1 {
            return Err(HistoryError::AmbiguousReference(reference.0.clone()));
        }

        if let Ok(commit_hash) = DoltHashId::try_from(reference.0.as_str()) {
            return Ok(Some(commit_hash));
        }

        Ok(None)
    }

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self, reference)))]
    fn resolve_operation_hash(&self, reference: &CommitRef) -> HistoryResult<DoltHashId> {
        self.resolve_ref(reference)?
            .ok_or_else(|| HistoryError::UnresolvedReference(reference.0.clone()))
    }

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self, reference)))]
    fn log_for_ref(
        &self,
        reference: &CommitRef,
        limit: Option<usize>,
    ) -> HistoryResult<Vec<HistoryEntry>> {
        let head_hash = self
            .resolve_ref(reference)?
            .ok_or_else(|| HistoryError::UnresolvedReference(reference.0.clone()))?;
        let history = head_log_entries(self.graph, &head_hash, limit)?
            .into_iter()
            .map(|entry| HistoryEntry {
                is_head: entry.commit_hash == head_hash,
                commit_hash: entry.commit_hash,
                parent_hash: entry.parent_hash,
                committer: entry.committer,
                email: entry.email,
                date: entry.date,
                message: entry.message,
            })
            .collect::<Vec<_>>();
        Ok(history)
    }

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self)))]
    fn log(&self, limit: Option<usize>) -> SqlResult<Vec<HistoryEntry>> {
        let head_hash = self.current_head()?;
        let history = match head_hash.as_ref() {
            Some(head_hash) => head_log_entries(self.graph, head_hash, limit)?,
            None => Vec::new(),
        }
        .into_iter()
        .map(|entry| HistoryEntry {
            is_head: head_hash.as_ref() == Some(&entry.commit_hash),
            commit_hash: entry.commit_hash,
            parent_hash: entry.parent_hash,
            committer: entry.committer,
            email: entry.email,
            date: entry.date,
            message: entry.message,
        })
        .collect::<Vec<_>>();
        Ok(history)
    }

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self)))]
    fn status(&self) -> SqlResult<Vec<HistoryStatus>> {
        status_rows(self.graph).map(|rows| {
            rows.into_iter()
                .map(|row| HistoryStatus {
                    table_name: row.table_name,
                    staged: row.staged,
                    status: row.status,
                })
                .collect()
        })
    }

    fn conflicts(&self) -> SqlResult<Vec<HistoryConflict>> {
        conflict_rows(self.graph)
    }

    fn commit_exists(&self, commit_hash: &DoltHashId) -> SqlResult<bool> {
        commit_exists(self.graph, commit_hash)
    }

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self, message)))]
    fn commit_all(&self, message: &str) -> SqlResult<DoltHashId> {
        commit_all_with_config(self.graph, self.config, message)
    }

    fn checkout_branch(&self, branch_name: &BranchName) -> SqlResult<()> {
        connect_branch(self.graph, &branch_name.0)
    }

    fn checkout_commit(&self, commit_hash: &DoltHashId) -> SqlResult<()> {
        run_history_statement(self.graph, "SELECT dolt_checkout(?1)", &[commit_hash])
    }

    fn create_branch(
        &self,
        branch_name: &BranchName,
        start_ref: Option<&CommitRef>,
    ) -> SqlResult<()> {
        match start_ref {
            Some(start_ref) => run_history_statement(
                self.graph,
                "SELECT dolt_branch(?1, ?2)",
                &[&branch_name.0, &start_ref.0],
            ),
            None => create_branch(self.graph, &branch_name.0),
        }
    }

    fn delete_branch(&self, branch_name: &BranchName) -> SqlResult<()> {
        delete_branch(self.graph, &branch_name.0)
    }

    fn merge_base(&self, source: &CommitRef, target: &CommitRef) -> SqlResult<DoltHashId> {
        merge_base(self.graph, &source.0, &target.0)
    }

    fn merge(&self, reference: &CommitRef) -> SqlResult<()> {
        merge(self.graph, &reference.0)
    }

    fn cherry_pick(&self, commit_hash: &DoltHashId) -> SqlResult<()> {
        cherry_pick(self.graph, &commit_hash.to_string())
    }

    fn reset_hard(&self, target: &CommitRef) -> SqlResult<()> {
        reset_hard(self.graph, &target.0)
    }

    fn push(&self, remote_name: &str, branch_name: &BranchName) -> SqlResult<()> {
        push(self.graph, remote_name, &branch_name.0)
    }

    fn pull(&self, remote_name: &str, branch_name: &BranchName) -> SqlResult<()> {
        pull(self.graph, remote_name, &branch_name.0)
    }

    fn fetch(&self, remote_name: &str, branch_name: Option<&BranchName>) -> SqlResult<()> {
        fetch(
            self.graph,
            remote_name,
            branch_name.map(|branch_name| branch_name.0.as_str()),
        )
    }

    fn graph(&self) -> &GraphConnection {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_core::{BranchName, CommitRef, DoltHashId, HashId};
    use tempfile::tempdir;

    use super::{
        DoltHistoryStore, active_branch, add_remote, branch_exists, branch_hash, branch_rows,
        checkout, commit_all, commit_staged_all, connect_branch, create_branch, diff_row_count,
        hash_of, is_current_branch_dirty, log_entries, log_entries_for_hashes,
        log_entries_for_revision, merge, merge_base, remote_rows, remove_remote, reset_hard,
        search_branch_names, set_commit_author_email, set_commit_author_name, status_rows,
    };
    use crate::{
        annotations::{AnnotationFileChecksumOverrides, add_annotation_file},
        assets::{AssetRef, AssetRole, OperationAsset, OperationKind},
        collection::Collection,
        db,
        history::{HistoryError, HistoryStore},
        operations::Defaults,
        sample::{NewSample, Sample},
        test_helpers::{get_connection, setup_gen_on_disk},
    };

    #[test]
    fn test_diff_row_count_counts_committed_sample_change() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "main-collection").expect("should insert collection");
        commit_all(&conn, "initial commit").expect("should create initial commit");
        Sample::create(
            &conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert sample");
        commit_all(&conn, "sample commit").expect("should commit sample change");

        let sample_diff_count = diff_row_count(&conn, "samples", "HEAD~1", "HEAD")
            .expect("should diff committed samples");
        assert_eq!(
            sample_diff_count, 1,
            "sample commit should add one sample row"
        );
    }

    #[test]
    fn test_resolve_operation_hash_rejects_an_unresolved_commit_ref() {
        let conn = get_connection(None).expect("should create graph database");
        let history_store = DoltHistoryStore::new(&conn);

        let error = history_store
            .resolve_operation_hash(&CommitRef("missing-commit".to_string()))
            .expect_err("should reject an unresolved commit reference");

        assert!(
            matches!(error, HistoryError::UnresolvedReference(reference) if reference == "missing-commit"),
            "missing commit should produce an unresolved-reference error"
        );
    }

    #[test]
    fn test_branch_rows_include_created_branch() {
        let conn = get_connection(None).expect("should create graph database");

        create_branch(&conn, "feature").expect("should create a feature branch");

        let branches = branch_rows(&conn).expect("should query branches");
        assert!(
            branches.iter().any(|branch| branch.name == "feature"),
            "branch list should include the created feature branch"
        );
        assert!(branch_exists(&conn, "feature").expect("should find feature branch"));
        assert!(!branch_exists(&conn, "missing").expect("should not find missing branch"));
        assert_eq!(
            branch_hash(&conn, "feature").expect("should read feature hash"),
            branches
                .iter()
                .find(|branch| branch.name == "feature")
                .map(|branch| branch.hash)
        );
        assert_eq!(
            Some(hash_of(&conn, "feature").expect("should resolve feature hash")),
            branch_hash(&conn, "feature").expect("should read feature hash")
        );
        assert_eq!(
            branch_hash(&conn, "missing").expect("missing branch should be optional"),
            None
        );
    }

    #[test]
    fn search_branch_names_ranks_exact_then_prefix_matches() {
        let conn = get_connection(None).expect("should create graph database");
        for name in ["feature", "feature-next", "my-feature"] {
            create_branch(&conn, name).expect("should create branch");
        }

        let branches = search_branch_names(&conn, "feature").expect("should search branches");

        assert_eq!(branches, vec!["feature", "feature-next", "my-feature"]);
    }

    #[test]
    fn test_merge_then_reset_hard_restores_pre_merge_state() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        commit_all(&conn, "initial commit").expect("should create initial commit");
        let main_branch = active_branch(&conn).expect("should resolve the active branch");
        create_branch(&conn, "feature").expect("should create a feature branch");
        checkout(&conn, "feature").expect("should checkout feature branch");
        Sample::create(
            &conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert feature branch graph row");
        commit_all(&conn, "feature commit").expect("should commit feature branch changes");
        checkout(&conn, &main_branch).expect("should checkout the original branch");
        Collection::create(&conn, "main-second-collection")
            .expect("should insert a main branch graph row");
        commit_all(&conn, "main branch commit").expect("should commit main branch change");
        merge(&conn, "feature").expect("should merge feature branch");

        let feature_sample_exists_after_merge = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM samples WHERE name = ?1)",
                ["feature-sample"],
                |row| row.get::<_, bool>(0),
            )
            .expect("should query merged sample state");
        assert!(
            feature_sample_exists_after_merge,
            "merge should add rows from the feature branch"
        );

        reset_hard(&conn, "HEAD~1").expect("should hard reset to the pre-merge commit");
        let feature_sample_exists = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM samples WHERE name = ?1)",
                ["feature-sample"],
                |row| row.get::<_, bool>(0),
            )
            .expect("should query reset sample state");
        assert!(
            !feature_sample_exists,
            "hard reset should remove rows introduced by the merge"
        );
    }

    #[test]
    fn test_merge_base_returns_common_ancestor_of_divergent_branches() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "base-collection").expect("should insert base graph row");
        let base = commit_all(&conn, "base commit").expect("should create base commit");
        let main_branch = active_branch(&conn).expect("should resolve the active branch");
        create_branch(&conn, "feature").expect("should create feature branch");

        Collection::create(&conn, "main-collection").expect("should insert main graph row");
        commit_all(&conn, "main commit").expect("should create main commit");

        checkout(&conn, "feature").expect("should checkout feature branch");
        Collection::create(&conn, "feature-collection").expect("should insert feature graph row");
        commit_all(&conn, "feature commit").expect("should create feature commit");

        let common_ancestor =
            merge_base(&conn, &main_branch, "feature").expect("should resolve merge base");
        assert_eq!(
            common_ancestor, base,
            "merge base should be common ancestor"
        );
    }

    // History order is for presentation, while `parent_hash` and tilde refs
    // represent directed first-parent edges. A merge must preserve both
    // contracts so callers never infer ancestry from adjacent history rows.
    #[test]
    fn test_merge_history_records_first_parents_and_resolves_tilde_refs() {
        let conn = get_connection(None).expect("should create graph database");
        let history_store = DoltHistoryStore::new(&conn);

        Collection::create(&conn, "base-collection").expect("should insert base graph row");
        let base_commit = history_store
            .commit_all("base")
            .expect("should commit base state");
        history_store
            .create_branch(&BranchName("feature".to_string()), None)
            .expect("should create feature branch");
        history_store
            .checkout_branch(&BranchName("feature".to_string()))
            .expect("should checkout feature branch");
        Collection::create(&conn, "feature-collection").expect("should insert feature graph row");
        let feature_commit = history_store
            .commit_all("feature")
            .expect("should commit feature state");

        history_store
            .checkout_branch(&BranchName("main".to_string()))
            .expect("should checkout main branch");
        Collection::create(&conn, "main-collection").expect("should insert main graph row");
        let main_commit = history_store
            .commit_all("main")
            .expect("should commit main state");
        history_store
            .merge(&CommitRef("feature".to_string()))
            .expect("should merge feature branch");
        let merge_commit = history_store
            .current_head()
            .expect("should query merged head")
            .expect("should create a merge commit");
        let entries = history_store.log(None).expect("should load merged history");
        let parent_for = |commit_hash| {
            entries
                .iter()
                .find(|entry| entry.commit_hash == commit_hash)
                .expect("should include commit in merged history")
                .parent_hash
        };

        assert_eq!(
            parent_for(merge_commit),
            Some(main_commit),
            "merge entry should record the main-side first parent"
        );
        assert_eq!(
            parent_for(main_commit),
            Some(base_commit),
            "main commit should retain its direct parent"
        );
        assert_eq!(
            parent_for(feature_commit),
            Some(base_commit),
            "feature commit should retain its direct parent"
        );
        assert!(
            entries.iter().any(|entry| entry.parent_hash.is_none()),
            "merged history should retain its root entry without a parent"
        );
        assert_eq!(
            history_store
                .resolve_ref(&CommitRef("HEAD~1".to_string()))
                .expect("should resolve first parent"),
            Some(main_commit),
            "HEAD~1 should follow the merge's first-parent edge"
        );
        assert_eq!(
            history_store
                .resolve_ref(&CommitRef("HEAD~2".to_string()))
                .expect("should resolve first-parent grandparent"),
            Some(base_commit),
            "HEAD~2 should remain on the main-side first-parent chain"
        );
    }

    #[test]
    fn test_remote_rows_include_added_remote() {
        let conn = get_connection(None).expect("should create graph database");

        let remote_dir = tempdir().expect("should create temporary remote directory");
        let remote_url = remote_dir.path().display().to_string();
        add_remote(&conn, "origin", &remote_url).expect("should add a file remote");
        let remotes = remote_rows(&conn).expect("should query remotes");
        assert!(
            remotes
                .iter()
                .any(|remote| remote.name == "origin" && remote.url == remote_url),
            "dolt_remotes should include the configured file remote"
        );

        remove_remote(&conn, "origin").expect("should remove the file remote");
        assert!(
            remote_rows(&conn).expect("should query remotes").is_empty(),
            "dolt_remotes should not include the removed remote"
        );
    }

    #[test]
    fn test_commit_staged_all_clears_status() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        let status_before_commit = status_rows(&conn).expect("should query dolt status");
        assert!(
            !status_before_commit.is_empty(),
            "status should report modified tables before commit"
        );

        let commit_hash =
            commit_staged_all(&conn, "initial commit").expect("should commit staged changes");
        assert!(
            commit_hash != DoltHashId::default(),
            "commit hash should be returned for explicit staged-all commit"
        );

        let clean_status = status_rows(&conn).expect("should query clean dolt status");
        assert!(
            clean_status.is_empty(),
            "explicit staged-all commit should clear dolt status"
        );
    }

    #[test]
    fn test_commit_all_uses_configured_commit_author() {
        let conn = get_connection(None).expect("should create graph database");
        set_commit_author_name(&conn, "Test User").expect("should configure commit author name");
        set_commit_author_email(&conn, "test@example.com")
            .expect("should configure commit author email");

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        commit_all(&conn, "initial commit").expect("should commit staged changes");

        let history = log_entries(&conn).expect("should query Dolt log");
        let head = history.first().expect("should contain committed entry");
        assert_eq!(head.committer, "Test User");
        assert_eq!(head.email, "test@example.com");
    }

    #[test]
    fn test_commit_all_uses_config_default_commit_author() {
        let context = setup_gen_on_disk();
        let graph_conn = context.graph().conn();
        let config_conn = context.config().conn();
        Defaults::set_default_committer_name(config_conn, "Config User")
            .expect("should configure default committer name");
        Defaults::set_default_committer_email(config_conn, "config@example.com")
            .expect("should configure default committer email");

        Collection::create(graph_conn, "main-collection").expect("should insert graph row");
        DoltHistoryStore::new_with_config(graph_conn, config_conn)
            .commit_all("initial commit")
            .expect("should commit staged changes");

        let log = log_entries(graph_conn).expect("should load commit log");
        let head = log.first().expect("should include committed head");
        assert_eq!(head.committer, "Config User");
        assert_eq!(head.email, "config@example.com");
    }

    #[test]
    fn test_commit_all_preserves_configured_commit_author() {
        let context = setup_gen_on_disk();
        let graph_conn = context.graph().conn();
        let config_conn = context.config().conn();
        Defaults::set_default_committer_name(config_conn, "Config User")
            .expect("should configure default committer name");
        Defaults::set_default_committer_email(config_conn, "config@example.com")
            .expect("should configure default committer email");
        set_commit_author_name(graph_conn, "Manual User")
            .expect("should configure graph committer name");
        set_commit_author_email(graph_conn, "manual@example.com")
            .expect("should configure graph committer email");

        Collection::create(graph_conn, "main-collection").expect("should insert graph row");
        DoltHistoryStore::new_with_config(graph_conn, config_conn)
            .commit_all("initial commit")
            .expect("should commit staged changes");

        let log = log_entries(graph_conn).expect("should load commit log");
        let head = log.first().expect("should include committed head");
        assert_eq!(head.committer, "Manual User");
        assert_eq!(head.email, "manual@example.com");
    }

    #[test]
    fn test_commit_staged_all_reports_nothing_to_commit_on_clean_repo() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        commit_all(&conn, "initial commit").expect("should commit initial state");

        let err = commit_staged_all(&conn, "second commit")
            .expect_err("clean repo should not produce a second commit");
        let err_text = err.to_string();
        assert!(
            err_text.contains("nothing to commit"),
            "clean repo error should explain no-op commit: {err_text}"
        );
    }

    #[test]
    fn test_is_current_branch_dirty_matches_status_rows() {
        let conn = get_connection(None).expect("should create graph database");

        assert_eq!(
            is_current_branch_dirty(&conn).expect("should query fresh branch dirty flag"),
            !status_rows(&conn)
                .expect("should query fresh dolt status rows")
                .is_empty(),
            "fresh repo dirty flag should match dolt_status emptiness"
        );

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        assert_eq!(
            is_current_branch_dirty(&conn).expect("should query dirty branch flag"),
            !status_rows(&conn)
                .expect("should query dirty dolt status rows")
                .is_empty(),
            "dirty repo dirty flag should match dolt_status emptiness"
        );

        commit_all(&conn, "initial commit").expect("should commit initial state");
        assert_eq!(
            is_current_branch_dirty(&conn).expect("should query clean branch flag"),
            !status_rows(&conn)
                .expect("should query clean dolt status rows")
                .is_empty(),
            "clean repo dirty flag should match dolt_status emptiness"
        );
    }

    #[test]
    fn test_history_store_resolves_refs_and_logs_commits() {
        let conn = get_connection(None).expect("should create graph database");
        let history_store = DoltHistoryStore::new(&conn);

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        let first_commit = history_store
            .commit_all("initial commit")
            .expect("should commit initial graph change");
        history_store
            .create_branch(&BranchName("feature".to_string()), None)
            .expect("should create branch through the history facade");
        history_store
            .checkout_branch(&BranchName("feature".to_string()))
            .expect("should checkout branch through the history facade");
        Sample::create(
            &conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert feature graph row");
        let second_commit = history_store
            .commit_all("feature commit")
            .expect("should commit feature graph change");
        Sample::create(
            &conn,
            NewSample {
                name: "second-feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert second feature graph row");
        let third_commit = history_store
            .commit_all("second feature commit")
            .expect("should commit second feature graph change");

        let current_head = history_store
            .current_head()
            .expect("should resolve current head")
            .expect("should return a head commit");
        assert_eq!(current_head, third_commit);

        let branch_name = history_store
            .current_branch()
            .expect("should resolve current branch")
            .expect("should return a branch name");
        assert_eq!(branch_name, BranchName("feature".to_string()));

        let head_ref = history_store
            .resolve_ref(&CommitRef("HEAD".to_string()))
            .expect("should resolve HEAD")
            .expect("should match current head");
        assert_eq!(head_ref, third_commit);

        let previous_ref = history_store
            .resolve_ref(&CommitRef("HEAD~1".to_string()))
            .expect("should resolve HEAD~1")
            .expect("should match parent commit");
        assert_eq!(previous_ref, second_commit);

        let grandparent_ref = history_store
            .resolve_ref(&CommitRef("HEAD~2".to_string()))
            .expect("should resolve HEAD~2")
            .expect("should match grandparent commit");
        assert_eq!(grandparent_ref, first_commit);

        let branch_ref = history_store
            .resolve_ref(&CommitRef("feature".to_string()))
            .expect("should resolve branch name")
            .expect("should match branch head");
        assert_eq!(branch_ref, third_commit);

        let third_commit_hex = third_commit.to_string();
        let short_hash = &third_commit_hex[..12];
        let prefix_ref = history_store
            .resolve_ref(&CommitRef(short_hash.to_string()))
            .expect("should resolve unique hash prefix")
            .expect("should match the prefixed commit");
        assert_eq!(prefix_ref, third_commit);

        let head_operation_hash = history_store
            .resolve_operation_hash(&CommitRef("HEAD".to_string()))
            .expect("should resolve HEAD as an operation hash");
        assert_eq!(head_operation_hash, third_commit);

        let history = history_store
            .log(Some(2))
            .expect("should query history through the facade");
        assert_eq!(history.len(), 2, "history log should honor the limit");
        assert!(
            history[0].is_head,
            "the newest history entry should be flagged as HEAD"
        );
        assert_eq!(history[0].message, "second feature commit");
        assert_eq!(history[1].message, "feature commit");
    }

    #[test]
    fn test_log_for_ref_reads_non_current_branch_without_changing_checkout() {
        let conn = get_connection(None).expect("should create graph database");
        let history_store = DoltHistoryStore::new(&conn);

        Collection::create(&conn, "main-collection").expect("should insert main graph row");
        history_store
            .commit_all("main commit")
            .expect("should commit main graph change");
        history_store
            .create_branch(&BranchName("feature".to_string()), None)
            .expect("should create feature branch");
        history_store
            .checkout_branch(&BranchName("feature".to_string()))
            .expect("should checkout feature branch");
        Sample::create(
            &conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert feature graph row");
        history_store
            .commit_all("feature commit")
            .expect("should commit feature graph change");
        history_store
            .checkout_branch(&BranchName("main".to_string()))
            .expect("should restore main before the history lookup");

        let feature_history = history_store
            .log_for_ref(&CommitRef("feature".to_string()), None)
            .expect("should load history from a non-current branch");

        assert_eq!(feature_history[0].message, "feature commit");
        assert_eq!(
            history_store
                .current_branch()
                .expect("should read current branch"),
            Some(BranchName("main".to_string())),
            "history lookup should not change the checkout"
        );
    }

    #[test]
    fn test_log_entries_for_hashes_loads_commits_across_branches_and_omits_missing_hashes() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "base-collection").expect("should insert base graph row");
        let base_commit = commit_all(&conn, "base commit").expect("should commit base change");
        create_branch(&conn, "feature").expect("should create feature branch");

        Collection::create(&conn, "main-collection").expect("should insert main graph row");
        let main_commit = commit_all(&conn, "main commit").expect("should commit main change");

        checkout(&conn, "feature").expect("should checkout feature branch");
        Collection::create(&conn, "feature-collection").expect("should insert feature graph row");
        let feature_commit =
            commit_all(&conn, "feature commit").expect("should commit feature change");
        checkout(&conn, "main").expect("should restore main branch");

        let entries = log_entries_for_hashes(
            &conn,
            &[
                feature_commit,
                main_commit,
                base_commit,
                DoltHashId::default(),
            ],
        )
        .expect("should load requested commits from all repository branches");

        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.commit_hash)
                .collect::<Vec<_>>(),
            vec![feature_commit, main_commit, base_commit],
            "all existing commits should be returned in request order while missing hashes are omitted"
        );
        assert_eq!(
            entries[0].parent_hash,
            Some(base_commit),
            "the feature commit should retain its first parent"
        );
        assert_eq!(
            entries[0].message, "feature commit",
            "the requested commit metadata should include its message"
        );
        assert_eq!(
            active_branch(&conn).expect("should read active branch"),
            "main",
            "commit metadata lookup should not change the active checkout"
        );
    }

    #[test]
    fn test_log_entries_for_revision_uses_dolt_commit_ranges() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "base-collection").expect("should insert base graph row");
        let base_commit = commit_all(&conn, "base commit").expect("should commit base change");
        create_branch(&conn, "feature").expect("should create feature branch");

        Collection::create(&conn, "main-collection").expect("should insert main graph row");
        let main_commit = commit_all(&conn, "main commit").expect("should commit main change");

        checkout(&conn, "feature").expect("should checkout feature branch");
        Collection::create(&conn, "feature-collection").expect("should insert feature graph row");
        let feature_commit =
            commit_all(&conn, "feature commit").expect("should commit feature change");
        checkout(&conn, "main").expect("should restore main branch");

        let feature_entries = log_entries_for_revision(&conn, "main..feature")
            .expect("should query the feature-only Dolt log range");
        assert_eq!(
            feature_entries
                .iter()
                .map(|entry| entry.commit_hash)
                .collect::<Vec<_>>(),
            vec![feature_commit],
            "Dolt range should return only commits unique to feature"
        );

        let main_entries = log_entries_for_revision(&conn, "feature..main")
            .expect("should query the main-only Dolt log range");
        assert_eq!(
            main_entries
                .iter()
                .map(|entry| entry.commit_hash)
                .collect::<Vec<_>>(),
            vec![main_commit],
            "reversing the Dolt range should return only commits unique to main"
        );

        let feature_from_base = log_entries_for_revision(&conn, &format!("{base_commit}..feature"))
            .expect("should query a hash-to-branch Dolt log range");
        assert_eq!(
            feature_from_base
                .iter()
                .map(|entry| entry.commit_hash)
                .collect::<Vec<_>>(),
            vec![feature_commit],
            "two-dot Dolt ranges should exclude the left revision"
        );
    }

    #[test]
    fn test_resolve_ref_errors_for_ambiguous_hash_prefix() {
        let conn = get_connection(None).expect("should create graph database");
        let history_store = DoltHistoryStore::new(&conn);
        Collection::create(&conn, "main-collection").expect("should insert initial graph row");
        let first_commit = history_store
            .commit_all("initial commit")
            .expect("should commit initial graph change");

        let mut commits = vec![first_commit];
        for index in 1..=16 {
            let sample_name = format!("sample-{index}");
            Sample::create(
                &conn,
                NewSample {
                    name: &sample_name,
                    is_reference: false,
                },
            )
            .expect("should insert graph row");
            let commit = history_store
                .commit_all(&format!("commit {index}"))
                .expect("should commit graph change");
            commits.push(commit);
        }

        let ambiguous_prefix = commits
            .iter()
            .fold(
                std::collections::HashMap::<char, usize>::new(),
                |mut counts, commit| {
                    let first_char = commit
                        .to_string()
                        .chars()
                        .next()
                        .expect("should contain at least one hex digit");
                    *counts.entry(first_char).or_default() += 1;
                    counts
                },
            )
            .into_iter()
            .find_map(|(character, count)| (count >= 2).then(|| character.to_string()))
            .expect("should find an ambiguous one-digit prefix after 17 commits");

        let err = history_store
            .resolve_ref(&CommitRef(ambiguous_prefix.clone()))
            .expect_err("ambiguous prefix should be reported as an error");
        let err_text = err.to_string();
        assert!(
            err_text.contains("ambiguous"),
            "ambiguous prefix should mention ambiguity: {err_text}"
        );
    }

    #[test]
    fn test_doltlite_sample_diff_contract() {
        let conn = get_connection(None).expect("should create graph database");

        Collection::create(&conn, "main-collection").expect("should insert collection");
        commit_all(&conn, "initial commit").expect("should commit initial graph change");
        Sample::create(
            &conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert sample");
        commit_all(&conn, "sample commit").expect("should commit sample change");

        let mut statement = conn
            .prepare("SELECT to_name, from_name, diff_type FROM dolt_diff_samples(?1, ?2)")
            .expect("should prepare dolt diff query");
        let mut rows = statement
            .query(["HEAD~1", "HEAD"])
            .expect("should query dolt diff rows");
        let row = rows
            .next()
            .expect("should yield a result row")
            .expect("should decode dolt diff row");
        let to_name: String = row.get(0).expect("should include the added sample name");
        let from_name: Option<String> = row.get(1).expect("should decode nullable from_name");
        let diff_type: String = row.get(2).expect("should decode diff_type");
        assert_eq!(to_name, "feature-sample");
        assert_eq!(from_name, None);
        assert_eq!(
            diff_type, "added",
            "dolt_diff_samples should label the inserted row as added"
        );
    }

    #[test]
    fn test_doltlite_annotation_asset_diff_contract() {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/simple.gff");

        add_annotation_file(
            &context,
            fixture_path.to_str().expect("should encode fixture path"),
            None,
            None,
            Some("fixture-track"),
            Some("annotation diff contract"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should commit annotation-file asset refs");

        let mut log_statement = conn
            .prepare(
                "SELECT to_id, to_operation_kind, diff_type \
                 FROM dolt_diff_gen_operation_log(?1, ?2)",
            )
            .expect("should prepare operation log diff query");
        let log_rows = log_statement
            .query_map(["HEAD~1", "HEAD"], |row| {
                Ok((
                    row.get::<_, Option<Vec<u8>>>(0)?,
                    row.get::<_, Option<OperationKind>>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .expect("should query operation log diffs")
            .collect::<Result<Vec<_>, _>>()
            .expect("should decode operation log diff rows");
        let log_id = log_rows
            .iter()
            .find_map(|(log_id, operation_kind, diff_type)| {
                (operation_kind.as_ref() == Some(&OperationKind::AnnotationFile)
                    && diff_type == "added")
                    .then(|| log_id.clone())
                    .flatten()
            });
        assert!(
            log_id.is_some(),
            "annotation-file commit should add a graph operation log row: {log_rows:?}"
        );
        let log_id = log_id.expect("should locate the committed log id");
        let log_id = HashId::try_from(log_id.as_slice()).expect("should decode log id");

        let mut asset_ref_statement = conn
            .prepare("SELECT diff_type FROM dolt_diff_gen_asset_refs(?1, ?2)")
            .expect("should prepare asset ref diff query");
        let asset_ref_rows = asset_ref_statement
            .query_map(["HEAD~1", "HEAD"], |row| row.get::<_, String>(0))
            .expect("should query asset ref diffs")
            .collect::<Result<Vec<_>, _>>()
            .expect("should decode asset ref diff rows");
        assert!(
            asset_ref_rows.iter().any(|diff_type| diff_type == "added"),
            "annotation-file commit should add an asset ref row: {asset_ref_rows:?}"
        );

        let committed_assets = OperationAsset::by_log_id(conn, &log_id)
            .into_iter()
            .filter_map(|asset| AssetRef::get_by_id(conn, &asset.asset_ref_id, None))
            .map(|asset_ref| (asset_ref.file_type, asset_ref.role, asset_ref.name))
            .collect::<Vec<_>>();
        assert!(
            committed_assets.iter().any(|(file_type, role, name)| {
                file_type == "gff3"
                    && *role == AssetRole::Annotation
                    && name.as_deref() == Some("fixture-track")
            }),
            "committed log should resolve to the annotation asset metadata: {committed_assets:?}"
        );
    }

    #[test]
    fn test_connect_branch_restores_clean_status_after_reopen() {
        let temp_dir = tempdir().expect("should create temporary repo directory");
        let db_path = temp_dir.path().join("default.db");
        let conn = db::get_connection(&db_path).expect("should open graph database");

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        commit_all(&conn, "initial commit").expect("should create initial commit");
        create_branch(&conn, "feature").expect("should create a feature branch");
        checkout(&conn, "feature").expect("should checkout feature branch");
        Sample::create(
            &conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert feature sample");
        commit_all(&conn, "feature commit").expect("should commit feature branch graph change");
        checkout(&conn, "main").expect("should switch back to main");
        let clean_status = status_rows(&conn).expect("should query clean status");
        assert!(
            clean_status.is_empty(),
            "checkout back to main should leave a clean working tree before reopen: {clean_status:?}"
        );
        drop(conn);

        let reopened = db::get_connection(&db_path).expect("should reopen graph database");
        connect_branch(&reopened, "main").expect("should reconnect main branch state");
        let reopened_status = status_rows(&reopened).expect("should query reopened status");
        assert!(
            reopened_status.is_empty(),
            "reopening main should keep a clean working tree: {reopened_status:?}"
        );
    }

    #[test]
    fn test_checkout_branch_created_from_relative_ref_after_copy() {
        let temp_dir = tempdir().expect("should create temporary repo directory");
        let db_path = temp_dir.path().join("default.db");
        let conn = db::get_connection(&db_path).expect("should open graph database");

        Collection::create(&conn, "main-collection").expect("should insert initial collection");
        commit_all(&conn, "initial commit").expect("should create initial commit");
        Sample::create(
            &conn,
            NewSample {
                name: "feature-sample",
                is_reference: false,
            },
        )
        .expect("should insert sample on main");
        commit_all(&conn, "sample commit").expect("should create second commit");

        let copied_dir = tempdir().expect("should create copied repo directory");
        let copied_path = copied_dir.path().join("default.db");
        std::fs::copy(&db_path, &copied_path).expect("should copy graph database");

        let copied = db::get_connection(&copied_path).expect("should open copied graph database");
        connect_branch(&copied, "main").expect("should reconnect copied main branch");
        let copied_history = DoltHistoryStore::new(&copied);
        copied_history
            .create_branch(
                &BranchName("snapshot".to_string()),
                Some(&CommitRef("main~1".to_string())),
            )
            .expect("should create a snapshot branch at the prior main commit");
        copied_history
            .checkout_branch(&BranchName("snapshot".to_string()))
            .expect("should checkout the copied snapshot branch");

        let feature_sample_exists = copied
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM samples WHERE name = ?1)",
                ["feature-sample"],
                |row| row.get::<_, bool>(0),
            )
            .expect("should query copied reset sample state");
        assert!(
            !feature_sample_exists,
            "snapshot branch checkout should expose the prior commit state"
        );
    }
}
