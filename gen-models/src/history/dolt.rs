use gen_core::{BranchName, CommitHash, CommitRef};
use rusqlite::{OptionalExtension, Result as SqlResult, params};

use crate::{
    db::GraphConnection,
    history::{
        HistoryConflict, HistoryEntry, HistoryError, HistoryResult, HistoryStatus, HistoryStore,
    },
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltLogEntry {
    pub commit_hash: CommitHash,
    pub committer: String,
    pub email: String,
    pub date: String,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltStatusRow {
    pub table_name: String,
    pub staged: bool,
    pub status: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltBranchRow {
    pub name: String,
    pub hash: String,
    pub latest_committer: String,
    pub latest_committer_email: String,
    pub latest_commit_date: String,
    pub latest_commit_message: String,
    pub remote: String,
    pub branch: String,
    pub dirty: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoltRemoteRow {
    pub name: String,
    pub url: String,
    pub fetch_specs: String,
    pub params: String,
}

const DEFAULT_COMMITTER_NAME: &str = "Gen";
const DEFAULT_COMMITTER_EMAIL: &str = "gen@genhub.bio";

fn run_history_statement(
    conn: &GraphConnection,
    query: &str,
    params: &[&dyn rusqlite::ToSql],
) -> SqlResult<()> {
    conn.query_row(query, params, |row| row.get_ref(0).map(|_| ()))?;
    Ok(())
}

pub struct DoltHistoryStore<'connection> {
    graph: &'connection GraphConnection,
}

impl<'connection> DoltHistoryStore<'connection> {
    pub fn new(graph: &'connection GraphConnection) -> Self {
        Self { graph }
    }
}

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

pub fn set_default_commit_author(conn: &GraphConnection) -> SqlResult<()> {
    if commit_author_name(conn)?.is_none() {
        set_commit_author_name(conn, DEFAULT_COMMITTER_NAME)?;
    }
    if commit_author_email(conn)?.is_none() {
        set_commit_author_email(conn, DEFAULT_COMMITTER_EMAIL)?;
    }
    Ok(())
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, message)))]
pub fn commit_all(conn: &GraphConnection, message: &str) -> SqlResult<CommitHash> {
    set_default_commit_author(conn)?;
    commit_staged_all(conn, message)
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, message)))]
pub fn commit_staged_all(conn: &GraphConnection, message: &str) -> SqlResult<CommitHash> {
    conn.query_row("SELECT dolt_commit('-A', '-m', ?1)", [message], |row| {
        row.get::<_, String>(0).map(CommitHash)
    })
}

pub fn active_branch(conn: &GraphConnection) -> SqlResult<String> {
    conn.query_row("SELECT active_branch()", [], |row| row.get(0))
}

pub fn log_entries(conn: &GraphConnection) -> SqlResult<Vec<DoltLogEntry>> {
    let mut statement =
        conn.prepare("SELECT commit_hash, committer, email, date, message FROM dolt_log")?;
    let rows = statement.query_map([], |row| {
        Ok(DoltLogEntry {
            commit_hash: CommitHash(row.get(0)?),
            committer: row.get(1)?,
            email: row.get(2)?,
            date: row.get(3)?,
            message: row.get(4)?,
        })
    })?;
    rows.collect()
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, head_hash)))]
fn head_log_entries(
    conn: &GraphConnection,
    head_hash: &CommitHash,
    limit: Option<usize>,
) -> SqlResult<Vec<DoltLogEntry>> {
    match limit {
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
                 SELECT log.commit_hash, log.committer, log.email, log.date, log.message \
                 FROM ancestry \
                 JOIN dolt_log AS log ON log.commit_hash = ancestry.commit_hash \
                 GROUP BY log.commit_hash, log.committer, log.email, log.date, log.message \
                 ORDER BY MIN(ancestry.depth), log.date DESC \
                 LIMIT ?2",
            )?;
            let rows = statement.query_map(params![head_hash.0, max_depth], |row| {
                Ok(DoltLogEntry {
                    commit_hash: CommitHash(row.get(0)?),
                    committer: row.get(1)?,
                    email: row.get(2)?,
                    date: row.get(3)?,
                    message: row.get(4)?,
                })
            })?;
            rows.collect()
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
                 SELECT log.commit_hash, log.committer, log.email, log.date, log.message \
                 FROM ancestry \
                 JOIN dolt_log AS log ON log.commit_hash = ancestry.commit_hash \
                 GROUP BY log.commit_hash, log.committer, log.email, log.date, log.message \
                 ORDER BY MIN(ancestry.depth), log.date DESC",
            )?;
            let rows = statement.query_map([&head_hash.0], |row| {
                Ok(DoltLogEntry {
                    commit_hash: CommitHash(row.get(0)?),
                    committer: row.get(1)?,
                    email: row.get(2)?,
                    date: row.get(3)?,
                    message: row.get(4)?,
                })
            })?;
            rows.collect()
        }
    }
}

#[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, head_hash)))]
fn nth_ancestor_hash(
    conn: &GraphConnection,
    head_hash: &CommitHash,
    offset: usize,
) -> SqlResult<Option<CommitHash>> {
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
             WHERE ancestors.parent_hash IS NOT NULL AND ancestry.depth < ?2 \
         ) \
         SELECT commit_hash \
         FROM ancestry \
         WHERE depth = ?2 \
         LIMIT 1",
        params![head_hash.0, offset],
        |row| row.get::<_, String>(0).map(CommitHash),
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

pub fn clone_remote(conn: &GraphConnection, remote_url: &str) -> SqlResult<()> {
    run_history_statement(conn, "SELECT dolt_clone(?1)", &[&remote_url])
}

pub fn branch_rows(conn: &GraphConnection) -> SqlResult<Vec<DoltBranchRow>> {
    let mut statement = conn.prepare(
        "SELECT name, hash, latest_committer, latest_committer_email, \
         latest_commit_date, latest_commit_message, remote, branch, dirty \
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

impl HistoryStore for DoltHistoryStore<'_> {
    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self)))]
    fn current_head(&self) -> SqlResult<Option<CommitHash>> {
        self.graph
            .query_row(
                "SELECT hash FROM dolt_branches WHERE name = active_branch()",
                [],
                |row| row.get::<_, String>(0).map(CommitHash),
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
    fn resolve_ref(&self, reference: &CommitRef) -> HistoryResult<Option<CommitHash>> {
        if reference.0 == "HEAD" {
            return self.current_head().map_err(HistoryError::from);
        }

        if let Some(offset) = reference.0.strip_prefix("HEAD~") {
            let offset = match offset.parse::<usize>() {
                Ok(offset) => offset,
                Err(_) => return Ok(None),
            };
            let Some(head_hash) = self.current_head().map_err(HistoryError::from)? else {
                return Ok(None);
            };
            return nth_ancestor_hash(self.graph, &head_hash, offset).map_err(HistoryError::from);
        }

        let branch_hash = self
            .graph
            .query_row(
                "SELECT hash FROM dolt_branches WHERE name = ?1",
                [&reference.0],
                |row| row.get::<_, String>(0).map(CommitHash),
            )
            .optional()
            .map_err(HistoryError::from)?;
        if branch_hash.is_some() {
            return Ok(branch_hash);
        }

        let prefix_matches = log_entries(self.graph)
            .map_err(HistoryError::from)?
            .into_iter()
            .filter(|entry| entry.commit_hash.0.starts_with(&reference.0))
            .map(|entry| entry.commit_hash)
            .collect::<Vec<_>>();
        if prefix_matches.len() == 1 {
            return Ok(prefix_matches.into_iter().next());
        }
        if prefix_matches.len() > 1 {
            return Err(HistoryError::AmbiguousReference(reference.0.clone()));
        }

        if reference.0.len() == 40
            && reference
                .0
                .chars()
                .all(|character| character.is_ascii_hexdigit())
        {
            return Ok(Some(CommitHash(reference.0.clone())));
        }

        Ok(None)
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

    #[cfg_attr(feature = "profiling", tracing::instrument(skip(self, message)))]
    fn commit_all(&self, message: &str) -> SqlResult<CommitHash> {
        commit_all(self.graph, message)
    }

    fn checkout_branch(&self, branch_name: &BranchName) -> SqlResult<()> {
        connect_branch(self.graph, &branch_name.0)
    }

    fn checkout_commit(&self, commit_hash: &CommitHash) -> SqlResult<()> {
        run_history_statement(self.graph, "SELECT dolt_checkout(?1)", &[&commit_hash.0])
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

    fn merge(&self, reference: &CommitRef) -> SqlResult<()> {
        merge(self.graph, &reference.0)
    }

    fn cherry_pick(&self, commit_hash: &CommitHash) -> SqlResult<()> {
        cherry_pick(self.graph, &commit_hash.0)
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
    use std::path::{Path, PathBuf};

    use gen_core::{BranchName, CommitRef};
    use rusqlite::Connection;
    use tempfile::tempdir;

    use super::{
        DoltHistoryStore, active_branch, add_remote, branch_rows, checkout, commit_all,
        commit_staged_all, connect_branch, create_branch, diff_row_count, is_current_branch_dirty,
        log_entries, merge, remote_rows, reset_hard, set_commit_author_email,
        set_commit_author_name, set_default_commit_author, status_rows,
    };
    use crate::{
        annotations::add_annotation_file,
        collection::Collection,
        db::GraphConnection,
        history::HistoryStore,
        migrations::run_migrations,
        sample::{NewSample, Sample},
        test_helpers::{get_connection, setup_gen_on_disk},
    };

    fn open_existing_connection(path: &Path) -> GraphConnection {
        let mut conn = Connection::open(path).expect("should reopen graph database");
        rusqlite::vtab::array::load_module(&conn).expect("should load array module");
        run_migrations(&mut conn);
        GraphConnection(conn)
    }

    #[test]
    fn test_doltlite_smoke_covers_history_contract() {
        let conn = get_connection(None).expect("should create graph database");
        set_default_commit_author(&conn).expect("should configure Dolt author");

        Collection::create(&conn, "main-collection").expect("should insert graph row");
        let status_before_commit = status_rows(&conn).expect("should query dolt status");
        assert!(
            status_before_commit
                .iter()
                .any(|row| row.table_name == "collections"),
            "status should report the modified collections table"
        );

        let initial_commit =
            commit_all(&conn, "initial commit").expect("should create initial commit");
        assert!(
            !initial_commit.0.is_empty(),
            "commit hashes should be returned as non-empty strings"
        );
        let clean_status = status_rows(&conn).expect("should query clean dolt status");
        assert!(
            clean_status.is_empty(),
            "status should be empty after commit"
        );

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
        let sample_diff_count = diff_row_count(&conn, "samples", "HEAD~1", "HEAD")
            .expect("should diff committed samples");
        assert_eq!(
            sample_diff_count, 1,
            "feature commit should add one sample row"
        );

        checkout(&conn, &main_branch).expect("should checkout the original branch");
        Collection::create(&conn, "main-second-collection")
            .expect("should insert a main branch graph row");
        commit_all(&conn, "main branch commit").expect("should commit main branch change");
        merge(&conn, "feature").expect("should merge feature branch");

        let history = log_entries(&conn).expect("should query Dolt log");
        assert!(
            history
                .iter()
                .any(|entry| entry.message == "initial commit"),
            "dolt log should include the initial commit message"
        );
        assert!(
            history
                .iter()
                .any(|entry| entry.message == "feature commit"),
            "dolt log should include the feature commit message"
        );

        let branches = branch_rows(&conn).expect("should query branches");
        assert!(
            branches.iter().any(|branch| branch.name == "feature"),
            "branch list should include the created feature branch"
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
            !commit_hash.0.is_empty(),
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

        let short_hash = &third_commit.0[..12];
        let prefix_ref = history_store
            .resolve_ref(&CommitRef(short_hash.to_string()))
            .expect("should resolve unique hash prefix")
            .expect("should match the prefixed commit");
        assert_eq!(prefix_ref, third_commit);

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
                        .0
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
            .prepare("SELECT * FROM dolt_diff_samples(?1, ?2)")
            .expect("should prepare dolt diff query");
        let column_names = statement
            .column_names()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            column_names,
            vec![
                "to_name".to_string(),
                "to_is_reference".to_string(),
                "to_commit".to_string(),
                "to_commit_date".to_string(),
                "from_name".to_string(),
                "from_is_reference".to_string(),
                "from_commit".to_string(),
                "from_commit_date".to_string(),
                "diff_type".to_string(),
            ],
            "dolt_diff_samples should preserve the expected row contract"
        );

        let mut rows = statement
            .query(["HEAD~1", "HEAD"])
            .expect("should query dolt diff rows");
        let row = rows
            .next()
            .expect("should yield a result row")
            .expect("should decode dolt diff row");
        let to_name: String = row.get(0).expect("should include the added sample name");
        let from_name: Option<String> = row.get(4).expect("should decode nullable from_name");
        let diff_type: String = row.get(8).expect("should decode diff_type");
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
        )
        .expect("should commit annotation-file asset refs");

        let mut log_statement = conn
            .prepare("SELECT * FROM dolt_diff_gen_operation_log(?1, ?2)")
            .expect("should prepare operation log diff query");
        let log_columns = log_statement
            .column_names()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            log_columns,
            vec![
                "to_id".to_string(),
                "to_operation_kind".to_string(),
                "to_command".to_string(),
                "to_created_on".to_string(),
                "to_commit".to_string(),
                "to_commit_date".to_string(),
                "from_id".to_string(),
                "from_operation_kind".to_string(),
                "from_command".to_string(),
                "from_created_on".to_string(),
                "from_commit".to_string(),
                "from_commit_date".to_string(),
                "diff_type".to_string(),
            ],
            "dolt_diff_gen_operation_log should preserve the expected row contract"
        );
        let log_rows = log_statement
            .query_map(["HEAD~1", "HEAD"], |row| {
                Ok((
                    row.get::<_, Option<Vec<u8>>>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(12)?,
                ))
            })
            .expect("should query operation log diffs")
            .collect::<Result<Vec<_>, _>>()
            .expect("should decode operation log diff rows");
        let log_id = log_rows
            .iter()
            .find_map(|(log_id, operation_kind, diff_type)| {
                (operation_kind.as_deref() == Some("annotation-file") && diff_type == "added")
                    .then(|| log_id.clone())
                    .flatten()
            });
        assert!(
            log_id.is_some(),
            "annotation-file commit should add a graph operation log row: {log_rows:?}"
        );
        let log_id = log_id.expect("should locate the committed log id");

        let asset_link_statement = conn
            .prepare("SELECT * FROM dolt_diff_gen_operation_assets(?1, ?2)")
            .expect("should prepare operation asset diff query");
        let asset_link_columns = asset_link_statement
            .column_names()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            asset_link_columns,
            vec![
                "to_log_id".to_string(),
                "to_asset_ref_id".to_string(),
                "to_role".to_string(),
                "to_commit".to_string(),
                "to_commit_date".to_string(),
                "from_log_id".to_string(),
                "from_asset_ref_id".to_string(),
                "from_role".to_string(),
                "from_commit".to_string(),
                "from_commit_date".to_string(),
                "diff_type".to_string(),
            ],
            "dolt_diff_gen_operation_assets should preserve the expected row contract"
        );
        let mut asset_ref_statement = conn
            .prepare("SELECT * FROM dolt_diff_gen_asset_refs(?1, ?2)")
            .expect("should prepare asset ref diff query");
        let asset_ref_columns = asset_ref_statement
            .column_names()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            asset_ref_columns,
            vec![
                "to_id".to_string(),
                "to_uri".to_string(),
                "to_file_type".to_string(),
                "to_checksum".to_string(),
                "to_size".to_string(),
                "to_role".to_string(),
                "to_logical_path".to_string(),
                "to_name".to_string(),
                "to_created_on".to_string(),
                "to_commit".to_string(),
                "to_commit_date".to_string(),
                "from_id".to_string(),
                "from_uri".to_string(),
                "from_file_type".to_string(),
                "from_checksum".to_string(),
                "from_size".to_string(),
                "from_role".to_string(),
                "from_logical_path".to_string(),
                "from_name".to_string(),
                "from_created_on".to_string(),
                "from_commit".to_string(),
                "from_commit_date".to_string(),
                "diff_type".to_string(),
            ],
            "dolt_diff_gen_asset_refs should preserve the expected row contract"
        );
        let asset_ref_rows = asset_ref_statement
            .query_map(["HEAD~1", "HEAD"], |row| row.get::<_, String>(22))
            .expect("should query asset ref diffs")
            .collect::<Result<Vec<_>, _>>()
            .expect("should decode asset ref diff rows");
        assert!(
            asset_ref_rows.iter().any(|diff_type| diff_type == "added"),
            "annotation-file commit should add an asset ref row: {asset_ref_rows:?}"
        );

        let committed_assets = conn
            .prepare(
                "SELECT asset_refs.file_type, asset_refs.role, asset_refs.name \
                 FROM gen_operation_assets operation_assets \
                 JOIN gen_asset_refs asset_refs ON asset_refs.id = operation_assets.asset_ref_id \
                 WHERE operation_assets.log_id = ?1 \
                 ORDER BY asset_refs.name, asset_refs.role",
            )
            .expect("should prepare committed asset lookup")
            .query_map([log_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .expect("should query committed asset rows")
            .collect::<Result<Vec<_>, _>>()
            .expect("should decode committed asset rows");
        assert!(
            committed_assets.iter().any(|(file_type, role, name)| {
                file_type == "gff3"
                    && role == "annotation"
                    && name.as_deref() == Some("fixture-track")
            }),
            "committed log should resolve to the annotation asset metadata: {committed_assets:?}"
        );
    }

    #[test]
    fn test_connect_branch_restores_clean_status_after_reopen() {
        let temp_dir = tempdir().expect("should create temporary repo directory");
        let db_path = temp_dir.path().join("default.db");
        let conn = open_existing_connection(&db_path);

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

        let reopened = open_existing_connection(&db_path);
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
        let conn = open_existing_connection(&db_path);

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

        let copied = open_existing_connection(&copied_path);
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
