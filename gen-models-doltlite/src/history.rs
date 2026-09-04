//! Doltlite-backed history primitives for the graph database.
//!
//! The following Dolt primitives are still unsupported at the Gen layer:
//! `dolt_rebase`
//! `dolt_revert`
//! `dolt_tag`
//! conflict-resolution helpers beyond read-only inspection through
//! `HistoryStore::conflicts()`
//!
//! That means:
//! Gen can detect merge/apply conflicts, but it does not yet expose a dedicated
//! conflict-resolution workflow.
//! Gen does not offer CLI flows for rebase, revert, or tags.
//! Any new history behavior should extend this module first rather than issuing
//! raw Dolt SQL from callers.

use gen_core::{BranchName, CommitRef, DoltHashId};
use rusqlite::Result as SqlResult;
use thiserror::Error;

use crate::db::GraphConnection;

pub mod dolt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HistoryEntry {
    pub commit_hash: DoltHashId,
    pub parent_hash: Option<DoltHashId>,
    pub committer: String,
    pub email: String,
    pub date: String,
    pub message: String,
    pub is_head: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HistoryStatus {
    pub table_name: String,
    pub staged: bool,
    pub status: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HistoryConflict {
    pub table_name: String,
    pub num_conflicts: i64,
}

#[derive(Debug, Error)]
pub enum HistoryError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("Commit reference '{0}' is ambiguous")]
    AmbiguousReference(String),
    #[error("No commit resolved for '{0}'")]
    UnresolvedReference(String),
}

pub type HistoryResult<T> = Result<T, HistoryError>;

pub trait HistoryStore {
    fn current_head(&self) -> SqlResult<Option<DoltHashId>>;
    fn current_branch(&self) -> SqlResult<Option<BranchName>>;
    fn resolve_ref(&self, reference: &CommitRef) -> HistoryResult<Option<DoltHashId>>;
    fn resolve_operation_hash(&self, reference: &CommitRef) -> HistoryResult<DoltHashId>;
    fn log_for_ref(
        &self,
        reference: &CommitRef,
        limit: Option<usize>,
    ) -> HistoryResult<Vec<HistoryEntry>>;
    fn log(&self, limit: Option<usize>) -> SqlResult<Vec<HistoryEntry>>;
    fn status(&self) -> SqlResult<Vec<HistoryStatus>>;
    fn conflicts(&self) -> SqlResult<Vec<HistoryConflict>>;
    fn commit_exists(&self, commit_hash: &DoltHashId) -> SqlResult<bool>;
    fn commit_all(&self, message: &str) -> SqlResult<DoltHashId>;
    fn checkout_branch(&self, branch_name: &BranchName) -> SqlResult<()>;
    fn checkout_commit(&self, commit_hash: &DoltHashId) -> SqlResult<()>;
    fn create_branch(
        &self,
        branch_name: &BranchName,
        start_ref: Option<&CommitRef>,
    ) -> SqlResult<()>;
    fn delete_branch(&self, branch_name: &BranchName) -> SqlResult<()>;
    fn merge_base(&self, source: &CommitRef, target: &CommitRef) -> SqlResult<DoltHashId>;
    fn merge(&self, reference: &CommitRef) -> SqlResult<()>;
    fn cherry_pick(&self, commit_hash: &DoltHashId) -> SqlResult<()>;
    fn reset_hard(&self, target: &CommitRef) -> SqlResult<()>;
    fn push(&self, remote_name: &str, branch_name: &BranchName) -> SqlResult<()>;
    fn pull(&self, remote_name: &str, branch_name: &BranchName) -> SqlResult<()>;
    fn fetch(&self, remote_name: &str, branch_name: Option<&BranchName>) -> SqlResult<()>;
    fn graph(&self) -> &GraphConnection;
}
