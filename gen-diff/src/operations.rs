//! Resolves history ranges and coordinates construction of operation graph diffs.
//!
//! [`collect_operation_diff`] is the public entry point used by
//! `gen view-diff`, operation history, patch previews, and GenHub. This module
//! determines which revisions and operations a request represents, asks Dolt
//! which database rows changed, groups those rows by block group, and delegates
//! graph construction to [`crate::graph`]. It does not perform a merge or
//! mutate either revision.
//!
//! # Comparison endpoints
//!
//! In merge terminology, the **source** branch supplies incoming changes and
//! the **target** branch receives them. The `source_hash` and `target_hash`
//! parameter names here predate that convention and instead describe the
//! direction of the graph comparison: `source_hash` is the baseline/left state
//! and `target_hash` is the changed/right state that will be rendered. A
//! merge-oriented caller therefore passes the receiving merge target as
//! `source_hash` and the incoming merge source as `target_hash`.
//!
//! [`DiffRange::TwoDot`] compares those two materialized endpoint states
//! directly. [`DiffRange::ThreeDot`] replaces the graph baseline with their
//! merge base and still displays the changes leading to `target_hash`. This is
//! why a three-dot call returns target-side operations even though the two
//! named branches may have diverged. A missing `source_hash` represents the
//! parent of a first/root operation and allows that operation to be shown
//! against an empty prior state.
//!
//! # Overall approach
//!
//! The pipeline deliberately uses table diffs and endpoint graphs for different
//! purposes:
//!
//! 1. Resolve the requested commits, merge base when needed, and the operation
//!    hashes represented by the comparison.
//! 2. Query `dolt_diff_block_groups`, `dolt_diff_block_group_edges`, and
//!    `dolt_diff_nodes`. These rows identify affected block-group IDs and carry
//!    old/new keys plus operation attribution without scanning every graph.
//! 3. Group those changes in `BlockGroupChanges` and build each affected block
//!    group once through `crate::graph::build_block_group_diff`.
//! 4. Return one [`OperationDiff`] containing history metadata and unified
//!    [`BlockGroupDiff`] graphs for rendering.
//!
//! Table rows alone are awkward display objects: they omit unchanged path
//! context, shared sequence-slice boundaries, and the connected terminal nodes
//! required by graph layout. Conversely, comparing every complete graph would
//! find structural changes but lose Dolt's efficient affected-row discovery and
//! operation attribution. Combining the two lets table diffs answer “what
//! changed and in which operation?” while endpoint snapshots answer “what
//! connected biological graph should be shown?”
//!
//! A viable alternative is to reconstruct the changed endpoint by applying
//! `block_group_edges` diff rows to the baseline graph. New child block groups
//! have complete copied membership rows, so this does not require a fallback
//! query. The current graph module loads both selected endpoint graphs instead,
//! primarily to keep real path membership and edit-site marker filtering
//! explicit. The lineage-parent selection used for a child block group is a
//! separate presentation decision explained in [`crate::graph`], not a claim
//! that the child's stored rows are incomplete.

use std::collections::{HashMap, HashSet};

use gen_core::{CommitRef, DoltHashId, HashId, errors::ConfigError};
use gen_models::{
    block_group_edge::AugmentedEdge,
    db::GraphConnection,
    errors::OperationError,
    history::{HistoryError, HistoryStore, dolt::DoltHistoryStore},
};
use thiserror::Error;

pub use crate::graph::{BlockGroupChangeKind, BlockGroupDiff};
use crate::graph::{BlockGroupDiffInputs, build_block_group_diff};

#[derive(Debug, Error)]
pub enum OperationDiffError {
    #[error("No current operation is checked out.")]
    NoCurrentOperation,
    #[error("Operation {0} not found.")]
    OperationMissing(DoltHashId),
    #[error("Unable to find path between {0} and {1}.")]
    PathNotFound(DoltHashId, DoltHashId),
    #[error("Config error: {0}")]
    Config(#[from] ConfigError),
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    OperationError(#[from] OperationError),
    #[error(transparent)]
    History(#[from] HistoryError),
}

/// Complete result of comparing two operation-history revisions.
///
/// Command and view code uses `operations` to identify the represented history and `diff_graph` to
/// render the consolidated changes for each affected block group.
#[derive(Clone, Debug)]
pub struct OperationDiff {
    pub operations: Vec<DoltHashId>,
    pub diff_graph: Vec<BlockGroupDiff>,
}

/// Selects the Git-style revision range semantics used to build a diff.
///
/// Given `source` and `target`, two-dot syntax (`source..target`) compares the
/// materialized source snapshot directly with the materialized target snapshot.
/// If the refs diverged, the result therefore reflects how both endpoint states
/// differ from one another.
///
/// Three-dot syntax (`source...target`) first finds the merge base, meaning the
/// best common ancestor commit from which both refs descended. It then uses
/// that common snapshot as the graph baseline and compares it with `target`.
/// This isolates the target-side changes made since the refs diverged, which is
/// what merge previews and branch-focused reviews generally need.
///
/// `collect_operation_diff` resolves these semantics before asking Dolt for
/// table changes and building the block-group graphs consumed by CLI, TUI,
/// patch-preview, and GenHub callers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffRange {
    /// Compare the source and target endpoint snapshots directly (`source..target`).
    TwoDot,
    /// Compare the source/target merge base with the target (`source...target`).
    ThreeDot,
}

impl DiffRange {
    fn between(self, source_ref: &str, target_ref: &str) -> String {
        let separator = match self {
            Self::TwoDot => "..",
            Self::ThreeDot => "...",
        };
        format!("{source_ref}{separator}{target_ref}")
    }
}

impl OperationDiff {
    fn empty() -> Self {
        Self {
            operations: Vec::new(),
            diff_graph: Vec::new(),
        }
    }
}

/// Builds the complete graph diff between two operation commits.
///
/// Returns the operations represented by the comparison and one consolidated graph diff for each
/// affected block group. The command and view layers use this as the complete comparison result.
pub fn collect_operation_diff(
    graph_conn: &GraphConnection,
    source_hash: Option<DoltHashId>,
    target_hash: DoltHashId,
    range: DiffRange,
) -> Result<OperationDiff, OperationDiffError> {
    let history_store = DoltHistoryStore::new(graph_conn);
    let target_ref = target_hash.to_string();
    let source_ref = source_hash
        .map(|hash| hash.to_string())
        .unwrap_or_else(|| format!("{target_ref}~1"));
    let (graph_source_ref, operation_start_hash) = match (range, source_hash) {
        (DiffRange::ThreeDot, Some(source_commit)) => {
            if !history_store.commit_exists(&source_commit)? {
                return Err(OperationDiffError::OperationMissing(source_commit));
            }
            if !history_store.commit_exists(&target_hash)? {
                return Err(OperationDiffError::OperationMissing(target_hash));
            }
            let merge_base = history_store.merge_base(
                &CommitRef(source_ref.clone()),
                &CommitRef(target_ref.clone()),
            )?;
            (merge_base.to_string(), Some(merge_base))
        }
        _ => (source_ref.clone(), source_hash),
    };
    let operations = operation_hashes_for_diff(&history_store, operation_start_hash, target_hash)?;
    if operations.is_empty() {
        return Ok(OperationDiff::empty());
    }

    let diff_ref = range.between(&source_ref, &target_ref);
    build_dolt_operation_diff(
        graph_conn,
        &graph_source_ref,
        &target_ref,
        &diff_ref,
        operations,
        target_hash,
    )
}

/// Returns an operation diff assembled from Dolt's table-level changes.
///
/// It groups the changed rows by block group, then builds the source and target graph states that
/// the command and view layers render for each group. What is added or removed is tracked as the
/// graph is being built, removing the need to compare graphs to figure out differences.
fn build_dolt_operation_diff(
    graph_conn: &GraphConnection,
    source_ref: &str,
    target_ref: &str,
    diff_ref: &str,
    operations: Vec<DoltHashId>,
    target_hash: DoltHashId,
) -> Result<OperationDiff, OperationDiffError> {
    let changes = collect_block_group_changes(graph_conn, diff_ref)?;
    let node_changes = collect_node_changes(graph_conn, diff_ref)?;
    let BlockGroupChanges {
        block_group_ids,
        edge_changes: edge_changes_by_block_group,
        operations_by_block_group_id: block_group_operations_by_id,
    } = changes;

    let mut block_group_ids = block_group_ids.into_iter().collect::<Vec<_>>();
    block_group_ids.sort();
    let diff_inputs = BlockGroupDiffInputs {
        source_ref,
        target_ref,
        default_operation: target_hash,
        edge_changes_by_block_group: &edge_changes_by_block_group,
        block_group_operations_by_id: &block_group_operations_by_id,
        node_changes: &node_changes,
    };
    let mut diff_graph = Vec::new();
    for block_group_id in block_group_ids {
        if let Some(block_group_diff) =
            build_block_group_diff(graph_conn, block_group_id, &diff_inputs)?
        {
            diff_graph.push(block_group_diff);
        }
    }

    Ok(OperationDiff {
        operations,
        diff_graph,
    })
}

/// Returns the operation hashes represented by a comparison.
///
/// For an ancestor range, the output is the target-side history after `source_hash`. For divergent
/// endpoints, it contains the source and target commits. Consumers use the result as comparison
/// metadata and to detect a comparison with no changes. Importantly, this just returns 0, 1, or 2
/// hashes indicating no changes, a target_hash, or two hashes marking the start/end of the diff.
fn operation_hashes_for_diff(
    history_store: &impl HistoryStore,
    source_hash: Option<DoltHashId>,
    target_hash: DoltHashId,
) -> Result<Vec<DoltHashId>, OperationDiffError> {
    if source_hash == Some(target_hash) {
        return Ok(Vec::new());
    }

    let target_ref = target_hash.to_string();
    let history_entries = history_store.log_for_ref(&CommitRef(target_ref), None)?;
    let history_hashes = history_entries
        .into_iter()
        .rev()
        .map(|entry| entry.commit_hash)
        .collect::<Vec<_>>();
    let target_index = history_hashes
        .iter()
        .position(|candidate| *candidate == target_hash)
        .ok_or(OperationDiffError::OperationMissing(target_hash))?;

    let operations = match source_hash {
        None => vec![target_hash],
        Some(source_hash) => match history_hashes
            .iter()
            .position(|candidate| *candidate == source_hash)
        {
            Some(source_index) if source_index > target_index => {
                return Err(OperationDiffError::PathNotFound(source_hash, target_hash));
            }
            Some(source_index) => history_hashes[source_index + 1..=target_index].to_vec(),
            None if history_store.commit_exists(&source_hash)? => {
                vec![source_hash, target_hash]
            }
            None => return Err(OperationDiffError::OperationMissing(source_hash)),
        },
    };
    Ok(operations)
}

/// Returns the affected block groups, their edge changes, and their operation attribution.
///
/// `build_dolt_operation_diff` consumes this aggregate to decide which block-group graphs to build
/// and how to match and attribute edge changes within each graph.
fn collect_block_group_changes(
    graph_conn: &GraphConnection,
    diff_ref: &str,
) -> Result<BlockGroupChanges, OperationDiffError> {
    let mut changes = BlockGroupChanges::default();
    let block_group_query = format!(
        "SELECT to_id,
                from_id,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                to_commit,
                diff_type
         FROM dolt_diff_block_groups('{diff_ref}')"
    );
    let mut block_group_statement = graph_conn.prepare(&block_group_query)?;
    let block_group_rows =
        block_group_statement.query_map([], DoltBlockGroupChangeRow::from_row)?;
    for row in block_group_rows {
        let row = row?;
        row.record_block_group(&mut changes);
    }

    let edge_query = format!(
        "SELECT to_block_group_id,
                from_block_group_id,
                to_edge_id,
                from_edge_id,
                to_chromosome_index,
                from_chromosome_index,
                to_phased,
                from_phased,
                to_commit,
                diff_type
         FROM dolt_diff_block_group_edges('{diff_ref}')"
    );
    let mut edge_statement = graph_conn.prepare(&edge_query)?;
    let edge_rows = edge_statement.query_map([], DoltBlockGroupChangeRow::from_row)?;
    for row in edge_rows {
        let row = row?;
        row.record_block_group(&mut changes);
        if let Some(edge_change) = row.edge_change() {
            changes
                .edge_changes
                .entry(edge_change.block_group_id())
                .or_default()
                .push(edge_change);
        }
    }

    Ok(changes)
}

/// Returns the added, removed, and modified node-ID transitions for a revision range.
///
/// Graph construction uses these transitions to match source nodes with target nodes and to
/// attribute the resulting graph changes to operations.
fn collect_node_changes(
    graph_conn: &GraphConnection,
    diff_ref: &str,
) -> Result<Vec<NodeChange>, OperationDiffError> {
    let query =
        format!("SELECT to_id, from_id, to_commit, diff_type FROM dolt_diff_nodes('{diff_ref}')");
    let mut node_changes = Vec::new();
    let mut statement = graph_conn.prepare(&query)?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, Option<HashId>>(0)?,
            row.get::<_, Option<HashId>>(1)?,
            row.get::<_, Option<DoltHashId>>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (to_node_id, from_node_id, to_commit, diff_type) = row?;
        let operation = to_commit;
        match diff_type.as_str() {
            "added" => {
                if let Some(new_node_id) = to_node_id {
                    node_changes.push(NodeChange::Added {
                        new_node_id,
                        operation,
                    });
                }
            }
            "removed" => {
                if let Some(old_node_id) = from_node_id {
                    node_changes.push(NodeChange::Removed {
                        old_node_id,
                        operation,
                    });
                }
            }
            "modified" => {
                if let (Some(old_node_id), Some(new_node_id)) = (from_node_id, to_node_id) {
                    node_changes.push(NodeChange::Modified {
                        old_node_id,
                        new_node_id,
                        old_operation: operation,
                        new_operation: operation,
                    });
                }
            }
            _ => {}
        }
    }

    Ok(node_changes)
}

/// Table-level changes collected before block-group graph construction.
///
/// This groups Dolt diff rows by block group so graph construction can determine which graphs to
/// build, which edges changed, and which operations should be used for attribution.
#[derive(Debug, Default)]
struct BlockGroupChanges {
    /// Every block group affected by a block-group or edge diff row.
    block_group_ids: HashSet<HashId>,
    /// Normalized edge changes keyed by their owning block group.
    edge_changes: HashMap<HashId, Vec<BlockGroupEdgeChange>>,
    /// Operations associated with each affected block group.
    operations_by_block_group_id: HashMap<HashId, HashSet<DoltHashId>>,
}

impl BlockGroupChanges {
    fn record_operation(&mut self, block_group_id: HashId, operation: Option<DoltHashId>) {
        if let Some(operation) = operation {
            self.operations_by_block_group_id
                .entry(block_group_id)
                .or_default()
                .insert(operation);
        }
    }
}

/// Normalized result row shared by the block-group and block-group-edge Dolt diff queries.
///
/// The optional source (`from_*`) and target (`to_*`) columns are converted into affected block
/// groups and `BlockGroupEdgeChange` values by the collection step.
#[derive(Debug)]
struct DoltBlockGroupChangeRow {
    /// Target-side block-group ID, absent when the block group was removed.
    to_block_group_id: Option<HashId>,
    /// Source-side block-group ID, absent when the block group was added.
    from_block_group_id: Option<HashId>,
    /// Target-side edge ID for edge rows.
    to_edge_id: Option<HashId>,
    /// Source-side edge ID for edge rows.
    from_edge_id: Option<HashId>,
    /// Target-side chromosome index for edge rows.
    to_chromosome_index: Option<i64>,
    /// Source-side chromosome index for edge rows.
    from_chromosome_index: Option<i64>,
    /// Target-side phasing value for edge rows.
    to_phased: Option<i64>,
    /// Source-side phasing value for edge rows.
    from_phased: Option<i64>,
    /// Commit Dolt reports for attributing the change.
    to_commit: Option<DoltHashId>,
    /// Dolt change classification: added, removed, or modified.
    diff_type: String,
}

impl DoltBlockGroupChangeRow {
    /// Decodes the normalized column shape shared by the block-group and edge diff queries.
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            to_block_group_id: row.get(0)?,
            from_block_group_id: row.get(1)?,
            to_edge_id: row.get(2)?,
            from_edge_id: row.get(3)?,
            to_chromosome_index: row.get(4)?,
            from_chromosome_index: row.get(5)?,
            to_phased: row.get(6)?,
            from_phased: row.get(7)?,
            to_commit: row.get(8)?,
            diff_type: row.get(9)?,
        })
    }

    /// Adds this row's affected block group and operation to the aggregate.
    fn record_block_group(&self, changes: &mut BlockGroupChanges) {
        if matches!(self.diff_type.as_str(), "added" | "removed" | "modified")
            && let Some(block_group_id) = self.to_block_group_id.or(self.from_block_group_id)
        {
            changes.block_group_ids.insert(block_group_id);
            changes.record_operation(block_group_id, self.change_operation());
        }
    }

    /// Returns the source/target edge keys represented by this diff row.
    ///
    /// The graph builder uses the result to match the row with the corresponding edges in the
    /// endpoint graphs. Rows without a recognized change type or complete edge key return `None`.
    fn edge_change(&self) -> Option<BlockGroupEdgeChange> {
        match self.diff_type.as_str() {
            "added" => {
                let block_group_id = self.to_block_group_id?;
                let new_key =
                    self.edge_key(self.to_edge_id, self.to_chromosome_index, self.to_phased)?;
                Some(BlockGroupEdgeChange {
                    block_group_id,
                    old_key: None,
                    new_key: Some(new_key),
                    old_operation: None,
                    new_operation: self.change_operation(),
                })
            }
            "removed" => {
                let block_group_id = self.from_block_group_id?;
                let old_key = self.edge_key(
                    self.from_edge_id,
                    self.from_chromosome_index,
                    self.from_phased,
                )?;
                Some(BlockGroupEdgeChange {
                    block_group_id,
                    old_key: Some(old_key),
                    new_key: None,
                    old_operation: self.change_operation(),
                    new_operation: None,
                })
            }
            "modified" => {
                let block_group_id = self.to_block_group_id.or(self.from_block_group_id)?;
                let old_key = self.edge_key(
                    self.from_edge_id,
                    self.from_chromosome_index,
                    self.from_phased,
                )?;
                let new_key =
                    self.edge_key(self.to_edge_id, self.to_chromosome_index, self.to_phased)?;
                Some(BlockGroupEdgeChange {
                    block_group_id,
                    old_key: Some(old_key),
                    new_key: Some(new_key),
                    old_operation: self.change_operation(),
                    new_operation: self.change_operation(),
                })
            }
            _ => None,
        }
    }

    /// Returns the stable identity of one side of an edge row when all key columns are present.
    fn edge_key(
        &self,
        edge_id: Option<HashId>,
        chromosome_index: Option<i64>,
        phased: Option<i64>,
    ) -> Option<BlockGroupEdgeKey> {
        Some(BlockGroupEdgeKey {
            edge_id: edge_id?,
            chromosome_index: chromosome_index?,
            phased: phased?,
        })
    }

    /// Returns the `to_commit` value Dolt supplies to attribute this row's change.
    fn change_operation(&self) -> Option<DoltHashId> {
        self.to_commit
    }
}

/// Stable identity of a block-group edge on one side of a comparison.
///
/// Graph construction uses this key to match edges loaded from an endpoint graph with rows returned
/// by Dolt's edge diff table.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BlockGroupEdgeKey {
    edge_id: HashId,
    chromosome_index: i64,
    phased: i64,
}

impl From<&AugmentedEdge> for BlockGroupEdgeKey {
    /// Returns the stable key used to match a graph edge with a Dolt edge-diff row.
    fn from(edge: &AugmentedEdge) -> Self {
        Self {
            edge_id: edge.edge.id,
            chromosome_index: edge.chromosome_index,
            phased: edge.phased,
        }
    }
}

/// Source/target node IDs and operation attribution for one Dolt node diff row.
///
/// Graph construction uses these values to identify nodes present on only one endpoint and annotate
/// their resulting graph changes with the responsible operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NodeChange {
    Added {
        new_node_id: HashId,
        operation: Option<DoltHashId>,
    },
    Removed {
        old_node_id: HashId,
        operation: Option<DoltHashId>,
    },
    Modified {
        old_node_id: HashId,
        new_node_id: HashId,
        old_operation: Option<DoltHashId>,
        new_operation: Option<DoltHashId>,
    },
}

impl NodeChange {
    /// Returns the source-side node ID, or `None` when the node exists only in the target.
    pub(crate) fn old_node_id(&self) -> Option<HashId> {
        match self {
            NodeChange::Added { .. } => None,
            NodeChange::Removed { old_node_id, .. } => Some(*old_node_id),
            NodeChange::Modified { old_node_id, .. } => Some(*old_node_id),
        }
    }

    /// Returns the target-side node ID, or `None` when the node exists only in the source.
    pub(crate) fn new_node_id(&self) -> Option<HashId> {
        match self {
            NodeChange::Added { new_node_id, .. } => Some(*new_node_id),
            NodeChange::Removed { .. } => None,
            NodeChange::Modified { new_node_id, .. } => Some(*new_node_id),
        }
    }

    /// Returns the operation attributed to the source-side half of this node change.
    pub(crate) fn old_operation(&self) -> Option<DoltHashId> {
        match self {
            NodeChange::Added { .. } => None,
            NodeChange::Removed { operation, .. } => *operation,
            NodeChange::Modified { old_operation, .. } => *old_operation,
        }
    }

    /// Returns the operation attributed to the target-side half of this node change.
    pub(crate) fn new_operation(&self) -> Option<DoltHashId> {
        match self {
            NodeChange::Added { operation, .. } => *operation,
            NodeChange::Removed { .. } => None,
            NodeChange::Modified { new_operation, .. } => *new_operation,
        }
    }
}

/// Source/target edge identities and operation attribution for one Dolt edge diff row.
///
/// Graph construction uses this to match endpoint edges, filter changes relevant to a block group,
/// and annotate the resulting graph changes with the responsible operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BlockGroupEdgeChange {
    block_group_id: HashId,
    /// Source-side edge identity, absent for an added edge.
    pub(crate) old_key: Option<BlockGroupEdgeKey>,
    /// Target-side edge identity, absent for a removed edge.
    pub(crate) new_key: Option<BlockGroupEdgeKey>,
    /// Operation attributed to the source-side half of the change.
    old_operation: Option<DoltHashId>,
    /// Operation attributed to the target-side half of the change.
    new_operation: Option<DoltHashId>,
}

impl BlockGroupEdgeChange {
    /// Returns the block group whose edge set contains this change.
    fn block_group_id(&self) -> HashId {
        self.block_group_id
    }

    /// Returns the source-side edge ID, or `None` for an added edge.
    pub(crate) fn old_edge_id(&self) -> Option<HashId> {
        self.old_key.map(|key| key.edge_id)
    }

    /// Returns the target-side edge ID, or `None` for a removed edge.
    pub(crate) fn new_edge_id(&self) -> Option<HashId> {
        self.new_key.map(|key| key.edge_id)
    }

    /// Returns the operation attributed to the source-side half of this edge change.
    pub(crate) fn old_operation(&self) -> Option<DoltHashId> {
        self.old_operation
    }

    /// Returns the operation attributed to the target-side half of this edge change.
    pub(crate) fn new_operation(&self) -> Option<DoltHashId> {
        self.new_operation
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use gen_core::{
        BranchName, CommitRef, DoltHashId, HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand,
        Workspace,
    };
    use gen_graph::{GenGraph, GraphEdge, GraphNode};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{AugmentedEdge, BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::DbContext,
        edge::{Edge, GroupBlock},
        history::{HistoryStore, dolt::DoltHistoryStore},
        node::Node,
        path::Path,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use petgraph::Direction;
    use rusqlite::params;
    use tempfile::tempdir;

    use super::{
        BlockGroupDiff, BlockGroupEdgeChange, BlockGroupEdgeKey, DiffRange, OperationDiffError,
        collect_block_group_changes, collect_operation_diff,
    };
    use crate::{
        graph::{
            AnnotatedStageGraph, BlockGroupStageGraphs, DiffChangeKind, DiffGenGraph, GraphEdgeKey,
            build_stage_graph_from_nodes, build_unified_diff_graph, path_end_graph_node,
            path_start_graph_node,
        },
        test_helpers::{get_config_connection, get_connection as test_get_connection},
    };

    fn setup_history_context() -> DbContext {
        let tmp_dir = tempdir().expect("should create temp dir").keep();
        let workspace = Workspace::new(tmp_dir);
        workspace.ensure_gen_dir();
        let graph_conn = test_get_connection(workspace.graph_db_path().unwrap().to_str().unwrap())
            .expect("should create graph database");
        let config_conn = get_config_connection(workspace.gen_db_path().unwrap().to_str().unwrap())
            .expect("should create config database");
        DbContext::new(workspace, graph_conn, config_conn).expect("should create db context")
    }

    fn create_linear_block_group(
        context: &gen_models::db::DbContext,
        collection_name: &str,
        sample_name: &str,
        block_group_name: &str,
        node_name: &str,
        sequence_bases: &str,
    ) -> BlockGroup {
        let graph_conn = context.graph().conn();
        Collection::get_or_create(graph_conn, collection_name).expect("should create collection");
        Sample::get_or_create(
            graph_conn,
            NewSample {
                name: sample_name,
                is_reference: false,
            },
        )
        .expect("should create sample");
        let block_group = BlockGroup::create(
            graph_conn,
            NewBlockGroup {
                collection_name,
                sample_name,
                name: block_group_name,
                ..Default::default()
            },
        )
        .expect("should create block group");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence_bases)
            .name(node_name)
            .save(graph_conn)
            .expect("should save sequence");
        let node_id = Node::create(
            graph_conn,
            &sequence.hash,
            &HashId::convert_str(&format!("{block_group_name}-{node_name}")),
        )
        .expect("should create node");
        let start_edge = Edge::create(
            graph_conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )
        .expect("should create start edge");
        let end_edge = Edge::create(
            graph_conn,
            node_id,
            sequence.length,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create end edge");
        BlockGroupEdge::bulk_create(
            graph_conn,
            &[
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: start_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: end_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                },
            ],
        );
        Path::create(
            graph_conn,
            block_group_name,
            &block_group.id,
            &[start_edge.id, end_edge.id],
        )
        .expect("should create path");
        block_group
    }

    fn replace_linear_block_group_path(
        context: &gen_models::db::DbContext,
        block_group: &BlockGroup,
        node_name: &str,
        sequence_bases: &str,
    ) {
        let graph_conn = context.graph().conn();
        let existing_edges =
            BlockGroupEdge::edges_for_block_group(graph_conn, &block_group.id, None)
                .into_iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.edge.id,
                    chromosome_index: edge.chromosome_index,
                    phased: edge.phased,
                })
                .collect::<Vec<_>>();
        Path::delete(graph_conn, &block_group.name, &block_group.id);
        BlockGroupEdge::bulk_delete(graph_conn, &existing_edges);

        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence_bases)
            .name(node_name)
            .save(graph_conn)
            .expect("should save replacement sequence");
        let node_id = Node::create(
            graph_conn,
            &sequence.hash,
            &HashId::convert_str(&format!("{}-{node_name}", block_group.name)),
        )
        .expect("should create replacement node");
        let start_edge = Edge::create(
            graph_conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )
        .expect("should create replacement start edge");
        let end_edge = Edge::create(
            graph_conn,
            node_id,
            sequence.length,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create replacement end edge");
        BlockGroupEdge::bulk_create(
            graph_conn,
            &[
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: start_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: end_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                },
            ],
        );
        Path::create(
            graph_conn,
            &block_group.name,
            &block_group.id,
            &[start_edge.id, end_edge.id],
        )
        .expect("should recreate path");
    }

    fn commit_operation(context: &gen_models::db::DbContext, message: &str) -> DoltHashId {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        history_store
            .commit_all(message)
            .expect("should commit graph changes")
    }

    fn commit_hash_ref(operation_hash: DoltHashId) -> CommitRef {
        CommitRef(operation_hash.to_string())
    }

    struct DivergentHistory {
        context: DbContext,
        source_block_group: BlockGroup,
        target_block_group: BlockGroup,
        source_hash: DoltHashId,
        target_hash: DoltHashId,
    }

    // Builds two branches with one unique block group on each side:
    //
    //                    source branch: source_hash (+ source_block_group)
    //                   /
    //     base_hash ----*
    //                   \
    //                    main branch:   target_hash (+ target_block_group)
    //
    // A two-dot diff compares the endpoint states, so the source block group is
    // removed and the target block group is added. A three-dot diff compares the
    // common base with the target, so only the target block group is added.
    fn setup_divergent_history() -> DivergentHistory {
        let mut context = setup_history_context();
        let graph_conn = context.graph().conn();
        let graph_db_path = context
            .workspace()
            .graph_db_path()
            .expect("should resolve graph database path");
        let history_store = DoltHistoryStore::new(graph_conn);

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let source_block_group =
            create_linear_block_group(&context, "c", "s", "main", "main", "CCCCC");
        let source_hash = commit_operation(&context, "main op");
        history_store
            .create_branch(
                &BranchName("source".to_string()),
                Some(&commit_hash_ref(source_hash)),
            )
            .expect("should preserve the source branch");
        history_store
            .reset_hard(&commit_hash_ref(base))
            .expect("should reset main to the common base");
        let reopened_graph = gen_models::db::get_connection(&graph_db_path)
            .expect("should reopen graph database after reset");
        gen_models::history::dolt::connect_branch(&reopened_graph, "main")
            .expect("should reconnect the main branch");
        context.set_graph(reopened_graph);
        let target_block_group =
            create_linear_block_group(&context, "c", "s", "feature", "feature", "GGGGG");
        let target_hash = commit_operation(&context, "feature op");

        DivergentHistory {
            context,
            source_block_group,
            target_block_group,
            source_hash,
            target_hash,
        }
    }

    fn switch_branch(context: &gen_models::db::DbContext, branch_name: &str) {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        history_store
            .checkout_branch(&BranchName(branch_name.to_string()))
            .expect("should checkout history branch");
    }

    fn create_feature_branch(
        context: &gen_models::db::DbContext,
        base_operation_hash: DoltHashId,
        branch_name: &str,
    ) {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        history_store
            .create_branch(
                &BranchName(branch_name.to_string()),
                Some(&commit_hash_ref(base_operation_hash)),
            )
            .expect("should create history branch");
        switch_branch(context, branch_name);
    }

    fn test_augmented_edge(
        edge_id: HashId,
        source_node_id: HashId,
        source_coordinate: i64,
        target_node_id: HashId,
        target_coordinate: i64,
    ) -> AugmentedEdge {
        AugmentedEdge {
            edge: Edge {
                id: edge_id,
                source_node_id,
                source_coordinate,
                source_strand: Strand::Forward,
                target_node_id,
                target_coordinate,
                target_strand: Strand::Forward,
            },
            chromosome_index: 0,
            phased: 0,
            created_on: 0,
        }
    }

    fn apply_edge_changes(
        graph_conn: &gen_models::db::GraphConnection,
        source_edges: &[AugmentedEdge],
        edge_changes: &[BlockGroupEdgeChange],
        target_ref: &str,
    ) -> Result<Vec<AugmentedEdge>, OperationDiffError> {
        let mut edges_by_key = source_edges
            .iter()
            .cloned()
            .map(|edge| (BlockGroupEdgeKey::from(&edge), edge))
            .collect::<HashMap<_, _>>();

        for edge_change in edge_changes {
            if let Some(old_key) = edge_change.old_key {
                edges_by_key.remove(&old_key);
            }
            if let Some(new_key) = edge_change.new_key {
                let edge = load_augmented_edge(
                    graph_conn,
                    edge_change.block_group_id,
                    new_key,
                    target_ref,
                )?;
                edges_by_key.insert(new_key, edge);
            }
        }

        let mut edges = edges_by_key.into_values().collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            right
                .created_on
                .cmp(&left.created_on)
                .then_with(|| left.edge.id.cmp(&right.edge.id))
                .then_with(|| left.chromosome_index.cmp(&right.chromosome_index))
                .then_with(|| left.phased.cmp(&right.phased))
        });
        Ok(edges)
    }

    fn load_augmented_edge(
        graph_conn: &gen_models::db::GraphConnection,
        block_group_id: HashId,
        edge_key: BlockGroupEdgeKey,
        history_ref: &str,
    ) -> Result<AugmentedEdge, OperationDiffError> {
        let query = "SELECT
                edges.id,
                edges.source_node_id,
                edges.source_coordinate,
                edges.source_strand,
                edges.target_node_id,
                edges.target_coordinate,
                edges.target_strand,
                block_group_edges.chromosome_index,
                block_group_edges.phased,
                block_group_edges.created_on
            FROM dolt_at_block_group_edges(?1) AS block_group_edges
            JOIN dolt_at_edges(?1) AS edges ON edges.id = block_group_edges.edge_id
            WHERE block_group_edges.block_group_id = ?2
              AND block_group_edges.edge_id = ?3
              AND block_group_edges.chromosome_index = ?4
              AND block_group_edges.phased = ?5";
        graph_conn
            .query_row(
                query,
                params![
                    history_ref,
                    block_group_id,
                    edge_key.edge_id,
                    edge_key.chromosome_index,
                    edge_key.phased
                ],
                |row| {
                    Ok(AugmentedEdge {
                        edge: Edge {
                            id: row.get(0)?,
                            source_node_id: row.get(1)?,
                            source_coordinate: row.get(2)?,
                            source_strand: row.get(3)?,
                            target_node_id: row.get(4)?,
                            target_coordinate: row.get(5)?,
                            target_strand: row.get(6)?,
                        },
                        chromosome_index: row.get(7)?,
                        phased: row.get(8)?,
                        created_on: row.get(9)?,
                    })
                },
            )
            .map_err(OperationDiffError::from)
    }

    fn build_stage_graph(
        stage_blocks: &[GroupBlock],
        stage_edges: &[AugmentedEdge],
        node_operations_by_id: &HashMap<HashId, DoltHashId>,
        edge_operations_by_id: &HashMap<HashId, DoltHashId>,
    ) -> AnnotatedStageGraph {
        let stage_nodes = stage_blocks
            .iter()
            .map(|block| GraphNode {
                node_id: block.node_id,
                sequence_start: block.start,
                sequence_end: block.end,
            })
            .collect::<HashSet<_>>();
        let continuation_spans = stage_nodes.clone();
        build_stage_graph_from_nodes(
            stage_nodes,
            &continuation_spans,
            stage_edges,
            node_operations_by_id,
            edge_operations_by_id,
        )
    }

    fn graph_edge_keys(graph: &GenGraph) -> HashSet<GraphEdgeKey> {
        graph
            .all_edges()
            .flat_map(|(source, target, edges)| {
                edges
                    .iter()
                    .copied()
                    .map(move |edge| GraphEdgeKey::new(source, target, edge))
            })
            .collect()
    }

    fn test_graph_node(node_id: HashId, sequence_start: i64, sequence_end: i64) -> GraphNode {
        GraphNode {
            node_id,
            sequence_start,
            sequence_end,
        }
    }

    fn test_terminal_block(block_index: i64, node_id: HashId) -> GroupBlock {
        GroupBlock::new(
            block_index,
            node_id,
            &Sequence::new().sequence_type("DNA").sequence("").build(),
            0,
            0,
        )
    }

    fn assert_diff_edge_kind(
        graph: &DiffGenGraph,
        source: GraphNode,
        target: GraphNode,
        expected_kind: DiffChangeKind,
    ) {
        let source = graph
            .nodes()
            .find(|node| node.node == source)
            .expect("should contain source node");
        let target = graph
            .nodes()
            .find(|node| node.node == target)
            .expect("should contain target node");
        let edge_changes = graph
            .edge_weight(source, target)
            .expect("should contain edge between nodes");
        assert!(
            edge_changes
                .iter()
                .any(|edge| edge.change.kind == expected_kind),
            "edge should contain expected change kind {expected_kind:?}: {edge_changes:?}"
        );
    }

    fn assert_all_nodes_connected_to(graph: &DiffGenGraph, start: GraphNode) {
        let start = graph
            .nodes()
            .find(|node| node.node == start)
            .expect("should contain start node");
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            for neighbor in graph
                .neighbors_directed(current, Direction::Outgoing)
                .chain(graph.neighbors_directed(current, Direction::Incoming))
            {
                if !visited.contains(&neighbor) {
                    stack.push(neighbor);
                }
            }
        }
        assert_eq!(
            visited.len(),
            graph.nodes().count(),
            "unified diff graph should be a single connected component"
        );
    }

    #[test]
    fn test_one_operation_diff_uses_dolt_history() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base_op = commit_operation(&context, "seed");
        let expected_block_group =
            create_linear_block_group(&context, "c", "s", "bg", "one", "CCCCC");
        let head = commit_operation(&context, "add");

        let diff = collect_operation_diff(graph_conn, Some(base_op), head, DiffRange::TwoDot)
            .expect("diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, expected_block_group.id);
        assert!(diff.diff_graph[0].source_block_group.is_none());
        assert_eq!(
            diff.diff_graph[0].target_block_group.as_ref(),
            Some(&expected_block_group)
        );
        assert_block_group_graph_shape(&diff.diff_graph[0]);
    }

    #[test]
    fn test_initial_operation_diff_contains_added_block_groups_from_history() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        let expected_block_group =
            create_linear_block_group(&context, "c", "s", "bg", "one", "AAAAA");
        let head = commit_operation(&context, "add");

        let diff = collect_operation_diff(graph_conn, None, head, DiffRange::TwoDot).expect("diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, expected_block_group.id);
        assert!(diff.diff_graph[0].source_block_group.is_none());
        assert_eq!(
            diff.diff_graph[0].target_block_group.as_ref(),
            Some(&expected_block_group)
        );
        assert_block_group_graph_shape(&diff.diff_graph[0]);
    }

    #[test]
    fn test_operation_diff_rejects_missing_from_commit() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        let missing_parent_hash = DoltHashId([1; 20]);
        create_linear_block_group(&context, "c", "s", "bg", "one", "AAAAA");
        let head = commit_operation(&context, "add");

        let error = collect_operation_diff(
            graph_conn,
            Some(missing_parent_hash),
            head,
            DiffRange::TwoDot,
        )
        .expect_err("should reject a missing from commit");
        assert!(matches!(
            error,
            OperationDiffError::OperationMissing(hash) if hash == missing_parent_hash
        ));
    }

    #[test]
    fn test_one_operation_diff_returns_removed_markers_in_unified_diff_graph() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        let removed_block_group =
            create_linear_block_group(&context, "c", "s", "bg", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let block_group_edges =
            BlockGroupEdge::edges_for_block_group(graph_conn, &removed_block_group.id, None)
                .into_iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: removed_block_group.id,
                    edge_id: edge.edge.id,
                    chromosome_index: edge.chromosome_index,
                    phased: edge.phased,
                })
                .collect::<Vec<_>>();
        Path::delete(
            graph_conn,
            &removed_block_group.name,
            &removed_block_group.id,
        );
        BlockGroupEdge::bulk_delete(graph_conn, &block_group_edges);
        BlockGroup::delete(
            graph_conn,
            &removed_block_group.collection_name,
            &removed_block_group.sample_name,
            &removed_block_group.name,
        )
        .expect("should delete block group");
        let head = commit_operation(&context, "remove bg");

        let diff =
            collect_operation_diff(graph_conn, Some(base), head, DiffRange::TwoDot).expect("diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, removed_block_group.id);
        assert_eq!(
            diff.diff_graph[0].source_block_group.as_ref(),
            Some(&removed_block_group)
        );
        assert!(diff.diff_graph[0].target_block_group.is_none());
        assert!(
            diff.diff_graph[0]
                .graph
                .nodes()
                .any(|node| node.change.kind == DiffChangeKind::Removed
                    && node.change.operation == Some(head)),
            "removed nodes should be marked as removed and attributed to the removing operation"
        );
        assert!(
            diff.diff_graph[0]
                .graph
                .all_edges()
                .any(|(_, _, edges)| edges
                    .iter()
                    .any(|edge| edge.change.kind == DiffChangeKind::Removed
                        && edge.change.operation == Some(head))),
            "removed edges should be marked as removed and attributed to the removing operation"
        );
    }

    #[test]
    fn test_modified_block_group_returns_added_and_removed_markers_in_unified_diff_graph() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        let block_group = create_linear_block_group(&context, "c", "s", "bg", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        replace_linear_block_group_path(&context, &block_group, "replacement", "CCCCC");
        let head = commit_operation(&context, "replace bg path");

        let diff =
            collect_operation_diff(graph_conn, Some(base), head, DiffRange::TwoDot).expect("diff");
        assert_eq!(diff.diff_graph.len(), 1);
        assert!(diff.diff_graph[0].source_block_group.is_some());
        assert!(diff.diff_graph[0].target_block_group.is_some());
        assert!(
            diff.diff_graph[0]
                .graph
                .nodes()
                .any(|node| node.change.kind == DiffChangeKind::Added),
            "modified block group should contain added nodes"
        );
        assert!(
            diff.diff_graph[0]
                .graph
                .nodes()
                .any(|node| node.change.kind == DiffChangeKind::Removed),
            "modified block group should contain removed nodes"
        );
        assert!(
            diff.diff_graph[0]
                .graph
                .all_edges()
                .any(|(_, _, edges)| edges
                    .iter()
                    .any(|edge| edge.change.kind == DiffChangeKind::Added)),
            "modified block group should contain added edges"
        );
        assert!(
            diff.diff_graph[0]
                .graph
                .all_edges()
                .any(|(_, _, edges)| edges
                    .iter()
                    .any(|edge| edge.change.kind == DiffChangeKind::Removed)),
            "modified block group should contain removed edges"
        );
    }

    #[test]
    fn test_source_graph_plus_dolt_changes_reconstructs_target_graph() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        let block_group = create_linear_block_group(&context, "c", "s", "bg", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        replace_linear_block_group_path(&context, &block_group, "replacement", "CCCCC");
        let head = commit_operation(&context, "replace bg path");
        let source_ref = commit_hash_ref(base).0;
        let target_ref = commit_hash_ref(head).0;

        let source_edges =
            BlockGroupEdge::edges_for_block_group(graph_conn, &block_group.id, Some(&source_ref));
        let changes = collect_block_group_changes(
            graph_conn,
            &DiffRange::TwoDot.between(&source_ref, &target_ref),
        )
        .expect("should collect block group changes");
        let edge_changes = changes
            .edge_changes
            .get(&block_group.id)
            .expect("should contain edge changes for replaced path");
        let reconstructed_edges =
            apply_edge_changes(graph_conn, &source_edges, edge_changes, &target_ref)
                .expect("should apply edge changes");

        let reconstructed_blocks = Edge::blocks_from_edges(
            graph_conn,
            context.workspace(),
            &block_group.id,
            &reconstructed_edges,
            Some(&target_ref),
        )
        .expect("should build reconstructed blocks");
        let target_edges =
            BlockGroupEdge::edges_for_block_group(graph_conn, &block_group.id, Some(&target_ref));
        let target_blocks = Edge::blocks_from_edges(
            graph_conn,
            context.workspace(),
            &block_group.id,
            &target_edges,
            Some(&target_ref),
        )
        .expect("should build target blocks");
        let reconstructed_graph = build_stage_graph(
            &reconstructed_blocks,
            &reconstructed_edges,
            &HashMap::new(),
            &HashMap::new(),
        );
        let target_graph = build_stage_graph(
            &target_blocks,
            &target_edges,
            &HashMap::new(),
            &HashMap::new(),
        );

        assert_eq!(
            graph_edge_keys(&reconstructed_graph.graph),
            graph_edge_keys(&target_graph.graph)
        );
        assert_eq!(
            reconstructed_graph.graph.nodes().collect::<HashSet<_>>(),
            target_graph.graph.nodes().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn test_graph_edge_keys_ignore_created_on() {
        let source = GraphNode {
            node_id: HashId::convert_str("source"),
            sequence_start: 0,
            sequence_end: 4,
        };
        let target = GraphNode {
            node_id: HashId::convert_str("target"),
            sequence_start: 0,
            sequence_end: 4,
        };
        let edge_id = HashId::convert_str("edge");
        let mut left_graph = GenGraph::new();
        left_graph.add_edge(
            source,
            target,
            vec![GraphEdge {
                edge_id,
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 1,
            }],
        );
        let mut right_graph = GenGraph::new();
        right_graph.add_edge(
            source,
            target,
            vec![GraphEdge {
                edge_id,
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 2,
            }],
        );

        assert_eq!(
            graph_edge_keys(&left_graph),
            graph_edge_keys(&right_graph),
            "edge comparison should ignore block-group edge timestamps"
        );
    }

    #[test]
    fn test_stage_graph_keeps_endpoint_local_node_slices() {
        let operation = DoltHashId([100; 20]);
        let node_id = HashId::convert_str("reference-node");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAGTTTTTT")
            .build();
        let start_edge_id = HashId::convert_str("start-edge");
        let end_edge_id = HashId::convert_str("end-edge");
        let source_blocks = vec![GroupBlock::new(0, node_id, &sequence, 0, 10)];
        let target_blocks = vec![
            GroupBlock::new(0, node_id, &sequence, 0, 3),
            GroupBlock::new(1, node_id, &sequence, 4, 10),
        ];
        let stage_edges = vec![
            test_augmented_edge(start_edge_id, PATH_START_NODE_ID, 0, node_id, 0),
            test_augmented_edge(end_edge_id, node_id, 10, PATH_END_NODE_ID, 0),
        ];
        let source_stage_graph = build_stage_graph(
            &source_blocks,
            &stage_edges,
            &HashMap::new(),
            &HashMap::new(),
        );
        let target_stage_graph = build_stage_graph(
            &target_blocks,
            &stage_edges,
            &HashMap::new(),
            &HashMap::from([(node_id, operation)]),
        );

        let source_nodes = source_stage_graph.graph.nodes().collect::<HashSet<_>>();
        assert!(
            source_nodes.contains(&GraphNode {
                node_id,
                sequence_start: 0,
                sequence_end: 10,
            }),
            "source graph should preserve its own unsplit block"
        );
        assert!(
            !source_nodes.contains(&GraphNode {
                node_id,
                sequence_start: 3,
                sequence_end: 4,
            }),
            "source graph should not synthesize slices from target-only boundaries"
        );
        let target_nodes = target_stage_graph.graph.nodes().collect::<HashSet<_>>();
        assert!(
            target_nodes.contains(&GraphNode {
                node_id,
                sequence_start: 0,
                sequence_end: 3,
            }) && target_nodes.contains(&GraphNode {
                node_id,
                sequence_start: 4,
                sequence_end: 10,
            }),
            "target graph should use target-local blocks"
        );
    }

    #[test]
    fn test_stage_graph_keeps_terminal_blocks_for_endpoint_lookup() {
        let node_id = HashId::convert_str("regular-node");
        let sequence = Sequence::new().sequence_type("DNA").sequence("ATC").build();
        let edge_id = HashId::convert_str("edge-to-start");
        let stage_blocks = vec![
            test_terminal_block(0, PATH_START_NODE_ID),
            GroupBlock::new(1, node_id, &sequence, 0, 3),
            test_terminal_block(2, PATH_END_NODE_ID),
        ];
        let stage_edges = vec![test_augmented_edge(
            edge_id,
            node_id,
            3,
            PATH_START_NODE_ID,
            0,
        )];

        let stage_graph = build_stage_graph(
            &stage_blocks,
            &stage_edges,
            &HashMap::new(),
            &HashMap::new(),
        );

        let regular_node = test_graph_node(node_id, 0, 3);
        assert!(
            stage_graph
                .graph
                .edge_weight(regular_node, path_start_graph_node())
                .is_some(),
            "terminal blocks should participate in endpoint lookup regardless of edge direction"
        );
    }

    #[test]
    fn test_unified_diff_graph_connects_deletion_and_addition_through_common_terminals() {
        let operation = DoltHashId([100; 20]);
        let prefix_node_id = HashId::convert_str("prefix");
        let deleted_node_id = HashId::convert_str("deleted");
        let middle_node_id = HashId::convert_str("middle");
        let inserted_node_id = HashId::convert_str("inserted");
        let suffix_node_id = HashId::convert_str("suffix");
        let prefix_sequence = Sequence::new().sequence_type("DNA").sequence("ATC").build();
        let deleted_sequence = Sequence::new().sequence_type("DNA").sequence("G").build();
        let middle_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGAT")
            .build();
        let inserted_sequence = Sequence::new().sequence_type("DNA").sequence("AGA").build();
        let suffix_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("CGATC")
            .build();
        let prefix_node = test_graph_node(prefix_node_id, 0, 3);
        let deleted_node = test_graph_node(deleted_node_id, 0, 1);
        let middle_node = test_graph_node(middle_node_id, 0, 6);
        let inserted_node = test_graph_node(inserted_node_id, 0, 3);
        let suffix_node = test_graph_node(suffix_node_id, 0, 5);
        let source_blocks = vec![
            test_terminal_block(0, PATH_START_NODE_ID),
            GroupBlock::new(1, prefix_node_id, &prefix_sequence, 0, 3),
            GroupBlock::new(2, deleted_node_id, &deleted_sequence, 0, 1),
            GroupBlock::new(3, middle_node_id, &middle_sequence, 0, 6),
            GroupBlock::new(4, suffix_node_id, &suffix_sequence, 0, 5),
            test_terminal_block(5, PATH_END_NODE_ID),
        ];
        let target_blocks = vec![
            test_terminal_block(0, PATH_START_NODE_ID),
            GroupBlock::new(1, prefix_node_id, &prefix_sequence, 0, 3),
            GroupBlock::new(2, middle_node_id, &middle_sequence, 0, 6),
            GroupBlock::new(3, inserted_node_id, &inserted_sequence, 0, 3),
            GroupBlock::new(4, suffix_node_id, &suffix_sequence, 0, 5),
            test_terminal_block(5, PATH_END_NODE_ID),
        ];
        let source_edges = vec![
            test_augmented_edge(
                HashId::convert_str("shared-start-edge"),
                PATH_START_NODE_ID,
                0,
                prefix_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("removed-prefix-deleted"),
                prefix_node_id,
                3,
                deleted_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("removed-deleted-middle"),
                deleted_node_id,
                1,
                middle_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("source-middle-suffix"),
                middle_node_id,
                6,
                suffix_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("shared-end-edge"),
                suffix_node_id,
                5,
                PATH_END_NODE_ID,
                0,
            ),
        ];
        let target_edges = vec![
            test_augmented_edge(
                HashId::convert_str("shared-start-edge"),
                PATH_START_NODE_ID,
                0,
                prefix_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("added-prefix-middle"),
                prefix_node_id,
                3,
                middle_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("added-middle-inserted"),
                middle_node_id,
                6,
                inserted_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("added-inserted-suffix"),
                inserted_node_id,
                3,
                suffix_node_id,
                0,
            ),
            test_augmented_edge(
                HashId::convert_str("shared-end-edge"),
                suffix_node_id,
                5,
                PATH_END_NODE_ID,
                0,
            ),
        ];
        let source_graph = build_stage_graph(
            &source_blocks,
            &source_edges,
            &HashMap::new(),
            &HashMap::new(),
        );
        let target_graph = build_stage_graph(
            &target_blocks,
            &target_edges,
            &HashMap::new(),
            &HashMap::new(),
        );
        let block_group = BlockGroup {
            id: HashId::pad_str(99),
            collection_name: "collection".to_string(),
            sample_name: "sample".to_string(),
            name: "block-group".to_string(),
            created_on: 0,
            parent_block_group_id: None,
            is_default: false,
        };
        let stage_graphs = BlockGroupStageGraphs {
            source_block_group: Some(block_group.clone()),
            target_block_group: Some(block_group),
            source_graph,
            reconstructed_target_graph: target_graph,
        };

        let graph = build_unified_diff_graph(&stage_graphs, operation);

        assert_eq!(
            graph
                .nodes()
                .filter(|node| node.node.node_id == PATH_START_NODE_ID)
                .count(),
            1,
            "source and target paths should share one start terminal"
        );
        assert_eq!(
            graph
                .nodes()
                .filter(|node| node.node.node_id == PATH_END_NODE_ID)
                .count(),
            1,
            "source and target paths should share one end terminal"
        );
        assert!(
            graph.nodes().any(
                |node| node.node == deleted_node && node.change.kind == DiffChangeKind::Removed
            ),
            "deleted G node should be rendered as removed"
        );
        assert!(
            graph.nodes().any(|node| node.node == inserted_node
                && node.change.kind == DiffChangeKind::Added),
            "inserted AGA node should be rendered as added"
        );
        assert_diff_edge_kind(&graph, prefix_node, deleted_node, DiffChangeKind::Removed);
        assert_diff_edge_kind(&graph, deleted_node, middle_node, DiffChangeKind::Removed);
        assert_diff_edge_kind(&graph, prefix_node, middle_node, DiffChangeKind::Added);
        assert_all_nodes_connected_to(&graph, path_start_graph_node());
        assert_diff_edge_kind(
            &graph,
            path_start_graph_node(),
            prefix_node,
            DiffChangeKind::Unchanged,
        );
        assert_diff_edge_kind(
            &graph,
            suffix_node,
            path_end_graph_node(),
            DiffChangeKind::Unchanged,
        );
    }

    #[test]
    fn test_merges_multiple_history_operations_preserve_operation_attribution() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        create_linear_block_group(&context, "c", "s", "bg1", "one", "CCCCC");
        let first_operation = commit_operation(&context, "add one");
        create_linear_block_group(&context, "c", "s", "bg2", "two", "GGGGG");
        let second_operation = commit_operation(&context, "add two");

        let expected_first_change =
            collect_operation_diff(graph_conn, Some(base), first_operation, DiffRange::TwoDot)
                .expect("first operation diff")
                .diff_graph
                .into_iter()
                .flat_map(|block_group| block_group.graph.nodes().collect::<Vec<_>>())
                .find(|node| {
                    node.change.kind == DiffChangeKind::Added
                        && node.change.operation == Some(first_operation)
                })
                .expect("first operation should add a node");
        let expected_second_change = collect_operation_diff(
            graph_conn,
            Some(first_operation),
            second_operation,
            DiffRange::TwoDot,
        )
        .expect("second operation diff")
        .diff_graph
        .into_iter()
        .flat_map(|block_group| block_group.graph.nodes().collect::<Vec<_>>())
        .find(|node| {
            node.change.kind == DiffChangeKind::Added
                && node.change.operation == Some(second_operation)
        })
        .expect("second operation should add a node");

        let diff =
            collect_operation_diff(graph_conn, Some(base), second_operation, DiffRange::TwoDot)
                .expect("diff");
        assert_eq!(diff.operations, vec![first_operation, second_operation]);

        let merged_changes = diff
            .diff_graph
            .iter()
            .flat_map(|block_group| block_group.graph.nodes())
            .map(|node| (node.node, node.change.kind))
            .collect::<HashSet<_>>();

        assert!(
            merged_changes.contains(&(
                expected_first_change.node,
                expected_first_change.change.kind
            )),
            "merged diff should contain the node change introduced by the first operation"
        );
        assert!(
            merged_changes.contains(&(
                expected_second_change.node,
                expected_second_change.change.kind,
            )),
            "merged diff should contain the node change introduced by the second operation"
        );
    }

    #[test]
    fn test_merges_multiple_history_operations() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let op1 = commit_operation(&context, "seed");
        let bg_one = create_linear_block_group(&context, "c", "s", "bg1", "one", "CCCCC");
        let op2 = commit_operation(&context, "add one");
        let bg_two = create_linear_block_group(&context, "c", "s", "bg2", "two", "GGGGG");
        let op3 = commit_operation(&context, "add two");

        let diff =
            collect_operation_diff(graph_conn, Some(op1), op3, DiffRange::TwoDot).expect("diff");
        assert_eq!(diff.operations, vec![op2, op3]);
        assert_eq!(diff.diff_graph.len(), 2);
        assert_eq!(
            diff.diff_graph
                .iter()
                .map(|block_group| block_group.id)
                .collect::<Vec<_>>(),
            vec![bg_one.id, bg_two.id]
        );
    }

    #[test]
    fn test_diff_to_non_head_commit_uses_target_history_state() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let bg_one = create_linear_block_group(&context, "c", "s", "bg1", "one", "CCCCC");
        let op_one = commit_operation(&context, "add one");
        let bg_two = create_linear_block_group(&context, "c", "s", "bg2", "two", "GGGGG");
        let op_two = commit_operation(&context, "add two");

        let diff = collect_operation_diff(graph_conn, Some(base), op_one, DiffRange::TwoDot)
            .expect("diff");
        assert_eq!(diff.operations, vec![op_one]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, bg_one.id);
        assert_ne!(diff.diff_graph[0].id, bg_two.id);

        let later_diff =
            collect_operation_diff(graph_conn, Some(op_one), op_two, DiffRange::TwoDot)
                .expect("diff");
        assert_eq!(later_diff.operations, vec![op_two]);
        assert_eq!(later_diff.diff_graph.len(), 1);
        assert_eq!(later_diff.diff_graph[0].id, bg_two.id);
    }

    #[test]
    fn diff_against_itself_is_empty() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();
        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let diff =
            collect_operation_diff(graph_conn, Some(base), base, DiffRange::TwoDot).expect("diff");
        assert!(diff.operations.is_empty());
    }

    #[test]
    fn test_two_dot_diff_reports_changes_on_both_divergent_branches() {
        let history = setup_divergent_history();
        let graph_conn = history.context.graph().conn();
        let source_ref = history.source_hash.to_string();
        let target_ref = history.target_hash.to_string();
        let raw_changes = collect_block_group_changes(
            graph_conn,
            &DiffRange::TwoDot.between(&source_ref, &target_ref),
        )
        .expect("should collect changes across branches");
        assert_eq!(
            raw_changes.block_group_ids,
            HashSet::from([history.source_block_group.id, history.target_block_group.id,]),
            "Dolt diff rows should contain the source removal and target addition"
        );

        let diff = collect_operation_diff(
            graph_conn,
            Some(history.source_hash),
            history.target_hash,
            DiffRange::TwoDot,
        )
        .expect("should collect endpoint diff");
        assert_eq!(
            diff.operations,
            vec![history.source_hash, history.target_hash]
        );
        assert_eq!(
            diff.diff_graph.len(),
            2,
            "expected source {source:?} and target {target:?} changes, got block groups {:?}",
            diff.diff_graph
                .iter()
                .map(|block_group| block_group.id)
                .collect::<Vec<_>>(),
            source = history.source_block_group.id,
            target = history.target_block_group.id,
        );
        let source_diff = diff
            .diff_graph
            .iter()
            .find(|block_group| block_group.id == history.source_block_group.id)
            .expect("source-only block group should be reported as removed");
        assert!(source_diff.source_block_group.is_some());
        assert!(source_diff.target_block_group.is_none());
        let target_diff = diff
            .diff_graph
            .iter()
            .find(|block_group| block_group.id == history.target_block_group.id)
            .expect("target-only block group should be reported as added");
        assert!(target_diff.source_block_group.is_none());
        assert!(target_diff.target_block_group.is_some());
    }

    #[test]
    fn test_three_dot_diff_reports_target_changes_from_merge_base() {
        let history = setup_divergent_history();
        let merge_base_diff = collect_operation_diff(
            history.context.graph().conn(),
            Some(history.source_hash),
            history.target_hash,
            DiffRange::ThreeDot,
        )
        .expect("should collect merge-base diff");
        assert_eq!(merge_base_diff.operations, vec![history.target_hash]);
        assert_eq!(
            merge_base_diff.diff_graph.len(),
            1,
            "three-dot diff should only report target changes from the merge base"
        );
        let target_diff = &merge_base_diff.diff_graph[0];
        assert_eq!(target_diff.id, history.target_block_group.id);
        assert!(target_diff.source_block_group.is_none());
        assert!(target_diff.target_block_group.is_some());
    }

    #[test]
    fn test_uses_default_graph_database_for_history_diffs() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let expected_block_group =
            create_linear_block_group(&context, "c", "s", "db-one", "one", "CCCCC");
        let head = commit_operation(&context, "add");

        let diff =
            collect_operation_diff(graph_conn, Some(base), head, DiffRange::TwoDot).expect("diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, expected_block_group.id);
    }

    #[test]
    fn test_diffs_follow_reset_replacement_history() {
        let mut context = setup_history_context();
        let graph_conn = context.graph().conn();
        let graph_db_path = context
            .workspace()
            .graph_db_path()
            .expect("should resolve graph db path");
        let history_store = DoltHistoryStore::new(graph_conn);

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let retained_block_group =
            create_linear_block_group(&context, "c", "s", "retained", "one", "CCCCC");
        let retained_commit = commit_operation(&context, "retained");
        let discarded_block_group =
            create_linear_block_group(&context, "c", "s", "discarded", "two", "GGGGG");
        commit_operation(&context, "discarded");

        history_store
            .reset_hard(&commit_hash_ref(retained_commit))
            .expect("should reset to retained commit");
        let reopened_graph =
            gen_models::db::get_connection(&graph_db_path).expect("should reopen graph db");
        gen_models::history::dolt::connect_branch(&reopened_graph, "main")
            .expect("should reconnect reopened main branch");
        context.set_graph(reopened_graph);
        let replacement_block_group =
            create_linear_block_group(&context, "c", "s", "replacement", "three", "TTTTT");
        let replacement_commit = commit_operation(&context, "replacement");
        let reopened_graph_after_replacement =
            gen_models::db::get_connection(&graph_db_path).expect("should reopen graph db");
        gen_models::history::dolt::connect_branch(&reopened_graph_after_replacement, "main")
            .expect("should reconnect reopened main branch");
        let discarded_exists = reopened_graph_after_replacement
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM block_groups WHERE id = ?1)",
                params![discarded_block_group.id],
                |row| row.get::<_, bool>(0),
            )
            .expect("should query reopened reset snapshot");
        let graph_conn = context.graph().conn();
        let diff = collect_operation_diff(
            graph_conn,
            Some(base),
            replacement_commit,
            DiffRange::TwoDot,
        )
        .expect("diff");
        assert_eq!(diff.operations, vec![retained_commit, replacement_commit]);
        assert!(
            !discarded_exists,
            "discarded block group should not remain in the post-reset snapshot"
        );
        assert_eq!(diff.diff_graph.len(), 2);
        let added_ids = diff
            .diff_graph
            .iter()
            .map(|block_group| block_group.id)
            .collect::<HashSet<_>>();
        assert_eq!(
            added_ids,
            HashSet::from([retained_block_group.id, replacement_block_group.id])
        );
        assert!(
            diff.diff_graph
                .iter()
                .all(|block_group| block_group.id != discarded_block_group.id),
            "discarded commit should not survive the reset path"
        );
    }

    #[test]
    fn test_merge_commit_diff_includes_merged_branch_state() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(graph_conn);

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let main_block_group =
            create_linear_block_group(&context, "c", "s", "main", "main", "CCCCC");
        let main_commit = commit_operation(&context, "main");

        create_feature_branch(&context, base, "feature");
        let feature_block_group =
            create_linear_block_group(&context, "c", "s", "feature", "feature", "GGGGG");
        commit_operation(&context, "feature");

        switch_branch(&context, "main");
        history_store
            .merge(&CommitRef("feature".to_string()))
            .expect("should merge feature branch");
        let merge_commit = history_store
            .current_head()
            .expect("should resolve current head")
            .expect("should have merge head");
        let merge_hash = merge_commit;

        let diff = collect_operation_diff(graph_conn, Some(base), merge_hash, DiffRange::TwoDot)
            .expect("diff");
        assert!(
            diff.operations.contains(&main_commit),
            "diff should include main-branch history"
        );
        assert!(
            diff.operations.contains(&merge_hash),
            "diff should include the merge commit"
        );
        let added_ids = diff
            .diff_graph
            .iter()
            .map(|block_group| block_group.id)
            .collect::<HashSet<_>>();
        assert_eq!(
            added_ids,
            HashSet::from([main_block_group.id, feature_block_group.id])
        );
    }

    fn assert_block_group_graph_shape(block_group_diff: &BlockGroupDiff) {
        assert_eq!(block_group_diff.graph.nodes().count(), 3);
        assert_eq!(block_group_diff.graph.all_edges().count(), 2);
    }
}
