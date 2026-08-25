//! Views a patch archive by materializing it and using Gen's normal operation-diff viewer.
//!
//! A patch archive contains ordered schema and data statements plus any referenced asset files; it
//! does not contain the domain-level [`OperationDiff`] consumed by the diff TUI. To construct that
//! diff, this module copies the current graph database into a temporary workspace, creates a
//! throwaway branch at the patch's base commit, applies the archive there, and compares the base
//! commit with the newly materialized target commit using [`collect_operation_diff`]. The result is
//! then displayed through [`view_diff`], keeping patch and history views consistent.
//!
//! The temporary workspace prevents patch statements, Dolt commits, and restored asset files from
//! changing the user's live workspace. The temporary branch serves a separate purpose: patch
//! application creates commits, so it needs a writable branch rooted at the exact base commit even
//! when that historical commit is not the head of an existing branch. Both exist only in the copied
//! repository and are removed with the temporary directory. The prepared view retains ownership of
//! that directory so its database and assets remain available until rendering finishes.
//!
//! A first-commit patch has no base state or statements to replay. In that case, the archived target
//! commit already present in the copied history is compared directly with an empty history.

use std::io::{Read, Seek};

use gen_core::{BranchName, CommitRef, HashId, errors::ConfigError};
use gen_diff::operations::{DiffRange, OperationDiff, OperationDiffError, collect_operation_diff};
use gen_models::{
    db::{DbContext, get_config_connection, get_connection},
    history::{HistoryStore, dolt::DoltHistoryStore},
};
use rusqlite::MAIN_DB;
use tempfile::{TempDir, tempdir};
use thiserror::Error;

use crate::{
    patch::{PatchError, apply_patch_archive_to_isolated_context, load_operation_patches},
    views::diff::view_diff,
};

#[derive(Debug, Error)]
pub enum PatchViewError {
    #[error("Applied patch did not create a target commit.")]
    MissingTargetCommit,
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    #[error("Diff error: {0}")]
    Diff(#[from] OperationDiffError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Patch error: {0}")]
    Patch(#[from] PatchError),
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

struct PreparedPatchView {
    _temporary_workspace: TempDir,
    context: DbContext,
    diff: OperationDiff,
}

fn temporary_branch_name(temporary_workspace: &TempDir) -> BranchName {
    let suffix = HashId::convert_str(&temporary_workspace.path().to_string_lossy());
    BranchName(format!("gen-patch-view-{suffix}"))
}

fn prepare_patch_view<R>(
    source_context: &DbContext,
    reader: &mut R,
) -> Result<PreparedPatchView, PatchViewError>
where
    R: Read + Seek,
{
    reader.rewind()?;
    let operation_patches = load_operation_patches(&mut *reader)?;
    let base_commit = operation_patches.base_commit_hash;
    let archived_target_commit = operation_patches.target_commit_hash;

    let temporary_workspace = tempdir()?;
    let workspace = gen_core::config::Workspace::new(temporary_workspace.path());
    workspace.ensure_gen_dir();
    let graph_path = workspace.graph_db_path()?;
    source_context
        .graph()
        .conn()
        .backup(MAIN_DB, &graph_path, None)?;

    let graph_conn = get_connection(&graph_path)?;
    let config_conn = get_config_connection(workspace.gen_db_path()?)?;
    let mut context = DbContext::new_raw(workspace, graph_conn, config_conn);
    let history_store = DoltHistoryStore::new(context.graph().conn());
    let temporary_branch = temporary_branch_name(&temporary_workspace);
    let branch_start = base_commit.as_ref().unwrap_or(&archived_target_commit);
    history_store.create_branch(
        &temporary_branch,
        Some(&CommitRef(branch_start.to_string())),
    )?;
    history_store.checkout_branch(&temporary_branch)?;

    let target_commit = if base_commit.is_some() {
        reader.rewind()?;
        apply_patch_archive_to_isolated_context(&mut context, reader)?;
        DoltHistoryStore::new(context.graph().conn())
            .current_head()?
            .ok_or(PatchViewError::MissingTargetCommit)?
    } else {
        archived_target_commit
    };
    let diff = collect_operation_diff(
        context.graph().conn(),
        base_commit,
        target_commit,
        DiffRange::TwoDot,
    )?;

    Ok(PreparedPatchView {
        _temporary_workspace: temporary_workspace,
        context,
        diff,
    })
}

pub fn view_patch<R>(source_context: &DbContext, reader: &mut R) -> Result<(), PatchViewError>
where
    R: Read + Seek,
{
    let prepared = prepare_patch_view(source_context, reader)?;
    view_diff(
        prepared.context.graph().conn(),
        prepared.context.workspace(),
        &prepared.diff,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{io::Cursor, path::PathBuf};

    use gen_diff::{
        graph::{DiffGraphEdge, DiffGraphNode},
        operations::{BlockGroupDiff, DiffRange, OperationDiff, collect_operation_diff},
    };
    use gen_models::{
        history::{HistoryStore, dolt::DoltHistoryStore},
        operations::commit_operation_summary,
        sample::Sample,
    };

    use super::prepare_patch_view;
    use crate::{
        imports::fasta::import_fasta, patch::create_patch, test_helpers::setup_gen_on_disk,
    };

    fn without_node_operation(mut node: DiffGraphNode) -> DiffGraphNode {
        node.change.operation = None;
        node
    }

    fn without_edge_operation(mut edge: DiffGraphEdge) -> DiffGraphEdge {
        edge.change.operation = None;
        edge
    }

    fn sorted_nodes(diff: &BlockGroupDiff) -> Vec<DiffGraphNode> {
        let mut nodes = diff
            .graph
            .nodes()
            .map(without_node_operation)
            .collect::<Vec<_>>();
        nodes.sort();
        nodes
    }

    fn sorted_edges(
        diff: &BlockGroupDiff,
    ) -> Vec<(DiffGraphNode, DiffGraphNode, Vec<DiffGraphEdge>)> {
        let mut edges = diff
            .graph
            .all_edges()
            .map(|(source, target, edges)| {
                (
                    without_node_operation(source),
                    without_node_operation(target),
                    edges.iter().copied().map(without_edge_operation).collect(),
                )
            })
            .collect::<Vec<_>>();
        edges.sort();
        edges
    }

    fn assert_same_graphical_diff(expected: &OperationDiff, actual: &OperationDiff) {
        let mut expected_graphs = expected.diff_graph.iter().collect::<Vec<_>>();
        expected_graphs.sort_by_key(|diff| diff.id);
        let mut actual_graphs = actual.diff_graph.iter().collect::<Vec<_>>();
        actual_graphs.sort_by_key(|diff| diff.id);

        assert_eq!(actual_graphs.len(), expected_graphs.len());
        for (actual, expected) in actual_graphs.into_iter().zip(expected_graphs) {
            assert_eq!(actual.id, expected.id);
            assert_eq!(actual.source_block_group, expected.source_block_group);
            assert_eq!(actual.target_block_group, expected.target_block_group);
            assert_eq!(sorted_nodes(actual), sorted_nodes(expected));
            assert_eq!(sorted_edges(actual), sorted_edges(expected));
        }
    }

    fn import_fixture(context: &gen_models::db::DbContext, sample: &str) {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple.fa")
            .to_string_lossy()
            .to_string();
        let operation = import_fasta(context, &fixture_path, "default", sample, false, &[])
            .expect("should import the FASTA fixture");
        commit_operation_summary(context, &operation).expect("should commit the FASTA import");
    }

    #[test]
    fn test_first_commit_patch_produces_same_diff_as_source_history() {
        let source_context = setup_gen_on_disk();
        import_fixture(&source_context, Sample::DEFAULT_NAME);
        let history = DoltHistoryStore::new(source_context.graph().conn())
            .log(None)
            .expect("should load source history");
        let target_hash = history[0].commit_hash;
        let expected = collect_operation_diff(
            source_context.graph().conn(),
            None,
            target_hash,
            DiffRange::TwoDot,
        )
        .expect("should collect the first source history diff");

        let mut archive = Cursor::new(Vec::new());
        create_patch(&source_context, &[target_hash], &mut archive)
            .expect("should export the first source operation as a patch");
        let prepared = prepare_patch_view(&source_context, &mut archive)
            .expect("should prepare the first-commit patch for viewing");

        assert_same_graphical_diff(&expected, &prepared.diff);
    }

    #[test]
    fn test_exported_patch_produces_same_diff_as_source_history() {
        let source_context = setup_gen_on_disk();
        import_fixture(&source_context, Sample::DEFAULT_NAME);
        import_fixture(&source_context, "patch-view-sample");

        let history = DoltHistoryStore::new(source_context.graph().conn())
            .log(None)
            .expect("should load source history");
        let target_hash = history[0].commit_hash;
        let base_hash = history[1].commit_hash;
        let expected = collect_operation_diff(
            source_context.graph().conn(),
            Some(base_hash),
            target_hash,
            DiffRange::TwoDot,
        )
        .expect("should collect the source history diff");

        let mut archive = Cursor::new(Vec::new());
        create_patch(&source_context, &[target_hash], &mut archive)
            .expect("should export the source operation as a patch");
        let prepared = prepare_patch_view(&source_context, &mut archive)
            .expect("should prepare the exported patch for viewing");

        assert_same_graphical_diff(&expected, &prepared.diff);
    }
}
