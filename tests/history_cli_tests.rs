use std::{
    collections::HashSet,
    fs,
    fs::File,
    path::{Path, PathBuf},
    process::Command,
};

use r#gen::{
    core::Workspace,
    diff::{
        graph::DiffChangeKind,
        operations::{DiffRange, collect_operation_diff},
    },
    get_connection,
    history::operations_history_entries,
    patch::load_patches,
};
use gen_models::{
    assets::AssetRef,
    block_group::BlockGroup,
    collection::Collection,
    db::GraphConnection,
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, status_rows},
    },
    node::Node,
    sample_lineage::SampleLineage,
    traits::Query,
};
use rusqlite::{Connection, params};
use tempfile::tempdir;

fn gen_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_gen"))
}

fn run_gen(repo_root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(gen_binary())
        .current_dir(repo_root)
        .args(args)
        .output()
        .expect("should run gen command")
}

fn assert_success(output: &std::process::Output, context: &str) {
    assert!(
        output.status.success(),
        "{context}: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn operations_stdout(repo_root: &Path) -> String {
    let operations_output = run_gen(repo_root, &["operations"]);
    assert_success(&operations_output, "gen operations should succeed");
    String::from_utf8_lossy(&operations_output.stdout).into_owned()
}

fn operations_stdout_for_branch(repo_root: &Path, branch_name: &str) -> String {
    let operations_output = run_gen(repo_root, &["operations", "--branch", branch_name]);
    assert_success(&operations_output, "gen operations --branch should succeed");
    String::from_utf8_lossy(&operations_output.stdout).into_owned()
}

fn commit_hash_for_summary(operations_output: &str, summary: &str) -> String {
    operations_output
        .lines()
        .find(|line| line.contains(summary))
        .and_then(|line| {
            let trimmed = line.trim_start_matches('>').trim();
            trimmed.split_whitespace().next()
        })
        .map(str::to_string)
        .unwrap_or_else(|| {
            panic!("should find commit summary '{summary}' in:\n{operations_output}")
        })
}

fn trailing_commit_hash(output: &std::process::Output, context: &str) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .split_whitespace()
        .last()
        .map(str::to_string)
        .unwrap_or_else(|| {
            panic!("{context}: should contain a trailing commit hash in stdout={stdout}")
        })
}

fn asset_refs(repo_root: &Path) -> Vec<AssetRef> {
    let graph_path = repo_root.join(".gen/default.db");
    let connection = get_connection(graph_path).expect("should reopen graph database");
    let mut asset_refs = AssetRef::all(&connection);
    asset_refs.sort_by(|left, right| {
        left.created_on
            .cmp(&right.created_on)
            .then_with(|| left.role.as_str().cmp(right.role.as_str()))
            .then_with(|| left.name.cmp(&right.name))
    });
    asset_refs
}

mod diff_views {
    use std::io::Write;

    use super::{
        DiffChangeKind, DiffRange, DoltHistoryStore, HashSet, HistoryStore, Node, Path, PathBuf,
        assert_success, collect_operation_diff, fs, get_connection, run_gen, tempdir,
    };

    fn maybe_export_diff_graph_debug(graph: &gen_diff::graph::DiffGenGraph, export_path: &Path) {
        let Some(parent_dir) = export_path.parent() else {
            return;
        };
        fs::create_dir_all(parent_dir).expect("should create debug export directory");
        let mut file = fs::File::create(export_path).expect("should create debug export file");

        for node in graph.nodes() {
            writeln!(
                file,
                "NODE\t{}\t{}\t{}\t{}\t{}\t{}",
                node.node.node_id,
                node.node.sequence_start,
                node.node.sequence_end,
                matches!(node.change.kind, DiffChangeKind::Added),
                matches!(node.change.kind, DiffChangeKind::Removed),
                node.change
                    .operation
                    .map(|operation| operation.to_string())
                    .unwrap_or_else(|| "none".to_string())
            )
            .expect("should write node export");
        }
        for (source, target, edges) in graph.all_edges() {
            for edge in edges {
                writeln!(
                    file,
                    "EDGE\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    source.node.node_id,
                    source.node.sequence_start,
                    source.node.sequence_end,
                    target.node.node_id,
                    target.node.sequence_start,
                    target.node.sequence_end,
                    matches!(edge.change.kind, DiffChangeKind::Added),
                    matches!(edge.change.kind, DiffChangeKind::Removed),
                    edge.change
                        .operation
                        .map(|operation| operation.to_string())
                        .unwrap_or_else(|| "none".to_string())
                )
                .expect("should write edge export");
            }
        }
    }

    fn append_graph_debug(graph: &gen_graph::GenGraph, export_path: &Path, prefix: &str) {
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(export_path)
            .expect("should reopen debug export file");
        for node in graph.nodes() {
            writeln!(
                file,
                "{prefix}_NODE\t{}\t{}\t{}",
                node.node_id, node.sequence_start, node.sequence_end
            )
            .expect("should append parent node export");
        }
        for (source, target, _) in graph.all_edges() {
            writeln!(
                file,
                "{prefix}_EDGE\t{}\t{}\t{}\t{}\t{}\t{}",
                source.node_id,
                source.sequence_start,
                source.sequence_end,
                target.node_id,
                target.sequence_start,
                target.sequence_end
            )
            .expect("should append parent edge export");
        }
    }

    #[test]
    fn test_view_diff_reports_vcf_added_and_removed_nodes_from_deltas() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let vcf_path = fixtures_dir.join("simple.vcf");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "default",
                    "--reference",
                    "default",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "vcf",
                    vcf_path.to_str().expect("should encode vcf path"),
                    "--name",
                    "default",
                    "--parent-samples",
                    "default",
                ],
            ),
            "vcf update should succeed",
        );

        let graph_conn = get_connection(repo_dir.path().join(".gen/default.db"))
            .expect("should reopen graph database");
        let history_entries = DoltHistoryStore::new(&graph_conn)
            .log(None)
            .expect("should query dolt history");
        let update_hash = history_entries
            .first()
            .map(|entry| entry.commit_hash)
            .expect("should contain update commit");
        let import_hash = history_entries
            .get(1)
            .map(|entry| entry.commit_hash)
            .expect("should contain import commit");

        let diff = collect_operation_diff(
            &graph_conn,
            Some(import_hash),
            update_hash,
            DiffRange::TwoDot,
        )
        .expect("should collect operation diff");
        let unknown_diff = diff
            .diff_graph
            .iter()
            .find(|diff| {
                diff.target_block_group.as_ref().is_some_and(|block_group| {
                    block_group.collection_name == "default"
                        && block_group.sample_name == "unknown"
                        && block_group.name == "m123"
                })
            })
            .expect("should contain unknown sample diff");
        assert!(
            unknown_diff.source_block_group.is_none(),
            "new unknown sample should not exist in the source revision"
        );
        let unknown_target_block_group = unknown_diff
            .target_block_group
            .as_ref()
            .expect("new unknown sample should exist in the target revision");
        assert_eq!(unknown_target_block_group.id, unknown_diff.id);
        let g1_diff = diff
            .diff_graph
            .iter()
            .find(|diff| {
                diff.target_block_group.as_ref().is_some_and(|block_group| {
                    block_group.collection_name == "default"
                        && block_group.sample_name == "G1"
                        && block_group.name == "m123"
                })
            })
            .expect("should contain target-only G1 sample diff even without sequence changes");
        assert!(
            g1_diff.source_block_group.is_none(),
            "new samples should be visible as added samples even when their graph matches the parent"
        );
        assert!(g1_diff.target_block_group.is_some());
        let parent_block_group_id = unknown_target_block_group
            .parent_block_group_id
            .expect("lineage-derived sample should retain parent block group id");
        let parent_graph =
            gen_graph::models::load_block_group_graph(&graph_conn, &parent_block_group_id, None)
                .expect("should load parent block group graph");
        assert!(
            parent_graph.nodes().next().is_some(),
            "parent block group graph should contain reference context"
        );

        if let Ok(export_path) = std::env::var("GEN_DEBUG_DIFF_EXPORT") {
            let export_path = Path::new(&export_path);
            maybe_export_diff_graph_debug(&unknown_diff.graph, export_path);
            append_graph_debug(&parent_graph, export_path, "PARENT");
        }

        let new_node_ids = unknown_diff
            .graph
            .nodes()
            .filter(|node| node.change.kind == DiffChangeKind::Added)
            .map(|node| node.node.node_id)
            .collect::<HashSet<_>>();
        assert!(
            !new_node_ids.is_empty(),
            "fixture update should introduce highlighted nodes"
        );
        let invalid_highlight_edges = unknown_diff
            .graph
            .all_edges()
            .flat_map(|(source, target, edges)| {
                edges.iter().filter_map(move |edge| {
                    (edge.change.kind == DiffChangeKind::Added
                        && source.change.kind == DiffChangeKind::Unchanged
                        && target.change.kind == DiffChangeKind::Unchanged
                        && source.node.node_id == target.node.node_id
                        && source.node.sequence_end == target.node.sequence_start)
                        .then_some((source.node.node_id, target.node.node_id, edge.edge.edge_id))
                })
            })
            .collect::<Vec<_>>();
        assert!(
            invalid_highlight_edges.is_empty(),
            "highlighted edges should not mark contiguous unchanged reference slices as new: {invalid_highlight_edges:?}"
        );

        let diff_node_ids = unknown_diff
            .graph
            .nodes()
            .map(|node| node.node.node_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let sequences_by_node_id =
            Node::get_sequences_by_node_ids(&graph_conn, &diff_node_ids, None);
        let removed_sequences = unknown_diff
            .graph
            .nodes()
            .filter(|node| {
                node.change.kind == DiffChangeKind::Removed
                    && node.node.sequence_end > node.node.sequence_start
            })
            .map(|node| {
                sequences_by_node_id[&node.node.node_id]
                    .get_sequence(node.node.sequence_start, node.node.sequence_end)
                    .expect("should slice removed node sequence")
            })
            .collect::<HashSet<_>>();
        let added_sequences = unknown_diff
            .graph
            .nodes()
            .filter(|node| {
                node.change.kind == DiffChangeKind::Added
                    && node.node.sequence_end > node.node.sequence_start
            })
            .map(|node| {
                sequences_by_node_id[&node.node.node_id]
                    .get_sequence(node.node.sequence_start, node.node.sequence_end)
                    .expect("should slice added node sequence")
            })
            .collect::<HashSet<_>>();
        assert_eq!(
            removed_sequences,
            HashSet::from_iter(["G".to_string()]),
            "vcf diff should show the deleted reference base as removed"
        );
        assert!(
            added_sequences.contains("AGA"),
            "vcf diff should include the inserted AGA node among added delta nodes: {added_sequences:?}"
        );
        let added_deletion_bypass_edges = unknown_diff
            .graph
            .all_edges()
            .filter(|(source, target, edges)| {
                source.node.node_id == target.node.node_id
                    && source.node.sequence_end == 3
                    && target.node.sequence_start == 4
                    && edges
                        .iter()
                        .any(|edge| edge.change.kind == DiffChangeKind::Added)
            })
            .count();
        assert!(
            added_deletion_bypass_edges > 0,
            "vcf diff should mark the edge that bypasses the deleted G as added"
        );
        let removed_insertion_passthrough_edges = unknown_diff
            .graph
            .all_edges()
            .filter(|(source, target, edges)| {
                source.node.node_id == target.node.node_id
                    && source.node.sequence_start == 4
                    && source.node.sequence_end == 10
                    && target.node.sequence_start == 10
                    && target.node.sequence_end == 34
                    && edges
                        .iter()
                        .any(|edge| edge.change.kind == DiffChangeKind::Removed)
            })
            .count();
        assert_eq!(
            removed_insertion_passthrough_edges, 0,
            "insertion should leave the original reference edge unhighlighted instead of marking it removed"
        );
    }
}

mod operation_history {
    use super::{
        PathBuf, assert_success, asset_refs, commit_hash_for_summary, operations_stdout, run_gen,
        tempdir, trailing_commit_hash,
    };

    #[test]
    fn test_import_fasta_creates_visible_history_entry() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");

        let init_output = run_gen(repo_dir.path(), &["init"]);
        assert_success(&init_output, "gen init should succeed");

        let import_output = run_gen(
            repo_dir.path(),
            &[
                "import",
                "fasta",
                fasta_path.to_str().expect("should encode fasta path"),
                "--name",
                "test-collection",
                "--sample",
                "test-sample",
            ],
        );
        assert_success(&import_output, "fasta import should succeed");

        let refs = asset_refs(repo_dir.path());
        assert!(
            refs.iter().any(|asset_ref| {
                asset_ref.uri == "file://.gen/outside_root/simple.fa"
                    && asset_ref.name.as_deref() == Some("simple.fa")
            }),
            "external FASTA should be namespaced away from workspace-root files: {refs:?}"
        );

        let stdout = operations_stdout(repo_dir.path());
        assert!(
            stdout.contains("m123:"),
            "operations output should include the fasta import commit summary: {stdout}"
        );
    }

    #[test]
    fn test_add_file_creates_visible_history_entry() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let annotation_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );

        let add_file_output = run_gen(
            repo_dir.path(),
            &[
                "add-file",
                annotation_path
                    .to_str()
                    .expect("should encode annotation path"),
                "--message",
                "add-gff",
            ],
        );
        assert_success(&add_file_output, "add-file should succeed");
        let printed_commit_hash =
            trailing_commit_hash(&add_file_output, "add-file should print the commit hash");

        let stdout = operations_stdout(repo_dir.path());
        assert!(
            stdout.contains("add-gff"),
            "operations output should include the add-file commit summary: {stdout}"
        );
        assert_eq!(
            printed_commit_hash,
            commit_hash_for_summary(&stdout, "add-gff"),
            "add-file should print the same commit hash shown by `gen operations`"
        );
    }

    #[test]
    fn test_add_annotation_file_creates_visible_history_entry() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let annotation_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );

        let add_annotation_output = run_gen(
            repo_dir.path(),
            &[
                "add-annotation-file",
                annotation_path
                    .to_str()
                    .expect("should encode annotation path"),
                "--name",
                "fixture-track",
                "--message",
                "add-annotation-sidecar",
            ],
        );
        assert_success(&add_annotation_output, "add-annotation-file should succeed");
        let printed_commit_hash = trailing_commit_hash(
            &add_annotation_output,
            "add-annotation-file should print the commit hash",
        );

        let stdout = operations_stdout(repo_dir.path());
        assert!(
            stdout.contains("add-annotation-sidecar"),
            "operations output should include the annotation-file commit summary: {stdout}"
        );
        assert_eq!(
            printed_commit_hash,
            commit_hash_for_summary(&stdout, "add-annotation-sidecar"),
            "add-annotation-file should print the same commit hash shown by `gen operations`"
        );
    }
}

mod patches {
    use super::{
        BlockGroup, File, PathBuf, Query, SampleLineage, Workspace, assert_success, fs,
        get_connection, load_patches, operations_stdout, operations_stdout_for_branch, params,
        run_gen, tempdir,
    };

    #[test]
    fn test_patch_create_accepts_history_range_selection() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let annotation_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "add-file",
                    annotation_path
                        .to_str()
                        .expect("should encode annotation path"),
                    "--message",
                    "add-gff",
                ],
            ),
            "add-file should succeed",
        );

        let patch_output = run_gen(
            repo_dir.path(),
            &["patch-create", "--name", "history-slice", "HEAD~1..HEAD"],
        );
        assert_success(
            &patch_output,
            "patch-create should succeed for a history range selection",
        );
        assert!(
            repo_dir.path().join("history-slice.gz").exists(),
            "patch-create should write the requested archive"
        );
    }

    #[test]
    fn test_patch_create_branch_uses_branch_head_against_current_head() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let annotation_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "add-file",
                    annotation_path
                        .to_str()
                        .expect("should encode annotation path"),
                    "--message",
                    "feature-gff",
                ],
            ),
            "feature branch add-file should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "main"]),
            "checkout main should succeed",
        );

        let patch_output = run_gen(
            repo_dir.path(),
            &[
                "patch-create",
                "--branch",
                "feature",
                "--name",
                "feature-head",
                "HEAD",
            ],
        );
        assert_success(
            &patch_output,
            "patch-create --branch should succeed for a diverged branch head",
        );

        let patch_path = repo_dir.path().join("feature-head.gz");
        assert!(
            patch_path.exists(),
            "patch-create should write the requested archive"
        );
        let patch_file = File::open(&patch_path).expect("should open the generated patch archive");
        let patches = load_patches(patch_file);
        assert_eq!(
            patches.len(),
            1,
            "only the branch-only commit should be selected"
        );
    }

    #[test]
    fn test_patch_apply_round_trips_from_repo_into_fresh_repo() {
        let source_repo = tempdir().expect("should create source repo directory");
        let target_repo = tempdir().expect("should create target repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let annotation_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(source_repo.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &[
                    "add-file",
                    annotation_path
                        .to_str()
                        .expect("should encode annotation path"),
                    "--message",
                    "add-gff",
                ],
            ),
            "add-file should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &["patch-create", "--name", "round-trip", "HEAD~2..HEAD"],
            ),
            "patch-create should succeed",
        );

        let patch_path = source_repo.path().join("round-trip.gz");
        assert!(patch_path.exists(), "patch archive should exist");

        assert_success(
            &run_gen(target_repo.path(), &["init"]),
            "target init should succeed",
        );
        assert_success(
            &run_gen(
                target_repo.path(),
                &[
                    "patch-apply",
                    patch_path.to_str().expect("should encode patch path"),
                ],
            ),
            "patch-apply should succeed for a fresh repo",
        );

        let target_operations = operations_stdout(target_repo.path());
        assert!(
            target_operations.contains("add-gff"),
            "fresh target history should include the add-file commit: {target_operations}"
        );
        assert!(
            target_operations.contains("m123:"),
            "fresh target history should include the import commit: {target_operations}"
        );
        let target_assets = Workspace::new(target_repo.path())
            .asset_dir()
            .expect("should resolve target asset dir");
        let asset_entries = fs::read_dir(target_assets)
            .expect("should list restored asset files")
            .count();
        assert!(
            asset_entries > 0,
            "patch apply should restore bundled asset payloads into the fresh repo"
        );
        let target_samples = run_gen(target_repo.path(), &["list-samples"]);
        assert_success(
            &target_samples,
            "list-samples should succeed after patch apply",
        );
        let target_samples_stdout = String::from_utf8_lossy(&target_samples.stdout);
        assert!(
            target_samples_stdout.contains("test-sample"),
            "patch apply should restore graph rows that make the imported sample visible: {target_samples_stdout}"
        );
    }

    #[test]
    fn test_patch_apply_branch_patch_to_another_branch() {
        let source_repo = tempdir().expect("should create source repo directory");
        let target_repo = tempdir().expect("should create target repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let annotation_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(source_repo.path(), &["init"]),
            "source init should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "source fasta import should succeed",
        );
        assert_success(
            &run_gen(source_repo.path(), &["branch", "--create", "feature"]),
            "source branch create should succeed",
        );
        assert_success(
            &run_gen(source_repo.path(), &["checkout", "--branch", "feature"]),
            "source checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &[
                    "add-file",
                    annotation_path
                        .to_str()
                        .expect("should encode annotation path"),
                    "--message",
                    "feature-gff",
                ],
            ),
            "source feature branch add-file should succeed",
        );
        assert_success(
            &run_gen(source_repo.path(), &["checkout", "main"]),
            "source checkout main should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &[
                    "patch-create",
                    "--branch",
                    "feature",
                    "--name",
                    "feature-head",
                    "HEAD",
                ],
            ),
            "source patch-create --branch should succeed",
        );

        let patch_path = source_repo.path().join("feature-head.gz");
        assert!(patch_path.exists(), "feature branch patch should exist");

        assert_success(
            &run_gen(target_repo.path(), &["init"]),
            "target init should succeed",
        );
        assert_success(
            &run_gen(
                target_repo.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "target fasta import should succeed",
        );
        assert_success(
            &run_gen(target_repo.path(), &["branch", "--create", "receiving"]),
            "target branch create should succeed",
        );
        assert_success(
            &run_gen(target_repo.path(), &["checkout", "--branch", "receiving"]),
            "target checkout receiving should succeed",
        );
        assert_success(
            &run_gen(
                target_repo.path(),
                &[
                    "patch-apply",
                    patch_path.to_str().expect("should encode patch path"),
                ],
            ),
            "patch-apply should succeed on the receiving branch",
        );

        let receiving_history = operations_stdout_for_branch(target_repo.path(), "receiving");
        assert!(
            receiving_history.contains("feature-gff"),
            "receiving branch should include the patched feature commit: {receiving_history}"
        );

        let main_history = operations_stdout_for_branch(target_repo.path(), "main");
        assert!(
            !main_history.contains("feature-gff"),
            "main branch should remain unchanged after applying the patch to another branch: {main_history}"
        );
    }

    #[test]
    fn test_patch_apply_preserves_sample_lineage_for_vcf_updates() {
        let source_repo = tempdir().expect("should create source repo directory");
        let target_repo = tempdir().expect("should create target repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let vcf_path = fixtures_dir.join("simple.vcf");

        assert_success(
            &run_gen(source_repo.path(), &["init"]),
            "source init should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "default",
                    "--reference",
                    "default",
                ],
            ),
            "source fasta import should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &[
                    "update",
                    "vcf",
                    vcf_path.to_str().expect("should encode vcf path"),
                    "--name",
                    "default",
                    "--parent-samples",
                    "default",
                ],
            ),
            "source vcf update should succeed",
        );
        assert_success(
            &run_gen(
                source_repo.path(),
                &["patch-create", "--name", "lineage-update", "HEAD"],
            ),
            "source patch-create should succeed",
        );

        let patch_path = source_repo.path().join("lineage-update.gz");
        assert!(patch_path.exists(), "lineage patch should exist");

        assert_success(
            &run_gen(target_repo.path(), &["init"]),
            "target init should succeed",
        );
        assert_success(
            &run_gen(
                target_repo.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "default",
                    "--reference",
                    "default",
                ],
            ),
            "target fasta import should succeed",
        );
        assert_success(
            &run_gen(
                target_repo.path(),
                &[
                    "patch-apply",
                    patch_path.to_str().expect("should encode patch path"),
                ],
            ),
            "patch-apply should succeed for lineage patch",
        );

        let graph_conn = get_connection(target_repo.path().join(".gen/default.db"))
            .expect("should reopen target graph database");
        assert_eq!(
            SampleLineage::get_parents(&graph_conn, "unknown", None),
            vec!["default".to_string()],
            "patch apply should preserve the lineage row for the derived sample"
        );
        let default_block_group = BlockGroup::query(
            &graph_conn,
            "SELECT * FROM block_groups
         WHERE collection_name = ?1 AND sample_name = ?2 AND name = ?3",
            params!["default", "default", "m123"],
        )
        .into_iter()
        .next()
        .expect("should retain the source block group");
        let imported_block_group = BlockGroup::query(
            &graph_conn,
            "SELECT * FROM block_groups
         WHERE collection_name = ?1 AND sample_name = ?2 AND name = ?3",
            params!["default", "unknown", "m123"],
        )
        .into_iter()
        .next()
        .expect("should import the derived block group");
        assert_eq!(
            imported_block_group.parent_block_group_id,
            Some(default_block_group.id),
            "patch apply should preserve the block-group parent lineage"
        );
    }
}

mod operation_updates {
    use super::{
        PathBuf, assert_success, commit_hash_for_summary, fs, operations_stdout, run_gen, tempdir,
    };

    #[test]
    fn test_update_fasta_creates_visible_history_entry() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let update_fasta_path = repo_dir.path().join("update.fa");

        fs::write(&update_fasta_path, ">m123\nTTTT\n").expect("should write update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );

        let update_output = run_gen(
            repo_dir.path(),
            &[
                "update",
                "fasta",
                update_fasta_path
                    .to_str()
                    .expect("should encode update fasta path"),
                "--name",
                "test-collection",
                "--sample",
                "test-sample",
                "--new-sample",
                "edited-sample",
                "--region-name",
                "m123:1-5",
            ],
        );
        assert_success(&update_output, "update fasta should succeed");

        let stdout = operations_stdout(repo_dir.path());
        assert!(
            stdout.contains("1 sequences inserted"),
            "operations output should include the update commit summary: {stdout}"
        );
    }

    #[test]
    fn test_reset_and_apply_restore_history_entries() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let annotation_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "add-file",
                    annotation_path
                        .to_str()
                        .expect("should encode annotation path"),
                    "--message",
                    "add-gff",
                ],
            ),
            "add-file should succeed",
        );

        let before_reset = operations_stdout(repo_dir.path());
        let add_file_hash = commit_hash_for_summary(&before_reset, "add-gff");
        let import_hash = commit_hash_for_summary(&before_reset, "m123:");

        assert_success(
            &run_gen(repo_dir.path(), &["reset", &import_hash]),
            "gen reset should succeed",
        );

        let after_reset = operations_stdout(repo_dir.path());
        let top_after_reset = after_reset
            .lines()
            .find(|line| line.trim_start().starts_with('>'))
            .expect("should have a HEAD line after reset");
        assert!(
            top_after_reset.contains(&import_hash),
            "reset should move HEAD to the import commit: {after_reset}"
        );

        assert_success(
            &run_gen(repo_dir.path(), &["apply", &add_file_hash]),
            "gen apply should succeed",
        );

        let after_apply = operations_stdout(repo_dir.path());
        let top_after_apply = after_apply
            .lines()
            .find(|line| line.trim_start().starts_with('>'))
            .expect("should have a HEAD line after apply");
        assert!(
            top_after_apply.contains("add-gff"),
            "apply should create a new HEAD commit with the add-file summary: {after_apply}"
        );
    }
}

mod revision_views {
    use super::{
        Collection, DoltHistoryStore, HistoryStore, Path, PathBuf, assert_success, fs,
        get_connection, operations_history_entries, operations_stdout,
        operations_stdout_for_branch, run_gen, status_rows, tempdir,
    };

    fn head_commit_hash(operations_output: &str) -> String {
        operations_output
            .lines()
            .find(|line| line.trim_start().starts_with('>'))
            .and_then(|line| {
                let trimmed = line.trim_start_matches('>').trim();
                trimmed.split_whitespace().next()
            })
            .map(str::to_string)
            .unwrap_or_else(|| panic!("should find a HEAD commit in:\n{operations_output}"))
    }

    fn setup_repo_with_feature_only_update(repo_root: &Path) -> (PathBuf, String, String) {
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let update_fasta_path = repo_root.join("feature-update.fa");

        fs::write(&update_fasta_path, ">m123\nTTTT\n")
            .expect("should write branch update fasta fixture");

        assert_success(&run_gen(repo_root, &["init"]), "gen init should succeed");
        assert_success(
            &run_gen(
                repo_root,
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(repo_root, &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_root, &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_root,
                &[
                    "update",
                    "fasta",
                    update_fasta_path
                        .to_str()
                        .expect("should encode update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "feature-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "feature branch update should succeed",
        );

        let feature_history = operations_stdout(repo_root);
        let feature_update_hash = head_commit_hash(&feature_history);

        assert_success(
            &run_gen(repo_root, &["checkout", "main"]),
            "checkout main should succeed",
        );

        (
            update_fasta_path,
            feature_update_hash,
            "feature-sample".to_string(),
        )
    }

    #[test]
    fn test_checkout_switches_branch_specific_graph_state() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let update_fasta_path = repo_dir.path().join("feature-update.fa");

        fs::write(&update_fasta_path, ">m123\nTTTT\n")
            .expect("should write branch update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    update_fasta_path
                        .to_str()
                        .expect("should encode update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "feature-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "feature branch update should succeed",
        );

        let feature_samples = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(&feature_samples, "list-samples on feature should succeed");
        let feature_stdout = String::from_utf8_lossy(&feature_samples.stdout);
        assert!(
            feature_stdout.contains("feature-sample"),
            "feature branch should expose the new sample: {feature_stdout}"
        );

        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "main"]),
            "checkout main should succeed",
        );

        let main_samples = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(&main_samples, "list-samples on main should succeed");
        let main_stdout = String::from_utf8_lossy(&main_samples.stdout);
        assert!(
            !main_stdout.contains("feature-sample"),
            "main branch should not expose the feature-only sample: {main_stdout}"
        );
    }

    #[test]
    fn test_checkout_commit_hash_reports_detached_head_not_supported() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let (_, feature_update_hash, _) = setup_repo_with_feature_only_update(repo_dir.path());

        let detached_checkout = run_gen(repo_dir.path(), &["checkout", &feature_update_hash]);
        assert!(
            !detached_checkout.status.success(),
            "checkout by resolved commit hash should fail: stdout={} stderr={}",
            String::from_utf8_lossy(&detached_checkout.stdout),
            String::from_utf8_lossy(&detached_checkout.stderr)
        );
        let stderr = String::from_utf8_lossy(&detached_checkout.stderr);
        assert!(
            stderr.contains("Detached HEAD checkouts are not supported"),
            "checkout by resolved commit hash should explain that detached HEAD is unsupported: {stderr}"
        );
    }

    #[test]
    fn test_list_samples_ref_reads_selected_commit_state() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let (_, feature_update_hash, feature_sample_name) =
            setup_repo_with_feature_only_update(repo_dir.path());

        let main_samples = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(&main_samples, "list-samples on main should succeed");
        let main_stdout = String::from_utf8_lossy(&main_samples.stdout);
        assert!(
            !main_stdout.contains(&feature_sample_name),
            "main branch should not expose the feature-only sample: {main_stdout}"
        );

        let ref_samples = run_gen(
            repo_dir.path(),
            &["list-samples", "--ref", &feature_update_hash],
        );
        assert_success(&ref_samples, "list-samples --ref should succeed");
        let ref_stdout = String::from_utf8_lossy(&ref_samples.stdout);
        assert!(
            ref_stdout.contains(&feature_sample_name),
            "list-samples --ref should expose the selected commit graph state: {ref_stdout}"
        );
    }

    #[test]
    fn test_list_graphs_ref_reads_selected_commit_state() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let (_, feature_update_hash, feature_sample_name) =
            setup_repo_with_feature_only_update(repo_dir.path());

        let main_graphs = run_gen(
            repo_dir.path(),
            &[
                "list-graphs",
                "--name",
                "test-collection",
                "--sample",
                &feature_sample_name,
            ],
        );
        assert_success(
            &main_graphs,
            "list-graphs on main should succeed even when the sample has no graphs",
        );
        assert!(
            String::from_utf8_lossy(&main_graphs.stdout)
                .trim()
                .is_empty(),
            "list-graphs on main should not list graphs for the feature-only sample: {}",
            String::from_utf8_lossy(&main_graphs.stdout)
        );

        let ref_graphs = run_gen(
            repo_dir.path(),
            &[
                "list-graphs",
                "--name",
                "test-collection",
                "--sample",
                &feature_sample_name,
                "--ref",
                &feature_update_hash,
            ],
        );
        assert_success(&ref_graphs, "list-graphs --ref should succeed");
        let ref_stdout = String::from_utf8_lossy(&ref_graphs.stdout);
        assert!(
            ref_stdout.contains("m123"),
            "list-graphs --ref should list graphs from the selected commit: {ref_stdout}"
        );
    }

    #[test]
    fn test_get_sequence_ref_reads_selected_commit_state() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let (_, feature_update_hash, feature_sample_name) =
            setup_repo_with_feature_only_update(repo_dir.path());

        let main_sequence = run_gen(
            repo_dir.path(),
            &[
                "get-sequence",
                "--name",
                "test-collection",
                "--sample",
                &feature_sample_name,
                "--graph",
                "m123",
                "--start",
                "0",
                "--end",
                "4",
            ],
        );
        assert!(
            !main_sequence.status.success(),
            "get-sequence on main should fail for the feature-only sample: stdout={} stderr={}",
            String::from_utf8_lossy(&main_sequence.stdout),
            String::from_utf8_lossy(&main_sequence.stderr)
        );

        let ref_sequence = run_gen(
            repo_dir.path(),
            &[
                "get-sequence",
                "--name",
                "test-collection",
                "--sample",
                &feature_sample_name,
                "--graph",
                "m123",
                "--start",
                "0",
                "--end",
                "4",
                "--ref",
                &feature_update_hash,
            ],
        );
        assert_success(&ref_sequence, "get-sequence --ref should succeed");
        assert_eq!(
            String::from_utf8_lossy(&ref_sequence.stdout).trim(),
            "ATTT",
            "get-sequence --ref should expose the selected commit graph state"
        );
    }

    #[test]
    fn test_export_fasta_ref_reads_selected_commit_state() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let (_, feature_update_hash, feature_sample_name) =
            setup_repo_with_feature_only_update(repo_dir.path());
        let export_path = repo_dir.path().join("feature.fa");

        let export_output = run_gen(
            repo_dir.path(),
            &[
                "export",
                "--ref",
                &feature_update_hash,
                "fasta",
                export_path.to_str().expect("should encode export path"),
                "--name",
                "test-collection",
                "--sample",
                &feature_sample_name,
            ],
        );
        assert_success(&export_output, "export fasta --ref should succeed");

        let exported_fasta =
            fs::read_to_string(&export_path).expect("should read the exported fasta file");
        assert!(
            exported_fasta.contains("TTTT"),
            "export fasta --ref should export sequence data from the selected commit: {exported_fasta}"
        );
    }

    #[test]
    fn test_operations_branch_lists_selected_branch_history() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let update_fasta_path = repo_dir.path().join("feature-update.fa");

        fs::write(&update_fasta_path, ">m123\nTTTT\n")
            .expect("should write branch update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    update_fasta_path
                        .to_str()
                        .expect("should encode update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "feature-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "feature branch update should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "main"]),
            "checkout main should succeed",
        );

        let main_history = operations_stdout(repo_dir.path());
        assert!(
            !main_history.contains("1 sequences inserted"),
            "main branch history should not include the feature-only update: {main_history}"
        );

        let graph = get_connection(repo_dir.path().join(".gen/default.db"))
            .expect("should open graph database before branch history read");
        Collection::create(&graph, "uncommitted-history-read")
            .expect("should create an uncommitted graph change");
        assert!(
            !status_rows(&graph)
                .expect("should read dirty status before branch history read")
                .is_empty(),
            "test repository should be dirty before reading another branch's history"
        );
        drop(graph);

        let feature_history = operations_stdout_for_branch(repo_dir.path(), "feature");
        assert!(
            feature_history.contains("Listing Dolt history for feature"),
            "feature branch output should mention the selected branch: {feature_history}"
        );
        assert!(
            feature_history.contains("1 sequences inserted"),
            "selected branch history should include the feature-only update: {feature_history}"
        );
        let graph = get_connection(repo_dir.path().join(".gen/default.db"))
            .expect("should open graph database for direct history read");
        let history = DoltHistoryStore::new(&graph);
        operations_history_entries(&history, Some("feature"))
            .expect("should read feature history directly");
        assert!(
            !status_rows(&graph)
                .expect("should read status after branch history read")
                .is_empty(),
            "branch history lookup should preserve the current working set"
        );
        assert_eq!(
            history
                .current_branch()
                .expect("should read branch after direct history read")
                .map(|branch| branch.0)
                .as_deref(),
            Some("main"),
            "branch history lookup should not change the history connection's active branch"
        );
    }
}

mod branch_history {
    use super::{
        Path, PathBuf, assert_success, commit_hash_for_summary, fs, get_connection,
        operations_stdout, params, run_gen, status_rows, tempdir,
    };

    fn graph_status(repo_root: &Path) -> String {
        let graph_path = repo_root.join(".gen/default.db");
        let connection = get_connection(graph_path).expect("should reopen graph database");
        let status_rows = status_rows(&connection).expect("should query Dolt status");
        format!("{status_rows:?}")
    }

    fn dirty_graph_with_uncommitted_log(repo_root: &Path) {
        let graph_path = repo_root.join(".gen/default.db");
        let connection = get_connection(graph_path).expect("should reopen graph database");
        connection
            .execute(
                "INSERT INTO gen_operation_log (id, operation_kind, command, created_on)
                 VALUES (?1, ?2, ?3, ?4)",
                params![[1_u8], "test", "dirty working set", 1_i64],
            )
            .expect("should create an uncommitted graph change");
    }

    #[test]
    fn test_merge_brings_feature_graph_state_into_main() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let update_fasta_path = repo_dir.path().join("feature-merge-update.fa");

        fs::write(&update_fasta_path, ">m123\nTTTT\n")
            .expect("should write branch update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    update_fasta_path
                        .to_str()
                        .expect("should encode update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "feature-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "feature branch update should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "main"]),
            "checkout main should succeed",
        );

        let merge_output = run_gen(repo_dir.path(), &["merge", "feature"]);
        assert!(
            merge_output.status.success(),
            "merge should succeed on a clean main branch: stdout={} stderr={} status={}",
            String::from_utf8_lossy(&merge_output.stdout),
            String::from_utf8_lossy(&merge_output.stderr),
            graph_status(repo_dir.path())
        );

        let main_samples = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(&main_samples, "list-samples on merged main should succeed");
        let main_stdout = String::from_utf8_lossy(&main_samples.stdout);
        assert!(
            main_stdout.contains("feature-sample"),
            "merged main branch should expose the feature sample: {main_stdout}"
        );
    }

    #[test]
    fn test_apply_and_reset_update_graph_state() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let update_fasta_path = repo_dir.path().join("feature-apply-update.fa");

        fs::write(&update_fasta_path, ">m123\nTTTT\n")
            .expect("should write branch update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );

        let main_history = operations_stdout(repo_dir.path());
        let import_hash = commit_hash_for_summary(&main_history, "m123:");

        assert_success(
            &run_gen(repo_dir.path(), &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    update_fasta_path
                        .to_str()
                        .expect("should encode update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "feature-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "feature branch update should succeed",
        );

        let feature_history = operations_stdout(repo_dir.path());
        let feature_update_hash = commit_hash_for_summary(&feature_history, "1 sequences inserted");

        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "main"]),
            "checkout main should succeed",
        );
        let main_before_apply = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(&main_before_apply, "list-samples on main should succeed");
        assert!(
            !String::from_utf8_lossy(&main_before_apply.stdout).contains("feature-sample"),
            "main branch should not expose the feature sample before apply"
        );

        assert_success(
            &run_gen(repo_dir.path(), &["apply", &feature_update_hash]),
            "apply should succeed",
        );
        let main_after_apply = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(&main_after_apply, "list-samples after apply should succeed");
        assert!(
            String::from_utf8_lossy(&main_after_apply.stdout).contains("feature-sample"),
            "apply should materialize the feature sample on main"
        );

        assert_success(
            &run_gen(repo_dir.path(), &["reset", &import_hash]),
            "reset should succeed",
        );
        let main_after_reset = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(&main_after_reset, "list-samples after reset should succeed");
        assert!(
            !String::from_utf8_lossy(&main_after_reset.stdout).contains("feature-sample"),
            "reset should remove the applied feature sample from main"
        );
    }

    #[test]
    fn test_reset_then_new_commit_replaces_head_lineage() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let first_update_path = repo_dir.path().join("first-update.fa");
        let second_update_path = repo_dir.path().join("second-update.fa");

        fs::write(&first_update_path, ">m123\nTTTT\n")
            .expect("should write first update fasta fixture");
        fs::write(&second_update_path, ">m123\nCCCC\n")
            .expect("should write second update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    first_update_path
                        .to_str()
                        .expect("should encode first update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "first-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "first update should succeed",
        );

        let history_after_first_update = operations_stdout(repo_dir.path());
        let import_hash = commit_hash_for_summary(&history_after_first_update, "m123:");

        assert_success(
            &run_gen(repo_dir.path(), &["reset", &import_hash]),
            "reset should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    second_update_path
                        .to_str()
                        .expect("should encode second update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "second-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "second update should succeed",
        );

        let final_history = operations_stdout(repo_dir.path());
        assert_eq!(
            final_history.matches("1 sequences inserted").count(),
            1,
            "active history should contain only one reachable post-import update after reset: {final_history}"
        );

        let sample_list = run_gen(repo_dir.path(), &["list-samples"]);
        assert_success(
            &sample_list,
            "list-samples should succeed after reset and recommit",
        );
        let sample_stdout = String::from_utf8_lossy(&sample_list.stdout);
        assert!(
            sample_stdout.contains("second-sample"),
            "new post-reset sample should be visible: {sample_stdout}"
        );
        assert!(
            !sample_stdout.contains("first-sample"),
            "reset should remove the earlier branch-local sample from the working set: {sample_stdout}"
        );
    }

    #[test]
    fn test_ref_changing_commands_fail_when_working_set_is_dirty() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let update_fasta_path = repo_dir.path().join("feature-dirty-update.fa");

        fs::write(&update_fasta_path, ">m123\nTTTT\n")
            .expect("should write branch update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );

        let main_history = operations_stdout(repo_dir.path());
        let import_hash = commit_hash_for_summary(&main_history, "m123:");

        assert_success(
            &run_gen(repo_dir.path(), &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    update_fasta_path
                        .to_str()
                        .expect("should encode update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "feature-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "feature branch update should succeed",
        );
        let feature_history = operations_stdout(repo_dir.path());
        let feature_update_hash = commit_hash_for_summary(&feature_history, "1 sequences inserted");

        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "main"]),
            "checkout main should succeed",
        );
        dirty_graph_with_uncommitted_log(repo_dir.path());

        for (args, action) in [
            (vec!["checkout", "feature"], "checkout"),
            (vec!["reset", import_hash.as_str()], "reset"),
            (vec!["apply", feature_update_hash.as_str()], "apply"),
            (vec!["merge", "feature"], "merge"),
        ] {
            let output = run_gen(repo_dir.path(), &args);
            assert!(
                !output.status.success(),
                "{action} should fail on a dirty working set: stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(&format!(
                    "Cannot {action}: the working set has uncommitted changes."
                )),
                "{action} should explain that Dolt status is dirty: {stderr}"
            );
        }
    }

    #[test]
    fn test_merge_conflict_is_reported_as_a_gen_error() {
        let repo_dir = tempdir().expect("should create temp repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let feature_update_path = repo_dir.path().join("feature-conflict.fa");
        let main_update_path = repo_dir.path().join("main-conflict.fa");

        fs::write(&feature_update_path, ">m123\nTTTT\n")
            .expect("should write feature update fasta fixture");
        fs::write(&main_update_path, ">m123\nGGGG\n")
            .expect("should write main update fasta fixture");

        assert_success(
            &run_gen(repo_dir.path(), &["init"]),
            "gen init should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "fasta import should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["branch", "--create", "feature"]),
            "branch create should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "--branch", "feature"]),
            "checkout feature should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    feature_update_path
                        .to_str()
                        .expect("should encode feature update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "edited-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "feature branch update should succeed",
        );
        assert_success(
            &run_gen(repo_dir.path(), &["checkout", "main"]),
            "checkout main should succeed",
        );
        assert_success(
            &run_gen(
                repo_dir.path(),
                &[
                    "update",
                    "fasta",
                    main_update_path
                        .to_str()
                        .expect("should encode main update fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                    "--new-sample",
                    "edited-sample",
                    "--region-name",
                    "m123:1-5",
                ],
            ),
            "main branch update should succeed",
        );

        let merge_output = run_gen(repo_dir.path(), &["merge", "feature"]);
        assert!(
            !merge_output.status.success(),
            "merge should fail when Dolt reports a conflict: stdout={} stderr={}",
            String::from_utf8_lossy(&merge_output.stdout),
            String::from_utf8_lossy(&merge_output.stderr)
        );
        let stderr = String::from_utf8_lossy(&merge_output.stderr);
        assert!(
            stderr.contains("Merge failed with Dolt conflicts."),
            "merge conflict should be reported as a Gen-level conflict error: {stderr}"
        );
    }
}

mod remotes {
    use super::{
        Connection, GraphConnection, Path, PathBuf, assert_success, asset_refs, fs, get_connection,
        operations_stdout, run_gen, status_rows, tempdir,
    };

    fn resolved_ref_hash(repo_root: &Path, reference: &str) -> String {
        let connection = get_connection(repo_root.join(".gen/default.db"))
            .expect("should open graph database for ref lookup");
        connection
            .query_row("SELECT dolt_hashof(?1)", [reference], |row| row.get(0))
            .unwrap_or_else(|error| panic!("should resolve {reference}: {error}"))
    }

    fn raw_graph_status(repo_root: &Path) -> Vec<gen_models::history::dolt::DoltStatusRow> {
        let connection = Connection::open(repo_root.join(".gen/default.db"))
            .expect("should open graph database without migrations");
        rusqlite::vtab::array::load_module(&connection).expect("should load array module");
        status_rows(&GraphConnection(connection)).expect("should query raw Dolt status")
    }

    fn collect_files(root: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let entries = fs::read_dir(root).expect("should read directory entries");
        for entry in entries {
            let entry = entry.expect("should read directory entry");
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_files(&path));
            } else {
                files.push(path);
            }
        }
        files
    }

    fn assert_remote_asset_is_not_materialized(repo_root: &Path) {
        let repo_asset_path = repo_root.join("simple.gff");
        assert!(
            !repo_asset_path.exists(),
            "remote transfer should keep asset payloads out of the repo checkout: {}",
            repo_asset_path.display()
        );

        let asset_cache_dir = repo_root.join(".gen/assets");
        let asset_cache_files = collect_files(&asset_cache_dir);
        assert!(
            asset_cache_files.is_empty(),
            "remote transfer should keep .gen/assets as an empty cache until materialization is requested: {asset_cache_files:?}"
        );
    }

    #[test]
    fn test_clone_of_schema_less_main_is_clean_and_can_checkout_schema_branch() {
        let remote_repo_dir = tempdir().expect("should create temp remote repo directory");
        let clone_parent_dir = tempdir().expect("should create temp clone parent directory");
        let remote_gen_dir = remote_repo_dir.path().join(".gen");
        fs::create_dir(&remote_gen_dir).expect("should create remote Gen directory");
        let remote_graph_path = remote_gen_dir.join("default.db");

        let raw_remote =
            Connection::open(&remote_graph_path).expect("should open raw remote graph");
        raw_remote
            .query_row("SELECT dolt_branch('foobar')", [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("should create foobar from the schema-less main branch");
        raw_remote
            .query_row("SELECT dolt_checkout('foobar')", [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("should check out foobar");
        raw_remote
            .query_row("SELECT dolt_config('user.name', 'Test User')", [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("should configure test committer name");
        raw_remote
            .query_row(
                "SELECT dolt_config('user.email', 'test@example.com')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("should configure test committer email");
        raw_remote
            .execute_batch(
                "CREATE TABLE branch_only(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO branch_only VALUES(1, 'visible on foobar');",
            )
            .expect("should create branch-only graph state");
        raw_remote
            .query_row(
                "SELECT dolt_commit('-A', '-m', 'Add branch-only state')",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("should commit the foobar state");
        raw_remote
            .query_row("SELECT dolt_checkout('main')", [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("should leave the remote on schema-less main");
        drop(raw_remote);

        let remote_url = format!("file://{}", remote_repo_dir.path().display());
        assert_success(
            &run_gen(clone_parent_dir.path(), &["clone", &remote_url]),
            "clone from schema-branch remote should succeed",
        );
        let cloned_repo_path = clone_parent_dir.path().join(
            remote_repo_dir
                .path()
                .file_name()
                .expect("remote temp dir should have a basename"),
        );

        assert!(
            raw_graph_status(&cloned_repo_path).is_empty(),
            "clone should not apply uncommitted migrations to main"
        );
        let cloned_graph = Connection::open(cloned_repo_path.join(".gen/default.db"))
            .expect("should open cloned graph for ref inspection");
        let active_branch: String = cloned_graph
            .query_row("SELECT active_branch()", [], |row| row.get(0))
            .expect("should read cloned active branch");
        let branch_names = cloned_graph
            .prepare("SELECT name FROM dolt_branches ORDER BY name")
            .expect("should prepare cloned branch query")
            .query_map([], |row| row.get::<_, String>(0))
            .expect("should query cloned branches")
            .collect::<Result<Vec<_>, _>>()
            .expect("should collect cloned branches");
        assert_eq!(active_branch, "main");
        assert!(
            branch_names.iter().any(|name| name == "foobar"),
            "clone should retain foobar; found {branch_names:?}"
        );
        drop(cloned_graph);
        assert_success(
            &run_gen(&cloned_repo_path, &["checkout", "foobar"]),
            "checkout of the schema branch after clone should succeed",
        );
        assert!(
            raw_graph_status(&cloned_repo_path).is_empty(),
            "checking out the migrated branch should remain clean"
        );
    }

    #[test]
    fn test_clone_from_file_remote_preserves_history_and_asset_refs() {
        let remote_repo_dir = tempdir().expect("should create temp remote repo directory");
        let clone_parent_dir = tempdir().expect("should create temp clone parent directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let gff_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(remote_repo_dir.path(), &["init"]),
            "remote gen init should succeed",
        );
        assert_success(
            &run_gen(
                remote_repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "remote fasta import should succeed",
        );
        assert_success(
            &run_gen(
                remote_repo_dir.path(),
                &[
                    "add-file",
                    gff_path.to_str().expect("should encode gff path"),
                    "--message",
                    "add-gff",
                ],
            ),
            "remote add-file should succeed",
        );
        let remote_graph = get_connection(remote_repo_dir.path().join(".gen/default.db"))
            .expect("should open remote graph for ref setup");
        let _: i64 = remote_graph
            .query_row("SELECT dolt_branch('feature')", [], |row| row.get(0))
            .expect("should create feature ref before clone");
        let _: i64 = remote_graph
            .query_row("SELECT dolt_tag('v1')", [], |row| row.get(0))
            .expect("should create tag before clone");
        drop(remote_graph);

        let remote_url = format!("file://{}", remote_repo_dir.path().display());
        assert_success(
            &run_gen(clone_parent_dir.path(), &["clone", &remote_url]),
            "clone from file remote should succeed",
        );

        let cloned_repo_path = clone_parent_dir.path().join(
            remote_repo_dir
                .path()
                .file_name()
                .expect("remote temp dir should have a basename"),
        );
        let cloned_history = operations_stdout(&cloned_repo_path);
        assert!(
            cloned_history.contains("m123:"),
            "cloned history should include the fasta import commit: {cloned_history}"
        );
        assert!(
            cloned_history.contains("add-gff"),
            "cloned history should include the add-file commit: {cloned_history}"
        );

        let refs = asset_refs(&cloned_repo_path);
        assert!(
            refs.iter()
                .any(|asset_ref| asset_ref.name.as_deref() == Some("simple.gff")),
            "cloned graph db should retain the asset reference rows: {refs:?}"
        );
        assert_remote_asset_is_not_materialized(&cloned_repo_path);
        assert_eq!(
            resolved_ref_hash(&cloned_repo_path, "feature"),
            resolved_ref_hash(remote_repo_dir.path(), "feature"),
            "clone should preserve branch refs"
        );
        assert_eq!(
            resolved_ref_hash(&cloned_repo_path, "v1"),
            resolved_ref_hash(remote_repo_dir.path(), "v1"),
            "clone should preserve tag refs"
        );
    }

    #[test]
    fn test_push_to_file_remote_and_clone_preserves_history_and_asset_refs() {
        let local_repo_dir = tempdir().expect("should create temp local repo directory");
        let remote_repo_dir = tempdir().expect("should create temp remote repo directory");
        let clone_parent_dir = tempdir().expect("should create temp clone parent directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let gff_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(local_repo_dir.path(), &["init"]),
            "local gen init should succeed",
        );
        assert_success(
            &run_gen(remote_repo_dir.path(), &["init"]),
            "remote gen init should succeed",
        );
        assert_success(
            &run_gen(
                local_repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "local fasta import should succeed",
        );
        assert_success(
            &run_gen(
                local_repo_dir.path(),
                &[
                    "add-file",
                    gff_path.to_str().expect("should encode gff path"),
                    "--message",
                    "add-gff",
                ],
            ),
            "local add-file should succeed",
        );

        let remote_url = format!("file://{}", remote_repo_dir.path().display());
        assert_success(
            &run_gen(
                local_repo_dir.path(),
                &["remote", "add", "origin", &remote_url],
            ),
            "remote add should succeed",
        );
        assert_success(
            &run_gen(local_repo_dir.path(), &["remote", "set-default", "origin"]),
            "remote set-default should succeed",
        );
        let local_graph = get_connection(local_repo_dir.path().join(".gen/default.db"))
            .expect("should open local graph for dirty push setup");
        local_graph
            .execute_batch(
                "CREATE TABLE uncommitted_marker(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO uncommitted_marker VALUES(1, 'local only');",
            )
            .expect("should create an unrelated dirty working-tree change");
        drop(local_graph);
        assert_success(
            &run_gen(local_repo_dir.path(), &["push"]),
            "push should permit an unrelated dirty working tree",
        );
        assert_eq!(
            resolved_ref_hash(local_repo_dir.path(), "origin/main"),
            resolved_ref_hash(local_repo_dir.path(), "main"),
            "post-push fetch should update the accepted tracking ref"
        );

        assert_success(
            &run_gen(clone_parent_dir.path(), &["clone", &remote_url]),
            "clone from pushed file remote should succeed",
        );

        let cloned_repo_path = clone_parent_dir.path().join(
            remote_repo_dir
                .path()
                .file_name()
                .expect("remote temp dir should have a basename"),
        );
        let cloned_history = operations_stdout(&cloned_repo_path);
        assert!(
            cloned_history.contains("m123:"),
            "cloned history should include the fasta import commit: {cloned_history}"
        );
        assert!(
            cloned_history.contains("add-gff"),
            "cloned history should include the add-file commit: {cloned_history}"
        );

        let refs = asset_refs(&cloned_repo_path);
        assert!(
            refs.iter()
                .any(|asset_ref| asset_ref.name.as_deref() == Some("simple.gff")),
            "cloned graph db should retain the pushed asset reference rows: {refs:?}"
        );
        assert_remote_asset_is_not_materialized(&cloned_repo_path);
        let cloned_graph = get_connection(cloned_repo_path.join(".gen/default.db"))
            .expect("should open clone for dirty marker check");
        let dirty_marker_exists: bool = cloned_graph
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name = 'uncommitted_marker')",
                [],
                |row| row.get(0),
            )
            .expect("should query cloned schema");
        assert!(
            !dirty_marker_exists,
            "push should transfer only committed history, not dirty working-tree changes"
        );
    }

    #[test]
    fn test_pull_rejects_an_absent_graph_without_creating_it() {
        let remote_repo_dir = tempdir().expect("should create temp remote repo directory");
        let local_repo_dir = tempdir().expect("should create temp local repo directory");
        let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fasta_path = fixtures_dir.join("simple.fa");
        let gff_path = fixtures_dir.join("simple.gff");

        assert_success(
            &run_gen(remote_repo_dir.path(), &["init"]),
            "remote gen init should succeed",
        );
        assert_success(
            &run_gen(
                remote_repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "remote fasta import should succeed",
        );
        assert_success(
            &run_gen(
                remote_repo_dir.path(),
                &[
                    "add-file",
                    gff_path.to_str().expect("should encode gff path"),
                    "--message",
                    "add-gff",
                ],
            ),
            "remote add-file should succeed",
        );

        assert_success(
            &run_gen(local_repo_dir.path(), &["init"]),
            "local gen init should succeed",
        );

        let remote_url = format!("file://{}", remote_repo_dir.path().display());
        assert_success(
            &run_gen(
                local_repo_dir.path(),
                &["remote", "add", "origin", &remote_url],
            ),
            "remote add should succeed",
        );
        assert_success(
            &run_gen(local_repo_dir.path(), &["remote", "set-default", "origin"]),
            "remote set-default should succeed",
        );
        assert!(
            !local_repo_dir.path().join(".gen/default.db").exists(),
            "remote configuration should not create the graph database before the first pull"
        );

        let pull = run_gen(local_repo_dir.path(), &["pull"]);
        assert!(
            !pull.status.success(),
            "pull should reject a workspace without a graph database"
        );
        assert!(
            String::from_utf8_lossy(&pull.stderr).contains("gen clone"),
            "pull should direct the user to the non-destructive initialization command: {}",
            String::from_utf8_lossy(&pull.stderr)
        );
        assert!(
            !local_repo_dir.path().join(".gen/default.db").exists(),
            "rejected pull should not create or replace the graph database"
        );
    }

    #[test]
    fn test_pull_preserves_an_existing_graph_database_with_uncommitted_changes() {
        let remote_repo_dir = tempdir().expect("should create temp remote repo directory");
        let local_repo_dir = tempdir().expect("should create temp local repo directory");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");

        assert_success(
            &run_gen(remote_repo_dir.path(), &["init"]),
            "remote gen init should succeed",
        );
        assert_success(
            &run_gen(
                remote_repo_dir.path(),
                &[
                    "import",
                    "fasta",
                    fasta_path.to_str().expect("should encode fasta path"),
                    "--name",
                    "test-collection",
                    "--sample",
                    "test-sample",
                ],
            ),
            "remote fasta import should succeed",
        );
        assert_success(
            &run_gen(local_repo_dir.path(), &["init"]),
            "local gen init should succeed",
        );
        let remote_url = format!("file://{}", remote_repo_dir.path().display());
        assert_success(
            &run_gen(
                local_repo_dir.path(),
                &["remote", "add", "origin", &remote_url],
            ),
            "remote add should succeed",
        );
        assert_success(
            &run_gen(local_repo_dir.path(), &["remote", "set-default", "origin"]),
            "remote set-default should succeed",
        );
        let local_graph = get_connection(local_repo_dir.path().join(".gen/default.db"))
            .expect("should create the local graph database");
        local_graph
            .execute_batch(
                "CREATE TABLE user_work(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO user_work VALUES(1, 'must survive failed pull');",
            )
            .expect("should create uncommitted user data");
        drop(local_graph);

        let pull = run_gen(local_repo_dir.path(), &["pull"]);
        assert!(
            !pull.status.success(),
            "pull should let Dolt reject an existing dirty database instead of replacing it"
        );
        let preserved_graph = get_connection(local_repo_dir.path().join(".gen/default.db"))
            .expect("the existing graph database should remain readable");
        let preserved_value: String = preserved_graph
            .query_row("SELECT value FROM user_work WHERE id = 1", [], |row| {
                row.get(0)
            })
            .expect("uncommitted user data should survive the rejected pull");
        assert_eq!(preserved_value, "must survive failed pull");
    }

    #[test]
    fn test_clone_rejects_existing_destination_without_removing_it() {
        let clone_parent = tempdir().expect("should create clone parent");
        let destination = clone_parent.path().join("existing-destination");
        fs::create_dir(&destination).expect("should create existing destination");
        let sentinel = destination.join("keep.txt");
        fs::write(&sentinel, "keep me").expect("should create destination sentinel");

        let output = run_gen(
            clone_parent.path(),
            &[
                "clone",
                "http://127.0.0.1:1/api/repos/alice/existing-destination",
            ],
        );

        assert!(
            !output.status.success(),
            "clone should reject an existing destination"
        );
        assert!(
            sentinel.exists(),
            "clone must not remove a pre-existing destination"
        );
    }

    #[test]
    fn test_clone_failure_removes_only_the_new_destination() {
        let clone_parent = tempdir().expect("should create clone parent");
        let unrelated = clone_parent.path().join("unrelated.txt");
        fs::write(&unrelated, "keep me").expect("should create unrelated file");
        let destination = clone_parent.path().join("failed-clone");

        let output = run_gen(
            clone_parent.path(),
            &["clone", "http://127.0.0.1:1/api/repos/alice/failed-clone"],
        );

        assert!(
            !output.status.success(),
            "clone should fail when GenHub is unavailable"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("failed to restore the canonical URL")
                && !stderr.contains("remote not found"),
            "clone failure before graph-remote creation should not emit a restoration warning: {stderr}"
        );
        assert!(
            !destination.exists(),
            "a failed clone should remove the destination it created"
        );
        assert!(
            unrelated.exists(),
            "clone cleanup must not remove unrelated paths"
        );
    }
}
