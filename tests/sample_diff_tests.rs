use std::{collections::HashSet, path::PathBuf};

use r#gen::{
    imports::fasta::import_fasta,
    test_helpers::setup_gen,
    updates::{sequence::update_with_sequence, vcf::update_with_vcf},
    views::diff_graph::build_diff_graph_component,
};
use gen_diff::{
    graph::DiffChangeKind,
    sample::{SampleDiff, build_sample_diff},
};
use gen_models::{db::DbContext, node::Node, sample::Sample};
use ratatui::style::Color;

fn simple_vcf_sample_diff(query_name: &str, base_name: &str) -> (DbContext, SampleDiff) {
    let context = setup_gen();
    let collection_name = "test";
    let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
    let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
    let fasta_path = fasta_path
        .to_str()
        .expect("should encode simple FASTA fixture path")
        .to_string();
    let vcf_path = vcf_path
        .to_str()
        .expect("should encode simple VCF fixture path")
        .to_string();

    import_fasta(
        &context,
        &fasta_path,
        collection_name,
        "reference",
        false,
        &[],
    )
    .expect("should import simple FASTA fixture");
    update_with_vcf(
        &context,
        &vcf_path,
        collection_name,
        String::new(),
        None,
        vec!["reference".to_string()],
        false,
    )
    .expect("should update simple FASTA with simple VCF");

    let diff = build_sample_diff(
        context.graph().conn(),
        collection_name,
        "m123",
        query_name,
        base_name,
        None,
    )
    .expect("should build sample diff");
    (context, diff)
}

fn changed_sequences(
    context: &DbContext,
    diff: &SampleDiff,
    change_kind: DiffChangeKind,
) -> HashSet<String> {
    let node_ids = diff
        .graph
        .nodes()
        .map(|node| node.node.node_id)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let sequences_by_node_id = Node::get_sequences_by_node_ids(
        context.graph().conn(),
        context.workspace(),
        &node_ids,
        None,
    );

    diff.graph
        .nodes()
        .filter(|node| {
            node.change.kind == change_kind && node.node.sequence_start < node.node.sequence_end
        })
        .map(|node| {
            sequences_by_node_id[&node.node.node_id]
                .get_sequence(node.node.sequence_start, node.node.sequence_end)
                .expect("should read changed graph-node sequence")
        })
        .collect()
}

#[test]
fn test_query_foo_base_unknown_marks_unknown_insertion_removed() {
    let (context, diff) = simple_vcf_sample_diff("foo", "unknown");

    assert_eq!(diff.query_block_group.sample_name, "foo");
    assert_eq!(diff.base_block_group.sample_name, "unknown");
    assert_eq!(
        changed_sequences(&context, &diff, DiffChangeKind::Removed),
        HashSet::from(["AGA".to_string()]),
        "base-only insertion should be removed"
    );
    assert!(
        changed_sequences(&context, &diff, DiffChangeKind::Added).is_empty(),
        "foo should not add sequence relative to unknown"
    );

    let node_ids = diff
        .graph
        .nodes()
        .map(|node| node.node.node_id)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let sequences_by_node_id = Node::get_sequences_by_node_ids(
        context.graph().conn(),
        context.workspace(),
        &node_ids,
        None,
    );
    let reference_node_id = sequences_by_node_id
        .iter()
        .find_map(|(node_id, sequence)| {
            (sequence
                .get_sequence(0, 34)
                .is_ok_and(|sequence| sequence == "ATCGATCGATCGATCGATCGGGAACACACAGAGA"))
            .then_some(*node_id)
        })
        .expect("should retain the simple FASTA backing node");
    let shared_reference_slices = diff
        .graph
        .nodes()
        .filter(|node| node.node.node_id == reference_node_id)
        .map(|node| {
            (
                node.node.sequence_start,
                node.node.sequence_end,
                node.change.kind,
            )
        })
        .collect::<HashSet<_>>();
    assert!(
        shared_reference_slices.contains(&(4, 10, DiffChangeKind::Unchanged))
            && shared_reference_slices.contains(&(10, 34, DiffChangeKind::Unchanged)),
        "the base-only insertion boundary should split shared query sequence: {shared_reference_slices:?}"
    );

    let neutral_query_continuation = diff.graph.all_edges().any(|(source, target, edges)| {
        source.node.node_id == reference_node_id
            && source.node.sequence_start == 4
            && source.node.sequence_end == 10
            && target.node.node_id == reference_node_id
            && target.node.sequence_start == 10
            && target.node.sequence_end == 34
            && edges
                .iter()
                .all(|edge| edge.change.kind == DiffChangeKind::Unchanged)
    });
    assert!(
        neutral_query_continuation,
        "foo's boundary reconnection around unknown's insertion should stay neutral"
    );

    let component = build_diff_graph_component(&diff.graph, String::new());
    assert!(
        component
            .highlighted_nodes
            .iter()
            .any(|(_, color)| *color == Color::Red),
        "removed base-only sequence should use the existing red diff highlight"
    );
    assert!(
        !component
            .highlighted_edges
            .iter()
            .any(|((source, target), _)| {
                source.node_id == reference_node_id
                    && source.sequence_start == 4
                    && source.sequence_end == 10
                    && target.node_id == reference_node_id
                    && target.sequence_start == 10
                    && target.sequence_end == 34
            }),
        "boundary reconnection should not receive a diff highlight"
    );
}

#[test]
fn test_reversing_query_and_base_marks_unknown_insertion_added() {
    let (context, diff) = simple_vcf_sample_diff("unknown", "foo");

    assert_eq!(
        changed_sequences(&context, &diff, DiffChangeKind::Added),
        HashSet::from(["AGA".to_string()]),
        "query-only insertion should be added"
    );
    assert!(
        changed_sequences(&context, &diff, DiffChangeKind::Removed).is_empty(),
        "unknown should not remove sequence relative to foo"
    );
}

/// Imports the simple reference, applies `(parent, child, region, sequence)`
/// sequence updates in order, then diffs `query_name` against `base_name`.
fn sequence_update_sample_diff(
    updates: &[(&str, &str, &str, &str)],
    query_name: &str,
    base_name: &str,
) -> (DbContext, SampleDiff) {
    let context = setup_gen();
    let collection_name = "test";
    let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
    import_fasta(
        &context,
        &fasta_path
            .to_str()
            .expect("should encode FASTA fixture path")
            .to_string(),
        collection_name,
        Sample::DEFAULT_NAME,
        false,
        &[],
    )
    .expect("should import simple FASTA fixture");
    for (parent, child, region, sequence) in updates {
        update_with_sequence(
            &context,
            collection_name,
            parent,
            child,
            region,
            sequence,
            false,
        )
        .expect("should apply sequence update");
    }
    let diff = build_sample_diff(
        context.graph().conn(),
        collection_name,
        "m123",
        query_name,
        base_name,
        None,
    )
    .expect("should build sample diff");
    (context, diff)
}

/// Counts non-neutral edges of one change kind, ignoring continuation edges.
fn changed_edge_count(diff: &SampleDiff, change_kind: DiffChangeKind) -> usize {
    diff.graph
        .all_edges()
        .flat_map(|(_, _, edges)| edges.iter())
        .filter(|edge| edge.change.kind == change_kind)
        .count()
}

/// Summarizes a diff as (added sequences, removed sequences, added edges, removed edges).
fn change_summary(
    context: &DbContext,
    diff: &SampleDiff,
) -> (HashSet<String>, HashSet<String>, usize, usize) {
    (
        changed_sequences(context, diff, DiffChangeKind::Added),
        changed_sequences(context, diff, DiffChangeKind::Removed),
        changed_edge_count(diff, DiffChangeKind::Added),
        changed_edge_count(diff, DiffChangeKind::Removed),
    )
}

fn sequences(values: &[&str]) -> HashSet<String> {
    values.iter().map(|value| value.to_string()).collect()
}

#[test]
fn test_snp_adds_one_node_and_two_edges_without_split_bookkeeping() {
    let updates = [(Sample::DEFAULT_NAME, "snp", "m123:10-11", "G")];

    let (context, diff) = sequence_update_sample_diff(&updates, "snp", Sample::DEFAULT_NAME);
    assert_eq!(
        change_summary(&context, &diff),
        (sequences(&["G"]), sequences(&[]), 2, 0)
    );

    let (context, diff) = sequence_update_sample_diff(&updates, Sample::DEFAULT_NAME, "snp");
    assert_eq!(
        change_summary(&context, &diff),
        (sequences(&[]), sequences(&["G"]), 0, 2)
    );
}

#[test]
fn test_deletion_is_a_single_edge_without_node_changes() {
    let updates = [(Sample::DEFAULT_NAME, "del", "m123:10-14", "")];

    let (context, diff) = sequence_update_sample_diff(&updates, "del", Sample::DEFAULT_NAME);
    assert_eq!(
        change_summary(&context, &diff),
        (sequences(&[]), sequences(&[]), 1, 0)
    );

    let (context, diff) = sequence_update_sample_diff(&updates, Sample::DEFAULT_NAME, "del");
    assert_eq!(
        change_summary(&context, &diff),
        (sequences(&[]), sequences(&[]), 0, 1)
    );
}

#[test]
fn test_edit_on_edit_marks_the_whole_extra_subgraph() {
    let updates = [
        (Sample::DEFAULT_NAME, "child", "m123:2-5", "AAAAAAAA"),
        ("child", "grandchild", "m123:4-6", "TTTTTTTT"),
    ];

    let (context, diff) = sequence_update_sample_diff(&updates, Sample::DEFAULT_NAME, "grandchild");
    assert_eq!(
        change_summary(&context, &diff),
        (sequences(&[]), sequences(&["AA", "AAAA", "TTTTTTTT"]), 0, 4)
    );

    let (context, diff) = sequence_update_sample_diff(&updates, "child", "grandchild");
    assert_eq!(
        change_summary(&context, &diff),
        (sequences(&[]), sequences(&["TTTTTTTT"]), 0, 2)
    );
}
