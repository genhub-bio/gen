use std::collections::{HashMap, HashSet};

use gen_core::{
    CommitHash, HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, errors::ConfigError, is_terminal,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{AugmentedEdge, BlockGroupEdge},
    db::GraphConnection,
    errors::OperationError,
    history::{HistoryStore, dolt::DoltHistoryStore},
};
use rusqlite::params;
use thiserror::Error;

use crate::graph::{
    DiffChange, DiffChangeKind, DiffGenGraph, DiffGraphEdge, DiffGraphNode, DiffPresence,
};

#[derive(Debug, Error)]
pub enum OperationDiffError {
    #[error("No current operation is checked out.")]
    NoCurrentOperation,
    #[error("Operation {0} not found.")]
    OperationMissing(HashId),
    #[error("Unable to find path between {0} and {1}.")]
    PathNotFound(HashId, HashId),
    #[error("Invalid operation hash for Dolt history lookup: {0}")]
    InvalidCommitHash(HashId),
    #[error("Config error: {0}")]
    Config(#[from] ConfigError),
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    OperationError(#[from] OperationError),
}

#[derive(Clone, Debug)]
pub struct BlockGroupDiff {
    pub id: HashId,
    pub block_group: Option<BlockGroup>,
    pub presence: DiffPresence,
    pub graph: DiffGenGraph,
}

#[derive(Clone, Debug)]
pub struct OperationDiff {
    pub operations: Vec<HashId>,
    pub diff_graph: Vec<BlockGroupDiff>,
}

impl OperationDiff {
    fn empty() -> Self {
        Self {
            operations: Vec::new(),
            diff_graph: Vec::new(),
        }
    }
}

pub fn collect_operation_diff(
    graph_conn: &GraphConnection,
    from_hash: Option<HashId>,
    to_hash: HashId,
) -> Result<OperationDiff, OperationDiffError> {
    let history_store = DoltHistoryStore::new(graph_conn);
    let history_selection = history_operations_in_order(&history_store, from_hash, to_hash)?;
    if history_selection.operations_in_order.is_empty() {
        return Ok(OperationDiff::empty());
    }

    build_dolt_operation_diff(graph_conn, &history_selection, to_hash)
}

fn build_dolt_operation_diff(
    graph_conn: &GraphConnection,
    history_selection: &HistoryOperationSelection,
    to_hash: HashId,
) -> Result<OperationDiff, OperationDiffError> {
    let to_ref = operation_history_ref(to_hash)?;
    let from_ref = history_selection
        .range_start_hash
        .map(operation_history_ref)
        .transpose()?
        .unwrap_or_else(|| format!("{to_ref}~1"));
    let default_operation = history_selection
        .operations_in_order
        .last()
        .copied()
        .ok_or(OperationDiffError::OperationMissing(to_hash))?;
    let changes = collect_block_group_changes(
        graph_conn,
        &from_ref,
        &to_ref,
        Some(&history_selection.target_history_operations),
    )?;
    let changed_node_ids = collect_node_changes(
        graph_conn,
        &from_ref,
        &to_ref,
        Some(&history_selection.target_history_operations),
    )?;
    let changed_block_groups = changes.presence_changes;
    let edge_changes_by_block_group = changes.edge_changes;
    let block_group_operations_by_id = changes.operations_by_block_group_id;

    let mut changed_block_group_ids = changed_block_groups.keys().copied().collect::<Vec<_>>();
    changed_block_group_ids.sort();
    let build_context = OperationDiffBuildContext {
        from_ref: &from_ref,
        to_ref: &to_ref,
        default_operation,
        edge_changes_by_block_group: &edge_changes_by_block_group,
        block_group_operations_by_id: &block_group_operations_by_id,
        changed_node_ids: &changed_node_ids,
    };
    let mut diff_graph = Vec::new();
    for block_group_id in changed_block_group_ids {
        if !changed_block_groups.contains_key(&block_group_id) {
            continue;
        }
        if let Some(block_group_diff) =
            build_block_group_diff(graph_conn, block_group_id, &build_context)?
        {
            diff_graph.push(block_group_diff);
        }
    }

    Ok(OperationDiff {
        operations: history_selection.operations_in_order.clone(),
        diff_graph,
    })
}

struct HistoryOperationSelection {
    operations_in_order: Vec<HashId>,
    target_history_operations: HashSet<HashId>,
    range_start_hash: Option<HashId>,
}

fn history_operations_in_order(
    history_store: &impl HistoryStore,
    from_hash: Option<HashId>,
    to_hash: HashId,
) -> Result<HistoryOperationSelection, OperationDiffError> {
    if from_hash == Some(to_hash) {
        return Ok(HistoryOperationSelection {
            operations_in_order: Vec::new(),
            target_history_operations: HashSet::new(),
            range_start_hash: None,
        });
    }

    let history_entries = history_store.log(None)?;
    let history_hashes = history_entries
        .into_iter()
        .rev()
        .map(|entry| HashId::pad_str(&entry.commit_hash.0))
        .collect::<Vec<_>>();
    let target_history_operations = history_hashes.iter().copied().collect();
    let to_index = history_hashes
        .iter()
        .position(|candidate| *candidate == to_hash)
        .ok_or(OperationDiffError::OperationMissing(to_hash))?;

    let (operations_in_order, range_start_hash) = match from_hash {
        None => (
            history_hashes[..=to_index]
                .last()
                .copied()
                .into_iter()
                .collect(),
            None,
        ),
        Some(from_hash) => match history_hashes
            .iter()
            .position(|candidate| *candidate == from_hash)
        {
            Some(from_index) if from_index > to_index => {
                return Err(OperationDiffError::PathNotFound(from_hash, to_hash));
            }
            Some(from_index) => (
                history_hashes[from_index + 1..=to_index].to_vec(),
                Some(from_hash),
            ),
            None if to_index <= 1 => (vec![to_hash], None),
            None => (vec![from_hash, to_hash], Some(from_hash)),
        },
    };
    Ok(HistoryOperationSelection {
        operations_in_order,
        target_history_operations,
        range_start_hash,
    })
}

fn operation_history_ref(hash: HashId) -> Result<String, OperationDiffError> {
    let commit_hash = operation_hash_to_commit_hash(hash)?;
    Ok(commit_hash.0)
}

fn operation_hash_to_commit_hash(hash: HashId) -> Result<CommitHash, OperationDiffError> {
    let hash = hash.to_string();
    if hash.len() < 40 {
        return Err(OperationDiffError::InvalidCommitHash(HashId::convert_str(
            hash.as_str(),
        )));
    }
    let commit_hash = hash[hash.len() - 40..].to_string();
    if !commit_hash
        .chars()
        .all(|character| character.is_ascii_hexdigit())
    {
        return Err(OperationDiffError::InvalidCommitHash(HashId::convert_str(
            hash.as_str(),
        )));
    }
    Ok(CommitHash(commit_hash))
}

fn operation_hash_from_commit(commit: Option<&str>) -> Option<HashId> {
    commit
        .filter(|commit| !commit.is_empty())
        .map(HashId::pad_str)
}

fn build_block_group_diff(
    graph_conn: &GraphConnection,
    block_group_id: HashId,
    context: &OperationDiffBuildContext<'_>,
) -> Result<Option<BlockGroupDiff>, OperationDiffError> {
    let Some(stage_graphs) = load_block_group_stage_graphs(graph_conn, block_group_id, context)?
    else {
        return Ok(None);
    };
    let graph = build_unified_diff_graph(&stage_graphs, context.default_operation);
    if !diff_graph_has_changes(&graph)
        && stage_graphs.presence == BlockGroupPresenceChange::Modified
    {
        return Ok(None);
    }

    Ok(Some(BlockGroupDiff {
        id: block_group_id,
        block_group: Some(stage_graphs.block_group),
        presence: stage_graphs.presence.diff_presence(),
        graph,
    }))
}

fn load_block_group_stage_graphs(
    graph_conn: &GraphConnection,
    block_group_id: HashId,
    context: &OperationDiffBuildContext<'_>,
) -> Result<Option<BlockGroupStageGraphs>, OperationDiffError> {
    let target_block_group =
        BlockGroup::get_by_id(graph_conn, &block_group_id, Some(context.to_ref)).ok();
    let source_block_group =
        BlockGroup::get_by_id(graph_conn, &block_group_id, Some(context.from_ref)).ok();
    let Some(presence) = BlockGroupPresenceChange::from_endpoint_presence(
        source_block_group.is_some(),
        target_block_group.is_some(),
    ) else {
        return Ok(None);
    };
    // Added or removed child block groups only exist on one endpoint. The
    // opposite endpoint is represented by the child's parent so the diff
    // compares the child graph against the graph it diverged from.
    let source_parent_block_group_id = target_block_group
        .as_ref()
        .and_then(|block_group| block_group.parent_block_group_id);
    let target_parent_block_group_id = source_block_group
        .as_ref()
        .and_then(|block_group| block_group.parent_block_group_id);
    let source_effective_block_group_id = source_block_group
        .as_ref()
        .map(|block_group| block_group.id)
        .or(source_parent_block_group_id);
    let target_effective_block_group_id = target_block_group
        .as_ref()
        .map(|block_group| block_group.id)
        .or(target_parent_block_group_id);
    // When the block group exists on the endpoint, read it from that endpoint's
    // ref. When it does not, the effective id is the parent id discovered from
    // the other endpoint's child row, so the parent must be read from that
    // endpoint's ref instead.
    let source_effective_history_ref = if source_block_group.is_some() {
        Some(context.from_ref)
    } else {
        Some(context.to_ref)
    };
    let target_effective_history_ref = if target_block_group.is_some() {
        Some(context.to_ref)
    } else {
        Some(context.from_ref)
    };
    let block_group = target_block_group.or(source_block_group);
    let Some(block_group) = block_group else {
        return Ok(None);
    };

    // Load both endpoint edge sets directly instead of replaying diff rows over
    // the source. Diff rows are still useful for attribution, but they are not
    // a complete graph description:
    //
    //   new sample with no sequence edits:
    //     source(parent):  start -> A -> end
    //     target(child):   start -> A -> end
    //
    // The copied child edges may have the same edge keys as the parent, so
    // replaying only changed keys can make the target look empty and hide the
    // created sample. Endpoint reads keep presence-only samples visible.
    //
    //   insertion/deletion edits:
    //     source: A -> B
    //     target: A -> I -> B, plus internal edit-site marker edges
    //
    // The target endpoint tells us which path edges are real graph choices and
    // lets the stage builder filter marker edges before rendering.
    let source_edges = load_block_group_edges(
        graph_conn,
        source_effective_block_group_id,
        source_effective_history_ref,
    );
    let raw_block_group_edge_changes = context
        .edge_changes_by_block_group
        .get(&block_group_id)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let block_group_edge_changes =
        effective_edge_changes_for_source(&source_edges, raw_block_group_edge_changes);
    let target_edges = load_block_group_edges(
        graph_conn,
        target_effective_block_group_id,
        target_effective_history_ref,
    );
    let source_spans = stage_spans_from_edges(&source_edges);
    let target_spans = stage_spans_from_edges(&target_edges);
    let stage_operation_change_lookup = collect_stage_operation_change_lookup(
        block_group_id,
        context,
        &block_group_edge_changes,
        &source_spans,
        &target_spans,
    );
    let (source_nodes, target_nodes) =
        split_stage_nodes_at_shared_boundaries(&source_spans, &target_spans);
    let source_graph = build_stage_graph_from_nodes(
        source_nodes,
        &source_spans,
        &source_edges,
        &stage_operation_change_lookup.source_node_operation_by_id,
        &stage_operation_change_lookup.source_edge_operation_by_id,
    );
    let reconstructed_target_graph = build_stage_graph_from_nodes(
        target_nodes,
        &target_spans,
        &target_edges,
        &stage_operation_change_lookup.target_node_operation_by_id,
        &stage_operation_change_lookup.target_edge_operation_by_id,
    );

    Ok(Some(BlockGroupStageGraphs {
        block_group,
        presence,
        source_graph,
        reconstructed_target_graph,
    }))
}

fn load_block_group_edges(
    graph_conn: &GraphConnection,
    block_group_id: Option<HashId>,
    history_ref: Option<&str>,
) -> Vec<AugmentedEdge> {
    block_group_id.map_or_else(Vec::new, |block_group_id| {
        BlockGroupEdge::edges_for_block_group(graph_conn, &block_group_id, history_ref)
    })
}

fn stage_spans_from_edges(edges: &[AugmentedEdge]) -> HashSet<GraphNode> {
    // A diff graph node only needs a backing node id and a sequence slice. Do
    // not call Edge::blocks_from_edges here: that query fetches sequence text
    // so GroupBlock::sequence can work, but the diff graph resolves sequence
    // text later through the regular GraphNode rendering path.
    //
    //   endpoint edges:
    //     start -> A@0
    //     A@10 -> end
    //
    //   edge-derived span:
    //     A[0..10]
    //
    // Keeping this as edge-derived spans lets edge attribution stay attached
    // until the graph builder has both the display nodes and real path edges.
    let mut starts_by_node_id = HashMap::<HashId, HashSet<i64>>::new();
    let mut ends_by_node_id = HashMap::<HashId, HashSet<i64>>::new();
    for edge in edges.iter().map(|edge| &edge.edge) {
        if !is_terminal(edge.source_node_id) {
            ends_by_node_id
                .entry(edge.source_node_id)
                .or_default()
                .insert(edge.source_coordinate);
        }
        if !is_terminal(edge.target_node_id) {
            starts_by_node_id
                .entry(edge.target_node_id)
                .or_default()
                .insert(edge.target_coordinate);
        }
    }

    let mut node_ids = starts_by_node_id.keys().copied().collect::<HashSet<_>>();
    node_ids.extend(ends_by_node_id.keys().copied());
    let mut spans = HashSet::new();
    for node_id in node_ids {
        let empty_starts = HashSet::new();
        let empty_ends = HashSet::new();
        let starts = starts_by_node_id.get(&node_id).unwrap_or(&empty_starts);
        let ends = ends_by_node_id.get(&node_id).unwrap_or(&empty_ends);
        for (sequence_start, sequence_end) in coordinate_spans(starts, ends) {
            spans.insert(GraphNode {
                node_id,
                sequence_start,
                sequence_end,
            });
        }
    }
    spans
}

fn coordinate_spans(starts: &HashSet<i64>, ends: &HashSet<i64>) -> Vec<(i64, i64)> {
    let mut coordinates = starts.union(ends).copied().collect::<Vec<_>>();
    coordinates.sort();
    coordinates.dedup();
    if coordinates.len() <= 1 {
        return coordinates
            .first()
            .copied()
            .map(|coordinate| vec![(coordinate, coordinate)])
            .unwrap_or_default();
    }
    coordinates
        .windows(2)
        .map(|window| (window[0], window[1]))
        .collect()
}

fn split_stage_nodes_at_shared_boundaries(
    source_spans: &HashSet<GraphNode>,
    target_spans: &HashSet<GraphNode>,
) -> (HashSet<GraphNode>, HashSet<GraphNode>) {
    let boundaries_by_node_id = collect_stage_boundaries(source_spans, target_spans);
    (
        split_spans_at_boundaries(source_spans, &boundaries_by_node_id),
        split_spans_at_boundaries(target_spans, &boundaries_by_node_id),
    )
}

fn collect_stage_boundaries(
    source_spans: &HashSet<GraphNode>,
    target_spans: &HashSet<GraphNode>,
) -> HashMap<HashId, HashSet<i64>> {
    let mut boundaries_by_node_id = HashMap::new();
    for span in source_spans.iter().chain(target_spans) {
        let boundaries = boundaries_by_node_id
            .entry(span.node_id)
            .or_insert_with(HashSet::new);
        boundaries.insert(span.sequence_start);
        boundaries.insert(span.sequence_end);
    }
    boundaries_by_node_id
}

fn split_spans_at_boundaries(
    spans: &HashSet<GraphNode>,
    boundaries_by_node_id: &HashMap<HashId, HashSet<i64>>,
) -> HashSet<GraphNode> {
    let mut nodes = HashSet::new();
    for span in spans {
        // Split both endpoint graphs on the union of observed edge boundaries
        // for each backing node. GraphNode coordinates are sequence slices, not
        // path coordinates, so this only divides already-present sequence spans.
        //
        //   source before VCF edit:
        //     [0................................34]
        //
        //   target after deletion/insertion:
        //     [0..3] [4..10] [10..............34] and inserted [AGA]
        //
        //   source display nodes:
        //     [0..3] [3..4] [4..10] [10.......34]
        //
        // Without this split, the deleted [3..4] base is hidden inside a large
        // source block and cannot be colored as removed.
        if span.sequence_start == span.sequence_end {
            nodes.insert(*span);
            continue;
        }
        let mut boundaries = boundaries_by_node_id
            .get(&span.node_id)
            .into_iter()
            .flat_map(|boundaries| boundaries.iter())
            .filter(|coordinate| {
                **coordinate >= span.sequence_start && **coordinate <= span.sequence_end
            })
            .copied()
            .collect::<Vec<_>>();
        boundaries.sort();
        boundaries.dedup();
        if boundaries.len() < 2 {
            nodes.insert(*span);
            continue;
        }
        for window in boundaries.windows(2) {
            let sequence_start = window[0];
            let sequence_end = window[1];
            if sequence_start == sequence_end {
                continue;
            }
            nodes.insert(GraphNode {
                node_id: span.node_id,
                sequence_start,
                sequence_end,
            });
        }
    }
    nodes
}

fn effective_edge_changes_for_source(
    source_edges: &[AugmentedEdge],
    edge_changes: &[BlockGroupEdgeChange],
) -> Vec<BlockGroupEdgeChange> {
    let source_edge_keys = source_edges
        .iter()
        .map(BlockGroupEdgeKey::from)
        .collect::<HashSet<_>>();
    edge_changes
        .iter()
        .copied()
        .filter(
            |edge_change| match (edge_change.old_key, edge_change.new_key) {
                (None, Some(new_key)) => !source_edge_keys.contains(&new_key),
                (Some(old_key), None) => source_edge_keys.contains(&old_key),
                (Some(old_key), Some(new_key)) => {
                    source_edge_keys.contains(&old_key) || !source_edge_keys.contains(&new_key)
                }
                (None, None) => false,
            },
        )
        .collect()
}

fn collect_stage_operation_change_lookup(
    block_group_id: HashId,
    context: &OperationDiffBuildContext<'_>,
    changed_edge_ids: &[BlockGroupEdgeChange],
    source_spans: &HashSet<GraphNode>,
    target_spans: &HashSet<GraphNode>,
) -> StageOperationChangeLookup {
    let mut change_lookup = StageOperationChangeLookup::default();
    let Some(block_group_operations) = context.block_group_operations_by_id.get(&block_group_id)
    else {
        return change_lookup;
    };
    let source_node_ids = span_node_ids(source_spans);
    let target_node_ids = span_node_ids(target_spans);
    for changed_node_id in context.changed_node_ids {
        if let (Some(old_node_id), Some(operation)) = (
            changed_node_id.old_node_id(),
            changed_node_id.old_operation(),
        ) && block_group_operations.contains(&operation)
            && source_node_ids.contains(&old_node_id)
            && !target_node_ids.contains(&old_node_id)
        {
            change_lookup
                .source_node_operation_by_id
                .insert(old_node_id, operation);
        }
        if let (Some(new_node_id), Some(operation)) = (
            changed_node_id.new_node_id(),
            changed_node_id.new_operation(),
        ) && block_group_operations.contains(&operation)
            && target_node_ids.contains(&new_node_id)
            && !source_node_ids.contains(&new_node_id)
        {
            change_lookup
                .target_node_operation_by_id
                .insert(new_node_id, operation);
        }
    }

    for changed_edge_id in changed_edge_ids {
        if let (Some(old_edge_id), Some(operation)) = (
            changed_edge_id.old_edge_id(),
            changed_edge_id.old_operation(),
        ) {
            change_lookup
                .source_edge_operation_by_id
                .insert(old_edge_id, operation);
        }
        if let (Some(new_edge_id), Some(operation)) = (
            changed_edge_id.new_edge_id(),
            changed_edge_id.new_operation(),
        ) {
            change_lookup
                .target_edge_operation_by_id
                .insert(new_edge_id, operation);
        }
    }
    change_lookup
}

fn span_node_ids(spans: &HashSet<GraphNode>) -> HashSet<HashId> {
    spans
        .iter()
        .filter(|span| !is_terminal(span.node_id))
        .map(|span| span.node_id)
        .collect()
}

fn build_unified_diff_graph(
    stage_graphs: &BlockGroupStageGraphs,
    default_operation: HashId,
) -> DiffGenGraph {
    let mut merged_nodes = stage_graphs
        .source_graph
        .node_changes
        .keys()
        .chain(stage_graphs.reconstructed_target_graph.node_changes.keys())
        .copied()
        .collect::<Vec<_>>();
    merged_nodes.sort();
    merged_nodes.dedup();

    let mut merged_edge_keys = stage_graphs
        .source_graph
        .edge_changes
        .keys()
        .chain(stage_graphs.reconstructed_target_graph.edge_changes.keys())
        .copied()
        .collect::<Vec<_>>();
    merged_edge_keys.sort();
    merged_edge_keys.dedup();
    let source_edge_pairs = annotated_edge_pairs(&stage_graphs.source_graph);
    let target_edge_pairs = annotated_edge_pairs(&stage_graphs.reconstructed_target_graph);

    let mut diff_graph = DiffGenGraph::new();
    for node in merged_nodes {
        diff_graph.add_node(diff_graph_node(node, stage_graphs, default_operation));
    }

    let mut merged_edges_by_node_pair = HashMap::<(GraphNode, GraphNode), Vec<GraphEdgeKey>>::new();
    for edge_key in merged_edge_keys {
        merged_edges_by_node_pair
            .entry((edge_key.source, edge_key.target))
            .or_default()
            .push(edge_key);
    }
    let mut merged_edge_pairs = merged_edges_by_node_pair.into_iter().collect::<Vec<_>>();
    merged_edge_pairs.sort_by_key(|((source, target), _)| (*source, *target));
    for ((source, target), edge_keys) in merged_edge_pairs {
        let source_node = diff_graph_node(source, stage_graphs, default_operation);
        let target_node = diff_graph_node(target, stage_graphs, default_operation);
        let unchanged_endpoints = source_node.change.kind == DiffChangeKind::Unchanged
            && target_node.change.kind == DiffChangeKind::Unchanged;
        let diff_edges = edge_keys
            .iter()
            .copied()
            .map(|edge_key| DiffGraphEdge {
                edge: graph_edge_for_key(stage_graphs, edge_key),
                change: merge_edge_stage_changes(
                    stage_graphs
                        .source_graph
                        .edge_changes
                        .get(&edge_key)
                        .copied(),
                    stage_graphs
                        .reconstructed_target_graph
                        .edge_changes
                        .get(&edge_key)
                        .copied(),
                    unchanged_endpoints,
                    source_edge_pairs.contains(&(source, target)),
                    target_edge_pairs.contains(&(source, target)),
                    default_operation,
                ),
            })
            .collect::<Vec<_>>();
        diff_graph.add_edge(source_node, target_node, diff_edges);
    }

    diff_graph
}

fn diff_graph_has_changes(graph: &DiffGenGraph) -> bool {
    graph
        .nodes()
        .any(|node| node.change.kind != DiffChangeKind::Unchanged)
        || graph.all_edges().any(|(_, _, edges)| {
            edges
                .iter()
                .any(|edge| edge.change.kind != DiffChangeKind::Unchanged)
        })
}

fn diff_graph_node(
    node: GraphNode,
    stage_graphs: &BlockGroupStageGraphs,
    default_operation: HashId,
) -> DiffGraphNode {
    let source_change = stage_graphs.source_graph.node_changes.get(&node).copied();
    let target_change = stage_graphs
        .reconstructed_target_graph
        .node_changes
        .get(&node)
        .copied();
    DiffGraphNode {
        node,
        change: merge_stage_changes(source_change, target_change, default_operation),
    }
}

fn annotated_edge_pairs(stage_graph: &AnnotatedStageGraph) -> HashSet<(GraphNode, GraphNode)> {
    stage_graph
        .edge_changes
        .keys()
        .map(GraphEdgeKey::node_pair)
        .collect()
}

fn graph_edge_for_key(stage_graphs: &BlockGroupStageGraphs, edge_key: GraphEdgeKey) -> GraphEdge {
    stage_graph_edge_for_key(&stage_graphs.reconstructed_target_graph, edge_key)
        .or_else(|| stage_graph_edge_for_key(&stage_graphs.source_graph, edge_key))
        .expect("should contain graph edge for merged diff key")
}

fn stage_graph_edge_for_key(
    stage_graph: &AnnotatedStageGraph,
    edge_key: GraphEdgeKey,
) -> Option<GraphEdge> {
    stage_graph
        .graph
        .edge_weight(edge_key.source, edge_key.target)?
        .iter()
        .copied()
        .find(|edge| edge_key.matches_edge(*edge))
}

fn merge_stage_changes(
    source_change: Option<DiffChange>,
    target_change: Option<DiffChange>,
    default_operation: HashId,
) -> DiffChange {
    match (source_change, target_change) {
        (None, None) => DiffChange::unchanged(),
        (None, Some(change)) => changed(DiffChangeKind::Added, change.operation, default_operation),
        (Some(change), None) => {
            changed(DiffChangeKind::Removed, change.operation, default_operation)
        }
        (Some(source_change), Some(target_change))
            if source_change.kind == DiffChangeKind::Removed =>
        {
            changed(
                DiffChangeKind::Removed,
                source_change.operation.or(target_change.operation),
                default_operation,
            )
        }
        (Some(source_change), Some(target_change))
            if target_change.kind == DiffChangeKind::Added =>
        {
            changed(
                DiffChangeKind::Added,
                target_change.operation.or(source_change.operation),
                default_operation,
            )
        }
        (Some(source_change), Some(target_change))
            if source_change.operation.is_some()
                || target_change.operation.is_some()
                || source_change.kind != DiffChangeKind::Unchanged
                || target_change.kind != DiffChangeKind::Unchanged =>
        {
            changed(
                DiffChangeKind::Modified,
                target_change.operation.or(source_change.operation),
                default_operation,
            )
        }
        (Some(_), Some(_)) => DiffChange::unchanged(),
    }
}

fn merge_edge_stage_changes(
    source_change: Option<DiffChange>,
    target_change: Option<DiffChange>,
    edge_has_unchanged_endpoints: bool,
    source_has_pair: bool,
    target_has_pair: bool,
    default_operation: HashId,
) -> DiffChange {
    match (source_change, target_change) {
        // Source-only continuation edges are context when both endpoint slices
        // survive in the target. They should stay grey for insertions:
        //
        //   source: A -------- B
        //   target: A -> I -> B
        //
        // Deletions still render red because their source-only edges touch a
        // removed node:
        //
        //   source: A -> G -> B
        //   target: A ------> B
        (Some(_), None) if edge_has_unchanged_endpoints => DiffChange::unchanged(),
        (None, Some(change)) if source_has_pair => {
            merge_pair_matched_edge_change(change, default_operation)
        }
        (Some(change), None) if target_has_pair => {
            merge_pair_matched_edge_change(change, default_operation)
        }
        (source_change, target_change) => {
            merge_stage_changes(source_change, target_change, default_operation)
        }
    }
}

fn merge_pair_matched_edge_change(change: DiffChange, default_operation: HashId) -> DiffChange {
    if change.kind == DiffChangeKind::Unchanged && change.operation.is_none() {
        DiffChange::unchanged()
    } else {
        changed(
            DiffChangeKind::Modified,
            change.operation,
            default_operation,
        )
    }
}

fn changed(
    kind: DiffChangeKind,
    operation: Option<HashId>,
    default_operation: HashId,
) -> DiffChange {
    DiffChange::new(kind, Some(operation.unwrap_or(default_operation)))
}

fn collect_block_group_changes(
    graph_conn: &GraphConnection,
    from_ref: &str,
    to_ref: &str,
    included_operations: Option<&HashSet<HashId>>,
) -> Result<BlockGroupChanges, OperationDiffError> {
    let query = "WITH block_group_changes AS (
             SELECT to_id AS to_block_group_id,
                    from_id AS from_block_group_id,
                    NULL AS to_edge_id,
                    NULL AS from_edge_id,
                    NULL AS to_chromosome_index,
                    NULL AS from_chromosome_index,
                    NULL AS to_phased,
                    NULL AS from_phased,
                    to_commit,
                    from_commit,
                    diff_type
             FROM dolt_diff_block_groups(?1, ?2)
             UNION ALL
             SELECT to_block_group_id,
                    from_block_group_id,
                    to_edge_id,
                    from_edge_id,
                    to_chromosome_index,
                    from_chromosome_index,
                    to_phased,
                    from_phased,
                    to_commit,
                    from_commit,
                    diff_type
             FROM dolt_diff_block_group_edges(?1, ?2)
         )
         SELECT to_block_group_id,
                from_block_group_id,
                to_edge_id,
                from_edge_id,
                to_chromosome_index,
                from_chromosome_index,
                to_phased,
                from_phased,
                to_commit,
                from_commit,
                diff_type
         FROM block_group_changes";
    let mut changes = BlockGroupChanges::default();
    let mut statement = graph_conn.prepare(query)?;
    let rows = statement.query_map(params![from_ref, to_ref], DoltBlockGroupChangeRow::from_row)?;
    for row in rows {
        let row = row?;
        if !range_change_included(
            row.diff_type.as_str(),
            row.change_operation(),
            row.source_operation(),
            included_operations,
        ) {
            continue;
        }
        row.record_presence_change(&mut changes);
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

fn collect_node_changes(
    graph_conn: &GraphConnection,
    from_ref: &str,
    to_ref: &str,
    included_operations: Option<&HashSet<HashId>>,
) -> Result<Vec<ChangedNodeIds>, OperationDiffError> {
    let query =
        "SELECT to_id, from_id, to_commit, from_commit, diff_type FROM dolt_diff_nodes(?1, ?2)";
    let mut changed_node_ids = Vec::new();
    let mut statement = graph_conn.prepare(query)?;
    let rows = statement.query_map(params![from_ref, to_ref], |row| {
        Ok((
            row.get::<_, Option<HashId>>(0)?,
            row.get::<_, Option<HashId>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;
    for row in rows {
        let (to_node_id, from_node_id, to_commit, from_commit, diff_type) = row?;
        let operation = operation_hash_from_commit(to_commit.as_deref());
        let source_operation = operation_hash_from_commit(from_commit.as_deref());
        if !range_change_included(
            diff_type.as_str(),
            operation,
            source_operation,
            included_operations,
        ) {
            continue;
        }
        match diff_type.as_str() {
            "added" => {
                if let Some(new_node_id) = to_node_id {
                    changed_node_ids.push(ChangedNodeIds::Added {
                        new_node_id,
                        operation,
                    });
                }
            }
            "removed" => {
                if let Some(old_node_id) = from_node_id {
                    changed_node_ids.push(ChangedNodeIds::Removed {
                        old_node_id,
                        operation,
                    });
                }
            }
            "modified" => {
                if let (Some(old_node_id), Some(new_node_id)) = (from_node_id, to_node_id) {
                    changed_node_ids.push(ChangedNodeIds::Modified {
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

    Ok(changed_node_ids)
}

fn range_change_included(
    diff_type: &str,
    operation: Option<HashId>,
    source_operation: Option<HashId>,
    included_operations: Option<&HashSet<HashId>>,
) -> bool {
    match included_operations {
        Some(included_operations) => operation
            .filter(|operation| included_operations.contains(operation))
            .is_some_and(|_| {
                !matches!(diff_type, "removed" | "modified")
                    || source_operation
                        .is_some_and(|operation| included_operations.contains(&operation))
            }),
        None => true,
    }
}

fn build_stage_graph_from_nodes(
    mut stage_nodes: HashSet<GraphNode>,
    continuation_spans: &HashSet<GraphNode>,
    stage_edges: &[AugmentedEdge],
    node_operations_by_id: &HashMap<HashId, HashId>,
    edge_operations_by_id: &HashMap<HashId, HashId>,
) -> AnnotatedStageGraph {
    let mut graph = GenGraph::new();
    let mut node_changes = HashMap::new();
    let mut edge_changes = HashMap::new();

    stage_nodes.insert(path_start_graph_node());
    stage_nodes.insert(path_end_graph_node());
    for node in &stage_nodes {
        graph.add_node(*node);
        node_changes.insert(
            *node,
            node_operations_by_id
                .get(&node.node_id)
                .map_or_else(DiffChange::unchanged, |operation| {
                    DiffChange::new(DiffChangeKind::Modified, Some(*operation))
                }),
        );
    }

    let blocks_by_start = stage_nodes
        .iter()
        .map(|node| ((node.node_id, node.sequence_start), *node))
        .collect::<HashMap<_, _>>();
    let blocks_by_end = stage_nodes
        .iter()
        .map(|node| ((node.node_id, node.sequence_end), *node))
        .collect::<HashMap<_, _>>();

    for augmented_edge in stage_edges {
        // Edit-site marker edges preserve coordinates for later graph
        // operations. They are not path choices, so rendering them as diff
        // edges makes inserted/deleted sequence look unchanged.
        if augmented_edge.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
            continue;
        }
        let source_node = blocks_by_end
            .get(&(
                augmented_edge.edge.source_node_id,
                augmented_edge.edge.source_coordinate,
            ))
            .copied();
        let target_node = blocks_by_start
            .get(&(
                augmented_edge.edge.target_node_id,
                augmented_edge.edge.target_coordinate,
            ))
            .copied();
        if let (Some(source_node), Some(target_node)) = (source_node, target_node) {
            let graph_edge = GraphEdge {
                edge_id: augmented_edge.edge.id,
                source_strand: augmented_edge.edge.source_strand,
                target_strand: augmented_edge.edge.target_strand,
                chromosome_index: augmented_edge.chromosome_index,
                phased: augmented_edge.phased,
                created_on: augmented_edge.created_on,
            };
            if let Some(existing_edges) = graph.edge_weight_mut(source_node, target_node) {
                existing_edges.push(graph_edge);
            } else {
                graph.add_edge(source_node, target_node, vec![graph_edge]);
            }
            edge_changes.insert(
                GraphEdgeKey::new(source_node, target_node, graph_edge),
                edge_operations_by_id
                    .get(&graph_edge.edge_id)
                    .map_or_else(DiffChange::unchanged, |operation| {
                        DiffChange::new(DiffChangeKind::Modified, Some(*operation))
                    }),
            );
        }
    }

    add_continuation_edges(
        &stage_nodes,
        continuation_spans,
        &mut graph,
        &mut edge_changes,
    );
    remove_unconnected_stage_nodes(&mut graph, &mut node_changes);

    AnnotatedStageGraph {
        graph,
        node_changes,
        edge_changes,
    }
}

fn path_start_graph_node() -> GraphNode {
    GraphNode {
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    }
}

fn path_end_graph_node() -> GraphNode {
    GraphNode {
        node_id: PATH_END_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    }
}

fn add_continuation_edges(
    stage_nodes: &HashSet<GraphNode>,
    continuation_spans: &HashSet<GraphNode>,
    graph: &mut GenGraph,
    edge_changes: &mut HashMap<GraphEdgeKey, DiffChange>,
) {
    for span in continuation_spans {
        // Delayed block construction starts from the endpoint edges and keeps
        // each endpoint's real covered spans separate before splitting them
        // into display nodes. Add neutral continuation edges only inside a
        // span that exists in this endpoint:
        //
        //   endpoint span [0..10] split to [0..3] [3..4] [4..10]
        //     add grey continuations 0..3 -> 3..4 -> 4..10
        //
        //   endpoint spans [0..3] and [4..10]
        //     do not synthesize 0..3 -> 3..4 -> 4..10
        //
        // This is the point where the graph knows enough to split display
        // nodes without asking a second question about whether a boundary came
        // from the source or target path.
        if is_terminal(span.node_id) || span.sequence_start == span.sequence_end {
            continue;
        }
        let mut nodes = stage_nodes
            .iter()
            .copied()
            .filter(|node| {
                node.node_id == span.node_id
                    && node.sequence_start >= span.sequence_start
                    && node.sequence_end <= span.sequence_end
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| (node.sequence_start, node.sequence_end));
        for window in nodes.windows(2) {
            let source_node = window[0];
            let target_node = window[1];
            if source_node.sequence_end != target_node.sequence_start {
                continue;
            }
            let synthetic_edge = GraphEdge {
                edge_id: HashId::convert_str(&format!(
                    "diff-continuation:{}:{}:{}:{}:{}",
                    span.node_id,
                    source_node.sequence_start,
                    source_node.sequence_end,
                    target_node.sequence_start,
                    target_node.sequence_end
                )),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                created_on: 0,
            };
            if let Some(existing_edges) = graph.edge_weight_mut(source_node, target_node) {
                if existing_edges.contains(&synthetic_edge) {
                    continue;
                }
                existing_edges.push(synthetic_edge);
            } else {
                graph.add_edge(source_node, target_node, vec![synthetic_edge]);
            }
            edge_changes.insert(
                GraphEdgeKey::new(source_node, target_node, synthetic_edge),
                DiffChange::unchanged(),
            );
        }
    }
}

fn remove_unconnected_stage_nodes(
    graph: &mut GenGraph,
    node_changes: &mut HashMap<GraphNode, DiffChange>,
) {
    let mut connected_nodes = HashSet::new();
    for (source, target, _) in graph.all_edges() {
        connected_nodes.insert(source);
        connected_nodes.insert(target);
    }
    let disconnected_nodes = graph
        .nodes()
        .filter(|node| !is_terminal(node.node_id) && !connected_nodes.contains(node))
        .collect::<Vec<_>>();
    for node in disconnected_nodes {
        graph.remove_node(node);
        node_changes.remove(&node);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
struct GraphEdgeKey {
    source: GraphNode,
    target: GraphNode,
    edge_id: HashId,
    source_strand: Strand,
    target_strand: Strand,
    chromosome_index: i64,
    phased: i64,
}

impl GraphEdgeKey {
    fn new(source: GraphNode, target: GraphNode, edge: GraphEdge) -> Self {
        Self {
            source,
            target,
            edge_id: edge.edge_id,
            source_strand: edge.source_strand,
            target_strand: edge.target_strand,
            chromosome_index: edge.chromosome_index,
            phased: edge.phased,
        }
    }

    fn node_pair(&self) -> (GraphNode, GraphNode) {
        (self.source, self.target)
    }

    fn matches_edge(self, edge: GraphEdge) -> bool {
        self.edge_id == edge.edge_id
            && self.source_strand == edge.source_strand
            && self.target_strand == edge.target_strand
            && self.chromosome_index == edge.chromosome_index
            && self.phased == edge.phased
    }
}

struct OperationDiffBuildContext<'a> {
    from_ref: &'a str,
    to_ref: &'a str,
    default_operation: HashId,
    edge_changes_by_block_group: &'a HashMap<HashId, Vec<BlockGroupEdgeChange>>,
    block_group_operations_by_id: &'a HashMap<HashId, HashSet<HashId>>,
    changed_node_ids: &'a [ChangedNodeIds],
}

struct BlockGroupStageGraphs {
    block_group: BlockGroup,
    presence: BlockGroupPresenceChange,
    source_graph: AnnotatedStageGraph,
    reconstructed_target_graph: AnnotatedStageGraph,
}

struct AnnotatedStageGraph {
    graph: GenGraph,
    node_changes: HashMap<GraphNode, DiffChange>,
    edge_changes: HashMap<GraphEdgeKey, DiffChange>,
}

#[derive(Debug, Default)]
struct StageOperationChangeLookup {
    source_node_operation_by_id: HashMap<HashId, HashId>,
    target_node_operation_by_id: HashMap<HashId, HashId>,
    source_edge_operation_by_id: HashMap<HashId, HashId>,
    target_edge_operation_by_id: HashMap<HashId, HashId>,
}

#[derive(Debug, Default)]
struct BlockGroupChanges {
    presence_changes: HashMap<HashId, BlockGroupPresenceChange>,
    edge_changes: HashMap<HashId, Vec<BlockGroupEdgeChange>>,
    operations_by_block_group_id: HashMap<HashId, HashSet<HashId>>,
}

impl BlockGroupChanges {
    fn record_operation(&mut self, block_group_id: HashId, operation: Option<HashId>) {
        if let Some(operation) = operation {
            self.operations_by_block_group_id
                .entry(block_group_id)
                .or_default()
                .insert(operation);
        }
    }
}

#[derive(Debug)]
struct DoltBlockGroupChangeRow {
    to_block_group_id: Option<HashId>,
    from_block_group_id: Option<HashId>,
    to_edge_id: Option<HashId>,
    from_edge_id: Option<HashId>,
    to_chromosome_index: Option<i64>,
    from_chromosome_index: Option<i64>,
    to_phased: Option<i64>,
    from_phased: Option<i64>,
    to_commit: Option<String>,
    from_commit: Option<String>,
    diff_type: String,
}

impl DoltBlockGroupChangeRow {
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
            from_commit: row.get(9)?,
            diff_type: row.get(10)?,
        })
    }

    fn record_presence_change(&self, changes: &mut BlockGroupChanges) {
        match self.diff_type.as_str() {
            "added" => {
                if let Some(block_group_id) = self.to_block_group_id {
                    changes
                        .presence_changes
                        .entry(block_group_id)
                        .and_modify(BlockGroupPresenceChange::mark_added)
                        .or_insert(BlockGroupPresenceChange::Added);
                    changes.record_operation(block_group_id, self.change_operation());
                }
            }
            "removed" => {
                if let Some(block_group_id) = self.from_block_group_id {
                    changes
                        .presence_changes
                        .entry(block_group_id)
                        .and_modify(BlockGroupPresenceChange::mark_removed)
                        .or_insert(BlockGroupPresenceChange::Removed);
                    changes.record_operation(block_group_id, self.change_operation());
                }
            }
            "modified" => {
                if let Some(block_group_id) = self.to_block_group_id.or(self.from_block_group_id) {
                    changes
                        .presence_changes
                        .insert(block_group_id, BlockGroupPresenceChange::Modified);
                    changes.record_operation(block_group_id, self.change_operation());
                }
            }
            _ => {}
        }
    }

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

    fn change_operation(&self) -> Option<HashId> {
        operation_hash_from_commit(self.to_commit.as_deref())
    }

    fn source_operation(&self) -> Option<HashId> {
        operation_hash_from_commit(self.from_commit.as_deref())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BlockGroupEdgeKey {
    edge_id: HashId,
    chromosome_index: i64,
    phased: i64,
}

impl From<&AugmentedEdge> for BlockGroupEdgeKey {
    fn from(edge: &AugmentedEdge) -> Self {
        Self {
            edge_id: edge.edge.id,
            chromosome_index: edge.chromosome_index,
            phased: edge.phased,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockGroupPresenceChange {
    Added,
    Removed,
    Modified,
}

impl BlockGroupPresenceChange {
    fn from_endpoint_presence(present_in_source: bool, present_in_target: bool) -> Option<Self> {
        match (present_in_source, present_in_target) {
            (false, false) => None,
            (false, true) => Some(BlockGroupPresenceChange::Added),
            (true, false) => Some(BlockGroupPresenceChange::Removed),
            (true, true) => Some(BlockGroupPresenceChange::Modified),
        }
    }

    fn mark_added(&mut self) {
        *self = match self {
            BlockGroupPresenceChange::Added => BlockGroupPresenceChange::Added,
            BlockGroupPresenceChange::Removed => BlockGroupPresenceChange::Modified,
            BlockGroupPresenceChange::Modified => BlockGroupPresenceChange::Modified,
        };
    }

    fn mark_removed(&mut self) {
        *self = match self {
            BlockGroupPresenceChange::Added => BlockGroupPresenceChange::Modified,
            BlockGroupPresenceChange::Removed => BlockGroupPresenceChange::Removed,
            BlockGroupPresenceChange::Modified => BlockGroupPresenceChange::Modified,
        };
    }

    fn diff_presence(self) -> DiffPresence {
        match self {
            BlockGroupPresenceChange::Added => DiffPresence::TargetOnly,
            BlockGroupPresenceChange::Removed => DiffPresence::SourceOnly,
            BlockGroupPresenceChange::Modified => DiffPresence::Both,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChangedNodeIds {
    Added {
        new_node_id: HashId,
        operation: Option<HashId>,
    },
    Removed {
        old_node_id: HashId,
        operation: Option<HashId>,
    },
    Modified {
        old_node_id: HashId,
        new_node_id: HashId,
        old_operation: Option<HashId>,
        new_operation: Option<HashId>,
    },
}

impl ChangedNodeIds {
    fn old_node_id(&self) -> Option<HashId> {
        match self {
            ChangedNodeIds::Added { .. } => None,
            ChangedNodeIds::Removed { old_node_id, .. } => Some(*old_node_id),
            ChangedNodeIds::Modified { old_node_id, .. } => Some(*old_node_id),
        }
    }

    fn new_node_id(&self) -> Option<HashId> {
        match self {
            ChangedNodeIds::Added { new_node_id, .. } => Some(*new_node_id),
            ChangedNodeIds::Removed { .. } => None,
            ChangedNodeIds::Modified { new_node_id, .. } => Some(*new_node_id),
        }
    }

    fn old_operation(&self) -> Option<HashId> {
        match self {
            ChangedNodeIds::Added { .. } => None,
            ChangedNodeIds::Removed { operation, .. } => *operation,
            ChangedNodeIds::Modified { old_operation, .. } => *old_operation,
        }
    }

    fn new_operation(&self) -> Option<HashId> {
        match self {
            ChangedNodeIds::Added { operation, .. } => *operation,
            ChangedNodeIds::Removed { .. } => None,
            ChangedNodeIds::Modified { new_operation, .. } => *new_operation,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockGroupEdgeChange {
    block_group_id: HashId,
    old_key: Option<BlockGroupEdgeKey>,
    new_key: Option<BlockGroupEdgeKey>,
    old_operation: Option<HashId>,
    new_operation: Option<HashId>,
}

impl BlockGroupEdgeChange {
    fn block_group_id(&self) -> HashId {
        self.block_group_id
    }

    fn old_edge_id(&self) -> Option<HashId> {
        self.old_key.map(|key| key.edge_id)
    }

    fn new_edge_id(&self) -> Option<HashId> {
        self.new_key.map(|key| key.edge_id)
    }

    fn old_operation(&self) -> Option<HashId> {
        self.old_operation
    }

    fn new_operation(&self) -> Option<HashId> {
        self.new_operation
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use gen_core::{
        BranchName, CommitRef, HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, Workspace,
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
        BlockGroupDiff, BlockGroupEdgeChange, BlockGroupEdgeKey, BlockGroupPresenceChange,
        BlockGroupStageGraphs, OperationDiffError, build_stage_graph_from_nodes,
        build_unified_diff_graph, collect_block_group_changes, collect_operation_diff,
        path_end_graph_node, path_start_graph_node,
    };
    use crate::{
        graph::{DiffChangeKind, DiffGenGraph},
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

    fn commit_operation(context: &gen_models::db::DbContext, message: &str) -> HashId {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        let commit_hash = history_store
            .commit_all(message)
            .expect("should commit graph changes");
        HashId::pad_str(&commit_hash.0)
    }

    fn commit_hash_ref(operation_hash: HashId) -> CommitRef {
        let hash_string = operation_hash.to_string();
        CommitRef(hash_string[hash_string.len() - 40..].to_string())
    }

    fn switch_branch(context: &gen_models::db::DbContext, branch_name: &str) {
        let history_store = DoltHistoryStore::new(context.graph().conn());
        history_store
            .checkout_branch(&BranchName(branch_name.to_string()))
            .expect("should checkout history branch");
    }

    fn create_feature_branch(
        context: &gen_models::db::DbContext,
        base_operation_hash: HashId,
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
        to_ref: &str,
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
                let edge =
                    load_augmented_edge(graph_conn, edge_change.block_group_id, new_key, to_ref)?;
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
        node_operations_by_id: &HashMap<HashId, HashId>,
        edge_operations_by_id: &HashMap<HashId, HashId>,
    ) -> super::AnnotatedStageGraph {
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

    fn graph_edge_keys(graph: &GenGraph) -> HashSet<super::GraphEdgeKey> {
        graph
            .all_edges()
            .flat_map(|(source, target, edges)| {
                edges
                    .iter()
                    .copied()
                    .map(move |edge| super::GraphEdgeKey::new(source, target, edge))
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
    fn one_operation_diff_uses_dolt_history() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base_op = commit_operation(&context, "seed");
        let expected_block_group =
            create_linear_block_group(&context, "c", "s", "bg", "one", "CCCCC");
        let head = commit_operation(&context, "add");

        let diff = collect_operation_diff(graph_conn, Some(base_op), head).expect("diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, expected_block_group.id);
        assert_block_group_graph_shape(&diff.diff_graph[0]);
    }

    #[test]
    fn initial_operation_diff_contains_added_block_groups_from_history() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        let expected_block_group =
            create_linear_block_group(&context, "c", "s", "bg", "one", "AAAAA");
        let head = commit_operation(&context, "add");

        let diff = collect_operation_diff(graph_conn, None, head).expect("diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, expected_block_group.id);
        assert_block_group_graph_shape(&diff.diff_graph[0]);
    }

    #[test]
    fn test_initial_diff_accepts_missing_parent_operation_row_with_history() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        let missing_parent_hash = HashId::pad_str(1);
        create_linear_block_group(&context, "c", "s", "bg", "one", "AAAAA");
        let head = commit_operation(&context, "add");

        let diff = collect_operation_diff(graph_conn, Some(missing_parent_hash), head)
            .expect("should treat a missing parent row as an initial diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
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

        let diff = collect_operation_diff(graph_conn, Some(base), head).expect("diff");
        assert_eq!(diff.operations, vec![head]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, removed_block_group.id);
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

        let diff = collect_operation_diff(graph_conn, Some(base), head).expect("diff");
        assert_eq!(diff.diff_graph.len(), 1);
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
        let changes = collect_block_group_changes(graph_conn, &source_ref, &target_ref, None)
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
            &block_group.id,
            &reconstructed_edges,
            Some(&target_ref),
        )
        .expect("should build reconstructed blocks");
        let target_edges =
            BlockGroupEdge::edges_for_block_group(graph_conn, &block_group.id, Some(&target_ref));
        let target_blocks = Edge::blocks_from_edges(
            graph_conn,
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
        let operation = HashId::pad_str(100);
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
        let operation = HashId::pad_str(100);
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
        let stage_graphs = BlockGroupStageGraphs {
            block_group: BlockGroup {
                id: HashId::pad_str(99),
                collection_name: "collection".to_string(),
                sample_name: "sample".to_string(),
                name: "block-group".to_string(),
                created_on: 0,
                parent_block_group_id: None,
                is_default: false,
            },
            presence: BlockGroupPresenceChange::Modified,
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
        let first_block_group =
            create_linear_block_group(&context, "c", "s", "bg1", "one", "CCCCC");
        let first_operation = commit_operation(&context, "add one");
        let second_block_group =
            create_linear_block_group(&context, "c", "s", "bg2", "two", "GGGGG");
        let second_operation = commit_operation(&context, "add two");

        let diff = collect_operation_diff(graph_conn, Some(base), second_operation).expect("diff");
        let first_diff = diff
            .diff_graph
            .iter()
            .find(|block_group| block_group.id == first_block_group.id)
            .expect("should keep first block group in unified diff");
        let second_diff = diff
            .diff_graph
            .iter()
            .find(|block_group| block_group.id == second_block_group.id)
            .expect("should keep second block group in unified diff");

        assert!(
            first_diff
                .graph
                .nodes()
                .any(|node| node.change.kind == DiffChangeKind::Added
                    && node.change.operation == Some(first_operation))
        );
        assert!(
            second_diff
                .graph
                .nodes()
                .any(|node| node.change.kind == DiffChangeKind::Added
                    && node.change.operation == Some(second_operation))
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

        let diff = collect_operation_diff(graph_conn, Some(op1), op3).expect("diff");
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

        let diff = collect_operation_diff(graph_conn, Some(base), op_one).expect("diff");
        assert_eq!(diff.operations, vec![op_one]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, bg_one.id);
        assert_ne!(diff.diff_graph[0].id, bg_two.id);

        let later_diff = collect_operation_diff(graph_conn, Some(op_one), op_two).expect("diff");
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
        let diff = collect_operation_diff(graph_conn, Some(base), base).expect("diff");
        assert!(diff.operations.is_empty());
    }

    #[test]
    fn diffs_across_branches_report_target_branch_history() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let main_block_group =
            create_linear_block_group(&context, "c", "s", "main", "main", "CCCCC");
        let op_main = commit_operation(&context, "main op");
        create_feature_branch(&context, base, "feature");
        let feature_block_group =
            create_linear_block_group(&context, "c", "s", "feature", "feature", "GGGGG");
        let op_feature = commit_operation(&context, "feature op");

        let diff = collect_operation_diff(graph_conn, Some(op_main), op_feature).expect("diff");
        assert_eq!(diff.operations, vec![op_main, op_feature]);
        assert_eq!(diff.diff_graph.len(), 1);
        assert_eq!(diff.diff_graph[0].id, feature_block_group.id);
        assert_ne!(
            diff.diff_graph[0].id, main_block_group.id,
            "divergent branch history still resolves to target-branch additions in the unified model"
        );
    }

    #[test]
    fn uses_default_graph_database_for_history_diffs() {
        let context = setup_history_context();
        let graph_conn = context.graph().conn();

        create_linear_block_group(&context, "c", "s", "seed", "seed", "AAAAA");
        let base = commit_operation(&context, "seed");
        let expected_block_group =
            create_linear_block_group(&context, "c", "s", "db-one", "one", "CCCCC");
        let head = commit_operation(&context, "add");

        let diff = collect_operation_diff(graph_conn, Some(base), head).expect("diff");
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
        let diff =
            collect_operation_diff(graph_conn, Some(base), replacement_commit).expect("diff");
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
        let merge_hash = HashId::pad_str(&merge_commit.0);

        let diff = collect_operation_diff(graph_conn, Some(base), merge_hash).expect("diff");
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
