use std::collections::{HashMap, HashSet};

use gen_core::{
    GenGraph, GraphNode, GraphNodePosition, HashId, NodeIntervalBlock,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, PathBlock,
};
use gen_models::{
    accession::AccessionSpan,
    annotations::persist_annotation,
    block_group::{
        BlockGroup, BlockGroupChange, BlockGroupError, IntervalTreeCache, IntervalTreeSource,
        SubgraphBoundary,
    },
    block_group_edge::{AugmentedEdgeData, BlockGroupEdge},
    db::GraphConnection,
    edge::{Edge, EdgeData},
    node::Node,
    operations::OperationSummary,
    region::{ResolvedGenRegion, ResolvedRegionKind},
    sample::{Sample, SampleError},
};
use intervaltree::IntervalTree;

use crate::{GraphError, all_intermediate_edges, flatten_to_interval_tree, graph_loader};

pub fn add_annotation(
    context: &gen_models::db::DbContext,
    collection: &str,
    name: &str,
    group: Option<&str>,
    sample: &str,
    region: &str,
) -> Result<OperationSummary, Box<dyn std::error::Error>> {
    let conn = context.graph().conn();
    let parsed_region = gen_core::region::Region::parse(region)?;
    let resolved_region = gen_models::region::resolve(&parsed_region, conn, collection, sample)?;
    let tree = region_intervaltree(&resolved_region, conn)?;
    let range = resolved_region.start..resolved_region.end;
    let spans = AccessionSpan::from_intervaltree_ranges(&tree, core::slice::from_ref(&range))?;
    persist_annotation(context, name, group, sample, &resolved_region, spans)
}

pub fn prune_graph(graph: &mut GenGraph) {
    graph_loader::prune_graph(graph);
}

#[expect(
    clippy::too_many_arguments,
    reason = "subgraph persistence requires both graph boundaries"
)]
pub fn derive_subgraph(
    conn: &GraphConnection,
    source_block_group_id: &HashId,
    start_block: &NodeIntervalBlock,
    end_block: &NodeIntervalBlock,
    start_node_coordinate: i64,
    end_node_coordinate: i64,
    target_block_group_id: &HashId,
    create_terminal_edges: bool,
) -> Result<(), BlockGroupError> {
    let graph = load_block_group_graph(conn, source_block_group_id, None)?;
    let start_node = graph
        .nodes()
        .find(|node| {
            node.node_id == start_block.node_id
                && node.sequence_start <= start_node_coordinate
                && node.sequence_end >= start_node_coordinate
        })
        .expect("should find the start boundary in the source graph");
    let end_node = graph
        .nodes()
        .find(|node| {
            node.node_id == end_block.node_id
                && node.sequence_start <= end_node_coordinate
                && node.sequence_end >= end_node_coordinate
        })
        .expect("should find the end boundary in the source graph");
    let edge_ids = all_intermediate_edges(&graph, start_node, end_node)
        .iter()
        .map(|(_source, _target, edge_info)| edge_info[0].edge_id)
        .collect::<Vec<_>>();
    BlockGroup::persist_subgraph(
        conn,
        source_block_group_id,
        &edge_ids,
        &SubgraphBoundary {
            block: *start_block,
            node_coordinate: start_node_coordinate,
        },
        &SubgraphBoundary {
            block: *end_block,
            node_coordinate: end_node_coordinate,
        },
        target_block_group_id,
        create_terminal_edges,
    )
}

pub fn get_all_sequences(
    conn: &GraphConnection,
    block_group_id: &HashId,
) -> Result<HashSet<String>, BlockGroupError> {
    get_all_sequences_with_pruning(conn, block_group_id, true)
}

pub fn get_all_sequences_with_pruning(
    conn: &GraphConnection,
    block_group_id: &HashId,
    prune: bool,
) -> Result<HashSet<String>, BlockGroupError> {
    let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
        .into_iter()
        .filter(|edge| edge.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .collect::<Vec<_>>();
    let blocks = Edge::blocks_from_edges(conn, block_group_id, &edges, None)?;
    let (load_edges, load_blocks) = Edge::graph_load_data(&edges, &blocks);
    let (mut graph, _) = graph_loader::build_graph(&load_edges, &load_blocks);
    if prune {
        graph_loader::prune_graph(&mut graph);
    }
    let sequences_by_node = blocks
        .iter()
        .map(|block| {
            (
                GraphNode {
                    node_id: block.node_id,
                    sequence_start: block.start,
                    sequence_end: block.end,
                },
                block.sequence(),
            )
        })
        .collect::<HashMap<_, _>>();
    Ok(graph_loader::get_all_sequences(&graph, &sequences_by_node))
}

pub fn get_sample_all_sequences(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    history_ref: Option<&str>,
) -> Result<HashSet<String>, SampleError> {
    let mut sequences = HashSet::new();
    for block_group in Sample::get_block_groups(conn, collection_name, sample_name, history_ref) {
        sequences.extend(get_all_sequences(conn, &block_group.id)?);
    }
    Ok(sequences)
}

/// Loads persisted block-group records and constructs their graph representation.
pub fn load_block_group_graph(
    conn: &GraphConnection,
    block_group_id: &HashId,
    history_ref: Option<&str>,
) -> Result<GenGraph, BlockGroupError> {
    let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id, history_ref);
    let blocks = Edge::blocks_from_edges(conn, block_group_id, &edges, history_ref)?;
    let (load_edges, load_blocks) = Edge::graph_load_data(&edges, &blocks);
    let (graph, _) = graph_loader::build_graph(&load_edges, &load_blocks);
    Ok(graph)
}

/// Loads a block group and projects its graph nodes into linear intervals.
pub fn load_block_group_intervaltree(
    conn: &GraphConnection,
    block_group_id: &HashId,
    remove_ambiguous_positions: bool,
) -> Result<IntervalTree<i64, NodeIntervalBlock>, BlockGroupError> {
    let mut graph = load_block_group_graph(conn, block_group_id, None)?;
    prune_graph(&mut graph);
    Ok(flatten_to_interval_tree(&graph, remove_ambiguous_positions))
}

pub fn load_sample_graph(
    conn: &GraphConnection,
    collection: &str,
    name: &str,
    history_ref: Option<&str>,
) -> Result<GenGraph, SampleError> {
    let block_groups = Sample::get_block_groups(conn, collection, name, history_ref);
    let mut sample_graph = GenGraph::new();
    for block_group in block_groups {
        let block_group_graph = load_block_group_graph(conn, &block_group.id, history_ref)?;
        for node in block_group_graph.nodes() {
            sample_graph.add_node(node);
        }
        for (source, target, edges) in block_group_graph.all_edges() {
            if let Some(existing_edges) = sample_graph.edge_weight_mut(source, target) {
                existing_edges.extend(edges.clone());
            } else {
                sample_graph.add_edge(source, target, edges.clone());
            }
        }
    }
    Ok(sample_graph)
}

fn region_intervaltree(
    region: &ResolvedGenRegion,
    conn: &GraphConnection,
) -> Result<IntervalTree<i64, NodeIntervalBlock>, BlockGroupError> {
    if region.kind == ResolvedRegionKind::BlockGroup {
        load_block_group_intervaltree(
            conn,
            &region.block_group.id,
            region.remove_ambiguous_positions,
        )
    } else {
        IntervalTreeSource::intervaltree(region, conn)
    }
}

pub fn resolve_region_positions(
    conn: &GraphConnection,
    block_group_id: &HashId,
    interval_tree: &IntervalTree<i64, NodeIntervalBlock>,
    start: i64,
    end: i64,
    start_offset: i64,
    end_offset: i64,
) -> Result<(Vec<GraphNodePosition>, Vec<GraphNodePosition>), GraphError> {
    graph_loader::resolve_region_positions(
        interval_tree,
        graph_loader::RegionPositionQuery {
            start,
            end,
            start_offset,
            end_offset,
        },
        |node_id| {
            Node::query_nodes_length(conn, &[node_id])
                .ok()
                .and_then(|lengths| lengths.get(&node_id).copied())
        },
        |graph, node_id| expand(conn, graph, block_group_id, node_id),
    )
}

pub fn positions_at_coordinate(
    interval_tree: &IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
) -> Vec<GraphNodePosition> {
    graph_loader::positions_at_coordinate(interval_tree, coordinate)
}

pub fn plan_region_edges(
    start_positions: &[GraphNodePosition],
    end_positions: &[GraphNodePosition],
    block: &PathBlock,
    preserve_edge: bool,
    chromosome_index: i64,
    phased: i64,
) -> Vec<AugmentedEdgeData> {
    graph_loader::plan_region_edges(
        start_positions,
        end_positions,
        block,
        preserve_edge,
        chromosome_index,
        phased,
    )
    .into_iter()
    .map(|edge| AugmentedEdgeData {
        edge_data: EdgeData {
            source_node_id: edge.source_node_id,
            source_coordinate: edge.source_coordinate,
            source_strand: edge.source_strand,
            target_node_id: edge.target_node_id,
            target_coordinate: edge.target_coordinate,
            target_strand: edge.target_strand,
        },
        chromosome_index: edge.chromosome_index,
        phased: edge.phased,
    })
    .collect()
}

pub fn find_region_graph_positions(
    region: &ResolvedGenRegion,
    conn: &GraphConnection,
    start_offset: i64,
    end_offset: i64,
) -> Result<ResolvedGenRegion, GraphError> {
    let interval_tree = region_intervaltree(region, conn).map_err(|_| GraphError::NoPath)?;
    let (start_positions, end_positions) = resolve_region_positions(
        conn,
        &region.block_group.id,
        &interval_tree,
        region.start,
        region.end,
        start_offset,
        end_offset,
    )?;
    let mut resolved = region.clone();
    resolved.start_anchors = Some(start_positions);
    resolved.end_anchors = Some(end_positions);
    Ok(resolved)
}

pub fn plan_region_change(
    region: &ResolvedGenRegion,
    conn: &GraphConnection,
    change: &BlockGroupChange,
    tree: Option<&IntervalTree<i64, NodeIntervalBlock>>,
) -> Result<Vec<AugmentedEdgeData>, BlockGroupError> {
    match region.kind {
        ResolvedRegionKind::Path | ResolvedRegionKind::BlockGroup => {
            let local_tree;
            let tree = match tree {
                Some(tree) => tree,
                None => {
                    local_tree = region_intervaltree(region, conn)?;
                    &local_tree
                }
            };
            return BlockGroup::set_up_new_edges(change, tree);
        }
        ResolvedRegionKind::Annotation | ResolvedRegionKind::Accession => {}
    }

    let graph_positions_from_tree = |coordinate| {
        let positions = positions_at_coordinate(tree?, coordinate);
        (!positions.is_empty()).then_some(positions)
    };
    let (start_positions, end_positions) =
        if let (Some(start), Some(end)) = (&region.start_anchors, &region.end_anchors) {
            (start.clone(), end.clone())
        } else if let Some(start_positions) = graph_positions_from_tree(region.start)
            && let Some(end_positions) = graph_positions_from_tree(region.end)
        {
            (start_positions, end_positions)
        } else {
            let resolved = find_region_graph_positions(region, conn, 0, 0)
                .map_err(|err| BlockGroupError::ChangeOutOfBounds(err.to_string()))?;
            (
                resolved.start_anchors.expect("should have start anchors"),
                resolved.end_anchors.expect("should have end anchors"),
            )
        };
    Ok(plan_region_edges(
        &start_positions,
        &end_positions,
        &change.block,
        change.preserve_edge,
        change.chromosome_index,
        change.phased,
    ))
}

pub fn insert_changes(
    conn: &GraphConnection,
    changes: &[BlockGroupChange],
    tree_map: Option<&mut IntervalTreeCache>,
) -> Result<(), BlockGroupError> {
    let mut new_augmented_edges_by_block_group = HashMap::new();
    let mut new_accession_edges = HashMap::new();
    let mut local_tree_map = HashMap::new();
    let tree_map = tree_map.unwrap_or(&mut local_tree_map);
    for change in changes {
        let cache_key = change.region.intervaltree_cache_key();
        if let std::collections::hash_map::Entry::Vacant(entry) = tree_map.entry(cache_key) {
            entry.insert(region_intervaltree(&change.region, conn)?);
        }
        let new_augmented_edges =
            plan_region_change(&change.region, conn, change, tree_map.get(&cache_key))?;
        new_augmented_edges_by_block_group
            .entry(change.region.block_group.id)
            .or_insert_with(Vec::new)
            .extend(new_augmented_edges.iter().cloned());
        if let Some(accession) = &change.path_accession {
            new_accession_edges
                .entry((change.region.block_group.id, accession.clone()))
                .or_insert_with(Vec::new)
                .extend(new_augmented_edges);
        }
    }
    BlockGroup::persist_insert_changes(
        conn,
        new_augmented_edges_by_block_group,
        new_accession_edges,
    )
}

pub fn insert_change(
    conn: &GraphConnection,
    change: &BlockGroupChange,
) -> Result<(), BlockGroupError> {
    let new_augmented_edges = plan_region_change(&change.region, conn, change, None)?;
    let mut new_augmented_edges_by_block_group = HashMap::new();
    new_augmented_edges_by_block_group
        .insert(change.region.block_group.id, new_augmented_edges.clone());
    let mut new_accession_edges = HashMap::new();
    if let Some(accession) = &change.path_accession {
        new_accession_edges.insert(
            (change.region.block_group.id, accession.clone()),
            new_augmented_edges,
        );
    }
    BlockGroup::persist_insert_changes(
        conn,
        new_augmented_edges_by_block_group,
        new_accession_edges,
    )
}

/// Merges edges adjacent to a node into an existing graph.
pub fn expand(
    conn: &GraphConnection,
    graph: &mut GenGraph,
    block_group_id: &HashId,
    node_id: HashId,
) -> bool {
    let candidate_edges =
        match Edge::edges_for_block_group_node_neighborhood(conn, block_group_id, node_id, None) {
            Ok(edges) => edges,
            Err(_) => return false,
        };
    let unloaded_edge_ids =
        graph_loader::unloaded_edge_ids(graph, candidate_edges.iter().map(|edge| edge.edge.id));
    let unloaded_edges = candidate_edges
        .into_iter()
        .filter(|edge| unloaded_edge_ids.contains(&edge.edge.id))
        .collect::<Vec<_>>();
    if unloaded_edges.is_empty() {
        return false;
    }
    let blocks = match Edge::blocks_from_edges(conn, block_group_id, &unloaded_edges, None) {
        Ok(blocks) => blocks,
        Err(_) => return false,
    };
    let (load_edges, load_blocks) = Edge::graph_load_data(&unloaded_edges, &blocks);
    let (fragment, _) = graph_loader::build_graph(&load_edges, &load_blocks);
    graph_loader::merge_fragment(graph, &fragment);
    true
}
