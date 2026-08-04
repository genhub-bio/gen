use std::collections::{HashMap, HashSet, VecDeque};

use gen_core::{
    GraphNodePosition, HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX,
    NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
    PathBlock, Strand, is_end_node, is_start_node, is_terminal,
};
use intervaltree::IntervalTree;
use itertools::Itertools;
use petgraph::Direction;

use crate::{
    GenGraph, GraphEdge, GraphError, GraphNode, MergeGraph, all_reachable_nodes, all_simple_paths,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphLoadBlock {
    pub id: i64,
    pub node_id: HashId,
    pub start: i64,
    pub end: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphLoadEdge {
    pub edge_id: HashId,
    pub source_node_id: HashId,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: HashId,
    pub target_coordinate: i64,
    pub target_strand: Strand,
    pub chromosome_index: i64,
    pub phased: i64,
    pub created_on: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegionPositionQuery {
    pub start: i64,
    pub end: i64,
    pub start_offset: i64,
    pub end_offset: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannedEdge {
    pub source_node_id: HashId,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: HashId,
    pub target_coordinate: i64,
    pub target_strand: Strand,
    pub chromosome_index: i64,
    pub phased: i64,
}

/// Resolves a linear region and its offsets to positions in an expandable graph.
///
/// Callers retain responsibility for persistence by supplying node lengths and graph expansion.
pub fn resolve_region_positions(
    interval_tree: &IntervalTree<i64, NodeIntervalBlock>,
    query: RegionPositionQuery,
    mut node_length: impl FnMut(HashId) -> Option<i64>,
    mut expand: impl FnMut(&mut GenGraph, HashId) -> bool,
) -> Result<(Vec<GraphNodePosition>, Vec<GraphNodePosition>), GraphError> {
    let filtered_tree = interval_tree
        .iter()
        .filter(|item| !is_terminal(item.value.node_id))
        .map(|item| (item.range.clone(), item.value))
        .collect::<IntervalTree<_, _>>();
    let mut graph = crate::graph_from_interval_tree(&filtered_tree);
    let start_anchor = resolve_anchor(
        &graph,
        &filtered_tree,
        query.start,
        &mut node_length,
        &mut expand,
    )?;
    let end_anchor = resolve_anchor(
        &graph,
        &filtered_tree,
        query.end,
        &mut node_length,
        &mut expand,
    )?;
    let start_positions = find_offset(&mut graph, &start_anchor, query.start_offset, &mut expand)?;
    let end_positions = find_offset(&mut graph, &end_anchor, query.end_offset, &mut expand)?;
    Ok((start_positions, end_positions))
}

/// Returns every non-terminal graph position covering a linear coordinate.
pub fn positions_at_coordinate(
    interval_tree: &IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
) -> Vec<GraphNodePosition> {
    let mut positions = interval_tree
        .query_point(coordinate)
        .map(|entry| entry.value)
        .filter(|block| !is_terminal(block.node_id))
        .map(|block| GraphNodePosition {
            graph_node: GraphNode {
                node_id: block.node_id,
                sequence_start: block.sequence_start,
                sequence_end: block.sequence_end,
            },
            offset: coordinate - block.start,
        })
        .collect::<Vec<_>>();
    positions.sort();
    positions.dedup();
    positions
}

/// Plans graph edges that preserve an edit site and connect a replacement block.
pub fn plan_region_edges(
    start_positions: &[GraphNodePosition],
    end_positions: &[GraphNodePosition],
    block: &PathBlock,
    preserve_edge: bool,
    chromosome_index: i64,
    phased: i64,
) -> Vec<PlannedEdge> {
    let preserve_chromosome_index = if preserve_edge {
        0
    } else {
        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
    };
    let mut edges = Vec::new();

    for position in start_positions.iter().chain(end_positions) {
        if is_terminal(position.graph_node.node_id) {
            continue;
        }
        let coordinate = position.coordinate();
        edges.push(PlannedEdge {
            source_node_id: position.graph_node.node_id,
            source_coordinate: coordinate,
            source_strand: Strand::Forward,
            target_node_id: position.graph_node.node_id,
            target_coordinate: coordinate,
            target_strand: Strand::Forward,
            chromosome_index: preserve_chromosome_index,
            phased: 0,
        });
    }

    if block.sequence_start == block.sequence_end {
        for start_position in start_positions {
            for end_position in end_positions {
                edges.push(PlannedEdge {
                    source_node_id: start_position.graph_node.node_id,
                    source_coordinate: start_position.coordinate(),
                    source_strand: Strand::Forward,
                    target_node_id: end_position.graph_node.node_id,
                    target_coordinate: end_position.coordinate(),
                    target_strand: Strand::Forward,
                    chromosome_index,
                    phased,
                });
            }
        }
    } else {
        for start_position in start_positions {
            edges.push(PlannedEdge {
                source_node_id: start_position.graph_node.node_id,
                source_coordinate: start_position.coordinate(),
                source_strand: Strand::Forward,
                target_node_id: block.node_id,
                target_coordinate: block.sequence_start,
                target_strand: Strand::Forward,
                chromosome_index,
                phased,
            });
        }
        for end_position in end_positions {
            edges.push(PlannedEdge {
                source_node_id: block.node_id,
                source_coordinate: block.sequence_end,
                source_strand: Strand::Forward,
                target_node_id: end_position.graph_node.node_id,
                target_coordinate: end_position.coordinate(),
                target_strand: Strand::Forward,
                chromosome_index,
                phased,
            });
        }
    }

    edges
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BlockKey {
    node_id: HashId,
    coordinate: i64,
}

/// Finds every graph position a signed sequence distance from an anchor.
///
/// The callback can add neighboring nodes when traversal reaches the boundary of the currently
/// loaded graph. It is first used at dead ends and then along existing paths if no result was
/// found.
pub fn find_offset(
    graph: &mut GenGraph,
    anchor: &GraphNodePosition,
    distance: i64,
    mut expand: impl FnMut(&mut GenGraph, HashId) -> bool,
) -> Result<Vec<GraphNodePosition>, GraphError> {
    if distance == 0 {
        return Ok(vec![*anchor]);
    }

    match find_offset_with_optional_expansion(graph, anchor, distance, false, &mut expand) {
        Ok(results) => Ok(results),
        Err(GraphError::OutOfBounds(_)) => {
            find_offset_with_optional_expansion(graph, anchor, distance, true, &mut expand)
        }
        Err(error) => Err(error),
    }
}

/// Resolves a linear coordinate to a graph position, expanding beyond the loaded interval when
/// necessary.
///
/// Persistence stays outside this function: callers provide node lengths and graph fragments
/// through callbacks.
pub fn resolve_anchor(
    graph: &GenGraph,
    interval_tree: &IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
    mut node_length: impl FnMut(HashId) -> Option<i64>,
    mut expand: impl FnMut(&mut GenGraph, HashId) -> bool,
) -> Result<GraphNodePosition, GraphError> {
    let block = interval_tree
        .query_point(coordinate)
        .map(|item| item.value)
        .find(|block| !is_terminal(block.node_id));
    if let Some(block) = block {
        return Ok(GraphNodePosition {
            graph_node: GraphNode {
                node_id: block.node_id,
                sequence_start: block.sequence_start,
                sequence_end: block.sequence_end,
            },
            offset: coordinate - block.start,
        });
    }

    let last_block = interval_tree
        .iter()
        .map(|item| item.value)
        .filter(|block| !is_terminal(block.node_id))
        .max_by_key(|block| block.end);
    let before_tree = coordinate < 0;
    let after_tree = last_block
        .map(|block| coordinate >= block.end)
        .unwrap_or(false);
    let boundary = if before_tree {
        interval_tree
            .iter()
            .map(|item| item.value)
            .filter(|block| !is_terminal(block.node_id))
            .min_by_key(|block| block.start)
    } else if after_tree {
        last_block
    } else {
        None
    }
    .ok_or(GraphError::NoPath)?;

    let boundary_node = GraphNode {
        node_id: boundary.node_id,
        sequence_start: boundary.sequence_start,
        sequence_end: boundary.sequence_end,
    };
    let boundary_anchor = GraphNodePosition {
        graph_node: boundary_node,
        offset: if before_tree {
            0
        } else {
            boundary_node.length()
        },
    };
    let boundary_distance = if before_tree {
        coordinate - boundary.start
    } else {
        coordinate - boundary.end
    };
    let same_node_coordinate = if before_tree {
        boundary.sequence_start + boundary_distance
    } else {
        boundary.sequence_end + boundary_distance
    };
    if same_node_coordinate >= 0
        && node_length(boundary.node_id).is_some_and(|length| same_node_coordinate <= length)
    {
        return Ok(GraphNodePosition {
            graph_node: boundary_node,
            offset: boundary_anchor.offset + boundary_distance,
        });
    }

    let mut expanded_graph = graph.clone();
    expand(&mut expanded_graph, boundary_node.node_id);
    let anchors = find_offset(
        &mut expanded_graph,
        &boundary_anchor,
        boundary_distance,
        expand,
    )?;

    anchors.into_iter().next().ok_or(GraphError::NoPath)
}

/// Returns candidate edge identifiers that are not already present in the loaded graph.
pub fn unloaded_edge_ids(
    graph: &GenGraph,
    candidate_edge_ids: impl IntoIterator<Item = HashId>,
) -> HashSet<HashId> {
    let loaded_edge_ids = graph
        .all_edges()
        .flat_map(|(_, _, edges)| edges.iter().map(|edge| edge.edge_id))
        .collect::<HashSet<_>>();
    candidate_edge_ids
        .into_iter()
        .filter(|edge_id| !loaded_edge_ids.contains(edge_id))
        .collect()
}

/// Merges a newly constructed local fragment into a loaded graph.
pub fn merge_fragment(graph: &mut GenGraph, fragment: &GenGraph) {
    graph.merge_graph(fragment);
}

fn find_offset_with_optional_expansion(
    graph: &mut GenGraph,
    anchor: &GraphNodePosition,
    distance: i64,
    expand_through_existing_paths: bool,
    expand: &mut impl FnMut(&mut GenGraph, HashId) -> bool,
) -> Result<Vec<GraphNodePosition>, GraphError> {
    let forward = distance > 0;
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut results = Vec::new();
    let mut result_seen = HashSet::new();

    queue.push_back((*anchor, distance));
    visited.insert((*anchor, distance));

    while let Some((position, remaining)) = queue.pop_front() {
        let node = position.graph_node;
        let node_length = node.length();

        if remaining == 0 {
            if result_seen.insert(position) {
                results.push(position);
            }
            continue;
        }

        if forward {
            let remaining_in_node = node_length - position.offset;
            if remaining <= remaining_in_node {
                let result = GraphNodePosition {
                    graph_node: node,
                    offset: position.offset + remaining,
                };
                if result_seen.insert(result) {
                    results.push(result);
                }
                continue;
            }

            let mut neighbors = graph
                .neighbors_directed(node, Direction::Outgoing)
                .collect::<Vec<_>>();
            if (neighbors.is_empty() || expand_through_existing_paths)
                && expand(graph, node.node_id)
            {
                neighbors = graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .collect();
            }

            for neighbor in neighbors {
                let next_position = GraphNodePosition {
                    graph_node: neighbor,
                    offset: 0,
                };
                let next_remaining = remaining - remaining_in_node;
                if visited.insert((next_position, next_remaining)) {
                    queue.push_back((next_position, next_remaining));
                }
            }
        } else {
            let remaining_in_node = position.offset;
            if -remaining <= remaining_in_node {
                let result = GraphNodePosition {
                    graph_node: node,
                    offset: position.offset + remaining,
                };
                if result_seen.insert(result) {
                    results.push(result);
                }
                continue;
            }

            let mut neighbors = graph
                .neighbors_directed(node, Direction::Incoming)
                .collect::<Vec<_>>();
            if (neighbors.is_empty() || expand_through_existing_paths)
                && expand(graph, node.node_id)
            {
                neighbors = graph
                    .neighbors_directed(node, Direction::Incoming)
                    .collect();
            }

            for neighbor in neighbors {
                let next_position = GraphNodePosition {
                    graph_node: neighbor,
                    offset: neighbor.length(),
                };
                let next_remaining = remaining + remaining_in_node;
                if visited.insert((next_position, next_remaining)) {
                    queue.push_back((next_position, next_remaining));
                }
            }
        }
    }

    if results.is_empty() {
        return Err(GraphError::OutOfBounds(distance));
    }

    Ok(results)
}

fn graph_node(block: &GraphLoadBlock) -> GraphNode {
    GraphNode {
        node_id: block.node_id,
        sequence_start: block.start,
        sequence_end: block.end,
    }
}

fn select_junction_or_sequence_blocks<'a>(
    blocks: &[&'a GraphLoadBlock],
) -> Vec<&'a GraphLoadBlock> {
    let junctions = blocks
        .iter()
        .copied()
        .filter(|block| block.start == block.end)
        .collect::<Vec<_>>();
    if junctions.is_empty() {
        blocks.to_vec()
    } else {
        junctions
    }
}

fn block_connections<'a>(
    edge: &GraphLoadEdge,
    source_blocks: &[&'a GraphLoadBlock],
    target_blocks: &[&'a GraphLoadBlock],
) -> Vec<(&'a GraphLoadBlock, &'a GraphLoadBlock)> {
    if edge.source_node_id == edge.target_node_id
        && edge.source_coordinate == edge.target_coordinate
    {
        let source_sequence_blocks = source_blocks
            .iter()
            .copied()
            .filter(|block| block.start != block.end)
            .collect::<Vec<_>>();
        let target_sequence_blocks = target_blocks
            .iter()
            .copied()
            .filter(|block| block.start != block.end)
            .collect::<Vec<_>>();
        let source_junctions = source_blocks
            .iter()
            .copied()
            .filter(|block| block.start == block.end)
            .collect::<Vec<_>>();
        let target_junctions = target_blocks
            .iter()
            .copied()
            .filter(|block| block.start == block.end)
            .collect::<Vec<_>>();

        if source_junctions.is_empty() && target_junctions.is_empty() {
            return source_blocks
                .iter()
                .copied()
                .cartesian_product(target_blocks.iter().copied())
                .collect();
        }

        return source_sequence_blocks
            .into_iter()
            .cartesian_product(target_junctions)
            .chain(
                source_junctions
                    .into_iter()
                    .cartesian_product(target_sequence_blocks),
            )
            .collect();
    }

    select_junction_or_sequence_blocks(source_blocks)
        .into_iter()
        .cartesian_product(select_junction_or_sequence_blocks(target_blocks))
        .collect()
}

/// Builds an in-memory sequence graph from records already loaded by the model layer.
pub fn build_graph(
    edges: &[GraphLoadEdge],
    blocks: &[GraphLoadBlock],
) -> (GenGraph, HashMap<(i64, i64), HashId>) {
    let blocks_by_start = blocks
        .iter()
        .map(|block| {
            (
                BlockKey {
                    node_id: block.node_id,
                    coordinate: block.start,
                },
                block,
            )
        })
        .into_group_map();
    let blocks_by_end = blocks
        .iter()
        .map(|block| {
            (
                BlockKey {
                    node_id: block.node_id,
                    coordinate: block.end,
                },
                block,
            )
        })
        .into_group_map();

    let mut graph = GenGraph::new();
    let mut edges_by_node_pair = HashMap::new();
    for block in blocks {
        graph.add_node(graph_node(block));
    }
    for edge in edges {
        let source_key = BlockKey {
            node_id: edge.source_node_id,
            coordinate: edge.source_coordinate,
        };
        let target_key = BlockKey {
            node_id: edge.target_node_id,
            coordinate: edge.target_coordinate,
        };
        if let Some(source_blocks) = blocks_by_end.get(&source_key)
            && let Some(target_blocks) = blocks_by_start.get(&target_key)
        {
            for (source_block, target_block) in
                block_connections(edge, source_blocks, target_blocks)
            {
                let source_node = graph_node(source_block);
                let target_node = graph_node(target_block);
                let graph_edge = GraphEdge {
                    edge_id: edge.edge_id,
                    source_strand: edge.source_strand,
                    target_strand: edge.target_strand,
                    chromosome_index: edge.chromosome_index,
                    phased: edge.phased,
                    created_on: edge.created_on,
                };
                if let Some(existing_edges) = graph.edge_weight_mut(source_node, target_node) {
                    existing_edges.push(graph_edge);
                } else {
                    graph.add_edge(source_node, target_node, vec![graph_edge]);
                }
                edges_by_node_pair.insert((source_block.id, target_block.id), edge.edge_id);
            }
        }
    }

    (graph, edges_by_node_pair)
}

/// Removes superseded chromosome paths and nodes that are no longer reachable from a graph root.
pub fn prune_graph(graph: &mut GenGraph) {
    let mut root_nodes = HashSet::new();
    let mut edges_to_remove = Vec::new();
    for node in graph.nodes() {
        if node.node_id == PATH_START_NODE_ID {
            root_nodes.insert(node);
        }
        let mut edges_by_chromosome = HashMap::new();
        for (source_node, target_node, edge_weights) in graph.edges(node) {
            for edge_weight in edge_weights {
                if edge_weight.chromosome_index == NO_CHROMOSOME_INDEX
                    || edge_weight.chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
                {
                    continue;
                }
                if edge_weight.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
                    edges_to_remove.push((source_node, target_node));
                    continue;
                }
                edges_by_chromosome
                    .entry(edge_weight.chromosome_index)
                    .and_modify(
                        |(source, target, created_on): &mut (GraphNode, GraphNode, i64)| {
                            if edge_weight.created_on > *created_on {
                                edges_to_remove.push((*source, *target));
                                *source = source_node;
                                *target = target_node;
                                *created_on = edge_weight.created_on;
                            } else {
                                edges_to_remove.push((source_node, target_node));
                            }
                        },
                    )
                    .or_insert((source_node, target_node, edge_weight.created_on));
            }
        }
    }

    for (source, target) in edges_to_remove {
        graph.remove_edge(source, target);
    }

    let reachable_nodes = all_reachable_nodes(&*graph, &Vec::from_iter(root_nodes));
    let nodes_to_remove = graph
        .nodes()
        .filter(|node| !reachable_nodes.contains(node))
        .collect::<Vec<_>>();
    for node in nodes_to_remove {
        graph.remove_node(node);
    }
}

/// Enumerates the sequence spelled by every simple start-to-end path in a loaded graph.
pub fn get_all_sequences(
    graph: &GenGraph,
    sequences_by_node: &HashMap<GraphNode, String>,
) -> HashSet<String> {
    let start_nodes = graph
        .nodes()
        .filter(|node| is_start_node(node.node_id))
        .collect::<Vec<_>>();
    let end_nodes = graph
        .nodes()
        .filter(|node| is_end_node(node.node_id))
        .collect::<Vec<_>>();
    let mut sequences = HashSet::new();

    for start_node in start_nodes {
        for end_node in &end_nodes {
            if start_node == *end_node {
                if start_node.node_id != PATH_START_NODE_ID
                    && start_node.node_id != PATH_END_NODE_ID
                {
                    sequences.insert(sequences_by_node[&start_node].clone());
                }
                continue;
            }
            for path in all_simple_paths(graph, start_node, *end_node) {
                let mut sequence = String::new();
                for node in path {
                    sequence.push_str(&sequences_by_node[&node]);
                }
                sequences.insert(sequence);
            }
        }
    }

    sequences
}
