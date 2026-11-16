use core::{fmt::Display, ops::Range};
use std::collections::HashSet;

use r#gen::graphs::{
    BlockGroupChunk, NodePoint, load_block_group_chunk,
    operators::{GraphOperationError, make_stitch_from_block_groups},
};
use gen_core::{HashId, NodeIntervalBlock, Strand, is_terminal};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup, SubgraphBoundary},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::DbContext,
    edge::Edge,
    sample::{NewSample, Sample},
};
use petgraph::algo::{is_cyclic_directed, kosaraju_scc};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

/// One piece of a stitch: a whole block group, or the span of a locus within one.
///
/// A locus is linear, but stitching it takes every variant route between its first and last
/// positions, so the result keeps the variation of that part of the graph without first
/// deriving a subgraph for it. Each locus range is a node and a span of that node's own sequence,
/// in the order the locus reads them.
pub(super) enum StitchSource {
    BlockGroup(HashId),
    Locus {
        block_group_id: HashId,
        ranges: Vec<(HashId, Range<i64>)>,
    },
}

fn stitch_error(error: impl Display) -> PyErr {
    PyRuntimeError::new_err(format!("Error stitching parts: {error}"))
}

/// Creates a new block group in `new_sample_name` by concatenating `sources` in order: the end of
/// each piece is connected to the start of the next. The result has a current path when every
/// piece does (a locus contributes the route it spells).
pub(super) fn stitch_sources(
    context: &DbContext,
    collection_name: &str,
    new_sample_name: &str,
    new_region_name: &str,
    sources: &[StitchSource],
) -> PyResult<BlockGroup> {
    let conn = context.graph().conn();
    conn.execute_batch("SAVEPOINT stitch_sources")
        .map_err(stitch_error)?;
    let result = create_stitched_from_sources(
        context,
        collection_name,
        new_sample_name,
        new_region_name,
        sources,
    );
    match result {
        Ok(block_group) => {
            conn.execute_batch("RELEASE stitch_sources")
                .map_err(stitch_error)?;
            Ok(block_group)
        }
        Err(error) => {
            conn.execute_batch("ROLLBACK TO stitch_sources")
                .map_err(stitch_error)?;
            conn.execute_batch("RELEASE stitch_sources")
                .map_err(stitch_error)?;
            Err(error)
        }
    }
}

fn create_stitched_from_sources(
    context: &DbContext,
    collection_name: &str,
    new_sample_name: &str,
    new_region_name: &str,
    sources: &[StitchSource],
) -> PyResult<BlockGroup> {
    let conn = context.graph().conn();
    let mut seen_block_group_ids = HashSet::new();
    let mut seen_edge_ids = HashSet::new();
    for source in sources {
        if let StitchSource::BlockGroup(block_group_id) = source {
            if !seen_block_group_ids.insert(*block_group_id) {
                return Err(stitch_error(GraphOperationError::InvalidStitchInput(
                    format!("sequence graph {block_group_id} appears more than once"),
                )));
            }
            // Stitching reads each graph's current path to build the new one.
            if BlockGroup::get_current_path(conn, block_group_id, None).is_err() {
                return Err(stitch_error(GraphOperationError::InvalidStitchInput(
                    format!("sequence graph {block_group_id} has no current path to stitch"),
                )));
            }
            for edge in BlockGroupEdge::edges_for_block_group(conn, block_group_id, None) {
                if !edge.edge.is_start_edge()
                    && !edge.edge.is_end_edge()
                    && !seen_edge_ids.insert(edge.edge.id)
                {
                    return Err(stitch_error(GraphOperationError::InvalidStitchInput(
                        format!("sequence graphs share edge {}", edge.edge.id),
                    )));
                }
            }
        }
    }

    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
            name: new_sample_name,
            ..Default::default()
        },
    );
    let child_block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name: new_sample_name,
            name: new_region_name,
            ..Default::default()
        },
    )
    .map_err(stitch_error)?;

    let mut chunks = Vec::with_capacity(sources.len());
    for source in sources {
        match source {
            StitchSource::BlockGroup(block_group_id) => {
                let block_group_edges =
                    BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
                        .into_iter()
                        .filter(|edge| !edge.edge.is_start_edge() && !edge.edge.is_end_edge())
                        .map(|edge| BlockGroupEdgeData {
                            block_group_id: child_block_group.id,
                            edge_id: edge.edge.id,
                            chromosome_index: edge.chromosome_index,
                            phased: edge.phased,
                        })
                        .collect::<Vec<_>>();
                BlockGroupEdge::bulk_create(conn, &block_group_edges);
                chunks.push(load_block_group_chunk(conn, *block_group_id));
            }
            StitchSource::Locus {
                block_group_id,
                ranges,
            } => chunks.push(add_locus_chunk(
                context,
                block_group_id,
                ranges,
                &child_block_group.id,
            )?),
        }
    }

    make_stitch_from_block_groups(context, &chunks, child_block_group.id, new_region_name)
        .map_err(stitch_error)?;
    validate_stitched_block_group_is_acyclic(context, &child_block_group.id)?;
    Ok(child_block_group)
}

/// Rejects a stitch whose graph loops through sequence blocks, which would read bases twice.
/// Zero-width routing blocks are contracted first, since an insertion inside a node meets its own
/// routing block again without reading any base twice.
fn validate_stitched_block_group_is_acyclic(
    context: &DbContext,
    block_group_id: &HashId,
) -> PyResult<()> {
    let mut graph = BlockGroup::get_graph(
        context.graph().conn(),
        context.workspace(),
        block_group_id,
        None,
    )
    .map_err(stitch_error)?;
    contract_zero_width_blocks(&mut graph);
    if !is_cyclic_directed(&graph) {
        return Ok(());
    }
    let describe = |block: &GraphNode| {
        format!(
            "{}:{}-{}",
            block.node_id, block.sequence_start, block.sequence_end
        )
    };
    let cycle = kosaraju_scc(&graph)
        .into_iter()
        .find(|component| component.len() > 1)
        .map(|component| {
            graph
                .all_edges()
                .filter(|(source, target, _)| {
                    component.contains(source) && component.contains(target)
                })
                .map(|(source, target, _)| {
                    format!("{} -> {}", describe(&source), describe(&target))
                })
                .collect::<Vec<_>>()
                .join("; ")
        })
        .unwrap_or_default();
    Err(stitch_error(GraphOperationError::StitchedGraphCycle(
        format!("block group {block_group_id} is cyclic through {cycle}"),
    )))
}

/// Replaces each zero-width routing block by direct edges between the blocks it joins.
///
/// Routing blocks consume no bases, so a route that crosses one spells what the same route
/// spells without it. Several routes through a coordinate can differ only in which routing
/// blocks they cross, which would list one sequence once per crossing; contracting them leaves
/// one edge per pair of blocks that read into each other. A cycle of routing blocks is crossed
/// once, and a block that would read into itself gains no edge, so simple-path enumeration
/// still ends.
fn contract_zero_width_blocks(graph: &mut GenGraph) {
    let routing = graph
        .nodes()
        .filter(|node| !is_terminal(node.node_id) && node.length() == 0)
        .collect::<HashSet<_>>();
    if routing.is_empty() {
        return;
    }
    let mut contracted = Vec::new();
    for source in graph.nodes().filter(|node| !routing.contains(node)) {
        let mut crossed = HashSet::new();
        let mut pending = Vec::new();
        for (_, target, weights) in graph.edges(source) {
            if routing.contains(&target) {
                if crossed.insert(target) {
                    pending.push(target);
                }
            } else if !graph.contains_edge(source, target) {
                contracted.push((source, target, weights.clone()));
            }
        }
        while let Some(current) = pending.pop() {
            for (_, target, weights) in graph.edges(current) {
                if routing.contains(&target) {
                    if crossed.insert(target) {
                        pending.push(target);
                    }
                } else if target != source && !graph.contains_edge(source, target) {
                    contracted.push((source, target, weights.clone()));
                }
            }
        }
    }
    for (source, target, weights) in contracted {
        if !graph.contains_edge(source, target) {
            graph.add_edge(source, target, weights);
        }
    }
    for node in routing {
        graph.remove_node(node);
    }
}

/// Copies the subgraph spanned by a locus into `target_block_group_id` and returns the chunk that
/// stitches it, whose path is the route the locus itself spells.
fn add_locus_chunk(
    context: &DbContext,
    block_group_id: &HashId,
    ranges: &[(HashId, Range<i64>)],
    target_block_group_id: &HashId,
) -> PyResult<BlockGroupChunk> {
    let conn = context.graph().conn();
    let (Some((first_node_id, first_range)), Some((last_node_id, last_range))) =
        (ranges.first(), ranges.last())
    else {
        return Err(stitch_error(GraphOperationError::InvalidStitchInput(
            "a locus to stitch must cover at least one base".to_string(),
        )));
    };
    let graph = BlockGroup::get_graph(conn, context.workspace(), block_group_id, None)
        .map_err(stitch_error)?;
    let holds = |node_id: &HashId, coordinate: i64| {
        graph.nodes().any(|node| {
            node.node_id == *node_id
                && node.sequence_start <= coordinate
                && node.sequence_end >= coordinate
        })
    };
    if !holds(first_node_id, first_range.start) || !holds(last_node_id, last_range.end) {
        return Err(stitch_error(GraphOperationError::InvalidStitchInput(
            "a locus to stitch is not part of its sequence graph".to_string(),
        )));
    }

    let boundary_block = |node_id: &HashId, range: &Range<i64>| NodeIntervalBlock {
        node_id: *node_id,
        start: 0,
        end: 0,
        sequence_start: range.start,
        sequence_end: range.end,
        strand: Strand::Forward,
    };
    let start_block = boundary_block(first_node_id, first_range);
    let end_block = boundary_block(last_node_id, last_range);
    BlockGroup::derive_subgraph(
        conn,
        context.workspace(),
        block_group_id,
        SubgraphBoundary {
            block: &start_block,
            sequence_coordinate: first_range.start,
        },
        SubgraphBoundary {
            block: &end_block,
            sequence_coordinate: last_range.end,
        },
        target_block_group_id,
        false,
    )
    .map_err(stitch_error)?;

    let mut path_edges = Vec::with_capacity(ranges.len().saturating_sub(1));
    for pair in ranges.windows(2) {
        path_edges.push(
            Edge::create(
                conn,
                pair[0].0,
                pair[0].1.end,
                Strand::Forward,
                pair[1].0,
                pair[1].1.start,
                Strand::Forward,
            )
            .map_err(stitch_error)?,
        );
    }
    let start_point = NodePoint {
        id: *first_node_id,
        coordinate: first_range.start,
        strand: Strand::Forward,
    };
    let end_point = NodePoint {
        id: *last_node_id,
        coordinate: last_range.end,
        strand: Strand::Forward,
    };
    Ok(BlockGroupChunk {
        entry_node_points: vec![start_point.clone()],
        exit_node_points: vec![end_point.clone()],
        path_edges,
        path_start_point: Some(start_point),
        path_end_point: Some(end_point),
    })
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
    use gen_graph::GraphEdge;

    use super::*;

    #[test]
    fn test_contracting_routing_blocks_leaves_one_route_per_pair_of_sequence_blocks() {
        let node = |name: &str, start: i64, end: i64| GraphNode {
            node_id: HashId::convert_str(name),
            sequence_start: start,
            sequence_end: end,
        };
        let edge = GraphEdge {
            edge_id: HashId::convert_str("edge"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 0,
            phased: 0,
            created_on: 0,
        };
        let start = node("start", 0, 0);
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            ..start
        };
        let end = GraphNode {
            node_id: PATH_END_NODE_ID,
            ..node("end", 0, 0)
        };
        let (left, right) = (node("sequence", 0, 5), node("sequence", 5, 10));
        let (first, second) = (node("sequence", 5, 5), node("other", 0, 0));
        let mut graph = GenGraph::new();
        // The left block reaches the right one directly and through a cycle of routing blocks,
        // and a routing block reads back into the left block.
        for (source, target) in [
            (start, left),
            (left, right),
            (left, first),
            (first, right),
            (first, second),
            (second, first),
            (first, left),
            (right, end),
        ] {
            graph.add_edge(source, target, vec![edge]);
        }

        contract_zero_width_blocks(&mut graph);

        let mut edges = graph
            .all_edges()
            .map(|(source, target, _)| (source, target))
            .collect::<Vec<_>>();
        edges.sort_unstable();
        let mut expected = vec![(start, left), (left, right), (right, end)];
        expected.sort_unstable();
        assert_eq!(edges, expected);
        assert_eq!(graph.node_count(), 4);
    }
}
