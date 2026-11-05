//! Read-only graph projection and region resolution shared by navigation and editing.

use core::ops::Range;
use std::collections::{HashMap, HashSet, VecDeque};

use gen_core::{
    HashId, NodeIntervalBlock, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, is_terminal,
};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::BlockGroupEdge,
    db::DbContext,
    locus::GraphLocus,
    region::{Region, ResolvedRegionKind, resolve},
};
use petgraph::Direction::{Incoming, Outgoing};
use pyo3::{
    PyResult,
    exceptions::{PyRuntimeError, PyValueError},
};

use super::{locus::GraphLocusExt as _, position::Position, utils::block_group_err_to_pyerr};

/// A locus and the sequence graph named by its region, when available.
pub(crate) struct LocusTarget {
    pub(crate) locus: GraphLocus,
    pub(crate) block_group_id: Option<HashId>,
}

/// Converts a region string into the block slices its coordinates cover.
pub(crate) fn locus_from_region(
    context: &DbContext,
    region: &str,
    collection_name: &str,
    sample_name: &str,
) -> PyResult<LocusTarget> {
    let conn = context.graph().conn();
    let parsed = Region::parse(region)
        .map_err(|err| PyValueError::new_err(format!("invalid region '{region}': {err}")))?;
    let resolved = resolve(&parsed, conn, collection_name, sample_name)
        .map_err(|err| PyValueError::new_err(format!("cannot resolve region '{region}': {err}")))?;
    // A block-group region's own tree spans every route through the graph, so once
    // edits add bubbles one coordinate maps to several blocks. Its current path gives
    // the single linear reading the coordinates were written against.
    let tree = match (resolved.kind, &resolved.path) {
        (ResolvedRegionKind::BlockGroup, Some(path)) => path
            .intervaltree(conn)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
        _ => resolved
            .intervaltree(conn, context.workspace())
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
    };
    let mut blocks: Vec<_> = tree
        .iter()
        .filter(|entry| !is_terminal(entry.value.node_id))
        .map(|entry| (entry.range.clone(), entry.value))
        .collect();
    blocks.sort_by_key(|(range, _)| (range.start, range.end));
    if blocks
        .windows(2)
        .any(|pair| pair[1].0.start < pair[0].0.end)
    {
        return Err(PyValueError::new_err(format!(
            "region '{region}' maps to more than one route through the graph"
        )));
    }

    let slice_of = |range: &Range<i64>, block: &NodeIntervalBlock, from: i64, to: i64| {
        let length = (block.sequence_end - block.sequence_start) as usize;
        let (local_from, local_to) = ((from - range.start) as usize, (to - range.start) as usize);
        let (start, end) = if block.strand == Strand::Reverse {
            (length - local_to, length - local_from)
        } else {
            (local_from, local_to)
        };
        GraphNodeSlice {
            block: GraphNode {
                node_id: block.node_id,
                sequence_start: block.sequence_start,
                sequence_end: block.sequence_end,
            },
            start,
            end,
            strand: block.strand,
        }
    };

    let slices: Vec<GraphNodeSlice> = blocks
        .iter()
        .filter(|(range, _)| range.start < resolved.end && resolved.start < range.end)
        .map(|(range, block)| {
            slice_of(
                range,
                block,
                range.start.max(resolved.start),
                range.end.min(resolved.end),
            )
        })
        .collect();
    if slices.is_empty() {
        return Err(PyValueError::new_err(format!(
            "region '{region}' does not cover any sequence"
        )));
    }
    let locus = GraphLocus { slices };
    if locus.length() as i64 != resolved.end - resolved.start {
        return Err(PyValueError::new_err(format!(
            "region '{region}' extends beyond the available sequence"
        )));
    }
    Ok(LocusTarget {
        locus,
        block_group_id: Some(resolved.block_group.id),
    })
}

/// The graph as sequence reading sees it: retired edit sites dropped, then pruned.
pub(crate) fn current_graph(context: &DbContext, block_group_id: &HashId) -> PyResult<GenGraph> {
    let mut graph = unpruned_graph(context, block_group_id)?;
    BlockGroup::prune_graph(&mut graph);
    Ok(graph)
}

/// The graph with retired edit sites dropped but not pruned, so sequence that no route from the
/// start reaches, such as an excised circle, keeps its blocks.
pub(crate) fn unpruned_graph(context: &DbContext, block_group_id: &HashId) -> PyResult<GenGraph> {
    let conn = context.graph().conn();
    // Markers are excluded before projection, as in sequence enumeration, because a later
    // edit can reuse a marker's edge as a live one.
    let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
        .into_iter()
        .filter(|edge| edge.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .collect::<Vec<_>>();
    BlockGroup::get_graph_from_edges(conn, context.workspace(), block_group_id, &edges)
        .map_err(block_group_err_to_pyerr)
}

pub(crate) fn forward_edge(graph: &GenGraph, source: GraphNode, target: GraphNode) -> bool {
    graph.edge_weight(source, target).is_some_and(|edges| {
        edges.iter().all(|edge| {
            edge.source_strand == Strand::Forward && edge.target_strand == Strand::Forward
        })
    })
}

/// Whether reading from `from` on its own strand can arrive at `to` on its own strand, where each
/// stored edge can be used as written or as its reverse complement, so a walk may be flipped any
/// number of times on the way. The search runs over a block together with the side of it being
/// read, because the side a walk is on depends on how many flips it has made and not on the block.
/// Within one block, reading on a strand moves toward higher coordinates on the forward strand and
/// lower ones on the reverse.
pub(crate) fn reaches_oriented(graph: &GenGraph, from: &Position, to: &Position) -> bool {
    let (Some(first), Some(last)) = (from.block(graph), to.block(graph)) else {
        return false;
    };
    let in_reading_order = |upstream: i64, downstream: i64, strand: Strand| {
        if strand == Strand::Reverse {
            upstream >= downstream
        } else {
            upstream <= downstream
        }
    };
    if first == last
        && from.strand == to.strand
        && in_reading_order(from.coordinate, to.coordinate, from.strand)
    {
        return true;
    }
    let mut pending = VecDeque::from([(first, from.strand)]);
    let mut seen = HashSet::from([(first, from.strand)]);
    while let Some((block, side)) = pending.pop_front() {
        let mut arrivals = Vec::new();
        for successor in graph.neighbors_directed(block, Outgoing) {
            for edge in graph.edge_weight(block, successor).into_iter().flatten() {
                if edge.source_strand == side {
                    arrivals.push((successor, edge.target_strand));
                }
            }
        }
        for predecessor in graph.neighbors_directed(block, Incoming) {
            for edge in graph.edge_weight(predecessor, block).into_iter().flatten() {
                if flipped(edge.target_strand) == side {
                    arrivals.push((predecessor, flipped(edge.source_strand)));
                }
            }
        }
        for (next, next_side) in arrivals {
            if is_terminal(next.node_id) || !seen.insert((next, next_side)) {
                continue;
            }
            if next == last && next_side == to.strand {
                return true;
            }
            pending.push_back((next, next_side));
        }
    }
    false
}

const fn flipped(strand: Strand) -> Strand {
    match strand {
        Strand::Forward => Strand::Reverse,
        Strand::Reverse => Strand::Forward,
        other => other,
    }
}

/// The fewest-block forward route from `start` to `end`, both included, if `end` can be reached.
pub(crate) fn shortest_route(
    graph: &GenGraph,
    start: GraphNode,
    end: GraphNode,
) -> Option<Vec<GraphNode>> {
    let mut parents: HashMap<GraphNode, GraphNode> = HashMap::new();
    let mut queue = VecDeque::from([start]);
    let mut seen = HashSet::from([start]);
    while let Some(block) = queue.pop_front() {
        if block == end {
            let mut route = vec![end];
            while let Some(parent) = parents.get(route.last()?) {
                route.push(*parent);
            }
            route.reverse();
            return Some(route);
        }
        for successor in graph.neighbors_directed(block, Outgoing) {
            if !is_terminal(successor.node_id)
                && forward_edge(graph, block, successor)
                && seen.insert(successor)
            {
                parents.insert(successor, block);
                queue.push_back(successor);
            }
        }
    }
    None
}
