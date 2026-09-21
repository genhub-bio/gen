//! Read-only graph projection and region resolution shared by navigation and editing.

use core::ops::Range;

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
use pyo3::{
    PyResult,
    exceptions::{PyRuntimeError, PyValueError},
};

use super::{locus::GraphLocusExt as _, utils::block_group_err_to_pyerr};

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
    let conn = context.graph().conn();
    // Markers are excluded before projection, as in sequence enumeration, because a later
    // edit can reuse a marker's edge as a live one.
    let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
        .into_iter()
        .filter(|edge| edge.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .collect::<Vec<_>>();
    let mut graph =
        BlockGroup::get_graph_from_edges(conn, context.workspace(), block_group_id, &edges)
            .map_err(block_group_err_to_pyerr)?;
    BlockGroup::prune_graph(&mut graph);
    Ok(graph)
}

pub(crate) fn forward_edge(graph: &GenGraph, source: GraphNode, target: GraphNode) -> bool {
    graph.edge_weight(source, target).is_some_and(|edges| {
        edges.iter().all(|edge| {
            edge.source_strand == Strand::Forward && edge.target_strand == Strand::Forward
        })
    })
}
