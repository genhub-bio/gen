//! Single graph edits addressed by Python targets.
//!
//! A region string, `Locus`, `Annotation`, or `Position` is canonicalized to node-absolute
//! ranges, checked against the current graph, and applied as one `BlockGroup::insert_change`
//! recorded as one operation.

use core::ops::Range;
use std::collections::HashMap;

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NodeIntervalBlock, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
    PathBlock, Strand, is_end_node, is_start_node, is_terminal,
};
use gen_graph::{GenGraph, GraphNode, GraphNodePosition, GraphNodeSlice};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange, BlockGroupError},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::{DbContext, GraphConnection},
    edge::Edge,
    errors::{OperationError, QueryError},
    history::dolt::hash_of,
    locus::GraphLocus,
    node::Node,
    operations::{OperationInfo, OperationSummary},
    path::Path,
    region::{Region, ResolvedGenRegion, ResolvedRegionKind, resolve},
    sequence::{Sequence, reverse_complement},
};
use petgraph::Direction::{Incoming, Outgoing};
use pyo3::{
    Bound, PyAny, PyRef, PyResult,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    types::PyAnyMethods as _,
};

use super::{
    annotation::PyAnnotation,
    block_group::PySequenceGraph,
    graph_search::{PositionSide, PyGraphLocus, PyGraphPos},
    locus::GraphLocusExt as _,
    repository::run_context_operation_write,
    utils::block_group_err_to_pyerr,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditKind {
    Insert,
    Replace,
    Delete,
}

impl EditKind {
    const fn verb(self) -> &'static str {
        match self {
            EditKind::Insert => "insert",
            EditKind::Replace => "replace",
            EditKind::Delete => "delete",
        }
    }
}

/// What to write at a target, and which sample the result lands in.
pub(crate) struct EditRequest<'a> {
    pub kind: EditKind,
    pub sequence: &'a str,
    pub message: Option<&'a str>,
    /// Add the edit as an alternative next to the target instead of superseding it. The
    /// current path is left unchanged.
    pub stack: bool,
}

/// A Python target in graph space. Region strings also name the sequence graph
/// they resolved in; other targets are located by the bases they cover.
struct EditTarget {
    locus: GraphLocus,
    block_group_id: Option<HashId>,
    /// Which block a zero-length target attaches to when it lands on a block edge.
    side: PositionSide,
}

/// The resolved attachment points of an edit in the destination graph.
struct EditSpan {
    start: GraphNodePosition,
    end: GraphNodePosition,
    is_reverse: bool,
    chromosome_index: i64,
    phased: i64,
}

/// Applies one edit to a sequence graph in place.
pub(crate) fn edit_sequence_graph(
    sequence_graph: &PySequenceGraph,
    target: &Bound<'_, PyAny>,
    request: &EditRequest<'_>,
) -> PyResult<Option<GraphLocus>> {
    let context = sequence_graph.require_context(request.kind.verb())?;
    let target = resolve_target(
        context,
        target,
        &sequence_graph.collection_name,
        &sequence_graph.sample_name,
    )?;
    if let Some(block_group_id) = target.block_group_id
        && block_group_id != sequence_graph.id
    {
        return Err(PyValueError::new_err(format!(
            "region resolves outside sequence graph '{}'",
            sequence_graph.name
        )));
    }
    apply_edit(
        context,
        &sequence_graph.id,
        &target.locus,
        target.side,
        request,
    )
}

fn resolve_target(
    context: &DbContext,
    target: &Bound<'_, PyAny>,
    collection_name: &str,
    sample_name: &str,
) -> PyResult<EditTarget> {
    let mut side = PositionSide::default();
    let locus = if let Ok(annotation) = target.extract::<PyRef<PyAnnotation>>() {
        annotation.graph_locus()
    } else if let Ok(locus) = target.extract::<PyRef<PyGraphLocus>>() {
        locus.inner.clone()
    } else if let Ok(position) = target.extract::<PyRef<PyGraphPos>>() {
        side = position.side;
        GraphLocus {
            slices: vec![GraphNodeSlice {
                block: position.inner.block,
                start: position.inner.offset,
                end: position.inner.offset,
                strand: Strand::Forward,
            }],
        }
    } else if let Ok(region) = target.extract::<&str>() {
        return locus_from_region(context, region, collection_name, sample_name);
    } else {
        return Err(PyTypeError::new_err(
            "edit target must be a region string, Locus, Annotation, or Position",
        ));
    };
    Ok(EditTarget {
        locus,
        block_group_id: None,
        side,
    })
}

/// Converts a region string into the block slices its coordinates cover.
fn locus_from_region(
    context: &DbContext,
    region: &str,
    collection_name: &str,
    sample_name: &str,
) -> PyResult<EditTarget> {
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

    let slices: Vec<GraphNodeSlice> = if resolved.start == resolved.end {
        let point = resolved.start;
        let containing = blocks
            .iter()
            .find(|(range, _)| range.start <= point && point < range.end);
        let ending = blocks.iter().find(|(range, _)| range.end == point);
        if let (Some((after_range, after_block)), Some((before_range, before_block))) =
            (containing, ending)
        {
            require_one_insertion_point(
                context,
                &resolved.block_group.id,
                region,
                &slice_of(before_range, before_block, point, point),
                &slice_of(after_range, after_block, point, point),
            )?;
        }
        containing
            .or(ending)
            .map(|(range, block)| slice_of(range, block, point, point))
            .into_iter()
            .collect()
    } else {
        blocks
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
            .collect()
    };
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
    Ok(EditTarget {
        locus,
        block_group_id: Some(resolved.block_group.id),
        side: PositionSide::default(),
    })
}

/// A path coordinate on a block edge names both the end of the block before it and the
/// start of the block after it. Where the graph forks or joins there, those are different
/// insertion points and the coordinate cannot say which one is meant.
fn require_one_insertion_point(
    context: &DbContext,
    block_group_id: &HashId,
    region: &str,
    before: &GraphNodeSlice,
    after: &GraphNodeSlice,
) -> PyResult<()> {
    let graph = current_graph(context, block_group_id)?;
    let anchors_of = |slice: &GraphNodeSlice| {
        let locus = GraphLocus {
            slices: vec![*slice],
        };
        locate_span(&graph, &locus.canonical(), PositionSide::Preceding)
            .map(|span| (span.start, span.end))
    };
    match (anchors_of(before), anchors_of(after)) {
        (Ok(before), Ok(after)) if before == after => Ok(()),
        (Err(_), Err(error)) => Err(error),
        _ => Err(PyValueError::new_err(format!(
            "region '{region}' is where the graph forks or joins, so it could mean the end of \
             the part before it or the start of the part after it; target one of those \
             positions instead"
        ))),
    }
}

fn apply_edit(
    context: &DbContext,
    source_block_group_id: &HashId,
    target: &GraphLocus,
    side: PositionSide,
    request: &EditRequest<'_>,
) -> PyResult<Option<GraphLocus>> {
    validate_request(target, request)?;
    let canonical = target.canonical();
    run_context_operation_write(
        context,
        |context| {
            let conn = context.graph().conn();
            let source = BlockGroup::get_by_id(conn, source_block_group_id, None)
                .map_err(block_group_err_to_pyerr)?;
            let graph = current_graph(context, &source.id)?;
            let mut span = locate_span(&graph, &canonical, side)?;
            if request.stack {
                // Pruning never competes on this index, so the existing route stays.
                span.chromosome_index = INDETERMINATE_CHROMOSOME_INDEX;
                span.phased = 0;
            }

            // Capture the path before insert_change adds edges without updating it.
            let path_target = if request.stack {
                None
            } else {
                locate_span_on_current_path(conn, &source.id, &span)?
            };

            let block = if request.kind == EditKind::Delete {
                PathBlock {
                    node_id: span.start.graph_node.node_id,
                    block_sequence: String::new(),
                    sequence_start: 0,
                    sequence_end: 0,
                    path_start: 0,
                    path_end: 0,
                    strand: Strand::Forward,
                }
            } else {
                inserted_block(conn, &source, &span, request.sequence)?
            };
            let change = BlockGroupChange {
                region: anchored_region(source.clone(), &span),
                path_accession: None,
                block: block.clone(),
                chromosome_index: span.chromosome_index,
                phased: span.phased,
                // Keeps the original route live for a stacked edit.
                preserve_edge: request.stack,
            };
            // Reactivating an older route (for example deleting a second insertion at
            // the same site) must receive this edit's timestamp for graph pruning.
            let reactivated = change
                .region
                .plan_edges(conn, context.workspace(), &change, None)
                .map_err(block_group_err_to_pyerr)?
                .into_iter()
                .filter(|edge| edge.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
                .map(|edge| {
                    BlockGroupEdgeData {
                        block_group_id: source.id,
                        edge_id: edge.edge_data.id_hash(),
                        chromosome_index: edge.chromosome_index,
                        phased: edge.phased,
                    }
                    .id_hash()
                })
                .collect::<Vec<_>>();
            BlockGroupEdge::select(conn)
                .delete_by_ids(reactivated)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            BlockGroup::insert_change(conn, context.workspace(), &change)
                .map_err(block_group_err_to_pyerr)?;

            // Update the materialized path only when the edit lies on its route.
            if let Some(path) = path_target {
                let middle_edge_ids = if request.kind == EditKind::Delete {
                    let deletion_edge = Edge::select(conn)
                        .source_node_id(span.start.graph_node.node_id)
                        .source_coordinate(span.start.coordinate())
                        .target_node_id(span.end.graph_node.node_id)
                        .target_coordinate(span.end.coordinate())
                        .load()
                        .map_err(|error| block_group_err_to_pyerr(error.into()))?
                        .into_iter()
                        .next()
                        .ok_or_else(|| PyRuntimeError::new_err("missing deletion edge"))?;
                    vec![deletion_edge.id]
                } else {
                    let edge_to_new_node = Edge::select(conn)
                        .target_node_id(block.node_id)
                        .load()
                        .map_err(|error| block_group_err_to_pyerr(error.into()))?
                        .into_iter()
                        .next()
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("missing edge into inserted node")
                        })?;
                    let edge_from_new_node = Edge::select(conn)
                        .source_node_id(block.node_id)
                        .load()
                        .map_err(|error| block_group_err_to_pyerr(error.into()))?
                        .into_iter()
                        .next()
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("missing edge from inserted node")
                        })?;
                    vec![edge_to_new_node.id, edge_from_new_node.id]
                };
                splice_current_path(conn, &path, &span, &middle_edge_ids)?;
            }

            let inserted = (request.kind != EditKind::Delete).then(|| GraphLocus {
                slices: vec![GraphNodeSlice {
                    block: GraphNode {
                        node_id: block.node_id,
                        sequence_start: block.sequence_start,
                        sequence_end: block.sequence_end,
                    },
                    start: 0,
                    end: (block.sequence_end - block.sequence_start) as usize,
                    strand: if span.is_reverse {
                        Strand::Reverse
                    } else {
                        Strand::Forward
                    },
                }],
            });
            let summary = OperationSummary::new(
                OperationInfo {
                    files: vec![],
                    description: "sequence_edit".to_string(),
                },
                request.message.map_or_else(
                    || {
                        format!(
                            "{}: {} {} in sample '{}'",
                            source.name,
                            request.kind.verb(),
                            describe_locus(&canonical),
                            source.sample_name
                        )
                    },
                    str::to_string,
                ),
            );
            Ok((inserted, summary))
        },
        |err| match err {
            OperationError::NoChanges => {
                PyValueError::new_err("edit made no changes to the sequence graph")
            }
            other => PyRuntimeError::new_err(format!("failed to record edit: {other}")),
        },
    )
}

fn validate_request(target: &GraphLocus, request: &EditRequest<'_>) -> PyResult<()> {
    if target.slices.is_empty() {
        return Err(PyValueError::new_err("edit target covers no sequence"));
    }
    let is_point = target.length() == 0;
    match request.kind {
        EditKind::Insert if !is_point => Err(PyValueError::new_err(
            "insert() needs a zero-length target such as a Position; use replace() to overwrite bases",
        )),
        EditKind::Replace | EditKind::Delete if is_point => Err(PyValueError::new_err(format!(
            "{}() needs a target that covers at least one base",
            request.kind.verb()
        ))),
        EditKind::Insert | EditKind::Replace if request.sequence.is_empty() => {
            Err(PyValueError::new_err(format!(
                "{}() needs a non-empty sequence; use delete() to remove bases",
                request.kind.verb()
            )))
        }
        _ => Ok(()),
    }
}

/// Resolve the complete target against the active route before selecting flanking
/// anchors. Reverse search hits may list slices in graph order, whereas a reverse
/// complement lists them in reading order; validate connectivity in either order.
fn locate_span(graph: &GenGraph, canonical: &GraphLocus, side: PositionSide) -> PyResult<EditSpan> {
    let is_reverse = canonical.slices[0].strand == Strand::Reverse;
    if canonical.slices.iter().any(|slice| {
        if is_reverse {
            slice.strand != Strand::Reverse
        } else {
            !matches!(slice.strand, Strand::Forward | Strand::Unknown)
        }
    }) {
        return Err(PyValueError::new_err(
            "target mixes strands; edit each strand separately",
        ));
    }
    let mut locus = if is_reverse {
        canonical.reverse_complement()
    } else {
        canonical.clone()
    };
    let slices = current_slices(graph, &locus, side).or_else(|error| {
        if !is_reverse {
            return Err(error);
        }
        locus.slices.reverse();
        current_slices(graph, &locus, side)
    })?;
    let first = slices.first().expect("should have a target slice");
    let last = slices.last().expect("should have a target slice");
    // A span that covers bases anchors on its own blocks, so superseding it never touches the
    // edges shared with neighbouring routes. An insertion anchors on its neighbours instead,
    // since both of its edges would otherwise meet at one junction and form a cycle.
    let (start, end) = if locus.length() == 0 {
        (boundary(graph, first, true)?, boundary(graph, last, false)?)
    } else {
        (
            GraphNodePosition {
                graph_node: first.block,
                offset: first.start as i64,
            },
            GraphNodePosition {
                graph_node: last.block,
                offset: last.end as i64,
            },
        )
    };
    let (chromosome_index, phased) = entry_chromosome(graph, first)?;
    Ok(EditSpan {
        start,
        end,
        is_reverse,
        chromosome_index,
        phased,
    })
}

fn forward_edge(graph: &GenGraph, source: GraphNode, target: GraphNode) -> bool {
    graph.edge_weight(source, target).is_some_and(|edges| {
        edges.iter().all(|edge| {
            edge.source_strand == Strand::Forward && edge.target_strand == Strand::Forward
        })
    })
}

fn current_graph(context: &DbContext, block_group_id: &HashId) -> PyResult<GenGraph> {
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

/// Resolve every immutable base, so an edit cannot silently skip missing interior segments.
/// A zero-length point on a block edge resolves to the block on `side` of it.
fn current_slices(
    graph: &GenGraph,
    locus: &GraphLocus,
    side: PositionSide,
) -> PyResult<Vec<GraphNodeSlice>> {
    let mut slices: Vec<GraphNodeSlice> = Vec::new();
    for slice in &locus.canonical().slices {
        let range = slice.block;
        let mut blocks = graph
            .nodes()
            .filter(|block| {
                block.node_id == range.node_id
                    && if range.length() == 0 {
                        block.sequence_start <= range.sequence_start
                            && range.sequence_end <= block.sequence_end
                    } else {
                        block.sequence_start < range.sequence_end
                            && range.sequence_start < block.sequence_end
                    }
            })
            .collect::<Vec<_>>();
        blocks.sort();
        if range.length() == 0 {
            match side {
                PositionSide::Preceding => blocks.truncate(1),
                PositionSide::Following => {
                    blocks.drain(..blocks.len().saturating_sub(1));
                }
            }
        }
        if blocks.is_empty() {
            return Err(PyValueError::new_err(
                "Target bases or position are not present in the current graph",
            ));
        }
        let mut coordinate = range.sequence_start;
        for block in blocks {
            if graph.edges_directed(block, Incoming).any(|(_, _, edges)| {
                edges
                    .iter()
                    .any(|edge| edge.target_strand != Strand::Forward)
            }) || graph.edges(block).any(|(_, _, edges)| {
                edges
                    .iter()
                    .any(|edge| edge.source_strand != Strand::Forward)
            }) {
                return Err(PyValueError::new_err(
                    "Target cannot be applied to the current graph orientation",
                ));
            }
            let start = range.sequence_start.max(block.sequence_start);
            let end = range.sequence_end.min(block.sequence_end);
            if start != coordinate {
                return Err(PyValueError::new_err(
                    "Target bases are not present in the current graph",
                ));
            }
            let current = GraphNodeSlice {
                block,
                start: (start - block.sequence_start) as usize,
                end: (end - block.sequence_start) as usize,
                strand: Strand::Forward,
            };
            if let Some(previous) = slices.last() {
                let connected = if previous.block == block {
                    previous.end == current.start
                } else {
                    previous.end == previous.block.length() as usize
                        && current.start == 0
                        && forward_edge(graph, previous.block, block)
                };
                if !connected {
                    return Err(PyValueError::new_err(
                        "Target is not a contiguous span in the current graph",
                    ));
                }
            }
            slices.push(current);
            coordinate = end;
        }
        if coordinate != range.sequence_end {
            return Err(PyValueError::new_err(
                "Target bases are not present in the current graph",
            ));
        }
    }
    Ok(slices)
}

/// Returns the current path when both edit anchors lie on its route.
fn locate_span_on_current_path(
    conn: &GraphConnection,
    block_group_id: &HashId,
    span: &EditSpan,
) -> PyResult<Option<Path>> {
    let path = match BlockGroup::get_current_path(conn, block_group_id, None) {
        Ok(path) => path,
        Err(BlockGroupError::QueryError(QueryError::ResultsNotFound(_))) => return Ok(None),
        Err(error) => return Err(block_group_err_to_pyerr(error)),
    };
    let path_blocks = path.coordinate_blocks(conn, None);
    let on_path = covering_block_for_entry(&path_blocks, &span.start).is_some()
        && covering_block_for_exit(&path_blocks, &span.end).is_some();
    Ok(on_path.then_some(path))
}

/// Prefers the block ending at a junction, so a start anchor keeps the preceding route.
fn covering_block_for_entry<'a>(
    path_blocks: &'a [PathBlock],
    position: &GraphNodePosition,
) -> Option<&'a PathBlock> {
    if is_start_node(position.graph_node.node_id) {
        return path_blocks
            .iter()
            .find(|block| is_start_node(block.node_id));
    }
    if is_end_node(position.graph_node.node_id) {
        return path_blocks.iter().find(|block| is_end_node(block.node_id));
    }
    let coordinate = position.coordinate();
    path_blocks
        .iter()
        .find(|block| {
            block.node_id == position.graph_node.node_id && block.sequence_end == coordinate
        })
        .or_else(|| {
            path_blocks.iter().find(|block| {
                block.node_id == position.graph_node.node_id
                    && block.sequence_start <= coordinate
                    && coordinate < block.sequence_end
            })
        })
}

/// Prefers the block starting at a junction, so an end anchor resumes on the later route.
fn covering_block_for_exit<'a>(
    path_blocks: &'a [PathBlock],
    position: &GraphNodePosition,
) -> Option<&'a PathBlock> {
    if is_start_node(position.graph_node.node_id) {
        return path_blocks
            .iter()
            .find(|block| is_start_node(block.node_id));
    }
    if is_end_node(position.graph_node.node_id) {
        return path_blocks.iter().find(|block| is_end_node(block.node_id));
    }
    let coordinate = position.coordinate();
    path_blocks
        .iter()
        .find(|block| {
            block.node_id == position.graph_node.node_id && block.sequence_start == coordinate
        })
        .or_else(|| {
            path_blocks.iter().find(|block| {
                block.node_id == position.graph_node.node_id
                    && block.sequence_start < coordinate
                    && coordinate <= block.sequence_end
            })
        })
}

/// Rebuilds `path` with the edges between the edit anchors replaced by `middle_edge_ids`.
fn splice_current_path(
    conn: &GraphConnection,
    path: &Path,
    span: &EditSpan,
    middle_edge_ids: &[HashId],
) -> PyResult<Path> {
    let path_blocks = path.coordinate_blocks(conn, None);
    let edges = Path::edges_for_path(conn, &path.id, None);
    let edges_by_target: HashMap<(HashId, i64), &Edge> = edges
        .iter()
        .map(|edge| ((edge.target_node_id, edge.target_coordinate), edge))
        .collect();
    let edges_by_source: HashMap<(HashId, i64), &Edge> = edges
        .iter()
        .map(|edge| ((edge.source_node_id, edge.source_coordinate), edge))
        .collect();

    let mut new_edge_ids = Vec::new();
    if !is_start_node(span.start.graph_node.node_id) {
        let block = covering_block_for_entry(&path_blocks, &span.start)
            .expect("should find span.start on the current path");
        let entry_edge = edges_by_target
            .get(&(block.node_id, block.sequence_start))
            .expect("should have an edge entering every path block");
        for edge in &edges {
            new_edge_ids.push(edge.id);
            if edge.id == entry_edge.id {
                break;
            }
        }
    }
    new_edge_ids.extend_from_slice(middle_edge_ids);
    if !is_end_node(span.end.graph_node.node_id) {
        let block = covering_block_for_exit(&path_blocks, &span.end)
            .expect("should find span.end on the current path");
        let exit_edge = edges_by_source
            .get(&(block.node_id, block.sequence_end))
            .expect("should have an edge leaving every path block");
        let mut past_exit_anchor = false;
        for edge in &edges {
            if edge.id == exit_edge.id {
                past_exit_anchor = true;
            }
            if past_exit_anchor {
                new_edge_ids.push(edge.id);
            }
        }
    }
    let new_name = format!(
        "{}-edit-{}-{}",
        path.name,
        describe_position(&span.start),
        describe_position(&span.end)
    );
    Path::create(conn, &new_name, &path.block_group_id, &new_edge_ids)
        .map_err(|error| block_group_err_to_pyerr(error.into()))
}

fn describe_position(position: &GraphNodePosition) -> String {
    format!("{}:{}", position.graph_node.node_id, position.coordinate())
}

/// The chromosome and phase an edit inherits from the route entering its first base.
///
/// Only a target starting at its block's first base has an entering edge to read; inside a
/// block the reference chromosome applies.
fn entry_chromosome(graph: &GenGraph, first: &GraphNodeSlice) -> PyResult<(i64, i64)> {
    if first.start != 0 {
        return Ok((0, 0));
    }
    let mut chromosomes = graph
        .edges_directed(first.block, Incoming)
        .flat_map(|(_, _, edges)| edges.iter())
        .map(|edge| (edge.chromosome_index, edge.phased))
        .collect::<Vec<_>>();
    chromosomes.sort_unstable();
    chromosomes.dedup();
    match chromosomes.as_slice() {
        [] => Ok((0, 0)),
        [chromosome] => Ok(*chromosome),
        _ => Err(PyValueError::new_err(
            "target boundary is ambiguous across chromosomes",
        )),
    }
}

fn boundary(graph: &GenGraph, slice: &GraphNodeSlice, start: bool) -> PyResult<GraphNodePosition> {
    let offset = if start { slice.start } else { slice.end } as i64;
    if (start && offset > 0) || (!start && offset < slice.block.length()) {
        return Ok(GraphNodePosition {
            graph_node: slice.block,
            offset,
        });
    }
    // A point at a block edge borrows its neighbour's address; with several neighbours it is
    // refused rather than attached to one route arbitrarily.
    let neighbors = graph
        .neighbors_directed(slice.block, if start { Incoming } else { Outgoing })
        .collect::<Vec<_>>();
    let neighbor = match (neighbors.as_slice(), start) {
        ([neighbor], _) => neighbor,
        ([], _) => {
            return Err(PyValueError::new_err(
                "insertion point has no route on its other side to attach to",
            ));
        }
        (_, true) => {
            return Err(PyValueError::new_err(
                "insertion point is a join where several routes arrive; insert at the end of \
                 one of the arriving parts instead",
            ));
        }
        (_, false) => {
            return Err(PyValueError::new_err(
                "insertion point is a fork where several routes leave; insert at the start of \
                 one of the leaving parts instead",
            ));
        }
    };
    let (source, target) = if start {
        (*neighbor, slice.block)
    } else {
        (slice.block, *neighbor)
    };
    if !forward_edge(graph, source, target) {
        return Err(PyValueError::new_err(
            "Target boundary cannot be applied to the current graph orientation",
        ));
    }
    Ok(GraphNodePosition {
        graph_node: *neighbor,
        offset: if start { neighbor.length() } else { 0 },
    })
}

/// Stores the new bases as a node. Reverse-strand targets are written reverse
/// complemented so that reading the edit site on the target's strand yields `sequence`.
fn inserted_block(
    conn: &GraphConnection,
    destination: &BlockGroup,
    span: &EditSpan,
    sequence: &str,
) -> PyResult<PathBlock> {
    let stored = if span.is_reverse {
        String::from_utf8(reverse_complement(sequence.as_bytes()))
            .expect("should reverse complement to valid UTF-8")
    } else {
        sequence.to_string()
    };
    let saved = Sequence::new()
        .sequence_type("DNA")
        .sequence(&stored)
        .save(conn)
        .map_err(|err| PyRuntimeError::new_err(format!("cannot store sequence: {err}")))?;
    // Include HEAD so repeated insert/delete cycles at one site get distinct nodes.
    let head = hash_of(conn, "HEAD").map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let node_key = format!(
        "python-edit:{head}:{}:{}:{}-{}:{}->{}",
        destination.id,
        span.start.graph_node.node_id,
        span.start.coordinate(),
        span.end.graph_node.node_id,
        span.end.coordinate(),
        saved.hash
    );
    let node_id = Node::create(conn, &saved.hash, &HashId::convert_str(&node_key))
        .map_err(|err| PyRuntimeError::new_err(format!("cannot create node: {err}")))?;
    Ok(PathBlock {
        node_id,
        block_sequence: stored,
        sequence_start: 0,
        sequence_end: saved.length,
        path_start: 0,
        path_end: 0,
        strand: Strand::Forward,
    })
}

/// A region whose edit sites are already resolved to graph positions, for `plan_edges`.
fn anchored_region(block_group: BlockGroup, span: &EditSpan) -> ResolvedGenRegion {
    ResolvedGenRegion {
        block_group,
        path: None,
        accession: None,
        annotation: None,
        kind: ResolvedRegionKind::Accession,
        anchor_start: 0,
        anchor_end: 0,
        feature_length: 0,
        start: span.start.coordinate(),
        end: span.end.coordinate(),
        start_anchors: Some(vec![span.start]),
        end_anchors: Some(vec![span.end]),
        remove_ambiguous_positions: false,
    }
}

fn describe_range(range: &GraphNode) -> String {
    format!(
        "{}:{}-{}",
        range.node_id, range.sequence_start, range.sequence_end
    )
}

fn describe_locus(locus: &GraphLocus) -> String {
    locus
        .slices
        .iter()
        .map(|slice| describe_range(&slice.block))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, ffi::CString};

    use r#gen::test_helpers::setup_gen_on_disk;
    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::{GenGraph, GraphEdge, GraphNode, GraphNodePosition, GraphNodeSlice};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::{DbContext, GraphConnection},
        edge::Edge,
        locus::GraphLocus,
        node::Node,
        path::Path,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use pyo3::{
        Py, PyErr, PyRef, Python, prepare_freethreaded_python,
        types::{PyDict, PyDictMethods as _},
    };

    use crate::python_api::{
        block_group::PySequenceGraph,
        editing::{EditSpan, locate_span, locate_span_on_current_path, locus_from_region},
        graph_search::PositionSide,
        locus::GraphLocusExt as _,
        repository::PyRepository,
    };

    fn run_edit_test(script: &str, expected: &str) {
        prepare_freethreaded_python();
        Python::with_gil(|python| {
            let context = setup_gen_on_disk();
            let repository = Py::new(
                python,
                PyRepository {
                    context: context.clone(),
                },
            )
            .expect("should wrap repository");
            let sample = repository
                .call_method1(
                    python,
                    "import_genbank",
                    (
                        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/editing.gbk"),
                        "parent",
                        "collection",
                    ),
                )
                .expect("should import annotated fixture");
            let sequence_graph = sample
                .call_method1(python, "__getitem__", (0,))
                .expect("should find sequence graph");
            let locals = PyDict::new(python);
            locals
                .set_item("repo", repository)
                .expect("should bind repository");
            locals
                .set_item("sample", sample)
                .expect("should bind sample");
            locals
                .set_item("graph", &sequence_graph)
                .expect("should bind graph");
            python
                .run(
                    &CString::new(script).expect("should encode script"),
                    Some(&locals),
                    None,
                )
                .unwrap_or_else(|error| {
                    error.print(python);
                    panic!("Python edit assertions failed: {error}")
                });
            let graph = sequence_graph
                .extract::<PyRef<PySequenceGraph>>(python)
                .expect("should extract graph");
            assert_sequence(&context, &graph, expected);
        });
    }

    fn assert_sequence(context: &DbContext, graph: &PySequenceGraph, expected: &str) {
        let sequences = BlockGroup::get_all_sequences(
            context.graph().conn(),
            context.workspace(),
            &graph.id,
            true,
        )
        .expect("should read edited sequences");
        assert_eq!(sequences, HashSet::from([expected.to_string()]));
    }

    #[test]
    fn test_edit_database_annotation_locus() {
        run_edit_test(
            r#"
annotations = graph.list_annotations()
assert len(annotations) == 3
for annotation in annotations:
    assert annotation.locus is not None
    assert len(annotation.locus) == len(annotation) == 3
    for segment, part in zip(annotation.segments, annotation.locus.slices):
        assert len(graph.get_node_sequence(part.node)) == segment['end'] - segment['start']
        assert part.strand == segment['strand']
        assert part.start == 0 and part.end == 3
assert any(annotation.metadata for annotation in annotations)
"#,
            "AAACCCGGGTTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_stable_annotation_locus_after_unrelated_edit() {
        run_edit_test(
            r#"
annotation = next(item for item in graph.list_annotations() if item.name == 'second')
before = annotation.segments
locus = annotation.locus
graph.insert('edits:1', 'AG')
assert annotation.segments == before
assert next(item for item in graph.list_annotations() if item.id == annotation.id).segments == before
graph.delete(locus)
"#,
            "AAGAACCCGGGAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_sequential_annotation_relative_deletions() {
        run_edit_test(
            r#"
for annotation in graph.list_annotations():
    graph.delete(annotation.locus)
"#,
            "AAAGGGAAAGGGTTT",
        );
    }

    #[test]
    fn test_edit_replacement_and_insertion() {
        run_edit_test(
            r#"
target = graph.search('CCCGGG', sequence_kind='exact')[0]
inserted = graph.replace(target, 'ATGC')
assert len(inserted) == 4
position = inserted.slice(2, 2)
middle = graph.insert(position, 'TT')
assert len(middle) == 2
graph.delete(middle)
graph.insert(inserted.end(), 'C')
"#,
            "AAAATGCCTTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_reverse_strand_targets() {
        run_edit_test(
            r#"
annotation = next(item for item in graph.list_annotations() if item.name == 'reverse')
assert annotation.locus.strand == '-'
inserted = graph.replace(annotation, 'AGT')
assert inserted.strand == '-'
assert inserted.start().offset == 3
assert inserted.end().offset == 0
graph.delete(inserted.slice(0, 1))
"#,
            "AAACCCGGGTTTAAAACGGGTTT",
        );
    }

    #[test]
    fn test_edit_missing_targets_and_failed_child_roll_back() {
        run_edit_test(
            r#"
annotation = next(item for item in graph.list_annotations() if item.name == 'first')
graph.delete(annotation)
count = len(repo.get_operations())
try:
    graph.delete(annotation)
except ValueError as error:
    assert 'not present' in str(error)
else:
    assert False, 'deleted target must fail'
assert len(repo.get_operations()) == count
"#,
            "AAAGGGTTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_one_operation_per_edit() {
        run_edit_test(
            r#"
count = len(repo.get_operations())
graph.delete('edits:3-6')
assert len(repo.get_operations()) == count + 1
inserted = graph.replace('edits:9-12', 'GA')
assert len(repo.get_operations()) == count + 2
graph.insert(inserted.start(), 'C')
assert len(repo.get_operations()) == count + 3
"#,
            // 'edits:9-12' resolves against the path after the first delete.
            "AAAGGGTTTCGACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_child_sample() {
        run_edit_test(
            r#"
annotation = next(item for item in graph.list_annotations() if item.name == 'first')
count = len(repo.get_operations())
child = sample.copy('child')
inserted = child[0].replace(annotation, 'ATGC')
assert len(inserted) == 4
assert len(repo.get_operations()) == count + 2
child[0].delete(inserted)
assert len(repo.get_operations()) == count + 3
assert child[0].search('AAAGGGTTT', sequence_kind='exact')
"#,
            "AAACCCGGGTTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_adjacent_deletions_and_terminal_insertions() {
        run_edit_test(
            r#"
whole = graph.search('AAACCCGGGTTTAAACCCGGGTTT', sequence_kind='exact')[0]
graph.delete(whole.slice(3, 6))
graph.delete(whole.slice(6, 9))
graph.delete(whole.slice(0, 3))
graph.delete(whole.slice(21, 24))
graph.insert(whole.slice(9, 9), 'AG')
graph.insert(whole.slice(21, 21), 'TC')
"#,
            "AGTTTAAACCCGGGTC",
        );
    }

    #[test]
    fn test_edit_deleted_interior_and_invalid_targets_fail() {
        run_edit_test(
            r#"
whole = graph.search('AAACCCGGGTTTAAACCCGGGTTT', sequence_kind='exact')[0]
graph.delete(whole.slice(3, 6))
count = len(repo.get_operations())
for target in [whole.slice(2, 7), whole.slice(4, 5), 'edits:50-60', 123]:
    try:
        graph.delete(target)
    except (ValueError, TypeError):
        pass
    else:
        assert False, 'invalid target must fail'
assert len(repo.get_operations()) == count
"#,
            "AAAGGGTTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_repeated_insertion_and_deletion() {
        run_edit_test(
            r#"
count = len(repo.get_operations())
for _ in range(3):
    inserted = graph.insert('edits:3', 'AG')
    graph.delete(inserted)
assert len(repo.get_operations()) == count + 6
"#,
            "AAACCCGGGTTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_reverse_span_across_nodes() {
        run_edit_test(
            r#"
graph.insert('edits:3', 'AG')
target = graph.search('AAGCC', sequence_kind='exact')[0]
assert len(target.slices) == 3
inserted = graph.replace(target.reverse_complement(), 'TCA')
assert len(inserted) == 3
"#,
            "AATGACGGGTTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_entire_graph_and_inserted_locus() {
        run_edit_test(
            r#"
inserted = graph.replace('edits', 'AGTC')
graph.delete(inserted)
"#,
            "",
        );
    }

    #[test]
    fn test_edit_annotation_region_uses_resolved_offsets() {
        run_edit_test(
            r#"
graph.delete('second:1-2')
"#,
            "AAACCCGGGTTAAACCCGGGTTT",
        );
    }

    #[test]
    fn test_edit_annotation_region_does_not_clip_missing_bases() {
        run_edit_test(
            r#"
count = len(repo.get_operations())
try:
    graph.delete('second:0-100')
except ValueError:
    pass
else:
    assert False, 'out-of-bounds annotation region must fail'
assert len(repo.get_operations()) == count
"#,
            "AAACCCGGGTTTAAACCCGGGTTT",
        );
    }

    fn graph_node(name: &str, start: i64, end: i64) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(name),
            sequence_start: start,
            sequence_end: end,
        }
    }

    fn forward_edge() -> GraphEdge {
        GraphEdge {
            edge_id: HashId::convert_str("edge"),
            source_strand: Strand::Forward,
            target_strand: Strand::Forward,
            chromosome_index: 0,
            phased: 0,
            created_on: 0,
        }
    }

    #[test]
    fn test_edit_rejects_disconnected_blocks_with_contiguous_node_coordinates() {
        let mut graph = GenGraph::new();
        graph.add_node(graph_node("sequence", 0, 5));
        graph.add_node(graph_node("sequence", 5, 10));
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice::full(
                graph_node("sequence", 2, 8),
                Strand::Forward,
            )],
        };
        assert!(
            locate_span(&graph, &locus, PositionSide::Preceding).is_err(),
            "coverage alone must not establish connectivity"
        );
    }

    #[test]
    fn test_edit_rejects_ambiguous_boundary_for_an_insertion() {
        let mut graph = GenGraph::new();
        let target = graph_node("target", 0, 10);
        graph.add_edge(graph_node("left", 0, 10), target, vec![forward_edge()]);
        graph.add_edge(graph_node("other", 0, 10), target, vec![forward_edge()]);
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice::full(
                graph_node("target", 0, 0),
                Strand::Forward,
            )],
        };
        assert!(
            locate_span(&graph, &locus, PositionSide::Preceding).is_err(),
            "an insertion at a block edge must not pick one flanking route arbitrarily"
        );
    }

    #[test]
    fn test_point_at_a_fork_resolves_to_the_block_on_its_side() {
        // The end of the forking block is ambiguous; the start of the block after it is not.
        let mut graph = GenGraph::new();
        let before = graph_node("sequence", 0, 5);
        let after = graph_node("sequence", 5, 10);
        graph.add_edge(before, after, vec![forward_edge()]);
        graph.add_edge(
            before,
            graph_node("alternative", 0, 5),
            vec![forward_edge()],
        );
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice {
                block: graph_node("sequence", 0, 10),
                start: 5,
                end: 5,
                strand: Strand::Forward,
            }],
        };

        assert!(
            locate_span(&graph, &locus, PositionSide::Preceding).is_err(),
            "the end of a forking block must not pick one of its routes"
        );
        let span = locate_span(&graph, &locus, PositionSide::Following)
            .expect("should locate the start of the route after the fork");
        assert_eq!(span.start.graph_node, before);
        assert_eq!(span.end.graph_node, after);
        assert_eq!(span.end.coordinate(), 5);
    }

    #[test]
    fn test_edit_anchors_a_whole_block_span_on_its_own_block() {
        // The shape of a library column, whose alternatives share their flanking edges.
        let mut graph = GenGraph::new();
        let target = graph_node("target", 0, 10);
        let left = graph_node("left", 0, 10);
        graph.add_edge(left, target, vec![forward_edge()]);
        graph.add_edge(left, graph_node("alternative", 0, 10), vec![forward_edge()]);
        graph.add_edge(target, graph_node("right", 0, 10), vec![forward_edge()]);
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice::full(target, Strand::Forward)],
        };

        let span = locate_span(&graph, &locus, PositionSide::Preceding)
            .expect("should locate a whole-block span");

        assert_eq!(span.start.graph_node, target);
        assert_eq!(span.start.coordinate(), 0);
        assert_eq!(span.end.graph_node, target);
        assert_eq!(span.end.coordinate(), 10);
    }

    #[test]
    fn test_edit_inherits_the_chromosome_of_the_route_it_lands_on() {
        let mut graph = GenGraph::new();
        let target = graph_node("target", 0, 10);
        graph.add_edge(
            graph_node("left", 0, 10),
            target,
            vec![GraphEdge {
                chromosome_index: 2,
                phased: 1,
                ..forward_edge()
            }],
        );
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice::full(target, Strand::Forward)],
        };

        let span = locate_span(&graph, &locus, PositionSide::Preceding)
            .expect("should locate a whole-block span");

        assert_eq!(span.chromosome_index, 2);
        assert_eq!(span.phased, 1);
    }

    #[test]
    fn test_edit_rejects_reverse_graph_edge_orientation() {
        let mut graph = GenGraph::new();
        graph.add_edge(
            graph_node("left", 0, 10),
            graph_node("target", 0, 10),
            vec![GraphEdge {
                target_strand: Strand::Reverse,
                ..forward_edge()
            }],
        );
        let locus = GraphLocus {
            slices: vec![GraphNodeSlice::full(
                graph_node("target", 2, 5),
                Strand::Forward,
            )],
        };
        assert!(
            locate_span(&graph, &locus, PositionSide::Preceding).is_err(),
            "unsupported graph orientation must fail before mutation"
        );
    }

    fn create_sequence_node(conn: &GraphConnection, sequence: &str) -> HashId {
        let saved = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn)
            .expect("should save sequence");
        Node::create(conn, &saved.hash, &HashId::convert_str(sequence)).expect("should create node")
    }

    fn full_node_position(node_id: HashId, length: i64, offset: i64) -> GraphNodePosition {
        GraphNodePosition {
            graph_node: GraphNode {
                node_id,
                sequence_start: 0,
                sequence_end: length,
            },
            offset,
        }
    }

    #[test]
    fn test_locate_span_on_current_path_distinguishes_bubble_from_current_route() {
        // `route` is on the current path; the parallel `bubble` is not.
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        Collection::create(conn, "collection").expect("should create collection");
        Sample::get_or_create(
            conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "collection",
                sample_name: "sample",
                name: "chr1",
                parent_block_group_id: None,
                is_default: true,
            },
        )
        .expect("should create block group");
        let route = create_sequence_node(conn, "AAAA");
        let bubble = create_sequence_node(conn, "CCCC");

        let start_to_route = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            route,
            0,
            Strand::Forward,
        )
        .expect("should create edge");
        let route_to_end = Edge::create(
            conn,
            route,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create edge");
        Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            bubble,
            0,
            Strand::Forward,
        )
        .expect("should create edge");
        Edge::create(
            conn,
            bubble,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create edge");

        Path::create_with_validated_edges(
            conn,
            "chr1",
            &block_group.id,
            &[start_to_route.id, route_to_end.id],
        )
        .expect("should create current path");

        let span_on_route = EditSpan {
            start: full_node_position(route, 4, 0),
            end: full_node_position(route, 4, 4),
            is_reverse: false,
            chromosome_index: 0,
            phased: 0,
        };
        let span_on_bubble = EditSpan {
            start: full_node_position(bubble, 4, 0),
            end: full_node_position(bubble, 4, 4),
            is_reverse: false,
            chromosome_index: 0,
            phased: 0,
        };

        assert!(
            locate_span_on_current_path(conn, &block_group.id, &span_on_route)
                .expect("should check path membership")
                .is_some(),
            "a span spanning the current path's own route must be found on it"
        );
        assert!(
            locate_span_on_current_path(conn, &block_group.id, &span_on_bubble)
                .expect("should check path membership")
                .is_none(),
            "a span on the bubble route must not be mistaken for the current path"
        );
    }

    /// `left` (AAAA) forks into `right_one` (CCCC) and `right_two` (GGGG), each on its own
    /// chromosome so pruning keeps both routes. `right_one` continues into `tail` (TTTT),
    /// which nothing else enters. The current path reads left, right_one, tail.
    fn setup_fork(context: &DbContext) {
        let conn = context.graph().conn();
        Collection::create(conn, "collection").expect("should create collection");
        Sample::get_or_create(
            conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "collection",
                sample_name: "sample",
                name: "chr1",
                parent_block_group_id: None,
                is_default: true,
            },
        )
        .expect("should create block group");
        let left = create_sequence_node(conn, "AAAA");
        let right_one = create_sequence_node(conn, "CCCC");
        let right_two = create_sequence_node(conn, "GGGG");
        let tail = create_sequence_node(conn, "TTTT");
        let create_edge = |source, source_coordinate, target, target_coordinate| {
            Edge::create(
                conn,
                source,
                source_coordinate,
                Strand::Forward,
                target,
                target_coordinate,
                Strand::Forward,
            )
            .expect("should create edge")
        };
        let path_edges = [
            create_edge(PATH_START_NODE_ID, 0, left, 0),
            create_edge(left, 4, right_one, 0),
            create_edge(right_one, 4, tail, 0),
            create_edge(tail, 4, PATH_END_NODE_ID, 0),
        ];
        let branch_edges = [
            create_edge(left, 4, right_two, 0),
            create_edge(right_two, 4, PATH_END_NODE_ID, 0),
        ];
        let block_group_edges = path_edges
            .iter()
            .map(|edge| (edge, 0))
            .chain(branch_edges.iter().map(|edge| (edge, 1)))
            .map(|(edge, chromosome_index)| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge.id,
                chromosome_index,
                phased: 0,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);
        Path::create_with_validated_edges(
            conn,
            "chr1",
            &block_group.id,
            &path_edges.iter().map(|edge| edge.id).collect::<Vec<_>>(),
        )
        .expect("should create current path");
    }

    fn error_message(error: PyErr) -> String {
        prepare_freethreaded_python();
        Python::with_gil(|python| error.value(python).to_string())
    }

    #[test]
    fn test_region_point_at_a_fork_is_rejected() {
        let context = setup_gen_on_disk();
        setup_fork(&context);

        let Err(error) = locus_from_region(&context, "chr1:4-4", "collection", "sample") else {
            panic!("a coordinate at a fork must not silently pick the path's route");
        };

        assert!(error_message(error).contains("forks or joins"));
    }

    #[test]
    fn test_region_point_between_single_routes_is_accepted() {
        let context = setup_gen_on_disk();
        setup_fork(&context);

        let target = locus_from_region(&context, "chr1:8-8", "collection", "sample")
            .unwrap_or_else(|error| panic!("{}", error_message(error)));

        assert_eq!(target.locus.length(), 0);
    }

    #[test]
    fn test_region_point_inside_a_block_is_accepted() {
        let context = setup_gen_on_disk();
        setup_fork(&context);

        let target = locus_from_region(&context, "chr1:2-2", "collection", "sample")
            .unwrap_or_else(|error| panic!("{}", error_message(error)));

        assert_eq!(target.locus.length(), 0);
    }
}
