//! Edits to a sequence graph made through the Python API, each recorded as one operation.
//!
//! Insertions are addressed by `SuperPosition`s and attach to every position a superposition
//! covers. The inserted sequence becomes one node wired between a set of left anchors and a set of
//! right anchors: an edge from every left anchor into the node's first position, and one from its
//! last position to every right anchor. An insertion given only `after` or only `before` finds the
//! missing side by stepping each position once, so a fork contributes an anchor per route. One given
//! both takes only the connections that run directly from an `after` position into a `before`
//! position, which picks out one route at a fork or join. Unless the insertion is stacked it retires
//! each connection it lands on, by demoting an edge or splitting a block with an edit-site marker,
//! because the connections replaced can sit on different chromosome indexes or on the never-pruned
//! one.
//!
//! Replacements and deletions are addressed by a region string, `Locus`, or `Annotation`. The
//! target is canonicalized to node-absolute ranges, checked against the current graph, and applied
//! by reconnecting every route at its boundaries and retiring the bypassed edges unless stacked.
//!
//! Both kinds store a new sequence as a node keyed by where it attaches, and splice the current path
//! when the edit lies on it.

use std::collections::HashSet;

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, PathBlock, Strand,
    is_end_node, is_start_node, is_terminal,
};
use gen_graph::{GenGraph, GraphNode, GraphNodePosition, GraphNodeSlice};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange, BlockGroupError},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::{DbContext, GraphConnection},
    edge::{Edge, EdgeData},
    errors::{OperationError, QueryError},
    locus::GraphLocus,
    node::Node,
    operations::{OperationInfo, OperationSummary},
    path::Path,
    region::{ResolvedGenRegion, ResolvedRegionKind},
    sequence::{Sequence, reverse_complement},
};
use petgraph::Direction::{Incoming, Outgoing};
use pyo3::{
    Bound, PyAny, PyErr, PyRef, PyResult,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    types::PyAnyMethods as _,
};

use super::{
    annotation::PyAnnotation,
    block_group::PySequenceGraph,
    graph_read::{LocusTarget, current_graph, forward_edge, locus_from_region},
    graph_search::PyGraphLocus,
    locus::GraphLocusExt as _,
    position::{Neighbor, Position, neighbors, reaches, require_blocks},
    repository::run_context_operation_write,
    utils::block_group_err_to_pyerr,
};

/// One end of an edit's new edges: a node coordinate, and the block of the destination graph that
/// coordinate falls in or bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Anchor {
    block: GraphNode,
    coordinate: i64,
}

impl Anchor {
    /// The point just after a position in node order, where the inserted node's entering edge leaves.
    fn left(graph: &GenGraph, neighbor: Neighbor) -> PyResult<Self> {
        Ok(match neighbor {
            Neighbor::Position(position) => Anchor {
                block: require_blocks(graph, &[position])?[0],
                coordinate: position.coordinate + 1,
            },
            Neighbor::Terminal(block) => Anchor {
                block,
                coordinate: block.sequence_end,
            },
        })
    }

    /// The point just before a position in node order, where the inserted node's exiting edge arrives.
    fn right(graph: &GenGraph, neighbor: Neighbor) -> PyResult<Self> {
        Ok(match neighbor {
            Neighbor::Position(position) => Anchor {
                block: require_blocks(graph, &[position])?[0],
                coordinate: position.coordinate,
            },
            Neighbor::Terminal(block) => Anchor {
                block,
                coordinate: block.sequence_start,
            },
        })
    }

    /// Whether an edge at this coordinate cuts its block in two.
    fn splits_block(&self) -> bool {
        self.block.sequence_start < self.coordinate && self.coordinate < self.block.sequence_end
    }

    fn describe(&self) -> String {
        format!("{}:{}", self.block.node_id, self.coordinate)
    }
}

impl From<GraphNodePosition> for Anchor {
    fn from(position: GraphNodePosition) -> Self {
        Self {
            block: position.graph_node,
            coordinate: position.coordinate(),
        }
    }
}

/// Where an insertion lands, for its operation summary: one point when its left and right anchors
/// name the same node coordinates, as between two positions of one node, else both sides.
fn describe_site(lefts: &[Anchor], rights: &[Anchor]) -> String {
    let (left, right) = (describe_anchors(lefts), describe_anchors(rights));
    if left == right {
        left
    } else {
        format!("{left} -> {right}")
    }
}

/// A sequence for an operation summary, shortened when long.
fn describe_sequence(sequence: &str) -> String {
    const SHOWN: usize = 20;
    let length = sequence.chars().count();
    if length <= SHOWN {
        sequence.to_string()
    } else {
        let shown = sequence.chars().take(SHOWN).collect::<String>();
        format!("{shown}... ({length} long)")
    }
}

fn describe_anchors(anchors: &[Anchor]) -> String {
    anchors
        .iter()
        .map(Anchor::describe)
        .collect::<Vec<_>>()
        .join(",")
}

/// Where an insertion goes relative to the positions it is given.
pub(crate) enum InsertSite<'a> {
    Before(&'a [Position]),
    After(&'a [Position]),
    /// On the connections from the `after` positions directly into the `before` positions.
    Junction {
        after: &'a [Position],
        before: &'a [Position],
    },
}

/// What an insertion retires.
enum Supersede {
    /// Nothing: the insertion is added alongside the existing routes.
    Stacked,
    /// The connection between each left and right anchor pair, which are neighbours.
    Adjacent(Vec<(Anchor, Anchor)>),
}

/// The edge rows an insertion writes and retires, decided against the graph before any change.
struct InsertPlan {
    /// Live rows of the edges that carried a superseded route out of a left or into a right anchor.
    retired: Vec<BlockGroupEdgeData>,
    /// Anchors inside a block, each split by a same-coordinate edge; `true` retires the route
    /// through the split.
    splits: Vec<(Anchor, bool)>,
    /// The chromosome index and phase of each edge into the inserted node, per left anchor.
    left_indexes: Vec<(Anchor, Vec<(i64, i64)>)>,
    /// The chromosome index and phase of each edge out of the inserted node, per right anchor.
    right_indexes: Vec<(Anchor, Vec<(i64, i64)>)>,
}

/// Inserts `sequence` at `site` in `sequence_graph` as one operation and returns its locus, read on
/// the positions' strand.
pub(crate) fn insert_at_positions(
    sequence_graph: &PySequenceGraph,
    site: &InsertSite<'_>,
    sequence: &str,
    message: Option<&str>,
    stack: bool,
) -> PyResult<GraphLocus> {
    if sequence.is_empty() {
        return Err(PyValueError::new_err("insert needs a non-empty sequence"));
    }
    let context = sequence_graph.require_context("insert")?;
    run_context_operation_write(
        context,
        |context| {
            let conn = context.graph().conn();
            let block_group = BlockGroup::get_by_id(conn, &sequence_graph.id, None)
                .map_err(block_group_err_to_pyerr)?;
            let graph = current_graph(context, &block_group.id)?;
            let sides = reading_sides(&graph, site)?;
            let is_reverse = reads_reverse(sides.before.iter().chain(&sides.after))?;
            // Edges run in node order, so a reverse-strand insertion is written reverse
            // complemented with what reads after it on the left.
            let orient = |before: Neighbor, after: Neighbor| {
                if is_reverse {
                    (after, before)
                } else {
                    (before, after)
                }
            };
            let (left_side, right_side) = if is_reverse {
                (sides.after, sides.before)
            } else {
                (sides.before, sides.after)
            };
            let lefts = anchors(&graph, left_side, Anchor::left)?;
            let rights = anchors(&graph, right_side, Anchor::right)?;
            refuse_junctions(&graph, lefts.iter().chain(&rights))?;
            refuse_loops(&graph, &lefts, &rights)?;

            let supersede = if stack {
                Supersede::Stacked
            } else {
                Supersede::Adjacent(
                    sides
                        .neighbours
                        .into_iter()
                        .map(|(before, after)| {
                            let (left, right) = orient(before, after);
                            Ok((Anchor::left(&graph, left)?, Anchor::right(&graph, right)?))
                        })
                        .collect::<PyResult<_>>()?,
                )
            };
            let plan = plan_insert(&graph, &block_group.id, &lefts, &rights, &supersede)?;
            let path_splice = if stack {
                None
            } else {
                current_path_pair(conn, &block_group.id, &lefts, &rights)?
            };
            let stored = stored_sequence(sequence, is_reverse);
            let (node_id, length) = inserted_node(conn, &block_group.id, &lefts, &rights, &stored)?;
            write_insert(conn, &block_group.id, plan, node_id, length)?;
            if let Some(splice) = path_splice {
                let middle_edge_ids = [
                    entry_edge(&splice.left, node_id).id_hash(),
                    exit_edge(&splice.right, node_id, length).id_hash(),
                ];
                splice_current_path(conn, &splice, &middle_edge_ids)?;
            }

            let locus = GraphLocus {
                slices: vec![GraphNodeSlice {
                    block: GraphNode {
                        node_id,
                        sequence_start: 0,
                        sequence_end: length,
                    },
                    start: 0,
                    end: length as usize,
                    strand: if is_reverse {
                        Strand::Reverse
                    } else {
                        Strand::Forward
                    },
                }],
            };
            let summary = edit_summary(message, || {
                format!(
                    "{}: insert {} at {} in sample '{}'",
                    block_group.name,
                    describe_sequence(sequence),
                    describe_site(&lefts, &rights),
                    block_group.sample_name
                )
            });
            Ok((locus, summary))
        },
        edit_error,
    )
}

/// What reads immediately before and after an insertion, in reading order.
struct ReadingSides {
    before: Vec<Neighbor>,
    after: Vec<Neighbor>,
    /// Each connection the insertion lands on, as the (before, after) neighbours it runs between.
    neighbours: Vec<(Neighbor, Neighbor)>,
}

fn reading_sides(graph: &GenGraph, site: &InsertSite<'_>) -> PyResult<ReadingSides> {
    let stepped = |positions: &[Position], forward: bool| -> PyResult<Vec<(Neighbor, Neighbor)>> {
        let mut pairs = vec![];
        for position in positions {
            for neighbor in neighbors(graph, position, forward)? {
                pairs.push(if forward {
                    (Neighbor::Position(*position), neighbor)
                } else {
                    (neighbor, Neighbor::Position(*position))
                });
            }
        }
        Ok(pairs)
    };
    let unique = |side: Vec<Neighbor>| {
        let mut unique = vec![];
        for neighbor in side {
            if !unique.contains(&neighbor) {
                unique.push(neighbor);
            }
        }
        unique
    };
    let pairs = match site {
        InsertSite::After(positions) => stepped(positions, true)?,
        InsertSite::Before(positions) => stepped(positions, false)?,
        InsertSite::Junction { after, before } => {
            // Positions on different strands never read into each other, so name the real problem.
            let given = after
                .iter()
                .chain(before.iter())
                .map(|position| Neighbor::Position(*position))
                .collect::<Vec<_>>();
            reads_reverse(given.iter())?;
            require_blocks(graph, before)?;
            let pairs = stepped(after, true)?
                .into_iter()
                .filter(|(_, next)| match next {
                    Neighbor::Position(position) => before.contains(position),
                    Neighbor::Terminal(_) => false,
                })
                .collect::<Vec<_>>();
            let unflanked_after = after.iter().find(|position| {
                !pairs
                    .iter()
                    .any(|(previous, _)| *previous == Neighbor::Position(**position))
            });
            let unflanked_before = before.iter().find(|position| {
                !pairs
                    .iter()
                    .any(|(_, next)| *next == Neighbor::Position(**position))
            });
            let unflanked = match (unflanked_after, unflanked_before) {
                (Some(position), _) => Some(format!(
                    "{} does not read directly into a position of before",
                    position.describe()
                )),
                (None, Some(position)) => Some(format!(
                    "{} is not read directly after a position of after",
                    position.describe()
                )),
                (None, None) => None,
            };
            if let Some(reason) = unflanked {
                return Err(PyValueError::new_err(format!(
                    "insert with both after and before needs them to flank a junction: every \
                     position of after must read directly into a position of before, but \
                     {reason}; to replace the sequence between two positions, use replace()"
                )));
            }
            pairs
        }
    };
    let sides = ReadingSides {
        before: unique(pairs.iter().map(|(before, _)| *before).collect()),
        after: unique(pairs.iter().map(|(_, after)| *after).collect()),
        neighbours: pairs,
    };
    if sides.before.is_empty() || sides.after.is_empty() {
        return Err(PyValueError::new_err(
            "insertion has no route on one of its sides to attach to",
        ));
    }
    Ok(sides)
}

/// Whether the positions read on the reverse strand. Every position must agree, since one inserted
/// node is stored in a single orientation.
fn reads_reverse<'a>(neighbors: impl Iterator<Item = &'a Neighbor>) -> PyResult<bool> {
    let strands = neighbors
        .filter_map(|neighbor| match neighbor {
            Neighbor::Position(position) => Some(position.strand == Strand::Reverse),
            Neighbor::Terminal(_) => None,
        })
        .collect::<HashSet<_>>();
    if strands.len() > 1 {
        return Err(PyValueError::new_err(
            "positions mix strands; insert on each strand separately",
        ));
    }
    Ok(strands.contains(&true))
}

fn anchors(
    graph: &GenGraph,
    side: Vec<Neighbor>,
    anchor: fn(&GenGraph, Neighbor) -> PyResult<Anchor>,
) -> PyResult<Vec<Anchor>> {
    let mut anchors = side
        .into_iter()
        .map(|neighbor| anchor(graph, neighbor))
        .collect::<PyResult<Vec<_>>>()?;
    anchors.sort_unstable();
    anchors.dedup();
    Ok(anchors)
}

/// An edge at a zero-length junction attaches to the junction rather than to the position beside it,
/// which would join the insertion to every route through the junction.
fn refuse_junctions<'a>(
    graph: &GenGraph,
    mut anchors: impl Iterator<Item = &'a Anchor>,
) -> PyResult<()> {
    let at_junction = anchors.any(|anchor| {
        graph.nodes().any(|block| {
            block.node_id == anchor.block.node_id
                && !is_terminal(block.node_id)
                && block.sequence_start == anchor.coordinate
                && block.sequence_end == anchor.coordinate
        })
    });
    if at_junction {
        return Err(PyValueError::new_err(
            "insertion point is at a junction left by an earlier edit, which is not supported yet",
        ));
    }
    Ok(())
}

/// Refuses an insertion whose right side leads back to its left side, which would let the
/// inserted sequence be read again after itself.
fn refuse_loops(graph: &GenGraph, lefts: &[Anchor], rights: &[Anchor]) -> PyResult<()> {
    let loops = rights.iter().any(|right| {
        lefts.iter().any(|left| {
            !is_terminal(right.block.node_id)
                && !is_terminal(left.block.node_id)
                && ((right.block == left.block && right.coordinate < left.coordinate)
                    || reaches(graph, right.block, left.block))
        })
    });
    if loops {
        return Err(PyValueError::new_err(
            "insertion would loop: a position on its right side leads back to a position on its left side",
        ));
    }
    Ok(())
}

/// Decides which connections the insertion retires and which chromosome index each new edge
/// carries.
///
/// An insertion with more than one anchor on either side is wired like a library column, on the
/// index pruning never competes on, as a stacked one is. A single-anchor insertion takes the index
/// of the connection it supersedes, falling back to the never-pruned index where that would compete
/// with an edge that stays live out of the same block.
fn plan_insert(
    graph: &GenGraph,
    block_group_id: &HashId,
    lefts: &[Anchor],
    rights: &[Anchor],
    supersede: &Supersede,
) -> PyResult<InsertPlan> {
    let mut retired_edges = HashSet::new();
    let mut retired_splits = HashSet::new();
    match supersede {
        Supersede::Stacked => {}
        Supersede::Adjacent(pairs) => {
            for (left, right) in pairs {
                if left.splits_block() {
                    retired_splits.insert(*left);
                } else if graph.edge_weight(left.block, right.block).is_some() {
                    retired_edges.insert((left.block, right.block));
                } else {
                    return Err(PyValueError::new_err(
                        "insertion point is next to a junction left by an earlier edit, which is \
                         not supported yet",
                    ));
                }
            }
        }
    }

    let mut splits = lefts
        .iter()
        .chain(rights)
        .filter(|anchor| anchor.splits_block())
        .map(|anchor| (*anchor, retired_splits.contains(anchor)))
        .collect::<Vec<_>>();
    splits.sort_unstable();
    splits.dedup();

    let never_pruned = vec![(INDETERMINATE_CHROMOSOME_INDEX, 0)];
    let (left_indexes, right_indexes) =
        if matches!(supersede, Supersede::Stacked) || lefts.len() > 1 || rights.len() > 1 {
            (
                lefts
                    .iter()
                    .map(|left| (*left, never_pruned.clone()))
                    .collect(),
                rights
                    .iter()
                    .map(|right| (*right, never_pruned.clone()))
                    .collect(),
            )
        } else {
            let (left, right) = (lefts[0], rights[0]);
            let inherited = |rows: Vec<BlockGroupEdgeData>| {
                let indexes = dedup(
                    rows.into_iter()
                        .filter(|row| row.chromosome_index >= 0)
                        .map(|row| (row.chromosome_index, row.phased))
                        .collect(),
                );
                if indexes.is_empty() {
                    vec![(0, 0)]
                } else {
                    indexes
                }
            };
            let left_indexes = if left.splits_block() {
                // Only the marker leaves the first half of a split block, and it competes only
                // while it stays live.
                if retired_splits.contains(&left) {
                    vec![(0, 0)]
                } else {
                    never_pruned.clone()
                }
            } else {
                let (retired, kept): (Vec<_>, Vec<_>) = graph
                    .neighbors_directed(left.block, Outgoing)
                    .partition(|successor| retired_edges.contains(&(left.block, *successor)));
                let staying = kept
                    .into_iter()
                    .flat_map(|successor| live_rows(graph, block_group_id, left.block, successor))
                    .map(|row| row.chromosome_index)
                    .collect::<HashSet<_>>();
                dedup(
                    inherited(
                        retired
                            .into_iter()
                            .flat_map(|successor| {
                                live_rows(graph, block_group_id, left.block, successor)
                            })
                            .collect(),
                    )
                    .into_iter()
                    .map(|(index, phased)| {
                        if staying.contains(&index) {
                            (INDETERMINATE_CHROMOSOME_INDEX, 0)
                        } else {
                            (index, phased)
                        }
                    })
                    .collect(),
                )
            };
            let right_indexes = if right.splits_block() {
                vec![(0, 0)]
            } else {
                inherited(
                    graph
                        .neighbors_directed(right.block, Incoming)
                        .filter(|predecessor| retired_edges.contains(&(*predecessor, right.block)))
                        .flat_map(|predecessor| {
                            live_rows(graph, block_group_id, predecessor, right.block)
                        })
                        .collect(),
                )
            };
            (vec![(left, left_indexes)], vec![(right, right_indexes)])
        };

    Ok(InsertPlan {
        retired: retired_edges
            .into_iter()
            .flat_map(|(source, target)| live_rows(graph, block_group_id, source, target))
            .collect(),
        splits,
        left_indexes,
        right_indexes,
    })
}

fn dedup(mut indexes: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    indexes.sort_unstable();
    indexes.dedup();
    indexes
}

/// Writes the planned edges and demotes the retired rows to the edit-site marker index.
///
/// Rows already recorded for the new edges, left by an earlier insertion of the same sequence at the
/// same anchors or an earlier split at the same coordinate, are dropped and written again, so
/// pruning ranks them as this operation's.
fn write_insert(
    conn: &GraphConnection,
    block_group_id: &HashId,
    plan: InsertPlan,
    node_id: HashId,
    length: i64,
) -> PyResult<()> {
    let row = |edge: &EdgeData, (chromosome_index, phased): (i64, i64)| BlockGroupEdgeData {
        block_group_id: *block_group_id,
        edge_id: edge.id_hash(),
        chromosome_index,
        phased,
    };
    let mut edges = vec![];
    let mut rows = vec![];
    for (anchor, retired) in &plan.splits {
        let marker = EdgeData {
            source_node_id: anchor.block.node_id,
            source_coordinate: anchor.coordinate,
            source_strand: Strand::Forward,
            target_node_id: anchor.block.node_id,
            target_coordinate: anchor.coordinate,
            target_strand: Strand::Forward,
        };
        let index = if *retired {
            PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
        } else {
            0
        };
        rows.push(row(&marker, (index, 0)));
        edges.push(marker);
    }
    for (anchor, indexes) in &plan.left_indexes {
        let edge = entry_edge(anchor, node_id);
        rows.extend(indexes.iter().map(|index| row(&edge, *index)));
        edges.push(edge);
    }
    for (anchor, indexes) in &plan.right_indexes {
        let edge = exit_edge(anchor, node_id, length);
        rows.extend(indexes.iter().map(|index| row(&edge, *index)));
        edges.push(edge);
    }
    edges.sort_unstable();
    edges.dedup();

    let edge_ids = edges.iter().map(EdgeData::id_hash).collect::<HashSet<_>>();
    let stale = BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
        .into_iter()
        .filter(|augmented_edge| edge_ids.contains(&augmented_edge.edge.id))
        .map(|augmented_edge| BlockGroupEdgeData {
            block_group_id: *block_group_id,
            edge_id: augmented_edge.edge.id,
            chromosome_index: augmented_edge.chromosome_index,
            phased: augmented_edge.phased,
        });
    let retired = plan
        .retired
        .into_iter()
        .filter(|row| !edge_ids.contains(&row.edge_id))
        .collect::<Vec<_>>();
    BlockGroupEdge::select(conn)
        .delete_by_ids(
            stale
                .chain(retired.iter().cloned())
                .map(|row| row.id_hash())
                .collect::<Vec<_>>(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Edge::bulk_create(conn, &edges);
    rows.extend(retired.into_iter().map(|row| BlockGroupEdgeData {
        chromosome_index: PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
        phased: 0,
        ..row
    }));
    BlockGroupEdge::bulk_create(conn, &rows);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EditKind {
    Replace,
    Delete,
}

impl EditKind {
    const fn verb(self) -> &'static str {
        match self {
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

/// The resolved attachment points of an edit in the destination graph.
struct EditSpan {
    starts: Vec<GraphNodePosition>,
    ends: Vec<GraphNodePosition>,
    is_reverse: bool,
    chromosome_index: i64,
    phased: i64,
    /// Block edges a span anchored on its neighbours steps over, which a superseding edit retires.
    bypassed_edges: Vec<(GraphNode, GraphNode)>,
}

/// Replaces or deletes the sequence a target covers, in place.
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
    apply_edit(context, &sequence_graph.id, &target.locus, request)
}

fn resolve_target(
    context: &DbContext,
    target: &Bound<'_, PyAny>,
    collection_name: &str,
    sample_name: &str,
) -> PyResult<LocusTarget> {
    let locus = if let Ok(annotation) = target.extract::<PyRef<PyAnnotation>>() {
        annotation.graph_locus()
    } else if let Ok(locus) = target.extract::<PyRef<PyGraphLocus>>() {
        locus.graph_locus()
    } else if let Ok(region) = target.extract::<&str>() {
        return locus_from_region(context, region, collection_name, sample_name);
    } else {
        return Err(PyTypeError::new_err(
            "edit target must be a region string, Locus, or Annotation",
        ));
    };
    Ok(LocusTarget {
        locus,
        block_group_id: None,
    })
}

fn apply_edit(
    context: &DbContext,
    source_block_group_id: &HashId,
    target: &GraphLocus,
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
            let mut span = locate_span(&graph, &canonical)?;
            if request.stack {
                // Pruning never competes on this index, so the existing route stays.
                span.chromosome_index = INDETERMINATE_CHROMOSOME_INDEX;
                span.phased = 0;
            }
            let lefts = span
                .starts
                .iter()
                .copied()
                .map(Anchor::from)
                .collect::<Vec<_>>();
            let rights = span
                .ends
                .iter()
                .copied()
                .map(Anchor::from)
                .collect::<Vec<_>>();
            if lefts.len() > 1
                || rights.len() > 1
                || lefts.iter().any(|left| {
                    !left.splits_block()
                        && graph.edges(left.block).any(|(source, target, edges)| {
                            !span.bypassed_edges.contains(&(source, target))
                                && edges
                                    .iter()
                                    .any(|edge| edge.chromosome_index == span.chromosome_index)
                        })
                })
            {
                // Several exits must coexist, and an untouched route must not compete with the edit.
                span.chromosome_index = INDETERMINATE_CHROMOSOME_INDEX;
                span.phased = 0;
            }

            // Capture the path before writing the edited routes.
            let path_splice = if request.stack {
                None
            } else {
                current_path_pair(conn, &source.id, &lefts, &rights)?
            };

            let block = match request.kind {
                EditKind::Delete => PathBlock {
                    node_id: lefts[0].block.node_id,
                    block_sequence: String::new(),
                    sequence_start: 0,
                    sequence_end: 0,
                    path_start: 0,
                    path_end: 0,
                    strand: Strand::Forward,
                },
                EditKind::Replace => {
                    let stored = stored_sequence(request.sequence, span.is_reverse);
                    let (node_id, length) =
                        inserted_node(conn, &source.id, &lefts, &rights, &stored)?;
                    PathBlock {
                        node_id,
                        block_sequence: stored,
                        sequence_start: 0,
                        sequence_end: length,
                        path_start: 0,
                        path_end: 0,
                        strand: Strand::Forward,
                    }
                }
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
            write_edit(context, &graph, &span, &change)?;

            // Update the materialized path only when the edit lies on its route.
            if let Some(splice) = path_splice {
                let path_edges = edit_edge_ids(&splice.left, &splice.right, &block, request.kind);
                splice_current_path(conn, &splice, &path_edges)?;
            }

            let inserted = (request.kind == EditKind::Replace).then(|| GraphLocus {
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
            let summary = edit_summary(request.message, || {
                format!(
                    "{}: {} {} in sample '{}'",
                    source.name,
                    request.kind.verb(),
                    describe_locus(&canonical),
                    source.sample_name
                )
            });
            Ok((inserted, summary))
        },
        edit_error,
    )
}

/// Writes the planned edit, refreshing reused edges and retiring bypassed routes unless stacked.
/// Only anchors inside a block need a split marker; markers at existing boundaries can create
/// empty junctions or retire connections shared by neighbouring routes.
fn write_edit(
    context: &DbContext,
    graph: &GenGraph,
    span: &EditSpan,
    change: &BlockGroupChange,
) -> PyResult<()> {
    let conn = context.graph().conn();
    let block_group_id = change.region.block_group.id;
    let splits = span
        .starts
        .iter()
        .chain(&span.ends)
        .copied()
        .map(Anchor::from)
        .filter(Anchor::splits_block)
        .map(|anchor| (anchor.block.node_id, anchor.coordinate))
        .collect::<HashSet<_>>();
    let planned = change
        .region
        .plan_edges(conn, context.workspace(), change, None)
        .map_err(block_group_err_to_pyerr)?
        .into_iter()
        .filter(|row| {
            let edge = &row.edge_data;
            edge.source_node_id != edge.target_node_id
                || edge.source_coordinate != edge.target_coordinate
                || splits.contains(&(edge.source_node_id, edge.source_coordinate))
        })
        .collect::<Vec<_>>();
    let edge_ids = planned
        .iter()
        .map(|row| row.edge_data.id_hash())
        .collect::<HashSet<_>>();
    let reactivated = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None)
        .into_iter()
        .filter(|row| edge_ids.contains(&row.edge.id))
        .map(|row| {
            BlockGroupEdgeData {
                block_group_id,
                edge_id: row.edge.id,
                chromosome_index: row.chromosome_index,
                phased: row.phased,
            }
            .id_hash()
        });
    let superseded = if change.preserve_edge {
        vec![]
    } else {
        span.bypassed_edges
            .iter()
            .flat_map(|(source, target)| live_rows(graph, &block_group_id, *source, *target))
            .filter(|row| !edge_ids.contains(&row.edge_id))
            .collect()
    };
    BlockGroupEdge::select(conn)
        .delete_by_ids(
            reactivated
                .chain(superseded.iter().map(BlockGroupEdgeData::id_hash))
                .collect::<Vec<_>>(),
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let mut edges = planned.iter().map(|row| row.edge_data).collect::<Vec<_>>();
    edges.sort_unstable();
    edges.dedup();
    Edge::bulk_create(conn, &edges);
    let rows = planned
        .into_iter()
        .map(|row| BlockGroupEdgeData {
            block_group_id,
            edge_id: row.edge_data.id_hash(),
            chromosome_index: row.chromosome_index,
            phased: row.phased,
        })
        .chain(superseded.into_iter().map(|row| BlockGroupEdgeData {
            chromosome_index: PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
            phased: 0,
            ..row
        }))
        .collect::<Vec<_>>();
    BlockGroupEdge::bulk_create(conn, &rows);
    Ok(())
}

fn validate_request(target: &GraphLocus, request: &EditRequest<'_>) -> PyResult<()> {
    if target.slices.is_empty() {
        return Err(PyValueError::new_err("edit target covers no sequence"));
    }
    if target.length() == 0 {
        return Err(PyValueError::new_err(format!(
            "{}() needs a target that covers at least one position",
            request.kind.verb()
        )));
    }
    if request.kind == EditKind::Replace && request.sequence.is_empty() {
        return Err(PyValueError::new_err(
            "replace() needs a non-empty sequence; use delete() to remove a target",
        ));
    }
    Ok(())
}

/// Resolve the complete target against the active route before selecting flanking
/// anchors. Reverse search hits may list slices in graph order, whereas a reverse
/// complement lists them in reading order; validate connectivity in either order.
fn locate_span(graph: &GenGraph, canonical: &GraphLocus) -> PyResult<EditSpan> {
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
    let slices = current_slices(graph, &locus).or_else(|error| {
        if !is_reverse {
            return Err(error);
        }
        locus.slices.reverse();
        current_slices(graph, &locus)
    })?;
    let first = slices.first().expect("should have a target slice");
    let last = slices.last().expect("should have a target slice");
    let start = span_anchors(graph, first, true)?;
    let end = span_anchors(graph, last, false)?;
    let (chromosome_index, phased) = entry_chromosome(graph, first);
    Ok(EditSpan {
        starts: start.anchors,
        ends: end.anchors,
        is_reverse,
        chromosome_index,
        phased,
        bypassed_edges: start
            .bypassed_edges
            .into_iter()
            .chain(end.bypassed_edges)
            .collect(),
    })
}

struct SpanBoundary {
    anchors: Vec<GraphNodePosition>,
    bypassed_edges: Vec<(GraphNode, GraphNode)>,
}

/// Attach inside a boundary block when sequence remains on that side; otherwise use every
/// neighbouring route. Retiring the edges into the removed sequence avoids creating an empty
/// junction at a node end and also handles routes that pruning never competes on.
fn span_anchors(graph: &GenGraph, slice: &GraphNodeSlice, start: bool) -> PyResult<SpanBoundary> {
    let offset = if start { slice.start } else { slice.end } as i64;
    let at_boundary = if start {
        offset == 0
    } else {
        offset == slice.block.length()
    };
    if !at_boundary {
        return Ok(SpanBoundary {
            anchors: vec![GraphNodePosition {
                graph_node: slice.block,
                offset,
            }],
            bypassed_edges: vec![],
        });
    }
    let mut neighbors = graph
        .neighbors_directed(slice.block, if start { Incoming } else { Outgoing })
        .collect::<Vec<_>>();
    neighbors.sort_unstable();
    if neighbors.is_empty() {
        return Err(PyValueError::new_err(
            "target boundary has no route on its other side to attach to",
        ));
    }
    let mut anchors = vec![];
    let mut bypassed = vec![];
    for neighbor in neighbors {
        let (source, target) = if start {
            (neighbor, slice.block)
        } else {
            (slice.block, neighbor)
        };
        if !forward_edge(graph, source, target) {
            return Err(PyValueError::new_err(
                "Target boundary cannot be applied to the current graph orientation",
            ));
        }
        anchors.push(GraphNodePosition {
            graph_node: neighbor,
            offset: if start { neighbor.length() } else { 0 },
        });
        bypassed.push((source, target));
    }
    Ok(SpanBoundary {
        anchors,
        bypassed_edges: bypassed,
    })
}

/// The chromosome and phase an edit inherits from the route entering its first position.
///
/// Only a target starting at its block's first position has an entering edge to read; inside a
/// block the reference chromosome applies.
///
/// Negative indexes do not identify chromosome copies, so a target with only those entering edges
/// falls back to the reference chromosome. A target shared by several copies uses the alternative
/// index, letting all its incoming routes coexist through the edit.
fn entry_chromosome(graph: &GenGraph, first: &GraphNodeSlice) -> (i64, i64) {
    if first.start != 0 {
        return (0, 0);
    }
    let mut chromosomes = graph
        .edges_directed(first.block, Incoming)
        .flat_map(|(_, _, edges)| edges.iter())
        .filter(|edge| edge.chromosome_index >= 0)
        .map(|edge| (edge.chromosome_index, edge.phased))
        .collect::<Vec<_>>();
    chromosomes.sort_unstable();
    chromosomes.dedup();
    match chromosomes.as_slice() {
        [] => (0, 0),
        [chromosome] => *chromosome,
        // A shared target is edited on every incoming route, as for a multi-anchor insertion.
        _ => (INDETERMINATE_CHROMOSOME_INDEX, 0),
    }
}

/// Resolve every immutable position, so an edit cannot silently skip missing interior segments.
fn current_slices(graph: &GenGraph, locus: &GraphLocus) -> PyResult<Vec<GraphNodeSlice>> {
    let mut slices: Vec<GraphNodeSlice> = Vec::new();
    for slice in &locus.canonical().slices {
        let range = slice.block;
        let mut blocks = graph
            .nodes()
            .filter(|block| {
                block.node_id == range.node_id
                    && block.sequence_start < range.sequence_end
                    && range.sequence_start < block.sequence_end
            })
            .collect::<Vec<_>>();
        blocks.sort();
        if blocks.is_empty() {
            return Err(PyValueError::new_err(
                "Target sequence is not present in the current graph",
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
                    "Target sequence is not present in the current graph",
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
                "Target sequence is not present in the current graph",
            ));
        }
    }
    Ok(slices)
}

/// A region whose edit sites are already resolved to graph positions for edge planning.
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
        start: span.starts[0].coordinate(),
        end: span.ends[0].coordinate(),
        start_anchors: Some(span.starts.clone()),
        end_anchors: Some(span.ends.clone()),
        remove_ambiguous_positions: false,
    }
}

/// The ordinary edges a replacement or deletion writes, in path order, named from the same anchors
/// and block it is given rather than searched for afterwards. `splice_current_path` puts them
/// between the path edges that reach the edit site.
fn edit_edge_ids(left: &Anchor, right: &Anchor, block: &PathBlock, kind: EditKind) -> Vec<HashId> {
    match kind {
        EditKind::Replace => vec![
            entry_edge(left, block.node_id).id_hash(),
            exit_edge(right, block.node_id, block.sequence_end).id_hash(),
        ],
        EditKind::Delete => vec![
            EdgeData {
                source_node_id: left.block.node_id,
                source_coordinate: left.coordinate,
                source_strand: Strand::Forward,
                target_node_id: right.block.node_id,
                target_coordinate: right.coordinate,
                target_strand: Strand::Forward,
            }
            .id_hash(),
        ],
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

/// The block group edge rows behind a live edge between two blocks.
fn live_rows(
    graph: &GenGraph,
    block_group_id: &HashId,
    source: GraphNode,
    target: GraphNode,
) -> Vec<BlockGroupEdgeData> {
    graph
        .edge_weight(source, target)
        .into_iter()
        .flatten()
        .map(|edge| BlockGroupEdgeData {
            block_group_id: *block_group_id,
            edge_id: edge.edge_id,
            chromosome_index: edge.chromosome_index,
            phased: edge.phased,
        })
        .collect()
}

/// The sequence as stored in node order. Reverse-strand edits are written reverse complemented so
/// that reading the edit site on its strand yields `sequence`.
fn stored_sequence(sequence: &str, is_reverse: bool) -> String {
    if is_reverse {
        String::from_utf8(reverse_complement(sequence.as_bytes()))
            .expect("should reverse complement to valid UTF-8")
    } else {
        sequence.to_string()
    }
}

/// Stores a new sequence as a node keyed by where it attaches, and returns its id and length.
///
/// The same sequence between the same anchors is the same node, so writing them again, such as after
/// deleting them, reuses the node and the edges already recorded for it instead of adding another.
fn inserted_node(
    conn: &GraphConnection,
    block_group_id: &HashId,
    lefts: &[Anchor],
    rights: &[Anchor],
    stored: &str,
) -> PyResult<(HashId, i64)> {
    let saved = Sequence::new()
        .sequence_type("DNA")
        .sequence(stored)
        .save(conn)
        .map_err(|error| PyRuntimeError::new_err(format!("cannot store sequence: {error}")))?;
    let node_key = format!(
        "python-edit:{block_group_id}:{}|{}:{}",
        describe_anchors(lefts),
        describe_anchors(rights),
        saved.hash
    );
    let node_id = Node::create(conn, &saved.hash, &HashId::convert_str(&node_key))
        .map_err(|error| PyRuntimeError::new_err(format!("cannot create node: {error}")))?;
    Ok((node_id, saved.length))
}

/// The edge from a left anchor into the first position of a new node.
fn entry_edge(anchor: &Anchor, node_id: HashId) -> EdgeData {
    EdgeData {
        source_node_id: anchor.block.node_id,
        source_coordinate: anchor.coordinate,
        source_strand: Strand::Forward,
        target_node_id: node_id,
        target_coordinate: 0,
        target_strand: Strand::Forward,
    }
}

/// The edge from the last position of a new node to a right anchor.
fn exit_edge(anchor: &Anchor, node_id: HashId, length: i64) -> EdgeData {
    EdgeData {
        source_node_id: node_id,
        source_coordinate: length,
        source_strand: Strand::Forward,
        target_node_id: anchor.block.node_id,
        target_coordinate: anchor.coordinate,
        target_strand: Strand::Forward,
    }
}

/// The current path, an edit's left and right anchor that lie on it, and the path blocks they
/// attach to.
struct PathSplice {
    path: Path,
    left: Anchor,
    right: Anchor,
    left_block: PathBlock,
    right_block: PathBlock,
}

/// Finds the anchor pair the current path runs through, if any.
///
/// Several anchors on one side can lie on the path, as when a deletion skips the arm the path reads
/// and an insertion after it steps to both the arm and the position after it, so the pair closest
/// together along the path is the one the edit replaces there.
fn current_path_pair(
    conn: &GraphConnection,
    block_group_id: &HashId,
    lefts: &[Anchor],
    rights: &[Anchor],
) -> PyResult<Option<PathSplice>> {
    let path = match BlockGroup::get_current_path(conn, block_group_id, None) {
        Ok(path) => path,
        Err(BlockGroupError::QueryError(QueryError::ResultsNotFound(_))) => return Ok(None),
        Err(error) => return Err(block_group_err_to_pyerr(error)),
    };
    let blocks = path.coordinate_blocks(conn, None);
    let mut pairs = vec![];
    for left in lefts {
        let Some(left_index) = path_block_leaving(&blocks, left) else {
            continue;
        };
        for right in rights {
            let Some(right_index) = path_block_entering(&blocks, right) else {
                continue;
            };
            if left_index < right_index
                || (left_index == right_index && left.coordinate <= right.coordinate)
            {
                pairs.push((
                    (right_index - left_index, right.coordinate - left.coordinate),
                    *left,
                    *right,
                    left_index,
                    right_index,
                ));
            }
        }
    }
    Ok(pairs
        .into_iter()
        .min_by_key(|(distance, ..)| *distance)
        .map(|(_, left, right, left_index, right_index)| PathSplice {
            left_block: blocks[left_index].clone(),
            right_block: blocks[right_index].clone(),
            path,
            left,
            right,
        }))
}

/// The index of the path block an edge leaving `anchor` departs from. The block ending at the
/// anchor comes first, so an anchor at a junction keeps the route before it.
fn path_block_leaving(blocks: &[PathBlock], anchor: &Anchor) -> Option<usize> {
    let node_id = anchor.block.node_id;
    if is_terminal(node_id) {
        return blocks.iter().position(|block| block.node_id == node_id);
    }
    blocks
        .iter()
        .position(|block| block.node_id == node_id && block.sequence_end == anchor.coordinate)
        .or_else(|| {
            blocks.iter().position(|block| {
                block.node_id == node_id
                    && block.sequence_start <= anchor.coordinate
                    && anchor.coordinate < block.sequence_end
            })
        })
}

/// The index of the path block an edge arriving at `anchor` lands in. The block starting at the
/// anchor comes first, so an anchor at a junction resumes on the route after it.
fn path_block_entering(blocks: &[PathBlock], anchor: &Anchor) -> Option<usize> {
    let node_id = anchor.block.node_id;
    if is_terminal(node_id) {
        return blocks.iter().position(|block| block.node_id == node_id);
    }
    blocks
        .iter()
        .position(|block| block.node_id == node_id && block.sequence_start == anchor.coordinate)
        .or_else(|| {
            blocks.iter().position(|block| {
                block.node_id == node_id
                    && block.sequence_start < anchor.coordinate
                    && anchor.coordinate <= block.sequence_end
            })
        })
}

/// Rebuilds the current path with the route between the splice's anchors replaced by
/// `middle_edge_ids`.
fn splice_current_path(
    conn: &GraphConnection,
    splice: &PathSplice,
    middle_edge_ids: &[HashId],
) -> PyResult<()> {
    let PathSplice {
        path,
        left,
        right,
        left_block,
        right_block,
    } = splice;
    let edges = Path::edges_for_path(conn, &path.id, None);
    let mut edge_ids = vec![];
    if !is_start_node(left.block.node_id) {
        let entry = edges
            .iter()
            .position(|edge| {
                edge.target_node_id == left_block.node_id
                    && edge.target_coordinate == left_block.sequence_start
            })
            .expect("should have an edge entering every path block");
        edge_ids.extend(edges[..=entry].iter().map(|edge| edge.id));
    }
    edge_ids.extend_from_slice(middle_edge_ids);
    if !is_end_node(right.block.node_id) {
        let exit = edges
            .iter()
            .position(|edge| {
                edge.source_node_id == right_block.node_id
                    && edge.source_coordinate == right_block.sequence_end
            })
            .expect("should have an edge leaving every path block");
        edge_ids.extend(edges[exit..].iter().map(|edge| edge.id));
    }
    let name = format!(
        "{}-edit-{}-{}",
        path.name,
        left.describe(),
        right.describe()
    );
    Path::create(conn, &name, &path.block_group_id, &edge_ids)
        .map_err(|error| block_group_err_to_pyerr(error.into()))?;
    Ok(())
}

fn edit_summary(message: Option<&str>, generated: impl FnOnce() -> String) -> OperationSummary {
    OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: "sequence_edit".to_string(),
        },
        message.map_or_else(generated, str::to_string),
    )
}

fn edit_error(error: OperationError) -> PyErr {
    match error {
        OperationError::NoChanges => {
            PyValueError::new_err("edit made no changes to the sequence graph")
        }
        other => PyRuntimeError::new_err(format!("failed to record edit: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use r#gen::test_helpers::setup_gen_on_disk;
    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::DbContext,
        edge::Edge,
        locus::GraphLocus,
        node::Node,
        path::Path,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use pyo3::{Py, PyErr, PyResult, Python, prepare_freethreaded_python};

    use crate::python_api::{
        block_group::PySequenceGraph,
        editing::{EditKind, EditRequest, InsertSite, edit_sequence_graph, insert_at_positions},
        graph_search::PyGraphLocus,
        position::Position,
    };

    const START: &str = "start";
    const END: &str = "end";

    /// A sequence graph whose nodes are named by their sequences.
    struct Fixture {
        context: DbContext,
        graph: PySequenceGraph,
        nodes: HashMap<&'static str, HashId>,
    }

    impl Fixture {
        fn position(&self, node: &str, coordinate: i64) -> Position {
            Position {
                node_id: self.nodes[node],
                coordinate,
                strand: Strand::Forward,
            }
        }

        fn reverse_position(&self, node: &str, coordinate: i64) -> Position {
            Position {
                strand: Strand::Reverse,
                ..self.position(node, coordinate)
            }
        }

        fn insert(
            &self,
            site: InsertSite<'_>,
            sequence: &str,
            stack: bool,
        ) -> PyResult<GraphLocus> {
            prepare_freethreaded_python();
            insert_at_positions(&self.graph, &site, sequence, None, stack)
        }

        /// Replaces or deletes the sequence of `locus`, passed in as a Python `Locus`.
        fn edit(
            &self,
            locus: GraphLocus,
            kind: EditKind,
            sequence: &str,
        ) -> PyResult<Option<GraphLocus>> {
            prepare_freethreaded_python();
            Python::with_gil(|python| {
                let locus = Py::new(python, PyGraphLocus::from_locus(locus))
                    .expect("should wrap the locus");
                edit_sequence_graph(
                    &self.graph,
                    locus.bind(python).as_any(),
                    &EditRequest {
                        kind,
                        sequence,
                        message: None,
                        stack: false,
                    },
                )
            })
        }

        fn sequences(&self) -> HashSet<String> {
            BlockGroup::get_all_sequences(
                self.context.graph().conn(),
                self.context.workspace(),
                &self.graph.id,
                true,
            )
            .expect("should read sequences")
        }

        fn path_sequence(&self) -> String {
            let conn = self.context.graph().conn();
            BlockGroup::get_current_path(conn, &self.graph.id, None)
                .expect("should have a current path")
                .sequence(conn, self.context.workspace(), None)
                .expect("should read the current path")
        }
    }

    fn strings(sequences: &[&str]) -> HashSet<String> {
        sequences.iter().map(ToString::to_string).collect()
    }

    fn error_message(error: PyErr) -> String {
        prepare_freethreaded_python();
        Python::with_gil(|python| error.value(python).to_string())
    }

    fn setup_sample(context: &DbContext) {
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
    }

    /// Builds sequence graph `name` from `edges`, each joining the last position of one node to
    /// the first position of the next on a chromosome index. The current path reads the nodes in `path`.
    fn build_graph(
        context: &DbContext,
        name: &str,
        edges: &[(&'static str, &'static str, i64)],
        path: &[&'static str],
    ) -> Fixture {
        let conn = context.graph().conn();
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "collection",
                sample_name: "sample",
                name,
                parent_block_group_id: None,
                is_default: true,
            },
        )
        .expect("should create block group");
        let mut nodes = HashMap::from([(START, PATH_START_NODE_ID), (END, PATH_END_NODE_ID)]);
        for (source, target, _) in edges {
            for sequence in [source, target] {
                nodes.entry(*sequence).or_insert_with(|| {
                    let saved = Sequence::new()
                        .sequence_type("DNA")
                        .sequence(sequence)
                        .save(conn)
                        .expect("should save sequence");
                    Node::create(conn, &saved.hash, &HashId::convert_str(sequence))
                        .expect("should create node")
                });
            }
        }
        let mut edge_ids = HashMap::new();
        let mut rows = vec![];
        for (source, target, chromosome_index) in edges {
            let source_coordinate = if *source == START {
                0
            } else {
                source.len() as i64
            };
            let edge = Edge::create(
                conn,
                nodes[source],
                source_coordinate,
                Strand::Forward,
                nodes[target],
                0,
                Strand::Forward,
            )
            .expect("should create edge");
            edge_ids.insert((*source, *target), edge.id);
            rows.push(BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge.id,
                chromosome_index: *chromosome_index,
                phased: 0,
            });
        }
        BlockGroupEdge::bulk_create(conn, &rows);
        let route = [START]
            .iter()
            .chain(path)
            .chain(&[END])
            .copied()
            .collect::<Vec<_>>();
        let path_edges = route
            .windows(2)
            .map(|pair| edge_ids[&(pair[0], pair[1])])
            .collect::<Vec<_>>();
        Path::create_with_validated_edges(conn, name, &block_group.id, &path_edges)
            .expect("should create current path");
        Fixture {
            context: context.clone(),
            graph: PySequenceGraph {
                id: block_group.id,
                collection_name: "collection".to_string(),
                sample_name: "sample".to_string(),
                name: name.to_string(),
                context: Some(context.clone()),
            },
            nodes,
        }
    }

    fn fixture(edges: &[(&'static str, &'static str, i64)], path: &[&'static str]) -> Fixture {
        let context = setup_gen_on_disk();
        setup_sample(&context);
        build_graph(&context, "chr1", edges, path)
    }

    /// `ABCD` forks into `EFGH` and `IJKL`, one per chromosome, which rejoin at `MNOP`.
    fn bubble() -> Fixture {
        fixture(
            &[
                (START, "ABCD", 0),
                ("ABCD", "EFGH", 0),
                ("ABCD", "IJKL", 1),
                ("EFGH", "MNOP", 0),
                ("IJKL", "MNOP", 1),
                ("MNOP", END, 0),
            ],
            &["ABCD", "EFGH", "MNOP"],
        )
    }

    /// The bubble in DNA, for tests that read the reverse strand: `AACC` forks into `GGGA` and
    /// `TTTC`, which rejoin at `CAAT`.
    fn dna_bubble() -> Fixture {
        fixture(
            &[
                (START, "AACC", 0),
                ("AACC", "GGGA", 0),
                ("AACC", "TTTC", 1),
                ("GGGA", "CAAT", 0),
                ("TTTC", "CAAT", 1),
                ("CAAT", END, 0),
            ],
            &["AACC", "GGGA", "CAAT"],
        )
    }

    /// Two parts that never meet, like two library members: `ABCD` then `EFGH`, and `IJKL` then
    /// `MNOP`.
    fn library() -> Fixture {
        fixture(
            &[
                (START, "ABCD", 0),
                (START, "IJKL", 1),
                ("ABCD", "EFGH", 0),
                ("IJKL", "MNOP", 1),
                ("EFGH", END, 0),
                ("MNOP", END, 1),
            ],
            &["ABCD", "EFGH"],
        )
    }

    /// A fully combinatorial layer, as a library import builds: `ABCD` and `EFGH` each read into
    /// both `IJKL` and `MNOP`.
    fn biclique() -> Fixture {
        fixture(
            &[
                (START, "ABCD", -3),
                (START, "EFGH", -3),
                ("ABCD", "IJKL", -3),
                ("ABCD", "MNOP", -3),
                ("EFGH", "IJKL", -3),
                ("EFGH", "MNOP", -3),
                ("IJKL", END, -3),
                ("MNOP", END, -3),
            ],
            &["ABCD", "IJKL"],
        )
    }

    /// Two combinatorial layers of three parts: `ABCD`, `EFGH` and `IJKL` each read into `MNOP`,
    /// `QRST` and `UVWX`.
    fn layers() -> Fixture {
        let mut edges = vec![];
        for left in ["ABCD", "EFGH", "IJKL"] {
            edges.push((START, left, -3));
            for right in ["MNOP", "QRST", "UVWX"] {
                edges.push((left, right, -3));
            }
        }
        for right in ["MNOP", "QRST", "UVWX"] {
            edges.push((right, END, -3));
        }
        fixture(&edges, &["ABCD", "MNOP"])
    }

    /// `ABCD` joined end to end to `EFGH` the way `stitch` and `import library` assemble parts,
    /// with every edge on the index pruning never competes on.
    fn seam() -> Fixture {
        fixture(
            &[(START, "ABCD", -3), ("ABCD", "EFGH", -3), ("EFGH", END, -3)],
            &["ABCD", "EFGH"],
        )
    }

    /// Edits driven from Python against the annotated `editing.gbk` fixture, which reads
    /// `AAACCCGGGTTTAAACCCGGGTTT`.
    mod python_scripts {
        use std::{collections::HashSet, ffi::CString};

        use r#gen::test_helpers::setup_gen_on_disk;
        use gen_models::{block_group::BlockGroup, db::DbContext};
        use pyo3::{
            Py, PyRef, Python, prepare_freethreaded_python,
            types::{PyDict, PyDictMethods as _},
        };

        use crate::python_api::{block_group::PySequenceGraph, repository::PyRepository};

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
annotations = graph.annotations
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
annotation = next(item for item in graph.annotations if item.name == 'second')
before = annotation.segments
locus = annotation.locus
[whole] = graph.search('AAACCCGGGTTTAAACCCGGGTTT', sequence_kind='exact')
graph.insert('AG', after=whole.start())
assert annotation.segments == before
assert next(item for item in graph.annotations if item.id == annotation.id).segments == before
graph.delete(locus)
"#,
                "AAGAACCCGGGAAACCCGGGTTT",
            );
        }

        #[test]
        fn test_edit_sequential_annotation_relative_deletions() {
            run_edit_test(
                r#"
for annotation in graph.annotations:
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
middle = graph.insert('TT', after=inserted.slice(1, 2).start())
assert len(middle) == 2
graph.delete(middle)
graph.insert('C', after=inserted.end())
"#,
                "AAAATGCCTTTAAACCCGGGTTT",
            );
        }

        #[test]
        fn test_edit_reverse_strand_targets() {
            run_edit_test(
                r#"
annotation = next(item for item in graph.annotations if item.name == 'reverse')
assert annotation.locus.strand == '-'
inserted = graph.replace(annotation, 'AGT')
assert inserted.strand == '-'
assert inserted.start().offset == 2
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
annotation = next(item for item in graph.annotations if item.name == 'first')
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
graph.insert('C', before=inserted.start())
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
annotation = next(item for item in graph.annotations if item.name == 'first')
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
[whole] = graph.search('AAACCCGGGTTTAAACCCGGGTTT', sequence_kind='exact')
graph.delete(whole.slice(3, 6))
graph.delete(whole.slice(6, 9))
graph.delete(whole.slice(0, 3))
graph.delete(whole.slice(21, 24))
graph.insert('AG', before=whole.slice(9, 10).start())
graph.insert('TC', after=whole.slice(20, 21).start())
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
        fn test_edit_bases_next_to_an_earlier_deletion() {
            run_edit_test(
                r#"
graph.delete('edits:3-6')
graph.replace('edits:3-4', 'T')
graph.replace('edits:2-3', 'C')
"#,
                "AACTGGTTTAAACCCGGGTTT",
            );
        }

        #[test]
        fn test_edit_repeated_insertion_and_deletion() {
            run_edit_test(
                r#"
[whole] = graph.search('AAACCCGGGTTTAAACCCGGGTTT', sequence_kind='exact')
count = len(repo.get_operations())
for _ in range(3):
    inserted = graph.insert('AG', after=whole.slice(2, 3).start())
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
[whole] = graph.search('AAACCCGGGTTTAAACCCGGGTTT', sequence_kind='exact')
graph.insert('AG', after=whole.slice(2, 3).start())
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
    }

    /// Deletion and replacement boundaries shared by several library routes.
    mod junction_edits {
        use gen_core::{PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, is_terminal};
        use gen_graph::{GraphNode, GraphNodeSlice};
        use gen_models::{block_group_edge::BlockGroupEdge, locus::GraphLocus};

        use super::{END, Fixture, START, dna_bubble, error_message, fixture, strings};
        use crate::python_api::{
            editing::{EditKind, EditRequest, InsertSite, apply_edit, current_graph},
            locus::GraphLocusExt as _,
        };

        fn target(fixture: &Fixture, sequence: &str, start: usize, end: usize) -> GraphLocus {
            GraphLocus {
                slices: vec![GraphNodeSlice {
                    block: GraphNode {
                        node_id: fixture.nodes[sequence],
                        sequence_start: 0,
                        sequence_end: sequence.len() as i64,
                    },
                    start,
                    end,
                    strand: Strand::Forward,
                }],
            }
        }

        fn assert_no_live_junctions(fixture: &Fixture) {
            let graph = current_graph(&fixture.context, &fixture.graph.id)
                .expect("should read the edited graph");
            assert!(
                graph
                    .nodes()
                    .all(|block| is_terminal(block.node_id) || block.length() > 0)
            );
        }

        #[test]
        fn test_delete_and_replace_at_a_heterozygous_join_on_both_strands() {
            for kind in [EditKind::Delete, EditKind::Replace] {
                for reverse in [false, true] {
                    let fixture = dna_bubble();
                    let locus = target(&fixture, "CAAT", 0, 2);
                    let locus = if reverse {
                        locus.reverse_complement()
                    } else {
                        locus
                    };
                    let replacement = if kind == EditKind::Delete {
                        ""
                    } else if reverse {
                        "CT"
                    } else {
                        "AG"
                    };
                    fixture
                        .edit(locus, kind, "AG")
                        .unwrap_or_else(|error| panic!("{}", error_message(error)));
                    assert_eq!(
                        fixture.sequences(),
                        strings(&[
                            &format!("AACCGGGA{replacement}AT"),
                            &format!("AACCTTTC{replacement}AT"),
                        ])
                    );
                    assert_eq!(fixture.path_sequence(), format!("AACCGGGA{replacement}AT"));
                    assert_no_live_junctions(&fixture);
                }
            }
        }

        #[test]
        fn test_junction_edits_preserve_stacked_routes_and_unrelated_alternatives() {
            for kind in [EditKind::Delete, EditKind::Replace] {
                for stack in [false, true] {
                    let fixture = fixture(
                        &[
                            (START, "AAAA", 0),
                            (START, "CCCC", 1),
                            ("AAAA", "GGGG", 0),
                            ("AAAA", "ACAC", 1),
                            ("CCCC", "GGGG", 0),
                            ("CCCC", "ACAC", 1),
                            ("GGGG", "TTTT", 0),
                            ("GGGG", "CTCT", 1),
                            ("ACAC", "TTTT", -3),
                            ("ACAC", "CTCT", -3),
                            ("TTTT", END, -3),
                            ("CTCT", END, -3),
                        ],
                        &["AAAA", "GGGG", "TTTT"],
                    );
                    let replacement = if kind == EditKind::Replace {
                        "ATAT"
                    } else {
                        ""
                    };
                    apply_edit(
                        &fixture.context,
                        &fixture.graph.id,
                        &target(&fixture, "GGGG", 0, 4),
                        &EditRequest {
                            kind,
                            sequence: replacement,
                            message: None,
                            stack,
                        },
                    )
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                    let mut expected = strings(&[]);
                    for left in ["AAAA", "CCCC"] {
                        for right in ["TTTT", "CTCT"] {
                            expected.insert(format!("{left}{replacement}{right}"));
                            expected.insert(format!("{left}ACAC{right}"));
                            if stack {
                                expected.insert(format!("{left}GGGG{right}"));
                            }
                        }
                    }
                    assert_eq!(fixture.sequences(), expected);
                    assert_eq!(
                        fixture.path_sequence(),
                        if stack {
                            "AAAAGGGGTTTT".to_string()
                        } else {
                            format!("AAAA{replacement}TTTT")
                        }
                    );
                    assert_no_live_junctions(&fixture);
                    if stack {
                        fixture
                            .edit(target(&fixture, "GGGG", 0, 4), EditKind::Replace, "TATA")
                            .unwrap_or_else(|error| panic!("{}", error_message(error)));
                        expected.retain(|sequence| !sequence.contains("GGGG"));
                        for left in ["AAAA", "CCCC"] {
                            for right in ["TTTT", "CTCT"] {
                                expected.insert(format!("{left}TATA{right}"));
                            }
                        }
                        assert_eq!(fixture.sequences(), expected);
                        assert_eq!(fixture.path_sequence(), "AAAATATATTTT");
                    }
                }
            }
        }

        #[test]
        fn test_delete_and_replace_at_a_join_allow_followup_insertions() {
            for kind in [EditKind::Delete, EditKind::Replace] {
                let fixture = fixture(
                    &[
                        (START, "TTTTTTTTTT", -3),
                        (START, "CCCCCCCCCC", -3),
                        ("TTTTTTTTTT", "GGGGGAAAAA", -3),
                        ("CCCCCCCCCC", "GGGGGAAAAA", -3),
                        ("GGGGGAAAAA", END, -3),
                    ],
                    &["TTTTTTTTTT", "GGGGGAAAAA"],
                );
                let replacement = if kind == EditKind::Replace {
                    "ATAT"
                } else {
                    ""
                };
                fixture
                    .edit(target(&fixture, "GGGGGAAAAA", 0, 5), kind, replacement)
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                assert_eq!(
                    fixture.sequences(),
                    strings(&[
                        &format!("TTTTTTTTTT{replacement}AAAAA"),
                        &format!("CCCCCCCCCC{replacement}AAAAA"),
                    ])
                );
                assert_eq!(
                    fixture.path_sequence(),
                    format!("TTTTTTTTTT{replacement}AAAAA")
                );
                assert_no_live_junctions(&fixture);
                let rows = BlockGroupEdge::edges_for_block_group(
                    fixture.context.graph().conn(),
                    &fixture.graph.id,
                    None,
                );
                for predecessor in ["TTTTTTTTTT", "CCCCCCCCCC"] {
                    assert!(rows.iter().any(|row| row.edge.source_node_id
                        == fixture.nodes[predecessor]
                        && row.edge.target_node_id == fixture.nodes["GGGGGAAAAA"]
                        && row.edge.target_coordinate == 0
                        && row.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX));
                }
                fixture
                    .insert(
                        InsertSite::Before(&[fixture.position("GGGGGAAAAA", 5)]),
                        "CG",
                        false,
                    )
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                fixture
                    .insert(
                        InsertSite::After(&[fixture.position("TTTTTTTTTT", 9)]),
                        "AC",
                        false,
                    )
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                assert_eq!(
                    fixture.sequences(),
                    strings(&[
                        &format!("TTTTTTTTTTAC{replacement}CGAAAAA"),
                        &format!("CCCCCCCCCC{replacement}CGAAAAA"),
                    ])
                );
                assert_no_live_junctions(&fixture);
            }
        }

        #[test]
        fn test_delete_and_replace_at_a_fork_keep_every_successor() {
            for kind in [EditKind::Delete, EditKind::Replace] {
                let fixture = fixture(
                    &[
                        (START, "AAAAAGGGGG", -3),
                        ("AAAAAGGGGG", "TTTTTTTTTT", -3),
                        ("AAAAAGGGGG", "CCCCCCCCCC", -3),
                        ("TTTTTTTTTT", END, -3),
                        ("CCCCCCCCCC", END, -3),
                    ],
                    &["AAAAAGGGGG", "TTTTTTTTTT"],
                );
                let replacement = if kind == EditKind::Replace {
                    "ATAT"
                } else {
                    ""
                };
                fixture
                    .edit(target(&fixture, "AAAAAGGGGG", 5, 10), kind, replacement)
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                assert_eq!(
                    fixture.sequences(),
                    strings(&[
                        &format!("AAAAA{replacement}TTTTTTTTTT"),
                        &format!("AAAAA{replacement}CCCCCCCCCC"),
                    ])
                );
                assert_eq!(
                    fixture.path_sequence(),
                    format!("AAAAA{replacement}TTTTTTTTTT")
                );
                assert_no_live_junctions(&fixture);
                fixture
                    .insert(
                        InsertSite::After(&[fixture.position("AAAAAGGGGG", 4)]),
                        "CG",
                        false,
                    )
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                assert_eq!(
                    fixture.sequences(),
                    strings(&[
                        &format!("AAAAACG{replacement}TTTTTTTTTT"),
                        &format!("AAAAACG{replacement}CCCCCCCCCC"),
                    ])
                );
            }
        }
    }

    /// Resolving replacement and deletion targets against the current graph and path.
    mod spans {
        use gen_core::{HashId, Strand};
        use gen_graph::{GenGraph, GraphEdge, GraphNode, GraphNodeSlice};
        use gen_models::locus::GraphLocus;

        use super::{END, START, fixture};
        use crate::python_api::editing::{Anchor, current_path_pair, locate_span};

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
                locate_span(&graph, &locus).is_err(),
                "coverage alone must not establish connectivity"
            );
        }

        #[test]
        fn test_edit_anchors_a_whole_node_span_on_its_neighbours() {
            // The shape of a library column: the alternative leaves the same block as the target.
            let mut graph = GenGraph::new();
            let target = graph_node("target", 0, 10);
            let left = graph_node("left", 0, 10);
            let right = graph_node("right", 0, 10);
            graph.add_edge(left, target, vec![forward_edge()]);
            graph.add_edge(left, graph_node("alternative", 0, 10), vec![forward_edge()]);
            graph.add_edge(target, right, vec![forward_edge()]);
            let locus = GraphLocus {
                slices: vec![GraphNodeSlice::full(target, Strand::Forward)],
            };

            let span = locate_span(&graph, &locus).expect("should locate a whole-node span");

            assert_eq!(span.starts[0].graph_node, left);
            assert_eq!(span.starts[0].coordinate(), 10);
            assert_eq!(span.ends[0].graph_node, right);
            assert_eq!(span.ends[0].coordinate(), 0);
            assert_eq!(span.bypassed_edges, vec![(left, target), (target, right)]);
        }

        #[test]
        fn test_edit_anchors_a_whole_node_span_on_every_route_at_a_join() {
            let mut graph = GenGraph::new();
            let target = graph_node("target", 0, 10);
            let right = graph_node("right", 0, 10);
            graph.add_edge(graph_node("left", 0, 10), target, vec![forward_edge()]);
            graph.add_edge(graph_node("other", 0, 10), target, vec![forward_edge()]);
            graph.add_edge(target, right, vec![forward_edge()]);
            let locus = GraphLocus {
                slices: vec![GraphNodeSlice::full(target, Strand::Forward)],
            };

            let span =
                locate_span(&graph, &locus).expect("should locate a whole-node span at a join");

            assert_eq!(span.starts.len(), 2);
            assert!(span.starts.iter().all(|anchor| anchor.coordinate() == 10));
            assert_eq!(span.ends[0].graph_node, right);
            assert_eq!(span.bypassed_edges.len(), 3);
            for predecessor in [graph_node("left", 0, 10), graph_node("other", 0, 10)] {
                assert!(span.bypassed_edges.contains(&(predecessor, target)));
            }
            assert!(span.bypassed_edges.contains(&(target, right)));
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

            graph.add_edge(target, graph_node("right", 0, 10), vec![forward_edge()]);
            let span = locate_span(&graph, &locus).expect("should locate a whole-block span");

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
                locate_span(&graph, &locus).is_err(),
                "unsupported graph orientation must fail before mutation"
            );
        }

        #[test]
        fn test_current_path_pair_distinguishes_bubble_from_current_route() {
            // `ABCD` is on the current path; the parallel `EFGH` is not.
            let fixture = fixture(
                &[
                    (START, "ABCD", 0),
                    (START, "EFGH", 1),
                    ("ABCD", END, 0),
                    ("EFGH", END, 1),
                ],
                &["ABCD"],
            );
            let conn = fixture.context.graph().conn();
            let whole_node = |name: &str, coordinate: i64| Anchor {
                block: GraphNode {
                    node_id: fixture.nodes[name],
                    sequence_start: 0,
                    sequence_end: 4,
                },
                coordinate,
            };
            let on_path = |name: &str| {
                current_path_pair(
                    conn,
                    &fixture.graph.id,
                    &[whole_node(name, 0)],
                    &[whole_node(name, 4)],
                )
                .expect("should check path membership")
                .is_some()
            };

            assert!(
                on_path("ABCD"),
                "a span on the current path's own route must be found on it"
            );
            assert!(
                !on_path("EFGH"),
                "a span on the bubble route must not be mistaken for the current path"
            );
        }
    }

    /// Insertions at `SuperPosition`s, and the edits that later reuse or remove them.
    mod inserts {
        use std::{collections::HashSet, ffi::CString};

        use gen_core::{INDETERMINATE_CHROMOSOME_INDEX, Strand};
        use gen_models::{
            block_group::BlockGroup, block_group_edge::BlockGroupEdge, locus::GraphLocus,
            sequence::reverse_complement,
        };
        use pyo3::{
            Py, Python, prepare_freethreaded_python,
            types::{PyDict, PyDictMethods as _},
        };

        use super::{
            END, Fixture, START, biclique, bubble, build_graph, dna_bubble, error_message, fixture,
            layers, library, seam, strings,
        };
        use crate::python_api::{
            editing::{EditKind, InsertSite, current_graph},
            position::{Position, PySuperPosition, require_blocks, step},
        };

        fn inserted_indexes(fixture: &Fixture, inserted: &GraphLocus) -> HashSet<i64> {
            let node_id = inserted.slices[0].block.node_id;
            BlockGroupEdge::edges_for_block_group(
                fixture.context.graph().conn(),
                &fixture.graph.id,
                None,
            )
            .into_iter()
            .filter(|row| row.edge.source_node_id == node_id || row.edge.target_node_id == node_id)
            .map(|row| row.chromosome_index)
            .collect()
        }

        #[test]
        fn test_step_splits_at_a_fork_and_joins_from_either_arm() {
            let fixture = bubble();
            let graph = current_graph(&fixture.context, &fixture.graph.id)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let step_or_panic = |position: Position, forward: bool| {
                step(&graph, &position, forward)
                    .unwrap_or_else(|error| panic!("{}", error_message(error)))
            };

            let fork = step_or_panic(fixture.position("ABCD", 3), true);
            let from_first_arm = step_or_panic(fixture.position("EFGH", 3), true);
            let from_second_arm = step_or_panic(fixture.position("IJKL", 3), true);
            let before_join = step_or_panic(fixture.position("MNOP", 0), false);

            assert_eq!(
                fork.into_iter().collect::<HashSet<_>>(),
                HashSet::from([fixture.position("EFGH", 0), fixture.position("IJKL", 0)])
            );
            assert_eq!(from_first_arm, vec![fixture.position("MNOP", 0)]);
            assert_eq!(from_second_arm, vec![fixture.position("MNOP", 0)]);
            assert_eq!(
                before_join.into_iter().collect::<HashSet<_>>(),
                HashSet::from([fixture.position("EFGH", 3), fixture.position("IJKL", 3)])
            );
        }

        #[test]
        fn test_step_on_the_reverse_strand_moves_toward_the_node_start() {
            let fixture = dna_bubble();
            let graph = current_graph(&fixture.context, &fixture.graph.id)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let stepped = step(&graph, &fixture.reverse_position("CAAT", 0), true)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                stepped.into_iter().collect::<HashSet<_>>(),
                HashSet::from([
                    fixture.reverse_position("GGGA", 3),
                    fixture.reverse_position("TTTC", 3)
                ])
            );
        }

        /// `ABCD` reads into `MNOP` directly as well as through `EFGH`, so one step from `ABCD`
        /// lands on the first position of `EFGH` and on the first position of `MNOP`, which `EFGH` reads
        /// into.
        #[test]
        fn test_step_past_a_skipped_arm_covers_the_arm_and_the_base_after_it() {
            let fixture = fixture(
                &[
                    (START, "ABCD", 0),
                    ("ABCD", "EFGH", 0),
                    ("ABCD", "MNOP", 1),
                    ("EFGH", "MNOP", 0),
                    ("MNOP", END, 0),
                ],
                &["ABCD", "EFGH", "MNOP"],
            );
            let graph = current_graph(&fixture.context, &fixture.graph.id)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let stepped = step(&graph, &fixture.position("ABCD", 3), true)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                stepped.into_iter().collect::<HashSet<_>>(),
                HashSet::from([fixture.position("EFGH", 0), fixture.position("MNOP", 0)])
            );
        }

        #[test]
        fn test_step_past_the_end_of_the_graph_fails() {
            let fixture = bubble();
            let graph = current_graph(&fixture.context, &fixture.graph.id)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let Err(error) = step(&graph, &fixture.position("MNOP", 3), true) else {
                panic!("stepping off the last position must fail");
            };

            assert!(error_message(error).contains("past the end"));
        }

        #[test]
        fn test_insert_after_a_fork_attaches_to_every_branch() {
            let fixture = bubble();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDXYEFGHMNOP", "ABCDXYIJKLMNOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABCDXYEFGHMNOP");
        }

        #[test]
        fn test_insert_before_a_join_matches_insert_after_its_arms() {
            let before = bubble();
            let after = bubble();

            before
                .insert(
                    InsertSite::Before(&[before.position("MNOP", 0)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            after
                .insert(
                    InsertSite::After(&[after.position("EFGH", 3), after.position("IJKL", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let expected = strings(&["ABCDEFGHXYMNOP", "ABCDIJKLXYMNOP"]);
            assert_eq!(before.sequences(), expected);
            assert_eq!(after.sequences(), expected);
            assert_eq!(before.path_sequence(), "ABCDEFGHXYMNOP");
            assert_eq!(after.path_sequence(), "ABCDEFGHXYMNOP");
        }

        #[test]
        fn test_insert_after_inside_a_node_splits_it() {
            let fixture = bubble();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 1)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABXYCDEFGHMNOP", "ABXYCDIJKLMNOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABXYCDEFGHMNOP");
        }

        /// `ABCD` reads into `MNOP` directly as well as through `EFGH`, as after a deletion.
        #[test]
        fn test_insert_after_a_base_whose_neighbours_share_a_route_keeps_every_route() {
            let fixture = fixture(
                &[
                    (START, "ABCD", 0),
                    ("ABCD", "EFGH", 0),
                    ("ABCD", "MNOP", 1),
                    ("EFGH", "MNOP", 0),
                    ("MNOP", END, 0),
                ],
                &["ABCD", "EFGH", "MNOP"],
            );

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDXYEFGHMNOP", "ABCDXYMNOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABCDXYEFGHMNOP");
        }

        #[test]
        fn test_insert_with_several_anchors_is_wired_like_a_library() {
            let fixture = bubble();

            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                inserted_indexes(&fixture, &inserted),
                HashSet::from([INDETERMINATE_CHROMOSOME_INDEX])
            );
        }

        #[test]
        fn test_insert_with_one_anchor_per_side_inherits_the_chromosome_it_replaces() {
            let fixture = bubble();

            let inserted = fixture
                .insert(
                    InsertSite::Junction {
                        after: &[fixture.position("ABCD", 3)],
                        before: &[fixture.position("IJKL", 0)],
                    },
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(inserted_indexes(&fixture, &inserted), HashSet::from([1]));
            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDEFGHMNOP", "ABCDXYIJKLMNOP"])
            );
        }

        #[test]
        fn test_insert_at_a_seam_does_not_inherit_the_never_pruned_index() {
            let fixture = seam();

            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(inserted_indexes(&fixture, &inserted), HashSet::from([0]));
            assert_eq!(fixture.sequences(), strings(&["ABCDXYEFGH"]));
            assert_eq!(fixture.path_sequence(), "ABCDXYEFGH");
        }

        /// One inserted node shared by parts that never meet lets each part's start read into the
        /// other part's end.
        #[test]
        fn test_insert_after_combined_positions_joins_every_left_to_every_right() {
            let fixture = library();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3), fixture.position("IJKL", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDXYEFGH", "ABCDXYMNOP", "IJKLXYEFGH", "IJKLXYMNOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABCDXYEFGH");
        }

        #[test]
        fn test_insert_at_a_fork_junction_leaves_the_other_branch_unchanged() {
            let fixture = bubble();

            fixture
                .insert(
                    InsertSite::Junction {
                        after: &[fixture.position("ABCD", 3)],
                        before: &[fixture.position("EFGH", 0)],
                    },
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDXYEFGHMNOP", "ABCDIJKLMNOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABCDXYEFGHMNOP");
        }

        #[test]
        fn test_insert_at_a_join_junction_leaves_the_other_arm_unchanged() {
            let fixture = bubble();

            fixture
                .insert(
                    InsertSite::Junction {
                        after: &[fixture.position("IJKL", 3)],
                        before: &[fixture.position("MNOP", 0)],
                    },
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDEFGHMNOP", "ABCDIJKLXYMNOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABCDEFGHMNOP");
        }

        #[test]
        fn test_insert_after_one_left_of_a_biclique_leaves_the_other_left_alone() {
            let fixture = biclique();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDXYIJKL", "ABCDXYMNOP", "EFGHIJKL", "EFGHMNOP"])
            );
        }

        #[test]
        fn test_insert_before_one_right_of_a_biclique_leaves_the_other_right_alone() {
            let fixture = biclique();

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("IJKL", 0)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDXYIJKL", "EFGHXYIJKL", "ABCDMNOP", "EFGHMNOP"])
            );
        }

        #[test]
        fn test_insert_at_one_junction_of_a_biclique_is_reached_only_through_it() {
            let fixture = biclique();

            fixture
                .insert(
                    InsertSite::Junction {
                        after: &[fixture.position("ABCD", 3)],
                        before: &[fixture.position("IJKL", 0)],
                    },
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDXYIJKL", "ABCDMNOP", "EFGHIJKL", "EFGHMNOP"])
            );
        }

        /// Every left or every right of the layer names all four connections, so every route
        /// runs through the insertion either way.
        #[test]
        fn test_insert_after_every_left_matches_insert_before_every_right_of_a_biclique() {
            let after = biclique();
            let before = biclique();

            after
                .insert(
                    InsertSite::After(&[after.position("ABCD", 3), after.position("EFGH", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            before
                .insert(
                    InsertSite::Before(&[before.position("IJKL", 0), before.position("MNOP", 0)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let expected = strings(&["ABCDXYIJKL", "ABCDXYMNOP", "EFGHXYIJKL", "EFGHXYMNOP"]);
            assert_eq!(after.sequences(), expected);
            assert_eq!(before.sequences(), expected);
        }

        /// Every route through the layer, reading `middle` between the parts where `through`
        /// holds for the pair.
        fn layer_routes(middle: &str, through: impl Fn(&str, &str) -> bool) -> HashSet<String> {
            let mut routes = HashSet::new();
            for left in ["ABCD", "EFGH", "IJKL"] {
                for right in ["MNOP", "QRST", "UVWX"] {
                    let between = if through(left, right) { middle } else { "" };
                    routes.insert(format!("{left}{between}{right}"));
                }
            }
            routes
        }

        #[test]
        fn test_insert_after_two_lefts_of_a_layer_leaves_the_third_left_alone() {
            let fixture = layers();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3), fixture.position("EFGH", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                layer_routes("XY", |left, _| left != "IJKL")
            );
        }

        #[test]
        fn test_insert_before_two_rights_of_a_layer_leaves_the_third_right_alone() {
            let fixture = layers();

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("MNOP", 0), fixture.position("QRST", 0)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                layer_routes("XY", |_, right| right != "UVWX")
            );
        }

        #[test]
        fn test_insert_between_two_lefts_and_two_rights_of_a_layer_covers_only_their_routes() {
            let fixture = layers();

            fixture
                .insert(
                    InsertSite::Junction {
                        after: &[fixture.position("ABCD", 3), fixture.position("EFGH", 3)],
                        before: &[fixture.position("MNOP", 0), fixture.position("QRST", 0)],
                    },
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                layer_routes("XY", |left, right| left != "IJKL" && right != "UVWX")
            );
        }

        /// One insertion per chain keeps each chain's own pairing, where a single insertion after
        /// both would let either chain read on into the other.
        #[test]
        fn test_one_insert_per_parallel_chain_keeps_the_chains_apart() {
            let fixture = library();

            let inserted = [fixture.position("ABCD", 3), fixture.position("IJKL", 3)].map(|left| {
                fixture
                    .insert(InsertSite::After(&[left]), "XY", false)
                    .unwrap_or_else(|error| panic!("{}", error_message(error)))
            });

            assert_eq!(fixture.sequences(), strings(&["ABCDXYEFGH", "IJKLXYMNOP"]));
            assert_ne!(
                inserted[0].slices[0].block.node_id,
                inserted[1].slices[0].block.node_id
            );
        }

        #[test]
        fn test_insert_at_a_junction_inside_a_node_matches_insert_after() {
            let junction = seam();
            let after = seam();

            junction
                .insert(
                    InsertSite::Junction {
                        after: &[junction.position("ABCD", 1)],
                        before: &[junction.position("ABCD", 2)],
                    },
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            after
                .insert(InsertSite::After(&[after.position("ABCD", 1)]), "XY", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(junction.sequences(), strings(&["ABXYCDEFGH"]));
            assert_eq!(junction.sequences(), after.sequences());
            assert_eq!(junction.path_sequence(), after.path_sequence());
        }

        #[test]
        fn test_insert_at_a_junction_on_the_reverse_strand_reads_on_that_strand() {
            let fixture = dna_bubble();

            // The reverse strand reads CAAT into GGGA and TTTC, which both read into AACC.
            let inserted = fixture
                .insert(
                    InsertSite::Junction {
                        after: &[fixture.reverse_position("CAAT", 0)],
                        before: &[fixture.reverse_position("GGGA", 3)],
                    },
                    "AC",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["AACCGGGAGTCAAT", "AACCTTTCCAAT"])
            );
            assert_eq!(inserted.slices[0].strand, Strand::Reverse);
        }

        #[test]
        fn test_insert_at_positions_that_do_not_flank_a_junction_fails() {
            let fixture = bubble();

            for (after, before) in [
                // Distant positions of one route.
                (fixture.position("ABCD", 1), fixture.position("EFGH", 2)),
                // A junction given the wrong way round.
                (fixture.position("EFGH", 0), fixture.position("ABCD", 3)),
                // Two arms that never read into each other.
                (fixture.position("EFGH", 3), fixture.position("IJKL", 0)),
            ] {
                let Err(error) = fixture.insert(
                    InsertSite::Junction {
                        after: &[after],
                        before: &[before],
                    },
                    "XY",
                    false,
                ) else {
                    panic!("positions that do not flank a junction must be refused");
                };

                let message = error_message(error);
                assert!(message.contains("flank a junction"), "{message}");
                assert!(message.contains("replace()"), "{message}");
            }
            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDEFGHMNOP", "ABCDIJKLMNOP"])
            );
        }

        /// Every position given must take part in a connection, even when another one does.
        #[test]
        fn test_insert_at_a_junction_with_an_unconnected_extra_position_fails() {
            let fixture = bubble();

            let Err(error) = fixture.insert(
                InsertSite::Junction {
                    after: &[fixture.position("ABCD", 3)],
                    before: &[fixture.position("EFGH", 0), fixture.position("MNOP", 0)],
                },
                "XY",
                false,
            ) else {
                panic!("a before position no after position reads into must be refused");
            };

            assert!(error_message(error).contains("is not read directly after"));
        }

        #[test]
        fn test_inserts_at_the_ends_of_a_graph_supersede_their_terminal_edges() {
            let fixture = seam();

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("ABCD", 0)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .insert(
                    InsertSite::After(&[fixture.position("EFGH", 3)]),
                    "WZ",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["XYABCDEFGHWZ"]));
            assert_eq!(fixture.path_sequence(), "XYABCDEFGHWZ");
        }

        #[test]
        fn test_repeated_inserts_after_one_base_stay_on_a_single_route() {
            let fixture = seam();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "WZ",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["ABCDWZXYEFGH"]));
            assert_eq!(fixture.path_sequence(), "ABCDWZXYEFGH");
        }

        #[test]
        fn test_insertion_chained_onto_an_insertion_keeps_the_path_connected() {
            let fixture = seam();
            let first = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let inserted_block = first.slices[0].block;
            let end_of_first = Position {
                node_id: inserted_block.node_id,
                coordinate: inserted_block.sequence_end - 1,
                strand: Strand::Forward,
            };

            fixture
                .insert(InsertSite::After(&[end_of_first]), "WZ", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["ABCDXYWZEFGH"]));
            assert_eq!(fixture.path_sequence(), "ABCDXYWZEFGH");
        }

        #[test]
        fn test_stacked_insert_after_a_fork_keeps_every_route() {
            let fixture = bubble();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    true,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&[
                    "ABCDEFGHMNOP",
                    "ABCDIJKLMNOP",
                    "ABCDXYEFGHMNOP",
                    "ABCDXYIJKLMNOP"
                ])
            );
            assert_eq!(fixture.path_sequence(), "ABCDEFGHMNOP");
        }

        #[test]
        fn test_replacing_a_stacked_insertion_keeps_the_route_it_was_stacked_beside() {
            let fixture = seam();
            let stacked = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    true,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            fixture
                .edit(stacked, EditKind::Replace, "WZ")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["ABCDWZEFGH", "ABCDEFGH"]));
        }

        #[test]
        fn test_deleting_an_insertion_reads_as_the_original() {
            let fixture = seam();
            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            fixture
                .edit(inserted, EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["ABCDEFGH"]));
            assert_eq!(fixture.path_sequence(), "ABCDEFGH");
        }

        #[test]
        fn test_insert_and_delete_cycles_do_not_grow_the_graph() {
            let fixture = seam();
            let shape = || {
                let full_graph = BlockGroup::get_graph(
                    fixture.context.graph().conn(),
                    fixture.context.workspace(),
                    &fixture.graph.id,
                    None,
                )
                .expect("should build the graph");
                (full_graph.node_count(), full_graph.all_edges().count())
            };
            let mut shapes = vec![];
            for _ in 0..3 {
                let inserted = fixture
                    .insert(
                        InsertSite::After(&[fixture.position("ABCD", 3)]),
                        "XY",
                        false,
                    )
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                fixture
                    .edit(inserted, EditKind::Delete, "")
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                shapes.push(shape());
            }

            assert!(shapes.windows(2).all(|pair| pair[0] == pair[1]));
            assert_eq!(fixture.sequences(), strings(&["ABCDEFGH"]));
        }

        #[test]
        fn test_insert_on_the_reverse_strand_reads_on_that_strand() {
            let fixture = fixture(&[(START, "AAAC", 0), ("AAAC", END, 0)], &["AAAC"]);

            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.reverse_position("AAAC", 3)]),
                    "AC",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .insert(
                    InsertSite::After(&[fixture.reverse_position("AAAC", 0)]),
                    "AC",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["GTAAAGTC"]));
            assert_eq!(fixture.path_sequence(), "GTAAAGTC");
            // The reverse strand reads G (the last position), AC, TTT, AC.
            assert_eq!(reverse_complement(b"GTAAAGTC"), b"GACTTTAC");
            assert_eq!(inserted.slices[0].strand, Strand::Reverse);
        }

        #[test]
        fn test_insert_with_positions_on_both_strands_fails() {
            let fixture = dna_bubble();

            let Err(error) = fixture.insert(
                InsertSite::Junction {
                    after: &[fixture.position("AACC", 3)],
                    before: &[fixture.reverse_position("GGGA", 0)],
                },
                "AC",
                false,
            ) else {
                panic!("positions on both strands must be refused");
            };

            assert!(error_message(error).contains("mix strands"));
        }

        #[test]
        fn test_positions_are_valid_on_graphs_that_reach_them() {
            let fixture = bubble();
            let other = build_graph(
                &fixture.context,
                "chr2",
                &[(START, "ABCD", 0), ("ABCD", "QRST", 0), ("QRST", END, 0)],
                &["ABCD", "QRST"],
            );
            let other_graph = current_graph(&other.context, &other.graph.id)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert!(require_blocks(&other_graph, &[fixture.position("ABCD", 3)]).is_ok());
            let Err(error) = require_blocks(&other_graph, &[fixture.position("EFGH", 0)]) else {
                panic!("a position the graph does not reach must be invalid");
            };
            assert!(error_message(error).contains("invalid position"));

            other
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(other.sequences(), strings(&["ABCDXYQRST"]));
            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDEFGHMNOP", "ABCDIJKLMNOP"])
            );
        }

        #[test]
        fn test_python_position_arithmetic_and_insert() {
            let fixture = bubble();
            prepare_freethreaded_python();
            Python::with_gil(|python| {
                let graph = Py::new(python, fixture.graph.clone()).expect("should wrap graph");
                let locals = PyDict::new(python);
                locals.set_item("graph", &graph).expect("should bind graph");
                locals
                    .set_item("SuperPosition", python.get_type::<PySuperPosition>())
                    .expect("should bind SuperPosition");
                let script = r#"
[abcd] = graph.search('ABCD', sequence_kind='exact')
last = SuperPosition(abcd.end())
assert len(last) == 1 and last.sequence_graph is not None
assert last + 1 == last.on(graph) + 1
last = last.on(graph)
fork = last + 1
assert len(fork) == 2
for walk, *arguments in [(fork.__add__, 1), (fork.__sub__, 1), (last.__add__, 2)]:
    try:
        walk(*arguments)
    except ValueError as error:
        assert 'cannot step' in str(error)
    else:
        assert False, 'a superposition covering several positions must not step'
[efgh] = graph.search('EFGH', sequence_kind='exact')
[ijkl] = graph.search('IJKL', sequence_kind='exact')
assert SuperPosition(efgh.start()).on(graph) + 3 == SuperPosition(efgh.end())
join = SuperPosition(efgh.end()).on(graph) + 1
assert join == SuperPosition(ijkl.end()).on(graph) + 1
assert len(join) == 1
arm_ends = SuperPosition(efgh.end(), ijkl.end())
assert arm_ends.sequence_graph is not None
assert arm_ends == SuperPosition(efgh.end()) | SuperPosition(ijkl.end())
assert arm_ends == efgh.end() | ijkl.end()
assert efgh.end() + 1 == ijkl.end() + 1
assert join - 1 == arm_ends
assert len(fork | join) == 3
[mnop_start] = join.positions
assert mnop_start.offset == 0 and mnop_start.strand == '+'
try:
    join + 4
except IndexError:
    pass
else:
    assert False, 'stepping off the graph must fail'
inserted = graph.insert('XY', before=join)
assert len(inserted) == 2
inserted = graph.insert('WZ', after=abcd.end())
assert len(inserted) == 2
"#;
                python
                    .run(
                        &CString::new(script).expect("should encode script"),
                        Some(&locals),
                        None,
                    )
                    .unwrap_or_else(|error| {
                        error.print(python);
                        panic!("Python position assertions failed: {error}")
                    });
            });

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDWZEFGHXYMNOP", "ABCDWZIJKLXYMNOP"])
            );
        }
    }
}
