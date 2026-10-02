//! Edits to a sequence graph made through the Python API, each recorded as one operation.
//!
//! An edit works directly on stored edges, which join node coordinates (ports). It replaces the
//! span between a left port and a right port with a new sequence node, or with a deletion edge, and
//! writes only those edges. Where several routes meet at a port, graph construction manifests the
//! port as a zero-width routing block, so every route there reaches the edit without the edit
//! naming each one, and pruning drops the retired routes.
//!
//! Insertions are addressed by `SuperPosition`s and attach to every position a superposition
//! covers. An insertion is given either `after` or `before` and finds the other side by stepping
//! each position once, so a fork contributes a route per branch. Which routes an insertion lands
//! on is chosen by the positions the superposition names.
//!
//! Replacements and deletions are addressed by a region string, `Locus`, or `Annotation`, which
//! is canonicalized to node-absolute ranges and checked against the current graph. The ranges
//! need only be present and connected, directly or across routing blocks; they need not lie on one
//! path.
//!
//! Unless stacked, an edit retires the routes it replaces: a marker edge at each port, which keeps
//! the original sequence continuous, is written as retired, and the stored connections an
//! insertion sits on are removed. A stacked edit keeps the markers live and gives its edges the
//! index pruning never competes on, so both routes remain. A new sequence is stored as a node keyed
//! by the ports it sits between and what it reads. The current path is spliced when the edit lies
//! on it.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Sha256Hash,
    Strand, is_terminal,
};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::{
    block_group::{BlockGroup, BlockGroupError},
    block_group_edge::{AugmentedEdge, BlockGroupEdge, BlockGroupEdgeData},
    db::{DbContext, GraphConnection},
    edge::{Edge, EdgeData},
    errors::{OperationError, QueryError},
    locus::GraphLocus,
    node::Node,
    operations::{OperationInfo, OperationSummary},
    path::Path,
    sequence::{Sequence, reverse_complement},
};
use petgraph::Direction::{self, Incoming, Outgoing};
use pyo3::{
    Bound, PyAny, PyErr, PyRef, PyResult,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    types::PyAnyMethods as _,
};

use super::{
    annotation::PyAnnotation,
    block_group::PySequenceGraph,
    graph_read::{LocusTarget, current_graph, locus_from_region},
    graph_search::PyGraphLocus,
    locus::GraphLocusExt as _,
    position::{Position, adjacent_blocks, is_routing_block, require_blocks},
    repository::run_context_operation_write,
    utils::block_group_err_to_pyerr,
};

/// The chromosome indexes an edge is recorded under. Edits never write phases.
type Rows = BTreeSet<i64>;

/// One end of a stored edge: a node coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Port {
    node_id: HashId,
    coordinate: i64,
}

impl Port {
    fn source(edge: &Edge) -> Self {
        Self {
            node_id: edge.source_node_id,
            coordinate: edge.source_coordinate,
        }
    }

    fn target(edge: &Edge) -> Self {
        Self {
            node_id: edge.target_node_id,
            coordinate: edge.target_coordinate,
        }
    }

    /// The forward edge joining this port to `target`.
    const fn edge_to(self, target: Port) -> EdgeData {
        EdgeData {
            source_node_id: self.node_id,
            source_coordinate: self.coordinate,
            source_strand: Strand::Forward,
            target_node_id: target.node_id,
            target_coordinate: target.coordinate,
            target_strand: Strand::Forward,
        }
    }
}

/// One end of an edit's new edges: a node coordinate, and the block of the destination graph that
/// coordinate falls in or bounds, which may be a zero-width routing block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Anchor {
    block: GraphNode,
    coordinate: i64,
}

impl Anchor {
    const fn end_of(block: GraphNode) -> Self {
        Self {
            block,
            coordinate: block.sequence_end,
        }
    }

    const fn start_of(block: GraphNode) -> Self {
        Self {
            block,
            coordinate: block.sequence_start,
        }
    }

    const fn port(&self) -> Port {
        Port {
            node_id: self.block.node_id,
            coordinate: self.coordinate,
        }
    }

    fn describe(&self) -> String {
        format!("{}:{}", self.block.node_id, self.coordinate)
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
            let connections = reading_connections(&graph, site)?;
            let is_reverse = connections.is_reverse;
            let stored = BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None);
            let named_left = matches!(site, InsertSite::After(_)) != is_reverse;
            refuse_splits_other_routes_share(&stored, &connections.pairs, &graph, named_left)?;
            let plan = insertion_plan(&graph, &stored, &connections.pairs, stack);

            // Capture the path before writing the edited routes.
            let path_before = if stack {
                None
            } else {
                current_path(conn, &block_group.id)?
            };
            let stored_text = stored_sequence(sequence, is_reverse);
            let (node_id, length) = inserted_node(conn, &plan.lefts, &plan.rights, &stored_text)?;
            write_edit(conn, &block_group.id, &plan, Some((node_id, length)))?;
            if let Some(path) = path_before {
                splice_path(
                    conn,
                    &path,
                    &PathEdit {
                        lefts: &plan.lefts,
                        rights: &plan.rights,
                        allele: Some((node_id, length)),
                        removed: None,
                    },
                )?;
            }

            let locus = inserted_locus(node_id, length, is_reverse);
            let summary = edit_summary(message, || {
                format!(
                    "{}: insert {} at {} in sample '{}'",
                    block_group.name,
                    describe_sequence(sequence),
                    describe_site(&plan.lefts, &plan.rights),
                    block_group.sample_name
                )
            });
            Ok((locus, summary))
        },
        edit_error,
    )
}

/// What an edit writes: the anchors its new edges leave from and arrive at, the chromosomes those
/// edges inherit, and the stored connections it supersedes.
struct EditPlan {
    lefts: Vec<Anchor>,
    rights: Vec<Anchor>,
    /// The rows of the edge leaving each left anchor, in the order of `lefts`.
    entry_rows: Vec<Rows>,
    /// The rows of the edge arriving at each right anchor, in the order of `rights`.
    exit_rows: Vec<Rows>,
    /// The rows of the edge that skips from the left anchor to the right one, for a deletion.
    skip_rows: Rows,
    /// Stored edges an insertion replaces, whose rows are removed.
    retired_edges: HashSet<HashId>,
    /// A node taken out of the graph by the edit, whose own edges are removed.
    dissolved: Option<HashId>,
    /// Ports inside a node where the edit splits its sequence, which needs a marker edge to say
    /// whether the sequence stays continuous across the split.
    split_ports: BTreeSet<Port>,
    /// The split ports where the sequence is cut, so the routes the edit replaces end there. At
    /// the others the original sequence stays continuous.
    cut_ports: BTreeSet<Port>,
    stack: bool,
}

impl EditPlan {
    /// The new edges and what each is recorded under. A stacked edit uses the index pruning never
    /// competes on, so the routes it sits beside stay. Otherwise edges inherit the routes they
    /// replace, and edges arriving from one node on the same index would prune each other, so
    /// those use the index pruning never competes on.
    fn edges(&self, allele: Option<(HashId, i64)>) -> Vec<(EdgeData, Rows)> {
        let never_pruned = || Rows::from([INDETERMINATE_CHROMOSOME_INDEX]);
        let inherited = |rows: &Rows| {
            if self.stack {
                never_pruned()
            } else if rows.is_empty() {
                Rows::from([0])
            } else {
                rows.clone()
            }
        };
        let Some((node_id, length)) = allele else {
            let skip_rows = inherited(&self.skip_rows);
            return self
                .lefts
                .iter()
                .flat_map(|left| {
                    self.rights
                        .iter()
                        .map(|right| (left.port().edge_to(right.port()), skip_rows.clone()))
                })
                .collect();
        };
        let start = Port {
            node_id,
            coordinate: 0,
        };
        let end = Port {
            node_id,
            coordinate: length,
        };
        let mut exit_rows = self.exit_rows.iter().map(inherited).collect::<Vec<_>>();
        let mut exits_by_index = BTreeMap::<i64, usize>::new();
        for rows in &exit_rows {
            for index in rows {
                *exits_by_index.entry(*index).or_default() += 1;
            }
        }
        for rows in &mut exit_rows {
            let contested = rows
                .iter()
                .any(|index| *index >= 0 && exits_by_index[index] > 1);
            if contested {
                *rows = never_pruned();
            }
        }
        let entries = self
            .lefts
            .iter()
            .zip(&self.entry_rows)
            .map(|(left, rows)| (left.port().edge_to(start), inherited(rows)));
        let exits = self
            .rights
            .iter()
            .zip(exit_rows)
            .map(|(right, rows)| (end.edge_to(right.port()), rows));
        entries.chain(exits).collect()
    }
}

fn row_data(block_group_id: &HashId, row: &AugmentedEdge) -> BlockGroupEdgeData {
    BlockGroupEdgeData {
        block_group_id: *block_group_id,
        edge_id: row.edge.id,
        chromosome_index: row.chromosome_index,
        phased: row.phased,
    }
}

/// Writes an edit's edges, the markers that keep or cut the sequence at its ports, and removes
/// the connections it supersedes, all in one block group.
///
/// Rows already recorded for an edge the edit writes are dropped and written again, so pruning
/// ranks them as this edit's. Writing a node again, such as after deleting it, first removes the
/// edges inside it that earlier edits made, so the node reads whole again.
fn write_edit(
    conn: &GraphConnection,
    block_group_id: &HashId,
    plan: &EditPlan,
    allele: Option<(HashId, i64)>,
) -> PyResult<()> {
    let existing = BlockGroupEdge::edges_for_block_group(conn, block_group_id, None);
    let new_row = |edge: &EdgeData, chromosome_index: i64| BlockGroupEdgeData {
        block_group_id: *block_group_id,
        edge_id: edge.id_hash(),
        chromosome_index,
        phased: 0,
    };
    let mut doomed = BTreeSet::<BlockGroupEdgeData>::new();
    let mut created = BTreeSet::<BlockGroupEdgeData>::new();
    let mut new_edges = BTreeSet::<EdgeData>::new();

    for existing_row in &existing {
        let superseded = existing_row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
            && plan.retired_edges.contains(&existing_row.edge.id);
        let inside_allele = allele
            .map(|(node_id, _)| node_id)
            .into_iter()
            .chain(plan.dissolved)
            .any(|node_id| {
                existing_row.edge.source_node_id == node_id
                    && existing_row.edge.target_node_id == node_id
            });
        if superseded || inside_allele {
            doomed.insert(row_data(block_group_id, existing_row));
        }
    }

    for port in &plan.split_ports {
        let port = *port;
        let marker = port.edge_to(port);
        let marker_rows = existing
            .iter()
            .filter(|existing_row| existing_row.edge.id == marker.id_hash())
            .collect::<Vec<_>>();
        let retired = marker_rows.iter().any(|existing_row| {
            existing_row.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
        });
        new_edges.insert(marker);
        if !plan.cut_ports.contains(&port) {
            if marker_rows.is_empty() {
                created.insert(new_row(&marker, 0));
            }
        } else {
            for existing_row in &marker_rows {
                if existing_row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
                    doomed.insert(row_data(block_group_id, existing_row));
                }
            }
            if !retired {
                created.insert(new_row(&marker, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX));
            }
        }
    }

    for (edge, rows) in plan.edges(allele) {
        new_edges.insert(edge);
        for chromosome_index in rows {
            let data = new_row(&edge, chromosome_index);
            if existing
                .iter()
                .any(|existing_row| row_data(block_group_id, existing_row) == data)
            {
                doomed.insert(data.clone());
            }
            created.insert(data);
        }
    }

    // Earlier paths still read the edges this edit retires, and a path's edges must stay in its
    // block group, so a retired edge keeps a row on the index that marks it retired.
    let live_after = created
        .iter()
        .filter(|data| data.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .map(|data| data.edge_id)
        .collect::<HashSet<_>>();
    let retired_before = existing
        .iter()
        .filter(|existing_row| existing_row.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .map(|existing_row| existing_row.edge.id)
        .collect::<HashSet<_>>();
    for data in &doomed {
        let kept = data.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
            || live_after.contains(&data.edge_id)
            || retired_before.contains(&data.edge_id);
        if !kept {
            created.insert(BlockGroupEdgeData {
                chromosome_index: PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
                phased: 0,
                ..data.clone()
            });
        }
    }

    BlockGroupEdge::bulk_delete(conn, &doomed.into_iter().collect::<Vec<_>>());
    Edge::bulk_create(conn, &new_edges.into_iter().collect::<Vec<_>>());
    BlockGroupEdge::bulk_create(conn, &created.into_iter().collect::<Vec<_>>());
    Ok(())
}

/// The chromosome indexes of the routes entering or leaving `block`, looking through routing
/// blocks. Negative indexes do not identify chromosome copies, so they are left out.
fn routed_rows(graph: &GenGraph, block: GraphNode, direction: Direction) -> Rows {
    let mut rows = Rows::new();
    let mut pending = vec![block];
    let mut visited = HashSet::from([block]);
    while let Some(current) = pending.pop() {
        for (source, target, edges) in graph.edges_directed(current, direction) {
            rows.extend(
                edges
                    .iter()
                    .filter(|edge| edge.chromosome_index >= 0)
                    .map(|edge| edge.chromosome_index),
            );
            let neighbor = if direction == Incoming {
                source
            } else {
                target
            };
            if is_routing_block(neighbor) && visited.insert(neighbor) {
                pending.push(neighbor);
            }
        }
    }
    rows
}

/// The stored connections from `left` to `right`, other than retired markers.
fn stored_connection<'a>(
    stored: &'a [AugmentedEdge],
    left: &Anchor,
    right: &Anchor,
) -> impl Iterator<Item = &'a AugmentedEdge> {
    let (left, right) = (left.port(), right.port());
    stored.iter().filter(move |row| {
        row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
            && row.edge.source_strand == Strand::Forward
            && row.edge.target_strand == Strand::Forward
            && Port::source(&row.edge) == left
            && Port::target(&row.edge) == right
    })
}

/// Plans an insertion over the connections it sits on, each a left and a right anchor in node
/// order. Inside a block both anchors are one point and the edit splits the block; between blocks
/// the stored connection from the left anchor to the right one is replaced.
fn insertion_plan(
    graph: &GenGraph,
    stored: &[AugmentedEdge],
    pairs: &[(Anchor, Anchor)],
    stack: bool,
) -> EditPlan {
    let unique = |mut anchors: Vec<Anchor>| {
        anchors.sort_unstable();
        anchors.dedup();
        anchors
    };
    let lefts = unique(pairs.iter().map(|(left, _)| *left).collect());
    let rights = unique(pairs.iter().map(|(_, right)| *right).collect());
    let side_rows = |anchor: &Anchor, entering: bool| {
        let mut rows = Rows::new();
        for (left, right) in pairs
            .iter()
            .filter(|(left, right)| anchor == if entering { left } else { right })
        {
            if left.port() == right.port() {
                rows.extend(routed_rows(graph, left.block, Incoming));
            } else {
                // The replacing edge takes over the chromosomes of the connection it replaces. A
                // connection only on never-pruned indexes passes them on, since an edge on a
                // chromosome index would prune the unrelated routes already on that index.
                let connection = stored_connection(stored, left, right)
                    .map(|connection| connection.chromosome_index)
                    .collect::<Rows>();
                let chromosomes = connection
                    .iter()
                    .filter(|index| **index >= 0)
                    .copied()
                    .collect::<Rows>();
                rows.extend(if chromosomes.is_empty() {
                    connection
                } else {
                    chromosomes
                });
            }
        }
        rows
    };
    let retired_edges = if stack {
        HashSet::new()
    } else {
        pairs
            .iter()
            .filter(|(left, right)| left.port() != right.port())
            .flat_map(|(left, right)| stored_connection(stored, left, right))
            .map(|connection| connection.edge.id)
            .collect()
    };
    // An insertion inside a block splits the sequence at its point and, unless stacked, cuts it
    // there. One between two stored ports replaces the connection joining them and leaves the
    // sequence at those ports alone.
    let split_ports = pairs
        .iter()
        .filter(|(left, right)| left.port() == right.port())
        .map(|(left, _)| left.port())
        .filter(|port| !is_terminal(port.node_id))
        .collect::<BTreeSet<_>>();
    let cut_ports = if stack {
        BTreeSet::new()
    } else {
        split_ports.clone()
    };
    EditPlan {
        entry_rows: lefts.iter().map(|left| side_rows(left, true)).collect(),
        exit_rows: rights.iter().map(|right| side_rows(right, false)).collect(),
        lefts,
        rights,
        skip_rows: Rows::new(),
        retired_edges,
        dissolved: None,
        split_ports,
        cut_ports,
        stack,
    }
}

/// Refuses an insertion that would split a point other routes also pass through.
///
/// Inside a node an insertion splits the sequence at one point, leaving and rejoining it there.
/// Graph construction joins every route arriving at a point to every route leaving it, so routes
/// already there would read the new sequence too: after the point, every route arriving at it;
/// before the point, every route leaving it. The original sequence of a stacked edit meets this at
/// its first and last base, which its alternative leaves from and rejoins at. An insertion between
/// two stored ports replaces the edge joining them and reaches exactly what that edge reached, so
/// it is not checked.
///
/// Routes through the blocks the caller named are wanted, such as the end of every arm of a join.
/// Earlier insertions at the same point leave and rejoin it too. Combining with them is how
/// insertions at one point behave, in either order, so their routes are no conflict either.
fn refuse_splits_other_routes_share(
    stored: &[AugmentedEdge],
    pairs: &[(Anchor, Anchor)],
    graph: &GenGraph,
    named_left: bool,
) -> PyResult<()> {
    let split_ports = pairs
        .iter()
        .filter(|(left, right)| left.port() == right.port())
        .map(|(left, _)| left.port())
        .filter(|port| !is_terminal(port.node_id))
        .collect::<BTreeSet<_>>();
    if split_ports.is_empty() {
        return Ok(());
    }
    let routes = stored
        .iter()
        .filter(|row| {
            row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                && Port::source(&row.edge) != Port::target(&row.edge)
        })
        .map(|row| (Port::source(&row.edge), Port::target(&row.edge)))
        .collect::<Vec<_>>();
    // A node entered from a split point and leaving back to it is an insertion at that point.
    let inserted_at = |port: &Port, node_id: HashId| {
        routes
            .iter()
            .any(|(source, target)| source == port && target.node_id == node_id)
            && routes
                .iter()
                .any(|(source, target)| source.node_id == node_id && target == port)
    };
    let named = pairs
        .iter()
        .map(|(left, right)| if named_left { left.block } else { right.block })
        .collect::<HashSet<_>>();
    for (source, target) in &routes {
        let (point, other) = if named_left {
            (target, sequence_block(graph, source, false))
        } else {
            (source, sequence_block(graph, target, true))
        };
        let Some(other) = other else {
            continue;
        };
        if !split_ports.contains(point)
            || named.contains(&other)
            || inserted_at(point, other.node_id)
        {
            continue;
        }
        let relation = if named_left {
            "arrives at"
        } else {
            "leaves from"
        };
        return Err(PyValueError::new_err(format!(
            "insert would also put the new sequence on the route through {}, which {relation} \
             the same point {}: every route reaching a point reads every route leaving it. This \
             happens at the first and last base of the original sequence of a stacked edit, \
             which its alternative leaves from and rejoins at. Insert inside that sequence, or \
             at the ends of the alternative, instead",
            describe_block(&other),
            describe_point(point),
        )));
    }
    Ok(())
}

/// A block for an error message: a short node id and the slice of the node it holds.
fn describe_block(block: &GraphNode) -> String {
    let node_id = block.node_id.to_string();
    format!(
        "{}[{}:{}]",
        &node_id[..8.min(node_id.len())],
        block.sequence_start,
        block.sequence_end
    )
}

/// A point between two bases of a node for an error message, as a short node id and coordinate.
fn describe_point(port: &Port) -> String {
    let node_id = port.node_id.to_string();
    format!("{}:{}", &node_id[..8.min(node_id.len())], port.coordinate)
}

/// The block holding sequence that starts at `port` when `starting`, else the one ending there.
fn sequence_block(graph: &GenGraph, port: &Port, starting: bool) -> Option<GraphNode> {
    graph.nodes().find(|block| {
        block.node_id == port.node_id
            && !is_routing_block(*block)
            && port.coordinate
                == if starting {
                    block.sequence_start
                } else {
                    block.sequence_end
                }
    })
}

/// The connections an insertion sits on, in node order.
struct Connections {
    pairs: Vec<(Anchor, Anchor)>,
    is_reverse: bool,
}

/// The connections one position has on the side an insertion reads from it, as a (left, right)
/// pair of anchors in node order for each block the graph continues into.
///
/// Inside a block the connection is a single point. At a block edge it names the neighbouring
/// block, which can be a zero-width routing block, and so is the stored port the edge reaches.
fn position_connections(
    graph: &GenGraph,
    position: &Position,
    toward_node_end: bool,
) -> PyResult<Vec<(Anchor, Anchor)>> {
    let block = require_blocks(graph, &[*position])?[0];
    if toward_node_end {
        let this = Anchor {
            block,
            coordinate: position.coordinate + 1,
        };
        if this.coordinate < block.sequence_end {
            return Ok(vec![(this, this)]);
        }
        Ok(adjacent_blocks(graph, block, true)?
            .into_iter()
            .map(|neighbor| (this, Anchor::start_of(neighbor)))
            .collect())
    } else {
        let this = Anchor {
            block,
            coordinate: position.coordinate,
        };
        if this.coordinate > block.sequence_start {
            return Ok(vec![(this, this)]);
        }
        Ok(adjacent_blocks(graph, block, false)?
            .into_iter()
            .map(|neighbor| (Anchor::end_of(neighbor), this))
            .collect())
    }
}

fn reading_connections(graph: &GenGraph, site: &InsertSite<'_>) -> PyResult<Connections> {
    let (positions, after) = match site {
        InsertSite::After(positions) => (*positions, true),
        InsertSite::Before(positions) => (*positions, false),
    };
    let is_reverse = reads_reverse(positions)?;
    require_blocks(graph, positions)?;
    let mut pairs = vec![];
    for position in positions {
        pairs.extend(position_connections(graph, position, after != is_reverse)?);
    }
    pairs.sort_unstable();
    pairs.dedup();
    if pairs.is_empty() {
        return Err(PyValueError::new_err(
            "insertion has no route on one of its sides to attach to",
        ));
    }
    Ok(Connections { pairs, is_reverse })
}

/// Whether the positions read on the reverse strand. Every position must agree, since one inserted
/// node is stored in a single orientation.
fn reads_reverse(positions: &[Position]) -> PyResult<bool> {
    let strands = positions
        .iter()
        .map(|position| position.strand == Strand::Reverse)
        .collect::<HashSet<_>>();
    if strands.len() > 1 {
        return Err(PyValueError::new_err(
            "positions mix strands; insert on each strand separately",
        ));
    }
    Ok(strands.contains(&true))
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
    start: Anchor,
    end: Anchor,
    is_reverse: bool,
    /// The routes entering the start of the target.
    entry_rows: Rows,
    /// The routes leaving the end of the target.
    exit_rows: Rows,
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
            let span = locate_span(&graph, &canonical)?;

            // Capture the path before writing the edited routes.
            let path_before = if request.stack {
                None
            } else {
                current_path(conn, &source.id)?
            };

            let stored = stored_sequence(request.sequence, span.is_reverse);
            let existing = BlockGroupEdge::edges_for_block_group(conn, &source.id, None);
            let lone = if request.stack {
                None
            } else {
                lone_insertion(&existing, &span)
            };
            // Taking out a whole insertion restores the original sequence at its site, so the
            // new sequence, if any, is written there as a fresh insertion.
            let (lefts, rights) = match &lone {
                Some(lone) => (vec![lone.left], vec![lone.right]),
                None => (vec![span.start], vec![span.end]),
            };
            let allele = match request.kind {
                EditKind::Delete => None,
                EditKind::Replace => Some(inserted_node(conn, &lefts, &rights, &stored)?),
            };
            let skip_rows = if span.entry_rows.is_empty() {
                span.exit_rows.clone()
            } else {
                span.entry_rows.clone()
            };
            let mut plan = EditPlan {
                lefts,
                rights,
                entry_rows: vec![span.entry_rows.clone()],
                exit_rows: vec![span.exit_rows.clone()],
                skip_rows,
                retired_edges: HashSet::new(),
                dissolved: None,
                split_ports: BTreeSet::from([span.start.port(), span.end.port()])
                    .into_iter()
                    .filter(|port| !is_terminal(port.node_id))
                    .collect(),
                cut_ports: BTreeSet::new(),
                stack: request.stack,
            };
            if !request.stack {
                plan.cut_ports = plan.split_ports.clone();
            }
            let mut removed = None;
            if let Some(lone) = &lone {
                plan.entry_rows = vec![lone.entry_rows.clone()];
                plan.exit_rows = vec![lone.exit_rows.clone()];
                plan.skip_rows = lone.entry_rows.union(&lone.exit_rows).copied().collect();
                plan.retired_edges = HashSet::from([lone.entry_edge, lone.exit_edge]);
                plan.dissolved = Some(lone.node_id);
                // A replacement inside a node cuts it at the point; between two ports it
                // replaces the connection the node interrupted.
                plan.split_ports = if lone.left.port() == lone.right.port() {
                    BTreeSet::from([lone.left.port()])
                } else {
                    BTreeSet::new()
                };
                plan.cut_ports = plan.split_ports.clone();
                removed = Some((lone.entry_edge, lone.exit_edge));
            }
            write_edit(conn, &source.id, &plan, allele)?;

            // Update the materialized path only when the edit lies on its route.
            if let Some(path) = path_before {
                splice_path(
                    conn,
                    &path,
                    &PathEdit {
                        lefts: &plan.lefts,
                        rights: &plan.rights,
                        allele,
                        removed,
                    },
                )?;
            }

            let inserted =
                allele.map(|(node_id, length)| inserted_locus(node_id, length, span.is_reverse));
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

/// Resolve the complete target against the current graph before selecting flanking anchors.
/// Reverse search hits may list slices in graph order, whereas a reverse complement lists them in
/// reading order; validate connectivity in either order.
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
    Ok(EditSpan {
        start: Anchor {
            block: first.block,
            coordinate: first.block.sequence_start + first.start as i64,
        },
        end: Anchor {
            block: last.block,
            coordinate: last.block.sequence_start + last.end as i64,
        },
        is_reverse,
        entry_rows: routed_rows(graph, first.block, Incoming),
        exit_rows: routed_rows(graph, last.block, Outgoing),
    })
}

/// Whether `target` follows `source` along forward edges that cross only routing blocks, which
/// hold no sequence and so leave the two adjacent in the sequence they read.
fn follows(graph: &GenGraph, source: GraphNode, target: GraphNode) -> PyResult<bool> {
    let mut pending = vec![source];
    let mut crossed = HashSet::from([source]);
    while let Some(current) = pending.pop() {
        for neighbor in adjacent_blocks(graph, current, true)? {
            if neighbor == target {
                return Ok(true);
            }
            if is_routing_block(neighbor) && crossed.insert(neighbor) {
                pending.push(neighbor);
            }
        }
    }
    Ok(false)
}

/// Resolve every immutable position, so an edit cannot silently skip missing interior segments.
///
/// Each base of the target must be in the graph and each block must follow the one before it,
/// directly or across routing blocks. The target need not lie on one path: several routes may
/// pass through it.
fn current_slices(graph: &GenGraph, locus: &GraphLocus) -> PyResult<Vec<GraphNodeSlice>> {
    let mut slices: Vec<GraphNodeSlice> = Vec::new();
    for slice in &locus.canonical().slices {
        let range = slice.block;
        let mut blocks = graph
            .nodes()
            .filter(|block| {
                block.node_id == range.node_id
                    && block.length() > 0
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
                        && follows(graph, previous.block, block)?
                };
                if !connected {
                    return Err(PyValueError::new_err(
                        "Target is not a connected span in the current graph",
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

/// The locus of a new node's whole sequence, read on the strand it was inserted on.
fn inserted_locus(node_id: HashId, length: i64, is_reverse: bool) -> GraphLocus {
    GraphLocus {
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
/// The key is the ports the node sits between and the sequence it reads (see `allele_node_id`),
/// so the same sequence in the same place is the same node whichever sample writes it: writing it
/// again, such as after deleting it, reuses the node and the edges already recorded for it
/// instead of adding another.
fn inserted_node(
    conn: &GraphConnection,
    lefts: &[Anchor],
    rights: &[Anchor],
    stored: &str,
) -> PyResult<(HashId, i64)> {
    let saved = Sequence::new()
        .sequence_type("DNA")
        .sequence(stored)
        .save(conn)
        .map_err(|error| PyRuntimeError::new_err(format!("cannot store sequence: {error}")))?;
    let node_id = Node::create(
        conn,
        &saved.hash,
        &allele_node_id(lefts, rights, &saved.hash),
    )
    .map_err(|error| PyRuntimeError::new_err(format!("cannot create node: {error}")))?;
    Ok((node_id, saved.length))
}

/// The id of the node holding `sequence_hash` between `starts` and `ends`. Only where it sits and
/// what it reads matter, so every edit putting the same sequence in the same place shares the
/// node, whichever sample or tool makes it.
fn allele_node_id(starts: &[Anchor], ends: &[Anchor], sequence_hash: &Sha256Hash) -> HashId {
    let describe = |anchors: &[Anchor]| {
        let mut points = anchors
            .iter()
            .map(|anchor| (anchor.block.node_id, anchor.coordinate))
            .collect::<Vec<_>>();
        points.sort_unstable();
        points.dedup();
        points
            .iter()
            .map(|(node_id, coordinate)| format!("{node_id}:{coordinate}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    HashId::convert_str(&format!(
        "{}|{}|{sequence_hash}",
        describe(starts),
        describe(ends)
    ))
}

/// A node an earlier insertion put on one connection, and nothing else uses.
struct LoneInsertion {
    node_id: HashId,
    /// The ports the insertion leaves from and returns to; one port for an insertion inside a
    /// node.
    left: Anchor,
    right: Anchor,
    entry_edge: HashId,
    exit_edge: HashId,
    entry_rows: Rows,
    exit_rows: Rows,
}

/// Finds whether `span` is one whole node with one edge entering it and one leaving it, which no
/// other edit touches.
///
/// Such a node is an insertion on a single connection. Inside a node the connection is a loop
/// through one port, which a deletion edge across the node would leave as a cycle of routing
/// blocks that reaches the sequence retired at that port again. So the node is taken out and the
/// connection it interrupted is restored, which also leaves the graph as it was before the
/// insertion rather than growing it.
fn lone_insertion(stored: &[AugmentedEdge], span: &EditSpan) -> Option<LoneInsertion> {
    let node_id = span.start.block.node_id;
    if span.end.block != span.start.block
        || span.start.coordinate != 0
        || span.start.block.sequence_start != 0
        || span.end.coordinate != span.start.block.sequence_end
        || is_terminal(node_id)
    {
        return None;
    }
    let incident = stored
        .iter()
        .filter(|row| {
            row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                && (row.edge.source_node_id == node_id || row.edge.target_node_id == node_id)
        })
        .collect::<Vec<_>>();
    let entries = incident
        .iter()
        .filter(|row| row.edge.target_node_id == node_id && row.edge.source_node_id != node_id)
        .collect::<Vec<_>>();
    let exits = incident
        .iter()
        .filter(|row| row.edge.source_node_id == node_id && row.edge.target_node_id != node_id)
        .collect::<Vec<_>>();
    let inside = incident.len() - entries.len() - exits.len();
    let entry_edges = entries
        .iter()
        .map(|row| row.edge.id)
        .collect::<HashSet<_>>();
    let exit_edges = exits.iter().map(|row| row.edge.id).collect::<HashSet<_>>();
    if inside != 0 || entry_edges.len() != 1 || exit_edges.len() != 1 {
        return None;
    }
    let (entry, exit) = (&entries[0].edge, &exits[0].edge);
    let (left, right) = (Port::source(entry), Port::target(exit));
    let whole_node =
        Port::target(entry).coordinate == 0 && Port::source(exit).coordinate == span.end.coordinate;
    if !whole_node {
        return None;
    }
    let anchor = |port: Port| Anchor {
        block: GraphNode {
            node_id: port.node_id,
            sequence_start: port.coordinate,
            sequence_end: port.coordinate,
        },
        coordinate: port.coordinate,
    };
    let rows = |rows: &[&&AugmentedEdge]| {
        rows.iter()
            .filter(|row| row.chromosome_index >= 0)
            .map(|row| row.chromosome_index)
            .collect::<Rows>()
    };
    Some(LoneInsertion {
        node_id,
        left: anchor(left),
        right: anchor(right),
        entry_edge: entry.id,
        exit_edge: exit.id,
        entry_rows: rows(&entries),
        exit_rows: rows(&exits),
    })
}

/// The sequence graph's current path, if it has one.
fn current_path(conn: &GraphConnection, block_group_id: &HashId) -> PyResult<Option<Path>> {
    match BlockGroup::get_current_path(conn, block_group_id, None) {
        Ok(path) => Ok(Some(path)),
        Err(BlockGroupError::QueryError(QueryError::ResultsNotFound(_))) => Ok(None),
        Err(error) => Err(block_group_err_to_pyerr(error)),
    }
}

/// An edit to splice into a path: the anchors it runs from and to, and the node it reads, or
/// nothing for a deletion.
struct PathEdit<'a> {
    lefts: &'a [Anchor],
    rights: &'a [Anchor],
    allele: Option<(HashId, i64)>,
    /// The entering and leaving edges of a node the edit takes out, which the path drops where it
    /// reads that node.
    removed: Option<(HashId, HashId)>,
}

/// A stretch of a path inside one node, from the coordinate its entering edge arrives at to the
/// coordinate its leaving edge departs from. A routing block gives an empty stretch.
struct Stretch {
    node_id: HashId,
    enter: i64,
    exit: i64,
}

/// Writes the path an edit leaves behind next to the current one, if the path runs through the
/// edit's anchors.
///
/// The path is a list of edges, so the edit replaces the edges between its anchors with its own.
/// Several anchors on one side can lie on the path, as when a deletion skips the arm the path
/// reads and an insertion after it steps to both the arm and the position after it, so the pair
/// closest together along the path is the one the edit replaces there. Only the path's own edges
/// are read, so the edit need not lie on a single linear route of the graph.
fn splice_path(conn: &GraphConnection, path: &Path, edit: &PathEdit<'_>) -> PyResult<()> {
    let edges = Path::edges_for_path(conn, &path.id, None);
    if edges.is_empty() {
        return Ok(());
    }
    if let Some((entry, exit)) = edit.removed {
        let Some(index) = edges
            .windows(2)
            .position(|pair| pair[0].id == entry && pair[1].id == exit)
        else {
            return Ok(());
        };
        let replacement =
            new_allele_edges(&edit.lefts[0].port(), &edit.rights[0].port(), edit.allele);
        return create_path(
            conn,
            path,
            edges[..index]
                .iter()
                .map(|edge| edge.id)
                .chain(replacement)
                .chain(edges[index + 2..].iter().map(|edge| edge.id))
                .collect(),
        );
    }
    let stretches = (0..=edges.len())
        .map(|index| {
            let entering = index.checked_sub(1).map(|previous| &edges[previous]);
            let leaving = edges.get(index);
            Stretch {
                node_id: entering.map_or(edges[0].source_node_id, |edge| edge.target_node_id),
                enter: entering.map_or(0, |edge| edge.target_coordinate),
                exit: leaving.map_or(0, |edge| edge.source_coordinate),
            }
        })
        .collect::<Vec<_>>();
    let holds = |stretch: &Stretch, anchor: &Anchor| {
        stretch.node_id == anchor.block.node_id
            && stretch.enter <= anchor.coordinate
            && anchor.coordinate <= stretch.exit
    };
    let mut candidates = vec![];
    for left in edit.lefts {
        for (left_index, left_stretch) in stretches.iter().enumerate() {
            if !holds(left_stretch, left) {
                continue;
            }
            for right in edit.rights {
                for (right_index, right_stretch) in stretches.iter().enumerate() {
                    if holds(right_stretch, right)
                        && (left_index < right_index
                            || (left_index == right_index && left.coordinate <= right.coordinate))
                    {
                        candidates.push((
                            (right_index - left_index, right.coordinate - left.coordinate),
                            left_index,
                            right_index,
                            left.port(),
                            right.port(),
                        ));
                    }
                }
            }
        }
    }
    let Some((_, left_index, right_index, left, right)) =
        candidates.into_iter().min_by_key(|candidate| candidate.0)
    else {
        return Ok(());
    };

    let replacement = new_allele_edges(&left, &right, edit.allele);
    create_path(
        conn,
        path,
        edges[..left_index]
            .iter()
            .map(|edge| edge.id)
            .chain(replacement)
            .chain(edges[right_index..].iter().map(|edge| edge.id))
            .collect(),
    )
}

/// The ids of the edges an edit writes between two ports: the pair around a new node, or the one
/// that skips a deletion. At one port an insertion is never empty, so a deletion there writes
/// nothing for the path.
fn new_allele_edges(left: &Port, right: &Port, allele: Option<(HashId, i64)>) -> Vec<HashId> {
    match allele {
        None if left == right => vec![],
        None => vec![left.edge_to(*right).id_hash()],
        Some((node_id, length)) => {
            let start = Port {
                node_id,
                coordinate: 0,
            };
            let end = Port {
                node_id,
                coordinate: length,
            };
            vec![left.edge_to(start).id_hash(), end.edge_to(*right).id_hash()]
        }
    }
}

/// Writes `edge_ids` as a path next to `path`, named for the edges so each edit's path is its own.
fn create_path(conn: &GraphConnection, path: &Path, edge_ids: Vec<HashId>) -> PyResult<()> {
    let edge_list = edge_ids
        .iter()
        .map(HashId::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let name = format!("{}-edit-{}", path.name, HashId::convert_str(&edge_list));
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
    use pyo3::{Py, PyErr, PyResult, Python};

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
            Python::initialize();
            insert_at_positions(&self.graph, &site, sequence, None, stack)
        }

        /// Replaces or deletes the sequence of `locus`, passed in as a Python `Locus`.
        fn edit(
            &self,
            locus: GraphLocus,
            kind: EditKind,
            sequence: &str,
        ) -> PyResult<Option<GraphLocus>> {
            Python::initialize();
            Python::attach(|python| {
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

        /// Replaces the sequence of `locus` as a stacked alternative.
        fn edit_stacked(&self, locus: GraphLocus, sequence: &str) -> PyResult<GraphLocus> {
            Python::initialize();
            Python::attach(|python| {
                let locus = Py::new(python, PyGraphLocus::from_locus(locus))
                    .expect("should wrap the locus");
                edit_sequence_graph(
                    &self.graph,
                    locus.bind(python).as_any(),
                    &EditRequest {
                        kind: EditKind::Replace,
                        sequence,
                        message: None,
                        stack: true,
                    },
                )
                .map(|inserted| inserted.expect("should return the replacement"))
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
        Python::initialize();
        Python::attach(|python| error.value(python).to_string())
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
    /// the first position of the next on a chromosome index. The current path reads the nodes in
    /// `path`.
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
            Py, PyRef, Python,
            types::{PyDict, PyDictMethods as _},
        };

        use crate::python_api::{block_group::PySequenceGraph, repository::PyRepository};

        fn run_edit_test(script: &str, expected: &str) {
            Python::initialize();
            Python::attach(|python| {
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
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
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
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
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
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
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
        use gen_core::Strand;
        use gen_graph::{GraphNode, GraphNodeSlice};
        use gen_models::locus::GraphLocus;

        use super::{END, Fixture, START, dna_bubble, error_message, fixture, strings};
        use crate::python_api::{
            editing::{EditKind, EditRequest, InsertSite, apply_edit},
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

        #[test]
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
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
                }
            }
        }

        #[test]
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
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
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
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
            }
        }

        #[test]
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
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

        use crate::python_api::editing::{Rows, locate_span};

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
        fn test_edit_anchors_a_whole_node_span_on_its_own_ends() {
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

            assert_eq!(span.start.block, target);
            assert_eq!(span.start.coordinate, 0);
            assert_eq!(span.end.block, target);
            assert_eq!(span.end.coordinate, 10);
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

            assert_eq!(span.entry_rows, Rows::from([2]));
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
    }

    /// Edits that meet zero-width routing blocks, where several routes share a port, and edits that
    /// repeat at one point.
    mod routing_edits {
        use std::collections::HashSet;

        use gen_core::Strand;
        use gen_graph::{GenGraph, GraphEdge, GraphNode, GraphNodeSlice};
        use gen_models::locus::GraphLocus;

        use super::{END, Fixture, START, bubble, error_message, fixture, strings};
        use crate::python_api::{
            editing::{EditKind, InsertSite},
            position::{Neighbor, Position, neighbors, step},
        };

        /// `sequence[start..end]` of a node named by its sequence.
        fn slice(fixture: &Fixture, sequence: &str, start: usize, end: usize) -> GraphLocus {
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

        fn single_node() -> Fixture {
            fixture(&[(START, "AAACCC", 0), ("AAACCC", END, 0)], &["AAACCC"])
        }

        fn after(fixture: &Fixture, coordinate: i64) -> [Position; 1] {
            [fixture.position("AAACCC", coordinate)]
        }

        #[test]
        fn test_stacked_insertions_at_one_point_spell_both_orders() {
            let fixture = single_node();
            for sequence in ["GG", "TT"] {
                fixture
                    .insert(InsertSite::After(&after(&fixture, 2)), sequence, true)
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
            }

            assert_eq!(
                fixture.sequences(),
                strings(&["AAACCC", "AAAGGCCC", "AAATTCCC", "AAAGGTTCCC", "AAATTGGCCC"])
            );
            assert_eq!(fixture.path_sequence(), "AAACCC");
        }

        #[test]
        fn test_repeated_insertions_at_one_point_stay_on_one_route() {
            let fixture = single_node();
            fixture
                .insert(InsertSite::After(&after(&fixture, 2)), "GG", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let second = fixture
                .insert(InsertSite::After(&after(&fixture, 2)), "TT", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["AAATTGGCCC"]));
            assert_eq!(fixture.path_sequence(), "AAATTGGCCC");

            fixture
                .edit(second, EditKind::Replace, "CA")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(fixture.sequences(), strings(&["AAACAGGCCC"]));
            assert_eq!(fixture.path_sequence(), "AAACAGGCCC");
        }

        #[test]
        fn test_replacing_and_deleting_an_insertion_inside_a_node_restores_its_site() {
            let fixture = single_node();
            let inserted = fixture
                .insert(InsertSite::After(&after(&fixture, 2)), "GG", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let replacement = fixture
                .edit(inserted, EditKind::Replace, "TT")
                .unwrap_or_else(|error| panic!("{}", error_message(error)))
                .expect("should return the replacement");
            assert_eq!(fixture.sequences(), strings(&["AAATTCCC"]));
            assert_eq!(fixture.path_sequence(), "AAATTCCC");

            fixture
                .edit(replacement, EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(fixture.sequences(), strings(&["AAACCC"]));
            assert_eq!(fixture.path_sequence(), "AAACCC");
        }

        #[test]
        fn test_replacing_part_of_an_inserted_node_keeps_the_rest() {
            let fixture = single_node();
            let inserted = fixture
                .insert(InsertSite::After(&after(&fixture, 2)), "GGTT", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let node_id = inserted.slices[0].block.node_id;
            let middle = GraphLocus {
                slices: vec![GraphNodeSlice {
                    block: GraphNode {
                        node_id,
                        sequence_start: 0,
                        sequence_end: 4,
                    },
                    start: 1,
                    end: 3,
                    strand: Strand::Forward,
                }],
            };

            fixture
                .edit(middle, EditKind::Replace, "C")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["AAAGCTCCC"]));
            assert_eq!(fixture.path_sequence(), "AAAGCTCCC");
        }

        #[test]
        fn test_a_target_across_a_stacked_split_is_one_connected_span() {
            let fixture = single_node();
            fixture
                .insert(InsertSite::After(&after(&fixture, 2)), "GG", true)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            fixture
                .edit(slice(&fixture, "AAACCC", 2, 4), EditKind::Replace, "TT")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["AATTCC"]));
            assert_eq!(fixture.path_sequence(), "AATTCC");
        }

        #[test]
        fn test_a_target_across_a_retired_split_is_not_connected() {
            let fixture = single_node();
            fixture
                .insert(InsertSite::After(&after(&fixture, 2)), "GG", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let Err(error) = fixture.edit(slice(&fixture, "AAACCC", 2, 4), EditKind::Replace, "TT")
            else {
                panic!("a target whose halves no route joins must be refused");
            };

            assert!(error_message(error).contains("not a connected span"));
            assert_eq!(fixture.sequences(), strings(&["AAAGGCCC"]));
        }

        #[test]
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
        fn test_edits_next_to_a_deleted_prefix_attach_to_the_routing_block() {
            let fixture = bubble();
            fixture
                .edit(slice(&fixture, "MNOP", 0, 2), EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(fixture.sequences(), strings(&["ABCDEFGHOP", "ABCDIJKLOP"]));

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("MNOP", 2)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .insert(
                    InsertSite::After(&[fixture.position("EFGH", 3), fixture.position("IJKL", 3)]),
                    "QQ",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDEFGHQQXYOP", "ABCDIJKLQQXYOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABCDEFGHQQXYOP");
        }

        #[test]
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
        fn test_adjacent_deletions_and_insertions_beside_them() {
            let node = "AAAACCCCGGGG";
            let fixture = fixture(&[(START, node, 0), (node, END, 0)], &[node]);
            for (start, end) in [(4, 8), (8, 10)] {
                fixture
                    .edit(slice(&fixture, node, start, end), EditKind::Delete, "")
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
            }
            assert_eq!(fixture.sequences(), strings(&["AAAAGG"]));

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position(node, 10)]),
                    "TT",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .insert(InsertSite::After(&[fixture.position(node, 3)]), "CC", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["AAAACCTTGG"]));
            assert_eq!(fixture.path_sequence(), "AAAACCTTGG");
        }

        #[test]
        fn test_insert_before_a_routing_block_sits_in_front_of_it() {
            let fixture = fixture(
                &[
                    (START, "ABCD", 0),
                    ("ABCD", "EFGH", 0),
                    ("EFGH", "IJKL", 0),
                    ("IJKL", END, 0),
                ],
                &["ABCD", "EFGH", "IJKL"],
            );
            fixture
                .edit(slice(&fixture, "EFGH", 0, 4), EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("IJKL", 0)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["ABCDXYIJKL"]));
            assert_eq!(fixture.path_sequence(), "ABCDXYIJKL");
        }

        /// A stacked alternative is a node of its own, so the base in front of its start is
        /// reached only through it.
        #[test]
        fn test_insert_before_a_stacked_alternative_reaches_only_the_alternative() {
            let fixture = single_node();
            let alternative = fixture
                .edit_stacked(slice(&fixture, "AAACCC", 3, 6), "TT")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let start = Position {
                node_id: alternative.slices[0].block.node_id,
                coordinate: 0,
                strand: Strand::Forward,
            };

            fixture
                .insert(InsertSite::Before(&[start]), "GG", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["AAACCC", "AAAGGTT"]));
        }

        /// The alternative of a stacked edit leaves at the first base of the sequence it sits
        /// beside and rejoins at its last, so an insertion before that first base or after that
        /// last base would put the new sequence on the alternative too.
        #[test]
        fn test_insert_at_the_ends_of_a_stacked_original_is_refused() {
            let fixture = single_node();
            fixture
                .edit_stacked(slice(&fixture, "AAACCC", 3, 6), "TT")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let before = fixture.sequences();
            assert_eq!(before, strings(&["AAACCC", "AAATT"]));

            for site in [
                InsertSite::Before(&[fixture.position("AAACCC", 3)]),
                InsertSite::After(&[fixture.position("AAACCC", 5)]),
            ] {
                let Err(error) = fixture.insert(site, "GG", false) else {
                    panic!("an insertion at an end of the original must be refused");
                };
                assert!(error_message(error).contains("would also put the new sequence"));
                assert_eq!(fixture.sequences(), before);
            }

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("AAACCC", 3)]),
                    "GG",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(fixture.sequences(), strings(&["AAACGGCC", "AAATT"]));
        }

        /// An insertion inside a node leaves and rejoins it at one port, where the original arm of
        /// a stacked edit also starts. Inserting before the original arm then puts the sequence
        /// between the earlier insertion and the original arm only, since the alternative is
        /// entered from the earlier insertion by an edge of its own.
        #[test]
        fn test_insert_before_the_original_arm_behind_an_insertion_in_front_of_the_fork() {
            let fixture = single_node();
            fixture
                .edit_stacked(slice(&fixture, "AAACCC", 3, 6), "TT")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .insert(
                    InsertSite::After(&[fixture.position("AAACCC", 2)]),
                    "GG",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(fixture.sequences(), strings(&["AAAGGCCC", "AAAGGTT"]));

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("AAACCC", 3)]),
                    "AC",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["AAAGGACCCC", "AAAGGTT"]));
        }

        #[test]
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands."]
        fn test_stepping_crosses_routing_blocks_without_consuming_bases() {
            let fixture = bubble();
            fixture
                .edit(slice(&fixture, "MNOP", 0, 2), EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let graph = super::super::current_graph(&fixture.context, &fixture.graph.id)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let stepped = step(&graph, &fixture.position("EFGH", 3), true)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(stepped, vec![fixture.position("MNOP", 2)]);
            let backward = step(&graph, &fixture.position("MNOP", 2), false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(
                backward.into_iter().collect::<HashSet<_>>(),
                HashSet::from([fixture.position("EFGH", 3), fixture.position("IJKL", 3)])
            );
        }

        #[test]
        fn test_stepping_through_a_cycle_of_routing_blocks_ends() {
            let node = |name: &str, start: i64, end: i64| GraphNode {
                node_id: gen_core::HashId::convert_str(name),
                sequence_start: start,
                sequence_end: end,
            };
            let edge = GraphEdge {
                edge_id: gen_core::HashId::convert_str("edge"),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            };
            let (left, right) = (node("sequence", 0, 4), node("sequence", 6, 10));
            let (first, second) = (node("sequence", 4, 4), node("sequence", 6, 6));
            let mut graph = GenGraph::new();
            for (source, target) in [
                (left, first),
                (first, second),
                (second, first),
                (second, right),
            ] {
                graph.add_edge(source, target, vec![edge]);
            }
            let position = Position {
                node_id: left.node_id,
                coordinate: 3,
                strand: Strand::Forward,
            };

            let found = neighbors(&graph, &position, true)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                found,
                vec![Neighbor::Position(Position {
                    coordinate: 6,
                    ..position
                })]
            );
        }
    }

    /// How edits treat the sequence graph's current path: they write a new one only when the edit
    /// lies on it and is not stacked, and the graph reads the same whichever route the path takes.
    mod current_path {
        use gen_models::path::Path;

        use super::{Fixture, bubble, error_message, strings};
        use crate::python_api::editing::{EditKind, InsertSite};

        fn path_count(fixture: &Fixture) -> usize {
            Path::query_for_collection_and_sample(
                fixture.context.graph().conn(),
                "collection",
                "sample",
            )
            .len()
        }

        #[test]
        fn test_an_edit_off_the_current_path_leaves_the_path_alone() {
            let fixture = bubble();
            let before = path_count(&fixture);

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("IJKL", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(path_count(&fixture), before);
            assert_eq!(fixture.path_sequence(), "ABCDEFGHMNOP");
            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDEFGHMNOP", "ABCDIJKLXYMNOP"])
            );
        }

        #[test]
        fn test_each_edit_on_the_current_path_writes_one_path() {
            let fixture = bubble();
            let before = path_count(&fixture);

            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("EFGH", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(path_count(&fixture), before + 1);
            assert_eq!(fixture.path_sequence(), "ABCDEFGHXYMNOP");

            fixture
                .edit(inserted, EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(path_count(&fixture), before + 2);
            assert_eq!(fixture.path_sequence(), "ABCDEFGHMNOP");
        }

        #[test]
        fn test_a_stacked_edit_never_writes_a_path() {
            let fixture = bubble();
            let before = path_count(&fixture);

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("EFGH", 3)]),
                    "XY",
                    true,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(path_count(&fixture), before);
            assert_eq!(fixture.path_sequence(), "ABCDEFGHMNOP");
        }

        #[test]
        fn test_every_earlier_path_stays_valid_after_superseding_edits() {
            let fixture = bubble();
            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("EFGH", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("MNOP", 0)]),
                    "QQ",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            fixture
                .edit(inserted, EditKind::Replace, "ZZ")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let conn = fixture.context.graph().conn();
            let paths = Path::query_for_collection_and_sample(conn, "collection", "sample");
            assert!(paths.len() > 1, "each edit on the path leaves its own path");
            for path in paths {
                let edge_ids = Path::edge_ids_for_path(conn, &path.id, None);
                Path::validate_edges(conn, &edge_ids, &fixture.graph.id)
                    .unwrap_or_else(|error| panic!("{}: {error}", path.name));
            }
        }

        #[test]
        fn test_the_current_path_is_always_one_of_the_graph_sequences() {
            let fixture = bubble();
            let positions = [
                fixture.position("ABCD", 1),
                fixture.position("EFGH", 2),
                fixture.position("MNOP", 3),
            ];
            for (round, position) in positions.into_iter().enumerate() {
                fixture
                    .insert(InsertSite::After(&[position]), &format!("X{round}"), false)
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                assert!(
                    fixture.sequences().contains(&fixture.path_sequence()),
                    "the path must stay a route of the graph after round {round}"
                );
            }
        }
    }

    /// Insertions at `SuperPosition`s, and the edits that later reuse or remove them.
    mod inserts {
        use std::{collections::HashSet, ffi::CString};

        use gen_core::{INDETERMINATE_CHROMOSOME_INDEX, Strand};
        use gen_graph::{GraphNode, GraphNodeSlice};
        use gen_models::{
            block_group::BlockGroup, block_group_edge::BlockGroupEdge, locus::GraphLocus,
            sequence::reverse_complement,
        };
        use pyo3::{
            Py, Python,
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
        /// lands on the first position of `EFGH` and on the first position of `MNOP`, which `EFGH`
        /// reads
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
        fn test_insert_after_a_fork_keeps_each_branch_on_its_own_chromosome() {
            let fixture = bubble();

            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(inserted_indexes(&fixture, &inserted), HashSet::from([0, 1]));
        }

        #[test]
        fn test_insert_before_one_branch_inherits_the_chromosome_it_replaces() {
            let fixture = bubble();

            let inserted = fixture
                .insert(
                    InsertSite::Before(&[fixture.position("IJKL", 0)]),
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
        fn test_insert_at_a_seam_keeps_the_never_pruned_index_of_its_connection() {
            let fixture = seam();

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
        fn test_insert_before_one_branch_of_a_fork_leaves_the_other_unchanged() {
            let fixture = bubble();

            fixture
                .insert(
                    InsertSite::Before(&[fixture.position("EFGH", 0)]),
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
        fn test_insert_after_one_arm_of_a_join_leaves_the_other_unchanged() {
            let fixture = bubble();

            fixture
                .insert(
                    InsertSite::After(&[fixture.position("IJKL", 3)]),
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
        fn test_insert_before_a_branch_on_the_reverse_strand_reads_on_that_strand() {
            let fixture = dna_bubble();

            // The reverse strand reads CAAT into GGGA and TTTC, which both read into AACC.
            let inserted = fixture
                .insert(
                    InsertSite::Before(&[fixture.reverse_position("GGGA", 3)]),
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

        /// Both sides of an insertion inside a node attach at the same node coordinate, which must
        /// not read as an empty deletion.
        #[test]
        fn test_deleting_an_insertion_inside_a_node_reads_as_the_original() {
            let fixture = bubble();
            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 1)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            fixture
                .edit(inserted, EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                fixture.sequences(),
                strings(&["ABCDEFGHMNOP", "ABCDIJKLMNOP"])
            );
            assert_eq!(fixture.path_sequence(), "ABCDEFGHMNOP");
        }

        /// The insertion and its deletion are keyed on the sequence around them, so repeating
        /// them reuses both nodes and edges, so no repeat grows the graph.
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
            let mut inserted_nodes = HashSet::new();
            for _ in 0..3 {
                let inserted = fixture
                    .insert(
                        InsertSite::After(&[fixture.position("ABCD", 3)]),
                        "XY",
                        false,
                    )
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                inserted_nodes.insert(inserted.slices[0].block.node_id);
                fixture
                    .edit(inserted, EditKind::Delete, "")
                    .unwrap_or_else(|error| panic!("{}", error_message(error)));
                shapes.push(shape());
            }

            assert_eq!(inserted_nodes.len(), 1);
            assert_eq!(shapes[0].0, shapes[1].0);
            assert_eq!(shapes[1], shapes[2]);
            assert_eq!(fixture.sequences(), strings(&["ABCDEFGH"]));
        }

        #[test]
        fn test_the_same_insertion_from_either_side_is_the_same_node() {
            let after = seam();
            let before = seam();
            let from_after = after
                .insert(InsertSite::After(&[after.position("ABCD", 3)]), "XY", false)
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let from_before = before
                .insert(
                    InsertSite::Before(&[before.position("EFGH", 0)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(
                from_after.slices[0].block.node_id,
                from_before.slices[0].block.node_id
            );
        }

        fn fixture_without_efgh() -> Fixture {
            fixture(
                &[(START, "ABCD", -3), ("ABCD", "IJKL", -3), ("IJKL", END, -3)],
                &["ABCD", "IJKL"],
            )
        }

        /// An insertion beside a deletion reads into what the deletion reads into, and is keyed as
        /// it would be without the deletion.
        #[test]
        fn test_insert_beside_a_deletion_reads_into_what_the_deletion_reads_into() {
            let fixture = fixture(
                &[
                    (START, "ABCD", -3),
                    ("ABCD", "EFGH", -3),
                    ("EFGH", "IJKL", -3),
                    ("IJKL", END, -3),
                ],
                &["ABCD", "EFGH", "IJKL"],
            );
            let never_had_efgh = fixture_without_efgh();
            let keyed_without_deletion = never_had_efgh
                .insert(
                    InsertSite::After(&[never_had_efgh.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));
            let efgh = GraphLocus {
                slices: vec![GraphNodeSlice {
                    block: GraphNode {
                        node_id: fixture.nodes["EFGH"],
                        sequence_start: 0,
                        sequence_end: 4,
                    },
                    start: 0,
                    end: 4,
                    strand: Strand::Forward,
                }],
            };
            fixture
                .edit(efgh, EditKind::Delete, "")
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let inserted = fixture
                .insert(
                    InsertSite::After(&[fixture.position("ABCD", 3)]),
                    "XY",
                    false,
                )
                .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences(), strings(&["ABCDXYIJKL"]));
            assert_eq!(fixture.path_sequence(), "ABCDXYIJKL");
            let graph = current_graph(&fixture.context, &fixture.graph.id)
                .expect("should read the edited graph");
            let inserted_block = graph
                .nodes()
                .find(|block| block.node_id == inserted.slices[0].block.node_id)
                .expect("should reach the insertion");
            let successors = graph.neighbors(inserted_block).collect::<Vec<_>>();
            assert_eq!(successors.len(), 1);
            assert_eq!(successors[0].length(), 4);
            assert_eq!(
                inserted.slices[0].block.node_id,
                keyed_without_deletion.slices[0].block.node_id
            );
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
                InsertSite::After(&[
                    fixture.position("AACC", 3),
                    fixture.reverse_position("GGGA", 0),
                ]),
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
            Python::initialize();
            Python::attach(|python| {
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
