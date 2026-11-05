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

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Display,
};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Sha256Hash,
    Strand, is_end_node, is_start_node, is_terminal, range::Range,
};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::{
    accession::{Accession, AccessionNode, AccessionSpan, NewAccession},
    annotations::{Annotation, NewAnnotation},
    block_group::{BlockGroup, BlockGroupError, NewBlockGroup},
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
    graph_read::{
        LocusTarget, current_graph, locus_from_region, reaches_oriented, unpruned_graph,
    },
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
    mode: EditMode,
) -> PyResult<GraphLocus> {
    let EditMode { stack } = mode;
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

/// A point between two bases of a node, with the strand an edge leaves or arrives on there.
#[derive(Clone, Copy, Debug)]
struct Junction {
    node_id: HashId,
    boundary: i64,
    strand: Strand,
}

impl Junction {
    /// Where an edge leaves from when it follows `position`. On the reverse strand that is toward
    /// the start of the node.
    fn following(position: &Position) -> Self {
        let boundary = if position.strand == Strand::Reverse {
            position.coordinate
        } else {
            position.coordinate + 1
        };
        Self {
            node_id: position.node_id,
            boundary,
            strand: position.strand,
        }
    }

    /// Where an edge arrives when it precedes `position`.
    fn preceding(position: &Position) -> Self {
        let boundary = if position.strand == Strand::Reverse {
            position.coordinate + 1
        } else {
            position.coordinate
        };
        Self {
            node_id: position.node_id,
            boundary,
            strand: position.strand,
        }
    }

    /// The port a stored edge names for this junction. A reverse-strand edge names the block it
    /// reads, by the start of the block when arriving and the end of the block when leaving, so
    /// the nearest boundary of the node beyond this one on the reading side is needed.
    fn stored_port(&self, leaving: bool, bounds: &BTreeSet<i64>) -> PyResult<Port> {
        let coordinate = match (self.strand == Strand::Reverse, leaving) {
            (false, _) => Some(self.boundary),
            (true, true) => bounds.range(self.boundary + 1..).next().copied(),
            (true, false) => bounds.range(..self.boundary).next_back().copied(),
        };
        coordinate
            .map(|coordinate| Port {
                node_id: self.node_id,
                coordinate,
            })
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "connect has no sequence to read on the reverse strand {} {}",
                    if leaving { "before" } else { "after" },
                    describe_point(&Port {
                        node_id: self.node_id,
                        coordinate: self.boundary,
                    }),
                ))
            })
    }

    fn describe(&self, port: &Port) -> String {
        format!(
            "{}{}",
            describe_point(port),
            if self.strand == Strand::Reverse {
                '-'
            } else {
                '+'
            }
        )
    }
}

/// The coordinates at which the blocks of `graph` cut `node_id`.
fn node_bounds(graph: &GenGraph, node_id: HashId) -> BTreeSet<i64> {
    graph
        .nodes()
        .filter(|block| block.node_id == node_id && !is_routing_block(*block))
        .flat_map(|block| [block.sequence_start, block.sequence_end])
        .collect()
}

/// Whether an edge is the marker that keeps a node's sequence continuous across a port.
fn is_marker(row: &AugmentedEdge) -> bool {
    Port::source(&row.edge) == Port::target(&row.edge)
        && row.edge.source_strand == Strand::Forward
        && row.edge.target_strand == Strand::Forward
}

/// A name for something moving into a graph that already uses `taken`, found by adding primes until
/// the name is free: 'P' becomes "P'", then "P''". A join renames rather than refuses by default, so
/// two graphs annotated with the same feature name can still be combined.
fn unclaimed_name(name: &str, taken: &HashSet<String>) -> String {
    let mut candidate = name.to_string();
    while taken.contains(&candidate) {
        candidate.push('\'');
    }
    candidate
}

/// The nodes an accession covers, in a form two accessions can be compared by: the same nodes,
/// coordinates and strands mean the same feature wherever the accession is stored.
fn accession_spans(conn: &GraphConnection, accession_id: &HashId) -> Vec<AccessionSpan> {
    let mut spans = Accession::get_nodes_by_id(conn, accession_id, None)
        .iter()
        .map(|node| AccessionSpan {
            node_id: node.node_id,
            range: Range {
                start: node.sequence_start,
                end: node.sequence_end,
            },
            strand: node.strand,
        })
        .collect::<Vec<_>>();
    spans.sort_by_key(|span| (span.node_id, span.range.start, span.range.end, span.strand));
    spans
}

/// The accession of `kept` under `parent` that has this name and covers exactly these spans.
fn find_duplicate<'a>(
    conn: &GraphConnection,
    kept: &'a [Accession],
    parent: Option<HashId>,
    name: &str,
    spans: &[AccessionSpan],
) -> Option<&'a Accession> {
    kept.iter().find(|existing| {
        existing.parent_accession_id == parent
            && existing.name == name
            && accession_spans(conn, &existing.id) == spans
    })
}

/// Gives `target` its own copy of every accession on `source`, with the annotations that name
/// them, so annotations of an absorbed graph keep resolving after it is removed. The copies are new
/// rows over the same stable node slices, so no accession changes the graph it belongs to; the
/// originals are deleted only because the graph they belong to is about to be. An accession that
/// `target` already has under the same name over exactly the same spans is the same feature, so it
/// is not copied again. One that only shares a name takes primes, unless `strict` asks for the
/// clash to be reported.
fn move_annotations(
    conn: &GraphConnection,
    source: &BlockGroup,
    target: &BlockGroup,
    strict: bool,
) -> PyResult<()> {
    let accessions = Accession::select(conn)
        .block_group_id(source.id)
        .load()
        .map_err(|error| read_error(&error))?;
    copy_annotations(conn, &source.name, &accessions, target, strict)?;
    delete_accessions(conn, &accessions)
}

fn read_error(error: &dyn Display) -> PyErr {
    PyRuntimeError::new_err(format!("failed to read annotations: {error}"))
}

fn write_error(error: &dyn Display) -> PyErr {
    PyRuntimeError::new_err(format!("failed to copy annotations: {error}"))
}

/// Copies `accessions`, which must hold the parent of every child in it, onto `target` with their
/// annotations. `source_name` only names the graph they come from in a clash reported when strict.
fn copy_annotations(
    conn: &GraphConnection,
    source_name: &str,
    accessions: &[Accession],
    target: &BlockGroup,
    strict: bool,
) -> PyResult<()> {
    let kept = Accession::select(conn)
        .block_group_id(target.id)
        .load()
        .map_err(|error| read_error(&error))?;
    if strict
        && let Some(clash) = accessions.iter().find(|accession| {
            accession.parent_accession_id.is_none()
                && kept.iter().any(|existing| {
                    existing.parent_accession_id.is_none() && existing.name == accession.name
                })
                && find_duplicate(
                    conn,
                    &kept,
                    None,
                    &accession.name,
                    &accession_spans(conn, &accession.id),
                )
                .is_none()
        })
    {
        return Err(PyValueError::new_err(format!(
            "connect() cannot join '{}' because both sequence graphs have an annotation named '{}'",
            source_name, clash.name
        )));
    }

    // A child accession is created after its parent, whose copy has a new id.
    let mut taken_names = HashMap::<Option<HashId>, HashSet<String>>::new();
    for existing in &kept {
        taken_names
            .entry(existing.parent_accession_id)
            .or_default()
            .insert(existing.name.clone());
    }
    let mut copies = HashMap::new();
    let mut pending = accessions.iter().collect::<Vec<_>>();
    while !pending.is_empty() {
        let (ready, waiting) = pending.into_iter().partition::<Vec<_>, _>(|accession| {
            accession.parent_accession_id.is_none_or(|parent| {
                copies.contains_key(&parent) || !accessions.iter().any(|other| other.id == parent)
            })
        });
        if ready.is_empty() {
            return Err(PyRuntimeError::new_err(
                "failed to copy annotations: accession parents form a cycle",
            ));
        }
        for accession in ready {
            let parent = accession
                .parent_accession_id
                .map(|parent| copies.get(&parent).copied().unwrap_or(parent));
            let spans = accession_spans(conn, &accession.id);
            if let Some(duplicate) = find_duplicate(conn, &kept, parent, &accession.name, &spans) {
                copies.insert(accession.id, duplicate.id);
                continue;
            }
            let taken = taken_names.entry(parent).or_default();
            let name = unclaimed_name(&accession.name, taken);
            taken.insert(name.clone());
            let renamed = name != accession.name;
            let copy = Accession::create(
                conn,
                &NewAccession {
                    name: name.clone(),
                    block_group_id: target.id,
                    parent_accession_id: parent,
                    spans,
                },
            )
            .map_err(|error| write_error(&error))?;
            let annotations = Annotation::select(conn)
                .accession_id(accession.id)
                .load()
                .map_err(|error| read_error(&error))?;
            for annotation in &annotations {
                // An annotation named after its accession follows the accession's new name, so an
                // annotation added by hand as 'P' reads as "P'" once moved rather than keeping the
                // name it clashed with. Imported names are their own and stay as they are.
                Annotation::create(
                    conn,
                    &annotation.group,
                    &NewAnnotation {
                        name: if renamed && annotation.name == accession.name {
                            &name
                        } else {
                            &annotation.name
                        },
                        accession_id: copy.id,
                        extra: annotation.extra.as_ref(),
                    },
                )
                .map_err(|error| write_error(&error))?;
            }
            copies.insert(accession.id, copy.id);
        }
        pending = waiting;
    }

    Ok(())
}

/// Removes `accessions` with the annotations and node spans that belong to them.
fn delete_accessions(conn: &GraphConnection, accessions: &[Accession]) -> PyResult<()> {
    let accession_ids = accessions
        .iter()
        .map(|accession| accession.id)
        .collect::<Vec<_>>();
    for accession_id in &accession_ids {
        let annotation_ids = Annotation::select(conn)
            .accession_id(*accession_id)
            .load()
            .map_err(|error| read_error(&error))?
            .into_iter()
            .map(|annotation| annotation.id)
            .collect::<Vec<_>>();
        Annotation::select(conn)
            .delete_by_ids(annotation_ids)
            .map_err(|error| write_error(&error))?;
        let node_ids = Accession::get_nodes_by_id(conn, accession_id, None)
            .into_iter()
            .map(|node| node.id)
            .collect::<Vec<_>>();
        AccessionNode::select(conn)
            .delete_by_ids(node_ids)
            .map_err(|error| write_error(&error))?;
    }
    Accession::select(conn)
        .delete_by_ids(accession_ids)
        .map_err(|error| write_error(&error))?;
    Ok(())
}

/// Folds the sequence graph `other` into `block_group`, so a connection can join two molecules of
/// one sample. Its edges and annotations move over and the emptied graph is removed, leaving one
/// sequence graph in which each molecule is a component that the connection then links. This is
/// how a recombinase exchanges strands between two molecules without a second graph to track.
/// Its paths are not moved. An annotation name already used here for different spans takes
/// primes, unless `strict` reports it.
fn absorb_graph(
    conn: &GraphConnection,
    block_group: &BlockGroup,
    other: &PySequenceGraph,
    strict: bool,
) -> PyResult<()> {
    let absorbed = match BlockGroup::get_by_id(conn, &other.id, None) {
        Ok(absorbed) => absorbed,
        // An earlier connection already folded this graph in, and its positions now resolve here.
        Err(BlockGroupError::QueryError(QueryError::ResultsNotFound(_))) => return Ok(()),
        Err(error) => return Err(block_group_err_to_pyerr(error)),
    };
    if absorbed.collection_name != block_group.collection_name
        || absorbed.sample_name != block_group.sample_name
    {
        return Err(PyValueError::new_err(format!(
            "connect() joins sequence graphs of one sample, but '{}' is in sample '{}' and '{}' is in sample '{}'",
            block_group.name, block_group.sample_name, absorbed.name, absorbed.sample_name
        )));
    }
    move_annotations(conn, &absorbed, block_group, strict)?;

    // Both molecules keep sharing the graph's start and end. Chromatid indices stop meaning
    // anything across molecules, so each live edge on one moves as indeterminate: pruning would
    // otherwise keep only the newest start edge per chromatid and drop a molecule. Retired edit
    // sites stay retired.
    // A circular molecule's marker from the end to the start, and the edges that read it linearly
    // from the start to its first node and from its last node to the end, are left behind too: the
    // marker flags a whole graph as circular, and the linear reading is not a state a molecule
    // can be in once it exchanges strands. Its own loop edge still makes it circular.
    let rows = BlockGroupEdge::edges_for_block_group(conn, &absorbed.id, None);
    let closes_circle = |row: &AugmentedEdge| {
        is_end_node(row.edge.source_node_id) && is_start_node(row.edge.target_node_id)
    };
    let circular = rows.iter().any(closes_circle);
    let moved = rows
        .iter()
        .filter(|row| {
            !closes_circle(row)
                && !(circular
                    && (is_start_node(row.edge.source_node_id)
                        || is_end_node(row.edge.target_node_id)))
        })
        .map(|row| {
            let chromosome_index = if row.chromosome_index >= 0 {
                INDETERMINATE_CHROMOSOME_INDEX
            } else {
                row.chromosome_index
            };
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: row.edge.id,
                chromosome_index,
                phased: row.phased,
            }
        })
        .collect::<Vec<_>>();
    BlockGroupEdge::bulk_create(conn, &moved);
    // Paths read the absorbed graph's own start and end, which no longer exist as one linear
    // reading once the molecules exchange strands, so they are dropped with the graph.
    let paths = Path::select(conn)
        .block_group_id(absorbed.id)
        .load()
        .map_err(|error| PyRuntimeError::new_err(format!("failed to read paths: {error}")))?;
    for path in &paths {
        Path::delete(conn, &path.name, &absorbed.id);
    }
    BlockGroupEdge::bulk_delete(
        conn,
        &rows
            .iter()
            .map(|row| row_data(&absorbed.id, row))
            .collect::<Vec<_>>(),
    );
    BlockGroup::select(conn)
        .delete_by_ids([absorbed.id])
        .map_err(|error| PyRuntimeError::new_err(format!("failed to remove graph: {error}")))?;
    Ok(())
}

/// One new edge of a connection: the junction it leaves, the junction it arrives at, the ports it
/// is stored on, and whether it restores the continuity it would otherwise retire.
struct Crossing<'a> {
    leaving: &'a Junction,
    arriving: &'a Junction,
    source: Port,
    target: Port,
    restores: bool,
}

/// The original continuity a new edge replaces: the ports inside a node where it is split, and the
/// stored edges that left the first point or arrived at the second. Inside a node the continuity
/// is the sequence itself, so it is only ever replaced by cutting the node at the port.
fn continuity_at(
    graph: &GenGraph,
    stored: &[AugmentedEdge],
    crossing: &Crossing<'_>,
) -> (BTreeSet<Port>, HashSet<HashId>) {
    let Crossing {
        leaving,
        arriving,
        source,
        target,
        restores,
    } = *crossing;
    let mut split_ports = BTreeSet::new();
    let mut retired_edges = HashSet::new();
    for junction in [leaving, arriving] {
        let bounds = node_bounds(graph, junction.node_id);
        let port = Port {
            node_id: junction.node_id,
            coordinate: junction.boundary,
        };
        let first = bounds.first().copied().unwrap_or(junction.boundary);
        let last = bounds.last().copied().unwrap_or(junction.boundary);
        if first < junction.boundary && junction.boundary < last {
            if !restores {
                split_ports.insert(port);
            }
            continue;
        }
        // At the start or end of a node the original continuity is a stored edge.
        for row in stored.iter().filter(|row| {
            row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                && row.edge.source_strand == Strand::Forward
                && row.edge.target_strand == Strand::Forward
        }) {
            let leaves_end = junction.boundary == last && Port::source(&row.edge) == port;
            let arrives_at_start = junction.boundary == first && Port::target(&row.edge) == port;
            if leaves_end || arrives_at_start {
                retired_edges.insert(row.edge.id);
            }
        }
    }
    // Edges that left the same point or arrived at the same point are superseded too.
    for row in stored.iter().filter(|row| {
        row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
    }) {
        let leaves_source =
            Port::source(&row.edge) == source && row.edge.source_strand == leaving.strand;
        let arrives_at_target =
            Port::target(&row.edge) == target && row.edge.target_strand == arriving.strand;
        if leaves_source || arrives_at_target {
            retired_edges.insert(row.edge.id);
        }
    }
    (split_ports, retired_edges)
}

/// Writes one raw edge from after `after` to before `before`, each read on its own strand, as one
/// operation. This is the primitive for rearrangements that reconnect existing sequence, such as
/// the strand exchanges of a recombinase, without writing any new sequence.
///
/// Unless stacked, the original continuity at both points is retired: a node split at a point is
/// cut there by a retired marker edge, and the stored edges that left the first point or arrived
/// at the second are retired. A point whose far side is the sequence the edge joins to is a
/// restoration of the original continuity, which writes the marker edge live.
pub(crate) fn connect_positions(
    sequence_graph: &PySequenceGraph,
    joined: &[PySequenceGraph],
    after: &Position,
    before: &Position,
    message: Option<&str>,
    mode: EditMode,
    strict: bool,
) -> PyResult<()> {
    let EditMode { stack } = mode;
    let context = sequence_graph.require_context("connect")?;
    run_context_operation_write(
        context,
        |context| {
            let conn = context.graph().conn();
            let block_group = BlockGroup::get_by_id(conn, &sequence_graph.id, None)
                .map_err(block_group_err_to_pyerr)?;
            for other in joined {
                absorb_graph(conn, &block_group, other, strict)?;
            }
            let graph = unpruned_graph(context, &block_group.id)?;
            let source_block = require_blocks(&graph, &[*after, *before])?[0];
            let leaving = Junction::following(after);
            let arriving = Junction::preceding(before);

            let bounds_at = |junction: &Junction| {
                let mut bounds = node_bounds(&graph, junction.node_id);
                for other in [&leaving, &arriving] {
                    if other.node_id == junction.node_id {
                        bounds.insert(other.boundary);
                    }
                }
                bounds
            };
            let source = leaving.stored_port(true, &bounds_at(&leaving))?;
            let target = arriving.stored_port(false, &bounds_at(&arriving))?;
            let edge = EdgeData {
                source_node_id: source.node_id,
                source_coordinate: source.coordinate,
                source_strand: leaving.strand,
                target_node_id: target.node_id,
                target_coordinate: target.coordinate,
                target_strand: arriving.strand,
            };
            let restores = source == target
                && leaving.strand == Strand::Forward
                && arriving.strand == Strand::Forward;

            let stored = BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None);
            let (split_ports, mut retired_edges) = continuity_at(
                &graph,
                &stored,
                &Crossing {
                    leaving: &leaving,
                    arriving: &arriving,
                    source,
                    target,
                    restores,
                },
            );
            retired_edges.remove(&edge.id_hash());
            if stack {
                retired_edges.clear();
            }

            let plan = EditPlan {
                lefts: vec![],
                rights: vec![],
                entry_rows: vec![],
                exit_rows: vec![],
                skip_rows: Rows::new(),
                retired_edges,
                dissolved: None,
                cut_ports: if stack {
                    BTreeSet::new()
                } else {
                    split_ports.clone()
                },
                split_ports,
                connections: vec![(edge, routed_rows(&graph, source_block, Incoming))],
                purge_markers: false,
                stack,
            };
            write_edit(conn, &block_group.id, &plan, None)?;

            let summary = edit_summary(message, || {
                let absorbed = joined
                    .iter()
                    .map(|other| format!(" (joining '{}')", other.name))
                    .collect::<String>();
                format!(
                    "{}: connect {} -> {} in sample '{}'{absorbed}",
                    block_group.name,
                    leaving.describe(&source),
                    arriving.describe(&target),
                    block_group.sample_name
                )
            });
            Ok(((), summary))
        },
        edit_error,
    )
}

/// Whether the sequence graph is a circle, which is marked by an edge from its end back to its
/// start.
fn is_circular(conn: &GraphConnection, block_group_id: &HashId) -> bool {
    BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
        .iter()
        .any(|row| {
            row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                && is_end_node(row.edge.source_node_id)
                && is_start_node(row.edge.target_node_id)
        })
}

/// Where a graph's own start edges arrive, which says which molecule a product continues.
fn start_ports(conn: &GraphConnection, block_group_id: &HashId) -> HashSet<Port> {
    BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
        .iter()
        .filter(|row| {
            row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                && is_start_node(row.edge.source_node_id)
        })
        .map(|row| Port::target(&row.edge))
        .collect()
}

/// The graphs a final exchange started from, so each product keeps the name of the molecule whose
/// start it continues.
struct Origins {
    host_starts: HashSet<Port>,
    other: Option<(BlockGroup, HashSet<Port>)>,
}

fn find_root(parents: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = index;
    while parents[current] != root {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

/// Gives every molecule that lost its start and end a symbolic reading frame. An excised circle
/// has neither, so paths and coordinates would have nothing to count from. The start arrives at the
/// recombination site the molecule's new edge arrives at, and the end leaves from where that edge
/// leaves, so the new edge still closes the loop and the linear reading starts at the site.
fn add_terminals(
    conn: &GraphConnection,
    rows: &[AugmentedEdge],
    destinations: &HashMap<usize, BlockGroup>,
    connections: &[EdgeData],
    mut molecule_of: impl FnMut(&AugmentedEdge) -> usize,
) {
    let live = rows
        .iter()
        .filter(|row| row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .map(|row| (molecule_of(row), row))
        .collect::<Vec<_>>();
    let connection_ids = connections
        .iter()
        .map(|edge| edge.id_hash())
        .collect::<HashSet<_>>();
    for (root, destination) in destinations {
        let own = live.iter().filter(|(owner, _)| owner == root);
        if own.clone().any(|(_, row)| is_start_node(row.edge.source_node_id)) {
            continue;
        }
        let Some((_, closing)) = own
            .clone()
            .find(|(_, row)| connection_ids.contains(&row.edge.id))
        else {
            continue;
        };
        let edge = &closing.edge;
        let start = EdgeData {
            source_node_id: PATH_START_NODE_ID,
            source_coordinate: 0,
            source_strand: Strand::Forward,
            target_node_id: edge.target_node_id,
            target_coordinate: edge.target_coordinate,
            target_strand: edge.target_strand,
        };
        let end = EdgeData {
            source_node_id: edge.source_node_id,
            source_coordinate: edge.source_coordinate,
            source_strand: edge.source_strand,
            target_node_id: PATH_END_NODE_ID,
            target_coordinate: 0,
            target_strand: Strand::Forward,
        };
        let created = Edge::bulk_create(conn, &[start, end])
            .into_iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: destination.id,
                edge_id,
                chromosome_index: INDETERMINATE_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &created);
    }
}

/// Gives each molecule left in `block_group` its own sequence graph. After a final exchange two
/// linear molecules are two molecules, and an excision leaves a line and a circle, while an
/// inversion or an integration leaves one. The molecules are the connected parts of the graph the
/// builder makes from the edges that were there before the exchange, joined by the exchange's new
/// edges, so what stays together is decided by the same blocks and junctions that read the graph.
///
/// The piece continuing the host's start keeps the host, the piece continuing the absorbed graph's
/// start gets that graph's name back, and any other piece, such as an excised circle, is named
/// after the host with a number. Edges and annotations follow their molecule; an annotation that
/// spans a breakpoint is kept by every molecule it touches, as an incomplete locus. Paths read
/// edges now spread over several graphs, so they are dropped.
fn split_molecules(
    context: &DbContext,
    block_group: &BlockGroup,
    origins: &Origins,
    new_edges: &[EdgeData],
) -> PyResult<Vec<BlockGroup>> {
    let conn = context.graph().conn();
    let rows = BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None);
    // The builder says what stays together: blocks are carved at every edge, the new ones
    // included, but the graph is built from the edges that were already there, so the pieces are
    // its connected parts. The new edges then only bridge pieces, and joining pieces by them
    // gives the molecules. Building with the new edges would add the junctions that re-create the
    // parental routes.
    let live = rows
        .iter()
        .filter(|row| row.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .cloned()
        .collect::<Vec<_>>();
    let new_ids = new_edges
        .iter()
        .map(EdgeData::id_hash)
        .collect::<HashSet<_>>();
    let blocks = Edge::blocks_from_edges(conn, context.workspace(), &block_group.id, &live, None)
        .map_err(|error| PyRuntimeError::new_err(format!("failed to carve blocks: {error}")))?;
    let earlier = live
        .iter()
        .filter(|row| !new_ids.contains(&row.edge.id))
        .cloned()
        .collect::<Vec<_>>();
    let (graph, _) = Edge::build_graph(&earlier, &blocks);
    let indices = graph
        .nodes()
        .enumerate()
        .map(|(index, node)| (node, index))
        .collect::<HashMap<_, _>>();
    let mut parents = (0..indices.len()).collect::<Vec<_>>();
    let mut linked = HashSet::<usize>::new();
    let mut anchors = HashMap::<HashId, GraphNode>::new();
    for (source, target, weights) in graph.all_edges() {
        linked.extend(
            [indices.get(&source), indices.get(&target)]
                .into_iter()
                .flatten()
                .copied(),
        );
        for weight in weights {
            anchors.insert(
                weight.edge_id,
                if is_terminal(source.node_id) {
                    target
                } else {
                    source
                },
            );
        }
        if !is_terminal(source.node_id) && !is_terminal(target.node_id) {
            let (source_root, target_root) = (
                find_root(&mut parents, indices[&source]),
                find_root(&mut parents, indices[&target]),
            );
            parents[source_root] = target_root;
        }
    }
    let block_at = |node_id: HashId, coordinate: i64, ending: bool| {
        blocks
            .iter()
            .filter(|block| {
                block.node_id == node_id
                    && if ending {
                        block.end == coordinate
                    } else {
                        block.start == coordinate
                    }
            })
            .max_by_key(|block| block.end - block.start)
            .map(|block| GraphNode {
                node_id: block.node_id,
                sequence_start: block.start,
                sequence_end: block.end,
            })
    };
    for edge in new_edges {
        let ends = (
            block_at(edge.source_node_id, edge.source_coordinate, true),
            block_at(edge.target_node_id, edge.target_coordinate, false),
        );
        if let (Some(source), Some(target)) = ends {
            let (source_root, target_root) = (
                find_root(&mut parents, indices[&source]),
                find_root(&mut parents, indices[&target]),
            );
            parents[source_root] = target_root;
            linked.extend([indices[&source], indices[&target]]);
            anchors.insert(edge.id_hash(), source);
        }
    }
    let mut molecules = BTreeMap::<usize, Vec<GraphNode>>::new();
    for (node, index) in &indices {
        if !is_terminal(node.node_id) {
            molecules
                .entry(find_root(&mut parents, *index))
                .or_default()
                .push(*node);
        }
    }
    // A stretch no stored edge reaches, such as what a deletion skipped, is not a molecule.
    molecules.retain(|_, nodes| {
        nodes
            .iter()
            .any(|node| !is_routing_block(*node) && linked.contains(&indices[node]))
    });
    if molecules.len() <= 1 {
        return Ok(vec![block_group.clone()]);
    }
    let owners = anchors
        .iter()
        .map(|(edge_id, node)| (*edge_id, find_root(&mut parents, indices[node])))
        .collect::<HashMap<_, _>>();
    let starting = |port: &Port| {
        indices
            .iter()
            .filter(|(node, _)| node.node_id == port.node_id && node.sequence_start == port.coordinate)
            .map(|(_, index)| *index)
            .collect::<Vec<_>>()
    };

    let root_of = |parents: &mut Vec<usize>, ports: &HashSet<Port>| {
        ports
            .iter()
            .flat_map(&starting)
            .map(|index| find_root(parents, index))
            .filter(|root| molecules.contains_key(root))
            .min()
    };
    let host_root = root_of(&mut parents, &origins.host_starts)
        .or_else(|| molecules.keys().next().copied())
        .expect("should have a molecule");
    let other_root = origins
        .other
        .as_ref()
        .and_then(|(_, starts)| root_of(&mut parents, starts))
        .filter(|root| *root != host_root);
    let mut destinations = HashMap::from([(host_root, block_group.clone())]);
    let mut extra = 2;
    for root in molecules.keys().filter(|root| **root != host_root) {
        let (name, parent, is_default) = match &origins.other {
            Some((other, _)) if other_root == Some(*root) => (
                other.name.clone(),
                other.parent_block_group_id,
                other.is_default,
            ),
            _ => {
                extra += 1;
                (
                    format!("{}-{}", block_group.name, extra - 1),
                    block_group.parent_block_group_id,
                    false,
                )
            }
        };
        let created = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &block_group.collection_name,
                sample_name: &block_group.sample_name,
                name: &name,
                parent_block_group_id: parent.as_ref(),
                is_default,
            },
        )
        .map_err(block_group_err_to_pyerr)?;
        destinations.insert(*root, created);
    }

    let mut moved = BTreeMap::<HashId, Vec<&AugmentedEdge>>::new();
    for row in &rows {
        let root = owners.get(&row.edge.id).copied().unwrap_or(host_root);
        let destination = &destinations[&root];
        if destination.id != block_group.id {
            moved.entry(destination.id).or_default().push(row);
        }
    }
    for (destination, moving) in &moved {
        BlockGroupEdge::bulk_delete(
            conn,
            &moving
                .iter()
                .map(|row| row_data(&block_group.id, row))
                .collect::<Vec<_>>(),
        );
        BlockGroupEdge::bulk_create(
            conn,
            &moving
                .iter()
                .map(|row| row_data(destination, row))
                .collect::<Vec<_>>(),
        );
    }
    add_terminals(conn, &rows, &destinations, new_edges, |row| {
        owners.get(&row.edge.id).copied().unwrap_or(host_root)
    });
    let paths = Path::select(conn)
        .block_group_id(block_group.id)
        .load()
        .map_err(|error| PyRuntimeError::new_err(format!("failed to read paths: {error}")))?;
    for path in &paths {
        Path::delete(conn, &path.name, &block_group.id);
    }

    let accessions = Accession::select(conn)
        .block_group_id(block_group.id)
        .load()
        .map_err(|error| read_error(&error))?;
    let root_accession = |accession: &Accession| {
        let mut current = accession;
        while let Some(parent) = current.parent_accession_id {
            match accessions.iter().find(|other| other.id == parent) {
                Some(found) => current = found,
                None => break,
            }
        }
        current.id
    };
    let touches = |accession_id: &HashId, segments: &[GraphNode]| {
        Accession::get_nodes_by_id(conn, accession_id, None)
            .iter()
            .any(|node| {
                segments.iter().any(|segment| {
                    segment.node_id == node.node_id
                        && node.sequence_start < segment.sequence_end
                        && segment.sequence_start < node.sequence_end
                })
            })
    };
    let touching = |roots: &HashSet<HashId>, segments: &[GraphNode]| {
        roots
            .iter()
            .copied()
            .filter(|root| touches(root, segments))
            .collect::<HashSet<_>>()
    };
    let roots = accessions
        .iter()
        .map(&root_accession)
        .collect::<HashSet<_>>();
    let mut hosted = HashSet::new();
    for (root, destination) in &destinations {
        let kept = touching(&roots, &molecules[root]);
        if destination.id == block_group.id {
            hosted = kept;
            continue;
        }
        let tree = accessions
            .iter()
            .filter(|accession| kept.contains(&root_accession(accession)))
            .cloned()
            .collect::<Vec<_>>();
        copy_annotations(conn, &block_group.name, &tree, destination, false)?;
    }
    // A tree with no span at all has no molecule to follow, so the host keeps it.
    let unspanned = roots
        .iter()
        .filter(|root| Accession::get_nodes_by_id(conn, root, None).is_empty())
        .copied();
    hosted.extend(unspanned);
    let dropped = accessions
        .iter()
        .filter(|accession| !hosted.contains(&root_accession(accession)))
        .cloned()
        .collect::<Vec<_>>();
    delete_accessions(conn, &dropped)?;

    let mut products = molecules
        .keys()
        .map(|root| destinations[root].clone())
        .collect::<Vec<_>>();
    products.sort_by_key(|product| product.id != block_group.id);
    Ok(products)
}

/// A cut across the sequence, named by the two nucleotides it falls between, read upstream to
/// downstream. Whether the graph joins them is checked, not assumed: inside a node on one strand it
/// always does, and elsewhere it takes an edge, which may cross strands. The graph has to join them: as neighbors within a node, or by an edge,
/// which is what tells one variant's junction from another's. The cut is blunt, so the other strand
/// is cut at the same place. A strand-specific overhang would name the second strand's cut
/// separately, offset from this one, and the same two positions would still pin the junction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Breakpoint {
    pub upstream: Position,
    pub downstream: Position,
}

impl Breakpoint {
    /// The same cut on the opposite strand: the two nucleotides swap places and flip.
    const fn reverse_complement(&self) -> Self {
        Self {
            upstream: complement(&self.downstream),
            downstream: complement(&self.upstream),
        }
    }
}

/// The two breakpoints an exchange crosses at. The first is `A|B` and the second `C|D`, and the
/// exchange joins `A` to `D` and `C` to `B`, which is the only pairing it makes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExchangeSites {
    pub first: Breakpoint,
    pub second: Breakpoint,
}

/// Whether `downstream` follows `upstream` in reading order: the next coordinate within a block,
/// or the first position past an edge of the graph, which may run through routing blocks. An edge
/// read on the other strand joins the same two positions, so it counts in either direction.
fn is_adjacent(graph: &GenGraph, upstream: &Position, downstream: &Position) -> bool {
    let (Some(from), Some(to)) = (upstream.block(graph), downstream.block(graph)) else {
        return false;
    };
    let reverse = |position: &Position| position.strand == Strand::Reverse;
    if from == to && upstream.strand == downstream.strand {
        let step = if reverse(upstream) { -1 } else { 1 };
        if downstream.coordinate == upstream.coordinate + step {
            return true;
        }
    }
    let leaves = if reverse(upstream) {
        from.sequence_start
    } else {
        from.sequence_end - 1
    };
    let arrives = if reverse(downstream) {
        to.sequence_end - 1
    } else {
        to.sequence_start
    };
    upstream.coordinate == leaves
        && downstream.coordinate == arrives
        && (joined(graph, from, to) || joined(graph, to, from))
}

/// Whether an edge leads from `from` to `to`, crossing only routing blocks on the way.
fn joined(graph: &GenGraph, from: GraphNode, to: GraphNode) -> bool {
    let mut pending = vec![from];
    let mut seen = HashSet::from([from]);
    while let Some(block) = pending.pop() {
        for next in graph.neighbors_directed(block, Outgoing) {
            if next == to {
                return true;
            }
            if is_routing_block(next) && seen.insert(next) {
                pending.push(next);
            }
        }
    }
    false
}

/// Exchanges the sequence across two breakpoints, as one operation. The first new edge leaves the
/// downstream side of the `source` nucleotide, `A` in the context `A|C`, and arrives at the upstream
/// side of the `target` nucleotide, `B` in the context `D|B`. The second joins `D` to `C`, so the
/// two edges are the products of a recombinase crossing over: a strand exchange between two
/// molecules, or between two points of one. Which strand each position is read on decides what the
/// exchange does to the sequence between them: an inversion, an excision, or a crossover.
///
/// Unless stacked, the original continuity at both breakpoints is deleted from the graph, and no
/// retired marker is left in its place, so the originals stay reachable only through history.
/// Stacked, the originals stay as an alternative, so both the exchange and what it replaced read.
/// A position attached to another sequence graph of the same sample folds that graph into this one
/// first.
///
/// Every molecule the exchange leaves stays in this one graph, like the ones a stacked exchange
/// keeps, unless `split` gives each its own graph. Without the split the retired continuity can
/// still be drawn as history, and the call returns only this graph.
pub(crate) fn exchange_positions(
    sequence_graph: &PySequenceGraph,
    other: Option<&PySequenceGraph>,
    sites: &ExchangeSites,
    message: Option<&str>,
    mode: EditMode,
    strict: bool,
    split: bool,
) -> PyResult<Vec<BlockGroup>> {
    let EditMode { stack } = mode;
    let context = sequence_graph.require_context("exchange")?;
    run_context_operation_write(
        context,
        |context| {
            let conn = context.graph().conn();
            let block_group = BlockGroup::get_by_id(conn, &sequence_graph.id, None)
                .map_err(block_group_err_to_pyerr)?;
            let both_linear = !is_circular(conn, &block_group.id)
                && other.is_none_or(|other| !is_circular(conn, &other.id));
            let origins = Origins {
                host_starts: start_ports(conn, &block_group.id),
                other: other.and_then(|other| {
                    BlockGroup::get_by_id(conn, &other.id, None)
                        .ok()
                        .map(|group| (group, start_ports(conn, &other.id)))
                }),
            };
            if let Some(other) = other {
                absorb_graph(conn, &block_group, other, strict)?;
            }
            let graph = unpruned_graph(context, &block_group.id)?;
            // Written entirely on the reverse strand, the exchange is the same as its mirror on
            // the forward strand, `(rc D, rc C)` and `(rc B, rc A)`, and only that spelling is
            // one a read from the start of the molecule can follow.
            let mirrored = ExchangeSites {
                first: sites.second.reverse_complement(),
                second: sites.first.reverse_complement(),
            };
            let sites = if sites.first.upstream.strand == Strand::Reverse
                && sites.second.downstream.strand == Strand::Reverse
            {
                &mirrored
            } else {
                sites
            };
            let (first, second) = (&sites.first, &sites.second);
            // Inverted sites in two linear molecules would join one arm to the other's reverse
            // complement. One product would then run from the end of a molecule backwards and
            // so have no start, and the other no end, and the graph has nowhere to put them.
            if other.is_some()
                && both_linear
                && first.upstream.strand != second.downstream.strand
            {
                return Err(PyValueError::new_err(
                    "exchange of inverted sites between two linear molecules is not supported: \
                     one product would have no start and the other no end",
                ));
            }
            require_blocks(
                &graph,
                &[
                    first.upstream,
                    first.downstream,
                    second.upstream,
                    second.downstream,
                ],
            )?;
            for breakpoint in [first, second] {
                if !is_adjacent(&graph, &breakpoint.upstream, &breakpoint.downstream) {
                    return Err(PyValueError::new_err(format!(
                        "exchange needs {} to follow {} in this graph, but they are not joined",
                        breakpoint.downstream.describe(),
                        breakpoint.upstream.describe()
                    )));
                }
            }
            // The breakpoints are `A|B` and `C|D`, joined as `A` to `D` and `C` to `B`. Below, `A`
            // is the source and `D` the target of the first edge, and `B` and `C` are the
            // successor and predecessor that the second edge joins.
            let (source, target) = (&first.upstream, &second.downstream);
            let (successor, predecessor) = (first.downstream, second.upstream);
            // The second edge joins `C` to `B`, which the read reaches as it is written unless
            // the two breakpoints lie on different strands of one molecule, as in an inversion.
            // There the read continues from `D` along its strand to the mirror of `B`, so the
            // second edge is the same junction written as the mirror of `B` into the mirror of
            // `C`. Between molecules, or around a circle, it is read as written.
            let mirror = source.strand != target.strand
                && (reaches_oriented(&graph, &successor, &complement(&predecessor))
                    || reaches_oriented(&graph, &complement(&predecessor), &successor));
            let reciprocal = if mirror {
                (complement(&successor), complement(&predecessor))
            } else {
                (predecessor, successor)
            };
            let joins = [(*source, *target), reciprocal];
            let junctions = joins
                .iter()
                .map(|(from, to)| (Junction::following(from), Junction::preceding(to)))
                .collect::<Vec<_>>();
            let bounds_at = |junction: &Junction| {
                let mut bounds = node_bounds(&graph, junction.node_id);
                for (from, to) in &junctions {
                    for other in [from, to] {
                        if other.node_id == junction.node_id {
                            bounds.insert(other.boundary);
                        }
                    }
                }
                bounds
            };
            let crossings = junctions
                .iter()
                .map(|(from, to)| {
                    Ok(Crossing {
                        leaving: from,
                        arriving: to,
                        source: from.stored_port(true, &bounds_at(from))?,
                        target: to.stored_port(false, &bounds_at(to))?,
                        restores: false,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let edges = crossings
                .iter()
                .map(|crossing| EdgeData {
                    source_node_id: crossing.source.node_id,
                    source_coordinate: crossing.source.coordinate,
                    source_strand: crossing.leaving.strand,
                    target_node_id: crossing.target.node_id,
                    target_coordinate: crossing.target.coordinate,
                    target_strand: crossing.arriving.strand,
                })
                .collect::<Vec<_>>();

            let stored = BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None);
            let mut split_ports = BTreeSet::new();
            let mut retired_edges = HashSet::new();
            for crossing in &crossings {
                let (ports, retired) = continuity_at(&graph, &stored, crossing);
                split_ports.extend(ports);
                retired_edges.extend(retired);
            }
            // A split gives each molecule a graph of its own, so the sequence is cut at each split
            // port by the edges that now leave and arrive there, and a marker that kept it
            // continuous is deleted rather than retired. Without one, the molecules share this
            // graph and the retired continuity stays in it as history, as `connect` leaves it.
            let cut_ports = if stack {
                retired_edges.clear();
                BTreeSet::new()
            } else if split {
                retired_edges.extend(split_ports.iter().map(|port| port.edge_to(*port).id_hash()));
                split_ports.clear();
                BTreeSet::new()
            } else {
                split_ports.clone()
            };
            for edge in &edges {
                retired_edges.remove(&edge.id_hash());
            }

            let plan = EditPlan {
                lefts: vec![],
                rights: vec![],
                entry_rows: vec![],
                exit_rows: vec![],
                skip_rows: Rows::new(),
                retired_edges,
                dissolved: None,
                cut_ports,
                split_ports,
                connections: edges
                    .iter()
                    .map(|edge| (*edge, Rows::from([INDETERMINATE_CHROMOSOME_INDEX])))
                    .collect(),
                purge_markers: split,
                stack,
            };
            write_edit(conn, &block_group.id, &plan, None)?;
            let products = if stack || !split {
                vec![block_group.clone()]
            } else {
                split_molecules(context, &block_group, &origins, &edges)?
            };

            let summary = edit_summary(message, || {
                let joined = other
                    .map(|other| format!(" (joining '{}')", other.name))
                    .unwrap_or_default();
                format!(
                    "{}: exchange {} -> {} in sample '{}'{joined}",
                    block_group.name,
                    crossings[0].leaving.describe(&crossings[0].source),
                    crossings[0].arriving.describe(&crossings[0].target),
                    block_group.sample_name
                )
            });
            Ok((products, summary))
        },
        edit_error,
    )
}

/// The same base read on the opposite strand.
const fn complement(position: &Position) -> Position {
    Position {
        strand: match position.strand {
            Strand::Forward => Strand::Reverse,
            Strand::Reverse => Strand::Forward,
            other => other,
        },
        ..*position
    }
}

/// Marks a cut in the node at the point just after (or just before) `position`, as one operation.
///
/// A cut is a block boundary that a later join can attach to, such as the end a restriction enzyme
/// leaves. It does not break the graph: the sequence stays continuous across it, so the graph is
/// still a supergraph of the molecules before and after the cut. A retired marker edge records the
/// cut, and a live one beside it keeps the sequence continuous if another edit splits the node at
/// the same point. A cut is read on the top strand and taken as blunt on both strands, so the
/// staggered ends of an overhang are not yet separate edges.
pub(crate) fn cut_position(
    sequence_graph: &PySequenceGraph,
    position: &Position,
    after: bool,
    message: Option<&str>,
) -> PyResult<()> {
    let context = sequence_graph.require_context("cut")?;
    run_context_operation_write(
        context,
        |context| {
            let conn = context.graph().conn();
            let block_group = BlockGroup::get_by_id(conn, &sequence_graph.id, None)
                .map_err(block_group_err_to_pyerr)?;
            let graph = unpruned_graph(context, &block_group.id)?;
            require_blocks(&graph, &[*position])?;
            let junction = if after {
                Junction::following(position)
            } else {
                Junction::preceding(position)
            };
            let bounds = node_bounds(&graph, junction.node_id);
            let first = bounds.first().copied().unwrap_or(junction.boundary);
            let last = bounds.last().copied().unwrap_or(junction.boundary);
            if junction.boundary <= first || last <= junction.boundary {
                return Err(PyValueError::new_err(format!(
                    "cut needs a point inside a node, but {} is at the end of one",
                    describe_point(&Port {
                        node_id: junction.node_id,
                        coordinate: junction.boundary,
                    }),
                )));
            }

            let port = Port {
                node_id: junction.node_id,
                coordinate: junction.boundary,
            };
            let marker = port.edge_to(port);
            let existing = BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None)
                .into_iter()
                .filter(|row| row.edge.id == marker.id_hash())
                .collect::<Vec<_>>();
            let cut_before = existing
                .iter()
                .any(|row| row.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX);
            if cut_before {
                return Err(PyValueError::new_err(format!(
                    "{} is already cut",
                    describe_point(&port)
                )));
            }
            let row = |chromosome_index| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: marker.id_hash(),
                chromosome_index,
                phased: 0,
            };
            let mut created = vec![row(PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)];
            if existing.is_empty() {
                created.push(row(0));
            }
            Edge::bulk_create(conn, &[marker]);
            BlockGroupEdge::bulk_create(conn, &created);

            let summary = edit_summary(message, || {
                format!(
                    "{}: cut at {} in sample '{}'",
                    block_group.name,
                    describe_point(&port),
                    block_group.sample_name
                )
            });
            Ok(((), summary))
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
    /// Raw edges written as given, with the rows each is recorded under, for a connection or an
    /// exchange.
    connections: Vec<(EdgeData, Rows)>,
    /// Whether a retired marker edge is deleted outright. Otherwise it keeps a row on the index
    /// that marks it retired, so a cut stays recorded where an edit replaced the sequence at it.
    purge_markers: bool,
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
        if !self.connections.is_empty() {
            return self
                .connections
                .iter()
                .map(|(edge, rows)| (*edge, inherited(rows)))
                .collect();
        }
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
    let purged = |data: &BlockGroupEdgeData| {
        plan.purge_markers
            && existing
                .iter()
                .any(|existing_row| existing_row.edge.id == data.edge_id && is_marker(existing_row))
    };
    for data in &doomed {
        let kept = data.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
            || live_after.contains(&data.edge_id)
            || retired_before.contains(&data.edge_id)
            || purged(data);
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
        connections: vec![],
        purge_markers: false,
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
    pub mode: EditMode,
}

/// The switch every editing method takes to choose how an edit is routed.
#[derive(Clone, Copy, Default)]
pub(crate) struct EditMode {
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
            let path_before = if request.mode.stack {
                None
            } else {
                current_path(conn, &source.id)?
            };

            let stored = stored_sequence(request.sequence, span.is_reverse);
            let existing = BlockGroupEdge::edges_for_block_group(conn, &source.id, None);
            let lone = if request.mode.stack {
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
                connections: vec![],
                purge_markers: false,
                stack: request.mode.stack,
            };
            if !request.mode.stack {
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

/// Writes `edge_ids` as an immutable path derived from its parent.
fn create_path(conn: &GraphConnection, path: &Path, edge_ids: Vec<HashId>) -> PyResult<()> {
    let name = Path::generate_path_name(&path.name, &edge_ids);
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

    use r#gen::test_helpers::{TestContext, setup_gen_on_disk};
    use gen_core::{
        HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand,
        range::Range,
    };
    use gen_models::{
        accession::{Accession, AccessionSpan, NewAccession},
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
        editing::{
            EditKind, EditMode, EditRequest, InsertSite, edit_sequence_graph, insert_at_positions,
        },
        graph_search::PyGraphLocus,
        position::Position,
    };

    const START: &str = "start";
    const END: &str = "end";

    /// A sequence graph whose nodes are named by their sequences.
    struct Fixture {
        graph: PySequenceGraph,
        nodes: HashMap<&'static str, HashId>,
        context: TestContext,
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
            self.insert_with(site, sequence, EditMode { stack })
        }

        fn insert_with(
            &self,
            site: InsertSite<'_>,
            sequence: &str,
            mode: EditMode,
        ) -> PyResult<GraphLocus> {
            Python::initialize();
            insert_at_positions(&self.graph, &site, sequence, None, mode)
        }

        /// Replaces or deletes the sequence of `locus`, passed in as a Python `Locus`.
        fn edit(
            &self,
            locus: GraphLocus,
            kind: EditKind,
            sequence: &str,
        ) -> PyResult<Option<GraphLocus>> {
            self.edit_with(locus, kind, sequence, EditMode::default())
        }

        /// Like `edit`, with the stacking and path switches given.
        fn edit_with(
            &self,
            locus: GraphLocus,
            kind: EditKind,
            sequence: &str,
            mode: EditMode,
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
                        mode,
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
                        mode: EditMode { stack: true },
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
        context: &TestContext,
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
            graph: PySequenceGraph {
                id: block_group.id,
                collection_name: "collection".to_string(),
                sample_name: "sample".to_string(),
                name: name.to_string(),
                context: Some(context.db_context().clone()),
            },
            nodes,
            context: context.clone(),
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
                        context: context.db_context().clone(),
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
    assert len(annotation.locus.sequence) == 3
    assert len(annotation.locus.start().node.sequence) >= 3
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
before = annotation.locus
locus = annotation.locus
[whole] = graph.search('AAACCCGGGTTTAAACCCGGGTTT', sequence_kind='exact')
graph.insert('AG', after=whole.start())
assert annotation.locus == before
assert next(item for item in graph.annotations if item.id == annotation.id).locus == before
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
        #[ignore = "Needs Path::validate_ordered_edges to accept edges that meet at the same coordinate; that relaxation is a separate PR. Re-enable when it lands. later targets are regions, which resolve against the current path."]
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
assert len({target[index].node for index in range(len(target))}) == 3
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
            editing::{EditKind, EditMode, EditRequest, InsertSite, apply_edit},
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
                            mode: EditMode { stack },
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
            editing::{EditKind, EditMode, InsertSite},
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
        fn test_stepping_crosses_routing_blocks_without_consuming_bases() {
            let fixture = bubble();
            fixture
                .edit_with(
                    slice(&fixture, "MNOP", 0, 2),
                    EditKind::Delete,
                    "",
                    EditMode { stack: false },
                )
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
        use gen_core::Strand;
        use gen_graph::{GraphNode, GraphNodeSlice};
        use gen_models::{block_group::BlockGroup, locus::GraphLocus, path::Path};

        use super::{Fixture, bubble, error_message, strings};
        use crate::python_api::editing::{EditKind, InsertSite, create_path};

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
        fn test_repeated_python_path_creation_reuses_the_same_name() {
            let fixture = bubble();
            let conn = fixture.context.graph().conn();
            let parent_path = BlockGroup::get_current_path(conn, &fixture.graph.id, None)
                .expect("should load the imported path");
            let edge_ids = Path::edge_ids_for_path(conn, &parent_path.id, None);
            let generated_name = Path::generate_path_name(&parent_path.name, &edge_ids);
            let before = path_count(&fixture);

            create_path(conn, &parent_path, edge_ids.clone())
                .unwrap_or_else(|error| panic!("{}", error));
            let generated_path = Path::select(conn)
                .block_group_id(parent_path.block_group_id)
                .name(generated_name.clone())
                .load()
                .expect("should load the first generated path")
                .into_iter()
                .next()
                .expect("should find the first generated path");
            let stored_created_on = generated_path.created_on;
            assert_eq!(path_count(&fixture), before + 1);

            create_path(conn, &parent_path, edge_ids).unwrap_or_else(|error| panic!("{}", error));
            let generated_paths = Path::select(conn)
                .block_group_id(parent_path.block_group_id)
                .name(generated_name)
                .load()
                .expect("should load the repeated generated path");

            assert_eq!(generated_paths.len(), 1);
            assert_eq!(generated_paths[0].id, generated_path.id);
            assert_eq!(generated_paths[0].created_on, stored_created_on);
            assert_eq!(path_count(&fixture), before + 1);
        }

        #[test]
        fn test_replacing_with_a_previous_sequence_updates_the_current_path() {
            let fixture = bubble();
            let conn = fixture.context.graph().conn();
            let target = GraphLocus {
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
            let first_replacement = fixture
                .edit(target, EditKind::Replace, "WXYZ")
                .unwrap_or_else(|error| panic!("{}", error_message(error)))
                .expect("should return the first replacement locus");
            let first_path = BlockGroup::get_current_path(conn, &fixture.graph.id, None)
                .expect("should select the first WXYZ path");
            let first_edge_ids = Path::edge_ids_for_path(conn, &first_path.id, None);

            let intermediate_replacement = fixture
                .edit(first_replacement, EditKind::Replace, "EFGH")
                .unwrap_or_else(|error| panic!("{}", error_message(error)))
                .expect("should return the intermediate replacement locus");
            let intermediate_path = BlockGroup::get_current_path(conn, &fixture.graph.id, None)
                .expect("should select the intermediate EFGH path");
            assert_eq!(fixture.path_sequence(), "ABCDEFGHMNOP");

            // Exercise A -> B -> A (WXYZ -> EFGH -> WXYZ). Reusing the first A path would keep its
            // old created_on and leave B current; hashing the parent distinguishes the final A path.
            let _final_replacement = fixture
                .edit(intermediate_replacement, EditKind::Replace, "WXYZ")
                .unwrap_or_else(|error| panic!("{}", error_message(error)))
                .expect("should return the final replacement locus");
            let final_path = BlockGroup::get_current_path(conn, &fixture.graph.id, None)
                .expect("should select the final WXYZ path");
            let final_edge_ids = Path::edge_ids_for_path(conn, &final_path.id, None);

            assert_eq!(first_edge_ids, final_edge_ids);
            assert_ne!(first_path.id, final_path.id);
            assert_ne!(intermediate_path.id, final_path.id);
            assert_eq!(fixture.path_sequence(), "ABCDWXYZMNOP");
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
assert len(last) == 1 and last.graph is not None
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
assert arm_ends.graph is not None
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

    mod connections {
        use super::*;
        use gen_core::{is_end_node, is_start_node};
        use crate::python_api::{
            editing::{Breakpoint, ExchangeSites, connect_positions, exchange_positions},
            graph_read::unpruned_graph,
            position::{Neighbor, neighbors},
        };

        /// The breakpoints of an exchange between two nucleotides, taking what lies across each
        /// from a graph with no earlier event, where it is unique.
        fn between(
            source_in: &Fixture,
            source: Position,
            target_in: &Fixture,
            target: Position,
        ) -> ExchangeSites {
            let across = |fixture: &Fixture, position: &Position, forward: bool| {
                let graph = unpruned_graph(&fixture.context, &fixture.graph.id)
                    .expect("should build the graph");
                match neighbors(&graph, position, forward)
                    .expect("should step from the position")
                    .as_slice()
                {
                    [Neighbor::Position(neighbor)] => *neighbor,
                    other => panic!("expected one neighbor, found {other:?}"),
                }
            };
            ExchangeSites {
                first: Breakpoint {
                    upstream: source,
                    downstream: across(source_in, &source, true),
                },
                second: Breakpoint {
                    upstream: across(target_in, &target, false),
                    downstream: target,
                },
            }
        }

        const MOLECULE: &str = "AAACCGTTT";

        /// One node whose middle `CCG` lies between two cut points, as between two loxP sites.
        fn molecule() -> Fixture {
            fixture(&[(START, MOLECULE, 0), (MOLECULE, END, 0)], &[MOLECULE])
        }

        /// Two sites, each split in two nodes at the point a recombinase cuts, around `CAC` and
        /// flanked by `AAA` and `TTT`.
        fn sites() -> Fixture {
            fixture(
                &[
                    (START, "AAA", 0),
                    ("AAA", "GG", 0),
                    ("GG", "CC", 0),
                    ("CC", "CAC", 0),
                    ("CAC", "GA", 0),
                    ("GA", "TC", 0),
                    ("TC", "TTT", 0),
                    ("TTT", END, 0),
                ],
                &["AAA", "GG", "CC", "CAC", "GA", "TC", "TTT"],
            )
        }

        impl Fixture {
            fn connect(&self, after: Position, before: Position, stack: bool) {
                Python::initialize();
                connect_positions(
                    &self.graph,
                    &[],
                    &after,
                    &before,
                    None,
                    EditMode { stack },
                    false,
                )
                .expect("should connect the positions");
            }
        }

        impl Fixture {
            fn exchange(&self, source: Position, target: Position, stack: bool) -> Vec<BlockGroup> {
                Python::initialize();
                exchange_positions(
&self.graph,
None,
&between(self, source, self, target),
None,
EditMode { stack },
false,
true,
)
                .unwrap_or_else(|error| panic!("{}", error_message(error)))
            }

            fn sequences_of(&self, block_group: &BlockGroup) -> HashSet<String> {
                BlockGroup::get_all_sequences(
                    self.context.graph().conn(),
                    self.context.workspace(),
                    &block_group.id,
                    true,
                )
                .expect("should read sequences")
            }

            fn retired_rows(&self) -> usize {
                BlockGroupEdge::edges_for_block_group(
                    self.context.graph().conn(),
                    &self.graph.id,
                    None,
                )
                .into_iter()
                .filter(|row| row.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
                .count()
            }
        }

        fn invert(fixture: &Fixture, stack: bool) {
            fixture.connect(
                fixture.position(MOLECULE, 2),
                fixture.reverse_position(MOLECULE, 5),
                stack,
            );
            fixture.connect(
                fixture.reverse_position(MOLECULE, 3),
                fixture.position(MOLECULE, 6),
                stack,
            );
        }

        fn excise(fixture: &Fixture) {
            fixture.connect(fixture.position("GG", 1), fixture.position("TC", 0), false);
            fixture.connect(fixture.position("GA", 1), fixture.position("CC", 0), false);
        }

        #[test]
        fn test_connect_inverts_the_segment_between_two_points() {
            let fixture = molecule();

            invert(&fixture, false);

            assert_eq!(fixture.sequences(), strings(&["AAACGGTTT"]));
        }

        #[test]
        fn test_connect_back_to_the_original_neighbors_reverses_an_inversion() {
            let fixture = molecule();
            invert(&fixture, false);

            fixture.connect(
                fixture.position(MOLECULE, 2),
                fixture.position(MOLECULE, 3),
                false,
            );
            fixture.connect(
                fixture.position(MOLECULE, 5),
                fixture.position(MOLECULE, 6),
                false,
            );

            assert_eq!(fixture.sequences(), strings(&[MOLECULE]));
        }

        #[test]
        fn test_stacked_connect_keeps_the_original_route() {
            let fixture = molecule();

            invert(&fixture, true);
            assert_eq!(fixture.sequences(), strings(&[MOLECULE, "AAACGGTTT"]));
        }

        #[test]
        fn test_exchange_inverts_the_segment_between_two_points() {
            let fixture = molecule();

            fixture.exchange(
                fixture.position(MOLECULE, 2),
                fixture.reverse_position(MOLECULE, 5),
                false,
            );

            assert_eq!(fixture.sequences(), strings(&["AAACGGTTT"]));
        }

        #[test]
        fn test_exchange_inversion_written_from_the_reverse_strand_inverts_the_segment() {
            let fixture = molecule();

            fixture.exchange(
                fixture.reverse_position(MOLECULE, 3),
                fixture.position(MOLECULE, 6),
                false,
            );

            assert_eq!(fixture.sequences(), strings(&["AAACGGTTT"]));
        }

        #[test]
        fn test_exchange_inside_nodes_between_graphs_gives_both_recombinants() {
            let fixture = fixture(&[(START, "AAAGG", 0), ("AAAGG", END, 0)], &["AAAGG"]);
            let other = build_graph(
                &fixture.context,
                "chr2",
                &[(START, "TTTCC", 0), ("TTTCC", END, 0)],
                &["TTTCC"],
            );
            Python::initialize();

            let products = exchange_positions(
&fixture.graph,
Some(&other.graph),
&between(&fixture, fixture.position("AAAGG", 2), &other, other.position("TTTCC", 3)),
None,
EditMode { stack: false },
false,
true,
)
            .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(fixture.sequences_of(&products[0]), strings(&["AAACC"]));
            assert_eq!(fixture.sequences_of(&products[1]), strings(&["TTTGG"]));
            assert_eq!(fixture.retired_rows(), 0);
        }

        #[test]
        fn test_exchange_excision_leaves_the_circle_in_its_own_graph() {
            let fixture = sites();

            let products =
                fixture.exchange(fixture.position("GG", 1), fixture.position("TC", 0), false);

            assert_eq!(products.len(), 2);
            assert_eq!(products[0].name, "chr1");
            assert_eq!(products[1].name, "chr1-2");
            assert_eq!(fixture.sequences_of(&products[0]), strings(&["AAAGGTCTTT"]));
        }

        #[test]
        fn test_exchange_gives_the_excised_circle_a_start_and_an_end() {
            let fixture = sites();

            let products =
                fixture.exchange(fixture.position("GG", 1), fixture.position("TC", 0), false);

            let rows = BlockGroupEdge::edges_for_block_group(
                fixture.context.graph().conn(),
                &products[1].id,
                None,
            );
            let starts = rows
                .iter()
                .filter(|row| is_start_node(row.edge.source_node_id))
                .count();
            let ends = rows
                .iter()
                .filter(|row| is_end_node(row.edge.target_node_id))
                .count();
            assert_eq!((starts, ends), (1, 1));
            assert!(!fixture.sequences_of(&products[1]).is_empty());
        }

        #[test]
        fn test_exchange_inversion_leaves_one_graph() {
            let fixture = molecule();

            let products = fixture.exchange(
                fixture.position(MOLECULE, 2),
                fixture.reverse_position(MOLECULE, 5),
                false,
            );

            assert_eq!(products.len(), 1);
        }

        #[test]
        fn test_stacked_exchange_between_graphs_stays_one_graph() {
            let fixture = fixture(
                &[(START, "AAA", 0), ("AAA", "GG", 0), ("GG", END, 0)],
                &["AAA", "GG"],
            );
            let other = build_graph(
                &fixture.context,
                "chr2",
                &[(START, "TTT", 0), ("TTT", "CC", 0), ("CC", END, 0)],
                &["TTT", "CC"],
            );
            Python::initialize();

            let products = exchange_positions(
&fixture.graph,
Some(&other.graph),
&between(&fixture, fixture.position("AAA", 2), &other, other.position("CC", 0)),
None,
EditMode { stack: true },
false,
true,
)
            .unwrap_or_else(|error| panic!("{}", error_message(error)));

            assert_eq!(products.len(), 1);
            assert_eq!(
                fixture.sequences(),
                strings(&["AAAGG", "AAACC", "TTTCC", "TTTGG"])
            );
        }

        #[test]
        fn test_annotations_follow_the_molecule_they_touch() {
            let fixture = fixture(&[(START, "AAAGG", 0), ("AAAGG", END, 0)], &["AAAGG"]);
            let other = build_graph(
                &fixture.context,
                "chr2",
                &[(START, "TTTCC", 0), ("TTTCC", END, 0)],
                &["TTTCC"],
            );
            let conn = fixture.context.graph().conn();
            for (name, start, end) in [("left", 0, 2), ("spanning", 0, 5)] {
                Accession::create(
                    conn,
                    &NewAccession {
                        name: name.to_string(),
                        block_group_id: fixture.graph.id,
                        parent_accession_id: None,
                        spans: vec![AccessionSpan {
                            node_id: HashId::convert_str("AAAGG"),
                            range: Range { start, end },
                            strand: Strand::Forward,
                        }],
                    },
                )
                .expect("should create accession");
            }
            Python::initialize();

            let products = exchange_positions(
&fixture.graph,
Some(&other.graph),
&between(&fixture, fixture.position("AAAGG", 2), &other, other.position("TTTCC", 3)),
None,
EditMode { stack: false },
false,
true,
)
            .unwrap_or_else(|error| panic!("{}", error_message(error)));

            let names = |block_group: &BlockGroup| {
                let mut names = Accession::select(conn)
                    .block_group_id(block_group.id)
                    .load()
                    .expect("should read accessions")
                    .into_iter()
                    .map(|accession| accession.name)
                    .collect::<Vec<_>>();
                names.sort();
                names
            };
            assert_eq!(names(&products[0]), vec!["left", "spanning"]);
            assert_eq!(names(&products[1]), vec!["spanning"]);
        }

        #[test]
        fn test_exchange_excises_a_circle_in_the_same_graph() {
            let fixture = sites();

            fixture.exchange(fixture.position("GG", 1), fixture.position("TC", 0), false);

            assert_eq!(fixture.sequences(), strings(&["AAAGGTCTTT"]));
        }

        #[test]
        fn test_exchange_inside_a_node_leaves_no_retired_marker() {
            let fixture = molecule();

            fixture.exchange(
                fixture.position(MOLECULE, 2),
                fixture.reverse_position(MOLECULE, 5),
                false,
            );

            assert_eq!(fixture.retired_rows(), 0);
        }

        #[test]
        fn test_stacked_exchange_keeps_the_original_route() {
            let fixture = molecule();

            fixture.exchange(
                fixture.position(MOLECULE, 2),
                fixture.reverse_position(MOLECULE, 5),
                true,
            );

            assert_eq!(fixture.sequences(), strings(&[MOLECULE, "AAACGGTTT"]));
        }

        #[test]
        fn test_exchange_between_graphs_gives_both_recombinants() {
            let fixture = fixture(
                &[(START, "AAA", 0), ("AAA", "GG", 0), ("GG", END, 0)],
                &["AAA", "GG"],
            );
            let other = build_graph(
                &fixture.context,
                "chr2",
                &[(START, "TTT", 0), ("TTT", "CC", 0), ("CC", END, 0)],
                &["TTT", "CC"],
            );
            Python::initialize();

            let products = exchange_positions(
&fixture.graph,
Some(&other.graph),
&between(&fixture, fixture.position("AAA", 2), &other, other.position("CC", 0)),
None,
EditMode { stack: false },
false,
true,
)
            .unwrap_or_else(|error| panic!("{}", error_message(error)));

            // Each recombinant is its own graph, and the second one is the absorbed graph again.
            assert_eq!(fixture.sequences(), strings(&["AAACC"]));
            let names = products
                .iter()
                .map(|product| product.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(names, vec!["chr1", "chr2"]);
            assert_eq!(products[1].id, other.graph.id);
            let second = BlockGroup::get_all_sequences(
                fixture.context.graph().conn(),
                fixture.context.workspace(),
                &products[1].id,
                true,
            )
            .expect("should read sequences");
            assert_eq!(second, strings(&["TTTGG"]));
        }

        #[test]
        fn test_connect_excises_a_circle_in_the_same_graph() {
            let fixture = sites();

            excise(&fixture);
            assert_eq!(fixture.sequences(), strings(&["AAAGGTCTTT"]));
        }

        #[test]
        fn test_connect_reintegrates_an_excised_circle() {
            let fixture = sites();
            excise(&fixture);

            fixture.connect(fixture.position("GG", 1), fixture.position("CC", 0), false);
            fixture.connect(fixture.position("GA", 1), fixture.position("TC", 0), false);

            assert_eq!(fixture.sequences(), strings(&["AAAGGCCCACGATCTTT"]));
        }
    }

    mod joined_annotations {
        use super::*;
        use crate::python_api::editing::connect_positions;

        /// Two graphs of one sample that both hold the node `AAAC`, with `arm` annotated on
        /// `start..end` of that node in each.
        fn graphs_annotated(host: (i64, i64), other: (i64, i64)) -> (Fixture, Fixture) {
            let fixture = fixture(&[(START, "AAAC", 0), ("AAAC", END, 0)], &["AAAC"]);
            let second = build_graph(
                &fixture.context,
                "chr2",
                &[(START, "AAAC", 0), ("AAAC", "GGGT", 0), ("GGGT", END, 0)],
                &["AAAC", "GGGT"],
            );
            for (graph, (start, end)) in [(&fixture, host), (&second, other)] {
                Accession::create(
                    fixture.context.graph().conn(),
                    &NewAccession {
                        name: "arm".to_string(),
                        block_group_id: graph.graph.id,
                        parent_accession_id: None,
                        spans: vec![AccessionSpan {
                            node_id: HashId::convert_str("AAAC"),
                            range: Range { start, end },
                            strand: Strand::Forward,
                        }],
                    },
                )
                .expect("should create accession");
            }
            (fixture, second)
        }

        fn join(host: &Fixture, other: &Fixture, strict: bool) -> PyResult<()> {
            connect_positions(
                &host.graph,
                std::slice::from_ref(&other.graph),
                &host.position("AAAC", 3),
                &other.position("GGGT", 0),
                None,
                EditMode { stack: true },
                strict,
            )
        }

        fn names(host: &Fixture) -> Vec<String> {
            let mut names = Accession::select(host.context.graph().conn())
                .block_group_id(host.graph.id)
                .load()
                .expect("should read accessions")
                .into_iter()
                .map(|accession| accession.name)
                .collect::<Vec<_>>();
            names.sort();
            names
        }

        #[test]
        fn test_an_accession_over_the_same_spans_is_not_added_again() {
            let (host, other) = graphs_annotated((0, 3), (0, 3));

            join(&host, &other, true).unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(names(&host), vec!["arm".to_string()]);
        }

        #[test]
        fn test_an_accession_over_other_spans_takes_a_prime() {
            let (host, other) = graphs_annotated((0, 3), (1, 4));

            join(&host, &other, false).unwrap_or_else(|error| panic!("{}", error_message(error)));
            assert_eq!(names(&host), vec!["arm".to_string(), "arm'".to_string()]);
        }

        #[test]
        fn test_an_accession_over_other_spans_is_reported_when_strict() {
            let (host, other) = graphs_annotated((0, 3), (1, 4));

            let Err(error) = join(&host, &other, true) else {
                panic!("a clash over different spans must be reported when strict");
            };
            assert!(error_message(error).contains("annotation named 'arm'"));
        }
    }
}
