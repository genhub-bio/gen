//! Positions in a sequence graph, alone or in superposition.
//!
//! Python sees a `Position` as an offset into the node slice it was read from. Internally it is
//! addressed by node coordinate, so it survives the block re-carving later edits cause and can move
//! between sequence graphs that share nodes. A `Position` attached to a sequence graph with `.on()`
//! can step with `+`/`-`: it stays a `Position` while the step lands on a single point, and splits
//! into a `SuperPosition` once the step lands at a fork. A `SuperPosition` holds several positions at
//! once: stepping it further splits one position into one per route, and `|` combines the positions
//! of separate superpositions, so a single edit can attach to every variant a superposition covers.

use std::collections::HashSet;

use gen_core::{HashId, Strand, is_start_node, is_terminal};
use gen_graph::{GenGraph, GraphNode};
use petgraph::Direction::{Incoming, Outgoing};
use pyo3::{
    Bound, Py, PyAny, PyRef, PyResult, Python,
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    pyclass, pymethods,
    types::{PyAnyMethods as _, PyTuple, PyTupleMethods as _},
};

use super::{
    block_group::PySequenceGraph,
    graph_node::PyGraphNode,
    graph_read::{current_graph, forward_edge},
};

/// A point in a node's sequence: the node, an offset into its sequence, and the strand it is read
/// on. Inside a superposition the offset names the sequence character at that offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Position {
    pub node_id: HashId,
    pub coordinate: i64,
    pub strand: Strand,
}

impl Position {
    /// The block of `graph` holding this position, if the graph still reaches it.
    pub(crate) fn block(&self, graph: &GenGraph) -> Option<GraphNode> {
        graph.nodes().find(|block| {
            block.node_id == self.node_id
                && block.sequence_start <= self.coordinate
                && self.coordinate < block.sequence_end
        })
    }

    fn is_reverse(&self) -> bool {
        self.strand == Strand::Reverse
    }

    pub(crate) fn describe(&self) -> String {
        let node_id = self.node_id.to_string();
        format!(
            "{}:{}{}",
            &node_id[..8.min(node_id.len())],
            self.coordinate,
            if self.is_reverse() { '-' } else { '+' }
        )
    }
}

/// What lies one step away from a position in reading order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Neighbor {
    Position(Position),
    /// The graph's start or end node, reached by stepping off its first or last position.
    Terminal(GraphNode),
}

/// Resolves every position against `graph`, failing on the first one the graph does not reach.
pub(crate) fn require_blocks(graph: &GenGraph, positions: &[Position]) -> PyResult<Vec<GraphNode>> {
    positions
        .iter()
        .map(|position| {
            position.block(graph).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "invalid position: {} is not reachable in this sequence graph",
                    position.describe()
                ))
            })
        })
        .collect()
}

/// The positions one step from `position` in reading order, toward the reading end when `forward`.
///
/// Inside a block that is the next coordinate. At a block edge it is the first position of every
/// block the graph continues into, passing through zero-length junctions, so a fork yields several.
pub(crate) fn neighbors(
    graph: &GenGraph,
    position: &Position,
    forward: bool,
) -> PyResult<Vec<Neighbor>> {
    let block = require_blocks(graph, &[*position])?[0];
    let toward_node_end = forward != position.is_reverse();
    let next = if toward_node_end {
        position.coordinate + 1
    } else {
        position.coordinate - 1
    };
    if block.sequence_start <= next && next < block.sequence_end {
        return Ok(vec![Neighbor::Position(Position {
            coordinate: next,
            ..*position
        })]);
    }
    let direction = if toward_node_end { Outgoing } else { Incoming };
    let mut found = vec![];
    let mut pending = vec![block];
    let mut visited = HashSet::from([block]);
    while let Some(current) = pending.pop() {
        for neighbor in graph.neighbors_directed(current, direction) {
            let (source, target) = if toward_node_end {
                (current, neighbor)
            } else {
                (neighbor, current)
            };
            if !forward_edge(graph, source, target) {
                return Err(PyValueError::new_err(
                    "position cannot step across a reverse-strand edge in this sequence graph",
                ));
            }
            let step = if is_terminal(neighbor.node_id) {
                Neighbor::Terminal(neighbor)
            } else if neighbor.sequence_start == neighbor.sequence_end {
                if visited.insert(neighbor) {
                    pending.push(neighbor);
                }
                continue;
            } else {
                Neighbor::Position(Position {
                    node_id: neighbor.node_id,
                    coordinate: if toward_node_end {
                        neighbor.sequence_start
                    } else {
                        neighbor.sequence_end - 1
                    },
                    strand: position.strand,
                })
            };
            if !found.contains(&step) {
                found.push(step);
            }
        }
    }
    Ok(found)
}

/// Whether `target` can be reached from `source` along at least one edge.
pub(crate) fn reaches(graph: &GenGraph, source: GraphNode, target: GraphNode) -> bool {
    let mut pending = vec![source];
    let mut visited = HashSet::new();
    while let Some(current) = pending.pop() {
        for successor in graph.neighbors_directed(current, Outgoing) {
            if successor == target {
                return true;
            }
            if visited.insert(successor) {
                pending.push(successor);
            }
        }
    }
    false
}

/// Moves one position a step along its strand, toward the reading end when `forward`, splitting
/// into one position per route at a fork.
///
/// Only a single position steps, so a walk can split only on its last step and the positions of a
/// superposition never travel down arms of different lengths to meet again.
pub(crate) fn step(
    graph: &GenGraph,
    position: &Position,
    forward: bool,
) -> PyResult<Vec<Position>> {
    let mut stepped = vec![];
    for neighbor in neighbors(graph, position, forward)? {
        match neighbor {
            Neighbor::Position(neighbor) => stepped.push(neighbor),
            Neighbor::Terminal(terminal) => {
                return Err(PyIndexError::new_err(format!(
                    "position steps past the {} of the sequence graph",
                    if is_start_node(terminal.node_id) {
                        "start"
                    } else {
                        "end"
                    }
                )));
            }
        }
    }
    Ok(canonical(stepped))
}

fn canonical(mut positions: Vec<Position>) -> Vec<Position> {
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// A position in a sequence graph: a node slice, an offset within it, and the strand it is read on.
///
/// Returned by ``Locus.start()`` and ``Locus.end()``, which name the first and last position of the
/// locus in reading order. Two positions are equal when they name the same point of the same node,
/// even if later edits split that node so their ``node`` and ``offset`` differ. Pass a position to
/// ``GraphWidget.go_to()``, as ``after`` or ``before`` to ``SequenceGraph.insert()``, or to
/// ``SuperPosition()`` to step from it.
///
/// A position taken from a ``Locus`` is attached to the sequence graph that locus came from, and can
/// step with ``pos + n`` and ``pos - n``: the result stays a ``Position`` while the step lands on a
/// single point, and becomes a ``SuperPosition`` once the step lands at a fork. ``pos.on(sg)``
/// attaches it to another sequence graph, such as a copy of the sample. ``a | b`` combines a
/// position with another position or a ``SuperPosition`` into a ``SuperPosition``.
#[pyclass(name = "Position", unsendable)]
#[derive(Clone)]
pub struct PyPosition {
    /// The point by node coordinate, which later edits leave unchanged.
    pub(crate) position: Position,
    /// The block the position was read from; `position` falls in or bounds it.
    pub(crate) block: GraphNode,
    /// The sequence graph this position steps through, or ``None`` when unattached.
    pub(crate) sequence_graph: Option<PySequenceGraph>,
}

impl PyPosition {
    /// The position `local_offset` into `block`, read on `strand`.
    pub(crate) fn in_block(block: GraphNode, local_offset: usize, strand: Strand) -> Self {
        let local_offset = local_offset as i64;
        assert!(
            local_offset <= block.length(),
            "position should lie within the block it was read from"
        );
        Self {
            position: Position {
                node_id: block.node_id,
                coordinate: block.sequence_start + local_offset,
                strand: if strand == Strand::Reverse {
                    Strand::Reverse
                } else {
                    Strand::Forward
                },
            },
            block,
            sequence_graph: None,
        }
    }

    /// This position attached to `sequence_graph`, so it can step through it.
    pub(crate) fn attached_to(mut self, sequence_graph: Option<PySequenceGraph>) -> Self {
        self.sequence_graph = sequence_graph;
        self
    }

    /// `position` in the block of `graph` holding it.
    fn located(graph: &GenGraph, position: Position) -> PyResult<Self> {
        Ok(Self {
            position,
            block: require_blocks(graph, &[position])?[0],
            sequence_graph: None,
        })
    }

    /// This position's own sequence graph, needed to step it.
    ///
    /// Mirrors `PySuperPosition::require_graph`'s error wording.
    fn require_graph(&self) -> PyResult<PySequenceGraph> {
        self.sequence_graph.clone().ok_or_else(|| {
            PyValueError::new_err(
                "position has no sequence graph to step through; attach one with pos.on(sg)",
            )
        })
    }

    /// The position or superposition `delta` steps away, reusing `SuperPosition::stepped` by
    /// wrapping this position alone. Stays a `Position` when the step lands on a single point,
    /// and becomes a `SuperPosition` once the step lands at a fork.
    fn stepped(&self, python: Python<'_>, delta: i64) -> PyResult<Py<PyAny>> {
        let sequence_graph = self.require_graph()?;
        let superposition = PySuperPosition {
            positions: vec![self.clone()],
            sequence_graph: Some(sequence_graph.clone()),
        };
        let stepped = superposition.stepped(delta)?;
        match stepped.positions.as_slice() {
            [position] => {
                let mut position = position.clone();
                position.sequence_graph = Some(sequence_graph);
                Py::new(python, position).map(Py::into_any)
            }
            _ => Py::new(python, stepped).map(Py::into_any),
        }
    }

    /// The block of `graph` holding this position and the offset into it, preferring the block
    /// the position was read from while `graph` still has it.
    pub(crate) fn locate(&self, graph: &GenGraph) -> Option<(GraphNode, i64)> {
        let coordinate = self.position.coordinate;
        let block = Some(self.block)
            .filter(|block| graph.contains_node(*block))
            .or_else(|| self.position.block(graph))
            .or_else(|| {
                // A point just past the last block of its node.
                graph.nodes().find(|block| {
                    block.node_id == self.position.node_id && block.sequence_end == coordinate
                })
            })?;
        Some((block, coordinate - block.sequence_start))
    }
}

#[pymethods]
impl PyPosition {
    /// The node slice this position is in.
    #[getter]
    fn node(&self) -> PyGraphNode {
        PyGraphNode::new(
            self.block.node_id,
            self.block.sequence_start,
            self.block.sequence_end,
        )
    }

    /// Offset within ``node`` (``0..node.length``).
    #[getter]
    fn offset(&self) -> i64 {
        self.position.coordinate - self.block.sequence_start
    }

    /// Strand this position is read on: ``"+"`` or ``"-"``.
    #[getter]
    fn strand(&self) -> &'static str {
        if self.position.is_reverse() { "-" } else { "+" }
    }

    /// The sequence graph this position steps through, or ``None`` when unattached.
    #[getter]
    fn sequence_graph(&self) -> Option<PySequenceGraph> {
        self.sequence_graph.clone()
    }

    /// This position attached to ``sequence_graph``.
    ///
    /// Raises ``ValueError`` if the position is not reachable in that sequence graph.
    fn on(&self, sequence_graph: PyRef<'_, PySequenceGraph>) -> PyResult<Self> {
        let context = sequence_graph.require_context("Position.on")?;
        let graph = current_graph(context, &sequence_graph.id)?;
        let mut located = Self::located(&graph, self.position)?;
        located.sequence_graph = Some(sequence_graph.clone());
        Ok(located)
    }

    /// The position ``delta`` steps along this position's strand, backward for a negative
    /// ``delta``. The result stays a ``Position`` while the step lands on a single point, and
    /// becomes a ``SuperPosition`` once the step lands at a fork.
    ///
    /// Raises ``ValueError`` when this position has no sequence graph attached, and
    /// ``IndexError`` when the step leaves the sequence graph.
    fn __add__(&self, python: Python<'_>, delta: i64) -> PyResult<Py<PyAny>> {
        self.stepped(python, delta)
    }

    /// The position ``delta`` steps back along this position's strand; the same as
    /// ``pos + -delta``.
    fn __sub__(&self, python: Python<'_>, delta: i64) -> PyResult<Py<PyAny>> {
        self.stepped(python, -delta)
    }

    /// A superposition covering this position and ``other``, a ``Position`` or a
    /// ``SuperPosition``.
    ///
    /// Raises ``ValueError`` when the two are attached to different sequence graphs.
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySuperPosition> {
        PySuperPosition::from(self.clone()).union(&PySuperPosition::from_operand(other)?)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<PyPosition>>()
            .is_ok_and(|other| other.position == self.position)
    }

    fn __hash__(&self) -> isize {
        position_hash(0, &self.position)
    }

    fn __repr__(&self) -> String {
        format!("Position({})", self.__str__())
    }

    fn __str__(&self) -> String {
        let node_id = self.block.node_id.to_string();
        format!(
            "{}[{}:{}][{}]{}",
            &node_id[..8.min(node_id.len())],
            self.block.sequence_start,
            self.block.sequence_end,
            self.offset(),
            self.strand()
        )
    }
}

fn position_hash(hash: isize, position: &Position) -> isize {
    let mut hash = hash;
    for &byte in &position.node_id.0 {
        hash = hash.wrapping_mul(31).wrapping_add(byte as isize);
    }
    hash = hash
        .wrapping_mul(31)
        .wrapping_add(position.coordinate as isize);
    hash.wrapping_mul(31)
        .wrapping_add(isize::from(position.is_reverse()))
}

/// The positions an insert method is given, from a ``Position`` or a ``SuperPosition``.
pub(crate) fn positions_of(value: &Bound<'_, PyAny>) -> PyResult<Vec<Position>> {
    if let Ok(position) = value.extract::<PyRef<PyPosition>>() {
        Ok(vec![position.position])
    } else if let Ok(superposition) = value.extract::<PyRef<PySuperPosition>>() {
        Ok(superposition
            .positions
            .iter()
            .map(|position| position.position)
            .collect())
    } else {
        Err(PyTypeError::new_err(
            "expected a Position or a SuperPosition",
        ))
    }
}

fn canonical_positions(mut positions: Vec<PyPosition>) -> Vec<PyPosition> {
    positions.sort_unstable_by_key(|position| position.position);
    positions.dedup_by_key(|position| position.position);
    positions
}

/// Several positions in a sequence graph held at once.
///
/// Build one from positions with ``SuperPosition(pos, ...)`` or ``a | b``. Where the graph holds
/// several variants a superposition can cover a position on each of them, and ``a | b`` combines
/// the positions of two superpositions, or of a superposition and a position. ``sp + n`` steps a single position along its strand, splitting
/// it across every route leaving a fork, and ``sp - n`` steps back the same way. A superposition
/// that already covers several positions does not step, so a walk splits only on its last step.
///
/// Stepping needs a sequence graph. A superposition of positions from one locus, or from loci of
/// one sequence graph, is attached to it already; ``sp.on(sg)`` attaches it to another sequence
/// graph that shares its nodes, and is invalid there only if one of its positions is not reachable. Pass a superposition as ``after`` or ``before`` to
/// ``SequenceGraph.insert()`` to insert at every position it covers.
#[pyclass(name = "SuperPosition", unsendable)]
#[derive(Clone)]
pub struct PySuperPosition {
    pub(crate) positions: Vec<PyPosition>,
    pub(crate) sequence_graph: Option<PySequenceGraph>,
}

impl From<PyPosition> for PySuperPosition {
    fn from(position: PyPosition) -> Self {
        Self {
            sequence_graph: position.sequence_graph.clone(),
            positions: vec![position],
        }
    }
}

impl PySuperPosition {
    /// The superposition an operand of `|` stands for: a `Position` or a `SuperPosition`.
    fn from_operand(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(position) = value.extract::<PyRef<PyPosition>>() {
            Ok(Self::from(position.clone()))
        } else if let Ok(superposition) = value.extract::<PyRef<Self>>() {
            Ok(superposition.clone())
        } else {
            Err(PyTypeError::new_err(
                "expected a Position or a SuperPosition",
            ))
        }
    }

    /// The sequence graph every position is attached to, or `None` unless they all share one.
    fn shared_graph(positions: &[PyPosition]) -> Option<PySequenceGraph> {
        let first = positions.first()?.sequence_graph.as_ref()?;
        positions
            .iter()
            .all(|position| {
                position
                    .sequence_graph
                    .as_ref()
                    .is_some_and(|graph| graph.id == first.id)
            })
            .then(|| first.clone())
    }

    /// A superposition covering the positions of both.
    fn union(&self, other: &Self) -> PyResult<Self> {
        let sequence_graph = match (&self.sequence_graph, &other.sequence_graph) {
            (Some(ours), Some(theirs)) if ours.id != theirs.id => {
                return Err(PyValueError::new_err(
                    "cannot combine positions attached to different sequence graphs",
                ));
            }
            (Some(graph), _) | (None, Some(graph)) => Some(graph.clone()),
            (None, None) => None,
        };
        Ok(Self {
            positions: canonical_positions(
                self.positions
                    .iter()
                    .chain(&other.positions)
                    .cloned()
                    .collect(),
            ),
            sequence_graph,
        })
    }

    fn require_graph(&self) -> PyResult<(GenGraph, &PySequenceGraph)> {
        let sequence_graph = self.sequence_graph.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "superposition has no sequence graph to step through; attach one with sp.on(sg)",
            )
        })?;
        let context = sequence_graph.require_context("SuperPosition")?;
        Ok((current_graph(context, &sequence_graph.id)?, sequence_graph))
    }

    pub(crate) fn stepped(&self, delta: i64) -> PyResult<Self> {
        let (graph, sequence_graph) = self.require_graph()?;
        let mut positions = self.positions.clone();
        for _ in 0..delta.unsigned_abs() {
            let [position] = positions.as_slice() else {
                return Err(PyValueError::new_err(format!(
                    "superposition covers {} positions, so it cannot step; step each position on \
                     its own and combine the results with |",
                    positions.len()
                )));
            };
            positions = step(&graph, &position.position, delta > 0)?
                .into_iter()
                .map(|position| PyPosition::located(&graph, position))
                .collect::<PyResult<_>>()?;
        }
        Ok(Self {
            positions,
            sequence_graph: Some(sequence_graph.clone()),
        })
    }
}

#[pymethods]
impl PySuperPosition {
    #[new]
    #[pyo3(signature = (*positions))]
    fn new(positions: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let positions = positions
            .iter()
            .map(|position| {
                position
                    .extract::<PyRef<PyPosition>>()
                    .map(|position| position.clone())
                    .map_err(|_| PyTypeError::new_err("SuperPosition() takes Position objects"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        if positions.is_empty() {
            return Err(PyValueError::new_err(
                "SuperPosition() needs at least one Position",
            ));
        }
        Ok(Self {
            sequence_graph: Self::shared_graph(&positions),
            positions: canonical_positions(positions),
        })
    }

    /// Every ``Position`` this superposition covers.
    #[getter]
    fn positions(&self) -> Vec<PyPosition> {
        self.positions.clone()
    }

    /// The sequence graph this superposition steps through, or ``None`` when unattached.
    #[getter]
    fn sequence_graph(&self) -> Option<PySequenceGraph> {
        self.sequence_graph.clone()
    }

    /// This superposition attached to ``sequence_graph``.
    ///
    /// Raises ``ValueError`` if any of its positions is not reachable in that sequence graph.
    fn on(&self, sequence_graph: PyRef<'_, PySequenceGraph>) -> PyResult<Self> {
        let context = sequence_graph.require_context("SuperPosition.on")?;
        let graph = current_graph(context, &sequence_graph.id)?;
        Ok(Self {
            positions: self
                .positions
                .iter()
                .map(|position| PyPosition::located(&graph, position.position))
                .collect::<PyResult<_>>()?,
            sequence_graph: Some(sequence_graph.clone()),
        })
    }

    /// A superposition covering the positions of both, given another ``SuperPosition`` or a
    /// ``Position``.
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.union(&Self::from_operand(other)?)
    }

    /// The positions ``delta`` steps along this superposition's strand, backward for a negative
    /// ``delta``.
    ///
    /// Raises ``ValueError`` when a step starts from more than one position, which happens once an
    /// earlier step passed a fork, and ``IndexError`` when a step leaves the sequence graph.
    fn __add__(&self, delta: i64) -> PyResult<Self> {
        self.stepped(delta)
    }

    /// The positions ``delta`` steps back along this superposition's strand; the same as
    /// ``sp + -delta``.
    fn __sub__(&self, delta: i64) -> PyResult<Self> {
        self.stepped(-delta)
    }

    fn __len__(&self) -> usize {
        self.positions.len()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<PySuperPosition>>()
            .is_ok_and(|other| {
                self.positions
                    .iter()
                    .map(|position| position.position)
                    .eq(other.positions.iter().map(|position| position.position))
            })
    }

    fn __hash__(&self) -> isize {
        self.positions
            .iter()
            .fold(0, |hash, position| position_hash(hash, &position.position))
    }

    fn __repr__(&self) -> String {
        format!(
            "SuperPosition([{}])",
            self.positions
                .iter()
                .map(|position| position.__str__())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use r#gen::test_helpers::{create_bg, setup_gen_on_disk};
    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::DbContext,
        edge::Edge,
        node::Node,
        sequence::Sequence,
    };
    use pyo3::{Python, prepare_freethreaded_python};

    use super::{Position, PyPosition, PySuperPosition, current_graph};
    use crate::python_api::block_group::PySequenceGraph;

    /// `ABCD` forks into `EFGH` and `IJKL`, which rejoin at `MNOP`. The current path reads
    /// `ABCD`, `EFGH`, `MNOP`, leaving `IJKL` as the other branch of the fork.
    fn bubble() -> (DbContext, PySequenceGraph, HashMap<&'static str, HashId>) {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        Collection::create(conn, "test").expect("should create collection");
        let block_group = create_bg(conn, "test", "test", "chr1");
        let mut nodes = HashMap::new();
        for name in ["ABCD", "EFGH", "IJKL", "MNOP"] {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(name)
                .save(conn)
                .expect("should save sequence");
            let node_id = Node::create(conn, &sequence.hash, &HashId::convert_str(name))
                .expect("should create node");
            nodes.insert(name, node_id);
        }
        let edges = [
            (PATH_START_NODE_ID, 0, nodes["ABCD"], 0),
            (nodes["ABCD"], 4, nodes["EFGH"], 0),
            (nodes["ABCD"], 4, nodes["IJKL"], 1),
            (nodes["EFGH"], 4, nodes["MNOP"], 0),
            (nodes["IJKL"], 4, nodes["MNOP"], 1),
            (nodes["MNOP"], 4, PATH_END_NODE_ID, 0),
        ];
        let mut path_edge_ids = vec![];
        let mut rows = vec![];
        for (index, (source, source_coordinate, target, chromosome_index)) in
            edges.into_iter().enumerate()
        {
            let edge = Edge::create(
                conn,
                source,
                source_coordinate,
                Strand::Forward,
                target,
                0,
                Strand::Forward,
            )
            .expect("should create edge");
            // The path follows the first branch of the fork (chromosome index 0): ABCD, EFGH, MNOP.
            // Indices 2 and 4 are the IJKL branch, which the current path skips.
            if index != 2 && index != 4 {
                path_edge_ids.push(edge.id);
            }
            rows.push(BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge.id,
                chromosome_index,
                phased: 0,
            });
        }
        BlockGroupEdge::bulk_create(conn, &rows);
        gen_models::path::Path::create(conn, "chr1", &block_group.id, &path_edge_ids)
            .expect("should create current path");
        let sequence_graph = PySequenceGraph {
            id: block_group.id,
            collection_name: "test".to_string(),
            sample_name: "test".to_string(),
            name: "chr1".to_string(),
            context: Some(context.clone()),
        };
        (context, sequence_graph, nodes)
    }

    /// The last position of node `node`, attached to `sequence_graph`.
    fn last_position(
        context: &DbContext,
        sequence_graph: &PySequenceGraph,
        nodes: &HashMap<&'static str, HashId>,
        node: &str,
        coordinate: i64,
    ) -> PyPosition {
        let graph = current_graph(context, &sequence_graph.id).expect("should read current graph");
        let mut position = PyPosition::located(
            &graph,
            Position {
                node_id: nodes[node],
                coordinate,
                strand: Strand::Forward,
            },
        )
        .expect("should locate position");
        position.sequence_graph = Some(sequence_graph.clone());
        position
    }

    #[test]
    fn test_position_add_without_fork_stays_a_position() {
        prepare_freethreaded_python();
        let (context, sequence_graph, nodes) = bubble();
        let position = last_position(&context, &sequence_graph, &nodes, "ABCD", 0);

        Python::with_gil(|python| {
            let stepped = position
                .__add__(python, 1)
                .expect("should step within a node");
            let stepped = stepped
                .extract::<PyPosition>(python)
                .expect("a step that stays on a single point should return a Position");
            assert_eq!(stepped.position.coordinate, 1);
            assert_eq!(
                stepped
                    .sequence_graph
                    .as_ref()
                    .expect("should carry the sequence graph")
                    .id,
                sequence_graph.id
            );

            let back = stepped
                .__sub__(python, 1)
                .expect("should step back within a node")
                .extract::<PyPosition>(python)
                .expect("stepping back should also stay a Position");
            assert_eq!(back.position, position.position);
        });
    }

    #[test]
    fn test_position_add_at_fork_becomes_a_superposition() {
        prepare_freethreaded_python();
        let (context, sequence_graph, nodes) = bubble();
        // The last position of ABCD steps into the first position of both EFGH and IJKL.
        let position = last_position(&context, &sequence_graph, &nodes, "ABCD", 3);

        Python::with_gil(|python| {
            let stepped = position
                .__add__(python, 1)
                .expect("should step to the fork");
            let superposition = stepped
                .extract::<PySuperPosition>(python)
                .expect("a step that lands at a fork should return a SuperPosition");
            let mut node_ids: Vec<HashId> = superposition
                .positions
                .iter()
                .map(|position| position.position.node_id)
                .collect();
            node_ids.sort();
            let mut expected = vec![nodes["EFGH"], nodes["IJKL"]];
            expected.sort();
            assert_eq!(node_ids, expected);
        });
    }

    #[test]
    fn test_position_add_without_sequence_graph_raises_value_error() {
        prepare_freethreaded_python();
        let (context, sequence_graph, nodes) = bubble();
        let mut position = last_position(&context, &sequence_graph, &nodes, "ABCD", 0);
        position.sequence_graph = None;

        Python::with_gil(|python| {
            let error = position
                .__add__(python, 1)
                .expect_err("stepping an unattached position should fail");
            assert!(
                error.to_string().contains("attach one with pos.on(sg)"),
                "unexpected error message: {error}"
            );
        });
    }

    #[test]
    fn test_position_add_past_terminal_raises_index_error() {
        prepare_freethreaded_python();
        let (context, sequence_graph, nodes) = bubble();
        let position = last_position(&context, &sequence_graph, &nodes, "MNOP", 3);

        Python::with_gil(|python| {
            let error = position
                .__add__(python, 1)
                .expect_err("stepping past the end of the graph should fail");
            assert!(
                error.is_instance_of::<pyo3::exceptions::PyIndexError>(python),
                "expected an IndexError, got: {error}"
            );
        });
    }
}
