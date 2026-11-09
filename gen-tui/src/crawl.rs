use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::Hash,
};

pub use gen_graph::{EagerSource, GraphSource};
use petgraph::{
    Direction,
    graph::NodeIndex,
    graphmap::DiGraphMap,
    stable_graph::StableDiGraph,
    visit::{
        EdgeRef, GraphBase, IntoEdges, IntoNeighborsDirected, IntoNodeIdentifiers, NodeIndexable,
    },
};

use crate::{
    curved::EdgeStyle,
    cycle_removal::remove_cycles,
    edge_ports::{EdgePorts, NodeSide},
    layout::{PinSide, WindowStructureBuilder},
    pinning::{
        LegEnd, OrderingConstraint, PinPlan, PinnedEdge, ordering_constraint, plan_curved_bends,
        plan_hairpin, plan_hairpin_arc, plan_sideways_bends, plan_vertical_bends,
    },
    window_graph::{WindowEdge, WindowGraph, WindowNode},
};

/// Preferred target for each boundary node and direction, retained across window rebuilds.
pub type PreferredWormholeDoors<NodeId> = HashMap<(NodeId, bool), NodeId>;

/// The domain edges that landed fully inside a built window, with their direction metadata
/// along the route. See `build_window_graph`.
pub type WindowPinnedEdges = Vec<PinnedEdge>;

/// Where a node belongs in the order a caller wants the layout to read left to right, if the
/// caller has one. See `build_window_graph_with_order`.
pub type OrderHintFn<G> = dyn Fn(<G as GraphBase>::NodeId) -> Option<i64> + Send + Sync;

/// How to read the ports of an edge from the graph that holds it.
pub type EdgePortsFn<G> = fn(&G, &<G as GraphBase>::EdgeId) -> EdgePorts;

/// The domain direction in which a collapsed wormhole door leaves its in-window boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalEdgeSide {
    /// The off-window target is an incoming neighbour of the boundary node.
    Predecessor,
    /// The off-window target is an outgoing neighbour of the boundary node.
    Successor,
}

/// One collapsed wormhole door: the in-window boundary node, its chosen navigation target,
/// every off-window domain node it represents, and the direction in which it leaves the
/// loaded window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalEdge<NodeId> {
    /// The domain node inside the loaded window.
    pub boundary: NodeId,
    /// The chosen off-window node used for navigation.
    pub target: NodeId,
    /// Every off-window neighbour represented by this collapsed door.
    pub all_collapsed: Vec<NodeId>,
    /// The domain direction from `boundary` toward the represented nodes.
    pub side: ExternalEdgeSide,
}

impl<NodeId> ExternalEdge<NodeId> {
    /// Construct a door representing missing incoming neighbours.
    pub fn predecessor(boundary: NodeId, target: NodeId, all_collapsed: Vec<NodeId>) -> Self {
        Self {
            boundary,
            target,
            all_collapsed,
            side: ExternalEdgeSide::Predecessor,
        }
    }

    /// Construct a door representing missing outgoing neighbours.
    pub fn successor(boundary: NodeId, target: NodeId, all_collapsed: Vec<NodeId>) -> Self {
        Self {
            boundary,
            target,
            all_collapsed,
            side: ExternalEdgeSide::Successor,
        }
    }
}

/// The domain graph paired with its lazy-loading source for the duration of a crawl - bundled
/// so the crawl functions below take one "the graph, however it gets loaded" parameter instead
/// of two.
pub struct GraphCursor<'a, G, S> {
    graph: &'a mut G,
    source: &'a mut S,
}

impl<'a, G, S> GraphCursor<'a, G, S>
where
    G: GraphBase,
    S: GraphSource<G>,
{
    pub fn new(graph: &'a mut G, source: &'a mut S) -> Self {
        Self { graph, source }
    }

    /// Borrow the underlying domain graph for reads that don't need to trigger loading.
    pub fn graph(&self) -> &G {
        self.graph
    }

    /// Make sure every node in `nodes` carries all its edges on both sides, in one request to
    /// the source for whichever of them are still frontier.
    fn complete(&mut self, nodes: &[G::NodeId]) -> Result<(), String>
    where
        G::NodeId: Copy,
    {
        let frontier: Vec<G::NodeId> = nodes
            .iter()
            .copied()
            .filter(|node| {
                self.source.is_frontier(*node, Direction::Outgoing)
                    || self.source.is_frontier(*node, Direction::Incoming)
            })
            .collect();
        if !frontier.is_empty() {
            self.source
                .expand_frontier(self.graph, &frontier)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

/// A size-independent window selected by [`neighborhood`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralSubgraph<NodeId> {
    /// The anchor the window is centred on.
    pub anchor: NodeId,
    /// The windowed nodes, sorted by domain index - purely for deterministic, repeatable
    /// window construction; `build_window_graph` only ever reads `node_id` from these in
    /// order, not the sort key itself.
    pub nodes: Vec<NodeId>,
    /// Domain edges `(source, target)` with both endpoints inside the window.
    pub edges: Vec<(NodeId, NodeId)>,
    /// The ports of every edge between a node pair, for pairs whose edges do not all attach
    /// east to west. A pair missing here has the default ports. See
    /// [`StructuralSubgraph::assign_edge_ports`].
    pub edge_ports: BTreeMap<(NodeId, NodeId), Vec<EdgePorts>>,
    /// Boundary edges collapsed to one wormhole door per node and direction.
    pub external_edges: Vec<ExternalEdge<NodeId>>,
}

/// Select up to `node_budget` nodes around `anchor`, including internal and boundary edges.
/// `cursor`'s source is asked to expand the frontier whenever the crawl needs to go past it;
/// pass an [`EagerSource`]-backed cursor for a graph that is already fully loaded. Nodes for
/// which `is_claimed` holds belong to another batch: the crawl stops at them and they become
/// doors instead.
pub fn neighborhood<G, S>(
    anchor: G::NodeId,
    node_budget: usize,
    cursor: &mut GraphCursor<G, S>,
    is_claimed: &dyn Fn(G::NodeId) -> bool,
    preferred_doors: &PreferredWormholeDoors<G::NodeId>,
) -> Result<StructuralSubgraph<G::NodeId>, String>
where
    G: GraphBase,
    G::NodeId: Copy + Eq + Hash + Ord,
    for<'b> &'b G:
        IntoNodeIdentifiers<NodeId = G::NodeId> + IntoNeighborsDirected<NodeId = G::NodeId>,
    S: GraphSource<G>,
{
    let members = crawl_batch(anchor, node_budget, cursor, is_claimed)?;
    Ok(subgraph_for(
        anchor,
        members,
        cursor.graph(),
        preferred_doors,
    ))
}

/// Claim up to `node_budget` unclaimed nodes around `anchor` as one batch, returned sorted by
/// domain identity. Every member comes back with all its edges loaded on both sides, so the
/// batch's window can later be rebuilt from its members alone, however much the graph has grown
/// in the meantime.
pub fn crawl_batch<G, S>(
    anchor: G::NodeId,
    node_budget: usize,
    cursor: &mut GraphCursor<G, S>,
    is_claimed: &dyn Fn(G::NodeId) -> bool,
) -> Result<Vec<G::NodeId>, String>
where
    G: GraphBase,
    G::NodeId: Copy + Eq + Hash + Ord,
    for<'b> &'b G:
        IntoNodeIdentifiers<NodeId = G::NodeId> + IntoNeighborsDirected<NodeId = G::NodeId>,
    S: GraphSource<G>,
{
    if !cursor
        .graph()
        .node_identifiers()
        .any(|node_id| node_id == anchor)
    {
        return Err("Anchor node is not in the graph".to_string());
    }

    let selected = crawl_neighborhood(anchor, node_budget, cursor, is_claimed)?;
    // Sorted by the domain node's own identity, not by wherever it landed in the underlying
    // graph structure - so window order is reproducible regardless of insertion order, which
    // a lazily-loaded graph never guarantees.
    let mut members: Vec<G::NodeId> = selected.into_iter().collect();
    members.sort();
    // Every member's neighbours are read to find in-window edges and boundary doors, so none
    // of them may stay frontier, whichever ones the crawl itself went past (a zero-budget crawl
    // never looks past its own seed).
    cursor.complete(&members)?;
    Ok(members)
}

/// The window for a batch: its members, the edges among them, and one door per member and side
/// toward everything outside it. `members` must be sorted and fully loaded, as
/// [`crawl_batch`] returns them.
pub fn subgraph_for<G>(
    anchor: G::NodeId,
    members: Vec<G::NodeId>,
    graph: &G,
    preferred_doors: &PreferredWormholeDoors<G::NodeId>,
) -> StructuralSubgraph<G::NodeId>
where
    G: GraphBase,
    G::NodeId: Copy + Eq + Hash + Ord,
    for<'b> &'b G: IntoNeighborsDirected<NodeId = G::NodeId>,
{
    let selected: HashSet<G::NodeId> = members.iter().copied().collect();
    let nodes = members;

    // Domain edges with both endpoints inside the window. Walking the outgoing neighbours of
    // the selected nodes keeps this bounded by the window rather than the whole graph.
    // Backward edges ride along naturally: the domain graph still carries them, and both
    // endpoints being in the window is enough to include them.
    let mut edges: Vec<(G::NodeId, G::NodeId)> = Vec::new();
    for &node_id in &nodes {
        for successor in graph.neighbors_directed(node_id, Direction::Outgoing) {
            if selected.contains(&successor) {
                edges.push((node_id, successor));
            }
        }
    }
    edges.sort();
    edges.dedup();

    // Collapse each side of a boundary node to one door. Reuse a valid preferred target;
    // otherwise choose the lowest domain index for deterministic navigation.
    let mut external_edges: Vec<ExternalEdge<G::NodeId>> = Vec::new();
    for &node_id in &nodes {
        for (direction, exits_toward_successor) in
            [(Direction::Outgoing, true), (Direction::Incoming, false)]
        {
            let mut candidates: Vec<G::NodeId> = graph
                .neighbors_directed(node_id, direction)
                .filter(|neighbor| !selected.contains(neighbor))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            if candidates.is_empty() {
                continue;
            }
            candidates.sort();

            let preferred = preferred_doors
                .get(&(node_id, exits_toward_successor))
                .copied();
            let target = match preferred {
                Some(preferred) if candidates.contains(&preferred) => preferred,
                _ => candidates[0],
            };
            let side = if exits_toward_successor {
                ExternalEdgeSide::Successor
            } else {
                ExternalEdgeSide::Predecessor
            };
            external_edges.push(ExternalEdge {
                boundary: node_id,
                target,
                all_collapsed: candidates,
                side,
            });
        }
    }
    external_edges.sort_by_key(|edge| (edge.boundary, edge.target));

    StructuralSubgraph {
        anchor,
        nodes,
        edges,
        edge_ports: BTreeMap::new(),
        external_edges,
    }
}

impl<NodeId: Copy + Eq + Hash + Ord> StructuralSubgraph<NodeId> {
    /// Read the ports of each edge inside the window from `graph`. Parallel edges between one
    /// node pair each keep their own ports, which is why `graph` must expose one petgraph edge
    /// per domain edge (see `gen_graph::PackedGraphMap`).
    pub fn assign_edge_ports<G>(&mut self, graph: &G, ports_of: EdgePortsFn<G>)
    where
        G: GraphBase<NodeId = NodeId>,
        for<'b> &'b G: GraphBase<NodeId = NodeId, EdgeId = G::EdgeId> + IntoEdges,
    {
        let members: HashSet<NodeId> = self.nodes.iter().copied().collect();
        let mut by_pair: BTreeMap<(NodeId, NodeId), Vec<EdgePorts>> = BTreeMap::new();
        for &node_id in &self.nodes {
            for edge in graph.edges(node_id) {
                if members.contains(&edge.target()) {
                    by_pair
                        .entry((edge.source(), edge.target()))
                        .or_default()
                        .push(ports_of(graph, &edge.id()));
                }
            }
        }
        for ports in by_pair.values_mut() {
            ports.sort();
            ports.dedup();
        }
        by_pair.retain(|_, ports| ports.as_slice() != [EdgePorts::default()]);
        self.edge_ports = by_pair;
    }

    /// The distinct ports of the edges from `source` to `target`.
    pub fn ports_of(&self, source: NodeId, target: NodeId) -> Vec<EdgePorts> {
        self.edge_ports
            .get(&(source, target))
            .cloned()
            .unwrap_or_else(|| vec![EdgePorts::default()])
    }
}

/// One domain edge as the window sees it.
struct DomainEdge<NodeId> {
    source: NodeId,
    target: NodeId,
    ports: EdgePorts,
    /// The caller declared the edge backward, whatever the ordering says.
    forced_backward: bool,
}

/// Build a self-contained Sugiyama input from a structural window.
///
/// Nodes are first ordered by the edges that ask for an order (cycle removal breaks any cycle
/// among them). Each edge is then drawn directly when its sides agree with that order, and
/// otherwise routed through left and right pins (see [`plan_sideways_bends`]). Every pin joins
/// the Sugiyama input, along with a temporary edge between the endpoints of each pinned route
/// that keeps their ranks in the chosen order; the builder removes it once layers are known.
/// Boundary edges become ranked wormhole nodes. `explicit_backward_edges` names node pairs to
/// treat as backward whatever the ordering says.
pub fn build_window_graph<G>(
    neighborhood: &StructuralSubgraph<G::NodeId>,
    graph: &G,
    explicit_backward_edges: Option<&HashSet<(G::NodeId, G::NodeId)>>,
) -> Result<(WindowGraph, WindowPinnedEdges), String>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash + Ord + 'static,
{
    build_window_graph_with_order(
        neighborhood,
        graph,
        explicit_backward_edges,
        None,
        EdgeStyle::default(),
    )
}

/// [`build_window_graph`] with an `order_hint` for the nodes whose left to right order the
/// caller knows, such as the nodes of an aligned reference path, whose position scale only
/// reads correctly when they lie in path order. Hinted nodes are ordered by their hints
/// whatever cycles the edges form, so an edge running against the hints is drawn as a backward
/// edge instead of cycle removal choosing which edge of a cycle to flip. `edge_style` decides
/// whether left and right pins join the Sugiyama input: the curved styles leave them out.
pub fn build_window_graph_with_order<G>(
    neighborhood: &StructuralSubgraph<G::NodeId>,
    graph: &G,
    explicit_backward_edges: Option<&HashSet<(G::NodeId, G::NodeId)>>,
    order_hint: Option<&OrderHintFn<G>>,
    edge_style: EdgeStyle,
) -> Result<(WindowGraph, WindowPinnedEdges), String>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash + Ord + 'static,
{
    let mut window_graph: StableDiGraph<WindowNode, WindowEdge, u32> = StableDiGraph::new();
    let mut node_to_local: HashMap<G::NodeId, NodeIndex<u32>> = HashMap::new();

    for &node_id in &neighborhood.nodes {
        let domain_idx = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
        let local_idx = window_graph.add_node(WindowNode::Data(domain_idx));
        node_to_local.insert(node_id, local_idx);
    }
    let to_domain =
        |node_id: G::NodeId| NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));

    let forced_pairs: HashSet<(G::NodeId, G::NodeId)> = explicit_backward_edges
        .into_iter()
        .flatten()
        .copied()
        .filter(|(source, target)| {
            node_to_local.contains_key(source) && node_to_local.contains_key(target)
        })
        .collect();

    let mut domain_edges: Vec<DomainEdge<G::NodeId>> = Vec::new();
    for &(source, target) in &neighborhood.edges {
        if forced_pairs.contains(&(source, target)) {
            continue;
        }
        for ports in neighborhood.ports_of(source, target) {
            domain_edges.push(DomainEdge {
                source,
                target,
                ports,
                forced_backward: false,
            });
        }
    }
    // A caller's explicit backward edge is looped even when the domain graph lacks it.
    for &(source, target) in &forced_pairs {
        domain_edges.push(DomainEdge {
            source,
            target,
            ports: EdgePorts::default(),
            forced_backward: true,
        });
    }
    domain_edges.sort_by_key(|edge| (edge.source, edge.target, edge.ports));
    domain_edges.dedup_by_key(|edge| (edge.source, edge.target, edge.ports));

    // The first sort: an order of the nodes that agrees with as many edges as possible. Only
    // edges that attach east to west or west to east say anything about the order.
    let mut ordering_graph: DiGraphMap<G::NodeId, ()> = DiGraphMap::new();
    for &node_id in &neighborhood.nodes {
        ordering_graph.add_node(node_id);
    }
    let hints: HashMap<G::NodeId, i64> = order_hint
        .map(|hint| {
            neighborhood
                .nodes
                .iter()
                .filter_map(|&node_id| Some((node_id, hint(node_id)?)))
                .collect()
        })
        .unwrap_or_default();
    let mut hinted_nodes: Vec<(i64, G::NodeId)> = hints
        .iter()
        .map(|(&node_id, &hint)| (hint, node_id))
        .collect();
    hinted_nodes.sort();
    for pair in hinted_nodes.windows(2) {
        ordering_graph.add_edge(pair[0].1, pair[1].1, ());
    }
    let against_hints = |edge: &DomainEdge<G::NodeId>| {
        matches!(
            (hints.get(&edge.source), hints.get(&edge.target)),
            (Some(source_hint), Some(target_hint)) if target_hint < source_hint
        )
    };
    for edge in domain_edges
        .iter()
        .filter(|edge| !edge.forced_backward && edge.source != edge.target && !against_hints(edge))
    {
        match ordering_constraint(edge.ports) {
            Some(OrderingConstraint::SourceFirst) => {
                ordering_graph.add_edge(edge.source, edge.target, ());
            }
            Some(OrderingConstraint::TargetFirst) => {
                ordering_graph.add_edge(edge.target, edge.source, ());
            }
            None => {}
        }
    }
    let position_in_order: HashMap<G::NodeId, usize> = remove_cycles(&ordering_graph, None, None)
        .ordering
        .into_iter()
        .enumerate()
        .map(|(position, node_id)| (node_id, position))
        .collect();

    let mut plans: Vec<PinPlan> = domain_edges
        .iter()
        .map(|edge| {
            // An edge that leaves a side of a node and returns to it is a hairpin, which a hair
            // pin beside that side turns around.
            if edge.source == edge.target && edge.ports.source == edge.ports.target {
                return if edge_style.is_curved() {
                    plan_hairpin_arc()
                } else {
                    plan_hairpin(edge.ports.source)
                };
            }
            // A loop's target never follows its source in the order, but a loop that leaves the
            // west side and returns on the east side travels the same legs the other way round.
            let target_follows_source = if edge.source == edge.target {
                edge.ports == EdgePorts::new(NodeSide::West, NodeSide::East)
            } else {
                !edge.forced_backward
                    && position_in_order[&edge.target] > position_in_order[&edge.source]
            };
            if edge_style.is_curved() {
                plan_curved_bends(
                    edge.ports,
                    target_follows_source,
                    edge.source == edge.target,
                )
            } else {
                plan_sideways_bends(edge.ports, target_follows_source)
            }
        })
        .collect();
    let ends: Vec<(G::NodeId, G::NodeId)> = domain_edges
        .iter()
        .map(|edge| (edge.source, edge.target))
        .collect();
    plan_vertical_bends(&mut plans, &ends);

    let mut pin_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)> = HashSet::new();
    let mut pinned_edges: WindowPinnedEdges = Vec::new();
    let mut plain_legs: HashSet<(NodeIndex<u32>, NodeIndex<u32>)> = HashSet::new();
    let mut ordering_edges: Vec<(NodeIndex<u32>, NodeIndex<u32>)> = Vec::new();
    for (edge, plan) in domain_edges.iter().zip(&plans) {
        let bundle = (to_domain(edge.source), to_domain(edge.target));
        let left_pin = plan
            .left_pin
            .then(|| window_graph.add_node(WindowNode::Pin(PinSide::Left)));
        let right_pin = plan
            .right_pin
            .then(|| window_graph.add_node(WindowNode::Pin(PinSide::Right)));
        let bend_pin = plan
            .bend_pin
            .map(|side| window_graph.add_node(WindowNode::Pin(side)));
        let hair_pin = plan
            .hair_pin
            .map(|side| window_graph.add_node(WindowNode::HairPin(side)));
        let resolve = |end: LegEnd| match end {
            LegEnd::Source => node_to_local[&edge.source],
            LegEnd::Target => node_to_local[&edge.target],
            LegEnd::LeftPin => left_pin.expect("should have added the left pin the plan names"),
            LegEnd::RightPin => right_pin.expect("should have added the right pin the plan names"),
            LegEnd::BendPin => bend_pin.expect("should have added the bend pin the plan names"),
            LegEnd::HairPin => hair_pin.expect("should have added the hair pin the plan names"),
        };
        for leg in &plan.legs {
            let (from, to) = (resolve(leg.from), resolve(leg.to));
            // Two domain edges can lay out as one plain window edge; the first one wins.
            if !plan.has_pins() && !plain_legs.insert((from, to)) {
                continue;
            }
            window_graph.add_edge(from, to, Some(bundle));
        }
        if let (Some(left_pin), Some(right_pin)) = (left_pin, right_pin) {
            pin_span_edges.insert((left_pin, right_pin));
        }
        // The pins alone do not tie the endpoints' ranks to the cycle-removal order, so a
        // pinned route also asks for its endpoints to stay in that order.
        if plan.has_pins() && edge.source != edge.target {
            let source_first = position_in_order[&edge.source] < position_in_order[&edge.target];
            let (earlier, later) = if source_first {
                (edge.source, edge.target)
            } else {
                (edge.target, edge.source)
            };
            let ordering_edge = (node_to_local[&earlier], node_to_local[&later]);
            if !ordering_edges.contains(&ordering_edge) {
                ordering_edges.push(ordering_edge);
            }
        }
        pinned_edges.push(PinnedEdge {
            source: bundle.0,
            target: bundle.1,
            ports: edge.ports,
        });
    }

    // Add each collapsed door as a ranked node on the correct side of its boundary.
    for door in &neighborhood.external_edges {
        let boundary_domain = NodeIndex::new(<G as NodeIndexable>::to_index(graph, door.boundary));
        let target_domain = NodeIndex::new(<G as NodeIndexable>::to_index(graph, door.target));
        let Some(&boundary_local) = node_to_local.get(&door.boundary) else {
            continue;
        };
        let wormhole_local = window_graph.add_node(WindowNode::Wormhole(target_domain));
        for &collapsed in &door.all_collapsed {
            let collapsed_domain = NodeIndex::new(<G as NodeIndexable>::to_index(graph, collapsed));
            let bundle = (boundary_domain, collapsed_domain);
            match door.side {
                ExternalEdgeSide::Successor => {
                    window_graph.add_edge(boundary_local, wormhole_local, Some(bundle));
                }
                ExternalEdgeSide::Predecessor => {
                    window_graph.add_edge(wormhole_local, boundary_local, Some(bundle));
                }
            }
        }
    }

    let mut window = WindowGraph {
        graph: window_graph,
        structure: None,
        pin_span_edges,
    };

    if window.graph.node_count() > 1 {
        let mut builder = WindowStructureBuilder::new(&window).with_ordering_edges(ordering_edges);
        builder.build_structure()?;
        window.structure = builder.structure();
    }

    Ok((window, pinned_edges))
}

/// How far past its node budget a batch may grow to take the whole of its last shell. Cutting a
/// shell leaves doors between nodes that are one hop apart, so this is only a safety limit for a
/// port with a very large number of edges: shells in ordinary graphs grow by a few nodes per
/// hop and never come near it.
pub(crate) const SHELL_CUTOFF_FACTOR: usize = 10;

/// Grow breadth-first shells around `anchor` until the batch holds at least `node_budget` nodes
/// or the graph is exhausted. Each shell is loaded on both sides in one request to the source
/// and expands into its neighbours in both directions, so the window grows evenly by distance
/// from `anchor`: forward-reached nodes find reconverging branches and backward-reached nodes
/// find sibling branches. A shell is taken whole unless that would pass
/// [`SHELL_CUTOFF_FACTOR`] times the budget; then predecessors and successors are taken in turn
/// up to that cutoff, each in domain order so repeat crawls of the same window are stable.
/// Claimed nodes remain boundaries.
fn crawl_neighborhood<G, S>(
    anchor: G::NodeId,
    node_budget: usize,
    cursor: &mut GraphCursor<G, S>,
    is_claimed: &dyn Fn(G::NodeId) -> bool,
) -> Result<HashSet<G::NodeId>, String>
where
    G: GraphBase,
    for<'b> &'b G: IntoNeighborsDirected<NodeId = G::NodeId>,
    G::NodeId: Copy + Eq + Hash + Ord,
    S: GraphSource<G>,
{
    let cutoff = node_budget.saturating_mul(SHELL_CUTOFF_FACTOR);
    let mut visited: HashSet<G::NodeId> = HashSet::from([anchor]);
    let mut shell = vec![anchor];
    while visited.len() < node_budget && !shell.is_empty() {
        cursor.complete(&shell)?;
        let graph = cursor.graph();
        let [predecessors, successors] =
            [Direction::Incoming, Direction::Outgoing].map(|direction| {
                let mut neighbors: Vec<G::NodeId> = shell
                    .iter()
                    .flat_map(|&node_id| graph.neighbors_directed(node_id, direction))
                    .filter(|neighbor| !visited.contains(neighbor) && !is_claimed(*neighbor))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                neighbors.sort();
                neighbors
            });
        let whole_shell: HashSet<G::NodeId> =
            predecessors.iter().chain(&successors).copied().collect();
        let room = cutoff.saturating_sub(visited.len());
        let mut next_shell: Vec<G::NodeId> = vec![];
        if whole_shell.len() <= room {
            next_shell.extend(whole_shell);
            next_shell.sort();
        } else {
            // Take predecessors and successors in turn so a cut shell still grows both sides;
            // an odd leftover goes backward.
            let (mut predecessors, mut successors) =
                (predecessors.into_iter(), successors.into_iter());
            let mut taken: HashSet<G::NodeId> = HashSet::new();
            while next_shell.len() < room {
                let backward = predecessors.by_ref().find(|node_id| taken.insert(*node_id));
                let forward = if next_shell.len() + usize::from(backward.is_some()) < room {
                    successors.by_ref().find(|node_id| taken.insert(*node_id))
                } else {
                    None
                };
                if backward.is_none() && forward.is_none() {
                    break;
                }
                next_shell.extend(backward.into_iter().chain(forward));
            }
        }
        visited.extend(next_shell.iter().copied());
        shell = next_shell;
    }
    Ok(visited)
}

#[cfg(test)]
mod tests {
    use petgraph::graphmap::DiGraphMap;

    use super::*;
    use crate::{assembly::assemble_window, layout::NodeRole};

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
    struct TestNode(i64);

    fn make_test_graph(edges: Vec<(i32, i32)>) -> DiGraphMap<TestNode, ()> {
        DiGraphMap::from_edges(
            edges
                .iter()
                .map(|(s, t)| (TestNode(*s as i64), TestNode(*t as i64))),
        )
    }

    fn rank_mismatch_edges() -> Vec<(i32, i32)> {
        vec![
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 3),
            (2, 1),
            (2, 6),
            (2, 7),
            (3, 2),
            (3, 5),
            (4, 6),
            (5, 2),
            (5, 7),
            (6, 7),
            (7, 6),
        ]
    }

    #[test]
    fn build_window_graph_with_backward_edge_fully_inside_adds_pins() {
        // Linear chain 0 -> 1 -> 2 -> 3 with a backward edge 3 -> 0 closing the loop,
        // exercised with an explicit backward-edge list (mirrors the GFA-import path).
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let mut graph = make_test_graph(edges);
        let backward_edges: HashSet<(TestNode, TestNode)> =
            HashSet::from([(TestNode(3), TestNode(0))]);

        // A budget covering every node: both endpoints of the backward edge land inside.
        let subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        let (window, _) = build_window_graph(&subgraph, &graph, Some(&backward_edges))
            .expect("should build the pinned acyclic window");

        // 4 data nodes + 2 pins.
        assert_eq!(window.graph.node_count(), 6);
        let pin_count = window
            .graph
            .node_indices()
            .filter(|&idx| matches!(window.graph.node_weight(idx), Some(WindowNode::Pin(_))))
            .count();
        assert_eq!(
            pin_count, 2,
            "should have injected exactly one left/right pin pair"
        );
        assert!(
            window.structure.is_some(),
            "a multi-node window should have a built Sugiyama structure"
        );

        let node_3_domain = NodeIndex::new(<DiGraphMap<TestNode, ()> as NodeIndexable>::to_index(
            &graph,
            TestNode(3),
        ));
        let node_0_domain = NodeIndex::new(<DiGraphMap<TestNode, ()> as NodeIndexable>::to_index(
            &graph,
            TestNode(0),
        ));
        let find_local = |domain: NodeIndex<u32>| {
            window
                .graph
                .node_indices()
                .find(|&idx| {
                    matches!(
                        window.graph.node_weight(idx),
                        Some(WindowNode::Data(found)) if *found == domain
                    )
                })
                .expect("should include the domain node in the window")
        };
        let node_3_local = find_local(node_3_domain);
        let node_0_local = find_local(node_0_domain);
        assert!(
            !window.graph.contains_edge(node_3_local, node_0_local),
            "backward edge should be rewired onto pins, not left as a direct edge"
        );
    }

    #[test]
    fn test_cycle_cut_keeps_endpoint_order_without_stale_constraint_edges() {
        // Every two-pin detour keeps the order chosen when the cycle was cut. The temporary
        // ordering edges must disappear before the structure is cached for assembly.
        let edges = rank_mismatch_edges();
        let mut graph = make_test_graph(edges.clone());
        let mut ordering_graph = DiGraphMap::new();
        for node in 0..8 {
            graph.add_node(TestNode(node));
            ordering_graph.add_node(TestNode(node));
        }
        let mut sorted_edges = edges.clone();
        sorted_edges.sort();
        for &(source, target) in &sorted_edges {
            ordering_graph.add_edge(TestNode(source.into()), TestNode(target.into()), ());
        }
        let positions: HashMap<_, _> = remove_cycles(&ordering_graph, None, None)
            .ordering
            .into_iter()
            .enumerate()
            .map(|(index, node)| (node, index))
            .collect();
        let preliminary = edges
            .iter()
            .filter(|&&(source, target)| {
                positions[&TestNode(target.into())] < positions[&TestNode(source.into())]
            })
            .count();
        let subgraph = StructuralSubgraph {
            anchor: TestNode(0),
            nodes: (0..8).map(TestNode).collect(),
            edges: edges
                .iter()
                .map(|&(source, target)| (TestNode(source.into()), TestNode(target.into())))
                .collect(),
            edge_ports: BTreeMap::new(),
            external_edges: Vec::new(),
        };
        let (window, _) =
            build_window_graph(&subgraph, &graph, None).expect("should build the pinned window");
        let final_count = window
            .graph
            .node_weights()
            .filter(|node| matches!(node, WindowNode::Pin(PinSide::Left)))
            .count();
        assert_eq!(final_count, preliminary);
        let structure = window
            .structure
            .as_ref()
            .expect("should cache the structure");
        assert!(
            structure
                .vertex_graph
                .edge_weights()
                .all(|edge| edge.input_node_idx_pair.is_some()),
            "temporary ordering edges must not enter the cached structure"
        );
        for (source, target) in edges {
            let local = |node: i32| {
                let domain_index = <DiGraphMap<TestNode, ()> as NodeIndexable>::to_index(
                    &graph,
                    TestNode(node.into()),
                );
                window
                    .graph
                    .node_indices()
                    .find(|&index| {
                        matches!(window.graph.node_weight(index), Some(WindowNode::Data(domain)) if domain.index() == domain_index)
                    })
                    .expect("should retain the data node")
            };
            let source_rank = structure
                .locate(local(source))
                .expect("should rank source")
                .0;
            let target_rank = structure
                .locate(local(target))
                .expect("should rank target")
                .0;
            assert_eq!(
                source_rank < target_rank,
                positions[&TestNode(source.into())] < positions[&TestNode(target.into())],
                "Sugiyama should preserve the chosen order for {source} -> {target}"
            );
        }
    }

    #[test]
    fn test_west_east_detour_keeps_reverse_constraint() {
        // Reversing every domain edge and its ports preserves the same preliminary
        // constraints. A detour keeps its pins.
        let edges: Vec<_> = rank_mismatch_edges()
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
        let mut graph = make_test_graph(edges.clone());
        let mut ordering_graph = DiGraphMap::new();
        for node in 0..8 {
            graph.add_node(TestNode(node));
            ordering_graph.add_node(TestNode(node));
        }
        let pairs: Vec<_> = edges
            .iter()
            .map(|&(source, target)| (TestNode(source.into()), TestNode(target.into())))
            .collect();
        let mut sorted_pairs = pairs.clone();
        sorted_pairs.sort();
        for &(source, target) in &sorted_pairs {
            ordering_graph.add_edge(target, source, ());
        }
        let positions: HashMap<_, _> = remove_cycles(&ordering_graph, None, None)
            .ordering
            .into_iter()
            .enumerate()
            .map(|(index, node)| (node, index))
            .collect();
        let preliminary = pairs
            .iter()
            .filter(|&&(source, target)| positions[&source] < positions[&target])
            .count();
        let subgraph = StructuralSubgraph {
            anchor: TestNode(0),
            nodes: (0..8).map(TestNode).collect(),
            edges: pairs.clone(),
            edge_ports: pairs
                .into_iter()
                .map(|pair| (pair, vec![EdgePorts::new(NodeSide::West, NodeSide::East)]))
                .collect(),
            external_edges: Vec::new(),
        };
        let (window, _) =
            build_window_graph(&subgraph, &graph, None).expect("should build the window");
        let final_count = window
            .graph
            .node_weights()
            .filter(|node| matches!(node, WindowNode::Pin(PinSide::Left)))
            .count();
        assert_eq!(final_count, preliminary);
    }

    #[test]
    fn build_window_graph_with_backward_edge_partially_inside_skips_pins() {
        // Same chain and backward edge, but a node budget of 1 crawls only the anchor
        // itself - the backward edge's other endpoint (node 3) is outside the window, so
        // it must not be pinned: a backward edge only gets a pin pair when both endpoints
        // are in-window. The anchor's one real off-window neighbour (node 1, via the
        // ordinary edge 0->1 - this test's graph has no actual 3->0 edge, only a
        // caller-declared `explicit_backward_edges` label) becomes a real wormhole node.
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let mut graph = make_test_graph(edges);
        let backward_edges: HashSet<(TestNode, TestNode)> =
            HashSet::from([(TestNode(3), TestNode(0))]);

        let subgraph = neighborhood(
            TestNode(0),
            1,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        assert_eq!(
            subgraph.nodes.len(),
            1,
            "a node budget of 1 should crawl only the anchor"
        );

        let (window, window_backward_edges) =
            build_window_graph(&subgraph, &graph, Some(&backward_edges))
                .expect("should build the anchor and wormhole window");

        assert!(
            window_backward_edges.is_empty(),
            "a backward edge with only one endpoint in the window should not be pinned"
        );
        assert_eq!(
            window.graph.node_count(),
            2,
            "no pin nodes should be added for a partially-windowed backward edge, but the \
             anchor's one real off-window neighbour still gets a wormhole node"
        );
        assert!(
            window.graph.node_weights().any(
                |node| matches!(node, WindowNode::Wormhole(target) if *target == NodeIndex::new(
                    <DiGraphMap<TestNode, ()> as NodeIndexable>::to_index(&graph, TestNode(1))
                ))
            ),
            "the anchor's off-window successor should be represented as a real wormhole node"
        );
        assert!(
            window.structure.is_some(),
            "a two-node window has a real Sugiyama structure"
        );
        assert!(
            subgraph
                .external_edges
                .iter()
                .any(|edge| (edge.boundary, edge.target) == (TestNode(0), TestNode(1))),
            "the crawl's own boundary edge should still surface in the structural query output"
        );
    }

    #[test]
    fn build_window_graph_detects_local_cycle_without_explicit_backward_edges() {
        // Same chain and backward edge as above, but with no explicit backward-edge list -
        // this is the auto-detect path, which must find and pin the loop purely from the
        // window's own crawled nodes/edges (no whole-graph pass involved).
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let mut graph = make_test_graph(edges);

        let subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        let (window, _) = build_window_graph(&subgraph, &graph, None)
            .expect("should build after detecting the local cycle");

        let pin_count = window
            .graph
            .node_indices()
            .filter(|&idx| matches!(window.graph.node_weight(idx), Some(WindowNode::Pin(_))))
            .count();
        assert_eq!(
            pin_count, 2,
            "should have injected exactly one left/right pin pair"
        );
        assert!(
            window.structure.is_some(),
            "a multi-node window should have a built Sugiyama structure"
        );
    }

    const WEST_WEST: EdgePorts = EdgePorts::new(NodeSide::West, NodeSide::West);
    const EAST_EAST: EdgePorts = EdgePorts::new(NodeSide::East, NodeSide::East);
    const WEST_EAST: EdgePorts = EdgePorts::new(NodeSide::West, NodeSide::East);

    /// Window of the chain 0 -> 1 -> 2 whose edge 0 -> 2 attaches to the given ports.
    fn window_with_ports(ports: EdgePorts, edge: (i32, i32)) -> (WindowGraph, WindowPinnedEdges) {
        let mut graph = make_test_graph(vec![(0, 1), (1, 2), (edge.0, edge.1)]);
        let mut subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        subgraph.edge_ports.insert(
            (TestNode(edge.0 as i64), TestNode(edge.1 as i64)),
            vec![ports],
        );
        build_window_graph(&subgraph, &graph, None).expect("should build the window")
    }

    fn pin_sides(window: &WindowGraph) -> Vec<PinSide> {
        let mut sides: Vec<PinSide> = window
            .graph
            .node_weights()
            .filter_map(|node| match node {
                WindowNode::Pin(side) => Some(*side),
                _ => None,
            })
            .collect();
        sides.sort_by_key(|side| matches!(side, PinSide::Right));
        sides
    }

    fn hair_pin_sides(window: &WindowGraph) -> Vec<NodeSide> {
        window
            .graph
            .node_weights()
            .filter_map(|node| match node {
                WindowNode::HairPin(side) => Some(*side),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_west_to_west_edge_adds_one_left_pin() {
        let (window, _) = window_with_ports(WEST_WEST, (0, 2));
        assert_eq!(pin_sides(&window), vec![PinSide::Left]);
    }

    #[test]
    fn test_east_to_east_edge_adds_one_right_pin() {
        let (window, _) = window_with_ports(EAST_EAST, (0, 2));
        assert_eq!(pin_sides(&window), vec![PinSide::Right]);
    }

    #[test]
    fn test_east_to_east_edge_from_a_node_to_itself_adds_one_hair_pin() {
        let (window, _) = window_with_ports(EAST_EAST, (1, 1));
        assert!(pin_sides(&window).is_empty());
        assert_eq!(hair_pin_sides(&window), vec![NodeSide::East]);
        let assembled = assemble_window(&window).expect("should assemble the window");
        assert!(
            assembled
                .graph
                .node_weights()
                .any(|node| matches!(node.role, NodeRole::HairPin(NodeSide::East)))
        );
    }

    #[test]
    fn test_hair_pin_has_one_edge_labelled_with_the_edge_it_stands_for() {
        for ports in [EAST_EAST, WEST_WEST] {
            let (window, _) = window_with_ports(ports, (1, 1));
            let hair_pin = window
                .graph
                .node_indices()
                .find(|&index| matches!(window.graph[index], WindowNode::HairPin(_)))
                .expect("should add a hair pin");
            let touching: Vec<_> = window
                .graph
                .edge_indices()
                .filter(|&edge| {
                    let (source, target) = window.graph.edge_endpoints(edge).expect("should exist");
                    source == hair_pin || target == hair_pin
                })
                .collect();
            assert_eq!(touching.len(), 1, "one edge, so ranking is not disturbed");
            assert!(
                matches!(window.graph[touching[0]], Some((source, target)) if source == target),
                "the edge carries the domain self-edge it stands for"
            );
        }
    }

    #[test]
    fn test_west_to_west_edge_from_a_node_to_itself_adds_one_hair_pin() {
        let (window, _) = window_with_ports(WEST_WEST, (1, 1));
        assert_eq!(hair_pin_sides(&window), vec![NodeSide::West]);
    }

    #[test]
    fn test_west_to_east_edge_forward_in_order_adds_both_pins() {
        let (window, _) = window_with_ports(WEST_EAST, (0, 2));
        assert_eq!(pin_sides(&window), vec![PinSide::Left, PinSide::Right]);
        assert_eq!(window.pin_span_edges.len(), 1);
    }

    #[test]
    fn test_west_to_east_edge_against_the_order_bends_down() {
        // 2 -> 0 attaching west to east reads as 0 -> 2 on the page.
        let (window, _) = window_with_ports(WEST_EAST, (2, 0));
        assert_eq!(pin_sides(&window), vec![PinSide::Down]);
        assert_eq!(window.graph.edge_count(), 4);
    }

    #[test]
    fn test_default_edge_against_the_order_adds_both_pins() {
        let (window, _) = window_with_ports(EdgePorts::default(), (2, 0));
        assert_eq!(pin_sides(&window), vec![PinSide::Left, PinSide::Right]);
    }

    #[test]
    fn test_parallel_edges_with_different_ports_keep_separate_routes() {
        let mut graph = make_test_graph(vec![(0, 1)]);
        let mut subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        subgraph.edge_ports.insert(
            (TestNode(0), TestNode(1)),
            vec![EdgePorts::default(), WEST_WEST, EAST_EAST],
        );
        let (window, pinned) =
            build_window_graph(&subgraph, &graph, None).expect("should build the window");
        assert_eq!(pin_sides(&window), vec![PinSide::Left, PinSide::Right]);
        assert_eq!(pinned.len(), 3);
    }

    #[test]
    fn test_opposite_straight_edges_get_an_up_and_a_down_pin() {
        // 0 -> 1 east to west and 1 -> 0 west to east both read left to right and right to
        // left between the same two nodes.
        let mut graph = make_test_graph(vec![(0, 1)]);
        let mut subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        subgraph
            .edge_ports
            .insert((TestNode(0), TestNode(1)), vec![EdgePorts::default()]);
        subgraph
            .edge_ports
            .insert((TestNode(1), TestNode(0)), vec![WEST_EAST]);
        subgraph.edges.push((TestNode(1), TestNode(0)));
        let (window, pinned) =
            build_window_graph(&subgraph, &graph, None).expect("should build the window");
        let mut sides = pin_sides(&window);
        sides.sort_by_key(|side| matches!(side, PinSide::Down));
        assert_eq!(sides, vec![PinSide::Up, PinSide::Down]);
        assert_eq!(pinned.len(), 2);
        assert_eq!(window.graph.edge_count(), 4);
    }

    #[test]
    fn test_assign_edge_ports_reads_each_parallel_edge() {
        let mut graph = make_test_graph(vec![(0, 1), (1, 2)]);
        let mut subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        subgraph.assign_edge_ports(&graph, |_, edge| {
            if edge.0 == TestNode(1) {
                WEST_WEST
            } else {
                EdgePorts::default()
            }
        });
        assert_eq!(
            subgraph.ports_of(TestNode(0), TestNode(1)),
            vec![EdgePorts::default()]
        );
        assert_eq!(subgraph.ports_of(TestNode(1), TestNode(2)), vec![WEST_WEST]);
    }

    #[test]
    fn test_neighborhood_shells_crawl_a_chain() {
        // 0 -> 1 -> 2 -> 3 -> 4, anchor 2, budget 3: after the anchor's own slot, one backward
        // hop (finds 1) and one forward hop (finds 3).
        let mut graph = make_test_graph(vec![(0, 1), (1, 2), (2, 3), (3, 4)]);

        let subgraph = neighborhood(
            TestNode(2),
            3,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(subgraph.anchor, TestNode(2));
        // Reached nodes remain real and receive doors for their missing sides.
        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(
            windowed,
            HashSet::from([TestNode(1), TestNode(2), TestNode(3)])
        );
        assert_eq!(
            subgraph.edges.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([(TestNode(1), TestNode(2)), (TestNode(2), TestNode(3))]),
        );
        assert_eq!(
            subgraph
                .external_edges
                .iter()
                .map(|edge| (edge.boundary, edge.target))
                .collect::<HashSet<_>>(),
            HashSet::from([(TestNode(1), TestNode(0)), (TestNode(3), TestNode(4))]),
        );
    }

    #[test]
    fn test_neighborhood_shells_grow_both_sides_by_distance() {
        // Chain 0 -> ... -> 6 with anchor 3, plus 10 -> 5 rejoining two hops downstream. A
        // forward sweep followed by a backward sweep from its whole reach would spend the
        // backward budget on 10 (one incoming hop from 5). Shells growing both ways at once
        // reach two nodes on each side of the anchor before 10 is ever a candidate.
        let mut graph = make_test_graph(vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (10, 5),
        ]);

        let subgraph = neighborhood(
            TestNode(3),
            5,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(
            subgraph.nodes,
            vec![
                TestNode(1),
                TestNode(2),
                TestNode(3),
                TestNode(4),
                TestNode(5)
            ],
            "the window should extend two hops on each side of the anchor"
        );
        assert_eq!(
            subgraph
                .external_edges
                .iter()
                .map(|edge| (edge.boundary, edge.target))
                .collect::<HashSet<_>>(),
            HashSet::from([
                (TestNode(1), TestNode(0)),
                (TestNode(5), TestNode(10)),
                (TestNode(5), TestNode(6)),
            ]),
        );
    }

    #[test]
    fn test_neighborhood_takes_a_shell_whole_past_the_budget() {
        // Chain 0 -> 1 -> 2 -> 3 -> 4 with anchor 2 and budget 2: the first shell {1, 3} passes
        // the budget, but stays under the cutoff, so it is taken whole rather than trimmed to one
        // side, and the crawl stops there.
        let mut graph = make_test_graph(vec![(0, 1), (1, 2), (2, 3), (3, 4)]);

        let subgraph = neighborhood(
            TestNode(2),
            2,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(subgraph.nodes, vec![TestNode(1), TestNode(2), TestNode(3)]);
    }

    #[test]
    fn test_neighborhood_cuts_only_a_shell_past_the_cutoff_keeping_both_sides() {
        // 0 -> 1 and 1 -> {2, 3, ...}: anchor 1's first shell holds more nodes than fit under the
        // cutoff. The cut takes the predecessor first and fills the rest with the lowest
        // successors, so both sides still grow.
        const BUDGET: usize = 3;
        let cutoff = BUDGET * SHELL_CUTOFF_FACTOR;
        let last_successor = cutoff as i32 + 5;
        let mut graph = make_test_graph(
            std::iter::once((0, 1))
                .chain((2..=last_successor).map(|successor| (1, successor)))
                .collect(),
        );

        let subgraph = neighborhood(
            TestNode(1),
            BUDGET,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(
            subgraph.nodes,
            (0..cutoff as i64).map(TestNode).collect::<Vec<_>>(),
            "the cut shell should fill exactly up to the cutoff, predecessor included"
        );
        assert_eq!(
            subgraph
                .external_edges
                .iter()
                .map(|edge| (edge.boundary, edge.target))
                .collect::<Vec<_>>(),
            vec![(TestNode(1), TestNode(cutoff as i64))],
            "the successors left out collapse into one door on the anchor"
        );
    }

    #[test]
    fn neighborhood_captures_whole_graph_when_budget_is_generous() {
        // Fork/join: 1 and 2 share a layer. Budget covers the entire (tiny) graph, so every
        // node's full neighbourhood is captured and no boundary doors are needed at all.
        let mut graph = make_test_graph(vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

        let subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(
            windowed,
            HashSet::from([TestNode(0), TestNode(1), TestNode(2), TestNode(3)]),
            "the whole 4-node graph should fit the budget"
        );
        assert!(subgraph.external_edges.is_empty());
        assert!(subgraph.nodes.contains(&TestNode(1)));
        assert!(subgraph.nodes.contains(&TestNode(2)));
    }

    #[test]
    fn neighborhood_unknown_anchor_errors() {
        let mut graph = make_test_graph(vec![(0, 1)]);
        assert!(
            neighborhood(
                TestNode(99),
                1,
                &mut GraphCursor::new(&mut graph, &mut EagerSource),
                &|_| false,
                &HashMap::new()
            )
            .is_err()
        );
    }

    #[test]
    fn neighborhood_shared_successor_stays_real_with_one_boundary_door() {
        // A1 and A2 converge on B, followed by C and then D. From A1 with a budget of three,
        // the second shell {A2, C} is taken whole and D falls outside.
        let mut graph = make_test_graph(vec![(0, 1), (2, 1), (1, 3), (3, 4)]);

        let subgraph = neighborhood(
            TestNode(0),
            3,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(
            windowed,
            HashSet::from([TestNode(0), TestNode(1), TestNode(2), TestNode(3)]),
            "A1 (0), B (1), A2 (2), and C (3) should all stay real"
        );
        assert_eq!(
            subgraph.edges.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([
                (TestNode(0), TestNode(1)),
                (TestNode(2), TestNode(1)),
                (TestNode(1), TestNode(3)),
            ]),
            "both A1's and A2's real edges into B should survive as ordinary window edges"
        );
        assert_eq!(
            subgraph.external_edges,
            vec![ExternalEdge::successor(
                TestNode(3),
                TestNode(4),
                vec![TestNode(4)]
            )],
            "only C needs a door, for the one side (successor D) it's actually missing"
        );
    }

    #[test]
    fn neighborhood_stops_at_claimed_nodes_and_shows_them_as_doors() {
        // 0 -> 1 -> 2 -> 3 with 1 already claimed by another batch: a crawl from 2 must not take
        // 1 back, however generous its budget, and 1 becomes the door home.
        let mut graph = make_test_graph(vec![(0, 1), (1, 2), (2, 3)]);

        let subgraph = neighborhood(
            TestNode(2),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|node| node == TestNode(1),
            &HashMap::new(),
        )
        .unwrap();

        assert_eq!(subgraph.nodes, vec![TestNode(2), TestNode(3)]);
        assert_eq!(
            subgraph.external_edges,
            vec![ExternalEdge::predecessor(
                TestNode(2),
                TestNode(1),
                vec![TestNode(1)]
            )],
        );
    }

    #[test]
    fn neighborhood_hub_collapses_many_missing_neighbours_into_one_door() {
        // Collapse five off-window successors into one deterministic door.
        let mut graph = make_test_graph(vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);

        let subgraph = neighborhood(
            TestNode(0),
            1,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(windowed, HashSet::from([TestNode(0)]));
        assert_eq!(
            subgraph.external_edges,
            vec![ExternalEdge::successor(
                TestNode(0),
                TestNode(1),
                vec![
                    TestNode(1),
                    TestNode(2),
                    TestNode(3),
                    TestNode(4),
                    TestNode(5)
                ],
            )],
            "5 missing outgoing neighbours should collapse to exactly one door, carrying every \
             one of them so the rendered stub's edge can represent them all"
        );
    }

    #[test]
    fn neighborhood_hub_door_prefers_a_still_valid_target() {
        // Same hub as above, but with a preference already recorded for node 4 - simulating a
        // door that was actually traveled through in an earlier build (see
        // `LayoutEngine::remember_wormhole_choice`). The collapsed door should keep pointing at
        // 4 instead of falling back to the lowest index.
        let mut graph = make_test_graph(vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
        let preferred_doors = HashMap::from([((TestNode(0), true), TestNode(4))]);

        let subgraph = neighborhood(
            TestNode(0),
            1,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &preferred_doors,
        )
        .unwrap();

        assert_eq!(
            subgraph.external_edges,
            vec![ExternalEdge::successor(
                TestNode(0),
                TestNode(4),
                vec![
                    TestNode(1),
                    TestNode(2),
                    TestNode(3),
                    TestNode(4),
                    TestNode(5)
                ],
            )],
            "a still-valid preferred target should be kept instead of the lowest index"
        );
    }

    #[test]
    fn neighborhood_never_disconnects_a_downstream_branch() {
        // R -> M -> {1, 2, ...}; anchor M's first shell is wider than the cutoff, so it is cut to
        // R and the lowest successors, leaving the rest of the downstream branches outside.
        const BUDGET: usize = 2;
        let cutoff = BUDGET * SHELL_CUTOFF_FACTOR;
        let root = cutoff as i32 + 10;
        let mut graph = make_test_graph(
            std::iter::once((root, 0))
                .chain((1..root).map(|successor| (0, successor)))
                .collect(),
        );
        let subgraph = neighborhood(
            TestNode(0),
            BUDGET,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(
            windowed,
            (0..cutoff as i64 - 1)
                .map(TestNode)
                .chain([TestNode(root as i64)])
                .collect::<HashSet<_>>(),
            "M, R, and the lowest successors should all stay in the window"
        );
        // Every windowed node must be reachable from every other, undirected - i.e. connected.
        let mut adjacency: HashMap<TestNode, Vec<TestNode>> = HashMap::new();
        for &(source, target) in &subgraph.edges {
            adjacency.entry(source).or_default().push(target);
            adjacency.entry(target).or_default().push(source);
        }
        let start = *windowed.iter().next().unwrap();
        let mut reached = HashSet::from([start]);
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            for &neighbor in adjacency.get(&node).unwrap_or(&Vec::new()) {
                if reached.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }
        assert_eq!(
            reached, windowed,
            "the windowed set must be fully connected"
        );
    }

    #[test]
    fn neighborhood_visits_a_rejoining_branch_exactly_once() {
        // Diamond: 0 -> {1, 2} -> 3. Treated as undirected, 3 is reachable from both 1 and 2 in
        // the same shell - it must be selected once, not duplicated or re-queued (which would
        // either double-count it or, on a graph with an actual longer cycle, loop forever).
        let mut graph = make_test_graph(vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

        let subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            &|_| false,
            &HashMap::new(),
        )
        .unwrap();

        let windowed = &subgraph.nodes;
        assert_eq!(
            windowed.len(),
            4,
            "each of the 4 nodes must appear exactly once"
        );
        assert_eq!(
            windowed.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([TestNode(0), TestNode(1), TestNode(2), TestNode(3)])
        );
    }

    /// Reveals a hidden graph one side of one node at a time, like a database-backed source
    /// would.
    struct RevealingSource {
        hidden: DiGraphMap<TestNode, ()>,
        loaded: HashSet<(TestNode, Direction)>,
    }

    impl GraphSource<DiGraphMap<TestNode, ()>> for RevealingSource {
        type Error = core::convert::Infallible;

        fn is_frontier(&self, node: TestNode, direction: Direction) -> bool {
            !self.loaded.contains(&(node, direction))
        }

        fn expand_frontier(
            &mut self,
            graph: &mut DiGraphMap<TestNode, ()>,
            frontier: &[TestNode],
        ) -> Result<(), Self::Error> {
            for &node in frontier {
                for direction in [Direction::Outgoing, Direction::Incoming] {
                    if !self.loaded.insert((node, direction)) {
                        continue;
                    }
                    for neighbor in self.hidden.neighbors_directed(node, direction) {
                        match direction {
                            Direction::Outgoing => graph.add_edge(node, neighbor, ()),
                            Direction::Incoming => graph.add_edge(neighbor, node, ()),
                        };
                    }
                }
            }
            Ok(())
        }
    }

    #[test]
    fn test_neighborhood_over_a_revealing_source_matches_the_eager_graph() {
        let shapes = vec![
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            vec![
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (4, 6),
                (6, 7),
                (5, 7),
            ],
            vec![(0, 2), (1, 2), (2, 3), (3, 4), (5, 4), (4, 6), (6, 0)],
        ];
        for edges in shapes {
            let eager_graph = make_test_graph(edges);
            let anchors: Vec<TestNode> = eager_graph.nodes().collect();
            for anchor in anchors {
                for budget in 0..=eager_graph.node_count() + 1 {
                    let expected = neighborhood(
                        anchor,
                        budget,
                        &mut GraphCursor::new(&mut eager_graph.clone(), &mut EagerSource),
                        &|_| false,
                        &HashMap::new(),
                    )
                    .unwrap();

                    let mut lazy_graph = DiGraphMap::new();
                    lazy_graph.add_node(anchor);
                    let mut source = RevealingSource {
                        hidden: eager_graph.clone(),
                        loaded: HashSet::new(),
                    };
                    let actual = neighborhood(
                        anchor,
                        budget,
                        &mut GraphCursor::new(&mut lazy_graph, &mut source),
                        &|_| false,
                        &HashMap::new(),
                    )
                    .unwrap();

                    assert_eq!(
                        actual, expected,
                        "anchor {anchor:?}, budget {budget}: a lazily revealed graph should \
                         produce the same window as the fully loaded one"
                    );
                }
            }
        }
    }

    /// Layers of nodes 0 to 4 in a window whose recombination edges run against the chain 0 to 4.
    fn chain_layers_with_hint(hint: Option<&OrderHintFn<DiGraphMap<TestNode, ()>>>) -> Vec<usize> {
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 1),
            (3, 2),
            (4, 1),
            (4, 2),
            (4, 3),
            (2, 1),
            (1, 0),
            (0, 4),
        ];
        let graph = make_test_graph(edges.clone());
        let subgraph = StructuralSubgraph {
            anchor: TestNode(0),
            nodes: (0..5).map(TestNode).collect(),
            edges: edges
                .iter()
                .map(|&(source, target)| (TestNode(source.into()), TestNode(target.into())))
                .collect(),
            edge_ports: BTreeMap::new(),
            external_edges: Vec::new(),
        };
        let (window, _) =
            build_window_graph_with_order(&subgraph, &graph, None, hint, EdgeStyle::default())
                .expect("should build the window");
        let structure = window
            .structure
            .as_ref()
            .expect("should cache the structure");
        (0..5)
            .map(|node: i64| {
                let domain_index =
                    <DiGraphMap<TestNode, ()> as NodeIndexable>::to_index(&graph, TestNode(node));
                let local = window
                    .graph
                    .node_indices()
                    .find(|&index| {
                        matches!(window.graph.node_weight(index), Some(WindowNode::Data(domain)) if domain.index() == domain_index)
                    })
                    .expect("should retain the data node");
                structure.locate(local).expect("should rank the node").0
            })
            .collect()
    }

    #[test]
    fn test_order_hint_lays_a_cyclic_chain_out_in_hint_order() {
        let hint = |node: TestNode| Some(node.0 * 100);
        let layers = chain_layers_with_hint(Some(&hint));
        assert!(
            layers.windows(2).all(|pair| pair[0] < pair[1]),
            "hinted nodes should rank in hint order, got {layers:?}"
        );
    }
}
