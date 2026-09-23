use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use petgraph::{
    Direction,
    graph::NodeIndex,
    graphmap::DiGraphMap,
    stable_graph::StableDiGraph,
    visit::{GraphBase, IntoNeighborsDirected, IntoNodeIdentifiers, NodeIndexable},
};

use crate::{
    cycle_removal::remove_cycles,
    layout::WindowStructureBuilder,
    window_graph::{WindowEdge, WindowGraph, WindowNode},
};

/// Maps a domain node to `(world_anchor, world_node_budget, world_boundary_node)` for the
/// most recent world that registered it as an external (wormhole) target. See
/// `LayoutEngine::wormhole_targets`.
pub type WormholeTargets<NodeId> = HashMap<NodeId, (NodeId, usize, NodeId)>;

/// Preferred target for each boundary node and direction, retained across window rebuilds.
pub type PreferredWormholeDoors<NodeId> = HashMap<(NodeId, bool), NodeId>;

/// The domain-index pairs of backward edges that landed fully inside a built window. See
/// `build_window_graph`.
pub type WindowBackwardEdges = Vec<(NodeIndex<u32>, NodeIndex<u32>)>;

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

/// Grows a domain graph `G` on demand as a crawl reaches a node whose neighbours aren't yet
/// loaded. This is the one seam through which a crawl ever touches whatever backs `G` -
/// gen-tui itself never needs to know how (or whether) a node's edges get fetched.
pub trait GraphSource<G: GraphBase> {
    /// Ensure `node`'s full neighbourhood (both directions) is present in `graph`, mutating it
    /// in place. Idempotent - a no-op once `node` is already loaded. Returns whether the call
    /// added anything.
    fn ensure_loaded(&mut self, graph: &mut G, node: G::NodeId) -> bool;
}

/// A source for a graph that is already fully loaded in memory, so a crawl never needs to grow
/// `G` - a permanent no-op. The default source for [`crate::layout_engine::LayoutEngine`].
#[derive(Debug, Clone, Copy, Default)]
pub struct EagerSource;

impl<G: GraphBase> GraphSource<G> for EagerSource {
    fn ensure_loaded(&mut self, _graph: &mut G, _node: G::NodeId) -> bool {
        false
    }
}

/// Any `FnMut(&mut G, G::NodeId) -> bool` works as a source too, for callers who would rather
/// pass a closure than name a type.
impl<G, F> GraphSource<G> for F
where
    G: GraphBase,
    F: FnMut(&mut G, G::NodeId) -> bool,
{
    fn ensure_loaded(&mut self, graph: &mut G, node: G::NodeId) -> bool {
        self(graph, node)
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

    /// Ensure `node`'s full neighbourhood is loaded. See [`GraphSource::ensure_loaded`].
    fn ensure_loaded(&mut self, node: G::NodeId) -> bool {
        self.source.ensure_loaded(self.graph, node)
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
    /// Boundary edges collapsed to one wormhole door per node and direction.
    pub external_edges: Vec<ExternalEdge<NodeId>>,
}

/// Select up to `node_budget` nodes around `anchor`, including internal and boundary edges.
/// `cursor` is consulted (via [`GraphSource::ensure_loaded`]) whenever the crawl needs a node's
/// neighbours; pass an [`EagerSource`]-backed cursor for a graph that is already fully loaded.
pub fn neighborhood<G, S>(
    anchor: G::NodeId,
    node_budget: usize,
    cursor: &mut GraphCursor<G, S>,
    forced_include: Option<G::NodeId>,
    wormhole_targets: &WormholeTargets<G::NodeId>,
    preferred_doors: &PreferredWormholeDoors<G::NodeId>,
) -> Result<StructuralSubgraph<G::NodeId>, String>
where
    G: GraphBase,
    G::NodeId: Copy + Eq + Hash + Ord,
    for<'b> &'b G:
        IntoNodeIdentifiers<NodeId = G::NodeId> + IntoNeighborsDirected<NodeId = G::NodeId>,
    S: GraphSource<G>,
{
    cursor.ensure_loaded(anchor);
    if !cursor
        .graph()
        .node_identifiers()
        .any(|node_id| node_id == anchor)
    {
        return Err("Anchor node is not in the graph".to_string());
    }

    let selected = crawl_neighborhood(
        anchor,
        node_budget,
        cursor,
        forced_include,
        wormhole_targets,
    );
    // Every selected node's neighbours are read below to find in-window edges and boundary
    // doors; make sure they are loaded regardless of exactly which nodes `crawl_neighborhood`
    // itself happened to expand along the way (e.g. a zero-budget crawl never touches its own
    // seed's neighbours).
    for &node_id in &selected {
        cursor.ensure_loaded(node_id);
    }
    let graph = cursor.graph();

    // Sorted by the domain node's own identity, not by wherever it landed in the underlying
    // graph structure - so window order is reproducible regardless of insertion order, which
    // a lazily-loaded graph never guarantees (and even an eagerly-loaded one only guaranteed
    // by an explicit whole-graph re-sort this crate no longer performs).
    let mut nodes: Vec<G::NodeId> = selected.iter().copied().collect();
    nodes.sort();

    // Domain edges with both endpoints inside the window. Walking the outgoing neighbours of
    // the selected nodes keeps this bounded by the window rather than the whole graph.
    // Backward edges ride along naturally: the domain graph still carries them, and both
    // endpoints being in the window is enough to include them.
    let mut edges: Vec<(G::NodeId, G::NodeId)> = Vec::new();
    for &node_id in &selected {
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
    for &node_id in &selected {
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

    Ok(StructuralSubgraph {
        anchor,
        nodes,
        edges,
        external_edges,
    })
}

/// Build a self-contained Sugiyama input from a structural window.
///
/// In-window backward edges are rewired through pins. Boundary edges become ranked wormhole
/// nodes. When no explicit backward-edge set is supplied, cycles are detected within the window.
pub fn build_window_graph<G>(
    neighborhood: &StructuralSubgraph<G::NodeId>,
    graph: &G,
    explicit_backward_edges: Option<&HashSet<(G::NodeId, G::NodeId)>>,
) -> Result<(WindowGraph, WindowBackwardEdges), String>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash + Ord + 'static,
{
    let mut window_graph: StableDiGraph<WindowNode, WindowEdge, u32> = StableDiGraph::new();
    let mut node_to_local: HashMap<G::NodeId, NodeIndex<u32>> = HashMap::new();
    let mut domain_to_local: HashMap<NodeIndex<u32>, NodeIndex<u32>> = HashMap::new();

    for &node_id in &neighborhood.nodes {
        let domain_idx = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
        let local_idx = window_graph.add_node(WindowNode::Data(domain_idx));
        node_to_local.insert(node_id, local_idx);
        domain_to_local.insert(domain_idx, local_idx);
    }

    // This window's own backward-edge set: honor an explicit caller-supplied list (filtered to
    // what's actually inside this window) if given, else detect cycles purely within this
    // window's own crawled nodes/edges.
    let window_backward_edges: HashSet<(G::NodeId, G::NodeId)> = match explicit_backward_edges {
        Some(explicit) => explicit
            .iter()
            .copied()
            .filter(|&(source, target)| {
                node_to_local.contains_key(&source) && node_to_local.contains_key(&target)
            })
            .collect(),
        None => {
            let mut local_graph: DiGraphMap<G::NodeId, ()> = DiGraphMap::new();
            for &node_id in &neighborhood.nodes {
                local_graph.add_node(node_id);
            }
            for &(source, target) in &neighborhood.edges {
                local_graph.add_edge(source, target, ());
            }
            remove_cycles(&local_graph, None, None).backward_edges
        }
    };

    // Backward edges are rewired onto pin nodes below, not added as ordinary edges -
    // `neighborhood.edges` still carries them raw (the domain graph does), and adding both
    // would leave the raw cyclic edge in place alongside the bypass.
    let excluded_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)> = window_backward_edges
        .iter()
        .map(|&(source, target)| {
            (
                NodeIndex::new(<G as NodeIndexable>::to_index(graph, source)),
                NodeIndex::new(<G as NodeIndexable>::to_index(graph, target)),
            )
        })
        .collect();

    for &(source, target) in &neighborhood.edges {
        let source_domain = NodeIndex::new(<G as NodeIndexable>::to_index(graph, source));
        let target_domain = NodeIndex::new(<G as NodeIndexable>::to_index(graph, target));
        if excluded_edges.contains(&(source_domain, target_domain)) {
            continue;
        }
        let source_local = node_to_local[&source];
        let target_local = node_to_local[&target];
        window_graph.add_edge(
            source_local,
            target_local,
            Some((source_domain, target_domain)),
        );
    }

    let mut backward_span_edges: HashSet<(NodeIndex<u32>, NodeIndex<u32>)> = HashSet::new();
    let mut backward_bundles: HashSet<(NodeIndex<u32>, NodeIndex<u32>)> = HashSet::new();
    for &(source_domain, target_domain) in &excluded_edges {
        let (Some(&source_local), Some(&target_local)) = (
            domain_to_local.get(&source_domain),
            domain_to_local.get(&target_domain),
        ) else {
            // Only one (or neither) endpoint is in the window - left to
            // `neighborhood.external_edges`, rendered downstream as a wormhole stub.
            continue;
        };

        let bundle = (source_domain, target_domain);
        backward_bundles.insert(bundle);

        let left_pin_idx = window_graph.add_node(WindowNode::Pin);
        let right_pin_idx = window_graph.add_node(WindowNode::Pin);
        window_graph.add_edge(left_pin_idx, target_local, Some(bundle));
        window_graph.add_edge(source_local, right_pin_idx, Some(bundle));
        window_graph.add_edge(left_pin_idx, right_pin_idx, Some(bundle));
        backward_span_edges.insert((left_pin_idx, right_pin_idx));
    }

    // Add each collapsed door as a ranked node on the correct side of its boundary.
    for door in &neighborhood.external_edges {
        let boundary_domain = NodeIndex::new(<G as NodeIndexable>::to_index(graph, door.boundary));
        let target_domain = NodeIndex::new(<G as NodeIndexable>::to_index(graph, door.target));
        let Some(&boundary_local) = domain_to_local.get(&boundary_domain) else {
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
        backward_span_edges,
    };

    let window_backward_edges: Vec<(NodeIndex<u32>, NodeIndex<u32>)> =
        backward_bundles.iter().copied().collect();

    if window.graph.node_count() > 1 {
        let mut builder = WindowStructureBuilder::new(&window);
        builder.build_structure()?;
        window.structure = builder.structure();
    }

    Ok((window, window_backward_edges))
}

/// Alternate outgoing and incoming breadth-first passes until the budget or graph is exhausted.
///
/// Incoming passes start from the full forward reach so they can discover reconverging branches.
/// `forced_include` is added outside the budget, while known wormhole targets remain boundaries.
fn crawl_neighborhood<G, S>(
    anchor: G::NodeId,
    node_budget: usize,
    cursor: &mut GraphCursor<G, S>,
    forced_include: Option<G::NodeId>,
    wormhole_targets: &WormholeTargets<G::NodeId>,
) -> HashSet<G::NodeId>
where
    G: GraphBase,
    for<'b> &'b G: IntoNeighborsDirected<NodeId = G::NodeId>,
    G::NodeId: Copy + Eq + Hash + Ord,
    S: GraphSource<G>,
{
    let mut visited: HashSet<G::NodeId> = HashSet::from([anchor]);
    if let Some(forced) = forced_include {
        cursor.ensure_loaded(forced);
        visited.insert(forced);
    }

    let remaining_budget = node_budget.saturating_sub(visited.len());
    let mut forward_budget = remaining_budget / 2;
    let mut backward_budget = remaining_budget - forward_budget;

    // Give unused budget from each directional pass to the next pass.
    loop {
        let before_len = visited.len();

        let forward_seed: Vec<G::NodeId> = {
            let mut nodes: Vec<G::NodeId> = visited.iter().copied().collect();
            nodes.sort();
            nodes
        };
        directional_bfs(
            &mut visited,
            forward_seed,
            &mut forward_budget,
            cursor,
            Direction::Outgoing,
            wormhole_targets,
            anchor,
        );
        backward_budget += forward_budget;
        forward_budget = 0;

        // Seed from the full reach to find predecessors and reconverging branches.
        let backward_seed: Vec<G::NodeId> = {
            let mut nodes: Vec<G::NodeId> = visited.iter().copied().collect();
            nodes.sort();
            nodes
        };
        directional_bfs(
            &mut visited,
            backward_seed,
            &mut backward_budget,
            cursor,
            Direction::Incoming,
            wormhole_targets,
            anchor,
        );
        forward_budget += backward_budget;
        backward_budget = 0;

        if forward_budget == 0 || visited.len() == before_len {
            break;
        }
    }

    visited
}

/// Crawl breadth-first in one direction, decrementing `budget` for each new node.
fn directional_bfs<G, S>(
    visited: &mut HashSet<G::NodeId>,
    mut current_shell: Vec<G::NodeId>,
    budget: &mut usize,
    cursor: &mut GraphCursor<G, S>,
    direction: Direction,
    wormhole_targets: &WormholeTargets<G::NodeId>,
    anchor: G::NodeId,
) where
    G: GraphBase,
    for<'b> &'b G: IntoNeighborsDirected<NodeId = G::NodeId>,
    G::NodeId: Copy + Eq + Hash + Ord,
    S: GraphSource<G>,
{
    while !current_shell.is_empty() && *budget > 0 {
        for &node_id in &current_shell {
            cursor.ensure_loaded(node_id);
        }
        let graph = cursor.graph();
        let mut candidates: Vec<G::NodeId> = current_shell
            .iter()
            .flat_map(|&node_id| graph.neighbors_directed(node_id, direction))
            .filter(|neighbor| {
                !visited.contains(neighbor)
                    && (*neighbor == anchor || !wormhole_targets.contains_key(neighbor))
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        // No rank bias - just a deterministic tie-break so repeat crawls of the same window
        // are stable, using the domain node's own identity rather than wherever it landed in
        // the underlying graph structure.
        candidates.sort();

        let mut next_shell = Vec::with_capacity(candidates.len());
        for node_id in candidates {
            if *budget == 0 {
                break;
            }
            visited.insert(node_id);
            next_shell.push(node_id);
            *budget -= 1;
        }
        if next_shell.is_empty() {
            break;
        }
        current_shell = next_shell;
    }
}

#[cfg(test)]
mod tests {
    use petgraph::graphmap::DiGraphMap;

    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
    struct TestNode(i64);

    fn make_test_graph(edges: Vec<(i32, i32)>) -> DiGraphMap<TestNode, ()> {
        DiGraphMap::from_edges(
            edges
                .iter()
                .map(|(s, t)| (TestNode(*s as i64), TestNode(*t as i64))),
        )
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
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        let (window, window_backward_edges) =
            build_window_graph(&subgraph, &graph, Some(&backward_edges))
                .expect("should build the pinned acyclic window");

        assert_eq!(
            window_backward_edges.len(),
            1,
            "the one backward edge should land fully inside the window"
        );
        // 4 data nodes + 2 pins.
        assert_eq!(window.graph.node_count(), 6);
        let pin_count = window
            .graph
            .node_indices()
            .filter(|&idx| matches!(window.graph.node_weight(idx), Some(WindowNode::Pin)))
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
            None,
            &HashMap::new(),
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
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect("should find the anchor in the graph");
        let (window, window_backward_edges) = build_window_graph(&subgraph, &graph, None)
            .expect("should build after detecting the local cycle");

        assert_eq!(
            window_backward_edges.len(),
            1,
            "the window's own local cycle should be detected and pinned"
        );
        let pin_count = window
            .graph
            .node_indices()
            .filter(|&idx| matches!(window.graph.node_weight(idx), Some(WindowNode::Pin)))
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

    #[test]
    fn neighborhood_forward_then_backward_crawls_a_chain() {
        // 0 -> 1 -> 2 -> 3 -> 4, anchor 2, budget 3: only the anchor's own budget slot is
        // guaranteed: 1 more forward (finds 3), 1 more backward seeded from {2,3} (finds 1).
        let mut graph = make_test_graph(vec![(0, 1), (1, 2), (2, 3), (3, 4)]);

        let subgraph = neighborhood(
            TestNode(2),
            3,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            None,
            &HashMap::new(),
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
    fn neighborhood_captures_whole_graph_when_budget_is_generous() {
        // Fork/join: 1 and 2 share a layer. Budget covers the entire (tiny) graph, so every
        // node's full neighbourhood is captured and no boundary doors are needed at all.
        let mut graph = make_test_graph(vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

        let subgraph = neighborhood(
            TestNode(0),
            10,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            None,
            &HashMap::new(),
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
                None,
                &HashMap::new(),
                &HashMap::new()
            )
            .is_err()
        );
    }

    #[test]
    fn neighborhood_shared_successor_stays_real_with_one_boundary_door() {
        // A1 and A2 converge on B; C falls outside the three-node budget.
        let mut graph = make_test_graph(vec![(0, 1), (2, 1), (1, 3)]);

        let subgraph = neighborhood(
            TestNode(0),
            3,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();

        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(
            windowed,
            HashSet::from([TestNode(0), TestNode(1), TestNode(2)]),
            "A1 (0), B (1), and A2 (2) should all stay real"
        );
        assert_eq!(
            subgraph.edges.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([(TestNode(0), TestNode(1)), (TestNode(2), TestNode(1))]),
            "both A1's and A2's real edges into B should survive as ordinary window edges"
        );
        assert_eq!(
            subgraph.external_edges,
            vec![ExternalEdge::successor(
                TestNode(1),
                TestNode(3),
                vec![TestNode(3)]
            )],
            "B needs exactly one door, for the one side (successor C) it's actually missing"
        );
    }

    #[test]
    fn neighborhood_forced_include_guarantees_presence_regardless_of_budget() {
        // 0 -> 1 -> 2 -> 3, anchor 3, budget 1: tight enough that an ordinary crawl finds
        // nothing beyond the anchor itself. Forcing 2 in seeds it unconditionally, on top of
        // (not competing with) that budget, guaranteeing it's part of the window regardless -
        // the wormhole-door guarantee `LayoutEngine::activate_world_at`'s `forced_include`
        // relies on when re-entering a window through a door other than the one it was built
        // from.
        let mut graph = make_test_graph(vec![(0, 1), (1, 2), (2, 3)]);

        let subgraph = neighborhood(
            TestNode(3),
            1,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            Some(TestNode(2)),
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();

        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(windowed, HashSet::from([TestNode(2), TestNode(3)]));
        assert_eq!(subgraph.edges, vec![(TestNode(2), TestNode(3))]);
        assert_eq!(
            subgraph.external_edges,
            vec![ExternalEdge::predecessor(
                TestNode(2),
                TestNode(1),
                vec![TestNode(1)]
            )],
            "the forced node's still-missing side should surface as its own external door"
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
            None,
            &HashMap::new(),
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
            None,
            &HashMap::new(),
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
        // R -> M -> {P, Q}; the budget reaches R, M, and P but not Q.
        let mut graph = make_test_graph(vec![(3, 0), (0, 1), (0, 2)]);
        // Anchor 0 (M); backward reaches R (3), forward reaches P (1) but budget runs out
        // before Q (2).
        let subgraph = neighborhood(
            TestNode(0),
            3,
            &mut GraphCursor::new(&mut graph, &mut EagerSource),
            None,
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();

        let windowed: HashSet<TestNode> = subgraph.nodes.iter().copied().collect();
        assert_eq!(
            windowed,
            HashSet::from([TestNode(0), TestNode(1), TestNode(3)]),
            "M, P, and R should all stay in the window"
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
            None,
            &HashMap::new(),
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
}
