use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use petgraph::{
    Direction,
    graph::NodeIndex,
    visit::{
        EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeCount, NodeIndexable, Visitable,
    },
};

use crate::{
    assembly::{AssembledLayout, assemble_window},
    crawl::{
        EagerSource, ExternalEdge, GraphCursor, GraphSource, PreferredWormholeDoors,
        WormholeTargets, build_window_graph, neighborhood,
    },
    layout::NodeRole,
};

/// The exact structural request that identifies a loaded rendering world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldKey<NodeId> {
    /// Domain node from which the structural crawl starts.
    pub anchor: NodeId,
    /// Fixed crawl budget chosen when the world is first requested.
    pub node_budget: usize,
    /// Optional boundary node that must be present even when it exceeds the budget.
    pub forced_include: Option<NodeId>,
}

/// One immutable structural world owned by [`LayoutEngine`]. Geometry is deliberately absent:
/// each view clones `layout` and applies its own renderer, gaps, area, and camera every frame.
#[derive(Clone)]
pub struct LoadedWorld<NodeId> {
    key: WorldKey<NodeId>,
    members: HashSet<NodeId>,
    external_edges: Vec<(NodeId, NodeId)>,
    layout: AssembledLayout,
}

impl<NodeId: Copy + Eq + Hash> LoadedWorld<NodeId> {
    /// Return the exact request that constructed this world.
    pub fn key(&self) -> WorldKey<NodeId> {
        self.key
    }

    /// Return whether `node` is a structural member of this world.
    pub fn contains(&self, node: NodeId) -> bool {
        self.members.contains(&node)
    }

    /// Iterate over the exact domain-node membership.
    pub fn members(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.members.iter().copied()
    }

    /// Return the neutral assembled layout cloned by renderers each frame.
    pub fn layout(&self) -> &AssembledLayout {
        &self.layout
    }

    /// Find the in-world boundary node whose collapsed edge points at `target`.
    pub fn boundary_for(&self, target: NodeId) -> Option<NodeId> {
        self.external_edges
            .iter()
            .find_map(|&(boundary, outside)| (outside == target).then_some(boundary))
    }
}

/// Divisor applied to the current viewport width (in columns) to derive the node budget for
/// the directional BFS crawl that builds the size-independent rendering neighbourhood (see
/// `crawl::neighborhood`). A smaller viewport keeps a smaller window.
const NEIGHBORHOOD_NODE_BUDGET_DIVISOR: usize = 3;
/// Floor under the derived budget, so a momentarily zero-width (uninitialized) viewport still
/// produces a usable window instead of degenerating to just the anchor node.
const MIN_NEIGHBORHOOD_NODE_BUDGET: usize = 10;

/// Maximum number of recently-built rendering worlds kept in `LayoutEngine::world_cache`.
/// A window's Sugiyama structure is cheap to rebuild (bounded by the neighbourhood's
/// `node_budget`, not by the size of the whole graph), so this only needs to be big enough to
/// make redrawing an unchanged window free.
const WORLD_CACHE_CAPACITY: usize = 8;

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StructuralBuildCounts {
    pub crawl: usize,
    pub sugiyama: usize,
    pub assembly: usize,
}

/// Owns the domain graph, one explicit active structural world, and a small LRU of previously
/// visited worlds. Multiple views consume the same active world without structural rebuilding.
///
/// `S` is how the engine grows `graph` on demand when a crawl reaches an unloaded node - see
/// [`crate::crawl::GraphSource`]. It defaults to [`EagerSource`], a no-op, for the common case
/// of an already-fully-loaded graph; a lazily-loaded graph plugs in its own source via
/// [`LayoutEngine::new_with_source`] instead. Either way there is exactly one crawl
/// implementation (`crawl::neighborhood`) underneath.
#[derive(Clone)]
pub struct LayoutEngine<G, S = EagerSource>
where
    G: GraphBase,
{
    graph: G,
    source: S,
    /// Anchor `ensure_initial_world` should open on instead of [`Self::default_anchor`]'s
    /// structural "index 0" guess - set by [`Self::set_preferred_initial_anchor`]. gen-tui has
    /// no notion of a domain-meaningful "start" (that's caller-specific, e.g. a sequence
    /// graph's `PATH_START` sentinel), so a caller that knows one sets it explicitly instead of
    /// this crate reordering the graph to fake a structural index for it.
    preferred_initial_anchor: Option<G::NodeId>,
    /// The caller's own already-known backward (cycle-closing) domain edges, if any (e.g.
    /// from GFA import's circular-genome/reverse-complement preprocessing). When `None`, each
    /// window detects its own backward edges by running cycle detection on just its own
    /// crawled nodes/edges - see `crawl::build_window_graph`.
    explicit_backward_edges: Option<HashSet<(G::NodeId, G::NodeId)>>,
    /// Recently-built worlds, in recency order (front = least recently used).
    world_cache: Vec<LoadedWorld<G::NodeId>>,
    active_world: Option<WorldKey<G::NodeId>>,
    /// Membership map used by the crawler to stop at known worlds.
    wormhole_targets: WormholeTargets<G::NodeId>,
    /// Exact construction request for each known wormhole target. This survives LRU eviction,
    /// allowing a revisited world to be rebuilt identically.
    wormhole_worlds: HashMap<G::NodeId, WorldKey<G::NodeId>>,
    /// Which specific off-window neighbour each boundary node's collapsed wormhole door should
    /// prefer, per side - set by `remember_wormhole_choice` whenever a door is actually
    /// traveled through, consulted by `neighborhood`'s external-edge collapsing on the next
    /// (re)build. See `crawl::PreferredWormholeDoors`.
    preferred_wormhole_doors: PreferredWormholeDoors<G::NodeId>,
    #[cfg(test)]
    build_counts: StructuralBuildCounts,
}

/// Constructors for the common case: `graph` is already fully loaded, so the engine never
/// needs to grow it and uses the no-op [`EagerSource`].
impl<G> LayoutEngine<G, EagerSource>
where
    G: GraphBase + EdgeIndexable + NodeIndexable + NodeCount + Visitable,
    G::NodeId: Copy + Eq + Hash + Ord + 'static,
    G::EdgeId: Clone,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNodeIdentifiers<NodeId = G::NodeId>
        + IntoEdgeReferences<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNeighborsDirected<NodeId = G::NodeId>,
    for<'b> &'b G::NodeId: Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
{
    /// Create a new LayoutEngine starting from an already-loaded graph. Backward
    /// (cycle-closing) edges, if any, are auto-detected per window as it is built - see
    /// `explicit_backward_edges`.
    pub fn new(graph: G) -> Self {
        Self::new_with_source(graph, EagerSource)
    }

    /// Create a new LayoutEngine, rewiring each backward edge `(source, target)` in
    /// `backward_edges` onto a pair of pin nodes within whichever window(s) it lands in
    /// fully, so it renders as a loop instead of crashing the (DAG-only) layout pipeline.
    pub fn new_with_backward_edges(graph: G, backward_edges: &[(G::NodeId, G::NodeId)]) -> Self {
        Self::new_with_source_and_backward_edges(graph, EagerSource, backward_edges)
    }
}

impl<G, S> LayoutEngine<G, S>
where
    G: GraphBase + EdgeIndexable + NodeIndexable + NodeCount + Visitable,
    G::NodeId: Copy + Eq + Hash + Ord + 'static,
    G::EdgeId: Clone,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNodeIdentifiers<NodeId = G::NodeId>
        + IntoEdgeReferences<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNeighborsDirected<NodeId = G::NodeId>,
    for<'b> &'b G::NodeId: Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
    S: GraphSource<G>,
{
    /// Create a new LayoutEngine that lazily grows `graph` via `source` whenever a crawl
    /// reaches a node it hasn't loaded yet - see [`crate::crawl::GraphSource`]. Backward
    /// (cycle-closing) edges, if any, are auto-detected per window - see
    /// `explicit_backward_edges`.
    pub fn new_with_source(graph: G, source: S) -> Self {
        Self {
            graph,
            source,
            preferred_initial_anchor: None,
            explicit_backward_edges: None,
            world_cache: Vec::new(),
            active_world: None,
            wormhole_targets: HashMap::new(),
            wormhole_worlds: HashMap::new(),
            preferred_wormhole_doors: HashMap::new(),
            #[cfg(test)]
            build_counts: StructuralBuildCounts::default(),
        }
    }

    /// [`Self::new_with_source`], additionally rewiring known backward edges - see
    /// [`Self::new_with_backward_edges`].
    pub fn new_with_source_and_backward_edges(
        graph: G,
        source: S,
        backward_edges: &[(G::NodeId, G::NodeId)],
    ) -> Self {
        Self {
            graph,
            source,
            preferred_initial_anchor: None,
            explicit_backward_edges: Some(backward_edges.iter().copied().collect()),
            world_cache: Vec::new(),
            active_world: None,
            wormhole_targets: HashMap::new(),
            wormhole_worlds: HashMap::new(),
            preferred_wormhole_doors: HashMap::new(),
            #[cfg(test)]
            build_counts: StructuralBuildCounts::default(),
        }
    }

    /// Get a reference to the original domain graph.
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// Mutably borrow the underlying domain graph, bypassing [`Self::source`] entirely -
    /// without otherwise touching the active world or window cache, unlike
    /// [`Self::window_for`]/[`Self::activate_world_at`], which both also navigate there. For a
    /// caller that needs to merge in data its own way rather than through the crawl's usual
    /// [`crate::crawl::GraphSource::ensure_loaded`] path - e.g. gen-python's `show_path`, which
    /// must be able to project a stored path onto the graph regardless of whether the active
    /// source is filtering some of that path's own edges out (see `SqlGraphSource::new_pruned`).
    /// A caller's designated "current path" is not guaranteed to be made up of exactly the
    /// edges a display-oriented pruning policy would keep.
    pub fn graph_mut(&mut self) -> &mut G {
        &mut self.graph
    }

    /// The rendering neighbourhood's node budget for a viewport of the given width
    /// (columns). See `NEIGHBORHOOD_NODE_BUDGET_DIVISOR`.
    pub fn neighborhood_node_budget(&self, viewport_width: usize) -> usize {
        (viewport_width / NEIGHBORHOOD_NODE_BUDGET_DIVISOR.max(1)).max(MIN_NEIGHBORHOOD_NODE_BUDGET)
    }

    /// Pick a default anchor: the domain graph's own lowest-index node. Deliberately not a
    /// "true" topologically-first node - that would need a whole-graph rank pass, which this
    /// engine never runs. `O(1)`: no scan, not even an in-degree check.
    pub fn default_anchor(&self) -> Option<G::NodeId> {
        if <G as NodeCount>::node_count(&self.graph) == 0 {
            return None;
        }
        Some(<G as NodeIndexable>::from_index(&self.graph, 0))
    }

    /// Record the anchor `ensure_initial_world` should open on, overriding
    /// [`Self::default_anchor`]'s structural guess. Callers that know a domain-meaningful
    /// starting point (e.g. a sequence graph's start sentinel) set it once after construction
    /// instead of gen-tui reordering the graph to fake a matching structural index.
    pub fn set_preferred_initial_anchor(&mut self, anchor: G::NodeId) {
        self.preferred_initial_anchor = Some(anchor);
    }

    /// Ensure the graph has an active world, choosing an anchor and a budget fixed from
    /// `viewport_width` only when no world has been loaded yet. Prefers
    /// [`Self::set_preferred_initial_anchor`]'s choice, falling back to [`Self::default_anchor`].
    pub fn ensure_initial_world(
        &mut self,
        viewport_width: usize,
    ) -> Result<Option<WorldKey<G::NodeId>>, String> {
        if let Some(key) = self.active_world {
            return Ok(Some(key));
        }
        let Some(anchor) = self
            .preferred_initial_anchor
            .or_else(|| self.default_anchor())
        else {
            return Ok(None);
        };
        let budget = self.neighborhood_node_budget(viewport_width);
        self.activate_world_at(anchor, budget, None).map(Some)
    }

    /// Build or reuse the exact requested world and make it active. Failed construction leaves
    /// the previous active world unchanged.
    pub fn activate_world_at(
        &mut self,
        anchor: G::NodeId,
        node_budget: usize,
        forced_include: Option<G::NodeId>,
    ) -> Result<WorldKey<G::NodeId>, String> {
        let key = WorldKey {
            anchor,
            node_budget,
            forced_include,
        };
        self.activate_known_world(key)?;
        Ok(key)
    }

    /// Activate a previously recorded exact world request, rebuilding it if its LRU entry was
    /// evicted.
    pub fn activate_known_world(&mut self, key: WorldKey<G::NodeId>) -> Result<(), String> {
        if let Some(position) = self.world_cache.iter().position(|world| world.key == key) {
            let world = self.world_cache.remove(position);
            self.world_cache.push(world);
            self.active_world = Some(key);
            return Ok(());
        }

        let subgraph = neighborhood(
            key.anchor,
            key.node_budget,
            &mut GraphCursor::new(&mut self.graph, &mut self.source),
            key.forced_include,
            &self.wormhole_targets,
            &self.preferred_wormhole_doors,
        )?;
        #[cfg(test)]
        {
            self.build_counts.crawl += 1;
        }
        let (window, backward_edges) = build_window_graph(
            &subgraph,
            &self.graph,
            self.explicit_backward_edges.as_ref(),
        )?;
        #[cfg(test)]
        {
            self.build_counts.sugiyama += 1;
        }
        let mut layout = assemble_window(&window).ok_or_else(|| {
            "assemble_window produced no window for a loaded neighbourhood".to_string()
        })?;
        #[cfg(test)]
        {
            self.build_counts.assembly += 1;
        }
        let to_node_index = |node_id: G::NodeId| {
            NodeIndex::new(<G as NodeIndexable>::to_index(&self.graph, node_id))
        };
        let anchor_node_index = to_node_index(key.anchor);
        layout.anchor_index = layout.graph.node_indices().find(|&node_index| {
            matches!(
                layout.graph[node_index].role,
                NodeRole::Data(domain_index) if domain_index == anchor_node_index
            )
        });
        layout.external_edges = subgraph
            .external_edges
            .iter()
            .map(|edge| ExternalEdge {
                boundary: to_node_index(edge.boundary),
                target: to_node_index(edge.target),
                all_collapsed: edge
                    .all_collapsed
                    .iter()
                    .map(|&node_id| to_node_index(node_id))
                    .collect(),
                side: edge.side,
            })
            .collect();
        layout.backward_edges = backward_edges;

        let world = LoadedWorld {
            key,
            members: subgraph.nodes.iter().copied().collect(),
            external_edges: subgraph
                .external_edges
                .iter()
                .map(|edge| (edge.boundary, edge.target))
                .collect(),
            layout,
        };

        self.world_cache.push(world);
        self.active_world = Some(key);
        while self.world_cache.len() > WORLD_CACHE_CAPACITY {
            let Some(position) = self
                .world_cache
                .iter()
                .position(|world| Some(world.key) != self.active_world)
            else {
                break;
            };
            self.world_cache.remove(position);
        }

        // Register membership only after every structural construction stage succeeded. An
        // external target is new territory; a later world's external edge back to one of these
        // members is what identifies this as a known return world.
        for &member in &subgraph.nodes {
            self.wormhole_targets
                .insert(member, (key.anchor, key.node_budget, member));
            self.wormhole_worlds.insert(member, key);
        }
        Ok(())
    }

    /// Return the exact key of the active structural world.
    pub fn active_world_key(&self) -> Option<WorldKey<G::NodeId>> {
        self.active_world
    }

    /// Return the active immutable structural world.
    pub fn active_world(&self) -> Option<&LoadedWorld<G::NodeId>> {
        let key = self.active_world?;
        self.world_cache.iter().find(|world| world.key == key)
    }

    /// Return whether `node` belongs to the active structural world.
    pub fn active_contains(&self, node: G::NodeId) -> bool {
        self.active_world()
            .is_some_and(|world| world.contains(node))
    }

    #[cfg(test)]
    pub(crate) fn structural_build_counts(&self) -> StructuralBuildCounts {
        self.build_counts
    }

    /// Compatibility helper that activates an ordinary world and returns a layout clone.
    /// Persistent views should consume [`Self::active_world`] instead.
    pub fn window_for(
        &mut self,
        anchor: G::NodeId,
        node_budget: usize,
    ) -> Result<AssembledLayout, String> {
        self.activate_world_at(anchor, node_budget, None)?;
        self.active_world()
            .map(|world| world.layout.clone())
            .ok_or_else(|| "activated world is missing from the cache".to_string())
    }

    /// Look up the exact known world containing a wormhole target, if one has been built.
    pub fn wormhole_world_for(&self, target: G::NodeId) -> Option<WorldKey<G::NodeId>> {
        self.wormhole_worlds.get(&target).copied()
    }

    /// Whether `to` is a direct successor of `from` in the domain graph (an outgoing edge
    /// `from -> to`), as opposed to a predecessor (`to -> from`). `from` and `to` are always
    /// directly adjacent when this matters in practice - click-to-teleport calls this on a
    /// boundary node and its off-screen wormhole target, which are adjacent by construction
    /// (that's what makes them an external-edge pair) - so this is an `O(degree)` local check,
    /// no whole-graph rank axis needed to answer "is the target ahead of or behind `from`."
    pub fn is_successor(&self, from: G::NodeId, to: G::NodeId) -> bool {
        self.graph
            .neighbors_directed(from, Direction::Outgoing)
            .any(|neighbor| neighbor == to)
    }

    /// Record that `boundary`'s collapsed wormhole door on the successor side
    /// (`exits_toward_successor == true`) or predecessor side (`false`) was just actually
    /// traveled through via `target`. Consulted the next time a window is (re)built with
    /// `boundary` on its edge, so re-collapsing its off-window neighbours on that side keeps
    /// landing on this same world instead of an arbitrary sibling - see
    /// `crawl::PreferredWormholeDoors`. Click-to-teleport calls this for every door it actually
    /// uses; a door's preference otherwise stays whatever the last recorded choice was, or the
    /// deterministic lowest-index default if it was never traveled through.
    pub fn remember_wormhole_choice(
        &mut self,
        boundary: G::NodeId,
        target: G::NodeId,
        exits_toward_successor: bool,
    ) {
        self.preferred_wormhole_doors
            .insert((boundary, exits_toward_successor), target);
    }
}

#[cfg(test)]
mod tests {
    use petgraph::{graph::NodeIndex, visit::NodeIndexable};

    use crate::{
        layout::NodeRole,
        layout_engine::{LayoutEngine, StructuralBuildCounts, WORLD_CACHE_CAPACITY, WorldKey},
        testing::mocks::{MockDomainGraph, TestGraphs},
    };

    #[test]
    fn test_window_for_sets_anchor_index() {
        let domain_graph = TestGraphs::domain_complex_dag();
        let mut engine = LayoutEngine::new(domain_graph);
        let anchor = engine.default_anchor().expect("should have graph nodes");

        let window = engine
            .window_for(anchor, 10)
            .expect("should build a window");
        let anchor_index = window
            .anchor_index
            .expect("should place the anchor in its window");
        assert!(matches!(
            window.graph[anchor_index].role,
            NodeRole::Data(domain_index) if domain_index == NodeIndex::new(<crate::testing::mocks::MockDomainGraph as NodeIndexable>::to_index(engine.graph(), anchor))
        ));
    }

    /// Wide, shallow graphs must remain bounded by the node budget.
    #[test]
    fn test_window_for_bounds_wide_shallow_graph() {
        let mut domain_graph = MockDomainGraph::default();
        const RANKS: usize = 4;
        const WIDTH: usize = 150;
        let mut layers: Vec<Vec<NodeIndex<u32>>> = Vec::new();
        for _ in 0..RANKS {
            let layer: Vec<NodeIndex<u32>> =
                (0..WIDTH).map(|_| domain_graph.add_node(())).collect();
            layers.push(layer);
        }
        // Wide fan-out/fan-in between ranks, not a clean one-to-one chain - closer to a real
        // assembly graph's branching than a linear stand-in would be.
        for rank in 0..RANKS - 1 {
            for (i, &source) in layers[rank].iter().enumerate() {
                for offset in 0..3 {
                    let target = layers[rank + 1][(i + offset) % WIDTH];
                    domain_graph.add_edge(source, target, ());
                }
            }
        }
        let total_nodes = RANKS * WIDTH;

        let mut engine = LayoutEngine::new(domain_graph);
        let anchor = layers[0][0];
        let node_budget = engine.neighborhood_node_budget(120);

        let window = engine
            .window_for(anchor, node_budget)
            .expect("should build a wide, shallow window");
        let data_node_count = window
            .graph
            .node_weights()
            .filter(|node| matches!(node.role, NodeRole::Data(_)))
            .count();

        assert!(
            data_node_count <= node_budget,
            "assembled window ({data_node_count} nodes) must not exceed the requested budget ({node_budget})"
        );
        assert!(
            data_node_count < total_nodes / 2,
            "the crawl must not degenerate into pulling in most of a wide/shallow graph: \
             got {data_node_count} of {total_nodes} total nodes"
        );
    }

    #[test]
    fn active_world_retrieval_does_not_repeat_structural_work() {
        let domain_graph = TestGraphs::domain_complex_dag();
        let mut engine = LayoutEngine::new(domain_graph);
        let key = engine
            .ensure_initial_world(80)
            .expect("should build an initial world")
            .expect("should have an initial world");

        let once = StructuralBuildCounts {
            crawl: 1,
            sugiyama: 1,
            assembly: 1,
        };
        assert_eq!(engine.structural_build_counts(), once);

        for _ in 0..5 {
            assert_eq!(engine.active_world_key(), Some(key));
            assert!(engine.active_world().is_some());
            engine
                .activate_known_world(key)
                .expect("should reactivate the cached world");
        }
        assert_eq!(engine.structural_build_counts(), once);
    }

    #[test]
    fn forced_include_is_part_of_the_exact_world_key() {
        let mut graph = MockDomainGraph::new();
        let first = graph.add_node(());
        let second = graph.add_node(());
        graph.add_edge(first, second, ());
        let mut engine = LayoutEngine::new(graph);

        let ordinary = engine
            .activate_world_at(first, 1, None)
            .expect("should build the ordinary world");
        let forced = engine
            .activate_world_at(first, 1, Some(second))
            .expect("should build the forced-inclusion world");

        assert_ne!(ordinary, forced);
        assert!(engine.active_contains(second));
        assert_eq!(engine.structural_build_counts().crawl, 2);
    }

    #[test]
    fn evicted_world_rebuilds_from_its_exact_key() {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..WORLD_CACHE_CAPACITY + 2)
            .map(|_| graph.add_node(()))
            .collect();
        for pair in nodes.windows(2) {
            graph.add_edge(pair[0], pair[1], ());
        }
        let mut engine = LayoutEngine::new(graph);
        let first_key = WorldKey {
            anchor: nodes[0],
            node_budget: 1,
            forced_include: None,
        };

        for &anchor in &nodes[..=WORLD_CACHE_CAPACITY] {
            engine
                .activate_world_at(anchor, 1, None)
                .expect("should build a small world");
        }

        let active_key = engine
            .active_world_key()
            .expect("should have an active world");
        assert!(
            engine
                .world_cache
                .iter()
                .any(|world| world.key == active_key)
        );
        assert_eq!(engine.world_cache.len(), WORLD_CACHE_CAPACITY);
        assert!(
            !engine
                .world_cache
                .iter()
                .any(|world| world.key == first_key)
        );
        let builds_before_reentry = engine.structural_build_counts().crawl;

        engine
            .activate_known_world(first_key)
            .expect("should rebuild the evicted exact world");

        assert_eq!(engine.active_world_key(), Some(first_key));
        assert!(
            engine
                .world_cache
                .iter()
                .any(|world| world.key == first_key)
        );
        assert_eq!(
            engine.structural_build_counts().crawl,
            builds_before_reentry + 1
        );
    }

    #[test]
    fn wormhole_loads_new_territory_and_returns_to_known_world() {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..6).map(|_| graph.add_node(())).collect();
        for pair in nodes.windows(2) {
            graph.add_edge(pair[0], pair[1], ());
        }
        let mut engine = LayoutEngine::new(graph);
        let first_key = engine
            .activate_world_at(nodes[0], 2, None)
            .expect("should build the first world");

        assert_eq!(
            engine.wormhole_world_for(nodes[2]),
            None,
            "an off-world target should represent new territory"
        );

        let second_key = engine
            .activate_world_at(nodes[2], 2, Some(nodes[1]))
            .expect("should build a new forced-inclusion world");
        assert_ne!(first_key, second_key);
        assert!(engine.active_contains(nodes[1]));
        assert_eq!(
            engine.wormhole_world_for(nodes[0]),
            Some(first_key),
            "the return target should resolve to its exact known world"
        );

        engine
            .activate_known_world(first_key)
            .expect("should return to the known world");
        assert!(engine.active_contains(nodes[0]));
        assert_eq!(engine.structural_build_counts().crawl, 2);
    }
}
