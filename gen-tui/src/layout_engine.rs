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
        build_window_graph, crawl_batch, subgraph_for,
    },
    layout::NodeRole,
};

/// Identifies one batch: a chunk of the domain graph claimed by a single crawl. Batches never
/// overlap, and a node keeps its batch for the engine's whole lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BatchId(usize);

/// The nodes one crawl claimed, in domain order, and the node it started from.
#[derive(Clone)]
struct Batch<NodeId> {
    anchor: NodeId,
    members: Vec<NodeId>,
}

/// One batch's immutable structural world. Geometry is deliberately absent: each view clones
/// `layout` and applies its own renderer, gaps, area, and camera every frame.
#[derive(Clone)]
pub struct LoadedWorld<NodeId> {
    batch: BatchId,
    anchor: NodeId,
    members: HashSet<NodeId>,
    external_edges: Vec<(NodeId, NodeId)>,
    layout: AssembledLayout,
}

impl<NodeId: Copy + Eq + Hash> LoadedWorld<NodeId> {
    /// The batch this world renders.
    pub fn batch(&self) -> BatchId {
        self.batch
    }

    /// The node the batch's crawl started from.
    pub fn anchor(&self) -> NodeId {
        self.anchor
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
/// the directional BFS crawl that claims a new batch (see `crawl::crawl_batch`). A smaller
/// viewport claims smaller batches.
const NEIGHBORHOOD_NODE_BUDGET_DIVISOR: usize = 3;
/// Floor under the derived budget, so a momentarily zero-width (uninitialized) viewport still
/// produces a usable window instead of degenerating to just the anchor node.
const MIN_NEIGHBORHOOD_NODE_BUDGET: usize = 10;

/// Maximum number of batch layouts kept in `LayoutEngine::world_cache`. A batch's Sugiyama
/// structure is cheap to rebuild from its stored members (bounded by the batch size, not by the
/// size of the whole graph), so this only needs to be big enough to make redrawing an unchanged
/// window free.
const WORLD_CACHE_CAPACITY: usize = 8;

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StructuralBuildCounts {
    pub crawl: usize,
    pub sugiyama: usize,
    pub assembly: usize,
}

/// Owns the domain graph, the batches it has been carved into so far, and a small LRU of their
/// built worlds. Multiple views consume the same active world without structural rebuilding.
///
/// The graph is rendered one batch at a time. Each crawl claims a fixed-size chunk of nodes no
/// other batch owns; wherever a batch borders another batch, or nodes nobody has claimed yet,
/// its window shows a wormhole door. Stepping through a door into claimed territory reopens that
/// batch; stepping into unclaimed territory claims a new one.
///
/// `S` is how the engine grows `graph` on demand when a crawl reaches a frontier node - see
/// [`crate::crawl::GraphSource`]. It defaults to [`EagerSource`], a no-op, for the common case
/// of an already-fully-loaded graph; a lazily-loaded graph plugs in its own source via
/// [`LayoutEngine::new_with_source`] instead.
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
    /// Every batch claimed so far, indexed by `BatchId`.
    batches: Vec<Batch<G::NodeId>>,
    /// Which batch each claimed node belongs to.
    batch_of: HashMap<G::NodeId, BatchId>,
    /// Recently-built worlds, in recency order (front = least recently used).
    world_cache: Vec<LoadedWorld<G::NodeId>>,
    active_batch: Option<BatchId>,
    /// Which specific off-window neighbour each boundary node's collapsed wormhole door should
    /// prefer, per side - set by `remember_wormhole_choice` whenever a door is actually
    /// traveled through, consulted when a batch's world is (re)built. See
    /// `crawl::PreferredWormholeDoors`.
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
    /// needs to go past a frontier node - see [`crate::crawl::GraphSource`]. Backward
    /// (cycle-closing) edges, if any, are auto-detected per window - see
    /// `explicit_backward_edges`.
    pub fn new_with_source(graph: G, source: S) -> Self {
        Self {
            graph,
            source,
            preferred_initial_anchor: None,
            explicit_backward_edges: None,
            batches: Vec::new(),
            batch_of: HashMap::new(),
            world_cache: Vec::new(),
            active_batch: None,
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
            explicit_backward_edges: Some(backward_edges.iter().copied().collect()),
            ..Self::new_with_source(graph, source)
        }
    }

    /// Get a reference to the original domain graph.
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// The node budget for a new batch claimed while the viewport is the given width
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
    /// instead of gen-tui reordering the graph to fake a matching structural index for it.
    pub fn set_preferred_initial_anchor(&mut self, anchor: G::NodeId) {
        self.preferred_initial_anchor = Some(anchor);
    }

    /// Ensure the graph has an active world, claiming the first batch with a budget fixed from
    /// `viewport_width` only when none is active yet. Prefers
    /// [`Self::set_preferred_initial_anchor`]'s choice, falling back to [`Self::default_anchor`].
    pub fn ensure_initial_world(
        &mut self,
        viewport_width: usize,
    ) -> Result<Option<BatchId>, String> {
        if let Some(batch) = self.active_batch {
            return Ok(Some(batch));
        }
        let Some(anchor) = self
            .preferred_initial_anchor
            .or_else(|| self.default_anchor())
        else {
            return Ok(None);
        };
        let budget = self.neighborhood_node_budget(viewport_width);
        self.activate_batch_containing(anchor, budget).map(Some)
    }

    /// Make the batch containing `node` active, claiming a new batch of up to `node_budget`
    /// nodes around it if no batch owns it yet. Failed construction leaves the previous active
    /// world unchanged.
    pub fn activate_batch_containing(
        &mut self,
        node: G::NodeId,
        node_budget: usize,
    ) -> Result<BatchId, String> {
        if let Some(&batch) = self.batch_of.get(&node) {
            self.activate_batch(batch)?;
            return Ok(batch);
        }

        let batch_of = &self.batch_of;
        let members = crawl_batch(
            node,
            node_budget,
            &mut GraphCursor::new(&mut self.graph, &mut self.source),
            &|candidate| batch_of.contains_key(&candidate),
        )?;
        #[cfg(test)]
        {
            self.build_counts.crawl += 1;
        }
        let batch = BatchId(self.batches.len());
        for &member in &members {
            self.batch_of.insert(member, batch);
        }
        self.batches.push(Batch {
            anchor: node,
            members,
        });
        self.activate_batch(batch)?;
        Ok(batch)
    }

    /// Make `batch` active, rebuilding its world from its stored members if the LRU evicted
    /// it. Rebuilding never crawls: a member's edges were fully loaded when it was claimed, so
    /// the same members always yield the same window.
    pub fn activate_batch(&mut self, batch: BatchId) -> Result<(), String> {
        if let Some(position) = self
            .world_cache
            .iter()
            .position(|world| world.batch == batch)
        {
            let world = self.world_cache.remove(position);
            self.world_cache.push(world);
            self.active_batch = Some(batch);
            return Ok(());
        }

        let Batch { anchor, members } = self
            .batches
            .get(batch.0)
            .cloned()
            .ok_or_else(|| format!("unknown batch {batch:?}"))?;
        let subgraph = subgraph_for(anchor, members, &self.graph, &self.preferred_wormhole_doors);
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
        let anchor_node_index = to_node_index(anchor);
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

        self.world_cache.push(LoadedWorld {
            batch,
            anchor,
            members: subgraph.nodes.iter().copied().collect(),
            external_edges: subgraph
                .external_edges
                .iter()
                .map(|edge| (edge.boundary, edge.target))
                .collect(),
            layout,
        });
        self.active_batch = Some(batch);
        while self.world_cache.len() > WORLD_CACHE_CAPACITY {
            let Some(position) = self
                .world_cache
                .iter()
                .position(|world| Some(world.batch) != self.active_batch)
            else {
                break;
            };
            self.world_cache.remove(position);
        }
        Ok(())
    }

    /// The batch whose world is active.
    pub fn active_batch(&self) -> Option<BatchId> {
        self.active_batch
    }

    /// Return the active immutable structural world.
    pub fn active_world(&self) -> Option<&LoadedWorld<G::NodeId>> {
        let batch = self.active_batch?;
        self.world_cache.iter().find(|world| world.batch == batch)
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

    /// Compatibility helper that activates the batch containing `anchor` and returns a layout
    /// clone. Persistent views should consume [`Self::active_world`] instead.
    pub fn window_for(
        &mut self,
        anchor: G::NodeId,
        node_budget: usize,
    ) -> Result<AssembledLayout, String> {
        self.activate_batch_containing(anchor, node_budget)?;
        self.active_world()
            .map(|world| world.layout.clone())
            .ok_or_else(|| "activated world is missing from the cache".to_string())
    }

    /// The batch that claimed `node`, if any has.
    pub fn batch_of(&self, node: G::NodeId) -> Option<BatchId> {
        self.batch_of.get(&node).copied()
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
    /// traveled through via `target`. Consulted the next time a batch's world is built with
    /// `boundary` on its edge, so re-collapsing its off-window neighbours on that side keeps
    /// landing on this same batch instead of an arbitrary sibling - see
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
        layout_engine::{LayoutEngine, StructuralBuildCounts, WORLD_CACHE_CAPACITY},
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
        let batch = engine
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
            assert_eq!(engine.active_batch(), Some(batch));
            assert!(engine.active_world().is_some());
            engine
                .activate_batch(batch)
                .expect("should reactivate the cached world");
        }
        assert_eq!(engine.structural_build_counts(), once);
    }

    fn chain(length: usize) -> (MockDomainGraph, Vec<NodeIndex<u32>>) {
        let mut graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..length).map(|_| graph.add_node(())).collect();
        for pair in nodes.windows(2) {
            graph.add_edge(pair[0], pair[1], ());
        }
        (graph, nodes)
    }

    /// The chain position just past the last node `batch` claimed.
    fn index_after(
        engine: &LayoutEngine<MockDomainGraph>,
        nodes: &[NodeIndex<u32>],
        batch: super::BatchId,
    ) -> usize {
        nodes
            .iter()
            .rposition(|node| engine.batch_of(*node) == Some(batch))
            .expect("the batch should own at least its anchor")
            + 1
    }

    #[test]
    fn test_batches_never_overlap() {
        let (graph, nodes) = chain(12);
        let mut engine = LayoutEngine::new(graph);
        let first = engine
            .activate_batch_containing(nodes[0], 4)
            .expect("should claim the first batch");
        let next = index_after(&engine, &nodes, first);

        let second = engine
            .activate_batch_containing(nodes[next], 6)
            .expect("should claim a batch next to the first");

        assert_ne!(first, second);
        for &node in &nodes[..next] {
            assert_eq!(engine.batch_of(node), Some(first));
            assert!(
                !engine.active_contains(node),
                "a new batch should never take a node another batch already owns"
            );
        }
    }

    #[test]
    fn test_claimed_node_reopens_its_batch_without_crawling() {
        let (graph, nodes) = chain(12);
        let mut engine = LayoutEngine::new(graph);
        let first = engine
            .activate_batch_containing(nodes[0], 3)
            .expect("should claim the first batch");
        let next = index_after(&engine, &nodes, first);
        engine
            .activate_batch_containing(nodes[next], 3)
            .expect("should claim the second batch");

        assert_eq!(
            engine
                .active_world()
                .and_then(|world| world.boundary_for(nodes[next - 1])),
            Some(nodes[next]),
            "the second batch should show a door back into the first"
        );
        let crawls = engine.structural_build_counts().crawl;
        assert_eq!(
            engine
                .activate_batch_containing(nodes[next - 1], 3)
                .expect("should return to the first batch"),
            first
        );
        assert!(engine.active_contains(nodes[0]));
        assert_eq!(engine.structural_build_counts().crawl, crawls);
    }

    #[test]
    fn test_evicted_batch_rebuilds_from_its_members_without_crawling() {
        let (graph, nodes) = chain(WORLD_CACHE_CAPACITY + 2);
        let mut engine = LayoutEngine::new(graph);
        let first = engine
            .activate_batch_containing(nodes[0], 1)
            .expect("should claim the first batch");
        for &anchor in &nodes[1..=WORLD_CACHE_CAPACITY] {
            engine
                .activate_batch_containing(anchor, 1)
                .expect("should claim a small batch");
        }
        assert_eq!(engine.world_cache.len(), WORLD_CACHE_CAPACITY);
        assert!(!engine.world_cache.iter().any(|world| world.batch == first));
        let counts_before = engine.structural_build_counts();

        engine
            .activate_batch(first)
            .expect("should rebuild the evicted batch");

        assert_eq!(engine.active_batch(), Some(first));
        assert!(engine.active_contains(nodes[0]));
        let counts_after = engine.structural_build_counts();
        assert_eq!(counts_after.crawl, counts_before.crawl);
        assert_eq!(counts_after.assembly, counts_before.assembly + 1);
    }
}
