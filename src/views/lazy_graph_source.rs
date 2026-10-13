//! A [`GraphSource`] that grows a [`GenGraph`] from SQLite on demand, so a
//! [`LayoutEngine`](gen_tui::layout_engine::LayoutEngine) can crawl a block group's stored graph
//! without first materializing all of it. Lives here (rather than in gen-python or gen-r)
//! because it is the shared `GenGraph`-viewer glue every binding sits on top of - see
//! `gen_graph_widget`.

use std::{collections::HashSet, path::PathBuf};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    db::GraphConnection,
    edge::Edge,
    port_crawl::{PortCrawler, PortGraphSource},
};
use gen_tui::crawl::{
    EagerSource, GraphCursor, GraphSource, PreferredWormholeDoors, StructuralSubgraph, crawl_batch,
    subgraph_for,
};
use petgraph::Direction;

use crate::{get_connection_for_branch, views::junction_projection::project_junctions};

/// Prepare the complete projected window before the layout engine claims any of its nodes.
/// The source resolves raw junctions while `crawl_batch` asks for adjacency; the returned
/// edges and doors therefore refer only to stable projected identities.
/// Complete a source's projected boundary after a batch's member selection has stopped.
pub trait ProjectedBatchSource: GraphSource<GenGraph> {
    fn finish_projected_batch(
        &mut self,
        graph: &mut GenGraph,
        members: &[GraphNode],
    ) -> Result<(), String>;
}

impl ProjectedBatchSource for EagerSource {
    fn finish_projected_batch(
        &mut self,
        _graph: &mut GenGraph,
        _members: &[GraphNode],
    ) -> Result<(), String> {
        Ok(())
    }
}

pub fn prepare_projected_batch<S: ProjectedBatchSource>(
    anchor: GraphNode,
    node_budget: usize,
    graph: &mut GenGraph,
    source: &mut S,
    is_claimed: &dyn Fn(GraphNode) -> bool,
    preferred_doors: &PreferredWormholeDoors<GraphNode>,
) -> Result<StructuralSubgraph<GraphNode>, String> {
    let members = crawl_batch(
        anchor,
        node_budget,
        &mut GraphCursor::new(graph, source),
        is_claimed,
    )?;
    source.finish_projected_batch(graph, &members)?;
    Ok(subgraph_for(anchor, members, graph, preferred_doors))
}

/// Omit routes that the shared projection marked superseded, then discard nodes reached only
/// through those routes once their incoming side is complete.
fn prune_projected_graph(
    projected: &GenGraph,
    surviving_routes: &GenGraph,
    is_frontier: impl Fn(GraphNode, Direction) -> bool,
) -> GenGraph {
    let mut hidden = HashSet::new();
    loop {
        let before = hidden.len();
        for node in projected.nodes() {
            if node.node_id == PATH_START_NODE_ID || is_frontier(node, Direction::Incoming) {
                continue;
            }
            if projected
                .neighbors_directed(node, Direction::Incoming)
                .all(|predecessor| {
                    !surviving_routes.contains_edge(predecessor, node)
                        || hidden.contains(&predecessor)
                })
            {
                hidden.insert(node);
            }
        }
        if hidden.len() == before {
            break;
        }
    }
    let mut retained = GenGraph::new();
    for node in projected.nodes() {
        if !hidden.contains(&node) {
            retained.add_node(node);
        }
    }
    for (source, target, records) in projected.all_edges() {
        if !hidden.contains(&source)
            && !hidden.contains(&target)
            && surviving_routes.contains_edge(source, target)
        {
            retained.add_edge(source, target, records.clone());
        }
    }
    retained
}

/// The fixed `(node_id, 0, 0)` value a `PATH_START`/`PATH_END` sentinel always carries (see
/// `GenGraphController::new` and how `Edge::create` records a `PATH_START`/`PATH_END` endpoint's own
/// coordinate) - the same two values regardless of which graph, or how much of it is loaded
/// so far, so naming them never needs a graph scan.
fn end_sentinel() -> GraphNode {
    GraphNode {
        node_id: PATH_END_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    }
}

fn start_sentinel() -> GraphNode {
    GraphNode {
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    }
}

/// Whether `block_group_id` carries the GFA-import synthetic `PATH_END -> PATH_START`
/// circular marker. The marker always arrives at `PATH_START`'s one port, so a single port
/// lookup answers this for the whole block group, independent of which part of it a crawl has
/// loaded so far.
fn block_group_is_circular(conn: &GraphConnection, block_group_id: &HashId) -> bool {
    Edge::edges_at_port_direction(
        conn,
        block_group_id,
        PATH_START_NODE_ID,
        0,
        Direction::Incoming,
    )
    .is_ok_and(|arriving| {
        arriving
            .iter()
            .any(|augmented_edge| augmented_edge.edge.source_node_id == PATH_END_NODE_ID)
    })
}

/// Discard the `PATH_END -> PATH_START` circular marker together with the sentinels' own
/// attachment edges (`PATH_START -> first_node`, `last_node -> PATH_END`) from `graph`,
/// wherever they currently appear in it - not just the marker.
///
/// The marker alone is redundant with the real edge between the genome's actual last and
/// first nodes (the crawl loads that one too, like any other edge) - keeping both would make
/// the same cycle show up twice once `LayoutEngine`'s per-window cycle detection tries to
/// bypass it. But leaving the sentinels' attachment edges in place would still draw the
/// `╟`/`╢` glyphs, which visually claims "this graph can be read either linearly or
/// circularly" - not true for a graph that only carries a circular marker. So a circular block
/// group takes the sentinels out of the rendered graph entirely, not just the marker edge.
///
/// Callers must only invoke this once they already know (via [`block_group_is_circular`],
/// cached) that the block group is circular, and must call it after every load - a crawl can
/// reach either sentinel from anywhere along the genome, so this strips unconditionally rather
/// than looking for the marker in `graph` first.
///
/// This fires unconditionally on every circular block group, because there is currently no
/// signal to gate it on further: every GFA-imported edge, attachment and marker alike, carries
/// the same `NO_CHROMOSOME_INDEX` today, so nothing distinguishes a
/// genuinely-both-linear-and-circular graph (if such a thing existed) from an ordinary
/// circular one. If a future importer ever wants that distinction, it should mark it with its
/// own chromosome_index and this should check for it before discarding the attachment edges.
fn discard_circular_marker_and_sentinels(graph: &mut GenGraph) {
    let end = end_sentinel();
    let start = start_sentinel();
    graph.remove_edge(end, start);
    let start_successors: Vec<GraphNode> = graph
        .neighbors_directed(start, Direction::Outgoing)
        .collect();
    for successor in start_successors {
        graph.remove_edge(start, successor);
    }
    let end_predecessors: Vec<GraphNode> =
        graph.neighbors_directed(end, Direction::Incoming).collect();
    for predecessor in end_predecessors {
        graph.remove_edge(predecessor, end);
    }
}

/// Build the graph a viewer should seed its `LayoutEngine` with for `block_group_id`, and
/// implicitly its starting anchor (the graph's own lowest-index node - see
/// `LayoutEngine::default_anchor`).
///
/// An ordinary (non-circular) block group seeds with just the `PATH_START` sentinel, same as
/// [`GenGraphController::new`](crate::views::gen_graph_controller::GenGraphController::new) -
/// `set_preferred_initial_anchor` then opens the view there, and the crawl finds the rest from
/// its one real outgoing edge.
///
/// A circular block group cannot open on `PATH_START` the same way: once its attachment edges
/// are discarded (see `discard_circular_marker_and_sentinels`, which every
/// [`ProjectedGraphSource::expand_frontier`] call applies), `PATH_START` has no edges left to crawl
/// from at all. So this seeds with the genome's real first node instead, found by loading
/// `PATH_START`'s outgoing side up front rather than guessing at its identity (a `GraphNode`
/// is a slice whose bounds depend on the other edges on its node - see `AGENTS.md`'s
/// model-nuances note - so it must come from the crawl's own carving, never be hand-built from
/// a raw edge row).
pub fn seed_block_group_graph(conn: &GraphConnection, block_group_id: &HashId) -> GenGraph {
    let start = start_sentinel();
    if !block_group_is_circular(conn, block_group_id) {
        let mut seed = GenGraph::new();
        seed.add_node(start);
        return seed;
    }

    let mut probe = GenGraph::new();
    probe.add_node(start);
    let _ = PortCrawler::new(*block_group_id, false).expand(
        conn,
        &mut probe,
        &[start],
        Direction::Outgoing,
        0,
    );

    let mut seed = GenGraph::new();
    if let Some(real_first) = probe.neighbors_directed(start, Direction::Outgoing).next() {
        seed.add_node(real_first);
    } else {
        seed.add_node(start);
    }
    seed
}

/// Crawl just far enough from `block_group_id`'s start to tell whether it has more than one
/// data node, for [`starting_zoom_level`](crate::views::gen_graph_widget::starting_zoom_level).
/// A viewer's seed holds only the start, and the full graph may be too large to load.
///
/// Loads the start's successors, and when there is exactly one, completes both of its sides.
/// The probe then holds a second data node whenever the block group has one reachable from the
/// start, since it either follows the start directly or attaches to the only node that does.
/// `prune` matches the viewer's source (see [`ProjectedGraphSource::new_pruned`]), so edges it would
/// never load don't carve the node. At most three port lookups, whatever the graph's size.
pub fn probe_block_group_start(
    conn: &GraphConnection,
    block_group_id: &HashId,
    prune: bool,
) -> GenGraph {
    let start = start_sentinel();
    let mut probe = GenGraph::new();
    probe.add_node(start);
    let mut crawler = PortCrawler::new(*block_group_id, prune);
    if crawler
        .expand(conn, &mut probe, &[start], Direction::Outgoing, 0)
        .is_err()
    {
        return probe;
    }
    let successors: Vec<GraphNode> = probe
        .neighbors_directed(start, Direction::Outgoing)
        .collect();
    if let [only] = successors[..] {
        let _ = crawler.complete(conn, &mut probe, &[only]);
    }
    probe
}

/// Grows a block group's `GenGraph` from SQLite as a crawl pushes past its frontier, one port
/// at a time (see [`PortCrawler`]).
///
/// Keep the raw graph paired with the fallible port source. Only the projected graph is passed
/// to the layout engine; junctions and their pruning provenance stay here.
#[derive(Clone, Debug)]
pub struct ProjectedGraphSource {
    db_path: PathBuf,
    branch: Option<String>,
    block_group_id: HashId,
    prune: bool,
    raw_graph: GenGraph,
    port_source: Option<PortGraphSource>,
    is_circular: Option<bool>,
    surviving_routes: GenGraph,
}

impl ProjectedGraphSource {
    pub fn new(db_path: PathBuf, block_group_id: HashId) -> Self {
        Self {
            db_path,
            branch: None,
            block_group_id,
            prune: false,
            raw_graph: GenGraph::new(),
            port_source: None,
            is_circular: None,
            surviving_routes: GenGraph::new(),
        }
    }

    pub fn new_pruned(db_path: PathBuf, block_group_id: HashId) -> Self {
        Self {
            prune: true,
            ..Self::new(db_path, block_group_id)
        }
    }

    /// Open this source's connection on `branch` rather than the default branch.
    pub fn with_branch(self, branch: Option<String>) -> Self {
        Self { branch, ..self }
    }

    pub(crate) fn surviving_routes(&self) -> &GenGraph {
        &self.surviving_routes
    }

    fn ensure_source(&mut self, graph: &GenGraph) -> Result<(), String> {
        if self.port_source.is_none() {
            let connection = get_connection_for_branch(&self.db_path, self.branch.as_deref())
                .map_err(|error| error.to_string())?;
            self.is_circular = Some(block_group_is_circular(&connection, &self.block_group_id));
            self.raw_graph = graph.clone();
            self.port_source = Some(PortGraphSource::new(connection, self.block_group_id, false));
        }
        Ok(())
    }

    fn refresh_projection(&mut self, graph: &mut GenGraph) {
        let mut projection = project_junctions(&self.raw_graph);
        if self.is_circular == Some(true) {
            discard_circular_marker_and_sentinels(&mut projection.graph);
            discard_circular_marker_and_sentinels(&mut projection.surviving_routes);
        }
        self.surviving_routes = projection.surviving_routes;
        if self.prune {
            *graph = prune_projected_graph(
                &projection.graph,
                &self.surviving_routes,
                |node, direction| {
                    self.port_source
                        .as_ref()
                        .is_some_and(|source| source.is_frontier(node, direction))
                },
            );
        } else {
            *graph = projection.graph;
        }
    }

    fn resolve_junction_frontier(&mut self) -> Result<(), String> {
        let source = self
            .port_source
            .as_mut()
            .expect("should have initialized port source");
        loop {
            let junctions: Vec<_> = self
                .raw_graph
                .nodes()
                .filter(|node| {
                    !gen_core::is_terminal(node.node_id)
                        && node.sequence_start == node.sequence_end
                        && (source.is_frontier(*node, Direction::Incoming)
                            || source.is_frontier(*node, Direction::Outgoing))
                })
                .collect();
            if junctions.is_empty() {
                return Ok(());
            }
            source
                .expand_frontier(&mut self.raw_graph, &junctions)
                .map_err(|error| error.to_string())?;
        }
    }

    /// Add a backing-node position to the paired raw graph, then expose its projected slice.
    pub fn locate(
        &mut self,
        graph: &mut GenGraph,
        node_id: HashId,
        coordinate: i64,
    ) -> Option<GraphNode> {
        self.ensure_source(graph).ok()?;
        let located = self
            .port_source
            .as_mut()?
            .locate(&mut self.raw_graph, node_id, coordinate)
            .ok()??;
        if located.sequence_start == located.sequence_end && !gen_core::is_terminal(located.node_id)
        {
            self.resolve_junction_frontier().ok()?;
            let mut outgoing: Vec<_> = self
                .raw_graph
                .neighbors_directed(located, Direction::Outgoing)
                .filter(|node| node.sequence_start != node.sequence_end)
                .collect();
            let mut incoming: Vec<_> = self
                .raw_graph
                .neighbors_directed(located, Direction::Incoming)
                .filter(|node| node.sequence_start != node.sequence_end)
                .collect();
            outgoing.sort();
            incoming.sort();
            self.refresh_projection(graph);
            return outgoing
                .into_iter()
                .chain(incoming)
                .find(|node| graph.contains_node(*node));
        }
        self.refresh_projection(graph);
        Some(located)
    }
}

impl GraphSource<GenGraph> for ProjectedGraphSource {
    type Error = String;

    fn is_frontier(&self, node: GraphNode, direction: Direction) -> bool {
        self.port_source
            .as_ref()
            .is_none_or(|source| source.is_frontier(node, direction))
    }

    fn expand_frontier(
        &mut self,
        graph: &mut GenGraph,
        frontier: &[GraphNode],
    ) -> Result<(), Self::Error> {
        self.ensure_source(graph)?;
        let source = self
            .port_source
            .as_mut()
            .expect("should have initialized port source");
        source
            .expand_frontier(&mut self.raw_graph, frontier)
            .map_err(|error| error.to_string())?;
        self.resolve_junction_frontier()?;
        self.refresh_projection(graph);
        Ok(())
    }
}

impl ProjectedBatchSource for ProjectedGraphSource {
    fn finish_projected_batch(
        &mut self,
        graph: &mut GenGraph,
        members: &[GraphNode],
    ) -> Result<(), String> {
        self.ensure_source(graph)?;
        let source = self
            .port_source
            .as_mut()
            .expect("should have initialized port source");
        let unfinished = members
            .iter()
            .copied()
            .filter(|node| {
                source.is_frontier(*node, Direction::Incoming)
                    || source.is_frontier(*node, Direction::Outgoing)
            })
            .collect::<Vec<_>>();
        if !unfinished.is_empty() {
            source
                .expand_frontier(&mut self.raw_graph, &unfinished)
                .map_err(|error| error.to_string())?;
        }
        self.resolve_junction_frontier()?;
        self.refresh_projection(graph);
        if members.iter().any(|node| !graph.contains_node(*node)) {
            return Err(
                "a projected batch member disappeared while resolving its frontier".to_string(),
            );
        }
        Ok(())
    }
}

/// The one `GraphSource` a viewer's `LayoutEngine` is built with, whichever loading strategy
/// this particular graph needed - so the engine has a single, fixed concrete type across its
/// whole lifetime (including reassignment when the viewer switches to a different block group)
/// regardless of which strategy any one graph actually used. Historical views keep their eager
/// graph and its projected route-survival decisions here.
#[derive(Clone, Debug)]
pub enum EagerOrProjectedSource {
    Eager { surviving_routes: GenGraph },
    Projected(Box<ProjectedGraphSource>),
}

impl EagerOrProjectedSource {
    pub(crate) fn surviving_routes(&self) -> Option<&GenGraph> {
        match self {
            Self::Eager { surviving_routes } => Some(surviving_routes),
            Self::Projected(source) => Some(source.surviving_routes()),
        }
    }

    /// The loaded block of `node_id` holding `coordinate`, or, for a lazily crawled graph, the
    /// one the crawl would carve there, added as frontier.
    pub fn locate(
        &mut self,
        graph: &mut GenGraph,
        node_id: HashId,
        coordinate: i64,
    ) -> Option<GraphNode> {
        let loaded = graph.nodes().find(|node| {
            node.node_id == node_id
                && node.sequence_start <= coordinate
                && coordinate < node.sequence_end
        });
        match self {
            Self::Eager { .. } => loaded,
            Self::Projected(source) => loaded.or_else(|| source.locate(graph, node_id, coordinate)),
        }
    }
}

impl GraphSource<GenGraph> for EagerOrProjectedSource {
    type Error = String;

    fn is_frontier(&self, node: GraphNode, direction: Direction) -> bool {
        match self {
            Self::Eager { .. } => false,
            Self::Projected(source) => source.is_frontier(node, direction),
        }
    }

    fn expand_frontier(
        &mut self,
        graph: &mut GenGraph,
        frontier: &[GraphNode],
    ) -> Result<(), Self::Error> {
        match self {
            Self::Eager { .. } => Ok(()),
            Self::Projected(source) => source.expand_frontier(graph, frontier),
        }
    }
}

impl ProjectedBatchSource for EagerOrProjectedSource {
    fn finish_projected_batch(
        &mut self,
        graph: &mut GenGraph,
        members: &[GraphNode],
    ) -> Result<(), String> {
        match self {
            Self::Eager { .. } => Ok(()),
            Self::Projected(source) => source.finish_projected_batch(graph, members),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::get_connection,
        edge::Edge,
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use gen_tui::layout_engine::LayoutEngine;

    use super::*;
    use crate::views::gen_graph_widget::{
        FULL_ZOOM_LEVEL, MINIMAL_ZOOM_LEVEL, starting_zoom_level,
    };

    /// Build a tiny on-disk `start -> x -> y -> z -> end` block group and return its id, so
    /// `ProjectedGraphSource` can be exercised against a real SQLite file (`ProjectedGraphSource` opens its
    /// own connection lazily via `get_connection`, which is why this needs a real file rather
    /// than an in-memory database).
    fn setup_chain_block_group(db_path: &std::path::Path) -> HashId {
        setup_labelled_chain_block_group(db_path, &["x", "y", "z"]).0
    }

    fn setup_junction_block_group(db_path: &std::path::Path) -> HashId {
        let (block_group_id, _) = setup_labelled_chain_block_group(db_path, &["x", "y"]);
        let connection = get_connection(db_path).unwrap();
        let node_id = HashId::convert_str("x");
        let edges = [(2, 4, 1), (2, 2, 0), (4, 4, 0)]
            .into_iter()
            .map(|(source_coordinate, target_coordinate, chromosome_index)| {
                let edge = Edge::create(
                    &connection,
                    node_id,
                    source_coordinate,
                    Strand::Forward,
                    node_id,
                    target_coordinate,
                    Strand::Forward,
                )
                .unwrap();
                BlockGroupEdgeData {
                    block_group_id,
                    edge_id: edge.id,
                    chromosome_index,
                    phased: 0,
                }
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(&connection, &edges);
        block_group_id
    }

    #[test]
    fn test_prune_projected_graph_omits_only_decided_superseded_routes() {
        let start = start_sentinel();
        let current = graph_node("current");
        let retired = graph_node("retired");
        let mut projected = GenGraph::new();
        projected.add_edge(start, current, Vec::new());
        projected.add_edge(start, retired, Vec::new());
        let mut surviving = GenGraph::new();
        surviving.add_edge(start, current, Vec::new());
        let pruned = prune_projected_graph(&projected, &surviving, |_, _| false);
        assert!(pruned.contains_edge(start, current));
        assert!(!pruned.contains_node(retired));
        let partial = prune_projected_graph(&projected, &surviving, |node, direction| {
            node == retired && direction == Direction::Incoming
        });
        assert!(partial.contains_node(retired));
        assert!(!partial.contains_edge(start, retired));
    }

    /// Build an on-disk `start -> labels[0] -> ... -> end` chain of 5-base nodes, each node id
    /// derived from its label. Returns the block group id and the chain's edge ids in order,
    /// ready to store as a path.
    pub(crate) fn setup_labelled_chain_block_group(
        db_path: &std::path::Path,
        labels: &[&str],
    ) -> (HashId, Vec<HashId>) {
        let conn = get_connection(db_path).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        Sample::get_or_create(
            &conn,
            NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let block_group = BlockGroup::create(
            &conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "chr1",
                ..Default::default()
            },
        )
        .unwrap();

        let mut node_ids = Vec::new();
        for label in labels {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAA")
                .save(&conn)
                .unwrap();
            node_ids
                .push(Node::create(&conn, &sequence.hash, &HashId::convert_str(label)).unwrap());
        }

        let mut edges = vec![
            Edge::create(
                &conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                node_ids[0],
                0,
                Strand::Forward,
            )
            .unwrap(),
        ];
        for window in node_ids.windows(2) {
            edges.push(
                Edge::create(
                    &conn,
                    window[0],
                    5,
                    Strand::Forward,
                    window[1],
                    0,
                    Strand::Forward,
                )
                .unwrap(),
            );
        }
        edges.push(
            Edge::create(
                &conn,
                *node_ids.last().unwrap(),
                5,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap(),
        );

        BlockGroupEdge::bulk_create(
            &conn,
            &edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );

        (block_group.id, edges.iter().map(|edge| edge.id).collect())
    }

    fn graph_node(label: &str) -> GraphNode {
        GraphNode {
            node_id: HashId::convert_str(label),
            sequence_start: 0,
            sequence_end: 5,
        }
    }

    /// Expand every node that is still frontier on either side until none is left.
    fn crawl_everything(source: &mut ProjectedGraphSource, graph: &mut GenGraph) {
        loop {
            let frontier: Vec<GraphNode> = graph
                .nodes()
                .filter(|node| {
                    source.is_frontier(*node, Direction::Outgoing)
                        || source.is_frontier(*node, Direction::Incoming)
                })
                .collect();
            if frontier.is_empty() {
                return;
            }
            source.expand_frontier(graph, &frontier).unwrap();
        }
    }

    #[test]
    fn test_adjacent_junction_is_resolved_before_projected_batch() {
        let temporary = tempfile::tempdir().unwrap();
        let db_path = temporary.path().join("graph.db");
        let block_group_id = setup_junction_block_group(&db_path);
        let connection = get_connection(&db_path).unwrap();
        let mut graph = seed_block_group_graph(&connection, &block_group_id);
        let start = graph.nodes().next().unwrap();
        let mut source = ProjectedGraphSource::new(db_path, block_group_id);
        source.expand_frontier(&mut graph, &[start]).unwrap();
        let adjacent = graph
            .nodes()
            .find(|node| node.node_id == HashId::convert_str("x"))
            .unwrap();
        let batch = prepare_projected_batch(
            adjacent,
            1,
            &mut graph,
            &mut source,
            &|_| false,
            &PreferredWormholeDoors::new(),
        )
        .unwrap();
        assert!(
            source
                .raw_graph
                .nodes()
                .any(|node| node.sequence_start == node.sequence_end
                    && !gen_core::is_terminal(node.node_id))
        );
        assert!(
            graph
                .nodes()
                .all(|node| node.sequence_start != node.sequence_end
                    || gen_core::is_terminal(node.node_id))
        );
        assert!(batch.nodes.iter().all(|node| graph.contains_node(*node)));
        let port_source = source.port_source.as_ref().unwrap();
        assert!(batch.nodes.iter().all(|node| {
            !port_source.is_frontier(*node, Direction::Incoming)
                && !port_source.is_frontier(*node, Direction::Outgoing)
        }));
        assert!(
            source
                .raw_graph
                .nodes()
                .filter(|node| {
                    node.sequence_start == node.sequence_end && !gen_core::is_terminal(node.node_id)
                })
                .all(|junction| {
                    !port_source.is_frontier(junction, Direction::Incoming)
                        && !port_source.is_frontier(junction, Direction::Outgoing)
                })
        );
        assert!(
            batch
                .external_edges
                .iter()
                .all(|door| graph.contains_node(door.boundary) && graph.contains_node(door.target))
        );
    }

    #[test]
    fn test_frontier_on_junction_is_completed_before_projection() {
        let temporary = tempfile::tempdir().unwrap();
        let db_path = temporary.path().join("graph.db");
        let block_group_id = setup_junction_block_group(&db_path);
        let connection = get_connection(&db_path).unwrap();
        let mut graph = seed_block_group_graph(&connection, &block_group_id);
        let start = graph.nodes().next().unwrap();
        let mut source = ProjectedGraphSource::new(db_path, block_group_id);
        source.ensure_source(&graph).unwrap();
        let port_source = source.port_source.as_mut().unwrap();
        port_source
            .expand_frontier(&mut source.raw_graph, &[start])
            .unwrap();
        let first = source
            .raw_graph
            .nodes()
            .find(|node| node.node_id == HashId::convert_str("x"))
            .unwrap();
        port_source
            .expand_frontier(&mut source.raw_graph, &[first])
            .unwrap();
        let junctions = source
            .raw_graph
            .nodes()
            .filter(|node| {
                node.sequence_start == node.sequence_end && !gen_core::is_terminal(node.node_id)
            })
            .collect::<Vec<_>>();
        assert!(!junctions.is_empty());
        source.resolve_junction_frontier().unwrap();
        for junction in junctions {
            assert!(
                !source
                    .port_source
                    .as_ref()
                    .unwrap()
                    .is_frontier(junction, Direction::Incoming)
            );
            assert!(
                !source
                    .port_source
                    .as_ref()
                    .unwrap()
                    .is_frontier(junction, Direction::Outgoing)
            );
        }
        source.refresh_projection(&mut graph);
        assert!(
            graph
                .nodes()
                .all(|node| node.sequence_start != node.sequence_end
                    || gen_core::is_terminal(node.node_id))
        );
    }

    #[test]
    fn test_expand_frontier_pulls_in_a_node_neighbourhood_from_sql() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let block_group_id = setup_chain_block_group(&db_path);

        let mut source = ProjectedGraphSource::new(db_path, block_group_id);
        let mut graph = GenGraph::new();
        let anchor = graph_node("y");
        graph.add_node(anchor);
        assert!(source.is_frontier(anchor, Direction::Outgoing));
        assert!(source.is_frontier(anchor, Direction::Incoming));

        source.expand_frontier(&mut graph, &[anchor]).unwrap();

        assert!(graph.contains_edge(graph_node("x"), anchor));
        assert!(graph.contains_edge(anchor, graph_node("z")));
        assert!(!source.is_frontier(anchor, Direction::Outgoing));
        assert!(!source.is_frontier(anchor, Direction::Incoming));
        assert!(
            source.is_frontier(graph_node("z"), Direction::Outgoing),
            "completing y should stop at its own neighbours"
        );
    }

    /// Build a tiny on-disk circular `start -> x -> y -> z -> end`, `z -> x` (real closure),
    /// `end -> start` (GFA-import synthetic marker) block group - the shape
    /// `discard_circular_marker_edge` is meant to see: two distinct closing edges, only one of
    /// which is the marker that should be dropped. Also returns the linear chain's edge ids in
    /// order, ready to store as a path.
    pub(crate) fn setup_circular_block_group(db_path: &std::path::Path) -> (HashId, Vec<HashId>) {
        setup_circular_labelled_chain_block_group(db_path, &["x", "y", "z"])
    }

    /// [`setup_labelled_chain_block_group`] closed into a circle: a real edge from the last
    /// node back to the first, plus the `PATH_END -> PATH_START` circular marker.
    fn setup_circular_labelled_chain_block_group(
        db_path: &std::path::Path,
        labels: &[&str],
    ) -> (HashId, Vec<HashId>) {
        let (block_group_id, chain_edge_ids) = setup_labelled_chain_block_group(db_path, labels);
        let conn = get_connection(db_path).unwrap();
        let real_closure = Edge::create(
            &conn,
            HashId::convert_str(labels.last().expect("should have at least one label")),
            5,
            Strand::Forward,
            HashId::convert_str(labels[0]),
            0,
            Strand::Forward,
        )
        .unwrap();
        let marker = Edge::create(
            &conn,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();
        BlockGroupEdge::bulk_create(
            &conn,
            &[real_closure, marker]
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        (block_group_id, chain_edge_ids)
    }

    #[test]
    fn test_expand_frontier_discards_marker_and_sentinel_attachment_edges() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_circular_block_group(&db_path);

        let mut source = ProjectedGraphSource::new(db_path, block_group_id);
        let mut graph = GenGraph::new();
        graph.add_node(graph_node("y"));
        crawl_everything(&mut source, &mut graph);

        let end = GraphNode {
            node_id: PATH_END_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let x = GraphNode {
            node_id: HashId::convert_str("x"),
            sequence_start: 0,
            sequence_end: 5,
        };
        let z = GraphNode {
            node_id: HashId::convert_str("z"),
            sequence_start: 0,
            sequence_end: 5,
        };
        assert!(
            !graph.contains_edge(end, start),
            "the synthetic PATH_END -> PATH_START marker should be discarded"
        );
        assert!(
            graph.contains_edge(z, x),
            "the real z -> x closure edge should be kept untouched, like any other edge"
        );
        assert!(
            !graph.contains_edge(start, x) && !graph.contains_edge(z, end),
            "the start/end sentinels' own attachment edges should be discarded too, so a \
             circular block group never draws the start/end glyphs at all"
        );
    }

    #[test]
    fn test_seed_block_group_graph_anchors_on_the_real_first_node_for_a_circular_block_group() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_circular_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();

        let seed = seed_block_group_graph(&conn, &block_group_id);

        let x = GraphNode {
            node_id: HashId::convert_str("x"),
            sequence_start: 0,
            sequence_end: 5,
        };
        assert_eq!(
            seed.nodes().collect::<Vec<_>>(),
            vec![x],
            "a circular block group should seed on its real first node, not PATH_START, since \
             PATH_START has no edges left once the crawl discards its attachment edge"
        );
    }

    #[test]
    fn test_seed_block_group_graph_anchors_on_path_start_for_a_linear_block_group() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let block_group_id = setup_chain_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();

        let seed = seed_block_group_graph(&conn, &block_group_id);

        assert_eq!(
            seed.nodes().collect::<Vec<_>>(),
            vec![start_sentinel()],
            "an ordinary (non-circular) block group should seed on PATH_START as before"
        );
    }

    #[test]
    fn test_layout_engine_crawls_a_lazily_loaded_graph() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let block_group_id = setup_chain_block_group(&db_path);
        let source = ProjectedGraphSource::new(db_path, block_group_id);

        // Seed the graph with only the anchor - everything else must come from `source`.
        let mut graph = GenGraph::new();
        let anchor = GraphNode {
            node_id: HashId::convert_str("y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        graph.add_node(anchor);

        let mut engine = LayoutEngine::new_with_source(graph, source);
        let window = engine
            .window_for(anchor, 10)
            .expect("should crawl the whole chain via SQL");
        let data_nodes = window
            .graph
            .node_weights()
            .filter(|node| matches!(node.role, gen_tui::layout::NodeRole::Data(_)))
            .count();
        assert_eq!(
            data_nodes, 5,
            "start, x, y, z, end should all be crawled in"
        );
    }

    /// The zoom level `starting_zoom_level` picks from the probe of a block group's start.
    fn probed_zoom_level(db_path: &std::path::Path, block_group_id: &HashId) -> usize {
        let conn = get_connection(db_path).unwrap();
        starting_zoom_level(&probe_block_group_start(&conn, block_group_id, false))
    }

    #[test]
    fn test_probe_block_group_start_finds_a_single_data_node() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x"]);
        assert_eq!(
            probed_zoom_level(&db_path, &block_group_id),
            FULL_ZOOM_LEVEL
        );
    }

    #[test]
    fn test_probe_block_group_start_finds_a_single_circular_data_node() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_circular_labelled_chain_block_group(&db_path, &["x"]);
        assert_eq!(
            probed_zoom_level(&db_path, &block_group_id),
            FULL_ZOOM_LEVEL
        );
    }

    #[test]
    fn test_probe_block_group_start_finds_a_second_data_node() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x", "y"]);
        assert_eq!(
            probed_zoom_level(&db_path, &block_group_id),
            MINIMAL_ZOOM_LEVEL
        );

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_circular_block_group(&db_path);
        assert_eq!(
            probed_zoom_level(&db_path, &block_group_id),
            MINIMAL_ZOOM_LEVEL
        );
    }

    #[test]
    fn test_probe_block_group_start_stays_near_the_start_of_a_long_chain() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels: Vec<String> = (0..80).map(|index| format!("n{index}")).collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();

        let probe = probe_block_group_start(&conn, &block_group_id, false);

        assert_eq!(probe.node_count(), 3, "should hold start, n0 and n1 only");
        assert_eq!(starting_zoom_level(&probe), MINIMAL_ZOOM_LEVEL);
    }
}
