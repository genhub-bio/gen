//! A [`GraphSource`] that grows a [`GenGraph`] from SQLite on demand, so a
//! [`LayoutEngine`](gen_tui::layout_engine::LayoutEngine) can crawl a block group's stored graph
//! without first materializing all of it. Lives here (rather than in gen-python or gen-r)
//! because it is the shared `GenGraph`-viewer glue every binding sits on top of - see
//! `gen_graph_widget`.

use std::{path::PathBuf, sync::Mutex};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, is_terminal};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    db::{GraphConnection, get_connection},
    edge::Edge,
    port_crawl::PortCrawler,
};
use gen_tui::crawl::{EagerSource, GraphSource};
use petgraph::Direction;

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
    Edge::edges_at_port(conn, block_group_id, PATH_START_NODE_ID, 0).is_ok_and(|port_edges| {
        port_edges
            .arriving
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
/// [`SqlGraphSource::expand_frontier`] call applies), `PATH_START` has no edges left to crawl
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

/// The real content slices a block group opens and closes on: every non-sentinel target of a
/// `PATH_START` edge and every non-sentinel source of a `PATH_END` edge.
///
/// Annotation loading uses these to hide an annotation that covers the whole block group end to
/// end. A lazily crawled graph cannot answer that by looking for nodes without predecessors or
/// successors, since every frontier node looks like one, so the bounds are resolved once per
/// block group, independent of how much of the graph is loaded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockGroupBounds {
    /// The first content slices, reached from `PATH_START`.
    pub roots: Vec<GraphNode>,
    /// The last content slices, leading into `PATH_END`.
    pub leaves: Vec<GraphNode>,
}

impl BlockGroupBounds {
    /// Resolve the live block group's bounds by probing only the two sentinel ports, the way
    /// [`seed_block_group_graph`] probes `PATH_START`. The probe carves the boundary slices
    /// through the crawl itself, so they match the slices a viewer's crawl later loads. A
    /// circular block group keeps its sentinel attachment edges in the database next to the
    /// `PATH_END -> PATH_START` marker, so its bounds come out the same way; the marker itself
    /// joins two sentinels and never counts as a bound.
    pub fn load(conn: &GraphConnection, block_group_id: &HashId) -> Self {
        let start = start_sentinel();
        let end = end_sentinel();
        let mut probe = GenGraph::new();
        probe.add_node(start);
        probe.add_node(end);
        let mut crawler = PortCrawler::new(*block_group_id, false);
        let _ = crawler.expand(conn, &mut probe, &[start], Direction::Outgoing, 0);
        let _ = crawler.expand(conn, &mut probe, &[end], Direction::Incoming, 0);
        Self::from_graph(&probe)
    }

    /// Read the bounds off a graph that already carries its sentinels' attachment edges, such
    /// as a historical view's fully materialized `BlockGroup::get_graph`.
    pub fn from_graph(graph: &GenGraph) -> Self {
        let content_neighbors = |sentinel: GraphNode, direction: Direction| -> Vec<GraphNode> {
            graph
                .neighbors_directed(sentinel, direction)
                .filter(|node| !is_terminal(node.node_id))
                .collect()
        };
        Self {
            roots: content_neighbors(start_sentinel(), Direction::Outgoing),
            leaves: content_neighbors(end_sentinel(), Direction::Incoming),
        }
    }

    /// The bounds for a viewer's block group: read off `graph` for a historical view, whose
    /// graph is always fully loaded and whose history the port queries cannot answer for, and
    /// probed from the live database otherwise.
    pub fn for_view(
        conn: &GraphConnection,
        graph: &GenGraph,
        block_group_id: &HashId,
        history_ref: Option<&str>,
    ) -> Self {
        if history_ref.is_some() {
            Self::from_graph(graph)
        } else {
            Self::load(conn, block_group_id)
        }
    }
}

/// Grows a block group's `GenGraph` from SQLite as a crawl pushes past its frontier, one port
/// at a time (see [`PortCrawler`]).
///
/// Caches one open `GraphConnection` per `SqlGraphSource` instance, opened lazily on the first
/// `expand_frontier` call and reused for the lifetime of a single crawl/view session - opening
/// a fresh connection per call re-runs full migrations (including a `dolt_status` query) on
/// every call, which made first-render of even a modest window painfully slow. This is
/// safe because nothing in gen-tui's synchronous, single-threaded render loop (or the
/// `render_gfa_snapshot` test helper in `gen_graph_widget.rs`, the only other caller) requires
/// `Send` or `Clone` on this type - gen-python's own per-operation connection pattern
/// (`gen-python/src/python_api/jupyter_widget.rs`) is unrelated and unaffected by this change.
#[derive(Debug)]
pub struct SqlGraphSource {
    db_path: PathBuf,
    block_group_id: HashId,
    /// Which ports of this block group are already in the graph. It describes the one
    /// `GenGraph` this source grows, which is why a `LayoutEngine` owns and clones the two
    /// together.
    crawler: PortCrawler,
    /// Whether this block group is circular, determined once (see
    /// [`block_group_is_circular`]) and cached - `None` until the first `expand_frontier` call.
    is_circular: Option<bool>,
    /// The connection opened on the first `expand_frontier` call, reused thereafter. Wrapped in
    /// a `Mutex` purely to make `SqlGraphSource` itself `Sync` - `rusqlite::Connection` holds a
    /// `RefCell`-backed statement cache and so is `Send` but not `Sync` (see gen-python's
    /// `jupyter_widget.rs`, whose `#[pyclass]` types must be both to survive ipykernel moving
    /// them between its cell-execution thread pool and its asyncio ioloop thread). Every access
    /// here goes through `&mut self` already, so this never actually contends - `get_mut`
    /// reaches the connection directly, bypassing the lock.
    connection: Mutex<Option<GraphConnection>>,
}

/// A `GraphConnection` can't itself be cloned (it wraps a live `rusqlite::Connection`), so a
/// clone starts with no connection and opens its own lazily on its own first `expand_frontier`
/// call - safe because nothing about the connection is observable state (see this struct's
/// own doc on why it's cached at all).
impl Clone for SqlGraphSource {
    fn clone(&self) -> Self {
        Self {
            db_path: self.db_path.clone(),
            block_group_id: self.block_group_id,
            crawler: self.crawler.clone(),
            is_circular: self.is_circular,
            connection: Mutex::new(None),
        }
    }
}

impl SqlGraphSource {
    pub fn new(db_path: PathBuf, block_group_id: HashId) -> Self {
        Self {
            db_path,
            block_group_id,
            crawler: PortCrawler::new(block_group_id, false),
            is_circular: None,
            connection: Mutex::new(None),
        }
    }

    /// Like [`Self::new`], but never loads an edge `BlockGroup::prune_graph` would remove, so
    /// pruned/retired edit-site edges (and anything only reachable through one) never enter the
    /// crawled graph. This is what backs gen-python's `show_history=False`.
    pub fn new_pruned(db_path: PathBuf, block_group_id: HashId) -> Self {
        Self {
            crawler: PortCrawler::new(block_group_id, true),
            ..Self::new(db_path, block_group_id)
        }
    }
}

impl GraphSource<GenGraph> for SqlGraphSource {
    fn is_frontier(&self, node: GraphNode, direction: Direction) -> bool {
        !self.crawler.is_complete(&node, direction)
    }

    /// A best-effort no-op on a connection or query failure - the nodes simply stay frontier,
    /// since gen-tui's `GraphSource` contract has no error channel of its own.
    fn expand_frontier(&mut self, graph: &mut GenGraph, frontier: &[GraphNode]) {
        let connection_slot = self
            .connection
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if connection_slot.is_none() {
            *connection_slot = get_connection(&self.db_path).ok();
        }
        let Some(conn) = connection_slot.as_ref() else {
            return;
        };
        let _ = self.crawler.complete(conn, graph, frontier);
        let is_circular = *self
            .is_circular
            .get_or_insert_with(|| block_group_is_circular(conn, &self.block_group_id));
        if is_circular {
            discard_circular_marker_and_sentinels(graph);
        }
    }
}

/// The one `GraphSource` a viewer's `LayoutEngine` is built with, whichever loading strategy
/// this particular graph needed - so the engine has a single, fixed concrete type across its
/// whole lifetime (including reassignment when the viewer switches to a different block group)
/// regardless of which strategy any one graph actually used. [`SqlGraphSource`] only reads the
/// live graph, so a historical view keeps using [`EagerSource`] over a fully materialized
/// `BlockGroup::get_graph` instead.
#[derive(Debug)]
pub enum EagerOrSqlSource {
    Eager(EagerSource),
    Sql(Box<SqlGraphSource>),
}

impl GraphSource<GenGraph> for EagerOrSqlSource {
    fn is_frontier(&self, node: GraphNode, direction: Direction) -> bool {
        match self {
            Self::Eager(source) => {
                <EagerSource as GraphSource<GenGraph>>::is_frontier(source, node, direction)
            }
            Self::Sql(source) => source.is_frontier(node, direction),
        }
    }

    fn expand_frontier(&mut self, graph: &mut GenGraph, frontier: &[GraphNode]) {
        match self {
            Self::Eager(source) => source.expand_frontier(graph, frontier),
            Self::Sql(source) => source.expand_frontier(graph, frontier),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, Workspace};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        edge::Edge,
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use gen_tui::layout_engine::LayoutEngine;

    use super::*;

    /// Build a tiny on-disk `start -> x -> y -> z -> end` block group and return its id, so
    /// `SqlGraphSource` can be exercised against a real SQLite file (`SqlGraphSource` opens its
    /// own connection lazily via `get_connection`, which is why this needs a real file rather
    /// than an in-memory database).
    fn setup_chain_block_group(db_path: &std::path::Path) -> HashId {
        setup_labelled_chain_block_group(db_path, &["x", "y", "z"]).0
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
    fn crawl_everything(source: &mut SqlGraphSource, graph: &mut GenGraph) {
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
            source.expand_frontier(graph, &frontier);
        }
    }

    #[test]
    fn test_expand_frontier_pulls_in_a_node_neighbourhood_from_sql() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let block_group_id = setup_chain_block_group(&db_path);

        let mut source = SqlGraphSource::new(db_path, block_group_id);
        let mut graph = GenGraph::new();
        let anchor = graph_node("y");
        graph.add_node(anchor);
        assert!(source.is_frontier(anchor, Direction::Outgoing));
        assert!(source.is_frontier(anchor, Direction::Incoming));

        source.expand_frontier(&mut graph, &[anchor]);

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
        let (block_group_id, chain_edge_ids) =
            setup_labelled_chain_block_group(db_path, &["x", "y", "z"]);
        let conn = get_connection(db_path).unwrap();
        let real_closure = Edge::create(
            &conn,
            HashId::convert_str("z"),
            5,
            Strand::Forward,
            HashId::convert_str("x"),
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

        let mut source = SqlGraphSource::new(db_path, block_group_id);
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
    fn test_block_group_bounds_load_the_first_and_last_content_slices() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let block_group_id = setup_chain_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();

        let bounds = BlockGroupBounds::load(&conn, &block_group_id);

        assert_eq!(
            bounds,
            BlockGroupBounds {
                roots: vec![graph_node("x")],
                leaves: vec![graph_node("z")],
            }
        );
        let full_graph =
            BlockGroup::get_graph(&conn, &Workspace::from_current_dir(), &block_group_id, None)
                .unwrap();
        assert_eq!(BlockGroupBounds::from_graph(&full_graph), bounds);
    }

    /// The `PATH_END -> PATH_START` marker joins the two sentinels, so a circular block group's
    /// bounds are still its real first and last slices, whether probed or read off the eager
    /// graph that keeps the marker.
    #[test]
    fn test_block_group_bounds_skip_the_circular_marker() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_circular_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();

        let bounds = BlockGroupBounds::load(&conn, &block_group_id);

        assert_eq!(
            bounds,
            BlockGroupBounds {
                roots: vec![graph_node("x")],
                leaves: vec![graph_node("z")],
            }
        );
        let full_graph =
            BlockGroup::get_graph(&conn, &Workspace::from_current_dir(), &block_group_id, None)
                .unwrap();
        assert!(full_graph.contains_edge(end_sentinel(), start_sentinel()));
        assert_eq!(BlockGroupBounds::from_graph(&full_graph), bounds);
        assert_eq!(
            BlockGroupBounds::for_view(&conn, &full_graph, &block_group_id, Some("main")),
            bounds,
            "a historical view reads the bounds off its full graph"
        );
    }

    #[test]
    fn test_block_group_bounds_are_empty_without_sentinel_edges() {
        let mut graph = GenGraph::new();
        graph.add_edge(graph_node("x"), graph_node("y"), Vec::new());

        assert_eq!(
            BlockGroupBounds::from_graph(&graph),
            BlockGroupBounds::default()
        );
    }

    #[test]
    fn test_layout_engine_crawls_a_lazily_loaded_graph() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let block_group_id = setup_chain_block_group(&db_path);
        let source = SqlGraphSource::new(db_path, block_group_id);

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
}
