//! A [`GraphSource`] that grows a [`GenGraph`] from SQLite on demand, so a
//! [`LayoutEngine`](gen_tui::layout_engine::LayoutEngine) can crawl a block group's stored graph
//! without first materializing all of it. Lives here (rather than in gen-python or gen-r)
//! because it is the shared `GenGraph`-viewer glue every binding sits on top of - see
//! `gen_graph_widget`.

use std::path::PathBuf;

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Workspace};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    db::{GraphConnection, get_connection},
    edge::Edge,
    graph::expand,
};
use gen_tui::crawl::{EagerSource, GraphSource};
use petgraph::Direction;

/// The fixed `(node_id, 0, 0)` value a `PATH_START`/`PATH_END` sentinel always carries (see
/// `get_empty_graph` and how `Edge::create` records a `PATH_START`/`PATH_END` endpoint's own
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
/// circular marker, via one direct, bounded query rooted at `PATH_START` - `PATH_START` is
/// always the marker edge's own target, so a query for edges touching `PATH_START` alone is
/// guaranteed to see it in a single 1-hop fetch. This is deliberately *not* inferred from
/// whether the marker happens to appear in whatever bounded batch an arbitrary node's own
/// `expand` call fetched (see `discard_circular_marker_and_sentinels`'s doc for why that's
/// unreliable): e.g. crawling from a node several hops from both sentinels can re-surface just
/// one sentinel's attachment edge, without the marker itself being in that same fetch.
fn block_group_is_circular(conn: &GraphConnection, block_group_id: &HashId) -> bool {
    Edge::edges_for_block_group_nodes(conn, block_group_id, &[PATH_START_NODE_ID], None)
        .unwrap_or_default()
        .iter()
        .any(|augmented_edge| {
            augmented_edge.edge.source_node_id == PATH_END_NODE_ID
                && augmented_edge.edge.target_node_id == PATH_START_NODE_ID
        })
}

/// Discard the `PATH_END -> PATH_START` circular marker together with the sentinels' own
/// attachment edges (`PATH_START -> first_node`, `last_node -> PATH_END`) from `graph`,
/// wherever they currently appear in it - not just the marker.
///
/// The marker alone is redundant with the real edge between the genome's actual last and
/// first nodes (`expand` loads that one too, like any other edge) - keeping both would make
/// the same cycle show up twice once `LayoutEngine`'s per-window cycle detection tries to
/// bypass it. But leaving the sentinels' attachment edges in place would still draw the
/// `╟`/`╢` glyphs, which visually claims "this graph can be read either linearly or
/// circularly" - not true for a graph that only carries a circular marker. So a circular block
/// group takes the sentinels out of the rendered graph entirely, not just the marker edge.
///
/// Callers must only invoke this once they already know (via [`block_group_is_circular`],
/// cached) that the block group is circular, and must call it after every load - this strips
/// unconditionally rather than checking for the marker's presence in `graph` first, since a
/// crawl reaching the sentinels from a node many hops away can reconnect one sentinel's
/// attachment edge without the marker itself being in that same load.
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
/// [`get_empty_graph`](crate::views::block_group::get_empty_graph) - `set_preferred_initial_anchor`
/// then opens the view there, and the crawl finds the rest from its one real outgoing edge.
///
/// A circular block group cannot open on `PATH_START` the same way: once its attachment edges
/// are discarded (see `discard_circular_marker_and_sentinels`, which every
/// [`SqlGraphSource::ensure_loaded`] call applies), `PATH_START` has no edges left to crawl
/// from at all. So this seeds with the genome's real first node instead - found with one
/// bounded `expand` call on `PATH_START` here, up front, rather than guessing at its identity
/// (a `GraphNode` is a content-addressed slice - see `AGENTS.md`'s model-nuances note - so it
/// must come from `expand`'s own construction, never be hand-built from a raw edge row).
pub fn seed_block_group_graph(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: &HashId,
) -> GenGraph {
    let start = start_sentinel();
    if !block_group_is_circular(conn, block_group_id) {
        let mut seed = GenGraph::new();
        seed.add_node(start);
        return seed;
    }

    let mut probe = GenGraph::new();
    probe.add_node(start);
    expand(conn, workspace, &mut probe, block_group_id, start.node_id);

    let mut seed = GenGraph::new();
    if let Some(real_first) = probe.neighbors_directed(start, Direction::Outgoing).next() {
        seed.add_node(real_first);
    } else {
        seed.add_node(start);
    }
    seed
}

/// Grows a block group's `GenGraph` one node's neighbourhood at a time, backed by SQLite.
///
/// Caches one open `GraphConnection` per `SqlGraphSource` instance, opened lazily on the first
/// `ensure_loaded` call and reused for the lifetime of a single crawl/view session - opening a
/// fresh connection per call re-runs full migrations (including a `dolt_status` query) on every
/// single crawled node, which made first-render of even a modest window painfully slow. This is
/// safe because nothing in gen-tui's synchronous, single-threaded render loop (or the
/// `render_gfa_snapshot` test helper in `gen_graph_widget.rs`, the only other caller) requires
/// `Send` or `Clone` on this type - gen-python's own per-operation connection pattern
/// (`gen-python/src/python_api/jupyter_widget.rs`) is unrelated and unaffected by this change.
#[derive(Debug)]
pub struct SqlGraphSource {
    db_path: PathBuf,
    workspace: Workspace,
    block_group_id: HashId,
    /// Whether this block group is circular, determined once (see
    /// [`block_group_is_circular`]) and cached - `None` until the first `ensure_loaded` call.
    is_circular: Option<bool>,
    /// The connection opened on the first `ensure_loaded` call, reused thereafter.
    connection: Option<GraphConnection>,
}

impl SqlGraphSource {
    pub fn new(db_path: PathBuf, workspace: Workspace, block_group_id: HashId) -> Self {
        Self {
            db_path,
            workspace,
            block_group_id,
            is_circular: None,
            connection: None,
        }
    }
}

impl GraphSource<GenGraph> for SqlGraphSource {
    /// Ensure `node`'s full neighbourhood is present in `graph`. A best-effort no-op on a
    /// connection failure - the crawl will simply see `node` as boundary-less rather than fail
    /// outright, since gen-tui's `GraphSource` contract has no error channel of its own.
    fn ensure_loaded(&mut self, graph: &mut GenGraph, node: GraphNode) -> bool {
        if self.connection.is_none() {
            self.connection = get_connection(&self.db_path).ok();
        }
        let Some(conn) = self.connection.as_ref() else {
            return false;
        };
        let added = expand(
            conn,
            &self.workspace,
            graph,
            &self.block_group_id,
            node.node_id,
        );
        let is_circular = *self
            .is_circular
            .get_or_insert_with(|| block_group_is_circular(conn, &self.block_group_id));
        if is_circular {
            discard_circular_marker_and_sentinels(graph);
        }
        added
    }
}

/// The one `GraphSource` a viewer's `LayoutEngine` is built with, whichever loading strategy
/// this particular graph needed - so the engine has a single, fixed concrete type across its
/// whole lifetime (including reassignment when the viewer switches to a different block group)
/// regardless of which strategy any one graph actually used. [`SqlGraphSource::ensure_loaded`]
/// has no `history_ref` of its own (`gen_models::graph::expand` always queries the live graph),
/// so a historical view keeps using [`EagerSource`] over a fully materialized
/// `BlockGroup::get_graph` instead.
#[derive(Debug)]
pub enum EagerOrSqlSource {
    Eager(EagerSource),
    Sql(SqlGraphSource),
}

impl GraphSource<GenGraph> for EagerOrSqlSource {
    fn ensure_loaded(&mut self, graph: &mut GenGraph, node: GraphNode) -> bool {
        match self {
            Self::Eager(source) => source.ensure_loaded(graph, node),
            Self::Sql(source) => source.ensure_loaded(graph, node),
        }
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
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

    /// Build a tiny on-disk `start -> x -> y -> z -> end` block group and return its DB path,
    /// workspace, and id, so `SqlGraphSource` can be exercised against a real SQLite file
    /// (`SqlGraphSource` opens its own connection lazily via `get_connection`, the same
    /// production entry point `ensure_loaded` itself uses, which is also why this needs a real
    /// file rather than an in-memory database).
    fn setup_chain_block_group(db_path: &std::path::Path) -> (Workspace, HashId) {
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
        for label in ["x", "y", "z"] {
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

        (Workspace::new(db_path.parent().unwrap()), block_group.id)
    }

    #[test]
    fn test_ensure_loaded_pulls_in_a_node_neighbourhood_from_sql() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (workspace, block_group_id) = setup_chain_block_group(&db_path);

        let mut source = SqlGraphSource::new(db_path, workspace, block_group_id);
        let mut graph = GenGraph::new();
        let anchor = GraphNode {
            node_id: HashId::convert_str("y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        graph.add_node(anchor);

        assert!(source.ensure_loaded(&mut graph, anchor));
        let node_ids: std::collections::HashSet<HashId> =
            graph.nodes().map(|node| node.node_id).collect();
        assert!(node_ids.contains(&HashId::convert_str("x")));
        assert!(node_ids.contains(&HashId::convert_str("z")));

        // Idempotent: nothing new to add once the neighbourhood is already loaded.
        assert!(!source.ensure_loaded(&mut graph.clone(), anchor));
    }

    /// Build a tiny on-disk circular `start -> x -> y -> z -> end`, `z -> x` (real closure),
    /// `end -> start` (GFA-import synthetic marker) block group - the shape
    /// `discard_circular_marker_edge` is meant to see: two distinct closing edges, only one of
    /// which is the marker that should be dropped.
    fn setup_circular_block_group(db_path: &std::path::Path) -> (Workspace, HashId) {
        let (workspace, block_group_id) = setup_chain_block_group(db_path);
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
        (workspace, block_group_id)
    }

    #[test]
    fn test_ensure_loaded_discards_marker_and_sentinel_attachment_edges() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (workspace, block_group_id) = setup_circular_block_group(&db_path);

        let mut source = SqlGraphSource::new(db_path, workspace, block_group_id);
        let mut graph = GenGraph::new();
        let anchor = GraphNode {
            node_id: HashId::convert_str("y"),
            sequence_start: 0,
            sequence_end: 5,
        };
        graph.add_node(anchor);
        // A few crawl steps to reach every node the closure edges touch, matching how the real
        // crawl in `crawl::neighborhood` would eventually load them.
        for node in [
            anchor,
            GraphNode {
                node_id: HashId::convert_str("x"),
                sequence_start: 0,
                sequence_end: 5,
            },
            GraphNode {
                node_id: HashId::convert_str("z"),
                sequence_start: 0,
                sequence_end: 5,
            },
            GraphNode {
                node_id: PATH_START_NODE_ID,
                sequence_start: 0,
                sequence_end: 0,
            },
            GraphNode {
                node_id: PATH_END_NODE_ID,
                sequence_start: 0,
                sequence_end: 0,
            },
        ] {
            source.ensure_loaded(&mut graph, node);
        }

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
        let (workspace, block_group_id) = setup_circular_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();

        let seed = seed_block_group_graph(&conn, &workspace, &block_group_id);

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
        let (workspace, block_group_id) = setup_chain_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();

        let seed = seed_block_group_graph(&conn, &workspace, &block_group_id);

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
        let (workspace, block_group_id) = setup_chain_block_group(&db_path);
        let source = SqlGraphSource::new(db_path, workspace, block_group_id);

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
