//! A [`GraphSource`] over main's [`PortCrawler`] that the lazy viewer shares across threads.
//! The crawler itself is single-threaded and infallible about its connection; the Jupyter widget
//! and the background batch loader need a `Send + Sync` source whose connection failures surface
//! as errors instead of panics, so this wraps one serialized connection around a crawler.

use std::sync::{Arc, Mutex};

use gen_core::HashId;
use gen_graph::{GenGraph, GraphNode, GraphSource};
use gen_models::{
    db::GraphConnection,
    edge::EdgeError,
    port_crawl::{LocateError, PortCrawler},
};
use petgraph::Direction;
use thiserror::Error;

/// A database-backed adjacency source for a retained local graph.
///
/// Start with an empty graph and seed it with [`Self::locate`] or a known graph slice.
/// Keep this source paired with that graph; clone both together. Clones share a serialized
/// database connection, preserving its selected branch, but keep independent crawl state.
/// The source is `Send + Sync`; loading still requires exclusive access to each graph/source pair.
/// The database must remain on the same block-group revision while adjacency is retained.
#[derive(Debug)]
pub struct PortGraphSource {
    connection: Arc<Mutex<GraphConnection>>,
    crawler: PortCrawler,
}

/// A failure to access the source's connection or load adjacency.
#[derive(Debug, Error)]
pub enum PortGraphSourceError {
    /// A database query or graph construction step failed.
    #[error(transparent)]
    Edge(#[from] EdgeError),
    /// A previous panic left the shared database session in an uncertain state.
    #[error("the graph source connection lock is poisoned")]
    ConnectionPoisoned,
}

impl Clone for PortGraphSource {
    fn clone(&self) -> Self {
        Self {
            connection: Arc::clone(&self.connection),
            crawler: self.crawler.clone(),
        }
    }
}

impl PortGraphSource {
    /// Take ownership of a connection and create an unexplored source for one block group.
    pub fn new(connection: GraphConnection, block_group_id: HashId, prune: bool) -> Self {
        Self {
            connection: Arc::new(Mutex::new(connection)),
            crawler: PortCrawler::new(block_group_id, prune),
        }
    }

    /// Add the slice containing a backing-node coordinate as an unexplored seed.
    /// Fails with [`LocateError::NotFound`] when the position does not belong to the block group
    /// and [`LocateError::Unreachable`] when a pruned source finds it cut off from the start.
    pub fn locate(
        &mut self,
        graph: &mut GenGraph,
        node_id: HashId,
        coordinate: i64,
    ) -> Result<GraphNode, LocateError> {
        let connection = self.connection.lock().map_err(|_| {
            LocateError::Unavailable(PortGraphSourceError::ConnectionPoisoned.to_string())
        })?;
        self.crawler.locate(&connection, graph, node_id, coordinate)
    }
}

impl GraphSource<GenGraph> for PortGraphSource {
    type Error = PortGraphSourceError;

    fn is_frontier(&self, node: GraphNode, direction: Direction) -> bool {
        !self.crawler.is_complete(&node, direction)
    }

    fn expand_frontier(
        &mut self,
        graph: &mut GenGraph,
        frontier: &[GraphNode],
    ) -> Result<(), Self::Error> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| PortGraphSourceError::ConnectionPoisoned)?;
        Ok(self.crawler.complete(&connection, graph, frontier)?)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, sync::Arc, thread};

    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::{GenGraph, GraphNode, GraphSource};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        node::Node,
        port_crawl::LocateError,
        sample::{NewSample, Sample},
        sequence::Sequence,
    };
    use petgraph::Direction;

    use super::{PortGraphSource, PortGraphSourceError};
    use crate::test_helpers::get_connection;

    /// One stored edge: `(source label, source coordinate, target label, target coordinate,
    /// chromosome index)`, where the labels "start" and "end" name the path sentinels.
    type EdgeSpec<'a> = (&'a str, i64, &'a str, i64, i64);

    type GraphShape = (
        BTreeSet<GraphNode>,
        BTreeSet<(GraphNode, GraphNode, Vec<HashId>)>,
    );

    fn node_id_for(label: &str) -> HashId {
        match label {
            "start" => PATH_START_NODE_ID,
            "end" => PATH_END_NODE_ID,
            _ => HashId::convert_str(label),
        }
    }

    fn block(label: &str, start: i64, end: i64) -> GraphNode {
        GraphNode {
            node_id: node_id_for(label),
            sequence_start: start,
            sequence_end: end,
        }
    }

    fn start_sentinel() -> GraphNode {
        GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    /// Store `nodes` and `edges` as one block group. Each edge gets its own `created_on`, in
    /// list order, so pruning has a well-defined newest edge per chromosome index.
    fn setup_block_group(
        conn: &GraphConnection,
        nodes: &[(&str, &str)],
        edges: &[EdgeSpec<'_>],
    ) -> HashId {
        Collection::get_or_create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "chr1",
                ..Default::default()
            },
        )
        .unwrap();
        for (label, sequence) in nodes {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(sequence)
                .save(conn)
                .unwrap();
            Node::create(conn, &sequence.hash, &HashId::convert_str(label)).unwrap();
        }
        for (order, (source, source_coordinate, target, target_coordinate, chromosome_index)) in
            edges.iter().enumerate()
        {
            let edge = Edge::create(
                conn,
                node_id_for(source),
                *source_coordinate,
                Strand::Forward,
                node_id_for(target),
                *target_coordinate,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: *chromosome_index,
                    phased: 0,
                }],
            );
            conn.execute(
                "UPDATE block_group_edges SET created_on = ?1 WHERE edge_id = ?2",
                (order as i64, edge.id),
            )
            .unwrap();
        }
        block_group.id
    }

    fn shape(graph: &GenGraph) -> GraphShape {
        let nodes = graph.nodes().collect();
        let edges = graph
            .all_edges()
            .map(|(source, target, graph_edges)| {
                let mut edge_ids: Vec<HashId> = graph_edges
                    .iter()
                    .map(|graph_edge| graph_edge.edge_id)
                    .collect();
                edge_ids.sort();
                (source, target, edge_ids)
            })
            .collect();
        (nodes, edges)
    }

    #[test]
    fn test_port_graph_source_moves_between_threads_with_shared_connection() {
        assert_send_sync::<PortGraphSource>();
        let connection = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &connection,
            &[("a", "AAAA")],
            &[("start", 0, "a", 0, 0), ("a", 4, "end", 0, 0)],
        );
        let mut source = PortGraphSource::new(connection, block_group_id, false);
        let mut graph = GenGraph::new();
        let anchor = source
            .locate(&mut graph, node_id_for("a"), 2)
            .expect("should locate the block");
        let mut cloned_source = source.clone();
        let mut cloned_graph = graph.clone();
        assert!(Arc::ptr_eq(&source.connection, &cloned_source.connection));
        let worker = thread::spawn(move || {
            cloned_source
                .expand_frontier(&mut cloned_graph, &[anchor])
                .unwrap();
            (cloned_source, cloned_graph)
        });
        let (mut cloned_source, mut cloned_graph) = worker
            .join()
            .expect("should complete adjacency on another thread");
        assert!(source.is_frontier(anchor, Direction::Outgoing));
        assert!(!cloned_source.is_frontier(anchor, Direction::Outgoing));
        let worker = thread::spawn(move || {
            source.expand_frontier(&mut graph, &[anchor]).unwrap();
            (source, graph)
        });
        // Both graph/source pairs can continue loading through the same serialized session.
        cloned_source
            .expand_frontier(&mut cloned_graph, &[start_sentinel()])
            .unwrap();
        let (source, graph) = worker
            .join()
            .expect("should expand through the shared connection");
        assert!(!source.is_frontier(anchor, Direction::Outgoing));
        assert_eq!(shape(&graph), shape(&cloned_graph));
    }

    #[test]
    fn test_port_graph_source_reports_poisoned_connection() {
        let mut source = PortGraphSource::new(
            get_connection(None).unwrap(),
            HashId::convert_str("group"),
            false,
        );
        let connection = Arc::clone(&source.connection);
        assert!(
            thread::spawn(move || {
                let _guard = connection.lock().expect("should lock the connection");
                panic!("simulate a panic while using the connection");
            })
            .join()
            .is_err()
        );
        let mut graph = GenGraph::new();
        let anchor = start_sentinel();
        graph.add_node(anchor);
        assert!(matches!(
            source.expand_frontier(&mut graph, &[anchor]),
            Err(PortGraphSourceError::ConnectionPoisoned)
        ));
        assert!(matches!(
            source.locate(&mut graph, node_id_for("a"), 2),
            Err(LocateError::Unavailable(_))
        ));
        assert!(source.is_frontier(anchor, Direction::Outgoing));
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_port_graph_source_retains_immediate_adjacency() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA"), ("b", "CCCC"), ("c", "GGGG")],
            &[
                ("start", 0, "a", 0, 0),
                ("a", 4, "b", 0, 0),
                ("b", 4, "c", 0, 0),
                ("c", 4, "end", 0, 0),
            ],
        );
        let mut source = PortGraphSource::new(conn, block_group_id, false);
        let mut graph = GenGraph::new();
        let anchor = source
            .locate(&mut graph, node_id_for("b"), 2)
            .expect("should locate the block");
        assert_eq!(graph.node_count(), 1);
        source.expand_frontier(&mut graph, &[anchor]).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert!(graph.contains_edge(block("a", 0, 4), anchor));
        assert!(graph.contains_edge(anchor, block("c", 0, 4)));
        for direction in [Direction::Incoming, Direction::Outgoing] {
            assert!(!source.is_frontier(anchor, direction));
            assert!(source.is_frontier(block("a", 0, 4), direction));
            assert!(source.is_frontier(block("c", 0, 4), direction));
        }
        let snapshot = |graph: &GenGraph| {
            (
                graph.nodes().collect::<BTreeSet<_>>(),
                graph
                    .all_edges()
                    .map(|(source, target, edges)| (source, target, edges.clone()))
                    .collect::<BTreeSet<_>>(),
            )
        };
        let expected = snapshot(&graph);
        source
            .expand_frontier(&mut graph, &[anchor, anchor])
            .unwrap();
        assert_eq!(
            snapshot(&graph),
            expected,
            "repeated requests should preserve edge metadata"
        );

        let mut cloned_source = source.clone();
        let mut cloned_graph = graph.clone();
        cloned_source
            .expand_frontier(&mut cloned_graph, &[block("a", 0, 4)])
            .unwrap();
        assert!(source.is_frontier(block("a", 0, 4), Direction::Incoming));
        assert_eq!(snapshot(&graph), expected);
        assert!(!cloned_source.is_frontier(block("a", 0, 4), Direction::Incoming));
        let start = start_sentinel();
        cloned_source
            .expand_frontier(&mut cloned_graph, &[start])
            .unwrap();
        assert!(
            !cloned_source.is_frontier(start, Direction::Incoming),
            "a known dead end should be complete"
        );
    }

    #[test]
    fn test_port_graph_source_retries_failed_adjacency() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA")],
            &[("start", 0, "a", 0, 0), ("a", 4, "end", 0, 0)],
        );
        let mut source = PortGraphSource::new(conn, block_group_id, false);
        let mut graph = GenGraph::new();
        let anchor = source
            .locate(&mut graph, node_id_for("a"), 2)
            .expect("should locate the block");
        source
            .connection
            .lock()
            .unwrap()
            .execute_batch("ALTER TABLE edges RENAME TO unavailable_edges;")
            .unwrap();
        assert!(source.expand_frontier(&mut graph, &[anchor]).is_err());
        for direction in [Direction::Incoming, Direction::Outgoing] {
            assert!(source.is_frontier(anchor, direction));
        }
        source
            .connection
            .lock()
            .unwrap()
            .execute_batch("ALTER TABLE unavailable_edges RENAME TO edges;")
            .unwrap();
        source.expand_frontier(&mut graph, &[anchor]).unwrap();
        for direction in [Direction::Incoming, Direction::Outgoing] {
            assert!(!source.is_frontier(anchor, direction));
        }
        assert_eq!(graph.edge_count(), 2);
    }
}
