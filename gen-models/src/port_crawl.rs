//! Grows a block group's `GenGraph` one `(node, coordinate)` port at a time, for viewers that
//! cannot afford to read every edge touching a node.
//!
//! A `GraphNode` is a slice of a backing node, and its incoming edges all arrive at its start
//! port while its outgoing edges all leave from its end port. So the graph around a port is fully
//! described by the edges at that port plus the nearest coordinates on either side of it, which
//! bound the slices that end or start there. Each of those is an indexed lookup, independent of
//! how many other edges the backing node carries.
//!
//! The crawler keeps two sets. A port is *loaded* once its edges and neighboring coordinates
//! have been read; a port is *materialized* once every one of its edges is in the graph, which
//! requires the far end of each edge to be loaded too, so the slice it lands on is known. A
//! graph node is complete in a direction when the port on that side is materialized. Every other
//! node in the graph is frontier on that side: its identity is exact, but some of its edges may
//! be missing.

use std::collections::{HashMap, HashSet, VecDeque};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, is_terminal,
};
use gen_graph::{GenGraph, GraphNode};
use petgraph::Direction;

use crate::{
    block_group_edge::AugmentedEdge,
    db::GraphConnection,
    edge::{Edge, EdgeError, GroupBlock, PortEdges},
};

/// One edge endpoint position: a coordinate on a backing node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Port {
    node_id: HashId,
    coordinate: i64,
}

impl Port {
    /// The port on `direction`'s side of `node`: its end for outgoing edges, its start for
    /// incoming ones. A junction or sentinel has the same port on both sides.
    fn of(node: &GraphNode, direction: Direction) -> Self {
        let coordinate = match direction {
            Direction::Outgoing => node.sequence_end,
            Direction::Incoming => node.sequence_start,
        };
        Port {
            node_id: node.node_id,
            coordinate,
        }
    }
}

/// What the database says about one port.
#[derive(Clone, Debug)]
struct LoadedPort {
    edges: PortEdges,
    /// The nearest lower coordinate on the same node with an edge endpoint, if any.
    previous: Option<i64>,
    /// The nearest higher coordinate on the same node with an edge endpoint, if any.
    next: Option<i64>,
}

impl LoadedPort {
    /// Whether an edge between two different coordinates of this port's own node both leaves
    /// and arrives here. `Edge::blocks_from_edges` turns such a coordinate into a zero-width
    /// junction so the two jumps can chain.
    fn is_jump_junction(&self, port: Port) -> bool {
        let jumps_out = self.edges.leaving.iter().any(|augmented_edge| {
            augmented_edge.edge.target_node_id == port.node_id
                && augmented_edge.edge.target_coordinate != port.coordinate
        });
        let jumps_in = self.edges.arriving.iter().any(|augmented_edge| {
            augmented_edge.edge.source_node_id == port.node_id
                && augmented_edge.edge.source_coordinate != port.coordinate
        });
        jumps_out && jumps_in
    }

    /// The `(start, end)` slices of the backing node that end or start at `port`, carved the same
    /// way `Edge::blocks_from_edges` carves them from the whole block group.
    fn blocks(&self, port: Port) -> Vec<(i64, i64)> {
        if is_terminal(port.node_id) {
            return vec![(0, 0)];
        }
        let coordinate = port.coordinate;
        // Mirrors `Edge::get_block_intervals`: a jump junction, or an outer junction where an
        // edge leaves the node's first coordinate or arrives at its last one.
        let has_junction = self.is_jump_junction(port)
            || (self.previous.is_none() && !self.edges.leaving.is_empty())
            || (self.next.is_none() && !self.edges.arriving.is_empty());
        let mut blocks = Vec::with_capacity(3);
        if let Some(previous) = self.previous {
            blocks.push((previous, coordinate));
        }
        if has_junction {
            blocks.push((coordinate, coordinate));
        }
        if let Some(next) = self.next {
            blocks.push((coordinate, next));
        }
        blocks
    }
}

/// Lazily grows one block group's graph from its ports. Owned by a viewer's graph source for as
/// long as it keeps growing the same `GenGraph`, since what it records describes that graph.
#[derive(Clone, Debug)]
pub struct PortCrawler {
    block_group_id: HashId,
    /// Leave out the edges `BlockGroup::prune_graph` would remove, deciding per source port.
    prune: bool,
    loaded: HashMap<Port, LoadedPort>,
    materialized: HashSet<Port>,
}

impl PortCrawler {
    pub fn new(block_group_id: HashId, prune: bool) -> Self {
        Self {
            block_group_id,
            prune,
            loaded: HashMap::new(),
            materialized: HashSet::new(),
        }
    }

    /// Whether the graph already carries every edge on `direction`'s side of `node`.
    pub fn is_complete(&self, node: &GraphNode, direction: Direction) -> bool {
        self.materialized.contains(&Port::of(node, direction))
    }

    /// Complete `frontier` on its `direction` side, then keep walking that way breadth-first,
    /// completing up to `budget` further nodes. A walk stops at any node that was already
    /// complete before this call, since everything past it is either loaded or its own frontier.
    pub fn expand(
        &mut self,
        conn: &GraphConnection,
        graph: &mut GenGraph,
        frontier: &[GraphNode],
        direction: Direction,
        budget: usize,
    ) -> Result<(), EdgeError> {
        let mut queue: VecDeque<GraphNode> = frontier.iter().copied().collect();
        let mut visited: HashSet<GraphNode> = frontier.iter().copied().collect();
        let mut materialized_here: HashSet<Port> = HashSet::new();
        let mut requested_remaining = frontier.len();
        let mut remaining_budget = budget;
        while let Some(node) = queue.pop_front() {
            let is_requested = requested_remaining > 0;
            if is_requested {
                requested_remaining -= 1;
            } else if remaining_budget == 0 {
                break;
            }
            let port = Port::of(&node, direction);
            if self.materialized.contains(&port) && !materialized_here.contains(&port) {
                continue;
            }
            if !is_requested {
                remaining_budget -= 1;
            }
            if !self.materialized.contains(&port) {
                self.materialize(conn, graph, port)?;
                materialized_here.insert(port);
            }
            for neighbor in graph.neighbors_directed(node, direction) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        Ok(())
    }

    /// Read `port`'s edges and neighboring coordinates, once.
    fn load(&mut self, conn: &GraphConnection, port: Port) -> Result<(), EdgeError> {
        if self.loaded.contains_key(&port) {
            return Ok(());
        }
        let edges = Edge::edges_at_port(conn, &self.block_group_id, port.node_id, port.coordinate)?;
        let (previous, next) = if is_terminal(port.node_id) {
            (None, None)
        } else {
            (
                Edge::adjacent_edge_coordinate(
                    conn,
                    &self.block_group_id,
                    port.node_id,
                    port.coordinate,
                    Direction::Incoming,
                )?,
                Edge::adjacent_edge_coordinate(
                    conn,
                    &self.block_group_id,
                    port.node_id,
                    port.coordinate,
                    Direction::Outgoing,
                )?,
            )
        };
        self.loaded.insert(
            port,
            LoadedPort {
                edges,
                previous,
                next,
            },
        );
        Ok(())
    }

    /// Add every slice at `port` and every edge leaving or arriving there to `graph`.
    fn materialize(
        &mut self,
        conn: &GraphConnection,
        graph: &mut GenGraph,
        port: Port,
    ) -> Result<(), EdgeError> {
        self.load(conn, port)?;
        let port_edges = self.loaded[&port].edges.clone();
        let far_ports: Vec<Port> = port_edges
            .leaving
            .iter()
            .map(|augmented_edge| Port {
                node_id: augmented_edge.edge.target_node_id,
                coordinate: augmented_edge.edge.target_coordinate,
            })
            .chain(port_edges.arriving.iter().map(|augmented_edge| Port {
                node_id: augmented_edge.edge.source_node_id,
                coordinate: augmented_edge.edge.source_coordinate,
            }))
            .collect();
        for far_port in &far_ports {
            self.load(conn, *far_port)?;
        }

        // A same-coordinate edge both leaves and arrives at its port; keep one copy.
        let mut edge_ids: HashSet<HashId> = HashSet::new();
        let edges: Vec<AugmentedEdge> = port_edges
            .leaving
            .iter()
            .chain(port_edges.arriving.iter())
            .filter(|augmented_edge| edge_ids.insert(augmented_edge.edge.id))
            .filter(|augmented_edge| !self.prune || self.survives_pruning(augmented_edge))
            .cloned()
            .collect();

        let mut ports: Vec<Port> = vec![port];
        ports.extend(far_ports);
        let mut seen_blocks: HashSet<(HashId, i64, i64)> = HashSet::new();
        let mut blocks: Vec<GroupBlock> = vec![];
        for block_port in ports {
            for (start, end) in self.loaded[&block_port].blocks(block_port) {
                if seen_blocks.insert((block_port.node_id, start, end)) {
                    blocks.push(GroupBlock::without_sequence(
                        blocks.len() as i64,
                        block_port.node_id,
                        start,
                        end,
                    ));
                }
            }
        }

        // Only the slices at `port` itself are added unconditionally: they are what this call
        // completes. A slice at a far port enters the graph through the edges that reach it.
        for (start, end) in self.loaded[&port].blocks(port) {
            graph.add_node(GraphNode {
                node_id: port.node_id,
                sequence_start: start,
                sequence_end: end,
            });
        }
        let (fragment, _) = Edge::build_graph(&edges, &blocks);
        for (source, target, graph_edges) in fragment.all_edges() {
            match graph.edge_weight_mut(source, target) {
                Some(existing) => {
                    for graph_edge in graph_edges {
                        if !existing
                            .iter()
                            .any(|present| present.edge_id == graph_edge.edge_id)
                        {
                            existing.push(*graph_edge);
                        }
                    }
                }
                None => {
                    graph.add_edge(source, target, graph_edges.clone());
                }
            }
        }
        self.materialized.insert(port);
        Ok(())
    }

    /// Whether `BlockGroup::prune_graph` keeps `augmented_edge`: an edit-site marker never
    /// survives, an edge with no chromosome index always does, and otherwise only the newest
    /// edge per chromosome index leaving the same source port is kept. The source port is always
    /// loaded by the time this runs, because it is either the port being materialized or the
    /// far end of one of its edges.
    fn survives_pruning(&self, augmented_edge: &AugmentedEdge) -> bool {
        let chromosome_index = augmented_edge.chromosome_index;
        if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
            return false;
        }
        if chromosome_index == NO_CHROMOSOME_INDEX
            || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
        {
            return true;
        }
        let source_port = Port {
            node_id: augmented_edge.edge.source_node_id,
            coordinate: augmented_edge.edge.source_coordinate,
        };
        self.loaded[&source_port]
            .edges
            .leaving
            .iter()
            .filter(|sibling| sibling.chromosome_index == chromosome_index)
            .max_by_key(|sibling| (sibling.created_on, sibling.edge.id))
            .is_some_and(|newest| newest.edge.id == augmented_edge.edge.id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};

    use super::*;
    use crate::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
        test_helpers::{get_connection, test_workspace},
    };

    /// One stored edge: `(source label, source coordinate, target label, target coordinate,
    /// chromosome index)`, where the labels "start" and "end" name the path sentinels.
    type EdgeSpec<'a> = (&'a str, i64, &'a str, i64, i64);

    fn node_id_for(label: &str) -> HashId {
        match label {
            "start" => PATH_START_NODE_ID,
            "end" => PATH_END_NODE_ID,
            _ => HashId::convert_str(label),
        }
    }

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

    /// A reference node with an insertion, two adjacent deletions that meet at a junction, a
    /// deletion of the first base, and the same-coordinate edges that keep the reference
    /// continuous across each split.
    fn junction_heavy_block_group(conn: &GraphConnection) -> HashId {
        setup_block_group(
            conn,
            &[("ref", "ACGTACGTAC"), ("insert", "GGGG")],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 10, "end", 0, 0),
                ("ref", 3, "insert", 0, 1),
                ("insert", 4, "ref", 5, 1),
                ("ref", 3, "ref", 3, 0),
                ("ref", 5, "ref", 5, 0),
                ("ref", 6, "ref", 8, 2),
                ("ref", 8, "ref", 9, 3),
                ("ref", 6, "ref", 6, 0),
                ("ref", 9, "ref", 9, 0),
                ("start", 0, "ref", 1, 4),
                ("ref", 1, "ref", 1, 0),
            ],
        )
    }

    fn start_sentinel() -> GraphNode {
        GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        }
    }

    /// Expand every incomplete node in both directions until nothing is left to load.
    fn crawl_to_exhaustion(
        conn: &GraphConnection,
        crawler: &mut PortCrawler,
        graph: &mut GenGraph,
    ) {
        loop {
            let incomplete: Vec<(GraphNode, Direction)> = graph
                .nodes()
                .flat_map(|node| [(node, Direction::Outgoing), (node, Direction::Incoming)])
                .filter(|(node, direction)| !crawler.is_complete(node, *direction))
                .collect();
            if incomplete.is_empty() {
                return;
            }
            for (node, direction) in incomplete {
                crawler
                    .expand(conn, graph, &[node], direction, usize::MAX)
                    .unwrap();
            }
        }
    }

    type GraphShape = (
        BTreeSet<GraphNode>,
        BTreeSet<(GraphNode, GraphNode, Vec<HashId>)>,
    );

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

    /// Drop nodes without edges, which the eager graph keeps for every slice but a crawl only
    /// reaches through an edge.
    fn connected_shape(graph: &GenGraph) -> GraphShape {
        let (nodes, edges) = shape(graph);
        let nodes = nodes
            .into_iter()
            .filter(|node| {
                graph
                    .neighbors_directed(*node, Direction::Outgoing)
                    .chain(graph.neighbors_directed(*node, Direction::Incoming))
                    .next()
                    .is_some()
            })
            .collect();
        (nodes, edges)
    }

    #[test]
    fn test_exhaustive_crawl_matches_the_eagerly_built_graph() {
        let conn = get_connection(None).unwrap();
        let block_group_id = junction_heavy_block_group(&conn);
        let eager = BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();

        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();
        graph.add_node(start_sentinel());
        crawl_to_exhaustion(&conn, &mut crawler, &mut graph);

        assert_eq!(connected_shape(&graph), connected_shape(&eager));
    }

    #[test]
    fn test_expand_with_zero_budget_completes_only_the_requested_side() {
        let conn = get_connection(None).unwrap();
        let block_group_id = junction_heavy_block_group(&conn);
        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();
        let start = start_sentinel();
        graph.add_node(start);

        crawler
            .expand(&conn, &mut graph, &[start], Direction::Outgoing, 0)
            .unwrap();

        assert!(crawler.is_complete(&start, Direction::Outgoing));
        let successors: BTreeSet<GraphNode> = graph
            .neighbors_directed(start, Direction::Outgoing)
            .collect();
        assert!(!successors.is_empty());
        for successor in successors {
            assert!(
                !crawler.is_complete(&successor, Direction::Outgoing),
                "a zero budget should leave {successor:?} as frontier"
            );
        }
    }

    #[test]
    fn test_expand_stops_after_the_budget() {
        let conn = get_connection(None).unwrap();
        let nodes: Vec<(String, &str)> = (0..20).map(|i| (format!("n{i}"), "ACGT")).collect();
        let node_refs: Vec<(&str, &str)> = nodes
            .iter()
            .map(|(label, sequence)| (label.as_str(), *sequence))
            .collect();
        let labels: Vec<&str> = nodes.iter().map(|(label, _)| label.as_str()).collect();
        let mut edges: Vec<EdgeSpec> = vec![("start", 0, labels[0], 0, 0)];
        for pair in labels.windows(2) {
            edges.push((pair[0], 4, pair[1], 0, 0));
        }
        edges.push((labels[19], 4, "end", 0, 0));
        let block_group_id = setup_block_group(&conn, &node_refs, &edges);
        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();
        let start = start_sentinel();
        graph.add_node(start);

        crawler
            .expand(&conn, &mut graph, &[start], Direction::Outgoing, 3)
            .unwrap();

        let complete = graph
            .nodes()
            .filter(|node| crawler.is_complete(node, Direction::Outgoing))
            .count();
        assert_eq!(complete, 4, "the start sentinel plus three budgeted nodes");
        assert!(graph.node_count() < 10);
    }

    #[test]
    fn test_pruning_keeps_the_newest_edge_per_source_port_and_chromosome_index() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[
                ("y", "ACGTA"),
                ("old", "CCCCC"),
                ("new", "GGGGG"),
                ("marked", "TTTTT"),
                ("tail", "AAAAA"),
            ],
            &[
                ("start", 0, "y", 0, 0),
                ("y", 5, "old", 0, 1),
                ("y", 5, "new", 0, 1),
                ("y", 5, "marked", 0, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
                ("y", 5, "tail", 0, NO_CHROMOSOME_INDEX),
            ],
        );
        let mut crawler = PortCrawler::new(block_group_id, true);
        let mut graph = GenGraph::new();
        graph.add_node(start_sentinel());
        crawl_to_exhaustion(&conn, &mut crawler, &mut graph);

        let reached: BTreeSet<HashId> = graph
            .all_edges()
            .flat_map(|(source, target, _)| [source.node_id, target.node_id])
            .collect();
        assert!(reached.contains(&HashId::convert_str("new")));
        assert!(reached.contains(&HashId::convert_str("tail")));
        assert!(!reached.contains(&HashId::convert_str("old")));
        assert!(!reached.contains(&HashId::convert_str("marked")));
    }
}
