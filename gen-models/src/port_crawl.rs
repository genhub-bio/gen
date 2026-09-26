//! Directional, lazy traversal of a block group's ports.
//!
//! Expanding a block consumes its closing port's edge group. Each jump opens a block at
//! the far port; sliding closes it at the next active port and caches that port's edges.
//! The closed block stays frontier until the viewer requests another expansion.

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
    edge::{Edge, EdgeError, GroupBlock},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Port {
    node_id: HashId,
    coordinate: i64,
}

impl Port {
    fn of(node: &GraphNode, direction: Direction) -> Self {
        Self {
            node_id: node.node_id,
            coordinate: match direction {
                Direction::Outgoing => node.sequence_end,
                Direction::Incoming => node.sequence_start,
            },
        }
    }

    fn endpoint(edge: &Edge, direction: Direction) -> Self {
        match direction {
            Direction::Outgoing => Self {
                node_id: edge.source_node_id,
                coordinate: edge.source_coordinate,
            },
            Direction::Incoming => Self {
                node_id: edge.target_node_id,
                coordinate: edge.target_coordinate,
            },
        }
    }
}

/// The blocks closed by sliding from one oriented port. A junction closes an arriving jump
/// immediately. Its continuity edge can subsequently open a sequence block at the same
/// oriented port without repeating the arriving jump.
#[derive(Clone, Debug, Default)]
struct PortSlides {
    landing: Option<GraphNode>,
    continuation: Option<GraphNode>,
}

/// Lazily grows one block group's graph. Keep the crawler paired with the graph it describes.
#[derive(Clone, Debug)]
pub struct PortCrawler {
    block_group_id: HashId,
    prune: bool,
    slide_cache: HashMap<(Port, Direction), PortSlides>,
    /// Frontier edge batches are known before their far blocks have been opened and closed.
    edge_groups: HashMap<(Port, Direction), Vec<AugmentedEdge>>,
    visited_edges: HashSet<(HashId, Direction)>,
    materialized: HashSet<(GraphNode, Direction)>,
}

/// A continuity edge joins the two sides of one coordinate rather than jumping elsewhere.
fn is_continuity(edge: &Edge) -> bool {
    edge.source_node_id == edge.target_node_id && edge.source_coordinate == edge.target_coordinate
}

/// Add the graph edges one stored edge projects between `source` and `target`. A junction
/// already in the graph at either endpoint's coordinate takes part too, so a continuity edge
/// connects through it the way the eager graph does.
fn merge_fragment(
    graph: &mut GenGraph,
    augmented_edge: AugmentedEdge,
    source: GraphNode,
    target: GraphNode,
) {
    let mut endpoints = vec![source, target];
    for (endpoint, coordinate) in [
        (source, augmented_edge.edge.source_coordinate),
        (target, augmented_edge.edge.target_coordinate),
    ] {
        let junction = GraphNode {
            node_id: endpoint.node_id,
            sequence_start: coordinate,
            sequence_end: coordinate,
        };
        if graph.contains_node(junction) && !endpoints.contains(&junction) {
            endpoints.push(junction);
        }
    }
    let blocks: Vec<GroupBlock> = endpoints
        .iter()
        .enumerate()
        .map(|(index, endpoint)| {
            GroupBlock::without_sequence(
                index as i64,
                endpoint.node_id,
                endpoint.sequence_start,
                endpoint.sequence_end,
            )
        })
        .collect();
    let (fragment, _) = Edge::build_graph(&[augmented_edge], &blocks);
    for (source, target, graph_edges) in fragment.all_edges() {
        if let Some(existing) = graph.edge_weight_mut(source, target) {
            for graph_edge in graph_edges {
                if !existing
                    .iter()
                    .any(|present| present.edge_id == graph_edge.edge_id)
                {
                    existing.push(*graph_edge);
                }
            }
        } else {
            graph.add_edge(source, target, graph_edges.clone());
        }
    }
}

impl PortCrawler {
    pub fn new(block_group_id: HashId, prune: bool) -> Self {
        Self {
            block_group_id,
            prune,
            slide_cache: HashMap::new(),
            edge_groups: HashMap::new(),
            visited_edges: HashSet::new(),
            materialized: HashSet::new(),
        }
    }

    /// Whether every edge on this side of the block has been resolved.
    pub fn is_complete(&self, node: &GraphNode, direction: Direction) -> bool {
        self.materialized.contains(&(*node, direction))
    }

    /// Complete both sides of the requested blocks, leaving new neighbours as frontier.
    pub fn complete(
        &mut self,
        conn: &GraphConnection,
        graph: &mut GenGraph,
        nodes: &[GraphNode],
    ) -> Result<(), EdgeError> {
        for node in nodes {
            for direction in [Direction::Outgoing, Direction::Incoming] {
                self.materialize(conn, graph, *node, direction)?;
            }
        }
        Ok(())
    }

    /// Complete `frontier` on its `direction` side, then keep walking that way breadth-first,
    /// completing up to `budget` further blocks. A walk stops at any block that was already
    /// complete before this call, since everything past it is either loaded or its own
    /// frontier. Requested blocks are queued first, so they are never cut off by the budget.
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
        let mut requested_remaining = frontier.len();
        let mut remaining_budget = budget;
        while let Some(node) = queue.pop_front() {
            let is_requested = requested_remaining > 0;
            if is_requested {
                requested_remaining -= 1;
            } else if remaining_budget == 0 {
                break;
            }
            if self.is_complete(&node, direction) {
                continue;
            }
            if !is_requested {
                remaining_budget -= 1;
            }
            self.materialize(conn, graph, node, direction)?;
            for neighbor in graph.neighbors_directed(node, direction) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        Ok(())
    }

    fn load_group(
        &mut self,
        conn: &GraphConnection,
        port: Port,
        direction: Direction,
    ) -> Result<Vec<AugmentedEdge>, EdgeError> {
        if let Some(edges) = self.edge_groups.get(&(port, direction)) {
            return Ok(edges.clone());
        }
        let edges = Edge::edges_at_port_direction(
            conn,
            &self.block_group_id,
            port.node_id,
            port.coordinate,
            direction,
        )?;
        self.edge_groups.insert((port, direction), edges.clone());
        Ok(edges)
    }

    /// Close the block opened at `port` and cache its closing port's edge group, which saves
    /// the next expansion's lookup.
    fn slide(
        &mut self,
        conn: &GraphConnection,
        port: Port,
        direction: Direction,
        include_landing: bool,
    ) -> Result<GraphNode, EdgeError> {
        let cached = self.slide_cache.get(&(port, direction)).and_then(|slides| {
            if include_landing {
                slides.landing
            } else {
                slides.continuation
            }
        });
        if let Some(node) = cached {
            return Ok(node);
        }
        let edges = if is_terminal(port.node_id) {
            Vec::new()
        } else {
            Edge::slide_edge_group(
                conn,
                &self.block_group_id,
                (port.node_id, port.coordinate),
                direction,
                include_landing,
            )?
        };
        let frontier = edges
            .first()
            .map_or(port, |edge| Port::endpoint(&edge.edge, direction));
        let node = GraphNode {
            node_id: port.node_id,
            sequence_start: port.coordinate.min(frontier.coordinate),
            sequence_end: port.coordinate.max(frontier.coordinate),
        };
        // A sentinel still needs an exact lookup when expanded. An advancing slide's
        // closing group and an inclusive landing query are already complete.
        if !is_terminal(port.node_id) && (frontier != port || include_landing) {
            self.edge_groups.insert((frontier, direction), edges);
        }
        let slides = self.slide_cache.entry((port, direction)).or_default();
        if include_landing {
            // If no junction interrupted the slide, continuity takes the same route.
            if frontier != port {
                slides.continuation = Some(node);
            }
            slides.landing = Some(node);
        } else {
            slides.continuation = Some(node);
        }
        Ok(node)
    }

    fn land(
        &mut self,
        conn: &GraphConnection,
        edge: &Edge,
        direction: Direction,
    ) -> Result<GraphNode, EdgeError> {
        let port = Port::endpoint(edge, direction.opposite());
        self.slide(conn, port, direction, !is_continuity(edge))
    }

    fn materialize(
        &mut self,
        conn: &GraphConnection,
        graph: &mut GenGraph,
        node: GraphNode,
        direction: Direction,
    ) -> Result<(), EdgeError> {
        if self.is_complete(&node, direction) {
            return Ok(());
        }
        let edges = self.load_group(conn, Port::of(&node, direction), direction)?;
        let port = Port::of(&node, direction);
        let junction = GraphNode {
            node_id: port.node_id,
            sequence_start: port.coordinate,
            sequence_end: port.coordinate,
        };
        let (has_continuity, has_jump) =
            edges
                .iter()
                .fold((false, false), |(has_continuity, has_jump), edge| {
                    if is_continuity(&edge.edge) {
                        (true, has_jump)
                    } else {
                        (has_continuity, true)
                    }
                });
        // An arbitrary viewer anchor can expose the sequence side of a junction before
        // any jump has discovered it. Resolve that ambiguity only for mixed edge groups.
        if node != junction
            && !graph.contains_node(junction)
            && has_continuity
            && has_jump
            && self.slide(conn, port, direction.opposite(), true)? == junction
        {
            graph.add_node(junction);
        }
        for augmented_edge in edges {
            let key = (augmented_edge.edge.id, direction);
            let same_coordinate = is_continuity(&augmented_edge.edge);
            let enters_junction = node != junction && graph.contains_node(junction);
            if enters_junction && !same_coordinate {
                // Non-continuity edges belong to the junction's frontier, not the
                // sequence block ending here. Expanding that junction executes them.
                continue;
            }
            // One continuity edge can project to both sequence -> junction and
            // junction -> sequence. The slide cache prevents repeating its database work.
            if self.visited_edges.contains(&key) && !same_coordinate {
                continue;
            }
            if self.prune && !self.survives_pruning(conn, &augmented_edge)? {
                self.visited_edges.insert(key);
                continue;
            }
            let far_node = if enters_junction {
                junction
            } else {
                self.land(conn, &augmented_edge.edge, direction)?
            };
            let (source, target) = match direction {
                Direction::Outgoing => (node, far_node),
                Direction::Incoming => (far_node, node),
            };
            merge_fragment(graph, augmented_edge, source, target);
            self.visited_edges.insert(key);
        }
        self.materialized.insert((node, direction));
        Ok(())
    }

    /// Reverse pruning needs the source's outgoing siblings; forward crawling already cached them.
    fn survives_pruning(
        &mut self,
        conn: &GraphConnection,
        augmented_edge: &AugmentedEdge,
    ) -> Result<bool, EdgeError> {
        let chromosome_index = augmented_edge.chromosome_index;
        if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
            return Ok(false);
        }
        if chromosome_index == NO_CHROMOSOME_INDEX
            || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
        {
            return Ok(true);
        }
        let edges = self.load_group(
            conn,
            Port::endpoint(&augmented_edge.edge, Direction::Outgoing),
            Direction::Outgoing,
        )?;
        Ok(edges
            .iter()
            .filter(|sibling| sibling.chromosome_index == chromosome_index)
            .max_by_key(|sibling| (sibling.created_on, sibling.edge.id))
            .is_some_and(|newest| newest.edge.id == augmented_edge.edge.id))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use gen_core::{
        HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID,
        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand,
    };
    use gen_graph::{GenGraph, GraphNode};
    use petgraph::Direction;

    use super::{Port, PortCrawler};
    use crate::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        node::Node,
        sample::{NewSample, Sample},
        sequence::Sequence,
        test_helpers::{get_connection, test_workspace},
    };

    fn block(label: &str, start: i64, end: i64) -> GraphNode {
        GraphNode {
            node_id: node_id_for(label),
            sequence_start: start,
            sequence_end: end,
        }
    }

    #[test]
    fn test_linear_crawl_stops_at_the_frontier_in_both_directions() {
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
        for direction in [Direction::Outgoing, Direction::Incoming] {
            let mut crawler = PortCrawler::new(block_group_id, false);
            let mut graph = GenGraph::new();
            let anchor = if direction == Direction::Outgoing {
                start_sentinel()
            } else {
                block("end", 0, 0)
            };
            graph.add_node(anchor);
            crawler
                .expand(&conn, &mut graph, &[anchor], direction, 2)
                .unwrap();
            let frontier = if direction == Direction::Outgoing {
                block("c", 0, 4)
            } else {
                block("a", 0, 4)
            };
            assert!(
                graph.contains_node(frontier),
                "the budgeted walk should reach the last block"
            );
            assert!(
                !crawler.is_complete(&frontier, direction),
                "the last block should stay frontier"
            );
            crawler
                .expand(&conn, &mut graph, &[frontier], direction, 0)
                .unwrap();
            assert!(
                crawler.is_complete(&frontier, direction),
                "expanding the frontier should complete it"
            );
        }
    }

    #[test]
    fn test_branching_convergence_reuses_slides_and_visits_edges_in_both_directions() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[
                ("a", "AAAA"),
                ("b", "CCCC"),
                ("joined", "GGGG"),
                ("tail", "TTTT"),
            ],
            &[
                ("start", 0, "a", 0, 0),
                ("start", 0, "b", 0, 1),
                ("a", 4, "joined", 0, 0),
                ("b", 4, "joined", 0, 0),
                ("joined", 4, "tail", 0, 0),
                ("tail", 4, "end", 0, 0),
            ],
        );
        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();
        graph.add_node(start_sentinel());
        crawler
            .expand(
                &conn,
                &mut graph,
                &[start_sentinel()],
                Direction::Outgoing,
                0,
            )
            .unwrap();
        assert_eq!(
            graph.neighbors(start_sentinel()).count(),
            2,
            "the start sentinel should fan out to both branches"
        );
        crawler
            .expand(
                &conn,
                &mut graph,
                &[block("a", 0, 4), block("b", 0, 4)],
                Direction::Outgoing,
                0,
            )
            .unwrap();
        assert!(
            !crawler.is_complete(&block("joined", 0, 4), Direction::Outgoing),
            "the converged block should stay frontier"
        );
        crawler
            .expand(
                &conn,
                &mut graph,
                &[block("joined", 0, 4)],
                Direction::Outgoing,
                10,
            )
            .unwrap();
        let forward_shape = shape(&graph);
        crawler
            .expand(
                &conn,
                &mut graph,
                &[block("end", 0, 0)],
                Direction::Incoming,
                10,
            )
            .unwrap();
        assert_eq!(
            shape(&graph),
            forward_shape,
            "a reverse crawl should add nothing to the forward graph"
        );
        assert_eq!(
            crawler.visited_edges.len(),
            12,
            "each of the six edges should be visited once per direction"
        );
        assert!(
            crawler.slide_cache.contains_key(&(
                Port {
                    node_id: node_id_for("joined"),
                    coordinate: 0
                },
                Direction::Outgoing
            )),
            "both branches land on the same cached slide"
        );
    }

    #[test]
    fn test_outer_junction_and_continuity_match_eager_graph_in_both_directions() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("ref", "AAAAAAAAAA")],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 0, "ref", 0, 0),
                ("ref", 0, "ref", 3, 1),
                ("ref", 3, "ref", 3, 0),
                ("ref", 3, "ref", 6, 1),
                ("ref", 6, "ref", 6, 0),
                ("ref", 6, "ref", 10, 1),
                ("ref", 10, "ref", 10, 0),
                ("ref", 10, "end", 0, 0),
            ],
        );
        let eager = BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
        for anchor in [start_sentinel(), block("end", 0, 0)] {
            let mut crawler = PortCrawler::new(block_group_id, false);
            let mut graph = GenGraph::new();
            graph.add_node(anchor);
            crawl_to_exhaustion(&conn, &mut crawler, &mut graph);
            assert_eq!(
                connected_shape(&graph),
                connected_shape(&eager),
                "the lazy crawl should match the eager graph"
            );
        }
    }

    #[test]
    fn test_jump_closes_at_junction_before_following_its_edge_batch() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("ref", "AAAAAAAAAA")],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 3, "ref", 6, 0),
                ("ref", 6, "ref", 9, 0),
                ("ref", 9, "end", 0, 0),
            ],
        );
        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();
        graph.add_node(start_sentinel());
        crawler
            .expand(
                &conn,
                &mut graph,
                &[start_sentinel()],
                Direction::Outgoing,
                1,
            )
            .unwrap();
        assert!(
            graph.contains_node(block("ref", 6, 6)),
            "the jump should close at the landing junction"
        );
        assert!(
            !graph.contains_node(block("ref", 9, 9)),
            "the junction's own jump should not be followed yet"
        );
        assert!(
            !crawler.is_complete(&block("ref", 6, 6), Direction::Outgoing),
            "the landing junction should stay frontier"
        );
    }

    #[test]
    fn test_continuity_stops_at_junction_without_expanding_it() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("ref", "AAAAAAAAAA")],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 0, "ref", 0, 0),
                ("ref", 0, "ref", 3, 1),
                ("ref", 3, "ref", 3, 0),
                ("ref", 3, "ref", 6, 1),
                ("ref", 6, "end", 0, 0),
            ],
        );
        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();
        graph.add_node(start_sentinel());
        crawler
            .expand(
                &conn,
                &mut graph,
                &[start_sentinel()],
                Direction::Outgoing,
                1,
            )
            .unwrap();
        crawler
            .expand(
                &conn,
                &mut graph,
                &[block("ref", 0, 3)],
                Direction::Outgoing,
                0,
            )
            .unwrap();
        assert!(
            graph.contains_edge(block("ref", 0, 3), block("ref", 3, 3)),
            "continuity should connect the block to its junction"
        );
        assert!(
            !graph.contains_node(block("ref", 3, 6)),
            "the junction should not be expanded past"
        );
        assert!(
            !graph.contains_node(block("ref", 6, 6)),
            "the junction's jump should not be followed"
        );
        crawler
            .expand(
                &conn,
                &mut graph,
                &[block("ref", 3, 3)],
                Direction::Outgoing,
                0,
            )
            .unwrap();
        assert!(
            graph.contains_edge(block("ref", 3, 3), block("ref", 3, 6)),
            "expanding the junction should open the next block"
        );
    }

    #[test]
    fn test_outer_continuity_and_internal_cross_node_ports_match_eager_graph() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("ref", "AAAAAAAAAA"), ("insert", "CCCC")],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 0, "ref", 0, 0),
                ("ref", 3, "ref", 3, 0),
                ("ref", 3, "insert", 0, 1),
                ("insert", 4, "ref", 3, 1),
                ("ref", 10, "ref", 10, 0),
                ("ref", 10, "end", 0, 0),
            ],
        );
        let eager = BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
        for anchor in [start_sentinel(), block("end", 0, 0)] {
            let mut crawler = PortCrawler::new(block_group_id, false);
            let mut graph = GenGraph::new();
            graph.add_node(anchor);
            crawl_to_exhaustion(&conn, &mut crawler, &mut graph);
            assert_eq!(
                connected_shape(&graph),
                connected_shape(&eager),
                "the lazy crawl should match the eager graph"
            );
        }
    }

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
