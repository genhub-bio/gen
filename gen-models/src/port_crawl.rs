//! Directional, lazy traversal of a block group's ports.
//!
//! Expanding a block reads the edges at one of its ports. Each edge's endpoints are resolved
//! against their adjacent sequence slices and any generated junction, matching the eager graph
//! builder without loading every edge of a long backing node.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, is_terminal,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use petgraph::Direction;
use thiserror::Error;

use crate::{
    block_group_edge::AugmentedEdge,
    db::GraphConnection,
    edge::{Edge, EdgeError, GroupBlock},
    path::Path,
};

/// A zero-based position on a stored node or along a path in the requested block group.
pub enum CrawlStart<'a> {
    /// A coordinate in the backing node's stored sequence.
    Node { node_id: HashId, coordinate: i64 },
    /// A coordinate along the path's assembled sequence.
    Path { path: &'a Path, coordinate: i64 },
}

#[derive(Debug, Error)]
pub enum CrawlError {
    /// A database query or graph construction step failed.
    #[error(transparent)]
    Edge(#[from] EdgeError),
    /// The returned graph must be able to contain its starting slice.
    #[error("the crawl budget must contain at least one graph node")]
    EmptyBudget,
    /// The requested path does not belong to the requested block group.
    #[error("the starting path belongs to another block group")]
    PathBlockGroupMismatch,
}

/// Return at most `node_budget` graph slices around a position, with all edges between them.
/// A position outside the block group returns `None`. Node coordinates refer to the backing
/// node's sequence; path coordinates count bases along the path, including reverse blocks.
/// The crawl may temporarily discover adjacent frontier nodes while resolving the selected
/// slices, but only selected slices and their internal edges appear in the returned graph.
pub fn crawl_graph(
    conn: &GraphConnection,
    block_group_id: HashId,
    start: CrawlStart<'_>,
    node_budget: usize,
) -> Result<Option<GenGraph>, CrawlError> {
    if node_budget == 0 {
        return Err(CrawlError::EmptyBudget);
    }
    let (node_id, coordinate) = match start {
        CrawlStart::Node {
            node_id,
            coordinate,
        } => (node_id, coordinate),
        CrawlStart::Path { path, coordinate } => {
            if path.block_group_id != block_group_id {
                return Err(CrawlError::PathBlockGroupMismatch);
            }
            let Some(block) = path
                .coordinate_blocks(conn, None)
                .into_iter()
                .find(|block| block.path_start <= coordinate && coordinate < block.path_end)
            else {
                return Ok(None);
            };
            let offset = coordinate - block.path_start;
            let node_coordinate = if block.strand == Strand::Reverse {
                block.sequence_end - offset - 1
            } else {
                block.sequence_start + offset
            };
            (block.node_id, node_coordinate)
        }
    };

    let mut crawler = PortCrawler::new(block_group_id, false);
    let mut loaded = GenGraph::new();
    let Some(anchor) = crawler.locate(conn, &mut loaded, node_id, coordinate)? else {
        return Ok(None);
    };
    let mut selected = HashSet::from([anchor]);
    let mut queue = VecDeque::from([anchor]);
    while selected.len() < node_budget {
        let Some(node) = queue.pop_front() else {
            break;
        };
        crawler.complete(conn, &mut loaded, &[node])?;
        let mut neighbors: Vec<GraphNode> = [Direction::Incoming, Direction::Outgoing]
            .into_iter()
            .flat_map(|direction| loaded.neighbors_directed(node, direction))
            .collect();
        neighbors.sort();
        neighbors.dedup();
        for neighbor in neighbors {
            if selected.len() == node_budget {
                break;
            }
            if selected.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    // Completing the selected boundary also discovers edges joining two selected slices by a
    // route that was not traversed while choosing the members.
    let members: Vec<GraphNode> = selected.iter().copied().collect();
    crawler.complete(conn, &mut loaded, &members)?;
    let mut result = GenGraph::new();
    for node in &members {
        result.add_node(*node);
    }
    for (source, target, edges) in loaded.all_edges() {
        if selected.contains(&source) && selected.contains(&target) {
            result.add_edge(source, target, edges.clone());
        }
    }
    Ok(Some(result))
}

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

/// Lazily grows one block group's graph. Keep the crawler paired with the graph it describes.
#[derive(Clone, Debug)]
pub struct PortCrawler {
    block_group_id: HashId,
    prune: bool,
    port_blocks: HashMap<Port, Vec<GraphNode>>,
    /// Frontier edge batches are known before their far blocks have been opened and closed.
    edge_groups: HashMap<(Port, Direction), Arc<[AugmentedEdge]>>,
    /// An edge is projected once, whichever of its two ports reached it first.
    visited_edges: HashSet<HashId>,
    materialized: HashSet<(GraphNode, Direction)>,
}

/// A jump skips sequence within one backing node, so its endpoints sit at different coordinates.
fn is_same_node_jump(augmented_edge: &AugmentedEdge) -> bool {
    let edge = &augmented_edge.edge;
    edge.source_node_id == edge.target_node_id && edge.source_coordinate != edge.target_coordinate
}

/// Whether any edge at one side of a port is a distinct, non-marker edge between real nodes.
/// A port with such edges on both sides is a shared position.
fn is_shared_position_edge_set(edges: &[AugmentedEdge]) -> bool {
    edges.iter().any(|augmented_edge| {
        let edge = &augmented_edge.edge;
        !edge.is_same_coordinate_edge()
            && !is_terminal(edge.source_node_id)
            && !is_terminal(edge.target_node_id)
    })
}

/// Whether a port is a shared position: ordinary edges both arrive at and leave it.
fn is_shared_position(incoming: &[AugmentedEdge], outgoing: &[AugmentedEdge]) -> bool {
    is_shared_position_edge_set(incoming) && is_shared_position_edge_set(outgoing)
}

/// Both sides of a stored edge, resolved to the blocks adjacent to its two ports.
struct EdgeEndpoints<'a> {
    source_blocks: &'a [GraphNode],
    target_blocks: &'a [GraphNode],
    source_is_shared_position: bool,
    target_is_shared_position: bool,
}

/// Project one stored edge onto the blocks at its two ports, the same way the eager graph
/// builder does for ordinary edges. A junction is a port with a self loop, and every edge at it
/// is loaded with the port, so the filtered product of the blocks arriving at and leaving the
/// port can be decided locally. Unlike the eager builder, the crawler does not add cycle-escape
/// edges for traversal algorithms that cannot revisit junctions, as can happen around insertions
/// anchored to a junction. Detecting those shortcuts requires reachability analysis over the full
/// graph. Without them, the crawled neighborhood still represents the routes through its junctions
/// and is suitable for the viewer widgets.
fn merge_fragment(graph: &mut GenGraph, augmented_edge: &AugmentedEdge, endpoints: &EdgeEndpoints) {
    let edge = &augmented_edge.edge;
    let group_block = |node: &GraphNode| {
        GroupBlock::without_sequence(0, node.node_id, node.sequence_start, node.sequence_end)
    };
    let source_blocks: Vec<GroupBlock> = endpoints
        .source_blocks
        .iter()
        .filter(|node| {
            node.node_id == edge.source_node_id && node.sequence_end == edge.source_coordinate
        })
        .map(group_block)
        .collect();
    let target_blocks: Vec<GroupBlock> = endpoints
        .target_blocks
        .iter()
        .filter(|node| {
            node.node_id == edge.target_node_id && node.sequence_start == edge.target_coordinate
        })
        .map(group_block)
        .collect();
    let connections = edge.block_connections(
        &source_blocks.iter().collect::<Vec<_>>(),
        &target_blocks.iter().collect::<Vec<_>>(),
        endpoints.source_is_shared_position,
        endpoints.target_is_shared_position,
    );
    let graph_edge = GraphEdge {
        edge_id: edge.id,
        source_strand: edge.source_strand,
        target_strand: edge.target_strand,
        chromosome_index: augmented_edge.chromosome_index,
        phased: augmented_edge.phased,
        created_on: augmented_edge.created_on,
    };
    for (source_block, target_block) in connections {
        let source = GraphNode {
            node_id: source_block.node_id,
            sequence_start: source_block.start,
            sequence_end: source_block.end,
        };
        let target = GraphNode {
            node_id: target_block.node_id,
            sequence_start: target_block.start,
            sequence_end: target_block.end,
        };
        if let Some(existing) = graph.edge_weight_mut(source, target) {
            if !existing
                .iter()
                .any(|present| present.edge_id == graph_edge.edge_id)
            {
                existing.push(graph_edge);
            }
        } else {
            graph.add_edge(source, target, vec![graph_edge]);
        }
    }
}

impl PortCrawler {
    pub fn new(block_group_id: HashId, prune: bool) -> Self {
        Self {
            block_group_id,
            prune,
            port_blocks: HashMap::new(),
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

    /// The graph slice holding a backing-node coordinate, added as frontier if needed.
    /// The surrounding port coordinates define its sequence bounds; a coordinate on an outer
    /// junction with no sequence to its right resolves to the zero-width junction itself.
    pub fn locate(
        &mut self,
        conn: &GraphConnection,
        graph: &mut GenGraph,
        node_id: HashId,
        coordinate: i64,
    ) -> Result<Option<GraphNode>, EdgeError> {
        if is_terminal(node_id) {
            return Ok(None);
        }
        let Some(start) = Edge::nearest_port_coordinate(
            conn,
            &self.block_group_id,
            node_id,
            coordinate,
            Direction::Incoming,
            true,
        )?
        else {
            return Ok(None);
        };
        if let Some(end) = Edge::nearest_port_coordinate(
            conn,
            &self.block_group_id,
            node_id,
            start,
            Direction::Outgoing,
            false,
        )? && coordinate < end
        {
            let block = GraphNode {
                node_id,
                sequence_start: start,
                sequence_end: end,
            };
            graph.add_node(block);
            return Ok(Some(block));
        }
        if coordinate == start {
            let junction = GraphNode {
                node_id,
                sequence_start: start,
                sequence_end: start,
            };
            if self
                .port_blocks(
                    conn,
                    Port {
                        node_id,
                        coordinate: start,
                    },
                )?
                .contains(&junction)
            {
                graph.add_node(junction);
                return Ok(Some(junction));
            }
        }
        Ok(None)
    }

    fn load_group(
        &mut self,
        conn: &GraphConnection,
        port: Port,
        direction: Direction,
    ) -> Result<Arc<[AugmentedEdge]>, EdgeError> {
        if let Some(edges) = self.edge_groups.get(&(port, direction)) {
            return Ok(Arc::clone(edges));
        }
        let edges: Arc<[AugmentedEdge]> = Edge::edges_at_port_direction(
            conn,
            &self.block_group_id,
            port.node_id,
            port.coordinate,
            direction,
        )?
        .into();
        self.edge_groups
            .insert((port, direction), Arc::clone(&edges));
        Ok(edges)
    }

    /// The sequence blocks adjacent to a port, plus its generated junction when the eager
    /// builder would place one there. A junction needs both an incoming and outgoing same-node
    /// jump at an internal port, a source edge at the first port, or a target edge at the last.
    fn port_blocks(
        &mut self,
        conn: &GraphConnection,
        port: Port,
    ) -> Result<Vec<GraphNode>, EdgeError> {
        if let Some(blocks) = self.port_blocks.get(&port) {
            return Ok(blocks.clone());
        }
        if is_terminal(port.node_id) {
            let sentinel = vec![GraphNode {
                node_id: port.node_id,
                sequence_start: 0,
                sequence_end: 0,
            }];
            self.port_blocks.insert(port, sentinel.clone());
            return Ok(sentinel);
        }
        let (previous, next) = Edge::neighboring_port_coordinates(
            conn,
            &self.block_group_id,
            port.node_id,
            port.coordinate,
        )?;
        let incoming = self.load_group(conn, port, Direction::Incoming)?;
        let outgoing = self.load_group(conn, port, Direction::Outgoing)?;
        let incoming_jump = incoming.iter().any(is_same_node_jump);
        let outgoing_jump = outgoing.iter().any(is_same_node_jump);
        // Mirrors the eager builder's junction rule: a jump endpoint meeting any other edge, or
        // two distinct edges meeting at the port, needs a zero-width block to route through.
        let junction = (outgoing_jump && !incoming.is_empty())
            || (incoming_jump && !outgoing.is_empty())
            || is_shared_position(&incoming, &outgoing);
        let mut blocks = Vec::new();
        if let Some(previous) = previous {
            blocks.push(GraphNode {
                node_id: port.node_id,
                sequence_start: previous,
                sequence_end: port.coordinate,
            });
        }
        if let Some(next) = next {
            blocks.push(GraphNode {
                node_id: port.node_id,
                sequence_start: port.coordinate,
                sequence_end: next,
            });
        }
        if (previous.is_none() && next.is_none())
            || (previous.is_none() && !outgoing.is_empty())
            || (next.is_none() && !incoming.is_empty())
            || junction
        {
            blocks.push(GraphNode {
                node_id: port.node_id,
                sequence_start: port.coordinate,
                sequence_end: port.coordinate,
            });
        }
        self.port_blocks.insert(port, blocks.clone());
        Ok(blocks)
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
        // Every stored slice sits on a route between the path sentinels, so an empty one without
        // edges on this side means the block group's edges are malformed.
        let is_dangling = node.length() == 0 && !is_terminal(node.node_id) && edges.is_empty();
        debug_assert!(
            !is_dangling,
            "dangling graph slice in block group {:?}: node {:?} has no {:?} edges",
            self.block_group_id, node, direction
        );
        for augmented_edge in edges.iter() {
            let edge_id = augmented_edge.edge.id;
            if self.visited_edges.contains(&edge_id) {
                continue;
            }
            if self.prune && !self.survives_pruning(conn, augmented_edge)? {
                self.visited_edges.insert(edge_id);
                continue;
            }
            let source_port = Port {
                node_id: augmented_edge.edge.source_node_id,
                coordinate: augmented_edge.edge.source_coordinate,
            };
            let target_port = Port {
                node_id: augmented_edge.edge.target_node_id,
                coordinate: augmented_edge.edge.target_coordinate,
            };
            let source_blocks = self.port_blocks(conn, source_port)?;
            let target_blocks = self.port_blocks(conn, target_port)?;
            let source_is_shared_position = self.is_shared_position_port(conn, source_port)?;
            let target_is_shared_position = self.is_shared_position_port(conn, target_port)?;
            merge_fragment(
                graph,
                augmented_edge,
                &EdgeEndpoints {
                    source_blocks: &source_blocks,
                    target_blocks: &target_blocks,
                    source_is_shared_position,
                    target_is_shared_position,
                },
            );
            self.visited_edges.insert(edge_id);
        }
        self.materialized.insert((node, direction));
        Ok(())
    }

    fn is_shared_position_port(
        &mut self,
        conn: &GraphConnection,
        port: Port,
    ) -> Result<bool, EdgeError> {
        let incoming = self.load_group(conn, port, Direction::Incoming)?;
        let outgoing = self.load_group(conn, port, Direction::Outgoing)?;
        Ok(is_shared_position(&incoming, &outgoing))
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
        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, is_terminal,
    };
    use gen_graph::{GenGraph, GraphNode};
    use petgraph::Direction;

    use super::{CrawlError, CrawlStart, PortCrawler, crawl_graph};
    use crate::{
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        node::Node,
        path::Path,
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
    fn test_crawl_graph_respects_node_budget_around_a_node_position() {
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
        let start = CrawlStart::Node {
            node_id: node_id_for("b"),
            coordinate: 2,
        };
        let graph = crawl_graph(&conn, block_group_id, start, 2)
            .unwrap()
            .expect("should find the requested node position");

        assert_eq!(graph.node_count(), 2);
        assert!(graph.contains_node(block("b", 0, 4)));
        assert!(
            graph
                .all_edges()
                .all(|(source, target, _)| graph.contains_node(source)
                    && graph.contains_node(target))
        );
        assert!(matches!(
            crawl_graph(
                &conn,
                block_group_id,
                CrawlStart::Node {
                    node_id: node_id_for("b"),
                    coordinate: 2,
                },
                0,
            ),
            Err(CrawlError::EmptyBudget)
        ));
    }

    #[test]
    fn test_crawl_graph_starts_at_a_path_position() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA"), ("b", "CCCC")],
            &[
                ("start", 0, "a", 0, 0),
                ("a", 4, "b", 0, 0),
                ("b", 4, "end", 0, 0),
            ],
        );
        let edge_ids = [
            (PATH_START_NODE_ID, 0),
            (node_id_for("a"), 4),
            (node_id_for("b"), 4),
        ]
        .map(|(node_id, coordinate)| {
            Edge::edges_at_port_direction(
                &conn,
                &block_group_id,
                node_id,
                coordinate,
                Direction::Outgoing,
            )
            .unwrap()[0]
                .edge
                .id
        });
        let path = Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let graph = crawl_graph(
            &conn,
            block_group_id,
            CrawlStart::Path {
                path: &path,
                coordinate: 5,
            },
            1,
        )
        .unwrap()
        .expect("should find a block at path coordinate five");

        assert_eq!(graph.node_count(), 1);
        assert!(graph.contains_node(block("b", 0, 4)));
    }

    #[test]
    fn test_crawl_graph_keeps_main_junctions_within_budget() {
        let conn = get_connection(None).unwrap();
        let block_group_id = junction_heavy_block_group(&conn);
        let eager = BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
        let graph = crawl_graph(
            &conn,
            block_group_id,
            CrawlStart::Node {
                node_id: node_id_for("ref"),
                coordinate: 9,
            },
            6,
        )
        .unwrap()
        .expect("should locate a reference slice beside a junction");

        assert!(graph.node_count() <= 6);
        assert!(graph.nodes().all(|node| eager.contains_node(node)));
        assert!(graph.nodes().any(|node| {
            node.node_id == node_id_for("ref") && node.sequence_start == node.sequence_end
        }));
        for (source, target, edges) in graph.all_edges() {
            let expected = eager
                .edge_weight(source, target)
                .expect("should preserve an eager graph edge");
            assert!(edges.iter().all(|edge| expected.contains(edge)));
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
            let complete = graph
                .nodes()
                .filter(|node| crawler.is_complete(node, direction))
                .count();
            assert_eq!(complete, 3, "the anchor plus the two budgeted blocks");
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
    #[cfg(debug_assertions)]
    #[should_panic(expected = "dangling graph slice")]
    fn test_forward_crawl_panics_on_a_dangling_empty_slice_in_debug_builds() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA"), ("b", "CCCC")],
            &[("start", 0, "a", 0, 0), ("a", 4, "b", 4, 0)],
        );
        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();
        graph.add_node(start_sentinel());
        crawl_to_exhaustion(&conn, &mut crawler, &mut graph);
    }

    #[test]
    fn test_converging_branches_project_each_edge_once_from_either_direction() {
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
            6,
            "each of the six edges should be projected once, from whichever side reached it"
        );
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

    /// The crawl has the eager graph's nodes and no edge the eager graph lacks. Each eager-only
    /// edge must be a loop-exit shortcut over a junction, whose walk the crawl carries through
    /// that junction.
    fn assert_matches_up_to_junction_shortcuts(lazy: &GenGraph, eager: &GenGraph) {
        assert_eq!(connected_shape(lazy).0, connected_shape(eager).0);
        for (source, target, _) in lazy.all_edges() {
            assert!(
                eager.contains_edge(source, target),
                "the crawl drew an edge the eager graph lacks: {source:?} -> {target:?}"
            );
        }
        for (source, target, _) in eager.all_edges() {
            let is_shortcut =
                lazy.neighbors_directed(source, Direction::Outgoing)
                    .any(|junction| {
                        !is_terminal(junction.node_id)
                            && junction.sequence_start == junction.sequence_end
                            && lazy.contains_edge(junction, target)
                    });
            assert!(
                lazy.contains_edge(source, target) || is_shortcut,
                "the crawl lacks {source:?} -> {target:?} and no junction carries the same walk"
            );
        }
    }

    /// A reference with an insertion whose end rejoins the reference, plus the same-coordinate
    /// edges that keep the reference continuous across the outer ports and the split.
    fn inserted_reference_block_group(conn: &GraphConnection) -> HashId {
        setup_block_group(
            conn,
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
        )
    }

    #[test]
    fn test_exhaustive_crawl_matches_the_eager_graph_from_either_end() {
        let fixtures: [fn(&GraphConnection) -> HashId; 3] = [
            junction_heavy_block_group,
            inserted_reference_block_group,
            |conn| {
                setup_block_group(
                    conn,
                    &[("a", "AAAA"), ("b", "CCCC")],
                    &[
                        ("start", 0, "a", 0, 0),
                        ("a", 4, "b", 4, 0),
                        ("b", 4, "end", 0, 0),
                    ],
                )
            },
        ];
        for setup in fixtures {
            let conn = get_connection(None).unwrap();
            let block_group_id = setup(&conn);
            let eager =
                BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
            for anchor in [start_sentinel(), block("end", 0, 0)] {
                let mut crawler = PortCrawler::new(block_group_id, false);
                let mut graph = GenGraph::new();
                graph.add_node(anchor);
                crawl_to_exhaustion(&conn, &mut crawler, &mut graph);
                assert_matches_up_to_junction_shortcuts(&graph, &eager);
            }
        }
    }

    /// Every coordinate of every block a full crawl carves locates to that block from a fresh
    /// crawler, and crawling on from the located block alone rebuilds the same graph, so a
    /// viewer that jumps somewhere first never carves a node differently.
    #[test]
    fn test_locate_finds_the_block_a_crawl_carves() {
        let fixtures: [fn(&GraphConnection) -> HashId; 2] =
            [junction_heavy_block_group, inserted_reference_block_group];
        for setup in fixtures {
            let conn = get_connection(None).unwrap();
            let block_group_id = setup(&conn);
            let mut crawler = PortCrawler::new(block_group_id, false);
            let mut crawled = GenGraph::new();
            crawled.add_node(start_sentinel());
            crawl_to_exhaustion(&conn, &mut crawler, &mut crawled);
            let expected = connected_shape(&crawled);
            for block in expected.0.iter().filter(|block| {
                !is_terminal(block.node_id) && block.sequence_end > block.sequence_start
            }) {
                for coordinate in block.sequence_start..block.sequence_end {
                    let mut crawler = PortCrawler::new(block_group_id, false);
                    let mut graph = GenGraph::new();
                    let located = crawler
                        .locate(&conn, &mut graph, block.node_id, coordinate)
                        .unwrap();
                    assert_eq!(located, Some(*block), "coordinate {coordinate}");
                    crawl_to_exhaustion(&conn, &mut crawler, &mut graph);
                    let (nodes, edges) = connected_shape(&graph);
                    assert!(
                        nodes == expected.0 && edges == expected.1,
                        "crawling on from {block:?} at {coordinate}: extra nodes {:?}, missing \
                         nodes {:?}, extra edges {:?}, missing edges {:?}",
                        nodes.difference(&expected.0).collect::<Vec<_>>(),
                        expected.0.difference(&nodes).collect::<Vec<_>>(),
                        edges.difference(&expected.1).collect::<Vec<_>>(),
                        expected.1.difference(&edges).collect::<Vec<_>>(),
                    );
                }
            }
        }
    }

    #[test]
    fn test_locate_finds_nothing_off_the_block_group() {
        let conn = get_connection(None).unwrap();
        let block_group_id = junction_heavy_block_group(&conn);
        let mut crawler = PortCrawler::new(block_group_id, false);
        let mut graph = GenGraph::new();

        assert_eq!(
            crawler
                .locate(&conn, &mut graph, node_id_for("unrelated"), 2)
                .unwrap(),
            None
        );
        assert_eq!(
            crawler
                .locate(&conn, &mut graph, PATH_START_NODE_ID, 0)
                .unwrap(),
            None
        );
        assert_eq!(graph.node_count(), 0);
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
                ("new", 5, "end", 0, 0),
                ("tail", 5, "end", 0, 0),
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
