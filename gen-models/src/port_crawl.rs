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
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX, NodeIntervalBlock,
    PATH_START_NODE_ID, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, is_terminal,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use intervaltree::IntervalTree;
use petgraph::Direction;
use thiserror::Error;

use crate::{
    accession::AccessionError,
    block_group_edge::AugmentedEdge,
    db::GraphConnection,
    edge::{Edge, EdgeError, GroupBlock},
    errors::PathError,
    region::ResolvedGenRegion,
};

/// Where a crawl begins.
pub enum CrawlStart<'a> {
    /// A coordinate in the backing node's stored sequence.
    Node { node_id: HashId, coordinate: i64 },
    /// The midpoint of a resolved region, which may name a path, block group, annotation or
    /// accession. Coordinates count bases along the region's path, or along its accession for
    /// annotations and accessions, including reverse blocks. To keep the crawl within the
    /// region, pass half its length as the radius.
    Region(&'a ResolvedGenRegion),
}

/// Why a backing-node position could not be turned into a block of the crawled graph.
#[derive(Debug, Error)]
pub enum LocateError {
    /// The position does not belong to the block group.
    #[error("the position is not in the block group")]
    NotFound,
    /// Edits cut the position off from the start of the block group. A pruned view omits it, as
    /// `BlockGroup::prune_graph` does, so there is no block to show.
    #[error("the position is cut off from the start of the block group by later edits")]
    Unreachable,
    /// A database query failed.
    #[error(transparent)]
    Edge(#[from] EdgeError),
    /// The graph source could not be used, for example because its connection was lost.
    #[error("the graph source is unavailable: {0}")]
    Unavailable(String),
}

#[derive(Debug, Error)]
pub enum CrawlError {
    /// A database query or graph construction step failed.
    #[error(transparent)]
    Edge(#[from] EdgeError),
    /// The returned graph must be able to contain its starting slice.
    #[error("the crawl budget must contain at least one graph node")]
    EmptyBudget,
    /// The requested region does not belong to the requested block group.
    #[error("the starting region belongs to another block group")]
    RegionBlockGroupMismatch,
    /// Path and block group regions are located through a path, annotation and accession regions
    /// through their accession.
    #[error("the starting region has neither a path nor an accession")]
    RegionWithoutBlocks,
    /// The path behind a path or block group region could not be loaded.
    #[error(transparent)]
    Path(#[from] PathError),
    /// The accession behind an annotation or accession region could not be loaded.
    #[error(transparent)]
    Accession(#[from] AccessionError),
}

/// Blocks laid out in the coordinates of `region.start` and `region.end`: the path's for path and
/// block group regions, the accession's for annotation and accession regions, which have no path.
/// A block group region is read through its current path rather than the block group's own tree,
/// which would load the whole graph.
fn region_blocks(
    conn: &GraphConnection,
    region: &ResolvedGenRegion,
) -> Result<Arc<IntervalTree<i64, NodeIntervalBlock>>, CrawlError> {
    if let Some(path) = &region.path {
        return Ok(path.intervaltree(conn)?);
    }
    let accession = region
        .accession
        .as_ref()
        .ok_or(CrawlError::RegionWithoutBlocks)?;
    Ok(accession.intervaltree(conn)?)
}

/// Return at most `node_budget` graph slices around a position, with all edges between them.
/// A position outside the block group returns `None`. Node coordinates refer to the backing
/// node's sequence; region coordinates count bases along its path. The crawl may temporarily
/// discover adjacent frontier nodes while resolving the selected slices, but only selected
/// slices and their internal edges appear in the returned graph.
///
/// With a `radius`, each branch stops once a slice begins that many bases from the start,
/// measured by the bases walked to reach it. Use it when the full sequence is on screen, or
/// around a hub node that spans the genome and would otherwise fill the budget with distant
/// slices. Insertions use up distance and deletions skip it. Turning around at a port is free,
/// which keeps the other legs of a bubble the same distance away as the leg holding the start.
/// A slice that straddles the limit is included whole. Without a radius the crawl follows
/// ports regardless of how much sequence they skip.
pub fn crawl_graph(
    conn: &GraphConnection,
    block_group_id: HashId,
    start: CrawlStart<'_>,
    node_budget: usize,
    radius: Option<i64>,
) -> Result<Option<GenGraph>, CrawlError> {
    if node_budget == 0 {
        return Err(CrawlError::EmptyBudget);
    }
    let (node_id, coordinate) = match start {
        CrawlStart::Node {
            node_id,
            coordinate,
        } => (node_id, coordinate),
        CrawlStart::Region(region) => {
            if region.block_group.id != block_group_id {
                return Err(CrawlError::RegionBlockGroupMismatch);
            }
            let region_coordinate = region.start + (region.end - region.start) / 2;
            let tree = region_blocks(conn, region)?;
            let Some(block) = tree
                .query_point(region_coordinate)
                .map(|element| &element.value)
                .find(|block| !is_terminal(block.node_id))
            else {
                return Ok(None);
            };
            let offset = region_coordinate - block.start;
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
    let anchor = match crawler.locate(conn, &mut loaded, node_id, coordinate) {
        Ok(anchor) => anchor,
        Err(LocateError::NotFound | LocateError::Unreachable | LocateError::Unavailable(_)) => {
            return Ok(None);
        }
        Err(LocateError::Edge(error)) => return Err(error.into()),
    };
    let mut selected = HashSet::from([anchor]);
    // Bases from the start to each slice's incoming and outgoing port, tracked only when the
    // crawl is bounded. The start sits inside the anchor, so its own offsets seed the walk.
    let mut distances: HashMap<GraphNode, PortDistances> = HashMap::from([(
        anchor,
        PortDistances {
            incoming: coordinate - anchor.sequence_start,
            outgoing: anchor.sequence_end - coordinate,
        },
    )]);
    let mut queue = VecDeque::from([anchor]);
    while selected.len() < node_budget {
        let Some(node) = queue.pop_front() else {
            break;
        };
        crawler.complete(conn, &mut loaded, &[node])?;
        let node_distances = distances.get(&node).copied().unwrap_or(PortDistances {
            incoming: 0,
            outgoing: 0,
        });
        let mut neighbors: Vec<(GraphNode, Direction)> = [Direction::Incoming, Direction::Outgoing]
            .into_iter()
            .filter(|direction| radius.is_none_or(|radius| node_distances.at(*direction) < radius))
            .flat_map(|direction| {
                loaded
                    .neighbors_directed(node, direction)
                    .map(move |neighbor| (neighbor, direction))
            })
            .collect();
        neighbors.sort();
        neighbors.dedup();
        for (neighbor, direction) in neighbors {
            if radius.is_some() {
                let entry = node_distances.at(direction);
                let length = neighbor.sequence_end - neighbor.sequence_start;
                let reached = match direction {
                    Direction::Outgoing => PortDistances {
                        incoming: entry,
                        outgoing: entry + length,
                    },
                    Direction::Incoming => PortDistances {
                        incoming: entry + length,
                        outgoing: entry,
                    },
                };
                let known = distances.entry(neighbor).or_insert(reached);
                let improved = known.merge(reached);
                if improved && selected.contains(&neighbor) && !queue.contains(&neighbor) {
                    queue.push_back(neighbor);
                }
            }
            if selected.len() < node_budget && selected.insert(neighbor) {
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

#[derive(Clone, Copy, Debug)]
struct PortDistances {
    incoming: i64,
    outgoing: i64,
}

impl PortDistances {
    fn at(&self, direction: Direction) -> i64 {
        match direction {
            Direction::Incoming => self.incoming,
            Direction::Outgoing => self.outgoing,
        }
    }

    /// Keep the shorter distance at each port; returns whether either shrank.
    fn merge(&mut self, other: Self) -> bool {
        let improved = other.incoming < self.incoming || other.outgoing < self.outgoing;
        self.incoming = self.incoming.min(other.incoming);
        self.outgoing = self.outgoing.min(other.outgoing);
        improved
    }
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
    /// Each chromosome record is projected once, whichever port reached it first. A stored
    /// edge ID alone is insufficient because several chromosome copies can share that edge.
    visited_edge_records: HashSet<GraphEdge>,
    materialized: HashSet<(GraphNode, Direction)>,
    /// Blocks known to be reached from a start sentinel by routes that survive pruning. A
    /// backward step stops its reachability walk at any of them, so checking a predecessor
    /// costs a few queries instead of a walk back to the start. It lives as long as the graph
    /// this crawler grows, so it is never staler than that graph.
    proven_live: HashSet<GraphNode>,
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

fn graph_edge_for(augmented_edge: &AugmentedEdge) -> GraphEdge {
    GraphEdge {
        edge_id: augmented_edge.edge.id,
        source_strand: augmented_edge.edge.source_strand,
        target_strand: augmented_edge.edge.target_strand,
        chromosome_index: augmented_edge.chromosome_index,
        phased: augmented_edge.phased,
        created_on: augmented_edge.created_on,
    }
}

/// The block-to-block links one stored edge produces at its two ports, the same way the eager
/// graph builder does for ordinary edges. A junction is a port with a self loop, and every edge at
/// it is loaded with the port, so the filtered product of the blocks arriving at and leaving the
/// port can be decided locally. Both `merge_fragment`, which draws the links, and
/// `PortCrawler::is_reachable`, which only follows them, call this so they can never disagree
/// about which blocks an edge joins.
fn connected_blocks(
    augmented_edge: &AugmentedEdge,
    endpoints: &EdgeEndpoints,
) -> Vec<(GraphNode, GraphNode)> {
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
    edge.block_connections(
        &source_blocks.iter().collect::<Vec<_>>(),
        &target_blocks.iter().collect::<Vec<_>>(),
        endpoints.source_is_shared_position,
        endpoints.target_is_shared_position,
    )
    .into_iter()
    .map(|(source_block, target_block)| {
        (
            GraphNode {
                node_id: source_block.node_id,
                sequence_start: source_block.start,
                sequence_end: source_block.end,
            },
            GraphNode {
                node_id: target_block.node_id,
                sequence_start: target_block.start,
                sequence_end: target_block.end,
            },
        )
    })
    .collect()
}

/// Project one stored edge onto the blocks at its two ports (see `connected_blocks`). Unlike the
/// eager builder, the crawler does not add cycle-escape edges for traversal algorithms that
/// cannot revisit junctions, as can happen around insertions anchored to a junction. Detecting
/// those shortcuts requires reachability analysis over the full graph. Without them, the crawled
/// neighborhood still represents the routes through its junctions and is suitable for the viewer
/// widgets.
fn merge_fragment(
    graph: &mut GenGraph,
    augmented_edge: &AugmentedEdge,
    connections: &[(GraphNode, GraphNode)],
) {
    let graph_edge = graph_edge_for(augmented_edge);
    for (source, target) in connections.iter().copied() {
        if let Some(existing) = graph.edge_weight_mut(source, target) {
            if !existing.contains(&graph_edge) {
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
            visited_edge_records: HashSet::new(),
            materialized: HashSet::new(),
            proven_live: HashSet::new(),
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
    /// With pruning on, a slice that no surviving route reaches from a start sentinel is refused,
    /// since a pruned view has no such block.
    pub fn locate(
        &mut self,
        conn: &GraphConnection,
        graph: &mut GenGraph,
        node_id: HashId,
        coordinate: i64,
    ) -> Result<GraphNode, LocateError> {
        let block = self
            .find_block(conn, node_id, coordinate)?
            .ok_or(LocateError::NotFound)?;
        if self.prune && !self.is_reachable(conn, block)? {
            return Err(LocateError::Unreachable);
        }
        graph.add_node(block);
        Ok(block)
    }

    /// The graph slice holding a backing-node coordinate, if the block group has one there.
    /// The surrounding port coordinates define its sequence bounds; a coordinate on an outer
    /// junction with no sequence to its right resolves to the zero-width junction itself.
    fn find_block(
        &mut self,
        conn: &GraphConnection,
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
            let graph_edge = graph_edge_for(augmented_edge);
            if self.visited_edge_records.contains(&graph_edge) {
                continue;
            }
            let connections = self.surviving_connections(conn, augmented_edge)?;
            let connections = self.live_connections(conn, connections)?;
            merge_fragment(graph, augmented_edge, &connections);
            self.visited_edge_records.insert(graph_edge);
        }
        self.materialized.insert((node, direction));
        Ok(())
    }

    /// The block-to-block links a stored edge produces, resolved against the blocks adjacent to
    /// both of its ports, plus the cycle-exit link the eager builder adds for an insertion.
    ///
    /// An insertion leaves a port and returns to it, so its returning edge can only reach the
    /// junction at that port. The reference continuation past the junction is carried by the cut
    /// marker, which pruning drops, so the returning edge also links its source straight to the
    /// sequence starting at the port, and keeps its own records.
    fn connections_of(
        &mut self,
        conn: &GraphConnection,
        augmented_edge: &AugmentedEdge,
    ) -> Result<Vec<(GraphNode, GraphNode)>, EdgeError> {
        let mut connections = self.edge_links(conn, augmented_edge)?;
        let edge = &augmented_edge.edge;
        if edge.is_same_coordinate_edge() {
            return Ok(connections);
        }
        let target_port = Port::endpoint(edge, Direction::Incoming);
        if !self.is_shared_position_port(conn, target_port)? {
            return Ok(connections);
        }
        let target_sequences: Vec<GraphNode> = self
            .port_blocks(conn, target_port)?
            .into_iter()
            .filter(|block| {
                block.node_id == edge.target_node_id
                    && block.sequence_start == edge.target_coordinate
                    && block.sequence_start != block.sequence_end
            })
            .collect();
        let entries: Vec<(GraphNode, GraphNode)> = connections
            .iter()
            .filter(|(_, target)| target.sequence_start == target.sequence_end)
            .copied()
            .collect();
        for (source, position) in entries {
            if !self.leads_back_to(conn, source, position)? {
                continue;
            }
            for target in &target_sequences {
                if !connections.contains(&(source, *target)) {
                    connections.push((source, *target));
                }
            }
        }
        Ok(connections)
    }

    /// The links one stored edge produces between the blocks at its two ports.
    fn edge_links(
        &mut self,
        conn: &GraphConnection,
        augmented_edge: &AugmentedEdge,
    ) -> Result<Vec<(GraphNode, GraphNode)>, EdgeError> {
        let source_port = Port::endpoint(&augmented_edge.edge, Direction::Outgoing);
        let target_port = Port::endpoint(&augmented_edge.edge, Direction::Incoming);
        let source_blocks = self.port_blocks(conn, source_port)?;
        let target_blocks = self.port_blocks(conn, target_port)?;
        let source_is_shared_position = self.is_shared_position_port(conn, source_port)?;
        let target_is_shared_position = self.is_shared_position_port(conn, target_port)?;
        Ok(connected_blocks(
            augmented_edge,
            &EdgeEndpoints {
                source_blocks: &source_blocks,
                target_blocks: &target_blocks,
                source_is_shared_position,
                target_is_shared_position,
            },
        ))
    }

    /// Whether `position` can reach `block` through the stored edges, which makes an edge from
    /// `block` into `position` a cycle. The walk goes backward from `block` over every stored
    /// edge, pruned or not, and across the boundaries where other edits split an inserted node.
    /// It never enters the backing node of `position`: the reference side is not part of a
    /// return path, and stopping there keeps the walk inside the inserted sequence.
    fn leads_back_to(
        &mut self,
        conn: &GraphConnection,
        block: GraphNode,
        position: GraphNode,
    ) -> Result<bool, EdgeError> {
        let mut pending = vec![block];
        let mut seen = HashSet::new();
        while let Some(current) = pending.pop() {
            if current.node_id == position.node_id || !seen.insert(current) {
                continue;
            }
            let start_port = Port::of(&current, Direction::Incoming);
            let arrivals = self.load_group(conn, start_port, Direction::Incoming)?;
            for augmented_edge in arrivals.iter() {
                for (source, target) in self.edge_links(conn, augmented_edge)? {
                    if target != current {
                        continue;
                    }
                    if source == position {
                        return Ok(true);
                    }
                    pending.push(source);
                }
            }
            for neighbor in self.port_blocks(conn, start_port)? {
                if neighbor.node_id == current.node_id
                    && neighbor.sequence_end == current.sequence_start
                    && neighbor.sequence_start != neighbor.sequence_end
                {
                    pending.push(neighbor);
                }
            }
        }
        Ok(false)
    }

    /// Whether a route that survives pruning leads from a start sentinel to `block`.
    ///
    /// The walk goes backward only, one port at a time: at each block it reads the edges
    /// arriving at that block's own incoming port, skips the ones pruning drops, and moves to
    /// the blocks they leave from. It never reads a whole backing node. A cut-off neighbourhood
    /// runs out of edges and answers `false`; a live block ends the walk at a start sentinel or
    /// at a block already proven live, and every block on that route is proven live too.
    /// Nothing is added to a graph, so a caller can ask before it materializes a block, whether
    /// that block comes from `locate` or is the source of a batch.
    pub fn is_reachable(
        &mut self,
        conn: &GraphConnection,
        block: GraphNode,
    ) -> Result<bool, EdgeError> {
        let mut pending = vec![block];
        let mut seen = HashSet::new();
        let mut reached_from: HashMap<GraphNode, GraphNode> = HashMap::new();
        while let Some(current) = pending.pop() {
            if current.node_id == PATH_START_NODE_ID || self.proven_live.contains(&current) {
                self.proven_live.insert(current);
                let mut route = current;
                while let Some(downstream) = reached_from.get(&route).copied() {
                    self.proven_live.insert(downstream);
                    route = downstream;
                }
                return Ok(true);
            }
            if !seen.insert(current) {
                continue;
            }
            let arriving = self.load_group(
                conn,
                Port::of(&current, Direction::Incoming),
                Direction::Incoming,
            )?;
            for augmented_edge in arriving.iter() {
                for (source, target) in self.surviving_connections(conn, augmented_edge)? {
                    if target == current {
                        // The starting block never gets a downstream, or a loop back to it would
                        // make the route marked below circular.
                        if source != block {
                            reached_from.entry(source).or_insert(current);
                        }
                        pending.push(source);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Narrow `connections` to the links whose source a surviving route reaches from a start
    /// sentinel. An edge read at one block's port also binds to every other block sharing that
    /// port, and a source nothing reaches is a dead block that pruning leaves out even though its
    /// own edge into a live block survives, as beside adjacent deletions or a superseded allele.
    /// The target of a kept link is live, so later steps stop their walk there.
    fn live_connections(
        &mut self,
        conn: &GraphConnection,
        connections: Vec<(GraphNode, GraphNode)>,
    ) -> Result<Vec<(GraphNode, GraphNode)>, EdgeError> {
        if !self.prune {
            return Ok(connections);
        }
        let mut live = Vec::new();
        for (source, target) in connections {
            if self.is_reachable(conn, source)? {
                self.proven_live.insert(target);
                live.push((source, target));
            }
        }
        Ok(live)
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

    /// The links of `connections_of` that pruning keeps. Like `BlockGroup::prune_graph`, which
    /// walks graph nodes, siblings compete per source block and chromosome index, not per port:
    /// a same-coordinate edge into a junction and the edge leaving that junction share a port
    /// but leave different blocks, so neither supersedes the other. Reverse pruning needs the
    /// source port's outgoing siblings; forward crawling already cached them.
    fn surviving_connections(
        &mut self,
        conn: &GraphConnection,
        augmented_edge: &AugmentedEdge,
    ) -> Result<Vec<(GraphNode, GraphNode)>, EdgeError> {
        let connections = self.connections_of(conn, augmented_edge)?;
        if !self.prune {
            return Ok(connections);
        }
        let chromosome_index = augmented_edge.chromosome_index;
        if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
            return Ok(Vec::new());
        }
        if chromosome_index == NO_CHROMOSOME_INDEX
            || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
        {
            return Ok(connections);
        }
        let siblings = self.load_group(
            conn,
            Port::endpoint(&augmented_edge.edge, Direction::Outgoing),
            Direction::Outgoing,
        )?;
        let mut newest_by_source: HashMap<GraphNode, (i64, HashId)> = HashMap::new();
        for sibling in siblings
            .iter()
            .filter(|sibling| sibling.chromosome_index == chromosome_index)
        {
            let rank = (sibling.created_on, sibling.edge.id);
            for (source, _) in self.connections_of(conn, sibling)? {
                newest_by_source
                    .entry(source)
                    .and_modify(|newest| *newest = (*newest).max(rank))
                    .or_insert(rank);
            }
        }
        let own_rank = (augmented_edge.created_on, augmented_edge.edge.id);
        Ok(connections
            .into_iter()
            .filter(|(source, _)| newest_by_source.get(source) == Some(&own_rank))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use gen_core::{
        HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID,
        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, is_terminal, range::Range,
    };
    use gen_graph::{GenGraph, GraphNode};
    use petgraph::Direction;

    use super::{CrawlError, CrawlStart, LocateError, PortCrawler, crawl_graph};
    use crate::{
        accession::{Accession, AccessionSpan, NewAccession},
        annotations::Annotation,
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        node::Node,
        path::Path,
        region::{Region, resolve},
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
        let graph = crawl_graph(&conn, block_group_id, start, 2, None)
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
                None,
            ),
            Err(CrawlError::EmptyBudget)
        ));
    }

    #[test]
    fn test_crawl_preserves_each_chromosome_record_on_a_shared_stored_edge() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA"), ("b", "CCCC"), ("c", "GGGG")],
            &[
                ("start", 0, "a", 0, 0),
                ("a", 4, "b", 0, 0),
                ("a", 4, "b", 0, 1),
                ("a", 4, "c", 0, 0),
                ("b", 4, "end", 0, 0),
                ("c", 4, "end", 0, 0),
            ],
        );

        for prune in [false, true] {
            let mut eager =
                BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
            if prune {
                BlockGroup::prune_graph(&mut eager);
            }
            let mut crawler = PortCrawler::new(block_group_id, prune);
            let mut lazy = GenGraph::new();
            lazy.add_node(start_sentinel());
            crawl_to_exhaustion(&conn, &mut crawler, &mut lazy);

            for (source, target, expected) in eager.all_edges() {
                let mut expected = expected.clone();
                expected.sort();
                let mut actual = lazy
                    .edge_weight(source, target)
                    .expect("should retain a connection with a surviving chromosome record")
                    .clone();
                actual.sort();
                assert_eq!(
                    actual, expected,
                    "chromosome records should match with prune={prune}"
                );
            }
        }
    }

    #[test]
    fn test_crawl_graph_starts_at_a_region_position() {
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
        Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let region = resolve(&Region::parse("chain:5").unwrap(), &conn, "test", "test").unwrap();
        let graph = crawl_graph(&conn, block_group_id, CrawlStart::Region(&region), 1, None)
            .unwrap()
            .expect("should find a block at path coordinate five");

        assert_eq!(graph.node_count(), 1);
        assert!(graph.contains_node(block("b", 0, 4)));
    }

    #[test]
    fn test_crawl_graph_radius_stops_walking_past_the_requested_distance() {
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
        let start = || CrawlStart::Node {
            node_id: node_id_for("b"),
            coordinate: 2,
        };
        let unlimited = crawl_graph(&conn, block_group_id, start(), usize::MAX, None)
            .unwrap()
            .expect("should find the requested node position");
        assert!(unlimited.contains_node(block("a", 0, 4)));
        assert!(unlimited.contains_node(block("c", 0, 4)));

        // Two bases remain on either side of the start inside `b`, so a span of three reaches
        // the neighboring slices without walking on to their own neighbors.
        let spanned = crawl_graph(&conn, block_group_id, start(), usize::MAX, Some(3))
            .unwrap()
            .expect("should find the requested node position");
        assert!(spanned.contains_node(block("b", 0, 4)));
        assert!(spanned.contains_node(block("a", 0, 4)));
        assert!(spanned.contains_node(block("c", 0, 4)));
        assert!(!spanned.contains_node(start_sentinel()));
        assert!(!spanned.contains_node(block("end", 0, 0)));
    }

    #[test]
    fn test_crawl_graph_radius_keeps_the_other_leg_of_a_bubble() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA"), ("b", "CCCC"), ("c", "GGGG"), ("d", "TTTT")],
            &[
                ("start", 0, "a", 0, 0),
                ("a", 4, "b", 0, 0),
                ("a", 4, "c", 0, 0),
                ("b", 4, "d", 0, 0),
                ("c", 4, "d", 0, 0),
                ("d", 4, "end", 0, 0),
            ],
        );
        let graph = crawl_graph(
            &conn,
            block_group_id,
            CrawlStart::Node {
                node_id: node_id_for("b"),
                coordinate: 2,
            },
            usize::MAX,
            Some(3),
        )
        .unwrap()
        .expect("should find the requested node position");

        // The other leg hangs off the same ports as the start's leg, so it is as close as the
        // leg that holds the start even though it is only reachable by turning around.
        assert!(graph.contains_node(block("b", 0, 4)));
        assert!(graph.contains_node(block("c", 0, 4)));
        assert!(graph.contains_node(block("a", 0, 4)));
        assert!(graph.contains_node(block("d", 0, 4)));
        assert!(!graph.contains_node(start_sentinel()));
        assert!(!graph.contains_node(block("end", 0, 0)));
    }

    #[test]
    fn test_crawl_graph_region_start_uses_its_midpoint() {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA"), ("b", "CCCC"), ("c", "GGGG"), ("d", "TTTT")],
            &[
                ("start", 0, "a", 0, 0),
                ("a", 4, "b", 0, 0),
                ("b", 4, "c", 0, 0),
                ("c", 4, "d", 0, 0),
                ("d", 4, "end", 0, 0),
            ],
        );
        let edge_ids = [
            (PATH_START_NODE_ID, 0),
            (node_id_for("a"), 4),
            (node_id_for("b"), 4),
            (node_id_for("c"), 4),
            (node_id_for("d"), 4),
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
        Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let region = resolve(&Region::parse("chain:5-11").unwrap(), &conn, "test", "test").unwrap();
        let graph = crawl_graph(
            &conn,
            block_group_id,
            CrawlStart::Region(&region),
            usize::MAX,
            Some(3),
        )
        .unwrap()
        .expect("should find the region midpoint");

        assert!(graph.contains_node(block("b", 0, 4)));
        assert!(graph.contains_node(block("c", 0, 4)));
        assert!(!graph.contains_node(block("a", 0, 4)));
        assert!(!graph.contains_node(block("d", 0, 4)));
    }

    fn chain_with_accession_region_inputs() -> (GraphConnection, HashId, Path) {
        let conn = get_connection(None).unwrap();
        let block_group_id = setup_block_group(
            &conn,
            &[("a", "AAAA"), ("b", "CCCC"), ("c", "GGGG"), ("d", "TTTT")],
            &[
                ("start", 0, "a", 0, 0),
                ("a", 4, "b", 0, 0),
                ("b", 4, "c", 0, 0),
                ("c", 4, "d", 0, 0),
                ("d", 4, "end", 0, 0),
            ],
        );
        let edge_ids = [
            (PATH_START_NODE_ID, 0),
            (node_id_for("a"), 4),
            (node_id_for("b"), 4),
            (node_id_for("c"), 4),
            (node_id_for("d"), 4),
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
        (conn, block_group_id, path)
    }

    #[test]
    fn test_crawl_graph_starts_at_an_accession_region_midpoint() {
        let (conn, block_group_id, path) = chain_with_accession_region_inputs();
        BlockGroup::add_accession(&conn, &path, "middle", 4, 12).unwrap();
        let region = resolve(&Region::parse("middle").unwrap(), &conn, "test", "test").unwrap();
        assert!(region.path.is_none());

        let graph = crawl_graph(
            &conn,
            block_group_id,
            CrawlStart::Region(&region),
            usize::MAX,
            Some(4),
        )
        .unwrap()
        .expect("should find the accession midpoint");

        assert!(graph.contains_node(block("b", 0, 4)));
        assert!(graph.contains_node(block("c", 0, 4)));
        assert!(!graph.contains_node(block("a", 0, 4)));
        assert!(!graph.contains_node(block("d", 0, 4)));
    }

    #[test]
    fn test_crawl_graph_starts_at_an_annotation_region_midpoint() {
        let (conn, block_group_id, path) = chain_with_accession_region_inputs();
        let accession = BlockGroup::add_accession(&conn, &path, "middle", 4, 12).unwrap();
        Annotation::get_or_create(&conn, "gene-middle", "genes", &accession.id, None).unwrap();
        let region = resolve(
            &Region::parse("gene-middle").unwrap(),
            &conn,
            "test",
            "test",
        )
        .unwrap();
        assert!(region.path.is_none());

        let graph = crawl_graph(
            &conn,
            block_group_id,
            CrawlStart::Region(&region),
            usize::MAX,
            Some(4),
        )
        .unwrap()
        .expect("should find the annotation midpoint");

        assert!(graph.contains_node(block("b", 0, 4)));
        assert!(graph.contains_node(block("c", 0, 4)));
        assert!(!graph.contains_node(block("a", 0, 4)));
        assert!(!graph.contains_node(block("d", 0, 4)));
    }

    #[test]
    fn test_crawl_graph_region_start_flips_offsets_in_reverse_blocks() {
        let conn = get_connection(None).unwrap();
        // Edges at coordinate four carve the node into two slices.
        let block_group_id = setup_block_group(
            &conn,
            &[("x", "AAAACCCC"), ("y", "GGGG")],
            &[
                ("start", 0, "x", 0, 0),
                ("x", 4, "y", 0, 0),
                ("y", 4, "x", 4, 0),
                ("x", 8, "end", 0, 0),
            ],
        );
        Accession::create(
            &conn,
            &NewAccession {
                name: "reverse".to_string(),
                block_group_id,
                parent_accession_id: None,
                spans: vec![AccessionSpan {
                    node_id: node_id_for("x"),
                    range: Range { start: 0, end: 8 },
                    strand: Strand::Reverse,
                }],
            },
        )
        .unwrap();
        // Accession offset one on a reversed node is node coordinate six, in the second slice.
        let region = resolve(
            &Region::parse("reverse:0-2").unwrap(),
            &conn,
            "test",
            "test",
        )
        .unwrap();

        let graph = crawl_graph(&conn, block_group_id, CrawlStart::Region(&region), 1, None)
            .unwrap()
            .expect("should find the reversed block");

        assert_eq!(graph.node_count(), 1);
        assert!(graph.contains_node(block("x", 4, 8)));
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
            None,
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
            crawler.visited_edge_records.len(),
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
        let timed_edges: Vec<(EdgeSpec<'_>, i64)> = edges
            .iter()
            .enumerate()
            .map(|(order, edge)| (*edge, order as i64))
            .collect();
        setup_block_group_at_times(conn, nodes, &timed_edges)
    }

    /// Like `setup_block_group`, with an explicit `created_on` per edge so a fixture can give
    /// edges the same creation time, as one VCF import chunk does.
    fn setup_block_group_at_times(
        conn: &GraphConnection,
        nodes: &[(&str, &str)],
        edges: &[(EdgeSpec<'_>, i64)],
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
        for (
            (source, source_coordinate, target, target_coordinate, chromosome_index),
            created_on,
        ) in edges.iter().copied()
        {
            let edge = Edge::create(
                conn,
                node_id_for(source),
                source_coordinate,
                Strand::Forward,
                node_id_for(target),
                target_coordinate,
                Strand::Forward,
            )
            .unwrap();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index,
                    phased: 0,
                }],
            );
            conn.execute(
                "UPDATE block_group_edges SET created_on = ?1 WHERE edge_id = ?2",
                (created_on, edge.id),
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

    const ADJACENT_EDIT_BASE: &str = "ATCGATCGATCGATCG";

    /// A stacked substitution of base 5 beside one of base 6: both edit sites share the junction
    /// at 6, and the crawl must carry `G -> junction -> A` as well as the other three routes.
    fn adjacent_substitutions() -> (
        Vec<(&'static str, &'static str)>,
        Vec<EdgeSpec<'static>>,
        Vec<&'static str>,
    ) {
        (
            vec![("ref", ADJACENT_EDIT_BASE), ("G", "G"), ("A", "A")],
            vec![
                ("start", 0, "ref", 0, 0),
                ("ref", 16, "end", 0, 0),
                ("ref", 5, "ref", 5, 0),
                ("ref", 5, "G", 0, -3),
                ("ref", 6, "A", 0, -3),
                ("ref", 6, "ref", 6, 0),
                ("ref", 7, "ref", 7, 0),
                ("G", 1, "ref", 6, -3),
                ("A", 1, "ref", 7, -3),
            ],
            vec![
                "ATCGAGAGATCGATCG",
                "ATCGAGCGATCGATCG",
                "ATCGATAGATCGATCG",
                "ATCGATCGATCGATCG",
            ],
        )
    }

    /// A stacked deletion of bases 5..7 beside a substitution of base 7.
    fn deletion_beside_substitution() -> (
        Vec<(&'static str, &'static str)>,
        Vec<EdgeSpec<'static>>,
        Vec<&'static str>,
    ) {
        (
            vec![("ref", ADJACENT_EDIT_BASE), ("A", "A")],
            vec![
                ("start", 0, "ref", 0, 0),
                ("ref", 16, "end", 0, 0),
                ("ref", 5, "ref", 5, 0),
                ("ref", 5, "ref", 7, -3),
                ("ref", 7, "A", 0, -3),
                ("ref", 7, "ref", 7, 0),
                ("ref", 8, "ref", 8, 0),
                ("A", 1, "ref", 8, -3),
            ],
            vec![
                "ATCGAAATCGATCG",
                "ATCGAGATCGATCG",
                "ATCGATCAATCGATCG",
                "ATCGATCGATCGATCG",
            ],
        )
    }

    /// A stacked insertion at 7 beside a replacement of bases 7..9.
    fn insertion_beside_replacement() -> (
        Vec<(&'static str, &'static str)>,
        Vec<EdgeSpec<'static>>,
        Vec<&'static str>,
    ) {
        (
            vec![("ref", ADJACENT_EDIT_BASE), ("GG", "GG"), ("TT", "TT")],
            vec![
                ("start", 0, "ref", 0, 0),
                ("ref", 16, "end", 0, 0),
                ("ref", 7, "ref", 7, 0),
                ("ref", 7, "GG", 0, -3),
                ("ref", 7, "TT", 0, -3),
                ("ref", 9, "ref", 9, 0),
                ("GG", 2, "ref", 7, -3),
                ("GG", 2, "TT", 0, -3),
                ("TT", 2, "ref", 9, -3),
            ],
            vec![
                "ATCGATCGATCGATCG",
                "ATCGATCGGGATCGATCG",
                "ATCGATCGGTTTCGATCG",
                "ATCGATCTTTCGATCG",
            ],
        )
    }

    /// Every sequence of the stored graph reads along some left-to-right walk of the crawl.
    fn spellings(graph: &GenGraph, sequences: &HashMap<HashId, &str>) -> BTreeSet<String> {
        fn walk(
            graph: &GenGraph,
            sequences: &HashMap<HashId, &str>,
            node: GraphNode,
            on_walk: &mut Vec<GraphNode>,
            text: &mut String,
            spelled: &mut BTreeSet<String>,
        ) {
            if node.node_id == PATH_END_NODE_ID {
                spelled.insert(text.clone());
                return;
            }
            let length_before = text.len();
            if !is_terminal(node.node_id) {
                let sequence = sequences[&node.node_id];
                text.push_str(&sequence[node.sequence_start as usize..node.sequence_end as usize]);
            }
            on_walk.push(node);
            let mut successors: Vec<GraphNode> = graph
                .neighbors_directed(node, Direction::Outgoing)
                .collect();
            successors.sort();
            for successor in successors {
                if !on_walk.contains(&successor) {
                    walk(graph, sequences, successor, on_walk, text, spelled);
                }
            }
            on_walk.pop();
            text.truncate(length_before);
        }
        let mut spelled = BTreeSet::new();
        walk(
            graph,
            sequences,
            start_sentinel(),
            &mut Vec::new(),
            &mut String::new(),
            &mut spelled,
        );
        spelled
    }

    /// Edits at neighbouring positions meet at a zero-width junction. The crawl must route every
    /// way into it to every way out of it, or a combination of the edits has no walk.
    #[test]
    fn test_adjacent_edits_keep_every_spelling_walkable_in_the_crawl() {
        let cases = [
            adjacent_substitutions(),
            deletion_beside_substitution(),
            insertion_beside_replacement(),
        ];
        for (nodes, edges, expected) in cases {
            let conn = get_connection(None).unwrap();
            let block_group_id = setup_block_group(&conn, &nodes, &edges);
            let sequences: HashMap<HashId, &str> = nodes
                .iter()
                .map(|(label, sequence)| (node_id_for(label), *sequence))
                .collect();
            let eager =
                BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
            let mut crawler = PortCrawler::new(block_group_id, false);
            let mut lazy = GenGraph::new();
            lazy.add_node(start_sentinel());
            crawl_to_exhaustion(&conn, &mut crawler, &mut lazy);

            assert_matches_up_to_junction_shortcuts(&lazy, &eager);
            let spelled = spellings(&lazy, &sequences);
            for sequence in expected {
                assert!(
                    spelled.contains(sequence),
                    "the crawl has no left-to-right walk spelling {sequence}"
                );
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
                        .expect("should locate a block of the crawled graph");
                    assert_eq!(located, *block, "coordinate {coordinate}");
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

        assert!(matches!(
            crawler.locate(&conn, &mut graph, node_id_for("unrelated"), 2),
            Err(LocateError::NotFound)
        ));
        assert!(matches!(
            crawler.locate(&conn, &mut graph, PATH_START_NODE_ID, 0),
            Err(LocateError::NotFound)
        ));
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

    /// Two deletions that meet at base 6, as the Python editing API writes them: a cut marker
    /// at each of the three ports and one skip edge per deletion. The blocks `[3,6)` and `[6,9)`
    /// are dead, but the second skip edge starts at the shared port 6, so it also leaves `[3,6)`.
    fn adjacent_deletions_block_group(conn: &GraphConnection) -> HashId {
        setup_block_group(
            conn,
            &[("ref", ADJACENT_EDIT_BASE)],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 16, "end", 0, 0),
                ("ref", 3, "ref", 3, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
                ("ref", 6, "ref", 6, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
                ("ref", 9, "ref", 9, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
                ("ref", 3, "ref", 6, 0),
                ("ref", 6, "ref", 9, 0),
            ],
        )
    }

    /// A heterozygous SNP at `[4,5)` followed by a homozygous one at `[5,6)`, imported in one
    /// chunk so every variant edge shares one creation time. The first record's right marker and
    /// the second record's entry edge both leave port 5 on chromosome 0. `swap_creation_order`
    /// flips which of the two is stored first, and so which has the lower edge ID.
    fn adjacent_heterozygous_then_homozygous_block_group(
        conn: &GraphConnection,
        swap_creation_order: bool,
    ) -> HashId {
        let first_record: [EdgeSpec<'_>; 4] = [
            ("ref", 4, "ref", 4, 0),
            ("ref", 5, "ref", 5, 0),
            ("ref", 4, "first", 0, 1),
            ("first", 1, "ref", 5, 1),
        ];
        let second_record: [EdgeSpec<'_>; 6] = [
            ("ref", 5, "ref", 5, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
            ("ref", 6, "ref", 6, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
            ("ref", 5, "second", 0, 0),
            ("ref", 5, "second", 0, 1),
            ("second", 1, "ref", 6, 0),
            ("second", 1, "ref", 6, 1),
        ];
        let mut edges: Vec<(EdgeSpec<'_>, i64)> = vec![
            (("start", 0, "ref", 0, 0), 0),
            (("ref", 12, "end", 0, 0), 0),
        ];
        let (earlier, later): (&[EdgeSpec<'_>], &[EdgeSpec<'_>]) = if swap_creation_order {
            (&second_record, &first_record)
        } else {
            (&first_record, &second_record)
        };
        edges.extend(earlier.iter().chain(later).map(|edge| (*edge, 1)));
        setup_block_group_at_times(
            conn,
            &[("ref", "ACGTACGTACGT"), ("first", "G"), ("second", "C")],
            &edges,
        )
    }

    /// A homozygous insertion at base 8, as a one-record VCF import writes it: the inserted
    /// node leaves port 8 and returns to the same port, and the only link between the reference
    /// blocks `[0,8)` and `[8,16)` is the cut marker with the reserved index, which pruning drops.
    fn homozygous_insertion_block_group(conn: &GraphConnection) -> HashId {
        setup_block_group(
            conn,
            &[("ref", "ATCGATCGATCGATCG"), ("inserted", "G")],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 16, "end", 0, 0),
                ("ref", 8, "ref", 8, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
                ("ref", 8, "inserted", 0, 0),
                ("ref", 8, "inserted", 0, 1),
                ("inserted", 1, "ref", 8, 0),
                ("inserted", 1, "ref", 8, 1),
            ],
        )
    }

    /// A homozygous insertion built from two nodes, so the return to port 8 is two edges long.
    fn two_node_insertion_block_group(conn: &GraphConnection) -> HashId {
        setup_block_group(
            conn,
            &[("ref", "ATCGATCGATCGATCG"), ("first", "G"), ("second", "C")],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 16, "end", 0, 0),
                ("ref", 8, "ref", 8, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
                ("ref", 8, "first", 0, 0),
                ("ref", 8, "first", 0, 1),
                ("first", 1, "second", 0, 0),
                ("first", 1, "second", 0, 1),
                ("second", 1, "ref", 8, 0),
                ("second", 1, "ref", 8, 1),
            ],
        )
    }

    /// A homozygous two-base insertion whose second base another edit replaces, which splits the
    /// inserted node into slices the return path has to cross.
    fn split_insertion_block_group(conn: &GraphConnection) -> HashId {
        setup_block_group(
            conn,
            &[
                ("ref", "ATCGATCGATCGATCG"),
                ("inserted", "GG"),
                ("replacement", "C"),
            ],
            &[
                ("start", 0, "ref", 0, 0),
                ("ref", 16, "end", 0, 0),
                ("ref", 8, "ref", 8, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX),
                ("ref", 8, "inserted", 0, 0),
                ("ref", 8, "inserted", 0, 1),
                ("inserted", 2, "ref", 8, 0),
                ("inserted", 2, "ref", 8, 1),
                (
                    "inserted",
                    1,
                    "inserted",
                    1,
                    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
                ),
                (
                    "inserted",
                    2,
                    "inserted",
                    2,
                    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX,
                ),
                ("inserted", 1, "replacement", 0, 0),
                ("replacement", 1, "inserted", 2, 0),
            ],
        )
    }

    type BlockGroupFixture = Box<dyn Fn(&GraphConnection) -> HashId>;

    fn pruning_fixtures() -> Vec<(&'static str, BlockGroupFixture)> {
        vec![
            (
                "adjacent deletions",
                Box::new(adjacent_deletions_block_group),
            ),
            (
                "inserted reference",
                Box::new(inserted_reference_block_group),
            ),
            ("junction heavy", Box::new(junction_heavy_block_group)),
            (
                "homozygous insertion",
                Box::new(homozygous_insertion_block_group),
            ),
            (
                "two node insertion",
                Box::new(two_node_insertion_block_group),
            ),
            ("split insertion", Box::new(split_insertion_block_group)),
        ]
    }

    #[test]
    fn test_pruned_crawl_has_the_blocks_the_pruned_eager_graph_keeps() {
        let mut problems = Vec::new();
        for (name, setup) in pruning_fixtures() {
            let conn = get_connection(None).unwrap();
            let block_group_id = setup(&conn);
            let mut eager =
                BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
            BlockGroup::prune_graph(&mut eager);
            for anchor in [start_sentinel(), block("end", 0, 0)] {
                let mut crawler = PortCrawler::new(block_group_id, true);
                let mut graph = GenGraph::new();
                graph.add_node(anchor);
                crawl_to_exhaustion(&conn, &mut crawler, &mut graph);
                let crawled = connected_shape(&graph).0;
                let kept = connected_shape(&eager).0;
                let extra: Vec<&GraphNode> = crawled.difference(&kept).collect();
                let missing: Vec<&GraphNode> = kept.difference(&crawled).collect();
                if !extra.is_empty() || !missing.is_empty() {
                    problems.push(format!(
                        "{name} from {anchor:?}: loaded blocks pruning removes {extra:?}; lacks blocks pruning keeps {missing:?}"
                    ));
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn test_is_reachable_agrees_with_pruning_for_every_block() {
        for (name, setup) in pruning_fixtures() {
            let conn = get_connection(None).unwrap();
            let block_group_id = setup(&conn);
            let everything =
                BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
            let mut pruned = everything.clone();
            BlockGroup::prune_graph(&mut pruned);
            let mut crawler = PortCrawler::new(block_group_id, true);
            for node in everything.nodes() {
                assert_eq!(
                    crawler.is_reachable(&conn, node).unwrap(),
                    pruned.contains_node(node),
                    "{name}: reachability of {node:?} should match what pruning keeps"
                );
            }
        }
    }

    #[test]
    fn test_is_reachable_finds_the_live_route_beside_adjacent_deletions() {
        let conn = get_connection(None).unwrap();
        let block_group_id = adjacent_deletions_block_group(&conn);
        let mut crawler = PortCrawler::new(block_group_id, true);
        for live in [block("ref", 0, 3), block("ref", 6, 6), block("ref", 9, 16)] {
            assert!(
                crawler.is_reachable(&conn, live).unwrap(),
                "{live:?} should be reachable"
            );
        }
        for dead in [block("ref", 3, 6), block("ref", 6, 9)] {
            assert!(
                !crawler.is_reachable(&conn, dead).unwrap(),
                "{dead:?} should be cut off by the retired markers"
            );
        }
    }

    #[test]
    fn test_is_reachable_without_pruning_follows_every_stored_edge() {
        let conn = get_connection(None).unwrap();
        let block_group_id = adjacent_deletions_block_group(&conn);
        let mut crawler = PortCrawler::new(block_group_id, false);
        assert!(crawler.is_reachable(&conn, block("ref", 3, 6)).unwrap());
    }

    /// Competing edges from one import chunk share a creation time, and edge IDs are content
    /// hashes, so neither the ID nor the storage order says which should win. Both prunings must
    /// keep the same edges whatever order the competing edges were stored in.
    #[test]
    fn test_pruning_breaks_creation_time_ties_the_same_way_eagerly_and_lazily() {
        for swap_creation_order in [false, true] {
            let conn = get_connection(None).unwrap();
            let block_group_id =
                adjacent_heterozygous_then_homozygous_block_group(&conn, swap_creation_order);
            let mut eager =
                BlockGroup::get_graph(&conn, test_workspace(), &block_group_id, None).unwrap();
            BlockGroup::prune_graph(&mut eager);
            let mut crawler = PortCrawler::new(block_group_id, true);
            let mut lazy = GenGraph::new();
            lazy.add_node(start_sentinel());
            crawl_to_exhaustion(&conn, &mut crawler, &mut lazy);
            assert_eq!(
                connected_shape(&lazy).0,
                connected_shape(&eager).0,
                "swap_creation_order={swap_creation_order}: both prunings should keep the same blocks"
            );
        }
    }
}
