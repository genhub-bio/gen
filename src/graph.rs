use crate::models::block_group::NodeIntervalBlock;
use crate::models::block_group_edge::NO_CHROMOSOME_INDEX;
use crate::models::node::{Node, PATH_START_NODE_ID};
use crate::models::path::PathBlock;
use crate::models::strand::Strand;
use interavl::IntervalTree as IT2;
use intervaltree::IntervalTree;
use petgraph::graphmap::DiGraphMap;
use petgraph::prelude::EdgeRef;
use petgraph::visit::{
    Dfs, GraphRef, IntoEdgeReferences, IntoEdges, IntoNeighbors, IntoNeighborsDirected, NodeCount,
    Reversed,
};
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::iter::from_fn;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct GraphNode {
    pub block_id: i64,
    pub node_id: i64,
    pub sequence_start: i64,
    pub sequence_end: i64,
}

impl fmt::Display for GraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}[{}-{}]",
            self.node_id, self.sequence_start, self.sequence_end
        )
    }
}

impl GraphNode {
    pub fn length(&self) -> i64 {
        self.sequence_end - self.sequence_start
    }
}

pub type GenGraph = DiGraphMap<GraphNode, Vec<GraphEdge>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct GraphEdge {
    pub edge_id: i64,
    pub source_strand: Strand,
    pub target_strand: Strand,
    pub chromosome_index: i64,
    pub phased: i64,
}

#[derive(Debug)]
pub struct OperationGraph {
    pub graph: DiGraphMap<usize, ()>,
    max_node_id: usize,
    pub node_ids: HashMap<String, usize>,
    reverse_map: HashMap<usize, String>,
}

impl Default for OperationGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationGraph {
    pub fn new() -> Self {
        OperationGraph {
            graph: DiGraphMap::new(),
            max_node_id: 0,
            node_ids: HashMap::new(),
            reverse_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, hash_id: &str) -> usize {
        let node_id = *self.node_ids.entry(hash_id.to_string()).or_insert_with(|| {
            let node_id = self.max_node_id;
            self.reverse_map.insert(node_id, hash_id.to_string());
            self.graph.add_node(node_id);
            self.max_node_id += 1;
            node_id
        });
        node_id
    }

    pub fn remove_node(&mut self, node_id: usize) {
        self.graph.remove_node(node_id);
        if let Some(key) = self.reverse_map.remove(&node_id) {
            self.node_ids.remove(&key).unwrap();
        }
    }

    pub fn remove_key(&mut self, hash_id: &str) {
        if let Some(node_index) = self.node_ids.remove(hash_id) {
            self.graph.remove_node(node_index);
            self.reverse_map.remove(&node_index).unwrap();
        }
    }

    pub fn get_node(&self, node_id: &str) -> usize {
        self.node_ids[node_id]
    }

    pub fn get_key(&self, index: usize) -> String {
        self.reverse_map[&index].clone()
    }

    pub fn add_edge(&mut self, src: &str, target: &str) {
        let src_node_id = self.add_node(src);
        let target_node_id = self.add_node(target);
        self.graph.add_edge(src_node_id, target_node_id, ());
    }
}

// hacked from https://docs.rs/petgraph/latest/src/petgraph/algo/simple_paths.rs.html#36-102 to support digraphmap

pub fn all_simple_paths<G>(
    graph: G,
    from: G::NodeId,
    to: G::NodeId,
) -> impl Iterator<Item = Vec<G::NodeId>>
where
    G: NodeCount,
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    // list of visited nodes
    let mut visited = vec![from];
    // list of childs of currently exploring path nodes,
    // last elem is list of childs of last visited node
    let mut stack = vec![graph.neighbors_directed(from, Direction::Outgoing)];

    from_fn(move || {
        while let Some(children) = stack.last_mut() {
            if let Some(child) = children.next() {
                if child == to {
                    let path = visited.iter().cloned().chain(Some(to)).collect::<_>();
                    return Some(path);
                } else if !visited.contains(&child) {
                    visited.push(child);
                    stack.push(graph.neighbors_directed(child, Direction::Outgoing));
                }
            } else {
                stack.pop();
                visited.pop();
            }
        }
        None
    })
}

pub fn all_intermediate_edges<G>(
    graph: G,
    from: G::NodeId,
    to: G::NodeId,
) -> Vec<<G as IntoEdgeReferences>::EdgeRef>
where
    G: GraphRef + IntoEdges + petgraph::visit::IntoNeighborsDirected + petgraph::visit::Visitable,
    G::NodeId: Eq + Hash + std::fmt::Display,
{
    let mut outgoing_nodes = HashSet::new();
    outgoing_nodes.insert(from);
    let mut dfs_outbound = Dfs::new(graph, from);

    while let Some(outgoing_node) = dfs_outbound.next(graph) {
        outgoing_nodes.insert(outgoing_node);
    }

    // This is a standard graph algorithm trick.  To get all nodes on paths from a source s to a
    // target t, but not any other nodes, first find all nodes reachable from s.  Then reverse all
    // the edges in the graph and find all nodes reachable from t.  The nodes in common between
    // those two sets are the exactly the ones on some path from s to t.
    let reversed_graph = Reversed(&graph);
    let mut incoming_nodes = HashSet::new();
    incoming_nodes.insert(to);
    let mut dfs_inbound = Dfs::new(reversed_graph, to);
    while let Some(incoming_node) = dfs_inbound.next(reversed_graph) {
        incoming_nodes.insert(incoming_node);
    }

    let common_nodes: HashSet<&<G as petgraph::visit::GraphBase>::NodeId> =
        outgoing_nodes.intersection(&incoming_nodes).collect();
    let mut common_edgerefs = vec![];
    for edge in graph.edge_references() {
        let (source, target) = (edge.source(), edge.target());

        if common_nodes.contains(&source) && common_nodes.contains(&target) {
            common_edgerefs.push(edge);
        }
    }

    common_edgerefs
}

pub fn all_simple_paths_by_edge<G>(
    graph: G,
    from: G::NodeId,
    to: G::NodeId,
) -> impl Iterator<Item = Vec<G::EdgeRef>>
where
    G: NodeCount + IntoEdges,
    G: IntoNeighborsDirected,
    G::NodeId: Eq + Hash,
{
    // list of visited nodes
    let mut visited = vec![from];
    // list of childs of currently exploring path nodes,
    // last elem is list of childs of last visited node
    let mut path: Vec<G::EdgeRef> = vec![];
    let mut stack = vec![graph.edges(from)];

    from_fn(move || {
        while let Some(edges) = stack.last_mut() {
            if let Some(edge) = edges.next() {
                let target = edge.target();
                if target == to {
                    let a_path = path.iter().cloned().chain(Some(edge)).collect::<_>();
                    return Some(a_path);
                } else if !visited.contains(&target) {
                    path.push(edge);
                    visited.push(target);
                    stack.push(graph.edges(target));
                }
            } else {
                stack.pop();
                path.pop();
                visited.pop();
            }
        }
        None
    })
}

pub fn all_reachable_nodes<G>(graph: G, nodes: &[G::NodeId]) -> HashSet<G::NodeId>
where
    G: GraphRef + IntoNeighbors,
    G::NodeId: Eq + Hash + Debug,
{
    let mut stack = VecDeque::new();
    let mut reachable = HashSet::new();
    for node in nodes.iter() {
        stack.push_front(*node);
        reachable.insert(*node);
        while let Some(nx) = stack.pop_front() {
            for succ in graph.neighbors(nx) {
                if !reachable.contains(&succ) {
                    reachable.insert(succ);
                    stack.push_back(succ);
                }
            }
        }
    }
    reachable
}

pub fn flatten_to_interval_tree(
    graph: &GenGraph,
    remove_ambiguous_positions: bool,
) -> IntervalTree<i64, NodeIntervalBlock> {
    #[derive(Clone, Debug, Ord, PartialOrd, Eq, Hash, PartialEq)]
    struct NodeP {
        x: i64,
        y: i64,
    }
    let mut excluded_nodes = HashSet::new();
    let mut node_tree: HashMap<i64, IT2<NodeP, i64>> = HashMap::new();

    let mut start_nodes = vec![];
    let mut end_nodes = vec![];
    for node in graph.nodes() {
        if Node::is_start_node(node.node_id) {
            start_nodes.push(node);
        }
        if Node::is_end_node(node.node_id) {
            end_nodes.push(node);
        }
    }

    let mut spans: HashSet<NodeIntervalBlock> = HashSet::new();

    for start in start_nodes.iter() {
        for end_node in end_nodes.iter() {
            for path in all_simple_paths_by_edge(&graph, *start, *end_node) {
                let mut offset = 0;
                for (source_node, target_node, edges) in path.iter() {
                    let block_len = source_node.length();
                    let node_start = offset;
                    let node_end = offset + block_len;
                    // Use the first edge in the vector for strand information
                    let edge = &edges[0];
                    spans.insert(NodeIntervalBlock {
                        block_id: source_node.block_id,
                        node_id: source_node.node_id,
                        start: node_start,
                        end: node_end,
                        sequence_start: source_node.sequence_start,
                        sequence_end: source_node.sequence_end,
                        strand: edge.source_strand,
                    });
                    spans.insert(NodeIntervalBlock {
                        block_id: target_node.block_id,
                        node_id: target_node.node_id,
                        start: node_end,
                        end: node_end + target_node.length(),
                        sequence_start: target_node.sequence_start,
                        sequence_end: target_node.sequence_end,
                        strand: edge.target_strand,
                    });
                    if remove_ambiguous_positions {
                        for (node_id, node_range) in [
                            (
                                source_node.node_id,
                                NodeP {
                                    x: node_start,
                                    y: source_node.sequence_start,
                                }..NodeP {
                                    x: node_end,
                                    y: source_node.sequence_end,
                                },
                            ),
                            (
                                target_node.node_id,
                                NodeP {
                                    x: node_end,
                                    y: target_node.sequence_start,
                                }..NodeP {
                                    x: node_end + target_node.length(),
                                    y: target_node.sequence_end,
                                },
                            ),
                        ] {
                            // TODO; This could be a bit better by trying to conserve subregions
                            // within a node that are not ambiguous instead of kicking the entire
                            // node out.
                            node_tree
                                .entry(node_id)
                                .and_modify(|tree| {
                                    for (stored_range, _stored_node_id) in
                                        tree.iter_overlaps(&node_range)
                                    {
                                        if *stored_range != node_range {
                                            excluded_nodes.insert(node_id);
                                            break;
                                        }
                                    }
                                    tree.insert(node_range.clone(), node_id);
                                })
                                .or_insert_with(|| {
                                    let mut t = IT2::default();
                                    t.insert(node_range.clone(), node_id);
                                    t
                                });
                        }
                    }
                    offset += block_len;
                }
            }
        }
    }

    let tree: IntervalTree<i64, NodeIntervalBlock> = spans
        .iter()
        .filter(|block| !remove_ambiguous_positions || !excluded_nodes.contains(&block.node_id))
        .map(|block| (block.start..block.end, *block))
        .collect();
    tree
}

pub fn project_path(graph: &GenGraph, path_blocks: &[PathBlock]) -> Vec<(GraphNode, Strand)> {
    // When a path is created, it will refer to node positions in the graph as it exists then.
    // If the graph is then updated, the path nodes may be split and the graph no longer contains
    // the corresponding nodes in the initial path. This takes the initial path and identifies the
    // set of nodes in the current graph corresponding to the path using a depth first search
    // approach.

    #[derive(Debug)]
    struct PathNode {
        node: (GraphNode, Strand),
        // path_index serves as a pointer at which path block the added node was at. This is because
        // in a DFS, we may reset to a previous position for a search and need to reset the position
        // in the path as well.
        path_index: usize,
        prev: Option<Rc<PathNode>>,
    }

    let start_nodes = graph
        .nodes()
        .filter(|node| node.node_id == PATH_START_NODE_ID)
        .collect::<Vec<GraphNode>>();
    let start_node = start_nodes.first().unwrap();

    let mut final_path = vec![];
    let mut stack: VecDeque<Rc<PathNode>> = VecDeque::new();
    let mut visited: HashSet<(GraphNode, Strand)> = HashSet::new();
    let mut path_index = 0;
    let mut current_pos;

    // Start with the initial path containing only the start node
    stack.push_back(Rc::new(PathNode {
        node: (*start_node, Strand::Forward),
        path_index,
        prev: None,
    }));

    while let Some(current_node) = stack.pop_back() {
        let current = current_node.node;
        let mut current_block = &path_blocks[current_node.path_index];

        if !visited.insert(current) {
            continue;
        }

        // if current completes the current block, move to the next path block
        if current.0.sequence_end == current_block.sequence_end {
            path_index += 1;
            if path_index < path_blocks.len() {
                current_block = &path_blocks[path_index];
                current_pos = current_block.sequence_start;
            } else {
                // we're done, path_block is fully consumed
                let mut path = vec![];
                let mut path_ref = Some(Rc::clone(&current_node));
                while let Some(pn) = path_ref {
                    path.push(pn.node);
                    path_ref = pn.prev.clone();
                }
                final_path = path.into_iter().rev().collect();
                break;
            }
        } else {
            current_pos = current.0.sequence_end;
        }

        for (_src, neighbor, edges) in graph.edges(current.0) {
            // Use the first edge in the vector for strand information
            if neighbor.node_id == current_block.node_id
                && neighbor.sequence_start == current_pos
                && edges
                    .iter()
                    .any(|edge| current_block.strand == edge.target_strand)
            {
                stack.push_back(Rc::new(PathNode {
                    node: (neighbor, current_block.strand),
                    path_index,
                    prev: Some(Rc::clone(&current_node)),
                }));
            }
        }
    }
    final_path
}

/// Find the articulation points of a directed graph using a non-recursive approach
/// This is a modified version of the algorithm found here:
/// https://en.wikipedia.org/wiki/Biconnected_component#Articulation_points
pub fn find_articulation_points(graph: &GenGraph) -> Vec<GraphNode> {
    let mut articulation_points: Vec<GraphNode> = Vec::new();
    let mut discovery_time: HashMap<GraphNode, usize> = HashMap::new();
    let mut low: HashMap<GraphNode, usize> = HashMap::new();
    let mut parent: HashMap<GraphNode, Option<GraphNode>> = HashMap::new();
    let mut time = 0;

    for node in graph.nodes() {
        if !discovery_time.contains_key(&node) {
            let mut stack = vec![(node, None, true)];
            while let Some((u, p, is_first_time)) = stack.pop() {
                if is_first_time {
                    // Initialize discovery time and low value
                    discovery_time.insert(u, time);
                    low.insert(u, time);
                    time += 1;
                    parent.insert(u, p);

                    // Push the node back with is_first_time = false to process after its neighbors
                    stack.push((u, p, false));

                    // Consider both incoming and outgoing edges as undirected
                    let neighbors: Vec<_> = graph
                        .neighbors_directed(u, Direction::Outgoing)
                        .chain(graph.neighbors_directed(u, Direction::Incoming))
                        .collect();

                    for v in neighbors {
                        if !discovery_time.contains_key(&v) {
                            stack.push((v, Some(u), true));
                        } else if Some(v) != p {
                            // Update low[u] if v is not parent
                            let current_low = low.get(&u).cloned().unwrap_or(usize::MAX);
                            let v_disc = discovery_time.get(&v).cloned().unwrap_or(usize::MAX);
                            low.insert(u, current_low.min(v_disc));
                        }
                    }
                } else {
                    // Post-processing after visiting all neighbors
                    let mut is_articulation = false;
                    let mut child_count = 0;

                    let neighbors: Vec<_> = graph
                        .neighbors_directed(u, Direction::Outgoing)
                        .chain(graph.neighbors_directed(u, Direction::Incoming))
                        .collect();

                    for v in neighbors {
                        if parent.get(&v).cloned() == Some(Some(u)) {
                            child_count += 1;
                            let v_low = low.get(&v).cloned().unwrap_or(usize::MAX);
                            let u_disc = discovery_time.get(&u).cloned().unwrap_or(usize::MAX);
                            if v_low >= u_disc {
                                is_articulation = true;
                            }
                            let current_low = low.get(&u).cloned().unwrap_or(usize::MAX);
                            let v_low = low.get(&v).cloned().unwrap_or(usize::MAX);
                            low.insert(u, current_low.min(v_low));
                        } else if Some(v) != parent.get(&u).cloned().unwrap_or(None) {
                            let v_disc = discovery_time.get(&v).cloned().unwrap_or(usize::MAX);
                            let current_low = low.get(&u).cloned().unwrap_or(usize::MAX);
                            low.insert(u, current_low.min(v_disc));
                        }
                    }

                    let u_parent = parent.get(&u).cloned().unwrap_or(None);
                    if (u_parent.is_some() && is_articulation)
                        || (u_parent.is_none() && child_count > 1)
                    {
                        articulation_points.push(u);
                    }
                }
            }
        }
    }

    articulation_points.sort();
    articulation_points.dedup();
    articulation_points
}

pub fn connect_all_boundary_edges(graph: &mut GenGraph) {
    let mut nodes_without_incoming: Vec<GraphNode> = vec![];
    let mut nodes_without_outgoing: Vec<GraphNode> = vec![];
    for node in graph.nodes() {
        if !Node::is_terminal(node.node_id)
            && graph
                .edges_directed(node, Direction::Incoming)
                .next()
                .is_none()
        {
            nodes_without_incoming.push(node);
        }
        if !Node::is_terminal(node.node_id)
            && graph
                .edges_directed(node, Direction::Outgoing)
                .next()
                .is_none()
        {
            nodes_without_outgoing.push(node);
        }
    }

    for node in nodes_without_incoming {
        if let Some(upstream_node) = graph.nodes().find(|other_node| {
            other_node.node_id == node.node_id && other_node.sequence_end == node.sequence_start
        }) {
            graph.add_edge(
                upstream_node,
                node,
                vec![GraphEdge {
                    edge_id: -1,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                }],
            );
        }
    }
    for node in nodes_without_outgoing {
        if let Some(downstream_node) = graph.nodes().find(|other_node| {
            other_node.node_id == node.node_id && other_node.sequence_start == node.sequence_end
        }) {
            graph.add_edge(
                node,
                downstream_node,
                vec![GraphEdge {
                    edge_id: -1,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                }],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graphmap::DiGraphMap;
    use std::collections::HashSet;

    #[test]
    fn test_path_graph() {
        let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);

        graph.add_edge(1, 2, ());
        graph.add_edge(2, 3, ());

        let paths = all_simple_paths(&graph, 1, 3).collect::<Vec<Vec<i64>>>();
        assert_eq!(paths.len(), 1);
        let path = paths.first().unwrap().clone();
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn test_get_simple_paths_by_edge() {
        let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);
        graph.add_node(5);
        graph.add_node(6);
        graph.add_node(7);
        graph.add_node(8);
        graph.add_node(9);

        graph.add_edge(1, 2, ());
        graph.add_edge(2, 3, ());
        graph.add_edge(3, 4, ());
        graph.add_edge(4, 5, ());
        graph.add_edge(2, 6, ());
        graph.add_edge(6, 7, ());
        graph.add_edge(7, 4, ());
        graph.add_edge(6, 8, ());
        graph.add_edge(8, 7, ());

        let edge_path =
            all_simple_paths_by_edge(&graph, 1, 5).collect::<Vec<Vec<(i64, i64, &())>>>();
        assert_eq!(
            edge_path,
            vec![
                vec![(1, 2, &()), (2, 3, &()), (3, 4, &()), (4, 5, &())],
                vec![
                    (1, 2, &()),
                    (2, 6, &()),
                    (6, 7, &()),
                    (7, 4, &()),
                    (4, 5, &())
                ],
                vec![
                    (1, 2, &()),
                    (2, 6, &()),
                    (6, 8, &()),
                    (8, 7, &()),
                    (7, 4, &()),
                    (4, 5, &())
                ]
            ]
        );
    }

    #[test]
    fn test_two_path_graph() {
        let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);

        graph.add_edge(1, 2, ());
        graph.add_edge(1, 3, ());
        graph.add_edge(2, 4, ());
        graph.add_edge(3, 4, ());

        let paths = all_simple_paths(&graph, 1, 4).collect::<Vec<Vec<i64>>>();
        assert_eq!(paths.len(), 2);
        assert_eq!(
            HashSet::<Vec<i64>>::from_iter::<Vec<Vec<i64>>>(paths),
            HashSet::from_iter(vec![vec![1, 2, 4], vec![1, 3, 4]])
        );
    }

    #[test]
    fn test_two_by_two_combinatorial_graph() {
        let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);
        graph.add_node(5);
        graph.add_node(6);
        graph.add_node(7);

        graph.add_edge(1, 2, ());
        graph.add_edge(1, 3, ());
        graph.add_edge(2, 4, ());
        graph.add_edge(3, 4, ());
        graph.add_edge(4, 5, ());
        graph.add_edge(4, 6, ());
        graph.add_edge(5, 7, ());
        graph.add_edge(6, 7, ());

        let paths = all_simple_paths(&graph, 1, 7).collect::<Vec<Vec<i64>>>();
        assert_eq!(paths.len(), 4);
        assert_eq!(
            HashSet::<Vec<i64>>::from_iter::<Vec<Vec<i64>>>(paths),
            HashSet::from_iter(vec![
                vec![1, 2, 4, 5, 7],
                vec![1, 3, 4, 5, 7],
                vec![1, 2, 4, 6, 7],
                vec![1, 3, 4, 6, 7]
            ])
        );
    }

    #[test]
    fn test_three_by_three_combinatorial_graph() {
        let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);
        graph.add_node(5);
        graph.add_node(6);
        graph.add_node(7);
        graph.add_node(8);
        graph.add_node(9);

        graph.add_edge(1, 2, ());
        graph.add_edge(1, 3, ());
        graph.add_edge(1, 4, ());
        graph.add_edge(2, 5, ());
        graph.add_edge(3, 5, ());
        graph.add_edge(4, 5, ());
        graph.add_edge(5, 6, ());
        graph.add_edge(5, 7, ());
        graph.add_edge(5, 8, ());
        graph.add_edge(6, 9, ());
        graph.add_edge(7, 9, ());
        graph.add_edge(8, 9, ());

        let paths = all_simple_paths(&graph, 1, 9).collect::<Vec<Vec<i64>>>();
        assert_eq!(paths.len(), 9);
        let expected_paths = vec![
            vec![1, 2, 5, 6, 9],
            vec![1, 3, 5, 6, 9],
            vec![1, 4, 5, 6, 9],
            vec![1, 2, 5, 7, 9],
            vec![1, 3, 5, 7, 9],
            vec![1, 4, 5, 7, 9],
            vec![1, 2, 5, 8, 9],
            vec![1, 3, 5, 8, 9],
            vec![1, 4, 5, 8, 9],
        ];
        assert_eq!(
            HashSet::<Vec<i64>>::from_iter::<Vec<Vec<i64>>>(paths),
            HashSet::from_iter(expected_paths)
        );
    }

    #[test]
    fn test_super_bubble_path() {
        // This graph looks like this:
        //              8
        //            /  \
        //          6  -> 7
        //         /        \
        //    1 -> 2 -> 3 -> 4 -> 5
        //
        //  We ensure that we capture all 3 paths from 1 -> 5
        let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);
        graph.add_node(5);
        graph.add_node(6);
        graph.add_node(7);
        graph.add_node(8);
        graph.add_node(9);

        graph.add_edge(1, 2, ());
        graph.add_edge(2, 3, ());
        graph.add_edge(3, 4, ());
        graph.add_edge(4, 5, ());
        graph.add_edge(2, 6, ());
        graph.add_edge(6, 7, ());
        graph.add_edge(7, 4, ());
        graph.add_edge(6, 8, ());
        graph.add_edge(8, 7, ());

        let paths = all_simple_paths(&graph, 1, 5).collect::<Vec<Vec<i64>>>();
        assert_eq!(
            HashSet::<Vec<i64>>::from_iter::<Vec<Vec<i64>>>(paths),
            HashSet::from_iter(vec![
                vec![1, 2, 3, 4, 5],
                vec![1, 2, 6, 7, 4, 5],
                vec![1, 2, 6, 8, 7, 4, 5]
            ])
        );
    }

    #[test]
    fn test_finds_all_reachable_nodes() {
        //
        //   1 -> 2 -> 3 -> 4 -> 5
        //           /
        //   6 -> 7
        //
        let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_node(4);
        graph.add_node(5);
        graph.add_node(6);
        graph.add_node(7);

        graph.add_edge(1, 2, ());
        graph.add_edge(2, 3, ());
        graph.add_edge(3, 4, ());
        graph.add_edge(4, 5, ());
        graph.add_edge(6, 7, ());
        graph.add_edge(7, 3, ());

        assert_eq!(
            all_reachable_nodes(&graph, &[1]),
            HashSet::from_iter(vec![1, 2, 3, 4, 5])
        );

        assert_eq!(
            all_reachable_nodes(&graph, &[1, 6]),
            HashSet::from_iter(vec![1, 2, 3, 4, 5, 6, 7])
        );

        assert_eq!(
            all_reachable_nodes(&graph, &[3]),
            HashSet::from_iter(vec![3, 4, 5])
        );

        assert_eq!(
            all_reachable_nodes(&graph, &[5]),
            HashSet::from_iter(vec![5])
        );
    }

    mod test_all_intermediate_edges {
        use super::*;
        #[test]
        fn test_one_part_group() {
            //
            //   1 -> 2 -> 3 -> 4 -> 5
            //         \-> 6 /
            //
            let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
            graph.add_node(1);
            graph.add_node(2);
            graph.add_node(3);
            graph.add_node(4);
            graph.add_node(5);
            graph.add_node(6);

            graph.add_edge(1, 2, ());
            graph.add_edge(2, 3, ());
            graph.add_edge(3, 4, ());
            graph.add_edge(4, 5, ());
            graph.add_edge(2, 6, ());
            graph.add_edge(6, 4, ());

            let result_edges = all_intermediate_edges(&graph, 2, 4);
            let intermediate_edges = result_edges
                .iter()
                .map(|(source, target, _weight)| (*source, *target))
                .collect::<Vec<(i64, i64)>>();
            assert_eq!(intermediate_edges, vec![(2, 3), (3, 4), (2, 6), (6, 4)]);
        }

        #[test]
        fn test_two_part_groups() {
            //
            //   1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
            //         \-> 8 /    \-> 9 /
            //
            let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
            graph.add_node(1);
            graph.add_node(2);
            graph.add_node(3);
            graph.add_node(4);
            graph.add_node(5);
            graph.add_node(6);
            graph.add_node(7);
            graph.add_node(8);
            graph.add_node(9);

            graph.add_edge(1, 2, ());
            graph.add_edge(2, 3, ());
            graph.add_edge(3, 4, ());
            graph.add_edge(4, 5, ());
            graph.add_edge(5, 6, ());
            graph.add_edge(6, 7, ());
            graph.add_edge(2, 8, ());
            graph.add_edge(8, 4, ());
            graph.add_edge(4, 9, ());
            graph.add_edge(9, 6, ());

            let result_edges = all_intermediate_edges(&graph, 2, 6);
            let intermediate_edges = result_edges
                .iter()
                .map(|(source, target, _weight)| (*source, *target))
                .collect::<Vec<(i64, i64)>>();
            assert_eq!(
                intermediate_edges,
                vec![
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (2, 8),
                    (8, 4),
                    (4, 9),
                    (9, 6)
                ]
            );
        }

        #[test]
        fn test_one_part_group_with_unrelated_edges() {
            //
            //        / ------- 7 \
            //   1 -> 2 -> 3 -> 4 -> 5
            //         \-> 6 /
            //
            // Because 7 has 5 as a target, it is excluded from the subgraph from 2 to 4
            let mut graph: DiGraphMap<i64, ()> = DiGraphMap::new();
            graph.add_node(1);
            graph.add_node(2);
            graph.add_node(3);
            graph.add_node(4);
            graph.add_node(5);
            graph.add_node(6);
            graph.add_node(7);

            graph.add_edge(1, 2, ());
            graph.add_edge(2, 3, ());
            graph.add_edge(3, 4, ());
            graph.add_edge(4, 5, ());
            graph.add_edge(2, 6, ());
            graph.add_edge(6, 4, ());
            graph.add_edge(2, 7, ());
            graph.add_edge(7, 5, ());

            let result_edges = all_intermediate_edges(&graph, 2, 4);
            let intermediate_edges = result_edges
                .iter()
                .map(|(source, target, _weight)| (*source, *target))
                .collect::<Vec<(i64, i64)>>();
            assert_eq!(intermediate_edges, vec![(2, 3), (3, 4), (2, 6), (6, 4)]);
        }
    }

    #[cfg(test)]
    mod project_path {
        use super::*;
        use crate::models::node::PATH_END_NODE_ID;
        use petgraph::graphmap::DiGraphMap;

        #[test]
        fn test_simple_path_projection() {
            // graph looks like
            //
            //           /-> 4.3.5 -> 5.0.3 ---\
            //  s -> 3.0.5 -------------------> 3.5.15 -> e
            //  initial path is defined as 3.0.0 to 3.0.15
            let mut graph: GenGraph = DiGraphMap::new();
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: PATH_START_NODE_ID,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 0,
                    sequence_end: 5,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 0,
                    sequence_end: 5,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 4,
                    sequence_start: 3,
                    sequence_end: 5,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 4,
                    sequence_start: 3,
                    sequence_end: 5,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 5,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 5,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 5,
                    sequence_end: 15,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 0,
                    sequence_end: 5,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 5,
                    sequence_end: 15,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 5,
                    sequence_end: 15,
                },
                GraphNode {
                    block_id: -1,
                    node_id: PATH_END_NODE_ID,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let path_blocks = vec![
                PathBlock {
                    id: 0,
                    node_id: PATH_START_NODE_ID,
                    block_sequence: "".to_string(),
                    sequence_start: 0,
                    sequence_end: 0,
                    path_start: 0,
                    path_end: 0,
                    strand: Strand::Forward,
                },
                PathBlock {
                    id: 0,
                    node_id: 3,
                    block_sequence: "".to_string(),
                    sequence_start: 0,
                    sequence_end: 15,
                    path_start: 0,
                    path_end: 0,
                    strand: Strand::Forward,
                },
                PathBlock {
                    id: 0,
                    node_id: PATH_END_NODE_ID,
                    block_sequence: "".to_string(),
                    sequence_start: 0,
                    sequence_end: 0,
                    path_start: 0,
                    path_end: 0,
                    strand: Strand::Forward,
                },
            ];
            let projection = project_path(&graph, &path_blocks);
            assert_eq!(
                projection,
                [
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: PATH_START_NODE_ID,
                            sequence_start: 0,
                            sequence_end: 0
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: 3,
                            sequence_start: 0,
                            sequence_end: 5
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: 3,
                            sequence_start: 5,
                            sequence_end: 15
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: PATH_END_NODE_ID,
                            sequence_start: 0,
                            sequence_end: 0
                        },
                        Strand::Forward
                    )
                ]
            )
        }

        #[test]
        fn test_nested_path_projection() {
            // graph looks like
            //                                                /-> 7.0.2 -\
            //           /-> 4.3.5 -> 5.0.3 ---\     /-> 6.0.3----------->6.3.7 -\
            //  s -> 3.0.5 -------------------> 3.5.10 ------------------------> 3.10.15 -> e
            //  initial path is defined as 3.0.0 -> 3.5.10 -> 6.0.7 -> 3.10.15
            let mut graph: GenGraph = DiGraphMap::new();
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: PATH_START_NODE_ID,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 0,
                    sequence_end: 5,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 0,
                    sequence_end: 5,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 4,
                    sequence_start: 3,
                    sequence_end: 5,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 4,
                    sequence_start: 3,
                    sequence_end: 5,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 5,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 5,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 5,
                    sequence_end: 10,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 0,
                    sequence_end: 5,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 5,
                    sequence_end: 10,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );

            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 5,
                    sequence_end: 10,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 10,
                    sequence_end: 15,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );

            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 5,
                    sequence_end: 10,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 6,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );

            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 6,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 6,
                    sequence_start: 3,
                    sequence_end: 7,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );

            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 6,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 7,
                    sequence_start: 0,
                    sequence_end: 2,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 7,
                    sequence_start: 0,
                    sequence_end: 2,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 6,
                    sequence_start: 3,
                    sequence_end: 7,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );

            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 6,
                    sequence_start: 3,
                    sequence_end: 7,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 10,
                    sequence_end: 15,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );

            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 3,
                    sequence_start: 10,
                    sequence_end: 15,
                },
                GraphNode {
                    block_id: -1,
                    node_id: PATH_END_NODE_ID,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let path_blocks = vec![
                PathBlock {
                    id: 0,
                    node_id: PATH_START_NODE_ID,
                    block_sequence: "".to_string(),
                    sequence_start: 0,
                    sequence_end: 0,
                    path_start: 0,
                    path_end: 0,
                    strand: Strand::Forward,
                },
                PathBlock {
                    id: 0,
                    node_id: 3,
                    block_sequence: "".to_string(),
                    sequence_start: 0,
                    sequence_end: 5,
                    path_start: 0,
                    path_end: 5,
                    strand: Strand::Forward,
                },
                PathBlock {
                    id: 0,
                    node_id: 3,
                    block_sequence: "".to_string(),
                    sequence_start: 5,
                    sequence_end: 10,
                    path_start: 5,
                    path_end: 10,
                    strand: Strand::Forward,
                },
                PathBlock {
                    id: 0,
                    node_id: 6,
                    block_sequence: "".to_string(),
                    sequence_start: 0,
                    sequence_end: 7,
                    path_start: 10,
                    path_end: 17,
                    strand: Strand::Forward,
                },
                PathBlock {
                    id: 0,
                    node_id: 3,
                    block_sequence: "".to_string(),
                    sequence_start: 10,
                    sequence_end: 15,
                    path_start: 17,
                    path_end: 23,
                    strand: Strand::Forward,
                },
                PathBlock {
                    id: 0,
                    node_id: PATH_END_NODE_ID,
                    block_sequence: "".to_string(),
                    sequence_start: 0,
                    sequence_end: 0,
                    path_start: 0,
                    path_end: 0,
                    strand: Strand::Forward,
                },
            ];
            let projection = project_path(&graph, &path_blocks);
            assert_eq!(
                projection,
                [
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: PATH_START_NODE_ID,
                            sequence_start: 0,
                            sequence_end: 0
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: 3,
                            sequence_start: 0,
                            sequence_end: 5
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: 3,
                            sequence_start: 5,
                            sequence_end: 10
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: 6,
                            sequence_start: 0,
                            sequence_end: 3
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: 6,
                            sequence_start: 3,
                            sequence_end: 7
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: 3,
                            sequence_start: 10,
                            sequence_end: 15
                        },
                        Strand::Forward
                    ),
                    (
                        GraphNode {
                            block_id: -1,
                            node_id: PATH_END_NODE_ID,
                            sequence_start: 0,
                            sequence_end: 0
                        },
                        Strand::Forward
                    )
                ]
            )
        }
    }

    #[cfg(test)]
    mod boundary_edges {
        use crate::models::node::PATH_END_NODE_ID;

        use super::super::*;

        #[test]
        fn test_connect_all_boundary_edges() {
            let mut graph = GenGraph::new();
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: PATH_START_NODE_ID,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 10,
                    sequence_start: 0,
                    sequence_end: 10,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 10,
                    sequence_start: 0,
                    sequence_end: 10,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 20,
                    sequence_start: 0,
                    sequence_end: 20,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 20,
                    sequence_start: 0,
                    sequence_end: 20,
                },
                GraphNode {
                    block_id: -1,
                    node_id: 10,
                    sequence_start: 20,
                    sequence_end: 30,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let orphan_node = graph.add_node(GraphNode {
                block_id: -1,
                node_id: 10,
                sequence_start: 10,
                sequence_end: 20,
            });
            graph.add_edge(
                GraphNode {
                    block_id: -1,
                    node_id: 10,
                    sequence_start: 20,
                    sequence_end: 30,
                },
                GraphNode {
                    block_id: -1,
                    node_id: PATH_END_NODE_ID,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                vec![GraphEdge {
                    edge_id: 0,
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let incoming_edges: Vec<_> = graph
                .edges_directed(orphan_node, Direction::Incoming)
                .collect();
            let outgoing_edges: Vec<_> = graph
                .edges_directed(orphan_node, Direction::Outgoing)
                .collect();
            assert!(incoming_edges.is_empty());
            assert!(outgoing_edges.is_empty());
            connect_all_boundary_edges(&mut graph);
            let incoming_edges: Vec<_> = graph
                .edges_directed(orphan_node, Direction::Incoming)
                .collect();
            let outgoing_edges: Vec<_> = graph
                .edges_directed(orphan_node, Direction::Outgoing)
                .collect();
            assert_eq!(
                incoming_edges,
                vec![(
                    GraphNode {
                        block_id: -1,
                        node_id: 10,
                        sequence_start: 0,
                        sequence_end: 10
                    },
                    orphan_node,
                    &vec![GraphEdge {
                        edge_id: -1,
                        source_strand: Strand::Forward,
                        target_strand: Strand::Forward,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0
                    }]
                )]
            );
            assert_eq!(
                outgoing_edges,
                vec![(
                    orphan_node,
                    GraphNode {
                        block_id: -1,
                        node_id: 10,
                        sequence_start: 20,
                        sequence_end: 30
                    },
                    &vec![GraphEdge {
                        edge_id: -1,
                        source_strand: Strand::Forward,
                        target_strand: Strand::Forward,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0
                    }]
                )]
            );
        }
    }
}
