use std::collections::{HashMap, HashSet};

use gen_core::{
    HashId, NO_CHROMOSOME_INDEX,
    Strand::{self, Forward},
    is_end_node, is_start_node, is_terminal,
};
use gen_graph::{ConnectBoundaryEdges, GenGraph, GraphEdge, GraphNode, connect_all_boundary_edges};
use gen_models::{
    block_group_edge::BlockGroupEdge, changesets::ChangesetModels, edge::Edge, node::Node,
    sequence::Sequence, session_operations::DependencyModels,
};
use itertools::Itertools;
use petgraph::{Direction, graphmap::DiGraphMap};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffGraphNode {
    pub node: GraphNode,
    pub is_new: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffGraphEdge {
    pub edge: GraphEdge,
    pub is_new: bool,
}

pub type DiffGenGraph = DiGraphMap<DiffGraphNode, Vec<DiffGraphEdge>>;

pub struct DiffGenGraphRef<'a>(pub &'a DiffGenGraph);

impl<'a> From<&'a DiffGenGraph> for DiffGenGraphRef<'a> {
    fn from(graph: &'a DiffGenGraph) -> Self {
        Self(graph)
    }
}

impl<'a> From<DiffGenGraphRef<'a>> for GenGraph {
    fn from(val: DiffGenGraphRef<'a>) -> Self {
        let mut graph = GenGraph::new();
        for node in val.0.nodes() {
            graph.add_node(node.node);
        }
        for (src, dest, edges) in val.0.all_edges() {
            let mapped_edges = edges.iter().map(|edge| edge.edge).collect::<Vec<_>>();
            graph.add_edge(src.node, dest.node, mapped_edges);
        }
        graph
    }
}

struct DiffGenGraphAdapter<'a>(&'a mut DiffGenGraph);

impl<'a> ConnectBoundaryEdges for DiffGenGraphAdapter<'a> {
    type Node = DiffGraphNode;
    type NodeIter<'b>
        = petgraph::graphmap::Nodes<'b, DiffGraphNode>
    where
        Self: 'b;

    fn graph_nodes(&self) -> Self::NodeIter<'_> {
        let graph: &DiffGenGraph = &*self.0;
        graph.nodes()
    }

    fn is_terminal(&self, node: Self::Node) -> bool {
        is_terminal(node.node.node_id)
    }

    fn can_connect_nodes(&self, node: Self::Node, other_node: Self::Node) -> bool {
        other_node.node.node_id == node.node.node_id
            && other_node.node.sequence_end == node.node.sequence_start
    }

    fn has_incoming(&self, node: Self::Node) -> bool {
        let graph: &DiffGenGraph = &*self.0;
        graph
            .edges_directed(node, Direction::Incoming)
            .next()
            .is_some()
    }

    fn has_outgoing(&self, node: Self::Node) -> bool {
        let graph: &DiffGenGraph = &*self.0;
        graph
            .edges_directed(node, Direction::Outgoing)
            .next()
            .is_some()
    }

    fn add_boundary_edge(&mut self, source: Self::Node, target: Self::Node) {
        let graph: &mut DiffGenGraph = &mut *self.0;
        graph.add_edge(
            source,
            target,
            vec![DiffGraphEdge {
                edge: GraphEdge {
                    edge_id: HashId::pad_str(0),
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                    created_on: 0,
                },
                is_new: source.is_new || target.is_new,
            }],
        );
    }
}

impl From<DiffGraphNode> for GraphNode {
    fn from(node: DiffGraphNode) -> Self {
        node.node
    }
}

impl From<DiffGraphEdge> for GraphEdge {
    fn from(edge: DiffGraphEdge) -> Self {
        edge.edge
    }
}

pub fn get_diff_graph(
    changes: &ChangesetModels,
    dependencies: &DependencyModels,
) -> HashMap<HashId, DiffGenGraph> {
    let start_node = Node::get_start_node();
    let end_node = Node::get_end_node();
    let mut bges_by_bg: HashMap<HashId, Vec<&BlockGroupEdge>> = HashMap::new();
    // the bool is marking whether or not the Edge/Node is new
    let mut edges_by_id: HashMap<HashId, (&Edge, bool)> = HashMap::new();
    let mut nodes_by_id: HashMap<HashId, (&Node, bool)> = HashMap::new();
    nodes_by_id.insert(start_node.id, (&start_node, false));
    nodes_by_id.insert(end_node.id, (&end_node, false));
    let mut sequences_by_hash: HashMap<HashId, &Sequence> = HashMap::new();
    let mut block_graphs: HashMap<HashId, DiffGenGraph> = HashMap::new();

    for bge in changes.block_group_edges.iter() {
        bges_by_bg
            .entry(bge.block_group_id)
            .and_modify(|l| l.push(bge))
            .or_insert_with(|| vec![bge]);
    }
    for edge in dependencies.edges.iter() {
        edges_by_id.insert(edge.id, (edge, false));
    }
    for edge in changes.edges.iter() {
        edges_by_id.insert(edge.id, (edge, true));
    }
    for node in dependencies.nodes.iter() {
        nodes_by_id.insert(node.id, (node, false));
    }
    for node in changes.nodes.iter() {
        nodes_by_id.insert(node.id, (node, true));
    }
    for seq in dependencies
        .sequences
        .iter()
        .chain(changes.sequences.iter())
    {
        sequences_by_hash.insert(seq.hash, seq);
    }

    for (bg_id, bg_edges) in bges_by_bg.iter() {
        let mut graph: DiGraphMap<HashId, Vec<(i64, i64, bool)>> = DiGraphMap::new();
        let mut block_graph = DiffGenGraph::new();
        block_graph.add_node(DiffGraphNode {
            node: GraphNode {
                block_id: -1,
                node_id: start_node.id,
                sequence_start: 0,
                sequence_end: 0,
            },
            is_new: false,
        });
        block_graph.add_node(DiffGraphNode {
            node: GraphNode {
                block_id: -1,
                node_id: end_node.id,
                sequence_start: 0,
                sequence_end: 0,
            },
            is_new: false,
        });
        for bg_edge in bg_edges {
            let (edge, edge_is_new) = *edges_by_id.get(&bg_edge.edge_id).unwrap();
            if let Some(weights) = graph.edge_weight_mut(edge.source_node_id, edge.target_node_id) {
                weights.push((edge.source_coordinate, edge.target_coordinate, edge_is_new));
            } else {
                graph.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    vec![(edge.source_coordinate, edge.target_coordinate, edge_is_new)],
                );
            }
        }

        for node in graph.nodes() {
            if is_terminal(node) {
                continue;
            }
            let in_ports = graph
                .edges_directed(node, Direction::Incoming)
                .flat_map(|(_src, _dest, weights)| {
                    weights.iter().map(|(_, tp, _)| *tp).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let out_ports = graph
                .edges_directed(node, Direction::Outgoing)
                .flat_map(|(_src, _dest, weights)| {
                    weights.iter().map(|(fp, _tp, _)| *fp).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let (node_obj, node_is_new) = *nodes_by_id.get(&node).unwrap();
            let sequence = *sequences_by_hash.get(&node_obj.sequence_hash).unwrap();
            let s_len = sequence.length;
            let mut block_starts: HashSet<i64> = HashSet::from_iter(in_ports.iter().copied());
            block_starts.insert(0);
            for x in out_ports.iter() {
                if *x < s_len - 1 {
                    block_starts.insert(*x);
                }
            }
            let mut block_ends: HashSet<i64> = HashSet::from_iter(out_ports.iter().copied());
            block_ends.insert(s_len);
            for x in in_ports.iter() {
                if *x > 0 {
                    block_ends.insert(*x);
                }
            }

            let block_starts = block_starts.into_iter().sorted().collect::<Vec<_>>();
            let block_ends = block_ends.into_iter().sorted().collect::<Vec<_>>();

            let mut blocks = vec![];
            for (i, j) in block_starts.iter().zip(block_ends.iter()) {
                let node = DiffGraphNode {
                    node: GraphNode {
                        block_id: -1,
                        node_id: node,
                        sequence_start: *i,
                        sequence_end: *j,
                    },
                    is_new: node_is_new,
                };
                block_graph.add_node(node);
                blocks.push(node);
            }

            for (i, j) in blocks.iter().tuple_windows() {
                block_graph.add_edge(
                    *i,
                    *j,
                    vec![DiffGraphEdge {
                        edge: GraphEdge {
                            edge_id: HashId::pad_str(1),
                            source_strand: Strand::Forward,
                            target_strand: Forward,
                            chromosome_index: 0,
                            phased: 0,
                            created_on: 0,
                        },
                        is_new: node_is_new,
                    }],
                );
            }
        }

        for (src, dest, weights) in graph.all_edges() {
            for (fp, tp, edge_is_new) in weights {
                if !(is_end_node(src) && is_start_node(dest)) {
                    let source_block = block_graph
                        .nodes()
                        .find(|node| node.node.node_id == src && node.node.sequence_end == *fp)
                        .unwrap();
                    let dest_block = block_graph
                        .nodes()
                        .find(|node| node.node.node_id == dest && node.node.sequence_start == *tp)
                        .unwrap();
                    block_graph.add_edge(
                        source_block,
                        dest_block,
                        vec![DiffGraphEdge {
                            edge: GraphEdge {
                                edge_id: HashId::pad_str(1),
                                source_strand: Strand::Forward,
                                target_strand: Forward,
                                chromosome_index: 0,
                                phased: 0,
                                created_on: 0,
                            },
                            is_new: *edge_is_new,
                        }],
                    );
                }
            }
        }

        block_graphs.insert(*bg_id, block_graph);
    }

    block_graphs
}

pub fn connect_all_boundary_edges_diff(graph: &mut DiffGenGraph) {
    connect_all_boundary_edges(&mut DiffGenGraphAdapter(graph));
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, NO_CHROMOSOME_INDEX, Strand};
    use gen_graph::{GraphEdge, GraphNode};
    use gen_models::{
        block_group::BlockGroup,
        block_group_edge::BlockGroupEdge,
        changesets::ChangesetModels,
        edge::Edge,
        node::Node,
        sequence::{NewSequence, Sequence},
        session_operations::DependencyModels,
    };

    use super::*;

    fn base_dependencies(start_node: &Node, end_node: &Node) -> DependencyModels {
        let mut start_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .name("start")
            .build();
        start_sequence.hash = start_node.sequence_hash;
        let mut end_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .name("end")
            .build();
        end_sequence.hash = end_node.sequence_hash;
        DependencyModels {
            collections: vec![],
            samples: vec![],
            sequences: vec![start_sequence, end_sequence],
            block_group: vec![],
            nodes: vec![start_node.clone(), end_node.clone()],
            edges: vec![],
            paths: vec![],
            accessions: vec![],
            accession_edges: vec![],
        }
    }

    fn find_edge(
        graph: &DiffGenGraph,
        src: DiffGraphNode,
        dest: DiffGraphNode,
    ) -> Option<&Vec<DiffGraphEdge>> {
        graph
            .all_edges()
            .find(|(edge_src, edge_dest, _)| *edge_src == src && *edge_dest == dest)
            .map(|(_, _, edges)| edges)
    }

    #[test]
    fn diff_graph_to_gen_graph_maps_nodes_and_edges() {
        let node_a = DiffGraphNode {
            node: GraphNode {
                block_id: 1,
                node_id: HashId::pad_str(1),
                sequence_start: 0,
                sequence_end: 5,
            },
            is_new: true,
        };
        let node_b = DiffGraphNode {
            node: GraphNode {
                block_id: 2,
                node_id: HashId::pad_str(2),
                sequence_start: 5,
                sequence_end: 10,
            },
            is_new: false,
        };
        let mut diff_graph = DiffGenGraph::new();
        diff_graph.add_node(node_a);
        diff_graph.add_node(node_b);
        let edge = DiffGraphEdge {
            edge: GraphEdge {
                edge_id: HashId::pad_str(9),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            is_new: true,
        };
        diff_graph.add_edge(node_a, node_b, vec![edge]);

        let graph: GenGraph = DiffGenGraphRef(&diff_graph).into();
        assert_eq!(graph.nodes().count(), 2);
        assert_eq!(graph.all_edges().count(), 1);
        let weights = graph
            .all_edges()
            .next()
            .map(|(_, _, edges)| edges.clone())
            .expect("graph edge");
        assert_eq!(weights, vec![edge.edge]);
    }

    #[test]
    fn get_diff_graph_splits_blocks_and_marks_new() {
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();
        let block_group = BlockGroup {
            id: HashId::pad_str(10),
            collection_name: "collection".to_string(),
            sample_name: Some("sample".to_string()),
            name: "bg".to_string(),
            created_on: 0,
        };
        let seq = NewSequence::new()
            .sequence_type("dna")
            .sequence("AAAAAAAAAA")
            .name("seq")
            .build();
        let node = Node {
            id: HashId::pad_str(11),
            sequence_hash: seq.hash,
        };
        let old_edge = Edge {
            id: HashId::convert_str("node-deletion"),
            source_node_id: node.id,
            source_coordinate: 3,
            source_strand: Strand::Forward,
            target_node_id: node.id,
            target_coordinate: 5,
            target_strand: Strand::Forward,
        };
        let new_seq = NewSequence::new()
            .sequence_type("dna")
            .sequence("TTTT")
            .name("new-seq")
            .build();
        let new_node = Node {
            id: HashId::pad_str(12),
            sequence_hash: new_seq.hash,
        };
        let edges = vec![
            Edge {
                id: HashId::convert_str("node-to-end-5"),
                source_node_id: node.id,
                source_coordinate: 5,
                source_strand: Strand::Forward,
                target_node_id: end_node.id,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            },
            Edge {
                id: HashId::convert_str("node-to-end-10"),
                source_node_id: node.id,
                source_coordinate: 10,
                source_strand: Strand::Forward,
                target_node_id: end_node.id,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            },
            Edge {
                id: HashId::convert_str("node-to-new-node"),
                source_node_id: node.id,
                source_coordinate: 10,
                source_strand: Strand::Forward,
                target_node_id: new_node.id,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            },
            Edge {
                id: HashId::convert_str("new-node-to-end"),
                source_node_id: new_node.id,
                source_coordinate: new_seq.length,
                source_strand: Strand::Forward,
                target_node_id: end_node.id,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            },
        ];
        let block_group_edges = edges
            .iter()
            .enumerate()
            .map(|(index, edge)| BlockGroupEdge {
                id: HashId::convert_str(&format!("bge-{index}")),
                block_group_id: block_group.id,
                edge_id: edge.id,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            })
            .collect::<Vec<_>>();
        let changes = ChangesetModels {
            collections: vec![],
            samples: vec![],
            sequences: vec![new_seq.clone()],
            block_groups: vec![block_group.clone()],
            nodes: vec![new_node.clone()],
            edges,
            block_group_edges,
            paths: vec![],
            path_edges: vec![],
            accessions: vec![],
            accession_edges: vec![],
            accession_paths: vec![],
        };
        let mut dependencies = base_dependencies(&start_node, &end_node);
        dependencies.sequences.push(seq.clone());
        dependencies.nodes.push(node.clone());
        dependencies.edges.push(old_edge);

        let diff_graphs = get_diff_graph(&changes, &dependencies);
        let graph = diff_graphs.get(&block_group.id).expect("block group graph");
        assert_eq!(graph.nodes().count(), 5);
        assert_eq!(graph.all_edges().count(), 5);

        let block_nodes = graph
            .nodes()
            .filter(|node_ref| node_ref.node.node_id == node.id)
            .collect::<Vec<_>>();
        assert_eq!(block_nodes.len(), 2);
        assert!(block_nodes.iter().all(|node_ref| !node_ref.is_new));

        let new_node_block = graph
            .nodes()
            .find(|node_ref| node_ref.node.node_id == new_node.id)
            .expect("new node block");
        assert!(new_node_block.is_new);

        let block_start = block_nodes
            .iter()
            .find(|node_ref| node_ref.node.sequence_start == 0)
            .copied()
            .expect("block start node");
        let block_mid = block_nodes
            .iter()
            .find(|node_ref| node_ref.node.sequence_start == 5)
            .copied()
            .expect("block mid node");
        dbg!(&graph.all_edges());
        let internal_edges = find_edge(graph, block_start, block_mid).expect("internal edge");
        assert_eq!(internal_edges.len(), 1);
        assert!(internal_edges[0].is_new);

        let start_block = graph
            .nodes()
            .find(|node_ref| node_ref.node.node_id == start_node.id)
            .expect("start node");
        let start_edges = find_edge(graph, start_block, block_start).expect("start edge");
        assert!(!start_edges[0].is_new);
    }

    #[test]
    fn connect_all_boundary_edges_connects_adjacent_blocks() {
        let mut graph = DiffGenGraph::new();
        let node_id = HashId::pad_str(12);
        let block_a = DiffGraphNode {
            node: GraphNode {
                block_id: 1,
                node_id,
                sequence_start: 0,
                sequence_end: 5,
            },
            is_new: true,
        };
        let block_b = DiffGraphNode {
            node: GraphNode {
                block_id: 2,
                node_id,
                sequence_start: 5,
                sequence_end: 10,
            },
            is_new: true,
        };
        graph.add_node(block_a);
        graph.add_node(block_b);

        connect_all_boundary_edges_diff(&mut graph);

        assert_eq!(graph.all_edges().count(), 1);
        let edges = find_edge(&graph, block_a, block_b).expect("boundary edge");
        assert_eq!(edges.len(), 1);
        let edge = edges[0];
        assert!(edge.is_new);
        assert_eq!(edge.edge.chromosome_index, NO_CHROMOSOME_INDEX);
    }
}
