use std::collections::{HashMap, HashSet};

use gen_core::{
    HashId, NO_CHROMOSOME_INDEX,
    Strand::{self, Forward},
    is_end_node, is_start_node, is_terminal,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use gen_models::{
    block_group_edge::BlockGroupEdge, changesets::ChangesetModels, edge::Edge,
    errors::OperationError, node::Node, operations::Operation, sequence::Sequence,
    session_operations::DependencyModels, traits::Query,
};
use html_escape;
use itertools::Itertools;
use petgraph::{Direction, graphmap::DiGraphMap};
use rusqlite::Connection;

use crate::patch::OperationPatch;

pub fn get_change_graph_from_hash(
    conn: &Connection,
    hash: &HashId,
) -> Result<HashMap<HashId, GenGraph>, OperationError> {
    let operation =
        Operation::get_by_id(conn, hash).ok_or(OperationError::NoOperation(format!("{hash}")))?;

    let changeset = operation.get_changeset();
    let dependencies = operation.get_changeset_dependencies();

    Ok(get_change_graph(&changeset.changes, &dependencies))
}

pub fn get_change_graph(
    changes: &ChangesetModels,
    dependencies: &DependencyModels,
) -> HashMap<HashId, GenGraph> {
    let start_node = Node::get_start_node();
    let end_node = Node::get_end_node();
    let mut bges_by_bg: HashMap<HashId, Vec<&BlockGroupEdge>> = HashMap::new();
    let mut edges_by_id: HashMap<HashId, &Edge> = HashMap::new();
    let mut nodes_by_id: HashMap<HashId, &Node> = HashMap::new();
    nodes_by_id.insert(start_node.id, &start_node);
    nodes_by_id.insert(end_node.id, &end_node);
    let mut sequences_by_hash: HashMap<HashId, &Sequence> = HashMap::new();
    let mut block_graphs: HashMap<HashId, GenGraph> = HashMap::new();

    for bge in changes.block_group_edges.iter() {
        bges_by_bg
            .entry(bge.block_group_id)
            .and_modify(|l| l.push(bge))
            .or_insert_with(|| vec![bge]);
    }
    for edge in changes.edges.iter().chain(dependencies.edges.iter()) {
        edges_by_id.insert(edge.id, edge);
    }
    for node in changes.nodes.iter().chain(dependencies.nodes.iter()) {
        nodes_by_id.insert(node.id, node);
    }
    for seq in changes
        .sequences
        .iter()
        .chain(dependencies.sequences.iter())
    {
        sequences_by_hash.insert(seq.hash, seq);
    }

    for (bg_id, bg_edges) in bges_by_bg.iter() {
        // There are 2 graphs created here. The first graph is our normal graph of nodes
        // and edges. This graph is then used to make our second graph representing the spans
        // of each node (blocks).
        let mut graph: DiGraphMap<HashId, Vec<(i64, i64)>> = DiGraphMap::new();
        let mut block_graph = GenGraph::new();
        block_graph.add_node(GraphNode {
            block_id: -1,
            node_id: start_node.id,
            sequence_start: 0,
            sequence_end: 0,
        });
        block_graph.add_node(GraphNode {
            block_id: -1,
            node_id: end_node.id,
            sequence_start: 0,
            sequence_end: 0,
        });
        for bg_edge in bg_edges {
            let edge = *edges_by_id.get(&bg_edge.edge_id).unwrap();
            if let Some(weights) = graph.edge_weight_mut(edge.source_node_id, edge.target_node_id) {
                weights.push((edge.source_coordinate, edge.target_coordinate));
            } else {
                graph.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    vec![(edge.source_coordinate, edge.target_coordinate)],
                );
            }
        }

        for node in graph.nodes() {
            // This is where we make the block graph. For this, we figure out the positions of
            // all incoming and outgoing edges from the node. Then we make blocks between those
            // positions.
            if is_terminal(node) {
                continue;
            }
            let in_ports = graph
                .edges_directed(node, Direction::Incoming)
                .flat_map(|(_src, _dest, weights)| {
                    weights.iter().map(|(_, tp)| *tp).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let out_ports = graph
                .edges_directed(node, Direction::Outgoing)
                .flat_map(|(_src, _dest, weights)| {
                    weights.iter().map(|(fp, _tp)| *fp).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let node_obj = *nodes_by_id.get(&node).unwrap();
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
                let node = GraphNode {
                    block_id: -1,
                    node_id: node,
                    sequence_start: *i,
                    sequence_end: *j,
                };
                block_graph.add_node(node);
                blocks.push(node);
            }

            for (i, j) in blocks.iter().tuple_windows() {
                block_graph.add_edge(
                    *i,
                    *j,
                    vec![GraphEdge {
                        edge_id: HashId::pad_str(1),
                        source_strand: Strand::Forward,
                        target_strand: Forward,
                        chromosome_index: 0,
                        phased: 0,
                        created_on: 0,
                    }],
                );
            }
        }

        for (src, dest, weights) in graph.all_edges() {
            for (fp, tp) in weights {
                if !(is_end_node(src) && is_start_node(dest)) {
                    let source_block = block_graph
                        .nodes()
                        .find(|node| node.node_id == src && node.sequence_end == *fp)
                        .unwrap();
                    let dest_block = block_graph
                        .nodes()
                        .find(|node| node.node_id == dest && node.sequence_start == *tp)
                        .unwrap();
                    block_graph.add_edge(
                        source_block,
                        dest_block,
                        vec![GraphEdge {
                            edge_id: HashId::pad_str(1),
                            source_strand: Strand::Forward,
                            target_strand: Forward,
                            chromosome_index: 0,
                            phased: 0,
                            created_on: 0,
                        }],
                    );
                }
            }
        }
        block_graphs.insert(*bg_id, block_graph);
    }
    block_graphs
}

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

pub fn diff_graph_to_gen_graph(diff_graph: &DiffGenGraph) -> GenGraph {
    let mut graph = GenGraph::new();
    for node in diff_graph.nodes() {
        graph.add_node(node.node);
    }
    for (src, dest, edges) in diff_graph.all_edges() {
        let mapped_edges = edges.iter().map(|edge| edge.edge).collect::<Vec<_>>();
        graph.add_edge(src.node, dest.node, mapped_edges);
    }
    graph
}

pub fn get_diff_graph(
    changes: &ChangesetModels,
    dependencies: &DependencyModels,
) -> HashMap<HashId, DiffGenGraph> {
    let start_node = Node::get_start_node();
    let end_node = Node::get_end_node();
    let mut bges_by_bg: HashMap<HashId, Vec<&BlockGroupEdge>> = HashMap::new();
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
    let mut nodes_without_incoming: Vec<DiffGraphNode> = vec![];
    let mut nodes_without_outgoing: Vec<DiffGraphNode> = vec![];
    for node in graph.nodes() {
        if !is_terminal(node.node.node_id)
            && graph
                .edges_directed(node, Direction::Incoming)
                .next()
                .is_none()
        {
            nodes_without_incoming.push(node);
        }
        if !is_terminal(node.node.node_id)
            && graph
                .edges_directed(node, Direction::Outgoing)
                .next()
                .is_none()
        {
            nodes_without_outgoing.push(node);
        }
    }

    for node in nodes_without_incoming {
        let upstream_node = graph.nodes().find(|other_node| {
            other_node.node.node_id == node.node.node_id
                && other_node.node.sequence_end == node.node.sequence_start
        });
        if let Some(upstream_node) = upstream_node {
            graph.add_edge(
                upstream_node,
                node,
                vec![DiffGraphEdge {
                    edge: GraphEdge {
                        edge_id: "0000000000000000000000000000000000000000000000000000000000000000"
                            .try_into()
                            .unwrap(),
                        source_strand: Strand::Forward,
                        target_strand: Strand::Forward,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0,
                        created_on: 0,
                    },
                    is_new: false,
                }],
            );
        }
    }
    for node in nodes_without_outgoing {
        let downstream_node = graph.nodes().find(|other_node| {
            other_node.node.node_id == node.node.node_id
                && other_node.node.sequence_start == node.node.sequence_end
        });
        if let Some(downstream_node) = downstream_node {
            graph.add_edge(
                node,
                downstream_node,
                vec![DiffGraphEdge {
                    edge: GraphEdge {
                        edge_id: "0000000000000000000000000000000000000000000000000000000000000000"
                            .try_into()
                            .unwrap(),
                        source_strand: Strand::Forward,
                        target_strand: Strand::Forward,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0,
                        created_on: 0,
                    },
                    is_new: false,
                }],
            );
        }
    }
}

pub fn view_patches(patches: &[OperationPatch]) -> HashMap<HashId, HashMap<HashId, String>> {
    // For each blockgroup in a patch, a .dot file is generated showing how the base sequence
    // has been updated.
    let mut diagrams: HashMap<HashId, HashMap<HashId, String>> = HashMap::new();

    for patch in patches {
        // The beginning work is loading the models from the patch as well as dependencies. Once
        // loaded, a graph is created of the added nodes and returned as a dot string
        let mut bg_dots: HashMap<HashId, String> = HashMap::new();

        let op_info = &patch.operation;
        let changeset = op_info.get_changeset();
        let dependencies = op_info.get_changeset_dependencies();

        let block_graphs = get_change_graph(&changeset.changes, &dependencies);

        let mut sequences_by_hash: HashMap<HashId, &Sequence> = HashMap::new();
        for seq in changeset
            .changes
            .sequences
            .iter()
            .chain(dependencies.sequences.iter())
        {
            sequences_by_hash.insert(seq.hash, seq);
        }
        let mut node_sequence_hashes: HashMap<HashId, HashId> = HashMap::new();
        for node in changeset
            .changes
            .nodes
            .iter()
            .chain(dependencies.nodes.iter())
        {
            node_sequence_hashes.insert(node.id, node.sequence_hash);
        }

        for (bg_id, block_graph) in block_graphs.iter() {
            let mut dot = "digraph {\n    rankdir=LR\n    node [shape=none]\n".to_string();
            for node in block_graph.nodes() {
                let node_id = node.node_id;
                let start = node.sequence_start;
                let end = node.sequence_end;
                let block_id = format!("{node_id}.{start}.{end}");
                if is_terminal(node.node_id) {
                    let label = if is_start_node(node.node_id) {
                        "start"
                    } else {
                        "end"
                    };
                    dot.push_str(&format!(
                        "\"{block_id}\" [label=\"{label}\", shape=ellipse]\n",
                    ));
                    continue;
                }

                let seq_hash = *node_sequence_hashes.get(&node.node_id).unwrap();
                let seq = *sequences_by_hash.get(&seq_hash).unwrap();
                let len = end - start;

                let formatted_seq = if len > 7 {
                    format!(
                        "{s}...{e}",
                        s = seq.get_sequence(start, start + 3),
                        e = seq.get_sequence(end - 3, end)
                    )
                } else {
                    seq.get_sequence(start, end)
                };

                let coordinates = format!("{node_id}:{start}-{end}");

                let label = format!(
                    "<\
                <TABLE BORDER='0'>\
                    <TR>\
                        <TD BORDER='1' ALIGN='CENTER' PORT='seq'>\
                            <FONT POINT-SIZE='12' FACE='Monospace'>{escaped_seq}</FONT>\
                        </TD>\
                    </TR>\
                    <TR>\
                        <TD ALIGN='CENTER'>\
                            <FONT POINT-SIZE='10'>{coordinates}</FONT>\
                        </TD>\
                    </TR>\
                </TABLE>\
                >",
                    escaped_seq = html_escape::encode_safe(&formatted_seq)
                );

                dot.push_str(&format!("\"{block_id}\" [label={label}]\n",));
            }

            for (src_node, dst_node, _) in block_graph.all_edges() {
                let src = src_node.node_id;
                let s_fp = src_node.sequence_start;
                let s_tp = src_node.sequence_end;
                let dest = dst_node.node_id;
                let d_fp = dst_node.sequence_start;
                let d_tp = dst_node.sequence_end;
                // Edges between adjacent blocks from the same node don't have an arrowhead
                // and are dashed because they represent the reference and can't be traversed.
                // TODO: In a heterozygous genome this isn't true. Check needs to be expanded.
                let style = if src == dest && d_fp == s_tp + 1 {
                    "dashed"
                } else {
                    "solid"
                };
                let arrow = if src == dest && d_fp == s_tp + 1 {
                    "none"
                } else {
                    "normal"
                };
                let headport = if is_end_node(dest) { "w" } else { "seq:w" };
                let tailport = if is_start_node(src) { "e" } else { "seq:e" };
                dot.push_str(&format!(
                    "\"{src}.{s_fp}.{s_tp}\" -> \"{dest}.{d_fp}.{d_tp}\" [arrowhead={arrow}, headport=\"{headport}\", tailport=\"{tailport}\", style=\"{style}\"]\n"
                ));
            }

            dot.push('}');
            bg_dots.insert(*bg_id, dot);
        }
        diagrams.insert(patch.operation.hash, bg_dots);
    }
    diagrams
}
