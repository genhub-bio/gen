use gen_core::HashId;
#[cfg(test)]
use gen_core::Strand;
use gen_graph::{GenGraph, GraphEdge, GraphNode};
use petgraph::graphmap::DiGraphMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DiffChangeKind {
    Unchanged,
    Added,
    Removed,
    Modified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffChange {
    pub kind: DiffChangeKind,
    pub operation: Option<HashId>,
}

impl DiffChange {
    pub const fn unchanged() -> Self {
        Self {
            kind: DiffChangeKind::Unchanged,
            operation: None,
        }
    }

    pub const fn new(kind: DiffChangeKind, operation: Option<HashId>) -> Self {
        Self { kind, operation }
    }

    pub fn operation_or(self, default_operation: HashId) -> HashId {
        self.operation.unwrap_or(default_operation)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DiffPresence {
    SourceOnly,
    TargetOnly,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffGraphNode {
    pub node: GraphNode,
    pub change: DiffChange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DiffGraphEdge {
    pub edge: GraphEdge,
    pub change: DiffChange,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_graph_to_gen_graph_maps_nodes_and_edges() {
        let node_a = DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(1),
                sequence_start: 0,
                sequence_end: 5,
            },
            change: DiffChange {
                kind: DiffChangeKind::Added,
                operation: Some(HashId::pad_str(10)),
            },
        };
        let node_b = DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(2),
                sequence_start: 5,
                sequence_end: 10,
            },
            change: DiffChange::unchanged(),
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
            change: DiffChange {
                kind: DiffChangeKind::Added,
                operation: Some(HashId::pad_str(10)),
            },
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
}
