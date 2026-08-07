//! End-to-end tests for the edge router.

use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

use crate::{
    edge_router::call_rust_router,
    geometry::LocalPos,
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

fn simple_graph() -> StableGraph<LayoutNode, LayoutEdge, Undirected, u32> {
    let mut graph = StableGraph::with_capacity(5, 5);
    let positions = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)];
    let nodes: Vec<_> = positions
        .into_iter()
        .enumerate()
        .map(|(index, (x, y))| {
            let domain_index = NodeIndex::new(index + 1);
            graph.add_node(LayoutNode::data(
                domain_index,
                LocalPos::new_xy(x, y),
                (1, 1),
                Some(y as i32),
            ))
        })
        .collect();

    for (source, target) in [(0, 2), (0, 3), (1, 2), (1, 3), (1, 4)] {
        graph.add_edge(
            nodes[source],
            nodes[target],
            LayoutEdge::new(NodeIndex::new(source + 1), NodeIndex::new(target + 1)),
        );
    }

    graph
}

fn node_positions(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
) -> Vec<(NodeRole, i64, i64)> {
    let mut positions: Vec<_> = graph
        .node_weights()
        .map(|node| (node.role.clone(), node.pos.x, node.pos.y))
        .collect();
    positions.sort_by_key(|(_, x, y)| (*x, *y));
    positions
}

#[test]
fn test_simple_graph_creation() {
    let graph = simple_graph();
    assert_eq!(graph.node_count(), 5);
    assert_eq!(graph.edge_count(), 5);
}

#[test]
fn test_router_preserves_or_adds_nodes() {
    let graph = simple_graph();
    let input_node_count = graph.node_count();
    let output = call_rust_router(graph).expect("should route the graph");

    assert!(output.node_count() >= input_node_count);
}

#[test]
fn test_router_is_deterministic() {
    let graph = simple_graph();
    let expected = call_rust_router(graph.clone()).expect("should route the graph");
    let expected_positions = node_positions(&expected);

    for _ in 0..9 {
        let output = call_rust_router(graph.clone()).expect("should route the graph");
        assert_eq!(output.node_count(), expected.node_count());
        assert_eq!(output.edge_count(), expected.edge_count());
        assert_eq!(node_positions(&output), expected_positions);
    }
}
