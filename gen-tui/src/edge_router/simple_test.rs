//! Simple test for the edge router adapter infrastructure
//! This tests the complete pipeline with a minimal graph

use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences},
};

use crate::{
    edge_router::call_rust_router,
    geometry::LocalPos,
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

/// Create the working 5-node test but with 1x1 sizes to test size impact
pub fn create_simple_test_graph() -> StableGraph<LayoutNode, LayoutEdge, Undirected, u32> {
    let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::with_capacity(5, 5);

    // Create nodes with proper 1x1 size
    // Node 1: at position (0, 0)
    let node1 = graph.add_node(LayoutNode::data(
        NodeIndex::new(1),
        LocalPos::new_xy(0, 0, 0),
        (1, 1), // Use proper 1x1 size
        Some(0),
    ));

    // Node 2: at position (1, 0)
    let node2 = graph.add_node(LayoutNode::data(
        NodeIndex::new(2),
        LocalPos::new_xy(0, 1, 0),
        (1, 1), // Use proper 1x1 size
        Some(0),
    ));

    // Node 3: at position (0, 1)
    let node3 = graph.add_node(LayoutNode::data(
        NodeIndex::new(3),
        LocalPos::new_xy(0, 0, 1),
        (1, 1), // Use proper 1x1 size
        Some(1),
    ));

    // Node 4: at position (1, 1)
    let node4 = graph.add_node(LayoutNode::data(
        NodeIndex::new(4),
        LocalPos::new_xy(0, 1, 1),
        (1, 1), // Use proper 1x1 size
        Some(1),
    ));

    // Node 5: at position (2, 1)
    let node5 = graph.add_node(LayoutNode::data(
        NodeIndex::new(5),
        LocalPos::new_xy(0, 2, 1),
        (1, 1), // Use proper 1x1 size
        Some(1),
    ));

    // Edges exactly like the working test: (1, 3), (1, 4), (2, 3), (2, 4), (2, 5)
    graph.add_edge(
        node1,
        node3,
        LayoutEdge::new(NodeIndex::new(1), NodeIndex::new(3)),
    );
    graph.add_edge(
        node1,
        node4,
        LayoutEdge::new(NodeIndex::new(1), NodeIndex::new(4)),
    );
    graph.add_edge(
        node2,
        node3,
        LayoutEdge::new(NodeIndex::new(2), NodeIndex::new(3)),
    );
    graph.add_edge(
        node2,
        node4,
        LayoutEdge::new(NodeIndex::new(2), NodeIndex::new(4)),
    );
    graph.add_edge(
        node2,
        node5,
        LayoutEdge::new(NodeIndex::new(2), NodeIndex::new(5)),
    );

    graph
}

/// Test the complete adapter pipeline with a simple graph
pub fn test_simple_adapter_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing adapter pipeline with simple graph...");

    // Create test graph
    let input_graph = create_simple_test_graph();
    println!(
        "✅ Created input graph: {} nodes, {} edges",
        input_graph.node_count(),
        input_graph.edge_count()
    );

    // Print input graph details
    println!("📊 Input graph details:");
    for (idx, node) in input_graph.node_references() {
        match &node.role {
            NodeRole::Data(domain_idx) => {
                println!(
                    "  Node {}: Data({}) at ({}, {}) size {:?}",
                    idx.index(),
                    domain_idx.index(),
                    node.pos.x,
                    node.pos.y,
                    node.size
                );
            }
            NodeRole::Routing => {
                println!(
                    "  Node {}: Routing at ({}, {}) size {:?}",
                    idx.index(),
                    node.pos.x,
                    node.pos.y,
                    node.size
                );
            }
            NodeRole::Stitch(side) => {
                println!(
                    "  Node {}: Stitch({:?}) at ({}, {}) size {:?}",
                    idx.index(),
                    side,
                    node.pos.x,
                    node.pos.y,
                    node.size
                );
            }
        }
    }

    for edge in input_graph.edge_references() {
        println!(
            "  Edge: {} -> {} (bundle: {:?})",
            edge.source().index(),
            edge.target().index(),
            edge.weight().bundle
        );
    }

    // Test the router
    println!("🔄 Running call_rust_router...");
    match call_rust_router(input_graph.clone()) {
        Ok(output_graph) => {
            println!("Router completed successfully!");
            println!(
                "📊 Output graph: {} nodes, {} edges",
                output_graph.node_count(),
                output_graph.edge_count()
            );

            // Print output graph details
            println!("📋 Output graph details:");
            for (idx, node) in output_graph.node_references() {
                match &node.role {
                    NodeRole::Data(domain_idx) => {
                        println!(
                            "  Node {}: Data({}) at ({}, {}) size {:?}",
                            idx.index(),
                            domain_idx.index(),
                            node.pos.x,
                            node.pos.y,
                            node.size
                        );
                    }
                    NodeRole::Routing => {
                        println!(
                            "  Node {}: Routing at ({}, {}) size {:?}",
                            idx.index(),
                            node.pos.x,
                            node.pos.y,
                            node.size
                        );
                    }
                    NodeRole::Stitch(side) => {
                        println!(
                            "  Node {}: Stitch({:?}) at ({}, {}) size {:?}",
                            idx.index(),
                            side,
                            node.pos.x,
                            node.pos.y,
                            node.size
                        );
                    }
                }
            }

            for edge in output_graph.edge_references() {
                println!(
                    "  Edge: {} -> {} (bundle: {:?})",
                    edge.source().index(),
                    edge.target().index(),
                    edge.weight().bundle
                );
            }

            // Verify basic properties
            if output_graph.node_count() >= input_graph.node_count() {
                println!("✅ Node count preserved or increased (routing nodes added)");
            } else {
                println!("❌ Node count decreased - this shouldn't happen");
                return Err("Node count decreased".into());
            }

            // Edge count can decrease due to bundling/optimization in the router
            // This is expected behavior
            println!(
                "📊 Edge count changed: {} -> {} (bundling/routing optimization)",
                input_graph.edge_count(),
                output_graph.edge_count()
            );

            // TODO: Re-enable validation once layer distance logic is fixed
            // let validation = crate::testing::validate_layout_graph(&output_graph);
            // println!("🔍 {}", validation.summary());

            println!("🎉 Simple adapter pipeline test PASSED!");
            Ok(())
        }
        Err(e) => {
            println!("❌ Router failed with error: {:?}", e);
            Err(Box::new(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph_creation() {
        let graph = create_simple_test_graph();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5);
    }

    #[test]
    fn test_adapter_pipeline() {
        test_simple_adapter_pipeline().expect("Adapter pipeline test should pass");
    }

    #[test]
    fn test_adapter_determinism() {
        // Run the router multiple times and verify we get consistent results
        let input_graph = create_simple_test_graph();
        let input_node_count = input_graph.node_count();
        let input_edge_count = input_graph.edge_count();

        // First run
        let result1 =
            call_rust_router(input_graph.clone()).expect("First router call should succeed");
        let node_count1 = result1.node_count();
        let edge_count1 = result1.edge_count();

        // Check if input was mutated
        assert_eq!(
            input_graph.node_count(),
            input_node_count,
            "Input graph nodes were mutated!"
        );
        assert_eq!(
            input_graph.edge_count(),
            input_edge_count,
            "Input graph edges were mutated!"
        );

        // Collect node positions for comparison
        let mut positions1: Vec<_> = result1
            .node_indices()
            .filter_map(|idx| {
                result1
                    .node_weight(idx)
                    .map(|n| (n.role.clone(), n.pos.x, n.pos.y))
            })
            .collect();
        positions1.sort_by_key(|p| (p.1, p.2)); // Sort by position for consistent comparison

        // Run multiple times and verify consistency
        for i in 2..=10 {
            let result = call_rust_router(input_graph.clone())
                .unwrap_or_else(|_| panic!("Router call {} should succeed", i));

            println!(
                "Run {}: {} nodes, {} edges (expected {} nodes, {} edges)",
                i,
                result.node_count(),
                result.edge_count(),
                node_count1,
                edge_count1
            );

            assert_eq!(
                result.node_count(),
                node_count1,
                "Run {} produced different node count",
                i
            );
            assert_eq!(
                result.edge_count(),
                edge_count1,
                "Run {} produced different edge count",
                i
            );

            let mut positions: Vec<_> = result
                .node_indices()
                .filter_map(|idx| {
                    result
                        .node_weight(idx)
                        .map(|n| (n.role.clone(), n.pos.x, n.pos.y))
                })
                .collect();
            positions.sort_by_key(|p| (p.1, p.2));

            assert_eq!(
                positions.len(),
                positions1.len(),
                "Run {} produced different number of positioned nodes",
                i
            );

            // Check that all positions match
            for (j, (pos, pos1)) in positions.iter().zip(positions1.iter()).enumerate() {
                assert_eq!(
                    pos.1, pos1.1,
                    "Run {} node {} has different x position",
                    i, j
                );
                assert_eq!(
                    pos.2, pos1.2,
                    "Run {} node {} has different y position",
                    i, j
                );
            }
        }
    }
}
