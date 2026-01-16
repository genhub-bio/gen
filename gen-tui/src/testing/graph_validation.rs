use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

use crate::layout::{LayoutEdge, LayoutNode};

/// Validation results for graph structure and layout quality
#[derive(Debug, Clone)]
pub struct GraphValidationResult {
    pub is_connected: bool,
    pub has_only_rectilinear_edges: bool,
    pub has_overlapping_nodes: bool,
    pub connected_components: usize,
    pub overlapping_pairs: Vec<(NodeIndex, NodeIndex)>,
    pub non_rectilinear_edges: Vec<(NodeIndex, NodeIndex)>,
}

impl GraphValidationResult {
    /// Check if the graph passes all validation criteria
    pub fn is_valid(&self) -> bool {
        self.is_connected && self.has_only_rectilinear_edges && !self.has_overlapping_nodes
    }

    /// Get a human-readable validation summary
    pub fn summary(&self) -> String {
        let mut issues = Vec::new();

        if !self.is_connected {
            issues.push(format!(
                "Graph is not connected ({} components)",
                self.connected_components
            ));
        }

        if !self.has_only_rectilinear_edges {
            issues.push(format!(
                "Graph has {} non-rectilinear edges",
                self.non_rectilinear_edges.len()
            ));
        }

        if self.has_overlapping_nodes {
            issues.push(format!(
                "Graph has {} overlapping node pairs",
                self.overlapping_pairs.len()
            ));
        }

        if issues.is_empty() {
            "Graph validation PASSED: connected, rectilinear, no overlaps".to_string()
        } else {
            format!("Graph validation FAILED: {}", issues.join(", "))
        }
    }
}

/// Comprehensive validation of a layout graph
pub fn validate_layout_graph(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> GraphValidationResult {
    let is_connected = is_graph_connected(graph);
    let connected_components = count_connected_components(graph);
    let (has_only_rectilinear_edges, non_rectilinear_edges) = check_rectilinear_edges(graph);
    let (has_overlapping_nodes, overlapping_pairs) = check_node_overlaps(graph);

    GraphValidationResult {
        is_connected,
        has_only_rectilinear_edges,
        has_overlapping_nodes,
        connected_components,
        overlapping_pairs,
        non_rectilinear_edges,
    }
}

/// Check if the graph is connected (single connected component)
fn is_graph_connected(graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>) -> bool {
    if graph.node_count() == 0 {
        return true; // Empty graph is considered connected
    }

    count_connected_components(graph) == 1
}

/// Count the number of connected components in the graph
fn count_connected_components(graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>) -> usize {
    let mut visited = HashSet::new();
    let mut components = 0;

    for node_index in graph.node_indices() {
        if !visited.contains(&node_index) {
            // Start BFS from this unvisited node
            let mut queue = VecDeque::new();
            queue.push_back(node_index);
            visited.insert(node_index);

            while let Some(current) = queue.pop_front() {
                for neighbor in graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            components += 1;
        }
    }

    components
}

/// Check if all edges are horizontal or vertical (rectilinear)
fn check_rectilinear_edges(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> (bool, Vec<(NodeIndex, NodeIndex)>) {
    let mut non_rectilinear_edges = Vec::new();

    for edge_index in graph.edge_indices() {
        if let Some((source_idx, target_idx)) = graph.edge_endpoints(edge_index) {
            let source_node = graph.node_weight(source_idx).unwrap();
            let target_node = graph.node_weight(target_idx).unwrap();

            let source_pos = (source_node.pos.x, source_node.pos.y);
            let target_pos = (target_node.pos.x, target_node.pos.y);

            // Edge is rectilinear if it's perfectly horizontal or vertical
            let is_horizontal = source_pos.1 == target_pos.1; // Same y-coordinate
            let is_vertical = source_pos.0 == target_pos.0; // Same x-coordinate

            if !is_horizontal && !is_vertical {
                non_rectilinear_edges.push((source_idx, target_idx));
            }
        }
    }

    let has_only_rectilinear_edges = non_rectilinear_edges.is_empty();
    (has_only_rectilinear_edges, non_rectilinear_edges)
}

/// Check for overlapping nodes (nodes at the same position)
fn check_node_overlaps(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> (bool, Vec<(NodeIndex, NodeIndex)>) {
    let mut position_to_nodes: HashMap<(i64, i64), Vec<NodeIndex>> = HashMap::new();
    let mut overlapping_pairs = Vec::new();

    // Group nodes by position
    for node_index in graph.node_indices() {
        let node = graph.node_weight(node_index).unwrap();
        let position = (node.pos.x, node.pos.y);
        position_to_nodes
            .entry(position)
            .or_default()
            .push(node_index);
    }

    // Find positions with multiple nodes (overlaps)
    for (_position, nodes) in position_to_nodes {
        if nodes.len() > 1 {
            // Generate all pairs of overlapping nodes at this position
            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    overlapping_pairs.push((nodes[i], nodes[j]));
                }
            }
        }
    }

    let has_overlapping_nodes = !overlapping_pairs.is_empty();
    (has_overlapping_nodes, overlapping_pairs)
}

/// Assert that a graph passes all validation criteria
/// This is useful for tests to catch graph structure bugs early
pub fn assert_valid_layout_graph(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
    test_name: &str,
) {
    let validation = validate_layout_graph(graph);

    if !validation.is_valid() {
        panic!(
            "Graph validation failed in test '{}': {}",
            test_name,
            validation.summary()
        );
    }
}

/// Utility function to print detailed validation information for debugging
pub fn print_validation_details(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
    validation: &GraphValidationResult,
) {
    println!("🔍 Graph Validation Details:");
    println!(
        "  📊 Nodes: {}, Edges: {}",
        graph.node_count(),
        graph.edge_count()
    );
    println!(
        "  🔗 Connected: {} ({} components)",
        validation.is_connected, validation.connected_components
    );
    println!(
        "  📐 Rectilinear edges: {}",
        validation.has_only_rectilinear_edges
    );
    println!("  🎯 No overlaps: {}", !validation.has_overlapping_nodes);

    if !validation.non_rectilinear_edges.is_empty() {
        println!("  ⚠️  Non-rectilinear edges:");
        for (source, target) in &validation.non_rectilinear_edges {
            let source_node = graph.node_weight(*source).unwrap();
            let target_node = graph.node_weight(*target).unwrap();
            println!(
                "    {:?} -> {:?}: ({},{}) -> ({},{})",
                source,
                target,
                source_node.pos.x,
                source_node.pos.y,
                target_node.pos.x,
                target_node.pos.y
            );
        }
    }

    if !validation.overlapping_pairs.is_empty() {
        println!("  ⚠️  Overlapping nodes:");
        for (node1, node2) in &validation.overlapping_pairs {
            let node1_data = graph.node_weight(*node1).unwrap();
            let _node2_data = graph.node_weight(*node2).unwrap();
            println!(
                "    {:?} and {:?} both at ({},{})",
                node1, node2, node1_data.pos.x, node1_data.pos.y
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::LocalPos, layout::NodeRole};

    #[test]
    fn test_validation_empty_graph() {
        let graph: StableGraph<LayoutNode, LayoutEdge, Undirected> = StableGraph::default();
        let validation = validate_layout_graph(&graph);

        assert!(validation.is_connected); // Empty graph is connected
        assert!(validation.has_only_rectilinear_edges);
        assert!(!validation.has_overlapping_nodes);
        assert!(validation.is_valid());
    }

    #[test]
    fn test_validation_connected_rectilinear_graph() {
        let mut graph = StableGraph::default();

        // Create a simple connected graph with rectilinear edges
        let node1 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(1)),
            LocalPos::new(0, (0, 0).into()),
            (1, 1),
            Some(0),
        ));

        let node2 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(2)),
            LocalPos::new(0, (1, 0).into()),
            (1, 1),
            Some(0),
        ));

        let node3 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(3)),
            LocalPos::new(0, (1, 1).into()),
            (1, 1),
            Some(0),
        ));

        // Add rectilinear edges
        graph.add_edge(node1, node2, LayoutEdge { bundle: vec![] }); // Horizontal
        graph.add_edge(node2, node3, LayoutEdge { bundle: vec![] }); // Vertical

        let validation = validate_layout_graph(&graph);

        assert!(validation.is_connected);
        assert!(validation.has_only_rectilinear_edges);
        assert!(!validation.has_overlapping_nodes);
        assert!(validation.is_valid());
    }

    #[test]
    fn test_validation_disconnected_graph() {
        let mut graph = StableGraph::default();

        // Create two disconnected components
        let node1 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(1)),
            LocalPos::new(0, (0, 0).into()),
            (1, 1),
            Some(0),
        ));

        let node2 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(2)),
            LocalPos::new(0, (1, 0).into()),
            (1, 1),
            Some(0),
        ));

        // Disconnected node
        let _node3 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(3)),
            LocalPos::new(0, (5, 5).into()),
            (1, 1),
            Some(0),
        ));

        graph.add_edge(node1, node2, LayoutEdge { bundle: vec![] });

        let validation = validate_layout_graph(&graph);

        assert!(!validation.is_connected);
        assert_eq!(validation.connected_components, 2);
        assert!(!validation.is_valid());
    }

    #[test]
    fn test_validation_non_rectilinear_edges() {
        let mut graph = StableGraph::default();

        // Create nodes with diagonal edge
        let node1 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(1)),
            LocalPos::new(0, (0, 0).into()),
            (1, 1),
            Some(0),
        ));

        let node2 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(2)),
            LocalPos::new(0, (1, 1).into()),
            (1, 1),
            Some(0),
        ));

        // Diagonal edge (non-rectilinear)
        graph.add_edge(node1, node2, LayoutEdge { bundle: vec![] });

        let validation = validate_layout_graph(&graph);

        assert!(validation.is_connected);
        assert!(!validation.has_only_rectilinear_edges);
        assert_eq!(validation.non_rectilinear_edges.len(), 1);
        assert!(!validation.is_valid());
    }

    #[test]
    fn test_validation_overlapping_nodes() {
        let mut graph = StableGraph::default();

        // Create overlapping nodes at the same position
        let _node1 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(1)),
            LocalPos::new(0, (0, 0).into()),
            (1, 1),
            Some(0),
        ));

        let _node2 = graph.add_node(LayoutNode::new(
            NodeRole::Data(NodeIndex::new(2)),
            LocalPos::new(0, (0, 0).into()), // Same position as node1
            (1, 1),
            Some(0),
        ));

        let validation = validate_layout_graph(&graph);

        assert!(validation.has_overlapping_nodes);
        assert_eq!(validation.overlapping_pairs.len(), 1);
        assert!(!validation.is_valid());
    }
}
