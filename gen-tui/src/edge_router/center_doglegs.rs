use std::collections::HashSet;

use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

use super::LayoutError;
use crate::layout::{LayoutEdge, LayoutNode, NodeRole};

/// Represents a dogleg pattern: 4 consecutive routing nodes forming vertical-horizontal-vertical
///
/// ```text
/// node_0
///  │
/// node_1───node_2
///           │
///          node_3
/// ```
struct Dogleg {
    node_0: NodeIndex,
    node_1: NodeIndex,
    node_2: NodeIndex,
    node_3: NodeIndex,
}

/// Centers dogleg patterns by moving the horizontal segment to the midpoint between
/// the vertical endpoints. This equalizes vertical edge lengths for better visual balance.
///
/// A dogleg is 4 consecutive routing nodes (node_0→node_1→node_2→node_3) where:
/// - node_0↔node_1 is vertical (same x)
/// - node_1↔node_2 is horizontal (same y)
/// - node_2↔node_3 is vertical (same x)
///
/// The function moves node_1 and node_2 to y = (node_0.y + node_3.y) / 2, checking that no other
/// routing nodes occupy the swept region.
pub fn center_doglegs(
    layer_graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected>,
) -> Result<(), LayoutError> {
    let doglegs = find_all_doglegs(layer_graph);

    for dogleg in doglegs {
        let node_0_y = layer_graph.node_weight(dogleg.node_0).unwrap().pos.y;
        let node_1 = layer_graph.node_weight(dogleg.node_1).unwrap();
        let node_2 = layer_graph.node_weight(dogleg.node_2).unwrap();
        let node_3_y = layer_graph.node_weight(dogleg.node_3).unwrap().pos.y;

        let (node_1_x, old_y) = (node_1.pos.x, node_1.pos.y);
        let node_2_x = node_2.pos.x;
        let new_y = (node_0_y + node_3_y) / 2;

        if old_y == new_y {
            continue;
        }

        let (x_min, x_max) = (node_1_x.min(node_2_x), node_1_x.max(node_2_x));
        let (y_min, y_max) = (old_y.min(new_y), old_y.max(new_y));

        let exclude: HashSet<NodeIndex> =
            [dogleg.node_0, dogleg.node_1, dogleg.node_2, dogleg.node_3]
                .into_iter()
                .collect();

        // Check for routing nodes in swept region
        let has_collision = layer_graph.node_indices().any(|idx| {
            if exclude.contains(&idx) {
                return false;
            }
            let Some(node) = layer_graph.node_weight(idx) else {
                return false;
            };
            if !matches!(node.role, NodeRole::Routing) {
                return false;
            }
            let (nx, ny) = (node.pos.x, node.pos.y);
            x_min <= nx && nx <= x_max && y_min <= ny && ny <= y_max
        });

        if has_collision {
            continue;
        }

        // Apply move
        layer_graph.node_weight_mut(dogleg.node_1).unwrap().pos.y = new_y;
        layer_graph.node_weight_mut(dogleg.node_2).unwrap().pos.y = new_y;
    }

    Ok(())
}

/// Find all dogleg patterns in the graph.
///
/// Searches for horizontal edges between degree-2 routing nodes, then verifies
/// that their outer neighbors form vertical connections.
fn find_all_doglegs(graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>) -> Vec<Dogleg> {
    let mut doglegs = Vec::new();

    // Iterate over all edges to find horizontal segments
    for edge_ref in graph.edge_references() {
        let node_1_idx = edge_ref.source();
        let node_2_idx = edge_ref.target();

        let Some(node_1) = graph.node_weight(node_1_idx) else {
            continue;
        };
        let Some(node_2) = graph.node_weight(node_2_idx) else {
            continue;
        };

        // Check if this is a horizontal edge between routing nodes
        if node_1.pos.y != node_2.pos.y {
            continue;
        }
        if !matches!(node_1.role, NodeRole::Routing) {
            continue;
        }
        if !matches!(node_2.role, NodeRole::Routing) {
            continue;
        }

        // Both must have exactly 2 neighbors
        let node_1_neighbors: Vec<_> = graph.neighbors(node_1_idx).collect();
        let node_2_neighbors: Vec<_> = graph.neighbors(node_2_idx).collect();
        if node_1_neighbors.len() != 2 || node_2_neighbors.len() != 2 {
            continue;
        }

        // Find the other neighbors (the vertical connections)
        let Some(&node_0_idx) = node_1_neighbors.iter().find(|&&idx| idx != node_2_idx) else {
            continue;
        };
        let Some(&node_3_idx) = node_2_neighbors.iter().find(|&&idx| idx != node_1_idx) else {
            continue;
        };

        let Some(node_0) = graph.node_weight(node_0_idx) else {
            continue;
        };
        let Some(node_3) = graph.node_weight(node_3_idx) else {
            continue;
        };

        // Verify both outer nodes are Routing
        if !matches!(node_0.role, NodeRole::Routing) {
            continue;
        }
        if !matches!(node_3.role, NodeRole::Routing) {
            continue;
        }

        // Verify vertical connections (same x coordinate)
        if node_0.pos.x != node_1.pos.x {
            continue;
        }
        if node_2.pos.x != node_3.pos.x {
            continue;
        }

        doglegs.push(Dogleg {
            node_0: node_0_idx,
            node_1: node_1_idx,
            node_2: node_2_idx,
            node_3: node_3_idx,
        });
    }

    doglegs
}

#[cfg(test)]
mod tests {
    use petgraph::{Undirected, stable_graph::StableGraph};

    use super::*;
    use crate::{geometry::LocalPos, layout::LayoutNode};

    #[test]
    fn test_center_doglegs_down_right_down_pull_up() {
        // Test a dogleg going down-right-down where horizontal needs to move UP to center
        // Pattern:  node_0 (y=0)
        //           │
        //           node_1───node_2  (currently at y=8, should move to y=5)
        //                    │
        //                   node_3 (y=10)
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        let node_0 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 0), (1, 1)));
        let node_1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 8), (1, 1)));
        let node_2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 8), (1, 1)));
        let node_3 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 10), (1, 1)));

        graph.add_edge(node_0, node_1, LayoutEdge::default());
        graph.add_edge(node_1, node_2, LayoutEdge::default());
        graph.add_edge(node_2, node_3, LayoutEdge::default());

        center_doglegs(&mut graph).unwrap();

        // Horizontal segment should now be at midpoint: (0 + 10) / 2 = 5
        assert_eq!(graph.node_weight(node_1).unwrap().pos.y, 5);
        assert_eq!(graph.node_weight(node_2).unwrap().pos.y, 5);
    }

    #[test]
    fn test_center_doglegs_down_right_down_pull_down() {
        // Test a dogleg going down-right-down where horizontal needs to move DOWN to center
        // Pattern:  node_0 (y=0)
        //           │
        //           node_1───node_2  (currently at y=2, should move to y=5)
        //                    │
        //                   node_3 (y=10)
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        let node_0 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 0), (1, 1)));
        let node_1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 2), (1, 1)));
        let node_2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 2), (1, 1)));
        let node_3 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 10), (1, 1)));

        graph.add_edge(node_0, node_1, LayoutEdge::default());
        graph.add_edge(node_1, node_2, LayoutEdge::default());
        graph.add_edge(node_2, node_3, LayoutEdge::default());

        center_doglegs(&mut graph).unwrap();

        // Horizontal segment should now be at midpoint: (0 + 10) / 2 = 5
        assert_eq!(graph.node_weight(node_1).unwrap().pos.y, 5);
        assert_eq!(graph.node_weight(node_2).unwrap().pos.y, 5);
    }

    #[test]
    fn test_center_doglegs_up_right_up_pull_down() {
        // Test a dogleg going up-right-up where horizontal needs to move DOWN to center
        // Pattern:  node_3 (y=0)
        //           │
        //           node_2───node_1  (currently at y=2, should move to y=5)
        //                    │
        //                   node_0 (y=10)
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        let node_0 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 10), (1, 1)));
        let node_1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 2), (1, 1)));
        let node_2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 2), (1, 1)));
        let node_3 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 0), (1, 1)));

        graph.add_edge(node_0, node_1, LayoutEdge::default());
        graph.add_edge(node_1, node_2, LayoutEdge::default());
        graph.add_edge(node_2, node_3, LayoutEdge::default());

        center_doglegs(&mut graph).unwrap();

        // Horizontal segment should now be at midpoint: (10 + 0) / 2 = 5
        assert_eq!(graph.node_weight(node_1).unwrap().pos.y, 5);
        assert_eq!(graph.node_weight(node_2).unwrap().pos.y, 5);
    }

    #[test]
    fn test_center_doglegs_up_right_up_pull_up() {
        // Test a dogleg going up-right-up where horizontal needs to move UP to center
        // Pattern:  node_3 (y=0)
        //           │
        //           node_2───node_1  (currently at y=8, should move to y=5)
        //                    │
        //                   node_0 (y=10)
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        let node_0 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 10), (1, 1)));
        let node_1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 8), (1, 1)));
        let node_2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 8), (1, 1)));
        let node_3 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 0), (1, 1)));

        graph.add_edge(node_0, node_1, LayoutEdge::default());
        graph.add_edge(node_1, node_2, LayoutEdge::default());
        graph.add_edge(node_2, node_3, LayoutEdge::default());

        center_doglegs(&mut graph).unwrap();

        // Horizontal segment should now be at midpoint: (10 + 0) / 2 = 5
        assert_eq!(graph.node_weight(node_1).unwrap().pos.y, 5);
        assert_eq!(graph.node_weight(node_2).unwrap().pos.y, 5);
    }

    #[test]
    fn test_center_doglegs_with_collision() {
        // Test that collision detection prevents centering when another routing node
        // occupies the swept region
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        // Dogleg nodes
        let node_0 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 0), (1, 1)));
        let node_1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 2), (1, 1)));
        let node_2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 2), (1, 1)));
        let node_3 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 10), (1, 1)));

        // Obstacle node in the swept region (would be at y=5 after centering)
        let obstacle = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 3, 5), (1, 1)));

        graph.add_edge(node_0, node_1, LayoutEdge::default());
        graph.add_edge(node_1, node_2, LayoutEdge::default());
        graph.add_edge(node_2, node_3, LayoutEdge::default());

        center_doglegs(&mut graph).unwrap();

        // Horizontal segment should NOT move due to collision
        assert_eq!(graph.node_weight(node_1).unwrap().pos.y, 2);
        assert_eq!(graph.node_weight(node_2).unwrap().pos.y, 2);
        // Obstacle should be unaffected
        assert_eq!(graph.node_weight(obstacle).unwrap().pos.y, 5);
    }

    #[test]
    fn test_center_doglegs_already_centered() {
        // Test that already-centered doglegs are not modified
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected>::default();

        let node_0 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 0), (1, 1)));
        let node_1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 5), (1, 1)));
        let node_2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 5), (1, 1)));
        let node_3 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 5, 10), (1, 1)));

        graph.add_edge(node_0, node_1, LayoutEdge::default());
        graph.add_edge(node_1, node_2, LayoutEdge::default());
        graph.add_edge(node_2, node_3, LayoutEdge::default());

        center_doglegs(&mut graph).unwrap();

        // Should remain at y=5 (already centered)
        assert_eq!(graph.node_weight(node_1).unwrap().pos.y, 5);
        assert_eq!(graph.node_weight(node_2).unwrap().pos.y, 5);
    }
}
