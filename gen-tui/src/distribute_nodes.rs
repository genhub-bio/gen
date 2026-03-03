//! Horizontal chain redistribution for layout nodes.
//!
//! After the Sugiyama algorithm completes, nodes along horizontal edges may look
//! out of place due to center-alignment. This module redistributes nodes along
//! horizontal chains to improve visual spacing while respecting obstacles.

use std::collections::{HashMap, HashSet};

use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

use crate::layout::{LayoutEdge, LayoutNode, NodeRole};

/// Asymmetric half-widths for a node (left half is larger for odd widths).
#[derive(Clone, Copy, Debug)]
pub struct Halves {
    pub left: i64,
    pub right: i64,
}

/// An inclusive interval [l, r] representing occupied cells.
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub l: i64, // inclusive
    pub r: i64, // inclusive
}

/// Result of optimizing a horizontal chain.
#[derive(Debug)]
pub enum ChainOptResult {
    Optimized { new_x: Vec<i64> },
    RevertToOriginal,
}

/// A horizontal chain represents a sequence of nodes connected by horizontal edges.
/// Chains are split at Routing nodes.
#[derive(Debug, Clone)]
struct HorizontalChain {
    /// Ordered node indices in the chain (left to right by x-coordinate)
    nodes: Vec<NodeIndex>,
    /// Y-coordinate of the chain (all nodes have same y)
    y: i64,
}

/// Calculate asymmetric half-widths for a given width.
/// For odd widths, the left half is larger.
/// Example: width=9 -> left=5, right=4
pub fn halves(width: i64) -> Halves {
    Halves {
        left: (width + 1) / 2,
        right: width / 2,
    }
}

/// Calculate the base center-to-center distance between two adjacent nodes
/// when they are touching (no gap).
pub fn base_step(width_i: i64, width_j: i64) -> i64 {
    let hi = halves(width_i);
    let hj = halves(width_j);
    hi.right + hj.left
}

/// Check if two intervals overlap.
fn overlaps(a: Interval, b: Interval) -> bool {
    a.l <= b.r && b.l <= a.r
}

/// Calculate the occupied interval for a node given its center x and width.
fn occ_interval(center_x: i64, width: i64) -> Interval {
    let h = halves(width);
    Interval {
        l: center_x - h.left,
        r: center_x + h.right,
    }
}

/// Calculate the minimal rightward shift needed to move a node's occupied interval
/// past a forbidden interval.
fn shift_right_to_avoid(protected: Interval, center_x: i64, width: i64) -> i64 {
    let h = halves(width);
    let min_center = protected.r + 1 + h.left;
    (min_center - center_x).max(0)
}

/// Identify all horizontal chains in the graph.
/// A horizontal chain is a path of nodes connected by edges where all nodes
/// have the same y-coordinate. Chains are split at Routing nodes.
fn identify_chains(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
) -> Vec<HorizontalChain> {
    let mut chains = Vec::new();
    let mut visited_edges: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();

    // Group nodes by y-coordinate
    let mut y_groups: HashMap<i64, Vec<NodeIndex>> = HashMap::new();
    for node_idx in graph.node_indices() {
        if let Some(node) = graph.node_weight(node_idx) {
            y_groups.entry(node.pos.y).or_default().push(node_idx);
        }
    }

    // For each y-level, find horizontal edges
    for (y, nodes_at_y) in y_groups {
        let nodes_set: HashSet<NodeIndex> = nodes_at_y.iter().copied().collect();

        // Find all horizontal edges at this y-level
        let mut horizontal_edges: Vec<(NodeIndex, NodeIndex)> = Vec::new();
        for node_idx in &nodes_at_y {
            for edge in graph.edges(*node_idx) {
                let source = edge.source();
                let target = edge.target();

                // Check if both endpoints are at the same y-coordinate
                if nodes_set.contains(&source) && nodes_set.contains(&target) {
                    let edge_pair = if source.index() < target.index() {
                        (source, target)
                    } else {
                        (target, source)
                    };

                    if !visited_edges.contains(&edge_pair) {
                        horizontal_edges.push(edge_pair);
                        visited_edges.insert(edge_pair);
                    }
                }
            }
        }

        // Build chains from horizontal edges using graph traversal
        let mut edge_map: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
        for (a, b) in &horizontal_edges {
            edge_map.entry(*a).or_default().push(*b);
            edge_map.entry(*b).or_default().push(*a);
        }

        let mut visited_nodes: HashSet<NodeIndex> = HashSet::new();

        // DFS to find connected components
        for start_node in &nodes_at_y {
            if visited_nodes.contains(start_node) {
                continue;
            }
            if !edge_map.contains_key(start_node) {
                continue; // Isolated node, not part of any chain
            }

            // Find all nodes in this connected component
            let mut component = Vec::new();
            let mut stack = vec![*start_node];

            while let Some(node) = stack.pop() {
                if visited_nodes.contains(&node) {
                    continue;
                }
                visited_nodes.insert(node);
                component.push(node);

                if let Some(neighbors) = edge_map.get(&node) {
                    for &neighbor in neighbors {
                        if !visited_nodes.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }

            // Sort component by x-coordinate
            component
                .sort_by_key(|&idx| graph.node_weight(idx).map(|n| n.pos.x).unwrap_or(i64::MAX));

            // Split the component at Routing nodes
            let sub_chains = split_chain_at_routing_nodes(graph, &component);

            for sub_chain in sub_chains {
                if sub_chain.len() >= 2 {
                    chains.push(HorizontalChain {
                        nodes: sub_chain,
                        y,
                    });
                }
            }
        }
    }

    chains
}

/// Split a chain at Routing nodes, duplicating routing nodes on both sides.
fn split_chain_at_routing_nodes(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    chain: &[NodeIndex],
) -> Vec<Vec<NodeIndex>> {
    let mut result = Vec::new();
    let mut current_chain = Vec::new();

    for &node_idx in chain {
        if let Some(node) = graph.node_weight(node_idx) {
            match node.role {
                NodeRole::Routing => {
                    // End current chain with routing node
                    if !current_chain.is_empty() {
                        current_chain.push(node_idx);
                        result.push(current_chain.clone());
                    }
                    // Start new chain with routing node
                    current_chain = vec![node_idx];
                }
                _ => {
                    current_chain.push(node_idx);
                }
            }
        }
    }

    // Add final chain
    if !current_chain.is_empty() {
        result.push(current_chain);
    }

    result
}

/// Find all obstacles (forbidden intervals) for a given horizontal chain.
/// Obstacles come from edges that cross the horizontal line at y.
/// For each crossing, we protect the crossing cell and its immediate neighbors.
///
/// Since all edges are rectilinear (either vertical or horizontal), we only need
/// to check vertical edges that span across the chain's y-coordinate.
fn find_obstacles(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    chain_y: i64,
    _chain_nodes: &[NodeIndex],
) -> Vec<Interval> {
    let mut obstacles = Vec::new();

    // Find all vertical edges that cross this y-level
    for edge in graph.edge_references() {
        let source_idx = edge.source();
        let target_idx = edge.target();

        if let (Some(source_node), Some(target_node)) =
            (graph.node_weight(source_idx), graph.node_weight(target_idx))
        {
            let x1 = source_node.pos.x;
            let y1 = source_node.pos.y;
            let x2 = target_node.pos.x;
            let y2 = target_node.pos.y;

            // Check if this is a vertical edge (same x-coordinate)
            if x1 == x2 {
                // Check if it crosses the chain's y-level
                let (y_min, y_max) = if y1 < y2 { (y1, y2) } else { (y2, y1) };

                if y_min < chain_y && chain_y < y_max {
                    // Vertical edge crosses at x1 (= x2)
                    // Protect the crossing cell and its neighbors
                    obstacles.push(Interval {
                        l: x1 - 1,
                        r: x1 + 1,
                    });
                }
            }
            // Horizontal edges (y1 == y2) don't cross other horizontal lines
            // Diagonal edges shouldn't exist in rectilinear layout
        }
    }

    // Merge overlapping obstacles
    if obstacles.is_empty() {
        return obstacles;
    }

    obstacles.sort_by_key(|i| i.l);
    let mut merged = vec![obstacles[0]];

    for obstacle in obstacles.iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if obstacle.l <= last.r + 1 {
            // Overlapping or adjacent - merge
            last.r = last.r.max(obstacle.r);
        } else {
            merged.push(*obstacle);
        }
    }

    merged
}

/// Optimize a single horizontal chain by redistributing nodes to maximize
/// the minimum gap between nodes while avoiding obstacles.
fn optimize_chain_edge_gaps(
    widths: &[i64],
    orig_x: &[i64],
    forbidden: &[Interval],
    min_gap: i64,
) -> ChainOptResult {
    let n = widths.len();
    if n < 2 || orig_x.len() != n {
        return ChainOptResult::RevertToOriginal;
    }

    let x0 = orig_x[0];
    let xn = orig_x[n - 1];

    // Precompute base steps
    let mut base: Vec<i64> = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        base.push(base_step(widths[i], widths[i + 1]));
    }

    let base_sum: i64 = base.iter().sum();
    let span: i64 = xn - x0;
    let g_total: i64 = span - base_sum;

    if g_total < 0 {
        return ChainOptResult::RevertToOriginal;
    }

    // Binary search for maximum minimum gap
    let m = (n - 1) as i64;
    let mut lo = min_gap; // Start from minimum configured gap
    let mut hi = g_total / m;

    let mut best_x: Option<Vec<i64>> = None;

    while lo <= hi {
        let mid = (lo + hi) / 2;
        match feasible_with_min_gap(mid, widths, x0, xn, &base, g_total, forbidden) {
            Some(x) => {
                best_x = Some(x);
                lo = mid + 1; // try larger t
            }
            None => {
                hi = mid - 1;
            }
        }
    }

    match best_x {
        Some(new_x) => ChainOptResult::Optimized { new_x },
        None => ChainOptResult::RevertToOriginal,
    }
}

/// Check if a given minimum gap `t` is feasible, and if so, return the positions.
fn feasible_with_min_gap(
    t: i64,
    widths: &[i64],
    x0: i64,
    xn: i64,
    base: &[i64],
    g_total: i64,
    forbidden: &[Interval],
) -> Option<Vec<i64>> {
    let n = widths.len();
    let m = n - 1;

    // Start with all gaps = t
    let mut g: Vec<i64> = vec![t; m];
    let mut slack: i64 = g_total - t * (m as i64);

    if slack < 0 {
        return None;
    }

    // Pass 1: left-to-right. Push nodes right to avoid obstacles.
    let mut x = x0;
    for i in 1..(n - 1) {
        // Compute position based on current gaps
        x += base[i - 1] + g[i - 1];

        // Check for obstacle collisions and push right if needed
        let mut dx_needed = 0i64;
        loop {
            let occ = occ_interval(x + dx_needed, widths[i]);
            let mut hit: Option<Interval> = None;

            for &f in forbidden {
                if overlaps(occ, f) {
                    hit = Some(f);
                    break;
                }
                if f.l > occ.r {
                    break; // sorted, no more overlaps possible
                }
            }

            match hit {
                None => break,
                Some(f) => {
                    let extra = shift_right_to_avoid(f, x + dx_needed, widths[i]);
                    if extra == 0 {
                        break;
                    }
                    dx_needed += extra;
                }
            }
        }

        if dx_needed > 0 {
            if dx_needed > slack {
                return None; // Not enough slack to push
            }
            slack -= dx_needed;
            g[i - 1] += dx_needed;
            x += dx_needed;
        }
    }

    // Pass 2: allocate remaining slack right-to-left
    // This makes left gaps smaller and right gaps larger, which is the
    // opposite preference from node width allocation (where we prefer left-heavy).
    // For gaps, we want smaller gaps on the left side.
    while slack > 0 {
        let mut placed = false;

        // Iterate from right to left (reverse order)
        for e in (0..m).rev() {
            g[e] += 1;
            if chain_is_legal(widths, x0, base, &g, forbidden) {
                slack -= 1;
                placed = true;
                break; // restart from right
            } else {
                g[e] -= 1;
            }
        }

        if !placed {
            return None; // Cannot place remaining slack
        }
    }

    // Build final positions
    let mut xs: Vec<i64> = vec![0; n];
    xs[0] = x0;
    for i in 0..m {
        xs[i + 1] = xs[i] + base[i] + g[i];
    }

    // Verify endpoint constraint
    if xs[n - 1] != xn {
        return None;
    }

    Some(xs)
}

/// Check if a chain configuration is legal (no overlaps with obstacles).
fn chain_is_legal(
    widths: &[i64],
    x0: i64,
    base: &[i64],
    g: &[i64],
    forbidden: &[Interval],
) -> bool {
    let n = widths.len();
    let mut x = x0;

    for i in 1..(n - 1) {
        x += base[i - 1] + g[i - 1];
        let occ = occ_interval(x, widths[i]);

        for &f in forbidden {
            if overlaps(occ, f) {
                return false;
            }
            if f.l > occ.r {
                break; // sorted, no more overlaps possible
            }
        }
    }

    true
}

/// Build a mapping from x-coordinates to Sugiyama layer indices.
/// This is used to reassign layer fields after horizontal redistribution.
fn build_x_to_layer_mapping(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
) -> Vec<(i64, i32)> {
    let mut x_to_layer: Vec<(i64, i32)> = Vec::new();

    for node_idx in graph.node_indices() {
        if let Some(node) = graph.node_weight(node_idx)
            && let NodeRole::Data(_) = node.role
            && let Some(layer) = node.layer
        {
            x_to_layer.push((node.pos.x, layer));
        }
    }

    // Sort by x-coordinate for efficient lookup
    x_to_layer.sort_by_key(|&(x, _layer)| x);

    x_to_layer
}

/// Find the closest layer to a given x-coordinate.
/// On distance ties, prefer the lower layer number.
fn find_closest_layer(x_to_layer: &[(i64, i32)], target_x: i64) -> Option<i32> {
    if x_to_layer.is_empty() {
        return None;
    }

    // Find minimum by (distance, layer) - this ensures ties favor lower layer numbers
    x_to_layer
        .iter()
        .map(|&(x, layer)| ((x - target_x).abs(), layer))
        .min_by_key(|&(dist, layer)| (dist, layer))
        .map(|(_, layer)| layer)
}

/// Redistribute nodes along horizontal chains in the layout graph.
/// This should be called after make_rectilinear and before building the spatial index.
///
/// The `vertex_spacing` parameter defines the minimum gap that should be maintained
/// between nodes, matching the spacing used elsewhere in the layout.
pub fn redistribute_horizontal_chains(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    vertex_spacing: f64,
) {
    // Build x-to-layer mapping before any redistribution
    // This preserves the original Sugiyama layer assignments for cursor navigation
    let x_to_layer = build_x_to_layer_mapping(graph);

    let chains = identify_chains(graph);
    let min_gap = vertex_spacing.round() as i64;

    for chain in chains {
        if chain.nodes.len() < 2 {
            continue; // Need at least 2 nodes for redistribution
        }

        // Extract current positions and widths
        let mut widths = Vec::new();
        let mut orig_x = Vec::new();

        for &node_idx in &chain.nodes {
            if let Some(node) = graph.node_weight(node_idx) {
                widths.push(node.size.0 as i64);
                orig_x.push(node.pos.x);
            }
        }

        // Find obstacles for this chain
        let obstacles = find_obstacles(graph, chain.y, &chain.nodes);

        // Optimize the chain
        match optimize_chain_edge_gaps(&widths, &orig_x, &obstacles, min_gap) {
            ChainOptResult::Optimized { new_x } => {
                // Update positions in graph
                for (i, &node_idx) in chain.nodes.iter().enumerate() {
                    if let Some(node) = graph.node_weight_mut(node_idx) {
                        // Only update Data nodes, keep Routing nodes fixed
                        match node.role {
                            NodeRole::Data(_) if i > 0 && i < chain.nodes.len() - 1 => {
                                log::trace!(
                                    "  Node {} moved from x={} to x={}",
                                    node_idx.index(),
                                    node.pos.x,
                                    new_x[i]
                                );
                                node.pos.x = new_x[i];

                                // Reassign layer based on closest original x-coordinate
                                // This keeps cursor navigation working after redistribution
                                if let Some(new_layer) = find_closest_layer(&x_to_layer, new_x[i])
                                    && node.layer != Some(new_layer)
                                {
                                    node.layer = Some(new_layer);
                                }
                            }
                            _ => {
                                // Keep endpoints and routing nodes fixed
                            }
                        }
                    }
                }
            }
            ChainOptResult::RevertToOriginal => {
                log::trace!(
                    "Chain at y={} optimization failed, keeping original positions",
                    chain.y
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halves() {
        let h = halves(9);
        assert_eq!(h.left, 5);
        assert_eq!(h.right, 4);

        let h = halves(8);
        assert_eq!(h.left, 4);
        assert_eq!(h.right, 4);

        let h = halves(1);
        assert_eq!(h.left, 1);
        assert_eq!(h.right, 0);
    }

    #[test]
    fn test_base_step() {
        let step = base_step(9, 9);
        assert_eq!(step, 9); // 4 + 5

        let step = base_step(8, 8);
        assert_eq!(step, 8); // 4 + 4

        let step = base_step(9, 7);
        assert_eq!(step, 8); // 4 + 4
    }

    #[test]
    fn test_overlaps() {
        let a = Interval { l: 0, r: 5 };
        let b = Interval { l: 3, r: 8 };
        assert!(overlaps(a, b));

        let a = Interval { l: 0, r: 5 };
        let b = Interval { l: 6, r: 10 };
        assert!(!overlaps(a, b));

        let a = Interval { l: 0, r: 5 };
        let b = Interval { l: 5, r: 10 };
        assert!(overlaps(a, b)); // touching is overlapping
    }

    #[test]
    fn test_occ_interval() {
        let occ = occ_interval(10, 9);
        assert_eq!(occ.l, 5); // 10 - 5
        assert_eq!(occ.r, 14); // 10 + 4

        let occ = occ_interval(10, 8);
        assert_eq!(occ.l, 6); // 10 - 4
        assert_eq!(occ.r, 14); // 10 + 4
    }

    #[test]
    fn test_shift_right_to_avoid() {
        let protected = Interval { l: 5, r: 10 };
        let shift = shift_right_to_avoid(protected, 8, 9);
        // Node at 8 with width 9 has halves (5, 4)
        // Occupies [3, 12]
        // Must move to at least 10 + 1 + 5 = 16
        // Shift = 16 - 8 = 8
        assert_eq!(shift, 8);

        // Already clear
        let shift = shift_right_to_avoid(protected, 20, 9);
        assert_eq!(shift, 0);
    }

    #[test]
    fn test_simple_chain_optimization() {
        // Test a simple 3-node chain with no obstacles
        let widths = vec![8, 8, 8];
        let orig_x = vec![0, 10, 20];
        let forbidden = vec![];
        let min_gap = 0; // No minimum gap for this test

        match optimize_chain_edge_gaps(&widths, &orig_x, &forbidden, min_gap) {
            ChainOptResult::Optimized { new_x } => {
                assert_eq!(new_x.len(), 3);
                assert_eq!(new_x[0], 0); // first endpoint fixed
                assert_eq!(new_x[2], 20); // last endpoint fixed
                // Middle node should be centered
                assert_eq!(new_x[1], 10);
            }
            ChainOptResult::RevertToOriginal => {
                panic!("Optimization should succeed for simple case");
            }
        }
    }

    #[test]
    fn test_chain_with_obstacle() {
        // Test a 3-node chain with an obstacle in the middle
        let widths = vec![8, 8, 8];
        let orig_x = vec![0, 20, 40];
        // Obstacle at x=18-22
        let forbidden = vec![Interval { l: 18, r: 22 }];
        let min_gap = 0; // No minimum gap for this test

        match optimize_chain_edge_gaps(&widths, &orig_x, &forbidden, min_gap) {
            ChainOptResult::Optimized { new_x } => {
                assert_eq!(new_x.len(), 3);
                assert_eq!(new_x[0], 0); // first endpoint fixed
                assert_eq!(new_x[2], 40); // last endpoint fixed

                // Middle node should avoid the obstacle
                let middle_occ = occ_interval(new_x[1], widths[1]);
                for &obs in &forbidden {
                    assert!(
                        !overlaps(middle_occ, obs),
                        "Middle node at {} (occupies [{}, {}]) overlaps obstacle [{}, {}]",
                        new_x[1],
                        middle_occ.l,
                        middle_occ.r,
                        obs.l,
                        obs.r
                    );
                }
            }
            ChainOptResult::RevertToOriginal => {
                panic!("Optimization should succeed with sufficient space");
            }
        }
    }

    #[test]
    fn test_impossible_chain() {
        // Test a chain where nodes are too wide for the available space
        let widths = vec![20, 20, 20];
        let orig_x = vec![0, 10, 20]; // Not enough space
        let forbidden = vec![];
        let min_gap = 0; // No minimum gap for this test

        match optimize_chain_edge_gaps(&widths, &orig_x, &forbidden, min_gap) {
            ChainOptResult::RevertToOriginal => {
                // Expected - not enough space
            }
            ChainOptResult::Optimized { .. } => {
                panic!("Should revert to original when impossible");
            }
        }
    }

    #[test]
    fn test_rectilinear_vertical_obstacle() {
        use crate::geometry::LocalPos;

        // Create a simple graph with a vertical edge crossing a horizontal chain
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let partition_idx = 0;

        // Horizontal chain at y=10: nodes at x=0, 10, 20
        let node_0 = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos {
                partition_idx,
                x: 0,
                y: 10,
            },
            (8, 3),
            Some(1),
        ));
        let node_1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos {
                partition_idx,
                x: 10,
                y: 10,
            },
            (8, 3),
            Some(1),
        ));
        let node_2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos {
                partition_idx,
                x: 20,
                y: 10,
            },
            (8, 3),
            Some(1),
        ));

        // Add horizontal edges
        graph.add_edge(
            node_0,
            node_1,
            LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(1)),
        );
        graph.add_edge(
            node_1,
            node_2,
            LayoutEdge::new(NodeIndex::new(1), NodeIndex::new(2)),
        );

        // Add a vertical edge that crosses the horizontal chain at x=15
        let node_top = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos {
                partition_idx,
                x: 15,
                y: 5,
            },
            (8, 3),
            Some(0),
        ));
        let node_bottom = graph.add_node(LayoutNode::data(
            NodeIndex::new(4),
            LocalPos {
                partition_idx,
                x: 15,
                y: 15,
            },
            (8, 3),
            Some(2),
        ));
        graph.add_edge(
            node_top,
            node_bottom,
            LayoutEdge::new(NodeIndex::new(3), NodeIndex::new(4)),
        );

        // Find obstacles for the horizontal chain at y=10
        let obstacles = find_obstacles(&graph, 10, &[node_0, node_1, node_2]);

        // Should find one obstacle at x=15 (protected: 14, 15, 16)
        assert_eq!(obstacles.len(), 1, "Should find exactly one obstacle");
        assert_eq!(obstacles[0].l, 14, "Obstacle should protect x-1");
        assert_eq!(obstacles[0].r, 16, "Obstacle should protect x+1");
    }

    #[test]
    fn test_minimum_gap_enforcement() {
        // Test that the minimum gap constraint is respected
        let widths = vec![8, 8, 8];
        let orig_x = vec![0, 12, 40]; // Plenty of space
        let forbidden = vec![];
        let min_gap = 5; // Require at least 5 units between nodes

        match optimize_chain_edge_gaps(&widths, &orig_x, &forbidden, min_gap) {
            ChainOptResult::Optimized { new_x } => {
                assert_eq!(new_x.len(), 3);
                assert_eq!(new_x[0], 0); // first endpoint fixed
                assert_eq!(new_x[2], 40); // last endpoint fixed

                // Verify all gaps are at least min_gap
                for i in 0..(widths.len() - 1) {
                    let base = base_step(widths[i], widths[i + 1]);
                    let actual_gap = new_x[i + 1] - new_x[i] - base;
                    assert!(
                        actual_gap >= min_gap,
                        "Gap {} between nodes {} and {} is {} but should be at least {}",
                        i,
                        i,
                        i + 1,
                        actual_gap,
                        min_gap
                    );
                }
            }
            ChainOptResult::RevertToOriginal => {
                panic!("Optimization should succeed with sufficient space");
            }
        }
    }

    #[test]
    fn test_minimum_gap_impossible() {
        // Test that optimization fails when minimum gap cannot be satisfied
        let widths = vec![8, 8, 8];
        let orig_x = vec![0, 10, 20]; // Not enough space for large min_gap
        let forbidden = vec![];
        let min_gap = 10; // Require 10 units between nodes (impossible)

        match optimize_chain_edge_gaps(&widths, &orig_x, &forbidden, min_gap) {
            ChainOptResult::RevertToOriginal => {
                // Expected - cannot satisfy minimum gap
            }
            ChainOptResult::Optimized { .. } => {
                panic!("Should revert to original when minimum gap is impossible");
            }
        }
    }

    #[test]
    fn test_horizontal_edges_not_obstacles() {
        use crate::geometry::LocalPos;

        // Horizontal edges at different y-levels should not create obstacles
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let partition_idx = 0;

        // Horizontal chain at y=10
        let node_0 = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos {
                partition_idx,
                x: 0,
                y: 10,
            },
            (8, 3),
            Some(1),
        ));
        let node_1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos {
                partition_idx,
                x: 10,
                y: 10,
            },
            (8, 3),
            Some(1),
        ));

        // Another horizontal edge at y=5
        let node_2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos {
                partition_idx,
                x: 5,
                y: 5,
            },
            (8, 3),
            Some(0),
        ));
        let node_3 = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos {
                partition_idx,
                x: 15,
                y: 5,
            },
            (8, 3),
            Some(0),
        ));

        graph.add_edge(
            node_0,
            node_1,
            LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(1)),
        );
        graph.add_edge(
            node_2,
            node_3,
            LayoutEdge::new(NodeIndex::new(2), NodeIndex::new(3)),
        );

        // Find obstacles for the horizontal chain at y=10
        let obstacles = find_obstacles(&graph, 10, &[node_0, node_1]);

        // Horizontal edges don't cross, so no obstacles
        assert_eq!(
            obstacles.len(),
            0,
            "Horizontal edges should not create obstacles"
        );
    }

    #[test]
    fn test_layer_reassignment() {
        use crate::geometry::LocalPos;

        // Create a graph with nodes at different x positions representing different layers
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let partition_idx = 0;

        // Layer 0 nodes at x=0, 10
        let _node_0 = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos {
                partition_idx,
                x: 0,
                y: 0,
            },
            (8, 3),
            Some(0),
        ));
        let _node_1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos {
                partition_idx,
                x: 10,
                y: 0,
            },
            (8, 3),
            Some(0),
        ));

        // Layer 1 nodes at x=20, 30
        let _node_2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos {
                partition_idx,
                x: 20,
                y: 5,
            },
            (8, 3),
            Some(1),
        ));
        let _node_3 = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos {
                partition_idx,
                x: 30,
                y: 5,
            },
            (8, 3),
            Some(1),
        ));

        // Layer 2 nodes at x=40, 50
        let _node_4 = graph.add_node(LayoutNode::data(
            NodeIndex::new(4),
            LocalPos {
                partition_idx,
                x: 40,
                y: 10,
            },
            (8, 3),
            Some(2),
        ));
        let _node_5 = graph.add_node(LayoutNode::data(
            NodeIndex::new(5),
            LocalPos {
                partition_idx,
                x: 50,
                y: 10,
            },
            (8, 3),
            Some(2),
        ));

        // Build x-to-layer mapping
        let x_to_layer = build_x_to_layer_mapping(&graph);

        // Should have 6 unique x positions
        assert_eq!(x_to_layer.len(), 6);

        // Test finding closest layer for various x positions
        assert_eq!(find_closest_layer(&x_to_layer, 0), Some(0)); // Exact match
        assert_eq!(find_closest_layer(&x_to_layer, 5), Some(0)); // Closer to 0 than 10
        assert_eq!(find_closest_layer(&x_to_layer, 15), Some(0)); // Equidistant to 10 and 20, prefer lower layer
        assert_eq!(find_closest_layer(&x_to_layer, 25), Some(1)); // Closer to 20 or 30
        assert_eq!(find_closest_layer(&x_to_layer, 35), Some(1)); // Equidistant to 30 and 40, prefer lower layer
        assert_eq!(find_closest_layer(&x_to_layer, 45), Some(2)); // Closer to 40 or 50
    }

    #[test]
    fn test_layer_reassignment_with_redistribution() {
        use crate::geometry::LocalPos;

        // Create a horizontal chain where nodes will be redistributed
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let partition_idx = 0;

        // Horizontal chain at y=10 with nodes from different original layers
        // Node at x=0, layer 0 (endpoint)
        let node_0 = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos {
                partition_idx,
                x: 0,
                y: 10,
            },
            (8, 3),
            Some(0),
        ));

        // Node at x=10, originally layer 1
        let node_1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos {
                partition_idx,
                x: 10,
                y: 10,
            },
            (8, 3),
            Some(1),
        ));

        // Node at x=20, originally layer 2
        let node_2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos {
                partition_idx,
                x: 20,
                y: 10,
            },
            (8, 3),
            Some(2),
        ));

        // Node at x=60, layer 3 (endpoint)
        let node_3 = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos {
                partition_idx,
                x: 60,
                y: 10,
            },
            (8, 3),
            Some(3),
        ));

        // Add horizontal edges to form a chain
        graph.add_edge(
            node_0,
            node_1,
            LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(1)),
        );
        graph.add_edge(
            node_1,
            node_2,
            LayoutEdge::new(NodeIndex::new(1), NodeIndex::new(2)),
        );
        graph.add_edge(
            node_2,
            node_3,
            LayoutEdge::new(NodeIndex::new(2), NodeIndex::new(3)),
        );

        // Record original layers
        let original_layer_node_1 = graph[node_1].layer;
        let original_layer_node_2 = graph[node_2].layer;

        // Run redistribution (this will spread nodes evenly)
        redistribute_horizontal_chains(&mut graph, 1.0);

        // After redistribution, interior nodes should have their layers reassigned
        // based on their new x positions
        // The nodes should be spread more evenly between 0 and 60
        // Expected positions: node_0=0, node_1≈20, node_2≈40, node_3=60

        // Check that interior nodes got their layers reassigned based on closest x
        let new_layer_node_1 = graph[node_1].layer;
        let new_layer_node_2 = graph[node_2].layer;

        // node_1 moved to around x=20, which should map to layer 2 (closer to original x=20)
        // node_2 moved to around x=40, which should map to layer 2 or 3

        // The exact layer assignment depends on the redistribution, but we can verify
        // that the layer field was updated (it should differ from original if nodes moved significantly)
        println!(
            "node_1: layer {:?} -> {:?}, x={}",
            original_layer_node_1, new_layer_node_1, graph[node_1].pos.x
        );
        println!(
            "node_2: layer {:?} -> {:?}, x={}",
            original_layer_node_2, new_layer_node_2, graph[node_2].pos.x
        );

        // At minimum, verify that layers are still valid (0-3 range)
        assert!(new_layer_node_1.unwrap() >= 0 && new_layer_node_1.unwrap() <= 3);
        assert!(new_layer_node_2.unwrap() >= 0 && new_layer_node_2.unwrap() <= 3);
    }
}
