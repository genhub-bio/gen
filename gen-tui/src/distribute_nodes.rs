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

/// Asymmetric halves for a node dimension
///   - `lo`: extent in the lower/negative direction from center
///     For odd sizes, `lo` gets the extra cell.
///   - `hi`: extent in the higher/positive direction from center
#[derive(Clone, Copy, Debug)]
pub struct Halves {
    pub lo: i64, // lower/negative direction (left for x, down for y)
    pub hi: i64, // higher/positive direction (right for x, up for y)
}

/// An inclusive interval [start, end] representing occupied cells.
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub start: i64, // inclusive
    pub end: i64,   // inclusive
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

/// Calculate asymmetric halves for a given dimension size.
/// For odd sizes, the lower half (lo) is larger.
/// Works for both width (x-axis) and height (y-axis).
///
/// Examples:
/// - size=9 -> lo=5, hi=4
/// - size=8 -> lo=4, hi=4
/// - size=1 -> lo=1, hi=0
pub fn halves(size: i64) -> Halves {
    Halves {
        lo: (size + 1) / 2,
        hi: size / 2,
    }
}

/// Calculate the base center-to-center distance between two adjacent nodes
/// when they are touching (no gap).
pub fn base_step(width_i: i64, width_j: i64) -> i64 {
    let hi = halves(width_i);
    let hj = halves(width_j);
    hi.hi + hj.lo
}

/// Check if two intervals overlap.
fn overlaps(a: Interval, b: Interval) -> bool {
    a.start <= b.end && b.start <= a.end
}

/// Calculate the occupied interval for a node given its center x and width.
fn occ_interval(center_x: i64, width: i64) -> Interval {
    let h = halves(width);
    Interval {
        start: center_x - h.lo,
        end: center_x + h.hi,
    }
}

/// Calculate the minimal rightward shift needed to move a node's occupied interval
/// past a forbidden interval.
fn shift_right_to_avoid(protected: Interval, center_x: i64, width: i64) -> i64 {
    let h = halves(width);
    let min_center = protected.end + 1 + h.lo;
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
///
/// Obstacles come from two sources:
/// 1. Vertical edges that cross the horizontal line at y
/// 2. Nodes whose bounding boxes intersect with the chain's envelope
///
/// The chain envelope is defined as the smallest rectangle spanning:
/// - x: from leftmost to rightmost chain node
/// - y: the chain's y-coordinate ± node_separation (to account for node heights)
///
/// For each obstacle, we protect the area with margins based on node_separation.
fn find_obstacles(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    chain_y: i64,
    chain_nodes: &[NodeIndex],
    node_separation: i64,
) -> Vec<Interval> {
    let mut obstacles = Vec::new();

    // Build set of chain node indices for quick lookup
    let chain_node_set: HashSet<NodeIndex> = chain_nodes.iter().copied().collect();

    // Calculate the chain envelope (exact bounding box of chain nodes)
    let (chain_x_min, chain_x_max, envelope_y_min, envelope_y_max) = {
        let mut x_min = i64::MAX;
        let mut x_max = i64::MIN;
        let mut max_half_down = 0i64; // Maximum downward extent from center
        let mut max_half_up = 0i64; // Maximum upward extent from center

        for &node_idx in chain_nodes {
            if let Some(node) = graph.node_weight(node_idx) {
                x_min = x_min.min(node.pos.x);
                x_max = x_max.max(node.pos.x);

                // Calculate how far this node extends above/below its center
                let h = halves(node.size.1 as i64);
                max_half_down = max_half_down.max(h.lo); // downward in y
                max_half_up = max_half_up.max(h.hi); // upward in y
            }
        }

        // Envelope is just the actual extent of chain nodes
        // node_separation is applied later when creating exclusion zones
        (x_min, x_max, chain_y - max_half_down, chain_y + max_half_up)
    };

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
                        start: x1 - 1,
                        end: x1 + 1,
                    });
                }
            }
            // Horizontal edges (y1 == y2) don't cross other horizontal lines
            // Diagonal edges shouldn't exist in rectilinear layout
        }
    }

    // Find all nodes (not in the chain) whose bounding boxes intersect the chain envelope
    for node_idx in graph.node_indices() {
        // Skip nodes that are part of the chain
        if chain_node_set.contains(&node_idx) {
            continue;
        }

        if let Some(node) = graph.node_weight(node_idx) {
            // Calculate node's bounding box using halves for both dimensions
            let node_h_width = halves(node.size.0 as i64);
            let node_x_min = node.pos.x - node_h_width.lo;
            let node_x_max = node.pos.x + node_h_width.hi;

            let node_h_height = halves(node.size.1 as i64);
            let node_y_min = node.pos.y - node_h_height.lo; // downward in y
            let node_y_max = node.pos.y + node_h_height.hi; // upward in y

            // Check if node's bounding box intersects the chain envelope
            let x_intersects = node_x_max >= chain_x_min && node_x_min <= chain_x_max;
            let y_intersects = node_y_max >= envelope_y_min && node_y_min <= envelope_y_max;

            if x_intersects && y_intersects {
                // Block the entire column at this node's x position ± node_separation
                obstacles.push(Interval {
                    start: node.pos.x - node_separation,
                    end: node.pos.x + node_separation,
                });
            }
        }
    }

    // Merge overlapping obstacles
    if obstacles.is_empty() {
        return obstacles;
    }

    obstacles.sort_by_key(|i| i.start);
    let mut merged = vec![obstacles[0]];

    for obstacle in obstacles.iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if obstacle.start <= last.end + 1 {
            // Overlapping or adjacent - merge
            last.end = last.end.max(obstacle.end);
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
                if f.start > occ.end {
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
            if f.start > occ.end {
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
        let obstacles = find_obstacles(graph, chain.y, &chain.nodes, min_gap);

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

                                // Snap layer to nearest pre-redistribution layer.
                                // The visual position moves freely for aesthetics, but the layer
                                // field (used for cursor navigation) is anchored to the closest
                                // original Sugiyama column. Stacking partners that don't straddle
                                // a column midpoint will share the same layer.
                                if let Some(snapped_layer) =
                                    find_closest_layer(&x_to_layer, new_x[i])
                                {
                                    node.layer = Some(snapped_layer);
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

/// The axis along which [`compress_dead_space`] operates.
#[derive(Clone, Copy, Debug)]
enum CompressAxis {
    X,
    Y,
}

/// Compute the cells occupied by a node along one axis, matching the
/// asymmetric extents of `BigRect::from_center_and_size`: even sizes extend
/// one cell further in the positive direction.
fn occupied_extent(center: i64, size: i64) -> Interval {
    Interval {
        start: center - ((size + 1) / 2 - 1),
        end: center + size / 2,
    }
}

/// Remove dead space from the layout by shrinking oversized gaps between
/// occupied bands down to `vertex_spacing`, on both axes.
///
/// The Brandes-Köpf coordinate assignment averages four extreme layouts. For
/// dense graphs (e.g. layers connected all-to-all) the four layouts disagree
/// strongly on block placement, and the average leaves large bands of empty
/// space between nodes that no individual layout had. This pass scans each
/// axis for bands of cells not covered by any node's bounding box and shrinks
/// them to the configured spacing. Gaps at or below the spacing are never
/// touched, and nothing is ever moved apart, so legitimate tight routing
/// tracks are preserved.
///
/// Edges are drawn as segments between node positions, so shifting whole
/// bands keeps the layout rectilinear: vertical and horizontal runs through a
/// removed gap simply get shorter.
///
/// Note: this pass is complementary to, not redundant with, the edge router's
/// `compress_graph` (see [`crate::edge_router::layout_graph_process`]). That
/// pass is a local normalizer: per layer pair, label-aware, and bidirectional,
/// meaning it may *expand* spacing so routing channels clear wide node labels.
/// This pass is a global, shrink-only cleaner: it removes slack spanning the
/// whole partition (e.g. Brandes-Köpf averaging artifacts) that no single
/// layer pair can see. Removing either reintroduces distortions: without
/// `compress_graph`, junctions can end up inside wide nodes; without this
/// pass, dense graphs keep large dead bands.
///
/// In the layout pipeline this runs both before and after
/// [`redistribute_horizontal_chains`]: before, because once nodes are
/// re-centered along their chains they straddle inter-column gaps and block
/// compression; after, because redistribution can vacate bands (e.g. a wide
/// node moving out of an inflated column), reopening dead space.
pub fn compress_dead_space(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    vertex_spacing: f64,
) {
    let min_gap = (vertex_spacing.round() as i64).max(1);
    compress_axis(graph, min_gap, CompressAxis::X);
    compress_axis(graph, min_gap, CompressAxis::Y);
}

fn compress_axis(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    min_gap: i64,
    axis: CompressAxis,
) {
    let extent = |node: &LayoutNode| match axis {
        CompressAxis::X => occupied_extent(node.pos.x, node.size.0 as i64),
        CompressAxis::Y => occupied_extent(node.pos.y, node.size.1 as i64),
    };

    let mut intervals: Vec<Interval> = graph.node_weights().map(extent).collect();
    if intervals.is_empty() {
        return;
    }
    intervals.sort_by_key(|interval| interval.start);

    // Merge intervals whose gap is at most min_gap: those gaps cannot shrink,
    // so the band boundary carries no shift change.
    let mut bands: Vec<Interval> = vec![intervals[0]];
    for interval in &intervals[1..] {
        let last = bands.last_mut().unwrap();
        if interval.start <= last.end + 1 + min_gap {
            last.end = last.end.max(interval.end);
        } else {
            bands.push(*interval);
        }
    }

    // Accumulate the leftward shift for each band: every gap larger than
    // min_gap is reduced to exactly min_gap.
    let mut shifts: Vec<i64> = vec![0; bands.len()];
    for i in 1..bands.len() {
        let gap = bands[i].start - bands[i - 1].end - 1;
        shifts[i] = shifts[i - 1] + (gap - min_gap);
    }

    if *shifts.last().unwrap() == 0 {
        return;
    }

    for node_idx in graph.node_indices().collect::<Vec<_>>() {
        let node = &mut graph[node_idx];
        let center = match axis {
            CompressAxis::X => node.pos.x,
            CompressAxis::Y => node.pos.y,
        };

        // Find the band containing this node's center
        let band_idx = bands.partition_point(|band| band.start <= center) - 1;

        match axis {
            CompressAxis::X => node.pos.x -= shifts[band_idx],
            CompressAxis::Y => node.pos.y -= shifts[band_idx],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halves() {
        let h = halves(9);
        assert_eq!(h.lo, 5);
        assert_eq!(h.hi, 4);

        let h = halves(8);
        assert_eq!(h.lo, 4);
        assert_eq!(h.hi, 4);

        let h = halves(1);
        assert_eq!(h.lo, 1);
        assert_eq!(h.hi, 0);
    }

    #[test]
    fn test_base_step() {
        let step = base_step(9, 9);
        assert_eq!(step, 9); // hi + lo = 4 + 5

        let step = base_step(8, 8);
        assert_eq!(step, 8); // hi + lo = 4 + 4

        let step = base_step(9, 7);
        assert_eq!(step, 8); // hi(9) + lo(7) = 4 + 4
    }

    #[test]
    fn test_overlaps() {
        let a = Interval { start: 0, end: 5 };
        let b = Interval { start: 3, end: 8 };
        assert!(overlaps(a, b));

        let a = Interval { start: 0, end: 5 };
        let b = Interval { start: 6, end: 10 };
        assert!(!overlaps(a, b));

        let a = Interval { start: 0, end: 5 };
        let b = Interval { start: 5, end: 10 };
        assert!(overlaps(a, b)); // touching is overlapping
    }

    #[test]
    fn test_occ_interval() {
        let occ = occ_interval(10, 9);
        assert_eq!(occ.start, 5); // 10 - lo(9) = 10 - 5
        assert_eq!(occ.end, 14); // 10 + hi(9) = 10 + 4

        let occ = occ_interval(10, 8);
        assert_eq!(occ.start, 6); // 10 - lo(8) = 10 - 4
        assert_eq!(occ.end, 14); // 10 + hi(8) = 10 + 4
    }

    #[test]
    fn test_shift_right_to_avoid() {
        let protected = Interval { start: 5, end: 10 };

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
        let forbidden = vec![Interval { start: 18, end: 22 }];
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
                        middle_occ.start,
                        middle_occ.end,
                        obs.start,
                        obs.end
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
        let node_separation = 1;
        let obstacles = find_obstacles(&graph, 10, &[node_0, node_1, node_2], node_separation);

        // Should find one obstacle at x=15 (protected: 14, 15, 16)
        assert_eq!(obstacles.len(), 1, "Should find exactly one obstacle");
        assert_eq!(obstacles[0].start, 14, "Obstacle should protect x-1");
        assert_eq!(obstacles[0].end, 16, "Obstacle should protect x+1");
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

        // Nodes far from the chain should not create obstacles
        // (no vertical edge crossings, no envelope intersection)
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

        // Another horizontal edge at y=0 (far enough to not intersect envelope)
        // Chain at y=10 with height=3 and node_separation=1 creates envelope [10-3, 10+3] = [7, 13]
        // Nodes at y=0 with height=3 span [-1, 1], which does not intersect [7, 13]
        let node_2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos {
                partition_idx,
                x: 5,
                y: 0,
            },
            (8, 3),
            Some(0),
        ));
        let node_3 = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos {
                partition_idx,
                x: 15,
                y: 0,
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
        let node_separation = 1;
        let obstacles = find_obstacles(&graph, 10, &[node_0, node_1], node_separation);

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

    #[test]
    fn test_node_intersection_creates_exclusion_zone() {
        use crate::geometry::LocalPos;

        // Test that nodes whose bounding boxes intersect with the chain envelope
        // create exclusion zones, preventing chain nodes from being placed there
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let partition_idx = 0;

        // Horizontal chain at y=10: nodes at x=0, 30, 60
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
        let node_1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos {
                partition_idx,
                x: 30,
                y: 10,
            },
            (8, 3),
            Some(1),
        ));
        let node_2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos {
                partition_idx,
                x: 60,
                y: 10,
            },
            (8, 3),
            Some(2),
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

        // Add a node above the chain that intersects the envelope
        // Chain is at y=10, chain nodes have height=3
        // halves(3) = {lo: 2, hi: 1}, so envelope spans [10-2, 10+1] = [8, 11]
        // This node at y=9 with height=3: halves(3) = {lo: 2, hi: 1}
        // Node spans [9-2, 9+1] = [7, 10], which intersects envelope [8, 11]
        let _blocking_node = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos {
                partition_idx,
                x: 20, // Between chain nodes 0 and 1
                y: 9,  // Above but overlapping with the chain
            },
            (6, 3), // Width=6, Height=3
            Some(0),
        ));

        // Add another node below the chain
        // This node at y=11 with height=3 spans [11-2, 11+1] = [9, 12]
        // Intersects envelope [8, 11]
        let _blocking_node_2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(4),
            LocalPos {
                partition_idx,
                x: 40, // Between chain nodes 1 and 2
                y: 11, // Below but overlapping with the chain
            },
            (6, 3),
            Some(2),
        ));

        // Find obstacles with node_separation=3
        let node_separation = 3;
        let obstacles = find_obstacles(&graph, 10, &[node_0, node_1, node_2], node_separation);

        // Should find two obstacles - one for each blocking node
        assert!(
            obstacles.len() >= 2,
            "Should find at least 2 obstacles from intersecting nodes, found {}",
            obstacles.len()
        );

        // Verify that the blocking nodes created exclusion zones
        // blocking_node at x=20 should create zone [20-3, 20+3] = [17, 23]
        // blocking_node_2 at x=40 should create zone [40-3, 40+3] = [37, 43]
        let has_obstacle_at_20 = obstacles.iter().any(|obs| obs.start <= 20 && 20 <= obs.end);
        let has_obstacle_at_40 = obstacles.iter().any(|obs| obs.start <= 40 && 40 <= obs.end);

        assert!(
            has_obstacle_at_20,
            "Should have exclusion zone around x=20 for blocking_node"
        );
        assert!(
            has_obstacle_at_40,
            "Should have exclusion zone around x=40 for blocking_node_2"
        );
    }

    /// Demonstrates that nodes sharing a Sugiyama rank (vertical stacking partners) can
    /// end up at different X coordinates after redistribution, splitting them into
    /// separate CroppedGraph layers and breaking vertical navigation.
    ///
    /// Graph: 0→{1,2}, 1→3, 2→{3,4}, 3→5, 4→5  (subcombinatorial_dag)
    ///
    /// Sugiyama ranks: 0=rank0, 1=rank1, 2=rank1, 3=rank2, 4=rank2, 5=rank3
    /// N3 and N4 are vertical stacking partners at rank 2.
    ///
    /// After redistribution the two horizontal chains:
    ///   upper: [R, N1, N3, R]  and  lower: [R, N2, N4, R]
    /// are optimized independently, so N3 and N4 can end up at different X values.
    #[test]
    fn test_stacking_partners_diverge_after_redistribution() {
        use petgraph::{Undirected, stable_graph::StableGraph};

        use crate::geometry::LocalPos;

        // Manually build the post-edge-routing layout for subcombinatorial_dag.
        // Positions below match what Sugiyama+edge-routing produces (all same rank
        // nodes share the same X before redistribution):
        //
        //   X:  0    10    20    30
        //   Y:  top chain:   R(0,5) - N1(10,5) - N3(20,5) - R(30,5)
        //       bottom chain: R(0,-5) - N2(10,-5) - N4(20,-5) - R(30,-5)
        //
        // N3 and N4 both start at X=20 (same Sugiyama rank=2).

        let partition_idx = 0usize;
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // Upper chain routing endpoints (fixed)
        let r_top_left = graph.add_node(LayoutNode::routing(
            LocalPos::new_xy(partition_idx, 0, 5),
            (1, 1),
        ));
        let r_top_right = graph.add_node(LayoutNode::routing(
            LocalPos::new_xy(partition_idx, 30, 5),
            (1, 1),
        ));

        // Lower chain routing endpoints (fixed)
        let r_bot_left = graph.add_node(LayoutNode::routing(
            LocalPos::new_xy(partition_idx, 0, -5),
            (1, 1),
        ));
        let r_bot_right = graph.add_node(LayoutNode::routing(
            LocalPos::new_xy(partition_idx, 30, -5),
            (1, 1),
        ));

        // N1 (rank=1) on upper chain at X=10 — note: only node at X=10,Y=5
        let n1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos::new_xy(partition_idx, 10, 5),
            (5, 3),
            Some(1),
        ));
        // N2 (rank=1) on lower chain at X=10 — vertical stacking partner of N1
        let n2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos::new_xy(partition_idx, 10, -5),
            (5, 3),
            Some(1),
        ));
        // N3 (rank=2) on upper chain at X=20 — vertical stacking partner of N4
        let n3 = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos::new_xy(partition_idx, 20, 5),
            (5, 3),
            Some(2),
        ));
        // N4 (rank=2) on lower chain at X=20 — vertical stacking partner of N3
        let n4 = graph.add_node(LayoutNode::data(
            NodeIndex::new(4),
            LocalPos::new_xy(partition_idx, 20, -5),
            (5, 3),
            Some(2),
        ));

        // Upper horizontal chain edges
        graph.add_edge(
            r_top_left,
            n1,
            LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(1)),
        );
        graph.add_edge(
            n1,
            n3,
            LayoutEdge::new(NodeIndex::new(1), NodeIndex::new(3)),
        );
        graph.add_edge(
            n3,
            r_top_right,
            LayoutEdge::new(NodeIndex::new(3), NodeIndex::new(5)),
        );

        // Lower horizontal chain edges
        graph.add_edge(
            r_bot_left,
            n2,
            LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(2)),
        );
        graph.add_edge(
            n2,
            n4,
            LayoutEdge::new(NodeIndex::new(2), NodeIndex::new(4)),
        );
        graph.add_edge(
            n4,
            r_bot_right,
            LayoutEdge::new(NodeIndex::new(4), NodeIndex::new(5)),
        );

        // Verify precondition: N3 and N4 start at the same X (same Sugiyama rank)
        assert_eq!(
            graph[n3].pos.x, graph[n4].pos.x,
            "N3 and N4 must start at same X"
        );
        let original_x = graph[n3].pos.x;
        println!(
            "Before redistribution: N3.x={}, N4.x={}",
            graph[n3].pos.x, graph[n4].pos.x
        );

        // Run redistribution with a non-trivial span so nodes actually move
        redistribute_horizontal_chains(&mut graph, 2.0);

        let n3_x = graph[n3].pos.x;
        let n4_x = graph[n4].pos.x;
        println!("After redistribution:  N3.x={}, N4.x={}", n3_x, n4_x);
        println!(
            "N3.layer={:?}, N4.layer={:?}",
            graph[n3].layer, graph[n4].layer
        );

        // Both moved from their original position (redistribution did something)
        println!(
            "N3 moved: {}, N4 moved: {}",
            n3_x != original_x,
            n4_x != original_x
        );

        // X positions may legitimately differ after redistribution — that is the whole point
        // of the aesthetic spacing. What must be equal is the *layer* field: both N3 and N4
        // started at X=20 (Sugiyama rank 2), so find_closest_layer maps their new positions
        // back to rank 2 as long as they don't straddle the midpoint to rank 1.
        let n3_layer = graph[n3].layer;
        let n4_layer = graph[n4].layer;
        assert_eq!(
            n3_layer, n4_layer,
            "Stacking partners N3 and N4 must share the same layer after redistribution \
             (N3.x={} layer={:?}, N4.x={} layer={:?})",
            n3_x, n3_layer, n4_x, n4_layer
        );
    }

    #[test]
    fn test_occupied_extent_matches_rect_convention() {
        // Odd size: symmetric
        let extent = occupied_extent(10, 3);
        assert_eq!((extent.start, extent.end), (9, 11));

        // Even size: extends one cell further in the positive direction
        let extent = occupied_extent(10, 4);
        assert_eq!((extent.start, extent.end), (9, 12));

        // Unit size occupies a single cell
        let extent = occupied_extent(10, 1);
        assert_eq!((extent.start, extent.end), (10, 10));
    }

    #[test]
    fn test_compress_dead_space_shrinks_oversized_gap() {
        use crate::geometry::LocalPos;

        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // Two rows of height 3 with centers 20 apart (occupied gap of 17 cells)
        // and two columns of width 5 with centers 30 apart.
        let top_left = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos::new_xy(0, 0, 0),
            (5, 3),
            Some(0),
        ));
        let bottom_right = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos::new_xy(0, 30, 20),
            (5, 3),
            Some(1),
        ));

        compress_dead_space(&mut graph, 1.0);

        // X: occupied bands [-2,2] and [28,32] -> gap 25 shrinks to 1,
        // so the second center moves from 30 to 6.
        // Y: occupied bands [-1,1] and [19,21] -> gap 17 shrinks to 1,
        // so the second center moves from 20 to 4.
        assert_eq!(graph[top_left].pos.x, 0);
        assert_eq!(graph[top_left].pos.y, 0);
        assert_eq!(graph[bottom_right].pos.x, 6);
        assert_eq!(graph[bottom_right].pos.y, 4);
    }

    #[test]
    fn test_compress_dead_space_preserves_tight_gaps() {
        use crate::geometry::LocalPos;

        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // Rows already at minimum spacing (centers 4 apart, height 3, gap 1)
        let positions = [(0, 0), (0, 4), (0, 8)];
        let nodes: Vec<_> = positions
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                graph.add_node(LayoutNode::data(
                    NodeIndex::new(i),
                    LocalPos::new_xy(0, x, y),
                    (5, 3),
                    Some(0),
                ))
            })
            .collect();

        compress_dead_space(&mut graph, 1.0);

        for (node_idx, &(x, y)) in nodes.iter().zip(positions.iter()) {
            assert_eq!(graph[*node_idx].pos.x, x, "tight layout must not move");
            assert_eq!(graph[*node_idx].pos.y, y, "tight layout must not move");
        }
    }

    /// Tests that a node's *layer* field is updated when redistribution shifts it
    /// far enough to cross the midpoint between two original Sugiyama columns.
    ///
    /// Setup: two columns — x=10 (layer 1) and x=50 (layer 2) — established by
    /// anchor nodes that sit at y=−30 and never participate in any chain.
    /// A single-node chain [R(0) − N(10) − R(80)] is redistributed; the optimizer
    /// centres N at x≈40, which is past the column midpoint (30) → layer snaps to 2.
    #[test]
    fn test_layer_changes_when_node_crosses_column_midpoint() {
        use petgraph::{Undirected, stable_graph::StableGraph};

        use crate::geometry::LocalPos;

        let partition_idx = 0usize;
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // Anchor nodes at y=−30 establish the column→layer mapping.
        // Their y-envelope [−32, −29] never intersects the chain at y=0, so they
        // are invisible to find_obstacles and don't affect redistribution.
        let _anchor_l1 = graph.add_node(LayoutNode::data(
            NodeIndex::new(100),
            LocalPos::new_xy(partition_idx, 10, -30),
            (3, 3),
            Some(1),
        ));
        let _anchor_l2 = graph.add_node(LayoutNode::data(
            NodeIndex::new(101),
            LocalPos::new_xy(partition_idx, 50, -30),
            (3, 3),
            Some(2),
        ));

        // Single-node horizontal chain at y=0, spanning x in [0, 80].
        // N starts at x=10 (column 1). The optimizer will centre it at x=40.
        let r_left = graph.add_node(LayoutNode::routing(
            LocalPos::new_xy(partition_idx, 0, 0),
            (1, 1),
        ));
        let n = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos::new_xy(partition_idx, 10, 0),
            (5, 3),
            Some(1),
        ));
        let r_right = graph.add_node(LayoutNode::routing(
            LocalPos::new_xy(partition_idx, 80, 0),
            (1, 1),
        ));

        graph.add_edge(
            r_left,
            n,
            LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(1)),
        );
        graph.add_edge(
            n,
            r_right,
            LayoutEdge::new(NodeIndex::new(1), NodeIndex::new(2)),
        );

        assert_eq!(graph[n].pos.x, 10, "N starts at x=10");
        assert_eq!(graph[n].layer, Some(1), "N starts at layer 1");

        redistribute_horizontal_chains(&mut graph, 1.0);

        let new_x = graph[n].pos.x;
        let new_layer = graph[n].layer;
        println!("After redistribution: N.x={new_x}, N.layer={new_layer:?}");

        // N should have moved significantly from its original position.
        assert_ne!(new_x, 10, "N must move from original x=10");

        // The column midpoint between x=10 and x=50 is 30.
        // N at x=40 is past that midpoint, so find_closest_layer returns layer 2.
        assert!(
            new_x > 30,
            "N (x={new_x}) should have crossed the column midpoint at x=30"
        );
        assert_eq!(
            new_layer,
            Some(2),
            "Layer must snap to 2 (closest column x=50) when N moves to x={new_x}"
        );
    }
}
