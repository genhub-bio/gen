//! Global layout compaction for layout graphs.
//!
//! Rebuilds routed coordinates with size-aware separation constraints.
//!
//! Aligned nodes move as rigid segments, and each segment is centered within its feasible range.

use std::collections::HashMap;

use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

use crate::{
    compaction::{AxisPlacement, Constraint, compact_axis_with_centering},
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

/// Extents on either side of a discrete center; `lo` receives the extra cell for odd sizes.
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

/// Split a discrete size into lower and upper extents.
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

/// Compute the cells occupied by a node along one axis, matching the
/// asymmetric extents of `BigRect::from_center_and_size`: even sizes extend
/// one cell further in the positive direction.
fn occupied_extent(center: i64, size: i64) -> Interval {
    Interval {
        start: center - ((size + 1) / 2 - 1),
        end: center + size / 2,
    }
}

/// The axis being solved. The "perpendicular" direction is the other axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

impl Axis {
    fn pos(self, node: &LayoutNode) -> i64 {
        match self {
            Axis::X => node.pos.x,
            Axis::Y => node.pos.y,
        }
    }

    fn perp_pos(self, node: &LayoutNode) -> i64 {
        match self {
            Axis::X => node.pos.y,
            Axis::Y => node.pos.x,
        }
    }

    fn size(self, node: &LayoutNode) -> i64 {
        match self {
            Axis::X => node.size.0 as i64,
            Axis::Y => node.size.1 as i64,
        }
    }

    fn perp_size(self, node: &LayoutNode) -> i64 {
        match self {
            Axis::X => node.size.1 as i64,
            Axis::Y => node.size.0 as i64,
        }
    }

    fn set_pos(self, node: &mut LayoutNode, value: i64) {
        match self {
            Axis::X => node.pos.x = value,
            Axis::Y => node.pos.y = value,
        }
    }
}

/// A maximal rigid group of nodes sharing one coordinate on the solved axis:
/// for x, a column connected by vertical edges; for y, a row connected by
/// horizontal edges. The whole segment moves as one constraint item.
struct AxisSegment {
    /// Current coordinate on the solved axis, used to direct constraints.
    pos: i64,
}

/// One collision-avoidance obstacle: either a real graph node or a
/// pass-through wire (an edge lying entirely within one segment, spanning
/// the perpendicular cells between its two same-coordinate endpoints).
/// Always carries its own real extent — never a segment aggregate — so a
/// wire only claims the specific perpendicular cells it actually occupies,
/// and a node only claims its own size, never a whole row's tallest member.
struct Obstacle {
    seg_id: usize,
    axis_pos: i64,
    /// Center coordinate captured before iterative compaction starts.
    incoming_axis_pos: i64,
    /// Size on the solved axis: the node's real size, or 1 for a wire (an
    /// edge line is one cell thick).
    axis_size: i64,
    /// Extent on the perpendicular axis: the node's own occupied extent, or
    /// the wire's own line interval (not the whole segment's span).
    perp: Interval,
    /// True for data and wormhole nodes, which share the data-node spacing policies.
    /// Pass-through wires, routing nodes, and pin nodes are routing obstacles.
    is_data: bool,
}

/// Plain union-find over dense node ids, used to group nodes into segments.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] != i {
            let root = self.find(self.parent[i]);
            self.parent[i] = root;
        }
        self.parent[i]
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Solved placement for one axis: the segment id of every node plus the
/// solver output for all segments.
struct AxisSolution {
    seg_of: HashMap<NodeIndex, usize>,
    placement: AxisPlacement,
}

/// Transforms an incoming free-space gap into the gap to enforce.
pub type GapSizer = fn(u64) -> u64;

/// Gap transforms for both axes, split by the kinds of obstacle in a pair.
///
/// Each function receives the pair's incoming free-space gap, captured before iterative
/// compaction, and returns the gap to enforce. `data_data_x`/`data_data_y` cover pairs of
/// data nodes; `data_routing_*`/`routing_routing_*` cover pairs touching routing, pin,
/// or wire obstacles. Wormhole nodes count as data nodes for policy selection.
#[derive(Clone, Copy)]
pub struct GapSizes {
    pub data_data_x: GapSizer,
    pub data_data_y: GapSizer,
    pub data_routing_x: GapSizer,
    pub data_routing_y: GapSizer,
    pub routing_routing_x: GapSizer,
    pub routing_routing_y: GapSizer,
}

impl Default for GapSizes {
    fn default() -> Self {
        Self {
            data_data_x: |_| 1,
            data_data_y: |_| 1,
            data_routing_x: |_| 1,
            data_routing_y: |_| 0,
            routing_routing_x: |_| 1,
            routing_routing_y: |_| 0,
        }
    }
}

impl GapSizes {
    fn between(&self, axis: Axis, a_is_data: bool, b_is_data: bool) -> GapSizer {
        match (axis, a_is_data && b_is_data, a_is_data || b_is_data) {
            (Axis::X, true, _) => self.data_data_x,
            (Axis::X, false, true) => self.data_routing_x,
            (Axis::X, false, false) => self.routing_routing_x,
            (Axis::Y, true, _) => self.data_data_y,
            (Axis::Y, false, true) => self.data_routing_y,
            (Axis::Y, false, false) => self.routing_routing_y,
        }
    }
}

/// Center-to-center minimum for an ordered obstacle pair.
fn transform_distance(
    gap_sizer: GapSizer,
    lower_incoming_pos: i64,
    lower_size: i64,
    upper_incoming_pos: i64,
    upper_size: i64,
) -> i64 {
    let base = base_step(lower_size, upper_size);
    let base_gap = u64::try_from(base).expect("should have a nonnegative base step");
    let incoming_center_distance = lower_incoming_pos.abs_diff(upper_incoming_pos);
    let suggested_gap = incoming_center_distance.saturating_sub(base_gap);
    let transformed_gap = gap_sizer(suggested_gap);
    let transformed_gap =
        i64::try_from(transformed_gap).expect("should fit transformed gap in layout coordinates");
    base.checked_add(transformed_gap)
        .expect("should fit constraint distance in layout coordinates")
}

/// Compact both axes until stable while enforcing the requested rendered gaps.
///
/// Alternating axes handles overlaps introduced by the preceding solve. Incoming coordinates
/// remain the baseline for gap transforms, and data layers are re-snapped after each X solve.
pub fn compact_layout(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    gaps: &GapSizes,
) {
    const MAX_ROUNDS: usize = 4;
    let incoming_positions: HashMap<NodeIndex, (i64, i64)> = graph
        .node_indices()
        .map(|node_idx| {
            let pos = graph[node_idx].pos;
            (node_idx, (pos.x, pos.y))
        })
        .collect();
    let mut previous: Option<Vec<(i64, i64)>> = None;
    for _ in 0..MAX_ROUNDS {
        compact_x(graph, gaps, &incoming_positions);
        if let Some(solution) = solve_axis(graph, Axis::Y, gaps, &incoming_positions) {
            apply_axis(graph, Axis::Y, &solution);
        }

        let current: Vec<(i64, i64)> = graph.node_weights().map(|n| (n.pos.x, n.pos.y)).collect();
        if previous.as_ref() == Some(&current) {
            break;
        }
        previous = Some(current);
    }
    snap_wormholes_to_neighbor(graph, gaps, &incoming_positions);
}

fn compact_x(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    gaps: &GapSizes,
    incoming_positions: &HashMap<NodeIndex, (i64, i64)>,
) {
    if let Some(solution) = solve_axis(graph, Axis::X, gaps, incoming_positions) {
        apply_axis(graph, Axis::X, &solution);
        snap_layers_to_anchor_columns(graph, &solution);
    }
}

/// Build segments and constraints for one axis and run the centering solver.
/// Returns `None` (leaving the graph untouched) if the graph is empty or the
/// solver rejects the constraint system.
fn solve_axis(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    axis: Axis,
    gaps: &GapSizes,
    incoming_positions: &HashMap<NodeIndex, (i64, i64)>,
) -> Option<AxisSolution> {
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    if nodes.is_empty() {
        return None;
    }
    let dense_of: HashMap<NodeIndex, usize> = nodes
        .iter()
        .enumerate()
        .map(|(dense, &idx)| (idx, dense))
        .collect();

    // Group nodes connected by axis-aligned edges (equal coordinate on the
    // solved axis) into rigid segments.
    let mut uf = UnionFind::new(nodes.len());
    for edge in graph.edge_references() {
        let source = &graph[edge.source()];
        let target = &graph[edge.target()];
        if axis.pos(source) == axis.pos(target) {
            uf.union(dense_of[&edge.source()], dense_of[&edge.target()]);
        }
    }

    // Compress union-find roots into dense segment ids.
    let mut seg_id_of_root: HashMap<usize, usize> = HashMap::new();
    let mut seg_of: HashMap<NodeIndex, usize> = HashMap::new();
    let mut segments: Vec<AxisSegment> = Vec::new();
    let mut obstacles: Vec<Obstacle> = Vec::new();
    for (dense, &node_idx) in nodes.iter().enumerate() {
        let root = uf.find(dense);
        let seg_id = *seg_id_of_root.entry(root).or_insert_with(|| {
            segments.push(AxisSegment {
                pos: axis.pos(&graph[node_idx]),
            });
            segments.len() - 1
        });
        seg_of.insert(node_idx, seg_id);

        let node = &graph[node_idx];
        let incoming_pos = incoming_positions[&node_idx];
        obstacles.push(Obstacle {
            seg_id,
            axis_pos: axis.pos(node),
            incoming_axis_pos: match axis {
                Axis::X => incoming_pos.0,
                Axis::Y => incoming_pos.1,
            },
            axis_size: axis.size(node),
            perp: occupied_extent(axis.perp_pos(node), axis.perp_size(node)),
            is_data: matches!(node.role, NodeRole::Data(_) | NodeRole::Wormhole(_)),
        });
    }

    // Edge lines within a segment occupy the perpendicular cells between
    // their endpoints (extent 1 on the solved axis — an edge line is one
    // cell thick). Including them as their own obstacle, at their own real
    // extent rather than the whole segment's, makes a vertical edge passing
    // through a row act as an obstacle only where it actually runs (and
    // symmetrically for horizontal edges crossing a column).
    for edge in graph.edge_references() {
        let source = &graph[edge.source()];
        let target = &graph[edge.target()];
        if axis.pos(source) != axis.pos(target) {
            continue;
        }
        let a = occupied_extent(axis.perp_pos(source), axis.perp_size(source));
        let b = occupied_extent(axis.perp_pos(target), axis.perp_size(target));
        let line = Interval {
            start: a.end.min(b.end) + 1,
            end: a.start.max(b.start) - 1,
        };
        if line.start <= line.end {
            let seg = seg_of[&edge.source()];
            let incoming_pos = incoming_positions[&edge.source()];
            obstacles.push(Obstacle {
                seg_id: seg,
                axis_pos: segments[seg].pos,
                incoming_axis_pos: match axis {
                    Axis::X => incoming_pos.0,
                    Axis::Y => incoming_pos.1,
                },
                axis_size: 1,
                perp: line,
                is_data: false,
            });
        }
    }

    let mut constraints = collision_constraints(&obstacles, axis, gaps);

    // Diagonal edges (routed directly rather than through rectilinear dummies)
    // carry no spacing requirement, but their endpoint order on each axis must
    // survive compaction.
    for edge in graph.edge_references() {
        let source = &graph[edge.source()];
        let target = &graph[edge.target()];
        if axis.pos(source) == axis.pos(target) || axis.perp_pos(source) == axis.perp_pos(target) {
            continue;
        }
        let (lower, upper) = if axis.pos(source) < axis.pos(target) {
            (edge.source(), edge.target())
        } else {
            (edge.target(), edge.source())
        };
        constraints.push(Constraint {
            from: seg_of[&lower],
            to: seg_of[&upper],
            gap: 0,
        });
    }

    match compact_axis_with_centering(0..segments.len(), &constraints) {
        Ok(placement) => Some(AxisSolution { seg_of, placement }),
        Err(error) => {
            log::warn!("layout compaction failed on {axis:?} axis: {error:?}");
            None
        }
    }
}

/// Group indices by transitive overlap of their perpendicular intervals.
///
/// Groups only prune candidate pairs; callers must still test direct overlap within a group.
fn overlap_groups<T>(items: &[T], perp: impl Fn(&T) -> Interval) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| perp(&items[i]).start);

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_end = i64::MIN;
    for i in order {
        let interval = perp(&items[i]);
        if !current.is_empty() && interval.start > current_end {
            groups.push(std::mem::take(&mut current));
            current_end = i64::MIN;
        }
        current_end = current_end.max(interval.end);
        current.push(i);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Build pairwise constraints for obstacles whose perpendicular extents overlap.
///
/// Each gap combines the pair's rectangular minimum with its transformed incoming free space.
fn collision_constraints(obstacles: &[Obstacle], axis: Axis, gaps: &GapSizes) -> Vec<Constraint> {
    let mut constraints = Vec::new();
    for group in overlap_groups(obstacles, |o| o.perp) {
        for (rank, &i) in group.iter().enumerate() {
            for &j in &group[rank + 1..] {
                // A stable total order (axis position, then index) on every pair,
                // not just a per-pair comparison, so the constraint graph — built
                // from many overlapping pairs, not a single sorted chain — stays
                // acyclic.
                let (i, j) = if (obstacles[i].axis_pos, i) <= (obstacles[j].axis_pos, j) {
                    (i, j)
                } else {
                    (j, i)
                };
                let (a, b) = (&obstacles[i], &obstacles[j]);
                if a.seg_id == b.seg_id {
                    continue;
                }
                if a.perp.start > b.perp.end || b.perp.start > a.perp.end {
                    continue;
                }
                let gap = constraint_distance(gaps, axis, a, b);
                constraints.push(Constraint {
                    from: a.seg_id,
                    to: b.seg_id,
                    gap,
                });
            }
        }
    }
    constraints
}

fn constraint_distance(gaps: &GapSizes, axis: Axis, lower: &Obstacle, upper: &Obstacle) -> i64 {
    let gap_sizer = gaps.between(axis, lower.is_data, upper.is_data);
    transform_distance(
        gap_sizer,
        lower.incoming_axis_pos,
        lower.axis_size,
        upper.incoming_axis_pos,
        upper.axis_size,
    )
}

/// Snap each degree-one wormhole beside its boundary after shared compaction settles.
fn snap_wormholes_to_neighbor(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    gaps: &GapSizes,
    incoming_positions: &HashMap<NodeIndex, (i64, i64)>,
) {
    let wormhole_indices: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|&idx| matches!(graph[idx].role, NodeRole::Wormhole(_)))
        .collect();
    for wormhole_idx in wormhole_indices {
        let Some(neighbor_idx) = graph.neighbors(wormhole_idx).next() else {
            continue;
        };
        let distance = transform_distance(
            gaps.data_routing_x,
            incoming_positions[&wormhole_idx].0,
            Axis::X.size(&graph[wormhole_idx]),
            incoming_positions[&neighbor_idx].0,
            Axis::X.size(&graph[neighbor_idx]),
        );
        let neighbor_x = graph[neighbor_idx].pos.x;
        let side = if graph[wormhole_idx].pos.x <= neighbor_x {
            -1
        } else {
            1
        };
        graph[wormhole_idx].pos.x = neighbor_x + side * distance;
    }
}

/// Write the centered placement back into the graph.
fn apply_axis(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    axis: Axis,
    solution: &AxisSolution,
) {
    for node_idx in graph.node_indices().collect::<Vec<_>>() {
        let seg = solution.seg_of[&node_idx];
        let value = solution.placement.centered[&seg];
        axis.set_pos(&mut graph[node_idx], value);
    }
}

/// Re-snap the Sugiyama `layer` field of data nodes that were moved within
/// horizontal slack. Fully-constrained columns (`low == high`) keep their
/// layers and act as anchors; floating nodes adopt the layer of the nearest
/// anchor column, preferring the lower layer on distance ties. This keeps
/// cursor navigation consistent: vertical stacking partners that end up at
/// the same x share the same layer.
fn snap_layers_to_anchor_columns(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    solution: &AxisSolution,
) {
    let has_slack = |seg: usize| solution.placement.low[&seg] != solution.placement.high[&seg];

    let mut anchors: Vec<(i64, i32)> = graph
        .node_indices()
        .filter(|idx| !has_slack(solution.seg_of[idx]))
        .filter_map(|idx| {
            let node = &graph[idx];
            match node.role {
                NodeRole::Data(_) => node.layer.map(|layer| (node.pos.x, layer)),
                _ => None,
            }
        })
        .collect();
    if anchors.is_empty() {
        return;
    }
    anchors.sort_unstable();

    for node_idx in graph.node_indices().collect::<Vec<_>>() {
        if !has_slack(solution.seg_of[&node_idx]) {
            continue;
        }
        let node = &mut graph[node_idx];
        if !matches!(node.role, NodeRole::Data(_)) || node.layer.is_none() {
            continue;
        }
        let snapped = anchors
            .iter()
            .map(|&(x, layer)| ((x - node.pos.x).abs(), layer))
            .min_by_key(|&(dist, layer)| (dist, layer))
            .map(|(_, layer)| layer);
        node.layer = snapped;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cross_coordinates::assign_cross_coordinates, geometry::LocalPos};

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
    fn iterative_rounds_reuse_the_incoming_baseline() {
        let gaps = GapSizes {
            data_data_x: |gap| gap.saturating_mul(2),
            ..GapSizes::default()
        };
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let left = graph.add_node(data_node(0, 0, 0, (3, 1), Some(0)));
        let right = graph.add_node(data_node(1, 10, 0, (3, 1), Some(1)));
        graph.add_edge(left, right, edge(0, 1));

        compact_layout(&mut graph, &gaps);

        assert_eq!(graph[left].pos.x.abs_diff(graph[right].pos.x), 17);
    }

    #[test]
    fn unconnected_wormhole_counts_as_data_for_gap_policy() {
        let gaps = GapSizes {
            data_data_x: |_| 2,
            data_routing_x: |_| 0,
            routing_routing_x: |_| 0,
            ..GapSizes::default()
        };
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let boundary = graph.add_node(data_node(0, 0, 0, (1, 1), Some(0)));
        let wormhole = graph.add_node(LayoutNode::new(
            NodeRole::Wormhole(NodeIndex::new(1)),
            LocalPos::new_xy(1, 0),
            (1, 1),
            None,
        ));

        compact_layout(&mut graph, &gaps);

        assert_eq!(graph[boundary].pos.x.abs_diff(graph[wormhole].pos.x), 3);
    }

    #[test]
    fn connected_wormhole_stays_within_data_routing_x_of_routing_neighbor() {
        let gaps = GapSizes {
            data_routing_x: |_| 1,
            routing_routing_x: |gap| gap,
            ..GapSizes::default()
        };
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();
        let wormhole = graph.add_node(LayoutNode::new(
            NodeRole::Wormhole(NodeIndex::new(1)),
            LocalPos::new_xy(0, 0),
            (1, 1),
            None,
        ));
        let neighbor = graph.add_node(LayoutNode::routing(LocalPos::new_xy(100, 0), (1, 1)));
        let vertical_anchor =
            graph.add_node(LayoutNode::routing(LocalPos::new_xy(100, 10), (1, 1)));
        let left_anchor = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 10), (1, 1)));
        graph.add_edge(wormhole, neighbor, edge(0, 1));
        graph.add_edge(neighbor, vertical_anchor, edge(0, 1));
        graph.add_edge(left_anchor, vertical_anchor, edge(0, 1));

        compact_layout(&mut graph, &gaps);

        assert_eq!(graph[wormhole].pos.x.abs_diff(graph[neighbor].pos.x), 2);
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

    fn data_node(
        domain: usize,
        x: i64,
        y: i64,
        size: (u64, u64),
        layer: Option<i32>,
    ) -> LayoutNode {
        LayoutNode::data(NodeIndex::new(domain), LocalPos::new_xy(x, y), size, layer)
    }

    fn edge(domain_a: usize, domain_b: usize) -> LayoutEdge {
        LayoutEdge::new(NodeIndex::new(domain_a), NodeIndex::new(domain_b))
    }

    /// Miniature bypass-arc scenario: a wide node D rides an arc above a row
    /// containing a wide grid node W.
    ///
    /// ```text
    ///   ╭──DDDDDD──╮
    ///   │          │
    ///   A WWWW...W B
    /// ```
    ///
    /// Solved x positions: the corner columns are pinned by W through A/B —
    /// both A/B and W are data nodes, so their pair gets the data-data X
    /// floor (2) on top of base_step(1,21)=11, landing W's column 13 away on
    /// each side — while D (routing-to-data, no floor) floats in [4, 21] and
    /// must be centered at 12.
    #[test]
    fn test_wide_node_centered_within_arc_slack() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let c1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0), (1, 1)));
        let d = graph.add_node(data_node(0, 3, 0, (6, 1), Some(1)));
        let c2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(25, 0), (1, 1)));
        let a = graph.add_node(data_node(1, 0, 4, (1, 1), Some(0)));
        let w = graph.add_node(data_node(2, 12, 4, (21, 1), Some(1)));
        let b = graph.add_node(data_node(3, 25, 4, (1, 1), Some(2)));

        graph.add_edge(c1, d, edge(1, 0));
        graph.add_edge(d, c2, edge(0, 3));
        graph.add_edge(c1, a, edge(1, 1));
        graph.add_edge(c2, b, edge(3, 3));
        graph.add_edge(a, w, edge(1, 2));
        graph.add_edge(w, b, edge(2, 3));

        compact_layout(&mut graph, &GapSizes::default());

        assert_eq!(graph[a].pos.x, 0);
        assert_eq!(graph[w].pos.x, 12);
        assert_eq!(graph[b].pos.x, 24);
        assert_eq!(graph[c2].pos.x, 24);

        // D floats in its slack range; centered placement is the midpoint.
        assert_eq!(graph[d].pos.x, 11);

        // D occupies [9, 14]: 8 free cells on the left arc, 9 on the right —
        // the 1-cell asymmetry is inherent to the odd total slack.
        let extent = occupied_extent(graph[d].pos.x, 6);
        let left_dashes = extent.start - graph[c1].pos.x - 1;
        let right_dashes = graph[c2].pos.x - extent.end - 1;
        assert!((left_dashes - right_dashes).abs() <= 2);

        // Vertical dead space between the rows is compacted to min_gap.
        assert_eq!(graph[c1].pos.y, 0);
        assert_eq!(graph[a].pos.y, 2);

        // D had slack, so its layer snaps to the nearest anchor column (W).
        assert_eq!(graph[d].layer, Some(1));
    }

    /// A row already at minimum spacing must not move.
    #[test]
    fn test_tight_row_unchanged() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // Widths 5 with a requested data-data gap of 1: centers base_step(5,5)+1 = 6 apart.
        let n0 = graph.add_node(data_node(0, 0, 0, (5, 1), Some(0)));
        let n1 = graph.add_node(data_node(1, 6, 0, (5, 1), Some(1)));
        let n2 = graph.add_node(data_node(2, 12, 0, (5, 1), Some(2)));
        graph.add_edge(n0, n1, edge(0, 1));
        graph.add_edge(n1, n2, edge(1, 2));

        compact_layout(&mut graph, &GapSizes::default());

        assert_eq!(graph[n0].pos.x, 0);
        assert_eq!(graph[n1].pos.x, 6);
        assert_eq!(graph[n2].pos.x, 12);
        // No slack anywhere, so layers are untouched.
        assert_eq!(graph[n1].layer, Some(1));
    }

    /// Oversized vertical gaps shrink to the minimum spacing.
    #[test]
    fn test_vertical_dead_space_removed() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let top = graph.add_node(data_node(0, 0, 0, (1, 1), Some(0)));
        let bottom = graph.add_node(data_node(1, 0, 10, (1, 1), Some(0)));
        graph.add_edge(top, bottom, edge(0, 1));

        compact_layout(&mut graph, &GapSizes::default());

        assert_eq!(graph[top].pos.y, 0);
        assert_eq!(graph[bottom].pos.y, 2);
    }

    /// A vertical edge passing through a row is an obstacle: the row's nodes
    /// keep a width-aware distance from the crossing column.
    #[test]
    fn test_crossing_edge_blocks_row_nodes() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // Vertical edge from (10, 0) to (10, 4) crossing the row at y=2.
        let top = graph.add_node(data_node(0, 10, 0, (1, 1), Some(1)));
        let bottom = graph.add_node(data_node(1, 10, 4, (1, 1), Some(1)));
        graph.add_edge(top, bottom, edge(0, 1));

        // Row at y=2: two nodes on either side of the crossing.
        let left = graph.add_node(data_node(2, 0, 2, (5, 1), Some(0)));
        let right = graph.add_node(data_node(3, 20, 2, (5, 1), Some(2)));
        graph.add_edge(left, right, edge(2, 3));

        compact_layout(&mut graph, &GapSizes::default());

        // The crossing column must stay strictly between the row nodes with
        // a one-cell clearance from each label: base_step(5,1)+1 = 4 from the
        // left center and base_step(1,5)+1 = 4 to the right center.
        let line_x = graph[top].pos.x;
        assert_eq!(graph[top].pos.x, graph[bottom].pos.x, "edge stays vertical");
        assert!(line_x - graph[left].pos.x >= 4);
        assert!(graph[right].pos.x - line_x >= 4);
    }

    /// A diamond whose two middle-layer branches have very different heights. The size-aware
    /// cross-coordinate assignment must (a) separate the branches by their real-height minimum,
    /// not a unit-size gap, and (b) centre the shared source and sink on the midpoint between the
    /// branches. A uniform-height diamond cannot distinguish this from a size-blind pass, which is
    /// why the branches here are 5x9 and 5x3.
    #[test]
    fn test_cross_coordinates_center_between_uneven_branches() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // Layer columns: source at x=0, the two branches at x=1, sink at x=2. Initial y only
        // fixes the within-layer order (tall branch above short branch).
        let source = graph.add_node(data_node(0, 0, 5, (5, 3), Some(0)));
        let tall = graph.add_node(data_node(1, 1, 0, (5, 9), Some(1)));
        let short = graph.add_node(data_node(2, 1, 20, (5, 3), Some(1)));
        let sink = graph.add_node(data_node(3, 2, 5, (5, 3), Some(2)));

        graph.add_edge(source, tall, edge(0, 1));
        graph.add_edge(source, short, edge(0, 2));
        graph.add_edge(tall, sink, edge(1, 3));
        graph.add_edge(short, sink, edge(2, 3));

        assign_cross_coordinates(&mut graph, 1);

        // The branches keep their order and are separated by their real-height minimum step
        // (base_step(9, 3) + min_gap = 6 + 1 = 7), well above the unit-size gap of 2.
        let branch_gap = graph[short].pos.y - graph[tall].pos.y;
        assert_eq!(branch_gap, base_step(9, 3) + 1);

        // The source and sink centre on the midpoint between the two branches, and on each other.
        // The true midpoint (0 + 7) / 2 = 3.5 is a half cell; round the same way the assignment
        // does rather than truncating, since the branch gap here is odd by construction.
        let midpoint = ((graph[tall].pos.y + graph[short].pos.y) as f64 / 2.0).round() as i64;
        assert_eq!(graph[source].pos.y, midpoint);
        assert_eq!(graph[sink].pos.y, midpoint);
        assert_eq!(graph[source].pos.y, graph[sink].pos.y);
    }

    /// A long edge routed through a dummy stays straight even when a tall real node shares the
    /// dummy's column: the dummy (top priority) holds the source-sink line and pushes the tall
    /// node aside rather than bending toward it.
    #[test]
    fn test_cross_coordinates_straighten_long_edge_past_tall_node() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        // source -> dummy -> sink is the only edge path; the tall node just occupies the middle
        // column above nothing, so its only interaction is being displaced by the dummy.
        let source = graph.add_node(data_node(0, 0, 0, (5, 3), Some(0)));
        let dummy = graph.add_node(LayoutNode::routing(LocalPos::new_xy(1, 0), (1, 1)));
        let tall = graph.add_node(data_node(1, 1, 10, (5, 15), Some(1)));
        let sink = graph.add_node(data_node(2, 2, 0, (5, 3), Some(2)));

        graph.add_edge(source, dummy, edge(0, 0));
        graph.add_edge(dummy, sink, edge(0, 2));

        assign_cross_coordinates(&mut graph, 1);

        // The dummy keeps the source-sink edge straight: all three share one y.
        assert_eq!(graph[source].pos.y, graph[dummy].pos.y);
        assert_eq!(graph[dummy].pos.y, graph[sink].pos.y);
        // The tall node kept the within-layer order (dummy above it) and was pushed clear.
        assert!(graph[tall].pos.y > graph[dummy].pos.y);
    }
}
