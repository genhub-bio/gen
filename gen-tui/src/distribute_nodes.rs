//! Global layout compaction for layout graphs.
//!
//! After the Sugiyama algorithm and edge routing complete, the layout contains
//! dead space (Brandes-Köpf averaging artifacts, raw routing channel tracks)
//! and off-center wide nodes. This module rebuilds all coordinates with a
//! separation-constraint solver (see [`crate::compaction`]): nodes that must
//! stay aligned move together as rigid segments, every ordered pair of
//! segments that overlaps in the perpendicular direction gets a width-aware
//! minimum-distance constraint, and each segment is placed at the midpoint of
//! its feasible range. Compression and centering therefore come out of a
//! single solve instead of separate heuristic passes.

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
    /// Largest member extent on the solved axis. Shrink-limit gaps are
    /// computed from this rather than per overlapping span, so a row (or
    /// column) behaves as one band of uniform thickness — keeping row pitch
    /// uniform, which zooming (disperse/contract) relies on.
    max_size: i64,
    /// Occupied intervals in the perpendicular direction with each occupant's
    /// extent on the solved axis (edge lines have extent 1). Sorted by start.
    /// These decide *whether* two segments constrain each other — a bypass
    /// arc whose spans never overlap a grid row can float past it — and the
    /// per-span extents detect genuine cell conflicts that require expansion.
    spans: Vec<(Interval, i64)>,
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

/// Compact the layout graph on both axes.
///
/// The x axis is solved first from the current geometry; the y axis is then
/// solved from the updated geometry. Solving sequentially (rather than both
/// axes from the same snapshot) is what keeps the result overlap-free: any
/// pair brought into x-overlap by the first solve is visible to the y sweep
/// and gets a y separation constraint.
///
/// The solver uses two gap regimes per [`required_gap`]:
/// * A *band gap* that prevents segments from compressing closer than their
///   maximum member extents allow (shrink-only, capped at the current gap).
/// * A *conflict gap* that expands spacing when overlapping span pairs are
///   too close for their cell extents (e.g. a junction column that starts
///   inside a wide node's label).
///
/// Data node `layer` fields (used for cursor navigation) are re-snapped after
/// the x solve: nodes with horizontal slack (e.g. a wide node centered along
/// a bypass arc) adopt the layer of the nearest fully-constrained column.
pub fn compact_layout(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    vertex_spacing: f64,
) {
    let min_gap = (vertex_spacing.round() as i64).max(1);

    compact_x(graph, min_gap);

    if let Some(solution) = solve_axis(graph, Axis::Y, min_gap) {
        apply_axis(graph, Axis::Y, &solution);
    }
}

/// Compact the layout graph horizontally only. Used for bridge layouts,
/// whose endpoints must keep the y coordinates they inherited from the
/// adjacent section layouts.
pub fn compact_layout_horizontal(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    vertex_spacing: f64,
) {
    let min_gap = (vertex_spacing.round() as i64).max(1);
    compact_x(graph, min_gap);
}

fn compact_x(graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>, min_gap: i64) {
    if let Some(solution) = solve_axis(graph, Axis::X, min_gap) {
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
    min_gap: i64,
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

    // Stitch nodes are the inter-partition interface: a boundary column (or
    // row) must move as one rigid item even where its members are not
    // connected inside this partition, or a pass-through edge can detach
    // from the boundary the neighboring partition continues it at.
    let mut stitch_group: HashMap<i64, usize> = HashMap::new();
    for (dense, &node_idx) in nodes.iter().enumerate() {
        let node = &graph[node_idx];
        if matches!(node.role, NodeRole::Stitch(_)) {
            match stitch_group.entry(axis.pos(node)) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    uf.union(dense, *entry.get());
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(dense);
                }
            }
        }
    }

    // The neighbors of a stitch node are the boundary interface itself: the
    // bridge places copies of them at the partition edge, and those copies
    // deduplicate by world position only while the originals stay together in
    // the boundary column. Keep same-coordinate stitch neighbors rigid.
    for &node_idx in &nodes {
        if !matches!(graph[node_idx].role, NodeRole::Stitch(_)) {
            continue;
        }
        let mut neighbor_group: HashMap<i64, usize> = HashMap::new();
        for neighbor in graph.neighbors(node_idx) {
            let neighbor_dense = dense_of[&neighbor];
            match neighbor_group.entry(axis.pos(&graph[neighbor])) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    uf.union(neighbor_dense, *entry.get());
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(neighbor_dense);
                }
            }
        }
    }

    // Bridge boundary copies all start at the same x-position (x=0 on the left,
    // x=min_width on the right) but share no edges and are not stitch nodes, so
    // without this they drift independently after centering, breaking coordinate
    // deduplication with the adjacent section partition.
    if matches!(axis, Axis::X) {
        let min_pos = nodes.iter().map(|&n| axis.pos(&graph[n])).min();
        let max_pos = nodes.iter().map(|&n| axis.pos(&graph[n])).max();
        if let (Some(mn), Some(mx)) = (min_pos, max_pos) {
            let mut min_rep: Option<usize> = None;
            let mut max_rep: Option<usize> = None;
            for (dense, &node_idx) in nodes.iter().enumerate() {
                let pos = axis.pos(&graph[node_idx]);
                if pos == mn {
                    match &mut min_rep {
                        None => min_rep = Some(dense),
                        Some(g) => uf.union(dense, *g),
                    }
                }
                if pos == mx && mx != mn {
                    match &mut max_rep {
                        None => max_rep = Some(dense),
                        Some(g) => uf.union(dense, *g),
                    }
                }
            }
        }
    }

    // Compress union-find roots into dense segment ids.
    let mut seg_id_of_root: HashMap<usize, usize> = HashMap::new();
    let mut seg_of: HashMap<NodeIndex, usize> = HashMap::new();
    let mut segments: Vec<AxisSegment> = Vec::new();
    for (dense, &node_idx) in nodes.iter().enumerate() {
        let root = uf.find(dense);
        let seg_id = *seg_id_of_root.entry(root).or_insert_with(|| {
            segments.push(AxisSegment {
                pos: axis.pos(&graph[node_idx]),
                max_size: 1,
                spans: Vec::new(),
            });
            segments.len() - 1
        });
        seg_of.insert(node_idx, seg_id);

        let node = &graph[node_idx];
        let segment = &mut segments[seg_id];
        segment.max_size = segment.max_size.max(axis.size(node));
        segment.spans.push((
            occupied_extent(axis.perp_pos(node), axis.perp_size(node)),
            axis.size(node),
        ));
    }

    // Edge lines within a segment occupy the perpendicular cells between
    // their endpoints with extent 1 on the solved axis. Including them makes
    // a vertical edge passing through a row act as an obstacle for that row
    // (and symmetrically for horizontal edges crossing a column).
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
            // For Y-axis, a horizontal wire spans the full row height; using the
            // segment's max_size lets wide data rows push routing stubs out to a
            // position outside the data node's visual extent.
            let obstacle_size = if matches!(axis, Axis::Y) {
                segments[seg].max_size
            } else {
                1
            };
            segments[seg].spans.push((line, obstacle_size));
        }
    }

    for segment in &mut segments {
        segment.spans.sort_by_key(|(interval, _)| interval.start);
    }

    let mut constraints = sweep_constraints(&segments, min_gap);

    // Stitch nodes mark partition boundaries. Their y-span is tiny (1×1 dummy
    // size), so sweep_constraints misses pairs where the stitch row and the
    // data row don't share any perpendicular cells. Force full x-clearance
    // between every stitch segment and every other segment.
    if matches!(axis, Axis::X) {
        // Accumulate the maximum required gap per ordered (from, to) pair so we
        // emit one constraint per pair rather than one per node combination.
        let mut stitch_gaps: HashMap<(usize, usize), i64> = HashMap::new();
        for &stitch_idx in &nodes {
            if !matches!(graph[stitch_idx].role, NodeRole::Stitch(_)) {
                continue;
            }
            let stitch_seg = seg_of[&stitch_idx];
            let stitch_x = axis.pos(&graph[stitch_idx]);
            let stitch_w = axis.size(&graph[stitch_idx]);
            for &other_idx in &nodes {
                let other_seg = seg_of[&other_idx];
                if other_seg == stitch_seg {
                    continue;
                }
                let other_x = axis.pos(&graph[other_idx]);
                if other_x == stitch_x {
                    continue;
                }
                let gap = base_step(stitch_w, axis.size(&graph[other_idx])) + min_gap;
                let (from, to) = if stitch_x < other_x {
                    (stitch_seg, other_seg)
                } else {
                    (other_seg, stitch_seg)
                };
                stitch_gaps
                    .entry((from, to))
                    .and_modify(|g| *g = (*g).max(gap))
                    .or_insert(gap);
            }
        }
        for ((from, to), gap) in stitch_gaps {
            constraints.push(Constraint { from, to, gap });
        }
    }

    // Diagonal edges (direct stitch connections that skipped rectilinear
    // routing) carry no spacing requirement, but their endpoint order on each
    // axis must survive compaction.
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

/// Generate ordered separation constraints between every pair of segments
/// whose perpendicular spans overlap, with a gap derived from the segments'
/// maximum member extents.
fn sweep_constraints(segments: &[AxisSegment], min_gap: i64) -> Vec<Constraint> {
    // Total order on (current position, segment id) directs every constraint
    // from lower to higher, which guarantees the constraint graph is acyclic.
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by_key(|&seg| (segments[seg].pos, seg));

    let mut constraints = Vec::new();
    for (rank, &left) in order.iter().enumerate() {
        for &right in &order[rank + 1..] {
            let current = segments[right].pos - segments[left].pos;
            if let Some(gap) = required_gap(&segments[left], &segments[right], min_gap, current) {
                constraints.push(Constraint {
                    from: left,
                    to: right,
                    gap,
                });
            }
        }
    }
    constraints
}

/// Separation needed between two segments at distance `current`, or `None`
/// if no spans overlap (the segments may pass each other).
///
/// Two regimes are combined:
/// - The band gap, from the segments' maximum member extents, acts only as a
///   shrink limit (clamped to `current`). Deliberately tight arrangements
///   (adjacent junction rows forming continuous lines, interleaved tall boxes,
///   routing tracks at one-cell spacing) must survive, and expanding would
///   also shift partitions out of alignment with their neighbors at stitch
///   boundaries.
/// - The conflict gap detects genuine cell overlaps per span pair: raw
///   routing channel tracks are not label-aware, so a junction column can
///   start inside a wide node's label. Such pairs expand to full clearance.
fn required_gap(
    left: &AxisSegment,
    right: &AxisSegment,
    min_gap: i64,
    current: i64,
) -> Option<i64> {
    let mut touching = false;
    let mut conflict_gap = 0i64;
    for &(left_interval, left_size) in &left.spans {
        for &(right_interval, right_size) in &right.spans {
            if overlaps(left_interval, right_interval) {
                touching = true;
                let step = base_step(left_size, right_size);
                if current < step {
                    conflict_gap = conflict_gap.max(step + min_gap);
                }
            }
        }
    }
    if !touching {
        return None;
    }
    let band_gap = base_step(left.max_size, right.max_size) + min_gap;
    Some(band_gap.min(current).max(conflict_gap))
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
    use crate::geometry::LocalPos;

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
        LayoutNode::data(
            NodeIndex::new(domain),
            LocalPos::new_xy(0, x, y),
            size,
            layer,
        )
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
    /// Solved x positions: the corner columns are pinned by W
    /// (base_step(1,21)+1 = 12 on each side, so the right column lands at 24),
    /// while D floats in [4, 19] and must be centered at 11.
    #[test]
    fn test_wide_node_centered_within_arc_slack() {
        let mut graph = StableGraph::<LayoutNode, LayoutEdge, Undirected, u32>::default();

        let c1 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0, 0), (1, 1)));
        let d = graph.add_node(data_node(0, 3, 0, (6, 1), Some(1)));
        let c2 = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 25, 0), (1, 1)));
        let a = graph.add_node(data_node(1, 0, 4, (1, 1), Some(0)));
        let w = graph.add_node(data_node(2, 12, 4, (21, 1), Some(1)));
        let b = graph.add_node(data_node(3, 25, 4, (1, 1), Some(2)));

        graph.add_edge(c1, d, edge(1, 0));
        graph.add_edge(d, c2, edge(0, 3));
        graph.add_edge(c1, a, edge(1, 1));
        graph.add_edge(c2, b, edge(3, 3));
        graph.add_edge(a, w, edge(1, 2));
        graph.add_edge(w, b, edge(2, 3));

        compact_layout(&mut graph, 1.0);

        assert_eq!(graph[a].pos.x, 0);
        assert_eq!(graph[w].pos.x, 12);
        assert_eq!(graph[b].pos.x, 24);
        assert_eq!(graph[c2].pos.x, 24);

        // D floats in [4, 19]; centered placement is the midpoint 11.
        assert_eq!(graph[d].pos.x, 11);

        // D occupies [9, 14]: 8 free cells on the left arc, 9 on the right —
        // the 1-cell asymmetry is inherent to the odd total slack.
        let extent = occupied_extent(graph[d].pos.x, 6);
        let left_dashes = extent.start - graph[c1].pos.x - 1;
        let right_dashes = graph[c2].pos.x - extent.end - 1;
        assert!((left_dashes - right_dashes).abs() <= 1);

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

        // Widths 5 with 1 free cell between: centers 6 apart.
        let n0 = graph.add_node(data_node(0, 0, 0, (5, 1), Some(0)));
        let n1 = graph.add_node(data_node(1, 6, 0, (5, 1), Some(1)));
        let n2 = graph.add_node(data_node(2, 12, 0, (5, 1), Some(2)));
        graph.add_edge(n0, n1, edge(0, 1));
        graph.add_edge(n1, n2, edge(1, 2));

        compact_layout(&mut graph, 1.0);

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

        compact_layout(&mut graph, 1.0);

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

        compact_layout(&mut graph, 1.0);

        // The crossing column must stay strictly between the row nodes with
        // a one-cell clearance from each label: base_step(5,1)+1 = 4 from the
        // left center and base_step(1,5)+1 = 4 to the right center.
        let line_x = graph[top].pos.x;
        assert_eq!(graph[top].pos.x, graph[bottom].pos.x, "edge stays vertical");
        assert!(line_x - graph[left].pos.x >= 4);
        assert!(graph[right].pos.x - line_x >= 4);
    }
}
