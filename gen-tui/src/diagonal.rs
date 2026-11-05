//! Unrouted edges for graphs too dense for the rectilinear router.
//!
//! These modes keep the Sugiyama layers, dummy nodes and loopback pins, but skip `edge_router`.
//! A terminal approximates their straight or cubic paths with braille dots. A caller that
//! rasterizes the buffer itself (PNG export) asks for the paths and draws them on its bitmap.

use std::collections::HashMap;

use ratatui::style::{Color, Style};

use crate::{
    edge_ports::NodeSide,
    geometry::{WorldPos, WorldRect},
    layout::{LayoutNode, NodeRole, PinSide},
    plotter::LineStyle,
    viewport_graph::ViewportGraph,
    viewport_state::WorldBuffer,
};

/// How a window's edges are routed and drawn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum EdgeStyle {
    /// Axis-aligned routed edges drawn with box-drawing characters.
    #[default]
    Rectilinear,
    /// Straight lines between nodes, bypassing the edge router.
    Diagonal(DiagonalDrawing),
    /// Cubic Bézier curves with horizontal entry and exit tangents.
    Bezier(DiagonalDrawing),
}

impl EdgeStyle {
    /// Whether edges skip the router.
    pub const fn is_diagonal(self) -> bool {
        matches!(self, Self::Diagonal(_) | Self::Bezier(_))
    }

    /// Whether a bitmap renderer draws the edges, so arrows are reported as marks too.
    pub const fn is_deferred(self) -> bool {
        matches!(
            self,
            Self::Diagonal(DiagonalDrawing::Deferred) | Self::Bezier(DiagonalDrawing::Deferred)
        )
    }
}

/// Who draws the paths of [`EdgeStyle::Diagonal`] or [`EdgeStyle::Bezier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagonalDrawing {
    /// Draw into the buffer with braille dots.
    Braille,
    /// Leave the buffer free of edge marks and report paths as [`DeferredEdge`]s, for callers
    /// that draw them on a bitmap. Everything else is placed exactly as in `Braille`.
    Deferred,
}

/// One straight edge segment, in buffer cell units: the origin is the top-left corner of the
/// buffer, x grows right and y grows down, and cell `(column, row)` spans
/// `[column, column + 1) x [row, row + 1)`. Endpoints on a node sit on its border.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiagonalLine {
    pub from: (f64, f64),
    pub to: (f64, f64),
    pub color: Color,
    pub line_style: LineStyle,
}

/// A cubic edge curve in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierCurve {
    pub from: (f64, f64),
    pub control1: (f64, f64),
    pub control2: (f64, f64),
    pub to: (f64, f64),
    pub color: Color,
    pub line_style: LineStyle,
}

/// A direction arrowhead for bitmap renderers, centered on a point that need not be a cell's
/// center, in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrowMark {
    pub center: (f64, f64),
    /// Direction of travel in buffer cell coordinates.
    pub direction: (f64, f64),
    pub color: Color,
}

/// The footprint of highlighted node cells (a whole node or a sub-range of one) for bitmap
/// renderers, which surround it with a glow like the one around highlighted edges. Corners are
/// in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeGlow {
    pub top_left: (f64, f64),
    pub bottom_right: (f64, f64),
    /// Whether the highlight continues into unhighlighted cells of the same node on that side,
    /// where the glow must stop at the cell boundary instead of spreading past it.
    pub clip_left: bool,
    pub clip_right: bool,
    pub color: Color,
}

/// The footprint of a highlighted single-glyph node, as drawn at minimal detail, for bitmap
/// renderers. The glyph is a filled disc smaller than its cell, so a rectangular [`NodeGlow`]
/// would sit away from it; instead the renderer redraws the disc slightly larger, with the same
/// layered glow a highlighted edge has. Center in buffer cell units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymbolGlow {
    pub center: (f64, f64),
    pub color: Color,
}

/// A deferred edge path for bitmap renderers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeferredEdge {
    Line(DiagonalLine),
    Bezier(BezierCurve),
    Arrow(ArrowMark),
    NodeGlow(NodeGlow),
    SymbolGlow(SymbolGlow),
}

/// Build the S-curve of an ordinary edge: a cubic with horizontal tangents at both ends and
/// handles extending halfway toward the opposite endpoint.
pub(crate) fn bezier_curve(
    from: (f64, f64),
    to: (f64, f64),
    color: Color,
    line_style: LineStyle,
) -> BezierCurve {
    let midpoint_x = (from.0 + to.0) / 2.0;
    BezierCurve {
        from,
        control1: (midpoint_x, from.1),
        control2: (midpoint_x, to.1),
        to,
        color,
        line_style,
    }
}

impl BezierCurve {
    /// The same curve with every point passed through `transform`, e.g. world to terminal
    /// coordinates.
    pub(crate) fn mapped(self, transform: impl Fn((f64, f64)) -> (f64, f64)) -> Self {
        Self {
            from: transform(self.from),
            control1: transform(self.control1),
            control2: transform(self.control2),
            to: transform(self.to),
            ..self
        }
    }

    /// The same curve traversed from its other end.
    fn reversed(self) -> Self {
        Self {
            from: self.to,
            control1: self.control2,
            control2: self.control1,
            to: self.from,
            ..self
        }
    }
}

/// Which way a loopback bulges away from the data node it leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopbackSide {
    Right,
    Left,
}

/// The two neighbors of a loopback pin: the data node the loop leaves or enters, and the first
/// node of the long return run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LoopbackNeighbors<Id> {
    pub anchor: Id,
    pub returning: Id,
}

/// Find the anchor and return run of a loopback pin: its first data neighbor and the first other
/// neighbor. A pin without a data neighbor (its anchor became a junction) or without a second
/// neighbor is not a recognizable loop, and its edges stay ordinary S-curves.
pub(crate) fn loopback_neighbors<'a, Id: Copy + PartialEq>(
    neighbors: impl IntoIterator<Item = (Id, &'a NodeRole)>,
) -> Option<LoopbackNeighbors<Id>> {
    let neighbors: Vec<_> = neighbors.into_iter().collect();
    let (anchor, _) = neighbors
        .iter()
        .find(|(_, role)| matches!(role, NodeRole::Data(_)))?;
    let (returning, _) = neighbors.iter().find(|(neighbor, _)| neighbor != anchor)?;
    Some(LoopbackNeighbors {
        anchor: *anchor,
        returning: *returning,
    })
}

/// A loopback bulges no further than this, however far its return run sits, so one tall loop
/// does not push its neighbors apart sideways.
const LOOPBACK_MAX_REACH_CELLS: f64 = 5.0;

/// Cells a loopback bulges away from its anchor, for a return run `height` rows from its start.
pub(crate) fn loopback_reach(height: f64) -> f64 {
    (height / 2.0).min(LOOPBACK_MAX_REACH_CELLS)
}

/// The turnaround of a loopback: a cubic that leaves `start` horizontally toward `side`, reverses
/// direction, and finishes back at `start`'s X on the row `return_y`, tangent to the horizontal
/// return run that continues from there. With the row separation `h`, handles reach `2h / 3` and the curve bulges `h / 2`. The loop depends only on the
/// anchor and the return row; the pin that identifies it need not lie on the curve.
pub(crate) fn loopback_curve(
    start: (f64, f64),
    return_y: f64,
    side: LoopbackSide,
    color: Color,
    line_style: LineStyle,
) -> BezierCurve {
    let handle = 4.0 * loopback_reach((return_y - start.1).abs()) / 3.0;
    let handle_x = match side {
        LoopbackSide::Right => start.0 + handle,
        LoopbackSide::Left => start.0 - handle,
    };
    BezierCurve {
        from: start,
        control1: (handle_x, start.1),
        control2: (handle_x, return_y),
        to: (start.0, return_y),
        color,
        line_style,
    }
}

/// A loopback in continuous world coordinates, found from its pin.
struct Loopback {
    anchor: WorldPos,
    returning: WorldPos,
    side: LoopbackSide,
    /// Where the straight departure from the anchor meets the turnaround.
    start: (f64, f64),
    anchor_end: (f64, f64),
    /// Where the straight return run meets the first node of the long span.
    return_end: (f64, f64),
}

fn loopback_at(
    viewport_graph: &ViewportGraph,
    pin: WorldPos,
    content_rects: &HashMap<WorldPos, WorldRect>,
) -> Option<Loopback> {
    let pin_node = viewport_graph.node_data_by_pos.get(&pin)?;
    let LoopbackNeighbors { anchor, returning } =
        loopback_neighbors(viewport_graph.neighbors(pin).filter_map(|neighbor| {
            let node = viewport_graph.node_data_by_pos.get(&neighbor)?;
            Some((neighbor, &node.role))
        }))?;
    let side = match anchor.x.cmp(&pin.x) {
        core::cmp::Ordering::Less => LoopbackSide::Right,
        core::cmp::Ordering::Greater => LoopbackSide::Left,
        core::cmp::Ordering::Equal => return None,
    };
    let anchor_node = viewport_graph.node_data_by_pos.get(&anchor)?;
    let returning_node = viewport_graph.node_data_by_pos.get(&returning)?;
    let (anchor_end, _) = segment_endpoints((anchor, anchor_node), (pin, pin_node), content_rects);
    let (return_end, _) =
        segment_endpoints((returning, returning_node), (pin, pin_node), content_rects);
    let bend_x = match side {
        LoopbackSide::Right => anchor_end.0.max(return_end.0) + 1.0,
        LoopbackSide::Left => anchor_end.0.min(return_end.0) - 1.0,
    };
    Some(Loopback {
        anchor,
        returning,
        side,
        start: (bend_x, anchor_end.1),
        anchor_end,
        return_end,
    })
}

/// How far an arch's handles reach toward its other end, as a fraction of the gap.
const BEND_HANDLE_FRACTION: f64 = 0.4;

/// How many rows an arch rises per column of gap, within [`BEND_MIN_REACH_ROWS`] and
/// [`BEND_MAX_REACH_ROWS`]. Cells are about twice as tall as wide, so this keeps arches shallow.
const BEND_REACH_PER_GAP: f64 = 0.12;
const BEND_MIN_REACH_ROWS: f64 = 0.5;
const BEND_MAX_REACH_ROWS: f64 = 1.0;

/// How an edge through an up or down pin is drawn: not through the pin, but as one arch between
/// the pin's two neighbors that leaves and arrives heading away from the line they lie on. The pin
/// determines the bend direction and reserves room for the arch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BendArch {
    /// The pin's neighbor on its west side, where the arch starts.
    pub west: WorldPos,
    /// The pin's neighbor on its east side, where the arch ends.
    pub east: WorldPos,
    pub curve: BezierCurve,
}

impl BendArch {
    /// The arch's midpoint, where it runs level.
    pub(crate) fn midpoint(&self) -> (f64, f64) {
        let curve = &self.curve;
        (
            (curve.from.0 + 3.0 * curve.control1.0 + 3.0 * curve.control2.0 + curve.to.0) / 8.0,
            (curve.from.1 + 3.0 * curve.control1.1 + 3.0 * curve.control2.1 + curve.to.1) / 8.0,
        )
    }
}

/// Build the arch of an up or down pin, or `None` when `pin` is not one or lacks a neighbor on
/// each side. The bend follows the pin's position relative to the line between its neighbors.
pub(crate) fn bend_arch(
    viewport_graph: &ViewportGraph,
    pin: WorldPos,
    content_rects: &HashMap<WorldPos, WorldRect>,
    color: Color,
    line_style: LineStyle,
) -> Option<BendArch> {
    let NodeRole::Pin(PinSide::Up | PinSide::Down) =
        viewport_graph.node_data_by_pos.get(&pin)?.role
    else {
        return None;
    };
    let neighbors: Vec<WorldPos> = viewport_graph.neighbors(pin).collect();
    let west = neighbors
        .iter()
        .copied()
        .filter(|n| n.x < pin.x)
        .min_by_key(|n| n.x)?;
    let east = neighbors
        .iter()
        .copied()
        .filter(|n| n.x > pin.x)
        .max_by_key(|n| n.x)?;
    let (from, to) = segment_endpoints(
        (west, viewport_graph.node_data_by_pos.get(&west)?),
        (east, viewport_graph.node_data_by_pos.get(&east)?),
        content_rects,
    );
    let gap = to.0 - from.0;
    let handle = gap * BEND_HANDLE_FRACTION;
    // A cubic with both controls `lift` off the line peaks at 3/4 of it.
    let peak = (gap * BEND_REACH_PER_GAP).clamp(BEND_MIN_REACH_ROWS, BEND_MAX_REACH_ROWS);
    let pin_row = pin.y as f64 + 0.5;
    let line_row = west.y as f64
        + 0.5
        + (pin.x - west.x) as f64 / (east.x - west.x) as f64 * (east.y - west.y) as f64;
    let lift = (pin_row - line_row).signum() * peak / 0.75;
    Some(BendArch {
        west,
        east,
        curve: BezierCurve {
            from,
            control1: (from.0 + handle, from.1 + lift),
            control2: (to.0 - handle, to.1 + lift),
            to,
            color,
            line_style,
        },
    })
}

/// An edge's cubic together with whether it is a loopback turnaround.
pub(crate) struct RoutedCurve {
    pub(crate) curve: BezierCurve,
    /// A horizontal departure from a loopback's anchor to its outward bend.
    pub(crate) stub: Option<DiagonalLine>,
    /// True for the curved [`loopback_curve`] half of a loopback, false for the straight return
    /// run and every ordinary S-curve. Arrow placement skips turnarounds and marks only the
    /// exactly horizontal sections of a route.
    pub(crate) is_turnaround: bool,
}

/// Build the cubic for an edge, including loopback routing and its optional departure stub.
pub(crate) fn routed_edge_curve(
    viewport_graph: &ViewportGraph,
    source: (WorldPos, &LayoutNode),
    target: (WorldPos, &LayoutNode),
    content_rects: &HashMap<WorldPos, WorldRect>,
    color: Color,
    line_style: LineStyle,
) -> RoutedCurve {
    [source, target]
        .into_iter()
        .find_map(|(pin, _)| bend_arch(viewport_graph, pin, content_rects, color, line_style))
        .map(|arch| RoutedCurve {
            curve: arch.curve,
            stub: None,
            is_turnaround: true,
        })
        .or_else(|| {
            self_link_span_curve(
                viewport_graph,
                source,
                target,
                content_rects,
                color,
                line_style,
            )
        })
        .or_else(|| {
            loopback_edge_curve(
                viewport_graph,
                source,
                target,
                content_rects,
                color,
                line_style,
            )
        })
        .or_else(|| hair_pin_bend(viewport_graph, source, target, color, line_style))
        .unwrap_or_else(|| {
            let (from, to) = segment_endpoints(source, target, content_rects);
            RoutedCurve {
                curve: bezier_curve(from, to, color, line_style),
                stub: None,
                is_turnaround: false,
            }
        })
}

/// How far, in cells, the apex of a hairpin arc lies from its attachment point.
const HAIRPIN_ARC_REACH_CELLS: f64 = 4.75;

/// The room, in cells, a layout keeps free beside a node side that has a hairpin. Neighboring
/// arcs may extend beyond this reservation, leaving a small gap between facing hairpins.
pub(crate) const HAIRPIN_ARC_RESERVED_CELLS: i64 = 3;

/// How far out the S-curves meet the bend at its top and bottom.
const HAIRPIN_ARC_TOP_CELLS: f64 = 3.0;

/// How far, in rows, the top and bottom of a hairpin arc lie from the row of its attachment.
/// The longer S-curves and tighter bend give the loop a droplet shape.
const HAIRPIN_ARC_RISE_ROWS: f64 = 1.0;

/// Position along the bend where its tangent is close to 45 degrees in a bitmap cell.
const HAIRPIN_ARROW_BEND_POSITION: f64 = 0.225;

/// The arc of a hairpin: an edge that leaves a side of a node and returns to the same side, drawn
/// as a loop that starts and ends at `attachment`, the middle of the node's side port. Both ends
/// run level along the row of `attachment`, so the loop leaves and rejoins the flat edge there
/// smoothly. In order, the three cubics climb to the top, round the far end, and descend back.
/// The bend has level tangents at both joins. `direction` is 1.0 for the east
/// side and -1.0 for the west. World y grows upward, so the top has the larger y.
pub(crate) fn hairpin_arc(
    attachment: (f64, f64),
    direction: f64,
    color: Color,
    line_style: LineStyle,
) -> [BezierCurve; 3] {
    let (x, y) = attachment;
    let top_x = x + direction * HAIRPIN_ARC_TOP_CELLS;
    let apex_x = x + direction * HAIRPIN_ARC_REACH_CELLS;
    // A symmetric cubic reaches three quarters of the distance to its controls at its middle.
    let bend_control_x = top_x + (apex_x - top_x) * 4.0 / 3.0;
    let rise = HAIRPIN_ARC_RISE_ROWS;
    let curve = |from, control1, control2, to| BezierCurve {
        from,
        control1,
        control2,
        to,
        color,
        line_style,
    };
    // The climbs are the S-curve of an ordinary edge, level at both ends.
    let half_top = x + (top_x - x) / 2.0;
    [
        curve(
            (x, y),
            (half_top, y),
            (half_top, y + rise),
            (top_x, y + rise),
        ),
        curve(
            (top_x, y + rise),
            (bend_control_x, y + rise),
            (bend_control_x, y - rise),
            (top_x, y - rise),
        ),
        curve(
            (top_x, y - rise),
            (half_top, y - rise),
            (half_top, y),
            (x, y),
        ),
    ]
}

/// One clockwise arrow on the bend, on the upper half for east and lower half for west.
pub(crate) fn hairpin_arc_arrow(arcs: &[BezierCurve; 3], color: Color) -> ArrowMark {
    let bend = &arcs[1];
    let west = bend.from.0 < arcs[0].from.0;
    let (t, direction) = if west {
        (1.0 - HAIRPIN_ARROW_BEND_POSITION, -1.0)
    } else {
        (HAIRPIN_ARROW_BEND_POSITION, 1.0)
    };
    let inverse = 1.0 - t;
    let point = |axis: usize| {
        inverse * inverse * inverse * [bend.from.0, bend.from.1][axis]
            + 3.0 * inverse * inverse * t * [bend.control1.0, bend.control1.1][axis]
            + 3.0 * inverse * t * t * [bend.control2.0, bend.control2.1][axis]
            + t * t * t * [bend.to.0, bend.to.1][axis]
    };
    let tangent = |axis: usize| {
        3.0 * inverse
            * inverse
            * ([bend.control1.0, bend.control1.1][axis] - [bend.from.0, bend.from.1][axis])
            + 6.0
                * inverse
                * t
                * ([bend.control2.0, bend.control2.1][axis]
                    - [bend.control1.0, bend.control1.1][axis])
            + 3.0
                * t
                * t
                * ([bend.to.0, bend.to.1][axis] - [bend.control2.0, bend.control2.1][axis])
    };
    ArrowMark {
        center: (point(0), point(1)),
        direction: (direction * tangent(0), direction * tangent(1)),
        color,
    }
}

/// The hairpin arc of a data node's self-edge, attached to its visible side.
pub(crate) fn hairpin_edge_arcs(
    (position, node): (WorldPos, &LayoutNode),
    side: NodeSide,
    content_rects: &HashMap<WorldPos, WorldRect>,
    color: Color,
    line_style: LineStyle,
) -> [BezierCurve; 3] {
    let direction = if side == NodeSide::West { -1.0 } else { 1.0 };
    let attachment = attach(
        (position, node),
        WorldPos::new(position.x + direction as i64, position.y),
        content_rects.get(&position),
    );
    hairpin_arc(attachment, direction, color, line_style)
}

/// How far, in cells, the bend at the end of a hairpin reaches past the flat edges it joins, for
/// each row between them. A cell is about twice as tall as it is wide, so a bend two rows tall
/// reads as a semicircle when it reaches two cells.
const HAIR_PIN_BEND_REACH_PER_ROW: f64 = 1.0;

/// The bend that closes a hairpin at the far end of its hair pin: the edge between the two routing
/// nodes there, which lie in one column and carry only a self-edge. It starts and
/// ends flat, bulging away from the flat edges that reach it, so it continues them as a semicircle.
fn hair_pin_bend(
    viewport_graph: &ViewportGraph,
    (source, source_node): (WorldPos, &LayoutNode),
    (target, target_node): (WorldPos, &LayoutNode),
    color: Color,
    line_style: LineStyle,
) -> Option<RoutedCurve> {
    let between_routing_nodes = matches!(source_node.role, NodeRole::Routing)
        && matches!(target_node.role, NodeRole::Routing);
    if !between_routing_nodes || source.x != target.x || source.y == target.y {
        return None;
    }
    let bundle = viewport_graph.graph.edge_weight(source, target)?;
    if bundle.is_empty() || !bundle.iter().all(|(first, second)| first == second) {
        return None;
    }
    let inward = viewport_graph
        .neighbors(source)
        .find(|&neighbor| neighbor != target)?;
    let direction = if inward.x < source.x { 1.0 } else { -1.0 };
    // A cubic with both handles this far out reaches three quarters of the way to them.
    let rows = (source.y - target.y).abs() as f64;
    let handle = direction * rows * HAIR_PIN_BEND_REACH_PER_ROW * 4.0 / 3.0;
    let from = (source.x as f64 + 0.5, source.y as f64 + 0.5);
    let to = (target.x as f64 + 0.5, target.y as f64 + 0.5);
    Some(RoutedCurve {
        curve: BezierCurve {
            from,
            control1: (from.0 + handle, from.1),
            control2: (to.0 + handle, to.1),
            to,
            color,
            line_style,
        },
        stub: None,
        is_turnaround: true,
    })
}

/// Join the two turnaround ends of a contracted self-link without extending toward either pin.
fn self_link_span_curve(
    viewport_graph: &ViewportGraph,
    (source, source_node): (WorldPos, &LayoutNode),
    (target, target_node): (WorldPos, &LayoutNode),
    content_rects: &HashMap<WorldPos, WorldRect>,
    color: Color,
    line_style: LineStyle,
) -> Option<RoutedCurve> {
    let opposite_sides = matches!(source_node.role, NodeRole::Pin(PinSide::Left))
        && matches!(target_node.role, NodeRole::Pin(PinSide::Right))
        || matches!(source_node.role, NodeRole::Pin(PinSide::Right))
            && matches!(target_node.role, NodeRole::Pin(PinSide::Left));
    if !opposite_sides {
        return None;
    }
    let first = loopback_at(viewport_graph, source, content_rects)?;
    let second = loopback_at(viewport_graph, target, content_rects)?;
    if first.anchor != second.anchor || first.returning != target || second.returning != source {
        return None;
    }
    Some(RoutedCurve {
        curve: straight_bezier(
            (first.start.0, first.return_end.1),
            (second.start.0, second.return_end.1),
            color,
            line_style,
        ),
        stub: None,
        is_turnaround: false,
    })
}

fn loopback_edge_curve(
    viewport_graph: &ViewportGraph,
    (source, source_node): (WorldPos, &LayoutNode),
    (target, target_node): (WorldPos, &LayoutNode),
    content_rects: &HashMap<WorldPos, WorldRect>,
    color: Color,
    line_style: LineStyle,
) -> Option<RoutedCurve> {
    let is_loopback_pin =
        |role: &NodeRole| matches!(role, NodeRole::Pin(side) if side.is_loopback());
    let (pin, other) = if is_loopback_pin(&source_node.role) {
        (source, target)
    } else if is_loopback_pin(&target_node.role) {
        (target, source)
    } else {
        return None;
    };
    let loopback = loopback_at(viewport_graph, pin, content_rects)?;
    // The loop is traced outward from its anchor: turnaround first, then the return run.
    let (canonical, canonical_source, is_turnaround, stub) = if other == loopback.anchor {
        let curve = loopback_curve(
            loopback.start,
            loopback.return_end.1,
            loopback.side,
            color,
            line_style,
        );
        let stub = DiagonalLine {
            from: loopback.anchor_end,
            to: loopback.start,
            color,
            line_style,
        };
        (curve, loopback.anchor, true, Some(stub))
    } else if other == loopback.returning {
        let corner = (loopback.start.0, loopback.return_end.1);
        (
            straight_bezier(corner, loopback.return_end, color, line_style),
            pin,
            false,
            None,
        )
    } else {
        return None;
    };
    let curve = if source == canonical_source {
        canonical
    } else {
        canonical.reversed()
    };
    Some(RoutedCurve {
        curve,
        stub,
        is_turnaround,
    })
}

/// Build a cubic whose control points lie on the straight segment between its endpoints.
pub(crate) fn straight_bezier(
    from: (f64, f64),
    to: (f64, f64),
    color: Color,
    line_style: LineStyle,
) -> BezierCurve {
    let interpolate = |fraction: f64| {
        (
            from.0 + (to.0 - from.0) * fraction,
            from.1 + (to.1 - from.1) * fraction,
        )
    };
    BezierCurve {
        from,
        control1: interpolate(1.0 / 3.0),
        control2: interpolate(2.0 / 3.0),
        to,
        color,
        line_style,
    }
}

/// Sample a cubic at evenly spaced parameter values for the terminal dot rasterizer.
pub(crate) fn sample_bezier(curve: &BezierCurve, steps: usize) -> Vec<(f64, f64)> {
    (0..=steps)
        .map(|step| {
            let t = step as f64 / steps as f64;
            let inverse = 1.0 - t;
            let weights = (
                inverse.powi(3),
                3.0 * inverse.powi(2) * t,
                3.0 * inverse * t.powi(2),
                t.powi(3),
            );
            (
                weights.0 * curve.from.0
                    + weights.1 * curve.control1.0
                    + weights.2 * curve.control2.0
                    + weights.3 * curve.to.0,
                weights.0 * curve.from.1
                    + weights.1 * curve.control1.1
                    + weights.2 * curve.control2.1
                    + weights.3 * curve.to.1,
            )
        })
        .collect()
}

/// Draw a sampled curve with one dot per raster step and continuous dash spacing.
pub(crate) fn draw_braille_bezier(buffer: &mut WorldBuffer, curve: &BezierCurve) {
    let points = sample_bezier(curve, 128);
    draw_braille_path(buffer, &points, curve.color, curve.line_style);
}

/// Braille dots per cell, horizontally and vertically.
const DOTS_X: i64 = 2;
const DOTS_Y: i64 = 4;

/// Dot bits of a braille cell, indexed `[row from the top][column]`.
const DOT_BITS: [[u8; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

/// Every fourth path dot is omitted for [`LineStyle::Dashed`].
const DASH_PERIOD: usize = 4;

/// Whether `character` is a braille pattern, which a line may add dots to.
pub(crate) fn is_braille(character: char) -> bool {
    matches!(character, '\u{2800}'..='\u{28FF}')
}

/// The ends of the straight line for the edge between `from` and `to`, in continuous world
/// coordinates (cell `(x, y)` spans `[x, x + 1) x [y, y + 1)`, y up). A line leaves a data or
/// wormhole node from the border facing the other end, at its attachment row. Routing nodes and
/// pins are met at their centers, so a chain of flat edges through them is one continuous line.
pub(crate) fn segment_endpoints(
    from: (WorldPos, &LayoutNode),
    to: (WorldPos, &LayoutNode),
    content_rects: &HashMap<WorldPos, WorldRect>,
) -> ((f64, f64), (f64, f64)) {
    (
        attach(from, to.0, content_rects.get(&from.0)),
        attach(to, from.0, content_rects.get(&to.0)),
    )
}

fn attach(
    (position, node): (WorldPos, &LayoutNode),
    other: WorldPos,
    content: Option<&WorldRect>,
) -> (f64, f64) {
    let row = position.y as f64 + 0.5;
    match node.role {
        NodeRole::Data(_) | NodeRole::Wormhole(_) => {
            let rect = content
                .copied()
                .unwrap_or_else(|| node.vertical_anchor.rect(position, node.size));
            let column = match other.x.cmp(&position.x) {
                core::cmp::Ordering::Greater => rect.max.x as f64 + 1.0,
                core::cmp::Ordering::Less => rect.min.x as f64,
                core::cmp::Ordering::Equal => position.x as f64 + 0.5,
            };
            (column, row)
        }
        // A hair pin is replaced by routing nodes before anything is drawn.
        NodeRole::Routing | NodeRole::Pin(_) | NodeRole::HairPin(_) => {
            (position.x as f64 + 0.5, row)
        }
    }
}

/// Draw the line from `from` to `to` (continuous terminal coordinates) into `buffer` as braille
/// dots. Dots merge into braille cells already there and never overwrite other glyphs.
pub(crate) fn draw_braille_line(
    buffer: &mut WorldBuffer,
    from: (f64, f64),
    to: (f64, f64),
    color: Color,
    line_style: LineStyle,
) {
    draw_braille_path(buffer, &[from, to], color, line_style);
}

/// The dot axis a path step advances along, which is where it lights exactly one dot per unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DotAxis {
    Column,
    Row,
}

type PathDot = ((i64, i64), DotAxis);

impl DotAxis {
    const fn major(self, dot: (i64, i64)) -> i64 {
        match self {
            Self::Column => dot.0,
            Self::Row => dot.1,
        }
    }
}

/// Append the dots a segment (in continuous dot units) lights, in path order. The dominant axis
/// gets one dot per dot line, sampled at the line's center, so shallow stretches stay one dot
/// thick instead of stair-stepping.
fn push_segment_dots(dots: &mut Vec<PathDot>, from: (f64, f64), to: (f64, f64)) {
    let delta_x = to.0 - from.0;
    let delta_y = to.1 - from.1;
    let (axis, start, end) = if delta_x.abs() >= delta_y.abs() {
        (DotAxis::Column, from.0, to.0)
    } else {
        (DotAxis::Row, from.1, to.1)
    };
    let low = start.min(end);
    let high = start.max(end);
    let first = (low - 0.5).ceil() as i64;
    let last = (high - 0.5).floor() as i64;
    let lines: Vec<i64> = if start <= end {
        (first..=last).collect()
    } else {
        (first..=last).rev().collect()
    };
    for line in lines {
        let center = line as f64 + 0.5;
        let fraction = (center - start) / (end - start);
        let dot = match axis {
            DotAxis::Column => (line, (from.1 + delta_y * fraction).floor() as i64),
            DotAxis::Row => ((from.0 + delta_x * fraction).floor() as i64, line),
        };
        if dots.last().map(|&(previous, _)| previous) != Some(dot) {
            dots.push((dot, axis));
        }
    }
}

/// Rasterize a polyline into path-ordered dots. Endpoints are always lit unless their dot line
/// already holds a dot, so short paths and line ends are not lost.
fn rasterize_path(points: &[(f64, f64)]) -> Vec<(i64, i64)> {
    let to_dot_units = |(x, y): (f64, f64)| (x * DOTS_X as f64, y * DOTS_Y as f64);
    let floor_dot = |point: (f64, f64)| (point.0.floor() as i64, point.1.floor() as i64);
    let scaled: Vec<(f64, f64)> = points.iter().copied().map(to_dot_units).collect();
    let mut dots: Vec<PathDot> = Vec::new();
    for segment in scaled.windows(2) {
        push_segment_dots(&mut dots, segment[0], segment[1]);
    }
    if let (Some(&first), Some(&last)) = (scaled.first(), scaled.last()) {
        let start = floor_dot(first);
        let end = floor_dot(last);
        match (dots.first().copied(), dots.last().copied()) {
            (Some((head, head_axis)), Some((tail, tail_axis))) => {
                if head_axis.major(head) != head_axis.major(start) {
                    dots.insert(0, (start, head_axis));
                }
                if tail_axis.major(tail) != tail_axis.major(end) {
                    dots.push((end, tail_axis));
                }
            }
            _ => dots.push((start, DotAxis::Column)),
        }
    }
    let mut path: Vec<(i64, i64)> = Vec::with_capacity(dots.len());
    for (dot, _) in dots {
        if path.last() != Some(&dot) {
            path.push(dot);
        }
    }
    path
}

fn draw_braille_path(
    buffer: &mut WorldBuffer,
    points: &[(f64, f64)],
    color: Color,
    line_style: LineStyle,
) {
    let mut cells: HashMap<(i64, i64), u8> = HashMap::new();
    for (index, (dot_x, dot_y)) in rasterize_path(points).into_iter().enumerate() {
        if line_style == LineStyle::Dashed && index % DASH_PERIOD == DASH_PERIOD - 1 {
            continue;
        }
        let row_from_top = dot_y.rem_euclid(DOTS_Y) as usize;
        let column = dot_x.rem_euclid(DOTS_X) as usize;
        *cells
            .entry((dot_x.div_euclid(DOTS_X), dot_y.div_euclid(DOTS_Y)))
            .or_default() |= DOT_BITS[row_from_top][column];
    }

    for ((x, y), bits) in cells {
        let position = WorldPos::new(x, y);
        let Some((existing, _)) = buffer.get_terminal_char_styled(position) else {
            continue;
        };
        let merged = if is_braille(existing) {
            bits | (existing as u32 - 0x2800) as u8
        } else if existing == ' ' {
            bits
        } else {
            continue;
        };
        let glyph = char::from_u32(0x2800 + u32::from(merged)).unwrap_or(' ');
        buffer.set_terminal_char_styled(position, glyph, Style::default().fg(color));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use petgraph::graph::NodeIndex;
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};

    use super::{
        BendArch, BezierCurve, DOT_BITS, HAIRPIN_ARC_REACH_CELLS, LoopbackNeighbors, LoopbackSide,
        bend_arch, bezier_curve, draw_braille_bezier, draw_braille_line, hairpin_arc,
        hairpin_arc_arrow, hairpin_edge_arcs, is_braille, loopback_curve, loopback_neighbors,
        loopback_reach, routed_edge_curve, sample_bezier, segment_endpoints,
    };
    use crate::{
        edge_ports::NodeSide,
        geometry::{LocalPos, WorldPos, WorldRect},
        layout::{LayoutNode, NodeRole, PinSide},
        plotter::LineStyle,
        viewport_graph::ViewportGraph,
        viewport_state::{ViewportState, WorldBuffer},
    };

    fn state() -> ViewportState {
        ViewportState {
            viewport_bounds: Rect::new(0, 0, 8, 4),
            ..ViewportState::default()
        }
    }

    fn rendered(from: (f64, f64), to: (f64, f64), line_style: LineStyle) -> Vec<String> {
        let state = state();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
        draw_braille_line(&mut world_buffer, from, to, Color::Red, line_style);
        (0..4)
            .map(|row| {
                (0..8)
                    .map(|col| buffer[(col, row)].symbol().to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_edges_attach_to_sequence_instead_of_annotation_margins() {
        let source_position = WorldPos::new(0, 0);
        let target_position = WorldPos::new(80, 5);
        let source = LayoutNode::new(
            NodeRole::Data(NodeIndex::new(0)),
            LocalPos::new_xy(0, 0),
            (40, 5),
            None,
        );
        let target = LayoutNode::new(
            NodeRole::Data(NodeIndex::new(1)),
            LocalPos::new_xy(80, 5),
            (50, 7),
            None,
        );
        let content = HashMap::from([
            (source_position, WorldRect::from_coords(-2, 0, 2, 0)),
            (target_position, WorldRect::from_coords(78, 5, 82, 5)),
        ]);
        assert_eq!(
            segment_endpoints(
                (source_position, &source),
                (target_position, &target),
                &content
            ),
            ((3.0, 0.5), (78.0, 5.5))
        );
        assert_eq!(
            segment_endpoints(
                (target_position, &target),
                (source_position, &source),
                &content
            ),
            ((78.0, 5.5), (3.0, 0.5))
        );
    }

    fn data_node(x: i64, y: i64) -> LayoutNode {
        LayoutNode::new(
            NodeRole::Data(NodeIndex::new(0)),
            LocalPos::new_xy(x, y),
            (1, 1),
            None,
        )
    }

    fn plain_node(role: NodeRole, x: i64, y: i64) -> LayoutNode {
        LayoutNode::new(role, LocalPos::new_xy(x, y), (1, 1), None)
    }

    /// A data node at `anchor` and a loopback pin at `pin`, whose other neighbor is a routing node
    /// at `returning`.
    fn loopback_graph(
        anchor: WorldPos,
        pin: WorldPos,
        returning: WorldPos,
    ) -> (ViewportGraph, [WorldPos; 3]) {
        let mut graph = ViewportGraph::empty();
        graph
            .node_data_by_pos
            .insert(anchor, data_node(anchor.x, anchor.y));
        graph
            .node_data_by_pos
            .insert(pin, plain_node(NodeRole::Pin(PinSide::Left), pin.x, pin.y));
        graph.node_data_by_pos.insert(
            returning,
            plain_node(NodeRole::Routing, returning.x, returning.y),
        );
        graph.graph.add_edge(anchor, pin, Vec::new());
        graph.graph.add_edge(pin, returning, Vec::new());
        (graph, [anchor, pin, returning])
    }

    fn curve_between(graph: &ViewportGraph, source: WorldPos, target: WorldPos) -> BezierCurve {
        routed_edge_curve(
            graph,
            (source, &graph.node_data_by_pos[&source]),
            (target, &graph.node_data_by_pos[&target]),
            &HashMap::new(),
            Color::Red,
            LineStyle::Normal,
        )
        .curve
    }

    fn extremes(curve: &BezierCurve) -> (f64, f64) {
        let xs: Vec<f64> = sample_bezier(curve, 1024)
            .into_iter()
            .map(|(x, _)| x)
            .collect();
        (
            xs.iter().copied().fold(f64::INFINITY, f64::min),
            xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }

    #[test]
    fn test_rightward_loopback_returning_above_start() {
        let curve = loopback_curve(
            (10.0, 2.5),
            8.5,
            LoopbackSide::Right,
            Color::Red,
            LineStyle::Normal,
        );
        let reach = loopback_reach(6.0);
        let handle = 2.0 * 6.0 / 3.0;
        assert_eq!(curve.from, (10.0, 2.5));
        assert_eq!(curve.control1, (10.0 + handle, 2.5));
        assert_eq!(curve.control2, (10.0 + handle, 8.5));
        assert_eq!(curve.to, (10.0, 8.5));
        let (min_x, max_x) = extremes(&curve);
        assert!((min_x - 10.0).abs() < 1e-9, "it should never bulge left");
        assert!((max_x - (10.0 + reach)).abs() < 1e-3);
    }

    #[test]
    fn test_rightward_loopback_returning_below_start() {
        let curve = loopback_curve(
            (10.0, 8.5),
            2.5,
            LoopbackSide::Right,
            Color::Red,
            LineStyle::Normal,
        );
        assert_eq!(curve.control1, (14.0, 8.5));
        assert_eq!(curve.control2, (14.0, 2.5));
        assert_eq!(curve.to, (10.0, 2.5));
        let (_, max_x) = extremes(&curve);
        assert!((max_x - 13.0).abs() < 1e-3);
    }

    #[test]
    fn test_leftward_loopback_mirrors_rightward_loopback() {
        for return_y in [8.5, -3.5] {
            let right = loopback_curve(
                (10.0, 2.5),
                return_y,
                LoopbackSide::Right,
                Color::Red,
                LineStyle::Normal,
            );
            let left = loopback_curve(
                (10.0, 2.5),
                return_y,
                LoopbackSide::Left,
                Color::Red,
                LineStyle::Normal,
            );
            let mirror = |(x, y): (f64, f64)| (20.0 - x, y);
            assert_eq!(left, right.mapped(mirror));
            let (min_x, max_x) = extremes(&left);
            assert!((max_x - 10.0).abs() < 1e-9, "it should never bulge right");
            assert!((min_x - (10.0 - (return_y - 2.5).abs() / 2.0)).abs() < 1e-3);
        }
    }

    #[test]
    fn test_loopback_is_horizontal_at_both_ends() {
        for side in [LoopbackSide::Right, LoopbackSide::Left] {
            let curve = loopback_curve((4.0, 1.5), -6.5, side, Color::Red, LineStyle::Normal);
            assert_eq!(curve.control1.1, curve.from.1, "leaves horizontally");
            assert_eq!(curve.control2.1, curve.to.1, "arrives horizontally");
            let outward = match side {
                LoopbackSide::Right => 1.0,
                LoopbackSide::Left => -1.0,
            };
            assert!(outward * (curve.control1.0 - curve.from.0) > 0.0);
            assert!(
                outward * (curve.to.0 - curve.control2.0) < 0.0,
                "it should return toward the anchor"
            );
        }
    }

    #[test]
    fn test_loopback_edges_split_into_turnaround_and_return_run() {
        for returning_x in [-3, -20] {
            let (graph, [anchor, pin, returning]) = loopback_graph(
                WorldPos::new(0, 0),
                WorldPos::new(6, 2),
                WorldPos::new(returning_x, -5),
            );
            let turnaround = curve_between(&graph, anchor, pin);
            let run = curve_between(&graph, pin, returning);
            // The bend starts one cell beyond the anchor's right border.
            assert_eq!(turnaround.from, (2.0, 0.5));
            assert_eq!(turnaround.to, (2.0, -4.5));
            assert_eq!(run.from, turnaround.to, "the run continues the turnaround");
            assert_eq!(run.to, (f64::from(returning_x as i32) + 0.5, -4.5));
            assert_eq!(run.from.1, run.to.1, "the return run is horizontal");
            assert_eq!(run.control1.1, run.from.1);
            assert_eq!(run.control2.1, run.to.1);
        }
    }

    #[test]
    fn test_loopback_bend_clears_both_variable_width_endpoints() {
        for (anchor_width, returning_width, expected_bend) in [(20, 2, 12.0), (2, 30, 7.0)] {
            let (mut graph, [anchor, pin, returning]) = loopback_graph(
                WorldPos::new(0, 2),
                WorldPos::new(30, 4),
                WorldPos::new(-10, -4),
            );
            graph.node_data_by_pos.insert(
                anchor,
                LayoutNode::new(
                    NodeRole::Data(NodeIndex::new(0)),
                    LocalPos::new_xy(0, 2),
                    (anchor_width, 1),
                    None,
                ),
            );
            graph.node_data_by_pos.insert(
                returning,
                LayoutNode::new(
                    NodeRole::Data(NodeIndex::new(1)),
                    LocalPos::new_xy(-10, -4),
                    (returning_width, 1),
                    None,
                ),
            );
            let routed = routed_edge_curve(
                &graph,
                (anchor, &graph.node_data_by_pos[&anchor]),
                (pin, &graph.node_data_by_pos[&pin]),
                &HashMap::new(),
                Color::Red,
                LineStyle::Normal,
            );
            let run = curve_between(&graph, pin, returning);
            let stub = routed.stub.expect("should have an anchor stub");
            assert_eq!(routed.curve.from.0, expected_bend);
            assert_eq!(routed.curve.to.0, expected_bend);
            assert_eq!(run.from, routed.curve.to);
            assert_eq!(stub.to, routed.curve.from);
            assert!(stub.to.0 - stub.from.0 >= 1.0);
            assert!(run.from.0 - run.to.0 >= 1.0);
            assert_eq!(routed.curve.control1.1, routed.curve.from.1);
            assert_eq!(routed.curve.control2.1, routed.curve.to.1);
        }
        for (anchor_width, returning_width, expected_bend) in [(20, 2, 0.0), (2, 30, 9.0)] {
            let anchor = WorldPos::new(10, 2);
            let pin = WorldPos::new(-20, 4);
            let returning = WorldPos::new(30, -4);
            let (mut graph, _) = loopback_graph(anchor, pin, returning);
            for (position, width, index) in
                [(anchor, anchor_width, 0), (returning, returning_width, 1)]
            {
                graph.node_data_by_pos.insert(
                    position,
                    LayoutNode::new(
                        NodeRole::Data(NodeIndex::new(index)),
                        LocalPos::new_xy(position.x, position.y),
                        (width, 1),
                        None,
                    ),
                );
            }
            let routed = routed_edge_curve(
                &graph,
                (anchor, &graph.node_data_by_pos[&anchor]),
                (pin, &graph.node_data_by_pos[&pin]),
                &HashMap::new(),
                Color::Red,
                LineStyle::Normal,
            );
            let run = curve_between(&graph, pin, returning);
            let stub = routed.stub.expect("should have an anchor stub");
            assert_eq!(routed.curve.from.0, expected_bend);
            assert_eq!(routed.curve.to.0, expected_bend);
            assert_eq!(run.from, routed.curve.to);
            assert_eq!(stub.to, routed.curve.from);
            assert!(stub.from.0 - stub.to.0 >= 1.0);
            assert!(run.to.0 - run.from.0 >= 1.0);
        }
    }

    #[test]
    fn test_variable_width_loopbacks_braille_snapshot() {
        for (anchor_width, returning_width) in [(20, 2), (2, 30)] {
            let (mut graph, [anchor, pin, returning]) = loopback_graph(
                WorldPos::new(20, 2),
                WorldPos::new(55, 4),
                WorldPos::new(4, -4),
            );
            for (position, width, index) in
                [(anchor, anchor_width, 0), (returning, returning_width, 1)]
            {
                graph.node_data_by_pos.insert(
                    position,
                    LayoutNode::new(
                        NodeRole::Data(NodeIndex::new(index)),
                        LocalPos::new_xy(position.x, position.y),
                        (width, 1),
                        None,
                    ),
                );
            }
            let turnaround = routed_edge_curve(
                &graph,
                (anchor, &graph.node_data_by_pos[&anchor]),
                (pin, &graph.node_data_by_pos[&pin]),
                &HashMap::new(),
                Color::Red,
                LineStyle::Normal,
            );
            let run = curve_between(&graph, pin, returning);
            let transform = |(x, y): (f64, f64)| (x + 8.0, 8.0 - y);
            let state = ViewportState {
                viewport_bounds: Rect::new(0, 0, 55, 14),
                ..ViewportState::default()
            };
            let mut buffer = Buffer::empty(state.viewport_bounds);
            let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
            let stub = turnaround.stub.expect("should have an anchor stub");
            draw_braille_line(
                &mut world_buffer,
                transform(stub.from),
                transform(stub.to),
                Color::Red,
                LineStyle::Normal,
            );
            draw_braille_bezier(&mut world_buffer, &turnaround.curve.mapped(transform));
            draw_braille_bezier(&mut world_buffer, &run.mapped(transform));
            let rows = (0..14)
                .map(|row| {
                    (0..55)
                        .map(|column| buffer[(column, row)].symbol().to_string())
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join("\n");
            insta::assert_snapshot!(
                format!("loopback_anchor_{anchor_width}_returning_{returning_width}"),
                rows
            );
        }
    }

    #[test]
    fn test_loopback_edges_follow_the_requested_direction() {
        let (graph, [anchor, pin, returning]) = loopback_graph(
            WorldPos::new(0, 0),
            WorldPos::new(6, 2),
            WorldPos::new(-4, -5),
        );
        let forward = curve_between(&graph, anchor, pin);
        let backward = curve_between(&graph, pin, anchor);
        assert_eq!(backward.from, forward.to);
        assert_eq!(backward.control1, forward.control2);
        assert_eq!(backward.control2, forward.control1);
        assert_eq!(backward.to, forward.from);
        let run = curve_between(&graph, pin, returning);
        let reversed_run = curve_between(&graph, returning, pin);
        assert_eq!(reversed_run.from, run.to);
        assert_eq!(reversed_run.to, run.from);
    }

    #[test]
    fn test_contracted_self_link_span_joins_both_turnarounds() {
        let anchor = WorldPos::new(10, 0);
        let left = WorldPos::new(2, 3);
        let right = WorldPos::new(18, 3);
        let mut graph = ViewportGraph::empty();
        graph.node_data_by_pos.insert(anchor, data_node(10, 0));
        graph
            .node_data_by_pos
            .insert(left, plain_node(NodeRole::Pin(PinSide::Left), 2, 3));
        graph
            .node_data_by_pos
            .insert(right, plain_node(NodeRole::Pin(PinSide::Right), 18, 3));
        graph.graph.add_edge(anchor, left, Vec::new());
        graph.graph.add_edge(left, right, Vec::new());
        graph.graph.add_edge(right, anchor, Vec::new());

        let left_turn = curve_between(&graph, anchor, left);
        let right_turn = curve_between(&graph, anchor, right);
        let span = curve_between(&graph, left, right);
        assert_eq!(span.from, left_turn.to);
        assert_eq!(span.to, right_turn.to);
    }

    #[test]
    fn test_leftward_loopback_pin_turns_around_to_the_left() {
        let (graph, [anchor, pin, _]) = loopback_graph(
            WorldPos::new(10, 0),
            WorldPos::new(4, 2),
            WorldPos::new(14, -5),
        );
        let turnaround = curve_between(&graph, anchor, pin);
        // The bend starts one cell beyond the anchor's left border.
        assert_eq!(turnaround.from, (9.0, 0.5));
        assert!(turnaround.control1.0 < turnaround.from.0);
        let (min_x, max_x) = extremes(&turnaround);
        assert!((max_x - 9.0).abs() < 1e-9);
        assert!((min_x - 6.5).abs() < 1e-3);
    }

    #[test]
    fn test_loopback_does_not_depend_on_the_pin_coordinates() {
        let anchor = WorldPos::new(0, 0);
        let returning = WorldPos::new(-4, -5);
        let (near, [_, near_pin, _]) = loopback_graph(anchor, WorldPos::new(2, 9), returning);
        let (far, [_, far_pin, _]) = loopback_graph(anchor, WorldPos::new(40, -30), returning);
        let near_curve = curve_between(&near, anchor, near_pin);
        let far_curve = curve_between(&far, anchor, far_pin);
        assert_eq!(near_curve, far_curve);
        let (_, max_x) = extremes(&near_curve);
        assert!(
            (max_x - 4.5).abs() < 1e-3,
            "the pin is neither on the curve nor a bound on it"
        );
    }

    #[test]
    fn test_loopback_without_vertical_separation_has_no_extent() {
        let curve = loopback_curve(
            (3.0, 1.5),
            1.5,
            LoopbackSide::Right,
            Color::Red,
            LineStyle::Normal,
        );
        assert_eq!(curve.from, curve.to);
        assert_eq!(curve.control1, curve.from);
        assert_eq!(curve.control2, curve.to);
        assert_eq!(loopback_reach(0.0), 0.0);
    }

    #[test]
    fn test_pin_without_data_neighbor_keeps_ordinary_s_curve() {
        let mut graph = ViewportGraph::empty();
        let first = WorldPos::new(20, 0);
        let second = WorldPos::new(-20, -15);
        for point in [first, second] {
            graph.node_data_by_pos.insert(
                point,
                plain_node(NodeRole::Pin(PinSide::Left), point.x, point.y),
            );
        }
        graph.graph.add_edge(first, second, Vec::new());
        let curve = curve_between(&graph, first, second);
        let (from, to) = (curve.from, curve.to);
        assert_eq!(curve.control1, ((from.0 + to.0) / 2.0, from.1));
        assert_eq!(curve.control2, ((from.0 + to.0) / 2.0, to.1));
    }

    #[test]
    fn test_loopback_neighbors_take_first_data_node_as_anchor() {
        let data = NodeRole::Data(NodeIndex::new(0));
        let anchored = |anchor, returning| Some(LoopbackNeighbors { anchor, returning });
        assert_eq!(
            loopback_neighbors([(1, &data), (2, &NodeRole::Routing)]),
            anchored(1, 2)
        );
        assert_eq!(
            loopback_neighbors([(1, &NodeRole::Routing), (2, &data)]),
            anchored(2, 1)
        );
        assert_eq!(loopback_neighbors([(1, &data), (2, &data)]), anchored(1, 2));
        assert_eq!(
            loopback_neighbors([(1, &NodeRole::Routing), (2, &NodeRole::Pin(PinSide::Left))]),
            None
        );
        assert_eq!(loopback_neighbors([(1, &data)]), None);
    }

    #[test]
    fn test_braille_line_is_connected_and_stays_between_its_endpoints() {
        let rows = rendered((0.0, 0.5), (8.0, 3.5), LineStyle::Normal);
        let lit: Vec<(usize, usize)> = rows
            .iter()
            .enumerate()
            .flat_map(|(row, line)| {
                line.chars()
                    .enumerate()
                    .filter(|(_, glyph)| is_braille(*glyph))
                    .map(move |(column, _)| (row, column))
            })
            .collect();
        assert!(!lit.is_empty());
        for column in 0..8 {
            assert!(
                lit.iter().any(|&(_, lit_column)| lit_column == column),
                "line should cross column {column}: {rows:?}"
            );
        }
    }

    #[test]
    fn test_braille_line_has_one_dot_per_major_axis_step() {
        let rows = rendered((0.0, 0.5), (8.0, 3.5), LineStyle::Normal);
        for dot_x in 0..16 {
            let lit = rows
                .iter()
                .flat_map(|row| row.chars().nth(dot_x / 2))
                .map(|glyph| (glyph as u32).saturating_sub(0x2800) as u8)
                .map(|bits| {
                    (0..4)
                        .filter(|dot_row| bits & DOT_BITS[*dot_row][dot_x % 2] != 0)
                        .count()
                })
                .sum::<usize>();
            assert_eq!(lit, 1, "line thickness at dot column {dot_x}");
        }
    }

    #[test]
    fn test_braille_line_does_not_overwrite_other_glyphs() {
        let state = state();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        buffer[(3, 1)].set_char('X');
        let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
        draw_braille_line(
            &mut world_buffer,
            (0.0, 0.0),
            (8.0, 4.0),
            Color::Red,
            LineStyle::Normal,
        );
        assert_eq!(buffer[(3, 1)].symbol(), "X");
    }

    #[test]
    fn test_braille_line_uses_terminal_coordinates_and_color() {
        let mut state = state();
        state.viewport_bounds = Rect::new(3, 2, 8, 4);
        state.camera_current = WorldPos::new(100, 100);
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
        draw_braille_line(
            &mut world_buffer,
            (3.0, 2.5),
            (10.0, 2.5),
            Color::Red,
            LineStyle::Normal,
        );
        for column in 3..=10 {
            let cell = &buffer[(column, 2)];
            assert!(is_braille(cell.symbol().chars().next().unwrap_or(' ')));
            assert_eq!(cell.fg, Color::Red);
        }
    }

    #[test]
    fn test_dashed_line_has_fewer_dots_than_normal_line() {
        let dots = |rows: Vec<String>| -> u32 {
            rows.iter()
                .flat_map(|row| row.chars())
                .filter(|glyph| is_braille(*glyph))
                .map(|glyph| (glyph as u32 - 0x2800).count_ones())
                .sum()
        };
        let normal = dots(rendered((0.0, 0.5), (8.0, 0.5), LineStyle::Normal));
        let dashed = dots(rendered((0.0, 0.5), (8.0, 0.5), LineStyle::Dashed));
        assert!(dashed < normal);
    }

    #[test]
    fn test_dashed_line_omits_only_single_dots() {
        let rows = rendered((0.0, 0.5), (8.0, 0.5), LineStyle::Dashed);
        let dot_row = 2;
        for dot_x in 0..16 {
            let glyph = rows[0].chars().nth(dot_x / 2).expect("should have a cell");
            let bits = (glyph as u32).saturating_sub(0x2800) as u8;
            let lit = bits & DOT_BITS[dot_row][dot_x % 2] != 0;
            assert_eq!(lit, dot_x % 4 != 3, "dash pattern at dot {dot_x}");
        }
    }

    #[test]
    fn test_dashed_bezier_omits_only_single_dots_across_segments() {
        let state = state();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let curve = bezier_curve((0.0, 0.5), (8.0, 0.5), Color::Red, LineStyle::Dashed);
        draw_braille_bezier(&mut WorldBuffer::new(&mut buffer, &state), &curve);
        for dot_x in 0..16 {
            let glyph = buffer[(dot_x / 2, 0)]
                .symbol()
                .chars()
                .next()
                .unwrap_or(' ');
            let bits = (glyph as u32).saturating_sub(0x2800) as u8;
            let lit = bits & DOT_BITS[2][usize::from(dot_x % 2)] != 0;
            assert_eq!(lit, dot_x % 4 != 3, "curve dash at dot {dot_x}");
        }
    }

    fn lit_dots(buffer: &Buffer) -> Vec<(i64, i64)> {
        let area = buffer.area;
        let mut dots = Vec::new();
        for row in 0..area.height {
            for column in 0..area.width {
                let glyph = buffer[(column, row)].symbol().chars().next().unwrap_or(' ');
                if !is_braille(glyph) {
                    continue;
                }
                let bits = (glyph as u32 - 0x2800) as u8;
                for (dot_row, row_bits) in DOT_BITS.iter().enumerate() {
                    for (dot_column, bit) in row_bits.iter().enumerate() {
                        if bits & bit != 0 {
                            dots.push((
                                i64::from(column) * 2 + dot_column as i64,
                                i64::from(row) * 4 + dot_row as i64,
                            ));
                        }
                    }
                }
            }
        }
        dots
    }

    fn draw_into_buffer(draw: impl FnOnce(&mut WorldBuffer)) -> Buffer {
        let state = state();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        draw(&mut WorldBuffer::new(&mut buffer, &state));
        buffer
    }

    #[test]
    fn test_shallow_sloped_line_has_one_dot_per_dot_column() {
        let buffer = draw_into_buffer(|world| {
            draw_braille_line(world, (0.0, 0.3), (7.5, 2.0), Color::Red, LineStyle::Normal);
        });
        let dots = lit_dots(&buffer);
        for dot_x in 0..15 {
            let count = dots.iter().filter(|(x, _)| *x == dot_x).count();
            assert_eq!(count, 1, "dots in dot column {dot_x}: {dots:?}");
        }
    }

    #[test]
    fn test_s_curve_has_one_dot_per_dot_column_where_x_dominant() {
        let curve = bezier_curve((0.0, 0.5), (8.0, 3.5), Color::Red, LineStyle::Normal);
        let buffer = draw_into_buffer(|world| draw_braille_bezier(world, &curve));
        let dots = lit_dots(&buffer);
        let points = sample_bezier(&curve, 128);
        let mut checked = 0;
        for dot_x in 0..16_i64 {
            let center = dot_x as f64 + 0.5;
            let steep = points.windows(2).any(|pair| {
                let (low, high) = (
                    pair[0].0.min(pair[1].0) * 2.0,
                    pair[0].0.max(pair[1].0) * 2.0,
                );
                let slope_x = (pair[1].0 - pair[0].0).abs() * 2.0;
                let slope_y = (pair[1].1 - pair[0].1).abs() * 4.0;
                low <= center + 0.5 && center - 0.5 <= high && slope_y > slope_x
            });
            if steep {
                continue;
            }
            checked += 1;
            let count = dots.iter().filter(|(x, _)| *x == dot_x).count();
            assert_eq!(count, 1, "dots in dot column {dot_x}: {dots:?}");
        }
        assert!(checked > 0);
    }

    #[test]
    fn test_vertical_line_has_one_dot_per_dot_row() {
        let buffer = draw_into_buffer(|world| {
            draw_braille_line(world, (2.2, 0.0), (2.2, 3.9), Color::Red, LineStyle::Normal);
        });
        let dots = lit_dots(&buffer);
        for dot_y in 0..15 {
            let count = dots.iter().filter(|(_, y)| *y == dot_y).count();
            assert_eq!(count, 1, "dots in dot row {dot_y}: {dots:?}");
        }
    }

    #[test]
    fn test_bend_arch_follows_pin_position_instead_of_role() {
        let west = WorldPos::new(0, 0);
        let east = WorldPos::new(4, 0);
        for (side, pin_row, expected_direction) in
            [(PinSide::Up, -2, -1.0), (PinSide::Down, 2, 1.0)]
        {
            let pin = WorldPos::new(2, pin_row);
            let mut graph = ViewportGraph::empty();
            graph.node_data_by_pos.insert(west, data_node(0, 0));
            graph.node_data_by_pos.insert(east, data_node(4, 0));
            graph
                .node_data_by_pos
                .insert(pin, plain_node(NodeRole::Pin(side), 2, pin_row));
            graph.graph.add_edge(west, pin, Vec::new());
            graph.graph.add_edge(pin, east, Vec::new());

            let arch = bend_arch(&graph, pin, &HashMap::new(), Color::Red, LineStyle::Normal)
                .expect("should build a bend through the pin");
            let (_, midpoint_row) = arch.midpoint();
            assert!(
                (midpoint_row - arch.curve.from.1) * expected_direction > 0.0,
                "pin {side:?} at row {pin_row} should bend toward its own position"
            );
        }
    }

    #[test]
    fn test_bend_arch_midpoint_is_level_above_the_line_for_an_up_pin() {
        let arch = BendArch {
            west: WorldPos::new(0, 0),
            east: WorldPos::new(8, 0),
            curve: BezierCurve {
                from: (2.0, 0.5),
                control1: (3.0, 2.0),
                control2: (5.0, 2.0),
                to: (6.0, 0.5),
                color: Color::Reset,
                line_style: LineStyle::Normal,
            },
        };
        let (column, row) = arch.midpoint();
        assert_eq!(column, 4.0);
        assert!(row > 0.5, "an up arch should rise above its ends");
    }

    /// The routing nodes of a hairpin's route across its hair pin, with the bend between the two
    /// at the far end.
    fn hair_pin_route(
        near_x: i64,
        far_x: i64,
        bend_bundle: Vec<(NodeIndex, NodeIndex)>,
    ) -> ViewportGraph {
        let mut graph = ViewportGraph::empty();
        let corners = [(near_x, 1), (near_x, -1), (far_x, 1), (far_x, -1)];
        for (x, y) in corners {
            graph
                .node_data_by_pos
                .insert(WorldPos::new(x, y), plain_node(NodeRole::Routing, x, y));
        }
        let self_edge = vec![(NodeIndex::new(0), NodeIndex::new(0))];
        for (first, second) in [(corners[0], corners[2]), (corners[1], corners[3])] {
            graph.graph.add_edge(
                WorldPos::new(first.0, first.1),
                WorldPos::new(second.0, second.1),
                self_edge.clone(),
            );
        }
        graph.graph.add_edge(
            WorldPos::new(far_x, 1),
            WorldPos::new(far_x, -1),
            bend_bundle,
        );
        graph
    }

    #[test]
    fn test_hair_pin_bend_continues_the_flat_edges_as_a_semicircle() {
        for (near_x, far_x, direction) in [(0, 3, 1.0), (3, 0, -1.0)] {
            let graph = hair_pin_route(near_x, far_x, vec![(NodeIndex::new(0), NodeIndex::new(0))]);
            let (top, bottom) = (WorldPos::new(far_x, 1), WorldPos::new(far_x, -1));
            let routed = routed_edge_curve(
                &graph,
                (top, &graph.node_data_by_pos[&top]),
                (bottom, &graph.node_data_by_pos[&bottom]),
                &HashMap::new(),
                Color::Reset,
                LineStyle::Normal,
            );

            // Two rows apart, so the bend reaches two cells.
            let handle = direction * 2.0 * 4.0 / 3.0;
            let column = far_x as f64 + 0.5;
            assert!(routed.is_turnaround);
            assert_eq!(routed.curve.from, (column, 1.5));
            assert_eq!(routed.curve.control1, (column + handle, 1.5));
            assert_eq!(routed.curve.control2, (column + handle, -0.5));
            assert_eq!(routed.curve.to, (column, -0.5));
        }
    }

    #[test]
    fn test_vertical_edge_that_is_not_a_self_edge_is_not_a_hair_pin_bend() {
        let graph = hair_pin_route(0, 3, vec![(NodeIndex::new(0), NodeIndex::new(1))]);
        let (top, bottom) = (WorldPos::new(3, 1), WorldPos::new(3, -1));
        let routed = routed_edge_curve(
            &graph,
            (top, &graph.node_data_by_pos[&top]),
            (bottom, &graph.node_data_by_pos[&bottom]),
            &HashMap::new(),
            Color::Reset,
            LineStyle::Normal,
        );
        assert!(!routed.is_turnaround);
    }

    #[test]
    fn test_hairpin_arc_has_three_curves_with_level_joins() {
        for direction in [1.0, -1.0] {
            let [climb, bend, descent] =
                hairpin_arc((10.0, 4.5), direction, Color::Reset, LineStyle::Normal);
            assert_eq!(climb.from, (10.0, 4.5));
            assert_eq!(descent.to, (10.0, 4.5));
            assert_eq!(climb.to, bend.from);
            assert_eq!(bend.to, descent.from);
            assert_eq!(climb.control1.1, climb.from.1);
            assert_eq!(descent.control2.1, descent.to.1);
            assert_eq!(climb.control2.1, climb.to.1);
            assert_eq!(bend.control1.1, bend.from.1);
            assert_eq!(bend.control2.1, bend.to.1);
            assert_eq!(descent.control1.1, descent.from.1);
            let middle = sample_bezier(&bend, 2)[1];
            assert!((middle.0 - (10.0 + direction * HAIRPIN_ARC_REACH_CELLS)).abs() < 1e-9);
        }
    }

    #[test]
    fn test_hairpin_arc_attaches_to_data_content_on_both_sides() {
        let domain_index = NodeIndex::new(0);
        let anchor_position = WorldPos::new(10, 0);
        let content = HashMap::from([(anchor_position, WorldRect::from_coords(8, 0, 12, 0))]);
        let node = LayoutNode::new(
            NodeRole::Data(domain_index),
            LocalPos::new_xy(10, 0),
            (9, 1),
            None,
        );
        for (side, expected_x) in [(NodeSide::West, 8.0), (NodeSide::East, 13.0)] {
            let [climb, _, descent] = hairpin_edge_arcs(
                (anchor_position, &node),
                side,
                &content,
                Color::Red,
                LineStyle::Normal,
            );
            assert_eq!(climb.from, (expected_x, 0.5));
            assert_eq!(descent.to, climb.from);
        }
    }

    #[test]
    fn test_hairpin_arc_arrow_follows_the_bend_clockwise() {
        for direction in [1.0, -1.0] {
            let arcs = hairpin_arc((10.0, 4.5), direction, Color::Reset, LineStyle::Normal);
            let arrow = hairpin_arc_arrow(&arcs, Color::Reset);
            assert!(arrow.center.1 < arcs[1].from.1);
            assert!(arrow.center.1 > arcs[1].to.1);
            assert_eq!((arrow.center.1 - 4.5).signum(), direction);
            assert_eq!(arrow.direction.0.signum(), direction);
            assert_eq!(arrow.direction.1.signum(), -direction);
            let pixel_slope = arrow.direction.1.abs() * 21.0 / (arrow.direction.0.abs() * 11.0);
            assert!((pixel_slope - 1.0).abs() < 0.1);
        }
    }
}
