// This module implements graph rendering using the ViewportGraph system.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use petgraph::{
    graph::NodeIndex,
    visit::{GraphBase, NodeIndexable},
};
use ratatui::{
    style::{Color, Style},
    symbols::merge::MergeStrategy,
};

use crate::{
    geometry::{WorldPos, WorldRect},
    graph_widget::NODE_GLYPH,
    layout::{JunctionSymbol, NodeRole},
    theme::Theme,
    viewport_graph::ViewportGraph,
    viewport_state::WorldBuffer,
};

/// Target distance in world cells per direction marker along a backward edge's full route.
const ARROW_GAPS: i64 = 16;

/// Line style for path highlighting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStyle {
    /// Normal weight box-drawing characters
    Normal,
    /// Heavy weight box-drawing characters
    Bold,
    /// Dashed box-drawing characters
    Dashed,
}

/// Style specification for highlighted paths
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathStyle {
    /// Color to use for tinting (Color::Reset means brighten instead)
    pub color: Color,
    /// Line weight style for edges and routing nodes
    pub line_style: LineStyle,
    /// Whether to merge glyphs with base layer or replace them outright
    pub merge_glyphs: bool,
}

pub(crate) type CellHighlight = (WorldPos, (i64, i64), (i64, i64), PathStyle);

impl PathStyle {
    /// Create a new PathStyle with default settings
    pub fn new(color: Color) -> Self {
        Self {
            color,
            line_style: LineStyle::Normal,
            merge_glyphs: false,
        }
    }

    /// Set the line style
    pub fn with_line_style(mut self, line_style: LineStyle) -> Self {
        self.line_style = line_style;
        self
    }

    /// Set whether to merge glyphs
    pub fn with_merge_glyphs(mut self, merge_glyphs: bool) -> Self {
        self.merge_glyphs = merge_glyphs;
        self
    }
}

/// Domain-side trait bundling node sizing and rendering for a single, fixed
/// level of detail. A widget or event loop picks which concrete `NodeRenderer`
/// is active (e.g. per zoom level) and borrows it immutably for rendering.
pub trait NodeRenderer<G>
where
    G: GraphBase,
{
    /// Get the dimensions (width, height) for a node.
    fn get_node_size(&self, node: &G::NodeId) -> (u64, u64);

    /// Get default dimensions for dummy/routing nodes
    fn get_dummy_size(&self) -> (u64, u64) {
        (1, 1)
    }

    /// Render a node in its allocated world area.
    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId);
}

/// Forward rendering through a boxed renderer.
impl<G, T> NodeRenderer<G> for Box<T>
where
    G: GraphBase,
    T: NodeRenderer<G> + ?Sized,
{
    fn get_node_size(&self, node: &G::NodeId) -> (u64, u64) {
        (**self).get_node_size(node)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (**self).get_dummy_size()
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId) {
        (**self).render_node(buffer, area, node_id);
    }
}

/// Forward rendering through a shared renderer.
impl<G, T> NodeRenderer<G> for Arc<T>
where
    G: GraphBase,
    T: NodeRenderer<G> + ?Sized,
{
    fn get_node_size(&self, node: &G::NodeId) -> (u64, u64) {
        (**self).get_node_size(node)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (**self).get_dummy_size()
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId) {
        (**self).render_node(buffer, area, node_id);
    }
}

pub(crate) struct PlotDecorations<'a> {
    pub node_highlights: &'a [(WorldPos, PathStyle)],
    pub edge_highlights: &'a [((WorldPos, WorldPos), PathStyle)],
    pub cell_highlights: &'a [CellHighlight],
    pub lowlights: &'a [(WorldPos, WorldPos)],
    pub node_lowlights: &'a [WorldPos],
}

/// Plot a viewport graph with resolved highlights and lowlights.
pub(crate) fn plot_viewport_graph_with_highlights<V, G>(
    viewport_graph: &ViewportGraph,
    buffer: &mut WorldBuffer<'_>,
    renderer: &V,
    original_graph: &G,
    decorations: PlotDecorations<'_>,
    theme: &Theme,
) where
    V: NodeRenderer<G>,
    G: GraphBase + NodeIndexable,
{
    let PlotDecorations {
        node_highlights,
        edge_highlights,
        cell_highlights,
        lowlights,
        node_lowlights,
    } = decorations;

    // Draw edges first so nodes appear on top
    for (source, target, bundle) in viewport_graph.edges() {
        // Check if edge is in any highlighted path
        // Later highlights in the vector take precedence over earlier ones
        let highlighted_style = edge_highlights
            .iter()
            .filter(|((s, t), _)| (*s == source && *t == target) || (*s == target && *t == source))
            .map(|(_, style)| *style)
            .next_back();

        let is_lowlight = lowlights
            .iter()
            .any(|(s, t)| (*s == source && *t == target) || (*s == target && *t == source));

        if let Some(style) = highlighted_style {
            let edge_color = match style.color {
                Color::Reset => theme[0x07],
                color => color,
            };
            // highlight+lowlight: dashed in the highlight color
            let line_style = if is_lowlight {
                LineStyle::Dashed
            } else {
                style.line_style
            };
            draw_edge_with_style(buffer, source, target, edge_color, line_style);
        } else if is_lowlight {
            draw_edge_with_style(buffer, source, target, theme[0x04], LineStyle::Dashed);
        } else if !bundle.is_empty() {
            // Normal edge with data
            draw_edge_with_style(buffer, source, target, theme[0x05], LineStyle::Normal);
        } else {
            // Edges that don't actually represent an original edge
            // (edges to/from terminal source/sink nodes) - draw with dashed lines
            draw_edge_with_style(buffer, source, target, theme[0x05], LineStyle::Dashed);
        }
    }

    // Draw nodes
    for (world_pos, node) in viewport_graph.nodes() {
        match &node.role {
            NodeRole::Data(domain_idx) => {
                let node_id = <G as NodeIndexable>::from_index(original_graph, domain_idx.index());
                let world_rect = WorldRect::from_center_and_size(*world_pos, node.size);
                renderer.render_node(buffer, world_rect, &node_id);

                // If lowlighted, dim the node background to theme[0x04].
                // Applied before highlights so highlights take priority.
                if node_lowlights.contains(world_pos) {
                    let dim = theme[0x04];
                    for y in world_rect.min.y..=world_rect.max.y {
                        for x in world_rect.min.x..=world_rect.max.x {
                            let pos = WorldPos::new(x, y);
                            if let Some((ch, style)) = buffer.get_char_styled(pos) {
                                let new_style = if ch == NODE_GLYPH {
                                    style.fg(dim)
                                } else {
                                    style.bg(dim)
                                };
                                buffer.set_char_styled(pos, ch, new_style);
                            }
                        }
                    }
                }

                // Check if this node is highlighted
                let highlighted_style = node_highlights
                    .iter()
                    .filter(|(pos, _)| pos == world_pos)
                    .map(|(_, style)| *style)
                    .next_back();

                // If highlighted, apply color directly (Color::Reset → cursor highlight slot)
                if let Some(path_style) = highlighted_style {
                    let hl = match path_style.color {
                        Color::Reset => theme[0x07],
                        c => c,
                    };
                    for y in world_rect.min.y..=world_rect.max.y {
                        for x in world_rect.min.x..=world_rect.max.x {
                            let pos = WorldPos::new(x, y);
                            if let Some((ch, style)) = buffer.get_char_styled(pos) {
                                let new_style = if ch == NODE_GLYPH {
                                    style.fg(hl)
                                } else {
                                    style.bg(hl)
                                };
                                buffer.set_char_styled(pos, ch, new_style);
                            }
                        }
                    }
                }

                // Sub-rect highlight pass: tint only the matched column/row range.
                // tl/br are node-local (col, row) offsets from world_rect.min, both inclusive.
                for (_, tl, br, path_style) in
                    cell_highlights.iter().filter(|(nwp, ..)| nwp == world_pos)
                {
                    let hl = match path_style.color {
                        Color::Reset => theme[0x07],
                        color => color,
                    };
                    let x0 = (world_rect.min.x + tl.0).max(world_rect.min.x);
                    let x1 = (world_rect.min.x + br.0).min(world_rect.max.x);
                    let y0 = (world_rect.min.y + tl.1).max(world_rect.min.y);
                    let y1 = (world_rect.min.y + br.1).min(world_rect.max.y);
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let pos = WorldPos::new(x, y);
                            if let Some((ch, style)) = buffer.get_char_styled(pos) {
                                let new_style = if ch == NODE_GLYPH {
                                    style.fg(hl)
                                } else {
                                    style.bg(hl)
                                };
                                buffer.set_char_styled(pos, ch, new_style);
                            }
                        }
                    }
                }
            }
            NodeRole::Routing | NodeRole::Pin | NodeRole::Wormhole(_) => {
                let edge_color = theme[0x05];
                let base_glyph =
                    compute_junction_glyph(viewport_graph.neighbors(*world_pos), *world_pos);

                // Check if this routing node is part of any highlighted edge
                let highlighted_style = edge_highlights
                    .iter()
                    .filter(|((s, t), _)| {
                        if s.x == t.x {
                            world_pos.x == s.x
                                && world_pos.y >= s.y.min(t.y)
                                && world_pos.y <= s.y.max(t.y)
                        } else {
                            world_pos.y == s.y
                                && world_pos.x >= s.x.min(t.x)
                                && world_pos.x <= s.x.max(t.x)
                        }
                    })
                    .map(|(_, style)| *style)
                    .next_back();

                // Collect neighbors once; reused for both glyph shape and color decisions.
                let all_neighbors: Vec<WorldPos> = viewport_graph.neighbors(*world_pos).collect();

                // Highlight takes full priority: skip lowlight logic entirely when present.
                let (character, fg_color) = if let Some(style) = highlighted_style {
                    let active_edges: Vec<_> = edge_highlights
                        .iter()
                        .filter(|(_, s)| *s == style)
                        .cloned()
                        .collect();
                    let highlight_graph = ViewportGraph::from_visual_edges(&active_edges);
                    let high_glyph =
                        compute_junction_glyph(highlight_graph.neighbors(*world_pos), *world_pos);
                    let ch = if style.merge_glyphs {
                        let high_char = match style.line_style {
                            LineStyle::Normal => high_glyph.glyph(),
                            LineStyle::Bold => high_glyph.heavy_glyph(),
                            LineStyle::Dashed => high_glyph.dashed_glyph(),
                        };
                        MergeStrategy::Fuzzy
                            .merge(&base_glyph.glyph().to_string(), &high_char.to_string())
                            .chars()
                            .next()
                            .unwrap_or('?')
                    } else {
                        match style.line_style {
                            LineStyle::Normal => high_glyph.glyph(),
                            LineStyle::Bold => high_glyph.heavy_glyph(),
                            LineStyle::Dashed => high_glyph.dashed_glyph(),
                        }
                    };
                    let color = match style.color {
                        Color::Reset => theme[0x07],
                        c => c,
                    };
                    (ch, color)
                } else {
                    // Compute once, share between glyph and color decisions below.
                    let lowlight_mask: Vec<bool> = all_neighbors
                        .iter()
                        .map(|&nb| {
                            is_lowlight_only_connection(*world_pos, nb, lowlights, edge_highlights)
                        })
                        .collect();
                    let all_dim = !lowlight_mask.is_empty() && lowlight_mask.iter().all(|&x| x);
                    let any_dim = lowlight_mask.iter().any(|&x| x);
                    // Mixed junction: exclude lowlight-only arms so the glyph reflects only the
                    // live connections, avoiding spurious branches pointing into dimmed edges.
                    let ch = if any_dim && !all_dim {
                        let lit = all_neighbors
                            .iter()
                            .zip(&lowlight_mask)
                            .filter_map(|(&nb, &dim)| (!dim).then_some(nb));
                        compute_junction_glyph(lit, *world_pos).glyph()
                    } else {
                        base_glyph.glyph()
                    };
                    let color = if all_dim { theme[0x04] } else { edge_color };
                    (ch, color)
                };

                // Render each degree-one wormhole as a horizontal direction marker.
                let character = if matches!(node.role, NodeRole::Wormhole(_)) {
                    match viewport_graph.neighbors(*world_pos).next() {
                        Some(neighbor) if neighbor.x > world_pos.x => '◁',
                        Some(neighbor) if neighbor.x < world_pos.x => '▷',
                        _ => character,
                    }
                } else {
                    character
                };
                buffer.set_char_styled(*world_pos, character, Style::default().fg(fg_color));
            }
        }
    }

    // Mark each backward edge along its full route, including the legs to its endpoints.
    for &(source, target) in &viewport_graph.backward_edges {
        draw_arrows(buffer, viewport_graph, source, target, ARROW_GAPS);
    }
}

/// Returns true if both `point_a` and `point_b` lie on the axis-aligned segment
/// from `seg_start` to `seg_end`.
fn both_points_on_segment(
    seg_start: WorldPos,
    seg_end: WorldPos,
    point_a: WorldPos,
    point_b: WorldPos,
) -> bool {
    if seg_start.x == seg_end.x && point_a.x == seg_start.x && point_b.x == seg_start.x {
        let lo = seg_start.y.min(seg_end.y);
        let hi = seg_start.y.max(seg_end.y);
        point_a.y >= lo && point_a.y <= hi && point_b.y >= lo && point_b.y <= hi
    } else if seg_start.y == seg_end.y && point_a.y == seg_start.y && point_b.y == seg_start.y {
        let lo = seg_start.x.min(seg_end.x);
        let hi = seg_start.x.max(seg_end.x);
        point_a.x >= lo && point_a.x <= hi && point_b.x >= lo && point_b.x <= hi
    } else {
        false
    }
}

/// Returns true if the connection from `routing_node` to `neighbor` lies on a lowlight
/// segment but not on any highlight segment — meaning it should be rendered dimmed.
fn is_lowlight_only_connection(
    routing_node: WorldPos,
    neighbor: WorldPos,
    lowlights: &[(WorldPos, WorldPos)],
    edge_highlights: &[((WorldPos, WorldPos), PathStyle)],
) -> bool {
    let on_lowlight = lowlights
        .iter()
        .any(|(start, end)| both_points_on_segment(*start, *end, routing_node, neighbor));
    let on_highlight = edge_highlights
        .iter()
        .any(|((start, end), _)| both_points_on_segment(*start, *end, routing_node, neighbor));
    on_lowlight && !on_highlight
}

/// Compute the junction glyph for a routing node given its neighbors.
fn compute_junction_glyph(
    neighbors: impl Iterator<Item = WorldPos>,
    pos: WorldPos,
) -> JunctionSymbol {
    let mut connections = 0u8;
    for neighbor in neighbors {
        if neighbor.y < pos.y {
            connections |= 0b0010; // South
        } else if neighbor.y > pos.y {
            connections |= 0b1000; // North
        }
        if neighbor.x < pos.x {
            connections |= 0b0001; // West
        } else if neighbor.x > pos.x {
            connections |= 0b0100; // East
        }
    }
    JunctionSymbol::new(connections)
}

/// Draw a rectilinear edge between two world positions (horizontal or vertical only),
/// using the specified line style for box-drawing characters.
fn draw_edge_with_style(
    buffer: &mut WorldBuffer,
    source: WorldPos,
    target: WorldPos,
    color: Color,
    line_style: LineStyle,
) {
    let style = Style::default().fg(color);

    let (v_ch, h_ch) = match line_style {
        LineStyle::Normal => ('│', '─'),
        LineStyle::Bold => ('┃', '━'),
        LineStyle::Dashed => ('┊', '╌'),
    };

    if source.x == target.x {
        // Vertical edge
        let (min_y, max_y) = if source.y < target.y {
            (source.y, target.y)
        } else {
            (target.y, source.y)
        };

        for y in min_y..=max_y {
            buffer.set_char_styled(WorldPos::new(source.x, y), v_ch, style);
        }
    } else if source.y == target.y {
        // Horizontal edge
        let (min_x, max_x) = if source.x < target.x {
            (source.x, target.x)
        } else {
            (target.x, source.x)
        };

        for x in min_x..=max_x {
            let pos = WorldPos::new(x, source.y);

            // Don't overwrite a vertical line with a horizontal one.
            // Vertical edges take priority at crossings (normal, heavy, and dashed).
            if !matches!(
                buffer.get_char(pos),
                Some('│') | Some('┃') | Some('┊') | Some('┆')
            ) {
                buffer.set_char_styled(pos, h_ch, style);
            }
        }
    }
}

/// Trace a backward edge's routed segments in domain source-to-target order.
/// Bundles distinguish the route from other edges sharing its junctions.
fn backward_edge_route(
    viewport_graph: &ViewportGraph,
    source: NodeIndex,
    target: NodeIndex,
) -> Option<Vec<WorldPos>> {
    let &source_pos = viewport_graph.node_positions.get(&source)?;
    let &target_pos = viewport_graph.node_positions.get(&target)?;
    let follows_edge = |start, end| {
        viewport_graph
            .graph
            .edge_weight(start, end)
            .is_some_and(|bundle| bundle.contains(&(source, target)))
    };

    // A self-loop has the same domain endpoint twice, so its undirected bundle alone
    // cannot distinguish the two directions. The layered router attaches outgoing
    // legs to the node's right port and incoming legs to its left port.
    let first_pos = if source == target {
        viewport_graph
            .neighbors(source_pos)
            .find(|&neighbor| neighbor.x > source_pos.x && follows_edge(source_pos, neighbor))?
    } else {
        source_pos
    };
    let mut previous = HashMap::from([(first_pos, source_pos)]);
    let mut pending = VecDeque::from([first_pos]);
    while let Some(position) = pending.pop_front() {
        if position == target_pos {
            let mut route = vec![position];
            let mut current = position;
            while current != first_pos {
                current = previous[&current];
                route.push(current);
            }
            if source == target {
                route.push(source_pos);
            }
            route.reverse();
            return Some(route);
        }
        for neighbor in viewport_graph.neighbors(position) {
            // Close a self-loop through its incoming leg, not by retracing its first segment.
            if source == target && position == first_pos && neighbor == source_pos {
                continue;
            }
            if follows_edge(position, neighbor) && !previous.contains_key(&neighbor) {
                previous.insert(neighbor, position);
                pending.push_back(neighbor);
            }
        }
    }
    None
}

/// Distribute direction markers by distance along the full routed backward edge.
/// Endpoint gaps and inter-marker gaps are equal to within one cell after rounding;
/// segment boundaries never restart the spacing. At bends, markers point along the
/// outgoing segment; data nodes keep their contents when they cover a marker.
fn draw_arrows(
    buffer: &mut WorldBuffer,
    viewport_graph: &ViewportGraph,
    source: NodeIndex,
    target: NodeIndex,
    gaps: i64,
) {
    let Some(route) = backward_edge_route(viewport_graph, source, target) else {
        return;
    };
    let total_length: i64 = route
        .windows(2)
        .map(|segment| (segment[1].x - segment[0].x).abs() + (segment[1].y - segment[0].y).abs())
        .sum();
    if total_length < 2 {
        return;
    }
    let arrow_count = (total_length / gaps).max(1);
    let intervals = arrow_count + 1;
    let mut arrow_index = 1;
    let mut traced_length = 0;
    for segment in route.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let delta = end - start;
        let length = delta.x.abs() + delta.y.abs();
        let direction = WorldPos::new(delta.x.signum(), delta.y.signum());
        let arrow = match (direction.x, direction.y) {
            (1, 0) => '▶',
            (-1, 0) => '◀',
            (0, 1) => '▲',
            (0, -1) => '▼',
            _ => return,
        };
        while arrow_index <= arrow_count {
            let distance = (arrow_index * total_length + intervals / 2) / intervals;
            if distance >= traced_length + length {
                break;
            }
            let position = start + direction * (distance - traced_length);
            let inside_node = viewport_graph.nodes().any(|(center, node)| {
                matches!(node.role, NodeRole::Data(_))
                    && WorldRect::from_center_and_size(*center, node.size).contains(position)
            });
            if !inside_node
                && let Some((character, style)) = buffer.get_char_styled(position)
                && matches!(character, '\u{2500}'..='\u{257f}')
            {
                buffer.set_char_styled(position, arrow, style);
            }
            arrow_index += 1;
        }
        traced_length += length;
    }
}

#[cfg(test)]
mod tests {
    use petgraph::graph::NodeIndex;
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};

    use super::{ARROW_GAPS, LineStyle, backward_edge_route, draw_arrows, draw_edge_with_style};
    use crate::{
        geometry::WorldPos,
        viewport_graph::ViewportGraph,
        viewport_state::{ViewportState, WorldBuffer},
    };

    fn routed_edge(points: &[WorldPos]) -> (ViewportGraph, NodeIndex, NodeIndex) {
        let source = NodeIndex::new(0);
        let target = NodeIndex::new(usize::from(points.first() != points.last()));
        let mut graph = ViewportGraph::empty();
        graph.node_positions.insert(source, points[0]);
        graph
            .node_positions
            .insert(target, points[points.len() - 1]);
        // Reverse insertion order so tracing cannot rely on graph storage order.
        for segment in points.windows(2).rev() {
            graph
                .graph
                .add_edge(segment[1], segment[0], vec![(source, target)]);
        }
        (graph, source, target)
    }

    fn rendered_arrows(points: &[WorldPos], state: &ViewportState) -> Vec<(WorldPos, char)> {
        let (graph, source, target) = routed_edge(points);
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, state);
        for (start, end, _) in graph.edges() {
            draw_edge_with_style(&mut world_buffer, start, end, Color::Red, LineStyle::Bold);
        }
        draw_arrows(&mut world_buffer, &graph, source, target, ARROW_GAPS);
        let mut arrows = Vec::new();
        let bounds = world_buffer.visible_world_area();
        for x in bounds.min.x..=bounds.max.x {
            for y in bounds.min.y..=bounds.max.y {
                let position = WorldPos::new(x, y);
                if let Some((character, style)) = world_buffer.get_char_styled(position)
                    && matches!(character, '▶' | '◀' | '▲' | '▼')
                {
                    assert_eq!(style.fg, Some(Color::Red));
                    arrows.push((position, character));
                }
            }
        }
        arrows
    }

    fn viewport() -> ViewportState {
        ViewportState {
            viewport_bounds: Rect::new(0, 0, 100, 80),
            ..ViewportState::default()
        }
    }

    #[test]
    fn test_backward_edge_route_follows_only_its_bundle() {
        let points = [
            WorldPos::new(20, 0),
            WorldPos::new(20, -10),
            WorldPos::new(-20, -10),
            WorldPos::new(-20, 0),
        ];
        let (mut graph, source, target) = routed_edge(&points);
        graph
            .graph
            .add_edge(points[0], points[3], vec![(target, source)]);
        graph.graph.add_edge(
            points[1],
            WorldPos::new(30, -10),
            vec![(source, NodeIndex::new(2))],
        );
        assert_eq!(
            backward_edge_route(&graph, source, target),
            Some(points.to_vec())
        );
    }

    #[test]
    fn test_arrows_are_evenly_spaced_along_all_four_directions() {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(20, 0),
            WorldPos::new(20, 20),
            WorldPos::new(-20, 20),
            WorldPos::new(-20, 0),
        ];
        // A 100-cell route has six arrows at rounded distances 100 * k / 7.
        assert_eq!(
            rendered_arrows(&points, &viewport()),
            vec![
                (WorldPos::new(-20, 14), '▼'),
                (WorldPos::new(-11, 20), '◀'),
                (WorldPos::new(3, 20), '◀'),
                (WorldPos::new(14, 0), '▶'),
                (WorldPos::new(17, 20), '◀'),
                (WorldPos::new(20, 9), '▲'),
            ]
        );
        let split_points = [
            points[0],
            WorldPos::new(7, 0),
            points[1],
            points[2],
            WorldPos::new(3, 20),
            points[3],
            points[4],
        ];
        assert_eq!(
            rendered_arrows(&split_points, &viewport()),
            rendered_arrows(&points, &viewport())
        );
    }

    #[test]
    fn test_arrows_keep_spacing_when_route_is_clipped() {
        let points = [WorldPos::new(-40, 0), WorldPos::new(40, 0)];
        let small_viewport = ViewportState {
            viewport_bounds: Rect::new(0, 0, 20, 10),
            ..ViewportState::default()
        };
        let expected: Vec<_> = rendered_arrows(&points, &viewport())
            .into_iter()
            .filter(|(position, _)| small_viewport.world_to_terminal(*position).is_some())
            .collect();
        assert_eq!(rendered_arrows(&points, &small_viewport), expected);
        assert!(!expected.is_empty());
    }

    #[test]
    fn test_backward_edge_route_traces_self_loop_from_outgoing_port() {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(20, 0),
            WorldPos::new(20, -20),
            WorldPos::new(-20, -20),
            WorldPos::new(-20, 0),
            WorldPos::new(0, 0),
        ];
        let (graph, source, target) = routed_edge(&points);
        assert_eq!(
            backward_edge_route(&graph, source, target),
            Some(points.to_vec())
        );
        assert_eq!(rendered_arrows(&points, &viewport()).len(), 7);
    }

    #[test]
    fn test_backward_edge_route_does_not_decorate_disconnected_segments() {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(20, 0),
            WorldPos::new(20, 20),
        ];
        let (mut graph, source, target) = routed_edge(&points);
        graph.graph.remove_edge(points[1], points[2]);
        assert!(backward_edge_route(&graph, source, target).is_none());
    }
}
