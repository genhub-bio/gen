// This module implements graph rendering using the ViewportGraph system.

use std::sync::Arc;

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

/// Number of world-x units between direction markers on a backward-edge bypass's main span.
const ARROW_GAPS: i64 = 16;

/// A main-span segment shorter than this (in world-x cells) gets no arrows at all, rather
/// than one landing right against a pin - short segments are common right next to a pin (see
/// `draw_arrows`), where there isn't enough run for a marker to read as "on the line" instead
/// of "touching the box".
const MIN_ARROW_SEGMENT_LENGTH: i64 = 3;

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

    // Mark the horizontal bypass of any rewired backward edge with direction arrows.
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

/// Place direction markers on the `left_pin -> right_pin` main-span segment(s) of a backward
/// edge rewired onto pin nodes by `crawl::build_window_graph`. The loop's other two legs
/// (`source -> right_pin`, `left_pin -> target`) are short excursions connecting a pin to its
/// real endpoint, not the backward direction itself, so they are never marked.
///
/// The main span always runs toward decreasing x (`◀`): a pin always floats to the extreme
/// rank of its own window, so `right_pin` sits to the right of `left_pin` by construction,
/// regardless of which part of the loop the current viewport happens to show.
fn draw_arrows(
    buffer: &mut WorldBuffer,
    viewport_graph: &ViewportGraph,
    source: NodeIndex,
    target: NodeIndex,
    gaps: i64,
) {
    // The main span may be split across several collinear segments that render as one
    // continuous line: a pin with 2+ neighbors survives `prune_pin_stubs` re-roled to
    // `Routing` (see `window_graph::WindowNode::Pin`) but, running after `simplify_graph`,
    // is never merged back into its neighbors, leaving it as a permanent break point.
    // Collect the segments first and compute a single margin from their combined span, so
    // the marker grid stays aligned across segment boundaries instead of each segment
    // centering itself.
    let mut segments: Vec<(WorldPos, WorldPos)> = Vec::new();
    let mut span: Option<(i64, i64)> = None;
    for (seg_a, seg_b, bundle) in viewport_graph.edges() {
        if seg_a.y != seg_b.y || !bundle.contains(&(source, target)) {
            continue;
        }

        let (lo, hi) = if seg_a.x <= seg_b.x {
            (seg_a, seg_b)
        } else {
            (seg_b, seg_a)
        };
        // Only the main span gets arrows - the loop's two excursions are skipped entirely.
        if !viewport_graph.backward_span_edges.contains(&(lo, hi)) {
            continue;
        }
        segments.push((lo, hi));
        span = Some(match span {
            Some((min_x, max_x)) => (min_x.min(lo.x), max_x.max(hi.x)),
            None => (lo.x, hi.x),
        });
    }

    let Some((min_x, max_x)) = span else {
        return;
    };

    // Center the markers across the full span: split the leftover space (total
    // length mod gaps) evenly between both ends, so the first/last arrow sits the
    // same distance from its endpoint as every other arrow sits from its neighbor.
    let margin = (max_x - min_x).rem_euclid(gaps) / 2;

    for (lo, hi) in segments {
        if hi.x - lo.x < MIN_ARROW_SEGMENT_LENGTH {
            continue;
        }
        let arrow = '◀';
        for x in lo.x..=hi.x {
            let pos = WorldPos::new(x, lo.y);
            if (x - min_x - margin).rem_euclid(gaps) == 0
                && matches!(
                    buffer.get_char(pos),
                    Some('─') | Some('━') | Some('┄') | Some('╌')
                )
                && let Some((_, style)) = buffer.get_char_styled(pos)
            {
                buffer.set_char_styled(pos, arrow, style);
            }
        }
    }
}
