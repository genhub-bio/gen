// This module implements graph rendering using the ViewportGraph system.

use std::{
    collections::{HashMap, HashSet, VecDeque, hash_map::Entry},
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
    diagonal::{
        ArrowMark, DeferredEdge, DiagonalDrawing, EdgeStyle, NodeGlow, SymbolGlow, bend_arch,
        draw_braille_bezier, draw_braille_line, hairpin_arc_arrow, hairpin_edge_arcs, is_braille,
        routed_edge_curve, segment_endpoints,
    },
    edge_ports::NodeSide,
    geometry::{WorldPos, WorldRect},
    graph_widget::NODE_GLYPH,
    layout::{JunctionSymbol, NodeRole, VerticalAnchor},
    pinning::PinnedEdge,
    theme::Theme,
    viewport_graph::ViewportGraph,
    viewport_state::WorldBuffer,
};

/// Rectilinear routes retain their established marker rhythm.
const ARROW_GAPS: i64 = 10;
/// Diagonal routes mark every edge, so repeat markers less often along long runs.
const DIAGONAL_ARROW_GAPS: i64 = 20;
/// A diagonal run with fewer free cells than this is too short to carry a readable arrow when it
/// starts or ends at a data node, where forward and reverse routes share the same few cells.
const MIN_ARROW_RUN_CELLS: i64 = 4;
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

/// Sequence columns available to the cursor, excluding decorative padding and omitted runs.
#[derive(Clone, Debug)]
pub enum CursorColumns {
    /// Every column in this inclusive range is a sequence character.
    Range(i64, i64),
    /// Only these ordered columns contain retained sequence characters.
    Positions(Vec<i64>),
}

impl CursorColumns {
    /// First and last accessible column, inclusive.
    pub fn bounds(&self) -> (i64, i64) {
        match self {
            Self::Range(first, last) => (*first, *last),
            Self::Positions(columns) => (
                *columns.first().unwrap_or(&0),
                *columns.last().unwrap_or(&0),
            ),
        }
    }

    /// Closest accessible column, preferring the left column on ties.
    pub fn nearest(&self, column: i64) -> i64 {
        match self {
            Self::Range(first, last) => column.clamp(*first, *last),
            Self::Positions(columns) => columns
                .iter()
                .copied()
                .min_by_key(|candidate| candidate.abs_diff(column))
                .unwrap_or(0),
        }
    }

    /// First accessible column at or beyond a requested column in the travel direction.
    pub fn next(&self, column: i64, forward: bool) -> Option<i64> {
        match self {
            Self::Range(first, last) => (*first..=*last).contains(&column).then_some(column),
            Self::Positions(columns) => {
                if forward {
                    columns
                        .iter()
                        .copied()
                        .find(|candidate| *candidate >= column)
                } else {
                    columns
                        .iter()
                        .rev()
                        .copied()
                        .find(|candidate| *candidate <= column)
                }
            }
        }
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

    /// Row used for layout alignment and edge attachment. Top-anchored renderers draw their
    /// sequence on the top row and grow decoration downward. Changes invalidate cached sizes.
    fn vertical_anchor(&self, _node: &G::NodeId) -> VerticalAnchor {
        VerticalAnchor::Center
    }

    /// Map a raw node-local coordinate to a displayed column.
    fn map_column(&self, _node: &G::NodeId, raw: i64) -> i64 {
        raw
    }

    /// Map a displayed column back to a raw node-local coordinate.
    fn raw_column(&self, _node: &G::NodeId, column: i64) -> i64 {
        column
    }

    /// Get default dimensions for dummy/routing nodes
    fn get_dummy_size(&self) -> (u64, u64) {
        (1, 1)
    }

    /// Render a node in its allocated world area.
    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId);

    /// Whether `node` is drawn. An invisible node is still placed in the layout, so its edges
    /// meet at it, but it is drawn as a junction of those edges rather than as a node.
    fn is_visible(&self, _node: &G::NodeId) -> bool {
        true
    }

    /// The one row of `node` the cursor may sit on, counted up from the bottom of its rect, or
    /// `None` to let the cursor use every row. A renderer that draws decoration around a node's
    /// content (e.g. annotation lanes above and below a sequence row) names the content row, so
    /// cursor moves, clicks, and jumps always land on it.
    fn cursor_row(&self, _node: &G::NodeId) -> Option<u64> {
        None
    }

    /// Columns containing sequence characters; `None` permits every column of the node.
    fn cursor_columns(&self, _node: &G::NodeId) -> Option<CursorColumns> {
        None
    }

    /// The node-local sequence rectangle that may receive highlights and dimming.
    /// Decoration outside this rectangle leaves previously painted edges untouched.
    fn content_rect(&self, node: &G::NodeId) -> WorldRect {
        let (width, height) = self.get_node_size(node);
        let (left, right) = self
            .cursor_columns(node)
            .map_or((0, width.saturating_sub(1) as i64), |columns| {
                columns.bounds()
            });
        let (bottom, top) = self
            .cursor_row(node)
            .map_or((0, height.saturating_sub(1) as i64), |row| {
                (row as i64, row as i64)
            });
        WorldRect::from_coords(left, bottom, right, top)
    }

    /// Whether the attachment row of `node` should sit at y = 0. Layout aligns nodes across layers while
    /// preserving within-layer spacing and straight routing chains where feasible. A renderer
    /// marks e.g. the reference sequence so the main path reads as a straight line.
    fn is_aligned(&self, _node: &G::NodeId) -> bool {
        false
    }

    /// A counter that changes whenever `get_node_size`, `get_dummy_size`, or `vertical_anchor` may answer
    /// differently than before. Views reuse a window's routed geometry until it changes, so a
    /// renderer whose sizes depend on mutable state (e.g. annotation lanes under each node) must
    /// bump it on every real change; a renderer with fixed sizes keeps the default.
    fn size_generation(&self) -> u64 {
        0
    }
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

    fn vertical_anchor(&self, node: &G::NodeId) -> VerticalAnchor {
        (**self).vertical_anchor(node)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (**self).get_dummy_size()
    }

    fn map_column(&self, node: &G::NodeId, raw: i64) -> i64 {
        (**self).map_column(node, raw)
    }

    fn raw_column(&self, node: &G::NodeId, column: i64) -> i64 {
        (**self).raw_column(node, column)
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId) {
        (**self).render_node(buffer, area, node_id);
    }

    fn is_visible(&self, node: &G::NodeId) -> bool {
        (**self).is_visible(node)
    }

    fn cursor_row(&self, node: &G::NodeId) -> Option<u64> {
        (**self).cursor_row(node)
    }

    fn cursor_columns(&self, node: &G::NodeId) -> Option<CursorColumns> {
        (**self).cursor_columns(node)
    }

    fn content_rect(&self, node: &G::NodeId) -> WorldRect {
        (**self).content_rect(node)
    }

    fn is_aligned(&self, node: &G::NodeId) -> bool {
        (**self).is_aligned(node)
    }

    fn size_generation(&self) -> u64 {
        (**self).size_generation()
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

    fn vertical_anchor(&self, node: &G::NodeId) -> VerticalAnchor {
        (**self).vertical_anchor(node)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (**self).get_dummy_size()
    }

    fn map_column(&self, node: &G::NodeId, raw: i64) -> i64 {
        (**self).map_column(node, raw)
    }

    fn raw_column(&self, node: &G::NodeId, column: i64) -> i64 {
        (**self).raw_column(node, column)
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &G::NodeId) {
        (**self).render_node(buffer, area, node_id);
    }

    fn is_visible(&self, node: &G::NodeId) -> bool {
        (**self).is_visible(node)
    }

    fn cursor_row(&self, node: &G::NodeId) -> Option<u64> {
        (**self).cursor_row(node)
    }

    fn cursor_columns(&self, node: &G::NodeId) -> Option<CursorColumns> {
        (**self).cursor_columns(node)
    }

    fn content_rect(&self, node: &G::NodeId) -> WorldRect {
        (**self).content_rect(node)
    }

    fn is_aligned(&self, node: &G::NodeId) -> bool {
        (**self).is_aligned(node)
    }

    fn size_generation(&self) -> u64 {
        (**self).size_generation()
    }
}

pub(crate) struct PlotDecorations<'a> {
    pub node_highlights: &'a [(WorldPos, PathStyle)],
    pub edge_highlights: &'a [((WorldPos, WorldPos), PathStyle)],
    pub hairpin_highlights: &'a [((NodeIndex, NodeSide), PathStyle)],
    pub cell_highlights: &'a [CellHighlight],
    pub lowlights: &'a [(WorldPos, WorldPos)],
    pub node_lowlights: &'a [WorldPos],
}

/// The glow around the highlighted cells of `rect`, a part of the node content `container`, in
/// buffer cell units.
fn cell_glow(
    buffer: &WorldBuffer<'_>,
    rect: WorldRect,
    container: WorldRect,
    color: Color,
) -> DeferredEdge {
    DeferredEdge::NodeGlow(NodeGlow {
        top_left: buffer.world_to_terminal_continuous((rect.min.x as f64, rect.max.y as f64 + 1.0)),
        bottom_right: buffer
            .world_to_terminal_continuous((rect.max.x as f64 + 1.0, rect.min.y as f64)),
        clip_left: rect.min.x > container.min.x,
        clip_right: rect.max.x < container.max.x,
        color,
    })
}

/// The glow for highlighted cells of `rect`. A lone node glyph (minimal detail) gets a halo
/// that follows the round symbol; everything else gets the rectangular glow of [`cell_glow`].
fn highlight_glow(
    buffer: &WorldBuffer<'_>,
    rect: WorldRect,
    container: WorldRect,
    color: Color,
) -> DeferredEdge {
    let is_symbol = rect.min == rect.max
        && buffer
            .get_char_styled(rect.min)
            .is_some_and(|(character, _)| character == NODE_GLYPH);
    if !is_symbol {
        return cell_glow(buffer, rect, container, color);
    }
    DeferredEdge::SymbolGlow(SymbolGlow {
        center: buffer
            .world_to_terminal_continuous((rect.min.x as f64 + 0.5, rect.min.y as f64 + 0.5)),
        color,
    })
}

/// Plot a viewport graph with resolved highlights and lowlights.
///
/// Returns the lines of a [`DiagonalDrawing::Deferred`] edge style, which the buffer does not
/// contain; every other style draws all of its edges into the buffer and returns none.
pub(crate) fn plot_viewport_graph_with_highlights<V, G>(
    viewport_graph: &ViewportGraph,
    buffer: &mut WorldBuffer<'_>,
    renderer: &V,
    original_graph: &G,
    decorations: PlotDecorations<'_>,
    theme: &Theme,
    edge_style: EdgeStyle,
) -> Vec<DeferredEdge>
where
    V: NodeRenderer<G>,
    G: GraphBase + NodeIndexable,
{
    let PlotDecorations {
        node_highlights,
        edge_highlights,
        hairpin_highlights,
        cell_highlights,
        lowlights,
        node_lowlights,
    } = decorations;

    // Annotation labels reserve layout space beyond the visible sequence. Attach edges to
    // sequence content, not those invisible margins, and use the same endpoints for arrows.
    let content_rects: HashMap<_, _> = viewport_graph
        .nodes()
        .filter_map(|(position, node)| {
            if !edge_style.is_diagonal() {
                return None;
            }
            let NodeRole::Data(domain_index) = node.role else {
                return None;
            };
            let node_id = <G as NodeIndexable>::from_index(original_graph, domain_index.index());
            let content = renderer.content_rect(&node_id);
            let layout = node.vertical_anchor.rect(*position, node.size);
            Some((
                *position,
                WorldRect::from_coords(
                    layout.min.x + content.min.x,
                    layout.min.y + content.min.y,
                    layout.min.x + content.max.x,
                    layout.min.y + content.max.y,
                ),
            ))
        })
        .collect();

    let glows_deferred = matches!(
        edge_style,
        EdgeStyle::Diagonal(DiagonalDrawing::Deferred)
            | EdgeStyle::Bezier(DiagonalDrawing::Deferred)
    );
    let ordinary_color = theme[0x05];
    let lowlight_color = theme[0x04];
    let mut deferred_lines = Vec::new();
    let mut stroke = |buffer: &mut WorldBuffer<'_>,
                      source: WorldPos,
                      target: WorldPos,
                      color: Color,
                      line_style: LineStyle| {
        let drawing = match edge_style {
            EdgeStyle::Diagonal(drawing) | EdgeStyle::Bezier(drawing) => drawing,
            EdgeStyle::Rectilinear => {
                draw_edge_with_style(buffer, source, target, color, line_style);
                return;
            }
        };
        let (Some(source_node), Some(target_node)) = (
            viewport_graph.node_data_by_pos.get(&source),
            viewport_graph.node_data_by_pos.get(&target),
        ) else {
            return;
        };
        if let EdgeStyle::Bezier(_) = edge_style {
            // An arch through an up or down pin is drawn once, with the pin's west half.
            let bend_pin = [source, target].into_iter().find(|&position| {
                bend_arch(viewport_graph, position, &content_rects, color, line_style).is_some()
            });
            if let Some(pin) = bend_pin
                && [source, target].iter().any(|position| position.x > pin.x)
            {
                return;
            }
            let routed = routed_edge_curve(
                viewport_graph,
                (source, source_node),
                (target, target_node),
                &content_rects,
                color,
                line_style,
            );
            let curve = routed
                .curve
                .mapped(|point| buffer.world_to_terminal_continuous(point));
            match drawing {
                DiagonalDrawing::Braille => {
                    if let Some(stub) = routed.stub {
                        draw_braille_line(
                            buffer,
                            buffer.world_to_terminal_continuous(stub.from),
                            buffer.world_to_terminal_continuous(stub.to),
                            color,
                            line_style,
                        );
                    }
                    draw_braille_bezier(buffer, &curve);
                }
                DiagonalDrawing::Deferred => {
                    if let Some(stub) = routed.stub {
                        deferred_lines.push(DeferredEdge::Line(crate::diagonal::DiagonalLine {
                            from: buffer.world_to_terminal_continuous(stub.from),
                            to: buffer.world_to_terminal_continuous(stub.to),
                            ..stub
                        }));
                    }
                    deferred_lines.push(DeferredEdge::Bezier(curve));
                }
            }
        } else {
            let (from, to) =
                segment_endpoints((source, source_node), (target, target_node), &content_rects);
            let from = buffer.world_to_terminal_continuous(from);
            let to = buffer.world_to_terminal_continuous(to);
            match drawing {
                DiagonalDrawing::Braille => draw_braille_line(buffer, from, to, color, line_style),
                DiagonalDrawing::Deferred => {
                    deferred_lines.push(DeferredEdge::Line(crate::diagonal::DiagonalLine {
                        from,
                        to,
                        color,
                        line_style,
                    }))
                }
            }
        }
    };

    // Draw edges first so nodes appear on top
    let mut edges: Vec<_> = viewport_graph.edges().collect();
    if edge_style.is_diagonal() {
        // Straight lines cross and share braille cells, where the last one drawn wins the
        // color: draw dimmed edges first and highlighted ones last so a highlight is never
        // painted over.
        let is_pair = |(s, t): &(WorldPos, WorldPos), source: WorldPos, target: WorldPos| {
            (*s == source && *t == target) || (*s == target && *t == source)
        };
        edges.sort_by_key(|&(source, target, _)| {
            if edge_highlights
                .iter()
                .any(|(pair, _)| is_pair(pair, source, target))
            {
                2
            } else if lowlights.iter().any(|pair| is_pair(pair, source, target)) {
                0
            } else {
                1
            }
        });
    }
    // How an edge between two positions is colored: highlights win over lowlights, and a plain
    // edge that stands for no original edge is dashed. Later highlights take precedence.
    let edge_look = |source: WorldPos, target: WorldPos, has_data: bool| {
        let is_pair = |(s, t): &(WorldPos, WorldPos)| {
            (*s == source && *t == target) || (*s == target && *t == source)
        };
        let highlighted_style = edge_highlights
            .iter()
            .filter(|(pair, _)| is_pair(pair))
            .map(|(_, style)| *style)
            .next_back();
        let is_lowlight = lowlights.iter().any(is_pair);
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
            (edge_color, line_style)
        } else if is_lowlight {
            (lowlight_color, LineStyle::Dashed)
        } else if has_data {
            (ordinary_color, LineStyle::Normal)
        } else {
            (ordinary_color, LineStyle::Dashed)
        }
    };
    for (source, target, bundle) in edges {
        if matches!(edge_style, EdgeStyle::Bezier(_)) && source == target {
            continue;
        }
        let (color, line_style) = edge_look(source, target, !bundle.is_empty());
        stroke(buffer, source, target, color, line_style);
    }

    if let EdgeStyle::Bezier(drawing) = edge_style {
        let mut hairpins: Vec<_> = viewport_graph
            .pinned_edges
            .iter()
            .filter(|edge| edge.source == edge.target && edge.ports.source == edge.ports.target)
            .collect();
        hairpins.sort_by_key(|edge| {
            hairpin_highlights
                .iter()
                .any(|((node, side), _)| *node == edge.source && *side == edge.ports.source)
        });
        let mut drawn = HashSet::new();
        for edge in hairpins {
            if !drawn.insert((edge.source, edge.ports.source)) {
                continue;
            }
            let Some(&position) = viewport_graph.node_positions.get(&edge.source) else {
                continue;
            };
            if !viewport_graph.graph.contains_edge(position, position) {
                continue;
            }
            let Some(node) = viewport_graph.node_data_by_pos.get(&position) else {
                continue;
            };
            let highlighted_style = hairpin_highlights
                .iter()
                .filter(|((domain_index, side), _)| {
                    *domain_index == edge.source && *side == edge.ports.source
                })
                .map(|(_, style)| *style)
                .next_back();
            let is_lowlight = lowlights
                .iter()
                .any(|&(from, to)| from == position && to == position);
            let (color, line_style) = if let Some(style) = highlighted_style {
                let color = if style.color == Color::Reset {
                    theme[0x07]
                } else {
                    style.color
                };
                let line_style = if is_lowlight {
                    LineStyle::Dashed
                } else {
                    style.line_style
                };
                (color, line_style)
            } else if is_lowlight {
                (lowlight_color, LineStyle::Dashed)
            } else {
                (ordinary_color, LineStyle::Normal)
            };
            let arcs = hairpin_edge_arcs(
                (position, node),
                edge.ports.source,
                &content_rects,
                color,
                line_style,
            )
            .map(|arc| arc.mapped(|point| buffer.world_to_terminal_continuous(point)));
            match drawing {
                DiagonalDrawing::Braille => {
                    arcs.iter().for_each(|arc| draw_braille_bezier(buffer, arc));
                }
                DiagonalDrawing::Deferred => {
                    deferred_lines.push(DeferredEdge::Arrow(hairpin_arc_arrow(
                        &arcs,
                        ordinary_color,
                    )));
                    deferred_lines.extend(arcs.into_iter().map(DeferredEdge::Bezier));
                }
            }
        }
    }

    // Draw nodes
    for (world_pos, node) in viewport_graph.nodes() {
        match &node.role {
            NodeRole::Data(domain_idx) => {
                let node_id = <G as NodeIndexable>::from_index(original_graph, domain_idx.index());
                let world_rect = node.vertical_anchor.rect(*world_pos, node.size);
                renderer.render_node(buffer, world_rect, &node_id);
                let content = renderer.content_rect(&node_id);
                let content_rect = WorldRect::from_coords(
                    world_rect.min.x + content.min.x,
                    world_rect.min.y + content.min.y,
                    world_rect.min.x + content.max.x,
                    world_rect.min.y + content.max.y,
                );

                // If lowlighted, dim the node background to theme[0x04].
                // Applied before highlights so highlights take priority.
                if node_lowlights.contains(world_pos) {
                    let dim = lowlight_color;
                    for y in content_rect.min.y..=content_rect.max.y {
                        for x in content_rect.min.x..=content_rect.max.x {
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
                    for y in content_rect.min.y..=content_rect.max.y {
                        for x in content_rect.min.x..=content_rect.max.x {
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
                    if glows_deferred {
                        deferred_lines.push(highlight_glow(buffer, content_rect, content_rect, hl));
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
                    let x0 = (world_rect.min.x + tl.0).max(content_rect.min.x);
                    let x1 = (world_rect.min.x + br.0).min(content_rect.max.x);
                    let y0 = (world_rect.min.y + tl.1).max(content_rect.min.y);
                    let y1 = (world_rect.min.y + br.1).min(content_rect.max.y);
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
                    if glows_deferred && x0 <= x1 && y0 <= y1 {
                        deferred_lines.push(highlight_glow(
                            buffer,
                            WorldRect::from_coords(x0, y0, x1, y1),
                            content_rect,
                            hl,
                        ));
                    }
                }
            }
            NodeRole::Routing | NodeRole::Pin(_) | NodeRole::HairPin(_) | NodeRole::Wormhole(_) => {
                // Straight lines meet at routing points without a junction glyph.
                if edge_style.is_diagonal() && !matches!(node.role, NodeRole::Wormhole(_)) {
                    continue;
                }
                let edge_color = ordinary_color;
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
                    let color = if all_dim { lowlight_color } else { edge_color };
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

    // Diagonal edges all carry direction markers; rectilinear edges retain markers only when
    // their routing would otherwise make the direction ambiguous.
    let two_way_cells = cells_in_both_directions(viewport_graph);
    for pinned_edge in &viewport_graph.pinned_edges {
        // A Bezier hairpin carries its arrow on the bend itself.
        if matches!(edge_style, EdgeStyle::Bezier(_))
            && pinned_edge.source == pinned_edge.target
            && pinned_edge.ports.source == pinned_edge.ports.target
        {
            continue;
        }
        if !edge_style.is_diagonal() && !route_runs_against_flow(viewport_graph, pinned_edge) {
            continue;
        }
        draw_arrows(
            buffer,
            viewport_graph,
            pinned_edge,
            if edge_style.is_diagonal() {
                DIAGONAL_ARROW_GAPS
            } else {
                ARROW_GAPS
            },
            (edge_style, &content_rects, ordinary_color),
            &two_way_cells,
            &mut deferred_lines,
        );
    }
    deferred_lines
}

/// The cells that routes run through in both horizontal directions, such as the bus a hairpin
/// shares with the edges beside it. An arrow there would point one way along a wire that carries
/// both, so none is drawn on them.
fn cells_in_both_directions(viewport_graph: &ViewportGraph) -> HashSet<WorldPos> {
    let mut rightward = HashSet::new();
    let mut leftward = HashSet::new();
    for pinned_edge in &viewport_graph.pinned_edges {
        let Some(route) = pinned_edge_route(viewport_graph, pinned_edge) else {
            continue;
        };
        for segment in route.windows(2) {
            let (start, end) = (segment[0], segment[1]);
            if start.y != end.y {
                continue;
            }
            let cells = if end.x > start.x {
                &mut rightward
            } else {
                &mut leftward
            };
            cells.extend(
                (start.x.min(end.x)..=start.x.max(end.x)).map(|x| WorldPos::new(x, start.y)),
            );
        }
    }
    rightward.intersection(&leftward).copied().collect()
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

/// Whether a domain edge's route runs against the layout's left-to-right flow, through a pin or
/// along a segment heading west, so a rectilinear drawing needs arrows to show its direction.
fn route_runs_against_flow(viewport_graph: &ViewportGraph, pinned_edge: &PinnedEdge) -> bool {
    pinned_edge_route(viewport_graph, pinned_edge).is_some_and(|route| {
        let passes_pin = route.iter().any(|position| {
            viewport_graph
                .node_data_by_pos
                .get(position)
                .is_some_and(|node| matches!(node.role, NodeRole::Pin(_)))
        });
        passes_pin || route.windows(2).any(|segment| segment[1].x < segment[0].x)
    })
}

/// Trace a domain edge's routed segments in source-to-target order, leaving its source
/// on the side its ports name and arriving at its target on the other named side. Bundles
/// distinguish the route from other edges sharing its junctions, and the sides from parallel
/// edges between the same two nodes.
fn pinned_edge_route(
    viewport_graph: &ViewportGraph,
    pinned_edge: &PinnedEdge,
) -> Option<Vec<WorldPos>> {
    trace_edge_route(viewport_graph, pinned_edge, |node| {
        viewport_graph.node_positions.get(&node).copied()
    })
}

/// [`pinned_edge_route`] for highlighting, which also follows edges into and out of invisible
/// junction nodes, drawn as routing points.
pub(crate) fn highlighted_edge_route(
    viewport_graph: &ViewportGraph,
    pinned_edge: &PinnedEdge,
) -> Option<Vec<WorldPos>> {
    trace_edge_route(viewport_graph, pinned_edge, |node| {
        viewport_graph
            .node_positions
            .get(&node)
            .or_else(|| viewport_graph.junction_positions.get(&node))
            .copied()
    })
}

fn trace_edge_route(
    viewport_graph: &ViewportGraph,
    pinned_edge: &PinnedEdge,
    position_of: impl Fn(NodeIndex) -> Option<WorldPos>,
) -> Option<Vec<WorldPos>> {
    let PinnedEdge {
        source,
        target,
        ports,
        ..
    } = *pinned_edge;
    let source_pos = position_of(source)?;
    let target_pos = position_of(target)?;
    let follows_edge = |start, end| {
        viewport_graph
            .graph
            .edge_weight(start, end)
            .is_some_and(|bundle| bundle.contains(&(source, target)))
    };
    // Legs reach a node from its west side through earlier layers and leave it on its east
    // side through later ones.
    let is_on_side = |neighbor: WorldPos, node: WorldPos, side: NodeSide| match side {
        NodeSide::West => neighbor.x < node.x,
        NodeSide::East => neighbor.x > node.x,
    };

    if source == target && ports.source == ports.target {
        // A hairpin leaves and returns by one side. Walking it without turning straight back along
        // the edge just used takes it out along its bus, around the turnaround and back, instead
        // of out and in by the same leg. Higher branches come first, so it leaves on the top row.
        let by_height = |position: WorldPos| {
            let mut neighbors: Vec<WorldPos> = viewport_graph.neighbors(position).collect();
            neighbors.sort_by_key(|neighbor| std::cmp::Reverse(neighbor.y));
            neighbors
        };
        let mut parent: HashMap<(WorldPos, WorldPos), (WorldPos, WorldPos)> = HashMap::new();
        let mut pending: VecDeque<(WorldPos, WorldPos)> = VecDeque::new();
        for neighbor in by_height(source_pos) {
            if follows_edge(source_pos, neighbor) && is_on_side(neighbor, source_pos, ports.source)
            {
                parent.insert((neighbor, source_pos), (source_pos, source_pos));
                pending.push_back((neighbor, source_pos));
            }
        }
        while let Some((position, came_from)) = pending.pop_front() {
            for neighbor in by_height(position) {
                if neighbor == came_from || !follows_edge(position, neighbor) {
                    continue;
                }
                if neighbor == source_pos {
                    if !is_on_side(position, source_pos, ports.target) {
                        continue;
                    }
                    let mut route = vec![source_pos, position];
                    let mut state = (position, came_from);
                    while parent[&state].0 != source_pos || parent[&state].1 != source_pos {
                        state = parent[&state];
                        route.insert(1, state.0);
                    }
                    route.push(source_pos);
                    return Some(route);
                }
                let next = (neighbor, position);
                if let Entry::Vacant(entry) = parent.entry(next) {
                    entry.insert((position, came_from));
                    pending.push_back(next);
                }
            }
        }
        return None;
    }

    let mut previous = HashMap::from([(source_pos, source_pos)]);
    let mut pending = VecDeque::from([source_pos]);
    while let Some(position) = pending.pop_front() {
        for neighbor in viewport_graph.neighbors(position) {
            if !follows_edge(position, neighbor)
                || (position == source_pos && !is_on_side(neighbor, source_pos, ports.source))
            {
                continue;
            }
            if neighbor == target_pos {
                // A loop closes through its incoming leg, not by retracing its first segment.
                if (position == source_pos && source == target)
                    || !is_on_side(position, target_pos, ports.target)
                {
                    continue;
                }
                let mut route = vec![target_pos, position];
                let mut current = position;
                while current != source_pos {
                    current = previous[&current];
                    route.push(current);
                }
                route.reverse();
                return Some(route);
            }
            if let Entry::Vacant(entry) = previous.entry(neighbor) {
                entry.insert(position);
                pending.push_back(neighbor);
            }
        }
    }
    None
}

/// The runs of a route that go one way in a straight line, so a run split by routing nodes
/// counts once.
fn straight_runs(route: &[WorldPos]) -> Vec<(WorldPos, WorldPos)> {
    let mut runs: Vec<(WorldPos, WorldPos)> = Vec::new();
    for segment in route.windows(2) {
        let (start, end) = (segment[0], segment[1]);
        let direction = WorldPos::new((end.x - start.x).signum(), (end.y - start.y).signum());
        if let Some((run_start, run_end)) = runs.last_mut() {
            let run_direction = WorldPos::new(
                (run_end.x - run_start.x).signum(),
                (run_end.y - run_start.y).signum(),
            );
            if *run_end == start && run_direction == direction {
                *run_end = end;
                continue;
            }
        }
        runs.push((start, end));
    }
    runs
}

/// The horizontal runs of a domain edge's route as drawn, each as its first and last cell in
/// the direction of travel. A rectilinear route yields its straight runs along one row. A curved
/// route combines adjacent flat stretches that meet end to end, but excludes
/// turnarounds and S-curves that change row.
fn horizontal_runs(
    viewport_graph: &ViewportGraph,
    route: &[WorldPos],
    edge_style: EdgeStyle,
    content_rects: &HashMap<WorldPos, WorldRect>,
) -> Vec<(WorldPos, WorldPos)> {
    if !edge_style.is_diagonal() {
        return straight_runs(route)
            .into_iter()
            .filter(|(start, end)| start.y == end.y)
            .collect();
    }
    let mut runs: Vec<(WorldPos, WorldPos)> = Vec::new();
    let mut can_merge = false;
    for pair in route.windows(2) {
        let flat = (|| {
            let first = (pair[0], viewport_graph.node_data_by_pos.get(&pair[0])?);
            let second = (pair[1], viewport_graph.node_data_by_pos.get(&pair[1])?);
            let routed = routed_edge_curve(
                viewport_graph,
                first,
                second,
                content_rects,
                Color::Reset,
                LineStyle::Normal,
            );
            let curve = routed.curve;
            let is_flat = [curve.control1, curve.control2, curve.to]
                .iter()
                .all(|point| (point.1 - curve.from.1).abs() < 1e-9);
            (!routed.is_turnaround && is_flat).then(|| {
                let row = curve.from.1.floor() as i64;
                (
                    WorldPos::new(curve.from.0.floor() as i64, row),
                    WorldPos::new(curve.to.0.floor() as i64, row),
                )
            })
        })();
        let Some((start, end)) = flat else {
            can_merge = false;
            continue;
        };
        if start.x == end.x {
            continue;
        }
        if let Some((run_start, run_end)) = runs.last_mut()
            && can_merge
            && *run_end == start
            && (end.x - start.x).signum() == (run_end.x - run_start.x).signum()
        {
            *run_end = end;
        } else {
            runs.push((start, end));
        }
        can_merge = true;
    }
    runs
}

/// Mark the direction of a domain edge on each horizontal run of its route (see
/// [`horizontal_runs`]). A run carries at least one arrow, unless it is shorter than
/// [`MIN_ARROW_RUN_CELLS`] free cells and touches a data node, with additional arrows exactly
/// `gaps` cells apart. The unused distance is split between the two ends. Vertical and slanted
/// stretches are left bare. A rectilinear route marks only its westward runs, with `◀`; a
/// diagonal one marks every run. An arrow replaces only the wire of a
/// rectilinear route, or the blank and braille cells of a curved one.
fn draw_arrows(
    buffer: &mut WorldBuffer,
    viewport_graph: &ViewportGraph,
    pinned_edge: &PinnedEdge,
    gaps: i64,
    (edge_style, content_rects, color): (EdgeStyle, &HashMap<WorldPos, WorldRect>, Color),
    two_way_cells: &HashSet<WorldPos>,
    deferred: &mut Vec<DeferredEdge>,
) {
    let Some(route) = pinned_edge_route(viewport_graph, pinned_edge) else {
        return;
    };
    let accepts = if edge_style.is_diagonal() {
        is_blank_or_braille
    } else {
        is_horizontal_wire
    };
    let is_inside_node = |position: WorldPos| {
        viewport_graph.nodes().any(|(center, node)| {
            matches!(node.role, NodeRole::Data(_))
                && node
                    .vertical_anchor
                    .rect(*center, node.size)
                    .contains(position)
        })
    };
    if edge_style.is_diagonal() {
        draw_bend_arrows(
            buffer,
            viewport_graph,
            &route,
            (edge_style, content_rects, color),
            deferred,
        );
    }
    for (start, end) in horizontal_runs(viewport_graph, &route, edge_style, content_rects) {
        let direction = WorldPos::new((end.x - start.x).signum(), 0);
        // Rightward flow is the default reading direction of a rectilinear drawing, so only
        // runs against it need an arrow. Diagonal drawings mark every run.
        if !edge_style.is_diagonal() && direction.x > 0 {
            continue;
        }
        let length = (end.x - start.x).abs();
        let free_cells: Vec<WorldPos> = (0..=length)
            .map(|distance| start + direction * distance)
            .filter(|&position| !is_inside_node(position) && !two_way_cells.contains(&position))
            .collect();
        let free_count = free_cells.len() as i64;
        if free_count == 0 {
            continue;
        }
        // Short runs between routing nodes and pins keep their arrow, since each one marks a
        // layer crossing of a long edge. A short run at a data node does not: routes in both
        // directions leave and enter it over the same cells.
        let touches_data_node = [start - direction, start, end, end + direction]
            .into_iter()
            .any(is_inside_node);
        if edge_style.is_diagonal() && free_count < MIN_ARROW_RUN_CELLS && touches_data_node {
            continue;
        }
        let arrow = if direction.x > 0 { '▶' } else { '◀' };
        for slot in arrow_slots(free_count, gaps, direction.x > 0) {
            // A slot outside the view stays skipped. One that lands on a corner or junction moves to
            // the nearest cell that can take the arrow.
            let Some((slot_character, _)) = buffer.get_char_styled(free_cells[slot as usize])
            else {
                continue;
            };
            let index = if accepts(slot_character) {
                slot
            } else {
                let Some(nearest) = (0..free_count)
                    .filter(|&index| {
                        buffer
                            .get_char_styled(free_cells[index as usize])
                            .is_some_and(|(character, _)| accepts(character))
                    })
                    .min_by_key(|&index| (index - slot).abs())
                else {
                    continue;
                };
                nearest
            };
            let position = free_cells[index as usize];
            if let Some((_, style)) = buffer.get_char_styled(position) {
                if edge_style.is_deferred() {
                    deferred.push(DeferredEdge::Arrow(ArrowMark {
                        center: buffer.world_to_terminal_continuous((
                            position.x as f64 + 0.5,
                            position.y as f64 + 0.5,
                        )),
                        direction: (if direction.x > 0 { 1.0 } else { -1.0 }, 0.0),
                        color,
                    }));
                } else {
                    buffer.set_char_styled(position, arrow, style);
                }
            }
        }
    }
}

/// Indices into a run's free cells, ordered along its direction of travel, that carry an
/// arrow. The leftover distance is split by integer division, so the odd cell must always fall
/// on the same physical side: otherwise a rightward edge and the leftward edge over the same
/// cells round opposite ways and their arrows sit one cell apart. The layout is computed
/// left to right and mirrored for leftward runs.
fn arrow_slots(free_count: i64, gaps: i64, points_right: bool) -> Vec<i64> {
    let arrow_count = ((free_count - 1) / gaps).max(1);
    let first_slot = (free_count - 1 - (arrow_count - 1) * gaps) / 2;
    (0..arrow_count)
        .map(|arrow_index| {
            let slot = first_slot + arrow_index * gaps;
            if points_right {
                slot
            } else {
                free_count - 1 - slot
            }
        })
        .collect()
}

/// Mark bend arches at their level midpoint. A bitmap renderer gets an arrow mark on the curve; a
/// terminal gets a glyph on the cell it falls in.
fn draw_bend_arrows(
    buffer: &mut WorldBuffer,
    viewport_graph: &ViewportGraph,
    route: &[WorldPos],
    (edge_style, content_rects, color): (EdgeStyle, &HashMap<WorldPos, WorldRect>, Color),
    deferred: &mut Vec<DeferredEdge>,
) {
    for window in route.windows(3) {
        let [before, pin, after] = window else {
            continue;
        };
        let Some(arch) = bend_arch(
            viewport_graph,
            *pin,
            content_rects,
            Color::Reset,
            LineStyle::Normal,
        ) else {
            continue;
        };
        let points_right = after.x >= before.x;
        let midpoint = arch.midpoint();
        if edge_style.is_deferred() {
            deferred.push(DeferredEdge::Arrow(ArrowMark {
                center: buffer.world_to_terminal_continuous(midpoint),
                direction: (if points_right { 1.0 } else { -1.0 }, 0.0),
                color,
            }));
            continue;
        }
        let position = WorldPos::new(midpoint.0.floor() as i64, midpoint.1.floor() as i64);
        let arrow = if points_right { '▶' } else { '◀' };
        if let Some((character, style)) = buffer.get_char_styled(position)
            && is_blank_or_braille(character)
        {
            buffer.set_char_styled(position, arrow, style);
        }
    }
}

/// Only a plain horizontal stroke can be replaced without hiding a turn or junction.
fn is_horizontal_wire(character: char) -> bool {
    matches!(character, '─' | '━' | '╌' | '┄' | '┈')
}

/// Whether a curved route's cell is blank or holds braille an arrow may replace.
fn is_blank_or_braille(character: char) -> bool {
    character == ' ' || is_braille(character)
}

#[cfg(test)]
mod diagonal_highlight_tests {
    use petgraph::graph::NodeIndex;
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};

    use super::{PathStyle, PlotDecorations, plot_viewport_graph_with_highlights};
    use crate::{
        diagonal::{DeferredEdge, DiagonalDrawing, EdgeStyle, is_braille},
        geometry::{LocalPos, WorldPos},
        graph_widget::MinimalNodeRenderer,
        layout::{LayoutNode, NodeRole},
        theme::current_theme,
        viewport_graph::ViewportGraph,
        viewport_state::{ViewportState, WorldBuffer},
    };

    /// Two crossing diagonals between routing points: edge 0 runs up, edge 1 runs down, and
    /// they meet in the middle cells.
    fn crossing_graph() -> (ViewportGraph, [WorldPos; 4]) {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(10, 6),
            WorldPos::new(0, 6),
            WorldPos::new(10, 0),
        ];
        let mut graph = ViewportGraph::empty();
        for point in &points {
            graph.node_data_by_pos.insert(
                *point,
                LayoutNode::new(
                    NodeRole::Routing,
                    LocalPos::new_xy(point.x, point.y),
                    (1, 1),
                    None,
                ),
            );
            graph.graph.add_node(*point);
        }
        graph.graph.add_edge(
            points[0],
            points[1],
            vec![(NodeIndex::new(0), NodeIndex::new(1))],
        );
        graph.graph.add_edge(
            points[2],
            points[3],
            vec![(NodeIndex::new(2), NodeIndex::new(3))],
        );
        (graph, points)
    }

    fn plot(
        graph: &ViewportGraph,
        drawing: DiagonalDrawing,
        highlights: &[((WorldPos, WorldPos), PathStyle)],
        lowlights: &[(WorldPos, WorldPos)],
    ) -> (Buffer, Vec<DeferredEdge>) {
        let state = ViewportState {
            camera_current: WorldPos::new(5, 3),
            viewport_bounds: Rect::new(0, 0, 12, 8),
            ..ViewportState::default()
        };
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let theme = current_theme();
        let lines = {
            let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
            plot_viewport_graph_with_highlights::<
                MinimalNodeRenderer,
                petgraph::graph::DiGraph<(), ()>,
            >(
                graph,
                &mut world_buffer,
                &MinimalNodeRenderer,
                &petgraph::graph::DiGraph::new(),
                PlotDecorations {
                    node_highlights: &[],
                    edge_highlights: highlights,
                    hairpin_highlights: &[],
                    cell_highlights: &[],
                    lowlights,
                    node_lowlights: &[],
                },
                &theme,
                EdgeStyle::Diagonal(drawing),
            )
        };
        (buffer, lines)
    }

    fn colors(buffer: &Buffer) -> Vec<Color> {
        buffer
            .content
            .iter()
            .filter(|cell| cell.symbol().chars().any(is_braille))
            .map(|cell| cell.fg)
            .collect()
    }

    #[test]
    fn test_highlighted_edge_keeps_its_color_where_edges_cross() {
        let (graph, points) = crossing_graph();
        for highlighted in [(points[0], points[1]), (points[2], points[3])] {
            let style = PathStyle::new(Color::Red);
            let (buffer, _) = plot(
                &graph,
                DiagonalDrawing::Braille,
                &[(highlighted, style)],
                &[],
            );
            let red = colors(&buffer).iter().filter(|c| **c == Color::Red).count();
            // The highlighted line has the same dots whichever edge is stored first, and its
            // crossing cells keep its color.
            assert!(
                red >= 8,
                "highlight should stay visible, got {red} red cells"
            );
        }
    }

    #[test]
    fn test_deferred_lines_list_highlights_last_and_lowlights_first() {
        let (graph, points) = crossing_graph();
        let bold_red = PathStyle::new(Color::Red).with_line_style(crate::plotter::LineStyle::Bold);
        let (buffer, lines) = plot(
            &graph,
            DiagonalDrawing::Deferred,
            &[((points[0], points[1]), bold_red)],
            &[(points[2], points[3])],
        );
        assert!(
            colors(&buffer).is_empty(),
            "deferred drawing leaves no dots"
        );
        assert_eq!(lines.len(), 2);
        let theme = current_theme();
        let DeferredEdge::Line(dimmed) = lines[0] else {
            panic!("should defer a line");
        };
        let DeferredEdge::Line(highlighted) = lines[1] else {
            panic!("should defer a line");
        };
        assert_eq!(dimmed.color, theme[0x04], "dimmed edge first");
        assert_eq!(dimmed.line_style, crate::plotter::LineStyle::Dashed);
        assert_eq!(highlighted.color, Color::Red, "highlight last");
        assert_eq!(highlighted.line_style, crate::plotter::LineStyle::Bold);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use petgraph::graph::NodeIndex;
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};

    use super::{
        ARROW_GAPS, DIAGONAL_ARROW_GAPS, LineStyle, cells_in_both_directions, draw_arrows,
        draw_edge_with_style, pinned_edge_route,
    };
    use crate::{
        diagonal::{
            DeferredEdge, DiagonalDrawing, EdgeStyle, draw_braille_bezier, draw_braille_line,
            routed_edge_curve,
        },
        edge_ports::{EdgePorts, NodeSide},
        geometry::{LocalPos, WorldPos},
        layout::{LayoutNode, NodeRole, PinSide},
        pinning::PinnedEdge,
        viewport_graph::ViewportGraph,
        viewport_state::{ViewportState, WorldBuffer},
    };

    fn pinned(source: NodeIndex, target: NodeIndex, ports: EdgePorts) -> PinnedEdge {
        PinnedEdge {
            source,
            target,
            ports,
        }
    }

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
        // A route leaving westward leaves its source's west side and arrives on its east side.
        let ports = if points[1].x < points[0].x {
            EdgePorts::new(NodeSide::West, NodeSide::East)
        } else {
            EdgePorts::default()
        };
        draw_arrows(
            &mut world_buffer,
            &graph,
            &pinned(source, target, ports),
            ARROW_GAPS,
            (EdgeStyle::Rectilinear, &HashMap::new(), Color::Reset),
            &HashSet::new(),
            &mut Vec::new(),
        );
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

    /// The arrows a diagonal route along one row carries when `end_nodes` stand at its two ends.
    fn diagonal_arrows_between(end_nodes: [NodeRole; 2], length: i64) -> usize {
        let (mut graph, source, target) =
            routed_edge(&[WorldPos::new(0, 0), WorldPos::new(length, 0)]);
        for (position, role) in [WorldPos::new(0, 0), WorldPos::new(length, 0)]
            .into_iter()
            .zip(end_nodes)
        {
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::new(role, LocalPos::new_xy(position.x, position.y), (1, 1), None),
            );
        }
        let state = viewport();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
        draw_arrows(
            &mut world_buffer,
            &graph,
            &pinned(source, target, EdgePorts::default()),
            DIAGONAL_ARROW_GAPS,
            (
                EdgeStyle::Diagonal(DiagonalDrawing::Braille),
                &HashMap::new(),
                Color::Reset,
            ),
            &HashSet::new(),
            &mut Vec::new(),
        );
        (-5..=length + 5)
            .filter(|&x| matches!(world_buffer.get_char(WorldPos::new(x, 0)), Some('▶' | '◀')))
            .count()
    }

    #[test]
    fn test_short_diagonal_run_between_routing_nodes_keeps_its_arrow() {
        assert_eq!(
            diagonal_arrows_between([NodeRole::Routing, NodeRole::Routing], 3),
            1
        );
    }

    #[test]
    fn test_short_diagonal_run_at_a_data_node_has_no_arrow() {
        let data = |index| NodeRole::Data(NodeIndex::new(index));
        assert_eq!(diagonal_arrows_between([data(0), NodeRole::Routing], 3), 0);
        assert_eq!(diagonal_arrows_between([NodeRole::Routing, data(1)], 3), 0);
        assert_eq!(diagonal_arrows_between([data(0), data(1)], 3), 0);
    }

    #[test]
    fn test_long_diagonal_run_at_a_data_node_keeps_its_arrow() {
        let data = |index| NodeRole::Data(NodeIndex::new(index));
        assert_eq!(diagonal_arrows_between([data(0), data(1)], 12), 1);
    }

    #[test]
    fn test_diagonal_direction_markers_use_wider_repeat_spacing() {
        let (mut graph, source, target) = routed_edge(&[WorldPos::new(0, 0), WorldPos::new(60, 0)]);
        for position in [WorldPos::new(0, 0), WorldPos::new(60, 0)] {
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::routing(LocalPos::new_xy(position.x, position.y), (1, 1)),
            );
        }
        let state = viewport();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
        draw_arrows(
            &mut world_buffer,
            &graph,
            &pinned(source, target, EdgePorts::default()),
            DIAGONAL_ARROW_GAPS,
            (
                EdgeStyle::Diagonal(DiagonalDrawing::Braille),
                &HashMap::new(),
                Color::Reset,
            ),
            &HashSet::new(),
            &mut Vec::new(),
        );
        let positions: Vec<_> = (0..=60)
            .filter(|&x| world_buffer.get_char(WorldPos::new(x, 0)) == Some('▶'))
            .collect();
        assert_eq!(positions, [10, 30, 50]);
    }

    fn viewport() -> ViewportState {
        ViewportState {
            viewport_bounds: Rect::new(0, 0, 100, 80),
            ..ViewportState::default()
        }
    }

    #[test]
    fn test_rectilinear_arrows_preserve_corners_and_junctions() {
        let points = [WorldPos::new(8, 0), WorldPos::new(0, 0)];
        let (graph, source, target) = routed_edge(&points);
        let state = viewport();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
        draw_edge_with_style(
            &mut world_buffer,
            points[1],
            points[0],
            Color::Red,
            LineStyle::Normal,
        );
        for (x, glyph) in [(2, '╮'), (4, '┼'), (6, '┬')] {
            world_buffer.set_char_styled(
                WorldPos::new(x, 0),
                glyph,
                ratatui::style::Style::default().fg(Color::Red),
            );
        }
        draw_arrows(
            &mut world_buffer,
            &graph,
            &pinned(
                source,
                target,
                EdgePorts::new(NodeSide::West, NodeSide::East),
            ),
            1,
            (EdgeStyle::Rectilinear, &HashMap::new(), Color::Reset),
            &HashSet::new(),
            &mut Vec::new(),
        );
        for (x, glyph) in [(2, '╮'), (4, '┼'), (6, '┬')] {
            assert_eq!(world_buffer.get_char(WorldPos::new(x, 0)), Some(glyph));
        }
        assert!((0..=8).any(|x| world_buffer.get_char(WorldPos::new(x, 0)) == Some('◀')));
    }

    #[test]
    fn test_curved_route_that_changes_row_carries_no_arrows() {
        let points = [WorldPos::new(20, 0), WorldPos::new(-20, -15)];
        let (mut graph, source, target) = routed_edge(&points);
        for point in points {
            graph.node_data_by_pos.insert(
                point,
                LayoutNode::new(
                    NodeRole::Pin(PinSide::Left),
                    LocalPos::new_xy(point.x, point.y),
                    (1, 1),
                    None,
                ),
            );
        }
        let state = viewport();
        let mut buffer = Buffer::empty(state.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
        draw_arrows(
            &mut world_buffer,
            &graph,
            &pinned(
                source,
                target,
                EdgePorts::new(NodeSide::West, NodeSide::East),
            ),
            ARROW_GAPS,
            (
                EdgeStyle::Bezier(DiagonalDrawing::Deferred),
                &HashMap::new(),
                Color::Reset,
            ),
            &HashSet::new(),
            &mut Vec::new(),
        );
        for x in -25..25 {
            for y in -20..5 {
                let character = world_buffer
                    .get_char_styled(WorldPos::new(x, y))
                    .map(|(character, _)| character);
                assert!(!matches!(character, Some('▶' | '◀' | '▲' | '▼')));
            }
        }
    }

    /// A data anchor, a loopback pin and a routing return node, joined as one backward route from
    /// the anchor to the return node.
    fn loopback_route(returning_x: i64) -> (ViewportGraph, NodeIndex, NodeIndex, [WorldPos; 3]) {
        let anchor = WorldPos::new(10, 0);
        let pin = WorldPos::new(30, 0);
        let returning = WorldPos::new(returning_x, -10);
        let source = NodeIndex::new(0);
        let target = NodeIndex::new(1);
        let mut graph = ViewportGraph::empty();
        graph.node_positions.insert(source, anchor);
        graph.node_positions.insert(target, returning);
        for (position, role) in [
            (anchor, NodeRole::Data(source)),
            (pin, NodeRole::Pin(PinSide::Left)),
            (returning, NodeRole::Routing),
        ] {
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::new(role, LocalPos::new_xy(position.x, position.y), (1, 1), None),
            );
        }
        graph.graph.add_edge(anchor, pin, vec![(source, target)]);
        graph.graph.add_edge(pin, returning, vec![(source, target)]);
        (graph, source, target, [anchor, pin, returning])
    }

    #[test]
    fn test_loopback_braille_path_and_arrows_snapshot() {
        let (graph, source, target, [anchor, pin, returning]) = loopback_route(-20);
        let state = ViewportState {
            camera_current: WorldPos::new(5, -5),
            viewport_bounds: Rect::new(0, 0, 55, 16),
            ..ViewportState::default()
        };
        let mut buffer = Buffer::empty(state.viewport_bounds);
        {
            let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
            for (first, second) in [(anchor, pin), (pin, returning)] {
                let routed = routed_edge_curve(
                    &graph,
                    (first, &graph.node_data_by_pos[&first]),
                    (second, &graph.node_data_by_pos[&second]),
                    &HashMap::new(),
                    Color::Red,
                    LineStyle::Normal,
                );
                if let Some(stub) = routed.stub {
                    let from = world_buffer.world_to_terminal_continuous(stub.from);
                    let to = world_buffer.world_to_terminal_continuous(stub.to);
                    draw_braille_line(&mut world_buffer, from, to, Color::Red, LineStyle::Normal);
                }
                let curve = routed
                    .curve
                    .mapped(|point| world_buffer.world_to_terminal_continuous(point));
                draw_braille_bezier(&mut world_buffer, &curve);
            }
            draw_arrows(
                &mut world_buffer,
                &graph,
                &pinned(
                    source,
                    target,
                    EdgePorts::new(NodeSide::East, NodeSide::East),
                ),
                ARROW_GAPS,
                (
                    EdgeStyle::Bezier(DiagonalDrawing::Braille),
                    &HashMap::new(),
                    Color::Red,
                ),
                &HashSet::new(),
                &mut Vec::new(),
            );
        }
        let rows = (0..16)
            .map(|row| {
                (0..55)
                    .map(|column| buffer[(column, row)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(rows.matches('◀').count(), 3);
        insta::assert_snapshot!("loopback_braille_path_and_arrows", rows);
    }

    #[test]
    fn test_pinned_edge_route_follows_only_its_bundle() {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(20, 0),
            WorldPos::new(20, -10),
            WorldPos::new(-20, -10),
            WorldPos::new(-20, 0),
            WorldPos::new(-10, 0),
        ];
        let (mut graph, source, target) = routed_edge(&points);
        graph
            .graph
            .add_edge(points[0], points[5], vec![(target, source)]);
        graph.graph.add_edge(
            points[2],
            WorldPos::new(30, -10),
            vec![(source, NodeIndex::new(2))],
        );
        assert_eq!(
            pinned_edge_route(&graph, &pinned(source, target, EdgePorts::default())),
            Some(points.to_vec())
        );
    }

    #[test]
    fn test_pinned_edge_route_picks_the_route_on_its_sides() {
        // Two parallel edges between the same nodes: one leaves and arrives on the east sides
        // through a pin to the right, the other on the west sides through a pin to the left.
        let source = NodeIndex::new(0);
        let target = NodeIndex::new(1);
        let (source_pos, target_pos) = (WorldPos::new(0, 0), WorldPos::new(20, 0));
        let (left, right) = (WorldPos::new(-10, 0), WorldPos::new(30, 5));
        let mut graph = ViewportGraph::empty();
        graph.node_positions.insert(source, source_pos);
        graph.node_positions.insert(target, target_pos);
        for (from, to) in [
            (source_pos, right),
            (right, target_pos),
            (source_pos, left),
            (left, target_pos),
        ] {
            graph.graph.add_edge(from, to, vec![(source, target)]);
        }
        let east_east = EdgePorts::new(NodeSide::East, NodeSide::East);
        let west_west = EdgePorts::new(NodeSide::West, NodeSide::West);
        assert_eq!(
            pinned_edge_route(&graph, &pinned(source, target, east_east)),
            Some(vec![source_pos, right, target_pos])
        );
        assert_eq!(
            pinned_edge_route(&graph, &pinned(source, target, west_west)),
            Some(vec![source_pos, left, target_pos])
        );
    }

    #[test]
    fn test_rectilinear_eastward_route_carries_no_arrows() {
        let points = [WorldPos::new(0, 0), WorldPos::new(60, 0)];
        assert!(rendered_arrows(&points, &viewport()).is_empty());
    }

    #[test]
    fn test_rectilinear_arrows_mark_only_westward_runs() {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(40, 0),
            WorldPos::new(40, 10),
            WorldPos::new(-40, 10),
            WorldPos::new(-40, 0),
            WorldPos::new(-20, 0),
        ];
        let arrows = rendered_arrows(&points, &viewport());
        let count_of = |expected: char| arrows.iter().filter(|(_, c)| *c == expected).count();
        // The eastward runs of 40 and 20 cells are implied; the 80 cell westward run carries
        // eight markers.
        assert_eq!(count_of('▶'), 0);
        assert_eq!(count_of('◀'), 8);
        assert_eq!(count_of('▲') + count_of('▼'), 0);
        assert!(
            arrows
                .iter()
                .filter(|(_, c)| *c == '◀')
                .all(|(position, _)| position.y == 10)
        );
    }

    #[test]
    fn test_arrows_are_spread_evenly_and_never_on_a_run_end() {
        for length in [2, 5, 20, 33, 80] {
            let points = [WorldPos::new(length, 0), WorldPos::new(0, 0)];
            let positions: Vec<i64> = rendered_arrows(&points, &viewport())
                .iter()
                .map(|(position, _)| position.x)
                .collect();
            assert!(!positions.is_empty());
            assert!(
                positions.iter().all(|&x| x > 0 && x < length),
                "{positions:?}"
            );
            let gaps: Vec<i64> = positions.windows(2).map(|pair| pair[1] - pair[0]).collect();
            assert!(
                gaps.iter().all(|gap| (gap - gaps[0]).abs() <= 1),
                "{gaps:?}"
            );
        }
    }

    #[test]
    fn test_arrows_are_ten_cells_apart_with_balanced_end_margins() {
        for (length, expected) in [
            (9, vec![4]),
            (19, vec![9]),
            (20, vec![5, 15]),
            (25, vec![7, 17]),
            (30, vec![5, 15, 25]),
        ] {
            // A leftward run marks the same cells as a rightward one over them.
            let points = [WorldPos::new(length, 0), WorldPos::new(0, 0)];
            let positions: Vec<i64> = rendered_arrows(&points, &viewport())
                .into_iter()
                .map(|(position, _)| position.x)
                .collect();
            assert_eq!(positions, expected, "arrow positions for length {length}");
        }
    }

    #[test]
    fn test_arrow_slots_mirror_for_leftward_runs() {
        for free_count in 1..=60 {
            let rightward = super::arrow_slots(free_count, 10, true);
            let mut leftward_mirrored: Vec<i64> = super::arrow_slots(free_count, 10, false)
                .into_iter()
                .map(|slot| free_count - 1 - slot)
                .collect();
            leftward_mirrored.sort_unstable();
            assert_eq!(rightward, leftward_mirrored, "free count {free_count}");
        }
    }

    #[test]
    fn test_arrow_slots_round_toward_the_same_physical_side() {
        // Twelve free cells leave an odd cell over: rightward travel puts the arrow on cell 5 from
        // the left, and leftward travel must put it on the same cell, which is 6 from its start.
        assert_eq!(super::arrow_slots(12, 10, true), vec![5]);
        assert_eq!(super::arrow_slots(12, 10, false), vec![6]);
        assert_eq!(super::arrow_slots(25, 10, true), vec![7, 17]);
        assert_eq!(super::arrow_slots(25, 10, false), vec![17, 7]);
    }

    #[test]
    fn test_arrows_ignore_how_routing_nodes_split_a_run() {
        let points = [WorldPos::new(60, 0), WorldPos::new(0, 0)];
        let split_points = [
            WorldPos::new(60, 0),
            WorldPos::new(33, 0),
            WorldPos::new(7, 0),
            WorldPos::new(0, 0),
        ];
        assert_eq!(
            rendered_arrows(&split_points, &viewport()),
            rendered_arrows(&points, &viewport())
        );
    }

    #[test]
    fn test_arrows_keep_spacing_when_route_is_clipped() {
        let points = [WorldPos::new(40, 0), WorldPos::new(-40, 0)];
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
    fn test_pinned_edge_route_traces_self_loop_from_outgoing_port() {
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
            pinned_edge_route(&graph, &pinned(source, target, EdgePorts::default())),
            Some(points.to_vec())
        );
        // Only the 40 cell westward run carries markers, four of them.
        assert_eq!(rendered_arrows(&points, &viewport()).len(), 4);
    }

    #[test]
    fn test_pinned_edge_route_does_not_decorate_disconnected_segments() {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(20, 0),
            WorldPos::new(40, 0),
        ];
        let (mut graph, source, target) = routed_edge(&points);
        graph.graph.remove_edge(points[1], points[2]);
        assert!(pinned_edge_route(&graph, &pinned(source, target, EdgePorts::default())).is_none());
    }

    /// A node at the origin and a hairpin route across a hair pin: the bus runs `bus` cells to a
    /// hub that feeds two flat edges, which a bend joins `width` cells further out.
    fn hairpin_route(bus: i64, width: i64) -> (ViewportGraph, NodeIndex, [WorldPos; 4], WorldPos) {
        let source = NodeIndex::new(0);
        let origin = WorldPos::new(0, 0);
        let hub = WorldPos::new(bus, 0);
        let direction = if width < 0 { -1 } else { 1 };
        let near = bus + direction * 2;
        let corners = [
            WorldPos::new(near, 1),
            WorldPos::new(near, -1),
            WorldPos::new(near + width, 1),
            WorldPos::new(near + width, -1),
        ];
        let mut graph = ViewportGraph::empty();
        graph.node_positions.insert(source, origin);
        let mut edges = vec![
            (corners[0], corners[2]),
            (corners[1], corners[3]),
            (corners[2], corners[3]),
        ];
        edges.extend([(hub, corners[0]), (hub, corners[1])]);
        if bus != 0 {
            edges.push((origin, hub));
        }
        for (first, second) in edges {
            graph.graph.add_edge(first, second, vec![(source, source)]);
        }
        (graph, source, corners, hub)
    }

    #[test]
    fn test_hairpin_route_leaves_by_the_top_leg_and_returns_by_the_bottom() {
        for (width, side) in [(3, NodeSide::East), (-3, NodeSide::West)] {
            let (graph, source, corners, _) = hairpin_route(0, width);
            let route =
                pinned_edge_route(&graph, &pinned(source, source, EdgePorts::new(side, side)));
            let origin = WorldPos::new(0, 0);
            assert_eq!(
                route,
                Some(vec![
                    origin, corners[0], corners[2], corners[3], corners[1], origin
                ])
            );
        }
    }

    #[test]
    fn test_hairpin_route_runs_out_and_back_along_a_shared_bus() {
        let (graph, source, corners, hub) = hairpin_route(4, 3);
        let route = pinned_edge_route(
            &graph,
            &pinned(
                source,
                source,
                EdgePorts::new(NodeSide::East, NodeSide::East),
            ),
        );
        let origin = WorldPos::new(0, 0);
        assert_eq!(
            route,
            Some(vec![
                origin, hub, corners[0], corners[2], corners[3], corners[1], hub, origin
            ])
        );
    }

    #[test]
    fn test_hairpin_arrows_leave_on_the_top_row_and_return_on_the_bottom() {
        // Route arrows follow both legs; the Bezier arc uses its own single bend arrow.
        for (bus, width, side, top_points_right) in [
            (4, 3, NodeSide::East, true),
            (-4, -3, NodeSide::West, false),
        ] {
            let (mut graph, source, corners, hub) = hairpin_route(bus, width);
            let origin = WorldPos::new(0, 0);
            graph.node_data_by_pos.insert(
                origin,
                LayoutNode::new(NodeRole::Data(source), LocalPos::new_xy(0, 0), (1, 1), None),
            );
            for position in corners.into_iter().chain([hub]) {
                graph.node_data_by_pos.insert(
                    position,
                    LayoutNode::new(
                        NodeRole::Routing,
                        LocalPos::new_xy(position.x, position.y),
                        (1, 1),
                        None,
                    ),
                );
            }
            let pinned_edge = pinned(source, source, EdgePorts::new(side, side));
            graph.pinned_edges = vec![pinned_edge];
            let state = viewport();
            let mut buffer = Buffer::empty(state.viewport_bounds);
            let mut world_buffer = WorldBuffer::new(&mut buffer, &state);
            let mut deferred = Vec::new();
            draw_arrows(
                &mut world_buffer,
                &graph,
                &pinned_edge,
                DIAGONAL_ARROW_GAPS,
                (
                    EdgeStyle::Bezier(DiagonalDrawing::Deferred),
                    &HashMap::new(),
                    Color::Red,
                ),
                &cells_in_both_directions(&graph),
                &mut deferred,
            );

            let mut arrows: Vec<_> = deferred
                .iter()
                .filter_map(|edge| match edge {
                    DeferredEdge::Arrow(mark) => Some(*mark),
                    _ => None,
                })
                .collect();
            arrows.sort_by(|first, second| first.center.1.total_cmp(&second.center.1));
            let directions: Vec<bool> = arrows.iter().map(|mark| mark.direction.0 > 0.0).collect();
            assert_eq!(directions, vec![top_points_right, !top_points_right]);
        }
    }
}
