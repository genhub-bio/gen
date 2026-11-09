//! Plots a `RoutedWindowGraph` into a cell buffer, clipping to the visible area as it goes.
//!
//! Rectilinear edges, junction glyphs, arrows and highlights are drawn into the buffer; a curved
//! layout mode instead returns overlay primitives for `pixel_overlay` to draw at pixel resolution.

use std::collections::{HashMap, HashSet, VecDeque, hash_map::Entry};

use petgraph::{
    graph::NodeIndex,
    visit::{GraphBase, NodeIndexable},
};
use ratatui::{
    style::{Color, Style},
    symbols::merge::MergeStrategy,
};

use crate::{
    curved_edges::{bend_arch, hairpin_arc_arrow, hairpin_edge_arcs, routed_edge_curve},
    edge_ports::{NodeSide, PortedEdge},
    geometry::{WorldPos, WorldRect},
    highlight_style::{CellHighlight, LineStyle, PathStyle},
    layout::{JunctionSymbol, NodeRole},
    layout_mode::LayoutMode,
    node_renderer::NodeRenderer,
    overlay_primitive::{ArrowMark, NodeGlow, OverlayPrimitive, StraightSegment, SymbolGlow},
    routed_window_graph::RoutedWindowGraph,
    theme::Theme,
    window_geometry::NODE_GLYPH,
    world_buffer::WorldBuffer,
};

/// Rectilinear routes retain their established marker rhythm.
const ARROW_GAPS: i64 = 10;
/// Curved runs are drawn smoothly and read easily, so repeat markers less often along them.
const CURVED_ARROW_GAPS: i64 = 20;
/// A curved run with fewer free cells than this is too short to carry a readable arrow when it
/// starts or ends at a data node, where forward and reverse routes share the same few cells.
const MIN_ARROW_RUN_CELLS: i64 = 4;

/// How far, in cells, a curved edge can stray from the box around its endpoints: the widest
/// loopback bulges 5 cells and a hairpin arc 4.75, and a stroke's halo adds a couple more.
const EDGE_REACH_CELLS: i64 = 8;

pub(crate) struct PlotDecorations<'a> {
    pub(crate) node_highlights: &'a [(WorldPos, PathStyle)],
    pub edge_highlights: &'a [((WorldPos, WorldPos), PathStyle)],
    pub(crate) hairpin_highlights: &'a [((NodeIndex, NodeSide), PathStyle)],
    pub(crate) cell_highlights: &'a [CellHighlight],
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
) -> OverlayPrimitive {
    OverlayPrimitive::NodeGlow(NodeGlow {
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
) -> OverlayPrimitive {
    let is_symbol = rect.min == rect.max
        && buffer
            .get_char_styled(rect.min)
            .is_some_and(|(character, _)| character == NODE_GLYPH);
    if !is_symbol {
        return cell_glow(buffer, rect, container, color);
    }
    OverlayPrimitive::SymbolGlow(SymbolGlow {
        center: buffer
            .world_to_terminal_continuous((rect.min.x as f64 + 0.5, rect.min.y as f64 + 0.5)),
        color,
    })
}

/// Plot a routed window graph with resolved highlights and lowlights.
///
/// Returns the overlay primitives of a curved layout mode, which the buffer does not contain; the
/// rectilinear layout mode draws all of its edges into the buffer and returns none.
pub(crate) fn plot_routed_window_graph_with_highlights<V, G>(
    routed_window_graph: &RoutedWindowGraph,
    buffer: &mut WorldBuffer<'_>,
    renderer: &V,
    original_graph: &G,
    decorations: PlotDecorations<'_>,
    theme: &Theme,
    layout_mode: LayoutMode,
) -> Vec<OverlayPrimitive>
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
    let content_rects: HashMap<_, _> = routed_window_graph
        .nodes()
        .filter_map(|(position, node)| {
            if !layout_mode.is_curved() {
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

    let ordinary_color = theme[0x05];
    let lowlight_color = theme[0x04];
    let visible_cells = buffer.visible_cells();
    // An edge is worth building only if the box around its endpoints, grown by how far a curve
    // can stray from them, meets the viewport. Edges whose endpoints are both off screen still
    // pass when they cross it.
    let may_reach_view = |endpoints: [WorldPos; 2]| {
        let Some(visible) = visible_cells else {
            return false;
        };
        let region = endpoints
            .into_iter()
            .map(|position| {
                content_rects.get(&position).copied().unwrap_or_else(|| {
                    WorldRect::from_coords(position.x, position.y, position.x, position.y)
                })
            })
            .reduce(|first, second| {
                WorldRect::from_coords(
                    first.min.x.min(second.min.x),
                    first.min.y.min(second.min.y),
                    first.max.x.max(second.max.x),
                    first.max.y.max(second.max.y),
                )
            })
            .expect("should have two endpoints");
        let margin = EDGE_REACH_CELLS;
        visible.intersects(&WorldRect::from_coords(
            region.min.x - margin,
            region.min.y - margin,
            region.max.x + margin,
            region.max.y + margin,
        ))
    };
    let mut overlay_primitives = Vec::new();
    let mut stroke = |buffer: &mut WorldBuffer<'_>,
                      source: WorldPos,
                      target: WorldPos,
                      color: Color,
                      line_style: LineStyle| {
        if !layout_mode.is_curved() {
            draw_rectilinear_segment(buffer, source, target, color, line_style);
            return;
        }
        let (Some(source_node), Some(target_node)) = (
            routed_window_graph.node_data_by_pos.get(&source),
            routed_window_graph.node_data_by_pos.get(&target),
        ) else {
            return;
        };
        if !may_reach_view([source, target]) {
            return;
        }
        // An arch through an up or down pin is drawn once, with the pin's west half.
        let bend_pin = [source, target].into_iter().find(|&position| {
            bend_arch(
                routed_window_graph,
                position,
                &content_rects,
                color,
                line_style,
            )
            .is_some()
        });
        if let Some(pin) = bend_pin
            && [source, target].iter().any(|position| position.x > pin.x)
        {
            return;
        }
        let routed = routed_edge_curve(
            routed_window_graph,
            (source, source_node),
            (target, target_node),
            &content_rects,
            color,
            line_style,
        );
        if let Some(stub) = routed.stub {
            overlay_primitives.push(OverlayPrimitive::Line(StraightSegment {
                from: buffer.world_to_terminal_continuous(stub.from),
                to: buffer.world_to_terminal_continuous(stub.to),
                ..stub
            }));
        }
        overlay_primitives.push(OverlayPrimitive::Bezier(
            routed
                .curve
                .mapped(|point| buffer.world_to_terminal_continuous(point)),
        ));
    };

    // Draw edges first so nodes appear on top
    let mut edges: Vec<_> = routed_window_graph.edges().collect();
    if layout_mode.is_curved() {
        // Curves are painted in order, so draw dimmed edges first and highlighted ones last: a
        // highlight is never painted over.
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
        if layout_mode.is_curved() && source == target {
            continue;
        }
        let (color, line_style) = edge_look(source, target, !bundle.is_empty());
        stroke(buffer, source, target, color, line_style);
    }

    if layout_mode.is_curved() {
        let mut hairpins: Vec<_> = routed_window_graph
            .domain_edges
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
            let Some(&position) = routed_window_graph.node_positions.get(&edge.source) else {
                continue;
            };
            if !routed_window_graph.graph.contains_edge(position, position) {
                continue;
            }
            let Some(node) = routed_window_graph.node_data_by_pos.get(&position) else {
                continue;
            };
            if !may_reach_view([position, position]) {
                continue;
            }
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
            overlay_primitives.push(OverlayPrimitive::Arrow(hairpin_arc_arrow(
                &arcs,
                ordinary_color,
            )));
            overlay_primitives.extend(arcs.into_iter().map(OverlayPrimitive::Bezier));
        }
    }

    // Draw nodes. The layout window holds more nodes than fit on screen, and rendering one reads
    // its sequence from the database, so nodes outside the viewport are left alone.
    let visible_cells = buffer.visible_cells();
    for (world_pos, node) in routed_window_graph.nodes() {
        match &node.role {
            NodeRole::Data(domain_idx) => {
                let node_id = <G as NodeIndexable>::from_index(original_graph, domain_idx.index());
                let world_rect = node.vertical_anchor.rect(*world_pos, node.size);
                let is_on_screen =
                    visible_cells.is_some_and(|visible| visible.intersects(&world_rect));
                if is_on_screen {
                    renderer.render_node(buffer, world_rect, &node_id);
                }
                let content = renderer.content_rect(&node_id);
                let content_rect = WorldRect::from_coords(
                    world_rect.min.x + content.min.x,
                    world_rect.min.y + content.min.y,
                    world_rect.min.x + content.max.x,
                    world_rect.min.y + content.max.y,
                );

                // If lowlighted, dim the node background to theme[0x04].
                // Applied before highlights so highlights take priority.
                if is_on_screen && node_lowlights.contains(world_pos) {
                    tint_cells(buffer, content_rect, lowlight_color);
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
                    if is_on_screen {
                        tint_cells(buffer, content_rect, hl);
                    }
                    if layout_mode.is_curved() {
                        overlay_primitives.push(highlight_glow(
                            buffer,
                            content_rect,
                            content_rect,
                            hl,
                        ));
                    }
                }

                // Sub-rect highlight pass: tint only the matched column/row range.
                // first_cell/last_cell are node-local (col, row) offsets from world_rect.min, both inclusive.
                for (_, first_cell, last_cell, path_style) in
                    cell_highlights.iter().filter(|(nwp, ..)| nwp == world_pos)
                {
                    let hl = match path_style.color {
                        Color::Reset => theme[0x07],
                        color => color,
                    };
                    let x0 = (world_rect.min.x + first_cell.0).max(content_rect.min.x);
                    let x1 = (world_rect.min.x + last_cell.0).min(content_rect.max.x);
                    let y0 = (world_rect.min.y + first_cell.1).max(content_rect.min.y);
                    let y1 = (world_rect.min.y + last_cell.1).min(content_rect.max.y);
                    if is_on_screen && x0 <= x1 && y0 <= y1 {
                        tint_cells(buffer, WorldRect::from_coords(x0, y0, x1, y1), hl);
                    }
                    if layout_mode.is_curved() && x0 <= x1 && y0 <= y1 {
                        overlay_primitives.push(highlight_glow(
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
                if layout_mode.is_curved() && !matches!(node.role, NodeRole::Wormhole(_)) {
                    continue;
                }
                let edge_color = ordinary_color;
                let base_glyph =
                    compute_junction_glyph(routed_window_graph.neighbors(*world_pos), *world_pos);

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

                // Collect neighbors once; reused for both glyph primitive and color decisions.
                let all_neighbors: Vec<WorldPos> =
                    routed_window_graph.neighbors(*world_pos).collect();

                // Highlight takes full priority: skip lowlight logic entirely when present.
                let (character, fg_color) = if let Some(style) = highlighted_style {
                    let active_edges: Vec<_> = edge_highlights
                        .iter()
                        .filter(|(_, s)| *s == style)
                        .cloned()
                        .collect();
                    let highlight_graph = RoutedWindowGraph::from_visual_edges(&active_edges);
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
                    match routed_window_graph.neighbors(*world_pos).next() {
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

    // Edges are marked only where their route runs against the left-to-right flow. Routes are
    // traced once and shared by every pass below.
    let routes: Vec<Option<Vec<WorldPos>>> = routed_window_graph
        .domain_edges
        .iter()
        .map(|edge| edge_route(routed_window_graph, edge))
        .collect();
    let obstacles = ArrowObstacles::new(routed_window_graph, &routes);
    for (edge, route) in routed_window_graph.domain_edges.iter().zip(&routes) {
        let Some(route) = route else {
            continue;
        };
        // A curved hairpin carries its arrow on the bend itself.
        if layout_mode.is_curved()
            && edge.source == edge.target
            && edge.ports.source == edge.ports.target
        {
            continue;
        }
        if !route_runs_against_flow(routed_window_graph, route) {
            continue;
        }
        draw_arrows(
            buffer,
            routed_window_graph,
            route,
            if layout_mode.is_curved() {
                CURVED_ARROW_GAPS
            } else {
                ARROW_GAPS
            },
            (layout_mode, &content_rects, ordinary_color),
            &obstacles,
            &mut overlay_primitives,
        );
    }
    overlay_primitives
}

/// Recolor the on-screen cells of a node region: sequence glyphs take the color as foreground and
/// everything else as background. Cells outside the viewport are never visited, so a node
/// megabases wide costs no more than the screen.
fn tint_cells(buffer: &mut WorldBuffer<'_>, region: WorldRect, color: Color) {
    let Some(visible) = buffer
        .visible_cells()
        .and_then(|visible| visible.intersection(&region))
    else {
        return;
    };
    for y in visible.min.y..=visible.max.y {
        for x in visible.min.x..=visible.max.x {
            let position = WorldPos::new(x, y);
            if let Some((character, style)) = buffer.get_char_styled(position) {
                let new_style = if character == NODE_GLYPH {
                    style.fg(color)
                } else {
                    style.bg(color)
                };
                buffer.set_char_styled(position, character, new_style);
            }
        }
    }
}

/// Inclusive x-spans per row, sorted and merged. Arrows reason about whole stretches of wire
/// rather than single cells, since a route can run to the center of a node megabases wide.
type RowSpans = HashMap<i64, Vec<(i64, i64)>>;

/// Sort `spans` and merge those that overlap or touch.
fn merge_spans(spans: &mut Vec<(i64, i64)>) {
    spans.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(spans.len());
    for &(first, last) in spans.iter() {
        match merged.last_mut() {
            Some((_, previous_last)) if first <= *previous_last + 1 => {
                *previous_last = (*previous_last).max(last);
            }
            _ => merged.push((first, last)),
        }
    }
    *spans = merged;
}

/// The spans covered by both of two sorted, merged lists.
fn intersect_spans(first: &[(i64, i64)], second: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut overlap = Vec::new();
    let (mut first_index, mut second_index) = (0, 0);
    while first_index < first.len() && second_index < second.len() {
        let low = first[first_index].0.max(second[second_index].0);
        let high = first[first_index].1.min(second[second_index].1);
        if low <= high {
            overlap.push((low, high));
        }
        if first[first_index].1 < second[second_index].1 {
            first_index += 1;
        } else {
            second_index += 1;
        }
    }
    overlap
}

/// What an arrow cannot sit on: the data nodes, and the stretches of wire that routes run
/// through in both horizontal directions, such as the bus a hairpin shares with the edges
/// beside it. An arrow there would point one way along a wire that carries both.
struct ArrowObstacles {
    data_nodes: RowSpans,
    two_way: RowSpans,
}

impl ArrowObstacles {
    fn new(routed_window_graph: &RoutedWindowGraph, routes: &[Option<Vec<WorldPos>>]) -> Self {
        let mut data_nodes = RowSpans::new();
        for (center, node) in routed_window_graph.nodes() {
            if !matches!(node.role, NodeRole::Data(_)) {
                continue;
            }
            let rect = node.vertical_anchor.rect(*center, node.size);
            for row in rect.min.y..=rect.max.y {
                data_nodes
                    .entry(row)
                    .or_default()
                    .push((rect.min.x, rect.max.x));
            }
        }
        data_nodes.values_mut().for_each(merge_spans);

        let mut rightward = RowSpans::new();
        let mut leftward = RowSpans::new();
        for route in routes.iter().flatten() {
            for segment in route.windows(2) {
                let (start, end) = (segment[0], segment[1]);
                if start.y != end.y {
                    continue;
                }
                let spans = if end.x > start.x {
                    &mut rightward
                } else {
                    &mut leftward
                };
                spans
                    .entry(start.y)
                    .or_default()
                    .push((start.x.min(end.x), start.x.max(end.x)));
            }
        }
        let mut two_way = RowSpans::new();
        for (row, mut right_spans) in rightward {
            let Some(mut left_spans) = leftward.remove(&row) else {
                continue;
            };
            merge_spans(&mut right_spans);
            merge_spans(&mut left_spans);
            let overlap = intersect_spans(&right_spans, &left_spans);
            if !overlap.is_empty() {
                two_way.insert(row, overlap);
            }
        }
        Self {
            data_nodes,
            two_way,
        }
    }

    fn is_inside_node(&self, position: WorldPos) -> bool {
        self.data_nodes.get(&position.y).is_some_and(|spans| {
            spans
                .iter()
                .any(|&(first, last)| first <= position.x && position.x <= last)
        })
    }

    /// The spans of `first..=last` on `row` where an arrow may go, in ascending order.
    fn free_spans(&self, row: i64, first: i64, last: i64) -> Vec<(i64, i64)> {
        let mut blocked: Vec<(i64, i64)> = [&self.data_nodes, &self.two_way]
            .into_iter()
            .filter_map(|spans| spans.get(&row))
            .flatten()
            .copied()
            .filter(|&(blocked_first, blocked_last)| blocked_last >= first && blocked_first <= last)
            .collect();
        merge_spans(&mut blocked);
        let mut free = Vec::new();
        let mut next = first;
        for (blocked_first, blocked_last) in blocked {
            if blocked_first > next {
                free.push((next, blocked_first - 1));
            }
            next = next.max(blocked_last + 1);
        }
        if next <= last {
            free.push((next, last));
        }
        free
    }
}

/// How many of `spans`' cells lie west of column `x`.
fn cells_west_of(spans: &[(i64, i64)], x: i64) -> i64 {
    spans
        .iter()
        .map(|&(first, last)| (last.min(x - 1) - first + 1).max(0))
        .sum()
}

/// The column of the `index`th cell, counted from the west, of `spans`.
fn column_of_cell(spans: &[(i64, i64)], index: i64) -> Option<i64> {
    let mut remaining = index;
    for &(first, last) in spans {
        let length = last - first + 1;
        if remaining < length {
            return Some(first + remaining);
        }
        remaining -= length;
    }
    None
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
fn draw_rectilinear_segment(
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
/// along a segment heading west. Only such edges need arrows: rightward flow is the default
/// reading direction, so an arrow is worth drawing where that implication fails.
fn route_runs_against_flow(routed_window_graph: &RoutedWindowGraph, route: &[WorldPos]) -> bool {
    let passes_pin = route.iter().any(|position| {
        routed_window_graph
            .node_data_by_pos
            .get(position)
            .is_some_and(|node| matches!(node.role, NodeRole::Pin(_)))
    });
    passes_pin || route.windows(2).any(|segment| segment[1].x < segment[0].x)
}

/// Trace a domain edge's routed segments in source-to-target order, leaving its source
/// on the side its ports name and arriving at its target on the other named side. Bundles
/// distinguish the route from other edges sharing its junctions, and the sides from parallel
/// edges between the same two nodes.
fn edge_route(
    routed_window_graph: &RoutedWindowGraph,
    edge: &PortedEdge<NodeIndex>,
) -> Option<Vec<WorldPos>> {
    trace_edge_route(routed_window_graph, edge, |node| {
        routed_window_graph.node_positions.get(&node).copied()
    })
}

/// [`edge_route`] for highlighting, which also follows edges into and out of invisible
/// junction nodes, drawn as routing points.
pub(crate) fn highlighted_edge_route(
    routed_window_graph: &RoutedWindowGraph,
    edge: &PortedEdge<NodeIndex>,
) -> Option<Vec<WorldPos>> {
    trace_edge_route(routed_window_graph, edge, |node| {
        routed_window_graph
            .node_positions
            .get(&node)
            .or_else(|| routed_window_graph.junction_positions.get(&node))
            .copied()
    })
}

fn trace_edge_route(
    routed_window_graph: &RoutedWindowGraph,
    edge: &PortedEdge<NodeIndex>,
    position_of: impl Fn(NodeIndex) -> Option<WorldPos>,
) -> Option<Vec<WorldPos>> {
    let PortedEdge {
        source,
        target,
        ports,
        ..
    } = *edge;
    let source_pos = position_of(source)?;
    let target_pos = position_of(target)?;
    let follows_edge = |start, end| {
        routed_window_graph
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
            let mut neighbors: Vec<WorldPos> = routed_window_graph.neighbors(position).collect();
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
        for neighbor in routed_window_graph.neighbors(position) {
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
    routed_window_graph: &RoutedWindowGraph,
    route: &[WorldPos],
    layout_mode: LayoutMode,
    content_rects: &HashMap<WorldPos, WorldRect>,
) -> Vec<(WorldPos, WorldPos)> {
    if !layout_mode.is_curved() {
        return straight_runs(route)
            .into_iter()
            .filter(|(start, end)| start.y == end.y)
            .collect();
    }
    let mut runs: Vec<(WorldPos, WorldPos)> = Vec::new();
    let mut can_merge = false;
    for pair in route.windows(2) {
        let flat = (|| {
            let first = (pair[0], routed_window_graph.node_data_by_pos.get(&pair[0])?);
            let second = (pair[1], routed_window_graph.node_data_by_pos.get(&pair[1])?);
            let routed = routed_edge_curve(
                routed_window_graph,
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

/// Mark the direction of a domain edge on each westward horizontal run of its route (see
/// [`horizontal_runs`]), with `◀`. Eastward runs are the default reading direction and stay
/// bare, so an arrow only appears where that implication fails. A run carries at least one
/// arrow, unless it is shorter than [`MIN_ARROW_RUN_CELLS`] free cells and touches a data node,
/// with additional arrows exactly `gaps` cells apart. The unused distance is split between the
/// two ends. Vertical and slanted stretches are left bare. An arrow replaces only the wire of a
/// rectilinear route, or the blank cells of a curved one.
///
/// Runs are handled as spans, and only the arrows that fall inside the viewport are visited, so
/// the cost follows the screen and not the length of the run.
fn draw_arrows(
    buffer: &mut WorldBuffer,
    routed_window_graph: &RoutedWindowGraph,
    route: &[WorldPos],
    gaps: i64,
    (layout_mode, content_rects, color): (LayoutMode, &HashMap<WorldPos, WorldRect>, Color),
    obstacles: &ArrowObstacles,
    overlay_primitives: &mut Vec<OverlayPrimitive>,
) {
    let Some(visible) = buffer.visible_cells() else {
        return;
    };
    let accepts = if layout_mode.is_curved() {
        is_blank
    } else {
        is_horizontal_wire
    };
    if layout_mode.is_curved() {
        draw_bend_arrows(
            buffer,
            routed_window_graph,
            route,
            (content_rects, color),
            overlay_primitives,
        );
    }
    for (start, end) in horizontal_runs(routed_window_graph, route, layout_mode, content_rects) {
        let direction = WorldPos::new((end.x - start.x).signum(), 0);
        if direction.x > 0 || start.y < visible.min.y || start.y > visible.max.y {
            continue;
        }
        let free_spans = obstacles.free_spans(start.y, end.x.min(start.x), end.x.max(start.x));
        let free_count: i64 = free_spans
            .iter()
            .map(|&(first, last)| last - first + 1)
            .sum();
        if free_count == 0 {
            continue;
        }
        // Short runs between routing nodes and pins keep their arrow, since each one marks a
        // layer crossing of a long edge. A short run at a data node does not: routes in both
        // directions leave and enter it over the same cells.
        let touches_data_node = [start - direction, start, end, end + direction]
            .into_iter()
            .any(|position| obstacles.is_inside_node(position));
        if layout_mode.is_curved() && free_count < MIN_ARROW_RUN_CELLS && touches_data_node {
            continue;
        }
        // Free cells are numbered from the west. The first and last ones on screen bound the
        // arrows worth looking at: a slot outside the view stays skipped.
        let first_visible = cells_west_of(&free_spans, visible.min.x);
        let last_visible = cells_west_of(&free_spans, visible.max.x + 1) - 1;
        let layout = ArrowLayout::new(free_count, gaps);
        for slot in layout.slots_between(first_visible, last_visible) {
            let Some(column) = column_of_cell(&free_spans, slot) else {
                continue;
            };
            // A slot that lands on a corner or junction moves to the nearest cell that can take
            // the arrow. Leftward travel breaks ties toward the east, where its run begins.
            let accepts_cell = |index: i64| {
                column_of_cell(&free_spans, index)
                    .and_then(|column| buffer.get_char_styled(WorldPos::new(column, start.y)))
                    .is_some_and(|(character, _)| accepts(character))
            };
            if buffer
                .get_char_styled(WorldPos::new(column, start.y))
                .is_none()
            {
                continue;
            }
            let nearest = (0..=gaps)
                .flat_map(|distance| [slot + distance, slot - distance])
                .find(|&index| (0..free_count).contains(&index) && accepts_cell(index));
            let Some(index) = nearest else {
                continue;
            };
            let Some(column) = column_of_cell(&free_spans, index) else {
                continue;
            };
            let position = WorldPos::new(column, start.y);
            if layout_mode.is_curved() {
                overlay_primitives.push(OverlayPrimitive::Arrow(ArrowMark {
                    center: buffer.world_to_terminal_continuous((
                        position.x as f64 + 0.5,
                        position.y as f64 + 0.5,
                    )),
                    direction: (-1.0, 0.0),
                    color,
                }));
            } else if let Some((_, style)) = buffer.get_char_styled(position) {
                buffer.set_char_styled(position, '◀', style);
            }
        }
    }
}

/// Where a run's arrows sit among its free cells, numbered from the west. The leftover
/// distance is split by integer division, so the odd cell always falls on the same physical
/// side: a rightward edge and the leftward edge over the same cells would otherwise round
/// opposite ways and their arrows would sit one cell apart.
struct ArrowLayout {
    first_slot: i64,
    arrow_count: i64,
    gaps: i64,
}

impl ArrowLayout {
    fn new(free_count: i64, gaps: i64) -> Self {
        let arrow_count = ((free_count - 1) / gaps).max(1);
        let first_slot = (free_count - 1 - (arrow_count - 1) * gaps) / 2;
        Self {
            first_slot,
            arrow_count,
            gaps,
        }
    }

    /// The slots among cells `first..=last`, in ascending order.
    fn slots_between(&self, first: i64, last: i64) -> impl Iterator<Item = i64> + use<> {
        let Self {
            first_slot,
            arrow_count,
            gaps,
        } = *self;
        let low = (first - first_slot + gaps - 1).div_euclid(gaps).max(0);
        let high = (last - first_slot).div_euclid(gaps).min(arrow_count - 1);
        (low..=high).map(move |arrow_index| first_slot + arrow_index * gaps)
    }
}

/// Mark bend arches at their level midpoint with an arrow mark on the curve.
fn draw_bend_arrows(
    buffer: &mut WorldBuffer,
    routed_window_graph: &RoutedWindowGraph,
    route: &[WorldPos],
    (content_rects, color): (&HashMap<WorldPos, WorldRect>, Color),
    overlay_primitives: &mut Vec<OverlayPrimitive>,
) {
    for window in route.windows(3) {
        let [before, pin, after] = window else {
            continue;
        };
        let Some(arch) = bend_arch(
            routed_window_graph,
            *pin,
            content_rects,
            Color::Reset,
            LineStyle::Normal,
        ) else {
            continue;
        };
        // As on straight runs, an arch heading east needs no arrow.
        if after.x >= before.x {
            continue;
        }
        let midpoint = arch.midpoint();
        overlay_primitives.push(OverlayPrimitive::Arrow(ArrowMark {
            center: buffer.world_to_terminal_continuous(midpoint),
            direction: (-1.0, 0.0),
            color,
        }));
    }
}

/// Only a plain horizontal stroke can be replaced without hiding a turn or junction.
fn is_horizontal_wire(character: char) -> bool {
    matches!(character, '─' | '━' | '╌' | '┄' | '┈')
}

/// Whether a curved route's cell is blank, where an arrow mark may be placed.
fn is_blank(character: char) -> bool {
    character == ' '
}

#[cfg(test)]
mod curved_highlight_tests {
    use petgraph::{graph::NodeIndex, visit::GraphBase};
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};

    use super::{
        NodeRenderer, PathStyle, PlotDecorations, plot_routed_window_graph_with_highlights,
    };
    use crate::{
        geometry::{LocalPos, WorldPos, WorldRect},
        layout::{LayoutNode, NodeRole},
        layout_mode::LayoutMode,
        overlay_primitive::OverlayPrimitive,
        routed_window_graph::RoutedWindowGraph,
        theme::current_theme,
        window_geometry::NODE_GLYPH,
        world_buffer::{WorldBuffer, WorldViewport},
    };

    /// Draws every node as a single glyph, regardless of domain data.
    #[derive(Debug, Clone, Copy, Default)]
    struct MinimalNodeRenderer;

    impl<G: GraphBase> NodeRenderer<G> for MinimalNodeRenderer {
        fn get_node_size(&self, _node: &G::NodeId) -> (u64, u64) {
            (1, 1)
        }

        fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, _node_id: &G::NodeId) {
            buffer.set_char(area.left_center(), NODE_GLYPH);
        }
    }

    /// Two plain edges between routing nodes at different coordinates, so each is a diagonal:
    /// edge 0 runs up, edge 1 runs down, and they cross in the middle cells.
    fn crossing_graph() -> (RoutedWindowGraph, [WorldPos; 4]) {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(10, 6),
            WorldPos::new(0, 6),
            WorldPos::new(10, 0),
        ];
        let mut graph = RoutedWindowGraph::empty();
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
        graph: &RoutedWindowGraph,
        highlights: &[((WorldPos, WorldPos), PathStyle)],
        lowlights: &[(WorldPos, WorldPos)],
    ) -> (Buffer, Vec<OverlayPrimitive>) {
        let world_viewport = WorldViewport {
            camera_current: WorldPos::new(5, 3),
            viewport_bounds: Rect::new(0, 0, 12, 8),
            ..WorldViewport::default()
        };
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let theme = current_theme();
        let overlay_primitives = {
            let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
            plot_routed_window_graph_with_highlights::<
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
                LayoutMode::Curved,
            )
        };
        (buffer, overlay_primitives)
    }

    #[test]
    fn test_overlay_primitives_list_highlights_last_and_lowlights_first() {
        let (graph, points) = crossing_graph();
        let bold_red =
            PathStyle::new(Color::Red).with_line_style(crate::highlight_style::LineStyle::Bold);
        let (buffer, overlay_primitives) = plot(
            &graph,
            &[((points[0], points[1]), bold_red)],
            &[(points[2], points[3])],
        );
        assert!(
            buffer.content.iter().all(|cell| cell.symbol() == " "),
            "curved drawing leaves the buffer empty"
        );
        assert_eq!(overlay_primitives.len(), 2);
        let theme = current_theme();
        let OverlayPrimitive::Bezier(dimmed) = overlay_primitives[0] else {
            panic!("should report a curve");
        };
        let OverlayPrimitive::Bezier(highlighted) = overlay_primitives[1] else {
            panic!("should report a curve");
        };
        assert_eq!(dimmed.color, theme[0x04], "dimmed edge first");
        assert_eq!(dimmed.line_style, crate::highlight_style::LineStyle::Dashed);
        assert_eq!(highlighted.color, Color::Red, "highlight last");
        assert_eq!(
            highlighted.line_style,
            crate::highlight_style::LineStyle::Bold
        );
    }

    /// One edge between two routing points on row 3, which the camera of [`plot`] looks at.
    fn edge_along_row(from_x: i64, to_x: i64) -> RoutedWindowGraph {
        let mut graph = RoutedWindowGraph::empty();
        let points = [WorldPos::new(from_x, 3), WorldPos::new(to_x, 3)];
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
        graph
    }

    #[test]
    fn test_edge_far_from_the_viewport_builds_no_overlay_primitives() {
        let (_, overlay_primitives) = plot(&edge_along_row(1_000, 1_100), &[], &[]);
        assert!(overlay_primitives.is_empty());
    }

    #[test]
    fn test_edge_crossing_the_viewport_keeps_its_overlay_primitive_when_both_endpoints_are_off_screen()
     {
        let (_, overlay_primitives) = plot(&edge_along_row(-1_000, 1_000), &[], &[]);
        assert!(
            overlay_primitives
                .iter()
                .any(|primitive| matches!(primitive, OverlayPrimitive::Bezier(_)))
        );
    }

    /// Counts how many nodes it is asked to draw.
    struct CountingRenderer {
        rendered: std::cell::Cell<usize>,
    }

    impl NodeRenderer<petgraph::graph::DiGraph<(), ()>> for CountingRenderer {
        fn get_node_size(&self, _node: &NodeIndex) -> (u64, u64) {
            (1, 1)
        }

        fn render_node(&self, _buffer: &mut WorldBuffer, _area: WorldRect, _node_id: &NodeIndex) {
            self.rendered.set(self.rendered.get() + 1);
        }
    }

    #[test]
    fn test_nodes_outside_the_viewport_are_not_rendered() {
        let mut graph = RoutedWindowGraph::empty();
        for (index, x) in [(0, 3), (1, 5_000)] {
            let position = WorldPos::new(x, 3);
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::new(
                    NodeRole::Data(NodeIndex::new(index)),
                    LocalPos::new_xy(position.x, position.y),
                    (1, 1),
                    None,
                ),
            );
            graph.graph.add_node(position);
        }
        let world_viewport = WorldViewport {
            camera_current: WorldPos::new(5, 3),
            viewport_bounds: Rect::new(0, 0, 12, 8),
            ..WorldViewport::default()
        };
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let renderer = CountingRenderer {
            rendered: std::cell::Cell::new(0),
        };
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        plot_routed_window_graph_with_highlights(
            &graph,
            &mut world_buffer,
            &renderer,
            &petgraph::graph::DiGraph::<(), ()>::new(),
            PlotDecorations {
                node_highlights: &[],
                edge_highlights: &[],
                hairpin_highlights: &[],
                cell_highlights: &[],
                lowlights: &[],
                node_lowlights: &[],
            },
            &current_theme(),
            LayoutMode::Curved,
        );
        assert_eq!(renderer.rendered.get(), 1);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use petgraph::graph::NodeIndex;
    use ratatui::{buffer::Buffer, layout::Rect, style::Color};

    use super::{
        ARROW_GAPS, ArrowLayout, ArrowObstacles, CURVED_ARROW_GAPS, LineStyle,
        draw_arrows as draw_route_arrows, draw_rectilinear_segment, edge_route,
    };
    use crate::{
        edge_ports::{EdgePorts, NodeSide, PortedEdge},
        geometry::{LocalPos, WorldPos},
        layout::{LayoutNode, NodeRole, PinSide},
        layout_mode::LayoutMode,
        overlay_primitive::OverlayPrimitive,
        routed_window_graph::RoutedWindowGraph,
        world_buffer::{WorldBuffer, WorldViewport},
    };

    fn ported_edge(
        source: NodeIndex,
        target: NodeIndex,
        ports: EdgePorts,
    ) -> PortedEdge<NodeIndex> {
        PortedEdge {
            source,
            target,
            ports,
        }
    }

    /// Draw the arrows of one domain edge, with the route traced and obstacles gathered the way
    /// painting does for the whole graph.
    fn draw_arrows(
        buffer: &mut WorldBuffer,
        graph: &RoutedWindowGraph,
        edge: &PortedEdge<NodeIndex>,
        gaps: i64,
        style: (
            LayoutMode,
            &HashMap<WorldPos, crate::geometry::WorldRect>,
            Color,
        ),
        overlay_primitives: &mut Vec<OverlayPrimitive>,
    ) {
        let route = edge_route(graph, edge);
        let routes = [route.clone()];
        let obstacles = ArrowObstacles::new(graph, &routes);
        if let Some(route) = route {
            draw_route_arrows(
                buffer,
                graph,
                &route,
                gaps,
                style,
                &obstacles,
                overlay_primitives,
            );
        }
    }

    fn routed_edge(points: &[WorldPos]) -> (RoutedWindowGraph, NodeIndex, NodeIndex) {
        let source = NodeIndex::new(0);
        let target = NodeIndex::new(usize::from(points.first() != points.last()));
        let mut graph = RoutedWindowGraph::empty();
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

    fn rendered_arrows(
        points: &[WorldPos],
        world_viewport: &WorldViewport,
    ) -> Vec<(WorldPos, char)> {
        let (graph, source, target) = routed_edge(points);
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, world_viewport);
        for (start, end, _) in graph.edges() {
            draw_rectilinear_segment(&mut world_buffer, start, end, Color::Red, LineStyle::Bold);
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
            &ported_edge(source, target, ports),
            ARROW_GAPS,
            (LayoutMode::Rectilinear, &HashMap::new(), Color::Reset),
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

    /// The arrows a westward curved route along one row carries when `end_nodes` stand at its
    /// two ends, the western one first. Eastward runs carry none (see
    /// `test_curved_eastward_run_has_no_arrow`).
    fn curved_arrows_between(end_nodes: [NodeRole; 2], length: i64) -> usize {
        let (mut graph, source, target) =
            routed_edge(&[WorldPos::new(length, 0), WorldPos::new(0, 0)]);
        for (position, role) in [WorldPos::new(0, 0), WorldPos::new(length, 0)]
            .into_iter()
            .zip(end_nodes)
        {
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::new(role, LocalPos::new_xy(position.x, position.y), (1, 1), None),
            );
        }
        let world_viewport = viewport();
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        let mut overlay_primitives = Vec::new();
        draw_arrows(
            &mut world_buffer,
            &graph,
            &ported_edge(
                source,
                target,
                EdgePorts::new(NodeSide::West, NodeSide::East),
            ),
            CURVED_ARROW_GAPS,
            (LayoutMode::Curved, &HashMap::new(), Color::Reset),
            &mut overlay_primitives,
        );
        overlay_primitives
            .iter()
            .filter(|primitive| matches!(primitive, OverlayPrimitive::Arrow(_)))
            .count()
    }

    #[test]
    fn test_curved_eastward_run_has_no_arrow() {
        let (mut graph, source, target) = routed_edge(&[WorldPos::new(0, 0), WorldPos::new(30, 0)]);
        for position in [WorldPos::new(0, 0), WorldPos::new(30, 0)] {
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::routing(LocalPos::new_xy(position.x, position.y), (1, 1)),
            );
        }
        let world_viewport = viewport();
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        let mut overlay_primitives = Vec::new();
        draw_arrows(
            &mut world_buffer,
            &graph,
            &ported_edge(source, target, EdgePorts::default()),
            CURVED_ARROW_GAPS,
            (LayoutMode::Curved, &HashMap::new(), Color::Reset),
            &mut overlay_primitives,
        );
        assert!(overlay_primitives.is_empty());
    }

    #[test]
    fn test_short_curved_run_between_routing_nodes_keeps_its_arrow() {
        assert_eq!(
            curved_arrows_between([NodeRole::Routing, NodeRole::Routing], 3),
            1
        );
    }

    #[test]
    fn test_short_curved_run_at_a_data_node_has_no_arrow() {
        let data = |index| NodeRole::Data(NodeIndex::new(index));
        assert_eq!(curved_arrows_between([data(0), NodeRole::Routing], 3), 0);
        assert_eq!(curved_arrows_between([NodeRole::Routing, data(1)], 3), 0);
        assert_eq!(curved_arrows_between([data(0), data(1)], 3), 0);
    }

    #[test]
    fn test_long_curved_run_at_a_data_node_keeps_its_arrow() {
        let data = |index| NodeRole::Data(NodeIndex::new(index));
        assert_eq!(curved_arrows_between([data(0), data(1)], 12), 1);
    }

    #[test]
    fn test_curved_direction_markers_use_wider_repeat_spacing() {
        let (mut graph, source, target) = routed_edge(&[WorldPos::new(60, 0), WorldPos::new(0, 0)]);
        for position in [WorldPos::new(0, 0), WorldPos::new(60, 0)] {
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::routing(LocalPos::new_xy(position.x, position.y), (1, 1)),
            );
        }
        let world_viewport = viewport();
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        let mut overlay_primitives = Vec::new();
        draw_arrows(
            &mut world_buffer,
            &graph,
            &ported_edge(
                source,
                target,
                EdgePorts::new(NodeSide::West, NodeSide::East),
            ),
            CURVED_ARROW_GAPS,
            (LayoutMode::Curved, &HashMap::new(), Color::Reset),
            &mut overlay_primitives,
        );
        let centers: Vec<_> = overlay_primitives
            .iter()
            .filter_map(|primitive| match primitive {
                OverlayPrimitive::Arrow(mark) => Some(mark.center),
                _ => None,
            })
            .collect();
        let expected: Vec<_> = [10, 30, 50]
            .into_iter()
            .map(|x| world_buffer.world_to_terminal_continuous((x as f64 + 0.5, 0.5)))
            .collect();
        assert_eq!(centers, expected);
    }

    fn viewport() -> WorldViewport {
        WorldViewport {
            viewport_bounds: Rect::new(0, 0, 100, 80),
            ..WorldViewport::default()
        }
    }

    #[test]
    fn test_rectilinear_arrows_preserve_corners_and_junctions() {
        let points = [WorldPos::new(8, 0), WorldPos::new(0, 0)];
        let (graph, source, target) = routed_edge(&points);
        let world_viewport = viewport();
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        draw_rectilinear_segment(
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
            &ported_edge(
                source,
                target,
                EdgePorts::new(NodeSide::West, NodeSide::East),
            ),
            1,
            (LayoutMode::Rectilinear, &HashMap::new(), Color::Reset),
            &mut Vec::new(),
        );
        for (x, glyph) in [(2, '╮'), (4, '┼'), (6, '┬')] {
            assert_eq!(world_buffer.get_char(WorldPos::new(x, 0)), Some(glyph));
        }
        assert!((0..=8).any(|x| world_buffer.get_char(WorldPos::new(x, 0)) == Some('◀')));
    }

    #[test]
    fn test_arrows_on_a_run_millions_of_cells_long_stay_on_screen() {
        let (mut graph, source, target) =
            routed_edge(&[WorldPos::new(5_000_000, 0), WorldPos::new(-5_000_000, 0)]);
        for position in [WorldPos::new(5_000_000, 0), WorldPos::new(-5_000_000, 0)] {
            graph.node_data_by_pos.insert(
                position,
                LayoutNode::routing(LocalPos::new_xy(position.x, position.y), (1, 1)),
            );
        }
        let world_viewport = viewport();
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        let mut overlay_primitives = Vec::new();
        draw_arrows(
            &mut world_buffer,
            &graph,
            &ported_edge(
                source,
                target,
                EdgePorts::new(NodeSide::West, NodeSide::East),
            ),
            CURVED_ARROW_GAPS,
            (LayoutMode::Curved, &HashMap::new(), Color::Reset),
            &mut overlay_primitives,
        );
        let visible = world_buffer.visible_world_area();
        let width = visible.max.x - visible.min.x + 1;
        let arrows = overlay_primitives
            .iter()
            .filter(|primitive| matches!(primitive, OverlayPrimitive::Arrow(_)))
            .count() as i64;
        assert!(arrows >= 1 && arrows <= width / CURVED_ARROW_GAPS + 1);
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
        let world_viewport = viewport();
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        draw_arrows(
            &mut world_buffer,
            &graph,
            &ported_edge(
                source,
                target,
                EdgePorts::new(NodeSide::West, NodeSide::East),
            ),
            ARROW_GAPS,
            (LayoutMode::Curved, &HashMap::new(), Color::Reset),
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
    fn loopback_route(
        returning_x: i64,
    ) -> (RoutedWindowGraph, NodeIndex, NodeIndex, [WorldPos; 3]) {
        let anchor = WorldPos::new(10, 0);
        let pin = WorldPos::new(30, 0);
        let returning = WorldPos::new(returning_x, -10);
        let source = NodeIndex::new(0);
        let target = NodeIndex::new(1);
        let mut graph = RoutedWindowGraph::empty();
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
    fn test_loopback_route_reports_three_arrows() {
        let (graph, source, target, _) = loopback_route(-20);
        let world_viewport = WorldViewport {
            camera_current: WorldPos::new(5, -5),
            viewport_bounds: Rect::new(0, 0, 55, 16),
            ..WorldViewport::default()
        };
        let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
        let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
        let mut overlay_primitives = Vec::new();
        draw_arrows(
            &mut world_buffer,
            &graph,
            &ported_edge(
                source,
                target,
                EdgePorts::new(NodeSide::East, NodeSide::East),
            ),
            ARROW_GAPS,
            (LayoutMode::Curved, &HashMap::new(), Color::Red),
            &mut overlay_primitives,
        );
        let arrows = overlay_primitives
            .iter()
            .filter(|primitive| matches!(primitive, OverlayPrimitive::Arrow(_)))
            .count();
        assert_eq!(arrows, 3);
    }

    #[test]
    fn test_edge_route_follows_only_its_bundle() {
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
            edge_route(&graph, &ported_edge(source, target, EdgePorts::default())),
            Some(points.to_vec())
        );
    }

    #[test]
    fn test_edge_route_picks_the_route_on_its_sides() {
        // Two parallel edges between the same nodes: one leaves and arrives on the east sides
        // through a pin to the right, the other on the west sides through a pin to the left.
        let source = NodeIndex::new(0);
        let target = NodeIndex::new(1);
        let (source_pos, target_pos) = (WorldPos::new(0, 0), WorldPos::new(20, 0));
        let (left, right) = (WorldPos::new(-10, 0), WorldPos::new(30, 5));
        let mut graph = RoutedWindowGraph::empty();
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
            edge_route(&graph, &ported_edge(source, target, east_east)),
            Some(vec![source_pos, right, target_pos])
        );
        assert_eq!(
            edge_route(&graph, &ported_edge(source, target, west_west)),
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
    fn test_arrow_layout_rounds_toward_the_same_physical_side() {
        // Twelve free cells leave an odd cell over: the arrow goes on cell 5 from the west,
        // whichever way the run travels.
        let slots = |free_count| {
            ArrowLayout::new(free_count, 10)
                .slots_between(0, free_count - 1)
                .collect::<Vec<_>>()
        };
        assert_eq!(slots(12), vec![5]);
        assert_eq!(slots(25), vec![7, 17]);
    }

    #[test]
    fn test_arrow_layout_visits_only_slots_in_range() {
        let layout = ArrowLayout::new(1_000_005, 10);
        let visible: Vec<_> = layout.slots_between(500_000, 500_030).collect();
        assert!(visible.len() <= 4);
        assert!(
            visible
                .iter()
                .all(|slot| (500_000..=500_030).contains(slot))
        );
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
        let small_viewport = WorldViewport {
            viewport_bounds: Rect::new(0, 0, 20, 10),
            ..WorldViewport::default()
        };
        let expected: Vec<_> = rendered_arrows(&points, &viewport())
            .into_iter()
            .filter(|(position, _)| small_viewport.world_to_terminal(*position).is_some())
            .collect();
        assert_eq!(rendered_arrows(&points, &small_viewport), expected);
        assert!(!expected.is_empty());
    }

    #[test]
    fn test_edge_route_traces_self_loop_from_outgoing_port() {
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
            edge_route(&graph, &ported_edge(source, target, EdgePorts::default())),
            Some(points.to_vec())
        );
        // Only the 40 cell westward run carries markers, four of them.
        assert_eq!(rendered_arrows(&points, &viewport()).len(), 4);
    }

    #[test]
    fn test_edge_route_does_not_decorate_disconnected_segments() {
        let points = [
            WorldPos::new(0, 0),
            WorldPos::new(20, 0),
            WorldPos::new(40, 0),
        ];
        let (mut graph, source, target) = routed_edge(&points);
        graph.graph.remove_edge(points[1], points[2]);
        assert!(edge_route(&graph, &ported_edge(source, target, EdgePorts::default())).is_none());
    }

    /// A node at the origin and a hairpin route across a hairpin node: the bus runs `bus` cells to a
    /// hub that feeds two flat edges, which a bend joins `width` cells further out.
    fn hairpin_route(
        bus: i64,
        width: i64,
    ) -> (RoutedWindowGraph, NodeIndex, [WorldPos; 4], WorldPos) {
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
        let mut graph = RoutedWindowGraph::empty();
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
            let route = edge_route(
                &graph,
                &ported_edge(source, source, EdgePorts::new(side, side)),
            );
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
        let route = edge_route(
            &graph,
            &ported_edge(
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
    fn test_hairpin_arrows_mark_only_the_leg_running_west() {
        // The leg that leaves eastward is implied; the one that returns westward gets an arrow.
        // The Bezier arc uses its own single bend arrow.
        for (bus, width, side) in [(4, 3, NodeSide::East), (-4, -3, NodeSide::West)] {
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
            let edge = ported_edge(source, source, EdgePorts::new(side, side));
            graph.domain_edges = vec![edge];
            let world_viewport = viewport();
            let mut buffer = Buffer::empty(world_viewport.viewport_bounds);
            let mut world_buffer = WorldBuffer::new(&mut buffer, &world_viewport);
            let mut overlay_primitives = Vec::new();
            draw_arrows(
                &mut world_buffer,
                &graph,
                &edge,
                CURVED_ARROW_GAPS,
                (LayoutMode::Curved, &HashMap::new(), Color::Red),
                &mut overlay_primitives,
            );

            let mut arrows: Vec<_> = overlay_primitives
                .iter()
                .filter_map(|edge| match edge {
                    OverlayPrimitive::Arrow(mark) => Some(*mark),
                    _ => None,
                })
                .collect();
            arrows.sort_by(|first, second| first.center.1.total_cmp(&second.center.1));
            let directions: Vec<bool> = arrows.iter().map(|mark| mark.direction.0 > 0.0).collect();
            assert_eq!(directions, vec![false]);
        }
    }
}
