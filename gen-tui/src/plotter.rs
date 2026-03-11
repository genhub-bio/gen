// This module implements graph rendering using the ViewportGraph system.
// All legacy rendering paths have been removed in favor of the unified ViewportGraph approach.

use std::{collections::HashSet, hash::Hash};

use petgraph::{
    graph::NodeIndex,
    visit::{
        EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeCount, NodeIndexable, Visitable,
    },
};
use ratatui::{
    style::{Color, Style},
    symbols::merge::MergeStrategy,
};

use crate::{
    color_utils::{brighten_colors, tint_colors},
    geometry::{BigRect, Point, WorldPos, WorldRect},
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::{GraphWidget, NODE_GLYPH},
    layout::{JunctionSymbol, NodeRole, VisualDetail},
    theme::Theme,
    viewport_graph::CroppedGraph,
};

/// Number of world-coordinate units between arrow markers on reversed edges.
const ARROW_GAPS: usize = 16;

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

// # Graph Rendering Architecture
//
// This module implements a three-layer graph rendering system that separates
// concerns between domain data, partitioning infrastructure, and visual layout.
//
// ## Three-Layer Architecture
//
// ### Layer 1: Domain Graph (Original Data)
// ```rust
// DiGraphMap<GraphNode, GraphEdge>  // GenGraph
// ```
// - **Purpose**: Original database domain data
// - **Contains**: `GraphNode { node_id, sequence_start, sequence_end, ... }`
// - **Responsibility**: Pure domain logic and data storage
// - **Used by**: Domain-specific operations, database queries, user interaction
//
// ### Layer 2: Partition System (Domain + Infrastructure + Management)
// ```rust
// PartitionTable<G> {
//     partitions: Vec<Partition<G>>,       // Collection of partition graphs
//     node_map: HashMap<G::NodeId, ...>,   // Cross-reference mapping
//     inter_partition_edges: HashMap<...>, // Cross-partition connections
// }
//
// // Each Partition contains:
// StableGraph<PartitionNode<G>, PartitionEdge<G>>
//
// enum PartitionNode<G> {
//     Data(G::NodeId),    // Wrapped domain node
//     LeftStitch,         // Partition boundary markers
//     RightStitch,
// }
// ```
// - **Purpose**: Partitioning system with domain data + infrastructure + management
// - **Contains**: Multiple partition graphs + stitching nodes + cross-partition mappings
// - **Responsibility**: Graph virtualization, memory management, coordinate stitching
// - **Used by**: Viewport management, layout algorithms, partition rendering
//
// ### Layer 3: Layout Graph (Positioning Only per Partition)
// ```rust
// StableGraph<LayoutNode, LayoutEdge>
//
// struct LayoutNode {
//     pos: LayoutPos,           // X, Y coordinates
//     size: (u64, u64),         // Width, height
//     role: NodeRole,           // Reference back to original graph via NodeIndex
// }
//
// enum NodeRole {
//     Data(NodeIndex<u32>),    // NodeIndex from original graph (Layer 1)
//                              // - GraphMap: converts to struct key (e.g., GraphNode)
//                              // - DiGraph: converts to NodeIndex directly
//     Routing(GlyphIndex),     // Layout-only routing nodes for edge routing
// }
// ```
// - **Purpose**: Pure visual positioning and layout coordinates
// - **Contains**: Coordinates, sizes, and direct references back to original graph
// - **Responsibility**: Spatial arrangement and rendering coordinates
// - **Used by**: Rendering engine, viewport calculations, hit testing
//
// ## Data Flow
//
// ```text
// Domain Graph (Layer 1)
//     ↓ [Partitioning Process]
// Partition System (Layer 2)
//     ↓ [Layout Algorithm per Partition]
// Layout Graphs (Layer 3)
//     ↓ [Rendering Pipeline]
// Visual Output
// ```
//
// ## Rendering Bridge
//
// The plot functions bridge these layers:
// 1. **Layout Graph (Layer 3)**: Provides positioning (`LayoutNode.pos`, `LayoutNode.size`)
// 2. **Partition Graph (Layer 3)**: Provides original graph references (`NodeRole::Data(original_node_index)`)
// 3. **Domain Graph (Layer 1)**: Converts NodeIndex to NodeId via `NodeIndexable::from_index()`
//    - For GraphMap (e.g., GenGraph): NodeId is the struct key (GraphNode)
//    - For DiGraph: NodeId is the NodeIndex itself
// 4. **Node Renderer**: Transforms domain data into visual representation
//
// This separation allows:
// - **Domain-agnostic rendering**: Layout and rendering logic doesn't know about sequences
// - **Flexible partitioning**: Can add infrastructure nodes without affecting domain logic
// - **Reusable components**: Same rendering engine works for different domain viewers
// - **Clean testing**: Each layer can be tested independently

/// Traits for domain-specific node rendering within the three-layer architecture.
///
/// This trait enables domain-specific viewers (like `GenGraphViewer`) to customize
/// how nodes are visually rendered while keeping the core graph rendering logic
/// completely domain-agnostic.
///
/// # Architecture Role
/// The `NodeSizer` and `NodeRenderer` sit at the boundary between the generic rendering
/// engine and domain-specific visualization logic:
/// - **Generic Engine**: Handles layout, viewport culling, coordinate transformation
/// - **Domain Renderer**: Interprets node data and produces visual representation
/// - **Domain Sizer**: Interprets node data and informs visual representation
///
/// # Data Flow
/// 1. `plot_graph` finds visible nodes and extracts positioning (Layer 3)
/// 2. `plot_graph` converts NodeIndex to NodeId using original graph (Layer 1)
/// 3. `NodeRenderer::render_node` transforms domain data into pixels (Domain-specific)
///
/// The `NodeKey` type specifies how nodes are identified in that context.
/// This could be a NodeIndex, node weight, or a combination etc.
/// Trait for computing node label dimensions at different levels of detail
pub trait NodeSizer<G>
where
    G: GraphBase,
{
    /// Get the dimensions (width, height) for a node at a specific level of detail
    fn get_node_size(&self, node: &G::NodeId, detail_level: VisualDetail) -> (u64, u64);

    /// Get default dimensions for dummy/routing nodes
    fn get_dummy_size(&self) -> (u64, u64) {
        (1, 1)
    }
}

/// Blanket implementation for Box<T> where T implements NodeSizer
impl<G, T> NodeSizer<G> for Box<T>
where
    G: GraphBase,
    T: NodeSizer<G> + ?Sized,
{
    fn get_node_size(&self, node: &G::NodeId, detail_level: VisualDetail) -> (u64, u64) {
        (**self).get_node_size(node, detail_level)
    }

    fn get_dummy_size(&self) -> (u64, u64) {
        (**self).get_dummy_size()
    }
}

pub trait NodeRenderer<G>
where
    G: GraphBase,
{
    /// Render a single node's visual representation.
    ///
    /// # Parameters
    /// - `buffer`: World coordinate buffer writer for drawing to the viewport
    /// - `area`: Screen rectangle allocated for this node (from layout graph)
    /// - `node_id`: NodeId of the node in the original graph
    /// - `detail_level`: Level of detail for rendering (matching NodeSizer trait)
    fn render_node(
        &mut self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        node_id: &G::NodeId,
        detail_level: VisualDetail,
    );
}

/// Plot a single layout (single partition) with a position offset
///
/// This is the main plotting function that bridges:
/// - **Layout Graph** (Layer 3): Provides spatial positioning via `layout`
/// - **Original Graph** (Layer 1): Provides NodeId conversion via `graph`
/// - **Domain Renderer**: Transforms data into visual representation via `renderer`
///
/// # Parameters
/// - `viewport_graph`: CroppedGraph containing only visible nodes and edges
/// - `buffer`: World coordinate buffer writer for drawing to the viewport
/// - `renderer`: Domain-specific rendering logic and data lookup
/// - `original_graph`: Original graph for NodeIndex to NodeId conversion
/// - `detail_level`: Level of detail for rendering
pub fn plot_viewport_graph<R, G>(
    viewport_graph: &CroppedGraph,
    buffer: &mut WorldBuffer<'_>,
    renderer: &mut R,
    original_graph: &G,
    detail_level: VisualDetail,
    theme: &Theme,
) where
    R: NodeRenderer<G>,
    G: GraphBase + NodeIndexable,
{
    plot_viewport_graph_with_highlights(
        viewport_graph,
        buffer,
        renderer,
        original_graph,
        detail_level,
        &[],
        &[],
        theme,
    )
}

/// Plot a single layout with path highlights
///
/// This is the main plotting function that bridges:
/// - **Layout Graph** (Layer 3): Provides spatial positioning via `layout`
/// - **Original Graph** (Layer 1): Provides NodeId conversion via `graph`
/// - **Domain Renderer**: Transforms data into visual representation via `renderer`
///
/// # Parameters
/// - `viewport_graph`: CroppedGraph containing only visible nodes and edges
/// - `buffer`: World coordinate buffer writer for drawing to the viewport
/// - `renderer`: Domain-specific rendering logic and data lookup
/// - `original_graph`: Original graph for NodeIndex to NodeId conversion
/// - `detail_level`: Level of detail for rendering
/// - `node_highlights`: List of node positions to highlight with their styles
/// - `edge_highlights`: List of edge segments to highlight with their styles
/// - `theme`: Theme colors for rendering
#[allow(clippy::too_many_arguments)]
pub fn plot_viewport_graph_with_highlights<R, G>(
    viewport_graph: &CroppedGraph,
    buffer: &mut WorldBuffer<'_>,
    renderer: &mut R,
    original_graph: &G,
    detail_level: VisualDetail,
    node_highlights: &[(WorldPos, PathStyle)],
    edge_highlights: &[((WorldPos, WorldPos), PathStyle)],
    theme: &Theme,
) where
    R: NodeRenderer<G>,
    G: GraphBase + NodeIndexable,
{
    // Draw edges first so nodes appear on top
    for (source, target, bundle) in viewport_graph.edges() {
        // Check if edge is in any highlighted path
        // Later highlights in the vector take precedence over earlier ones
        let highlighted_style = edge_highlights
            .iter()
            .filter(|((s, t), _)| (*s == source && *t == target) || (*s == target && *t == source))
            .map(|(_, style)| *style)
            .next_back();

        if let Some(style) = highlighted_style {
            let edge_color = match style.color {
                Color::Reset => {
                    // Brighten the edge color instead of tinting
                    let (brightened, _) = brighten_colors(theme.edge_fg, theme.edge_fg, 0.2);
                    brightened
                }
                _ => {
                    // Compute tinted colors as they would be applied to nodes
                    let (_, tinted_bg) =
                        tint_colors(theme.node_fg, theme.node_bg, style.color, 0.4);
                    // Use the tinted bg as the edge fg color
                    tinted_bg
                }
            };
            draw_edge_with_style(buffer, source, target, edge_color, style.line_style);
        } else if !bundle.is_empty() {
            // Normal edge with data
            let color = theme.edge_fg;
            draw_edge_with_style(buffer, source, target, color, LineStyle::Normal);
        } else {
            // Edges that don't actually represent an original edge
            // (edges to/from terminal source/sink nodes) - draw with dashed lines
            let color = theme.edge_fg;
            draw_edge_with_style(buffer, source, target, color, LineStyle::Dashed);
        }
    }

    // Draw nodes
    for (world_pos, node) in viewport_graph.nodes() {
        match &node.role {
            NodeRole::Data(domain_idx) => {
                let node_id = <G as NodeIndexable>::from_index(original_graph, domain_idx.index());
                let world_rect = WorldRect::from_center_and_size(*world_pos, node.size);
                renderer.render_node(buffer, world_rect, &node_id, detail_level);

                // Check if this node is highlighted
                let highlighted_style = node_highlights
                    .iter()
                    .filter(|(pos, _)| pos == world_pos)
                    .map(|(_, style)| *style)
                    .next_back();

                // If highlighted, either tint with color or brighten (for Color::Reset)
                if let Some(path_style) = highlighted_style {
                    for y in world_rect.min.y..=world_rect.max.y {
                        for x in world_rect.min.x..=world_rect.max.x {
                            let pos = WorldPos::new(x, y);
                            if let Some((ch, style)) = buffer.get_char_styled(pos) {
                                let fg = style.fg.unwrap_or(Color::Reset);
                                let bg = style.bg.unwrap_or(Color::Reset);
                                let (new_fg, new_bg) = if ch == NODE_GLYPH {
                                    let new_fg = match path_style.color {
                                        Color::Reset => brighten_colors(fg, bg, 0.2).0,
                                        color => color,
                                    };
                                    (new_fg, bg)
                                } else {
                                    match path_style.color {
                                        Color::Reset => brighten_colors(fg, bg, 0.2),
                                        _ => tint_colors(fg, bg, path_style.color, 0.4),
                                    }
                                };
                                let new_style = style.fg(new_fg).bg(new_bg);
                                buffer.set_char_styled(pos, ch, new_style);
                            }
                        }
                    }
                }
            }
            NodeRole::Routing => {
                let edge_color = theme.edge_fg;
                let base_glyph = compute_junction_glyph(viewport_graph, *world_pos);

                // Check if this routing node is part of any highlighted edge
                let highlighted_style = edge_highlights
                    .iter()
                    .filter(|((s, t), _)| {
                        // A routing node is highlighted if it lies on a highlighted edge segment
                        // Since edges are either horizontal or vertical, we can check if the point lies between endpoints
                        if s.x == t.x {
                            // Vertical edge
                            world_pos.x == s.x
                                && world_pos.y >= s.y.min(t.y)
                                && world_pos.y <= s.y.max(t.y)
                        } else {
                            // Horizontal edge
                            world_pos.y == s.y
                                && world_pos.x >= s.x.min(t.x)
                                && world_pos.x <= s.x.max(t.x)
                        }
                    })
                    .map(|(_, style)| *style)
                    .next_back();

                let character = if let Some(style) = highlighted_style {
                    // Create a temporary graph for the active highlight style
                    // to compute the correct glyph
                    let active_edges: Vec<_> = edge_highlights
                        .iter()
                        .filter(|(_, s)| *s == style)
                        .cloned()
                        .collect();
                    let highlight_graph = CroppedGraph::from_visual_edges(&active_edges);
                    let high_glyph = compute_junction_glyph(&highlight_graph, *world_pos);

                    // Decision tree:
                    // 1. Is this routing node part of any highlighted path?
                    // 2. If it is, do we merge with the current glyph or replace it?
                    //    - Merge if merge_glyphs is true
                    //    - Replace if merge_glyphs is false (avoids spiney artefacts with color tinting)
                    if style.merge_glyphs {
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
                        // Replace mode: use the highlight glyph directly
                        match style.line_style {
                            LineStyle::Normal => high_glyph.glyph(),
                            LineStyle::Bold => high_glyph.heavy_glyph(),
                            LineStyle::Dashed => high_glyph.dashed_glyph(),
                        }
                    }
                } else {
                    base_glyph.glyph()
                };

                let fg_color = match highlighted_style {
                    None => edge_color,
                    Some(style) => match style.color {
                        Color::Reset => {
                            // Brighten the edge color instead of tinting
                            let (brightened, _) = brighten_colors(edge_color, edge_color, 0.2);
                            brightened
                        }
                        _ => {
                            // Compute tinted colors as they would be applied to nodes
                            let (_, tinted_bg) =
                                tint_colors(theme.node_fg, theme.node_bg, style.color, 0.4);
                            // Use the tinted bg as the edge fg color
                            tinted_bg
                        }
                    },
                };
                let style = Style::default().fg(fg_color);
                buffer.set_char_styled(*world_pos, character, style);
            }
            NodeRole::Stitch(_) => {
                // Stitch nodes should have been replaced by actual content in ViewportGraph
                // If we see one, it means there's no actual content at this position
                buffer.set_char(*world_pos, '◆');
            }
        }
    }

    // Draw direction markers on reversed (backward) edges identified during cycle removal.
    let mut drawn_reversed: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
    for (_, _, bundle) in viewport_graph.edges() {
        for &(src, tgt) in bundle {
            if src == tgt || !drawn_reversed.insert((src, tgt)) {
                continue;
            }
            if viewport_graph.backward_edges.contains(&(src, tgt)) {
                draw_arrows(buffer, viewport_graph, src, tgt, ARROW_GAPS);
            }
        }
    }
}

/// Compute the junction glyph for a routing node based on its connections
fn compute_junction_glyph(viewport_graph: &CroppedGraph, pos: WorldPos) -> JunctionSymbol {
    let mut connections = 0u8;

    // Check connections in all four directions
    for neighbor in viewport_graph.neighbors(pos) {
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
        LineStyle::Dashed => ('┆', '┄'),
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
            if !matches!(buffer.get_char(pos), Some('│') | Some('┃') | Some('┆')) {
                buffer.set_char_styled(pos, h_ch, style);
            }
        }
    }
}

/// Draw direction marker arrows along the visual path of an edge.
///
/// Finds the visual path for the domain edge `(source, target)` by extracting the
/// subgraph of segments whose bundles contain that pair, then walks the path from the
/// Place directional arrow markers on every segment of a reversed edge.
///
/// Arrow placement uses a diagonal grid: a marker is placed at every position where
/// `(pos.x + pos.y).rem_euclid(gaps) == 0`, keeping markers aligned across partition
/// boundaries regardless of where each viewport window starts.
///
/// Direction is determined by walking the edge's subgraph from a known anchor node:
/// - If source is visible, walk forward from source.
/// - If only target is visible, walk backward from target.
/// - Otherwise, walk from the rightmost degree-1 node (assumed source side).
///   Segments unreachable from any anchor fall back to geometry:
/// - Vertical: above y=0 → `▼` (facing down toward nodes), below y=0 → `▲`.
/// - Horizontal: `◀` (backward edges always go right-to-left).
///
/// Only cells already containing a straight line character are overwritten;
/// the existing style is preserved.
pub fn draw_arrows(
    buffer: &mut WorldBuffer,
    viewport_graph: &CroppedGraph,
    source: NodeIndex,
    target: NodeIndex,
    gaps: usize,
) {
    if gaps == 0 || source == target {
        return;
    }
    let g = gaps as i64;

    // The subgraph already contains exactly the segments labelled with (source, target).
    // No search needed — we just walk this pre-filtered graph from an anchor.
    let sub = viewport_graph.subgraph(|bundle| bundle.contains(&(source, target)));
    if sub.graph.node_count() == 0 {
        return;
    }

    // Find world position of the source or target domain node within the subgraph.
    let source_pos = sub
        .node_data_by_pos
        .iter()
        .find(|(_, n)| matches!(n.role, NodeRole::Data(idx) if idx == source))
        .map(|(pos, _)| *pos);
    let target_pos = sub
        .node_data_by_pos
        .iter()
        .find(|(_, n)| matches!(n.role, NodeRole::Data(idx) if idx == target))
        .map(|(pos, _)| *pos);

    // Choose walk anchor and direction. Prefer source (forward), else target (backward),
    // else the rightmost degree-1 node (assumed source side for a backward edge).
    let (start, forward) = if let Some(p) = source_pos {
        (p, true)
    } else if let Some(p) = target_pos {
        (p, false)
    } else {
        let endpoint = sub
            .graph
            .nodes()
            .filter(|&p| sub.graph.neighbors(p).count() == 1)
            .max_by_key(|p| p.x)
            .or_else(|| sub.graph.nodes().next());
        match endpoint {
            Some(p) => (p, true),
            None => return,
        }
    };

    // Walk from `start` through the subgraph, recording directed (from, to) per edge.
    // At each junction we simply continue to every unvisited neighbor — no search required.
    // Key: normalised (lo, hi); value: directed (from, to).
    let mut directed: Vec<((WorldPos, WorldPos), (WorldPos, WorldPos))> = Vec::new();
    let mut visited: HashSet<WorldPos> = HashSet::new();
    let mut stack: Vec<WorldPos> = vec![start];
    visited.insert(start);

    while let Some(cur) = stack.pop() {
        for next in sub.graph.neighbors(cur) {
            let key = if cur <= next {
                (cur, next)
            } else {
                (next, cur)
            };
            if directed.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let (from, to) = if forward { (cur, next) } else { (next, cur) };
            directed.push((key, (from, to)));
            if !visited.contains(&next) {
                visited.insert(next);
                stack.push(next);
            }
        }
    }

    // Offset the diagonal grid so that the outer markers on the longest segment are
    // equidistant from their respective endpoints.  For a segment of length L with
    // spacing g, the unused remainder is L % g; splitting that equally gives a margin
    // of m = (L % g) / 2 at each end, so the first marker lands at lo + m.
    let grid_offset = {
        let mut best_len: i64 = -1;
        let mut offset: i64 = 0;
        let mut seen_len: HashSet<(WorldPos, WorldPos)> = HashSet::new();
        for (seg_a, seg_b, _) in sub.edges() {
            let key = if seg_a <= seg_b {
                (seg_a, seg_b)
            } else {
                (seg_b, seg_a)
            };
            if !seen_len.insert(key) {
                continue;
            }
            let (lo, hi) = key;
            let len = if lo.x == hi.x {
                hi.y - lo.y
            } else {
                hi.x - lo.x
            };
            if len > best_len {
                best_len = len;
                let margin = len.rem_euclid(g) / 2;
                offset = (lo.x + lo.y + margin).rem_euclid(g);
            }
        }
        offset
    };

    // Place markers on every segment in the subgraph.
    let mut seen: HashSet<(WorldPos, WorldPos)> = HashSet::new();
    for (seg_a, seg_b, _) in sub.edges() {
        let key = if seg_a <= seg_b {
            (seg_a, seg_b)
        } else {
            (seg_b, seg_a)
        };
        if !seen.insert(key) {
            continue;
        }
        let (lo, hi) = key;

        let arrow_ch = if let Some((_, (from, to))) = directed.iter().find(|(k, _)| *k == key) {
            let (from, to) = (*from, *to);
            if to.x > from.x {
                '▶'
            } else if to.x < from.x {
                '◀'
            } else if to.y > from.y {
                '▲' // world y increases upward → ▲
            } else {
                '▼'
            }
        } else if lo.x == hi.x {
            let mid_y = (lo.y + hi.y) / 2;
            if mid_y >= 0 { '▼' } else { '▲' }
        } else {
            '◀'
        };

        if lo.x == hi.x {
            for y in lo.y..=hi.y {
                let pos = WorldPos::new(lo.x, y);
                if (pos.x + pos.y - grid_offset).rem_euclid(g) == 0
                    && matches!(buffer.get_char(pos), Some('│') | Some('┃') | Some('┆'))
                    && let Some((_, style)) = buffer.get_char_styled(pos)
                {
                    buffer.set_char_styled(pos, arrow_ch, style);
                }
            }
        } else {
            for x in lo.x..=hi.x {
                let pos = WorldPos::new(x, lo.y);
                if (pos.x + pos.y - grid_offset).rem_euclid(g) == 0
                    && matches!(buffer.get_char(pos), Some('─') | Some('━') | Some('┄'))
                    && let Some((_, style)) = buffer.get_char_styled(pos)
                {
                    buffer.set_char_styled(pos, arrow_ch, style);
                }
            }
        }
    }
}

/// Render any graph widget to string representation using TestBackend
///
/// This is a domain-agnostic function that can work with any graph type and custom renderers.
/// It replaces the functionality previously in GenGraphViewer::plot_to_string().
pub fn plot_graph_to_string<G, S, R>(
    controller: &mut GraphController<G, S>,
    renderer: R,
    detail_level: Option<VisualDetail>,
    offset: Option<(i64, i64)>,
    size: Option<(i64, i64)>,
) -> Result<(String, u16, u16), String>
where
    G: GraphBase
        + Clone
        + EdgeIndexable
        + NodeIndexable
        + NodeCount
        + Visitable
        + IntoNodeIdentifiers
        + IntoEdgeReferences
        + IntoNeighborsDirected,
    G::NodeId: Copy + Eq + Hash + Ord,
    G::EdgeId: Clone,
    S: NodeSizer<G>,
    R: NodeRenderer<G>,
{
    use ratatui::{Terminal, backend::TestBackend};

    if let Some(s) = detail_level {
        controller.set_detail_level(s);
    }

    // Get the bounding box of the entire graph
    // (this will trigger loading of all partitions)
    let bbox = controller.calculate_total_bounds()?;

    // Crop to a camera of at most u16::MAX x u16::MAX
    let mut crop_to = if let Some(size) = size {
        BigRect::from_coords(0, 0, size.0 + 1, size.1 + 1)
    } else {
        BigRect::from_corners(
            Point::new(0, 0),
            Point::new(u16::MAX as i64, u16::MAX as i64),
        )
    };

    // Place the camera in the bottom left corner of the bbox,
    // and apply an offset if provided
    if let Some(offset) = offset {
        crop_to = crop_to.transform(|pos| pos + bbox.center() + offset);
    } else {
        crop_to = crop_to.transform(|pos| pos + bbox.center());
    };

    // Perform the actual cropping
    let intersection = bbox
        .intersection(&crop_to)
        .ok_or("Plot offset out of bounds")?;
    let width = intersection.width() as u16;
    let height = intersection.height() as u16;

    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).map_err(|e| e.to_string())?;

    // TODO: instead of using the center, use the origin of the partition
    // that currently holds the camera center
    let camera_center = intersection.center();
    controller.viewport_state.camera_current = camera_center;
    controller.viewport_state.camera_target = camera_center;
    controller.viewport_state.viewport_bounds =
        ratatui::layout::Rect::new(0, 0, width + 5, height + 5);
    controller.viewport_state.focus();

    // Render using the domain-agnostic graph widget
    let result = terminal.draw(|f| {
        let area = f.area();

        // Update viewport bounds to match area
        controller.viewport_state.viewport_bounds = area;

        let current_detail_level = controller.get_detail_level();

        // Create the domain-agnostic graph widget
        let widget = GraphWidget::with_renderer(renderer).detail_level(current_detail_level);

        f.render_stateful_widget(widget, area, controller);
    });

    result.map_err(|e| e.to_string())?;

    // Convert buffer to string
    let buffer = terminal.backend().buffer();
    let mut result = String::new();

    for y in 0..buffer.area().height {
        for x in 0..buffer.area().width {
            let cell = &buffer[(x, y)];
            result.push(cell.symbol().chars().next().unwrap_or(' '));
        }
        if y < buffer.area().height - 1 {
            result.push('\n');
        }
    }

    Ok((result, width, height))
}
