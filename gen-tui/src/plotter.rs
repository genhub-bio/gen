// This module implements graph rendering using the ViewportGraph system.
// All legacy rendering paths have been removed in favor of the unified ViewportGraph approach.

use std::hash::Hash;

use petgraph::visit::{
    EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCount, NodeIndexable, Visitable,
};
use ratatui::style::{Color, Style};

use crate::{
    geometry::{BigRect, Point, WorldPos, WorldRect},
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::GraphWidget,
    layout::{JunctionSymbol, NodeRole, VisualDetail},
    theme::get_theme_color,
    viewport_graph::CroppedGraph,
};

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
/// - `path_highlights`: Path highlights with colors for emphasized rendering (later highlights take precedence)
pub fn plot_viewport_graph_with_highlights<R, G>(
    viewport_graph: &CroppedGraph,
    buffer: &mut WorldBuffer<'_>,
    renderer: &mut R,
    original_graph: &G,
    detail_level: VisualDetail,
    path_highlights: &[(
        petgraph::graphmap::DiGraphMap<petgraph::graph::NodeIndex, ()>,
        Color,
    )],
) where
    R: NodeRenderer<G>,
    G: GraphBase + NodeIndexable,
{
    // Project the highlight paths (consist of only data nodes) onto the rectilinear graph
    // so that we end up with subgraphs that have all the routing done.
    let projected_highlights: Vec<(CroppedGraph, Color)> = path_highlights
        .iter()
        .map(|(path_graph, color)| {
            let projected = viewport_graph.subgraph(|bundle| {
                bundle
                    .iter()
                    .any(|&(u, v)| path_graph.contains_edge(u, v) || path_graph.contains_edge(v, u))
            });
            (projected, *color)
        })
        .collect();

    // Draw edges first so nodes appear on top
    for (source, target, bundle) in viewport_graph.edges() {
        // Ommit the edges that don't actually represent an original edge
        // (edges to/from terminal source/sink nodes)
        if !bundle.is_empty() {
            // Check if edge is in any projected highlight
            // Later highlights in the vector take precedence over earlier ones
            let mut highlighted_color = None;

            for (graph, color) in &projected_highlights {
                if graph.graph.contains_edge(source, target) {
                    highlighted_color = Some(*color);
                }
            }

            if let Some(color) = highlighted_color {
                draw_bold_edge(buffer, source, target, color);
            } else {
                let color = get_theme_color("edge").unwrap_or(ratatui::style::Color::Gray);
                draw_edge(buffer, source, target, color);
            }
        }
    }

    // Draw nodes
    for (world_pos, node) in viewport_graph.nodes() {
        match &node.role {
            NodeRole::Data(domain_idx) => {
                let node_id = <G as NodeIndexable>::from_index(original_graph, domain_idx.index());
                let world_rect = WorldRect::from_center_and_size(*world_pos, node.size);
                renderer.render_node(buffer, world_rect, &node_id, detail_level);
            }
            NodeRole::Routing => {
                // By default, render the junction as it appears in the full graph
                let mut glyph = compute_junction_glyph(viewport_graph, *world_pos);
                let mut style = Style::default()
                    .fg(get_theme_color("edge").unwrap_or(ratatui::style::Color::Gray));
                let mut is_highlighted = false;

                // If this node is part of a highlighted path, overlay the bolded bends overtop.
                for (graph, color) in &projected_highlights {
                    if graph.contains_node(world_pos) {
                        glyph = compute_junction_glyph(graph, *world_pos);
                        style = Style::default().fg(*color);
                        is_highlighted = true;
                    }
                }

                let character = if is_highlighted {
                    glyph.heavy_glyph()
                } else {
                    glyph.glyph()
                };

                buffer.set_char_styled(*world_pos, character, style);
            }
            NodeRole::Stitch(_) => {
                // Stitch nodes should have been replaced by actual content in ViewportGraph
                // If we see one, it means there's no actual content at this position
                buffer.set_char(*world_pos, '◆');
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
/// optionally using heavy box-drawing characters.
fn draw_edge_with_weight(
    buffer: &mut WorldBuffer,
    source: WorldPos,
    target: WorldPos,
    color: Color,
    heavy: bool,
) {
    let style = Style::default().fg(color);

    let (v_ch, h_ch) = if heavy {
        ('┃', '━')
    } else {
        ('│', '─')
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
            // Vertical edges take priority at crossings (both normal and heavy).
            if !matches!(buffer.get_char(pos), Some('│') | Some('┃')) {
                buffer.set_char_styled(pos, h_ch, style);
            }
        }
    }
}

/// Draw a rectilinear edge using normal-weight box-drawing characters.
fn draw_edge(buffer: &mut WorldBuffer, source: WorldPos, target: WorldPos, color: Color) {
    draw_edge_with_weight(buffer, source, target, color, false);
}

/// Draw a rectilinear edge using heavy-weight box-drawing characters.
fn draw_bold_edge(buffer: &mut WorldBuffer, source: WorldPos, target: WorldPos, color: Color) {
    draw_edge_with_weight(buffer, source, target, color, true);
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
