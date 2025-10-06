// This module implements graph rendering using the ViewportGraph system.
// All legacy rendering paths have been removed in favor of the unified ViewportGraph approach.

use std::hash::Hash;

use petgraph::visit::{
    EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCount, NodeIndexable, Visitable,
};
use ratatui::style::Style;

use crate::{
    geometry::{BigRect, Point, WorldPos, WorldRect},
    graph_controller::{GraphController, WorldBuffer},
    graph_widget::GraphWidget,
    layout::{JunctionSymbol, NodeRole, VisualDetail},
    theme::get_theme_color,
    viewport_graph::ViewportGraph,
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
/// - `layout`: Layer 3 positioning information (LayoutNode with coordinates)
/// - `buffer`: World coordinate buffer writer for drawing to the viewport
/// - `renderer`: Domain-specific rendering logic and data lookup
/// - `offset`: Global positioning offset used to stitch chunks of the world together
/// - `graph`: Original graph for NodeIndex to NodeId conversion
///
/// The ViewportGraph contains only visible nodes and edges
pub fn plot_viewport_graph<R, G>(
    viewport_graph: &ViewportGraph,
    buffer: &mut WorldBuffer<'_>,
    renderer: &mut R,
    original_graph: &G,
    detail_level: VisualDetail,
) where
    R: NodeRenderer<G>,
    G: GraphBase + NodeIndexable,
{
    // Draw edges first so nodes appear on top
    for (source, target, bundle) in viewport_graph.edges() {
        // Ommit the edges that don't actually represent an original edge
        // (edges to/from terminal source/sink nodes)
        if !bundle.is_empty() {
            draw_edge(buffer, source, target);
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
                let glyph = compute_junction_glyph(viewport_graph, *world_pos);
                buffer.set_char(*world_pos, glyph.glyph());
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
fn compute_junction_glyph(viewport_graph: &ViewportGraph, pos: WorldPos) -> JunctionSymbol {
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

/// Draw a rectilinear edge between two world positions
fn draw_edge(buffer: &mut WorldBuffer, source: WorldPos, target: WorldPos) {
    let style = Style::default().fg(get_theme_color("edge").unwrap_or(ratatui::style::Color::Gray));

    if source.x == target.x {
        // Vertical edge
        let (min_y, max_y) = if source.y < target.y {
            (source.y, target.y)
        } else {
            (target.y, source.y)
        };
        for y in min_y..=max_y {
            buffer.set_char_styled(WorldPos::new(source.x, y), '│', style);
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
            // Don't overwrite a vertical line with a horizontal one
            // Vertical edges take priority at crossings
            if buffer.get_char(pos) != Some('│') {
                buffer.set_char_styled(pos, '─', style);
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use petgraph::stable_graph::{NodeIndex, StableDiGraph};
    use ratatui::{Terminal, backend::TestBackend};

    use super::*;
    use crate::graph_controller::*;

    // Mock domain graph type - use StableDiGraph to match LayoutEngine
    type MockGraph = StableDiGraph<(), ()>;

    // Helper to create a mock domain graph with nodes 0-7
    fn create_mock_domain_graph() -> MockGraph {
        let mut graph = MockGraph::new();
        // Add 8 nodes (0-7) to match the complex_dag test
        for _i in 0..8 {
            graph.add_node(());
        }
        graph
    }

    // Mock renderer that draws node index
    struct MockRenderer;

    impl NodeRenderer<MockGraph> for MockRenderer {
        fn render_node(
            &mut self,
            buffer: &mut WorldBuffer,
            area: WorldRect,
            node_id: &NodeIndex,
            _detail_level: VisualDetail,
        ) {
            use ratatui::style::{Color, Style};

            // Fill the entire node area with a solid background block
            let bg_style = Style::default().bg(Color::Blue).fg(Color::Blue);
            for y in area.min.y..=area.max.y {
                for x in area.min.x..=area.max.x {
                    buffer.set_char_styled(WorldPos::new(x, y), '█', bg_style);
                }
            }

            // Calculate centered position for the text
            let label = format!("N{}", node_id.index());
            let label_width = label.len() as i64;
            let area_width = area.max.x - area.min.x + 1;
            let area_height = area.max.y - area.min.y + 1;

            // Center horizontally (for odd widths, keep larger space on the left)
            let text_x = area.min.x + (area_width - label_width) / 2;
            // Center vertically
            let text_y = area.min.y + area_height / 2;

            // Render the centered text in a contrasting color
            let text_style = Style::default().bg(Color::Blue).fg(Color::White);
            buffer.set_string_styled(WorldPos::new(text_x, text_y), &label, text_style);
        }
    }

    // Helper to create a simple layout using LayoutEngine (proper pipeline)
    fn create_simple_layout() -> crate::layout::PartitionLayout {
        use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph};

        use crate::{
            layout::LayoutEngine,
            partition::{PartitionEdge, PartitionNode, StitchSide},
        };

        // Create a partition graph for the LayoutEngine with required stitch nodes
        let mut partition_graph = StableDiGraph::new();

        // Add stitch nodes first (as done by PartitionTable::new)
        let left_stitch = partition_graph.add_node(PartitionNode::Stitch(StitchSide::Left));
        let right_stitch = partition_graph.add_node(PartitionNode::Stitch(StitchSide::Right));

        // Add data nodes
        let n1 = partition_graph.add_node(PartitionNode::Data(NodeIndex::new(0)));
        let n2 = partition_graph.add_node(PartitionNode::Data(NodeIndex::new(1)));

        // Connect stitch nodes to data nodes (as done by PartitionTable::new)
        partition_graph.add_edge(left_stitch, n1, None);
        partition_graph.add_edge(n1, n2, Some((NodeIndex::new(0), NodeIndex::new(1))));
        partition_graph.add_edge(n2, right_stitch, None);

        struct TestNodeSizer;
        impl NodeSizer<&StableDiGraph<PartitionNode, PartitionEdge, u32>> for TestNodeSizer {
            fn get_node_size(
                &self,
                _node: &NodeIndex<u32>,
                _detail_level: VisualDetail,
            ) -> (u64, u64) {
                (10, 3)
            }
        }

        let detail_level = VisualDetail::Minimal;
        let node_sizer = TestNodeSizer;
        // Use LayoutEngine to compute layout with proper pipeline
        let mut engine = LayoutEngine::new(&partition_graph, 0);
        engine
            .compute_layout(&node_sizer, detail_level) // Default total_partitions for testing
            .expect("Layout computation failed")
    }

    // Helper to create a branched layout using LayoutEngine (proper pipeline)
    fn create_branched_layout() -> crate::layout::PartitionLayout {
        use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph};

        use crate::{
            layout::LayoutEngine,
            partition::{PartitionEdge, PartitionNode, StitchSide},
        };

        // Create a partition graph for the LayoutEngine with required stitch nodes
        let mut partition_graph = StableDiGraph::new();

        // Add stitch nodes first (as done by PartitionTable::new)
        let left_stitch = partition_graph.add_node(PartitionNode::Stitch(StitchSide::Left));
        let right_stitch = partition_graph.add_node(PartitionNode::Stitch(StitchSide::Right));

        // Add data nodes
        let n1 = partition_graph.add_node(PartitionNode::Data(NodeIndex::new(0)));
        let n2 = partition_graph.add_node(PartitionNode::Data(NodeIndex::new(1)));
        let n3 = partition_graph.add_node(PartitionNode::Data(NodeIndex::new(2)));

        // Connect stitch nodes and data nodes (as done by PartitionTable::new)
        partition_graph.add_edge(left_stitch, n1, None);
        partition_graph.add_edge(n1, n2, Some((NodeIndex::new(0), NodeIndex::new(1))));
        partition_graph.add_edge(n1, n3, Some((NodeIndex::new(0), NodeIndex::new(2))));
        partition_graph.add_edge(n2, right_stitch, None);
        partition_graph.add_edge(n3, right_stitch, None);

        struct TestNodeSizer;
        impl NodeSizer<&StableDiGraph<PartitionNode, PartitionEdge, u32>> for TestNodeSizer {
            fn get_node_size(
                &self,
                _node: &NodeIndex<u32>,
                _detail_level: VisualDetail,
            ) -> (u64, u64) {
                (10, 3)
            }
        }

        // Use LayoutEngine to compute layout with proper pipeline
        let mut engine = LayoutEngine::new(&partition_graph, 0);
        let node_sizer = TestNodeSizer;
        engine
            .compute_layout(&node_sizer, VisualDetail::Minimal) // Default total_partitions for testing
            .expect("Layout computation failed")
    }

    // Helper function for tests - disabled due to legacy rendering removal
    #[cfg(test)]
    #[allow(dead_code)]
    fn test_plot_layout<R, G>(
        _layout: &crate::layout::PartitionLayout,
        _buffer: &mut WorldBuffer<'_>,
        _renderer: &mut R,
        _offset: (i64, i64),
        _graph: &G,
        _detail_level: VisualDetail,
    ) where
        R: NodeRenderer<G>,
        G: GraphBase + NodeIndexable,
    {
        // Legacy rendering removed - tests need to be rewritten to use ViewportGraph
    }

    #[test]
    #[ignore] // Disabled: uses legacy rendering that was removed
    fn test_plot_layout_simple() {
        let layout = create_simple_layout();
        let mock_graph = create_mock_domain_graph(); // Empty, but needed for type
        let mut renderer = MockRenderer;

        let backend = TestBackend::new(60, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = ViewportState::new();
        state.camera_current = WorldPos::new(30, 0);

        terminal
            .draw(|f| {
                let area = f.area();
                state.update(Duration::ZERO, (area.width, area.height));

                let mut buffer = WorldBuffer::new(f.buffer_mut(), &state);
                test_plot_layout(
                    &layout,
                    &mut buffer,
                    &mut renderer,
                    (0, 0),
                    &mock_graph,
                    VisualDetail::Minimal,
                );
            })
            .unwrap();

        insta::assert_snapshot!(terminal.backend());
    }

    #[test]
    #[ignore] // Disabled: uses legacy rendering that was removed
    fn test_plot_layout_branching() {
        let layout = create_branched_layout();
        let mock_graph = create_mock_domain_graph();
        let mut renderer = MockRenderer;

        let backend = TestBackend::new(60, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        let mut state = ViewportState::new();
        state.camera_current = WorldPos::new(30, 0);

        terminal
            .draw(|f| {
                let area = f.area();
                state.update(Duration::ZERO, (area.width, area.height));

                let mut buffer = WorldBuffer::new(f.buffer_mut(), &state);
                test_plot_layout(
                    &layout,
                    &mut buffer,
                    &mut renderer,
                    (0, 0),
                    &mock_graph,
                    VisualDetail::Minimal,
                );
            })
            .unwrap();

        insta::assert_snapshot!(terminal.backend());
    }
}
