use std::{cmp, hash::Hash, marker::PhantomData};

use petgraph::{
    graph::NodeIndex,
    visit::{
        EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeCount, NodeIndexable, Visitable,
    },
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    prelude::StatefulWidget,
    style::Style,
    widgets::{Block, Widget},
};

use crate::{
    geometry::{WorldPos, WorldRect},
    graph_controller::{GraphController, ViewportState, WorldBuffer},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer, plot_viewport_graph},
    theme::get_theme_color,
};

pub const NODE_GLYPH: char = '●'; // changed from '⏺', which renders as an emoji in some fonts;

/// Default NodeSizer that always returns 1x1 size
#[derive(Debug, Clone, Copy)]
pub struct DefaultNodeSizer;

impl<G> NodeSizer<G> for DefaultNodeSizer
where
    G: GraphBase,
{
    fn get_node_size(&self, _node: &G::NodeId, _detail_level: VisualDetail) -> (u64, u64) {
        (1, 1)
    }
}

/// Default NodeRenderer that draws a NodeGlyph at the center of the allocated area
#[derive(Debug, Default)]
pub struct DefaultNodeRenderer;

impl<G> NodeRenderer<G> for DefaultNodeRenderer
where
    G: GraphBase,
{
    fn render_node(
        &mut self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        _node_id: &G::NodeId,
        _detail_level: VisualDetail,
    ) {
        // Draw the node glyph at the center of the allocated area
        let center = area.center();
        buffer.set_char(center, NODE_GLYPH);
    }
}

/// Type alias for a GraphWidget with default renderer
pub type DefaultGraphWidget<'a, G, S> = GraphWidget<'a, G, S, DefaultNodeRenderer>;

/// A reusable Ratatui widget that renders a "world-space" viewport into a terminal area.
pub struct GraphWidget<'a, G, S, R>
where
    G: GraphBase + Clone,
    S: NodeSizer<G>,
    R: NodeRenderer<G>,
{
    /// Optional Block (border + title) around the widget.
    pub block: Option<Block<'a>>,

    /// Rendering style for the Block or background.
    pub style: Style,

    /// Optional user-provided callback that writes directly to the buffer using world coordinates.
    #[allow(clippy::type_complexity)]
    pub buffer_paint_fn: Option<Box<dyn FnMut(&mut WorldBuffer<'_>, &ViewportState) + 'a>>,

    /// Node renderer for domain-specific visualization
    pub renderer: R,

    /// Level of detail for the labels (minimal, full, truncated)
    pub detail_level: Option<VisualDetail>,

    /// Whether to enable cursor visibility
    pub cursor_enabled: bool,

    /// PhantomData to connect G and S type parameters
    _phantom: PhantomData<(G, S)>,
}

// Implementation for default renderer case
impl<'a, G, S> GraphWidget<'a, G, S, DefaultNodeRenderer>
where
    G: GraphBase + Clone,
    S: NodeSizer<G>,
{
    /// Create a new Graph widget with default renderer.
    pub fn new() -> GraphWidget<'a, G, S, DefaultNodeRenderer> {
        GraphWidget::<'a, G, S, DefaultNodeRenderer> {
            block: None,
            style: Style::default(),
            buffer_paint_fn: None,
            renderer: DefaultNodeRenderer,
            detail_level: None,
            cursor_enabled: false,
            _phantom: PhantomData,
        }
    }
}

// Implementation for any renderer type
impl<'a, G, S, R> GraphWidget<'a, G, S, R>
where
    G: GraphBase + Clone,
    S: NodeSizer<G>,
    R: NodeRenderer<G>,
{
    /// Create a new GraphWidget with a specific renderer
    pub fn with_renderer(renderer: R) -> GraphWidget<'a, G, S, R> {
        GraphWidget::<'a, G, S, R> {
            block: None,
            style: Style::default(),
            buffer_paint_fn: None,
            renderer,
            detail_level: None,
            cursor_enabled: false,
            _phantom: PhantomData,
        }
    }

    /// Set an optional Block (border + title) around the widget.
    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    /// Set the rendering style for the Block or background.
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Set the node renderer for domain-specific visualization.
    pub fn renderer(mut self, renderer: R) -> Self {
        self.renderer = renderer;
        self
    }

    /// Set the scale for level of detail.
    pub fn detail_level(mut self, detail_level: VisualDetail) -> Self {
        self.detail_level = Some(detail_level);
        self
    }

    /// Stores an optional closure that writes directly to the buffer using world coordinates,
    /// and is called after the graph is plotted.
    pub fn paint<F>(mut self, render_fn: F) -> Self
    where
        F: FnMut(&mut WorldBuffer<'_>, &ViewportState) + 'a,
    {
        self.buffer_paint_fn = Some(Box::new(render_fn));
        self
    }

    /// Enable cursor visibility for this widget.
    pub fn cursor(mut self) -> Self {
        self.cursor_enabled = true;
        self
    }
}

impl<G, S> Default for GraphWidget<'_, G, S, DefaultNodeRenderer>
where
    G: GraphBase + Clone,
    S: NodeSizer<G>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, S, R> StatefulWidget for GraphWidget<'_, G, S, R>
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
    for<'a> &'a G: IntoNodeIdentifiers + IntoEdgeReferences + IntoNeighborsDirected,
    for<'a> &'a G::NodeId: Hash + Ord,
    for<'a> &'a G::EdgeId: Clone,
    S: NodeSizer<G>,
    R: NodeRenderer<G>,
{
    type State = GraphController<G, S>;

    /// Render method that automatically plots the graph using the controller and renderer
    fn render(mut self, area: Rect, buf: &mut Buffer, controller: &mut Self::State) {
        // Clear the area with the widget's style
        buf.set_style(area, self.style);

        // Apply the block if present and get the inner area
        let inner_area = if let Some(block) = &self.block {
            let inner = block.inner(area);
            block.clone().render(area, buf);
            inner
        } else {
            area
        };

        // Check if viewport bounds changed
        let bounds_changed = controller.viewport_state.viewport_bounds != inner_area;

        // Update viewport bounds in state
        controller.viewport_state.viewport_bounds = inner_area;

        // Force a rebuild if bounds changed or if we still don't have nowieence view v
        if bounds_changed || controller.viewport_graph.graph.node_count() == 0 {
            controller.trigger_rebuild();
        }

        // Sync scale if provided
        if let Some(detail_level) = self.detail_level
            && controller.get_detail_level() != detail_level
        {
            controller.set_detail_level(detail_level);
        }

        // Detect motion and set rebuild flag if needed
        if controller.detect_motion() {
            controller.trigger_rebuild();
        }

        // Update ViewportGraph only if rebuild is needed
        // This ensures we always have the correct visible nodes/edges
        // Note: Camera and cursor adjustments for layout changes are handled by disperse/contract/set_detail_level
        if controller.needs_rebuild()
            && let Err(e) = controller.rebuild_viewport_graph()
        {
            log::error!("Error rebuilding viewport graph: {}", e);
        }

        // Render the graph using the renderer
        let mut world_buffer = WorldBuffer::new(buf, &controller.viewport_state);

        // Extract data from controller and render directly via ViewportGraph
        let viewport_graph = controller.get_viewport_graph();
        let detail_level = controller.get_detail_level();

        plot_viewport_graph(
            viewport_graph,
            &mut world_buffer,
            &mut self.renderer,
            &controller.graph,
            detail_level,
        );

        // Render path highlights if any exist
        self.render_path_highlights(&mut world_buffer, controller);

        // Call user supplied closure if provided
        if let Some(mut buffer_render_fn) = self.buffer_paint_fn {
            let mut buffer_writer = WorldBuffer::new(buf, &controller.viewport_state);
            buffer_render_fn(&mut buffer_writer, &controller.viewport_state);
        }

        if controller.cursor.is_visible() {
            // Get cursor world position for rendering
            if let Some(cursor_world_pos) = controller
                .cursor
                .to_world_pos(controller.get_viewport_graph())
            {
                let mut cursor_buffer = WorldBuffer::new(buf, &controller.viewport_state);

                // Get the current character at cursor position
                if let Some(current_char) = cursor_buffer.get_char(cursor_world_pos) {
                    // Get cursor colors from theme
                    let cursor_bg =
                        get_theme_color("cursor_bg").unwrap_or(ratatui::style::Color::White);
                    let cursor_fg =
                        get_theme_color("cursor_fg").unwrap_or(ratatui::style::Color::Black);

                    let cursor_style = Style::default().bg(cursor_bg).fg(cursor_fg);

                    // Write the same character back with cursor styling
                    cursor_buffer.set_char_styled(cursor_world_pos, current_char, cursor_style);
                }
            }
        }
    }
}

impl<'a, G, S, R> GraphWidget<'a, G, S, R>
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
    for<'b> &'b G: IntoNodeIdentifiers + IntoEdgeReferences + IntoNeighborsDirected,
    for<'b> &'b G::NodeId: Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
    S: NodeSizer<G>,
    R: NodeRenderer<G>,
{
    /// Render path highlights on top of the existing graph
    fn render_path_highlights(
        &self,
        buffer: &mut WorldBuffer<'_>,
        controller: &GraphController<G, S>,
    ) {
        let path_highlights = controller.get_path_highlights();
        let node_highlights = controller.get_node_highlights();
        let viewport_graph = controller.get_viewport_graph();

        if path_highlights.is_empty() && node_highlights.is_empty() {
            return;
        }

        // Render path highlights (edges)
        for (color, path_graph) in path_highlights {
            for (source, target, _) in path_graph.all_edges() {
                // Convert node IDs to domain indices
                let source_idx =
                    NodeIndex::new(<G as NodeIndexable>::to_index(&controller.graph, source));
                let target_idx =
                    NodeIndex::new(<G as NodeIndexable>::to_index(&controller.graph, target));

                // Find the visual positions of the source and target nodes
                if let (Some(&source_pos), Some(&target_pos)) = (
                    viewport_graph.node_positions.get(&source_idx),
                    viewport_graph.node_positions.get(&target_idx),
                ) {
                    // Draw highlighted edge with path color
                    self.draw_highlighted_edge(buffer, source_pos, target_pos, *color);
                }
            }
        }

        // Render node highlights
        for (color, highlighted_nodes) in node_highlights {
            for node_id in highlighted_nodes {
                // Convert node ID to domain index
                let node_idx =
                    NodeIndex::new(<G as NodeIndexable>::to_index(&controller.graph, *node_id));

                // Find the visual position of this node in the viewport
                if let Some(&node_pos) = viewport_graph.node_positions.get(&node_idx) {
                    // Get the current character at node position
                    if let Some(current_char) = buffer.get_char(node_pos) {
                        let highlight_style = Style::default()
                            .fg(*color)
                            .bg(get_theme_color("canvas").unwrap_or(ratatui::style::Color::Black));
                        buffer.set_char_styled(node_pos, current_char, highlight_style);
                    }
                }
            }
        }
    }

    /// Draw a highlighted edge between two world positions with a specific color
    fn draw_highlighted_edge(
        &self,
        buffer: &mut WorldBuffer<'_>,
        head: WorldPos,
        tail: WorldPos,
        color: ratatui::style::Color,
    ) {
        let edge_style = Style::default().fg(color);
        // Ensure the ordering
        let start = cmp::min(head, tail);
        let end = cmp::max(head, tail);

        if start.x == end.x {
            for y in start.y..=end.y {
                let pos = WorldPos { x: start.x, y };
                buffer.set_char_styled(pos, '┃', edge_style);
            }
        } else if start.y == end.y {
            for x in start.x..=end.x {
                let pos = crate::geometry::WorldPos { x, y: start.y };
                buffer.set_char_styled(pos, '━', edge_style);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ratatui::{
        style::{Color, Style},
        widgets::Block,
    };

    use super::*;

    // Mock types for testing
    type MockGraph = petgraph::stable_graph::StableDiGraph<(), ()>;

    struct MockNodeSizer;
    impl NodeSizer<MockGraph> for MockNodeSizer {
        fn get_node_size(
            &self,
            _node: &petgraph::graph::NodeIndex,
            _detail_level: VisualDetail,
        ) -> (u64, u64) {
            (1, 1)
        }
    }

    struct MockRenderer;
    impl NodeRenderer<MockGraph> for MockRenderer {
        fn render_node(
            &mut self,
            _buffer: &mut WorldBuffer,
            _area: crate::geometry::WorldRect,
            _node_id: &petgraph::graph::NodeIndex,
            _detail_level: VisualDetail,
        ) {
        }
    }

    #[test]
    fn test_builder_pattern() {
        let block = Block::default().title("Test Graph");
        let style = Style::default().bg(Color::Blue);

        let widget =
            GraphWidget::<'_, MockGraph, MockNodeSizer, MockRenderer>::with_renderer(MockRenderer)
                .block(block)
                .style(style);

        // Verify the builder pattern worked
        assert!(widget.block.is_some());
        assert_eq!(widget.style.bg, Some(Color::Blue));
    }

    #[test]
    fn test_default_implementation() {
        let widget = GraphWidget::<'_, MockGraph, MockNodeSizer, DefaultNodeRenderer>::default();

        assert!(widget.block.is_none());
        assert_eq!(widget.style, Style::default());
    }
}
