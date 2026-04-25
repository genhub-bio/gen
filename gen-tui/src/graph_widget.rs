use std::{hash::Hash, marker::PhantomData};

use petgraph::visit::{
    EdgeIndexable, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
    NodeCount, NodeIndexable, Visitable,
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    widgets::{Block, StatefulWidget, Widget},
};

use crate::{
    geometry::{BigRect, WorldPos, WorldRect},
    graph_controller::{GraphController, ViewportState, WorldBuffer},
    layout::VisualDetail,
    plotter::{NodeRenderer, NodeSizer, plot_viewport_graph_with_highlights},
    theme::current_theme,
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
    G: GraphBase,
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
    G: GraphBase,
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
    G: GraphBase,
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
    G: GraphBase,
    S: NodeSizer<G>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, S, R> StatefulWidget for GraphWidget<'_, G, S, R>
where
    G: GraphBase + EdgeIndexable + NodeIndexable + NodeCount + Visitable,
    G::NodeId: Copy + Eq + Hash + Ord,
    G::EdgeId: Clone,
    for<'a> &'a G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNodeIdentifiers<NodeId = G::NodeId>
        + IntoEdgeReferences<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNeighborsDirected<NodeId = G::NodeId>,
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

        // Force a rebuild if on sufficient movement, change in window size or it's still empty
        if controller.detect_motion()
            || bounds_changed
            || controller.viewport_graph.graph.node_count() == 0
        {
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
        let theme = current_theme();
        let mut world_buffer = WorldBuffer::new(buf, &controller.viewport_state);

        // Extract data from controller and render directly via ViewportGraph
        let viewport_graph = controller.get_viewport_graph();
        let detail_level = controller.get_detail_level();
        let node_highlights = controller.get_node_highlights();
        let edge_highlights = controller.get_edge_highlights();
        let cell_highlights = controller.get_cell_highlights();
        plot_viewport_graph_with_highlights(
            viewport_graph,
            &mut world_buffer,
            &mut self.renderer,
            controller.graph(),
            detail_level,
            node_highlights,
            edge_highlights,
            cell_highlights,
            &theme,
        );

        // Call user supplied closure if provided
        if let Some(mut buffer_render_fn) = self.buffer_paint_fn {
            let mut buffer_writer = WorldBuffer::new(buf, &controller.viewport_state);
            buffer_render_fn(&mut buffer_writer, &controller.viewport_state);
        }

        if controller.cursor.is_visible() {
            let viewport_graph = controller.get_viewport_graph();
            // Apply color, formatting and/or glyph swap
            let apply_cursor_style = |ch: char, style: Style| -> (char, Style) {
                if ch == NODE_GLYPH {
                    let new_style = style
                        .fg(theme[0x06])
                        .bg(theme[0x00])
                        .remove_modifier(Modifier::all());
                    ('█', new_style) // alternative: ○
                } else {
                    let new_style = style
                        .fg(theme[0x00])
                        .bg(theme[0x06])
                        .remove_modifier(Modifier::all())
                        .add_modifier(Modifier::BOLD);
                    (ch, new_style)
                }
            };

            if controller.cursor.is_coarse_mode() {
                // Highlight the whole node in coarse mode
                if let Some(node_idx) = controller.cursor.node_idx()
                    && let Some(&node_center) = viewport_graph.node_positions.get(&node_idx)
                    && let Some(node_data) = viewport_graph.node_data_by_pos.get(&node_center)
                {
                    let node_rect = BigRect::from_center_and_size(node_center, node_data.size);
                    let mut cursor_buffer = WorldBuffer::new(buf, &controller.viewport_state);

                    for y in node_rect.bottom()..=node_rect.top() {
                        for x in node_rect.left()..=node_rect.right() {
                            let pos = WorldPos::new(x, y);
                            if let Some((current_char, current_style)) =
                                cursor_buffer.get_char_styled(pos)
                            {
                                let (char_to_render, new_style) =
                                    apply_cursor_style(current_char, current_style);
                                cursor_buffer.set_char_styled(
                                    pos,
                                    char_to_render,
                                    new_style.add_modifier(Modifier::UNDERLINED),
                                );
                            }
                        }
                    }
                }
            } else {
                // Single cell cursor in normal mode
                if let Some(cursor_world_pos) = controller.cursor.to_world_pos(viewport_graph) {
                    let mut cursor_buffer = WorldBuffer::new(buf, &controller.viewport_state);

                    if let Some((current_char, current_style)) =
                        cursor_buffer.get_char_styled(cursor_world_pos)
                    {
                        let (char_to_render, new_style) =
                            apply_cursor_style(current_char, current_style);
                        cursor_buffer.set_char_styled(cursor_world_pos, char_to_render, new_style);
                    }

                    let below_pos = WorldPos::new(cursor_world_pos.x, cursor_world_pos.y - 1);
                    cursor_buffer.set_char(below_pos, '^');
                }
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
            _area: WorldRect,
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
