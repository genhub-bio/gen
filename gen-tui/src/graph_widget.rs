use petgraph::visit::GraphBase;
use ratatui::style::Style;

use crate::{
    assembly::AssembledLayout,
    distribute_nodes::GapSizes,
    geometry::{WorldPos, WorldRect},
    layout::{NodeRole, WindowGeometry},
    plotter::NodeRenderer,
    theme::Theme,
    viewport_state::WorldBuffer,
};

/// Single-cell glyph used by [`MinimalNodeRenderer`].
pub const NODE_GLYPH: char = '●';

/// A `NodeRenderer` that draws every node as a single glyph, regardless of domain data.
/// Needs no sequence/domain lookup, so it works for any `GraphBase` - the default,
/// always-available renderer for the lowest zoom level.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinimalNodeRenderer;

impl<G: GraphBase> NodeRenderer<G> for MinimalNodeRenderer {
    fn get_node_size(&self, _node: &G::NodeId) -> (u64, u64) {
        (1, 1)
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, _node_id: &G::NodeId) {
        buffer.set_char(area.left_center(), NODE_GLYPH);
    }
}

/// Route and compact an assembled neighbourhood window into renderable geometry.
///
/// This is the widget half of the level-of-detail split: the controller assembles the window
/// into a `StableGraph<LayoutNode, LayoutEdge>` (see [`crate::assembly::assemble_window`]),
/// and the widget re-sizes every node at the current
/// level of detail, then assigns cross coordinates, routes, and compacts it. Sizing is applied
/// here before coordinate assignment, so the geometry reflects the detail the visual currently
/// holds; the assembled neutral sizes are discarded.
///
/// `size_of` supplies the rendered size for each node role.
pub fn build_window_geometry(
    assembled: AssembledLayout,
    gaps: &GapSizes,
    mut size_of: impl FnMut(&NodeRole) -> (u64, u64),
) -> WindowGeometry {
    let mut graph = assembled.graph;
    for node in graph.node_weights_mut() {
        node.size = size_of(&node.role);
    }
    WindowGeometry::new(graph, gaps)
}

pub(crate) fn style_cursor_cell(buffer: &mut WorldBuffer, pos: WorldPos, theme: &Theme) {
    if let Some((ch, style)) = buffer.get_char_styled(pos) {
        let foreground = if ch == NODE_GLYPH || style.bg == Some(theme[0x00]) {
            style.fg.or(style.bg)
        } else {
            style.bg.or(style.fg)
        };
        buffer.set_char_styled(
            pos,
            ch,
            Style {
                fg: foreground.or(Some(theme[0x05])),
                bg: Some(theme[0x03]),
                ..style
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use ratatui::{buffer::Buffer, layout::Rect, style::Style};

    use crate::{
        geometry::WorldPos,
        graph_widget::{NODE_GLYPH, style_cursor_cell},
        theme::current_theme,
        viewport_state::{ViewportState, WorldBuffer},
    };

    #[test]
    fn cursor_cell_promotes_node_or_annotation_color_onto_base03() {
        let area = Rect::new(0, 0, 5, 5);
        let mut viewport = ViewportState::new();
        viewport.camera_current = WorldPos::new(2, 2);
        viewport.viewport_bounds = area;
        let theme = current_theme();
        let mut buffer = Buffer::empty(area);
        let annotation_pos = WorldPos::new(1, 2);
        let minimal_pos = WorldPos::new(2, 2);
        let sequence_pos = WorldPos::new(3, 2);

        let mut world_buffer = WorldBuffer::new(&mut buffer, &viewport);
        world_buffer.set_char_styled(
            annotation_pos,
            'A',
            Style::default().fg(theme[0x00]).bg(theme[0x08]),
        );
        world_buffer.set_char_styled(
            minimal_pos,
            NODE_GLYPH,
            Style::default().fg(theme[0x0C]).bg(theme[0x00]),
        );
        world_buffer.set_char_styled(
            sequence_pos,
            'C',
            Style::default().fg(theme[0x00]).bg(theme[0x05]),
        );

        style_cursor_cell(&mut world_buffer, annotation_pos, &theme);
        style_cursor_cell(&mut world_buffer, minimal_pos, &theme);
        style_cursor_cell(&mut world_buffer, sequence_pos, &theme);

        for (position, foreground) in [
            ((1, 2), theme[0x08]),
            ((2, 2), theme[0x0C]),
            ((3, 2), theme[0x05]),
        ] {
            assert_eq!(buffer[position].fg, foreground);
            assert_eq!(buffer[position].bg, theme[0x03]);
        }
    }

    mod window_geometry {
        use std::collections::HashMap;

        use super::super::*;
        use crate::{
            assembly::assemble_window,
            crawl::{EagerSource, GraphCursor, build_window_graph, neighborhood},
            layout::NodeRole,
            testing::mocks::{MockDomainGraph, TestGraphs},
        };

        /// Crawl the whole graph from its first node and assemble it into one window graph.
        fn assembled_window(graph: &MockDomainGraph) -> AssembledLayout {
            let anchor = graph
                .node_indices()
                .next()
                .expect("should have graph nodes");
            let mut graph = graph.clone();
            let subgraph = neighborhood(
                anchor,
                graph.node_count(),
                &mut GraphCursor::new(&mut graph, &mut EagerSource),
                &|_| false,
                &HashMap::new(),
            )
            .expect("should find the anchor in the graph");
            let (window, _) =
                build_window_graph(&subgraph, &graph, None).expect("should build the window");
            assemble_window(&window).expect("should assemble the window")
        }

        /// Size every data node 5x3 and every routing/pin node 1x1, standing in for a domain
        /// visual at a fixed level of detail.
        fn size_of(role: &NodeRole) -> (u64, u64) {
            match role {
                NodeRole::Data(_) => (5, 3),
                _ => (1, 1),
            }
        }

        #[test]
        fn test_window_geometry_keeps_data_nodes() {
            let graph = TestGraphs::domain_extended_diamond();
            let assembled = assembled_window(&graph);
            let expected_data = assembled
                .graph
                .node_weights()
                .filter(|node| matches!(node.role, NodeRole::Data(_)))
                .count();
            assert!(expected_data > 0);

            let geometry = build_window_geometry(assembled, &GapSizes::default(), size_of);

            let data_nodes = geometry
                .graph
                .node_weights()
                .filter(|node| matches!(node.role, NodeRole::Data(_)))
                .count();
            assert_eq!(
                data_nodes, expected_data,
                "routing and compaction must preserve every data node"
            );
        }

        #[test]
        fn test_window_geometry_applies_current_detail_sizes() {
            let graph = TestGraphs::domain_extended_diamond();
            let assembled = assembled_window(&graph);

            let geometry = build_window_geometry(assembled, &GapSizes::default(), size_of);

            for node in geometry.graph.node_weights() {
                if matches!(node.role, NodeRole::Data(_)) {
                    assert_eq!(
                        node.size,
                        (5, 3),
                        "data nodes must be re-sized from the widget's visual, not the \
                         assembled provisional size"
                    );
                }
            }
        }
    }
}
