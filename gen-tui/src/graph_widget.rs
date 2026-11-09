use std::collections::HashSet;

use petgraph::{graph::NodeIndex, visit::GraphBase};
use ratatui::style::Style;

use crate::{
    assembly::AssembledLayout,
    curved::EdgeStyle,
    distribute_nodes::GapSizes,
    geometry::{WorldPos, WorldRect},
    layout::{HAIR_PIN_SIZE, NodeRole, VerticalAnchor, WindowGeometry},
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
/// `size_of` supplies the rendered size for each node role, and `is_visible` says which domain
/// nodes are drawn; the rest become junctions of their edges (see [`WindowGeometry::new`]) and
/// should be given a routing node's size. `is_aligned` marks nodes that pin their layer to
/// y = 0 (see `NodeRenderer::is_aligned`).
pub fn build_window_geometry(
    assembled: AssembledLayout,
    gaps: &GapSizes,
    size_of: impl FnMut(&NodeRole) -> (u64, u64),
    is_visible: impl Fn(NodeIndex) -> bool,
    is_aligned: impl Fn(NodeIndex) -> bool,
) -> WindowGeometry {
    build_window_geometry_with_anchors(
        assembled,
        gaps,
        EdgeStyle::Rectilinear,
        size_of,
        is_visible,
        is_aligned,
        |_| VerticalAnchor::Center,
    )
}

/// Build geometry with a per-node vertical attachment row, shared by layout and rendering.
pub fn build_window_geometry_with_anchors(
    assembled: AssembledLayout,
    gaps: &GapSizes,
    edge_style: EdgeStyle,
    mut size_of: impl FnMut(&NodeRole) -> (u64, u64),
    is_visible: impl Fn(NodeIndex) -> bool,
    is_aligned: impl Fn(NodeIndex) -> bool,
    vertical_anchor: impl Fn(NodeIndex) -> VerticalAnchor,
) -> WindowGeometry {
    let mut graph = assembled.graph;
    let mut junctions = HashSet::new();
    let mut centered = HashSet::new();
    for node in graph.node_weights_mut() {
        node.size = size_of(&node.role);
        match node.role {
            NodeRole::Data(domain_index) => {
                node.vertical_anchor = vertical_anchor(domain_index);
                if !is_visible(domain_index) {
                    junctions.insert(domain_index);
                }
                if is_aligned(domain_index) {
                    centered.insert(domain_index);
                }
            }
            // A turnaround has a glyph of its own whose top row is the row its bus meets.
            NodeRole::HairPin(_) => {
                node.size = HAIR_PIN_SIZE;
                node.vertical_anchor = VerticalAnchor::Top;
            }
            _ => {}
        }
    }
    WindowGeometry::new(
        graph,
        gaps,
        &junctions,
        &centered,
        &assembled.pinned_edges,
        edge_style,
    )
}

/// Arrow drawn above an accent-styled cursor cell.
pub const CURSOR_ARROW: char = '↓';

/// Fill the cursor's cell with the theme's cursor accent, keeping its character, and point an
/// arrow at it from the cell above. The arrow never replaces a letter or digit, so a node's
/// sequence or label is not overwritten by it.
pub(crate) fn style_cursor_cell(buffer: &mut WorldBuffer, pos: WorldPos, theme: &Theme) {
    let accent = theme[0x0D];
    if let Some((ch, style)) = buffer.get_char_styled(pos) {
        buffer.set_char_styled(
            pos,
            ch,
            Style {
                fg: Some(theme[0x00]),
                bg: Some(accent),
                ..style
            },
        );
    }
    let above = WorldPos::new(pos.x, pos.y + 1);
    if let Some((ch, style)) = buffer.get_char_styled(above)
        && !ch.is_alphanumeric()
    {
        buffer.set_char_styled(
            above,
            CURSOR_ARROW,
            Style {
                fg: Some(accent),
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
        graph_widget::{CURSOR_ARROW, style_cursor_cell},
        theme::current_theme,
        viewport_state::{ViewportState, WorldBuffer},
    };

    #[test]
    fn test_cursor_fills_the_cell_and_points_at_it_from_above() {
        let area = Rect::new(0, 0, 5, 5);
        let mut viewport = ViewportState::new();
        viewport.camera_current = WorldPos::new(2, 2);
        viewport.viewport_bounds = area;
        let theme = current_theme();
        let mut buffer = Buffer::empty(area);
        let cursor = WorldPos::new(2, 2);
        let letter_above = WorldPos::new(4, 3);

        let mut world_buffer = WorldBuffer::new(&mut buffer, &viewport);
        world_buffer.set_char_styled(
            cursor,
            'C',
            Style::default().fg(theme[0x00]).bg(theme[0x05]),
        );
        world_buffer.set_char_styled(letter_above, 'G', Style::default());
        style_cursor_cell(&mut world_buffer, cursor, &theme);
        style_cursor_cell(&mut world_buffer, WorldPos::new(4, 2), &theme);

        assert_eq!(buffer[(2, 2)].symbol(), "C");
        assert_eq!(buffer[(2, 2)].bg, theme[0x0D]);
        assert_eq!(buffer[(2, 1)].symbol(), CURSOR_ARROW.to_string());
        assert_eq!(buffer[(2, 1)].fg, theme[0x0D]);
        assert_eq!(
            buffer[(4, 1)].symbol(),
            "G",
            "should not overwrite a letter with the arrow"
        );
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
        fn test_top_anchors_survive_routing_and_compaction() {
            let graph = TestGraphs::domain_extended_diamond();
            let geometry = build_window_geometry_with_anchors(
                assembled_window(&graph),
                &GapSizes::default(),
                EdgeStyle::Rectilinear,
                |role| match role {
                    NodeRole::Data(index) => (5, 2 + index.index() as u64 % 5),
                    _ => (1, 1),
                },
                |_| true,
                |_| false,
                |_| VerticalAnchor::Top,
            );
            for index in geometry.graph.node_indices() {
                let node = &geometry.graph[index];
                if matches!(node.role, NodeRole::Data(_)) {
                    let rect = node.rect(WorldPos::ZERO);
                    assert_eq!(node.vertical_anchor, VerticalAnchor::Top);
                    assert_eq!(rect.max.y, node.pos.y);
                    assert_eq!(rect.max.y - rect.min.y + 1, node.size.1 as i64);
                    for neighbor in geometry.graph.neighbors(index) {
                        assert_eq!(
                            geometry.graph[neighbor].pos.y, rect.max.y,
                            "edges must attach horizontally on the top row"
                        );
                    }
                }
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

            let geometry = build_window_geometry(
                assembled,
                &GapSizes::default(),
                size_of,
                |_| true,
                |_| false,
            );

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

            let geometry = build_window_geometry(
                assembled,
                &GapSizes::default(),
                size_of,
                |_| true,
                |_| false,
            );

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
