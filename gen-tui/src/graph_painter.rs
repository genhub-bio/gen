use std::{collections::HashSet, hash::Hash};

use petgraph::{
    graph::NodeIndex,
    visit::{GraphBase, NodeIndexable},
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    widgets::{Block, Widget},
};

use crate::{
    assembly::AssembledLayout,
    distribute_nodes::GapSizes,
    frame_index::{FrameIndex, PlacedNode},
    geometry::{WorldPos, WorldRect, floor_half},
    graph_widget::build_window_geometry,
    layout::{NodeRole, WindowGeometry},
    plotter::{
        CellHighlight, NodeRenderer, PathStyle, PlotDecorations,
        plot_viewport_graph_with_highlights,
    },
    theme::current_theme,
    viewport_graph::ViewportGraph,
    viewport_state::{ViewportState, WorldBuffer},
};

/// Every placed wormhole (`NodeRole::Wormhole`) stub's screen rect, boundary node, and
/// off-screen domain target. See `GraphPainter::render`.
pub type WormholeStubs<NodeId> = Vec<(WorldRect, NodeId, NodeId)>;

/// Type of element to highlight in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HighlightKind<N> {
    /// A single node
    Node(N),
    /// An edge between two nodes (source, target)
    Edge(N, N),
    /// Tint a sub-rectangle of a single node.
    /// tl/br are (col, row) offsets from the node's top-left corner; br is exclusive.
    Cells {
        node: N,
        tl: (i64, i64),
        br: (i64, i64),
    },
}

/// Where the painter should anchor the window it renders, and the hard-zone boundary
/// `snap_camera` uses to decide when the anchor needs to move. `anchor`/`anchor_fraction`
/// identify a specific pixel within a specific node (matching `Cursor`'s node + fractional
/// position semantics); `anchor_screen` is the signed screen position (relative to the
/// render area's top-left, allowed to go negative or past the far edge) that pixel is pinned
/// to across window changes.
#[derive(Debug, Clone, Copy)]
pub struct Camera<N> {
    pub anchor: N,
    pub anchor_fraction: (f64, f64),
    pub anchor_screen: (i64, i64),
    pub hard_zone: u16,
}

/// Persistent, domain-terms highlight/lowlight state, resolved to screen positions fresh on
/// every `GraphPainter::render` call rather than baked into any structure that survives a
/// frame.
#[derive(Debug, Clone)]
pub struct Highlights<N> {
    pub styles: Vec<(HighlightKind<N>, PathStyle)>,
    pub edge_lowlights: Vec<(N, N)>,
    pub node_lowlights: Vec<N>,
}

impl<N> Default for Highlights<N> {
    fn default() -> Self {
        Self {
            styles: Vec::new(),
            edge_lowlights: Vec::new(),
            node_lowlights: Vec::new(),
        }
    }
}

/// Keep the cursor inside `hard_zone` by shifting the camera anchor when needed.
pub fn snap_camera(
    cursor_screen: (i64, i64),
    area: Rect,
    anchor_screen: (i64, i64),
    hard_zone: u16,
) -> (i64, i64) {
    let (width, height) = (area.width, area.height);
    if width == 0 || height == 0 {
        return anchor_screen;
    }

    // Terminal cells are roughly twice as tall as they are wide.
    let min_dimension = width.min(height);
    let hard_zone = if hard_zone > min_dimension / 2 {
        1
    } else {
        hard_zone
    } as i64;
    let hard_zone_y = hard_zone / 2;

    let min_x = hard_zone;
    let max_x = width as i64 - hard_zone;
    let min_y = hard_zone_y;
    let max_y = height as i64 - hard_zone_y;

    let mut screen = anchor_screen;
    if cursor_screen.0 < min_x {
        screen.0 += min_x - cursor_screen.0;
    } else if cursor_screen.0 >= max_x {
        screen.0 -= cursor_screen.0 - (max_x - 1);
    }
    if cursor_screen.1 < min_y {
        screen.1 += min_y - cursor_screen.1;
    } else if cursor_screen.1 >= max_y {
        screen.1 -= cursor_screen.1 - (max_y - 1);
    }
    screen
}

/// Pure ephemeral graph painter: renders one `AssembledLayout` window into a buffer and
/// returns the `FrameIndex` describing where everything landed. Never touches
/// `LayoutEngine` or any cursor/view state - callers own the anchor and the highlight
/// lists and pass them in fresh every call.
pub struct GraphPainter<'a, G, V>
where
    G: GraphBase,
{
    window: AssembledLayout,
    graph: &'a G,
    visual: &'a V,
    gaps: GapSizes,
    block: Option<Block<'a>>,
    style: Style,
}

impl<'a, G, V> GraphPainter<'a, G, V>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash,
    V: NodeRenderer<G>,
{
    /// Construct a painter for one assembled window. `graph` is the domain graph the window
    /// was assembled from (needed only to translate its `NodeRole::Data` indices to/from
    /// `G::NodeId` when talking to `visual` and to the caller's domain-terms `Camera`/
    /// `Highlights`).
    pub fn new(window: AssembledLayout, graph: &'a G, visual: &'a V) -> Self {
        Self {
            window,
            graph,
            visual,
            gaps: GapSizes::default(),
            block: None,
            style: Style::default(),
        }
    }

    /// Set the minimum inter-node gap for each axis (the current zoom level's target).
    pub fn spacing(mut self, gaps: &GapSizes) -> Self {
        self.gaps = *gaps;
        self
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Size every node via `visual`, route and compact the window, place it so the camera's
    /// anchor pixel lands at `camera.anchor_screen`, paint the visible parts into `buf`, and
    /// return a `FrameIndex` over every placed Data node (including ones outside `area`),
    /// alongside a flat list of every placed wormhole (`NodeRole::Wormhole`) stub's screen rect,
    /// boundary node, and off-screen domain target - kept separate from `FrameIndex` rather
    /// than folded into it, since `FrameIndex`'s `N` key space is domain node ids only and
    /// wormhole stubs are synthetic routing nodes with no domain id of their own.
    pub fn render(
        mut self,
        area: Rect,
        buf: &mut Buffer,
        camera: &Camera<G::NodeId>,
        highlights: &Highlights<G::NodeId>,
    ) -> (FrameIndex<G::NodeId>, WormholeStubs<G::NodeId>) {
        buf.set_style(area, self.style);
        let inner_area = if let Some(block) = &self.block {
            let inner = block.inner(area);
            block.clone().render(area, buf);
            inner
        } else {
            area
        };

        let graph = self.graph;
        let visual = self.visual;
        let backward_edges = std::mem::take(&mut self.window.backward_edges);
        let geometry = build_window_geometry(self.window, &self.gaps, |role| match role {
            NodeRole::Data(node_index) => {
                let node_id = <G as NodeIndexable>::from_index(graph, node_index.index());
                visual.get_node_size(&node_id)
            }
            _ => visual.get_dummy_size(),
        });

        let offset = anchor_offset(&geometry, graph, camera);

        let viewport_graph = ViewportGraph::from_window_geometry(&geometry, &backward_edges);
        paint(
            &viewport_graph,
            inner_area,
            buf,
            offset,
            graph,
            visual,
            highlights,
        );

        (
            frame_index(&geometry, graph, offset, inner_area),
            wormhole_index(&geometry, graph, offset),
        )
    }
}

/// The translation from `geometry`'s local coordinate space to screen space such that the
/// camera's anchor pixel (its node's position plus `anchor_fraction`) lands exactly at
/// `camera.anchor_screen`. The anchor's node rect, offset by the fractional position
/// within it, is the point being pinned.
fn anchor_offset<G>(geometry: &WindowGeometry, graph: &G, camera: &Camera<G::NodeId>) -> WorldPos
where
    G: GraphBase + NodeIndexable,
{
    let anchor_node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, camera.anchor));
    let anchor_layout = geometry
        .graph
        .node_weights()
        .find_map(|node| match node.role {
            NodeRole::Data(node_index) if node_index == anchor_node_index => {
                Some((node.pos, node.size))
            }
            _ => None,
        });

    let Some((pos, size)) = anchor_layout else {
        return WorldPos::ZERO;
    };

    let anchor_point = WorldRect::from_center_and_size(pos.point(), size)
        .point_at_fraction(camera.anchor_fraction);

    WorldPos::new(
        camera.anchor_screen.0 - anchor_point.x,
        camera.anchor_screen.1 - anchor_point.y,
    )
}

/// Paint the window into `buf`, resolving `highlights` from domain terms to screen positions
/// for this call only (nothing is baked into `viewport_graph` beyond this paint).
fn paint<G, V>(
    viewport_graph: &ViewportGraph,
    area: Rect,
    buf: &mut Buffer,
    offset: WorldPos,
    graph: &G,
    visual: &V,
    highlights: &Highlights<G::NodeId>,
) where
    G: GraphBase + NodeIndexable,
    V: NodeRenderer<G>,
{
    let half_width = floor_half(area.width as i64);
    let half_height = floor_half(area.height as i64);
    let camera_current = WorldPos::new(half_width - offset.x, half_height - offset.y);

    let mut viewport_state = ViewportState::new();
    viewport_state.camera_current = camera_current;
    viewport_state.viewport_bounds = area;

    let mut node_highlights = Vec::new();
    let mut edge_highlights = Vec::new();
    let mut cell_highlights = Vec::new();
    for (kind, style) in &highlights.styles {
        match kind {
            HighlightKind::Node(node_id) => {
                apply_node_highlight(
                    viewport_graph,
                    graph,
                    *node_id,
                    *style,
                    &mut node_highlights,
                );
            }
            HighlightKind::Edge(source, target) => {
                apply_edge_highlight(
                    viewport_graph,
                    graph,
                    *source,
                    *target,
                    *style,
                    &mut edge_highlights,
                );
            }
            HighlightKind::Cells { node, tl, br } => {
                apply_cell_highlight(
                    viewport_graph,
                    graph,
                    *node,
                    *tl,
                    *br,
                    *style,
                    &mut cell_highlights,
                );
            }
        }
    }

    let edge_lowlights = resolve_edge_lowlights(viewport_graph, graph, &highlights.edge_lowlights);
    let node_lowlights = resolve_node_lowlights(viewport_graph, graph, &highlights.node_lowlights);

    let theme = current_theme();
    let mut world_buffer = WorldBuffer::new(buf, &viewport_state);
    plot_viewport_graph_with_highlights(
        viewport_graph,
        &mut world_buffer,
        visual,
        graph,
        PlotDecorations {
            node_highlights: &node_highlights,
            edge_highlights: &edge_highlights,
            cell_highlights: &cell_highlights,
            lowlights: &edge_lowlights,
            node_lowlights: &node_lowlights,
        },
        &theme,
    );
}

fn apply_node_highlight<G: GraphBase + NodeIndexable>(
    viewport_graph: &ViewportGraph,
    graph: &G,
    node_id: G::NodeId,
    style: PathStyle,
    out: &mut Vec<(WorldPos, PathStyle)>,
) {
    let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
    if let Some(pos) = viewport_graph.node_positions.get(&node_index) {
        out.push((*pos, style));
    }
}

fn apply_edge_highlight<G: GraphBase + NodeIndexable>(
    viewport_graph: &ViewportGraph,
    graph: &G,
    source_id: G::NodeId,
    target_id: G::NodeId,
    style: PathStyle,
    out: &mut Vec<((WorldPos, WorldPos), PathStyle)>,
) {
    let source = NodeIndex::new(<G as NodeIndexable>::to_index(graph, source_id));
    let target = NodeIndex::new(<G as NodeIndexable>::to_index(graph, target_id));

    let edges: Vec<(WorldPos, WorldPos)> = viewport_graph
        .edges()
        .filter(|(_, _, bundle)| {
            bundle.contains(&(source, target)) || bundle.contains(&(target, source))
        })
        .map(|(source_pos, target_pos, _)| (source_pos, target_pos))
        .collect();

    out.extend(edges.into_iter().map(|edge| (edge, style)));
}

fn apply_cell_highlight<G: GraphBase + NodeIndexable>(
    viewport_graph: &ViewportGraph,
    graph: &G,
    node_id: G::NodeId,
    tl: (i64, i64),
    br: (i64, i64),
    style: PathStyle,
    out: &mut Vec<CellHighlight>,
) {
    let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
    if let Some(&world_pos) = viewport_graph.node_positions.get(&node_index) {
        out.push((world_pos, tl, br, style));
    }
}

fn resolve_edge_lowlights<G: GraphBase + NodeIndexable>(
    viewport_graph: &ViewportGraph,
    graph: &G,
    domain_lowlights: &[(G::NodeId, G::NodeId)],
) -> Vec<(WorldPos, WorldPos)> {
    let lowlight_pairs: HashSet<(NodeIndex, NodeIndex)> = domain_lowlights
        .iter()
        .map(|(source, target)| {
            let source = NodeIndex::new(<G as NodeIndexable>::to_index(graph, *source));
            let target = NodeIndex::new(<G as NodeIndexable>::to_index(graph, *target));
            if source <= target {
                (source, target)
            } else {
                (target, source)
            }
        })
        .collect();

    viewport_graph
        .edges()
        .filter(|(_, _, bundle)| {
            !bundle.is_empty()
                && bundle.iter().all(|&(source, target)| {
                    let pair = if source <= target {
                        (source, target)
                    } else {
                        (target, source)
                    };
                    lowlight_pairs.contains(&pair)
                })
        })
        .map(|(source_pos, target_pos, _)| (source_pos, target_pos))
        .collect()
}

fn resolve_node_lowlights<G: GraphBase + NodeIndexable>(
    viewport_graph: &ViewportGraph,
    graph: &G,
    domain_node_lowlights: &[G::NodeId],
) -> Vec<WorldPos> {
    domain_node_lowlights
        .iter()
        .filter_map(|&node_id| {
            let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
            viewport_graph.node_positions.get(&node_index).copied()
        })
        .collect()
}

/// Build the `FrameIndex` over every placed Data node.
fn frame_index<G>(
    geometry: &WindowGeometry,
    graph: &G,
    offset: WorldPos,
    area: Rect,
) -> FrameIndex<G::NodeId>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash,
{
    let placed: Vec<PlacedNode<G::NodeId>> = geometry
        .graph
        .node_weights()
        .filter_map(|node| {
            let NodeRole::Data(node_index) = node.role else {
                return None;
            };
            let node_id = <G as NodeIndexable>::from_index(graph, node_index.index());
            let center = WorldPos::new(node.pos.x + offset.x, node.pos.y + offset.y);
            let rect = WorldRect::from_center_and_size(center, node.size);
            Some(PlacedNode {
                id: node_id,
                rect,
                layer: node.layer.unwrap_or_default(),
            })
        })
        .collect();

    let clip_area = WorldRect::from_coords(
        0,
        0,
        (area.width as i64).saturating_sub(1),
        (area.height as i64).saturating_sub(1),
    );
    FrameIndex::build(placed, clip_area)
}

/// Every placed `NodeRole::Wormhole` (wormhole) stub's screen rect, the boundary node whose
/// edge produced it, and its off-screen domain target - the click-to-teleport counterpart of
/// `frame_index`'s Data-only index. Sourced from the same `external_edges` already fed to the
/// router. The boundary comes from the stub's own incident
/// edge bundle, which the router labels with every collapsed `(boundary, target)` domain edge.
/// Matching by `NodeRole::Wormhole(target)` alone is insufficient because two different
/// boundaries may choose the same off-window target; doing so would map both hit entries to the
/// first rendered stub and leave the second arrow unclickable. Not clipped to `area`: mirrors
/// `frame_index`'s "including ones outside area" contract, since a stub just past the visible
/// edge is still a legitimate teleport target once panned into view.
fn wormhole_index<G>(
    geometry: &WindowGeometry,
    graph: &G,
    offset: WorldPos,
) -> WormholeStubs<G::NodeId>
where
    G: GraphBase + NodeIndexable,
{
    geometry
        .graph
        .node_indices()
        .filter_map(|stub_index| {
            let placed = &geometry.graph[stub_index];
            let NodeRole::Wormhole(target_index) = &placed.role else {
                return None;
            };
            let target_index = *target_index;
            let boundary_index = geometry
                .graph
                .edges(stub_index)
                .find_map(|edge| edge.weight().bundle.first().map(|&(boundary, _)| boundary))?;
            let boundary_id = <G as NodeIndexable>::from_index(graph, boundary_index.index());
            let target_id = <G as NodeIndexable>::from_index(graph, target_index.index());
            let center = WorldPos::new(placed.pos.x + offset.x, placed.pos.y + offset.y);
            let rect = WorldRect::from_center_and_size(center, placed.size);
            Some((rect, boundary_id, target_id))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use petgraph::graph::NodeIndex;
    use ratatui::{buffer::Buffer, layout::Rect};

    use super::*;
    use crate::{
        geometry::{LocalPos, WorldRect},
        layout::{LayoutEdge, LayoutNode, WindowGeometry},
        layout_engine::LayoutEngine,
        testing::mocks::{MockDomainGraph, TestGraphs},
    };

    #[derive(Clone)]
    struct FixedSizeVisual {
        size: (u64, u64),
    }

    impl NodeRenderer<MockDomainGraph> for FixedSizeVisual {
        fn get_node_size(&self, _node: &NodeIndex) -> (u64, u64) {
            self.size
        }

        fn get_dummy_size(&self) -> (u64, u64) {
            (1, 1)
        }

        fn render_node(&self, _buffer: &mut WorldBuffer, _area: WorldRect, _node_id: &NodeIndex) {}
    }

    #[test]
    fn test_snap_camera_leaves_anchor_alone_inside_hard_zone() {
        let area = Rect::new(0, 0, 20, 20);
        let anchor_screen = (10, 10);
        let cursor_screen = (10, 10);
        assert_eq!(
            snap_camera(cursor_screen, area, anchor_screen, 2),
            anchor_screen
        );
    }

    #[test]
    fn test_snap_camera_snaps_when_cursor_crosses_hard_zone() {
        let area = Rect::new(0, 0, 20, 20);
        let anchor_screen = (10, 10);
        // Cursor moved to the far right edge, well past the hard zone boundary.
        let cursor_screen = (19, 10);
        let snapped = snap_camera(cursor_screen, area, anchor_screen, 2);
        assert_ne!(snapped, anchor_screen, "camera should snap immediately");
        assert_eq!(snapped.0, 8);
        assert_eq!(
            cursor_screen.0 + snapped.0 - anchor_screen.0,
            area.width as i64 - 3,
            "camera translation should put the cursor on the right hard-zone boundary"
        );
    }

    #[test]
    fn test_snap_camera_corrects_each_viewport_edge_toward_the_interior() {
        let area = Rect::new(0, 0, 20, 20);

        assert_eq!(snap_camera((-3, 10), area, (-3, 10), 2).0, 2);
        assert_eq!(snap_camera((22, 10), area, (22, 10), 2).0, 17);
        assert_eq!(snap_camera((10, -3), area, (10, -3), 2).1, 1);
        assert_eq!(snap_camera((10, 22), area, (10, 22), 2).1, 18);
    }

    #[test]
    fn test_render_places_anchor_at_requested_screen_position() {
        let domain_graph = TestGraphs::domain_extended_diamond();
        let mut engine = LayoutEngine::new(domain_graph);
        let anchor = engine.default_anchor().expect("should have graph nodes");
        let node_budget = engine.neighborhood_node_budget(20);
        let window = engine
            .window_for(anchor, node_budget)
            .expect("should build a window");

        let visual = FixedSizeVisual { size: (5, 3) };
        let graph = engine.graph();
        let camera = Camera {
            anchor,
            anchor_fraction: (0.5, 0.5),
            anchor_screen: (7, 4),
            hard_zone: 2,
        };
        let highlights = Highlights::default();

        let area = Rect::new(0, 0, 30, 15);
        let mut buffer = Buffer::empty(area);
        let painter = GraphPainter::new(window, graph, &visual);
        let (frame, _wormhole) = painter.render(area, &mut buffer, &camera, &highlights);

        let anchor_rect = frame.rect_of(anchor).expect("should place the anchor");
        assert_eq!(
            anchor_rect.center(),
            crate::geometry::Point::new(7, 4),
            "anchor's center should land exactly at camera.anchor_screen for a (0.5, 0.5) fraction"
        );
    }

    #[test]
    fn test_render_frame_index_covers_every_data_node() {
        let domain_graph = TestGraphs::domain_extended_diamond();
        let expected_data_nodes =
            <MockDomainGraph as petgraph::visit::NodeCount>::node_count(&domain_graph);
        let mut engine = LayoutEngine::new(domain_graph);
        let anchor = engine.default_anchor().expect("should have graph nodes");
        let node_budget = engine.neighborhood_node_budget(40).max(10);
        let window = engine
            .window_for(anchor, node_budget)
            .expect("should build a window");

        let visual = FixedSizeVisual { size: (3, 1) };
        let graph = engine.graph();
        let camera = Camera {
            anchor,
            anchor_fraction: (0.0, 0.5),
            anchor_screen: (2, 5),
            hard_zone: 2,
        };
        let highlights = Highlights::default();

        let area = Rect::new(0, 0, 60, 20);
        let mut buffer = Buffer::empty(area);
        let painter = GraphPainter::new(window, graph, &visual);
        let (frame, _wormhole) = painter.render(area, &mut buffer, &camera, &highlights);

        for node_index in 0..expected_data_nodes {
            let node_id = NodeIndex::new(node_index);
            assert!(
                frame.rect_of(node_id).is_some(),
                "every domain node in the window should be placed, missing {:?}",
                node_id
            );
        }
    }

    #[test]
    fn wormhole_index_distinguishes_stubs_with_the_same_target() {
        let mut domain_graph = MockDomainGraph::new();
        let boundary_a = domain_graph.add_node(());
        let boundary_b = domain_graph.add_node(());
        let target = domain_graph.add_node(());

        let mut graph = petgraph::stable_graph::StableGraph::<
            LayoutNode,
            LayoutEdge,
            petgraph::Undirected,
            u32,
        >::default();
        let stub_a = graph.add_node(LayoutNode::new(
            NodeRole::Wormhole(target),
            LocalPos::new_xy(2, 1),
            (1, 1),
            None,
        ));
        let route_a = graph.add_node(LayoutNode::routing(LocalPos::new_xy(1, 1), (1, 1)));
        graph.add_edge(stub_a, route_a, LayoutEdge::new(boundary_a, target));

        let stub_b = graph.add_node(LayoutNode::new(
            NodeRole::Wormhole(target),
            LocalPos::new_xy(8, 3),
            (1, 1),
            None,
        ));
        let route_b = graph.add_node(LayoutNode::routing(LocalPos::new_xy(7, 3), (1, 1)));
        graph.add_edge(stub_b, route_b, LayoutEdge::new(boundary_b, target));

        let geometry = WindowGeometry {
            graph,
            width: 8,
            height: 3,
        };

        let indexed = wormhole_index(&geometry, &domain_graph, WorldPos::ZERO);

        assert_eq!(indexed.len(), 2);
        assert!(indexed.contains(&(
            WorldRect::from_center_and_size(WorldPos::new(2, 1), (1, 1)),
            boundary_a,
            target,
        )));
        assert!(indexed.contains(&(
            WorldRect::from_center_and_size(WorldPos::new(8, 3), (1, 1)),
            boundary_b,
            target,
        )));
    }
}
