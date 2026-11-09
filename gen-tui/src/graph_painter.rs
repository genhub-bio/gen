use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use petgraph::{
    graph::NodeIndex,
    visit::{GraphBase, NodeIndexable},
};
use ratatui::{buffer::Buffer, layout::Rect};

use crate::{
    assembly::AssembledLayout,
    curved::{LayoutMode, OverlayPrimitive},
    edge_ports::{EdgePorts, EdgeSelector, NodeSide, PortedEdge},
    frame_index::{FrameIndex, PlacedNode},
    geometry::{WorldPos, WorldRect, floor_half},
    layout::{LayoutNode, NodeRole, WindowGeometry},
    layout_compaction::GapPolicy,
    plotter::{
        CellHighlight, NodeRenderer, PathStyle, PlotDecorations, highlighted_edge_route,
        plot_routed_window_graph_with_highlights,
    },
    routed_window_graph::RoutedWindowGraph,
    theme::current_theme,
    viewport_state::{ViewportState, WorldBuffer},
    window_geometry::build_window_geometry_with_anchors,
};

/// Every placed wormhole (`NodeRole::Wormhole`) stub's screen rect, boundary node, and
/// off-screen domain target. See `GraphPainter::render`.
pub(crate) type WormholeStubs<NodeId> = Vec<(WorldRect, NodeId, NodeId)>;

/// What painting a window produced besides the cells it drew into the buffer.
pub(crate) struct RenderedWindow<NodeId> {
    /// Where every data node landed.
    pub frame: FrameIndex<NodeId>,
    /// Every placed wormhole stub.
    pub wormhole_stubs: WormholeStubs<NodeId>,
    /// Overlay primitives in buffer cell units. Empty unless the layout mode is curved.
    pub overlay_primitives: Vec<OverlayPrimitive>,
}

/// Type of element to highlight in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HighlightKind<N> {
    /// A single node
    Node(N),
    /// One edge, named in full or by its endpoints (see [`EdgeSelector`])
    Edge(EdgeSelector<N>),
    /// Tint a sub-rectangle of a single node.
    /// first_cell/last_cell are (col, row) offsets from the node's top-left corner; last_cell is inclusive.
    Cells {
        node: N,
        first_cell: (i64, i64),
        last_cell: (i64, i64),
    },
}

/// Where the painter should anchor the window it renders, and the hard-zone boundary
/// `snap_camera` uses to decide when the anchor needs to move. `anchor`/`anchor_fraction`
/// identify a specific pixel within a specific node (matching `CursorState`'s node + fractional
/// position semantics); `anchor_screen` is the signed screen position (relative to the
/// render area's bottom-left corner with Y growing upward, allowed to go negative or past the
/// far edge) that pixel is pinned to across window changes.
#[derive(Debug, Clone, Copy)]
pub struct Camera<N> {
    pub anchor: N,
    pub anchor_fraction: (f64, f64),
    pub(crate) anchor_screen: (i64, i64),
    pub(crate) hard_zone: u16,
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
pub(crate) fn snap_camera(
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

/// One window's routed, compacted geometry at fixed node sizes and gaps, with the routed window
/// graph built from it. Neither depends on the camera, so a view that only pans or moves its
/// cursor keeps painting the same scene at a new offset instead of re-routing the window.
#[derive(Clone)]
pub(crate) struct WindowScene {
    geometry: WindowGeometry,
    routed_window_graph: RoutedWindowGraph,
    layout_mode: LayoutMode,
}

impl WindowScene {
    /// Size every node of `window` via `visual`, then route and compact it with `gap_policy`.
    pub fn new<G, V>(
        window: AssembledLayout,
        graph: &G,
        visual: &V,
        gap_policy: &GapPolicy,
        layout_mode: LayoutMode,
    ) -> Self
    where
        G: GraphBase + NodeIndexable,
        V: NodeRenderer<G>,
    {
        let domain_edges = window.domain_edges.clone();
        let is_visible = |node_index: NodeIndex| {
            visual.is_visible(&<G as NodeIndexable>::from_index(graph, node_index.index()))
        };
        let geometry = build_window_geometry_with_anchors(
            window,
            gap_policy,
            layout_mode,
            |role| match role {
                NodeRole::Data(node_index) if is_visible(*node_index) => {
                    let node_id = <G as NodeIndexable>::from_index(graph, node_index.index());
                    visual.get_node_size(&node_id)
                }
                _ => visual.get_dummy_size(),
            },
            is_visible,
            |node_index| {
                visual.is_aligned(&<G as NodeIndexable>::from_index(graph, node_index.index()))
            },
            |node_index| {
                visual.vertical_anchor(&<G as NodeIndexable>::from_index(graph, node_index.index()))
            },
        );
        let routed_window_graph = RoutedWindowGraph::from_window_geometry(&geometry, &domain_edges);
        Self {
            geometry,
            routed_window_graph,
            layout_mode,
        }
    }

    /// Place the scene so the camera's anchor pixel lands at `camera.anchor_screen`, paint the
    /// visible parts into `buffer`, and index where everything landed (see
    /// [`GraphPainter::render`]). `graph` and `visual` must be the ones the scene was built
    /// with.
    pub fn paint<G, V>(
        &self,
        graph: &G,
        visual: &V,
        area: Rect,
        buffer: &mut Buffer,
        camera: &Camera<G::NodeId>,
        highlights: &Highlights<G::NodeId>,
    ) -> RenderedWindow<G::NodeId>
    where
        G: GraphBase + NodeIndexable,
        G::NodeId: Copy + Eq + Hash,
        V: NodeRenderer<G>,
    {
        let offset = anchor_offset(&self.geometry, graph, camera);
        let overlay_primitives = paint(self, area, buffer, offset, graph, visual, highlights);
        RenderedWindow {
            frame: frame_index(&self.geometry, graph, visual, offset, area),
            wormhole_stubs: wormhole_index(&self.geometry, graph, offset),
            overlay_primitives,
        }
    }

    /// Where `paint` would draw the point at `fraction` within `node` under `camera`, if the
    /// scene places `node` as a data node. Lets a caller correct the camera before painting.
    pub(crate) fn screen_point<G>(
        &self,
        graph: &G,
        camera: &Camera<G::NodeId>,
        node: G::NodeId,
        fraction: (f64, f64),
    ) -> Option<WorldPos>
    where
        G: GraphBase + NodeIndexable,
    {
        let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node));
        let node = data_node_placement(&self.geometry, node_index)?;
        let local = node.rect(WorldPos::ZERO).point_at_fraction(fraction);
        let offset = anchor_offset(&self.geometry, graph, camera);
        Some(WorldPos::new(local.x + offset.x, local.y + offset.y))
    }
}

/// The local position and size `geometry` gives the data node for `node_index`.
fn data_node_placement(geometry: &WindowGeometry, node_index: NodeIndex) -> Option<&LayoutNode> {
    geometry
        .graph
        .node_weights()
        .find(|node| matches!(node.role, NodeRole::Data(data_index) if data_index == node_index))
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
    let anchor_layout = data_node_placement(geometry, anchor_node_index);

    // An invisible anchor was placed as the routing node its edges meet at.
    let Some(node) = anchor_layout.or_else(|| geometry.junction_node(anchor_node_index)) else {
        return WorldPos::ZERO;
    };

    let anchor_point = node
        .rect(WorldPos::ZERO)
        .point_at_fraction(camera.anchor_fraction);

    WorldPos::new(
        camera.anchor_screen.0 - anchor_point.x,
        camera.anchor_screen.1 - anchor_point.y,
    )
}

/// Paint the window into `buffer`, resolving `highlights` from domain terms to screen positions
/// for this call only (nothing is baked into `routed_window_graph` beyond this paint).
fn paint<G, V>(
    scene: &WindowScene,
    area: Rect,
    buffer: &mut Buffer,
    offset: WorldPos,
    graph: &G,
    visual: &V,
    highlights: &Highlights<G::NodeId>,
) -> Vec<OverlayPrimitive>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash,
    V: NodeRenderer<G>,
{
    let routed_window_graph = &scene.routed_window_graph;
    let layout_mode = scene.layout_mode;
    let half_width = floor_half(area.width as i64);
    let half_height = floor_half(area.height as i64);
    let camera_current = WorldPos::new(half_width - offset.x, half_height - offset.y);

    let mut viewport_state = ViewportState::new();
    viewport_state.camera_current = camera_current;
    viewport_state.viewport_bounds = area;

    let mut node_highlights = Vec::new();
    let mut edge_highlights = Vec::new();
    let mut hairpin_highlights = Vec::new();
    let mut cell_highlights = Vec::new();
    for (kind, style) in &highlights.styles {
        match kind {
            HighlightKind::Node(node_id) => {
                apply_node_highlight(
                    routed_window_graph,
                    graph,
                    *node_id,
                    *style,
                    &mut node_highlights,
                );
            }
            HighlightKind::Edge(selector) => {
                apply_edge_highlight(
                    routed_window_graph,
                    graph,
                    *selector,
                    *style,
                    &mut edge_highlights,
                    &mut hairpin_highlights,
                );
            }
            HighlightKind::Cells {
                node,
                first_cell,
                last_cell,
            } => {
                apply_cell_highlight(
                    routed_window_graph,
                    graph,
                    *node,
                    *first_cell,
                    *last_cell,
                    *style,
                    &mut cell_highlights,
                );
            }
        }
    }

    let edge_lowlights =
        resolve_edge_lowlights(routed_window_graph, graph, &highlights.edge_lowlights);
    let node_lowlights =
        resolve_node_lowlights(routed_window_graph, graph, &highlights.node_lowlights);

    let theme = current_theme();
    let mut world_buffer = WorldBuffer::new(buffer, &viewport_state);
    plot_routed_window_graph_with_highlights(
        routed_window_graph,
        &mut world_buffer,
        visual,
        graph,
        PlotDecorations {
            node_highlights: &node_highlights,
            edge_highlights: &edge_highlights,
            hairpin_highlights: &hairpin_highlights,
            cell_highlights: &cell_highlights,
            lowlights: &edge_lowlights,
            node_lowlights: &node_lowlights,
        },
        &theme,
        layout_mode,
    )
}

fn apply_node_highlight<G: GraphBase + NodeIndexable>(
    routed_window_graph: &RoutedWindowGraph,
    graph: &G,
    node_id: G::NodeId,
    style: PathStyle,
    out: &mut Vec<(WorldPos, PathStyle)>,
) {
    let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
    if let Some(pos) = routed_window_graph.node_positions.get(&node_index) {
        out.push((*pos, style));
    }
}

/// Light the routed segments of the one domain edge `selector` names. Parallel edges between
/// the same two nodes take different routes (they leave and enter by different sides), so the
/// highlight follows the edge's own route rather than everything bundled under the node pair.
///
/// A pair-only selector picks the west-to-east edge between the nodes, or the only edge joining
/// them when none reads west to east, and nothing when that is ambiguous. A pair with no domain
/// edge in the window at all is a wormhole stub, which has no ports and is matched by its pair.
fn apply_edge_highlight<G: GraphBase + NodeIndexable>(
    routed_window_graph: &RoutedWindowGraph,
    graph: &G,
    selector: EdgeSelector<G::NodeId>,
    style: PathStyle,
    out: &mut Vec<((WorldPos, WorldPos), PathStyle)>,
    hairpin_out: &mut Vec<((NodeIndex, NodeSide), PathStyle)>,
) {
    let source = NodeIndex::new(<G as NodeIndexable>::to_index(graph, selector.source()));
    let target = NodeIndex::new(<G as NodeIndexable>::to_index(graph, selector.target()));

    let joins_pair = |ported_edge: &&PortedEdge<NodeIndex>| {
        (ported_edge.source, ported_edge.target) == (source, target)
            || (ported_edge.source, ported_edge.target) == (target, source)
    };
    if !routed_window_graph
        .domain_edges
        .iter()
        .any(|ported_edge| joins_pair(&ported_edge))
    {
        out.extend(bundled_segments(routed_window_graph, source, target, style));
        return;
    }

    let candidates: Vec<&PortedEdge<NodeIndex>> = routed_window_graph
        .domain_edges
        .iter()
        .filter(|ported_edge| (ported_edge.source, ported_edge.target) == (source, target))
        .collect();
    let chosen: Vec<&PortedEdge<NodeIndex>> = match selector {
        EdgeSelector::Exact(exact) => candidates
            .into_iter()
            .filter(|ported_edge| ported_edge.ports == exact.ports)
            .collect(),
        EdgeSelector::Between { .. } => {
            let forward: Vec<&PortedEdge<NodeIndex>> = candidates
                .iter()
                .copied()
                .filter(|ported_edge| ported_edge.ports == EdgePorts::default())
                .collect();
            if forward.is_empty() && candidates.len() == 1 {
                candidates
            } else {
                forward
            }
        }
    };

    for ported_edge in chosen {
        if ported_edge.source == ported_edge.target
            && ported_edge.ports.source == ported_edge.ports.target
            && routed_window_graph
                .node_positions
                .get(&source)
                .is_some_and(|position| {
                    routed_window_graph
                        .graph
                        .contains_edge(*position, *position)
                })
        {
            hairpin_out.push(((source, ported_edge.ports.source), style));
            continue;
        }
        match highlighted_edge_route(routed_window_graph, ported_edge) {
            Some(route) => out.extend(route.windows(2).map(|hop| ((hop[0], hop[1]), style))),
            // A junction's routing point has no sides to tell its edges apart by, so when the
            // ports cannot be followed through one, the pair's bundled segments stand in.
            None if routed_window_graph.junction_positions.contains_key(&source)
                || routed_window_graph.junction_positions.contains_key(&target) =>
            {
                out.extend(bundled_segments(routed_window_graph, source, target, style));
            }
            None => {}
        }
    }
}

/// Every routed segment carrying a domain edge between `source` and `target` in either direction.
fn bundled_segments(
    routed_window_graph: &RoutedWindowGraph,
    source: NodeIndex,
    target: NodeIndex,
    style: PathStyle,
) -> Vec<((WorldPos, WorldPos), PathStyle)> {
    routed_window_graph
        .edges()
        .filter(|(_, _, bundle)| {
            bundle.contains(&(source, target)) || bundle.contains(&(target, source))
        })
        .map(|(source_pos, target_pos, _)| ((source_pos, target_pos), style))
        .collect()
}

fn apply_cell_highlight<G: GraphBase + NodeIndexable>(
    routed_window_graph: &RoutedWindowGraph,
    graph: &G,
    node_id: G::NodeId,
    first_cell: (i64, i64),
    last_cell: (i64, i64),
    style: PathStyle,
    out: &mut Vec<CellHighlight>,
) {
    let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node_id));
    if let Some(&world_pos) = routed_window_graph.node_positions.get(&node_index) {
        out.push((world_pos, first_cell, last_cell, style));
    }
}

/// The domain nodes the window places, by id. Lowlights are recorded as the crawl loads edges,
/// so one can name a node the graph has since re-carved or dropped; looking nodes up through
/// the window instead of indexing into the graph leaves such stale entries with nothing to dim.
fn window_node_indices<G>(
    routed_window_graph: &RoutedWindowGraph,
    graph: &G,
) -> HashMap<G::NodeId, NodeIndex>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash,
{
    routed_window_graph
        .node_positions
        .keys()
        .chain(routed_window_graph.junction_positions.keys())
        .filter(|index| index.index() < graph.node_bound())
        .map(|&index| {
            (
                <G as NodeIndexable>::from_index(graph, index.index()),
                index,
            )
        })
        .collect()
}

fn resolve_edge_lowlights<G>(
    routed_window_graph: &RoutedWindowGraph,
    graph: &G,
    domain_lowlights: &[(G::NodeId, G::NodeId)],
) -> Vec<(WorldPos, WorldPos)>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash,
{
    let window_nodes = window_node_indices(routed_window_graph, graph);
    let lowlight_pairs: HashSet<(NodeIndex, NodeIndex)> = domain_lowlights
        .iter()
        .filter_map(|(source, target)| {
            let source = *window_nodes.get(source)?;
            let target = *window_nodes.get(target)?;
            Some(if source <= target {
                (source, target)
            } else {
                (target, source)
            })
        })
        .collect();

    routed_window_graph
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

fn resolve_node_lowlights<G>(
    routed_window_graph: &RoutedWindowGraph,
    graph: &G,
    domain_node_lowlights: &[G::NodeId],
) -> Vec<WorldPos>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash,
{
    let window_nodes = window_node_indices(routed_window_graph, graph);
    domain_node_lowlights
        .iter()
        .filter_map(|node_id| {
            let node_index = window_nodes.get(node_id)?;
            routed_window_graph.node_positions.get(node_index).copied()
        })
        .collect()
}

/// Build the `FrameIndex` over every placed Data node, with the row `visual` holds a cursor to
/// on each.
fn frame_index<G, V>(
    geometry: &WindowGeometry,
    graph: &G,
    visual: &V,
    offset: WorldPos,
    area: Rect,
) -> FrameIndex<G::NodeId>
where
    G: GraphBase + NodeIndexable,
    G::NodeId: Copy + Eq + Hash,
    V: NodeRenderer<G>,
{
    let placed: Vec<PlacedNode<G::NodeId>> = geometry
        .graph
        .node_weights()
        .filter_map(|node| {
            let NodeRole::Data(node_index) = node.role else {
                return None;
            };
            let node_id = <G as NodeIndexable>::from_index(graph, node_index.index());
            let rect = node.rect(offset);
            Some(PlacedNode {
                id: node_id,
                rect,
                layer: node.layer.unwrap_or_default(),
                cursor_row: visual.cursor_row(&node_id),
                cursor_columns: visual.cursor_columns(&node_id),
            })
        })
        .collect();

    let junctions = geometry
        .junctions()
        .map(|(node_index, node)| {
            (
                <G as NodeIndexable>::from_index(graph, node_index.index()),
                WorldPos::new(node.pos.x + offset.x, node.pos.y + offset.y),
            )
        })
        .collect();

    let clip_area = WorldRect::from_coords(
        0,
        0,
        (area.width as i64).saturating_sub(1),
        (area.height as i64).saturating_sub(1),
    );
    FrameIndex::build(placed, clip_area).with_junctions(junctions)
}

/// Every placed `NodeRole::Wormhole` (wormhole) stub's screen rect, the boundary node whose
/// edge produced it, and its off-screen domain target - the click-to-teleport counterpart of
/// `frame_index`'s Data-only index. Sourced from the same `wormhole_doors` already fed to the
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
            let rect = placed.rect(offset);
            Some((rect, boundary_id, target_id))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use petgraph::graph::NodeIndex;
    use ratatui::{
        buffer::Buffer,
        layout::Rect,
        style::{Color, Style},
    };

    use super::*;
    use crate::{
        edge_ports::PortedEdge,
        geometry::{LocalPos, WorldRect},
        layout::{LayoutEdge, LayoutNode, WindowGeometry},
        layout_engine::LayoutEngine,
        testing::mocks::{MockDomainGraph, TestGraphs},
    };

    /// Pure ephemeral graph painter for tests: renders one `AssembledLayout` window into a buffer
    /// and returns the `RenderedWindow` describing where everything landed. Never touches
    /// `LayoutEngine` or any cursor/view state - callers own the anchor and the highlight
    /// lists and pass them in fresh every call.
    struct GraphPainter<'a, G, V>
    where
        G: GraphBase,
    {
        window: AssembledLayout,
        graph: &'a G,
        visual: &'a V,
        gap_policy: GapPolicy,
        layout_mode: LayoutMode,
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
        fn new(window: AssembledLayout, graph: &'a G, visual: &'a V) -> Self {
            Self {
                window,
                graph,
                visual,
                gap_policy: GapPolicy::default(),
                layout_mode: LayoutMode::default(),
            }
        }

        /// Set the minimum inter-node gap for each axis (the current zoom level's target).
        fn spacing(mut self, gap_policy: &GapPolicy) -> Self {
            self.gap_policy = *gap_policy;
            self
        }

        /// Choose how edges are routed and drawn. Defaults to [`LayoutMode::Rectilinear`].
        fn layout_mode(mut self, layout_mode: LayoutMode) -> Self {
            self.layout_mode = layout_mode;
            self
        }

        /// Size every node via `visual`, route and compact the window, place it so the camera's
        /// anchor pixel lands at `camera.anchor_screen`, paint the visible parts into `buffer`, and
        /// return a [`RenderedWindow`]: a `FrameIndex` over every placed Data node (including
        /// ones outside `area`), a flat list of every placed wormhole (`NodeRole::Wormhole`)
        /// stub's screen rect, boundary node, and off-screen domain target - kept separate from
        /// `FrameIndex` since its `N` key space is domain node ids only and wormhole stubs are
        /// synthetic routing nodes with no domain id of their own - and the overlay primitives
        /// of a curved layout mode.
        fn render(
            self,
            area: Rect,
            buffer: &mut Buffer,
            camera: &Camera<G::NodeId>,
            highlights: &Highlights<G::NodeId>,
        ) -> RenderedWindow<G::NodeId> {
            let scene = WindowScene::new(
                self.window,
                self.graph,
                self.visual,
                &self.gap_policy,
                self.layout_mode,
            );
            scene.paint(self.graph, self.visual, area, buffer, camera, highlights)
        }
    }

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
    fn test_lowlights_naming_nodes_missing_from_the_graph_dim_nothing() {
        // A lowlight recorded for a slice the crawl later re-carved names a node the graph no
        // longer holds. Painting must skip it instead of indexing into the graph.
        let mut graph = petgraph::graphmap::DiGraphMap::<u32, ()>::new();
        graph.add_node(10);
        graph.add_node(20);
        let first = NodeIndex::new(<_ as NodeIndexable>::to_index(&graph, 10));
        let second = NodeIndex::new(<_ as NodeIndexable>::to_index(&graph, 20));
        let mut routed_window_graph = RoutedWindowGraph::empty();
        routed_window_graph
            .node_positions
            .insert(first, WorldPos::new(0, 0));
        routed_window_graph
            .node_positions
            .insert(second, WorldPos::new(8, 0));

        let node_lowlights = resolve_node_lowlights(&routed_window_graph, &graph, &[10, 99]);
        let edge_lowlights =
            resolve_edge_lowlights(&routed_window_graph, &graph, &[(10, 99), (99, 20)]);

        assert_eq!(node_lowlights, vec![WorldPos::new(0, 0)]);
        assert!(edge_lowlights.is_empty());
    }

    #[test]
    fn test_curved_hairpin_highlight_keeps_its_side() {
        let mut graph = MockDomainGraph::new();
        let domain_index = graph.add_node(());
        let position = WorldPos::new(0, 0);
        let mut routed_window_graph = RoutedWindowGraph::empty();
        routed_window_graph
            .node_positions
            .insert(domain_index, position);
        routed_window_graph
            .graph
            .add_edge(position, position, vec![(domain_index, domain_index)]);
        routed_window_graph.domain_edges = [NodeSide::West, NodeSide::East]
            .map(|side| PortedEdge {
                source: domain_index,
                target: domain_index,
                ports: EdgePorts::new(side, side),
            })
            .into();

        for side in [NodeSide::West, NodeSide::East] {
            let mut ordinary = Vec::new();
            let mut hairpins = Vec::new();
            apply_edge_highlight(
                &routed_window_graph,
                &graph,
                EdgeSelector::Exact(PortedEdge::new(
                    domain_index,
                    domain_index,
                    EdgePorts::new(side, side),
                )),
                PathStyle::new(Color::Red),
                &mut ordinary,
                &mut hairpins,
            );
            assert!(ordinary.is_empty());
            assert_eq!(
                hairpins,
                vec![((domain_index, side), PathStyle::new(Color::Red))]
            );
        }
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
        let RenderedWindow { frame, .. } = painter.render(area, &mut buffer, &camera, &highlights);

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
        let RenderedWindow { frame, .. } = painter.render(area, &mut buffer, &camera, &highlights);

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
    fn test_wormhole_index_distinguishes_stubs_with_the_same_target() {
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
            junctions: Vec::new(),
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

    mod curved_edges {
        use ratatui::{buffer::Buffer, layout::Rect};

        use super::*;

        struct BoxVisual;

        impl NodeRenderer<MockDomainGraph> for BoxVisual {
            fn get_node_size(&self, _node: &NodeIndex) -> (u64, u64) {
                (5, 1)
            }

            fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &NodeIndex) {
                let label = format!("[{:^3}]", node_id.index());
                buffer.set_string_styled(area.min, &label, Style::default());
            }
        }

        fn render_text(
            graph: MockDomainGraph,
            layout_mode: LayoutMode,
            area: Rect,
        ) -> (String, Vec<OverlayPrimitive>) {
            let mut engine = LayoutEngine::new(graph);
            let anchor = engine.default_anchor().expect("should have graph nodes");
            let window = engine
                .window_for(anchor, 200)
                .expect("should build a window");
            let camera = Camera {
                anchor,
                anchor_fraction: (0.5, 0.5),
                anchor_screen: (i64::from(area.width) / 2, i64::from(area.height) / 2),
                hard_zone: 2,
            };
            let mut buffer = Buffer::empty(area);
            let RenderedWindow {
                overlay_primitives, ..
            } = GraphPainter::new(window, engine.graph(), &BoxVisual)
                .spacing(&GapPolicy {
                    data_data_x: |_| 2,
                    data_data_y: |_| 1,
                    data_routing_x: |_| 1,
                    data_routing_y: |_| 0,
                    routing_routing_x: |_| 0,
                    routing_routing_y: |_| 0,
                })
                .layout_mode(layout_mode)
                .render(area, &mut buffer, &camera, &Highlights::default());
            let text = (0..area.height)
                .map(|row| {
                    (0..area.width)
                        .map(|column| buffer[(column, row)].symbol().to_string())
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join("\n");
            (text, overlay_primitives)
        }

        #[test]
        fn test_curved_edges_report_overlay_primitives_instead_of_drawing_them() {
            let (text, overlay_primitives) = render_text(
                TestGraphs::domain_complex_dag(),
                LayoutMode::Curved,
                Rect::new(0, 0, 100, 30),
            );
            assert!(
                !text
                    .chars()
                    .any(|glyph| matches!(glyph, '\u{2800}'..='\u{28FF}')),
                "no edge dots should be drawn into the buffer:\n{text}"
            );
            assert!(!text.contains(['─', '│', '┼', '├', '┤', '┬', '┴']));
            assert!(
                overlay_primitives
                    .iter()
                    .any(|primitive| matches!(primitive, OverlayPrimitive::Bezier(_)))
            );
        }

        #[test]
        fn test_curved_edges_leave_forward_horizontal_routes_unmarked() {
            let (_, overlay_primitives) = render_text(
                TestGraphs::domain_simple_chain(),
                LayoutMode::Curved,
                Rect::new(0, 0, 100, 15),
            );
            let arrows = overlay_primitives
                .iter()
                .filter(|primitive| matches!(primitive, OverlayPrimitive::Arrow(_)))
                .count();
            assert_eq!(
                arrows, 0,
                "left-to-right edges are implied and carry no marker"
            );
            let (rectilinear, _) = render_text(
                TestGraphs::domain_simple_chain(),
                LayoutMode::Rectilinear,
                Rect::new(0, 0, 100, 15),
            );
            assert!(!rectilinear.contains('▶'));
        }

        #[test]
        fn test_curve_leaves_a_node_from_its_border_at_its_row() {
            let area = Rect::new(0, 0, 60, 10);
            let (text, overlay_primitives) =
                render_text(TestGraphs::domain_simple_chain(), LayoutMode::Curved, area);
            assert!(
                overlay_primitives.len() >= 6,
                "side ports should split the chain edges:\n{text}"
            );
            let row_of = |label: &str| {
                text.lines()
                    .position(|line| line.contains(label))
                    .expect("should draw the node") as f64
                    + 0.5
            };
            let right_border = |label: &str| {
                let row = text.lines().find(|line| line.contains(label)).unwrap();
                (row.find(label).unwrap() + label.len()) as f64
            };
            let first = overlay_primitives
                .iter()
                .find_map(|edge| match edge {
                    OverlayPrimitive::Bezier(curve)
                        if (curve.from.0 - right_border("[ 0 ]")).abs() < 1e-9 =>
                    {
                        Some(curve)
                    }
                    _ => None,
                })
                .expect("a curve should leave the first node's right border");
            assert_eq!(first.from.1, row_of("[ 0 ]"));
        }

        #[test]
        fn test_curved_geometry_keeps_data_layers_in_columns_with_side_ports() {
            let graph = TestGraphs::domain_strut(4, 3);
            let mut engine = LayoutEngine::new(graph);
            let anchor = engine.default_anchor().expect("should have graph nodes");
            let window = engine
                .window_for(anchor, 200)
                .expect("should build a window");
            let assembled_nodes = window.graph.node_count();
            let assembled_edges = window.graph.edge_count();
            let geometry = build_window_geometry_with_anchors(
                window,
                &GapPolicy::default(),
                LayoutMode::Curved,
                |role| match role {
                    NodeRole::Data(_) => (5, 1),
                    _ => (1, 1),
                },
                |_| true,
                |_| false,
                |_| crate::layout::VerticalAnchor::Center,
            );
            assert!(geometry.graph.node_count() > assembled_nodes);
            assert!(geometry.graph.edge_count() > assembled_edges);
            let mut column_of_layer = HashMap::new();
            for node in geometry.graph.node_weights() {
                if let NodeRole::Data(_) = node.role
                    && let Some(layer) = node.layer
                {
                    assert_eq!(
                        *column_of_layer.entry(layer).or_insert(node.pos.x),
                        node.pos.x,
                        "layer {layer} should stay in one column"
                    );
                }
            }
            for edge in geometry.graph.edge_indices() {
                let (source, target) = geometry.graph.edge_endpoints(edge).unwrap();
                let (left, right) = (&geometry.graph[source], &geometry.graph[target]);
                assert!(
                    (left.pos.x - right.pos.x).abs()
                        >= i64::try_from((left.size.0 + right.size.0) / 2).unwrap(),
                    "an edge's ends should not overlap horizontally"
                );
            }
        }

        #[test]
        fn test_curved_side_ports_clear_widest_node_in_each_layer() {
            let mut engine = LayoutEngine::new(TestGraphs::domain_extended_diamond());
            let anchor = engine.default_anchor().expect("should have graph nodes");
            let window = engine
                .window_for(anchor, 200)
                .expect("should build a window");
            let geometry = build_window_geometry_with_anchors(
                window,
                &GapPolicy::default(),
                LayoutMode::Curved,
                |role| match role {
                    NodeRole::Data(index) if index.index() % 2 == 0 => (11, 1),
                    NodeRole::Data(_) => (3, 1),
                    _ => (1, 1),
                },
                |_| true,
                |_| false,
                |_| crate::layout::VerticalAnchor::Center,
            );
            let mut extents_by_layer = HashMap::<i32, (i64, i64, i64)>::new();
            for node in geometry.graph.node_weights() {
                if let NodeRole::Data(_) = node.role
                    && let Some(layer) = node.layer
                {
                    let rect = node.rect(crate::geometry::WorldPos::ZERO);
                    let extents = extents_by_layer
                        .entry(layer)
                        .or_insert((node.pos.x, rect.min.x, rect.max.x));
                    assert_eq!(extents.0, node.pos.x);
                    extents.1 = extents.1.min(rect.min.x);
                    extents.2 = extents.2.max(rect.max.x);
                }
            }
            for node in geometry.graph.node_weights() {
                if node.side_port_offset != 0 {
                    let (center, left, right) = extents_by_layer[&node.layer.unwrap()];
                    assert_eq!(node.pos.x, center + node.side_port_offset);
                    let anchor = node
                        .side_port_anchor
                        .expect("should link a side port to its data node");
                    assert_eq!(node.pos.y, geometry.graph[anchor].pos.y);
                    if node.side_port_offset < 0 {
                        assert_eq!(node.pos.x, left - 2);
                    } else {
                        assert_eq!(node.pos.x, right + 2);
                    }
                }
            }
        }

        #[test]
        fn test_curved_dummies_get_side_ports_like_data_nodes() {
            let mut engine = LayoutEngine::new(TestGraphs::domain_skip_layer());
            let anchor = engine.default_anchor().expect("should have graph nodes");
            let window = engine
                .window_for(anchor, 200)
                .expect("should build a window");
            let geometry = build_window_geometry_with_anchors(
                window,
                &GapPolicy::default(),
                LayoutMode::Curved,
                |role| match role {
                    NodeRole::Data(_) => (5, 1),
                    _ => (1, 1),
                },
                |_| true,
                |_| false,
                |_| crate::layout::VerticalAnchor::Center,
            );
            let graph = &geometry.graph;
            let dummies: Vec<_> = graph
                .node_indices()
                .filter(|&index| {
                    graph[index].role == NodeRole::Routing
                        && graph[index].side_port_anchor.is_none()
                })
                .collect();
            assert!(!dummies.is_empty(), "the skip edge should leave a dummy");
            for dummy in dummies {
                let ports: Vec<_> = graph
                    .node_indices()
                    .filter(|&index| graph[index].side_port_anchor == Some(dummy))
                    .collect();
                assert_eq!(ports.len(), 2, "a dummy should get a left and a right port");
                for port in ports {
                    assert_eq!(graph[port].pos.y, graph[dummy].pos.y);
                    assert_eq!(
                        graph[port].pos.x,
                        graph[dummy].pos.x + graph[port].side_port_offset
                    );
                    assert!(graph[port].side_port_offset.abs() >= 2);
                }
            }
        }

        #[test]
        fn test_curved_loopback_pins_keep_their_data_anchor() {
            let mut graph = MockDomainGraph::new();
            let nodes: Vec<_> = (0..4).map(|_| graph.add_node(())).collect();
            for pair in nodes.windows(2) {
                graph.add_edge(pair[0], pair[1], ());
            }
            graph.add_edge(nodes[3], nodes[0], ());
            let mut engine = LayoutEngine::new_with_backward_edges(graph, &[(nodes[3], nodes[0])]);
            let window = engine
                .window_for(nodes[0], 200)
                .expect("should build a window");
            let geometry = build_window_geometry_with_anchors(
                window,
                &GapPolicy::default(),
                LayoutMode::Curved,
                |role| match role {
                    NodeRole::Data(_) => (5, 1),
                    _ => (1, 1),
                },
                |_| true,
                |_| false,
                |_| crate::layout::VerticalAnchor::Center,
            );
            let pins: Vec<_> = geometry
                .graph
                .node_indices()
                .filter(|&index| {
                    matches!(geometry.graph[index].role, NodeRole::Pin(side) if side.is_loopback())
                })
                .collect();
            assert!(
                !pins.is_empty(),
                "the backward edge should create loopback pins"
            );
            for pin in pins {
                assert!(
                    geometry
                        .graph
                        .neighbors(pin)
                        .any(|index| matches!(geometry.graph[index].role, NodeRole::Data(_))),
                    "a loopback pin needs a direct data anchor for its turnaround"
                );
            }
        }

        #[test]
        fn test_wider_drop_gets_wider_column_gap() {
            let layer_gap = |width: usize| {
                let mut engine = LayoutEngine::new(TestGraphs::domain_strut(2, width));
                let anchor = engine.default_anchor().expect("should have graph nodes");
                let window = engine
                    .window_for(anchor, 200)
                    .expect("should build a window");
                let geometry = build_window_geometry_with_anchors(
                    window,
                    &GapPolicy::default(),
                    LayoutMode::Curved,
                    |role| match role {
                        NodeRole::Data(_) => (5, 1),
                        _ => (1, 1),
                    },
                    |_| true,
                    |_| false,
                    |_| crate::layout::VerticalAnchor::Center,
                );
                let columns: std::collections::BTreeSet<i64> = geometry
                    .graph
                    .node_weights()
                    .map(|node| node.pos.x)
                    .collect();
                columns.last().unwrap() - columns.first().unwrap()
            };
            assert!(layer_gap(6) > layer_gap(1));
        }
    }

    mod parallel_edge_highlights {
        use petgraph::graph::EdgeIndex;
        use ratatui::style::Color;

        use super::*;
        use crate::edge_ports::{EdgePorts, NodeSide, PortedEdge};

        const EAST_TO_EAST: EdgePorts = EdgePorts::new(NodeSide::East, NodeSide::East);
        const WEST_TO_WEST: EdgePorts = EdgePorts::new(NodeSide::West, NodeSide::West);

        /// The first parallel edge reads west to east; the second leaves and enters by the east
        /// side, so it detours around its target.
        fn ports_by_edge_index(_graph: &MockDomainGraph, edge: &EdgeIndex) -> EdgePorts {
            if edge.index() == 0 {
                EdgePorts::default()
            } else {
                EAST_TO_EAST
            }
        }

        fn both_sides_west(_graph: &MockDomainGraph, edge: &EdgeIndex) -> EdgePorts {
            if edge.index() == 0 {
                WEST_TO_WEST
            } else {
                EAST_TO_EAST
            }
        }

        fn only_east_to_east(_graph: &MockDomainGraph, _edge: &EdgeIndex) -> EdgePorts {
            EAST_TO_EAST
        }

        /// The cells painted red when `selector` is highlighted on two nodes joined by two
        /// parallel edges whose ports `ports_of` assigns.
        fn red_cells(
            ports_of: fn(&MockDomainGraph, &EdgeIndex) -> EdgePorts,
            edge_count: usize,
            selector: impl Fn(NodeIndex, NodeIndex) -> EdgeSelector<NodeIndex>,
        ) -> HashSet<(u16, u16)> {
            let mut domain_graph = MockDomainGraph::new();
            let source = domain_graph.add_node(());
            let target = domain_graph.add_node(());
            for _ in 0..edge_count {
                domain_graph.add_edge(source, target, ());
            }
            assert_eq!(
                domain_graph.edge_count(),
                edge_count,
                "should keep every parallel edge"
            );
            let mut engine = LayoutEngine::new(domain_graph);
            engine.set_edge_ports(ports_of);
            let anchor = engine.default_anchor().expect("should have graph nodes");
            let window = engine
                .window_for(anchor, 20)
                .expect("should build a window");
            let visual = FixedSizeVisual { size: (5, 1) };
            let camera = Camera {
                anchor,
                anchor_fraction: (0.5, 0.5),
                anchor_screen: (20, 10),
                hard_zone: 2,
            };
            let highlights = Highlights {
                styles: vec![(
                    HighlightKind::Edge(selector(source, target)),
                    PathStyle::new(Color::Red),
                )],
                ..Highlights::default()
            };
            let area = Rect::new(0, 0, 60, 20);
            let mut buffer = Buffer::empty(area);
            GraphPainter::new(window, engine.graph(), &visual).render(
                area,
                &mut buffer,
                &camera,
                &highlights,
            );
            let mut cells = HashSet::new();
            for y in 0..area.height {
                for x in 0..area.width {
                    if buffer[(x, y)].fg == Color::Red {
                        cells.insert((x, y));
                    }
                }
            }
            cells
        }

        fn exact(ports: EdgePorts) -> impl Fn(NodeIndex, NodeIndex) -> EdgeSelector<NodeIndex> {
            move |source, target| EdgeSelector::Exact(PortedEdge::new(source, target, ports))
        }

        fn between(source: NodeIndex, target: NodeIndex) -> EdgeSelector<NodeIndex> {
            EdgeSelector::Between { source, target }
        }

        #[test]
        fn test_exact_selector_lights_only_its_own_parallel_edge() {
            let forward = red_cells(ports_by_edge_index, 2, exact(EdgePorts::default()));
            let detour = red_cells(ports_by_edge_index, 2, exact(EAST_TO_EAST));

            assert!(!forward.is_empty(), "the forward edge should light up");
            assert!(!detour.is_empty(), "the detouring edge should light up");
            assert!(
                !detour.is_subset(&forward) && !forward.is_subset(&detour),
                "each parallel edge should light cells the other does not"
            );
        }

        #[test]
        fn test_node_pair_lights_the_forward_edge_and_not_its_parallel_edges() {
            let forward = red_cells(ports_by_edge_index, 2, exact(EdgePorts::default()));
            let pair = red_cells(ports_by_edge_index, 2, between);

            assert_eq!(
                pair, forward,
                "a bare node pair should light the west-to-east edge alone"
            );
        }

        #[test]
        fn test_node_pair_lights_the_only_edge_when_none_is_west_to_east() {
            let only = red_cells(only_east_to_east, 1, exact(EAST_TO_EAST));
            let pair = red_cells(only_east_to_east, 1, between);

            assert!(!only.is_empty());
            assert_eq!(pair, only);
        }

        #[test]
        fn test_node_pair_lights_nothing_when_parallel_edges_are_all_non_default() {
            let pair = red_cells(both_sides_west, 2, between);

            assert!(
                pair.is_empty(),
                "should not guess between two non-default parallel edges"
            );
        }

        #[test]
        fn test_exact_selector_for_an_absent_edge_lights_nothing() {
            let missing = red_cells(ports_by_edge_index, 2, exact(WEST_TO_WEST));

            assert!(missing.is_empty());
        }
    }
}
