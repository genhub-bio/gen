use std::{collections::HashMap, hash::Hash};

#[cfg(feature = "crossterm")]
use crossterm::event::{KeyCode, KeyEvent};
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
    style::{Color, Style},
    widgets::{Block, StatefulWidget, Widget},
};

use crate::{
    assembly::AssembledLayout,
    crawl::{EagerSource, GraphSource},
    distribute_nodes::GapSizes,
    frame_index::{Direction, FrameIndex},
    graph_painter::{Camera, GraphPainter, HighlightKind, Highlights, snap_camera},
    layout::NodeRole,
    layout_engine::{BatchId, LayoutEngine},
    navigator::{CursorOverlay, CursorState, Navigator},
    plotter::{NodeRenderer, PathStyle},
    theme::current_theme,
};

const DEFAULT_HARD_ZONE: u16 = 2;

/// A domain node's local (pre-offset) position within an already-assembled window, if it's
/// present as a `NodeRole::Data` node there. Used by the `go_to_node_framed` go-to path to
/// compute a screen-position delta between two nodes in the same window without needing a full
/// paint pass (`GraphPainter`'s `anchor_offset` does the equivalent lookup, but only for the
/// camera's own anchor and only as part of painting).
fn node_pos_in_window<G>(window: &AssembledLayout, graph: &G, node: G::NodeId) -> Option<(i64, i64)>
where
    G: GraphBase + NodeIndexable,
{
    let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node));
    window
        .graph
        .node_weights()
        .find_map(|layout_node| match layout_node.role {
            NodeRole::Data(domain_index) if domain_index == node_index => {
                Some((layout_node.pos.x, layout_node.pos.y))
            }
            _ => None,
        })
}

/// A domain node's Sugiyama rank (layer) within an already-assembled window, if it's present
/// as a `NodeRole::Data` node there. `probe.graph`'s own `pos.x` is a layer ordinal, not a
/// pixel column (see `AssembledLayout`'s doc comment), so callers that need real pixel widths
/// pair this with `rank_widths`/`content_width_left_of_rank` instead of reading `pos` directly.
fn node_rank_in_window<G>(probe: &AssembledLayout, graph: &G, node: G::NodeId) -> Option<i32>
where
    G: GraphBase + NodeIndexable,
{
    let node_index = NodeIndex::new(<G as NodeIndexable>::to_index(graph, node));
    probe
        .graph
        .node_weights()
        .find_map(|layout_node| match layout_node.role {
            NodeRole::Data(domain_index) if domain_index == node_index => layout_node.layer,
            _ => None,
        })
}

/// The widest node's pixel width (a dummy's size, for routing/pin roles) per Sugiyama rank
/// present in `probe`. Used to estimate real pixel-column widths from `probe`'s rank-only
/// `AssembledLayout` before the real routing/compaction pass has run - see
/// `content_width_left_of_rank`.
fn rank_widths<G, V>(probe: &AssembledLayout, graph: &G, visual: &V) -> HashMap<i32, u64>
where
    G: NodeIndexable,
    V: NodeRenderer<G>,
{
    let mut width_by_rank: HashMap<i32, u64> = HashMap::new();
    for node in probe.graph.node_weights() {
        let Some(rank) = node.layer else { continue };
        let width = match node.role {
            NodeRole::Data(node_index) => {
                let node_id = <G as NodeIndexable>::from_index(graph, node_index.index());
                visual.get_node_size(&node_id).0
            }
            _ => visual.get_dummy_size().0,
        };
        width_by_rank
            .entry(rank)
            .and_modify(|current| *current = (*current).max(width))
            .or_insert(width);
    }
    width_by_rank
}

/// Estimate the pixel width of complete ranks before `target_rank`.
fn content_width_left_of_rank(
    width_by_rank: &HashMap<i32, u64>,
    min_rank: i32,
    target_rank: i32,
) -> i64 {
    (min_rank..target_rank)
        .map(|rank| width_by_rank.get(&rank).copied().unwrap_or(1) as i64 + 1)
        .sum()
}

/// Shift a target right just enough to avoid clipping its first partially visible preceding rank.
///
/// The shift is bounded to one rank so distant preceding content cannot push the target off-screen.
fn min_shift_to_avoid_left_clip(
    width_by_rank: &HashMap<i32, u64>,
    min_rank: i32,
    target_rank: i32,
    left_of_col_x: i64,
) -> i64 {
    let mut shift = if left_of_col_x < 0 { -left_of_col_x } else { 0 };
    let boundary = left_of_col_x + shift;

    let mut cumulative = 0i64;
    let mut rank = target_rank - 1;
    while rank >= min_rank {
        let width = width_by_rank.get(&rank).copied().unwrap_or(1) as i64 + 1;
        if cumulative + width > boundary {
            let hidden = cumulative + width - boundary;
            if hidden < width {
                shift += hidden;
            }
            break;
        }
        cumulative += width;
        rank -= 1;
    }
    shift
}

/// Navigation, camera, zoom, and highlighting state retained between renders.
///
/// `zoom_index` is an opaque position into whatever renderer/`GapSizes` table the owning
/// widget/event loop maintains - `gen-tui` has no notion of which concrete zoom levels
/// exist; it only tracks which position in that (external) table this view is currently
/// showing, and applies `gaps` as told.
#[derive(Clone)]
pub struct GraphViewState<N> {
    pub frame: FrameIndex<N>,
    /// Every placed wormhole (`NodeRole::Wormhole`) stub's screen rect, boundary node, and
    /// off-screen domain target from the last render - the click-to-teleport counterpart of
    /// `frame`, kept separate since `FrameIndex`'s key space is domain node ids only. See
    /// `wormhole_hit`.
    pub wormhole: Vec<(crate::geometry::WorldRect, N, N)>,
    pub cursor: CursorState<N>,
    pub camera: Option<Camera<N>>,
    pub zoom_index: usize,
    /// Minimum inter-node gap for each axis - the current zoom level's target distance,
    /// enforced by compaction (see `distribute_nodes::compact_layout`).
    pub gaps: GapSizes,
    pub highlights: Highlights<N>,
    /// The area passed to the most recent `GraphView::render` call. Used for click-to-world
    /// conversion and the `snap_camera` hard-zone boundary - facts knowable only from the
    /// last time this view was actually rendered, not a cached geometry.
    last_area: Rect,
    go_to_pending: bool,
    go_to_snap_left: bool,
    go_to_snap_right: bool,
    /// Batch whose world produced `frame` and `wormhole`.
    last_batch: Option<BatchId>,
    /// Cursor state restored if an explicit jump cannot construct its requested world.
    go_to_previous_cursor: Option<CursorState<N>>,
    /// Most recent wormhole arrival node. Painted with the theme's Base0B color until the
    /// next wormhole traversal replaces it.
    wormhole_entry: Option<N>,
    /// Independent overview views center each newly observed active world.
    center_world_on_change: bool,
    /// Set by `go_to_node_framed`: selects an independent camera anchor inside the active world
    /// while the cursor node is framed at the requested screen position.
    go_to_frame_anchor: Option<N>,
}

impl<N> Default for GraphViewState<N> {
    fn default() -> Self {
        Self {
            frame: FrameIndex::empty(),
            wormhole: Vec::new(),
            cursor: CursorState::default(),
            camera: None,
            zoom_index: 0,
            gaps: GapSizes::default(),
            highlights: Highlights::default(),
            last_area: Rect::new(0, 0, 0, 0),
            go_to_pending: false,
            go_to_frame_anchor: None,
            go_to_snap_left: false,
            go_to_snap_right: false,
            last_batch: None,
            go_to_previous_cursor: None,
            wormhole_entry: None,
            center_world_on_change: false,
        }
    }
}

impl<N> GraphViewState<N> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<N: Copy + Eq + Hash + Ord> GraphViewState<N> {
    /// The width of the area passed to the most recent `GraphView::render` call. Lets a caller
    /// outside `render()` compute the same fixed budget for an explicit world activation.
    pub fn last_area_width(&self) -> u16 {
        self.last_area.width
    }

    pub fn last_batch(&self) -> Option<BatchId> {
        self.last_batch
    }

    /// Discard per-frame products so this state can be reused as an independent view of the
    /// controller's active world.
    pub fn reset_render_state(&mut self) {
        self.frame = FrameIndex::empty();
        self.wormhole.clear();
        self.camera = None;
        self.last_batch = None;
        self.wormhole_entry = None;
        self.center_world_on_change = true;
    }

    /// Returns true when the cursor is visible (keyboard-driven mode).
    pub fn is_cursor_visible(&self) -> bool {
        self.cursor.visible
    }

    /// Hide the cursor and let the camera move freely. The cursor still tracks its node so
    /// it is ready when switching back.
    pub fn hide_cursor(&mut self) {
        self.cursor.visible = false;
    }

    /// Show the cursor and re-enable camera-following.
    pub fn show_cursor(&mut self) {
        self.cursor.visible = true;
    }

    /// Mark the node at which the most recent wormhole traversal arrived.
    pub fn mark_wormhole_entry(&mut self, node: N) {
        self.wormhole_entry = Some(node);
    }

    /// Return the most recent wormhole arrival node, if any.
    pub fn wormhole_entry(&self) -> Option<N> {
        self.wormhole_entry
    }

    /// Jump to a specific node at a fractional offset within it. Shows the cursor and queues
    /// a one-shot viewport centering for the next render (see `GraphView::render`'s go-to
    /// handling).
    pub fn go_to_node(&mut self, node: N, offset: (f64, f64)) {
        if !self.go_to_pending {
            self.go_to_previous_cursor = Some(self.cursor);
        }
        self.cursor.set_node(node, offset);
        self.cursor.visible = true;
        self.go_to_pending = true;
    }

    /// Like `go_to_node`, but uses `camera_anchor` for camera translation while framing
    /// `frame_node`. Both nodes should belong to the controller's active world.
    pub fn go_to_node_framed(&mut self, camera_anchor: N, frame_node: N, offset: (f64, f64)) {
        if !self.go_to_pending {
            self.go_to_previous_cursor = Some(self.cursor);
        }
        self.cursor.set_node(frame_node, offset);
        self.cursor.visible = true;
        self.go_to_pending = true;
        self.go_to_frame_anchor = Some(camera_anchor);
    }

    /// Request that the next pending go-to snap left rather than center. Call immediately
    /// after `go_to_node`.
    pub fn queue_snap_left(&mut self) {
        self.go_to_snap_left = true;
    }

    /// Request that the next pending go-to snap right rather than center. Call immediately
    /// after `go_to_node`. Mirrors `queue_snap_left` for the opposite edge - e.g. a wormhole
    /// jump that exits a window toward a lower rank should enter the new window from its right
    /// edge, matching the direction you'd naturally keep moving in.
    pub fn queue_snap_right(&mut self) {
        self.go_to_snap_right = true;
    }

    /// Pan the camera by a drag delta expressed in terminal coordinates, hiding the cursor
    /// (free-camera mode). The Y-axis flip (world Y+ is up, terminal Y+ is down) is handled
    /// here so callers can pass raw terminal deltas directly, matching `terminal_to_screen`
    /// (X unflipped, Y flipped). Applied immediately, no easing.
    pub fn move_by_terminal(&mut self, terminal_dx: i16, terminal_dy: i16) {
        self.hide_cursor();
        let world_dx = terminal_dx as i64;
        let world_dy = -(terminal_dy as i64);
        if let Some(camera) = &mut self.camera {
            camera.anchor_screen.0 += world_dx;
            camera.anchor_screen.1 += world_dy;
        }
    }

    /// Query (no mutation) whether a click at the given terminal coordinates hits a wormhole
    /// (`NodeRole::Wormhole`) stub, returning `(boundary_node, target_node)` if so - the local
    /// node the stub is attached to, and the off-screen domain node it leads to. Callers that
    /// want click-to-teleport should check this *before* `handle_click` - a hit here means the
    /// click needs engine-level handling (which world to load, forced inclusion, etc., none of
    /// which `GraphViewState` has access to) instead of ordinary cursor placement.
    pub fn wormhole_hit(&self, terminal_x: u16, terminal_y: u16) -> Option<(N, N)> {
        let screen = self.terminal_to_screen(terminal_x, terminal_y)?;
        let point = crate::geometry::Point::new(screen.0, screen.1);
        self.wormhole.iter().find_map(|&(rect, boundary, target)| {
            rect.contains(point).then_some((boundary, target))
        })
    }

    /// Query (without mutation) the domain node at the given terminal coordinates and the
    /// clicked fractional position within it. Coordinates are interpreted relative to this
    /// state's most recently rendered area.
    pub fn node_hit(&self, terminal_x: u16, terminal_y: u16) -> Option<(N, (f64, f64))> {
        let screen = self.terminal_to_screen(terminal_x, terminal_y)?;
        self.frame
            .hit(crate::geometry::Point::new(screen.0, screen.1))
    }

    /// Handle a click at the given terminal coordinates.
    ///
    /// - If a placed node occupies the clicked cell: places the cursor on that node,
    ///   switches to cursor-anchored mode, and returns `true`.
    /// - Otherwise: switches to free-camera mode and returns `false`.
    pub fn handle_click(&mut self, terminal_x: u16, terminal_y: u16) -> bool {
        match self.node_hit(terminal_x, terminal_y) {
            Some((node, frac)) => {
                self.cursor.set_node(node, frac);
                self.show_cursor();
                self.rebase_camera_to_cursor();
                true
            }
            None => {
                self.hide_cursor();
                false
            }
        }
    }

    /// Rebase a free camera onto the closest visible node without moving the hidden cursor.
    ///
    /// `anchor_screen` stays at the query position the caller actually panned to - only the
    /// fraction is derived from a rect-clamped point. Snapping `anchor_screen` itself into the
    /// closest node's rect would undo the pan whenever that node stays closest across the move
    /// (e.g. a drag small enough to stay within one wide node's rect never escapes it).
    pub fn rebase_camera_to_closest_node(&mut self) {
        let Some(camera) = self.camera else {
            return;
        };
        let query = crate::geometry::Point::new(camera.anchor_screen.0, camera.anchor_screen.1);
        let Some(node) = self.frame.closest(query) else {
            return;
        };
        let Some(rect) = self.frame.rect_of(node) else {
            return;
        };
        let clamped_x = query.x.clamp(rect.left(), rect.right());
        let clamped_y = query.y.clamp(rect.bottom(), rect.top());
        let frac = rect.fraction_of(crate::geometry::Point::new(clamped_x, clamped_y));
        self.camera = Some(Camera {
            anchor: node,
            anchor_fraction: frac,
            anchor_screen: camera.anchor_screen,
            hard_zone: camera.hard_zone,
        });
    }

    /// Convert terminal coordinates to `frame`'s screen space (origin at `last_area`'s
    /// bottom-left, Y-up), the inverse of the flip `CursorOverlay`/the painter apply.
    fn terminal_to_screen(&self, terminal_x: u16, terminal_y: u16) -> Option<(i64, i64)> {
        let area = self.last_area;
        if area.width == 0
            || area.height == 0
            || terminal_x < area.x
            || terminal_x >= area.x + area.width
            || terminal_y < area.y
            || terminal_y >= area.y + area.height
        {
            return None;
        }
        let screen_x = (terminal_x - area.x) as i64;
        let screen_y = (area.height - 1 - (terminal_y - area.y)) as i64;
        Some((screen_x, screen_y))
    }

    /// Convert a position in `frame`'s screen space (origin at `last_area`'s bottom-left,
    /// Y-up) to terminal coordinates, the inverse of `terminal_to_screen`. Returns `None`
    /// when the position falls outside the area rendered on the last call to
    /// `GraphView::render`, matching `terminal_to_screen`'s bounds check.
    pub fn screen_to_terminal(&self, x: i64, y: i64) -> Option<(u16, u16)> {
        let area = self.last_area;
        if area.width == 0
            || area.height == 0
            || x < 0
            || y < 0
            || x >= area.width as i64
            || y >= area.height as i64
        {
            return None;
        }
        let terminal_x = area.x + x as u16;
        let terminal_y = area.y + (area.height - 1 - y as u16);
        Some((terminal_x, terminal_y))
    }

    /// Return the wormhole reached by a horizontal move in the direction of its placed arrow.
    /// Wormholes live outside `FrameIndex`, so they must be checked explicitly before ordinary
    /// domain-node navigation. Use rendered geometry rather than domain edge direction: routing
    /// may place a successor door to the left of its boundary (or a predecessor to the right).
    fn horizontal_wormhole(&self, delta: i64) -> Option<(N, N)> {
        let boundary = self.cursor.node?;
        let boundary_rect = self.frame.rect_of(boundary)?;
        let cursor = boundary_rect.point_at_fraction(self.cursor.fractional);
        let next_x = cursor.x.saturating_add(delta);
        if next_x >= boundary_rect.left() && next_x <= boundary_rect.right() {
            return None;
        }
        let distance = |rect: crate::geometry::WorldRect| {
            let closest = rect.find_closest_cell(cursor);
            let dx = closest.x - cursor.x;
            let dy = closest.y - cursor.y;
            dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy))
        };

        let wormhole = self
            .wormhole
            .iter()
            .filter(|&&(rect, candidate_boundary, _)| {
                candidate_boundary == boundary
                    && if delta > 0 {
                        rect.center().x > boundary_rect.center().x
                    } else {
                        rect.center().x < boundary_rect.center().x
                    }
            })
            .min_by_key(|&&(rect, _, target)| (distance(rect), rect.left(), rect.bottom(), target))
            .copied();
        let (_, boundary, target) = wormhole?;
        Some((boundary, target))
    }

    /// Handle keyboard events for graph navigation and control. Returns the wormhole reached
    /// by a navigation key, if any; the owning application performs the engine-level world
    /// activation because `GraphViewState` deliberately does not own a `LayoutEngine`.
    #[cfg(feature = "crossterm")]
    pub fn handle_key_event(&mut self, key: KeyEvent) -> Result<Option<(N, N)>, String> {
        match key.code {
            KeyCode::Left | KeyCode::Char('h') => {
                if let Some(wormhole) = self.horizontal_wormhole(-1) {
                    return Ok(Some(wormhole));
                }
                Navigator::move_horizontal(&mut self.cursor, -1, &self.frame)?;
                self.rebase_camera_to_cursor();
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if let Some(wormhole) = self.horizontal_wormhole(1) {
                    return Ok(Some(wormhole));
                }
                Navigator::move_horizontal(&mut self.cursor, 1, &self.frame)?;
                self.rebase_camera_to_cursor();
            }
            // Note: in world/screen coordinates, Y increases upward.
            KeyCode::Up | KeyCode::Char('k') => {
                Navigator::move_vertical(&mut self.cursor, 1, &self.frame)?;
                self.rebase_camera_to_cursor();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                Navigator::move_vertical(&mut self.cursor, -1, &self.frame)?;
                self.rebase_camera_to_cursor();
            }
            _ => (),
        }
        Ok(None)
    }

    /// Move the cursor to the next stop to its right (`forward`) or left, as listed per node
    /// by `stops` in columns from the node's left edge, and keep the camera following it.
    /// See `Navigator::move_to_stop`. The search stays within the active batch: it does not
    /// pass through wormhole doors, so it fails and leaves the cursor in place when no
    /// placed node ahead has a stop.
    pub fn move_cursor_to_stop(
        &mut self,
        forward: bool,
        stops: impl Fn(N) -> Vec<i64>,
    ) -> Result<(), String> {
        let direction = if forward {
            Direction::Right
        } else {
            Direction::Left
        };
        Navigator::move_to_stop(&mut self.cursor, direction, &self.frame, stops)?;
        self.rebase_camera_to_cursor();
        Ok(())
    }

    /// Rebase the camera's anchor onto the cursor's current node, pinning it at the screen
    /// position the cursor already occupies (so nothing visually jumps), then let
    /// `snap_camera` push it back if the cursor has crossed into the hard zone.
    fn rebase_camera_to_cursor(&mut self) {
        let Some(node) = self.cursor.node else {
            return;
        };
        let Some(rect) = self.frame.rect_of(node) else {
            return;
        };
        let point = rect.point_at_fraction(self.cursor.fractional);
        let screen = (point.x, point.y);
        let hard_zone = self
            .camera
            .map(|c| c.hard_zone)
            .unwrap_or(DEFAULT_HARD_ZONE);
        let anchor_screen = snap_camera(screen, self.last_area, screen, hard_zone);
        self.camera = Some(Camera {
            anchor: node,
            anchor_fraction: self.cursor.fractional,
            anchor_screen,
            hard_zone,
        });
    }

    /// Pick the next accent color from the theme (slots 0x08-0x0F), cycling sequentially
    /// after each call.
    pub fn next_accent_color(&self) -> Color {
        let theme = current_theme();
        const ACCENTS: [usize; 8] = [0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F];
        let accent_colors: Vec<Color> = ACCENTS.iter().map(|&i| theme[i]).collect();

        let last_idx = self
            .highlights
            .styles
            .iter()
            .rev()
            .find_map(|(_, s)| accent_colors.iter().position(|&c| c == s.color));

        let next_idx = match last_idx {
            None => 0,
            Some(i) => {
                let n = (i + 1) % 8;
                if n == 0 {
                    log::warn!("all 8 accent colours have been used; cycling back to the first");
                }
                n
            }
        };

        accent_colors[next_idx]
    }

    pub fn set_node_highlight(&mut self, node: N, style: PathStyle) {
        self.highlights
            .styles
            .push((HighlightKind::Node(node), style));
    }

    pub fn set_edge_highlight(&mut self, edge: (N, N), style: PathStyle) {
        self.highlights
            .styles
            .push((HighlightKind::Edge(edge.0, edge.1), style));
    }

    pub fn set_cell_highlight(
        &mut self,
        node: N,
        tl: (i64, i64),
        br: (i64, i64),
        style: PathStyle,
    ) {
        self.highlights
            .styles
            .push((HighlightKind::Cells { node, tl, br }, style));
    }

    pub fn has_highlight(&self, style: &PathStyle) -> bool {
        self.highlights.styles.iter().any(|(_, s)| s == style)
    }

    pub fn clear_highlight(&mut self, style: &PathStyle) {
        self.highlights.styles.retain(|(_, s)| s != style);
    }

    pub fn clear_all_highlights(&mut self) {
        self.highlights.styles.clear();
    }

    pub fn dim_edge(&mut self, edge: (N, N)) {
        self.highlights.edge_lowlights.push(edge);
    }

    pub fn dim_node(&mut self, node: N) {
        self.highlights.node_lowlights.push(node);
    }
}

/// The actual `StatefulWidget`: per-frame geometry and painting over the controller's active
/// structural world. Applications hold `GraphViewState` and construct a fresh, borrowing
/// `GraphView` for each render call.
type OverlayFn<'a, NodeId> = Box<dyn FnOnce(&mut Buffer, &FrameIndex<NodeId>) + 'a>;

pub struct GraphView<'a, G, V, S = EagerSource>
where
    G: GraphBase,
{
    engine: &'a mut LayoutEngine<G, S>,
    visual: &'a V,
    block: Option<Block<'a>>,
    style: Style,
    overlay_fn: Option<OverlayFn<'a, G::NodeId>>,
}

impl<'a, G, V, S> GraphView<'a, G, V, S>
where
    G: GraphBase,
{
    pub fn new(engine: &'a mut LayoutEngine<G, S>, visual: &'a V) -> Self {
        Self {
            engine,
            visual,
            block: None,
            style: Style::default(),
            overlay_fn: None,
        }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Paint into the render buffer after the graph and before the cursor overlay.
    pub fn overlay<F>(mut self, overlay_fn: F) -> Self
    where
        F: FnOnce(&mut Buffer, &FrameIndex<G::NodeId>) + 'a,
    {
        self.overlay_fn = Some(Box::new(overlay_fn));
        self
    }
}

impl<G, V, S> StatefulWidget for GraphView<'_, G, V, S>
where
    G: GraphBase + EdgeIndexable + NodeIndexable + NodeCount + Visitable,
    G::NodeId: Copy + Eq + Hash + Ord + 'static,
    G::EdgeId: Clone,
    for<'b> &'b G: GraphBase<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNodeIdentifiers<NodeId = G::NodeId>
        + IntoEdgeReferences<NodeId = G::NodeId, EdgeId = G::EdgeId>
        + IntoNeighborsDirected<NodeId = G::NodeId>,
    for<'b> &'b G::NodeId: Hash + Ord,
    for<'b> &'b G::EdgeId: Clone,
    V: NodeRenderer<G>,
    S: GraphSource<G>,
{
    type State = GraphViewState<G::NodeId>;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        buf.set_style(area, self.style);
        let inner_area = if let Some(block) = &self.block {
            let inner = block.inner(area);
            block.clone().render(area, buf);
            inner
        } else {
            area
        };
        state.last_area = inner_area;

        let node_budget = self
            .engine
            .neighborhood_node_budget(inner_area.width as usize);

        let initial_batch = match self.engine.ensure_initial_world(inner_area.width as usize) {
            Ok(Some(key)) => key,
            Ok(None) => {
                state.frame = FrameIndex::empty();
                state.wormhole.clear();
                state.last_batch = None;
                return;
            }
            Err(error) => {
                log::error!("GraphView: initial world construction failed: {}", error);
                state.frame = FrameIndex::empty();
                state.wormhole.clear();
                return;
            }
        };

        // An ordinary go-to inside the active world only reframes the camera. A target outside
        // it is the one render-time operation allowed to switch batches, claiming a new one if
        // nothing owns the target yet.
        if state.go_to_pending
            && let Some(target) = state.cursor.node
            && !self.engine.active_contains(target)
            && let Err(error) = self.engine.activate_batch_containing(target, node_budget)
        {
            log::error!("GraphView: jump world construction failed: {}", error);
            if let Some(previous) = state.go_to_previous_cursor.take() {
                state.cursor = previous;
            }
            state.go_to_pending = false;
            state.go_to_frame_anchor = None;
            state.go_to_snap_left = false;
            state.go_to_snap_right = false;
        }

        let active_batch = self.engine.active_batch().unwrap_or(initial_batch);
        let had_rendered_world = state.last_batch.is_some();
        let world_changed = state.last_batch != Some(active_batch);
        if world_changed {
            state.frame = FrameIndex::empty();
            state.wormhole.clear();
            state.last_batch = Some(active_batch);
        }

        let Some(active_world) = self.engine.active_world() else {
            state.frame = FrameIndex::empty();
            state.wormhole.clear();
            return;
        };
        let window = active_world.layout().clone();
        let structural_anchor = active_world.anchor();
        let half_height = inner_area.height as i64 / 2;

        // The main view initially frames from the left; secondary views center the anchor.
        if world_changed && !state.go_to_pending && (had_rendered_world || state.camera.is_none()) {
            let use_main_initial_framing =
                state.cursor.node.is_none() && !state.center_world_on_change;
            if state.cursor.node.is_none()
                || !active_world.contains(state.cursor.node.unwrap_or(structural_anchor))
            {
                state.cursor.set_node(structural_anchor, (0.0, 0.5));
            }
            let (fraction, screen_x) = if use_main_initial_framing {
                let anchor_screen_x =
                    node_rank_in_window(&window, self.engine.graph(), structural_anchor)
                        .and_then(|anchor_rank| {
                            let min_rank = window
                                .graph
                                .node_weights()
                                .filter_map(|node| node.layer)
                                .min()?;
                            let width_by_rank =
                                rank_widths(&window, self.engine.graph(), self.visual);
                            Some(
                                DEFAULT_HARD_ZONE as i64
                                    + content_width_left_of_rank(
                                        &width_by_rank,
                                        min_rank,
                                        anchor_rank,
                                    ),
                            )
                        })
                        .unwrap_or(DEFAULT_HARD_ZONE as i64);
                ((0.0, 0.5), anchor_screen_x)
            } else {
                ((0.5, 0.5), inner_area.width as i64 / 2)
            };
            state.camera = Some(Camera {
                anchor: structural_anchor,
                anchor_fraction: fraction,
                anchor_screen: (screen_x, half_height),
                hard_zone: DEFAULT_HARD_ZONE,
            });
        } else if state.camera.is_none() {
            state.camera = Some(Camera {
                anchor: structural_anchor,
                anchor_fraction: (0.5, 0.5),
                anchor_screen: (inner_area.width as i64 / 2, half_height),
                hard_zone: DEFAULT_HARD_ZONE,
            });
        }

        // A pending go-to centers the cursor's screen position so the anchor projection
        // lands the target exactly at (or snapped to an edge of) the requested spot.
        if state.go_to_pending {
            let col_x = if state.go_to_snap_left {
                state.go_to_snap_left = false;
                state.go_to_snap_right = false;
                if state.cursor.visible {
                    DEFAULT_HARD_ZONE as i64
                } else {
                    1
                }
            } else if state.go_to_snap_right {
                state.go_to_snap_right = false;
                let right_edge = inner_area.width as i64 - 1;
                if state.cursor.visible {
                    right_edge - DEFAULT_HARD_ZONE as i64
                } else {
                    right_edge
                }
            } else {
                inner_area.width as i64 / 2
            };
            let frame_anchor = state.go_to_frame_anchor.take();
            if let (Some(camera), Some(frame_node)) = (&mut state.camera, state.cursor.node) {
                match frame_anchor {
                    // Frame one active-world node while retaining another as the independent
                    // camera anchor.
                    Some(camera_anchor) => {
                        let placed =
                            node_pos_in_window(&window, self.engine.graph(), camera_anchor)
                                .zip(node_pos_in_window(&window, self.engine.graph(), frame_node));
                        camera.anchor = camera_anchor;
                        camera.anchor_fraction = (0.5, 0.5);
                        camera.anchor_screen = match placed {
                            Some((anchor_pos, frame_pos)) => (
                                col_x + (anchor_pos.0 - frame_pos.0),
                                half_height + (anchor_pos.1 - frame_pos.1),
                            ),
                            None => (col_x, half_height),
                        };
                    }
                    None => {
                        // Rebase within the active world and avoid leaving one partially clipped
                        // rank at the viewport's left edge.
                        let frac_x = state.cursor.fractional.0;
                        let adjusted_col_x =
                            node_rank_in_window(&window, self.engine.graph(), frame_node)
                                .and_then(|frame_rank| {
                                    let min_rank = window
                                        .graph
                                        .node_weights()
                                        .filter_map(|node| node.layer)
                                        .min()?;
                                    let width_by_rank =
                                        rank_widths(&window, self.engine.graph(), self.visual);
                                    let own_width =
                                        width_by_rank.get(&frame_rank).copied().unwrap_or(1) as i64;
                                    let left_of_col_x =
                                        col_x - (frac_x * own_width as f64).round() as i64;
                                    let shift = min_shift_to_avoid_left_clip(
                                        &width_by_rank,
                                        min_rank,
                                        frame_rank,
                                        left_of_col_x,
                                    );
                                    Some(col_x + shift)
                                })
                                .unwrap_or(col_x);

                        camera.anchor = frame_node;
                        camera.anchor_fraction = state.cursor.fractional;
                        camera.anchor_screen = (adjusted_col_x, half_height);
                    }
                }
            }
            state.go_to_pending = false;
            state.go_to_previous_cursor = None;
        }

        let Some(camera) = state.camera else {
            state.frame = FrameIndex::empty();
            state.wormhole = Vec::new();
            return;
        };

        let graph = self.engine.graph();
        let mut painter = GraphPainter::new(window, graph, self.visual).spacing(&state.gaps);
        if let Some(entry) = state.wormhole_entry {
            painter = painter.node_overlay(entry, PathStyle::new(current_theme()[0x0B]));
        }
        let (frame, wormhole) = painter.render(inner_area, buf, &camera, &state.highlights);
        if let Some(overlay_fn) = self.overlay_fn {
            overlay_fn(buf, &frame);
        }
        CursorOverlay::render(inner_area, buf, &state.cursor, &frame);
        state.frame = frame;
        state.wormhole = wormhole;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ratatui::layout::Rect;

    use super::*;
    use crate::{
        graph_widget::NODE_GLYPH,
        testing::mocks::{MockDomainGraph, TestGraphs},
    };

    #[derive(Clone)]
    struct FixedSizeVisual;

    impl NodeRenderer<MockDomainGraph> for FixedSizeVisual {
        fn get_node_size(&self, _node: &petgraph::graph::NodeIndex) -> (u64, u64) {
            (3, 1)
        }

        fn render_node(
            &self,
            buffer: &mut crate::viewport_state::WorldBuffer,
            area: crate::geometry::WorldRect,
            _node_id: &petgraph::graph::NodeIndex,
        ) {
            buffer.set_char_styled(area.center(), NODE_GLYPH, Style::default());
        }
    }

    #[test]
    fn test_first_render_initializes_cursor_and_camera() {
        let domain_graph = TestGraphs::domain_extended_diamond();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();

        let area = Rect::new(0, 0, 40, 10);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);

        assert!(state.cursor.node.is_some());
        assert!(state.camera.is_some());
        assert!(!state.frame.is_empty());
    }

    #[test]
    fn test_render_twice_preserves_cursor_screen_position() {
        let domain_graph = TestGraphs::domain_extended_diamond();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();

        let area = Rect::new(0, 0, 40, 10);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        let node = state.cursor.node.expect("should have a cursor node");
        let first_rect = state.frame.rect_of(node).expect("should place the node");

        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        let second_rect = state
            .frame
            .rect_of(node)
            .expect("should place the node again");

        assert_eq!(
            first_rect.center(),
            second_rect.center(),
            "anchor node should hold its screen position across an unchanged re-render"
        );
    }

    #[test]
    fn wormhole_entry_uses_base0b_over_other_node_highlights() {
        let domain_graph = TestGraphs::domain_simple_chain();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();
        let area = Rect::new(0, 0, 40, 10);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        let entry = state.cursor.node.expect("should initialize the cursor");

        state.set_node_highlight(entry, PathStyle::new(Color::Red));
        state.mark_wormhole_entry(entry);
        state.hide_cursor();
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);

        let center = state
            .frame
            .rect_of(entry)
            .expect("should place the entry node")
            .center();
        let terminal = state
            .screen_to_terminal(center.x, center.y)
            .expect("should keep the entry node visible");
        assert_eq!(buffer[terminal].fg, current_theme()[0x0B]);
    }

    #[test]
    fn test_navigate_right_moves_cursor_to_next_layer() {
        let domain_graph = TestGraphs::domain_simple_chain();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();

        let area = Rect::new(0, 0, 40, 10);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);

        #[cfg(feature = "crossterm")]
        {
            use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
            let first_node = state.cursor.node.expect("should have a cursor node");
            for _ in 0..5 {
                state
                    .handle_key_event(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE))
                    .expect("should navigate");
                if state.cursor.node != Some(first_node) {
                    break;
                }
            }
            assert_ne!(state.cursor.node, Some(first_node));
        }
    }

    #[test]
    fn test_go_to_node_centers_on_next_render() {
        let domain_graph = TestGraphs::domain_extended_diamond();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();

        let area = Rect::new(0, 0, 40, 10);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);

        let target = engine.default_anchor().expect("should have graph nodes");
        state.go_to_node(target, (0.5, 0.5));
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);

        let rect = state
            .frame
            .rect_of(target)
            .expect("should place the target");
        assert_eq!(rect.center().x, (area.width as i64) / 2);
    }

    #[test]
    fn node_hit_accounts_for_render_area_terminal_offset() {
        let domain_graph = TestGraphs::domain_extended_diamond();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();

        let area = Rect::new(7, 11, 40, 10);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);

        let target = state.cursor.node.expect("should have a cursor node");
        let target_center = state
            .frame
            .rect_of(target)
            .expect("should place the target")
            .center();
        let terminal = state
            .screen_to_terminal(target_center.x, target_center.y)
            .expect("should keep the target center within the rendered area");
        let cursor_before = state.cursor;

        let hit = state
            .node_hit(terminal.0, terminal.1)
            .expect("should hit the target at the offset coordinate");

        assert_eq!(hit.0, target);
        assert_eq!(state.cursor.node, cursor_before.node);
        assert_eq!(state.cursor.fractional, cursor_before.fractional);
        assert_eq!(state.cursor.visible, cursor_before.visible);
    }

    #[test]
    fn repeated_render_preserves_complete_buffer_and_frame() {
        let domain_graph = TestGraphs::domain_complex_dag();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();
        let area = Rect::new(0, 0, 60, 18);

        let mut first_buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut first_buffer, &mut state);
        let mut first_rects: Vec<_> = state
            .frame
            .ids()
            .filter_map(|node| state.frame.rect_of(node).map(|rect| (node, rect)))
            .collect();
        first_rects.sort_by_key(|(node, _)| node.index());

        let mut second_buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut second_buffer, &mut state);
        let mut second_rects: Vec<_> = state
            .frame
            .ids()
            .filter_map(|node| state.frame.rect_of(node).map(|rect| (node, rect)))
            .collect();
        second_rects.sort_by_key(|(node, _)| node.index());

        assert_eq!(first_buffer, second_buffer);
        assert_eq!(first_rects, second_rects);
        assert_eq!(engine.structural_build_counts().crawl, 1);
    }

    #[test]
    fn two_view_states_share_one_structural_world() {
        let domain_graph = TestGraphs::domain_complex_dag();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut main_state = GraphViewState::default();
        let mut second_state = GraphViewState::default();
        second_state.reset_render_state();
        second_state.hide_cursor();
        let main_area = Rect::new(0, 0, 60, 18);
        let second_area = Rect::new(0, 0, 30, 8);

        let mut main_buffer = ratatui::buffer::Buffer::empty(main_area);
        GraphView::new(&mut engine, &visual).render(main_area, &mut main_buffer, &mut main_state);
        let key = engine.active_batch().expect("should have an active world");
        let membership: HashSet<_> = engine
            .active_world()
            .expect("should retain the active world")
            .members()
            .collect();
        let counts = engine.structural_build_counts();

        let mut second_buffer = ratatui::buffer::Buffer::empty(second_area);
        GraphView::new(&mut engine, &visual).render(
            second_area,
            &mut second_buffer,
            &mut second_state,
        );

        assert_eq!(engine.active_batch(), Some(key));
        assert_eq!(engine.structural_build_counts(), counts);
        let second_camera = second_state
            .camera
            .expect("should give the second view a camera");
        let anchor = engine
            .active_world()
            .expect("should retain the active world")
            .anchor();
        assert_eq!(second_camera.anchor, anchor);
        assert_eq!(second_camera.anchor_screen.0, second_area.width as i64 / 2);
        assert_eq!(
            engine
                .active_world()
                .expect("should retain the active world")
                .members()
                .collect::<HashSet<_>>(),
            membership
        );
        assert_eq!(main_state.frame.ids().collect::<HashSet<_>>(), membership);
        assert_eq!(second_state.frame.ids().collect::<HashSet<_>>(), membership);
    }

    #[test]
    fn interactions_and_resize_do_not_rebuild_active_world() {
        let domain_graph = TestGraphs::domain_complex_dag();
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();
        let area = Rect::new(0, 0, 60, 18);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        let key = engine.active_batch().expect("should have an active world");
        let counts = engine.structural_build_counts();
        let target = engine
            .active_world()
            .expect("should retain the active world")
            .members()
            .last()
            .expect("should have nodes in the world");

        state.go_to_node(target, (0.5, 0.5));
        state.move_by_terminal(2, -1);
        state.rebase_camera_to_closest_node();
        state.zoom_index += 1;
        state.gaps.data_data_x = |_| 2;
        state.set_node_highlight(target, PathStyle::new(Color::Red));
        let resized = Rect::new(0, 0, 42, 12);
        let mut resized_buffer = ratatui::buffer::Buffer::empty(resized);
        GraphView::new(&mut engine, &visual).render(resized, &mut resized_buffer, &mut state);

        assert_eq!(engine.active_batch(), Some(key));
        assert_eq!(engine.structural_build_counts(), counts);
    }

    #[test]
    fn go_to_outside_active_world_builds_once() {
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..30).map(|_| domain_graph.add_node(())).collect();
        for pair in nodes.windows(2) {
            domain_graph.add_edge(pair[0], pair[1], ());
        }
        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();
        let area = Rect::new(0, 0, 30, 12);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        assert!(!engine.active_contains(nodes[29]));

        state.go_to_node(nodes[29], (0.5, 0.5));
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        let loaded_key = engine.active_batch().expect("should load the target world");
        assert_eq!(
            engine.active_world().map(|world| world.anchor()),
            Some(nodes[29])
        );
        assert!(engine.active_contains(nodes[29]));
        assert_eq!(engine.structural_build_counts().crawl, 2);

        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        assert_eq!(engine.active_batch(), Some(loaded_key));
        assert_eq!(engine.structural_build_counts().crawl, 2);
    }

    /// Rendering the same structural world must not depend on temporary hash-map order.
    #[test]
    fn render_is_stable_across_repeated_renders_of_an_unchanged_world() {
        let mut domain_graph = MockDomainGraph::new();
        const LAYERS: usize = 6;
        const WIDTH: usize = 4;
        let mut layer_nodes: Vec<Vec<NodeIndex>> = Vec::new();
        for _ in 0..LAYERS {
            layer_nodes.push((0..WIDTH).map(|_| domain_graph.add_node(())).collect());
        }
        for layer in 0..LAYERS - 1 {
            for (index, &source) in layer_nodes[layer].iter().enumerate() {
                for offset in 0..2 {
                    let target = layer_nodes[layer + 1][(index + offset) % WIDTH];
                    domain_graph.add_edge(source, target, ());
                }
            }
        }

        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();
        let area = Rect::new(0, 0, 24, 16);
        let mut buffer = ratatui::buffer::Buffer::empty(area);

        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        let world_key = engine.active_batch().expect("should have an active world");
        assert!(
            !state.wormhole.is_empty(),
            "test graph should be large enough that the node budget forces a wormhole boundary"
        );
        let first_rects: Vec<(usize, crate::geometry::WorldRect)> = state
            .frame
            .ids()
            .filter_map(|node| state.frame.rect_of(node).map(|rect| (node.index(), rect)))
            .collect();

        for _ in 0..5 {
            let mut repeat_buffer = ratatui::buffer::Buffer::empty(area);
            GraphView::new(&mut engine, &visual).render(area, &mut repeat_buffer, &mut state);
            assert_eq!(
                engine.active_batch(),
                Some(world_key),
                "an unchanged render loop must not rebuild the structural world"
            );
            let mut repeat_rects: Vec<(usize, crate::geometry::WorldRect)> = state
                .frame
                .ids()
                .filter_map(|node| state.frame.rect_of(node).map(|rect| (node.index(), rect)))
                .collect();
            repeat_rects.sort_by_key(|&(index, _)| index);
            let mut expected_rects = first_rects.clone();
            expected_rects.sort_by_key(|&(index, _)| index);
            assert_eq!(
                repeat_rects, expected_rects,
                "domain node positions must not change across renders of the same world"
            );
        }
    }

    #[test]
    fn rendered_wormhole_is_hit_testable_at_its_glyph() {
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..30).map(|_| domain_graph.add_node(())).collect();
        for pair in nodes.windows(2) {
            domain_graph.add_edge(pair[0], pair[1], ());
        }

        let mut engine = LayoutEngine::new(domain_graph);
        let visual = FixedSizeVisual;
        let mut state = GraphViewState::default();
        let area = Rect::new(7, 5, 30, 12);
        let mut buffer = ratatui::buffer::Buffer::empty(Rect::new(0, 0, 50, 25));

        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);
        let boundary = state
            .wormhole
            .first()
            .map(|&(_, boundary, _)| boundary)
            .expect("should render a wormhole");
        state.go_to_node(boundary, (0.5, 0.5));
        GraphView::new(&mut engine, &visual).render(area, &mut buffer, &mut state);

        let (rect, boundary, target) = state
            .wormhole
            .iter()
            .copied()
            .find(|(rect, _, _)| {
                state
                    .screen_to_terminal(rect.center().x, rect.center().y)
                    .is_some()
            })
            .expect("should render a visible wormhole");
        let terminal = state
            .screen_to_terminal(rect.center().x, rect.center().y)
            .expect("should keep the wormhole center inside the view");

        assert_eq!(
            state.wormhole_hit(terminal.0, terminal.1),
            Some((boundary, target))
        );
        assert!(matches!(
            buffer.cell(terminal).map(|cell| cell.symbol()),
            Some("◁" | "▷")
        ));

        #[cfg(feature = "crossterm")]
        {
            let boundary_rect = state
                .frame
                .rect_of(boundary)
                .expect("should place the wormhole boundary");
            let exits_right = rect.center().x > boundary_rect.center().x;
            state
                .cursor
                .set_node(boundary, (if exits_right { 1.0 } else { 0.0 }, 0.5));
            state.show_cursor();
            let key = crossterm::event::KeyEvent::new(
                if exits_right {
                    crossterm::event::KeyCode::Right
                } else {
                    crossterm::event::KeyCode::Left
                },
                crossterm::event::KeyModifiers::NONE,
            );

            assert_eq!(
                state.handle_key_event(key),
                Ok(Some((boundary, target))),
                "moving from a boundary toward its arrow should request wormhole activation"
            );
        }
    }
}
