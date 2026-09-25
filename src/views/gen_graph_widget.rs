use core::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use gen_core::{HashId, Strand, Workspace, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::{db::GraphConnection, locus::GraphLocus, node::Node, sequence::SequenceError};
use gen_tui::{
    crawl::{EagerSource, GraphSource},
    distribute_nodes::GapSizes,
    frame_index::FrameIndex,
    geometry::{WorldPos, WorldRect, floor_half},
    graph_view::GraphViewState,
    graph_widget::NODE_GLYPH,
    layout::VisualDetail,
    layout_engine::{BatchId, LayoutEngine},
    plotter::{NodeRenderer, PathStyle},
    theme::current_theme,
    viewport_state::WorldBuffer,
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

use crate::views::{
    annotation_track::{
        AnnotationSpan, LoadedNodeSlices, graph_locus_from_annotation_span,
        locus_should_show_in_truncated, span_covered_by_later, span_label_text,
        span_should_show_in_truncated,
    },
    graph_dimming::GraphDimming,
    graph_overlay::{AnnotationColorCache, GraphOverlay, OverlaySource},
    inline_label_placement::draw_label_near_pos,
};

/// Ordered `+`/`-` zoom gap-size steps, tightest to loosest, paired with a `NodeRenderer` in
/// [`build_zoom_levels`] to form the actual zoom table. Graph-agnostic - kept separate from
/// the renderer so tests can pair the same gap progression with mock renderers.
///
/// The first two steps use minimal/near-zero gaps (matched with the glyph-only renderer,
/// which has nothing to protect). The rest floor `data_data_y` at the layout's own suggested
/// gap (`y.max(1)`) but fix `data_data_x` to a flat constant rather than flooring the
/// suggested gap - the x-axis suggestion from the upstream Brandes-Köpf pass carries
/// averaging artifacts/dead space (see the module doc) that a floor alone can't compact
/// away, since it can only ever push a small value up, never a large one down. The two
/// widest steps use odd multiples (`×3`, `×5`) rather than `×2`/`×4`: doubling an odd base
/// flips it even, which throws off `halves`' asymmetric lo/hi split and visibly unbalances
/// node centering - odd multiples of an odd base stay odd.
///
/// Lives outside gen-tui deliberately: gen-tui only exposes the primitives (the public
/// `gaps` field, `zoom_index`) and has no notion of which concrete levels exist or their
/// gaps - that policy belongs to the widget.
pub const ZOOM_GAP_SIZES: [GapSizes; 6] = [
    GapSizes {
        data_data_x: |_| 1,
        data_data_y: |_| 0,
        data_routing_x: |_| 1,
        data_routing_y: |_| 0,
        routing_routing_x: |_| 0,
        routing_routing_y: |_| 0,
    },
    GapSizes {
        data_data_x: |_| 1,
        data_data_y: |y| y.max(1),
        data_routing_x: |_| 1,
        data_routing_y: |y| y,
        routing_routing_x: |_| 1,
        routing_routing_y: |y| y,
    },
    GapSizes {
        data_data_x: |_| 2,
        data_data_y: |y| y.max(1),
        data_routing_x: |_| 1,
        data_routing_y: |y| y,
        routing_routing_x: |_| 0,
        routing_routing_y: |y| y,
    },
    GapSizes {
        data_data_x: |_| 2,
        data_data_y: |y| y.max(1),
        data_routing_x: |_| 1,
        data_routing_y: |y| y,
        routing_routing_x: |_| 0,
        routing_routing_y: |y| y,
    },
    GapSizes {
        data_data_x: |_| 5,
        data_data_y: |y| 3 * y.max(1),
        data_routing_x: |_| 1,
        data_routing_y: |y| y,
        routing_routing_x: |_| 0,
        routing_routing_y: |y| y,
    },
    GapSizes {
        data_data_x: |_| 9,
        data_data_y: |y| 5 * y.max(1),
        data_routing_x: |_| 1,
        data_routing_y: |y| y,
        routing_routing_x: |_| 0,
        routing_routing_y: |y| y,
    },
];

/// Index of the canonical `Minimal` step - used by callers exposing a named "minimal detail"
/// action (e.g. the Jupyter widget's `minimize_sequences`) rather than raw zoom in/out.
pub const MINIMAL_ZOOM_LEVEL: usize = 1;

/// Index into a zoom table matching a widget's initial `Truncated` / default-gap state -
/// the third entry built by [`build_zoom_levels`] (index 2: after the two `Minimal` steps).
pub const DEFAULT_ZOOM_LEVEL: usize = 2;

/// Index of the canonical `Full` step - used by callers exposing a named "full detail"
/// action rather than raw zoom in/out.
pub const FULL_ZOOM_LEVEL: usize = 3;

/// A widget/event-loop-owned table of `(renderer, gap sizes)` pairs, one per zoom step.
/// `GraphViewState::zoom_index` is an index into this table; nothing about which renderer
/// is active lives on the widget, the controller, or the renderer itself. [`VisualDetail`]
/// here is inert per-entry metadata (not renderer state) used only by column-clamping code
/// ([`clamp_col`]) that needs to know how a node's interior columns map at this zoom step.
///
/// Bounded by `'a`, not `Send + Sync + 'static`: the common case (TUI, R bindings) builds
/// this from a borrowed `&GraphConnection`, which is neither `Send` nor `Sync` and never
/// outlives the caller's scope. See [`SendSyncZoomLevels`] for the one caller (the Jupyter
/// widget) that needs the stronger bound.
pub type ZoomLevels<'a> = Vec<(VisualDetail, Arc<dyn NodeRenderer<GenGraph> + 'a>, GapSizes)>;

/// The six `(VisualDetail, GapSizes)` gap-size steps paired with fresh renderer instances,
/// tightest to loosest - shared by [`build_zoom_levels`] and [`build_send_sync_zoom_levels`].
/// The truncated and full renderers each get their own instance (sharing one sequence cache
/// across the three full-detail entries via `Arc`, not cloning it) so a cache warmed at one
/// gap size stays warm when the user only changes spacing.
macro_rules! zoom_entries {
    ($minimal:expr, $truncated:expr, $full:expr) => {
        vec![
            (VisualDetail::Minimal, $minimal.clone(), ZOOM_GAP_SIZES[0]),
            (VisualDetail::Minimal, $minimal, ZOOM_GAP_SIZES[1]),
            (VisualDetail::Truncated, $truncated, ZOOM_GAP_SIZES[2]),
            (VisualDetail::Full, $full.clone(), ZOOM_GAP_SIZES[3]),
            (VisualDetail::Full, $full.clone(), ZOOM_GAP_SIZES[4]),
            (VisualDetail::Full, $full, ZOOM_GAP_SIZES[5]),
        ]
    };
}

/// Build the standard six-step zoom table for a GenGraph sequence source: two minimal-glyph
/// steps, one truncated-sequence step, and three full-sequence steps at increasing gaps.
pub fn build_zoom_levels<'a, S>(source: S) -> ZoomLevels<'a>
where
    S: SequenceSource + Clone + 'a,
{
    let minimal: Arc<dyn NodeRenderer<GenGraph> + 'a> = Arc::new(GenGraphMinimalRenderer);
    let truncated: Arc<dyn NodeRenderer<GenGraph> + 'a> =
        Arc::new(GenGraphTruncatedRenderer::new(source.clone()));
    let full: Arc<dyn NodeRenderer<GenGraph> + 'a> = Arc::new(GenGraphFullRenderer::new(source));
    zoom_entries!(minimal, truncated, full)
}

/// Like [`build_zoom_levels`], but with [`GenGraphAnnotatedRenderer`] at the three
/// full-detail steps, drawing whatever `layer` currently holds under each node.
pub fn build_annotated_zoom_levels<'a, S>(source: S, layer: NodeAnnotationLayer) -> ZoomLevels<'a>
where
    S: SequenceSource + Clone + 'a,
{
    let minimal: Arc<dyn NodeRenderer<GenGraph> + 'a> = Arc::new(GenGraphMinimalRenderer);
    let truncated: Arc<dyn NodeRenderer<GenGraph> + 'a> =
        Arc::new(GenGraphTruncatedRenderer::new(source.clone()));
    let full: Arc<dyn NodeRenderer<GenGraph> + 'a> =
        Arc::new(GenGraphAnnotatedRenderer::new(source, layer));
    zoom_entries!(minimal, truncated, full)
}

/// A [`ZoomLevels`] table whose renderers are `Send + Sync + 'static`, for callers (the
/// Jupyter widget's `#[pyclass]`) that must store the table as a field of a type pyo3
/// requires to be `Send + Sync`.
pub type SendSyncZoomLevels = Vec<(
    VisualDetail,
    Arc<dyn NodeRenderer<GenGraph> + Send + Sync>,
    GapSizes,
)>;

/// Like [`build_zoom_levels`], but for a `Send + Sync + 'static` sequence source (e.g.
/// `PathSequenceSource`), producing a table that itself is `Send + Sync`.
pub fn build_send_sync_zoom_levels<S>(source: S) -> SendSyncZoomLevels
where
    S: SequenceSource + Clone + Send + Sync + 'static,
{
    let minimal: Arc<dyn NodeRenderer<GenGraph> + Send + Sync> = Arc::new(GenGraphMinimalRenderer);
    let truncated: Arc<dyn NodeRenderer<GenGraph> + Send + Sync> =
        Arc::new(GenGraphTruncatedRenderer::new(source.clone()));
    let full: Arc<dyn NodeRenderer<GenGraph> + Send + Sync> =
        Arc::new(GenGraphFullRenderer::new(source));
    zoom_entries!(minimal, truncated, full)
}

/// Build thread-safe zoom levels with annotation flags at full detail for Jupyter.
pub fn build_send_sync_annotated_zoom_levels<S>(
    source: S,
    layer: NodeAnnotationLayer,
) -> SendSyncZoomLevels
where
    S: SequenceSource + Clone + Send + Sync + 'static,
{
    let minimal: Arc<dyn NodeRenderer<GenGraph> + Send + Sync> = Arc::new(GenGraphMinimalRenderer);
    let truncated: Arc<dyn NodeRenderer<GenGraph> + Send + Sync> =
        Arc::new(GenGraphTruncatedRenderer::new(source.clone()));
    let full: Arc<dyn NodeRenderer<GenGraph> + Send + Sync> =
        Arc::new(GenGraphAnnotatedRenderer::new(source, layer));
    zoom_entries!(minimal, truncated, full)
}

/// Apply `levels[index]` (clamped in range) to a view state's zoom index and gaps. Generic
/// over the renderer's trait-object bound so it works for both [`ZoomLevels`] and
/// [`SendSyncZoomLevels`] - it only ever reads the `GapSizes` element.
pub fn apply_zoom_level<R>(
    view_state: &mut GraphViewState<GraphNode>,
    index: usize,
    levels: &[(VisualDetail, R, GapSizes)],
) {
    let index = index.min(levels.len() - 1);
    view_state.zoom_index = index;
    view_state.gaps = levels[index].2;
}

/// Step one zoom level in (more detail / more spread). No-op at the last level.
pub fn zoom_in<R>(
    view_state: &mut GraphViewState<GraphNode>,
    levels: &[(VisualDetail, R, GapSizes)],
) {
    apply_zoom_level(view_state, view_state.zoom_index + 1, levels);
}

/// Step one zoom level out (less detail / tighter packing). No-op at the first level.
pub fn zoom_out<R>(
    view_state: &mut GraphViewState<GraphNode>,
    levels: &[(VisualDetail, R, GapSizes)],
) {
    apply_zoom_level(view_state, view_state.zoom_index.saturating_sub(1), levels);
}

/// Labels for special start/end nodes
pub mod label {
    pub const START: &str = "╟";
    pub const END: &str = "╢";
}

/// Where a `GenGraphNodeRenderer` fetches genomic sequence data from. Implemented for a
/// live `&GraphConnection` (a session-long connection, e.g. the TUI and R bindings,
/// where the visual and the connection share one thread for the widget's lifetime) and
/// for `PathSequenceSource` (a lazily-opened, reused connection, e.g. the Jupyter widget,
/// whose controller is `Send` and moved between the ipykernel executor thread and the
/// asyncio thread, so it cannot hold a live connection at construction time, but can
/// still own one once opened — `rusqlite::Connection` is `Send`, just not `Sync`).
pub trait SequenceSource {
    fn get_node_sequence(
        &self,
        node_id: HashId,
        start: i64,
        end: i64,
    ) -> Result<String, SequenceError>;
}

impl SequenceSource for &GraphConnection {
    fn get_node_sequence(
        &self,
        node_id: HashId,
        start: i64,
        end: i64,
    ) -> Result<String, SequenceError> {
        let database_path = self
            .path()
            .map(PathBuf::from)
            .ok_or_else(|| SequenceError::Io("graph database has no file path".to_string()))?;
        let workspace_path = database_path
            .parent()
            .and_then(|path| path.parent())
            .ok_or_else(|| {
                SequenceError::Io("graph database is not inside a workspace".to_string())
            })?;
        let workspace = Workspace::new(workspace_path);
        (*self, &workspace).get_node_sequence(node_id, start, end)
    }
}

impl<'a> SequenceSource for (&'a GraphConnection, &'a Workspace) {
    fn get_node_sequence(
        &self,
        node_id: HashId,
        start: i64,
        end: i64,
    ) -> Result<String, SequenceError> {
        Ok(
            Node::get_sequence_range(self.0, self.1, node_id, start, end)?
                .unwrap_or_else(|| "?".repeat((end - start).max(0) as usize)),
        )
    }
}

/// A `SequenceSource` that opens its `GraphConnection` lazily on first use and then
/// reuses it for the rest of its life, instead of reopening (and re-running migrations)
/// on every cache-missed node. The connection is wrapped in a `Mutex` both to make this
/// type `Sync` (`pyo3`'s `#[pyclass]` requires it, and `rusqlite::Connection` is `Send`
/// but not `Sync`) and to provide interior mutability for lazy-open under `&self`.
/// Cloning drops the cached connection rather than trying to duplicate a live one
/// (`Connection` isn't `Clone`); the clone reopens lazily on its own first use.
pub struct PathSequenceSource {
    db_path: PathBuf,
    conn: std::sync::Mutex<Option<GraphConnection>>,
}

impl PathSequenceSource {
    pub fn new(db_path: PathBuf) -> Self {
        Self {
            db_path,
            conn: std::sync::Mutex::new(None),
        }
    }
}

impl Clone for PathSequenceSource {
    fn clone(&self) -> Self {
        Self::new(self.db_path.clone())
    }
}

impl SequenceSource for PathSequenceSource {
    fn get_node_sequence(
        &self,
        node_id: HashId,
        start: i64,
        end: i64,
    ) -> Result<String, SequenceError> {
        if self.conn.is_poisoned() {
            self.conn.clear_poison();
        }
        let mut conn = self
            .conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if conn.is_none() {
            *conn = Some(
                crate::get_connection(self.db_path.clone())
                    .map_err(|e| SequenceError::Io(e.to_string()))?,
            );
        }
        conn.as_ref()
            .unwrap()
            .get_node_sequence(node_id, start, end)
    }
}

/// Fetch a node's sequence through `source`, caching the result in `cache`. Shared by
/// [`GenGraphTruncatedRenderer`] and [`GenGraphFullRenderer`] - the only two levels that
/// need genomic sequence data at all.
fn fetch_cached_sequence<S: SequenceSource>(
    source: &S,
    cache: &Mutex<HashMap<GraphNode, String>>,
    node_key: &GraphNode,
) -> Result<String, SequenceError> {
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(cached) = cache.get(node_key) {
        return Ok(cached.clone());
    }

    let sequence = source.get_node_sequence(
        node_key.node_id,
        node_key.sequence_start,
        node_key.sequence_end,
    )?;

    cache.insert(*node_key, sequence.clone());
    Ok(sequence)
}

/// Render start/end nodes with their fixed label, or fall through to `render_glyph`/
/// `render_sequence` for ordinary nodes. Shared sizing/rendering shape for all three
/// GenGraph renderer levels - only the ordinary-node behavior differs between them.
fn start_end_node_size(node: &GraphNode) -> Option<(u64, u64)> {
    if is_start_node(node.node_id) {
        return Some((label::START.chars().count() as u64, 1u64));
    }
    if is_end_node(node.node_id) {
        return Some((label::END.chars().count() as u64, 1u64));
    }
    None
}

/// Whether a node is drawn. A zero-length slice is where an edit meets a node (a junction), so it
/// is left for its edges to meet at instead; the zero-length start and end sentinels are drawn
/// with their labels.
fn is_drawn_node(node: &GraphNode) -> bool {
    node.length() > 0 || is_start_node(node.node_id) || is_end_node(node.node_id)
}

/// Render start/end nodes with their fixed label; returns `true` if it did (nothing further
/// to render for this node).
fn render_start_end_node(buffer: &mut WorldBuffer, area: WorldRect, node_id: &GraphNode) -> bool {
    let theme = current_theme();
    if is_start_node(node_id.node_id) {
        let edge_style = Style::default().bg(theme[0x00]).fg(theme[0x05]);
        buffer.set_string_styled(area.left_center(), label::START, edge_style);
        return true;
    }
    if is_end_node(node_id.node_id) {
        let edge_style = Style::default().bg(theme[0x00]).fg(theme[0x05]);
        buffer.set_string_styled(area.left_center(), label::END, edge_style);
        return true;
    }
    false
}

/// `NodeRenderer` for the lowest GenGraph zoom level: every ordinary node collapses to a
/// single glyph. Needs no sequence source, so it carries no data at all.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenGraphMinimalRenderer;

impl NodeRenderer<GenGraph> for GenGraphMinimalRenderer {
    fn get_node_size(&self, node: &GraphNode) -> (u64, u64) {
        start_end_node_size(node).unwrap_or((1, 1))
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &GraphNode) {
        let theme = current_theme();
        let background_style = Style::default().bg(theme[0x05]);
        buffer.fill_rect(area, ' ');
        buffer.set_char_styled(area.left_center(), ' ', background_style);

        if render_start_end_node(buffer, area, node_id) {
            return;
        }
        let text_style = Style::default().fg(theme[0x05]).bg(theme[0x00]);
        buffer.set_string_styled(area.left_center(), &NODE_GLYPH.to_string(), text_style);
    }

    fn is_visible(&self, node: &GraphNode) -> bool {
        is_drawn_node(node)
    }
}

/// `NodeRenderer` for the middle GenGraph zoom level: nodes show their genomic sequence,
/// inner-truncated to 13 cells (5 border bases + `...` + 5 border bases) when longer.
pub struct GenGraphTruncatedRenderer<S> {
    source: S,
    cache: Mutex<HashMap<GraphNode, String>>,
}

impl<S: SequenceSource> GenGraphTruncatedRenderer<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl<S: SequenceSource> NodeRenderer<GenGraph> for GenGraphTruncatedRenderer<S> {
    fn get_node_size(&self, node: &GraphNode) -> (u64, u64) {
        start_end_node_size(node).unwrap_or_else(|| {
            let sequence_length = (node.sequence_end - node.sequence_start) as u64;
            (sequence_length.min(13), 1u64) // 13 = 5 border + 3 mid + 5 border
        })
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &GraphNode) {
        let theme = current_theme();
        let background_style = Style::default().bg(theme[0x05]);
        let text_style = Style::default().bg(theme[0x05]).fg(theme[0x00]);
        buffer.fill_rect(area, ' ');
        buffer.set_char_styled(area.left_center(), ' ', background_style);

        if render_start_end_node(buffer, area, node_id) {
            return;
        }
        let sequence = fetch_cached_sequence(&self.source, &self.cache, node_id)
            .unwrap_or_else(|_| "Unknown Sequence".to_string());
        let truncated = inner_truncation(&sequence, 13);
        buffer.set_string_styled(area.left_center(), &truncated, text_style);
    }

    fn is_visible(&self, node: &GraphNode) -> bool {
        is_drawn_node(node)
    }
}

/// `NodeRenderer` for the highest GenGraph zoom levels: nodes show their complete genomic
/// sequence (clipped only by the viewport, not truncated).
pub struct GenGraphFullRenderer<S> {
    source: S,
    cache: Mutex<HashMap<GraphNode, String>>,
}

impl<S: SequenceSource> GenGraphFullRenderer<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl<S: SequenceSource> NodeRenderer<GenGraph> for GenGraphFullRenderer<S> {
    fn get_node_size(&self, node: &GraphNode) -> (u64, u64) {
        start_end_node_size(node)
            .unwrap_or(((node.sequence_end - node.sequence_start) as u64, 1u64))
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &GraphNode) {
        let theme = current_theme();
        let background_style = Style::default().bg(theme[0x05]);
        let text_style = Style::default().bg(theme[0x05]).fg(theme[0x00]);
        buffer.fill_rect(area, ' ');
        buffer.set_char_styled(area.left_center(), ' ', background_style);

        if render_start_end_node(buffer, area, node_id) {
            return;
        }
        let sequence = fetch_cached_sequence(&self.source, &self.cache, node_id)
            .unwrap_or_else(|_| "Unknown Sequence".to_string());
        buffer.set_string_styled(area.left_center(), &sequence, text_style);
    }

    fn is_visible(&self, node: &GraphNode) -> bool {
        is_drawn_node(node)
    }
}

/// Where an [`AnnotationFlag`]'s name is drawn relative to its bar, decided by
/// [`pack_annotation_flags`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelPlacement {
    /// Inside the bar, breaking the line but never covering the bar's first or last cell.
    Inside,
    /// Beside the bar with one cell of air, starting at this node-local column. Preferred
    /// to the left of the bar, falling back to the right when the bar sits too close to the
    /// node's start.
    Beside(i64),
    /// Nowhere under the node: the bar is drawn alone and the name is left to the floating
    /// label pass (`draw_annotation_labels`), which places it as close to the node as the
    /// surrounding cells allow.
    Floating,
}

/// One annotation's presence on one node, in that node's local column space, as drawn by
/// [`GenGraphAnnotatedRenderer`]: a bar over `bar_start..bar_end` with a direction cap on
/// whichever end is the feature's true end, plus its name. `lane` and `label` are filled in
/// by [`pack_annotation_flags`]; lane 0 is the row directly under the sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationFlag {
    /// The annotation this piece belongs to; pieces of one span on different nodes share it.
    pub id: HashId,
    /// Position of this piece among the span's pieces, in path order.
    pub piece: usize,
    pub name: String,
    pub color: Color,
    pub strand: Strand,
    /// First node-local column the bar covers (inclusive).
    pub bar_start: i64,
    /// Node-local column just past the bar (exclusive).
    pub bar_end: i64,
    /// The annotation continues on a preceding node, so this bar has no start cap.
    pub continues_left: bool,
    /// The annotation continues on a following node, so this bar has no end cap.
    pub continues_right: bool,
    /// Whether this piece carries the name. A span split over several nodes names only its
    /// widest piece, so a sliver of it on a short node is just a bar.
    pub show_label: bool,
    pub label: LabelPlacement,
    pub lane: usize,
}

impl AnnotationFlag {
    fn label_width(&self) -> i64 {
        if self.show_label {
            self.name.chars().count() as i64
        } else {
            0
        }
    }

    /// The node-local columns (`start..end`) this flag occupies for packing: its bar plus,
    /// when the name sits beside the bar, the name too.
    fn extent(&self) -> (i64, i64) {
        match self.label {
            LabelPlacement::Inside | LabelPlacement::Floating => (self.bar_start, self.bar_end),
            LabelPlacement::Beside(start) => (
                start.min(self.bar_start),
                (start + self.label_width()).max(self.bar_end),
            ),
        }
    }

    /// Whether `column` is the feature's directional end, where the cap is drawn.
    fn is_cap(&self, column: i64) -> bool {
        match self.strand {
            Strand::Forward => !self.continues_right && column == self.bar_end - 1,
            Strand::Reverse => !self.continues_left && column == self.bar_start,
            _ => false,
        }
    }

    fn cap_at(&self, column: i64) -> Option<char> {
        if !self.is_cap(column) {
            return None;
        }
        Some(match self.strand {
            Strand::Forward => ANNOTATION_FORWARD_CAP,
            _ => ANNOTATION_REVERSE_CAP,
        })
    }

    /// The bar columns (`start..end`) the name may be written over: the bar's interior, so
    /// its first and last cell (and any cap) stay visible around the name.
    fn text_columns(&self) -> (i64, i64) {
        let start = self.bar_start + 1;
        (start, (self.bar_end - 1).max(start))
    }
}

/// Flags are drawn as a double line so they cannot be mistaken for a graph edge (single
/// line) or a node (solid block), capped with a triangle on the feature's directional end.
const ANNOTATION_BAR: char = '═';
const ANNOTATION_FORWARD_CAP: char = '▶';
const ANNOTATION_REVERSE_CAP: char = '◀';

/// Decide every flag's label placement and lane for a node `node_width` columns wide, and
/// return how many lanes the node needs. Flags are sorted into drawing order.
///
/// A name that fits inside its bar's interior goes there. Otherwise it is placed one cell
/// to the left of the bar, or to the right when the left would spill past the node's
/// start, and from then on counts as part of the flag for spacing. A name that fits on
/// neither side is left to the floating label pass. Lanes are then assigned by interval
/// partitioning: flags in column order each take the first lane whose previous flag ends
/// at least one column before this one starts, which uses the fewest lanes possible for
/// the given extents.
pub fn pack_annotation_flags(flags: &mut [AnnotationFlag], node_width: i64) -> usize {
    for flag in flags.iter_mut() {
        let label_width = flag.label_width();
        let (text_start, text_end) = flag.text_columns();
        flag.label = if label_width <= text_end - text_start {
            LabelPlacement::Inside
        } else if flag.bar_start > label_width {
            LabelPlacement::Beside(flag.bar_start - 1 - label_width)
        } else if flag.bar_end + 1 + label_width <= node_width {
            LabelPlacement::Beside(flag.bar_end + 1)
        } else {
            LabelPlacement::Floating
        };
    }
    flags.sort_by(|a, b| {
        let (a_start, a_end) = a.extent();
        let (b_start, b_end) = b.extent();
        a_start
            .cmp(&b_start)
            .then((b_end - b_start).cmp(&(a_end - a_start)))
            .then(a.name.cmp(&b.name))
    });

    // Exclusive end column of the last flag placed in each lane.
    let mut lane_ends: Vec<i64> = Vec::new();
    for flag in flags.iter_mut() {
        let (start, end) = flag.extent();
        let lane = lane_ends
            .iter()
            .position(|lane_end| *lane_end < start)
            .unwrap_or_else(|| {
                lane_ends.push(i64::MIN);
                lane_ends.len() - 1
            });
        lane_ends[lane] = end;
        flag.lane = lane;
    }
    lane_ends.len()
}

#[derive(Clone, Debug, Default)]
struct PackedAnnotations {
    flags: Vec<AnnotationFlag>,
    lanes: usize,
}

/// The per-node annotation flags a [`GenGraphAnnotatedRenderer`] draws, shared between the
/// renderer (which only ever holds an `Arc<dyn NodeRenderer>` inside a zoom table) and the
/// viewer that owns the overlays. The viewer refills it with [`update_node_annotations`]
/// whenever its overlays change; the renderer reads it on every size query and paint.
#[derive(Clone, Debug, Default)]
pub struct NodeAnnotationLayer {
    packed: Arc<Mutex<HashMap<GraphNode, PackedAnnotations>>>,
    /// Bumped whenever `replace` changes any node's lane count, the only thing the layer
    /// contributes to node sizes, so views know when to re-route their geometry.
    size_generation: Arc<AtomicU64>,
}

impl NodeAnnotationLayer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pack `flags` per node and make them the layer's contents.
    pub fn replace(&self, flags_by_node: HashMap<GraphNode, Vec<AnnotationFlag>>) {
        let packed: HashMap<GraphNode, PackedAnnotations> = flags_by_node
            .into_iter()
            .map(|(node, mut flags)| {
                let lanes = pack_annotation_flags(&mut flags, node.length());
                (node, PackedAnnotations { flags, lanes })
            })
            .collect();
        let mut current = self.lock();
        let lanes_in = |contents: &HashMap<GraphNode, PackedAnnotations>, node: &GraphNode| {
            contents.get(node).map_or(0, |packed| packed.lanes)
        };
        let sizes_changed = packed
            .iter()
            .any(|(node, entry)| lanes_in(&current, node) != entry.lanes)
            || current
                .iter()
                .any(|(node, entry)| lanes_in(&packed, node) != entry.lanes);
        *current = packed;
        if sizes_changed {
            self.size_generation.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Changes whenever a `replace` changed how many lanes any node needs. See
    /// [`NodeRenderer::size_generation`].
    pub fn size_generation(&self) -> u64 {
        self.size_generation.load(Ordering::Relaxed)
    }

    /// Rows of annotation lanes `node` needs under its sequence.
    pub fn lanes(&self, node: &GraphNode) -> usize {
        self.lock().get(node).map_or(0, |packed| packed.lanes)
    }

    /// The annotations whose name found no room under their node, so they still need a
    /// floating label.
    pub fn floating_span_ids(&self) -> HashSet<HashId> {
        self.lock()
            .values()
            .flat_map(|packed| packed.flags.iter())
            .filter(|flag| flag.label == LabelPlacement::Floating)
            .map(|flag| flag.id)
            .collect()
    }

    /// Every flag, grouped by annotation and ordered by piece, with the node each is on.
    fn pieces(&self) -> Vec<Vec<(GraphNode, AnnotationFlag)>> {
        let mut by_id: HashMap<HashId, Vec<(GraphNode, AnnotationFlag)>> = HashMap::new();
        for (node, packed) in self.lock().iter() {
            for flag in &packed.flags {
                by_id
                    .entry(flag.id)
                    .or_default()
                    .push((*node, flag.clone()));
            }
        }
        let mut pieces: Vec<_> = by_id.into_values().collect();
        for span in &mut pieces {
            span.sort_by_key(|(_, flag)| flag.piece);
        }
        pieces.sort_by_key(|span| span[0].1.id);
        pieces
    }

    fn flags(&self, node: &GraphNode) -> Vec<AnnotationFlag> {
        self.lock()
            .get(node)
            .map(|packed| packed.flags.clone())
            .unwrap_or_default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<GraphNode, PackedAnnotations>> {
        self.packed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Where annotations start on each loaded node, whether or not their flags are drawn: the
/// stops the viewers' `w`/`b` keys move the cursor between. Built from the overlays against
/// the loaded graph, the same way [`update_node_annotations`] builds the flags, so every loaded
/// node of the active batch has its stops, on screen or not.
#[derive(Clone, Debug, Default)]
pub struct AnnotationStarts(HashMap<GraphNode, Vec<i64>>);

impl AnnotationStarts {
    pub fn new<S: GraphSource<GenGraph>>(
        engine: &LayoutEngine<GenGraph, S>,
        overlays: &[GraphOverlay],
    ) -> Self {
        Self(
            annotation_flags_by_node(engine, overlays)
                .into_iter()
                .map(|(node, flags)| (node, flag_starts(&flags)))
                .collect(),
        )
    }

    /// Node-local columns where an annotation starts on `node`, in reading direction: the
    /// first column of a forward or unstranded annotation's first piece, and the last column
    /// of a reverse-strand annotation's last piece.
    pub fn on(&self, node: &GraphNode) -> Vec<i64> {
        self.0.get(node).cloned().unwrap_or_default()
    }
}

/// The sorted, distinct start columns of one node's flags; see [`AnnotationStarts::on`].
fn flag_starts(flags: &[AnnotationFlag]) -> Vec<i64> {
    let mut starts: Vec<i64> = flags
        .iter()
        .filter_map(|flag| match flag.strand {
            Strand::Reverse => (!flag.continues_right).then_some(flag.bar_end - 1),
            _ => (!flag.continues_left).then_some(flag.bar_start),
        })
        .collect();
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// Refill `layer` from the span overlays, using the colors [`reapply_overlays`] settled on
/// (so call it after that pass). Returns the overlays whose name found no room under their
/// node, for the caller to hand to [`draw_annotation_labels`].
pub fn update_node_annotations<S: GraphSource<GenGraph>>(
    layer: &NodeAnnotationLayer,
    engine: &LayoutEngine<GenGraph, S>,
    overlays: &[GraphOverlay],
) -> Vec<GraphOverlay> {
    layer.replace(annotation_flags_by_node(engine, overlays));
    let floating = layer.floating_span_ids();
    overlays
        .iter()
        .filter(|overlay| {
            overlay
                .span()
                .is_some_and(|span| floating.contains(&span.id))
        })
        .cloned()
        .collect()
}

/// Join consecutive slices of a locus that continue each other on the same node, such as the
/// parts of a GenBank `join(541..546,547..564)` location, so they are drawn as one bar. Parts
/// that don't meet stay separate: the two halves of a feature wrapping a circular sequence's
/// origin sit at opposite ends of its node, one running off the end and one starting at the
/// start.
fn join_touching_slices(slices: &[GraphNodeSlice]) -> Vec<GraphNodeSlice> {
    let mut joined: Vec<GraphNodeSlice> = Vec::with_capacity(slices.len());
    for slice in slices {
        if let Some(previous) = joined.last_mut()
            && previous.block == slice.block
            && previous.strand == slice.strand
            && match slice.strand {
                Strand::Reverse => slice.end == previous.start,
                _ => previous.end == slice.start,
            }
        {
            previous.start = previous.start.min(slice.start);
            previous.end = previous.end.max(slice.end);
            continue;
        }
        joined.push(*slice);
    }
    joined
}

/// Every span overlay's flags, unpacked, on the loaded nodes it covers. Every piece of a span
/// (its segments, with touching ones joined by [`join_touching_slices`]) becomes a flag on its
/// own node; the direction cap is only drawn on the piece that is the feature's true end,
/// which is what `continues_left`/`continues_right` record.
fn annotation_flags_by_node<S: GraphSource<GenGraph>>(
    engine: &LayoutEngine<GenGraph, S>,
    overlays: &[GraphOverlay],
) -> HashMap<GraphNode, Vec<AnnotationFlag>> {
    let theme = current_theme();
    let loaded = LoadedNodeSlices::new(engine.graph());
    let mut flags_by_node: HashMap<GraphNode, Vec<AnnotationFlag>> = HashMap::new();
    for overlay in overlays {
        let Some(span) = overlay.span().filter(|span| !span.name.is_empty()) else {
            continue;
        };
        let Some(locus) = graph_locus_from_annotation_span(span, &loaded) else {
            continue;
        };
        let color = match overlay.style.color {
            Color::Reset => theme[0x06],
            other => other,
        };
        let slices = join_touching_slices(&locus.slices);
        let last = slices.len().saturating_sub(1);
        let widest = slices
            .iter()
            .enumerate()
            .max_by_key(|(index, slice)| (slice.end - slice.start, -(*index as i64)))
            .map_or(0, |(index, _)| index);
        for (index, slice) in slices.iter().enumerate() {
            let width = slice.block.length();
            let bar_start = (slice.start as i64).clamp(0, width);
            let bar_end = (slice.end as i64).clamp(0, width);
            if bar_end <= bar_start {
                continue;
            }
            flags_by_node
                .entry(slice.block)
                .or_default()
                .push(AnnotationFlag {
                    id: span.id,
                    piece: index,
                    name: span.name.clone(),
                    color,
                    strand: slice.strand,
                    bar_start,
                    bar_end,
                    continues_left: index > 0,
                    continues_right: index < last,
                    show_label: index == widest,
                    label: LabelPlacement::Inside,
                    lane: 0,
                });
        }
    }
    flags_by_node
}

/// Braille dot bit for a dot at sub-cell `(column, row)` of a 2×4 braille cell.
const fn braille_bit(column: i64, row: i64) -> u32 {
    match (column, row) {
        (0, 0) => 0x01,
        (0, 1) => 0x02,
        (0, 2) => 0x04,
        (0, 3) => 0x40,
        (1, 0) => 0x08,
        (1, 1) => 0x10,
        (1, 2) => 0x20,
        _ => 0x80,
    }
}

/// Visit a thin, eight-connected line, including both endpoints, without supercover dots.
fn rasterize_braille_line(from: (i64, i64), to: (i64, i64), visit: &mut impl FnMut((i64, i64))) {
    let (mut x, mut y) = from;
    let delta_x = (to.0 - x).abs();
    let delta_y = -(to.1 - y).abs();
    let step_x = (to.0 - x).signum();
    let step_y = (to.1 - y).signum();
    let mut error = delta_x + delta_y;
    loop {
        visit((x, y));
        if (x, y) == to {
            break;
        }
        let twice_error = 2 * error;
        if twice_error >= delta_y {
            error += delta_y;
            x += step_x;
        }
        if twice_error <= delta_x {
            error += delta_x;
            y += step_y;
        }
    }
}

/// Flatten in dot space before rounding, so dense curve samples cannot widen the line.
fn rasterize_braille_cubic(controls: &[(f64, f64); 4], visit: &mut impl FnMut((i64, i64))) {
    let [start, first, second, end] = *controls;
    let chord = (end.0 - start.0, end.1 - start.1);
    let chord_length = chord.0.hypot(chord.1);
    let distance = |point: (f64, f64)| {
        let offset = (point.0 - start.0, point.1 - start.1);
        if chord_length == 0.0 {
            offset.0.hypot(offset.1)
        } else {
            (chord.0 * offset.1 - chord.1 * offset.0).abs() / chord_length
        }
    };
    // A quarter dot keeps the polyline close to the curve on the binary Braille grid.
    if distance(first).max(distance(second)) <= 0.25 {
        // Resolve half-dot ties evenly so reflecting a connector preserves its rounding.
        rasterize_braille_line(
            (
                start.0.round_ties_even() as i64,
                start.1.round_ties_even() as i64,
            ),
            (
                end.0.round_ties_even() as i64,
                end.1.round_ties_even() as i64,
            ),
            visit,
        );
        return;
    }

    let midpoint =
        |left: (f64, f64), right: (f64, f64)| ((left.0 + right.0) / 2.0, (left.1 + right.1) / 2.0);
    let start_first = midpoint(start, first);
    let first_second = midpoint(first, second);
    let second_end = midpoint(second, end);
    let left_control = midpoint(start_first, first_second);
    let right_control = midpoint(first_second, second_end);
    let middle = midpoint(left_control, right_control);
    // Both halves reuse the same endpoint so rounding cannot leave a gap at the join.
    rasterize_braille_cubic(&[start, start_first, left_control, middle], visit);
    rasterize_braille_cubic(&[middle, right_control, second_end, end], visit);
}

/// Adaptively flatten and rasterize a thin braille cubic between two terminal cells
/// (inclusive, beside their vertical middle), adding dots only to empty or braille cells,
/// so edge lines and nodes in the way are left intact.
fn draw_braille_curve(buf: &mut Buffer, from: (u16, u16), to: (u16, u16), color: Color) {
    // Work in dot space: 2 columns × 4 rows of dots per cell.
    // Braille has no center dot. Choose the middle row facing the curve at each
    // bar so the joins bend inward; terminal row coordinates increase downward.
    let start_row = 1 + i64::from(to.1 > from.1);
    let end_row = 1 + i64::from(to.1 < from.1);
    let (x0, y0) = (2 * from.0 as i64, 4 * from.1 as i64 + start_row);
    let (x1, y1) = (2 * to.0 as i64 + 1, 4 * to.1 as i64 + end_row);
    // Mid-gap controls at each endpoint's height give symmetric, horizontal joins.
    let middle_x = (x0 + x1) as f64 / 2.0;
    let controls = [
        (x0 as f64, y0 as f64),
        (middle_x, y0 as f64),
        (middle_x, y1 as f64),
        (x1 as f64, y1 as f64),
    ];
    let mut dots: HashMap<(u16, u16), u32> = HashMap::new();
    rasterize_braille_cubic(&controls, &mut |(x, y)| {
        let cell = ((x / 2) as u16, (y / 4) as u16);
        *dots.entry(cell).or_default() |= braille_bit(x % 2, y % 4);
    });
    for (cell, bits) in dots {
        let Some(target) = buf.cell_mut(cell) else {
            continue;
        };
        let existing = target.symbol().chars().next().unwrap_or(' ');
        let merged = match existing {
            ' ' => bits,
            braille if ('\u{2800}'..='\u{28FF}').contains(&braille) => {
                bits | (braille as u32 - 0x2800)
            }
            _ => continue,
        };
        let glyph = char::from_u32(0x2800 + merged).expect("should be a valid braille glyph");
        target.set_char(glyph);
        target.set_fg(color);
    }
}

/// Connect the pieces of the focused annotation, when it spans several nodes, with a dotted
/// braille cubic curve from the end of one piece's bar to the start of the next, drawn after
/// the graph so it can use the placed node rects in `frame`. Connectors for every annotation
/// at once crowd the graph, so only the span `focused` names is connected, and nothing is
/// drawn without a focus. Only pieces whose nodes were both placed by the last render are
/// connected, and the curve only ever fills empty cells, so it never breaks an edge or a node
/// it crosses.
pub fn draw_annotation_connectors(
    buf: &mut Buffer,
    area: Rect,
    frame: &FrameIndex<GraphNode>,
    layer: &NodeAnnotationLayer,
    focused: Option<HashId>,
) {
    let Some(focused) = focused else {
        return;
    };
    let to_terminal = |x: i64, y: i64| -> Option<(u16, u16)> {
        if x < 0 || y < 0 || x >= area.width as i64 || y >= area.height as i64 {
            return None;
        }
        Some((area.x + x as u16, area.y + area.height - 1 - y as u16))
    };
    let lane_row = |rect: WorldRect, flag: &AnnotationFlag| -> i64 {
        sequence_row(rect) - 1 - flag.lane as i64
    };
    for pieces in layer
        .pieces()
        .into_iter()
        .filter(|pieces| pieces[0].1.id == focused)
    {
        for pair in pieces.windows(2) {
            let ((from_node, from_flag), (to_node, to_flag)) = (&pair[0], &pair[1]);
            let (Some(from_rect), Some(to_rect)) =
                (frame.rect_of(*from_node), frame.rect_of(*to_node))
            else {
                continue;
            };
            let from_x = from_rect.min.x + from_flag.bar_end;
            let to_x = to_rect.min.x + to_flag.bar_start - 1;
            if to_x < from_x {
                continue;
            }
            let (Some(from), Some(to)) = (
                to_terminal(from_x, lane_row(from_rect, from_flag)),
                to_terminal(to_x, lane_row(to_rect, to_flag)),
            ) else {
                continue;
            };
            draw_braille_curve(buf, from, to, from_flag.color);
        }
    }
}

/// The row a node's sequence sits on within `area`: the layout's center row, which is where
/// edges attach and where `highlight_match_range` paints. Everything below it is annotation
/// lanes. The layout centers a node on its position with `WorldRect::from_center_and_size`,
/// which puts `floor_half(rows)` rows below the center and the rest above (the extra row of
/// an even height goes above), so an annotated node asks for an odd height: `2 × lanes + 1`
/// rows leave exactly `lanes` rows under the sequence, mirrored by as many blank rows above
/// it that keep the sequence on the row the layout routes edges to.
fn sequence_row(area: WorldRect) -> i64 {
    let rows = area.max.y - area.min.y + 1;
    area.min.y + floor_half(rows)
}

/// `NodeRenderer` for the highest GenGraph zoom levels with annotations drawn under each
/// node as packed flags: a bar over the feature's columns, a triangle cap on its
/// directional end, and its name inside the bar (inverted) when it fits or beside it when
/// not. Sizing grows with the number of lanes the node's flags pack into, so the layout
/// reserves the rows instead of letting neighbors overdraw them.
pub struct GenGraphAnnotatedRenderer<S> {
    source: S,
    cache: Mutex<HashMap<GraphNode, String>>,
    layer: NodeAnnotationLayer,
}

impl<S: SequenceSource> GenGraphAnnotatedRenderer<S> {
    pub fn new(source: S, layer: NodeAnnotationLayer) -> Self {
        Self {
            source,
            cache: Mutex::new(HashMap::new()),
            layer,
        }
    }

    fn render_flag(
        &self,
        buffer: &mut WorldBuffer,
        area: WorldRect,
        row: i64,
        flag: &AnnotationFlag,
    ) {
        let theme = current_theme();
        let bar_style = Style::default().fg(flag.color).bg(theme[0x00]);
        let local_x = |column: i64| area.min.x + column;

        for column in flag.bar_start..flag.bar_end {
            let glyph = flag.cap_at(column).unwrap_or(ANNOTATION_BAR);
            buffer.set_char_styled(WorldPos::new(local_x(column), row), glyph, bar_style);
        }

        match flag.label {
            LabelPlacement::Beside(start) => {
                buffer.set_string_styled(WorldPos::new(local_x(start), row), &flag.name, bar_style);
            }
            LabelPlacement::Floating => {}
            LabelPlacement::Inside => {
                // Center the name in whatever part of the bar is on screen right now, so a
                // long feature scrolled half off the viewport still shows its name where
                // the user can see it; once even that is too narrow, clip the name.
                let visible = buffer.visible_world_area();
                let (text_start, text_end) = flag.text_columns();
                let visible_start = text_start.max(visible.min.x - area.min.x);
                let visible_end = text_end.min(visible.max.x - area.min.x + 1);
                let visible_width = visible_end - visible_start;
                let label_width = flag.label_width();
                if visible_width <= 0 || label_width == 0 {
                    return;
                }
                let text_x = if label_width <= visible_width {
                    visible_start + (visible_width - label_width) / 2
                } else {
                    visible_start
                };
                let text: String = flag.name.chars().take(visible_width as usize).collect();
                buffer.set_string_styled(WorldPos::new(local_x(text_x), row), &text, bar_style);
            }
        }
    }
}

impl<S: SequenceSource> NodeRenderer<GenGraph> for GenGraphAnnotatedRenderer<S> {
    fn get_node_size(&self, node: &GraphNode) -> (u64, u64) {
        start_end_node_size(node).unwrap_or_else(|| {
            let lanes = self.layer.lanes(node) as u64;
            (node.length() as u64, 2 * lanes + 1)
        })
    }

    fn size_generation(&self) -> u64 {
        self.layer.size_generation()
    }

    /// The sequence row (see [`sequence_row`]); the flag lanes around it are only decoration.
    fn cursor_row(&self, node: &GraphNode) -> Option<u64> {
        Some(floor_half(self.get_node_size(node).1 as i64) as u64)
    }

    fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &GraphNode) {
        let theme = current_theme();
        let background_style = Style::default().bg(theme[0x05]);
        let text_style = Style::default().bg(theme[0x05]).fg(theme[0x00]);
        buffer.fill_rect_styled(area, ' ', Style::default().bg(theme[0x00]));
        let sequence_y = sequence_row(area);
        let sequence_pos = WorldPos::new(area.min.x, sequence_y);
        buffer.fill_rect_styled(
            WorldRect::from_coords(area.min.x, sequence_y, area.max.x, sequence_y),
            ' ',
            background_style,
        );

        if render_start_end_node(buffer, area, node_id) {
            return;
        }
        let sequence = fetch_cached_sequence(&self.source, &self.cache, node_id)
            .unwrap_or_else(|_| "Unknown Sequence".to_string());
        buffer.set_string_styled(sequence_pos, &sequence, text_style);

        for flag in self.layer.flags(node_id) {
            let row = sequence_pos.y - 1 - flag.lane as i64;
            if row < area.min.y {
                continue;
            }
            self.render_flag(buffer, area, row, &flag);
        }
    }

    fn is_visible(&self, node: &GraphNode) -> bool {
        is_drawn_node(node)
    }
}

/// Truncate a genomic sequence from the inside, keeping the beginning and end.
///
/// # Arguments
/// * `s` - The sequence string to truncate
/// * `target_length` - Maximum length for the output string
///
/// # Returns
/// A string showing beginning...end if truncation needed, or original if short enough.
pub fn inner_truncation(s: &str, target_length: u32) -> String {
    if s.len() <= target_length as usize {
        return s.to_string();
    } else if target_length < 5 {
        return NODE_GLYPH.to_string(); // ⏺ is U+23FA
    }
    // length - 3 because we need space for the ellipsis
    let left_len = (target_length - 3) / 2 + ((target_length - 3) % 2);
    let right_len = (target_length - 3) / 2;

    let left = &s[..left_len as usize];
    let right = &s[(s.len() - right_len as usize)..];

    format!("{}...{}", left, right)
}

/// Create a `LayoutEngine`/`ZoomLevels`/`GraphViewState` triple for a GenGraph with the
/// standard theme and settings.
///
/// This is the standard way to initialize a `GraphView` for GenGraph visualization: it dims
/// pruned edges and the nodes only they lead into (see [`GraphDimming`]), starts at [`DEFAULT_ZOOM_LEVEL`], and starts in
/// free-camera mode (cursor hidden until the user clicks a node or uses keyboard nav).
///
/// # Arguments
/// * `graph` - The GenGraph to visualize
/// * `source` - Sequence source the renderers use for sequence fetching (a live
///   `&GraphConnection`, or a `PathBuf` for callers that cannot hold one)
///
/// # Returns
/// The engine, zoom table, and view state ready to be passed to `GraphView::new`/rendered.
pub fn create_gen_graph_engine<'a, S: SequenceSource + Clone + 'a>(
    graph: GenGraph,
    source: S,
) -> (
    LayoutEngine<GenGraph>,
    ZoomLevels<'a>,
    GraphViewState<GraphNode>,
) {
    build_gen_graph_engine(graph, build_zoom_levels(source))
}

/// Like [`create_gen_graph_engine`], but with annotation flags drawn under nodes at the
/// full-detail zoom steps from `layer` - see [`build_annotated_zoom_levels`].
pub fn create_annotated_gen_graph_engine<'a, S: SequenceSource + Clone + 'a>(
    graph: GenGraph,
    source: S,
    layer: NodeAnnotationLayer,
) -> (
    LayoutEngine<GenGraph>,
    ZoomLevels<'a>,
    GraphViewState<GraphNode>,
) {
    build_gen_graph_engine(graph, build_annotated_zoom_levels(source, layer))
}

/// Like [`create_gen_graph_engine`], but for a `Send + Sync + 'static` sequence source
/// (e.g. `PathSequenceSource`), producing a zoom table that itself is `Send + Sync` - for
/// callers (the Jupyter widget's `#[pyclass]`) that must store the table as a field of a
/// type pyo3 requires to be `Send + Sync`.
pub fn create_send_sync_gen_graph_engine<S: SequenceSource + Clone + Send + Sync + 'static>(
    graph: GenGraph,
    source: S,
) -> (
    LayoutEngine<GenGraph>,
    SendSyncZoomLevels,
    GraphViewState<GraphNode>,
) {
    build_gen_graph_engine(graph, build_send_sync_zoom_levels(source))
}

/// Create a thread-safe graph engine with annotation flags at full detail for Jupyter.
pub fn create_send_sync_annotated_gen_graph_engine<S>(
    graph: GenGraph,
    source: S,
    layer: NodeAnnotationLayer,
) -> (
    LayoutEngine<GenGraph>,
    SendSyncZoomLevels,
    GraphViewState<GraphNode>,
)
where
    S: SequenceSource + Clone + Send + Sync + 'static,
{
    build_gen_graph_engine(graph, build_send_sync_annotated_zoom_levels(source, layer))
}

type GraphEngineSetup<R> = (
    LayoutEngine<GenGraph>,
    Vec<(VisualDetail, R, GapSizes)>,
    GraphViewState<GraphNode>,
);

/// Shared body of [`create_gen_graph_engine`]/[`create_send_sync_gen_graph_engine`]: dims
/// pruned edges and the nodes only they lead into, starts at [`DEFAULT_ZOOM_LEVEL`], and starts in
/// free-camera mode (cursor hidden until the user clicks a node or uses keyboard nav).
///
/// Cycle safety (including a circular genome's `PATH_END -> PATH_START` closure) is left
/// entirely to `LayoutEngine`'s own per-window cycle detection (`crawl::build_window_graph`)
/// rather than a whole-graph pre-pass here - for the whole-graph-as-one-window case this is
/// today's only caller, that detects exactly the same cycles a dedicated whole-graph DFS would.
/// Reverse-complement link collapsing (redundant `A+ -> B+` / `B- -> A-` GFA pairs) is not
/// performed here either; that GFA-import cleanup is deferred for now.
fn build_gen_graph_engine<R>(
    graph: GenGraph,
    levels: Vec<(VisualDetail, R, GapSizes)>,
) -> GraphEngineSetup<R> {
    let start_node = graph.nodes().find(|node| is_start_node(node.node_id));
    let mut view_state = GraphViewState::default();
    // The graph never grows, so one sync decides all of its dimming.
    GraphDimming::default().sync(&graph, &EagerSource, &mut view_state);
    let mut engine = LayoutEngine::new(graph);
    if let Some(start_node) = start_node {
        engine.set_preferred_initial_anchor(start_node);
    }

    apply_zoom_level(&mut view_state, DEFAULT_ZOOM_LEVEL, &levels);
    view_state.hide_cursor();

    (engine, levels, view_state)
}

type GraphEngineSetupLazy<R, S> = (
    LayoutEngine<GenGraph, S>,
    Vec<(VisualDetail, R, GapSizes)>,
    GraphViewState<GraphNode>,
);

/// Like [`build_gen_graph_engine`], but for a `graph` that is only seeded (e.g. just its
/// starting anchor) and grows lazily through `source` as the crawl reaches unloaded nodes -
/// see [`crate::views::lazy_graph_source::SqlGraphSource`]. The view state starts undimmed:
/// callers keep a [`GraphDimming`] and sync it whenever the crawl may have grown
/// `engine.graph()`, which any render can do.
fn build_gen_graph_engine_lazy<R, S>(
    graph: GenGraph,
    source: S,
    levels: Vec<(VisualDetail, R, GapSizes)>,
) -> GraphEngineSetupLazy<R, S>
where
    S: GraphSource<GenGraph>,
{
    let start_node = graph.nodes().find(|node| is_start_node(node.node_id));
    let mut view_state = GraphViewState::default();
    let mut engine = LayoutEngine::new_with_source(graph, source);
    if let Some(start_node) = start_node {
        engine.set_preferred_initial_anchor(start_node);
    }

    apply_zoom_level(&mut view_state, DEFAULT_ZOOM_LEVEL, &levels);
    view_state.hide_cursor();

    (engine, levels, view_state)
}

/// Like [`create_annotated_gen_graph_engine`], but for a lazily-loaded `graph`/`source` pair -
/// see [`build_gen_graph_engine_lazy`], including how the caller keeps it dimmed.
pub fn create_annotated_gen_graph_engine_lazy<'a, Src, Seq>(
    graph: GenGraph,
    source: Src,
    sequence_source: Seq,
    layer: NodeAnnotationLayer,
) -> (
    LayoutEngine<GenGraph, Src>,
    ZoomLevels<'a>,
    GraphViewState<GraphNode>,
)
where
    Src: GraphSource<GenGraph>,
    Seq: SequenceSource + Clone + 'a,
{
    build_gen_graph_engine_lazy(
        graph,
        source,
        build_annotated_zoom_levels(sequence_source, layer),
    )
}

/// Like [`create_send_sync_annotated_gen_graph_engine`], but for a lazily-loaded `graph`/`source`
/// pair - see [`build_gen_graph_engine_lazy`], including how the caller keeps it dimmed.
pub fn create_send_sync_annotated_gen_graph_engine_lazy<Src, Seq>(
    graph: GenGraph,
    source: Src,
    sequence_source: Seq,
    layer: NodeAnnotationLayer,
) -> (
    LayoutEngine<GenGraph, Src>,
    SendSyncZoomLevels,
    GraphViewState<GraphNode>,
)
where
    Src: GraphSource<GenGraph>,
    Seq: SequenceSource + Clone + Send + Sync + 'static,
{
    build_gen_graph_engine_lazy(
        graph,
        source,
        build_send_sync_annotated_zoom_levels(sequence_source, layer),
    )
}

/// The slice and local offset (0-based, within that slice's block) at the
/// midpoint of a locus's total sequence length. `None` for an empty locus.
pub fn locus_midpoint(locus: &GraphLocus) -> Option<(GraphNodeSlice, usize)> {
    let total: usize = locus
        .slices
        .iter()
        .map(|slice| slice.end - slice.start)
        .sum();
    if total == 0 {
        return None;
    }
    let half = total / 2;
    let mut consumed = 0;
    for (index, slice) in locus.slices.iter().enumerate() {
        let len = slice.end - slice.start;
        let is_last = index == locus.slices.len() - 1;
        if consumed + len > half || is_last {
            return Some((*slice, slice.start + (half - consumed)));
        }
        consumed += len;
    }
    None
}

/// Compute the screen-space bounding corners of the matched region in a `GraphLocus`,
/// against the placed-node rects of the most recently rendered `GraphView` frame.
///
/// Returns `(left_pos, right_pos)` where:
/// - `left_pos` is the screen position of the first matched column in `blocks[0]`,
///   with the minimum y across all blocks
/// - `right_pos` is the screen position of the last matched column in `blocks[last]`,
///   with the maximum y across all blocks
///
/// Column offsets are clamped with `clamp_col` so they map correctly in every
/// detail level (e.g. truncated nodes collapse interior columns to the `...` cell).
///
/// Returns `None` if no block in the locus was placed by that render (all off-screen).
pub fn locus_label_bounds(
    locus: &GraphLocus,
    frame: &FrameIndex<GraphNode>,
    detail_level: VisualDetail,
) -> Option<(WorldPos, WorldPos)> {
    let block_screen_pos = |block: GraphNode, col_raw: i64| -> Option<WorldPos> {
        let rect = frame.rect_of(block)?;
        let col = clamp_col(col_raw, block.length(), detail_level);
        Some(WorldPos::new(rect.min.x + col, rect.center().y))
    };

    let last = locus.slices.len() - 1;

    let left_pos = locus.slices.iter().enumerate().find_map(|(i, s)| {
        let col_raw = if i == 0 { s.start as i64 } else { 0 };
        block_screen_pos(s.block, col_raw)
    })?;

    let right_pos = locus.slices.iter().enumerate().rev().find_map(|(i, s)| {
        let col_raw = if i == last {
            s.end.saturating_sub(1) as i64
        } else {
            s.block.length() - 1
        };
        block_screen_pos(s.block, col_raw)
    })?;

    let mut y_min = i64::MAX;
    let mut y_max = i64::MIN;
    for s in &locus.slices {
        let Some(rect) = frame.rect_of(s.block) else {
            continue;
        };
        y_min = y_min.min(rect.center().y);
        y_max = y_max.max(rect.center().y);
    }

    if y_min == i64::MAX {
        return None;
    }

    let left_pos = WorldPos::new(left_pos.x, y_max);
    let right_pos = WorldPos::new(right_pos.x, y_min);

    Some((left_pos, right_pos))
}

/// Apply detail-level clamping to a raw column offset.
///
/// In `Truncated` mode, interior columns of long nodes map to the `...` region.
fn clamp_col(col_raw: i64, block_seq_len: i64, detail_level: VisualDetail) -> i64 {
    match detail_level {
        VisualDetail::Minimal => 0,
        VisualDetail::Truncated if block_seq_len > 13 => {
            clamp_truncated_col(col_raw, block_seq_len)
        }
        _ => col_raw,
    }
}

/// Map a sequence offset to a visual cell column inside a 13-cell truncated node.
///
/// Display layout: `AAAAA...BBBBB` — first 5 bases (cells 0-4), `...` (cells 5-7),
/// last 5 bases (cells 8-12). Interior positions that fall in the `...` region clamp
/// to cell 6 (the centre dot).
fn clamp_truncated_col(value: i64, block_seq_len: i64) -> i64 {
    if value < 5 {
        value
    } else if block_seq_len - value <= 5 {
        13 - (block_seq_len - value)
    } else {
        6
    }
}

/// Highlight only the matched bytes of a `GraphLocus`, using sub-rect tinting.
///
/// - Start node: tinted from `start_offset` to its right edge.
/// - Middle nodes: fully tinted.
/// - End node: tinted from its left edge to `end_offset` (exclusive).
///
/// In `Truncated` detail level the column offsets are clamped so that interior
/// positions map to the `...` cell rather than a precise (wrong) location.
pub fn highlight_match_range<R>(
    view_state: &mut GraphViewState<GraphNode>,
    levels: &[(VisualDetail, R, GapSizes)],
    m: &GraphLocus,
    style: PathStyle,
) where
    R: NodeRenderer<GenGraph>,
{
    let (detail_level, renderer, _) = &levels[view_state.zoom_index.min(levels.len() - 1)];

    for s in &m.slices {
        let block_seq_len = s.block.length();
        let col_start_raw = s.start as i64;
        let col_end_raw = s.end.saturating_sub(1) as i64;
        let (col_start, col_end) = (
            clamp_col(col_start_raw, block_seq_len, *detail_level),
            clamp_col(col_end_raw, block_seq_len, *detail_level),
        );
        // Cell highlights are offset from the node rect's bottom row; the sequence sits on
        // the center row, above any annotation lanes the renderer reserved under it.
        let row = floor_half(renderer.get_node_size(&s.block).1 as i64);
        view_state.set_cell_highlight(s.block, (col_start, row), (col_end, row), style);
    }

    for (s, t) in m.slices.iter().zip(m.slices.iter().skip(1)) {
        view_state.set_edge_highlight((s.block, t.block), style);
    }
}

/// A mapped cell rectangle: the node it's on, and its top-left/bottom-right columns.
type CellRegion = (GraphNode, (i64, i64), (i64, i64));

/// The mapped column range a `GraphNodeSlice` occupies once rendered, using the same
/// `clamp_col` math `highlight_match_range` uses to paint it. Computing conflicts against
/// this (rather than against raw, unmapped sequence coordinates) is what catches
/// collisions that only exist after mapping - e.g. two annotations that don't overlap at
/// `Full` detail can still both collapse onto the same cell once a node is small enough to
/// be `Truncated`.
fn slice_region(detail_level: VisualDetail, slice: &GraphNodeSlice) -> CellRegion {
    let block_seq_len = slice.block.length();
    let col_start = slice.start as i64;
    let col_end = slice.end.saturating_sub(1) as i64;
    let tl = (clamp_col(col_start, block_seq_len, detail_level), 0);
    let br = (clamp_col(col_end, block_seq_len, detail_level), 0);
    (slice.block, tl, br)
}

/// Whether two mapped cell regions occupy any of the same cells.
fn regions_overlap(a: &CellRegion, b: &CellRegion) -> bool {
    a.0 == b.0 && a.1.0 <= b.2.0 && b.1.0 <= a.2.0
}

/// The 8 theme accent slots annotation colors and `AnnotationColorCache::next_color` are
/// drawn from.
fn accent_colors() -> [Color; 8] {
    let theme = current_theme();
    [
        theme[0x08],
        theme[0x09],
        theme[0x0A],
        theme[0x0B],
        theme[0x0C],
        theme[0x0D],
        theme[0x0E],
        theme[0x0F],
    ]
}

/// Assign annotation colors and replace the path highlights on `view_state`.
///
/// Span overlays are processed longest-first. Each span's color
/// is chosen greedily: its color from a previous pass (`color_cache`) if it's still
/// conflict-free, or else the next color in rotation, so spans that never conflict with
/// anything still get spread across distinct colors instead of colors reshuffling across
/// frames or collapsing onto one repeated color. Only hunts for a different free accent
/// slot when the preferred color is actually taken by a previously-processed span whose
/// mapped cell range overlaps this one; only gives up and accepts a collision if every slot
/// is already taken by a genuine neighbor - with only 8 slots, a dense pile of
/// mutually-overlapping annotations can still collide, but this makes collisions the
/// exception rather than the default.
///
/// `overlays` is written back with the colors used by annotation bars, connectors, and
/// labels. Annotation spans leave sequence cells and graph edges in their normal colors.
/// Callers run this after zoom or detail changes, or every frame in the live TUI viewers
/// because the overlay set changes with scrolling.
pub fn reapply_overlays<R, S>(
    engine: &LayoutEngine<GenGraph, S>,
    view_state: &mut GraphViewState<GraphNode>,
    levels: &[(VisualDetail, R, GapSizes)],
    overlays: &mut [GraphOverlay],
    color_cache: &mut AnnotationColorCache,
) where
    R: NodeRenderer<GenGraph>,
    S: GraphSource<GenGraph>,
{
    let detail_level = levels[view_state.zoom_index.min(levels.len() - 1)].0;
    let graph = engine.graph();
    let loaded = LoadedNodeSlices::new(graph);

    // DB-loaded tracks are too busy to paint at minimal detail; a span confined to a
    // partial slice of a single node is also dropped at truncated detail, mirroring the
    // label suppression below.
    let mut span_indices: Vec<usize> = overlays
        .iter()
        .enumerate()
        .filter_map(|(idx, overlay)| {
            overlay
                .span()
                .filter(|_| {
                    !matches!(
                        (detail_level, &overlay.source),
                        (VisualDetail::Minimal, OverlaySource::Track(_))
                    )
                })
                .filter(|span| {
                    detail_level != VisualDetail::Truncated
                        || span_should_show_in_truncated(span, &loaded)
                })
                .map(|_| idx)
        })
        .collect();
    span_indices.sort_by_key(|&idx| {
        let span = overlays[idx]
            .span()
            .expect("filtered to span overlays above");
        -(span
            .segments
            .iter()
            .map(|segment| segment.end - segment.start)
            .sum::<i64>())
    });

    // Use mapped span extents to keep overlapping annotation colors distinct.
    let accents = accent_colors();
    let mut occupied: Vec<(CellRegion, Color)> = Vec::new();
    for idx in span_indices {
        let span = overlays[idx]
            .span()
            .expect("filtered to span overlays above");
        let Some(locus) = graph_locus_from_annotation_span(span, &loaded) else {
            continue;
        };
        let regions: Vec<CellRegion> = locus
            .slices
            .iter()
            .map(|slice| slice_region(detail_level, slice))
            .collect();
        let used: Vec<Color> = occupied
            .iter()
            .filter(|(placed, _)| regions.iter().any(|region| regions_overlap(placed, region)))
            .map(|(_, color)| *color)
            .collect();

        // Prefer this span's previous color, or, the first time it's seen, the next color
        // in rotation, so spans that never conflict with anything still get spread across
        // distinct colors instead of repeatedly landing on the same one. Only hunt for a
        // different free accent slot when the preferred color is actually taken by
        // something this span overlaps; only give up and accept a collision if every slot
        // is taken.
        let preferred = color_cache
            .get(&span.id)
            .unwrap_or_else(|| color_cache.next_color(&accents));
        let color = if used.contains(&preferred) {
            accents
                .into_iter()
                .find(|c| !used.contains(c))
                .unwrap_or(preferred)
        } else {
            preferred
        };
        color_cache.set(span.id, color);

        for region in regions {
            occupied.push((region, color));
        }
        overlays[idx].style.color = color;
    }

    // A path is resolved against whatever is loaded right now, so a batch the crawl adds
    // later picks up its share of the highlight on the next reapply.
    view_state.clear_all_highlights();
    for overlay in overlays.iter() {
        if let Some(path) = overlay.path() {
            let route = path.loaded_route(graph);
            for node in route.nodes {
                view_state.set_node_highlight(node, overlay.style);
            }
            for edge in route.edges {
                view_state.set_edge_highlight(edge, overlay.style);
            }
        }
    }
}

/// Everything besides the overlays themselves that highlights, annotation flags, and floating
/// labels are resolved against: the zoom step (which picks the detail level) and the loaded
/// graph, identified by the active batch and how many nodes have been crawled in. Viewers
/// rebuild that overlay-derived state only when this or their overlays change, instead of on
/// every frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlayInputs {
    zoom_index: usize,
    active_batch: Option<BatchId>,
    loaded_nodes: usize,
}

impl OverlayInputs {
    pub fn current<S: GraphSource<GenGraph>>(
        engine: &LayoutEngine<GenGraph, S>,
        view_state: &GraphViewState<GraphNode>,
    ) -> Self {
        Self {
            zoom_index: view_state.zoom_index,
            active_batch: engine.active_batch(),
            loaded_nodes: engine.graph().node_count(),
        }
    }
}

/// One floating label resolved against the loaded graph, waiting only for a camera to place it.
#[derive(Clone)]
struct PendingLabel {
    locus: GraphLocus,
    text: String,
    color: Color,
}

/// The floating labels for a set of overlays at one detail level, mapped onto the loaded graph
/// and with covered or truncated-away spans already dropped. Only placement depends on the
/// camera, so a viewer rebuilds this when its overlays, zoom level, or loaded batch change and
/// hands it to [`draw_annotation_labels`] every frame.
#[derive(Clone)]
pub struct AnnotationLabels {
    detail_level: VisualDetail,
    labels: Vec<PendingLabel>,
    /// Whether a labelled span was dropped before placement, which counts as hidden.
    any_suppressed: bool,
}

impl Default for AnnotationLabels {
    fn default() -> Self {
        Self {
            detail_level: VisualDetail::Minimal,
            labels: Vec::new(),
            any_suppressed: false,
        }
    }
}

impl AnnotationLabels {
    /// Resolve the labels of `overlays` at `detail_level` against `graph`.
    ///
    /// Overlays are labelled longest-first so the covered-by-later check matches highlight
    /// paint order. A label is suppressed when its span is fully covered by a shorter overlay
    /// on top, or when it collapses into a truncated node.
    pub fn new(graph: &GenGraph, detail_level: VisualDetail, overlays: &[GraphOverlay]) -> Self {
        let mut labeled: Vec<(&AnnotationSpan, PathStyle)> = overlays
            .iter()
            .filter_map(|overlay| {
                overlay
                    .span()
                    .filter(|span| !span.name.is_empty())
                    .filter(|_| {
                        !matches!(
                            (detail_level, &overlay.source),
                            (VisualDetail::Minimal, OverlaySource::Track(_))
                        )
                    })
                    .map(|span| (span, overlay.style))
            })
            .collect();
        let mut result = Self {
            detail_level,
            ..Self::default()
        };
        if labeled.is_empty() {
            return result;
        }
        labeled.sort_by_key(|(span, _)| {
            -(span
                .segments
                .iter()
                .map(|segment| segment.end - segment.start)
                .sum::<i64>())
        });

        let span_refs: Vec<&AnnotationSpan> = labeled.iter().map(|(span, _)| *span).collect();
        let theme = current_theme();
        let loaded = LoadedNodeSlices::new(graph);
        for (idx, (span, style)) in labeled.iter().enumerate() {
            let Some(locus) = graph_locus_from_annotation_span(span, &loaded) else {
                continue;
            };
            if span_covered_by_later(span, idx, &span_refs)
                || (detail_level == VisualDetail::Truncated
                    && !locus_should_show_in_truncated(&locus))
            {
                result.any_suppressed = true;
                continue;
            }
            let color = match style.color {
                Color::Reset => theme[0x06],
                other => other,
            };
            result.labels.push(PendingLabel {
                locus,
                text: span_label_text(span),
                color,
            });
        }
        result
    }
}

/// Draw `labels` near their spans after the graph has been rendered into `buf`.
///
/// A label is hidden when no free cell is found near its span. Returns `true` if any label
/// was hidden here or suppressed when `labels` was built, so the caller can show a single
/// "some annotations hidden" hint.
pub fn draw_annotation_labels(
    buf: &mut Buffer,
    area: Rect,
    view_state: &GraphViewState<GraphNode>,
    labels: &AnnotationLabels,
) -> bool {
    let max_distance = if labels.detail_level == VisualDetail::Minimal {
        10
    } else {
        5
    };
    let mut any_hidden = labels.any_suppressed;
    for label in &labels.labels {
        let Some(bounds) = locus_label_bounds(&label.locus, &view_state.frame, labels.detail_level)
        else {
            continue;
        };
        if draw_label_near_pos(
            buf,
            area,
            bounds,
            &label.text,
            label.color,
            view_state,
            max_distance,
        )
        .is_none()
        {
            any_hidden = true;
        }
    }
    any_hidden
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_graph::GraphEdge;
    use gen_models::{block_group::BlockGroup, sample::Sample};
    use gen_tui::{
        geometry::{WorldPos, WorldRect},
        graph_view::GraphView,
        plotter::LineStyle,
        testing::{TestGraphs, create_test_terminal, mocks::MockDomainGraph},
        viewport_state::{ViewportState, WorldBuffer},
    };
    use petgraph::graph::NodeIndex;
    use ratatui::{backend::TestBackend, widgets::StatefulWidget as _};

    use super::*;
    use crate::{
        imports::gfa::import_gfa,
        test_helpers::setup_gen,
        views::{annotation_track::AnnotationSegment, graph_overlay::OverlayContent},
    };

    const ZOOM_SNAPSHOT_SEQUENCES: [&str; 10] = [
        "A",
        "CG",
        "GAT",
        "TTAC",
        "ACGTA",
        "GATTACA",
        "CCGTAAGCT",
        "TTAACCGGATC",
        "ACGTTGCAACGTAGC",
        "GCTAACGTTAGGCCATGTA",
    ];

    #[derive(Clone, Copy)]
    struct SyntheticSequenceSource;

    impl SequenceSource for SyntheticSequenceSource {
        fn get_node_sequence(
            &self,
            _node_id: HashId,
            start: i64,
            end: i64,
        ) -> Result<String, SequenceError> {
            Ok("ACGTACGTAC"[start as usize..end as usize].to_string())
        }
    }

    struct ZoomSnapshotMinimal;

    impl NodeRenderer<MockDomainGraph> for ZoomSnapshotMinimal {
        fn get_node_size(&self, _node_id: &NodeIndex) -> (u64, u64) {
            (1, 1)
        }

        fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, _node_id: &NodeIndex) {
            buffer.set_char(area.left_center(), NODE_GLYPH);
        }
    }

    struct ZoomSnapshotTruncated;

    impl ZoomSnapshotTruncated {
        fn sequence(node_id: NodeIndex) -> &'static str {
            ZOOM_SNAPSHOT_SEQUENCES[node_id.index()]
        }
    }

    impl NodeRenderer<MockDomainGraph> for ZoomSnapshotTruncated {
        fn get_node_size(&self, node_id: &NodeIndex) -> (u64, u64) {
            (Self::sequence(*node_id).len().min(13) as u64, 1)
        }

        fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &NodeIndex) {
            let sequence = inner_truncation(Self::sequence(*node_id), 13);
            buffer.set_string_styled(area.left_center(), &sequence, Style::default());
        }
    }

    struct ZoomSnapshotFull;

    impl ZoomSnapshotFull {
        fn sequence(node_id: NodeIndex) -> &'static str {
            ZOOM_SNAPSHOT_SEQUENCES[node_id.index()]
        }
    }

    impl NodeRenderer<MockDomainGraph> for ZoomSnapshotFull {
        fn get_node_size(&self, node_id: &NodeIndex) -> (u64, u64) {
            (Self::sequence(*node_id).len() as u64, 1)
        }

        fn render_node(&self, buffer: &mut WorldBuffer, area: WorldRect, node_id: &NodeIndex) {
            buffer.set_string_styled(
                area.left_center(),
                Self::sequence(*node_id),
                Style::default(),
            );
        }
    }

    /// Mirrors [`ZOOM_GAP_SIZES`] paired with the mock renderers above, so
    /// `complex_dag_at_each_zoom_level` exercises the same six-step progression the
    /// production `build_zoom_levels` builds for a real GenGraph.
    fn zoom_snapshot_table() -> Vec<(Box<dyn NodeRenderer<MockDomainGraph>>, GapSizes)> {
        vec![
            (Box::new(ZoomSnapshotMinimal), ZOOM_GAP_SIZES[0]),
            (Box::new(ZoomSnapshotMinimal), ZOOM_GAP_SIZES[1]),
            (Box::new(ZoomSnapshotTruncated), ZOOM_GAP_SIZES[2]),
            (Box::new(ZoomSnapshotFull), ZOOM_GAP_SIZES[3]),
            (Box::new(ZoomSnapshotFull), ZOOM_GAP_SIZES[4]),
            (Box::new(ZoomSnapshotFull), ZOOM_GAP_SIZES[5]),
        ]
    }

    #[test]
    fn complex_dag_at_each_zoom_level() {
        let graph = TestGraphs::domain_complex_dag();

        for (level, (visual, gaps)) in zoom_snapshot_table().into_iter().enumerate() {
            let mut terminal = create_test_terminal(120, 40);
            let mut engine = LayoutEngine::new(graph.clone());
            let mut view_state = GraphViewState::default();
            view_state.zoom_index = level;
            view_state.gaps = gaps;

            terminal
                .draw(|frame| {
                    let area = frame.area();
                    GraphView::new(&mut engine, &visual).render(
                        area,
                        frame.buffer_mut(),
                        &mut view_state,
                    );
                })
                .expect("should render the complex DAG");

            insta::assert_snapshot!(format!("complex_dag_zoom_{level}"), terminal.backend());
        }
    }

    /// Importing a GFA cycle (no explicit path covering it) should not crash
    /// `create_gen_graph_engine`. All back edges in the graph (the GFA link cycle plus
    /// the synthetic `PATH_END -> PATH_START` edge added by import) must be detected and
    /// rewired via `LayoutEngine::new_with_backward_edges` so toposort sees an acyclic
    /// graph.
    #[test]
    fn test_create_gen_graph_engine_handles_circular_genome() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/gfa/cycle_no_path.gfa");
        let collection_name = "cycle".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME).unwrap();

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let graph =
            BlockGroup::get_graph(conn, context.workspace(), &block_group_id, None).unwrap();
        let (mut engine, _visual, _view_state) =
            create_gen_graph_engine(graph, (conn, context.workspace()));
        let anchor = engine
            .default_anchor()
            .expect("circular genome graph should have nodes");
        let node_budget = engine.neighborhood_node_budget(80);
        engine
            .window_for(anchor, node_budget)
            .expect("layout should succeed on a circular genome instead of panicking in toposort");
    }

    #[test]
    fn create_gen_graph_engine_handles_multiple_cycles() {
        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/gfa/multiple_cycles_no_path.gfa");
        let collection_name = "multiple-cycles".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME).unwrap();

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let graph =
            BlockGroup::get_graph(conn, context.workspace(), &block_group_id, None).unwrap();
        let (mut engine, _visual, _view_state) =
            create_gen_graph_engine(graph, (conn, context.workspace()));
        let anchor = engine
            .default_anchor()
            .expect("multiple-cycle graph should have nodes");
        let node_budget = engine.neighborhood_node_budget(80);
        engine.window_for(anchor, node_budget).expect(
            "layout should detect every cycle instead of excluding only the synthetic edge",
        );
    }

    /// Test coordinate handling for very large genomic sequences
    ///
    /// Genomic sequences can span hundreds of thousands of base pairs, creating
    /// world coordinates that exceed u16::MAX (65,535) when rendered. This test
    /// verifies that the coordinate conversion system handles such large values
    /// correctly without integer overflow or wraparound artifacts.
    #[test]
    fn test_coordinate_overflow_with_large_genomic_sequences() {
        // Set up a viewport for rendering genomic data
        let mut viewport_state = ViewportState::new();
        viewport_state.viewport_bounds = ratatui::layout::Rect::new(0, 0, 80, 20);

        // Position camera to simulate viewing a region of a large genome
        // where sequence coordinates naturally reach high values
        let camera_center = WorldPos::new(40000, 0);
        viewport_state.camera_current = camera_center;

        let backend = TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let mut buffer = terminal.current_buffer_mut().clone();
        let world_buffer = WorldBuffer::new(&mut buffer, &viewport_state);

        // Test a coordinate representing the end of a 100K base pair genomic sequence
        // Such large sequences are common in genomics (genes, regulatory regions, etc.)
        let large_genomic_pos = WorldPos::new(camera_center.x + 70000, 0); // ~110K coordinate

        // The coordinate conversion should handle large values gracefully:
        // - Return None if outside viewport (correct behavior)
        // - Never wrap around due to u16 overflow (incorrect behavior)
        let result = world_buffer.world_to_viewport(large_genomic_pos);

        assert!(
            result.is_none(),
            "Large genomic coordinates outside viewport should return None, not wrap around"
        );

        // Verify that normal-sized coordinates still work correctly
        let normal_pos = WorldPos::new(camera_center.x, camera_center.y);
        assert!(
            world_buffer.world_to_viewport(normal_pos).is_some(),
            "Coordinates within normal range should convert successfully"
        );
    }

    #[test]
    fn test_inner_truncation_no_truncation_needed() {
        let s = "hello";
        let truncated = inner_truncation(s, 10);
        assert_eq!(truncated, "hello");
    }

    #[test]
    fn test_inner_truncation_truncate_to_odd_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 7);
        assert_eq!(truncated, "he...ld");
    }

    #[test]
    fn test_inner_truncation_truncate_to_even_length() {
        let s = "hello world";
        let truncated = inner_truncation(s, 8);
        assert_eq!(truncated, "hel...ld");
    }

    #[test]
    fn test_inner_truncation_empty_string() {
        let s = "";
        let truncated = inner_truncation(s, 5);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_inner_truncation_short_target() {
        let s = "hello world";
        let truncated = inner_truncation(s, 3);
        assert_eq!(truncated, NODE_GLYPH.to_string());
    }

    #[test]
    fn snapshot_zygosity_pruned_edges() {
        use std::path::PathBuf;

        use gen_models::sample::Sample;
        use gen_tui::{graph_view::GraphView, testing::create_test_terminal};
        use ratatui::widgets::StatefulWidget as _;

        use crate::{
            imports::fasta::import_fasta, test_helpers::setup_gen_on_disk,
            updates::vcf::update_with_vcf,
        };

        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple.fa")
            .to_str()
            .unwrap()
            .to_string();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_zygosity.vcf")
            .to_str()
            .unwrap()
            .to_string();

        import_fasta(
            &context,
            &fasta_path,
            collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        update_with_vcf(
            &context,
            &vcf_path,
            collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        let gen_graph =
            Sample::get_graph(conn, context.workspace(), collection, "SAMPLE1", None).unwrap();
        let (mut engine, zoom_levels, mut view_state) =
            create_gen_graph_engine(gen_graph, (conn, context.workspace()));
        let visual = &zoom_levels[view_state.zoom_index].1;

        let mut terminal = create_test_terminal(132, 43);
        terminal
            .draw(|f| {
                let area = f.area();
                GraphView::new(&mut engine, visual).render(area, f.buffer_mut(), &mut view_state);
            })
            .unwrap();

        insta::assert_snapshot!("zygosity_pruned_edges", terminal.backend().to_string());
    }

    /// Renders through `SqlGraphSource` (the same lazy-crawl path the live TUI viewer uses for
    /// every non-historical block group - see `views::block_group::load_block_group_graph`),
    /// not the eager `BlockGroup::get_graph`, which the viewers no longer use for this case.
    fn render_gfa_snapshot(gfa_fixture: &str, collection_name: &str) -> String {
        use std::path::PathBuf;

        use gen_models::{block_group::BlockGroup, sample::Sample};
        use gen_tui::{graph_view::GraphView, testing::create_test_terminal};
        use ratatui::widgets::StatefulWidget as _;

        use crate::{
            imports::gfa::import_gfa,
            test_helpers::setup_gen_on_disk,
            views::lazy_graph_source::{SqlGraphSource, seed_block_group_graph},
        };

        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(gfa_fixture);
        import_gfa(&context, &gfa_path, collection_name, Sample::DEFAULT_NAME).unwrap();

        let block_group_id = BlockGroup::get_id(collection_name, Sample::DEFAULT_NAME, "", None);
        let db_path = PathBuf::from(conn.path().expect("graph database has no file path"));
        let source = SqlGraphSource::new(db_path, block_group_id);
        let seed = seed_block_group_graph(conn, &block_group_id);
        let (mut engine, zoom_levels, mut view_state) = create_annotated_gen_graph_engine_lazy(
            seed,
            source,
            (conn, context.workspace()),
            NodeAnnotationLayer::new(),
        );
        let visual = &zoom_levels[view_state.zoom_index].1;

        let mut terminal = create_test_terminal(132, 43);
        terminal
            .draw(|f| {
                let area = f.area();
                GraphView::new(&mut engine, visual).render(area, f.buffer_mut(), &mut view_state);
            })
            .unwrap();
        terminal.backend().to_string()
    }

    #[test]
    fn snapshot_gfa_cycle_no_path() {
        let snapshot = render_gfa_snapshot("fixtures/gfa/cycle_no_path.gfa", "cycle_no_path");
        insta::assert_snapshot!("gfa_cycle_no_path", snapshot);
    }

    #[test]
    fn snapshot_gfa_cycle_with_path() {
        let snapshot = render_gfa_snapshot("fixtures/gfa/cycle_with_path.gfa", "cycle_with_path");
        insta::assert_snapshot!("gfa_cycle_with_path", snapshot);
    }

    fn flag(name: &str, bar_start: i64, bar_end: i64, strand: Strand) -> AnnotationFlag {
        AnnotationFlag {
            id: HashId::convert_str(name),
            piece: 0,
            name: name.to_string(),
            color: Color::Red,
            strand,
            bar_start,
            bar_end,
            continues_left: false,
            continues_right: false,
            show_label: true,
            label: LabelPlacement::Inside,
            lane: 0,
        }
    }

    fn placement_of<'a>(flags: &'a [AnnotationFlag], name: &str) -> &'a AnnotationFlag {
        flags
            .iter()
            .find(|flag| flag.name == name)
            .expect("should contain the named flag")
    }

    #[test]
    fn test_pack_annotation_flags_places_names_inside_or_beside() {
        let mut flags = vec![
            flag("AmpR", 10, 20, Strand::Forward),
            flag("snug", 14, 18, Strand::Forward),
            flag("long name here", 20, 24, Strand::Unknown),
            flag("promoter", 0, 4, Strand::Forward),
            flag("no room for this name", 5, 9, Strand::Reverse),
        ];
        pack_annotation_flags(&mut flags, 25);

        assert_eq!(placement_of(&flags, "AmpR").label, LabelPlacement::Inside);
        assert_eq!(
            placement_of(&flags, "snug").label,
            LabelPlacement::Beside(9),
            "a name as wide as its bar would cover the bar's ends, so it goes beside"
        );
        assert_eq!(
            placement_of(&flags, "long name here").label,
            LabelPlacement::Beside(5),
            "a name wider than its bar goes one cell to the left of the bar"
        );
        assert_eq!(
            placement_of(&flags, "promoter").label,
            LabelPlacement::Beside(5),
            "falls back to the right when the left would spill past the node start"
        );
        assert_eq!(
            placement_of(&flags, "no room for this name").label,
            LabelPlacement::Floating,
            "a name that fits on neither side is left to the floating label pass"
        );
    }

    #[test]
    fn test_pack_annotation_flags_shares_lanes_for_disjoint_extents() {
        let mut flags = vec![
            flag("a", 0, 5, Strand::Forward),
            flag("b", 6, 10, Strand::Forward),
            flag("c", 12, 30, Strand::Reverse),
        ];
        assert_eq!(pack_annotation_flags(&mut flags, 40), 1);
        assert!(flags.iter().all(|flag| flag.lane == 0));

        // Bars touching without a gap column cannot share a lane.
        let mut flags = vec![
            flag("a", 0, 5, Strand::Forward),
            flag("b", 5, 10, Strand::Forward),
        ];
        assert_eq!(pack_annotation_flags(&mut flags, 40), 2);

        // A name beside its bar counts toward the flag's extent.
        let mut flags = vec![
            flag("z", 2, 4, Strand::Unknown),
            flag("abcdef", 10, 12, Strand::Unknown),
        ];
        assert_eq!(pack_annotation_flags(&mut flags, 40), 2);
        assert_eq!(
            placement_of(&flags, "abcdef").label,
            LabelPlacement::Beside(3)
        );

        // Three mutually overlapping bars need three lanes; a fourth disjoint one reuses
        // the first.
        let mut flags = vec![
            flag("a", 0, 20, Strand::Forward),
            flag("b", 5, 25, Strand::Forward),
            flag("c", 10, 30, Strand::Forward),
            flag("d", 22, 30, Strand::Forward),
        ];
        assert_eq!(pack_annotation_flags(&mut flags, 40), 3);
        assert_eq!(placement_of(&flags, "d").lane, 0);
    }

    #[test]
    fn test_annotated_renderer_size_grows_with_lanes() {
        let node = GraphNode {
            node_id: HashId::convert_str("annotated"),
            sequence_start: 0,
            sequence_end: 40,
        };
        let layer = NodeAnnotationLayer::new();
        let renderer = GenGraphAnnotatedRenderer::new(SyntheticSequenceSource, layer.clone());
        assert_eq!(renderer.get_node_size(&node), (40, 1));

        layer.replace(HashMap::from([(
            node,
            vec![flag("a", 0, 10, Strand::Forward)],
        )]));
        assert_eq!(renderer.get_node_size(&node), (40, 3));

        layer.replace(HashMap::from([(
            node,
            vec![
                flag("a", 0, 10, Strand::Forward),
                flag("b", 5, 15, Strand::Forward),
            ],
        )]));
        assert_eq!(
            renderer.get_node_size(&node),
            (40, 5),
            "two lanes under the sequence need a 5-row rect so the sequence stays on the \
             layout's center row"
        );
    }

    #[test]
    fn test_annotated_renderer_size_generation_moves_only_when_lanes_change() {
        let node = GraphNode {
            node_id: HashId::convert_str("annotated"),
            sequence_start: 0,
            sequence_end: 40,
        };
        let layer = NodeAnnotationLayer::new();
        let renderer = GenGraphAnnotatedRenderer::new(SyntheticSequenceSource, layer.clone());
        let initial = renderer.size_generation();

        layer.replace(HashMap::new());
        assert_eq!(renderer.size_generation(), initial);

        layer.replace(HashMap::from([(
            node,
            vec![flag("a", 0, 10, Strand::Forward)],
        )]));
        let one_lane = renderer.size_generation();
        assert_ne!(one_lane, initial);

        // Moving a flag within its lane changes what is drawn, not how big the node is.
        layer.replace(HashMap::from([(
            node,
            vec![flag("a", 2, 12, Strand::Forward)],
        )]));
        assert_eq!(renderer.size_generation(), one_lane);

        layer.replace(HashMap::from([(
            node,
            vec![
                flag("a", 0, 10, Strand::Forward),
                flag("b", 5, 15, Strand::Forward),
            ],
        )]));
        let two_lanes = renderer.size_generation();
        assert_ne!(two_lanes, one_lane);

        layer.replace(HashMap::new());
        assert_ne!(renderer.size_generation(), two_lanes);
    }

    #[derive(Clone, Copy)]
    struct RepeatingSequenceSource;

    impl SequenceSource for RepeatingSequenceSource {
        fn get_node_sequence(
            &self,
            _node_id: HashId,
            start: i64,
            end: i64,
        ) -> Result<String, SequenceError> {
            Ok("ACGT"
                .chars()
                .cycle()
                .skip(start as usize)
                .take((end - start) as usize)
                .collect())
        }
    }

    fn span_overlay(name: &str, segments: Vec<(GraphNode, i64, i64, Strand)>) -> GraphOverlay {
        GraphOverlay {
            content: OverlayContent::Span(AnnotationSpan {
                id: HashId::convert_str(name),
                name: name.to_string(),
                segments: segments
                    .into_iter()
                    .map(|(node, start, end, strand)| AnnotationSegment {
                        node_id: node.node_id,
                        start: node.sequence_start + start,
                        end: node.sequence_start + end,
                        strand,
                    })
                    .collect(),
            }),
            source: OverlaySource::Track("features".to_string()),
            style: PathStyle {
                color: Color::Reset,
                line_style: LineStyle::Normal,
                merge_glyphs: true,
            },
        }
    }

    /// A linear start → a → b → c → end graph with annotations covering every flag case:
    /// a name inside its bar, beside it, clipped inside a bar too narrow for it, a
    /// multi-node span whose cap only shows on its last node, and overlapping features
    /// that pack into a second lane.
    #[test]
    fn snapshot_annotated_full_detail() {
        use gen_tui::{graph_view::GraphView, testing::create_test_terminal};

        let node = |name: &str, length: i64| GraphNode {
            node_id: HashId::convert_str(name),
            sequence_start: 0,
            sequence_end: length,
        };
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let end = GraphNode {
            node_id: PATH_END_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let node_a = node("a", 30);
        let node_b = node("b", 12);
        let node_c = node("c", 30);
        let mut graph = GenGraph::new();
        let edge = |index: usize| {
            vec![GraphEdge {
                edge_id: HashId::convert_str(&format!("edge {index}")),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            }]
        };
        graph.add_edge(start, node_a, edge(0));
        graph.add_edge(node_a, node_b, edge(1));
        graph.add_edge(node_b, node_c, edge(2));
        graph.add_edge(node_c, end, edge(3));

        let mut overlays = vec![
            span_overlay("AmpR", vec![(node_a, 4, 24, Strand::Forward)]),
            span_overlay("ori", vec![(node_a, 25, 30, Strand::Reverse)]),
            span_overlay(
                "lacZ",
                vec![
                    (node_a, 20, 30, Strand::Forward),
                    (node_b, 0, 12, Strand::Forward),
                    (node_c, 0, 5, Strand::Forward),
                ],
            ),
            span_overlay("promoter", vec![(node_b, 2, 6, Strand::Forward)]),
            span_overlay("terminator_region", vec![(node_c, 10, 20, Strand::Unknown)]),
            span_overlay("tag", vec![(node_c, 25, 27, Strand::Reverse)]),
        ];

        let layer = NodeAnnotationLayer::new();
        let (mut engine, zoom_levels, mut view_state) =
            create_annotated_gen_graph_engine(graph, RepeatingSequenceSource, layer.clone());
        apply_zoom_level(&mut view_state, FULL_ZOOM_LEVEL, &zoom_levels);
        let mut colors = AnnotationColorCache::new();

        let mut terminal = create_test_terminal(100, 16);
        let area = terminal.get_frame().area();
        // Two frames: the first establishes the camera and frame index, the second
        // paints with every node placed.
        for _ in 0..2 {
            reapply_overlays(
                &engine,
                &mut view_state,
                &zoom_levels,
                &mut overlays,
                &mut colors,
            );
            let floating = update_node_annotations(&layer, &engine, &overlays);
            let labels = AnnotationLabels::new(
                engine.graph(),
                zoom_levels[view_state.zoom_index].0,
                &floating,
            );
            let visual = &zoom_levels[view_state.zoom_index].1;
            terminal
                .draw(|f| {
                    GraphView::new(&mut engine, visual).render(
                        area,
                        f.buffer_mut(),
                        &mut view_state,
                    );
                    draw_annotation_connectors(
                        f.buffer_mut(),
                        area,
                        &view_state.frame,
                        &layer,
                        None,
                    );
                    draw_annotation_labels(f.buffer_mut(), area, &view_state, &labels);
                })
                .unwrap();
        }

        // The sequence row stays on the rect's center row with the flags under it.
        let rect = view_state
            .frame
            .rect_of(node_a)
            .expect("should have placed node a");
        assert_eq!(
            rect.max.y - rect.min.y + 1,
            5,
            "AmpR and lacZ overlap: two lanes"
        );
        let sequence_y = area.height as i64 - 1 - sequence_row(rect);
        let buffer = terminal.backend().buffer();
        let lane = &buffer[((rect.min.x + 4) as u16, sequence_y as u16 + 1)];
        assert_eq!(lane.symbol(), "═");

        insta::assert_snapshot!("annotated_full_detail", terminal.backend().to_string());
    }

    mod annotation_snapshots {
        use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
        use gen_graph::{GenGraph, GraphEdge, GraphNode};
        use gen_tui::{
            GraphViewState, frame_index::Direction, graph_view::GraphView,
            testing::create_test_terminal,
        };
        use ratatui::{style::Color, widgets::StatefulWidget as _};

        use super::{RepeatingSequenceSource, span_overlay};
        use crate::views::{
            gen_graph_widget::{
                AnnotationLabels, AnnotationStarts, FULL_ZOOM_LEVEL, NodeAnnotationLayer,
                apply_zoom_level, create_annotated_gen_graph_engine, draw_annotation_connectors,
                draw_annotation_labels, draw_braille_curve, reapply_overlays,
                update_node_annotations,
            },
            graph_overlay::{AnnotationColorCache, GraphOverlay},
        };

        fn node(name: &str, length: i64) -> GraphNode {
            GraphNode {
                node_id: HashId::convert_str(name),
                sequence_start: 0,
                sequence_end: length,
            }
        }

        fn graph(edges: &[(GraphNode, GraphNode)]) -> GenGraph {
            let mut graph = GenGraph::new();
            let start = GraphNode {
                node_id: PATH_START_NODE_ID,
                sequence_start: 0,
                sequence_end: 0,
            };
            let end = GraphNode {
                node_id: PATH_END_NODE_ID,
                sequence_start: 0,
                sequence_end: 0,
            };
            let first = edges.first().expect("should have a first edge").0;
            let last = edges.last().expect("should have a last edge").1;
            for (index, (source, target)) in core::iter::once((start, first))
                .chain(edges.iter().copied())
                .chain(core::iter::once((last, end)))
                .enumerate()
            {
                graph.add_edge(
                    source,
                    target,
                    vec![GraphEdge {
                        edge_id: HashId::convert_str(&format!("annotation edge {index}")),
                        source_strand: Strand::Forward,
                        target_strand: Strand::Forward,
                        chromosome_index: 0,
                        phased: 0,
                        created_on: 0,
                    }],
                );
            }
            graph
        }

        // Exercise the viewer's complete overlay pass, including the second frame that
        // uses the placed nodes for connectors and floating labels.
        fn render(
            graph: GenGraph,
            overlays: Vec<GraphOverlay>,
            zoom_level: usize,
            size: (u16, u16),
            focused: Option<HashId>,
        ) -> String {
            render_view(graph, overlays, zoom_level, size, focused).0
        }

        /// `render`, also returning the annotation starts and view state it rendered with.
        fn render_view(
            graph: GenGraph,
            mut overlays: Vec<GraphOverlay>,
            zoom_level: usize,
            size: (u16, u16),
            focused: Option<HashId>,
        ) -> (String, AnnotationStarts, GraphViewState<GraphNode>) {
            let layer = NodeAnnotationLayer::new();
            let (mut engine, zoom_levels, mut view_state) =
                create_annotated_gen_graph_engine(graph, RepeatingSequenceSource, layer.clone());
            apply_zoom_level(&mut view_state, zoom_level, &zoom_levels);
            let mut colors = AnnotationColorCache::new();
            let mut terminal = create_test_terminal(size.0, size.1);
            for _ in 0..2 {
                reapply_overlays(
                    &engine,
                    &mut view_state,
                    &zoom_levels,
                    &mut overlays,
                    &mut colors,
                );
                let floating = update_node_annotations(&layer, &engine, &overlays);
                let labels = AnnotationLabels::new(
                    engine.graph(),
                    zoom_levels[view_state.zoom_index].0,
                    &floating,
                );
                terminal
                    .draw(|frame| {
                        let area = frame.area();
                        GraphView::new(&mut engine, &zoom_levels[view_state.zoom_index].1).render(
                            area,
                            frame.buffer_mut(),
                            &mut view_state,
                        );
                        draw_annotation_connectors(
                            frame.buffer_mut(),
                            area,
                            &view_state.frame,
                            &layer,
                            focused,
                        );
                        draw_annotation_labels(frame.buffer_mut(), area, &view_state, &labels);
                    })
                    .expect("should render annotations");
            }
            let starts = AnnotationStarts::new(&engine, &overlays);
            (terminal.backend().to_string(), starts, view_state)
        }

        #[test]
        fn test_annotation_simple_strands() {
            let sequence = node("sequence", 24);
            let neighbor = node("unannotated", 4);
            for (name, strand) in [
                ("forward", Strand::Forward),
                ("reverse", Strand::Reverse),
                ("unknown", Strand::Unknown),
            ] {
                let snapshot = render(
                    graph(&[(sequence, neighbor)]),
                    vec![span_overlay("gene", vec![(sequence, 0, 24, strand)])],
                    FULL_ZOOM_LEVEL,
                    (44, 7),
                    None,
                );
                insta::assert_snapshot!(format!("annotation_simple_{name}"), snapshot);
            }
        }

        #[test]
        fn test_annotation_packed_labels() {
            // A nonzero sequence origin also exercises translation into local columns.
            let sequence = GraphNode {
                sequence_start: 100,
                sequence_end: 140,
                ..node("packed", 40)
            };
            let neighbor = node("unannotated", 4);
            let snapshot = render(
                graph(&[(sequence, neighbor)]),
                vec![
                    span_overlay("promoter", vec![(sequence, 0, 2, Strand::Forward)]),
                    span_overlay("gene", vec![(sequence, 5, 26, Strand::Forward)]),
                    span_overlay("antisense", vec![(sequence, 8, 28, Strand::Reverse)]),
                    span_overlay("site", vec![(sequence, 30, 31, Strand::Reverse)]),
                    span_overlay("tag", vec![(sequence, 33, 40, Strand::Unknown)]),
                    span_overlay(
                        "a_label_too_long_for_either_side",
                        vec![(sequence, 18, 22, Strand::Unknown)],
                    ),
                ],
                FULL_ZOOM_LEVEL,
                (64, 15),
                None,
            );
            assert!(snapshot.contains("a_label_too_long_for_either_side"));
            insta::assert_snapshot!("annotation_packed_labels", snapshot);
        }

        /// A fork and join carrying spans that cross it on either branch, a nested and a
        /// one-column span, and a span continuing past the join.
        fn branching_spans() -> (GenGraph, Vec<GraphOverlay>) {
            let first = node("first", 16);
            let upper = node("upper", 22);
            let lower = node("lower", 12);
            let merge = node("merge", 16);
            let last = node("last", 10);
            let graph = graph(&[
                (first, upper),
                (first, lower),
                (upper, merge),
                (lower, merge),
                (merge, last),
            ]);
            let overlays = vec![
                span_overlay(
                    "coding",
                    vec![
                        (first, 6, 16, Strand::Forward),
                        (upper, 0, 22, Strand::Forward),
                        (merge, 0, 9, Strand::Forward),
                    ],
                ),
                span_overlay(
                    "reverse",
                    vec![
                        (first, 10, 16, Strand::Reverse),
                        (lower, 0, 12, Strand::Reverse),
                        (merge, 0, 6, Strand::Reverse),
                    ],
                ),
                span_overlay("nested", vec![(upper, 3, 18, Strand::Reverse)]),
                span_overlay("site", vec![(lower, 5, 6, Strand::Forward)]),
                span_overlay(
                    "tail",
                    vec![
                        (merge, 11, 16, Strand::Unknown),
                        (last, 0, 10, Strand::Unknown),
                    ],
                ),
            ];
            (graph, overlays)
        }

        fn braille_cells(rendered: &str) -> Vec<(usize, usize)> {
            rendered
                .lines()
                .enumerate()
                .flat_map(|(row, line)| {
                    line.chars()
                        .enumerate()
                        .filter(|(_, glyph)| ('\u{2801}'..='\u{28FF}').contains(glyph))
                        .map(move |(column, _)| (row, column))
                })
                .collect()
        }

        #[test]
        fn test_annotation_connectors_only_for_focused_span() {
            let (graph, overlays) = branching_spans();
            let render_focused = |focused: Option<&str>| {
                render(
                    graph.clone(),
                    overlays.clone(),
                    FULL_ZOOM_LEVEL,
                    (110, 31),
                    focused.map(HashId::convert_str),
                )
            };
            assert!(
                braille_cells(&render_focused(None)).is_empty(),
                "should draw no connectors without a focused annotation"
            );
            assert!(
                braille_cells(&render_focused(Some("site"))).is_empty(),
                "should draw no connectors for a focused span on a single node"
            );
            let coding = braille_cells(&render_focused(Some("coding")));
            let reverse = braille_cells(&render_focused(Some("reverse")));
            let tail = braille_cells(&render_focused(Some("tail")));
            assert!(!coding.is_empty(), "should connect the focused span");
            assert!(!reverse.is_empty(), "should connect the focused span");
            assert_ne!(coding, reverse, "should connect only the focused span");
            // `tail` joins pieces on adjacent nodes, so its connector spans the gap between them.
            assert!(!tail.is_empty(), "should connect the focused span");
            assert!(
                tail.iter().all(|cell| !coding.contains(cell)),
                "should not draw another span's connector"
            );
        }

        /// The cursor's node and its column within that node.
        fn cursor_column(view_state: &GraphViewState<GraphNode>) -> (GraphNode, i64) {
            let node = view_state.cursor.node.expect("should have a cursor node");
            let rect = view_state
                .frame
                .rect_of(node)
                .expect("should place the cursor node");
            (
                node,
                rect.point_at_fraction(view_state.cursor.fractional).x - rect.left(),
            )
        }

        #[test]
        fn test_annotation_start_jumps_across_a_fork() {
            let (graph, overlays) = branching_spans();
            let (_, annotation_starts, mut view_state) =
                render_view(graph, overlays, FULL_ZOOM_LEVEL, (110, 31), None);
            let (first, upper, lower, merge) = (
                node("first", 16),
                node("upper", 22),
                node("lower", 12),
                node("merge", 16),
            );
            // Forward and unstranded spans start at their first column; reverse-strand spans
            // (`nested`, and `reverse` on `merge`) start at their last column.
            assert_eq!(annotation_starts.on(&first), vec![6]);
            assert_eq!(annotation_starts.on(&upper), vec![17]);
            assert_eq!(annotation_starts.on(&lower), vec![5]);
            assert_eq!(annotation_starts.on(&merge), vec![5, 11]);

            let starts = |node: GraphNode| annotation_starts.on(&node);
            let branch_start = |branch: GraphNode| {
                let column = if branch == upper { 17 } else { 5 };
                (branch, column)
            };
            view_state.cursor.set_node(first, (0.0, 0.5));
            let mut forward = Vec::new();
            while view_state.move_cursor_to_stop(true, starts).is_ok() {
                forward.push(cursor_column(&view_state));
            }
            // At the fork, the jump takes whichever branch the cursor's right-arrow step would.
            let forward_branch = view_state
                .frame
                .neighbor(first, Direction::Right)
                .expect("should have a branch after the fork");
            assert_eq!(
                forward,
                vec![
                    (first, 6),
                    branch_start(forward_branch),
                    (merge, 5),
                    (merge, 11)
                ]
            );
            assert_eq!(
                cursor_column(&view_state),
                (merge, 11),
                "should stay put at the batch's last annotation start"
            );

            let mut backward = Vec::new();
            while view_state.move_cursor_to_stop(false, starts).is_ok() {
                backward.push(cursor_column(&view_state));
            }
            let backward_branch = view_state
                .frame
                .neighbor(merge, Direction::Left)
                .expect("should have a branch before the join");
            assert_eq!(
                backward,
                vec![(merge, 5), branch_start(backward_branch), (first, 6)]
            );
        }

        #[test]
        fn test_annotation_branching_spans() {
            let (graph, overlays) = branching_spans();
            for zoom_level in [FULL_ZOOM_LEVEL, 5] {
                let snapshot = render(graph.clone(), overlays.clone(), zoom_level, (110, 31), None);
                for label in ["coding", "reverse", "nested", "site", "tail"] {
                    assert_eq!(
                        snapshot.matches(label).count(),
                        1,
                        "should label {label} once"
                    );
                }
                insta::assert_snapshot!(
                    format!("annotation_branching_zoom_{zoom_level}"),
                    snapshot
                );
            }
        }

        #[test]
        fn test_annotation_connector_curves() {
            let mut terminal = create_test_terminal(66, 16);
            terminal
                .draw(|frame| {
                    let buffer = frame.buffer_mut();
                    // Paired slopes expose the symmetry; short gaps and level bars cover
                    // the limiting shapes. The crossing must preserve existing graph ink.
                    for (from, to) in [
                        ((2, 1), (26, 6)),
                        ((36, 6), (60, 1)),
                        ((2, 9), (26, 9)),
                        ((36, 9), (39, 14)),
                        ((48, 9), (48, 14)),
                    ] {
                        buffer[(from.0 - 1, from.1)].set_char('═');
                        buffer[(to.0 + 1, to.1)].set_char('═');
                        draw_braille_curve(buffer, from, to, Color::Red);
                    }
                    buffer[(14, 12)].set_char('│');
                    buffer[(19, 12)].set_char('A');
                    draw_braille_curve(buffer, (2, 12), (26, 12), Color::Blue);
                    assert_eq!(buffer[(14, 12)].symbol(), "│");
                    assert_eq!(buffer[(19, 12)].symbol(), "A");
                })
                .expect("should render connector curves");
            insta::assert_snapshot!(
                "annotation_connector_curves",
                terminal.backend().to_string()
            );
        }
    }

    #[test]
    fn test_zero_length_slices_are_invisible_at_every_zoom_level() {
        let junction = GraphNode {
            node_id: HashId::convert_str("zero-length-junction"),
            sequence_start: 3,
            sequence_end: 3,
        };
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let end = GraphNode {
            node_id: PATH_END_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let content = GraphNode {
            node_id: HashId::convert_str("content"),
            sequence_start: 0,
            sequence_end: 5,
        };

        let plain = build_zoom_levels(SyntheticSequenceSource);
        let annotated =
            build_annotated_zoom_levels(SyntheticSequenceSource, NodeAnnotationLayer::new());
        for (detail_level, renderer, _) in plain.iter().chain(annotated.iter()) {
            assert!(
                !renderer.is_visible(&junction),
                "a zero-length slice should be a junction at {detail_level:?}"
            );
            for node in [start, end, content] {
                assert!(
                    renderer.is_visible(&node),
                    "{node:?} should be drawn at {detail_level:?}"
                );
            }
        }
    }

    #[test]
    fn test_join_touching_slices_joins_only_parts_that_meet() {
        let node = GraphNode {
            node_id: HashId::convert_str("plasmid"),
            sequence_start: 0,
            sequence_end: 100,
        };
        let other = GraphNode {
            node_id: HashId::convert_str("other"),
            sequence_start: 0,
            sequence_end: 100,
        };
        let slice = |block: GraphNode, start: usize, end: usize, strand: Strand| GraphNodeSlice {
            block,
            start,
            end,
            strand,
        };

        assert_eq!(
            join_touching_slices(&[
                slice(node, 40, 46, Strand::Forward),
                slice(node, 46, 64, Strand::Forward),
                slice(node, 64, 71, Strand::Forward),
            ]),
            vec![slice(node, 40, 71, Strand::Forward)],
            "forward parts meeting end to end should join"
        );
        assert_eq!(
            join_touching_slices(&[
                slice(node, 60, 70, Strand::Reverse),
                slice(node, 50, 60, Strand::Reverse),
            ]),
            vec![slice(node, 50, 70, Strand::Reverse)],
            "reverse parts meeting end to end should join"
        );
        let wrapping_origin = [
            slice(node, 90, 100, Strand::Forward),
            slice(node, 0, 10, Strand::Forward),
        ];
        assert_eq!(
            join_touching_slices(&wrapping_origin),
            wrapping_origin.to_vec(),
            "the halves of a feature wrapping the origin don't meet"
        );
        let apart = [
            slice(node, 0, 10, Strand::Forward),
            slice(node, 11, 20, Strand::Forward),
            slice(other, 20, 30, Strand::Forward),
            slice(other, 30, 40, Strand::Reverse),
        ];
        assert_eq!(
            join_touching_slices(&apart),
            apart.to_vec(),
            "gaps, other nodes and other strands should stay separate"
        );
    }

    /// A GenBank `join(...)` whose parts meet end to end is drawn as one bar in one lane, and a
    /// feature wrapping the origin is drawn as two capless pieces at the node's ends with one
    /// name between them.
    #[test]
    fn test_node_annotations_join_touching_parts_and_wrap_the_origin() {
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let plasmid = GraphNode {
            node_id: HashId::convert_str("plasmid"),
            sequence_start: 0,
            sequence_end: 100,
        };
        let end = GraphNode {
            node_id: PATH_END_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let mut graph = GenGraph::new();
        graph.add_edge(start, plasmid, Vec::new());
        graph.add_edge(plasmid, end, Vec::new());
        let overlays = vec![
            // Ends exactly where the promoter below starts, like pUC19's AmpR promoter.
            span_overlay("neighbour", vec![(plasmid, 30, 40, Strand::Forward)]),
            span_overlay(
                "promoter",
                vec![
                    (plasmid, 40, 46, Strand::Forward),
                    (plasmid, 46, 64, Strand::Forward),
                    (plasmid, 64, 71, Strand::Forward),
                ],
            ),
            span_overlay(
                "ori",
                vec![
                    (plasmid, 90, 100, Strand::Forward),
                    (plasmid, 0, 10, Strand::Forward),
                ],
            ),
        ];
        let layer = NodeAnnotationLayer::new();
        let (engine, _, _) =
            create_annotated_gen_graph_engine(graph, RepeatingSequenceSource, layer.clone());

        update_node_annotations(&layer, &engine, &overlays);

        let flags = layer.flags(&plasmid);
        let named = |name: &str| {
            let mut named: Vec<&AnnotationFlag> =
                flags.iter().filter(|flag| flag.name == name).collect();
            named.sort_by_key(|flag| flag.piece);
            named
        };
        let promoter = named("promoter");
        assert_eq!(
            promoter.len(),
            1,
            "the promoter's parts should join into one bar"
        );
        assert_eq!((promoter[0].bar_start, promoter[0].bar_end), (40, 71));
        assert!(!promoter[0].continues_left && !promoter[0].continues_right);
        assert!(promoter[0].show_label);

        let ori = named("ori");
        assert_eq!(ori.len(), 2, "the halves of ori should stay two bars");
        assert_eq!((ori[0].bar_start, ori[0].bar_end), (90, 100));
        assert!(!ori[0].continues_left && ori[0].continues_right);
        assert_eq!((ori[1].bar_start, ori[1].bar_end), (0, 10));
        assert!(ori[1].continues_left && !ori[1].continues_right);
        assert_eq!(
            ori.iter().filter(|flag| flag.show_label).count(),
            1,
            "ori should be named once"
        );
    }
}
