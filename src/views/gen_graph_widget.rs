use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, is_end_node, is_start_node,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode, GraphNodeSlice};
use gen_models::{db::GraphConnection, locus::GraphLocus, node::Node, sequence::SequenceError};
use gen_tui::{
    cycle_removal::remove_cycles,
    distribute_nodes::GapSizes,
    frame_index::FrameIndex,
    geometry::{WorldPos, WorldRect},
    graph_view::GraphViewState,
    graph_widget::NODE_GLYPH,
    layout::VisualDetail,
    layout_engine::LayoutEngine,
    plotter::{NodeRenderer, PathStyle},
    theme::current_theme,
    viewport_state::WorldBuffer,
};
use petgraph::{
    Direction,
    visit::{DfsEvent, depth_first_search},
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
};

use crate::views::{
    annotation_track::{
        AnnotationSpan, graph_locus_from_annotation_span, span_covered_by_later, span_label_text,
        span_should_hide_in_truncated,
    },
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
        let sequences = Node::get_sequences_by_node_ids(self, &[node_id], None);
        match sequences.get(&node_id) {
            Some(seq) => seq.get_sequence(start, end),
            None => Ok("?".repeat((end - start).max(0) as usize)),
        }
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

/// Compute which edges would be removed by `BlockGroup::prune_graph`.
///
/// Mirrors the per-source-node, per-chromosome_index deduplication logic: for each
/// chromosome_index appearing on outgoing edges of a node, the edge with the highest
/// `created_on` is kept; all others are dimmed. Edges with
/// `PRESERVE_EDIT_SITE_CHROMOSOME_INDEX` are always dimmed; edges with
/// `NO_CHROMOSOME_INDEX` or `INDETERMINATE_CHROMOSOME_INDEX` are never dimmed.
fn compute_pruned_edges(graph: &GenGraph) -> HashSet<(GraphNode, GraphNode)> {
    let mut pruned: HashSet<(GraphNode, GraphNode)> = HashSet::new();

    for node in graph.nodes() {
        // chromosome_index -> (source, target, best_created_on)
        let mut edges_by_ci: HashMap<i64, (GraphNode, GraphNode, i64)> = HashMap::new();

        for (source_node, target_node, edge_weights) in graph.edges(node) {
            for edge_weight in edge_weights {
                let GraphEdge {
                    chromosome_index,
                    created_on,
                    ..
                } = *edge_weight;

                if chromosome_index == NO_CHROMOSOME_INDEX
                    || chromosome_index == INDETERMINATE_CHROMOSOME_INDEX
                {
                    continue;
                }
                if chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
                    pruned.insert((source_node, target_node));
                    continue;
                }
                edges_by_ci
                    .entry(chromosome_index)
                    .and_modify(|(best_src, best_tgt, best_ts)| {
                        if created_on > *best_ts {
                            pruned.insert((*best_src, *best_tgt));
                            *best_src = source_node;
                            *best_tgt = target_node;
                            *best_ts = created_on;
                        } else {
                            pruned.insert((source_node, target_node));
                        }
                    })
                    .or_insert((source_node, target_node, created_on));
            }
        }
    }

    pruned
}

/// Find nodes that become inaccessible when all pruned edges are removed.
///
/// BFS from all start nodes following only non-pruned edges. Any node not reached
/// is only reachable through pruned (lowlighted) edges and should be dimmed.
fn compute_inaccessible_nodes(
    graph: &GenGraph,
    pruned: &HashSet<(GraphNode, GraphNode)>,
) -> Vec<GraphNode> {
    let mut reachable: HashSet<GraphNode> = HashSet::new();
    let mut queue: VecDeque<GraphNode> =
        graph.nodes().filter(|n| is_start_node(n.node_id)).collect();
    for &node in &queue {
        reachable.insert(node);
    }

    while let Some(node) = queue.pop_front() {
        for (src, tgt, _) in graph.edges(node) {
            if !pruned.contains(&(src, tgt)) && reachable.insert(tgt) {
                queue.push_back(tgt);
            }
        }
    }

    graph.nodes().filter(|n| !reachable.contains(n)).collect()
}

/// Collapse redundant reverse-complement representations of the same bidirected GFA link.
///
/// GFA producers commonly emit both `A+ -> B+` and its equivalent reverse-complement
/// `B- -> A-`. `GenGraph` has unoriented nodes, so retaining both makes that one adjacency
/// look like a directed two-node cycle. Keep the direction selected by the cycle ordering
/// and remove only backward edges whose strand metadata proves that the opposite edge is
/// the same bidirected link.
fn collapse_reverse_complement_edges(graph: &mut GenGraph) {
    let backward_edges = remove_cycles(&*graph, None, None).backward_edges;
    let redundant_edges: Vec<(GraphNode, GraphNode)> = backward_edges
        .into_iter()
        .filter(|(source, target)| source != target)
        .filter(|(source, target)| {
            let Some(source_edges) = graph.edge_weight(*source, *target) else {
                return false;
            };
            let Some(target_edges) = graph.edge_weight(*target, *source) else {
                return false;
            };
            source_edges.iter().all(|source_edge| {
                target_edges.iter().any(|target_edge| {
                    source_edge.source_strand == target_edge.target_strand.complement()
                        && source_edge.target_strand == target_edge.source_strand.complement()
                })
            })
        })
        .collect();

    for (source, target) in redundant_edges {
        graph.remove_edge(source, target);
    }
}

/// Extract enough backward edges to make the layout input acyclic.
///
/// When GFA import supplied the synthetic `PATH_END -> PATH_START` marker, preserve that
/// marker as the rendered circular-genome loop and discard the equivalent raw closure.
/// Then scan for any additional cycles instead of assuming the synthetic marker was the
/// graph's only cycle.
fn extract_backward_edges(graph: &mut GenGraph) -> Vec<(GraphNode, GraphNode)> {
    let end_node = graph.nodes().find(|node| is_end_node(node.node_id));
    let start_node = graph.nodes().find(|node| is_start_node(node.node_id));
    let synthetic_edge = if let (Some(end_node), Some(start_node)) = (end_node, start_node)
        && graph.contains_edge(end_node, start_node)
    {
        let predecessors: Vec<GraphNode> = graph
            .neighbors_directed(end_node, Direction::Incoming)
            .collect();
        let successors: Vec<GraphNode> = graph
            .neighbors_directed(start_node, Direction::Outgoing)
            .collect();
        for predecessor in &predecessors {
            for successor in &successors {
                graph.remove_edge(*predecessor, *successor);
            }
        }
        graph.remove_edge(end_node, start_node);
        Some((end_node, start_node))
    } else {
        None
    };

    let mut backward_edges = Vec::new();
    let starts: Vec<GraphNode> = graph.nodes().collect();
    depth_first_search(&*graph, starts, |event| {
        if let DfsEvent::BackEdge(source, target) = event {
            backward_edges.push((source, target));
        }
        petgraph::visit::Control::<()>::Continue
    });
    backward_edges.extend(synthetic_edge);
    backward_edges
}

/// Create a `LayoutEngine`/`ZoomLevels`/`GraphViewState` triple for a GenGraph with the
/// standard theme and settings.
///
/// This is the standard way to initialize a `GraphView` for GenGraph visualization: it dims
/// pruned edges and inaccessible nodes, starts at [`DEFAULT_ZOOM_LEVEL`], and starts in
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
    mut graph: GenGraph,
    source: S,
) -> (
    LayoutEngine<GenGraph>,
    ZoomLevels<'a>,
    GraphViewState<GraphNode>,
) {
    collapse_reverse_complement_edges(&mut graph);
    let backward_edges = extract_backward_edges(&mut graph);
    let pruned = compute_pruned_edges(&graph);
    let inaccessible = compute_inaccessible_nodes(&graph, &pruned);
    let levels = build_zoom_levels(source);
    let engine = if backward_edges.is_empty() {
        LayoutEngine::new(graph)
    } else {
        LayoutEngine::new_with_backward_edges(graph, &backward_edges)
    };

    let mut view_state = GraphViewState::default();
    for edge in pruned {
        view_state.dim_edge(edge);
    }
    for node in inaccessible {
        view_state.dim_node(node);
    }
    apply_zoom_level(&mut view_state, DEFAULT_ZOOM_LEVEL, &levels);
    view_state.hide_cursor();

    (engine, levels, view_state)
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
) {
    let detail_level = levels[view_state.zoom_index.min(levels.len() - 1)].0;

    for s in &m.slices {
        let block_seq_len = s.block.length();
        let col_start_raw = s.start as i64;
        let col_end_raw = s.end.saturating_sub(1) as i64;
        let (col_start, col_end) = (
            clamp_col(col_start_raw, block_seq_len, detail_level),
            clamp_col(col_end_raw, block_seq_len, detail_level),
        );
        view_state.set_cell_highlight(s.block, (col_start, 0), (col_end, 0), style);
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

/// Re-register every overlay highlight on `view_state`, replacing whatever highlights were
/// previously set.
///
/// Span overlays are processed longest-first so shorter (inner) spans paint on top; any
/// path overlay is applied last so the route paints over the span tints. Each span's color
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
/// `overlays` is written back with the colors actually used, so `draw_annotation_labels`
/// (which reads `overlay.style` separately, after this runs) labels each span in the same
/// color that got painted. Callers run this after any change that invalidates mapped
/// highlight columns (zoom, detail change) or, in the live TUI viewers, every frame because
/// the overlay set changes with scrolling.
pub fn reapply_overlays<R>(
    engine: &LayoutEngine<GenGraph>,
    view_state: &mut GraphViewState<GraphNode>,
    levels: &[(VisualDetail, R, GapSizes)],
    overlays: &mut [GraphOverlay],
    color_cache: &mut AnnotationColorCache,
) {
    let detail_level = levels[view_state.zoom_index.min(levels.len() - 1)].0;
    let graph = engine.graph();

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
                        || !span_should_hide_in_truncated(span, graph)
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

    // Decide every span's locus and color first (only needs `&engine`); painting
    // (`&mut view_state`) happens in a second pass once every color is settled.
    let accents = accent_colors();
    let mut occupied: Vec<(CellRegion, Color)> = Vec::new();
    let mut decisions: Vec<(usize, GraphLocus, Color)> = Vec::new();
    for idx in span_indices {
        let span = overlays[idx]
            .span()
            .expect("filtered to span overlays above");
        let Some(locus) = graph_locus_from_annotation_span(span, graph) else {
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
        decisions.push((idx, locus, color));
    }

    view_state.clear_all_highlights();
    for (idx, locus, color) in &decisions {
        overlays[*idx].style.color = *color;
        highlight_match_range(view_state, levels, locus, overlays[*idx].style);
    }
    for overlay in overlays.iter() {
        if let Some(nodes) = overlay.path_nodes() {
            view_state.set_path_highlight(overlay.style, nodes.to_vec());
        }
    }
}

/// Draw floating labels for `overlays` after the graph has been rendered into `buf`.
///
/// Overlays are labelled longest-first so the covered-by-later check matches highlight
/// paint order. A label is suppressed when its span is fully covered by a shorter overlay
/// on top, when it collapses into a truncated node, or when no free cell is found near its
/// span. Returns `true` if any labelled overlay was suppressed, so the caller can show a
/// single "some annotations hidden" hint.
pub fn draw_annotation_labels<R>(
    buf: &mut Buffer,
    area: Rect,
    engine: &LayoutEngine<GenGraph>,
    view_state: &GraphViewState<GraphNode>,
    levels: &[(VisualDetail, R, GapSizes)],
    overlays: &[GraphOverlay],
) -> bool {
    let detail_level = levels[view_state.zoom_index.min(levels.len() - 1)].0;
    let graph = engine.graph();
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
    if labeled.is_empty() {
        return false;
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
    let max_distance = if detail_level == VisualDetail::Minimal {
        10
    } else {
        5
    };

    let mut any_hidden = false;
    for (idx, (span, style)) in labeled.iter().enumerate() {
        let Some(locus) = graph_locus_from_annotation_span(span, graph) else {
            continue;
        };
        if span_covered_by_later(span, idx, &span_refs) {
            any_hidden = true;
            continue;
        }
        if detail_level == VisualDetail::Truncated && span_should_hide_in_truncated(span, graph) {
            any_hidden = true;
            continue;
        }
        let Some(bounds) = locus_label_bounds(&locus, &view_state.frame, detail_level) else {
            continue;
        };
        let color = match style.color {
            Color::Reset => theme[0x06],
            other => other,
        };
        let label = span_label_text(span);
        if draw_label_near_pos(buf, area, bounds, &label, color, view_state, max_distance).is_none()
        {
            any_hidden = true;
        }
    }
    any_hidden
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_core::Strand;
    use gen_models::{block_group::BlockGroup, sample::Sample};
    use gen_tui::{
        geometry::{WorldPos, WorldRect},
        graph_view::GraphView,
        testing::{TestGraphs, create_test_terminal, mocks::MockDomainGraph},
        viewport_state::{ViewportState, WorldBuffer},
    };
    use petgraph::graph::NodeIndex;
    use ratatui::{backend::TestBackend, widgets::StatefulWidget as _};

    use super::*;
    use crate::{imports::gfa::import_gfa, test_helpers::setup_gen};

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
        let graph = BlockGroup::get_graph(conn, &block_group_id, None).unwrap();
        let (mut engine, _visual, _view_state) = create_gen_graph_engine(graph, conn);
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
        let graph = BlockGroup::get_graph(conn, &block_group_id, None).unwrap();
        let (mut engine, _visual, _view_state) = create_gen_graph_engine(graph, conn);
        let anchor = engine
            .default_anchor()
            .expect("multiple-cycle graph should have nodes");
        let node_budget = engine.neighborhood_node_budget(80);
        engine.window_for(anchor, node_budget).expect(
            "layout should detect every cycle instead of excluding only the synthetic edge",
        );
    }

    #[test]
    fn collapses_redundant_reverse_complement_links() {
        let source = GraphNode {
            node_id: HashId::convert_str("source"),
            sequence_start: 0,
            sequence_end: 3,
        };
        let target = GraphNode {
            node_id: HashId::convert_str("target"),
            sequence_start: 0,
            sequence_end: 3,
        };
        let edge = |edge_id, source_strand, target_strand| GraphEdge {
            edge_id: HashId::convert_str(edge_id),
            source_strand,
            target_strand,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            created_on: 0,
        };
        let mut graph = GenGraph::new();
        graph.add_edge(
            source,
            target,
            vec![edge("forward", Strand::Forward, Strand::Forward)],
        );
        graph.add_edge(
            target,
            source,
            vec![edge("reverse", Strand::Reverse, Strand::Reverse)],
        );

        collapse_reverse_complement_edges(&mut graph);

        assert_eq!(graph.edge_count(), 1);
        assert_ne!(
            graph.contains_edge(source, target),
            graph.contains_edge(target, source)
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

        let gen_graph = Sample::get_graph(conn, collection, "SAMPLE1", None).unwrap();
        let (mut engine, zoom_levels, mut view_state) = create_gen_graph_engine(gen_graph, conn);
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

    fn render_gfa_snapshot(gfa_fixture: &str, collection_name: &str) -> String {
        use std::path::PathBuf;

        use gen_models::{block_group::BlockGroup, sample::Sample};
        use gen_tui::{graph_view::GraphView, testing::create_test_terminal};
        use ratatui::widgets::StatefulWidget as _;

        use crate::{imports::gfa::import_gfa, test_helpers::setup_gen_on_disk};

        let context = setup_gen_on_disk();
        let conn = context.graph().conn();

        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(gfa_fixture);
        import_gfa(&context, &gfa_path, collection_name, Sample::DEFAULT_NAME).unwrap();

        let block_group_id = BlockGroup::get_id(collection_name, Sample::DEFAULT_NAME, "", None);
        let graph = BlockGroup::get_graph(conn, &block_group_id, None).unwrap();
        let (mut engine, zoom_levels, mut view_state) = create_gen_graph_engine(graph, conn);
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
}
