use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, is_end_node, is_start_node,
};
use gen_graph::{GenGraph, GraphEdge, GraphNode};
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
use ratatui::{buffer::Buffer, layout::Rect, style::Style};

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
