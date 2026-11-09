use std::{
    collections::{HashMap, HashSet},
    error::Error,
    io::Write,
    time::{Duration, Instant},
};

use crossterm::event::{
    self, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use gen_core::{HashId, Workspace, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode, PackedGenGraph};
use gen_models::{block_group::BlockGroup, db::GraphConnection, history::dolt::active_branch};
use gen_tui::{
    crawl::GraphSource,
    graph_view::GraphViewState,
    kitty_graphics::{KittyPixelOverlay, supports_kitty_graphics},
    layout_engine::{BatchId, LayoutEngine},
    theme::current_theme,
};
use log::warn;
use ratatui::{
    layout::{Constraint, Direction, HorizontalAlignment, Layout, Position, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, List, ListItem, Padding, Paragraph, Wrap},
};

use crate::{
    progress_bar::{get_handler, get_time_elapsed_bar},
    views::{
        annotation_files::AnnotationFileEntry,
        annotation_loading::{
            AnnotationLoadCoordinator, AnnotationLoadIdentity, AnnotationLoadResult,
            AnnotationLoadSpec,
        },
        annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
        collection::{CollectionExplorer, CollectionExplorerState, FocusZone},
        gen_graph_controller::{ClickOutcome, GenGraphController, GraphKeyOutcome, WorldSync},
        graph_database::GraphDatabase,
        graph_overlay::{
            GraphOverlay, OverlaySource, file_track_key, group_track_key, remove_track_overlays,
            replace_track_overlays,
        },
        lazy_graph_source::{EagerOrProjectedSource, seed_block_group_graph},
        panels::{render_status_bar, render_with_optional_clear},
        region_search::{
            RegionSearchContext, RegionSearchMatch, RegionSearchTask, RegionSearchWorker,
            RegionSearchWorkerPoll, RegionSearchWorkerResult, activate_search_match,
            remove_search_overlay,
        },
        tui_runtime::TuiSession,
    },
};

#[derive(Debug, Default)]
struct RegionSearchState {
    query: String,
    matches: Vec<RegionSearchMatch>,
    selected_match: Option<usize>,
    completed_query: Option<String>,
    completed_feedback: Option<String>,
    focused: bool,
    generation: u64,
    search_due: Option<Instant>,
    selection_pending: bool,
}

impl RegionSearchState {
    fn clear_search_state(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.search_due = None;
        self.selection_pending = false;
        self.clear_cached_results();
    }

    fn schedule_query(&mut self, now: Instant) {
        self.generation = self.generation.wrapping_add(1);
        self.clear_cached_results();
        self.selection_pending = false;
        self.search_due = (!self.query.trim().is_empty()).then_some(now + REGION_SEARCH_DEBOUNCE);
    }

    fn blur(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.search_due = None;
        self.selection_pending = false;
        self.focused = false;
    }

    fn clear_matches(&mut self) {
        self.matches.clear();
        self.selected_match = None;
    }

    fn clear_cached_results(&mut self) {
        self.clear_matches();
        self.completed_query = None;
        self.completed_feedback = None;
    }

    fn complete_query(&mut self, result: Result<Vec<RegionSearchMatch>, String>) {
        self.completed_query = Some(self.query.trim().to_string());
        match result {
            Ok(matches) => {
                self.matches = matches;
                self.selected_match = None;
                self.completed_feedback = None;
            }
            Err(feedback) => {
                self.clear_matches();
                self.completed_feedback = Some(feedback);
            }
        }
    }

    fn move_selection(&mut self, delta: isize) {
        if self.matches.is_empty() {
            self.selected_match = None;
            return;
        }
        let count = self.matches.len() as isize;
        self.selected_match = Some(match self.selected_match {
            Some(selected_match) => ((selected_match as isize + delta).rem_euclid(count)) as usize,
            None if delta < 0 => self.matches.len() - 1,
            None => 0,
        });
    }

    fn handle_key_at(&mut self, key: KeyEvent, now: Instant) -> RegionSearchInputAction {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('u') {
            self.query.clear();
            self.clear_search_state();
            return RegionSearchInputAction::Cleared;
        }
        match key.code {
            KeyCode::Char(character) => {
                self.query.push(character);
                self.schedule_query(now);
                RegionSearchInputAction::Changed
            }
            KeyCode::Backspace => {
                if self.query.pop().is_some() {
                    if self.query.is_empty() {
                        self.clear_search_state();
                        RegionSearchInputAction::Cleared
                    } else {
                        self.schedule_query(now);
                        RegionSearchInputAction::Changed
                    }
                } else {
                    RegionSearchInputAction::Ignored
                }
            }
            KeyCode::Up => {
                self.move_selection(-1);
                RegionSearchInputAction::Ignored
            }
            KeyCode::Down => {
                self.move_selection(1);
                RegionSearchInputAction::Ignored
            }
            KeyCode::Enter if !self.selection_pending => self
                .selected_match
                .and_then(|selected_match| self.matches.get(selected_match))
                .cloned()
                .map_or(RegionSearchInputAction::Ignored, |region_match| {
                    RegionSearchInputAction::Selected(Box::new(region_match))
                }),
            KeyCode::Enter => RegionSearchInputAction::Ignored,
            KeyCode::Esc => {
                self.blur();
                RegionSearchInputAction::Closed
            }
            _ => RegionSearchInputAction::Ignored,
        }
    }

    fn due_query(&mut self, now: Instant) -> Option<(u64, String)> {
        if self.search_due.is_some_and(|search_due| now >= search_due) {
            self.search_due = None;
            Some((self.generation, self.query.trim().to_string()))
        } else {
            None
        }
    }
}

const REGION_SEARCH_DEBOUNCE: Duration = Duration::from_millis(150);

fn is_region_search_command(key_code: KeyCode) -> bool {
    key_code == KeyCode::Char('g')
}

fn focus_region_search(state: &mut RegionSearchState, now: Instant) {
    state.focused = true;
    if state.completed_query.as_deref() != Some(state.query.trim()) {
        state.schedule_query(now);
    }
}

fn current_region_search_context(
    workspace: &Workspace,
    collection_name: &str,
    current_block_group: Option<&BlockGroup>,
    history_ref: Option<&str>,
    annotation_files: &[AnnotationFileEntry],
    active_annotation_files: &HashSet<HashId>,
) -> Option<RegionSearchContext> {
    current_block_group.map(|block_group| {
        RegionSearchContext::new(
            workspace,
            collection_name,
            block_group.sample_name.as_str(),
            block_group,
            history_ref,
            annotation_files,
            active_annotation_files,
        )
    })
}

#[derive(Clone, Debug)]
enum RegionSearchInputAction {
    Changed,
    Cleared,
    Selected(Box<RegionSearchMatch>),
    Closed,
    Ignored,
}

fn queue_region_search_task(
    worker: &RegionSearchWorker,
    task_in_flight: &mut bool,
    pending_task: &mut Option<RegionSearchTask>,
    task: RegionSearchTask,
) -> Result<(), String> {
    if *task_in_flight {
        *pending_task = Some(task);
        Ok(())
    } else {
        worker.submit(task)?;
        *task_in_flight = true;
        Ok(())
    }
}

fn select_region_search_match(
    state: &mut RegionSearchState,
    worker: &RegionSearchWorker,
    context: Option<RegionSearchContext>,
    task_in_flight: &mut bool,
    pending_task: &mut Option<RegionSearchTask>,
    search_match: RegionSearchMatch,
) -> Result<Option<RegionSearchMatch>, String> {
    if state.selection_pending {
        return Ok(None);
    }
    if search_match.needs_selection_resolution() {
        let context = context.ok_or_else(|| "select a block group first".to_string())?;
        let task = RegionSearchTask::resolve_selection(state.generation, context, search_match);
        queue_region_search_task(worker, task_in_flight, pending_task, task)?;
        state.selection_pending = true;
        Ok(None)
    } else {
        Ok(Some(search_match))
    }
}

fn search_match_window_start(selected_match: Option<usize>, match_count: usize) -> usize {
    selected_match
        .unwrap_or(0)
        .saturating_sub(4)
        .min(match_count.saturating_sub(5))
}

fn is_current_search_generation(state: &RegionSearchState, generation: u64) -> bool {
    state.focused && state.generation == generation
}

#[derive(Debug, Eq, PartialEq)]
struct RegionSearchViewIdentity {
    collection_name: String,
    selected_block_group_id: Option<HashId>,
    current_block_group_id: Option<HashId>,
    current_block_group_name: Option<String>,
    current_sample_name: Option<String>,
    history_ref: Option<String>,
    active_file_ids: Vec<HashId>,
    annotation_files: Vec<AnnotationFileEntry>,
}

impl RegionSearchViewIdentity {
    fn new(
        collection_name: &str,
        selected_block_group_id: Option<HashId>,
        current_block_group: Option<&BlockGroup>,
        history_ref: Option<&str>,
        annotation_files: &[AnnotationFileEntry],
        active_annotation_files: &HashSet<HashId>,
    ) -> Self {
        let mut active_file_ids = active_annotation_files.iter().copied().collect::<Vec<_>>();
        active_file_ids.sort_unstable();
        Self {
            collection_name: collection_name.to_string(),
            selected_block_group_id,
            current_block_group_id: current_block_group.map(|block_group| block_group.id),
            current_block_group_name: current_block_group
                .map(|block_group| block_group.name.clone()),
            current_sample_name: current_block_group
                .map(|block_group| block_group.sample_name.clone()),
            history_ref: history_ref.map(str::to_string),
            active_file_ids,
            annotation_files: annotation_files.to_vec(),
        }
    }
}

// Frequency by which we check for external updates to the db
const REFRESH_INTERVAL: u64 = 3; // seconds
const MESSAGE_BUFFER_LIMIT: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelMode {
    Details,
    Messages,
}

/// A block group's graph as a viewer opens it.
pub(crate) struct BlockGroupGraph {
    /// The full graph for a historical view, otherwise just the seed the crawl grows from.
    pub graph: GenGraph,
    /// What the viewer's `LayoutEngine` crawls through.
    pub source: EagerOrProjectedSource,
}

/// Load `block_group_id`'s graph and the source its `LayoutEngine` crawls through.
///
/// A historical view (`history_ref: Some(_)`) always eager-loads the full graph up front:
/// the port queries the lazy path below is built on can only answer for the live graph.
/// Otherwise, the graph is seeded with just its `PATH_START` sentinel and grown lazily from
/// SQLite as the viewer's crawl pushes past its frontier - see `ProjectedGraphSource`, which is what
/// turns opening a large block group from a full-graph-materializing stall into an
/// near-instant open.
pub(crate) fn load_block_group_graph(
    database: &mut GraphDatabase,
    block_group_id: &HashId,
    history_ref: Option<&str>,
    prune_history: bool,
) -> Result<BlockGroupGraph, Box<dyn Error>> {
    if history_ref.is_some() {
        let workspace = database.workspace().clone();
        let graph = BlockGroup::get_graph(
            database.connection()?,
            &workspace,
            block_group_id,
            history_ref,
        )?;
        let projection = crate::views::junction_projection::project_junctions(&graph);
        return Ok(BlockGroupGraph {
            graph: projection.graph,
            source: EagerOrProjectedSource::Eager {
                surviving_routes: projection.surviving_routes,
            },
        });
    }
    let source = database.graph_source(*block_group_id, prune_history);
    let conn = database.connection()?;
    Ok(BlockGroupGraph {
        graph: seed_block_group_graph(conn, block_group_id),
        source: EagerOrProjectedSource::Projected(Box::new(source)),
    })
}

/// Node IDs in the currently active crawled neighborhood (excluding terminal start/end
/// nodes) - the same deliberately-constrained local window `LayoutEngine` already
/// subsets the graph to, so annotation loading has no need to subset any further by
/// what happens to be on-screen right now.
pub(crate) fn active_neighborhood_node_ids<S: GraphSource<PackedGenGraph>>(
    engine: &LayoutEngine<PackedGenGraph, S>,
) -> HashSet<HashId> {
    let Some(world) = engine.active_world() else {
        return HashSet::new();
    };
    world
        .members()
        .map(|node| node.node_id)
        .filter(|&id| !is_start_node(id) && !is_end_node(id))
        .collect()
}

/// Compute the coordinate window (min sequence start, max sequence end) spanned by the
/// currently active crawled neighborhood.
pub(crate) fn active_neighborhood_coordinate_window<S: GraphSource<PackedGenGraph>>(
    engine: &LayoutEngine<PackedGenGraph, S>,
) -> Option<(i64, i64)> {
    let world = engine.active_world()?;
    let mut start = i64::MAX;
    let mut end = i64::MIN;

    for node in world.members() {
        if is_start_node(node.node_id) || is_end_node(node.node_id) {
            continue;
        }
        start = start.min(node.sequence_start);
        end = end.max(node.sequence_end);
    }

    (start <= end).then_some((start, end))
}

pub(crate) fn expand_query_window(window: (i64, i64)) -> (i64, i64) {
    let span = (window.1 - window.0).max(1);
    (window.0.saturating_sub(span), window.1.saturating_add(span))
}

/// Mirror a [`GroupReload`](crate::views::gen_graph_controller::GroupReload) into the
/// sidebar's active annotation groups and the message bar.
fn apply_group_reload(
    sync: WorldSync,
    explorer_state: &mut CollectionExplorerState,
    messages: &mut crate::views::messages::MessageBuffer,
) {
    let Some(group_reload) = sync.group_reload else {
        return;
    };
    explorer_state.active_annotation_groups.clear();
    explorer_state
        .active_annotation_groups
        .extend(group_reload.loaded);
    for warning in group_reload.warnings {
        messages.push_warn(warning);
    }
}

/// Everything `handle_annotation_toggle_requests` needs to service a pending file/group
/// toggle from the sidebar - grouped into one struct since it's all `&`-borrowed context
/// gathered from several owning locals in `view_block_group`, shared unchanged between
/// the keyboard and mouse call sites.
struct AnnotationToggleContext<'a, S: GraphSource<PackedGenGraph>> {
    conn: &'a GraphConnection,
    history_ref: Option<&'a str>,
    collection_name: &'a str,
    current_block_group: Option<&'a BlockGroup>,
    graph_engine: &'a LayoutEngine<PackedGenGraph, S>,
    explorer: &'a CollectionExplorer,
    is_loading: bool,
}

/// Service a pending annotation file or group toggle request left on `explorer_state` by
/// the sidebar's input/mouse handling. Shared by the keyboard and mouse event branches in
/// `view_block_group`'s event loop, which both route sidebar interaction through the same
/// `CollectionExplorerState` toggle-request fields. File tracks load in the background, so
/// activating one only queues the load; groups still load inline from the active neighborhood.
fn handle_annotation_toggle_requests<S: GraphSource<PackedGenGraph>>(
    ctx: &AnnotationToggleContext<S>,
    explorer_state: &mut CollectionExplorerState,
    overlays: &mut Vec<GraphOverlay>,
    annotation_loads: &mut AnnotationLoadCoordinator,
    annotation_file_index_available: &mut HashMap<HashId, bool>,
    annotation_file_loaded_windows: &mut HashMap<HashId, (i64, i64)>,
    messages: &mut crate::views::messages::MessageBuffer,
) {
    if let Some(toggled_id) = explorer_state.annotation_file_toggle_requested.take() {
        if explorer_state.is_annotation_file_active(&toggled_id) {
            if !ctx.is_loading
                && let Some(entry) = ctx.explorer.annotation_file_entry(&toggled_id)
                && let Some(block_group) = ctx.current_block_group
            {
                let node_filter = active_neighborhood_node_ids(ctx.graph_engine);
                let context = AnnotationFileActivationContext {
                    collection_name: ctx.collection_name,
                    history_ref: ctx.history_ref,
                    block_group,
                    node_filter: &node_filter,
                    query_window: active_neighborhood_coordinate_window(ctx.graph_engine)
                        .map(expand_query_window),
                };
                annotation_loads.queue_activation(annotation_file_load_spec(&context, entry));
            }
        } else {
            annotation_loads.cancel_file(toggled_id);
            remove_track_overlays(overlays, &file_track_key(&toggled_id));
            annotation_file_index_available.remove(&toggled_id);
            annotation_file_loaded_windows.remove(&toggled_id);
        }
    }

    if let Some(toggled_group) = explorer_state.annotation_group_toggle_requested.take() {
        if explorer_state.is_annotation_group_active(&toggled_group) {
            if ctx.current_block_group.is_some() {
                let node_ids = active_neighborhood_node_ids(ctx.graph_engine);
                let entry = ctx.explorer.annotation_group_entry(&toggled_group);
                let spans = match entry.map(|entry| {
                    load_annotations_for_group(&AnnotationGroupTrackRequest {
                        conn: ctx.conn,
                        history_ref: ctx.history_ref,
                        entry,
                        projection_graph: ctx.graph_engine.graph(),
                        node_ids: &node_ids,
                    })
                }) {
                    Some(Ok(spans)) => spans,
                    Some(Err(err)) => {
                        messages.push_warn(format!(
                            "Failed to load annotations for group {toggled_group}: {err}"
                        ));
                        Vec::new()
                    }
                    None => Vec::new(),
                };
                if spans.is_empty() {
                    explorer_state.deactivate_annotation_group(&toggled_group);
                } else {
                    replace_track_overlays(overlays, &group_track_key(&toggled_group), spans);
                }
            }
        } else {
            remove_track_overlays(overlays, &group_track_key(&toggled_group));
        }
    }
}

/// Shared inputs for loading an annotation file's track, common to activating one file
/// (on checkbox toggle) and auto-activating every available file on load. `node_filter` is the
/// active crawled neighborhood at queue time, so the worker never materializes the full graph.
struct AnnotationFileActivationContext<'a> {
    collection_name: &'a str,
    history_ref: Option<&'a str>,
    block_group: &'a BlockGroup,
    node_filter: &'a HashSet<HashId>,
    query_window: Option<(i64, i64)>,
}

fn annotation_file_load_spec(
    context: &AnnotationFileActivationContext,
    entry: &AnnotationFileEntry,
) -> AnnotationLoadSpec {
    AnnotationLoadSpec {
        collection_name: context.collection_name.to_string(),
        sample_name: context.block_group.sample_name.clone(),
        block_group_id: context.block_group.id,
        block_group_name: context.block_group.name.clone(),
        history_ref: context.history_ref.map(str::to_string),
        query_window: context.query_window,
        node_filter: context.node_filter.clone(),
        entry: entry.clone(),
    }
}

/// Queue automatic file loading without holding up input or redraws.
/// Files are activated fresh whenever the viewer opens, the collection or block group changes, or
/// the file list changes, so every listed file is offered to the next activation pass.
fn annotation_file_ids(entries: &[AnnotationFileEntry]) -> Vec<HashId> {
    entries.iter().map(|entry| entry.file_addition.id).collect()
}

fn auto_activate_annotation_files_for_viewport(
    context: &AnnotationFileActivationContext,
    entries: &[AnnotationFileEntry],
    explorer_state: &mut CollectionExplorerState,
    annotation_file_index_available: &HashMap<HashId, bool>,
    annotation_loads: &mut AnnotationLoadCoordinator,
) {
    for entry in entries {
        let id = entry.file_addition.id;
        let active = explorer_state.is_annotation_file_active(&id);
        if !active {
            explorer_state.active_annotation_files.insert(id);
        }
        if (!active || !annotation_file_index_available.contains_key(&id))
            && !annotation_loads.has_current_work_for_file(id)
        {
            annotation_loads.queue_activation(annotation_file_load_spec(context, entry));
        }
    }
}

fn annotation_file_result_matches_view(
    identity: &AnnotationLoadIdentity,
    current_collection_name: &str,
    selected_block_group_id: Option<HashId>,
    current_block_group: Option<&BlockGroup>,
    history_ref: Option<&str>,
) -> bool {
    identity.collection_name == current_collection_name
        && selected_block_group_id == Some(identity.block_group_id)
        && identity.history_ref.as_deref() == history_ref
        && current_block_group.is_some_and(|block_group| {
            identity.block_group_id == block_group.id
                && identity.block_group_name == block_group.name
                && identity.sample_name == block_group.sample_name
        })
}

fn annotation_result_window_covers(
    indexed: bool,
    loaded_window: Option<(i64, i64)>,
    visible_window: Option<(i64, i64)>,
) -> bool {
    if !indexed || visible_window.is_none() {
        return true;
    }
    let (Some((loaded_start, loaded_end)), Some((visible_start, visible_end))) =
        (loaded_window, visible_window)
    else {
        return false;
    };
    loaded_start <= visible_start && loaded_end >= visible_end
}

struct AnnotationLoadView<'a> {
    collection_name: &'a str,
    selected_block_group_id: Option<HashId>,
    block_group: Option<&'a BlockGroup>,
    history_ref: Option<&'a str>,
    visible_window: Option<(i64, i64)>,
    explorer: &'a CollectionExplorer,
}

struct AnnotationLoadUiState<'a> {
    explorer_state: &'a mut CollectionExplorerState,
    overlays: &'a mut Vec<GraphOverlay>,
    index_available: &'a mut HashMap<HashId, bool>,
    loaded_windows: &'a mut HashMap<HashId, (i64, i64)>,
    messages: &'a mut crate::views::messages::MessageBuffer,
}

fn apply_annotation_load_result(
    result: AnnotationLoadResult,
    annotation_loads: &mut AnnotationLoadCoordinator,
    view: AnnotationLoadView<'_>,
    state: AnnotationLoadUiState<'_>,
) {
    let file_id = result.identity.entry.file_addition.id;
    if !annotation_loads.is_current(&result.identity)
        || !state.explorer_state.is_annotation_file_active(&file_id)
        || !annotation_file_result_matches_view(
            &result.identity,
            view.collection_name,
            view.selected_block_group_id,
            view.block_group,
            view.history_ref,
        )
        || view.explorer.annotation_file_entry(&file_id) != Some(&result.identity.entry)
    {
        return;
    }

    match result.result {
        Ok(load) => {
            state.index_available.insert(file_id, load.index_available);
            if !annotation_result_window_covers(
                load.index_available,
                load.loaded_window,
                view.visible_window,
            ) {
                return;
            }
            replace_track_overlays(
                state.overlays,
                &file_track_key(&file_id),
                load.track.annotations,
            );
            if let Some(window) = load.loaded_window {
                state.loaded_windows.insert(file_id, window);
                annotation_loads.cancel_pending_reloads(file_id);
            } else {
                state.loaded_windows.remove(&file_id);
            }
        }
        Err(error) => {
            state.messages.push_warn(format!(
                "Failed to load annotation file {}: {error}",
                result.identity.entry.display_name
            ));
            annotation_loads.cancel_file(file_id);
            state.explorer_state.deactivate_annotation_file(&file_id);
            remove_track_overlays(state.overlays, &file_track_key(&file_id));
            state.index_available.remove(&file_id);
            state.loaded_windows.remove(&file_id);
        }
    }
}

/// Handle a click on a wormhole (`NodeRole::Wormhole`) stub: `boundary` is the node inside the
/// current window the stub is attached to, `target` is the off-screen domain node it leads to
/// (see `GraphViewState::wormhole_hit`).
///
/// If another batch already owns `target`, that batch's world reopens; otherwise a new batch is
/// claimed around `target`. Either way `boundary` stays with the batch we are leaving, so the
/// world we arrive in shows a door straight back to it.
///
/// Direction (which edge of the new window to enter from) is decided by whether `target` is a
/// successor or predecessor of `boundary` (`LayoutEngine::is_successor`): exiting toward a
/// successor enters the new window from the left, exiting toward a predecessor enters from the
/// right - the same direction you'd naturally keep moving in.
pub(crate) fn teleport_through_wormhole<S: GraphSource<PackedGenGraph>>(
    graph_engine: &mut LayoutEngine<PackedGenGraph, S>,
    graph_view_state: &mut GraphViewState<GraphNode>,
    boundary: GraphNode,
    target: GraphNode,
) {
    // Direction is always about what we're actually leaving (the clicked stub's own boundary
    // node) versus what we're heading toward (its target). `boundary` and `target` are always
    // directly adjacent (that's what makes them an external-edge pair), so whether we're
    // heading "ahead" is just whether `target` is a successor of `boundary` - no whole-graph
    // rank needed.
    let exits_toward_successor = graph_engine.is_successor(boundary, target);
    let entry_fraction = if exits_toward_successor {
        (0.0, 0.5)
    } else {
        (1.0, 0.5)
    };
    graph_engine.remember_wormhole_choice(boundary, target, exits_toward_successor);
    // Also record the reverse, so `target`'s door on the side facing `boundary` leads straight
    // back to where we came from rather than to whichever neighbor on that side sorts lowest.
    graph_engine.remember_wormhole_choice(target, boundary, !exits_toward_successor);

    let node_budget =
        graph_engine.neighborhood_node_budget(graph_view_state.last_area_width() as usize);
    if graph_engine
        .activate_batch_containing(target, node_budget)
        .is_err()
    {
        return;
    }
    let Some(anchor) = graph_engine.active_world().map(|world| world.anchor()) else {
        return;
    };
    graph_view_state.go_to_node_framed(anchor, target, entry_fraction);

    if exits_toward_successor {
        graph_view_state.queue_snap_left();
    } else {
        graph_view_state.queue_snap_right();
    }
}

/// How long the event loop may sleep before background work needs another look. Annotation
/// loads and region searches finish on worker threads, and a debounced search query comes due
/// on a timer, so neither produces an input event of its own.
fn background_work_wait(
    annotation_loads: &AnnotationLoadCoordinator,
    search_state: &RegionSearchState,
    search_is_working: bool,
) -> Duration {
    let mut wait = if annotation_loads.is_busy() {
        Duration::from_millis(75)
    } else {
        Duration::from_secs(3600)
    };
    if let Some(search_due) = search_state.search_due {
        wait = wait.min(search_due.saturating_duration_since(Instant::now()));
    }
    if search_is_working || search_state.selection_pending {
        wait = wait.min(Duration::from_millis(75));
    }
    wait
}

/// Read and drop every input event already waiting, returning whether one was a resize. The
/// viewers call it after drawing the world a door led into, since input that queued up while
/// that world loaded was aimed at the screen the user left.
pub(crate) fn discard_pending_input() -> std::io::Result<bool> {
    let mut resized = false;
    while event::poll(Duration::ZERO)? {
        resized |= matches!(event::read()?, event::Event::Resize(..));
    }
    Ok(resized)
}

/// Initial graph selection and navigation for the full-screen viewer.
pub struct BlockGroupViewOptions<'a> {
    /// Graph to select when opening the viewer.
    pub name: Option<String>,
    /// Sample containing the selected graph.
    pub sample_name: Option<String>,
    /// Collection to browse.
    pub collection_name: &'a str,
    /// Requested node ID and offset.
    pub position: Option<String>,
    /// Historical revision to display, if requested.
    pub history_ref: Option<&'a str>,
    /// A graph already on screen, such as the inline widget's, to keep drawing instead of
    /// loading the selected graph again.
    pub controller: Option<Box<GenGraphController>>,
}

/// The controller the full-screen viewer starts on: `handed_over` when given, otherwise a new
/// one, with `selected` open unless it already is.
///
/// A handed-over controller keeps its loaded batches, overlays and annotation groups. Its
/// camera was placed for the previous viewer's area, so it is reframed on the cursor node.
fn starting_controller(
    conn: &GraphConnection,
    workspace: &Workspace,
    history_ref: Option<&str>,
    handed_over: Option<Box<GenGraphController>>,
    selected: Option<&BlockGroup>,
) -> Result<GenGraphController, Box<dyn Error>> {
    let mut controller = match handed_over {
        Some(controller) => *controller,
        None => GenGraphController::new(
            GraphDatabase::for_connection(conn, workspace)?,
            history_ref.map(String::from),
        ),
    };
    if let Some(selected) = selected
        && controller.block_group().map(|open| open.id) != Some(selected.id)
    {
        controller.open_block_group(&selected.id)?;
    }
    controller.reframe_on_cursor();
    Ok(controller)
}

pub fn view_block_group<'a>(
    conn: &'a GraphConnection,
    config_conn: &gen_models::db::ConfigConnection,
    workspace: &'a Workspace,
    options: BlockGroupViewOptions<'a>,
) -> Result<(), Box<dyn Error>> {
    let BlockGroupViewOptions {
        name,
        sample_name,
        collection_name,
        position,
        history_ref,
        controller: handed_over,
    } = options;
    let progress_bar = get_handler();
    let bar = progress_bar.add(get_time_elapsed_bar());
    let _ = progress_bar.println("Loading block group");

    let mut focus_zone = FocusZone::Sidebar;
    let mut explorer_state = CollectionExplorerState::new();
    if let Some(ref s) = sample_name {
        explorer_state.set_sample_expanded(s, true);
    }

    let selected = match (name, sample_name.as_ref()) {
        (Some(name), Some(sample_name)) => Some(
            BlockGroup::get_by_name(conn, collection_name, sample_name, &name, history_ref)
                .unwrap_or_else(|_| {
                    panic!(
                        "No block group found with name {:?} and sample {:?} in collection {} ",
                        name, sample_name, collection_name
                    )
                }),
        ),
        _ => None,
    };
    // The graph, its view, dimming and overlays; drawn into the canvas area each frame.
    let mut controller =
        starting_controller(conn, workspace, history_ref, handed_over, selected.as_ref())?;
    let block_group_id = controller.block_group().map(|block_group| block_group.id);
    if block_group_id.is_some() {
        explorer_state.selected_block_group_id = block_group_id;
        focus_zone = FocusZone::Canvas;
    }
    explorer_state.active_annotation_groups.extend(
        controller
            .loaded_annotation_groups()
            .map(ToString::to_string),
    );

    bar.finish();

    let mut messages = crate::views::messages::MessageBuffer::new(MESSAGE_BUFFER_LIMIT);
    let mut annotation_file_index_available: HashMap<HashId, bool> = HashMap::new();
    let mut annotation_file_loaded_windows: HashMap<HashId, (i64, i64)> = HashMap::new();
    let mut current_block_group = controller.block_group().cloned();
    let mut current_collection_name = collection_name.to_string();

    // Create explorer and its state that persists across frames
    let mut explorer = CollectionExplorer::new(
        conn,
        config_conn,
        sample_name.as_deref(),
        current_block_group.as_ref(),
        collection_name,
        history_ref,
    );

    let mut pending_annotation_file_loads = annotation_file_ids(&explorer.data.annotation_files);

    // Create the graph controller and initial graph
    let bar = progress_bar.add(get_time_elapsed_bar());

    // TODO: Handle origin positioning - not directly supported in new widget yet
    if position.is_some() {
        warn!("Origin positioning not yet supported in GenGraphController");
    }

    bar.finish();

    let worker_branch = if history_ref.is_none() {
        Some(active_branch(conn)?)
    } else {
        None
    };
    let mut annotation_loads = AnnotationLoadCoordinator::new(
        workspace.graph_db_path()?,
        workspace.clone(),
        worker_branch.clone(),
        history_ref.map(str::to_string),
    )?;
    let region_search_worker = RegionSearchWorker::new(
        workspace.graph_db_path()?,
        worker_branch,
        history_ref.map(str::to_string),
    )?;

    // Setup terminal
    let mut session = TuiSession::enter()?;
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;
    let terminal = session.terminal_mut();

    // Basic event loop
    let mut kitty_overlay = KittyPixelOverlay::default();
    // The terminal is asked once, in raw mode and before any event is read, because its answer
    // arrives on standard input. Curved edges are drawn through the Kitty graphics protocol, so
    // a terminal without it keeps the rectilinear style.
    let curved_edges_available = supports_kitty_graphics();
    controller.set_curved_edges_available(curved_edges_available);
    let mut show_panel = false;
    let mut panel_mode = PanelMode::Details;
    let show_sidebar = true;
    let mut tui_layout_change = false;

    // Mouse drag state
    let mut mouse_last_pos: Option<(u16, u16)> = None;
    let mut mouse_is_dragging = false;
    let mut last_sidebar_area = Rect::default();
    let mut last_search_area = Rect::default();
    let mut last_search_dropdown_area = Rect::default();
    let mut search_state = RegionSearchState::default();
    let mut search_error: Option<String> = None;
    let mut search_task_in_flight = false;
    let mut pending_search_task: Option<RegionSearchTask> = None;
    let mut search_view_identity: Option<RegionSearchViewIdentity> = None;

    // Track the last selected block group to detect changes
    let mut last_selected_block_group_id = block_group_id;
    // Track if we're loading a new block group
    let mut is_loading = false;
    let mut last_refresh = Instant::now();
    let mut should_quit = false;
    // Mouse capture reports every pointer movement; only events that can change the screen
    // earn a redraw.
    let mut needs_redraw = true;
    let mut annotation_loads_were_busy = false;
    // The batch the active annotation files were last projected onto.
    let mut annotation_files_batch: Option<BatchId> = None;
    // Set when a key or click takes a door. Input after it is left unread until the new world
    // is drawn, then discarded.
    let mut entered_door = false;
    loop {
        // Drain ALL pending input events before doing any work
        while !entered_door && crossterm::event::poll(Duration::from_millis(0))? {
            let input_event = event::read()?;
            if !matches!(
                input_event,
                event::Event::Mouse(MouseEvent {
                    kind: MouseEventKind::Moved,
                    ..
                })
            ) {
                needs_redraw = true;
            }
            if let event::Event::Mouse(mouse) = &input_event
                && matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
                && search_state.focused
                && !last_search_area.contains(Position {
                    x: mouse.column,
                    y: mouse.row,
                })
                && !last_search_dropdown_area.contains(Position {
                    x: mouse.column,
                    y: mouse.row,
                })
            {
                search_state.blur();
                pending_search_task = None;
                search_error = None;
            }
            match input_event {
                event::Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if search_state.focused && !matches!(key.code, KeyCode::Tab | KeyCode::BackTab)
                    {
                        match search_state.handle_key_at(key, Instant::now()) {
                            RegionSearchInputAction::Changed => search_error = None,
                            RegionSearchInputAction::Cleared => {
                                search_error = None;
                                pending_search_task = None;
                                remove_search_overlay(controller.overlays_mut());
                            }
                            RegionSearchInputAction::Selected(search_match) => {
                                let context = current_region_search_context(
                                    workspace,
                                    &current_collection_name,
                                    current_block_group.as_ref(),
                                    history_ref,
                                    &explorer.data.annotation_files,
                                    &explorer_state.active_annotation_files,
                                );
                                match select_region_search_match(
                                    &mut search_state,
                                    &region_search_worker,
                                    context,
                                    &mut search_task_in_flight,
                                    &mut pending_search_task,
                                    *search_match,
                                ) {
                                    Ok(Some(search_match)) => {
                                        match activate_search_match(
                                            &mut controller,
                                            &search_match,
                                            conn,
                                            workspace,
                                        ) {
                                            Ok(()) => {
                                                search_state.blur();
                                                pending_search_task = None;
                                                search_error = None;
                                                focus_zone = FocusZone::Canvas;
                                            }
                                            Err(error) => search_error = Some(error),
                                        }
                                    }
                                    Ok(None) => search_error = None,
                                    Err(error) => search_error = Some(error),
                                }
                            }
                            RegionSearchInputAction::Closed => {
                                search_state.blur();
                                pending_search_task = None;
                                search_error = None;
                            }
                            RegionSearchInputAction::Ignored => {}
                        }
                        continue;
                    }

                    if search_state.focused {
                        search_state.blur();
                        pending_search_task = None;
                        search_error = None;
                    }

                    if is_region_search_command(key.code) {
                        if !search_state.focused {
                            focus_region_search(&mut search_state, Instant::now());
                            search_error = search_state.completed_feedback.clone();
                        }
                        continue;
                    }

                    // Any keyboard navigation shows the cursor.
                    if !controller.view_state().is_cursor_visible()
                        && matches!(
                            key.code,
                            KeyCode::Left
                                | KeyCode::Right
                                | KeyCode::Up
                                | KeyCode::Down
                                | KeyCode::Char('h' | 'j' | 'k' | 'l' | 'w' | 'b' | 'e')
                        )
                    {
                        controller.view_state_mut().show_cursor();
                    }

                    // Global handlers
                    match key.code {
                        KeyCode::Char('q') => {
                            should_quit = true;
                            break;
                        }
                        KeyCode::Char('m') => {
                            if show_panel && panel_mode == PanelMode::Messages {
                                show_panel = false;
                                focus_zone = FocusZone::Canvas;
                            } else {
                                show_panel = true;
                                panel_mode = PanelMode::Messages;
                                focus_zone = FocusZone::Panel;
                            }
                            tui_layout_change = true;
                        }
                        KeyCode::Tab => {
                            // Tab - cycle forwards
                            focus_zone = match focus_zone {
                                FocusZone::Canvas => {
                                    if show_panel {
                                        FocusZone::Panel
                                    } else {
                                        FocusZone::Sidebar
                                    }
                                }
                                FocusZone::Sidebar => FocusZone::Canvas,
                                FocusZone::Panel => FocusZone::Sidebar,
                            }
                        }
                        KeyCode::BackTab => {
                            // Shift+Tab - cycle backwards
                            focus_zone = match focus_zone {
                                FocusZone::Canvas => FocusZone::Sidebar,
                                FocusZone::Sidebar => {
                                    if show_panel {
                                        FocusZone::Panel
                                    } else {
                                        FocusZone::Canvas
                                    }
                                }
                                FocusZone::Panel => FocusZone::Canvas,
                            }
                        }
                        _ => {}
                    }

                    // Focus-specific handlers
                    match focus_zone {
                        FocusZone::Canvas => match key.code {
                            KeyCode::Enter => {
                                // TODO: Node selection not yet supported, always show panel for now
                                show_panel = true;
                                panel_mode = PanelMode::Details;
                                focus_zone = FocusZone::Panel;
                                tui_layout_change = true;
                            }
                            KeyCode::Esc => {
                                if !controller.view_state().is_cursor_visible() {
                                    controller.view_state_mut().show_cursor();
                                } else if !show_panel {
                                    focus_zone = FocusZone::Sidebar;
                                }
                            }
                            // Zoom, the path toggle, annotation stops and navigation
                            // behave the same in every viewer.
                            _ => {
                                let outcome = controller.handle_key(key);
                                if key.code == KeyCode::Char('s')
                                    && outcome == GraphKeyOutcome::Ignore
                                {
                                    messages.push_warn(
                                        "curved edges need a terminal that supports the Kitty graphics protocol"
                                            .to_string(),
                                    );
                                    needs_redraw = true;
                                }
                                entered_door = outcome == GraphKeyOutcome::EnteredDoor;
                            }
                        },
                        FocusZone::Panel => match key.code {
                            KeyCode::Esc => {
                                show_panel = false;
                                focus_zone = FocusZone::Canvas;
                                tui_layout_change = true;
                            }
                            KeyCode::Char('c') if panel_mode == PanelMode::Messages => {
                                messages.clear();
                            }
                            _ => {}
                        },
                        FocusZone::Sidebar => {
                            explorer.handle_input(&mut explorer_state, key);
                            // Check if focus change was requested by the explorer
                            if let Some(requested_zone) = explorer_state.focus_change_requested {
                                focus_zone = requested_zone;
                                explorer_state.focus_change_requested = None;
                            }
                            let (graph_engine, overlays) = controller.engine_and_overlays_mut();
                            handle_annotation_toggle_requests(
                                &AnnotationToggleContext {
                                    conn,
                                    history_ref,
                                    collection_name: &current_collection_name,
                                    current_block_group: current_block_group.as_ref(),
                                    graph_engine,
                                    explorer: &explorer,
                                    is_loading,
                                },
                                &mut explorer_state,
                                overlays,
                                &mut annotation_loads,
                                &mut annotation_file_index_available,
                                &mut annotation_file_loaded_windows,
                                &mut messages,
                            );
                        }
                    }
                }
                event::Event::Mouse(mouse)
                    if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
                        && (last_search_area.contains(Position {
                            x: mouse.column,
                            y: mouse.row,
                        }) || (search_state.focused
                            && last_search_dropdown_area.contains(Position {
                                x: mouse.column,
                                y: mouse.row,
                            }))) =>
                {
                    let was_focused = search_state.focused;
                    if !was_focused {
                        focus_region_search(&mut search_state, Instant::now());
                        search_error = search_state.completed_feedback.clone();
                    }
                    focus_zone = FocusZone::Canvas;
                    if last_search_dropdown_area.contains(Position {
                        x: mouse.column,
                        y: mouse.row,
                    }) {
                        let row = mouse
                            .row
                            .saturating_sub(last_search_dropdown_area.top() + 1)
                            as usize
                            + search_match_window_start(
                                search_state.selected_match,
                                search_state.matches.len(),
                            );
                        if row < search_state.matches.len() {
                            search_state.selected_match = Some(row);
                            let search_match = search_state.matches[row].clone();
                            let context = current_region_search_context(
                                workspace,
                                &current_collection_name,
                                current_block_group.as_ref(),
                                history_ref,
                                &explorer.data.annotation_files,
                                &explorer_state.active_annotation_files,
                            );
                            match select_region_search_match(
                                &mut search_state,
                                &region_search_worker,
                                context,
                                &mut search_task_in_flight,
                                &mut pending_search_task,
                                search_match,
                            ) {
                                Ok(Some(search_match)) => {
                                    match activate_search_match(
                                        &mut controller,
                                        &search_match,
                                        conn,
                                        workspace,
                                    ) {
                                        Ok(()) => {
                                            search_state.blur();
                                            pending_search_task = None;
                                            search_error = None;
                                            focus_zone = FocusZone::Canvas;
                                        }
                                        Err(error) => search_error = Some(error),
                                    }
                                }
                                Ok(None) => search_error = None,
                                Err(error) => search_error = Some(error),
                            }
                        }
                    }
                }
                event::Event::Mouse(mouse)
                    if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
                        && last_sidebar_area.contains(Position {
                            x: mouse.column,
                            y: mouse.row,
                        }) =>
                {
                    mouse_last_pos = None;
                    mouse_is_dragging = false;
                    focus_zone = FocusZone::Sidebar;
                    explorer.handle_mouse(&mut explorer_state, mouse.column, mouse.row);
                    if let Some(requested_zone) = explorer_state.focus_change_requested {
                        focus_zone = requested_zone;
                        explorer_state.focus_change_requested = None;
                    }
                    let (graph_engine, overlays) = controller.engine_and_overlays_mut();
                    handle_annotation_toggle_requests(
                        &AnnotationToggleContext {
                            conn,
                            history_ref,
                            collection_name: &current_collection_name,
                            current_block_group: current_block_group.as_ref(),
                            graph_engine,
                            explorer: &explorer,
                            is_loading,
                        },
                        &mut explorer_state,
                        overlays,
                        &mut annotation_loads,
                        &mut annotation_file_index_available,
                        &mut annotation_file_loaded_windows,
                        &mut messages,
                    );
                }
                event::Event::Mouse(mouse) if focus_zone == FocusZone::Canvas => match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) => {
                        mouse_last_pos = Some((mouse.column, mouse.row));
                        mouse_is_dragging = false;
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        if let Some((last_x, last_y)) = mouse_last_pos {
                            let dx = mouse.column as i16 - last_x as i16;
                            let dy = mouse.row as i16 - last_y as i16;
                            controller.pan(dx, dy);
                            mouse_is_dragging = true;
                        }
                        mouse_last_pos = Some((mouse.column, mouse.row));
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        if !mouse_is_dragging
                            && controller.click(mouse.column, mouse.row)
                                == ClickOutcome::EnteredDoor
                        {
                            entered_door = true;
                        }
                        mouse_last_pos = None;
                        mouse_is_dragging = false;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        if should_quit {
            break;
        }

        if let Some(selected_collection) = explorer_state.collection_change_requested.take() {
            annotation_loads.invalidate_view();
            current_collection_name = selected_collection;
            explorer_state.selected_block_group_id = None;
            last_selected_block_group_id = None;
            current_block_group = None;
            controller = GenGraphController::new(
                GraphDatabase::for_connection(conn, workspace)?,
                history_ref.map(String::from),
            );
            controller.set_curved_edges_available(curved_edges_available);
            explorer.refresh(
                conn,
                config_conn,
                None,
                None,
                &current_collection_name,
                history_ref,
            );
            explorer.force_reload(&mut explorer_state);
            explorer_state.list_state.select(Some(0));
            explorer_state.retain_annotation_files(&explorer.data.annotation_files);
            pending_annotation_file_loads = annotation_file_ids(&explorer.data.annotation_files);
            explorer_state.retain_annotation_groups(&explorer.data.annotation_groups);
            annotation_file_index_available.clear();
            annotation_file_loaded_windows.clear();
            explorer_state.active_annotation_groups.clear();
            is_loading = false;
            last_refresh = Instant::now();
            search_state.focused = false;
            search_state.clear_search_state();
            pending_search_task = None;
            search_error = None;
            needs_redraw = true;
        }

        // Trigger reload if selection changed to a new block group
        if explorer_state.selected_block_group_id != last_selected_block_group_id {
            annotation_loads.invalidate_view();
            is_loading = true;
            last_selected_block_group_id = explorer_state.selected_block_group_id;
        }

        // Refresh explorer data and force reload on change.
        // Skipped when loading — we want the draw to happen first so the loading
        // indicator is shown without any extra latency.
        // I do this every REFRESH_INTERVAL seconds.
        if !is_loading && last_refresh.elapsed() >= Duration::from_secs(REFRESH_INTERVAL) {
            let previous_annotation_files = explorer.data.annotation_files.clone();
            let selected_sample = current_block_group
                .as_ref()
                .map(|bg| bg.sample_name.as_str());
            if explorer.refresh(
                conn,
                config_conn,
                selected_sample,
                current_block_group.as_ref(),
                &current_collection_name,
                history_ref,
            ) {
                let annotation_files_changed =
                    previous_annotation_files != explorer.data.annotation_files;
                explorer.force_reload(&mut explorer_state);
                explorer_state.retain_annotation_files(&explorer.data.annotation_files);
                explorer_state.retain_annotation_groups(&explorer.data.annotation_groups);
                needs_redraw = true;
                annotation_file_index_available
                    .retain(|id, _| explorer_state.is_annotation_file_active(id));
                annotation_file_loaded_windows
                    .retain(|id, _| explorer_state.is_annotation_file_active(id));
                let active_file_keys: HashSet<String> = explorer_state
                    .active_annotation_files
                    .iter()
                    .map(file_track_key)
                    .collect();
                controller
                    .overlays_mut()
                    .retain(|overlay| match &overlay.source {
                        OverlaySource::Track(key) if key.starts_with("file:") => {
                            active_file_keys.contains(key)
                        }
                        OverlaySource::Track(key) => {
                            key.strip_prefix("group:").is_some_and(|group_id| {
                                explorer_state.is_annotation_group_active(group_id)
                            })
                        }
                        _ => true,
                    });
                if annotation_files_changed {
                    annotation_loads.invalidate_view();
                    annotation_file_index_available.clear();
                    annotation_file_loaded_windows.clear();
                    pending_annotation_file_loads =
                        annotation_file_ids(&explorer.data.annotation_files);
                }
            }
            last_refresh = Instant::now();
        }

        // Queue indexed file reloads after the viewport leaves their loaded window. Only one
        // request per file can wait behind the worker, so repeated pan events replace that file's
        // queued window instead of building a backlog. The neighborhood's window and node ids
        // are only worth computing while an indexed file is active.
        let has_indexed_annotation_file = explorer.data.annotation_files.iter().any(|entry| {
            let id = entry.file_addition.id;
            explorer_state.is_annotation_file_active(&id)
                && annotation_file_index_available
                    .get(&id)
                    .copied()
                    .unwrap_or(false)
        });
        if !is_loading
            && has_indexed_annotation_file
            && let Some(block_group) = current_block_group.as_ref()
            && let Some(visible_window) = active_neighborhood_coordinate_window(controller.engine())
        {
            let query_window = expand_query_window(visible_window);
            let node_filter = active_neighborhood_node_ids(controller.engine());
            for entry in &explorer.data.annotation_files {
                let id = entry.file_addition.id;
                if !explorer_state.is_annotation_file_active(&id) {
                    continue;
                }
                if !annotation_file_index_available
                    .get(&id)
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }

                let needs_reload = match annotation_file_loaded_windows.get(&id) {
                    Some((loaded_start, loaded_end)) => {
                        visible_window.0 < *loaded_start || visible_window.1 > *loaded_end
                    }
                    None => true,
                };

                if !needs_reload {
                    continue;
                }
                if annotation_loads.has_current_request_covering(id, visible_window) {
                    continue;
                }
                let context = AnnotationFileActivationContext {
                    history_ref,
                    collection_name: &current_collection_name,
                    block_group,
                    node_filter: &node_filter,
                    query_window: Some(query_window),
                };
                annotation_loads.queue_reload(annotation_file_load_spec(&context, entry));
            }
        }

        // A file's features are projected onto the nodes of the batch that was active when it
        // loaded, so entering another batch through a door reloads every active file against
        // the new batch. Indexed files need this too: their loaded window can still cover the
        // new batch's coordinates while its nodes are not the ones that were projected.
        let active_batch = controller.engine().active_batch();
        if !is_loading
            && active_batch.is_some()
            && active_batch != annotation_files_batch
            && let Some(block_group) = current_block_group.as_ref()
        {
            annotation_files_batch = active_batch;
            let node_filter = active_neighborhood_node_ids(controller.engine());
            let query_window =
                active_neighborhood_coordinate_window(controller.engine()).map(expand_query_window);
            let context = AnnotationFileActivationContext {
                history_ref,
                collection_name: &current_collection_name,
                block_group,
                node_filter: &node_filter,
                query_window,
            };
            for entry in &explorer.data.annotation_files {
                let id = entry.file_addition.id;
                if explorer_state.is_annotation_file_active(&id) {
                    annotation_file_loaded_windows.remove(&id);
                    annotation_loads.queue_reload(annotation_file_load_spec(&context, entry));
                }
            }
        }

        let visible_window = active_neighborhood_coordinate_window(controller.engine());
        while let Some(result) = annotation_loads.try_receive() {
            needs_redraw = true;
            apply_annotation_load_result(
                result,
                &mut annotation_loads,
                AnnotationLoadView {
                    collection_name: &current_collection_name,
                    selected_block_group_id: explorer_state.selected_block_group_id,
                    block_group: current_block_group.as_ref(),
                    history_ref,
                    visible_window,
                    explorer: &explorer,
                },
                AnnotationLoadUiState {
                    explorer_state: &mut explorer_state,
                    overlays: controller.overlays_mut(),
                    index_available: &mut annotation_file_index_available,
                    loaded_windows: &mut annotation_file_loaded_windows,
                    messages: &mut messages,
                },
            );
        }
        annotation_loads.dispatch_pending();
        if annotation_loads.is_busy() != annotation_loads_were_busy {
            annotation_loads_were_busy = annotation_loads.is_busy();
            needs_redraw = true;
        }

        let current_search_view_identity = RegionSearchViewIdentity::new(
            &current_collection_name,
            explorer_state.selected_block_group_id,
            current_block_group.as_ref(),
            history_ref,
            &explorer.data.annotation_files,
            &explorer_state.active_annotation_files,
        );
        if search_view_identity.as_ref() != Some(&current_search_view_identity) {
            search_view_identity = Some(current_search_view_identity);
            search_state.clear_search_state();
            search_error = None;
            pending_search_task = None;
            if search_state.focused {
                search_state.schedule_query(Instant::now());
            }
        }

        loop {
            let response = match region_search_worker.try_receive() {
                RegionSearchWorkerPoll::Response(response) => *response,
                RegionSearchWorkerPoll::Empty => break,
                RegionSearchWorkerPoll::Disconnected => {
                    search_task_in_flight = false;
                    pending_search_task = None;
                    if search_state.focused {
                        search_state.selection_pending = false;
                        search_state.search_due = None;
                        search_state.clear_cached_results();
                        search_error = Some("region search worker stopped".to_string());
                        needs_redraw = true;
                    }
                    break;
                }
            };
            search_task_in_flight = false;
            if is_current_search_generation(&search_state, response.generation) {
                needs_redraw = true;
                match response.result {
                    RegionSearchWorkerResult::Matches(result) => {
                        search_state.complete_query(result);
                        search_error = search_state.completed_feedback.clone();
                    }
                    RegionSearchWorkerResult::Selection(Ok(search_match)) => {
                        search_state.selection_pending = false;
                        match activate_search_match(
                            &mut controller,
                            search_match.as_ref(),
                            conn,
                            workspace,
                        ) {
                            Ok(()) => {
                                search_state.blur();
                                pending_search_task = None;
                                search_error = None;
                                focus_zone = FocusZone::Canvas;
                            }
                            Err(error) => search_error = Some(error),
                        }
                    }
                    RegionSearchWorkerResult::Selection(Err(error)) => {
                        search_state.selection_pending = false;
                        search_error = Some(error);
                    }
                }
            }
            if let Some(task) = pending_search_task.take()
                && search_state.focused
                && task.generation() == search_state.generation
            {
                match region_search_worker.submit(task) {
                    Ok(()) => search_task_in_flight = true,
                    Err(error) => search_error = Some(error),
                }
            }
        }

        if let Some((generation, query)) = search_state.due_query(Instant::now()) {
            needs_redraw = true;
            if let Some(context) = current_region_search_context(
                workspace,
                &current_collection_name,
                current_block_group.as_ref().filter(|_| !is_loading),
                history_ref,
                &explorer.data.annotation_files,
                &explorer_state.active_annotation_files,
            ) {
                let task = RegionSearchTask::query(generation, context, query);
                if let Err(error) = queue_region_search_task(
                    &region_search_worker,
                    &mut search_task_in_flight,
                    &mut pending_search_task,
                    task,
                ) {
                    search_error = Some(error);
                }
            } else {
                search_error = Some(if is_loading {
                    "loading selected block group".to_string()
                } else {
                    "select a block group first".to_string()
                });
            }
        }

        // A teleport or jump handled above may already have grown the graph, or moved it into
        // another batch whose annotation groups are loaded here.
        let pre_draw_sync = controller.sync_active_world();
        if pre_draw_sync.changed {
            needs_redraw = true;
        }
        apply_group_reload(pre_draw_sync, &mut explorer_state, &mut messages);

        // Nothing that reaches the screen changed (e.g. only the mouse moved): wait for the
        // next event instead of redrawing an identical frame.
        if !needs_redraw {
            let _ = crossterm::event::poll(background_work_wait(
                &annotation_loads,
                &search_state,
                search_task_in_flight || pending_search_task.is_some(),
            ));
            continue;
        }
        needs_redraw = false;

        // Draw the UI
        let completed_frame = terminal.draw(|frame| {
            let status_bar_height: u16 = 1;

            // The outer layout is a vertical split between the main area, optional message bar, and status bar
            let show_message_bar = !messages.is_empty();

            let mut outer_constraints = vec![Constraint::Min(1)];
            if show_message_bar {
                outer_constraints.push(Constraint::Length(1)); // Message bar
            }
            outer_constraints.push(Constraint::Length(status_bar_height));

            let outer_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(outer_constraints)
                .split(frame.area());

            let status_bar_area = *outer_layout.last().unwrap();
            let message_bar_area = if show_message_bar {
                Some(outer_layout[outer_layout.len() - 2])
            } else {
                None
            };

            // The sidebar is a horizontal split of the area above the status bar (and message bar)
            let sidebar_layout = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(vec![Constraint::Percentage(20), Constraint::Percentage(80)])
                .split(outer_layout[0]);
            let sidebar_area = sidebar_layout[0];
            last_sidebar_area = sidebar_area;
            let viewer_root_area = sidebar_layout[1];

            let search_is_pending = search_state.focused
                && !search_state.query.trim().is_empty()
                && (search_state.search_due.is_some()
                    || search_state.selection_pending
                    || search_task_in_flight
                    || pending_search_task.is_some());
            let visible_search_matches = if search_state.focused {
                search_state.matches.len().min(5)
            } else {
                0
            };
            let search_dropdown_rows = if visible_search_matches > 0 {
                visible_search_matches as u16 + 2
            } else if search_state.focused && (search_error.is_some() || search_is_pending) {
                3
            } else {
                0
            };
            let search_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Length(3),
                    Constraint::Length(search_dropdown_rows),
                    Constraint::Min(1),
                ])
                .split(viewer_root_area);
            let search_input_area = search_layout[0];
            let search_dropdown_area = search_layout[1];
            last_search_area = search_input_area;
            last_search_dropdown_area = if search_state.focused {
                search_dropdown_area
            } else {
                Rect::default()
            };
            let viewer_content_area = search_layout[2];

            // The panel pops up in the graph area, it does not overlap with the sidebar
            let panel_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Percentage(80), Constraint::Percentage(20)])
                .split(viewer_content_area);
            let panel_area = panel_layout[1];

            let canvas_area = if show_panel {
                panel_layout[0]
            } else {
                viewer_content_area
            };

            // Sidebar
            explorer_state.has_focus = focus_zone == FocusZone::Sidebar;
            if show_sidebar {
                let theme = current_theme();
                let sidebar_block = Block::default().padding(Padding::new(0, 0, 1, 1)).style(
                    Style::default().bg(theme[0x01]).fg(theme[0x05]),
                );
                let sidebar_content_area = sidebar_block.inner(sidebar_area);

                frame.render_widget(sidebar_block.clone(), sidebar_area);
                frame.render_stateful_widget(&explorer, sidebar_content_area, &mut explorer_state);

                // Draw the vertical separator line at the right edge of the sidebar
                let line_char = "▕";
                let line_style = Style::default().fg(theme[0x02]);
                let x = sidebar_area.right() - 1;
                for y in sidebar_area.top()..sidebar_area.bottom() {
                    frame.buffer_mut().set_string(x, y, line_char, line_style);
                }
            }

            let search_border_style = if search_state.focused {
                Style::default()
                    .fg(current_theme()[0x07])
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(current_theme()[0x05])
            };
            let search_value = if search_state.query.is_empty() {
                Span::styled(
                    "type a region such as chr1:20-30 (1-based)",
                    Style::default()
                        .fg(current_theme()[0x04])
                        .add_modifier(Modifier::DIM),
                )
            } else {
                Span::styled(
                    search_state.query.clone(),
                    Style::default().fg(current_theme()[0x05]),
                )
            };
            let search_input = Paragraph::new(Line::from(vec![Span::raw(" "), search_value]))
                .block(
                    Block::bordered()
                        .title("Region search (g) ")
                        .border_style(search_border_style),
                );
            frame.render_widget(search_input, search_input_area);

            if search_state.focused && search_dropdown_rows > 0 {
                let dropdown_items = if search_state.matches.is_empty() {
                    let feedback = search_error.as_deref().unwrap_or(if search_state
                        .selection_pending
                    {
                        "Resolving selected annotation…"
                    } else {
                        "Searching…"
                    });
                    vec![ListItem::new(Line::from(Span::styled(
                        feedback,
                        Style::default().fg(current_theme()[0x04]),
                    )))]
                } else {
                    let window_start = search_match_window_start(
                        search_state.selected_match,
                        search_state.matches.len(),
                    );
                    search_state
                        .matches
                        .iter()
                        .skip(window_start)
                        .take(visible_search_matches)
                        .enumerate()
                        .map(|(index, search_match)| {
                            let style = if search_state.selected_match
                                == Some(index + window_start)
                            {
                                Style::default()
                                    .fg(current_theme()[0x00])
                                    .bg(current_theme()[0x07])
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(current_theme()[0x05])
                            };
                            ListItem::new(Line::from(Span::styled(
                                format!(" {}", search_match.label),
                                style,
                            )))
                        })
                        .collect()
                };
                let dropdown = List::new(dropdown_items).block(
                    Block::bordered()
                        .title("Matches (↑↓, Enter)")
                        .border_style(search_border_style),
                );
                frame.render_widget(dropdown, search_dropdown_area);
            }

            // Render message bar if there are messages
            if let Some(area) = message_bar_area
                && let Some(msg) = messages.latest()
            {
                let message_text = Text::from(msg.as_str());
                let message_bar = Paragraph::new(message_text)
                    .style(Style::default().fg(current_theme()[0x09]).bg(current_theme()[0x00]));
                frame.render_widget(message_bar, area);
            }

            // Status bar
            let mut status_message = if search_state.focused {
                "type region | *↑↓* choose match | *enter* go | *ctrl-u* clear | *esc* close"
                    .to_string()
            } else {
                match focus_zone {
                    FocusZone::Canvas => {
                        let tab_dest = if show_panel { "to panel" } else { "to sidebar" };
                        if !controller.view_state().is_cursor_visible() {
                            format!("*drag* pan | *click* select | *↑↓←→* show cursor | *g* search | *tab* {tab_dest}")
                        } else {
                            format!("*←↓↑→/hjkl* move | *wbe* annotations | *enter* details | *+/-* zoom | *p* path | *m* messages | *g* search | *tab* {tab_dest}")
                        }
                    }
                    FocusZone::Panel => match panel_mode {
                        PanelMode::Messages => "*c* clear | *esc* close | *tab* to sidebar".to_string(),
                        PanelMode::Details => "*esc* close | *tab* to sidebar".to_string(),
                    },
                    FocusZone::Sidebar => CollectionExplorer::get_status_line(),
                }
            };
            if let Some(error) = search_error.as_deref() {
                status_message = format!("search: {error}");
            }
            if let Some(status) = annotation_loads.status() {
                let loading_message = if status.waiting {
                    format!("Waiting to load annotation {}", status.display_name)
                } else {
                    format!("Loading annotation {}", status.display_name)
                };
                status_message = format!("{loading_message} | {status_message}");
            }
            if !search_state.focused && focus_zone != FocusZone::Canvas {
                status_message.push_str(" | *g* search");
            }
            status_message.push_str(" | *q* quit"); // Universal controls
            render_status_bar(frame, status_bar_area, &status_message);

            // Canvas area
            if is_loading {
                let loading_text = Text::styled(
                    "Loading…",
                    Style::default()
                        .fg(current_theme()[0x05])
                        .add_modifier(Modifier::BOLD),
                );
                let loading_para =
                    Paragraph::new(loading_text).alignment(HorizontalAlignment::Center);

                // Center the loading message vertically in the canvas area
                let loading_area = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(45),
                        Constraint::Length(1),
                        Constraint::Percentage(45),
                    ])
                    .split(canvas_area)[1];

                render_with_optional_clear(frame, canvas_area, loading_area, true, loading_para);
            } else if explorer_state.selected_block_group_id.is_none() {
                // Render splash screen
                let splashscreen_lines = [
                    " ██████╗ ███████╗███╗   ██╗",
                    "██╔════╝ ██╔════╝████╗  ██║",
                    "██║  ███╗█████╗  ██╔██╗ ██║",
                    "██║   ██║██╔══╝  ██║╚██╗██║",
                    "╚██████╔╝███████╗██║ ╚████║",
                    " ╚═════╝ ╚══════╝╚═╝  ╚═══╝",
                ];

                let splash_text = Text::from(
                    splashscreen_lines
                        .iter()
                        .map(|&l| {
                            Line::from(Span::styled(
                                l,
                                Style::default().fg(current_theme()[0x07]),
                            ))
                        })
                        .collect::<Vec<_>>(),
                );

                let splash_para =
                    Paragraph::new(splash_text).alignment(HorizontalAlignment::Center);

                // Center the splash screen vertically in the canvas area
                let splash_area = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(40),
                        Constraint::Length(splashscreen_lines.len() as u16),
                        Constraint::Percentage(40),
                    ])
                    .split(canvas_area)[1];

                render_with_optional_clear(frame, canvas_area, splash_area, true, splash_para);
            } else {
                let canvas_style = Style::default().bg(current_theme()[0x00]);

                let main_canvas_area = canvas_area;

                // At full detail annotations are drawn as flags under their nodes, so only the
                // names that found no room there still float.
                controller.render(
                    frame,
                    main_canvas_area,
                    canvas_style,
                );
            }

            // Panel
            if show_panel {
                let panel_title = match panel_mode {
                    PanelMode::Details => "Details",
                    PanelMode::Messages => "Messages",
                };
                let panel_block = Block::bordered()
                    .padding(Padding::new(2, 2, 1, 1))
                    .title(panel_title)
                    .style(Style::default().bg(current_theme()[0x01]).fg(current_theme()[0x05]))
                    .border_style(if focus_zone == FocusZone::Panel {
                        Style::default()
                            .fg(current_theme()[0x07])
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(current_theme()[0x05])
                    });

                let panel_text = match panel_mode {
                    PanelMode::Details => {
                        let mut lines = vec![];

                        let cursor = controller.view_state().cursor;
                        if let Some(graph_node) = cursor.node {
                            let node_id_short =
                                graph_node.node_id.to_string().chars().take(12).collect::<String>();
                            let (frac_x, _) = cursor.fractional;
                            let block_width = graph_node.sequence_end - graph_node.sequence_start;
                            let pos_on_node = graph_node.sequence_start
                                + (frac_x * block_width as f64).round() as i64;
                            let block_spec = format!(
                                "{}:{}-{} (cursor at {})",
                                node_id_short,
                                graph_node.sequence_start,
                                graph_node.sequence_end,
                                pos_on_node
                            );
                            lines.push(Line::from(vec![
                                Span::styled(
                                    "Block: ",
                                    Style::default().add_modifier(Modifier::BOLD),
                                ),
                                Span::raw(block_spec),
                            ]));
                        } else {
                            lines.push(Line::from(Span::styled(
                                "No node selected",
                                Style::default()
                                    .fg(current_theme()[0x04])
                                    .add_modifier(Modifier::ITALIC),
                            )));
                        }

                        lines
                    }
                    PanelMode::Messages => {
                        if messages.is_empty() {
                            vec![Line::from(vec![Span::styled(
                                "No messages",
                                Style::default().fg(current_theme()[0x04]),
                            )])]
                        } else {
                            messages
                                .iter()
                                .enumerate()
                                .map(|(idx, message)| {
                                    Line::from(vec![Span::raw(format!(
                                        "{:>2}. {message}",
                                        idx + 1
                                    ))])
                                })
                                .collect()
                        }
                    }
                };

                let panel_content = Paragraph::new(panel_text)
                    .wrap(Wrap { trim: true })
                    .alignment(HorizontalAlignment::Left)
                    .block(panel_block);

                render_with_optional_clear(
                    frame,
                    panel_area,
                    panel_area,
                    tui_layout_change,
                    panel_content,
                );

                // Reset the layout change flag
                tui_layout_change = false;
            }
        })?;

        // Edges drawn through the Kitty graphics protocol go out after the cells, as one image.
        if curved_edges_available {
            let mut escapes = Vec::new();
            if controller.layout_mode().is_curved() {
                kitty_overlay.update(
                    &mut escapes,
                    completed_frame.buffer,
                    controller.overlay_primitives(),
                )?;
            } else {
                kitty_overlay.clear(&mut escapes)?;
            }
            terminal.backend_mut().write_all(&escapes)?;
        }

        // The render just above may have claimed a new world, growing the graph and needing
        // that batch's annotation groups; if anything drawn changed, the frame is redrawn
        // below.
        let post_draw_sync = controller.sync_active_world();
        let changed_after_draw = post_draw_sync.changed;
        apply_group_reload(post_draw_sync, &mut explorer_state, &mut messages);

        // Rendering claims the initial graph neighborhood. File projection needs those nodes,
        // so load defaults only after that first draw (also after changing block groups).
        if !is_loading
            && controller.engine().active_world().is_some()
            && let Some(block_group) = current_block_group.as_ref()
        {
            let pending_entries: Vec<AnnotationFileEntry> = explorer
                .data
                .annotation_files
                .iter()
                .filter(|entry| {
                    let id = entry.file_addition.id;
                    pending_annotation_file_loads.contains(&id)
                })
                .cloned()
                .collect();
            pending_annotation_file_loads.clear();
            if !pending_entries.is_empty() {
                let node_filter = active_neighborhood_node_ids(controller.engine());
                let context = AnnotationFileActivationContext {
                    collection_name: &current_collection_name,
                    history_ref,
                    block_group,
                    node_filter: &node_filter,
                    query_window: active_neighborhood_coordinate_window(controller.engine())
                        .map(expand_query_window),
                };
                auto_activate_annotation_files_for_viewport(
                    &context,
                    &pending_entries,
                    &mut explorer_state,
                    &annotation_file_index_available,
                    &mut annotation_loads,
                );
                annotation_loads.dispatch_pending();
                needs_redraw = true;
            }
        }

        // Update the graph controller if a new block group was selected.
        // This runs after terminal.draw() so the loading indicator is visible
        // for the full duration of the blocking DB work.
        if is_loading && let Some(ref new_block_group_id) = explorer_state.selected_block_group_id {
            // Replace the graph, its view and its overlays with the selected block group's.
            controller.open_block_group(new_block_group_id)?;
            current_block_group = controller.block_group().cloned();
            let selected_sample = current_block_group
                .as_ref()
                .map(|bg| bg.sample_name.as_str());
            if explorer.refresh(
                conn,
                config_conn,
                selected_sample,
                current_block_group.as_ref(),
                &current_collection_name,
                history_ref,
            ) {
                explorer.force_reload(&mut explorer_state);
                explorer_state.retain_annotation_groups(&explorer.data.annotation_groups);
            }
            annotation_file_index_available.clear();
            annotation_file_loaded_windows.clear();
            explorer_state.active_annotation_groups.clear();
            explorer_state.active_annotation_files.clear();
            pending_annotation_file_loads = annotation_file_ids(&explorer.data.annotation_files);

            is_loading = false;
            search_state.focused = false;
            search_state.clear_search_state();
            pending_search_task = None;
            search_error = None;
            needs_redraw = true;
            continue;
        }

        // The overlays, dimming, or loaded graph changed after the frame was rendered. Draw
        // them immediately instead of waiting for another input event.
        if changed_after_draw || needs_redraw {
            needs_redraw = true;
            continue;
        }

        // The door's world is fully drawn; drop what was typed or clicked while it loaded.
        if entered_door {
            entered_door = false;
            needs_redraw = discard_pending_input()?;
            if needs_redraw {
                continue;
            }
        }

        // Rendering is event-driven (no animation to advance): sleep until the next input event
        // or until background annotation loads and region searches need another look.
        let _ = crossterm::event::poll(background_work_wait(
            &annotation_loads,
            &search_state,
            search_task_in_flight || pending_search_task.is_some(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use gen_core::{HashId, Workspace};
    use gen_graph::{GenGraph, GraphNode, PackedGenGraph};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        db::get_connection,
        path::Path,
    };
    use gen_tui::{graph_view::GraphViewState, layout_engine::LayoutEngine};
    use ratatui::{Terminal, backend::TestBackend, style::Style};

    use super::{
        REGION_SEARCH_DEBOUNCE, RegionSearchInputAction, RegionSearchState, focus_region_search,
        is_current_search_generation, is_region_search_command, starting_controller,
        teleport_through_wormhole,
    };
    use crate::views::{
        gen_graph_controller::GenGraphController,
        graph_database::GraphDatabase,
        graph_overlay::has_path_overlay,
        lazy_graph_source::tests::setup_labelled_chain_block_group,
        region_search::{resolve_region_search_matches, search_request_fixture},
    };

    #[test]
    fn test_region_search_state_handles_dropdown_selection_without_default() {
        assert!(is_region_search_command(KeyCode::Char('g')));
        assert!(!is_region_search_command(KeyCode::Char('/')));
        assert!(!is_region_search_command(KeyCode::Tab));

        let request = search_request_fixture();
        let match_template = resolve_region_search_matches(&request.request(), "duplicate-gene")
            .expect("should load a search match for state testing")
            .into_iter()
            .next()
            .expect("fixture should provide a search match");
        let matches = vec![match_template.clone(), match_template];

        let mut state = RegionSearchState {
            focused: true,
            ..RegionSearchState::default()
        };
        state.complete_query(Ok(matches.clone()));
        assert_eq!(state.selected_match, None);
        let now = Instant::now();
        assert!(matches!(
            state.handle_key_at(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), now),
            RegionSearchInputAction::Ignored
        ));
        state.move_selection(1);
        assert_eq!(state.selected_match, Some(0));
        state.move_selection(1);
        assert_eq!(state.selected_match, Some(1));
        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
                now + Duration::from_millis(10),
            ),
            RegionSearchInputAction::Changed
        ));
        assert_eq!(state.selected_match, None);
        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                now + Duration::from_millis(10),
            ),
            RegionSearchInputAction::Ignored
        ));
        state.complete_query(Ok(matches));
        state.move_selection(1);
        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
                now + Duration::from_millis(10),
            ),
            RegionSearchInputAction::Selected(_)
        ));
    }

    #[test]
    fn test_region_search_clear_action_empties_query_and_matches() {
        let request = search_request_fixture();
        let mut state = RegionSearchState {
            query: "chr1:5-10".to_string(),
            focused: true,
            ..RegionSearchState::default()
        };
        state.complete_query(Ok(resolve_region_search_matches(
            &request.request(),
            "chr1:5-10",
        )
        .expect("should resolve the clear-action fixture query")));
        state.selected_match = Some(0);

        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
                Instant::now(),
            ),
            RegionSearchInputAction::Cleared
        ));
        assert!(state.query.is_empty());
        assert!(state.matches.is_empty());
        assert_eq!(state.completed_query, None);
        assert_eq!(state.completed_feedback, None);
        assert_eq!(state.selected_match, None);
        assert!(state.focused);

        state.query = "x".to_string();
        state.complete_query(Ok(resolve_region_search_matches(
            &request.request(),
            "chr1:5-10",
        )
        .expect("should resolve the query before backspace clears it")));
        state.selected_match = Some(0);
        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
                Instant::now(),
            ),
            RegionSearchInputAction::Cleared
        ));
        assert!(state.query.is_empty());
        assert!(state.matches.is_empty());
        assert_eq!(state.completed_query, None);
    }

    #[test]
    fn test_region_search_refocus_reuses_matches_and_selected_index() {
        let request = search_request_fixture();
        let mut state = RegionSearchState {
            query: "chr1:5-10".to_string(),
            focused: true,
            ..RegionSearchState::default()
        };
        state.complete_query(Ok(resolve_region_search_matches(
            &request.request(),
            &state.query,
        )
        .expect("should resolve the preserved query")));
        let matches = state.matches.clone();
        state.selected_match = Some(0);
        let completed_generation = state.generation;
        state.blur();
        let blurred_generation = state.generation;
        assert!(!is_current_search_generation(&state, completed_generation));

        let now = Instant::now();
        focus_region_search(&mut state, now);
        assert_eq!(state.query, "chr1:5-10");
        assert_eq!(state.matches.len(), matches.len());
        assert_eq!(state.selected_match, Some(0));
        assert!(state.search_due.is_none());
        assert_eq!(state.generation, blurred_generation);
        assert!(state.due_query(now + REGION_SEARCH_DEBOUNCE).is_none());
    }

    #[test]
    fn test_region_search_refocus_reuses_completed_not_found_feedback() {
        let mut state = RegionSearchState {
            query: "missing-region".to_string(),
            focused: true,
            ..RegionSearchState::default()
        };
        state.complete_query(Err("no region matched missing-region".to_string()));
        state.blur();

        let now = Instant::now();
        focus_region_search(&mut state, now);
        assert!(state.matches.is_empty());
        assert_eq!(
            state.completed_feedback.as_deref(),
            Some("no region matched missing-region")
        );
        assert!(state.search_due.is_none());
        assert!(state.due_query(now + REGION_SEARCH_DEBOUNCE).is_none());
    }

    #[test]
    fn test_region_search_refocus_retries_query_cancelled_before_completion() {
        let now = Instant::now();
        let mut state = RegionSearchState {
            focused: true,
            ..RegionSearchState::default()
        };
        state.handle_key_at(KeyEvent::new(KeyCode::Char('U'), KeyModifiers::NONE), now);
        let cancelled_generation = state.generation;
        state.blur();
        assert!(!is_current_search_generation(&state, cancelled_generation));
        assert!(state.search_due.is_none());

        focus_region_search(&mut state, now + Duration::from_millis(50));
        assert_eq!(state.due_query(now + Duration::from_millis(199)), None);
        assert_eq!(
            state.due_query(now + Duration::from_millis(200)),
            Some((state.generation, "U".to_string()))
        );
    }

    #[test]
    fn test_region_search_debounce_only_sends_the_final_query() {
        let now = Instant::now();
        let mut state = RegionSearchState {
            focused: true,
            ..RegionSearchState::default()
        };
        assert!(matches!(
            state.handle_key_at(KeyEvent::new(KeyCode::Char('U'), KeyModifiers::NONE), now,),
            RegionSearchInputAction::Changed
        ));
        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE),
                now + Duration::from_millis(100),
            ),
            RegionSearchInputAction::Changed
        ));
        assert!(state.due_query(now + Duration::from_millis(249)).is_none());
        assert_eq!(
            state.due_query(now + Duration::from_millis(250)),
            Some((state.generation, "U2".to_string()))
        );
    }

    #[test]
    fn test_region_search_clear_and_escape_cancel_debounced_lookup() {
        let now = Instant::now();
        let mut state = RegionSearchState {
            focused: true,
            ..RegionSearchState::default()
        };
        state.handle_key_at(KeyEvent::new(KeyCode::Char('U'), KeyModifiers::NONE), now);
        let cancelled_generation = state.generation;
        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
                now + Duration::from_millis(50),
            ),
            RegionSearchInputAction::Cleared
        ));
        assert!(state.due_query(now + Duration::from_secs(1)).is_none());
        assert!(state.generation > cancelled_generation);

        state.handle_key_at(
            KeyEvent::new(KeyCode::Char('U'), KeyModifiers::NONE),
            now + Duration::from_secs(2),
        );
        assert!(matches!(
            state.handle_key_at(
                KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
                now + Duration::from_secs(2) + Duration::from_millis(50),
            ),
            RegionSearchInputAction::Closed
        ));
        assert!(state.due_query(now + Duration::from_secs(3)).is_none());
    }

    #[test]
    fn test_region_search_discards_results_from_a_stale_generation() {
        let now = Instant::now();
        let mut state = RegionSearchState {
            focused: true,
            ..RegionSearchState::default()
        };
        state.handle_key_at(KeyEvent::new(KeyCode::Char('U'), KeyModifiers::NONE), now);
        let old_generation = state.generation;
        assert!(is_current_search_generation(&state, old_generation));

        state.handle_key_at(
            KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE),
            now + Duration::from_millis(20),
        );
        assert!(!is_current_search_generation(&state, old_generation));
    }

    fn three_node_chain() -> (LayoutEngine<PackedGenGraph>, [GraphNode; 3]) {
        let nodes = ["left", "middle", "right"].map(|label| GraphNode {
            node_id: HashId::convert_str(label),
            sequence_start: 0,
            sequence_end: 5,
        });
        let mut graph = GenGraph::new();
        graph.add_edge(nodes[0], nodes[1], Vec::new());
        graph.add_edge(nodes[1], nodes[2], Vec::new());
        (LayoutEngine::new(PackedGenGraph::from(graph)), nodes)
    }

    #[test]
    fn test_wormhole_entry_frames_target_without_highlighting_it() {
        let (mut predecessor_engine, predecessor_nodes) = three_node_chain();
        let mut predecessor_state = GraphViewState::default();
        teleport_through_wormhole(
            &mut predecessor_engine,
            &mut predecessor_state,
            predecessor_nodes[1],
            predecessor_nodes[0],
        );

        assert_eq!(predecessor_state.cursor.node, Some(predecessor_nodes[0]));
        assert_eq!(predecessor_state.cursor.fractional, (1.0, 0.5));
        assert!(predecessor_state.highlights.styles.is_empty());

        let (mut successor_engine, successor_nodes) = three_node_chain();
        successor_engine
            .activate_batch_containing(successor_nodes[2], 10)
            .expect("should preload the successor's world");
        let mut successor_state = GraphViewState::default();
        teleport_through_wormhole(
            &mut successor_engine,
            &mut successor_state,
            successor_nodes[1],
            successor_nodes[2],
        );

        assert_eq!(successor_state.cursor.node, Some(successor_nodes[2]));
        assert_eq!(successor_state.cursor.fractional, (0.0, 0.5));
        assert!(successor_state.highlights.styles.is_empty());
    }

    /// Draw `controller` once into a narrow terminal, so it crawls its first batch.
    fn draw_first_batch(controller: &mut GenGraphController) {
        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        controller.sync_active_world();
        terminal
            .draw(|frame| {
                controller.render(frame, frame.area(), Style::default());
            })
            .expect("should draw the first batch");
        controller.sync_active_world();
    }

    #[test]
    fn test_starting_controller_keeps_a_handed_over_controller_without_reloading() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels: Vec<String> = (0..80).map(|index| format!("n{index}")).collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, edge_ids) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let block_group = BlockGroup::get_by_id(&conn, &block_group_id, None).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut inline = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        inline.view_state_mut().show_cursor();
        inline.add_path(&path);
        assert!(inline.toggle_path());
        draw_first_batch(&mut inline);
        let anchor = inline
            .engine()
            .active_world()
            .expect("should have an active world after drawing")
            .anchor();
        let anchor_batch = inline.engine().batch_of(anchor);
        let active_batch = inline.engine().active_batch();
        let node_count = inline.engine().graph().node_count();
        let cursor = inline.view_state().cursor;
        assert!(node_count > 2 && node_count < labels.len());
        assert!(cursor.node.is_some());

        let full = starting_controller(
            &conn,
            &workspace,
            None,
            Some(Box::new(inline)),
            Some(&block_group),
        )
        .expect("should start the full viewer");

        assert_eq!(full.engine().batch_of(anchor), anchor_batch);
        assert_eq!(full.engine().active_batch(), active_batch);
        assert_eq!(full.engine().graph().node_count(), node_count);
        assert_eq!(full.view_state().cursor.node, cursor.node);
        assert_eq!(full.view_state().cursor.fractional, cursor.fractional);
        assert!(has_path_overlay(full.overlays()));
    }

    #[test]
    fn test_starting_controller_opens_another_selected_block_group() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels: Vec<String> = (0..80).map(|index| format!("n{index}")).collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let other = BlockGroup::create(
            &conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "chr2",
                ..Default::default()
            },
        )
        .unwrap();
        let workspace = Workspace::from_current_dir();
        let mut inline = GenGraphController::for_block_group(
            GraphDatabase::for_connection(&conn, &workspace)
                .expect("should open the graph database"),
            &block_group_id,
            None,
        )
        .expect("should load the block group");
        draw_first_batch(&mut inline);
        assert!(inline.engine().graph().node_count() > 2);

        let full = starting_controller(
            &conn,
            &workspace,
            None,
            Some(Box::new(inline)),
            Some(&other),
        )
        .expect("should start the full viewer");

        assert_eq!(full.block_group().map(|open| open.id), Some(other.id));
        assert!(full.engine().graph().node_count() <= 2);
        assert_eq!(full.engine().active_batch(), None);
        assert!(full.overlays().is_empty());
    }
}
