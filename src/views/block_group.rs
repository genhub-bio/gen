use std::{
    collections::{HashMap, HashSet},
    error::Error,
    path::PathBuf,
    time::{Duration, Instant},
};

use crossterm::event::{
    self, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use gen_core::{HashId, Workspace, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection};
use gen_tui::{
    crawl::{EagerSource, GraphSource},
    graph_view::GraphViewState,
    layout_engine::LayoutEngine,
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
        annotations::{
            AnnotationFileTrackRequest, AnnotationGroupTrackRequest, load_annotation_file_track,
            load_annotations_for_group,
        },
        collection::{CollectionExplorer, CollectionExplorerState, FocusZone},
        gen_graph_controller::{AnnotationDisplay, GenGraphController, GraphKeyOutcome, WorldSync},
        gen_graph_widget::starting_zoom_level,
        graph_overlay::{
            GraphOverlay, OverlaySource, file_track_key, group_track_key, remove_track_overlays,
            replace_track_overlays,
        },
        lazy_graph_source::{
            EagerOrSqlSource, SqlGraphSource, probe_block_group_start, seed_block_group_graph,
        },
        panels::{render_status_bar, render_with_optional_clear},
        region_search::{
            RegionSearchMatch, RegionSearchRequest, activate_search_match, remove_search_overlay,
            resolve_region_search_matches,
        },
        tui_runtime::TuiSession,
    },
};

#[derive(Debug, Default)]
struct RegionSearchState {
    query: String,
    matches: Vec<RegionSearchMatch>,
    selected_match: Option<usize>,
    focused: bool,
}

impl RegionSearchState {
    fn clear_matches(&mut self) {
        self.matches.clear();
        self.selected_match = None;
    }

    fn set_matches(&mut self, matches: Vec<RegionSearchMatch>) {
        self.matches = matches;
        self.selected_match = self.selected_match.and_then(|selected_match| {
            (selected_match < self.matches.len()).then_some(selected_match)
        });
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

    fn handle_key(&mut self, key: KeyEvent) -> RegionSearchInputAction {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('u') {
            self.query.clear();
            self.clear_matches();
            return RegionSearchInputAction::Cleared;
        }
        match key.code {
            KeyCode::Char(character) => {
                self.query.push(character);
                self.selected_match = None;
                RegionSearchInputAction::Changed
            }
            KeyCode::Backspace => {
                if self.query.pop().is_some() {
                    self.selected_match = None;
                    if self.query.is_empty() {
                        self.clear_matches();
                        RegionSearchInputAction::Cleared
                    } else {
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
            KeyCode::Enter => self
                .selected_match
                .and_then(|selected_match| self.matches.get(selected_match))
                .cloned()
                .map_or(RegionSearchInputAction::Ignored, |region_match| {
                    RegionSearchInputAction::Selected(Box::new(region_match))
                }),
            KeyCode::Esc => RegionSearchInputAction::Closed,
            _ => RegionSearchInputAction::Ignored,
        }
    }
}

fn is_region_search_command(key_code: KeyCode) -> bool {
    key_code == KeyCode::Char('g')
}

fn focus_region_search(state: &mut RegionSearchState) {
    state.focused = true;
    state.clear_matches();
}

#[derive(Clone, Debug)]
enum RegionSearchInputAction {
    Changed,
    Cleared,
    Selected(Box<RegionSearchMatch>),
    Closed,
    Ignored,
}

fn refresh_region_search(
    state: &mut RegionSearchState,
    search_error: &mut Option<String>,
    request: Option<&RegionSearchRequest<'_>>,
) {
    state.selected_match = None;
    *search_error = None;
    if state.query.trim().is_empty() {
        state.clear_matches();
        return;
    }
    let Some(request) = request else {
        state.clear_matches();
        *search_error = Some("select a block group first".to_string());
        return;
    };
    match resolve_region_search_matches(request, state.query.trim()) {
        Ok(matches) => state.set_matches(matches),
        Err(error) => {
            state.clear_matches();
            *search_error = Some(error);
        }
    }
}

fn search_match_window_start(selected_match: Option<usize>, match_count: usize) -> usize {
    selected_match
        .unwrap_or(0)
        .saturating_sub(4)
        .min(match_count.saturating_sub(5))
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
    pub source: EagerOrSqlSource,
    /// The zoom level the viewer opens at (see `starting_zoom_level`).
    pub zoom_index: usize,
}

/// Load `block_group_id`'s graph, the source its `LayoutEngine` should crawl through, and the
/// zoom level to open it at.
///
/// A historical view (`history_ref: Some(_)`) always eager-loads the full graph up front:
/// the port queries the lazy path below is built on can only answer for the live graph.
/// Otherwise, the graph is seeded with just its `PATH_START` sentinel and grown lazily from
/// SQLite as the viewer's crawl pushes past its frontier - see `SqlGraphSource`, which is what
/// turns opening a large block group from a full-graph-materializing stall into an
/// near-instant open. The seed can't tell how many nodes there are, so the zoom level comes
/// from a small probe crawl of the start instead.
pub(crate) fn load_block_group_graph(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: &HashId,
    history_ref: Option<&str>,
) -> Result<BlockGroupGraph, Box<dyn Error>> {
    if history_ref.is_some() {
        let graph = BlockGroup::get_graph(conn, workspace, block_group_id, history_ref)?;
        return Ok(BlockGroupGraph {
            zoom_index: starting_zoom_level(&graph),
            graph,
            source: EagerOrSqlSource::Eager(EagerSource),
        });
    }
    let db_path = conn
        .path()
        .map(PathBuf::from)
        .ok_or("graph database has no file path")?;
    Ok(BlockGroupGraph {
        graph: seed_block_group_graph(conn, block_group_id),
        source: EagerOrSqlSource::Sql(Box::new(SqlGraphSource::new(db_path, *block_group_id))),
        zoom_index: starting_zoom_level(&probe_block_group_start(conn, block_group_id, false)),
    })
}

/// Node IDs in the currently active crawled neighborhood (excluding terminal start/end
/// nodes) - the same deliberately-constrained local window `LayoutEngine` already
/// subsets the graph to, so annotation loading has no need to subset any further by
/// what happens to be on-screen right now.
pub(crate) fn active_neighborhood_node_ids<S: GraphSource<GenGraph>>(
    engine: &LayoutEngine<GenGraph, S>,
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
pub(crate) fn active_neighborhood_coordinate_window<S: GraphSource<GenGraph>>(
    engine: &LayoutEngine<GenGraph, S>,
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
struct AnnotationToggleContext<'a, S: GraphSource<GenGraph>> {
    conn: &'a GraphConnection,
    history_ref: Option<&'a str>,
    workspace: &'a Workspace,
    collection_name: &'a str,
    current_block_group: Option<&'a BlockGroup>,
    graph_engine: &'a LayoutEngine<GenGraph, S>,
    explorer: &'a CollectionExplorer,
}

/// Service a pending annotation file or group toggle request left on `explorer_state` by
/// the sidebar's input/mouse handling. Shared by the keyboard and mouse event branches in
/// `view_block_group`'s event loop, which both route sidebar interaction through the same
/// `CollectionExplorerState` toggle-request fields.
fn handle_annotation_toggle_requests<S: GraphSource<GenGraph>>(
    ctx: &AnnotationToggleContext<S>,
    explorer_state: &mut CollectionExplorerState,
    overlays: &mut Vec<GraphOverlay>,
    annotation_file_index_available: &mut HashMap<HashId, bool>,
    annotation_file_loaded_windows: &mut HashMap<HashId, (i64, i64)>,
    messages: &mut crate::views::messages::MessageBuffer,
) {
    if let Some(toggled_id) = explorer_state.annotation_file_toggle_requested.take() {
        if explorer_state.is_annotation_file_active(&toggled_id) {
            if let Some(entry) = ctx.explorer.annotation_file_entry(&toggled_id)
                && let Some(block_group) = ctx.current_block_group
            {
                let query_window = active_neighborhood_coordinate_window(ctx.graph_engine)
                    .map(expand_query_window);
                let node_filter = active_neighborhood_node_ids(ctx.graph_engine);
                let request = AnnotationFileTrackRequest {
                    conn: ctx.conn,
                    history_ref: ctx.history_ref,
                    workspace: ctx.workspace,
                    collection_name: ctx.collection_name,
                    sample_name: block_group.sample_name.as_str(),
                    block_group_name: Some(&block_group.name),
                    query_window,
                    node_filter: &node_filter,
                    entry,
                };
                match load_annotation_file_track(&request) {
                    Ok(load) => {
                        replace_track_overlays(
                            overlays,
                            &file_track_key(&toggled_id),
                            load.track.annotations,
                        );
                        annotation_file_index_available.insert(toggled_id, load.index_available);
                        if let Some(window) = load.loaded_window {
                            annotation_file_loaded_windows.insert(toggled_id, window);
                        } else {
                            annotation_file_loaded_windows.remove(&toggled_id);
                        }
                    }
                    Err(err) => {
                        messages.push_warn(format!("{err}"));
                        explorer_state.deactivate_annotation_file(&toggled_id);
                        remove_track_overlays(overlays, &file_track_key(&toggled_id));
                        annotation_file_index_available.remove(&toggled_id);
                        annotation_file_loaded_windows.remove(&toggled_id);
                    }
                }
            }
        } else {
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
pub(crate) fn teleport_through_wormhole<S: GraphSource<GenGraph>>(
    graph_engine: &mut LayoutEngine<GenGraph, S>,
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
    // back to where we came from rather than to whichever neighbour on that side sorts lowest.
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
    pub controller: Option<Box<GenGraphController<'a>>>,
}

/// The controller the full-screen viewer starts on: `handed_over` when given, otherwise a new
/// one, with `selected` open unless it already is.
///
/// A handed-over controller keeps its loaded batches, overlays and annotation groups. Its
/// camera was placed for the previous viewer's area, so it is reframed on the cursor node.
fn starting_controller<'a>(
    conn: &'a GraphConnection,
    workspace: &'a Workspace,
    history_ref: Option<&'a str>,
    handed_over: Option<Box<GenGraphController<'a>>>,
    selected: Option<&BlockGroup>,
) -> Result<GenGraphController<'a>, Box<dyn Error>> {
    let mut controller = match handed_over {
        Some(controller) => *controller,
        None => GenGraphController::new(conn, workspace, history_ref),
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

    // Create explorer and its state that persists across frames
    let mut explorer = CollectionExplorer::new(
        conn,
        config_conn,
        sample_name.as_deref(),
        current_block_group.as_ref(),
        collection_name,
        history_ref,
    );

    // Create the graph controller and initial graph
    let bar = progress_bar.add(get_time_elapsed_bar());

    // TODO: Handle origin positioning - not directly supported in new widget yet
    if position.is_some() {
        warn!("Origin positioning not yet supported in GenGraphWidget");
    }

    bar.finish();

    // Setup terminal
    let mut session = TuiSession::enter()?;
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;
    let terminal = session.terminal_mut();

    // Basic event loop
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

    // Track the last selected block group to detect changes
    let mut last_selected_block_group_id = block_group_id;
    // Track if we're loading a new block group
    let mut is_loading = false;
    let mut last_refresh = Instant::now();
    let mut should_quit = false;
    // Mouse capture reports every pointer movement; only events that can change the screen
    // earn a redraw.
    let mut needs_redraw = true;
    // Set when a key or click takes a door. Input after it is left unread until the new world
    // is drawn, then discarded.
    let mut entered_door = false;
    loop {
        // Drain ALL pending input events before doing any work
        while !entered_door && crossterm::event::poll(Duration::from_millis(0))? {
            let input = event::read()?;
            if !matches!(
                input,
                event::Event::Mouse(MouseEvent {
                    kind: MouseEventKind::Moved,
                    ..
                })
            ) {
                needs_redraw = true;
            }
            match input {
                event::Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if search_state.focused && !matches!(key.code, KeyCode::Tab | KeyCode::BackTab)
                    {
                        match search_state.handle_key(key) {
                            RegionSearchInputAction::Changed => {
                                let request = current_block_group.as_ref().map(|block_group| {
                                    RegionSearchRequest {
                                        conn,
                                        collection_name,
                                        sample_name: block_group.sample_name.as_str(),
                                    }
                                });
                                refresh_region_search(
                                    &mut search_state,
                                    &mut search_error,
                                    request.as_ref(),
                                );
                            }
                            RegionSearchInputAction::Cleared => {
                                search_error = None;
                                remove_search_overlay(controller.overlays_mut());
                            }
                            RegionSearchInputAction::Selected(search_match) => {
                                let (graph, graph_view_state, overlays) =
                                    controller.graph_view_and_overlays_mut();
                                match activate_search_match(
                                    graph_view_state,
                                    graph,
                                    overlays,
                                    search_match.as_ref(),
                                    conn,
                                    workspace,
                                ) {
                                    Ok(()) => {
                                        search_state.focused = false;
                                        search_state.clear_matches();
                                        search_error = None;
                                        focus_zone = FocusZone::Canvas;
                                    }
                                    Err(error) => search_error = Some(error),
                                }
                            }
                            RegionSearchInputAction::Closed => {
                                search_state.focused = false;
                                search_state.clear_matches();
                                search_error = None;
                            }
                            RegionSearchInputAction::Ignored => {}
                        }
                        continue;
                    }

                    if search_state.focused {
                        search_state.focused = false;
                        search_state.clear_matches();
                        search_error = None;
                    }

                    if is_region_search_command(key.code) {
                        if !search_state.focused {
                            focus_region_search(&mut search_state);
                            let request = current_block_group.as_ref().map(|block_group| {
                                RegionSearchRequest {
                                    conn,
                                    collection_name,
                                    sample_name: block_group.sample_name.as_str(),
                                }
                            });
                            refresh_region_search(
                                &mut search_state,
                                &mut search_error,
                                request.as_ref(),
                            );
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
                                | KeyCode::Char('h' | 'j' | 'k' | 'l' | 'w' | 'b')
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
                                entered_door =
                                    controller.handle_key(key) == GraphKeyOutcome::EnteredDoor;
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
                                    workspace,
                                    collection_name,
                                    current_block_group: current_block_group.as_ref(),
                                    graph_engine,
                                    explorer: &explorer,
                                },
                                &mut explorer_state,
                                overlays,
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
                        focus_region_search(&mut search_state);
                        let request =
                            current_block_group
                                .as_ref()
                                .map(|block_group| RegionSearchRequest {
                                    conn,
                                    collection_name,
                                    sample_name: block_group.sample_name.as_str(),
                                });
                        refresh_region_search(
                            &mut search_state,
                            &mut search_error,
                            request.as_ref(),
                        );
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
                            let (graph, graph_view_state, overlays) =
                                controller.graph_view_and_overlays_mut();
                            match activate_search_match(
                                graph_view_state,
                                graph,
                                overlays,
                                &search_match,
                                conn,
                                workspace,
                            ) {
                                Ok(()) => {
                                    search_state.focused = false;
                                    search_state.clear_matches();
                                    search_error = None;
                                    focus_zone = FocusZone::Canvas;
                                }
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
                            workspace,
                            collection_name,
                            current_block_group: current_block_group.as_ref(),
                            graph_engine,
                            explorer: &explorer,
                        },
                        &mut explorer_state,
                        overlays,
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
                            let graph_view_state = controller.view_state_mut();
                            graph_view_state.move_by_terminal(dx, dy);
                            graph_view_state.rebase_camera_to_closest_node();
                            mouse_is_dragging = true;
                        }
                        mouse_last_pos = Some((mouse.column, mouse.row));
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        if !mouse_is_dragging {
                            match controller
                                .view_state()
                                .wormhole_hit(mouse.column, mouse.row)
                            {
                                Some((boundary, target)) => {
                                    controller.teleport_through_wormhole(boundary, target);
                                    entered_door = true;
                                }
                                None => {
                                    controller
                                        .view_state_mut()
                                        .handle_click(mouse.column, mouse.row);
                                }
                            }
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

        // Trigger reload if selection changed to a new block group
        if explorer_state.selected_block_group_id != last_selected_block_group_id {
            is_loading = true;
            last_selected_block_group_id = explorer_state.selected_block_group_id;
        }

        // Refresh explorer data and force reload on change.
        // Skipped when loading — we want the draw to happen first so the loading
        // indicator is shown without any extra latency.
        // I do this every REFRESH_INTERVAL seconds.
        if !is_loading && last_refresh.elapsed() >= Duration::from_secs(REFRESH_INTERVAL) {
            let selected_sample = current_block_group
                .as_ref()
                .map(|bg| bg.sample_name.as_str());
            if explorer.refresh(
                conn,
                config_conn,
                selected_sample,
                current_block_group.as_ref(),
                collection_name,
                history_ref,
            ) {
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
            }
            last_refresh = Instant::now();
        }

        // Reload indexed annotation file tracks when the crawled neighborhood has changed
        // enough that the loaded window no longer covers it. The neighborhood's window and node
        // ids are only worth computing while an indexed file is active.
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

                let request = AnnotationFileTrackRequest {
                    conn,
                    history_ref,
                    workspace,
                    collection_name,
                    sample_name: block_group.sample_name.as_str(),
                    block_group_name: Some(&block_group.name),
                    query_window: Some(query_window),
                    node_filter: &node_filter,
                    entry,
                };
                needs_redraw = true;
                match load_annotation_file_track(&request) {
                    Ok(load) => {
                        replace_track_overlays(
                            controller.overlays_mut(),
                            &file_track_key(&id),
                            load.track.annotations,
                        );
                        if let Some(window) = load.loaded_window {
                            annotation_file_loaded_windows.insert(id, window);
                        } else {
                            annotation_file_loaded_windows.remove(&id);
                        }
                        annotation_file_index_available.insert(id, load.index_available);
                    }
                    Err(err) => {
                        messages.push_warn(format!("{err}"));
                        explorer_state.deactivate_annotation_file(&id);
                        remove_track_overlays(controller.overlays_mut(), &file_track_key(&id));
                        annotation_file_index_available.remove(&id);
                        annotation_file_loaded_windows.remove(&id);
                    }
                }
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
            let _ = crossterm::event::poll(Duration::from_secs(3600));
            continue;
        }
        needs_redraw = false;

        // Draw the UI
        terminal.draw(|frame| {
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

            let visible_search_matches = search_state.matches.len().min(5);
            let search_dropdown_rows = if visible_search_matches > 0 {
                visible_search_matches as u16 + 2
            } else if search_state.focused && search_error.is_some() {
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
                    vec![ListItem::new(Line::from(Span::styled(
                        search_error.as_deref().unwrap_or("no matching regions"),
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
                            format!("*←→↑↓* move | *w/b* next/prev annotation | *enter* details | *+/-* zoom | *p* path | *m* messages | *g* search | *tab* {tab_dest}")
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
                    AnnotationDisplay::FlagsUnderNodes,
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

        // The render just above may have claimed a new world, growing the graph and needing
        // that batch's annotation groups; if anything drawn changed, the frame is redrawn
        // below.
        let post_draw_sync = controller.sync_active_world();
        let changed_after_draw = post_draw_sync.changed;
        apply_group_reload(post_draw_sync, &mut explorer_state, &mut messages);

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
                collection_name,
                history_ref,
            ) {
                explorer.force_reload(&mut explorer_state);
                explorer_state.retain_annotation_files(&explorer.data.annotation_files);
                explorer_state.retain_annotation_groups(&explorer.data.annotation_groups);
            }
            annotation_file_index_available.clear();
            annotation_file_loaded_windows.clear();
            explorer_state.active_annotation_groups.clear();
            if let Some(block_group) = current_block_group.as_ref() {
                let node_filter = active_neighborhood_node_ids(controller.engine());
                let query_window = active_neighborhood_coordinate_window(controller.engine())
                    .map(expand_query_window);
                for entry in &explorer.data.annotation_files {
                    let id = entry.file_addition.id;
                    if !explorer_state.is_annotation_file_active(&id) {
                        continue;
                    }
                    let request = AnnotationFileTrackRequest {
                        conn,
                        history_ref,
                        workspace,
                        collection_name,
                        sample_name: block_group.sample_name.as_str(),
                        block_group_name: Some(&block_group.name),
                        query_window,
                        node_filter: &node_filter,
                        entry,
                    };
                    match load_annotation_file_track(&request) {
                        Ok(load) => {
                            replace_track_overlays(
                                controller.overlays_mut(),
                                &file_track_key(&id),
                                load.track.annotations,
                            );
                            if let Some(window) = load.loaded_window {
                                annotation_file_loaded_windows.insert(id, window);
                            }
                            annotation_file_index_available.insert(id, load.index_available);
                        }
                        Err(err) => {
                            messages.push_warn(format!("{err}"));
                            explorer_state.deactivate_annotation_file(&id);
                        }
                    }
                }
            }

            is_loading = false;
            search_state.clear_matches();
            search_state.focused = false;
            search_error = None;
            needs_redraw = true;
            continue;
        }

        // The overlays, dimming, or loaded graph changed after the frame was rendered. Draw
        // them immediately instead of waiting for the next keyboard or mouse event to wake the
        // idle viewer.
        if changed_after_draw {
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

        // Rendering is event-driven (no animation to advance): block indefinitely until the
        // next input event wakes us.
        let _ = crossterm::event::poll(Duration::from_secs(3600));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use gen_core::{HashId, Workspace};
    use gen_graph::{GenGraph, GraphNode};
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup},
        db::get_connection,
        path::Path,
    };
    use gen_tui::{graph_view::GraphViewState, layout_engine::LayoutEngine};
    use ratatui::{Terminal, backend::TestBackend, style::Style};

    use super::{
        RegionSearchInputAction, RegionSearchState, focus_region_search, is_region_search_command,
        refresh_region_search, starting_controller, teleport_through_wormhole,
    };
    use crate::views::{
        gen_graph_controller::{AnnotationDisplay, GenGraphController},
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

        let mut state = RegionSearchState::default();
        state.set_matches(matches);
        assert_eq!(state.selected_match, None);
        assert!(matches!(
            state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            RegionSearchInputAction::Ignored
        ));
        state.move_selection(1);
        assert_eq!(state.selected_match, Some(0));
        state.move_selection(1);
        assert_eq!(state.selected_match, Some(1));
        assert!(matches!(
            state.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE)),
            RegionSearchInputAction::Changed
        ));
        assert_eq!(state.selected_match, None);
        assert!(matches!(
            state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            RegionSearchInputAction::Ignored
        ));
        state.move_selection(1);
        assert!(matches!(
            state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            RegionSearchInputAction::Selected(_)
        ));
    }

    #[test]
    fn test_region_search_clear_action_empties_query_and_matches() {
        let request = search_request_fixture();
        let mut state = RegionSearchState {
            query: "chr1:5-10".to_string(),
            matches: resolve_region_search_matches(&request.request(), "chr1:5-10")
                .expect("should resolve the clear-action fixture query"),
            selected_match: Some(0),
            focused: true,
        };

        assert!(matches!(
            state.handle_key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL)),
            RegionSearchInputAction::Cleared
        ));
        assert!(state.query.is_empty());
        assert!(state.matches.is_empty());
        assert_eq!(state.selected_match, None);
        assert!(state.focused);

        state.query = "x".to_string();
        state.matches = resolve_region_search_matches(&request.request(), "chr1:5-10")
            .expect("should resolve the query before backspace clears it");
        state.selected_match = Some(0);
        assert!(matches!(
            state.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE)),
            RegionSearchInputAction::Cleared
        ));
        assert!(state.query.is_empty());
        assert!(state.matches.is_empty());
    }

    #[test]
    fn test_region_search_refocus_preserves_query_without_default_selection() {
        let request = search_request_fixture();
        let mut state = RegionSearchState {
            query: "chr1:5-10".to_string(),
            ..RegionSearchState::default()
        };
        state.set_matches(
            resolve_region_search_matches(&request.request(), &state.query)
                .expect("should resolve the preserved query"),
        );
        state.selected_match = Some(0);
        state.focused = false;

        focus_region_search(&mut state);
        assert_eq!(state.query, "chr1:5-10");
        assert_eq!(state.selected_match, None);
        assert!(state.matches.is_empty());

        let mut search_error = None;
        refresh_region_search(&mut state, &mut search_error, Some(&request.request()));
        assert_eq!(state.query, "chr1:5-10");
        assert_eq!(state.matches.len(), 1);
        assert_eq!(state.selected_match, None);
        assert!(search_error.is_none());
    }

    fn three_node_chain() -> (LayoutEngine<GenGraph>, [GraphNode; 3]) {
        let nodes = ["left", "middle", "right"].map(|label| GraphNode {
            node_id: HashId::convert_str(label),
            sequence_start: 0,
            sequence_end: 5,
        });
        let mut graph = GenGraph::new();
        graph.add_edge(nodes[0], nodes[1], Vec::new());
        graph.add_edge(nodes[1], nodes[2], Vec::new());
        (LayoutEngine::new(graph), nodes)
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
                controller.render(
                    frame,
                    frame.area(),
                    AnnotationDisplay::FloatingLabels,
                    Style::default(),
                );
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
        let mut inline =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
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
        let mut inline =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
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
