use std::{
    collections::{HashMap, HashSet},
    error::Error,
    path::PathBuf,
    time::{Duration, Instant},
};

use crossterm::event::{
    self, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use gen_core::{HashId, PATH_START_NODE_ID, Workspace, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection};
use gen_tui::{
    LineStyle,
    crawl::{EagerSource, GraphSource},
    graph_view::{GraphView, GraphViewState},
    layout::VisualDetail,
    layout_engine::{BatchId, LayoutEngine},
    plotter::PathStyle,
    theme::current_theme,
};
use log::{info, warn};
use ratatui::{
    layout::{Constraint, Direction, HorizontalAlignment, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, List, ListItem, Padding, Paragraph, Wrap},
};

use crate::{
    progress_bar::{get_handler, get_time_elapsed_bar},
    views::{
        annotation_groups::load_annotation_group_entries,
        annotations::{
            AnnotationFileTrackRequest, AnnotationGroupTrackRequest, load_annotation_file_track,
            load_annotations_for_group,
        },
        collection::{CollectionExplorer, CollectionExplorerState, FocusZone},
        gen_graph_widget::{
            self, NodeAnnotationLayer, create_annotated_gen_graph_engine_lazy,
            draw_annotation_connectors, draw_annotation_labels, reapply_overlays,
            update_node_annotations,
        },
        graph_dimming::GraphDimming,
        graph_overlay::{
            AnnotationColorCache, GraphOverlay, OverlaySource, PathMembership, file_track_key,
            group_track_key, has_path_overlay, remove_path_overlay, remove_track_overlays,
            replace_track_overlays, set_path_overlay,
        },
        lazy_graph_source::{EagerOrSqlSource, SqlGraphSource, seed_block_group_graph},
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

fn get_empty_graph() -> GenGraph {
    let mut g = GenGraph::new();
    g.add_node(GraphNode {
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    });
    g
}

/// Load `block_group_id`'s graph and the source its `LayoutEngine` should crawl through.
///
/// A historical view (`history_ref: Some(_)`) always eager-loads the full graph up front:
/// the port queries the lazy path below is built on can only answer for the live graph.
/// Otherwise, the graph is seeded with just its `PATH_START` sentinel and grown lazily from
/// SQLite as the viewer's crawl pushes past its frontier - see `SqlGraphSource`, which is what
/// turns opening a large block group from a full-graph-materializing stall into an
/// near-instant open.
pub(crate) fn load_block_group_graph(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: &gen_core::HashId,
    history_ref: Option<&str>,
) -> Result<(GenGraph, EagerOrSqlSource), Box<dyn Error>> {
    if history_ref.is_some() {
        let graph = BlockGroup::get_graph(conn, workspace, block_group_id, history_ref)?;
        return Ok((graph, EagerOrSqlSource::Eager(EagerSource)));
    }
    let db_path = conn
        .path()
        .map(PathBuf::from)
        .ok_or("graph database has no file path")?;
    let source = SqlGraphSource::new(db_path, *block_group_id);
    let seed = seed_block_group_graph(conn, block_group_id);
    Ok((seed, EagerOrSqlSource::Sql(Box::new(source))))
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

#[expect(
    clippy::too_many_arguments,
    reason = "keeps the neighborhood loader's database and view state explicit"
)]
fn load_annotation_groups_for_neighborhood(
    conn: &GraphConnection,
    workspace: &Workspace,
    history_ref: Option<&str>,
    block_group: &BlockGroup,
    node_ids: &HashSet<HashId>,
    explorer_state: &mut CollectionExplorerState,
    overlays: &mut Vec<GraphOverlay>,
    messages: &mut crate::views::messages::MessageBuffer,
) {
    for entry in load_annotation_group_entries(conn, block_group, history_ref) {
        let spans = match load_annotations_for_group(&AnnotationGroupTrackRequest {
            conn,
            workspace,
            history_ref,
            current_block_group: block_group,
            entry: &entry,
            node_ids,
        }) {
            Ok(spans) => spans,
            Err(err) => {
                messages.push_warn(format!(
                    "Failed to load annotations for group {}: {err}",
                    entry.id
                ));
                continue;
            }
        };
        if spans.is_empty() {
            continue;
        }
        explorer_state
            .active_annotation_groups
            .insert(entry.id.clone());
        replace_track_overlays(overlays, &group_track_key(&entry.id), spans);
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
            if let Some(block_group) = ctx.current_block_group {
                let node_ids = active_neighborhood_node_ids(ctx.graph_engine);
                let entry = ctx.explorer.annotation_group_entry(&toggled_group);
                let spans = match entry.map(|entry| {
                    load_annotations_for_group(&AnnotationGroupTrackRequest {
                        conn: ctx.conn,
                        workspace: ctx.workspace,
                        history_ref: ctx.history_ref,
                        current_block_group: block_group,
                        entry,
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

/// Toggle path highlighting for a block group.
///
/// The path lives in `overlays` alongside the annotation overlays and is repainted each
/// frame by the render loop, so this only adds or removes it. The block group's current path
/// is fetched once as edge membership; each repaint highlights whichever of its edges the
/// crawl has loaded so far. Returns whether the path overlay is now enabled.
fn toggle_path_highlight(
    conn: &GraphConnection,
    history_ref: Option<&str>,
    block_group_id: &gen_core::HashId,
    color: ratatui::style::Color,
    overlays: &mut Vec<GraphOverlay>,
) -> Result<bool, String> {
    if has_path_overlay(overlays) {
        remove_path_overlay(overlays);
        Ok(false)
    } else {
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        let path = BlockGroup::get_current_path(conn, block_group_id, history_ref)
            .map_err(|error| format!("Failed to query path: {error}"))?;
        let membership = PathMembership::load(conn, &path.id, history_ref);
        if membership.is_empty() {
            return Err("Path has no edges".to_string());
        }
        set_path_overlay(overlays, style, membership);
        Ok(true)
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
}

pub fn view_block_group(
    conn: &GraphConnection,
    config_conn: &gen_models::db::ConfigConnection,
    workspace: &Workspace,
    options: BlockGroupViewOptions<'_>,
) -> Result<(), Box<dyn Error>> {
    let BlockGroupViewOptions {
        name,
        sample_name,
        collection_name,
        position,
        history_ref,
    } = options;
    let progress_bar = get_handler();
    let bar = progress_bar.add(get_time_elapsed_bar());
    let _ = progress_bar.println("Loading block group");

    let mut block_graph;
    let mut graph_source;
    let mut block_group_id: Option<gen_core::HashId> = None;
    let mut focus_zone = FocusZone::Sidebar;
    let mut explorer_state = CollectionExplorerState::new();
    if let Some(ref s) = sample_name {
        explorer_state.set_sample_expanded(s, true);
    }

    if let (Some(name), Some(sample_name)) = (name, sample_name.as_ref()) {
        let block_group =
            BlockGroup::get_by_name(conn, collection_name, sample_name, &name, history_ref)
                .unwrap_or_else(|_| {
                    panic!(
                        "No block group found with name {:?} and sample {:?} in collection {} ",
                        name, sample_name, collection_name
                    )
                });
        block_group_id = Some(block_group.id);
        (block_graph, graph_source) =
            load_block_group_graph(conn, workspace, &block_group.id, history_ref)?;
        explorer_state.selected_block_group_id = Some(block_group.id);
        focus_zone = FocusZone::Canvas;
    } else {
        block_graph = get_empty_graph();
        graph_source = EagerOrSqlSource::Eager(EagerSource);
    }

    bar.finish();

    let mut messages = crate::views::messages::MessageBuffer::new(MESSAGE_BUFFER_LIMIT);
    // Every annotation currently painted on the canvas, from both loaded files and
    // loaded groups, keyed by track (see `graph_overlay::file_track_key`/`group_track_key`).
    let mut overlays: Vec<GraphOverlay> = Vec::new();
    let mut annotation_colors = AnnotationColorCache::new();
    let mut annotation_file_index_available: HashMap<HashId, bool> = HashMap::new();
    let mut annotation_file_loaded_windows: HashMap<HashId, (i64, i64)> = HashMap::new();
    let mut current_block_group =
        block_group_id.map(
            |bg_id| match BlockGroup::get_by_id(conn, &bg_id, history_ref) {
                Ok(bg) => bg,
                Err(err) => {
                    // TODO: Handle these with messages instead of panic'ing
                    panic!("Failed to load block group {bg_id}: {err}");
                }
            },
        );

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
    let _ = progress_bar.println("Pre-computing layout in chunks");

    // Annotation flags drawn under nodes at full detail; refilled from `overlays` each frame.
    let node_annotations = NodeAnnotationLayer::new();
    let (mut graph_engine, mut graph_zoom_levels, mut graph_view_state) =
        create_annotated_gen_graph_engine_lazy(
            block_graph,
            graph_source,
            conn,
            node_annotations.clone(),
        );
    // Pruned edges and the nodes only they lead into, kept up to date as the crawl grows the
    // graph: synced before each draw, and again after it since rendering can crawl too.
    let mut graph_dimming = GraphDimming::default();

    // TODO: Handle origin positioning - not directly supported in new widget yet
    if position.is_some() {
        warn!("Origin positioning not yet supported in GenGraphWidget");
    }

    bar.finish();

    let mut annotation_groups_loaded = false;
    // The active neighborhood the annotation groups were last loaded for. The crawled
    // neighborhood is already the deliberately-constrained local window, so a reload is
    // only needed when it changes (block group switch, wormhole teleport into a
    // different world) - not on every pan/zoom within the same neighborhood.
    let mut annotation_groups_world: Option<BatchId> = None;

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
    loop {
        // Drain ALL pending input events before doing any work
        while crossterm::event::poll(Duration::from_millis(0))? {
            match event::read()? {
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
                                remove_search_overlay(&mut overlays);
                            }
                            RegionSearchInputAction::Selected(search_match) => {
                                match activate_search_match(
                                    &mut graph_view_state,
                                    graph_engine.graph(),
                                    &mut overlays,
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
                    if !graph_view_state.is_cursor_visible()
                        && matches!(
                            key.code,
                            KeyCode::Left
                                | KeyCode::Right
                                | KeyCode::Up
                                | KeyCode::Down
                                | KeyCode::Char('h' | 'j' | 'k' | 'l' | 'w' | 'b')
                        )
                    {
                        graph_view_state.show_cursor();
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
                                if !graph_view_state.is_cursor_visible() {
                                    graph_view_state.show_cursor();
                                } else if !show_panel {
                                    focus_zone = FocusZone::Sidebar;
                                }
                            }
                            // Annotation starts are only known in screen columns where the
                            // annotations are drawn under their nodes, at full detail.
                            KeyCode::Char(key_char @ ('w' | 'b'))
                                if graph_zoom_levels[graph_view_state.zoom_index].0
                                    == VisualDetail::Full =>
                            {
                                // Reaching the edge of the loaded batch leaves the cursor
                                // where it is, like an arrow key with nothing beyond it.
                                let _ = graph_view_state
                                    .move_cursor_to_stop(key_char == 'w', |node| {
                                        node_annotations.annotation_starts(&node)
                                    });
                            }
                            KeyCode::Char('p') => {
                                if let Some(ref block_group_id) =
                                    explorer_state.selected_block_group_id
                                {
                                    match toggle_path_highlight(
                                        conn,
                                        history_ref,
                                        block_group_id,
                                        Color::Red,
                                        &mut overlays,
                                    ) {
                                        Ok(highlighting_enabled) => {
                                            if highlighting_enabled {
                                                info!(
                                                    "Path highlighting enabled for block group {}",
                                                    block_group_id
                                                );
                                            } else {
                                                info!("Path highlighting disabled");
                                            }
                                        }
                                        Err(err) => {
                                            warn!("Failed to toggle path highlighting: {}", err);
                                        }
                                    }
                                } else {
                                    warn!("No block group selected for path highlighting");
                                }
                            }
                            KeyCode::Char('+') | KeyCode::Char('=') => {
                                gen_graph_widget::zoom_in(
                                    &mut graph_view_state,
                                    &graph_zoom_levels,
                                );
                            }
                            KeyCode::Char('-') => {
                                gen_graph_widget::zoom_out(
                                    &mut graph_view_state,
                                    &graph_zoom_levels,
                                );
                            }
                            _ => {
                                if let Ok(Some((boundary, target))) =
                                    graph_view_state.handle_key_event(key)
                                {
                                    teleport_through_wormhole(
                                        &mut graph_engine,
                                        &mut graph_view_state,
                                        boundary,
                                        target,
                                    );
                                }
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
                            handle_annotation_toggle_requests(
                                &AnnotationToggleContext {
                                    conn,
                                    history_ref,
                                    workspace,
                                    collection_name,
                                    current_block_group: current_block_group.as_ref(),
                                    graph_engine: &graph_engine,
                                    explorer: &explorer,
                                },
                                &mut explorer_state,
                                &mut overlays,
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
                            match activate_search_match(
                                &mut graph_view_state,
                                graph_engine.graph(),
                                &mut overlays,
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
                    handle_annotation_toggle_requests(
                        &AnnotationToggleContext {
                            conn,
                            history_ref,
                            workspace,
                            collection_name,
                            current_block_group: current_block_group.as_ref(),
                            graph_engine: &graph_engine,
                            explorer: &explorer,
                        },
                        &mut explorer_state,
                        &mut overlays,
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
                            graph_view_state.move_by_terminal(dx, dy);
                            graph_view_state.rebase_camera_to_closest_node();
                            mouse_is_dragging = true;
                        }
                        mouse_last_pos = Some((mouse.column, mouse.row));
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        if !mouse_is_dragging {
                            match graph_view_state.wormhole_hit(mouse.column, mouse.row) {
                                Some((boundary, target)) => teleport_through_wormhole(
                                    &mut graph_engine,
                                    &mut graph_view_state,
                                    boundary,
                                    target,
                                ),
                                None => {
                                    graph_view_state.handle_click(mouse.column, mouse.row);
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
                annotation_file_index_available
                    .retain(|id, _| explorer_state.is_annotation_file_active(id));
                annotation_file_loaded_windows
                    .retain(|id, _| explorer_state.is_annotation_file_active(id));
                let active_file_keys: HashSet<String> = explorer_state
                    .active_annotation_files
                    .iter()
                    .map(file_track_key)
                    .collect();
                overlays.retain(|overlay| match &overlay.source {
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
        // enough that the loaded window no longer covers it. Annotation group reload
        // piggybacks on the same neighborhood-changed signal.
        if !is_loading
            && let Some(block_group) = current_block_group.as_ref()
            && let Some(visible_window) = active_neighborhood_coordinate_window(&graph_engine)
        {
            if graph_engine.active_batch() != annotation_groups_world {
                annotation_groups_loaded = false;
            }
            let query_window = expand_query_window(visible_window);
            let node_filter = active_neighborhood_node_ids(&graph_engine);
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
                match load_annotation_file_track(&request) {
                    Ok(load) => {
                        replace_track_overlays(
                            &mut overlays,
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
                        remove_track_overlays(&mut overlays, &file_track_key(&id));
                        annotation_file_index_available.remove(&id);
                        annotation_file_loaded_windows.remove(&id);
                    }
                }
            }
        }

        // A teleport or jump handled above may already have grown the graph.
        graph_dimming.sync(
            graph_engine.graph(),
            graph_engine.source(),
            &mut graph_view_state,
        );

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
                        if !graph_view_state.is_cursor_visible() {
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

                // Re-register overlay highlights before rendering. This reruns every frame
                // because `overlays` can change between frames (file/group toggles,
                // scroll-triggered reloads).
                reapply_overlays(
                    &graph_engine,
                    &mut graph_view_state,
                    &graph_zoom_levels,
                    &mut overlays,
                    &mut annotation_colors,
                );
                // Names with no room under their node fall back to floating labels.
                let floating_overlays =
                    update_node_annotations(&node_annotations, &graph_engine, &overlays);

                let active_renderer = &graph_zoom_levels[graph_view_state.zoom_index].1;
                let view = GraphView::new(&mut graph_engine, active_renderer).style(canvas_style);
                frame.render_stateful_widget(view, main_canvas_area, &mut graph_view_state);

                // Draw floating labels after the graph, then a single hint if any were hidden.
                // At full detail the renderer draws annotations as flags under their nodes,
                // so only the names that found no room there still float.
                let detail_level = graph_zoom_levels[graph_view_state.zoom_index].0;
                let labelled_overlays = if detail_level == VisualDetail::Full {
                    draw_annotation_connectors(
                        frame.buffer_mut(),
                        main_canvas_area,
                        &graph_view_state.frame,
                        &node_annotations,
                        None,
                    );
                    &floating_overlays
                } else {
                    &overlays
                };
                let any_hidden = draw_annotation_labels(
                    frame.buffer_mut(),
                    main_canvas_area,
                    &graph_engine,
                    &graph_view_state,
                    &graph_zoom_levels,
                    labelled_overlays,
                );
                if any_hidden {
                    let note = if detail_level == VisualDetail::Full {
                        " some annotations hidden due to space constraints "
                    } else {
                        " some annotations hidden in truncated view "
                    };
                    let note_style =
                        Style::default().fg(current_theme()[0x09]).bg(current_theme()[0x00]);
                    frame.buffer_mut().set_string(
                        main_canvas_area.x,
                        main_canvas_area.bottom().saturating_sub(1),
                        note,
                        note_style,
                    );
                }
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

                        if let Some(graph_node) = graph_view_state.cursor.node {
                            let node_id_short =
                                graph_node.node_id.to_string().chars().take(12).collect::<String>();
                            let (frac_x, _) = graph_view_state.cursor.fractional;
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

        // The render just above may have claimed a new world, growing the graph; if that
        // changed any dimming, the frame is redrawn below.
        let dimming_changed_after_draw = graph_dimming.sync(
            graph_engine.graph(),
            graph_engine.source(),
            &mut graph_view_state,
        );

        // Load (or reload) annotation groups for the active crawled neighborhood - that
        // neighborhood is already the deliberately-constrained local window, so this is
        // the sole source for which segments to fetch (no further viewport subsetting).
        let mut annotation_groups_loaded_after_draw = false;
        if !annotation_groups_loaded && let Some(block_group) = current_block_group.as_ref() {
            let node_ids = active_neighborhood_node_ids(&graph_engine);
            if !node_ids.is_empty() {
                overlays.retain(
                    |o| !matches!(&o.source, OverlaySource::Track(k) if k.starts_with("group:")),
                );
                explorer_state.active_annotation_groups.clear();
                load_annotation_groups_for_neighborhood(
                    conn,
                    workspace,
                    history_ref,
                    block_group,
                    &node_ids,
                    &mut explorer_state,
                    &mut overlays,
                    &mut messages,
                );
                annotation_groups_loaded = true;
                annotation_groups_loaded_after_draw = true;
                annotation_groups_world = graph_engine.active_batch();
            }
        }

        // Update the graph controller if a new block group was selected.
        // This runs after terminal.draw() so the loading indicator is visible
        // for the full duration of the blocking DB work.
        if is_loading && let Some(ref new_block_group_id) = explorer_state.selected_block_group_id {
            // Create a new graph (and matching source) for the selected block group
            (block_graph, graph_source) =
                load_block_group_graph(conn, workspace, new_block_group_id, history_ref)?;
            // Update the graph engine
            (graph_engine, graph_zoom_levels, graph_view_state) =
                create_annotated_gen_graph_engine_lazy(
                    block_graph,
                    graph_source,
                    conn,
                    node_annotations.clone(),
                );
            graph_dimming = GraphDimming::default();
            let block_group = match BlockGroup::get_by_id(conn, new_block_group_id, history_ref) {
                Ok(bg) => bg,
                Err(err) => {
                    // TODO: Handle these with messages instead of panic'ing
                    panic!("Failed to load block group {}: {err}", new_block_group_id);
                }
            };

            current_block_group = Some(block_group);
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
            overlays.clear();
            annotation_file_index_available.clear();
            annotation_file_loaded_windows.clear();
            explorer_state.active_annotation_groups.clear();
            annotation_groups_loaded = false;
            annotation_groups_world = None;
            if let Some(block_group) = current_block_group.as_ref() {
                let node_filter = active_neighborhood_node_ids(&graph_engine);
                let query_window =
                    active_neighborhood_coordinate_window(&graph_engine).map(expand_query_window);
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
                                &mut overlays,
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
            continue;
        }

        // The overlays or dimming changed after the frame was rendered. Draw them immediately
        // instead of waiting for the next keyboard or mouse event to wake the idle viewer.
        if annotation_groups_loaded_after_draw || dimming_changed_after_draw {
            continue;
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
    use gen_core::HashId;
    use gen_graph::{GenGraph, GraphNode};
    use gen_tui::{graph_view::GraphViewState, layout_engine::LayoutEngine};

    use super::{
        RegionSearchInputAction, RegionSearchState, focus_region_search, is_region_search_command,
        refresh_region_search, teleport_through_wormhole,
    };
    use crate::views::region_search::{resolve_region_search_matches, search_request_fixture};

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
}
