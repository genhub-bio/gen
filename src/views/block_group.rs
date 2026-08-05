use std::{
    collections::HashSet,
    error::Error,
    time::{Duration, Instant},
};

use crossterm::event::{self, KeyCode, KeyEventKind, MouseButton, MouseEventKind};
use gen_core::{HashId, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection, node::Node, traits::Query};
use gen_tui::{
    LineStyle, graph_controller::GraphController, layout::VisualDetail, plotter::PathStyle,
    theme::current_theme,
};
use log::{info, warn};
use ratatui::{
    layout::{Constraint, Direction, HorizontalAlignment, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Padding, Paragraph, Wrap},
};
use rusqlite::params;

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
            GenGraphNodeSizer, create_gen_graph_controller, create_gen_graph_widget,
            draw_annotation_labels, reapply_overlays,
        },
        graph_overlay::{
            AnnotationColorCache, GraphOverlay, OverlaySource, file_track_key, group_track_key,
            has_path_overlay, project_path_overlay_nodes, remove_path_overlay,
            remove_track_overlays, replace_track_overlays, set_path_overlay,
        },
        panels::{render_status_bar, render_with_optional_clear},
        tui_runtime::TuiSession,
    },
};

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

/// Get the most recent path for a block group and map it to GraphNodes in the current graph
fn get_block_group_path_nodes(
    conn: &GraphConnection,
    block_group_id: &gen_core::HashId,
    graph: &GenGraph,
) -> Result<Vec<gen_graph::GraphNode>, String> {
    use gen_models::path::Path;

    // Query the database for the most recent path for this block group
    let path = Path::get(
        conn,
        "SELECT * FROM paths WHERE block_group_id = ?1 ORDER BY created_on DESC LIMIT 1",
        rusqlite::params![block_group_id],
    )
    .map_err(|e| format!("Failed to query path: {}", e))?;

    let path_blocks = path
        .blocks(conn, None)
        .map_err(|err| format!("Failed to load path blocks: {err}"))?;

    let path_nodes = project_path_overlay_nodes(graph, &path_blocks);
    if path_nodes.is_empty() {
        return Err("Path nodes not found in current graph state".to_string());
    }

    Ok(path_nodes)
}

/// Toggle the path overlay for a block group.
///
/// The path lives in `overlays` alongside the annotation overlays and is repainted each
/// frame by the render loop, so this only adds or removes it. Returns whether the path
/// overlay is now enabled.
fn toggle_path_highlight(
    conn: &GraphConnection,
    controller: &GraphController<GenGraph, GenGraphNodeSizer>,
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
        let path_nodes = get_block_group_path_nodes(conn, block_group_id, controller.graph())?;
        set_path_overlay(overlays, style, path_nodes);
        Ok(true)
    }
}

/// Node IDs present in the current viewport (excluding terminal start/end nodes).
pub(crate) fn extract_viewport_node_ids(
    controller: &GraphController<GenGraph, GenGraphNodeSizer>,
) -> HashSet<HashId> {
    use gen_core::{is_end_node, is_start_node};
    use petgraph::visit::NodeIndexable;
    let graph = controller.graph();
    controller
        .get_viewport_graph()
        .data_nodes()
        .map(|(_, idx, _)| <&GenGraph as NodeIndexable>::from_index(&graph, idx.index()).node_id)
        .filter(|&id| !is_start_node(id) && !is_end_node(id))
        .collect()
}

/// Compute the coordinate window (min sequence start, max sequence end) of visible blocks
/// in the current viewport, using the graph controller's viewport graph.
pub(crate) fn current_view_coordinate_window(
    controller: &GraphController<GenGraph, GenGraphNodeSizer>,
) -> Option<(i64, i64)> {
    use gen_core::{is_end_node, is_start_node};
    use petgraph::visit::NodeIndexable;

    let viewport_graph = controller.get_viewport_graph();
    let graph = controller.graph();
    let mut start = i64::MAX;
    let mut end = i64::MIN;

    for (_world_pos, domain_idx, _layout_node) in viewport_graph.data_nodes() {
        let block = <&GenGraph as NodeIndexable>::from_index(&graph, domain_idx.index());
        if is_start_node(block.node_id) || is_end_node(block.node_id) {
            continue;
        }
        start = start.min(block.sequence_start);
        end = end.max(block.sequence_end);
    }

    (start <= end).then_some((start, end))
}

pub(crate) fn expand_query_window(window: (i64, i64)) -> (i64, i64) {
    let span = (window.1 - window.0).max(1);
    (window.0.saturating_sub(span), window.1.saturating_add(span))
}

fn load_annotation_groups_for_viewport(
    conn: &GraphConnection,
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

#[expect(
    clippy::too_many_arguments,
    reason = "CLI entrypoint needs to forward explicit view selection and history state"
)]
pub fn view_block_group(
    conn: &GraphConnection,
    config_conn: &gen_models::db::ConfigConnection,
    workspace: &gen_core::config::Workspace,
    name: Option<String>,
    sample_name: Option<String>,
    collection_name: &str,
    position: Option<String>, // Node ID and offset
    history_ref: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let mut current_collection_name = collection_name.to_string();
    let progress_bar = get_handler();
    let bar = progress_bar.add(get_time_elapsed_bar());
    let _ = progress_bar.println("Loading block group");

    // Get the node object corresponding to the position given by the user
    let origin = if let Some(position_str) = position {
        let parts = position_str.split(":").collect::<Vec<&str>>();
        if parts.len() != 2 {
            panic!("Invalid position: {}", position_str);
        }
        let node_id = parts[0].parse::<i64>().unwrap();
        let offset = parts[1].parse::<i64>().unwrap();
        Some((
            Node::get(conn, "select * from nodes where id = ?1", params![node_id]).unwrap(),
            offset,
        ))
    } else {
        None
    };

    let mut block_graph;
    let mut block_group_id: Option<gen_core::HashId> = None;
    let mut focus_zone = FocusZone::Sidebar;
    let mut explorer_state = CollectionExplorerState::new();
    if let Some(ref s) = sample_name {
        explorer_state.set_sample_expanded(s, true);
    }

    if let (Some(name), Some(sample_name)) = (name, sample_name.as_ref()) {
        let block_group = BlockGroup::get_by_name(
            conn,
            &current_collection_name,
            sample_name,
            &name,
            history_ref,
        );

        if block_group.is_err() {
            panic!(
                "No block group found with name {:?} and sample {:?} in collection {} ",
                name,
                sample_name.clone(),
                current_collection_name
            );
        }

        let block_group = block_group.unwrap();
        block_group_id = Some(block_group.id);
        block_graph =
            gen_graph::models::load_block_group_graph(conn, &block_group.id, history_ref)?;
        explorer_state.selected_block_group_id = Some(block_group.id);
        focus_zone = FocusZone::Canvas;
    } else {
        block_graph = get_empty_graph();
    }

    bar.finish();

    let mut messages = crate::views::messages::MessageBuffer::new(MESSAGE_BUFFER_LIMIT);
    // Every annotation currently painted on the canvas, from both loaded files and
    // loaded groups, keyed by track (see `graph_overlay::file_track_key`/`group_track_key`).
    let mut overlays: Vec<GraphOverlay> = Vec::new();
    let mut annotation_colors = AnnotationColorCache::new();
    let mut annotation_file_index_available: std::collections::HashMap<HashId, bool> =
        std::collections::HashMap::new();
    let mut annotation_file_loaded_windows: std::collections::HashMap<HashId, (i64, i64)> =
        std::collections::HashMap::new();
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
        &current_collection_name,
        history_ref,
    );

    // Create the graph controller and initial graph
    let bar = progress_bar.add(get_time_elapsed_bar());
    let _ = progress_bar.println("Pre-computing layout in chunks");

    let mut graph_controller = create_gen_graph_controller(block_graph.clone());

    // TODO: Handle origin positioning - not directly supported in new widget yet
    if origin.is_some() {
        warn!("Origin positioning not yet supported in GenGraphWidget");
    }

    bar.finish();

    let mut annotation_groups_loaded = false;

    // Setup terminal
    let mut session = TuiSession::enter()?;
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;
    let terminal = session.terminal_mut();

    // Basic event loop
    let tick_rate = Duration::from_millis(16); // ~60fps
    let mut last_tick = Instant::now();
    let mut last_frame_time = Instant::now();
    let mut show_panel = false;
    let mut panel_mode = PanelMode::Details;
    let show_sidebar = true;
    let mut tui_layout_change = false;

    // Mouse drag state
    let mut mouse_last_pos: Option<(u16, u16)> = None;
    let mut mouse_is_dragging = false;
    let mut last_sidebar_area = Rect::default();

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
                    // Any keyboard navigation shows the cursor.
                    if !graph_controller.is_cursor_visible()
                        && matches!(
                            key.code,
                            KeyCode::Left
                                | KeyCode::Right
                                | KeyCode::Up
                                | KeyCode::Down
                                | KeyCode::Char('h' | 'j' | 'k' | 'l')
                        )
                    {
                        graph_controller.show_cursor();
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
                                if graph_controller.cursor.is_coarse_mode() {
                                    graph_controller.cursor.set_coarse_mode(false);
                                } else {
                                    // TODO: Node selection not yet supported, always show panel for now
                                    show_panel = true;
                                    panel_mode = PanelMode::Details;
                                    focus_zone = FocusZone::Panel;
                                    tui_layout_change = true;
                                }
                            }
                            KeyCode::Esc => {
                                if !graph_controller.is_cursor_visible() {
                                    graph_controller.show_cursor();
                                } else if !graph_controller.cursor.is_coarse_mode() {
                                    graph_controller.cursor.set_coarse_mode(true);
                                } else if !show_panel {
                                    focus_zone = FocusZone::Sidebar;
                                }
                            }
                            KeyCode::Char('p') => {
                                if let Some(ref block_group_id) =
                                    explorer_state.selected_block_group_id
                                {
                                    match toggle_path_highlight(
                                        conn,
                                        &graph_controller,
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
                            _ => {
                                graph_controller.handle_key_event(key).ok();
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
                            // Handle annotation file toggle requests
                            if let Some(toggled_id) =
                                explorer_state.annotation_file_toggle_requested.take()
                            {
                                if explorer_state.is_annotation_file_active(&toggled_id) {
                                    if let Some(entry) = explorer.annotation_file_entry(&toggled_id)
                                        && let Some(bg) = current_block_group.as_ref()
                                    {
                                        let query_window =
                                            current_view_coordinate_window(&graph_controller)
                                                .map(expand_query_window);
                                        let node_filter: std::collections::HashSet<HashId> =
                                            block_graph.nodes().map(|node| node.node_id).collect();
                                        let request = AnnotationFileTrackRequest {
                                            conn,
                                            history_ref,
                                            workspace,
                                            collection_name: &current_collection_name,
                                            sample_name: bg.sample_name.as_str(),
                                            block_group_name: Some(&bg.name),
                                            query_window,
                                            node_filter: &node_filter,
                                            entry,
                                        };
                                        match load_annotation_file_track(&request) {
                                            Ok(load) => {
                                                replace_track_overlays(
                                                    &mut overlays,
                                                    &file_track_key(&toggled_id),
                                                    load.track.annotations,
                                                );
                                                annotation_file_index_available
                                                    .insert(toggled_id, load.index_available);
                                                if let Some(window) = load.loaded_window {
                                                    annotation_file_loaded_windows
                                                        .insert(toggled_id, window);
                                                } else {
                                                    annotation_file_loaded_windows
                                                        .remove(&toggled_id);
                                                }
                                            }
                                            Err(err) => {
                                                messages.push_warn(format!("{err}"));
                                                explorer_state
                                                    .deactivate_annotation_file(&toggled_id);
                                                remove_track_overlays(
                                                    &mut overlays,
                                                    &file_track_key(&toggled_id),
                                                );
                                                annotation_file_index_available.remove(&toggled_id);
                                                annotation_file_loaded_windows.remove(&toggled_id);
                                            }
                                        }
                                    }
                                } else {
                                    remove_track_overlays(
                                        &mut overlays,
                                        &file_track_key(&toggled_id),
                                    );
                                    annotation_file_index_available.remove(&toggled_id);
                                    annotation_file_loaded_windows.remove(&toggled_id);
                                }
                            }
                            // Handle annotation group toggle requests
                            if let Some(toggled_group) =
                                explorer_state.annotation_group_toggle_requested.take()
                            {
                                if explorer_state.is_annotation_group_active(&toggled_group) {
                                    if current_block_group.is_some() {
                                        let node_ids = extract_viewport_node_ids(&graph_controller);
                                        let entry = explorer.annotation_group_entry(&toggled_group);
                                        let spans = match entry.map(|entry| {
                                            load_annotations_for_group(
                                                &AnnotationGroupTrackRequest {
                                                    conn,
                                                    history_ref,
                                                    current_block_group: current_block_group
                                                        .as_ref()
                                                        .expect("current block group should exist"),
                                                    entry,
                                                    node_ids: &node_ids,
                                                },
                                            )
                                        }) {
                                            Some(Ok(spans)) => spans,
                                            Some(Err(err)) => {
                                                messages.push_warn(format!(
                                                    "Failed to load annotations for group {}: {err}",
                                                    toggled_group
                                                ));
                                                Vec::new()
                                            }
                                            None => Vec::new(),
                                        };
                                        if spans.is_empty() {
                                            explorer_state
                                                .deactivate_annotation_group(&toggled_group);
                                        } else {
                                            replace_track_overlays(
                                                &mut overlays,
                                                &group_track_key(&toggled_group),
                                                spans,
                                            );
                                        }
                                    }
                                } else {
                                    remove_track_overlays(
                                        &mut overlays,
                                        &group_track_key(&toggled_group),
                                    );
                                }
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
                    focus_zone = FocusZone::Sidebar;
                    explorer.handle_mouse(&mut explorer_state, mouse.column, mouse.row);
                    if let Some(requested_zone) = explorer_state.focus_change_requested {
                        focus_zone = requested_zone;
                        explorer_state.focus_change_requested = None;
                    }
                    // Handle annotation file toggle requests
                    if let Some(toggled_id) = explorer_state.annotation_file_toggle_requested.take()
                    {
                        if explorer_state.is_annotation_file_active(&toggled_id) {
                            if let Some(entry) = explorer.annotation_file_entry(&toggled_id)
                                && let Some(bg) = current_block_group.as_ref()
                            {
                                let query_window =
                                    current_view_coordinate_window(&graph_controller)
                                        .map(expand_query_window);
                                let node_filter: std::collections::HashSet<HashId> =
                                    block_graph.nodes().map(|node| node.node_id).collect();
                                let request = AnnotationFileTrackRequest {
                                    conn,
                                    history_ref,
                                    workspace,
                                    collection_name: &current_collection_name,
                                    sample_name: bg.sample_name.as_str(),
                                    block_group_name: Some(&bg.name),
                                    query_window,
                                    node_filter: &node_filter,
                                    entry,
                                };
                                match load_annotation_file_track(&request) {
                                    Ok(load) => {
                                        replace_track_overlays(
                                            &mut overlays,
                                            &file_track_key(&toggled_id),
                                            load.track.annotations,
                                        );
                                        annotation_file_index_available
                                            .insert(toggled_id, load.index_available);
                                        if let Some(window) = load.loaded_window {
                                            annotation_file_loaded_windows
                                                .insert(toggled_id, window);
                                        } else {
                                            annotation_file_loaded_windows.remove(&toggled_id);
                                        }
                                    }
                                    Err(err) => {
                                        messages.push_warn(format!("{err}"));
                                        explorer_state.deactivate_annotation_file(&toggled_id);
                                        remove_track_overlays(
                                            &mut overlays,
                                            &file_track_key(&toggled_id),
                                        );
                                        annotation_file_index_available.remove(&toggled_id);
                                        annotation_file_loaded_windows.remove(&toggled_id);
                                    }
                                }
                            }
                        } else {
                            remove_track_overlays(&mut overlays, &file_track_key(&toggled_id));
                            annotation_file_index_available.remove(&toggled_id);
                            annotation_file_loaded_windows.remove(&toggled_id);
                        }
                    }
                    // Handle annotation group toggle requests
                    if let Some(toggled_group) =
                        explorer_state.annotation_group_toggle_requested.take()
                    {
                        if explorer_state.is_annotation_group_active(&toggled_group) {
                            if let Some(bg) = current_block_group.as_ref() {
                                let node_ids = extract_viewport_node_ids(&graph_controller);
                                let entry = explorer.annotation_group_entry(&toggled_group);
                                let spans = match entry.map(|entry| {
                                    load_annotations_for_group(&AnnotationGroupTrackRequest {
                                        conn,
                                        history_ref,
                                        current_block_group: bg,
                                        entry,
                                        node_ids: &node_ids,
                                    })
                                }) {
                                    Some(Ok(spans)) => spans,
                                    Some(Err(err)) => {
                                        messages.push_warn(format!(
                                            "Failed to load annotations for group {}: {err}",
                                            toggled_group
                                        ));
                                        Vec::new()
                                    }
                                    None => Vec::new(),
                                };
                                if spans.is_empty() {
                                    explorer_state.deactivate_annotation_group(&toggled_group);
                                } else {
                                    replace_track_overlays(
                                        &mut overlays,
                                        &group_track_key(&toggled_group),
                                        spans,
                                    );
                                }
                            }
                        } else {
                            remove_track_overlays(&mut overlays, &group_track_key(&toggled_group));
                        }
                    }
                }
                event::Event::Mouse(mouse) if focus_zone == FocusZone::Canvas => match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) => {
                        mouse_last_pos = Some((mouse.column, mouse.row));
                        mouse_is_dragging = false;
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        if let Some((lx, ly)) = mouse_last_pos {
                            let dx = mouse.column as i16 - lx as i16;
                            let dy = mouse.row as i16 - ly as i16;
                            graph_controller.move_by_terminal(dx, dy);
                            graph_controller.sync_cursor_to_closest_node();
                            mouse_is_dragging = true;
                        }
                        mouse_last_pos = Some((mouse.column, mouse.row));
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        if !mouse_is_dragging {
                            graph_controller.handle_click(mouse.column, mouse.row);
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
            current_collection_name = selected_collection;
            explorer_state.selected_block_group_id = None;
            last_selected_block_group_id = None;
            current_block_group = None;
            block_graph = get_empty_graph();
            graph_controller = create_gen_graph_controller(block_graph.clone());
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
            explorer_state.retain_annotation_groups(&explorer.data.annotation_groups);
            overlays.clear();
            annotation_file_index_available.clear();
            annotation_file_loaded_windows.clear();
            explorer_state.active_annotation_groups.clear();
            annotation_groups_loaded = false;
            is_loading = false;
            last_refresh = Instant::now();
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
                &current_collection_name,
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
                overlays.retain(|o| match &o.source {
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

        // Reload indexed annotation file tracks when the user scrolls past the loaded window.
        // Annotation group reload piggybacks on the viewport-rebuild signal: when the camera has
        // moved far enough that the cropped-graph node set changes, invalidate and re-load.
        if !is_loading
            && let Some(bg) = current_block_group.as_ref()
            && let Some(visible_window) = current_view_coordinate_window(&graph_controller)
        {
            if graph_controller.detect_motion() {
                annotation_groups_loaded = false;
            }
            let query_window = expand_query_window(visible_window);
            let node_filter: std::collections::HashSet<HashId> =
                block_graph.nodes().map(|node| node.node_id).collect();
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
                    collection_name: &current_collection_name,
                    sample_name: bg.sample_name.as_str(),
                    block_group_name: Some(&bg.name),
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

        // Calculate frame delta for smooth animations
        let now = Instant::now();
        let frame_delta = now.duration_since(last_frame_time);
        last_frame_time = now;

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

            // The panel pops up in the graph area, it does not overlap with the sidebar
            let panel_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Percentage(80), Constraint::Percentage(20)])
                .split(viewer_root_area);
            let panel_area = panel_layout[1];

            let canvas_area = if show_panel {
                panel_layout[0]
            } else {
                viewer_root_area
            };

            // Set viewport bounds to the actual canvas area before updating animations
            graph_controller.viewport_state.viewport_bounds = canvas_area;

            // Update animations with frame delta for smooth camera and cursor animations
            graph_controller.update_animations(frame_delta);

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
            let mut status_message = match focus_zone {
                FocusZone::Canvas => {
                    let tab_dest = if show_panel { "to panel" } else { "to sidebar" };
                    if !graph_controller.is_cursor_visible() {
                        format!("*drag* pan | *click* select | *↑↓←→* show cursor | *tab* {tab_dest}")
                    } else if graph_controller.cursor.is_coarse_mode() {
                        format!("*←→↑↓* navigate by block | *enter* by character | *+/-* zoom | *p* path | *m* messages | *tab* {tab_dest}")
                    } else {
                        format!("*←→↑↓* navigate by character | *enter* details | *+/-* zoom | *p* path | *m* messages | *tab* {tab_dest}")
                    }
                }
                FocusZone::Panel => match panel_mode {
                    PanelMode::Messages => "*c* clear | *esc* close | *tab* to sidebar".to_string(),
                    PanelMode::Details => "*esc* close | *tab* to sidebar".to_string(),
                },
                FocusZone::Sidebar => CollectionExplorer::get_status_line(),
            };
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
                graph_controller.viewport_state.focus();

                // Re-register overlay highlights before rendering. This reruns every frame
                // because `overlays` can change between frames (file/group toggles,
                // scroll-triggered reloads).
                reapply_overlays(&mut graph_controller, &mut overlays, &mut annotation_colors);

                let canvas_style = Style::default().bg(current_theme()[0x00]);
                let widget = create_gen_graph_widget(conn)
                    .detail_level(graph_controller.get_detail_level())
                    .style(canvas_style)
                    .cursor();
                frame.render_stateful_widget(widget, canvas_area, &mut graph_controller);

                // Draw floating labels after the graph, then a single hint if any were hidden.
                let detail_level = graph_controller.get_detail_level();
                let any_hidden = draw_annotation_labels(
                    frame.buffer_mut(),
                    canvas_area,
                    &graph_controller,
                    &overlays,
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
                        canvas_area.x,
                        canvas_area.bottom().saturating_sub(1),
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
                        use petgraph::visit::NodeIndexable;

                        let mut lines = vec![];

                        if let Some(node_idx) = graph_controller.cursor.node_idx() {
                            let graph_node = <&GenGraph as NodeIndexable>::from_index(
                                &graph_controller.graph(),
                                node_idx.index(),
                            );
                            let node_id_short =
                                graph_node.node_id.to_string().chars().take(12).collect::<String>();
                            let block_spec = if graph_controller.get_detail_level()
                                == VisualDetail::Full
                            {
                                let (frac_x, _) = graph_controller.cursor.fractional_pos();
                                let block_width =
                                    graph_node.sequence_end - graph_node.sequence_start;
                                let pos_on_node = graph_node.sequence_start
                                    + (frac_x * block_width as f64).round() as i64;
                                format!(
                                    "{}:{}-{} (cursor at {})",
                                    node_id_short,
                                    graph_node.sequence_start,
                                    graph_node.sequence_end,
                                    pos_on_node
                                )
                            } else {
                                format!(
                                    "{}:{}-{}",
                                    node_id_short,
                                    graph_node.sequence_start,
                                    graph_node.sequence_end
                                )
                            };
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

        // After the first draw the viewport is populated. Load (or reload) annotation groups
        // using the viewport node IDs so only on-screen segments are fetched.
        let mut annotation_groups_loaded_after_draw = false;
        if !annotation_groups_loaded && let Some(block_group) = current_block_group.as_ref() {
            let node_ids = extract_viewport_node_ids(&graph_controller);
            if !node_ids.is_empty() {
                overlays.retain(
                    |o| !matches!(&o.source, OverlaySource::Track(k) if k.starts_with("group:")),
                );
                explorer_state.active_annotation_groups.clear();
                load_annotation_groups_for_viewport(
                    conn,
                    history_ref,
                    block_group,
                    &node_ids,
                    &mut explorer_state,
                    &mut overlays,
                    &mut messages,
                );
                annotation_groups_loaded = true;
                annotation_groups_loaded_after_draw = true;
            }
        }

        // Update the graph controller if a new block group was selected.
        // This runs after terminal.draw() so the loading indicator is visible
        // for the full duration of the blocking DB work.
        if is_loading && let Some(ref new_block_group_id) = explorer_state.selected_block_group_id {
            // Create a new graph for the selected block group
            block_graph =
                gen_graph::models::load_block_group_graph(conn, new_block_group_id, history_ref)?;
            // Update the graph controller
            graph_controller = create_gen_graph_controller(block_graph.clone());
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
                &current_collection_name,
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
            if let Some(bg) = current_block_group.as_ref() {
                let node_filter: std::collections::HashSet<HashId> =
                    block_graph.nodes().map(|node| node.node_id).collect();
                let query_window =
                    current_view_coordinate_window(&graph_controller).map(expand_query_window);
                for entry in explorer.data.annotation_files.iter() {
                    let id = entry.file_addition.id;
                    if !explorer_state.is_annotation_file_active(&id) {
                        continue;
                    }
                    let request = AnnotationFileTrackRequest {
                        conn,
                        history_ref,
                        workspace,
                        collection_name: &current_collection_name,
                        sample_name: bg.sample_name.as_str(),
                        block_group_name: Some(&bg.name),
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
            continue;
        }

        // The overlays were populated after the frame was rendered. Draw them immediately
        // instead of waiting for the next keyboard or mouse event to wake the idle viewer.
        if annotation_groups_loaded_after_draw {
            continue;
        }

        // If an animation is running, wake up after tick_rate to advance it.
        // If the display is idle, block indefinitely — the next input event will wake us.
        let wait = if graph_controller.is_animating() {
            tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::from_secs(3600)
        };
        let _ = crossterm::event::poll(wait);

        // Update tick
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    Ok(())
}
