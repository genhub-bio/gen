use std::{
    error::Error,
    time::{Duration, Instant},
};

use crossterm::event::{self, KeyCode, KeyEventKind, MouseButton, MouseEventKind};
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection, node::Node, traits::Query};
use gen_tui::{LineStyle, graph_controller::GraphController, plotter::PathStyle};
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
    theme::get_theme_color,
    views::{
        annotation_track::AnnotationTrack,
        annotations::{
            AnnotationFileTrackRequest, load_annotation_file_track, load_annotations_for_group,
        },
        collection::{CollectionExplorer, CollectionExplorerState, FocusZone},
        gen_graph_widget::{
            GenGraphNodeSizer, create_gen_graph_controller, create_gen_graph_widget,
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
    use gen_graph::project_path;
    use gen_models::path::Path;

    // Query the database for the most recent path for this block group
    let path = Path::get(
        conn,
        "SELECT * FROM paths WHERE block_group_id = ?1 ORDER BY created_on DESC LIMIT 1",
        rusqlite::params![block_group_id],
    )
    .map_err(|e| format!("Failed to query path: {}", e))?;

    // Get the path blocks from the database
    let path_blocks = path.blocks(conn);

    // Project the path blocks onto the current graph state
    let projected_path = project_path(graph, &path_blocks);

    // Filter out terminal nodes (start and end) and convert to GraphNodes
    let path_nodes: Vec<gen_graph::GraphNode> = projected_path
        .iter()
        .filter_map(|(node, _)| {
            // Filter out terminal nodes
            if node.node_id != PATH_START_NODE_ID && node.node_id != PATH_END_NODE_ID {
                Some(*node)
            } else {
                None
            }
        })
        .collect();

    if path_nodes.is_empty() {
        return Err("Path nodes not found in current graph state".to_string());
    }

    Ok(path_nodes)
}

/// Toggle path highlighting for a block group
fn toggle_path_highlight(
    conn: &GraphConnection,
    controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    block_group_id: &gen_core::HashId,
    color: ratatui::style::Color,
) -> Result<bool, String> {
    let style = PathStyle::new(color)
        .with_line_style(LineStyle::Bold)
        .with_merge_glyphs(true);
    // Check if highlighting is already active for this style
    if controller.has_highlight(&style) {
        controller.clear_highlight(&style);
        Ok(false)
    } else {
        // Get the path nodes for this block group
        let path_nodes = get_block_group_path_nodes(conn, block_group_id, controller.graph())?;

        // Set the path highlight using GraphNodes directly
        controller.set_path_highlight(style, path_nodes);
        Ok(true)
    }
}

fn visible_ranges_by_node(
    block_graph: &GenGraph,
) -> std::collections::HashMap<HashId, Vec<(i64, i64)>> {
    let mut ranges_by_node: std::collections::HashMap<HashId, Vec<(i64, i64)>> =
        std::collections::HashMap::new();
    for node in block_graph.nodes() {
        if node.sequence_end <= node.sequence_start {
            continue;
        }
        ranges_by_node
            .entry(node.node_id)
            .or_default()
            .push((node.sequence_start, node.sequence_end));
    }
    ranges_by_node
}

/// Compute the coordinate window (min sequence start, max sequence end) of visible blocks
/// in the current viewport, using the graph controller's viewport graph.
fn current_view_coordinate_window(
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

fn expand_query_window(window: (i64, i64)) -> (i64, i64) {
    let span = (window.1 - window.0).max(1);
    (window.0.saturating_sub(span), window.1.saturating_add(span))
}

pub fn view_block_group(
    conn: &GraphConnection,
    op_conn: &gen_models::db::OperationsConnection,
    workspace: &gen_core::config::Workspace,
    name: Option<String>,
    sample_name: Option<String>,
    collection_name: &str,
    position: Option<String>, // Node ID and offset
) -> Result<(), Box<dyn Error>> {
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

    // Create explorer and its state that persists across frames
    let mut explorer =
        CollectionExplorer::new(conn, op_conn, sample_name.as_deref(), collection_name);
    let mut explorer_state = CollectionExplorerState::new();
    if let Some(ref s) = sample_name {
        explorer_state.set_sample_expanded(s, true);
    }

    let mut block_graph;
    let mut block_group_id: Option<gen_core::HashId> = None;
    let mut focus_zone = FocusZone::Sidebar;

    if let (Some(name), Some(sample_name)) = (name, sample_name.as_ref()) {
        let block_group = BlockGroup::get(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2 AND name = ?3",
            params![collection_name, sample_name, name],
        );

        if block_group.is_err() {
            panic!(
                "No block group found with name {:?} and sample {:?} in collection {} ",
                name,
                sample_name.clone(),
                collection_name
            );
        }

        let block_group = block_group.unwrap();
        block_group_id = Some(block_group.id);
        block_graph = BlockGroup::get_graph(conn, &block_group.id);
        explorer_state.selected_block_group_id = Some(block_group.id);
        focus_zone = FocusZone::Canvas;
    } else {
        block_graph = get_empty_graph();
    }

    bar.finish();

    let mut messages = crate::views::messages::MessageBuffer::new(MESSAGE_BUFFER_LIMIT);
    let mut annotation_file_tracks: std::collections::HashMap<HashId, AnnotationTrack> =
        std::collections::HashMap::new();
    let mut annotation_file_index_available: std::collections::HashMap<HashId, bool> =
        std::collections::HashMap::new();
    let mut annotation_file_loaded_windows: std::collections::HashMap<HashId, (i64, i64)> =
        std::collections::HashMap::new();
    let mut annotation_group_tracks: std::collections::HashMap<String, AnnotationTrack> =
        std::collections::HashMap::new();
    let mut current_block_group =
        block_group_id.map(|bg_id| match BlockGroup::get_by_id(conn, &bg_id) {
            Ok(bg) => bg,
            Err(err) => {
                // TODO: Handle these with messages instead of panic'ing
                panic!("Failed to load block group {bg_id}: {err}");
            }
        });

    // Create the graph controller and initial graph
    let bar = progress_bar.add(get_time_elapsed_bar());
    let _ = progress_bar.println("Pre-computing layout in chunks");

    let mut graph_controller = create_gen_graph_controller(block_graph.clone());

    // TODO: Handle origin positioning - not directly supported in new widget yet
    if origin.is_some() {
        warn!("Origin positioning not yet supported in GenGraphWidget");
    }

    bar.finish();

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
                                        &mut graph_controller,
                                        block_group_id,
                                        Color::Red,
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
                            KeyCode::Char('c') => {
                                if panel_mode == PanelMode::Messages {
                                    messages.clear();
                                }
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
                                            workspace,
                                            collection_name,
                                            sample_name: bg.sample_name.as_str(),
                                            block_group_name: Some(&bg.name),
                                            query_window,
                                            node_filter: &node_filter,
                                            entry,
                                        };
                                        match load_annotation_file_track(&request) {
                                            Ok(load) => {
                                                annotation_file_tracks
                                                    .insert(toggled_id, load.track);
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
                                                annotation_file_tracks.remove(&toggled_id);
                                                annotation_file_index_available.remove(&toggled_id);
                                                annotation_file_loaded_windows.remove(&toggled_id);
                                            }
                                        }
                                    }
                                } else {
                                    annotation_file_tracks.remove(&toggled_id);
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
                                        let visible_node_ranges =
                                            visible_ranges_by_node(&block_graph);
                                        let spans = match load_annotations_for_group(
                                            conn,
                                            &toggled_group,
                                            &visible_node_ranges,
                                        ) {
                                            Ok(spans) => spans,
                                            Err(err) => {
                                                messages.push_warn(format!(
                                                    "Failed to load annotations for group {}: {err}",
                                                    toggled_group
                                                ));
                                                Vec::new()
                                            }
                                        };
                                        if spans.is_empty() {
                                            explorer_state
                                                .deactivate_annotation_group(&toggled_group);
                                        } else {
                                            annotation_group_tracks.insert(
                                                toggled_group.clone(),
                                                AnnotationTrack::new(toggled_group, spans),
                                            );
                                        }
                                    }
                                } else {
                                    annotation_group_tracks.remove(&toggled_group);
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
                    MouseEventKind::ScrollUp => {
                        graph_controller.zoom_in();
                    }
                    MouseEventKind::ScrollDown => {
                        graph_controller.zoom_out();
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
            if explorer.refresh(conn, op_conn, selected_sample, collection_name) {
                explorer.force_reload(&mut explorer_state);
                explorer_state.retain_annotation_files(&explorer.data.annotation_files);
                explorer_state.retain_annotation_groups(&explorer.data.annotation_groups);
                annotation_file_tracks.retain(|id, _| explorer_state.is_annotation_file_active(id));
                annotation_file_index_available
                    .retain(|id, _| explorer_state.is_annotation_file_active(id));
                annotation_file_loaded_windows
                    .retain(|id, _| explorer_state.is_annotation_file_active(id));
                annotation_group_tracks
                    .retain(|name, _| explorer_state.is_annotation_group_active(name));
            }
            last_refresh = Instant::now();
        }

        // Reload indexed annotation file tracks when the user scrolls past the loaded window
        if !is_loading
            && let Some(bg) = current_block_group.as_ref()
            && let Some(visible_window) = current_view_coordinate_window(&graph_controller)
        {
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
                    workspace,
                    collection_name,
                    sample_name: bg.sample_name.as_str(),
                    block_group_name: Some(&bg.name),
                    query_window: Some(query_window),
                    node_filter: &node_filter,
                    entry,
                };
                match load_annotation_file_track(&request) {
                    Ok(load) => {
                        annotation_file_tracks.insert(id, load.track);
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
                        annotation_file_tracks.remove(&id);
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

            // Collect annotation tracks to display
            let mut track_panels: Vec<(&AnnotationTrack, u16)> = Vec::new();
            for track in annotation_file_tracks.values() {
                let height = crate::views::annotation_track::annotation_panel_height(track, 10);
                if height > 0 {
                    track_panels.push((track, height));
                }
            }
            for track in annotation_group_tracks.values() {
                let height = crate::views::annotation_track::annotation_panel_height(track, 10);
                if height > 0 {
                    track_panels.push((track, height));
                }
            }

            let total_annotation_height: u16 = track_panels.iter().map(|(_, h)| *h).sum();

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

            // Split viewer area between graph and annotation panels
            let viewer_constraints = if total_annotation_height > 0 {
                vec![
                    Constraint::Min(10),                         // Graph area (minimum 10 lines)
                    Constraint::Length(total_annotation_height), // Annotation panels
                ]
            } else {
                vec![Constraint::Percentage(100)] // Just the graph
            };

            let viewer_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(viewer_constraints)
                .split(viewer_root_area);

            let graph_root_area = viewer_layout[0];
            let annotation_area = if viewer_layout.len() > 1 {
                Some(viewer_layout[1])
            } else {
                None
            };

            // The panel pops up in the graph area, it does not overlap with the sidebar or annotations
            let panel_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Percentage(80), Constraint::Percentage(20)])
                .split(graph_root_area);
            let panel_area = panel_layout[1];

            let canvas_area = if show_panel {
                panel_layout[0]
            } else {
                graph_root_area
            };

            // Set viewport bounds to the actual canvas area before updating animations
            graph_controller.viewport_state.viewport_bounds = canvas_area;

            // Update animations with frame delta for smooth camera and cursor animations
            graph_controller.update_animations(frame_delta);

            // Sidebar
            explorer_state.has_focus = focus_zone == FocusZone::Sidebar;
            if show_sidebar {
                let sidebar_block = Block::default().padding(Padding::new(0, 0, 1, 1)).style(
                    Style::default()
                        .bg(get_theme_color("sidebar").unwrap())
                        .fg(get_theme_color("text").unwrap()),
                );
                let sidebar_content_area = sidebar_block.inner(sidebar_area);

                frame.render_widget(sidebar_block.clone(), sidebar_area);
                frame.render_stateful_widget(&explorer, sidebar_content_area, &mut explorer_state);

                // Draw the vertical separator line at the right edge of the sidebar
                let line_char = "▕";
                let line_style = Style::default().fg(get_theme_color("separator").unwrap());
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
                let message_bar = Paragraph::new(message_text).style(
                    Style::default()
                        .fg(get_theme_color("warning").unwrap_or(Color::Yellow))
                        .bg(get_theme_color("status_bar").unwrap_or(Color::Black)),
                );
                frame.render_widget(message_bar, area);
            }

            // Status bar
            let mut status_message = match focus_zone {
                FocusZone::Canvas => {
                    if !graph_controller.is_cursor_visible() {
                        "*drag*: pan | *click node*: select | *scroll*: zoom | *arrows*: keyboard nav"
                            .to_string()
                    } else if graph_controller.cursor.is_coarse_mode() {
                        "*←→↑↓* move | *enter* fine nav | *+/-* zoom | *p* path | *esc* sidebar"
                            .to_string()
                    } else {
                        "*←→↑↓* move | *enter* details | *+/-* zoom | *p* path | *esc* coarse nav"
                            .to_string()
                    }
                }
                FocusZone::Panel => match panel_mode {
                    PanelMode::Messages => "*c* clear | *esc* close panel".to_string(),
                    PanelMode::Details => "*esc* close panel".to_string(),
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
                        .fg(get_theme_color("text").unwrap())
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
                                Style::default().fg(get_theme_color("highlight").unwrap()),
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
                let canvas_style = Style::default().bg(get_theme_color("canvas").unwrap());
                let widget = create_gen_graph_widget(conn)
                    .detail_level(graph_controller.get_detail_level())
                    .style(canvas_style)
                    .cursor();
                frame.render_stateful_widget(widget, canvas_area, &mut graph_controller);

                // Render annotation track panels below the graph
                if let Some(annotation_area) = annotation_area {
                    let mut current_y = annotation_area.y;
                    for (track, height) in track_panels.iter() {
                        if current_y + height > annotation_area.y + annotation_area.height {
                            break; // No more room
                        }
                        let track_area = Rect {
                            x: annotation_area.x,
                            y: current_y,
                            width: annotation_area.width,
                            height: *height,
                        };
                        crate::views::annotation_track::draw_annotations_panel(
                            frame,
                            track_area,
                            track,
                            &graph_controller,
                        );
                        current_y = current_y.saturating_add(*height);
                    }
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
                    .style(
                        Style::default()
                            .bg(get_theme_color("panel").unwrap())
                            .fg(get_theme_color("text").unwrap()),
                    )
                    .border_style(if focus_zone == FocusZone::Panel {
                        Style::default()
                            .fg(get_theme_color("highlight").unwrap())
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(get_theme_color("text").unwrap())
                    });

                let panel_text = match panel_mode {
                    PanelMode::Details => {
                        use gen_tui::layout::VisualDetail;
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
                                    .fg(get_theme_color("text_muted").unwrap_or(Color::DarkGray))
                                    .add_modifier(Modifier::ITALIC),
                            )));
                        }

                        lines
                    }
                    PanelMode::Messages => {
                        if messages.is_empty() {
                            vec![Line::from(vec![Span::styled(
                                "No messages",
                                Style::default().fg(get_theme_color("text_muted").unwrap()),
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

        // Update the graph controller if a new block group was selected.
        // This runs after terminal.draw() so the loading indicator is visible
        // for the full duration of the blocking DB work.
        if is_loading && let Some(ref new_block_group_id) = explorer_state.selected_block_group_id {
            // Create a new graph for the selected block group
            block_graph = BlockGroup::get_graph(conn, new_block_group_id);
            // Update the graph controller
            graph_controller = create_gen_graph_controller(block_graph.clone());
            let block_group = match BlockGroup::get_by_id(conn, new_block_group_id) {
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
            if explorer.refresh(conn, op_conn, selected_sample, collection_name) {
                explorer.force_reload(&mut explorer_state);
                explorer_state.retain_annotation_files(&explorer.data.annotation_files);
                explorer_state.retain_annotation_groups(&explorer.data.annotation_groups);
            }
            annotation_file_tracks.clear();
            annotation_file_index_available.clear();
            annotation_file_loaded_windows.clear();
            annotation_group_tracks.clear();
            if let Some(bg) = current_block_group.as_ref() {
                let node_filter: std::collections::HashSet<HashId> =
                    block_graph.nodes().map(|node| node.node_id).collect();
                let visible_node_ranges = visible_ranges_by_node(&block_graph);
                let query_window =
                    current_view_coordinate_window(&graph_controller).map(expand_query_window);
                for entry in explorer.data.annotation_groups.iter() {
                    if explorer_state.is_annotation_group_active(&entry.name) {
                        let spans = match load_annotations_for_group(
                            conn,
                            &entry.name,
                            &visible_node_ranges,
                        ) {
                            Ok(spans) => spans,
                            Err(err) => {
                                messages.push_warn(format!(
                                    "Failed to load annotations for group {}: {err}",
                                    entry.name
                                ));
                                Vec::new()
                            }
                        };
                        if spans.is_empty() {
                            continue;
                        }
                        annotation_group_tracks.insert(
                            entry.name.clone(),
                            AnnotationTrack::new(entry.name.clone(), spans),
                        );
                    }
                }
                for entry in explorer.data.annotation_files.iter() {
                    let id = entry.file_addition.id;
                    if !explorer_state.is_annotation_file_active(&id) {
                        continue;
                    }
                    let request = AnnotationFileTrackRequest {
                        conn,
                        workspace,
                        collection_name,
                        sample_name: bg.sample_name.as_str(),
                        block_group_name: Some(&bg.name),
                        query_window,
                        node_filter: &node_filter,
                        entry,
                    };
                    match load_annotation_file_track(&request) {
                        Ok(load) => {
                            annotation_file_tracks.insert(id, load.track);
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

    crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture).ok();
    Ok(())
}
