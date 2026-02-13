use std::{
    collections::{HashMap, HashSet},
    error::Error,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_core::{HashId, PATH_START_NODE_ID, Workspace, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode, connect_all_boundary_edges};
use gen_models::{
    block_group::BlockGroup,
    db::{GraphConnection, OperationsConnection},
    node::Node,
    path::Path,
    traits::Query,
};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Padding, Paragraph, Wrap},
};
use rusqlite::params;

use crate::{
    config::get_theme_color,
    progress_bar::{get_handler, get_time_elapsed_bar},
    views::{
        annotation_track::AnnotationTrack,
        annotations::{
            AnnotationFileTrackRequest, load_annotation_file_track, load_annotations_for_group,
        },
        block_group_viewer::{PlotParameters, Viewer},
        collection::{CollectionExplorer, CollectionExplorerState, FocusZone},
        messages::MessageBuffer,
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
        block_id: -1,
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    });
    g
}

/// Parses a string with markdown-like asterisk syntax for highlighting.
/// Segments surrounded by '*' are styled with `highlight_style`.
/// Other segments are styled with `default_style`.
fn style_text(text: &str, default_style: Style, highlight_style: Style) -> Line<'_> {
    let mut spans = Vec::new();
    let mut is_highlighted = false;
    for part in text.split('*') {
        if !part.is_empty() {
            spans.push(Span::styled(
                part,
                if is_highlighted {
                    highlight_style
                } else {
                    default_style
                },
            ));
        }
        is_highlighted = !is_highlighted;
    }
    Line::from(spans)
}

fn current_view_coordinate_window(viewer: &Viewer<'_>) -> Option<(i64, i64)> {
    let mut start = i64::MAX;
    let mut end = i64::MIN;

    for &block in viewer.scaled_layout.labels.keys() {
        if is_start_node(block.node_id) || is_end_node(block.node_id) {
            continue;
        }
        if !viewer.is_block_visible(block) {
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

fn visible_ranges_by_node(block_graph: &GenGraph) -> HashMap<HashId, Vec<(i64, i64)>> {
    let mut ranges_by_node: HashMap<HashId, Vec<(i64, i64)>> = HashMap::new();
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

pub fn view_block_group(
    conn: &GraphConnection,
    op_conn: &OperationsConnection,
    workspace: &Workspace,
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
            panic!("Invalid position: {position_str}");
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
        explorer_state.toggle_sample(s);
    }

    let mut block_graph;
    let mut block_group_id: Option<HashId> = None;
    let mut focus_zone = FocusZone::Sidebar;

    if let Some(name) = name {
        // Get the block group for two cases: with and without a sample
        let block_group = if let Some(ref sample_name) = sample_name {
            BlockGroup::get(
                conn,
                "select * from block_groups where collection_name = ?1 AND sample_name = ?2 AND name = ?3",
                params![collection_name, sample_name, name],
            )
        } else {
            // modified version:
            BlockGroup::get(
                conn,
                "select * from block_groups where collection_name = ?1 AND sample_name is null AND name = ?2",
                params![collection_name, name],
            )
        };

        if block_group.is_err() {
            panic!(
                "No block group found with name {:?} and sample {:?} in collection {} ",
                name,
                sample_name.clone().unwrap_or_else(|| "null".to_string()),
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

    connect_all_boundary_edges(&mut block_graph);

    bar.finish();

    // Create the viewer and the initial graph
    let bar = progress_bar.add(get_time_elapsed_bar());
    let _ = progress_bar.println("Pre-computing layout in chunks");

    let mut viewer = if let Some(origin) = origin {
        Viewer::with_origin(&block_graph, conn, PlotParameters::default(), origin)
    } else {
        Viewer::new(&block_graph, conn, PlotParameters::default())
    };

    bar.finish();

    let mut messages = MessageBuffer::new(MESSAGE_BUFFER_LIMIT);
    let mut annotation_file_tracks: HashMap<HashId, AnnotationTrack> = HashMap::new();
    let mut annotation_file_index_available: HashMap<HashId, bool> = HashMap::new();
    let mut annotation_file_loaded_windows: HashMap<HashId, (i64, i64)> = HashMap::new();
    let mut annotation_group_tracks: HashMap<String, AnnotationTrack> = HashMap::new();
    let mut current_block_group = block_group_id.map(|bg_id| BlockGroup::get_by_id(conn, &bg_id));

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = ratatui::init();

    // Basic event loop
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();
    let mut show_panel = false;
    let mut panel_mode = PanelMode::Details;
    let mut message_scroll: u16 = 0;
    let show_sidebar = true;
    let mut tui_layout_change = false;

    // Track the last selected block group to detect changes
    let mut last_selected_block_group_id = block_group_id;
    // Track if we're loading a new block group
    let mut is_loading = false;
    let mut last_refresh = Instant::now();
    loop {
        // Refresh explorer data and force reload on change
        // I do this every REFRESH_INTERVAL seconds.
        if last_refresh.elapsed() >= Duration::from_secs(REFRESH_INTERVAL) {
            let selected_sample = current_block_group
                .as_ref()
                .and_then(|bg| bg.sample_name.as_deref());
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

        // Trigger reload if selection changed to a new block group
        if explorer_state.selected_block_group_id != last_selected_block_group_id {
            is_loading = true;
            last_selected_block_group_id = explorer_state.selected_block_group_id;
        }

        if !is_loading
            && let Some(bg) = current_block_group.as_ref()
            && let Some(visible_window) = current_view_coordinate_window(&viewer)
        {
            let query_window = expand_query_window(visible_window);
            let node_filter: HashSet<HashId> =
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
                    sample_name: bg.sample_name.as_deref(),
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

        // Draw the UI
        terminal.draw(|frame| {
            let status_bar_height: u16 = 1;
            let show_message_bar = !messages.is_empty();

            // The outer layout is a vertical split between the status bar and everything else
            let mut outer_constraints = vec![ratatui::layout::Constraint::Min(1)];
            if show_message_bar {
                outer_constraints.push(ratatui::layout::Constraint::Length(1));
            }
            outer_constraints.push(ratatui::layout::Constraint::Length(status_bar_height));

            let outer_layout = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints(outer_constraints)
                .split(frame.area());
            let status_bar_area = *outer_layout.last().unwrap();
            let message_bar_area = if show_message_bar {
                Some(outer_layout[1])
            } else {
                None
            };

            // The sidebar is a horizontal split of the area above the status bar
            let sidebar_layout = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints(vec![Constraint::Percentage(20), Constraint::Percentage(80)])
                .split(outer_layout[0]);
            let sidebar_area = sidebar_layout[0];

            let viewer_root_area = sidebar_layout[1];

            // The panel pops up in the viewer area, it does not overlap with the sidebar
            let panel_layout = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints(vec![Constraint::Percentage(80), Constraint::Percentage(20)])
                .split(viewer_root_area);
            let panel_area = panel_layout[1];

            let canvas_area = if show_panel {
                panel_layout[0]
            } else {
                viewer_root_area
            };

            let mut annotation_tracks: Vec<&AnnotationTrack> = Vec::new();
            for entry in explorer.data.annotation_groups.iter() {
                if let Some(track) = annotation_group_tracks.get(&entry.name) {
                    annotation_tracks.push(track);
                }
            }
            for entry in explorer.data.annotation_files.iter() {
                if let Some(track) = annotation_file_tracks.get(&entry.file_addition.id) {
                    annotation_tracks.push(track);
                }
            }

            let min_graph_height = 1;
            let mut remaining_height = canvas_area.height;
            let mut track_panels: Vec<(&AnnotationTrack, u16)> = Vec::new();
            for track in annotation_tracks {
                if remaining_height <= min_graph_height {
                    break;
                }
                let max_for_track = remaining_height - min_graph_height;
                let height = viewer.annotation_panel_height(track, max_for_track);
                if height == 0 {
                    continue;
                }
                track_panels.push((track, height));
                remaining_height = remaining_height.saturating_sub(height);
            }

            let graph_area = Rect {
                x: canvas_area.x,
                y: canvas_area.y,
                width: canvas_area.width,
                height: remaining_height.max(1),
            };

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

            // Status bar
            let mut status_message = match focus_zone {
                FocusZone::Canvas => {
                    Viewer::get_status_line()
                        + " | *p* toggle current path | *m* messages | *esc* back to sidebar"
                }
                FocusZone::Panel => match panel_mode {
                    PanelMode::Messages => {
                        "*↑/↓* scroll messages | *c* clear | *esc* close panel".to_string()
                    }
                    PanelMode::Details => "*esc* close panel".to_string(),
                },
                FocusZone::Sidebar => CollectionExplorer::get_status_line(),
            };
            status_message.push_str(" | *q* quit"); // Universal controls

            let status_bar_contents = format!(
                "{status_message:^width$}",
                width = status_bar_area.width as usize
            );

            // Style the status bar text
            let status_line = style_text(
                &status_bar_contents,
                Style::default().fg(get_theme_color("text_muted").unwrap()), // default color
                Style::default().fg(get_theme_color("highlight").unwrap()),  // highlight color
            );

            let status_bar = Paragraph::new(status_line)
                .style(Style::default().bg(get_theme_color("statusbar").unwrap()));

            frame.render_widget(status_bar, status_bar_area);

            // Message bar (latest warning)
            if let Some(area) = message_bar_area
                && let Some(latest) = messages.latest()
            {
                let extra = messages.len().saturating_sub(1);
                let suffix = if extra > 0 {
                    format!(" (+{extra})")
                } else {
                    String::new()
                };
                let message_line = format!("WARN: {latest}{suffix}");
                let message_bar = Paragraph::new(message_line).style(
                    Style::default()
                        .fg(get_theme_color("error").unwrap())
                        .bg(get_theme_color("statusbar").unwrap()),
                );
                frame.render_widget(message_bar, area);
            }

            // Canvas area
            if is_loading {
                // Draw loading message in canvas area
                let loading_text = Text::styled(
                    "Loading...",
                    Style::default()
                        .fg(get_theme_color("text").unwrap())
                        .add_modifier(Modifier::BOLD),
                );
                let loading_para =
                    Paragraph::new(loading_text).alignment(ratatui::layout::Alignment::Center);

                // Center the loading message vertically in the canvas area
                let loading_area = ratatui::layout::Layout::default()
                    .direction(ratatui::layout::Direction::Vertical)
                    .constraints([
                        ratatui::layout::Constraint::Percentage(45),
                        ratatui::layout::Constraint::Length(1),
                        ratatui::layout::Constraint::Percentage(45),
                    ])
                    .split(canvas_area)[1];

                frame.render_widget(Clear, canvas_area); // Clear the canvas area first
                frame.render_widget(loading_para, loading_area);
            } else {
                // Ask the viewer to paint the canvas
                viewer.has_focus = focus_zone == FocusZone::Canvas;
                viewer.draw(frame, graph_area);
                if !track_panels.is_empty() {
                    let mut current_y = graph_area.y + graph_area.height;
                    for (track, height) in track_panels.iter() {
                        let panel_area = Rect {
                            x: canvas_area.x,
                            y: current_y,
                            width: canvas_area.width,
                            height: *height,
                        };
                        viewer.draw_annotations_panel(frame, panel_area, track);
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
                        if let Some(selected_block) = viewer.state.selected_block {
                            vec![
                                Line::from(vec![
                                    Span::styled(
                                        "Block ID: ",
                                        Style::default().add_modifier(Modifier::BOLD),
                                    ),
                                    Span::raw(selected_block.block_id.to_string()),
                                ]),
                                Line::from(vec![
                                    Span::styled(
                                        "Node ID: ",
                                        Style::default().add_modifier(Modifier::BOLD),
                                    ),
                                    Span::raw(selected_block.node_id.to_string()),
                                ]),
                                Line::from(vec![
                                    Span::styled(
                                        "Start: ",
                                        Style::default().add_modifier(Modifier::BOLD),
                                    ),
                                    Span::raw(selected_block.sequence_start.to_string()),
                                ]),
                                Line::from(vec![
                                    Span::styled(
                                        "End: ",
                                        Style::default().add_modifier(Modifier::BOLD),
                                    ),
                                    Span::raw(selected_block.sequence_end.to_string()),
                                ]),
                            ]
                        } else {
                            vec![Line::from(vec![Span::styled(
                                "No block selected",
                                Style::default()
                                    .fg(get_theme_color("text").unwrap())
                                    .add_modifier(Modifier::BOLD),
                            )])]
                        }
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

                let mut panel_content = Paragraph::new(panel_text)
                    .wrap(Wrap { trim: true })
                    .alignment(ratatui::layout::Alignment::Left)
                    .block(panel_block);
                if panel_mode == PanelMode::Messages {
                    panel_content = panel_content.scroll((message_scroll, 0));
                }

                // Clear the panel area if we just changed the layout
                if tui_layout_change {
                    frame.render_widget(Clear, panel_area);
                }
                frame.render_widget(panel_content, panel_area);

                // Reset the layout change flag
                tui_layout_change = false;
            }
        })?;

        // After drawing, update the viewer if needed
        if is_loading && let Some(ref new_block_group_id) = explorer_state.selected_block_group_id {
            // Create a new graph for the selected block group
            block_graph = BlockGroup::get_graph(conn, new_block_group_id);
            messages.push_warn(format!("{block_graph:?}"));
            connect_all_boundary_edges(&mut block_graph);
            // Update the viewer
            viewer = Viewer::new(&block_graph, conn, PlotParameters::default());
            current_block_group = Some(BlockGroup::get_by_id(conn, new_block_group_id));
            let selected_sample = current_block_group
                .as_ref()
                .and_then(|bg| bg.sample_name.as_deref());
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
                let node_filter: HashSet<HashId> =
                    block_graph.nodes().map(|node| node.node_id).collect();
                let visible_node_ranges = visible_ranges_by_node(&block_graph);
                let query_window = current_view_coordinate_window(&viewer).map(expand_query_window);
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
                    if explorer_state.is_annotation_file_active(&id) {
                        let request = AnnotationFileTrackRequest {
                            conn,
                            workspace,
                            collection_name,
                            sample_name: bg.sample_name.as_deref(),
                            block_group_name: Some(&bg.name),
                            query_window,
                            node_filter: &node_filter,
                            entry,
                        };
                        match load_annotation_file_track(&request) {
                            Ok(load) => {
                                annotation_file_tracks.insert(id, load.track);
                                annotation_file_index_available.insert(id, load.index_available);
                                if let Some(window) = load.loaded_window {
                                    annotation_file_loaded_windows.insert(id, window);
                                }
                            }
                            Err(err) => {
                                messages.push_warn(format!("{err}"));
                                explorer_state.deactivate_annotation_file(&id);
                                annotation_file_index_available.remove(&id);
                                annotation_file_loaded_windows.remove(&id);
                            }
                        }
                    }
                }
            }
            viewer.state.selected_block = None;
            is_loading = false;
        }

        // Handle input
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));
        if crossterm::event::poll(timeout)?
            && let event::Event::Key(key) = event::read()?
        {
            if viewer.state.show_splash_screen {
                viewer.state.show_splash_screen = false;
            }
            if key.kind == KeyEventKind::Press {
                // Global handlers
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('m') => {
                        if show_panel && panel_mode == PanelMode::Messages {
                            show_panel = false;
                            focus_zone = FocusZone::Canvas;
                        } else {
                            show_panel = true;
                            panel_mode = PanelMode::Messages;
                            focus_zone = FocusZone::Panel;
                            message_scroll = 0;
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
                            FocusZone::Panel => FocusZone::Sidebar,
                            FocusZone::Sidebar => FocusZone::Canvas,
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
                            if viewer.state.selected_block.is_some() {
                                show_panel = true;
                                panel_mode = PanelMode::Details;
                                focus_zone = FocusZone::Panel;
                                tui_layout_change = true;
                            }
                        }
                        KeyCode::Esc => {
                            if !show_panel {
                                focus_zone = FocusZone::Sidebar;
                                viewer.state.selected_block = None;
                            }
                        }
                        KeyCode::Char('p') => {
                            // TODO: make current path highlighted by default, and boundary edges indicated as dashed lines
                            // Toggle current path highlighting
                            if viewer.has_highlight(Color::Red) {
                                viewer.clear_highlight(Color::Red);
                            } else if let Some(ref bg_id) = explorer_state.selected_block_group_id {
                                // BlockGroup::get_current_path will panic if there's no path,
                                // so we roll our own query here. (todo: have get_current_path return an Option)
                                let current_path = <Path as Query>::get(
                                    conn,
                                    "SELECT * FROM paths WHERE block_group_id = ?1 ORDER BY created_on DESC LIMIT 1",
                                    params![bg_id],
                                );
                                match current_path {
                                    Ok(path) => {
                                        if let Err(err) = viewer
                                            .show_path(&path, get_theme_color("error").unwrap())
                                        {
                                            // todo: pop up a message in the panel
                                            messages.push_warn(err.to_string());
                                        }
                                    }
                                    Err(err) => {
                                        messages.push_warn(format!(
                                            "No path found for block group {bg_id}: {err}"
                                        ));
                                    }
                                }
                            } else {
                                messages.push_warn("No block group selected");
                            }
                        }
                        _ => {
                            viewer.handle_input(key);
                        }
                    },
                    FocusZone::Panel => match key.code {
                        KeyCode::Esc => {
                            show_panel = false;
                            focus_zone = FocusZone::Canvas;
                            tui_layout_change = true;
                        }
                        KeyCode::Up => {
                            if panel_mode == PanelMode::Messages {
                                message_scroll = message_scroll.saturating_sub(1);
                            }
                        }
                        KeyCode::Down => {
                            if panel_mode == PanelMode::Messages {
                                message_scroll = message_scroll.saturating_add(1);
                            }
                        }
                        KeyCode::Char('c') => {
                            if panel_mode == PanelMode::Messages {
                                messages.clear();
                                message_scroll = 0;
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
                        if let Some(toggled_id) =
                            explorer_state.annotation_file_toggle_requested.take()
                        {
                            if explorer_state.is_annotation_file_active(&toggled_id) {
                                if let Some(entry) = explorer.annotation_file_entry(&toggled_id)
                                    && let Some(bg) = current_block_group.as_ref()
                                {
                                    let query_window = current_view_coordinate_window(&viewer)
                                        .map(expand_query_window);
                                    let node_filter: HashSet<HashId> =
                                        block_graph.nodes().map(|node| node.node_id).collect();
                                    let request = AnnotationFileTrackRequest {
                                        conn,
                                        workspace,
                                        collection_name,
                                        sample_name: bg.sample_name.as_deref(),
                                        block_group_name: Some(&bg.name),
                                        query_window,
                                        node_filter: &node_filter,
                                        entry,
                                    };
                                    match load_annotation_file_track(&request) {
                                        Ok(load) => {
                                            annotation_file_tracks.insert(toggled_id, load.track);
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
                        if let Some(toggled_group) =
                            explorer_state.annotation_group_toggle_requested.take()
                        {
                            if explorer_state.is_annotation_group_active(&toggled_group) {
                                if current_block_group.is_some() {
                                    let visible_node_ranges = visible_ranges_by_node(&block_graph);
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
                                        explorer_state.deactivate_annotation_group(&toggled_group);
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
        }
        // Update tick
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Clean up terminal
    disable_raw_mode()?;
    let stdout = terminal.backend_mut();
    execute!(stdout, LeaveAlternateScreen)?;
    Ok(())
}
