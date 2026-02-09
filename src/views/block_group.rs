use std::{
    error::Error,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode, connect_all_boundary_edges};
use gen_models::{block_group::BlockGroup, db::GraphConnection, node::Node, traits::Query};
use gen_tui::{graph_controller::GraphController, plotter::PathStyle};
use log::{info, warn};
use ratatui::{
    layout::Constraint,
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Padding, Paragraph, Wrap},
};
use rusqlite::params;

use crate::{
    config::get_theme_color,
    progress_bar::{get_handler, get_time_elapsed_bar},
    views::{
        collection::{CollectionExplorer, CollectionExplorerState, FocusZone},
        gen_graph_widget::{
            GenGraphNodeSizer, create_gen_graph_controller, create_gen_graph_widget,
        },
        helpers::{install_tui_panic_hook, style_text},
    },
};

// Frequency by which we check for external updates to the db
const REFRESH_INTERVAL: u64 = 3; // seconds

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
    controller: &mut GraphController<&GenGraph, GenGraphNodeSizer>,
    block_group_id: &gen_core::HashId,
    color: ratatui::style::Color,
) -> Result<bool, String> {
    let style = PathStyle::new(color)
        .with_line_style(gen_tui::plotter::LineStyle::Bold)
        .with_merge_glyphs(true);
    // Check if highlighting is already active for this style
    if controller.has_highlight(&style) {
        controller.clear_highlight(&style);
        Ok(false)
    } else {
        // Get the path nodes for this block group
        let path_nodes = get_block_group_path_nodes(conn, block_group_id, controller.graph)?;

        // Set the path highlight using GraphNodes directly
        controller.set_path_highlight(style, path_nodes);
        Ok(true)
    }
}

pub fn view_block_group(
    conn: &GraphConnection,
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
    let mut explorer = CollectionExplorer::new(conn, collection_name);
    let mut explorer_state = CollectionExplorerState::new();
    if let Some(ref s) = sample_name {
        explorer_state.toggle_sample(s);
    }

    let mut block_graph;
    let mut block_group_id: Option<gen_core::HashId> = None;
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

    // Create the graph controller and initial graph
    let bar = progress_bar.add(get_time_elapsed_bar());
    let _ = progress_bar.println("Pre-computing layout in chunks");

    let mut graph_controller = create_gen_graph_controller(&block_graph);

    // TODO: Handle origin positioning - not directly supported in new widget yet
    if origin.is_some() {
        warn!("Origin positioning not yet supported in GenGraphWidget");
    }

    bar.finish();

    // Setup terminal
    install_tui_panic_hook();

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = ratatui::init();

    // Basic event loop
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();
    let mut last_frame_time = Instant::now();
    let mut show_panel = false;
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
            if explorer.refresh(conn, collection_name) {
                explorer.force_reload(&mut explorer_state);
            }
            last_refresh = Instant::now();
        }

        // Trigger reload if selection changed to a new block group
        if explorer_state.selected_block_group_id != last_selected_block_group_id {
            is_loading = true;
            last_selected_block_group_id = explorer_state.selected_block_group_id;
        }

        // Calculate frame delta for smooth animations
        let now = Instant::now();
        let frame_delta = now.duration_since(last_frame_time);
        last_frame_time = now;

        // Draw the UI
        terminal.draw(|frame| {
            let status_bar_height: u16 = 1;

            // The outer layout is a vertical split between the status bar and everything else
            let outer_layout = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints(vec![
                    ratatui::layout::Constraint::Min(1),
                    ratatui::layout::Constraint::Length(status_bar_height),
                ])
                .split(frame.area());
            let status_bar_area = outer_layout[1];

            // The sidebar is a horizontal split of the area above the status bar
            let sidebar_layout = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints(vec![Constraint::Percentage(20), Constraint::Percentage(80)])
                .split(outer_layout[0]);
            let sidebar_area = sidebar_layout[0];

            // The panel pops up in the canvas area, it does not overlap with the sidebar
            let panel_layout = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints(vec![Constraint::Percentage(80), Constraint::Percentage(20)])
                .split(sidebar_layout[1]);
            let panel_area = panel_layout[1];

            let canvas_area = if show_panel {
                panel_layout[0]
            } else {
                sidebar_layout[1]
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

            // Status bar
            let mut status_message = match focus_zone {
                FocusZone::Canvas => {
                    "*←→↑↓* pan | *+/-* zoom | *p* toggle path | *esc* back to sidebar".to_string()
                }
                FocusZone::Panel => "*esc* close panel".to_string(),
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

            // Canvas area
            if is_loading {
                // Draw loading message in canvas area
                let loading_text = Text::styled(
                    "Loading...",
                    Style::default()
                        .fg(get_theme_color("text").unwrap())
                        .add_modifier(Modifier::BOLD),
                );
                let loading_para = Paragraph::new(loading_text)
                    .alignment(ratatui::layout::HorizontalAlignment::Center);

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
                graph_controller.viewport_state.focus();
                let canvas_style = Style::default().bg(get_theme_color("canvas").unwrap());
                let widget = create_gen_graph_widget(conn)
                    .detail_level(graph_controller.get_detail_level())
                    .style(canvas_style)
                    .cursor();
                frame.render_stateful_widget(widget, canvas_area, &mut graph_controller);
            }

            // Panel
            if show_panel {
                let panel_block = Block::bordered()
                    .padding(Padding::new(2, 2, 1, 1))
                    .title("Details")
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

                // TODO: Node selection not yet supported in GenGraphWidget
                let panel_text = vec![
                    Line::from(vec![
                        Span::styled(
                            "Camera Position: ",
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(format!(
                            "({}, {})",
                            graph_controller.viewport_state.camera_current.x,
                            graph_controller.viewport_state.camera_current.y
                        )),
                    ]),
                    Line::from(vec![
                        Span::styled(
                            "Detail Level: ",
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(format!("{:?}", graph_controller.get_detail_level())),
                    ]),
                    Line::from(vec![Span::styled(
                        "Node selection not yet supported",
                        Style::default()
                            .fg(get_theme_color("text").unwrap())
                            .add_modifier(Modifier::ITALIC),
                    )]),
                ];

                let panel_content = Paragraph::new(panel_text)
                    .wrap(Wrap { trim: true })
                    .alignment(ratatui::layout::HorizontalAlignment::Left)
                    .block(panel_block);

                // Clear the panel area if we just changed the layout
                if tui_layout_change {
                    frame.render_widget(Clear, panel_area);
                }
                frame.render_widget(panel_content, panel_area);

                // Reset the layout change flag
                tui_layout_change = false;
            }
        })?;

        // After drawing, update the graph controller if needed
        if is_loading && let Some(ref new_block_group_id) = explorer_state.selected_block_group_id {
            // Create a new graph for the selected block group
            block_graph = BlockGroup::get_graph(conn, new_block_group_id);
            connect_all_boundary_edges(&mut block_graph);
            // Update the graph controller
            graph_controller = create_gen_graph_controller(&block_graph);

            is_loading = false;
        }

        // Handle input
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));
        if crossterm::event::poll(timeout)?
            && let event::Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            // Global handlers
            match key.code {
                KeyCode::Char('q') => break,
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
                        focus_zone = FocusZone::Panel;
                        tui_layout_change = true;
                    }
                    KeyCode::Esc => {
                        if !show_panel {
                            focus_zone = FocusZone::Sidebar;
                        }
                    }
                    KeyCode::Char('p') => {
                        if let Some(ref block_group_id) = explorer_state.selected_block_group_id {
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
                FocusZone::Panel => {
                    if key.code == KeyCode::Esc {
                        show_panel = false;
                        focus_zone = FocusZone::Canvas;
                        tui_layout_change = true;
                    }
                }
                FocusZone::Sidebar => {
                    explorer.handle_input(&mut explorer_state, key);
                    // Check if focus change was requested by the explorer
                    if let Some(requested_zone) = explorer_state.focus_change_requested {
                        focus_zone = requested_zone;
                        explorer_state.focus_change_requested = None;
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
