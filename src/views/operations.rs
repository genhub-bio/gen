use std::{collections::HashMap, io, rc::Rc, time::Instant};

use crossterm::{
    event::{self, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_core::{HashId, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    operations::{Operation, OperationSummary},
    traits::Query,
};
use itertools::Itertools;
use rat_text::{
    HasScreenCursor,
    text_area::{TextArea, TextAreaState},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    prelude::{Color, StatefulWidget, Style},
    style::Modifier,
    widgets::{Block, Borders, Paragraph, Row, Table},
};
use rusqlite::{params, types::Value};

use crate::{
    config::get_theme_color,
    views::{
        gen_graph_widget::{create_gen_graph_controller, create_gen_graph_widget},
        helpers::{install_tui_panic_hook, style_text},
        patch::get_change_graph_from_hash,
    },
};

fn clip_text(t: &str, limit: usize) -> String {
    let t = t.replace("\n", " ");
    if t.len() > limit - 3 {
        format!("{trunc}...", trunc = &t[0..limit - 3])
    } else {
        t.to_string()
    }
}

struct OperationRow<'a> {
    operation: &'a Operation,
    summary: OperationSummary,
}

pub fn view_operations(context: &DbContext, operations: &[Operation]) -> Result<(), io::Error> {
    let conn = context.graph().conn();
    let op_conn = context.operations().conn();
    install_tui_panic_hook();

    let operation_by_hash: HashMap<_, &Operation> = HashMap::from_iter(
        operations
            .iter()
            .map(|op| (op.hash, op))
            .collect::<Vec<(_, &Operation)>>(),
    );
    let summaries = OperationSummary::query(
        op_conn,
        "select * from operation_summaries where operation_hash in rarray(?1)",
        params![Rc::new(
            operations
                .iter()
                .map(|x| Value::from(x.hash))
                .collect::<Vec<Value>>()
        )],
    );
    let mut operation_summaries = summaries
        .iter()
        .map(|summary| OperationRow {
            operation: operation_by_hash[&summary.operation_hash],
            summary: summary.clone(),
        })
        .collect::<Vec<_>>();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut textarea = TextAreaState::new();
    let mut empty_graph: GenGraph = GenGraph::new();
    let mut blockgroup_graphs: Vec<(HashId, String, GenGraph)> = vec![];
    let mut selected_blockgroup_graph: usize = 0;
    empty_graph.add_node(GraphNode {
        node_id: PATH_START_NODE_ID,
        block_id: 0,
        sequence_start: 0,
        sequence_end: 1,
    });

    let mut graph_controller = create_gen_graph_controller(&empty_graph);

    let mut view_message_panel = false;
    let mut view_graph = false;
    let mut panel_focus = "operations";
    let mut focus_rotation = vec!["operations"];
    let mut focus_index: usize = 0;
    let mut panel_activated = true; // Start with operations panel activated
    let focused_style = Style::default()
        .fg(Color::Blue)
        .add_modifier(Modifier::BOLD);
    let selected_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);
    let unfocused_style = Style::default().fg(Color::Gray);
    let status_bar_height: u16 = 1;

    let mut selected = 0;
    let mut last_frame_time = Instant::now();

    loop {
        let now = Instant::now();
        let frame_delta = now.duration_since(last_frame_time);
        last_frame_time = now;

        terminal.draw(|f| {
            let rows: Vec<Row> = operation_summaries
                .iter()
                .enumerate()
                .map(|(i, op)| {
                    let style = if i == selected {
                        Style::default().add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };

                    Row::new(vec![
                        clip_text(&op.operation.hash.to_string(), 40),
                        clip_text(&op.operation.change_type, 20),
                        clip_text(&op.summary.summary, 50),
                    ])
                    .style(style)
                })
                .collect();

            let ops_title = if !panel_activated && panel_focus == "operations" {
                "[Operations]" // Selected in nav mode
            } else {
                "Operations"
            };

            let table = Table::new(
                rows,
                [
                    Constraint::Length(40),
                    Constraint::Length(20),
                    Constraint::Length(50),
                ],
            )
            .header(
                Row::new(vec!["Operation Hash", "Change Type", "Summary"])
                    .style(Style::default().add_modifier(Modifier::UNDERLINED)),
            )
            .block(
                Block::default()
                    .title(ops_title)
                    .borders(Borders::ALL)
                    .border_style(if panel_activated && panel_focus == "operations" {
                        focused_style // Blue bold when active
                    } else if !panel_activated && panel_focus == "operations" {
                        selected_style // Cyan bold when selected but not active
                    } else {
                        unfocused_style // Gray when not selected
                    }),
            );

            let outer_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Min(1),
                    Constraint::Length(status_bar_height),
                ])
                .split(f.area());

            let main_area = outer_layout[0];
            let status_bar_area = outer_layout[1];

            let panel_messages = if !panel_activated {
                // NAVIGATION MODE - show cycling and panel management controls
                let mut msg = "*tab/arrows* navigate | *enter* activate panel".to_string();
                if panel_focus == "message_editor" || panel_focus == "graph_view" {
                    msg.push_str(" | *x* close panel");
                }
                msg.push_str(" | *q* quit");
                msg
            } else {
                // ACTIVE MODE - show panel-specific controls
                if panel_focus == "operations" {
                    "*↑↓* select | *e* edit msg | *v* view graph | *esc* leave panel".to_string()
                } else if panel_focus == "message_editor" {
                    "*ctrl+s* save | *esc* leave panel".to_string()
                } else if panel_focus == "graph_view" {
                    "*tab* cycle block groups | *←→↑↓* pan | *+/-* zoom | *esc* leave panel"
                        .to_string()
                } else {
                    "*esc* leave panel".to_string()
                }
            };

            let msg_editor_title = if !panel_activated && panel_focus == "message_editor" {
                "[Operation Summary]" // Selected in nav mode
            } else {
                "Operation Summary"
            };

            let msg_editor_block = Block::default()
                .title(msg_editor_title)
                .borders(Borders::ALL)
                .border_style(if panel_activated && panel_focus == "message_editor" {
                    focused_style // Blue bold when active
                } else if !panel_activated && panel_focus == "message_editor" {
                    selected_style // Cyan bold when selected but not active
                } else {
                    unfocused_style // Gray when not selected
                });

            // Determine the canvas area for the graph and render all panels
            let canvas_area = if view_message_panel {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(main_area);
                f.render_widget(table, chunks[0]);
                if view_graph {
                    let sub_chunk = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(chunks[1]);
                    TextArea::new().block(msg_editor_block.clone()).render(
                        sub_chunk[0],
                        f.buffer_mut(),
                        &mut textarea,
                    );
                    Some(sub_chunk[1])
                } else {
                    TextArea::new().block(msg_editor_block.clone()).render(
                        chunks[1],
                        f.buffer_mut(),
                        &mut textarea,
                    );
                    None
                }
            } else if view_graph {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(main_area);
                f.render_widget(table, chunks[0]);
                Some(chunks[1])
            } else {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(100)].as_ref())
                    .split(main_area);
                f.render_widget(table, chunks[0]);
                None
            };

            // Render the graph widget if a canvas area is available
            if let Some(canvas_area) = canvas_area {
                let graph_title = if blockgroup_graphs.is_empty() {
                    "Change Graph".to_string()
                } else {
                    format!(
                        "Change Graph {name}",
                        name = blockgroup_graphs[selected_blockgroup_graph].1
                    )
                };
                let graph_title = if !panel_activated && panel_focus == "graph_view" {
                    format!("[{}]", graph_title)
                } else {
                    graph_title
                };
                let graph_border_style = if panel_activated && panel_focus == "graph_view" {
                    focused_style
                } else if !panel_activated && panel_focus == "graph_view" {
                    selected_style
                } else {
                    unfocused_style
                };

                let graph_block = Block::default()
                    .title(graph_title)
                    .borders(Borders::ALL)
                    .border_style(graph_border_style);
                let inner_canvas = graph_block.inner(canvas_area);

                // Update viewport with exact inner bounds before animations
                graph_controller.viewport_state.focus();
                graph_controller.viewport_state.viewport_bounds = inner_canvas;
                graph_controller.update_animations(frame_delta);

                f.render_widget(graph_block, canvas_area);

                let canvas_style = Style::default().bg(get_theme_color("canvas").unwrap());
                let widget = create_gen_graph_widget(conn)
                    .detail_level(graph_controller.get_detail_level())
                    .style(canvas_style)
                    .cursor();
                f.render_stateful_widget(widget, inner_canvas, &mut graph_controller);
            }
            let status_bar_contents = format!(
                "{panel_messages:^width$}",
                width = status_bar_area.width as usize
            );
            let status_line = style_text(
                &status_bar_contents,
                Style::default().fg(get_theme_color("text_muted").unwrap()),
                Style::default().fg(get_theme_color("highlight").unwrap()),
            );
            let status_bar = Paragraph::new(status_line)
                .style(Style::default().bg(get_theme_color("statusbar").unwrap()));
            f.render_widget(status_bar, status_bar_area);

            // Set cursor position for the message editor when focused
            if view_message_panel
                && panel_activated
                && panel_focus == "message_editor"
                && let Some((cursor_x, cursor_y)) = textarea.screen_cursor()
            {
                f.set_cursor_position((cursor_x, cursor_y));
            }
        })?;

        if event::poll(std::time::Duration::from_millis(100))?
            && let event::Event::Key(key) = event::read()?
        {
            if !panel_activated {
                // NAVIGATION MODE - handle Tab/Shift+Tab, Enter, x, q
                match key.code {
                    KeyCode::Tab => {
                        focus_index += 1;
                        if focus_index >= focus_rotation.len() {
                            focus_index = 0;
                        }
                        panel_focus = focus_rotation[focus_index];
                    }
                    KeyCode::BackTab => {
                        if focus_index > 0 {
                            focus_index -= 1;
                        } else {
                            focus_index = focus_rotation.len() - 1;
                        }
                        panel_focus = focus_rotation[focus_index];
                    }
                    // *   Down from "Operations" moves to the first available bottom panel ("Message Editor" or "Graph View").
                    //     *   Up from any bottom panel returns to "Operations".
                    //     *   Right moves from "Message Editor" to "Graph View" (if visible).
                    //     *   Left moves from "Graph View" to "Message Editor" (if visible).
                    KeyCode::Up => {
                        if panel_focus != "operations" {
                            panel_focus = "operations";
                            if let Some(idx) = focus_rotation.iter().position(|&s| s == panel_focus)
                            {
                                focus_index = idx;
                            }
                        }
                    }
                    KeyCode::Down => {
                        if panel_focus == "operations" {
                            if view_message_panel {
                                panel_focus = "message_editor";
                            } else if view_graph {
                                panel_focus = "graph_view";
                            }
                            if let Some(idx) = focus_rotation.iter().position(|&s| s == panel_focus)
                            {
                                focus_index = idx;
                            }
                        }
                    }
                    KeyCode::Left => {
                        if panel_focus == "graph_view" && view_message_panel {
                            panel_focus = "message_editor";
                            if let Some(idx) = focus_rotation.iter().position(|&s| s == panel_focus)
                            {
                                focus_index = idx;
                            }
                        }
                    }
                    KeyCode::Right => {
                        if panel_focus == "message_editor" && view_graph {
                            panel_focus = "graph_view";
                            if let Some(idx) = focus_rotation.iter().position(|&s| s == panel_focus)
                            {
                                focus_index = idx;
                            }
                        }
                    }
                    KeyCode::Enter => {
                        // Activate the currently selected panel
                        panel_activated = true;
                    }
                    KeyCode::Char('x') => {
                        // Close the currently selected panel
                        if panel_focus == "message_editor" {
                            view_message_panel = false;
                            // Remove from focus_rotation
                            if let Some((p, _)) = focus_rotation
                                .iter()
                                .find_position(|s| **s == "message_editor")
                            {
                                focus_rotation.remove(p);
                            }
                            // Adjust focus_index and cycle to next panel
                            if focus_index >= focus_rotation.len() {
                                focus_index = 0;
                            }
                            panel_focus = focus_rotation[focus_index];
                        } else if panel_focus == "graph_view" {
                            view_graph = false;
                            // Remove from focus_rotation
                            if let Some((p, _)) =
                                focus_rotation.iter().find_position(|s| **s == "graph_view")
                            {
                                focus_rotation.remove(p);
                            }
                            // Adjust focus_index and cycle to next panel
                            if focus_index >= focus_rotation.len() {
                                focus_index = 0;
                            }
                            panel_focus = focus_rotation[focus_index];
                        }
                        // Operations panel cannot be closed
                    }
                    KeyCode::Char('q') => {
                        // Exit application
                        break;
                    }
                    _ => {}
                }
            } else {
                // ACTIVE MODE - delegate to active panel, Esc leaves (doesn't close)
                if key.code == KeyCode::Esc {
                    panel_activated = false;
                } else {
                    match panel_focus {
                        "operations" => {
                            // Operations table in active mode handles: e, v, up/down
                            match key.code {
                                KeyCode::Up => {
                                    if selected > 0 {
                                        selected = selected.saturating_sub(1);
                                    }
                                }
                                KeyCode::Down => {
                                    if selected < operations.len() - 1 {
                                        selected += 1;
                                    }
                                }
                                KeyCode::Char('e') => {
                                    // Open message editor
                                    textarea
                                        .set_text(&operation_summaries[selected].summary.summary);
                                    view_message_panel = true;
                                    // Add to focus_rotation if not present
                                    focus_index = if let Some((i, _)) = focus_rotation
                                        .iter()
                                        .find_position(|s| **s == "message_editor")
                                    {
                                        i
                                    } else {
                                        focus_rotation.push("message_editor");
                                        focus_rotation.len() - 1
                                    };
                                    panel_focus = focus_rotation[focus_index];
                                    // panel_activated remains true (auto-activate new panel)
                                }
                                KeyCode::Char('v') => {
                                    // Open graph view
                                    view_graph = true;
                                    // Add to focus_rotation if not present
                                    focus_index = if let Some((i, _)) =
                                        focus_rotation.iter().find_position(|s| **s == "graph_view")
                                    {
                                        i
                                    } else {
                                        focus_rotation.push("graph_view");
                                        focus_rotation.len() - 1
                                    };
                                    panel_focus = focus_rotation[focus_index];
                                    // Load graphs for selected operation
                                    let hash = operation_summaries[selected].operation.hash;
                                    let graphs =
                                        get_change_graph_from_hash(context, &hash).unwrap();
                                    blockgroup_graphs.clear();
                                    let bg_info = BlockGroup::query_by_ids(
                                        conn,
                                        &graphs.keys().cloned().collect::<Vec<_>>(),
                                    );
                                    let bg_map: HashMap<HashId, &BlockGroup> =
                                        HashMap::from_iter(bg_info.iter().map(|k| (k.id, k)));
                                    for (i, v) in graphs {
                                        blockgroup_graphs.push((
                                            i,
                                            format!(
                                                "{collection} {sample} {name}",
                                                collection = bg_map[&i].collection_name.clone(),
                                                sample = bg_map[&i]
                                                    .sample_name
                                                    .clone()
                                                    .unwrap_or("Reference".to_string()),
                                                name = bg_map[&i].name.clone()
                                            ),
                                            v,
                                        ));
                                    }
                                    selected_blockgroup_graph = 0;
                                    graph_controller = if blockgroup_graphs.is_empty() {
                                        create_gen_graph_controller(&empty_graph)
                                    } else {
                                        create_gen_graph_controller(
                                            &blockgroup_graphs[selected_blockgroup_graph].2,
                                        )
                                    };
                                    // panel_activated remains true (auto-activate new panel)
                                }
                                _ => {}
                            }
                        }
                        "message_editor" => {
                            if key.code == KeyCode::Char('s')
                                && key.modifiers.contains(KeyModifiers::CONTROL)
                            {
                                // Save message
                                let new_summary = textarea.text();
                                let _ = OperationSummary::set_message(
                                    op_conn,
                                    operation_summaries[selected].summary.id,
                                    &new_summary,
                                );
                                operation_summaries[selected].summary.summary = new_summary;
                            } else {
                                // Convert crossterm KeyEvent to rat-text event
                                let _outcome = rat_text::text_area::handle_events(
                                    &mut textarea,
                                    true, // focused
                                    &crossterm::event::Event::Key(key),
                                );
                            }
                        }
                        "graph_view" => {
                            if key.code == KeyCode::Tab || key.code == KeyCode::BackTab {
                                // Cycle block groups
                                if key.code == KeyCode::BackTab {
                                    if selected_blockgroup_graph == 0 {
                                        selected_blockgroup_graph = blockgroup_graphs.len() - 1;
                                    } else {
                                        selected_blockgroup_graph -= 1;
                                    }
                                } else {
                                    selected_blockgroup_graph += 1;
                                    if selected_blockgroup_graph >= blockgroup_graphs.len() {
                                        selected_blockgroup_graph = 0;
                                    }
                                }
                                // Update the graph controller with the new graph
                                graph_controller = create_gen_graph_controller(
                                    &blockgroup_graphs[selected_blockgroup_graph].2,
                                );
                            } else {
                                let _ = graph_controller.handle_key_event(key);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
