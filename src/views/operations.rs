// Note: Using patched tui-textarea from https://github.com/phsym/tui-textarea
// This fork has ratatui 0.30.0 compatibility fixes that haven't been merged upstream yet

use std::{backtrace::Backtrace, collections::HashMap, io, rc::Rc, time::Instant};

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
use gen_tui::{graph_controller::GraphController, layout::VisualDetail, theme::Theme};
use itertools::Itertools;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    prelude::{Color, Style, Text},
    style::Modifier,
    widgets::{Block, Borders, Paragraph, Row, Table},
};
use rusqlite::{params, types::Value};
use tui_textarea::TextArea;

use crate::{
    config::get_theme_color,
    views::{
        gen_graph_widget::{GenGraphNodeSizer, create_gen_graph_widget},
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

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
}

pub fn view_operations(context: &DbContext, operations: &[Operation]) -> Result<(), io::Error> {
    let conn = context.graph().conn();
    let op_conn = context.operations().conn();
    std::panic::set_hook(Box::new(|info| {
        restore_terminal();
        eprintln!("Application crashed: {info}");
        let backtrace = Backtrace::capture();
        eprintln!("Stack trace:\n{backtrace}");
    }));

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

    let mut textarea = TextArea::default();
    let mut empty_graph: GenGraph = GenGraph::new();
    let mut blockgroup_graphs: Vec<(HashId, String, GenGraph)> = vec![];
    let mut selected_blockgroup_graph: usize = 0;
    empty_graph.add_node(GraphNode {
        node_id: PATH_START_NODE_ID,
        block_id: 0,
        sequence_start: 0,
        sequence_end: 1,
    });

    // Create graph controller instead of Viewer
    let node_sizer = GenGraphNodeSizer;
    let mut graph_controller = GraphController::new(&empty_graph, node_sizer).with_theme(Theme {
        canvas: get_theme_color("canvas").unwrap(),
        node_fg: get_theme_color("text").unwrap(),
        node_bg: get_theme_color("node").unwrap(),
        edge_fg: get_theme_color("edge").unwrap(),
        edge_bg: get_theme_color("canvas").unwrap(),
        cursor_fg: get_theme_color("cursor_fg").unwrap(),
        cursor_bg: get_theme_color("cursor_bg").unwrap(),
    });
    graph_controller.set_detail_level(VisualDetail::Truncated);
    graph_controller.show_cursor();
    graph_controller.initialize_cursor();
    graph_controller
        .rebuild_viewport_graph()
        .map_err(io::Error::other)?;

    let mut view_message_panel = false;
    let mut view_graph = false;
    let mut panel_focus = "operations";
    let mut focus_rotation = vec!["operations"];
    let mut focus_index: usize = 0;
    let focused_style = Style::default()
        .fg(Color::Blue)
        .add_modifier(Modifier::BOLD);
    let unfocused_style = Style::default().fg(Color::Gray);
    let status_bar_height: u16 = 1;

    let mut selected = 0;
    let mut last_frame_time = Instant::now();

    loop {
        // Calculate frame delta for smooth animations
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
                    .title("Operations")
                    .borders(Borders::ALL)
                    .border_style(if panel_focus == "operations" {
                        focused_style
                    } else {
                        unfocused_style
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

            let mut panel_messages = " Controls: ctrl+up/down=cycle focus".to_string();

            // for ease, we just set all panels to unfocused here
            textarea.set_block(
                Block::default()
                    .title("Operation Summary")
                    .borders(Borders::ALL)
                    .border_style(unfocused_style),
            );

            if panel_focus == "message_editor" {
                panel_messages.push_str("| ctrl+s=save message | esc=close message editor");
                textarea.set_block(
                    Block::default()
                        .title("Operation Summary")
                        .borders(Borders::ALL)
                        .border_style(focused_style),
                );
            } else if panel_focus == "operations" {
                panel_messages
                    .push_str(" | e or enter=edit message | v=view graph | esc or q=exit");
            } else if panel_focus == "graph_view" {
                panel_messages
                    .push_str(" | tab = cycle block group | ←→↑↓ pan | +/- zoom | esc or q=exit");
            }

            if view_message_panel {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(main_area);
                if view_graph {
                    let sub_chunk = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(chunks[1]);
                    f.render_widget(&textarea, sub_chunk[0]);

                    // Update graph controller viewport and render widget
                    let canvas_area = sub_chunk[1];
                    graph_controller.viewport_state.focus();
                    graph_controller.viewport_state.viewport_bounds = canvas_area;
                    graph_controller.update_animations(frame_delta);

                    let canvas_style = Style::default().bg(get_theme_color("canvas").unwrap());
                    let graph_title = if blockgroup_graphs.is_empty() {
                        "Change Graph".to_string()
                    } else {
                        format!(
                            "Change Graph {name}",
                            name = blockgroup_graphs[selected_blockgroup_graph].1
                        )
                    };
                    let graph_block = Block::default()
                        .title(graph_title)
                        .borders(Borders::ALL)
                        .border_style(if panel_focus == "graph_view" {
                            focused_style
                        } else {
                            unfocused_style
                        });

                    let inner_canvas = graph_block.inner(canvas_area);
                    f.render_widget(graph_block, canvas_area);

                    let widget = create_gen_graph_widget(conn)
                        .detail_level(graph_controller.get_detail_level())
                        .style(canvas_style)
                        .cursor();
                    f.render_stateful_widget(widget, inner_canvas, &mut graph_controller);
                } else {
                    f.render_widget(&textarea, chunks[1]);
                }
                f.render_widget(table, chunks[0]);
            } else if view_graph {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(main_area);
                f.render_widget(table, chunks[0]);

                // Update graph controller viewport and render widget
                let canvas_area = chunks[1];
                graph_controller.viewport_state.focus();
                graph_controller.viewport_state.viewport_bounds = canvas_area;
                graph_controller.update_animations(frame_delta);

                let canvas_style = Style::default().bg(get_theme_color("canvas").unwrap());
                let graph_title = if blockgroup_graphs.is_empty() {
                    "Change Graph".to_string()
                } else {
                    format!(
                        "Change Graph {name}",
                        name = blockgroup_graphs[selected_blockgroup_graph].1
                    )
                };
                let graph_block = Block::default()
                    .title(graph_title)
                    .borders(Borders::ALL)
                    .border_style(if panel_focus == "graph_view" {
                        focused_style
                    } else {
                        unfocused_style
                    });

                let inner_canvas = graph_block.inner(canvas_area);
                f.render_widget(graph_block, canvas_area);

                let widget = create_gen_graph_widget(conn)
                    .detail_level(graph_controller.get_detail_level())
                    .style(canvas_style)
                    .cursor();
                f.render_stateful_widget(widget, inner_canvas, &mut graph_controller);
            } else {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(100)].as_ref())
                    .split(main_area);
                f.render_widget(table, chunks[0]);
            };
            let status_bar_contents = format!(
                "{panel_messages:width$}",
                width = status_bar_area.width as usize
            );
            let status_bar = Paragraph::new(Text::styled(
                status_bar_contents,
                Style::default().bg(Color::DarkGray).fg(Color::White),
            ));
            f.render_widget(status_bar, status_bar_area);
        })?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let event::Event::Key(key) = event::read()? {
                if key.modifiers == KeyModifiers::CONTROL
                    && (key.code == KeyCode::Up || key.code == KeyCode::Down)
                {
                    if key.code == KeyCode::Down {
                        focus_index += 1;
                        if focus_index >= focus_rotation.len() {
                            focus_index = 0;
                        }
                        panel_focus = focus_rotation[focus_index];
                    } else {
                        if focus_index > 0 {
                            focus_index -= 1;
                        } else {
                            focus_index = focus_rotation.len() - 1;
                        }
                        panel_focus = focus_rotation[focus_index];
                    }
                } else if panel_focus == "message_editor" {
                    if key.code == KeyCode::Esc {
                        view_message_panel = false;
                        if let Some((p, _)) = focus_rotation
                            .iter()
                            .find_position(|s| **s == "message_editor")
                        {
                            focus_rotation.remove(p);
                        }
                        if focus_index >= focus_rotation.len() {
                            focus_index = 0;
                        }
                        panel_focus = focus_rotation[focus_index];
                    } else if key.code == KeyCode::Char('s')
                        && key.modifiers == KeyModifiers::CONTROL
                    {
                        let new_summary = textarea.lines().iter().join("\n");
                        let _ = OperationSummary::set_message(
                            op_conn,
                            operation_summaries[selected].summary.id,
                            &new_summary,
                        );
                        operation_summaries[selected].summary.summary = new_summary;
                    } else {
                        textarea.input(key);
                    }
                } else if panel_focus == "graph_view" {
                    if key.code == KeyCode::Esc || key.code == KeyCode::Char('q') {
                        view_graph = false;
                        if let Some((p, _)) =
                            focus_rotation.iter().find_position(|s| **s == "graph_view")
                        {
                            focus_rotation.remove(p);
                        }
                        if focus_index >= focus_rotation.len() {
                            focus_index = 0;
                        }
                        panel_focus = focus_rotation[focus_index];
                    } else if key.code == KeyCode::Tab || key.code == KeyCode::BackTab {
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
                        let node_sizer = GenGraphNodeSizer;
                        graph_controller = GraphController::new(
                            &blockgroup_graphs[selected_blockgroup_graph].2,
                            node_sizer,
                        )
                        .with_theme(Theme {
                            canvas: get_theme_color("canvas").unwrap(),
                            node_fg: get_theme_color("text").unwrap(),
                            node_bg: get_theme_color("node").unwrap(),
                            edge_fg: get_theme_color("edge").unwrap(),
                            edge_bg: get_theme_color("canvas").unwrap(),
                            cursor_fg: get_theme_color("cursor_fg").unwrap(),
                            cursor_bg: get_theme_color("cursor_bg").unwrap(),
                        });
                        graph_controller.set_detail_level(VisualDetail::Truncated);
                        graph_controller.show_cursor();
                        graph_controller.initialize_cursor();
                        // Ignore rebuild errors during block group switching
                        let _ = graph_controller.rebuild_viewport_graph();
                    } else {
                        let _ = graph_controller.handle_key_event(key);
                    }
                } else {
                    let code = key.code;
                    match code {
                        KeyCode::Esc | KeyCode::Char('q') => break,
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
                        KeyCode::Enter | KeyCode::Char('e') => {
                            textarea = TextArea::from_iter(
                                operation_summaries[selected].summary.summary.split("\n"),
                            );
                            view_message_panel = true;
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
                        }
                        KeyCode::Char('v') => {
                            view_graph = true;
                            focus_index = if let Some((i, _)) =
                                focus_rotation.iter().find_position(|s| **s == "graph_view")
                            {
                                i
                            } else {
                                focus_rotation.push("graph_view");
                                focus_rotation.len() - 1
                            };
                            panel_focus = focus_rotation[focus_index];
                            let hash = operation_summaries[selected].operation.hash;
                            let graphs = get_change_graph_from_hash(context, &hash).unwrap();
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
                            if blockgroup_graphs.is_empty() {
                                let node_sizer = GenGraphNodeSizer;
                                graph_controller = GraphController::new(&empty_graph, node_sizer)
                                    .with_theme(Theme {
                                        canvas: get_theme_color("canvas").unwrap(),
                                        node_fg: get_theme_color("text").unwrap(),
                                        node_bg: get_theme_color("node").unwrap(),
                                        edge_fg: get_theme_color("edge").unwrap(),
                                        edge_bg: get_theme_color("canvas").unwrap(),
                                        cursor_fg: get_theme_color("cursor_fg").unwrap(),
                                        cursor_bg: get_theme_color("cursor_bg").unwrap(),
                                    });
                                graph_controller.set_detail_level(VisualDetail::Truncated);
                                graph_controller.show_cursor();
                                graph_controller.initialize_cursor();
                                // Ignore rebuild errors for empty graph
                                let _ = graph_controller.rebuild_viewport_graph();
                            } else {
                                let node_sizer = GenGraphNodeSizer;
                                graph_controller = GraphController::new(
                                    &blockgroup_graphs[selected_blockgroup_graph].2,
                                    node_sizer,
                                )
                                .with_theme(Theme {
                                    canvas: get_theme_color("canvas").unwrap(),
                                    node_fg: get_theme_color("text").unwrap(),
                                    node_bg: get_theme_color("node").unwrap(),
                                    edge_fg: get_theme_color("edge").unwrap(),
                                    edge_bg: get_theme_color("canvas").unwrap(),
                                    cursor_fg: get_theme_color("cursor_fg").unwrap(),
                                    cursor_bg: get_theme_color("cursor_bg").unwrap(),
                                });
                                graph_controller.set_detail_level(VisualDetail::Truncated);
                                graph_controller.show_cursor();
                                graph_controller.initialize_cursor();
                                // Ignore rebuild errors when initially loading graph
                                let _ = graph_controller.rebuild_viewport_graph();
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
