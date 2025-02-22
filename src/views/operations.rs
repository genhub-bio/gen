use crate::graph::{GraphEdge, GraphNode};
use crate::models::block_group::BlockGroup;
use crate::models::operations::{Operation, OperationSummary};
use crate::models::traits::Query;
use crate::views::block_group_viewer::{PlotParameters, Viewer};
use crossterm::event::KeyModifiers;
use crossterm::{
    event::{self, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use itertools::Itertools;
use petgraph::prelude::DiGraphMap;
use ratatui::prelude::{Color, Style, Text};
use ratatui::style::Modifier;
use ratatui::widgets::Paragraph;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Row, Table},
    Terminal,
};
use rusqlite::{params, types::Value, Connection};
use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::io;
use std::rc::Rc;
use tui_textarea::TextArea;

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

pub fn view_operations(
    conn: &Connection,
    op_conn: &Connection,
    operations: &[Operation],
) -> Result<(), io::Error> {
    std::panic::set_hook(Box::new(|info| {
        restore_terminal();
        eprintln!("Application crashed: {info}");
        let backtrace = Backtrace::capture();
        eprintln!("Stack trace:\n{}", backtrace);
    }));

    let operation_by_hash: HashMap<String, &Operation> = HashMap::from_iter(
        operations
            .iter()
            .map(|op| (op.hash.clone(), op))
            .collect::<Vec<(String, &Operation)>>(),
    );
    let summaries = OperationSummary::query(
        op_conn,
        "select * from operation_summary where operation_hash in rarray(?1)",
        params![Rc::new(
            operations
                .iter()
                .map(|x| Value::from(x.hash.clone()))
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
    let mut empty_graph: DiGraphMap<GraphNode, GraphEdge> = DiGraphMap::new();
    let mut blockgroup_graphs: Vec<(i64, String, DiGraphMap<GraphNode, GraphEdge>)> = vec![];
    let mut selected_blockgroup_graph: usize = 0;
    empty_graph.add_node(GraphNode {
        node_id: 1,
        block_id: 0,
        sequence_start: 0,
        sequence_end: 1,
    });
    let mut graph_viewer = Viewer::new(&empty_graph, conn, PlotParameters::default());
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
    loop {
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
                        clip_text(&op.operation.hash, 40),
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
            if view_graph {
                graph_viewer.set_block(
                    Block::default()
                        .title(if blockgroup_graphs.is_empty() {
                            "Change Graph".to_string()
                        } else {
                            format!(
                                "Change Graph {name}",
                                name = blockgroup_graphs[selected_blockgroup_graph].1
                            )
                        })
                        .borders(Borders::ALL)
                        .border_style(unfocused_style),
                );
            }

            if panel_focus == "message_editor" {
                panel_messages.push_str(", ctrl+s=save message, esc=close message editor");
                textarea.set_block(
                    Block::default()
                        .title("Operation Summary")
                        .borders(Borders::ALL)
                        .border_style(focused_style),
                );
            } else if panel_focus == "operations" {
                panel_messages.push_str(", e or enter=edit message, v=view graph, esc or q=exit");
            } else if panel_focus == "graph_view" {
                panel_messages.push_str(&format!(
                    ", tab = cycle block group, {l}",
                    l = Viewer::get_status_line()
                ));
                graph_viewer.set_block(
                    Block::default()
                        .title(if blockgroup_graphs.is_empty() {
                            "Change Graph".to_string()
                        } else {
                            format!(
                                "Change Graph {name}",
                                name = blockgroup_graphs[selected_blockgroup_graph].1
                            )
                        })
                        .borders(Borders::ALL)
                        .border_style(focused_style),
                );
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
                    graph_viewer.draw(f, sub_chunk[1]);
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
                graph_viewer.draw(f, chunks[1]);
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
                    if key.code == KeyCode::Esc {
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
                    } else if key.code == KeyCode::Tab {
                        if key.modifiers == KeyModifiers::SHIFT {
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
                        graph_viewer = Viewer::new(
                            &blockgroup_graphs[selected_blockgroup_graph].2,
                            conn,
                            PlotParameters::default(),
                        );
                    } else {
                        graph_viewer.handle_input(key);
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
                            let hash = &operation_summaries[selected].operation.hash;
                            let graphs = Operation::get_change_graph(op_conn, hash).unwrap();
                            blockgroup_graphs.clear();
                            let bg_info = BlockGroup::get_by_ids(
                                conn,
                                &graphs.keys().copied().collect::<Vec<i64>>(),
                            );
                            let bg_map: HashMap<i64, &BlockGroup> =
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
                                graph_viewer =
                                    Viewer::new(&empty_graph, conn, PlotParameters::default());
                            } else {
                                graph_viewer = Viewer::new(
                                    &blockgroup_graphs[selected_blockgroup_graph].2,
                                    conn,
                                    PlotParameters::default(),
                                );
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
