use std::{collections::HashMap, io, rc::Rc, time::Instant};

use crossterm::event::{self, KeyCode, KeyModifiers};
use gen_core::{HashId, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    operations::{Operation, OperationSummary},
    traits::Query,
};
use rat_text::{
    HasScreenCursor,
    text_area::{TextArea, TextAreaState},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    prelude::{StatefulWidget, Style},
    style::Modifier,
    widgets::{Row, Table},
};
use rusqlite::{params, types::Value};

use crate::{
    config::get_theme_color,
    views::{
        gen_graph_widget::{create_gen_graph_controller, create_gen_graph_widget},
        panels::{PanelFocus, PanelStyles, panel_block, render_status_bar},
        patch::get_change_graph_from_hash,
        tui_runtime::TuiSession,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationPanel {
    Operations,
    MessageEditor,
    GraphView,
}

pub fn view_operations(context: &DbContext, operations: &[Operation]) -> Result<(), io::Error> {
    let conn = context.graph().conn();
    let op_conn = context.operations().conn();

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

    let mut session = TuiSession::enter()?;
    let terminal = session.terminal_mut();

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
    let mut panel_focus = PanelFocus::new(OperationPanel::Operations);
    let panel_styles = PanelStyles::default();
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
            .block(panel_block(
                "Operations",
                &panel_focus,
                OperationPanel::Operations,
                panel_styles,
            ));

            let outer_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Min(1),
                    Constraint::Length(status_bar_height),
                ])
                .split(f.area());

            let main_area = outer_layout[0];
            let status_bar_area = outer_layout[1];

            let panel_messages = if panel_focus.is_navigation() {
                let mut msg = "*tab/arrows* navigate | *enter* activate panel".to_string();
                if matches!(
                    panel_focus.current(),
                    OperationPanel::MessageEditor | OperationPanel::GraphView
                ) {
                    msg.push_str(" | *x* close panel");
                }
                msg.push_str(" | *q* quit");
                msg
            } else {
                match panel_focus.current() {
                    OperationPanel::Operations => {
                        "*↑↓* select | *e* edit msg | *v* view graph | *esc* leave panel"
                            .to_string()
                    }
                    OperationPanel::MessageEditor => {
                        "*ctrl+s* save | *esc* leave panel".to_string()
                    }
                    OperationPanel::GraphView => {
                        "*tab* cycle block groups | *←→↑↓* pan | *+/-* zoom | *esc* leave panel"
                            .to_string()
                    }
                }
            };

            let msg_editor_block = panel_block(
                "Operation Summary",
                &panel_focus,
                OperationPanel::MessageEditor,
                panel_styles,
            );

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

            if let Some(canvas_area) = canvas_area {
                let graph_title = if blockgroup_graphs.is_empty() {
                    "Change Graph".to_string()
                } else {
                    format!(
                        "Change Graph {name}",
                        name = blockgroup_graphs[selected_blockgroup_graph].1
                    )
                };

                let graph_block = panel_block(
                    graph_title,
                    &panel_focus,
                    OperationPanel::GraphView,
                    panel_styles,
                );
                let inner_canvas = graph_block.inner(canvas_area);

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

            render_status_bar(f, status_bar_area, &panel_messages);

            if view_message_panel
                && panel_focus.is_active_panel(OperationPanel::MessageEditor)
                && let Some((cursor_x, cursor_y)) = textarea.screen_cursor()
            {
                f.set_cursor_position((cursor_x, cursor_y));
            }
        })?;

        if event::poll(std::time::Duration::from_millis(100))?
            && let event::Event::Key(key) = event::read()?
        {
            if panel_focus.is_navigation() {
                match key.code {
                    KeyCode::Tab => {
                        panel_focus.cycle_next();
                    }
                    KeyCode::BackTab => {
                        panel_focus.cycle_prev();
                    }
                    KeyCode::Up => {
                        panel_focus.focus(OperationPanel::Operations);
                    }
                    KeyCode::Down => {
                        if panel_focus.current() == OperationPanel::Operations {
                            if view_message_panel {
                                panel_focus.focus(OperationPanel::MessageEditor);
                            } else if view_graph {
                                panel_focus.focus(OperationPanel::GraphView);
                            }
                        }
                    }
                    KeyCode::Left => {
                        if panel_focus.current() == OperationPanel::GraphView && view_message_panel
                        {
                            panel_focus.focus(OperationPanel::MessageEditor);
                        }
                    }
                    KeyCode::Right => {
                        if panel_focus.current() == OperationPanel::MessageEditor && view_graph {
                            panel_focus.focus(OperationPanel::GraphView);
                        }
                    }
                    KeyCode::Enter => {
                        panel_focus.activate();
                    }
                    KeyCode::Char('x') => match panel_focus.current() {
                        OperationPanel::MessageEditor => {
                            view_message_panel = false;
                            panel_focus.remove_panel(OperationPanel::MessageEditor);
                        }
                        OperationPanel::GraphView => {
                            view_graph = false;
                            panel_focus.remove_panel(OperationPanel::GraphView);
                        }
                        OperationPanel::Operations => {}
                    },
                    KeyCode::Char('q') => {
                        break;
                    }
                    _ => {}
                }
            } else if key.code == KeyCode::Esc {
                panel_focus.deactivate();
            } else {
                match panel_focus.current() {
                    OperationPanel::Operations => match key.code {
                        KeyCode::Up => {
                            if selected > 0 {
                                selected = selected.saturating_sub(1);
                            }
                        }
                        KeyCode::Down => {
                            if selected + 1 < operations.len() {
                                selected += 1;
                            }
                        }
                        KeyCode::Char('e') => {
                            if operation_summaries.is_empty() {
                                continue;
                            }
                            textarea.set_text(&operation_summaries[selected].summary.summary);
                            view_message_panel = true;
                            panel_focus.include_panel(OperationPanel::MessageEditor);
                            panel_focus.focus(OperationPanel::MessageEditor);
                            panel_focus.activate();
                        }
                        KeyCode::Char('v') => {
                            if operation_summaries.is_empty() {
                                continue;
                            }
                            view_graph = true;
                            panel_focus.include_panel(OperationPanel::GraphView);
                            panel_focus.focus(OperationPanel::GraphView);
                            panel_focus.activate();

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
                            graph_controller = if blockgroup_graphs.is_empty() {
                                create_gen_graph_controller(&empty_graph)
                            } else {
                                create_gen_graph_controller(
                                    &blockgroup_graphs[selected_blockgroup_graph].2,
                                )
                            };
                        }
                        _ => {}
                    },
                    OperationPanel::MessageEditor => {
                        if key.code == KeyCode::Char('s')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            if operation_summaries.is_empty() {
                                continue;
                            }
                            let new_summary = textarea.text();
                            let _ = OperationSummary::set_message(
                                op_conn,
                                operation_summaries[selected].summary.id,
                                &new_summary,
                            );
                            operation_summaries[selected].summary.summary = new_summary;
                        } else {
                            let _outcome = rat_text::text_area::handle_events(
                                &mut textarea,
                                true,
                                &crossterm::event::Event::Key(key),
                            );
                        }
                    }
                    OperationPanel::GraphView => {
                        if key.code == KeyCode::Tab || key.code == KeyCode::BackTab {
                            if blockgroup_graphs.is_empty() {
                                continue;
                            }
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
                            graph_controller = create_gen_graph_controller(
                                &blockgroup_graphs[selected_blockgroup_graph].2,
                            );
                        } else {
                            let _ = graph_controller.handle_key_event(key);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
