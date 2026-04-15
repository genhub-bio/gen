use std::{collections::HashMap, io, rc::Rc, time::Instant};

use crossterm::event::{self, KeyCode, KeyModifiers};
use gen_core::PATH_START_NODE_ID;
use gen_diff::operations::{BlockGroupDiff, collect_operation_diff};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    db::DbContext,
    operations::{Operation, OperationSummary},
    traits::Query,
};
use gen_tui::graph_controller::GraphController;
use rat_text::{
    HasScreenCursor,
    text_area::{TextArea, TextAreaState},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    prelude::{StatefulWidget, Style},
    style::Modifier,
    widgets::{Block, List, ListItem, Row, Table},
};
use rusqlite::{params, types::Value};

use crate::{
    config::get_theme_color,
    views::{
        diff_graph::{
            DiffGraphComponent, apply_diff_highlights, block_group_label,
            build_diff_graph_component, highlight_color_for_change_label,
            split_connected_components,
        },
        gen_graph_widget::{
            GenGraphNodeSizer, create_gen_graph_controller, create_gen_graph_widget,
        },
        panels::{PanelFocus, PanelStyles, panel_block, render_status_bar},
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

type OperationDiffComponent = DiffGraphComponent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationPanel {
    Operations,
    MessageEditor,
    GraphView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphViewFocus {
    DiffList,
    GraphCanvas,
}

fn graph_subpanel_border_style(
    panel_focus: &PanelFocus<OperationPanel>,
    active_subpanel: GraphViewFocus,
    subpanel: GraphViewFocus,
    panel_styles: PanelStyles,
) -> Style {
    if panel_focus.is_active_panel(OperationPanel::GraphView) {
        if active_subpanel == subpanel {
            panel_styles.focused
        } else {
            panel_styles.unfocused
        }
    } else if panel_focus.is_navigation_selected(OperationPanel::GraphView) {
        panel_styles.selected
    } else {
        panel_styles.unfocused
    }
}

fn collect_diff_components(
    db_path: &str,
    graph_diffs: &[BlockGroupDiff],
    change_label: &'static str,
) -> Vec<OperationDiffComponent> {
    let highlight_color = highlight_color_for_change_label(change_label);
    let mut components = Vec::new();

    for graph_diff in graph_diffs {
        let base_title = format!(
            "{change_label} [{db_path}] {block_group}",
            block_group = block_group_label(graph_diff)
        );
        let parts = split_connected_components(&graph_diff.graph);
        if parts.len() <= 1 {
            components.push(build_diff_graph_component(
                &graph_diff.graph,
                base_title,
                highlight_color,
            ));
            continue;
        }

        let total = parts.len();
        for (index, diff_graph) in parts.into_iter().enumerate() {
            components.push(build_diff_graph_component(
                &diff_graph,
                format!("{base_title} (part {}/{total})", index + 1),
                highlight_color,
            ));
        }
    }

    components
}

fn load_diff_components_for_operation(
    context: &DbContext,
    operation: &Operation,
) -> Vec<OperationDiffComponent> {
    let op_conn = context.operations().conn();
    let diffs = match collect_operation_diff(
        context.workspace(),
        op_conn,
        operation.parent_hash,
        operation.hash,
        None,
    ) {
        Ok(diffs) => diffs,
        Err(_) => return Vec::new(),
    };
    let mut db_order = diffs.keys().cloned().collect::<Vec<_>>();
    db_order.sort();

    let mut components = Vec::new();
    for db_path in db_order {
        if let Some(diff) = diffs.get(&db_path)
            && let Some(db_diff) = diff.dbs.get(&db_path)
        {
            components.extend(collect_diff_components(
                &db_path,
                &db_diff.added_block_groups,
                "Add",
            ));
            components.extend(collect_diff_components(
                &db_path,
                &db_diff.removed_block_groups,
                "Remove",
            ));
        }
    }

    components
}

fn build_graph_controller(
    components: &[OperationDiffComponent],
    selected_component: usize,
    empty_graph: &GenGraph,
) -> GraphController<GenGraph, GenGraphNodeSizer> {
    if let Some(component) = components.get(selected_component) {
        let mut controller = create_gen_graph_controller(component.graph.clone());
        apply_diff_highlights(&mut controller, component);
        controller
    } else {
        create_gen_graph_controller(empty_graph.clone())
    }
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
    let mut diff_components: Vec<OperationDiffComponent> = vec![];
    let mut selected_diff_component: usize = 0;
    empty_graph.add_node(GraphNode {
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 1,
    });

    let mut graph_controller =
        build_graph_controller(&diff_components, selected_diff_component, &empty_graph);

    let mut view_message_panel = false;
    let mut view_graph = false;
    let mut graph_view_focus = GraphViewFocus::DiffList;
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
                        if graph_view_focus == GraphViewFocus::DiffList {
                            "*↑↓* select diff part | *tab* graph | *esc* leave panel".to_string()
                        } else {
                            "*←→↑↓* pan | *+/-* zoom | *tab* diff list | *esc* leave panel"
                                .to_string()
                        }
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
                let graph_panel = panel_block(
                    "Operation Diff",
                    &panel_focus,
                    OperationPanel::GraphView,
                    panel_styles,
                );
                let panel_inner = graph_panel.inner(canvas_area);
                f.render_widget(graph_panel, canvas_area);

                let graph_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Length(45), Constraint::Min(1)])
                    .split(panel_inner);

                let list_items: Vec<ListItem> = diff_components
                    .iter()
                    .enumerate()
                    .map(|(index, component)| {
                        let item_style = if index == selected_diff_component {
                            Style::default()
                                .fg(ratatui::style::Color::Cyan)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        };
                        ListItem::new(component.title.clone()).style(item_style)
                    })
                    .collect();

                let list_border_style = graph_subpanel_border_style(
                    &panel_focus,
                    graph_view_focus,
                    GraphViewFocus::DiffList,
                    panel_styles,
                );
                let list = List::new(list_items).block(
                    Block::default()
                        .title("Diff Parts")
                        .borders(ratatui::widgets::Borders::ALL)
                        .border_style(list_border_style),
                );
                f.render_widget(list, graph_chunks[0]);

                let graph_title = if diff_components.is_empty() {
                    "No Diff Parts".to_string()
                } else {
                    diff_components[selected_diff_component].title.clone()
                };
                let graph_border_style = graph_subpanel_border_style(
                    &panel_focus,
                    graph_view_focus,
                    GraphViewFocus::GraphCanvas,
                    panel_styles,
                );
                let graph_block = Block::default()
                    .title(graph_title)
                    .borders(ratatui::widgets::Borders::ALL)
                    .border_style(graph_border_style);
                let inner_canvas = graph_block.inner(graph_chunks[1]);
                f.render_widget(graph_block, graph_chunks[1]);

                graph_controller.viewport_state.focus();
                graph_controller.viewport_state.viewport_bounds = inner_canvas;
                graph_controller.update_animations(frame_delta);

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
                            if operation_summaries.is_empty() {
                                continue;
                            }
                            let previous_selected = selected;
                            if selected > 0 {
                                selected = selected.saturating_sub(1);
                            }
                            if view_graph && selected != previous_selected {
                                diff_components = load_diff_components_for_operation(
                                    context,
                                    operation_summaries[selected].operation,
                                );
                                selected_diff_component = 0;
                                graph_controller = build_graph_controller(
                                    &diff_components,
                                    selected_diff_component,
                                    &empty_graph,
                                );
                            }
                        }
                        KeyCode::Down => {
                            if operation_summaries.is_empty() {
                                continue;
                            }
                            let previous_selected = selected;
                            if selected + 1 < operation_summaries.len() {
                                selected += 1;
                            }
                            if view_graph && selected != previous_selected {
                                diff_components = load_diff_components_for_operation(
                                    context,
                                    operation_summaries[selected].operation,
                                );
                                selected_diff_component = 0;
                                graph_controller = build_graph_controller(
                                    &diff_components,
                                    selected_diff_component,
                                    &empty_graph,
                                );
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
                            graph_view_focus = GraphViewFocus::DiffList;
                            panel_focus.include_panel(OperationPanel::GraphView);
                            panel_focus.focus(OperationPanel::GraphView);
                            panel_focus.activate();

                            diff_components = load_diff_components_for_operation(
                                context,
                                operation_summaries[selected].operation,
                            );
                            selected_diff_component = 0;
                            graph_controller = build_graph_controller(
                                &diff_components,
                                selected_diff_component,
                                &empty_graph,
                            );
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
                            graph_view_focus = if graph_view_focus == GraphViewFocus::DiffList {
                                GraphViewFocus::GraphCanvas
                            } else {
                                GraphViewFocus::DiffList
                            };
                        } else if graph_view_focus == GraphViewFocus::DiffList {
                            match key.code {
                                KeyCode::Up => {
                                    if selected_diff_component > 0 {
                                        selected_diff_component -= 1;
                                        graph_controller = build_graph_controller(
                                            &diff_components,
                                            selected_diff_component,
                                            &empty_graph,
                                        );
                                    }
                                }
                                KeyCode::Down => {
                                    if selected_diff_component + 1 < diff_components.len() {
                                        selected_diff_component += 1;
                                        graph_controller = build_graph_controller(
                                            &diff_components,
                                            selected_diff_component,
                                            &empty_graph,
                                        );
                                    }
                                }
                                _ => {}
                            }
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
