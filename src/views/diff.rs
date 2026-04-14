use std::{collections::HashMap, io, time::Instant};

use crossterm::event::{self, Event, KeyCode};
use gen_diff::{
    graph::DiffGenGraph,
    operations::{BlockGroupDiff, OperationDiff},
};
use gen_models::db::GraphConnection;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{List, ListItem},
};

use gen_tui::theme::current_theme;

use crate::{
    views::{
        diff_graph::{
            DiffGraphComponent, apply_diff_highlights, block_group_label,
            build_diff_graph_component, highlight_color_for_change_label,
            split_connected_components,
        },
        gen_graph_widget::{create_gen_graph_controller, create_gen_graph_widget},
        panels::{PanelFocus, PanelStyles, panel_block, render_status_bar},
        tui_runtime::TuiSession,
    },
};

struct DiffComponent {
    render: DiffGraphComponent,
    collection: String,
    sample: String,
    block_group: String,
    part_label: Option<String>,
    change_label: &'static str,
}

struct ListEntry {
    label: String,
    component_index: Option<usize>,
    db_path: String,
    is_header: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffPanel {
    List,
    Graph,
}

pub fn view_diff(
    conn: &GraphConnection,
    diffs: &HashMap<String, OperationDiff>,
) -> Result<(), io::Error> {
    let mut components: Vec<DiffComponent> = vec![];
    let mut components_by_db: HashMap<String, Vec<usize>> = HashMap::new();
    let mut db_order = diffs.keys().cloned().collect::<Vec<_>>();
    db_order.sort();

    for db_path in &db_order {
        if let Some(diff) = diffs.get(db_path)
            && let Some(db_diff) = diff.dbs.get(db_path)
        {
            for component in collect_components(&db_diff.added_block_groups, "Add") {
                let entry = components_by_db.entry(db_path.clone()).or_default();
                entry.push(components.len());
                components.push(component);
            }
            for component in collect_components(&db_diff.removed_block_groups, "Remove") {
                let entry = components_by_db.entry(db_path.clone()).or_default();
                entry.push(components.len());
                components.push(component);
            }
        }
    }

    if components.is_empty() {
        println!("No differences to display.");
        return Ok(());
    }

    let mut session = TuiSession::enter()?;
    let terminal = session.terminal_mut();

    let mut selected = 0usize;
    let mut expanded_db = db_order.first().cloned();
    let mut current_component = 0usize;

    let mut panel_focus = PanelFocus::new(DiffPanel::List);
    panel_focus.include_panel(DiffPanel::Graph);
    let panel_styles = PanelStyles::default();

    let mut graph_controller =
        create_gen_graph_controller(components[current_component].render.graph.clone());
    apply_diff_highlights(&mut graph_controller, &components[current_component].render);

    let mut last_frame_time = Instant::now();

    loop {
        let entries = build_entries(&db_order, &components, &components_by_db, &expanded_db);
        if entries.is_empty() {
            break;
        }
        if selected >= entries.len() {
            selected = 0;
        }
        let desired_component =
            resolve_selected_component(&entries, selected, &components_by_db, expanded_db.as_ref())
                .unwrap_or(0);
        if desired_component != current_component {
            current_component = desired_component;
            graph_controller =
                create_gen_graph_controller(components[current_component].render.graph.clone());
            apply_diff_highlights(&mut graph_controller, &components[current_component].render);
        }

        let now = Instant::now();
        let frame_delta = now.duration_since(last_frame_time);
        last_frame_time = now;

        terminal.draw(|f| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(1)])
                .split(f.area());

            let main = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(45), Constraint::Min(1)])
                .split(outer[0]);

            let list_items: Vec<ListItem> = entries
                .iter()
                .enumerate()
                .map(|(i, entry)| {
                    let style = if i == selected {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    ListItem::new(entry.label.clone()).style(style)
                })
                .collect();

            let list = List::new(list_items).block(panel_block(
                "Diff Parts",
                &panel_focus,
                DiffPanel::List,
                panel_styles,
            ));
            f.render_widget(list, main[0]);

            let graph_title = format!(
                "{} ({}/{})",
                components[current_component].render.title,
                current_component + 1,
                components.len()
            );
            let graph_block =
                panel_block(graph_title, &panel_focus, DiffPanel::Graph, panel_styles);
            let inner_canvas = graph_block.inner(main[1]);

            graph_controller.viewport_state.focus();
            graph_controller.viewport_state.viewport_bounds = inner_canvas;
            graph_controller.update_animations(frame_delta);

            f.render_widget(graph_block, main[1]);

            let canvas_style = Style::default().bg(current_theme()[0x00]);
            let widget = create_gen_graph_widget(conn)
                .detail_level(graph_controller.get_detail_level())
                .style(canvas_style)
                .cursor();
            f.render_stateful_widget(widget, inner_canvas, &mut graph_controller);

            let panel_messages = if panel_focus.is_navigation() {
                "*tab* toggle focus | *enter* activate | *q* quit"
            } else if panel_focus.current() == DiffPanel::List {
                "*↑↓* select | *tab* toggle focus | *esc* leave panel | *q* quit"
            } else {
                "*←→↑↓* pan | *+/-* zoom | *tab* toggle focus | *esc* leave panel | *q* quit"
            };
            render_status_bar(f, outer[1], panel_messages);
        })?;

        if event::poll(std::time::Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            if panel_focus.is_navigation() {
                match key.code {
                    KeyCode::Tab | KeyCode::Left | KeyCode::Right => {
                        panel_focus.cycle_next();
                    }
                    KeyCode::Enter => {
                        panel_focus.activate();
                    }
                    KeyCode::Esc | KeyCode::Char('q') => break,
                    _ => {}
                }
            } else if key.code == KeyCode::Esc {
                panel_focus.deactivate();
            } else if key.code == KeyCode::Tab {
                panel_focus.cycle_next();
            } else if key.code == KeyCode::Char('q') {
                break;
            } else if panel_focus.current() == DiffPanel::List {
                match key.code {
                    KeyCode::Up => {
                        if selected > 0 {
                            selected -= 1;
                            if let Some(entry) = entries.get(selected) {
                                expanded_db = Some(entry.db_path.clone());
                            }
                        }
                    }
                    KeyCode::Down => {
                        if selected + 1 < entries.len() {
                            selected += 1;
                            if let Some(entry) = entries.get(selected) {
                                expanded_db = Some(entry.db_path.clone());
                            }
                        }
                    }
                    _ => {}
                }
            } else {
                let _ = graph_controller.handle_key_event(key);
            }
        }
    }

    Ok(())
}

fn collect_components(graphs: &[BlockGroupDiff], change_label: &'static str) -> Vec<DiffComponent> {
    let mut components = Vec::new();
    let highlight_color = highlight_color_for_change_label(change_label);
    for graph_diff in graphs {
        let parts = split_connected_components(&graph_diff.graph);
        let (collection, sample, block_group) = if let Some(bg) = &graph_diff.block_group {
            (
                bg.collection_name.clone(),
                bg.sample_name.clone(),
                bg.name.clone(),
            )
        } else {
            (
                String::from("Unknown"),
                String::from("Unknown"),
                String::from("Unknown"),
            )
        };
        if parts.len() <= 1 {
            components.push(build_component(
                &graph_diff.graph,
                change_label,
                highlight_color,
                &block_group_label(graph_diff),
                collection,
                sample,
                block_group,
                None,
            ));
        } else {
            let total = parts.len();
            for (idx, diff_graph) in parts.into_iter().enumerate() {
                components.push(build_component(
                    &diff_graph,
                    change_label,
                    highlight_color,
                    &format!(
                        "{} (part {}/{})",
                        block_group_label(graph_diff),
                        idx + 1,
                        total
                    ),
                    collection.clone(),
                    sample.clone(),
                    block_group.clone(),
                    Some(format!("part {}/{}", idx + 1, total)),
                ));
            }
        }
    }
    components
}

#[allow(clippy::too_many_arguments)]
fn build_component(
    diff_graph: &DiffGenGraph,
    change_label: &'static str,
    highlight_color: Color,
    title: &str,
    collection: String,
    sample: String,
    block_group: String,
    part_label: Option<String>,
) -> DiffComponent {
    DiffComponent {
        render: build_diff_graph_component(
            diff_graph,
            format!("{change_label} {title}"),
            highlight_color,
        ),
        collection,
        sample,
        block_group,
        part_label,
        change_label,
    }
}

fn build_entries(
    db_order: &[String],
    components: &[DiffComponent],
    components_by_db: &HashMap<String, Vec<usize>>,
    expanded_db: &Option<String>,
) -> Vec<ListEntry> {
    let mut entries = Vec::new();
    for db_path in db_order {
        entries.push(ListEntry {
            label: db_path.clone(),
            component_index: None,
            db_path: db_path.clone(),
            is_header: true,
        });
        if expanded_db.as_ref() == Some(db_path)
            && let Some(indices) = components_by_db.get(db_path)
        {
            for index in indices {
                let component = &components[*index];
                let part = component
                    .part_label
                    .as_ref()
                    .map(|p| format!(" | {p}"))
                    .unwrap_or_default();
                let label = format!(
                    "  {change} | {collection} | {sample} | {bg}{part}",
                    change = component.change_label,
                    collection = component.collection,
                    sample = component.sample,
                    bg = component.block_group,
                    part = part
                );
                entries.push(ListEntry {
                    label,
                    component_index: Some(*index),
                    db_path: db_path.clone(),
                    is_header: false,
                });
            }
        }
    }
    entries
}

fn resolve_selected_component(
    entries: &[ListEntry],
    selected: usize,
    components_by_db: &HashMap<String, Vec<usize>>,
    expanded_db: Option<&String>,
) -> Option<usize> {
    if let Some(entry) = entries.get(selected) {
        if let Some(index) = entry.component_index {
            return Some(index);
        }
        if entry.is_header
            && let Some(indices) = components_by_db.get(&entry.db_path)
        {
            return indices.first().copied();
        }
    }
    if let Some(db_path) = expanded_db
        && let Some(indices) = components_by_db.get(db_path)
    {
        return indices.first().copied();
    }
    None
}
