use std::{collections::HashMap, io, time::Instant};

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_diff::{
    graph::{DiffGenGraph, DiffGenGraphRef, DiffGraphNode},
    operations::{BlockGroupDiff, OperationDiff},
};
use gen_models::db::GraphConnection;
use gen_tui::{graph_controller::GraphController, plotter::PathStyle};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::{
    config::get_theme_color,
    views::{
        gen_graph_widget::{
            GenGraphNodeSizer, create_gen_graph_controller, create_gen_graph_widget,
        },
        helpers::{install_tui_panic_hook, style_text},
    },
};

struct DiffComponent {
    title: String,
    collection: String,
    sample: String,
    block_group: String,
    part_label: Option<String>,
    graph: gen_graph::GenGraph,
    highlight_nodes: Vec<gen_graph::GraphNode>,
    highlight_edges: Vec<(gen_graph::GraphNode, gen_graph::GraphNode)>,
    highlight_color: Color,
    change_label: &'static str,
}

struct ListEntry {
    label: String,
    component_index: Option<usize>,
    db_path: String,
    is_header: bool,
}

/// Apply diff highlights (nodes and edges) to a graph controller.
///
/// This is the only direct GraphController mutation beyond setup — it configures
/// per-node and per-edge highlights for disconnected diff regions that can't use
/// set_path_highlight.
fn apply_diff_highlights(
    controller: &mut GraphController<&gen_graph::GenGraph, GenGraphNodeSizer>,
    component: &DiffComponent,
) {
    let style = PathStyle::new(component.highlight_color)
        .with_line_style(gen_tui::plotter::LineStyle::Bold)
        .with_merge_glyphs(true);
    for &node in &component.highlight_nodes {
        controller.set_node_highlight(node, style);
    }
    for &(src, tgt) in &component.highlight_edges {
        controller.set_edge_highlight((src, tgt), style);
    }
}

/// This splits a graph into its connected components. It's needed because a change may happen in the middle of a graph where no
/// start/end nodes are present and the viewer crashes without splitting it.
fn split_connected_components(graph: &DiffGenGraph) -> Vec<DiffGenGraph> {
    use std::collections::HashSet;

    use petgraph::Direction;

    let mut visited: HashSet<DiffGraphNode> = HashSet::new();
    let mut components = vec![];

    for node in graph.nodes() {
        if visited.contains(&node) {
            continue;
        }
        let mut stack = vec![node];
        let mut component_nodes: HashSet<DiffGraphNode> = HashSet::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            component_nodes.insert(current);
            for neighbor in graph
                .neighbors_directed(current, Direction::Outgoing)
                .chain(graph.neighbors_directed(current, Direction::Incoming))
            {
                if !visited.contains(&neighbor) {
                    stack.push(neighbor);
                }
            }
        }

        let mut subgraph = DiffGenGraph::new();
        for n in &component_nodes {
            subgraph.add_node(*n);
        }
        for (src, dest, edges) in graph.all_edges() {
            if component_nodes.contains(&src) && component_nodes.contains(&dest) {
                subgraph.add_edge(src, dest, edges.clone());
            }
        }
        components.push(subgraph);
    }

    components
}

fn block_group_label(diff: &BlockGroupDiff) -> String {
    if let Some(bg) = &diff.block_group {
        format!(
            "{collection} {sample} {name}",
            collection = bg.collection_name,
            sample = bg
                .sample_name
                .clone()
                .unwrap_or_else(|| "Reference".to_string()),
            name = bg.name
        )
    } else {
        format!("BlockGroup {}", diff.id)
    }
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

    install_tui_panic_hook();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut selected = 0usize;
    let mut expanded_db = db_order.first().cloned();
    let mut current_component = 0usize;

    // Panel focus: two-mode system matching operations view
    let mut panel_focus = "diff_list"; // "diff_list" or "graph_view"
    let mut panel_activated = true; // Start with diff list activated

    // Three-state border styles matching operations view
    let focused_style = Style::default()
        .fg(Color::Blue)
        .add_modifier(Modifier::BOLD);
    let selected_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);
    let unfocused_style = Style::default().fg(Color::Gray);

    // Setup: create controller and apply highlights for initial component
    let mut graph_controller = create_gen_graph_controller(&components[current_component].graph);
    apply_diff_highlights(&mut graph_controller, &components[current_component]);

    let mut last_frame_time = Instant::now();

    let result = (|| -> Result<(), io::Error> {
        loop {
            let entries = build_entries(&db_order, &components, &components_by_db, &expanded_db);
            if entries.is_empty() {
                break;
            }
            if selected >= entries.len() {
                selected = 0;
            }
            let desired_component = resolve_selected_component(
                &entries,
                selected,
                &components_by_db,
                expanded_db.as_ref(),
            )
            .unwrap_or(0);
            if desired_component != current_component {
                current_component = desired_component;
                // Setup: create new controller and apply highlights on component switch
                graph_controller =
                    create_gen_graph_controller(&components[current_component].graph);
                apply_diff_highlights(&mut graph_controller, &components[current_component]);
            }

            // Calculate frame delta for smooth animations
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

                // Left panel: diff list
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

                let list_title = if !panel_activated && panel_focus == "diff_list" {
                    "[Diff Parts]"
                } else {
                    "Diff Parts"
                };

                let list_border_style = if panel_activated && panel_focus == "diff_list" {
                    focused_style
                } else if !panel_activated && panel_focus == "diff_list" {
                    selected_style
                } else {
                    unfocused_style
                };

                let list = List::new(list_items).block(
                    Block::default()
                        .title(list_title)
                        .borders(Borders::ALL)
                        .border_style(list_border_style),
                );
                f.render_widget(list, main[0]);

                // Right panel: graph via widget
                let graph_title_text = format!(
                    "{} ({}/{})",
                    components[current_component].title,
                    current_component + 1,
                    components.len()
                );
                let graph_title = if !panel_activated && panel_focus == "graph_view" {
                    format!("[{}]", graph_title_text)
                } else {
                    graph_title_text
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
                let inner_canvas = graph_block.inner(main[1]);

                // Pre-render animation tick
                graph_controller.viewport_state.focus();
                graph_controller.viewport_state.viewport_bounds = inner_canvas;
                graph_controller.update_animations(frame_delta);

                f.render_widget(graph_block, main[1]);

                let canvas_style = Style::default().bg(get_theme_color("canvas").unwrap());
                let widget = create_gen_graph_widget(conn)
                    .detail_level(graph_controller.get_detail_level())
                    .style(canvas_style)
                    .cursor();
                f.render_stateful_widget(widget, inner_canvas, &mut graph_controller);

                // Status bar with themed styling
                let status_bar_area = outer[1];
                let panel_messages = if !panel_activated {
                    "*tab* toggle focus | *enter* activate | *q* quit"
                } else if panel_focus == "diff_list" {
                    "*↑↓* select | *tab* toggle focus | *esc* leave panel | *q* quit"
                } else {
                    "*←→↑↓* pan | *+/-* zoom | *tab* toggle focus | *esc* leave panel | *q* quit"
                };

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
            })?;

            if event::poll(std::time::Duration::from_millis(100))?
                && let Event::Key(key) = event::read()?
            {
                if !panel_activated {
                    // NAVIGATION MODE: Tab to toggle focus, Enter to activate, q to quit
                    match key.code {
                        KeyCode::Tab | KeyCode::Left | KeyCode::Right => {
                            panel_focus = if panel_focus == "diff_list" {
                                "graph_view"
                            } else {
                                "diff_list"
                            };
                        }
                        KeyCode::Enter => {
                            panel_activated = true;
                        }
                        KeyCode::Esc | KeyCode::Char('q') => break,
                        _ => {}
                    }
                } else {
                    // ACTIVE MODE: Esc to deactivate; panel-specific controls
                    if key.code == KeyCode::Esc {
                        panel_activated = false;
                    } else if key.code == KeyCode::Tab {
                        // Toggle focus between panels while staying activated
                        panel_focus = if panel_focus == "diff_list" {
                            "graph_view"
                        } else {
                            "diff_list"
                        };
                    } else if key.code == KeyCode::Char('q') {
                        break;
                    } else if panel_focus == "diff_list" {
                        // List panel: Up/Down to select
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
                        // Graph panel: delegate to controller
                        let _ = graph_controller.handle_key_event(key);
                    }
                }
            }
        }
        Ok(())
    })();

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    result
}

fn collect_components(graphs: &[BlockGroupDiff], change_label: &'static str) -> Vec<DiffComponent> {
    let mut components = Vec::new();
    let highlight_color = match change_label {
        "Add" => Color::Green,
        "Remove" => Color::Red,
        _ => Color::White,
    };
    for graph_diff in graphs {
        let parts = split_connected_components(&graph_diff.graph);
        let (collection, sample, block_group) = if let Some(bg) = &graph_diff.block_group {
            (
                bg.collection_name.clone(),
                bg.sample_name
                    .clone()
                    .unwrap_or_else(|| "Reference".to_string()),
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
            let diff_graph = graph_diff.graph.clone();
            components.push(build_component(
                &diff_graph,
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
    let graph: gen_graph::GenGraph = DiffGenGraphRef(diff_graph).into();
    let highlight_edges = collect_highlight_edges(diff_graph);
    let highlight_nodes = collect_highlight_nodes(diff_graph);
    DiffComponent {
        title: format!("{change_label} {title}"),
        collection,
        sample,
        block_group,
        part_label,
        graph,
        highlight_nodes,
        highlight_edges,
        highlight_color,
        change_label,
    }
}

/// Collect edges that are new in the diff graph, returning them as a flat
/// vector of (source, target) pairs suitable for set_edge_highlight.
fn collect_highlight_edges(
    diff_graph: &DiffGenGraph,
) -> Vec<(gen_graph::GraphNode, gen_graph::GraphNode)> {
    let mut edges = Vec::new();
    for (src, dest, edge_data) in diff_graph.all_edges() {
        if edge_data.iter().any(|edge| edge.is_new) {
            edges.push((src.node, dest.node));
        }
    }
    edges
}

/// Collect nodes that are new in the diff graph, returning them as a flat
/// vector suitable for set_node_highlight.
fn collect_highlight_nodes(diff_graph: &DiffGenGraph) -> Vec<gen_graph::GraphNode> {
    diff_graph
        .nodes()
        .filter_map(|node| node.is_new.then_some(node.node))
        .collect()
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
