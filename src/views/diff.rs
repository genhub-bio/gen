use std::{
    collections::{HashMap, HashSet},
    io,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_diff::{
    graph::{DiffGenGraph, DiffGenGraphRef, DiffGraphNode},
    operations::{BlockGroupDiff, OperationDiff},
};
use gen_models::db::GraphConnection;
use gen_tui::{
    graph_controller::GraphController, layout::VisualDetail, plotter::PathStyle, theme::Theme,
};
use petgraph::graphmap::DiGraphMap;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::Text,
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::{
    config::get_theme_color,
    views::gen_graph_widget::{GenGraphNodeSizer, create_gen_graph_widget},
};

struct DiffComponent {
    title: String,
    collection: String,
    sample: String,
    block_group: String,
    part_label: Option<String>,
    graph: gen_graph::GenGraph,
    highlight_graph: DiGraphMap<gen_graph::GraphNode, ()>,
    highlight_nodes: HashSet<gen_graph::GraphNode>,
    highlight_color: Color,
    change_label: &'static str,
}

struct ListEntry {
    label: String,
    component_index: Option<usize>,
    db_path: String,
    is_header: bool,
}

/// This splits a graph into its connected components. It's needed because a change may happen in the middle of a graph where no
/// start/end nodes are present and the viewer crashes without splitting it.
fn split_connected_components(graph: &DiffGenGraph) -> Vec<DiffGenGraph> {
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

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut selected = 0usize;
    let mut expanded_db = db_order.first().cloned();
    let mut graph_focus = false;
    let mut current_component_idx = 0usize;

    // Initial controller setup
    let node_sizer = GenGraphNodeSizer;
    let mut graph_controller =
        GraphController::new(&components[current_component_idx].graph, node_sizer).with_theme(
            Theme {
                canvas: get_theme_color("canvas").unwrap(),
                node_fg: get_theme_color("text").unwrap(),
                node_bg: get_theme_color("node").unwrap(),
                edge_fg: get_theme_color("edge").unwrap(),
                edge_bg: get_theme_color("canvas").unwrap(),
                cursor_fg: get_theme_color("cursor_fg").unwrap(),
                cursor_bg: get_theme_color("cursor_bg").unwrap(),
            },
        );
    graph_controller.set_detail_level(VisualDetail::Truncated);
    graph_controller.show_cursor();

    // Apply initial highlights
    apply_component_highlights(&mut graph_controller, &components[current_component_idx]);

    let mut last_frame_time = Instant::now();
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

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

        if desired_component != current_component_idx {
            current_component_idx = desired_component;

            // Recreate controller for new component
            let node_sizer = GenGraphNodeSizer;
            graph_controller =
                GraphController::new(&components[current_component_idx].graph, node_sizer)
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

            apply_component_highlights(&mut graph_controller, &components[current_component_idx]);
        }

        // Calculate frame delta
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

            let list = List::new(list_items).block(
                Block::default()
                    .title("Diff Parts")
                    .borders(Borders::ALL)
                    .border_style(if graph_focus {
                        Style::default()
                    } else {
                        Style::default().fg(Color::Blue)
                    }),
            );
            f.render_widget(list, main[0]);

            // Render Graph
            let graph_block = Block::default()
                .title(format!(
                    "{} ({}/{})",
                    components[current_component_idx].title,
                    current_component_idx + 1,
                    components.len()
                ))
                .borders(Borders::ALL)
                .border_style(if graph_focus {
                    Style::default().fg(Color::Blue)
                } else {
                    Style::default()
                });

            let canvas_area = graph_block.inner(main[1]);
            f.render_widget(graph_block, main[1]);

            // Update viewport
            graph_controller.viewport_state.focus();
            graph_controller.viewport_state.viewport_bounds = canvas_area;
            graph_controller.update_animations(frame_delta);

            let canvas_style = Style::default().bg(get_theme_color("canvas").unwrap());
            let widget = create_gen_graph_widget(conn)
                .detail_level(graph_controller.get_detail_level())
                .style(canvas_style)
                .cursor();
            f.render_stateful_widget(widget, canvas_area, &mut graph_controller);

            let status = Paragraph::new(Text::styled(
                "↑/↓ select | tab toggle focus | graph: ←→↑↓ pan, +/- zoom | q to exit",
                Style::default().bg(Color::DarkGray).fg(Color::White),
            ));
            f.render_widget(status, outer[1]);
        })?;

        // Input handling
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => break,
                KeyCode::Tab => {
                    graph_focus = !graph_focus;
                }
                KeyCode::Up if !graph_focus => {
                    if selected > 0 {
                        selected -= 1;
                        if let Some(entry) = entries.get(selected) {
                            expanded_db = Some(entry.db_path.clone());
                        }
                    }
                }
                KeyCode::Down if !graph_focus => {
                    if selected + 1 < entries.len() {
                        selected += 1;
                        if let Some(entry) = entries.get(selected) {
                            expanded_db = Some(entry.db_path.clone());
                        }
                    }
                }
                _ => {
                    if graph_focus {
                        let _ = graph_controller.handle_key_event(key);
                    }
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn apply_component_highlights(
    controller: &mut GraphController<&gen_graph::GenGraph, GenGraphNodeSizer>,
    component: &DiffComponent,
) {
    let highlight_style = PathStyle::new(component.highlight_color);

    // Highlight edges
    for (src, dst, _) in component.highlight_graph.all_edges() {
        controller.set_edge_highlight((src, dst), highlight_style);
    }

    // Highlight nodes
    for node in &component.highlight_nodes {
        controller.set_node_highlight(*node, highlight_style);
    }
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
    let highlight_graph = build_edge_highlight_graph(diff_graph);
    let highlight_nodes = build_node_highlights(diff_graph);
    DiffComponent {
        title: format!("{change_label} {title}"),
        collection,
        sample,
        block_group,
        part_label,
        graph,
        highlight_graph,
        highlight_nodes,
        highlight_color,
        change_label,
    }
}

fn build_edge_highlight_graph(diff_graph: &DiffGenGraph) -> DiGraphMap<gen_graph::GraphNode, ()> {
    let mut highlight_graph = DiGraphMap::new();
    for (src, dest, edges) in diff_graph.all_edges() {
        if edges.iter().any(|edge| edge.is_new) {
            highlight_graph.add_node(src.node);
            highlight_graph.add_node(dest.node);
            highlight_graph.add_edge(src.node, dest.node, ());
        }
    }
    highlight_graph
}

fn build_node_highlights(diff_graph: &DiffGenGraph) -> HashSet<gen_graph::GraphNode> {
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
