use std::{collections::HashMap, io, time::Duration};

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_models::traits::Query;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::Text,
    widgets::{Block, Borders, List, ListItem, Paragraph},
};
use rusqlite::Connection;

use crate::{
    core::HashId,
    diffs::operations::{BlockGroupDiff, OperationDiff},
    graph::connect_all_boundary_edges,
    models::node::Node,
    views::block_group_viewer::{PlotParameters, Viewer},
};

struct DiffComponent {
    title: String,
    collection: String,
    sample: String,
    block_group: String,
    part_label: Option<String>,
    graph: gen_graph::GenGraph,
    change_label: &'static str,
    db_path: String,
}

struct ListEntry {
    label: String,
    component_index: Option<usize>,
    db_path: String,
    is_header: bool,
}

fn split_connected_components(graph: &gen_graph::GenGraph) -> Vec<gen_graph::GenGraph> {
    use std::collections::HashSet;

    use gen_graph::GraphNode;
    use petgraph::Direction;

    let mut visited: HashSet<GraphNode> = HashSet::new();
    let mut components = vec![];

    for node in graph.nodes() {
        if visited.contains(&node) {
            continue;
        }
        let mut stack = vec![node];
        let mut component_nodes: HashSet<GraphNode> = HashSet::new();
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

        let mut subgraph = gen_graph::GenGraph::new();
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

fn choose_origin(conn: &Connection, graph: &gen_graph::GenGraph) -> (Node, i64) {
    if let Some(start_block) = graph
        .nodes()
        .find(|n| n.node_id == gen_core::PATH_START_NODE_ID)
    {
        return (Node::get_start_node(), start_block.sequence_start);
    }

    if let Some(first) = graph.nodes().next() {
        let node = Node::get(
            conn,
            "select * from nodes where id = ?1",
            rusqlite::params![first.node_id],
        )
        .unwrap_or(Node {
            id: first.node_id,
            sequence_hash: HashId::pad_str(0),
        });
        return (node, first.sequence_start);
    }

    (Node::get_start_node(), 0)
}

fn make_viewer<'a>(
    conn: &'a Connection,
    graph: &'a gen_graph::GenGraph,
    params: PlotParameters,
) -> Viewer<'a> {
    let origin = choose_origin(conn, graph);
    Viewer::with_origin(graph, conn, params, origin)
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
    conn: &Connection,
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
            for component in
                collect_components(&db_diff.added_block_groups, "Add", &db_diff.db_path)
            {
                let entry = components_by_db.entry(db_path.clone()).or_default();
                entry.push(components.len());
                components.push(component);
            }
            for component in
                collect_components(&db_diff.removed_block_groups, "Remove", &db_diff.db_path)
            {
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
    let mut current_component = 0usize;
    let mut viewer = make_viewer(
        conn,
        &components[current_component].graph,
        PlotParameters::default(),
    );

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
                viewer = make_viewer(
                    conn,
                    &components[current_component].graph,
                    PlotParameters::default(),
                );
            }

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

                viewer.set_block(
                    Block::default()
                        .title(format!(
                            "{} ({}/{})",
                            components[current_component].title,
                            current_component + 1,
                            components.len()
                        ))
                        .borders(Borders::ALL)
                        .border_style(if graph_focus {
                            Style::default().fg(Color::Blue)
                        } else {
                            Style::default()
                        }),
                );
                viewer.has_focus = graph_focus;
                viewer.draw(f, main[1]);

                let status = Paragraph::new(Text::styled(
                    format!(
                        "↑/↓ select | tab/enter toggle focus | shift+tab list | graph: {} | q to exit",
                        Viewer::get_status_line()
                    ),
                    Style::default().bg(Color::DarkGray).fg(Color::White),
                ));
                f.render_widget(status, outer[1]);
            })?;

            if event::poll(Duration::from_millis(100))?
                && let Event::Key(key) = event::read()?
            {
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') => break,
                    KeyCode::Tab | KeyCode::Enter => {
                        graph_focus = true;
                        viewer.has_focus = true;
                    }
                    KeyCode::BackTab => {
                        graph_focus = false;
                        viewer.has_focus = false;
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
                            viewer.handle_input(key);
                        }
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

fn collect_components(
    graphs: &[BlockGroupDiff],
    change_label: &'static str,
    db_path: &str,
) -> Vec<DiffComponent> {
    let mut components = Vec::new();
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
            let mut graph = graph_diff.graph.clone();
            connect_all_boundary_edges(&mut graph);
            components.push(DiffComponent {
                title: format!("{change_label} {}", block_group_label(graph_diff)),
                collection,
                sample,
                block_group,
                part_label: None,
                graph,
                change_label,
                db_path: db_path.to_string(),
            });
        } else {
            let total = parts.len();
            for (idx, mut graph) in parts.into_iter().enumerate() {
                connect_all_boundary_edges(&mut graph);
                components.push(DiffComponent {
                    title: format!(
                        "{change_label} {} (part {}/{})",
                        block_group_label(graph_diff),
                        idx + 1,
                        total
                    ),
                    collection: collection.clone(),
                    sample: sample.clone(),
                    block_group: block_group.clone(),
                    part_label: Some(format!("part {}/{}", idx + 1, total)),
                    graph,
                    change_label,
                    db_path: db_path.to_string(),
                });
            }
        }
    }
    components
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
