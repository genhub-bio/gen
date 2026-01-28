use std::{collections::HashMap, io, time::Duration};

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_diff::{
    graph::{DiffGenGraph, DiffGenGraphRef, DiffGraphNode},
    operations::{BlockGroupDiff, OperationDiff},
};
use gen_models::{db::GraphConnection, traits::Query};
use petgraph::graphmap::DiGraphMap;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::Text,
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::{core::HashId, models::node::Node};

// Temporary no-op stubs to keep build green while TUI parts are disabled
#[derive(Clone)]
struct PlotParameters;
impl PlotParameters {
    fn default() -> Self {
        PlotParameters
    }
}

struct Viewer<'a> {
    has_focus: bool,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Viewer<'a> {
    fn with_origin(
        _graph: &'a gen_graph::GenGraph,
        _conn: &'a GraphConnection,
        _params: PlotParameters,
        _origin: (Node, i64),
    ) -> Self {
        let _ = _origin;
        Viewer {
            has_focus: false,
            _phantom: std::marker::PhantomData,
        }
    }

    fn get_status_line() -> &'static str {
        "diff viewer"
    }

    // Accept the shapes used below and ignore them
    fn set_highlights(
        &mut self,
        _highlights: Vec<(
            Color,
            petgraph::graphmap::DiGraphMap<gen_graph::GraphNode, ()>,
        )>,
    ) {
    }

    fn set_node_highlights(
        &mut self,
        _highlights: Vec<(Color, std::collections::HashSet<gen_graph::GraphNode>)>,
    ) {
    }

    // Generic args so the signature stays permissive
    fn set_block<T>(&mut self, _block: T) {}

    fn draw<T>(&self, _f: &mut T, _area: ratatui::layout::Rect) {}

    fn handle_input<T>(&mut self, _key: T) {}
}

struct DiffComponent {
    title: String,
    collection: String,
    sample: String,
    block_group: String,
    part_label: Option<String>,
    graph: gen_graph::GenGraph,
    highlight_graph: DiGraphMap<gen_graph::GraphNode, ()>,
    highlight_nodes: std::collections::HashSet<gen_graph::GraphNode>,
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

/// This positions the viewer on either the start node if it exists, or the first node it can find otherwise.
fn choose_origin(conn: &GraphConnection, graph: &gen_graph::GenGraph) -> (Node, i64) {
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
    conn: &'a GraphConnection,
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
    let mut current_component = 0usize;
    let mut viewer = make_viewer(
        conn,
        &components[current_component].graph,
        PlotParameters::default(),
    );
    viewer.set_highlights(vec![(
        components[current_component].highlight_color,
        components[current_component].highlight_graph.clone(),
    )]);
    viewer.set_node_highlights(vec![(
        components[current_component].highlight_color,
        components[current_component].highlight_nodes.clone(),
    )]);

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
                viewer.set_highlights(vec![(
                    components[current_component].highlight_color,
                    components[current_component].highlight_graph.clone(),
                )]);
                viewer.set_node_highlights(vec![(
                    components[current_component].highlight_color,
                    components[current_component].highlight_nodes.clone(),
                )]);
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
                        "↑/↓ select | tab toggle focus | graph: {} | q to exit",
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
                    KeyCode::Tab => {
                        graph_focus = !graph_focus;
                        viewer.has_focus = graph_focus;
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

/// This function looks a little odd because it goes into the existing `highlights` of block_group_viewer, where nodes are defined
/// by HashId, start, end. So we are not actually losing our edges here because the nodes contain the sequence start/end and have been
/// split up into their own nodes by this point.
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

fn build_node_highlights(
    diff_graph: &DiffGenGraph,
) -> std::collections::HashSet<gen_graph::GraphNode> {
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
