use std::{io, time::Duration};

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

pub fn view_diff(conn: &Connection, diff: &OperationDiff) -> Result<(), io::Error> {
    let mut components: Vec<DiffComponent> = vec![];
    collect_components(&diff.added_block_groups, "Add", &mut components);
    collect_components(&diff.removed_block_groups, "Remove", &mut components);

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
    let mut graph_focus = false;
    let mut viewer = make_viewer(conn, &components[selected].graph, PlotParameters::default());

    let result = (|| -> Result<(), io::Error> {
        loop {
            terminal.draw(|f| {
                let outer = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(1), Constraint::Length(1)])
                    .split(f.area());

                let main = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Length(45), Constraint::Min(1)])
                    .split(outer[0]);

                let list_items: Vec<ListItem> = components
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let part = c
                            .part_label
                            .as_ref()
                            .map(|p| format!(" | {p}"))
                            .unwrap_or_default();
                        let content = format!(
                            "{change} | {collection} | {sample} | {bg}{part}",
                            change = c.change_label,
                            collection = c.collection,
                            sample = c.sample,
                            bg = c.block_group,
                            part = part
                        );
                        let style = if i == selected {
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        };
                        ListItem::new(content).style(style)
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
                            components[selected].title,
                            selected + 1,
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
                            viewer = make_viewer(
                                conn,
                                &components[selected].graph,
                                PlotParameters::default(),
                            );
                        }
                    }
                    KeyCode::Down if !graph_focus => {
                        if selected + 1 < components.len() {
                            selected += 1;
                            viewer = make_viewer(
                                conn,
                                &components[selected].graph,
                                PlotParameters::default(),
                            );
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
    components: &mut Vec<DiffComponent>,
) {
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
                });
            }
        }
    }
}
