use std::collections::HashSet;

use gen_diff::{
    graph::{DiffGenGraph, DiffGenGraphRef, DiffGraphNode},
    operations::BlockGroupDiff,
};
use gen_graph::{GenGraph, GraphNode};
use gen_tui::{LineStyle, graph_controller::GraphController, plotter::PathStyle};
use petgraph::Direction;
use ratatui::style::Color;

use crate::views::gen_graph_widget::GenGraphNodeSizer;

#[derive(Clone)]
pub struct DiffGraphComponent {
    pub title: String,
    pub graph: GenGraph,
    pub highlight_nodes: Vec<GraphNode>,
    pub highlight_edges: Vec<(GraphNode, GraphNode)>,
    pub highlight_color: Color,
}

/// Apply diff highlights (nodes and edges) to a graph controller.
pub fn apply_diff_highlights(
    controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    component: &DiffGraphComponent,
) {
    let style = PathStyle::new(component.highlight_color)
        .with_line_style(LineStyle::Bold)
        .with_merge_glyphs(true);
    for &node in &component.highlight_nodes {
        controller.set_node_highlight(node, style);
    }
    for &(src, target) in &component.highlight_edges {
        controller.set_edge_highlight((src, target), style);
    }
}

/// Split a diff graph into connected components to keep the viewer stable when
/// a change region doesn't include start/end nodes.
pub fn split_connected_components(graph: &DiffGenGraph) -> Vec<DiffGenGraph> {
    let mut visited: HashSet<DiffGraphNode> = HashSet::new();
    let mut components = Vec::new();

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
        for graph_node in &component_nodes {
            subgraph.add_node(*graph_node);
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

pub fn block_group_label(diff: &BlockGroupDiff) -> String {
    if let Some(bg) = &diff.block_group {
        format!(
            "{collection} {sample} {name}",
            collection = bg.collection_name,
            sample = bg.sample_name.clone(),
            name = bg.name
        )
    } else {
        format!("BlockGroup {}", diff.id)
    }
}

pub fn highlight_color_for_change_label(change_label: &str) -> Color {
    match change_label {
        "Add" => Color::Green,
        "Remove" => Color::Red,
        _ => Color::White,
    }
}

pub fn build_diff_graph_component(
    diff_graph: &DiffGenGraph,
    title: String,
    highlight_color: Color,
) -> DiffGraphComponent {
    let graph: GenGraph = DiffGenGraphRef(diff_graph).into();
    let highlight_edges = collect_highlight_edges(diff_graph);
    let highlight_nodes = collect_highlight_nodes(diff_graph);
    DiffGraphComponent {
        title,
        graph,
        highlight_nodes,
        highlight_edges,
        highlight_color,
    }
}

fn collect_highlight_edges(diff_graph: &DiffGenGraph) -> Vec<(GraphNode, GraphNode)> {
    let mut edges = Vec::new();
    for (src, dest, edge_data) in diff_graph.all_edges() {
        if edge_data.iter().any(|edge| edge.is_new) {
            edges.push((src.node, dest.node));
        }
    }
    edges
}

fn collect_highlight_nodes(diff_graph: &DiffGenGraph) -> Vec<GraphNode> {
    diff_graph
        .nodes()
        .filter_map(|node| node.is_new.then_some(node.node))
        .collect()
}
