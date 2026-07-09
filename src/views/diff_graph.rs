use std::collections::HashSet;

use gen_diff::{
    graph::{DiffChangeKind, DiffGenGraph, DiffGenGraphRef, DiffGraphNode},
    operations::{BlockGroupChangeKind, BlockGroupDiff},
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
    pub highlighted_nodes: Vec<(GraphNode, Color)>,
    pub highlighted_edges: Vec<((GraphNode, GraphNode), Color)>,
}

/// Apply diff highlights (nodes and edges) to a graph controller.
pub fn apply_diff_highlights(
    controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    component: &DiffGraphComponent,
) {
    for &(node, color) in &component.highlighted_nodes {
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        controller.set_node_highlight(node, style);
    }
    for &((src, target), color) in &component.highlighted_edges {
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
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
    if let Some(bg) = diff
        .target_block_group
        .as_ref()
        .or(diff.source_block_group.as_ref())
    {
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

struct GraphChangePresence {
    has_added: bool,
    has_removed: bool,
    has_modified: bool,
}

impl GraphChangePresence {
    fn label(&self) -> &'static str {
        if self.has_added && self.has_removed {
            "Change"
        } else if self.has_added {
            "Add"
        } else if self.has_removed {
            "Remove"
        } else if self.has_modified {
            "Modify"
        } else {
            "Unchanged"
        }
    }
}

pub fn change_label_for_graph(diff_graph: &DiffGenGraph) -> &'static str {
    GraphChangePresence {
        has_added: diff_graph
            .nodes()
            .any(|node| node.change.kind == DiffChangeKind::Added)
            || diff_graph.all_edges().any(|(_, _, edges)| {
                edges
                    .iter()
                    .any(|edge| edge.change.kind == DiffChangeKind::Added)
            }),
        has_removed: diff_graph
            .nodes()
            .any(|node| node.change.kind == DiffChangeKind::Removed)
            || diff_graph.all_edges().any(|(_, _, edges)| {
                edges
                    .iter()
                    .any(|edge| edge.change.kind == DiffChangeKind::Removed)
            }),
        has_modified: diff_graph
            .nodes()
            .any(|node| node.change.kind == DiffChangeKind::Modified)
            || diff_graph.all_edges().any(|(_, _, edges)| {
                edges
                    .iter()
                    .any(|edge| edge.change.kind == DiffChangeKind::Modified)
            }),
    }
    .label()
}

pub fn change_label_for_block_group(diff: &BlockGroupDiff) -> &'static str {
    let graph_label = change_label_for_graph(&diff.graph);
    if graph_label != "Unchanged" {
        return graph_label;
    }
    match diff.change_kind() {
        Some(BlockGroupChangeKind::Added) => "Created",
        Some(BlockGroupChangeKind::Removed) => "Removed",
        Some(BlockGroupChangeKind::Modified) => graph_label,
        None => "Unknown",
    }
}

pub fn build_diff_graph_component(diff_graph: &DiffGenGraph, title: String) -> DiffGraphComponent {
    let graph: GenGraph = DiffGenGraphRef(diff_graph).into();
    let highlighted_edges = collect_highlight_edges(diff_graph);
    let highlighted_nodes = collect_highlight_nodes(diff_graph);
    DiffGraphComponent {
        title,
        graph,
        highlighted_nodes,
        highlighted_edges,
    }
}

fn collect_highlight_edges(diff_graph: &DiffGenGraph) -> Vec<((GraphNode, GraphNode), Color)> {
    let mut edges = Vec::new();
    for (src, dest, edge_data) in diff_graph.all_edges() {
        if edge_data
            .iter()
            .any(|edge| edge.change.kind == DiffChangeKind::Added)
        {
            edges.push(((src.node, dest.node), Color::Green));
        }
        if edge_data
            .iter()
            .any(|edge| edge.change.kind == DiffChangeKind::Removed)
        {
            edges.push(((src.node, dest.node), Color::Red));
        }
        if edge_data
            .iter()
            .any(|edge| edge.change.kind == DiffChangeKind::Modified)
        {
            edges.push(((src.node, dest.node), Color::Yellow));
        }
    }
    edges
}

fn collect_highlight_nodes(diff_graph: &DiffGenGraph) -> Vec<(GraphNode, Color)> {
    diff_graph
        .nodes()
        .filter_map(|node| match node.change.kind {
            DiffChangeKind::Added => Some((node.node, Color::Green)),
            DiffChangeKind::Removed => Some((node.node, Color::Red)),
            DiffChangeKind::Modified => Some((node.node, Color::Yellow)),
            DiffChangeKind::Unchanged => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use gen_core::{DoltHashId, HashId, Strand};
    use gen_diff::graph::{DiffChange, DiffChangeKind, DiffGenGraph, DiffGraphEdge, DiffGraphNode};
    use gen_graph::{GraphEdge, GraphNode};

    use super::split_connected_components;

    fn graph_node(id: i64, start: i64, end: i64, is_new: bool) -> DiffGraphNode {
        DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(id),
                sequence_start: start,
                sequence_end: end,
            },
            change: if is_new {
                DiffChange::new(DiffChangeKind::Added, Some(DoltHashId([100; 20])))
            } else {
                DiffChange::unchanged()
            },
        }
    }

    fn graph_edge(id: i64, is_new: bool) -> Vec<DiffGraphEdge> {
        vec![DiffGraphEdge {
            edge: GraphEdge {
                edge_id: HashId::pad_str(id),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            change: if is_new {
                DiffChange::new(DiffChangeKind::Added, Some(DoltHashId([100; 20])))
            } else {
                DiffChange::unchanged()
            },
        }]
    }

    #[test]
    fn test_split_connected_components_returns_each_disconnected_subgraph() {
        let left_start = graph_node(1, 0, 3, true);
        let left_end = graph_node(2, 3, 6, true);
        let right_start = graph_node(3, 0, 2, false);
        let right_end = graph_node(4, 2, 4, false);

        let mut diff_graph = DiffGenGraph::new();
        diff_graph.add_edge(left_start, left_end, graph_edge(10, true));
        diff_graph.add_edge(right_start, right_end, graph_edge(11, false));

        assert_eq!(split_connected_components(&diff_graph).len(), 2);
    }
}
