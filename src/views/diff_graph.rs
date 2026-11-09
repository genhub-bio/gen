use std::collections::HashSet;

use crossterm::event::{KeyCode, KeyEvent};
use gen_diff::{
    graph::{DiffChangeKind, DiffGenGraph, DiffGenGraphRef, DiffGraphNode},
    operations::{BlockGroupChangeKind, BlockGroupDiff},
};
use gen_graph::{GenGraph, GraphNode};
use gen_tui::{
    LineStyle, distribute_nodes::GapSizes, edge_ports::EdgeKey, graph_view::GraphViewState,
    layout::VisualDetail, plotter::PathStyle,
};
use petgraph::Direction;
use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

use crate::views::gen_graph_widget::{stored_edge_key, zoom_in, zoom_out};

#[derive(Clone)]
pub struct DiffGraphComponent {
    pub title: String,
    pub graph: GenGraph,
    pub highlighted_nodes: Vec<(GraphNode, Color)>,
    pub highlighted_edges: Vec<(EdgeKey<GraphNode>, Color)>,
}

impl DiffGraphComponent {
    /// Whether any node or edge is marked as modified, which the legend then has to explain.
    pub fn has_modified_marks(&self) -> bool {
        self.highlighted_nodes
            .iter()
            .any(|(_, color)| *color == Color::Yellow)
            || self
                .highlighted_edges
                .iter()
                .any(|(_, color)| *color == Color::Yellow)
    }
}

/// What the colored marks of a diff stand for, named after the two states the user compared.
/// Every diff viewer shows it so the colors can be read without knowing the highlight rules.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffLegend {
    added: String,
    removed: String,
}

impl DiffLegend {
    /// The legend of a query compared against a base: green is only in the query, red only in
    /// the base.
    pub fn between(query: &str, base: &str) -> Self {
        Self {
            added: format!("only in {query}"),
            removed: format!("only in {base}"),
        }
    }

    /// The legend of a diff whose two states have no names, such as a patch or an operation.
    pub fn changes() -> Self {
        Self {
            added: "added".to_string(),
            removed: "removed".to_string(),
        }
    }

    /// The legend as one line of text with the color names drawn in their own colors, using
    /// the terminal palette so it follows the user's theme.
    pub fn line(&self, includes_modified: bool) -> Line<'static> {
        let mut spans = vec![
            Span::styled("green", Style::default().fg(Color::Green)),
            Span::raw(format!(" = {}", self.added)),
            Span::raw("  "),
            Span::styled("red", Style::default().fg(Color::Red)),
            Span::raw(format!(" = {}", self.removed)),
        ];
        if includes_modified {
            spans.push(Span::raw("  "));
            spans.push(Span::styled("yellow", Style::default().fg(Color::Yellow)));
            spans.push(Span::raw(" = modified"));
        }
        Line::from(spans)
    }
}

/// Zoom a diff viewer's view for `+`/`=` (in) and `-`/`_` (out). Returns whether `key` was a
/// zoom key. The diff viewers drive a bare view state rather than a controller, and the view
/// state alone does not bind zoom, so each viewer routes its keys through here.
pub fn zoom_for_key<R>(
    view_state: &mut GraphViewState<GraphNode>,
    zoom_levels: &[(VisualDetail, R, GapSizes)],
    key: KeyEvent,
) -> bool {
    match key.code {
        KeyCode::Char('+') | KeyCode::Char('=') => zoom_in(view_state, zoom_levels),
        KeyCode::Char('-') | KeyCode::Char('_') => zoom_out(view_state, zoom_levels),
        _ => return false,
    }
    true
}

/// Apply one key to a diff viewer's graph: zoom for the zoom keys, navigation otherwise.
pub fn handle_diff_key<R>(
    view_state: &mut GraphViewState<GraphNode>,
    zoom_levels: &[(VisualDetail, R, GapSizes)],
    key: KeyEvent,
) {
    if !zoom_for_key(view_state, zoom_levels, key) {
        let _ = view_state.handle_key_event(key);
    }
}

/// Apply diff highlights (nodes and edges) to a graph view's state.
pub fn apply_diff_highlights(
    view_state: &mut GraphViewState<GraphNode>,
    component: &DiffGraphComponent,
) {
    for &(node, color) in &component.highlighted_nodes {
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        view_state.set_node_highlight(node, style);
    }
    for &(edge, color) in &component.highlighted_edges {
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        view_state.set_edge_highlight(edge, style);
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

/// The kind of change an edge is drawn with. An edge that changed keeps its own kind. A neutral
/// continuation edge, which only joins two slices of one stretch of sequence, cannot exist
/// without both slices, so it takes the change of an endpoint that is on one side only: the
/// edges flanking an added node are added and the edges flanking a removed node are removed.
fn displayed_edge_kind(
    edge_kind: DiffChangeKind,
    source: DiffChangeKind,
    target: DiffChangeKind,
) -> DiffChangeKind {
    if edge_kind != DiffChangeKind::Unchanged {
        return edge_kind;
    }
    [source, target]
        .into_iter()
        .find(|kind| matches!(kind, DiffChangeKind::Added | DiffChangeKind::Removed))
        .unwrap_or(DiffChangeKind::Unchanged)
}

/// One highlight per changed edge, keyed with its strands so a parallel edge that did not change
/// stays unlit.
fn collect_highlight_edges(diff_graph: &DiffGenGraph) -> Vec<(EdgeKey<GraphNode>, Color)> {
    let mut edges = Vec::new();
    for (source, target, edge_data) in diff_graph.all_edges() {
        for diff_edge in edge_data {
            let color = match displayed_edge_kind(
                diff_edge.change.kind,
                source.change.kind,
                target.change.kind,
            ) {
                DiffChangeKind::Added => Color::Green,
                DiffChangeKind::Removed => Color::Red,
                DiffChangeKind::Modified => Color::Yellow,
                DiffChangeKind::Unchanged => continue,
            };
            let key = stored_edge_key(source.node, target.node, &diff_edge.edge);
            if !edges.contains(&(key, color)) {
                edges.push((key, color));
            }
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
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use gen_core::{DoltHashId, HashId, Strand};
    use gen_diff::graph::{DiffChange, DiffChangeKind, DiffGenGraph, DiffGraphEdge, DiffGraphNode};
    use gen_graph::{GraphEdge, GraphNode};
    use ratatui::style::Color;

    use super::{
        DiffLegend, build_diff_graph_component, displayed_edge_kind, split_connected_components,
        zoom_for_key,
    };
    use crate::{
        test_helpers::setup_gen,
        views::gen_graph_widget::{create_gen_graph_engine, stored_edge_key},
    };

    fn change(kind: DiffChangeKind) -> DiffChange {
        match kind {
            DiffChangeKind::Unchanged => DiffChange::unchanged(),
            kind => DiffChange::new(kind, Some(DoltHashId([100; 20]))),
        }
    }

    fn graph_node(id: i64, start: i64, end: i64, is_new: bool) -> DiffGraphNode {
        diff_node(
            id,
            start,
            end,
            if is_new {
                DiffChangeKind::Added
            } else {
                DiffChangeKind::Unchanged
            },
        )
    }

    fn diff_node(id: i64, start: i64, end: i64, kind: DiffChangeKind) -> DiffGraphNode {
        DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(id),
                sequence_start: start,
                sequence_end: end,
            },
            change: change(kind),
        }
    }

    fn diff_edge(id: i64, kind: DiffChangeKind) -> Vec<DiffGraphEdge> {
        vec![DiffGraphEdge {
            edge: GraphEdge {
                edge_id: HashId::pad_str(id),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            change: change(kind),
        }]
    }

    fn graph_edge(id: i64, is_new: bool) -> Vec<DiffGraphEdge> {
        diff_edge(
            id,
            if is_new {
                DiffChangeKind::Added
            } else {
                DiffChangeKind::Unchanged
            },
        )
    }

    /// The color the component draws the edge from `source` to `target` with, if it is lit.
    fn edge_color(
        graph: &DiffGenGraph,
        component: &super::DiffGraphComponent,
        source: DiffGraphNode,
        target: DiffGraphNode,
    ) -> Option<Color> {
        let diff_edges = graph
            .edge_weight(source, target)
            .expect("should hold the edge between the two nodes");
        let key = stored_edge_key(source.node, target.node, &diff_edges[0].edge);
        component
            .highlighted_edges
            .iter()
            .find(|(highlighted, _)| *highlighted == key)
            .map(|(_, color)| *color)
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

    #[test]
    fn test_continuation_edges_take_the_change_of_a_one_sided_endpoint() {
        // A deletion re-cuts the base's node: the base has `left removed right` joined by
        // continuation edges, the query skips `removed` with an explicit edge. An insertion
        // adds `inserted` between `right` and `tail`, which the query joins to its neighbors.
        let left = diff_node(1, 0, 3, DiffChangeKind::Unchanged);
        let removed = diff_node(1, 3, 4, DiffChangeKind::Removed);
        let right = diff_node(1, 4, 10, DiffChangeKind::Unchanged);
        let tail = diff_node(1, 10, 34, DiffChangeKind::Unchanged);
        let inserted = diff_node(2, 0, 3, DiffChangeKind::Added);
        let mut diff_graph = DiffGenGraph::new();
        diff_graph.add_edge(left, removed, diff_edge(20, DiffChangeKind::Unchanged));
        diff_graph.add_edge(removed, right, diff_edge(21, DiffChangeKind::Unchanged));
        diff_graph.add_edge(left, right, diff_edge(22, DiffChangeKind::Added));
        diff_graph.add_edge(right, inserted, diff_edge(23, DiffChangeKind::Added));
        diff_graph.add_edge(inserted, tail, diff_edge(24, DiffChangeKind::Added));
        diff_graph.add_edge(right, tail, diff_edge(25, DiffChangeKind::Unchanged));

        let component = build_diff_graph_component(&diff_graph, String::new());

        assert_eq!(
            edge_color(&diff_graph, &component, left, removed),
            Some(Color::Red),
            "the continuation into a removed node is removed"
        );
        assert_eq!(
            edge_color(&diff_graph, &component, removed, right),
            Some(Color::Red),
            "the continuation out of a removed node is removed"
        );
        assert_eq!(
            edge_color(&diff_graph, &component, left, right),
            Some(Color::Green),
            "the explicit deletion edge that skips it is added"
        );
        assert_eq!(
            edge_color(&diff_graph, &component, right, inserted),
            Some(Color::Green)
        );
        assert_eq!(
            edge_color(&diff_graph, &component, inserted, tail),
            Some(Color::Green)
        );
        assert_eq!(
            edge_color(&diff_graph, &component, right, tail),
            None,
            "a continuation between two shared slices stays unlit"
        );
    }

    #[test]
    fn test_continuation_edges_flanking_an_added_node_are_added() {
        // The same re-cut seen from the other side: the added node is a slice only the query
        // has, so the edges that join it to the shared slices are only in the query too.
        let left = diff_node(1, 0, 3, DiffChangeKind::Unchanged);
        let added = diff_node(1, 3, 4, DiffChangeKind::Added);
        let right = diff_node(1, 4, 10, DiffChangeKind::Unchanged);
        let mut diff_graph = DiffGenGraph::new();
        diff_graph.add_edge(left, added, diff_edge(30, DiffChangeKind::Unchanged));
        diff_graph.add_edge(added, right, diff_edge(31, DiffChangeKind::Unchanged));

        let component = build_diff_graph_component(&diff_graph, String::new());

        assert_eq!(
            edge_color(&diff_graph, &component, left, added),
            Some(Color::Green)
        );
        assert_eq!(
            edge_color(&diff_graph, &component, added, right),
            Some(Color::Green)
        );
    }

    #[test]
    fn test_displayed_edge_kind_keeps_the_kind_of_a_changed_edge() {
        use DiffChangeKind::{Added, Modified, Removed, Unchanged};

        assert_eq!(displayed_edge_kind(Added, Removed, Unchanged), Added);
        assert_eq!(displayed_edge_kind(Modified, Added, Added), Modified);
        assert_eq!(
            displayed_edge_kind(Unchanged, Unchanged, Unchanged),
            Unchanged
        );
        assert_eq!(
            displayed_edge_kind(Unchanged, Unchanged, Modified),
            Unchanged
        );
        assert_eq!(displayed_edge_kind(Unchanged, Removed, Unchanged), Removed);
        assert_eq!(displayed_edge_kind(Unchanged, Unchanged, Added), Added);
    }

    #[test]
    fn test_diff_legend_names_the_query_and_the_base() {
        let legend = DiffLegend::between("unknown", "reference").line(false);
        let text: String = legend
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();

        assert_eq!(text, "green = only in unknown  red = only in reference");
        assert_eq!(legend.spans[0].style.fg, Some(Color::Green));
        assert_eq!(legend.spans[3].style.fg, Some(Color::Red));
        assert!(
            DiffLegend::changes()
                .line(true)
                .spans
                .iter()
                .any(|span| span.content == " = modified"),
            "the legend explains yellow when something is modified"
        );
    }

    #[test]
    fn test_zoom_keys_step_the_zoom_level_and_stay_within_the_levels() {
        let context = setup_gen();
        let mut diff_graph = DiffGenGraph::new();
        diff_graph.add_edge(
            graph_node(1, 0, 4, true),
            graph_node(2, 4, 8, true),
            graph_edge(10, true),
        );
        let component = build_diff_graph_component(&diff_graph, String::new());
        let (_engine, zoom_levels, mut view_state) = create_gen_graph_engine(
            component.graph.clone(),
            (context.graph().conn(), context.workspace()),
        );
        let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
        let start = view_state.zoom_index;

        assert!(zoom_for_key(
            &mut view_state,
            &zoom_levels,
            press(KeyCode::Char('+'))
        ));
        assert_eq!(view_state.zoom_index, start + 1);
        assert!(zoom_for_key(
            &mut view_state,
            &zoom_levels,
            press(KeyCode::Char('='))
        ));
        assert_eq!(view_state.zoom_index, start + 2);
        assert!(zoom_for_key(
            &mut view_state,
            &zoom_levels,
            press(KeyCode::Char('-'))
        ));
        assert_eq!(view_state.zoom_index, start + 1);
        assert!(zoom_for_key(
            &mut view_state,
            &zoom_levels,
            press(KeyCode::Char('_'))
        ));
        assert_eq!(view_state.zoom_index, start);
        for _ in 0..zoom_levels.len() + 2 {
            zoom_for_key(&mut view_state, &zoom_levels, press(KeyCode::Char('+')));
        }
        assert_eq!(view_state.zoom_index, zoom_levels.len() - 1);
        assert!(
            !zoom_for_key(&mut view_state, &zoom_levels, press(KeyCode::Right)),
            "navigation keys are not zoom keys"
        );
    }
}
