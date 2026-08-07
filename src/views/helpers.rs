use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};
use gen_graph::{GenGraph, GraphNode, project_path};
use gen_models::{db::GraphConnection, path::Path};
use ratatui::{
    style::Style,
    text::{Line, Span},
};

/// Project a path's blocks onto the current graph state and return its non-terminal
/// `GraphNode`s (the start/end sentinel nodes are never real path content).
pub fn project_path_nodes(
    conn: &GraphConnection,
    path: &Path,
    graph: &GenGraph,
) -> Result<Vec<GraphNode>, String> {
    let path_blocks = path
        .blocks(conn, None)
        .map_err(|err| format!("Failed to load path blocks: {err}"))?;

    let projected_path = project_path(graph, &path_blocks);
    let path_nodes: Vec<GraphNode> = projected_path
        .iter()
        .filter_map(|(node, _)| {
            if node.node_id != PATH_START_NODE_ID && node.node_id != PATH_END_NODE_ID {
                Some(*node)
            } else {
                None
            }
        })
        .collect();

    if path_nodes.is_empty() {
        return Err("Path nodes not found in current graph state".to_string());
    }

    Ok(path_nodes)
}

/// Parses a string with markdown-like asterisk syntax for highlighting.
/// Segments surrounded by '*' are styled with `highlight_style`.
/// Other segments are styled with `default_style`.
pub fn style_text(text: &str, default_style: Style, highlight_style: Style) -> Line<'_> {
    let mut spans = Vec::new();
    let mut is_highlighted = false;
    for part in text.split('*') {
        if !part.is_empty() {
            spans.push(Span::styled(
                part,
                if is_highlighted {
                    highlight_style
                } else {
                    default_style
                },
            ));
        }
        is_highlighted = !is_highlighted;
    }
    Line::from(spans)
}
