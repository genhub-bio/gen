use std::{collections::HashMap, fmt::Write};

use gen_widget::layout::{LayoutEdge, LayoutNode, NodeRole};
use petgraph::{
    Undirected,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

/// DOT export modes for different visualization needs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotExportMode {
    /// Full debug output with all routing nodes, bundles, and coordinates
    Debug,
    /// Simplified view with routing nodes as small points, no bundle details
    Simplified,
    /// Only original graph nodes, no routing nodes at all
    InputOnly,
}

/// Export graph to DOT format with configurable detail level
pub fn export_graph_to_dot(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    filename: &str,
    mode: DotExportMode,
) -> Result<(), std::io::Error> {
    let dot_content = match mode {
        DotExportMode::Debug => generate_dot_debug(graph),
        DotExportMode::Simplified => generate_dot_simplified(graph),
        DotExportMode::InputOnly => generate_dot_input_only(graph),
    };

    std::fs::write(filename, &dot_content)?;

    println!("Graph exported to: {}", filename);
    println!(
        "View with: neato -Tpng {} -o {}.png",
        filename,
        filename.replace(".dot", "")
    );

    Ok(())
}

/// Generate DOT with full debug information including bundles and coordinates
fn generate_dot_debug(graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>) -> String {
    let mut dot = String::new();
    writeln!(&mut dot, "graph {{").unwrap();
    writeln!(&mut dot, "    layout=neato;").unwrap();
    writeln!(&mut dot, "    overlap=false;").unwrap();
    writeln!(&mut dot, "    rankdir=LR;").unwrap();

    // Write nodes with full debug info
    let mut sorted_node_indices: Vec<_> = graph.node_indices().collect();
    sorted_node_indices.sort_by(|&a, &b| {
        let node_a = graph.node_weight(a).unwrap();
        let node_b = graph.node_weight(b).unwrap();
        (node_a.pos.x, node_a.pos.y).cmp(&(node_b.pos.x, node_b.pos.y))
    });

    for idx in sorted_node_indices {
        if let Some(node) = graph.node_weight(idx) {
            let node_id = idx.index();
            let x = node.pos.x;
            let y = node.pos.y;

            // Create label based on node role with coordinates
            let node_title = match &node.role {
                NodeRole::Data(payload) => format!("D{}", payload.index()),
                NodeRole::Routing => {
                    let glyph_index = compute_routing_glyph_index(graph, idx);
                    format!("R{:04b}", glyph_index)
                }
                NodeRole::Stitch(side) => match side {
                    gen_widget::partition::StitchSide::Left => "S_L".to_string(),
                    gen_widget::partition::StitchSide::Right => "S_R".to_string(),
                },
            };

            let label = format!("{}\\n({},{})", node_title, x, y);

            // Shape and color based on role
            let (shape, color) = match &node.role {
                NodeRole::Data(_) => ("box", "lightblue"),
                NodeRole::Routing => ("circle", "lightgreen"),
                NodeRole::Stitch(_) => ("diamond", "orange"),
            };

            writeln!(
                &mut dot,
                "    n{} [label=\"{}\", pos=\"{},{}!\", shape=\"{}\", fillcolor=\"{}\", style=\"filled\", pin=true];",
                node_id, label, x, y, shape, color
            )
            .unwrap();
        }
    }

    // Write edges with bundle information
    let edge_colors = ["black", "red", "blue", "green", "purple", "orange", "brown"];
    let mut bundle_to_color = HashMap::new();
    let mut color_index = 0;

    for edge_ref in graph.edge_references() {
        let source_idx = edge_ref.source();
        let target_idx = edge_ref.target();
        let source = source_idx.index();
        let target = target_idx.index();
        let edge_data = edge_ref.weight();

        if edge_data.bundle.is_empty() {
            // Empty bundle - show as gray dashed line
            writeln!(
                &mut dot,
                "    n{} -- n{} [label=\"EMPTY\", color=\"gray\", style=\"dashed\"];",
                source, target
            )
            .unwrap();
        } else {
            // Show bundle contents
            let bundle_str = edge_data
                .bundle
                .iter()
                .map(|(s, t)| format!("({},{})", s.index(), t.index()))
                .collect::<Vec<_>>()
                .join("\\n");

            // Assign consistent colors to different bundle patterns
            let bundle_key = format!("{:?}", edge_data.bundle);
            if !bundle_to_color.contains_key(&bundle_key) {
                bundle_to_color.insert(
                    bundle_key.clone(),
                    edge_colors[color_index % edge_colors.len()],
                );
                color_index += 1;
            }
            let color = bundle_to_color[&bundle_key];

            // Thickness based on bundle size
            let thickness = (edge_data.bundle.len()).clamp(1, 5);

            writeln!(
                &mut dot,
                "    n{} -- n{} [label=\"{}\", color=\"{}\", penwidth={}];",
                source, target, bundle_str, color, thickness
            )
            .unwrap();
        }
    }

    writeln!(&mut dot, "}}").unwrap();
    dot
}

/// Generate simplified DOT with routing nodes as small points
fn generate_dot_simplified(
    _graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
) -> String {
    // TODO: Implement simplified view
    unimplemented!("Simplified DOT export not yet implemented")
}

/// Generate DOT with only original input nodes (no routing nodes)
fn generate_dot_input_only(
    _graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
) -> String {
    // TODO: Implement input-only view
    unimplemented!("Input-only DOT export not yet implemented")
}

/// Compute routing glyph index based on connectivity
fn compute_routing_glyph_index(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    node_idx: petgraph::graph::NodeIndex,
) -> u8 {
    let node = graph.node_weight(node_idx).unwrap();
    let mut glyph_index = 0u8;

    // Check connections in each direction
    for edge in graph.edges(node_idx) {
        let neighbor_idx = edge.target();
        if let Some(neighbor) = graph.node_weight(neighbor_idx) {
            let dx = neighbor.pos.x - node.pos.x;
            let dy = neighbor.pos.y - node.pos.y;

            // Determine primary direction
            if dx.abs() > dy.abs() {
                if dx > 0 {
                    glyph_index |= 0b0100; // East
                } else {
                    glyph_index |= 0b0001; // West
                }
            } else if dy > 0 {
                glyph_index |= 0b0010; // South
            } else {
                glyph_index |= 0b1000; // North
            }
        }
    }

    glyph_index
}
