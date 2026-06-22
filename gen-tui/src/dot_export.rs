use std::{collections::HashMap, fmt::Write};

use petgraph::{Undirected, stable_graph::StableGraph};

use crate::{
    geometry::WorldPos,
    layout::{LayoutEdge, LayoutNode, NodeRole},
    partition::StitchSide,
    viewport_graph::CroppedGraph,
};

pub fn export_to_dot(viewport_graph: &CroppedGraph, filename: &str) -> Result<(), std::io::Error> {
    let mut dot = String::new();
    writeln!(&mut dot, "graph {{").unwrap();
    writeln!(&mut dot, "    layout=neato;").unwrap();
    writeln!(&mut dot, "    overlap=false;").unwrap();
    writeln!(&mut dot, "    rankdir=LR;").unwrap();

    // Create a mapping from WorldPos to sequential node IDs for DOT output
    let mut world_to_id: HashMap<WorldPos, usize> = HashMap::new();

    // Collect and sort world positions for consistent output
    let mut sorted_positions: Vec<_> = viewport_graph.nodes().map(|(pos, _)| *pos).collect();
    sorted_positions.sort_by(|a, b| {
        // Sort by x coordinate first, then y coordinate
        a.x.cmp(&b.x).then(a.y.cmp(&b.y))
    });

    // Assign sequential IDs
    for (next_id, pos) in sorted_positions.iter().enumerate() {
        world_to_id.insert(*pos, next_id);
    }

    // Write nodes with coordinates, in sorted order by node ID
    for pos in sorted_positions.iter() {
        let node_id = world_to_id[pos];
        let node = viewport_graph
            .nodes()
            .find(|(world_pos, _)| *world_pos == pos)
            .map(|(_, n)| n)
            .unwrap();
        let x = pos.x;
        let y = pos.y;

        // Create label based on node role
        let label = match &node.role {
            NodeRole::Data(payload) => format!("D{}", payload.index()),
            NodeRole::Routing | NodeRole::Pin => "".to_string(),
            NodeRole::Stitch(side) => match side {
                StitchSide::Left => "S_L".to_string(),
                StitchSide::Right => "S_R".to_string(),
            },
        };

        // Shape and color based on node role
        let (shape, color) = match &node.role {
            NodeRole::Data(_) => ("box", "lightblue"),
            NodeRole::Routing | NodeRole::Pin => ("point", "red"),
            NodeRole::Stitch(_) => ("diamond", "orange"),
        };

        // Adjust font size and size based on node size
        let (width, height) = node.size;
        let fontsize = (width / 10).clamp(8, 16);
        let width_inches = width as f64 / 72.0; // Convert to inches
        let height_inches = height as f64 / 72.0;

        // Include pos attribute for all nodes
        match &node.role {
            NodeRole::Routing => {
                // Point shape doesn't use width/height attributes, no label but include coordinates as custom attribute
                writeln!(
                    &mut dot,
                    "    n{} [label=\"\", pos=\"{},{}!\", shape=\"{}\", fillcolor=\"{}\", style=\"filled\", pin=true, coords=\"({},{})\"];",
                    node_id, x, y, shape, color, x, y
                )
                .unwrap();
            }
            _ => {
                // Regular nodes with width and height
                writeln!(
                    &mut dot,
                    "    n{} [label=\"{}\", pos=\"{},{}!\", shape=\"{}\", fillcolor=\"{}\", style=\"filled\", pin=true, fontsize=\"{}\", width=\"{}\", height=\"{}\"];",
                    node_id, label, x, y, shape, color, fontsize, width_inches, height_inches
                )
                .unwrap();
            }
        }
    }

    // Write edges (with bundle information as custom attributes)
    // ViewportGraph already excludes stitch nodes, so we don't need to filter them
    for (source_pos, target_pos, edge_bundle) in viewport_graph.edges() {
        // Check if both nodes exist in our mapping (they might be at partition boundaries)
        let source_id = world_to_id.get(&source_pos);
        let target_id = world_to_id.get(&target_pos);

        match (source_id, target_id) {
            (Some(source_id), Some(target_id)) => {
                // Both nodes are in viewport - render the edge normally
                if edge_bundle.is_empty() {
                    // Empty bundle - show as dashed line with bundle info as custom attribute
                    writeln!(
                        &mut dot,
                        "    n{} -- n{} [style=\"dashed\", bundle=\"EMPTY\"];",
                        source_id, target_id
                    )
                    .unwrap();
                } else {
                    // Show bundle as comma-separated list in custom attribute
                    let bundle_str = edge_bundle
                        .iter()
                        .map(|(s, t)| format!("({},{})", s.index(), t.index()))
                        .collect::<Vec<_>>()
                        .join(",");

                    // Thickness based on bundle size
                    let thickness = edge_bundle.len().clamp(1, 5);

                    writeln!(
                        &mut dot,
                        "    n{} -- n{} [penwidth={}, bundle=\"{}\"];",
                        source_id, target_id, thickness, bundle_str
                    )
                    .unwrap();
                }
            }
            _ => {
                // One or both nodes are missing (likely at partition boundaries)
                // Add a comment explaining what happened
                let bundle_info = if edge_bundle.is_empty() {
                    "EMPTY".to_string()
                } else {
                    edge_bundle
                        .iter()
                        .map(|(s, t)| format!("({},{})", s.index(), t.index()))
                        .collect::<Vec<_>>()
                        .join(",")
                };

                writeln!(
                    &mut dot,
                    "    // OMITTED EDGE: source_pos=({},{}) -> target_pos=({},{}) bundle=[{}] (nodes at partition boundary)",
                    source_pos.x, source_pos.y, target_pos.x, target_pos.y, bundle_info
                )
                .unwrap();
            }
        }
    }

    writeln!(&mut dot, "}}").unwrap();

    std::fs::write(filename, &dot)?;

    Ok(())
}

/// Dump a single partition's raw `LayoutNode` graph (local coordinates) to a `dot` file.
/// Unlike `export_to_dot`, this includes Stitch and Pin nodes, since the whole point is
/// to inspect alignment-relevant structure that's invisible by the time a `CroppedGraph`
/// is built (stitches are always stripped there, and pins are stripped after this point
/// too, by `prune_pin_stubs`). Intended for ad hoc debugging, e.g. dumping the graph
/// right after `align_partition_to_origin` to see whether Pin/Stitch rows actually landed
/// at y=0 and where Data/Routing nodes sit relative to them.
pub fn export_layout_graph_to_dot(
    layout_graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    filename: &str,
) -> Result<(), std::io::Error> {
    // Inches per layout unit. Large enough that even a 1x1 dummy node's box (and its
    // label) is legible, and consistent between position and size so boxes don't overlap
    // their neighbors purely as a rendering artifact.
    const SCALE: f64 = 0.3;

    let mut dot = String::new();
    writeln!(&mut dot, "graph {{").unwrap();
    writeln!(&mut dot, "    layout=neato;").unwrap();
    writeln!(&mut dot, "    overlap=false;").unwrap();
    writeln!(&mut dot, "    rankdir=LR;").unwrap();

    for node_idx in layout_graph.node_indices() {
        let node = &layout_graph[node_idx];
        let x = node.pos.x as f64 * SCALE;
        // Graphviz's `pos` is math-convention (y increases upward); `LocalPos.y` increases
        // downward like a terminal row. Negate so the rendered image reads top-to-bottom
        // the same way the ASCII snapshot does.
        let y = -(node.pos.y as f64) * SCALE;

        let (label, color) = match &node.role {
            NodeRole::Data(payload) => (format!("D{}", payload.index()), "lightblue"),
            NodeRole::Routing => ("R".to_string(), "red"),
            NodeRole::Pin => ("PIN".to_string(), "purple"),
            NodeRole::Stitch(StitchSide::Left) => ("S_L".to_string(), "orange"),
            NodeRole::Stitch(StitchSide::Right) => ("S_R".to_string(), "orange"),
        };

        // Draw every node as a rectangle sized by its actual `node.size`, so unequal
        // dummy/pin/stitch/data heights (which drive how much vertical clearance the
        // compaction/separation solver reserves around each node) are visible directly,
        // rather than hidden behind a fixed point/diamond/star glyph.
        let (width, height) = node.size;
        let width_inches = width.max(1) as f64 * SCALE;
        let height_inches = height.max(1) as f64 * SCALE;

        writeln!(
            &mut dot,
            "    n{} [label=\"{}\\n{}x{}\", pos=\"{},{}!\", shape=\"box\", fixedsize=\"true\", width=\"{}\", height=\"{}\", fontsize=\"8\", fillcolor=\"{}\", style=\"filled\", pin=true];",
            node_idx.index(),
            label,
            width,
            height,
            x,
            y,
            width_inches,
            height_inches,
            color
        )
        .unwrap();
    }

    for edge_idx in layout_graph.edge_indices() {
        if let Some((source, target)) = layout_graph.edge_endpoints(edge_idx) {
            let bundle_str = layout_graph[edge_idx]
                .bundle
                .iter()
                .map(|(s, t)| format!("({},{})", s.index(), t.index()))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                &mut dot,
                "    n{} -- n{} [label=\"{}\"];",
                source.index(),
                target.index(),
                bundle_str
            )
            .unwrap();
        }
    }

    writeln!(&mut dot, "}}").unwrap();

    std::fs::write(filename, &dot)?;

    Ok(())
}
