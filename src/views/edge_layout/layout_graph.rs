use std::collections::HashMap;

use itertools::Itertools;
use petgraph::stable_graph::StableGraph;

use super::{
    layout_layer::layout_layer,
    process_graph::{
        assign_glyph_idx, assign_ports, compress_graph, preprocess_graph, simplify_graph,
    },
    temp_graph::TempGraph,
    EdgeData, LayoutError, NodeData,
};

pub fn layout_graph(
    graph: StableGraph<NodeData, EdgeData>,
) -> Result<StableGraph<NodeData, EdgeData>, LayoutError> {
    // We assume a left-to-right layout with all positions annotated at this point
    // But we still need to do some pre-processing to make sure the positions are correct and positive.

    let mut temp_graph = TempGraph::new();
    for node_index in graph.node_indices() {
        let mut node = graph.node_weight(node_index).unwrap().clone();
        node.node_id = node_index.index() as u64;
        temp_graph.add_node(
            node_index.index() as u64,
            graph.node_weight(node_index).unwrap().clone(),
        );
    }

    for edge_index in graph.edge_indices() {
        let (source, target) = graph.edge_endpoints(edge_index).unwrap();
        let edge_data = graph.edge_weight(edge_index).unwrap();
        temp_graph.add_edge(
            edge_index.index() as u64,
            source.index() as u64,
            target.index() as u64,
            edge_data.clone(),
        )?;
    }

    preprocess_graph(&mut temp_graph);

    // Sort by x-coordinate, then by y-coordinate and group nodes by x-coordinate
    let node_indices = temp_graph
        .node_indices()
        .sorted_by_key(|node_index| temp_graph.get_node(*node_index).unwrap().position)
        .collect::<Vec<_>>();
    let node_indices_by_layer = node_indices
        .iter()
        .map(|node_index| {
            (
                temp_graph.get_node(*node_index).unwrap().position.0,
                *node_index,
            )
        })
        .into_group_map();

    let mut combined_graph = TempGraph::new();

    println!("--------------------------------");
    println!("laying out graph");
    let nodes = temp_graph.nodes().collect::<Vec<_>>();
    let min_x = nodes
        .iter()
        .min_by_key(|node| node.position.0)
        .unwrap()
        .position
        .0;
    let max_x = nodes
        .iter()
        .max_by_key(|node| node.position.0)
        .unwrap()
        .position
        .0;
    println!("original graph goes from {min_x} to {max_x}");

    let mut x_offset = 0;
    for (x_left, x_right) in node_indices_by_layer.keys().tuple_windows() {
        let left_node_indices = node_indices_by_layer.get(x_left).unwrap();
        let right_node_indices = node_indices_by_layer.get(x_right).unwrap();
        // Edges between left and right
        let edges = left_node_indices
            .iter()
            .cartesian_product(right_node_indices.iter())
            .filter(|(node_index1, node_index2)| {
                temp_graph.contains_edge(**node_index1, **node_index2)
            })
            .map(|(node_index1, node_index2)| (*node_index1, *node_index2))
            .collect::<Vec<_>>();
        // Grab the data associated with the nodes to use in position calculations
        // and to pass on to the output graph.
        let left_data = left_node_indices
            .iter()
            .map(|index| temp_graph.get_node(*index).unwrap().clone())
            .collect::<Vec<_>>();
        let right_data = right_node_indices
            .iter()
            .map(|index| temp_graph.get_node(*index).unwrap().clone())
            .collect::<Vec<_>>();

        let mut layer_graph = layout_layer(&left_data, &right_data, &edges)?;

        // Set margins so that we don't print the label overtop of the terminals
        let mut left_label_width = left_data.iter().map(|node| node.size.0).max().unwrap();
        if left_label_width < 1 {
            left_label_width = 1;
        }

        let mut right_label_width = right_data.iter().map(|node| node.size.0).max().unwrap();
        if right_label_width < 1 {
            right_label_width = 1;
        }

        let left_margin = (left_label_width + 1) / 2;
        let right_margin = (right_label_width + 1) / 2;

        // Post-processing steps (modify the subgraph in place)
        assign_ports(&mut layer_graph)?;
        simplify_graph(&mut layer_graph)?;
        compress_graph(&mut layer_graph, 0, 1, Some((left_margin, right_margin)))?;

        // Note that the subgraph remains bounded by node positions (referenced to their center),
        // so that we can stitch subgraphs together seamlessly. This means that the padding
        // for each layer is actually added over two iterations.
        // ensure_padding(G_layer_routed, left=U_margin, right=V_margin+1)
        // pad_graph(G_layer_routed, left=U_margin, right=V_margin)

        // TODO: This is a hack to ensure that the layer width is correct.
        // if combined_graph.nodes():
        // Close the gap between the right column of the previous layer and the left column of the current layer
        // Calculate offset based on the rightmost x-coordinate in combined_graph
        // and the x-coordinate of the U-nodes in the current layer (which is the left side of G_layer_routed)
        // max_x_combined = max(x for n_id, (x,y) in combined_graph.nodes(data='pos'))
        // x_offset = (max_x_combined - x_u)

        let mut new_nodes_by_old_node = HashMap::new();
        for node_index in layer_graph.node_indices() {
            let node = layer_graph.get_node(node_index).unwrap();
            // Attributes from G_layer_routed should already be correct
            // We just need to adjust the x-coordinate of pos.
            let (x, y) = node.position;
            let new_x = x + x_offset;

            combined_graph.add_node(
                node.node_id,
                NodeData {
                    node_id: node.node_id,
                    position: (new_x, y),
                    node_type: node.node_type.clone(),
                    ports: node.ports,
                    glyph_index: node.glyph_index,
                    size: node.size,
                },
            );

            new_nodes_by_old_node.insert(node_index, node.node_id);
        }

        for edge_index in layer_graph.edge_indices() {
            if let Some((source, target)) = layer_graph.edge_adjacencies(edge_index) {
                if let Some(edge_data) = layer_graph.get_edge(edge_index) {
                    combined_graph.add_edge(edge_index, source, target, edge_data.clone())?;
                }
            }
        }

        // Make sure the next layer stitches seamlessly to the current layer
        let min_x = layer_graph
            .nodes()
            .map(|node| node.position.0)
            .min()
            .unwrap();
        let max_x = layer_graph
            .nodes()
            .map(|node| node.position.0)
            .max()
            .unwrap();
        let putative_layer_width = x_right - x_left;
        let actual_layer_width = max_x - min_x;
        x_offset += actual_layer_width - putative_layer_width;
    }

    // The following functions modify the graph in place
    assign_ports(&mut combined_graph)?;
    simplify_graph(&mut combined_graph)?;
    assign_glyph_idx(&mut combined_graph)?;

    Ok(combined_graph.to_stable_graph())
}
