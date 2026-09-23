use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

use super::{
    LayoutError, NodeData,
    layout_graph_process::{BundledLegEdges, simplify_graph},
    route_channel::Router,
    temp_graph::TempGraph,
};
use crate::{
    geometry::LocalPos,
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

#[derive(Clone, Debug)]
struct Terminal {
    node_id: u64,
    position: i64,
    net_index: u64,
}

#[derive(Clone, Debug)]
struct Pin {
    net: u64,
    position: i64,
}

type Biclique = (HashSet<u64>, HashSet<u64>);

fn enumerate_bicliques(edges: &[(u64, u64)]) -> Result<Vec<Biclique>, LayoutError> {
    // MBEA from Zhang et al. (2014), "On finding bicliques in bipartite graphs."
    let mut part1 = HashSet::new();
    let mut part2 = HashSet::new();
    for edge in edges {
        part1.insert(edge.0);
        part2.insert(edge.1);
    }

    // The algorithm assumes that the left side is smaller than the right side,
    // so we need to check for that and swap if necessary.
    let swapped = if part1.len() > part2.len() {
        part1.clear();
        part2.clear();
        for edge in edges {
            part1.insert(edge.1);
            part2.insert(edge.0);
        }
        true
    } else {
        false
    };

    let mut graph = TempGraph::new();
    for node_id in edges.iter().flat_map(|edge| [edge.0, edge.1]) {
        graph.add_node(
            node_id,
            NodeData {
                node_id,
                position: (0, 0),
                node_type: None,
                size: (1, 1),
                original_node_id: None,
                layer: None,
            },
        );
    }

    for edge in edges {
        graph.add_edge(edge.0, edge.1)?;
        graph.add_edge(edge.1, edge.0)?;
    }

    let left_neighbors = part1.clone();
    let right_neighbors: HashSet<u64> = HashSet::new();
    let mut right_candidates = part2.clone().into_iter().collect::<Vec<u64>>();
    // Sort with stable key: first by neighbor count, then by node id for determinism
    right_candidates.sort_by_key(|x| {
        let neighbor_count = graph.neighbors(*x).len();
        (neighbor_count, *x)
    });
    let clique_outsiders: Vec<u64> = vec![];

    let mut bicliques = find_biclique(
        &graph,
        &left_neighbors,
        &right_neighbors,
        &right_candidates,
        &clique_outsiders,
    );

    if swapped {
        for entry in bicliques.iter_mut() {
            let (l, r) = entry;
            *entry = (r.clone(), l.clone());
        }
    }

    // Sanity check: all edges from the bicliques should be in the original graph,
    // and all edges from the original graph should be in the bicliques
    let edge_set = edges.iter().copied().collect::<HashSet<(u64, u64)>>();
    for biclique in &bicliques {
        // Sort HashSet elements for deterministic iteration
        let u_sorted: Vec<_> = biclique.0.iter().copied().collect();
        let v_sorted: Vec<_> = biclique.1.iter().copied().collect();
        for u in &u_sorted {
            for v in &v_sorted {
                assert!(edge_set.contains(&(*u, *v)));
            }
        }
    }
    for (u, v) in edge_set {
        let mut found_biclique = false;
        for biclique in &bicliques {
            if biclique.0.contains(&u) && biclique.1.contains(&v) {
                found_biclique = true;
            }
        }
        assert!(found_biclique);
    }

    Ok(bicliques)
}

fn find_biclique(
    graph: &TempGraph,
    left_neighbors: &HashSet<u64>,
    right_neighbors: &HashSet<u64>,
    right_candidates: &[u64],
    clique_outsiders: &[u64],
) -> Vec<Biclique> {
    let mut found_bicliques = vec![];

    let mut clique_outsiders_copy = clique_outsiders.to_vec();

    for (i, candidate) in right_candidates.iter().enumerate() {
        // (*1) Select next candidate from right_candidates and attempt to extend the biclique
        let mut new_right_neighbors = right_neighbors.clone();
        new_right_neighbors.insert(*candidate);
        let candidate_neighbors = graph
            .neighbors(*candidate)
            .iter()
            .copied()
            .collect::<HashSet<u64>>();
        let new_left_neighbors: HashSet<u64> = left_neighbors
            .intersection(&candidate_neighbors)
            .copied()
            .collect();

        // New containers for the *next* level's candidates / non-candidates
        let mut new_right_candidates: Vec<u64> = vec![];
        let mut new_clique_outsiders: Vec<u64> = vec![];

        // (*2)  Check if the biclique is maximal by looking at new_left_neighbors
        let mut is_maximal = true;
        for outsider in &clique_outsiders_copy {
            let outsider_neighbors = graph
                .neighbors(*outsider)
                .iter()
                .copied()
                .collect::<HashSet<u64>>();
            if new_left_neighbors == outsider_neighbors {
                is_maximal = false;
                break;
            } else if !new_left_neighbors
                .intersection(&outsider_neighbors)
                .collect::<Vec<_>>()
                .is_empty()
            {
                new_clique_outsiders.push(*outsider);
            }
        }

        clique_outsiders_copy.push(*candidate);

        // Stop exploring this branch if the biclique is not maximal
        if !is_maximal {
            continue;
        }

        // (*3)  Expand R' to maximal size
        for remaining_candidate in &right_candidates[i + 1..] {
            let candidate_neighbors = graph.neighbors(*remaining_candidate);
            if new_left_neighbors.is_subset(&candidate_neighbors) {
                // fully connected
                new_right_neighbors.insert(*remaining_candidate); // extend biclique
            } else if !new_left_neighbors
                .intersection(&candidate_neighbors)
                .collect::<Vec<_>>()
                .is_empty()
            {
                // partially connected
                new_right_candidates.push(*remaining_candidate); // stay as candidate
            }
        }

        // Store the biclique
        found_bicliques.push((new_left_neighbors.clone(), new_right_neighbors.clone()));

        // (*4)  Recurse if there are still candidates left
        if !new_right_candidates.is_empty() {
            let result_bicliques = find_biclique(
                graph,
                &new_left_neighbors,
                &new_right_neighbors,
                &new_right_candidates,
                &new_clique_outsiders,
            );
            found_bicliques.extend(result_bicliques);
        }
    }

    found_bicliques
}

fn make_nets(bicliques: &mut [(HashSet<u64>, HashSet<u64>)]) -> Vec<HashSet<(u64, u64)>> {
    // Assign each edge to one net, preferring the largest containing biclique.
    let mut nets: Vec<HashSet<(u64, u64)>> = vec![];

    // Sort the bicliques by number of edges (descending)
    // Note: Sorting on negative value to get the largest bicliques first
    for biclique in bicliques
        .iter()
        .sorted_by_key(|biclique| -(biclique.0.len() as i64 * biclique.1.len() as i64))
    {
        let mut biclique_edges = HashSet::new();
        // Sort HashSet elements for deterministic iteration
        let mut u_sorted: Vec<_> = biclique.0.iter().copied().collect();
        u_sorted.sort_unstable();
        let mut v_sorted: Vec<_> = biclique.1.iter().copied().collect();
        v_sorted.sort_unstable();
        for u in &u_sorted {
            for v in &v_sorted {
                biclique_edges.insert((*u, *v));
            }
        }

        let unique_edges: HashSet<(u64, u64)> = biclique_edges
            .difference(&nets.clone().into_iter().flatten().collect())
            .cloned()
            .collect();
        if !unique_edges.is_empty() {
            nets.push(unique_edges);
        }
    }

    nets
}

fn place_terminals(node_position: i64, last_position: Option<i64>, num_terminals: i64) -> Vec<i64> {
    // Attempts to place num_terminals terminals symmetrically around node_pos,
    // unless the position is already taken, in which case
    // it returns the next available position.
    let mut result = vec![];

    let mut last_pos = 0;
    if let Some(pos) = last_position {
        last_pos = pos;
    }

    for i in 0..num_terminals {
        // Mirror Python's symmetric half-step spacing around the node position
        let center = (num_terminals as f64 - 1.0) / 2.0;
        let d = i as f64 - center;
        let attempt_position = node_position + (d.signum() * d.abs().ceil()) as i64;

        if last_position.is_none() || last_pos < attempt_position {
            result.push(attempt_position);
            last_pos = attempt_position;
        } else {
            result.push(last_pos + 1);
            last_pos += 1;
        }
    }

    result
}

fn nudge_terminals(
    left_terminals: &mut [Terminal],
    right_terminals: &mut [Terminal],
    node_coord: &HashMap<u64, (i64, i64)>,
) -> Vec<Terminal> {
    // Eliminate unnecessary bends when two nodes are facing each other,
    // even if this breaks symmetry.

    for i in 0..left_terminals.len() {
        for j in 0..right_terminals.len() {
            let left_node_pos = node_coord.get(&left_terminals[i].node_id).unwrap().1;
            let right_node_pos = node_coord.get(&right_terminals[j].node_id).unwrap().1;

            if left_node_pos != right_node_pos {
                continue;
            }

            // Left side:
            let current_pos = left_terminals[i].position;
            let delta = current_pos - left_node_pos;
            let occupied_positions = left_terminals
                .iter()
                .map(|t| t.position)
                .collect::<HashSet<i64>>();
            let target_pos = current_pos - delta;
            if !occupied_positions.contains(&target_pos) {
                left_terminals[i].position = target_pos;
            }

            // Right side:
            let current_pos = right_terminals[j].position;
            let delta = current_pos - right_node_pos;
            let occupied_positions = right_terminals
                .iter()
                .map(|t| t.position)
                .collect::<HashSet<i64>>();
            let target_pos = current_pos - delta;
            if !occupied_positions.contains(&target_pos) {
                right_terminals[j].position = target_pos;
            }
        }
    }

    [left_terminals.to_owned(), right_terminals.to_owned()].concat()
}

fn make_terminals(
    nets: &[HashSet<(u64, u64)>],
    node_coord: HashMap<u64, (i64, i64)>,
) -> Vec<Terminal> {
    // Nodes connect to nets via terminal nodes that become the pins in the routing algorithm.
    // We place the terminal nodes at the same within-rank position as the original nodes, except
    // if there is already a terminal node there, in which case we move everything up.

    // Returns a list of Terminal objects that each refer to a node,
    // net index (starting from 1) and within-rank position.

    // Assign nodes and an average position to each net

    let mut net_nodes = vec![];
    for (i, net) in nets.iter().enumerate() {
        let mut left_nodes = HashSet::new();
        let mut right_nodes = HashSet::new();
        for edge in net {
            left_nodes.insert(edge.0);
            right_nodes.insert(edge.1);
        }

        let nodes: HashSet<u64> = left_nodes.union(&right_nodes).copied().collect();
        let node_coordinate_sum: i64 = nodes
            .iter()
            .map(|node| node_coord.get(node).unwrap().1)
            .sum();
        net_nodes.push((
            i + 1,
            node_coordinate_sum / nodes.len() as i64,
            left_nodes,
            right_nodes,
        ));
    }

    net_nodes.sort_by_key(|node| node.1);

    // Assign nets to each node, in the order of the position of the net
    let mut left_node_nets = HashMap::new();
    let mut right_node_nets = HashMap::new();
    for (net_index, _net_position, left_nodes, right_nodes) in net_nodes {
        for node in left_nodes {
            left_node_nets.entry(node).or_insert(vec![]).push(net_index);
        }
        for node in right_nodes {
            right_node_nets
                .entry(node)
                .or_insert(vec![])
                .push(net_index);
        }
    }

    // Loop over the nodes in the order of their position within each rank
    // to place the terminals.
    let mut left_terminals: Vec<Terminal> = vec![];
    let mut right_terminals: Vec<Terminal> = vec![];

    // Place the terminals for the left nodes
    for node in left_node_nets
        .keys()
        .sorted_by_key(|x| node_coord.get(x).unwrap().1)
    {
        let node_position = node_coord.get(node).unwrap().1;
        let last_position = if left_terminals.is_empty() {
            None
        } else {
            Some(left_terminals[left_terminals.len() - 1].position)
        };

        let num_terminals = left_node_nets.get(node).unwrap().len();
        let terminal_places = place_terminals(node_position, last_position, num_terminals as i64);
        for (i, net_index) in left_node_nets.get(node).unwrap().iter().enumerate() {
            left_terminals.push(Terminal {
                node_id: *node,
                position: terminal_places[i],
                net_index: *net_index as u64,
            });
        }
    }

    // Same for the right nodes
    for node in right_node_nets
        .keys()
        .sorted_by_key(|x| node_coord.get(x).unwrap().1)
    {
        let node_position = node_coord.get(node).unwrap().1;
        let last_position = if right_terminals.is_empty() {
            None
        } else {
            Some(right_terminals[right_terminals.len() - 1].position)
        };

        let num_terminals = right_node_nets.get(node).unwrap().len();
        let terminal_places = place_terminals(node_position, last_position, num_terminals as i64);
        for (i, net_index) in right_node_nets.get(node).unwrap().iter().enumerate() {
            right_terminals.push(Terminal {
                node_id: *node,
                position: terminal_places[i],
                net_index: *net_index as u64,
            });
        }
    }

    nudge_terminals(&mut left_terminals, &mut right_terminals, &node_coord)
}

fn make_pin_lists(left_pins: Vec<Pin>, right_pins: Vec<Pin>) -> (Vec<u64>, Vec<u64>) {
    let all_pins = [left_pins.clone(), right_pins.clone()].concat();

    assert!(!all_pins.is_empty(), "Both lists are empty");

    // Always shift so the minimum position becomes 0
    let min_pos = all_pins.iter().map(|pin| pin.position).min().unwrap();
    let max_pos = all_pins.iter().map(|pin| pin.position).max().unwrap();

    let offset = -min_pos;
    let length = (max_pos - min_pos + 1) as usize;

    let mut left_out = vec![0u64; length];
    for pin in left_pins {
        log::debug!(
            "pin position: {}, adjusted: {}",
            pin.position,
            pin.position + offset
        );
        let adjusted_pos = (pin.position + offset) as usize;
        left_out[adjusted_pos] = pin.net;
    }

    let mut right_out = vec![0u64; length];
    for pin in right_pins {
        let adjusted_pos = (pin.position + offset) as usize;
        right_out[adjusted_pos] = pin.net;
    }

    (left_out, right_out)
}

fn translate_graph(graph: &mut TempGraph, x_offset: i64, y_offset: i64) -> Result<(), LayoutError> {
    for node_index in graph.node_indices() {
        let mut node_data = graph.get_node(node_index).unwrap();
        let (x, y) = node_data.position;
        node_data.position = (x + x_offset, y + y_offset);
        graph.update_node(node_index, node_data)?;
    }

    Ok(())
}

pub fn layout_layer(
    left_positions: &[LayoutNode],
    right_positions: &[LayoutNode],
    edges: &[(NodeIndex, NodeIndex)],
    edge_bundles: &BundledLegEdges,
) -> Result<StableGraph<LayoutNode, LayoutEdge, Undirected>, LayoutError> {
    let mut selected =
        layout_layer_internal(left_positions, right_positions, edges, edge_bundles, false)?;
    simplify_graph(&mut selected)?;
    let mut selected_score = (total_edge_length(&selected), selected.node_count());
    for (swap_layers, reverse_order) in [(false, true), (true, false), (true, true)] {
        let mut candidate = if swap_layers {
            layout_layer_swapped(
                left_positions,
                right_positions,
                edges,
                edge_bundles,
                reverse_order,
            )?
        } else {
            layout_layer_internal(
                left_positions,
                right_positions,
                edges,
                edge_bundles,
                reverse_order,
            )?
        };
        simplify_graph(&mut candidate)?;
        let score = (total_edge_length(&candidate), candidate.node_count());
        if score < selected_score {
            selected = candidate;
            selected_score = score;
        }
    }
    Ok(selected)
}

/// Route from the opposite layer in the requested vertical orientation.
fn layout_layer_swapped(
    left_positions: &[LayoutNode],
    right_positions: &[LayoutNode],
    edges: &[(NodeIndex, NodeIndex)],
    edge_bundles: &BundledLegEdges,
    reverse_order: bool,
) -> Result<StableGraph<LayoutNode, LayoutEdge, Undirected>, LayoutError> {
    let reflect_positions = |positions: &[LayoutNode]| {
        positions
            .iter()
            .map(|node| {
                let mut reflected = node.clone();
                reflected.pos.x = -node.pos.x;
                reflected
            })
            .collect::<Vec<_>>()
    };
    let swapped_edges = edges
        .iter()
        .map(|&(left, right)| (right, left))
        .collect::<Vec<_>>();
    let swapped_bundles = edge_bundles
        .iter()
        .map(|(&(left, right), bundle)| ((right, left), bundle.clone()))
        .collect();
    let mut graph = layout_layer_internal(
        &reflect_positions(right_positions),
        &reflect_positions(left_positions),
        &swapped_edges,
        &swapped_bundles,
        reverse_order,
    )?;

    if let Some(left_node) = left_positions.first() {
        let routed_left = graph[NodeIndex::new(right_positions.len())].pos.x;
        for node in graph.node_weights_mut() {
            node.pos.x = left_node.pos.x + (routed_left - node.pos.x);
        }
    }
    Ok(graph)
}

fn total_edge_length(graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>) -> i64 {
    graph
        .edge_indices()
        .filter_map(|edge_index| {
            let (source_index, target_index) = graph.edge_endpoints(edge_index)?;
            let source = graph.node_weight(source_index)?;
            let target = graph.node_weight(target_index)?;
            Some((source.pos.x - target.pos.x).abs() + (source.pos.y - target.pos.y).abs())
        })
        .sum()
}

fn layout_layer_internal(
    left_positions: &[LayoutNode],
    right_positions: &[LayoutNode],
    edges: &[(NodeIndex, NodeIndex)],
    edge_bundles: &BundledLegEdges,
    reverse_order: bool,
) -> Result<StableGraph<LayoutNode, LayoutEdge, Undirected>, LayoutError> {
    // If reverse_order is true, mirror both layers across the horizontal axis y=0.
    // A global negation is its own inverse regardless of either layer's extent.
    let (left_positions_flipped, right_positions_flipped) = if reverse_order {
        let left_flipped = left_positions
            .iter()
            .map(|node| {
                let mut flipped = node.clone();
                flipped.pos.y = -node.pos.y;
                flipped
            })
            .collect();
        let right_flipped = right_positions
            .iter()
            .map(|node| {
                let mut flipped = node.clone();
                flipped.pos.y = -node.pos.y;
                flipped
            })
            .collect();

        (left_flipped, right_flipped)
    } else {
        (left_positions.to_vec(), right_positions.to_vec())
    };

    let left_positions = &left_positions_flipped;
    let right_positions = &right_positions_flipped;

    // Route consecutive layers through terminal channels and shared buses.

    // Create a mapping from edge NodeIndex to sequential u64 IDs for routing algorithm
    // The edges use indices into left_positions and right_positions arrays
    let left_node_ids: HashSet<u64> = (0..left_positions.len() as u64).collect();
    let right_node_ids: HashSet<u64> = (left_positions.len() as u64
        ..(left_positions.len() + right_positions.len()) as u64)
        .collect();

    // Convert edges to u64 pairs for routing algorithm
    // Source indices are relative to left_positions, target indices are relative to right_positions
    let edges_u64: Vec<(u64, u64)> = edges
        .iter()
        .map(|(source, target)| {
            let source_id = source.index() as u64;
            let target_id = (left_positions.len() + target.index()) as u64;
            (source_id, target_id)
        })
        .collect();

    for (source, target) in &edges_u64 {
        assert!(
            left_node_ids.contains(source),
            "Source {} not in left_node_ids",
            source
        );
        assert!(
            right_node_ids.contains(target),
            "Target {} not in right_node_ids",
            target
        );
        assert!(
            !left_node_ids.contains(target),
            "Target {} should not be in left_node_ids",
            target
        );
        assert!(
            !right_node_ids.contains(source),
            "Source {} should not be in right_node_ids",
            source
        );
    }

    // Nets are non-overlapping subgraphs where every node on one side,
    // has an edge to every node on the other side.
    // This is similar to the definition of bicliques, except that those may overlap.
    let mut bicliques = enumerate_bicliques(&edges_u64)?;
    let nets = make_nets(&mut bicliques);

    // For each node, we create one terminal per net that it is part of.
    // Each terminal is defined by a linear position within its rank, and its net index.
    let mut all_positions_by_node = HashMap::new();

    for (i, node) in left_positions.iter().enumerate() {
        let node_id = i as u64;
        all_positions_by_node.insert(node_id, (node.pos.x, node.pos.y));
    }
    for (i, node) in right_positions.iter().enumerate() {
        let node_id = (left_positions.len() + i) as u64;
        all_positions_by_node.insert(node_id, (node.pos.x, node.pos.y));
    }

    let terminals = make_terminals(&nets, all_positions_by_node);

    // First channel: left nodes to left terminals (as nodes)
    let left_pins = left_positions
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let node_id = i as u64;
            Pin {
                net: node_id + 1,
                position: node.pos.y,
            }
        })
        .collect::<Vec<Pin>>();
    let right_pins = terminals
        .iter()
        .filter(|terminal| left_node_ids.contains(&terminal.node_id))
        .map(|terminal| Pin {
            net: terminal.node_id + 1,
            position: terminal.position,
        })
        .collect::<Vec<Pin>>();

    let (left_pin_list, right_pin_list) = make_pin_lists(left_pins, right_pins);

    let mut router = Router::new(
        left_pin_list.clone(),
        right_pin_list.clone(),
        None, // Use computed density
        1,    // minimum_jog_length
        10,   // steady_net_constant
    );

    let mut graph1 = router.route()?;

    // Only anchor and translate if graph1 has nodes
    if graph1.node_count() > 0 {
        // Anchor the bottom-left node of G1 to the bottom-left node of U_pos
        let left_anchor = left_positions
            .iter()
            .sorted_by_key(|node| (node.pos.x, node.pos.y))
            .collect::<Vec<&LayoutNode>>()[0]
            .pos;
        // `node_id` breaks ties deterministically: `TempGraph::nodes()` iterates a `HashMap`, so
        // without a tie-break, two nodes sharing this corner's position would sort in whatever
        // arbitrary bucket order the map happens to produce - which can differ between otherwise
        // identical calls (see the `node_id` sort in the channel-merge loop below for the same
        // hazard). Since `node_id`s are assigned in construction order, sorting by it recovers a
        // stable order that only depends on the graph's own topology.
        let graph1_nodes: Vec<_> = graph1
            .nodes()
            .sorted_by_key(|node| (node.position, node.node_id))
            .collect();
        if !graph1_nodes.is_empty() {
            let graph1_anchor = graph1_nodes[0].position;
            let offset_x = left_anchor.x - graph1_anchor.0;
            let offset_y = left_anchor.y - graph1_anchor.1;

            translate_graph(&mut graph1, offset_x, offset_y)?;
        }
    }

    // Second channel: U terminals (as nets) to V terminals (as nets)
    let left_pins2 = terminals
        .iter()
        .filter(|terminal| left_node_ids.contains(&terminal.node_id))
        .map(|terminal| Pin {
            net: terminal.net_index + 1,
            position: terminal.position,
        })
        .collect::<Vec<Pin>>();
    let right_pins2 = terminals
        .iter()
        .filter(|terminal| right_node_ids.contains(&terminal.node_id))
        .map(|terminal| Pin {
            net: terminal.net_index + 1,
            position: terminal.position,
        })
        .collect::<Vec<Pin>>();

    let (left_pin_list2, right_pin_list2) = make_pin_lists(left_pins2, right_pins2);

    let mut router2 = Router::new(
        left_pin_list2.clone(),
        right_pin_list2.clone(),
        None, // Use computed density
        1,    // minimum_jog_length
        10,   // steady_net_constant
    );

    let mut graph2 = router2.route()?;

    // Only anchor and translate if both graphs have nodes
    if graph1.node_count() > 0 && graph2.node_count() > 0 {
        // Anchor the bottom-left node of G2 to the bottom-right node of G1
        // For G1, we sort the nodes descending by x coordinate and ascending by y coordinate to get bottom-right.
        let graph1_nodes: Vec<_> = graph1
            .nodes()
            .sorted_by_key(|node| (-node.position.0, node.position.1, node.node_id))
            .collect();
        // For G2 conventional sorting (x and y ascending) is sufficient to get bottom-left
        let graph2_nodes: Vec<_> = graph2
            .nodes()
            .sorted_by_key(|node| (node.position.0, node.position.1, node.node_id))
            .collect();

        if !graph1_nodes.is_empty() && !graph2_nodes.is_empty() {
            let graph1_bottom_right_anchor = graph1_nodes[0].position;
            let graph2_bottom_left_anchor = graph2_nodes[0].position;

            let offset_x_2 = graph1_bottom_right_anchor.0 - graph2_bottom_left_anchor.0;
            let offset_y_2 = graph1_bottom_right_anchor.1 - graph2_bottom_left_anchor.1;
            translate_graph(&mut graph2, offset_x_2, offset_y_2)?;
        }
    }

    // Third channel: V terminals (as nodes) to V nodes
    let left_pins3 = terminals
        .iter()
        .filter(|terminal| right_node_ids.contains(&terminal.node_id))
        .map(|terminal| Pin {
            net: terminal.node_id + 1,
            position: terminal.position,
        })
        .collect::<Vec<Pin>>();
    let right_pins3 = right_positions
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let node_id = (left_positions.len() + i) as u64;
            Pin {
                net: node_id + 1,
                position: node.pos.y,
            }
        })
        .collect::<Vec<Pin>>();

    let (left_pin_list3, right_pin_list3) = make_pin_lists(left_pins3, right_pins3);

    let mut router3 = Router::new(
        left_pin_list3.clone(),
        right_pin_list3.clone(),
        None, // Use computed density
        1,    // minimum_jog_length
        10,   // steady_net_constant
    );

    let mut graph3 = router3.route()?;

    // Only anchor and translate if both graphs have nodes
    if graph2.node_count() > 0 && graph3.node_count() > 0 {
        // Anchor the bottom-left node of G3 to the bottom-right node of G2.
        let graph2_nodes: Vec<_> = graph2
            .nodes()
            .sorted_by_key(|node| (-node.position.0, node.position.1, node.node_id))
            .collect();
        let graph3_nodes: Vec<_> = graph3
            .nodes()
            .sorted_by_key(|node| (node.position.0, node.position.1, node.node_id))
            .collect();

        if !graph2_nodes.is_empty() && !graph3_nodes.is_empty() {
            let graph2_bottom_right_anchor = graph2_nodes[0].position;
            let graph3_bottom_left_anchor = graph3_nodes[0].position;

            let offset_x_3 = graph2_bottom_right_anchor.0 - graph3_bottom_left_anchor.0;
            let offset_y_3 = graph2_bottom_right_anchor.1 - graph3_bottom_left_anchor.1;
            translate_graph(&mut graph3, offset_x_3, offset_y_3)?;
        }
    }

    // Combine all channel graphs into a single StableGraph layer graph, merging nodes by position
    // Use StableGraph with LayoutNode and LayoutEdge for the combined graph
    let mut layer_graph: StableGraph<LayoutNode, LayoutEdge, Undirected> = StableGraph::default();

    // Map from (position) to NodeIndex in layer_graph for coordinate-based deduplication
    let mut position_to_node_idx: HashMap<(i64, i64), NodeIndex> = HashMap::new();

    // Helper function to convert NodeData to LayoutNode
    let convert_to_layout_node =
        |node: &NodeData, adjusted_position: (i64, i64)| -> Result<LayoutNode, LayoutError> {
            let pos = LocalPos::new(adjusted_position.into());

            // Determine node role
            let role = if let Some(ref node_type) = node.node_type {
                if node_type == "Routing" {
                    NodeRole::Routing
                } else if let Some(original_id) = node.original_node_id {
                    NodeRole::Data(NodeIndex::new(original_id as usize))
                } else {
                    return Err(LayoutError::MissingOriginalNodeId(node.node_id));
                }
            } else if let Some(original_id) = node.original_node_id {
                NodeRole::Data(NodeIndex::new(original_id as usize))
            } else {
                return Err(LayoutError::MissingOriginalNodeId(node.node_id));
            };

            // Convert i64 to u64 for size, preserve original layer information
            Ok(LayoutNode::new(
                role,
                pos,
                (node.size.0 as u64, node.size.1 as u64),
                node.layer, // Preserve the original layer from NodeData
            ))
        };

    // The left side corresponds to the original U nodes, including their position
    // Track mapping from array index to layer_graph NodeIndex
    let mut left_array_idx_to_node: HashMap<usize, NodeIndex> = HashMap::new();
    for (array_idx, node) in left_positions.iter().enumerate() {
        let position = (node.pos.x, node.pos.y);
        let node_idx = layer_graph.add_node(node.clone());
        position_to_node_idx.insert(position, node_idx);
        left_array_idx_to_node.insert(array_idx, node_idx);
    }

    // The right side corresponds to the original V nodes, with updated x coordinates.
    let right_boundary = graph3
        .nodes()
        .map(|node| node.position.0)
        .max()
        .unwrap_or(0);

    let mut right_array_idx_to_node: HashMap<usize, NodeIndex> = HashMap::new();
    for (array_idx, node) in right_positions.iter().enumerate() {
        let new_position = (right_boundary, node.pos.y);
        // Create layout node with updated x position
        let new_pos = LocalPos::new(new_position.into());
        let updated_node = LayoutNode::new(node.role.clone(), new_pos, node.size, node.layer);
        let node_idx = layer_graph.add_node(updated_node);
        position_to_node_idx.insert(new_position, node_idx);
        right_array_idx_to_node.insert(array_idx, node_idx);
    }

    // Sort temporary nodes by geometry so randomized hash order cannot affect insertion order.
    for source_graph in [graph1, graph2, graph3] {
        let mut graph_nodes: Vec<NodeData> = source_graph.nodes().collect();
        graph_nodes.sort_by_key(|node| (node.position, node.node_id));
        for node_data in &graph_nodes {
            let position = node_data.position;

            // Check if a node already exists at this position (coordinate-based deduplication).
            if position_to_node_idx.contains_key(&position) {
                // Already added (e.g. an original right or left node position)
                continue;
            }

            // Create new routing node with proper node_type
            let mut final_node_data = node_data.clone();
            if final_node_data.node_type.is_none() {
                final_node_data.node_type = Some("Routing".to_string());
            }

            let layout_node = convert_to_layout_node(&final_node_data, position)?;
            let node_idx = layer_graph.add_node(layout_node);
            position_to_node_idx.insert(position, node_idx);
        }

        // Add edges using coordinate-based mapping
        for node1_data in &graph_nodes {
            let node1_pos = node1_data.position;

            // Get the mapped node index for this position in the layer graph
            if let Some(&layer_node1_idx) = position_to_node_idx.get(&node1_pos) {
                // Iterate through neighbors in the source graph, sorted by their deterministic
                // geometry for the same reason as `graph_nodes` above. Sorting the randomized
                // `HashSet` by router-assigned id is not sufficient: those ids inherit any
                // nondeterminism in node creation order and would make edge insertion order vary.
                let mut sorted_neighbors: Vec<NodeData> = source_graph
                    .neighbors(node1_data.node_id)
                    .into_iter()
                    .filter_map(|node_id| source_graph.get_node(node_id))
                    .collect();
                sorted_neighbors.sort_by_key(|node| (node.position, node.node_id));
                for node2_data in sorted_neighbors {
                    let node2_pos = node2_data.position;
                    if let Some(&layer_node2_idx) = position_to_node_idx.get(&node2_pos)
                        && layer_node1_idx != layer_node2_idx
                    {
                        // Check if edge already exists to avoid duplicates
                        if layer_graph
                            .find_edge(layer_node1_idx, layer_node2_idx)
                            .is_none()
                            && layer_graph
                                .find_edge(layer_node2_idx, layer_node1_idx)
                                .is_none()
                        {
                            // An ordinary edge is born empty; bundles (including a wormhole's,
                            // which is now a real domain-labelled edge like any other) are
                            // applied via BFS below, once every edge exists to search over.
                            let layout_edge = LayoutEdge {
                                bundle: Vec::new(),
                                is_backward_span: false,
                            };
                            layer_graph.add_edge(layer_node1_idx, layer_node2_idx, layout_edge);
                        }
                    }
                }
            }
        }
    }

    // Apply bundles to routing paths
    // For each edge in edge_bundles, find the path in layer_graph and label all edges with the bundle
    // Sort the keys to ensure deterministic iteration order
    let mut sorted_bundles: Vec<_> = edge_bundles.iter().collect();
    sorted_bundles.sort_by_key(|((left, right), _)| (*left, *right));
    for ((left_idx, right_idx), (bundle, is_backward_span)) in sorted_bundles {
        let start_node = match left_array_idx_to_node.get(&left_idx.index()) {
            Some(&node) => node,
            None => continue, // Node not found, skip
        };
        let end_node = match right_array_idx_to_node.get(&right_idx.index()) {
            Some(&node) => node,
            None => continue, // Node not found, skip
        };

        // Use BFS to find the path between start and end nodes
        if let Some(path) = find_path_bfs(&layer_graph, start_node, end_node) {
            // Label all edges in the path with the bundle and is_backward_span flag
            for i in 0..path.len() - 1 {
                let u = path[i];
                let v = path[i + 1];

                let edge_idx = layer_graph
                    .find_edge(u, v)
                    .or_else(|| layer_graph.find_edge(v, u));
                if let Some(edge_idx) = edge_idx
                    && let Some(edge_weight) = layer_graph.edge_weight_mut(edge_idx)
                {
                    // Add the bundle to this edge (avoiding duplicates)
                    for label in bundle {
                        if !edge_weight.bundle.contains(label) {
                            edge_weight.bundle.push(*label);
                        }
                    }
                    edge_weight.is_backward_span |= *is_backward_span;
                }
            }
        }
    }

    // If we flipped the positions for routing, flip the result back. Negation
    // about y=0 is its own exact inverse regardless of either side's vertical
    // extent, so this always restores the original coordinates.
    if reverse_order {
        for node in layer_graph.node_weights_mut() {
            node.pos.y = -node.pos.y;
        }

        for node_index in layer_graph.node_indices() {
            let node = &layer_graph[node_index];
            if matches!(node.role, NodeRole::Routing) {
                let incident_bundles: Vec<_> = layer_graph
                    .edges(node_index)
                    .map(|edge| &edge.weight().bundle)
                    .collect();
                assert!(
                    incident_bundles.len() != 2 || incident_bundles[0] == incident_bundles[1],
                    "degree-two routing node {node_index:?} joins different bundles"
                );
            }
        }
    }

    Ok(layer_graph)
}

/// BFS to find path between two nodes in the layer graph
fn find_path_bfs(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected>,
    start: NodeIndex,
    end: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    use std::collections::{HashMap as StdHashMap, VecDeque};

    let mut queue = VecDeque::new();
    let mut visited = std::collections::HashSet::new();
    let mut predecessors = StdHashMap::new();

    queue.push_back(start);
    visited.insert(start);

    while let Some(current) = queue.pop_front() {
        if current == end {
            // Reconstruct path
            let mut path = Vec::new();
            let mut node = end;

            while node != start {
                path.push(node);
                node = predecessors[&node];
            }
            path.push(start);
            path.reverse();

            return Some(path);
        }

        for neighbor in graph.neighbors(current) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                predecessors.insert(neighbor, current);
                queue.push_back(neighbor);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use more_asserts::{assert_ge, assert_gt};

    use super::*;

    #[test]
    fn test_enumerate_bicliques() {
        // Simple graph
        let edges = vec![(1, 3), (1, 4), (2, 3), (2, 4)];
        let bicliques = enumerate_bicliques(&edges);

        assert!(bicliques.is_ok());
        let bicliques = bicliques.unwrap();

        // Expected output should have exactly one biclique with nodes 1,2 on one side and 3,4 on the other
        let expected = [(HashSet::from([1, 2]), HashSet::from([3, 4]))];
        assert_eq!(bicliques, expected);

        // # Another test case with multiple bicliques
        let edges2 = vec![(1, 3), (1, 4), (2, 3), (2, 4), (2, 5)];
        let bicliques2 = enumerate_bicliques(&edges2);

        assert!(bicliques2.is_ok());
        let bicliques2 = bicliques2.unwrap();

        // Expected output should have two bicliques
        let expected2 = [
            (HashSet::from([1, 2]), HashSet::from([3, 4])),
            (HashSet::from([2]), HashSet::from([3, 4, 5])),
        ];
        assert_eq!(
            bicliques2
                .iter()
                .sorted_by_key(|biclique| (biclique.0.len(), biclique.1.len()))
                .collect::<Vec<_>>(),
            expected2
                .iter()
                .sorted_by_key(|biclique| (biclique.0.len(), biclique.1.len()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_make_nets() {
        // Simple case with two bicliques
        let mut bicliques = vec![
            (HashSet::from([1, 2]), HashSet::from([3, 4])),
            (HashSet::from([2]), HashSet::from([5])),
        ];
        let nets = make_nets(&mut bicliques);

        // Verify we get two nets, first one should be larger
        assert_eq!(nets.len(), 2);
        assert_eq!(nets[0].len(), 4);
        assert_eq!(nets[1].len(), 1);

        // Check contents of the nets
        assert!(
            nets[0] == HashSet::from([(1, 3), (1, 4), (2, 3), (2, 4)])
                || nets[0] == HashSet::from([(3, 1), (4, 1), (3, 2), (4, 2)])
        );
        assert!(nets[1] == HashSet::from([(2, 5)]) || nets[1] == HashSet::from([(5, 2)]));
    }

    #[test]
    fn test_make_terminals() {
        let nets = vec![
            HashSet::from([(1, 3), (1, 4)]),
            HashSet::from([(2, 3), (2, 4), (2, 5)]),
        ];
        let positions_by_node = HashMap::from([
            (1, (0, 0)),
            (2, (1, 0)),
            (3, (0, 1)),
            (4, (1, 1)),
            (5, (2, 1)),
        ]);

        let terminals = make_terminals(&nets, positions_by_node.clone());

        assert_eq!(terminals.len(), 7);

        // Verify terminal properties
        for terminal in &terminals {
            assert!(positions_by_node.contains_key(&terminal.node_id));
            assert!(HashSet::from([1, 2]).contains(&terminal.net_index));
        }
    }

    #[test]
    fn test_make_pin_lists() {
        let left = vec![
            Pin {
                net: 1,
                position: 2,
            },
            Pin {
                net: 2,
                position: 4,
            },
        ];
        let right = vec![
            Pin {
                net: 1,
                position: 1,
            },
            Pin {
                net: 2,
                position: 3,
            },
        ];

        let (left_list, right_list) = make_pin_lists(left, right);

        // After normalization (subtracting min position of 1):
        // left pins will be at positions 1 and 3 (2-1=1, 4-1=3)
        // right pins will be at positions 0 and 2 (1-1=0, 3-1=2)
        let expected_left = vec![0, 1, 0, 2];
        let expected_right = vec![1, 0, 2, 0];

        assert_eq!(left_list, expected_left);
        assert_eq!(right_list, expected_right);
    }

    #[test]
    #[should_panic]
    fn test_make_pin_lists_with_empty_inputs() {
        // Test with empty lists--should cause an assertion error
        make_pin_lists(vec![], vec![]);
    }

    #[test]
    fn test_route_layer_example_1() {
        // Create LayoutNodes for the test
        let left_nodes = vec![
            LayoutNode::data(NodeIndex::new(1), LocalPos::new_xy(0, 1), (0, 0), None),
            LayoutNode::data(NodeIndex::new(2), LocalPos::new_xy(1, 1), (0, 0), None),
        ];
        let right_nodes = vec![
            LayoutNode::data(NodeIndex::new(3), LocalPos::new_xy(0, 2), (0, 0), None),
            LayoutNode::data(NodeIndex::new(4), LocalPos::new_xy(1, 2), (0, 0), None),
            LayoutNode::data(NodeIndex::new(5), LocalPos::new_xy(2, 2), (0, 0), None),
        ];

        // Create edges using NodeIndex
        let edges = vec![
            (NodeIndex::new(0), NodeIndex::new(0)), // left[0] to right[0]
            (NodeIndex::new(0), NodeIndex::new(1)), // left[0] to right[1]
            (NodeIndex::new(1), NodeIndex::new(0)), // left[1] to right[0]
            (NodeIndex::new(1), NodeIndex::new(1)), // left[1] to right[1]
            (NodeIndex::new(1), NodeIndex::new(2)), // left[1] to right[2]
        ];

        let edge_bundles = HashMap::new();
        let graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles);

        let graph = graph.expect("should route the first example");

        // Verify nodes and edges exist
        assert_ge!(graph.node_count(), 5); // At least original nodes
        assert_ge!(graph.edge_count(), 5); // At least original edges
    }

    #[test]
    fn test_route_layer_example_2() {
        let left_nodes = vec![
            LayoutNode::data(NodeIndex::new(5), LocalPos::new_xy(2, 0), (0, 0), None),
            LayoutNode::data(NodeIndex::new(2), LocalPos::new_xy(2, 2), (0, 0), None),
        ];
        let right_nodes = vec![LayoutNode::data(
            NodeIndex::new(3),
            LocalPos::new_xy(3, 1),
            (0, 0),
            None,
        )];

        let edges = vec![
            (NodeIndex::new(0), NodeIndex::new(0)), // left[0] to right[0]
            (NodeIndex::new(1), NodeIndex::new(0)), // left[1] to right[0]
        ];

        let edge_bundles = HashMap::new();
        let graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles);

        let graph = graph.expect("should route the second example");

        // Verify the result has the expected structure
        assert_gt!(graph.node_count(), 2); // Should have original nodes plus routing nodes
        assert_gt!(graph.edge_count(), 1); // Should have original edges plus routing edges
    }

    #[test]
    fn test_route_layer_example_3() {
        let left_nodes = vec![
            LayoutNode::data(NodeIndex::new(1), LocalPos::new_xy(1, 1), (0, 0), None),
            LayoutNode::data(NodeIndex::new(5), LocalPos::new_xy(2, 0), (0, 0), None),
        ];
        let right_nodes = vec![LayoutNode::data(
            NodeIndex::new(2),
            LocalPos::new_xy(2, 2),
            (0, 0),
            None,
        )];

        let edges = vec![
            (NodeIndex::new(0), NodeIndex::new(0)), // left[0] to right[0]
            (NodeIndex::new(1), NodeIndex::new(0)), // left[1] to right[0]
        ];

        let edge_bundles = HashMap::new();
        let graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles);

        graph.expect("should route the third example");
    }

    /// A wormhole door is a real node from crawl time onward (see
    /// `crawl::build_window_graph`), not synthesized by `layout_layer` - it's routed exactly
    /// like any other node given in `right_nodes` with a real edge to its boundary.
    #[test]
    fn test_layout_layer_routes_a_wormhole_node_on_the_right_like_any_other_node() {
        let external_target = NodeIndex::new(999);
        let left_nodes = vec![
            LayoutNode::data(NodeIndex::new(1), LocalPos::new_xy(0, 0), (0, 0), None),
            LayoutNode::data(NodeIndex::new(2), LocalPos::new_xy(0, 5), (0, 0), None),
        ];
        let right_nodes = vec![
            LayoutNode::data(NodeIndex::new(3), LocalPos::new_xy(3, 0), (0, 0), None),
            LayoutNode::new(
                NodeRole::Wormhole(external_target),
                LocalPos::new_xy(3, 5),
                (0, 0),
                None,
            ),
        ];
        let edges = vec![
            (NodeIndex::new(0), NodeIndex::new(0)), // left[0] to right[0]
            (NodeIndex::new(1), NodeIndex::new(1)), // left[1] (boundary) to right[1] (wormhole)
        ];
        let edge_bundles = HashMap::new();

        let graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles)
            .expect("should route a wormhole node on the right");

        let wormhole_count = graph
            .node_weights()
            .filter(
                |node| matches!(node.role, NodeRole::Wormhole(target) if target == external_target),
            )
            .count();
        assert_eq!(
            wormhole_count, 1,
            "expected exactly one wormhole node, routed like any other"
        );

        let data_count = graph
            .node_weights()
            .filter(|node| matches!(node.role, NodeRole::Data(_)))
            .count();
        assert_eq!(
            data_count, 3,
            "the wormhole node must not disturb any of the real data nodes"
        );
    }

    #[test]
    fn test_layout_layer_routes_a_wormhole_node_on_the_left_like_any_other_node() {
        let external_target = NodeIndex::new(998);
        let left_nodes = vec![
            LayoutNode::data(NodeIndex::new(1), LocalPos::new_xy(0, 0), (0, 0), None),
            LayoutNode::new(
                NodeRole::Wormhole(external_target),
                LocalPos::new_xy(0, 5),
                (0, 0),
                None,
            ),
        ];
        let right_nodes = vec![
            LayoutNode::data(NodeIndex::new(3), LocalPos::new_xy(3, 0), (0, 0), None),
            LayoutNode::data(NodeIndex::new(4), LocalPos::new_xy(3, 5), (0, 0), None),
        ];
        let edges = vec![
            (NodeIndex::new(0), NodeIndex::new(0)), // left[0] to right[0]
            (NodeIndex::new(1), NodeIndex::new(1)), // left[1] (wormhole) to right[1] (boundary)
        ];
        let edge_bundles = HashMap::new();

        let graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles)
            .expect("should route a wormhole node on the left");

        let wormhole_count = graph
            .node_weights()
            .filter(
                |node| matches!(node.role, NodeRole::Wormhole(target) if target == external_target),
            )
            .count();
        assert_eq!(
            wormhole_count, 1,
            "expected exactly one wormhole node, routed like any other"
        );
    }

    #[test]
    fn test_layout_layer_keeps_two_wormhole_nodes_at_distinct_positions() {
        let left_nodes = vec![
            LayoutNode::data(NodeIndex::new(1), LocalPos::new_xy(0, 0), (0, 0), None),
            LayoutNode::data(NodeIndex::new(2), LocalPos::new_xy(0, 1), (0, 0), None),
        ];
        let target_a = NodeIndex::new(901);
        let target_b = NodeIndex::new(902);
        let right_nodes = vec![
            LayoutNode::new(
                NodeRole::Wormhole(target_a),
                LocalPos::new_xy(3, 0),
                (0, 0),
                None,
            ),
            LayoutNode::new(
                NodeRole::Wormhole(target_b),
                LocalPos::new_xy(3, 100),
                (0, 0),
                None,
            ),
        ];
        let edges = vec![
            (NodeIndex::new(0), NodeIndex::new(0)), // left[0] to right[0]
            (NodeIndex::new(1), NodeIndex::new(1)), // left[1] to right[1]
        ];
        let edge_bundles = HashMap::new();

        let graph = layout_layer(&left_nodes, &right_nodes, &edges, &edge_bundles)
            .expect("should route two wormhole nodes");

        let wormhole_positions: Vec<_> = graph
            .node_weights()
            .filter_map(|node| match node.role {
                NodeRole::Wormhole(target) if target == target_a || target == target_b => {
                    Some(node.pos)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            wormhole_positions.len(),
            2,
            "both wormhole nodes must be present"
        );
        assert_ne!(
            wormhole_positions[0], wormhole_positions[1],
            "distinct wormhole nodes must land at distinct positions"
        );
    }
}
