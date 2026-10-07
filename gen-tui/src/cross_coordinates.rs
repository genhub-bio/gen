//! Brandes–Köpf cross-coordinate assignment for assembled windows.

use std::collections::{BTreeMap, HashMap, HashSet};

use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    unionfind::UnionFind,
    visit::{EdgeRef, IntoEdgeReferences},
};

use crate::{
    compaction::{Constraint, compact_axis_with_centering},
    distribute_nodes::base_step,
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

/// Mirror simple branch-and-merge bubbles when this moves centered nodes toward the first row.
/// Sugiyama's routing vertices stay on their branch; cycle-bypass edges are excluded. The
/// ordinal y values produced by assembly are still intact at this point.
pub fn orient_centered_bubbles(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    centered: &HashSet<NodeIndex>,
) {
    if centered.is_empty() {
        return;
    }

    let mut layers: BTreeMap<i64, Vec<NodeIndex<u32>>> = BTreeMap::new();
    for node_index in graph.node_indices() {
        layers
            .entry(graph[node_index].pos.x)
            .or_default()
            .push(node_index);
    }
    let mut layers: Vec<Vec<NodeIndex<u32>>> = layers.into_values().collect();
    for layer in &mut layers {
        layer.sort_by_key(|&node_index| graph[node_index].pos.y);
    }

    let backward_bundles: HashSet<_> = graph
        .edge_references()
        .filter(|edge| edge.weight().is_backward_span)
        .flat_map(|edge| edge.weight().bundle.iter().copied())
        .collect();
    let mut outgoing: HashMap<NodeIndex<u32>, Vec<NodeIndex<u32>>> = HashMap::new();
    let mut incoming: HashMap<NodeIndex<u32>, Vec<NodeIndex<u32>>> = HashMap::new();
    for edge in graph.edge_references() {
        let (first, second) = (edge.source(), edge.target());
        if matches!(graph[first].role, NodeRole::Pin | NodeRole::Wormhole(_))
            || matches!(graph[second].role, NodeRole::Pin | NodeRole::Wormhole(_))
            || edge.weight().is_backward_span
            || edge
                .weight()
                .bundle
                .iter()
                .any(|bundle| backward_bundles.contains(bundle))
        {
            continue;
        }
        let (left, right) = if graph[first].pos.x + 1 == graph[second].pos.x {
            (first, second)
        } else if graph[second].pos.x + 1 == graph[first].pos.x {
            (second, first)
        } else {
            continue;
        };
        outgoing.entry(left).or_default().push(right);
        incoming.entry(right).or_default().push(left);
    }
    for neighbours in outgoing.values_mut().chain(incoming.values_mut()) {
        neighbours.sort_unstable();
        neighbours.dedup();
    }

    let columns: HashMap<i64, usize> = layers
        .iter()
        .enumerate()
        .map(|(index, layer)| (graph[layer[0]].pos.x, index))
        .collect();
    for source in graph.node_indices().collect::<Vec<_>>() {
        if !matches!(graph[source].role, NodeRole::Data(_)) {
            continue;
        }
        let Some(branches) = outgoing.get(&source).filter(|branches| branches.len() > 1) else {
            continue;
        };
        let mut sink = None;
        let mut interior = HashSet::new();
        let mut valid = true;
        for &branch in branches {
            let mut current = branch;
            loop {
                if incoming
                    .get(&current)
                    .is_some_and(|parents| parents.len() > 1)
                {
                    if sink.is_some_and(|existing| existing != current) {
                        valid = false;
                    }
                    sink = Some(current);
                    break;
                }
                if !interior.insert(current) {
                    valid = false;
                    break;
                }
                let Some(children) = outgoing
                    .get(&current)
                    .filter(|children| children.len() == 1)
                else {
                    valid = false;
                    break;
                };
                current = children[0];
            }
            if !valid {
                break;
            }
        }
        let Some(sink) = sink.filter(|_| valid && !interior.is_empty()) else {
            continue;
        };
        if !matches!(graph[sink].role, NodeRole::Data(_))
            || !incoming[&sink]
                .iter()
                .all(|parent| *parent == source || interior.contains(parent))
        {
            continue;
        }

        let mut bubble_layers: BTreeMap<i64, Vec<NodeIndex<u32>>> = BTreeMap::new();
        for &node_index in &interior {
            bubble_layers
                .entry(graph[node_index].pos.x)
                .or_default()
                .push(node_index);
        }
        let mut ranges = Vec::new();
        let mut improvement = 0_isize;
        for (column, members) in bubble_layers {
            let layer_index = columns[&column];
            let layer = &layers[layer_index];
            let positions: Vec<_> = layer
                .iter()
                .enumerate()
                .filter_map(|(position, node_index)| {
                    interior.contains(node_index).then_some(position)
                })
                .collect();
            let start = positions[0];
            if positions.len() != members.len()
                || positions.last() != Some(&(start + members.len() - 1))
            {
                valid = false;
                break;
            }
            let end = start + members.len();
            let block = &layer[start..end];
            if let Some(position) = centered_position(graph, block, centered) {
                improvement += 2 * position as isize - (block.len() - 1) as isize;
            }
            ranges.push((layer_index, start, end));
        }
        if !valid || improvement <= 0 {
            continue;
        }
        for (layer_index, start, end) in ranges {
            let layer = &mut layers[layer_index];
            layer[start..end].reverse();
            for (position, &node_index) in layer[start..end].iter().enumerate() {
                graph[node_index].pos.y = (start + position) as i64;
            }
        }
    }
}

fn centered_position(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    block: &[NodeIndex<u32>],
    centered: &HashSet<NodeIndex>,
) -> Option<usize> {
    let centered_data: Vec<_> = block
        .iter()
        .enumerate()
        .filter_map(|(position, &node_index)| match graph[node_index].role {
            NodeRole::Data(domain_index) if centered.contains(&domain_index) => Some(position),
            _ => None,
        })
        .collect();
    match centered_data.as_slice() {
        [position] => return Some(*position),
        [] => {}
        _ => return None,
    }
    let centered_routing: Vec<_> = block
        .iter()
        .enumerate()
        .filter(|&(_, &node_index)| {
            matches!(graph[node_index].role, NodeRole::Routing)
                && graph
                    .edges(node_index)
                    .flat_map(|edge| edge.weight().bundle.iter())
                    .any(|(source, target)| centered.contains(source) || centered.contains(target))
        })
        .map(|(position, _)| position)
        .collect();
    match centered_routing.as_slice() {
        [position] => Some(*position),
        _ => None,
    }
}

/// Assign size-aware cross-axis coordinates while preserving Sugiyama's within-layer order.
pub fn assign_cross_coordinates(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    min_gap: i64,
) {
    let min_gap = min_gap.max(1);
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    if node_indices.is_empty() {
        return;
    }
    let dense: HashMap<NodeIndex, usize> = node_indices
        .iter()
        .enumerate()
        .map(|(dense_id, &node_index)| (node_index, dense_id))
        .collect();

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); node_indices.len()];
    for edge in graph.edge_references() {
        let source = dense[&edge.source()];
        let target = dense[&edge.target()];
        adjacency[source].push(target);
        adjacency[target].push(source);
    }
    let cross_graph = CrossGraph {
        column: node_indices
            .iter()
            .map(|&node_index| graph[node_index].pos.x)
            .collect(),
        order: node_indices
            .iter()
            .map(|&node_index| graph[node_index].pos.y)
            .collect(),
        height: node_indices
            .iter()
            .map(|&node_index| graph[node_index].size.1 as i64)
            .collect(),
        center_offset: node_indices
            .iter()
            .map(|&node_index| {
                let node = &graph[node_index];
                node.vertical_anchor.center_offset(node.size.1)
            })
            .collect(),
        is_dummy: node_indices
            .iter()
            .map(|&node_index| matches!(graph[node_index].role, NodeRole::Routing))
            .collect(),
        adjacency,
        min_gap,
    };

    let coordinates = cross_graph.assign();
    for (dense_id, &node_index) in node_indices.iter().enumerate() {
        graph[node_index].pos.y = coordinates[dense_id];
    }
}

/// Place centered nodes on one row using the same symmetric separation solver as compaction.
/// Routing chains that Sugiyama left straight are tied together, so pinning data nodes does not
/// introduce an avoidable bend before edge routing. A centered edge can pin a layer with no
/// centered data node; ambiguous layers are left without an anchor.
pub fn center_layers(
    graph: &mut StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    centered: &HashSet<NodeIndex>,
) {
    if centered.is_empty() {
        return;
    }
    let mut layers: BTreeMap<i64, Vec<NodeIndex<u32>>> = BTreeMap::new();
    for node_index in graph.node_indices() {
        layers
            .entry(graph[node_index].pos.x)
            .or_default()
            .push(node_index);
    }
    let mut constraints = Vec::new();
    let mut anchors = Vec::new();
    for layer in layers.values_mut() {
        layer.sort_by_key(|&node_index| graph[node_index].pos.y);
        for pair in layer.windows(2) {
            constraints.push(Constraint {
                from: pair[0].index(),
                to: pair[1].index(),
                gap: graph[pair[0]].vertical_separation(&graph[pair[1]]) + 1,
            });
        }
        let data: Vec<_> = layer.iter().copied().filter(|&node_index| {
            matches!(graph[node_index].role, NodeRole::Data(domain_index) if centered.contains(&domain_index))
        }).collect();
        match data.as_slice() {
            [only] => anchors.push(*only),
            [] => {
                let routing: Vec<_> = layer
                    .iter()
                    .copied()
                    .filter(|&node_index| {
                        matches!(graph[node_index].role, NodeRole::Routing)
                            && graph
                                .edges(node_index)
                                .flat_map(|edge| edge.weight().bundle.iter())
                                .any(|(source, target)| {
                                    centered.contains(source) || centered.contains(target)
                                })
                    })
                    .collect();
                if let [only] = routing.as_slice() {
                    anchors.push(*only);
                }
            }
            _ => {}
        }
    }
    if anchors.is_empty() {
        return;
    }
    let mut aligned_routing = Vec::new();
    for edge in graph.edge_references() {
        let source = edge.source();
        let target = edge.target();
        if matches!(graph[source].role, NodeRole::Routing)
            && matches!(graph[target].role, NodeRole::Routing)
            && graph[source].pos.y == graph[target].pos.y
        {
            aligned_routing.push((source, target));
        }
    }
    let placement = solve_centered_layers(graph, &constraints, &anchors, &aligned_routing)
        .or_else(|| solve_centered_layers(graph, &constraints, &anchors, &[]));
    let Some(placement) = placement else {
        log::warn!("centered cross-coordinate constraints are infeasible");
        return;
    };
    for node_index in graph.node_indices().collect::<Vec<_>>() {
        graph[node_index].pos.y = placement[&node_index];
    }
}

/// Equality groups are contracted before solving. The generic solver can handle zero-gap cycles,
/// but contraction keeps this frequent pre-routing pass on its acyclic fast path.
fn solve_centered_layers(
    graph: &StableGraph<LayoutNode, LayoutEdge, Undirected, u32>,
    separations: &[Constraint],
    anchors: &[NodeIndex<u32>],
    aligned_routing: &[(NodeIndex<u32>, NodeIndex<u32>)],
) -> Option<HashMap<NodeIndex<u32>, i64>> {
    let node_indices: Vec<_> = graph.node_indices().collect();
    let dense: HashMap<_, _> = node_indices
        .iter()
        .enumerate()
        .map(|(index, &node_index)| (node_index, index))
        .collect();
    let mut groups = UnionFind::<usize>::new(node_indices.len());
    for &node_index in &anchors[1..] {
        groups.union(dense[&anchors[0]], dense[&node_index]);
    }
    for &(source, target) in aligned_routing {
        groups.union(dense[&source], dense[&target]);
    }

    let mut group_ids = HashMap::new();
    let group_of: HashMap<_, _> = node_indices
        .iter()
        .map(|&node_index| {
            let root = groups.find(dense[&node_index]);
            let next_id = group_ids.len();
            let group_id = *group_ids.entry(root).or_insert(next_id);
            (node_index, group_id)
        })
        .collect();
    let mut constraints = Vec::with_capacity(separations.len());
    for &constraint in separations {
        let from = group_of[&NodeIndex::new(constraint.from)];
        let to = group_of[&NodeIndex::new(constraint.to)];
        if from == to {
            return None;
        }
        constraints.push(Constraint {
            from,
            to,
            gap: constraint.gap,
        });
    }
    let placement = compact_axis_with_centering(0..group_ids.len(), &constraints).ok()?;
    let origin = placement.centered[&group_of[&anchors[0]]];
    Some(
        group_of
            .into_iter()
            .map(|(node_index, group_id)| (node_index, placement.centered[&group_id] - origin))
            .collect(),
    )
}

/// Dense representation used by the four Brandes–Köpf alignment passes.
///
/// Only plain routing nodes are dummy chain segments. Data, pin, and wormhole nodes retain
/// ordinary alignment behavior.
struct CrossGraph {
    column: Vec<i64>,
    order: Vec<i64>,
    height: Vec<i64>,
    center_offset: Vec<i64>,
    is_dummy: Vec<bool>,
    adjacency: Vec<Vec<usize>>,
    min_gap: i64,
}

/// The layered view of a [`CrossGraph`]: layers of node ids (ordered), plus reverse lookups.
struct CrossLayered {
    layers: Vec<Vec<usize>>,
    layer_of: Vec<usize>,
    position_of: Vec<usize>,
}

/// A vertical alignment: `root[v]` is the topmost node of `v`'s block, `align[v]` the next node in
/// the block after `v` (cyclic, so `align[last] == root`).
struct CrossAlignment {
    root: Vec<usize>,
    align: Vec<usize>,
}

impl CrossGraph {
    fn len(&self) -> usize {
        self.column.len()
    }

    /// Group nodes into layers by `column` (ascending) with each layer ordered by `(order, id)`,
    /// and index that structure for layer/position/neighbour lookups.
    fn layered(&self) -> CrossLayered {
        let mut columns: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for node_index in 0..self.len() {
            columns
                .entry(self.column[node_index])
                .or_default()
                .push(node_index);
        }
        let mut layers: Vec<Vec<usize>> = columns.into_values().collect();
        for layer in &mut layers {
            layer.sort_by_key(|&node_index| (self.order[node_index], node_index));
        }
        let mut layer_of = vec![0usize; self.len()];
        let mut position_of = vec![0usize; self.len()];
        for (layer_index, layer) in layers.iter().enumerate() {
            for (position, &node_index) in layer.iter().enumerate() {
                layer_of[node_index] = layer_index;
                position_of[node_index] = position;
            }
        }
        CrossLayered {
            layers,
            layer_of,
            position_of,
        }
    }

    /// Minimum attachment-row distance for consecutive nodes in increasing world Y.
    fn separation(&self, lower: usize, upper: usize) -> i64 {
        base_step(self.height[lower], self.height[upper]) + self.center_offset[lower]
            - self.center_offset[upper]
            + self.min_gap
    }

    /// Run the four Brandes–Köpf passes and balance them into one coordinate per node.
    fn assign(&self) -> Vec<i64> {
        let layered = self.layered();
        let adjacent = AdjacentNeighbours::new(self, &layered);
        let marked = mark_type1_conflicts(self, &layered, &adjacent);

        let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(4);
        for &align_to_upper in &[true, false] {
            for &prefer_low in &[true, false] {
                let alignment = vertical_alignment(
                    self,
                    &layered,
                    &adjacent,
                    &marked,
                    align_to_upper,
                    prefer_low,
                );
                candidates.push(horizontal_compaction(
                    self, &layered, &alignment, prefer_low,
                ));
            }
        }
        balance(self, candidates)
    }
}

/// Per-node adjacent-layer neighbour lists in within-layer order, computed once per solve.
/// `CrossLayered::upper_neighbours`/`lower_neighbours` refilter and re-sort the adjacency list on
/// every call, and the passes query the same lists many times over.
struct AdjacentNeighbours {
    upper: Vec<Vec<usize>>,
    lower: Vec<Vec<usize>>,
}

impl AdjacentNeighbours {
    fn new(graph: &CrossGraph, layered: &CrossLayered) -> AdjacentNeighbours {
        AdjacentNeighbours {
            upper: (0..graph.len())
                .map(|node_index| layered.upper_neighbours(graph, node_index))
                .collect(),
            lower: (0..graph.len())
                .map(|node_index| layered.lower_neighbours(graph, node_index))
                .collect(),
        }
    }
}

impl CrossLayered {
    /// Neighbours in the layer immediately to the left (`column - 1`), in within-layer order.
    fn upper_neighbours(&self, graph: &CrossGraph, node_index: usize) -> Vec<usize> {
        match self.layer_of[node_index].checked_sub(1) {
            Some(layer_index) => self.neighbours_in_layer(graph, node_index, layer_index),
            None => Vec::new(),
        }
    }

    /// Neighbours in the layer immediately to the right (`column + 1`), in within-layer order.
    fn lower_neighbours(&self, graph: &CrossGraph, node_index: usize) -> Vec<usize> {
        self.neighbours_in_layer(graph, node_index, self.layer_of[node_index] + 1)
    }

    fn neighbours_in_layer(
        &self,
        graph: &CrossGraph,
        node_index: usize,
        layer_index: usize,
    ) -> Vec<usize> {
        let mut neighbours: Vec<usize> = graph.adjacency[node_index]
            .iter()
            .copied()
            .filter(|&neighbour| self.layer_of[neighbour] == layer_index)
            .collect();
        neighbours.sort_by_key(|&neighbour| self.position_of[neighbour]);
        neighbours
    }
}

/// Mark type-1 conflicts: non-inner segments that cross an inner (dummy–dummy) segment. Marked
/// segments are excluded from alignment so long-edge/bypass dummy chains stay straight. Returned as
/// `(upper endpoint, lower endpoint)` pairs, upper meaning the smaller layer.
fn mark_type1_conflicts(
    graph: &CrossGraph,
    layered: &CrossLayered,
    adjacent: &AdjacentNeighbours,
) -> HashSet<(usize, usize)> {
    let mut marked = HashSet::new();
    let layer_count = layered.layers.len();
    if layer_count < 2 {
        return marked;
    }

    let is_inner = |upper: usize, lower: usize| graph.is_dummy[upper] && graph.is_dummy[lower];

    for layer_index in 0..layer_count - 1 {
        let lower_layer = &layered.layers[layer_index + 1];
        let mut boundary_position = 0usize;
        let mut previous_scanned = 0usize;
        for (scan_position, &lower_node) in lower_layer.iter().enumerate() {
            let inner_upper = adjacent.upper[lower_node]
                .iter()
                .copied()
                .find(|&upper| is_inner(upper, lower_node));
            let at_layer_end = scan_position == lower_layer.len() - 1;
            if inner_upper.is_none() && !at_layer_end {
                continue;
            }
            let next_boundary = match inner_upper {
                Some(upper) => layered.position_of[upper],
                None => layered.layers[layer_index].len().saturating_sub(1),
            };
            for &scanned_node in &lower_layer[previous_scanned..=scan_position] {
                for &upper in &adjacent.upper[scanned_node] {
                    let upper_position = layered.position_of[upper];
                    let crosses =
                        upper_position < boundary_position || upper_position > next_boundary;
                    if crosses && !is_inner(upper, scanned_node) {
                        marked.insert((upper, scanned_node));
                    }
                }
            }
            previous_scanned = scan_position + 1;
            boundary_position = next_boundary;
        }
    }
    marked
}

/// Align each node to the median of its unmarked neighbours in the adjacent layer, keeping a
/// layer's alignment edges mutually non-crossing so blocks never cross. `align_to_upper` picks which
/// adjacent layer to align against; `prefer_low` picks the lower-position median on ties (and scans
/// low-to-high), otherwise the higher-position median scanning high-to-low.
///
/// Mirrors `gen_sugiyama::p3_calculate_coordinates::create_vertical_alignments`: one scan per
/// layer, trying the lower-then-upper (or reverse, per `prefer_low`) median candidate for each
/// node with no dummy/data distinction. A candidate is skipped only when its segment is a marked
/// type-1 conflict (see `mark_type1_conflicts`) or already claimed, and a claim is rejected if it
/// would cross an alignment edge already made in this layer.
fn vertical_alignment(
    graph: &CrossGraph,
    layered: &CrossLayered,
    adjacent: &AdjacentNeighbours,
    marked: &HashSet<(usize, usize)>,
    align_to_upper: bool,
    prefer_low: bool,
) -> CrossAlignment {
    let node_count = graph.len();
    let mut root: Vec<usize> = (0..node_count).collect();
    let mut align: Vec<usize> = (0..node_count).collect();
    let mut claimed = vec![false; node_count];

    let layer_order: Vec<usize> = if align_to_upper {
        (0..layered.layers.len()).collect()
    } else {
        (0..layered.layers.len()).rev().collect()
    };

    for layer_index in layer_order {
        let scan: Vec<usize> = if prefer_low {
            layered.layers[layer_index].clone()
        } else {
            layered.layers[layer_index].iter().rev().copied().collect()
        };
        let mut claims: Vec<(i64, i64)> = Vec::new();
        for &node_index in &scan {
            let neighbours = if align_to_upper {
                &adjacent.upper[node_index]
            } else {
                &adjacent.lower[node_index]
            };
            if neighbours.is_empty() {
                continue;
            }
            let degree = neighbours.len();
            let median_pair = [(degree - 1) / 2, degree / 2];
            let candidate_order = if prefer_low {
                median_pair
            } else {
                [median_pair[1], median_pair[0]]
            };
            for median_index in candidate_order {
                if align[node_index] != node_index {
                    break;
                }
                let medium = neighbours[median_index];
                let segment = if align_to_upper {
                    (medium, node_index)
                } else {
                    (node_index, medium)
                };
                if marked.contains(&segment) || claimed[medium] {
                    continue;
                }
                let node_position = layered.position_of[node_index] as i64;
                let medium_position = layered.position_of[medium] as i64;
                let crosses = claims.iter().any(|&(claim_node, claim_medium)| {
                    (node_position - claim_node) * (medium_position - claim_medium) < 0
                });
                if crosses {
                    continue;
                }
                align[medium] = node_index;
                root[node_index] = root[medium];
                align[node_index] = root[node_index];
                claims.push((node_position, medium_position));
                claimed[medium] = true;
            }
        }
    }
    CrossAlignment { root, align }
}

/// Place every block root, then every node, so consecutive nodes in a layer keep at least
/// `graph.separation(above, below)` between them. `prefer_low` matches the alignment pass: a
/// low-preferring pass packs blocks toward low coordinates, a high-preferring one toward high
/// coordinates (compacting the mirrored order and negating).
fn horizontal_compaction(
    graph: &CrossGraph,
    layered: &CrossLayered,
    alignment: &CrossAlignment,
    prefer_low: bool,
) -> Vec<f64> {
    let node_count = graph.len();
    let mut sink: Vec<usize> = (0..node_count).collect();
    let mut shift: Vec<f64> = vec![f64::INFINITY; node_count];
    let mut inner_coordinate: Vec<Option<f64>> = vec![None; node_count];

    let predecessor_of = |node_index: usize| -> Option<usize> {
        let position = layered.position_of[node_index];
        let layer = &layered.layers[layered.layer_of[node_index]];
        if prefer_low {
            position.checked_sub(1).map(|earlier| layer[earlier])
        } else {
            layer.get(position + 1).copied()
        }
    };
    let separation_to = |node_index: usize, predecessor: usize| -> f64 {
        let gap = if prefer_low {
            graph.separation(predecessor, node_index)
        } else {
            graph.separation(node_index, predecessor)
        };
        gap as f64
    };

    fn place_block(
        root_node: usize,
        alignment: &CrossAlignment,
        sink: &mut Vec<usize>,
        shift: &mut Vec<f64>,
        inner_coordinate: &mut Vec<Option<f64>>,
        predecessor_of: &dyn Fn(usize) -> Option<usize>,
        separation_to: &dyn Fn(usize, usize) -> f64,
    ) {
        if inner_coordinate[root_node].is_some() {
            return;
        }
        inner_coordinate[root_node] = Some(0.0);
        let mut current = root_node;
        loop {
            if let Some(predecessor) = predecessor_of(current) {
                let predecessor_root = alignment.root[predecessor];
                place_block(
                    predecessor_root,
                    alignment,
                    sink,
                    shift,
                    inner_coordinate,
                    predecessor_of,
                    separation_to,
                );
                if sink[root_node] == root_node {
                    sink[root_node] = sink[predecessor_root];
                }
                let required = inner_coordinate[predecessor_root].unwrap()
                    + separation_to(current, predecessor);
                if sink[root_node] == sink[predecessor_root] {
                    let updated = inner_coordinate[root_node].unwrap().max(required);
                    inner_coordinate[root_node] = Some(updated);
                } else {
                    let candidate_shift = inner_coordinate[root_node].unwrap() - required;
                    shift[sink[predecessor_root]] =
                        shift[sink[predecessor_root]].min(candidate_shift);
                }
            }
            current = alignment.align[current];
            if current == root_node {
                break;
            }
        }
    }

    for node_index in 0..node_count {
        if alignment.root[node_index] == node_index {
            place_block(
                node_index,
                alignment,
                &mut sink,
                &mut shift,
                &mut inner_coordinate,
                &predecessor_of,
                &separation_to,
            );
        }
    }

    let mut coordinates: Vec<f64> = (0..node_count)
        .map(|node_index| {
            let root_node = alignment.root[node_index];
            let mut coordinate = inner_coordinate[root_node].unwrap();
            let sink_shift = shift[sink[root_node]];
            if sink_shift < f64::INFINITY {
                coordinate += sink_shift;
            }
            coordinate
        })
        .collect();

    if !prefer_low {
        for coordinate in &mut coordinates {
            *coordinate = -*coordinate;
        }
    }
    coordinates
}

/// Align the four candidate coordinate sets to a common minimum, take each node's median (average
/// of the two middle values), and shift so the minimum coordinate is 0.
///
/// The paper aligns left-directed runs to the narrowest candidate's minimum and right-directed runs
/// to its maximum. When candidate widths differ that pins the two ends of a mirror-image pass pair
/// at different offsets, skewing the medians by half the width difference — enough to break exact
/// row co-registration of the grid fixtures. Aligning every candidate to a common minimum keeps
/// mirror pairs coincident, and the final normalisation to `min = 0` makes the reference irrelevant.
fn balance(graph: &CrossGraph, mut candidates: Vec<Vec<f64>>) -> Vec<i64> {
    let node_count = graph.len();
    if node_count == 0 {
        return Vec::new();
    }

    for coordinates in candidates.iter_mut() {
        let minimum = coordinates.iter().cloned().fold(f64::INFINITY, f64::min);
        for coordinate in coordinates.iter_mut() {
            *coordinate -= minimum;
        }
    }

    let mut balanced = vec![0.0f64; node_count];
    for node_index in 0..node_count {
        let values: Vec<f64> = candidates
            .iter()
            .map(|coordinates| coordinates[node_index])
            .collect();
        balanced[node_index] = values.iter().sum::<f64>() / values.len() as f64;
    }

    let minimum = balanced.iter().cloned().fold(f64::INFINITY, f64::min);
    balanced
        .iter()
        .map(|&coordinate| round_half_even(coordinate - minimum))
        .collect()
}

/// Round to the nearest integer, ties to even, so a pair of half-integer values symmetric about an
/// integer centre stays symmetric after rounding (unlike round-half-away-from-zero).
fn round_half_even(value: f64) -> i64 {
    let floor = value.floor();
    let fraction = value - floor;
    let floor = floor as i64;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1
    } else if floor % 2 == 0 {
        floor
    } else {
        floor + 1
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::{Undirected, graph::NodeIndex, stable_graph::StableGraph};

    use crate::{
        assembly::assemble_window,
        crawl::{EagerSource, GraphCursor, build_window_graph, neighborhood},
        cross_coordinates::{assign_cross_coordinates, center_layers, orient_centered_bubbles},
        geometry::LocalPos,
        layout::{LayoutEdge, LayoutNode, NodeRole},
        testing::mocks::MockDomainGraph,
    };

    #[test]
    fn test_orient_centered_three_layer_bubble() {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let left = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos::new_xy(0, 0),
            (1, 1),
            Some(0),
        ));
        let alternate = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos::new_xy(1, 0),
            (1, 1),
            Some(1),
        ));
        let reference = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos::new_xy(1, 1),
            (1, 1),
            Some(1),
        ));
        let right = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos::new_xy(2, 0),
            (1, 1),
            Some(2),
        ));
        for middle in [alternate, reference] {
            graph.add_edge(left, middle, LayoutEdge::empty());
            graph.add_edge(middle, right, LayoutEdge::empty());
        }

        orient_centered_bubbles(&mut graph, &HashSet::from([NodeIndex::new(2)]));

        assert_eq!(graph[reference].pos.y, 0);
        assert_eq!(graph[alternate].pos.y, 1);
        assert_eq!(graph[left].pos.y, 0);
        assert_eq!(graph[right].pos.y, 0);
    }

    #[test]
    fn test_orient_centered_bubble_mirrors_all_interior_layers() {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let mut nodes = Vec::new();
        for (column, order) in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)] {
            let domain_index = NodeIndex::new(nodes.len());
            nodes.push(graph.add_node(LayoutNode::data(
                domain_index,
                LocalPos::new_xy(column, order),
                (1, 1),
                Some(column as i32),
            )));
        }
        for (source, target) in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)] {
            graph.add_edge(nodes[source], nodes[target], LayoutEdge::empty());
        }

        orient_centered_bubbles(
            &mut graph,
            &HashSet::from([NodeIndex::new(2), NodeIndex::new(4)]),
        );

        assert_eq!(graph[nodes[2]].pos.y, 0);
        assert_eq!(graph[nodes[4]].pos.y, 0);
        assert_eq!(graph[nodes[1]].pos.y, 1);
        assert_eq!(graph[nodes[3]].pos.y, 1);
    }

    #[test]
    fn test_orient_centered_bubble_skips_unconnected_layers() {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let positions = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)];
        for (domain_index, (column, order)) in positions.into_iter().enumerate() {
            graph.add_node(LayoutNode::data(
                NodeIndex::new(domain_index),
                LocalPos::new_xy(column, order),
                (1, 1),
                Some(column as i32),
            ));
        }

        orient_centered_bubbles(&mut graph, &HashSet::from([NodeIndex::new(2)]));

        assert_eq!(graph[NodeIndex::new(2)].pos.y, 1);
    }

    #[test]
    fn test_orient_centered_bubble_skips_dead_end_branch() {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let nodes: Vec<_> = [(0, 0), (1, 0), (1, 1), (2, 0)]
            .into_iter()
            .enumerate()
            .map(|(domain_index, (column, order))| {
                graph.add_node(LayoutNode::data(
                    NodeIndex::new(domain_index),
                    LocalPos::new_xy(column, order),
                    (1, 1),
                    Some(column as i32),
                ))
            })
            .collect();
        for (source, target) in [(0, 1), (0, 2), (1, 3)] {
            graph.add_edge(nodes[source], nodes[target], LayoutEdge::empty());
        }

        orient_centered_bubbles(&mut graph, &HashSet::from([NodeIndex::new(2)]));

        assert_eq!(graph[nodes[2]].pos.y, 1);
    }

    #[test]
    fn test_orient_centered_bubble_inside_cycle_bypass() {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let backward = (NodeIndex::new(3), NodeIndex::new(0));
        let left_pin = graph.add_node(LayoutNode::new(
            crate::layout::NodeRole::Pin,
            LocalPos::new_xy(0, 0),
            (1, 1),
            None,
        ));
        let source = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos::new_xy(0, 1),
            (1, 1),
            Some(0),
        ));
        let bypass = graph.add_node(LayoutNode::routing(LocalPos::new_xy(1, 0), (1, 1)));
        let alternate = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos::new_xy(1, 1),
            (1, 1),
            Some(1),
        ));
        let reference = graph.add_node(LayoutNode::data(
            NodeIndex::new(2),
            LocalPos::new_xy(1, 2),
            (1, 1),
            Some(1),
        ));
        let right_pin = graph.add_node(LayoutNode::new(
            crate::layout::NodeRole::Pin,
            LocalPos::new_xy(2, 0),
            (1, 1),
            None,
        ));
        let sink = graph.add_node(LayoutNode::data(
            NodeIndex::new(3),
            LocalPos::new_xy(2, 1),
            (1, 1),
            Some(2),
        ));
        for middle in [alternate, reference] {
            graph.add_edge(source, middle, LayoutEdge::empty());
            graph.add_edge(middle, sink, LayoutEdge::empty());
        }
        for (first, second) in [(left_pin, source), (sink, right_pin)] {
            graph.add_edge(first, second, LayoutEdge::new(backward.0, backward.1));
        }
        for (first, second) in [(left_pin, bypass), (bypass, right_pin)] {
            let mut edge = LayoutEdge::new(backward.0, backward.1);
            edge.is_backward_span = true;
            graph.add_edge(first, second, edge);
        }

        orient_centered_bubbles(&mut graph, &HashSet::from([NodeIndex::new(2)]));

        assert_eq!(graph[reference].pos.y, 1);
        assert_eq!(graph[alternate].pos.y, 2);
        assert_eq!(graph[bypass].pos.y, 0);
    }

    #[test]
    fn test_orient_centered_bubble_after_circular_sugiyama() {
        let mut domain_graph = MockDomainGraph::new();
        let [source, alternate, reference, sink] = [(); 4].map(|_| domain_graph.add_node(()));
        for (from, to) in [
            (source, alternate),
            (source, reference),
            (alternate, sink),
            (reference, sink),
            (sink, source),
        ] {
            domain_graph.add_edge(from, to, ());
        }
        let backward_edges = HashSet::from([(sink, source)]);
        let subgraph = neighborhood(
            source,
            10,
            &mut GraphCursor::new(&mut domain_graph, &mut EagerSource),
            &|_| false,
            &std::collections::HashMap::new(),
        )
        .expect("should crawl circular bubble");
        let (window, _) = build_window_graph(&subgraph, &domain_graph, Some(&backward_edges))
            .expect("should build circular window");
        let mut assembled = assemble_window(&window).expect("should assemble circular window");
        let bubble_nodes: Vec<_> = assembled
            .graph
            .node_indices()
            .filter(|&node_index| {
                matches!(assembled.graph[node_index].role, NodeRole::Data(domain_index) if domain_index == alternate || domain_index == reference)
            })
            .collect();
        assert_eq!(bubble_nodes.len(), 2);
        let (first, second) =
            if assembled.graph[bubble_nodes[0]].pos.y < assembled.graph[bubble_nodes[1]].pos.y {
                (bubble_nodes[0], bubble_nodes[1])
            } else {
                (bubble_nodes[1], bubble_nodes[0])
            };
        let centered_domain = match assembled.graph[second].role {
            NodeRole::Data(domain_index) => domain_index,
            _ => unreachable!(),
        };
        let routing_rows: Vec<_> = assembled
            .graph
            .node_indices()
            .filter(|&node_index| {
                matches!(assembled.graph[node_index].role, NodeRole::Routing)
                    && assembled.graph[node_index].pos.x == assembled.graph[second].pos.x
            })
            .map(|node_index| (node_index, assembled.graph[node_index].pos.y))
            .collect();
        assert!(
            !routing_rows.is_empty(),
            "cycle bypass should cross the bubble layer"
        );

        orient_centered_bubbles(&mut assembled.graph, &HashSet::from([centered_domain]));

        assert!(assembled.graph[second].pos.y < assembled.graph[first].pos.y);
        for (node_index, row) in routing_rows {
            assert_eq!(assembled.graph[node_index].pos.y, row);
        }

        assign_cross_coordinates(&mut assembled.graph, 1);
        center_layers(
            &mut assembled.graph,
            &HashSet::from([source, centered_domain, sink]),
        );
        let bypass_rows: Vec<_> = assembled
            .graph
            .node_indices()
            .filter(|&node_index| {
                matches!(assembled.graph[node_index].role, NodeRole::Routing)
                    && assembled
                        .graph
                        .edges(node_index)
                        .any(|edge| edge.weight().bundle.contains(&(sink, source)))
            })
            .map(|node_index| assembled.graph[node_index].pos.y)
            .collect();
        assert_eq!(bypass_rows.len(), 3);
        assert!(bypass_rows.iter().all(|&row| row == bypass_rows[0]));
    }

    #[test]
    fn test_center_layers_balances_routing_around_centered_data() {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let first_routing = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, -5), (1, 1)));
        let first = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos::new_xy(0, 3),
            (1, 1),
            Some(0),
        ));
        let second_routing = graph.add_node(LayoutNode::routing(LocalPos::new_xy(1, 0), (1, 1)));
        let second = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos::new_xy(1, 3),
            (1, 1),
            Some(1),
        ));

        center_layers(
            &mut graph,
            &HashSet::from([NodeIndex::new(0), NodeIndex::new(1)]),
        );

        assert_eq!(graph[first].pos.y, 0);
        assert_eq!(graph[second].pos.y, 0);
        assert_eq!(graph[first_routing].pos.y, -2);
        assert_eq!(graph[second_routing].pos.y, -2);
    }

    #[test]
    fn test_center_layers_relaxes_incompatible_routing_alignment() {
        let mut graph: StableGraph<LayoutNode, LayoutEdge, Undirected, u32> =
            StableGraph::default();
        let first_routing = graph.add_node(LayoutNode::routing(LocalPos::new_xy(0, 0), (1, 1)));
        let first_data = graph.add_node(LayoutNode::data(
            NodeIndex::new(0),
            LocalPos::new_xy(0, 2),
            (1, 1),
            Some(0),
        ));
        let second_data = graph.add_node(LayoutNode::data(
            NodeIndex::new(1),
            LocalPos::new_xy(1, -2),
            (1, 1),
            Some(1),
        ));
        let second_routing = graph.add_node(LayoutNode::routing(LocalPos::new_xy(1, 0), (1, 1)));
        graph.add_edge(first_routing, second_routing, LayoutEdge::empty());

        center_layers(
            &mut graph,
            &HashSet::from([NodeIndex::new(0), NodeIndex::new(1)]),
        );

        assert_eq!(graph[first_data].pos.y, 0);
        assert_eq!(graph[second_data].pos.y, 0);
        assert!(graph[first_routing].pos.y <= -2);
        assert!(graph[second_routing].pos.y >= 2);
    }
}
