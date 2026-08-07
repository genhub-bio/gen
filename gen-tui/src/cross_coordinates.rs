//! Brandes–Köpf cross-coordinate assignment for assembled windows.

use std::collections::{BTreeMap, HashMap, HashSet};

use petgraph::{
    Undirected,
    graph::NodeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

use crate::{
    distribute_nodes::base_step,
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

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

/// Dense representation used by the four Brandes–Köpf alignment passes.
///
/// Only plain routing nodes are dummy chain segments. Data, pin, and wormhole nodes retain
/// ordinary alignment behavior.
struct CrossGraph {
    column: Vec<i64>,
    order: Vec<i64>,
    height: Vec<i64>,
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

    /// Minimum centre-to-centre distance between two nodes consecutive in a layer, `above` over
    /// `below`, from their real heights plus `min_gap`.
    fn separation(&self, above: usize, below: usize) -> i64 {
        base_step(self.height[above], self.height[below]) + self.min_gap
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
