#[cfg(test)]
mod tests;
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::{Deref, DerefMut},
};

use log::{debug, info, trace};
use petgraph::{
    Direction::{Incoming, Outgoing},
    algo::toposort,
    stable_graph::{NodeIndex, StableDiGraph},
    visit::EdgeRef,
};

use super::{
    Edge, Vertex,
    config::CrossingMinimization,
    slack,
    util::{IterDir, iterate, radix_sort},
};

#[derive(Clone)]
struct Order {
    _inner: Vec<Vec<NodeIndex>>,
    positions: HashMap<NodeIndex, usize>,
}

impl Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for row in &self._inner {
            for c in row {
                s.push_str(&c.index().to_string());
                s.push(',')
            }
            s.push('\n');
        }
        f.write_str(&s)
    }
}

impl Order {
    fn new(layers: Vec<Vec<NodeIndex>>) -> Self {
        let mut positions = HashMap::new();
        for l in &layers {
            for (pos, v) in l.iter().enumerate() {
                positions.insert(*v, pos);
            }
        }
        Self {
            _inner: layers,
            positions,
        }
    }

    fn max_rank(&self) -> usize {
        self.len()
    }

    fn exchange(&mut self, a: usize, b: usize, r: usize) {
        // first update positions, then swap
        *self.positions.get_mut(&self._inner[r][a]).unwrap() = b;
        *self.positions.get_mut(&self._inner[r][b]).unwrap() = a;
        self._inner[r].swap(a, b);
    }

    fn cross_count_two_vertices(
        &self,
        v: NodeIndex,
        w: NodeIndex,
        graph: &StableDiGraph<Vertex, Edge>,
    ) -> usize {
        let mut crossings = 0;
        for dir in [Incoming, Outgoing] {
            let mut v_adjacent = graph
                .neighbors_directed(v, dir)
                .map(|n| *self.positions.get(&n).unwrap())
                .collect::<Vec<_>>();
            let mut w_adjacent = graph
                .neighbors_directed(w, dir)
                .map(|n| *self.positions.get(&n).unwrap())
                .collect::<Vec<_>>();
            v_adjacent.sort();
            w_adjacent.sort();
            crossings += Self::calculate_cross_count_two_vertices(&v_adjacent, &w_adjacent);
        }
        crossings
    }

    fn calculate_cross_count_two_vertices(v_adjacent: &[usize], w_adjacent: &[usize]) -> usize {
        let mut all_crossings = 0;
        let mut k = 0;
        for i in v_adjacent {
            let i = *i;
            let mut crossings = k;
            while k < w_adjacent.len() && w_adjacent[k] < i {
                let j = w_adjacent[k];
                if i > j {
                    crossings += 1;
                }
                k += 1;
            }
            all_crossings += crossings;
        }
        all_crossings
    }

    fn crossings(&self, graph: &StableDiGraph<Vertex, Edge>) -> usize {
        let mut cross_count = 0;
        for rank in 0..self.max_rank() - 1 {
            cross_count += self.bilayer_cross_count(graph, rank);
        }
        cross_count
    }

    fn bilayer_cross_count(&self, graph: &StableDiGraph<Vertex, Edge>, rank: usize) -> usize {
        // find initial edge order
        let north = &self[rank];
        let south = &self[rank + 1];
        let mut len = south.len();
        let mut key_length = 0;
        while len > 0 {
            len /= 10;
            key_length += 1;
        }
        let edge_endpoint_positions = north
            .iter()
            .flat_map(|v| {
                radix_sort(
                    graph
                        .neighbors_directed(*v, Outgoing)
                        .filter(|n| graph[*v].rank.abs_diff(graph[*n].rank) == 1)
                        .filter_map(|n| self.positions.get(&n))
                        .copied()
                        .collect(),
                    key_length,
                )
            })
            .collect::<Vec<_>>();
        Self::count_crossings(edge_endpoint_positions, south.len())
    }

    fn count_crossings(endpoints: Vec<usize>, south_len: usize) -> usize {
        // build the accumulator tree
        let mut c = 0;
        while 1 << c < south_len {
            c += 1
        }
        let tree_size = (1 << (c + 1)) - 1;
        let first_index = (1 << c) - 1;
        let mut tree = vec![0; tree_size];

        let mut cross_count = 0;

        // traverse through the positions and adjust tree nodes
        for pos in endpoints {
            let mut index = pos + first_index;
            tree[index] += 1;
            while index > 0 {
                // traverse up the tree, incrementing the nodes of the tree
                // each time we visit them.
                //
                // When visiting a left node, add the value of the node on the right to
                // the cross count;
                if index % 2 == 1 {
                    cross_count += tree[index + 1]
                }
                index = (index - 1) / 2;
                tree[index] += 1;
            }
        }
        cross_count
    }

    #[allow(dead_code)]
    fn print(&self) {
        for line in &self._inner {
            for v in line {
                print!("{v:>2?} ");
            }
            println!();
        }
    }
}

impl Deref for Order {
    type Target = Vec<Vec<NodeIndex>>;

    fn deref(&self) -> &Self::Target {
        &self._inner
    }
}

impl DerefMut for Order {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self._inner
    }
}

pub(super) fn insert_dummy_vertices(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    dummy_size: f64,
) -> Vec<(NodeIndex, NodeIndex, Edge)> {
    // find all edges that have slack of greater than 0.
    // and insert dummy vertices
    info!(target: "crossing_reduction", "Inserting dummy vertices for edges spanning more than {minimum_length} ranks");
    let mut removed_edges = Vec::new();
    for edge in graph.edge_indices().collect::<Vec<_>>() {
        if slack(graph, edge, minimum_length) > 0 {
            let (mut tail, head) = graph.edge_endpoints(edge).unwrap();

            trace!(target: "crossing_reduction", 
                "Inserting {} dummy vertices between: ({}, {})", 
                graph[head].rank - graph[tail].rank - 1, 
                tail.index(), 
                head.index());

            // Store the removed edge data
            let edge_data = graph.remove_edge(edge).unwrap();
            removed_edges.push((tail, head, edge_data));

            for rank in (graph[tail].rank + 1)..graph[head].rank {
                // usize usize::MAX id as reserved value for a dummy vertex
                let d = Vertex {
                    is_dummy: true,
                    size: (dummy_size, 0.0),
                    ..Default::default()
                };
                let new = graph.add_node(d);
                graph[new].align = new;
                graph[new].root = new;
                graph[new].sink = new;
                graph[new].rank = rank;
                let edge = if let Some(label) = edge_data.input_node_idx_pair {
                    Edge::default().with_label(label)
                } else {
                    Edge::default()
                };
                graph.add_edge(tail, new, edge);
                tail = new;
            }

            let edge = if let Some(label) = edge_data.input_node_idx_pair {
                Edge::default().with_label(label)
            } else {
                Edge::default()
            };
            graph.add_edge(tail, head, edge); // add last dummy edge connecting to the head
        }
    }
    removed_edges
}

pub(super) fn redistribute_dummy_vertices(graph: &mut StableDiGraph<Vertex, Edge>) {
    let mut visited_interior = HashSet::new();
    let mut paths = Vec::new();

    let node_indices: Vec<_> = graph.node_indices().collect();
    for &n in &node_indices {
        if is_interior(graph, n) && !visited_interior.contains(&n) {
            let mut chain = vec![n];
            visited_interior.insert(n);

            // Go backwards to find the start of the interior chain
            let mut curr = n;
            while let Some(prev) = graph.neighbors_directed(curr, Incoming).next() {
                if is_interior(graph, prev) && !visited_interior.contains(&prev) {
                    chain.insert(0, prev);
                    visited_interior.insert(prev);
                    curr = prev;
                } else {
                    break;
                }
            }
            // Go forwards to find the end of the interior chain
            let mut curr = n;
            while let Some(next) = graph.neighbors_directed(curr, Outgoing).next() {
                if is_interior(graph, next) && !visited_interior.contains(&next) {
                    chain.push(next);
                    visited_interior.insert(next);
                    curr = next;
                } else {
                    break;
                }
            }

            // Maximal chain of interior nodes c1...cm
            let c1 = chain[0];
            let cm = *chain.last().unwrap();
            let x = graph.neighbors_directed(c1, Incoming).next();
            let y = graph.neighbors_directed(cm, Outgoing).next();

            // A node L is a left endpoint if out_degree == 1 and its neighbor is interior.
            // A node R is a right endpoint if in_degree == 1 and its neighbor is interior.
            // We search for the maximal such L and R around our chain.
            let l = if let Some(x_node) = x {
                if graph.neighbors_directed(x_node, Outgoing).count() == 1 {
                    Some(x_node)
                } else if chain.len() >= 2 {
                    // If x_node has out_degree > 1, c1 itself is the maximal left endpoint
                    // (if it has an interior neighbor, which it does: c2).
                    Some(c1)
                } else {
                    None
                }
            } else if chain.len() >= 2 {
                Some(c1)
            } else {
                None
            };

            let r = if let Some(y_node) = y {
                if graph.neighbors_directed(y_node, Incoming).count() == 1 {
                    Some(y_node)
                } else if chain.len() >= 2 {
                    // If y_node has in_degree > 1, cm itself is the maximal right endpoint.
                    Some(cm)
                } else {
                    None
                }
            } else if chain.len() >= 2 {
                Some(cm)
            } else {
                None
            };

            if let (Some(l_node), Some(r_node)) = (l, r) {
                let mut full_chain = Vec::new();
                if let Some(x_node) = x {
                    full_chain.push(x_node);
                }
                full_chain.extend(chain);
                if let Some(y_node) = y {
                    full_chain.push(y_node);
                }

                let l_idx = full_chain.iter().position(|&n| n == l_node).unwrap();
                let r_idx = full_chain.iter().position(|&n| n == r_node).unwrap();
                let path = full_chain[l_idx..=r_idx].to_vec();

                // Paths shorter than 3 vertices (no interior) are ignored.
                if path.len() >= 3 {
                    paths.push(path);
                }
            }
        }
    }

    for path in paths {
        let n_nodes: Vec<_> = path
            .iter()
            .filter(|&&n| !graph[n].is_dummy)
            .cloned()
            .collect();
        let d_nodes: Vec<_> = path
            .iter()
            .filter(|&&n| graph[n].is_dummy)
            .cloned()
            .collect();

        // Skip if no dummies or no non-dummies to interleave.
        if n_nodes.is_empty() || d_nodes.is_empty() {
            continue;
        }

        let m = d_nodes.len();
        let start_dummies = m / 2;
        let end_dummies = m - start_dummies;

        let mut new_sequence = Vec::new();
        let mut d_iter = d_nodes.into_iter();
        for _ in 0..start_dummies {
            new_sequence.push(d_iter.next().unwrap());
        }
        new_sequence.extend(n_nodes);
        for _ in 0..end_dummies {
            new_sequence.push(d_iter.next().unwrap());
        }

        let first_node = path[0];
        let last_node = *path.last().unwrap();
        let start_rank = graph[first_node].rank;
        let path_set: HashSet<_> = path.iter().cloned().collect();

        // Collect external edges and internal edge data before modifying the graph.
        let in_edges: Vec<_> = graph
            .edges_directed(first_node, Incoming)
            .filter(|e| !path_set.contains(&e.source()))
            .map(|e| (e.source(), *e.weight()))
            .collect();
        let out_edges: Vec<_> = graph
            .edges_directed(last_node, Outgoing)
            .filter(|e| !path_set.contains(&e.target()))
            .map(|e| (e.target(), *e.weight()))
            .collect();

        let mut internal_edge_data = Vec::new();
        for i in 0..path.len() - 1 {
            let edge = graph.find_edge(path[i], path[i + 1]).unwrap();
            internal_edge_data.push(*graph.edge_weight(edge).unwrap());
        }

        // Remove old edges from the path.
        for i in 0..path.len() - 1 {
            if let Some(e) = graph.find_edge(path[i], path[i + 1]) {
                graph.remove_edge(e);
            }
        }
        for (src, _) in &in_edges {
            if let Some(e) = graph.find_edge(*src, first_node) {
                graph.remove_edge(e);
            }
        }
        for (target, _) in &out_edges {
            if let Some(e) = graph.find_edge(last_node, *target) {
                graph.remove_edge(e);
            }
        }

        // Re-assign ranks based on the new interleaved sequence.
        for (i, &node) in new_sequence.iter().enumerate() {
            graph[node].rank = start_rank + i as i32;
        }

        // Reconnect edges in the new sequence.
        let mut added_edges = Vec::new();
        for i in 0..new_sequence.len() - 1 {
            let e = graph.add_edge(new_sequence[i], new_sequence[i + 1], internal_edge_data[i]);
            added_edges.push(e);
        }
        // Reconnect external edges to the new endpoints.
        for (src, data) in in_edges {
            let e = graph.add_edge(src, new_sequence[0], data);
            added_edges.push(e);
        }
        for (target, data) in out_edges {
            let e = graph.add_edge(*new_sequence.last().unwrap(), target, data);
            added_edges.push(e);
        }

        // Update labels for all touched edges based on new non-dummy neighbors.
        for e_idx in added_edges {
            let (u, v) = graph.edge_endpoints(e_idx).unwrap();
            let s_node = find_nearest_non_dummy(graph, u, Incoming);
            let t_node = find_nearest_non_dummy(graph, v, Outgoing);

            if let (Some(s_orig), Some(t_orig)) =
                (graph[s_node].input_node_idx, graph[t_node].input_node_idx)
            {
                graph.edge_weight_mut(e_idx).unwrap().input_node_idx_pair = Some((s_orig, t_orig));
            }
        }
    }
}

fn find_nearest_non_dummy(
    graph: &StableDiGraph<Vertex, Edge>,
    start: NodeIndex,
    dir: petgraph::Direction,
) -> NodeIndex {
    let mut curr = start;
    while graph[curr].is_dummy {
        if let Some(next) = graph.neighbors_directed(curr, dir).next() {
            curr = next;
        } else {
            break;
        }
    }
    curr
}

fn is_interior(graph: &StableDiGraph<Vertex, Edge>, n: NodeIndex) -> bool {
    graph.neighbors_directed(n, Incoming).count() == 1
        && graph.neighbors_directed(n, Outgoing).count() == 1
}

pub(super) fn remove_dummy_vertices(
    graph: &mut StableDiGraph<Vertex, Edge>,
    order: &mut [Vec<NodeIndex>],
) {
    // go through all nodes in topological order
    // see if any outgoing neighbors are dummies
    // follow them until the other non dummy node is found
    // insert old edge
    // remove all dummy nodes
    info!(target: "crossing_reduction", "Removing dummy vertices and inserting original edges.");
    let vertices = toposort(&*graph, None).unwrap();
    for v in vertices {
        let mut edges = Vec::new();
        for mut n in graph.neighbors_directed(v, Outgoing) {
            if graph[n].is_dummy {
                while graph[n].is_dummy {
                    let dummy_neighbors = graph.neighbors_directed(n, Outgoing).collect::<Vec<_>>();
                    //assert_eq!(dummy_neighbors.len(), 1);
                    n = dummy_neighbors[0];
                }
                edges.push((v, n));
            }
        }
        for (tail, head) in edges {
            graph.add_edge(tail, head, Edge::default());
        }
    }
    // remove from order
    for l in order {
        l.retain(|v| !graph[*v].is_dummy);
    }
    graph.retain_nodes(|g, v| !g[v].is_dummy);
}

// TODO: Maybe write store all upper neighbors on vertex directly
pub(super) fn ordering(
    graph: &mut StableDiGraph<Vertex, Edge>,
    crossing_minimization: CrossingMinimization,
    transpose: bool,
) -> Vec<Vec<NodeIndex>> {
    let order = init_order(graph);
    // move downwards for crossing reduction
    let cm_method = match crossing_minimization {
        CrossingMinimization::Barycenter => self::barycenter,
        CrossingMinimization::Median => self::median,
    };
    let order = reduce_crossings_bilayer_sweep(graph, order, cm_method, transpose);
    order._inner
}

type CMMethod =
    fn(&StableDiGraph<Vertex, Edge>, NodeIndex, bool, &HashMap<NodeIndex, usize>) -> f64;

fn init_order(graph: &StableDiGraph<Vertex, Edge>) -> Order {
    info!(target: "crossing_reduction", 
        "Initializing order of vertices in each rank via dfs.");

    fn dfs(
        v: NodeIndex,
        order: &mut Vec<Vec<NodeIndex>>,
        graph: &StableDiGraph<Vertex, Edge>,
        visited: &mut HashSet<NodeIndex>,
    ) {
        if !visited.contains(&v) {
            visited.insert(v);
            order[graph[v].rank as usize].push(v);
            graph
                .neighbors_directed(v, Outgoing)
                .for_each(|n| dfs(n, order, graph, visited))
        }
    }

    let max_rank = graph
        .node_weights()
        .map(|v| v.rank as usize)
        .max_by(|r1, r2| r1.cmp(r2))
        .expect("Got invalid ranking");
    let mut order = vec![Vec::new(); max_rank + 1];
    let mut visited = HashSet::new();

    // build initial order via dfs
    graph
        .node_indices()
        .for_each(|v| dfs(v, &mut order, graph, &mut visited));

    Order::new(order)
}

fn reduce_crossings_bilayer_sweep(
    graph: &StableDiGraph<Vertex, Edge>,
    mut order: Order,
    cm_method: CMMethod,
    transpose: bool,
) -> Order {
    info!(target: "crossing_reduction", "Reducing crossings via bilayer sweep");
    let mut best_crossings = order.crossings(graph);
    debug!(target: "crossing_reduction", "Initial number of crossings: {best_crossings}");
    let mut last_best = 0;
    let mut best = order.clone();

    fn prefer_order(graph: &StableDiGraph<Vertex, Edge>, a: &Order, b: &Order) -> bool {
        // Lexicographic compare by (sort_bias, node_index) across layers.
        for r in 0..a.max_rank() {
            let la = &a._inner[r];
            let lb = &b._inner[r];
            for (va, vb) in la.iter().zip(lb.iter()) {
                let ka = (graph[*va].sort_bias, va.index());
                let kb = (graph[*vb].sort_bias, vb.index());
                if ka != kb {
                    return ka < kb;
                }
            }
        }
        false
    }

    for i in 0.. {
        order = order_layer(graph, i % 2 == 0, &order, cm_method);
        if transpose {
            self::transpose(graph, &mut order, i % 2 == 0);
        }
        let crossings = order.crossings(graph);
        trace!(target: "crossing_reduction", "Current number of crossings: {crossings}");

        let improves_crossings = crossings < best_crossings;
        let improves_preference = crossings == best_crossings && prefer_order(graph, &order, &best);

        if improves_crossings {
            best_crossings = crossings;
            debug!(target: "crossing_reduction", "Lowest number of crossings so far: {best_crossings}");
            best = order.clone();
            // Only reset the convergence counter on strict crossing improvements.
            last_best = 0;
        } else {
            if improves_preference {
                best = order.clone();
            }
            // Preference-only improvements do not extend the sweep loop.
            last_best += 1;
        }
        if last_best == 4 {
            info!(target: "crossing_reduction", "Didn't improve after 4 sweeps, returning");
            return best;
        }
    }
    best
}

fn transpose(graph: &StableDiGraph<Vertex, Edge>, order: &mut Order, move_down: bool) {
    trace!(target: "crossings_reduction", 
        "Using transpose, try to swap vertices in each layer manually to reduce cross count");

    let mut improved = true;
    let iter_dir = if move_down {
        IterDir::Forward
    } else {
        IterDir::Backward
    };

    while improved {
        improved = false;
        for r in iterate(iter_dir, order.max_rank()) {
            trace!(target: "reduce_crossings", "Transpose vertices in rank {r}");
            for i in 0..order._inner[r].len() - 1 {
                let v = order._inner[r][i];
                let w = order._inner[r][i + 1];
                let v_w_crossing = order.cross_count_two_vertices(v, w, graph);
                let w_v_crossing = order.cross_count_two_vertices(w, v, graph);
                if v_w_crossing > w_v_crossing {
                    improved = true;
                    order.exchange(i, i + 1, r);
                }
            }
        }
        trace!(target: "reduce_crossings", "Did improve: {improved}");
    }
}

fn order_layer(
    graph: &StableDiGraph<Vertex, Edge>,
    move_down: bool,
    cur_order: &Order,
    cm_method: CMMethod,
) -> Order {
    let mut new_order = vec![Vec::new(); cur_order.max_rank()];
    let mut positions = cur_order.positions.clone();
    let dir: Vec<usize> = if move_down {
        new_order[0].clone_from(&cur_order._inner[0]);
        (1..cur_order.max_rank()).collect()
    } else {
        new_order[cur_order.max_rank() - 1].clone_from(&cur_order._inner[cur_order.max_rank() - 1]);
        (0..cur_order.max_rank() - 1).rev().collect()
    };

    for rank in dir {
        trace!(target: "crossing_reduction", "Updating order of vertices in rank {rank}");
        trace!(target: "crossing_reduction", "Original order: {:?}",
            cur_order[rank]
                .iter()
                .map(|v| v.index())
                .collect::<Vec<_>>()
                .as_slice()
        );

        new_order[rank].clone_from(&cur_order[rank]);
        let ordering = new_order[rank]
            .iter()
            .map(|n| (*n, cm_method(graph, *n, move_down, &positions)))
            .collect::<HashMap<NodeIndex, f64>>();

        new_order[rank].sort_by(|a, b| {
            let a_val = ordering.get(a).unwrap();
            let b_val = ordering.get(b).unwrap();
            a_val
                .partial_cmp(b_val)
                .unwrap()
                // If multiple orderings are equally good (same heuristic score),
                // apply a stable, caller-provided preference.
                .then_with(|| graph[*a].sort_bias.cmp(&graph[*b].sort_bias))
                .then_with(|| a.index().cmp(&b.index()))
        });

        new_order[rank].iter().enumerate().for_each(|(pos, v)| {
            positions.insert(*v, pos);
        });
        trace!(target: "crossing_reduction", "Updated order : {:?}",
            new_order[rank]
                .iter()
                .map(|v| v.index())
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    Order::new(new_order)
}

fn barycenter(
    graph: &StableDiGraph<Vertex, Edge>,
    vertex: NodeIndex,
    move_down: bool,
    positions: &HashMap<NodeIndex, usize>,
) -> f64 {
    let neighbors: Vec<_> = if move_down {
        graph.neighbors_directed(vertex, Incoming).collect()
    } else {
        graph.neighbors_directed(vertex, Outgoing).collect()
    };

    if neighbors.is_empty() {
        return *positions.get(&vertex).unwrap() as f64;
    }

    // Only look at direct neighbors
    let adjacent = neighbors
        .into_iter()
        // .filter(|n| graph[vertex].rank.abs_diff(graph[*n].rank) == 1)
        .map(|n| *positions.get(&n).unwrap())
        .collect::<Vec<usize>>();

    adjacent.iter().sum::<usize>() as f64 / adjacent.len() as f64
}

fn median(
    graph: &StableDiGraph<Vertex, Edge>,
    vertex: NodeIndex,
    move_down: bool,
    positions: &HashMap<NodeIndex, usize>,
) -> f64 {
    let neighbors = if move_down {
        graph.neighbors_directed(vertex, Incoming)
    } else {
        graph.neighbors_directed(vertex, Outgoing)
    };
    // Only look at direct neighbors
    let mut adjacent = neighbors
        .filter(|n| graph[vertex].rank.abs_diff(graph[*n].rank) == 1)
        .map(|n| *positions.get(&n).unwrap())
        .collect::<Vec<_>>();

    adjacent.sort();

    let length_p = adjacent.len();
    let m = length_p / 2;
    if length_p == 0 {
        f64::MAX
    } else if length_p % 2 == 1 {
        adjacent[m] as f64
    } else if length_p == 2 {
        (adjacent[0] + adjacent[1]) as f64 / 2.
    } else {
        let left = adjacent[m - 1] - adjacent[0];
        let right = adjacent[length_p - 1] - adjacent[m];
        (adjacent[m - 1] * right + adjacent[m] * left) as f64 / (left + right) as f64
    }
}
