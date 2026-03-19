use std::{
    collections::{HashSet, VecDeque},
    hash::Hash,
};

use log::{info, trace};
use petgraph::{
    algo::toposort,
    visit::{
        EdgeRef, GraphBase, IntoEdgeReferences, IntoNeighborsDirected, IntoNodeIdentifiers,
        NodeCount, NodeIndexable, Visitable,
    },
};

/// Result of cycle removal: a linear ordering of nodes and the set of backward edges.
pub struct CycleRemovalResult<NodeId> {
    /// Nodes in topological-like order (sources first, sinks last).
    pub ordering: Vec<NodeId>,
    /// Edges that point backward relative to the ordering.
    /// For self-loops (u == v), both endpoints are the same.
    pub backward_edges: HashSet<(NodeId, NodeId)>,
}

/// Compute a linear ordering of nodes and identify backward edges.
///
/// For acyclic graphs, uses petgraph's `toposort` (fast path).
/// For cyclic graphs, uses the Eades–Lin–Smyth heuristic to find a
/// feedback arc set with minimal backward edges.
///
/// Optional `pin_source` / `pin_sink` force specific nodes to the
/// beginning / end of the ordering.
pub fn remove_cycles<G>(
    graph: &G,
    pin_source: Option<G::NodeId>,
    pin_sink: Option<G::NodeId>,
) -> CycleRemovalResult<G::NodeId>
where
    G: GraphBase + NodeIndexable + NodeCount + Visitable,
    for<'a> &'a G: IntoNodeIdentifiers<NodeId = G::NodeId>
        + IntoEdgeReferences<EdgeRef: EdgeRef<NodeId = G::NodeId>>
        + IntoNeighborsDirected,
    G::NodeId: Copy + Eq + Hash + Ord,
{
    // Fast path: try toposort first (works for acyclic graphs)
    if let Ok(sorted) = toposort(graph, None) {
        // Collect self-loops (toposort succeeds even with self-loops in some petgraph versions,
        // but we still need to detect them)
        let mut backward_edges = HashSet::new();
        for e in graph.edge_references() {
            if e.source() == e.target() {
                backward_edges.insert((e.source(), e.target()));
            }
        }
        if backward_edges.is_empty() {
            trace!(target: "cycle_removal", "Graph is acyclic, using toposort ordering");
            return CycleRemovalResult {
                ordering: sorted,
                backward_edges,
            };
        }
    }

    info!(
        target: "cycle_removal",
        "Graph contains cycles, computing Eades ordering (pin_source={}, pin_sink={})",
        pin_source.is_some(),
        pin_sink.is_some()
    );

    let ordering = eades_ordering(graph, pin_source, pin_sink);

    // Build rank map: node -> position in ordering
    let mut rank = vec![0usize; graph.node_bound()];
    for (i, &node) in ordering.iter().enumerate() {
        rank[graph.to_index(node)] = i;
    }

    // Identify backward edges
    let mut backward_edges = HashSet::new();
    for e in graph.edge_references() {
        let u = e.source();
        let v = e.target();
        if u == v || rank[graph.to_index(u)] > rank[graph.to_index(v)] {
            backward_edges.insert((u, v));
        }
    }

    info!(
        target: "cycle_removal",
        "Found {} backward edges",
        backward_edges.len()
    );

    CycleRemovalResult {
        ordering,
        backward_edges,
    }
}

/// Eades–Lin–Smyth heuristic ordering.
///
/// Iteratively removes sources (in-degree 0) to the front of the ordering,
/// sinks (out-degree 0) to the back, and breaks ties by picking the node
/// with maximum (out_degree - in_degree).
fn eades_ordering<G>(
    graph: &G,
    pinned_source: Option<G::NodeId>,
    pinned_sink: Option<G::NodeId>,
) -> Vec<G::NodeId>
where
    G: GraphBase + NodeIndexable + NodeCount,
    for<'a> &'a G: IntoNodeIdentifiers<NodeId = G::NodeId>
        + IntoEdgeReferences<EdgeRef: EdgeRef<NodeId = G::NodeId>>,
    G::NodeId: Copy + Eq + Hash + Ord,
{
    let bound = graph.node_bound();
    let mut active = vec![false; bound];
    let mut active_count = 0usize;
    let mut outs: Vec<Vec<usize>> = vec![Vec::new(); bound];
    let mut ins: Vec<Vec<usize>> = vec![Vec::new(); bound];
    let mut in_deg = vec![0i32; bound];
    let mut out_deg = vec![0i32; bound];

    // Map index back to NodeId
    let mut index_to_node: Vec<Option<G::NodeId>> = vec![None; bound];

    for n in graph.node_identifiers() {
        let ni = graph.to_index(n);
        active[ni] = true;
        active_count += 1;
        index_to_node[ni] = Some(n);
    }

    for e in graph.edge_references() {
        let u = graph.to_index(e.source());
        let v = graph.to_index(e.target());
        // Skip self-loops for degree computation
        if u == v {
            continue;
        }
        outs[u].push(v);
        ins[v].push(u);
        out_deg[u] += 1;
        in_deg[v] += 1;
    }

    let mut sources: VecDeque<usize> = VecDeque::new();
    let mut sinks: VecDeque<usize> = VecDeque::new();
    let mut prefix: Vec<usize> = Vec::with_capacity(active_count);
    let mut suffix: VecDeque<usize> = VecDeque::with_capacity(active_count);

    let is_active = |i: usize, active: &[bool]| i < active.len() && active[i];

    // Helper: remove a node and update neighbors
    let remove_node = |vi: usize,
                       active: &mut Vec<bool>,
                       active_count: &mut usize,
                       in_deg: &mut Vec<i32>,
                       out_deg: &mut Vec<i32>,
                       outs: &Vec<Vec<usize>>,
                       ins: &Vec<Vec<usize>>,
                       sources: &mut VecDeque<usize>,
                       sinks: &mut VecDeque<usize>| {
        if !active[vi] {
            return;
        }
        active[vi] = false;
        *active_count -= 1;

        for &w in &outs[vi] {
            if active[w] {
                in_deg[w] -= 1;
                if in_deg[w] == 0 {
                    sources.push_back(w);
                }
            }
        }
        for &u in &ins[vi] {
            if active[u] {
                out_deg[u] -= 1;
                if out_deg[u] == 0 {
                    sinks.push_back(u);
                }
            }
        }
    };

    // Pin source
    if let Some(s) = pinned_source {
        let si = graph.to_index(s);
        if is_active(si, &active) {
            prefix.push(si);
            remove_node(
                si,
                &mut active,
                &mut active_count,
                &mut in_deg,
                &mut out_deg,
                &outs,
                &ins,
                &mut sources,
                &mut sinks,
            );
        }
    }

    // Pin sink
    if let Some(t) = pinned_sink {
        let ti = graph.to_index(t);
        if is_active(ti, &active) && Some(t) != pinned_source {
            suffix.push_back(ti);
            remove_node(
                ti,
                &mut active,
                &mut active_count,
                &mut in_deg,
                &mut out_deg,
                &outs,
                &ins,
                &mut sources,
                &mut sinks,
            );
        }
    }

    // Seed sources and sinks
    for i in 0..bound {
        if active[i] {
            if in_deg[i] == 0 {
                sources.push_back(i);
            }
            if out_deg[i] == 0 {
                sinks.push_back(i);
            }
        }
    }

    // Main loop
    while active_count > 0 {
        let mut progressed = false;

        while let Some(v) = sources.pop_front() {
            if is_active(v, &active) {
                progressed = true;
                prefix.push(v);
                remove_node(
                    v,
                    &mut active,
                    &mut active_count,
                    &mut in_deg,
                    &mut out_deg,
                    &outs,
                    &ins,
                    &mut sources,
                    &mut sinks,
                );
            }
        }

        while let Some(v) = sinks.pop_front() {
            if is_active(v, &active) {
                progressed = true;
                suffix.push_front(v);
                remove_node(
                    v,
                    &mut active,
                    &mut active_count,
                    &mut in_deg,
                    &mut out_deg,
                    &outs,
                    &ins,
                    &mut sources,
                    &mut sinks,
                );
            }
        }

        if progressed {
            continue;
        }

        // Pick node with max (out - in), tie-break by lower index
        let mut best: Option<usize> = None;
        let mut best_score = i32::MIN;
        for i in 0..bound {
            if active[i] {
                let score = out_deg[i] - in_deg[i];
                if score > best_score || (score == best_score && best.is_none_or(|b| i < b)) {
                    best_score = score;
                    best = Some(i);
                }
            }
        }

        if let Some(b) = best {
            prefix.push(b);
            remove_node(
                b,
                &mut active,
                &mut active_count,
                &mut in_deg,
                &mut out_deg,
                &outs,
                &ins,
                &mut sources,
                &mut sinks,
            );
        } else {
            break;
        }
    }

    // Construct final ordering
    let mut order = Vec::with_capacity(prefix.len() + suffix.len());
    order.extend(prefix);
    order.extend(suffix);

    order.into_iter().filter_map(|i| index_to_node[i]).collect()
}

#[cfg(test)]
mod tests {
    use petgraph::stable_graph::StableDiGraph;

    use super::*;

    #[test]
    fn test_acyclic_graph_no_backward_edges() {
        let mut g = StableDiGraph::<&str, ()>::new();
        let node_a = g.add_node("a");
        let node_b = g.add_node("b");
        let node_c = g.add_node("c");
        g.add_edge(node_a, node_b, ());
        g.add_edge(node_b, node_c, ());

        let result = remove_cycles(&g, None, None);
        assert!(result.backward_edges.is_empty());
        assert_eq!(result.ordering.len(), 3);
    }

    #[test]
    fn test_simple_cycle() {
        let mut g = StableDiGraph::<&str, ()>::new();
        let node_a = g.add_node("a");
        let node_b = g.add_node("b");
        let node_c = g.add_node("c");
        g.add_edge(node_a, node_b, ());
        g.add_edge(node_b, node_c, ());
        g.add_edge(node_c, node_a, ());

        let result = remove_cycles(&g, None, None);
        assert_eq!(result.backward_edges.len(), 1);
        assert_eq!(result.ordering.len(), 3);
    }

    #[test]
    fn test_self_loop() {
        let mut g = StableDiGraph::<&str, ()>::new();
        let node_a = g.add_node("a");
        g.add_edge(node_a, node_a, ());

        let result = remove_cycles(&g, None, None);
        assert_eq!(result.backward_edges.len(), 1);
        assert!(result.backward_edges.contains(&(node_a, node_a)));
    }

    #[test]
    fn test_pinned_source_determines_backward_edge() {
        let mut g = StableDiGraph::<&str, ()>::new();
        let node_a = g.add_node("a");
        let node_b = g.add_node("b");
        let node_c = g.add_node("c");
        g.add_edge(node_a, node_b, ());
        g.add_edge(node_b, node_c, ());
        g.add_edge(node_c, node_a, ());

        // Pin node_a as source => ordering [node_a, node_b, node_c], back-edge is node_c->node_a
        let result = remove_cycles(&g, Some(node_a), None);
        assert_eq!(result.backward_edges.len(), 1);
        assert!(result.backward_edges.contains(&(node_c, node_a)));
    }

    #[test]
    fn test_pinned_source_and_sink() {
        let mut g = StableDiGraph::<&str, ()>::new();
        let node_a = g.add_node("a");
        let node_b = g.add_node("b");
        let node_c = g.add_node("c");
        g.add_edge(node_a, node_b, ());
        g.add_edge(node_b, node_c, ());
        g.add_edge(node_c, node_a, ());

        // Pin node_a as source, node_c as sink => ordering [node_a, node_b, node_c], back-edge node_c->node_a
        let result = remove_cycles(&g, Some(node_a), Some(node_c));
        assert_eq!(result.backward_edges.len(), 1);
        assert!(result.backward_edges.contains(&(node_c, node_a)));
    }
}
