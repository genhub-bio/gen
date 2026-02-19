// TODOS: Keep non graph edges during rank() procedure in vecdeque to be able to cyclically search through them
//! Executes the second phase of sugiyamas algorithm, which assigns each vertex
//! a rank.
//! Currently three ranking algorithm are implmented:
//!
//! 1. Original - tries to move each vertex as close to neighbors as possible.
//! 2. MinimizeEdgeLength - builds a feasible tight tree in order to minimize
//!    edge lengths. This is the technique describe in the paper by Gansner et al.
//! 3. Up - Move vertices as far up as possible
//! 4. Down - Move vertices as far down as possible.
//!
mod cut_values;
mod low_lim;
pub(super) mod ranking;
#[cfg(test)]
pub(crate) mod tests;

use log::info;
use petgraph::{
    stable_graph::{EdgeIndex, StableDiGraph},
    visit::IntoNodeIdentifiers,
};

use self::{
    cut_values::update_cutvalues,
    low_lim::update_low_lim,
    ranking::{feasible_tree, init_rank, move_vertices_down, move_vertices_up, update_ranks},
};
use super::{Edge, Vertex, config::RankingType, slack};

pub(super) fn rank(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    ranking_type: RankingType,
) {
    info!(target: "ranking", "Start ranking, ranking type: {ranking_type:?}, minimum_length: {minimum_length}");
    init_rank(graph, minimum_length);
    match ranking_type {
        RankingType::Original => original(graph, minimum_length),
        RankingType::MinimizeEdgeLength => minimize_edge_length(graph, minimum_length),
        RankingType::Up => move_vertices_up(graph, minimum_length),
        RankingType::Down => move_vertices_down(graph, minimum_length),
    }
}

fn minimize_edge_length(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    feasible_tree(graph, minimum_length);
    while let Some(removed_edge) = leave_edge(graph) {
        // swap edges and calculate cut value
        let swap_edge = enter_edge(graph, removed_edge, minimum_length);
        exchange(graph, removed_edge, swap_edge, minimum_length);
    }

    // don't balance ranks since we want maximum width to
    // give indication about number of parallel processes running
    normalize(graph);
}

fn original(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    move_vertices_up(graph, minimum_length);
    move_vertices_down(graph, minimum_length);
}

fn leave_edge(graph: &StableDiGraph<Vertex, Edge>) -> Option<EdgeIndex> {
    for edge in graph.edge_indices() {
        if let Some(cut_value) = graph[edge].cut_value
            && cut_value < 0
        {
            return Some(edge);
        }
    }
    None
}

fn enter_edge(
    graph: &mut StableDiGraph<Vertex, Edge>,
    edge: EdgeIndex,
    minimum_length: i32,
) -> EdgeIndex {
    // find a non-tree edge to replace e.
    // remove e from tree
    // consider all edges going from head to tail component.
    // choose edge with minimum slack.
    let (mut u, mut v) = graph
        .edge_endpoints(edge)
        .map(|(t, h)| (graph[t], graph[h]))
        .unwrap();
    let is_root_in_head = u.lim < v.lim;
    if !is_root_in_head {
        std::mem::swap(&mut u, &mut v);
    }

    graph
        .edge_indices()
        .filter(|e| !graph[*e].is_tree_edge && is_head_to_tail(graph, *e, u, is_root_in_head))
        .min_by(|e1, e2| slack(graph, *e1, minimum_length).cmp(&slack(graph, *e2, minimum_length)))
        .unwrap()
}

fn exchange(
    graph: &mut StableDiGraph<Vertex, Edge>,
    removed_edge: EdgeIndex,
    swap_edge: EdgeIndex,
    minimum_length: i32,
) {
    // swap edges
    graph[removed_edge].is_tree_edge = false;
    graph[swap_edge].is_tree_edge = true;

    // update the graph
    let least_common_ancestor = update_cutvalues(graph, removed_edge, swap_edge);
    update_low_lim(graph, least_common_ancestor);
    update_ranks(graph, minimum_length);
}

fn normalize(graph: &mut StableDiGraph<Vertex, Edge>) {
    let min_rank = graph
        .node_identifiers()
        .map(|v| graph[v].rank)
        .min()
        .unwrap();
    for v in graph.node_weights_mut() {
        v.rank -= min_rank;
    }
}

fn is_head_to_tail(
    graph: &StableDiGraph<Vertex, Edge>,
    edge: EdgeIndex,
    u: Vertex,
    is_root_in_head: bool,
) -> bool {
    // edge needs to go from head to tail. e.g. tail neads to be in head component, and head in tail component
    let (tail, head) = graph
        .edge_endpoints(edge)
        .map(|(t, h)| (graph[t], graph[h]))
        .unwrap();
    // check if head is in tail component
    is_root_in_head == (u.low <= head.lim && head.lim <= u.lim) &&
    // check if tail is in head component
    is_root_in_head != (u.low <= tail.lim && tail.lim <= u.lim)
}
