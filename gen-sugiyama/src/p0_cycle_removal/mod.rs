use log::{debug, info};
use petgraph::{
    algo::{greedy_feedback_arc_set, is_cyclic_directed},
    stable_graph::{EdgeIndex, StableDiGraph},
    visit::EdgeRef,
};

use super::{Edge, Vertex};

#[allow(dead_code)]
/// Removes all the edges that contribute to cycles in the graph
/// Does so by finding a greedy feedback arc set and then reversing the
/// direction of the edges from that set.
/// Is not guaranteed to find the minimum fas.
pub(crate) fn remove_cycles(graph: &mut StableDiGraph<Vertex, Edge>) -> Vec<EdgeIndex> {
    if !is_cyclic_directed(&*graph) {
        info!(target: "Cycle Removal", "Graph contains no cycle");
        return Vec::new();
    }

    info!(target: "Cycle Removal", "Graph contains cycle, reversing edges");

    // get the feedback arc set
    let fas: Vec<EdgeIndex> = greedy_feedback_arc_set(&*graph).map(|e| e.id()).collect();
    let mut reversed_edges = Vec::new();

    // reverse the direction of the edges
    for edge in fas {
        if let Some((tail, head)) = graph.edge_endpoints(edge) {
            // get the weight
            let weight = graph[edge];
            // add new edge in reversed direction
            let reversed_edge = graph.add_edge(head, tail, weight);
            reversed_edges.push(reversed_edge);
            // remove the old edge
            graph.remove_edge(edge);
        }
    }

    assert!(!is_cyclic_directed(&*graph));

    debug!(target: "Cycle Removal", "Reversed {} edges", reversed_edges.len());

    reversed_edges
}

#[cfg(test)]
mod tests {
    use petgraph::{algo::is_cyclic_directed, stable_graph::StableDiGraph};

    use super::remove_cycles;
    use crate::{Edge, Vertex};

    #[test]
    fn test_graph_simple_no_cycles() {
        let mut graph = StableDiGraph::<Vertex, Edge>::from_edges([(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert!(!is_cyclic_directed(&graph));
        assert!(remove_cycles(&mut graph).is_empty());
    }

    #[test]
    fn test_graph_simple_contains_cycle() {
        let mut graph = StableDiGraph::<Vertex, Edge>::from_edges([
            (2, 1),
            (1, 4),
            (4, 5),
            (2, 3),
            (3, 6),
            (6, 5),
            (5, 2),
        ]);

        assert!(is_cyclic_directed(&graph));
        let _ = remove_cycles(&mut graph);
        assert!(!is_cyclic_directed(&graph));
    }

    #[test]
    fn test_graph_complex_contains_cycle() {
        let mut graph = StableDiGraph::<Vertex, Edge>::from_edges([
            (1, 2),
            (2, 5),
            (2, 6),
            (2, 3),
            (3, 4),
            (4, 3),
            (4, 8),
            (8, 4),
            (8, 7),
            (3, 7),
            (6, 7),
            (7, 6),
            (5, 6),
            (5, 1),
        ]);

        assert!(is_cyclic_directed(&graph));
        let edges = remove_cycles(&mut graph);
        println!(
            "test_graph_complex_contains_cycle: Reversed {} edges",
            edges.len()
        );
        assert!(!is_cyclic_directed(&graph));
    }
}
