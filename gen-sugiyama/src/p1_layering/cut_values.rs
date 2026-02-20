use std::collections::VecDeque;

use log::{debug, info, trace};
use petgraph::{
    Direction::{self, Incoming, Outgoing},
    stable_graph::{EdgeIndex, NodeIndex, StableDiGraph},
    visit::EdgeRef,
};

use super::{Edge, Vertex};

#[derive(Debug)]
struct NeighborhoodInfo {
    cut_value_sum: i32,
    tree_edge_weight_sum: i32,
    non_tree_edge_weight_sum: i32,
    missing: Option<NodeIndex>,
}

pub(super) fn init_cutvalues(graph: &mut StableDiGraph<Vertex, Edge>) {
    // TODO: check if it is faster to collect tree edges or to do unecessary iterations
    info!(target: "cut_values", "Initializing cut values");
    let queue = leaves(graph);
    debug!(target: "cut_values", "Leaves of tree: {:?}", queue);
    // traverse tree inward via breadth first starting from leaves
    calculate_cut_values(graph, queue);
}

pub(super) fn update_cutvalues(
    graph: &mut StableDiGraph<Vertex, Edge>,
    removed_edge: EdgeIndex,
    swap_edge: EdgeIndex,
) -> NodeIndex {
    info!(target: "cut_values", "Updating outdated cut values");
    let least_common_ancestor = remove_outdated_cut_values(graph, swap_edge, removed_edge);
    let queue = VecDeque::from([graph.edge_endpoints(removed_edge).unwrap().0]);
    debug!(target: "cut_values", "Leaves of tree: {queue:?}");
    calculate_cut_values(graph, queue);
    least_common_ancestor
}

fn leaves(graph: &StableDiGraph<Vertex, Edge>) -> VecDeque<NodeIndex> {
    graph
        .node_indices()
        .filter(|v| {
            1 == graph
                .edges_directed(*v, Incoming)
                .chain(graph.edges_directed(*v, Outgoing))
                .filter(|e| e.weight().is_tree_edge)
                .count()
        })
        .collect::<VecDeque<_>>()
}

fn calculate_cut_values(graph: &mut StableDiGraph<Vertex, Edge>, mut queue: VecDeque<NodeIndex>) {
    info!(target: "cut_values", "Calculating cut values of tree edges");
    while let Some(vertex) = queue.pop_front() {
        debug!(target: "cut_values", "Calculating cut values for tree edges incident to: {}", vertex.index());
        let incoming = get_neighborhood_info(graph, vertex, Incoming);
        let outgoing = get_neighborhood_info(graph, vertex, Outgoing);

        // if we can't calculate cut value yet, or the value is already known
        let (mut incoming, mut outgoing) = match (incoming, outgoing) {
            (Some(inc), Some(out)) => (inc, out),
            _ => continue,
        };

        let missing = match (incoming.missing, outgoing.missing) {
            (Some(u), None) => u,
            (None, Some(v)) => v,
            _ => continue,
        };

        let edge = match graph.find_edge(vertex, missing) {
            Some(e) => {
                // switch direction, if vertex is tail component of edge
                std::mem::swap(&mut incoming, &mut outgoing);
                e
            }
            None => graph.find_edge(missing, vertex).unwrap(),
        };

        graph[edge].cut_value = Some(calculate_cut_value(graph[edge].weight, incoming, outgoing));
        trace!(target: "cut_values", "Cut values for edge: {}, {:?}", edge.index(), graph[edge].cut_value);
        // continue traversing tree in direction of edge whose vertex was missing before
        queue.push_back(missing);
    }
}

fn calculate_cut_value(
    edge_weight: i32,
    incoming: NeighborhoodInfo,
    outgoing: NeighborhoodInfo,
) -> i32 {
    trace!(target: "cut_values", "Calculating cut value: edge_weight: {edge_weight}, data of incoming edges: {incoming:?}, data of outgoing edges: {outgoing:?}");
    edge_weight + incoming.non_tree_edge_weight_sum - incoming.cut_value_sum
        + incoming.tree_edge_weight_sum
        - outgoing.non_tree_edge_weight_sum
        + outgoing.cut_value_sum
        - outgoing.tree_edge_weight_sum
}

fn get_neighborhood_info(
    graph: &StableDiGraph<Vertex, Edge>,
    vertex: NodeIndex,
    direction: Direction,
) -> Option<NeighborhoodInfo> {
    // return the sum of all cut values,
    // sum of weights of cut value edges
    // missing cut value (only if there is one)
    // sum of weights of edges who are not tree edges
    let mut cut_value_sum = 0;
    let mut tree_edge_weight_sum = 0;
    let mut non_tree_edge_weight_sum = 0;
    let mut missing = None;

    for edge in graph.edges_directed(vertex, direction) {
        let (tail, head) = (edge.source(), edge.target());
        let edge = *edge.weight();
        if !edge.is_tree_edge {
            non_tree_edge_weight_sum += edge.weight;
        } else if let Some(cut_value) = edge.cut_value {
            cut_value_sum += cut_value;
            tree_edge_weight_sum += edge.weight;
        } else if missing.is_none() {
            missing = Some(if tail == vertex { head } else { tail });
        } else {
            return None;
        }
    }
    Some(NeighborhoodInfo {
        cut_value_sum,
        tree_edge_weight_sum,
        non_tree_edge_weight_sum,
        missing,
    })
}

fn remove_outdated_cut_values(
    graph: &mut StableDiGraph<Vertex, Edge>,
    swap_edge: EdgeIndex,
    removed_edge: EdgeIndex,
) -> NodeIndex {
    info!(target: "cut_values", "Remove outtdated cut_values, in order to calculate new ones");
    graph[removed_edge].cut_value = None;
    let (mut w, mut x) = graph.edge_endpoints(swap_edge).unwrap();
    if graph[w].lim > graph[x].lim {
        std::mem::swap(&mut w, &mut x)
    }
    debug!(target: "cut_values", 
        "looking for path connecting endpoints of new edge ({}, {}), removing cut values on the way", 
        w.index(), 
        x.index());

    // follow path back until least common ancestor is found
    // and remove cut_values on the way
    let least_common_ancestor = match graph[w].parent {
        None => w,
        Some(mut parent) => {
            let mut l = w;
            loop {
                let edge = graph.find_edge_undirected(l, parent).unwrap().0;
                graph[edge].cut_value = None;
                l = parent;
                trace!(target: "cut_values", "current node in path: {}", l.index());
                if graph[l].low <= graph[w].lim && graph[x].lim <= graph[l].lim
                    || graph[l].parent.is_none()
                {
                    break l;
                }
                parent = graph[l].parent.unwrap();
            }
        }
    };
    // record path from x to l
    // we don't need to care about the order in which the edges are added,
    // since we only need them to remove the outdated cutvalues.
    let mut l = x;
    while l != least_common_ancestor {
        let parent = graph[l].parent.unwrap();
        let edge = graph.find_edge_undirected(l, parent).unwrap().0;
        graph[edge].cut_value = None;
        l = parent;
    }

    debug!(target: "cut_values", 
        "found least common ancestor in path connecting {} {}: {}", 
        w.index(), 
        x.index(), 
        least_common_ancestor.index());

    least_common_ancestor
}

#[cfg(test)]
mod tests {

    use petgraph::stable_graph::NodeIndex;

    use crate::p1_layering::{
        Edge,
        cut_values::{init_cutvalues, remove_outdated_cut_values, update_cutvalues},
        low_lim::init_low_lim,
        tests::{
            CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE, EXAMPLE_GRAPH,
            EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE,
            GraphBuilder, LOW_LIM_GRAPH, LOW_LIM_GRAPH_LOW_LIM_VALUES,
        },
    };

    #[test]
    fn test_cut_values_one_negative() {
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE;
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&edges)
            .build();

        init_cutvalues(&mut graph);

        assert_eq!(
            graph[graph
                .find_edge(edges[0].0.into(), edges[0].1.into())
                .unwrap()]
            .cut_value,
            Some(3)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[1].0.into(), edges[1].1.into())
                .unwrap()]
            .cut_value,
            Some(3)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[2].0.into(), edges[2].1.into())
                .unwrap()]
            .cut_value,
            Some(3)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[3].0.into(), edges[3].1.into())
                .unwrap()]
            .cut_value,
            Some(3)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[4].0.into(), edges[4].1.into())
                .unwrap()]
            .cut_value,
            Some(0)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[5].0.into(), edges[5].1.into())
                .unwrap()]
            .cut_value,
            Some(0)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[6].0.into(), edges[6].1.into())
                .unwrap()]
            .cut_value,
            Some(-1)
        );
    }

    #[test]
    fn test_cut_values_all_positive() {
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE;
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&edges)
            .build();

        init_cutvalues(&mut graph);

        assert_eq!(
            graph[graph
                .find_edge(edges[0].0.into(), edges[0].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[1].0.into(), edges[1].1.into())
                .unwrap()]
            .cut_value,
            Some(1)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[2].0.into(), edges[2].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[3].0.into(), edges[3].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[4].0.into(), edges[4].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[5].0.into(), edges[5].1.into())
                .unwrap()]
            .cut_value,
            Some(1)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[6].0.into(), edges[6].1.into())
                .unwrap()]
            .cut_value,
            Some(0)
        );
    }

    #[test]
    fn remove_outdated_cut_values_low_lim_graph() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
            .with_tree_edges(&LOW_LIM_GRAPH)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .build();

        init_cutvalues(&mut graph);

        let tail = 6.into();
        let head = 8.into();
        let edge = graph.add_edge(tail, head, Edge::default());
        let actual_lca = remove_outdated_cut_values(&mut graph, edge, edge);
        let expected_path = [(5, 6), (4, 5), (4, 8)]
            .into_iter()
            .map(|(t, h)| graph.find_edge(t.into(), h.into()).unwrap())
            .collect::<Vec<_>>();
        for edge in graph.edge_indices() {
            if expected_path.contains(&edge) {
                assert!(graph[edge].cut_value.is_none());
            } else if graph[edge].is_tree_edge {
                assert!(graph[edge].cut_value.is_some())
            }
        }
        let expected_lca = NodeIndex::from(4_u32);
        assert_eq!(actual_lca, expected_lca);
    }

    #[test]
    fn remove_outdated_cut_values_low_lim_graph_lca_is_root() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
            .with_tree_edges(&LOW_LIM_GRAPH)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .build();

        init_cutvalues(&mut graph);

        let tail = 3.into();
        let head = 8.into();
        let edge = graph.add_edge(tail, head, Edge::default());
        let actual_lca = remove_outdated_cut_values(&mut graph, edge, edge);
        let expected_path = [(1, 3), (0, 1), (4, 8), (0, 4)]
            .into_iter()
            .map(|(t, h)| graph.find_edge(t.into(), h.into()).unwrap())
            .collect::<Vec<_>>();
        for edge in graph.edge_indices() {
            if expected_path.contains(&edge) {
                assert!(graph[edge].cut_value.is_none());
            } else if graph[edge].is_tree_edge {
                assert!(graph[edge].cut_value.is_some())
            }
        }
        let expected_lca = NodeIndex::from(0_u32);
        assert_eq!(actual_lca, expected_lca);
    }
    #[test]
    fn update_cutvalues_updated_correctly() {
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
            .with_cut_values(&CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .build();

        init_low_lim(&mut graph);

        let swap_edge = graph.find_edge(0.into(), 4.into()).unwrap();
        let removed_edge = graph.find_edge(6.into(), 7.into()).unwrap();
        graph[swap_edge].is_tree_edge = true;
        graph[removed_edge].is_tree_edge = false;
        update_cutvalues(&mut graph, removed_edge, swap_edge);
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE;

        assert_eq!(
            graph[graph
                .find_edge(edges[0].0.into(), edges[0].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[1].0.into(), edges[1].1.into())
                .unwrap()]
            .cut_value,
            Some(1)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[2].0.into(), edges[2].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[3].0.into(), edges[3].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[4].0.into(), edges[4].1.into())
                .unwrap()]
            .cut_value,
            Some(2)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[5].0.into(), edges[5].1.into())
                .unwrap()]
            .cut_value,
            Some(1)
        );
        assert_eq!(
            graph[graph
                .find_edge(edges[6].0.into(), edges[6].1.into())
                .unwrap()]
            .cut_value,
            Some(0)
        );
    }

    #[test]
    fn update_cutvalues_only_tree_edges_have_cut_values() {
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
            .with_cut_values(&CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .build();

        init_low_lim(&mut graph);

        let swap_edge = graph.find_edge(0.into(), 4.into()).unwrap();
        let removed_edge = graph.find_edge(6.into(), 7.into()).unwrap();
        graph[swap_edge].is_tree_edge = true;
        graph[removed_edge].is_tree_edge = false;
        update_cutvalues(&mut graph, removed_edge, swap_edge);

        for e in graph.edge_weights() {
            if e.is_tree_edge {
                assert!(e.cut_value.is_some());
            } else {
                assert!(e.cut_value.is_none());
            }
        }
    }
}
