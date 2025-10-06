use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

use super::{Edge, Vertex};
use crate::standalone_sugiyama::p1_layering::{
    cut_values::init_cutvalues, enter_edge, is_head_to_tail, leave_edge, low_lim::init_low_lim,
};

pub(crate) const EXAMPLE_GRAPH: [(u32, u32); 9] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (4, 6),
    (5, 6),
    (6, 7),
    (0, 4),
    (0, 5),
];
pub(crate) const EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING: [(u32, u32); 7] =
    [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
pub(crate) const EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE: [(u32, u32); 7] =
    [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
pub(crate) const EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE: [(u32, u32); 7] =
    [(0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6)];
pub(crate) const LOW_LIM_GRAPH: [(u32, u32); 8] = [
    (0, 1),
    (1, 2),
    (1, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (4, 7),
    (4, 8),
];
pub(crate) const LOW_LIM_GRAPH_AFTER_UPDATE: [(u32, u32); 8] = [
    (0, 1),
    (1, 2),
    (1, 3),
    (0, 4),
    (5, 6),
    (4, 7),
    (4, 8),
    (6, 7),
];
pub(crate) const LOW_LIM_GRAPH_LOW_LIM_VALUES: [(u32, u32, u32, Option<u32>); 9] = [
    (0, 1, 9, None),
    (1, 1, 3, Some(0)),
    (2, 1, 1, Some(1)),
    (3, 2, 2, Some(1)),
    (4, 4, 8, Some(0)),
    (5, 4, 5, Some(4)),
    (6, 4, 4, Some(5)),
    (7, 6, 6, Some(4)),
    (8, 7, 7, Some(4)),
];
pub(crate) const EXAMPLE_GRAPH_LOW_LIM_VALUES_NEG_CUT_VALUE: [(u32, u32, u32, Option<u32>); 8] = [
    (0, 1, 8, None),
    (1, 1, 7, Some(0)),
    (2, 1, 6, Some(1)),
    (3, 1, 5, Some(2)),
    (4, 1, 1, Some(6)),
    (5, 2, 2, Some(6)),
    (6, 1, 3, Some(7)),
    (7, 1, 4, Some(3)),
];

pub(crate) const CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE: [(u32, u32, i32); 7] = [
    (0, 1, 3),
    (1, 2, 3),
    (2, 3, 3),
    (3, 7, 3),
    (4, 6, 0),
    (5, 6, 0),
    (6, 7, -1),
];

pub(super) struct GraphBuilder {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    connecting_path: Vec<EdgeIndex>,
    removed_edge: EdgeIndex,
    least_common_ancestor: NodeIndex,
}

impl GraphBuilder {
    pub(super) fn new(edges: &[(u32, u32)]) -> Self {
        let graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
        Self {
            graph,
            minimum_length: 1,
            connecting_path: Vec::new(),
            removed_edge: EdgeIndex::from(0),
            least_common_ancestor: NodeIndex::from(0),
        }
    }

    #[allow(dead_code)]
    pub(super) fn minimum_length(&mut self) -> &mut i32 {
        &mut self.minimum_length
    }

    pub(super) fn build(
        self,
    ) -> (
        StableDiGraph<Vertex, Edge>,
        i32,
        Vec<EdgeIndex>,
        EdgeIndex,
        NodeIndex,
    ) {
        let Self {
            graph,
            minimum_length,
            connecting_path,
            removed_edge,
            least_common_ancestor,
        } = self;
        (
            graph,
            minimum_length,
            connecting_path,
            removed_edge,
            least_common_ancestor,
        )
    }

    // Blanket implementations
    pub(super) fn with_minimum_length(mut self, minimum_length: u32) -> Self
    where
        Self: Sized,
    {
        self.minimum_length = minimum_length as i32;
        self
    }

    pub(super) fn with_cut_values(mut self, cut_values: &[(u32, u32, i32)]) -> Self
    where
        Self: Sized,
    {
        for (tail, head, cut_value) in cut_values {
            // ignore any edges that do not exist
            if let Some(edge) = self.graph.find_edge((*tail).into(), (*head).into()) {
                self.graph[edge].cut_value = Some(*cut_value);
            }
        }

        self
    }

    pub(super) fn with_tree_edges(mut self, tree_edges: &[(u32, u32)]) -> Self
    where
        Self: Sized,
    {
        for (tail, head) in tree_edges {
            self.graph[NodeIndex::from(*tail)].is_tree_vertex = true;
            self.graph[NodeIndex::from(*head)].is_tree_vertex = true;
            let edge = self
                .graph
                .find_edge((*tail).into(), (*head).into())
                .expect("Edge not found in tree.");
            self.graph[edge].is_tree_edge = true;
        }
        self
    }

    /// Add low_lim_values to the graph.
    ///
    /// Input is given by a slice of (NodeIndex, low, lim, parent).
    ///
    /// Panics if vertex is not contained in tree.
    pub(super) fn with_low_lim_values(mut self, low_lim: &[(u32, u32, u32, Option<u32>)]) -> Self
    where
        Self: Sized,
    {
        for (id, low, lim, parent) in low_lim {
            let weight = self.graph.node_weight_mut((*id).into()).unwrap();
            weight.low = *low;
            weight.lim = *lim;
            weight.parent = parent.map(|p| p.into());
        }

        self
    }

    #[allow(dead_code)]
    pub(super) fn with_connecting_path(mut self, connecting_path: &[(u32, u32)]) -> Self {
        self.connecting_path = connecting_path
            .iter()
            .map(|(tail, head)| {
                self.graph
                    .find_edge_undirected((*tail).into(), (*head).into())
                    .unwrap()
                    .0
            })
            .collect::<Vec<_>>();
        self
    }

    /// Set the edge that was removed during rank procedure.
    /// Panics if edge is not contained in graph.
    #[allow(dead_code)]
    pub(super) fn with_removed_edge(mut self, tail: u32, head: u32) -> Self {
        self.removed_edge = self.graph.find_edge(tail.into(), head.into()).unwrap();
        self.graph[self.removed_edge].is_tree_edge = false;
        self
    }

    /// Set the ranks, of the vertices.
    /// Used to check if outtdated ranks are updated
    /// correctly after update_ranks is run.
    ///
    /// Input is  [(NodeIndex, rank)].
    ///
    /// Panics if node is not part of graph.
    pub(super) fn with_ranks(mut self, ranks: &[(u32, i32)]) -> Self {
        for (n, r) in ranks {
            self.graph[NodeIndex::from(*n)].rank = *r;
        }
        self
    }

    /// Set the least common ancestor, which is the root
    /// of the path connecting the edge that will be added
    /// to the graph.
    ///
    /// Panics if vertex is not part of graph.
    pub(super) fn with_least_common_ancestor(mut self, lca: u32) -> Self {
        assert!(self.graph.contains_node(NodeIndex::from(lca)));
        self.least_common_ancestor = lca.into();
        self
    }
}

#[test]
fn leave_edge_has_negative_cut_value() {
    let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
        .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
        .build();

    init_cutvalues(&mut graph);
    init_low_lim(&mut graph);

    let leave_edge = leave_edge(&graph);
    assert!(leave_edge.is_some());
    let (tail, head) = graph.edge_endpoints(leave_edge.unwrap()).unwrap();
    assert_eq!(tail, NodeIndex::from(6));
    assert_eq!(head, NodeIndex::from(7));
}

#[test]
fn leave_edge_has_no_negative_cut_value() {
    let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
        .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
        .build();

    init_cutvalues(&mut graph);
    init_low_lim(&mut graph);

    let leave_edge = leave_edge(&graph);
    assert!(leave_edge.is_none());
}

#[test]
fn test_is_head_to_tail_true_root_in_tail() {
    // u is always considered to be the tail of the edge to be swapped
    let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
        .with_tree_edges(&LOW_LIM_GRAPH)
        .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
        .build();
    let u = 5;
    let u = *graph.node_weight(u.into()).unwrap();
    let tail = 6;
    let head = 7;
    let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
    assert!(is_head_to_tail(&graph, edge, u, false));
}

#[test]
fn test_is_head_to_tail_true_root_in_head() {
    let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
        .with_tree_edges(&LOW_LIM_GRAPH)
        .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
        .build();
    let u = 4;
    let u = *graph.node_weight(u.into()).unwrap();
    let tail = 3;
    let head = 5;
    let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
    assert!(is_head_to_tail(&graph, edge, u, true));
}

#[test]
fn test_is_head_to_tail_false_root_in_tail() {
    let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
        .with_tree_edges(&LOW_LIM_GRAPH)
        .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
        .build();
    let u = 3;
    let u = *graph.node_weight(u.into()).unwrap();
    let tail = 4;
    let head = 3;
    let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
    assert!(!is_head_to_tail(&graph, edge, u, false));
}

#[test]
fn test_is_head_to_tail_false_root_in_head() {
    let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
        .with_tree_edges(&LOW_LIM_GRAPH)
        .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
        .build();
    let u = 2;
    let u = *graph.node_weight(u.into()).unwrap();
    let tail = 2;
    let head = 0;
    let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
    assert!(!is_head_to_tail(&graph, edge, u, true));
}

#[test]
fn enter_edge_find_edge() {
    let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
        .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
        .build();

    init_cutvalues(&mut graph);
    init_low_lim(&mut graph);

    let leave_edge = leave_edge(&graph).unwrap();
    let enter_edge = enter_edge(&mut graph, leave_edge, 1);
    let (tail, head) = graph.edge_endpoints(enter_edge).unwrap();
    assert!(tail == NodeIndex::from(0));
    assert!(head == NodeIndex::from(4) || head == NodeIndex::from(5));
}

mod integration {

    use petgraph::stable_graph::StableDiGraph;

    use super::{EXAMPLE_GRAPH, GraphBuilder};
    use crate::standalone_sugiyama::{
        config::RankingType,
        p1_layering::{Edge, Vertex, rank, slack},
    };

    fn is_correct(graph: StableDiGraph<Vertex, Edge>, minimum_length: i32) -> bool {
        // all cut values must be positive,
        0 <= graph.edge_indices()
            .filter(|e| graph[*e].is_tree_edge)
            .filter_map(|e| graph[e].cut_value)
            .min()
            .unwrap_or(0)
        &&
        // tree must be tight
        0 == graph.edge_indices()
            .filter(|e| graph[*e].is_tree_edge)
            .map(|e| slack(&graph, e, minimum_length))
            .max()
            .unwrap()
        &&
        // minimum rank must be 0
        0 == graph.node_weights()
            .map(|w| w.rank)
            .min()
            .unwrap()
    }

    #[test]
    fn run_algorithm_example_graph() {
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH).build();
        rank(&mut graph, 1, RankingType::MinimizeEdgeLength);
        assert!(is_correct(graph, 1));
    }
}
