use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use super::{Edge, Vertex};

const ONE_DUMMY: [(u32, u32); 9] = [
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
const ONE_DUMMY_RANKS: [(u32, u32); 8] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 1),
    (5, 1),
    (6, 2),
    (7, 4),
];

const THREE_DUMMIES: [(u32, u32); 10] = [
    (0, 1),
    (0, 2),
    (1, 4),
    (1, 5),
    (2, 3),
    (3, 8),
    (4, 6),
    (5, 7),
    (6, 7),
    (7, 8),
];
const THREE_DUMMIES_RANKS: [(u32, u32); 9] = [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 3),
    (7, 4),
    (8, 5),
];

const COMPLEX_EXAMPLE: [(u32, u32); 21] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 13),
    (3, 9),
    (3, 11),
    (4, 10),
    (5, 11),
    (6, 11),
    (7, 11),
    (8, 15),
    (11, 12),
    (11, 13),
    (12, 14),
    (12, 15),
    (12, 13),
];
const COMPLEX_EXAMPLE_RANKS: [(u32, u32); 16] = [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
    (8, 2),
    (9, 2),
    (10, 3),
    (11, 3),
    (12, 4),
    (14, 5),
    (15, 5),
    (13, 5),
];
const _TYPE_2_CONFLICT_2_COLS: [(u32, u32); 8] = [
    (0, 3),
    (1, 2),
    (2, 5),
    (3, 4),
    (4, 7),
    (5, 6),
    (6, 8),
    (7, 8),
];

const _TYPE_2_CONFLICT_2_COLS_RANKS: [(u32, u32); 9] = [
    (0, 0),
    (1, 0),
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 2),
    (6, 3),
    (7, 3),
    (8, 4),
];

const _TYPE_2_CONFLICT_2_COLS_DUMMIES: [u32; 4] = [2, 3, 4, 5];

struct GraphBuilder {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl GraphBuilder {
    fn new_from_edges_with_ranking(edges: &[(u32, u32)], ranks: &[(u32, u32)]) -> Self {
        let mut graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
        for (v, rank) in ranks {
            graph[NodeIndex::from(*v)].rank = *rank as i32;
        }

        Self {
            graph,
            minimum_length: 1,
        }
    }

    #[allow(dead_code)]
    fn with_minimum_length(mut self, minimum_length: i32) -> Self {
        self.minimum_length = minimum_length;
        self
    }

    #[allow(dead_code)]
    fn with_dummies(mut self, dummies: &[u32]) -> Self {
        for dummy in dummies {
            self.graph[NodeIndex::from(*dummy)].is_dummy = true;
        }
        self
    }

    fn build(self) -> (StableDiGraph<Vertex, Edge>, i32) {
        (self.graph, self.minimum_length)
    }
}

#[cfg(test)]
mod insert_dummy_vertices {
    use super::{GraphBuilder, ONE_DUMMY_RANKS};
    use crate::p2_reduce_crossings::{
        insert_dummy_vertices,
        tests::{ONE_DUMMY, THREE_DUMMIES, THREE_DUMMIES_RANKS},
    };

    #[test]
    fn insert_dummy_vertices_one_dummy() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&ONE_DUMMY, &ONE_DUMMY_RANKS).build();
        let n_vertices = graph.node_count();
        let removed_edges = insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 1);
        // one more vertex
        assert_eq!(n_vertices + 1, graph.node_count());
        // should have removed one edge
        assert_eq!(removed_edges.len(), 1);
    }

    #[test]
    fn insert_dummy_vertices_three_dummies() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&THREE_DUMMIES, &THREE_DUMMIES_RANKS).build();
        println!("graph: {:?}", graph);
        let n_vertices = graph.node_count();
        let removed_edges = insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        println!("graph: {:?}", graph);

        println!("removed_edges: {:?}", removed_edges);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 3);
        // one more vertex
        assert_eq!(n_vertices + 3, graph.node_count());
        // should have removed three edges (todo: double check this)
        assert_eq!(removed_edges.len(), 2);
    }
}

mod init_order {
    use super::{
        COMPLEX_EXAMPLE, COMPLEX_EXAMPLE_RANKS, GraphBuilder, ONE_DUMMY, ONE_DUMMY_RANKS,
        THREE_DUMMIES, THREE_DUMMIES_RANKS,
    };
    use crate::p2_reduce_crossings::insert_dummy_vertices;

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_one_dummy() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&ONE_DUMMY, &ONE_DUMMY_RANKS).build();
        let _removed_edges = insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_three_dummies() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&THREE_DUMMIES, &THREE_DUMMIES_RANKS).build();
        let _removed_edges = insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_seven_dummies() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&COMPLEX_EXAMPLE, &COMPLEX_EXAMPLE_RANKS)
                .build();

        let _removed_edges = insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }
}

// TODO: Add new tests for Order crosscount
#[cfg(test)]
mod order {
    use petgraph::stable_graph::StableDiGraph;

    use crate::{Edge, Vertex, p2::order_layer, p2_reduce_crossings::Order};

    /// Shorthand for creating a default vertex with a specified rank.
    fn vertex_with_rank(rank: i32) -> Vertex {
        Vertex {
            rank,
            ..Default::default()
        }
    }

    #[test]
    fn two_crossings() {
        let mut graph = StableDiGraph::new();
        let n0 = graph.add_node(vertex_with_rank(0));
        let n1 = graph.add_node(vertex_with_rank(0));
        let n2 = graph.add_node(vertex_with_rank(0));
        let s0 = graph.add_node(vertex_with_rank(1));
        let s1 = graph.add_node(vertex_with_rank(1));

        graph.add_edge(n0, s1, Edge::default());
        graph.add_edge(n1, s0, Edge::default());
        graph.add_edge(n2, s0, Edge::default());

        let order = Order::new(vec![vec![n0, n1, n2], vec![s0, s1]]);
        assert_eq!(order.bilayer_cross_count(&graph, 0), 2);
    }

    #[test]
    fn four_crossings() {
        let mut graph = StableDiGraph::new();
        let n0 = graph.add_node(vertex_with_rank(0));
        let n1 = graph.add_node(vertex_with_rank(0));
        let n2 = graph.add_node(vertex_with_rank(0));
        let n3 = graph.add_node(vertex_with_rank(0));
        let s0 = graph.add_node(vertex_with_rank(1));
        let s1 = graph.add_node(vertex_with_rank(1));
        let s2 = graph.add_node(vertex_with_rank(1));
        let s3 = graph.add_node(vertex_with_rank(1));

        graph.add_edge(n0, s3, Edge::default());
        graph.add_edge(n1, s2, Edge::default());
        graph.add_edge(n2, s1, Edge::default());
        graph.add_edge(n3, s0, Edge::default());

        let order = Order::new(vec![vec![n0, n1, n2, n3], vec![s0, s1, s2, s3]]);

        assert_eq!(order.bilayer_cross_count(&graph, 0), 6);
    }

    #[test]
    fn twelve_crossings() {
        let mut g = StableDiGraph::<Vertex, Edge>::new();
        let n0 = g.add_node(vertex_with_rank(0));
        let n1 = g.add_node(vertex_with_rank(0));
        let n2 = g.add_node(vertex_with_rank(0));
        let n3 = g.add_node(vertex_with_rank(0));
        let n4 = g.add_node(vertex_with_rank(0));
        let n5 = g.add_node(vertex_with_rank(0));
        let s0 = g.add_node(vertex_with_rank(1));
        let s1 = g.add_node(vertex_with_rank(1));
        let s2 = g.add_node(vertex_with_rank(1));
        let s3 = g.add_node(vertex_with_rank(1));
        let s4 = g.add_node(vertex_with_rank(1));

        g.add_edge(n0, s0, Edge::default());
        g.add_edge(n1, s1, Edge::default());
        g.add_edge(n1, s2, Edge::default());
        g.add_edge(n2, s0, Edge::default());
        g.add_edge(n2, s3, Edge::default());
        g.add_edge(n2, s4, Edge::default());
        g.add_edge(n3, s0, Edge::default());
        g.add_edge(n3, s3, Edge::default());
        g.add_edge(n4, s3, Edge::default());
        g.add_edge(n5, s2, Edge::default());
        g.add_edge(n5, s4, Edge::default());

        let order = Order::new(vec![vec![n0, n1, n2, n3, n4, n5], vec![s0, s1, s2, s3, s4]]);
        assert_eq!(order.crossings(&g), 12);
    }

    #[test]
    fn test_barycenter() {
        let mut graph = StableDiGraph::new();
        let n0 = graph.add_node(vertex_with_rank(0)); // 33
        let n1 = graph.add_node(vertex_with_rank(0)); // 28
        let n2 = graph.add_node(vertex_with_rank(0)); // 6
        let n3 = graph.add_node(vertex_with_rank(0)); // 42
        let n4 = graph.add_node(vertex_with_rank(0)); // 31
        let n5 = graph.add_node(vertex_with_rank(0)); // 25
        let n6 = graph.add_node(vertex_with_rank(0)); // 38
        let n7 = graph.add_node(vertex_with_rank(0)); // 34
        let s0 = graph.add_node(vertex_with_rank(1)); // 9
        let s1 = graph.add_node(vertex_with_rank(1)); // 43
        let s2 = graph.add_node(vertex_with_rank(1)); // 8
        let s3 = graph.add_node(vertex_with_rank(1)); // 39
        let s4 = graph.add_node(vertex_with_rank(1)); // 35

        graph.add_edge(n0, s0, Edge::default());
        graph.add_edge(n1, s0, Edge::default());
        graph.add_edge(n2, s0, Edge::default());
        graph.add_edge(n2, s2, Edge::default());
        graph.add_edge(n3, s1, Edge::default());
        graph.add_edge(n4, s2, Edge::default());
        graph.add_edge(n5, s2, Edge::default());
        graph.add_edge(n6, s3, Edge::default());
        graph.add_edge(n7, s4, Edge::default());

        let _inner = vec![
            vec![n0, n2, n4, n3, n6, n7, n1, n5],
            vec![s0, s1, s2, s3, s4],
        ];
        let order = Order::new(_inner);
        let expected_order = order_layer(
            &graph,
            false,
            &order,
            crate::p2_reduce_crossings::barycenter,
        );
        assert_eq!(
            expected_order._inner[0],
            vec![n0, n1, n2, n3, n4, n5, n6, n7]
        );
    }

    #[test]
    fn deterministic_ordering_with_tiebreaker() {
        let mut graph = StableDiGraph::new();
        let l0_n0 = graph.add_node(vertex_with_rank(0));
        let l0_n1 = graph.add_node(vertex_with_rank(0));
        let l1_n0 = graph.add_node(vertex_with_rank(1));
        let l1_n1 = graph.add_node(vertex_with_rank(1));
        let l2_n0 = graph.add_node(vertex_with_rank(2));
        let l2_n1 = graph.add_node(vertex_with_rank(2));
        let l3_n0 = graph.add_node(vertex_with_rank(3));
        let l3_n1 = graph.add_node(vertex_with_rank(3));

        graph.add_edge(l0_n0, l1_n0, Edge::default());
        graph.add_edge(l0_n0, l1_n1, Edge::default());
        graph.add_edge(l0_n1, l1_n0, Edge::default());
        graph.add_edge(l0_n1, l1_n1, Edge::default());

        graph.add_edge(l1_n0, l2_n0, Edge::default());
        graph.add_edge(l1_n0, l2_n1, Edge::default());
        graph.add_edge(l1_n1, l2_n0, Edge::default());
        graph.add_edge(l1_n1, l2_n1, Edge::default());

        graph.add_edge(l2_n0, l3_n0, Edge::default());
        graph.add_edge(l2_n0, l3_n1, Edge::default());
        graph.add_edge(l2_n1, l3_n0, Edge::default());
        graph.add_edge(l2_n1, l3_n1, Edge::default());

        let layer0 = vec![l0_n0, l0_n1];
        let layer1 = vec![l1_n0, l1_n1];
        let layer2 = vec![l2_n0, l2_n1];
        let layer3 = vec![l3_n0, l3_n1];

        let original_order = vec![
            layer0.clone(),
            layer1.clone(),
            layer2.clone(),
            layer3.clone(),
        ];
        let order = Order::new(original_order);

        let result = order_layer(&graph, true, &order, crate::p2_reduce_crossings::barycenter);

        let l0_result = &result._inner[0];
        let l1_result = &result._inner[1];
        let l2_result = &result._inner[2];
        let l3_result = &result._inner[3];

        assert_eq!(l0_result[0], l0_n0);
        assert_eq!(l0_result[1], l0_n1);
        assert_eq!(l1_result[0], l1_n0);
        assert_eq!(l1_result[1], l1_n1);
        assert_eq!(l2_result[0], l2_n0);
        assert_eq!(l2_result[1], l2_n1);
        assert_eq!(l3_result[0], l3_n0);
        assert_eq!(l3_result[1], l3_n1);
    }

    #[test]
    fn deterministic_ordering_with_random_initial_orders() {
        let mut graph = StableDiGraph::new();
        let l0_n0 = graph.add_node(vertex_with_rank(0));
        let l0_n1 = graph.add_node(vertex_with_rank(0));
        let l1_n0 = graph.add_node(vertex_with_rank(1));
        let l1_n1 = graph.add_node(vertex_with_rank(1));
        let l2_n0 = graph.add_node(vertex_with_rank(2));
        let l2_n1 = graph.add_node(vertex_with_rank(2));
        let l3_n0 = graph.add_node(vertex_with_rank(3));
        let l3_n1 = graph.add_node(vertex_with_rank(3));

        graph.add_edge(l0_n0, l1_n0, Edge::default());
        graph.add_edge(l0_n0, l1_n1, Edge::default());
        graph.add_edge(l0_n1, l1_n0, Edge::default());
        graph.add_edge(l0_n1, l1_n1, Edge::default());

        graph.add_edge(l1_n0, l2_n0, Edge::default());
        graph.add_edge(l1_n0, l2_n1, Edge::default());
        graph.add_edge(l1_n1, l2_n0, Edge::default());
        graph.add_edge(l1_n1, l2_n1, Edge::default());

        graph.add_edge(l2_n0, l3_n0, Edge::default());
        graph.add_edge(l2_n0, l3_n1, Edge::default());
        graph.add_edge(l2_n1, l3_n0, Edge::default());
        graph.add_edge(l2_n1, l3_n1, Edge::default());

        let results: Vec<Vec<Vec<petgraph::stable_graph::NodeIndex>>> = (0..20)
            .map(|_| {
                let layer0 = vec![l0_n0, l0_n1];
                let layer1 = vec![l1_n1, l1_n0];
                let layer2 = vec![l2_n1, l2_n0];
                let layer3 = vec![l3_n1, l3_n0];

                let order = Order::new(vec![layer0, layer1, layer2, layer3]);
                let result =
                    order_layer(&graph, true, &order, crate::p2_reduce_crossings::barycenter);
                result._inner.clone()
            })
            .collect();

        let first = &results[0];
        for result in &results {
            assert_eq!(
                result, first,
                "All runs should produce the same ordering due to tiebreaker"
            );
        }
    }

    #[test]
    fn deterministic_ordering_exhaustive_permutations_in_symmetric_k3_3() {
        let mut graph = StableDiGraph::new();

        let l0_n0 = graph.add_node(vertex_with_rank(0));
        let l0_n1 = graph.add_node(vertex_with_rank(0));
        let l0_n2 = graph.add_node(vertex_with_rank(0));

        let l1_n0 = graph.add_node(vertex_with_rank(1));
        let l1_n1 = graph.add_node(vertex_with_rank(1));
        let l1_n2 = graph.add_node(vertex_with_rank(1));

        let l2_n0 = graph.add_node(vertex_with_rank(2));
        let l2_n1 = graph.add_node(vertex_with_rank(2));
        let l2_n2 = graph.add_node(vertex_with_rank(2));

        // Fully connect layer 0 -> layer 1 and layer 1 -> layer 2.
        for &u in &[l0_n0, l0_n1, l0_n2] {
            for &v in &[l1_n0, l1_n1, l1_n2] {
                graph.add_edge(u, v, Edge::default());
            }
        }
        for &u in &[l1_n0, l1_n1, l1_n2] {
            for &v in &[l2_n0, l2_n1, l2_n2] {
                graph.add_edge(u, v, Edge::default());
            }
        }

        let layer0 = vec![l0_n0, l0_n1, l0_n2];

        let perms_l1 = vec![
            vec![l1_n0, l1_n1, l1_n2],
            vec![l1_n0, l1_n2, l1_n1],
            vec![l1_n1, l1_n0, l1_n2],
            vec![l1_n1, l1_n2, l1_n0],
            vec![l1_n2, l1_n0, l1_n1],
            vec![l1_n2, l1_n1, l1_n0],
        ];
        let perms_l2 = vec![
            vec![l2_n0, l2_n1, l2_n2],
            vec![l2_n0, l2_n2, l2_n1],
            vec![l2_n1, l2_n0, l2_n2],
            vec![l2_n1, l2_n2, l2_n0],
            vec![l2_n2, l2_n0, l2_n1],
            vec![l2_n2, l2_n1, l2_n0],
        ];

        // In this symmetric graph, every vertex in layer 1 and layer 2 has the same barycenter.
        // The final order must therefore be determined entirely by the tiebreaker.
        for layer1 in &perms_l1 {
            for layer2 in &perms_l2 {
                let order = Order::new(vec![layer0.clone(), layer1.clone(), layer2.clone()]);
                let result =
                    order_layer(&graph, true, &order, crate::p2_reduce_crossings::barycenter);

                assert_eq!(result._inner[1], vec![l1_n0, l1_n1, l1_n2]);
                assert_eq!(result._inner[2], vec![l2_n0, l2_n1, l2_n2]);
            }
        }
    }

    #[test]
    fn deterministic_ordering_uses_sort_bias_on_ties() {
        let mut graph = StableDiGraph::new();

        let l0 = graph.add_node(vertex_with_rank(0));

        let l1_a = graph.add_node(vertex_with_rank(1));
        let l1_b = graph.add_node(vertex_with_rank(1));
        let l1_c = graph.add_node(vertex_with_rank(1));

        // Connect the single upper node to all three, giving them identical barycenters.
        graph.add_edge(l0, l1_a, Edge::default());
        graph.add_edge(l0, l1_b, Edge::default());
        graph.add_edge(l0, l1_c, Edge::default());

        // Bias ordering: c (lowest) -> a -> b (highest).
        graph[l1_a].set_sort_bias(0);
        graph[l1_b].set_sort_bias(5);
        graph[l1_c].set_sort_bias(-5);

        // Scramble initial order to ensure tie-breaking is what fixes it.
        let order = Order::new(vec![vec![l0], vec![l1_b, l1_a, l1_c]]);
        let result = order_layer(&graph, true, &order, crate::p2_reduce_crossings::barycenter);

        assert_eq!(result._inner[1], vec![l1_c, l1_a, l1_b]);
    }
}
