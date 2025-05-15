#[cfg(test)]
mod standalone_sugiyama_tests {
    use crate::views::standalone_sugiyama::algorithm::{ 
        build_layout_graph, Edge, Vertex,
    };
    use crate::views::standalone_sugiyama::configure::{Config};
    use petgraph::stable_graph::StableDiGraph;

    #[cfg(feature = "python-bindings")]
    use crate::views::standalone_sugiyama::algorithm::call_python_with_layout_graph;


    fn setup_test(edges: Vec<(usize, usize)>) -> StableDiGraph<Vertex, Edge> {
        let mut graph = StableDiGraph::<Vertex, Edge>::new();
        let max_node_id = edges.iter().map(|(u, v)| *u.max(v)).max().unwrap();

        let nodes: Vec<_> = (0..=max_node_id).map(|i| graph.add_node(
            Vertex::new(i, (1.0, 1.0))
        )).collect();

        for (u, v) in edges {
            graph.add_edge(nodes[u], nodes[v], Edge::default());
        }

        graph
    }

    #[test]
    fn test_build_layout_graph() {
        // Simple graph with 5 nodes and no dummy nodes
        let edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (3, 4),
        ];

        let graph = setup_test(edges.to_vec());
        let config = Config::default();
        let layout_graph = build_layout_graph(graph.clone(), &config);

        let coords = layout_graph.node_weights()
                                 .map(|v| (v.original_id.unwrap(), (v.x, v.y)))
                                 .collect::<Vec<_>>();
        assert_eq!(coords, vec![
            (0, (1, 0)),
            (1, (2, 1)),
            (2, (0, 1)),
            (3, (1, 2)),
            (4, (1, 3))
        ]);
    }

    #[test]
    fn test_build_layout_graph_with_dummies() {
        // The edge from 1 to 3 skips a layer => we need a dummy node
        let edges = [
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
        ];

        let graph = setup_test(edges.to_vec());
        let config = Config::default();
        let layout_graph = build_layout_graph(graph.clone(), &config);

        // We need to handle dummy nodes here, hence the string conversion
        let coords = layout_graph.node_indices()
            .map(|node_idx| {
                let node = &layout_graph[node_idx];
                let id_str = node.original_id
                                 .map_or(
                                    format!("d{}", node_idx.index()), 
                                    |id| id.to_string()
                                );
            (id_str, (node.x, node.y))
             })
            .collect::<Vec<_>>();

        assert_eq!(coords, vec![
            ("0".to_string(), (1, 0)), 
            ("1".to_string(), (1, 1)), 
            ("2".to_string(), (2, 2)), 
            ("3".to_string(), (1, 3)), 
            ("4".to_string(), (1, 4)),
            ("d5".to_string(), (0, 2)), // dummy node
        ]);
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn test_call_python_with_layout_graph() {
        let edges = [
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
        ];

        let graph = setup_test(edges.to_vec());
        let config = Config::default();
        let layout_graph = build_layout_graph(graph.clone(), &config);

        pyo3::prepare_freethreaded_python();
        let graph_from_python = call_python_with_layout_graph(&layout_graph, "route_graph", "route_graph");
        
        // For initial tests, we just manipulated the coordinates of the nodes
        assert!(graph_from_python.unwrap().node_weights().all(|v| v.x == 100 && v.y == 100));
    }
    
}
