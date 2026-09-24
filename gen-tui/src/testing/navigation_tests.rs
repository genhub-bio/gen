#[cfg(test)]
mod tests {
    use ratatui::{buffer::Buffer, layout::Rect, widgets::StatefulWidget as _};

    use crate::{
        graph_view::{GraphView, GraphViewState},
        layout_engine::LayoutEngine,
        navigator::Navigator,
        testing::mocks::{MockDomainGraph, TestNodeSizers},
    };

    #[test]
    fn test_layer_based_navigation() {
        // Create a diamond-shaped test graph to test layer navigation
        // Layer 0: node 0
        // Layer 1: nodes 1, 2
        // Layer 2: node 3
        let mut domain_graph = MockDomainGraph::new();
        let n0 = domain_graph.add_node(());
        let n1 = domain_graph.add_node(());
        let n2 = domain_graph.add_node(());
        let n3 = domain_graph.add_node(());
        // Create diamond structure
        domain_graph.add_edge(n0, n1, ());
        domain_graph.add_edge(n0, n2, ());
        domain_graph.add_edge(n1, n3, ());
        domain_graph.add_edge(n2, n3, ());
        // Use 1x1 nodes to make navigation easier - any movement crosses boundaries
        let node_sizer = TestNodeSizers::fixed_1x1();
        let mut engine = LayoutEngine::new(domain_graph.clone());
        let mut state = GraphViewState::default();

        // Render once (also initializes the cursor onto the engine's default anchor) to see
        // the entire graph.
        let area = Rect::new(0, 0, 100, 50);
        let mut buffer = Buffer::empty(area);
        GraphView::new(&mut engine, &node_sizer).render(area, &mut buffer, &mut state);

        let initial_node = state.cursor.node.expect("should initialize the cursor");
        Navigator::move_horizontal(&mut state.cursor, 1, &state.frame)
            .expect("should move to the next layer");
        let node_after_right = state.cursor.node.expect("should keep a cursor node");
        let initial_layer = state
            .frame
            .layer_of(initial_node)
            .expect("should place the initial node in a layer");
        let right_layer = state
            .frame
            .layer_of(node_after_right)
            .expect("should place the next node in a layer");
        assert_ne!(initial_layer, right_layer);

        Navigator::move_horizontal(&mut state.cursor, 1, &state.frame)
            .expect("should move to the final layer");
        let final_node = state.cursor.node.expect("should keep a cursor node");
        let final_layer = state
            .frame
            .layer_of(final_node)
            .expect("should place the final node in a layer");
        assert_ne!(right_layer, final_layer);

        Navigator::move_horizontal(&mut state.cursor, -1, &state.frame)
            .expect("should return to the middle layer");
        let middle_node = state.cursor.node.expect("should keep a cursor node");
        assert_eq!(state.frame.layer_of(middle_node), Some(right_layer));

        Navigator::move_horizontal(&mut state.cursor, -1, &state.frame)
            .expect("should return to the initial layer");
        let returned_node = state.cursor.node.expect("should keep a cursor node");
        assert_eq!(state.frame.layer_of(returned_node), Some(initial_layer));
    }

    #[test]
    fn test_cursor_navigates_along_chain() {
        let mut domain_graph = MockDomainGraph::new();
        let nodes: Vec<_> = (0..5).map(|_| domain_graph.add_node(())).collect();
        for i in 0..4 {
            domain_graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        let node_sizer = TestNodeSizers::fixed_1x1();
        let mut engine = LayoutEngine::new(domain_graph.clone());
        let mut state = GraphViewState::default();

        let area = Rect::new(0, 0, 200, 50);
        let mut buffer = Buffer::empty(area);
        GraphView::new(&mut engine, &node_sizer).render(area, &mut buffer, &mut state);

        state.cursor.set_node(nodes[1], (1.0, 0.5));
        Navigator::move_horizontal(&mut state.cursor, 1, &state.frame).unwrap();

        assert_eq!(state.cursor.node, Some(nodes[2]));
    }
}
