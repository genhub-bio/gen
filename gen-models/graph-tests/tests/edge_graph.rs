use gen_core::{GraphNode, HashId, Strand};
use gen_graph::graph_loader::{GraphLoadBlock, GraphLoadEdge, build_graph};

fn graph_block(id: i64, node_id: HashId, start: i64, end: i64) -> GraphLoadBlock {
    GraphLoadBlock {
        id,
        node_id,
        start,
        end,
    }
}

fn graph_edge(
    id: &str,
    source_node_id: HashId,
    source_coordinate: i64,
    target_node_id: HashId,
    target_coordinate: i64,
) -> GraphLoadEdge {
    GraphLoadEdge {
        edge_id: HashId::convert_str(id),
        source_node_id,
        source_coordinate,
        source_strand: Strand::Forward,
        target_node_id,
        target_coordinate,
        target_strand: Strand::Forward,
        chromosome_index: 0,
        phased: 0,
        created_on: 0,
    }
}

fn graph_node(block: &GraphLoadBlock) -> GraphNode {
    GraphNode {
        node_id: block.node_id,
        sequence_start: block.start,
        sequence_end: block.end,
    }
}

#[test]
fn test_build_graph_routes_incoming_edge_to_junction() {
    let source_node_id = HashId::convert_str("incoming-source");
    let target_node_id = HashId::convert_str("incoming-target");
    let blocks = vec![
        graph_block(0, source_node_id, 0, 3),
        graph_block(1, target_node_id, 0, 0),
        graph_block(2, target_node_id, 0, 1),
    ];
    let edges = vec![graph_edge(
        "incoming-edge",
        source_node_id,
        3,
        target_node_id,
        0,
    )];

    let (graph, _) = build_graph(&edges, &blocks);

    assert!(
        graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[1])),
        "incoming edge should terminate at the junction"
    );
    assert!(
        !graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[2])),
        "incoming edge should not bypass the junction"
    );
    assert_eq!(graph.edge_count(), 1, "should project one block edge");
}

#[test]
fn test_build_graph_routes_outgoing_edge_from_junction() {
    let source_node_id = HashId::convert_str("outgoing-source");
    let target_node_id = HashId::convert_str("outgoing-target");
    let blocks = vec![
        graph_block(0, source_node_id, 0, 1),
        graph_block(1, source_node_id, 1, 1),
        graph_block(2, target_node_id, 0, 2),
    ];
    let edges = vec![graph_edge(
        "outgoing-edge",
        source_node_id,
        1,
        target_node_id,
        0,
    )];

    let (graph, _) = build_graph(&edges, &blocks);

    assert!(
        graph.contains_edge(graph_node(&blocks[1]), graph_node(&blocks[2])),
        "outgoing edge should originate at the junction"
    );
    assert!(
        !graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[2])),
        "outgoing edge should not bypass the junction"
    );
    assert_eq!(graph.edge_count(), 1, "should create one block edge");
}

#[test]
fn test_build_graph_creates_same_coordinate_edge_from_start_junction() {
    let node_id = HashId::convert_str("same-coordinate-node");
    let blocks = vec![graph_block(0, node_id, 0, 0), graph_block(1, node_id, 0, 1)];
    let edges = vec![graph_edge("same-coordinate-edge", node_id, 0, node_id, 0)];

    let (graph, _) = build_graph(&edges, &blocks);

    assert!(
        graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[1])),
        "same-coordinate edge should connect the junction to adjacent sequence"
    );
    assert!(
        !graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[0])),
        "same-coordinate edge should not add a redundant junction self-loop"
    );
    assert_eq!(graph.edge_count(), 1, "should create one block edge");
}

#[test]
fn test_build_graph_creates_same_coordinate_edge_into_end_junction() {
    let node_id = HashId::convert_str("ending-same-coordinate-node");
    let blocks = vec![graph_block(0, node_id, 0, 1), graph_block(1, node_id, 1, 1)];
    let edges = vec![graph_edge(
        "ending-same-coordinate-edge",
        node_id,
        1,
        node_id,
        1,
    )];

    let (graph, _) = build_graph(&edges, &blocks);

    assert!(
        graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[1])),
        "same-coordinate edge should connect adjacent sequence into the junction"
    );
    assert!(
        !graph.contains_edge(graph_node(&blocks[1]), graph_node(&blocks[1])),
        "same-coordinate edge should not add a redundant junction self-loop"
    );
    assert_eq!(graph.edge_count(), 1, "should create one block edge");
}

#[test]
fn test_build_graph_creates_same_coordinate_edge_without_junction_directly() {
    let node_id = HashId::convert_str("interior-same-coordinate-node");
    let blocks = vec![graph_block(0, node_id, 0, 1), graph_block(1, node_id, 1, 2)];
    let edges = vec![graph_edge(
        "interior-same-coordinate-edge",
        node_id,
        1,
        node_id,
        1,
    )];

    let (graph, _) = build_graph(&edges, &blocks);

    assert!(
        graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[1])),
        "same-coordinate edge should directly connect sequence blocks without a junction"
    );
    assert_eq!(graph.edge_count(), 1, "should create one block edge");
}

#[test]
fn test_build_graph_omits_same_coordinate_edge_without_adjacent_sequence() {
    let node_id = HashId::convert_str("isolated-junction");
    let blocks = vec![graph_block(0, node_id, 0, 0)];
    let edges = vec![graph_edge("isolated-edge", node_id, 0, node_id, 0)];

    let (graph, edges_by_node_pair) = build_graph(&edges, &blocks);

    assert!(
        !graph.contains_edge(graph_node(&blocks[0]), graph_node(&blocks[0])),
        "a junction without adjacent sequence should not create a graph self-loop"
    );
    assert_eq!(graph.edge_count(), 0, "should not create a block edge");
    assert!(
        edges_by_node_pair.is_empty(),
        "omitted self-loop should not have a block-pair mapping"
    );
}
