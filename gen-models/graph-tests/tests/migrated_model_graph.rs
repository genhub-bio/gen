use gen_core::{GraphNode, HashId, NodeIntervalBlock, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange, BlockGroupError, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    edge::Edge,
    node::Node,
    region::{ResolvedGenRegion, ResolvedRegionKind},
    sample::{NewSample, Sample},
    sequence::Sequence,
};
use gen_models_graph_tests::{get_connection, setup_block_group};

#[test]
fn test_get_graph_branched_graph() {
    let conn = get_connection(None).expect("should create an in-memory graph database");
    Collection::get_or_create(&conn, "test").expect("should create the collection");
    Sample::get_or_create(
        &conn,
        NewSample {
            name: "test",
            ..Default::default()
        },
    )
    .expect("should create the sample");
    let block_group = BlockGroup::create(
        &conn,
        NewBlockGroup {
            collection_name: "test",
            sample_name: "test",
            name: "branched",
            ..Default::default()
        },
    )
    .expect("should create the block group");

    let node_ids = [
        ("AAA", "node-aaa"),
        ("GGG", "node-ggg"),
        ("TTT", "node-ttt"),
        ("CCC", "node-ccc"),
        ("ATC", "node-atc"),
    ]
    .map(|(bases, name)| {
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(bases)
            .save(&conn)
            .expect("should save the sequence");
        Node::create(&conn, &sequence.hash, &HashId::convert_str(name))
            .expect("should create the node")
    });
    let [node_aaa, node_ggg, node_ttt, node_ccc, node_atc] = node_ids;
    let edges = [
        (node_aaa, 3, node_ttt, 0),
        (node_ggg, 3, node_ttt, 0),
        (node_ttt, 3, node_ccc, 0),
        (node_ttt, 3, node_atc, 0),
    ]
    .map(|(source, source_coordinate, target, target_coordinate)| {
        Edge::create(
            &conn,
            source,
            source_coordinate,
            Strand::Forward,
            target,
            target_coordinate,
            Strand::Forward,
        )
        .expect("should create the edge")
    });
    BlockGroupEdge::bulk_create(
        &conn,
        &edges
            .iter()
            .map(|edge| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge.id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>(),
    );

    let graph = gen_graph::models::load_block_group_graph(&conn, &block_group.id, None)
        .expect("should load the branched graph");
    assert_eq!(graph.nodes().len(), 7);
    assert_eq!(graph.all_edges().count(), 4);
    let expected_edges = [
        (node_aaa, 3, 3, node_ttt, 0, 3, edges[0].id),
        (node_ggg, 3, 3, node_ttt, 0, 3, edges[1].id),
        (node_ttt, 0, 3, node_ccc, 0, 0, edges[2].id),
        (node_ttt, 0, 3, node_atc, 0, 0, edges[3].id),
    ];
    for (source, source_start, source_end, target, target_start, target_end, edge_id) in
        expected_edges
    {
        let source = GraphNode {
            node_id: source,
            sequence_start: source_start,
            sequence_end: source_end,
        };
        let target = GraphNode {
            node_id: target,
            sequence_start: target_start,
            sequence_end: target_end,
        };
        let weights = graph
            .edge_weight(source, target)
            .unwrap_or_else(|| panic!("should contain graph edge {source:?} -> {target:?}"));
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].edge_id, edge_id);
    }
}

#[test]
fn test_error_on_out_of_bounds_change() {
    let conn = get_connection(None).expect("should create an in-memory graph database");
    let (block_group_id, path) = setup_block_group(&conn);
    let sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .expect("should save the deletion sequence");
    let node_id = Node::create(&conn, &sequence.hash, &HashId::convert_str("1"))
        .expect("should create the deletion node");
    let deletion = PathBlock {
        node_id,
        block_sequence: sequence
            .get_sequence(None, None)
            .expect("should load the deletion sequence"),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 350,
        path_end: 400,
        strand: Strand::Forward,
    };
    for (start, end) in [(350, 400), (-300, 400)] {
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, start, end)
            .expect("should resolve the requested path region");
        let result = gen_graph::models::insert_change(
            &conn,
            &BlockGroupChange {
                region,
                path_accession: None,
                block: deletion.clone(),
                chromosome_index: 1,
                phased: 0,
                preserve_edge: true,
            },
        );
        assert!(matches!(result, Err(BlockGroupError::ChangeOutOfBounds(_))));
    }
}

fn values_at(
    tree: &intervaltree::IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
) -> Vec<NodeIntervalBlock> {
    let mut values = tree
        .query_point(coordinate)
        .map(|entry| entry.value)
        .collect::<Vec<_>>();
    values.sort();
    values
}

#[test]
fn test_blockgroup_interval_tree() {
    let conn = get_connection(None).expect("should create an in-memory graph database");
    let (block_group_id, _path) = setup_block_group(&conn);
    Sample::get_or_create(
        &conn,
        NewSample {
            name: "child",
            ..Default::default()
        },
    )
    .expect("should create the child sample");
    let child_id = BlockGroup::get_or_create_sample_block_groups(
        &conn,
        "test",
        "child",
        "chr1",
        vec!["test".to_string()],
    )
    .expect("should create the child block group")[0]
        .id;
    let sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .expect("should save the inserted sequence");
    let node_id = Node::create(&conn, &sequence.hash, &HashId::convert_str("insert-node"))
        .expect("should create the inserted node");
    let child =
        BlockGroup::get_by_id(&conn, &child_id, None).expect("should load the child block group");
    gen_graph::models::insert_change(
        &conn,
        &BlockGroupChange {
            region: ResolvedGenRegion {
                block_group: child,
                path: None,
                accession: None,
                annotation: None,
                kind: ResolvedRegionKind::BlockGroup,
                anchor_start: 0,
                anchor_end: 0,
                feature_length: 0,
                start: 7,
                end: 15,
                start_anchors: None,
                end_anchors: None,
                remove_ambiguous_positions: true,
            },
            path_accession: None,
            block: PathBlock {
                node_id,
                block_sequence: sequence
                    .get_sequence(0, 4)
                    .expect("should load the inserted sequence"),
                sequence_start: 0,
                sequence_end: 4,
                path_start: 7,
                path_end: 15,
                strand: Strand::Forward,
            },
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        },
    )
    .expect("should insert the block-group change");

    let original = gen_graph::models::load_block_group_intervaltree(&conn, &block_group_id, false)
        .expect("should load the original interval tree");
    let original_unambiguous =
        gen_graph::models::load_block_group_intervaltree(&conn, &block_group_id, true)
            .expect("should load the original unambiguous interval tree");
    assert_eq!(values_at(&original, 3), values_at(&original_unambiguous, 3));
    assert_eq!(
        values_at(&original, 35),
        values_at(&original_unambiguous, 35)
    );

    let child_tree = gen_graph::models::load_block_group_intervaltree(&conn, &child_id, false)
        .expect("should load the child interval tree");
    let child_unambiguous =
        gen_graph::models::load_block_group_intervaltree(&conn, &child_id, true)
            .expect("should load the child unambiguous interval tree");
    assert_eq!(values_at(&child_tree, 3), values_at(&child_unambiguous, 3));
    assert_eq!(values_at(&child_tree, 30).len(), 2);
    assert!(values_at(&child_unambiguous, 30).is_empty());
    assert_eq!(values_at(&child_unambiguous, 9).len(), 2);
}

#[test]
fn test_blocks_from_edges_after_change() {
    let conn = get_connection(None).expect("should create an in-memory graph database");
    let (block_group_id, path) = setup_block_group(&conn);
    let edges = BlockGroupEdge::edges_for_block_group(&conn, &block_group_id, None);
    assert_eq!(
        Edge::blocks_from_edges(&conn, &block_group_id, &edges, None)
            .expect("should derive the original blocks")
            .len(),
        6
    );
    let sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .expect("should save the inserted sequence");
    let node_id = Node::create(&conn, &sequence.hash, &HashId::convert_str("1"))
        .expect("should create the inserted node");
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15)
        .expect("should resolve the changed region");
    gen_graph::models::insert_change(
        &conn,
        &BlockGroupChange {
            region,
            path_accession: None,
            block: PathBlock {
                node_id,
                block_sequence: sequence
                    .get_sequence(0, 4)
                    .expect("should load the inserted sequence"),
                sequence_start: 0,
                sequence_end: 4,
                path_start: 7,
                path_end: 15,
                strand: Strand::Forward,
            },
            chromosome_index: 0,
            phased: 0,
            preserve_edge: true,
        },
    )
    .expect("should insert the change");
    let mut edges = BlockGroupEdge::edges_for_block_group(&conn, &block_group_id, None);
    assert_eq!(
        Edge::blocks_from_edges(&conn, &block_group_id, &edges, None)
            .expect("should derive the changed blocks")
            .len(),
        9
    );
    edges.reverse();
    assert_eq!(
        Edge::blocks_from_edges(&conn, &block_group_id, &edges, None)
            .expect("should derive the changed blocks independent of edge ordering")
            .len(),
        9
    );
}
