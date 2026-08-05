use core::ops::Range;
use std::collections::HashSet;

use gen_core::{HashId, NO_CHROMOSOME_INDEX, Strand, calculate_hash};
use gen_graph::models::{derive_subgraph, get_all_sequences};
use gen_models::{
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    edge::Edge,
    node::Node,
    sequence::Sequence,
    traits::Query as _,
};
use gen_models_graph_tests::{create_block_group, get_connection, setup_block_group};
use rusqlite::params;

#[test]
fn test_derive_subgraph_one_insertion() {
    /*
    AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -> GGGGGGGGGG
                      \-> AAAAAAAA ->/
    Subgraph range:  |-----------------|
    Sequences of the subgraph are TAAAAAAAAC, TTTTTCCCCC
     */
    let conn = &get_connection(None).unwrap();
    let (block_group1_id, original_path) = setup_block_group(conn);

    let intervaltree = original_path.intervaltree(conn).unwrap();
    let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
    let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAAAAAA")
        .save(conn)
        .unwrap();
    let insert_node_id = Node::create(
        conn,
        &insert_sequence.hash,
        &HashId(calculate_hash(&format!(
            "test-insert-a-node.{}",
            insert_sequence.hash
        ))),
    )
    .unwrap();
    let edge_into_insert = Edge::create(
        conn,
        insert_start_node_id,
        6,
        Strand::Forward,
        insert_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge_out_of_insert = Edge::create(
        conn,
        insert_node_id,
        8,
        Strand::Forward,
        insert_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_1 = Edge::create(
        conn,
        insert_start_node_id,
        6,
        Strand::Forward,
        insert_start_node_id,
        6,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_2 = Edge::create(
        conn,
        insert_end_node_id,
        4,
        Strand::Forward,
        insert_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();

    let edge_ids = [
        &edge_into_insert.id,
        &edge_out_of_insert.id,
        &ref_heal_1.id,
        &ref_heal_2.id,
    ];
    let block_group_edges = edge_ids
        .iter()
        .enumerate()
        .map(|(i, edge_id)| BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: *(*edge_id),
            chromosome_index: if i < 2 { 1 } else { 0 },
            phased: 0,
        })
        .collect::<Vec<BlockGroupEdgeData>>();
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let insert_path = original_path
        .new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert)
        .unwrap();
    assert_eq!(
        insert_path.sequence(conn, None).unwrap(),
        "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
    );

    let all_sequences = get_all_sequences(conn, &block_group1_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
        ])
    );

    let mut blocks = intervaltree
        .query(Range { start: 15, end: 25 })
        .map(|x| x.value)
        .collect::<Vec<_>>();
    blocks.sort_by_key(|a| a.start);
    let start_block = blocks[0];
    let start_node_coordinate = 15 - start_block.start + start_block.sequence_start;
    let end_block = blocks[blocks.len() - 1];
    let end_node_coordinate = 25 - end_block.start + end_block.sequence_start;

    let block_group2 = create_block_group(conn, "test", "test", "chr1.1");
    let node_count_before = Node::query(conn, "SELECT * FROM nodes", params![]).len();
    derive_subgraph(
        conn,
        &block_group1_id,
        &start_block,
        &end_block,
        start_node_coordinate,
        end_node_coordinate,
        &block_group2.id,
        true,
    )
    .unwrap();
    let node_count_after = Node::query(conn, "SELECT * FROM nodes", params![]).len();
    assert_eq!(node_count_after, node_count_before);
    let all_sequences2 = get_all_sequences(conn, &block_group2.id).unwrap();
    assert_eq!(
        all_sequences2,
        HashSet::from_iter(vec!["TTTTTCCCCC".to_string(), "TAAAAAAAAC".to_string(),])
    );
}

#[test]
fn test_derive_subgraph_two_independent_insertions() {
    /*
    AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -----> GGGGGGGGGG
                          \-> AAAAAAAA ->/  \->TTTTTTTT -/
    Subgraph range:     |----------------------------------|
     */
    let conn = &get_connection(None).unwrap();
    let (block_group1_id, original_path) = setup_block_group(conn);

    let intervaltree = original_path.intervaltree(conn).unwrap();
    let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
    let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAAAAAA")
        .save(conn)
        .unwrap();
    let insert_node_id = Node::create(
        conn,
        &insert_sequence.hash,
        &HashId(calculate_hash(&format!(
            "test-insert-a-node.{}",
            insert_sequence.hash
        ))),
    )
    .unwrap();
    let edge_into_insert = Edge::create(
        conn,
        insert_start_node_id,
        6,
        Strand::Forward,
        insert_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge_out_of_insert = Edge::create(
        conn,
        insert_node_id,
        8,
        Strand::Forward,
        insert_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_1 = Edge::create(
        conn,
        insert_start_node_id,
        6,
        Strand::Forward,
        insert_start_node_id,
        6,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_2 = Edge::create(
        conn,
        insert_end_node_id,
        4,
        Strand::Forward,
        insert_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();

    let edge_ids = [
        &edge_into_insert.id,
        &edge_out_of_insert.id,
        &ref_heal_1.id,
        &ref_heal_2.id,
    ];
    let block_group_edges = edge_ids
        .iter()
        .enumerate()
        .map(|(i, edge_id)| BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: *(*edge_id),
            chromosome_index: if i < 2 { 1 } else { 0 },
            phased: 0,
        })
        .collect::<Vec<BlockGroupEdgeData>>();
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let insert_path = original_path
        .new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert)
        .unwrap();
    assert_eq!(
        insert_path.sequence(conn, None).unwrap(),
        "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
    );

    let insert2_start_node_id = intervaltree.query_point(28).next().unwrap().value.node_id;
    let insert2_end_node_id = intervaltree.query_point(32).next().unwrap().value.node_id;

    let insert2_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTTTTTTT")
        .save(conn)
        .unwrap();
    let insert2_node_id = Node::create(
        conn,
        &insert2_sequence.hash,
        &HashId(calculate_hash(&format!(
            "test-insert-t-node.{}",
            insert2_sequence.hash
        ))),
    )
    .unwrap();
    let edge_into_insert2 = Edge::create(
        conn,
        insert2_start_node_id,
        6,
        Strand::Forward,
        insert2_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge_out_of_insert2 = Edge::create(
        conn,
        insert2_node_id,
        8,
        Strand::Forward,
        insert2_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_1 = Edge::create(
        conn,
        insert2_start_node_id,
        6,
        Strand::Forward,
        insert2_start_node_id,
        6,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_2 = Edge::create(
        conn,
        insert2_end_node_id,
        4,
        Strand::Forward,
        insert2_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();

    let edge_ids = [
        &edge_into_insert2.id,
        &edge_out_of_insert2.id,
        &ref_heal_1.id,
        &ref_heal_2.id,
    ];
    let block_group_edges = edge_ids
        .iter()
        .enumerate()
        .map(|(i, edge_id)| BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: *(*edge_id),
            chromosome_index: if i < 2 { 1 } else { 0 },
            phased: 0,
        })
        .collect::<Vec<BlockGroupEdgeData>>();
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let insert2_path = insert_path
        .new_path_with(conn, 28, 32, &edge_into_insert2, &edge_out_of_insert2)
        .unwrap();
    assert_eq!(
        insert2_path.sequence(conn, None).unwrap(),
        "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG"
    );

    let all_sequences = get_all_sequences(conn, &block_group1_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTCCCCCCTTTTTTTTGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG".to_string(),
        ])
    );

    let mut blocks = intervaltree
        .query(Range { start: 15, end: 36 })
        .map(|x| x.value)
        .collect::<Vec<_>>();
    blocks.sort_by_key(|a| a.start);
    let start_block = blocks[0];
    let start_node_coordinate = 15 - start_block.start + start_block.sequence_start;
    let end_block = blocks[blocks.len() - 1];
    let end_node_coordinate = 36 - end_block.start + end_block.sequence_start;

    let block_group2 = create_block_group(conn, "test", "test", "chr1.1");
    derive_subgraph(
        conn,
        &block_group1_id,
        &start_block,
        &end_block,
        start_node_coordinate,
        end_node_coordinate,
        &block_group2.id,
        true,
    )
    .unwrap();
    let all_sequences2 = get_all_sequences(conn, &block_group2.id).unwrap();
    assert_eq!(
        all_sequences2,
        HashSet::from_iter(vec![
            "TTTTTCCCCCCCCCCGGGGGG".to_string(),
            "TAAAAAAAACCCCCCGGGGGG".to_string(),
            "TTTTTCCCCCCTTTTTTTTGG".to_string(),
            "TAAAAAAAACCTTTTTTTTGG".to_string(),
        ])
    );
}

#[test]
fn test_derive_subgraph_two_independent_insertions_and_one_deletion() {
    /*
               /--------------------------------------------\  (<-- Deletion edge)
    AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -----> GGGGGGGGGG
                          \-> AAAAAAAA ->/  \->TTTTTTTT -/
    Subgraph range: |----------------------------------|

    Confirms that deletion edge is ignored and not added to subgraph
     */
    let conn = &get_connection(None).unwrap();
    let (block_group1_id, original_path) = setup_block_group(conn);

    let intervaltree = original_path.intervaltree(conn).unwrap();
    let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
    let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAAAAAA")
        .save(conn)
        .unwrap();
    let insert_node_id = Node::create(
        conn,
        &insert_sequence.hash,
        &HashId(calculate_hash(&format!(
            "test-insert-a-node.{}",
            insert_sequence.hash
        ))),
    )
    .unwrap();
    let edge_into_insert = Edge::create(
        conn,
        insert_start_node_id,
        6,
        Strand::Forward,
        insert_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge_out_of_insert = Edge::create(
        conn,
        insert_node_id,
        8,
        Strand::Forward,
        insert_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_1 = Edge::create(
        conn,
        insert_start_node_id,
        6,
        Strand::Forward,
        insert_start_node_id,
        6,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_2 = Edge::create(
        conn,
        insert_end_node_id,
        4,
        Strand::Forward,
        insert_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();

    let edge_ids = [
        &edge_into_insert.id,
        &edge_out_of_insert.id,
        &ref_heal_1.id,
        &ref_heal_2.id,
    ];
    let block_group_edges = edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: *(*edge_id),
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
        })
        .collect::<Vec<BlockGroupEdgeData>>();
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let insert_path = original_path
        .new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert)
        .unwrap();
    assert_eq!(
        insert_path.sequence(conn, None).unwrap(),
        "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
    );

    let insert2_start_node_id = intervaltree.query_point(28).next().unwrap().value.node_id;
    let insert2_end_node_id = intervaltree.query_point(32).next().unwrap().value.node_id;

    let insert2_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTTTTTTT")
        .save(conn)
        .unwrap();
    let insert2_node_id = Node::create(
        conn,
        &insert2_sequence.hash,
        &HashId(calculate_hash(&format!(
            "test-insert-t-node.{}",
            insert2_sequence.hash
        ))),
    )
    .unwrap();
    let edge_into_insert2 = Edge::create(
        conn,
        insert2_start_node_id,
        6,
        Strand::Forward,
        insert2_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge_out_of_insert2 = Edge::create(
        conn,
        insert2_node_id,
        8,
        Strand::Forward,
        insert2_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_1 = Edge::create(
        conn,
        insert2_start_node_id,
        6,
        Strand::Forward,
        insert2_start_node_id,
        6,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_2 = Edge::create(
        conn,
        insert2_end_node_id,
        4,
        Strand::Forward,
        insert2_end_node_id,
        4,
        Strand::Forward,
    )
    .unwrap();

    let edge_ids = [
        &edge_into_insert2.id,
        &edge_out_of_insert2.id,
        &ref_heal_1.id,
        &ref_heal_2.id,
    ];
    let block_group_edges = edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: *(*edge_id),
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
        })
        .collect::<Vec<BlockGroupEdgeData>>();
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let insert2_path = insert_path
        .new_path_with(conn, 28, 32, &edge_into_insert2, &edge_out_of_insert2)
        .unwrap();
    assert_eq!(
        insert2_path.sequence(conn, None).unwrap(),
        "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG"
    );

    let deletion_end_node_id = intervaltree.query_point(38).next().unwrap().value.node_id;
    let deletion_edge = Edge::create(
        conn,
        insert_node_id,
        8,
        Strand::Forward,
        deletion_end_node_id,
        8,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_1 = Edge::create(
        conn,
        insert_node_id,
        8,
        Strand::Forward,
        insert_node_id,
        8,
        Strand::Forward,
    )
    .unwrap();
    let ref_heal_2 = Edge::create(
        conn,
        deletion_end_node_id,
        8,
        Strand::Forward,
        deletion_end_node_id,
        8,
        Strand::Forward,
    )
    .unwrap();
    let block_group_edges = [
        BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: deletion_edge.id,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: ref_heal_1.id,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group1_id,
            edge_id: ref_heal_2.id,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
        },
    ];
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let all_sequences = get_all_sequences(conn, &block_group1_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTCCCCCCTTTTTTTTGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTAAAAAAAAGG".to_string(), // Sequence including deletion
        ])
    );

    let mut blocks = intervaltree
        .query(Range { start: 15, end: 36 })
        .map(|x| x.value)
        .collect::<Vec<_>>();
    blocks.sort_by_key(|a| a.start);
    let start_block = blocks[0];
    let start_node_coordinate = 15 - start_block.start + start_block.sequence_start;
    let end_block = blocks[blocks.len() - 1];
    let end_node_coordinate = 36 - end_block.start + end_block.sequence_start;

    let block_group2 = create_block_group(conn, "test", "test", "chr1.1");
    derive_subgraph(
        conn,
        &block_group1_id,
        &start_block,
        &end_block,
        start_node_coordinate,
        end_node_coordinate,
        &block_group2.id,
        true,
    )
    .unwrap();
    let all_sequences2 = get_all_sequences(conn, &block_group2.id).unwrap();
    assert_eq!(
        all_sequences2,
        // The deletion is not included in the cloned subgraph since one end of it is
        // outside the specified range
        HashSet::from_iter(vec![
            "TTTTTCCCCCCCCCCGGGGGG".to_string(),
            "TAAAAAAAACCCCCCGGGGGG".to_string(),
            "TTTTTCCCCCCTTTTTTTTGG".to_string(),
            "TAAAAAAAACCTTTTTTTTGG".to_string(),
        ])
    );
}
