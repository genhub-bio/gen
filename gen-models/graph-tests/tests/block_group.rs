use std::collections::HashSet;

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    block_group::BlockGroupChange, node::Node, region::ResolvedGenRegion, sequence::Sequence,
};
use gen_models_graph_tests::{get_all_sequences, get_connection, setup_block_group};

#[test]
fn test_insert_and_deletion_sequences() {
    let conn = get_connection(None).expect("should create an in-memory graph database");
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .expect("should save the inserted sequence");
    let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"))
        .expect("should create the inserted node");
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence
            .get_sequence(0, 4)
            .expect("should load the inserted sequence"),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 7,
        path_end: 15,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15)
        .expect("should resolve the insertion region");
    gen_graph::models::insert_change(
        &conn,
        &BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        },
    )
    .expect("should insert the sequence change");

    assert_eq!(
        get_all_sequences(&conn, &block_group_id)
            .expect("should enumerate sequences after insertion"),
        HashSet::from([
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );

    let deletion_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .expect("should save the deletion sequence");
    let deletion_node_id = Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("2"))
        .expect("should create the deletion node");
    let deletion = PathBlock {
        node_id: deletion_node_id,
        block_sequence: deletion_sequence
            .get_sequence(None, None)
            .expect("should load the deletion sequence"),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 19,
        path_end: 31,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 19, 31)
        .expect("should resolve the deletion region");
    gen_graph::models::insert_change(
        &conn,
        &BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        },
    )
    .expect("should insert the deletion change");

    assert_eq!(
        get_all_sequences(&conn, &block_group_id)
            .expect("should enumerate sequences after deletion"),
        HashSet::from([
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTGGGGGGGGG".to_string(),
            "AAAAAAANNNNTTTTGGGGGGGGG".to_string(),
        ])
    );
}
