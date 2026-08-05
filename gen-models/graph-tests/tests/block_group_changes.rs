use std::collections::HashSet;

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    accession::Accession,
    annotations::Annotation as ModelAnnotation,
    block_group::{BlockGroup, BlockGroupChange, PathCache},
    node::Node,
    region::ResolvedGenRegion,
    sequence::Sequence,
    traits::Query as _,
};
use gen_models_graph_tests::{get_all_sequences, get_connection, setup_block_group};

#[test]
fn test_insert_accession_change_get_all() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let mut path_cache = PathCache::new(&conn);
    let accession =
        BlockGroup::add_accession(&conn, &path, "test-accession", 10, 30, &mut path_cache).unwrap();
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id = Node::create(
        &conn,
        &insert_sequence.hash,
        &HashId::convert_str("acc-insert-node"),
    )
    .unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 5,
        path_end: 15,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_accession(&conn, &accession, 5, 15).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };

    gen_graph::models::insert_change(&conn, &change).unwrap();
    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTNNNNCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_insert_annotation_change_get_all() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let mut path_cache = PathCache::new(&conn);
    let accession =
        BlockGroup::add_accession(&conn, &path, "test-accession", 10, 30, &mut path_cache).unwrap();
    let annotation =
        ModelAnnotation::get_or_create(&conn, "gene-1", "track-1", &accession.id, None).unwrap();
    let deletion_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .unwrap();
    let deletion_node_id = Node::create(
        &conn,
        &deletion_sequence.hash,
        &HashId::convert_str("annotation-delete-node"),
    )
    .unwrap();
    let deletion = PathBlock {
        node_id: deletion_node_id,
        block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 5,
        path_end: 15,
        strand: Strand::Forward,
    };
    let annotation_accession = Accession::get_by_id(&conn, &annotation.accession_id, None).unwrap();
    let region =
        ResolvedGenRegion::from_annotation(&conn, &annotation, &annotation_accession, 5, 15)
            .unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: deletion,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };

    gen_graph::models::insert_change(&conn, &change).unwrap();
    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_simple_insert_get_all() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 7,
        path_end: 15,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_on_block_boundary_middle() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 15,
        path_end: 15,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 15, 15).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTNNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_within_block() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 12,
        path_end: 17,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 12, 17).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTNNNNTTTCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_on_block_boundary_start() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 10,
        path_end: 10,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 10).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAANNNNTTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_on_block_boundary_end() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 9,
        path_end: 9,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 9, 9).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAANNNNATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_across_entire_block_boundary() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 10,
        path_end: 20,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 20).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAANNNNCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_across_two_blocks() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 15,
        path_end: 25,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 15, 25).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTNNNNCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_spanning_blocks() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 5,
        path_end: 35,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 5, 35).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAANNNNGGGGG".to_string()
        ])
    );
}

#[test]
fn test_simple_deletion() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let deletion_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .unwrap();
    let deletion_node_id =
        Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
    let deletion = PathBlock {
        node_id: deletion_node_id,
        block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 19,
        path_end: 31,
        strand: Strand::Forward,
    };

    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 19, 31).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: deletion,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };

    // take out an entire block
    gen_graph::models::insert_change(&conn, &change).unwrap();
    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_doesnt_apply_same_insert_twice() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 7,
        path_end: 15,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
fn test_insert_at_beginning_of_path() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 0,
        path_end: 0,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 0, 0).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "NNNNAAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_homozygous_insert_at_beginning_of_path() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 0,
        path_end: 0,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 0, 0).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 0,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "NNNNAAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_insert_at_end_of_path() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);

    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 40,
        path_end: 40,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 40, 40).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGGNNNN".to_string(),
        ])
    );
}

#[test]
fn test_insert_at_one_bp_into_block() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 10,
        path_end: 11,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 11).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_insert_at_one_bp_from_end_of_block() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(&conn)
        .unwrap();
    let insert_node_id =
        Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 19,
        path_end: 20,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 19, 20).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTNNNNCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_delete_at_beginning_of_path() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let deletion_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .unwrap();
    let deletion_node_id =
        Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
    let deletion = PathBlock {
        node_id: deletion_node_id,
        block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 0,
        path_end: 1,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 0, 1).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: deletion,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_delete_at_end_of_path() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let deletion_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .unwrap();
    let deletion_node_id =
        Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
    let deletion = PathBlock {
        node_id: deletion_node_id,
        block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 35,
        path_end: 40,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 35, 40).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: deletion,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_deletion_starting_at_block_boundary() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let deletion_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .unwrap();
    let deletion_node_id =
        Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
    let deletion = PathBlock {
        node_id: deletion_node_id,
        block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 10,
        path_end: 12,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 12).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: deletion,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}

#[test]
fn test_deletion_ending_at_block_boundary() {
    let conn = get_connection(None).unwrap();
    let (block_group_id, path) = setup_block_group(&conn);
    let deletion_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("")
        .save(&conn)
        .unwrap();
    let deletion_node_id =
        Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
    let deletion = PathBlock {
        node_id: deletion_node_id,
        block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
        sequence_start: 0,
        sequence_end: 0,
        path_start: 18,
        path_end: 20,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 18, 20).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: deletion,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(&conn, &change).unwrap();

    let all_sequences = get_all_sequences(&conn, &block_group_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );
}
