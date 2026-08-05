use std::collections::{HashMap, HashSet};

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    edge::Edge,
    node::Node,
    path::Path,
    region::{ResolvedGenRegion, ResolvedRegionKind},
    sample::{NewSample, Sample},
    sequence::Sequence,
    traits::Query as _,
};
use gen_models_graph_tests::{
    create_block_group, get_all_sequences, get_connection, get_sample_all_sequences,
    get_single_block_group_id, setup_block_group,
};
use rusqlite::{params, types::Value as SQLValue};

#[test]
fn test_blockgroup_copies_immediate_parent_block_groups() {
    let conn = &get_connection(None).unwrap();
    Collection::create(conn, "test").unwrap();
    Sample::get_or_create(
        conn,
        NewSample {
            name: "parent_a",
            ..Default::default()
        },
    )
    .unwrap();
    Sample::get_or_create(
        conn,
        NewSample {
            name: "parent_b",
            ..Default::default()
        },
    )
    .unwrap();
    Sample::get_or_create(
        conn,
        NewSample {
            name: "child",
            ..Default::default()
        },
    )
    .unwrap();

    let parent_a_bg = create_block_group(conn, "test", "parent_a", "chr1");
    let parent_b_bg = create_block_group(conn, "test", "parent_b", "chr1");

    let seq_a = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAA")
        .save(conn)
        .unwrap();
    let seq_b = Sequence::new()
        .sequence_type("DNA")
        .sequence("CCCC")
        .save(conn)
        .unwrap();
    let node_a = Node::create(conn, &seq_a.hash, &HashId::convert_str("merge-parent-a")).unwrap();
    let node_b = Node::create(conn, &seq_b.hash, &HashId::convert_str("merge-parent-b")).unwrap();

    let parent_a_edges = [
        Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_a,
            0,
            Strand::Forward,
        )
        .unwrap(),
        Edge::create(
            conn,
            node_a,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap(),
    ];
    let parent_b_edges = [
        Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_b,
            0,
            Strand::Forward,
        )
        .unwrap(),
        Edge::create(
            conn,
            node_b,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap(),
    ];

    BlockGroupEdge::bulk_create(
        conn,
        &parent_a_edges
            .iter()
            .map(|edge| BlockGroupEdgeData {
                block_group_id: parent_a_bg.id,
                edge_id: edge.id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>(),
    );
    BlockGroupEdge::bulk_create(
        conn,
        &parent_b_edges
            .iter()
            .map(|edge| BlockGroupEdgeData {
                block_group_id: parent_b_bg.id,
                edge_id: edge.id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>(),
    );

    let child_block_groups = BlockGroup::get_or_create_sample_block_groups(
        conn,
        "test",
        "child",
        "chr1",
        vec!["parent_a".to_string(), "parent_b".to_string()],
    )
    .unwrap();
    assert_eq!(child_block_groups.len(), 2);

    let child_by_parent = child_block_groups
        .iter()
        .map(|block_group| (block_group.parent_block_group_id.unwrap(), block_group))
        .collect::<HashMap<_, _>>();

    let child_a = child_by_parent.get(&parent_a_bg.id).unwrap();
    let child_b = child_by_parent.get(&parent_b_bg.id).unwrap();

    let child_a_edges = BlockGroupEdge::query(
        conn,
        "select * from block_group_edges where block_group_id = ?1",
        params![child_a.id],
    );
    let child_b_edges = BlockGroupEdge::query(
        conn,
        "select * from block_group_edges where block_group_id = ?1",
        params![child_b.id],
    );
    assert_eq!(
        child_a_edges
            .iter()
            .map(|edge| edge.edge_id)
            .collect::<HashSet<_>>(),
        parent_a_edges
            .iter()
            .map(|edge| edge.id)
            .collect::<HashSet<_>>()
    );
    assert_eq!(
        child_b_edges
            .iter()
            .map(|edge| edge.edge_id)
            .collect::<HashSet<_>>(),
        parent_b_edges
            .iter()
            .map(|edge| edge.id)
            .collect::<HashSet<_>>()
    );

    assert_eq!(
        get_all_sequences(conn, &child_a.id).unwrap(),
        HashSet::from_iter(vec!["AAAA".to_string()])
    );
    assert_eq!(
        get_all_sequences(conn, &child_b.id).unwrap(),
        HashSet::from_iter(vec!["CCCC".to_string()])
    );
    assert_eq!(
        get_sample_all_sequences(conn, "test", "child", None).unwrap(),
        HashSet::from_iter(vec!["AAAA".to_string(), "CCCC".to_string()])
    );
}

#[test]
fn test_changes_against_derivative_blockgroups() {
    let conn = &get_connection(None).unwrap();
    let (_block_group_id, _path) = setup_block_group(conn);
    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
            name: "child",
            ..Default::default()
        },
    )
    .unwrap();
    let new_bg_id =
        get_single_block_group_id(conn, "test", "child", "chr1", vec!["test".to_string()]);
    let new_path = Path::query(
        conn,
        "select * from paths where block_group_id = ?1",
        rusqlite::params!(SQLValue::from(new_bg_id)),
    );
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(conn)
        .unwrap();
    let insert_node_id =
        Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 7,
        path_end: 15,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion::from_path(conn, new_bg_id, &new_path[0], 7, 15).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: false,
    };

    // note we are making our change against the new blockgroup, and not the parent blockgroup
    gen_graph::models::insert_change(conn, &change).unwrap();
    let all_sequences = get_all_sequences(conn, &new_bg_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec!["AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),])
    );

    // Now, we make a change against another descendant
    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
            name: "grandchild",
            ..Default::default()
        },
    )
    .unwrap();
    let gc_bg_id = get_single_block_group_id(
        conn,
        "test",
        "grandchild",
        "chr1",
        vec!["child".to_string()],
    );
    let _new_path = Path::query(
        conn,
        "select * from paths where block_group_id = ?1",
        rusqlite::params!(SQLValue::from(gc_bg_id)),
    );

    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 7,
        path_end: 15,
        strand: Strand::Forward,
    };
    let gc_bg = BlockGroup::get_by_id(conn, &gc_bg_id, None).unwrap();
    let gc_region = ResolvedGenRegion {
        block_group: gc_bg,
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
    };
    let change = BlockGroupChange {
        region: gc_region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: false,
    };
    gen_graph::models::insert_change(conn, &change).unwrap();
    let all_sequences = get_all_sequences(conn, &gc_bg_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec!["AAAAAAANNNNTCCCCCCCCCCGGGGGGGGGG".to_string(),])
    );
}

#[test]
fn test_changes_against_derivative_diploid_blockgroups() {
    // This test ensures that if we have heterozygous changes that do not introduce frameshifts,
    // we can modify regions downstream of them.
    let conn = &get_connection(None).unwrap();
    let (_block_group_id, _path) = setup_block_group(conn);
    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
            name: "child",
            ..Default::default()
        },
    )
    .unwrap();
    let new_bg_id =
        get_single_block_group_id(conn, "test", "child", "chr1", vec!["test".to_string()]);
    let _new_path = Path::query(
        conn,
        "select * from paths where block_group_id = ?1",
        params![new_bg_id],
    );
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(conn)
        .unwrap();
    let insert_node_id =
        Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 7,
        path_end: 11,
        strand: Strand::Forward,
    };
    let bg = BlockGroup::get_by_id(conn, &new_bg_id, None).unwrap();
    let region = ResolvedGenRegion {
        block_group: bg,
        path: None,
        accession: None,
        annotation: None,
        kind: ResolvedRegionKind::BlockGroup,
        anchor_start: 0,
        anchor_end: 0,
        feature_length: 0,
        start: 7,
        end: 11,
        start_anchors: None,
        end_anchors: None,
        remove_ambiguous_positions: true,
    };
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(conn, &change).unwrap();
    let all_sequences = get_all_sequences(conn, &new_bg_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );

    // Now, we make a change against another descendant
    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
            name: "grandchild",
            ..Default::default()
        },
    )
    .unwrap();
    let gc_bg_id = get_single_block_group_id(
        conn,
        "test",
        "grandchild",
        "chr1",
        vec!["child".to_string()],
    );
    let _new_path = Path::query(
        conn,
        "select * from paths where block_group_id = ?1",
        params![gc_bg_id],
    );

    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(conn)
        .unwrap();
    let insert_node_id = Node::create(
        conn,
        &insert_sequence.hash,
        &HashId::convert_str("new-hash"),
    )
    .unwrap();

    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 20,
        path_end: 24,
        strand: Strand::Forward,
    };
    let gc_bg = BlockGroup::get_by_id(conn, &gc_bg_id, None).unwrap();
    let gc_region = ResolvedGenRegion {
        block_group: gc_bg,
        path: None,
        accession: None,
        annotation: None,
        kind: ResolvedRegionKind::BlockGroup,
        anchor_start: 0,
        anchor_end: 0,
        feature_length: 0,
        start: 20,
        end: 24,
        start_anchors: None,
        end_anchors: None,
        remove_ambiguous_positions: true,
    };
    let change = BlockGroupChange {
        region: gc_region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(conn, &change).unwrap();
    let all_sequences = get_all_sequences(conn, &gc_bg_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTNNNNCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAANNNNTTTTTTTTTNNNNCCCCCCGGGGGGGGGG".to_string()
        ])
    );
}

#[test]
#[should_panic]
fn test_prohibits_out_of_frame_changes_against_derivative_diploid_blockgroups() {
    // This test ensures that we do not allow ambiguous changes by coordinates
    let conn = &get_connection(None).unwrap();
    let (_block_group_id, _path) = setup_block_group(conn);
    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
            name: "child",
            ..Default::default()
        },
    )
    .unwrap();
    let new_bg_id =
        get_single_block_group_id(conn, "test", "child", "chr1", vec!["test".to_string()]);
    let _new_path = Path::query(
        conn,
        "select * from paths where block_group_id = ?1",
        rusqlite::params!(SQLValue::from(new_bg_id)),
    );
    // This is a heterozygous replacement of 5 bases with 4 bases, so positions
    // downstream of this are not addressable.
    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(conn)
        .unwrap();
    let insert_node_id =
        Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 7,
        path_end: 12,
        strand: Strand::Forward,
    };
    let bg = BlockGroup::get_by_id(conn, &new_bg_id, None).unwrap();
    let region = ResolvedGenRegion {
        block_group: bg,
        path: None,
        accession: None,
        annotation: None,
        kind: ResolvedRegionKind::BlockGroup,
        anchor_start: 0,
        anchor_end: 0,
        feature_length: 0,
        start: 7,
        end: 12,
        start_anchors: None,
        end_anchors: None,
        remove_ambiguous_positions: true,
    };
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };

    // note we are making our change against the new blockgroup, and not the parent blockgroup
    gen_graph::models::insert_change(conn, &change).unwrap();
    let all_sequences = get_all_sequences(conn, &new_bg_id).unwrap();
    assert_eq!(
        all_sequences,
        HashSet::from_iter(vec![
            "AAAAAAANNNNTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
        ])
    );

    // Now, we make a change against another descendant and get an error
    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
            name: "grandchild",
            ..Default::default()
        },
    )
    .unwrap();
    let gc_bg_id = get_single_block_group_id(
        conn,
        "test",
        "grandchild",
        "chr1",
        vec!["child".to_string()],
    );
    let _new_path = Path::query(
        conn,
        "select * from paths where block_group_id = ?1",
        rusqlite::params!(SQLValue::from(gc_bg_id)),
    );

    let insert_sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence("NNNN")
        .save(conn)
        .unwrap();
    let insert_node_id =
        Node::create(conn, &insert_sequence.hash, &HashId::pad_str("new-hash")).unwrap();

    let insert = PathBlock {
        node_id: insert_node_id,
        block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
        sequence_start: 0,
        sequence_end: 4,
        path_start: 20,
        path_end: 24,
        strand: Strand::Forward,
    };
    let gc_bg = BlockGroup::get_by_id(conn, &gc_bg_id, None).unwrap();
    let gc_region = ResolvedGenRegion {
        block_group: gc_bg,
        path: None,
        accession: None,
        annotation: None,
        kind: ResolvedRegionKind::BlockGroup,
        anchor_start: 0,
        anchor_end: 0,
        feature_length: 0,
        start: 20,
        end: 24,
        start_anchors: None,
        end_anchors: None,
        remove_ambiguous_positions: true,
    };
    let change = BlockGroupChange {
        region: gc_region,
        path_accession: None,
        block: insert,
        chromosome_index: 1,
        phased: 0,
        preserve_edge: true,
    };
    gen_graph::models::insert_change(conn, &change).unwrap();
}
