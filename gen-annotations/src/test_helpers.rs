use std::sync::OnceLock;

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand, Workspace};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::GraphConnection,
    edge::Edge,
    migrations::run_migrations,
    node::Node,
    path::Path,
    region::ResolvedGenRegion,
    sample::Sample,
    sequence::Sequence,
};
use rusqlite::Connection;
use tempfile::tempdir;

pub fn test_workspace() -> &'static Workspace {
    static WORKSPACE: OnceLock<Workspace> = OnceLock::new();

    WORKSPACE.get_or_init(|| {
        let workspace = Workspace::new(tempdir().expect("should create test directory").keep());
        workspace.ensure_gen_dir();
        workspace
    })
}

pub fn get_connection() -> GraphConnection {
    let mut conn = Connection::open_in_memory().expect("should open in-memory database");
    rusqlite::vtab::array::load_module(&conn).expect("should load rarray module");
    run_migrations(&mut conn);
    GraphConnection(conn)
}

pub fn setup_test_data(conn: &GraphConnection) {
    let collection = Collection::create(conn, "test").unwrap();
    Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: Sample::DEFAULT_NAME,
            ..Default::default()
        },
    )
    .unwrap();
    let seq1 = Sequence::new()
        .sequence_type("DNA")
        .sequence("ATCGATCGATCGATCGA")
        .save(conn)
        .unwrap();
    let seq2 = Sequence::new()
        .sequence_type("DNA")
        .sequence("TCGGGAACACACAGAGA")
        .save(conn)
        .unwrap();
    let node1 = Node::create(
        conn,
        &seq1.hash,
        &HashId::convert_str(&format!(
            "{collection}.m123:{hash}",
            collection = collection.name,
            hash = seq1.hash
        )),
    )
    .unwrap();
    let node2 = Node::create(
        conn,
        &seq2.hash,
        &HashId::convert_str(&format!(
            "{collection}.m123:{hash}",
            collection = collection.name,
            hash = seq2.hash
        )),
    )
    .unwrap();
    let block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name: &collection.name,
            sample_name: Sample::DEFAULT_NAME,
            name: "m123",
            parent_block_group_id: None,
            is_default: false,
        },
    )
    .unwrap();

    let edge_into = Edge::create(
        conn,
        PATH_START_NODE_ID,
        0,
        Strand::Forward,
        node1,
        0,
        Strand::Forward,
    )
    .unwrap();
    let middle_edge = Edge::create(
        conn,
        node1,
        seq1.length,
        Strand::Forward,
        node2,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge_out_of = Edge::create(
        conn,
        node2,
        seq2.length,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )
    .unwrap();

    let new_block_group_edges = vec![
        BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge_into.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: middle_edge.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge_out_of.id,
            chromosome_index: 0,
            phased: 0,
        },
    ];

    BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
    Path::create(
        conn,
        "m123",
        &block_group.id,
        &[edge_into.id, middle_edge.id, edge_out_of.id],
    )
    .unwrap();

    Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: "foo",
            ..Default::default()
        },
    )
    .unwrap();
    let _ =
        Sample::get_or_create_child(conn, "test", "foo", vec![Sample::DEFAULT_NAME.to_string()]);

    let sample_bg_id = BlockGroup::get_or_create_sample_block_groups(
        conn,
        "test",
        "foo",
        "m123",
        vec![Sample::DEFAULT_NAME.to_string()],
    )
    .expect("should create child block group")[0]
        .id;
    let sample_path = BlockGroup::get_current_path(conn, &sample_bg_id, None).unwrap();
    let alt_seq = "C";

    let sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence(alt_seq)
        .save(conn)
        .unwrap();
    let node_id = Node::create(
        conn,
        &sequence.hash,
        &HashId::convert_str(&format!(
            "{path_id}:3-4->{sequence_hash}",
            path_id = sample_path.id,
            sequence_hash = sequence.hash,
        )),
    )
    .unwrap();
    let region = ResolvedGenRegion::from_path(conn, sample_bg_id, &sample_path, 3, 4).unwrap();
    let change = BlockGroupChange {
        region,
        path_accession: None,
        block: PathBlock {
            node_id,
            block_sequence: alt_seq.to_string(),
            sequence_start: 0,
            sequence_end: alt_seq.len() as i64,
            path_start: 3,
            path_end: 4,
            strand: Strand::Forward,
        },
        chromosome_index: 0,
        phased: 0,
        preserve_edge: false,
    };

    BlockGroup::insert_change(conn, test_workspace(), &change)
        .expect("should apply variant change");
}

// Shared two-node fixture: both paths traverse B then A with the shown edge endpoint strands.
// Positive stored ranges select bases first. The edge strand shown is the source/target endpoint
// orientation for that traversal.
//
// Stored node A:
//   index: 012345678
//   DNA:   AACCGGTTA
//   span:  ^--------^ [0,9)
//
// Stored node B:
//   index: 012345678
//   DNA:   CCGTTGCAT
//   span:  ^-------^ [0,8); stored index 8 (T) is excluded.
//
// Case 1: path edges traverse B then A; edge endpoints at B and A have strand +.
//   path index:    01234567890123456
//   path nodes:    BBBBBBBBAAAAAAAAA
//   path spans:    ^-------^--------^
//   path DNA:      CCGTTGCAAACCGGTTA
//
// Case 2: path edges traverse B then A; edge endpoints at B and A have strand -. Path DNA
// shown is negative strand. It is stored as positive strand
//   path index:    01234567890123456
//   path nodes:    BBBBBBBBAAAAAAAAA
//   path spans:    ^-------^--------^
//   path DNA:      TGCAACGGTAACCGGTT
pub(crate) fn create_two_node_test_block_group(
    conn: &gen_models::db::GraphConnection,
    name: &str,
    strand: Strand,
) -> (BlockGroup, HashId, HashId) {
    let block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            name,
            parent_block_group_id: None,
            is_default: false,
        },
    )
    .expect("should create test block group");
    let sequence_a_text = "AACCGGTTA";
    let sequence_b_text = "CCGTTGCAT";
    let sequence_a = Sequence::new()
        .sequence_type("DNA")
        .sequence(sequence_a_text)
        .save(conn)
        .expect("should save first path sequence");
    let node_a_id = HashId::convert_str(&format!("{name}-path-node-a"));
    Node::create(conn, &sequence_a.hash, &node_a_id).expect("should create first path node");
    let sequence_b = Sequence::new()
        .sequence_type("DNA")
        .sequence(sequence_b_text)
        .save(conn)
        .expect("should save second path sequence");
    let node_b_id = HashId::convert_str(&format!("{name}-path-node-b"));
    Node::create(conn, &sequence_b.hash, &node_b_id).expect("should create second path node");

    let edge_start = Edge::create(
        conn,
        PATH_START_NODE_ID,
        0,
        Strand::Forward,
        node_b_id,
        0,
        strand,
    )
    .expect("should create path start edge");
    let edge_between = Edge::create(conn, node_b_id, 8, strand, node_a_id, 0, strand)
        .expect("should connect path nodes");
    let edge_end = Edge::create(
        conn,
        node_a_id,
        9,
        strand,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )
    .expect("should create path end edge");
    let edge_ids = [edge_start.id, edge_between.id, edge_end.id];
    BlockGroupEdge::bulk_create(
        conn,
        &edge_ids
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>(),
    );
    let path =
        Path::create(conn, name, &block_group.id, &edge_ids).expect("should create test path");
    let sequences_by_node_id = Node::get_sequences_by_node_ids(
        conn,
        crate::test_helpers::test_workspace(),
        &[node_a_id, node_b_id],
        None,
    );
    assert_eq!(
        sequences_by_node_id[&node_a_id]
            .get_sequence(None, None)
            .expect("should fetch stored sequence for first node"),
        sequence_a_text,
        "node DNA should stay in its stored 5′-to-3′ orientation"
    );
    assert_eq!(
        sequences_by_node_id[&node_b_id]
            .get_sequence(None, None)
            .expect("should fetch stored sequence for second node"),
        sequence_b_text,
        "node DNA should stay in its stored 5′-to-3′ orientation"
    );
    let path_sequence = path
        .sequence(conn, crate::test_helpers::test_workspace(), None)
        .expect("should load test path sequence");
    assert_eq!(
        path_sequence,
        if strand == Strand::Reverse {
            "TGCAACGGTAACCGGTT"
        } else {
            "CCGTTGCAAACCGGTTA"
        },
        "path retrieval should orient node DNA from the path strand"
    );
    assert_eq!(
        &path_sequence[7..9],
        if strand == Strand::Reverse {
            "GT"
        } else {
            "AA"
        },
        "the split-node feature fixture should spell its strand-specific path bases"
    );

    (block_group, node_a_id, node_b_id)
}

pub(crate) fn create_reverse_test_block_group(
    conn: &gen_models::db::GraphConnection,
) -> (BlockGroup, HashId, HashId) {
    create_two_node_test_block_group(conn, "reverse", Strand::Reverse)
}

pub(crate) fn create_forward_test_block_group(
    conn: &gen_models::db::GraphConnection,
) -> (BlockGroup, HashId, HashId) {
    create_two_node_test_block_group(conn, "forward", Strand::Forward)
}
