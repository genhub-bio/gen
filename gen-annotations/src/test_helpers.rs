use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, PathChange},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::GraphConnection,
    edge::Edge,
    migrations::run_migrations,
    node::Node,
    path::Path,
    sample::Sample,
    sequence::Sequence,
};
use rusqlite::Connection;

pub fn get_connection() -> GraphConnection {
    let mut conn = Connection::open_in_memory().expect("should open in-memory database");
    rusqlite::vtab::array::load_module(&conn).expect("should load rarray module");
    run_migrations(&mut conn);
    GraphConnection(conn)
}

pub fn setup_test_data(conn: &GraphConnection) {
    let collection = Collection::create(conn, "test");
    let seq1 = Sequence::new()
        .sequence_type("DNA")
        .sequence("ATCGATCGATCGATCGA")
        .save(conn);
    let seq2 = Sequence::new()
        .sequence_type("DNA")
        .sequence("TCGGGAACACACAGAGA")
        .save(conn);
    let node1 = Node::create(
        conn,
        &seq1.hash,
        &HashId::convert_str(&format!(
            "{collection}.m123:{hash}",
            collection = collection.name,
            hash = seq1.hash
        )),
    );
    let node2 = Node::create(
        conn,
        &seq2.hash,
        &HashId::convert_str(&format!(
            "{collection}.m123:{hash}",
            collection = collection.name,
            hash = seq2.hash
        )),
    );
    let block_group = BlockGroup::create(conn, &collection.name, None, "m123");

    let edge_into = Edge::create(
        conn,
        PATH_START_NODE_ID,
        0,
        Strand::Forward,
        node1,
        0,
        Strand::Forward,
    );
    let middle_edge = Edge::create(
        conn,
        node1,
        seq1.length,
        Strand::Forward,
        node2,
        0,
        Strand::Forward,
    );
    let edge_out_of = Edge::create(
        conn,
        node2,
        seq2.length,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    );

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
    );

    Sample::get_or_create(conn, "foo");
    let _ = Sample::get_or_create_child(conn, "test", "foo", None);

    let sample_bg_id =
        BlockGroup::get_or_create_sample_block_group(conn, "test", "foo", "m123", None)
            .expect("should create child block group");
    let sample_path = BlockGroup::get_current_path(conn, &sample_bg_id);
    let tree = sample_path.intervaltree(conn);

    let alt_seq = "C";

    let sequence = Sequence::new()
        .sequence_type("DNA")
        .sequence(alt_seq)
        .save(conn);
    let node_id = Node::create(
        conn,
        &sequence.hash,
        &HashId::convert_str(&format!(
            "{path_id}:3-4->{sequence_hash}",
            path_id = sample_path.id,
            sequence_hash = sequence.hash,
        )),
    );
    let change = PathChange {
        block_group_id: sample_bg_id,
        path: sample_path,
        path_accession: None,
        start: 3,
        end: 4,
        block: PathBlock {
            id: 0,
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

    BlockGroup::insert_change(conn, &change, &tree)
        .expect("should apply variant change from simple.vcf");
}
