use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::GraphConnection,
    edge::Edge,
    migrations::run_migrations,
    node::Node,
    path::Path,
    sequence::Sequence,
};
use rusqlite::Connection;

pub fn get_connection() -> GraphConnection {
    let mut conn = Connection::open_in_memory().expect("should open in-memory database");
    rusqlite::vtab::array::load_module(&conn).expect("should load rarray module");
    run_migrations(&mut conn);
    GraphConnection(conn)
}

pub fn get_simple_sequence(conn: &GraphConnection) -> HashId {
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
    block_group.id
}
