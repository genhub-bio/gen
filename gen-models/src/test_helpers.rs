use std::{fmt::Debug, fs, ops::Add, path::PathBuf};

use gen_core::{
    config::{get_or_create_gen_dir, BASE_DIR},
    errors::ConnectionError,
    HashId, Strand, PATH_END_NODE_ID, PATH_START_NODE_ID,
};
use intervaltree::IntervalTree;
use rusqlite::Connection;
use tempfile::tempdir;

use crate::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    edge::Edge,
    file_types::FileTypes,
    migrations::{run_migrations, run_operation_migrations},
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
};

pub fn get_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<Connection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to remove database entry.");
        }
        conn = Connection::open(v).map_err(|e| ConnectionError::OpenFailed(e))?;
    } else {
        conn = Connection::open_in_memory().map_err(|e| ConnectionError::OpenFailed(e))?;
    }
    rusqlite::vtab::array::load_module(&conn)?;
    run_migrations(&mut conn);
    Ok(conn)
}

pub fn get_operation_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<Connection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to remove database entry.");
        }
        conn = Connection::open(v).map_err(|e| ConnectionError::OpenFailed(e))?;
    } else {
        conn = Connection::open_in_memory().map_err(|e| ConnectionError::OpenFailed(e))?;
    }
    rusqlite::vtab::array::load_module(&conn)?;
    run_operation_migrations(&mut conn);
    Ok(conn)
}

pub fn setup_gen_dir() -> PathBuf {
    let tmp_dir = tempdir().unwrap().keep();
    {
        BASE_DIR.with(|v| {
            let mut writer = v.write().unwrap();
            *writer = tmp_dir;
        });
    }
    get_or_create_gen_dir()
}

pub fn setup_block_group(conn: &Connection) -> (HashId, Path) {
    let a_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAAAAAAAA")
        .save(conn);
    let a_node_id = Node::create(conn, &a_seq.hash, &HashId::convert_str("test-a-node"));
    let t_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTTTTTTTTT")
        .save(conn);
    let t_node_id = Node::create(conn, &t_seq.hash, &HashId::convert_str("test-t-node"));
    let c_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("CCCCCCCCCC")
        .save(conn);
    let c_node_id = Node::create(conn, &c_seq.hash, &HashId::convert_str("test-c-node"));
    let g_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("GGGGGGGGGG")
        .save(conn);
    let g_node_id = Node::create(conn, &g_seq.hash, &HashId::convert_str("test-g-node"));
    let _collection = Collection::create(conn, "test");
    let block_group = BlockGroup::create(conn, "test", None, "chr1");
    let edge0 = Edge::create(
        conn,
        PATH_START_NODE_ID,
        0,
        Strand::Forward,
        a_node_id,
        0,
        Strand::Forward,
    );
    let edge1 = Edge::create(
        conn,
        a_node_id,
        10,
        Strand::Forward,
        t_node_id,
        0,
        Strand::Forward,
    );
    let edge2 = Edge::create(
        conn,
        t_node_id,
        10,
        Strand::Forward,
        c_node_id,
        0,
        Strand::Forward,
    );
    let edge3 = Edge::create(
        conn,
        c_node_id,
        10,
        Strand::Forward,
        g_node_id,
        0,
        Strand::Forward,
    );
    let edge4 = Edge::create(
        conn,
        g_node_id,
        10,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    );

    let block_group_edges = vec![
        BlockGroupEdgeData {
            block_group_id: block_group.id.clone(),
            edge_id: edge0.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id.clone(),
            edge_id: edge1.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id.clone(),
            edge_id: edge2.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id.clone(),
            edge_id: edge3.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id.clone(),
            edge_id: edge4.id,
            chromosome_index: 0,
            phased: 0,
        },
    ];
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let path = Path::create(
        conn,
        "chr1",
        &block_group.id,
        &[edge0.id, edge1.id, edge2.id, edge3.id, edge4.id],
    );
    (block_group.id, path)
}

pub fn interval_tree_verify<K, V>(tree: &IntervalTree<K, V>, i: K, expected: &[V])
where
    K: Ord + Add<i64, Output = K> + Copy,
    V: Copy + Ord + Debug,
{
    let mut v1: Vec<_> = tree.query_point(i).map(|x| x.value).collect();
    v1.sort();
    let mut v2: Vec<_> = tree.query(i..(i + 1)).map(|x| x.value).collect();
    v2.sort();
    assert_eq!(v1, expected);
    assert_eq!(v2, expected);
}

pub fn create_operation(
    conn: &Connection,
    op_conn: &Connection,
    file_path: &str,
    file_type: FileTypes,
    description: &str,
    hash: impl Into<Option<HashId>>,
) -> Operation {
    let mut session = start_operation(conn);
    end_operation(
        conn,
        op_conn,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: file_path.to_string(),
                file_type,
            }],
            description: description.to_string(),
        },
        "test operation",
        hash,
    )
    .unwrap()
}
