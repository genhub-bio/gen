use std::{fmt::Debug, fs, ops::Add};

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, config::Workspace,
    errors::ConnectionError,
};
use intervaltree::IntervalTree;
use rusqlite::Connection;
use tempfile::tempdir;

use crate::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::{DbContext, GraphConnection, OperationsConnection},
    edge::Edge,
    file_types::FileTypes,
    migrations::{run_migrations, run_operation_migrations},
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::{NewPath, Path},
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
};

pub fn get_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<GraphConnection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to remove database entry.");
        }
        conn = Connection::open(v).map_err(ConnectionError::OpenFailed)?;
    } else {
        conn = Connection::open_in_memory().map_err(ConnectionError::OpenFailed)?;
    }
    rusqlite::vtab::array::load_module(&conn)?;
    run_migrations(&mut conn);
    Ok(GraphConnection(conn))
}

pub fn get_operation_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<OperationsConnection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to remove database entry.");
        }
        conn = Connection::open(v).map_err(ConnectionError::OpenFailed)?;
    } else {
        conn = Connection::open_in_memory().map_err(ConnectionError::OpenFailed)?;
    }
    rusqlite::vtab::array::load_module(&conn)?;
    run_operation_migrations(&mut conn);
    Ok(OperationsConnection(conn))
}

pub fn setup_gen() -> DbContext {
    let tmp_dir = tempdir().unwrap().keep();
    let workspace = Workspace::new(tmp_dir);
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(None).unwrap();
    let operation_conn = get_operation_connection(None).unwrap();
    DbContext::new(workspace, graph_conn, operation_conn)
}

pub fn setup_block_group(conn: &GraphConnection) -> (HashId, Path) {
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
    Sample::get_or_create(conn, "test");
    let block_group = BlockGroup::create(conn, "test", "test", "chr1");
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
            block_group_id: block_group.id,
            edge_id: edge0.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge1.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge2.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge3.id,
            chromosome_index: 0,
            phased: 0,
        },
        BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge4.id,
            chromosome_index: 0,
            phased: 0,
        },
    ];
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let path = Path::create(
        conn,
        NewPath {
            name: "chr1",
            block_group_id: &block_group.id,
            edge_ids: &[edge0.id, edge1.id, edge2.id, edge3.id, edge4.id],
        },
    );
    (block_group.id, path)
}

pub fn create_path(
    conn: &GraphConnection,
    name: &str,
    block_group_id: &HashId,
    edge_ids: &[HashId],
) -> Path {
    Path::create(
        conn,
        NewPath {
            name,
            block_group_id,
            edge_ids,
        },
    )
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
    context: &DbContext,
    file_path: &str,
    file_type: FileTypes,
    description: &str,
    hash: impl Into<Option<HashId>>,
) -> Operation {
    let repo_root = context.repo_root().unwrap();
    if file_type != FileTypes::Changeset && file_type != FileTypes::None {
        let full_path = if std::path::Path::new(file_path).is_absolute() {
            std::path::PathBuf::from(file_path)
        } else {
            repo_root.join(file_path)
        };
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        if !full_path.exists() {
            fs::write(&full_path, b"test file content").unwrap();
        }
    }

    let conn = context.graph().conn();
    let mut session = start_operation(conn);
    end_operation(
        context,
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
