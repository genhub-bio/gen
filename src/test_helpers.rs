use std::{fmt::Debug, fs, io::Write, ops::Add, path::PathBuf};

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, config::Workspace,
    errors::ConnectionError,
};
use gen_graph::GenGraph;
use gen_models::{
    block_group::{BlockGroup, BlockGroupError, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::{DbContext, GraphConnection, OperationsConnection},
    edge::Edge,
    file_types::FileTypes,
    migrations::{run_migrations, run_operation_migrations},
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
};

pub fn create_bg(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    name: &str,
) -> BlockGroup {
    BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name,
            name,
            ..Default::default()
        },
    )
}
use intervaltree::IntervalTree;
use rusqlite::Connection;
use tempfile::tempdir;

pub fn get_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<GraphConnection, ConnectionError> {
    let path: Option<&str> = db_path.into();
    let mut conn;
    if let Some(v) = path {
        if fs::metadata(v).is_ok() {
            fs::remove_file(v).expect("Unable to delete existing file");
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
            fs::remove_file(v).expect("Unable to delete existing file");
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
    let graph_conn = get_connection(None).expect("unable to open graph connection");
    let operation_conn =
        get_operation_connection(None).expect("unable to open operations connection");
    DbContext::new(workspace, graph_conn, operation_conn)
}

pub fn setup_gen_on_disk() -> DbContext {
    let tmp_dir = tempdir().unwrap().keep();
    let workspace = Workspace::new(tmp_dir);
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(
        workspace
            .ensure_gen_dir()
            .join("default.db")
            .to_str()
            .unwrap(),
    )
    .expect("unable to open graph connection");
    let operation_conn =
        get_operation_connection(workspace.ensure_gen_dir().join("gen.db").to_str().unwrap())
            .expect("unable to open operations connection");
    DbContext::new(workspace, graph_conn, operation_conn)
}

pub fn setup_block_group(conn: &GraphConnection) -> Result<(HashId, Path), BlockGroupError> {
    let a_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAAAAAAAA")
        .save(conn)
        .unwrap();
    let a_node_id = Node::create(
        conn,
        &a_seq.hash,
        &HashId::convert_str(&format!("test-a-node.{}", a_seq.hash)),
    )?;
    let t_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTTTTTTTTT")
        .save(conn)
        .unwrap();
    let t_node_id = Node::create(
        conn,
        &t_seq.hash,
        &HashId::convert_str(&format!("test-t-node.{}", a_seq.hash)),
    )?;
    let c_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("CCCCCCCCCC")
        .save(conn)
        .unwrap();
    let c_node_id = Node::create(
        conn,
        &c_seq.hash,
        &HashId::convert_str(&format!("test-c-node.{}", a_seq.hash)),
    )?;
    let g_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("GGGGGGGGGG")
        .save(conn)
        .unwrap();
    let g_node_id = Node::create(
        conn,
        &g_seq.hash,
        &HashId::convert_str(&format!("test-g-node.{}", a_seq.hash)),
    )?;
    let _collection = Collection::create(conn, "test");
    Sample::get_or_create(conn, "test").unwrap();
    let block_group = create_bg(conn, "test", "test", "chr1");
    let edge0 = Edge::create(
        conn,
        PATH_START_NODE_ID,
        0,
        Strand::Forward,
        a_node_id,
        0,
        Strand::Forward,
    )?;
    let edge1 = Edge::create(
        conn,
        a_node_id,
        10,
        Strand::Forward,
        t_node_id,
        0,
        Strand::Forward,
    )?;
    let edge2 = Edge::create(
        conn,
        t_node_id,
        10,
        Strand::Forward,
        c_node_id,
        0,
        Strand::Forward,
    )?;
    let edge3 = Edge::create(
        conn,
        c_node_id,
        10,
        Strand::Forward,
        g_node_id,
        0,
        Strand::Forward,
    )?;
    let edge4 = Edge::create(
        conn,
        g_node_id,
        10,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )?;

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
        "chr1",
        &block_group.id,
        &[edge0.id, edge1.id, edge2.id, edge3.id, edge4.id],
    )
    .unwrap();
    Ok((block_group.id, path))
}

pub fn save_graph(graph: &GenGraph, path: &str) {
    use std::fs::File;

    use petgraph::dot::{Config, Dot};
    let mut file = File::create(path).unwrap();
    let _ = file.write_all(
        format!(
            "{dot:?}",
            dot = Dot::with_attr_getters(
                &graph,
                &[Config::NodeNoLabel, Config::EdgeNoLabel],
                &|_, (_, _, edge_weights)| format!(
                    "label = \"{}\"",
                    edge_weights
                        .iter()
                        .map(|ew| ew.chromosome_index.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                &|_, (node, _weight)| format!(
                    "label = \"{}[{}-{}]\"",
                    node.node_id, node.sequence_start, node.sequence_end
                ),
            ),
        )
        .as_bytes(),
    );
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

pub fn get_sample_bg(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
) -> BlockGroup {
    let mut results = Sample::get_block_groups(conn, collection_name, sample_name);
    results.pop().unwrap()
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
            PathBuf::from(file_path)
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

    let mut session = start_operation(context.graph().conn());
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
        hash.into(),
    )
    .unwrap()
}
