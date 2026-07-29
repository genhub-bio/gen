use std::{fmt::Debug, fs, io::Write, ops::Add};

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, config::Workspace,
    errors::ConnectionError,
};
use gen_graph::GenGraph;
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::{ConfigConnection, DbContext, GraphConnection},
    edge::Edge,
    migrations::{run_config_migrations, run_migrations},
    node::Node,
    path::Path,
    sample::Sample,
    sequence::Sequence,
};

pub fn create_bg(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    name: &str,
) -> BlockGroup {
    Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample_name,
            ..Default::default()
        },
    )
    .unwrap();
    BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name,
            name,
            ..Default::default()
        },
    )
    .unwrap()
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

pub fn get_config_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<ConfigConnection, ConnectionError> {
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
    run_config_migrations(&mut conn);
    Ok(ConfigConnection(conn))
}

pub fn setup_gen() -> DbContext {
    let tmp_dir = tempdir().unwrap().keep();
    let workspace = Workspace::new(tmp_dir);
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(None).expect("unable to open graph connection");
    let config_conn = get_config_connection(None).expect("unable to open config connection");
    DbContext::new(workspace, graph_conn, config_conn).unwrap()
}

pub fn setup_gen_on_disk() -> DbContext {
    let tmp_dir = tempdir().unwrap().keep();
    let workspace = Workspace::new(tmp_dir);
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(workspace.graph_db_path().unwrap().to_str().unwrap())
        .expect("unable to open graph connection");
    let config_conn =
        get_config_connection(workspace.ensure_gen_dir().join("gen.db").to_str().unwrap())
            .expect("unable to open config connection");
    DbContext::new(workspace, graph_conn, config_conn).unwrap()
}

pub fn setup_block_group(conn: &GraphConnection) -> (HashId, Path) {
    let a_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAAAAAAAA")
        .save(conn)
        .unwrap();
    let a_node_id = Node::create(
        conn,
        &a_seq.hash,
        &HashId::convert_str(&format!("test-a-node.{}", a_seq.hash)),
    )
    .unwrap();
    let t_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTTTTTTTTT")
        .save(conn)
        .unwrap();
    let t_node_id = Node::create(
        conn,
        &t_seq.hash,
        &HashId::convert_str(&format!("test-t-node.{}", a_seq.hash)),
    )
    .unwrap();
    let c_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("CCCCCCCCCC")
        .save(conn)
        .unwrap();
    let c_node_id = Node::create(
        conn,
        &c_seq.hash,
        &HashId::convert_str(&format!("test-c-node.{}", a_seq.hash)),
    )
    .unwrap();
    let g_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("GGGGGGGGGG")
        .save(conn)
        .unwrap();
    let g_node_id = Node::create(
        conn,
        &g_seq.hash,
        &HashId::convert_str(&format!("test-g-node.{}", a_seq.hash)),
    )
    .unwrap();
    let _collection = Collection::create(conn, "test");
    Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: "test",
            ..Default::default()
        },
    )
    .unwrap();
    let block_group = create_bg(conn, "test", "test", "chr1");
    let edge0 = Edge::create(
        conn,
        PATH_START_NODE_ID,
        0,
        Strand::Forward,
        a_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge1 = Edge::create(
        conn,
        a_node_id,
        10,
        Strand::Forward,
        t_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge2 = Edge::create(
        conn,
        t_node_id,
        10,
        Strand::Forward,
        c_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge3 = Edge::create(
        conn,
        c_node_id,
        10,
        Strand::Forward,
        g_node_id,
        0,
        Strand::Forward,
    )
    .unwrap();
    let edge4 = Edge::create(
        conn,
        g_node_id,
        10,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )
    .unwrap();

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
    (block_group.id, path)
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
    let mut results = Sample::get_block_groups(conn, collection_name, sample_name, None);
    results.pop().unwrap()
}
