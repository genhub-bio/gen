use std::{
    fmt::Debug,
    fs,
    io::Write,
    ops::{Add, Deref, DerefMut},
    rc::Rc,
    sync::OnceLock,
};

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
use tempfile::{TempDir, tempdir};

/// Owns a database context and the temporary workspace created for a test.
///
/// Cloned fixtures share the workspace guard. Raw context or handle clones must not outlive their
/// owning fixture because production database models do not own test directories.
#[derive(Clone)]
pub struct TestContext {
    context: DbContext,
    _temp_dir: Rc<TempDir>,
}

impl TestContext {
    /// Keeps the temporary workspace alive until the final fixture clone drops.
    pub fn new(context: DbContext, temp_dir: TempDir) -> Self {
        Self {
            context,
            _temp_dir: Rc::new(temp_dir),
        }
    }

    /// Borrows the context for runtime structures that store a raw database context.
    ///
    /// Keep this fixture alive as long as any raw context or handle clone is used.
    pub fn db_context(&self) -> &DbContext {
        &self.context
    }
}

impl Deref for TestContext {
    type Target = DbContext;

    fn deref(&self) -> &Self::Target {
        &self.context
    }
}

impl DerefMut for TestContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.context
    }
}

pub fn test_workspace() -> &'static Workspace {
    static WORKSPACE: OnceLock<Workspace> = OnceLock::new();

    WORKSPACE.get_or_init(|| {
        let workspace = Workspace::new(tempdir().expect("should create test directory").keep());
        workspace.ensure_gen_dir();
        workspace
    })
}

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

pub fn setup_gen() -> TestContext {
    let tmp_dir = tempdir().unwrap();
    let workspace = Workspace::new(tmp_dir.path());
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(None).expect("unable to open graph connection");
    let config_conn = get_config_connection(None).expect("unable to open config connection");
    TestContext::new(
        DbContext::new(workspace, graph_conn, config_conn).unwrap(),
        tmp_dir,
    )
}

pub fn setup_gen_on_disk() -> TestContext {
    let tmp_dir = tempdir().unwrap();
    let workspace = Workspace::new(tmp_dir.path());
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(workspace.graph_db_path().unwrap().to_str().unwrap())
        .expect("unable to open graph connection");
    let config_conn =
        get_config_connection(workspace.ensure_gen_dir().join("gen.db").to_str().unwrap())
            .expect("unable to open config connection");
    TestContext::new(
        DbContext::new(workspace, graph_conn, config_conn).unwrap(),
        tmp_dir,
    )
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{TestContext, setup_gen, setup_gen_on_disk};

    fn assert_temp_dir_lives_until_last_context_clone_drop(context: TestContext) {
        let base_dir = context.workspace().base_dir().to_path_buf();
        let nested_asset = context
            .workspace()
            .asset_dir()
            .expect("should resolve workspace asset directory")
            .join("nested/fixture.txt");
        fs::create_dir_all(
            nested_asset
                .parent()
                .expect("should have a parent directory"),
        )
        .expect("should create nested fixture asset directory");
        fs::write(&nested_asset, "temporary workspace asset")
            .expect("should write nested fixture asset");

        let context_clone = context.clone();
        drop(context);
        assert!(
            base_dir.is_dir(),
            "context clone should keep workspace alive"
        );
        assert_eq!(
            fs::read_to_string(&nested_asset).expect("should read nested fixture asset"),
            "temporary workspace asset"
        );

        assert_eq!(
            context_clone
                .graph()
                .conn()
                .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                .expect("should query through cloned context"),
            1
        );
        drop(context_clone);
        assert!(
            !base_dir.exists(),
            "workspace should be removed after its last owner drops"
        );
    }

    #[test]
    fn test_root_setup_gen_temp_dir_cleanup() {
        assert_temp_dir_lives_until_last_context_clone_drop(setup_gen());
    }

    #[test]
    fn test_root_setup_gen_on_disk_temp_dir_cleanup() {
        assert_temp_dir_lives_until_last_context_clone_drop(setup_gen_on_disk());
    }
}
