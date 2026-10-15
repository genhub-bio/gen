use std::{fmt::Debug, fs, ops::Add, sync::OnceLock};

use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, config::Workspace,
    errors::ConnectionError,
};
use intervaltree::IntervalTree;
use rusqlite::Connection;
use tempfile::tempdir;

use crate::{
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::{ConfigConnection, DbContext, GraphConnection},
    edge::Edge,
    migrations::{run_config_migrations, run_migrations},
    node::Node,
    path::Path,
    sample::{NewSample, Sample},
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
        NewSample {
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

pub fn get_config_connection<'a>(
    db_path: impl Into<Option<&'a str>>,
) -> Result<ConfigConnection, ConnectionError> {
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
    run_config_migrations(&mut conn);
    Ok(ConfigConnection(conn))
}

pub fn setup_gen() -> DbContext {
    let tmp_dir = tempdir().unwrap();
    let workspace = Workspace::new(tmp_dir.path());
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(None).unwrap();
    let config_conn = get_config_connection(None).unwrap();
    DbContext::new(workspace, graph_conn, config_conn)
        .unwrap()
        .with_temp_dir(tmp_dir)
}

pub fn setup_gen_on_disk() -> DbContext {
    let tmp_dir = tempdir().unwrap();
    let workspace = Workspace::new(tmp_dir.path());
    workspace.ensure_gen_dir();
    let graph_conn = get_connection(workspace.graph_db_path().unwrap().to_str().unwrap()).unwrap();
    let config_conn = get_config_connection(workspace.gen_db_path().unwrap().to_str()).unwrap();
    DbContext::new(workspace, graph_conn, config_conn)
        .unwrap()
        .with_temp_dir(tmp_dir)
}

pub fn test_workspace() -> &'static Workspace {
    static WORKSPACE: OnceLock<Workspace> = OnceLock::new();

    WORKSPACE.get_or_init(|| {
        let workspace = Workspace::new(tempdir().unwrap().keep());
        workspace.ensure_gen_dir();
        workspace
    })
}

pub fn setup_block_group(conn: &GraphConnection) -> (HashId, Path) {
    let a_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAAAAAAAAA")
        .save(conn)
        .unwrap();
    let a_node_id = Node::create(conn, &a_seq.hash, &HashId::convert_str("test-a-node")).unwrap();
    let t_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTTTTTTTTT")
        .save(conn)
        .unwrap();
    let t_node_id = Node::create(conn, &t_seq.hash, &HashId::convert_str("test-t-node")).unwrap();
    let c_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("CCCCCCCCCC")
        .save(conn)
        .unwrap();
    let c_node_id = Node::create(conn, &c_seq.hash, &HashId::convert_str("test-c-node")).unwrap();
    let g_seq = Sequence::new()
        .sequence_type("DNA")
        .sequence("GGGGGGGGGG")
        .save(conn)
        .unwrap();
    let g_node_id = Node::create(conn, &g_seq.hash, &HashId::convert_str("test-g-node")).unwrap();
    Collection::get_or_create(conn, "test").unwrap();
    Sample::get_or_create(
        conn,
        NewSample {
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

pub fn interval_tree_verify<K, V>(tree: &IntervalTree<K, V>, i: K, expected: &[V])
where
    K: Ord + Add<i64, Output = K> + Copy,
    V: Copy + Ord + Debug,
{
    let mut v1: Vec<_> = tree.query_point(i).map(|x| x.value).collect();
    v1.sort();
    let mut v2: Vec<_> = tree.query(i..(i + 1)).map(|x| x.value).collect();
    v2.sort();
    let mut expected = expected.to_vec();
    expected.sort();
    assert_eq!(v1, expected);
    assert_eq!(v2, expected);
}

pub fn run_query(conn: &Connection, query: &str) {
    let mut stmt = conn.prepare(query).unwrap();
    for _ in stmt
        .query_map([], |row| {
            dbg!(&row);
            Ok(())
        })
        .unwrap()
    {}
}

#[cfg(test)]
mod tests {
    use std::{fs, panic::AssertUnwindSafe};

    use super::{setup_gen, setup_gen_on_disk};
    use crate::db::DbContext;

    fn assert_temp_dir_lives_until_last_context_or_handle_drop(context: DbContext) {
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
        let graph_handle = context.graph().clone();
        let config_handle = context.config().clone();
        drop(context);
        assert!(
            base_dir.is_dir(),
            "context clone should keep workspace alive"
        );
        assert_eq!(
            fs::read_to_string(&nested_asset).expect("should read nested fixture asset"),
            "temporary workspace asset"
        );

        drop(context_clone);
        assert!(
            base_dir.is_dir(),
            "cloned handles should keep workspace alive"
        );
        assert_eq!(
            graph_handle
                .conn()
                .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                .expect("should query through cloned graph handle"),
            1
        );
        assert_eq!(
            config_handle
                .conn()
                .query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                .expect("should query through cloned config handle"),
            1
        );

        drop(graph_handle);
        assert!(
            base_dir.is_dir(),
            "config handle should keep workspace alive"
        );

        drop(config_handle);
        assert!(
            !base_dir.exists(),
            "workspace should be removed after its last owner drops"
        );
    }

    #[test]
    fn test_setup_gen_temp_dir_cleanup() {
        assert_temp_dir_lives_until_last_context_or_handle_drop(setup_gen());
    }

    #[test]
    fn test_setup_gen_on_disk_temp_dir_cleanup() {
        assert_temp_dir_lives_until_last_context_or_handle_drop(setup_gen_on_disk());
    }

    #[test]
    fn test_setup_gen_on_disk_temp_dir_cleanup_during_unwind() {
        let mut base_dir = None;
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let context = setup_gen_on_disk();
            base_dir = Some(context.workspace().base_dir().to_path_buf());
            let fixture_asset = context
                .workspace()
                .asset_dir()
                .expect("should resolve workspace asset directory")
                .join("nested/fixture.txt");
            fs::create_dir_all(
                fixture_asset
                    .parent()
                    .expect("should have a parent directory"),
            )
            .expect("should create nested fixture asset directory");
            fs::write(fixture_asset, "temporary workspace asset")
                .expect("should write nested fixture asset");
            panic!("exercise temporary workspace cleanup during unwind");
        }));

        assert!(result.is_err(), "closure should unwind");
        let base_dir = base_dir.expect("should reach panic after workspace setup");
        assert!(
            !base_dir.exists(),
            "workspace should be removed during unwind"
        );
    }
}
