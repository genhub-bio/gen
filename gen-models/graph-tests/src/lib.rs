use std::{
    collections::{HashMap, HashSet},
    fs,
};

use gen_core::{
    GraphNode, HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, Strand, errors::ConnectionError,
};
use gen_graph::{all_intermediate_edges, graph_loader};
use gen_models::{
    block_group::{BlockGroup, BlockGroupError, NewBlockGroup, SubgraphBoundary},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::GraphConnection,
    edge::Edge,
    migrations::run_migrations,
    node::Node,
    path::Path,
    sample::{NewSample, Sample, SampleError},
    sequence::Sequence,
};
use rusqlite::Connection;

#[expect(
    clippy::too_many_arguments,
    reason = "test helper mirrors both graph boundaries"
)]
pub fn derive_subgraph(
    conn: &GraphConnection,
    source_block_group_id: &HashId,
    start_block: &NodeIntervalBlock,
    end_block: &NodeIntervalBlock,
    start_node_coordinate: i64,
    end_node_coordinate: i64,
    target_block_group_id: &HashId,
    create_terminal_edges: bool,
) -> Result<(), BlockGroupError> {
    let graph = BlockGroup::get_graph(conn, source_block_group_id, None)?;
    let start_node = graph
        .nodes()
        .find(|node| {
            node.node_id == start_block.node_id
                && node.sequence_start <= start_node_coordinate
                && node.sequence_end >= start_node_coordinate
        })
        .expect("should find the start boundary in the source graph");
    let end_node = graph
        .nodes()
        .find(|node| {
            node.node_id == end_block.node_id
                && node.sequence_start <= end_node_coordinate
                && node.sequence_end >= end_node_coordinate
        })
        .expect("should find the end boundary in the source graph");
    let edge_ids = all_intermediate_edges(&graph, start_node, end_node)
        .iter()
        .map(|(_source, _target, edge_info)| edge_info[0].edge_id)
        .collect::<Vec<_>>();
    BlockGroup::persist_subgraph(
        conn,
        source_block_group_id,
        &edge_ids,
        &SubgraphBoundary {
            block: *start_block,
            node_coordinate: start_node_coordinate,
        },
        &SubgraphBoundary {
            block: *end_block,
            node_coordinate: end_node_coordinate,
        },
        target_block_group_id,
        create_terminal_edges,
    )
}

pub fn get_all_sequences(
    conn: &GraphConnection,
    block_group_id: &HashId,
) -> Result<HashSet<String>, BlockGroupError> {
    let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
        .into_iter()
        .filter(|edge| edge.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
        .collect::<Vec<_>>();
    let blocks = Edge::blocks_from_edges(conn, block_group_id, &edges, None)?;
    let (load_edges, load_blocks) = Edge::graph_load_data(&edges, &blocks);
    let (mut graph, _) = graph_loader::build_graph(&load_edges, &load_blocks);
    graph_loader::prune_graph(&mut graph);
    let sequences_by_node = blocks
        .iter()
        .map(|block| {
            (
                GraphNode {
                    node_id: block.node_id,
                    sequence_start: block.start,
                    sequence_end: block.end,
                },
                block.sequence(),
            )
        })
        .collect::<HashMap<_, _>>();
    Ok(graph_loader::get_all_sequences(&graph, &sequences_by_node))
}

pub fn get_all_sequences_with_pruning(
    conn: &GraphConnection,
    block_group_id: &HashId,
    _prune: bool,
) -> Result<HashSet<String>, BlockGroupError> {
    get_all_sequences(conn, block_group_id)
}

pub fn get_sample_all_sequences(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    history_ref: Option<&str>,
) -> Result<HashSet<String>, SampleError> {
    let mut sequences = HashSet::new();
    for block_group in Sample::get_block_groups(conn, collection_name, sample_name, history_ref) {
        sequences.extend(get_all_sequences(conn, &block_group.id)?);
    }
    Ok(sequences)
}

pub fn get_connection<'a>(
    database_path: impl Into<Option<&'a str>>,
) -> Result<GraphConnection, ConnectionError> {
    let database_path = database_path.into();
    if let Some(database_path) = database_path
        && fs::metadata(database_path).is_ok()
    {
        fs::remove_file(database_path).expect("should remove the existing test database");
    }
    let mut conn = if let Some(database_path) = database_path {
        Connection::open(database_path).map_err(ConnectionError::OpenFailed)?
    } else {
        Connection::open_in_memory().map_err(ConnectionError::OpenFailed)?
    };
    rusqlite::vtab::array::load_module(&conn)?;
    run_migrations(&mut conn);
    Ok(GraphConnection(conn))
}

pub fn create_block_group(
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
    .expect("should create the test sample");
    BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name,
            name,
            ..Default::default()
        },
    )
    .expect("should create the test block group")
}

pub fn setup_block_group(conn: &GraphConnection) -> (HashId, Path) {
    let sequences = [
        ("AAAAAAAAAA", "test-a-node"),
        ("TTTTTTTTTT", "test-t-node"),
        ("CCCCCCCCCC", "test-c-node"),
        ("GGGGGGGGGG", "test-g-node"),
    ];
    let node_ids = sequences
        .into_iter()
        .map(|(sequence, node_name)| {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(sequence)
                .save(conn)
                .expect("should save the test sequence");
            Node::create(conn, &sequence.hash, &HashId::convert_str(node_name))
                .expect("should create the test node")
        })
        .collect::<Vec<_>>();

    Collection::get_or_create(conn, "test").expect("should create the test collection");
    let block_group = create_block_group(conn, "test", "test", "chr1");
    let node_path = [
        (PATH_START_NODE_ID, 0, node_ids[0], 0),
        (node_ids[0], 10, node_ids[1], 0),
        (node_ids[1], 10, node_ids[2], 0),
        (node_ids[2], 10, node_ids[3], 0),
        (node_ids[3], 10, PATH_END_NODE_ID, 0),
    ];
    let edges = node_path
        .into_iter()
        .map(
            |(source_node_id, source_coordinate, target_node_id, target_coordinate)| {
                Edge::create(
                    conn,
                    source_node_id,
                    source_coordinate,
                    Strand::Forward,
                    target_node_id,
                    target_coordinate,
                    Strand::Forward,
                )
                .expect("should create the test edge")
            },
        )
        .collect::<Vec<_>>();
    let block_group_edges = edges
        .iter()
        .map(|edge| BlockGroupEdgeData {
            block_group_id: block_group.id,
            edge_id: edge.id,
            chromosome_index: 0,
            phased: 0,
        })
        .collect::<Vec<_>>();
    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let edge_ids = edges.iter().map(|edge| edge.id).collect::<Vec<_>>();
    let path = Path::create(conn, "chr1", &block_group.id, &edge_ids)
        .expect("should create the test path");
    (block_group.id, path)
}

pub fn get_single_block_group_id(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    group_name: &str,
    parent_samples: Vec<String>,
) -> HashId {
    BlockGroup::get_or_create_sample_block_groups(
        conn,
        collection_name,
        sample_name,
        group_name,
        parent_samples,
    )
    .expect("should create the sample block group")
    .into_iter()
    .next()
    .expect("should return one block group")
    .id
}
