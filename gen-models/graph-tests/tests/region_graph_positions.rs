use std::collections::HashSet;

use gen_core::{GraphNodePosition, HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    accession::Accession,
    block_group::{BlockGroup, NewBlockGroup, PathCache},
    block_group_edge::{AugmentedEdge, BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::GraphConnection,
    edge::Edge,
    node::Node,
    path::Path,
    region::{ResolvedGenRegion, ResolvedRegionKind},
    sample::{NewSample, Sample},
    sequence::Sequence,
};
use gen_models_graph_tests::get_connection;

trait FindGraphPositions {
    fn find_graph_positions(
        &self,
        conn: &GraphConnection,
        start_offset: i64,
        end_offset: i64,
    ) -> Result<ResolvedGenRegion, gen_graph::GraphError>;
}

impl FindGraphPositions for ResolvedGenRegion {
    fn find_graph_positions(
        &self,
        conn: &GraphConnection,
        start_offset: i64,
        end_offset: i64,
    ) -> Result<ResolvedGenRegion, gen_graph::GraphError> {
        gen_graph::models::find_region_graph_positions(self, conn, start_offset, end_offset)
    }
}

fn setup_graph() -> (GraphConnection, HashId) {
    let conn = get_connection(None).unwrap();
    Collection::get_or_create(&conn, "test").unwrap();
    Sample::get_or_create(
        &conn,
        NewSample {
            name: "test",
            ..Default::default()
        },
    )
    .unwrap();
    let block_group = BlockGroup::create(
        &conn,
        NewBlockGroup {
            collection_name: "test",
            sample_name: "test",
            name: "chr1",
            ..Default::default()
        },
    )
    .unwrap();

    let seq_x = Sequence::new()
        .sequence_type("DNA")
        .sequence("XXXXX")
        .save(&conn)
        .unwrap();
    let seq_y = Sequence::new()
        .sequence_type("DNA")
        .sequence("YYYYY")
        .save(&conn)
        .unwrap();
    let seq_z = Sequence::new()
        .sequence_type("DNA")
        .sequence("ZZZZZ")
        .save(&conn)
        .unwrap();

    let node_x = Node::create(&conn, &seq_x.hash, &HashId::convert_str("node-x")).unwrap();
    let node_y = Node::create(&conn, &seq_y.hash, &HashId::convert_str("node-y")).unwrap();
    let node_z = Node::create(&conn, &seq_z.hash, &HashId::convert_str("node-z")).unwrap();

    let e_start = Edge::create(
        &conn,
        PATH_START_NODE_ID,
        -1,
        Strand::Forward,
        node_x,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_xy = Edge::create(
        &conn,
        node_x,
        5,
        Strand::Forward,
        node_y,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_yz = Edge::create(
        &conn,
        node_y,
        5,
        Strand::Forward,
        node_z,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_end = Edge::create(
        &conn,
        node_z,
        5,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )
    .unwrap();

    BlockGroupEdge::bulk_create(
        &conn,
        &[
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: e_start.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: e_xy.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: e_yz.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: e_end.id,
                chromosome_index: 0,
                phased: 0,
            },
        ],
    );

    (conn, block_group.id)
}

fn create_accession(
    conn: &GraphConnection,
    block_group_id: HashId,
    name: &str,
    start: i64,
    end: i64,
) -> Accession {
    let edges = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None);
    let mut by_source: std::collections::HashMap<HashId, &AugmentedEdge> =
        std::collections::HashMap::new();
    for ae in &edges {
        by_source.insert(ae.edge.source_node_id, ae);
    }
    let mut ordered = vec![];
    let mut current = Some(PATH_START_NODE_ID);
    while let Some(src) = current {
        if let Some(ae) = by_source.get(&src) {
            ordered.push(ae.edge.id);
            current = if ae.edge.target_node_id == PATH_END_NODE_ID {
                None
            } else {
                Some(ae.edge.target_node_id)
            };
        } else {
            break;
        }
    }
    let path = Path::create(conn, name, &block_group_id, &ordered).unwrap();
    let mut path_cache = PathCache::new(conn);
    let accession =
        BlockGroup::add_accession(conn, &path, name, start, end, &mut path_cache).unwrap();
    Path::delete(conn, name, &block_group_id);
    accession
}

fn create_accession_from_edges(
    conn: &GraphConnection,
    block_group_id: HashId,
    name: &str,
    edge_ids: &[HashId],
    start: i64,
    end: i64,
) -> Accession {
    let path = Path::create(conn, name, &block_group_id, edge_ids).unwrap();
    let mut path_cache = PathCache::new(conn);
    let accession =
        BlockGroup::add_accession(conn, &path, name, start, end, &mut path_cache).unwrap();
    Path::delete(conn, name, &block_group_id);
    accession
}

fn make_region(
    bg: BlockGroup,
    accession: Accession,
    anchor_start: i64,
    anchor_end: i64,
    feature_length: i64,
    start: i64,
    end: i64,
) -> ResolvedGenRegion {
    ResolvedGenRegion {
        block_group: bg,
        path: None,
        accession: Some(accession),
        annotation: None,
        kind: ResolvedRegionKind::Accession,
        anchor_start,
        anchor_end,
        feature_length,
        start,
        end,
        start_anchors: None,
        end_anchors: None,
        remove_ambiguous_positions: false,
    }
}

/// Creates a branched graph: {AAA,GGG} → TTT → {CCC,ATC}
/// Path: AAA→TTT→CCC (positions 0..9)
fn setup_branched_graph() -> (GraphConnection, HashId) {
    let conn = get_connection(None).unwrap();
    Collection::get_or_create(&conn, "test").unwrap();
    Sample::get_or_create(
        &conn,
        NewSample {
            name: "test",
            ..Default::default()
        },
    )
    .unwrap();
    let bg = BlockGroup::create(
        &conn,
        NewBlockGroup {
            collection_name: "test",
            sample_name: "test",
            name: "branched",
            ..Default::default()
        },
    )
    .unwrap();

    let seq_aaa = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAA")
        .save(&conn)
        .unwrap();
    let seq_ggg = Sequence::new()
        .sequence_type("DNA")
        .sequence("GGG")
        .save(&conn)
        .unwrap();
    let seq_ttt = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTT")
        .save(&conn)
        .unwrap();
    let seq_ccc = Sequence::new()
        .sequence_type("DNA")
        .sequence("CCC")
        .save(&conn)
        .unwrap();
    let seq_atc = Sequence::new()
        .sequence_type("DNA")
        .sequence("ATC")
        .save(&conn)
        .unwrap();

    let n_aaa = Node::create(&conn, &seq_aaa.hash, &HashId::convert_str("node-aaa")).unwrap();
    let n_ggg = Node::create(&conn, &seq_ggg.hash, &HashId::convert_str("node-ggg")).unwrap();
    let n_ttt = Node::create(&conn, &seq_ttt.hash, &HashId::convert_str("node-ttt")).unwrap();
    let n_ccc = Node::create(&conn, &seq_ccc.hash, &HashId::convert_str("node-ccc")).unwrap();
    let n_atc = Node::create(&conn, &seq_atc.hash, &HashId::convert_str("node-atc")).unwrap();

    let e_start = Edge::create(
        &conn,
        PATH_START_NODE_ID,
        -1,
        Strand::Forward,
        n_aaa,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_ggg_start = Edge::create(
        &conn,
        PATH_START_NODE_ID,
        -1,
        Strand::Forward,
        n_ggg,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_aaa_ttt =
        Edge::create(&conn, n_aaa, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
    let e_ttt_ccc =
        Edge::create(&conn, n_ttt, 3, Strand::Forward, n_ccc, 0, Strand::Forward).unwrap();
    let e_ccc_end = Edge::create(
        &conn,
        n_ccc,
        3,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_atc_end = Edge::create(
        &conn,
        n_atc,
        3,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_ggg_ttt =
        Edge::create(&conn, n_ggg, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
    let e_ttt_atc =
        Edge::create(&conn, n_ttt, 3, Strand::Forward, n_atc, 0, Strand::Forward).unwrap();

    BlockGroupEdge::bulk_create(
        &conn,
        &[
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_start.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_ggg_start.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_aaa_ttt.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_ttt_ccc.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_ccc_end.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_atc_end.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_ggg_ttt.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_ttt_atc.id,
                chromosome_index: 0,
                phased: 0,
            },
        ],
    );

    (conn, bg.id)
}

struct GraphFixture {
    conn: GraphConnection,
    block_group_id: HashId,
    path: Vec<HashId>,
}

/// Creates a branched graph: AAA -> {CC, GGGG} -> TTT.
fn setup_variable_length_branched_graph() -> GraphFixture {
    let conn = get_connection(None).unwrap();
    Collection::get_or_create(&conn, "test").unwrap();
    Sample::get_or_create(
        &conn,
        NewSample {
            name: "test",
            ..Default::default()
        },
    )
    .unwrap();
    let bg = BlockGroup::create(
        &conn,
        NewBlockGroup {
            collection_name: "test",
            sample_name: "test",
            name: "variable-length-branched",
            ..Default::default()
        },
    )
    .unwrap();

    let seq_aaa = Sequence::new()
        .sequence_type("DNA")
        .sequence("AAA")
        .save(&conn)
        .unwrap();
    let seq_cc = Sequence::new()
        .sequence_type("DNA")
        .sequence("CC")
        .save(&conn)
        .unwrap();
    let seq_gggg = Sequence::new()
        .sequence_type("DNA")
        .sequence("GGGG")
        .save(&conn)
        .unwrap();
    let seq_ttt = Sequence::new()
        .sequence_type("DNA")
        .sequence("TTT")
        .save(&conn)
        .unwrap();

    let n_aaa = Node::create(&conn, &seq_aaa.hash, &HashId::convert_str("node-aaa")).unwrap();
    let n_cc = Node::create(&conn, &seq_cc.hash, &HashId::convert_str("node-cc")).unwrap();
    let n_gggg = Node::create(&conn, &seq_gggg.hash, &HashId::convert_str("node-gggg")).unwrap();
    let n_ttt = Node::create(&conn, &seq_ttt.hash, &HashId::convert_str("node-ttt")).unwrap();

    let e_start = Edge::create(
        &conn,
        PATH_START_NODE_ID,
        -1,
        Strand::Forward,
        n_aaa,
        0,
        Strand::Forward,
    )
    .unwrap();
    let e_aaa_cc =
        Edge::create(&conn, n_aaa, 3, Strand::Forward, n_cc, 0, Strand::Forward).unwrap();
    let e_aaa_gggg =
        Edge::create(&conn, n_aaa, 3, Strand::Forward, n_gggg, 0, Strand::Forward).unwrap();
    let e_cc_ttt =
        Edge::create(&conn, n_cc, 2, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
    let e_gggg_ttt =
        Edge::create(&conn, n_gggg, 4, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
    let e_end = Edge::create(
        &conn,
        n_ttt,
        3,
        Strand::Forward,
        PATH_END_NODE_ID,
        0,
        Strand::Forward,
    )
    .unwrap();

    BlockGroupEdge::bulk_create(
        &conn,
        &[
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_start.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_aaa_cc.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_aaa_gggg.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_cc_ttt.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_gggg_ttt.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: bg.id,
                edge_id: e_end.id,
                chromosome_index: 0,
                phased: 0,
            },
        ],
    );

    GraphFixture {
        conn,
        block_group_id: bg.id,
        path: vec![e_start.id, e_aaa_cc.id, e_cc_ttt.id, e_end.id],
    }
}

fn position_set(positions: &[GraphNodePosition]) -> HashSet<(HashId, i64)> {
    positions
        .iter()
        .map(|pos| (pos.graph_node.node_id, pos.offset))
        .collect()
}

#[test]
fn test_finds_graph_positions_within_node() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "within", 0, 15);
    let region = make_region(bg, acc, 0, 15, 15, 7, 7);

    let resolved = region.find_graph_positions(&conn, 2, 2).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-y")
    );
    assert_eq!(start_pos[0].offset, 4);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-y"));
    assert_eq!(end_pos[0].offset, 4);
}

#[test]
fn test_finds_graph_positions_forward_across_nodes() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "fwd", 0, 15);
    let region = make_region(bg, acc, 0, 15, 15, 7, 7);

    let resolved = region.find_graph_positions(&conn, 5, 5).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-z")
    );
    assert_eq!(start_pos[0].offset, 2);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-z"));
    assert_eq!(end_pos[0].offset, 2);
}

#[test]
fn test_finds_graph_positions_backwards_across_nodes() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "bwd", 0, 15);
    let region = make_region(bg, acc, 0, 15, 15, 7, 7);

    let resolved = region.find_graph_positions(&conn, -5, -5).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-x")
    );
    assert_eq!(start_pos[0].offset, 2);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-x"));
    assert_eq!(end_pos[0].offset, 2);
}

#[test]
fn test_reports_out_of_bounds() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "oob", 0, 15);
    let region = make_region(bg, acc, 0, 15, 15, 7, 7);

    assert!(region.find_graph_positions(&conn, 100, 100).is_err());
}

#[test]
fn test_finds_graph_positions_from_start() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "start", 0, 15);
    let region = make_region(bg, acc, 0, 15, 15, 0, 0);

    let resolved = region.find_graph_positions(&conn, 12, 12).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-z")
    );
    assert_eq!(start_pos[0].offset, 2);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-z"));
    assert_eq!(end_pos[0].offset, 2);
}

#[test]
fn test_finds_graph_positions_from_end() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "end", 0, 15);
    let region = make_region(bg, acc, 0, 15, 15, 14, 14);

    let resolved = region.find_graph_positions(&conn, -14, -14).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-x")
    );
    assert_eq!(start_pos[0].offset, 0);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-x"));
    assert_eq!(end_pos[0].offset, 0);
}

#[test]
fn test_finds_graph_positions_within_accessions() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "acc-within", 5, 10);
    let region = make_region(bg, acc, 5, 10, 5, 2, 2);

    let resolved = region.find_graph_positions(&conn, 1, 1).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-y")
    );
    assert_eq!(start_pos[0].offset, 3);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-y"));
    assert_eq!(end_pos[0].offset, 3);
}

#[test]
fn test_finds_graph_positions_expands_accession_forward() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "acc-fwd", 5, 10);
    let region = make_region(bg, acc, 5, 10, 5, 3, 3);

    let resolved = region.find_graph_positions(&conn, 5, 5).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-z")
    );
    assert_eq!(start_pos[0].offset, 3);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-z"));
    assert_eq!(end_pos[0].offset, 3);
}

#[test]
fn test_finds_graph_positions_expands_accession_backward() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "acc-bwd", 5, 10);
    let region = make_region(bg, acc, 5, 10, 5, 1, 1);

    let resolved = region.find_graph_positions(&conn, -4, -4).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 1);
    assert_eq!(
        start_pos[0].graph_node.node_id,
        HashId::convert_str("node-x")
    );
    assert_eq!(start_pos[0].offset, 2);
    assert_eq!(end_pos.len(), 1);
    assert_eq!(end_pos[0].graph_node.node_id, HashId::convert_str("node-x"));
    assert_eq!(end_pos[0].offset, 2);
}

#[test]
fn test_finds_graph_positions_reports_accession_out_of_bounds() {
    let (conn, bg_id) = setup_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    let acc = create_accession(&conn, bg_id, "acc-oob", 5, 10);
    let region = make_region(bg, acc, 5, 10, 5, 2, 2);

    assert!(region.find_graph_positions(&conn, 100, 100).is_err());
}

#[test]
fn test_finds_graph_positions_in_branched_graph_backwards_returns_all_positions() {
    let (conn, bg_id) = setup_branched_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    // Accession on TTT: path positions 3..6, accession-relative 0..3
    let acc = create_accession(&conn, bg_id, "branched-bwd", 3, 6);
    let region = make_region(bg, acc, 3, 6, 3, 0, 0);

    // Backward 3 from TTT offset 0 → should find AAA and GGG at offset 0
    let resolved = region.find_graph_positions(&conn, -3, -3).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 2);
    let start_ids: Vec<HashId> = start_pos.iter().map(|p| p.graph_node.node_id).collect();
    assert!(start_ids.contains(&HashId::convert_str("node-aaa")));
    assert!(start_ids.contains(&HashId::convert_str("node-ggg")));
    for pos in start_pos {
        assert_eq!(pos.offset, 0);
    }
    assert_eq!(end_pos.len(), 2);
    let end_ids: Vec<HashId> = end_pos.iter().map(|p| p.graph_node.node_id).collect();
    assert!(end_ids.contains(&HashId::convert_str("node-aaa")));
    assert!(end_ids.contains(&HashId::convert_str("node-ggg")));
}

#[test]
fn test_finds_graph_positions_in_branched_graph_forwardsgr_returns_all_positions() {
    let (conn, bg_id) = setup_branched_graph();
    let bg = BlockGroup::get_by_id(&conn, &bg_id, None).unwrap();
    // Accession on TTT: path positions 3..6, accession-relative 0..3
    let acc = create_accession(&conn, bg_id, "branched-fwd", 3, 6);
    let region = make_region(bg, acc, 3, 6, 3, 2, 2);

    // Forward 3 from TTT offset 2 → should find CCC and ATC at offset 2
    let resolved = region.find_graph_positions(&conn, 3, 3).unwrap();
    let start_pos = resolved.start_anchors.as_ref().unwrap();
    let end_pos = resolved.end_anchors.as_ref().unwrap();
    assert_eq!(start_pos.len(), 2);
    let start_ids: Vec<HashId> = start_pos.iter().map(|p| p.graph_node.node_id).collect();
    assert!(start_ids.contains(&HashId::convert_str("node-ccc")));
    assert!(start_ids.contains(&HashId::convert_str("node-atc")));
    for pos in start_pos {
        assert_eq!(pos.offset, 2);
    }
    assert_eq!(end_pos.len(), 2);
    let end_ids: Vec<HashId> = end_pos.iter().map(|p| p.graph_node.node_id).collect();
    assert!(end_ids.contains(&HashId::convert_str("node-ccc")));
    assert!(end_ids.contains(&HashId::convert_str("node-atc")));
}

#[test]
fn test_finds_graph_positions_in_variable_length_branch_finds_middle_nodes() {
    let fixture = setup_variable_length_branched_graph();
    let bg = BlockGroup::get_by_id(&fixture.conn, &fixture.block_group_id, None).unwrap();
    let aaa_acc = create_accession_from_edges(
        &fixture.conn,
        fixture.block_group_id,
        "variable-aaa",
        &fixture.path,
        0,
        3,
    );
    let aaa_region = make_region(bg.clone(), aaa_acc, 0, 3, 3, 2, 2);

    let from_aaa = aaa_region
        .find_graph_positions(&fixture.conn, 2, 2)
        .unwrap();
    assert_eq!(
        position_set(&from_aaa.start_anchors.unwrap()),
        HashSet::from([
            (HashId::convert_str("node-cc"), 1),
            (HashId::convert_str("node-gggg"), 1)
        ])
    );

    let ttt_acc = create_accession_from_edges(
        &fixture.conn,
        fixture.block_group_id,
        "variable-ttt",
        &fixture.path,
        5,
        8,
    );
    let ttt_region = make_region(bg, ttt_acc, 5, 8, 3, 0, 0);

    let from_ttt = ttt_region
        .find_graph_positions(&fixture.conn, -2, -2)
        .unwrap();
    assert_eq!(
        position_set(&from_ttt.start_anchors.unwrap()),
        HashSet::from([
            (HashId::convert_str("node-cc"), 0),
            (HashId::convert_str("node-gggg"), 2)
        ])
    );
}

#[test]
fn test_finds_graph_positions_in_variable_length_branch_returns_single_position() {
    let fixture = setup_variable_length_branched_graph();
    let bg = BlockGroup::get_by_id(&fixture.conn, &fixture.block_group_id, None).unwrap();
    let acc = create_accession_from_edges(
        &fixture.conn,
        fixture.block_group_id,
        "variable-single",
        &fixture.path,
        0,
        3,
    );
    let region = make_region(bg, acc, 0, 3, 3, 1, 1);

    let positions = region.find_graph_positions(&fixture.conn, 1, 1).unwrap();
    assert_eq!(
        position_set(&positions.start_anchors.unwrap()),
        HashSet::from([(HashId::convert_str("node-aaa"), 2)])
    );
}

#[test]
fn test_finds_graph_positions_in_variable_length_branch_finds_different_ttt_offsets() {
    let fixture = setup_variable_length_branched_graph();
    let bg = BlockGroup::get_by_id(&fixture.conn, &fixture.block_group_id, None).unwrap();
    let acc = create_accession_from_edges(
        &fixture.conn,
        fixture.block_group_id,
        "variable-ttt-offsets",
        &fixture.path,
        0,
        3,
    );
    let region = make_region(bg, acc, 0, 3, 3, 2, 2);

    let positions = region.find_graph_positions(&fixture.conn, 6, 6).unwrap();
    assert_eq!(
        position_set(&positions.start_anchors.unwrap()),
        HashSet::from([
            (HashId::convert_str("node-ttt"), 1),
            (HashId::convert_str("node-ttt"), 3)
        ])
    );
}
