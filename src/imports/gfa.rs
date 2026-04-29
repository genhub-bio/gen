use std::{collections::HashMap, path::Path as FilePath};

use gen_core::{
    HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, is_end_node,
    is_start_node,
};
use gen_graph::{GraphEdge, GraphNode};
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::{Edge, EdgeData},
    errors::{OperationError, SequenceError},
    file_types::FileTypes,
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
    traits::Query,
};
use indexmap::IndexSet;
use itertools::Itertools;
use petgraph::{algo::kosaraju_scc, prelude::UnGraphMap, visit::Dfs};
use thiserror::Error;

use crate::{
    gfa::bool_to_strand,
    gfa_reader::Gfa,
    progress_bar::{get_handler, get_message_bar, get_progress_bar, get_time_elapsed_bar},
};

#[derive(Debug, Error, PartialEq)]
pub enum GFAImportError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Sequence Error: {0}")]
    Sequence(#[from] SequenceError),
}

pub fn import_gfa(
    context: &DbContext,
    gfa_path: &FilePath,
    collection_name: &str,
    sample_name: &str,
) -> Result<Operation, GFAImportError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let mut session = start_operation(conn);
    Collection::create(conn, collection_name);
    Sample::get_or_create(conn, sample_name);
    let block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name,
            name: "",
            ..Default::default()
        },
    );
    let bar = progress_bar.add(get_time_elapsed_bar());
    bar.set_message("Parsing GFA");
    let gfa: Gfa<String, (), ()> = Gfa::parse_gfa_file(gfa_path.to_str().unwrap());
    let mut sequences_by_segment_id: HashMap<&String, Sequence> = HashMap::new();
    let mut node_ids_by_segment_id: HashMap<&String, HashId> = HashMap::new();
    bar.finish();

    let bar = progress_bar.add(get_progress_bar(gfa.segments.len() as u64));
    bar.set_message("Parsing Segments");
    for segment in &gfa.segments {
        let input_sequence = segment.sequence.get_string(&gfa.sequence);
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(input_sequence)
            .save(conn);
        sequences_by_segment_id.insert(&segment.id, sequence.clone());
        // TODO: Node hash is always new, it's sorted by insert time via being a v7 uuid but maybe want to
        // define the hash itself for idempotency?
        let node_id = Node::create(conn, &sequence.hash, &HashId::uuid7());
        node_ids_by_segment_id.insert(&segment.id, node_id);
        bar.inc(1);
    }
    bar.finish();

    let mut edges = IndexSet::new();
    let bar = progress_bar.add(get_progress_bar(gfa.links.len() as u64));
    let mut source_refs_in_links = IndexSet::new();
    let mut target_refs_in_links = IndexSet::new();

    bar.set_message("Parsing Links");
    for link in &gfa.links {
        let source = sequences_by_segment_id.get(&link.from).unwrap();
        let source_node_id = *node_ids_by_segment_id.get(&link.from).unwrap();
        source_refs_in_links.insert(&link.from);
        let target_node_id = *node_ids_by_segment_id.get(&link.to).unwrap();
        target_refs_in_links.insert(&link.to);
        edges.insert(edge_data_from_fields(
            source_node_id,
            source.length,
            bool_to_strand(link.from_dir),
            target_node_id,
            bool_to_strand(link.to_dir),
        ));
        bar.inc(1);
    }
    bar.finish();

    let pure_source_refs = source_refs_in_links
        .difference(&target_refs_in_links)
        .collect::<IndexSet<_>>();
    let pure_target_refs = target_refs_in_links
        .difference(&source_refs_in_links)
        .collect::<IndexSet<_>>();
    for source_ref in pure_source_refs {
        let source_node_id = *node_ids_by_segment_id.get(source_ref).unwrap();
        edges.insert(edge_data_from_fields(
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            source_node_id,
            Strand::Forward,
        ));
    }

    for target_ref in pure_target_refs {
        let target_node_id = *node_ids_by_segment_id.get(target_ref).unwrap();
        let target_sequence = sequences_by_segment_id.get(target_ref).unwrap();
        edges.insert(edge_data_from_fields(
            target_node_id,
            target_sequence.length,
            Strand::Forward,
            PATH_END_NODE_ID,
            Strand::Forward,
        ));
    }

    let bar = progress_bar.add(get_progress_bar(gfa.paths.len() as u64));
    bar.set_message("Parsing Paths");
    for input_path in &gfa.paths {
        let mut source_node_id = PATH_START_NODE_ID;
        let mut source_coordinate = 0;
        let mut source_strand = Strand::Forward;
        for (index, segment_id) in input_path.segments.iter().enumerate() {
            let target = sequences_by_segment_id.get(segment_id).unwrap();
            let target_node_id = *node_ids_by_segment_id.get(segment_id).unwrap();
            let target_strand = bool_to_strand(input_path.strands[index]);
            edges.insert(edge_data_from_fields(
                source_node_id,
                source_coordinate,
                source_strand,
                target_node_id,
                target_strand,
            ));
            source_node_id = target_node_id;
            source_coordinate = target.length;
            source_strand = target_strand;
        }
        edges.insert(edge_data_from_fields(
            source_node_id,
            source_coordinate,
            source_strand,
            PATH_END_NODE_ID,
            Strand::Forward,
        ));
        bar.inc(1);
    }
    bar.finish();

    let bar = progress_bar.add(get_progress_bar(gfa.paths.len() as u64));
    bar.set_message("Parsing Walks");
    for input_walk in &gfa.walk {
        let mut source_node_id = PATH_START_NODE_ID;
        let mut source_coordinate = 0;
        let mut source_strand = Strand::Forward;
        for (index, segment_id) in input_walk.segments.iter().enumerate() {
            let target = sequences_by_segment_id.get(segment_id).unwrap();
            let target_node_id = *node_ids_by_segment_id.get(segment_id).unwrap();
            let target_strand = bool_to_strand(input_walk.strands[index]);
            edges.insert(edge_data_from_fields(
                source_node_id,
                source_coordinate,
                source_strand,
                target_node_id,
                target_strand,
            ));
            source_node_id = target_node_id;
            source_coordinate = target.length;
            source_strand = target_strand;
        }
        edges.insert(edge_data_from_fields(
            source_node_id,
            source_coordinate,
            source_strand,
            PATH_END_NODE_ID,
            Strand::Forward,
        ));
        bar.inc(1);
    }
    bar.finish();

    let gen_bar = progress_bar.add(get_time_elapsed_bar());
    gen_bar.set_message("Creating Gen Objects");
    let edge_ids = Edge::bulk_create(conn, &edges.into_iter().collect::<Vec<EdgeData>>());

    let saved_edges = Edge::query_by_ids(conn, &edge_ids);
    let mut edge_ids_by_data = HashMap::new();
    for edge in saved_edges {
        let key = edge_data_from_fields(
            edge.source_node_id,
            edge.source_coordinate,
            edge.source_strand,
            edge.target_node_id,
            edge.target_strand,
        );
        edge_ids_by_data.insert(key, edge.id);
    }

    let mut created_blockgroup_edges: IndexSet<HashId> = IndexSet::new();

    for input_path in &gfa.paths {
        let path_name = &input_path.name;
        let mut source_node_id = PATH_START_NODE_ID;
        let mut source_coordinate = 0;
        let mut source_strand = Strand::Forward;
        let mut path_edge_ids = vec![];
        for (index, segment_id) in input_path.segments.iter().enumerate() {
            let target = sequences_by_segment_id.get(segment_id).unwrap();
            let target_node_id = *node_ids_by_segment_id.get(segment_id).unwrap();
            let target_strand = bool_to_strand(input_path.strands[index]);
            let key = edge_data_from_fields(
                source_node_id,
                source_coordinate,
                source_strand,
                target_node_id,
                target_strand,
            );
            let edge_id = *edge_ids_by_data.get(&key).unwrap();
            path_edge_ids.push(edge_id);
            source_node_id = target_node_id;
            source_coordinate = target.length;
            source_strand = target_strand;
        }
        let key = edge_data_from_fields(
            source_node_id,
            source_coordinate,
            source_strand,
            PATH_END_NODE_ID,
            Strand::Forward,
        );
        let edge_id = *edge_ids_by_data.get(&key).unwrap();
        path_edge_ids.push(edge_id);
        created_blockgroup_edges.extend(path_edge_ids.iter());

        BlockGroupEdge::bulk_create(
            conn,
            &path_edge_ids
                .iter()
                .map(|id| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: *id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>(),
        );
        Path::create(conn, path_name, &block_group.id, &path_edge_ids);
    }

    for input_walk in &gfa.walk {
        let path_name = &input_walk.sample_id;
        let mut source_node_id = PATH_START_NODE_ID;
        let mut source_coordinate = 0;
        let mut source_strand = Strand::Forward;
        let mut path_edge_ids = vec![];
        for (index, segment_id) in input_walk.segments.iter().enumerate() {
            let target = sequences_by_segment_id.get(segment_id).unwrap();
            let target_node_id = *node_ids_by_segment_id.get(segment_id).unwrap();
            let target_strand = bool_to_strand(input_walk.strands[index]);
            let key = edge_data_from_fields(
                source_node_id,
                source_coordinate,
                source_strand,
                target_node_id,
                target_strand,
            );
            let edge_id = *edge_ids_by_data.get(&key).unwrap();
            path_edge_ids.push(edge_id);
            source_node_id = target_node_id;
            source_coordinate = target.length;
            source_strand = target_strand;
        }
        let key = edge_data_from_fields(
            source_node_id,
            source_coordinate,
            source_strand,
            PATH_END_NODE_ID,
            Strand::Forward,
        );
        let edge_id = *edge_ids_by_data.get(&key).unwrap();
        path_edge_ids.push(edge_id);
        created_blockgroup_edges.extend(path_edge_ids.iter());

        BlockGroupEdge::bulk_create(
            conn,
            &path_edge_ids
                .iter()
                .map(|id| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: *id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>(),
        );
        Path::create(conn, path_name, &block_group.id, &path_edge_ids);
    }

    // make any block group edges not in paths or walks
    BlockGroupEdge::bulk_create(
        conn,
        &edge_ids
            .iter()
            .filter_map(|id| {
                if !created_blockgroup_edges.contains(id) {
                    Some(BlockGroupEdgeData {
                        block_group_id: block_group.id,
                        edge_id: *id,
                        chromosome_index: NO_CHROMOSOME_INDEX,
                        phased: 0,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<BlockGroupEdgeData>>(),
    );

    // check the graph for cycles and make start/end nodes if so
    let bar = progress_bar.add(get_progress_bar(None));
    bar.set_message("Breaking cycles");
    let message_bar = progress_bar.add(get_message_bar());
    let graph = BlockGroup::get_graph(conn, &block_group.id)?;
    let mut undirected_graph: UnGraphMap<GraphNode, GraphEdge> = UnGraphMap::new();
    for node in graph.nodes() {
        undirected_graph.add_node(node);
    }
    for (src, dst, weights) in graph.all_edges() {
        undirected_graph.add_edge(src, dst, weights[0]);
    }
    let connected_components = kosaraju_scc(&undirected_graph);
    let mut new_edges = vec![];
    for subgraph in connected_components.iter() {
        if subgraph.len() >= 3 {
            let mut has_start = false;
            let mut has_end = false;
            for node in subgraph.iter() {
                if !has_start && is_start_node(node.node_id) {
                    has_start = true;
                } else if !has_end && is_end_node(node.node_id) {
                    has_end = true;
                };
                if has_start && has_end {
                    break;
                }
            }
            // For graphs with just one enter/exit point, we log a message
            if !has_start && !has_end {
                // from the subgraph, we want to find a deterministic sort of ordered elements.
                // Kosaraju returns nodes in arbitrary order. We use DFS and then rotate the vector
                // so the first node_id starts the list for consistency. If a node in the DFS is in
                // a known start node for a path, we use that one.
                let mut order = vec![];
                let mut dfs = Dfs::new(&graph, subgraph[0]);
                while let Some(nx) = dfs.next(&graph) {
                    order.push(nx);
                }
                let min_index = order.iter().enumerate().min_set_by_key(|(_, k)| k.node_id)[0].0;
                order.rotate_left(min_index);
                bar.inc(1);
                new_edges.push(edge_data_from_fields(
                    PATH_START_NODE_ID,
                    0,
                    Strand::Forward,
                    order[0].node_id,
                    Strand::Forward,
                ));
                let last_node = order.last().unwrap();
                new_edges.push(edge_data_from_fields(
                    last_node.node_id,
                    last_node.sequence_end,
                    Strand::Forward,
                    PATH_END_NODE_ID,
                    Strand::Forward,
                ));
                new_edges.push(edge_data_from_fields(
                    PATH_END_NODE_ID,
                    0,
                    Strand::Forward,
                    PATH_START_NODE_ID,
                    Strand::Forward,
                ));
            } else if has_start && has_end {
                // there's a cycle, but has a start/end already. At some point we should track this
                // so we know ahead of time where the cycles are
            } else {
                message_bar.set_message("Path encountered with cycle after start/end node, no cycle breaking will apply.");
            }
        }
    }
    message_bar.finish();
    let new_edge_ids = Edge::bulk_create(conn, &new_edges.into_iter().collect::<Vec<EdgeData>>());
    BlockGroupEdge::bulk_create(
        conn,
        &new_edge_ids
            .iter()
            .map(|id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>(),
    );
    bar.finish();

    let op = end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: gfa_path.to_str().unwrap().to_string(),
                file_type: FileTypes::GFA,
            }],
            description: "gfa_import".to_string(),
        },
        &format!("Imported GFA {path}", path = gfa_path.to_str().unwrap()),
        None,
    )
    .map_err(GFAImportError::OperationError);
    gen_bar.finish();
    op
}

fn edge_data_from_fields(
    source_node_id: HashId,
    source_coordinate: i64,
    source_strand: Strand,
    target_node_id: HashId,
    target_strand: Strand,
) -> EdgeData {
    EdgeData {
        source_node_id,
        source_coordinate,
        source_strand,
        target_node_id,
        target_coordinate: 0,
        target_strand,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_models::traits::*;
    use rusqlite::params;

    use super::*;
    use crate::{test_helpers::setup_gen, track_database};

    #[test]
    fn test_import_simple_gfa() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/simple.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        track_database(conn, context.operations().conn()).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "m123"],
        )[0]
        .clone();

        let result = path.sequence(conn).unwrap();
        assert_eq!(result, "ATCGATCGATCGATCGATCGGGAACACACAGAGA");

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_creates_sample() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/simple.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        track_database(conn, context.operations().conn()).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, "new-sample");
        assert_eq!(
            Sample::get_by_name(conn, "new-sample").unwrap().name,
            "new-sample"
        );
    }

    #[test]
    fn test_import_no_path_gfa() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/no_path.gfa");
        let collection_name = "no path".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        track_database(conn, context.operations().conn()).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAAATTTTGGGGCCCC".to_string()])
        );

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_import_gfa_with_walk() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/walk.gfa");
        let collection_name = "walk".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        track_database(conn, context.operations().conn()).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "291344"],
        )[0]
        .clone();

        let result = path.sequence(conn).unwrap();
        assert_eq!(result, "ACCTACAAATTCAAAC");

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_import_gfa_with_reverse_strand_edges() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/reverse_strand.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        track_database(conn, context.operations().conn()).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "124"],
        )[0]
        .clone();

        let result = path.sequence(conn).unwrap();
        assert_eq!(result, "TATGCCAGCTGCGAATA");

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 6);
    }

    #[test]
    fn test_import_anderson_promoters() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/anderson_promoters.gfa");
        let collection_name = "anderson promoters".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let paths = Path::query_for_collection(conn, &collection_name);
        assert_eq!(paths.len(), 20);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "BBa_J23100"],
        )[0]
        .clone();

        let result = path.sequence(conn).unwrap();
        let big_part = "TGCTAGCTACTAGTGAAAGAGGAGAAATACTAGATGGCTTCCTCCGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAACGCTGATAGTGCTAGTGTAGATCGCTACTAGAGCCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATATACTAGAAGCGGCCGCTGCAGGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATACGGTTATCCACAGAATCAGGGGATAACGCAGGAAAGAACATGTGAGCAAAAGGCCAGCAAAAGGCCAGGAACCGTAAAAAGGCCGCGTTGCTGGCGTTTTTCCATAGGCTCCGCCCCCCTGACGAGCATCACAAAAATCGACGCTCAAGTCAGAGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCCTGGAAGCTCCCTCGTGCGCTCTCCTGTTCCGACCCTGCCGCTTACCGGATACCTGTCCGCCTTTCTCCCTTCGGGAAGCGTGGCGCTTTCTCATAGCTCACGCTGTAGGTATCTCAGTTCGGTGTAGGTCGTTCGCTCCAAGCTGGGCTGTGTGCACGAACCCCCCGTTCAGCCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGTAAGACACGACTTATCGCCACTGGCAGCAGCCACTGGTAACAGGATTAGCAGAGCGAGGTATGTAGGCGGTGCTACAGAGTTCTTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGGACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGCAAGCAGCAGATTACGCGCAGAAAAAAAGGATCTCAAGAAGATCCTTTGATCTTTTCTACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGTTACCAATGCTTAATCAGTGAGGCACCTATCTCAGCGATCTGTCTATTTCGTTCATCCATAGTTGCCTGACTCCCCGTCGTGTAGATAACTACGATACGGGAGGGCTTACCATCTGGCCCCAGTGCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAAGGGCCGAGCGCAGAAGTGGTCCTGCAACTTTATCCGCCTCCATCCAGTCTATTAATTGTTGCCGGGAAGCTAGAGTAAGTAGTTCGCCAGTTAATAGTTTGCGCAACGTTGTTGCCATTGCTACAGGCATCGTGGTGTCACGCTCGTCGTTTGGTATGGCTTCATTCAGCTCCGGTTCCCAACGATCAAGGCGAGTTACATGATCCCCCATGTTGTGCAAAAAAGCGGTTAGCTCCTTCGGTCCTCCGATCGTTGTCAGAAGTAAGTTGGCCGCAGTGTTATCACTCATGGTTATGGCAGCACTGCATAATTCTCTTACTGTCATGCCATCCGTAAGATGCTTTTCTGTGACTGGTGAGTACTCAACCAAGTCATTCTGAGAATAGTGTATGCGGCGACCGAGTTGCTCTTGCCCGGCGTCAATACGGGATAATACCGCGCCACATAGCAGAACTTTAAAAGTGCTCATCATTGGAAAACGTTCTTCGGGGCGAAAACTCTCAAGGATCTTACCGCTGTTGAGATCCAGTTCGATGTAACCCACTCGTGCACCCAACTGATCTTCAGCATCTTTTACTTTCACCAGCGTTTCTGGGTGAGCAAAAACAGGAAGGCAAAATGCCGCAAAAAAGGGAATAAGGGCGACACGGAAATGTTGAATACTCATACTCTTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTATTGTCTCATGAGCGGATACATATTTGAATGTATTTAGAAAAATAAACAAATAGGGGTTCCGCGCACATTTCCCCGAAAAGTGCCACCTGACGTCTAAGAAACCATTATTATCATGACATTAACCTATAAAAATAGGCGTATCACGAGGCAGAATTTCAGATAAAAAAAATCCTTAGCTTTCGCTAAGGATGATTTCTGGAATTCGCGGCCGCATCTAGAG";
        let expected_sequence_parts = vec![
            "T",
            "T",
            "G",
            "A",
            "C",
            "G",
            "GCTAGCTCAG",
            "T",
            "CCT",
            "A",
            "GG",
            "T",
            "A",
            "C",
            "A",
            "G",
            big_part,
        ];

        let expected_sequence = expected_sequence_parts.join("");
        assert_eq!(result, expected_sequence);

        let part1 = "T";
        let part3 = "T";
        let part4_5 = vec!["G", "T"];
        let part6 = "A";
        let part7_8 = vec!["C", "T"];
        let part9_10 = vec!["A", "G"];
        let part11 = "GCTAGCTCAG";
        let part12_13 = vec!["T", "C"];
        let part14 = "CCT";
        let part15_16 = vec!["A", "T"];
        let part17 = "GG";
        let part18_19 = vec!["T", "G"];
        let part20 = "A";
        let part21_22 = vec!["T", "C"];
        let part23_24 = vec!["A", "T"];
        let part25_26 = vec!["A", "G"];

        let mut expected_sequences = HashSet::new();
        for part_a in &part4_5 {
            for part_b in &part7_8 {
                for part_c in &part9_10 {
                    for part_d in &part12_13 {
                        for part_e in &part15_16 {
                            for part_f in &part18_19 {
                                for part_g in &part21_22 {
                                    for part_h in &part23_24 {
                                        for part_i in &part25_26 {
                                            let expected_sequence_parts1 = vec![
                                                part1, part3, part_a, part6, part_b, part_c,
                                                part11, part_d, part14, part_e, part17, part_f,
                                                part20, part_g, part_h, part_i, big_part,
                                            ];
                                            let temp_sequence1 = expected_sequence_parts1.join("");
                                            let expected_sequence_parts2 = vec![
                                                part3, part_a, part6, part_b, part_c, part11,
                                                part_d, part14, part_e, part17, part_f, part20,
                                                part_g, part_h, part_i, big_part,
                                            ];
                                            let temp_sequence2 = expected_sequence_parts2.join("");
                                            expected_sequences.insert(temp_sequence1);
                                            expected_sequences.insert(temp_sequence2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(all_sequences.len(), 1024);
        assert_eq!(all_sequences, expected_sequences);

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 28);
    }

    #[test]
    fn test_import_aa_gfa() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/aa.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "124"],
        )[0]
        .clone();

        let result = path.sequence(conn).unwrap();
        assert_eq!(result, "AA");

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(all_sequences, HashSet::from_iter(vec!["AA".to_string()]));

        let node_count = Node::query(conn, "select * from nodes", rusqlite::params!()).len() as i64;
        assert_eq!(node_count, 4);
    }

    #[test]
    fn test_imports_gfa_with_cycle() {
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/cycle_no_path.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAACCCTTTGGGACTCTA".to_string()])
        );
    }

    #[test]
    fn test_breaks_cycle_using_path_node() {
        // here the fixture has a path indicting the cycle starts in the middle of where it would
        // normally be created
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/cycle_with_path.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["TTTGGGACTCTAAAACCC".to_string()])
        );
    }
}
