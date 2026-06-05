use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path as FilePath,
};

use gen_core::{
    HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, calculate_hash,
    is_end_node, is_start_node,
};
use gen_graph::{GraphEdge, GraphNode};
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::{Edge, EdgeData},
    errors::{
        BlockGroupError, CollectionError, NodeError, OperationError, PathError, SampleError,
        SequenceError,
    },
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
    #[error("Collection creation error: {0}")]
    CollectionError(#[from] CollectionError),
    #[error("Sample creation error: {0}")]
    SampleError(#[from] SampleError),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Path creation error: {0}")]
    PathError(#[from] PathError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Block group creation error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
}

pub fn import_gfa(
    context: &DbContext,
    gfa_path: &FilePath,
    collection_name: &str,
    sample_name: &str,
) -> Result<(Operation, Vec<BlockGroup>), GFAImportError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let mut session = start_operation(conn);
    match Collection::create(conn, collection_name) {
        Ok(_) => {}
        Err(CollectionError::Duplicate(_)) => {}
        Err(e) => return Err(GFAImportError::CollectionError(e)),
    }
    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample_name,
            ..Default::default()
        },
    ) {
        Ok(_) => {}
        Err(e) => {
            return Err(GFAImportError::SampleError(e));
        }
    }

    let bar = progress_bar.add(get_time_elapsed_bar());
    bar.set_message("Parsing GFA");
    let gfa: Gfa<String, (), ()> = Gfa::parse_gfa_file(gfa_path.to_str().unwrap());
    let sn_tags = read_sn_tags(gfa_path);
    bar.finish();

    // Determine block group name per segment: SN:Z: tags > longest path/walk > filename stem
    let bg_name_by_segment: HashMap<String, String> = if !sn_tags.is_empty() {
        gfa.segments
            .iter()
            .map(|seg| {
                (
                    seg.id.clone(),
                    sn_tags.get(&seg.id).cloned().unwrap_or_default(),
                )
            })
            .collect()
    } else if !gfa.paths.is_empty() || !gfa.walk.is_empty() {
        let longest_name = gfa
            .paths
            .iter()
            .map(|p| (p.name.as_str(), p.segments.len()))
            .chain(
                gfa.walk
                    .iter()
                    .map(|w| (w.sample_id.as_str(), w.segments.len())),
            )
            .max_by_key(|&(_, len)| len)
            .map(|(name, _)| name.to_string())
            .unwrap_or_default();
        gfa.segments
            .iter()
            .map(|seg| (seg.id.clone(), longest_name.clone()))
            .collect()
    } else {
        let filename_stem = gfa_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        gfa.segments
            .iter()
            .map(|seg| (seg.id.clone(), filename_stem.clone()))
            .collect()
    };

    // Create one block group per unique name
    let unique_bg_names: IndexSet<String> = bg_name_by_segment.values().cloned().collect();
    let mut block_groups_by_name: HashMap<String, BlockGroup> = HashMap::new();
    for bg_name in &unique_bg_names {
        let bg = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name,
                sample_name,
                name: bg_name,
                ..Default::default()
            },
        )?;
        block_groups_by_name.insert(bg_name.clone(), bg);
    }

    let mut sequences_by_segment_id: HashMap<&String, Sequence> = HashMap::new();
    let mut node_ids_by_segment_id: HashMap<&String, HashId> = HashMap::new();

    let bar = progress_bar.add(get_progress_bar(gfa.segments.len() as u64));
    bar.set_message("Parsing Segments");
    for segment in &gfa.segments {
        let input_sequence = segment.sequence.get_string(&gfa.sequence);
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(input_sequence)
            .save(conn)?;
        sequences_by_segment_id.insert(&segment.id, sequence.clone());
        let node_hash = HashId(calculate_hash(&format!(
            "{collection_name}.{seg_id}:{seq_hash}",
            seg_id = segment.id,
            seq_hash = sequence.hash
        )));
        let node_id = Node::create(conn, &sequence.hash, &node_hash)?;
        node_ids_by_segment_id.insert(&segment.id, node_id);
        bar.inc(1);
    }
    bar.finish();

    // Map node_id → block_group_id for routing edges to the correct block group
    let mut bg_id_by_node_id: HashMap<HashId, HashId> = HashMap::new();
    for (seg_id, node_id) in &node_ids_by_segment_id {
        if let Some(bg_id) = bg_name_by_segment
            .get(*seg_id)
            .and_then(|name| block_groups_by_name.get(name))
            .map(|bg| bg.id)
        {
            bg_id_by_node_id.insert(*node_id, bg_id);
        }
    }

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
    let mut edge_ids_by_data: HashMap<EdgeData, HashId> = HashMap::new();
    let mut edge_data_by_id: HashMap<HashId, EdgeData> = HashMap::new();
    for edge in saved_edges {
        let key = edge_data_from_fields(
            edge.source_node_id,
            edge.source_coordinate,
            edge.source_strand,
            edge.target_node_id,
            edge.target_strand,
        );
        edge_ids_by_data.insert(key, edge.id);
        edge_data_by_id.insert(edge.id, key);
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

        let first_seg = &input_path.segments[0];
        let first_node_id = *node_ids_by_segment_id.get(first_seg).unwrap();
        let bg_id = bg_id_by_node_id
            .get(&first_node_id)
            .copied()
            .unwrap_or_else(|| block_groups_by_name.values().next().unwrap().id);

        BlockGroupEdge::bulk_create(
            conn,
            &path_edge_ids
                .iter()
                .map(|id| BlockGroupEdgeData {
                    block_group_id: bg_id,
                    edge_id: *id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>(),
        );
        Path::create(conn, path_name, &bg_id, &path_edge_ids)?;
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

        let first_seg = &input_walk.segments[0];
        let first_node_id = *node_ids_by_segment_id.get(first_seg).unwrap();
        let bg_id = bg_id_by_node_id
            .get(&first_node_id)
            .copied()
            .unwrap_or_else(|| block_groups_by_name.values().next().unwrap().id);

        BlockGroupEdge::bulk_create(
            conn,
            &path_edge_ids
                .iter()
                .map(|id| BlockGroupEdgeData {
                    block_group_id: bg_id,
                    edge_id: *id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>(),
        );
        Path::create(conn, path_name, &bg_id, &path_edge_ids)?;
    }

    // make any block group edges not in paths or walks
    let leftover_bge: Vec<BlockGroupEdgeData> = edge_ids
        .iter()
        .filter_map(|id| {
            if created_blockgroup_edges.contains(id) {
                return None;
            }
            let edge_data = edge_data_by_id.get(id)?;
            let node_key = if is_start_node(edge_data.source_node_id) {
                edge_data.target_node_id
            } else {
                edge_data.source_node_id
            };
            let bg_id = bg_id_by_node_id
                .get(&node_key)
                .copied()
                .unwrap_or_else(|| block_groups_by_name.values().next().unwrap().id);
            Some(BlockGroupEdgeData {
                block_group_id: bg_id,
                edge_id: *id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
        })
        .collect();
    BlockGroupEdge::bulk_create(conn, &leftover_bge);

    // check each block group graph for cycles and wire up start/end nodes
    let bar = progress_bar.add(get_progress_bar(None));
    bar.set_message("Breaking cycles");
    let message_bar = progress_bar.add(get_message_bar());
    let mut all_new_cycle_edges: Vec<(EdgeData, HashId)> = vec![];
    for bg in block_groups_by_name.values() {
        let graph = BlockGroup::get_graph(conn, &bg.id)?;
        let mut undirected_graph: UnGraphMap<GraphNode, GraphEdge> = UnGraphMap::new();
        for node in graph.nodes() {
            undirected_graph.add_node(node);
        }
        for (src, dst, weights) in graph.all_edges() {
            undirected_graph.add_edge(src, dst, weights[0]);
        }
        let connected_components = kosaraju_scc(&undirected_graph);
        for subgraph in connected_components.iter() {
            if subgraph.len() >= 3 {
                // For graphs with just one enter/exit point, we log a message
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
                    let min_index =
                        order.iter().enumerate().min_set_by_key(|(_, k)| k.node_id)[0].0;
                    order.rotate_left(min_index);
                    bar.inc(1);
                    all_new_cycle_edges.push((
                        edge_data_from_fields(
                            PATH_START_NODE_ID,
                            0,
                            Strand::Forward,
                            order[0].node_id,
                            Strand::Forward,
                        ),
                        bg.id,
                    ));
                    let last_node = order.last().unwrap();
                    all_new_cycle_edges.push((
                        edge_data_from_fields(
                            last_node.node_id,
                            last_node.sequence_end,
                            Strand::Forward,
                            PATH_END_NODE_ID,
                            Strand::Forward,
                        ),
                        bg.id,
                    ));
                    all_new_cycle_edges.push((
                        edge_data_from_fields(
                            PATH_END_NODE_ID,
                            0,
                            Strand::Forward,
                            PATH_START_NODE_ID,
                            Strand::Forward,
                        ),
                        bg.id,
                    ));
                } else if has_start && has_end {
                    // there's a cycle, but has a start/end already. At some point we should track this
                    // so we know ahead of time where the cycles are
                } else {
                    message_bar.set_message(
                        "Path encountered with cycle after start/end node, no cycle breaking will apply.",
                    );
                }
            }
        }
    }
    message_bar.finish();
    let new_edge_data: Vec<EdgeData> = all_new_cycle_edges.iter().map(|(e, _)| *e).collect();
    let new_edge_ids = Edge::bulk_create(conn, &new_edge_data);
    let cycle_bge: Vec<BlockGroupEdgeData> = new_edge_ids
        .iter()
        .zip(all_new_cycle_edges.iter())
        .map(|(id, (_, bg_id))| BlockGroupEdgeData {
            block_group_id: *bg_id,
            edge_id: *id,
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
        })
        .collect();
    BlockGroupEdge::bulk_create(conn, &cycle_bge);
    bar.finish();

    let mut block_groups: Vec<BlockGroup> = block_groups_by_name.into_values().collect();
    block_groups.sort_by(|a, b| a.name.cmp(&b.name));

    let op = end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![
                OperationFile::new(gfa_path.to_str().unwrap().to_string())
                    .set_file_type(FileTypes::GFA),
            ],
            description: "gfa_import".to_string(),
        },
        &format!("Imported GFA {path}", path = gfa_path.to_str().unwrap()),
        None,
    )
    .map_err(GFAImportError::OperationError);
    gen_bar.finish();
    op.map(|op| (op, block_groups))
}

fn read_sn_tags(gfa_path: &FilePath) -> HashMap<String, String> {
    let Ok(file) = File::open(gfa_path) else {
        return HashMap::new();
    };
    let mut sn_tags = HashMap::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if !line.starts_with('S') {
            continue;
        }
        let mut fields = line.splitn(10, '\t');
        fields.next(); // 'S'
        let Some(seg_id) = fields.next() else {
            continue;
        };
        fields.next(); // sequence
        for opt in fields {
            if let Some(sn_value) = opt.strip_prefix("SN:Z:") {
                sn_tags.insert(seg_id.to_string(), sn_value.to_string());
                break;
            }
        }
    }
    sn_tags
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

        let block_group_id =
            BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "m123", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "m123"],
        )[0]
        .clone();

        let result = path.sequence(conn);
        assert_eq!(result.unwrap(), "ATCGATCGATCGATCGATCGGGAACACACAGAGA");

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
    fn test_double_import_gfa_is_idempotent() {
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/simple.gfa");
        let collection_name = "test".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();

        track_database(conn, context.operations().conn()).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let node_count_after_first =
            Node::query(conn, "select * from nodes", rusqlite::params!()).len();

        let second_result = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);
        assert!(
            matches!(
                second_result,
                Err(GFAImportError::OperationError(
                    gen_models::errors::OperationError::NoChanges
                ))
            ),
            "expected NoChanges on duplicate import, got {second_result:?}"
        );

        let node_count_after_second =
            Node::query(conn, "select * from nodes", rusqlite::params!()).len();
        assert_eq!(
            node_count_after_first, node_count_after_second,
            "duplicate import must not create new nodes"
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

        let block_group_id =
            BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "no_path", None);
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

        let block_group_id =
            BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "291344", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "291344"],
        )[0]
        .clone();

        let result = path.sequence(conn);
        assert_eq!(result.unwrap(), "ACCTACAAATTCAAAC");

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

        let block_group_id =
            BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "123", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "124"],
        )[0]
        .clone();

        let result = path.sequence(conn);
        assert_eq!(result.unwrap(), "TATGCCAGCTGCGAATA");

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

        let block_group_id = BlockGroup::get_id(
            &collection_name,
            Sample::DEFAULT_NAME,
            "BBa_J23119#0#BBa_J23119",
            None,
        );
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "BBa_J23100"],
        )[0]
        .clone();

        let result = path.sequence(conn);
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
        assert_eq!(result.unwrap(), expected_sequence);

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

        let block_group_id =
            BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "123", None);
        let path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 AND name = ?2",
            params![block_group_id, "124"],
        )[0]
        .clone();

        let result = path.sequence(conn);
        assert_eq!(result.unwrap(), "AA");

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

        let block_group_id = BlockGroup::get_id(
            &collection_name,
            Sample::DEFAULT_NAME,
            "cycle_no_path",
            None,
        );

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAACCCTTTGGGACTCTA".to_string()])
        );
    }

    #[test]
    fn test_import_disjoint_graphs() {
        // Two separate linear chains with no cross-links; both end up in one block group via shared start/end nodes.
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/disjoint_graphs.gfa");
        let collection_name = "disjoint".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(
            &collection_name,
            Sample::DEFAULT_NAME,
            "disjoint_graphs",
            None,
        );
        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAAACCCC".to_string(), "TTTTGGGG".to_string()])
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

        let block_group_id =
            BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "m123", None);

        let all_sequences = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["TTTGGGACTCTAAAACCC".to_string()])
        );
    }

    // --- Chris' proposed naming cases ---

    #[test]
    fn test_import_gfa_ref_tag_names_block_group() {
        // SN:Z: tag present on all segments; block group should be named by that value.
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/ref_tag_single.gfa");
        let collection_name = "ref_tag_single".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1",
            params![collection_name],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(block_groups[0].name, "chr1");
    }

    #[test]
    fn test_import_gfa_multi_ref_tags_create_multiple_block_groups() {
        // Different SN:Z: values on disjoint subgraphs; one block group per unique SN value.
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/ref_tag_multi.gfa");
        let collection_name = "ref_tag_multi".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1",
            params![collection_name],
        );
        let names: HashSet<String> = block_groups.iter().map(|bg| bg.name.clone()).collect();
        assert_eq!(
            names,
            HashSet::from_iter(vec!["chr1".to_string(), "chr2".to_string()])
        );
    }

    #[test]
    fn test_import_gfa_longest_path_names_block_group() {
        // No SN:Z: tags; block group named by the path with the most segments.
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/paths_no_ref.gfa");
        let collection_name = "paths_no_ref".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1",
            params![collection_name],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(block_groups[0].name, "longer_path");
    }

    #[test]
    fn test_import_gfa_single_graph_no_ref_uses_filename() {
        // No SN:Z: tags, no paths, single connected graph; block group named by GFA filename stem.
        let gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/no_path.gfa");
        let collection_name = "no_path_filename".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1",
            params![collection_name],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(block_groups[0].name, "no_path");
    }

    #[test]
    fn test_import_gfa_multiple_graphs_no_ref_uses_filename() {
        // No SN:Z: tags, no paths, multiple disjoint subgraphs; all in one block group named by filename stem.
        let gfa_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gfa/disjoint_graphs.gfa");
        let collection_name = "disjoint_filename".to_string();
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1",
            params![collection_name],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(block_groups[0].name, "disjoint_graphs");
    }
}
