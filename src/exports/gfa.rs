use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use gen_core::{HashId, Workspace, is_terminal, strand::Strand};
use gen_graph::{GenGraph, GraphNode, project_path};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::BlockGroupEdge,
    db::GraphConnection,
    edge::{Edge, EdgeError},
    errors::{PathError, SequenceError},
    path::Path,
    sample::Sample,
};
use itertools::Itertools;
use thiserror::Error;

use crate::gfa::{Link, Path as GFAPath, Segment, path_line, write_links, write_segments};

#[derive(Debug, Error)]
pub enum GfaExportError {
    #[error("I/O error while exporting GFA: {0}")]
    Io(#[from] std::io::Error),
    #[error("Sequence error while exporting GFA: {0}")]
    Sequence(#[from] SequenceError),
    #[error("Path error while exporting GFA: {0}")]
    Path(#[from] PathError),
    #[error("Path error while exporting GFA for path '{path_name}': {source}")]
    PathForName {
        path_name: String,
        #[source]
        source: PathError,
    },
    #[error("Edge error while exporting GFA: {source}")]
    Edge {
        #[source]
        source: EdgeError,
    },
    #[error("No block groups found for collection {collection_name} and sample {sample_name}")]
    MissingBlockGroups {
        collection_name: String,
        sample_name: String,
    },
}

pub fn export_gfa(
    conn: &GraphConnection,
    workspace: &Workspace,
    collection_name: &str,
    filename: &PathBuf,
    sample_name: &str,
    max_size: impl Into<Option<i64>>,
    history_ref: Option<&str>,
) -> Result<(), GfaExportError> {
    let chunk_size = max_size.into().unwrap_or(i64::MAX);
    // General note about how we encode segment IDs.  The node ID and the start coordinate in the
    // sequence are all that's needed, because the end coordinate can be inferred from the length of
    // the segment's sequence.  So the segment ID is of the form <node ID>.<start coordinate>
    let mut edge_set = HashSet::new();
    let sample_block_groups =
        Sample::get_block_groups(conn, collection_name, sample_name, history_ref);
    if sample_block_groups.is_empty() {
        return Err(GfaExportError::MissingBlockGroups {
            collection_name: collection_name.to_string(),
            sample_name: sample_name.to_string(),
        });
    }
    let mut blocks = vec![];
    let mut seen_blocks = HashSet::new();
    for block_group in sample_block_groups {
        let block_group_edges =
            BlockGroupEdge::edges_for_block_group(conn, &block_group.id, history_ref);
        for mut block in Edge::blocks_from_edges(
            conn,
            workspace,
            &block_group.id,
            &block_group_edges,
            history_ref,
        )
        .map_err(|source| GfaExportError::Edge { source })?
        {
            if seen_blocks.insert((block.node_id, block.start, block.end)) {
                block.id = blocks.len() as i64;
                blocks.push(block);
            }
        }
        edge_set.extend(block_group_edges);
    }

    let edges = edge_set.into_iter().collect::<Vec<_>>();

    blocks.sort_by_key(|a| a.node_id);

    let (gen_graph, _edges_by_node_pair) = Edge::build_graph(&edges, &blocks);

    // Create GenGraph from the built graph
    let mut graph = GenGraph::new();
    graph.extend(
        gen_graph
            .all_edges()
            .map(|(src, dest, weight)| (src, dest, weight.clone())),
    );

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    let mut segments = BTreeSet::new();
    let mut split_segments = HashMap::new();
    for block in &blocks {
        // A deletion's node spells nothing, so it is exported as no segment; links bridge it.
        if !is_terminal(block.node_id) && block.start != block.end {
            if block.end - block.start > chunk_size {
                let mut sub_segments = vec![];
                let block_sequence = block.sequence();
                for (index, sub_start) in (block.start..block.end)
                    .step_by(chunk_size as usize)
                    .enumerate()
                {
                    let sub_end = (sub_start + chunk_size).min(block.end);
                    let seq_start = index as i64 * chunk_size;
                    let seq_end =
                        ((index as i64 + 1) * chunk_size).min(block_sequence.len() as i64);
                    segments.insert(Segment {
                        sequence: block_sequence[seq_start as usize..seq_end as usize].to_string(),
                        node_id: block.node_id,
                        sequence_start: sub_start,
                        sequence_end: sub_end,
                        // NOTE: We can't easily get the value for strand, but it doesn't matter
                        // because this value is only used for writing segments
                        strand: Strand::Forward,
                    });
                    sub_segments.push((sub_start, sub_end));
                }
                split_segments.insert(block.node_id, sub_segments);
            } else {
                segments.insert(Segment {
                    sequence: block.sequence(),
                    node_id: block.node_id,
                    sequence_start: block.start,
                    sequence_end: block.end,
                    // NOTE: We can't easily get the value for strand, but it doesn't matter
                    // because this value is only used for writing segments
                    strand: Strand::Forward,
                });
            }
        }
    }

    let mut links = BTreeSet::new();
    for (source, target, source_strand, target_strand) in segment_links(&graph) {
        if !is_terminal(source.node_id) && !is_terminal(target.node_id) {
            let source_segment = if let Some(splits) = split_segments.get(&source.node_id) {
                let last_split = splits.last().unwrap();
                Segment {
                    sequence: "".to_string(),
                    node_id: source.node_id,
                    sequence_start: last_split.0,
                    sequence_end: last_split.1,
                    strand: source_strand,
                }
            } else {
                Segment {
                    sequence: "".to_string(),
                    node_id: source.node_id,
                    sequence_start: source.sequence_start,
                    sequence_end: source.sequence_end,
                    strand: source_strand,
                }
            };

            let target_segment = if let Some(splits) = split_segments.get(&target.node_id) {
                let first_split = splits.first().unwrap();
                Segment {
                    sequence: "".to_string(),
                    node_id: target.node_id,
                    sequence_start: first_split.0,
                    sequence_end: first_split.1,
                    strand: source_strand,
                }
            } else {
                Segment {
                    sequence: "".to_string(),
                    node_id: target.node_id,
                    sequence_start: target.sequence_start,
                    sequence_end: target.sequence_end,
                    strand: target_strand,
                }
            };

            links.insert(Link {
                source_segment_id: source_segment.segment_id(),
                source_strand,
                target_segment_id: target_segment.segment_id(),
                target_strand,
            });
        }
    }

    for (node_id, splits) in split_segments.iter() {
        for ((src_start, src_end), (dst_start, dst_end)) in splits.iter().tuple_windows() {
            let left = Segment {
                sequence: "".to_string(),
                node_id: *node_id,
                sequence_start: *src_start,
                sequence_end: *src_end,
                strand: Strand::Forward,
            };
            let right = Segment {
                sequence: "".to_string(),
                node_id: *node_id,
                sequence_start: *dst_start,
                sequence_end: *dst_end,
                strand: Strand::Forward,
            };
            links.insert(Link {
                source_segment_id: left.segment_id(),
                source_strand: Strand::Forward,
                target_segment_id: right.segment_id(),
                target_strand: Strand::Forward,
            });
        }
    }

    let paths = get_paths(conn, collection_name, sample_name, &graph, &split_segments)?;
    write_segments(&mut writer, &segments.iter().collect::<Vec<&Segment>>())?;
    write_links(&mut writer, &links.iter().collect::<Vec<&Link>>())?;
    write_paths(&mut writer, paths)?;

    Ok(())
}

/// Whether `node` is a deletion's node, which spells nothing and so is no segment.
fn is_deletion_node(node: &GraphNode) -> bool {
    !is_terminal(node.node_id) && node.sequence_start == node.sequence_end
}

/// The links between the nodes that are segments, with their strands. A link into a chain of
/// deletion nodes is bridged to every node the chain leads to, keeping the strand the route
/// leaves its source on and the one it enters its target on. The bridges show the routes the
/// deletions leave in place; which deletions a route passes through is not recorded in GFA.
fn segment_links(graph: &GenGraph) -> BTreeSet<(GraphNode, GraphNode, Strand, Strand)> {
    let mut links = BTreeSet::new();
    for (source, target, edge_info) in graph.all_edges() {
        if is_deletion_node(&source) {
            continue;
        }
        let source_strand = edge_info[0].source_strand;
        let mut pending = vec![(target, edge_info[0].target_strand)];
        let mut visited = HashSet::new();
        while let Some((node, strand)) = pending.pop() {
            if !is_deletion_node(&node) {
                links.insert((source, node, source_strand, strand));
                continue;
            }
            if !visited.insert(node) {
                continue;
            }
            for (_, next, next_info) in graph.edges(node) {
                pending.push((next, next_info[0].target_strand));
            }
        }
    }
    links
}

fn get_paths(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    graph: &GenGraph,
    split_segments: &HashMap<HashId, Vec<(i64, i64)>>,
) -> Result<HashMap<String, Vec<(String, Strand)>>, GfaExportError> {
    let paths = Path::query_for_collection_and_sample(conn, collection_name, sample_name);

    let mut path_links: HashMap<String, Vec<(String, Strand)>> = HashMap::new();

    for path in paths {
        let block_group = match BlockGroup::get_by_id(conn, &path.block_group_id, None) {
            Ok(block_group) => block_group,
            Err(_) => {
                return Err(GfaExportError::MissingBlockGroups {
                    collection_name: collection_name.to_string(),
                    sample_name: sample_name.to_string(),
                });
            }
        };
        let sample_name = block_group.sample_name;

        let path_blocks = path.coordinate_blocks(conn, None);
        let projected_path = project_path(graph, &path_blocks)
            .into_iter()
            .filter(|(node, _)| !is_deletion_node(node))
            .collect::<Vec<_>>();
        let spells_nothing = projected_path
            .iter()
            .all(|(node, _)| is_terminal(node.node_id));

        if spells_nothing && !path_blocks.is_empty() {
            println!(
                "Path {name} spells no sequence, and a GFA path needs at least one segment; it is not exported.",
                name = path.name
            );
        } else if !projected_path.is_empty() {
            let full_path_name = if !sample_name.is_empty() {
                format!("{}.{}", path.name, sample_name)
            } else {
                path.name
            };
            path_links.insert(
                full_path_name,
                projected_path
                    .iter()
                    .filter_map(|(node, strand)| {
                        if !is_terminal(node.node_id) {
                            if let Some(splits) = split_segments.get(&node.node_id) {
                                Some(
                                    splits
                                        .iter()
                                        .map(|(start, end)| {
                                            (
                                                format!("{id}.{start}.{end}", id = node.node_id),
                                                *strand,
                                            )
                                        })
                                        .collect::<Vec<_>>(),
                                )
                            } else {
                                Some(vec![(
                                    format!(
                                        "{id}.{ss}.{se}",
                                        id = node.node_id,
                                        ss = node.sequence_start,
                                        se = node.sequence_end
                                    ),
                                    *strand,
                                )])
                            }
                        } else {
                            None
                        }
                    })
                    .flatten()
                    .collect::<Vec<_>>(),
            );
        } else {
            println!(
                "Path {name} is not translatable to current graph.",
                name = path.name
            );
        }
    }
    Ok(path_links)
}

fn write_paths(
    writer: &mut BufWriter<File>,
    path_links: HashMap<String, Vec<(String, Strand)>>,
) -> std::io::Result<()> {
    for (name, links) in path_links.iter() {
        let mut segment_ids = vec![];
        let mut node_strands = vec![];
        for (segment_id, strand) in links.iter() {
            segment_ids.push(segment_id.clone());
            node_strands.push(*strand);
        }
        let path = GFAPath {
            name: name.clone(),
            segment_ids,
            node_strands,
        };
        writer.write_all(&path_line(&path).into_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, path::PathBlock};
    use gen_graph::{GraphEdge, GraphNode};
    use gen_models::{
        annotations::add_annotation,
        block_group::{BlockGroup, BlockGroupChange},
        block_group_edge::BlockGroupEdgeData,
        collection::Collection,
        node::Node,
        region::ResolvedGenRegion,
        sequence::Sequence,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
        graphs::combinatorial_library::parse_library,
        imports::{fasta::import_fasta, gfa::import_gfa},
        test_helpers::{get_sample_bg, setup_block_group, setup_gen},
        updates::{library::update_with_library, sequence::update_with_sequence},
    };

    #[test]
    fn test_simple_export() {
        // Sets up a basic graph and then exports it to a GFA file
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection_name = "test collection";
        Collection::create(conn, collection_name).unwrap();

        Sample::create(
            conn,
            gen_models::sample::NewSample {
                name: Sample::DEFAULT_NAME,
                is_reference: false,
            },
        )
        .unwrap();
        let block_group = BlockGroup::create(
            conn,
            gen_models::block_group::NewBlockGroup {
                collection_name,
                sample_name: Sample::DEFAULT_NAME,
                name: "test block group",
                ..Default::default()
            },
        )
        .unwrap();
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAA")
            .save(conn)
            .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTT")
            .save(conn)
            .unwrap();
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGG")
            .save(conn)
            .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCC")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();

        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            4,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            4,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node3_id,
            4,
            Strand::Forward,
            node4_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node4_id,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let new_block_group_edges = vec![
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
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge5.id,
                chromosome_index: 0,
                phased: 0,
            },
        ];
        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);

        Path::create(
            conn,
            "1234",
            &block_group.id,
            &[edge1.id, edge2.id, edge3.id, edge4.id, edge5.id],
        )
        .unwrap();

        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group.id,
            false,
        )
        .unwrap();

        let temp_dir = tempdir().expect("Couldn't get handle to temp directory");
        let mut gfa_path = PathBuf::from(temp_dir.path());
        gfa_path.push("intermediate.gfa");

        export_gfa(
            conn,
            context.workspace(),
            collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .unwrap();
        // NOTE: Not directly checking file contents because segments are written in random order
        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let block_group2 = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();

        assert_eq!(all_sequences, all_sequences2);

        let paths = Path::query_for_collection(conn, "test collection 2");
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0].sequence(conn, context.workspace(), None).unwrap(),
            "AAAATTTTGGGGCCCC"
        );
    }

    #[test]
    fn test_front_deletion_in_combinatorial_library_exports_every_route() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");
        let fasta_path = fasta_path
            .to_str()
            .expect("should have a UTF-8 FASTA path")
            .to_string();
        let parts_path = parts_path
            .to_str()
            .expect("should have a UTF-8 parts path")
            .to_string();
        let library_path = library_path
            .to_str()
            .expect("should have a UTF-8 library path")
            .to_string();

        import_fasta(
            &context,
            &fasta_path,
            collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        add_annotation(
            &context,
            collection,
            "SITE",
            None,
            Sample::DEFAULT_NAME,
            "m123:7-20",
        )
        .unwrap();
        update_with_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "design",
            "SITE",
            parse_library(&parts_path, &library_path).unwrap(),
            Some(&parts_path),
            Some(&library_path),
        )
        .unwrap();
        update_with_sequence(
            &context, collection, "design", "deleted", "cds1:0-1", "", false,
        )
        .unwrap();

        let block_group = get_sample_bg(conn, collection, "deleted");
        let graph = BlockGroup::get_graph(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group.id,
            None,
        )
        .unwrap();
        let node_ids = graph.nodes().map(|node| node.node_id).collect::<Vec<_>>();
        let sequences = Node::get_sequences_by_node_ids(conn, context.workspace(), &node_ids, None);
        let rendered_sequence = |node: GraphNode| {
            sequences[&node.node_id]
                .get_sequence(node.sequence_start, node.sequence_end)
                .unwrap()
        };
        let deleted_target = graph
            .nodes()
            .find(|node| rendered_sequence(*node) == "TGATAA")
            .expect("should contain the remainder of cds1");
        let original_first_base = graph
            .nodes()
            .find(|node| node.node_id == deleted_target.node_id && rendered_sequence(*node) == "A")
            .expect("should contain the original first base of cds1");
        let upstream_parts = graph
            .nodes()
            .filter(|node| ["AAAA", "CAAC", "TAAT"].contains(&rendered_sequence(*node).as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            upstream_parts.len(),
            3,
            "should contain every combinatorial prefix"
        );

        let temp_dir = tempdir().expect("should create a temporary directory");
        let gfa_path = temp_dir.path().join("front-deletion.gfa");
        export_gfa(
            conn,
            context.workspace(),
            collection,
            &gfa_path,
            "deleted",
            None,
            None,
        )
        .unwrap();
        let gfa_lines = fs::read_to_string(&gfa_path)
            .expect("should read the exported GFA")
            .lines()
            .map(str::to_string)
            .collect::<HashSet<_>>();
        let segment_id = |node: GraphNode| {
            format!(
                "{}.{}.{}",
                node.node_id, node.sequence_start, node.sequence_end
            )
        };
        let link_line = |source: GraphNode, target: GraphNode| {
            format!(
                "L\t{}\t+\t{}\t+\t0M",
                segment_id(source),
                segment_id(target)
            )
        };

        for upstream_part in upstream_parts {
            assert!(
                gfa_lines.contains(&link_line(upstream_part, original_first_base)),
                "each combinatorial prefix should link to the original first base"
            );
            assert!(
                gfa_lines.contains(&link_line(upstream_part, deleted_target)),
                "each combinatorial prefix should link past the deleted base"
            );
        }
        assert!(
            !gfa_lines
                .iter()
                .any(|line| line.starts_with("S\t") && line.split('\t').nth(2) == Some("")),
            "the front deletion should export no empty segment"
        );
    }

    #[test]
    fn test_errors_when_sample_has_no_block_groups() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let temp_dir = tempdir().expect("Couldn't get handle to temp directory");
        let gfa_path = PathBuf::from(temp_dir.path()).join("missing.gfa");

        let result = export_gfa(
            conn,
            context.workspace(),
            "missing",
            &gfa_path,
            Sample::DEFAULT_NAME,
            None,
            None,
        );

        assert!(matches!(
            result,
            Err(GfaExportError::MissingBlockGroups {
                collection_name: _,
                sample_name: _,
            })
        ));
    }

    #[test]
    fn test_splits_nodes() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let (bg_id, _path) = setup_block_group(conn);
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &bg_id,
            false,
        )
        .unwrap();

        let temp_dir = tempdir().expect("Couldn't get handle to temp directory");
        let gfa_path = PathBuf::from(temp_dir.path()).join("split.gfa");

        export_gfa(
            conn,
            context.workspace(),
            "test",
            &gfa_path,
            "test",
            5,
            None,
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let block_group2 = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();

        assert_eq!(all_sequences, all_sequences2);

        let graph = BlockGroup::get_graph(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            None,
        )
        .unwrap();
        let graph_nodes = graph
            .nodes()
            .filter_map(|node| {
                if is_terminal(node.node_id) {
                    None
                } else {
                    Some(node.node_id)
                }
            })
            .collect::<Vec<_>>();
        let node_sequences =
            Node::get_sequences_by_node_ids(conn, context.workspace(), &graph_nodes, None);
        assert!(node_sequences.len() > 1);
        for sequence in node_sequences.values() {
            assert!(
                sequence.length <= 5,
                "Sequence length {l} > 5",
                l = sequence.length
            );
        }
    }

    #[test]
    fn test_simple_round_trip() {
        let context = setup_gen();
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/simple.gfa");
        let collection_name = "test".to_string();
        let conn = context.graph().conn();

        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group_id,
            false,
        )
        .unwrap();

        let temp_dir = tempdir().expect("Couldn't get handle to temp directory");
        let mut gfa_path = PathBuf::from(temp_dir.path());
        gfa_path.push("intermediate.gfa");

        export_gfa(
            conn,
            context.workspace(),
            &collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .unwrap();
        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let block_group2 = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();

        assert_eq!(all_sequences, all_sequences2);
    }

    #[test]
    fn test_anderson_round_trip() {
        let context = setup_gen();
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/anderson_promoters.gfa");
        let collection_name = "anderson promoters".to_string();
        let conn = context.graph().conn();

        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group_id,
            false,
        )
        .unwrap();

        let temp_dir = tempdir().expect("Couldn't get handle to temp directory");
        let mut gfa_path = PathBuf::from(temp_dir.path());
        gfa_path.push("intermediate.gfa");

        export_gfa(
            conn,
            context.workspace(),
            &collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .unwrap();
        let _ = import_gfa(
            &context,
            &gfa_path,
            "anderson promoters 2",
            Sample::DEFAULT_NAME,
        );

        let block_group2 = Collection::get_block_groups(conn, "anderson promoters 2", None)
            .pop()
            .unwrap();
        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();

        assert_eq!(all_sequences, all_sequences2);
    }

    #[test]
    fn test_reverse_strand_round_trip() {
        let context = setup_gen();
        let mut gfa_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gfa_path.push("fixtures/reverse_strand.gfa");
        let collection_name = "test".to_string();
        let conn = context.graph().conn();

        let _ = import_gfa(&context, &gfa_path, &collection_name, Sample::DEFAULT_NAME);

        let block_group_id = BlockGroup::get_id(&collection_name, Sample::DEFAULT_NAME, "", None);
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group_id,
            false,
        )
        .unwrap();

        let temp_dir = tempdir().expect("Couldn't get handle to temp directory");
        let mut gfa_path = PathBuf::from(temp_dir.path());
        gfa_path.push("intermediate.gfa");

        export_gfa(
            conn,
            context.workspace(),
            &collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .unwrap();
        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let block_group2 = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();

        assert_eq!(all_sequences, all_sequences2);
    }

    #[test]
    fn test_sequence_is_split_into_multiple_segments() {
        // Confirm that if edges are added to or from a sequence, that results in the sequence being
        // split into multiple segments in the exported GFA, and that the multiple segments are
        // re-imported as multiple sequences
        let context = setup_gen();
        let conn = context.graph().conn();

        let (block_group_id, path) = setup_block_group(conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn)
            .unwrap();
        let insert_node_id =
            Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(conn, block_group_id, &path, 7, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, crate::test_helpers::test_workspace(), &change).unwrap();

        let augmented_edges = BlockGroupEdge::edges_for_block_group(conn, &block_group_id, None);
        let mut node_ids = HashSet::new();
        let mut edge_ids = HashSet::new();
        for augmented_edge in augmented_edges {
            let edge = &augmented_edge.edge;
            if !is_terminal(edge.source_node_id) {
                node_ids.insert(edge.source_node_id);
            }
            if !is_terminal(edge.target_node_id) {
                node_ids.insert(edge.target_node_id);
            }
            if !is_terminal(edge.source_node_id) && !is_terminal(edge.target_node_id) {
                edge_ids.insert(edge.id);
            }
        }

        // The original 10-length A, T, C, G sequences, plus NNNN
        assert_eq!(node_ids.len(), 5);
        // 3 edges from A sequence -> T sequence, T sequence -> C sequence, C sequence -> G sequence
        // 2 edges to and from NNNN
        // 2 edges healing the reference
        // 7 total
        assert_eq!(edge_ids.len(), 7);

        let node_ids_for_query = node_ids.iter().copied().collect::<Vec<_>>();
        let nodes = Node::select(conn)
            .query_by_ids(node_ids_for_query)
            .expect("should load nodes by id");
        let mut node_hashes = HashSet::new();
        for node in nodes {
            if !is_terminal(node.id) {
                node_hashes.insert(node.sequence_hash);
            }
        }

        // The original 10-length A, T, C, G sequences, plus NNNN
        assert_eq!(node_hashes.len(), 5);

        let temp_dir = tempdir().expect("Couldn't get handle to temp directory");
        let mut gfa_path = PathBuf::from(temp_dir.path());
        gfa_path.push("intermediate.gfa");
        export_gfa(
            conn,
            context.workspace(),
            "test",
            &gfa_path,
            "test",
            None,
            None,
        )
        .unwrap();
        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let block_group2 = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();

        let augmented_edges2 = BlockGroupEdge::edges_for_block_group(conn, &block_group2.id, None);
        let mut node_ids2 = HashSet::new();
        let mut edge_ids2 = HashSet::new();
        for augmented_edge in augmented_edges2 {
            let edge = &augmented_edge.edge;
            if !is_terminal(edge.source_node_id) {
                node_ids2.insert(edge.source_node_id);
            }
            if !is_terminal(edge.target_node_id) {
                node_ids2.insert(edge.target_node_id);
            }
            if !is_terminal(edge.source_node_id) && !is_terminal(edge.target_node_id) {
                edge_ids2.insert(edge.id);
            }
        }

        // The 10-length A and T sequences have now been split in two (showing up as different
        // segments in the exported GFA), so expect two more nodes
        assert_eq!(node_ids2.len(), 7);
        // 3 edges from A sequence -> T sequence, T sequence -> C sequence, C sequence -> G sequence
        // 2 edges to and from NNNN
        // 2 edges healing the reference in NNNN
        // 7 total
        assert_eq!(edge_ids2.len(), 7);

        let node_ids_for_query = node_ids2.iter().copied().collect::<Vec<_>>();
        let nodes2 = Node::select(conn)
            .query_by_ids(node_ids_for_query)
            .expect("should load nodes by id");
        let mut node_hashes2 = HashSet::new();
        for node in nodes2 {
            if !is_terminal(node.id) {
                node_hashes2.insert(node.sequence_hash);
            }
        }

        // The 10-length A and T sequences have now been split in two, but since the T sequences was
        // split in half, there's just one new TTTTT sequence shared by 2 nodes
        assert_eq!(node_hashes2.len(), 6);
    }

    /// Deletion nodes spell no sequence, so the export leaves them out: no empty segment, a
    /// link across them, and paths that step from one sequence segment to the next.
    #[test]
    fn test_export_bridges_deletion_nodes() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let collection = "test";
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        // Two deletions meeting at 4: the path steps through both.
        update_with_sequence(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "first",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context, collection, "first", "second", "m123:2-4", "", false,
        )
        .unwrap();

        let temp_dir = tempdir().expect("should create a temporary directory");
        let gfa_path = temp_dir.path().join("deletions.gfa");
        export_gfa(
            conn,
            context.workspace(),
            collection,
            &gfa_path,
            "second",
            None,
            None,
        )
        .unwrap();
        let gfa = fs::read_to_string(&gfa_path).expect("should read the exported GFA");
        let segments = gfa
            .lines()
            .filter(|line| line.starts_with("S\t"))
            .map(|line| {
                let fields = line.split('\t').collect::<Vec<_>>();
                (fields[1].to_string(), fields[2].to_string())
            })
            .collect::<HashMap<_, _>>();
        assert!(
            segments.values().all(|sequence| !sequence.is_empty()),
            "should export no empty segment"
        );
        let links = gfa
            .lines()
            .filter(|line| line.starts_with("L\t"))
            .map(|line| {
                let fields = line.split('\t').collect::<Vec<_>>();
                (fields[1].to_string(), fields[3].to_string())
            })
            .collect::<HashSet<_>>();
        for (source, target) in &links {
            assert!(
                segments.contains_key(source) && segments.contains_key(target),
                "link {source} -> {target} should join exported segments"
            );
        }
        let mut spelled = HashSet::new();
        for path_line in gfa.lines().filter(|line| line.starts_with("P\t")) {
            let steps = path_line
                .split('\t')
                .nth(2)
                .unwrap()
                .split(',')
                .map(|step| step.trim_end_matches(['+', '-']))
                .collect::<Vec<_>>();
            spelled.insert(
                steps
                    .iter()
                    .map(|step| segments[*step].as_str())
                    .collect::<String>(),
            );
            for (source, target) in steps.iter().tuple_windows() {
                let pair = (source.to_string(), target.to_string());
                assert!(
                    links.contains(&pair),
                    "path step {pair:?} should have a link"
                );
            }
        }
        assert!(
            spelled.contains("ATCGATCGATCGATCGGGAACACACAGAGA"),
            "should export the path through both deletions, among {spelled:?}"
        );
    }

    /// Every route into a chain of deletion nodes links to every route out of it: two sources
    /// and three targets around two chained deletions give six links, and none touch a deletion.
    #[test]
    fn test_segment_links_join_every_route_across_deletion_nodes() {
        let node = |name: &str, length: i64| GraphNode {
            node_id: HashId::convert_str(name),
            sequence_start: 0,
            sequence_end: length,
        };
        let weight = || {
            vec![GraphEdge {
                edge_id: HashId::convert_str("edge"),
                source_strand: Strand::Forward,
                target_strand: Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            }]
        };
        let sources = [node("left", 3), node("other left", 2)];
        let targets = [node("right", 4), node("middle", 1), node("other right", 5)];
        let (first_deletion, second_deletion) = (node("first", 0), node("second", 0));
        let mut graph = GenGraph::new();
        for source in sources {
            graph.add_edge(source, first_deletion, weight());
        }
        graph.add_edge(first_deletion, second_deletion, weight());
        for target in targets {
            graph.add_edge(second_deletion, target, weight());
        }

        let links = segment_links(&graph)
            .into_iter()
            .map(|(source, target, _, _)| (source, target))
            .collect::<HashSet<_>>();
        let expected = sources
            .into_iter()
            .cartesian_product(targets)
            .collect::<HashSet<_>>();
        assert_eq!(links, expected);
    }
}
