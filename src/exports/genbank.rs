#![allow(warnings)]
use std::{borrow::Cow, collections::HashSet, fs::File, hash::Hash, iter::zip, path::PathBuf, str};

use gb_io::{self, seq::Location};
use gen_core::{is_terminal, path::PathBlock};
use gen_graph::{GenGraph, GraphEdge, GraphNode, all_simple_paths};
use gen_models::{block_group::BlockGroup, db::GraphConnection, node::Node, sample::Sample};
use itertools::Itertools;
use petgraph::{prelude::DiGraphMap, visit::Dfs};
use rusqlite::{self, Connection};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenbankExportError {
    #[error("I/O error while exporting GenBank: {0}")]
    Io(#[from] std::io::Error),
}

fn merge_nodes(nodes: &[GraphNode]) -> Vec<GraphNode> {
    // This is purposefully not sorted, as the input may be a path of nodes from a path where
    // nodes are disordered. The purpose of this function is to merge a vector of nodes in
    // the order a path is traversed, and to combine any nodes that happen to be contiguous.
    let mut merged = vec![*nodes.first().unwrap()];
    if nodes.len() > 1 {
        for node in nodes[1..].iter() {
            let last_node = merged.last().unwrap();
            if node.node_id == last_node.node_id && node.sequence_start == last_node.sequence_end {
                merged.last_mut().unwrap().sequence_end = node.sequence_end;
            } else {
                merged.push(*node);
            }
        }
    }
    merged
}

fn get_path_nodes(graph: &GenGraph, path_blocks: &[PathBlock]) -> Vec<GraphNode> {
    // From a graph, return graph nodes that traverse a given path. The approach here
    // is to create a reduced graph containing only nodes present in the path. These nodes
    // may not be the ones we want, however, as nodes can be reused in a graph. So we traverse
    // the graph, and match nodes along with the reduced graph. If the traversed set of nodes
    // is an exact match for the path, we return it. Note there can possibly be multiple traversals
    // that satisfy the path, but we just return 1.

    // first, reduce the graph down to nodes and connections that are possible
    let mut path_nodes = HashSet::new();
    let mut path_edges = HashSet::new();
    let mut path_it = path_blocks.iter().peekable();
    while let Some(block) = path_it.next() {
        path_nodes.insert(block.node_id);
        if let Some(next_block) = path_it.peek() {
            path_nodes.insert(next_block.node_id);
            path_edges.insert((block.node_id, next_block.node_id));
        }
    }

    let mut path_graph = GenGraph::new();
    for node in graph.nodes() {
        if path_nodes.contains(&node.node_id) {
            path_graph.add_node(node);
            for (_src_node, target_node, edges) in graph.edges(node) {
                // we include self-node connections because nodes are often split in the graph creation, so this ensures
                // connections like node_1 -> node_1 are kept since they will not be path edges.
                if (node.node_id == target_node.node_id
                    && node.sequence_end == target_node.sequence_start)
                    || (path_edges.contains(&(node.node_id, target_node.node_id)))
                {
                    path_graph.add_edge(node, target_node, edges.clone());
                }
            }
        }
    }

    let first_block = path_blocks.first().unwrap();
    let last_block = path_blocks.last().unwrap();
    let start_nodes = path_graph
        .nodes()
        .filter(|node| {
            if node.node_id == first_block.node_id {
                return node.sequence_start <= first_block.sequence_start
                    && node.sequence_end <= first_block.sequence_end;
            }
            false
        })
        .collect::<Vec<GraphNode>>();
    let end_nodes = path_graph
        .nodes()
        .filter(|node| {
            if node.node_id == last_block.node_id {
                return node.sequence_end >= last_block.sequence_end
                    && node.sequence_start >= last_block.sequence_start
                    && node.sequence_start <= last_block.sequence_end;
            }
            false
        })
        .collect::<Vec<GraphNode>>();

    for start_node in start_nodes.iter() {
        for end_node in end_nodes.iter() {
            for node_path in all_simple_paths(&path_graph, *start_node, *end_node) {
                let merged_path = merge_nodes(&node_path);
                let mut invalid = false;
                for (putative_path_node, path_block) in zip(merged_path, path_blocks) {
                    if !(putative_path_node.sequence_start <= path_block.sequence_start
                        && putative_path_node.sequence_end == path_block.sequence_end)
                    {
                        invalid = true;
                        break;
                    }
                }
                if !invalid {
                    return node_path;
                }
            }
        }
    }

    vec![]
}

pub fn export_genbank(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    filename: &PathBuf,
) -> Result<(), GenbankExportError> {
    // GenBank don't really support graph like structures. Programs like Geneious use features to
    // mark where changes have occurred, and for now we replicate this approach. However, we are
    // only able to show one alternative path. The assumption is GenBank will predominantly be used
    // for haploid organisms and plasmids.

    // To carry out the export and mark engineering, we find the paths for a sample, and identify
    // all places that diverge from the path. The initial genbank import is setup so the path matches
    // the unmodified sequence with changes to it implemented as new graph edges. So when our graph
    // has a connection point that is not in that path, we traverse the new node until we enter the
    // path again and record this change in the sequence. Because GenBank files generally represent
    // the fully engineered sequence, all these changes to the path are incorporated in to the final
    // sequence returned. Once again, we assume there is only one graph bubble when we encounter
    // them, so there is only 1 change to represent. We do not guard against this being an incorrect
    // assumption.
    let block_groups = Sample::get_block_groups(conn, collection_name, sample_name);

    let file = File::create(filename)?;
    let mut writer = gb_io::writer::SeqWriter::new(file);

    for block_group in block_groups.iter() {
        let path = BlockGroup::get_current_path(conn, &block_group.id);
        let path_blocks = path
            .blocks(conn)
            .into_iter()
            .filter(|block| !is_terminal(block.node_id))
            .collect::<Vec<_>>();
        let mut seq = gb_io::seq::Seq::empty();
        seq.name = Some(block_group.name.clone());
        seq.seq = path.sequence(conn).into_bytes();

        // Identify the node traversal corresponding to our path.
        let graph = BlockGroup::get_graph(conn, &block_group.id);
        let path_nodes = get_path_nodes(&graph, &path_blocks);
        let path_node_set: HashSet<&GraphNode> = HashSet::from_iter(&path_nodes);
        let mut node_it = path_nodes.iter().peekable();

        let mut position = 0;
        let mut offset = 0;

        // current_node and next_node correspond to the nodes in our path traversal.
        while let Some(current_node) = node_it.next() {
            position += current_node.length();

            // we evaluate all edges from our node, and if the connection point is not the expected
            // next node of the path, it's a bubble and a change we incorporate.
            for (_source_node, target_node, _edges) in graph.edges(*current_node) {
                if let Some(next_node) = node_it.peek()
                    && &&target_node != next_node
                {
                    // To trace out the bubble, we do a simple DFS until we are back in our path,
                    // as genbank can't support graphs we assume there is simple engineering
                    // here with only 1 alternative path
                    let mut sub_path = vec![];
                    let mut dfs = Dfs::new(&graph, target_node);
                    let mut reentry_node = None;
                    while let Some(nx) = dfs.next(&graph) {
                        if path_node_set.contains(&nx) {
                            reentry_node = Some(nx);
                            break;
                        }
                        sub_path.push(nx)
                    }

                    let mut sequence = String::new();
                    for sub_node in sub_path.iter() {
                        let seqs = Node::get_sequences_by_node_ids(conn, &[sub_node.node_id]);
                        let seq = &seqs[&sub_node.node_id];
                        sequence.push_str(
                            &seq.get_sequence(sub_node.sequence_start, sub_node.sequence_end),
                        );
                    }
                    let mut qualifiers = vec![];

                    let upos = (position + offset) as usize;
                    let mut location = None;

                    // we did an insertion/replacement
                    if target_node.node_id != current_node.node_id {
                        // to distinguish between a replacement and an insertion, we look at the
                        // next node after our target node. If it is the same as our next_node, it's
                        // an insertion. Otherwise, it's a replacement. The 2 events look like this:
                        // A is current_node, B/A is next_node, C is target_node
                        // Insertion:
                        //        A
                        //        | \
                        //        |  C
                        //        | /
                        //        A
                        // Replacement:
                        //        A
                        //       / \
                        //      B   C
                        //       \ /
                        //        A
                        if let Some(entry_node) = reentry_node {
                            location = Some(
                                seq.range_to_location(upos as i64, (upos + sequence.len()) as i64),
                            );
                            if entry_node == **next_node {
                                offset += sequence.len() as i64;
                                seq.seq
                                    .splice(upos..upos, sequence.into_bytes())
                                    .collect::<Vec<_>>();
                                qualifiers.push((
                                    Cow::Borrowed("note"),
                                    Some("Geneious type: Editing History Insertion".to_string()),
                                ));
                                qualifiers.push((Cow::Borrowed("Original_Bases"), None));
                            } else {
                                let end_pos = upos + next_node.length() as usize;
                                offset += sequence.len() as i64 - next_node.length();
                                let original_bases = seq
                                    .seq
                                    .splice(upos..end_pos, sequence.into_bytes())
                                    .collect::<Vec<u8>>();
                                qualifiers.push((
                                    Cow::Borrowed("note"),
                                    Some("Geneious type: Editing History Replacement".to_string()),
                                ));
                                qualifiers.push((
                                    Cow::Borrowed("Original_Bases"),
                                    Some(str::from_utf8(&original_bases).unwrap().to_string()),
                                ));
                            }
                        } else {
                            panic!("unsupported. Maybe insert at end of sequence?");
                        }
                    } else if target_node.node_id == current_node.node_id
                        && target_node.sequence_start != current_node.sequence_end
                    {
                        // if we're not contiguous, it's a deletion
                        offset -= next_node.length();
                        let original_bases = seq
                            .seq
                            .splice(
                                upos..upos + next_node.length() as usize,
                                sequence.into_bytes(),
                            )
                            .collect::<Vec<_>>();
                        // range_to_location always returns a Location::Join, whereas we want location::between. However, since this method
                        // handles circles/linear/etc. we use it to find the location and then convert it to a between.
                        let (ls, le) = seq
                            .range_to_location(upos as i64, (upos + 1) as i64)
                            .find_bounds()
                            .unwrap();
                        location = Some(Location::Between(ls - 1, le - 1));
                        qualifiers.push((
                            Cow::Borrowed("note"),
                            Some("Geneious type: Editing History Deletion".to_string()),
                        ));
                        qualifiers.push((
                            Cow::Borrowed("Original_Bases"),
                            Some(str::from_utf8(&original_bases).unwrap().to_string()),
                        ));
                    }
                    if let Some(l) = location {
                        seq.features.push(gb_io::seq::Feature {
                            kind: Cow::Borrowed("misc_feature"),
                            location: l,
                            qualifiers,
                        });
                    } else {
                        println!("We are unable to determine the type of edit being exported.");
                    }
                }
            }
        }

        writer.write(&seq)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use std::{io, io::BufReader, path::PathBuf, str};

    use gb_io::reader;
    use gen_core::{HashId, strand::Strand::Forward};
    use gen_models::{
        file_types::FileTypes,
        metadata,
        operations::{OperationFile, OperationInfo},
    };
    use tempfile;

    use super::*;
    use crate::{
        imports::genbank::{GenBankImportOptions, import_genbank},
        test_helpers::setup_gen,
        track_database,
    };

    fn compare_genbanks(a: &PathBuf, b: &PathBuf) {
        let a = reader::parse_file(a).unwrap();
        let a_seq = str::from_utf8(&a[0].seq).unwrap().to_string();
        let b = reader::parse_file(b).unwrap();
        let b_seq = str::from_utf8(&b[0].seq).unwrap().to_string();
        assert_eq!(a_seq, b_seq);

        let mut a_features = vec![];
        for feature in a[0].features.iter() {
            for (k, v) in feature.qualifiers.iter() {
                if k == "note"
                    && let Some(v) = v
                    && v.starts_with("Geneious type: Editing")
                {
                    let original_bases = &feature
                        .qualifiers
                        .iter()
                        .filter(|(k, _v)| k == "Original_Bases")
                        .map(|(_k, v)| v.clone())
                        .collect::<Option<String>>();
                    a_features.push((
                        feature.location.find_bounds().unwrap(),
                        original_bases.clone(),
                        v.clone(),
                    ))
                }
            }
        }

        let mut b_features = vec![];
        for feature in b[0].features.iter() {
            for (k, v) in feature.qualifiers.iter() {
                if k == "note"
                    && let Some(v) = v
                    && v.starts_with("Geneious type: Editing")
                {
                    let original_bases = &feature
                        .qualifiers
                        .iter()
                        .filter(|(k, _v)| k == "Original_Bases")
                        .map(|(_k, v)| v.clone())
                        .collect::<Option<String>>();
                    b_features.push((
                        feature.location.find_bounds().unwrap(),
                        original_bases.clone(),
                        v.clone(),
                    ))
                }
            }
        }
        assert_eq!(a_features, b_features);
    }

    #[test]
    fn test_import_then_export_insertion() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/insertion.gb");
        let file = File::open(&path).unwrap();
        let operation = import_genbank(
            &context,
            BufReader::new(file),
            None,
            Sample::DEFAULT_NAME,
            OperationInfo {
                files: vec![OperationFile {
                    file_path: path.to_str().unwrap().to_string(),
                    file_type: FileTypes::GenBank,
                }],
                description: "test".to_string(),
            },
            GenBankImportOptions::default(),
        )
        .unwrap();
        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.gb");
        export_genbank(conn, "", Sample::DEFAULT_NAME, &filename).unwrap();
        compare_genbanks(&path, &filename);
    }

    #[test]
    fn test_import_then_export_replacement() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/deletion_and_insertion.gb");
        let file = File::open(&path).unwrap();
        let operation = import_genbank(
            &context,
            BufReader::new(file),
            None,
            Sample::DEFAULT_NAME,
            OperationInfo {
                files: vec![OperationFile {
                    file_path: path.to_str().unwrap().to_string(),
                    file_type: FileTypes::GenBank,
                }],
                description: "test".to_string(),
            },
            GenBankImportOptions::default(),
        )
        .unwrap();
        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.gb");
        export_genbank(conn, "", Sample::DEFAULT_NAME, &filename).unwrap();
        compare_genbanks(&path, &filename);
    }

    #[test]
    fn test_import_then_export_multiple_operations() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/multiple_insertions_deletions.gb");
        let file = File::open(&path).unwrap();
        let operation = import_genbank(
            &context,
            BufReader::new(file),
            None,
            Sample::DEFAULT_NAME,
            OperationInfo {
                files: vec![OperationFile {
                    file_path: path.to_str().unwrap().to_string(),
                    file_type: FileTypes::GenBank,
                }],
                description: "test".to_string(),
            },
            GenBankImportOptions::default(),
        )
        .unwrap();
        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.gb");
        export_genbank(conn, "", Sample::DEFAULT_NAME, &filename).unwrap();
        compare_genbanks(&path, &filename);
    }

    #[test]
    fn test_get_path_graph() {
        let mut graph = GenGraph::new();
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 0,
                sequence_end: 10,
            },
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 10,
                sequence_end: 20,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        // second starting point for the graph, this also represents a node that is part of the path, but part of the sequence we don't want to use in our path
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("20"),
                sequence_start: 0,
                sequence_end: 10,
            },
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 10,
                sequence_end: 20,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        // represent node_id being split into 3 pieces
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 10,
                sequence_end: 20,
            },
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 20,
                sequence_end: 30,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        // put the same node_id 1 somewhere random in the graph on an edge we don't want to follow
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 0,
                sequence_end: 10,
            },
            GraphNode {
                node_id: HashId::convert_str("30"),
                sequence_start: 0,
                sequence_end: 10,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("30"),
                sequence_start: 0,
                sequence_end: 10,
            },
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 10,
                sequence_end: 20,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 10,
                sequence_end: 20,
            },
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 20,
                sequence_end: 30,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        // final part of path block
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("10"),
                sequence_start: 20,
                sequence_end: 30,
            },
            GraphNode {
                node_id: HashId::convert_str("20"),
                sequence_start: 30,
                sequence_end: 40,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("20"),
                sequence_start: 30,
                sequence_end: 40,
            },
            GraphNode {
                node_id: HashId::convert_str("20"),
                sequence_start: 40,
                sequence_end: 60,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("20"),
                sequence_start: 30,
                sequence_end: 40,
            },
            GraphNode {
                node_id: HashId::convert_str("40"),
                sequence_start: 40,
                sequence_end: 60,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        let path_blocks = vec![
            PathBlock {
                node_id: HashId::convert_str("10"),
                block_sequence: String::new(),
                sequence_start: 0,
                sequence_end: 30,
                path_start: 0,
                path_end: 30,
                strand: Forward,
            },
            PathBlock {
                node_id: HashId::convert_str("20"),
                block_sequence: String::new(),
                sequence_start: 30,
                sequence_end: 60,
                path_start: 30,
                path_end: 60,
                strand: Forward,
            },
        ];
        assert_eq!(
            get_path_nodes(&graph, &path_blocks),
            vec![
                GraphNode {
                    node_id: HashId::convert_str("10"),
                    sequence_start: 0,
                    sequence_end: 10
                },
                GraphNode {
                    node_id: HashId::convert_str("10"),
                    sequence_start: 10,
                    sequence_end: 20
                },
                GraphNode {
                    node_id: HashId::convert_str("10"),
                    sequence_start: 20,
                    sequence_end: 30
                },
                GraphNode {
                    node_id: HashId::convert_str("20"),
                    sequence_start: 30,
                    sequence_end: 40
                },
                GraphNode {
                    node_id: HashId::convert_str("20"),
                    sequence_start: 40,
                    sequence_end: 60
                }
            ]
        )
    }

    #[test]
    fn test_get_path_graph_single_path_block() {
        let mut graph = GenGraph::new();
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("3"),
                sequence_start: 0,
                sequence_end: 1425,
            },
            GraphNode {
                node_id: HashId::convert_str("3"),
                sequence_start: 2220,
                sequence_end: 8302,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("3"),
                sequence_start: 0,
                sequence_end: 1425,
            },
            GraphNode {
                node_id: HashId::convert_str("3"),
                sequence_start: 1425,
                sequence_end: 2220,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        graph.add_edge(
            GraphNode {
                node_id: HashId::convert_str("3"),
                sequence_start: 1425,
                sequence_end: 2220,
            },
            GraphNode {
                node_id: HashId::convert_str("3"),
                sequence_start: 2220,
                sequence_end: 8302,
            },
            vec![GraphEdge {
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                source_strand: Forward,
                target_strand: Forward,
                created_on: 0,
            }],
        );
        let path_blocks = vec![PathBlock {
            node_id: HashId::convert_str("3"),
            block_sequence: String::new(),
            sequence_start: 0,
            sequence_end: 8302,
            path_start: 0,
            path_end: 8302,
            strand: Forward,
        }];
        assert_eq!(
            get_path_nodes(&graph, &path_blocks),
            vec![
                GraphNode {
                    node_id: HashId::convert_str("3"),
                    sequence_start: 0,
                    sequence_end: 1425
                },
                GraphNode {
                    node_id: HashId::convert_str("3"),
                    sequence_start: 1425,
                    sequence_end: 2220
                },
                GraphNode {
                    node_id: HashId::convert_str("3"),
                    sequence_start: 2220,
                    sequence_end: 8302
                }
            ]
        )
    }
}
