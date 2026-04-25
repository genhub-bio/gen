#![allow(warnings)]
use std::{
    borrow::Cow,
    cmp::{max, min},
    collections::HashSet,
    fs::File,
    hash::Hash,
    iter::zip,
    path::PathBuf,
    str,
};

use gb_io::{self, seq::Location};
use gen_core::{
    Strand, is_terminal,
    path::PathBlock,
    range::{OrderedMerge, Range, merge_ordered_items},
};
use gen_graph::{GenGraph, GraphEdge, GraphNode, all_simple_paths};
use gen_models::{
    accession::{Accession, AccessionEdge},
    annotations::{Annotation, GenBankLocationOperator},
    block_group::BlockGroup,
    db::GraphConnection,
    node::Node,
    sample::Sample,
    traits::Query,
};
use itertools::Itertools;
use petgraph::{prelude::DiGraphMap, visit::Dfs};
use rusqlite::{self, Connection};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GenbankExportError {
    #[error("I/O error while exporting GenBank: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AnnotationSegment {
    node_id: gen_core::HashId,
    range: Range,
    strand: Strand,
}

impl OrderedMerge for AnnotationSegment {
    fn should_merge_with(&self, next: &Self) -> bool {
        self.strand == next.strand
            && next.range.start >= self.range.start
            && next.range.start <= self.range.end
    }

    fn merge_with(&mut self, next: &Self) {
        self.range.end = max(self.range.end, next.range.end);
    }
}

fn accession_edges_to_segments(edges: &[AccessionEdge]) -> Vec<AnnotationSegment> {
    let mut segments = Vec::new();
    let mut current_node = None;
    let mut current_start = None;
    let mut current_strand = None;

    for edge in edges {
        if edge.source_coordinate < 0 {
            current_node = Some(edge.target_node_id);
            current_start = Some(edge.target_coordinate);
            current_strand = Some(edge.target_strand);
            continue;
        }

        if let (Some(node_id), Some(start), Some(strand)) =
            (current_node, current_start, current_strand)
        {
            let (segment_start, segment_end) = if start <= edge.source_coordinate {
                (start, edge.source_coordinate)
            } else {
                (edge.source_coordinate, start)
            };
            if segment_end > segment_start {
                segments.push(AnnotationSegment {
                    node_id,
                    range: Range {
                        start: segment_start,
                        end: segment_end,
                    },
                    strand,
                });
            }
        }

        if edge.target_coordinate < 0 {
            break;
        }

        current_node = Some(edge.target_node_id);
        current_start = Some(edge.target_coordinate);
        current_strand = Some(edge.target_strand);
    }

    segments
}

fn merge_annotation_segments(segments: Vec<AnnotationSegment>) -> Vec<AnnotationSegment> {
    merge_ordered_items(
        segments
            .into_iter()
            .filter(|segment| segment.range.end > segment.range.start)
            .collect(),
    )
}

fn project_single_annotation_segment(
    segment: &AnnotationSegment,
    path_blocks: &[PathBlock],
) -> Vec<AnnotationSegment> {
    merge_annotation_segments(
        path_blocks
            .iter()
            .filter_map(|block| {
                if block.node_id != segment.node_id {
                    return None;
                }
                let overlap_start = max(segment.range.start, block.sequence_start);
                let overlap_end = min(segment.range.end, block.sequence_end);
                if overlap_end <= overlap_start {
                    return None;
                }

                let (start, end) = if block.strand == Strand::Reverse {
                    (
                        block.path_start + (block.sequence_end - overlap_end),
                        block.path_start + (block.sequence_end - overlap_start),
                    )
                } else {
                    (
                        block.path_start + (overlap_start - block.sequence_start),
                        block.path_start + (overlap_end - block.sequence_start),
                    )
                };
                let strand = if block.strand == Strand::Reverse {
                    segment.strand.complement()
                } else {
                    segment.strand
                };

                Some(AnnotationSegment {
                    node_id: block.node_id,
                    range: Range { start, end },
                    strand,
                })
            })
            .collect(),
    )
}

fn project_annotation_segments(
    accession_segments: &[AnnotationSegment],
    path_blocks: &[PathBlock],
    preserve_part_boundaries: bool,
) -> Vec<AnnotationSegment> {
    let projected = accession_segments
        .iter()
        .flat_map(|segment| project_single_annotation_segment(segment, path_blocks))
        .collect::<Vec<_>>();

    if preserve_part_boundaries {
        projected
    } else {
        merge_annotation_segments(projected)
    }
}

fn build_annotation_location(
    locations: Vec<Location>,
    operator: Option<&GenBankLocationOperator>,
) -> Option<Location> {
    match locations.len() {
        0 => None,
        1 => locations.into_iter().next(),
        _ => Some(match operator {
            Some(GenBankLocationOperator::Join) | None => Location::Join(locations),
            Some(GenBankLocationOperator::Order) => Location::Order(locations),
            Some(GenBankLocationOperator::Bond) => Location::Bond(locations),
            Some(GenBankLocationOperator::OneOf) => Location::OneOf(locations),
        }),
    }
}

fn annotation_location(
    segments: &[AnnotationSegment],
    operator: Option<&GenBankLocationOperator>,
) -> Option<Location> {
    let mut locations = segments
        .iter()
        .filter(|segment| segment.range.end > segment.range.start)
        .map(|segment| Location::simple_range(segment.range.start, segment.range.end))
        .collect::<Vec<_>>();
    if locations.is_empty() {
        return None;
    }

    let strand = segments.first()?.strand;
    let location = build_annotation_location(locations, operator)?;

    Some(if strand == Strand::Reverse {
        Location::Complement(Box::new(location))
    } else {
        location
    })
}

fn export_annotations(
    conn: &GraphConnection,
    path: &gen_models::path::Path,
    path_blocks: &[PathBlock],
    seq: &mut gb_io::seq::Seq,
    sample_name: &str,
) {
    let normalize_qualifier_text =
        |value: &str| value.split_whitespace().collect::<Vec<_>>().join(" ");
    let annotations = Annotation::query_by_sample(conn, sample_name)
        .expect("should load sample annotations for GenBank export");
    for annotation in annotations {
        let Some(accession) = Accession::get_by_id(conn, &annotation.accession_id) else {
            continue;
        };
        if accession.path_id != path.id {
            continue;
        }

        let accession_segments = accession_edges_to_segments(&Accession::get_edges_by_id(
            conn,
            &annotation.accession_id,
        ));
        let genbank_extra = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.genbank.as_ref());
        let projected_segments = project_annotation_segments(
            &accession_segments,
            path_blocks,
            genbank_extra
                .and_then(|extra| extra.location_operator.as_ref())
                .is_some(),
        );
        let Some(location) = annotation_location(
            &projected_segments,
            genbank_extra.and_then(|extra| extra.location_operator.as_ref()),
        ) else {
            continue;
        };
        let kind = genbank_extra
            .map(|extra| Cow::Owned(extra.kind.clone()))
            .unwrap_or_else(|| Cow::Borrowed("misc_feature"));
        let qualifiers = genbank_extra
            .map(|extra| {
                extra
                    .qualifiers
                    .iter()
                    .map(|qualifier| {
                        (
                            Cow::Owned(qualifier.key.clone()),
                            qualifier.value.as_deref().map(normalize_qualifier_text),
                        )
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![(Cow::Borrowed("label"), Some(annotation.name.clone()))]);
        seq.features.push(gb_io::seq::Feature {
            kind,
            location,
            qualifiers,
        });
    }
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
        export_annotations(conn, &path, &path_blocks, &mut seq, sample_name);

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
    use gen_core::{
        HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, is_terminal, strand::Strand::Forward,
    };
    use gen_models::{
        accession::{Accession, AccessionEdge, AccessionEdgeData, AccessionPath},
        annotations::{Annotation, AnnotationExtra, GenBankExtra, GenBankLocationOperator},
        block_group::BlockGroup,
        file_types::FileTypes,
        metadata,
        operations::{OperationFile, OperationInfo},
        path::Path,
    };
    use tempfile;

    use super::*;
    use crate::{
        imports::genbank::{GenBankImportOptions, import_genbank},
        test_helpers::{setup_block_group, setup_gen},
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

    fn feature_label(feature: &gb_io::seq::Feature) -> Option<String> {
        feature
            .qualifiers
            .iter()
            .find_map(|(key, value)| (*key == "label").then(|| value.clone()).flatten())
    }

    fn feature_qualifier(feature: &gb_io::seq::Feature, key: &str) -> Option<String> {
        feature
            .qualifiers
            .iter()
            .find_map(|(qualifier_key, value)| {
                ((*qualifier_key).as_ref() == key)
                    .then(|| value.clone())
                    .flatten()
            })
    }

    fn create_annotation_with_segments(
        conn: &gen_models::db::GraphConnection,
        path: &Path,
        name: &str,
        segments: &[(usize, i64, i64, Strand)],
        operator: Option<GenBankLocationOperator>,
        sample_name: &str,
    ) {
        let blocks = path
            .blocks(conn)
            .into_iter()
            .filter(|block| !is_terminal(block.node_id))
            .collect::<Vec<_>>();
        let first = segments
            .first()
            .expect("should contain at least one annotation segment");
        let mut edges = vec![AccessionEdgeData {
            source_node_id: PATH_START_NODE_ID,
            source_coordinate: -1,
            source_strand: Strand::Forward,
            target_node_id: blocks[first.0].node_id,
            target_coordinate: first.1,
            target_strand: first.3,
            chromosome_index: 0,
        }];
        for window in segments.windows(2) {
            let current = window[0];
            let next = window[1];
            edges.push(AccessionEdgeData {
                source_node_id: blocks[current.0].node_id,
                source_coordinate: current.2,
                source_strand: current.3,
                target_node_id: blocks[next.0].node_id,
                target_coordinate: next.1,
                target_strand: next.3,
                chromosome_index: 0,
            });
        }
        let last = segments
            .last()
            .expect("should contain at least one annotation segment");
        edges.push(AccessionEdgeData {
            source_node_id: blocks[last.0].node_id,
            source_coordinate: last.2,
            source_strand: last.3,
            target_node_id: PATH_END_NODE_ID,
            target_coordinate: -1,
            target_strand: Strand::Forward,
            chromosome_index: 0,
        });

        let accession = Accession::get_or_create(conn, name, &path.id, None);
        let edge_ids = AccessionEdge::bulk_create(conn, &edges);
        AccessionPath::create(conn, &accession.id, &edge_ids);
        Annotation::create_with_samples(
            conn,
            name,
            "export-track",
            &accession.id,
            Some(&AnnotationExtra {
                genbank: Some(GenBankExtra {
                    kind: "misc_feature".to_string(),
                    qualifiers: vec![gen_models::annotations::GenBankQualifier {
                        key: "label".to_string(),
                        value: Some(name.to_string()),
                    }],
                    location_operator: operator,
                }),
                ..AnnotationExtra::default()
            }),
            &[sample_name],
        )
        .unwrap();
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
            GenBankImportOptions::default().annotation_name_from_path(&path),
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
            GenBankImportOptions::default().annotation_name_from_path(&path),
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
            GenBankImportOptions::default().annotation_name_from_path(&path),
        )
        .unwrap();
        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.gb");
        export_genbank(conn, "", Sample::DEFAULT_NAME, &filename).unwrap();
        compare_genbanks(&path, &filename);
    }

    #[test]
    fn test_import_then_export_annotations() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/puc19.gb");
        let file = File::open(&path).unwrap();
        import_genbank(
            &context,
            BufReader::new(file),
            Some("fixtures"),
            "puc19-export",
            OperationInfo {
                files: vec![OperationFile {
                    file_path: path.to_str().unwrap().to_string(),
                    file_type: FileTypes::GenBank,
                }],
                description: "test".to_string(),
            },
            GenBankImportOptions::default().annotation_name_from_path(&path),
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.gb");
        export_genbank(conn, "fixtures", "puc19-export", &filename).unwrap();

        let output = reader::parse_file(&filename).unwrap();
        let features = &output[0].features;
        let labels = features
            .iter()
            .filter_map(feature_label)
            .collect::<HashSet<_>>();
        assert!(labels.contains("AmpR"), "export should include AmpR");
        assert!(
            labels.contains("lac promoter"),
            "export should include lac promoter"
        );
        assert!(labels.contains("ori"), "export should include ori");
        assert!(
            labels.contains("M13 Forward"),
            "export should include reverse-strand annotations"
        );

        let amp_r = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("AmpR"))
            .unwrap();
        assert_eq!(amp_r.kind.as_ref(), "CDS");
        assert_eq!(
            feature_qualifier(amp_r, "product").as_deref(),
            Some("beta-lactamase")
        );

        let lac_promoter = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("lac promoter"))
            .unwrap();
        assert_eq!(
            lac_promoter.location,
            Location::Join(vec![
                Location::simple_range(540, 546),
                Location::simple_range(546, 564),
                Location::simple_range(564, 571),
            ])
        );

        let exported_text = std::fs::read_to_string(&filename).unwrap();
        assert!(
            exported_text.contains(
                "                     /note=\"CAP binding activates transcription in the presence\n                     of cAMP.\""
            ),
            "export should wrap the CAP binding site note without inserting a blank line"
        );
        assert!(
            !exported_text.contains(
                "                     /note=\"CAP binding activates transcription in the presence\n\n                     of cAMP.\""
            ),
            "export should not insert a blank line inside wrapped note text"
        );

        let m13_forward = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("M13 Forward"))
            .unwrap();
        assert_eq!(
            m13_forward.location,
            Location::Complement(Box::new(Location::simple_range(688, 706)))
        );

        let ori = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("ori"))
            .unwrap();
        assert_eq!(
            ori.location,
            Location::Join(vec![
                Location::simple_range(2314, 2686),
                Location::simple_range(0, 217),
            ])
        );
    }

    #[test]
    fn test_export_genbank_annotations_preserves_location_operators() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let (_block_group_id, path) = setup_block_group(conn);

        create_annotation_with_segments(
            conn,
            &path,
            "join-annotation",
            &[(0, 1, 3, Strand::Forward), (1, 2, 5, Strand::Forward)],
            Some(GenBankLocationOperator::Join),
            "test",
        );
        create_annotation_with_segments(
            conn,
            &path,
            "order-annotation",
            &[(0, 4, 6, Strand::Forward), (1, 6, 8, Strand::Forward)],
            Some(GenBankLocationOperator::Order),
            "test",
        );
        create_annotation_with_segments(
            conn,
            &path,
            "bond-annotation",
            &[(1, 1, 2, Strand::Forward), (2, 3, 4, Strand::Forward)],
            Some(GenBankLocationOperator::Bond),
            "test",
        );
        create_annotation_with_segments(
            conn,
            &path,
            "oneof-annotation",
            &[(2, 5, 7, Strand::Forward), (3, 1, 3, Strand::Forward)],
            Some(GenBankLocationOperator::OneOf),
            "test",
        );
        create_annotation_with_segments(
            conn,
            &path,
            "reverse-order-annotation",
            &[(0, 7, 9, Strand::Reverse), (1, 0, 2, Strand::Reverse)],
            Some(GenBankLocationOperator::Order),
            "test",
        );
        create_annotation_with_segments(
            conn,
            &path,
            "reverse-single-annotation",
            &[(3, 4, 7, Strand::Reverse)],
            None,
            "test",
        );

        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.gb");
        export_genbank(conn, "test", "test", &filename).unwrap();

        let output = reader::parse_file(&filename).unwrap();
        let features = &output[0].features;

        let join_annotation = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("join-annotation"))
            .unwrap();
        assert_eq!(
            join_annotation.location,
            Location::Join(vec![
                Location::simple_range(1, 3),
                Location::simple_range(12, 15),
            ])
        );

        let order_annotation = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("order-annotation"))
            .unwrap();
        assert_eq!(
            order_annotation.location,
            Location::Order(vec![
                Location::simple_range(4, 6),
                Location::simple_range(16, 18),
            ])
        );

        let bond_annotation = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("bond-annotation"))
            .unwrap();
        assert_eq!(
            bond_annotation.location,
            Location::Bond(vec![
                Location::simple_range(11, 12),
                Location::simple_range(23, 24),
            ])
        );

        let oneof_annotation = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("oneof-annotation"))
            .unwrap();
        assert_eq!(
            oneof_annotation.location,
            Location::OneOf(vec![
                Location::simple_range(25, 27),
                Location::simple_range(31, 33),
            ])
        );

        let reverse_order_annotation = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("reverse-order-annotation"))
            .unwrap();
        assert_eq!(
            reverse_order_annotation.location,
            Location::Complement(Box::new(Location::Order(vec![
                Location::simple_range(7, 9),
                Location::simple_range(10, 12),
            ])))
        );

        let reverse_single_annotation = features
            .iter()
            .find(|feature| feature_label(feature).as_deref() == Some("reverse-single-annotation"))
            .unwrap();
        assert_eq!(
            reverse_single_annotation.location,
            Location::Complement(Box::new(Location::simple_range(34, 37)))
        );
    }

    #[test]
    fn test_export_genbank_normalizes_legacy_qualifier_newlines() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let (_block_group_id, path) = setup_block_group(conn);

        create_annotation_with_segments(
            conn,
            &path,
            "legacy-note",
            &[(0, 0, 25, Strand::Forward)],
            None,
            "test",
        );

        let annotation = Annotation::query_by_group(conn, "export-track")
            .unwrap()
            .into_iter()
            .find(|annotation| annotation.name == "legacy-note")
            .unwrap();
        let mut extra = annotation.extra.unwrap();
        extra.genbank.as_mut().unwrap().qualifiers = vec![
            gen_models::annotations::GenBankQualifier {
                key: "label".to_string(),
                value: Some("legacy-note".to_string()),
            },
            gen_models::annotations::GenBankQualifier {
                key: "note".to_string(),
                value: Some(
                    "CAP binding activates transcription in the presence\n\nof cAMP.".to_string(),
                ),
            },
        ];
        conn.execute(
            "update annotations set extra = ?1 where id = ?2",
            rusqlite::params![serde_json::to_string(&extra).unwrap(), annotation.id],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.gb");
        export_genbank(conn, "test", "test", &filename).unwrap();

        let exported_text = std::fs::read_to_string(&filename).unwrap();
        assert!(
            exported_text.contains(
                "                     /note=\"CAP binding activates transcription in the presence\n                     of cAMP.\""
            ),
            "export should collapse embedded qualifier newlines to plain spaces before normal wrapping"
        );
        assert!(
            !exported_text.contains(
                "                     /note=\"CAP binding activates transcription in the presence\n\n                     of cAMP.\""
            ),
            "export should not preserve blank lines from legacy stored qualifier text"
        );
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
