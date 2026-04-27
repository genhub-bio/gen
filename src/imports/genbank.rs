use std::{
    cmp::{max, min},
    io::Read,
    path::Path as FsPath,
    str,
};

use gb_io::reader;
use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand,
    range::{OrderedMerge, Range, merge_ordered_items},
};
use gen_models::{
    accession::{Accession, AccessionEdge, AccessionEdgeData, AccessionPath},
    annotations::Annotation,
    block_group::{BlockGroup, NewBlockGroup, PathChange},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::{Collection, CollectionError},
    db::DbContext,
    edge::Edge,
    node::Node,
    operations::{Operation, OperationInfo},
    path::Path,
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
};

use crate::{
    genbank::{
        EditType, GenBankAnnotation, GenBankAnnotationSegment, GenBankEdit, GenBankError,
        process_sequence,
    },
    progress_bar::{add_saving_operation_bar, get_handler, get_progress_bar},
};

#[derive(Clone, Debug)]
pub struct GenBankImportOptions {
    pub add_annotations: bool,
    pub annotation_name: Option<String>,
    pub annotation_group: Option<String>,
}

impl Default for GenBankImportOptions {
    fn default() -> Self {
        Self {
            add_annotations: true,
            annotation_name: None,
            annotation_group: None,
        }
    }
}

impl GenBankImportOptions {
    pub fn annotation_name_from_path(mut self, path: impl AsRef<FsPath>) -> Self {
        self.annotation_name = path
            .as_ref()
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .map(str::to_string);
        self
    }
}

/// Maps a half-open interval in the final edited GenBank sequence back to a canonical node range.
///
/// `final_range` is a coordinate interval in the fully edited sequence from the imported GenBank
/// record. `node_id` and `sequence_range` identify the corresponding
/// half-open range on the canonical node that stores that portion of sequence in the graph.
#[derive(Clone, Debug, Eq, PartialEq)]
struct FinalSequenceSegment {
    final_range: Range,
    node_id: HashId,
    sequence_range: Range,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NodeSequenceSegment {
    node_id: HashId,
    sequence_range: Range,
    strand: Strand,
}

impl OrderedMerge for NodeSequenceSegment {
    fn should_merge_with(&self, next: &Self) -> bool {
        self.node_id == next.node_id
            && self.strand == next.strand
            && next.sequence_range.start >= self.sequence_range.start
            && next.sequence_range.start <= self.sequence_range.end
    }

    fn merge_with(&mut self, next: &Self) {
        self.sequence_range.end = self.sequence_range.end.max(next.sequence_range.end);
    }
}

struct LocusAnnotationImport<'a> {
    path: &'a Path,
    wt_node_id: HashId,
    wt_length: i64,
    final_sequence: &'a str,
    annotations: &'a [GenBankAnnotation],
    changes: &'a [(GenBankEdit, Option<HashId>)],
    annotation_name: Option<&'a str>,
    annotation_group: Option<&'a str>,
    collection: &'a str,
    sample: &'a str,
    locus_name: &'a str,
}

fn annotation_file_label(annotation_name: Option<&str>, fallback: &str) -> String {
    annotation_name
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| (!fallback.is_empty()).then(|| fallback.to_string()))
        .unwrap_or_else(|| "genbank".to_string())
}

fn annotation_group_name(
    annotation_name: Option<&str>,
    collection: &str,
    sample: &str,
    locus_name: &str,
) -> String {
    let file_label = annotation_file_label(annotation_name, locus_name);
    format!("GenBank {collection}/{sample}/{file_label}")
}

/// Builds a coordinate translation table from the final imported GenBank sequence back to the
/// graph nodes that store it.
///
/// GenBank annotations are expressed against the fully edited sequence from the file, but after
/// import that sequence may be split across the WT node plus any inserted or replacement nodes
/// created for edits. Each `FinalSequenceSegment` says which half-open range in the final sequence
/// maps to which node and node-local coordinate range, and `map_annotation_segments` uses that to
/// project annotation spans onto the correct graph nodes.
///
/// A confusing part of this code is that edit.start refers to the position in the wildtype sequence
/// where an edit begins. That is how GenBankEdit defines start/end.
fn build_final_sequence_segments(
    wt_node_id: HashId,
    wt_length: i64,
    changes: &[(GenBankEdit, Option<HashId>)],
) -> Vec<FinalSequenceSegment> {
    let mut segments = Vec::new();
    let mut wt_cursor = 0;
    let mut final_cursor = 0;

    for (edit, change_node_id) in changes {
        // If we are not at the position of the edit, add the segment of the wildtype sequence
        // to the chain
        if wt_cursor < edit.start {
            let reference_length = edit.start - wt_cursor;
            segments.push(FinalSequenceSegment {
                final_range: Range {
                    start: final_cursor,
                    end: final_cursor + reference_length,
                },
                node_id: wt_node_id,
                sequence_range: Range {
                    start: wt_cursor,
                    end: edit.start,
                },
            });
            wt_cursor = edit.start;
            final_cursor += reference_length;
        }

        match edit.edit_type {
            EditType::Insertion => {
                if let Some(change_node_id) = change_node_id {
                    let new_length = edit.new_sequence.len() as i64;
                    if new_length > 0 {
                        segments.push(FinalSequenceSegment {
                            final_range: Range {
                                start: final_cursor,
                                end: final_cursor + new_length,
                            },
                            node_id: *change_node_id,
                            sequence_range: Range {
                                start: 0,
                                end: new_length,
                            },
                        });
                    }
                    final_cursor += new_length;
                }
            }
            EditType::Replacement => {
                wt_cursor = edit.end;
                if let Some(change_node_id) = change_node_id {
                    let new_length = edit.new_sequence.len() as i64;
                    if new_length > 0 {
                        segments.push(FinalSequenceSegment {
                            final_range: Range {
                                start: final_cursor,
                                end: final_cursor + new_length,
                            },
                            node_id: *change_node_id,
                            sequence_range: Range {
                                start: 0,
                                end: new_length,
                            },
                        });
                    }
                    final_cursor += new_length;
                }
            }
            EditType::Deletion => {
                wt_cursor = edit.end;
            }
        }
    }

    if wt_cursor < wt_length {
        let reference_length = wt_length - wt_cursor;
        segments.push(FinalSequenceSegment {
            final_range: Range {
                start: final_cursor,
                end: final_cursor + reference_length,
            },
            node_id: wt_node_id,
            sequence_range: Range {
                start: wt_cursor,
                end: wt_length,
            },
        });
    }

    segments
}

/// Coalesces consecutive overlapping or adjacent node-local segments that are already ordered.
///
/// This is not a general interval merge. It assumes `segments` arrive in the correct traversal
/// order, so only the most recently merged segment needs to be checked. When two consecutive
/// segments are on the same node and strand and overlap or touch, the merged segment keeps the
/// original `sequence_range.start` and extends `sequence_range.end` to the farthest right boundary.
///
/// For example:
/// - `[0, 3)` then `[3, 5)` merges to `[0, 5)`
/// - `[0, 3)` then `[2, 7)` merges to `[0, 7)`
/// - `[0, 3)` then `[5, 7)` does not merge
/// - Same coordinates on different nodes or strands do not merge
fn merge_node_sequence_segments(
    segments: Vec<NodeSequenceSegment>,
) -> Result<Vec<NodeSequenceSegment>, GenBankError> {
    for segment in &segments {
        if segment.sequence_range.end <= segment.sequence_range.start {
            return Err(GenBankError::ParseError(format!(
                "Invalid node sequence segment: start {} must be less than end {}",
                segment.sequence_range.start, segment.sequence_range.end
            )));
        }
    }

    Ok(merge_ordered_items(segments))
}

/// Maps annotation spans from final edited-sequence coordinates onto node-local coordinates.
///
/// `overlap_end <= overlap_start` means the annotation segment and the current final-sequence
/// segment do not contribute a positive-width overlap, so that pair is skipped.
///
/// Completely before:
/// ```text
/// annotation: [----)
/// final:           [----)
///
/// overlap_start = final.start
/// overlap_end   = annotation.end
///
/// annotation.end <= final.start
/// => overlap_end <= overlap_start
/// ```
///
/// Completely after:
/// ```text
/// annotation:        [----)
/// final:      [----)
///
/// overlap_start = annotation.start
/// overlap_end   = final.end
///
/// final.end <= annotation.start
/// => overlap_end <= overlap_start
/// ```
///
/// Touching at a boundary:
/// ```text
/// annotation: [----)
/// final:           [----)
///
/// annotation.end == final.start
/// => overlap_end == overlap_start
/// ```
fn map_annotation_segments(
    annotation_segments: &[GenBankAnnotationSegment],
    final_segments: &[FinalSequenceSegment],
    preserve_part_boundaries: bool,
) -> Result<Vec<NodeSequenceSegment>, GenBankError> {
    let mapped = annotation_segments
        .iter()
        .map(|annotation_segment| {
            merge_node_sequence_segments(
                final_segments
                    .iter()
                    .filter_map(|final_segment| {
                        let overlap_start =
                            max(annotation_segment.start, final_segment.final_range.start);
                        let overlap_end =
                            min(annotation_segment.end, final_segment.final_range.end);
                        if overlap_end <= overlap_start {
                            return None;
                        }

                        let segment_start = final_segment.sequence_range.start
                            + (overlap_start - final_segment.final_range.start);
                        let segment_end = final_segment.sequence_range.start
                            + (overlap_end - final_segment.final_range.start);

                        Some(NodeSequenceSegment {
                            node_id: final_segment.node_id,
                            sequence_range: Range {
                                start: segment_start,
                                end: segment_end,
                            },
                            strand: annotation_segment.strand,
                        })
                    })
                    .collect(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    if preserve_part_boundaries {
        Ok(mapped)
    } else {
        merge_node_sequence_segments(mapped)
    }
}

fn create_accession_for_segments(
    conn: &gen_models::db::GraphConnection,
    path: &Path,
    accession_name: &str,
    segments: &[NodeSequenceSegment],
) -> Result<HashId, GenBankError> {
    let accession = match Accession::get_or_create(conn, accession_name, &path.id, None) {
        Ok(accession) => accession,
        Err(err) => return Err(GenBankError::AccessionError(err)),
    };
    let mut edges = Vec::with_capacity(segments.len() + 1);

    let first = segments.first().ok_or_else(|| {
        GenBankError::ParseError("Annotation has no mappable segments".to_string())
    })?;
    edges.push(AccessionEdgeData {
        source_node_id: PATH_START_NODE_ID,
        source_coordinate: -1,
        source_strand: Strand::Forward,
        target_node_id: first.node_id,
        target_coordinate: first.sequence_range.start,
        target_strand: first.strand,
        chromosome_index: 0,
    });

    for window in segments.windows(2) {
        let current = &window[0];
        let next = &window[1];
        edges.push(AccessionEdgeData {
            source_node_id: current.node_id,
            source_coordinate: current.sequence_range.end,
            source_strand: current.strand,
            target_node_id: next.node_id,
            target_coordinate: next.sequence_range.start,
            target_strand: next.strand,
            chromosome_index: 0,
        });
    }

    let last = segments.last().unwrap();
    edges.push(AccessionEdgeData {
        source_node_id: last.node_id,
        source_coordinate: last.sequence_range.end,
        source_strand: last.strand,
        target_node_id: PATH_END_NODE_ID,
        target_coordinate: -1,
        target_strand: Strand::Forward,
        chromosome_index: 0,
    });

    let edge_ids = AccessionEdge::bulk_create(conn, &edges);
    AccessionPath::create(conn, &accession.id, &edge_ids)?;
    Ok(accession.id)
}

fn import_locus_annotations(
    conn: &gen_models::db::GraphConnection,
    input: LocusAnnotationImport<'_>,
) -> Result<(), GenBankError> {
    if input.annotations.is_empty() {
        return Ok(());
    }

    let final_segments =
        build_final_sequence_segments(input.wt_node_id, input.wt_length, input.changes);
    let annotation_group = input
        .annotation_group
        .map(str::to_string)
        .unwrap_or_else(|| {
            annotation_group_name(
                input.annotation_name,
                input.collection,
                input.sample,
                input.locus_name,
            )
        });

    for annotation in input.annotations.iter() {
        let mapped_segments = map_annotation_segments(
            &annotation.segments,
            &final_segments,
            annotation
                .extra
                .as_ref()
                .and_then(|extra| extra.genbank.as_ref())
                .and_then(|extra| extra.location_operator.as_ref())
                .is_some(),
        )?;
        if mapped_segments.is_empty() {
            continue;
        }

        let annotation_sequence_key = annotation
            .segments
            .iter()
            .map(|segment| {
                let segment_sequence =
                    &input.final_sequence[segment.start as usize..segment.end as usize];
                format!(
                    "{}:{}:{}:{segment_sequence}",
                    segment.start, segment.end, segment.strand
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        let mapped_segment_key = mapped_segments
            .iter()
            .map(|segment| {
                format!(
                    "{}:{}:{}:{}",
                    segment.node_id,
                    segment.sequence_range.start,
                    segment.sequence_range.end,
                    segment.strand
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        let annotation_sequence_hash = HashId::convert_str(&format!(
            "{annotation_sequence_key};mapped_segments={mapped_segment_key}"
        ));
        let accession_name = format!(
            "{annotation_group}:annotation_sequence_hash={annotation_sequence_hash}:{}",
            annotation.name,
        );
        let accession_id =
            create_accession_for_segments(conn, input.path, &accession_name, &mapped_segments)?;

        let _ = Annotation::create_with_samples(
            conn,
            &annotation.name,
            &annotation_group,
            &accession_id,
            annotation.extra.as_ref(),
            &[input.sample],
        )?;
    }

    Ok(())
}

pub fn import_genbank<'a, R>(
    context: &DbContext,
    data: R,
    collection: impl Into<Option<&'a str>>,
    sample: &str,
    operation_info: OperationInfo,
    options: GenBankImportOptions,
) -> Result<Operation, GenBankError>
where
    R: Read,
{
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let mut session = start_operation(conn);
    let reader = reader::SeqReader::new(data);
    let collection = match Collection::create(conn, collection.into().unwrap_or_default()) {
        Ok(c) => c,
        Err(CollectionError::Duplicate(c)) => c,
        Err(e) => {
            return Err(GenBankError::CollectionError(e));
        }
    };
    match Sample::get_or_create(conn, sample) {
        Ok(_) => {}
        Err(e) => {
            return Err(GenBankError::SampleError(e));
        }
    }

    let _ = progress_bar.println("Parsing GenBank");
    let bar = progress_bar.add(get_progress_bar(None));
    bar.set_message("Entries parsed");
    for result in reader {
        match result {
            Ok(seq) => {
                let locus = process_sequence(seq)?;
                let original_seq = locus.original_sequence();
                let mut seq_model = Sequence::new().sequence(&original_seq);
                if !locus.name.is_empty() {
                    seq_model = seq_model.name(&locus.name);
                }
                if let Some(ref mol_type) = locus.molecule_type {
                    seq_model = seq_model.sequence_type(mol_type);
                }
                let sequence = seq_model.save(conn)?;
                let wt_node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!(
                        "{collection}.{contig}:{hash}",
                        collection = &collection.name,
                        contig = &locus.name,
                        hash = sequence.hash
                    )),
                )?;

                let block_group = BlockGroup::create(
                    conn,
                    NewBlockGroup {
                        collection_name: &collection.name,
                        sample_name: sample,
                        name: &locus.name,
                        ..Default::default()
                    },
                )?;
                let edge_into = Edge::create(
                    conn,
                    PATH_START_NODE_ID,
                    0,
                    Strand::Forward,
                    wt_node_id,
                    0,
                    Strand::Forward,
                )?;
                let edge_out_of = Edge::create(
                    conn,
                    wt_node_id,
                    sequence.length,
                    Strand::Forward,
                    PATH_END_NODE_ID,
                    0,
                    Strand::Forward,
                )?;
                BlockGroupEdge::bulk_create(
                    conn,
                    &[
                        BlockGroupEdgeData {
                            block_group_id: block_group.id,
                            edge_id: edge_into.id,
                            chromosome_index: 0,
                            phased: 0,
                        },
                        BlockGroupEdgeData {
                            block_group_id: block_group.id,
                            edge_id: edge_out_of.id,
                            chromosome_index: 0,
                            phased: 0,
                        },
                    ],
                );
                let path = Path::create(
                    conn,
                    &locus.name,
                    &block_group.id,
                    &[edge_into.id, edge_out_of.id],
                )?;

                let wt_changes = locus.changes_to_wt();
                let mut applied_changes = Vec::with_capacity(wt_changes.len());
                for edit in wt_changes {
                    let start = edit.start;
                    let end = edit.end;
                    let change_node_id = None;
                    let change = match edit.edit_type {
                        EditType::Insertion | EditType::Replacement => {
                            let change_seq = Sequence::new()
                                .sequence(&edit.new_sequence)
                                .name(&format!(
                                    "Geneious type: Editing History {edit_type}",
                                    edit_type = edit.edit_type
                                ))
                                .sequence_type("DNA")
                                .save(conn)?;
                            let change_node_id = Node::create(
                                conn,
                                &change_seq.hash,
                                &HashId::convert_str(&format!(
                                    "{parent_hash}:{start}-{end}->{new_hash}",
                                    parent_hash = &sequence.hash,
                                    new_hash = &change_seq.hash,
                                )),
                            )?;
                            PathChange {
                                block_group_id: block_group.id,
                                path: path.clone(),
                                path_accession: None,
                                start,
                                end,
                                block: PathBlock {
                                    node_id: change_node_id,
                                    block_sequence: edit.new_sequence.clone(),
                                    sequence_start: 0,
                                    sequence_end: change_seq.length,
                                    path_start: start,
                                    path_end: end + change_seq.length,
                                    strand: Strand::Forward,
                                },
                                chromosome_index: 1,
                                phased: 0,
                                preserve_edge: true,
                            }
                        }
                        EditType::Deletion => PathChange {
                            block_group_id: block_group.id,
                            path: path.clone(),
                            path_accession: None,
                            start,
                            end,
                            block: PathBlock {
                                node_id: wt_node_id,
                                block_sequence: "".to_string(),
                                sequence_start: 0,
                                sequence_end: 0,
                                path_start: start,
                                path_end: end,
                                strand: Strand::Forward,
                            },
                            chromosome_index: 1,
                            phased: 0,
                            preserve_edge: true,
                        },
                    };
                    let tree = path.intervaltree(conn);
                    BlockGroup::insert_change(conn, &change, &tree).unwrap();
                    applied_changes.push((edit, change_node_id));
                }

                if options.add_annotations {
                    import_locus_annotations(
                        conn,
                        LocusAnnotationImport {
                            path: &path,
                            wt_node_id,
                            wt_length: sequence.length,
                            final_sequence: &locus.sequence,
                            annotations: &locus.annotations,
                            changes: &applied_changes,
                            annotation_name: options.annotation_name.as_deref(),
                            annotation_group: options.annotation_group.as_deref(),
                            collection: &collection.name,
                            sample,
                            locus_name: &locus.name,
                        },
                    )?;
                }
            }
            Err(e) => return Err(GenBankError::ParseError(format!("Failed to parse {e}"))),
        }
        bar.inc(1);
    }
    bar.finish();
    let bar = add_saving_operation_bar(&progress_bar);
    let op = end_operation(
        context,
        &mut session,
        &operation_info,
        &format!(
            "Genbank Import of {files}",
            files = operation_info
                .files
                .iter()
                .map(|f| f.file_path.clone())
                .collect::<Vec<_>>()
                .join(",")
        ),
        None,
    )
    .map_err(GenBankError::OperationError);
    bar.finish();
    op
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs::File,
        io::BufReader,
        path::PathBuf,
    };

    use gen_core::is_terminal;
    use gen_models::{
        annotations::{Annotation, AnnotationGroup, GenBankLocationOperator},
        file_types::FileTypes,
        operations::OperationFile,
        traits::Query,
    };
    use noodles::fasta;

    use super::*;
    use crate::{
        test_helpers::setup_gen, track_database, views::annotations::load_annotations_for_group,
    };

    fn get_unmodified_sequence() -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/unmodified.fa");
        let mut reader = fasta::io::reader::Builder.build_from_path(path).unwrap();
        let mut records = reader.records();
        let record = records.next().unwrap().unwrap();
        let seq = record.sequence();
        str::from_utf8(seq.as_ref()).unwrap().to_string()
    }

    fn import_puc19(
        context: &DbContext,
        sample_name: &str,
        options: GenBankImportOptions,
    ) -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/puc19.gb");
        let file = File::open(&path).unwrap();
        let _ = import_genbank(
            context,
            BufReader::new(file),
            Some("fixtures"),
            sample_name,
            OperationInfo {
                files: vec![OperationFile {
                    file_path: path.to_str().unwrap().to_string(),
                    file_type: FileTypes::GenBank,
                }],
                description: "test".to_string(),
            },
            options.annotation_name_from_path(&path),
        )
        .unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn test_build_final_sequence_segments() {
        let wt_node_id = HashId::convert_str("wt");
        let insertion_node_id = HashId::convert_str("insertion");
        let replacement_node_id = HashId::convert_str("replacement");
        let segments = build_final_sequence_segments(
            wt_node_id,
            10,
            &[
                (
                    GenBankEdit {
                        start: 2,
                        end: 2,
                        old_sequence: String::new(),
                        new_sequence: "GG".to_string(),
                        edit_type: EditType::Insertion,
                    },
                    Some(insertion_node_id),
                ),
                (
                    GenBankEdit {
                        start: 5,
                        end: 7,
                        old_sequence: "AA".to_string(),
                        new_sequence: "TTT".to_string(),
                        edit_type: EditType::Replacement,
                    },
                    Some(replacement_node_id),
                ),
                (
                    GenBankEdit {
                        start: 8,
                        end: 9,
                        old_sequence: "C".to_string(),
                        new_sequence: String::new(),
                        edit_type: EditType::Deletion,
                    },
                    None,
                ),
            ],
        );

        assert_eq!(
            segments,
            vec![
                FinalSequenceSegment {
                    final_range: Range { start: 0, end: 2 },
                    node_id: wt_node_id,
                    sequence_range: Range { start: 0, end: 2 },
                },
                FinalSequenceSegment {
                    final_range: Range { start: 2, end: 4 },
                    node_id: insertion_node_id,
                    sequence_range: Range { start: 0, end: 2 },
                },
                FinalSequenceSegment {
                    final_range: Range { start: 4, end: 7 },
                    node_id: wt_node_id,
                    sequence_range: Range { start: 2, end: 5 },
                },
                FinalSequenceSegment {
                    final_range: Range { start: 7, end: 10 },
                    node_id: replacement_node_id,
                    sequence_range: Range { start: 0, end: 3 },
                },
                FinalSequenceSegment {
                    final_range: Range { start: 10, end: 11 },
                    node_id: wt_node_id,
                    sequence_range: Range { start: 7, end: 8 },
                },
                FinalSequenceSegment {
                    final_range: Range { start: 11, end: 12 },
                    node_id: wt_node_id,
                    sequence_range: Range { start: 9, end: 10 },
                },
            ]
        );
    }

    #[test]
    fn test_merge_node_sequence_segments() {
        let node_id = HashId::convert_str("node");
        let other_node_id = HashId::convert_str("other");

        let merged = merge_node_sequence_segments(vec![
            NodeSequenceSegment {
                node_id,
                sequence_range: Range { start: 0, end: 3 },
                strand: Strand::Forward,
            },
            NodeSequenceSegment {
                node_id,
                sequence_range: Range { start: 3, end: 5 },
                strand: Strand::Forward,
            },
            NodeSequenceSegment {
                node_id,
                sequence_range: Range { start: 4, end: 7 },
                strand: Strand::Forward,
            },
            NodeSequenceSegment {
                node_id,
                sequence_range: Range { start: 1, end: 2 },
                strand: Strand::Reverse,
            },
            NodeSequenceSegment {
                node_id: other_node_id,
                sequence_range: Range { start: 0, end: 1 },
                strand: Strand::Forward,
            },
        ])
        .unwrap();

        assert_eq!(
            merged,
            vec![
                NodeSequenceSegment {
                    node_id,
                    sequence_range: Range { start: 0, end: 7 },
                    strand: Strand::Forward,
                },
                NodeSequenceSegment {
                    node_id,
                    sequence_range: Range { start: 1, end: 2 },
                    strand: Strand::Reverse,
                },
                NodeSequenceSegment {
                    node_id: other_node_id,
                    sequence_range: Range { start: 0, end: 1 },
                    strand: Strand::Forward,
                },
            ]
        );
    }

    #[test]
    fn test_merge_node_sequence_segments_errors_on_invalid_range() {
        let node_id = HashId::convert_str("node");

        assert_eq!(
            merge_node_sequence_segments(vec![NodeSequenceSegment {
                node_id,
                sequence_range: Range { start: 7, end: 7 },
                strand: Strand::Forward,
            }]),
            Err(GenBankError::ParseError(
                "Invalid node sequence segment: start 7 must be less than end 7".to_string()
            ))
        );
    }

    #[test]
    fn test_map_annotation_segments() {
        let wt_node_id = HashId::convert_str("wt");
        let insertion_node_id = HashId::convert_str("insertion");

        let mapped = map_annotation_segments(
            &[GenBankAnnotationSegment {
                start: 1,
                end: 6,
                strand: Strand::Forward,
            }],
            &[
                FinalSequenceSegment {
                    final_range: Range { start: 0, end: 2 },
                    node_id: wt_node_id,
                    sequence_range: Range { start: 0, end: 2 },
                },
                FinalSequenceSegment {
                    final_range: Range { start: 2, end: 4 },
                    node_id: insertion_node_id,
                    sequence_range: Range { start: 0, end: 2 },
                },
                FinalSequenceSegment {
                    final_range: Range { start: 4, end: 7 },
                    node_id: wt_node_id,
                    sequence_range: Range { start: 2, end: 5 },
                },
            ],
            false,
        )
        .unwrap();

        assert_eq!(
            mapped,
            vec![
                NodeSequenceSegment {
                    node_id: wt_node_id,
                    sequence_range: Range { start: 1, end: 2 },
                    strand: Strand::Forward,
                },
                NodeSequenceSegment {
                    node_id: insertion_node_id,
                    sequence_range: Range { start: 0, end: 2 },
                    strand: Strand::Forward,
                },
                NodeSequenceSegment {
                    node_id: wt_node_id,
                    sequence_range: Range { start: 2, end: 4 },
                    strand: Strand::Forward,
                },
            ]
        );
    }

    #[test]
    fn test_map_annotation_segments_preserves_part_boundaries() {
        let wt_node_id = HashId::convert_str("wt");

        let mapped = map_annotation_segments(
            &[
                GenBankAnnotationSegment {
                    start: 0,
                    end: 2,
                    strand: Strand::Forward,
                },
                GenBankAnnotationSegment {
                    start: 2,
                    end: 5,
                    strand: Strand::Forward,
                },
            ],
            &[FinalSequenceSegment {
                final_range: Range { start: 0, end: 5 },
                node_id: wt_node_id,
                sequence_range: Range { start: 0, end: 5 },
            }],
            true,
        )
        .unwrap();

        assert_eq!(
            mapped,
            vec![
                NodeSequenceSegment {
                    node_id: wt_node_id,
                    sequence_range: Range { start: 0, end: 2 },
                    strand: Strand::Forward,
                },
                NodeSequenceSegment {
                    node_id: wt_node_id,
                    sequence_range: Range { start: 2, end: 5 },
                    strand: Strand::Forward,
                },
            ]
        );
    }

    #[test]
    fn test_error_on_invalid_file() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        assert_eq!(
            import_genbank(
                &context,
                BufReader::new("this is not valid".as_bytes()),
                None,
                Sample::DEFAULT_NAME,
                OperationInfo {
                    files: vec![OperationFile {
                        file_path: "".to_string(),
                        file_type: FileTypes::GenBank,
                    }],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default(),
            ),
            Err(GenBankError::ParseError(
                "Failed to parse Syntax error: Error MapRes while parsing [this is not valid]"
                    .to_string()
            ))
        )
    }

    #[test]
    fn test_records_operation() {
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
        assert_eq!(
            Operation::get_by_id(op_conn, &operation.hash).unwrap(),
            operation
        );
    }

    #[test]
    fn test_creates_sample() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/geneious_genbank/insertion.gb");
        let file = File::open(&path).unwrap();
        let _ = import_genbank(
            &context,
            BufReader::new(file),
            None,
            "new-sample",
            OperationInfo {
                files: vec![OperationFile {
                    file_path: "".to_string(),
                    file_type: FileTypes::GenBank,
                }],
                description: "test".to_string(),
            },
            GenBankImportOptions::default(),
        );
        assert_eq!(
            Sample::get_by_name(conn, "new-sample").unwrap().name,
            "new-sample"
        );
    }

    #[test]
    fn test_imports_puc19_annotations_by_default() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_puc19(&context, "puc19-sample", GenBankImportOptions::default());

        let groups = AnnotationGroup::query_by_sample(conn, "puc19-sample");
        assert_eq!(groups.len(), 1);
        assert!(groups[0].name.contains("puc19"));

        let annotations = Annotation::query_by_group(conn, &groups[0].name).unwrap();
        assert_eq!(annotations.len(), 21);
        assert!(
            annotations
                .iter()
                .any(|annotation| annotation.name == "AmpR")
        );
        assert!(
            annotations
                .iter()
                .any(|annotation| annotation.name == "lac promoter")
        );
        assert!(
            annotations
                .iter()
                .any(|annotation| annotation.name == "ori")
        );
        let amp_r = annotations
            .iter()
            .find(|annotation| annotation.name == "AmpR")
            .unwrap();
        let amp_r_extra = amp_r
            .extra
            .as_ref()
            .and_then(|extra| extra.genbank.as_ref())
            .unwrap();
        assert_eq!(amp_r_extra.kind, "CDS");
        assert!(
            amp_r_extra
                .qualifiers
                .iter()
                .any(|qualifier| qualifier.key == "product"
                    && qualifier.value.as_deref() == Some("beta-lactamase"))
        );

        let ori_annotation = annotations
            .iter()
            .find(|annotation| annotation.name == "ori")
            .unwrap();
        assert_eq!(
            ori_annotation
                .extra
                .as_ref()
                .and_then(|extra| extra.genbank.as_ref())
                .and_then(|extra| extra.location_operator.as_ref()),
            Some(&GenBankLocationOperator::Join)
        );

        let block_group = Sample::get_block_groups(conn, "fixtures", "puc19-sample")
            .into_iter()
            .next()
            .unwrap();
        let path = BlockGroup::get_current_path(conn, &block_group.id);
        let mut visible_ranges_by_node: HashMap<HashId, Vec<(i64, i64)>> = HashMap::new();
        for block in path.blocks(conn) {
            if is_terminal(block.node_id) {
                continue;
            }
            visible_ranges_by_node
                .entry(block.node_id)
                .or_default()
                .push((block.sequence_start, block.sequence_end));
        }

        let spans =
            load_annotations_for_group(conn, &groups[0].name, &visible_ranges_by_node).unwrap();
        let m13_forward = spans
            .iter()
            .find(|annotation| annotation.name == "M13 Forward")
            .unwrap();
        assert_eq!(m13_forward.segments.len(), 1);
        assert_eq!(m13_forward.segments[0].start, 688);
        assert_eq!(m13_forward.segments[0].end, 706);
        assert_eq!(m13_forward.segments[0].strand, Strand::Reverse);

        let ori = spans
            .iter()
            .find(|annotation| annotation.name == "ori")
            .unwrap();
        assert_eq!(ori.segments.len(), 2);
        assert!(
            ori.segments
                .iter()
                .any(|segment| segment.start == 2314 && segment.end == 2686)
        );
        assert!(
            ori.segments
                .iter()
                .any(|segment| segment.start == 0 && segment.end == 217)
        );
    }

    #[test]
    fn test_skips_puc19_annotations_with_option() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let _ = import_puc19(
            &context,
            "no-annotation-sample",
            GenBankImportOptions {
                add_annotations: false,
                annotation_name: None,
                annotation_group: None,
            },
        );

        assert!(AnnotationGroup::query_by_sample(conn, "no-annotation-sample").is_empty());
        let annotations = Annotation::query(conn, "select * from annotations", rusqlite::params!());
        assert!(annotations.is_empty());
    }

    #[cfg(test)]
    mod geneious_genbanks {
        use super::*;
        use crate::{normalize_string, track_database};

        #[test]
        fn test_parses_insertion() {
            // this file has an insertion from 1426-2220
            let context = setup_gen();
            let conn = context.graph().conn();
            let op_conn = context.operations().conn();

            track_database(conn, op_conn).unwrap();

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/insertion.gb");
            let file = File::open(&path).unwrap();
            let _ = import_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                OperationInfo {
                    files: vec![OperationFile {
                        file_path: "".to_string(),
                        file_type: FileTypes::GenBank,
                    }],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default(),
            );
            let f = reader::parse_file(&path).unwrap();
            let seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            let block_group_id = BlockGroup::get_id("", Sample::DEFAULT_NAME, "insertion", None);
            let seqs = BlockGroup::get_all_sequences(conn, &block_group_id, false);
            assert_eq!(
                seqs,
                HashSet::from_iter([
                    seq.clone(),
                    format!("{}{}", &seq[..1425].to_string(), &seq[2220..].to_string()).to_string()
                ])
            );
        }

        #[test]
        fn test_parses_deletion() {
            // this file has a deletion from 765-766
            let context = setup_gen();
            let conn = context.graph().conn();
            let op_conn = context.operations().conn();

            track_database(conn, op_conn).unwrap();

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/deletion.gb");
            let file = File::open(&path).unwrap();
            let _ = import_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                OperationInfo {
                    files: vec![OperationFile {
                        file_path: "".to_string(),
                        file_type: FileTypes::GenBank,
                    }],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default(),
            );
            let f = reader::parse_file(&path).unwrap();
            let seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            let deleted: String = normalize_string(
                "TTACGCCCCGCCCTGCCACTCATCGCAGTACTGTTGTAATT
        CATTAAGCATTCTGCCGACATGGAAGCCATCACAAACGGCATGATGAACCTGAATCGCCAGCG
        GCATCAGCACCTTGTCGCCTTGCGTATAATATTTGCCCATGGTGAAAACGGGGGCGAAGAAGT
        TGTCCATATTGGCCACGTTTAAATCAAAACTGGTGAAACTCACCCAGGGATTGGCTGAGACGA
        AAAACATATTCTCAATAAACCCTTTAGGGAAATAGGCCAGGTTTTCACCGTAACACGCCACAT
        CTTGCGAATATATGTGTAGAAACTGCCGGAAATCGTCGTGGTATTCACTCCAGAGCGATGAAA
        ACGTTTCAGTTTGCTCATGGAAAACGGTGTAACAAGGGTGAACACTATCCCATATCACCAGCT
        CACCGTCTTTCATTGCCATACGGAATTCCGGATGAGCATTCATCAGGCGGGCAAGAATGTGAA
        TAAAGGCCGGATAAAACTTGTGCTTATTTTTCTTTACGGTCTTTAAAAAGGCCGTAATATCCA
        GCTGAACGGTCTGGTTATAGGTACATTGAGCAACTGACTGAAATGCCTCAAAATGTTCTTTAC
        GATGCCATTGGGATATATCAACGGTGGTATATCCAGTGATTTTTTTCTCCAT",
            );
            let block_group_id = BlockGroup::get_id("", Sample::DEFAULT_NAME, "deletion", None);
            let seqs = BlockGroup::get_all_sequences(conn, &block_group_id, false);
            assert_eq!(
                seqs,
                HashSet::from_iter([
                    seq.clone(),
                    format!(
                        "{}{deleted}{}",
                        &seq[..765].to_string(),
                        &seq[765..].to_string()
                    )
                    .to_string()
                ])
            );
        }

        #[test]
        fn test_parses_deletion_and_insertion() {
            let context = setup_gen();
            let conn = context.graph().conn();
            let op_conn = context.operations().conn();

            track_database(conn, op_conn).unwrap();

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/deletion_and_insertion.gb");
            let file = File::open(&path).unwrap();
            let _ = import_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                OperationInfo {
                    files: vec![OperationFile {
                        file_path: "".to_string(),
                        file_type: FileTypes::GenBank,
                    }],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default(),
            );
            let f = reader::parse_file(&path).unwrap();
            let seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            let deleted: String = normalize_string(
                "TACGCCCCGCCCTGCCACTCATCGCAGTACTGTTGTAATTC
             ATTAAGCATTCTGCCGACATGGAAGCCATCACAAACGGCATGATGAACCTGAATCGCC
             AGCGGCATCAGCACCTTGTCGCCTTGCGTATAATATTTGCCCATGGTGAAAACGGGGG
             CGAAGAAGTTGTCCATATTGGCCACGTTTAAATCAAAACTGGTGAAACTCACCCAGGG
             ATTGGCTGAGACGAAAAACATATTCTCAATAAACCCTTTAGGGAAATAGGCCAGGTTT
             TCACCGTAACACGCCACATCTTGCGAATATATGTGTAGAAACTGCCGGAAATCGTCGT
             GGTATTCACTCCAGAGCGATGAAAACGTTTCAGTTTGCTCATGGAAAACGGTGTAACA
             AGGGTGAACACTATCCCATATCACCAGCTCACCGTCTTTCATTGCCATACGGAATTCC
             GGATGAGCATTCATCAGGCGGGCAAGAATGTGAATAAAGGCCGGATAAAACTTGTGCT
             TATTTTTCTTTACGGTCTTTAAAAAGGCCGTAATATCCAGCTGAACGGTCTGGTTATA
             GGTACATTGAGCAACTGACTGAAATGCCTCAAAATGTTCTTTACGATGCCATTGGGAT
             ATATCAACGGTGGTATATCCAGTGATTTTTTTCTC",
            );
            let seqs = BlockGroup::get_all_sequences(
                conn,
                &BlockGroup::get_id("", Sample::DEFAULT_NAME, "deletion_and_insertion", None),
                false,
            );
            assert_eq!(
                seqs,
                HashSet::from_iter([
                    seq.clone(),
                    format!(
                        "{}{deleted}{}",
                        &seq[..766].to_string(),
                        &seq[1557..].to_string()
                    )
                    .to_string()
                ])
            );
        }

        #[test]
        fn test_parses_substitution() {
            // replacing a sequence ends up with the same result as doing a compound delete + insert
            // in the above test.
            let context = setup_gen();
            let conn = context.graph().conn();
            let op_conn = context.operations().conn();

            track_database(conn, op_conn).unwrap();

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/substitution.gb");
            let file = File::open(&path).unwrap();
            let _ = import_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                OperationInfo {
                    files: vec![OperationFile {
                        file_path: "".to_string(),
                        file_type: FileTypes::GenBank,
                    }],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default(),
            );
            let f = reader::parse_file(&path).unwrap();
            let seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            let deleted: String = normalize_string(
                "TACGCCCCGCCCTGCCACTCATCGCAGTACTGTTGTAATTC
             ATTAAGCATTCTGCCGACATGGAAGCCATCACAAACGGCATGATGAACCTGAATCGCC
             AGCGGCATCAGCACCTTGTCGCCTTGCGTATAATATTTGCCCATGGTGAAAACGGGGG
             CGAAGAAGTTGTCCATATTGGCCACGTTTAAATCAAAACTGGTGAAACTCACCCAGGG
             ATTGGCTGAGACGAAAAACATATTCTCAATAAACCCTTTAGGGAAATAGGCCAGGTTT
             TCACCGTAACACGCCACATCTTGCGAATATATGTGTAGAAACTGCCGGAAATCGTCGT
             GGTATTCACTCCAGAGCGATGAAAACGTTTCAGTTTGCTCATGGAAAACGGTGTAACA
             AGGGTGAACACTATCCCATATCACCAGCTCACCGTCTTTCATTGCCATACGGAATTCC
             GGATGAGCATTCATCAGGCGGGCAAGAATGTGAATAAAGGCCGGATAAAACTTGTGCT
             TATTTTTCTTTACGGTCTTTAAAAAGGCCGTAATATCCAGCTGAACGGTCTGGTTATA
             GGTACATTGAGCAACTGACTGAAATGCCTCAAAATGTTCTTTACGATGCCATTGGGAT
             ATATCAACGGTGGTATATCCAGTGATTTTTTTCTC",
            );
            let seqs = BlockGroup::get_all_sequences(
                conn,
                &BlockGroup::get_id("", Sample::DEFAULT_NAME, "substitution", None),
                false,
            );
            assert_eq!(
                seqs,
                HashSet::from_iter([
                    seq.clone(),
                    format!(
                        "{}{deleted}{}",
                        &seq[..766].to_string(),
                        &seq[1557..].to_string()
                    )
                    .to_string()
                ])
            );
        }

        #[test]
        fn test_parses_multiple_changes() {
            let context = setup_gen();
            let conn = context.graph().conn();
            let op_conn = context.operations().conn();

            track_database(conn, op_conn).unwrap();

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/multiple_insertions_deletions.gb");
            let file = File::open(&path).unwrap();
            let _ = import_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                OperationInfo {
                    files: vec![OperationFile {
                        file_path: "".to_string(),
                        file_type: FileTypes::GenBank,
                    }],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default(),
            );
            // there would be 4! sequences so we just check we have the fully changed and unchanged sequence
            let f = reader::parse_file(&path).unwrap();
            let mod_seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            let sequences: HashSet<String> = BlockGroup::get_all_sequences(
                conn,
                &BlockGroup::get_id("", Sample::DEFAULT_NAME, "insertion", None),
                false,
            )
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
            let unchanged_seq = get_unmodified_sequence();
            assert!(sequences.contains(&mod_seq));
            assert!(sequences.contains(&unchanged_seq));
        }
    }
}
