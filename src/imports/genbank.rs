use std::{
    cmp::{max, min},
    io::Read,
    path::Path as FsPath,
    str,
};

use gb_io::reader;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand};
use gen_models::{
    accession::{Accession, AccessionEdge, AccessionEdgeData, AccessionPath},
    annotations::Annotation,
    block_group::{BlockGroup, NewBlockGroup, PathChange},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
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

#[derive(Clone, Copy, Debug)]
pub struct GenBankImportOptions {
    pub add_annotations: bool,
}

impl Default for GenBankImportOptions {
    fn default() -> Self {
        Self {
            add_annotations: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FinalSequenceSegment {
    final_start: i64,
    final_end: i64,
    node_id: HashId,
    sequence_start: i64,
    sequence_end: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NodeSequenceSegment {
    node_id: HashId,
    sequence_start: i64,
    sequence_end: i64,
}

struct LocusAnnotationImport<'a> {
    path: &'a Path,
    wt_node_id: HashId,
    wt_length: i64,
    annotations: &'a [GenBankAnnotation],
    changes: &'a [(GenBankEdit, Option<HashId>)],
    operation_info: &'a OperationInfo,
    collection: &'a str,
    sample: Option<&'a str>,
    locus_name: &'a str,
}

fn annotation_file_label(operation_info: &OperationInfo, fallback: &str) -> String {
    operation_info
        .files
        .first()
        .and_then(|file| {
            FsPath::new(&file.file_path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .filter(|stem| !stem.is_empty())
                .map(str::to_string)
        })
        .or_else(|| (!fallback.is_empty()).then(|| fallback.to_string()))
        .unwrap_or_else(|| "genbank".to_string())
}

fn annotation_group_name(
    operation_info: &OperationInfo,
    collection: &str,
    sample: Option<&str>,
    locus_name: &str,
) -> String {
    let file_label = annotation_file_label(operation_info, locus_name);
    let sample_label = sample.unwrap_or("reference");
    format!("GenBank {collection}/{sample_label}/{file_label}")
}

fn build_final_sequence_segments(
    wt_node_id: HashId,
    wt_length: i64,
    changes: &[(GenBankEdit, Option<HashId>)],
) -> Vec<FinalSequenceSegment> {
    let mut segments = Vec::new();
    let mut wt_cursor = 0;
    let mut final_cursor = 0;

    for (edit, change_node_id) in changes {
        if wt_cursor < edit.start {
            let reference_length = edit.start - wt_cursor;
            segments.push(FinalSequenceSegment {
                final_start: final_cursor,
                final_end: final_cursor + reference_length,
                node_id: wt_node_id,
                sequence_start: wt_cursor,
                sequence_end: edit.start,
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
                            final_start: final_cursor,
                            final_end: final_cursor + new_length,
                            node_id: *change_node_id,
                            sequence_start: 0,
                            sequence_end: new_length,
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
                            final_start: final_cursor,
                            final_end: final_cursor + new_length,
                            node_id: *change_node_id,
                            sequence_start: 0,
                            sequence_end: new_length,
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
            final_start: final_cursor,
            final_end: final_cursor + reference_length,
            node_id: wt_node_id,
            sequence_start: wt_cursor,
            sequence_end: wt_length,
        });
    }

    segments
}

fn merge_node_sequence_segments(segments: Vec<NodeSequenceSegment>) -> Vec<NodeSequenceSegment> {
    let mut merged: Vec<NodeSequenceSegment> = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment.sequence_end <= segment.sequence_start {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && last.node_id == segment.node_id
            && segment.sequence_start >= last.sequence_start
            && segment.sequence_start <= last.sequence_end
        {
            last.sequence_end = last.sequence_end.max(segment.sequence_end);
            continue;
        }
        merged.push(segment);
    }
    merged
}

fn map_annotation_segments(
    annotation_segments: &[GenBankAnnotationSegment],
    final_segments: &[FinalSequenceSegment],
) -> Vec<NodeSequenceSegment> {
    merge_node_sequence_segments(
        annotation_segments
            .iter()
            .flat_map(|annotation_segment| {
                final_segments.iter().filter_map(|final_segment| {
                    let overlap_start = max(annotation_segment.start, final_segment.final_start);
                    let overlap_end = min(annotation_segment.end, final_segment.final_end);
                    if overlap_end <= overlap_start {
                        return None;
                    }

                    let segment_start =
                        final_segment.sequence_start + (overlap_start - final_segment.final_start);
                    let segment_end =
                        final_segment.sequence_start + (overlap_end - final_segment.final_start);

                    Some(NodeSequenceSegment {
                        node_id: final_segment.node_id,
                        sequence_start: segment_start,
                        sequence_end: segment_end,
                    })
                })
            })
            .collect(),
    )
}

fn create_accession_for_segments(
    conn: &gen_models::db::GraphConnection,
    path: &Path,
    accession_name: &str,
    segments: &[NodeSequenceSegment],
) -> Result<HashId, GenBankError> {
    let accession = Accession::create(conn, accession_name, &path.id, None)?;
    let mut edges = Vec::with_capacity(segments.len() + 1);

    let first = segments.first().ok_or_else(|| {
        GenBankError::ParseError("Annotation has no mappable segments".to_string())
    })?;
    edges.push(AccessionEdgeData {
        source_node_id: PATH_START_NODE_ID,
        source_coordinate: -1,
        source_strand: Strand::Forward,
        target_node_id: first.node_id,
        target_coordinate: first.sequence_start,
        target_strand: Strand::Forward,
        chromosome_index: 0,
    });

    for window in segments.windows(2) {
        let current = &window[0];
        let next = &window[1];
        edges.push(AccessionEdgeData {
            source_node_id: current.node_id,
            source_coordinate: current.sequence_end,
            source_strand: Strand::Forward,
            target_node_id: next.node_id,
            target_coordinate: next.sequence_start,
            target_strand: Strand::Forward,
            chromosome_index: 0,
        });
    }

    let last = segments.last().unwrap();
    edges.push(AccessionEdgeData {
        source_node_id: last.node_id,
        source_coordinate: last.sequence_end,
        source_strand: Strand::Forward,
        target_node_id: PATH_END_NODE_ID,
        target_coordinate: -1,
        target_strand: Strand::Forward,
        chromosome_index: 0,
    });

    let edge_ids = AccessionEdge::bulk_create(conn, &edges);
    AccessionPath::create(conn, &accession.id, &edge_ids);
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
    let annotation_group = annotation_group_name(
        input.operation_info,
        input.collection,
        input.sample,
        input.locus_name,
    );

    for (annotation_index, annotation) in input.annotations.iter().enumerate() {
        let mapped_segments = map_annotation_segments(&annotation.segments, &final_segments);
        if mapped_segments.is_empty() {
            continue;
        }

        let accession_name = format!("{annotation_group}:{annotation_index}:{}", annotation.name);
        let accession_id =
            create_accession_for_segments(conn, input.path, &accession_name, &mapped_segments)?;

        if let Some(sample_name) = input.sample {
            let _ = Annotation::create_with_samples(
                conn,
                &annotation.name,
                &annotation_group,
                &accession_id,
                &[sample_name],
            )?;
        } else {
            let _ = Annotation::get_or_create(
                conn,
                &annotation.name,
                &annotation_group,
                &accession_id,
            )?;
        }
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
    let collection = Collection::create(conn, collection.into().unwrap_or_default());
    Sample::get_or_create(conn, sample);

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
                let sequence = seq_model.save(conn);
                let wt_node_id = Node::create(
                    conn,
                    &sequence.hash,
                    &HashId::convert_str(&format!(
                        "{collection}.{contig}:{hash}",
                        collection = &collection.name,
                        contig = &locus.name,
                        hash = sequence.hash
                    )),
                );

                let block_group = BlockGroup::create(
                    conn,
                    NewBlockGroup {
                        collection_name: &collection.name,
                        sample_name: sample,
                        name: &locus.name,
                        ..Default::default()
                    },
                );
                let edge_into = Edge::create(
                    conn,
                    PATH_START_NODE_ID,
                    0,
                    Strand::Forward,
                    wt_node_id,
                    0,
                    Strand::Forward,
                );
                let edge_out_of = Edge::create(
                    conn,
                    wt_node_id,
                    sequence.length,
                    Strand::Forward,
                    PATH_END_NODE_ID,
                    0,
                    Strand::Forward,
                );
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
                );

                let wt_changes = locus.changes_to_wt();
                let mut applied_changes = Vec::with_capacity(wt_changes.len());
                for edit in wt_changes {
                    let start = edit.start;
                    let end = edit.end;
                    let mut change_node_id = None;
                    let change = match edit.edit_type {
                        EditType::Insertion | EditType::Replacement => {
                            let change_seq = Sequence::new()
                                .sequence(&edit.new_sequence)
                                .name(&format!(
                                    "Geneious type: Editing History {edit_type}",
                                    edit_type = edit.edit_type
                                ))
                                .sequence_type("DNA")
                                .save(conn);
                            let change_node = Node::create(
                                conn,
                                &change_seq.hash,
                                &HashId::convert_str(&format!(
                                    "{parent_hash}:{start}-{end}->{new_hash}",
                                    parent_hash = &sequence.hash,
                                    new_hash = &change_seq.hash,
                                )),
                            );
                            change_node_id = Some(change_node);
                            PathChange {
                                block_group_id: block_group.id,
                                path: path.clone(),
                                path_accession: None,
                                start,
                                end,
                                block: PathBlock {
                                    node_id: change_node,
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
                            annotations: &locus.annotations,
                            changes: &applied_changes,
                            operation_info: &operation_info,
                            collection: &collection.name,
                            sample: Some(sample),
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
        annotations::{Annotation, AnnotationGroup},
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
            Some(sample_name),
            OperationInfo {
                files: vec![OperationFile {
                    file_path: path.to_str().unwrap().to_string(),
                    file_type: FileTypes::GenBank,
                }],
                description: "test".to_string(),
            },
            options,
        )
        .unwrap();
        path.to_string_lossy().to_string()
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

        let block_group = Sample::get_block_groups(conn, "fixtures", Some("puc19-sample"))
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
