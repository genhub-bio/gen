use std::{io::Read, path::Path as FsPath, str};

use gb_io::reader;
use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand,
    calculate_hash,
};
use gen_models::{
    accession::{Accession, AccessionSpanBlock, AccessionSpanCreate, ResolvedAccessionSpan},
    annotations::Annotation,
    block_group::{BlockGroup, BlockGroupChange, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::{Edge, EdgeData},
    errors::CollectionError,
    node::Node,
    operations::{Operation, OperationInfo},
    path::Path,
    region::ResolvedGenRegion,
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
    traits::Query,
};
use itertools::Itertools;

use crate::{
    genbank::{
        EditType, GenBankAnnotation, GenBankAnnotationSegment, GenBankError, process_sequence,
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

fn blocks_at_coordinate(
    tree: &intervaltree::IntervalTree<i64, NodeIntervalBlock>,
    coordinate: i64,
) -> Vec<&NodeIntervalBlock> {
    tree.query_point(coordinate)
        .map(|item| &item.value)
        .collect()
}

fn annotation_segment_span(
    segment: &GenBankAnnotationSegment,
    tree: &intervaltree::IntervalTree<i64, NodeIntervalBlock>,
    all_path_edges: &[Edge],
) -> ResolvedAccessionSpan {
    let start_blocks = blocks_at_coordinate(tree, segment.start);
    assert_eq!(start_blocks.len(), 1);
    let start_block = start_blocks[0];
    let end_blocks = blocks_at_coordinate(tree, segment.end - 1);
    assert_eq!(end_blocks.len(), 1);
    let end_block = end_blocks[0];

    let (start_edge_index, _) = all_path_edges
        .iter()
        .enumerate()
        .find(|(_, edge)| {
            edge.target_node_id == start_block.node_id
                && edge.target_coordinate == start_block.sequence_start
        })
        .expect("should find path edge entering annotation start block");
    let (end_edge_index, _) = all_path_edges
        .iter()
        .enumerate()
        .find(|(_, edge)| {
            edge.source_node_id == end_block.node_id
                && edge.source_coordinate == end_block.sequence_end
        })
        .expect("should find path edge leaving annotation end block");
    assert!(
        start_edge_index <= end_edge_index,
        "annotation start edge should precede end edge"
    );

    let blocks = tree
        .query(segment.start..segment.end)
        .map(|item| item.value)
        .filter(|block| block.node_id != PATH_START_NODE_ID && block.node_id != PATH_END_NODE_ID)
        .sorted_by(|a, b| a.start.cmp(&b.start))
        .map(|block| {
            let source_edge = all_path_edges
                .iter()
                .find(|edge| {
                    edge.target_node_id == block.node_id
                        && edge.target_coordinate == block.sequence_start
                })
                .expect("should find edge entering annotation block");
            let target_edge = all_path_edges
                .iter()
                .find(|edge| {
                    edge.source_node_id == block.node_id
                        && edge.source_coordinate == block.sequence_end
                })
                .expect("should find edge leaving annotation block");
            AccessionSpanBlock {
                node_id: block.node_id,
                sequence_start: block.sequence_start + (segment.start - block.start).max(0),
                sequence_end: block.sequence_end - (block.end - segment.end).max(0),
                strand: block.strand,
                source_edge_id: source_edge.id,
                target_edge_id: target_edge.id,
            }
        })
        .collect();
    ResolvedAccessionSpan { blocks }
}

fn annotation_sequence(annotation: &GenBankAnnotation, sequence: &str) -> String {
    annotation
        .segments
        .iter()
        .map(|segment| sequence[segment.start as usize..segment.end as usize].to_string())
        .collect::<Vec<_>>()
        .join("")
}

fn annotation_sequence_hash(annotation: &GenBankAnnotation, sequence: &str) -> HashId {
    HashId(calculate_hash(&annotation_sequence(annotation, sequence)))
}

fn annotation_accession_name(annotation_group: &str, annotation_sequence_hash: &HashId) -> String {
    HashId(calculate_hash(&format!(
        "{annotation_group}:{annotation_sequence_hash}"
    )))
    .to_string()
}

fn add_sequence_block(
    blocks: &mut Vec<PathBlock>,
    node_id: HashId,
    sequence_start: i64,
    sequence_end: i64,
    final_position: &mut i64,
) {
    if sequence_end <= sequence_start {
        return;
    }
    let block_len = sequence_end - sequence_start;
    blocks.push(PathBlock {
        node_id,
        block_sequence: String::new(),
        sequence_start,
        sequence_end,
        path_start: *final_position,
        path_end: *final_position + block_len,
        strand: Strand::Forward,
    });
    *final_position += block_len;
}

fn updated_sequence_blocks(
    wt_node_id: HashId,
    original_length: i64,
    edited_blocks: &[(i64, i64, Option<PathBlock>)],
) -> Vec<PathBlock> {
    let mut blocks = Vec::new();
    let mut original_position = 0;
    let mut final_position = 0;
    for (edit_start, edit_end, inserted_block) in edited_blocks {
        add_sequence_block(
            &mut blocks,
            wt_node_id,
            original_position,
            *edit_start,
            &mut final_position,
        );
        if let Some(block) = inserted_block {
            add_sequence_block(
                &mut blocks,
                block.node_id,
                block.sequence_start,
                block.sequence_end,
                &mut final_position,
            );
        }
        original_position = *edit_end;
    }
    add_sequence_block(
        &mut blocks,
        wt_node_id,
        original_position,
        original_length,
        &mut final_position,
    );
    blocks
}

fn updated_sequence_edges(
    conn: &gen_models::db::GraphConnection,
    blocks: &[PathBlock],
) -> Result<Vec<Edge>, GenBankError> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    let mut edges = Vec::new();
    let first = blocks.first().expect("should contain first updated block");
    edges.push(EdgeData {
        source_node_id: PATH_START_NODE_ID,
        source_coordinate: 0,
        source_strand: Strand::Forward,
        target_node_id: first.node_id,
        target_coordinate: first.sequence_start,
        target_strand: Strand::Forward,
    });
    for window in blocks.windows(2) {
        let source = &window[0];
        let target = &window[1];
        edges.push(EdgeData {
            source_node_id: source.node_id,
            source_coordinate: source.sequence_end,
            source_strand: Strand::Forward,
            target_node_id: target.node_id,
            target_coordinate: target.sequence_start,
            target_strand: Strand::Forward,
        });
    }
    let last = blocks.last().expect("should contain last updated block");
    edges.push(EdgeData {
        source_node_id: last.node_id,
        source_coordinate: last.sequence_end,
        source_strand: Strand::Forward,
        target_node_id: PATH_END_NODE_ID,
        target_coordinate: 0,
        target_strand: Strand::Forward,
    });
    let edge_ids = edges.iter().map(EdgeData::id_hash).collect::<Vec<_>>();
    let existing_edges = Edge::query_by_ids(conn, &edge_ids);
    if existing_edges.len() != edge_ids.len() {
        let existing_edge_ids = existing_edges
            .iter()
            .map(|edge| edge.id)
            .collect::<std::collections::HashSet<_>>();
        let missing = edge_ids
            .into_iter()
            .find(|edge_id| !existing_edge_ids.contains(edge_id))
            .expect("should find missing edge id");
        return Err(GenBankError::LookupError(format!(
            "Annotation walk edge {missing} was not found in existing graph topology"
        )));
    }
    Ok(existing_edges)
}

fn updated_sequence_intervaltree(
    blocks: &[PathBlock],
) -> intervaltree::IntervalTree<i64, NodeIntervalBlock> {
    blocks
        .iter()
        .map(|block| {
            (
                block.path_start..block.path_end,
                NodeIntervalBlock {
                    node_id: block.node_id,
                    start: block.path_start,
                    end: block.path_end,
                    sequence_start: block.sequence_start,
                    sequence_end: block.sequence_end,
                    strand: block.strand,
                },
            )
        })
        .collect()
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
    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample,
            ..Default::default()
        },
    ) {
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
                let sequence_type = if locus.circular {
                    locus
                        .molecule_type
                        .as_ref()
                        .map(|mol_type| format!("circular {mol_type}"))
                        .unwrap_or_else(|| "circular".to_string())
                } else {
                    locus
                        .molecule_type
                        .clone()
                        .unwrap_or_else(|| "DNA".to_string())
                };
                seq_model = seq_model.sequence_type(&sequence_type);
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
                let mut edited_blocks = Vec::<(i64, i64, Option<PathBlock>)>::new();
                for edit in wt_changes {
                    let start = edit.start;
                    let end = edit.end;
                    let region =
                        ResolvedGenRegion::from_path(conn, block_group.id, &path, start, end)?;
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
                            let block = PathBlock {
                                node_id: change_node_id,
                                block_sequence: edit.new_sequence.clone(),
                                sequence_start: 0,
                                sequence_end: change_seq.length,
                                path_start: start,
                                path_end: end + change_seq.length,
                                strand: Strand::Forward,
                            };
                            edited_blocks.push((start, end, Some(block.clone())));
                            BlockGroupChange {
                                region: region.clone(),
                                path_accession: None,
                                block,
                                chromosome_index: 1,
                                phased: 0,
                                preserve_edge: true,
                            }
                        }
                        EditType::Deletion => {
                            edited_blocks.push((start, end, None));
                            BlockGroupChange {
                                region,
                                path_accession: None,
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
                            }
                        }
                    };
                    BlockGroup::insert_change(conn, &change).unwrap();
                }

                if options.add_annotations {
                    let annotation_group = options.annotation_group.clone().unwrap_or_else(|| {
                        annotation_group_name(
                            options.annotation_name.as_deref(),
                            &collection.name,
                            sample,
                            &locus.name,
                        )
                    });
                    let annotation_blocks =
                        updated_sequence_blocks(wt_node_id, sequence.length, &edited_blocks);
                    let path_tree = updated_sequence_intervaltree(&annotation_blocks);
                    let all_path_edges = updated_sequence_edges(conn, &annotation_blocks)?;
                    for annotation in locus.annotations.iter() {
                        let sequence_hash = annotation_sequence_hash(annotation, &locus.sequence);
                        let accession_name =
                            annotation_accession_name(&annotation_group, &sequence_hash);
                        let spans = annotation
                            .segments
                            .iter()
                            .map(|segment| {
                                annotation_segment_span(segment, &path_tree, &all_path_edges)
                            })
                            .collect::<Vec<_>>();
                        let accession = Accession::get_or_create_from_spans(
                            conn,
                            AccessionSpanCreate {
                                name: &accession_name,
                                block_group_id: &block_group.id,
                                parent_accession_id: None,
                                spans: &spans,
                            },
                        )?;
                        Annotation::create_with_samples(
                            conn,
                            &annotation.name,
                            &annotation_group,
                            &accession.id,
                            annotation.extra.as_ref(),
                            &[sample],
                        )?;
                    }
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
    use std::{collections::HashSet, fs::File, io::BufReader, path::PathBuf};

    use gen_models::{
        accession::Accession,
        annotations::{Annotation, AnnotationGroup, GenBankLocationOperator},
        file_types::FileTypes,
        operations::OperationFile,
        traits::Query,
    };
    use noodles::fasta;

    use super::*;
    use crate::{test_helpers::setup_gen, track_database};

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
                files: vec![
                    OperationFile::new(path.to_str().unwrap().to_string())
                        .set_file_type(FileTypes::GenBank),
                ],
                description: "test".to_string(),
            },
            options.annotation_name_from_path(&path),
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
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
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
                files: vec![
                    OperationFile::new(path.to_str().unwrap().to_string())
                        .set_file_type(FileTypes::GenBank),
                ],
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
                files: vec![OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank)],
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

    #[test]
    fn imports_puc19_annotations() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let path = import_puc19(&context, "puc19-import", GenBankImportOptions::default());

        let expected_group = "GenBank fixtures/puc19-import/puc19";
        assert_eq!(
            AnnotationGroup::query_by_sample(conn, "puc19-import"),
            vec![AnnotationGroup {
                name: expected_group.to_string()
            }]
        );
        let annotations = Annotation::query_by_group(conn, expected_group).unwrap();
        let labels = annotations
            .iter()
            .map(|annotation| annotation.name.clone())
            .collect::<HashSet<_>>();
        assert!(labels.contains("AmpR"));
        assert!(labels.contains("lac promoter"));
        assert!(labels.contains("M13 Forward"));
        assert!(labels.contains("ori"));

        let m13_rev = annotations
            .iter()
            .find(|annotation| annotation.name == "M13 rev")
            .unwrap();
        let m13_reverse = annotations
            .iter()
            .find(|annotation| annotation.name == "M13 Reverse")
            .unwrap();
        assert_eq!(m13_rev.accession_id, m13_reverse.accession_id);

        let parsed = reader::parse_file(PathBuf::from(path)).unwrap();
        let locus = process_sequence(parsed.into_iter().next().unwrap()).unwrap();
        let original_sequence = locus.original_sequence();
        let source_annotation = locus
            .annotations
            .into_iter()
            .find(|annotation| annotation.name == "M13 rev")
            .unwrap();
        let sequence_hash = annotation_sequence_hash(&source_annotation, &original_sequence);
        let expected_accession_name = annotation_accession_name(expected_group, &sequence_hash);
        let accession = Accession::get_by_id(conn, &m13_rev.accession_id).unwrap();
        assert_eq!(accession.name, expected_accession_name);

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
        assert_eq!(
            amp_r_extra.location_operator,
            Some(GenBankLocationOperator::Join)
        );
        assert!(amp_r_extra.qualifiers.iter().any(|qualifier| {
            qualifier.key == "product" && qualifier.value.as_deref() == Some("beta-lactamase")
        }));
        let amp_r_blocks = Accession::get_by_id(conn, &amp_r.accession_id)
            .unwrap()
            .blocks(conn)
            .unwrap()
            .into_iter()
            .filter(|block| {
                block.node_id != PATH_START_NODE_ID && block.node_id != PATH_END_NODE_ID
            })
            .map(|block| (block.sequence_start, block.sequence_end))
            .collect::<Vec<_>>();
        assert_eq!(amp_r_blocks, vec![(1283, 1352), (1352, 2144)]);

        let ori = annotations
            .iter()
            .find(|annotation| annotation.name == "ori")
            .unwrap();
        let ori_blocks = Accession::get_by_id(conn, &ori.accession_id)
            .unwrap()
            .blocks(conn)
            .unwrap()
            .into_iter()
            .filter(|block| {
                block.node_id != PATH_START_NODE_ID && block.node_id != PATH_END_NODE_ID
            })
            .map(|block| (block.sequence_start, block.sequence_end))
            .collect::<Vec<_>>();
        assert_eq!(ori_blocks, vec![(2314, 2686), (0, 217)]);
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
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default(),
            );
            let f = reader::parse_file(&path).unwrap();
            let seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            let block_group_id = BlockGroup::get_id("", Sample::DEFAULT_NAME, "insertion", None);
            let seqs = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
            assert_eq!(
                seqs,
                HashSet::from_iter([
                    seq.clone(),
                    format!("{}{}", &seq[..1425].to_string(), &seq[2220..].to_string()).to_string()
                ])
            );
        }

        #[test]
        fn imports_annotations_when_genbank_has_edits() {
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
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default().annotation_name_from_path(&path),
            );

            let expected_group = "GenBank /reference/insertion";
            assert_eq!(
                AnnotationGroup::query_by_sample(conn, Sample::DEFAULT_NAME),
                vec![AnnotationGroup {
                    name: expected_group.to_string()
                }]
            );
            let annotations = Annotation::query_by_group(conn, expected_group).unwrap();
            let lac_z = annotations
                .iter()
                .find(|annotation| annotation.name == "lacZalpha")
                .unwrap();
            let lac_z_blocks = Accession::get_by_id(conn, &lac_z.accession_id)
                .unwrap()
                .blocks(conn)
                .unwrap()
                .into_iter()
                .filter(|block| {
                    block.node_id != PATH_START_NODE_ID && block.node_id != PATH_END_NODE_ID
                })
                .map(|block| (block.sequence_start, block.sequence_end))
                .collect::<Vec<_>>();
            assert_eq!(lac_z_blocks, vec![(55, 436)]);

            let inserted_cds = annotations
                .iter()
                .find(|annotation| annotation.name == "AAA73390.1")
                .unwrap();
            let inserted_cds_blocks = Accession::get_by_id(conn, &inserted_cds.accession_id)
                .unwrap()
                .blocks(conn)
                .unwrap()
                .into_iter()
                .filter(|block| {
                    block.node_id != PATH_START_NODE_ID && block.node_id != PATH_END_NODE_ID
                })
                .map(|block| (block.sequence_start, block.sequence_end))
                .collect::<Vec<_>>();
            assert_eq!(inserted_cds_blocks, vec![(0, 795)]);
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
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
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
            let seqs = BlockGroup::get_all_sequences(conn, &block_group_id, false).unwrap();
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
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
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
            )
            .unwrap();
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
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
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
            )
            .unwrap();
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
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
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
            .unwrap()
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
            let unchanged_seq = get_unmodified_sequence();
            assert!(sequences.contains(&mod_seq));
            assert!(sequences.contains(&unchanged_seq));
        }
    }
}
