use std::{
    cmp::{max, min},
    io::Read,
    path::Path as FsPath,
    str,
};

use gb_io::reader;
use gen_annotations::projection::AnnotationSegment;
use gen_core::{
    HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand,
    range::{Range, merge_ordered_items},
};
use gen_models::{
    accession::{Accession, AccessionEdge, AccessionEdgeData},
    annotations::Annotation,
    block_group::{BlockGroup, BlockGroupChange, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::Edge,
    errors::CollectionError,
    node::Node,
    operations::{Operation, OperationInfo},
    path::Path,
    region::ResolvedGenRegion,
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
                            BlockGroupChange {
                                region: region.clone(),
                                path_accession: None,
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
                        EditType::Deletion => BlockGroupChange {
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
                        },
                    };
                    BlockGroup::insert_change(conn, &change).unwrap();
                }

                if options.add_annotations {}
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
