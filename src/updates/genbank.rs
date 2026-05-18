use std::{io::Read, str};

use gb_io::reader;
use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup, PathChange},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::Edge,
    errors::CollectionError,
    node::Node,
    operations::{Operation, OperationInfo},
    path::Path,
    sequence::Sequence,
    session_operations::end_operation,
    traits::Query,
};
use itertools::Itertools;
use rusqlite::{params, types::Value};

use crate::genbank::{EditType, GenBankError, process_sequence};

pub fn update_with_genbank<'a, R>(
    context: &DbContext,
    data: R,
    collection: impl Into<Option<&'a str>>,
    sample_name: &str,
    create_missing: bool,
    operation_info: &OperationInfo,
) -> Result<Operation, GenBankError>
where
    R: Read,
{
    let conn = context.graph().conn();
    let mut session = gen_models::session_operations::start_operation(conn);
    let reader = reader::SeqReader::new(data);
    let collection = match Collection::create(conn, collection.into().unwrap_or_default()) {
        Ok(c) => c,
        Err(CollectionError::Duplicate(c)) => c,
        Err(e) => return Err(GenBankError::CollectionError(e)),
    };
    for result in reader {
        match result {
            Ok(seq) => {
                let locus = process_sequence(seq)?;
                let original_sequence = locus.original_sequence();
                let mut seq_model = Sequence::new().sequence(&original_sequence);
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
                        contig = &locus.name,
                        collection = &collection.name,
                        hash = sequence.hash
                    )),
                )?;

                let block_group = if let Ok(bg) = BlockGroup::get(
                    conn,
                    "select * from block_groups where collection_name = ?1 AND sample_name = ?2 AND name = ?3",
                    params![
                        Value::from(collection.name.clone()),
                        Value::from(sample_name.to_string()),
                        Value::from(locus.name.clone())
                    ],
                ) {
                    bg
                } else {
                    if !create_missing {
                        return Err(GenBankError::LookupError(format!(
                            "No block group named {contig} exists. Try importing first or pass --create-missing.",
                            contig = &locus.name
                        )));
                    }
                    BlockGroup::create(
                        conn,
                        NewBlockGroup {
                            collection_name: &collection.name,
                            sample_name,
                            name: &locus.name,
                            ..Default::default()
                        },
                    )?
                };
                let paths = Path::query(
                    conn,
                    "select * from paths where block_group_id = ?1 AND name = ?2",
                    params![Value::from(block_group.id), Value::from(locus.name.clone())],
                );
                let path = if let Some(first) = paths.first() {
                    first.clone()
                } else {
                    if !create_missing {
                        return Err(GenBankError::LookupError(format!(
                            "No path named {contig} exists. Try importing first or pass --create-missing.",
                            contig = &locus.name
                        )));
                    }
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
                    Path::create(
                        conn,
                        &locus.name,
                        &block_group.id,
                        &[edge_into.id, edge_out_of.id],
                    )?
                };
                for edit in locus.changes_to_wt() {
                    let start = edit.start;
                    let end = edit.end;
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
                            let change_node = Node::create(
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
                    let tree = path.intervaltree(conn)?;
                    BlockGroup::insert_change(conn, &change, &tree).unwrap();
                }
            }
            Err(e) => return Err(GenBankError::ParseError(format!("Failed to parse {e}"))),
        }
    }
    end_operation(
        context,
        &mut session,
        operation_info,
        &format!(
            "Update with GenBank {files}.",
            files = operation_info
                .files
                .iter()
                .map(|f| f.file_path.clone())
                .join(",")
        ),
        None,
    )
    .map_err(GenBankError::OperationError)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs::File, io::BufReader, path::PathBuf};

    use gen_models::{file_types::FileTypes, operations::OperationFile, sample::Sample};
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

    #[test]
    fn test_error_on_invalid_file() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        assert_eq!(
            update_with_genbank(
                &context,
                BufReader::new("this is not valid".as_bytes()),
                None,
                Sample::DEFAULT_NAME,
                false,
                &OperationInfo {
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
                    description: "test".to_string(),
                }
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
        Sample::create(
            conn,
            gen_models::sample::NewSample {
                name: Sample::DEFAULT_NAME,
                is_reference: false,
            },
        )
        .unwrap();
        let operation = update_with_genbank(
            &context,
            BufReader::new(file),
            None,
            Sample::DEFAULT_NAME,
            true,
            &OperationInfo {
                files: vec![
                    OperationFile::new(path.to_str().unwrap().to_string())
                        .set_file_type(FileTypes::GenBank),
                ],
                description: "test".to_string(),
            },
        )
        .unwrap();
        assert_eq!(
            Operation::get_by_id(op_conn, &operation.hash).unwrap(),
            operation
        );
    }

    #[cfg(test)]
    mod geneious_genbanks {
        use gen_models::{operations::OperationFile, sample::Sample};

        use super::*;
        use crate::{
            imports::genbank::{GenBankImportOptions, import_genbank},
            test_helpers::setup_gen,
            track_database,
        };

        #[test]
        fn test_incorporates_updates() {
            // This tests that we are able to take a genbank that has been further modified
            // and update it, mimicking a workflow of going between gen <-> 3rd party tool <-> gen
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

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/multiple_insertions_deletions.gb");
            let file = File::open(&path).unwrap();
            let _ = update_with_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                true,
                &OperationInfo {
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
                    description: "test".to_string(),
                },
            );

            let f = reader::parse_file(&path).unwrap();
            let mod_seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            let block_group_id = BlockGroup::get_id("", Sample::DEFAULT_NAME, "insertion", None);
            let sequences: HashSet<String> =
                BlockGroup::get_all_sequences(conn, &block_group_id, false)
                    .iter()
                    .map(|s| s.to_lowercase())
                    .collect();
            let unchanged_seq = get_unmodified_sequence();
            assert!(sequences.contains(&mod_seq));
            assert!(sequences.contains(&unchanged_seq));
        }

        #[test]
        fn test_creates_missing_entries() {
            // This tests that we are able to take a genbank that has been further modified
            // and includes new sequences and update it.
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

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/concat.gb");
            let file = File::open(&path).unwrap();
            let _ = update_with_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                true,
                &OperationInfo {
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
                    description: "test".to_string(),
                },
            );

            let f = reader::parse_file(&path).unwrap();
            let block_group_id = BlockGroup::get_id("", Sample::DEFAULT_NAME, "insertion", None);
            let sequences: HashSet<String> =
                BlockGroup::get_all_sequences(conn, &block_group_id, false)
                    .iter()
                    .map(|s| s.to_lowercase())
                    .collect();
            let unchanged_seq = get_unmodified_sequence();
            assert!(sequences.contains(&unchanged_seq));
            let mod_seq = str::from_utf8(&f[0].seq).unwrap().to_string();
            assert!(sequences.contains(&mod_seq));

            // we have a new blockgroup called deletion that uses the same base sequence but
            // has a deletion in it.
            let block_group_id = BlockGroup::get_id("", Sample::DEFAULT_NAME, "deletion", None);
            let sequences: HashSet<String> =
                BlockGroup::get_all_sequences(conn, &block_group_id, false)
                    .iter()
                    .map(|s| s.to_lowercase())
                    .collect();
            let mod_seq = str::from_utf8(&f[1].seq).unwrap().to_string();
            assert!(sequences.contains(&unchanged_seq));
            assert!(sequences.contains(&mod_seq));
        }

        #[test]
        fn test_errors_on_missing_locus() {
            // This tests that if a genbank file has sequences we are missing, it's an error. This
            // is an attempt to avoid updating the database with the wrong file.
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

            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/geneious_genbank/concat.gb");
            let file = File::open(&path).unwrap();
            let op = update_with_genbank(
                &context,
                BufReader::new(file),
                None,
                Sample::DEFAULT_NAME,
                false,
                &OperationInfo {
                    files: vec![
                        OperationFile::new("".to_string()).set_file_type(FileTypes::GenBank),
                    ],
                    description: "test".to_string(),
                },
            );
            assert!(op.is_err());
            assert_eq!(op, Err(GenBankError::LookupError("No block group named deletion exists. Try importing first or pass --create-missing.".to_string())));
        }
    }
}
