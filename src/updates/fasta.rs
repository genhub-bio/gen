use std::str;

use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, PathChange},
    edge::Edge,
    file_types::FileTypes,
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    sample::Sample,
    sequence::Sequence,
    traits::*,
};
use noodles::fasta;
use rusqlite::{self, Connection, types::Value as SQLValue};

use crate::fasta::FastaError;

#[allow(clippy::too_many_arguments)]
pub fn update_with_fasta(
    conn: &Connection,
    operation_conn: &Connection,
    collection_name: &str,
    parent_sample_name: Option<&str>,
    new_sample_name: &str,
    region_name: &str,
    start_coordinate: i64,
    end_coordinate: i64,
    fasta_file_path: &str,
    disable_reference_path_update: bool,
) -> Result<Operation, FastaError> {
    let mut session = gen_models::session_operations::start_operation(conn);

    let mut fasta_reader = fasta::io::reader::Builder.build_from_path(fasta_file_path)?;

    let _new_sample = Sample::get_or_create(conn, new_sample_name);
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name);

    let mut found_bg_id = None;
    for block_group in block_groups {
        let new_bg_id = BlockGroup::get_or_create_sample_block_group(
            conn,
            collection_name,
            new_sample_name,
            &block_group.name,
            parent_sample_name,
        )?;

        if block_group.name == region_name {
            found_bg_id = Some(new_bg_id);
        }
    }

    let new_block_group_id = if let Some(x) = found_bg_id {
        x
    } else {
        panic!("No region found with name: {region_name}");
    };

    let path = BlockGroup::get_current_path(conn, &new_block_group_id);
    let interval_tree = path.intervaltree(conn);

    // Assuming just one entry in the fasta file
    let mut first_node = None;
    let mut change_count = 0;
    for (index, result) in fasta_reader.records().enumerate() {
        let record = result?;

        let sequence = str::from_utf8(record.sequence().as_ref())
            .unwrap()
            .to_string();
        if sequence.is_empty() {
            // We assume this is a deletion.
            let node_id = HashId::convert_str("");
            let path_block = PathBlock {
                id: -1,
                node_id,
                block_sequence: sequence,
                sequence_start: 0,
                sequence_end: 0,
                path_start: start_coordinate,
                path_end: end_coordinate,
                strand: Strand::Forward,
            };

            let path_change = PathChange {
                block_group_id: new_block_group_id,
                path: path.clone(),
                path_accession: None,
                start: start_coordinate,
                end: end_coordinate,
                block: path_block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };

            BlockGroup::insert_change(conn, &path_change, &interval_tree).unwrap();
            if index == 0 {
                first_node = Some(node_id);
            } else if first_node.is_some() {
                first_node = None;
            }
        } else {
            let seq = Sequence::new()
                .sequence_type("DNA")
                .sequence(&sequence)
                .save(conn);
            let node_id = Node::create(
                conn,
                &seq.hash,
                &HashId::convert_str(&format!(
                    "{path_id}:{ref_start}-{ref_end}->{sequence_hash}",
                    path_id = path.id,
                    ref_start = 0,
                    ref_end = seq.length,
                    sequence_hash = seq.hash
                )),
            );

            let path_block = PathBlock {
                id: -1,
                node_id,
                block_sequence: sequence,
                sequence_start: 0,
                sequence_end: seq.length,
                path_start: start_coordinate,
                path_end: end_coordinate,
                strand: Strand::Forward,
            };

            let path_change = PathChange {
                block_group_id: new_block_group_id,
                path: path.clone(),
                path_accession: None,
                start: start_coordinate,
                end: end_coordinate,
                block: path_block,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
                preserve_edge: true,
            };

            BlockGroup::insert_change(conn, &path_change, &interval_tree).unwrap();
            if index == 0 {
                first_node = Some(node_id);
            } else if first_node.is_some() {
                first_node = None;
            }
        }

        change_count += 1;
    }

    if !disable_reference_path_update && let Some(node_id) = first_node {
        if node_id == HashId::convert_str("") {
            let _ = path.new_path_with_deletion(conn, start_coordinate, end_coordinate);
        } else {
            let edge_to_new_node = Edge::query(
                conn,
                "select * from edges where target_node_id = ?1",
                rusqlite::params!(SQLValue::from(node_id)),
            )[0]
            .clone();
            let edge_from_new_node = Edge::query(
                conn,
                "select * from edges where source_node_id = ?1",
                rusqlite::params!(SQLValue::from(node_id)),
            )[0]
            .clone();
            path.new_path_with(
                conn,
                start_coordinate,
                end_coordinate,
                &edge_to_new_node,
                &edge_from_new_node,
            );
        }
    }

    let summary_str = format!("{change_count} sequences inserted");
    let op = gen_models::session_operations::end_operation(
        conn,
        operation_conn,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: fasta_file_path.to_string(),
                file_type: FileTypes::Fasta,
            }],
            description: "fasta_update".to_string(),
        },
        &summary_str,
        None,
    )
    .unwrap();

    println!("Updated with fasta file: {fasta_file_path}");

    Ok(op)
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use std::{collections::HashSet, path::PathBuf};

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_connection, get_operation_connection, get_sample_bg, setup_gen_dir},
        track_database,
    };

    #[test]
    fn test_update_with_fasta() {
        /*
        Graph after fasta update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> AAAAAAAA --/
        */
        setup_gen_dir();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aaaaaaaa.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("child sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_disable_reference_path_update() {
        // This tests if we stop updating the reference path if explicitly asked for when there
        // is a single insert occurring
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fasta_update_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/aaaaaaaa.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "other sample",
            "m123",
            2,
            5,
            fasta_update_path.to_str().unwrap(),
            true,
        );

        let child_blockgroup = get_sample_bg(conn, &collection, "child sample").id;
        let other_blockgroup = get_sample_bg(conn, &collection, "other sample").id;
        let child_path = BlockGroup::get_current_path(conn, &child_blockgroup);
        let other_path = BlockGroup::get_current_path(conn, &other_blockgroup);
        assert_eq!(
            child_path.sequence(conn),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(
            other_path.sequence(conn),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_update_with_multiple_entries() {
        /*
        Graph after fasta update:
           /-> GGGG --\
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> CCCC --/
        */
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fasta_update_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/multiple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATGGGGTCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATCCCCTCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("child sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_within_update() {
        /*
        Graph after fasta updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> TTTTTTTT --/
        */
        setup_gen_dir();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        let _ = import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing part of the first update sequence
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
            "grandchild sample",
            "m123",
            4,
            6,
            fasta_update2_path.to_str().unwrap(),
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAATTTTTTTTAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("grandchild sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_fastas_partial_leading_overlap() {
        /*
        Graph after fasta updates:
        A --> T --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
         \       \-> AAAA -------> AAAA --/
          \--> TTTTTTTT --/
        */
        setup_gen_dir();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
            "grandchild sample",
            "m123",
            1,
            6,
            fasta_update2_path.to_str().unwrap(),
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATTTTTTTTAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("grandchild sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_fastas_partial_trailing_overlap() {
        /*
        Graph after fasta updates:
        A --> T --------------> CGA ----------------> TC --> GATCGATCGATCGGGAACACACAGAGA
         \       \-----> AAAAAAAA ---------/             /
          \-------------> TTTTTTTT ---------------------/
        */
        /*
        Graph after fasta updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        setup_gen_dir();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
            "grandchild sample",
            "m123",
            1,
            12,
            fasta_update2_path.to_str().unwrap(),
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATTTTTTTTGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("grandchild sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_fastas_second_over_first() {
        /*
        Graph after fasta updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        setup_gen_dir();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
            "grandchild sample",
            "m123",
            6,
            12,
            fasta_update2_path.to_str().unwrap(),
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAATTTTTTTTGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("grandchild sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_same_fasta_twice() {
        /*
        Graph after fasta updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> AAAAAAAA --/
        */
        setup_gen_dir();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aaaaaaaa.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );
        // Same fasta second time
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
            "grandchild sample",
            "m123",
            4,
            6,
            fasta_update_path.to_str().unwrap(),
            false,
        );
        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("grandchild sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_deletion() {
        /*
        Graph after fasta update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> -------- --/
        */
        setup_gen_dir();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/empty.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let _ = update_with_fasta(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATTCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            rusqlite::params!(
                SQLValue::from(collection),
                SQLValue::from("child sample".to_string()),
            ),
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );

        // TODO: Check path of child sample
    }
}
