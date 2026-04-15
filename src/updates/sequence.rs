use std::str;

use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, PathChange},
    db::DbContext,
    edge::Edge,
    node::Node,
    operations::{Operation, OperationInfo},
    sample::Sample,
    sequence::Sequence,
    traits::*,
};
use rusqlite::{self, params};

use crate::errors::SequenceUpdateError;

#[allow(clippy::too_many_arguments)]
pub fn update_with_sequence(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_name: &str,
    start_coordinate: i64,
    end_coordinate: i64,
    sequence: &str,
    disable_reference_path_update: bool,
) -> Result<Operation, SequenceUpdateError> {
    let conn = context.graph().conn();
    let mut session = gen_models::session_operations::start_operation(conn);

    let _new_sample = Sample::get_or_create(conn, new_sample_name);
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name);

    let mut target_block_groups = vec![];
    for block_group in block_groups {
        let new_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            collection_name,
            new_sample_name,
            &block_group.name,
            vec![parent_sample_name.to_string()],
        )?;

        if block_group.name == region_name {
            target_block_groups = new_block_groups;
        }
    }

    if target_block_groups.is_empty() {
        panic!("No region found with name: {region_name}");
    }

    for target_block_group in &target_block_groups {
        let path = BlockGroup::get_current_path(conn, &target_block_group.id);
        let interval_tree = path.intervaltree(conn);
        let node_id = if sequence.is_empty() {
            let node_id = HashId::convert_str("");
            let path_block = PathBlock {
                node_id,
                block_sequence: sequence.to_string(),
                sequence_start: 0,
                sequence_end: 0,
                path_start: start_coordinate,
                path_end: end_coordinate,
                strand: Strand::Forward,
            };

            let path_change = PathChange {
                block_group_id: target_block_group.id,
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
            node_id
        } else {
            let seq = Sequence::new()
                .sequence_type("DNA")
                .sequence(sequence)
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
            )?;

            let path_block = PathBlock {
                node_id,
                block_sequence: sequence.to_string(),
                sequence_start: 0,
                sequence_end: seq.length,
                path_start: start_coordinate,
                path_end: end_coordinate,
                strand: Strand::Forward,
            };

            let path_change = PathChange {
                block_group_id: target_block_group.id,
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
            node_id
        };

        if !disable_reference_path_update {
            if node_id == HashId::convert_str("") {
                let _ = path.new_path_with_deletion(conn, start_coordinate, end_coordinate);
            } else {
                let edge_to_new_node = Edge::query(
                    conn,
                    "select * from edges where target_node_id = ?1",
                    params![node_id],
                )[0]
                .clone();
                let edge_from_new_node = Edge::query(
                    conn,
                    "select * from edges where source_node_id = ?1",
                    params![node_id],
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
    }

    let summary_str =
        format!("Sequences {mod}", mod=if sequence.is_empty() { "deleted" } else { "inserted" });
    let op = gen_models::session_operations::end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: "fasta_update".to_string(),
        },
        &summary_str,
        None,
    )
    .unwrap();

    println!("Updated with sequence.");

    Ok(op)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_gen},
        track_database,
    };

    #[test]
    fn test_update_with_sequence() {
        /*
        Graph after sequence update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> AAAAAAAA --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "child sample"],
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
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "other sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
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
    fn test_update_within_update() {
        /*
        Graph after sequence updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> TTTTTTTT --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        let _ = import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing part of the first update sequence
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123",
            4,
            6,
            "TTTTTTTT",
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
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_sequences_partial_leading_overlap() {
        /*
        Graph after sequence updates:
        A --> T --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
         \       \-> AAAA -------> AAAA --/
          \--> TTTTTTTT --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123",
            1,
            6,
            "TTTTTTTT",
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
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_sequences_partial_trailing_overlap() {
        /*
        Graph after sequence updates:
        A --> T --------------> CGA ----------------> TC --> GATCGATCGATCGGGAACACACAGAGA
         \       \-----> AAAAAAAA ---------/             /
          \-------------> TTTTTTTT ---------------------/
        */
        /*
        Graph after sequence updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123",
            1,
            12,
            "TTTTTTTT",
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
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_two_sequences_second_over_first() {
        /*
        Graph after sequence updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123",
            6,
            12,
            "TTTTTTTT",
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
            params![collection, "grandchild sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );
    }

    #[test]
    fn test_update_with_same_sequence_twice() {
        /*
        Graph after sequence updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> AAAAAAAA --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Same sequence second time
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123",
            4,
            6,
            "AAAAAAAA",
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
            params![collection, "grandchild sample"],
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
        Graph after sequence update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> -------- --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            "",
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATTCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "child sample"],
        );
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false),
            HashSet::from_iter(expected_sequences),
        );

        let latest_path = BlockGroup::get_current_path(conn, &block_groups[0].id);
        assert_eq!(
            latest_path.sequence(conn),
            "ATTCGATCGATCGATCGGGAACACACAGAGA"
        );
    }
}
