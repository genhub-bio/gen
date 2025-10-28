use std::str;

use gen_core::{HashId, NO_CHROMOSOME_INDEX, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, PathChange},
    edge::Edge,
    node::Node,
    operations::{Operation, OperationInfo},
    sample::Sample,
    sequence::Sequence,
    traits::*,
};
use rusqlite::{self, Connection, params};

use crate::errors::SequenceUpdateError;

#[allow(clippy::too_many_arguments)]
pub fn update_with_sequence(
    conn: &Connection,
    operation_conn: &Connection,
    collection_name: &str,
    parent_sample_name: Option<&str>,
    new_sample_name: &str,
    region_name: &str,
    start_coordinate: i64,
    end_coordinate: i64,
    sequence: &str,
    disable_reference_path_update: bool,
) -> Result<Operation, SequenceUpdateError> {
    let mut session = gen_models::session_operations::start_operation(conn);

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
    let node_id = if sequence.is_empty() {
        // We assume this is a deletion.
        let node_id = HashId::convert_str("");
        // This path block represents a deletion, so will not actually be
        // used to create a new node.  So the node ID can be anything, which
        // is why we're setting it to the HashId of the empty string.  The
        // important part is that sequence_start == sequence_end (the 0
        // values for them are arbitrary), which flags that it's a deletion
        // to the logic in BlockGroup::insert_change.
        let path_block = PathBlock {
            id: -1,
            node_id,
            block_sequence: sequence.to_string(),
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
        );

        let path_block = PathBlock {
            id: -1,
            node_id,
            block_sequence: sequence.to_string(),
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

    let summary_str =
        format!("Sequences {mod}", mod=if sequence.is_empty() { "deleted" } else { "inserted" });
    let op = gen_models::session_operations::end_operation(
        conn,
        operation_conn,
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
        test_helpers::{get_connection, get_operation_connection, get_sample_bg, setup_gen_dir},
        track_database,
    };

    #[test]
    fn test_update_with_sequence() {
        /*
        Graph after fasta update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> AAAAAAAA --/
        */
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
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
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
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
        Graph after fasta updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> TTTTTTTT --/
        */
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second fasta update replacing part of the first update sequence
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
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
        Graph after fasta updates:
        A --> T --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
         \       \-> AAAA -------> AAAA --/
          \--> TTTTTTTT --/
        */
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
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
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
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
        Graph after fasta updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
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
        Graph after fasta updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> AAAAAAAA --/
        */
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
            "child sample",
            "m123",
            2,
            5,
            "AAAAAAAA",
            false,
        );
        // Same fasta second time
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            Some("child sample"),
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
        Graph after fasta update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> -------- --/
        */
        setup_gen_dir();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
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
        let _ = update_with_sequence(
            conn,
            op_conn,
            &collection,
            None,
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
