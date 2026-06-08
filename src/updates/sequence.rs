use std::str;

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    edge::Edge,
    node::Node,
    operations::{Operation, OperationInfo},
    region::{GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind},
    sample::Sample,
    sequence::Sequence,
    traits::*,
};
use rusqlite::{self, params};

use crate::{
    errors::SequenceUpdateError,
    updates::{
        InsertChangeData, insert_update_change, resolve_update_region, target_update_region,
    },
};

#[allow(clippy::too_many_arguments)]
pub fn update_with_sequence(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_name: &str,
    sequence: &str,
    disable_reference_path_update: bool,
) -> Result<Operation, SequenceUpdateError> {
    let conn = context.graph().conn();
    let mut session = gen_models::session_operations::start_operation(conn);
    let parsed_region = Region::parse(region_name).map_err(GenRegionError::from)?;
    let resolved_region =
        resolve_update_region(&parsed_region, conn, collection_name, parent_sample_name)?;
    if parsed_region.start.is_none() && parsed_region.end.is_none() {
        return Err(SequenceUpdateError::MissingCoordinates(
            region_name.to_string(),
        ));
    }
    let _new_sample = Sample::get_or_create_child(
        conn,
        collection_name,
        new_sample_name,
        vec![parent_sample_name.to_string()],
    )?;
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

        if block_group.name == resolved_region.block_group.name {
            target_block_groups = new_block_groups;
        }
    }

    if target_block_groups.is_empty() {
        return Err(GenRegionError::NotFound(region_name.to_string()).into());
    }

    for target_block_group in &target_block_groups {
        let path = BlockGroup::get_current_path(conn, &target_block_group.id)?;
        let (start_coordinate, end_coordinate) = (resolved_region.start, resolved_region.end);
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

            insert_sequence_change(
                conn,
                &resolved_region,
                target_block_group,
                &path,
                path_block,
            )?;
            node_id
        } else {
            let seq = Sequence::new()
                .sequence_type("DNA")
                .sequence(sequence)
                .save(conn)?;
            let node_id = Node::create(
                conn,
                &seq.hash,
                &HashId::convert_str(&format!(
                    "{block_group_id}:{ref_start}-{ref_end}->{sequence_hash}",
                    block_group_id = target_block_group.id,
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

            insert_sequence_change(
                conn,
                &resolved_region,
                target_block_group,
                &path,
                path_block,
            )?;
            node_id
        };

        if !disable_reference_path_update && resolved_region.kind == ResolvedRegionKind::Path {
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
                )?;
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

fn insert_sequence_change(
    conn: &gen_models::db::GraphConnection,
    region: &ResolvedGenRegion,
    target_block_group: &BlockGroup,
    path: &gen_models::path::Path,
    block: PathBlock,
) -> Result<(), SequenceUpdateError> {
    let source = target_update_region(conn, region, target_block_group.id, Some(path))?;
    let data = InsertChangeData::new(block);
    insert_update_change(conn, source, data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_core::NO_CHROMOSOME_INDEX;
    use gen_models::{
        annotations::{Annotation, add_annotation},
        block_group::{BlockGroup, BlockGroupChange, PathCache},
        path::Path,
        region::{ResolvedGenRegion, resolve_annotation},
        sample_lineage::SampleLineage,
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_block_group, setup_gen},
        track_database,
    };

    fn insertion_block(
        conn: &gen_models::db::GraphConnection,
        name: &str,
        sequence: &str,
    ) -> PathBlock {
        let seq = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn)
            .unwrap();
        let node_id = Node::create(conn, &seq.hash, &HashId::convert_str(name)).unwrap();

        PathBlock {
            node_id,
            block_sequence: sequence.to_string(),
            sequence_start: 0,
            sequence_end: seq.length,
            path_start: 0,
            path_end: 0,
            strand: Strand::Forward,
        }
    }

    #[test]
    fn accession_update_does_not_require_target_path() {
        let conn = crate::test_helpers::get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let mut path_cache = PathCache::new(&conn);
        let accession =
            BlockGroup::add_accession(&conn, &path, "target-acc", 10, 30, &mut path_cache).unwrap();
        Path::delete(&conn, "chr1", &block_group_id);

        let region = ResolvedGenRegion::from_accession(&conn, &accession, 5, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insertion_block(&conn, "acc-no-path-node", "NNNN"),
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge: true,
        };

        BlockGroup::insert_change(&conn, &change).unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap(),
            HashSet::from_iter([
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTNNNNCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn annotation_update_does_not_require_target_path() {
        let conn = crate::test_helpers::get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let mut path_cache = PathCache::new(&conn);
        let accession =
            BlockGroup::add_accession(&conn, &path, "target-ann-acc", 10, 30, &mut path_cache)
                .unwrap();
        let annotation =
            Annotation::get_or_create(&conn, "target-ann", "genes", &accession.id, None).unwrap();
        Path::delete(&conn, "chr1", &block_group_id);

        let region =
            ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, -5, 25).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insertion_block(&conn, "ann-no-path-node", "NNNN"),
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge: true,
        };

        BlockGroup::insert_change(&conn, &change).unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap(),
            HashSet::from_iter([
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAANNNNGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn update_sequence_with_annotation_negative_start_after_fasta_import() {
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
            "simple",
            false,
        )
        .unwrap();
        add_annotation(&context, &collection, "foobar", None, "simple", "m123:5-20").unwrap();
        assert!(
            resolve_annotation(
                &Region::parse("foobar:-3-5").unwrap(),
                conn,
                &collection,
                "simple"
            )
            .is_ok()
        );

        let result = update_with_sequence(
            &context,
            &collection,
            "simple",
            "derived",
            "foobar:-3-5",
            "AAA",
            false,
        );

        assert!(result.is_ok(), "{result:?}");
        assert!(
            resolve_annotation(
                &Region::parse("foobar:-3-5").unwrap(),
                conn,
                &collection,
                "derived"
            )
            .is_ok()
        );
        let block_group = get_sample_bg(conn, &collection, "derived");
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_group.id, false).unwrap(),
            HashSet::from_iter([
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATAAACGATCGATCGGGAACACACAGAGA".to_string(),
            ])
        );
    }

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
            "m123:2-5",
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
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
            HashSet::from_iter(expected_sequences),
        );
        assert_eq!(
            SampleLineage::get_parents(conn, "child sample"),
            vec![Sample::DEFAULT_NAME.to_string()],
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
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        let _ = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "other sample",
            "m123:2-5",
            "AAAAAAAA",
            true,
        );

        let child_blockgroup = get_sample_bg(conn, &collection, "child sample").id;
        let other_blockgroup = get_sample_bg(conn, &collection, "other sample").id;
        let child_path = BlockGroup::get_current_path(conn, &child_blockgroup).unwrap();
        let other_path = BlockGroup::get_current_path(conn, &other_blockgroup).unwrap();
        assert_eq!(
            child_path.sequence(conn).unwrap(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(
            other_path.sequence(conn).unwrap(),
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
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing part of the first update sequence
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:4-6",
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
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
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
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:1-6",
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
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
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
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:1-12",
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
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
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
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Second sequence update replacing parts of both the original and first update sequences
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:6-12",
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
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
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
            "m123:2-5",
            "AAAAAAAA",
            false,
        );
        // Same sequence second time
        let _ = update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:4-6",
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
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
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
            "m123:2-5",
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
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
            HashSet::from_iter(expected_sequences),
        );

        let latest_path = BlockGroup::get_current_path(conn, &block_groups[0].id).unwrap();
        assert_eq!(
            latest_path.sequence(conn).unwrap(),
            "ATTCGATCGATCGATCGGGAACACACAGAGA"
        );
    }
}
