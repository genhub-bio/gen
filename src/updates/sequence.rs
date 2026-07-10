use std::str;

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    edge::Edge,
    node::Node,
    operations::{OperationInfo, OperationSummary},
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
) -> Result<OperationSummary, SequenceUpdateError> {
    let conn = context.graph().conn();
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
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name, None);

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
        let path = BlockGroup::get_current_path(conn, &target_block_group.id, None)?;
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
    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: "fasta_update".to_string(),
        },
        summary_str,
    );

    println!("Updated with sequence.");

    Ok(operation_summary)
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
        assets::{OperationKind, OperationLog},
        block_group::{BlockGroup, BlockGroupChange, PathCache},
        history::{HistoryStore, dolt::DoltHistoryStore},
        operations::commit_operation_summary,
        path::Path,
        region::{ResolvedGenRegion, resolve_annotation},
        sample_lineage::SampleLineage,
    };

    use super::*;
    use crate::{
        graphs::combinatorial_library::parse_library,
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_block_group, setup_gen},
        track_database,
        updates::library::update_with_library,
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
        let history_store = DoltHistoryStore::new(conn);

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
        let operation_summary = update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            "AAAAAAAA",
            false,
        )
        .unwrap();
        let commit_hash = commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));
        let mut operation_logs = OperationLog::all(conn);
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("fasta_update".to_string())
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
            SampleLineage::get_parents(conn, "child sample", None),
            vec![Sample::DEFAULT_NAME.to_string()],
        );
    }

    #[test]
    fn test_disable_reference_path_update() {
        // This tests if we stop updating the reference path if explicitly asked for when there
        // is a single insert occurring
        let context = setup_gen();
        let conn = context.graph().conn();

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
        let child_path = BlockGroup::get_current_path(conn, &child_blockgroup, None).unwrap();
        let other_path = BlockGroup::get_current_path(conn, &other_blockgroup, None).unwrap();
        assert_eq!(
            child_path.sequence(conn, None).unwrap(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(
            other_path.sequence(conn, None).unwrap(),
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

        let latest_path = BlockGroup::get_current_path(conn, &block_groups[0].id, None).unwrap();
        assert_eq!(
            latest_path.sequence(conn, None).unwrap(),
            "ATTCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    // Reproduces https://github.com/genhub-bio/gen/issues/211: deleting a
    // sub-region of a part that a combinatorial library wired in from
    // multiple entry points (p1/p2/p3 -> cds1) should create a bypass edge
    // for each entry point, not just one. Mirrors the bash reproduction:
    // import -> add-annotation -> update library -> update sequence "".
    #[test]
    fn test_deletion_inside_combinatorial_library_part_creates_bypass_for_every_entry_point() {
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
            "wt",
            false,
        )
        .unwrap();
        add_annotation(&context, &collection, "SITE", None, "wt", "m123:7-20").unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path).unwrap();

        update_with_library(
            &context,
            &collection,
            "wt",
            "design",
            "SITE",
            parts_list,
            Some(parts_path),
            Some(library_path),
        )
        .unwrap();

        // Delete the first base of cds1's start codon, same as
        // `gen update sequence "" --region-name cds1:0-1` in the issue.
        let _ = update_with_sequence(
            &context,
            &collection,
            "design",
            "deleted",
            "cds1:0-1",
            "",
            false,
        );

        let block_groups = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection, "deleted"],
        );
        assert_eq!(block_groups.len(), 1);

        // Every combo that routes through cds1 (via p1, p2, or p3) should
        // gain a bypass-edge variant with the first base of cds1 removed,
        // alongside the untouched combos and the original unedited route.
        let expected_sequences = HashSet::from_iter(
            [
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
                "ATCGATCAAAAATGATAAGGAACACACAGAGA",
                "ATCGATCAAAAATGTTAAGGAACACACAGAGA",
                "ATCGATCAAAAATGCTAAGGAACACACAGAGA",
                "ATCGATCTAATATGATAAGGAACACACAGAGA",
                "ATCGATCTAATATGTTAAGGAACACACAGAGA",
                "ATCGATCTAATATGCTAAGGAACACACAGAGA",
                "ATCGATCCAACATGATAAGGAACACACAGAGA",
                "ATCGATCCAACATGTTAAGGAACACACAGAGA",
                "ATCGATCCAACATGCTAAGGAACACACAGAGA",
                // bypass variants: cds1's leading "A" removed, one per entry point
                "ATCGATCAAAAATGATAGGAACACACAGAGA",
                "ATCGATCTAATATGATAGGAACACACAGAGA",
                "ATCGATCCAACATGATAGGAACACACAGAGA",
            ]
            .map(String::from),
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &block_groups[0].id, false).unwrap(),
            expected_sequences,
        );
    }
}
