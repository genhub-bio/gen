use std::str;

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    node::Node,
    operations::{OperationInfo, OperationSummary},
    region::{GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind},
    sample::Sample,
    sequence::Sequence,
};

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
                context.workspace(),
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
                context.workspace(),
                &resolved_region,
                target_block_group,
                &path,
                path_block,
            )?;
            node_id
        };

        if !disable_reference_path_update && resolved_region.kind == ResolvedRegionKind::Path {
            if node_id == HashId::convert_str("") {
                path.new_path_with_deletion(conn, start_coordinate, end_coordinate)?;
            } else {
                // The new node can have an edge from every route meeting the edit's
                // boundaries; splice in only the pair that continues the selected path.
                let (edge_to_new_node, edge_from_new_node) =
                    path.splice_edges_for_node(conn, node_id, start_coordinate, end_coordinate)?;
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
    workspace: &gen_core::Workspace,
    region: &ResolvedGenRegion,
    target_block_group: &BlockGroup,
    path: &gen_models::path::Path,
    block: PathBlock,
) -> Result<(), SequenceUpdateError> {
    let source = target_update_region(conn, region, target_block_group.id, Some(path))?;
    let data = InsertChangeData::new(block);
    insert_update_change(conn, workspace, source, data)?;
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
    use petgraph::algo::is_cyclic_directed;

    use super::*;
    use crate::{
        graphs::combinatorial_library::parse_library,
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_block_group, setup_gen},
        updates::library::update_with_library,
    };

    fn block_groups_for_sample(
        conn: &gen_models::db::GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Vec<BlockGroup> {
        BlockGroup::select(conn)
            .collection_name(collection_name)
            .sample_name(sample_name)
            .load()
            .expect("should query block groups for the sample")
    }

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

        BlockGroup::insert_change(&conn, crate::test_helpers::test_workspace(), &change).unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                &conn,
                crate::test_helpers::test_workspace(),
                &block_group_id,
                false
            )
            .unwrap(),
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

        BlockGroup::insert_change(&conn, crate::test_helpers::test_workspace(), &change).unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                &conn,
                crate::test_helpers::test_workspace(),
                &block_group_id,
                false
            )
            .unwrap(),
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
            &[],
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
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_group.id,
                false
            )
            .unwrap(),
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
            &[],
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
        let mut operation_logs = OperationLog::all(conn).expect("should load operation logs");
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("fasta_update".to_string())
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
        ];
        let block_groups = block_groups_for_sample(conn, &collection, "child sample");
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_groups[0].id,
                false
            )
            .unwrap(),
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
            &[],
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
            child_path
                .sequence(conn, context.workspace(), None)
                .unwrap(),
            "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(
            other_path
                .sequence(conn, context.workspace(), None)
                .unwrap(),
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
            &[],
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
        let block_groups = block_groups_for_sample(conn, &collection, "grandchild sample");
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_groups[0].id,
                false
            )
            .unwrap(),
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
            &[],
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
        let block_groups = block_groups_for_sample(conn, &collection, "grandchild sample");
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_groups[0].id,
                false
            )
            .unwrap(),
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
            &[],
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
        let block_groups = block_groups_for_sample(conn, &collection, "grandchild sample");
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_groups[0].id,
                false
            )
            .unwrap(),
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
            &[],
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
        let block_groups = block_groups_for_sample(conn, &collection, "grandchild sample");
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_groups[0].id,
                false
            )
            .unwrap(),
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
            &[],
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
        let block_groups = block_groups_for_sample(conn, &collection, "grandchild sample");
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_groups[0].id,
                false
            )
            .unwrap(),
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
            &[],
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
        let block_groups = block_groups_for_sample(conn, &collection, "child sample");
        assert_eq!(block_groups.len(), 1);
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &block_groups[0].id,
                false
            )
            .unwrap(),
            HashSet::from_iter(expected_sequences),
        );

        let latest_path = BlockGroup::get_current_path(conn, &block_groups[0].id, None).unwrap();
        assert_eq!(
            latest_path
                .sequence(conn, context.workspace(), None)
                .unwrap(),
            "ATTCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    const LIBRARY_PARTS: [&str; 3] = ["AAAA", "CAAC", "TAAT"];

    /// Puts the combinatorial library of `parts.fa` and `combinatorial_design.csv` into
    /// `m123:7-20` of `simple.fa` as sample "design": each of the three parts is followed by each
    /// of `cds1` (`ATGATAA`), `cds2` and `cds3`, so three routes arrive at the start of `cds1`.
    fn import_library_design(context: &DbContext, collection: &str) {
        let collection = &collection.to_string();
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let fixture = |name: &str| fixtures.join(name).to_str().unwrap().to_string();
        let parts_path = fixture("parts.fa");
        let library_path = fixture("combinatorial_design.csv");
        import_fasta(
            context,
            &fixture("simple.fa"),
            collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        add_annotation(
            context,
            collection,
            "SITE",
            None,
            Sample::DEFAULT_NAME,
            "m123:7-20",
        )
        .unwrap();
        update_with_library(
            context,
            collection,
            Sample::DEFAULT_NAME,
            "design",
            "SITE",
            parse_library(&parts_path, &library_path).unwrap(),
            Some(&parts_path),
            Some(&library_path),
        )
        .unwrap();
    }

    /// The sequences `sample` spells.
    fn sample_sequences(context: &DbContext, collection: &str, sample: &str) -> HashSet<String> {
        let block_group = get_sample_bg(context.graph().conn(), collection, sample);
        BlockGroup::get_all_sequences(
            context.graph().conn(),
            context.workspace(),
            &block_group.id,
            false,
        )
        .unwrap()
    }

    /// `m123` with the library site holding each of the three parts followed by `coding`.
    fn after_every_part(coding: &str) -> HashSet<String> {
        LIBRARY_PARTS
            .iter()
            .map(|part| format!("ATCGATC{part}{coding}GGAACACACAGAGA"))
            .collect()
    }

    /// Deleting the first base of `cds1`, where the three parts arrive, applies to all three.
    #[test]
    fn test_deletion_at_a_node_start_applies_to_every_route_into_it() {
        let context = setup_gen();
        let collection = "test";
        import_library_design(&context, collection);
        update_with_sequence(
            &context, collection, "design", "deleted", "cds1:0-1", "", false,
        )
        .unwrap();

        let mut expected = sample_sequences(&context, collection, "design");
        expected.extend(after_every_part("TGATAA"));
        assert_eq!(sample_sequences(&context, collection, "deleted"), expected);
    }

    /// Deleting the first base of a block applies to every route arriving at it, the same as at
    /// the start of a node. After `GG` is inserted at `cds1:3`, both the rest of `cds1` and `GG`
    /// arrive at `cds1:3`, so deleting `cds1:3-4` drops that base after either.
    #[test]
    fn test_deletion_at_a_block_start_applies_to_every_route_into_it() {
        let context = setup_gen();
        let collection = "test";
        import_library_design(&context, collection);
        update_with_sequence(
            &context, collection, "design", "inserted", "cds1:3-3", "GG", false,
        )
        .unwrap();
        let mut inserted = sample_sequences(&context, collection, "design");
        inserted.extend(after_every_part("ATGGGATAA"));
        assert_eq!(sample_sequences(&context, collection, "inserted"), inserted);

        update_with_sequence(
            &context,
            collection,
            "inserted",
            "inserted_deleted",
            "cds1:3-4",
            "",
            false,
        )
        .unwrap();

        let mut expected = inserted;
        expected.extend(after_every_part("ATGTAA"));
        expected.extend(after_every_part("ATGGGTAA"));
        assert_eq!(
            sample_sequences(&context, collection, "inserted_deleted"),
            expected
        );
    }

    /// Deleting the first base of `cds1`, then the next one, keeps every combination: either
    /// base alone or both, after each of the three parts.
    #[test]
    fn test_iterative_deletion_at_combinatorial_part_start_preserves_connectivity() {
        let context = setup_gen();
        let collection = "test";
        import_library_design(&context, collection);
        update_with_sequence(
            &context, collection, "design", "deleted", "cds1:0-1", "", false,
        )
        .unwrap();
        update_with_sequence(
            &context, collection, "deleted", "deleted2", "cds1:1-2", "", false,
        )
        .unwrap();

        let mut expected = sample_sequences(&context, collection, "design");
        expected.extend(after_every_part("TGATAA"));
        expected.extend(after_every_part("AGATAA"));
        expected.extend(after_every_part("GATAA"));
        assert_eq!(sample_sequences(&context, collection, "deleted2"), expected);
    }

    /// Inserting at the start or the end of `cds3` puts the insertion on every route through
    /// that end: after each of the three parts arriving at its start, and before the rest of
    /// `m123` after its end. The graph stays acyclic.
    #[test]
    fn test_insertion_at_a_node_start_or_end_applies_to_every_route_through_it() {
        for (region, spelled) in [("cds3:0-0", "GGATGCTAA"), ("cds3:7-7", "ATGCTAAGG")] {
            let context = setup_gen();
            let collection = "test";
            import_library_design(&context, collection);
            update_with_sequence(
                &context, collection, "design", "inserted", region, "GG", false,
            )
            .unwrap();

            let block_group = get_sample_bg(context.graph().conn(), collection, "inserted");
            let graph = BlockGroup::get_graph(
                context.graph().conn(),
                context.workspace(),
                &block_group.id,
                None,
            )
            .unwrap();
            assert!(
                !is_cyclic_directed(&graph),
                "{region} should leave the graph acyclic"
            );
            let mut expected = sample_sequences(&context, collection, "design");
            expected.extend(after_every_part(spelled));
            assert_eq!(
                sample_sequences(&context, collection, "inserted"),
                expected,
                "{region}"
            );
        }
    }

    fn import_simple_fixture(context: &gen_models::db::DbContext, collection: &str) {
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            context,
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
    }

    fn current_path_sequence(
        context: &gen_models::db::DbContext,
        collection: &str,
        sample_name: &str,
    ) -> String {
        let conn = context.graph().conn();
        let block_group = get_sample_bg(conn, collection, sample_name);
        let path = BlockGroup::get_current_path(conn, &block_group.id, None).unwrap();
        path.sequence(conn, context.workspace(), None).unwrap()
    }

    #[test]
    fn test_deletion_at_contig_start_updates_reference_path() {
        // Reference: ATCGATCGATCGATCGATCGGGAACACACAGAGA. Deleting 0-2 should
        // leave the selected path spelling the suffix, not the full reference.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:0-2",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "child sample"),
            "CGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_second_deletion_on_updated_path_updates_reference_path() {
        // First deletion removes CG at 2-4, leaving ATATCG... The second deletion
        // is resolved against the updated path, so 2-4 now removes AT.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "child sample"),
            "ATATCGATCGATCGATCGGGAACACACAGAGA"
        );
        update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "grandchild sample"),
            "ATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_deletion_after_insertion_updates_reference_path() {
        // Insert GG at 4, then delete 6-8 resolved against the inserted path.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:4",
            "GG",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "child sample"),
            "ATCGGGATCGATCGATCGATCGGGAACACACAGAGA"
        );
        update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:6-8",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "grandchild sample"),
            "ATCGGGCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_insertion_where_two_routes_arrive_updates_reference_path() {
        // After deleting 2-4, position 2 has two arriving routes (reference and
        // deletion bypass). Inserting there must splice the updated path, not fail
        // while picking one of several edges into the new node.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:2",
            "GG",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "grandchild sample"),
            "ATGGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_deletion_at_contig_end_updates_reference_path() {
        // Deleting the last bases of the contig leaves the path ending at the deletion
        // bypass, which runs straight into the path end node.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:30-34",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "child sample"),
            "ATCGATCGATCGATCGATCGGGAACACACA"
        );
    }

    #[test]
    fn test_whole_contig_deletion_updates_reference_path() {
        // Deleting the whole contig leaves a path that goes from the start node straight to
        // the end node and spells nothing.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:0-34",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "child sample"),
            ""
        );
    }

    #[test]
    fn test_deleting_whole_insertion_restores_reference_path() {
        // Deleting exactly the inserted GG, resolved against the inserted path, restores the
        // original reference.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:4-4",
            "GG",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:4-6",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "grandchild sample"),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_deletion_spanning_insertion_updates_reference_path() {
        // A deletion that spans the inserted GG and one reference base on each side splices
        // the path from the left reference block to the right one, skipping the insertion.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:4-4",
            "GG",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:3-7",
            "",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "grandchild sample"),
            "ATCTCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_replacement_after_deletion_keeps_deletion_in_reference_path() {
        // After deleting CG at 2-4, position 2 on the updated path is the junction where the
        // deletion bypass leaves the first block. A replacement there must splice in through
        // the bypass the path actually takes, not through a reference edge that would bring
        // the deleted CG back.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:2-3",
            "T",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "grandchild sample"),
            "ATTTCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_insertion_at_contig_start_updates_reference_path() {
        // Inserting at position 0 splices the new node in directly after the path start node.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:0-0",
            "GG",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "child sample"),
            "GGATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_insertion_at_contig_end_updates_reference_path() {
        // Inserting at the contig length splices the new node in directly before the path end
        // node.
        let context = setup_gen();
        let collection = "test".to_string();
        import_simple_fixture(&context, &collection);
        update_with_sequence(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:34-34",
            "GG",
            false,
        )
        .unwrap();
        assert_eq!(
            current_path_sequence(&context, &collection, "child sample"),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGAGG"
        );
    }
}
