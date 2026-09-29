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
            let allele = (!sequence.is_empty()).then_some((node_id, 0, sequence.len() as i64));
            path.new_path_with_edit(conn, start_coordinate, end_coordinate, allele)?;
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

    use gen_core::{NO_CHROMOSOME_INDEX, is_end_node, is_start_node};
    use gen_graph::all_simple_paths;
    use gen_models::{
        annotations::{Annotation, add_annotation},
        assets::{OperationKind, OperationLog},
        block_group::{BlockGroup, BlockGroupChange, PathCache},
        block_group_edge::BlockGroupEdge,
        db::DbContext,
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

    /// The deletion edges `sample_name`'s current path takes on `m123` of `simple.fa`, in order:
    /// edges between the reference node and the path's start and end that skip bases.
    fn path_deletion_edges(
        context: &gen_models::db::DbContext,
        collection: &str,
        sample_name: &str,
    ) -> Vec<HashId> {
        let conn = context.graph().conn();
        let block_group = get_sample_bg(conn, collection, sample_name);
        let path = BlockGroup::get_current_path(conn, &block_group.id, None).unwrap();
        let reference = get_sample_bg(conn, collection, Sample::DEFAULT_NAME);
        let reference_path = BlockGroup::get_current_path(conn, &reference.id, None).unwrap();
        let reference_node_id =
            Path::edges_for_path(conn, &reference_path.id, None)[0].target_node_id;
        let position = |node_id: HashId, coordinate: i64| {
            if node_id == reference_node_id {
                Some(coordinate)
            } else if is_start_node(node_id) {
                Some(0)
            } else if is_end_node(node_id) {
                Some(34)
            } else {
                None
            }
        };
        Path::edges_for_path(conn, &path.id, None)
            .into_iter()
            .filter(|edge| {
                let source = position(edge.source_node_id, edge.source_coordinate);
                let target = position(edge.target_node_id, edge.target_coordinate);
                matches!((source, target), (Some(source), Some(target)) if source < target)
            })
            .map(|edge| edge.id)
            .collect()
    }

    /// Whether `sample`'s block group holds the edge `edge_id`.
    fn graph_has_edge(
        context: &gen_models::db::DbContext,
        collection: &str,
        sample: &str,
        edge_id: HashId,
    ) -> bool {
        let conn = context.graph().conn();
        let block_group = get_sample_bg(conn, collection, sample);
        BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None)
            .iter()
            .any(|augmented_edge| augmented_edge.edge.id == edge_id)
    }

    /// A second deletion starting where the first ended, on the updated path, removes the next
    /// bases. The route through both is an edge of its own, skipping both, and the path takes
    /// it; the first deletion stays in the graph.
    #[test]
    fn test_sequential_deletions_are_spliced_in_through_their_combination() {
        let context = setup_gen();
        let collection = "test";
        import_simple_fixture(&context, collection);
        update_with_sequence(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context,
            collection,
            "child sample",
            "grandchild sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();

        let first = path_deletion_edges(&context, collection, "child sample");
        let both = path_deletion_edges(&context, collection, "grandchild sample");
        assert_eq!(first.len(), 1);
        assert_eq!(both.len(), 1);
        assert_ne!(both[0], first[0]);
        assert!(graph_has_edge(
            &context,
            collection,
            "grandchild sample",
            first[0]
        ));
        assert_eq!(
            current_path_sequence(&context, collection, "grandchild sample"),
            "ATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    /// A deletion ending where an earlier one starts is spliced in through the edge skipping
    /// both; the earlier deletion stays in the graph.
    #[test]
    fn test_deletion_before_an_earlier_one_is_spliced_in_through_their_combination() {
        let context = setup_gen();
        let collection = "test";
        import_simple_fixture(&context, collection);
        update_with_sequence(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context,
            collection,
            "child sample",
            "grandchild sample",
            "m123:0-2",
            "",
            false,
        )
        .unwrap();

        let first = path_deletion_edges(&context, collection, "child sample");
        let both = path_deletion_edges(&context, collection, "grandchild sample");
        assert_eq!(both.len(), 1);
        assert_ne!(both[0], first[0]);
        assert!(graph_has_edge(
            &context,
            collection,
            "grandchild sample",
            first[0]
        ));
        assert_eq!(
            current_path_sequence(&context, collection, "grandchild sample"),
            "ATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    /// An insertion where a deletion ends goes after the deletion. The combination is an edge of
    /// its own, from where the deletion starts into the insertion, and the path takes it.
    #[test]
    fn test_insertion_after_a_deletion_is_spliced_in_through_their_combination() {
        let context = setup_gen();
        let collection = "test";
        import_simple_fixture(&context, collection);
        update_with_sequence(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-4",
            "",
            false,
        )
        .unwrap();
        update_with_sequence(
            &context,
            collection,
            "child sample",
            "grandchild sample",
            "m123:2",
            "GG",
            false,
        )
        .unwrap();

        let conn = context.graph().conn();
        let grandchild = get_sample_bg(conn, collection, "grandchild sample");
        let grandchild_path = BlockGroup::get_current_path(conn, &grandchild.id, None).unwrap();
        let reference_node_id =
            Path::edges_for_path(conn, &grandchild_path.id, None)[0].target_node_id;
        assert!(
            Path::edges_for_path(conn, &grandchild_path.id, None)
                .iter()
                .any(|edge| edge.source_node_id == reference_node_id
                    && edge.source_coordinate == 2
                    && edge.target_node_id != reference_node_id),
            "the path should leave the reference where the deletion starts, into the insertion"
        );
        assert_eq!(
            current_path_sequence(&context, collection, "grandchild sample"),
            "ATGGATCGATCGATCGATCGGGAACACACAGAGA"
        );
        let graph = BlockGroup::get_graph(conn, context.workspace(), &grandchild.id, None).unwrap();
        assert!(!is_cyclic_directed(&graph));
    }

    /// Two samples deleting the same bases share one deletion edge.
    #[test]
    fn test_same_deletion_in_two_samples_shares_its_edge() {
        let context = setup_gen();
        let collection = "test";
        import_simple_fixture(&context, collection);
        for sample in ["one", "other"] {
            update_with_sequence(
                &context,
                collection,
                Sample::DEFAULT_NAME,
                sample,
                "m123:10-12",
                "",
                false,
            )
            .unwrap();
        }

        let one = path_deletion_edges(&context, collection, "one");
        assert_eq!(one.len(), 1);
        assert_eq!(one, path_deletion_edges(&context, collection, "other"));
    }

    /// Deleting a whole contig leaves a path of one edge from the path start to its end.
    #[test]
    fn test_whole_contig_deletion_is_one_deletion_edge() {
        let context = setup_gen();
        let collection = "test";
        import_simple_fixture(&context, collection);
        update_with_sequence(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:0-34",
            "",
            false,
        )
        .unwrap();

        assert_eq!(
            path_deletion_edges(&context, collection, "child sample").len(),
            1
        );
        assert_eq!(
            current_path_sequence(&context, collection, "child sample"),
            ""
        );
    }

    /// Deleting bases a sample has already deleted along another route changes nothing and
    /// makes no cycle.
    #[test]
    fn test_reapplying_a_deletion_changes_nothing() {
        let context = setup_gen();
        let collection = "test";
        import_simple_fixture(&context, collection);
        update_with_sequence(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:10-12",
            "",
            true,
        )
        .unwrap();
        update_with_sequence(
            &context,
            collection,
            "child sample",
            "grandchild sample",
            "m123:10-12",
            "",
            true,
        )
        .unwrap();

        assert_eq!(
            sample_sequences(&context, collection, "grandchild sample"),
            sample_sequences(&context, collection, "child sample")
        );
        let conn = context.graph().conn();
        let block_group = get_sample_bg(conn, collection, "grandchild sample");
        let graph =
            BlockGroup::get_graph(conn, context.workspace(), &block_group.id, None).unwrap();
        assert!(!is_cyclic_directed(&graph));
    }

    /// Repeating an insertion and a deletion at one point, each on a new child sample, keeps
    /// every sample on its own route: the second insertion goes in front of the first, which is
    /// its sibling, rather than after it, and deleting it rejoins the path where it left.
    #[test]
    fn test_cli_insert_delete_cycle_at_one_point_does_not_chain_alternatives() {
        let context = setup_gen();
        let collection = "test";
        import_simple_fixture(&context, collection);

        update_with_sequence(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "s0",
            "m123:10-10",
            "CC",
            false,
        )
        .unwrap();
        update_with_sequence(&context, collection, "s0", "s1", "m123:10-12", "", false).unwrap();
        update_with_sequence(&context, collection, "s1", "s2", "m123:10-10", "CC", false).unwrap();
        update_with_sequence(&context, collection, "s2", "s3", "m123:10-12", "", false).unwrap();

        assert_eq!(
            current_path_sequence(&context, collection, "s3"),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    /// The number of routes from the path start to its end in `sample`'s pruned graph.
    fn route_count(context: &DbContext, collection: &str, sample: &str) -> usize {
        let block_group = get_sample_bg(context.graph().conn(), collection, sample);
        let mut graph = BlockGroup::get_graph(
            context.graph().conn(),
            context.workspace(),
            &block_group.id,
            None,
        )
        .unwrap();
        BlockGroup::prune_graph(&mut graph);
        let start = graph
            .nodes()
            .find(|node| is_start_node(node.node_id))
            .expect("should have a start node");
        graph
            .nodes()
            .filter(|node| is_end_node(node.node_id))
            .map(|end| all_simple_paths(&graph, start, end).count())
            .sum()
    }

    /// `m123` of `simple.fa` with `edits` applied, each replacing `start..end` with `sequence`
    /// in reference coordinates.
    fn with_edits(edits: &[&(usize, usize, &str)]) -> String {
        const REFERENCE: &str = "ATCGATCGATCGATCGATCGGGAACACACAGAGA";
        let mut edits = edits.to_vec();
        edits.sort_by_key(|(start, end, _)| (*start, *end));
        let mut spelled = String::new();
        let mut position = 0;
        for (start, end, sequence) in edits {
            spelled.push_str(&REFERENCE[position..*start]);
            spelled.push_str(sequence);
            position = *end;
        }
        spelled.push_str(&REFERENCE[position..]);
        spelled
    }

    /// Every pair of touching edits (deletion, substitution or insertion on either side of the
    /// point 10), made one after the other in either order and each keeping the reference, gives
    /// the four combinations of reference and alternative (RR, AR, RA and AA) through exactly four
    /// routes: every combination is written once, as an edge of its own.
    #[test]
    fn test_adjacent_edits_give_every_combination_once() {
        type Edit = (usize, usize, &'static str);
        const LEFT_OF_TEN: [Edit; 2] = [(9, 10, ""), (9, 10, "G")];
        const RIGHT_OF_TEN: [Edit; 2] = [(10, 11, ""), (10, 11, "A")];
        const INSERTION_AT_TEN: Edit = (10, 10, "GG");
        const INSERTION_AT_ELEVEN: Edit = (11, 11, "GG");

        let mut pairs = vec![];
        for left in LEFT_OF_TEN {
            for right in RIGHT_OF_TEN {
                pairs.push((left, right));
            }
            pairs.push((left, INSERTION_AT_TEN));
        }
        for right in RIGHT_OF_TEN {
            pairs.push((INSERTION_AT_TEN, right));
            pairs.push((right, INSERTION_AT_ELEVEN));
        }

        // The second edit is made on the first's path, so its region shifts by the length the
        // first added or removed unless it lies before it.
        let region_after = |first: &Edit, second: &Edit| {
            let shift = if second.0 >= first.1 && first != second {
                first.2.len() as i64 - (first.1 - first.0) as i64
            } else {
                0
            };
            format!(
                "m123:{}-{}",
                second.0 as i64 + shift,
                second.1 as i64 + shift
            )
        };

        let mut cases = vec![];
        for (left, right) in pairs {
            cases.push((left, right));
            cases.push((right, left));
        }
        let mut failures = vec![];
        for (first_edit, second_edit) in cases {
            // Once a base is deleted, its position on the path is the point after it, so an
            // insertion before that base cannot be addressed there.
            let is_deletion = first_edit.2.is_empty() && first_edit.1 > first_edit.0;
            if is_deletion && second_edit.0 == second_edit.1 && second_edit.0 == first_edit.0 {
                continue;
            }
            let first_region = format!("m123:{}-{}", first_edit.0, first_edit.1);
            let first = first_edit.2;
            let second_region = region_after(&first_edit, &second_edit);
            let second = second_edit.2;
            let context = setup_gen();
            let collection = "test";
            import_simple_fixture(&context, collection);
            update_with_sequence(
                &context,
                collection,
                Sample::DEFAULT_NAME,
                "first",
                &first_region,
                first,
                false,
            )
            .unwrap();
            update_with_sequence(
                &context,
                collection,
                "first",
                "second",
                &second_region,
                second,
                false,
            )
            .unwrap();

            let case = format!("{first_region} {first:?} then {second_region} {second:?}");
            let expected = HashSet::from([
                with_edits(&[]),
                with_edits(&[&first_edit]),
                with_edits(&[&second_edit]),
                with_edits(&[&first_edit, &second_edit]),
            ]);
            let sequences = sample_sequences(&context, collection, "second");
            let routes = route_count(&context, collection, "second");
            let current = current_path_sequence(&context, collection, "second");
            if sequences != expected
                || routes != 4
                || current != with_edits(&[&first_edit, &second_edit])
            {
                failures.push(format!(
                    "{case}: {} sequences, {routes} routes",
                    sequences.len()
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// Deleting a whole coding part, where three parts arrive, joins each of them to what
    /// follows the deleted part.
    #[test]
    fn test_whole_node_deletion_joins_every_node_before_it_to_the_node_after_it() {
        let context = setup_gen();
        let collection = "test";
        import_library_design(&context, collection);
        update_with_sequence(
            &context, collection, "design", "deleted", "cds1:0-7", "", false,
        )
        .unwrap();

        let mut expected = sample_sequences(&context, collection, "design");
        expected.extend(after_every_part(""));
        assert_eq!(sample_sequences(&context, collection, "deleted"), expected);
    }
}
