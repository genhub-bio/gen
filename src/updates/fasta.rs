use std::str;

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    block_group::BlockGroup,
    db::DbContext,
    edge::Edge,
    file_types::FileTypes,
    node::Node,
    operations::{OperationFile, OperationInfo, OperationSummary},
    region::{GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind},
    sample::Sample,
    sequence::Sequence,
};
use noodles::fasta;

use crate::{
    fasta::FastaError,
    updates::{
        InsertChangeData, insert_update_change, resolve_update_region, target_update_region,
    },
};
#[allow(clippy::too_many_arguments)]
pub fn update_with_fasta(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_name: &str,
    fasta_file_path: &str,
    disable_reference_path_update: bool,
) -> Result<OperationSummary, FastaError> {
    let conn = context.graph().conn();
    let parsed_region = Region::parse(region_name).map_err(GenRegionError::from)?;
    let resolved_region =
        resolve_update_region(&parsed_region, conn, collection_name, parent_sample_name)?;
    let (start_coordinate, end_coordinate) =
        if parsed_region.start.is_some() || parsed_region.end.is_some() {
            (resolved_region.start, resolved_region.end)
        } else {
            return Err(FastaError::MissingCoordinates(region_name.to_string()));
        };

    let mut fasta_reader = fasta::io::reader::Builder.build_from_path(fasta_file_path)?;

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

    struct TargetBlockGroupState {
        block_group_id: HashId,
        path: Option<gen_models::path::Path>,
        first_node: Option<HashId>,
    }

    let mut target_states = target_block_groups
        .iter()
        .map(|target_block_group| -> Result<_, FastaError> {
            let path = if resolved_region.kind == ResolvedRegionKind::Path {
                Some(BlockGroup::get_current_path(
                    conn,
                    &target_block_group.id,
                    None,
                )?)
            } else {
                None
            };
            Ok(TargetBlockGroupState {
                block_group_id: target_block_group.id,
                path,
                first_node: None,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut change_count = 0;

    for (index, result) in fasta_reader.records().enumerate() {
        let record = result?;
        let sequence = str::from_utf8(record.sequence().as_ref()).unwrap();

        for state in &mut target_states {
            if sequence.is_empty() {
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

                insert_fasta_change(
                    conn,
                    context.workspace(),
                    &resolved_region,
                    state.block_group_id,
                    state.path.as_ref(),
                    path_block,
                )?;
                if index == 0 {
                    state.first_node = Some(node_id);
                } else if state.first_node.is_some() {
                    state.first_node = None;
                }
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
                        block_group_id = state.block_group_id,
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

                insert_fasta_change(
                    conn,
                    context.workspace(),
                    &resolved_region,
                    state.block_group_id,
                    state.path.as_ref(),
                    path_block,
                )?;
                if index == 0 {
                    state.first_node = Some(node_id);
                } else if state.first_node.is_some() {
                    state.first_node = None;
                }
            }
            change_count += 1;
        }
    }

    for state in target_states {
        if !disable_reference_path_update
            && resolved_region.kind == ResolvedRegionKind::Path
            && let Some(node_id) = state.first_node
            && let Some(path) = state.path
        {
            if node_id == HashId::convert_str("") {
                let _ = path.new_path_with_deletion(conn, start_coordinate, end_coordinate);
            } else {
                let edge_to_new_node = Edge::select(conn)
                    .target_node_id(node_id)
                    .load()
                    .expect("should load edge to inserted node")[0]
                    .clone();
                let edge_from_new_node = Edge::select(conn)
                    .source_node_id(node_id)
                    .load()
                    .expect("should load edge from inserted node")[0]
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

    let summary_str = format!("{change_count} sequences inserted");
    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![
                OperationFile::new(fasta_file_path.to_string()).set_file_type(FileTypes::Fasta),
            ],
            description: "fasta_update".to_string(),
        },
        summary_str,
    );

    println!("Updated with fasta file: {fasta_file_path}");

    Ok(operation_summary)
}

fn insert_fasta_change(
    conn: &gen_models::db::GraphConnection,
    workspace: &gen_core::Workspace,
    region: &ResolvedGenRegion,
    target_block_group_id: HashId,
    path: Option<&gen_models::path::Path>,
    block: PathBlock,
) -> Result<(), FastaError> {
    let source = target_update_region(conn, region, target_block_group_id, path)?;
    let data = InsertChangeData::new(block);
    insert_update_change(conn, workspace, source, data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use std::{collections::HashSet, io::Write, path::PathBuf};

    use gen_models::{
        annotations::add_annotation,
        assets::{OperationKind, OperationLog},
        history::{HistoryStore, dolt::DoltHistoryStore},
        operations::commit_operation_summary,
        path::Path,
        traits::Query,
    };
    use rusqlite::types::Value as SQLValue;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_gen},
    };

    #[test]
    fn test_update_with_fasta() {
        /*
        Graph after fasta update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> AAAAAAAA --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(conn);

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aaaaaaaa.fa");

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
        let operation_summary = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update_path.to_str().unwrap(),
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
    fn test_disable_reference_path_update() {
        // This tests if we stop updating the reference path if explicitly asked for when there
        // is a single insert occurring
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fasta_update_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/aaaaaaaa.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update_path.to_str().unwrap(),
            false,
        );
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "other sample",
            "m123:2-5",
            fasta_update_path.to_str().unwrap(),
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
    fn test_update_with_multiple_entries() {
        /*
        Graph after fasta update:
           /-> GGGG --\
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> CCCC --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let fasta_update_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/fastas/multiple.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
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
    fn test_update_within_update() {
        /*
        Graph after fasta updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> TTTTTTTT --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing part of the first update sequence
        let _ = update_with_fasta(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:4-6",
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
    fn test_update_with_two_fastas_partial_leading_overlap() {
        /*
        Graph after fasta updates:
        A --> T --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
         \       \-> AAAA -------> AAAA --/
          \--> TTTTTTTT --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_fasta(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:1-6",
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
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_fasta(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:1-12",
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
    fn test_update_with_two_fastas_second_over_first() {
        /*
        Graph after fasta updates:
        AT --------------> CGA ------------> TC --> GATCGATCGATCGGGAACACACAGAGA
              \-> AAAA -------> AAAA ----/        /
                           \--> TTTTTTTT --------/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update1_path.push("fixtures/aaaaaaaa.fa");
        let mut fasta_update2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update2_path.push("fixtures/tttttttt.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update1_path.to_str().unwrap(),
            false,
        );
        // Second fasta update replacing parts of both the original and first update sequences
        let _ = update_with_fasta(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:6-12",
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
    fn test_update_with_same_fasta_twice() {
        /*
        Graph after fasta updates:
        AT --------------> CGA ----------------> TCGATCGATCGATCGGGAACACACAGAGA
            \-> AA -----> AA -------> AAAA --/
                   \--> AAAAAAAA --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aaaaaaaa.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update_path.to_str().unwrap(),
            false,
        );
        // Same fasta second time
        let _ = update_with_fasta(
            &context,
            &collection,
            "child sample",
            "grandchild sample",
            "m123:4-6",
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
    fn update_with_fasta_accepts_annotation_region_without_target_path() {
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

        Sample::get_or_create_child(conn, &collection, "derived", vec!["simple".to_string()])
            .unwrap();
        let derived_block_group = get_sample_bg(conn, &collection, "derived");
        Path::delete(conn, "m123", &derived_block_group.id);

        let mut update_fasta = NamedTempFile::new().unwrap();
        update_fasta.write_all(b">update\nAAA\n").unwrap();

        let result = update_with_fasta(
            &context,
            &collection,
            "simple",
            "derived",
            "foobar:-3-5",
            update_fasta.path().to_str().unwrap(),
            false,
        );

        assert!(result.is_ok(), "{result:?}");
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
    fn test_deletion() {
        /*
        Graph after fasta update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> -------- --/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/empty.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
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
            rusqlite::params![collection, "child sample".to_string()],
        );
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

    #[test]
    fn test_update_before_start() {
        /*
        Graph after fasta update:
        <start node> ----> ATCGA ------> TCGATCGATCGATCGGGAACACACAGAGA
                    \---> AAAAAAAA ----/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aaaaaaaa.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:0-5",
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "AAAAAAAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
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
    fn test_update_after_end() {
        /*
        Graph after fasta update:
        ATCGATCGATCGATCGATCGGGAACACAC ---> AGAGA ------> <end node>
                                     \---> AAAAAAAA ----/
        */
        let context = setup_gen();
        let conn = context.graph().conn();

        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aaaaaaaa.fa");

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
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:29-34",
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let expected_sequences = vec![
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
            "ATCGATCGATCGATCGATCGGGAACACACAAAAAAAA".to_string(),
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
}
