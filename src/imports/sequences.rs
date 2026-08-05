use std::collections::HashMap;

use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::DbContext,
    edge::Edge,
    errors::{CollectionError, SampleError},
    node::Node,
    operations::{OperationInfo, OperationSummary},
    path::Path,
    sample::Sample,
    sequence::Sequence,
};

use crate::fasta::FastaError;
pub fn import_sequences(
    context: &DbContext,
    entries: &[(String, String)],
    collection_name: &str,
    sample: &str,
) -> Result<OperationSummary, FastaError> {
    let conn = context.graph().conn();

    let collection = match Collection::create(conn, collection_name) {
        Ok(collection) => collection,
        Err(CollectionError::Duplicate(collection)) => collection,
        Err(e) => return Err(FastaError::CollectionError(e)),
    };

    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample,
            ..Default::default()
        },
    ) {
        Ok(_) => {}
        Err(SampleError::Duplicate(_)) => {}
        Err(e) => return Err(FastaError::SampleError(e)),
    }

    let mut summary: HashMap<String, i64> = HashMap::new();

    for (name, sequence) in entries {
        let sequence_length = sequence.len() as i64;
        let seq = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn)?;
        let node_id = Node::create(
            conn,
            &seq.hash,
            &HashId::convert_str(&format!(
                "{collection}.{name}:{hash}",
                collection = collection.name,
                hash = seq.hash
            )),
        )?;
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &collection.name,
                sample_name: sample,
                name,
                ..Default::default()
            },
        )?;
        let edge_into = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )?;
        let edge_out_of = Edge::create(
            conn,
            node_id,
            sequence_length,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )?;

        let new_block_group_edges = vec![
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
        ];

        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        let path = Path::create(conn, name, &block_group.id, &[edge_into.id, edge_out_of.id])?;
        summary.entry(path.name).or_insert(sequence_length);
    }

    let mut summary_str = String::new();
    for (path_name, change_count) in &summary {
        summary_str.push_str(&format!(" {path_name}: {change_count} changes.\n"));
    }

    Ok(OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: "sequence_addition".to_string(),
        },
        summary_str,
    ))
}

/// Stores each reference sequence as a node (idempotent — same sequence hash reuses the same
/// node) and creates one BlockGroup per region that slices into that node at [start, end).
/// Coordinates are 0-based, half-open. Calling this multiple times with overlapping reference
/// sequences is safe and cheap: the node is stored once regardless.
pub fn import_genomic_regions(
    context: &DbContext,
    reference_sequences: &[(String, String)],
    regions: &[(String, String, i64, i64)],
    collection_name: &str,
    sample: &str,
) -> Result<OperationSummary, FastaError> {
    let conn = context.graph().conn();

    let collection = match Collection::create(conn, collection_name) {
        Ok(collection) => collection,
        Err(CollectionError::Duplicate(collection)) => collection,
        Err(e) => return Err(FastaError::CollectionError(e)),
    };

    match Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: sample,
            ..Default::default()
        },
    ) {
        Ok(_) => {}
        Err(SampleError::Duplicate(_)) => {}
        Err(e) => return Err(FastaError::SampleError(e)),
    }

    // Store each reference sequence and record its node id. Node::create is idempotent on
    // constraint violation, so the same chromosome imported twice reuses the existing node.
    let mut node_ids: HashMap<&str, HashId> = HashMap::new();
    for (seq_name, sequence) in reference_sequences {
        let seq = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn)?;
        let node_id = Node::create(
            conn,
            &seq.hash,
            &HashId::convert_str(&format!(
                "{collection}.{seq_name}:{hash}",
                collection = collection.name,
                hash = seq.hash
            )),
        )?;
        node_ids.insert(seq_name.as_str(), node_id);
    }

    let mut summary: HashMap<String, i64> = HashMap::new();

    for (region_name, seq_name, start, end) in regions {
        let node_id = node_ids.get(seq_name.as_str()).copied().ok_or_else(|| {
            FastaError::IOError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no reference sequence '{seq_name}' for region '{region_name}'"),
            ))
        })?;

        let region_len = end - start;
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &collection.name,
                sample_name: sample,
                name: region_name,
                ..Default::default()
            },
        )?;
        let edge_into = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            *start,
            Strand::Forward,
        )?;
        let edge_out_of = Edge::create(
            conn,
            node_id,
            *end,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )?;

        let new_block_group_edges = vec![
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
        ];

        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        let path = Path::create(
            conn,
            region_name,
            &block_group.id,
            &[edge_into.id, edge_out_of.id],
        )?;
        summary.entry(path.name).or_insert(region_len);
    }

    let mut summary_str = String::new();
    for (path_name, change_count) in &summary {
        summary_str.push_str(&format!(" {path_name}: {change_count} changes.\n"));
    }

    Ok(OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: "genomic_region_addition".to_string(),
        },
        summary_str,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use gen_models::{
        assets::{OperationKind, OperationLog},
        errors::OperationError,
        history::{HistoryStore, dolt::DoltHistoryStore},
        operations::commit_operation_summary,
        traits::Query,
    };

    use super::*;
    use crate::test_helpers::setup_gen;

    #[test]
    fn test_import_sequences_detects_no_changes() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(conn);
        let entries = vec![("chr1".to_string(), "ATCG".to_string())];

        let operation_summary =
            import_sequences(&context, &entries, "test", Sample::DEFAULT_NAME).unwrap();
        let commit_hash = commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));
        let mut operation_logs = OperationLog::all(conn);
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("sequence_addition".to_string())
        );

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "chr1", None);
        assert_eq!(
            gen_models_graph_tests::get_all_sequences_with_pruning(conn, &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCG".to_string()])
        );

        let operation_summary =
            import_sequences(&context, &entries, "test", Sample::DEFAULT_NAME).unwrap();
        let err = commit_operation_summary(&context, &operation_summary).unwrap_err();
        assert_eq!(err, OperationError::NoChanges);
    }

    #[test]
    fn test_import_genomic_regions_returns_commit_hash() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(conn);
        let reference_sequences = vec![("chr1".to_string(), "ATCGAT".to_string())];
        let regions = vec![("region-a".to_string(), "chr1".to_string(), 1, 5)];

        let operation_summary = import_genomic_regions(
            &context,
            &reference_sequences,
            &regions,
            "test",
            Sample::DEFAULT_NAME,
        )
        .unwrap();
        let commit_hash = commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));
        let mut operation_logs = OperationLog::all(conn);
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("genomic_region_addition".to_string())
        );

        let block_group_id = BlockGroup::get_id("test", Sample::DEFAULT_NAME, "region-a", None);
        assert_eq!(
            gen_models_graph_tests::get_all_sequences_with_pruning(conn, &block_group_id, false)
                .unwrap(),
            HashSet::from_iter(vec!["TCGA".to_string()])
        );
    }
}
