use core::ops::Range;

use anyhow::{Error, Result};
use gen_models::{
    db::DbContext,
    errors::OperationError,
    operations::OperationInfo,
    session_operations::{end_operation, start_operation},
};
use itertools::Itertools;
use thiserror::Error;

use crate::{
    commands::get_default_collection,
    graphs::operators::{derive_chunks, get_path},
};

#[derive(Debug, Error, PartialEq)]
pub enum DeriveChunksOperationError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("No chunking method provided: {0}")]
    NoChunkingMethod(String),
    #[error("No chunk coordinates provided: {0}")]
    NoChunkCoordinates(String),
    #[error("At least one chunk coordinate exceeds path length: {0}")]
    PathLengthExceeded(String),
}

/// Given a sample and region (block group), splits the block group into subgraphs based on coordinates along a path.  A
/// path can be specified by the backbone parameter, otherwise we use the latest path for the block group.  The chunk
/// boundaries can be specified either with breakpoints (specific coordinates given as a comma separated string) or with
/// chunk_size.  Given a chunk size of n, we split up the sequence graph into subgraphs of length n along the input
/// path, along with some remainder graph of length <= n at the downstream end.
#[allow(clippy::too_many_arguments)]
pub fn derive_chunks_operation(
    db_context: &DbContext,
    name: Option<String>,
    sample: String,
    new_sample: String,
    region: String,
    backbone: Option<String>,
    breakpoints: Option<Vec<i64>>,
    chunk_size: Option<i64>,
) -> Result<(), Error> {
    let operation_conn = db_context.operations().conn();
    let graph_conn = db_context.graph().conn();

    let mut session = start_operation(graph_conn);

    graph_conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let collection_name = &(match name {
        Some(collection) => collection,
        None => get_default_collection(operation_conn),
    });
    let sample_name = sample.clone();
    let new_sample_name = new_sample.clone();
    let region_name = region.clone();

    let path_length = get_path(
        graph_conn,
        collection_name,
        sample_name.as_str(),
        &region_name.to_string(),
        backbone.as_deref(),
    )?
    .length(graph_conn)?;

    let chunk_points = if let Some(breakpoints) = breakpoints {
        if breakpoints.is_empty() {
            return Err(DeriveChunksOperationError::NoChunkCoordinates(
                "No chunk coordinates provided.".to_string(),
            )
            .into());
        }
        breakpoints.into_iter().sorted().collect::<Vec<i64>>()
    } else if let Some(chunk_size) = chunk_size {
        if chunk_size >= path_length {
            vec![]
        } else {
            let chunk_count = path_length / chunk_size;
            (1..=chunk_count)
                .map(|i| i * chunk_size)
                .filter(|&p| p < path_length)
                .collect::<Vec<i64>>()
        }
    } else {
        return Err(DeriveChunksOperationError::NoChunkingMethod(
            "No chunking method specified.".to_string(),
        )
        .into());
    };
    if chunk_points.last().is_some_and(|&p| p > path_length) {
        return Err(DeriveChunksOperationError::PathLengthExceeded(
            "At least one chunk coordinate exceeds path length.".to_string(),
        )
        .into());
    }

    let mut range_start = 0;
    let mut chunk_ranges = vec![];
    for chunk_point in chunk_points {
        chunk_ranges.push(Range {
            start: range_start,
            end: chunk_point,
        });
        range_start = chunk_point;
    }
    chunk_ranges.push(Range {
        start: range_start,
        end: path_length,
    });

    let chunk_range_length = chunk_ranges.len();

    if let Err(err) = derive_chunks(
        db_context,
        collection_name,
        sample_name.as_str(),
        &new_sample_name,
        &region_name.to_string(),
        backbone.as_deref(),
        chunk_ranges,
        None,
        true,
    ) {
        graph_conn.execute("ROLLBACK TRANSACTION;", [])?;
        operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(err.into());
    }

    let summary_str = format!(
        " {}: {} new derived sequence graph(s)",
        new_sample_name, chunk_range_length,
    );

    let _op = end_operation(
        db_context,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: "derive chunks".to_string(),
        },
        &summary_str,
        None,
    )
    .map_err(DeriveChunksOperationError::OperationError)?;

    graph_conn.execute("END TRANSACTION;", [])?;
    operation_conn.execute("END TRANSACTION;", [])?;

    println!("Derive chunks succeeded.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_models::{block_group::BlockGroup, collection::Collection, sample::Sample};

    use super::*;
    use crate::{imports::fasta::import_fasta, test_helpers::setup_gen, track_database};

    fn setup_with_fasta(fasta: &str) -> DbContext {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();
        Collection::create(conn, "test").unwrap();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(fasta);
        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        context
    }

    fn chunk_sequences(context: &DbContext, sample: &str) -> Vec<String> {
        let conn = context.graph().conn();
        let mut chunks = Sample::get_block_groups(conn, "test", sample);
        chunks.sort_by_key(|bg| bg.name.clone());
        chunks
            .iter()
            .map(|bg| {
                BlockGroup::get_current_path(conn, &bg.id)
                    .sequence(conn)
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn empty_breakpoints_errors() {
        let context = setup_with_fasta("fixtures/simple.fa");
        let result = derive_chunks_operation(
            &context,
            Some("test".into()),
            Sample::DEFAULT_NAME.into(),
            "chunks".into(),
            "m123".into(),
            None,
            Some(vec![]),
            None,
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("No chunk coordinates provided"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn chunk_size_with_remainder() {
        // simple.fa: m123 = 34 bp.  chunk_size=10 → 3 full chunks + 1 remainder of 4.
        // Regression: the previous (0..chunk_count) range produced a 0..0 first chunk,
        // causing an empty interval-tree query and an index-out-of-bounds panic.
        let context = setup_with_fasta("fixtures/simple.fa");
        derive_chunks_operation(
            &context,
            Some("test".into()),
            Sample::DEFAULT_NAME.into(),
            "chunks".into(),
            "m123".into(),
            None,
            None,
            Some(10),
        )
        .unwrap();

        let seqs = chunk_sequences(&context, "chunks");
        assert_eq!(seqs.len(), 4, "expected 3 full chunks + 1 remainder");
        assert!(seqs.iter().all(|s| !s.is_empty()), "no chunk may be empty");
        assert_eq!(seqs.concat(), "ATCGATCGATCGATCGATCGGGAACACACAGAGA");
    }

    #[test]
    fn chunk_size_divides_evenly() {
        // chunk_size=17 divides 34 bp exactly → 2 chunks, no empty trailing range.
        let context = setup_with_fasta("fixtures/simple.fa");
        derive_chunks_operation(
            &context,
            Some("test".into()),
            Sample::DEFAULT_NAME.into(),
            "chunks".into(),
            "m123".into(),
            None,
            None,
            Some(17),
        )
        .unwrap();

        let seqs = chunk_sequences(&context, "chunks");
        assert_eq!(seqs.len(), 2);
        assert!(seqs.iter().all(|s| !s.is_empty()), "no chunk may be empty");
        assert_eq!(seqs.concat(), "ATCGATCGATCGATCGATCGGGAACACACAGAGA");
    }

    #[test]
    fn chunk_size_larger_than_sequence() {
        // chunk_size >= path length → single chunk covering the whole sequence.
        let context = setup_with_fasta("fixtures/simple.fa");
        derive_chunks_operation(
            &context,
            Some("test".into()),
            Sample::DEFAULT_NAME.into(),
            "chunks".into(),
            "m123".into(),
            None,
            None,
            Some(100),
        )
        .unwrap();

        let seqs = chunk_sequences(&context, "chunks");
        assert_eq!(seqs.len(), 1);
        assert_eq!(seqs[0], "ATCGATCGATCGATCGATCGGGAACACACAGAGA");
    }
}
