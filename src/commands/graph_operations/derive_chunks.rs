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
    #[error("Invalid breakpoint: {0}")]
    InvalidBreakpoint(String),
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
    sample: Option<String>,
    new_sample: String,
    region: String,
    backbone: Option<String>,
    breakpoints: Option<String>,
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
        sample_name.as_deref(),
        &region_name.to_string(),
        backbone.as_deref(),
    )?
    .length(graph_conn);

    let chunk_points = if let Some(breakpoints) = breakpoints {
        let mut result = vec![];
        for breakpoint in breakpoints.split(",") {
            match breakpoint.parse::<i64>() {
                Ok(parsed_value) => result.push(parsed_value),
                Err(_) => {
                    return Err(DeriveChunksOperationError::InvalidBreakpoint(format!(
                        "Invalid breakpoint: {breakpoint}"
                    ))
                    .into());
                }
            }
        }

        result.into_iter().sorted().collect::<Vec<i64>>()
    } else if let Some(chunk_size) = chunk_size {
        let chunk_count = path_length / chunk_size;
        (0..chunk_count)
            .map(|i| i * chunk_size)
            .collect::<Vec<i64>>()
    } else {
        return Err(DeriveChunksOperationError::NoChunkingMethod(
            "No chunking method specified.".to_string(),
        )
        .into());
    };

    if chunk_points.is_empty() {
        return Err(DeriveChunksOperationError::NoChunkCoordinates(
            "No chunk coordinates provided.".to_string(),
        )
        .into());
    }
    if chunk_points[chunk_points.len() - 1] > path_length {
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
        sample_name.as_deref(),
        &new_sample_name,
        &region_name.to_string(),
        backbone.as_deref(),
        chunk_ranges,
    ) {
        graph_conn.execute("ROLLBACK TRANSACTION;", [])?;
        operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(err.into());
    }

    let summary_str = format!(
        " {}: {} new derived block group(s)",
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
