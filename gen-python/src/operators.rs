use core::ops::Range;
use log::debug;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::python_api::block_group::PyBlockGroup;
use r#gen::graph_operators::{derive_chunks, make_stitch, GraphOperationError};

/// Helper function to handle transactions and execute an operation
pub(crate) fn with_transaction<F, T>(
    context: &gen_models::db::DbContext,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(&gen_models::db::DbContext) -> Result<T, GraphOperationError>,
{
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    if let Err(err) = r#gen::track_database(conn, operation_conn) {
        return Err(PyRuntimeError::new_err(format!(
            "Error tracking database: {err}"
        )));
    }

    conn.execute("BEGIN TRANSACTION", [])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to begin transaction: {e}")))?;
    operation_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to begin transaction: {e}")))?;

    let result = operation(context);

    match result {
        Ok(value) => {
            conn.execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit transaction: {e}")))?;
            operation_conn
                .execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit transaction: {e}")))?;
            Ok(value)
        }
        Err(err) => {
            let _ = conn.execute("ROLLBACK TRANSACTION;", []);
            let _ = operation_conn.execute("ROLLBACK TRANSACTION;", []);
            Err(map_graph_operation_error(err))
        }
    }
}

/// Maps GraphOperationError to Python exceptions with detailed error messages
pub(crate) fn map_graph_operation_error(err: GraphOperationError) -> PyErr {
    match err {
        GraphOperationError::OperationError(op_err) => {
            PyRuntimeError::new_err(format!("Operation failed: {op_err}"))
        }
        GraphOperationError::InvalidCoordinate(msg) => {
            PyRuntimeError::new_err(format!("Invalid coordinate: {msg}"))
        }
        GraphOperationError::RegionNotFound(msg) => {
            PyRuntimeError::new_err(format!("Region not found: {msg}"))
        }
        GraphOperationError::PathNotFound(msg) => {
            PyRuntimeError::new_err(format!("Path not found: {msg}"))
        }
    }
}


/// TODO: consider dropping this requirement here and on the Rust side
/// Validates that all block groups are from the same collection and sample
pub(crate) fn validate_block_groups_for_stitch(
    block_groups: &[PyBlockGroup],
) -> PyResult<(&str, Option<&str>)> {
    if block_groups.is_empty() {
        return Err(PyRuntimeError::new_err(
            "At least one parent block group is required.",
        ));
    }

    let collection_name = &block_groups[0].collection_name;
    let parent_sample_name = block_groups[0].sample_name.as_deref();

    for bg in block_groups {
        if bg.collection_name != *collection_name {
            return Err(PyRuntimeError::new_err(
                "All block groups must be from the same collection.",
            ));
        }
        if bg.sample_name.as_deref() != parent_sample_name {
            return Err(PyRuntimeError::new_err(
                "All block groups must be from the same sample.",
            ));
        }
    }

    Ok((collection_name, parent_sample_name))
}

/// Derives chunks from a parent block group.
///
/// Args:
///     context: The database context
///     parent_block_group: The parent block group to derive from
///     new_sample_name: Name of the new sample to create
///     chunk_ranges: List of (start, end) tuples defining the chunk ranges
///     backbone: Optional backbone path name
///
/// Returns:
///     Success message string
#[pyfunction]
pub fn derive_chunks_py(
    context: PyRef<'_, crate::PyDbContext>,
    parent_block_group: PyBlockGroup,
    new_sample_name: String,
    chunk_ranges: Vec<(i64, i64)>,
    backbone: Option<String>,
) -> PyResult<String> {
    debug!("derive_chunks called for block group: {}", parent_block_group.name);

    let context = &context.0;

    // Convert Python tuples to Rust Range<i64>
    let ranges: Vec<Range<i64>> = chunk_ranges
        .into_iter()
        .map(|(start, end)| Range { start, end })
        .collect();

    with_transaction(context, |ctx| {
        derive_chunks(
            ctx,
            &parent_block_group.collection_name,
            parent_block_group.sample_name.as_deref(),
            &new_sample_name,
            &parent_block_group.name,
            backbone.as_deref(),
            ranges,
        )
    })?;

    Ok("Chunks derived successfully.".to_string())
}

// TODO: clarify how "region" is used here and elsewhere, it basically the sequence name (or block group name)
/// Stitches multiple block groups together into a new sample, under a new name.
///
/// Args:
///     context: The database context
///     parent_block_groups: List of parent block groups to stitch together
///     new_sample_name: Name of the new sample to create
///     new_region_name: Name for the new stitched region
///
/// Returns:
///     Success message string
#[pyfunction]
pub fn make_stitch_py(
    context: PyRef<'_, crate::PyDbContext>,
    parent_block_groups: Vec<PyBlockGroup>,
    new_sample_name: String,
    new_region_name: String,
) -> PyResult<String> {
    debug!(
        "make_stitch called for {} block groups, new region: {}",
        parent_block_groups.len(),
        new_region_name
    );

    let context = &context.0;

    let (collection_name, parent_sample_name) =
        validate_block_groups_for_stitch(&parent_block_groups)?;

    let region_names: Vec<&str> = parent_block_groups
        .iter()
        .map(|bg| bg.name.as_str())
        .collect();

    with_transaction(context, |ctx| {
        make_stitch(
            ctx,
            collection_name,
            parent_sample_name,
            &new_sample_name,
            &region_names,
            &new_region_name,
        )
    })?;

    Ok("Stitch completed successfully.".to_string())
}
