use core::ops::Range;
use log::debug;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::python_api::block_group::PyBlockGroup;
use r#gen::graph_operators::{create_block_group, derive_chunks, make_stitch, GraphOperationError};

// NOTE: The transaction management code has been removed because it should include the operation tracking
// as well, ideally in a Python context manager (implement __enter__ and __exit__ methods). The problem is
// that the stitch and chunk operations are tracked within their function, so we can't use a context manager
// to do something like this:
// with repo.commit("Design knockouts"):
//     for gene in genes:
//         # find CRISPR target
//         # design deletion (chunk + stitch for example, probably a dedicated function )
//         derive_chunks(genome, [0, ])
//         # extract homology regions
//         # assemble payload

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

/// Creates a new block group with optional sequence data.
///
/// Args:
///     context: The database context
///     name: Name of the block group
///     collection_name: Name of the collection
///     sample_name: Optional sample name
///     sequence: Optional DNA sequence string
///
/// Returns:
///     PyBlockGroup instance
#[pyfunction]
#[pyo3(signature = (context, name, collection_name, sample_name = None, sequence = None))]
pub fn create_block_group_py(
    context: PyRef<'_, crate::PyDbContext>,
    name: String,
    collection_name: String,
    sample_name: Option<String>,
    sequence: Option<String>,
) -> PyResult<PyBlockGroup> {
    let context = &context.0;
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

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

    match create_block_group(
        context,
        &name,
        &collection_name,
        sample_name.as_deref(),
        sequence.as_deref(),
    ) {
        Ok(_op) => {
            conn.execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit: {e}")))?;
            operation_conn
                .execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit: {e}")))?;

            // Query for the created block group to get its actual ID
            use gen_models::{block_group::BlockGroup, traits::Query};
            let block_groups = BlockGroup::query(
                conn,
                "SELECT * FROM block_groups WHERE collection_name = ?1 AND name = ?2 ORDER BY created_on DESC LIMIT 1",
                rusqlite::params![&collection_name, &name],
            );

            if let Some(bg) = block_groups.into_iter().next() {
                Ok(PyBlockGroup {
                    id: bg.id,
                    collection_name: bg.collection_name,
                    sample_name: bg.sample_name,
                    name: bg.name,
                })
            } else {
                Err(PyRuntimeError::new_err("Failed to retrieve created block group"))
            }
        }
        Err(err) => {
            let _ = conn.execute("ROLLBACK TRANSACTION;", []);
            let _ = operation_conn.execute("ROLLBACK TRANSACTION;", []);
            Err(map_graph_operation_error(err))
        }
    }
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
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

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

    // Convert Python tuples to Rust Range<i64>
    let ranges: Vec<Range<i64>> = chunk_ranges
        .into_iter()
        .map(|(start, end)| Range { start, end })
        .collect();

    match derive_chunks(
        context,
        &parent_block_group.collection_name,
        parent_block_group.sample_name.as_deref(),
        &new_sample_name,
        &parent_block_group.name,
        backbone.as_deref(),
        ranges,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit: {e}")))?;
            operation_conn
                .execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit: {e}")))?;
            Ok("Chunks derived successfully.".to_string())
        }
        Err(err) => {
            let _ = conn.execute("ROLLBACK TRANSACTION;", []);
            let _ = operation_conn.execute("ROLLBACK TRANSACTION;", []);
            Err(map_graph_operation_error(err))
        }
    }
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
    let conn = context.graph().conn();
    let operation_conn = context.operations().conn();

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

    let (collection_name, parent_sample_name) =
        validate_block_groups_for_stitch(&parent_block_groups)?;

    let region_names: Vec<&str> = parent_block_groups
        .iter()
        .map(|bg| bg.name.as_str())
        .collect();

    match make_stitch(
        context,
        collection_name,
        parent_sample_name,
        &new_sample_name,
        &region_names,
        &new_region_name,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit: {e}")))?;
            operation_conn
                .execute("END TRANSACTION;", [])
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to commit: {e}")))?;
            Ok("Stitch completed successfully.".to_string())
        }
        Err(err) => {
            let _ = conn.execute("ROLLBACK TRANSACTION;", []);
            let _ = operation_conn.execute("ROLLBACK TRANSACTION;", []);
            Err(map_graph_operation_error(err))
        }
    }
}
