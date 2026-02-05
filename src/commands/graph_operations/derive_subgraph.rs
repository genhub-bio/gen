use core::ops::Range;

use anyhow::{Error, Result};
use gen_core::region::Region;
use gen_models::{
    db::DbContext,
    errors::OperationError,
    operations::OperationInfo,
    session_operations::{end_operation, start_operation},
};
use thiserror::Error;

use crate::{commands::get_default_collection, graphs::operators::derive_chunks};

#[derive(Debug, Error, PartialEq)]
pub enum DeriveSubgraphOperationError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Invalid region boundary: {0}")]
    RegionBoundary(String),
}

/// Given a sample and region (sequence graph name plus a coordinate range), creates a new sample and sequence graph
/// that is the result of cutting a subgraph out of the sequence graph of the parent sample.  The cutting part is
/// defined by the coordinates given by the region parameter, and can either be along the latest path or one specified
/// by the backbone parameter.
pub fn derive_subgraph_operation(
    db_context: &DbContext,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    region: String,
    backbone: Option<String>,
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

    let parsed_region = if let Some(region) = Region::parse(&region) {
        region
    } else {
        return Err(Error::msg(format!(
            "Failed to parse region string: {}",
            region
        )));
    };

    let start_coordinate = parsed_region.start;
    let end_coordinate = parsed_region.end;

    if let Err(err) = derive_chunks(
        db_context,
        collection_name,
        sample_name.as_deref(),
        &new_sample_name,
        &parsed_region.name.to_string(),
        backbone.as_deref(),
        vec![Range {
            start: start_coordinate,
            end: end_coordinate,
        }],
        None,
        true,
    ) {
        graph_conn.execute("ROLLBACK TRANSACTION;", [])?;
        operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(err.into());
    }

    let summary_str = format!(" {}: new derived block group", new_sample_name,);

    let _op = end_operation(
        db_context,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: "derive subgraph".to_string(),
        },
        &summary_str,
        None,
    )
    .map_err(DeriveSubgraphOperationError::OperationError)?;

    graph_conn.execute("END TRANSACTION;", [])?;
    operation_conn.execute("END TRANSACTION;", [])?;

    println!("Derive subgraph succeeded.");

    Ok(())
}
