use anyhow::{Error, Result};
use gen_models::{
    db::DbContext,
    errors::OperationError,
    operations::OperationInfo,
    session_operations::{end_operation, start_operation},
};
use thiserror::Error;

use crate::{
    commands::get_default_collection,
    graph_operators::{GraphOperationError, make_stitch},
};

#[derive(Debug, Error, PartialEq)]
pub enum MakeStitchOperationError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Error stitching subgraphs: {0}")]
    StitchingError(String),
}

pub fn make_stitch_operation(
    db_context: &DbContext,
    name: Option<String>,
    sample: Option<String>,
    new_sample: String,
    regions: String,
    new_region: String,
) -> Result<(), Error> {
    println!("Make stitch called");

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

    let region_names = regions.split(",").collect::<Vec<&str>>();

    match make_stitch(
        db_context,
        collection_name,
        sample_name.as_deref(),
        &new_sample_name,
        &region_names,
        &new_region,
    ) {
        Ok(_) => {}
        Err(GraphOperationError::OperationError(OperationError::NoChanges)) => {
            println!("Warning: No changes made.");
        }
        Err(e) => {
            graph_conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(MakeStitchOperationError::StitchingError(format!(
                "Error stitching subgraphs: {e}"
            ))
            .into());
        }
    }

    let summary_str = format!(
        " {}: stitched {} chunks into new graph",
        new_sample_name,
        region_names.len()
    );

    let _op = end_operation(
        db_context,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: "make stitch".to_string(),
        },
        &summary_str,
        None,
    )
    .map_err(GraphOperationError::OperationError)?;

    graph_conn.execute("END TRANSACTION;", [])?;
    operation_conn.execute("END TRANSACTION;", [])?;

    println!("Make stitch finished.");

    Ok(())
}
