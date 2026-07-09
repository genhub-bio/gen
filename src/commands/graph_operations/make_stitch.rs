use anyhow::{Error, Result};
use gen_models::{
    db::DbContext,
    errors::OperationError,
    operations::{OperationInfo, OperationSummary},
};
use thiserror::Error;

use crate::{
    commands::{commit_operation, get_default_collection},
    graphs::operators::{GraphOperationError, make_stitch},
};

#[derive(Debug, Error, PartialEq)]
pub enum MakeStitchOperationError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Error stitching subgraphs: {0}")]
    StitchingError(String),
}

/// Given a sample and regions (sequence graph names) in the sample, creates a
/// new sequence graph in the new sample that is the result of "concatenating"
/// each region with the next.  The end nodes of each preceding region are given
/// edges to all the start nodes of the following region.
pub fn make_stitch_operation(
    db_context: &DbContext,
    name: Option<String>,
    sample: String,
    new_sample: String,
    regions: String,
    new_region: String,
) -> Result<(), Error> {
    let operation_conn = db_context.config().conn();
    let graph_conn = db_context.graph().conn();

    graph_conn.execute("BEGIN TRANSACTION", [])?;

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
        sample_name.as_str(),
        &new_sample_name,
        &region_names,
        &new_region,
    ) {
        Ok(_) => {}
        Err(GraphOperationError::OperationError(OperationError::NoChanges)) => {
            graph_conn.execute("ROLLBACK TRANSACTION;", [])?;
            println!("Warning: No changes made.");
            return Ok(());
        }
        Err(e) => {
            graph_conn.execute("ROLLBACK TRANSACTION;", [])?;
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

    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: "make stitch".to_string(),
        },
        summary_str,
    );
    match commit_operation(db_context, &operation_summary) {
        Ok(_) => {}
        Err(OperationError::NoChanges) => {
            println!("Warning: No changes made.");
            return Ok(());
        }
        Err(err) => return Err(GraphOperationError::OperationError(err).into()),
    }

    println!(
        "Stitched chunks successfully into new region {} in sample {}.",
        new_region, new_sample_name
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_models::{block_group::BlockGroup, collection::Collection, sample::Sample};

    use super::*;
    use crate::{imports::fasta::import_fasta, test_helpers::setup_gen};

    fn setup_with_chunks() -> DbContext {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let config_conn = context.config().conn();
        Collection::create(graph_conn, "test").unwrap();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            &context,
            &fasta_path.to_string_lossy().to_string(),
            "test",
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        crate::commands::graph_operations::derive_chunks::derive_chunks_operation(
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
        let _ = config_conn;
        context
    }

    #[test]
    fn test_make_stitch_operation_reconstructs_sequence() {
        let context = setup_with_chunks();
        let graph_conn = context.graph().conn();

        make_stitch_operation(
            &context,
            Some("test".into()),
            "chunks".into(),
            "stitched".into(),
            "m123.1,m123.2".into(),
            "m123".into(),
        )
        .unwrap();

        let stitched_block_group_id = BlockGroup::get_id("test", "stitched", "m123", None);
        let stitched_sequence =
            BlockGroup::get_current_path(graph_conn, &stitched_block_group_id, None)
                .unwrap()
                .sequence(graph_conn, None)
                .unwrap();
        assert_eq!(stitched_sequence, "ATCGATCGATCGATCGATCGGGAACACACAGAGA");
    }
}
