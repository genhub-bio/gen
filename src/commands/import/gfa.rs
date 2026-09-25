use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use gen_models::{
    errors::OperationError,
    sample::{NewSample, Sample},
};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    imports::gfa::import_gfa,
};

/// Import a GFA file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// Override the Dolt commit message
    #[arg(short = 'm', long)]
    message: Option<String>,
    /// The name of the collection to store the entry under
    #[arg(short = 'c', long)]
    collection: Option<String>,
    /// A sample name to associate the GFA file with
    #[arg(short, long, conflicts_with = "reference")]
    sample: Option<String>,
    /// Create or use a reference sample with this name
    #[arg(long, conflicts_with = "sample")]
    reference: Option<String>,
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(cli_context, cmd))
)]
pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("GFA import called");

    let context = cli_context.context;
    let config_conn = context.config().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", []).unwrap();

    let collection_name = &cmd
        .collection
        .clone()
        .unwrap_or_else(|| get_default_collection(config_conn));
    let (sample_name, is_reference) = crate::commands::import::resolve_import_sample(
        cmd.sample.as_deref(),
        cmd.reference.as_deref(),
    )?;
    if is_reference
        && let Err(e) = Sample::get_or_create(
            conn,
            NewSample {
                name: sample_name,
                is_reference: true,
            },
        )
    {
        conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(e.into());
    }
    match import_gfa(
        context,
        &PathBuf::from(cmd.path.clone()),
        collection_name,
        sample_name,
    ) {
        Ok(operation_summary) => {
            conn.execute("END TRANSACTION", [])?;
            let mut operation_summary = operation_summary;
            if let Some(message) = cmd.message {
                operation_summary.summary = message;
            }
            match commit_operation(context, &operation_summary) {
                Ok(_) => {
                    println!("GFA imported.");
                    Ok(())
                }
                Err(OperationError::NoChanges) => {
                    println!("GFA already exists.");
                    Ok(())
                }
                Err(e) => Err(e.into()),
            }
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            println!("Import failed.");
            Err(e.into())
        }
    }
}
