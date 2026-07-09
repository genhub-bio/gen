use anyhow::Result;
use clap::Args;
use gen_models::{errors::OperationError, sample::Sample};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    updates::gaf::update_with_gaf,
};

/// Update with a GAF file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GAF file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long, default_value_t = Sample::DEFAULT_NAME.to_string())]
    sample: String,
    /// The csv describing changes to make
    #[arg(short, long)]
    csv: String,
    /// If specified, the newly created sample will inherit this sample's existing graph
    #[arg(short, long, default_value_t = Sample::DEFAULT_NAME.to_string())]
    parent_sample: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with GAF called");

    let context = cli_context.context;
    let operation_conn = context.config().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    match update_with_gaf(
        context,
        &cmd.path,
        &cmd.csv,
        name,
        cmd.sample.as_str(),
        Some(cmd.parent_sample.as_str()),
    ) {
        Ok(operation_summary) => match commit_operation(context, &operation_summary) {
            Ok(_) | Err(OperationError::NoChanges) => {}
            Err(err) => return Err(err.into()),
        },
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(err.into());
        }
    };

    Ok(())
}
