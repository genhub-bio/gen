use anyhow::Result;
use clap::Args;
use gen_models::{errors::OperationError, sample::Sample};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    updates::sequence::update_with_sequence,
};

/// Update with a fasta file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Sequence to use
    #[clap(index = 1)]
    pub sequence: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long, default_value_t = Sample::DEFAULT_NAME.to_string())]
    sample: String,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
    /// The region to update (path, accession, or annotation; e.g. "chr1:2-5")
    #[arg(long)]
    region_name: String,
    /// Do not update the sample's reference path
    #[arg(long, action)]
    no_reference_path_update: bool,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with sequence called");

    let context = cli_context.context;
    let config_conn = context.config().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(config_conn));

    match update_with_sequence(
        context,
        name,
        cmd.sample.as_str(),
        &cmd.new_sample,
        &cmd.region_name,
        &cmd.sequence,
        cmd.no_reference_path_update,
    ) {
        Ok(operation_summary) => {
            conn.execute("END TRANSACTION", [])?;
            match commit_operation(context, &operation_summary) {
                Ok(_) | Err(OperationError::NoChanges) => {}
                Err(err) => return Err(err.into()),
            }
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(err.into());
        }
    };

    Ok(())
}
