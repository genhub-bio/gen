use anyhow::Result;
use clap::Args;
use gen_models::{errors::OperationError, sample::Sample};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    updates::gfa::update_with_gfa,
};

/// Update with a GFA file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long, default_value_t = Sample::DEFAULT_NAME.to_string())]
    sample: String,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(cli_context, cmd))
)]
pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with GFA called");

    let context = cli_context.context;
    let config_conn = context.config().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(config_conn));

    match update_with_gfa(
        context,
        name,
        cmd.sample.as_str(),
        &cmd.new_sample,
        &cmd.path,
    ) {
        Ok(operation_summary) => {
            conn.execute("END TRANSACTION", [])?;
            match commit_operation(context, &operation_summary) {
                Ok(_) | Err(OperationError::NoChanges) => {}
                Err(e) => return Err(e.into()),
            }
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(e.into());
        }
    }

    Ok(())
}
