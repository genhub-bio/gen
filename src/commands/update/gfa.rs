use anyhow::Result;
use clap::Args;
use gen_models::sample::Sample;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
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
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    match update_with_gfa(
        context,
        name,
        cmd.sample.as_str(),
        &cmd.new_sample,
        &cmd.path,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", [])?;
            operation_conn.execute("END TRANSACTION;", [])?;
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(e.into());
        }
    }

    Ok(())
}
