use anyhow::Result;
use clap::Args;
use gen_models::sample::Sample;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
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
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    if let Err(err) = update_with_gaf(
        context,
        &cmd.path,
        &cmd.csv,
        name,
        cmd.sample.as_str(),
        Some(cmd.parent_sample.as_str()),
    ) {
        conn.execute("ROLLBACK TRANSACTION;", [])?;
        operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(err.into());
    }

    conn.execute("END TRANSACTION;", [])?;
    operation_conn.execute("END TRANSACTION;", [])?;

    Ok(())
}
