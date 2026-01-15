use anyhow::Result;
use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    updates::library::update_with_library,
};

/// Update with library files
#[derive(Debug, Args)]
pub struct Command {
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long)]
    sample: Option<String>,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
    /// The name of the path to add the library to
    #[arg(short, long)]
    path_name: String,
    /// The start coordinate for the region to add the library to
    #[arg(long)]
    start: i64,
    /// The end coordinate for the region to add the library to
    #[arg(short, long)]
    end: i64,
    /// A CSV with combinatorial library information
    #[arg(short, long)]
    library: String,
    /// A fasta with the combinatorial library parts
    #[arg(long)]
    parts: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with library called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    if let Err(err) = update_with_library(
        context,
        name,
        cmd.sample.clone().as_deref(),
        &cmd.new_sample,
        &cmd.path_name,
        cmd.start,
        cmd.end,
        &cmd.parts,
        &cmd.library,
    ) {
        conn.execute("ROLLBACK TRANSACTION;", [])?;
        operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(err.into());
    }

    conn.execute("END TRANSACTION;", [])?;
    operation_conn.execute("END TRANSACTION;", [])?;

    Ok(())
}
