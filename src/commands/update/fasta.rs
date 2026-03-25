use anyhow::Result;
use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    updates::fasta::update_with_fasta,
};

/// Update with a fasta file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Fasta file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long)]
    sample: String,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
    /// The name of the region to update (eg "chr1")
    #[arg(long)]
    region_name: String,
    /// The start coordinate for the region to add the library to
    #[arg(long)]
    start: i64,
    /// The end coordinate for the region to add the library to
    #[arg(short, long)]
    end: i64,
    /// Do not update the sample's reference path if there is a single fasta entry
    #[arg(long, action)]
    no_reference_path_update: bool,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with fasta called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    if let Err(err) = update_with_fasta(
        context,
        name,
        cmd.sample.as_str(),
        &cmd.new_sample,
        &cmd.region_name,
        cmd.start,
        cmd.end,
        &cmd.path,
        cmd.no_reference_path_update,
    ) {
        conn.execute("ROLLBACK TRANSACTION;", [])?;
        operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(err.into());
    }

    conn.execute("END TRANSACTION;", [])?;
    operation_conn.execute("END TRANSACTION;", [])?;

    Ok(())
}
