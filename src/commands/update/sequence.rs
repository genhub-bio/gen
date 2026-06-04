use anyhow::Result;
use clap::Args;
use gen_models::sample::Sample;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
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
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    if let Err(err) = update_with_sequence(
        context,
        name,
        cmd.sample.as_str(),
        &cmd.new_sample,
        &cmd.region_name,
        &cmd.sequence,
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
