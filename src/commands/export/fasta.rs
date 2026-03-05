use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    exports::fasta::export_fasta,
};

/// Export a FASTA file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// FASTA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection for exporting
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample for exporting
    #[arg(short, long)]
    sample: Option<String>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("FASTA export called");
    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));
    export_fasta(
        conn,
        name,
        cmd.sample.clone().as_deref(),
        &PathBuf::from(cmd.path),
    )?;

    conn.execute("END TRANSACTION", [])?;
    operation_conn.execute("END TRANSACTION", [])?;

    Ok(())
}
