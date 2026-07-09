use std::{fs::File, path::PathBuf};

use anyhow::Result;
use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    exports::genbank::export_genbank,
};

/// Export a GenBank file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GenBank file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection for exporting
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample for exporting
    #[arg(short, long)]
    sample: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("GFA export called");
    let context = cli_context.context;
    let operation_conn = context.config().conn();
    let conn = context.graph().conn();

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));
    let file = File::create(PathBuf::from(cmd.path))?;
    export_genbank(
        conn,
        name,
        cmd.sample.as_str(),
        file,
        cli_context.history_ref,
    )?;

    conn.execute("END TRANSACTION", [])?;

    Ok(())
}
