use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    exports::gfa::export_gfa,
};

/// Export a GFA file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection for exporting
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample for exporting
    #[arg(short, long)]
    sample: String,
    /// The max sequence length per node
    #[arg(long)]
    node_max: Option<i64>,
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
    export_gfa(
        conn,
        name,
        &PathBuf::from(cmd.path),
        cmd.sample.as_str(),
        cmd.node_max,
        cli_context.history_ref,
    )?;

    conn.execute("END TRANSACTION", [])?;

    Ok(())
}
