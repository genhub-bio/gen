use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    exports::gfa::export_gfa,
    get_connection, get_operation_connection,
};

/// Export a GFA file
#[derive(Debug, Args)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection for exporting
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample for exporting
    #[arg(short, long)]
    sample: Option<String>,
    /// The max sequence length per node
    #[arg(long)]
    node_max: Option<i64>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("GFA export called");
    let operation_conn = get_operation_connection(None)?;
    let db = get_db_for_command(cli_context.db.clone(), &operation_conn);
    let conn = get_connection(&db)?;

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));
    export_gfa(
        &conn,
        name,
        &PathBuf::from(cmd.path),
        cmd.sample.clone(),
        cmd.node_max,
    );

    conn.execute("END TRANSACTION", [])?;
    operation_conn.execute("END TRANSACTION", [])?;

    Ok(())
}
