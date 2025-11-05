use anyhow::Result;
use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    get_connection, get_operation_connection,
    updates::gaf::update_with_gaf,
};

/// Update with a GAF file
#[derive(Debug, Args)]
pub struct Command {
    /// GAF file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long)]
    sample: Option<String>,
    /// The csv describing changes to make
    #[arg(short, long)]
    csv: String,
    /// If specified, the newly created sample will inherit this sample's existing graph
    #[arg(short, long)]
    parent_sample: Option<String>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with GAF called");

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

    if let Err(err) = update_with_gaf(
        &conn,
        &operation_conn,
        &cmd.path,
        &cmd.csv,
        name,
        cmd.sample.as_deref(),
        cmd.parent_sample.as_deref(),
    ) {
        conn.execute("ROLLBACK TRANSACTION;", [])?;
        operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(err.into());
    }

    conn.execute("END TRANSACTION;", [])?;
    operation_conn.execute("END TRANSACTION;", [])?;

    Ok(())
}
