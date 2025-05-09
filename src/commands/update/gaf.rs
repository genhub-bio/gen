use crate::commands::cli_context::CliContext;
use crate::commands::{get_db_for_command, get_default_collection};
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::models::metadata;
use crate::models::operations::setup_db;
use crate::updates::gaf::update_with_gaf;
use clap::Args;

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

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with GAF called");

    let operation_conn = get_operation_connection(None);
    let db = get_db_for_command(cli_context, &operation_conn);
    let conn = get_connection(&db);
    let db_uuid = metadata::get_db_uuid(&conn);

    // initialize the selected database if needed.
    setup_db(&operation_conn, &db_uuid);
    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));

    update_with_gaf(
        &conn,
        &operation_conn,
        &cmd.path,
        &cmd.csv,
        name,
        cmd.sample.as_deref(),
        cmd.parent_sample.as_deref(),
    );

    conn.execute("END TRANSACTION;", []).unwrap();
    operation_conn.execute("END TRANSACTION;", []).unwrap();
}
