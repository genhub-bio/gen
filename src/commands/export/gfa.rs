use crate::commands::cli_context::CliContext;
use crate::commands::{get_db_for_command, get_default_collection};
use crate::config::get_operation_connection;
use crate::exports::gfa::export_gfa;
use crate::get_connection;
use crate::models::metadata;
use crate::models::operations::setup_db;
use clap::Args;
use std::path::PathBuf;

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

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("GFA export called");
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
    export_gfa(
        &conn,
        name,
        &PathBuf::from(cmd.path),
        cmd.sample.clone(),
        cmd.node_max,
    );

    conn.execute("END TRANSACTION", []).unwrap();
    operation_conn.execute("END TRANSACTION", []).unwrap();
}
