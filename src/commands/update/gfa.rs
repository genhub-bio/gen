use crate::commands::cli_context::CliContext;
use crate::commands::{get_db_for_command, get_default_collection};
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::models::metadata;
use crate::models::operations::setup_db;
use crate::updates::gfa::update_with_gfa;
use clap::Args;

/// Update with a GFA file
#[derive(Debug, Args)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long)]
    sample: Option<String>,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with GFA called");

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

    match update_with_gfa(
        &conn,
        &operation_conn,
        name,
        cmd.sample.clone().as_deref(),
        &cmd.new_sample,
        &cmd.path,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            panic!("Failed to update. Error is: {e}");
        }
    }
}
