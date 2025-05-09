use crate::commands::cli_context::CliContext;
use crate::commands::{get_db_for_command, get_default_collection};
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::imports::gfa::{import_gfa, GFAImportError};
use crate::models::metadata;
use crate::models::operations::setup_db;
use crate::operation_management::OperationError;
use clap::Args;
use std::path::PathBuf;

/// Import a GFA file
#[derive(Debug, Args)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to store the entry under
    #[arg(short, long)]
    name: Option<String>,
    /// A sample name to associate the GFA file with
    #[arg(short, long)]
    sample: Option<String>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("GFA import called");

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
    match import_gfa(
        &PathBuf::from(cmd.path.clone()),
        name,
        cmd.sample.as_deref(),
        &conn,
        &operation_conn,
    ) {
        Ok(_) => {
            println!("GFA imported.");
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
        }
        Err(GFAImportError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            println!("GFA already exists.")
        }
        Err(_) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            panic!("Import failed.");
        }
    }
}
