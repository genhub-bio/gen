use crate::commands::cli_context::CliContext;
use crate::commands::get_db_for_command;
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::models::file_types::FileTypes;
use crate::models::metadata;
use crate::models::operations::{setup_db, OperationFile, OperationInfo};
use crate::updates::genbank::update_with_genbank;
use clap::Args;
use rusqlite::Connection;
use std::fs::File;

/// Update with a GenBank file
#[derive(Debug, Args)]
pub struct Command {
    /// GenBank file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// If a new entity is found, create it as a normal import
    #[arg(long, action, alias = "cm")]
    create_missing: bool,
}

fn get_default_collection(conn: &Connection) -> String {
    let mut stmt = conn
        .prepare("select collection_name from defaults where id = 1")
        .unwrap();
    stmt.query_row((), |row| row.get(0))
        .unwrap_or("default".to_string())
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with GenBank called");

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

    let f = File::open(&cmd.path).unwrap();
    match update_with_genbank(
        &conn,
        &operation_conn,
        &f,
        name.as_ref(),
        cmd.create_missing,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: cmd.path.clone(),
                file_type: FileTypes::GenBank,
            }],
            description: "Update from GenBank".to_string(),
        },
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
