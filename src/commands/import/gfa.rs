use crate::commands::cli_context::CliContext;
use crate::config::{get_gen_dir, get_operation_connection};
use crate::get_connection;
use crate::imports::gfa::{import_gfa, GFAImportError};
use crate::models::metadata;
use crate::models::operations::setup_db;
use crate::operation_management::OperationError;
use clap::Args;
use rusqlite::Connection;
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

fn get_default_collection(conn: &Connection) -> String {
    let mut stmt = conn
        .prepare("select collection_name from defaults where id = 1")
        .unwrap();
    stmt.query_row((), |row| row.get(0))
        .unwrap_or("default".to_string())
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("GFA import called");

    let operation_conn = get_operation_connection(None);

    let binding = cli_context.db.clone().unwrap_or_else(|| {
        let mut stmt = operation_conn
            .prepare("select db_name from defaults where id = 1;")
            .unwrap();
        let row: Option<String> = stmt.query_row((), |row| row.get(0)).unwrap();
        row.unwrap_or_else(|| match get_gen_dir() {
            Some(dir) => PathBuf::from(dir)
                .join("default.db")
                .to_str()
                .unwrap()
                .to_string(),
            None => {
                panic!("No .gen directory found. Please run 'gen init' first.")
            }
        })
    });
    let db = binding.as_str();
    let conn = get_connection(db);
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
        Ok(_) => println!("GFA imported."),
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
