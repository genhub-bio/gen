use crate::commands::cli_context::CliContext;
use crate::commands::get_db_for_command;
use crate::config::get_operation_connection;
use crate::exports::fasta::export_fasta;
use crate::get_connection;
use crate::models::metadata;
use crate::models::operations::setup_db;
use clap::Args;
use rusqlite::Connection;
use std::path::PathBuf;

/// Export a FASTA file
#[derive(Debug, Args)]
pub struct Command {
    /// FASTA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection for exporting
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample for exporting
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
    export_fasta(
        &conn,
        name,
        cmd.sample.clone().as_deref(),
        &PathBuf::from(cmd.path),
    );

    conn.execute("END TRANSACTION", []).unwrap();
    operation_conn.execute("END TRANSACTION", []).unwrap();
}
