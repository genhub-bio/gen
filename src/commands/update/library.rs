use crate::commands::cli_context::CliContext;
use crate::commands::get_db_for_command;
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::models::metadata;
use crate::models::operations::setup_db;
use crate::updates::library::update_with_library;
use clap::Args;
use rusqlite::Connection;

/// Update with library files
#[derive(Debug, Args)]
pub struct Command {
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long)]
    sample: Option<String>,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
    /// The name of the path to add the library to
    #[arg(short, long)]
    path_name: String,
    /// The start coordinate for the region to add the library to
    #[arg(long)]
    start: i64,
    /// The end coordinate for the region to add the library to
    #[arg(short, long)]
    end: i64,
    /// A CSV with combinatorial library information
    #[arg(short, long)]
    library: String,
    /// A fasta with the combinatorial library parts
    #[arg(long)]
    parts: String,
}

fn get_default_collection(conn: &Connection) -> String {
    let mut stmt = conn
        .prepare("select collection_name from defaults where id = 1")
        .unwrap();
    stmt.query_row((), |row| row.get(0))
        .unwrap_or("default".to_string())
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with library called");

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

    update_with_library(
        &conn,
        &operation_conn,
        name,
        cmd.sample.clone().as_deref(),
        &cmd.new_sample,
        &cmd.path_name,
        cmd.start,
        cmd.end,
        &cmd.parts,
        &cmd.library,
    )
    .unwrap();

    conn.execute("END TRANSACTION;", []).unwrap();
    operation_conn.execute("END TRANSACTION;", []).unwrap();
}
