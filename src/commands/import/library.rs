use crate::commands::get_db_for_command;
use crate::commands::cli_context::CliContext;
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::imports::library::import_library;
use crate::models::metadata;
use crate::models::operations::setup_db;
use clap::Args;
use rusqlite::Connection;

/// Import Library files
#[derive(Debug, Args)]
pub struct Command {
    /// The name of the region
    #[clap(index = 1)]
    region_name: String,
    /// The path to the combinatorial library parts fasta file
    #[clap(index = 2)]
    parts: Option<String>,
    /// The path to the combinatorial library csv file
    #[clap(index = 3)]
    library: Option<String>,
    /// The name of the collection to store the entry under
    #[arg(short, long)]
    name: Option<String>,
    /// A sample name to associate the library with
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
    println!("Library import called");

    let operation_conn = get_operation_connection(None);
    let db = get_db_for_command(cli_context, &operation_conn);
    let conn = get_connection(&db);
    let db_uuid = metadata::get_db_uuid(&conn);

    // initialize the selected database if needed.
    setup_db(&operation_conn, &db_uuid);
    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd.name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));
    import_library(
        &conn,
        &operation_conn,
        name,
        cmd.sample.as_deref(),
        cmd.parts.as_deref().unwrap(),
        cmd.library.as_deref().unwrap(),
        &cmd.region_name,
    )
        .unwrap();
}
