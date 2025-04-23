use crate::commands::cli_context::CliContext;
use crate::commands::get_db_for_command;
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::imports::genbank::import_genbank;
use crate::models::file_types::FileTypes;
use crate::models::metadata;
use crate::models::operations::{setup_db, OperationFile, OperationInfo};
use clap::Args;
use rusqlite::Connection;
use std::fs::File;

/// Import a Genbank file
#[derive(Debug, Args)]
pub struct Command {
    /// Genbank file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to store the entry under
    #[arg(short, long)]
    name: Option<String>,
    /// A sample name to associate the Genbank file with
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
    println!("Genbank import called");

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
    let mut reader: Box<dyn std::io::Read> = if cmd.path.ends_with(".gz") {
        let file = File::open(cmd.path.clone()).unwrap();
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(File::open(cmd.path.clone()).unwrap())
    };
    match import_genbank(
        &conn,
        &operation_conn,
        &mut reader,
        name.as_ref(),
        cmd.sample.as_deref(),
        OperationInfo {
            files: vec![OperationFile {
                file_path: cmd.path.clone(),
                file_type: FileTypes::GenBank,
            }],
            description: "GenBank Import".to_string(),
        },
    ) {
        Ok(_) => {
            println!("GenBank imported.");
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            panic!("Import failed: {err:?}");
        }
    }
}
