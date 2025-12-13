use anyhow::Result;
use clap::Args;
use gen_models::errors::OperationError;

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    get_connection, get_operation_connection,
    imports::library::{LibraryImportError, import_library},
};

/// Import Library files
#[derive(Debug, Args)]
pub struct Command {
    /// The name of the library
    #[clap(index = 1)]
    library_name: String,
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

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Library import called");

    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(cli_context.db.clone(), &operation_conn);
    let conn = get_connection(&db).unwrap();

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));
    match import_library(
        &conn,
        &operation_conn,
        name,
        cmd.sample.as_deref(),
        cmd.parts.as_deref().unwrap(),
        cmd.library.as_deref().unwrap(),
        &cmd.library_name,
    ) {
        Ok(_) => {
            println!("Library imported.");
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok(())
        }
        Err(LibraryImportError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            println!("Library already exists.");
            Ok(())
        }
        Err(e) => {
            println!("Library import failed: {}", e);
            Err(e.into())
        }
    }
}
