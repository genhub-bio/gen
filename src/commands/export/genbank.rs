use std::path::PathBuf;

use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    exports::genbank::export_genbank,
    get_connection, get_operation_connection,
};

/// Export a GenBank file
#[derive(Debug, Args)]
pub struct Command {
    /// GenBank file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection for exporting
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample for exporting
    #[arg(short, long)]
    sample: Option<String>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("GFA export called");
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
    export_genbank(
        &conn,
        name,
        cmd.sample.clone().as_deref(),
        &PathBuf::from(cmd.path),
    );

    conn.execute("END TRANSACTION", []).unwrap();
    operation_conn.execute("END TRANSACTION", []).unwrap();
}
