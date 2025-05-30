use crate::commands::cli_context::CliContext;
use crate::commands::{get_db_for_command, get_default_collection};
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::models::metadata;
use crate::models::operations::setup_db;
use crate::updates::fasta::update_with_fasta;
use clap::Args;

/// Update with a fasta file
#[derive(Debug, Args)]
pub struct Command {
    /// Fasta file path
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
    /// The name of the region to update (eg "chr1")
    #[arg(long)]
    region_name: String,
    /// The start coordinate for the region to add the library to
    #[arg(long)]
    start: i64,
    /// The end coordinate for the region to add the library to
    #[arg(short, long)]
    end: i64,
    /// Do not update the sample's reference path if there is a single fasta entry
    #[arg(long, action)]
    no_reference_path_update: bool,
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with fasta called");

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

    update_with_fasta(
        &conn,
        &operation_conn,
        name,
        cmd.sample.clone().as_deref(),
        &cmd.new_sample,
        &cmd.region_name,
        cmd.start,
        cmd.end,
        &cmd.path,
        cmd.no_reference_path_update,
    )
    .unwrap();

    conn.execute("END TRANSACTION;", []).unwrap();
    operation_conn.execute("END TRANSACTION;", []).unwrap();
}
