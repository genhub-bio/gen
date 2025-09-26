use clap::Args;
use gen_models::operations::setup_db;

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    get_connection, get_operation_connection,
    updates::fasta::update_with_fasta,
};

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

    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(cli_context.db.clone(), &operation_conn);
    let conn = get_connection(&db).unwrap();

    // initialize the selected database if needed.
    setup_db(&operation_conn);
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
