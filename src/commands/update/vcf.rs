use crate::commands::cli_context::CliContext;
use crate::commands::get_db_for_command;
use crate::config::get_operation_connection;
use crate::get_connection;
use crate::models::metadata;
use crate::models::operations::setup_db;
use crate::operation_management::OperationError;
use crate::updates::vcf::{update_with_vcf, VcfError};
use clap::Args;
use rusqlite::Connection;

/// Update with a VCF file
#[derive(Debug, Args)]
pub struct Command {
    /// VCF file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// If no genotype is provided, enter the genotype to assign variants
    #[arg(short, long)]
    genotype: Option<String>,
    /// The name of the sample to update
    #[arg(short, long)]
    sample: Option<String>,
    /// Use the given sample as the parent sample for changes.
    #[arg(long, alias = "cf")]
    coordinate_frame: Option<String>,
}

fn get_default_collection(conn: &Connection) -> String {
    let mut stmt = conn
        .prepare("select collection_name from defaults where id = 1")
        .unwrap();
    stmt.query_row((), |row| row.get(0))
        .unwrap_or("default".to_string())
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with VCF called");

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

    match update_with_vcf(
        &cmd.path,
        name,
        cmd.genotype.clone().unwrap_or("".to_string()),
        cmd.sample.clone().unwrap_or("".to_string()),
        &conn,
        &operation_conn,
        cmd.coordinate_frame.as_deref(),
    ) {
        Ok(_) => {
	    conn.execute("END TRANSACTION;", []).unwrap();
	    operation_conn.execute("END TRANSACTION;", []).unwrap();
	},
        Err(VcfError::OperationError(OperationError::NoChanges)) => println!("No changes made. If the VCF lacks a sample or genotype, they need to be provided via --sample and --genotype."),
        Err(e) => panic!("Error updating with vcf: {e}"),
    }
}
