use clap::Args;
use gen_models::{errors::OperationError, operations::setup_db};

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    get_connection, get_operation_connection,
    updates::vcf::{update_with_vcf, VcfError},
};

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

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with VCF called");

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
