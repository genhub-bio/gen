use anyhow::Result;
use clap::Args;
use gen_models::errors::OperationError;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    updates::vcf::{VcfError, update_with_vcf},
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

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with VCF called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    match update_with_vcf(
        &cmd.path,
        name,
        cmd.genotype.clone().unwrap_or("".to_string()),
        cmd.sample.clone().unwrap_or("".to_string()),
        context,
        cmd.coordinate_frame.as_deref(),
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", [])?;
            operation_conn.execute("END TRANSACTION;", [])?;
        }
        Err(VcfError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            println!(
                "No changes made. If the VCF lacks a sample or genotype, they need to be provided via --sample and --genotype."
            );
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(e.into());
        }
    }

    Ok(())
}
