use anyhow::Result;
use clap::Args;
use gen_models::errors::OperationError;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    updates::vcf::{VcfError, update_with_vcf},
};

/// Update with a VCF file
#[derive(Debug, Args, Clone)]
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
    /// Optional sample override. If omitted, use samples defined in the VCF.
    #[arg(short, long)]
    sample: Option<String>,
    /// Use the given samples as the parent samples for changes. Repeat the flag or use commas.
    #[arg(
        long = "parent-samples",
        aliases = ["parent-sample", "ps", "reference"],
        value_delimiter = ','
    )]
    parent_samples: Vec<String>,
    /// Apply edits in-place instead of using parent sample's reference coordinates
    #[arg(long = "inplace")]
    in_place: bool,
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(cli_context, cmd))
)]
pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with VCF called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    if cmd.sample.is_none() && cmd.parent_samples.is_empty() {
        return Err(anyhow::anyhow!(
            "one of --sample or --reference must be provided"
        ));
    }

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    match update_with_vcf(
        context,
        &cmd.path,
        name,
        cmd.genotype.clone().unwrap_or("".to_string()),
        cmd.sample.as_deref(),
        cmd.parent_samples.clone(),
        cmd.in_place,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", [])?;
            operation_conn.execute("END TRANSACTION;", [])?;
        }
        Err(VcfError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            println!(
                "No changes made. If the VCF lacks samples or genotypes, provide them via --sample and --genotype."
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
