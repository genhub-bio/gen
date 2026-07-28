use anyhow::Result;
use clap::Args;
use gen_models::{errors::OperationError, sample::Sample};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    graphs::combinatorial_library::parse_library,
    updates::library::update_with_library,
};

/// Update with library files
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long, default_value_t = Sample::DEFAULT_NAME.to_string())]
    sample: String,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
    /// The region to replace with the library (path, accession, or annotation)
    #[arg(short, long = "region-name", alias = "path-name")]
    region_name: String,
    /// A CSV with combinatorial library information
    #[arg(short, long)]
    library: String,
    /// A fasta with the combinatorial library parts
    #[arg(long)]
    parts: String,
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(cli_context, cmd))
)]
pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with library called");

    let context = cli_context.context;
    let config_conn = context.config().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(config_conn));

    let parts_list = parse_library(&cmd.parts, &cmd.library)?;

    match update_with_library(
        context,
        name,
        cmd.sample.as_str(),
        &cmd.new_sample,
        &cmd.region_name,
        parts_list,
        Some(&cmd.parts),
        Some(&cmd.library),
    ) {
        Ok(operation_summary) => {
            conn.execute("END TRANSACTION", [])?;
            match commit_operation(context, &operation_summary) {
                Ok(_) | Err(OperationError::NoChanges) => {}
                Err(err) => return Err(err.into()),
            }
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(err.into());
        }
    };

    println!("Updated with library file: {0}", cmd.library);

    Ok(())
}
