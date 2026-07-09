use anyhow::Result;
use clap::Args;
use gen_models::{
    errors::OperationError,
    sample::{NewSample, Sample},
};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    graphs::combinatorial_library::parse_library,
    imports::library::import_library,
};

/// Import Library files
#[derive(Debug, Args, Clone)]
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
    #[arg(short, long, conflicts_with = "reference")]
    sample: Option<String>,
    /// Create or use a reference sample with this name
    #[arg(long, conflicts_with = "sample")]
    reference: Option<String>,
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(cli_context, cmd))
)]
pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Library import called");

    let context = cli_context.context;
    let operation_conn = context.config().conn();
    let conn = context.graph().conn();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));
    let (sample_name, is_reference) = crate::commands::import::resolve_import_sample(
        cmd.sample.as_deref(),
        cmd.reference.as_deref(),
    )?;
    let parts_list = parse_library(&cmd.parts.clone().unwrap(), &cmd.library.clone().unwrap())?;

    let parts_path = cmd.parts.unwrap();
    let library_path = cmd.library.unwrap();

    conn.execute("BEGIN TRANSACTION", [])?;

    if is_reference
        && let Err(e) = Sample::get_or_create(
            conn,
            NewSample {
                name: sample_name,
                is_reference: true,
            },
        )
    {
        conn.execute("ROLLBACK TRANSACTION;", [])?;
        return Err(e.into());
    }

    match import_library(
        context,
        name,
        sample_name,
        &cmd.library_name,
        parts_list,
        Some(&parts_path),
        Some(&library_path),
    ) {
        Ok(operation_summary) => match commit_operation(context, &operation_summary) {
            Ok(_) => {
                println!("Imported library file {library_path} and parts file {parts_path}");
                Ok(())
            }
            Err(OperationError::NoChanges) => {
                println!("Library already exists.");
                Ok(())
            }
            Err(e) => Err(e.into()),
        },
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            println!("Library import failed: {}", e);
            Err(e.into())
        }
    }
}
