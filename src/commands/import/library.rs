use anyhow::Result;
use clap::Args;
use gen_models::{
    errors::OperationError,
    sample::{NewSample, Sample},
};

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    end_transaction_if_active,
    graphs::combinatorial_library::parse_library,
    imports::library::{LibraryImportError, import_library},
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

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));
    let (sample_name, is_reference) = crate::commands::import::resolve_import_sample(
        cmd.sample.as_deref(),
        cmd.reference.as_deref(),
    )?;
    if is_reference {
        Sample::get_or_create(
            conn,
            NewSample {
                name: sample_name,
                is_reference: true,
            },
        )?;
    }
    let parts_list = parse_library(&cmd.parts.clone().unwrap(), &cmd.library.clone().unwrap())?;

    let parts_path = cmd.parts.unwrap();
    let library_path = cmd.library.unwrap();

    match import_library(
        context,
        name,
        sample_name,
        &cmd.library_name,
        parts_list,
        Some(&parts_path),
        Some(&library_path),
    ) {
        Ok(_) => {
            println!("Imported library file {library_path} and parts file {parts_path}");
            end_transaction_if_active(conn).unwrap();
            end_transaction_if_active(operation_conn).unwrap();
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
