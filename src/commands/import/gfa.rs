use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use gen_models::{
    errors::OperationError,
    sample::{NewSample, Sample},
};

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    imports::gfa::{GFAImportError, import_gfa},
};

/// Import a GFA file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to store the entry under
    #[arg(short, long)]
    name: Option<String>,
    /// A sample name to associate the GFA file with
    #[arg(short, long, conflicts_with = "reference")]
    sample: Option<String>,
    /// Create or use a reference sample with this name
    #[arg(long, conflicts_with = "sample")]
    reference: Option<String>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("GFA import called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
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
    match import_gfa(context, &PathBuf::from(cmd.path.clone()), name, sample_name) {
        Ok(_) => {
            println!("GFA imported.");
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok(())
        }
        Err(GFAImportError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            println!("GFA already exists.");
            Ok(())
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            println!("Import failed.");
            Err(e.into())
        }
    }
}
