use anyhow::Result;
use clap::Args;
use gen_models::errors::OperationError;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    imports::alignment::{AlignmentImportError, import_alignment_aln},
};

/// Import an alignment file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Alignment file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to store the alignment under
    #[arg(short, long)]
    name: Option<String>,
    /// Override the sample name for all imported alignment rows
    #[arg(short, long)]
    sample: Option<String>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Alignment import called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    match import_alignment_aln(context, &cmd.path.clone(), name, cmd.sample.as_deref()) {
        Ok(_) => {
            println!("Alignment imported.");
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok(())
        }
        Err(AlignmentImportError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            println!("Alignment contents already exist.");
            Ok(())
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(e.into())
        }
    }
}
