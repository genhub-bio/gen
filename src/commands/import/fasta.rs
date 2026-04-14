use anyhow::Result;
use clap::Args;
use gen_models::errors::OperationError;

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    fasta::FastaError,
    imports::fasta::import_fasta,
};

/// Import a fasta file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Fasta file path
    #[clap(index = 1)]
    pub path: String,
    /// Don't store the sequence in the database, instead store the filename
    #[arg(long, action)]
    shallow: bool,
    /// The name of the collection to store the entry under
    #[arg(short, long)]
    name: Option<String>,
    /// A sample name to associate the fasta file with
    #[arg(short, long, default_value = "reference")]
    sample: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Fasta import called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));
    match import_fasta(
        context,
        &cmd.path.clone(),
        name,
        cmd.sample.as_str(),
        cmd.shallow,
    ) {
        Ok(_) => {
            println!("Fasta imported.");
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok(())
        }
        Err(FastaError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            println!("Fasta contents already exist.");
            Ok(())
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            Err(e.into())
        }
    }
}
