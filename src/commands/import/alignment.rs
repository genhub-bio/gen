use anyhow::Result;
use clap::Args;
use gen_models::{
    errors::OperationError,
    sample::{NewSample, Sample},
};

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
    /// A sample name to associate the alignment with
    #[arg(short, long, conflicts_with = "reference")]
    sample: Option<String>,
    /// Create or use a reference sample with this name
    #[arg(long, conflicts_with = "sample")]
    reference: Option<String>,
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
    let default_sample = Sample::DEFAULT_NAME.to_string();
    let (sample_name, is_reference) = match (cmd.sample.as_deref(), cmd.reference.as_deref()) {
        (Some(sample), None) => (sample, false),
        (None, Some(reference)) => (reference, true),
        (None, None) => (default_sample.as_str(), false),
        (Some(_), Some(_)) => {
            return Err(anyhow::anyhow!(
                "--sample and --reference are mutually exclusive"
            ));
        }
    };
    if is_reference {
        Sample::get_or_create(
            conn,
            NewSample {
                name: sample_name,
                is_reference: true,
            },
        )?;
    }

    match import_alignment_aln(context, &cmd.path.clone(), name, sample_name) {
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
