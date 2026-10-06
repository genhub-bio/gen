use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use gen_models::errors::OperationError;

use crate::{
    commands::{cli_context::CliContext, commit_operation},
    imports::sample_metadata::import_sample_metadata,
};

/// Import sample metadata from an exported TSV file.
#[derive(Clone, Debug, Args)]
pub struct Command {
    /// Input TSV file path
    pub path: PathBuf,
    /// Override the Dolt commit message
    #[arg(short = 'm', long)]
    message: Option<String>,
}

pub fn execute(cli_context: &CliContext, command: Command) -> Result<()> {
    let context = cli_context.context;
    let connection = context.graph().conn();
    connection.execute("BEGIN TRANSACTION", [])?;
    match import_sample_metadata(connection, &command.path) {
        Ok(mut summary) => {
            connection.execute("END TRANSACTION", [])?;
            if let Some(message) = command.message {
                summary.summary = message;
            }
            match commit_operation(context, &summary) {
                Ok(_) | Err(OperationError::NoChanges) => Ok(()),
                Err(error) => Err(error.into()),
            }
        }
        Err(error) => {
            connection.execute("ROLLBACK TRANSACTION", [])?;
            Err(error)
        }
    }
}
