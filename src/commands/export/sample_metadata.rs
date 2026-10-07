use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::{commands::cli_context::CliContext, exports::sample_metadata::export_sample_metadata};

/// Export sample metadata as TSV or JSON.
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Output file path
    pub path: PathBuf,
    /// Export only this sample's metadata; omit to export all samples
    #[arg(short, long)]
    pub sample: Option<String>,
    /// Export only these comma-separated metadata keys; omit to export all keys
    #[arg(long, value_delimiter = ',')]
    pub keys: Option<Vec<String>>,
    /// Export JSON grouped by sample instead of TSV
    #[arg(long)]
    pub json: bool,
}

pub fn execute(cli_context: &CliContext, command: Command) -> Result<()> {
    export_sample_metadata(
        cli_context.context.graph().conn(),
        command.sample.as_deref(),
        &command.path,
        cli_context.history_ref,
        command.keys.as_deref(),
        command.json,
    )
}
