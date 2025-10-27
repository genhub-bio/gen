use anyhow;
use clap::{Args, Subcommand};
use clap_nested_commands::generate_sync_commands;

use crate::commands::cli_context::CliContext;

mod fasta;
mod genbank;
mod gfa;
mod library;

/// Import commands
#[derive(Debug, Args)]
pub struct Command {
    #[command(subcommand)]
    pub command: Commands,
}

generate_sync_commands!(return_type = Result<(), anyhow::Error>; fasta, genbank, gfa, library);
