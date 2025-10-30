use clap::{Args, Subcommand};
use clap_nested_commands::generate_sync_commands;

use crate::commands::cli_context::CliContext;

mod fasta;
mod gaf;
mod genbank;
mod gfa;
mod library;
mod sequence;
mod vcf;

/// Import commands
#[derive(Debug, Args)]
pub struct Command {
    #[command(subcommand)]
    pub command: Commands,
}

generate_sync_commands!(fasta, gaf, genbank, gfa, library, sequence, vcf);
