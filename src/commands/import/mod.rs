use crate::commands::cli_context::CliContext;
use clap::{Args, Subcommand};
use clap_nested_commands::generate_sync_commands;

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

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
/// Import a new sequence collection.
enum Import {
    /// Import a library
    #[command(arg_required_else_help(true))]
    Library {},
}

generate_sync_commands!(fasta, genbank, gfa, library);
