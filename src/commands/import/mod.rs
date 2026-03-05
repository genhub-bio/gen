use anyhow;
use clap::{Args, Subcommand};

use crate::commands::cli_context::CliContext;

mod fasta;
mod genbank;
mod gfa;
mod library;

#[derive(Debug, Args, Clone)]
pub struct Command {
    #[command(subcommand)]
    pub command: Commands,
}

/// Import commands
#[derive(Clone, Debug, Subcommand)]
pub enum Commands {
    /// Import fasta
    Fasta(fasta::Command),
    /// Import genbank
    Genbank(genbank::Command),
    /// Import gfa
    Gfa(gfa::Command),
    /// Import library
    Library(library::Command),
}

pub fn execute(ctx: &CliContext, command: Command) -> anyhow::Result<()> {
    match command.command {
        Commands::Fasta(cmd) => crate::commands::import::fasta::execute(ctx, cmd),
        Commands::Genbank(cmd) => crate::commands::import::genbank::execute(ctx, cmd),
        Commands::Gfa(cmd) => crate::commands::import::gfa::execute(ctx, cmd),
        Commands::Library(cmd) => crate::commands::import::library::execute(ctx, cmd),
    }
}
