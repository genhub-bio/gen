use anyhow;
use clap::{Args, Subcommand};

use crate::commands::cli_context::CliContext;

mod fasta;
mod genbank;
mod gfa;

#[derive(Debug, Args, Clone)]
pub struct Command {
    #[arg(long = "ref")]
    pub history_ref: Option<String>,
    #[command(subcommand)]
    pub command: Commands,
}

/// Export commands
#[derive(Clone, Debug, Subcommand)]
pub enum Commands {
    /// Export fasta
    Fasta(fasta::Command),
    /// Export genbank
    Genbank(genbank::Command),
    /// Export gfa
    Gfa(gfa::Command),
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(ctx, command))
)]
pub fn execute(ctx: &CliContext, command: Command) -> anyhow::Result<()> {
    match command.command {
        Commands::Fasta(cmd) => crate::commands::export::fasta::execute(ctx, cmd),
        Commands::Genbank(cmd) => crate::commands::export::genbank::execute(ctx, cmd),
        Commands::Gfa(cmd) => crate::commands::export::gfa::execute(ctx, cmd),
    }
}
