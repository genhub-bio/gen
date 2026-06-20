use anyhow;
use clap::{Args, Subcommand};

use crate::commands::cli_context::CliContext;

mod alignment;
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
    /// Import alignment
    Alignment(alignment::Command),
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
        Commands::Alignment(cmd) => crate::commands::import::alignment::execute(ctx, cmd),
        Commands::Fasta(cmd) => crate::commands::import::fasta::execute(ctx, cmd),
        Commands::Genbank(cmd) => crate::commands::import::genbank::execute(ctx, cmd),
        Commands::Gfa(cmd) => crate::commands::import::gfa::execute(ctx, cmd),
        Commands::Library(cmd) => crate::commands::import::library::execute(ctx, cmd),
    }
}

pub fn resolve_import_sample<'a>(
    sample: Option<&'a str>,
    reference: Option<&'a str>,
) -> anyhow::Result<(&'a str, bool)> {
    match (sample, reference) {
        (Some(sample), None) => Ok((sample, false)),
        (None, Some(reference)) => Ok((reference, true)),
        (None, None) => Err(anyhow::anyhow!(
            "one of --sample or --reference must be provided"
        )),
        (Some(_), Some(_)) => Err(anyhow::anyhow!(
            "--sample and --reference are mutually exclusive"
        )),
    }
}
