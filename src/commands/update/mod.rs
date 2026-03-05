use anyhow;
use clap::{Args, Subcommand};

use crate::commands::cli_context::CliContext;

mod fasta;
mod gaf;
mod genbank;
mod gfa;
mod library;
mod sequence;
mod vcf;

#[derive(Debug, Args, Clone)]
pub struct Command {
    #[command(subcommand)]
    pub command: Commands,
}

/// Update commands
#[derive(Clone, Debug, Subcommand)]
pub enum Commands {
    /// Update with fasta
    Fasta(fasta::Command),
    /// Update with GAF
    Gaf(gaf::Command),
    /// Update with genbank
    Genbank(genbank::Command),
    /// Update with gfa
    Gfa(gfa::Command),
    /// Update with library
    Library(library::Command),
    /// Update with sequence
    Sequence(sequence::Command),
    /// Update with vcf
    Vcf(vcf::Command),
}

pub fn execute(ctx: &CliContext, command: Command) -> anyhow::Result<()> {
    match command.command {
        Commands::Fasta(cmd) => crate::commands::update::fasta::execute(ctx, cmd),
        Commands::Gaf(cmd) => crate::commands::update::gaf::execute(ctx, cmd),
        Commands::Genbank(cmd) => crate::commands::update::genbank::execute(ctx, cmd),
        Commands::Gfa(cmd) => crate::commands::update::gfa::execute(ctx, cmd),
        Commands::Library(cmd) => crate::commands::update::library::execute(ctx, cmd),
        Commands::Sequence(cmd) => crate::commands::update::sequence::execute(ctx, cmd),
        Commands::Vcf(cmd) => crate::commands::update::vcf::execute(ctx, cmd),
    }
}
