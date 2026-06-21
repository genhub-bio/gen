use clap::{Args, Subcommand};

use crate::commands::cli_context::CliContext;

mod chunks;
mod subgraph;
mod translation;

#[derive(Debug, Args, Clone)]
pub struct Command {
    #[command(subcommand)]
    pub command: Commands,
}

/// Derive commands
#[derive(Clone, Debug, Subcommand)]
pub enum Commands {
    /// Translate a gene annotation into a protein sequence graph
    Translation(translation::Command),
    /// Replace a sequence graph with a subgraph in a coordinate range
    Subgraph(subgraph::Command),
    /// Replace a sequence graph with subgraphs at specified breakpoints
    Chunks(chunks::Command),
}

pub fn execute(ctx: &CliContext, command: Command) -> anyhow::Result<()> {
    match command.command {
        Commands::Translation(cmd) => translation::execute(ctx, cmd),
        Commands::Subgraph(cmd) => subgraph::execute(ctx, cmd),
        Commands::Chunks(cmd) => chunks::execute(ctx, cmd),
    }
}
