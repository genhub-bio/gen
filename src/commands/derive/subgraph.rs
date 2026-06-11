use clap::Args;

use crate::commands::{
    cli_context::CliContext, graph_operations::derive_subgraph::derive_subgraph_operation,
};

/// Derive a subgraph from a coordinate range
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// The name of the collection (defaults to the current default collection)
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the parent sample
    #[arg(short, long)]
    sample: String,
    /// The name of the new sample
    #[arg(long)]
    new_sample: String,
    /// The region to derive (e.g. chr1:1000-2000)
    #[arg(short, long)]
    region: String,
    /// Name of an alternate path to use as the backbone
    #[arg(long)]
    backbone: Option<String>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> anyhow::Result<()> {
    derive_subgraph_operation(
        cli_context.context,
        cmd.name,
        cmd.sample,
        cmd.new_sample,
        cmd.region,
        cmd.backbone,
    )
}
