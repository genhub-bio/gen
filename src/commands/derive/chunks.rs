use clap::Args;

use crate::commands::{
    cli_context::CliContext, graph_operations::derive_chunks::derive_chunks_operation,
};

/// Derive subgraphs at specified breakpoints
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
    /// The region to split (e.g. chr1:1000-2000)
    #[arg(short, long)]
    region: String,
    /// Name of an alternate path to use as the backbone
    #[arg(long)]
    backbone: Option<String>,
    /// Comma-separated breakpoints (e.g. 1000,2000,3000)
    #[arg(long)]
    breakpoints: Option<String>,
    /// Uniform chunk size in bp
    #[arg(long)]
    chunk_size: Option<i64>,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> anyhow::Result<()> {
    let parsed_breakpoints = cmd
        .breakpoints
        .map(|s| {
            s.split(',')
                .map(|bp| {
                    bp.parse::<i64>()
                        .map_err(|_| anyhow::anyhow!("Invalid breakpoint: {bp}"))
                })
                .collect::<Result<Vec<i64>, _>>()
        })
        .transpose()?;

    derive_chunks_operation(
        cli_context.context,
        cmd.name,
        cmd.sample,
        cmd.new_sample,
        cmd.region,
        cmd.backbone,
        parsed_breakpoints,
        cmd.chunk_size,
    )
}
