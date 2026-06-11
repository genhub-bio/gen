use clap::Args;

use crate::commands::{
    cli_context::CliContext,
    graph_operations::derive_translation::{DeriveTranslationArgs, derive_translation_operation},
};

/// Translate a sequence graph or annotation into a protein sequence graph.
///
/// Pass --region with just a name to translate an entire sequence graph, or
/// include coordinates (e.g. mreB:10-200) to translate a subrange.
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Collection (defaults to the current default collection)
    #[arg(short, long)]
    collection: Option<String>,
    /// The sample that owns the sequence graph or annotation
    #[arg(short, long)]
    sample: String,
    /// Region to translate: a sequence graph name, annotation name, or name with
    /// coordinates (e.g. mreB, mreB:10-200, chr1:1000-2000)
    #[arg(short, long)]
    region: String,
    /// Sample name for the output protein block group
    #[arg(long)]
    new_sample: String,
    /// Strand of the CDS ("forward" or "reverse"; inferred from the annotation when omitted)
    #[arg(long)]
    strand: Option<String>,
    /// Initial reading frame offset (0, 1, or 2)
    #[arg(long, default_value = "0")]
    frame: u8,
    /// NCBI codon table ID (default: 1 = Standard)
    #[arg(long, default_value = "1")]
    codon_table: u8,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> anyhow::Result<()> {
    derive_translation_operation(
        cli_context.context,
        DeriveTranslationArgs {
            collection: cmd.collection,
            sample: cmd.sample,
            region: cmd.region,
            output_sample: cmd.new_sample,
            strand: cmd.strand,
            frame: cmd.frame,
            codon_table: cmd.codon_table,
        },
    )
}
