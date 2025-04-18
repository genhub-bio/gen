use crate::commands::cli_context::CliContext;
use clap::Args;

/// Import a fasta file
#[derive(Debug, Args)]
pub struct Command {
    /// Fasta file path
    #[arg(long)]
    pub path: String,
}

pub fn execute(_cli_context: &CliContext, _cmd: Command) {
    println!("Fasta import called");
}
