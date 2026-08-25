use anyhow::Result;
use clap::Args;
use gen_models::{
    errors::OperationError,
    sample::{NewSample, Sample},
};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    fasta::FastaError,
    imports::fasta::import_fasta,
};

/// Import a fasta file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Fasta file path
    #[clap(index = 1)]
    pub path: String,
    /// Don't store the sequence in the database, instead store a reference to an asset
    #[arg(long, action)]
    shallow: bool,
    /// Index associated with the shallow FASTA. May be specified multiple times.
    #[arg(long, requires = "shallow")]
    index: Vec<String>,
    /// The name of the collection to store the entry under
    #[arg(short, long)]
    name: Option<String>,
    /// A sample name to associate the fasta file with
    #[arg(short, long, conflicts_with = "reference")]
    sample: Option<String>,
    /// Create or use a reference sample with this name
    #[arg(long, conflicts_with = "sample")]
    reference: Option<String>,
}

#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(cli_context, cmd))
)]
pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Fasta import called");

    let context = cli_context.context;
    let config_conn = context.config().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(config_conn));
    let (sample_name, is_reference) = crate::commands::import::resolve_import_sample(
        cmd.sample.as_deref(),
        cmd.reference.as_deref(),
    )?;
    if is_reference {
        Sample::get_or_create(
            conn,
            NewSample {
                name: sample_name,
                is_reference: true,
            },
        )?;
    }
    match import_fasta(
        context,
        &cmd.path.clone(),
        name,
        sample_name,
        cmd.shallow,
        &cmd.index,
    ) {
        Ok(operation_summary) => {
            conn.execute("END TRANSACTION", [])?;
            match commit_operation(context, &operation_summary) {
                Ok(_) => {
                    println!("Fasta imported.");
                    Ok(())
                }
                Err(OperationError::NoChanges) => {
                    println!("Fasta contents already exist.");
                    Ok(())
                }
                Err(e) => Err(e.into()),
            }
        }
        Err(FastaError::OperationError(OperationError::NoChanges)) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            println!("Fasta contents already exist.");
            Ok(())
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            Err(e.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Command;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        command: Command,
    }

    #[test]
    fn test_accepts_multiple_remote_fasta_indexes() {
        let cli = TestCli::try_parse_from([
            "test",
            "https://example.test/reference.fa.bgz",
            "--shallow",
            "--index",
            "https://example.test/reference.fa.bgz.fai",
            "--index",
            "https://example.test/reference.fa.bgz.gzi",
        ])
        .expect("should parse multiple shallow FASTA indexes");

        assert_eq!(
            cli.command.index,
            [
                "https://example.test/reference.fa.bgz.fai",
                "https://example.test/reference.fa.bgz.gzi",
            ],
            "repeatable --index values should preserve every supplied URI"
        );
    }
}
