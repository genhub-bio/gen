use anyhow::Result;
use clap::Args;
use gen_models::{errors::OperationError, sample::Sample};

use crate::{
    commands::{cli_context::CliContext, commit_operation, get_default_collection},
    updates::gaf::update_with_gaf,
};

/// Update with a GAF file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GAF file path
    #[clap(index = 1)]
    pub path: String,
    /// Override the Dolt commit message
    #[arg(short = 'm', long)]
    message: Option<String>,
    /// The name of the collection to update
    #[arg(short = 'c', long)]
    collection: Option<String>,
    /// The name of the sample to update
    #[arg(short, long, default_value_t = Sample::DEFAULT_NAME.to_string())]
    sample: String,
    /// The csv describing changes to make
    #[arg(long)]
    csv: String,
    /// If specified, the newly created sample will inherit this sample's existing graph
    #[arg(short, long, default_value_t = Sample::DEFAULT_NAME.to_string())]
    parent_sample: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with GAF called");

    let context = cli_context.context;
    let config_conn = context.config().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;

    let collection_name = &cmd
        .collection
        .clone()
        .unwrap_or_else(|| get_default_collection(config_conn));

    match update_with_gaf(
        context,
        &cmd.path,
        &cmd.csv,
        collection_name,
        cmd.sample.as_str(),
        Some(cmd.parent_sample.as_str()),
    ) {
        Ok(operation_summary) => {
            conn.execute("END TRANSACTION", [])?;
            let mut operation_summary = operation_summary;
            if let Some(message) = cmd.message {
                operation_summary.summary = message;
            }
            match commit_operation(context, &operation_summary) {
                Ok(_) | Err(OperationError::NoChanges) => {}
                Err(err) => return Err(err.into()),
            }
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(err.into());
        }
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::commands::{Cli, Commands, update::Commands as UpdateCommands};

    #[test]
    fn test_update_gaf_accepts_collection_short_and_csv_long_options() {
        for collection_flag in ["--collection", "-c"] {
            let cli = Cli::try_parse_from([
                "gen",
                "update",
                "gaf",
                "alignment.gaf",
                collection_flag,
                "selected",
                "--csv",
                "changes.csv",
            ])
            .expect("should parse update gaf collection and csv options");

            let Some(Commands::Update(update_command)) = cli.command else {
                panic!("should parse update command");
            };
            let UpdateCommands::Gaf(command) = update_command.command else {
                panic!("should parse GAF update command");
            };

            assert_eq!(command.collection.as_deref(), Some("selected"));
            assert_eq!(command.csv, "changes.csv");
        }
    }
}
