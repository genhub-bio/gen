use anyhow::Result;
use clap::Args;
use gen_models::{
    errors::OperationError,
    sample::{NewSample, Sample},
};

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    fasta::FastaError,
    imports::fasta::import_fasta,
};

/// Import a fasta file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Fasta file path
    #[clap(index = 1)]
    pub path: String,
    /// Don't store the sequence in the database, instead store the filename
    #[arg(long, action)]
    shallow: bool,
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
    let operation_conn = context.config().conn();
    let conn = context.graph().conn();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));
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
    match import_fasta(context, &cmd.path.clone(), name, sample_name, cmd.shallow) {
        Ok(_) => {
            println!("Fasta imported.");
            Ok(())
        }
        Err(FastaError::OperationError(OperationError::NoChanges)) => {
            println!("Fasta contents already exist.");
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use gen_core::config::Workspace;
    use gen_models::db::DbContext;

    use super::{Command, execute};
    use crate::{commands::cli_context::CliContext, get_config_connection, get_connection};

    #[test]
    fn test_execute_imports_fasta_in_fresh_repo_without_manual_database_tracking() {
        let temp_dir = tempfile::tempdir().expect("should create temp directory");
        let workspace = Workspace::new(temp_dir.path());
        workspace.ensure_gen_dir();

        let graph_connection = get_connection(
            workspace
                .graph_db_path()
                .expect("should resolve graph db path"),
        )
        .expect("should open graph connection");
        let operation_connection = get_config_connection(Some(
            workspace
                .gen_db_path()
                .expect("should resolve config db path"),
        ))
        .expect("should open config connection");
        let db_context = DbContext::new(workspace, graph_connection, operation_connection).unwrap();
        let cli_context = CliContext {
            context: &db_context,
            history_ref: None,
        };

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let command = Command {
            path: fasta_path.to_string_lossy().to_string(),
            shallow: false,
            name: Some("test-collection".to_string()),
            sample: Some("test-sample".to_string()),
            reference: None,
        };

        let result = execute(&cli_context, command);
        assert!(
            result.is_ok(),
            "fresh CLI import should succeed with default graph/config setup: {result:?}"
        );

        let operation_count = fs::read_dir(
            db_context
                .workspace()
                .asset_dir()
                .expect("should resolve asset directory"),
        )
        .expect("should read asset directory")
        .count();
        assert!(
            operation_count > 0,
            "import should materialize at least one asset"
        );
    }
}
