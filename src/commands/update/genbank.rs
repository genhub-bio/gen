use std::fs::File;

use anyhow::Result;
use clap::Args;
use gen_models::{
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo},
};

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    get_connection, get_operation_connection,
    updates::genbank::update_with_genbank,
};

/// Update with a GenBank file
#[derive(Debug, Args)]
pub struct Command {
    /// GenBank file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// If a new entity is found, create it as a normal import
    #[arg(long, action, alias = "cm")]
    create_missing: bool,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with GenBank called");

    let operation_conn = get_operation_connection(None)?;
    let db = get_db_for_command(cli_context.db.clone(), &operation_conn);
    let conn = get_connection(&db)?;

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));

    let f = File::open(&cmd.path)?;
    match update_with_genbank(
        &conn,
        &operation_conn,
        &f,
        name.as_ref(),
        cmd.create_missing,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: cmd.path.clone(),
                file_type: FileTypes::GenBank,
            }],
            description: "Update from GenBank".to_string(),
        },
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", [])?;
            operation_conn.execute("END TRANSACTION;", [])?;
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", [])?;
            operation_conn.execute("ROLLBACK TRANSACTION;", [])?;
            return Err(e.into());
        }
    }

    Ok(())
}
