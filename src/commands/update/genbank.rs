use std::fs::File;

use anyhow::Result;
use clap::Args;
use gen_models::{
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo},
};

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    updates::genbank::update_with_genbank,
};

/// Update with a GenBank file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// GenBank file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long, default_value = "reference")]
    sample: String,
    /// If a new entity is found, create it as a normal import
    #[arg(long, action, alias = "cm")]
    create_missing: bool,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Update with GenBank called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));

    let f = File::open(&cmd.path)?;
    match update_with_genbank(
        context,
        &f,
        name.as_ref(),
        &cmd.sample,
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
