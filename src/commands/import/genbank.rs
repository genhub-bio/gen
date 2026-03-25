use std::fs::File;

use anyhow::Result;
use clap::Args;
use gen_models::{
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo},
};

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    imports::genbank::import_genbank,
};

/// Import a Genbank file
#[derive(Debug, Args, Clone)]
pub struct Command {
    /// Genbank file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to store the entry under
    #[arg(short, long)]
    name: Option<String>,
    /// A sample name to associate the Genbank file with
    #[arg(short, long)]
    sample: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) -> Result<()> {
    println!("Genbank import called");

    let context = cli_context.context;
    let operation_conn = context.operations().conn();
    let conn = context.graph().conn();

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(operation_conn));
    let mut reader: Box<dyn std::io::Read> = if cmd.path.ends_with(".gz") {
        let file = File::open(cmd.path.clone()).unwrap();
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(File::open(cmd.path.clone()).unwrap())
    };
    match import_genbank(
        context,
        &mut reader,
        name.as_ref(),
        cmd.sample.as_str(),
        OperationInfo {
            files: vec![OperationFile {
                file_path: cmd.path.clone(),
                file_type: FileTypes::GenBank,
            }],
            description: "GenBank Import".to_string(),
        },
    ) {
        Ok(_) => {
            println!("GenBank imported.");
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
            Ok(())
        }
        Err(err) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            println!("Import failed: {err:?}");
            Err(err.into())
        }
    }
}
