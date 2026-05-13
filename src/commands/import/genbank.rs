use std::fs::File;

use anyhow::Result;
use clap::Args;
use gen_models::{
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo},
    sample::Sample,
};

use crate::{
    commands::{cli_context::CliContext, get_default_collection},
    imports::genbank::{GenBankImportOptions, import_genbank},
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
    #[arg(short, long, conflicts_with = "reference")]
    sample: Option<String>,
    /// Create or use a reference sample with this name
    #[arg(long, conflicts_with = "sample")]
    reference: Option<String>,
    /// Override the annotation group name created for imported GenBank annotations
    #[arg(long = "annotation-group")]
    annotation_group: Option<String>,
    /// Skip importing GenBank feature annotations
    #[arg(long)]
    no_annotations: bool,
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
    let (sample_name, is_reference) = crate::commands::import::resolve_import_sample(
        cmd.sample.as_deref(),
        cmd.reference.as_deref(),
    )?;
    if is_reference {
        Sample::get_or_create_reference(conn, sample_name)?;
    }
    let mut options = GenBankImportOptions::default().annotation_name_from_path(&cmd.path);
    options.add_annotations = !cmd.no_annotations;
    options.annotation_group = cmd.annotation_group.clone();
    match import_genbank(
        context,
        &mut reader,
        name.as_ref(),
        sample_name,
        OperationInfo {
            files: vec![OperationFile::new(cmd.path.clone()).set_file_type(FileTypes::GenBank)],
            description: "GenBank Import".to_string(),
        },
        options,
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
