use crate::commands::cli_context::CliContext;
use clap::{Args, Subcommand};
use clap_nested_commands::generate_sync_commands;

mod fasta;

/// Import commands
#[derive(Debug, Args)]
pub struct Command {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
/// Import a new sequence collection.
enum Import {
    /// Import a fasta
    #[command(arg_required_else_help(true))]
    Fasta {
        /// Fasta file path
        #[clap(index = 1)]
        fasta: String,
        /// Don't store the sequence in the database, instead store the filename
        #[arg(long, action)]
        shallow: bool,
        /// The name of the collection to store the entry under
        #[arg(short, long)]
        name: Option<String>,
        /// A sample name to associate the fasta file with
        #[arg(short, long)]
        sample: Option<String>,
    },
    /// Import a genbank
    #[command(arg_required_else_help(true))]
    Genbank {
        /// Genbank file path
        #[clap(index = 1)]
        gb: String,
        /// The name of the collection to store the entry under
        #[arg(short, long)]
        name: Option<String>,
        /// A sample name to associate the genbank file with
        #[arg(short, long)]
        sample: Option<String>,
    },
    /// Import a GFA
    #[command(arg_required_else_help(true))]
    Gfa {
        /// GFA file path
        #[clap(index = 1)]
        gfa: String,
        /// The name of the collection to store the entry under
        #[arg(short, long)]
        name: Option<String>,
        /// A sample name to associate the GFA file with
        #[arg(short, long)]
        sample: Option<String>,
    },
    /// Import a library
    #[command(arg_required_else_help(true))]
    Library {
        /// The name of the region
        #[arg(long)]
        region_name: Option<String>,
        /// The path to the combinatorial library parts
        #[arg(long)]
        parts: Option<String>,
        /// The path to the combinatorial library csv
        #[arg(long)]
        library: Option<String>,
        /// The name of the collection to store the entry under
        #[arg(short, long)]
        name: Option<String>,
        /// A sample name to associate the library with
        #[arg(short, long)]
        sample: Option<String>,
    },
}

// fn blah() {
//             Some(Commands::Import {
//             fasta,
//             gb,
//             gfa,
//             name,
//             shallow,
//             sample,
//             region_name,
//             parts,
//             library,
//         }) => {
//             conn.execute("BEGIN TRANSACTION", []).unwrap();
//             operation_conn.execute("BEGIN TRANSACTION", []).unwrap();
//             let name = &name
//                 .clone()
//                 .unwrap_or_else(|| get_default_collection(&operation_conn));
//             if fasta.is_some() {
//                 match import_fasta(
//                     &fasta.clone().unwrap(),
//                     name,
//                     sample.as_deref(),
//                     *shallow,
//                     &conn,
//                     &operation_conn,
//                 ) {
//                     Ok(_) => println!("Fasta imported."),
//                     Err(FastaError::OperationError(OperationError::NoChanges)) => {
//                         println!("Fasta contents already exist.")
//                     }
//                     Err(_) => {
//                         conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                         operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                         panic!("Import failed.");
//                     }
//                 }
//             } else if gfa.is_some() {
//                 match import_gfa(
//                     &PathBuf::from(gfa.clone().unwrap()),
//                     name,
//                     sample.as_deref(),
//                     &conn,
//                     &operation_conn,
//                 ) {
//                     Ok(_) => println!("GFA Imported."),
//                     Err(GFAImportError::OperationError(OperationError::NoChanges)) => {
//                         println!("GFA already exists.")
//                     }
//                     Err(_) => {
//                         conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                         operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                         panic!("Import failed.");
//                     }
//                 }
//             } else if let Some(gb) = gb {
//                 let mut reader: Box<dyn std::io::Read> = if gb.ends_with(".gz") {
//                     let file = File::open(gb.clone()).unwrap();
//                     Box::new(flate2::read::GzDecoder::new(file))
//                 } else {
//                     Box::new(File::open(gb.clone()).unwrap())
//                 };
//                 match import_genbank(
//                     &conn,
//                     &operation_conn,
//                     &mut reader,
//                     name.deref(),
//                     sample.as_deref(),
//                     OperationInfo {
//                         files: vec![OperationFile {
//                             file_path: gb.clone(),
//                             file_type: FileTypes::GenBank,
//                         }],
//                         description: "GenBank Import".to_string(),
//                     },
//                 ) {
//                     Ok(_) => println!("GenBank Imported."),
//                     Err(err) => {
//                         conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                         operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                         panic!("Import failed: {err:?}");
//                     }
//                 }
//             } else if region_name.is_some() && parts.is_some() && library.is_some() {
//                 import_library(
//                     &conn,
//                     &operation_conn,
//                     name,
//                     sample.as_deref(),
//                     parts.as_deref().unwrap(),
//                     library.as_deref().unwrap(),
//                     region_name.as_deref().unwrap(),
//                 )
//                 .unwrap();
//             } else {
//                 conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                 operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
//                 panic!(
//                     "ERROR: Import command attempted but no recognized file format was specified"
//                 );
//             }
//             conn.execute("END TRANSACTION", []).unwrap();
//             operation_conn.execute("END TRANSACTION", []).unwrap();
//             }
// }

generate_sync_commands!(fasta);
