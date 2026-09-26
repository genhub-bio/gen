use std::{fs::File, path::PathBuf};

use gen_core::Workspace;
use gen_models::{
    block_group::{BlockGroup, BlockGroupError},
    collection::Collection,
    db::GraphConnection,
    errors::PathError,
    sample::Sample,
};
use noodles::fasta;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FastaExportError {
    /// The sample, collection, and revision selection matched no block groups.
    #[error("No block groups found for {0}; check the sample, collection, and revision")]
    NoMatchingBlockGroups(String),
    #[error("I/O error while exporting FASTA: {0}")]
    Io(#[from] std::io::Error),
    #[error("Path error while exporting FASTA: {0}")]
    Path(#[from] PathError),
    #[error("Block group error while exporting FASTA: {0}")]
    BlockGroup(#[from] BlockGroupError),
}

pub fn export_fasta(
    conn: &GraphConnection,
    workspace: &Workspace,
    collection_name: &str,
    sample_name: Option<&str>,
    filename: &PathBuf,
    history_ref: Option<&str>,
) -> Result<(), FastaExportError> {
    let block_groups = if let Some(sample_name) = sample_name {
        Sample::get_block_groups(conn, collection_name, sample_name, history_ref)
    } else {
        Collection::get_block_groups(conn, collection_name, history_ref)
    };

    // A failed selection must not look like a successful export or truncate an
    // existing output file. Missing paths within a matching group error below.
    if block_groups.is_empty() {
        let selection = match sample_name {
            Some(sample_name) => {
                format!("sample '{sample_name}' in collection '{collection_name}'")
            }
            None => format!("collection '{collection_name}'"),
        };
        let selection = match history_ref {
            Some(history_ref) => format!("{selection} at revision '{history_ref}'"),
            None => selection,
        };
        return Err(FastaExportError::NoMatchingBlockGroups(selection));
    }

    let file = File::create(filename)?;
    let mut writer = fasta::io::Writer::new(file);

    for block_group in block_groups {
        let path = BlockGroup::get_current_path(conn, &block_group.id, history_ref)?;

        let definition = fasta::record::Definition::new(block_group.name, None);
        let sequence = fasta::record::Sequence::from(
            path.sequence(conn, workspace, history_ref)?.into_bytes(),
        );
        let record = fasta::Record::new(definition, sequence);

        writer.write_record(&record)?;
    }

    println!("Exported to file {}", filename.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use std::{io, path::PathBuf, str};

    use noodles::fasta;
    use tempfile;

    use super::*;
    use crate::{
        imports::fasta::import_fasta, test_helpers::setup_gen, updates::fasta::update_with_fasta,
    };

    #[test]
    fn test_export_missing_selection_preserves_output() {
        let context = setup_gen();
        let directory = tempfile::tempdir().unwrap();
        let filename = directory.path().join("out.fa");
        std::fs::write(&filename, "existing output").unwrap();
        for sample_name in [Some("sample_0001"), None] {
            let error = export_fasta(
                context.graph().conn(),
                context.workspace(),
                "missing-collection",
                sample_name,
                &filename,
                None,
            )
            .unwrap_err();
            assert!(matches!(error, FastaExportError::NoMatchingBlockGroups(_)));
            assert!(error.to_string().contains("missing-collection"));
            assert_eq!(
                std::fs::read_to_string(&filename).unwrap(),
                "existing output"
            );
        }
    }

    #[test]
    fn test_import_then_export() {
        let context = setup_gen();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.fa");
        export_fasta(
            conn,
            context.workspace(),
            &collection,
            None,
            &filename,
            None,
        )
        .unwrap();

        let mut fasta_reader = fasta::io::reader::Builder
            .build_from_path(filename)
            .unwrap();
        let record = fasta_reader
            .records()
            .next()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "No records found in fasta file")
            })
            .unwrap()
            .unwrap();

        let sequence = str::from_utf8(record.sequence().as_ref())
            .unwrap()
            .to_string();
        assert_eq!(sequence, "ATCGATCGATCGATCGATCGGGAACACACAGAGA");
    }

    #[test]
    fn test_import_fasta_update_with_fasta_export() {
        /*
        Graph after fasta update:
        AT ----> CGA ------> TCGATCGATCGATCGGGAACACACAGAGA
           \-> AAAAAAAA --/
        */
        let context = setup_gen();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aaaaaaaa.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123:2-5",
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.fa");
        export_fasta(
            conn,
            context.workspace(),
            &collection,
            Some("child sample"),
            &filename,
            None,
        )
        .unwrap();

        let mut fasta_reader = fasta::io::reader::Builder
            .build_from_path(filename)
            .unwrap();
        let record = fasta_reader
            .records()
            .next()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "No records found in fasta file")
            })
            .unwrap()
            .unwrap();

        let sequence = str::from_utf8(record.sequence().as_ref())
            .unwrap()
            .to_string();
        assert_eq!(sequence, "ATAAAAAAAATCGATCGATCGATCGGGAACACACAGAGA");
    }
}
