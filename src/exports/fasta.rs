use std::{fs::File, path::PathBuf};

use gen_models::{
    block_group::BlockGroup, collection::Collection, db::GraphConnection, sample::Sample,
};
use noodles::fasta;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FastaExportError {
    #[error("I/O error while exporting FASTA: {0}")]
    Io(#[from] std::io::Error),
}

pub fn export_fasta(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: Option<&str>,
    filename: &PathBuf,
) -> Result<(), FastaExportError> {
    let block_groups = if let Some(sample_name) = sample_name {
        Sample::get_block_groups(conn, collection_name, sample_name)
    } else {
        Collection::get_block_groups(conn, collection_name)
    };

    let file = File::create(filename)?;
    let mut writer = fasta::io::Writer::new(file);

    for block_group in block_groups {
        let path = BlockGroup::get_current_path(conn, &block_group.id);

        let definition = fasta::record::Definition::new(block_group.name, None);
        let sequence = fasta::record::Sequence::from(path.sequence(conn).into_bytes());
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
        imports::fasta::import_fasta, test_helpers::setup_gen, track_database,
        updates::fasta::update_with_fasta,
    };

    #[test]
    fn test_import_then_export() {
        let context = setup_gen();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.fa");
        export_fasta(conn, &collection, None, &filename).unwrap();

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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        let _ = update_with_fasta(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "child sample",
            "m123",
            2,
            5,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let tmp_dir = tempfile::tempdir().unwrap().keep();
        let filename = tmp_dir.join("out.fa");
        export_fasta(conn, &collection, Some("child sample"), &filename).unwrap();

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
