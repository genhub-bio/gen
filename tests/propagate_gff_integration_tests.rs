#[cfg(test)]
mod propagate_gff_integration_tests {
    use std::{fs::File, io::BufReader, path::PathBuf};

    use r#gen::{
        imports::fasta::import_fasta, test_helpers::setup_gen, track_database,
        updates::fasta::update_with_fasta,
    };
    use gen_annotations::gff::propagate_gff;
    use noodles::gff;
    use tempfile::tempdir;

    #[test]
    fn test_simple_propagate() {
        let context = setup_gen();
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aa.fa");
        let mut gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        gff_path.push("fixtures/simple.gff");
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            "test",
            None,
            false,
        )
        .unwrap();

        let _ = update_with_fasta(
            &context,
            "test",
            None,
            "child sample",
            "m123",
            15,
            25,
            fasta_update_path.to_str().unwrap(),
            false,
        );

        let temp_dir = tempdir().expect("should create temp directory");
        let mut output_path = PathBuf::from(temp_dir.path());
        output_path.push("output.gff");
        let _ = propagate_gff(
            conn,
            "test",
            None,
            "child sample",
            gff_path.to_str().unwrap(),
            output_path.to_str().unwrap(),
        );

        let reader = File::open(output_path.to_str().unwrap())
            .map(BufReader::new)
            .map(gff::io::Reader::new);

        for (i, result) in reader
            .expect("should read output file")
            .record_bufs()
            .enumerate()
        {
            let record = result.unwrap();
            assert_eq!(record.reference_sequence_name(), "m123");
            if i == 0 {
                assert_eq!(record.reference_sequence_name(), "m123");
                assert_eq!(record.source(), "gen-test");
                assert_eq!(record.ty(), "Region");
                assert_eq!(record.start().get(), 1);
                assert_eq!(record.end().get(), 26);
            } else {
                assert_eq!(record.reference_sequence_name(), "m123");
                assert_eq!(record.source(), "gen-test");
                assert_eq!(record.ty(), "Gene");
                assert_eq!(record.start().get(), 5);
                assert_eq!(record.end().get(), 15);
            }
        }
    }
}
