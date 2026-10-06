use std::{fs, path::Path, process::Command};

use tempfile::tempdir;

fn run_gen(repository: &Path, arguments: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_gen"))
        .current_dir(repository)
        .args(arguments)
        .output()
        .expect("should run gen command");
    assert!(
        output.status.success(),
        "command should succeed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn test_export_sample_metadata_cli() {
    let directory = tempdir().expect("should create repository directory");
    run_gen(directory.path(), &["init"]);
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let fasta = fixtures.join("simple.fa");
    let variants = directory.path().join("metadata.vcf");
    let input = fs::read_to_string(fixtures.join("simple-metadata.vcf"))
        .expect("should read VCF fixture")
        .replace("Score=90.68>", "Score=90.68,Count=1>")
        .replace("Score=27.79>", "Score=27.79,Count=2>");
    fs::write(&variants, input).expect("should write metadata VCF");
    run_gen(
        directory.path(),
        &[
            "import",
            "fasta",
            fasta.to_str().expect("should encode FASTA path"),
            "--collection",
            "test",
            "--sample",
            "reference",
        ],
    );
    run_gen(
        directory.path(),
        &[
            "update",
            "vcf",
            variants.to_str().expect("should encode VCF path"),
            "--collection",
            "test",
            "--parent-samples",
            "reference",
        ],
    );
    run_gen(
        directory.path(),
        &["export", "sample-metadata", "without-metadata.tsv"],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("without-metadata.tsv"))
            .expect("should read export without metadata"),
        "sample_name\tkey\tvalue_type\tvalue\n"
    );
    run_gen(
        directory.path(),
        &[
            "update",
            "vcf",
            variants.to_str().expect("should encode VCF path"),
            "--collection",
            "test",
            "--read-metadata",
            "--parent-samples",
            "reference",
        ],
    );
    run_gen(
        directory.path(),
        &["export", "sample-metadata", "metadata.tsv"],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("metadata.tsv")).expect("should read TSV"),
        "sample_name\tkey\tvalue_type\tvalue\nsample_001\tCount\tinteger\t1\nsample_001\tScore\tfloat\t90.68\nsample_002\tCount\tinteger\t2\nsample_002\tScore\tfloat\t27.79\n"
    );
    run_gen(
        directory.path(),
        &[
            "export",
            "--ref",
            "HEAD",
            "sample-metadata",
            "filtered.tsv",
            "--sample",
            "sample_002",
            "--keys",
            "missing,Score",
        ],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("filtered.tsv"))
            .expect("should read filtered TSV"),
        "sample_name\tkey\tvalue_type\tvalue\nsample_002\tScore\tfloat\t27.79\n"
    );
}

#[test]
fn test_export_sample_metadata_cli_unmatched_keys() {
    let directory = tempdir().expect("should create repository directory");
    run_gen(directory.path(), &["init"]);
    run_gen(
        directory.path(),
        &[
            "export",
            "sample-metadata",
            "metadata.tsv",
            "--keys",
            "missing,other",
        ],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("metadata.tsv")).expect("should read TSV"),
        "sample_name\tkey\tvalue_type\tvalue\n"
    );
}
