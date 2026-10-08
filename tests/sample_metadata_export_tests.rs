use std::{fs, path::Path, process::Command};

use serde_json::{Value, json};
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
            "sample-metadata",
            "samples.json",
            "--json",
            "--keys",
            "Score",
        ],
    );
    let samples: Value = serde_json::from_str(
        &fs::read_to_string(directory.path().join("samples.json"))
            .expect("should read samples JSON"),
    )
    .expect("should parse samples JSON");
    assert_eq!(
        samples,
        json!([
            {"name": "sample_001", "metadata": {"Score": 90.68}},
            {"name": "sample_002", "metadata": {"Score": 27.79}}
        ])
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

#[test]
fn test_import_sample_metadata_cli_round_trip_and_validation() {
    let directory = tempdir().expect("should create repository directory");
    run_gen(directory.path(), &["init"]);
    let fasta = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
    run_gen(
        directory.path(),
        &[
            "import",
            "fasta",
            fasta.to_str().expect("should encode FASTA path"),
            "--collection",
            "test",
            "--sample",
            "sample",
        ],
    );
    let input = "sample_name\tkey\tvalue_type\tvalue\nsample\tboolean\tboolean\tfalse\nsample\tfloat\tfloat\t1.25\nsample\tinteger\tinteger\t42\nsample\ttext\ttext\t\"tabs\tand\nquotes\"\"\"\n";
    fs::write(directory.path().join("input.tsv"), input).expect("should write metadata TSV");
    run_gen(directory.path(), &["import", "metadata", "input.tsv"]);
    run_gen(
        directory.path(),
        &["export", "sample-metadata", "metadata.json", "--json"],
    );
    let exported: Value = serde_json::from_str(
        &fs::read_to_string(directory.path().join("metadata.json")).expect("should read JSON"),
    )
    .expect("should parse JSON");
    assert_eq!(
        exported,
        json!([{"name": "sample", "metadata": {
            "boolean": false, "float": 1.25, "integer": 42, "text": "tabs\tand\nquotes\""
        }}])
    );
    run_gen(
        directory.path(),
        &[
            "export",
            "--ref",
            "HEAD",
            "sample-metadata",
            "filtered.json",
            "--json",
            "--sample",
            "sample",
            "--keys",
            "boolean,integer",
        ],
    );
    let filtered: Value = serde_json::from_str(
        &fs::read_to_string(directory.path().join("filtered.json"))
            .expect("should read filtered JSON"),
    )
    .expect("should parse filtered JSON");
    assert_eq!(
        filtered,
        json!([{"name": "sample", "metadata": {"boolean": false, "integer": 42}}])
    );
    run_gen(
        directory.path(),
        &[
            "export",
            "sample-metadata",
            "empty.json",
            "--json",
            "--keys",
            "missing",
        ],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("empty.json")).expect("should read empty JSON"),
        "[]"
    );

    run_gen(
        directory.path(),
        &["export", "sample-metadata", "output.tsv"],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("output.tsv")).expect("should read export"),
        input
    );
    run_gen(directory.path(), &["branch", "--create", "metadata-import"]);
    run_gen(
        directory.path(),
        &["checkout", "--branch", "metadata-import"],
    );
    fs::write(directory.path().join("update.tsv"),
        "sample_name\tkey\tvalue_type\tvalue\nsample\tboolean\tinteger\t1\nsample\tboolean\tboolean\ttrue\n")
        .expect("should write update TSV");
    run_gen(
        directory.path(),
        &["import", "sample-metadata", "update.tsv"],
    );
    let updated = input.replace("boolean\tfalse", "boolean\ttrue");
    run_gen(
        directory.path(),
        &["export", "sample-metadata", "output.tsv"],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("output.tsv"))
            .expect("should read updated export"),
        updated
    );
    for invalid in [
        "wrong\theader\n",
        "sample_name\tkey\tvalue_type\tvalue\nsample\tnew\ttext\tvalid\nsample\tbad\tfloat\tNaN\n",
        "sample_name\tkey\tvalue_type\tvalue\nsample\tbad\tinteger\t9223372036854775808\n",
        "sample_name\tkey\tvalue_type\tvalue\nsample\tbad\tboolean\t2\n",
        "sample_name\tkey\tvalue_type\tvalue\nsample\tbad\tunknown\tvalue\n",
        "sample_name\tkey\tvalue_type\tvalue\nmissing\tbad\ttext\tvalue\n",
        "sample_name\tkey\tvalue_type\tvalue\nsample\tbad\ttext\n",
    ] {
        fs::write(directory.path().join("invalid.tsv"), invalid).expect("should write invalid TSV");
        let result = Command::new(env!("CARGO_BIN_EXE_gen"))
            .current_dir(directory.path())
            .args(["import", "metadata", "invalid.tsv"])
            .output()
            .expect("should run invalid import");
        assert!(!result.status.success(), "invalid TSV should be rejected");
        run_gen(
            directory.path(),
            &["export", "sample-metadata", "output.tsv"],
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("output.tsv"))
                .expect("should read preserved export"),
            updated
        );
    }
    run_gen(directory.path(), &["checkout", "--branch", "main"]);
    run_gen(
        directory.path(),
        &["export", "sample-metadata", "output.tsv"],
    );
    assert_eq!(
        fs::read_to_string(directory.path().join("output.tsv"))
            .expect("should read original export"),
        input
    );
}
