use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Result, bail};
use csv::WriterBuilder;
use gen_models::{
    db::GraphConnection,
    sample::Sample,
    sample_metadata::{MetadataValue, SampleMetadata},
};
use serde::Serialize;
use serde_json::Value;

/// Export typed sample metadata as TSV or grouped JSON, ordered by sample name and key.
///
/// The header is `sample_name`, `key`, `value_type`, and `value`. Value types
/// are `text`, `integer`, `float`, and `boolean`. Fields containing tabs, newlines, or
/// quotes use CSV-style quoting with a tab delimiter.
pub fn export_sample_metadata(
    connection: &GraphConnection,
    sample_name: Option<&str>,
    filename: &Path,
    history_ref: Option<&str>,
    keys: Option<&[String]>,
    json: bool,
) -> Result<()> {
    let mut selector = SampleMetadata::select(connection).with_ref(history_ref);
    if let Some(sample_name) = sample_name {
        if Sample::select(connection)
            .with_ref(history_ref)
            .name(sample_name)
            .load()?
            .is_empty()
        {
            bail!("Sample not found: {sample_name}");
        }
        selector = selector.sample_name(sample_name);
    }
    let mut metadata = selector.load()?;
    if let Some(keys) = keys {
        metadata.retain(|entry| keys.contains(&entry.key));
    }
    metadata.sort_by(|left, right| {
        left.sample_name
            .cmp(&right.sample_name)
            .then_with(|| left.key.cmp(&right.key))
    });

    // Load and validate the selection before opening the output so an invalid
    // sample or revision cannot truncate an existing export.
    if json {
        let mut samples: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
        for entry in metadata {
            let value = match entry.value {
                MetadataValue::Text(value) => Value::from(value),
                MetadataValue::Integer(value) => Value::from(value),
                MetadataValue::Float(value) => Value::from(value),
                MetadataValue::Boolean(value) => Value::from(value),
            };
            samples
                .entry(entry.sample_name)
                .or_default()
                .insert(entry.key, value);
        }
        let samples: Vec<_> = samples
            .into_iter()
            .map(|(name, metadata)| JsonSample { name, metadata })
            .collect();
        fs::write(filename, serde_json::to_vec_pretty(&samples)?)?;
        return Ok(());
    }
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(filename)?;
    writer.write_record(["sample_name", "key", "value_type", "value"])?;
    for entry in metadata {
        let (value_type, value) = match entry.value {
            MetadataValue::Text(text) => ("text", text),
            MetadataValue::Integer(integer) => ("integer", integer.to_string()),
            MetadataValue::Float(float) => ("float", float.to_string()),
            MetadataValue::Boolean(boolean) => ("boolean", boolean.to_string()),
        };
        writer.write_record([entry.sample_name.as_str(), &entry.key, value_type, &value])?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Serialize)]
struct JsonSample {
    name: String,
    metadata: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use std::fs;

    use csv::ReaderBuilder;
    use gen_models::{
        history::dolt::commit_all,
        sample::{NewSample, Sample},
        sample_metadata::{MetadataValue, SampleMetadata},
    };
    use tempfile::tempdir;

    use crate::{
        exports::sample_metadata::export_sample_metadata, test_helpers::setup_gen_on_disk,
    };

    #[test]
    fn test_export_sample_metadata_types_escaping_and_filtering() {
        let context = setup_gen_on_disk();
        let connection = context.graph().conn();
        for name in ["sample", "other", "empty"] {
            Sample::create(
                connection,
                NewSample {
                    name,
                    ..Default::default()
                },
            )
            .expect("should create sample");
        }
        let special = "tabs\tnewlines\nquotes\" and Unicode: é";
        for (sample_name, key, value) in [
            ("sample", "text", MetadataValue::Text(special.to_string())),
            ("sample", "integer", MetadataValue::Integer(i64::MAX)),
            ("sample", "minimum", MetadataValue::Integer(i64::MIN)),
            ("sample", "float", MetadataValue::Float(1.25)),
            ("sample", "boolean_false", MetadataValue::Boolean(false)),
            ("sample", "boolean_true", MetadataValue::Boolean(true)),
            (
                "sample",
                "numeric_text",
                MetadataValue::Text("1".to_string()),
            ),
            ("sample", "empty_text", MetadataValue::Text(String::new())),
            ("other", "key\t\"\n", MetadataValue::Float(1.0)),
        ] {
            SampleMetadata::create(connection, sample_name, key, &value)
                .expect("should create metadata");
        }
        let directory = tempdir().expect("should create output directory");
        let output = directory.path().join("metadata.tsv");
        export_sample_metadata(connection, None, &output, None, None, false)
            .expect("should export all metadata");
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(&output)
            .expect("should open TSV");
        assert_eq!(
            reader
                .headers()
                .expect("should read header")
                .iter()
                .collect::<Vec<_>>(),
            vec!["sample_name", "key", "value_type", "value"]
        );
        let records = reader
            .records()
            .map(|record| {
                record
                    .expect("should parse TSV record")
                    .iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            records,
            vec![
                vec!["other", "key\t\"\n", "float", "1"],
                vec!["sample", "boolean_false", "boolean", "false"],
                vec!["sample", "boolean_true", "boolean", "true"],
                vec!["sample", "empty_text", "text", ""],
                vec!["sample", "float", "float", "1.25"],
                vec!["sample", "integer", "integer", "9223372036854775807"],
                vec!["sample", "minimum", "integer", "-9223372036854775808"],
                vec!["sample", "numeric_text", "text", "1"],
                vec!["sample", "text", "text", special],
            ]
        );
        export_sample_metadata(connection, Some("other"), &output, None, None, false)
            .expect("should export selected sample");
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(&output)
            .expect("should open filtered TSV");
        assert_eq!(reader.records().count(), 1);
        export_sample_metadata(connection, Some("empty"), &output, None, None, false)
            .expect("should export sample without metadata");
        let header = "sample_name\tkey\tvalue_type\tvalue\n";
        assert_eq!(
            fs::read_to_string(&output).expect("should read empty export"),
            header
        );
        assert!(
            export_sample_metadata(connection, Some("missing"), &output, None, None, false)
                .is_err()
        );
        assert_eq!(
            fs::read_to_string(&output).expect("should preserve output"),
            header
        );
        assert!(
            export_sample_metadata(connection, None, directory.path(), None, None, false).is_err()
        );
    }

    #[test]
    fn test_export_sample_metadata_keys() {
        let context = setup_gen_on_disk();
        let connection = context.graph().conn();
        for name in ["first", "second"] {
            Sample::create(
                connection,
                NewSample {
                    name,
                    ..Default::default()
                },
            )
            .expect("should create sample");
            for key in ["a", "b", "c"] {
                SampleMetadata::create(connection, name, key, &MetadataValue::Integer(1))
                    .expect("should create metadata");
            }
        }
        let directory = tempdir().expect("should create output directory");
        let output = directory.path().join("metadata.tsv");
        let keys = [
            "c".to_string(),
            "a".to_string(),
            "a".to_string(),
            "missing".to_string(),
        ];
        export_sample_metadata(connection, None, &output, None, Some(&keys), false)
            .expect("should export selected keys across samples");
        assert_eq!(
            fs::read_to_string(&output).expect("should read export"),
            "sample_name\tkey\tvalue_type\tvalue\nfirst\ta\tinteger\t1\nfirst\tc\tinteger\t1\nsecond\ta\tinteger\t1\nsecond\tc\tinteger\t1\n"
        );
        export_sample_metadata(
            connection,
            Some("second"),
            &output,
            None,
            Some(&keys),
            false,
        )
        .expect("should combine sample and key filters");
        assert_eq!(
            fs::read_to_string(&output).expect("should read export"),
            "sample_name\tkey\tvalue_type\tvalue\nsecond\ta\tinteger\t1\nsecond\tc\tinteger\t1\n"
        );
        export_sample_metadata(
            connection,
            None,
            &output,
            None,
            Some(&["missing".to_string()]),
            false,
        )
        .expect("should export header for unmatched keys");
        assert_eq!(
            fs::read_to_string(&output).expect("should read export"),
            "sample_name\tkey\tvalue_type\tvalue\n"
        );
    }

    #[test]
    fn test_export_sample_metadata_history_and_empty_repository() {
        let context = setup_gen_on_disk();
        let connection = context.graph().conn();
        let directory = tempdir().expect("should create output directory");
        let output = directory.path().join("metadata.tsv");
        export_sample_metadata(connection, None, &output, None, None, false)
            .expect("should export empty repository");
        assert_eq!(
            fs::read_to_string(&output).expect("should read export"),
            "sample_name\tkey\tvalue_type\tvalue\n"
        );
        Sample::create(
            connection,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        SampleMetadata::create(connection, "sample", "score", &MetadataValue::Integer(1))
            .expect("should create metadata");
        let original = commit_all(connection, "original metadata").expect("should commit metadata");
        SampleMetadata::upsert(connection, "sample", "score", &MetadataValue::Float(2.5))
            .expect("should update metadata");
        commit_all(connection, "updated metadata").expect("should commit updated metadata");
        export_sample_metadata(
            connection,
            Some("sample"),
            &output,
            Some(&original.to_string()),
            None,
            false,
        )
        .expect("should export historical metadata");
        assert_eq!(
            fs::read_to_string(&output).expect("should read historical export"),
            "sample_name\tkey\tvalue_type\tvalue\nsample\tscore\tinteger\t1\n"
        );
        assert!(
            export_sample_metadata(
                connection,
                None,
                &output,
                Some("missing-revision"),
                None,
                false
            )
            .is_err()
        );
        export_sample_metadata(connection, Some("sample"), &output, None, None, false)
            .expect("should export current metadata");
        assert_eq!(
            fs::read_to_string(&output).expect("should read current export"),
            "sample_name\tkey\tvalue_type\tvalue\nsample\tscore\tfloat\t2.5\n"
        );
    }
}
