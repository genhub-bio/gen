use std::path::Path;

use anyhow::{Context, Result, bail};
use csv::ReaderBuilder;
use gen_models::{
    db::GraphConnection,
    operations::{OperationFile, OperationInfo, OperationSummary},
    sample::Sample,
    sample_metadata::{MetadataValue, SampleMetadata},
};

/// Import exported TSV metadata into existing samples, replacing matching keys.
pub fn import_sample_metadata(
    connection: &GraphConnection,
    filename: &Path,
) -> Result<OperationSummary> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_path(filename)?;
    if reader.headers()?.iter().collect::<Vec<_>>() != ["sample_name", "key", "value_type", "value"]
    {
        bail!("metadata TSV header must be sample_name, key, value_type, value");
    }
    // Validate the entire input before writing so a bad row cannot leave a partial import.
    let mut entries = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let row = index + 2;
        let record = record.with_context(|| format!("invalid metadata TSV row {row}"))?;
        let value = match &record[2] {
            "text" => MetadataValue::Text(record[3].to_string()),
            "integer" => MetadataValue::Integer(
                record[3]
                    .parse()
                    .with_context(|| format!("invalid integer at row {row}"))?,
            ),
            "float" => {
                let value: f64 = record[3]
                    .parse()
                    .with_context(|| format!("invalid float at row {row}"))?;
                if !value.is_finite() {
                    bail!("metadata float must be finite at row {row}");
                }
                MetadataValue::Float(value)
            }
            "boolean" => MetadataValue::Boolean(record[3].parse().with_context(|| {
                format!("invalid boolean at row {row}; expected true or false")
            })?),
            value_type => bail!("unknown metadata value type {value_type:?} at row {row}"),
        };
        if Sample::select(connection)
            .name(&record[0])
            .load()?
            .is_empty()
        {
            bail!("sample not found at row {row}: {}", &record[0]);
        }
        entries.push((record[0].to_string(), record[1].to_string(), value));
    }
    for (sample_name, key, value) in &entries {
        SampleMetadata::upsert(connection, sample_name, key, value)?;
    }
    Ok(OperationSummary::new(
        OperationInfo {
            files: vec![OperationFile::new(filename.to_string_lossy().into_owned())],
            description: "import sample metadata".to_string(),
        },
        format!("Imported {} sample metadata rows", entries.len()),
    ))
}
