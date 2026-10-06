use gen_core::Sha256Hash;
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{ModelSelect, db::GraphConnection};

/// A typed sample metadata value.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub enum MetadataValue {
    /// A text value.
    Text(String),
    /// A signed 64-bit integer.
    Integer(i64),
    /// A finite double-precision value.
    Float(f64),
    /// A boolean value.
    Boolean(bool),
}

/// One metadata value associated with a sample and key.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, ModelSelect)]
#[model_select(
    table = "sample_metadata",
    select = "hash, sample_name, key, value_text, value_integer, value_float, value_boolean",
    from_row = SampleMetadata::from_row
)]
pub struct SampleMetadata {
    /// Content hash including the sample, key, value type, and value.
    #[model_select(primary_key)]
    pub hash: Sha256Hash,
    /// The sample's primary key in the samples table.
    pub sample_name: String,
    /// The metadata key, unique within the sample.
    pub key: String,
    /// Exactly one typed metadata value.
    #[model_select(skip)]
    pub value: MetadataValue,
}

/// Errors storing sample metadata.
#[derive(Debug, Error, PartialEq)]
pub enum SampleMetadataError {
    /// The database rejected the operation.
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
    /// Non-finite values cannot be persisted reliably in SQLite.
    #[error("sample metadata float values must be finite")]
    NonFiniteFloat,
}

impl SampleMetadata {
    /// Store a new value; an existing key for this sample causes a uniqueness error.
    pub fn create(
        conn: &GraphConnection,
        sample_name: &str,
        key: &str,
        value: &MetadataValue,
    ) -> Result<Self, SampleMetadataError> {
        Self::store(conn, sample_name, key, value, false)
    }

    /// Store a value, replacing the value and content hash for an existing sample key.
    pub fn upsert(
        conn: &GraphConnection,
        sample_name: &str,
        key: &str,
        value: &MetadataValue,
    ) -> Result<Self, SampleMetadataError> {
        Self::store(conn, sample_name, key, value, true)
    }

    fn store(
        conn: &GraphConnection,
        sample_name: &str,
        key: &str,
        value: &MetadataValue,
        replace: bool,
    ) -> Result<Self, SampleMetadataError> {
        let value = match value {
            MetadataValue::Float(float) if !float.is_finite() => {
                return Err(SampleMetadataError::NonFiniteFloat);
            }
            MetadataValue::Float(float) if *float == 0.0 => MetadataValue::Float(0.0),
            value => value.clone(),
        };
        let (text, integer, float, boolean) = match &value {
            MetadataValue::Text(text) => (Some(text.as_str()), None, None, None),
            MetadataValue::Integer(integer) => (None, Some(*integer), None, None),
            MetadataValue::Float(float) => (None, None, Some(*float), None),
            MetadataValue::Boolean(boolean) => (None, None, None, Some(*boolean)),
        };

        // Length prefixes and a type tag keep field boundaries and numeric types distinct.
        let mut hasher = Sha256::new();
        for field in [sample_name, key] {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field.as_bytes());
        }
        match &value {
            MetadataValue::Text(text) => {
                hasher.update([0]);
                hasher.update(text.as_bytes());
            }
            MetadataValue::Integer(integer) => {
                hasher.update([1]);
                hasher.update(integer.to_be_bytes());
            }
            MetadataValue::Float(float) => {
                hasher.update([2]);
                hasher.update(float.to_bits().to_be_bytes());
            }
            MetadataValue::Boolean(boolean) => {
                hasher.update([3]);
                hasher.update([u8::from(*boolean)]);
            }
        }
        let hash = Sha256Hash(hasher.finalize().into());
        let mut sql = String::from(
            "INSERT INTO sample_metadata
             (hash, sample_name, key, value_text, value_integer, value_float, value_boolean)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        );
        if replace {
            sql.push_str(
                " ON CONFLICT(sample_name, key) DO UPDATE SET
                  hash = excluded.hash, value_text = excluded.value_text,
                  value_integer = excluded.value_integer, value_float = excluded.value_float,
                  value_boolean = excluded.value_boolean",
            );
        }
        conn.execute(
            &sql,
            params![hash, sample_name, key, text, integer, float, boolean],
        )?;
        Ok(Self {
            hash,
            sample_name: sample_name.to_string(),
            key: key.to_string(),
            value,
        })
    }

    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        let values = (
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<f64>>(5)?,
            row.get::<_, Option<i64>>(6)?,
        );
        let value = match values {
            (Some(text), None, None, None) => MetadataValue::Text(text),
            (None, Some(integer), None, None) => MetadataValue::Integer(integer),
            (None, None, Some(float), None) if float.is_finite() => MetadataValue::Float(float),
            (None, None, None, Some(boolean @ (0 | 1))) => MetadataValue::Boolean(boolean == 1),
            _ => return Err(rusqlite::Error::InvalidQuery),
        };
        Ok(Self {
            hash: row.get(0)?,
            sample_name: row.get(1)?,
            key: row.get(2)?,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::params;

    use crate::{
        sample::{NewSample, Sample},
        sample_metadata::{MetadataValue, SampleMetadata, SampleMetadataError},
        test_helpers::get_connection,
    };

    #[test]
    fn test_sample_metadata_upsert_replaces_type_and_hash() {
        let conn = get_connection(None).expect("should open database");
        Sample::create(
            &conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        let original = SampleMetadata::upsert(
            &conn,
            "sample",
            "Score",
            &MetadataValue::Text("pending".to_string()),
        )
        .expect("should insert metadata");
        let updated =
            SampleMetadata::upsert(&conn, "sample", "Score", &MetadataValue::Float(90.68))
                .expect("should update metadata");
        assert_ne!(original.hash, updated.hash);
        assert_eq!(
            SampleMetadata::select(&conn)
                .load()
                .expect("should load metadata"),
            vec![updated.clone()],
        );
        let repeated =
            SampleMetadata::upsert(&conn, "sample", "Score", &MetadataValue::Float(90.68))
                .expect("should repeat update");
        assert_eq!(updated, repeated);
    }

    #[test]
    fn test_sample_metadata_boolean_hash_and_type_changes() {
        let conn = get_connection(None).expect("should open database");
        Sample::create(
            &conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        let mut hashes = Vec::new();
        for value in [
            MetadataValue::Integer(1),
            MetadataValue::Boolean(true),
            MetadataValue::Boolean(false),
            MetadataValue::Text("false".to_string()),
            MetadataValue::Float(0.0),
        ] {
            let metadata = SampleMetadata::upsert(&conn, "sample", "flag", &value)
                .expect("should replace metadata type");
            assert!(
                !hashes.contains(&metadata.hash),
                "typed values should have distinct hashes"
            );
            hashes.push(metadata.hash);
            assert_eq!(
                SampleMetadata::select(&conn)
                    .load()
                    .expect("should load metadata"),
                vec![metadata]
            );
        }
    }

    #[test]
    fn test_sample_metadata_round_trip_and_uniqueness() {
        let conn = get_connection(None).expect("should open database");
        Sample::create(
            &conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        for (key, value) in [
            ("text", MetadataValue::Text("1".to_string())),
            ("integer", MetadataValue::Integer(i64::MAX)),
            ("float", MetadataValue::Float(1.25)),
            ("true", MetadataValue::Boolean(true)),
            ("false", MetadataValue::Boolean(false)),
        ] {
            let metadata = SampleMetadata::create(&conn, "sample", key, &value)
                .expect("should create metadata");
            let loaded = SampleMetadata::select(&conn)
                .sample_name("sample")
                .key(key)
                .load()
                .expect("should load metadata");
            assert_eq!(loaded, vec![metadata]);
            assert!(
                SampleMetadata::create(&conn, "sample", key, &MetadataValue::Integer(2)).is_err()
            );
        }
        assert!(
            SampleMetadata::create(&conn, "missing", "key", &MetadataValue::Integer(1)).is_err()
        );
        for float in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                SampleMetadata::create(&conn, "sample", "invalid", &MetadataValue::Float(float)),
                Err(SampleMetadataError::NonFiniteFloat)
            ));
        }
    }

    #[test]
    fn test_sample_metadata_database_value_constraints() {
        let conn = get_connection(None).expect("should open database");
        Sample::create(
            &conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        for values in [
            "NULL, NULL, NULL, NULL",
            "NULL, NULL, NULL, 2",
            "NULL, NULL, NULL, -1",
            "NULL, NULL, NULL, 0.5",
            "NULL, NULL, NULL, 'invalid'",
            "NULL, 1, NULL, 1",
            "'text', 1, NULL, NULL",
            "NULL, 'invalid', NULL, NULL",
            "NULL, 1.5, NULL, NULL",
            "NULL, NULL, 'invalid', NULL",
        ] {
            let sql = format!(
                "INSERT INTO sample_metadata (hash, sample_name, key, value_text, value_integer, value_float, value_boolean)
                 VALUES (?1, 'sample', 'key', {values})"
            );
            assert!(conn.execute(&sql, params![vec![0u8; 32]]).is_err());
        }
    }

    #[test]
    fn test_sample_metadata_hash_encoding() {
        let conn = get_connection(None).expect("should open database");
        for name in ["a", "ab"] {
            Sample::create(
                &conn,
                NewSample {
                    name,
                    ..Default::default()
                },
            )
            .expect("should create sample");
        }
        let text = SampleMetadata::create(&conn, "a", "bc", &MetadataValue::Text("1".to_string()))
            .expect("should create text");
        SampleMetadata::select(&conn)
            .delete_by_ids([text.hash])
            .expect("should delete metadata");
        let integer = SampleMetadata::create(&conn, "a", "bc", &MetadataValue::Integer(1))
            .expect("should create integer");
        SampleMetadata::select(&conn)
            .delete_by_ids([integer.hash])
            .expect("should delete metadata");
        let float = SampleMetadata::create(&conn, "a", "bc", &MetadataValue::Float(1.0))
            .expect("should create float");
        assert_ne!(text.hash, integer.hash);
        assert_ne!(integer.hash, float.hash);
        let other = SampleMetadata::create(&conn, "ab", "c", &MetadataValue::Float(1.0))
            .expect("should create metadata for sample 'ab' with key 'c'");
        assert_ne!(
            float.hash, other.hash,
            "sample/key pairs ('a', 'bc') and ('ab', 'c') should hash differently even though both concatenate to 'abc'"
        );
        let zero = SampleMetadata::create(&conn, "a", "zero", &MetadataValue::Float(-0.0))
            .expect("should create zero");
        SampleMetadata::select(&conn)
            .delete_by_ids([zero.hash])
            .expect("should delete zero");
        let positive_zero = SampleMetadata::create(&conn, "a", "zero", &MetadataValue::Float(0.0))
            .expect("should recreate zero");
        assert_eq!(zero.hash, positive_zero.hash);
    }
}
