use gen_core::Sha256Hash;
use rusqlite::{Row, params, types::Type};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{ModelSelect, db::GraphConnection};

/// A typed sample metadata value.
///
/// Postcard persists enum variant indices. Preserve this order and append new variants.
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
    select = "hash, sample_name, key, value",
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
    /// The metadata value could not be serialized.
    #[error(transparent)]
    Serialization(#[from] postcard::Error),
    /// Metadata floats must remain valid in all supported export formats.
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
        let encoded_value = postcard::to_stdvec(&value)?;

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
             (hash, sample_name, key, value)
             VALUES (?1, ?2, ?3, ?4)",
        );
        if replace {
            sql.push_str(
                " ON CONFLICT(sample_name, key) DO UPDATE SET
                  hash = excluded.hash, value = excluded.value",
            );
        }
        conn.execute(&sql, params![hash, sample_name, key, encoded_value])?;
        Ok(Self {
            hash,
            sample_name: sample_name.to_string(),
            key: key.to_string(),
            value,
        })
    }

    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        let encoded_value: Vec<u8> = row.get(3)?;
        let (value, remaining) = postcard::take_from_bytes::<MetadataValue>(&encoded_value)
            .map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(3, Type::Blob, Box::new(error))
            })?;
        if !remaining.is_empty() {
            return Err(rusqlite::Error::InvalidQuery);
        }
        if matches!(&value, MetadataValue::Float(float) if !float.is_finite()) {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                3,
                Type::Blob,
                Box::new(SampleMetadataError::NonFiniteFloat),
            ));
        }
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
        for value in ["NULL", "'invalid'", "1", "1.5"] {
            let sql = format!(
                "INSERT INTO sample_metadata (hash, sample_name, key, value)
                 VALUES (?1, 'sample', 'key', {value})"
            );
            assert!(conn.execute(&sql, params![vec![0u8; 32]]).is_err());
        }
    }

    #[test]
    fn test_sample_metadata_rejects_invalid_encoded_values() {
        let conn = get_connection(None).expect("should open database");
        Sample::create(
            &conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .expect("should create sample");
        let mut trailing_bytes =
            postcard::to_stdvec(&MetadataValue::Boolean(true)).expect("should encode boolean");
        trailing_bytes.push(0);
        let mut invalid_values = vec![vec![], vec![255], vec![3, 2], trailing_bytes];
        for float in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            invalid_values.push(
                postcard::to_stdvec(&MetadataValue::Float(float))
                    .expect("should encode non-finite float for corruption test"),
            );
        }
        for encoded_value in invalid_values {
            conn.execute(
                "INSERT INTO sample_metadata (hash, sample_name, key, value)
                 VALUES (?1, 'sample', 'key', ?2)",
                params![vec![0u8; 32], encoded_value],
            )
            .expect("should insert corrupt metadata");
            assert!(SampleMetadata::select(&conn).load().is_err());
            conn.execute("DELETE FROM sample_metadata", [])
                .expect("should remove corrupt metadata");
        }
    }

    #[test]
    fn test_sample_metadata_persisted_encoding() {
        // These bytes define the stored format independently of Rust enum refactors.
        for (value, encoded_value) in [
            (MetadataValue::Text("x".to_string()), vec![0, 1, b'x']),
            (MetadataValue::Integer(-1), vec![1, 1]),
            (
                MetadataValue::Float(1.0),
                vec![2, 0, 0, 0, 0, 0, 0, 240, 63],
            ),
            (MetadataValue::Boolean(false), vec![3, 0]),
            (MetadataValue::Boolean(true), vec![3, 1]),
        ] {
            assert_eq!(
                postcard::to_stdvec(&value).expect("should encode value"),
                encoded_value
            );
            assert_eq!(
                postcard::from_bytes::<MetadataValue>(&encoded_value)
                    .expect("should decode stored value"),
                value
            );
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
