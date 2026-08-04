use std::{convert::TryFrom, fmt, hash::Hash};

use rand::Rng;
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_128;

pub mod change;
pub mod config;
pub mod errors;
#[allow(clippy::all)]
pub mod generated;
pub mod graph;
pub mod path;
pub mod range;
pub mod region;
pub mod strand;
pub mod traits;

pub use change::BlockGroupChange;
pub use config::Workspace;
use errors::HashError;
pub use generated::gen_core_capnp;
pub use graph::{GenGraph, GraphEdge, GraphNode, GraphNodePosition, GraphNodeSlice};
pub use path::PathBlock;
#[cfg(feature = "python-bindings")]
use pyo3::pyclass;
pub use strand::Strand;

pub static NO_CHROMOSOME_INDEX: i64 = -1;
pub static PRESERVE_EDIT_SITE_CHROMOSOME_INDEX: i64 = -2;
pub static INDETERMINATE_CHROMOSOME_INDEX: i64 = -3;
/// Number of bytes in a native Dolt commit hash.
pub const DOLT_HASH_SIZE: usize = 20;
/// Number of bytes in a Gen domain identifier.
pub const HASH_ID_SIZE: usize = 16;
/// Number of bytes in a SHA-256 content hash.
pub const SHA256_HASH_SIZE: usize = 32;

// these are just the written out hex from the inserted values from sql migrations
pub const PATH_START_NODE_ID: HashId = HashId([
    0x84, 0xd6, 0xad, 0xbd, 0x53, 0x95, 0x28, 0x19, 0x33, 0xfe, 0x41, 0xe8, 0x77, 0xd3, 0xa7, 0xf0,
]);
pub const PATH_END_NODE_ID: HashId = HashId([
    0x1c, 0x7d, 0xfc, 0x64, 0x97, 0x7b, 0x08, 0x38, 0xaf, 0x07, 0x62, 0xd7, 0x33, 0x3d, 0xcb, 0x64,
]);
pub const PATH_START_SEQUENCE_HASH: Sha256Hash = Sha256Hash([
    0x84, 0xd6, 0xad, 0xbd, 0x53, 0x95, 0x28, 0x19, 0x33, 0xfe, 0x41, 0xe8, 0x77, 0xd3, 0xa7, 0xf0,
    0x2a, 0x3b, 0x19, 0x90, 0xa6, 0x5b, 0xe1, 0x90, 0x1b, 0x2c, 0x91, 0xfc, 0x68, 0x5e, 0x08, 0x3b,
]);
pub const PATH_END_SEQUENCE_HASH: Sha256Hash = Sha256Hash([
    0x1c, 0x7d, 0xfc, 0x64, 0x97, 0x7b, 0x08, 0x38, 0xaf, 0x07, 0x62, 0xd7, 0x33, 0x3d, 0xcb, 0x64,
    0xc1, 0x75, 0xb1, 0x5e, 0x65, 0xa7, 0x00, 0x99, 0xec, 0x38, 0xf4, 0x6b, 0xf1, 0xa1, 0x5e, 0xa3,
]);

pub fn is_terminal(node_id: HashId) -> bool {
    is_start_node(node_id) || is_end_node(node_id)
}

pub fn is_start_node(node_id: HashId) -> bool {
    node_id == PATH_START_NODE_ID
}

pub fn is_end_node(node_id: HashId) -> bool {
    node_id == PATH_END_NODE_ID
}

#[derive(
    Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
pub struct CommitRef(pub String);

#[derive(
    Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
pub struct BranchName(pub String);

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DoltHashId(pub [u8; DOLT_HASH_SIZE]);

impl serde::Serialize for DoltHashId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for DoltHashId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct DoltHashIdVisitor;

        impl<'de> serde::de::Visitor<'de> for DoltHashIdVisitor {
            type Value = DoltHashId;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 40-character hex string")
            }

            fn visit_str<E>(self, value: &str) -> Result<DoltHashId, E>
            where
                E: serde::de::Error,
            {
                DoltHashId::try_from(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(DoltHashIdVisitor)
    }
}

impl fmt::Display for DoltHashId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for DoltHashId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "DoltHashId({self})")
    }
}

impl TryFrom<String> for DoltHashId {
    type Error = HashError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::try_from(value.as_str())
    }
}

impl TryFrom<&str> for DoltHashId {
    type Error = HashError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let bytes = hex::decode(value)?;
        let actual = bytes.len();
        let array = bytes.try_into().map_err(|_| HashError::InvalidLength {
            expected: DOLT_HASH_SIZE,
            actual,
        })?;
        Ok(Self(array))
    }
}

impl TryFrom<&[u8]> for DoltHashId {
    type Error = HashError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        let actual = value.len();
        let array = value.try_into().map_err(|_| HashError::InvalidLength {
            expected: DOLT_HASH_SIZE,
            actual,
        })?;
        Ok(Self(array))
    }
}

#[cfg_attr(feature = "python-bindings", pyclass)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
/// A 128-bit Gen domain identifier, normally derived with XXH3-128.
pub struct HashId(pub [u8; HASH_ID_SIZE]);

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
/// A 256-bit SHA-256 content hash used for sequences and file checksums.
pub struct Sha256Hash(pub [u8; SHA256_HASH_SIZE]);

use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};

macro_rules! impl_fixed_hash {
    ($hash_type:ident, $size:expr) => {
        impl serde::Serialize for $hash_type {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&self.to_string())
            }
        }

        impl<'de> serde::Deserialize<'de> for $hash_type {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct HashVisitor;

                impl<'de> serde::de::Visitor<'de> for HashVisitor {
                    type Value = $hash_type;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        write!(formatter, "a {}-character hex string", $size * 2)
                    }

                    fn visit_str<E>(self, value: &str) -> Result<$hash_type, E>
                    where
                        E: serde::de::Error,
                    {
                        $hash_type::try_from(value).map_err(E::custom)
                    }
                }

                deserializer.deserialize_str(HashVisitor)
            }
        }

        impl fmt::Display for $hash_type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                for byte in &self.0 {
                    write!(formatter, "{byte:02x}")?;
                }
                Ok(())
            }
        }

        impl fmt::Debug for $hash_type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}({self})", stringify!($hash_type))
            }
        }

        impl TryFrom<String> for $hash_type {
            type Error = HashError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::try_from(value.as_str())
            }
        }

        impl TryFrom<&str> for $hash_type {
            type Error = HashError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                let bytes = hex::decode(value)?;
                let actual = bytes.len();
                let array = bytes.try_into().map_err(|_| HashError::InvalidLength {
                    expected: $size,
                    actual,
                })?;
                Ok(Self(array))
            }
        }

        impl TryFrom<&[u8]> for $hash_type {
            type Error = HashError;

            fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
                let actual = value.len();
                let array = value.try_into().map_err(|_| HashError::InvalidLength {
                    expected: $size,
                    actual,
                })?;
                Ok(Self(array))
            }
        }

        impl PartialEq<[u8; $size]> for $hash_type {
            fn eq(&self, other: &[u8; $size]) -> bool {
                &self.0 == other
            }
        }

        impl PartialEq<$hash_type> for [u8; $size] {
            fn eq(&self, other: &$hash_type) -> bool {
                self == &other.0
            }
        }

        impl FromSql for $hash_type {
            fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
                match value {
                    ValueRef::Blob(bytes) => {
                        Self::try_from(bytes).map_err(|error| FromSqlError::Other(Box::new(error)))
                    }
                    _ => Err(FromSqlError::InvalidType),
                }
            }
        }

        impl ToSql for $hash_type {
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                Ok(ToSqlOutput::from(self.0.as_ref()))
            }
        }

        impl From<$hash_type> for Value {
            fn from(hash: $hash_type) -> Self {
                Value::Blob(hash.0.to_vec())
            }
        }
    };
}

impl_fixed_hash!(HashId, HASH_ID_SIZE);
impl_fixed_hash!(Sha256Hash, SHA256_HASH_SIZE);

impl FromSql for DoltHashId {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value {
            ValueRef::Text(text) => {
                let text = core::str::from_utf8(text).map_err(|error| {
                    FromSqlError::Other(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        error,
                    )))
                })?;
                Self::try_from(text).map_err(|error| FromSqlError::Other(Box::new(error)))
            }
            ValueRef::Blob(bytes) => {
                Self::try_from(bytes).map_err(|error| FromSqlError::Other(Box::new(error)))
            }
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for DoltHashId {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Owned(rusqlite::types::Value::Text(
            self.to_string(),
        )))
    }
}

use rusqlite::types::Value;

impl From<uuid::Uuid> for HashId {
    fn from(uuid: uuid::Uuid) -> Self {
        Self(*uuid.as_bytes())
    }
}

impl HashId {
    /// Left-pads a hexadecimal fixture value to the full identifier width.
    pub fn pad_str<T: ToString>(input: T) -> Self {
        let s = input.to_string();
        let hex = format!("{s:0>32}");
        let bytes = hex::decode(hex).expect("invalid hex string");
        HashId(bytes.try_into().expect("should contain 16 bytes"))
    }

    /// Derives a stable domain identifier from a string using XXH3-128.
    pub fn convert_str(s: &str) -> Self {
        HashId(calculate_hash(s))
    }

    pub fn random_str() -> Self {
        let mut rng = rand::rng();
        let mut random_bytes = [0u8; HASH_ID_SIZE];
        rng.fill_bytes(&mut random_bytes);
        Self(random_bytes)
    }

    pub fn uuid7() -> Self {
        uuid::Uuid::now_v7().into()
    }

    // this is a hack for the library code, which we have a hack for chromosome_index to be the edge_id
    pub fn extract_digits(&self) -> i64 {
        let hex = format!("{self}");
        let digits: String = hex
            .chars()
            .map(|c| {
                if c.is_ascii_digit() {
                    c
                } else {
                    // ascii digits start at 48, alphabet at 97 so subtract 49 to get char digits. We add 1 to make 'a' = 1
                    // to prevent leading zeros
                    (c.to_ascii_lowercase() as u8 - 49 + 1) as char
                }
            })
            .take(15) // limit to first 15 digits if you want
            .collect();

        digits.parse().unwrap_or(0)
    }

    pub fn starts_with(&self, prefix: &str) -> bool {
        if prefix.len() > HASH_ID_SIZE * 2 || prefix.is_empty() {
            return false;
        }

        let end_byte = prefix.len() / 2 + prefix.len() % 2;
        let encoded = hex::encode(&self.0[..end_byte]);
        encoded.starts_with(prefix)
    }
}

impl Sha256Hash {
    /// Calculates the SHA-256 digest of a string.
    pub fn convert_str(value: &str) -> Self {
        Self(calculate_sha256(value))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeIntervalBlock {
    pub node_id: HashId,
    pub start: i64,
    pub end: i64,
    pub sequence_start: i64,
    pub sequence_end: i64,
    pub strand: Strand,
}

/// Calculates a 128-bit XXH3 Gen domain identifier.
pub fn calculate_hash(value: &str) -> [u8; HASH_ID_SIZE] {
    xxh3_128(value.as_bytes()).to_le_bytes()
}

/// Calculates a SHA-256 content hash.
pub fn calculate_sha256(value: &str) -> [u8; SHA256_HASH_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(value);
    let result = hasher.finalize();
    result.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_id_uses_xxh3_128() {
        let result = HashId::convert_str("a test");
        assert_eq!(calculate_hash("a test"), result);
        assert_eq!(result.to_string(), "88f0cfc88597fce07ca40a6af6070b00");
        assert_eq!(std::mem::size_of::<HashId>(), HASH_ID_SIZE);
        assert_ne!(result, HashId::convert_str("another test"));
    }

    #[test]
    fn test_calculate_sha256() {
        let result: Sha256Hash = "a82639b6f8c3a6e536d8cc562c3b86ff4b012c84ab230c1e5be649aa9ad26d21"
            .try_into()
            .unwrap();
        assert_eq!(calculate_sha256("a test"), result);
        assert_eq!(std::mem::size_of::<Sha256Hash>(), SHA256_HASH_SIZE);
    }

    #[cfg(test)]
    mod hashid {
        use super::*;

        #[test]
        fn test_starts_with() {
            let hash = HashId::convert_str("a test");
            let encoded = hash.to_string();
            assert!(hash.starts_with(&encoded[..4]));
            assert!(hash.starts_with(&encoded));
            assert!(!hash.starts_with("ffff"));
            assert!(!hash.starts_with(&format!("{encoded}0")));
            assert!(!hash.starts_with(""));
            assert!(!hash.starts_with(&format!("{encoded}9")));
        }
    }

    mod dolt_hash_id {
        use super::*;

        #[test]
        fn test_round_trips_dolt_hash_hex() {
            let hex = "8a7f64798afa5f3f66b37357717c3e57fa7cdf06";
            let hash = DoltHashId::try_from(hex).unwrap();

            assert_eq!(hash.to_string(), hex);
        }

        #[test]
        fn test_rejects_non_dolt_hash_length() {
            let error = DoltHashId::try_from(
                "0000000000000000000000008a7f64798afa5f3f66b37357717c3e57fa7cdf06",
            )
            .unwrap_err();

            assert_eq!(
                error,
                HashError::InvalidLength {
                    expected: DOLT_HASH_SIZE,
                    actual: 32,
                }
            );
        }
    }
}
