use std::{convert::TryFrom, fmt, hash::Hash};

use hex::FromHex;
use rand::RngCore;
use sha2::{Digest, Sha256};

pub mod config;
pub mod errors;
#[allow(clippy::all)]
pub mod generated;
pub mod path;
pub mod range;
pub mod strand;
pub mod traits;

pub use generated::gen_core_capnp;
pub use path::PathBlock;
pub use strand::Strand;

pub static NO_CHROMOSOME_INDEX: i64 = -1;

// these are just the written out hex from the inserted values from sql migrations
pub const PATH_START_NODE_ID: HashId = HashId([
    0x84, 0xd6, 0xad, 0xbd, 0x53, 0x95, 0x28, 0x19, 0x33, 0xfe, 0x41, 0xe8, 0x77, 0xd3, 0xa7, 0xf0,
    0x2a, 0x3b, 0x19, 0x90, 0xa6, 0x5b, 0xe1, 0x90, 0x1b, 0x2c, 0x91, 0xfc, 0x68, 0x5e, 0x08, 0x3b,
]);
pub const PATH_END_NODE_ID: HashId = HashId([
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HashId(pub [u8; 32]);

impl serde::Serialize for HashId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let hex_str = hex::encode(self.0);
        serializer.serialize_str(&hex_str)
    }
}

impl<'de> serde::Deserialize<'de> for HashId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HashIdVisitor;

        impl<'de> serde::de::Visitor<'de> for HashIdVisitor {
            type Value = HashId;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a 64-character hex string")
            }

            fn visit_str<E>(self, v: &str) -> Result<HashId, E>
            where
                E: serde::de::Error,
            {
                let bytes = hex::decode(v).map_err(E::custom)?;
                if bytes.len() != 32 {
                    return Err(E::custom("expected 32 bytes"));
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                Ok(HashId(arr))
            }
        }

        deserializer.deserialize_str(HashIdVisitor)
    }
}

impl fmt::Display for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl TryFrom<String> for HashId {
    type Error = hex::FromHexError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        let bytes = <[u8; 32]>::from_hex(&s)?;
        Ok(HashId(bytes))
    }
}

impl fmt::Debug for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashId({})", hex::encode(self.0))
    }
}

impl TryFrom<&[u8]> for HashId {
    type Error = &'static str;

    fn try_from(slice: &[u8]) -> Result<Self, Self::Error> {
        if slice.len() != 32 {
            return Err("slice must be exactly 32 bytes long");
        }

        let mut array = [0u8; 32];
        array.copy_from_slice(slice);
        Ok(HashId(array))
    }
}

impl TryFrom<&str> for HashId {
    type Error = &'static str;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        let bytes = hex::decode(s).expect("invalid hex string");
        Ok(HashId(bytes.try_into().expect("not 32 bytes")))
    }
}

impl PartialEq<[u8; 32]> for HashId {
    fn eq(&self, other: &[u8; 32]) -> bool {
        &self.0 == other
    }
}

impl PartialEq<HashId> for [u8; 32] {
    fn eq(&self, other: &HashId) -> bool {
        self == &other.0
    }
}

use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, ValueRef};

impl FromSql for HashId {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value {
            ValueRef::Blob(b) => {
                if b.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(b);
                    Ok(HashId(arr))
                } else {
                    Err(FromSqlError::Other(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("expected 32-byte blob, got {}", b.len()),
                    ))))
                }
            }
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

impl ToSql for HashId {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.0.as_ref())) // &[u8]
    }
}

use rusqlite::types::Value;

impl From<HashId> for Value {
    fn from(h: HashId) -> Self {
        Value::Blob(h.0.to_vec())
    }
}

impl From<uuid::Uuid> for HashId {
    fn from(uuid: uuid::Uuid) -> Self {
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(uuid.as_bytes());
        Self(buf)
    }
}

impl HashId {
    pub fn pad_str<T: ToString>(input: T) -> Self {
        let s = input.to_string();
        let hex = format!("{:0>64}", s); // pad to 64 hex chars
        let bytes = hex::decode(hex).expect("invalid hex string");
        HashId(bytes.try_into().expect("not 32 bytes"))
    }

    pub fn convert_str(s: &str) -> Self {
        HashId(calculate_hash(s))
    }

    pub fn random_str() -> Self {
        let mut rng = rand::rng();
        let mut random_bytes = [0u8; 32];
        rng.fill_bytes(&mut random_bytes);
        Self(random_bytes)
    }

    pub fn uuid7() -> Self {
        uuid::Uuid::now_v7().into()
    }

    // this is a hack for the library code, which we have a hack for chromosome_index to be the edge_id
    pub fn extract_digits(&self) -> i64 {
        let hex = format!("{}", self);
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
        if prefix.len() > 64 || prefix.is_empty() {
            return false;
        }

        let end_byte = prefix.len() / 2 + prefix.len() % 2;
        let encoded = hex::encode(&self.0[..end_byte]);
        encoded.starts_with(prefix)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeIntervalBlock {
    pub block_id: i64,
    pub node_id: HashId,
    pub start: i64,
    pub end: i64,
    pub sequence_start: i64,
    pub sequence_end: i64,
    pub strand: Strand,
}

pub fn calculate_hash(t: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(t);
    let result = hasher.finalize();
    result.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_hashes() {
        let result: HashId = "a82639b6f8c3a6e536d8cc562c3b86ff4b012c84ab230c1e5be649aa9ad26d21"
            .try_into()
            .unwrap();
        assert_eq!(calculate_hash("a test"), result);
    }

    #[cfg(test)]
    mod hashid {
        use super::*;

        #[test]
        fn test_starts_with() {
            let hash: HashId = "a82639b6f8c3a6e536d8cc562c3b86ff4b012c84ab230c1e5be649aa9ad26d21"
                .try_into()
                .unwrap();
            assert!(hash.starts_with("a826"));
            assert!(hash
                .starts_with("a82639b6f8c3a6e536d8cc562c3b86ff4b012c84ab230c1e5be649aa9ad26d21"));
            assert!(!hash.starts_with("b826"));
            assert!(!hash
                .starts_with("a82639b6f8c3a6e536d8cc562c3b86ff4b012c84ab230c1e5be649aa9ad26d210"));
            assert!(!hash.starts_with(""));
            assert!(!hash
                .starts_with("a82639b6f8c3a6e536d8cc562c3b86ff4b012c84ab230c1e5be649aa9ad26d219"));
        }
    }
}
