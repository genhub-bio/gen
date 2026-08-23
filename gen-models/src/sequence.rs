use core::hash::{Hash, Hasher};
use std::{io::BufReader, ops::Range, rc::Rc, str, sync};

use cached::proc_macro::cached;
use flate2::read::MultiGzDecoder;
use gen_core::{HashId, Sha256Hash, Workspace, traits::Capnp};
use indexmap::IndexMap;
use noodles::{
    bgzf::{self, gzi},
    core::Region,
    fasta::{self, fai, io::indexed_reader::Builder as IndexBuilder},
};
use rusqlite::{Row, ToSql, params, types::Value};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    assets::{AssetRef, AssetUri, LocalAssetUri},
    db::GraphConnection,
    gen_models_capnp::sequence,
    traits::{Query, max_rows_per_batch},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Sequence {
    pub hash: Sha256Hash,
    pub sequence_type: String,
    sequence: String,
    // These fields are only relevant when the sequence is stored externally.
    pub name: String,
    pub asset_ref_id: Option<HashId>,
    pub length: i64,
    // Indicates whether the sequence is stored externally, a quick flag instead of checking the
    // sequence and asset reference in each caller.
    pub external_sequence: bool,
    #[serde(skip)]
    asset_ref: Option<AssetRef>,
    #[serde(skip)]
    index_asset_refs: Vec<AssetRef>,
    #[serde(skip)]
    workspace: Option<Workspace>,
    #[serde(skip)]
    asset_resolution_error: Option<String>,
}

impl PartialEq for Sequence {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.sequence_type == other.sequence_type
            && self.sequence == other.sequence
            && self.name == other.name
            && self.asset_ref_id == other.asset_ref_id
            && self.length == other.length
            && self.external_sequence == other.external_sequence
    }
}

impl Eq for Sequence {}

impl Hash for Sequence {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
        self.sequence_type.hash(state);
        self.sequence.hash(state);
        self.name.hash(state);
        self.asset_ref_id.hash(state);
        self.length.hash(state);
        self.external_sequence.hash(state);
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum SequenceError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Sequence or asset_ref_id must be set.")]
    NoSequence(),
    #[error("An asset reference must have an accompanying sequence name.")]
    AssetMissingSequenceName(),
    #[error("Sequence length must be specified.")]
    MissingSequenceLength(),
    #[error("Sequence cache lock poisoned: {0}")]
    CachePoisoned(String),
    #[error("Invalid indexed sequence region '{name}': {reason}")]
    InvalidRegion { name: String, reason: String },
    #[error("Sequence I/O error: {0}")]
    Io(String),
    #[error("Invalid UTF-8 while reading sequence data: {0}")]
    Utf8(String),
    #[error("Sequence id '{name}' not found in fasta file {file_path}")]
    IdMissing { file_path: String, name: String },
    #[error("Sequence bounds out of range: start={start}, end={end}, length={length}")]
    BoundsError { start: i64, end: i64, length: i64 },
    #[error("Unable to resolve sequence asset: {0}")]
    AssetResolution(String),
}

impl<'a> Capnp<'a> for Sequence {
    type Builder = sequence::Builder<'a>;
    type Reader = sequence::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_hash(&self.hash.0).unwrap();
        builder.set_sequence_type(&self.sequence_type);
        builder.set_sequence(&self.sequence);
        builder.set_name(&self.name);
        builder.set_length(self.length);
        builder.set_external_sequence(self.external_sequence);
        if let Some(asset_ref_id) = self.asset_ref_id {
            builder.set_asset_ref_id(&asset_ref_id.0).unwrap();
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let hash = reader
            .get_hash()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let sequence_type = reader.get_sequence_type().unwrap().to_string().unwrap();
        let sequence = reader.get_sequence().unwrap().to_string().unwrap();
        let name = reader.get_name().unwrap().to_string().unwrap();
        let length = reader.get_length();
        let asset_ref_id = reader.get_asset_ref_id().ok().and_then(|value| {
            value
                .as_slice()
                .and_then(|bytes| bytes.try_into().ok())
                .map(HashId)
        });
        let external_sequence = asset_ref_id.is_some();

        Sequence {
            hash,
            sequence_type,
            sequence,
            name,
            asset_ref_id,
            length,
            external_sequence,
            asset_ref: None,
            index_asset_refs: Vec::new(),
            workspace: None,
            asset_resolution_error: None,
        }
    }
}

#[derive(Default, Debug)]
pub struct NewSequence<'a> {
    sequence_type: Option<&'a str>,
    sequence: Option<&'a str>,
    name: Option<&'a str>,
    asset_ref_id: Option<&'a HashId>,
    length: Option<i64>,
    shallow: bool,
}

impl<'a> From<&'a Sequence> for NewSequence<'a> {
    fn from(value: &'a Sequence) -> NewSequence<'a> {
        NewSequence::new()
            .sequence_type(&value.sequence_type)
            .sequence(&value.sequence)
            .name(&value.name)
            .asset_ref_id(value.asset_ref_id.as_ref())
            .length(value.length)
    }
}

impl<'a> NewSequence<'a> {
    pub fn new() -> NewSequence<'static> {
        NewSequence {
            shallow: false,
            ..NewSequence::default()
        }
    }

    pub fn shallow(mut self, setting: bool) -> Self {
        self.shallow = setting;
        self
    }

    pub fn sequence_type(mut self, seq_type: &'a str) -> Self {
        self.sequence_type = Some(seq_type);
        self
    }

    pub fn sequence(mut self, sequence: &'a str) -> Self {
        self.sequence = Some(sequence);
        self.length = Some(sequence.len() as i64);
        self
    }

    pub fn name(mut self, name: &'a str) -> Self {
        self.name = Some(name);
        self
    }

    pub fn asset_ref_id(mut self, asset_ref_id: Option<&'a HashId>) -> Self {
        if asset_ref_id.is_some() {
            self.asset_ref_id = asset_ref_id;
            self.shallow = true;
        }
        self
    }

    pub fn length(mut self, length: i64) -> Self {
        self.length = Some(length);
        self
    }

    pub fn hash(&self) -> Sha256Hash {
        let mut hasher = Sha256::new();
        hasher.update(self.sequence_type.expect("Sequence type must be defined."));
        hasher.update(";");
        if let Some(v) = self.sequence {
            hasher.update(v);
        } else {
            hasher.update("");
        }
        hasher.update(";");
        if let Some(v) = self.name {
            hasher.update(v);
        } else {
            hasher.update("");
        }
        hasher.update(";");
        if let Some(value) = self.asset_ref_id {
            hasher.update(value.0);
        } else {
            hasher.update("");
        }
        hasher.update(";");
        Sha256Hash(hasher.finalize().into())
    }

    pub fn build(self) -> Sequence {
        let external_sequence = self.asset_ref_id.is_some();
        Sequence {
            hash: self.hash(),
            sequence_type: self.sequence_type.unwrap().to_string(),
            sequence: self.sequence.unwrap_or("").to_string(),
            name: self.name.unwrap_or("").to_string(),
            asset_ref_id: self.asset_ref_id.copied(),
            length: self.length.unwrap(),
            external_sequence,
            asset_ref: None,
            index_asset_refs: Vec::new(),
            workspace: None,
            asset_resolution_error: None,
        }
    }

    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(self, conn))
    )]
    pub fn save(self, conn: &GraphConnection) -> Result<Sequence, SequenceError> {
        let mut length = 0;
        if self.sequence.is_none() && self.asset_ref_id.is_none() {
            return Err(SequenceError::NoSequence());
        }
        if self.asset_ref_id.is_some() && self.name.is_none() {
            return Err(SequenceError::AssetMissingSequenceName());
        }
        if self.length.is_none() {
            if let Some(v) = self.sequence {
                length = v.len() as i64;
            } else {
                // TODO: if name/path specified, grab length automatically
                return Err(SequenceError::MissingSequenceLength());
            }
        }
        let hash = self.hash();
        match conn.query_row(
            "SELECT hash from sequences where hash = ?1;",
            [hash],
            |row| row.get::<_, Sha256Hash>(0),
        ) {
            Ok(_) => {}
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                let mut stmt = conn.prepare("INSERT INTO sequences (hash, sequence_type, sequence, name, asset_ref_id, length) VALUES (?1, ?2, ?3, ?4, ?5, ?6);")?;
                match stmt.execute(params![
                    hash,
                    self.sequence_type.unwrap().to_string(),
                    if self.shallow {
                        ""
                    } else {
                        self.sequence.unwrap()
                    },
                    self.name.unwrap_or(""),
                    self.asset_ref_id,
                    self.length.unwrap_or(length)
                ]) {
                    Ok(_) => {}
                    Err(err) => return Err(SequenceError::DatabaseError(err)),
                }
            }
            Err(err) => return Err(SequenceError::DatabaseError(err)),
        };

        Ok(Sequence {
            hash,
            sequence_type: self.sequence_type.unwrap().to_string(),
            sequence: self.sequence.unwrap_or("").to_string(),
            name: self.name.unwrap_or("").to_string(),
            asset_ref_id: self.asset_ref_id.copied(),
            length: self.length.unwrap_or(length),
            external_sequence: self.asset_ref_id.is_some(),
            asset_ref: None,
            index_asset_refs: Vec::new(),
            workspace: None,
            asset_resolution_error: None,
        })
    }
}

#[cached(
    key = "String",
    convert = r#"{ format!("{}:{}", workspace.base_dir().display(), asset_ref.id) }"#
)]
fn fasta_index(workspace: &Workspace, asset_ref: &AssetRef) -> Option<fai::Index> {
    let reader = asset_ref.reader(workspace).ok()?;
    fai::io::Reader::new(reader).read_index().ok()
}

#[cached(
    key = "String",
    convert = r#"{ format!("{}:{}", workspace.base_dir().display(), asset_ref.id) }"#
)]
fn fasta_gzi_index(workspace: &Workspace, asset_ref: &AssetRef) -> Option<gzi::Index> {
    let reader = asset_ref.reader(workspace).ok()?;
    gzi::io::Reader::new(reader).read_index().ok()
}

fn asset_extension(asset_ref: &AssetRef) -> Option<String> {
    <dyn AssetUri>::from_uri(&asset_ref.uri)
        .suffix()
        .and_then(|suffix| suffix.rsplit('.').next().map(str::to_string))
}

fn validate_sequence_bounds(
    start: i64,
    end: i64,
    length: i64,
) -> Result<(usize, usize), SequenceError> {
    if start < 0 || end < 0 || start > length || end > length {
        return Err(SequenceError::BoundsError { start, end, length });
    }

    Ok((start as usize, end as usize))
}

fn circular_sequence_slice(sequence: &str, start: i64, end: i64) -> Result<String, SequenceError> {
    let length = sequence.len() as i64;
    if length == 0 {
        if start == 0 && end == 0 {
            return Ok(String::new());
        }
        return Err(SequenceError::BoundsError { start, end, length });
    }

    let (start, end) = validate_sequence_bounds(start, end, length)?;
    if start <= end {
        return Ok(sequence[start..end].to_string());
    }

    let mut result = String::with_capacity((length as usize - start) + end);
    result.push_str(&sequence[start..]);
    result.push_str(&sequence[..end]);

    Ok(result)
}

fn sequence_slice(
    sequence: &str,
    start: i64,
    end: i64,
    circular: bool,
) -> Result<String, SequenceError> {
    if circular {
        return circular_sequence_slice(sequence, start, end);
    }

    let length = sequence.len() as i64;
    let (start, end) = validate_sequence_bounds(start, end, length)?;
    if start <= end {
        return Ok(sequence[start..end].to_string());
    }

    Err(SequenceError::BoundsError {
        start: start as i64,
        end: end as i64,
        length,
    })
}

type SequenceCache = sync::RwLock<Option<((HashId, String), Option<String>)>>;

pub fn cached_sequence(
    workspace: &Workspace,
    sequence_asset: &AssetRef,
    index_assets: &[AssetRef],
    name: &str,
    range: Range<i64>,
    circular: bool,
) -> Result<String, SequenceError> {
    static SEQUENCE_CACHE: sync::LazyLock<SequenceCache> =
        sync::LazyLock::new(|| sync::RwLock::new(None));
    let cache_key = (sequence_asset.id, name.to_string());

    {
        let cache = SEQUENCE_CACHE
            .read()
            .map_err(|err| SequenceError::CachePoisoned(err.to_string()))?;
        if let Some((cached_key, cached_sequence)) = cache.as_ref()
            && cached_key == &cache_key
        {
            if let Some(sequence) = cached_sequence {
                return sequence_slice(sequence, range.start, range.end, circular);
            }
            return Err(SequenceError::IdMissing {
                file_path: sequence_asset.uri.clone(),
                name: name.to_string(),
            });
        }
    }

    let mut cache = SEQUENCE_CACHE
        .write()
        .map_err(|err| SequenceError::CachePoisoned(err.to_string()))?;

    let mut sequence: Option<String> = None;
    let fasta_index_asset = index_assets
        .iter()
        .find(|asset_ref| asset_extension(asset_ref).as_deref() == Some("fai"));
    let gzip_index_asset = index_assets
        .iter()
        .find(|asset_ref| asset_extension(asset_ref).as_deref() == Some("gzi"));
    let sequence_extension = asset_extension(sequence_asset);
    let compressed = matches!(sequence_extension.as_deref(), Some("gz" | "bgz"));
    if let Some(index) = fasta_index_asset.and_then(|asset_ref| fasta_index(workspace, asset_ref)) {
        let region = name
            .parse::<Region>()
            .map_err(|err| SequenceError::InvalidRegion {
                name: name.to_string(),
                reason: err.to_string(),
            })?;
        let builder = IndexBuilder::default().set_index(index);
        if let Some(gzi_index) =
            gzip_index_asset.and_then(|asset_ref| fasta_gzi_index(workspace, asset_ref))
        {
            let sequence_reader = sequence_asset
                .reader(workspace)
                .map_err(|err| SequenceError::Io(err.to_string()))?;
            let bgzf_reader = bgzf::io::indexed_reader::Builder::default()
                .set_index(gzi_index)
                .build_from_reader(sequence_reader)
                .map_err(|err| SequenceError::Io(err.to_string()))?;
            let mut reader = builder
                .build_from_reader(bgzf_reader)
                .map_err(|err| SequenceError::Io(err.to_string()))?;
            sequence = Some(
                str::from_utf8(
                    reader
                        .query(&region)
                        .map_err(|err| SequenceError::Io(err.to_string()))?
                        .sequence()
                        .as_ref(),
                )
                .map_err(|err| SequenceError::Utf8(err.to_string()))?
                .to_string(),
            )
        } else if !compressed {
            let sequence_reader = sequence_asset
                .reader(workspace)
                .map_err(|err| SequenceError::Io(err.to_string()))?;
            let mut reader = builder
                .build_from_reader(sequence_reader)
                .map_err(|err| SequenceError::Io(err.to_string()))?;
            sequence = Some(
                str::from_utf8(
                    reader
                        .query(&region)
                        .map_err(|err| SequenceError::Io(err.to_string()))?
                        .sequence()
                        .as_ref(),
                )
                .map_err(|err| SequenceError::Utf8(err.to_string()))?
                .to_string(),
            );
        }
    }
    if sequence.is_none() {
        let sequence_reader = sequence_asset
            .reader(workspace)
            .map_err(|err| SequenceError::Io(err.to_string()))?;
        let reader_stream: Box<dyn std::io::BufRead> = match sequence_extension.as_deref() {
            Some("gz") => Box::new(BufReader::new(MultiGzDecoder::new(sequence_reader))),
            Some("bgz") => Box::new(bgzf::io::Reader::new(sequence_reader)),
            _ => Box::new(sequence_reader),
        };
        let mut reader = fasta::io::reader::Builder
            .build_from_reader(reader_stream)
            .map_err(|err| SequenceError::Io(err.to_string()))?;
        for result in reader.records() {
            let record = result.map_err(|err| SequenceError::Io(err.to_string()))?;
            if String::from_utf8(record.name().to_vec())
                .map_err(|err| SequenceError::Utf8(err.to_string()))?
                == name
            {
                sequence = Some(
                    str::from_utf8(record.sequence().as_ref())
                        .map_err(|err| SequenceError::Utf8(err.to_string()))?
                        .to_string(),
                );
                break;
            }
        }
    }
    // This is an LRU cache setup; keep only the last sequence fetched so loading multiple plant
    // genomes does not retain every genome in memory.
    *cache = Some((cache_key, sequence));
    // we do this to avoid a clone of potentially large data.
    if let Some((_, Some(seq))) = cache.as_ref() {
        return sequence_slice(seq, range.start, range.end, circular);
    }
    Err(SequenceError::IdMissing {
        file_path: sequence_asset.uri.clone(),
        name: name.to_string(),
    })
}

impl Sequence {
    /// Saves sequences in bounded batches, retaining any row already stored under the same hash.
    #[cfg_attr(feature = "profiling", tracing::instrument(skip(conn, sequences)))]
    pub fn bulk_save(conn: &GraphConnection, sequences: &[Sequence]) -> Result<(), SequenceError> {
        let batch_size = max_rows_per_batch(conn, 6);

        for chunk in sequences.chunks(batch_size) {
            let mut sql = String::from(
                "INSERT OR IGNORE INTO sequences
                 (hash, sequence_type, sequence, name, asset_ref_id, length) VALUES ",
            );
            for row_index in 0..chunk.len() {
                if row_index > 0 {
                    sql.push(',');
                }
                sql.push_str("(?, ?, ?, ?, ?, ?)");
            }
            sql.push(';');

            let mut values = Vec::with_capacity(chunk.len() * 6);
            for sequence in chunk {
                values.push(Value::from(sequence.hash));
                values.push(Value::from(sequence.sequence_type.clone()));
                values.push(Value::from(sequence.sequence.clone()));
                values.push(Value::from(sequence.name.clone()));
                values.push(Value::from(sequence.asset_ref_id));
                values.push(Value::from(sequence.length));
            }
            let mut statement = conn.prepare_cached(&sql)?;
            statement.execute(rusqlite::params_from_iter(values))?;
        }

        Ok(())
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> NewSequence<'static> {
        NewSequence::new()
    }

    fn is_circular(&self) -> bool {
        self.sequence_type
            .split_whitespace()
            .any(|part| part.eq_ignore_ascii_case("circular"))
    }

    pub fn query_by_ids<T>(
        conn: &GraphConnection,
        workspace: &Workspace,
        ids: &[T],
        history_ref: Option<&str>,
    ) -> Vec<Self>
    where
        T: Clone,
        rusqlite::types::Value: From<T>,
    {
        let sequence_table = <Self as Query>::table_name_with_history_ref(history_ref);
        let asset_table = AssetRef::table_name_with_history_ref(history_ref);
        let query = format!(
            "WITH arr AS (
                SELECT value, rowid AS pos FROM rarray(:ids)
             )
             SELECT sequences.*, asset_refs.*, index_assets.*
             FROM {sequence_table} AS sequences
             LEFT JOIN {asset_table} AS asset_refs
               ON sequences.asset_ref_id = asset_refs.id
             LEFT JOIN {asset_table} AS index_assets
               ON asset_refs.id = index_assets.upstream_asset_ref_id
              AND index_assets.role = 'sequence-index'
             JOIN arr ON sequences.hash = arr.value
             ORDER BY arr.pos;"
        );
        let values: Vec<Value> = ids.iter().map(|id| Value::from(id.clone())).collect();
        let id_values = Rc::new(values);
        let mut query_params: Vec<(&str, &dyn ToSql)> = vec![(":ids", &id_values)];
        if let Some(history_ref) = history_ref.as_ref() {
            query_params.push((":history_ref", history_ref));
        }
        let mut statement = conn.prepare(&query).unwrap();
        let mut sequences: IndexMap<Sha256Hash, Self> = IndexMap::new();
        for row in statement
            .query_map(&query_params[..], |row| {
                Ok(Self::process_joined_row(row, workspace))
            })
            .unwrap()
        {
            let sequence = row.unwrap();
            let sequence_hash = sequence.hash;
            if let Some(existing) = sequences.get_mut(&sequence_hash) {
                existing.index_asset_refs.extend(sequence.index_asset_refs);
            } else {
                sequences.insert(sequence_hash, sequence);
            }
        }
        sequences.into_values().collect()
    }

    fn process_joined_row(row: &Row, workspace: &Workspace) -> Self {
        let mut sequence = <Self as Query>::process_row(row);
        if let Some(asset_ref_id) = sequence.asset_ref_id {
            let Some(joined_asset_ref_id): Option<HashId> = row.get(6).unwrap() else {
                sequence.asset_resolution_error =
                    Some(format!("asset reference {asset_ref_id} does not exist"));
                return sequence;
            };
            let asset_ref = AssetRef {
                id: joined_asset_ref_id,
                uri: row.get(7).unwrap(),
                file_type: row.get(8).unwrap(),
                checksum: row.get(9).unwrap(),
                size: row.get(10).unwrap(),
                role: row.get(11).unwrap(),
                logical_path: row.get(12).unwrap(),
                name: row.get(13).unwrap(),
                created_on: row.get(14).unwrap(),
                upstream_asset_ref_id: row.get(15).unwrap(),
            };
            if LocalAssetUri::is_local_path_or_file_uri(&asset_ref.uri)
                && let Err(error) = asset_ref.versioned_store_path(workspace)
            {
                sequence.asset_resolution_error = Some(error.to_string());
                return sequence;
            }
            sequence.asset_ref = Some(asset_ref);
            sequence.workspace = Some(workspace.clone());

            let index_asset_ref_id: Option<HashId> = row.get(16).unwrap();
            if let Some(index_asset_ref_id) = index_asset_ref_id {
                let index_asset = AssetRef {
                    id: index_asset_ref_id,
                    uri: row.get(17).unwrap(),
                    file_type: row.get(18).unwrap(),
                    checksum: row.get(19).unwrap(),
                    size: row.get(20).unwrap(),
                    role: row.get(21).unwrap(),
                    logical_path: row.get(22).unwrap(),
                    name: row.get(23).unwrap(),
                    created_on: row.get(24).unwrap(),
                    upstream_asset_ref_id: row.get(25).unwrap(),
                };
                if !LocalAssetUri::is_local_path_or_file_uri(&index_asset.uri)
                    || index_asset.versioned_store_path(workspace).is_ok()
                {
                    sequence.index_asset_refs.push(index_asset);
                }
            }
        }
        sequence
    }

    pub fn get_sequence(
        &self,
        start: impl Into<Option<i64>>,
        end: impl Into<Option<i64>>,
    ) -> Result<String, SequenceError> {
        // todo: handle circles

        let start: Option<i64> = start.into();
        let end: Option<i64> = end.into();
        let start = start.unwrap_or(0);
        let end = end.unwrap_or(self.length);
        if self.external_sequence {
            if let Some(error) = self.asset_resolution_error.as_ref() {
                return Err(SequenceError::AssetResolution(error.clone()));
            }
            let asset_ref = self.asset_ref.as_ref().ok_or_else(|| {
                SequenceError::AssetResolution(
                    "sequence asset was loaded without repository context".to_string(),
                )
            })?;
            let workspace = self.workspace.as_ref().ok_or_else(|| {
                SequenceError::AssetResolution(
                    "sequence asset was loaded without repository context".to_string(),
                )
            })?;
            return cached_sequence(
                workspace,
                asset_ref,
                &self.index_asset_refs,
                &self.name,
                start..end,
                self.is_circular(),
            );
        }
        if start == 0 && end == self.length {
            return Ok(self.sequence.clone());
        }
        sequence_slice(&self.sequence, start, end, self.is_circular())
    }

    pub fn delete_by_hash(conn: &GraphConnection, hash: &Sha256Hash) {
        let mut stmt = conn
            .prepare("delete from sequences where hash = ?1;")
            .unwrap();
        stmt.execute(params![hash]).unwrap();
    }

    pub fn query_by_blockgroup(
        conn: &GraphConnection,
        workspace: &Workspace,
        block_group_id: &HashId,
    ) -> Vec<Sequence> {
        let asset_table = AssetRef::table_name_with_history_ref(None);
        let query = format!(
            "SELECT sequences.*, asset_refs.*, index_assets.*
             FROM block_group_edges bge
             LEFT JOIN edges ON bge.edge_id = edges.id
             LEFT JOIN nodes ON (edges.source_node_id = nodes.id OR edges.target_node_id = nodes.id)
             LEFT JOIN sequences ON (nodes.sequence_hash = sequences.hash)
             LEFT JOIN {asset_table} AS asset_refs ON sequences.asset_ref_id = asset_refs.id
             LEFT JOIN {asset_table} AS index_assets
               ON asset_refs.id = index_assets.upstream_asset_ref_id
              AND index_assets.role = 'sequence-index'
             WHERE bge.block_group_id = ?1;"
        );
        let mut statement = conn.prepare(&query).unwrap();
        let mut sequences: IndexMap<Sha256Hash, Self> = IndexMap::new();
        for row in statement
            .query_map(params![block_group_id], |row| {
                Ok(Self::process_joined_row(row, workspace))
            })
            .unwrap()
        {
            let sequence = row.unwrap();
            let sequence_hash = sequence.hash;
            if let Some(existing) = sequences.get_mut(&sequence_hash) {
                existing.index_asset_refs.extend(sequence.index_asset_refs);
            } else {
                sequences.insert(sequence_hash, sequence);
            }
        }
        sequences.into_values().collect()
    }
}

impl Query for Sequence {
    type Model = Sequence;

    const PRIMARY_KEY: &'static str = "hash";
    const TABLE_NAME: &'static str = "sequences";

    fn process_row(row: &Row) -> Self::Model {
        let asset_ref_id: Option<HashId> = row.get(4).unwrap();
        let hash: Sha256Hash = row.get(0).unwrap();
        let sequence = row.get(2).unwrap();
        Sequence {
            hash,
            sequence_type: row.get(1).unwrap(),
            sequence,
            name: row.get(3).unwrap(),
            asset_ref_id,
            length: row.get(5).unwrap(),
            external_sequence: asset_ref_id.is_some(),
            asset_ref: None,
            index_asset_refs: Vec::new(),
            workspace: None,
            asset_resolution_error: None,
        }
    }
}

pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&base| match base.to_ascii_uppercase() {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            b'U' => b'A',
            b'N' => b'N',
            b'R' => b'Y',
            b'Y' => b'R',
            b'S' => b'S',
            b'W' => b'W',
            b'K' => b'M',
            b'M' => b'K',
            b'B' => b'V',
            b'V' => b'B',
            b'D' => b'H',
            b'H' => b'D',
            _ => base,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{fs, fs::OpenOptions, io::Write, time};

    use gen_core::traits::Capnp as _;
    use rand::RngExt;
    use sha2::Digest as _;

    use super::{Sequence, SequenceError};
    use crate::{
        assets::{AssetRef, AssetRole},
        gen_models_capnp::sequence,
        operations::OperationFile,
        test_helpers::{get_connection, setup_gen_on_disk},
        traits::Query,
    };

    fn prepare_asset(context: &crate::db::DbContext, path: &str, role: AssetRole) -> AssetRef {
        let operation_file = OperationFile::new(path).set_role(role);
        let asset_ref = operation_file
            .prepare_asset_ref(context.workspace(), 1)
            .expect("should retain test asset");
        AssetRef::create(context.graph().conn(), &asset_ref)
            .expect("should create test asset reference");
        asset_ref
    }

    fn prepare_derived_asset(
        context: &crate::db::DbContext,
        path: &str,
        role: AssetRole,
        upstream_asset_ref_id: &gen_core::HashId,
    ) -> AssetRef {
        let operation_file = OperationFile::new(path)
            .set_role(role)
            .set_upstream_asset_ref_id(upstream_asset_ref_id);
        let asset_ref = operation_file
            .prepare_asset_ref(context.workspace(), 1)
            .expect("should retain derived test asset");
        AssetRef::create(context.graph().conn(), &asset_ref)
            .expect("should create derived test asset reference");
        asset_ref
    }

    #[test]
    fn test_builder() {
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCG")
            .build();
        assert_eq!(sequence.length, 4);
        assert_eq!(sequence.sequence, "ATCG");
        assert_eq!(
            sequence.hash.to_string(),
            "4ab41cf208e5a797ed0052a3fc5f87bed18f94fc1fc7fdd6ed3199b4e1c34ace"
        );
    }

    #[test]
    fn test_builder_with_asset_ref() {
        let asset_ref_id = gen_core::HashId::convert_str("asset");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .name("chr1")
            .asset_ref_id(Some(&asset_ref_id))
            .length(50)
            .build();
        assert_eq!(sequence.length, 50);
        assert_eq!(sequence.sequence, "");
    }

    #[test]
    fn test_create_sequence_in_db() {
        let conn = &get_connection(None).unwrap();
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AACCTT")
            .save(conn)
            .unwrap();
        assert_eq!(&sequence.sequence, "AACCTT");
        assert_eq!(sequence.sequence_type, "DNA");
        assert!(!sequence.external_sequence);
    }

    #[test]
    fn test_delete_sequence_by_hash() {
        let conn = &get_connection(None).unwrap();
        let before_count = Sequence::all(conn).len();
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AACCTT")
            .save(conn)
            .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AACCTTAA")
            .save(conn)
            .unwrap();

        let sequences = Sequence::all(conn);
        assert_eq!(sequences.len(), before_count + 2);

        Sequence::delete_by_hash(conn, &sequence.hash);

        let sequences = Sequence::all(conn);
        assert_eq!(sequences.len(), before_count + 1);
        assert!(sequences.iter().any(|s| s.hash == sequence2.hash));
    }

    #[test]
    fn test_create_sequence_on_disk() {
        let context = setup_gen_on_disk();
        let fasta_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("reference.fa");
        fs::write(&fasta_path, ">chr1\nAACCGGTTAA\n").unwrap();
        let asset_ref = prepare_asset(&context, fasta_path.to_str().unwrap(), AssetRole::Input);
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .name("chr1")
            .asset_ref_id(Some(&asset_ref.id))
            .length(10)
            .save(context.graph().conn())
            .unwrap();
        assert_eq!(sequence.sequence_type, "DNA");
        assert_eq!(&sequence.sequence, "");
        assert_eq!(sequence.name, "chr1");
        assert_eq!(sequence.asset_ref_id, Some(asset_ref.id));
        assert_eq!(sequence.length, 10);
        assert!(sequence.external_sequence);
    }

    #[test]
    fn test_get_sequence() {
        let conn = &get_connection(None).unwrap();
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCGATCGATCGATCGGGAACACACAGAGA")
            .save(conn)
            .unwrap();
        assert_eq!(
            sequence.get_sequence(None, None).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(sequence.get_sequence(0, 5).unwrap(), "ATCGA");
        assert_eq!(sequence.get_sequence(10, 15).unwrap(), "CGATC");
        assert_eq!(
            sequence.get_sequence(3, None).unwrap(),
            "GATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(sequence.get_sequence(None, 5).unwrap(), "ATCGA");
        assert_eq!(
            sequence.get_sequence(0, 100),
            Err(SequenceError::BoundsError {
                start: 0,
                end: 100,
                length: 34,
            })
        );
        assert_eq!(
            sequence.get_sequence(5, 2),
            Err(SequenceError::BoundsError {
                start: 5,
                end: 2,
                length: 34,
            })
        );
    }

    #[test]
    fn test_get_sequence_circular() {
        let conn = &get_connection(None).unwrap();
        let sequence = Sequence::new()
            .sequence_type("circular")
            .sequence("AAACCCTTT")
            .save(conn)
            .unwrap();
        assert_eq!(sequence.get_sequence(4, 2).unwrap(), "CCTTTAA");
        assert_eq!(
            sequence.get_sequence(0, 10),
            Err(SequenceError::BoundsError {
                start: 0,
                end: 10,
                length: 9,
            })
        );
    }

    #[test]
    fn test_get_sequence_from_disk() {
        let context = setup_gen_on_disk();
        let temp_file_path = context.workspace().repo_root().unwrap().join("simple.fa");
        fs::write(
            &temp_file_path,
            ">m123\nATCGATCGATCGATCGATCGGGAACACACAGAGA\n",
        )
        .unwrap();
        let asset_ref = prepare_asset(&context, temp_file_path.to_str().unwrap(), AssetRole::Input);
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .name("m123")
            .asset_ref_id(Some(&asset_ref.id))
            .length(34)
            .save(context.graph().conn())
            .unwrap();
        let seq = Sequence::query_by_ids(
            context.graph().conn(),
            context.workspace(),
            &[sequence.hash],
            None,
        )
        .remove(0);
        // Overwrite temp_file_path with a garbage sequence, showing that sequence access uses the versioned file
        // and not any local materialized file.
        fs::write(
            &temp_file_path,
            ">m123\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n",
        )
        .unwrap();
        assert_eq!(
            seq.get_sequence(None, None).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(seq.get_sequence(0, 5).unwrap(), "ATCGA");
        assert_eq!(seq.get_sequence(10, 15).unwrap(), "CGATC");
        assert_eq!(
            seq.get_sequence(3, None).unwrap(),
            "GATCGATCGATCGATCGGGAACACACAGAGA"
        );
        assert_eq!(seq.get_sequence(None, 5).unwrap(), "ATCGA");
    }

    #[cfg(unix)]
    #[test]
    fn test_on_disk_sequence_rejects_asset_outside_repository() {
        use std::os::unix::fs::symlink;

        let context = setup_gen_on_disk();
        let outside_directory = tempfile::tempdir().unwrap();
        let outside_path = outside_directory.path().join("secret.fa");
        let contents = b">secret\nPRIVATE\n";
        fs::write(&outside_path, contents).unwrap();
        let checksum = gen_core::Sha256Hash(sha2::Sha256::digest(contents).into());
        let uri = format!("file://{}", outside_path.display());
        let asset_ref = AssetRef {
            id: AssetRef::id_hash(
                &uri,
                "fasta",
                Some(&checksum),
                &AssetRole::Input,
                outside_path.to_str(),
                Some("secret.fa"),
                None,
            ),
            uri,
            file_type: "fasta".to_string(),
            checksum: Some(checksum),
            size: None,
            role: AssetRole::Input,
            logical_path: Some(outside_path.to_string_lossy().to_string()),
            name: Some("secret.fa".to_string()),
            created_on: 1,
            upstream_asset_ref_id: None,
        };
        AssetRef::create(context.graph().conn(), &asset_ref).unwrap();
        let asset_filename =
            <dyn crate::assets::AssetUri>::from_uri(&asset_ref.uri).hashed_filename(&checksum);
        symlink(
            &outside_path,
            context
                .workspace()
                .asset_dir()
                .unwrap()
                .join(asset_filename),
        )
        .unwrap();
        let unresolved_sequence = Sequence::new()
            .sequence_type("DNA")
            .name("secret")
            .asset_ref_id(Some(&asset_ref.id))
            .length(7)
            .save(context.graph().conn())
            .unwrap();
        let sequence = Sequence::query_by_ids(
            context.graph().conn(),
            context.workspace(),
            &[unresolved_sequence.hash],
            None,
        )
        .remove(0);

        let error = sequence
            .get_sequence(None, None)
            .expect_err("should not read an asset outside the repository");
        assert!(
            matches!(&error, SequenceError::AssetResolution(reason) if reason.contains("not within repo root")),
            "should report the repository boundary: {error}"
        );
    }

    #[test]
    fn test_get_sequence_from_disk_circular() {
        let context = setup_gen_on_disk();
        let temp_file_path = context.workspace().repo_root().unwrap().join("simple.fa");
        fs::write(&temp_file_path, ">m123\nAAACCCTTT\n").unwrap();
        let asset_ref = prepare_asset(&context, temp_file_path.to_str().unwrap(), AssetRole::Input);
        let sequence = Sequence::new()
            .sequence_type("circular")
            .name("m123")
            .asset_ref_id(Some(&asset_ref.id))
            .length(9)
            .save(context.graph().conn())
            .unwrap();
        let seq = Sequence::query_by_ids(
            context.graph().conn(),
            context.workspace(),
            &[sequence.hash],
            None,
        )
        .remove(0);
        assert_eq!(seq.get_sequence(4, 2).unwrap(), "CCTTTAA");
        assert_eq!(
            seq.get_sequence(0, 10),
            Err(SequenceError::BoundsError {
                start: 0,
                end: 10,
                length: 9,
            })
        );
    }

    #[test]
    fn test_on_disk_sequence_uses_retained_index_asset() {
        let context = setup_gen_on_disk();
        let repo_root = context.workspace().repo_root().unwrap();
        let fasta_path = repo_root.join("indexed.fa");
        let index_path = repo_root.join("indexed.fa.fai");
        fs::write(&fasta_path, ">chr1\nACGTACGT\n").unwrap();
        fs::write(&index_path, "chr1\t8\t6\t8\t9\n").unwrap();
        let asset_ref = prepare_asset(&context, fasta_path.to_str().unwrap(), AssetRole::Input);
        let index_asset_ref = prepare_derived_asset(
            &context,
            index_path.to_str().unwrap(),
            AssetRole::SequenceIndex,
            &asset_ref.id,
        );
        let unresolved_sequence = Sequence::new()
            .sequence_type("DNA")
            .name("chr1")
            .asset_ref_id(Some(&asset_ref.id))
            .length(8)
            .save(context.graph().conn())
            .unwrap();
        let sequence = Sequence::query_by_ids(
            context.graph().conn(),
            context.workspace(),
            &[unresolved_sequence.hash],
            None,
        )
        .remove(0);
        fs::remove_file(&fasta_path).unwrap();
        fs::remove_file(&index_path).unwrap();

        assert_eq!(
            sequence.get_sequence(2, 6).unwrap(),
            "GTAC",
            "should read the retained sequence through its index"
        );
        assert!(
            sequence
                .index_asset_refs
                .iter()
                .any(|asset_ref| asset_ref.id == index_asset_ref.id),
            "resolved sequence should include its retained index AssetRef"
        );
        assert_eq!(
            index_asset_ref.upstream_asset_ref_id,
            Some(asset_ref.id),
            "index AssetRef should link to the immutable sequence AssetRef"
        );
    }

    #[test]
    // #[cfg(feature = "benchmark")]
    fn test_cached_sequence_performance() {
        let context = setup_gen_on_disk();
        let temp_file_path = context.workspace().repo_root().unwrap().join("large.fa");
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&temp_file_path)
            .unwrap();
        writeln!(file, ">chr22").unwrap();
        for _ in 1..3_000_000 {
            writeln!(
                file,
                "ATCGATCGATCGATCGATCGGGAACACACAGAGAATCGATCGATCGATCGATCGGGAACACACAGAGA"
            )
            .unwrap();
        }
        // write index
        let index_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("large.fa.fai");
        fs::write(&index_path, "chr22	203999932	7	68	69\n").unwrap();
        let asset_ref = prepare_asset(&context, temp_file_path.to_str().unwrap(), AssetRole::Input);
        prepare_derived_asset(
            &context,
            index_path.to_str().unwrap(),
            AssetRole::SequenceIndex,
            &asset_ref.id,
        );
        let unresolved_sequence = Sequence::new()
            .sequence_type("DNA")
            .asset_ref_id(Some(&asset_ref.id))
            .name("chr22")
            .length(203_999_932)
            .save(context.graph().conn())
            .unwrap();
        let sequence = Sequence::query_by_ids(
            context.graph().conn(),
            context.workspace(),
            &[unresolved_sequence.hash],
            None,
        )
        .remove(0);
        let s = time::Instant::now();
        for _ in 1..1_000_000 {
            let start = rand::rng().random_range(1..200_000_000);

            sequence.get_sequence(start, start + 20).unwrap();
        }
        let elapsed = s.elapsed().as_secs();
        assert!(
            elapsed < 5,
            "Cached sequence benchmark failed: {elapsed}s elapsed"
        );
    }

    #[test]
    fn test_capnp_serialization() {
        use capnp::message::TypedBuilder;

        let sequence = Sequence {
            hash: gen_core::Sha256Hash::convert_str("test_hash"),
            sequence_type: "DNA".to_string(),
            sequence: "ATCG".to_string(),
            name: "test_seq".to_string(),
            asset_ref_id: Some(gen_core::HashId::convert_str("asset")),
            length: 4,
            external_sequence: true,
            asset_ref: None,
            index_asset_refs: Vec::new(),
            workspace: None,
            asset_resolution_error: None,
        };

        let mut message = TypedBuilder::<sequence::Owned>::new_default();
        let mut root = message.init_root();
        sequence.write_capnp(&mut root);

        let deserialized = Sequence::read_capnp(root.into_reader());
        assert_eq!(sequence, deserialized);
    }
}
