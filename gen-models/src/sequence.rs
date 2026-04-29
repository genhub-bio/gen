use std::{collections::HashMap, fs, str, sync};

use cached::proc_macro::cached;
use gen_core::{HashId, traits::Capnp};
use noodles::{
    bgzf::{self, gzi},
    core::Region,
    fasta::{self, fai, io::indexed_reader::Builder as IndexBuilder},
};
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{db::GraphConnection, errors::SequenceError, gen_models_capnp::sequence, traits::*};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct Sequence {
    pub hash: HashId,
    pub sequence_type: String,
    sequence: String,
    // these 2 fields are only relevant when the sequence is stored externally
    pub name: String,
    pub file_path: String,
    pub length: i64,
    // indicates whether the sequence is stored externally, a quick flag instead of having to
    // check sequence or file_path and do the logic in function calls.
    pub external_sequence: bool,
}

impl<'a> Capnp<'a> for Sequence {
    type Builder = sequence::Builder<'a>;
    type Reader = sequence::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_hash(&self.hash.0).unwrap();
        builder.set_sequence_type(&self.sequence_type);
        builder.set_sequence(&self.sequence);
        builder.set_name(&self.name);
        builder.set_file_path(&self.file_path);
        builder.set_length(self.length);
        builder.set_external_sequence(self.external_sequence);
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
        let file_path = reader.get_file_path().unwrap().to_string().unwrap();
        let length = reader.get_length();
        let external_sequence = reader.get_external_sequence();

        Sequence {
            hash,
            sequence_type,
            sequence,
            name,
            file_path,
            length,
            external_sequence,
        }
    }
}

#[derive(Default, Debug)]
pub struct NewSequence<'a> {
    sequence_type: Option<&'a str>,
    sequence: Option<&'a str>,
    name: Option<&'a str>,
    file_path: Option<&'a str>,
    length: Option<i64>,
    shallow: bool,
}

impl<'a> From<&'a Sequence> for NewSequence<'a> {
    fn from(value: &'a Sequence) -> NewSequence<'a> {
        NewSequence::new()
            .sequence_type(&value.sequence_type)
            .sequence(&value.sequence)
            .name(&value.name)
            .file_path(&value.file_path)
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

    pub fn file_path(mut self, path: &'a str) -> Self {
        if !path.is_empty() {
            self.file_path = Some(path);
            self.shallow = true;
        }
        self
    }

    pub fn length(mut self, length: i64) -> Self {
        self.length = Some(length);
        self
    }

    pub fn hash(&self) -> HashId {
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
        if let Some(v) = self.file_path {
            hasher.update(v);
        } else {
            hasher.update("");
        }
        hasher.update(";");

        HashId(hasher.finalize().into())
    }

    pub fn build(self) -> Sequence {
        let file_path = self.file_path.unwrap_or("").to_string();
        let external_sequence = !file_path.is_empty();
        Sequence {
            hash: self.hash(),
            sequence_type: self.sequence_type.unwrap().to_string(),
            sequence: self.sequence.unwrap_or("").to_string(),
            name: self.name.unwrap_or("").to_string(),
            file_path,
            length: self.length.unwrap(),
            external_sequence,
        }
    }

    pub fn save(self, conn: &GraphConnection) -> Sequence {
        let mut length = 0;
        if self.sequence.is_none() && self.file_path.is_none() {
            panic!("Sequence or file_path must be set.");
        }
        if self.file_path.is_some() && self.name.is_none() {
            panic!("A filepath must have an accompanying sequence name");
        }
        if self.length.is_none() {
            if let Some(v) = self.sequence {
                length = v.len() as i64;
            } else {
                // TODO: if name/path specified, grab length automatically
                panic!("Sequence length must be specified.");
            }
        }
        let hash = self.hash();
        match conn.query_row(
            "SELECT hash from sequences where hash = ?1;",
            [hash],
            |row| row.get::<_, HashId>(0),
        ) {
            Ok(_) => {}
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                let mut stmt = conn.prepare("INSERT INTO sequences (hash, sequence_type, sequence, name, file_path, length) VALUES (?1, ?2, ?3, ?4, ?5, ?6);").unwrap();
                stmt.execute(params![
                    hash,
                    self.sequence_type.unwrap().to_string(),
                    if self.shallow {
                        ""
                    } else {
                        self.sequence.unwrap()
                    },
                    self.name.unwrap_or(""),
                    self.file_path.unwrap_or(""),
                    self.length.unwrap_or(length)
                ])
                .unwrap();
            }
            Err(_e) => {
                panic!("something bad happened querying the database")
            }
        };
        Sequence {
            hash,
            sequence_type: self.sequence_type.unwrap().to_string(),
            sequence: self.sequence.unwrap_or("").to_string(),
            name: self.name.unwrap_or("").to_string(),
            file_path: self.file_path.unwrap_or("").to_string(),
            length: self.length.unwrap_or(length),
            external_sequence: !self.file_path.unwrap_or("").is_empty(),
        }
    }
}

#[cached(key = "String", convert = r#"{ format!("{}", path) }"#)]
fn fasta_index(path: &str) -> Option<fai::Index> {
    let index_path = format!("{path}.fai");
    if fs::metadata(&index_path).is_ok() {
        return Some(fai::fs::read(&index_path).unwrap());
    }
    None
}

#[cached(key = "String", convert = r#"{ format!("{}", path) }"#)]
fn fasta_gzi_index(path: &str) -> Option<gzi::Index> {
    let index_path = format!("{path}.gzi");
    if fs::metadata(&index_path).is_ok() {
        return Some(gzi::fs::read(&index_path).unwrap());
    }
    None
}

pub fn cached_sequence(
    file_path: &str,
    name: &str,
    start: usize,
    end: usize,
) -> Result<String, SequenceError> {
    static SEQUENCE_CACHE: sync::LazyLock<sync::RwLock<HashMap<String, Option<String>>>> =
        sync::LazyLock::new(|| sync::RwLock::new(HashMap::new()));
    let key = format!("{file_path}-{name}");

    {
        let cache = SEQUENCE_CACHE.read().unwrap();
        if let Some(cached_sequence) = cache.get(&key) {
            if let Some(sequence) = cached_sequence {
                if start > end || end > sequence.len() {
                    return Err(SequenceError::OutOfBounds {
                        start,
                        end,
                        length: sequence.len(),
                    });
                }
                return Ok(sequence[start..end].to_string());
            }
            return Err(SequenceError::NotFound {
                file_path: file_path.to_string(),
                name: name.to_string(),
            });
        }
    }

    let mut cache = SEQUENCE_CACHE.write().unwrap();

    let mut sequence: Option<String> = None;
    let region = name.parse::<Region>().unwrap();
    if let Some(index) = fasta_index(file_path) {
        let builder = IndexBuilder::default().set_index(index);
        if let Some(gzi_index) = fasta_gzi_index(file_path) {
            let bgzf_reader = bgzf::io::indexed_reader::Builder::default()
                .set_index(gzi_index)
                .build_from_path(file_path)
                .unwrap();
            let mut reader = builder.build_from_reader(bgzf_reader).unwrap();
            sequence = Some(
                str::from_utf8(reader.query(&region).unwrap().sequence().as_ref())
                    .unwrap()
                    .to_string(),
            )
        } else {
            let mut reader = builder.build_from_path(file_path).unwrap();
            sequence = Some(
                str::from_utf8(reader.query(&region).unwrap().sequence().as_ref())
                    .unwrap()
                    .to_string(),
            );
        }
    } else {
        let mut reader = fasta::io::reader::Builder
            .build_from_path(file_path)
            .unwrap();
        for result in reader.records() {
            let record = result.unwrap();
            if String::from_utf8(record.name().to_vec()).unwrap() == name {
                sequence = Some(
                    str::from_utf8(record.sequence().as_ref())
                        .unwrap()
                        .to_string(),
                );
                break;
            }
        }
    }
    // this is a LRU cache setup, we just keep the last sequence we fetched so we don't end up loading
    // plant genomes into memory.
    cache.clear();
    cache.insert(key.clone(), sequence);
    // we do this to avoid a clone of potentially large data.
    if let Some(seq) = &cache[&key] {
        if start > end || end > seq.len() {
            return Err(SequenceError::OutOfBounds {
                start,
                end,
                length: seq.len(),
            });
        }
        return Ok(seq[start..end].to_string());
    }
    Err(SequenceError::NotFound {
        file_path: file_path.to_string(),
        name: name.to_string(),
    })
}

impl Sequence {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> NewSequence<'static> {
        NewSequence::new()
    }

    pub fn get_sequence(
        &self,
        start: impl Into<Option<i64>>,
        end: impl Into<Option<i64>>,
    ) -> Result<String, SequenceError> {
        // todo: handle circles

        let start: Option<i64> = start.into();
        let end: Option<i64> = end.into();
        let start = start.unwrap_or(0) as usize;
        let end = end.unwrap_or(self.length) as usize;
        if self.external_sequence {
            return cached_sequence(&self.file_path, &self.name, start, end);
        }
        if start > end || end > self.sequence.len() {
            return Err(SequenceError::OutOfBounds {
                start,
                end,
                length: self.sequence.len(),
            });
        }
        if start == 0 && end as i64 == self.length {
            return Ok(self.sequence.clone());
        }
        Ok(self.sequence[start..end].to_string())
    }

    pub fn delete_by_hash(conn: &GraphConnection, hash: &HashId) {
        let mut stmt = conn
            .prepare("delete from sequences where hash = ?1;")
            .unwrap();
        stmt.execute(params![hash]).unwrap();
    }

    pub fn query_by_blockgroup(conn: &GraphConnection, block_group_id: &HashId) -> Vec<Sequence> {
        Sequence::query(
            conn,
            "select sequences.* from block_group_edges bge left join edges on bge.edge_id = edges.id left join nodes on (edges.source_node_id = nodes.id or edges.target_node_id = nodes.id) left join sequences on (nodes.sequence_hash = sequences.hash) where bge.block_group_id = ?1;",
            params![block_group_id],
        )
    }
}

impl Query for Sequence {
    type Model = Sequence;

    const PRIMARY_KEY: &'static str = "hash";
    const TABLE_NAME: &'static str = "sequences";

    fn process_row(row: &Row) -> Self::Model {
        let file_path: String = row.get(4).unwrap();
        let mut external_sequence = false;
        if !file_path.is_empty() {
            external_sequence = true;
        }
        let hash: HashId = row.get(0).unwrap();
        let sequence = row.get(2).unwrap();
        Sequence {
            hash,
            sequence_type: row.get(1).unwrap(),
            sequence,
            name: row.get(3).unwrap(),
            file_path,
            length: row.get(5).unwrap(),
            external_sequence,
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    #[allow(unused_imports)]
    use std::time;
    use std::{fs::OpenOptions, io::Write};

    use rand::RngExt;

    use super::*;
    use crate::test_helpers::get_connection;

    #[test]
    fn test_builder() {
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCG")
            .build();
        assert_eq!(sequence.length, 4);
        assert_eq!(sequence.sequence, "ATCG");
    }

    #[test]
    fn test_builder_with_from_disk() {
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .name("chr1")
            .file_path("/foo/bar")
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
            .save(conn);
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
            .save(conn);
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AACCTTAA")
            .save(conn);

        let sequences = Sequence::all(conn);
        assert_eq!(sequences.len(), before_count + 2);

        Sequence::delete_by_hash(conn, &sequence.hash);

        let sequences = Sequence::all(conn);
        assert_eq!(sequences.len(), before_count + 1);
        assert!(sequences.iter().any(|s| s.hash == sequence2.hash));
    }

    #[test]
    fn test_create_sequence_on_disk() {
        let conn = &get_connection(None).unwrap();
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .name("chr1")
            .file_path("/some/path.fa")
            .length(10)
            .save(conn);
        assert_eq!(sequence.sequence_type, "DNA");
        assert_eq!(&sequence.sequence, "");
        assert_eq!(sequence.name, "chr1");
        assert_eq!(sequence.file_path, "/some/path.fa");
        assert_eq!(sequence.length, 10);
        assert!(sequence.external_sequence);
    }

    #[test]
    fn test_get_sequence() {
        let conn = &get_connection(None).unwrap();
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATCGATCGATCGATCGATCGGGAACACACAGAGA")
            .save(conn);
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
            sequence.get_sequence(100, 110),
            Err(SequenceError::OutOfBounds {
                start: 100,
                end: 110,
                length: 34,
            })
        );
    }

    #[test]
    fn test_get_sequence_from_disk() {
        let conn = &get_connection(None).unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("simple.fa");
        fs::write(
            &temp_file_path,
            ">m123\nATCGATCGATCGATCGATCGGGAACACACAGAGA\n",
        )
        .unwrap();
        let seq = Sequence::new()
            .sequence_type("DNA")
            .name("m123")
            .file_path(temp_file_path.to_str().unwrap())
            .length(34)
            .save(conn);
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
        assert_eq!(
            seq.get_sequence(100, 110),
            Err(SequenceError::OutOfBounds {
                start: 100,
                end: 110,
                length: 34,
            })
        );
    }

    #[test]
    fn test_get_sequence_returns_bounds_error_for_stale_length() {
        let conn = &get_connection(None).unwrap();
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("TCGATCGGG")
            .length(10)
            .save(conn);

        assert_eq!(
            sequence.get_sequence(0, 10),
            Err(SequenceError::OutOfBounds {
                start: 0,
                end: 10,
                length: 9,
            })
        );
    }

    #[test]
    // #[cfg(feature = "benchmark")]
    fn test_cached_sequence_performance() {
        let conn = &get_connection(None).unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_file_path = temp_dir.path().join("large.fa");
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
        let index_path = temp_dir.path().join("large.fa.fai");
        fs::write(&index_path, "chr22	203999932	7	68	69\n").unwrap();
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .file_path(temp_file_path.to_str().unwrap())
            .name("chr22")
            .length(203_999_932)
            .save(conn);
        let s = time::Instant::now();
        for _ in 1..1_000_000 {
            let start = rand::rng().random_range(1..200_000_000);

            let _ = sequence.get_sequence(start, start + 20);
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
            hash: HashId::convert_str("test_hash"),
            sequence_type: "DNA".to_string(),
            sequence: "ATCG".to_string(),
            name: "test_seq".to_string(),
            file_path: "/path/to/file".to_string(),
            length: 4,
            external_sequence: false,
        };

        let mut message = TypedBuilder::<sequence::Owned>::new_default();
        let mut root = message.init_root();
        sequence.write_capnp(&mut root);

        let deserialized = Sequence::read_capnp(root.into_reader());
        assert_eq!(sequence, deserialized);
    }
}
