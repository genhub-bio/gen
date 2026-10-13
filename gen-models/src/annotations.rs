use std::{
    fs::{self, File},
    io::{self, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use flate2::read::MultiGzDecoder;
use gen_core::{
    DoltHashId, HashId, NodeIntervalBlock, Sha256Hash, calculate_hash,
    config::Workspace,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use intervaltree::IntervalTree;
use rusqlite::{Result as SqlResult, Row, params, types::Type};
use serde::{Deserialize, Serialize};
use tempfile::{Builder as TempFileBuilder, TempDir};
use thiserror::Error;

use crate::{
    Direction, ForeignKeyCache, ModelSelect, ModelSelectError,
    accession::{Accession, AccessionError, AccessionSelect, AccessionSpan, NewAccession},
    assets::{
        ANNOTATION_ATTRIBUTE_INDEX_ROLE, AssetRef, AssetRole, AssetUri, ChecksummedWriter,
        CompressionType, LocalAssetUri, OperationKind,
    },
    db::{DbContext, GraphConnection, max_rows_per_batch},
    errors::{FileAdditionError, OperationError},
    file_types::FileTypes,
    gen_models_capnp::{annotation, annotation_group, annotation_group_sample},
    history::{HistoryStore, dolt::DoltHistoryStore},
    interval_tree::IntervalTreeSource,
    operations::{
        FileAddition, FileSource, OperationFile, OperationInfo, OperationSummary,
        calculate_reader_checksum, track_asset_refs,
    },
    region::GenRegionError,
    select::query_models,
};
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize, ModelSelect)]
#[model_select(table = "annotation_groups")]
pub struct AnnotationGroup {
    #[model_select(primary_key)]
    pub name: String,
}

impl AnnotationGroup {
    pub fn create(conn: &GraphConnection, name: &str) -> rusqlite::Result<AnnotationGroup> {
        let mut stmt = conn
            .prepare("INSERT INTO annotation_groups (name) VALUES (?1) returning (name);")
            .unwrap();
        stmt.query_row((name,), |row| Ok(AnnotationGroup { name: row.get(0)? }))
    }

    pub fn get_or_create(
        conn: &GraphConnection,
        name: &str,
    ) -> Result<AnnotationGroup, AnnotationGroupError> {
        let mut statement = conn.prepare(
            "INSERT INTO annotation_groups (name) VALUES (?1) ON CONFLICT(name) DO NOTHING;",
        )?;
        statement.execute(params![name])?;
        Ok(AnnotationGroup {
            name: name.to_string(),
        })
    }

    pub fn query_by_sample(
        conn: &GraphConnection,
        sample_name: &str,
        history_ref: Option<&str>,
    ) -> Vec<AnnotationGroup> {
        let samples = AnnotationGroupSample::select(conn)
            .sample_name(sample_name)
            .with_ref(history_ref);
        AnnotationGroup::select(conn)
            .join_filtered_on(
                AnnotationGroupSelect::Name,
                AnnotationGroupSampleSelect::AnnotationGroup,
                samples,
            )
            .order_by(AnnotationGroupSelect::Name, Direction::Asc)
            .load()
            .expect("should load annotation groups for sample")
    }
}

impl<'a> Capnp<'a> for AnnotationGroup {
    type Builder = annotation_group::Builder<'a>;
    type Reader = annotation_group::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_name(&self.name);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        AnnotationGroup {
            name: reader.get_name().unwrap().to_string().unwrap(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ModelSelect)]
#[model_select(table = "annotations", from_row = annotation_from_row)]
pub struct Annotation {
    pub id: HashId,
    pub name: String,
    #[model_select(column = "annotation_group")]
    pub group: String,
    #[model_select(foreign_key(Accession, cache = cached_accession))]
    pub accession_id: HashId,
    /// Runtime-only related accession cache used by file translators and database-loaded relations.
    #[serde(skip)]
    #[model_select(skip)]
    pub cached_accession: ForeignKeyCache<Accession>,
    /// Runtime-only interval tree attached by a file annotation translator.
    #[serde(skip)]
    #[model_select(skip)]
    pub cached_interval_tree: Option<Arc<IntervalTree<i64, NodeIntervalBlock>>>,
    /// History ref this runtime model was selected from.
    #[serde(skip)]
    #[model_select(history_ref)]
    pub history_ref: Option<String>,
    #[model_select(skip)]
    pub extra: Option<AnnotationExtra>,
}

impl PartialEq for Annotation {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.name == other.name
            && self.group == other.group
            && self.accession_id == other.accession_id
            && self.extra == other.extra
    }
}

impl Eq for Annotation {}

fn annotation_from_row(row: &Row) -> SqlResult<Annotation> {
    Ok(Annotation {
        id: row.get(0)?,
        name: row.get(1)?,
        group: row.get(2)?,
        accession_id: row.get(3)?,
        cached_accession: ForeignKeyCache::default(),
        cached_interval_tree: None,
        history_ref: None,
        extra: deserialize_annotation_extra(row.get(4)?).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(4, Type::Text, Box::new(error))
        })?,
    })
}

/// For use with create/bulk_create methods.
#[derive(Clone, Debug)]
pub struct NewAnnotation<'a> {
    /// User-facing annotation name.
    pub name: &'a str,
    /// Accession covered by this annotation.
    pub accession_id: HashId,
    /// Format-specific annotation metadata.
    pub extra: Option<&'a AnnotationExtra>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct AnnotationExtra {
    pub genbank: Option<GenBankExtra>,
    pub gff: Option<GffExtra>,
    pub bed: Option<BedExtra>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct GenBankExtra {
    pub kind: String,
    pub qualifiers: Vec<GenBankQualifier>,
    pub location_operator: Option<GenBankLocationOperator>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct GenBankQualifier {
    pub key: String,
    pub value: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum GenBankLocationOperator {
    Join,
    Order,
    Bond,
    OneOf,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct GffExtra {
    pub source: Option<String>,
    pub ty: String,
    pub score: Option<String>,
    pub phase: Option<String>,
    pub attributes: Vec<GffAttribute>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct GffAttribute {
    pub key: String,
    pub values: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct BedExtra {
    pub score: Option<String>,
    pub thick_start: Option<i64>,
    pub thick_end: Option<i64>,
    pub item_rgb: Option<String>,
    pub block_count: Option<u64>,
    pub block_sizes: Option<Vec<i64>>,
    pub block_starts: Option<Vec<i64>>,
    pub other_fields: Vec<String>,
}

fn serialize_annotation_extra(extra: Option<&AnnotationExtra>) -> Result<String, AnnotationError> {
    serde_json::to_string(extra.unwrap_or(&AnnotationExtra::default()))
        .map_err(|err| AnnotationError::SerializationError(err.to_string()))
}

fn deserialize_annotation_extra(
    value: Option<String>,
) -> Result<Option<AnnotationExtra>, AnnotationError> {
    value
        .map(|value| serde_json::from_str(&value))
        .transpose()
        .map(|value| value.filter(|extra| extra != &AnnotationExtra::default()))
        .map_err(|err| AnnotationError::SerializationError(err.to_string()))
}

impl<'a> Capnp<'a> for Annotation {
    type Builder = annotation::Builder<'a>;
    type Reader = annotation::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_name(&self.name);
        builder.set_annotation_group(&self.group);
        builder.set_accession_id(&self.accession_id.0).unwrap();
        builder.set_extra(
            serialize_annotation_extra(self.extra.as_ref())
                .unwrap_or_default()
                .as_str(),
        );
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let name = reader.get_name().unwrap().to_string().unwrap();
        let group = reader.get_annotation_group().unwrap().to_string().unwrap();
        let accession_id = reader
            .get_accession_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let extra = deserialize_annotation_extra(
            reader
                .get_extra()
                .ok()
                .map(|value| value.to_string().unwrap())
                .filter(|value| !value.is_empty()),
        )
        .unwrap_or(None);

        Annotation {
            id,
            name,
            group,
            accession_id,
            cached_accession: ForeignKeyCache::default(),
            cached_interval_tree: None,
            history_ref: None,
            extra,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum AnnotationError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Selector error: {0}")]
    ModelSelect(#[from] ModelSelectError),
    #[error("Annotation group error: {0}")]
    AnnotationGroupError(#[from] AnnotationGroupError),
    #[error("Accession error: {0}")]
    AccessionError(#[from] AccessionError),
    #[error("Annotation extra serialization error: {0}")]
    SerializationError(String),
}

#[derive(Debug, Error)]
pub enum AddAnnotationError {
    #[error("Region parse error: {0}")]
    RegionParse(#[from] gen_core::region::RegionParseError),
    #[error("Region resolution error: {0}")]
    RegionResolution(#[from] GenRegionError),
    #[error("Accession error: {0}")]
    Accession(#[from] AccessionError),
    #[error("Annotation error: {0}")]
    Annotation(#[from] AnnotationError),
}

impl Annotation {
    pub fn generate_id(name: &str, group: &str, accession_id: &HashId) -> HashId {
        HashId(calculate_hash(&format!("{name}:{group}:{accession_id}",)))
    }

    pub fn create(
        conn: &GraphConnection,
        group: &str,
        annotation: &NewAnnotation<'_>,
    ) -> Result<Annotation, AnnotationError> {
        let id = Annotation::generate_id(annotation.name, group, &annotation.accession_id);
        let query = "INSERT INTO annotations (id, name, annotation_group, accession_id, extra) VALUES (?1, ?2, ?3, ?4, ?5);";
        let mut stmt = conn.prepare(query)?;
        let extra_json = serialize_annotation_extra(annotation.extra)?;
        stmt.execute(params![
            id,
            annotation.name,
            group,
            annotation.accession_id,
            extra_json
        ])?;
        Ok(Annotation {
            id,
            name: annotation.name.to_string(),
            group: group.to_string(),
            accession_id: annotation.accession_id,
            cached_accession: ForeignKeyCache::default(),
            cached_interval_tree: None,
            history_ref: None,
            extra: annotation.extra.cloned(),
        })
    }

    pub fn get_or_create(
        conn: &GraphConnection,
        name: &str,
        group: &str,
        accession_id: &HashId,
        extra: Option<&AnnotationExtra>,
    ) -> Result<Annotation, AnnotationError> {
        AnnotationGroup::get_or_create(conn, group)?;
        let id = Annotation::generate_id(name, group, accession_id);
        let mut statement = conn.prepare(
            "INSERT INTO annotations (id, name, annotation_group, accession_id, extra) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(accession_id, annotation_group, name) DO NOTHING;",
        )?;
        let extra_json = serialize_annotation_extra(extra)?;
        statement.execute(params![id, name, group, accession_id, extra_json])?;
        Ok(Annotation {
            id,
            name: name.to_string(),
            group: group.to_string(),
            accession_id: *accession_id,
            cached_accession: ForeignKeyCache::default(),
            cached_interval_tree: None,
            history_ref: None,
            extra: extra.cloned(),
        })
    }

    /// Creates annotations for one group, preserving existing rows with matching identities.
    pub fn bulk_create(
        conn: &GraphConnection,
        group: &str,
        annotations: &[NewAnnotation<'_>],
    ) -> Result<(), AnnotationError> {
        if annotations.is_empty() {
            return Ok(());
        }

        AnnotationGroup::get_or_create(conn, group)?;
        let batch_size = max_rows_per_batch(conn, 5);
        for annotation_batch in annotations.chunks(batch_size) {
            let mut rows = Vec::with_capacity(annotation_batch.len());
            let mut parameters: Vec<Box<dyn rusqlite::ToSql>> =
                Vec::with_capacity(annotation_batch.len() * 5);
            for annotation in annotation_batch {
                parameters.push(Box::new(Annotation::generate_id(
                    annotation.name,
                    group,
                    &annotation.accession_id,
                )));
                parameters.push(Box::new(annotation.name.to_string()));
                parameters.push(Box::new(group.to_string()));
                parameters.push(Box::new(annotation.accession_id));
                parameters.push(Box::new(serialize_annotation_extra(annotation.extra)?));
                rows.push("(?, ?, ?, ?, ?)");
            }
            let query = format!(
                "INSERT INTO annotations (id, name, annotation_group, accession_id, extra) VALUES {} ON CONFLICT(accession_id, annotation_group, name) DO NOTHING;",
                rows.join(",")
            );
            conn.execute(&query, rusqlite::params_from_iter(parameters))?;
        }
        Ok(())
    }

    pub fn create_with_samples(
        conn: &GraphConnection,
        name: &str,
        group: &str,
        accession_id: &HashId,
        extra: Option<&AnnotationExtra>,
        sample_names: &[&str],
    ) -> Result<Annotation, AnnotationError> {
        let annotation = Annotation::get_or_create(conn, name, group, accession_id, extra)?;
        annotation.add_samples(conn, sample_names)?;
        Ok(annotation)
    }

    pub fn add_samples(
        &self,
        conn: &GraphConnection,
        sample_names: &[&str],
    ) -> Result<(), AnnotationError> {
        if sample_names.is_empty() {
            return Ok(());
        }
        AnnotationGroup::get_or_create(conn, &self.group)?;
        let query = "INSERT INTO annotation_group_samples (annotation_group, sample_name) VALUES (?1, ?2) ON CONFLICT(annotation_group, sample_name) DO NOTHING;";
        let mut stmt = conn.prepare(query)?;
        for sample_name in sample_names {
            stmt.execute(params![self.group, sample_name])?;
        }
        Ok(())
    }

    pub fn get_samples(
        conn: &GraphConnection,
        annotation_group: &str,
    ) -> Result<Vec<String>, AnnotationError> {
        Ok(AnnotationGroupSample::select(conn)
            .annotation_group(annotation_group)
            .only(AnnotationGroupSampleSelect::SampleName)
            .load()?)
    }

    pub fn query_by_sample(
        conn: &GraphConnection,
        sample_name: &str,
        history_ref: Option<&str>,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let samples = AnnotationGroupSample::select(conn)
            .sample_name(sample_name)
            .with_ref(history_ref);
        Ok(Annotation::select(conn)
            .join_filtered_on(
                AnnotationSelect::Group,
                AnnotationGroupSampleSelect::AnnotationGroup,
                samples,
            )
            .load()?)
    }

    pub fn query_by_group(
        conn: &GraphConnection,
        group: &str,
        history_ref: Option<&str>,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let select = Annotation::select(conn).group(group).with_ref(history_ref);
        Ok(select.load()?)
    }

    pub fn query_by_group_and_block_group(
        conn: &GraphConnection,
        group: &str,
        block_group_id: &HashId,
        history_ref: Option<&str>,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let accessions = Accession::select(conn)
            .block_group_id(*block_group_id)
            .with_ref(history_ref);
        Ok(Annotation::select(conn)
            .group(group)
            .join_filtered_on(
                AnnotationSelect::AccessionId,
                AccessionSelect::Id,
                accessions,
            )
            .load()?)
    }

    /// List every annotation on a block group, following the sample lineage.
    ///
    /// Collects annotations attached to the block group named `block_group_name`
    /// in `collection_name`, considering `sample_name` together with all of its
    /// ancestor samples. This mirrors how a block group inherits its parents'
    /// annotations. Results are de-duplicated by annotation id and ordered from
    /// the closest sample in the lineage outward. Two annotations that share a
    /// name but come from different samples (e.g. a parent and child both
    /// annotating "mreB") are both included, since each is a distinct
    /// annotation record; unlike [`RegionResolver::resolve`], this is a listing
    /// operation and does not need to disambiguate by name.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - Collection the block group belongs to.
    /// * `sample_name` - Sample whose lineage is walked, including the sample
    ///   itself.
    /// * `block_group_name` - Name shared by the block group across the lineage.
    ///
    /// # Errors
    ///
    /// Returns [`AnnotationError::DatabaseError`] if the query fails.
    pub fn query_with_lineage(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        block_group_name: &str,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let query = "WITH RECURSIVE visible_samples(name, depth, path) AS (
                 SELECT ?2, 0, ',' || ?2 || ','
                 UNION ALL
                 SELECT sl.parent_sample_name,
                        visible.depth + 1,
                        visible.path || sl.parent_sample_name || ','
                 FROM sample_lineage sl
                 JOIN visible_samples visible ON sl.child_sample_name = visible.name
                 WHERE instr(visible.path, ',' || sl.parent_sample_name || ',') = 0
             )
             SELECT a.id,
                    a.name,
                    a.annotation_group,
                    a.accession_id,
                    a.extra \
             FROM annotations a \
             JOIN accessions acc ON a.accession_id = acc.id \
             JOIN block_groups bg ON acc.block_group_id = bg.id \
             JOIN visible_samples visible ON bg.sample_name = visible.name \
             WHERE bg.collection_name = ?1 \
               AND bg.name = ?3
             GROUP BY a.id, a.name, a.annotation_group, a.accession_id, a.extra
             ORDER BY min(visible.depth), a.name, a.id";
        Ok(query_models::<Annotation, _>(
            conn,
            query,
            params![collection_name, sample_name, block_group_name],
        )?)
    }

    pub(crate) fn resolve_candidates(
        region: &Region,
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<Vec<Self>, AnnotationError> {
        query_models::<Annotation, _>(
            conn,
            "WITH RECURSIVE visible_samples(name, depth, path) AS (
                 SELECT ?2, 0, ',' || ?2 || ','
                 UNION ALL
                 SELECT sl.parent_sample_name,
                        visible.depth + 1,
                        visible.path || sl.parent_sample_name || ','
                 FROM sample_lineage sl
                 JOIN visible_samples visible ON sl.child_sample_name = visible.name
                 WHERE instr(visible.path, ',' || sl.parent_sample_name || ',') = 0
             ),
             matching_annotations AS (
                 SELECT a.id,
                        a.name,
                        a.annotation_group,
                        a.accession_id,
                        a.extra,
                        min(visible.depth) AS depth \
                 FROM annotations a \
                 JOIN accessions acc ON a.accession_id = acc.id \
                 JOIN block_groups bg ON acc.block_group_id = bg.id \
                 JOIN visible_samples visible ON bg.sample_name = visible.name \
                 WHERE bg.collection_name = ?1 \
                   AND lower(a.name) = lower(?3)
                 GROUP BY a.id, a.name, a.annotation_group, a.accession_id, a.extra
             )
             SELECT id, name, annotation_group, accession_id, extra \
             FROM matching_annotations \
             WHERE depth = (SELECT min(depth) FROM matching_annotations)",
            params![collection_name, sample_name, region.name],
        )
        .map_err(AnnotationError::from)
    }

    pub fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<Arc<IntervalTree<i64, NodeIntervalBlock>>, AnnotationError> {
        IntervalTreeSource::intervaltree(self, conn)
    }

    pub fn length(&self, conn: &GraphConnection) -> Result<i64, AnnotationError> {
        let interval_tree = self.intervaltree(conn)?;
        crate::accession::intervaltree_length(&interval_tree).map_err(Into::into)
    }
}

impl IntervalTreeSource for Annotation {
    type Error = AnnotationError;

    fn cached_interval_tree(&self) -> Option<&Arc<IntervalTree<i64, NodeIntervalBlock>>> {
        self.cached_interval_tree.as_ref()
    }

    fn set_cached_interval_tree(
        &mut self,
        interval_tree: Option<Arc<IntervalTree<i64, NodeIntervalBlock>>>,
    ) {
        self.cached_interval_tree = interval_tree;
    }

    fn load_interval_tree(
        &self,
        conn: &GraphConnection,
    ) -> Result<Arc<IntervalTree<i64, NodeIntervalBlock>>, Self::Error> {
        let accession = self.get_accession(conn)?;
        accession.intervaltree(conn).map_err(Into::into)
    }
}

impl RegionResolver for Annotation {
    type Connection = GraphConnection;
    type Error = AnnotationError;

    fn resolve(
        region: &Region,
        conn: &Self::Connection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<Self, RegionResolutionError<Self::Error>> {
        let matches = Self::resolve_candidates(region, conn, collection_name, sample_name)?;

        match matches.len() {
            0 => Err(RegionResolutionError::NotFound(region.name.clone())),
            1 => Ok(matches.into_iter().next().unwrap()),
            _ => Err(RegionResolutionError::Ambiguous(format!(
                "multiple annotations named {}",
                region.name
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize, ModelSelect)]
#[model_select(
    table = "annotation_group_samples",
    default_sort(annotation_group = "asc", sample_name = "asc")
)]
pub struct AnnotationGroupSample {
    #[model_select(primary_key)]
    pub annotation_group: String,
    #[model_select(primary_key)]
    pub sample_name: String,
}

impl<'a> Capnp<'a> for AnnotationGroupSample {
    type Builder = annotation_group_sample::Builder<'a>;
    type Reader = annotation_group_sample::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_annotation_group(&self.annotation_group);
        builder.set_sample_name(&self.sample_name);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let annotation_group = reader.get_annotation_group().unwrap().to_string().unwrap();
        let sample_name = reader.get_sample_name().unwrap().to_string().unwrap();
        AnnotationGroupSample {
            annotation_group,
            sample_name,
        }
    }
}

impl AnnotationGroupSample {
    pub fn create(
        conn: &GraphConnection,
        annotation_group: &str,
        sample_name: &str,
    ) -> Result<(), AnnotationError> {
        AnnotationGroup::get_or_create(conn, annotation_group)?;
        let query = "INSERT INTO annotation_group_samples (annotation_group, sample_name) VALUES (?1, ?2) ON CONFLICT(annotation_group, sample_name) DO NOTHING;";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![annotation_group, sample_name])?;
        Ok(())
    }

    pub fn delete(
        conn: &GraphConnection,
        annotation_group: &str,
        sample_name: &str,
    ) -> Result<(), AnnotationError> {
        let query = "DELETE FROM annotation_group_samples WHERE annotation_group = ?1 AND sample_name = ?2;";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![annotation_group, sample_name])?;
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum AnnotationGroupError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

#[derive(Debug, Error)]
pub enum AnnotationFileError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("File addition error: {0}")]
    FileAdditionError(#[from] FileAdditionError),
    #[error("Annotation file I/O error: {0}")]
    IoError(#[from] io::Error),
    #[error("Unable to sort or index annotation file: {0}")]
    GaiError(#[from] gai::Error),
    #[error("Operation error: {0}")]
    OperationError(#[from] OperationError),
    #[error(
        "Unable to detect annotation file format from the file extension. Use --format to specify it explicitly."
    )]
    MissingFileExtension,
    #[error("Unsupported annotation file type: {0}")]
    UnsupportedFileType(String),
}

pub fn parse_annotation_file_type(value: &str) -> Result<FileTypes, AnnotationFileError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "gff3" | "gff" => Ok(FileTypes::Gff3),
        "bed" => Ok(FileTypes::Bed),
        "genbank" | "gb" => Ok(FileTypes::GenBank),
        other => Err(AnnotationFileError::UnsupportedFileType(other.to_string())),
    }
}

pub fn annotation_file_extension(path: &str) -> Option<String> {
    let path = Path::new(path);
    let mut ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    if matches!(ext.as_deref(), Some("gz") | Some("bgz") | Some("bgzf")) {
        ext = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| Path::new(stem).extension().and_then(|ext| ext.to_str()))
            .map(|ext| ext.to_ascii_lowercase());
    }
    ext
}

pub fn annotation_index_file_path(
    workspace: &Workspace,
    path: &str,
    explicit_index_path: Option<&str>,
) -> Option<String> {
    if let Some(index_path) = explicit_index_path {
        return Some(index_path.to_string());
    }

    let mut candidates = vec![format!("{path}.csi"), format!("{path}.tbi")];
    let path_buf = PathBuf::from(path);
    if let Some(extension) = path_buf.extension().and_then(|ext| ext.to_str()) {
        for suffix in ["csi", "tbi"] {
            let mut extension_candidate = path_buf.clone();
            extension_candidate.set_extension(format!("{extension}.{suffix}"));
            let extension_candidate = extension_candidate.to_string_lossy().to_string();
            if !candidates
                .iter()
                .any(|candidate| candidate == &extension_candidate)
            {
                candidates.push(extension_candidate);
            }
        }
    }

    for candidate in candidates {
        let exists = if Path::new(&candidate).is_absolute() {
            Path::new(&candidate).exists()
        } else {
            workspace
                .repo_root()
                .ok()
                .is_some_and(|repo_root| repo_root.join(&candidate).exists())
        };
        if exists {
            return Some(candidate);
        }
    }

    None
}

pub fn add_annotation(
    context: &DbContext,
    collection: &str,
    name: &str,
    group: Option<&str>,
    sample: &str,
    region: &str,
) -> Result<OperationSummary, AddAnnotationError> {
    let graph_conn = context.graph().conn();
    let parsed_region = Region::parse(region)?;
    let resolved_region = crate::region::resolve(&parsed_region, graph_conn, collection, sample)?;
    let spans = AccessionSpan::from_resolved_region(
        graph_conn,
        context.workspace(),
        &resolved_region,
        None,
    )?;

    let accession = Accession::get_or_create(
        graph_conn,
        &NewAccession {
            name: name.to_string(),
            block_group_id: resolved_region.block_group.id,
            parent_accession_id: None,
            spans,
        },
    )?;

    let annotation_group = group.unwrap_or("default");
    let annotation =
        Annotation::get_or_create(graph_conn, name, annotation_group, &accession.id, None)?;
    AnnotationGroupSample::create(graph_conn, &annotation.group, sample)?;

    Ok(OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: format!("add annotation {name}"),
        },
        format!("add annotation {name}"),
    ))
}

/// Optional content checksums collected while annotation assets were already being streamed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AnnotationFileChecksumOverrides {
    /// Checksum of the annotation file.
    pub annotation: Option<Sha256Hash>,
    /// Checksum of the annotation index.
    pub index: Option<Sha256Hash>,
}

/// A stage reported while preparing a local GFF3 or BED annotation file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnnotationFileProgressStage {
    /// Sort source records before creating offset-based indexes.
    Sorting,
    /// Compress the sorted source into the retained BGZF asset.
    Compressing,
    /// Build the searchable attribute index for the retained source.
    BuildingIndex,
}

/// Options shared by annotation-file addition and its optional progress callback.
pub struct AnnotationFileAddOptions<'a> {
    /// Checksums collected while reading annotation assets.
    pub checksum_overrides: AnnotationFileChecksumOverrides,
    /// Called immediately before each local GFF3 or BED preparation stage.
    pub progress: Option<&'a mut dyn FnMut(AnnotationFileProgressStage)>,
}

fn report_annotation_file_progress(
    progress: &mut Option<&mut dyn FnMut(AnnotationFileProgressStage)>,
    stage: AnnotationFileProgressStage,
) {
    if let Some(callback) = progress.as_mut() {
        callback(stage);
    }
}

// GAI finishes sorting before it emits sorted rows, so the first BGZF write marks the transition.
struct ReportCompressionOnFirstWrite<W, F> {
    writer: W,
    on_first_write: Option<F>,
}

impl<W, F> ReportCompressionOnFirstWrite<W, F> {
    fn new(writer: W, on_first_write: F) -> Self {
        Self {
            writer,
            on_first_write: Some(on_first_write),
        }
    }
}

impl<W: Write, F: FnMut()> Write for ReportCompressionOnFirstWrite<W, F> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if !buffer.is_empty()
            && let Some(mut on_first_write) = self.on_first_write.take()
        {
            on_first_write();
        }

        self.writer.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
fn prepare_local_sorted_annotation(
    workspace: &Workspace,
    path: &str,
    file_type: FileTypes,
    checksum_override: Option<Sha256Hash>,
    progress: &mut Option<&mut dyn FnMut(AnnotationFileProgressStage)>,
) -> Result<(FileAddition, FileAddition, FileAddition), AnnotationFileError> {
    let cache_dir = workspace
        .find_cache_dir()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "workspace cache directory is unavailable",
            )
        })?
        .join("annotation-indexes");
    fs::create_dir_all(&cache_dir)?;

    let asset_uri = <dyn AssetUri>::new(workspace, path);
    if let Some(expected_checksum) = checksum_override {
        let source_checksum = calculate_reader_checksum(asset_uri.reader(workspace)?)?;
        if source_checksum != expected_checksum {
            return Err(FileAdditionError::ChecksumError(
                "local source checksum does not match the provided checksum".to_string(),
            )
            .into());
        }
    }

    let source_reader = asset_uri.reader(workspace)?;
    let (compression_type, replayed_reader) =
        CompressionType::sniff(source_reader).map_err(FileAdditionError::FileReadError)?;
    let source_reader: Box<dyn Read> = match compression_type {
        CompressionType::Plain => Box::new(replayed_reader),
        CompressionType::Gzip | CompressionType::Bgzf => {
            Box::new(MultiGzDecoder::new(replayed_reader))
        }
    };
    let asset_dir = workspace
        .asset_dir()
        .map_err(FileAdditionError::ConfigError)?;
    fs::create_dir_all(&asset_dir)?;
    let mut sorted_source = TempFileBuilder::new()
        .suffix(&format!(".{}.bgz", FileTypes::suffix(file_type)))
        .tempfile_in(&asset_dir)?;
    report_annotation_file_progress(progress, AnnotationFileProgressStage::Sorting);
    let format = match file_type {
        FileTypes::Gff3 => gai::SortFormat::Gff,
        FileTypes::Bed => gai::SortFormat::Bed,
        other => {
            return Err(AnnotationFileError::UnsupportedFileType(
                other.as_str().to_string(),
            ));
        }
    };

    // GAI sorts, writes a .bgz and returns an index all in one pass.
    let mut checksummed_writer = ChecksummedWriter::new(sorted_source.as_file_mut());
    let coordinate_index = {
        let mut compressed_writer =
            ReportCompressionOnFirstWrite::new(&mut checksummed_writer, || {
                report_annotation_file_progress(progress, AnnotationFileProgressStage::Compressing)
            });
        let mut bgzf_writer = noodles::bgzf::io::Writer::new(&mut compressed_writer);
        let coordinate_index = match gai::sort_bgzf_with_csi(
            BufReader::new(source_reader),
            &mut bgzf_writer,
            format,
            true,
        ) {
            Ok(coordinate_index) => coordinate_index,
            Err(error) => {
                bgzf_writer.into_inner();
                return Err(error.into());
            }
        };
        bgzf_writer.finish()?.flush()?;
        coordinate_index
    };
    let archive_checksum = checksummed_writer.checksum();
    let materialized_checksum =
        calculate_reader_checksum(noodles::bgzf::io::Reader::new(sorted_source.reopen()?))?;
    let file_addition = FileAddition::prepare_from_source(
        workspace,
        path,
        file_type,
        FileSource::StagedBgzf {
            path: sorted_source.path().to_path_buf(),
            archive_checksum,
            materialized_checksum,
        },
        true,
    )?;

    let sidecar_temp_dir = TempDir::new_in(&cache_dir)?;
    let coordinate_index_path = sidecar_temp_dir.path().join("coordinate.csi");
    let mut coordinate_index_writer =
        noodles::csi::io::Writer::new(File::create(&coordinate_index_path)?);
    coordinate_index_writer.write_index(&coordinate_index)?;
    coordinate_index_writer.into_inner().finish()?.flush()?;

    let source_path = asset_dir.join(
        file_addition
            .hashed_filename()
            .expect("should have a checksum for retained local annotations"),
    );
    let index_file_addition = FileAddition::prepare(
        workspace,
        &coordinate_index_path.to_string_lossy(),
        FileTypes::Tabix,
        None,
    )?;
    report_annotation_file_progress(progress, AnnotationFileProgressStage::BuildingIndex);
    let attribute_index_path = sidecar_temp_dir.path().join("attributes.gai");
    let name_index_options = match format {
        gai::SortFormat::Gff => {
            gai::NameIndexOptions::new(["Name", "name", "gene", "gene_name", "db_xref"], false)?
        }
        gai::SortFormat::Bed => gai::NameIndexOptions::bed(false),
    };
    gai::build_name_index_with_options(
        &source_path,
        &coordinate_index_path,
        &attribute_index_path,
        &name_index_options,
        &gai::BuildOptions::default(),
    )?;
    let attribute_index_file_addition = FileAddition::prepare(
        workspace,
        &attribute_index_path.to_string_lossy(),
        FileTypes::None,
        None,
    )?;

    Ok((
        file_addition,
        index_file_addition,
        attribute_index_file_addition,
    ))
}

/// Adds an annotation asset and its available indexes to operation history.
///
/// Callers pass checksums only when the corresponding remote streams were already consumed for
/// annotation work. This keeps operation creation independent of remote credentials while local
/// annotation files are sorted, retained, and hashed during preparation.
pub fn add_annotation_file(
    context: &DbContext,
    path: &str,
    format: Option<&str>,
    index: Option<&str>,
    name: Option<&str>,
    message: Option<&str>,
    checksum_overrides: AnnotationFileChecksumOverrides,
) -> Result<DoltHashId, AnnotationFileError> {
    add_annotation_file_with_progress(
        context,
        path,
        format,
        index,
        name,
        message,
        AnnotationFileAddOptions {
            checksum_overrides,
            progress: None,
        },
    )
}

/// Adds an annotation file and reports local GFF3/BED preparation stages when requested.
pub fn add_annotation_file_with_progress(
    context: &DbContext,
    path: &str,
    format: Option<&str>,
    index: Option<&str>,
    name: Option<&str>,
    message: Option<&str>,
    options: AnnotationFileAddOptions<'_>,
) -> Result<DoltHashId, AnnotationFileError> {
    let AnnotationFileAddOptions {
        checksum_overrides,
        mut progress,
    } = options;
    let workspace = context.workspace();
    let graph_conn = context.graph().conn();

    let file_type = match format {
        Some(format) => parse_annotation_file_type(format)?,
        None => {
            let ext =
                annotation_file_extension(path).ok_or(AnnotationFileError::MissingFileExtension)?;
            parse_annotation_file_type(&ext)?
        }
    };
    let name = name.or_else(|| Path::new(path).file_name().and_then(|value| value.to_str()));
    let name_value = name.unwrap_or_default();
    let (file_addition, logical_path, sidecars) =
        if matches!(file_type, FileTypes::Gff3 | FileTypes::Bed)
            && LocalAssetUri::is_local_path_or_file_uri(path)
        {
            // Sorting changes compressed offsets, so local GFF/BED sidecars are always rebuilt against
            // the retained BGZF bytes. A supplied coordinate index describes the original input bytes.
            let (file_addition, index_file_addition, attribute_index_file_addition) =
                prepare_local_sorted_annotation(
                    workspace,
                    path,
                    file_type,
                    checksum_overrides.annotation,
                    &mut progress,
                )?;
            let logical_path =
                OperationFile::storage_file_path(workspace, path, file_addition.checksum.as_ref())?;
            let filename = Path::new(&logical_path)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("annotation");
            let sidecars = vec![
                (
                    index_file_addition,
                    AssetRole::AnnotationIndex,
                    format!("{logical_path}.csi"),
                    format!("{filename}.csi"),
                ),
                (
                    attribute_index_file_addition,
                    AssetRole::Other(ANNOTATION_ATTRIBUTE_INDEX_ROLE.to_string()),
                    format!("{logical_path}.gai"),
                    format!("{filename}.gai"),
                ),
            ];
            (file_addition, logical_path, sidecars)
        } else {
            let file_addition =
                FileAddition::prepare(workspace, path, file_type, checksum_overrides.annotation)?;
            let logical_path =
                OperationFile::storage_file_path(workspace, path, file_addition.checksum.as_ref())?;
            let mut sidecars = Vec::new();
            if let Some(index_path) = annotation_index_file_path(workspace, path, index) {
                let index_file_type = FileTypes::infer_from_path(&index_path);
                let index_file_addition = FileAddition::prepare(
                    workspace,
                    &index_path,
                    index_file_type,
                    checksum_overrides.index,
                )?;
                let index_logical_path = OperationFile::storage_file_path(
                    workspace,
                    &index_path,
                    index_file_addition.checksum.as_ref(),
                )?;
                let index_name = Path::new(&index_path)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| index_path.clone());
                sidecars.push((
                    index_file_addition,
                    AssetRole::AnnotationIndex,
                    index_logical_path,
                    index_name,
                ));
            }
            (file_addition, logical_path, sidecars)
        };
    let created_on = chrono::Utc::now()
        .timestamp_nanos_opt()
        .expect("should create annotation asset timestamp");
    let annotation_asset_ref = AssetRef::from_file_addition(
        &file_addition,
        AssetRole::Annotation,
        Some(&logical_path),
        name,
        None,
        created_on,
    );
    let annotation_asset_ref_id = annotation_asset_ref.id;
    let index_file_addition_ids = sidecars
        .iter()
        .map(|(file_addition, _, _, _)| file_addition.id.to_string())
        .collect::<Vec<_>>()
        .join(":");
    let log_id = HashId(calculate_hash(&format!(
        "{file_addition_id}:{name_value}:{index_file_addition_ids}",
        file_addition_id = file_addition.id
    )));
    let summary = message
        .map(str::to_string)
        .unwrap_or_else(|| format!("Add annotation file {path}"));
    let mut asset_refs = vec![annotation_asset_ref];
    for (file_addition, role, logical_path, index_name) in &sidecars {
        asset_refs.push(AssetRef::from_file_addition(
            file_addition,
            role.clone(),
            Some(logical_path),
            Some(index_name),
            Some(&annotation_asset_ref_id),
            created_on,
        ));
    }

    track_asset_refs(
        graph_conn,
        Some(&log_id),
        &OperationKind::AnnotationFile,
        &summary,
        &asset_refs,
    )?;
    let history_store = DoltHistoryStore::new_with_config(graph_conn, context.config().conn());
    if history_store.status()?.is_empty() {
        return Err(OperationError::NoChanges.into());
    }

    let commit_hash = history_store.commit_all(&summary)?;

    Ok(commit_hash)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        fs::{self, File},
        sync::Arc,
    };

    use gen_core::{HashId, NodeIntervalBlock, Strand, region::RegionResolutionError};
    use noodles::bgzf;

    use super::*;
    use crate::{
        assets::{AssetRef, AssetRole, OperationAsset, OperationLog},
        block_group::BlockGroup,
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        errors::OperationError,
        operations::{
            calculate_file_checksum, calculate_reader_checksum, commit_operation_summary,
        },
        path::Path,
        sample::Sample,
        sample_lineage::SampleLineage,
        test_helpers::{create_bg, get_connection, setup_block_group, setup_gen},
    };

    mod region_resolver {
        use super::*;

        #[test]
        fn resolves_annotation_by_name_case_insensitively() {
            let conn = get_connection(None).unwrap();
            let (_block_group_id, path) = setup_block_group(&conn);
            let accession = BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5).unwrap();
            let annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();
            assert!(annotation.cached_interval_tree().is_none());
            assert!(
                annotation
                    .intervaltree(&conn)
                    .expect("should load the persisted annotation tree")
                    .query(0..5)
                    .next()
                    .is_some()
            );

            let region = Region::parse("MREB").unwrap();
            let resolved = Annotation::resolve(&region, &conn, "test", "test").unwrap();
            assert_eq!(resolved.id, annotation.id);
        }

        #[test]
        fn returns_not_found_for_missing_annotation() {
            let conn = get_connection(None).unwrap();
            let (_block_group_id, _path) = setup_block_group(&conn);

            let region = Region::parse("missing").unwrap();
            let err = Annotation::resolve(&region, &conn, "test", "test").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::NotFound(name) if name == "missing"
            ));
        }

        #[test]
        fn returns_ambiguous_for_multiple_matching_annotations() {
            let conn = get_connection(None).unwrap();
            let (_block_group_id, path) = setup_block_group(&conn);
            let accession = BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5).unwrap();
            let _ = Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

            let other_block_group = create_bg(&conn, "test", "test", "other");
            let edge_ids = Path::edges_for_path(&conn, &path.id, None)
                .into_iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>();
            let block_group_edges = edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: other_block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>();
            crate::block_group_edge::BlockGroupEdge::bulk_create(&conn, &block_group_edges);
            let other_path =
                Path::create(&conn, "other-path", &other_block_group.id, &edge_ids).unwrap();
            let other_accession =
                BlockGroup::add_accession(&conn, &other_path, "ann-accession-2", 0, 5).unwrap();
            let _ = Annotation::get_or_create(&conn, "MREB", "genes", &other_accession.id, None)
                .unwrap();

            let region = Region::parse("mreB").unwrap();
            let err = Annotation::resolve(&region, &conn, "test", "test").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::Ambiguous(name) if name == "multiple annotations named mreB"
            ));
        }

        #[test]
        fn resolves_closest_annotation_in_sample_lineage() {
            let conn = get_connection(None).unwrap();
            let (_parent_block_group_id, parent_path) = setup_block_group(&conn);
            let parent_accession =
                BlockGroup::add_accession(&conn, &parent_path, "parent-accession", 0, 5).unwrap();
            let _parent_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &parent_accession.id, None)
                    .unwrap();

            let child_block_group = create_bg(&conn, "test", "child", "chr1");
            let edge_ids = Path::edges_for_path(&conn, &parent_path.id, None)
                .into_iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>();
            let block_group_edges = edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: child_block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>();
            crate::block_group_edge::BlockGroupEdge::bulk_create(&conn, &block_group_edges);
            let child_path =
                Path::create(&conn, "child-path", &child_block_group.id, &edge_ids).unwrap();
            let child_accession =
                BlockGroup::add_accession(&conn, &child_path, "child-accession", 0, 5).unwrap();
            let child_annotation =
                Annotation::get_or_create(&conn, "MREB", "genes", &child_accession.id, None)
                    .unwrap();
            SampleLineage::create(&conn, "test", "child").unwrap();

            let region = Region::parse("mreB").unwrap();
            let resolved = Annotation::resolve(&region, &conn, "test", "child").unwrap();

            assert_eq!(resolved.id, child_annotation.id);
        }

        #[test]
        fn falls_back_to_parent_annotation_when_child_has_none() {
            let conn = get_connection(None).unwrap();
            let (_parent_block_group_id, parent_path) = setup_block_group(&conn);
            let parent_accession =
                BlockGroup::add_accession(&conn, &parent_path, "parent-only-accession", 0, 5)
                    .unwrap();
            let parent_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &parent_accession.id, None)
                    .unwrap();

            let child_block_group = create_bg(&conn, "test", "child", "chr1");
            let edge_ids = Path::edges_for_path(&conn, &parent_path.id, None)
                .into_iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>();
            let block_group_edges = edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: child_block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>();
            crate::block_group_edge::BlockGroupEdge::bulk_create(&conn, &block_group_edges);
            let _child_path =
                Path::create(&conn, "child-path", &child_block_group.id, &edge_ids).unwrap();
            SampleLineage::create(&conn, "test", "child").unwrap();

            let region = Region::parse("mreB").unwrap();
            let resolved = Annotation::resolve(&region, &conn, "test", "child").unwrap();

            assert_eq!(resolved.id, parent_annotation.id);
        }
    }

    mod lineage_listing {
        use super::*;

        /// Attach a block group that shares `parent_path`'s edges, give it its
        /// own path and accession, and annotate it. Returns the annotation.
        fn annotate_block_group(
            conn: &GraphConnection,
            parent_path: &Path,
            collection_name: &str,
            sample_name: &str,
            block_group_name: &str,
            accession_name: &str,
            annotation_name: &str,
        ) -> Annotation {
            let block_group = create_bg(conn, collection_name, sample_name, block_group_name);
            let edge_ids = Path::edges_for_path(conn, &parent_path.id, None)
                .into_iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>();
            let block_group_edges = edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>();
            BlockGroupEdge::bulk_create(conn, &block_group_edges);
            let path = Path::create(conn, accession_name, &block_group.id, &edge_ids).unwrap();
            let accession = BlockGroup::add_accession(conn, &path, accession_name, 0, 5).unwrap();
            Annotation::get_or_create(conn, annotation_name, "genes", &accession.id, None).unwrap()
        }

        #[test]
        fn test_lists_annotations_on_block_group() {
            let conn = get_connection(None).unwrap();
            let (_block_group_id, path) = setup_block_group(&conn);
            let accession = BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5).unwrap();
            let annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

            let related_accession = annotation
                .get_accession(&conn)
                .expect("should load persisted annotation accession");
            let cached_accession = annotation
                .cached_accession
                .snapshot()
                .expect("should cache persisted annotation accession");
            assert!(Arc::ptr_eq(&related_accession, &cached_accession));
            assert!(Arc::ptr_eq(
                &related_accession,
                &annotation
                    .get_accession(&conn)
                    .expect("should reuse cached annotation accession")
            ));

            let listed = Annotation::query_with_lineage(&conn, "test", "test", "chr1").unwrap();

            assert_eq!(listed, vec![annotation]);
        }

        #[test]
        fn test_includes_parent_annotations_via_lineage() {
            let conn = get_connection(None).unwrap();
            let (_parent_block_group_id, parent_path) = setup_block_group(&conn);
            let parent_accession =
                BlockGroup::add_accession(&conn, &parent_path, "parent-accession", 0, 5).unwrap();
            let parent_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &parent_accession.id, None)
                    .unwrap();

            let child_annotation = annotate_block_group(
                &conn,
                &parent_path,
                "test",
                "child",
                "chr1",
                "child-accession",
                "ftsZ",
            );
            SampleLineage::create(&conn, "test", "child").unwrap();

            let listed = Annotation::query_with_lineage(&conn, "test", "child", "chr1").unwrap();

            let ids: HashSet<HashId> = listed.iter().map(|a| a.id).collect();
            assert_eq!(
                ids,
                HashSet::from([parent_annotation.id, child_annotation.id])
            );
        }

        #[test]
        fn test_excludes_annotations_on_other_block_group_names() {
            let conn = get_connection(None).unwrap();
            let (_block_group_id, path) = setup_block_group(&conn);
            let accession =
                BlockGroup::add_accession(&conn, &path, "chr1-accession", 0, 5).unwrap();
            let chr1_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

            let _chr2_annotation = annotate_block_group(
                &conn,
                &path,
                "test",
                "test",
                "chr2",
                "chr2-accession",
                "ftsZ",
            );

            let listed = Annotation::query_with_lineage(&conn, "test", "test", "chr1").unwrap();

            assert_eq!(listed, vec![chr1_annotation]);
        }

        #[test]
        fn test_excludes_annotations_from_samples_outside_the_lineage() {
            let conn = get_connection(None).unwrap();
            let (_block_group_id, path) = setup_block_group(&conn);
            let accession =
                BlockGroup::add_accession(&conn, &path, "test-accession", 0, 5).unwrap();
            let test_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

            let _unrelated_annotation = annotate_block_group(
                &conn,
                &path,
                "test",
                "unrelated",
                "chr1",
                "unrelated-accession",
                "ftsZ",
            );

            let listed = Annotation::query_with_lineage(&conn, "test", "test", "chr1").unwrap();

            assert_eq!(listed, vec![test_annotation]);
        }

        #[test]
        fn test_keeps_both_when_parent_and_child_share_an_annotation_name() {
            let conn = get_connection(None).unwrap();
            let (_parent_block_group_id, parent_path) = setup_block_group(&conn);
            let parent_accession =
                BlockGroup::add_accession(&conn, &parent_path, "parent-accession", 0, 5).unwrap();
            let parent_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &parent_accession.id, None)
                    .unwrap();

            let child_annotation = annotate_block_group(
                &conn,
                &parent_path,
                "test",
                "child",
                "chr1",
                "child-accession",
                "mreB",
            );
            SampleLineage::create(&conn, "test", "child").unwrap();

            let listed = Annotation::query_with_lineage(&conn, "test", "child", "chr1").unwrap();

            // Same name, distinct annotation ids: listing keeps both rather than
            // collapsing by name. `resolve` would pick only the child's here
            // since it's the closer match (depth 0 vs. depth 1) and error on tie.
            assert_eq!(listed, vec![child_annotation, parent_annotation]);
        }
    }

    #[test]
    fn annotation_groups_all_lists_groups_in_name_order() {
        let conn = get_connection(None).unwrap();
        AnnotationGroup::create(&conn, "zebra").unwrap();
        AnnotationGroup::create(&conn, "ant").unwrap();

        let groups = AnnotationGroup::all(&conn).expect("should load annotation groups");

        assert_eq!(
            groups,
            vec![
                AnnotationGroup { name: "ant".into() },
                AnnotationGroup {
                    name: "zebra".into()
                }
            ]
        );
    }

    #[test]
    fn create_annotation_with_samples() {
        let conn = get_connection(None).unwrap();
        let (_block_group_id, path) = setup_block_group(&conn);

        let _ = Sample::create(
            &conn,
            crate::sample::NewSample {
                name: "sample-1",
                is_reference: false,
            },
        )
        .unwrap();
        let _ = Sample::create(
            &conn,
            crate::sample::NewSample {
                name: "sample-2",
                is_reference: false,
            },
        )
        .unwrap();
        let accession = BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5).unwrap();

        let annotations = [NewAnnotation {
            name: "gene-a",
            accession_id: accession.id,
            extra: None,
        }];
        Annotation::bulk_create(&conn, "project-tracks", &annotations).unwrap();
        AnnotationGroupSample::create(&conn, "project-tracks", "sample-1").unwrap();
        AnnotationGroupSample::create(&conn, "project-tracks", "sample-2").unwrap();
        let annotation = Annotation::query_by_group(&conn, "project-tracks", None)
            .unwrap()
            .pop()
            .expect("should create annotation");
        Annotation::bulk_create(&conn, "project-tracks", &annotations).unwrap();

        assert_eq!(
            Annotation::query_by_group(&conn, "project-tracks", None).unwrap(),
            vec![annotation.clone()]
        );

        let mut samples = Annotation::get_samples(&conn, &annotation.group).unwrap();
        samples.sort();
        assert_eq!(
            samples,
            vec!["sample-1".to_string(), "sample-2".to_string()]
        );

        let by_sample = Annotation::query_by_sample(&conn, "sample-1", None).unwrap();
        assert_eq!(by_sample.len(), 1);
        assert_eq!(by_sample[0], annotation);

        let by_group = Annotation::query_by_group(&conn, "project-tracks", None).unwrap();
        assert_eq!(by_group, vec![annotation]);
    }

    #[test]
    fn deserialize_annotation_extra_defaults_missing_fields() {
        // Ensure that if we add new fields to the AnnotationExtra struct, like new_annotation_format we still parse old ones
        let extra = deserialize_annotation_extra(Some(r#"{"genbank":{"kind":"CDS"}}"#.to_string()))
            .unwrap()
            .unwrap();

        assert_eq!(
            extra.genbank,
            Some(GenBankExtra {
                kind: "CDS".to_string(),
                ..GenBankExtra::default()
            })
        );
        assert_eq!(extra.gff, None);
        assert_eq!(extra.bed, None);
    }

    #[test]
    fn deserialize_annotation_extra_ignores_unknown_fields() {
        // ensure if we drop a format, removed_top_level key here, we still parse
        let extra = deserialize_annotation_extra(Some(
            r#"{"genbank":{"kind":"CDS","legacy_field":"value"},"removed_top_level":true}"#
                .to_string(),
        ))
        .unwrap()
        .unwrap();

        assert_eq!(
            extra.genbank,
            Some(GenBankExtra {
                kind: "CDS".to_string(),
                ..GenBankExtra::default()
            })
        );
        assert_eq!(extra.gff, None);
        assert_eq!(extra.bed, None);
    }

    #[test]
    fn deserialize_and_reserialize_annotation_extra_keeps_compatible_fields() {
        // Ensure that if we parse an older field, we can still reserialize it with the compatible fields kept.
        let extra = deserialize_annotation_extra(Some(
            r#"{"genbank":{"kind":"CDS","legacy_field":"value"},"removed_top_level":true}"#
                .to_string(),
        ))
        .unwrap()
        .unwrap();

        let reserialized = serialize_annotation_extra(Some(&extra)).unwrap();
        let reparsed: AnnotationExtra = serde_json::from_str(&reserialized).unwrap();

        assert_eq!(
            reparsed.genbank,
            Some(GenBankExtra {
                kind: "CDS".to_string(),
                ..GenBankExtra::default()
            })
        );
        assert_eq!(reparsed.gff, None);
        assert_eq!(reparsed.bed, None);
    }

    #[test]
    fn parse_annotation_file_type_values() {
        assert_eq!(parse_annotation_file_type("gff3").unwrap(), FileTypes::Gff3);
        assert_eq!(parse_annotation_file_type("GFF").unwrap(), FileTypes::Gff3);
        assert_eq!(parse_annotation_file_type("bed").unwrap(), FileTypes::Bed);
        assert_eq!(
            parse_annotation_file_type("GenBank").unwrap(),
            FileTypes::GenBank
        );
        assert_eq!(
            parse_annotation_file_type("gb").unwrap(),
            FileTypes::GenBank
        );
        let err = parse_annotation_file_type("bam").unwrap_err();
        assert!(matches!(err, AnnotationFileError::UnsupportedFileType(_)));
    }

    #[test]
    fn add_annotation_creates_annotation() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(graph_conn);
        let _ = setup_block_group(graph_conn);

        let operation_summary = add_annotation(
            &context,
            "test",
            "gene-a",
            Some("track-1"),
            "test",
            "chr1:1-5",
        )
        .unwrap();
        let commit_hash = commit_operation_summary(&context, &operation_summary).unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));
        let mut operation_logs = OperationLog::all(graph_conn).expect("should load operation logs");
        operation_logs.sort_by_key(|operation_log| std::cmp::Reverse(operation_log.created_on));
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::Other("add annotation gene-a".to_string())
        );

        let annotations = Annotation::query_by_group(graph_conn, "track-1", None).unwrap();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].name, "gene-a");
    }

    #[test]
    fn test_add_annotation_detects_no_changes() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let _ = setup_block_group(graph_conn);

        let operation_summary = add_annotation(
            &context,
            "test",
            "gene-a",
            Some("track-1"),
            "test",
            "chr1:1-5",
        )
        .unwrap();
        commit_operation_summary(&context, &operation_summary).unwrap();

        let operation_summary = add_annotation(
            &context,
            "test",
            "gene-a",
            Some("track-1"),
            "test",
            "chr1:1-5",
        )
        .unwrap();
        let error = commit_operation_summary(&context, &operation_summary).unwrap_err();
        assert_eq!(error, OperationError::NoChanges);
    }

    #[test]
    fn add_annotation_file_creates_operation() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let history_store = DoltHistoryStore::new(graph_conn);

        let repo_root = context.workspace().repo_root().unwrap();
        let annotation_path = repo_root.join("fixtures").join("annotation.gff3");
        fs::create_dir_all(annotation_path.parent().unwrap()).unwrap();
        fs::write(&annotation_path, "##gff-version 3\n").unwrap();
        let annotation_path_str = annotation_path.to_string_lossy().to_string();

        let commit_hash = add_annotation_file(
            &context,
            &annotation_path_str,
            None,
            None,
            Some("track-1"),
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .unwrap();
        assert_eq!(history_store.current_head().unwrap(), Some(commit_hash));

        let mut asset_refs = AssetRef::all(graph_conn).expect("should load asset references");
        asset_refs.sort_by(|left, right| left.role.as_str().cmp(right.role.as_str()));
        let mut operation_logs = OperationLog::all(graph_conn).expect("should load operation logs");
        operation_logs.sort_by_key(|operation_log| operation_log.created_on);
        assert_eq!(asset_refs.len(), 3);
        let annotation_asset = asset_refs
            .iter()
            .find(|asset| asset.role == AssetRole::Annotation)
            .expect("should track the annotation source");
        assert_eq!(annotation_asset.name.as_deref(), Some("track-1"));
        assert!(
            annotation_asset
                .logical_path
                .as_deref()
                .unwrap_or_default()
                .ends_with("fixtures/annotation.gff3")
        );
        assert!(asset_refs.iter().any(|asset| {
            asset.role == AssetRole::AnnotationIndex
                && asset.upstream_asset_ref_id == Some(annotation_asset.id)
        }));
        assert!(asset_refs.iter().any(|asset| {
            asset.role == AssetRole::Other(ANNOTATION_ATTRIBUTE_INDEX_ROLE.to_string())
                && asset.upstream_asset_ref_id == Some(annotation_asset.id)
        }));
        assert_eq!(operation_logs.len(), 1);
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::AnnotationFile
        );
        let operation_assets = OperationAsset::by_log_id(graph_conn, &operation_logs[0].id);
        assert_eq!(operation_assets.len(), 3);
        assert!(
            history_store
                .log(None)
                .unwrap()
                .iter()
                .any(|entry| entry.commit_hash == commit_hash
                    && entry.message.contains("Add annotation file"))
        );

        let error = add_annotation_file(
            &context,
            &annotation_path_str,
            None,
            None,
            Some("track-1"),
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            AnnotationFileError::OperationError(OperationError::NoChanges)
        ));
    }

    #[test]
    fn test_add_annotation_file_rebuilds_local_gff_indexes_after_sorting() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();

        let repo_root = context
            .workspace()
            .repo_root()
            .expect("should have repo root");
        let annotation_path = repo_root
            .join("fixtures")
            .join("annotation-with-index.gff3");
        let index_path = repo_root.join("fixtures").join("annotation-with-index.csi");
        let annotation_contents = b"##gff-version 3\n";
        let index_contents = b"index";
        fs::create_dir_all(
            annotation_path
                .parent()
                .expect("should have annotation parent directory"),
        )
        .expect("should create fixture directory");
        fs::write(&annotation_path, annotation_contents).expect("should write annotation fixture");
        fs::write(&index_path, index_contents).expect("should write index fixture");

        add_annotation_file(
            &context,
            annotation_path
                .to_str()
                .expect("should encode annotation path"),
            None,
            Some(index_path.to_str().expect("should encode index path")),
            Some("track-with-index"),
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should create annotation file operation");

        let mut asset_refs = AssetRef::all(graph_conn).expect("should load asset references");
        asset_refs.sort_by(|left, right| {
            left.role
                .as_str()
                .cmp(right.role.as_str())
                .then_with(|| left.logical_path.cmp(&right.logical_path))
        });
        assert_eq!(asset_refs.len(), 3);
        let annotation_asset = asset_refs
            .iter()
            .find(|asset| asset.role == AssetRole::Annotation)
            .expect("should track the annotation source");
        let coordinate_index = asset_refs
            .iter()
            .find(|asset| asset.role == AssetRole::AnnotationIndex)
            .expect("should rebuild a coordinate index");
        let attribute_index = asset_refs
            .iter()
            .find(|asset| {
                asset.role == AssetRole::Other(ANNOTATION_ATTRIBUTE_INDEX_ROLE.to_string())
            })
            .expect("should build an attribute index");
        assert_eq!(
            coordinate_index.upstream_asset_ref_id,
            Some(annotation_asset.id)
        );
        assert_eq!(
            attribute_index.upstream_asset_ref_id,
            Some(annotation_asset.id)
        );
        assert_eq!(
            annotation_asset.checksum,
            Some(
                calculate_file_checksum(
                    annotation_asset
                        .versioned_store_path(context.workspace())
                        .expect("should resolve retained BGZF annotation")
                )
                .expect("should checksum retained BGZF annotation")
            )
        );
        assert_eq!(
            annotation_asset.materialized_checksum,
            Some(calculate_reader_checksum(annotation_contents.as_slice()).unwrap())
        );
        assert!(annotation_asset.uri.ends_with(".gff3.bgz"));
        let archive_path = annotation_asset
            .versioned_store_path(context.workspace())
            .expect("should resolve retained BGZF annotation");
        let mut decoded = noodles::bgzf::io::Reader::new(
            fs::File::open(archive_path).expect("should open retained BGZF annotation"),
        );
        let mut decoded_contents = Vec::new();
        decoded
            .read_to_end(&mut decoded_contents)
            .expect("should decompress retained annotation bytes");
        assert_eq!(decoded_contents, annotation_contents);
        assert_eq!(fs::read(&annotation_path).unwrap(), annotation_contents);
        assert_eq!(fs::read(&index_path).unwrap(), index_contents);
        assert_eq!(coordinate_index.file_type.as_str(), "tabix");
        assert!(coordinate_index.uri.ends_with(".csi"));
        assert_eq!(attribute_index.file_type.as_str(), "none");
        assert!(attribute_index.uri.ends_with(".gai"));

        let err = add_annotation_file(
            &context,
            annotation_path
                .to_str()
                .expect("should encode annotation path"),
            None,
            Some(index_path.to_str().expect("should encode index path")),
            Some("track-with-index"),
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AnnotationFileError::OperationError(OperationError::NoChanges)
        ));
    }

    #[test]
    fn test_add_annotation_file_sorts_local_gff() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let annotation_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("features.gff3");
        let input = concat!(
            "##gff-version 3\n",
            "m123\tsource\tgene\t20\t25\t.\t+\t.\tName=Beta-feature\n",
            "m123\tsource\tgene\t2\t4\t.\t+\t.\tName=Alpha-feature\n",
            "m123\tsource\tgene\t10\t12\t.\t-\t.\tName=Gamma-feature\n",
        );
        let source_checksum = calculate_reader_checksum(input.as_bytes())
            .expect("should checksum original unsorted GFF input");
        fs::write(&annotation_path, input).expect("should write unsorted GFF input");

        add_annotation_file(
            &context,
            annotation_path
                .to_str()
                .expect("should encode annotation path"),
            None,
            None,
            Some("sorted-gff"),
            None,
            AnnotationFileChecksumOverrides {
                annotation: Some(source_checksum),
                index: None,
            },
        )
        .expect("should sort and index local GFF input");

        assert_eq!(fs::read_to_string(&annotation_path).unwrap(), input);
        let assets = AssetRef::all(graph_conn).expect("should load generated annotation assets");
        let annotation = assets
            .iter()
            .find(|asset| asset.role == AssetRole::Annotation)
            .expect("should track GFF source");

        let source_path = annotation
            .versioned_store_path(context.workspace())
            .expect("should resolve retained BGZF source");
        let retained_bgzf_bytes = fs::read(&source_path).expect("should read retained BGZF bytes");
        let archive_checksum = calculate_reader_checksum(retained_bgzf_bytes.as_slice())
            .expect("should checksum retained BGZF archive bytes");
        assert_eq!(annotation.checksum, Some(archive_checksum));
        let mut bgzf_reader = noodles::bgzf::io::Reader::new(
            fs::File::open(&source_path).expect("should open retained BGZF source"),
        );
        let mut sorted_text = String::new();
        bgzf_reader
            .read_to_string(&mut sorted_text)
            .expect("should decode sorted GFF source");
        let sorted_checksum = calculate_reader_checksum(sorted_text.as_bytes())
            .expect("should checksum sorted annotation content");
        assert_eq!(annotation.materialized_checksum, Some(sorted_checksum));
        assert_ne!(annotation.materialized_checksum, Some(source_checksum));
        let sorted_positions = sorted_text
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split('\t').nth(3).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sorted_positions, ["2", "10", "20"]);
    }

    #[test]
    fn test_add_annotation_file_builds_local_gff_csi() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let annotation_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("features.gff3");
        let input = concat!(
            "##gff-version 3\n",
            "m123\tsource\tgene\t600000002\t600000004\t.\t+\t.\tName=Alpha-feature\n",
            "m123\tsource\tgene\t600000020\t600000025\t.\t+\t.\tName=Beta-feature\n",
        );
        fs::write(&annotation_path, input).expect("should write sorted GFF input");

        add_annotation_file(
            &context,
            annotation_path
                .to_str()
                .expect("should encode annotation path"),
            None,
            None,
            Some("indexed-gff"),
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should index local GFF input");

        let assets = AssetRef::all(graph_conn).expect("should load generated annotation assets");
        assert_eq!(assets.len(), 3);
        let annotation = assets
            .iter()
            .find(|asset| asset.role == AssetRole::Annotation)
            .expect("should track GFF source");
        let coordinate_index = assets
            .iter()
            .find(|asset| asset.role == AssetRole::AnnotationIndex)
            .expect("should track CSI coordinate index");
        assert!(annotation.uri.ends_with(".gff3.bgz"));
        assert!(coordinate_index.uri.ends_with(".csi"));
        assert_eq!(coordinate_index.file_type, FileTypes::Tabix.as_str());
        assert_eq!(
            coordinate_index.upstream_asset_ref_id,
            Some(annotation.id),
            "CSI metadata should link to its annotation source"
        );

        let source_path = annotation
            .versioned_store_path(context.workspace())
            .expect("should resolve retained BGZF source");
        let coordinate_index_path = coordinate_index
            .versioned_store_path(context.workspace())
            .expect("should resolve retained CSI index");
        let coordinate_index = noodles::csi::fs::read(&coordinate_index_path)
            .expect("should read coordinate index generated from BGZF bytes");
        let mut indexed_reader = noodles::csi::io::IndexedReader::new(
            File::open(&source_path).unwrap(),
            coordinate_index,
        );
        let region: noodles::core::Region = "m123:600000002-600000002".parse().unwrap();
        let records = indexed_reader
            .query(&region)
            .unwrap()
            .map(|result| result.unwrap().as_ref().as_bytes().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 1);
        assert!(String::from_utf8_lossy(&records[0]).contains("Alpha-feature"));
    }

    #[test]
    fn test_add_annotation_file_failure_stops_stage_notifications() {
        let context = setup_gen();
        let annotation_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("invalid.gff3");
        fs::write(&annotation_path, "not a GFF feature record\n")
            .expect("should write invalid GFF input");
        let mut progress_events = Vec::new();
        let error = {
            let mut on_progress = |stage| progress_events.push(stage);
            add_annotation_file_with_progress(
                &context,
                annotation_path
                    .to_str()
                    .expect("should encode invalid annotation path"),
                None,
                None,
                Some("invalid-gff"),
                None,
                AnnotationFileAddOptions {
                    checksum_overrides: AnnotationFileChecksumOverrides::default(),
                    progress: Some(&mut on_progress),
                },
            )
            .expect_err("should reject malformed GFF input while sorting")
        };

        assert!(matches!(error, AnnotationFileError::GaiError(_)));
        assert_eq!(progress_events, [AnnotationFileProgressStage::Sorting]);
    }

    #[test]
    fn test_add_annotation_file_indexes_bed4_names_after_sorting() {
        let context = setup_gen();
        let annotation_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("features.bed");
        let input = "m123\t10\t16\texon-z\nm123\t1\t5\texon-a\n";
        fs::write(&annotation_path, input).expect("should write unsorted BED4 input");
        let mut progress_events = Vec::new();
        {
            let mut on_progress = |stage| progress_events.push(stage);
            add_annotation_file_with_progress(
                &context,
                annotation_path
                    .to_str()
                    .expect("should encode annotation path"),
                None,
                None,
                Some("search-bed"),
                None,
                AnnotationFileAddOptions {
                    checksum_overrides: AnnotationFileChecksumOverrides::default(),
                    progress: Some(&mut on_progress),
                },
            )
            .expect("should sort and index local BED4 input");
        }
        assert_eq!(
            progress_events,
            [
                AnnotationFileProgressStage::Sorting,
                AnnotationFileProgressStage::Compressing,
                AnnotationFileProgressStage::BuildingIndex,
            ]
        );

        assert_eq!(fs::read_to_string(&annotation_path).unwrap(), input);
        let assets =
            AssetRef::all(context.graph().conn()).expect("should load generated annotation assets");
        let annotation = assets
            .iter()
            .find(|asset| asset.role == AssetRole::Annotation)
            .expect("should track BED source");
        let coordinate_index = assets
            .iter()
            .find(|asset| asset.role == AssetRole::AnnotationIndex)
            .expect("should track BED coordinate index");
        let attribute_index = assets
            .iter()
            .find(|asset| {
                asset.role == AssetRole::Other(ANNOTATION_ATTRIBUTE_INDEX_ROLE.to_string())
            })
            .expect("should track BED name index");
        let source_path = annotation
            .versioned_store_path(context.workspace())
            .expect("should resolve BGZF source");
        let coordinate_index_path = coordinate_index
            .versioned_store_path(context.workspace())
            .expect("should resolve CSI index");
        let attribute_index_path = attribute_index
            .versioned_store_path(context.workspace())
            .expect("should resolve GAI index");
        let mut bgzf_reader = bgzf::io::Reader::new(File::open(&source_path).unwrap());
        let mut sorted_text = String::new();
        bgzf_reader.read_to_string(&mut sorted_text).unwrap();
        assert_eq!(
            sorted_text.lines().collect::<Vec<_>>(),
            ["m123\t1\t5\texon-a", "m123\t10\t16\texon-z"]
        );

        let mut indexed_source =
            gai::IndexedSource::open(&source_path, &coordinate_index_path, &attribute_index_path)
                .expect("should open BED source with both indexes");
        let hits = indexed_source
            .query_with_mode("exon-", gai::MatchMode::Prefix)
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].raw_line, "m123\t1\t5\texon-a");
        assert_eq!(hits[1].raw_line, "m123\t10\t16\texon-z");
    }

    #[test]
    fn test_add_annotation_file_builds_local_gff_attribute_index() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let annotation_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("features.gff3");
        let input = concat!(
            "##gff-version 3\n",
            "m123\tsource\tgene\t2\t4\t.\t+\t.\tID=id-only-term;Name=Alpha-feature\n",
            "m123\tsource\tgene\t10\t12\t.\t-\t.\tID=other-id;Name=Gamma-feature\n",
            "m123\tsource\tgene\t20\t25\t.\t+\t.\tID=private-id;Name=Beta-feature;name=foobar;gene=alternate-beta;db_xref=GeneID:123\n",
            "m123\tsource\tgene\t30\t35\t.\t+\t.\tID=private-gene-name-only;gene_name=BRCA1\n",
        );
        fs::write(&annotation_path, input).expect("should write GFF input with searchable names");

        add_annotation_file(
            &context,
            annotation_path
                .to_str()
                .expect("should encode annotation path"),
            None,
            None,
            Some("indexed-gff"),
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should index configured GFF attributes");

        let assets = AssetRef::all(graph_conn).expect("should load generated annotation assets");
        assert_eq!(assets.len(), 3);
        let annotation = assets
            .iter()
            .find(|asset| asset.role == AssetRole::Annotation)
            .expect("should track GFF source");
        let coordinate_index = assets
            .iter()
            .find(|asset| asset.role == AssetRole::AnnotationIndex)
            .expect("should track CSI coordinate index");
        let attribute_index = assets
            .iter()
            .find(|asset| {
                asset.role == AssetRole::Other(ANNOTATION_ATTRIBUTE_INDEX_ROLE.to_string())
            })
            .expect("should track GAI attribute index");
        assert!(attribute_index.uri.ends_with(".gai"));
        assert_eq!(attribute_index.file_type, FileTypes::None.as_str());
        assert_eq!(
            attribute_index.upstream_asset_ref_id,
            Some(annotation.id),
            "GAI metadata should link to its annotation source"
        );

        let source_path = annotation
            .versioned_store_path(context.workspace())
            .expect("should resolve retained BGZF source");
        let coordinate_index_path = coordinate_index
            .versioned_store_path(context.workspace())
            .expect("should resolve retained CSI index");
        let attribute_index_path = attribute_index
            .versioned_store_path(context.workspace())
            .expect("should resolve retained GAI index");
        assert_eq!(
            gai::inspect_index(&attribute_index_path)
                .expect("should inspect retained GAI index")
                .attributes,
            vec![
                "Name".to_string(),
                "name".to_string(),
                "gene".to_string(),
                "gene_name".to_string(),
                "db_xref".to_string(),
            ]
        );
        let mut indexed_source =
            gai::IndexedSource::open(&source_path, &coordinate_index_path, &attribute_index_path)
                .expect("should open retained BGZF, CSI, and GAI files");
        assert_eq!(
            indexed_source.metadata().attributes,
            ["Name", "name", "gene", "gene_name", "db_xref"]
        );
        for query in ["beta", "alternate", "foobar", "GeneID:123"] {
            assert_eq!(
                indexed_source
                    .query_with_mode(query, gai::MatchMode::Prefix)
                    .expect("should query a configured GFF attribute")
                    .len(),
                1,
                "should find the value {query}"
            );
        }
        let gene_name_hits = indexed_source
            .query_with_mode("brca1", gai::MatchMode::Prefix)
            .expect("should query configured gene_name attribute");
        assert_eq!(gene_name_hits.len(), 1);
        assert!(gene_name_hits[0].raw_line.contains("gene_name=BRCA1"));
        assert!(
            indexed_source
                .query_with_mode("private-gene-name-only", gai::MatchMode::Prefix)
                .expect("should query for a private ID")
                .is_empty()
                && indexed_source
                    .query_with_mode("id-only-term", gai::MatchMode::Prefix)
                    .expect("should query for an ID-only term")
                    .is_empty(),
            "the ID attribute should not be indexed"
        );
    }

    #[test]
    fn test_add_annotation_file_does_not_read_remote_assets() {
        let context = setup_gen();
        let annotation_uri = "http://127.0.0.1:1/annotation.gff3".to_string();
        let index_uri = "http://127.0.0.1:1/annotation.gff3.csi".to_string();
        let mut progress_events = Vec::new();

        {
            let mut on_progress = |stage| progress_events.push(stage);
            add_annotation_file_with_progress(
                &context,
                &annotation_uri,
                None,
                Some(&index_uri),
                Some("remote-annotation"),
                None,
                AnnotationFileAddOptions {
                    checksum_overrides: AnnotationFileChecksumOverrides::default(),
                    progress: Some(&mut on_progress),
                },
            )
            .expect("should track remote annotation assets");
        }
        assert!(progress_events.is_empty());

        let asset_refs =
            AssetRef::all(context.graph().conn()).expect("should load asset references");
        assert_eq!(asset_refs.len(), 2);
        assert!(
            asset_refs
                .iter()
                .all(|asset_ref| asset_ref.checksum.is_none())
        );
        assert!(
            asset_refs
                .iter()
                .any(|asset_ref| asset_ref.uri == annotation_uri)
        );
        assert!(
            asset_refs
                .iter()
                .any(|asset_ref| asset_ref.uri == index_uri)
        );
    }

    #[test]
    fn test_add_non_gff_bed_annotation_file_has_no_sort_progress() {
        let context = setup_gen();
        let annotation_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("annotation.gb");
        fs::write(&annotation_path, "LOCUS       test\n")
            .expect("should write non-GFF/BED annotation input");
        let mut progress_events = Vec::new();

        {
            let mut on_progress = |stage| progress_events.push(stage);
            add_annotation_file_with_progress(
                &context,
                annotation_path
                    .to_str()
                    .expect("should encode non-GFF/BED annotation path"),
                None,
                None,
                Some("genbank-annotation"),
                None,
                AnnotationFileAddOptions {
                    checksum_overrides: AnnotationFileChecksumOverrides::default(),
                    progress: Some(&mut on_progress),
                },
            )
            .expect("should add non-GFF/BED annotation without sort progress");
        }

        assert!(progress_events.is_empty());
    }

    #[test]
    fn test_add_annotation_file_passes_remote_checksum_overrides() {
        let context = setup_gen();
        let annotation_uri = "http://127.0.0.1:1/annotation.gff3";
        let index_uri = "http://127.0.0.1:1/annotation.gff3.csi";
        let annotation_checksum = Sha256Hash::convert_str("known annotation contents");
        let index_checksum = Sha256Hash::convert_str("known index contents");

        add_annotation_file(
            &context,
            annotation_uri,
            None,
            Some(index_uri),
            Some("remote-annotation"),
            None,
            AnnotationFileChecksumOverrides {
                annotation: Some(annotation_checksum),
                index: Some(index_checksum),
            },
        )
        .expect("should track checksummed remote assets without reading them");

        let asset_refs =
            AssetRef::all(context.graph().conn()).expect("should load asset references");
        assert_eq!(asset_refs.len(), 2);
        let annotation = asset_refs
            .iter()
            .find(|asset_ref| asset_ref.role == AssetRole::Annotation)
            .expect("should track annotation asset");
        let index = asset_refs
            .iter()
            .find(|asset_ref| asset_ref.role == AssetRole::AnnotationIndex)
            .expect("should track annotation index");
        assert_eq!(annotation.uri, annotation_uri);
        assert_eq!(annotation.checksum, Some(annotation_checksum));
        assert_eq!(index.uri, index_uri);
        assert_eq!(index.checksum, Some(index_checksum));
    }

    #[test]
    fn test_add_annotation_file_does_not_store_absolute_external_index_path() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let repo_root = context
            .workspace()
            .repo_root()
            .expect("should have repo root");
        let annotation_path = repo_root.join("annotation.gff3");
        fs::write(&annotation_path, "##gff-version 3\n").expect("should write annotation fixture");

        let external_dir = tempfile::tempdir().expect("should create external directory");
        let index_path = external_dir.path().join("private-index.csi");
        fs::write(&index_path, "private index contents").expect("should write external index");
        let index_path = index_path
            .to_str()
            .expect("should encode external index path");

        add_annotation_file(
            &context,
            annotation_path
                .to_str()
                .expect("should encode annotation path"),
            None,
            Some(index_path),
            None,
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should create annotation file operation");

        let assets = AssetRef::all(graph_conn).expect("should load asset references");
        let index_asset = assets
            .iter()
            .find(|asset| asset.role == AssetRole::AnnotationIndex)
            .expect("should store generated coordinate index asset");
        assert_eq!(
            index_asset.logical_path.as_deref(),
            Some("annotation.gff3.csi")
        );
        assert!(!index_asset.uri.contains(index_path));
        assert!(assets.iter().any(|asset| {
            asset.role == AssetRole::Other(ANNOTATION_ATTRIBUTE_INDEX_ROLE.to_string())
                && asset
                    .logical_path
                    .as_deref()
                    .is_some_and(|path| path.ends_with(".gai"))
        }));
        assert!(!assets.iter().any(|asset| asset.uri.contains(index_path)));
    }

    #[test]
    fn test_annotation_and_accession_interval_tree_caches_are_runtime_only() {
        let conn = get_connection(None).unwrap();
        let annotation_id = HashId::convert_str("cached-annotation");
        let accession_id = HashId::convert_str("cached-accession");
        let block_group_id = HashId::convert_str("cached-block-group");
        let tree: IntervalTree<i64, NodeIntervalBlock> = [(
            0..4,
            NodeIntervalBlock {
                node_id: HashId::convert_str("cached-node"),
                start: 0,
                end: 4,
                sequence_start: 3,
                sequence_end: 7,
                strand: Strand::Reverse,
            },
        )]
        .into_iter()
        .collect();
        let shared_tree = Arc::new(tree);
        let mut accession = Accession {
            id: accession_id,
            name: "cached-accession".to_string(),
            block_group_id,
            parent_accession_id: None,
            cached_interval_tree: None,
            history_ref: Some("saved-ref".to_string()),
        };
        accession.set_cached_interval_tree(Some(Arc::clone(&shared_tree)));
        let mut annotation = Annotation {
            id: annotation_id,
            name: "cached-annotation".to_string(),
            group: "test".to_string(),
            accession_id,
            cached_accession: ForeignKeyCache::default(),
            cached_interval_tree: None,
            history_ref: Some("saved-ref".to_string()),
            extra: None,
        };
        annotation.set_cached_interval_tree(Some(Arc::clone(&shared_tree)));
        let shared_accession = Arc::new(accession.clone());
        annotation.set_accession(Arc::clone(&shared_accession));

        assert!(
            Accession::select(&conn)
                .id(accession_id)
                .load()
                .expect("should query absent accession")
                .is_empty()
        );
        assert!(
            Annotation::select(&conn)
                .id(annotation_id)
                .load()
                .expect("should query absent annotation")
                .is_empty()
        );
        let annotation_tree = annotation
            .intervaltree(&conn)
            .expect("should return cached annotation tree without database rows");
        let annotation_accession = annotation
            .get_accession(&conn)
            .expect("should return cached annotation accession without database rows");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should return cached accession tree without database rows");
        assert!(Arc::ptr_eq(&annotation_tree, &shared_tree));
        assert!(Arc::ptr_eq(&annotation_accession, &shared_accession));
        assert!(Arc::ptr_eq(&accession_tree, &shared_tree));
        let mut annotation_without_tree = annotation.clone();
        annotation_without_tree.set_cached_interval_tree(None);
        let tree_from_cached_accession = annotation_without_tree
            .intervaltree(&conn)
            .expect("should load annotation tree from cached accession");
        assert!(Arc::ptr_eq(&tree_from_cached_accession, &shared_tree));
        assert_eq!(
            annotation_without_tree
                .length(&conn)
                .expect("should get annotation length from cached accession tree"),
            4
        );
        assert_eq!(
            accession
                .length(&conn)
                .expect("should get cached accession length"),
            4
        );

        let cloned_annotation = annotation.clone();
        let cloned_accession = accession.clone();
        assert!(Arc::ptr_eq(
            &annotation_tree,
            cloned_annotation
                .cached_interval_tree()
                .expect("should clone annotation cache")
        ));
        let cloned_related_accession = cloned_annotation
            .cached_accession
            .snapshot()
            .expect("should clone annotation accession cache");
        assert!(Arc::ptr_eq(
            &annotation_accession,
            &cloned_related_accession
        ));
        assert!(Arc::ptr_eq(
            &accession_tree,
            cloned_accession
                .cached_interval_tree()
                .expect("should clone accession cache")
        ));

        let mut uncached_annotation = annotation.clone();
        uncached_annotation.set_cached_interval_tree(None);
        uncached_annotation.cached_accession.replace(None);
        assert!(uncached_annotation.cached_accession.snapshot().is_none());
        assert!(annotation.cached_accession.snapshot().is_some());
        let mut uncached_accession = accession.clone();
        uncached_accession.set_cached_interval_tree(None);
        assert_eq!(annotation, uncached_annotation);
        assert_eq!(accession, uncached_accession);
        assert_eq!(
            serde_json::to_value(&annotation).expect("should serialize annotation"),
            serde_json::to_value(&uncached_annotation)
                .expect("should serialize uncached annotation")
        );
        assert_eq!(
            serde_json::to_value(&accession).expect("should serialize accession"),
            serde_json::to_value(&uncached_accession).expect("should serialize uncached accession")
        );
        assert!(
            serde_json::to_value(&annotation)
                .expect("should serialize annotation")
                .get("history_ref")
                .is_none()
        );
        assert!(
            serde_json::to_value(&accession)
                .expect("should serialize accession")
                .get("history_ref")
                .is_none()
        );
        assert!(
            serde_json::from_value::<Annotation>(
                serde_json::to_value(&annotation).expect("should serialize annotation")
            )
            .expect("should deserialize annotation")
            .cached_interval_tree()
            .is_none()
        );
        assert!(
            serde_json::from_value::<Annotation>(
                serde_json::to_value(&annotation).expect("should serialize annotation")
            )
            .expect("should deserialize annotation")
            .cached_accession
            .snapshot()
            .is_none()
        );
        assert_eq!(
            serde_json::from_value::<Annotation>(
                serde_json::to_value(&annotation).expect("should serialize annotation")
            )
            .expect("should deserialize annotation")
            .history_ref(),
            None
        );
        assert!(
            serde_json::from_value::<Accession>(
                serde_json::to_value(&accession).expect("should serialize accession")
            )
            .expect("should deserialize accession")
            .cached_interval_tree()
            .is_none()
        );
    }

    #[test]
    fn test_annotation_history_ref_loads_historical_accession_nodes() {
        let conn = get_connection(None).unwrap();
        let (_block_group_id, path) = setup_block_group(&conn);
        let accession =
            BlockGroup::add_accession(&conn, &path, "history-accession", 0, 10).unwrap();
        let annotation =
            Annotation::get_or_create(&conn, "history-annotation", "test", &accession.id, None)
                .unwrap();
        let history_ref: String = conn
            .query_row(
                "SELECT dolt_commit('-A', '-m', 'save historical annotation accession')",
                [],
                |row| row.get(0),
            )
            .unwrap();

        conn.execute(
            "DELETE FROM accession_nodes WHERE accession_id = ?1",
            [accession.id],
        )
        .unwrap();
        assert!(Accession::get_nodes_by_id(&conn, &accession.id, None).is_empty());

        let annotation = Annotation::select(&conn)
            .id(annotation.id)
            .with_ref(history_ref.as_str())
            .get()
            .unwrap()
            .unwrap();
        assert_eq!(annotation.history_ref(), Some(history_ref.as_str()));

        let accession = annotation.get_accession(&conn).unwrap();
        assert_eq!(accession.history_ref(), Some(history_ref.as_str()));
        assert_eq!(accession.length(&conn).unwrap(), 10);
        assert_eq!(annotation.length(&conn).unwrap(), 10);
        assert!(
            annotation
                .intervaltree(&conn)
                .unwrap()
                .query(0..10)
                .next()
                .is_some()
        );
    }
}
