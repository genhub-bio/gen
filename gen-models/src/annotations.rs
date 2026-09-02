use std::path::{Path, PathBuf};

use gen_core::{
    DoltHashId, HashId, NodeIntervalBlock, Sha256Hash, calculate_hash,
    config::Workspace,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use intervaltree::IntervalTree;
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    accession::{Accession, AccessionError, AccessionSpan, NewAccession},
    assets::{AssetRef, AssetRole, OperationKind},
    db::{DbContext, GraphConnection},
    errors::{FileAdditionError, OperationError},
    file_types::FileTypes,
    gen_models_capnp::{annotation, annotation_group, annotation_group_sample},
    history::{HistoryStore, dolt::DoltHistoryStore},
    operations::{FileAddition, OperationFile, OperationInfo, OperationSummary, track_asset_refs},
    region::GenRegionError,
    traits::{Query, max_rows_per_batch},
};
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AnnotationGroup {
    pub name: String,
}

impl Query for AnnotationGroup {
    type Model = AnnotationGroup;

    const PRIMARY_KEY: &'static str = "name";
    const TABLE_NAME: &'static str = "annotation_groups";

    fn process_row(row: &Row) -> Self::Model {
        AnnotationGroup {
            name: row.get(0).unwrap(),
        }
    }
}

impl AnnotationGroup {
    /// Lists annotation groups visible at an optional historical reference.
    pub fn all(conn: &GraphConnection, history_ref: Option<&str>) -> Vec<AnnotationGroup> {
        let table = AnnotationGroup::table_name_with_history_ref(history_ref);
        let mut params: Vec<(&str, &dyn rusqlite::ToSql)> = Vec::new();
        if let Some(history_ref) = history_ref.as_ref() {
            params.push((":history_ref", history_ref));
        }
        AnnotationGroup::query(
            conn,
            &format!("SELECT * FROM {table} ORDER BY name"),
            &params[..],
        )
    }

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
        let groups_table = AnnotationGroup::table_name_with_history_ref(history_ref);
        let samples_table = if history_ref.is_some() {
            "dolt_at_annotation_group_samples(:history_ref)"
        } else {
            "annotation_group_samples"
        };
        let query = format!(
            "\
            select ag.* \
            from {groups_table} ag \
            join {samples_table} s \
                on ag.name = s.annotation_group \
            where s.sample_name = :sample_name \
            order by ag.name;"
        );
        let mut params: Vec<(&str, &dyn rusqlite::ToSql)> = vec![(":sample_name", &sample_name)];
        if let Some(history_ref) = history_ref.as_ref() {
            params.push((":history_ref", history_ref));
        }
        AnnotationGroup::query(conn, &query, &params[..])
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

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Annotation {
    pub id: HashId,
    pub name: String,
    pub group: String,
    pub accession_id: HashId,
    pub extra: Option<AnnotationExtra>,
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
            extra,
        }
    }
}

impl Query for Annotation {
    type Model = Annotation;

    const TABLE_NAME: &'static str = "annotations";

    fn process_row(row: &Row) -> Self::Model {
        Annotation {
            id: row.get(0).unwrap(),
            name: row.get(1).unwrap(),
            group: row.get(2).unwrap(),
            accession_id: row.get(3).unwrap(),
            extra: deserialize_annotation_extra(row.get(4).unwrap())
                .expect("should deserialize annotation extra from database"),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum AnnotationError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
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
        let query = "SELECT sample_name FROM annotation_group_samples WHERE annotation_group = ?1;";
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(params![annotation_group], |row| row.get(0))?;
        let mut samples = Vec::new();
        for row in rows {
            samples.push(row?);
        }
        Ok(samples)
    }

    pub fn query_by_sample(
        conn: &GraphConnection,
        sample_name: &str,
        history_ref: Option<&str>,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let annotations_table = Annotation::table_name_with_history_ref(history_ref);
        let samples_table = if history_ref.is_some() {
            "dolt_at_annotation_group_samples(:history_ref)"
        } else {
            "annotation_group_samples"
        };
        let query = format!(
            "select annotations.* from {annotations_table} annotations \
             join {samples_table} samples \
               on annotations.annotation_group = samples.annotation_group \
             where samples.sample_name = :sample_name"
        );
        let mut params: Vec<(&str, &dyn rusqlite::ToSql)> = vec![(":sample_name", &sample_name)];
        if let Some(history_ref) = history_ref.as_ref() {
            params.push((":history_ref", history_ref));
        }
        Ok(Annotation::query(conn, &query, &params[..]))
    }

    pub fn query_by_group(
        conn: &GraphConnection,
        group: &str,
        history_ref: Option<&str>,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let query = format!(
            "select * from {} where annotation_group = :group",
            Annotation::table_name_with_history_ref(history_ref)
        );
        let mut params: Vec<(&str, &dyn rusqlite::ToSql)> = vec![(":group", &group)];
        if let Some(history_ref) = history_ref.as_ref() {
            params.push((":history_ref", history_ref));
        }
        Ok(Annotation::query(conn, &query, &params[..]))
    }

    pub fn query_by_group_and_block_group(
        conn: &GraphConnection,
        group: &str,
        block_group_id: &HashId,
        history_ref: Option<&str>,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let annotations_table = Annotation::table_name_with_history_ref(history_ref);
        let accessions_table = Accession::table_name_with_history_ref(history_ref);
        let query = format!(
            "SELECT annotations.* FROM {annotations_table} annotations \
             JOIN {accessions_table} accessions ON accessions.id = annotations.accession_id \
             WHERE annotations.annotation_group = :group \
               AND accessions.block_group_id = :block_group_id"
        );
        let mut params: Vec<(&str, &dyn rusqlite::ToSql)> =
            vec![(":group", &group), (":block_group_id", block_group_id)];
        if let Some(history_ref) = history_ref.as_ref() {
            params.push((":history_ref", history_ref));
        }
        Ok(Annotation::query(conn, &query, &params[..]))
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
        Ok(Annotation::query(
            conn,
            query,
            params![collection_name, sample_name, block_group_name],
        ))
    }

    pub fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, AnnotationError> {
        let accession = Accession::get(
            conn,
            "select * from accessions where id = ?1",
            params![self.accession_id],
        )?;
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
        let matches = Annotation::query(
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
        );

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

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AnnotationGroupSample {
    pub annotation_group: String,
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
    if matches!(ext.as_deref(), Some("gz") | Some("bgz")) {
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

    let mut candidates = vec![format!("{path}.tbi")];
    let path_buf = PathBuf::from(path);
    if let Some(extension) = path_buf.extension().and_then(|ext| ext.to_str()) {
        let mut extension_candidate = path_buf.clone();
        extension_candidate.set_extension(format!("{extension}.tbi"));
        let extension_candidate = extension_candidate.to_string_lossy().to_string();
        if !candidates
            .iter()
            .any(|candidate| candidate == &extension_candidate)
        {
            candidates.push(extension_candidate);
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

/// Adds an annotation asset and optional index to operation history.
///
/// Callers pass checksums only when the corresponding remote streams were already consumed for
/// annotation work. This keeps operation creation independent of remote credentials while local
/// assets are retained and hashed during preparation.
pub fn add_annotation_file(
    context: &DbContext,
    path: &str,
    format: Option<&str>,
    index: Option<&str>,
    name: Option<&str>,
    message: Option<&str>,
    checksum_overrides: AnnotationFileChecksumOverrides,
) -> Result<DoltHashId, AnnotationFileError> {
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
    let file_addition =
        FileAddition::prepare(workspace, path, file_type, checksum_overrides.annotation)?;
    let annotation_logical_path =
        OperationFile::storage_file_path(workspace, path, file_addition.checksum.as_ref())?;
    let prepared_index =
        if let Some(index_path) = annotation_index_file_path(workspace, path, index) {
            let index_file_type = FileTypes::infer_from_path(&index_path);
            let file_addition = FileAddition::prepare(
                workspace,
                &index_path,
                index_file_type,
                checksum_overrides.index,
            )?;
            let logical_path = OperationFile::storage_file_path(
                workspace,
                &index_path,
                file_addition.checksum.as_ref(),
            )?;
            let name = Path::new(&index_path)
                .file_name()
                .and_then(|value| value.to_str())
                .map(str::to_string);
            Some((file_addition, logical_path, name))
        } else {
            None
        };
    let name = name.or_else(|| Path::new(path).file_name().and_then(|value| value.to_str()));
    let name_value = name.unwrap_or_default();
    let annotation_asset_ref_id = AssetRef::id_hash(
        &file_addition.asset_uri,
        file_addition.file_type.as_str(),
        file_addition.checksum.as_ref(),
        &AssetRole::Annotation,
        Some(&annotation_logical_path),
        name,
        None,
    );
    let index_file_addition_id = prepared_index
        .as_ref()
        .map(|(index_file, _, _)| index_file.id.to_string())
        .unwrap_or_default();
    let log_id = HashId(calculate_hash(&format!(
        "{file_addition_id}:{name_value}:{index_file_addition_id}",
        file_addition_id = file_addition.id
    )));
    let summary = message
        .map(str::to_string)
        .unwrap_or_else(|| format!("Add annotation file {path}"));
    let created_on = chrono::Utc::now()
        .timestamp_nanos_opt()
        .expect("should create annotation asset timestamp");
    let mut tracked_assets = vec![AssetRef::from_file_addition(
        &file_addition,
        AssetRole::Annotation,
        Some(&annotation_logical_path),
        name,
        None,
        created_on,
    )];
    if let Some((index_file_addition, index_logical_path, index_name)) = prepared_index.as_ref() {
        tracked_assets.push(AssetRef::from_file_addition(
            index_file_addition,
            AssetRole::AnnotationIndex,
            Some(index_logical_path),
            index_name.as_deref(),
            Some(&annotation_asset_ref_id),
            created_on,
        ));
    }
    track_asset_refs(
        graph_conn,
        Some(&log_id),
        &OperationKind::AnnotationFile,
        &summary,
        &tracked_assets,
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
    use std::{collections::HashSet, fs};

    use gen_core::{HashId, region::RegionResolutionError};

    use super::*;
    use crate::{
        assets::{AssetRef, AssetRole, OperationAsset, OperationLog},
        block_group::{BlockGroup, PathCache},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        errors::OperationError,
        operations::{calculate_reader_checksum, commit_operation_summary},
        path::Path,
        sample::Sample,
        sample_lineage::SampleLineage,
        test_helpers::{create_bg, get_connection, setup_block_group, setup_gen},
        traits::Query,
    };

    mod region_resolver {
        use super::*;

        #[test]
        fn resolves_annotation_by_name_case_insensitively() {
            let conn = get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone()).unwrap();
            let accession =
                BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5, &mut cache).unwrap();
            let annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

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
            let (block_group_id, path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone()).unwrap();
            let accession =
                BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5, &mut cache).unwrap();
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
                BlockGroup::add_accession(&conn, &other_path, "ann-accession-2", 0, 5, &mut cache)
                    .unwrap();
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
            let (parent_block_group_id, parent_path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &parent_block_group_id, parent_path.name.clone())
                .unwrap();
            let parent_accession = BlockGroup::add_accession(
                &conn,
                &parent_path,
                "parent-accession",
                0,
                5,
                &mut cache,
            )
            .unwrap();
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
                BlockGroup::add_accession(&conn, &child_path, "child-accession", 0, 5, &mut cache)
                    .unwrap();
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
            let (parent_block_group_id, parent_path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &parent_block_group_id, parent_path.name.clone())
                .unwrap();
            let parent_accession = BlockGroup::add_accession(
                &conn,
                &parent_path,
                "parent-only-accession",
                0,
                5,
                &mut cache,
            )
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
        #[expect(clippy::too_many_arguments, reason = "test helper wiring")]
        fn annotate_block_group(
            conn: &GraphConnection,
            parent_path: &Path,
            cache: &mut PathCache,
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
            let accession =
                BlockGroup::add_accession(conn, &path, accession_name, 0, 5, cache).unwrap();
            Annotation::get_or_create(conn, annotation_name, "genes", &accession.id, None).unwrap()
        }

        #[test]
        fn test_lists_annotations_on_block_group() {
            let conn = get_connection(None).unwrap();
            let (block_group_id, path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone()).unwrap();
            let accession =
                BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5, &mut cache).unwrap();
            let annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

            let listed = Annotation::query_with_lineage(&conn, "test", "test", "chr1").unwrap();

            assert_eq!(listed, vec![annotation]);
        }

        #[test]
        fn test_includes_parent_annotations_via_lineage() {
            let conn = get_connection(None).unwrap();
            let (parent_block_group_id, parent_path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &parent_block_group_id, parent_path.name.clone())
                .unwrap();
            let parent_accession = BlockGroup::add_accession(
                &conn,
                &parent_path,
                "parent-accession",
                0,
                5,
                &mut cache,
            )
            .unwrap();
            let parent_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &parent_accession.id, None)
                    .unwrap();

            let child_annotation = annotate_block_group(
                &conn,
                &parent_path,
                &mut cache,
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
            let (block_group_id, path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone()).unwrap();
            let accession =
                BlockGroup::add_accession(&conn, &path, "chr1-accession", 0, 5, &mut cache)
                    .unwrap();
            let chr1_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

            let _chr2_annotation = annotate_block_group(
                &conn,
                &path,
                &mut cache,
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
            let (block_group_id, path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone()).unwrap();
            let accession =
                BlockGroup::add_accession(&conn, &path, "test-accession", 0, 5, &mut cache)
                    .unwrap();
            let test_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &accession.id, None).unwrap();

            let _unrelated_annotation = annotate_block_group(
                &conn,
                &path,
                &mut cache,
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
            let (parent_block_group_id, parent_path) = setup_block_group(&conn);
            let mut cache = PathCache::new(&conn);
            let _ = PathCache::lookup(&mut cache, &parent_block_group_id, parent_path.name.clone())
                .unwrap();
            let parent_accession = BlockGroup::add_accession(
                &conn,
                &parent_path,
                "parent-accession",
                0,
                5,
                &mut cache,
            )
            .unwrap();
            let parent_annotation =
                Annotation::get_or_create(&conn, "mreB", "genes", &parent_accession.id, None)
                    .unwrap();

            let child_annotation = annotate_block_group(
                &conn,
                &parent_path,
                &mut cache,
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

        let groups = AnnotationGroup::all(&conn, None);

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
        let (block_group_id, path) = setup_block_group(&conn);

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

        let mut cache = PathCache::new(&conn);
        let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone()).unwrap();
        let accession =
            BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5, &mut cache).unwrap();

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
        let mut operation_logs = OperationLog::all(graph_conn);
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
        let err = commit_operation_summary(&context, &operation_summary).unwrap_err();
        assert_eq!(err, OperationError::NoChanges);
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

        let mut asset_refs = AssetRef::all(graph_conn);
        asset_refs.sort_by(|left, right| left.role.as_str().cmp(right.role.as_str()));
        let mut operation_logs = OperationLog::all(graph_conn);
        operation_logs.sort_by_key(|operation_log| operation_log.created_on);
        assert_eq!(asset_refs.len(), 1);
        assert_eq!(asset_refs[0].role, AssetRole::Annotation);
        assert_eq!(asset_refs[0].name.as_deref(), Some("track-1"));
        assert!(
            asset_refs[0]
                .logical_path
                .as_deref()
                .unwrap_or_default()
                .ends_with("fixtures/annotation.gff3")
        );
        assert_eq!(operation_logs.len(), 1);
        assert_eq!(
            operation_logs[0].operation_kind,
            OperationKind::AnnotationFile
        );
        let operation_assets = OperationAsset::by_log_id(graph_conn, &operation_logs[0].id);
        assert_eq!(operation_assets.len(), 1);
        assert_eq!(operation_assets[0].role, AssetRole::Annotation);
        assert!(
            history_store
                .log(None)
                .unwrap()
                .iter()
                .any(|entry| entry.commit_hash == commit_hash
                    && entry.message.contains("Add annotation file"))
        );
        let err = add_annotation_file(
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
            err,
            AnnotationFileError::OperationError(OperationError::NoChanges)
        ));
    }

    #[test]
    fn test_add_annotation_file_tracks_explicit_non_tabix_index() {
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
        fs::create_dir_all(
            annotation_path
                .parent()
                .expect("should have annotation parent directory"),
        )
        .expect("should create fixture directory");
        fs::write(&annotation_path, "##gff-version 3\n").expect("should write annotation fixture");
        fs::write(&index_path, "index").expect("should write index fixture");

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

        let mut asset_refs = AssetRef::all(graph_conn);
        asset_refs.sort_by(|left, right| {
            left.role
                .as_str()
                .cmp(right.role.as_str())
                .then_with(|| left.logical_path.cmp(&right.logical_path))
        });
        assert_eq!(asset_refs.len(), 2);
        assert_eq!(asset_refs[0].role, AssetRole::Annotation);
        assert_eq!(asset_refs[1].role, AssetRole::AnnotationIndex);
        assert_eq!(
            asset_refs[1].upstream_asset_ref_id,
            Some(asset_refs[0].id),
            "annotation index should identify its immutable upstream annotation"
        );
        assert_eq!(
            asset_refs[0].checksum,
            Some(calculate_reader_checksum("##gff-version 3\n".as_bytes()).unwrap())
        );
        assert_eq!(
            asset_refs[1].checksum,
            Some(calculate_reader_checksum("index".as_bytes()).unwrap())
        );
        assert_eq!(asset_refs[1].file_type.as_str(), "none");
        assert_eq!(
            asset_refs[1]
                .logical_path
                .as_deref()
                .expect("should store index logical path"),
            "fixtures/annotation-with-index.csi"
        );
    }

    #[test]
    fn test_add_annotation_file_does_not_read_remote_assets() {
        let context = setup_gen();
        let annotation_uri = "http://127.0.0.1:1/annotation.gff3".to_string();
        let index_uri = "http://127.0.0.1:1/annotation.gff3.csi".to_string();

        add_annotation_file(
            &context,
            &annotation_uri,
            None,
            Some(&index_uri),
            Some("remote-annotation"),
            None,
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should track remote annotation assets");

        let asset_refs = AssetRef::all(context.graph().conn());
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

        let asset_refs = AssetRef::all(context.graph().conn());
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

        let index_asset = AssetRef::all(graph_conn)
            .into_iter()
            .find(|asset| asset.role == AssetRole::AnnotationIndex)
            .expect("should store annotation index asset");
        let checksum = index_asset
            .checksum
            .expect("local index asset should have checksum");
        let expected_logical_path = format!(".gen/assets/{checksum}.csi");
        assert_eq!(
            index_asset.logical_path.as_deref(),
            Some(expected_logical_path.as_str())
        );
        assert_ne!(index_asset.logical_path.as_deref(), Some(index_path));
        assert!(!index_asset.uri.contains(index_path));
    }
}
