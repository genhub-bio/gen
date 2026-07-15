use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
};

#[cfg(all(feature = "remote-storage", not(target_os = "emscripten")))]
use anyhow::anyhow;
use gen_core::{
    HashId, NodeIntervalBlock, calculate_hash,
    config::Workspace,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use intervaltree::IntervalTree;
use rusqlite::{Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    accession::{Accession, AccessionError, AccessionSpan, NewAccession},
    db::{DbContext, GraphConnection, OperationsConnection},
    errors::FileAdditionError,
    file_types::FileTypes,
    gen_models_capnp::{annotation, annotation_group, annotation_group_sample},
    operations::{FileAddition, Operation, OperationInfo},
    session_operations::{end_operation, start_operation},
    traits::Query,
};
#[cfg(all(feature = "remote-storage", not(target_os = "emscripten")))]
use crate::{
    changesets::{ChangesetModels, DatabaseChangeset, write_changeset},
    errors::OperationError,
    files::GenDatabase,
    metadata,
    operations::{OperationFile, OperationSummary},
    session_operations::DependencyModels,
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
        match AnnotationGroup::create(conn, name) {
            Ok(group) => Ok(group),
            Err(rusqlite::Error::SqliteFailure(err, _details))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                AnnotationGroup::get_by_id(conn, &name.to_string())
                    .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows.into())
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn query_by_sample(conn: &GraphConnection, sample_name: &str) -> Vec<AnnotationGroup> {
        let query = "\
            select ag.* \
            from annotation_groups ag \
            join annotation_group_samples s \
                on ag.name = s.annotation_group \
            where s.sample_name = ?1 \
            order by ag.name;";
        AnnotationGroup::query(conn, query, params![sample_name])
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

impl Annotation {
    pub fn generate_id(name: &str, group: &str, accession_id: &HashId) -> HashId {
        HashId(calculate_hash(&format!("{name}:{group}:{accession_id}",)))
    }

    pub fn create(
        conn: &GraphConnection,
        name: &str,
        group: &str,
        accession_id: &HashId,
        extra: Option<&AnnotationExtra>,
    ) -> Result<Annotation, AnnotationError> {
        let id = Annotation::generate_id(name, group, accession_id);
        let query = "INSERT INTO annotations (id, name, annotation_group, accession_id, extra) VALUES (?1, ?2, ?3, ?4, ?5);";
        let mut stmt = conn.prepare(query)?;
        let extra_json = serialize_annotation_extra(extra)?;
        stmt.execute(params![id, name, group, accession_id, extra_json])?;
        Ok(Annotation {
            id,
            name: name.to_string(),
            group: group.to_string(),
            accession_id: *accession_id,
            extra: extra.cloned(),
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
        match Annotation::create(conn, name, group, accession_id, extra) {
            Ok(annotation) => Ok(annotation),
            Err(AnnotationError::DatabaseError(rusqlite::Error::SqliteFailure(err, _details)))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                let id = Annotation::generate_id(name, group, accession_id);
                Ok(Annotation {
                    id,
                    name: name.to_string(),
                    group: group.to_string(),
                    accession_id: *accession_id,
                    extra: extra.cloned(),
                })
            }
            Err(err) => Err(err),
        }
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
        let query = "INSERT OR IGNORE INTO annotation_group_samples (annotation_group, sample_name) VALUES (?1, ?2);";
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
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let query = "select a.* from annotations a left join annotation_group_samples s on (a.annotation_group = s.annotation_group) where s.sample_name = ?1";
        Ok(Annotation::query(conn, query, params![sample_name]))
    }

    pub fn query_by_group(
        conn: &GraphConnection,
        group: &str,
    ) -> Result<Vec<Annotation>, AnnotationError> {
        let query = "select * from annotations where annotation_group = ?1";
        Ok(Annotation::query(conn, query, params![group]))
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
        let query = "INSERT OR IGNORE INTO annotation_group_samples (annotation_group, sample_name) VALUES (?1, ?2);";
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

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct AnnotationFileInfo {
    pub file_addition: FileAddition,
    pub index_file_addition: Option<FileAddition>,
    pub name: Option<String>,
    pub file_path: String,
    pub index_file_path: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AnnotationFileAdditionInput {
    pub file_path: String,
    pub file_type: FileTypes,
    pub checksum_override: Option<HashId>,
    pub name: Option<String>,
    pub index_file_path: Option<String>,
}

#[derive(Debug, Error)]
pub enum AnnotationFileError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("File addition error: {0}")]
    FileAdditionError(#[from] FileAdditionError),
    #[error("Index file must be Tabix, got: {0:?}")]
    InvalidIndexFileType(FileTypes),
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
) -> Result<Operation, Box<dyn std::error::Error>> {
    let graph_conn = context.graph().conn();
    let operation_conn = context.operations().conn();
    let parsed_region = Region::parse(region)?;
    let resolved_region = crate::region::resolve(&parsed_region, graph_conn, collection, sample)?;
    let spans = AccessionSpan::from_resolved_region(graph_conn, &resolved_region, None)?;

    let mut session = start_operation(graph_conn);
    graph_conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let accession = Accession::create(
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

    let operation = end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![],
            description: format!("add annotation {name}"),
        },
        &format!("add annotation {name}"),
        None,
    )?;

    graph_conn.execute("END TRANSACTION", [])?;
    operation_conn.execute("END TRANSACTION", [])?;

    Ok(operation)
}

#[cfg(all(feature = "remote-storage", not(target_os = "emscripten")))]
pub fn add_annotation_file(
    context: &DbContext,
    path: &str,
    format: Option<&str>,
    index: Option<&str>,
    name: Option<&str>,
    message: Option<&str>,
) -> Result<Operation, Box<dyn std::error::Error>> {
    let workspace = context.workspace();
    let operation_conn = context.operations().conn();
    let graph_conn = context.graph().conn();
    let db_uuid = metadata::get_db_uuid(graph_conn);

    let file_type = match format {
        Some(format) => parse_annotation_file_type(format)?,
        None => {
            let ext = annotation_file_extension(path).ok_or_else(|| {
                anyhow!(
                    "Unable to detect annotation file format from the file extension. Use --format to specify it explicitly."
                )
            })?;
            parse_annotation_file_type(&ext)?
        }
    };
    let file_addition =
        FileAddition::get_or_create(workspace, operation_conn, path, file_type, None)?;
    let index_file_path = annotation_index_file_path(workspace, path, index);
    let index_file_addition = if let Some(index_file_path) = index_file_path {
        Some(FileAddition::get_or_create(
            workspace,
            operation_conn,
            &index_file_path,
            FileTypes::Tabix,
            None,
        )?)
    } else {
        None
    };

    let stored_file_path =
        OperationFile::storage_file_path(workspace, path, &file_addition.checksum)?;
    let stored_index_file_path = if let Some(index_file_addition) = index_file_addition.as_ref() {
        Some(OperationFile::storage_file_path(
            workspace,
            path,
            &index_file_addition.checksum,
        )?)
    } else {
        None
    };

    let name_value = name.unwrap_or_default();
    let index_file_addition_id = index_file_addition
        .as_ref()
        .map(|index_file| index_file.id.to_string())
        .unwrap_or_default();
    let stored_index_file_path_value = stored_index_file_path.as_deref().unwrap_or_default();

    let operation_hash = HashId(calculate_hash(&format!(
        "{file_addition_id}:{name_value}:{index_file_addition_id}:{stored_file_path}:{stored_index_file_path_value}",
        file_addition_id = file_addition.id
    )));
    let operation = match Operation::create(operation_conn, "annotation-file", &operation_hash) {
        Ok(operation) => operation,
        Err(rusqlite::Error::SqliteFailure(err, _details))
            if err.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            return Err(OperationError::NoChanges.into());
        }
        Err(err) => return Err(err.into()),
    };
    AnnotationFile::link_to_operation(
        operation_conn,
        &operation.hash,
        &file_addition.id,
        index_file_addition
            .as_ref()
            .map(|index_file| &index_file.id),
        name,
        &stored_file_path,
        stored_index_file_path.as_deref(),
    )?;
    Operation::add_database(operation_conn, &operation.hash, &db_uuid)?;
    let summary = message
        .map(str::to_string)
        .unwrap_or_else(|| format!("Add annotation file {path}"));
    OperationSummary::create(operation_conn, &operation.hash, &summary);

    let gen_db = GenDatabase::get_by_uuid(operation_conn, &db_uuid)?;
    write_changeset(
        workspace,
        &operation,
        DatabaseChangeset {
            db_path: gen_db.path,
            changes: ChangesetModels::default(),
        },
        &DependencyModels::default(),
    );

    if file_type != FileTypes::Changeset && file_type != FileTypes::None {
        file_addition.store_file(workspace)?;
        if let Some(index_file_addition) = index_file_addition {
            index_file_addition.store_file(workspace)?;
        }
    }

    Ok(operation)
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AnnotationFile {
    pub id: i64,
    pub operation_hash: HashId,
    pub file_addition_id: HashId,
    pub index_file_addition_id: Option<HashId>,
    pub name: Option<String>,
    pub file_path: String,
    pub index_file_path: Option<String>,
}

impl Query for AnnotationFile {
    type Model = AnnotationFile;

    const TABLE_NAME: &'static str = "annotation_files";

    fn process_row(row: &Row) -> Self::Model {
        AnnotationFile {
            id: row.get(0).unwrap(),
            operation_hash: row.get(1).unwrap(),
            file_addition_id: row.get(2).unwrap(),
            index_file_addition_id: row.get(3).unwrap(),
            name: row.get(4).unwrap(),
            file_path: row.get(5).unwrap(),
            index_file_path: row.get(6).unwrap(),
        }
    }
}

impl AnnotationFile {
    pub fn load_index(
        conn: &OperationsConnection,
        file_addition_id: Option<&HashId>,
    ) -> Result<Option<FileAddition>, AnnotationFileError> {
        let Some(file_addition_id) = file_addition_id else {
            return Ok(None);
        };
        let index_file_addition = FileAddition::get_by_id(conn, file_addition_id).ok_or(
            AnnotationFileError::DatabaseError(rusqlite::Error::QueryReturnedNoRows),
        )?;
        if index_file_addition.file_type != FileTypes::Tabix {
            return Err(AnnotationFileError::InvalidIndexFileType(
                index_file_addition.file_type,
            ));
        }
        Ok(Some(index_file_addition))
    }

    pub fn link_to_operation(
        conn: &OperationsConnection,
        operation_hash: &HashId,
        file_addition_id: &HashId,
        index_file_addition_id: Option<&HashId>,
        name: Option<&str>,
        file_path: &str,
        index_file_path: Option<&str>,
    ) -> Result<(), AnnotationFileError> {
        AnnotationFile::load_index(conn, index_file_addition_id)?;
        let query = "INSERT INTO annotation_files (operation_hash, file_addition_id, index_file_addition_id, name, file_path, index_file_path) VALUES (?1, ?2, ?3, ?4, ?5, ?6);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            operation_hash,
            file_addition_id,
            index_file_addition_id,
            name,
            file_path,
            index_file_path
        ])?;
        Ok(())
    }

    #[cfg(all(feature = "remote-storage", not(target_os = "emscripten")))]
    pub fn add_to_operation(
        workspace: &Workspace,
        conn: &OperationsConnection,
        operation_hash: &HashId,
        input: &AnnotationFileAdditionInput,
    ) -> Result<FileAddition, AnnotationFileError> {
        let file_addition = FileAddition::get_or_create(
            workspace,
            conn,
            &input.file_path,
            input.file_type,
            input.checksum_override,
        )?;
        let index_file_addition = if let Some(index_file_path) = input.index_file_path.as_ref() {
            Some(FileAddition::get_or_create(
                workspace,
                conn,
                index_file_path,
                FileTypes::Tabix,
                None,
            )?)
        } else {
            None
        };

        let file_path =
            OperationFile::storage_file_path(workspace, &input.file_path, &file_addition.checksum)?;
        let index_file_path = if let Some(index_file_addition) = index_file_addition.as_ref() {
            if let Some(index_file_path) = input.index_file_path.as_deref() {
                Some(OperationFile::storage_file_path(
                    workspace,
                    index_file_path,
                    &index_file_addition.checksum,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        AnnotationFile::link_to_operation(
            conn,
            operation_hash,
            &file_addition.id,
            index_file_addition.as_ref().map(|index| &index.id),
            input.name.as_deref(),
            &file_path,
            index_file_path.as_deref(),
        )?;
        Ok(file_addition)
    }

    pub fn get_files_for_operation(
        conn: &OperationsConnection,
        operation_hash: &HashId,
    ) -> Vec<AnnotationFileInfo> {
        let query = "select fa.*, af.index_file_addition_id, af.name, af.file_path, af.index_file_path from file_additions fa join annotation_files af on (fa.id = af.file_addition_id) where af.operation_hash = ?1";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(params![operation_hash], |row| {
                Ok((
                    FileAddition::process_row(row),
                    row.get::<_, Option<HashId>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                ))
            })
            .unwrap();
        rows.map(|row| {
            let (file_addition, index_file_addition_id, name, file_path, index_file_path) =
                row.unwrap();
            AnnotationFileInfo {
                file_addition,
                index_file_addition: AnnotationFile::load_index(
                    conn,
                    index_file_addition_id.as_ref(),
                )
                .unwrap(),
                name,
                file_path,
                index_file_path,
            }
        })
        .collect()
    }

    pub fn get_all_files(conn: &OperationsConnection) -> Vec<AnnotationFileInfo> {
        let query = "select fa.*, af.index_file_addition_id, af.name, af.file_path, af.index_file_path from file_additions fa join annotation_files af on (fa.id = af.file_addition_id)";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    FileAddition::process_row(row),
                    row.get::<_, Option<HashId>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                ))
            })
            .unwrap();
        let mut entries: Vec<AnnotationFileInfo> = rows
            .map(|row| {
                let (file_addition, index_file_addition_id, name, file_path, index_file_path) =
                    row.unwrap();
                AnnotationFileInfo {
                    file_addition,
                    index_file_addition: AnnotationFile::load_index(
                        conn,
                        index_file_addition_id.as_ref(),
                    )
                    .unwrap(),
                    name,
                    file_path,
                    index_file_path,
                }
            })
            .collect();
        entries.sort_by(|a, b| {
            let a_path = a.file_addition.file_path();
            let b_path = b.file_addition.file_path();
            let a_name = std::path::Path::new(a_path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| a_path.to_string());
            let b_name = std::path::Path::new(b_path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| b_path.to_string());
            a_name.cmp(&b_name).then_with(|| a_path.cmp(b_path))
        });
        entries
    }

    pub fn query_by_operations(
        conn: &OperationsConnection,
        operations: &[HashId],
    ) -> Result<HashMap<HashId, Vec<FileAddition>>, AnnotationFileError> {
        let query = "select fa.*, af.operation_hash from file_additions fa left join annotation_files af on (fa.id = af.file_addition_id) where af.operation_hash in rarray(?1)";
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(
            params![Rc::new(
                operations
                    .iter()
                    .map(|h| Value::from(*h))
                    .collect::<Vec<Value>>()
            )],
            |row| Ok((FileAddition::process_row(row), row.get::<_, HashId>(4)?)),
        )?;
        rows.into_iter()
            .try_fold(HashMap::new(), |mut acc: HashMap<_, Vec<_>>, row| {
                let (item, hash) = row?;
                acc.entry(hash).or_default().push(item);
                Ok(acc)
            })
            .map_err(AnnotationFileError::DatabaseError)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs};

    use gen_core::{HashId, region::RegionResolutionError};

    use super::*;
    use crate::{
        block_group::{BlockGroup, PathCache},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        errors::OperationError,
        files::GenDatabase,
        metadata,
        path::Path,
        path_edge::PathEdge,
        sample::Sample,
        sample_lineage::SampleLineage,
        test_helpers::{create_bg, get_connection, setup_block_group, setup_gen},
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
            let edge_ids = PathEdge::edges_for_path(&conn, &path.id)
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
            let edge_ids = PathEdge::edges_for_path(&conn, &parent_path.id)
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
            let edge_ids = PathEdge::edges_for_path(&conn, &parent_path.id)
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
            let edge_ids = PathEdge::edges_for_path(conn, &parent_path.id)
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

        let annotation =
            Annotation::get_or_create(&conn, "gene-a", "project-tracks", &accession.id, None)
                .unwrap();
        annotation
            .add_samples(&conn, &["sample-1", "sample-2"])
            .unwrap();

        let mut samples = Annotation::get_samples(&conn, &annotation.group).unwrap();
        samples.sort();
        assert_eq!(
            samples,
            vec!["sample-1".to_string(), "sample-2".to_string()]
        );

        let by_sample = Annotation::query_by_sample(&conn, "sample-1").unwrap();
        assert_eq!(by_sample.len(), 1);
        assert_eq!(by_sample[0], annotation);

        let by_group = Annotation::query_by_group(&conn, "project-tracks").unwrap();
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
    fn add_annotation_file_to_operation() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let workspace = context.workspace();
        let repo_root = workspace.repo_root().unwrap();
        let annotation_path = repo_root.join("fixtures").join("annotation.gff3");
        fs::create_dir_all(annotation_path.parent().unwrap()).unwrap();
        fs::write(&annotation_path, "##gff-version 3\n").unwrap();

        let op_hash = HashId::random_str();
        let _ = crate::operations::Operation::create(op_conn, "annotation-file", &op_hash)
            .expect("should create operation");

        let file_addition = AnnotationFile::add_to_operation(
            workspace,
            op_conn,
            &op_hash,
            &AnnotationFileAdditionInput {
                file_path: annotation_path.to_string_lossy().to_string(),
                file_type: FileTypes::Gff3,
                checksum_override: None,
                name: Some("fixtures-annotation".to_string()),
                index_file_path: None,
            },
        )
        .unwrap();

        let files = AnnotationFile::get_files_for_operation(op_conn, &op_hash);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_addition, file_addition);
        assert_eq!(files[0].file_path, "fixtures/annotation.gff3");
        assert!(files[0].index_file_addition.is_none());
        assert!(files[0].index_file_path.is_none());
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
        let operation_conn = context.operations().conn();
        let db_uuid = metadata::get_db_uuid(graph_conn);
        let _ = GenDatabase::create(operation_conn, &db_uuid, "test-db", "test-db-path").unwrap();
        let _ = setup_block_group(graph_conn);

        let operation = add_annotation(
            &context,
            "test",
            "gene-a",
            Some("track-1"),
            "test",
            "chr1:1-5",
        )
        .unwrap();
        assert_eq!(operation.change_type, "add annotation gene-a");

        let annotations = Annotation::query_by_group(graph_conn, "track-1").unwrap();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0].name, "gene-a");
    }

    #[test]
    fn add_annotation_file_creates_operation() {
        let context = setup_gen();
        let graph_conn = context.graph().conn();
        let operation_conn = context.operations().conn();
        let db_uuid = metadata::get_db_uuid(graph_conn);
        let _ = GenDatabase::create(operation_conn, &db_uuid, "test-db", "test-db-path").unwrap();

        let repo_root = context.workspace().repo_root().unwrap();
        let annotation_path = repo_root.join("fixtures").join("annotation.gff3");
        fs::create_dir_all(annotation_path.parent().unwrap()).unwrap();
        fs::write(&annotation_path, "##gff-version 3\n").unwrap();
        let annotation_path_str = annotation_path.to_string_lossy().to_string();

        let operation = add_annotation_file(
            &context,
            &annotation_path_str,
            None,
            None,
            Some("track-1"),
            None,
        )
        .unwrap();
        assert_eq!(operation.change_type, "annotation-file");

        let files = AnnotationFile::get_files_for_operation(operation_conn, &operation.hash);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name.as_deref(), Some("track-1"));

        let err = add_annotation_file(
            &context,
            &annotation_path_str,
            None,
            None,
            Some("track-1"),
            None,
        )
        .unwrap_err();
        let op_err = err
            .downcast_ref::<OperationError>()
            .expect("should be an OperationError");
        assert_eq!(*op_err, OperationError::NoChanges);
    }
}
