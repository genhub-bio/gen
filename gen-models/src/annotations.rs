use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
};

use anyhow::anyhow;
use gen_core::{
    HashId, NodeIntervalBlock, calculate_hash, config::Workspace, region::Region, traits::Capnp,
};
use intervaltree::IntervalTree;
use rusqlite::{Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    accession::Accession,
    block_group::{BlockGroup, PathCache},
    changesets::{ChangesetModels, DatabaseChangeset, write_changeset},
    db::{DbContext, GraphConnection, OperationsConnection},
    errors::{FileAdditionError, OperationError},
    file_types::FileTypes,
    files::GenDatabase,
    gen_models_capnp::{annotation, annotation_group, annotation_group_sample},
    metadata,
    operations::{FileAddition, Operation, OperationInfo, OperationSummary},
    sample::Sample,
    session_operations::{DependencyModels, end_operation, start_operation},
    traits::Query,
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

    pub fn intervaltree(&self, conn: &GraphConnection) -> IntervalTree<i64, NodeIntervalBlock> {
        Accession::get_by_id(conn, &self.accession_id)
            .unwrap_or_else(|| {
                panic!(
                    "Missing accession {} for annotation {}",
                    self.accession_id, self.id
                )
            })
            .intervaltree(conn)
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
    let start = parsed_region.start;
    let end = parsed_region.end;

    let block_groups = Sample::get_block_groups(graph_conn, collection, sample);
    let block_group = block_groups
        .iter()
        .find(|bg| bg.name == parsed_region.name)
        .ok_or_else(|| anyhow!("Graph {} not found for sample {sample}", parsed_region.name))?;
    let path = BlockGroup::get_current_path(graph_conn, &block_group.id);
    let path_length = path.length(graph_conn)?;
    if start < 0 || end < 0 || start > end || end > path_length {
        return Err(anyhow!("Region {region} is outside the path bounds (0-{path_length})").into());
    }

    let mut session = start_operation(graph_conn);
    graph_conn.execute("BEGIN TRANSACTION", [])?;
    operation_conn.execute("BEGIN TRANSACTION", [])?;

    let mut cache = PathCache::new(graph_conn);
    let _ = PathCache::lookup(&mut cache, &block_group.id, path.name.clone())?;
    let accession = BlockGroup::add_accession(graph_conn, &path, name, start, end, &mut cache)?;

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
    let index_file_addition = annotation_index_file_path(workspace, path, index)
        .map(|index_path| {
            FileAddition::get_or_create(
                workspace,
                operation_conn,
                &index_path,
                FileTypes::Tabix,
                None,
            )
        })
        .transpose()?;
    let name_value = name.unwrap_or_default();
    let index_file_addition_id = index_file_addition
        .as_ref()
        .map(|index_file| index_file.id.to_string())
        .unwrap_or_default();
    let operation_hash = HashId(calculate_hash(&format!(
        "{file_addition_id}:{name_value}:{index_file_addition_id}",
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
    ) -> Result<(), AnnotationFileError> {
        AnnotationFile::load_index(conn, index_file_addition_id)?;
        let query = "INSERT INTO annotation_files (operation_hash, file_addition_id, index_file_addition_id, name) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            operation_hash,
            file_addition_id,
            index_file_addition_id,
            name
        ])?;
        Ok(())
    }

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
        let index_file_addition = input
            .index_file_path
            .as_deref()
            .map(|path| FileAddition::get_or_create(workspace, conn, path, FileTypes::Tabix, None))
            .transpose()?;
        AnnotationFile::link_to_operation(
            conn,
            operation_hash,
            &file_addition.id,
            index_file_addition.as_ref().map(|index| &index.id),
            input.name.as_deref(),
        )?;
        Ok(file_addition)
    }

    pub fn get_files_for_operation(
        conn: &OperationsConnection,
        operation_hash: &HashId,
    ) -> Vec<AnnotationFileInfo> {
        let query = "select fa.*, af.index_file_addition_id, af.name from file_additions fa join annotation_files af on (fa.id = af.file_addition_id) where af.operation_hash = ?1";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(params![operation_hash], |row| {
                Ok((
                    FileAddition::process_row(row),
                    row.get::<_, Option<HashId>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                ))
            })
            .unwrap();
        rows.map(|row| {
            let (file_addition, index_file_addition_id, name) = row.unwrap();
            AnnotationFileInfo {
                file_addition,
                index_file_addition: AnnotationFile::load_index(
                    conn,
                    index_file_addition_id.as_ref(),
                )
                .unwrap(),
                name,
            }
        })
        .collect()
    }

    pub fn get_all_files(conn: &OperationsConnection) -> Vec<AnnotationFileInfo> {
        let query = "select fa.*, af.index_file_addition_id, af.name from file_additions fa join annotation_files af on (fa.id = af.file_addition_id)";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    FileAddition::process_row(row),
                    row.get::<_, Option<HashId>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                ))
            })
            .unwrap();
        let mut entries: Vec<AnnotationFileInfo> = rows
            .map(|row| {
                let (file_addition, index_file_addition_id, name) = row.unwrap();
                AnnotationFileInfo {
                    file_addition,
                    index_file_addition: AnnotationFile::load_index(
                        conn,
                        index_file_addition_id.as_ref(),
                    )
                    .unwrap(),
                    name,
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
    use std::fs;

    use gen_core::HashId;

    use super::*;
    use crate::{
        block_group::{BlockGroup, PathCache},
        errors::OperationError,
        files::GenDatabase,
        metadata,
        sample::Sample,
        test_helpers::{get_connection, setup_block_group, setup_gen},
    };

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
        assert!(files[0].index_file_addition.is_none());
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
