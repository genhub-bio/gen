use std::{collections::HashMap, fmt, rc::Rc};

use gen_core::{HashId, calculate_hash, config::Workspace, traits::Capnp};
use rusqlite::{
    Row, params,
    types::{FromSql, FromSqlResult, ToSql, ToSqlOutput, Value, ValueRef},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    db::{GraphConnection, OperationsConnection},
    errors::FileAdditionError,
    file_types::FileTypes,
    gen_models_capnp::{annotation_sample, stored_annotation},
    operations::FileAddition,
    traits::Query,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq, Deserialize, Serialize)]
pub struct AnnotationKind(String);

impl AnnotationKind {
    pub fn new(value: impl AsRef<str>) -> Self {
        AnnotationKind(value.as_ref().trim().to_ascii_lowercase())
    }

    pub fn gff3() -> Self {
        AnnotationKind::new("gff3")
    }

    pub fn bed() -> Self {
        AnnotationKind::new("bed")
    }

    pub fn genbank() -> Self {
        AnnotationKind::new("genbank")
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn to_file_type(&self) -> Option<FileTypes> {
        match self.0.as_str() {
            "gff3" | "gff" => Some(FileTypes::Gff3),
            "bed" => Some(FileTypes::Bed),
            "genbank" | "gb" => Some(FileTypes::GenBank),
            _ => None,
        }
    }
}

impl fmt::Display for AnnotationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for AnnotationKind {
    fn from(value: &str) -> Self {
        AnnotationKind::new(value)
    }
}

impl From<String> for AnnotationKind {
    fn from(value: String) -> Self {
        AnnotationKind::new(value)
    }
}

impl ToSql for AnnotationKind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Owned(Value::Text(self.0.clone())))
    }
}

impl FromSql for AnnotationKind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let raw = value.as_str()?;
        Ok(AnnotationKind::new(raw))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Annotation {
    pub id: HashId,
    pub name: String,
    pub kind: AnnotationKind,
    pub accession_id: HashId,
}

impl<'a> Capnp<'a> for Annotation {
    type Builder = stored_annotation::Builder<'a>;
    type Reader = stored_annotation::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_name(&self.name);
        builder.set_annotation_type(self.kind.as_str());
        builder.set_accession_id(&self.accession_id.0).unwrap();
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
        let kind = AnnotationKind::new(reader.get_annotation_type().unwrap().to_string().unwrap());
        let accession_id = reader
            .get_accession_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();

        Annotation {
            id,
            name,
            kind,
            accession_id,
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
            kind: row.get(2).unwrap(),
            accession_id: row.get(3).unwrap(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AnnotationSample {
    pub annotation_id: HashId,
    pub sample_name: String,
}

impl<'a> Capnp<'a> for AnnotationSample {
    type Builder = annotation_sample::Builder<'a>;
    type Reader = annotation_sample::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_annotation_id(&self.annotation_id.0).unwrap();
        builder.set_sample_name(&self.sample_name);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let annotation_id = reader
            .get_annotation_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let sample_name = reader.get_sample_name().unwrap().to_string().unwrap();
        AnnotationSample {
            annotation_id,
            sample_name,
        }
    }
}

impl AnnotationSample {
    pub fn insert(
        conn: &GraphConnection,
        annotation_sample: &AnnotationSample,
    ) -> Result<(), AnnotationError> {
        let query =
            "INSERT OR IGNORE INTO annotations_sample (annotation_id, sample_name) VALUES (?1, ?2);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            annotation_sample.annotation_id,
            annotation_sample.sample_name
        ])?;
        Ok(())
    }

    pub fn create(
        conn: &GraphConnection,
        annotation_id: &HashId,
        sample_name: &str,
    ) -> Result<(), AnnotationError> {
        let query =
            "INSERT OR IGNORE INTO annotations_sample (annotation_id, sample_name) VALUES (?1, ?2);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![annotation_id, sample_name])?;
        Ok(())
    }

    pub fn delete(
        conn: &GraphConnection,
        annotation_id: &HashId,
        sample_name: &str,
    ) -> Result<(), AnnotationError> {
        let query =
            "DELETE FROM annotations_sample WHERE annotation_id = ?1 AND sample_name = ?2;";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![annotation_id, sample_name])?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum AnnotationError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl Annotation {
    pub fn generate_id(name: &str, kind: &AnnotationKind, accession_id: &HashId) -> HashId {
        HashId(calculate_hash(&format!("{accession_id}:{name}:{kind}",)))
    }

    pub fn insert(conn: &GraphConnection, annotation: &Annotation) -> Result<(), AnnotationError> {
        let query = "INSERT OR IGNORE INTO annotations (id, name, annotation_type, accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            annotation.id,
            annotation.name,
            annotation.kind,
            annotation.accession_id
        ])?;
        Ok(())
    }

    pub fn create(
        conn: &GraphConnection,
        name: &str,
        kind: AnnotationKind,
        accession_id: &HashId,
    ) -> Result<Annotation, AnnotationError> {
        let id = Annotation::generate_id(name, &kind, accession_id);
        let query = "INSERT INTO annotations (id, name, annotation_type, accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query)?;
        let insert_result = stmt.execute(params![id, name, kind, accession_id]);
        match insert_result {
            Ok(_) => Ok(Annotation {
                id,
                name: name.to_string(),
                kind,
                accession_id: *accession_id,
            }),
            Err(rusqlite::Error::SqliteFailure(err, _details))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Annotation::get_by_id(conn, &id)
                    .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows.into())
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn create_with_samples(
        conn: &GraphConnection,
        name: &str,
        kind: AnnotationKind,
        accession_id: &HashId,
        sample_names: &[&str],
    ) -> Result<Annotation, AnnotationError> {
        let annotation = Annotation::create(conn, name, kind, accession_id)?;
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
        let query = "INSERT OR IGNORE INTO annotations_sample (annotation_id, sample_name) VALUES (?1, ?2);";
        let mut stmt = conn.prepare(query)?;
        for sample_name in sample_names {
            stmt.execute(params![self.id, sample_name])?;
        }
        Ok(())
    }

    pub fn get_samples(
        conn: &GraphConnection,
        annotation_id: &HashId,
    ) -> Result<Vec<String>, AnnotationError> {
        let query = "SELECT sample_name FROM annotations_sample WHERE annotation_id = ?1;";
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map(params![annotation_id], |row| row.get(0))?;
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
        let query = "select a.* from annotations a left join annotations_sample s on (a.id = s.annotation_id) where s.sample_name = ?1";
        Ok(Annotation::query(conn, query, params![sample_name]))
    }
}

#[derive(Debug, Error)]
pub enum AnnotationFileError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("File addition error: {0}")]
    FileAdditionError(#[from] FileAdditionError),
    #[error("Unsupported annotation file type: {0}")]
    UnsupportedFileType(String),
}

pub fn parse_annotation_file_type(value: &str) -> Result<FileTypes, AnnotationFileError> {
    let annotation_kind = AnnotationKind::new(value);
    annotation_kind
        .to_file_type()
        .ok_or_else(|| AnnotationFileError::UnsupportedFileType(value.to_string()))
}

pub struct AnnotationFile;

impl AnnotationFile {
    pub fn link_to_operation(
        conn: &OperationsConnection,
        operation_hash: &HashId,
        file_addition_id: &HashId,
    ) -> Result<(), AnnotationFileError> {
        let query =
            "INSERT INTO annotation_files (operation_hash, file_addition_id) VALUES (?1, ?2);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![operation_hash, file_addition_id])?;
        Ok(())
    }

    pub fn add_to_operation(
        workspace: &Workspace,
        conn: &OperationsConnection,
        operation_hash: &HashId,
        file_path: &str,
        file_type: FileTypes,
        checksum_override: Option<HashId>,
    ) -> Result<FileAddition, AnnotationFileError> {
        let file_addition =
            FileAddition::get_or_create(workspace, conn, file_path, file_type, checksum_override)?;
        AnnotationFile::link_to_operation(conn, operation_hash, &file_addition.id)?;
        Ok(file_addition)
    }

    pub fn get_files_for_operation(
        conn: &OperationsConnection,
        operation_hash: &HashId,
    ) -> Vec<FileAddition> {
        let query = "select fa.* from file_additions fa left join annotation_files af on (fa.id = af.file_addition_id) where af.operation_hash = ?1";
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map(params![operation_hash], |row| {
                Ok(FileAddition::process_row(row))
            })
            .unwrap();
        rows.map(|row| row.unwrap()).collect()
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
        sample::Sample,
        test_helpers::{setup_block_group, setup_gen},
    };

    #[test]
    fn create_annotation_with_samples() {
        let conn = crate::test_helpers::get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);

        let _ = Sample::create(&conn, "sample-1").unwrap();
        let _ = Sample::create(&conn, "sample-2").unwrap();

        let mut cache = PathCache::new(&conn);
        let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone());
        let accession = BlockGroup::add_accession(&conn, &path, "ann-accession", 0, 5, &mut cache);

        let annotation =
            Annotation::create(&conn, "gene-a", AnnotationKind::gff3(), &accession.id).unwrap();
        annotation
            .add_samples(&conn, &["sample-1", "sample-2"])
            .unwrap();

        let mut samples = Annotation::get_samples(&conn, &annotation.id).unwrap();
        samples.sort();
        assert_eq!(
            samples,
            vec!["sample-1".to_string(), "sample-2".to_string()]
        );

        let by_sample = Annotation::query_by_sample(&conn, "sample-1").unwrap();
        assert_eq!(by_sample.len(), 1);
        assert_eq!(by_sample[0], annotation);
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
            annotation_path.to_string_lossy().as_ref(),
            FileTypes::Gff3,
            None,
        )
        .unwrap();

        let files = AnnotationFile::get_files_for_operation(op_conn, &op_hash);
        assert_eq!(files, vec![file_addition]);
    }
}
