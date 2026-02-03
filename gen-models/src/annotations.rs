use std::{collections::HashMap, rc::Rc};

use gen_core::{HashId, calculate_hash, config::Workspace, traits::Capnp};
use rusqlite::{Row, params, types::Value};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    db::{GraphConnection, OperationsConnection},
    errors::FileAdditionError,
    file_types::FileTypes,
    gen_models_capnp::{annotation_group, annotation_group_sample, stored_annotation},
    operations::FileAddition,
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

    pub fn get_or_create(conn: &GraphConnection, name: &str) -> AnnotationGroup {
        match AnnotationGroup::create(conn, name) {
            Ok(group) => group,
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    AnnotationGroup {
                        name: name.to_string(),
                    }
                } else {
                    panic!("something bad happened querying the database")
                }
            }
            Err(_) => {
                panic!("something bad happened.")
            }
        }
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
}

impl<'a> Capnp<'a> for Annotation {
    type Builder = stored_annotation::Builder<'a>;
    type Reader = stored_annotation::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_name(&self.name);
        builder.set_annotation_group(&self.group);
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
        let group = reader.get_annotation_group().unwrap().to_string().unwrap();
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
            group,
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
            group: row.get(2).unwrap(),
            accession_id: row.get(3).unwrap(),
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
    pub fn insert(
        conn: &GraphConnection,
        annotation_sample: &AnnotationGroupSample,
    ) -> Result<(), AnnotationError> {
        AnnotationGroup::get_or_create(conn, &annotation_sample.annotation_group);
        let query = "INSERT OR IGNORE INTO annotation_group_samples (annotation_group, sample_name) VALUES (?1, ?2);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            annotation_sample.annotation_group,
            annotation_sample.sample_name
        ])?;
        Ok(())
    }

    pub fn create(
        conn: &GraphConnection,
        annotation_group: &str,
        sample_name: &str,
    ) -> Result<(), AnnotationError> {
        AnnotationGroup::get_or_create(conn, annotation_group);
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

#[derive(Debug, Error)]
pub enum AnnotationError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl Annotation {
    pub fn generate_id(name: &str, group: &str, accession_id: &HashId) -> HashId {
        HashId(calculate_hash(&format!("{accession_id}:{name}:{group}",)))
    }

    pub fn insert(conn: &GraphConnection, annotation: &Annotation) -> Result<(), AnnotationError> {
        AnnotationGroup::get_or_create(conn, &annotation.group);
        let query = "INSERT OR IGNORE INTO annotations (id, name, annotation_group, accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            annotation.id,
            annotation.name,
            annotation.group,
            annotation.accession_id
        ])?;
        Ok(())
    }

    pub fn create(
        conn: &GraphConnection,
        name: &str,
        group: &str,
        accession_id: &HashId,
    ) -> Result<Annotation, AnnotationError> {
        AnnotationGroup::get_or_create(conn, group);
        let id = Annotation::generate_id(name, group, accession_id);
        let query = "INSERT INTO annotations (id, name, annotation_group, accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query)?;
        let insert_result = stmt.execute(params![id, name, group, accession_id]);
        match insert_result {
            Ok(_) => Ok(Annotation {
                id,
                name: name.to_string(),
                group: group.to_string(),
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
        group: &str,
        accession_id: &HashId,
        sample_names: &[&str],
    ) -> Result<Annotation, AnnotationError> {
        let annotation = Annotation::create(conn, name, group, accession_id)?;
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
        AnnotationGroup::get_or_create(conn, &self.group);
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
    match value.trim().to_ascii_lowercase().as_str() {
        "gff3" | "gff" => Ok(FileTypes::Gff3),
        "bed" => Ok(FileTypes::Bed),
        "genbank" | "gb" => Ok(FileTypes::GenBank),
        other => Err(AnnotationFileError::UnsupportedFileType(other.to_string())),
    }
}

pub struct AnnotationFile;

impl AnnotationFile {
    pub fn link_to_operation(
        conn: &OperationsConnection,
        operation_hash: &HashId,
        file_addition_id: &HashId,
        name: Option<&str>,
    ) -> Result<(), AnnotationFileError> {
        let query = "INSERT INTO annotation_files (operation_hash, file_addition_id, name) VALUES (?1, ?2, ?3);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![operation_hash, file_addition_id, name])?;
        Ok(())
    }

    pub fn add_to_operation(
        workspace: &Workspace,
        conn: &OperationsConnection,
        operation_hash: &HashId,
        file_path: &str,
        file_type: FileTypes,
        checksum_override: Option<HashId>,
        name: Option<&str>,
    ) -> Result<FileAddition, AnnotationFileError> {
        let file_addition =
            FileAddition::get_or_create(workspace, conn, file_path, file_type, checksum_override)?;
        AnnotationFile::link_to_operation(conn, operation_hash, &file_addition.id, name)?;
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
            Annotation::create(&conn, "gene-a", "project-tracks", &accession.id).unwrap();
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
            Some("fixtures-annotation"),
        )
        .unwrap();

        let files = AnnotationFile::get_files_for_operation(op_conn, &op_hash);
        assert_eq!(files, vec![file_addition]);
    }
}
