use gen_core::{HashId, traits::Capnp};
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    db::GraphConnection,
    gen_models_capnp::{annotation_sample, stored_annotation},
    traits::Query,
};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Annotation {
    pub id: HashId,
    pub name: String,
    pub annotation_type: String,
    pub accession_id: HashId,
}

impl<'a> Capnp<'a> for Annotation {
    type Builder = stored_annotation::Builder<'a>;
    type Reader = stored_annotation::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_name(&self.name);
        builder.set_annotation_type(&self.annotation_type);
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
        let annotation_type = reader.get_annotation_type().unwrap().to_string().unwrap();
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
            annotation_type,
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
            annotation_type: row.get(2).unwrap(),
            accession_id: row.get(3).unwrap(),
        }
    }
}

impl Annotation {
    pub fn insert(conn: &GraphConnection, annotation: &Annotation) -> Result<(), AnnotationError> {
        let query = "INSERT OR IGNORE INTO annotations (id, name, annotation_type, accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            annotation.id,
            annotation.name,
            annotation.annotation_type,
            annotation.accession_id
        ])?;
        Ok(())
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
        let query = "INSERT OR IGNORE INTO annotations_sample (annotation_id, sample_name) VALUES (?1, ?2);";
        let mut stmt = conn.prepare(query)?;
        stmt.execute(params![
            annotation_sample.annotation_id,
            annotation_sample.sample_name
        ])?;
        Ok(())
    }

    pub fn delete(
        conn: &GraphConnection,
        annotation_id: &HashId,
        sample_name: &str,
    ) -> Result<(), AnnotationError> {
        let query = "DELETE FROM annotations_sample WHERE annotation_id = ?1 AND sample_name = ?2;";
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
