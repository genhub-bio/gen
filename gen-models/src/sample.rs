use std::{collections::HashSet, rc::Rc};

use gen_core::traits::Capnp;
use gen_graph::GenGraph;
use rusqlite::{Result as SQLResult, Row, params, types::Value as SQLValue};
use serde::{Deserialize, Serialize};

use crate::{
    block_group::BlockGroup, db::GraphConnection, errors::SampleError, gen_models_capnp::sample,
    sample_lineage::SampleLineage, traits::Query,
};

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct Sample {
    pub name: String,
}

impl<'a> Capnp<'a> for Sample {
    type Builder = sample::Builder<'a>;
    type Reader = sample::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_name(&self.name);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let name = reader.get_name().unwrap().to_string().unwrap();
        Sample { name }
    }
}

impl Query for Sample {
    type Model = Sample;

    const PRIMARY_KEY: &'static str = "name";
    const TABLE_NAME: &'static str = "samples";

    fn process_row(row: &Row) -> Self::Model {
        Sample {
            name: row.get(0).unwrap(),
        }
    }
}

impl Sample {
    pub const DEFAULT_NAME: &str = "reference";

    pub fn get_parent_names(conn: &GraphConnection, sample_name: &str) -> Vec<String> {
        SampleLineage::get_parents(conn, sample_name)
    }

    pub fn create(conn: &GraphConnection, name: &str) -> Result<Sample, SampleError> {
        let mut stmt =
            match conn.prepare("INSERT INTO samples (name) VALUES (?1) returning (name);") {
                Ok(stmt) => stmt,
                Err(err) => return Err(SampleError::SqliteError(err)),
            };

        match stmt.query_row((name,), |row| Ok(Sample { name: row.get(0)? })) {
            Ok(sample) => Ok(sample),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(SampleError::Duplicate(Sample {
                    name: name.to_string(),
                }))
            }
            Err(err) => Err(SampleError::SqliteError(err)),
        }
    }

    pub fn get_or_create(conn: &GraphConnection, name: &str) -> Result<Sample, SampleError> {
        match Sample::create(conn, name) {
            Ok(sample) => Ok(sample),
            Err(SampleError::Duplicate(sample)) => Ok(sample),
            Err(e) => Err(e),
        }
    }

    pub fn delete_by_name(conn: &GraphConnection, name: &str) {
        let mut stmt = conn.prepare("delete from samples where name = ?1").unwrap();
        stmt.execute([name]).unwrap();
    }

    pub fn get_graph(conn: &GraphConnection, collection: &str, name: &str) -> GenGraph {
        let block_groups = Sample::get_block_groups(conn, collection, name);
        let mut sample_graph = GenGraph::new();
        for bg in block_groups {
            let bg_graph = BlockGroup::get_graph(conn, &bg.id);
            // Add nodes and edges from block group graph to sample graph
            for node in bg_graph.nodes() {
                sample_graph.add_node(node);
            }
            for (source, dest, edges) in bg_graph.all_edges() {
                if let Some(existing_edges) = sample_graph.edge_weight_mut(source, dest) {
                    existing_edges.extend(edges.clone());
                } else {
                    sample_graph.add_edge(source, dest, edges.clone());
                }
            }
        }
        sample_graph
    }

    pub fn get_all_sequences(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        prune: bool,
    ) -> HashSet<String> {
        Sample::get_block_groups(conn, collection_name, sample_name)
            .into_iter()
            .flat_map(|block_group| BlockGroup::get_all_sequences(conn, &block_group.id, prune))
            .collect()
    }

    pub fn get_or_create_child(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        parent_samples: Vec<String>,
    ) -> Result<Sample, SampleError> {
        match Sample::create(conn, sample_name) {
            Ok(new_sample) => {
                if !parent_samples.is_empty() {
                    let parent_block_groups = BlockGroup::query(
                        conn,
                        "select * from block_groups
                         where collection_name = ?1 AND sample_name IN rarray(?2)
                         ORDER BY name, sample_name, created_on, id",
                        params![
                            collection_name,
                            Rc::new(
                                parent_samples
                                    .iter()
                                    .cloned()
                                    .map(SQLValue::from)
                                    .collect::<Vec<_>>()
                            ),
                        ],
                    );
                    let group_names = parent_block_groups
                        .into_iter()
                        .map(|parent_block_group| parent_block_group.name)
                        .collect::<HashSet<_>>();

                    for group_name in group_names {
                        BlockGroup::get_or_create_sample_block_groups(
                            conn,
                            collection_name,
                            &new_sample.name,
                            &group_name,
                            parent_samples.clone(),
                        )
                        .map_err(SampleError::from)?;
                    }

                    for parent_sample in parent_samples {
                        SampleLineage::create(conn, &parent_sample, &new_sample.name)
                            .map_err(SampleError::from)?;
                    }
                }

                Ok(new_sample)
            }
            Err(SampleError::Duplicate(sample)) => Ok(sample),
            Err(e) => Err(e),
        }
    }

    pub fn get_block_groups(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Vec<BlockGroup> {
        BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
            params![collection_name, sample_name],
        )
    }

    pub fn get_all_names(conn: &GraphConnection) -> Vec<String> {
        let samples = Sample::query(conn, "select * from samples;", rusqlite::params!());
        samples.iter().map(|s| s.name.clone()).collect()
    }

    pub fn get_by_name(conn: &GraphConnection, name: &str) -> SQLResult<Sample> {
        Sample::get(
            conn,
            "select * from samples where name = ?1;",
            rusqlite::params!(name),
        )
    }

    pub fn search_name(conn: &GraphConnection, name: &str) -> Vec<Sample> {
        Sample::query(
            conn,
            "select * from samples
             where instr(lower(name), lower(?1)) > 0
             order by name;",
            rusqlite::params!(name),
        )
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::{
        collection::Collection,
        errors::SampleError,
        test_helpers::{create_bg, get_connection},
    };

    #[test]
    fn test_capnp_serialization() {
        let sample = Sample {
            name: "test_sample".to_string(),
        };

        let mut message = TypedBuilder::<sample::Owned>::new_default();
        let mut root = message.init_root();
        sample.write_capnp(&mut root);

        let deserialized = Sample::read_capnp(root.into_reader());
        assert_eq!(sample, deserialized);
    }

    #[test]
    fn test_delete_by_name() {
        let conn = &get_connection(None).unwrap();

        let _ = Sample::create(conn, "sample1").unwrap();
        let _ = Sample::create(conn, "sample2").unwrap();

        assert!(Sample::get_by_name(conn, "sample1").is_ok());
        assert!(Sample::get_by_name(conn, "sample2").is_ok());

        Sample::delete_by_name(conn, "sample1");

        assert!(Sample::get_by_name(conn, "sample1").is_err());
        assert!(Sample::get_by_name(conn, "sample2").is_ok());
    }

    #[test]
    fn test_search_name_returns_partial_matches() {
        let conn = &get_connection(None).unwrap();

        for sample in ["alpha", "BarFooBaz", "foo", "QuxFood", "zzz"] {
            Sample::create(conn, sample).unwrap();
        }

        let matches = Sample::search_name(conn, "FoO")
            .into_iter()
            .map(|sample| sample.name)
            .collect::<Vec<_>>();

        assert_eq!(matches, vec!["BarFooBaz", "QuxFood", "foo"]);
    }

    #[test]
    fn test_get_or_create_child_does_not_add_lineage_for_existing_sample() {
        let conn = &get_connection(None).unwrap();
        Sample::get_or_create(conn, "parent").unwrap();
        Sample::get_or_create(conn, "child").unwrap();

        Sample::get_or_create_child(conn, "test", "child", vec!["parent".to_string()]).unwrap();

        assert!(SampleLineage::get_parents(conn, "child").is_empty());
    }

    #[test]
    fn test_get_or_create_child_returns_sample_error_for_invalid_lineage() {
        let conn = &get_connection(None).unwrap();

        let err = Sample::get_or_create_child(conn, "test", "child", vec!["child".to_string()])
            .unwrap_err();

        assert!(matches!(
            err,
            SampleError::SqliteError(rusqlite::Error::SqliteFailure(code, _))
                if code.code == rusqlite::ErrorCode::ConstraintViolation
        ));
    }

    #[test]
    fn test_get_or_create_child_multiple_parents() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();

        create_bg(conn, "test", "parent_a", "chr1");
        create_bg(conn, "test", "parent_a", "chr2");
        create_bg(conn, "test", "parent_b", "chr2");
        create_bg(conn, "test", "parent_c", "chr3");

        let child = Sample::get_or_create_child(
            conn,
            "test",
            "child",
            vec![
                "parent_a".to_string(),
                "parent_b".to_string(),
                "parent_c".to_string(),
            ],
        )
        .unwrap();

        let mut block_group_names = Sample::get_block_groups(conn, "test", &child.name)
            .into_iter()
            .map(|block_group| block_group.name)
            .collect::<Vec<_>>();
        block_group_names.sort();
        assert_eq!(block_group_names, vec!["chr1", "chr2", "chr2", "chr3"]);
        assert_eq!(
            SampleLineage::get_parents(conn, &child.name),
            vec![
                "parent_a".to_string(),
                "parent_b".to_string(),
                "parent_c".to_string(),
            ]
        );
    }
}
