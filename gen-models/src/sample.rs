use std::fmt::*;

use gen_core::traits::Capnp;
use gen_graph::GenGraph;
use rusqlite::{params, Connection, Result as SQLResult, Row};
use serde::{Deserialize, Serialize};

use crate::{block_group::BlockGroup, gen_models_capnp::sample, traits::*};

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
    pub fn create(conn: &Connection, name: &str) -> SQLResult<Sample> {
        let mut stmt = conn
            .prepare("INSERT INTO samples (name) VALUES (?1) returning (name);")
            .unwrap();
        stmt.query_row((name,), |row| Ok(Sample { name: row.get(0)? }))
    }

    pub fn get_or_create(conn: &Connection, name: &str) -> Sample {
        match Sample::create(conn, name) {
            Ok(sample) => sample,
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    Sample {
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

    pub fn delete_by_name(conn: &Connection, name: &str) {
        let mut stmt = conn.prepare("delete from samples where name = ?1").unwrap();
        stmt.execute([name]).unwrap();
    }

    pub fn get_graph<'a>(
        conn: &Connection,
        collection: &str,
        name: impl Into<Option<&'a str>>,
    ) -> GenGraph {
        let name = name.into();
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

    pub fn get_or_create_child(
        conn: &Connection,
        collection_name: &str,
        sample_name: &str,
        parent_sample: Option<&str>,
    ) -> Sample {
        if let Ok(new_sample) = Sample::create(conn, sample_name) {
            let bgs = if let Some(parent) = parent_sample {
                BlockGroup::query(
                    conn,
                    "select * from block_groups where collection_name = ?1 AND sample_name = ?2",
                    params!(collection_name, parent),
                )
            } else {
                BlockGroup::query(conn, "select * from block_groups where collection_name = ?1 AND sample_name is null;", params!(collection_name))
            };
            for bg in bgs.iter() {
                BlockGroup::get_or_create_sample_block_group(
                    conn,
                    collection_name,
                    &new_sample.name,
                    &bg.name,
                    parent_sample,
                )
                .expect("failed to get or create blockgroup clone.");
            }
            new_sample
        } else {
            Sample {
                name: sample_name.to_string(),
            }
        }
    }

    pub fn get_block_groups(
        conn: &Connection,
        collection_name: &str,
        sample_name: Option<&str>,
    ) -> Vec<BlockGroup> {
        if let Some(sample) = sample_name {
            BlockGroup::query(
                conn,
                "select * from block_groups where collection_name = ?1 AND sample_name = ?2;",
                params![collection_name, sample],
            )
        } else {
            BlockGroup::query(
                conn,
                "select * from block_groups where collection_name = ?1 AND sample_name IS NULL;",
                params![collection_name],
            )
        }
    }

    pub fn get_all_names(conn: &Connection) -> Vec<String> {
        let samples = Sample::query(conn, "select * from samples;", rusqlite::params!());
        samples.iter().map(|s| s.name.clone()).collect()
    }

    pub fn get_by_name(conn: &Connection, name: &str) -> SQLResult<Sample> {
        Sample::get(
            conn,
            "select * from samples where name = ?1;",
            rusqlite::params!(name),
        )
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::test_helpers::get_connection;

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

        let sample1 = Sample::create(conn, "sample1").unwrap();
        let sample2 = Sample::create(conn, "sample2").unwrap();

        assert!(Sample::get_by_name(&conn, "sample1").is_ok());
        assert!(Sample::get_by_name(&conn, "sample2").is_ok());

        Sample::delete_by_name(&conn, "sample1");

        assert!(Sample::get_by_name(&conn, "sample1").is_err());
        assert!(Sample::get_by_name(&conn, "sample2").is_ok());
    }
}
