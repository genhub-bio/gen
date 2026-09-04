use std::collections::HashSet;

use gen_core::{Workspace, traits::Capnp};
use gen_graph::GenGraph;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Direction, ModelSelect, ModelSelectError,
    block_group::{BlockGroup, BlockGroupSelect},
    db::GraphConnection,
    errors::{BlockGroupError, QueryError},
    gen_models_capnp::sample,
    sample_lineage::SampleLineage,
};

#[derive(Debug, Deserialize, Serialize, PartialEq, ModelSelect)]
#[model_select(table = "samples")]
pub struct Sample {
    #[model_select(primary_key)]
    pub name: String,
    pub is_reference: bool,
}

#[derive(Clone, Debug, Default)]
pub struct NewSample<'a> {
    pub name: &'a str,
    pub is_reference: bool,
}

#[derive(Debug, Error, PartialEq)]
pub enum SampleError {
    #[error("Query Error: {0}")]
    QueryError(#[from] QueryError),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
    #[error("Selector error: {0}")]
    ModelSelect(#[from] ModelSelectError),
    #[error("Sample already exists")]
    Duplicate(Sample),
    #[error("Sample not found: {0}")]
    NotFound(String),
    #[error("Sample is not a reference sample: {0}")]
    NotReference(String),
    #[error("Block group creation error: {0}")]
    BlockGroup(#[from] BlockGroupError),
}

impl<'a> Capnp<'a> for Sample {
    type Builder = sample::Builder<'a>;
    type Reader = sample::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_name(&self.name);
        builder.set_is_reference(self.is_reference);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let name = reader.get_name().unwrap().to_string().unwrap();
        let is_reference = reader.get_is_reference();
        Sample { name, is_reference }
    }
}

impl Sample {
    pub const DEFAULT_NAME: &str = "reference";

    pub fn get_parent_names(
        conn: &GraphConnection,
        sample_name: &str,
        history_ref: Option<&str>,
    ) -> Vec<String> {
        SampleLineage::get_parents(conn, sample_name, history_ref)
    }

    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, new_sample))
    )]
    pub fn create(
        conn: &GraphConnection,
        new_sample: NewSample<'_>,
    ) -> Result<Sample, SampleError> {
        let mut stmt =
            match conn.prepare("INSERT INTO samples (name, is_reference) VALUES (?1, ?2);") {
                Ok(stmt) => stmt,
                Err(err) => return Err(SampleError::SqliteError(err)),
            };

        match stmt.execute(params![new_sample.name, new_sample.is_reference]) {
            Ok(_) => Ok(Sample {
                name: new_sample.name.to_string(),
                is_reference: new_sample.is_reference,
            }),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(SampleError::Duplicate(Sample {
                    name: new_sample.name.to_string(),
                    is_reference: new_sample.is_reference,
                }))
            }
            Err(err) => Err(SampleError::SqliteError(err)),
        }
    }

    #[cfg_attr(
        all(debug_assertions, feature = "profiling"),
        tracing::instrument(skip(conn, new_sample))
    )]
    pub fn get_or_create(
        conn: &GraphConnection,
        new_sample: NewSample<'_>,
    ) -> Result<Sample, SampleError> {
        match Sample::create(conn, new_sample.clone()) {
            Ok(sample) => Ok(sample),
            Err(SampleError::Duplicate(sample)) => Ok(sample),
            Err(e) => Err(e),
        }
    }

    pub fn set_reference(
        conn: &GraphConnection,
        name: &str,
        is_reference: bool,
    ) -> Result<(), SampleError> {
        conn.execute(
            "UPDATE samples SET is_reference = ?2 WHERE name = ?1",
            params![name, is_reference],
        )?;
        Ok(())
    }

    pub fn get_reference_samples(conn: &GraphConnection, history_ref: Option<&str>) -> Vec<Sample> {
        let mut select = Sample::select(conn)
            .is_reference(true)
            .order_by(SampleSelect::Name, Direction::Asc);
        if let Some(history_ref) = history_ref {
            select = select.with_ref(history_ref);
        }
        select.load().expect("should load reference samples")
    }

    pub fn get_sample_reference_block_groups(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<Vec<BlockGroup>, SampleError> {
        let sample = Sample::get_by_name(conn, sample_name)
            .map_err(|_| SampleError::NotFound(sample_name.to_string()))?;

        if !sample.is_reference {
            return Err(SampleError::NotReference(sample_name.to_string()));
        }

        Ok(Sample::get_block_groups(
            conn,
            collection_name,
            sample_name,
            None,
        ))
    }

    pub fn delete_by_name(conn: &GraphConnection, name: &str) {
        let mut stmt = conn.prepare("delete from samples where name = ?1").unwrap();
        stmt.execute([name]).unwrap();
    }

    pub fn get_graph(
        conn: &GraphConnection,
        workspace: &Workspace,
        collection: &str,
        name: &str,
        history_ref: Option<&str>,
    ) -> Result<GenGraph, SampleError> {
        let block_groups = Sample::get_block_groups(conn, collection, name, history_ref);
        let mut sample_graph = GenGraph::new();
        for bg in block_groups {
            let bg_graph = BlockGroup::get_graph(conn, workspace, &bg.id, history_ref)?;
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
        Ok(sample_graph)
    }

    pub fn get_all_sequences(
        conn: &GraphConnection,
        workspace: &Workspace,
        collection_name: &str,
        sample_name: &str,
        prune: bool,
        history_ref: Option<&str>,
    ) -> Result<HashSet<String>, SampleError> {
        let mut sequences = HashSet::new();
        for block_group in Sample::get_block_groups(conn, collection_name, sample_name, history_ref)
        {
            sequences.extend(BlockGroup::get_all_sequences(
                conn,
                workspace,
                &block_group.id,
                prune,
            )?);
        }
        Ok(sequences)
    }

    pub fn get_or_create_child(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        parent_samples: Vec<String>,
    ) -> Result<Sample, SampleError> {
        match Sample::create(
            conn,
            NewSample {
                name: sample_name,
                is_reference: false,
            },
        ) {
            Ok(new_sample) => {
                if !parent_samples.is_empty() {
                    let parent_block_groups = BlockGroup::select(conn)
                        .collection_name(collection_name)
                        .sample_name_in(parent_samples.iter())
                        .order_by(BlockGroupSelect::Name, Direction::Asc)
                        .order_by(BlockGroupSelect::SampleName, Direction::Asc)
                        .order_by(BlockGroupSelect::CreatedOn, Direction::Asc)
                        .order_by(BlockGroupSelect::Id, Direction::Asc)
                        .load()?;
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
        history_ref: Option<&str>,
    ) -> Vec<BlockGroup> {
        let mut select = BlockGroup::select(conn)
            .collection_name(collection_name)
            .sample_name(sample_name);
        if let Some(history_ref) = history_ref {
            select = select.with_ref(history_ref);
        }
        select.load().expect("should load block groups")
    }

    pub fn get_all_names(conn: &GraphConnection, history_ref: Option<&str>) -> Vec<String> {
        let mut select = Sample::select(conn);
        if let Some(history_ref) = history_ref {
            select = select.with_ref(history_ref);
        }
        let samples = select.load().expect("should load sample names");
        samples.iter().map(|s| s.name.clone()).collect()
    }

    pub fn get_by_name(conn: &GraphConnection, name: &str) -> Result<Sample, SampleError> {
        Sample::select(conn)
            .name(name)
            .load()?
            .into_iter()
            .next()
            .ok_or_else(|| SampleError::NotFound(name.to_string()))
    }

    pub fn search_name(
        conn: &GraphConnection,
        name: &str,
        history_ref: Option<&str>,
    ) -> Vec<Sample> {
        let mut select = Sample::select(conn)
            .name_contains(name)
            .order_by(SampleSelect::Name, Direction::Asc);
        if let Some(history_ref) = history_ref {
            select = select.with_ref(history_ref);
        }
        select.load().expect("should load matching samples")
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::{
        collection::Collection,
        errors::SampleError,
        history::dolt::commit_staged_all,
        test_helpers::{create_bg, get_connection},
    };

    #[test]
    fn test_capnp_serialization() {
        let sample = Sample {
            name: "test_sample".to_string(),
            is_reference: true,
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

        let _ = Sample::create(
            conn,
            NewSample {
                name: "sample1",
                is_reference: false,
            },
        )
        .unwrap();
        let _ = Sample::create(
            conn,
            NewSample {
                name: "sample2",
                is_reference: false,
            },
        )
        .unwrap();

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
            Sample::create(
                conn,
                NewSample {
                    name: sample,
                    is_reference: false,
                },
            )
            .unwrap();
        }

        let matches = Sample::search_name(conn, "FoO", None)
            .into_iter()
            .map(|sample| sample.name)
            .collect::<Vec<_>>();

        assert_eq!(matches, vec!["BarFooBaz", "QuxFood", "foo"]);
    }

    #[test]
    fn test_search_name_returns_matches_at_history_ref() {
        let conn = &get_connection(None).unwrap();
        Sample::create(
            conn,
            NewSample {
                name: "historical-foo",
                is_reference: false,
            },
        )
        .expect("should create historical sample");
        let historical_commit =
            commit_staged_all(conn, "add historical sample").expect("should commit sample");
        let historical_ref = historical_commit.to_string();
        Sample::create(
            conn,
            NewSample {
                name: "current-foo",
                is_reference: false,
            },
        )
        .expect("should create current sample");

        let matches = Sample::search_name(conn, "foo", Some(&historical_ref))
            .into_iter()
            .map(|sample| sample.name)
            .collect::<Vec<_>>();

        assert_eq!(matches, vec!["historical-foo"]);
    }

    #[test]
    fn test_get_or_create_child_does_not_add_lineage_for_existing_sample() {
        let conn = &get_connection(None).unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "parent",
                ..Default::default()
            },
        )
        .unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "child",
                ..Default::default()
            },
        )
        .unwrap();

        Sample::get_or_create_child(conn, "test", "child", vec!["parent".to_string()]).unwrap();

        assert!(SampleLineage::get_parents(conn, "child", None).is_empty());
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

        let mut block_group_names = Sample::get_block_groups(conn, "test", &child.name, None)
            .into_iter()
            .map(|block_group| block_group.name)
            .collect::<Vec<_>>();
        block_group_names.sort();
        assert_eq!(block_group_names, vec!["chr1", "chr2", "chr2", "chr3"]);
        assert_eq!(
            SampleLineage::get_parents(conn, &child.name, None),
            vec![
                "parent_a".to_string(),
                "parent_b".to_string(),
                "parent_c".to_string(),
            ]
        );
    }

    #[test]
    fn test_get_reference_samples_returns_only_reference_samples() {
        let conn = &get_connection(None).unwrap();
        Sample::create(
            conn,
            NewSample {
                name: "reference-a",
                is_reference: true,
            },
        )
        .unwrap();
        Sample::create(
            conn,
            NewSample {
                name: "sample-b",
                is_reference: false,
            },
        )
        .unwrap();
        Sample::create(
            conn,
            NewSample {
                name: "reference-c",
                is_reference: true,
            },
        )
        .unwrap();

        let reference_names = Sample::get_reference_samples(conn, None)
            .into_iter()
            .map(|sample| sample.name)
            .collect::<Vec<_>>();

        assert_eq!(reference_names, vec!["reference-a", "reference-c"]);
    }

    #[test]
    fn test_get_sample_reference_block_groups_returns_reference_block_groups() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();

        Sample::create(
            conn,
            NewSample {
                name: "reference-root",
                is_reference: true,
            },
        )
        .unwrap();
        create_bg(conn, "test", "reference-root", "chr1");
        create_bg(conn, "test", "reference-root", "chr2");

        let reference_block_group_names =
            Sample::get_sample_reference_block_groups(conn, "test", "reference-root")
                .unwrap()
                .into_iter()
                .map(|block_group| block_group.name)
                .collect::<Vec<_>>();

        assert_eq!(reference_block_group_names, vec!["chr1", "chr2"]);
    }

    #[test]
    fn test_get_sample_reference_block_groups_errors_for_non_reference_sample() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::create(
            conn,
            NewSample {
                name: "child",
                is_reference: false,
            },
        )
        .unwrap();

        let err = Sample::get_sample_reference_block_groups(conn, "test", "child").unwrap_err();

        assert_eq!(err, SampleError::NotReference("child".to_string()));
    }

    #[test]
    fn test_get_sample_reference_block_groups_errors_for_missing_sample() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();

        let err = Sample::get_sample_reference_block_groups(conn, "test", "missing").unwrap_err();

        assert_eq!(err, SampleError::NotFound("missing".to_string()));
    }
}
