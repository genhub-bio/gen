use std::str;

use gen_core::{HashId, traits::Capnp};
use rusqlite::session;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    accession::{Accession, AccessionEdge},
    block_group::BlockGroup,
    changesets::{DatabaseChangeset, process_changesetiter, write_changeset},
    collection::Collection,
    db::{GraphDb, OperationsDb},
    edge::Edge,
    errors::OperationError,
    files::GenDatabase,
    gen_models_capnp::dependency_models,
    metadata::{self, get_db_uuid},
    node::Node,
    operations::{FileAddition, Operation, OperationInfo, OperationState, OperationSummary},
    path::Path,
    sample::Sample,
    sequence::Sequence,
};

pub fn start_operation(conn: &impl GraphDb) -> session::Session<'_> {
    let mut session = session::Session::new(conn.graph_conn()).unwrap();
    attach_session(&mut session);
    session
}

#[allow(clippy::too_many_arguments)]
pub fn end_operation(
    conn: &impl GraphDb,
    operation_conn: &impl OperationsDb,
    session: &mut session::Session,
    operation_info: &OperationInfo,
    summary_str: &str,
    force_hash: impl Into<Option<HashId>>,
) -> Result<Operation, OperationError> {
    let conn = conn.graph_conn();
    let operation_conn = operation_conn.operations_conn();
    let db_uuid = metadata::get_db_uuid(conn);
    // determine if this operation has already happened
    let mut output = Vec::new();
    session.changeset_strm(&mut output).unwrap();

    let (changeset_models, dependencies) = process_changesetiter(conn, &output);

    let hash = if let Some(hash) = force_hash.into() {
        hash
    } else {
        if output.is_empty() {
            return Err(OperationError::NoChanges);
        }
        let mut hasher = Sha256::new();
        hasher.update(&db_uuid[..]);
        hasher.update(&output[..]);
        HashId(hasher.finalize().into())
    };

    operation_conn
        .execute("SAVEPOINT new_operation;", [])
        .unwrap();

    match Operation::create(operation_conn, &operation_info.description, &hash) {
        Ok(operation) => {
            for op_file in operation_info.files.iter() {
                let fa = match FileAddition::get_or_create(
                    operation_conn,
                    &op_file.file_path,
                    op_file.file_type,
                ) {
                    Ok(fa) => fa,
                    Err(err) => return Err(OperationError::SQLError(format!("{err}"))),
                };
                Operation::add_file(operation_conn, &operation.hash, &fa.id)
                    .map_err(|err| OperationError::SQLError(format!("{err}")))?
            }
            Operation::add_database(operation_conn, &operation.hash, &db_uuid)
                .map_err(|err| OperationError::SQLError(format!("{err}")))?;
            OperationSummary::create(operation_conn, &operation.hash, summary_str);
            let db_uuid = get_db_uuid(conn);
            let gen_db = GenDatabase::get_by_uuid(operation_conn, &db_uuid).unwrap();
            write_changeset(
                &operation,
                DatabaseChangeset {
                    db_path: gen_db.path,
                    changes: changeset_models,
                },
                &dependencies,
            );
            OperationState::set_operation(operation_conn, &operation.hash);
            operation_conn
                .execute("RELEASE SAVEPOINT new_operation;", [])
                .unwrap();
            Ok(operation)
        }
        Err(rusqlite::Error::SqliteFailure(err, details)) => {
            operation_conn
                .execute("ROLLBACK TRANSACTION TO SAVEPOINT new_operation;", [])
                .unwrap();
            if err.code == rusqlite::ErrorCode::ConstraintViolation {
                Err(OperationError::OperationExists)
            } else {
                panic!("something bad happened querying the database {details:?}");
            }
        }
        Err(e) => {
            operation_conn
                .execute("ROLLBACK TRANSACTION TO SAVEPOINT new_operation;", [])
                .unwrap();
            panic!("something bad happened querying the database {e:?}");
        }
    }
}

pub fn attach_session(session: &mut session::Session) {
    for table in [
        "collections",
        "samples",
        "sequences",
        "block_groups",
        "paths",
        "nodes",
        "edges",
        "path_edges",
        "block_group_edges",
        "accessions",
        "accession_edges",
        "accession_paths",
    ] {
        session.attach(Some(table)).unwrap();
    }
}

#[derive(Default, Deserialize, Serialize, Debug, PartialEq)]
pub struct DependencyModels {
    pub collections: Vec<Collection>,
    pub samples: Vec<Sample>,
    pub sequences: Vec<Sequence>,
    pub block_group: Vec<BlockGroup>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub paths: Vec<Path>,
    pub accessions: Vec<Accession>,
    pub accession_edges: Vec<AccessionEdge>,
}

impl<'a> Capnp<'a> for DependencyModels {
    type Builder = dependency_models::Builder<'a>;
    type Reader = dependency_models::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        // Write collections
        let mut collections_builder = builder
            .reborrow()
            .init_collections(self.collections.len() as u32);
        for (i, collection) in self.collections.iter().enumerate() {
            let mut collection_builder = collections_builder.reborrow().get(i as u32);
            collection.write_capnp(&mut collection_builder);
        }

        // Write samples
        let mut samples_builder = builder.reborrow().init_samples(self.samples.len() as u32);
        for (i, sample) in self.samples.iter().enumerate() {
            let mut sample_builder = samples_builder.reborrow().get(i as u32);
            sample.write_capnp(&mut sample_builder);
        }

        // Write sequences
        let mut sequences_builder = builder
            .reborrow()
            .init_sequences(self.sequences.len() as u32);
        for (i, sequence) in self.sequences.iter().enumerate() {
            let mut sequence_builder = sequences_builder.reborrow().get(i as u32);
            sequence.write_capnp(&mut sequence_builder);
        }

        // Write block groups (note: field name is blockGroup in capnp schema)
        let mut block_group_builder = builder
            .reborrow()
            .init_block_group(self.block_group.len() as u32);
        for (i, block_group) in self.block_group.iter().enumerate() {
            let mut bg_builder = block_group_builder.reborrow().get(i as u32);
            block_group.write_capnp(&mut bg_builder);
        }

        // Write nodes
        let mut nodes_builder = builder.reborrow().init_nodes(self.nodes.len() as u32);
        for (i, node) in self.nodes.iter().enumerate() {
            let mut node_builder = nodes_builder.reborrow().get(i as u32);
            node.write_capnp(&mut node_builder);
        }

        // Write edges
        let mut edges_builder = builder.reborrow().init_edges(self.edges.len() as u32);
        for (i, edge) in self.edges.iter().enumerate() {
            let mut edge_builder = edges_builder.reborrow().get(i as u32);
            edge.write_capnp(&mut edge_builder);
        }

        // Write paths
        let mut paths_builder = builder.reborrow().init_paths(self.paths.len() as u32);
        for (i, path) in self.paths.iter().enumerate() {
            let mut path_builder = paths_builder.reborrow().get(i as u32);
            path.write_capnp(&mut path_builder);
        }

        // Write accessions
        let mut accessions_builder = builder
            .reborrow()
            .init_accessions(self.accessions.len() as u32);
        for (i, accession) in self.accessions.iter().enumerate() {
            let mut accession_builder = accessions_builder.reborrow().get(i as u32);
            accession.write_capnp(&mut accession_builder);
        }

        // Write accession edges (note: field name is accessionEdges in capnp schema)
        let mut accession_edges_builder = builder
            .reborrow()
            .init_accession_edges(self.accession_edges.len() as u32);
        for (i, accession_edge) in self.accession_edges.iter().enumerate() {
            let mut accession_edge_builder = accession_edges_builder.reborrow().get(i as u32);
            accession_edge.write_capnp(&mut accession_edge_builder);
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        // Read collections
        let collections_reader = reader.get_collections().unwrap();
        let mut collections = Vec::new();
        for collection_reader in collections_reader.iter() {
            collections.push(Collection::read_capnp(collection_reader));
        }

        // Read samples
        let samples_reader = reader.get_samples().unwrap();
        let mut samples = Vec::new();
        for sample_reader in samples_reader.iter() {
            samples.push(Sample::read_capnp(sample_reader));
        }

        // Read sequences
        let sequences_reader = reader.get_sequences().unwrap();
        let mut sequences = Vec::new();
        for sequence_reader in sequences_reader.iter() {
            sequences.push(Sequence::read_capnp(sequence_reader));
        }

        // Read block groups
        let block_group_reader = reader.get_block_group().unwrap();
        let mut block_group = Vec::new();
        for bg_reader in block_group_reader.iter() {
            block_group.push(BlockGroup::read_capnp(bg_reader));
        }

        // Read nodes
        let nodes_reader = reader.get_nodes().unwrap();
        let mut nodes = Vec::new();
        for node_reader in nodes_reader.iter() {
            nodes.push(Node::read_capnp(node_reader));
        }

        // Read edges
        let edges_reader = reader.get_edges().unwrap();
        let mut edges = Vec::new();
        for edge_reader in edges_reader.iter() {
            edges.push(Edge::read_capnp(edge_reader));
        }

        // Read paths
        let paths_reader = reader.get_paths().unwrap();
        let mut paths = Vec::new();
        for path_reader in paths_reader.iter() {
            paths.push(Path::read_capnp(path_reader));
        }

        // Read accessions
        let accessions_reader = reader.get_accessions().unwrap();
        let mut accessions = Vec::new();
        for accession_reader in accessions_reader.iter() {
            accessions.push(Accession::read_capnp(accession_reader));
        }

        // Read accession edges
        let accession_edges_reader = reader.get_accession_edges().unwrap();
        let mut accession_edges = Vec::new();
        for accession_edge_reader in accession_edges_reader.iter() {
            accession_edges.push(AccessionEdge::read_capnp(accession_edge_reader));
        }

        DependencyModels {
            collections,
            samples,
            sequences,
            block_group,
            nodes,
            edges,
            paths,
            accessions,
            accession_edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;
    use gen_core::Strand;

    use super::*;
    use crate::sequence::NewSequence;

    #[test]
    fn test_dependency_models_capnp_serialization() {
        let dependency_models = DependencyModels {
            collections: vec![Collection {
                name: "test_collection".to_string(),
            }],
            samples: vec![Sample {
                name: "test_sample".to_string(),
            }],
            sequences: vec![
                NewSequence::new()
                    .sequence_type("DNA")
                    .sequence("ATCG")
                    .name("test_seq")
                    .build(),
            ],
            block_group: vec![BlockGroup {
                id: HashId::pad_str(1),
                collection_name: "test_collection".to_string(),
                sample_name: Some("test_sample".to_string()),
                name: "test_bg".to_string(),
                created_on: 0,
            }],
            nodes: vec![Node {
                id: HashId::convert_str("node_hash"),
                sequence_hash: HashId::convert_str("test_hash"),
            }],
            edges: vec![Edge {
                id: HashId::pad_str(1),
                source_node_id: HashId::convert_str("1"),
                source_coordinate: 0,
                source_strand: Strand::Forward,
                target_node_id: HashId::convert_str("2"),
                target_coordinate: 0,
                target_strand: Strand::Forward,
            }],
            paths: vec![Path {
                id: HashId::pad_str(1),
                block_group_id: HashId::pad_str(1),
                name: "test_path".to_string(),
                created_on: 0,
            }],
            accessions: vec![Accession {
                id: HashId::pad_str(1),
                name: "test_accession".to_string(),
                path_id: HashId::pad_str(1),
                parent_accession_id: None,
            }],
            accession_edges: vec![AccessionEdge {
                id: HashId::pad_str(1),
                source_node_id: HashId::convert_str("1"),
                source_coordinate: 0,
                source_strand: Strand::Forward,
                target_node_id: HashId::convert_str("2"),
                target_coordinate: 0,
                target_strand: Strand::Forward,
                chromosome_index: 0,
            }],
        };

        let mut message = TypedBuilder::<dependency_models::Owned>::new_default();
        let mut root = message.init_root();
        dependency_models.write_capnp(&mut root);

        let deserialized = DependencyModels::read_capnp(root.into_reader());

        assert_eq!(dependency_models, deserialized);
    }
}
