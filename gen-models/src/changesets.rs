use std::{
    collections::{HashMap, HashSet},
    convert::TryInto,
    fs,
    io::Read,
    path::{Path as StdPath, PathBuf},
    str,
};

use gen_core::{HashId, Strand, config::Workspace, is_terminal, traits::Capnp};
use itertools::Itertools;
use rusqlite::{
    session::{ChangesetItem, ChangesetIter},
    types::FromSql,
};
use serde::{Deserialize, Serialize};

use crate::{
    accession::{Accession, AccessionEdge, AccessionEdgeData, AccessionPath},
    annotations::{Annotation, AnnotationGroup, AnnotationGroupSample},
    block_group::{BlockGroup, NewBlockGroup},
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    collection::Collection,
    db::GraphConnection,
    edge::{Edge, EdgeData},
    errors::{AnnotationError, AnnotationGroupError, ChangesetError, CollectionError},
    gen_models_capnp::{changeset_models, database_changeset},
    node::Node,
    operations::Operation,
    path::Path,
    path_edge::PathEdge,
    sample::Sample,
    sample_lineage::SampleLineage,
    sequence::{NewSequence, Sequence},
    session_operations::DependencyModels,
    traits::Query,
};

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct DatabaseChangeset {
    pub db_path: String,
    pub changes: ChangesetModels,
}

impl DatabaseChangeset {
    pub fn get_db_path(path: &StdPath) -> String {
        use capnp::serialize_packed;

        let file = fs::File::open(path).unwrap();
        let mut reader = std::io::BufReader::new(file);

        let message_reader =
            serialize_packed::read_message(&mut reader, capnp::message::ReaderOptions::new())
                .unwrap();
        let root = message_reader
            .get_root::<database_changeset::Reader>()
            .unwrap();
        root.get_db_path().unwrap().to_string().unwrap()
    }
}

impl<'a> Capnp<'a> for DatabaseChangeset {
    type Builder = database_changeset::Builder<'a>;
    type Reader = database_changeset::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_db_path(&self.db_path);
        let mut changeset_builder = builder.reborrow().init_changes();
        self.changes.write_capnp(&mut changeset_builder);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let db_path = reader.get_db_path().unwrap().to_string().unwrap();
        let changeset_reader = reader.get_changes().unwrap();
        let changes = ChangesetModels::read_capnp(changeset_reader);
        DatabaseChangeset { db_path, changes }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct ChangesetModels {
    pub collections: Vec<crate::collection::Collection>,
    pub samples: Vec<crate::sample::Sample>,
    pub sample_lineages: Vec<SampleLineage>,
    pub sequences: Vec<Sequence>,
    pub block_groups: Vec<BlockGroup>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub block_group_edges: Vec<crate::block_group_edge::BlockGroupEdge>,
    pub paths: Vec<Path>,
    pub path_edges: Vec<PathEdge>,
    pub accessions: Vec<Accession>,
    pub accession_edges: Vec<AccessionEdge>,
    pub accession_paths: Vec<AccessionPath>,
    pub annotation_groups: Vec<AnnotationGroup>,
    pub annotations: Vec<Annotation>,
    pub annotation_group_samples: Vec<AnnotationGroupSample>,
}

impl<'a> Capnp<'a> for ChangesetModels {
    type Builder = changeset_models::Builder<'a>;
    type Reader = changeset_models::Reader<'a>;

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

        let mut sample_lineages_builder = builder
            .reborrow()
            .init_sample_lineages(self.sample_lineages.len() as u32);
        for (i, sample_lineage) in self.sample_lineages.iter().enumerate() {
            let mut sample_lineage_builder = sample_lineages_builder.reborrow().get(i as u32);
            sample_lineage.write_capnp(&mut sample_lineage_builder);
        }

        // Write sequences
        let mut sequences_builder = builder
            .reborrow()
            .init_sequences(self.sequences.len() as u32);
        for (i, sequence) in self.sequences.iter().enumerate() {
            let mut sequence_builder = sequences_builder.reborrow().get(i as u32);
            sequence.write_capnp(&mut sequence_builder);
        }

        // Write block groups
        let mut block_groups_builder = builder
            .reborrow()
            .init_block_groups(self.block_groups.len() as u32);
        for (i, block_group) in self.block_groups.iter().enumerate() {
            let mut block_group_builder = block_groups_builder.reborrow().get(i as u32);
            block_group.write_capnp(&mut block_group_builder);
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

        // Write block group edges
        let mut block_group_edges_builder = builder
            .reborrow()
            .init_block_group_edges(self.block_group_edges.len() as u32);
        for (i, block_group_edge) in self.block_group_edges.iter().enumerate() {
            let mut block_group_edge_builder = block_group_edges_builder.reborrow().get(i as u32);
            block_group_edge.write_capnp(&mut block_group_edge_builder);
        }

        // Write paths
        let mut paths_builder = builder.reborrow().init_paths(self.paths.len() as u32);
        for (i, path) in self.paths.iter().enumerate() {
            let mut path_builder = paths_builder.reborrow().get(i as u32);
            path.write_capnp(&mut path_builder);
        }

        // Write path edges
        let mut path_edges_builder = builder
            .reborrow()
            .init_path_edges(self.path_edges.len() as u32);
        for (i, path_edge) in self.path_edges.iter().enumerate() {
            let mut path_edge_builder = path_edges_builder.reborrow().get(i as u32);
            path_edge.write_capnp(&mut path_edge_builder);
        }

        // Write accessions
        let mut accessions_builder = builder
            .reborrow()
            .init_accessions(self.accessions.len() as u32);
        for (i, accession) in self.accessions.iter().enumerate() {
            let mut accession_builder = accessions_builder.reborrow().get(i as u32);
            accession.write_capnp(&mut accession_builder);
        }

        // Write accession edges
        let mut accession_edges_builder = builder
            .reborrow()
            .init_accession_edges(self.accession_edges.len() as u32);
        for (i, accession_edge) in self.accession_edges.iter().enumerate() {
            let mut accession_edge_builder = accession_edges_builder.reborrow().get(i as u32);
            accession_edge.write_capnp(&mut accession_edge_builder);
        }

        // Write accession paths
        let mut accession_paths_builder = builder
            .reborrow()
            .init_accession_paths(self.accession_paths.len() as u32);
        for (i, accession_path) in self.accession_paths.iter().enumerate() {
            let mut accession_path_builder = accession_paths_builder.reborrow().get(i as u32);
            accession_path.write_capnp(&mut accession_path_builder);
        }

        // Write annotation groups
        let mut annotation_groups_builder = builder
            .reborrow()
            .init_annotation_groups(self.annotation_groups.len() as u32);
        for (i, annotation_group) in self.annotation_groups.iter().enumerate() {
            let mut annotation_group_builder = annotation_groups_builder.reborrow().get(i as u32);
            annotation_group.write_capnp(&mut annotation_group_builder);
        }

        // Write annotations
        let mut annotations_builder = builder
            .reborrow()
            .init_annotations(self.annotations.len() as u32);
        for (i, annotation) in self.annotations.iter().enumerate() {
            let mut annotation_builder = annotations_builder.reborrow().get(i as u32);
            annotation.write_capnp(&mut annotation_builder);
        }

        // Write annotation group samples
        let mut annotation_group_samples_builder = builder
            .reborrow()
            .init_annotation_group_samples(self.annotation_group_samples.len() as u32);
        for (i, annotation_group_sample) in self.annotation_group_samples.iter().enumerate() {
            let mut annotation_group_sample_builder =
                annotation_group_samples_builder.reborrow().get(i as u32);
            annotation_group_sample.write_capnp(&mut annotation_group_sample_builder);
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        // Read collections
        let collections_reader = reader.get_collections().unwrap();
        let mut collections = Vec::new();
        for collection_reader in collections_reader.iter() {
            collections.push(crate::collection::Collection::read_capnp(collection_reader));
        }

        // Read samples
        let samples_reader = reader.get_samples().unwrap();
        let mut samples = Vec::new();
        for sample_reader in samples_reader.iter() {
            samples.push(crate::sample::Sample::read_capnp(sample_reader));
        }

        let sample_lineages_reader = reader.get_sample_lineages().unwrap();
        let mut sample_lineages = Vec::new();
        for sample_lineage_reader in sample_lineages_reader.iter() {
            sample_lineages.push(SampleLineage::read_capnp(sample_lineage_reader));
        }

        // Read sequences
        let sequences_reader = reader.get_sequences().unwrap();
        let mut sequences = Vec::new();
        for sequence_reader in sequences_reader.iter() {
            sequences.push(Sequence::read_capnp(sequence_reader));
        }

        // Read block groups
        let block_groups_reader = reader.get_block_groups().unwrap();
        let mut block_groups = Vec::new();
        for block_group_reader in block_groups_reader.iter() {
            block_groups.push(BlockGroup::read_capnp(block_group_reader));
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

        // Read block group edges
        let block_group_edges_reader = reader.get_block_group_edges().unwrap();
        let mut block_group_edges = Vec::new();
        for block_group_edge_reader in block_group_edges_reader.iter() {
            block_group_edges.push(crate::block_group_edge::BlockGroupEdge::read_capnp(
                block_group_edge_reader,
            ));
        }

        // Read paths
        let paths_reader = reader.get_paths().unwrap();
        let mut paths = Vec::new();
        for path_reader in paths_reader.iter() {
            paths.push(Path::read_capnp(path_reader));
        }

        // Read path edges
        let path_edges_reader = reader.get_path_edges().unwrap();
        let mut path_edges = Vec::new();
        for path_edge_reader in path_edges_reader.iter() {
            path_edges.push(PathEdge::read_capnp(path_edge_reader));
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

        // Read accession paths
        let accession_paths_reader = reader.get_accession_paths().unwrap();
        let mut accession_paths = Vec::new();
        for accession_path_reader in accession_paths_reader.iter() {
            accession_paths.push(AccessionPath::read_capnp(accession_path_reader));
        }

        // Read annotation groups
        let annotation_groups_reader = reader.get_annotation_groups().unwrap();
        let mut annotation_groups = Vec::new();
        for annotation_group_reader in annotation_groups_reader.iter() {
            annotation_groups.push(AnnotationGroup::read_capnp(annotation_group_reader));
        }

        // Read annotations
        let annotations_reader = reader.get_annotations().unwrap();
        let mut annotations = Vec::new();
        for annotation_reader in annotations_reader.iter() {
            annotations.push(Annotation::read_capnp(annotation_reader));
        }

        // Read annotation group samples
        let annotation_group_samples_reader = reader.get_annotation_group_samples().unwrap();
        let mut annotation_group_samples = Vec::new();
        for annotation_group_sample_reader in annotation_group_samples_reader.iter() {
            annotation_group_samples.push(AnnotationGroupSample::read_capnp(
                annotation_group_sample_reader,
            ));
        }

        ChangesetModels {
            collections,
            samples,
            sample_lineages,
            sequences,
            block_groups,
            nodes,
            edges,
            block_group_edges,
            paths,
            path_edges,
            accessions,
            accession_edges,
            accession_paths,
            annotation_groups,
            annotations,
            annotation_group_samples,
        }
    }
}

// Helper functions for parsing changeset items
pub fn parse_string(item: &ChangesetItem, col: usize) -> String {
    str::from_utf8(item.new_value(col).unwrap().as_bytes().unwrap())
        .unwrap()
        .to_string()
}

pub fn parse_maybe_string(item: &ChangesetItem, col: usize) -> Option<String> {
    item.new_value(col)
        .unwrap()
        .as_bytes_or_null()
        .unwrap()
        .map(|v| str::from_utf8(v).unwrap().to_string())
}

pub fn parse_blob(item: &ChangesetItem, col: usize) -> [u8; 32] {
    let bytes = item.new_value(col).unwrap().as_bytes().unwrap();

    bytes.try_into().expect("blob must be exactly 32 bytes")
}

pub fn parse_maybe_blob(item: &ChangesetItem, col: usize) -> Option<[u8; 32]> {
    item.new_value(col)
        .unwrap()
        .as_bytes_or_null()
        .unwrap()
        .map(|v| v.try_into().expect("blob must be exactly 32 bytes"))
}

pub fn parse_hashid(item: &ChangesetItem, col: usize) -> HashId {
    HashId(parse_blob(item, col))
}

pub fn parse_maybe_hashid(item: &ChangesetItem, col: usize) -> Option<HashId> {
    parse_maybe_blob(item, col).map(HashId)
}

pub fn parse_number(item: &ChangesetItem, col: usize) -> i64 {
    item.new_value(col).unwrap().as_i64().unwrap()
}

pub fn parse_maybe_number(item: &ChangesetItem, col: usize) -> Option<i64> {
    item.new_value(col).unwrap().as_i64_or_null().unwrap()
}

pub fn process_changesetiter(
    conn: &GraphConnection,
    mut changes: &[u8],
) -> (ChangesetModels, DependencyModels) {
    use fallible_streaming_iterator::FallibleStreamingIterator;
    use gen_core::is_terminal;
    use itertools::Itertools;

    use crate::{
        accession::{Accession, AccessionEdge},
        block_group::BlockGroup,
        edge::Edge,
        node::Node,
        path::Path,
        sequence::Sequence,
        traits::Query,
    };

    // Initialize collections for changeset models
    let mut created_block_groups = vec![];
    let mut created_edges = vec![];
    let mut created_nodes = vec![];
    let mut created_sequences = vec![];
    let mut created_bg_edges = vec![];
    let mut created_samples = vec![];
    let mut created_sample_lineages = vec![];
    let mut created_collections = vec![];
    let mut created_paths = vec![];
    let mut created_path_edges = vec![];
    let mut created_accessions = vec![];
    let mut created_accession_edges = vec![];
    let mut created_accession_paths = vec![];
    let mut created_annotation_groups = vec![];
    let mut created_annotations = vec![];
    let mut created_annotation_group_samples = vec![];

    // Initialize collections for dependency tracking
    let mut previous_collections = HashSet::new();
    let mut previous_samples = HashSet::new();
    let mut previous_block_groups = HashSet::new();
    let mut previous_edges = HashSet::new();
    let mut previous_paths = HashSet::new();
    let mut previous_accessions = HashSet::new();
    let mut previous_nodes = HashSet::new();
    let mut previous_sequences = HashSet::new();
    let mut previous_accession_edges = HashSet::new();
    let mut created_block_groups_set = HashSet::new();
    let mut created_paths_set = HashSet::new();
    let mut created_accessions_set = HashSet::new();
    let mut created_edges_set = HashSet::new();
    let mut created_accession_edges_set = HashSet::new();
    let mut created_nodes_set = HashSet::new();
    let mut created_sequences_set = HashSet::new();
    let mut created_samples_set: HashSet<String> = HashSet::new();
    let mut created_collections_set: HashSet<String> = HashSet::new();

    let input: &mut dyn Read = &mut changes;
    let mut iter = ChangesetIter::start_strm(&input).unwrap();

    while let Some(item) = iter.next().unwrap() {
        let op = item.op().unwrap();
        // info on indirect changes: https://www.sqlite.org/draft/session/sqlite3session_indirect.html
        if !op.indirect() {
            let table = op.table_name();
            let pk_column = item
                .pk()
                .unwrap()
                .iter()
                .find_position(|item| **item == 1)
                .unwrap()
                .0;
            match table {
                "sequences" => {
                    let hash = parse_hashid(item, pk_column);
                    let sequence = Sequence::new()
                        .sequence_type(&parse_string(item, 1))
                        .sequence(&parse_string(item, 2))
                        .name(&parse_string(item, 3))
                        .file_path(&parse_string(item, 4))
                        .length(parse_number(item, 5))
                        .build();
                    assert_eq!(hash, sequence.hash);
                    created_sequences.push(sequence);
                    created_sequences_set.insert(hash);
                }
                "block_groups" => {
                    let bg_pk = HashId(parse_blob(item, pk_column));
                    let collection = parse_string(item, 1);
                    let sample_name = parse_string(item, 2);
                    let name = parse_string(item, 3);
                    let created_on = parse_number(item, 4);
                    let parent_block_group_id = parse_maybe_hashid(item, 5);

                    created_block_groups.push(BlockGroup {
                        id: bg_pk,
                        collection_name: collection.clone(),
                        sample_name: sample_name.clone(),
                        name,
                        created_on,
                        parent_block_group_id,
                        is_default: parse_number(item, 6) != 0,
                    });

                    created_block_groups_set.insert(bg_pk);
                    if let Some(parent_block_group_id) = parent_block_group_id
                        && !created_block_groups_set.contains(&parent_block_group_id)
                    {
                        previous_block_groups.insert(parent_block_group_id);
                    }
                    if !created_collections_set.contains(&collection) {
                        previous_collections.insert(collection);
                    }
                    if !created_samples_set.contains(&sample_name) {
                        previous_samples.insert(sample_name);
                    }
                }
                "nodes" => {
                    let node_id = parse_hashid(item, pk_column);
                    let sequence_hash = parse_hashid(item, 1);

                    created_nodes.push(Node {
                        id: node_id,
                        sequence_hash,
                    });

                    created_nodes_set.insert(node_id);
                    if !created_sequences_set.contains(&sequence_hash) {
                        previous_sequences.insert(sequence_hash);
                    }
                }
                "edges" => {
                    let edge_id = HashId(parse_blob(item, pk_column));
                    let source_node_id = parse_hashid(item, 1);
                    let target_node_id = parse_hashid(item, 4);

                    created_edges.push(Edge {
                        id: edge_id,
                        source_node_id,
                        source_coordinate: parse_number(item, 2),
                        source_strand: Strand::column_result(item.new_value(3).unwrap()).unwrap(),
                        target_node_id,
                        target_coordinate: parse_number(item, 5),
                        target_strand: Strand::column_result(item.new_value(6).unwrap()).unwrap(),
                    });

                    created_edges_set.insert(edge_id);
                    let nodes = Node::query_by_ids(conn, &[source_node_id, target_node_id]);
                    for node in nodes.iter() {
                        if !created_nodes_set.contains(&node.id) && !is_terminal(node.id) {
                            previous_sequences.insert(node.sequence_hash);
                            previous_nodes.insert(node.id);
                        }
                    }
                }
                "block_group_edges" => {
                    let bg_id = HashId(parse_blob(item, 1));
                    let edge_id = HashId(parse_blob(item, 2));

                    created_bg_edges.push(BlockGroupEdge {
                        id: HashId(parse_blob(item, pk_column)),
                        block_group_id: bg_id,
                        edge_id,
                        chromosome_index: parse_number(item, 3),
                        phased: parse_number(item, 4),
                        created_on: parse_number(item, 5),
                    });

                    if !created_edges_set.contains(&edge_id) {
                        previous_edges.insert(edge_id);
                    }
                    if !created_block_groups_set.contains(&bg_id) {
                        previous_block_groups.insert(bg_id);
                    }
                }
                "samples" => {
                    let name = parse_string(item, pk_column);
                    created_samples.push(Sample {
                        name: name.clone(),
                        is_reference: parse_number(item, 1) != 0,
                    });
                    created_samples_set.insert(name);
                }
                "sample_lineage" => {
                    let parent_sample_name = parse_string(item, 0);
                    let child_sample_name = parse_string(item, 1);

                    created_sample_lineages.push(SampleLineage {
                        parent_sample_name: parent_sample_name.clone(),
                        child_sample_name: child_sample_name.clone(),
                    });

                    if !created_samples_set.contains(&parent_sample_name) {
                        previous_samples.insert(parent_sample_name);
                    }
                    if !created_samples_set.contains(&child_sample_name) {
                        previous_samples.insert(child_sample_name);
                    }
                }
                "collections" => {
                    let name = parse_string(item, pk_column);
                    created_collections.push(Collection { name: name.clone() });
                    created_collections_set.insert(name);
                }
                "paths" => {
                    let path_id = HashId(parse_blob(item, pk_column));
                    let bg_id = HashId(parse_blob(item, 1));

                    created_paths.push(Path {
                        id: path_id,
                        block_group_id: bg_id,
                        name: parse_string(item, 2),
                        created_on: parse_number(item, 3),
                    });

                    created_paths_set.insert(path_id);
                    if !created_block_groups_set.contains(&bg_id) {
                        previous_block_groups.insert(bg_id);
                    }
                }
                "path_edges" => {
                    let path_id = HashId(parse_blob(item, 1));
                    let edge_id = HashId(parse_blob(item, 2));

                    created_path_edges.push(PathEdge {
                        id: HashId(parse_blob(item, pk_column)),
                        path_id,
                        edge_id,
                        index_in_path: parse_number(item, 3),
                    });

                    if !created_paths_set.contains(&path_id) {
                        previous_paths.insert(path_id);
                    }
                    if !created_edges_set.contains(&edge_id) {
                        previous_edges.insert(edge_id);
                    }
                }
                "accessions" => {
                    let accession_id = HashId(parse_blob(item, pk_column));
                    let path_id = HashId(parse_blob(item, 2));
                    let parent_accession_id = parse_maybe_hashid(item, 3);

                    created_accessions.push(Accession {
                        id: accession_id,
                        name: parse_string(item, 1),
                        path_id,
                        parent_accession_id,
                    });

                    created_accessions_set.insert(accession_id);
                    if !created_paths_set.contains(&path_id) {
                        previous_paths.insert(path_id);
                    }
                    if let Some(id) = parent_accession_id
                        && !created_accessions_set.contains(&id)
                    {
                        previous_accessions.insert(id);
                    }
                }
                "accession_edges" => {
                    let edge_id = parse_hashid(item, pk_column);
                    let source_node_id = parse_hashid(item, 1);
                    let target_node_id = parse_hashid(item, 4);

                    created_accession_edges.push(AccessionEdge {
                        id: edge_id,
                        source_node_id,
                        source_coordinate: parse_number(item, 2),
                        source_strand: Strand::column_result(item.new_value(3).unwrap()).unwrap(),
                        target_node_id,
                        target_coordinate: parse_number(item, 5),
                        target_strand: Strand::column_result(item.new_value(6).unwrap()).unwrap(),
                        chromosome_index: parse_number(item, 7),
                    });

                    created_accession_edges_set.insert(edge_id);
                    let nodes = Node::query_by_ids(conn, &[source_node_id, target_node_id]);
                    if !created_nodes_set.contains(&source_node_id) {
                        previous_sequences.insert(nodes[0].sequence_hash);
                    }
                    if source_node_id != target_node_id
                        && !created_nodes_set.contains(&target_node_id)
                    {
                        previous_sequences.insert(nodes[1].sequence_hash);
                    }
                }
                "accession_paths" => {
                    let accession_id = parse_hashid(item, 1);
                    let edge_id = parse_hashid(item, 3);

                    created_accession_paths.push(AccessionPath {
                        id: parse_hashid(item, pk_column),
                        accession_id,
                        index_in_path: parse_number(item, 2),
                        edge_id,
                    });

                    if !created_accessions_set.contains(&accession_id) {
                        previous_accessions.insert(accession_id);
                    }
                    if !created_accession_edges_set.contains(&edge_id) {
                        previous_accession_edges.insert(edge_id);
                    }
                }
                "annotations" => {
                    let id = parse_hashid(item, pk_column);
                    let name = parse_string(item, 1);
                    let group = parse_string(item, 2);
                    let accession_id = parse_hashid(item, 3);
                    let extra = parse_string(item, 4);

                    created_annotations.push(Annotation {
                        id,
                        name,
                        group,
                        accession_id,
                        extra: if extra.is_empty() {
                            None
                        } else {
                            Some(serde_json::from_str(&extra).unwrap())
                        },
                    });

                    if !created_accessions_set.contains(&accession_id) {
                        previous_accessions.insert(accession_id);
                    }
                }
                "annotation_groups" => {
                    let name = parse_string(item, pk_column);

                    created_annotation_groups.push(AnnotationGroup { name });
                }
                "annotation_group_samples" => {
                    let annotation_group = parse_string(item, 0);
                    let sample_name = parse_string(item, 1);

                    created_annotation_group_samples.push(AnnotationGroupSample {
                        annotation_group,
                        sample_name: sample_name.clone(),
                    });

                    if !created_samples_set.contains(&sample_name) {
                        previous_samples.insert(sample_name);
                    }
                }
                t => {
                    println!("unhandled table {t}")
                }
            }
        }
    }

    // Process additional dependencies for edges
    let existing_edges = Edge::query_by_ids(
        conn,
        &previous_edges.clone().into_iter().collect::<Vec<_>>(),
    );
    let mut new_nodes = vec![];
    for edge in existing_edges.iter() {
        if !previous_nodes.contains(&edge.source_node_id) && !is_terminal(edge.source_node_id) {
            previous_nodes.insert(edge.source_node_id);
            new_nodes.push(edge.source_node_id);
        }
        if !previous_nodes.contains(&edge.target_node_id) && !is_terminal(edge.target_node_id) {
            previous_nodes.insert(edge.target_node_id);
            new_nodes.push(edge.target_node_id);
        }
    }
    for node in Node::query_by_ids(conn, &new_nodes) {
        previous_sequences.insert(node.sequence_hash);
    }

    let changeset_models = ChangesetModels {
        sequences: created_sequences,
        block_groups: created_block_groups,
        nodes: created_nodes,
        edges: created_edges,
        block_group_edges: created_bg_edges,
        samples: created_samples,
        sample_lineages: created_sample_lineages,
        collections: created_collections,
        paths: created_paths,
        path_edges: created_path_edges,
        accessions: created_accessions,
        accession_edges: created_accession_edges,
        accession_paths: created_accession_paths,
        annotation_groups: created_annotation_groups,
        annotations: created_annotations,
        annotation_group_samples: created_annotation_group_samples,
    };

    let dependency_models = DependencyModels {
        collections: Collection::query_by_ids(conn, &previous_collections),
        samples: Sample::query_by_ids(conn, &previous_samples),
        sequences: Sequence::query_by_ids(conn, &previous_sequences),
        block_group: BlockGroup::query_by_ids(conn, &previous_block_groups),
        nodes: Node::query_by_ids(conn, &previous_nodes),
        edges: Edge::query_by_ids(conn, &previous_edges),
        paths: Path::query_by_ids(conn, &previous_paths),
        accessions: Accession::query_by_ids(conn, &previous_accessions),
        accession_edges: AccessionEdge::query_by_ids(conn, &previous_accession_edges),
    };

    (changeset_models, dependency_models)
}

pub fn apply_changeset(
    conn: &GraphConnection,
    changeset: &ChangesetModels,
    dependencies: &DependencyModels,
) -> Result<(), ChangesetError> {
    for collection in dependencies.collections.iter() {
        match Collection::create(conn, &collection.name) {
            Ok(_) => {}
            Err(CollectionError::Duplicate(_)) => {}
            Err(err) => return Err(ChangesetError::CollectionError(err)),
        }
    }

    for sample in dependencies.samples.iter() {
        Sample::get_or_create(conn, &sample.name)?;
    }

    for sequence in dependencies.sequences.iter() {
        NewSequence::from(sequence).save(conn)?;
    }
    for node in dependencies.nodes.iter() {
        if !is_terminal(node.id) {
            assert!(Sequence::get_by_id(conn, &node.sequence_hash).is_some());
        }
    }

    for bg in block_groups_parent_first(&dependencies.block_group) {
        BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &bg.collection_name,
                sample_name: &bg.sample_name,
                name: &bg.name,
                parent_block_group_id: bg.parent_block_group_id.as_ref(),
                is_default: bg.is_default,
            },
        )?;
    }

    for node in dependencies.nodes.iter() {
        Node::create(conn, &node.sequence_hash, &node.id)?;
    }

    Edge::bulk_create(
        conn,
        &dependencies
            .edges
            .iter()
            .map(EdgeData::from)
            .collect::<Vec<_>>(),
    );

    for path in dependencies.paths.iter() {
        Path::create(conn, &path.name, &path.block_group_id, &[])?;
    }

    AccessionEdge::bulk_create(
        conn,
        &dependencies
            .accession_edges
            .iter()
            .map(AccessionEdgeData::from)
            .collect::<Vec<AccessionEdgeData>>(),
    );

    for accession in dependencies.accessions.iter() {
        Accession::get_or_create(
            conn,
            &accession.name,
            &accession.path_id,
            accession.parent_accession_id.as_ref(),
        )?;
    }

    for collection in &changeset.collections {
        Collection::create(conn, &collection.name)?;
    }
    for sample in &changeset.samples {
        Sample::get_or_create(conn, &sample.name)?;
    }
    for sample_lineage in &changeset.sample_lineages {
        SampleLineage::create(
            conn,
            &sample_lineage.parent_sample_name,
            &sample_lineage.child_sample_name,
        )?;
    }
    for sequence in &changeset.sequences {
        NewSequence::from(sequence).save(conn)?;
    }
    for bg in block_groups_parent_first(&changeset.block_groups) {
        BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &bg.collection_name,
                sample_name: &bg.sample_name,
                name: &bg.name,
                parent_block_group_id: bg.parent_block_group_id.as_ref(),
                is_default: bg.is_default,
            },
        )?;
    }
    for node in &changeset.nodes {
        Node::create(conn, &node.sequence_hash, &node.id)?;
    }

    Edge::bulk_create(
        conn,
        &changeset
            .edges
            .iter()
            .map(EdgeData::from)
            .collect::<Vec<_>>(),
    );
    BlockGroupEdge::bulk_create(
        conn,
        &changeset
            .block_group_edges
            .iter()
            .map(BlockGroupEdgeData::from)
            .collect::<Vec<BlockGroupEdgeData>>(),
    );

    for path in &changeset.paths {
        Path::create(conn, &path.name, &path.block_group_id, &[])?;
        let edges = changeset
            .path_edges
            .iter()
            .filter(|edge| edge.path_id == path.id)
            .sorted_by(|e1, e2| Ord::cmp(&e1.index_in_path, &e2.index_in_path))
            .map(|path_edge| path_edge.edge_id);
        let _ = PathEdge::bulk_create(conn, &path.id, &edges.collect::<Vec<_>>());
    }

    AccessionEdge::bulk_create(
        conn,
        &changeset
            .accession_edges
            .iter()
            .map(AccessionEdgeData::from)
            .collect::<Vec<_>>(),
    );

    for accession in &changeset.accessions {
        Accession::get_or_create(
            conn,
            &accession.name,
            &accession.path_id,
            accession.parent_accession_id.as_ref(),
        )?;
        let edges = changeset
            .accession_paths
            .iter()
            .filter(|ap| ap.accession_id == accession.id)
            .sorted_by(|e1, e2| Ord::cmp(&e1.index_in_path, &e2.index_in_path))
            .map(|ap| ap.edge_id);
        AccessionPath::create(conn, &accession.id, &edges.collect::<Vec<_>>())?;
    }

    for annotation_group in &changeset.annotation_groups {
        AnnotationGroup::get_or_create(conn, &annotation_group.name).map_err(|err| match err {
            AnnotationGroupError::DatabaseError(inner) => inner,
        })?;
    }

    for annotation in &changeset.annotations {
        Annotation::get_or_create(
            conn,
            &annotation.name,
            &annotation.group,
            &annotation.accession_id,
            annotation.extra.as_ref(),
        )
        .map_err(|err| match err {
            AnnotationError::DatabaseError(inner) => ChangesetError::SqliteError(inner),
            AnnotationError::AnnotationGroupError(AnnotationGroupError::DatabaseError(inner)) => {
                ChangesetError::SqliteError(inner)
            }
            AnnotationError::SerializationError(message) => {
                ChangesetError::SerializationError(message)
            }
        })?;
    }

    for annotation_group_sample in &changeset.annotation_group_samples {
        AnnotationGroupSample::create(
            conn,
            &annotation_group_sample.annotation_group,
            &annotation_group_sample.sample_name,
        )
        .map_err(|err| match err {
            AnnotationError::DatabaseError(inner) => ChangesetError::SqliteError(inner),
            AnnotationError::AnnotationGroupError(AnnotationGroupError::DatabaseError(inner)) => {
                ChangesetError::SqliteError(inner)
            }
            AnnotationError::SerializationError(message) => {
                ChangesetError::SerializationError(message)
            }
        })?;
    }
    Ok(())
}

pub fn revert_changeset(
    conn: &GraphConnection,
    changeset: &ChangesetModels,
) -> Result<(), ChangesetError> {
    for sample_lineage in &changeset.sample_lineages {
        SampleLineage::delete(
            conn,
            &sample_lineage.parent_sample_name,
            &sample_lineage.child_sample_name,
        )?;
    }
    for annotation_group_sample in &changeset.annotation_group_samples {
        AnnotationGroupSample::delete(
            conn,
            &annotation_group_sample.annotation_group,
            &annotation_group_sample.sample_name,
        )
        .map_err(|err| match err {
            AnnotationError::DatabaseError(inner) => ChangesetError::SqliteError(inner),
            AnnotationError::AnnotationGroupError(AnnotationGroupError::DatabaseError(inner)) => {
                ChangesetError::SqliteError(inner)
            }
            AnnotationError::SerializationError(message) => {
                ChangesetError::SerializationError(message)
            }
        })?;
    }
    Annotation::delete_by_ids(
        conn,
        &changeset
            .annotations
            .iter()
            .map(|obj| obj.id)
            .collect::<Vec<_>>(),
    );
    AnnotationGroup::delete_by_ids(
        conn,
        &changeset
            .annotation_groups
            .iter()
            .map(|group| group.name.clone())
            .collect::<Vec<_>>(),
    );
    AccessionPath::delete_by_ids(
        conn,
        &changeset
            .accession_paths
            .iter()
            .map(|obj| obj.id)
            .collect::<Vec<_>>(),
    );
    // TODO: edges can be made by other operations in other dbs earlier and reused, so we likely want to allow FK failures delete for deletions
    AccessionEdge::delete_by_ids(
        conn,
        &changeset
            .accession_edges
            .iter()
            .map(|pe| pe.id)
            .collect::<Vec<_>>(),
    );
    Accession::delete_by_ids(
        conn,
        &changeset
            .accessions
            .iter()
            .map(|obj| obj.id)
            .collect::<Vec<_>>(),
    );
    PathEdge::delete_by_ids(
        conn,
        &changeset
            .path_edges
            .iter()
            .map(|pe| pe.id)
            .collect::<Vec<_>>(),
    );
    Path::delete_by_ids(
        conn,
        &changeset.paths.iter().map(|p| p.id).collect::<Vec<_>>(),
    );

    BlockGroupEdge::delete_by_ids(
        conn,
        &changeset
            .block_group_edges
            .iter()
            .map(|obj| obj.id)
            .collect::<Vec<_>>(),
    );
    Edge::delete_by_ids(
        conn,
        &changeset.edges.iter().map(|obj| obj.id).collect::<Vec<_>>(),
    );

    BlockGroup::delete_by_ids(
        conn,
        &changeset
            .block_groups
            .iter()
            .map(|obj| obj.id)
            .collect::<Vec<_>>(),
    );

    Node::delete_by_ids(
        conn,
        &changeset.nodes.iter().map(|obj| obj.id).collect::<Vec<_>>(),
    );
    Collection::delete_by_ids(
        conn,
        &changeset
            .collections
            .iter()
            .map(|obj| obj.name.clone())
            .collect::<Vec<_>>(),
    );
    Sample::delete_by_ids(
        conn,
        &changeset
            .samples
            .iter()
            .map(|obj| obj.name.clone())
            .collect::<Vec<_>>(),
    );
    Sequence::delete_by_ids(
        conn,
        &changeset
            .sequences
            .iter()
            .map(|obj| obj.hash)
            .collect::<Vec<_>>(),
    );
    Ok(())
}

pub fn get_changeset_from_path(path: PathBuf) -> DatabaseChangeset {
    use capnp::serialize_packed;
    let file = fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);

    let message_reader =
        serialize_packed::read_message(&mut reader, capnp::message::ReaderOptions::new()).unwrap();
    let root = message_reader
        .get_root::<database_changeset::Reader>()
        .unwrap();
    DatabaseChangeset::read_capnp(root)
}

pub fn get_changeset_dependencies_from_path(path: PathBuf) -> DependencyModels {
    use capnp::serialize_packed;

    let file = fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let message_reader =
        serialize_packed::read_message(&mut reader, capnp::message::ReaderOptions::new()).unwrap();
    let root = message_reader
        .get_root::<crate::gen_models_capnp::dependency_models::Reader>()
        .unwrap();
    DependencyModels::read_capnp(root)
}

// This sorts parents first so when creating blockgroups, we don't have foreign key issues when a child is made before a parent.
fn block_groups_parent_first(block_groups: &[BlockGroup]) -> Vec<&BlockGroup> {
    let by_id = block_groups
        .iter()
        .map(|block_group| (block_group.id, block_group))
        .collect::<HashMap<_, _>>();

    fn walk<'a>(
        block_group_id: HashId,
        by_id: &HashMap<HashId, &'a BlockGroup>,
        visiting: &mut HashSet<HashId>,
        visited: &mut HashSet<HashId>,
        ordered: &mut Vec<&'a BlockGroup>,
    ) {
        if visited.contains(&block_group_id) || !visiting.insert(block_group_id) {
            return;
        }

        let block_group = by_id[&block_group_id];
        if let Some(parent_id) = block_group.parent_block_group_id
            && by_id.contains_key(&parent_id)
        {
            walk(parent_id, by_id, visiting, visited, ordered);
        }

        visiting.remove(&block_group_id);
        if visited.insert(block_group_id) {
            ordered.push(block_group);
        }
    }

    let mut ordered = Vec::new();
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    for block_group in block_groups {
        walk(
            block_group.id,
            &by_id,
            &mut visiting,
            &mut visited,
            &mut ordered,
        );
    }

    ordered
}

pub fn write_changeset(
    workspace: &Workspace,
    operation: &Operation,
    changes: DatabaseChangeset,
    dependencies: &DependencyModels,
) {
    use capnp::{message::Builder, serialize_packed};

    let change_path = operation.get_changeset_path(workspace);
    let dependency_path = operation.get_changeset_dependencies_path(workspace);

    // Write dependencies using capnp
    let mut dependency_file = fs::File::create_new(&dependency_path)
        .unwrap_or_else(|_| panic!("Unable to open {dependency_path:?}"));
    let mut message = Builder::new_default();
    let mut dep_root = message.init_root();
    dependencies.write_capnp(&mut dep_root);
    serialize_packed::write_message(&mut dependency_file, &message).unwrap();

    // Write changes using capnp
    let mut file = fs::File::create_new(&change_path)
        .unwrap_or_else(|_| panic!("Unable to open {change_path:?}"));
    let mut message = Builder::new_default();
    let mut change_root = message.init_root();
    changes.write_capnp(&mut change_root);
    serialize_packed::write_message(&mut file, &message).unwrap();
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use super::*;
    use crate::{
        file_types::FileTypes,
        operations::{OperationFile, OperationInfo},
        session_operations::{end_operation, start_operation},
        test_helpers::{setup_block_group, setup_gen},
        traits::Query,
    };

    #[test]
    fn test_database_changeset_capnp_serialization() {
        use capnp::message::TypedBuilder;
        use gen_core::Strand;

        let changeset_models = ChangesetModels {
            collections: vec![crate::collection::Collection {
                name: "test_collection".to_string(),
            }],
            samples: vec![crate::sample::Sample {
                name: "test_sample".to_string(),
                is_reference: false,
            }],
            sample_lineages: vec![SampleLineage {
                parent_sample_name: "parent_sample".to_string(),
                child_sample_name: "test_sample".to_string(),
            }],
            sequences: vec![
                NewSequence::new()
                    .sequence("ATCG")
                    .sequence_type("DNA")
                    .name("test_seq")
                    .build(),
            ],
            block_groups: vec![BlockGroup {
                id: HashId::pad_str(1),
                collection_name: "test_collection".to_string(),
                sample_name: "test_sample".to_string(),
                name: "test_bg".to_string(),
                created_on: Utc::now().timestamp_nanos_opt().unwrap(),
                parent_block_group_id: None,
                is_default: false,
            }],
            nodes: vec![
                Node {
                    id: HashId::convert_str("node_hash"),
                    sequence_hash: HashId::convert_str("test_hash"),
                },
                Node {
                    id: HashId::convert_str("node_hash_2"),
                    sequence_hash: HashId::convert_str("test_hash_2"),
                },
            ],
            edges: vec![Edge {
                id: HashId::pad_str(1),
                source_node_id: HashId::convert_str("1"),
                source_coordinate: 0,
                source_strand: Strand::Forward,
                target_node_id: HashId::convert_str("2"),
                target_coordinate: 0,
                target_strand: Strand::Forward,
            }],
            block_group_edges: vec![crate::block_group_edge::BlockGroupEdge {
                id: HashId::pad_str(1),
                block_group_id: HashId::pad_str(1),
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            }],
            paths: vec![Path {
                id: HashId::pad_str(1),
                block_group_id: HashId::pad_str(1),
                name: "test_path".to_string(),
                created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            }],
            path_edges: vec![PathEdge {
                id: HashId::pad_str(1),
                path_id: HashId::pad_str(1),
                index_in_path: 0,
                edge_id: HashId::pad_str(1),
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
            accession_paths: vec![AccessionPath {
                id: HashId::pad_str(1),
                accession_id: HashId::pad_str(1),
                index_in_path: 0,
                edge_id: HashId::pad_str(1),
            }],
            annotation_groups: vec![AnnotationGroup {
                name: "gff3".to_string(),
            }],
            annotations: vec![Annotation {
                id: HashId::pad_str(1),
                name: "test_annotation".to_string(),
                group: "gff3".to_string(),
                accession_id: HashId::pad_str(1),
                extra: None,
            }],
            annotation_group_samples: vec![AnnotationGroupSample {
                annotation_group: "gff3".to_string(),
                sample_name: "test_sample".to_string(),
            }],
        };

        let changeset = DatabaseChangeset {
            db_path: "/path/to/db".to_string(),
            changes: changeset_models,
        };

        let mut message = TypedBuilder::<database_changeset::Owned>::new_default();
        let mut root = message.init_root();
        changeset.write_capnp(&mut root);

        let deserialized = DatabaseChangeset::read_capnp(root.into_reader());
        assert_eq!(changeset, deserialized);
    }

    #[test]
    fn test_database_changeset_get_db_path() {
        use std::io::Write;

        use capnp::{message::Builder, serialize_packed};
        use tempfile::NamedTempFile;

        let expected_db_path = "/tmp/test_db_path";
        let changeset = DatabaseChangeset {
            db_path: expected_db_path.to_string(),
            changes: ChangesetModels {
                collections: vec![],
                samples: vec![],
                sample_lineages: vec![],
                sequences: vec![],
                block_groups: vec![],
                nodes: vec![],
                edges: vec![],
                block_group_edges: vec![],
                paths: vec![],
                path_edges: vec![],
                accessions: vec![],
                accession_edges: vec![],
                accession_paths: vec![],
                annotation_groups: vec![],
                annotations: vec![],
                annotation_group_samples: vec![],
            },
        };

        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        let mut message = Builder::new_default();
        let mut root = message.init_root();
        changeset.write_capnp(&mut root);
        serialize_packed::write_message(&mut temp_file, &message).unwrap();
        temp_file.flush().unwrap();

        let db_path = DatabaseChangeset::get_db_path(path.as_path());

        assert_eq!(db_path, expected_db_path);
    }

    #[test]
    fn test_changeset_models_capnp_serialization() {
        use capnp::message::TypedBuilder;
        use gen_core::Strand;

        let changeset_models = ChangesetModels {
            collections: vec![crate::collection::Collection {
                name: "test_collection".to_string(),
            }],
            samples: vec![crate::sample::Sample {
                name: "test_sample".to_string(),
                is_reference: false,
            }],
            sample_lineages: vec![SampleLineage {
                parent_sample_name: "parent_sample".to_string(),
                child_sample_name: "test_sample".to_string(),
            }],
            sequences: vec![
                NewSequence::new()
                    .sequence("ATCG")
                    .sequence_type("DNA")
                    .name("test_seq")
                    .build(),
            ],
            block_groups: vec![BlockGroup {
                id: HashId::pad_str(1),
                collection_name: "test_collection".to_string(),
                sample_name: "test_sample".to_string(),
                name: "test_bg".to_string(),
                created_on: Utc::now().timestamp_nanos_opt().unwrap(),
                parent_block_group_id: None,
                is_default: false,
            }],
            nodes: vec![
                Node {
                    id: HashId::convert_str("node_hash"),
                    sequence_hash: HashId::convert_str("test_hash"),
                },
                Node {
                    id: HashId::convert_str("node_hash_2"),
                    sequence_hash: HashId::convert_str("test_hash_2"),
                },
            ],
            edges: vec![Edge {
                id: HashId::pad_str(1),
                source_node_id: HashId::convert_str("1"),
                source_coordinate: 0,
                source_strand: Strand::Forward,
                target_node_id: HashId::convert_str("2"),
                target_coordinate: 0,
                target_strand: Strand::Forward,
            }],
            block_group_edges: vec![crate::block_group_edge::BlockGroupEdge {
                id: HashId::pad_str(1),
                block_group_id: HashId::pad_str(1),
                edge_id: HashId::pad_str(1),
                chromosome_index: 0,
                phased: 0,
                created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            }],
            paths: vec![Path {
                id: HashId::pad_str(1),
                block_group_id: HashId::pad_str(1),
                name: "test_path".to_string(),
                created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            }],
            path_edges: vec![PathEdge {
                id: HashId::pad_str(1),
                path_id: HashId::pad_str(1),
                index_in_path: 0,
                edge_id: HashId::pad_str(1),
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
            accession_paths: vec![AccessionPath {
                id: HashId::pad_str(1),
                accession_id: HashId::pad_str(1),
                index_in_path: 0,
                edge_id: HashId::pad_str(1),
            }],
            annotation_groups: vec![AnnotationGroup {
                name: "gff3".to_string(),
            }],
            annotations: vec![Annotation {
                id: HashId::pad_str(1),
                name: "test_annotation".to_string(),
                group: "gff3".to_string(),
                accession_id: HashId::pad_str(1),
                extra: None,
            }],
            annotation_group_samples: vec![AnnotationGroupSample {
                annotation_group: "gff3".to_string(),
                sample_name: "test_sample".to_string(),
            }],
        };

        let mut message = TypedBuilder::<changeset_models::Owned>::new_default();
        let mut root = message.init_root();
        changeset_models.write_capnp(&mut root);

        let deserialized = ChangesetModels::read_capnp(root.into_reader());

        assert_eq!(changeset_models, deserialized);
    }

    #[test]
    fn test_changeset_includes_annotations() {
        use crate::block_group::PathCache;

        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let _ = Sample::create(
            conn,
            crate::sample::NewSample {
                name: "sample-1",
                is_reference: false,
            },
        )
        .unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let mut cache = PathCache::new(conn);
        let _ = PathCache::lookup(&mut cache, &block_group_id, path.name.clone()).unwrap();
        let accession =
            BlockGroup::add_accession(conn, &path, "ann-accession", 0, 5, &mut cache).unwrap();

        let mut session = start_operation(conn);
        let annotation =
            Annotation::get_or_create(conn, "gene-a", "gff3", &accession.id, None).unwrap();
        annotation.add_samples(conn, &["sample-1"]).unwrap();

        let operation = end_operation(
            &context,
            &mut session,
            &OperationInfo {
                files: vec![],
                description: "annotation op".to_string(),
            },
            "annotation op",
            None,
        )
        .unwrap();

        let changeset = operation.get_changeset(context.workspace());
        assert_eq!(
            changeset.changes.annotation_groups,
            vec![AnnotationGroup {
                name: "gff3".to_string(),
            }]
        );
        assert_eq!(changeset.changes.annotations, vec![annotation.clone()]);
        assert_eq!(
            changeset.changes.annotation_group_samples,
            vec![AnnotationGroupSample {
                annotation_group: annotation.group.clone(),
                sample_name: "sample-1".to_string(),
            }]
        );

        let dependencies = operation.get_changeset_dependencies(context.workspace());
        assert_eq!(dependencies.samples.len(), 1);
        assert_eq!(dependencies.samples[0].name, "sample-1");
        assert_eq!(dependencies.accessions.len(), 1);
        assert_eq!(dependencies.accessions[0].id, accession.id);
    }

    #[test]
    fn test_changeset_includes_sample_lineage() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        let db_uuid = crate::metadata::get_db_uuid(conn);
        crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path").unwrap();

        let _ = Sample::create(
            conn,
            crate::sample::NewSample {
                name: "parent",
                is_reference: false,
            },
        )
        .unwrap();

        let mut session = start_operation(conn);
        let _ = Sample::create(
            conn,
            crate::sample::NewSample {
                name: "child",
                is_reference: false,
            },
        )
        .unwrap();
        SampleLineage::create(conn, "parent", "child").unwrap();

        let operation = end_operation(
            &context,
            &mut session,
            &OperationInfo {
                files: vec![],
                description: "sample lineage op".to_string(),
            },
            "sample lineage op",
            None,
        )
        .unwrap();

        let changeset = operation.get_changeset(context.workspace());
        assert_eq!(
            changeset.changes.sample_lineages,
            vec![SampleLineage {
                parent_sample_name: "parent".to_string(),
                child_sample_name: "child".to_string(),
            }]
        );

        let dependencies = operation.get_changeset_dependencies(context.workspace());
        assert_eq!(dependencies.samples.len(), 1);
        assert_eq!(dependencies.samples[0].name, "parent");
    }

    #[test]
    fn test_block_groups_parent_first() {
        let parent = BlockGroup {
            id: HashId::pad_str(1),
            collection_name: "test".to_string(),
            sample_name: "parent".to_string(),
            name: "bg".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: None,
            is_default: false,
        };
        let child = BlockGroup {
            id: HashId::pad_str(2),
            collection_name: "test".to_string(),
            sample_name: "child".to_string(),
            name: "bg".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: Some(parent.id),
            is_default: false,
        };

        let block_groups = [child.clone(), parent.clone()];
        let ordered = block_groups_parent_first(&block_groups);
        assert_eq!(ordered, vec![&parent, &child]);
    }

    #[test]
    fn test_block_groups_parent_first_with_three_level_lineage() {
        let parent = BlockGroup {
            id: HashId::pad_str(1),
            collection_name: "test".to_string(),
            sample_name: "parent".to_string(),
            name: "bg".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: None,
            is_default: false,
        };
        let child = BlockGroup {
            id: HashId::pad_str(2),
            collection_name: "test".to_string(),
            sample_name: "child".to_string(),
            name: "bg".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: Some(parent.id),
            is_default: false,
        };
        let grandchild = BlockGroup {
            id: HashId::pad_str(3),
            collection_name: "test".to_string(),
            sample_name: "grandchild".to_string(),
            name: "bg".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: Some(child.id),
            is_default: false,
        };

        let block_groups = [grandchild.clone(), child.clone(), parent.clone()];
        let ordered = block_groups_parent_first(&block_groups);
        assert_eq!(ordered, vec![&parent, &child, &grandchild]);
    }

    #[test]
    fn test_block_groups_parent_first_when_every_entry_has_a_parent() {
        let root_id = HashId::pad_str(1);
        let child = BlockGroup {
            id: HashId::pad_str(2),
            collection_name: "test".to_string(),
            sample_name: "child".to_string(),
            name: "bg".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: Some(root_id),
            is_default: false,
        };
        let grandchild = BlockGroup {
            id: HashId::pad_str(3),
            collection_name: "test".to_string(),
            sample_name: "grandchild".to_string(),
            name: "bg".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: Some(child.id),
            is_default: false,
        };

        let block_groups = [grandchild.clone(), child.clone()];
        let ordered = block_groups_parent_first(&block_groups);
        assert_eq!(ordered, vec![&child, &grandchild]);
    }

    #[cfg(test)]
    mod changeset_dependencies {
        use super::*;

        #[test]
        fn test_tracks_nodes_and_sequences_from_previous_block_group_edges() {
            let context = setup_gen();
            let conn = context.graph().conn();
            let op_conn = context.operations().conn();

            let db_uuid = crate::metadata::get_db_uuid(conn);
            crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path")
                .unwrap();

            let (bg_id, _path_id) = setup_block_group(conn);
            let old_edges = BlockGroupEdge::edges_for_block_group(conn, &bg_id);

            let mut session = start_operation(conn);
            // make a blockgroup with an edge from our parent blockgroup
            let _ = Sample::create(
                conn,
                crate::sample::NewSample {
                    name: "new",
                    is_reference: false,
                },
            )
            .unwrap();
            let new_bg = BlockGroup::create(
                conn,
                NewBlockGroup {
                    collection_name: "test",
                    sample_name: "new",
                    name: "new-bg",
                    ..Default::default()
                },
            )
            .unwrap();
            let shared_edge = old_edges[0].edge.clone();
            BlockGroupEdge::bulk_create(
                conn,
                &[BlockGroupEdgeData {
                    block_group_id: new_bg.id,
                    edge_id: shared_edge.id,
                    chromosome_index: 0,
                    phased: 0,
                }],
            );
            let operation = end_operation(
                &context,
                &mut session,
                &OperationInfo {
                    files: vec![],
                    description: "test".to_string(),
                },
                "test",
                None,
            )
            .unwrap();

            let dependencies = operation.get_changeset_dependencies(context.workspace());
            assert_eq!(dependencies.nodes[0].id, shared_edge.target_node_id);
            assert_eq!(dependencies.nodes.len(), 1);
            let nodes = Node::query_by_ids(conn, &[shared_edge.target_node_id]);
            assert_eq!(dependencies.sequences[0].hash, nodes[0].sequence_hash);
            assert_eq!(dependencies.sequences.len(), 1);
        }

        #[test]
        fn test_records_patch_dependencies() {
            let context = setup_gen();
            let conn = context.graph().conn();
            let op_conn = context.operations().conn();

            let db_uuid = crate::metadata::get_db_uuid(conn);
            crate::files::GenDatabase::create(op_conn, &db_uuid, "test_db", "test_db_path")
                .unwrap();

            // create some stuff before we attach to our main session that will be required as extra information
            let (bg_id, _path_id) = setup_block_group(conn);
            let dep_bg = BlockGroup::get_by_id(conn, &bg_id).unwrap();

            let existing_seq = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAATTTT")
                .save(conn)
                .unwrap();
            let existing_node_id =
                Node::create(conn, &existing_seq.hash, &HashId::convert_str("1")).unwrap();

            let mut session = start_operation(conn);

            let random_seq = Sequence::new()
                .sequence_type("DNA")
                .sequence("ATCG")
                .save(conn)
                .unwrap();
            let random_node_id =
                Node::create(conn, &random_seq.hash, &HashId::convert_str("2")).unwrap();

            let new_edge = Edge::create(
                conn,
                random_node_id,
                0,
                Strand::Forward,
                existing_node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            let block_group_edge = BlockGroupEdgeData {
                block_group_id: bg_id,
                edge_id: new_edge.id,
                chromosome_index: 0,
                phased: 0,
            };
            BlockGroupEdge::bulk_create(conn, &[block_group_edge]);
            fs::write(
                context.workspace().repo_root().unwrap().join("test"),
                b"test file content",
            )
            .unwrap();
            let operation = end_operation(
                &context,
                &mut session,
                &OperationInfo {
                    files: vec![
                        OperationFile::new("test".to_string()).set_file_type(FileTypes::None),
                    ],
                    description: "test".to_string(),
                },
                "test",
                None,
            )
            .unwrap();

            let dependencies = operation.get_changeset_dependencies(context.workspace());
            assert_eq!(dependencies.sequences.len(), 1);
            assert_eq!(
                dependencies.block_group[0].collection_name,
                dep_bg.collection_name
            );
            assert_eq!(dependencies.block_group[0].name, dep_bg.name);
            assert_eq!(dependencies.block_group[0].sample_name, dep_bg.sample_name);
        }
    }
}
