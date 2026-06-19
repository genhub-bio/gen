use std::{cell::RefCell, collections::HashSet};

use gen_core::{HashId, is_terminal};

use crate::{
    accession::{Accession, AccessionNode},
    annotations::{Annotation, AnnotationGroup, AnnotationGroupSample},
    block_group::BlockGroup,
    block_group_edge::BlockGroupEdge,
    changesets::ChangesetModels,
    collection::Collection,
    db::GraphConnection,
    edge::Edge,
    path::Path,
    path_edge::PathEdge,
    sample::Sample,
    sample_lineage::SampleLineage,
    sequence::Sequence,
    session_operations::DependencyModels,
    traits::Query,
};

#[derive(Default)]
pub struct OperationRecorder {
    changes: ChangesetModels,
}

thread_local! {
    static ACTIVE_RECORDER: RefCell<Option<OperationRecorder>> = const { RefCell::new(None) };
}

impl OperationRecorder {
    pub fn start() -> Self {
        ACTIVE_RECORDER.with(|active| {
            *active.borrow_mut() = Some(OperationRecorder::default());
        });
        OperationRecorder::default()
    }

    pub fn finish(&mut self, conn: &GraphConnection) -> (ChangesetModels, DependencyModels) {
        let changes = ACTIVE_RECORDER.with(|active| {
            active
                .borrow_mut()
                .take()
                .map(|recorder| recorder.changes)
                .unwrap_or_default()
        });
        let dependencies = dependencies_for_changes(conn, &changes);
        (changes, dependencies)
    }
}

fn with_active_recorder(f: impl FnOnce(&mut OperationRecorder)) {
    ACTIVE_RECORDER.with(|active| {
        if let Some(recorder) = active.borrow_mut().as_mut() {
            f(recorder);
        }
    });
}

pub fn record_collection(collection: Collection) {
    with_active_recorder(|recorder| recorder.changes.collections.push(collection));
}

pub fn record_sample(sample: Sample) {
    with_active_recorder(|recorder| recorder.changes.samples.push(sample));
}

pub fn record_sample_lineage(sample_lineage: SampleLineage) {
    with_active_recorder(|recorder| recorder.changes.sample_lineages.push(sample_lineage));
}

pub fn record_sequence(sequence: Sequence) {
    with_active_recorder(|recorder| recorder.changes.sequences.push(sequence));
}

pub fn record_block_group(block_group: BlockGroup) {
    with_active_recorder(|recorder| recorder.changes.block_groups.push(block_group));
}

pub fn record_path(path: Path) {
    with_active_recorder(|recorder| recorder.changes.paths.push(path));
}

pub fn record_node(node: NodeRecord) {
    with_active_recorder(|recorder| recorder.changes.nodes.push(node.into()));
}

pub fn record_edge(edge: Edge) {
    with_active_recorder(|recorder| recorder.changes.edges.push(edge));
}

pub fn record_block_group_edge(block_group_edge: BlockGroupEdge) {
    with_active_recorder(|recorder| recorder.changes.block_group_edges.push(block_group_edge));
}

pub fn record_path_edge(path_edge: PathEdge) {
    with_active_recorder(|recorder| recorder.changes.path_edges.push(path_edge));
}

pub fn record_accession(accession: Accession) {
    with_active_recorder(|recorder| recorder.changes.accessions.push(accession));
}

pub fn record_accession_node(accession_node: AccessionNode) {
    with_active_recorder(|recorder| recorder.changes.accession_nodes.push(accession_node));
}

pub fn record_annotation_group(annotation_group: AnnotationGroup) {
    with_active_recorder(|recorder| recorder.changes.annotation_groups.push(annotation_group));
}

pub fn record_annotation(annotation: Annotation) {
    with_active_recorder(|recorder| recorder.changes.annotations.push(annotation));
}

pub fn record_annotation_group_sample(annotation_group_sample: AnnotationGroupSample) {
    with_active_recorder(|recorder| {
        recorder
            .changes
            .annotation_group_samples
            .push(annotation_group_sample);
    });
}

pub struct NodeRecord {
    pub id: HashId,
    pub sequence_hash: HashId,
}

impl From<NodeRecord> for crate::node::Node {
    fn from(value: NodeRecord) -> Self {
        crate::node::Node {
            id: value.id,
            sequence_hash: value.sequence_hash,
        }
    }
}

fn dependencies_for_changes(conn: &GraphConnection, changes: &ChangesetModels) -> DependencyModels {
    let created_collections = changes
        .collections
        .iter()
        .map(|collection| collection.name.clone())
        .collect::<HashSet<_>>();
    let created_samples = changes
        .samples
        .iter()
        .map(|sample| sample.name.clone())
        .collect::<HashSet<_>>();
    let created_block_groups = changes
        .block_groups
        .iter()
        .map(|block_group| block_group.id)
        .collect::<HashSet<_>>();
    let created_paths = changes
        .paths
        .iter()
        .map(|path| path.id)
        .collect::<HashSet<_>>();
    let created_accessions = changes
        .accessions
        .iter()
        .map(|accession| accession.id)
        .collect::<HashSet<_>>();
    let created_edges = changes
        .edges
        .iter()
        .map(|edge| edge.id)
        .collect::<HashSet<_>>();
    let created_nodes = changes
        .nodes
        .iter()
        .map(|node| node.id)
        .collect::<HashSet<_>>();
    let created_sequences = changes
        .sequences
        .iter()
        .map(|sequence| sequence.hash)
        .collect::<HashSet<_>>();

    let mut previous_collections = HashSet::new();
    let mut previous_samples = HashSet::new();
    let mut previous_block_groups = HashSet::new();
    let mut previous_edges = HashSet::new();
    let mut previous_paths = HashSet::new();
    let mut previous_accessions = HashSet::new();
    let mut previous_nodes = HashSet::new();
    let mut previous_sequences = HashSet::new();

    for block_group in &changes.block_groups {
        if !created_collections.contains(&block_group.collection_name) {
            previous_collections.insert(block_group.collection_name.clone());
        }
        if !created_samples.contains(&block_group.sample_name) {
            previous_samples.insert(block_group.sample_name.clone());
        }
        if let Some(parent_block_group_id) = block_group.parent_block_group_id
            && !created_block_groups.contains(&parent_block_group_id)
        {
            previous_block_groups.insert(parent_block_group_id);
        }
    }

    for node in &changes.nodes {
        if !created_sequences.contains(&node.sequence_hash) {
            previous_sequences.insert(node.sequence_hash);
        }
    }

    for edge in &changes.edges {
        for node_id in [edge.source_node_id, edge.target_node_id] {
            if !created_nodes.contains(&node_id) && !is_terminal(node_id) {
                previous_nodes.insert(node_id);
            }
        }
    }

    for block_group_edge in &changes.block_group_edges {
        if !created_edges.contains(&block_group_edge.edge_id) {
            previous_edges.insert(block_group_edge.edge_id);
        }
        if !created_block_groups.contains(&block_group_edge.block_group_id) {
            previous_block_groups.insert(block_group_edge.block_group_id);
        }
    }

    for sample_lineage in &changes.sample_lineages {
        if !created_samples.contains(&sample_lineage.parent_sample_name) {
            previous_samples.insert(sample_lineage.parent_sample_name.clone());
        }
        if !created_samples.contains(&sample_lineage.child_sample_name) {
            previous_samples.insert(sample_lineage.child_sample_name.clone());
        }
    }

    for path in &changes.paths {
        if !created_block_groups.contains(&path.block_group_id) {
            previous_block_groups.insert(path.block_group_id);
        }
    }

    for path_edge in &changes.path_edges {
        if !created_paths.contains(&path_edge.path_id) {
            previous_paths.insert(path_edge.path_id);
        }
        if !created_edges.contains(&path_edge.edge_id) {
            previous_edges.insert(path_edge.edge_id);
        }
    }

    for accession in &changes.accessions {
        if !created_block_groups.contains(&accession.block_group_id) {
            previous_block_groups.insert(accession.block_group_id);
        }
        if let Some(parent_accession_id) = accession.parent_accession_id
            && !created_accessions.contains(&parent_accession_id)
        {
            previous_accessions.insert(parent_accession_id);
        }
    }

    for accession_node in &changes.accession_nodes {
        if !created_accessions.contains(&accession_node.accession_id) {
            previous_accessions.insert(accession_node.accession_id);
        }
        if !created_nodes.contains(&accession_node.node_id) && !is_terminal(accession_node.node_id)
        {
            previous_nodes.insert(accession_node.node_id);
        }
    }

    for annotation in &changes.annotations {
        if !created_accessions.contains(&annotation.accession_id) {
            previous_accessions.insert(annotation.accession_id);
        }
    }

    for annotation_group_sample in &changes.annotation_group_samples {
        if !created_samples.contains(&annotation_group_sample.sample_name) {
            previous_samples.insert(annotation_group_sample.sample_name.clone());
        }
    }

    let existing_edges =
        Edge::query_by_ids(conn, &previous_edges.iter().copied().collect::<Vec<_>>());
    for edge in &existing_edges {
        for node_id in [edge.source_node_id, edge.target_node_id] {
            if !created_nodes.contains(&node_id) && !is_terminal(node_id) {
                previous_nodes.insert(node_id);
            }
        }
    }

    for node in
        crate::node::Node::query_by_ids(conn, &previous_nodes.iter().copied().collect::<Vec<_>>())
    {
        if !created_sequences.contains(&node.sequence_hash) {
            previous_sequences.insert(node.sequence_hash);
        }
    }

    let previous_accession_nodes = AccessionNode::query_accessions(
        conn,
        &previous_accessions.iter().copied().collect::<Vec<_>>(),
    )
    .unwrap_or_default()
    .into_values()
    .flatten()
    .collect::<Vec<_>>();

    DependencyModels {
        collections: Collection::query_by_ids(conn, &previous_collections),
        samples: Sample::query_by_ids(conn, &previous_samples),
        sequences: Sequence::query_by_ids(conn, &previous_sequences),
        block_group: BlockGroup::query_by_ids(conn, &previous_block_groups),
        nodes: crate::node::Node::query_by_ids(
            conn,
            &previous_nodes.iter().copied().collect::<Vec<_>>(),
        ),
        edges: Edge::query_by_ids(conn, &previous_edges.iter().copied().collect::<Vec<_>>()),
        paths: Path::query_by_ids(conn, &previous_paths),
        accessions: Accession::query_by_ids(conn, &previous_accessions),
        accession_nodes: previous_accession_nodes,
    }
}
