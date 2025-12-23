use std::collections::{HashMap, HashSet};

use gen_core::HashId;
use gen_graph::{GenGraph, connect_all_boundary_edges};
use gen_models::{
    block_group::BlockGroup, changesets::ChangesetModels, errors::OperationError,
    operations::Operation, session_operations::DependencyModels, traits::Query,
};
use petgraph::Direction;
use rusqlite::Connection;
use thiserror::Error;

use crate::views::patch::get_change_graph;

#[derive(Debug, Error)]
pub enum OperationDiffError {
    #[error("No current operation is checked out.")]
    NoCurrentOperation,
    #[error("Operation {0} not found.")]
    OperationMissing(HashId),
    #[error("Unable to find path between {0} and {1}.")]
    PathNotFound(HashId, HashId),
    #[error("Missing changeset data for operation {0}.")]
    MissingChangeset(HashId),
    #[error(transparent)]
    OperationError(#[from] OperationError),
}

#[derive(Clone, Debug)]
pub struct BlockGroupDiff {
    pub id: HashId,
    pub block_group: Option<BlockGroup>,
    pub graph: GenGraph,
}

#[derive(Clone, Debug)]
pub struct OperationDiff {
    pub operations: Vec<HashId>,
    pub added_block_groups: Vec<BlockGroupDiff>,
    pub removed_block_groups: Vec<BlockGroupDiff>,
}

pub fn collect_operation_diff(
    op_conn: &Connection,
    from_hash: HashId,
    to_hash: HashId,
) -> Result<OperationDiff, OperationDiffError> {
    if from_hash == to_hash {
        return Ok(OperationDiff {
            operations: vec![],
            added_block_groups: vec![],
            removed_block_groups: vec![],
        });
    }

    let path = Operation::get_path_between(op_conn, from_hash, to_hash);
    if path.is_empty() {
        return Err(OperationDiffError::PathNotFound(from_hash, to_hash));
    }

    let mut operations_in_order = vec![];
    let mut added_ops = vec![];
    let mut removed_ops = vec![];
    for (src, direction, dest) in path {
        let op_hash = match direction {
            Direction::Outgoing => dest,
            Direction::Incoming => src,
        };
        operations_in_order.push(op_hash);
        match direction {
            Direction::Outgoing => {
                added_ops.push(op_hash);
            }
            Direction::Incoming => {
                removed_ops.push(op_hash);
            }
        }
    }

    let added_graphs = build_block_group_diffs(op_conn, &added_ops)?;
    let removed_graphs = build_block_group_diffs(op_conn, &removed_ops)?;

    Ok(OperationDiff {
        operations: operations_in_order,
        added_block_groups: added_graphs,
        removed_block_groups: removed_graphs,
    })
}

fn build_block_group_diffs(
    op_conn: &Connection,
    operations: &[HashId],
) -> Result<Vec<BlockGroupDiff>, OperationDiffError> {
    if operations.is_empty() {
        return Ok(vec![]);
    }

    let mut block_group_info: HashMap<HashId, BlockGroup> = HashMap::new();

    let mut block_groups = HashSet::new();
    let mut edges = HashSet::new();
    let mut bges = HashSet::new();
    let mut nodes = HashSet::new();
    let mut sequences = HashSet::new();

    let mut dep_edges = HashSet::new();
    let mut dep_nodes = HashSet::new();
    let mut dep_sequences = HashSet::new();

    for op_hash in operations {
        let operation = Operation::get_by_id(op_conn, op_hash)
            .ok_or_else(|| OperationDiffError::OperationMissing(*op_hash))?;
        let changeset = operation.get_changeset().changes;
        let dependencies = operation.get_changeset_dependencies();

        for block_group in changeset
            .block_groups
            .iter()
            .chain(dependencies.block_group.iter())
        {
            block_group_info
                .entry(block_group.id)
                .or_insert_with(|| block_group.clone());
        }

        block_groups.extend(changeset.block_groups);
        edges.extend(changeset.edges);
        bges.extend(changeset.block_group_edges);
        nodes.extend(changeset.nodes);
        sequences.extend(changeset.sequences);
        dep_edges.extend(dependencies.edges);
        dep_nodes.extend(dependencies.nodes);
        dep_sequences.extend(dependencies.sequences);
    }

    let merged_graphs = get_change_graph(
        &ChangesetModels {
            block_groups: block_groups.into_iter().collect(),
            edges: edges.into_iter().collect(),
            block_group_edges: bges.into_iter().collect(),
            nodes: nodes.into_iter().collect(),
            sequences: sequences.into_iter().collect(),
            ..Default::default()
        },
        &DependencyModels {
            edges: dep_edges.into_iter().collect(),
            nodes: dep_nodes.into_iter().collect(),
            sequences: dep_sequences.into_iter().collect(),
            ..Default::default()
        },
    );

    let mut block_groups = merged_graphs
        .into_iter()
        .map(|(id, mut graph)| {
            let block_group = block_group_info.get(&id).cloned();
            connect_all_boundary_edges(&mut graph);
            BlockGroupDiff {
                id,
                block_group,
                graph,
            }
        })
        .collect::<Vec<_>>();
    block_groups.sort_by_key(|a| {
        if let Some(bg) = &a.block_group {
            (
                bg.collection_name.clone(),
                bg.sample_name
                    .clone()
                    .unwrap_or_else(|| "Reference".to_string()),
                bg.name.clone(),
                format!("{id}", id = a.id),
            )
        } else {
            (
                String::new(),
                String::new(),
                String::new(),
                format!("{id}", id = a.id),
            )
        }
    });

    Ok(block_groups)
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, Strand};
    use gen_models::{
        block_group::BlockGroup,
        block_group_edge::BlockGroupEdge,
        changesets::{ChangesetModels, DatabaseChangeset, write_changeset},
        edge::Edge,
        node::Node,
        operations::{Branch, Operation, OperationState},
        sequence::{NewSequence, Sequence},
    };

    use super::*;
    use crate::test_helpers::{get_operation_connection, setup_gen_dir};

    fn base_dependencies(start_node: &Node, end_node: &Node) -> DependencyModels {
        let mut start_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .name("start")
            .build();
        start_sequence.hash = start_node.sequence_hash;
        let mut end_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .name("end")
            .build();
        end_sequence.hash = end_node.sequence_hash;
        DependencyModels {
            collections: vec![],
            samples: vec![],
            sequences: vec![start_sequence, end_sequence],
            block_group: vec![],
            nodes: vec![start_node.clone(), end_node.clone()],
            edges: vec![],
            paths: vec![],
            accessions: vec![],
            accession_edges: vec![],
        }
    }

    fn init_branch() -> Connection {
        setup_gen_dir();
        let op_conn = get_operation_connection(None).unwrap();
        Branch::get_or_create(&op_conn, "main");
        OperationState::set_branch(&op_conn, "main");
        op_conn
    }

    fn simple_changeset(
        block_group: &BlockGroup,
        node: &Node,
        seq: &Sequence,
        start_node: &Node,
        end_node: &Node,
    ) -> (ChangesetModels, DependencyModels) {
        let edges = vec![
            Edge {
                id: HashId::convert_str(&format!("{}-{}-start", block_group.id, node.id)),
                source_node_id: start_node.id,
                source_coordinate: 0,
                source_strand: Strand::Forward,
                target_node_id: node.id,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            },
            Edge {
                id: HashId::convert_str(&format!("{}-{}-end", block_group.id, node.id)),
                source_node_id: node.id,
                source_coordinate: seq.length,
                source_strand: Strand::Forward,
                target_node_id: end_node.id,
                target_coordinate: 0,
                target_strand: Strand::Forward,
            },
        ];
        let block_group_edges = vec![
            BlockGroupEdge {
                id: HashId::convert_str(&format!("{}-{}-start-bge", block_group.id, node.id)),
                block_group_id: block_group.id,
                edge_id: edges[0].id,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            BlockGroupEdge {
                id: HashId::convert_str(&format!("{}-{}-end-bge", block_group.id, node.id)),
                block_group_id: block_group.id,
                edge_id: edges[1].id,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
        ];
        let changeset = ChangesetModels {
            collections: vec![],
            samples: vec![],
            sequences: vec![seq.clone()],
            block_groups: vec![block_group.clone()],
            nodes: vec![node.clone()],
            edges,
            block_group_edges,
            paths: vec![],
            path_edges: vec![],
            accessions: vec![],
            accession_edges: vec![],
            accession_paths: vec![],
        };
        let dependencies = base_dependencies(start_node, end_node);
        (changeset, dependencies)
    }

    #[test]
    fn diff_with_single_operation_uses_current_as_default_target() {
        let op_conn = init_branch();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let base_op =
            Operation::create(&op_conn, "seed", &HashId::pad_str(1)).expect("create base op");

        let seq_one = NewSequence::new()
            .sequence_type("dna")
            .sequence("AAAAA")
            .name("one")
            .build();
        let node_one = Node {
            id: HashId::pad_str(10),
            sequence_hash: seq_one.hash,
        };
        let block_group = BlockGroup {
            id: HashId::pad_str(3),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "bg".to_string(),
            created_on: 0,
        };

        let head = Operation::create(&op_conn, "add", &HashId::pad_str(2)).expect("create op");
        let (changeset, dependencies) =
            simple_changeset(&block_group, &node_one, &seq_one, &start_node, &end_node);
        write_changeset(
            &head,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset,
            },
            &dependencies,
        );

        let diff = collect_operation_diff(&op_conn, base_op.hash, head.hash).expect("diff");
        assert_eq!(diff.operations, vec![head.hash]);
        assert_eq!(diff.added_block_groups.len(), 1);
        assert!(diff.removed_block_groups.is_empty());
        let graph = &diff.added_block_groups[0].graph;
        assert_eq!(graph.nodes().count(), 3);
        assert_eq!(graph.all_edges().count(), 2);

        let _ = base_op; // silence unused warning for explicitly named base operation
    }

    #[test]
    fn diff_merges_multiple_operations() {
        let op_conn = init_branch();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let op1 = Operation::create(&op_conn, "seed", &HashId::pad_str(1)).expect("create base op");

        let bg_one = BlockGroup {
            id: HashId::pad_str(3),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "bg1".to_string(),
            created_on: 0,
        };
        let seq_one = NewSequence::new()
            .sequence_type("dna")
            .sequence("AAAAA")
            .name("one")
            .build();
        let node_one = Node {
            id: HashId::pad_str(10),
            sequence_hash: seq_one.hash,
        };
        let op2 = Operation::create(&op_conn, "add", &HashId::pad_str(2)).expect("create op2");
        let (changeset_one, dependencies_one) =
            simple_changeset(&bg_one, &node_one, &seq_one, &start_node, &end_node);
        write_changeset(
            &op2,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset_one,
            },
            &dependencies_one,
        );

        let bg_two = BlockGroup {
            id: HashId::pad_str(4),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "bg2".to_string(),
            created_on: 0,
        };
        let seq_two = NewSequence::new()
            .sequence_type("dna")
            .sequence("CCCCC")
            .name("two")
            .build();
        let node_two = Node {
            id: HashId::pad_str(11),
            sequence_hash: seq_two.hash,
        };
        let op3 = Operation::create(&op_conn, "add", &HashId::pad_str(3)).expect("create op3");
        let (changeset_two, dependencies_two) =
            simple_changeset(&bg_two, &node_two, &seq_two, &start_node, &end_node);
        write_changeset(
            &op3,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset_two,
            },
            &dependencies_two,
        );

        let diff = collect_operation_diff(&op_conn, op1.hash, op3.hash).expect("diff");
        assert_eq!(diff.operations, vec![op2.hash, op3.hash]);
        assert_eq!(diff.added_block_groups.len(), 2);
        for bg in &diff.added_block_groups {
            assert_eq!(bg.graph.all_edges().count(), 2);
        }
    }

    #[test]
    fn diff_merges_nested_changes_in_same_block_group() {
        let op_conn = init_branch();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let op1 = Operation::create(&op_conn, "seed", &HashId::pad_str(1)).expect("create base op");

        let block_group = BlockGroup {
            id: HashId::pad_str(5),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "bg".to_string(),
            created_on: 0,
        };

        let seq_one = NewSequence::new()
            .sequence_type("dna")
            .sequence("AAAAA")
            .name("one")
            .build();
        let node_one = Node {
            id: HashId::pad_str(12),
            sequence_hash: seq_one.hash,
        };
        let op2 = Operation::create(&op_conn, "add", &HashId::pad_str(2)).expect("create op2");
        let (changeset_one, mut dependencies_one) =
            simple_changeset(&block_group, &node_one, &seq_one, &start_node, &end_node);
        dependencies_one.block_group.push(block_group.clone());
        write_changeset(
            &op2,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset_one,
            },
            &dependencies_one,
        );

        let seq_two = NewSequence::new()
            .sequence_type("dna")
            .sequence("CCCC")
            .name("two")
            .build();
        let node_two = Node {
            id: HashId::pad_str(13),
            sequence_hash: seq_two.hash,
        };
        let op3 = Operation::create(&op_conn, "add", &HashId::pad_str(3)).expect("create op3");
        let (changeset_two, mut dependencies_two) =
            simple_changeset(&block_group, &node_two, &seq_two, &start_node, &end_node);
        dependencies_two.block_group.push(block_group.clone());
        write_changeset(
            &op3,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset_two,
            },
            &dependencies_two,
        );

        let diff = collect_operation_diff(&op_conn, op1.hash, op3.hash).expect("diff");
        assert_eq!(diff.operations, vec![op2.hash, op3.hash]);
        assert_eq!(diff.added_block_groups.len(), 1);
        let graph = &diff.added_block_groups[0].graph;
        assert_eq!(graph.nodes().count(), 4);
        assert_eq!(graph.all_edges().count(), 4);
    }

    #[test]
    fn diff_against_itself_is_empty() {
        let op_conn = init_branch();
        let base =
            Operation::create(&op_conn, "seed", &HashId::pad_str(1)).expect("create base op");
        let diff = collect_operation_diff(&op_conn, base.hash, base.hash).expect("diff");
        assert!(diff.operations.is_empty());
        assert!(diff.added_block_groups.is_empty());
        assert!(diff.removed_block_groups.is_empty());
    }

    #[test]
    fn diff_populates_removed_block_groups_on_branch_change() {
        let op_conn = init_branch();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let base = Operation::create(&op_conn, "seed", &HashId::pad_str(1)).expect("base op");

        let main_block_group = BlockGroup {
            id: HashId::pad_str(20),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "main".to_string(),
            created_on: 0,
        };
        let main_seq = NewSequence::new()
            .sequence_type("dna")
            .sequence("AAAAA")
            .name("main")
            .build();
        let main_node = Node {
            id: HashId::pad_str(21),
            sequence_hash: main_seq.hash,
        };
        let op_main = Operation::create(&op_conn, "add", &HashId::pad_str(2)).expect("main op");
        let (main_changeset, main_deps) = simple_changeset(
            &main_block_group,
            &main_node,
            &main_seq,
            &start_node,
            &end_node,
        );
        write_changeset(
            &op_main,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: main_changeset,
            },
            &main_deps,
        );

        let feature_branch = Branch::create_with_remote(&op_conn, "feature", None).unwrap();
        OperationState::set_branch(&op_conn, &feature_branch.name);
        OperationState::set_operation(&op_conn, &base.hash);

        let feature_block_group = BlockGroup {
            id: HashId::pad_str(30),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "feature".to_string(),
            created_on: 0,
        };
        let feature_seq = NewSequence::new()
            .sequence_type("dna")
            .sequence("CCCCC")
            .name("feature")
            .build();
        let feature_node = Node {
            id: HashId::pad_str(31),
            sequence_hash: feature_seq.hash,
        };
        let op_feature =
            Operation::create(&op_conn, "add", &HashId::pad_str(3)).expect("feature op");
        let (feature_changeset, feature_deps) = simple_changeset(
            &feature_block_group,
            &feature_node,
            &feature_seq,
            &start_node,
            &end_node,
        );
        write_changeset(
            &op_feature,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: feature_changeset,
            },
            &feature_deps,
        );

        let diff = collect_operation_diff(&op_conn, op_main.hash, op_feature.hash).expect("diff");
        assert_eq!(diff.operations, vec![op_main.hash, op_feature.hash]);
        assert_eq!(diff.added_block_groups.len(), 1);
        assert_eq!(diff.removed_block_groups.len(), 1);
        assert_eq!(diff.added_block_groups[0].id, feature_block_group.id);
        assert_eq!(diff.removed_block_groups[0].id, main_block_group.id);
    }
}
