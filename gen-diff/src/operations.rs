use std::collections::{HashMap, HashSet};

use gen_core::{HashId, Workspace};
use gen_models::{
    block_group::BlockGroup, changesets::ChangesetModels, db::OperationsConnection,
    errors::OperationError, operations::Operation, session_operations::DependencyModels,
    traits::Query,
};
use petgraph::Direction;
use thiserror::Error;

use crate::graph::{DiffGenGraph, connect_all_boundary_edges_diff, get_diff_graph};

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
    pub graph: DiffGenGraph,
}

#[derive(Clone, Debug)]
pub struct OperationDiff {
    pub operations: Vec<HashId>,
    pub dbs: HashMap<String, DbDiff>,
}

#[derive(Clone, Debug)]
pub struct DbDiff {
    pub db_path: String,
    pub added_block_groups: Vec<BlockGroupDiff>,
    pub removed_block_groups: Vec<BlockGroupDiff>,
}

#[derive(Clone, Debug, Default)]
pub struct BlockGroupDiffs {
    pub operations: Vec<HashId>,
    pub block_group_diffs: Vec<BlockGroupDiff>,
}

fn build_operation_diffs(
    operations_in_order: &[HashId],
    added_graphs: &HashMap<String, BlockGroupDiffs>,
    removed_graphs: &HashMap<String, BlockGroupDiffs>,
) -> HashMap<String, OperationDiff> {
    let mut db_paths = HashSet::new();
    db_paths.extend(added_graphs.keys().cloned());
    db_paths.extend(removed_graphs.keys().cloned());

    let mut diffs = HashMap::new();
    for db_path in db_paths {
        let mut op_set = HashSet::new();
        if let Some(diffs) = added_graphs.get(&db_path) {
            op_set.extend(diffs.operations.iter().copied());
        }
        if let Some(diffs) = removed_graphs.get(&db_path) {
            op_set.extend(diffs.operations.iter().copied());
        }
        let operations = operations_in_order
            .iter()
            .copied()
            .filter(|hash| op_set.contains(hash))
            .collect::<Vec<_>>();
        let db_diff = DbDiff {
            db_path: db_path.clone(),
            added_block_groups: added_graphs
                .get(&db_path)
                .map(|diffs| diffs.block_group_diffs.clone())
                .unwrap_or_default(),
            removed_block_groups: removed_graphs
                .get(&db_path)
                .map(|diffs| diffs.block_group_diffs.clone())
                .unwrap_or_default(),
        };
        let mut dbs = HashMap::new();
        dbs.insert(db_path.clone(), db_diff);
        diffs.insert(db_path, OperationDiff { operations, dbs });
    }

    diffs
}

pub fn collect_operation_diff(
    workspace: &Workspace,
    op_conn: &OperationsConnection,
    from_hash: Option<HashId>,
    to_hash: HashId,
    db_path: Option<&str>,
) -> Result<HashMap<String, OperationDiff>, OperationDiffError> {
    let (operations_in_order, added_ops, removed_ops) = if let Some(from_hash) = from_hash {
        if from_hash == to_hash {
            return Ok(HashMap::new());
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

        (operations_in_order, added_ops, removed_ops)
    } else {
        (vec![to_hash], vec![to_hash], vec![])
    };

    let added_graphs = build_block_group_diffs(workspace, op_conn, &added_ops, db_path)?;
    let removed_graphs = build_block_group_diffs(workspace, op_conn, &removed_ops, db_path)?;

    Ok(build_operation_diffs(
        &operations_in_order,
        &added_graphs,
        &removed_graphs,
    ))
}

/// The idea here is to build a merged changeset from the changesets of operations. This changeset is then fed into
/// the machinery for rendering single operation changesets. If multiple operations are passed in, this is a merged
/// changeset that represents the combined effect of all the operations.
fn build_block_group_diffs(
    workspace: &Workspace,
    op_conn: &OperationsConnection,
    operations: &[HashId],
    db_path: Option<&str>,
) -> Result<HashMap<String, BlockGroupDiffs>, OperationDiffError> {
    if operations.is_empty() {
        return Ok(HashMap::new());
    }

    #[derive(Default)]
    struct DbAccumulator {
        operations: Vec<HashId>,
        block_group_info: HashMap<HashId, BlockGroup>,
        block_groups: HashSet<gen_models::block_group::BlockGroup>,
        edges: HashSet<gen_models::edge::Edge>,
        block_group_edges: HashSet<gen_models::block_group_edge::BlockGroupEdge>,
        nodes: HashSet<gen_models::node::Node>,
        sequences: HashSet<gen_models::sequence::Sequence>,
        dep_edges: HashSet<gen_models::edge::Edge>,
        dep_nodes: HashSet<gen_models::node::Node>,
        dep_sequences: HashSet<gen_models::sequence::Sequence>,
    }

    let mut accumulators: HashMap<String, DbAccumulator> = HashMap::new();

    for op_hash in operations {
        let operation = Operation::get_by_id(op_conn, op_hash)
            .ok_or_else(|| OperationDiffError::OperationMissing(*op_hash))?;
        let changeset = operation.get_changeset(workspace);
        if let Some(db_path) = db_path
            && changeset.db_path != db_path
        {
            continue;
        }
        let changeset_db = changeset.db_path.clone();
        let changeset = changeset.changes;
        let dependencies = operation.get_changeset_dependencies(workspace);

        let entry = accumulators.entry(changeset_db).or_default();
        entry.operations.push(*op_hash);

        for block_group in changeset
            .block_groups
            .iter()
            .chain(dependencies.block_group.iter())
        {
            entry
                .block_group_info
                .entry(block_group.id)
                .or_insert_with(|| block_group.clone());
        }

        entry.block_groups.extend(changeset.block_groups);
        entry.edges.extend(changeset.edges);
        entry.block_group_edges.extend(changeset.block_group_edges);
        entry.nodes.extend(changeset.nodes);
        entry.sequences.extend(changeset.sequences);
        entry.dep_edges.extend(dependencies.edges);
        entry.dep_nodes.extend(dependencies.nodes);
        entry.dep_sequences.extend(dependencies.sequences);
    }

    let mut results = HashMap::new();
    for (db_path, acc) in accumulators {
        let merged_graphs = get_diff_graph(
            &ChangesetModels {
                block_groups: acc.block_groups.into_iter().collect(),
                edges: acc.edges.into_iter().collect(),
                block_group_edges: acc.block_group_edges.into_iter().collect(),
                nodes: acc.nodes.into_iter().collect(),
                sequences: acc.sequences.into_iter().collect(),
                ..Default::default()
            },
            &DependencyModels {
                edges: acc.dep_edges.into_iter().collect(),
                nodes: acc.dep_nodes.into_iter().collect(),
                sequences: acc.dep_sequences.into_iter().collect(),
                ..Default::default()
            },
        );

        let mut block_groups = merged_graphs
            .into_iter()
            .map(|(id, mut graph)| {
                let block_group = acc.block_group_info.get(&id).cloned();
                connect_all_boundary_edges_diff(&mut graph);
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
        results.insert(
            db_path,
            BlockGroupDiffs {
                operations: acc.operations,
                block_group_diffs: block_groups,
            },
        );
    }

    Ok(results)
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
    use crate::test_helpers::setup_gen;

    fn get_db_diff<'a>(diffs: &'a HashMap<String, OperationDiff>, db_path: &str) -> &'a DbDiff {
        diffs
            .get(db_path)
            .and_then(|diff| diff.dbs.get(db_path))
            .expect("db diff")
    }

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
            annotation_groups: vec![],
            annotations: vec![],
            annotation_samples: vec![],
        };
        let dependencies = base_dependencies(start_node, end_node);
        (changeset, dependencies)
    }

    #[test]
    fn one_operation_diff() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let workspace = context.workspace();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let base_op =
            Operation::create(op_conn, "seed", &HashId::pad_str(1)).expect("create base op");

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

        let head = Operation::create(op_conn, "add", &HashId::pad_str(2)).expect("create op");
        let (changeset, dependencies) =
            simple_changeset(&block_group, &node_one, &seq_one, &start_node, &end_node);
        write_changeset(
            workspace,
            &head,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset,
            },
            &dependencies,
        );

        let diffs = collect_operation_diff(workspace, op_conn, Some(base_op.hash), head.hash, None)
            .expect("diff");
        let diff = diffs.get("diff.db").expect("diff db");
        let db_diff = get_db_diff(&diffs, "diff.db");
        assert_eq!(diff.operations, vec![head.hash]);
        assert_eq!(db_diff.added_block_groups.len(), 1);
        assert!(db_diff.removed_block_groups.is_empty());
        let graph = &db_diff.added_block_groups[0].graph;
        assert_eq!(graph.nodes().count(), 3);
        assert_eq!(graph.all_edges().count(), 2);
    }

    #[test]
    fn initial_operation_diff_contains_added_block_groups() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let workspace = context.workspace();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

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

        let head = Operation::create(op_conn, "add", &HashId::pad_str(2)).expect("create op");
        let (changeset, dependencies) =
            simple_changeset(&block_group, &node_one, &seq_one, &start_node, &end_node);
        write_changeset(
            workspace,
            &head,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset,
            },
            &dependencies,
        );

        let diffs =
            collect_operation_diff(workspace, op_conn, None, head.hash, None).expect("diff");
        let diff = diffs.get("diff.db").expect("diff db");
        let db_diff = get_db_diff(&diffs, "diff.db");
        assert_eq!(diff.operations, vec![head.hash]);
        assert_eq!(db_diff.added_block_groups.len(), 1);
        assert!(db_diff.removed_block_groups.is_empty());
        let graph = &db_diff.added_block_groups[0].graph;
        assert_eq!(graph.nodes().count(), 3);
        assert_eq!(graph.all_edges().count(), 2);
    }

    #[test]
    fn merges_multiple_operations() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let workspace = context.workspace();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let op1 = Operation::create(op_conn, "seed", &HashId::pad_str(1)).expect("create base op");

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
        let op2 = Operation::create(op_conn, "add", &HashId::pad_str(2)).expect("create op2");
        let (changeset_one, dependencies_one) =
            simple_changeset(&bg_one, &node_one, &seq_one, &start_node, &end_node);
        write_changeset(
            workspace,
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
        let op3 = Operation::create(op_conn, "add", &HashId::pad_str(3)).expect("create op3");
        let (changeset_two, dependencies_two) =
            simple_changeset(&bg_two, &node_two, &seq_two, &start_node, &end_node);
        write_changeset(
            workspace,
            &op3,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: changeset_two,
            },
            &dependencies_two,
        );

        let diffs = collect_operation_diff(workspace, op_conn, Some(op1.hash), op3.hash, None)
            .expect("diff");
        let diff = diffs.get("diff.db").expect("diff db");
        let db_diff = get_db_diff(&diffs, "diff.db");
        assert_eq!(diff.operations, vec![op2.hash, op3.hash]);
        assert_eq!(db_diff.added_block_groups.len(), 2);
    }

    #[test]
    fn diff_against_itself_is_empty() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let workspace = context.workspace();
        let base = Operation::create(op_conn, "seed", &HashId::pad_str(1)).expect("create base op");
        let diffs = collect_operation_diff(workspace, op_conn, Some(base.hash), base.hash, None)
            .expect("diff");
        assert!(diffs.is_empty());
    }

    #[test]
    fn diffs_across_branches() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let workspace = context.workspace();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let base = Operation::create(op_conn, "seed", &HashId::pad_str(1)).expect("base op");

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
        let op_main = Operation::create(op_conn, "add", &HashId::pad_str(2)).expect("main op");
        let (main_changeset, main_deps) = simple_changeset(
            &main_block_group,
            &main_node,
            &main_seq,
            &start_node,
            &end_node,
        );
        write_changeset(
            workspace,
            &op_main,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: main_changeset,
            },
            &main_deps,
        );

        let feature_branch = Branch::create_with_remote(op_conn, "feature", None).unwrap();
        OperationState::set_branch(op_conn, &feature_branch.name);
        OperationState::set_operation(op_conn, &base.hash);

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
            Operation::create(op_conn, "add", &HashId::pad_str(3)).expect("feature op");
        let (feature_changeset, feature_deps) = simple_changeset(
            &feature_block_group,
            &feature_node,
            &feature_seq,
            &start_node,
            &end_node,
        );
        write_changeset(
            workspace,
            &op_feature,
            DatabaseChangeset {
                db_path: "diff.db".to_string(),
                changes: feature_changeset,
            },
            &feature_deps,
        );

        let diffs = collect_operation_diff(
            workspace,
            op_conn,
            Some(op_main.hash),
            op_feature.hash,
            None,
        )
        .expect("diff");
        let diff = diffs.get("diff.db").expect("diff db");
        let db_diff = get_db_diff(&diffs, "diff.db");
        assert_eq!(diff.operations, vec![op_main.hash, op_feature.hash]);
        assert_eq!(db_diff.added_block_groups.len(), 1);
        assert_eq!(db_diff.removed_block_groups.len(), 1);
        assert_eq!(db_diff.added_block_groups[0].id, feature_block_group.id);
        assert_eq!(db_diff.removed_block_groups[0].id, main_block_group.id);
    }

    #[test]
    fn filters_by_database_path() {
        let context = setup_gen();
        let op_conn = context.operations().conn();
        let workspace = context.workspace();
        let start_node = Node::get_start_node();
        let end_node = Node::get_end_node();

        let base = Operation::create(op_conn, "seed", &HashId::pad_str(1)).expect("base op");

        let block_group_one = BlockGroup {
            id: HashId::pad_str(40),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "db-one".to_string(),
            created_on: 0,
        };
        let seq_one = NewSequence::new()
            .sequence_type("dna")
            .sequence("AAAAA")
            .name("one")
            .build();
        let node_one = Node {
            id: HashId::pad_str(41),
            sequence_hash: seq_one.hash,
        };
        let op_one = Operation::create(op_conn, "add", &HashId::pad_str(2)).expect("op one");
        let (changeset_one, deps_one) = simple_changeset(
            &block_group_one,
            &node_one,
            &seq_one,
            &start_node,
            &end_node,
        );
        write_changeset(
            workspace,
            &op_one,
            DatabaseChangeset {
                db_path: "db-one.db".to_string(),
                changes: changeset_one,
            },
            &deps_one,
        );

        let block_group_two = BlockGroup {
            id: HashId::pad_str(50),
            collection_name: "c".to_string(),
            sample_name: Some("s".to_string()),
            name: "db-two".to_string(),
            created_on: 0,
        };
        let seq_two = NewSequence::new()
            .sequence_type("dna")
            .sequence("CCCCC")
            .name("two")
            .build();
        let node_two = Node {
            id: HashId::pad_str(51),
            sequence_hash: seq_two.hash,
        };
        let op_two = Operation::create(op_conn, "add", &HashId::pad_str(3)).expect("op two");
        let (changeset_two, deps_two) = simple_changeset(
            &block_group_two,
            &node_two,
            &seq_two,
            &start_node,
            &end_node,
        );
        write_changeset(
            workspace,
            &op_two,
            DatabaseChangeset {
                db_path: "db-two.db".to_string(),
                changes: changeset_two,
            },
            &deps_two,
        );

        let diffs = collect_operation_diff(
            workspace,
            op_conn,
            Some(base.hash),
            op_two.hash,
            Some("db-one.db"),
        )
        .expect("diff");
        let diff = diffs.get("db-one.db").expect("diff db");
        let db_diff = get_db_diff(&diffs, "db-one.db");
        assert_eq!(diff.operations, vec![op_one.hash]);
        assert_eq!(db_diff.added_block_groups.len(), 1);
        assert_eq!(db_diff.added_block_groups[0].id, block_group_one.id);

        let diff_none = collect_operation_diff(
            workspace,
            op_conn,
            Some(base.hash),
            op_two.hash,
            Some("missing.db"),
        )
        .expect("diff");
        assert!(diff_none.is_empty());
    }
}
