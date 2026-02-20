use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
};

use gen_core::{
    HashId, NO_CHROMOSOME_INDEX, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID,
    PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, PathBlock, Strand, calculate_hash, is_end_node,
    is_start_node, is_terminal, traits::Capnp,
};
use gen_graph::{
    GenGraph, GraphNode, all_intermediate_edges, all_reachable_nodes, all_simple_paths,
    flatten_to_interval_tree,
};
use intervaltree::IntervalTree;
use rusqlite::{Row, params, types::Value as SQLValue};
use serde::{Deserialize, Serialize};

use crate::{
    accession::{Accession, AccessionEdge, AccessionEdgeData, AccessionPath},
    block_group_edge::{AugmentedEdgeData, BlockGroupEdge, BlockGroupEdgeData},
    db::GraphConnection,
    edge::{Edge, EdgeData, GroupBlock},
    errors::{ChangeError, QueryError},
    gen_models_capnp::block_group,
    node::Node,
    path::{Path, PathData},
    path_edge::PathEdge,
    traits::*,
};

#[derive(Clone, Debug, Deserialize, Eq, Hash, Serialize, PartialEq)]
pub struct BlockGroup {
    pub id: HashId,
    pub collection_name: String,
    pub sample_name: Option<String>,
    pub name: String,
    pub created_on: i64,
}

impl<'a> Capnp<'a> for BlockGroup {
    type Builder = block_group::Builder<'a>;
    type Reader = block_group::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_collection_name(&self.collection_name);
        match &self.sample_name {
            None => {
                builder.reborrow().get_sample_name().set_none(());
            }
            Some(n) => {
                builder.reborrow().get_sample_name().set_some(n.clone());
            }
        }
        builder.set_name(&self.name);
        builder.set_created_on(self.created_on);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let collection_name = reader.get_collection_name().unwrap().to_string().unwrap();
        let sample_name: Option<String> = match reader.get_sample_name().which().unwrap() {
            block_group::sample_name::None(()) => None,
            block_group::sample_name::Some(n) => Some(n.unwrap().to_string().unwrap()),
        };
        let name = reader.get_name().unwrap().to_string().unwrap();
        let created_on = reader.get_created_on();

        BlockGroup {
            id,
            collection_name,
            sample_name,
            name,
            created_on,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockGroupData<'a> {
    pub collection_name: &'a str,
    pub sample_name: Option<&'a str>,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct PathChange {
    pub block_group_id: HashId,
    pub path: Path,
    pub path_accession: Option<String>,
    pub start: i64,
    pub end: i64,
    pub block: PathBlock,
    pub chromosome_index: i64,
    pub phased: i64,
    pub preserve_edge: bool,
}

pub struct PathCache<'a> {
    pub cache: HashMap<PathData, Path>,
    pub intervaltree_cache: HashMap<Path, IntervalTree<i64, NodeIntervalBlock>>,
    pub conn: &'a GraphConnection,
}

impl<'a> PathCache<'a> {
    pub fn new(conn: &'a GraphConnection) -> PathCache<'a> {
        PathCache {
            cache: HashMap::<PathData, Path>::new(),
            intervaltree_cache: HashMap::<Path, IntervalTree<i64, NodeIntervalBlock>>::new(),
            conn,
        }
    }

    pub fn lookup(path_cache: &mut PathCache, block_group_id: &HashId, name: String) -> Path {
        let path_key = PathData {
            name: name.clone(),
            block_group_id: *block_group_id,
        };
        let path_lookup = path_cache.cache.get(&path_key);
        if let Some(path) = path_lookup {
            path.clone()
        } else {
            let conn = path_cache.conn;
            let new_path = Path::query(
                conn,
                "select * from paths where block_group_id = ?1 AND name = ?2",
                params![block_group_id, name],
            )[0]
            .clone();

            path_cache.cache.insert(path_key, new_path.clone());
            let tree = new_path.intervaltree(conn);
            path_cache.intervaltree_cache.insert(new_path.clone(), tree);
            new_path
        }
    }

    pub fn get_intervaltree<'b>(
        path_cache: &'b PathCache<'_>,
        path: &Path,
    ) -> Option<&'b IntervalTree<i64, NodeIntervalBlock>> {
        path_cache.intervaltree_cache.get(path)
    }
}

impl BlockGroup {
    pub fn create(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: Option<&str>,
        name: &str,
    ) -> BlockGroup {
        let hash = HashId(calculate_hash(&format!(
            "{collection_name}:{sample_name:?}:{name}"
        )));
        let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap();
        let query = "INSERT INTO block_groups (id, collection_name, sample_name, name, created_on) VALUES (?1, ?2, ?3, ?4, ?5);";
        let mut stmt = conn.prepare(query).unwrap();
        let bg = BlockGroup {
            id: hash,
            collection_name: collection_name.to_string(),
            sample_name: sample_name.map(|s| s.to_string()),
            name: name.to_string(),
            created_on: timestamp,
        };
        match stmt.execute((hash, collection_name, sample_name, name, timestamp)) {
            Ok(_) => bg,
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    bg
                } else {
                    panic!("something bad happened querying the database")
                }
            }
            Err(_) => {
                panic!("something bad happened querying the database")
            }
        }
    }

    pub fn delete(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: Option<&str>,
        name: &str,
    ) {
        if let Some(n) = sample_name {
            let query = "delete from block_groups where collection_name = ?1 and sample_name = ?2 and name = ?3;";
            conn.execute(
                query,
                params![collection_name.to_string(), n.to_string(), name.to_string()],
            )
            .unwrap();
        } else {
            let query = "delete from block_groups where collection_name = ?1 and sample_name is null and name = ?2;";
            conn.execute(
                query,
                params![collection_name.to_string(), name.to_string()],
            )
            .unwrap();
        }
    }

    pub fn get_by_id(conn: &GraphConnection, id: &HashId) -> BlockGroup {
        let query = "SELECT * FROM block_groups WHERE id = ?1";
        let mut stmt = conn.prepare(query).unwrap();
        match stmt.query_row([id], |row| Ok(Self::process_row(row))) {
            Ok(res) => res,
            Err(rusqlite::Error::QueryReturnedNoRows) => panic!("No block group with id {id}"),
            Err(_) => {
                panic!("something bad happened querying the database")
            }
        }
    }

    pub fn duplicate(&self, conn: &GraphConnection, target_block_group_id: &HashId) {
        let existing_paths = Path::query(
            conn,
            "SELECT * from paths where block_group_id = ?1;",
            params![self.id],
        );

        let augmented_edges = BlockGroupEdge::edges_for_block_group(conn, &self.id);
        let edge_ids = augmented_edges
            .iter()
            .map(|edge| edge.edge.id)
            .collect::<Vec<_>>();
        let new_block_group_edges = edge_ids
            .iter()
            .enumerate()
            .map(|(i, edge_id)| BlockGroupEdgeData {
                block_group_id: *target_block_group_id,
                edge_id: *edge_id,
                chromosome_index: augmented_edges[i].chromosome_index,
                phased: augmented_edges[i].phased,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);

        let mut path_map = HashMap::new();

        for path in existing_paths.iter() {
            let edge_ids = PathEdge::edges_for_path(conn, &path.id)
                .into_iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>();
            let new_path = Path::create(conn, &path.name, target_block_group_id, &edge_ids);
            path_map.insert(&path.id, new_path.id);
        }

        for accession in Accession::query(
            conn,
            "select * from accessions where path_id IN rarray(?1);",
            rusqlite::params!(Rc::new(
                existing_paths
                    .iter()
                    .map(|path| SQLValue::from(path.id))
                    .collect::<Vec<SQLValue>>()
            )),
        ) {
            let edges = AccessionPath::query(
                conn,
                "Select * from accession_paths where accession_id = ?1 order by index_in_path ASC;",
                rusqlite::params!(SQLValue::from(accession.id)),
            );
            let new_path_id = &path_map[&accession.path_id];
            let obj = Accession::create(
                conn,
                &accession.name,
                new_path_id,
                accession.parent_accession_id.as_ref(),
            )
            .expect("Unable to create accession in clone.");
            AccessionPath::create(
                conn,
                &obj.id,
                &edges.iter().map(|ap| ap.edge_id).collect::<Vec<_>>(),
            );
        }
    }

    pub fn get_or_create_sample_block_group(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        group_name: &str,
        parent_sample: Option<&str>,
    ) -> Result<HashId, QueryError> {
        let binding = BlockGroup::query(
            conn,
            "select * from block_groups where collection_name = ?1 AND sample_name = ?2 AND name = ?3",
            params![collection_name, sample_name, group_name],
        );
        let initial_query_result = binding.first();

        if let Some(block_group) = initial_query_result {
            return Ok(block_group.id);
        }

        let block_groups = if let Some(parent_sample_name) = parent_sample {
            // use the base reference group if it exists
            BlockGroup::query(
                conn,
                "select * from block_groups where collection_name = ?1 AND sample_name = ?2 AND name = ?3",
                params![collection_name, parent_sample_name, group_name],
            )
        } else {
            BlockGroup::query(
                conn,
                "select * from block_groups where collection_name = ?1 AND sample_name IS null AND name = ?2",
                params![collection_name, group_name],
            )
        };

        if let Some(parent_block_group) = block_groups.first() {
            let new_block_group =
                BlockGroup::create(conn, collection_name, Some(sample_name), group_name);

            // clone parent blocks/edges/path
            parent_block_group.duplicate(conn, &new_block_group.id);

            Ok(new_block_group.id)
        } else {
            let error_message = if let Some(parent_sample_name) = parent_sample {
                format!(
                    "Block group not found for either new sample ({sample_name}) or parent sample ({parent_sample_name})"
                )
            } else {
                format!(
                    "Block group not found for new sample ({sample_name}) and default parent sample"
                )
            };
            Err(QueryError::ResultsNotFound(error_message))
        }
    }

    pub fn get_id(collection_name: &str, sample_name: Option<&str>, group_name: &str) -> HashId {
        HashId(calculate_hash(&format!(
            "{collection_name}:{sample_name:?}:{group_name}"
        )))
    }

    pub fn get_graph(conn: &GraphConnection, block_group_id: &HashId) -> GenGraph {
        let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id);
        let blocks = Edge::blocks_from_edges(conn, &edges);
        let (graph, _) = Edge::build_graph(&edges, &blocks);
        graph
    }

    pub fn prune_graph(graph: &mut GenGraph) {
        // Prunes a graph by removing edges on the same chromosome_index. This means if 2 edges are
        // both "chromosome index 0", we keep the newer one.
        let mut root_nodes = HashSet::new();
        let mut edges_to_remove: Vec<(GraphNode, GraphNode)> = vec![];
        for node in graph.nodes() {
            if node.node_id == PATH_START_NODE_ID {
                root_nodes.insert(node);
            }
            let mut edges_by_ci: HashMap<i64, (GraphNode, GraphNode, i64)> = HashMap::new();
            for (source_node, target_node, edge_weights) in graph.edges(node) {
                for edge_weight in edge_weights {
                    if edge_weight.chromosome_index == NO_CHROMOSOME_INDEX {
                        continue;
                    }
                    if edge_weight.chromosome_index == PRESERVE_EDIT_SITE_CHROMOSOME_INDEX {
                        edges_to_remove.push((source_node, target_node));
                        continue;
                    }
                    edges_by_ci
                        .entry(edge_weight.chromosome_index)
                        .and_modify(|(source, target, created_on)| {
                            if edge_weight.created_on > *created_on {
                                edges_to_remove.push((*source, *target));
                                *source = source_node;
                                *target = target_node;
                                *created_on = edge_weight.created_on;
                            } else {
                                edges_to_remove.push((source_node, target_node));
                            }
                        })
                        .or_insert((source_node, target_node, edge_weight.created_on));
                }
            }
        }

        for (source, target) in edges_to_remove.iter() {
            graph.remove_edge(*source, *target);
        }

        let reachable_nodes = all_reachable_nodes(&*graph, &Vec::from_iter(root_nodes));
        let mut to_remove = vec![];
        for node in graph.nodes() {
            if !reachable_nodes.contains(&node) {
                to_remove.push(node);
            }
        }
        for node in to_remove {
            graph.remove_node(node);
        }
    }

    pub fn get_all_sequences(
        conn: &GraphConnection,
        block_group_id: &HashId,
        _prune: bool,
    ) -> HashSet<String> {
        let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id)
            .into_iter()
            .filter(|edge| edge.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
            .collect::<Vec<_>>();
        let blocks = Edge::blocks_from_edges(conn, &edges);

        let (mut graph, _) = Edge::build_graph(&edges, &blocks);
        BlockGroup::prune_graph(&mut graph);

        let mut start_nodes = vec![];
        let mut end_nodes = vec![];
        for node in graph.nodes() {
            if is_start_node(node.node_id) {
                start_nodes.push(node);
            } else if is_end_node(node.node_id) {
                end_nodes.push(node);
            }
        }
        let blocks_by_id = blocks
            .clone()
            .into_iter()
            .map(|block| (block.id, block))
            .collect::<HashMap<i64, GroupBlock>>();
        let mut sequences = HashSet::<String>::new();

        for start_node in start_nodes {
            for end_node in &end_nodes {
                // TODO: maybe make all_simple_paths return a single path id where start == end
                if start_node == *end_node {
                    let block = blocks_by_id.get(&start_node.block_id).unwrap();
                    if block.node_id != PATH_START_NODE_ID && block.node_id != PATH_END_NODE_ID {
                        sequences.insert(block.sequence());
                    }
                } else {
                    for path in all_simple_paths(&graph, start_node, *end_node) {
                        let mut current_sequence = "".to_string();
                        for node in path {
                            let block = blocks_by_id.get(&node.block_id).unwrap();
                            let block_sequence = block.sequence();
                            current_sequence.push_str(&block_sequence);
                        }
                        sequences.insert(current_sequence);
                    }
                }
            }
        }

        sequences
    }

    pub fn add_accession(
        conn: &GraphConnection,
        path: &Path,
        name: &str,
        start: i64,
        end: i64,
        cache: &mut PathCache,
    ) -> Accession {
        let tree = PathCache::get_intervaltree(cache, path).unwrap();
        let start_blocks: Vec<&NodeIntervalBlock> =
            tree.query_point(start).map(|x| &x.value).collect();
        assert_eq!(start_blocks.len(), 1);
        let start_block = start_blocks[0];
        let end_blocks: Vec<&NodeIntervalBlock> = tree.query_point(end).map(|x| &x.value).collect();
        assert_eq!(end_blocks.len(), 1);
        let end_block = end_blocks[0];
        // we make a start/end edge for the accession start/end, then fill in the middle
        // with any existing edges
        let start_edge = AccessionEdgeData {
            source_node_id: PATH_START_NODE_ID,
            source_coordinate: -1,
            source_strand: Strand::Forward,
            target_node_id: start_block.node_id,
            target_coordinate: start - start_block.start + start_block.sequence_start,
            target_strand: Strand::Forward,
            chromosome_index: 0,
        };
        let end_edge = AccessionEdgeData {
            source_node_id: end_block.node_id,
            source_coordinate: end - end_block.start + end_block.sequence_start,
            source_strand: Strand::Forward,
            target_node_id: PATH_END_NODE_ID,
            target_coordinate: -1,
            target_strand: Strand::Forward,
            chromosome_index: 0,
        };
        let accession =
            Accession::create(conn, name, &path.id, None).expect("Unable to create accession.");
        let mut path_edges = vec![start_edge];
        if start_block == end_block {
            path_edges.push(end_edge);
        } else {
            let mut in_range = false;
            let path_blocks: Vec<&NodeIntervalBlock> = tree
                .iter_sorted()
                .map(|x| &x.value)
                .filter(|block| {
                    if block.block_id == start_block.block_id {
                        in_range = true;
                    } else if block.block_id == end_block.block_id {
                        in_range = false;
                        return true;
                    }
                    in_range
                })
                .collect::<Vec<_>>();
            // if start and end block are not the same, we will always have at least 2 elements in path_blocks
            for (block, next_block) in path_blocks.iter().zip(path_blocks[1..].iter()) {
                path_edges.push(AccessionEdgeData {
                    source_node_id: block.node_id,
                    source_coordinate: block.sequence_end,
                    source_strand: block.strand,
                    target_node_id: next_block.node_id,
                    target_coordinate: next_block.sequence_start,
                    target_strand: next_block.strand,
                    chromosome_index: 0,
                })
            }
            path_edges.push(end_edge);
        }
        AccessionPath::create(
            conn,
            &accession.id,
            &AccessionEdge::bulk_create(conn, &path_edges),
        );
        accession
    }

    pub fn insert_changes(
        conn: &GraphConnection,
        changes: &[PathChange],
        cache: &mut PathCache,
        modify_blockgroup: bool,
    ) -> Result<(), ChangeError> {
        let mut new_augmented_edges_by_block_group =
            HashMap::<&HashId, Vec<AugmentedEdgeData>>::new();
        let mut new_accession_edges = HashMap::new();
        let mut tree_map = HashMap::new();
        for change in changes {
            let tree = if modify_blockgroup {
                tree_map.entry(change.block_group_id).or_insert_with(|| {
                    BlockGroup::intervaltree_for(conn, &change.block_group_id, true)
                })
            } else {
                PathCache::get_intervaltree(cache, &change.path).unwrap()
            };
            let new_augmented_edges = BlockGroup::set_up_new_edges(change, tree)?;
            new_augmented_edges_by_block_group
                .entry(&change.block_group_id)
                .and_modify(|new_edge_data| new_edge_data.extend(new_augmented_edges.clone()))
                .or_insert_with(|| new_augmented_edges.clone());
            if let Some(accession) = &change.path_accession {
                new_accession_edges
                    .entry((&change.path, accession))
                    .and_modify(|new_edge_data: &mut Vec<AugmentedEdgeData>| {
                        new_edge_data.extend(new_augmented_edges.clone())
                    })
                    .or_insert_with(|| new_augmented_edges.clone());
            }
        }

        let mut edge_data_map = HashMap::new();

        for (block_group_id, new_augmented_edges) in new_augmented_edges_by_block_group {
            let new_edges = new_augmented_edges
                .iter()
                .map(|augmented_edge| augmented_edge.edge_data)
                .collect::<Vec<_>>();
            let edge_ids = Edge::bulk_create(conn, &new_edges);
            for (i, edge_data) in new_edges.iter().enumerate() {
                edge_data_map.insert(*edge_data, edge_ids[i]);
            }
            let new_block_group_edges = edge_ids
                .iter()
                .enumerate()
                .map(|(i, edge_id)| BlockGroupEdgeData {
                    block_group_id: *block_group_id,
                    edge_id: *edge_id,
                    chromosome_index: new_augmented_edges[i].chromosome_index,
                    phased: new_augmented_edges[i].phased,
                })
                .collect::<Vec<_>>();
            BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        }

        for ((path, accession_name), path_edges) in new_accession_edges {
            match Accession::get(
                conn,
                "select * from accessions where name = ?1 AND path_id = ?2",
                params![accession_name, path.id],
            ) {
                Ok(_) => {
                    println!(
                        "accession already exists, consider a better matching algorithm to determine if this is an error."
                    );
                }
                Err(_) => {
                    let acc_edges = AccessionEdge::bulk_create(
                        conn,
                        &path_edges
                            .iter()
                            .map(AccessionEdgeData::from)
                            .collect::<Vec<_>>(),
                    );
                    let acc = Accession::create(conn, accession_name, &path.id, None)
                        .expect("Accession could not be created.");
                    AccessionPath::create(conn, &acc.id, &acc_edges);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::ptr_arg)]
    #[allow(clippy::needless_late_init)]
    pub fn insert_change(
        conn: &GraphConnection,
        change: &PathChange,
        tree: &IntervalTree<i64, NodeIntervalBlock>,
    ) -> Result<(), ChangeError> {
        let new_augmented_edges = BlockGroup::set_up_new_edges(change, tree)?;
        let new_edges = new_augmented_edges
            .iter()
            .map(|augmented_edge| augmented_edge.edge_data)
            .collect::<Vec<_>>();
        let edge_ids = Edge::bulk_create(conn, &new_edges);
        let new_block_group_edges = edge_ids
            .iter()
            .enumerate()
            .map(|(i, edge_id)| BlockGroupEdgeData {
                block_group_id: change.block_group_id,
                edge_id: *edge_id,
                chromosome_index: new_augmented_edges[i].chromosome_index,
                phased: new_augmented_edges[i].phased,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        Ok(())
    }

    fn set_up_new_edges(
        change: &PathChange,
        tree: &IntervalTree<i64, NodeIntervalBlock>,
    ) -> Result<Vec<AugmentedEdgeData>, ChangeError> {
        let start_blocks: Vec<&NodeIntervalBlock> =
            tree.query_point(change.start).map(|x| &x.value).collect();
        assert_eq!(start_blocks.len(), 1);
        // NOTE: This may not be used but needs to be initialized here instead of inside the if
        // statement that uses it, so that the borrow checker is happy
        let previous_start_blocks: Vec<&NodeIntervalBlock> = tree
            .query_point(change.start - 1)
            .map(|x| &x.value)
            .collect();
        assert_eq!(previous_start_blocks.len(), 1);
        let start_block = if start_blocks[0].start == change.start {
            // First part of this block will be replaced/deleted, need to get previous block to add
            // edge including it
            previous_start_blocks[0]
        } else {
            start_blocks[0]
        };

        // Ensure the change is within the path bounds. The logic here is a bit backwards, where
        // we check if the start is before the start block's end and the end is before the end
        // block's start. This is because the terminal blocks start and end at the bounds of the
        // interval tree. So while it's ok to have a start/end block be the start/end block (for
        // changes at the extremes, it's not ok for the change to start beyond the current
        // boundaries.
        if is_start_node(start_block.node_id) && change.start < start_block.end {
            return Err(ChangeError::OutOfBounds(format!(
                "Invalid change specified. Coordinate {pos} is before start of path range ({path_pos}).",
                pos = change.start,
                path_pos = start_block.end
            )));
        }

        let end_blocks: Vec<&NodeIntervalBlock> =
            tree.query_point(change.end).map(|x| &x.value).collect();
        assert_eq!(end_blocks.len(), 1);
        let end_block = end_blocks[0];

        if is_end_node(end_block.node_id) && change.end > end_block.start {
            return Err(ChangeError::OutOfBounds(format!(
                "Invalid change specified. Coordinate {pos} is before start of path range ({path_pos}).",
                pos = change.end,
                path_pos = end_block.start
            )));
        }

        let mut new_edges = vec![];

        if change.block.sequence_start == change.block.sequence_end {
            // Deletion
            let source_coordinate = change.start - start_block.start + start_block.sequence_start;
            let target_coordinate = change.end - end_block.start + end_block.sequence_start;
            let mut aug_edges = vec![];
            let new_edge = EdgeData {
                source_node_id: start_block.node_id,
                source_coordinate,
                source_strand: Strand::Forward,
                target_node_id: end_block.node_id,
                target_coordinate,
                target_strand: Strand::Forward,
            };
            aug_edges.push(AugmentedEdgeData {
                edge_data: new_edge,
                chromosome_index: change.chromosome_index,
                phased: change.phased,
            });

            // NOTE: If the deletion is happening at the very beginning of a path, we need to add
            // an edge from the dedicated start node to the end of the deletion, to indicate it's
            // another start point in the block group DAG.
            if change.start == 0 {
                let target_coordinate = change.end - end_block.start + end_block.sequence_start;
                let new_beginning_edge = EdgeData {
                    source_node_id: PATH_START_NODE_ID,
                    source_coordinate: 0,
                    source_strand: Strand::Forward,
                    target_node_id: end_block.node_id,
                    target_coordinate,
                    target_strand: Strand::Forward,
                };
                aug_edges.push(AugmentedEdgeData {
                    edge_data: new_beginning_edge,
                    chromosome_index: change.chromosome_index,
                    phased: change.phased,
                });
                if !is_terminal(end_block.node_id) {
                    new_edges.push(AugmentedEdgeData {
                        edge_data: EdgeData {
                            source_node_id: end_block.node_id,
                            source_coordinate: target_coordinate,
                            source_strand: Strand::Forward,
                            target_node_id: end_block.node_id,
                            target_coordinate,
                            target_strand: Strand::Forward,
                        },
                        chromosome_index: if change.preserve_edge {
                            0
                        } else {
                            PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                        },
                        phased: 0,
                    });
                }
            } else {
                if !is_terminal(start_block.node_id) {
                    new_edges.push(AugmentedEdgeData {
                        edge_data: EdgeData {
                            source_node_id: start_block.node_id,
                            source_coordinate,
                            source_strand: Strand::Forward,
                            target_node_id: start_block.node_id,
                            target_coordinate: source_coordinate,
                            target_strand: Strand::Forward,
                        },
                        chromosome_index: if change.preserve_edge {
                            0
                        } else {
                            PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                        },
                        phased: 0,
                    });
                };
                if !is_terminal(end_block.node_id) {
                    new_edges.push(AugmentedEdgeData {
                        edge_data: EdgeData {
                            source_node_id: end_block.node_id,
                            source_coordinate: target_coordinate,
                            source_strand: Strand::Forward,
                            target_node_id: end_block.node_id,
                            target_coordinate,
                            target_strand: Strand::Forward,
                        },
                        chromosome_index: if change.preserve_edge {
                            0
                        } else {
                            PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                        },
                        phased: 0,
                    });
                }
            }
            new_edges.extend(aug_edges);
        // NOTE: If the deletion is happening at the very end of a path, we might add an edge
        // from the beginning of the deletion to the dedicated end node, but in practice it
        // doesn't affect sequence readouts, so it may not be worth it.
        } else {
            // Insertion/replacement
            let insertion_start_coordinate =
                change.start - start_block.start + start_block.sequence_start;
            let new_start_edge = EdgeData {
                source_node_id: start_block.node_id,
                source_coordinate: insertion_start_coordinate,
                source_strand: Strand::Forward,
                target_node_id: change.block.node_id,
                target_coordinate: change.block.sequence_start,
                target_strand: Strand::Forward,
            };
            let new_augmented_start_edge = AugmentedEdgeData {
                edge_data: new_start_edge,
                chromosome_index: change.chromosome_index,
                phased: change.phased,
            };
            let insertion_end_coordinate = change.end - end_block.start + end_block.sequence_start;
            let new_end_edge = EdgeData {
                source_node_id: change.block.node_id,
                source_coordinate: change.block.sequence_end,
                source_strand: Strand::Forward,
                target_node_id: end_block.node_id,
                target_coordinate: insertion_end_coordinate,
                target_strand: Strand::Forward,
            };
            let new_augmented_end_edge = AugmentedEdgeData {
                edge_data: new_end_edge,
                chromosome_index: change.chromosome_index,
                phased: change.phased,
            };

            if change.start == 0 {
                new_edges.push(AugmentedEdgeData {
                    edge_data: EdgeData {
                        source_node_id: PATH_START_NODE_ID,
                        source_coordinate: 0,
                        source_strand: Strand::Forward,
                        target_node_id: change.block.node_id,
                        target_coordinate: change.block.sequence_start,
                        target_strand: Strand::Forward,
                    },
                    chromosome_index: change.chromosome_index,
                    phased: 0,
                });
            }

            if !is_terminal(start_block.node_id) {
                new_edges.push(AugmentedEdgeData {
                    edge_data: EdgeData {
                        source_node_id: start_block.node_id,
                        source_coordinate: insertion_start_coordinate,
                        source_strand: Strand::Forward,
                        target_node_id: start_block.node_id,
                        target_coordinate: insertion_start_coordinate,
                        target_strand: Strand::Forward,
                    },
                    chromosome_index: if change.preserve_edge {
                        0
                    } else {
                        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                    },
                    phased: 0,
                });
            }
            if !is_terminal(end_block.node_id) {
                new_edges.push(AugmentedEdgeData {
                    edge_data: EdgeData {
                        source_node_id: end_block.node_id,
                        source_coordinate: insertion_end_coordinate,
                        source_strand: Strand::Forward,
                        target_node_id: end_block.node_id,
                        target_coordinate: insertion_end_coordinate,
                        target_strand: Strand::Forward,
                    },
                    chromosome_index: if change.preserve_edge {
                        0
                    } else {
                        PRESERVE_EDIT_SITE_CHROMOSOME_INDEX
                    },
                    phased: 0,
                });
            }

            new_edges.push(new_augmented_start_edge);
            new_edges.push(new_augmented_end_edge);
        }

        Ok(new_edges)
    }

    pub fn intervaltree_for(
        conn: &GraphConnection,
        block_group_id: &HashId,
        remove_ambiguous_positions: bool,
    ) -> IntervalTree<i64, NodeIntervalBlock> {
        // make a tree where every node has a span in the graph.
        let mut graph = BlockGroup::get_graph(conn, block_group_id);
        BlockGroup::prune_graph(&mut graph);
        flatten_to_interval_tree(&graph, remove_ambiguous_positions)
    }

    pub fn get_current_path(conn: &GraphConnection, block_group_id: &HashId) -> Path {
        let paths = Path::query(
            conn,
            "SELECT * FROM paths WHERE block_group_id = ?1 ORDER BY created_on DESC",
            params![block_group_id],
        );
        paths[0].clone()
    }

    pub fn get_path_by_name(
        conn: &GraphConnection,
        block_group_id: &HashId,
        path_name: &str,
    ) -> Option<Path> {
        let paths = Path::query(
            conn,
            "SELECT * FROM paths WHERE block_group_id = ?1 ORDER BY created_on DESC",
            params![block_group_id],
        );

        for path in &paths {
            if path.name == path_name {
                return Some(path.clone());
            }
        }

        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn derive_subgraph(
        conn: &GraphConnection,
        collection_name: &str,
        child_sample_name: &str,
        source_block_group_id: &HashId,
        start_block: &NodeIntervalBlock,
        end_block: &NodeIntervalBlock,
        start_node_coordinate: i64,
        end_node_coordinate: i64,
        target_block_group_id: &HashId,
        create_terminal_edges: bool,
    ) -> HashMap<HashId, HashId> {
        let current_graph = BlockGroup::get_graph(conn, source_block_group_id);
        let start_node = current_graph
            .nodes()
            .find(|node| {
                node.node_id == start_block.node_id
                    && node.sequence_start <= start_node_coordinate
                    && node.sequence_end >= start_node_coordinate
            })
            .unwrap();
        let end_node = current_graph
            .nodes()
            .find(|node| {
                node.node_id == end_block.node_id
                    && node.sequence_start <= end_node_coordinate
                    && node.sequence_end >= end_node_coordinate
            })
            .unwrap();
        let subgraph_edges = all_intermediate_edges(&current_graph, start_node, end_node);

        // Filter out internal edges (boundary edges) that don't exist in the database
        let subgraph_edge_ids = subgraph_edges
            .iter()
            .map(|(_to, _from, edge_info)| edge_info[0].edge_id)
            .collect::<Vec<_>>();
        let source_edges = Edge::query_by_ids(conn, &subgraph_edge_ids);

        // Instead of reusing the existing edges, we create copies of the nodes and edges.  This
        // makes it easier to recombine subgraphs later.
        // TODO: Annotation support. We do not reuse node ids because nodes control the graph
        // topology. If we duplicated nodes via these operations, it would leave to unintentional
        // cycles. A node cannot exist in 2 places. This will be non-intuitive for annotation
        // propagation and needs to be handled.
        let mut old_node_ids = HashSet::new();
        for source_edge in &source_edges {
            old_node_ids.insert(source_edge.source_node_id);
            old_node_ids.insert(source_edge.target_node_id);
        }

        old_node_ids.insert(start_block.node_id);
        old_node_ids.insert(end_block.node_id);

        let old_nodes = Node::query_by_ids(conn, &old_node_ids);
        let old_nodes_by_id = old_nodes
            .iter()
            .map(|node| (node.id, node))
            .collect::<HashMap<_, _>>();

        let mut new_node_ids_by_old = HashMap::new();
        let old_start_node = old_nodes_by_id.get(&start_block.node_id).unwrap();
        let new_start_node_hash = HashId(calculate_hash(&format!(
            "{}.{}.{}.bg-{}",
            collection_name, child_sample_name, old_start_node.id, target_block_group_id
        )));
        new_node_ids_by_old.insert(
            old_start_node.id,
            Node::create(conn, &old_start_node.sequence_hash, &new_start_node_hash),
        );
        let old_end_node = old_nodes_by_id.get(&end_block.node_id).unwrap();
        let new_end_node_hash = HashId(calculate_hash(&format!(
            "{}.{}.{}.bg-{}",
            collection_name, child_sample_name, old_end_node.id, target_block_group_id
        )));
        new_node_ids_by_old.insert(
            old_end_node.id,
            Node::create(conn, &old_end_node.sequence_hash, &new_end_node_hash),
        );

        let mut new_edges = vec![];
        for source_edge in &source_edges {
            let new_source_node_id = *new_node_ids_by_old
                .entry(source_edge.source_node_id)
                .or_insert_with(|| {
                    let old_source_node = old_nodes_by_id.get(&source_edge.source_node_id).unwrap();
                    let new_source_node_hash = HashId(calculate_hash(&format!(
                        "{}.{}.{}.bg-{}",
                        collection_name,
                        child_sample_name,
                        old_source_node.id,
                        target_block_group_id
                    )));
                    Node::create(conn, &old_source_node.sequence_hash, &new_source_node_hash)
                });
            let new_target_node_id = *new_node_ids_by_old
                .entry(source_edge.target_node_id)
                .or_insert_with(|| {
                    let old_target_node = old_nodes_by_id.get(&source_edge.target_node_id).unwrap();
                    let new_target_node_hash = HashId(calculate_hash(&format!(
                        "{}.{}.{}.bg-{}",
                        collection_name,
                        child_sample_name,
                        old_target_node.id,
                        target_block_group_id
                    )));
                    Node::create(conn, &old_target_node.sequence_hash, &new_target_node_hash)
                });

            new_edges.push(EdgeData {
                source_node_id: new_source_node_id,
                source_coordinate: source_edge.source_coordinate,
                source_strand: source_edge.source_strand,
                target_node_id: new_target_node_id,
                target_coordinate: source_edge.target_coordinate,
                target_strand: source_edge.target_strand,
            });
        }

        let new_edge_ids = Edge::bulk_create(conn, &new_edges);
        let new_edge_ids_by_old = source_edges
            .iter()
            .zip(new_edge_ids.iter())
            .map(|(source_edge, new_edge_id)| (&source_edge.id, new_edge_id))
            .collect::<HashMap<_, _>>();

        let source_block_group_edges = BlockGroupEdge::specific_edges_for_block_group(
            conn,
            source_block_group_id,
            &subgraph_edge_ids,
        );
        let source_edge_ids = source_edges
            .iter()
            .map(|edge| &edge.id)
            .collect::<HashSet<_>>();
        let source_block_group_edges = source_block_group_edges
            .iter()
            .filter(|block_group_edge| source_edge_ids.contains(&block_group_edge.edge.id))
            .collect::<Vec<_>>();
        let source_block_group_edges_by_edge_id = source_block_group_edges
            .iter()
            .map(|block_group_edge| (&block_group_edge.edge.id, block_group_edge))
            .collect::<HashMap<_, _>>();

        let subgraph_edge_inputs = source_block_group_edges
            .iter()
            .map(|edge| {
                let block_group_edge = source_block_group_edges_by_edge_id
                    .get(&edge.edge.id)
                    .unwrap();
                let new_edge_id = new_edge_ids_by_old.get(&edge.edge.id).unwrap();
                BlockGroupEdgeData {
                    block_group_id: *target_block_group_id,
                    edge_id: **new_edge_id,
                    chromosome_index: block_group_edge.chromosome_index,
                    phased: block_group_edge.phased,
                }
            })
            .collect::<Vec<_>>();

        let mut all_edges = subgraph_edge_inputs.clone();

        if create_terminal_edges {
            let new_start_node_id = new_node_ids_by_old.get(&start_block.node_id).unwrap();
            let new_start_edge = Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                *new_start_node_id,
                start_node_coordinate,
                start_block.strand,
            );
            let new_start_edge_data = BlockGroupEdgeData {
                block_group_id: *target_block_group_id,
                edge_id: new_start_edge.id,
                chromosome_index: 0,
                phased: 0,
            };
            let new_end_node_id = new_node_ids_by_old.get(&end_block.node_id).unwrap();
            let new_end_edge = Edge::create(
                conn,
                *new_end_node_id,
                end_node_coordinate,
                end_block.strand,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            );
            let new_end_edge_data = BlockGroupEdgeData {
                block_group_id: *target_block_group_id,
                edge_id: new_end_edge.id,
                chromosome_index: 0,
                phased: 0,
            };

            all_edges.push(new_start_edge_data);
            all_edges.push(new_end_edge_data);
        }

        BlockGroupEdge::bulk_create(conn, &all_edges);

        new_node_ids_by_old
    }
}

impl Query for BlockGroup {
    type Model = BlockGroup;

    const TABLE_NAME: &'static str = "block_groups";

    fn process_row(row: &Row) -> Self::Model {
        BlockGroup {
            id: row.get(0).unwrap(),
            collection_name: row.get(1).unwrap(),
            sample_name: row.get(2).unwrap(),
            name: row.get(3).unwrap(),
            created_on: row.get(4).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Range;

    use capnp::message::TypedBuilder;
    use chrono::Utc;
    use gen_core::NO_CHROMOSOME_INDEX;

    use super::*;
    use crate::{
        collection::Collection,
        node::Node,
        sample::Sample,
        sequence::Sequence,
        test_helpers::{get_connection, interval_tree_verify, setup_block_group},
    };

    #[test]
    fn test_capnp_serialization() {
        let block_group = BlockGroup {
            id: HashId::pad_str(42),
            collection_name: "test_collection".to_string(),
            sample_name: Some("test_sample".to_string()),
            name: "test_block_group".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
        };

        let mut message = TypedBuilder::<block_group::Owned>::new_default();
        let mut root = message.init_root();
        block_group.write_capnp(&mut root);

        let deserialized = BlockGroup::read_capnp(root.into_reader());
        assert_eq!(block_group, deserialized);
    }

    #[test]
    fn test_capnp_serialization_no_sample() {
        let block_group = BlockGroup {
            id: HashId::pad_str(43),
            collection_name: "test_collection_2".to_string(),
            sample_name: None,
            name: "test_block_group_2".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
        };

        let mut message = TypedBuilder::<block_group::Owned>::new_default();
        let mut root = message.init_root();
        block_group.write_capnp(&mut root);

        let deserialized = BlockGroup::read_capnp(root.into_reader());
        assert_eq!(block_group, deserialized);
    }

    #[test]
    fn test_blockgroup_create() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test");
        let bg1 = BlockGroup::create(conn, "test", None, "hg19");
        assert_eq!(bg1.collection_name, "test");
        assert_eq!(bg1.name, "hg19");
        Sample::get_or_create(conn, "sample");
        let bg2 = BlockGroup::create(conn, "test", Some("sample"), "hg19");
        assert_eq!(bg2.collection_name, "test");
        assert_eq!(bg2.name, "hg19");
        assert_eq!(bg2.sample_name, Some("sample".to_string()));
        assert_ne!(&bg1.id, &bg2.id);
    }

    #[test]
    fn test_blockgroup_delete_without_sample() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test");
        let bg1 = BlockGroup::create(conn, "test", None, "hg19");
        let bg2 = BlockGroup::create(conn, "test", None, "hg38");

        BlockGroup::delete(conn, "test", None, &bg1.name);

        let bgs = BlockGroup::all(conn);
        assert_eq!(bgs.len(), 1);
        assert_eq!(bgs[0], bg2);
    }

    #[test]
    fn test_blockgroup_delete_with_sample() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test");
        Sample::get_or_create(conn, "sample1");
        Sample::get_or_create(conn, "sample2");
        let bg1 = BlockGroup::create(conn, "test", Some("sample1"), "hg19");
        let bg2 = BlockGroup::create(conn, "test", Some("sample2"), "hg19");

        BlockGroup::delete(conn, "test", bg1.sample_name.as_deref(), &bg1.name);

        let bgs = BlockGroup::all(conn);
        assert_eq!(bgs.len(), 1);
        assert_eq!(bgs[0], bg2);
    }

    #[test]
    fn test_blockgroup_clone() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test");
        let bg1 = BlockGroup::create(conn, "test", None, "hg19");
        assert_eq!(bg1.collection_name, "test");
        assert_eq!(bg1.name, "hg19");
        Sample::get_or_create(conn, "sample");
        let bg2 =
            BlockGroup::get_or_create_sample_block_group(conn, "test", "sample", "hg19", None)
                .unwrap();
        assert_eq!(
            BlockGroupEdge::edges_for_block_group(conn, &bg1.id),
            BlockGroupEdge::edges_for_block_group(conn, &bg2)
        );
    }

    #[test]
    fn test_blockgroup_clone_passes_accessions() {
        let conn = &get_connection(None).unwrap();
        let (bg_1, path) = setup_block_group(conn);
        let mut path_cache = PathCache::new(conn);
        PathCache::lookup(&mut path_cache, &bg_1, path.name.clone());
        let acc_1 = BlockGroup::add_accession(conn, &path, "test", 3, 7, &mut path_cache);
        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                rusqlite::params!(SQLValue::from("test".to_string())),
            ),
            vec![Accession {
                id: acc_1.id,
                name: "test".to_string(),
                path_id: path.id,
                parent_accession_id: None,
            }]
        );

        Sample::get_or_create(conn, "sample2");
        let _bg2 =
            BlockGroup::get_or_create_sample_block_group(conn, "test", "sample2", "chr1", None)
                .unwrap();
        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                rusqlite::params!(SQLValue::from("test".to_string())),
            )
            .len(),
            2
        );
    }

    #[test]
    fn insert_and_deletion_get_all() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 7,
            end: 15,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );

        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn);
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("2"));
        let deletion = PathBlock {
            id: 0,
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 19,
            path_end: 31,
            strand: Strand::Forward,
        };

        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 19,
            end: 31,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        // take out an entire block.
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTGGGGGGGGG".to_string(),
            ])
        )
    }

    #[test]
    fn simple_insert_get_all() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 7,
            end: 15,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_on_block_boundary_middle() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 15,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 15,
            end: 15,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTNNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_within_block() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 12,
            path_end: 17,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 12,
            end: 17,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTNNNNTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_on_block_boundary_start() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 10,
            path_end: 10,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 10,
            end: 10,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAANNNNTTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_on_block_boundary_end() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 9,
            path_end: 9,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 9,
            end: 9,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAANNNNATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_across_entire_block_boundary() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 10,
            path_end: 20,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 10,
            end: 20,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAANNNNCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_across_two_blocks() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 15,
            path_end: 25,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 15,
            end: 25,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTNNNNCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_spanning_blocks() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 5,
            path_end: 35,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 5,
            end: 35,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAANNNNGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn simple_deletion() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn);
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1"));
        let deletion = PathBlock {
            id: 0,
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 19,
            path_end: 31,
            strand: Strand::Forward,
        };

        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 19,
            end: 31,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };

        // take out an entire block.
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTGGGGGGGGG".to_string(),
            ])
        )
    }

    #[test]
    fn doesnt_apply_same_insert_twice() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 7,
            end: 15,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );

        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    fn insert_at_beginning_of_path() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 0,
            path_end: 0,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 0,
            end: 0,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "NNNNAAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn homozygous_insert_at_beginning_of_path() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 0,
            path_end: 0,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 0,
            end: 0,
            block: insert,
            chromosome_index: 0,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "NNNNAAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn insert_at_end_of_path() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 40,
            path_end: 40,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 40,
            end: 40,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGGNNNN".to_string(),
            ])
        );
    }

    #[test]
    fn insert_at_one_bp_into_block() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 10,
            path_end: 11,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 10,
            end: 11,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn insert_at_one_bp_from_end_of_block() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn);
        let insert_node_id = Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 19,
            path_end: 20,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 19,
            end: 20,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTNNNNCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn delete_at_beginning_of_path() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn);
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1"));
        let deletion = PathBlock {
            id: 0,
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 0,
            path_end: 1,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 0,
            end: 1,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn delete_at_end_of_path() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn);
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1"));
        let deletion = PathBlock {
            id: 0,
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 35,
            path_end: 40,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 35,
            end: 40,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn error_on_out_of_bounds_change() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn);
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1"));
        let deletion = PathBlock {
            id: 0,
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 350,
            path_end: 400,
            strand: Strand::Forward,
        };
        let after_end_change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 350,
            end: 400,
            block: deletion.clone(),
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let before_start_change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: -300,
            end: 400,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        let res = BlockGroup::insert_change(&conn, &after_end_change, &tree);
        assert!(matches!(res, Err(ChangeError::OutOfBounds(_))));
        let res = BlockGroup::insert_change(&conn, &before_start_change, &tree);
        assert!(matches!(res, Err(ChangeError::OutOfBounds(_))));
    }

    #[test]
    fn deletion_starting_at_block_boundary() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn);
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1"));
        let deletion = PathBlock {
            id: 0,
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 10,
            path_end: 12,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 10,
            end: 12,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn deletion_ending_at_block_boundary() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn);
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1"));
        let deletion = PathBlock {
            id: 0,
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 18,
            path_end: 20,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id,
            path: path.clone(),
            path_accession: None,
            start: 18,
            end: 20,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(&conn);
        BlockGroup::insert_change(&conn, &change, &tree).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn test_blockgroup_interval_tree() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let _new_sample = Sample::get_or_create(conn, "child");
        let new_bg_id =
            BlockGroup::get_or_create_sample_block_group(conn, "test", "child", "chr1", None)
                .unwrap();
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![new_bg_id],
        );
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn);
        let insert_node_id = Node::create(
            conn,
            &insert_sequence.hash,
            &HashId::convert_str("insert-node"),
        );
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id: new_bg_id,
            path: new_path[0].clone(),
            path_accession: None,
            start: 7,
            end: 15,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let tree = path.intervaltree(conn);
        BlockGroup::insert_change(conn, &change, &tree).unwrap();

        let tree = BlockGroup::intervaltree_for(conn, &block_group_id, false);
        let tree2 = BlockGroup::intervaltree_for(conn, &block_group_id, true);
        interval_tree_verify(
            &tree,
            3,
            &[NodeIntervalBlock {
                block_id: 3,
                node_id: HashId::convert_str("test-a-node"),
                start: 0,
                end: 10,
                sequence_start: 0,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree2,
            3,
            &[NodeIntervalBlock {
                block_id: 3,
                node_id: HashId::convert_str("test-a-node"),
                start: 0,
                end: 10,
                sequence_start: 0,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree,
            35,
            &[NodeIntervalBlock {
                block_id: 0,
                node_id: HashId::convert_str("test-g-node"),
                start: 30,
                end: 40,
                sequence_start: 0,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree2,
            35,
            &[NodeIntervalBlock {
                block_id: 0,
                node_id: HashId::convert_str("test-g-node"),
                start: 30,
                end: 40,
                sequence_start: 0,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );

        // This blockgroup has a change from positions 7-15 of 4 base pairs -- so any changes after this will be ambiguous
        let tree = BlockGroup::intervaltree_for(conn, &new_bg_id, false);
        let tree2 = BlockGroup::intervaltree_for(conn, &new_bg_id, true);
        interval_tree_verify(
            &tree,
            3,
            &[NodeIntervalBlock {
                block_id: 5,
                node_id: HashId::convert_str("test-a-node"),
                start: 0,
                end: 7,
                sequence_start: 0,
                sequence_end: 7,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree2,
            3,
            &[NodeIntervalBlock {
                block_id: 5,
                node_id: HashId::convert_str("test-a-node"),
                start: 0,
                end: 7,
                sequence_start: 0,
                sequence_end: 7,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree,
            30,
            &[
                NodeIntervalBlock {
                    block_id: 0,
                    node_id: HashId::convert_str("test-g-node"),
                    start: 26,
                    end: 36,
                    sequence_start: 0,
                    sequence_end: 10,
                    strand: Strand::Forward,
                },
                NodeIntervalBlock {
                    block_id: 0,
                    node_id: HashId::convert_str("test-g-node"),
                    start: 30,
                    end: 40,
                    sequence_start: 0,
                    sequence_end: 10,
                    strand: Strand::Forward,
                },
            ],
        );
        interval_tree_verify(&tree2, 30, &[]);
        // TODO: This case should return [] because there are 2 distinct nodes at this position and thus it is ambiguous.
        // currently, the caller needs to filter these out.
        interval_tree_verify(
            &tree2,
            9,
            &[
                NodeIntervalBlock {
                    block_id: 4,
                    node_id: HashId::convert_str("insert-node"),
                    start: 7,
                    end: 11,
                    sequence_start: 0,
                    sequence_end: 4,
                    strand: Strand::Forward,
                },
                NodeIntervalBlock {
                    block_id: 6,
                    node_id: HashId::convert_str("test-a-node"),
                    start: 7,
                    end: 10,
                    sequence_start: 7,
                    sequence_end: 10,
                    strand: Strand::Forward,
                },
            ],
        );
    }

    #[test]
    fn test_changes_against_derivative_blockgroups() {
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, _path) = setup_block_group(conn);
        let _new_sample = Sample::get_or_create(conn, "child");
        let new_bg_id =
            BlockGroup::get_or_create_sample_block_group(conn, "test", "child", "chr1", None)
                .unwrap();
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(new_bg_id)),
        );
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn);
        let insert_node_id = Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id: new_bg_id,
            path: new_path[0].clone(),
            path_accession: None,
            start: 7,
            end: 15,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: false,
        };
        // note we are making our change against the new blockgroup, and not the parent blockgroup
        let tree = BlockGroup::intervaltree_for(conn, &new_bg_id, true);
        BlockGroup::insert_change(conn, &change, &tree).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_bg_id, true);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),])
        );

        // Now, we make a change against another descendant
        let _new_sample = Sample::get_or_create(conn, "grandchild");
        let gc_bg_id = BlockGroup::get_or_create_sample_block_group(
            conn,
            "test",
            "grandchild",
            "chr1",
            Some("child"),
        )
        .unwrap();
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(gc_bg_id)),
        );

        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id: gc_bg_id,
            path: new_path[0].clone(),
            path_accession: None,
            start: 7,
            end: 15,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: false,
        };
        // take out an entire block.
        let tree = BlockGroup::intervaltree_for(conn, &gc_bg_id, true);
        BlockGroup::insert_change(conn, &change, &tree).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &gc_bg_id, true);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAAAAAANNNNTCCCCCCCCCCGGGGGGGGGG".to_string(),])
        )
    }

    #[test]
    fn test_changes_against_derivative_diploid_blockgroups() {
        // This test ensures that if we have heterozygous changes that do not introduce frameshifts,
        // we can modify regions downstream of them.
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, _path) = setup_block_group(conn);
        let _new_sample = Sample::get_or_create(conn, "child");
        let new_bg_id =
            BlockGroup::get_or_create_sample_block_group(conn, "test", "child", "chr1", None)
                .unwrap();
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![new_bg_id],
        );
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn);
        let insert_node_id = Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 11,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id: new_bg_id,
            path: new_path[0].clone(),
            path_accession: None,
            start: 7,
            end: 11,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        // note we are making our change against the new blockgroup, and not the parent blockgroup
        let tree = BlockGroup::intervaltree_for(conn, &new_bg_id, true);
        BlockGroup::insert_change(conn, &change, &tree).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_bg_id, true);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );

        // Now, we make a change against another descendant
        let _new_sample = Sample::get_or_create(conn, "grandchild");
        let gc_bg_id = BlockGroup::get_or_create_sample_block_group(
            conn,
            "test",
            "grandchild",
            "chr1",
            Some("child"),
        )
        .unwrap();
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![gc_bg_id],
        );

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn);
        let insert_node_id = Node::create(
            conn,
            &insert_sequence.hash,
            &HashId::convert_str("new-hash"),
        );

        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 20,
            path_end: 24,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id: gc_bg_id,
            path: new_path[0].clone(),
            path_accession: None,
            start: 20,
            end: 24,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        // take out an entire block.
        let tree = BlockGroup::intervaltree_for(conn, &gc_bg_id, true);
        BlockGroup::insert_change(conn, &change, &tree).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &gc_bg_id, true);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTNNNNCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTTTTTNNNNCCCCCCGGGGGGGGGG".to_string()
            ])
        )
    }

    #[test]
    #[should_panic]
    fn test_prohibits_out_of_frame_changes_against_derivative_diploid_blockgroups() {
        // This test ensures that we do not allow ambiguous changes by coordinates
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, _path) = setup_block_group(conn);
        let _new_sample = Sample::get_or_create(conn, "child");
        let new_bg_id =
            BlockGroup::get_or_create_sample_block_group(conn, "test", "child", "chr1", None)
                .unwrap();
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(new_bg_id)),
        );
        // This is a heterozygous replacement of 5 bases with 4 bases, so positions
        // downstream of this are not addressable.
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn);
        let insert_node_id = Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1"));
        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 12,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id: new_bg_id,
            path: new_path[0].clone(),
            path_accession: None,
            start: 7,
            end: 12,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        // note we are making our change against the new blockgroup, and not the parent blockgroup
        let tree = BlockGroup::intervaltree_for(conn, &new_bg_id, true);
        BlockGroup::insert_change(conn, &change, &tree).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_bg_id, true);
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAANNNNTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );

        // Now, we make a change against another descendant and get an error
        let _new_sample = Sample::get_or_create(conn, "grandchild");
        let gc_bg_id = BlockGroup::get_or_create_sample_block_group(
            conn,
            "test",
            "grandchild",
            "chr1",
            Some("child"),
        )
        .unwrap();
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(gc_bg_id)),
        );

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn);
        let insert_node_id =
            Node::create(conn, &insert_sequence.hash, &HashId::pad_str("new-hash"));

        let insert = PathBlock {
            id: 0,
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).to_string(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 20,
            path_end: 24,
            strand: Strand::Forward,
        };
        let change = PathChange {
            block_group_id: gc_bg_id,
            path: new_path[0].clone(),
            path_accession: None,
            start: 20,
            end: 24,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        // take out an entire block.
        let tree = BlockGroup::intervaltree_for(conn, &gc_bg_id, true);
        BlockGroup::insert_change(conn, &change, &tree).unwrap();
    }

    mod test_derive_subgraph {

        use super::*;
        use crate::{
            collection::Collection,
            node::Node,
            sequence::Sequence,
            test_helpers::{get_connection, setup_block_group},
        };

        #[test]
        fn test_derive_subgraph_one_insertion() {
            /*
            AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -> GGGGGGGGGG
                              \-> AAAAAAAA ->/
            Subgraph range:  |-----------------|
            Sequences of the subgraph are TAAAAAAAAC, TTTTTCCCCC
             */
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test");
            let (block_group1_id, original_path) = setup_block_group(conn);

            let intervaltree = original_path.intervaltree(conn);
            let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
            let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

            let insert_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAAAAA")
                .save(conn);
            let insert_node_id = Node::create(
                conn,
                &insert_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-a-node.{}",
                    insert_sequence.hash
                ))),
            );
            let edge_into_insert = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_node_id,
                0,
                Strand::Forward,
            );
            let edge_out_of_insert = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            );
            let ref_heal_1 = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_start_node_id,
                6,
                Strand::Forward,
            );
            let ref_heal_2 = Edge::create(
                conn,
                insert_end_node_id,
                4,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            );

            let edge_ids = [
                &edge_into_insert.id,
                &edge_out_of_insert.id,
                &ref_heal_1.id,
                &ref_heal_2.id,
            ];
            let block_group_edges = edge_ids
                .iter()
                .enumerate()
                .map(|(i, edge_id)| BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: *(*edge_id),
                    chromosome_index: if i < 2 { 1 } else { 0 },
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>();
            BlockGroupEdge::bulk_create(conn, &block_group_edges);

            let insert_path =
                original_path.new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert);
            assert_eq!(
                insert_path.sequence(conn),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
            );

            let all_sequences = BlockGroup::get_all_sequences(conn, &block_group1_id, false);
            assert_eq!(
                all_sequences,
                HashSet::from_iter(vec![
                    "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
                ])
            );

            let mut blocks = intervaltree
                .query(Range { start: 15, end: 25 })
                .map(|x| x.value)
                .collect::<Vec<_>>();
            blocks.sort_by(|a, b| a.start.cmp(&b.start));
            let start_block = blocks[0];
            let start_node_coordinate = 15 - start_block.start + start_block.sequence_start;
            let end_block = blocks[blocks.len() - 1];
            let end_node_coordinate = 25 - end_block.start + end_block.sequence_start;

            let block_group2 = BlockGroup::create(conn, "test", None, "chr1.1");
            BlockGroup::derive_subgraph(
                conn,
                "test",
                "test",
                &block_group1_id,
                &start_block,
                &end_block,
                start_node_coordinate,
                end_node_coordinate,
                &block_group2.id,
                true,
            );
            let all_sequences2 = BlockGroup::get_all_sequences(conn, &block_group2.id, false);
            assert_eq!(
                all_sequences2,
                HashSet::from_iter(vec!["TTTTTCCCCC".to_string(), "TAAAAAAAAC".to_string(),])
            );
        }

        #[test]
        fn test_derive_subgraph_two_independent_insertions() {
            /*
            AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -----> GGGGGGGGGG
                                  \-> AAAAAAAA ->/  \->TTTTTTTT -/
            Subgraph range:     |----------------------------------|
             */
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test");
            let (block_group1_id, original_path) = setup_block_group(conn);

            let intervaltree = original_path.intervaltree(conn);
            let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
            let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

            let insert_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAAAAA")
                .save(conn);
            let insert_node_id = Node::create(
                conn,
                &insert_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-a-node.{}",
                    insert_sequence.hash
                ))),
            );
            let edge_into_insert = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_node_id,
                0,
                Strand::Forward,
            );
            let edge_out_of_insert = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            );
            let ref_heal_1 = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_start_node_id,
                6,
                Strand::Forward,
            );
            let ref_heal_2 = Edge::create(
                conn,
                insert_end_node_id,
                4,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            );

            let edge_ids = [
                &edge_into_insert.id,
                &edge_out_of_insert.id,
                &ref_heal_1.id,
                &ref_heal_2.id,
            ];
            let block_group_edges = edge_ids
                .iter()
                .enumerate()
                .map(|(i, edge_id)| BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: *(*edge_id),
                    chromosome_index: if i < 2 { 1 } else { 0 },
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>();
            BlockGroupEdge::bulk_create(conn, &block_group_edges);

            let insert_path =
                original_path.new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert);
            assert_eq!(
                insert_path.sequence(conn),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
            );

            let insert2_start_node_id = intervaltree.query_point(28).next().unwrap().value.node_id;
            let insert2_end_node_id = intervaltree.query_point(32).next().unwrap().value.node_id;

            let insert2_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("TTTTTTTT")
                .save(conn);
            let insert2_node_id = Node::create(
                conn,
                &insert2_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-t-node.{}",
                    insert2_sequence.hash
                ))),
            );
            let edge_into_insert2 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_node_id,
                0,
                Strand::Forward,
            );
            let edge_out_of_insert2 = Edge::create(
                conn,
                insert2_node_id,
                8,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            );
            let ref_heal_1 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_start_node_id,
                6,
                Strand::Forward,
            );
            let ref_heal_2 = Edge::create(
                conn,
                insert2_end_node_id,
                4,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            );

            let edge_ids = [
                &edge_into_insert2.id,
                &edge_out_of_insert2.id,
                &ref_heal_1.id,
                &ref_heal_2.id,
            ];
            let block_group_edges = edge_ids
                .iter()
                .enumerate()
                .map(|(i, edge_id)| BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: *(*edge_id),
                    chromosome_index: if i < 2 { 1 } else { 0 },
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>();
            BlockGroupEdge::bulk_create(conn, &block_group_edges);

            let insert2_path =
                insert_path.new_path_with(conn, 28, 32, &edge_into_insert2, &edge_out_of_insert2);
            assert_eq!(
                insert2_path.sequence(conn),
                "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG"
            );

            let all_sequences = BlockGroup::get_all_sequences(conn, &block_group1_id, false);
            assert_eq!(
                all_sequences,
                HashSet::from_iter(vec![
                    "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTTTTTCCCCCCTTTTTTTTGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG".to_string(),
                ])
            );

            let mut blocks = intervaltree
                .query(Range { start: 15, end: 36 })
                .map(|x| x.value)
                .collect::<Vec<_>>();
            blocks.sort_by(|a, b| a.start.cmp(&b.start));
            let start_block = blocks[0];
            let start_node_coordinate = 15 - start_block.start + start_block.sequence_start;
            let end_block = blocks[blocks.len() - 1];
            let end_node_coordinate = 36 - end_block.start + end_block.sequence_start;

            let block_group2 = BlockGroup::create(conn, "test", None, "chr1.1");
            BlockGroup::derive_subgraph(
                conn,
                "test",
                "test",
                &block_group1_id,
                &start_block,
                &end_block,
                start_node_coordinate,
                end_node_coordinate,
                &block_group2.id,
                true,
            );
            let all_sequences2 = BlockGroup::get_all_sequences(conn, &block_group2.id, false);
            assert_eq!(
                all_sequences2,
                HashSet::from_iter(vec![
                    "TTTTTCCCCCCCCCCGGGGGG".to_string(),
                    "TAAAAAAAACCCCCCGGGGGG".to_string(),
                    "TTTTTCCCCCCTTTTTTTTGG".to_string(),
                    "TAAAAAAAACCTTTTTTTTGG".to_string(),
                ])
            );
        }

        #[test]
        fn test_derive_subgraph_two_independent_insertions_and_one_deletion() {
            /*
                       /--------------------------------------------\  (<-- Deletion edge)
            AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -----> GGGGGGGGGG
                                  \-> AAAAAAAA ->/  \->TTTTTTTT -/
            Subgraph range: |----------------------------------|

            Confirms that deletion edge is ignored and not added to subgraph
             */
            let conn = &get_connection(None).unwrap();
            Collection::create(conn, "test");
            let (block_group1_id, original_path) = setup_block_group(conn);

            let intervaltree = original_path.intervaltree(conn);
            let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
            let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

            let insert_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAAAAA")
                .save(conn);
            let insert_node_id = Node::create(
                conn,
                &insert_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-a-node.{}",
                    insert_sequence.hash
                ))),
            );
            let edge_into_insert = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_node_id,
                0,
                Strand::Forward,
            );
            let edge_out_of_insert = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            );
            let ref_heal_1 = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_start_node_id,
                6,
                Strand::Forward,
            );
            let ref_heal_2 = Edge::create(
                conn,
                insert_end_node_id,
                4,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            );

            let edge_ids = [
                &edge_into_insert.id,
                &edge_out_of_insert.id,
                &ref_heal_1.id,
                &ref_heal_2.id,
            ];
            let block_group_edges = edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: *(*edge_id),
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>();
            BlockGroupEdge::bulk_create(conn, &block_group_edges);

            let insert_path =
                original_path.new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert);
            assert_eq!(
                insert_path.sequence(conn),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
            );

            let insert2_start_node_id = intervaltree.query_point(28).next().unwrap().value.node_id;
            let insert2_end_node_id = intervaltree.query_point(32).next().unwrap().value.node_id;

            let insert2_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("TTTTTTTT")
                .save(conn);
            let insert2_node_id = Node::create(
                conn,
                &insert2_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-t-node.{}",
                    insert2_sequence.hash
                ))),
            );
            let edge_into_insert2 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_node_id,
                0,
                Strand::Forward,
            );
            let edge_out_of_insert2 = Edge::create(
                conn,
                insert2_node_id,
                8,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            );
            let ref_heal_1 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_start_node_id,
                6,
                Strand::Forward,
            );
            let ref_heal_2 = Edge::create(
                conn,
                insert2_end_node_id,
                4,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            );

            let edge_ids = [
                &edge_into_insert2.id,
                &edge_out_of_insert2.id,
                &ref_heal_1.id,
                &ref_heal_2.id,
            ];
            let block_group_edges = edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: *(*edge_id),
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<BlockGroupEdgeData>>();
            BlockGroupEdge::bulk_create(conn, &block_group_edges);

            let insert2_path =
                insert_path.new_path_with(conn, 28, 32, &edge_into_insert2, &edge_out_of_insert2);
            assert_eq!(
                insert2_path.sequence(conn),
                "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG"
            );

            let deletion_end_node_id = intervaltree.query_point(38).next().unwrap().value.node_id;
            let deletion_edge = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                deletion_end_node_id,
                8,
                Strand::Forward,
            );
            let ref_heal_1 = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_node_id,
                8,
                Strand::Forward,
            );
            let ref_heal_2 = Edge::create(
                conn,
                deletion_end_node_id,
                8,
                Strand::Forward,
                deletion_end_node_id,
                8,
                Strand::Forward,
            );
            let block_group_edges = [
                BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: deletion_edge.id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: ref_heal_1.id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                },
                BlockGroupEdgeData {
                    block_group_id: block_group1_id,
                    edge_id: ref_heal_2.id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                },
            ];
            BlockGroupEdge::bulk_create(conn, &block_group_edges);

            let all_sequences = BlockGroup::get_all_sequences(conn, &block_group1_id, false);
            assert_eq!(
                all_sequences,
                HashSet::from_iter(vec![
                    "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTTTTTCCCCCCTTTTTTTTGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG".to_string(),
                    "AAAAAAAAAATTTTTTAAAAAAAAGG".to_string(), // Sequence including deletion
                ])
            );

            let mut blocks = intervaltree
                .query(Range { start: 15, end: 36 })
                .map(|x| x.value)
                .collect::<Vec<_>>();
            blocks.sort_by(|a, b| a.start.cmp(&b.start));
            let start_block = blocks[0];
            let start_node_coordinate = 15 - start_block.start + start_block.sequence_start;
            let end_block = blocks[blocks.len() - 1];
            let end_node_coordinate = 36 - end_block.start + end_block.sequence_start;

            let block_group2 = BlockGroup::create(conn, "test", None, "chr1.1");
            BlockGroup::derive_subgraph(
                conn,
                "test",
                "test",
                &block_group1_id,
                &start_block,
                &end_block,
                start_node_coordinate,
                end_node_coordinate,
                &block_group2.id,
                true,
            );
            let all_sequences2 = BlockGroup::get_all_sequences(conn, &block_group2.id, false);
            assert_eq!(
                all_sequences2,
                // The deletion is not included in the cloned subgraph since one end of it is
                // outside the specified range
                HashSet::from_iter(vec![
                    "TTTTTCCCCCCCCCCGGGGGG".to_string(),
                    "TAAAAAAAACCCCCCGGGGGG".to_string(),
                    "TTTTTCCCCCCTTTTTTTTGG".to_string(),
                    "TAAAAAAAACCTTTTTTTTGG".to_string(),
                ])
            );
        }
    }
}
