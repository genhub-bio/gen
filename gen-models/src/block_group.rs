use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
};

use gen_core::{
    HashId, INDETERMINATE_CHROMOSOME_INDEX, NO_CHROMOSOME_INDEX, NodeIntervalBlock,
    PATH_END_NODE_ID, PATH_START_NODE_ID, PRESERVE_EDIT_SITE_CHROMOSOME_INDEX, PathBlock, Strand,
    calculate_hash, is_end_node, is_start_node, is_terminal,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use gen_graph::{
    GenGraph, GraphNode, all_intermediate_edges, all_reachable_nodes, all_simple_paths,
    flatten_to_interval_tree,
};
use intervaltree::IntervalTree;
use rusqlite::{Row, params, types::Value as SQLValue};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    accession::{Accession, AccessionEdge, AccessionEdgeData},
    annotations::AnnotationError,
    block_group_edge::{AugmentedEdge, AugmentedEdgeData, BlockGroupEdge, BlockGroupEdgeData},
    db::GraphConnection,
    edge::{Edge, EdgeData, GroupBlock},
    errors::{AccessionError, EdgeError, NodeError, PathError, QueryError, SequenceError},
    gen_models_capnp::block_group,
    node::Node,
    path::{Path, PathData},
    path_edge::PathEdge,
    region::{ResolvedGenRegion, ResolvedRegionKind},
    sample::Sample,
    traits::*,
};

#[derive(Clone, Debug, Deserialize, Eq, Hash, Serialize, PartialEq)]
pub struct BlockGroup {
    pub id: HashId,
    pub collection_name: String,
    pub sample_name: String,
    pub name: String,
    pub created_on: i64,
    pub parent_block_group_id: Option<HashId>,
    pub is_default: bool,
}

#[derive(Debug, Error, PartialEq)]
pub enum BlockGroupError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Path creation error: {0}")]
    PathError(#[from] PathError),
    #[error("Edge creation error: {0}")]
    EdgeError(#[from] EdgeError),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Accession creation error: {0}")]
    AccessionError(#[from] AccessionError),
    #[error("Annotation error: {0}")]
    AnnotationError(#[from] AnnotationError),
    #[error("Query error: {0}")]
    QueryError(#[from] QueryError),
    #[error("Change error: {0}")]
    ChangeOutOfBounds(String),
    #[error("Sequence error: {0}")]
    SequenceError(#[from] SequenceError),
}

impl<'a> Capnp<'a> for BlockGroup {
    type Builder = block_group::Builder<'a>;
    type Reader = block_group::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_collection_name(&self.collection_name);
        builder.set_sample_name(&self.sample_name);
        builder.set_name(&self.name);
        builder.set_created_on(self.created_on);
        match &self.parent_block_group_id {
            None => {
                builder.reborrow().get_parent_block_group_id().set_none(());
            }
            Some(parent_block_group_id) => {
                builder
                    .reborrow()
                    .get_parent_block_group_id()
                    .set_some(&parent_block_group_id.0)
                    .unwrap();
            }
        }
        builder.set_is_default(self.is_default);
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
        let sample_name = reader.get_sample_name().unwrap().to_string().unwrap();
        let name = reader.get_name().unwrap().to_string().unwrap();
        let created_on = reader.get_created_on();
        let parent_block_group_id =
            reader
                .get_parent_block_group_id()
                .which()
                .ok()
                .and_then(|parent_block_group_id| match parent_block_group_id {
                    block_group::parent_block_group_id::None(()) => None,
                    block_group::parent_block_group_id::Some(parent_block_group_id) => {
                        if let Ok(parent_block_group_id) = parent_block_group_id {
                            if let Some(parent_block_group_id) = parent_block_group_id.as_slice() {
                                parent_block_group_id.try_into().ok()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                });
        let is_default = reader.get_is_default();

        BlockGroup {
            id,
            collection_name,
            sample_name,
            name,
            created_on,
            parent_block_group_id,
            is_default,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockGroupData<'a> {
    pub collection_name: &'a str,
    pub sample_name: &'a str,
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct NewBlockGroup<'a> {
    pub collection_name: &'a str,
    pub sample_name: &'a str,
    pub name: &'a str,
    pub parent_block_group_id: Option<&'a HashId>,
    pub is_default: bool,
}

#[derive(Clone, Debug)]
pub struct BlockGroupChange {
    pub region: ResolvedGenRegion,
    pub path_accession: Option<String>,
    pub block: PathBlock,
    pub chromosome_index: i64,
    pub phased: i64,
    pub preserve_edge: bool,
}

pub trait IntervalTreeSource {
    fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, BlockGroupError>;
}

pub type IntervalTreeCache =
    HashMap<(HashId, ResolvedRegionKind), IntervalTree<i64, NodeIntervalBlock>>;

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

    pub fn lookup(
        path_cache: &mut PathCache,
        block_group_id: &HashId,
        name: String,
    ) -> Result<Path, PathError> {
        let path_key = PathData {
            name: name.clone(),
            block_group_id: *block_group_id,
        };
        let path_lookup = path_cache.cache.get(&path_key);
        if let Some(path) = path_lookup {
            Ok(path.clone())
        } else {
            let conn = path_cache.conn;
            let new_path = Path::query(
                conn,
                "select * from paths where block_group_id = ?1 AND name = ?2",
                params![block_group_id, name],
            )[0]
            .clone();

            path_cache.cache.insert(path_key, new_path.clone());
            let tree = new_path.intervaltree(conn)?;
            path_cache.intervaltree_cache.insert(new_path.clone(), tree);
            Ok(new_path)
        }
    }

    pub fn get_intervaltree<'b>(
        path_cache: &'b mut PathCache<'_>,
        path: &Path,
    ) -> Result<&'b IntervalTree<i64, NodeIntervalBlock>, PathError> {
        if !path_cache.intervaltree_cache.contains_key(path) {
            let tree = path.intervaltree(path_cache.conn)?;
            path_cache.intervaltree_cache.insert(path.clone(), tree);
        }

        path_cache.intervaltree_cache.get(path).ok_or_else(|| {
            PathError::Missing(format!("Missing interval tree for path {}", path.id))
        })
    }
}

impl BlockGroup {
    pub fn create(
        conn: &GraphConnection,
        new_block_group: NewBlockGroup<'_>,
    ) -> Result<BlockGroup, BlockGroupError> {
        let hash = BlockGroup::get_id(
            new_block_group.collection_name,
            new_block_group.sample_name,
            new_block_group.name,
            new_block_group.parent_block_group_id,
        );
        let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap();
        let query = "INSERT INTO block_groups (
                id,
                collection_name,
                sample_name,
                name,
                created_on,
                parent_block_group_id,
                is_default
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);";
        let mut stmt = match conn.prepare(query) {
            Ok(stmt) => stmt,
            Err(e) => return Err(BlockGroupError::DatabaseError(e)),
        };
        let bg = BlockGroup {
            id: hash,
            collection_name: new_block_group.collection_name.to_string(),
            sample_name: new_block_group.sample_name.to_string(),
            name: new_block_group.name.to_string(),
            created_on: timestamp,
            parent_block_group_id: new_block_group.parent_block_group_id.copied(),
            is_default: new_block_group.is_default,
        };
        match stmt.execute(params![
            hash,
            new_block_group.collection_name,
            new_block_group.sample_name,
            new_block_group.name,
            timestamp,
            new_block_group.parent_block_group_id,
            new_block_group.is_default,
        ]) {
            Ok(_) => Ok(bg),
            Err(rusqlite::Error::SqliteFailure(err, _details))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Ok(bg)
            }
            Err(e) => Err(BlockGroupError::DatabaseError(e)),
        }
    }

    pub fn delete(conn: &GraphConnection, collection_name: &str, sample_name: &str, name: &str) {
        let query = "delete from block_groups where collection_name = ?1 and sample_name = ?2 and name = ?3;";
        conn.execute(
            query,
            params![
                collection_name.to_string(),
                sample_name.to_string(),
                name.to_string()
            ],
        )
        .unwrap();
    }

    pub fn get_by_id(conn: &GraphConnection, id: &HashId) -> Result<BlockGroup, BlockGroupError> {
        let query = "SELECT * FROM block_groups WHERE id = ?1";
        let mut stmt = match conn.prepare(query) {
            Ok(stmt) => stmt,
            Err(e) => return Err(BlockGroupError::DatabaseError(e)),
        };
        match stmt.query_row([id], |row| Ok(Self::process_row(row))) {
            Ok(res) => Ok(res),
            Err(rusqlite::Error::QueryReturnedNoRows) => Err(BlockGroupError::QueryError(
                QueryError::ResultsNotFound(format!("BlockGroup with id {id} not found")),
            )),
            Err(_) => Err(BlockGroupError::DatabaseError(
                rusqlite::Error::QueryReturnedNoRows,
            )),
        }
    }

    pub fn get_reference_block_groups(
        conn: &GraphConnection,
        collection_name: &str,
    ) -> Vec<BlockGroup> {
        BlockGroup::query(
            conn,
            "select bg.*
             from block_groups bg
             join samples s on s.name = bg.sample_name
             where bg.collection_name = ?1 and s.is_reference = 1
             order by bg.name, bg.sample_name, bg.created_on, bg.id;",
            params![collection_name],
        )
    }

    fn copy_paths_and_accessions_into(
        &self,
        conn: &GraphConnection,
        target_block_group_id: &HashId,
    ) -> Result<(), BlockGroupError> {
        let existing_paths = Path::query(
            conn,
            "SELECT * from paths where block_group_id = ?1;",
            params![self.id],
        );

        for path in &existing_paths {
            let edge_ids = PathEdge::edges_for_path(conn, &path.id)
                .into_iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>();
            Path::create(conn, &path.name, target_block_group_id, &edge_ids)?;
        }

        Ok(())
    }

    pub fn get_or_create_sample_block_groups(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        group_name: &str,
        parent_samples: Vec<String>,
    ) -> Result<Vec<BlockGroup>, BlockGroupError> {
        let existing_block_groups = BlockGroup::query(
            conn,
            "select * from block_groups
             where collection_name = ?1 AND sample_name = ?2 AND name = ?3
             order by created_on, id",
            params![collection_name, sample_name, group_name],
        );

        if !existing_block_groups.is_empty() {
            return Ok(existing_block_groups);
        }

        let parent_block_groups = BlockGroup::find_parent_block_groups(
            conn,
            collection_name,
            sample_name,
            group_name,
            &parent_samples,
        );

        if parent_block_groups.is_empty() {
            return Ok(vec![BlockGroup::create(
                conn,
                NewBlockGroup {
                    collection_name,
                    sample_name,
                    name: group_name,
                    ..Default::default()
                },
            )?]);
        }

        let mut new_block_groups = vec![];
        for parent_block_group in &parent_block_groups {
            let new_block_group = BlockGroup::create(
                conn,
                NewBlockGroup {
                    collection_name,
                    sample_name,
                    name: group_name,
                    parent_block_group_id: Some(&parent_block_group.id),
                    ..Default::default()
                },
            )?;
            new_block_group.copy_contents_from(conn, parent_block_group)?;
            new_block_groups.push(new_block_group);
        }

        Ok(new_block_groups)
    }

    pub fn find_parent_block_groups(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        group_name: &str,
        parent_samples: &[String],
    ) -> Vec<BlockGroup> {
        let parent_samples = if parent_samples.is_empty() {
            Sample::get_parent_names(conn, sample_name)
        } else {
            parent_samples.to_vec()
        };
        if parent_samples.is_empty() {
            return vec![];
        }

        BlockGroup::query(
            conn,
            "select * from block_groups
             where collection_name = ?1 AND sample_name IN rarray(?2) AND name = ?3
             order by sample_name, created_on, id",
            params![
                collection_name,
                Rc::new(
                    parent_samples
                        .iter()
                        .cloned()
                        .map(SQLValue::from)
                        .collect::<Vec<_>>()
                ),
                group_name
            ],
        )
    }

    fn copy_contents_from(
        &self,
        conn: &GraphConnection,
        source_block_group: &BlockGroup,
    ) -> Result<(), BlockGroupError> {
        let new_block_group_edges =
            BlockGroupEdge::edges_for_block_group(conn, &source_block_group.id)
                .into_iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: self.id,
                    edge_id: edge.edge.id,
                    chromosome_index: edge.chromosome_index,
                    phased: edge.phased,
                })
                .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        source_block_group.copy_paths_and_accessions_into(conn, &self.id)?;
        Ok(())
    }

    pub fn get_id(
        collection_name: &str,
        sample_name: &str,
        group_name: &str,
        parent_block_group_id: Option<&HashId>,
    ) -> HashId {
        let lineage_key = parent_block_group_id
            .map(ToString::to_string)
            .unwrap_or_default();
        HashId(calculate_hash(&format!(
            "{collection_name}:{sample_name}:{group_name}:{lineage_key}"
        )))
    }

    pub fn get_graph(
        conn: &GraphConnection,
        block_group_id: &HashId,
    ) -> Result<GenGraph, BlockGroupError> {
        let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id);
        let blocks = Edge::blocks_from_edges(conn, block_group_id, &edges)?;
        let (graph, _) = Edge::build_graph(&edges, &blocks);
        Ok(graph)
    }

    pub fn get_graph_from_edges(
        conn: &GraphConnection,
        block_group_id: &HashId,
        edges: &[AugmentedEdge],
    ) -> Result<GenGraph, BlockGroupError> {
        let blocks = Edge::blocks_from_edges(conn, block_group_id, edges)?;
        let edges_vec = edges.to_vec();
        let (graph, _) = Edge::build_graph(&edges_vec, &blocks);
        Ok(graph)
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
                    if edge_weight.chromosome_index == INDETERMINATE_CHROMOSOME_INDEX {
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
    ) -> Result<HashSet<String>, BlockGroupError> {
        let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id)
            .into_iter()
            .filter(|edge| edge.chromosome_index != PRESERVE_EDIT_SITE_CHROMOSOME_INDEX)
            .collect::<Vec<_>>();
        let blocks = Edge::blocks_from_edges(conn, block_group_id, &edges)?;

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
        let blocks_by_node = blocks
            .iter()
            .map(|block| {
                (
                    GraphNode {
                        node_id: block.node_id,
                        sequence_start: block.start,
                        sequence_end: block.end,
                    },
                    block,
                )
            })
            .collect::<HashMap<GraphNode, &GroupBlock>>();
        let mut sequences = HashSet::<String>::new();

        for start_node in start_nodes {
            for end_node in &end_nodes {
                // TODO: maybe make all_simple_paths return a single path id where start == end
                if start_node == *end_node {
                    let block = blocks_by_node.get(&start_node).unwrap();
                    if block.node_id != PATH_START_NODE_ID && block.node_id != PATH_END_NODE_ID {
                        sequences.insert(block.sequence());
                    }
                } else {
                    for path in all_simple_paths(&graph, start_node, *end_node) {
                        let mut current_sequence = "".to_string();
                        for node in path {
                            let block = blocks_by_node.get(&node).unwrap();
                            let block_sequence = block.sequence();
                            current_sequence.push_str(&block_sequence);
                        }
                        sequences.insert(current_sequence);
                    }
                }
            }
        }

        Ok(sequences)
    }

    pub fn add_accession(
        conn: &GraphConnection,
        path: &Path,
        name: &str,
        start: i64,
        end: i64,
        cache: &mut PathCache,
    ) -> Result<Accession, BlockGroupError> {
        let tree = PathCache::get_intervaltree(cache, path)?;
        let start_blocks: Vec<&NodeIntervalBlock> =
            tree.query_point(start).map(|x| &x.value).collect();
        assert_eq!(start_blocks.len(), 1);
        let start_block = start_blocks[0];
        let end_blocks: Vec<&NodeIntervalBlock> =
            tree.query_point(end - 1).map(|x| &x.value).collect();
        assert_eq!(end_blocks.len(), 1);
        let end_block = end_blocks[0];
        let all_path_edges = PathEdge::edges_for_path(conn, &path.id);
        let start_coordinate = start - start_block.start + start_block.sequence_start;
        let end_coordinate = end - end_block.start + end_block.sequence_start;
        let start_edge = all_path_edges
            .iter()
            .find(|edge| {
                edge.target_node_id == start_block.node_id
                    && edge.target_coordinate == start_block.sequence_start
            })
            .expect("should find path edge entering accession start block");
        let end_edge = all_path_edges
            .iter()
            .find(|edge| {
                edge.source_node_id == end_block.node_id
                    && edge.source_coordinate == end_block.sequence_end
            })
            .expect("should find path edge leaving accession end block");
        let start_edge = AccessionEdgeData {
            edge_id: start_edge.id,
            source_offset: None,
            target_offset: Some(start_coordinate - start_edge.target_coordinate),
        };
        let end_edge = AccessionEdgeData {
            edge_id: end_edge.id,
            source_offset: Some(end_coordinate - end_edge.source_coordinate),
            target_offset: None,
        };
        let accession = Accession::create(conn, name, &path.block_group_id, None)
            .expect("Unable to create accession.");
        let mut accession_path_edges = vec![start_edge];
        if start_block == end_block {
            accession_path_edges.push(end_edge);
        } else {
            let mut in_range = false;
            let path_blocks: Vec<&NodeIntervalBlock> = tree
                .iter_sorted()
                .map(|x| &x.value)
                .filter(|block| {
                    if *block == start_block {
                        in_range = true;
                    } else if *block == end_block {
                        in_range = false;
                        return true;
                    }
                    in_range
                })
                .collect::<Vec<_>>();
            for (block, next_block) in path_blocks.iter().zip(path_blocks[1..].iter()) {
                let edge = all_path_edges
                    .iter()
                    .find(|edge| {
                        edge.source_node_id == block.node_id
                            && edge.source_coordinate == block.sequence_end
                            && edge.target_node_id == next_block.node_id
                            && edge.target_coordinate == next_block.sequence_start
                    })
                    .expect("should find path edge between accession blocks");
                accession_path_edges.push(AccessionEdgeData {
                    edge_id: edge.id,
                    source_offset: Some(0),
                    target_offset: Some(0),
                })
            }
            accession_path_edges.push(end_edge);
        }
        AccessionEdge::bulk_create(conn, &accession.id, &accession_path_edges);
        Ok(accession)
    }

    pub fn insert_change(
        conn: &GraphConnection,
        change: &BlockGroupChange,
    ) -> Result<(), BlockGroupError> {
        let new_augmented_edges = change.region.plan_edges(conn, change, None)?;
        let mut new_augmented_edges_by_block_group = HashMap::new();
        new_augmented_edges_by_block_group
            .insert(change.region.block_group.id, new_augmented_edges.clone());
        let mut new_accession_edges = HashMap::new();
        if let Some(accession) = &change.path_accession {
            new_accession_edges.insert(
                (change.region.block_group.id, accession.clone()),
                new_augmented_edges,
            );
        }
        Self::persist_insert_changes(
            conn,
            new_augmented_edges_by_block_group,
            new_accession_edges,
        )
    }

    pub fn insert_changes(
        conn: &GraphConnection,
        changes: &[BlockGroupChange],
        tree_map: Option<&mut IntervalTreeCache>,
    ) -> Result<(), BlockGroupError> {
        let mut new_augmented_edges_by_block_group =
            HashMap::<HashId, Vec<AugmentedEdgeData>>::new();
        let mut new_accession_edges = HashMap::<(HashId, String), Vec<AugmentedEdgeData>>::new();
        let mut local_tree_map = HashMap::new();
        let tree_map = match tree_map {
            Some(tree_map) => tree_map,
            None => &mut local_tree_map,
        };
        for change in changes {
            let cache_key = change.region.intervaltree_cache_key();
            #[expect(
                clippy::map_entry,
                reason = "entry API doesn't work with ? error propagation"
            )]
            if !tree_map.contains_key(&cache_key) {
                tree_map.insert(
                    cache_key,
                    IntervalTreeSource::intervaltree(&change.region, conn)?,
                );
            }
            let tree = tree_map.get(&cache_key);
            let new_augmented_edges = change.region.plan_edges(conn, change, tree)?;
            new_augmented_edges_by_block_group
                .entry(change.region.block_group.id)
                .and_modify(|new_edge_data| new_edge_data.extend(new_augmented_edges.clone()))
                .or_insert_with(|| new_augmented_edges.clone());
            if let Some(accession) = &change.path_accession {
                new_accession_edges
                    .entry((change.region.block_group.id, accession.clone()))
                    .and_modify(|new_edge_data: &mut Vec<AugmentedEdgeData>| {
                        new_edge_data.extend(new_augmented_edges.clone())
                    })
                    .or_insert_with(|| new_augmented_edges.clone());
            }
        }
        Self::persist_insert_changes(
            conn,
            new_augmented_edges_by_block_group,
            new_accession_edges,
        )
    }

    fn persist_insert_changes(
        conn: &GraphConnection,
        new_augmented_edges_by_block_group: HashMap<HashId, Vec<AugmentedEdgeData>>,
        new_accession_edges: HashMap<(HashId, String), Vec<AugmentedEdgeData>>,
    ) -> Result<(), BlockGroupError> {
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
                    block_group_id,
                    edge_id: *edge_id,
                    chromosome_index: new_augmented_edges[i].chromosome_index,
                    phased: new_augmented_edges[i].phased,
                })
                .collect::<Vec<_>>();
            BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        }

        for ((block_group_id, accession_name), path_edges) in new_accession_edges {
            match Accession::get(
                conn,
                "select * from accessions where name = ?1 AND block_group_id = ?2",
                params![accession_name, block_group_id],
            ) {
                Ok(_) => {
                    println!(
                        "accession already exists, consider a better matching algorithm to determine if this is an error."
                    );
                }
                Err(_) => {
                    let accession_edge_data = path_edges
                        .iter()
                        .map(|augmented_edge| AccessionEdgeData {
                            edge_id: edge_data_map[&augmented_edge.edge_data],
                            source_offset: Some(0),
                            target_offset: Some(0),
                        })
                        .collect::<Vec<_>>();
                    let acc = Accession::create(conn, &accession_name, &block_group_id, None)
                        .expect("Accession could not be created.");
                    AccessionEdge::bulk_create(conn, &acc.id, &accession_edge_data);
                }
            }
        }
        Ok(())
    }

    pub fn set_up_new_edges(
        change: &BlockGroupChange,
        tree: &IntervalTree<i64, NodeIntervalBlock>,
    ) -> Result<Vec<AugmentedEdgeData>, BlockGroupError> {
        let start_blocks: Vec<&NodeIntervalBlock> = tree
            .query_point(change.region.start)
            .map(|x| &x.value)
            .collect();
        assert_eq!(start_blocks.len(), 1);
        // NOTE: This may not be used but needs to be initialized here instead of inside the if
        // statement that uses it, so that the borrow checker is happy
        let previous_start_blocks: Vec<&NodeIntervalBlock> = tree
            .query_point(change.region.start - 1)
            .map(|x| &x.value)
            .collect();
        assert_eq!(previous_start_blocks.len(), 1);
        let start_block = if start_blocks[0].start == change.region.start {
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
        if is_start_node(start_block.node_id) && change.region.start < start_block.end {
            return Err(BlockGroupError::ChangeOutOfBounds(format!(
                "Invalid change specified. Coordinate {pos} is before start of path range ({path_pos}).",
                pos = change.region.start,
                path_pos = start_block.end
            )));
        }
        let end_blocks: Vec<&NodeIntervalBlock> = tree
            .query_point(change.region.end)
            .map(|x| &x.value)
            .collect();
        assert_eq!(end_blocks.len(), 1);
        let end_block = end_blocks[0];

        if is_end_node(end_block.node_id) && change.region.end > end_block.start {
            return Err(BlockGroupError::ChangeOutOfBounds(format!(
                "Invalid change specified. Coordinate {pos} is before start of path range ({path_pos}).",
                pos = change.region.end,
                path_pos = end_block.start
            )));
        }

        let mut new_edges = vec![];

        if change.block.sequence_start == change.block.sequence_end {
            // Deletion
            let source_coordinate =
                change.region.start - start_block.start + start_block.sequence_start;
            let target_coordinate = change.region.end - end_block.start + end_block.sequence_start;
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
            if change.region.start == 0 {
                let target_coordinate =
                    change.region.end - end_block.start + end_block.sequence_start;
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
                change.region.start - start_block.start + start_block.sequence_start;
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
            let insertion_end_coordinate =
                change.region.end - end_block.start + end_block.sequence_start;
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

            if change.region.start == 0 {
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
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, BlockGroupError> {
        // make a tree where every node has a span in the graph.
        let mut graph = BlockGroup::get_graph(conn, block_group_id)?;
        BlockGroup::prune_graph(&mut graph);
        Ok(flatten_to_interval_tree(&graph, remove_ambiguous_positions))
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
    ) -> Result<HashMap<HashId, HashId>, BlockGroupError> {
        let current_graph = BlockGroup::get_graph(conn, source_block_group_id)?;
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
            Node::create(conn, &old_start_node.sequence_hash, &new_start_node_hash)?,
        );
        let old_end_node = old_nodes_by_id.get(&end_block.node_id).unwrap();
        let new_end_node_hash = HashId(calculate_hash(&format!(
            "{}.{}.{}.bg-{}",
            collection_name, child_sample_name, old_end_node.id, target_block_group_id
        )));
        new_node_ids_by_old.insert(
            old_end_node.id,
            Node::create(conn, &old_end_node.sequence_hash, &new_end_node_hash)?,
        );

        let mut new_edges = vec![];
        for source_edge in &source_edges {
            let new_source_node_id = if let Some(node_id) =
                new_node_ids_by_old.get(&source_edge.source_node_id)
            {
                *node_id
            } else {
                let old_source_node = old_nodes_by_id.get(&source_edge.source_node_id).unwrap();
                let new_source_node_hash = HashId(calculate_hash(&format!(
                    "{}.{}.{}.bg-{}",
                    collection_name, child_sample_name, old_source_node.id, target_block_group_id
                )));
                let new_node_id =
                    Node::create(conn, &old_source_node.sequence_hash, &new_source_node_hash)?;
                new_node_ids_by_old.insert(old_source_node.id, new_node_id);
                new_node_id
            };
            let new_target_node_id = if let Some(node_id) =
                new_node_ids_by_old.get(&source_edge.target_node_id)
            {
                *node_id
            } else {
                let old_target_node = old_nodes_by_id.get(&source_edge.target_node_id).unwrap();
                let new_target_node_hash = HashId(calculate_hash(&format!(
                    "{}.{}.{}.bg-{}",
                    collection_name, child_sample_name, old_target_node.id, target_block_group_id
                )));
                let new_node_id =
                    Node::create(conn, &old_target_node.sequence_hash, &new_target_node_hash)?;
                new_node_ids_by_old.insert(old_target_node.id, new_node_id);
                new_node_id
            };
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
            )?;
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
            )?;
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

        Ok(new_node_ids_by_old)
    }
}

impl RegionResolver for BlockGroup {
    type Connection = GraphConnection;
    type Error = BlockGroupError;

    fn resolve(
        region: &Region,
        conn: &Self::Connection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<Self, RegionResolutionError<Self::Error>> {
        let matches = BlockGroup::query(
            conn,
            "SELECT * FROM block_groups \
             WHERE collection_name = ?1 \
               AND sample_name = ?2 \
               AND lower(name) = lower(?3)",
            params![collection_name, sample_name, region.name],
        );

        match matches.len() {
            0 => Err(RegionResolutionError::NotFound(region.name.clone())),
            1 => Ok(matches.into_iter().next().unwrap()),
            _ => Err(RegionResolutionError::Ambiguous(format!(
                "multiple block groups named {}",
                region.name
            ))),
        }
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
            parent_block_group_id: row.get(5).unwrap(),
            is_default: row.get(6).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Range;
    use std::collections::HashSet;

    use capnp::message::TypedBuilder;
    use chrono::Utc;
    use gen_core::{NO_CHROMOSOME_INDEX, region::RegionResolutionError};

    use super::*;
    use crate::{
        annotations::Annotation as ModelAnnotation,
        collection::Collection,
        node::Node,
        region::{ResolvedGenRegion, ResolvedRegionKind},
        sample::{NewSample, Sample},
        sequence::Sequence,
        test_helpers::{create_bg, get_connection, interval_tree_verify, setup_block_group},
    };

    mod region_resolver {
        use super::*;

        #[test]
        fn resolves_block_group_by_name_case_insensitively() {
            let conn = &get_connection(None).unwrap();
            let (block_group_id, _path) = setup_block_group(conn);

            let region = Region::parse("CHR1").unwrap();
            let resolved = BlockGroup::resolve(&region, conn, "test", "test").unwrap();
            assert_eq!(resolved.id, block_group_id);
        }

        #[test]
        fn returns_not_found_for_missing_block_group() {
            let conn = &get_connection(None).unwrap();
            let (_block_group_id, _path) = setup_block_group(conn);

            let region = Region::parse("missing").unwrap();
            let err = BlockGroup::resolve(&region, conn, "test", "test").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::NotFound(name) if name == "missing"
            ));
        }

        #[test]
        fn returns_ambiguous_for_multiple_matching_block_groups() {
            let conn = &get_connection(None).unwrap();
            let (_block_group_id, _path) = setup_block_group(conn);
            let _ = create_bg(conn, "test", "test", "CHR1");

            let region = Region::parse("chr1").unwrap();
            let err = BlockGroup::resolve(&region, conn, "test", "test").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::Ambiguous(name)
                    if name == "multiple block groups named chr1"
            ));
        }
    }

    fn get_single_bg_id(
        conn: &GraphConnection,
        collection_name: &str,
        sample_name: &str,
        group_name: &str,
        parent_samples: Vec<String>,
    ) -> HashId {
        BlockGroup::get_or_create_sample_block_groups(
            conn,
            collection_name,
            sample_name,
            group_name,
            parent_samples,
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .id
    }

    #[test]
    fn test_capnp_serialization() {
        let block_group = BlockGroup {
            id: HashId::pad_str(42),
            collection_name: "test_collection".to_string(),
            sample_name: "test_sample".to_string(),
            name: "test_block_group".to_string(),
            created_on: Utc::now().timestamp_nanos_opt().unwrap(),
            parent_block_group_id: None,
            is_default: false,
        };

        let mut message = TypedBuilder::<block_group::Owned>::new_default();
        let mut root = message.init_root();
        block_group.write_capnp(&mut root);

        let deserialized = BlockGroup::read_capnp(root.into_reader());
        assert_eq!(block_group, deserialized);
    }

    #[test]
    fn test_capnp_deserialization_defaults_missing_parent_to_none() {
        let created_on = Utc::now().timestamp_nanos_opt().unwrap();

        let mut message = TypedBuilder::<block_group::Owned>::new_default();
        let mut root = message.init_root();
        root.set_id(&HashId::pad_str(42).0).unwrap();
        root.set_collection_name("test_collection");
        root.set_sample_name("test_sample");
        root.set_name("test_block_group");
        root.set_created_on(created_on);

        let deserialized = BlockGroup::read_capnp(root.into_reader());
        assert_eq!(deserialized.parent_block_group_id, None);
        assert!(!deserialized.is_default);
    }

    #[test]
    fn test_blockgroup_create() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: Sample::DEFAULT_NAME,
                ..Default::default()
            },
        )
        .unwrap();
        let bg1 = create_bg(conn, "test", Sample::DEFAULT_NAME, "hg19");
        assert_eq!(bg1.collection_name, "test");
        assert_eq!(bg1.name, "hg19");
        Sample::get_or_create(
            conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .unwrap();
        let bg2 = create_bg(conn, "test", "sample", "hg19");
        assert_eq!(bg2.collection_name, "test");
        assert_eq!(bg2.name, "hg19");
        assert_eq!(bg2.sample_name, "sample".to_string());
        assert_ne!(&bg1.id, &bg2.id);
    }

    #[test]
    fn test_blockgroup_delete() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "sample1",
                ..Default::default()
            },
        )
        .unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "sample2",
                ..Default::default()
            },
        )
        .unwrap();
        let bg1 = create_bg(conn, "test", "sample1", "hg19");
        let bg2 = create_bg(conn, "test", "sample2", "hg19");

        BlockGroup::delete(conn, "test", &bg1.sample_name, &bg1.name);

        let bgs = BlockGroup::all(conn);
        assert_eq!(bgs.len(), 1);
        assert_eq!(bgs[0], bg2);
    }

    #[test]
    fn test_blockgroup_clone() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: Sample::DEFAULT_NAME,
                ..Default::default()
            },
        )
        .unwrap();
        let bg1 = create_bg(conn, "test", Sample::DEFAULT_NAME, "hg19");
        assert_eq!(bg1.collection_name, "test");
        assert_eq!(bg1.name, "hg19");
        Sample::get_or_create(
            conn,
            NewSample {
                name: "sample",
                ..Default::default()
            },
        )
        .unwrap();
        let bg2 = get_single_bg_id(
            conn,
            "test",
            "sample",
            "hg19",
            vec![Sample::DEFAULT_NAME.to_string()],
        );
        assert_eq!(
            BlockGroupEdge::edges_for_block_group(conn, &bg1.id),
            BlockGroupEdge::edges_for_block_group(conn, &bg2)
        );
    }

    #[test]
    fn test_blockgroup_copies_immediate_parent_block_groups() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "parent_a",
                ..Default::default()
            },
        )
        .unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "parent_b",
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

        let parent_a_bg = create_bg(conn, "test", "parent_a", "chr1");
        let parent_b_bg = create_bg(conn, "test", "parent_b", "chr1");

        let seq_a = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAA")
            .save(conn)
            .unwrap();
        let seq_b = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCC")
            .save(conn)
            .unwrap();
        let node_a =
            Node::create(conn, &seq_a.hash, &HashId::convert_str("merge-parent-a")).unwrap();
        let node_b =
            Node::create(conn, &seq_b.hash, &HashId::convert_str("merge-parent-b")).unwrap();

        let parent_a_edges = [
            Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                node_a,
                0,
                Strand::Forward,
            )
            .unwrap(),
            Edge::create(
                conn,
                node_a,
                4,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap(),
        ];
        let parent_b_edges = [
            Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                node_b,
                0,
                Strand::Forward,
            )
            .unwrap(),
            Edge::create(
                conn,
                node_b,
                4,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap(),
        ];

        BlockGroupEdge::bulk_create(
            conn,
            &parent_a_edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: parent_a_bg.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        BlockGroupEdge::bulk_create(
            conn,
            &parent_b_edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: parent_b_bg.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );

        let child_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            "test",
            "child",
            "chr1",
            vec!["parent_a".to_string(), "parent_b".to_string()],
        )
        .unwrap();
        assert_eq!(child_block_groups.len(), 2);

        let child_by_parent = child_block_groups
            .iter()
            .map(|block_group| (block_group.parent_block_group_id.unwrap(), block_group))
            .collect::<HashMap<_, _>>();

        let child_a = child_by_parent.get(&parent_a_bg.id).unwrap();
        let child_b = child_by_parent.get(&parent_b_bg.id).unwrap();

        let child_a_edges = BlockGroupEdge::query(
            conn,
            "select * from block_group_edges where block_group_id = ?1",
            params![child_a.id],
        );
        let child_b_edges = BlockGroupEdge::query(
            conn,
            "select * from block_group_edges where block_group_id = ?1",
            params![child_b.id],
        );
        assert_eq!(
            child_a_edges
                .iter()
                .map(|edge| edge.edge_id)
                .collect::<HashSet<_>>(),
            parent_a_edges
                .iter()
                .map(|edge| edge.id)
                .collect::<HashSet<_>>()
        );
        assert_eq!(
            child_b_edges
                .iter()
                .map(|edge| edge.edge_id)
                .collect::<HashSet<_>>(),
            parent_b_edges
                .iter()
                .map(|edge| edge.id)
                .collect::<HashSet<_>>()
        );

        assert_eq!(
            BlockGroup::get_all_sequences(conn, &child_a.id, false).unwrap(),
            HashSet::from_iter(vec!["AAAA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &child_b.id, false).unwrap(),
            HashSet::from_iter(vec!["CCCC".to_string()])
        );
        assert_eq!(
            Sample::get_all_sequences(conn, "test", "child", false).unwrap(),
            HashSet::from_iter(vec!["AAAA".to_string(), "CCCC".to_string()])
        );
    }

    #[test]
    fn test_get_or_create_sample_block_groups_creates_root_block_group_if_no_parents() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "root_sample",
                ..Default::default()
            },
        )
        .unwrap();

        let block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            "test",
            "root_sample",
            "chr1",
            vec![],
        )
        .unwrap();

        assert_eq!(block_groups.len(), 1);
        assert_eq!(block_groups[0].collection_name, "test");
        assert_eq!(block_groups[0].sample_name, "root_sample");
        assert_eq!(block_groups[0].name, "chr1");
        assert!(block_groups[0].parent_block_group_id.is_none());
    }

    #[test]
    fn test_get_or_create_sample_block_groups_seeds_from_parents_without_block_groups() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "parent_sample",
                ..Default::default()
            },
        )
        .unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "child_sample",
                ..Default::default()
            },
        )
        .unwrap();

        let block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            "test",
            "child_sample",
            "chr1",
            vec!["parent_sample".to_string()],
        )
        .unwrap();

        assert_eq!(block_groups.len(), 1);
        assert_eq!(block_groups[0].collection_name, "test");
        assert_eq!(block_groups[0].sample_name, "child_sample");
        assert_eq!(block_groups[0].name, "chr1");
        assert!(block_groups[0].parent_block_group_id.is_none());
        assert!(BlockGroupEdge::edges_for_block_group(conn, &block_groups[0].id).is_empty());
    }

    #[test]
    fn test_get_graph_branched_graph() {
        // Branched graph: {AAA,GGG} → TTT → {CCC,ATC}
        // TTT has 2 incoming edges and 2 outgoing edges.
        // There are explicitly no start/end nodes to ensure we can build graphs purely from coordinates
        let conn = get_connection(None).unwrap();
        Collection::get_or_create(&conn, "test").unwrap();
        crate::sample::Sample::get_or_create(
            &conn,
            crate::sample::NewSample {
                name: "test",
                ..Default::default()
            },
        )
        .unwrap();
        let bg = BlockGroup::create(
            &conn,
            crate::block_group::NewBlockGroup {
                collection_name: "test",
                sample_name: "test",
                name: "branched",
                ..Default::default()
            },
        )
        .unwrap();

        let seq_aaa = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAA")
            .save(&conn)
            .unwrap();
        let seq_ggg = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGG")
            .save(&conn)
            .unwrap();
        let seq_ttt = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTT")
            .save(&conn)
            .unwrap();
        let seq_ccc = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCC")
            .save(&conn)
            .unwrap();
        let seq_atc = Sequence::new()
            .sequence_type("DNA")
            .sequence("ATC")
            .save(&conn)
            .unwrap();

        let n_aaa = Node::create(&conn, &seq_aaa.hash, &HashId::convert_str("node-aaa")).unwrap();
        let n_ggg = Node::create(&conn, &seq_ggg.hash, &HashId::convert_str("node-ggg")).unwrap();
        let n_ttt = Node::create(&conn, &seq_ttt.hash, &HashId::convert_str("node-ttt")).unwrap();
        let n_ccc = Node::create(&conn, &seq_ccc.hash, &HashId::convert_str("node-ccc")).unwrap();
        let n_atc = Node::create(&conn, &seq_atc.hash, &HashId::convert_str("node-atc")).unwrap();

        // Edges: AAA→TTT, GGG→TTT, TTT→CCC, TTT→ATC
        let e_aaa_ttt =
            Edge::create(&conn, n_aaa, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
        let e_ggg_ttt =
            Edge::create(&conn, n_ggg, 3, Strand::Forward, n_ttt, 0, Strand::Forward).unwrap();
        let e_ttt_ccc =
            Edge::create(&conn, n_ttt, 3, Strand::Forward, n_ccc, 0, Strand::Forward).unwrap();
        let e_ttt_atc =
            Edge::create(&conn, n_ttt, 3, Strand::Forward, n_atc, 0, Strand::Forward).unwrap();

        let expected_edges = [
            (
                GraphNode {
                    node_id: n_aaa,
                    sequence_start: 3,
                    sequence_end: 3,
                },
                GraphNode {
                    node_id: n_ttt,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                e_aaa_ttt.id,
            ),
            (
                GraphNode {
                    node_id: n_ggg,
                    sequence_start: 3,
                    sequence_end: 3,
                },
                GraphNode {
                    node_id: n_ttt,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                e_ggg_ttt.id,
            ),
            (
                GraphNode {
                    node_id: n_ttt,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                GraphNode {
                    node_id: n_ccc,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                e_ttt_ccc.id,
            ),
            (
                GraphNode {
                    node_id: n_ttt,
                    sequence_start: 0,
                    sequence_end: 3,
                },
                GraphNode {
                    node_id: n_atc,
                    sequence_start: 0,
                    sequence_end: 0,
                },
                e_ttt_atc.id,
            ),
        ];

        let block_group_edges = [
            e_aaa_ttt.clone(),
            e_ggg_ttt.clone(),
            e_ttt_ccc.clone(),
            e_ttt_atc.clone(),
        ]
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id: bg.id,
            edge_id: edge_id.id,
            chromosome_index: 0,
            phased: 0,
        })
        .collect::<Vec<_>>();

        BlockGroupEdge::bulk_create(&conn, &block_group_edges);
        let graph = BlockGroup::get_graph(&conn, &bg.id).unwrap();

        // 5 non-terminal nodes: AAA, GGG, TTT, CCC, ATC
        // 2 terminal blocks: START, END
        // Total: 7
        assert_eq!(
            graph.nodes().len(),
            7,
            "expected 7 blocks (5 nodes + 2 terminals), got {}",
            graph.nodes().len()
        );

        assert_eq!(
            graph.all_edges().count(),
            expected_edges.len(),
            "expected exactly the 4 branch edges"
        );
        for (source, target, edge_id) in expected_edges {
            let weights = graph
                .edge_weight(source, target)
                .unwrap_or_else(|| panic!("missing graph edge {source:?} -> {target:?}"));
            assert_eq!(
                weights.len(),
                1,
                "expected a single edge weight for {source:?} -> {target:?}"
            );
            assert_eq!(weights[0].edge_id, edge_id);
        }
    }

    #[test]
    fn test_blockgroup_merge_from_multiple_parents_preserves_paths_and_accessions() {
        let conn = &get_connection(None).unwrap();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "parent_a",
                ..Default::default()
            },
        )
        .unwrap();
        Sample::get_or_create(
            conn,
            NewSample {
                name: "parent_b",
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

        let parent_a_bg = create_bg(conn, "test", "parent_a", "chr1");
        let parent_b_bg = create_bg(conn, "test", "parent_b", "chr1");

        let seq_a = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAA")
            .save(conn)
            .unwrap();
        let seq_b = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCC")
            .save(conn)
            .unwrap();
        let node_a =
            Node::create(conn, &seq_a.hash, &HashId::convert_str("metadata-parent-a")).unwrap();
        let node_b =
            Node::create(conn, &seq_b.hash, &HashId::convert_str("metadata-parent-b")).unwrap();

        let parent_a_edges = [
            Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                node_a,
                0,
                Strand::Forward,
            )
            .unwrap(),
            Edge::create(
                conn,
                node_a,
                4,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap(),
        ];
        let parent_b_edges = [
            Edge::create(
                conn,
                PATH_START_NODE_ID,
                0,
                Strand::Forward,
                node_b,
                0,
                Strand::Forward,
            )
            .unwrap(),
            Edge::create(
                conn,
                node_b,
                4,
                Strand::Forward,
                PATH_END_NODE_ID,
                0,
                Strand::Forward,
            )
            .unwrap(),
        ];

        BlockGroupEdge::bulk_create(
            conn,
            &parent_a_edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: parent_a_bg.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        BlockGroupEdge::bulk_create(
            conn,
            &parent_b_edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: parent_b_bg.id,
                    edge_id: edge.id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );

        let parent_a_path = Path::create(
            conn,
            "chr1",
            &parent_a_bg.id,
            &parent_a_edges
                .iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let parent_b_path = Path::create(
            conn,
            "chr1",
            &parent_b_bg.id,
            &parent_b_edges
                .iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let parent_b_alt_path = Path::create(
            conn,
            "chr1-alt",
            &parent_b_bg.id,
            &parent_b_edges
                .iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let mut path_cache = PathCache::new(conn);
        let parent_a_path_len = parent_a_path.length(conn).unwrap();
        let parent_b_path_len = parent_b_path.length(conn).unwrap();
        let parent_b_alt_path_len = parent_b_alt_path.length(conn).unwrap();
        BlockGroup::add_accession(
            conn,
            &parent_a_path,
            "parent-a-acc",
            0,
            parent_a_path_len,
            &mut path_cache,
        )
        .unwrap();
        BlockGroup::add_accession(
            conn,
            &parent_b_path,
            "parent-b-acc",
            0,
            parent_b_path_len,
            &mut path_cache,
        )
        .unwrap();
        BlockGroup::add_accession(
            conn,
            &parent_b_alt_path,
            "parent-b-alt-acc",
            0,
            parent_b_alt_path_len,
            &mut path_cache,
        )
        .unwrap();

        let child_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            "test",
            "child",
            "chr1",
            vec!["parent_a".to_string(), "parent_b".to_string()],
        )
        .unwrap();
        let child_by_parent = child_block_groups
            .iter()
            .map(|block_group| (block_group.parent_block_group_id.unwrap(), block_group))
            .collect::<HashMap<_, _>>();
        let child_a = child_by_parent.get(&parent_a_bg.id).unwrap();
        let child_b = child_by_parent.get(&parent_b_bg.id).unwrap();

        let child_a_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name",
            params![child_a.id],
        );
        assert_eq!(
            child_a_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["chr1"]
        );

        let child_b_paths = Path::query(
            conn,
            "select * from paths where block_group_id = ?1 order by name",
            params![child_b.id],
        );
        assert_eq!(
            child_b_paths
                .iter()
                .map(|path| path.name.as_str())
                .collect::<Vec<_>>(),
            vec!["chr1", "chr1-alt"]
        );

        let child_a_accessions = Accession::query(
            conn,
            "select * from accessions where block_group_id = ?1 order by name",
            params![child_a.id],
        );
        assert_eq!(
            child_a_accessions
                .iter()
                .map(|accession| accession.name.as_str())
                .collect::<Vec<_>>(),
            Vec::<&str>::new()
        );

        let child_b_accessions = Accession::query(
            conn,
            "select * from accessions where block_group_id = ?1 order by name",
            params![child_b.id],
        );
        assert_eq!(
            child_b_accessions
                .iter()
                .map(|accession| accession.name.as_str())
                .collect::<Vec<_>>(),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn test_blockgroup_clone_does_not_copy_accessions() {
        let conn = &get_connection(None).unwrap();
        let (_bg_1, path) = setup_block_group(conn);
        let mut path_cache = PathCache::new(conn);
        let acc_1 = BlockGroup::add_accession(conn, &path, "test", 3, 7, &mut path_cache).unwrap();
        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                rusqlite::params!(SQLValue::from("test".to_string())),
            ),
            vec![Accession {
                id: acc_1.id,
                name: "test".to_string(),
                block_group_id: path.block_group_id,
                parent_accession_id: None,
            }]
        );

        Sample::get_or_create(
            conn,
            NewSample {
                name: "sample2",
                ..Default::default()
            },
        )
        .unwrap();
        let _bg2 = get_single_bg_id(conn, "test", "sample2", "chr1", vec!["test".to_string()]);
        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                rusqlite::params!(SQLValue::from("test".to_string())),
            )
            .len(),
            1
        );
    }

    #[test]
    fn accession_end_coordinate_is_not_included() {
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, path) = setup_block_group(conn);
        let mut path_cache = PathCache::new(conn);

        let accession =
            BlockGroup::add_accession(conn, &path, "test", 3, 10, &mut path_cache).unwrap();
        let edges = Accession::get_edges_by_id(conn, &accession.id);

        assert_eq!(accession.length(conn).unwrap(), 7);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].source_offset, None);
        assert_eq!(edges[0].target_offset, Some(3));
        assert_eq!(edges[1].source_offset, Some(0));
        assert_eq!(edges[1].target_offset, None);
    }

    #[test]
    fn insert_accession_change_get_all() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let mut path_cache = PathCache::new(&conn);
        let accession =
            BlockGroup::add_accession(&conn, &path, "test-accession", 10, 30, &mut path_cache)
                .unwrap();
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn)
            .unwrap();
        let insert_node_id = Node::create(
            &conn,
            &insert_sequence.hash,
            &HashId::convert_str("acc-insert-node"),
        )
        .unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 5,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_accession(&conn, &accession, 5, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };

        BlockGroup::insert_change(&conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTNNNNCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn insert_annotation_change_get_all() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let mut path_cache = PathCache::new(&conn);
        let accession =
            BlockGroup::add_accession(&conn, &path, "test-accession", 10, 30, &mut path_cache)
                .unwrap();
        let annotation =
            ModelAnnotation::get_or_create(&conn, "gene-1", "track-1", &accession.id, None)
                .unwrap();
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn)
            .unwrap();
        let deletion_node_id = Node::create(
            &conn,
            &deletion_sequence.hash,
            &HashId::convert_str("annotation-delete-node"),
        )
        .unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 5,
            path_end: 15,
            strand: Strand::Forward,
        };
        let annotation_accession = Accession::get_by_id(&conn, &annotation.accession_id).unwrap();
        let region =
            ResolvedGenRegion::from_annotation(&conn, &annotation, &annotation_accession, 5, 15)
                .unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };

        BlockGroup::insert_change(&conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTCCCCCGGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn insert_and_deletion_get_all() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("2")).unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 19,
            path_end: 31,
            strand: Strand::Forward,
        };

        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 19, 31).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn simple_insert_get_all() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 15,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 15, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 12,
            path_end: 17,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 12, 17).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 10,
            path_end: 10,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 10).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 9,
            path_end: 9,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 9, 9).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 10,
            path_end: 20,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 20).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 15,
            path_end: 25,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 15, 25).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 5,
            path_end: 35,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 5, 35).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 19,
            path_end: 31,
            strand: Strand::Forward,
        };

        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 19, 31).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };

        BlockGroup::insert_change(&conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTGGGGGGGGG".to_string(),
            ])
        );
    }

    #[test]
    fn doesnt_apply_same_insert_twice() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 7, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string()
            ])
        );
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 0,
            path_end: 0,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 0, 0).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 0,
            path_end: 0,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 0, 0).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 0,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 40,
            path_end: 40,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 40, 40).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 10,
            path_end: 11,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 11).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let insert_node_id =
            Node::create(&conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 19,
            path_end: 20,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 19, 20).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 0,
            path_end: 1,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 0, 1).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 35,
            path_end: 40,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 35, 40).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 350,
            path_end: 400,
            strand: Strand::Forward,
        };
        let after_end_region =
            ResolvedGenRegion::from_path(&conn, block_group_id, &path, 350, 400).unwrap();
        let after_end_change = BlockGroupChange {
            region: after_end_region,
            path_accession: None,
            block: deletion.clone(),
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let before_start_region =
            ResolvedGenRegion::from_path(&conn, block_group_id, &path, -300, 400).unwrap();
        let before_start_change = BlockGroupChange {
            region: before_start_region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        let res = BlockGroup::insert_change(&conn, &after_end_change);
        assert!(matches!(res, Err(BlockGroupError::ChangeOutOfBounds(_))));
        let res = BlockGroup::insert_change(&conn, &before_start_change);
        assert!(matches!(res, Err(BlockGroupError::ChangeOutOfBounds(_))));
    }

    #[test]
    fn deletion_starting_at_block_boundary() {
        let conn = get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(&conn);
        let deletion_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("")
            .save(&conn)
            .unwrap();
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 10,
            path_end: 12,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 10, 12).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
            .save(&conn)
            .unwrap();
        let deletion_node_id =
            Node::create(&conn, &deletion_sequence.hash, &HashId::convert_str("1")).unwrap();
        let deletion = PathBlock {
            node_id: deletion_node_id,
            block_sequence: deletion_sequence.get_sequence(None, None).unwrap(),
            sequence_start: 0,
            sequence_end: 0,
            path_start: 18,
            path_end: 20,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(&conn, block_group_id, &path, 18, 20).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: deletion,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(&conn, &change).unwrap();

        let all_sequences = BlockGroup::get_all_sequences(&conn, &block_group_id, false).unwrap();
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
        let (block_group_id, _path) = setup_block_group(conn);
        let _new_sample = Sample::get_or_create(
            conn,
            NewSample {
                name: "child",
                ..Default::default()
            },
        )
        .unwrap();
        let new_bg_id = get_single_bg_id(conn, "test", "child", "chr1", vec!["test".to_string()]);
        let _new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![new_bg_id],
        );
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn)
            .unwrap();
        let insert_node_id = Node::create(
            conn,
            &insert_sequence.hash,
            &HashId::convert_str("insert-node"),
        )
        .unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let bg = BlockGroup::get_by_id(conn, &new_bg_id).unwrap();
        let region = ResolvedGenRegion {
            block_group: bg,
            path: None,
            accession: None,
            annotation: None,
            kind: ResolvedRegionKind::BlockGroup,
            anchor_start: 0,
            anchor_end: 0,
            feature_length: 0,
            start: 7,
            end: 15,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: true,
        };
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, &change).unwrap();

        let tree = BlockGroup::intervaltree_for(conn, &block_group_id, false).unwrap();
        let tree2 = BlockGroup::intervaltree_for(conn, &block_group_id, true).unwrap();
        interval_tree_verify(
            &tree,
            3,
            &[NodeIntervalBlock {
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
                node_id: HashId::convert_str("test-g-node"),
                start: 30,
                end: 40,
                sequence_start: 0,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );

        // This blockgroup has a change from positions 7-15 of 4 base pairs -- so any changes after this will be ambiguous
        let tree = BlockGroup::intervaltree_for(conn, &new_bg_id, false).unwrap();
        let tree2 = BlockGroup::intervaltree_for(conn, &new_bg_id, true).unwrap();
        interval_tree_verify(
            &tree,
            3,
            &[NodeIntervalBlock {
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
                    node_id: HashId::convert_str("test-g-node"),
                    start: 26,
                    end: 36,
                    sequence_start: 0,
                    sequence_end: 10,
                    strand: Strand::Forward,
                },
                NodeIntervalBlock {
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
                    node_id: HashId::convert_str("insert-node"),
                    start: 7,
                    end: 11,
                    sequence_start: 0,
                    sequence_end: 4,
                    strand: Strand::Forward,
                },
                NodeIntervalBlock {
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
        let _new_sample = Sample::get_or_create(
            conn,
            NewSample {
                name: "child",
                ..Default::default()
            },
        )
        .unwrap();
        let new_bg_id = get_single_bg_id(conn, "test", "child", "chr1", vec!["test".to_string()]);
        let new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(new_bg_id)),
        );
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn)
            .unwrap();
        let insert_node_id =
            Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let region = ResolvedGenRegion::from_path(conn, new_bg_id, &new_path[0], 7, 15).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: false,
        };
        BlockGroup::insert_change(conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_bg_id, true).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAAAAAANNNNTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),])
        );

        // Now, we make a change against another descendant
        let _new_sample = Sample::get_or_create(
            conn,
            NewSample {
                name: "grandchild",
                ..Default::default()
            },
        )
        .unwrap();
        let gc_bg_id = get_single_bg_id(
            conn,
            "test",
            "grandchild",
            "chr1",
            vec!["child".to_string()],
        );
        let _new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(gc_bg_id)),
        );

        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 15,
            strand: Strand::Forward,
        };
        let gc_bg = BlockGroup::get_by_id(conn, &gc_bg_id).unwrap();
        let gc_region = ResolvedGenRegion {
            block_group: gc_bg,
            path: None,
            accession: None,
            annotation: None,
            kind: ResolvedRegionKind::BlockGroup,
            anchor_start: 0,
            anchor_end: 0,
            feature_length: 0,
            start: 7,
            end: 15,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: true,
        };
        let change = BlockGroupChange {
            region: gc_region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: false,
        };
        BlockGroup::insert_change(conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &gc_bg_id, true).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec!["AAAAAAANNNNTCCCCCCCCCCGGGGGGGGGG".to_string(),])
        );
    }

    #[test]
    fn test_changes_against_derivative_diploid_blockgroups() {
        // This test ensures that if we have heterozygous changes that do not introduce frameshifts,
        // we can modify regions downstream of them.
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, _path) = setup_block_group(conn);
        let _new_sample = Sample::get_or_create(
            conn,
            NewSample {
                name: "child",
                ..Default::default()
            },
        )
        .unwrap();
        let new_bg_id = get_single_bg_id(conn, "test", "child", "chr1", vec!["test".to_string()]);
        let _new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![new_bg_id],
        );
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn)
            .unwrap();
        let insert_node_id =
            Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 11,
            strand: Strand::Forward,
        };
        let bg = BlockGroup::get_by_id(conn, &new_bg_id).unwrap();
        let region = ResolvedGenRegion {
            block_group: bg,
            path: None,
            accession: None,
            annotation: None,
            kind: ResolvedRegionKind::BlockGroup,
            anchor_start: 0,
            anchor_end: 0,
            feature_length: 0,
            start: 7,
            end: 11,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: true,
        };
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_bg_id, true).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );

        // Now, we make a change against another descendant
        let _new_sample = Sample::get_or_create(
            conn,
            NewSample {
                name: "grandchild",
                ..Default::default()
            },
        )
        .unwrap();
        let gc_bg_id = get_single_bg_id(
            conn,
            "test",
            "grandchild",
            "chr1",
            vec!["child".to_string()],
        );
        let _new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            params![gc_bg_id],
        );

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn)
            .unwrap();
        let insert_node_id = Node::create(
            conn,
            &insert_sequence.hash,
            &HashId::convert_str("new-hash"),
        )
        .unwrap();

        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 20,
            path_end: 24,
            strand: Strand::Forward,
        };
        let gc_bg = BlockGroup::get_by_id(conn, &gc_bg_id).unwrap();
        let gc_region = ResolvedGenRegion {
            block_group: gc_bg,
            path: None,
            accession: None,
            annotation: None,
            kind: ResolvedRegionKind::BlockGroup,
            anchor_start: 0,
            anchor_end: 0,
            feature_length: 0,
            start: 20,
            end: 24,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: true,
        };
        let change = BlockGroupChange {
            region: gc_region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &gc_bg_id, true).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAANNNNTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTNNNNCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAANNNNTTTTTTTTTNNNNCCCCCCGGGGGGGGGG".to_string()
            ])
        );
    }

    #[test]
    #[should_panic]
    fn test_prohibits_out_of_frame_changes_against_derivative_diploid_blockgroups() {
        // This test ensures that we do not allow ambiguous changes by coordinates
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, _path) = setup_block_group(conn);
        let _new_sample = Sample::get_or_create(
            conn,
            NewSample {
                name: "child",
                ..Default::default()
            },
        )
        .unwrap();
        let new_bg_id = get_single_bg_id(conn, "test", "child", "chr1", vec!["test".to_string()]);
        let _new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(new_bg_id)),
        );
        // This is a heterozygous replacement of 5 bases with 4 bases, so positions
        // downstream of this are not addressable.
        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn)
            .unwrap();
        let insert_node_id =
            Node::create(conn, &insert_sequence.hash, &HashId::convert_str("1")).unwrap();
        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 7,
            path_end: 12,
            strand: Strand::Forward,
        };
        let bg = BlockGroup::get_by_id(conn, &new_bg_id).unwrap();
        let region = ResolvedGenRegion {
            block_group: bg,
            path: None,
            accession: None,
            annotation: None,
            kind: ResolvedRegionKind::BlockGroup,
            anchor_start: 0,
            anchor_end: 0,
            feature_length: 0,
            start: 7,
            end: 12,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: true,
        };
        let change = BlockGroupChange {
            region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, &change).unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_bg_id, true).unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAANNNNTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
            ])
        );

        // Now, we make a change against another descendant and get an error
        let _new_sample = Sample::get_or_create(
            conn,
            NewSample {
                name: "grandchild",
                ..Default::default()
            },
        )
        .unwrap();
        let gc_bg_id = get_single_bg_id(
            conn,
            "test",
            "grandchild",
            "chr1",
            vec!["child".to_string()],
        );
        let _new_path = Path::query(
            conn,
            "select * from paths where block_group_id = ?1",
            rusqlite::params!(SQLValue::from(gc_bg_id)),
        );

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("NNNN")
            .save(conn)
            .unwrap();
        let insert_node_id =
            Node::create(conn, &insert_sequence.hash, &HashId::pad_str("new-hash")).unwrap();

        let insert = PathBlock {
            node_id: insert_node_id,
            block_sequence: insert_sequence.get_sequence(0, 4).unwrap(),
            sequence_start: 0,
            sequence_end: 4,
            path_start: 20,
            path_end: 24,
            strand: Strand::Forward,
        };
        let gc_bg = BlockGroup::get_by_id(conn, &gc_bg_id).unwrap();
        let gc_region = ResolvedGenRegion {
            block_group: gc_bg,
            path: None,
            accession: None,
            annotation: None,
            kind: ResolvedRegionKind::BlockGroup,
            anchor_start: 0,
            anchor_end: 0,
            feature_length: 0,
            start: 20,
            end: 24,
            start_anchors: None,
            end_anchors: None,
            remove_ambiguous_positions: true,
        };
        let change = BlockGroupChange {
            region: gc_region,
            path_accession: None,
            block: insert,
            chromosome_index: 1,
            phased: 0,
            preserve_edge: true,
        };
        BlockGroup::insert_change(conn, &change).unwrap();
    }

    mod test_derive_subgraph {

        use super::*;
        use crate::{
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
            let (block_group1_id, original_path) = setup_block_group(conn);

            let intervaltree = original_path.intervaltree(conn).unwrap();
            let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
            let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

            let insert_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAAAAA")
                .save(conn)
                .unwrap();
            let insert_node_id = Node::create(
                conn,
                &insert_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-a-node.{}",
                    insert_sequence.hash
                ))),
            )
            .unwrap();
            let edge_into_insert = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            let edge_out_of_insert = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_1 = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_start_node_id,
                6,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_2 = Edge::create(
                conn,
                insert_end_node_id,
                4,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();

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

            let insert_path = original_path
                .new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert)
                .unwrap();
            assert_eq!(
                insert_path.sequence(conn).unwrap(),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
            );

            let all_sequences =
                BlockGroup::get_all_sequences(conn, &block_group1_id, false).unwrap();
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

            let block_group2 = create_bg(conn, "test", "test", "chr1.1");
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
            )
            .unwrap();
            let all_sequences2 =
                BlockGroup::get_all_sequences(conn, &block_group2.id, false).unwrap();
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
            let (block_group1_id, original_path) = setup_block_group(conn);

            let intervaltree = original_path.intervaltree(conn).unwrap();
            let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
            let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

            let insert_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAAAAA")
                .save(conn)
                .unwrap();
            let insert_node_id = Node::create(
                conn,
                &insert_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-a-node.{}",
                    insert_sequence.hash
                ))),
            )
            .unwrap();
            let edge_into_insert = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            let edge_out_of_insert = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_1 = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_start_node_id,
                6,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_2 = Edge::create(
                conn,
                insert_end_node_id,
                4,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();

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

            let insert_path = original_path
                .new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert)
                .unwrap();
            assert_eq!(
                insert_path.sequence(conn).unwrap(),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
            );

            let insert2_start_node_id = intervaltree.query_point(28).next().unwrap().value.node_id;
            let insert2_end_node_id = intervaltree.query_point(32).next().unwrap().value.node_id;

            let insert2_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("TTTTTTTT")
                .save(conn)
                .unwrap();
            let insert2_node_id = Node::create(
                conn,
                &insert2_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-t-node.{}",
                    insert2_sequence.hash
                ))),
            )
            .unwrap();
            let edge_into_insert2 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            let edge_out_of_insert2 = Edge::create(
                conn,
                insert2_node_id,
                8,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_1 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_start_node_id,
                6,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_2 = Edge::create(
                conn,
                insert2_end_node_id,
                4,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();

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

            let insert2_path = insert_path
                .new_path_with(conn, 28, 32, &edge_into_insert2, &edge_out_of_insert2)
                .unwrap();
            assert_eq!(
                insert2_path.sequence(conn).unwrap(),
                "AAAAAAAAAATTTTTTAAAAAAAACCTTTTTTTTGGGGGG"
            );

            let all_sequences =
                BlockGroup::get_all_sequences(conn, &block_group1_id, false).unwrap();
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

            let block_group2 = create_bg(conn, "test", "test", "chr1.1");
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
            )
            .unwrap();
            let all_sequences2 =
                BlockGroup::get_all_sequences(conn, &block_group2.id, false).unwrap();
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
            let (block_group1_id, original_path) = setup_block_group(conn);

            let intervaltree = original_path.intervaltree(conn).unwrap();
            let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
            let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

            let insert_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("AAAAAAAA")
                .save(conn)
                .unwrap();
            let insert_node_id = Node::create(
                conn,
                &insert_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-a-node.{}",
                    insert_sequence.hash
                ))),
            )
            .unwrap();
            let edge_into_insert = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            let edge_out_of_insert = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_1 = Edge::create(
                conn,
                insert_start_node_id,
                6,
                Strand::Forward,
                insert_start_node_id,
                6,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_2 = Edge::create(
                conn,
                insert_end_node_id,
                4,
                Strand::Forward,
                insert_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();

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

            let insert_path = original_path
                .new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert)
                .unwrap();
            assert_eq!(
                insert_path.sequence(conn).unwrap(),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
            );

            let insert2_start_node_id = intervaltree.query_point(28).next().unwrap().value.node_id;
            let insert2_end_node_id = intervaltree.query_point(32).next().unwrap().value.node_id;

            let insert2_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence("TTTTTTTT")
                .save(conn)
                .unwrap();
            let insert2_node_id = Node::create(
                conn,
                &insert2_sequence.hash,
                &HashId(calculate_hash(&format!(
                    "test-insert-t-node.{}",
                    insert2_sequence.hash
                ))),
            )
            .unwrap();
            let edge_into_insert2 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_node_id,
                0,
                Strand::Forward,
            )
            .unwrap();
            let edge_out_of_insert2 = Edge::create(
                conn,
                insert2_node_id,
                8,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_1 = Edge::create(
                conn,
                insert2_start_node_id,
                6,
                Strand::Forward,
                insert2_start_node_id,
                6,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_2 = Edge::create(
                conn,
                insert2_end_node_id,
                4,
                Strand::Forward,
                insert2_end_node_id,
                4,
                Strand::Forward,
            )
            .unwrap();

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

            let insert2_path = insert_path
                .new_path_with(conn, 28, 32, &edge_into_insert2, &edge_out_of_insert2)
                .unwrap();
            assert_eq!(
                insert2_path.sequence(conn).unwrap(),
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
            )
            .unwrap();
            let ref_heal_1 = Edge::create(
                conn,
                insert_node_id,
                8,
                Strand::Forward,
                insert_node_id,
                8,
                Strand::Forward,
            )
            .unwrap();
            let ref_heal_2 = Edge::create(
                conn,
                deletion_end_node_id,
                8,
                Strand::Forward,
                deletion_end_node_id,
                8,
                Strand::Forward,
            )
            .unwrap();
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

            let all_sequences =
                BlockGroup::get_all_sequences(conn, &block_group1_id, false).unwrap();
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

            let block_group2 = create_bg(conn, "test", "test", "chr1.1");
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
            )
            .unwrap();
            let all_sequences2 =
                BlockGroup::get_all_sequences(conn, &block_group2.id, false).unwrap();
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
