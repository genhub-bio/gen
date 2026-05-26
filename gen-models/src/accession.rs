use std::collections::HashSet;

use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, calculate_hash,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use intervaltree::IntervalTree;
use itertools::Itertools;
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group::{
        AccessionChange, BlockGroup, BlockGroupTreeSource, IntervalTreeSource,
        ResolvedBlockGroupChange,
    },
    block_group_edge::{AugmentedEdgeData, BlockGroupEdge},
    db::GraphConnection,
    edge::EdgeData,
    gen_models_capnp::{accession, accession_edge, accession_path},
    traits::*,
};

#[derive(Clone, Deserialize, Serialize, Debug, Eq, PartialEq)]
pub struct Accession {
    pub id: HashId,
    pub name: String,
    pub block_group_id: HashId,
    pub parent_accession_id: Option<HashId>,
}

#[derive(Debug, Error, PartialEq)]
pub enum AccessionError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Block group error: {0}")]
    BlockGroup(String),
    #[error("Duplicate entry with uuid: {0}")]
    Duplicate(String),
    #[error("Accession {0} has no edges in accession_paths")]
    MissingPath(HashId),
    #[error("Accession not found: {0}")]
    NotFound(String),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum AnchorPoint {
    PathStart,
    Node { node_id: HashId, coordinate: i64 },
}

impl<'a> Capnp<'a> for Accession {
    type Builder = accession::Builder<'a>;
    type Reader = accession::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_name(&self.name);
        builder.set_block_group_id(&self.block_group_id.0).unwrap();
        match &self.parent_accession_id {
            None => {
                builder.reborrow().get_parent_accession_id().set_none(());
            }
            Some(n) => {
                builder
                    .reborrow()
                    .get_parent_accession_id()
                    .set_some(&n.0)
                    .unwrap();
            }
        }
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let name = reader.get_name().unwrap().to_string().unwrap();
        let block_group_id = reader
            .get_block_group_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let parent_accession_id: Option<HashId> =
            match reader.get_parent_accession_id().which().unwrap() {
                accession::parent_accession_id::None(()) => None,
                accession::parent_accession_id::Some(n) => {
                    Some(n.unwrap().as_slice().unwrap().try_into().unwrap())
                }
            };

        Accession {
            id,
            name,
            block_group_id,
            parent_accession_id,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash)]
pub struct AccessionEdge {
    pub id: HashId,
    pub source_node_id: HashId,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: HashId,
    pub target_coordinate: i64,
    pub target_strand: Strand,
    pub chromosome_index: i64,
}

#[derive(Debug, Error, PartialEq)]
pub enum AccessionEdgeError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl<'a> Capnp<'a> for AccessionEdge {
    type Builder = accession_edge::Builder<'a>;
    type Reader = accession_edge::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_source_node_id(&self.source_node_id.0).unwrap();
        builder.set_source_coordinate(self.source_coordinate);
        builder.set_source_strand(self.source_strand.into());
        builder.set_target_node_id(&self.target_node_id.0).unwrap();
        builder.set_target_coordinate(self.target_coordinate);
        builder.set_target_strand(self.target_strand.into());
        builder.set_chromosome_index(self.chromosome_index);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let source_node_id = reader
            .get_source_node_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let source_coordinate = reader.get_source_coordinate();
        let source_strand = reader.get_source_strand().unwrap().into();
        let target_node_id = reader
            .get_target_node_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let target_coordinate = reader.get_target_coordinate();
        let target_strand = reader.get_target_strand().unwrap().into();
        let chromosome_index = reader.get_chromosome_index();

        AccessionEdge {
            id,
            source_node_id,
            source_coordinate,
            source_strand,
            target_node_id,
            target_coordinate,
            target_strand,
            chromosome_index,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq)]
pub struct AccessionPath {
    pub id: HashId,
    pub accession_id: HashId,
    pub index_in_path: i64,
    pub edge_id: HashId,
}

#[derive(Debug, Error, PartialEq)]
pub enum AccessionPathError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl<'a> Capnp<'a> for AccessionPath {
    type Builder = accession_path::Builder<'a>;
    type Reader = accession_path::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_accession_id(&self.accession_id.0).unwrap();
        builder.set_index_in_path(self.index_in_path);
        builder.set_edge_id(&self.edge_id.0).unwrap();
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let id = reader
            .get_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let accession_id = reader
            .get_accession_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let index_in_path = reader.get_index_in_path();
        let edge_id = reader
            .get_edge_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();

        AccessionPath {
            id,
            accession_id,
            index_in_path,
            edge_id,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AccessionEdgeData {
    pub source_node_id: HashId,
    pub source_coordinate: i64,
    pub source_strand: Strand,
    pub target_node_id: HashId,
    pub target_coordinate: i64,
    pub target_strand: Strand,
    pub chromosome_index: i64,
}

impl AccessionEdgeData {
    pub fn id_hash(&self) -> HashId {
        HashId(calculate_hash(&format!(
            "{}:{}:{}:{}:{}:{}:{}",
            self.source_node_id,
            self.source_coordinate,
            self.source_strand,
            self.target_node_id,
            self.target_coordinate,
            self.target_strand,
            self.chromosome_index
        )))
    }
}

impl From<&AccessionEdge> for AccessionEdgeData {
    fn from(item: &AccessionEdge) -> Self {
        AccessionEdgeData {
            source_node_id: item.source_node_id,
            source_coordinate: item.source_coordinate,
            source_strand: item.source_strand,
            target_node_id: item.target_node_id,
            target_coordinate: item.target_coordinate,
            target_strand: item.target_strand,
            chromosome_index: item.chromosome_index,
        }
    }
}

impl From<&AugmentedEdgeData> for AccessionEdgeData {
    fn from(item: &AugmentedEdgeData) -> Self {
        AccessionEdgeData {
            source_node_id: item.edge_data.source_node_id,
            source_coordinate: item.edge_data.source_coordinate,
            source_strand: item.edge_data.source_strand,
            target_node_id: item.edge_data.target_node_id,
            target_coordinate: item.edge_data.target_coordinate,
            target_strand: item.edge_data.target_strand,
            chromosome_index: item.chromosome_index,
        }
    }
}

impl Accession {
    fn id_hash(
        block_group_id: &HashId,
        parent_accession_id: Option<&HashId>,
        name: &str,
    ) -> HashId {
        HashId(calculate_hash(&format!(
            "{block_group_id}:{parent_accession_id:?}:{name}"
        )))
    }

    pub fn create(
        conn: &GraphConnection,
        name: &str,
        block_group_id: &HashId,
        parent_accession_id: Option<&HashId>,
    ) -> Result<Accession, AccessionError> {
        let query = "INSERT INTO accessions (id, name, block_group_id, parent_accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = match conn.prepare(query) {
            Ok(s) => s,
            Err(e) => return Err(AccessionError::DatabaseError(e)),
        };

        let hash = Accession::id_hash(block_group_id, parent_accession_id, name);
        match stmt.execute((hash, name, block_group_id, parent_accession_id)) {
            Ok(_) => Ok(Accession {
                id: hash,
                name: name.to_string(),
                block_group_id: *block_group_id,
                parent_accession_id: parent_accession_id.copied(),
            }),
            Err(rusqlite::Error::SqliteFailure(err, _details))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(AccessionError::Duplicate(format!(
                    "An accession with the same name, block_group_id, and parent_accession_id already exists. name: {name}, block_group_id: {block_group_id}, parent_accession_id: {parent_accession_id:?}"
                )))
            }
            Err(e) => Err(AccessionError::DatabaseError(e)),
        }
    }

    pub fn get_or_create(
        conn: &GraphConnection,
        name: &str,
        block_group_id: &HashId,
        parent_accession_id: Option<&HashId>,
    ) -> Result<Accession, AccessionError> {
        match Accession::create(conn, name, block_group_id, parent_accession_id) {
            Ok(accession) => Ok(accession),
            Err(AccessionError::Duplicate(_)) => {
                let hash = Accession::id_hash(block_group_id, parent_accession_id, name);
                Ok(Accession {
                    id: hash,
                    name: name.to_string(),
                    block_group_id: *block_group_id,
                    parent_accession_id: parent_accession_id.copied(),
                })
            }
            Err(e) => Err(e),
        }
    }

    pub fn get_edges_by_id(conn: &GraphConnection, accession_id: &HashId) -> Vec<AccessionEdge> {
        let query = "\
            select ae.* \
            from accession_edges ae \
            join accession_paths ap on ap.edge_id = ae.id \
            where ap.accession_id = ?1 \
            order by ap.index_in_path;";
        AccessionEdge::query(conn, query, params![accession_id])
    }

    pub fn blocks(&self, conn: &GraphConnection) -> Result<Vec<NodeIntervalBlock>, AccessionError> {
        let edges = Self::get_edges_by_id(conn, &self.id);
        if edges.is_empty() {
            return Err(AccessionError::MissingPath(self.id));
        }

        let mut offset = 0;
        let mut blocks = vec![NodeIntervalBlock {
            node_id: PATH_START_NODE_ID,
            start: i64::MIN + 1,
            end: 0,
            sequence_start: 0,
            sequence_end: 0,
            strand: Strand::Forward,
        }];

        for (into, out_of) in edges.iter().tuple_windows() {
            let block_len = out_of.source_coordinate - into.target_coordinate;
            blocks.push(NodeIntervalBlock {
                node_id: into.target_node_id,
                start: offset,
                end: offset + block_len,
                sequence_start: into.target_coordinate,
                sequence_end: out_of.source_coordinate,
                strand: into.target_strand,
            });
            offset += block_len;
        }

        blocks.push(NodeIntervalBlock {
            node_id: PATH_END_NODE_ID,
            start: offset,
            end: i64::MAX - 1,
            sequence_start: 0,
            sequence_end: 0,
            strand: Strand::Forward,
        });

        Ok(blocks)
    }

    pub fn length(&self, conn: &GraphConnection) -> Result<i64, AccessionError> {
        let blocks = self.blocks(conn)?;
        Ok(blocks.last().unwrap().start)
    }

    pub fn intervaltree(
        &self,
        conn: &GraphConnection,
    ) -> Result<IntervalTree<i64, NodeIntervalBlock>, AccessionError> {
        Ok(self
            .blocks(conn)?
            .into_iter()
            .map(|block| (block.start..block.end, block))
            .collect())
    }

    pub fn coordinate_to_block_group(
        &self,
        conn: &GraphConnection,
        target_block_group_id: &HashId,
        coordinate: i64,
    ) -> Result<i64, AccessionError> {
        let tree = BlockGroup::intervaltree_for(conn, target_block_group_id, false);
        let interval_blocks = tree
            .iter_sorted()
            .map(|entry| entry.value)
            .collect::<Vec<_>>();
        let accession_blocks = self.blocks(conn)?;
        let accession_length = self.length(conn)?;
        let accession_start = accession_blocks
            .iter()
            .find(|block| block.start == 0)
            .ok_or_else(|| AccessionError::NotFound(self.name.clone()))?;
        let accession_end = accession_blocks
            .iter()
            .find(|block| block.end == accession_length)
            .ok_or_else(|| AccessionError::NotFound(self.name.clone()))?;

        if coordinate < 0 {
            return Ok(block_sequence_coordinate_to_interval(
                &interval_blocks,
                accession_start,
                accession_start.sequence_start,
                &self.name,
            )? + coordinate);
        }
        if coordinate > accession_length {
            return Ok(block_sequence_coordinate_to_interval(
                &interval_blocks,
                accession_end,
                accession_end.sequence_end,
                &self.name,
            )? + (coordinate - accession_length));
        }

        let (block, sequence_coordinate) = {
            let block = accession_blocks
                .iter()
                .find(|block| {
                    block.start <= coordinate
                        && coordinate <= block.end
                        && block.node_id != PATH_START_NODE_ID
                        && block.node_id != PATH_END_NODE_ID
                })
                .ok_or_else(|| AccessionError::NotFound(self.name.clone()))?;
            (block, block.sequence_start + (coordinate - block.start))
        };

        block_sequence_coordinate_to_interval(
            &interval_blocks,
            block,
            sequence_coordinate,
            &self.name,
        )
    }

    pub fn coordinates_to_block_group(
        &self,
        conn: &GraphConnection,
        target_block_group_id: &HashId,
        start: i64,
        end: i64,
    ) -> Result<(i64, i64), AccessionError> {
        Ok((
            self.coordinate_to_block_group(conn, target_block_group_id, start)?,
            self.coordinate_to_block_group(conn, target_block_group_id, end)?,
        ))
    }

    pub fn query_target_blocks(
        &self,
        conn: &GraphConnection,
        target_block_group_id: &HashId,
        coordinate: i64,
    ) -> Result<Vec<NodeIntervalBlock>, AccessionError> {
        let mapped_coordinate =
            self.coordinate_to_block_group(conn, target_block_group_id, coordinate)?;
        let tree = BlockGroup::intervaltree_for(conn, target_block_group_id, false);
        Ok(tree
            .query_point(mapped_coordinate)
            .map(|entry| entry.value)
            .collect())
    }

    pub fn resolve_block_group_change(
        &self,
        conn: &GraphConnection,
        change: &AccessionChange,
    ) -> Result<ResolvedBlockGroupChange, AccessionError> {
        if let Some(new_edges) = insert_change_from_negative_start(self, conn, change)? {
            return Ok(ResolvedBlockGroupChange::DirectEdges(new_edges));
        }

        let (start, end) = self.coordinates_to_block_group(
            conn,
            &change.block_group_id,
            change.start,
            change.end,
        )?;
        let tree = BlockGroupTreeSource {
            block_group_id: change.block_group_id,
            remove_ambiguous_positions: false,
        }
        .intervaltree(conn)
        .map_err(|err| AccessionError::BlockGroup(err.to_string()))?;
        let use_block_group_insert = tree.query_point(start).count() == 1
            && tree.query_point(start - 1).count() == 1
            && tree.query_point(end).count() == 1;

        if use_block_group_insert {
            return Ok(ResolvedBlockGroupChange::Interval { tree, start, end });
        }

        if let Some(new_edges) = insert_ambiguous_start_change(conn, change, start, end)? {
            return Ok(ResolvedBlockGroupChange::DirectEdges(new_edges));
        }

        Err(AccessionError::NotFound(self.name.clone()))
    }
}

fn block_sequence_coordinate_to_interval(
    interval_blocks: &[NodeIntervalBlock],
    block: &NodeIntervalBlock,
    sequence_coordinate: i64,
    accession_name: &str,
) -> Result<i64, AccessionError> {
    let interval_block = interval_blocks
        .iter()
        .find(|interval_block| {
            interval_block.node_id == block.node_id
                && interval_block.sequence_start <= sequence_coordinate
                && sequence_coordinate <= interval_block.sequence_end
        })
        .ok_or_else(|| AccessionError::NotFound(accession_name.to_string()))?;

    Ok(interval_block.start + (sequence_coordinate - interval_block.sequence_start))
}

fn insert_ambiguous_start_change(
    conn: &GraphConnection,
    change: &AccessionChange,
    start: i64,
    end: i64,
) -> Result<Option<Vec<AugmentedEdgeData>>, AccessionError> {
    if change.block.sequence_start == change.block.sequence_end {
        return Ok(None);
    }

    let tree = BlockGroup::intervaltree_for(conn, &change.block_group_id, false);
    let start_blocks = tree
        .query_point(start)
        .map(|entry| entry.value)
        .collect::<Vec<_>>();
    let end_blocks = tree
        .query_point(end)
        .map(|entry| entry.value)
        .collect::<Vec<_>>();

    if start != 0 || start_blocks.len() <= 1 || end_blocks.len() != 1 {
        return Ok(None);
    }

    let end_block = end_blocks[0];
    let end_coordinate = end - end_block.start + end_block.sequence_start;
    Ok(Some(vec![
        AugmentedEdgeData {
            edge_data: EdgeData {
                source_node_id: PATH_START_NODE_ID,
                source_coordinate: 0,
                source_strand: Strand::Forward,
                target_node_id: change.block.node_id,
                target_coordinate: change.block.sequence_start,
                target_strand: Strand::Forward,
            },
            chromosome_index: change.chromosome_index,
            phased: change.phased,
        },
        AugmentedEdgeData {
            edge_data: EdgeData {
                source_node_id: change.block.node_id,
                source_coordinate: change.block.sequence_end,
                source_strand: Strand::Forward,
                target_node_id: end_block.node_id,
                target_coordinate: end_coordinate,
                target_strand: Strand::Forward,
            },
            chromosome_index: change.chromosome_index,
            phased: change.phased,
        },
    ]))
}

fn insert_change_from_negative_start(
    accession: &Accession,
    conn: &GraphConnection,
    change: &AccessionChange,
) -> Result<Option<Vec<AugmentedEdgeData>>, AccessionError> {
    if change.start >= 0 || change.block.sequence_start == change.block.sequence_end {
        return Ok(None);
    }

    let accession_blocks = accession.blocks(conn)?;
    let accession_length = accession.length(conn)?;
    let start_block = accession_blocks
        .iter()
        .find(|block| block.start == 0 && block.node_id != PATH_START_NODE_ID)
        .ok_or_else(|| AccessionError::NotFound(accession.name.clone()))?;
    let start_points = walk_back_to_anchor_points(
        conn,
        &change.block_group_id,
        start_block.node_id,
        start_block.sequence_start,
        -change.start,
    )?;
    let end_points = if change.start == change.end {
        points_after_anchor(conn, &change.block_group_id, &start_points)?
    } else {
        vec![
            coordinate_to_anchor_point(
                &accession_blocks,
                accession_length,
                change.end,
                &accession.name,
            )?
            .ok_or_else(|| AccessionError::NotFound(accession.name.clone()))?,
        ]
    };

    let mut new_edges = Vec::new();
    for start_point in start_points {
        match start_point {
            AnchorPoint::PathStart => new_edges.push(AugmentedEdgeData {
                edge_data: EdgeData {
                    source_node_id: PATH_START_NODE_ID,
                    source_coordinate: 0,
                    source_strand: Strand::Forward,
                    target_node_id: change.block.node_id,
                    target_coordinate: change.block.sequence_start,
                    target_strand: Strand::Forward,
                },
                chromosome_index: change.chromosome_index,
                phased: change.phased,
            }),
            AnchorPoint::Node {
                node_id,
                coordinate,
            } => new_edges.push(AugmentedEdgeData {
                edge_data: EdgeData {
                    source_node_id: node_id,
                    source_coordinate: coordinate,
                    source_strand: Strand::Forward,
                    target_node_id: change.block.node_id,
                    target_coordinate: change.block.sequence_start,
                    target_strand: Strand::Forward,
                },
                chromosome_index: change.chromosome_index,
                phased: change.phased,
            }),
        }
    }
    for end_point in end_points {
        match end_point {
            AnchorPoint::PathStart => {
                return Err(AccessionError::NotFound(accession.name.clone()));
            }
            AnchorPoint::Node {
                node_id,
                coordinate,
            } => new_edges.push(AugmentedEdgeData {
                edge_data: EdgeData {
                    source_node_id: change.block.node_id,
                    source_coordinate: change.block.sequence_end,
                    source_strand: Strand::Forward,
                    target_node_id: node_id,
                    target_coordinate: coordinate,
                    target_strand: Strand::Forward,
                },
                chromosome_index: change.chromosome_index,
                phased: change.phased,
            }),
        }
    }

    Ok(Some(new_edges))
}

fn points_after_anchor(
    conn: &GraphConnection,
    target_block_group_id: &HashId,
    anchor_points: &[AnchorPoint],
) -> Result<Vec<AnchorPoint>, AccessionError> {
    let edges = BlockGroupEdge::edges_for_block_group(conn, target_block_group_id);
    let mut points = HashSet::new();
    for anchor in anchor_points {
        match anchor {
            AnchorPoint::PathStart => {
                for edge in edges
                    .iter()
                    .filter(|edge| edge.edge.source_node_id == PATH_START_NODE_ID)
                {
                    points.insert(AnchorPoint::Node {
                        node_id: edge.edge.target_node_id,
                        coordinate: edge.edge.target_coordinate,
                    });
                }
            }
            AnchorPoint::Node {
                node_id,
                coordinate,
            } => {
                points.insert(AnchorPoint::Node {
                    node_id: *node_id,
                    coordinate: *coordinate,
                });
            }
        }
    }
    Ok(points.into_iter().collect())
}

fn coordinate_to_anchor_point(
    accession_blocks: &[NodeIntervalBlock],
    accession_length: i64,
    coordinate: i64,
    accession_name: &str,
) -> Result<Option<AnchorPoint>, AccessionError> {
    if !(0..=accession_length).contains(&coordinate) {
        return Ok(None);
    }
    let block = accession_blocks
        .iter()
        .find(|block| {
            block.start <= coordinate
                && coordinate <= block.end
                && block.node_id != PATH_START_NODE_ID
                && block.node_id != PATH_END_NODE_ID
        })
        .ok_or_else(|| AccessionError::NotFound(accession_name.to_string()))?;
    Ok(Some(AnchorPoint::Node {
        node_id: block.node_id,
        coordinate: block.sequence_start + (coordinate - block.start),
    }))
}

fn walk_back_to_anchor_points(
    conn: &GraphConnection,
    target_block_group_id: &HashId,
    node_id: HashId,
    coordinate: i64,
    remaining: i64,
) -> Result<Vec<AnchorPoint>, AccessionError> {
    let edges = BlockGroupEdge::edges_for_block_group(conn, target_block_group_id);
    let mut points = HashSet::new();
    collect_backtrack_points(&edges, node_id, coordinate, remaining, &mut points);
    Ok(points.into_iter().collect())
}

fn collect_backtrack_points(
    edges: &[crate::block_group_edge::AugmentedEdge],
    node_id: HashId,
    coordinate: i64,
    remaining: i64,
    out: &mut HashSet<AnchorPoint>,
) {
    if remaining == 0 {
        out.insert(AnchorPoint::Node {
            node_id,
            coordinate,
        });
        return;
    }
    if remaining < coordinate {
        out.insert(AnchorPoint::Node {
            node_id,
            coordinate: coordinate - remaining,
        });
        return;
    }

    let remainder = remaining - coordinate;
    let incoming = edges
        .iter()
        .filter(|edge| edge.edge.target_node_id == node_id && edge.edge.target_coordinate == 0)
        .collect::<Vec<_>>();
    if incoming.is_empty() {
        if remainder == 0 {
            out.insert(AnchorPoint::PathStart);
        }
        return;
    }

    for edge in incoming {
        if edge.edge.source_node_id == PATH_START_NODE_ID {
            if remainder == 0 {
                out.insert(AnchorPoint::PathStart);
            }
        } else {
            collect_backtrack_points(
                edges,
                edge.edge.source_node_id,
                edge.edge.source_coordinate,
                remainder,
                out,
            );
        }
    }
}

impl RegionResolver for Accession {
    type Connection = GraphConnection;
    type Error = AccessionError;

    fn resolve(
        region: &Region,
        conn: &Self::Connection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<Self, RegionResolutionError<Self::Error>> {
        let matches = Accession::query(
            conn,
            "SELECT a.* \
             FROM accessions a \
             JOIN block_groups bg ON a.block_group_id = bg.id \
             WHERE bg.collection_name = ?1 \
               AND bg.sample_name = ?2 \
               AND lower(a.name) = lower(?3)",
            params![collection_name, sample_name, region.name],
        );

        match matches.len() {
            0 => Err(RegionResolutionError::NotFound(region.name.clone())),
            1 => Ok(matches.into_iter().next().unwrap()),
            _ => Err(RegionResolutionError::Ambiguous(format!(
                "multiple accessions named {}",
                region.name
            ))),
        }
    }
}

impl Query for Accession {
    type Model = Accession;

    const TABLE_NAME: &'static str = "accessions";

    fn process_row(row: &Row) -> Self::Model {
        Accession {
            id: row.get(0).unwrap(),
            name: row.get(1).unwrap(),
            block_group_id: row.get(2).unwrap(),
            parent_accession_id: row.get(3).unwrap(),
        }
    }
}

impl AccessionEdge {
    pub fn create(
        conn: &GraphConnection,
        edge: AccessionEdgeData,
    ) -> Result<AccessionEdge, AccessionEdgeError> {
        let hash = HashId(calculate_hash(&format!(
            "{}:{}:{}:{}:{}:{}:{}",
            edge.source_node_id,
            edge.source_coordinate,
            edge.source_strand,
            edge.target_node_id,
            edge.target_coordinate,
            edge.target_strand,
            edge.chromosome_index
        )));
        // TODO: handle get-or-create
        let insert_statement = "INSERT INTO accession_edges (id, source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand, chromosome_index) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);";
        let mut stmt = conn.prepare(insert_statement).unwrap();
        match stmt.execute(params![
            hash,
            edge.source_node_id,
            edge.source_coordinate,
            edge.source_strand,
            edge.target_node_id,
            edge.target_coordinate,
            edge.target_strand,
            edge.chromosome_index
        ]) {
            Ok(_) => {}
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {}
            }
            Err(err) => return Err(AccessionEdgeError::DatabaseError(err)),
        };
        Ok(AccessionEdge {
            id: hash,
            source_node_id: edge.source_node_id,
            source_coordinate: edge.source_coordinate,
            source_strand: edge.source_strand,
            target_node_id: edge.target_node_id,
            target_coordinate: edge.target_coordinate,
            target_strand: edge.target_strand,
            chromosome_index: edge.chromosome_index,
        })
    }

    pub fn bulk_create(conn: &GraphConnection, edges: &[AccessionEdgeData]) -> Vec<HashId> {
        let edge_ids = edges.iter().map(|edge| edge.id_hash()).collect::<Vec<_>>();
        let query = AccessionEdge::query_by_ids(conn, &edge_ids);
        let existing_edges = query.iter().map(|edge| &edge.id).collect::<HashSet<_>>();

        let mut edges_to_insert = HashSet::new();
        for (index, edge) in edge_ids.iter().enumerate() {
            if !existing_edges.contains(edge) {
                edges_to_insert.insert(&edges[index]);
            }
        }

        let batch_size = max_rows_per_batch(conn, 8);

        for chunk in &edges_to_insert.iter().chunks(batch_size) {
            let mut rows = vec![];
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            for edge in chunk {
                params.push(Box::new(edge.id_hash()));
                params.push(Box::new(edge.source_node_id));
                params.push(Box::new(edge.source_coordinate));
                params.push(Box::new(edge.source_strand));
                params.push(Box::new(edge.target_node_id));
                params.push(Box::new(edge.target_coordinate));
                params.push(Box::new(edge.target_strand));
                params.push(Box::new(edge.chromosome_index));
                rows.push("(?, ?, ?, ?, ?, ?, ?, ?)");
            }
            let sql = format!(
                "INSERT INTO accession_edges (id, source_node_id, source_coordinate, source_strand, target_node_id, target_coordinate, target_strand, chromosome_index) VALUES {};",
                rows.join(",")
            );
            conn.execute(&sql, rusqlite::params_from_iter(params))
                .unwrap();
        }
        edge_ids
    }

    pub fn bulk_delete(conn: &GraphConnection, edges: &[AccessionEdgeData]) {
        let ids = edges.iter().map(|e| e.id_hash()).collect::<Vec<_>>();
        AccessionEdge::delete_by_ids(conn, &ids);
    }

    pub fn to_data(edge: AccessionEdge) -> AccessionEdgeData {
        AccessionEdgeData {
            source_node_id: edge.source_node_id,
            source_coordinate: edge.source_coordinate,
            source_strand: edge.source_strand,
            target_node_id: edge.target_node_id,
            target_coordinate: edge.target_coordinate,
            target_strand: edge.target_strand,
            chromosome_index: edge.chromosome_index,
        }
    }
}

impl Query for AccessionEdge {
    type Model = AccessionEdge;

    const TABLE_NAME: &'static str = "accession_edges";

    fn process_row(row: &Row) -> Self::Model {
        AccessionEdge {
            id: row.get(0).unwrap(),
            source_node_id: row.get(1).unwrap(),
            source_coordinate: row.get(2).unwrap(),
            source_strand: row.get(3).unwrap(),
            target_node_id: row.get(4).unwrap(),
            target_coordinate: row.get(5).unwrap(),
            target_strand: row.get(6).unwrap(),
            chromosome_index: row.get(7).unwrap(),
        }
    }
}

impl AccessionPath {
    pub fn create(
        conn: &GraphConnection,
        accession_id: &HashId,
        edge_ids: &[HashId],
    ) -> Result<(), AccessionPathError> {
        let batch_size = max_rows_per_batch(conn, 4);

        for (index1, chunk) in edge_ids.chunks(batch_size).enumerate() {
            let mut rows_to_insert = vec![];
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            for (index2, edge_id) in chunk.iter().enumerate() {
                rows_to_insert.push("(?, ?, ?, ?)".to_string());
                let index_of = index1 * 100000 + index2;
                let hash = HashId(calculate_hash(&format!(
                    "{accession_id}:{edge_ids:?}:{index_of}",
                )));
                params.push(Box::new(hash));
                params.push(Box::new(accession_id));
                params.push(Box::new(edge_id));
                params.push(Box::new(index_of));
            }

            let sql = format!(
                "INSERT OR IGNORE INTO accession_paths (id, accession_id, edge_id, index_in_path) VALUES {};",
                rows_to_insert.join(", ")
            );

            let mut stmt = match conn.prepare(&sql) {
                Ok(s) => s,
                Err(e) => return Err(AccessionPathError::DatabaseError(e)),
            };
            match stmt.execute(rusqlite::params_from_iter(params)) {
                Ok(_) => {}
                Err(e) => return Err(AccessionPathError::DatabaseError(e)),
            }
        }

        Ok(())
    }
}

impl Query for AccessionPath {
    type Model = AccessionPath;

    const TABLE_NAME: &'static str = "accession_paths";

    fn process_row(row: &Row) -> AccessionPath {
        AccessionPath {
            id: row.get(0).unwrap(),
            accession_id: row.get(1).unwrap(),
            index_in_path: row.get(2).unwrap(),
            edge_id: row.get(3).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use capnp::message::TypedBuilder;
    use gen_core::{HashId, region::RegionResolutionError};

    use super::*;
    use crate::{
        block_group::{BlockGroup, PathCache},
        block_group_edge::BlockGroupEdgeData,
        path::Path,
        path_edge::PathEdge,
        test_helpers::{create_bg, get_connection, interval_tree_verify, setup_block_group},
    };

    mod region_resolver {
        use super::*;

        #[test]
        fn resolves_accession_by_name_case_insensitively() {
            let conn = &get_connection(None).unwrap();
            let (_bg, path) = setup_block_group(conn);
            let mut path_cache = PathCache::new(conn);
            let accession =
                BlockGroup::add_accession(conn, &path, "mreB", 5, 15, &mut path_cache).unwrap();

            let region = Region::parse("MREB").unwrap();
            let resolved = Accession::resolve(&region, conn, "test", "test").unwrap();
            assert_eq!(resolved.id, accession.id);
        }

        #[test]
        fn returns_not_found_for_missing_accession() {
            let conn = &get_connection(None).unwrap();
            let (_bg, _path) = setup_block_group(conn);

            let region = Region::parse("missing").unwrap();
            let err = Accession::resolve(&region, conn, "test", "test").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::NotFound(name) if name == "missing"
            ));
        }

        #[test]
        fn returns_ambiguous_for_multiple_matching_accessions() {
            let conn = &get_connection(None).unwrap();
            let (_bg, path) = setup_block_group(conn);
            let mut path_cache = PathCache::new(conn);
            let _ = BlockGroup::add_accession(conn, &path, "mreB", 5, 15, &mut path_cache).unwrap();

            let other_block_group = create_bg(conn, "test", "test", "other");
            let edge_ids = PathEdge::edges_for_path(conn, &path.id)
                .into_iter()
                .map(|edge| edge.id)
                .collect::<Vec<_>>();
            let block_group_edges = edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: other_block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>();
            crate::block_group_edge::BlockGroupEdge::bulk_create(conn, &block_group_edges);
            let other_path =
                Path::create(conn, "other-path", &other_block_group.id, &edge_ids).unwrap();
            let _ = BlockGroup::add_accession(conn, &other_path, "MREB", 5, 15, &mut path_cache)
                .unwrap();

            let region = Region::parse("mreB").unwrap();
            let err = Accession::resolve(&region, conn, "test", "test").unwrap_err();
            assert!(matches!(
                err,
                RegionResolutionError::Ambiguous(name) if name == "multiple accessions named mreB"
            ));
        }
    }

    #[test]
    fn test_accession_capnp_serialization() {
        let accession = Accession {
            id: "0000000000000000000000000000000000000000000000000000000000000200"
                .try_into()
                .unwrap(),
            name: "test_accession".to_string(),
            block_group_id: "0000000000000000000000000000000000000000000000000000000000000150"
                .try_into()
                .unwrap(),
            parent_accession_id: Some(
                "0000000000000000000000000000000000000000000000000000000000000100"
                    .try_into()
                    .unwrap(),
            ),
        };

        let mut message = TypedBuilder::<accession::Owned>::new_default();
        let mut root = message.init_root();
        accession.write_capnp(&mut root);

        let deserialized = Accession::read_capnp(root.into_reader());
        assert_eq!(accession, deserialized);
    }

    #[test]
    fn test_accession_capnp_serialization_no_parent() {
        let accession = Accession {
            id: "0000000000000000000000000000000000000000000000000000000000000201"
                .try_into()
                .unwrap(),
            name: "test_accession_2".to_string(),
            block_group_id: "0000000000000000000000000000000000000000000000000000000000000151"
                .try_into()
                .unwrap(),
            parent_accession_id: None,
        };

        let mut message = TypedBuilder::<accession::Owned>::new_default();
        let mut root = message.init_root();
        accession.write_capnp(&mut root);

        let deserialized = Accession::read_capnp(root.into_reader());
        assert_eq!(accession, deserialized);
    }

    #[test]
    fn test_accession_edge_capnp_serialization() {
        let accession_edge = AccessionEdge {
            id: "0000000000000000000000000000030000000000000000000000000000000000"
                .try_into()
                .unwrap(),
            source_node_id: HashId::convert_str("10"),
            source_coordinate: 100,
            source_strand: Strand::Forward,
            target_node_id: HashId::convert_str("20"),
            target_coordinate: 200,
            target_strand: Strand::Reverse,
            chromosome_index: 1,
        };

        let mut message = TypedBuilder::<accession_edge::Owned>::new_default();
        let mut root = message.init_root();
        accession_edge.write_capnp(&mut root);

        let deserialized = AccessionEdge::read_capnp(root.into_reader());
        assert_eq!(accession_edge, deserialized);
    }

    #[test]
    fn test_accession_create_query() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let accession = Accession::create(conn, "test", &block_group_id, None).unwrap();
        let _accession_2 = Accession::create(conn, "test2", &block_group_id, None).unwrap();
        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                params!["test"],
            ),
            vec![Accession {
                id: accession.id,
                name: "test".to_string(),
                block_group_id,
                parent_accession_id: None,
            }]
        );
    }

    #[test]
    fn test_intervaltree() {
        let conn = &get_connection(None).unwrap();
        let (_bg, path) = setup_block_group(conn);
        let mut path_cache = PathCache::new(conn);
        let accession =
            BlockGroup::add_accession(conn, &path, "test", 5, 35, &mut path_cache).unwrap();

        let tree = accession.intervaltree(conn).unwrap();
        interval_tree_verify(
            &tree,
            0,
            &[NodeIntervalBlock {
                node_id: HashId::convert_str("test-a-node"),
                start: 0,
                end: 5,
                sequence_start: 5,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree,
            5,
            &[NodeIntervalBlock {
                node_id: HashId::convert_str("test-t-node"),
                start: 5,
                end: 15,
                sequence_start: 0,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree,
            15,
            &[NodeIntervalBlock {
                node_id: HashId::convert_str("test-c-node"),
                start: 15,
                end: 25,
                sequence_start: 0,
                sequence_end: 10,
                strand: Strand::Forward,
            }],
        );
        interval_tree_verify(
            &tree,
            25,
            &[NodeIntervalBlock {
                node_id: HashId::convert_str("test-g-node"),
                start: 25,
                end: 30,
                sequence_start: 0,
                sequence_end: 5,
                strand: Strand::Forward,
            }],
        );
    }

    #[test]
    fn test_length() {
        let conn = &get_connection(None).unwrap();
        let (_bg, path) = setup_block_group(conn);
        let mut path_cache = PathCache::new(conn);
        let accession =
            BlockGroup::add_accession(conn, &path, "test", 5, 35, &mut path_cache).unwrap();

        assert_eq!(accession.length(conn).unwrap(), 30);
    }
}
