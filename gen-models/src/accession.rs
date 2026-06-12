use std::collections::{HashMap, HashSet};

use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, calculate_hash,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use intervaltree::IntervalTree;
use itertools::Itertools;
use rusqlite::{Result, Row, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group_edge::BlockGroupEdge,
    db::GraphConnection,
    edge::Edge,
    gen_models_capnp::{accession, accession_edge},
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
    #[error("Duplicate entry with uuid: {0}")]
    Duplicate(String),
    #[error("Accession {0} has no edges in accession_edges")]
    MissingPath(HashId),
    #[error("Accession span cannot be empty")]
    EmptySpan,
    #[error("Accession span edge {0} does not exist")]
    MissingEdge(HashId),
    #[error("Invalid accession span: {0}")]
    InvalidSpan(String),
    #[error("No accession span walk found: {0}")]
    NoWalk(String),
    #[error("Ambiguous accession span walk: {0}")]
    AmbiguousWalk(String),
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

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq, Eq, Hash)]
pub struct AccessionEdge {
    pub id: HashId,
    pub accession_id: HashId,
    pub edge_id: HashId,
    pub index_in_path: i64,
    pub source_offset: Option<i64>,
    pub target_offset: Option<i64>,
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
        builder.set_accession_id(&self.accession_id.0).unwrap();
        builder.set_index_in_path(self.index_in_path);
        builder.set_edge_id(&self.edge_id.0).unwrap();
        match self.source_offset {
            Some(offset) => builder.reborrow().get_source_offset().set_some(offset),
            None => builder.reborrow().get_source_offset().set_none(()),
        }
        match self.target_offset {
            Some(offset) => builder.reborrow().get_target_offset().set_some(offset),
            None => builder.reborrow().get_target_offset().set_none(()),
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
        let edge_id = reader
            .get_edge_id()
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
        let source_offset = match reader.get_source_offset().which().unwrap() {
            accession_edge::source_offset::None(()) => None,
            accession_edge::source_offset::Some(offset) => Some(offset),
        };
        let target_offset = match reader.get_target_offset().which().unwrap() {
            accession_edge::target_offset::None(()) => None,
            accession_edge::target_offset::Some(offset) => Some(offset),
        };

        AccessionEdge {
            id,
            accession_id,
            index_in_path,
            edge_id,
            source_offset,
            target_offset,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AccessionEdgeData {
    pub edge_id: HashId,
    pub source_offset: Option<i64>,
    pub target_offset: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedAccessionSpan {
    pub blocks: Vec<NodeIntervalBlock>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccessionSpanCreate<'a> {
    pub name: &'a str,
    pub block_group_id: &'a HashId,
    pub parent_accession_id: Option<&'a HashId>,
    pub spans: &'a [ResolvedAccessionSpan],
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NodeSlice {
    pub node_id: HashId,
    pub sequence_start: i64,
    pub sequence_end: i64,
    pub strand: Strand,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccessionSpanSearch<'a> {
    pub block_group_id: &'a HashId,
    pub anchors: Vec<NodeSlice>,
    pub policy: SearchPolicy,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SearchPolicy {
    ExactAdjacentOnly,
}

impl AccessionEdgeData {
    pub fn id_hash(&self, accession_id: &HashId, index_in_path: i64) -> HashId {
        HashId(calculate_hash(&format!(
            "{}:{}:{}:{:?}:{:?}",
            accession_id, self.edge_id, index_in_path, self.source_offset, self.target_offset
        )))
    }
}

impl From<&AccessionEdge> for AccessionEdgeData {
    fn from(item: &AccessionEdge) -> Self {
        AccessionEdgeData {
            edge_id: item.edge_id,
            source_offset: item.source_offset,
            target_offset: item.target_offset,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AccessionEdgeInfo {
    accession_edge: AccessionEdge,
    edge: Edge,
}

impl Query for AccessionEdgeInfo {
    type Model = AccessionEdgeInfo;

    const PRIMARY_KEY: &'static str = "id";
    const TABLE_NAME: &'static str = "accession_edges";

    fn process_row(row: &Row) -> Self::Model {
        AccessionEdgeInfo {
            accession_edge: AccessionEdge::process_row(row),
            edge: Edge {
                id: row.get(6).unwrap(),
                source_node_id: row.get(7).unwrap(),
                source_coordinate: row.get(8).unwrap(),
                source_strand: row.get(9).unwrap(),
                target_node_id: row.get(10).unwrap(),
                target_coordinate: row.get(11).unwrap(),
                target_strand: row.get(12).unwrap(),
            },
        }
    }
}

impl AccessionEdgeInfo {
    pub fn get_by_accession_id(
        conn: &GraphConnection,
        accession_id: &HashId,
    ) -> Result<Vec<AccessionEdgeInfo>> {
        let query = "\
            select ae.*, e.* \
            from accession_edges ae \
            join edges e on e.id = ae.edge_id \
            where ae.accession_id = ?1 \
            order by ae.index_in_path;";
        Ok(Self::query(conn, query, params![accession_id]))
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

    pub fn create_from_spans(
        conn: &GraphConnection,
        create: AccessionSpanCreate,
    ) -> Result<Accession, AccessionError> {
        let accession = Accession::create(
            conn,
            create.name,
            create.block_group_id,
            create.parent_accession_id,
        )?;
        let accession_edges = Accession::edge_data_from_spans(conn, create.spans)?;
        AccessionEdge::bulk_create(conn, &accession.id, &accession_edges);
        Ok(accession)
    }

    pub fn get_or_create_from_spans(
        conn: &GraphConnection,
        create: AccessionSpanCreate,
    ) -> Result<Accession, AccessionError> {
        let accession = Accession::get_or_create(
            conn,
            create.name,
            create.block_group_id,
            create.parent_accession_id,
        )?;
        if Accession::get_edges_by_id(conn, &accession.id)?.is_empty() {
            let accession_edges = Accession::edge_data_from_spans(conn, create.spans)?;
            AccessionEdge::bulk_create(conn, &accession.id, &accession_edges);
        }
        Ok(accession)
    }

    pub fn edge_data_from_spans(
        conn: &GraphConnection,
        spans: &[ResolvedAccessionSpan],
    ) -> Result<Vec<AccessionEdgeData>, AccessionError> {
        if spans.is_empty() {
            return Err(AccessionError::EmptySpan);
        }
        let mut edge_ids = HashSet::new();
        for span in spans {
            if span.blocks.is_empty() {
                return Err(AccessionError::EmptySpan);
            }
            for block in &span.blocks {
                if block.sequence_end <= block.sequence_start {
                    return Err(AccessionError::InvalidSpan(format!(
                        "block {} has non-positive length {}..{}",
                        block.node_id, block.sequence_start, block.sequence_end
                    )));
                }
                let source_edge_id = block.source_edge_id.ok_or_else(|| {
                    AccessionError::InvalidSpan(format!(
                        "block {} is missing source edge provenance",
                        block.node_id
                    ))
                })?;
                let target_edge_id = block.target_edge_id.ok_or_else(|| {
                    AccessionError::InvalidSpan(format!(
                        "block {} is missing target edge provenance",
                        block.node_id
                    ))
                })?;
                edge_ids.insert(source_edge_id);
                edge_ids.insert(target_edge_id);
            }
        }

        let edge_ids = edge_ids.into_iter().collect::<Vec<_>>();
        let edges = Edge::query_by_ids(conn, &edge_ids)
            .into_iter()
            .map(|edge| (edge.id, edge))
            .collect::<HashMap<_, _>>();
        for edge_id in &edge_ids {
            if !edges.contains_key(edge_id) {
                return Err(AccessionError::MissingEdge(*edge_id));
            }
        }

        let mut accession_edges = Vec::new();
        for span in spans {
            let first_block = span.blocks.first().ok_or(AccessionError::EmptySpan)?;
            let first_source_edge_id = first_block.source_edge_id.ok_or_else(|| {
                AccessionError::InvalidSpan(format!(
                    "block {} is missing source edge provenance",
                    first_block.node_id
                ))
            })?;
            let source_edge = edges
                .get(&first_source_edge_id)
                .ok_or(AccessionError::MissingEdge(first_source_edge_id))?;
            Accession::validate_source_edge(source_edge, first_block)?;
            accession_edges.push(AccessionEdgeData {
                edge_id: source_edge.id,
                source_offset: None,
                target_offset: Some(first_block.sequence_start - source_edge.target_coordinate),
            });

            for (index, block) in span.blocks.iter().enumerate() {
                let block_target_edge_id = block.target_edge_id.ok_or_else(|| {
                    AccessionError::InvalidSpan(format!(
                        "block {} is missing target edge provenance",
                        block.node_id
                    ))
                })?;
                let target_edge = edges
                    .get(&block_target_edge_id)
                    .ok_or(AccessionError::MissingEdge(block_target_edge_id))?;
                Accession::validate_target_edge(target_edge, block)?;
                let next_block = span.blocks.get(index + 1);
                let target_offset = match next_block {
                    Some(next_block) => {
                        let next_source_edge_id = next_block.source_edge_id.ok_or_else(|| {
                            AccessionError::InvalidSpan(format!(
                                "block {} is missing source edge provenance",
                                next_block.node_id
                            ))
                        })?;
                        if next_source_edge_id != block_target_edge_id {
                            return Err(AccessionError::InvalidSpan(format!(
                                "block {} exits on edge {}, but next block {} enters on edge {}",
                                block.node_id,
                                block_target_edge_id,
                                next_block.node_id,
                                next_source_edge_id
                            )));
                        }
                        Accession::validate_source_edge(target_edge, next_block)?;
                        Some(next_block.sequence_start - target_edge.target_coordinate)
                    }
                    None => None,
                };
                accession_edges.push(AccessionEdgeData {
                    edge_id: target_edge.id,
                    source_offset: Some(block.sequence_end - target_edge.source_coordinate),
                    target_offset,
                });
            }
        }

        Ok(accession_edges)
    }

    pub fn search_span(
        conn: &GraphConnection,
        search: AccessionSpanSearch,
    ) -> Result<ResolvedAccessionSpan, AccessionError> {
        match search.policy {
            SearchPolicy::ExactAdjacentOnly => {
                Accession::search_exact_adjacent_span(conn, search.block_group_id, &search.anchors)
            }
        }
    }

    fn search_exact_adjacent_span(
        conn: &GraphConnection,
        block_group_id: &HashId,
        anchors: &[NodeSlice],
    ) -> Result<ResolvedAccessionSpan, AccessionError> {
        if anchors.is_empty() {
            return Err(AccessionError::EmptySpan);
        }
        for anchor in anchors {
            if anchor.sequence_end <= anchor.sequence_start {
                return Err(AccessionError::InvalidSpan(format!(
                    "anchor {} has non-positive length {}..{}",
                    anchor.node_id, anchor.sequence_start, anchor.sequence_end
                )));
            }
        }

        let mut seen_edge_ids = HashSet::new();
        let edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id)
            .into_iter()
            .filter_map(|augmented_edge| {
                if seen_edge_ids.insert(augmented_edge.edge.id) {
                    Some(augmented_edge.edge)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let first_anchor = anchors.first().ok_or(AccessionError::EmptySpan)?;
        let first_source_edge =
            Accession::unique_search_edge(&edges, "entering first anchor", |edge| {
                edge.target_node_id == first_anchor.node_id
                    && edge.target_strand == first_anchor.strand
                    && edge.target_coordinate <= first_anchor.sequence_start
            })?;

        let mut blocks = Vec::new();
        let mut source_edge_id = first_source_edge.id;
        for (index, anchor) in anchors.iter().enumerate() {
            let target_edge = match anchors.get(index + 1) {
                Some(next_anchor) => {
                    Accession::unique_search_edge(&edges, "between adjacent anchors", |edge| {
                        edge.source_node_id == anchor.node_id
                            && edge.source_strand == anchor.strand
                            && edge.source_coordinate == anchor.sequence_end
                            && edge.target_node_id == next_anchor.node_id
                            && edge.target_strand == next_anchor.strand
                            && edge.target_coordinate == next_anchor.sequence_start
                    })?
                }
                None => Accession::unique_search_edge(&edges, "leaving final anchor", |edge| {
                    edge.source_node_id == anchor.node_id
                        && edge.source_strand == anchor.strand
                        && edge.source_coordinate >= anchor.sequence_end
                })?,
            };
            blocks.push(NodeIntervalBlock {
                node_id: anchor.node_id,
                start: 0,
                end: anchor.sequence_end - anchor.sequence_start,
                sequence_start: anchor.sequence_start,
                sequence_end: anchor.sequence_end,
                strand: anchor.strand,
                source_edge_id: Some(source_edge_id),
                target_edge_id: Some(target_edge.id),
            });
            source_edge_id = target_edge.id;
        }

        Ok(ResolvedAccessionSpan { blocks })
    }

    fn unique_search_edge(
        edges: &[Edge],
        description: &str,
        predicate: impl Fn(&Edge) -> bool,
    ) -> Result<Edge, AccessionError> {
        let matches = edges
            .iter()
            .filter(|edge| predicate(edge))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(AccessionError::NoWalk(description.to_string())),
            [edge] => Ok((*edge).clone()),
            _ => Err(AccessionError::AmbiguousWalk(description.to_string())),
        }
    }

    fn validate_source_edge(edge: &Edge, block: &NodeIntervalBlock) -> Result<(), AccessionError> {
        if edge.target_node_id != block.node_id {
            return Err(AccessionError::InvalidSpan(format!(
                "edge {} targets node {}, not block node {}",
                edge.id, edge.target_node_id, block.node_id
            )));
        }
        if edge.target_strand != block.strand {
            return Err(AccessionError::InvalidSpan(format!(
                "edge {} target strand {:?} does not match block strand {:?}",
                edge.id, edge.target_strand, block.strand
            )));
        }
        if edge.target_coordinate > block.sequence_start {
            return Err(AccessionError::InvalidSpan(format!(
                "edge {} enters at {}, after block start {}",
                edge.id, edge.target_coordinate, block.sequence_start
            )));
        }
        Ok(())
    }

    fn validate_target_edge(edge: &Edge, block: &NodeIntervalBlock) -> Result<(), AccessionError> {
        if edge.source_node_id != block.node_id {
            return Err(AccessionError::InvalidSpan(format!(
                "edge {} sources node {}, not block node {}",
                edge.id, edge.source_node_id, block.node_id
            )));
        }
        if edge.source_strand != block.strand {
            return Err(AccessionError::InvalidSpan(format!(
                "edge {} source strand {:?} does not match block strand {:?}",
                edge.id, edge.source_strand, block.strand
            )));
        }
        if edge.source_coordinate < block.sequence_end {
            return Err(AccessionError::InvalidSpan(format!(
                "edge {} exits at {}, before block end {}",
                edge.id, edge.source_coordinate, block.sequence_end
            )));
        }
        Ok(())
    }

    pub fn get_edges_by_id(
        conn: &GraphConnection,
        accession_id: &HashId,
    ) -> Result<Vec<AccessionEdge>, AccessionError> {
        Ok(AccessionEdge::query(
            conn,
            "select * from accession_edges where accession_id = ?1",
            params![accession_id],
        ))
    }

    pub fn blocks(&self, conn: &GraphConnection) -> Result<Vec<NodeIntervalBlock>, AccessionError> {
        let edges = AccessionEdgeInfo::get_by_accession_id(conn, &self.id)?;
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
            source_edge_id: None,
            target_edge_id: None,
        }];

        for (into, out_of) in edges.iter().tuple_windows() {
            let Some(target_offset) = into.accession_edge.target_offset else {
                continue;
            };
            let Some(source_offset) = out_of.accession_edge.source_offset else {
                continue;
            };
            let sequence_start = into.edge.target_coordinate + target_offset;
            let sequence_end = out_of.edge.source_coordinate + source_offset;
            let block_len = sequence_end - sequence_start;
            if block_len <= 0 {
                continue;
            }
            blocks.push(NodeIntervalBlock {
                node_id: into.edge.target_node_id,
                start: offset,
                end: offset + block_len,
                sequence_start,
                sequence_end,
                strand: into.edge.target_strand,
                source_edge_id: Some(into.accession_edge.edge_id),
                target_edge_id: Some(out_of.accession_edge.edge_id),
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
            source_edge_id: None,
            target_edge_id: None,
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
        accession_id: &HashId,
        index_in_path: i64,
        edge: AccessionEdgeData,
    ) -> Result<AccessionEdge, AccessionEdgeError> {
        let hash = edge.id_hash(accession_id, index_in_path);
        // TODO: handle get-or-create
        let insert_statement = "INSERT INTO accession_edges (id, accession_id, edge_id, index_in_path, source_offset, target_offset) VALUES (?1, ?2, ?3, ?4, ?5, ?6);";
        let mut stmt = conn.prepare(insert_statement).unwrap();
        match stmt.execute(params![
            hash,
            accession_id,
            edge.edge_id,
            index_in_path,
            edge.source_offset,
            edge.target_offset
        ]) {
            Ok(_) => {}
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {}
            }
            Err(err) => return Err(AccessionEdgeError::DatabaseError(err)),
        };
        Ok(AccessionEdge {
            id: hash,
            accession_id: *accession_id,
            edge_id: edge.edge_id,
            index_in_path,
            source_offset: edge.source_offset,
            target_offset: edge.target_offset,
        })
    }

    pub fn bulk_create(
        conn: &GraphConnection,
        accession_id: &HashId,
        edges: &[AccessionEdgeData],
    ) -> Vec<HashId> {
        let edge_ids = edges
            .iter()
            .enumerate()
            .map(|(index, edge)| edge.id_hash(accession_id, index as i64))
            .collect::<Vec<_>>();
        let query = AccessionEdge::query_by_ids(conn, &edge_ids);
        let existing_edges = query.iter().map(|edge| &edge.id).collect::<HashSet<_>>();

        let mut edges_to_insert = Vec::new();
        for (index, edge) in edge_ids.iter().enumerate() {
            if !existing_edges.contains(edge) {
                edges_to_insert.push((index, &edges[index]));
            }
        }

        let batch_size = max_rows_per_batch(conn, 6);

        for chunk in &edges_to_insert.iter().chunks(batch_size) {
            let mut rows = vec![];
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            for (index, edge) in chunk {
                let index_in_path = *index as i64;
                params.push(Box::new(edge.id_hash(accession_id, index_in_path)));
                params.push(Box::new(*accession_id));
                params.push(Box::new(edge.edge_id));
                params.push(Box::new(index_in_path));
                params.push(Box::new(edge.source_offset));
                params.push(Box::new(edge.target_offset));
                rows.push("(?, ?, ?, ?, ?, ?)");
            }
            let sql = format!(
                "INSERT INTO accession_edges (id, accession_id, edge_id, index_in_path, source_offset, target_offset) VALUES {};",
                rows.join(",")
            );
            conn.execute(&sql, rusqlite::params_from_iter(params))
                .unwrap();
        }
        edge_ids
    }

    pub fn to_data(edge: AccessionEdge) -> AccessionEdgeData {
        AccessionEdgeData {
            edge_id: edge.edge_id,
            source_offset: edge.source_offset,
            target_offset: edge.target_offset,
        }
    }
}

impl Query for AccessionEdge {
    type Model = AccessionEdge;

    const TABLE_NAME: &'static str = "accession_edges";

    fn process_row(row: &Row) -> Self::Model {
        AccessionEdge {
            id: row.get(0).unwrap(),
            accession_id: row.get(1).unwrap(),
            edge_id: row.get(2).unwrap(),
            index_in_path: row.get(3).unwrap(),
            source_offset: row.get(4).unwrap(),
            target_offset: row.get(5).unwrap(),
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
    fn create_from_spans_indexes_single_node_fragment() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let path_edges = PathEdge::edges_for_path(conn, &path.id);
        let accession = Accession::create_from_spans(
            conn,
            AccessionSpanCreate {
                name: "a-fragment",
                block_group_id: &block_group_id,
                parent_accession_id: None,
                spans: &[ResolvedAccessionSpan {
                    blocks: vec![NodeIntervalBlock {
                        node_id: HashId::convert_str("test-a-node"),
                        start: 0,
                        end: 5,
                        sequence_start: 3,
                        sequence_end: 8,
                        strand: Strand::Forward,
                        source_edge_id: Some(path_edges[0].id),
                        target_edge_id: Some(path_edges[1].id),
                    }],
                }],
            },
        )
        .unwrap();

        let edges = Accession::get_edges_by_id(conn, &accession.id).unwrap();
        assert_eq!(accession.length(conn).unwrap(), 5);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].edge_id, path_edges[0].id);
        assert_eq!(edges[0].source_offset, None);
        assert_eq!(edges[0].target_offset, Some(3));
        assert_eq!(edges[1].edge_id, path_edges[1].id);
        assert_eq!(edges[1].source_offset, Some(-2));
        assert_eq!(edges[1].target_offset, None);
    }

    #[test]
    fn create_from_spans_indexes_explicit_existing_walk() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let path_edges = PathEdge::edges_for_path(conn, &path.id);
        let accession = Accession::create_from_spans(
            conn,
            AccessionSpanCreate {
                name: "explicit-walk",
                block_group_id: &block_group_id,
                parent_accession_id: None,
                spans: &[ResolvedAccessionSpan {
                    blocks: vec![
                        NodeIntervalBlock {
                            node_id: HashId::convert_str("test-a-node"),
                            start: 0,
                            end: 5,
                            sequence_start: 5,
                            sequence_end: 10,
                            strand: Strand::Forward,
                            source_edge_id: Some(path_edges[0].id),
                            target_edge_id: Some(path_edges[1].id),
                        },
                        NodeIntervalBlock {
                            node_id: HashId::convert_str("test-t-node"),
                            start: 5,
                            end: 11,
                            sequence_start: 0,
                            sequence_end: 6,
                            strand: Strand::Forward,
                            source_edge_id: Some(path_edges[1].id),
                            target_edge_id: Some(path_edges[2].id),
                        },
                    ],
                }],
            },
        )
        .unwrap();

        let edges = Accession::get_edges_by_id(conn, &accession.id).unwrap();
        assert_eq!(accession.length(conn).unwrap(), 11);
        assert_eq!(
            edges.iter().map(|edge| edge.edge_id).collect::<Vec<_>>(),
            vec![path_edges[0].id, path_edges[1].id, path_edges[2].id]
        );
        assert_eq!(edges[0].target_offset, Some(5));
        assert_eq!(edges[1].source_offset, Some(0));
        assert_eq!(edges[1].target_offset, Some(0));
        assert_eq!(edges[2].source_offset, Some(-4));
    }

    #[test]
    fn search_spans_finds_exact_adjacent_existing_walk() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, path) = setup_block_group(conn);
        let path_edges = PathEdge::edges_for_path(conn, &path.id);

        let span = Accession::search_span(
            conn,
            AccessionSpanSearch {
                block_group_id: &block_group_id,
                anchors: vec![
                    NodeSlice {
                        node_id: HashId::convert_str("test-a-node"),
                        sequence_start: 5,
                        sequence_end: 10,
                        strand: Strand::Forward,
                    },
                    NodeSlice {
                        node_id: HashId::convert_str("test-t-node"),
                        sequence_start: 0,
                        sequence_end: 6,
                        strand: Strand::Forward,
                    },
                ],
                policy: SearchPolicy::ExactAdjacentOnly,
            },
        )
        .unwrap();

        assert_eq!(
            span.blocks
                .iter()
                .map(|block| (block.source_edge_id, block.target_edge_id))
                .collect::<Vec<_>>(),
            vec![
                (Some(path_edges[0].id), Some(path_edges[1].id)),
                (Some(path_edges[1].id), Some(path_edges[2].id))
            ]
        );
        let accession = Accession::create_from_spans(
            conn,
            AccessionSpanCreate {
                name: "searched-walk",
                block_group_id: &block_group_id,
                parent_accession_id: None,
                spans: &[span],
            },
        )
        .unwrap();
        assert_eq!(accession.length(conn).unwrap(), 11);
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
            accession_id: HashId::convert_str("1"),
            index_in_path: 2,
            edge_id: HashId::convert_str("10"),
            source_offset: Some(100),
            target_offset: Some(200),
        };

        let mut message = TypedBuilder::<accession_edge::Owned>::new_default();
        let mut root = message.init_root();
        accession_edge.write_capnp(&mut root);

        let deserialized = AccessionEdge::read_capnp(root.into_reader());
        assert_eq!(accession_edge, deserialized);
    }

    #[test]
    fn test_accession_edges_stores_block_group_edge_offsets() {
        let conn = &get_connection(None).unwrap();
        let (_bg, path) = setup_block_group(conn);
        let mut path_cache = PathCache::new(conn);

        let accession =
            BlockGroup::add_accession(conn, &path, "test", 5, 35, &mut path_cache).unwrap();
        let edges = Accession::get_edges_by_id(conn, &accession.id).unwrap();

        assert_eq!(edges.len(), 5);
        assert_eq!(edges[0].source_offset, None);
        assert_eq!(edges[0].target_offset, Some(5));
        assert_eq!(edges[1].source_offset, Some(0));
        assert_eq!(edges[1].target_offset, Some(0));
        assert_eq!(edges[4].source_offset, Some(-5));
        assert_eq!(edges[4].target_offset, None);
    }

    #[test]
    fn test_accession_edges_store_accession_path_order() {
        let conn = &get_connection(None).unwrap();
        let (_bg, path) = setup_block_group(conn);
        let mut path_cache = PathCache::new(conn);

        let accession =
            BlockGroup::add_accession(conn, &path, "test", 5, 35, &mut path_cache).unwrap();
        let edges = Accession::get_edges_by_id(conn, &accession.id).unwrap();

        assert_eq!(edges.len(), 5);
        assert!(edges.iter().all(|edge| edge.accession_id == accession.id));
        assert_eq!(
            edges
                .iter()
                .map(|edge| edge.index_in_path)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert!(
            conn.prepare("select * from accession_paths").is_err(),
            "accession_paths table should be merged into accession_edges"
        );
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
        let path_edges = PathEdge::edges_for_path(conn, &path.id);

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
                source_edge_id: Some(path_edges[0].id),
                target_edge_id: Some(path_edges[1].id),
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
                source_edge_id: Some(path_edges[1].id),
                target_edge_id: Some(path_edges[2].id),
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
                source_edge_id: Some(path_edges[2].id),
                target_edge_id: Some(path_edges[3].id),
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
                source_edge_id: Some(path_edges[3].id),
                target_edge_id: Some(path_edges[4].id),
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
