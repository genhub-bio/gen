use std::{collections::HashMap, ops::Range as StdRange, rc::Rc};

use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, Workspace,
    calculate_hash, is_terminal,
    range::Range,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use intervaltree::IntervalTree;
use rusqlite::{Row, params, types::Value as SQLValue};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group_edge::AugmentedEdgeData,
    db::GraphConnection,
    errors::QueryError,
    gen_models_capnp::{accession, accession_node},
    region::ResolvedGenRegion,
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
    #[error("Accession node creation error: {0}")]
    AccessionNodeError(#[from] AccessionNodeError),
    #[error("Duplicate entry with uuid: {0}")]
    Duplicate(String),
    #[error("Accession {0} has no nodes in accession_nodes")]
    MissingPath(HashId),
    #[error("Accession has no spans")]
    EmptySpans,
    #[error("Invalid accession range: {start}-{end}")]
    InvalidRange { start: i64, end: i64 },
    #[error("Unable to determine interval tree length")]
    MissingIntervalTreeLength,
    #[error("Unable to project region into accession spans: {0}")]
    RegionProjection(String),
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
pub struct AccessionNode {
    pub id: HashId,
    pub accession_id: HashId,
    pub node_id: HashId,
    pub sequence_start: i64,
    pub sequence_end: i64,
    pub strand: Strand,
    pub index_in_path: i64,
}

#[derive(Debug, Error, PartialEq)]
pub enum AccessionNodeError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl<'a> Capnp<'a> for AccessionNode {
    type Builder = accession_node::Builder<'a>;
    type Reader = accession_node::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_accession_id(&self.accession_id.0).unwrap();
        builder.set_node_id(&self.node_id.0).unwrap();
        builder.set_sequence_start(self.sequence_start);
        builder.set_sequence_end(self.sequence_end);
        builder.set_strand(self.strand.into());
        builder.set_index_in_path(self.index_in_path);
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
        let node_id = reader
            .get_node_id()
            .unwrap()
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let sequence_start = reader.get_sequence_start();
        let sequence_end = reader.get_sequence_end();
        let strand = reader.get_strand().unwrap().into();
        let index_in_path = reader.get_index_in_path();

        AccessionNode {
            id,
            accession_id,
            node_id,
            sequence_start,
            sequence_end,
            strand,
            index_in_path,
        }
    }
}

/// AccessionNodeData is a non-database form of AccessionNode. It allows callers to construct
/// an AccessionNode without having to calculate the id.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AccessionNodeData {
    pub accession_id: HashId,
    pub node_id: HashId,
    pub sequence_start: i64,
    pub sequence_end: i64,
    pub strand: Strand,
    pub index_in_path: i64,
}

/// AccessionSpan is similar to AnnotationSegment in shape, but its primary use is
/// for creating AccessionNodes
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccessionSpan {
    pub node_id: HashId,
    pub range: Range,
    pub strand: Strand,
}

impl AccessionNodeData {
    pub fn id_hash(&self) -> HashId {
        HashId(calculate_hash(&format!(
            "{}:{}:{}:{}:{}:{}",
            self.accession_id,
            self.node_id,
            self.sequence_start,
            self.sequence_end,
            self.strand,
            self.index_in_path
        )))
    }
}

impl From<&AccessionNode> for AccessionNodeData {
    fn from(item: &AccessionNode) -> Self {
        AccessionNodeData {
            accession_id: item.accession_id,
            node_id: item.node_id,
            sequence_start: item.sequence_start,
            sequence_end: item.sequence_end,
            strand: item.strand,
            index_in_path: item.index_in_path,
        }
    }
}

impl From<AccessionNodeData> for AccessionNode {
    fn from(item: AccessionNodeData) -> Self {
        AccessionNode {
            id: item.id_hash(),
            accession_id: item.accession_id,
            node_id: item.node_id,
            sequence_start: item.sequence_start,
            sequence_end: item.sequence_end,
            strand: item.strand,
            index_in_path: item.index_in_path,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewAccession {
    pub name: String,
    pub block_group_id: HashId,
    pub parent_accession_id: Option<HashId>,
    pub spans: Vec<AccessionSpan>,
}

impl Accession {
    /// An accession is an ordered array of slices of nodes. The purpose
    /// of an accession is to provide additional layers of information to
    /// a graph. These extra pieces of information can be features such
    /// as gene annotations, epigenetic markers, translocations, and so on.
    ///
    /// From a modeling stance, there is a table AccessionNodes, which holds
    /// the positions within a Node the accession covers, its strand, and what
    /// position it is in the array. Accessions are not meant to be restricted
    /// to a graph topology. Take the following example of a spliced gene:
    ///
    /// ```text
    /// AAAAAAAAAAAAAAAATTTTTTTTTTTTCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGG
    ///     [part-a]--->[part-b]---->[part-c]-->[part-d]
    /// ```
    ///
    /// There are no edges in the primary sequence, but there are 4 accession nodes
    /// linked together. There is similarly no guarantee that nodes are on the same
    /// blockgroup, which enables modeling of events such as translocations.
    ///
    /// Similarly, we do not model an accession on the edges of the graph
    /// because the variation in the graph does not impact the accession. Take an example
    /// where a variant is introduced in the intronic region of a gene:
    ///
    /// ```text
    ///             A
    ///            / \
    /// AAAAAAAAAA-TTT-TTTTTTCCCCCCCCCCC
    /// [part-1]----------->[part-2]
    /// ```
    ///
    /// If we annotated according to edges, the introduction of the A variant questions
    /// if we should store part-1/part-2 also on the new edge between them. Storing it
    /// on nodes means it is up to the context the graph is being used to determine how to
    /// treat the graph topology.
    fn id_hash(
        block_group_id: &HashId,
        parent_accession_id: Option<&HashId>,
        name: &str,
    ) -> HashId {
        HashId(calculate_hash(&format!(
            "{block_group_id}:{parent_accession_id:?}:{name}"
        )))
    }

    pub fn create(conn: &GraphConnection, new: &NewAccession) -> Result<Accession, AccessionError> {
        if new.spans.is_empty() {
            return Err(AccessionError::EmptySpans);
        }
        let query = "INSERT INTO accessions (id, name, block_group_id, parent_accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = match conn.prepare(query) {
            Ok(s) => s,
            Err(e) => return Err(AccessionError::DatabaseError(e)),
        };
        let parent_accession_id = new.parent_accession_id.as_ref();
        let hash = Accession::id_hash(&new.block_group_id, parent_accession_id, &new.name);
        let accession = match stmt.execute((
            hash,
            &new.name,
            &new.block_group_id,
            parent_accession_id,
        )) {
            Ok(_) => Accession {
                id: hash,
                name: new.name.clone(),
                block_group_id: new.block_group_id,
                parent_accession_id: new.parent_accession_id,
            },
            Err(rusqlite::Error::SqliteFailure(err, _details))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                return Err(AccessionError::Duplicate(format!(
                    "An accession with the same name, block_group_id, and parent_accession_id already exists. name: {}, block_group_id: {}, parent_accession_id: {:?}",
                    new.name, new.block_group_id, new.parent_accession_id
                )));
            }
            Err(e) => return Err(AccessionError::DatabaseError(e)),
        };
        Self::insert_spans(conn, &accession.id, &new.spans)?;
        Ok(accession)
    }

    pub fn create_from_edges(
        conn: &GraphConnection,
        name: &str,
        block_group_id: &HashId,
        parent_accession_id: Option<&HashId>,
        edges: &[AugmentedEdgeData],
    ) -> Result<Accession, AccessionError> {
        let spans = edges
            .windows(2)
            .map(|edge_pair| {
                let into = &edge_pair[0].edge_data;
                let out_of = &edge_pair[1].edge_data;
                AccessionSpan {
                    node_id: into.target_node_id,
                    range: Range {
                        start: into.target_coordinate,
                        end: out_of.source_coordinate,
                    },
                    strand: into.target_strand,
                }
            })
            .collect::<Vec<_>>();
        Self::create(
            conn,
            &NewAccession {
                name: name.to_string(),
                block_group_id: *block_group_id,
                parent_accession_id: parent_accession_id.copied(),
                spans,
            },
        )
    }

    pub fn get_or_create(
        conn: &GraphConnection,
        new: &NewAccession,
    ) -> Result<Accession, AccessionError> {
        match Accession::create(conn, new) {
            Ok(accession) => Ok(accession),
            Err(AccessionError::Duplicate(message)) => {
                let accession = Accession {
                    id: Accession::id_hash(
                        &new.block_group_id,
                        new.parent_accession_id.as_ref(),
                        &new.name,
                    ),
                    name: new.name.clone(),
                    block_group_id: new.block_group_id,
                    parent_accession_id: new.parent_accession_id,
                };
                let nodes = Self::get_nodes_by_id(conn, &accession.id, None);
                if nodes.is_empty() {
                    Self::insert_spans(conn, &accession.id, &new.spans)?;
                } else if !Self::nodes_match_spans(&nodes, &new.spans) {
                    return Err(AccessionError::Duplicate(message));
                }
                Ok(accession)
            }
            Err(err) => Err(err),
        }
    }

    fn nodes_match_spans(nodes: &[AccessionNode], spans: &[AccessionSpan]) -> bool {
        nodes.len() == spans.len()
            && nodes
                .iter()
                .zip(spans)
                .enumerate()
                .all(|(index, (node, span))| {
                    node.index_in_path == index as i64
                        && node.node_id == span.node_id
                        && node.sequence_start == span.range.start
                        && node.sequence_end == span.range.end
                        && node.strand == span.strand
                })
    }

    fn insert_spans(
        conn: &GraphConnection,
        accession_id: &HashId,
        spans: &[AccessionSpan],
    ) -> Result<(), AccessionError> {
        let nodes = spans
            .iter()
            .enumerate()
            .map(|(index, span)| AccessionNodeData {
                accession_id: *accession_id,
                node_id: span.node_id,
                sequence_start: span.range.start,
                sequence_end: span.range.end,
                strand: span.strand,
                index_in_path: index as i64,
            })
            .collect::<Vec<_>>();
        AccessionNode::bulk_create(conn, &nodes)?;
        Ok(())
    }

    pub fn get_nodes_by_id(
        conn: &GraphConnection,
        accession_id: &HashId,
        history_ref: Option<&str>,
    ) -> Vec<AccessionNode> {
        let query = format!(
            "select * from {} where accession_id = :accession_id order by index_in_path;",
            AccessionNode::table_name_with_history_ref(history_ref)
        );
        let mut params: Vec<(&str, &dyn rusqlite::ToSql)> = vec![(":accession_id", accession_id)];
        if let Some(history_ref) = history_ref.as_ref() {
            params.push((":history_ref", history_ref));
        }
        AccessionNode::query(conn, &query, &params[..])
    }

    pub fn blocks(&self, conn: &GraphConnection) -> Result<Vec<NodeIntervalBlock>, AccessionError> {
        let nodes = Self::get_nodes_by_id(conn, &self.id, None);
        if nodes.is_empty() {
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

        for node in nodes {
            let block_len = node.sequence_end - node.sequence_start;
            blocks.push(NodeIntervalBlock {
                node_id: node.node_id,
                start: offset,
                end: offset + block_len,
                sequence_start: node.sequence_start,
                sequence_end: node.sequence_end,
                strand: node.strand,
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
        let nodes = Self::get_nodes_by_id(conn, &self.id, None);
        if nodes.is_empty() {
            return Err(AccessionError::MissingPath(self.id));
        }

        Ok(nodes
            .iter()
            .map(|node| node.sequence_end - node.sequence_start)
            .sum())
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

impl AccessionSpan {
    /// Given an intervaltree, create AccessionSpans on the provided range positions.
    /// For example:
    /// from_intervaltree_ranges(tree, [(1..3), (5..10)])
    /// would create 2 AccessionSpans corresponding to the Nodes and positions in the
    /// input ranges.
    pub fn from_intervaltree_ranges(
        tree: &IntervalTree<i64, NodeIntervalBlock>,
        ranges: &[StdRange<i64>],
    ) -> Result<Vec<AccessionSpan>, AccessionError> {
        let length = intervaltree_length(tree)?;
        let mut spans = Vec::new();
        for range in ranges {
            if range.start > range.end {
                // TODO: When circular stuff is better supported, this should not be an error. We should
                // check if the blockgroup is circular and wrap around.
                return Err(AccessionError::InvalidRange {
                    start: range.start,
                    end: range.end,
                });
            }
            if range.start < 0 || range.end < 0 || range.start > length || range.end > length {
                return Err(AccessionError::InvalidRange {
                    start: range.start,
                    end: range.end,
                });
            }
            if range.start == range.end {
                continue;
            }
            let mut blocks = tree
                .query(range.start..range.end)
                .map(|entry| &entry.value)
                .filter(|block| !is_terminal(block.node_id))
                .collect::<Vec<_>>();
            blocks.sort_by_key(|block| block.start);
            spans.extend(blocks.into_iter().map(|block| {
                let clipped_start = range.start.max(block.start);
                let clipped_end = range.end.min(block.end);
                AccessionSpan {
                    node_id: block.node_id,
                    range: Range {
                        start: clipped_start - block.start + block.sequence_start,
                        end: clipped_end - block.start + block.sequence_start,
                    },
                    strand: block.strand,
                }
            }));
        }
        Ok(spans)
    }

    /// convert a ResolvedGenRegion to an AccessionSpan. If no sub-region is selected,
    /// the entire region is used.
    pub fn from_resolved_region(
        conn: &GraphConnection,
        workspace: &Workspace,
        region: &ResolvedGenRegion,
        ranges: Option<&[StdRange<i64>]>,
    ) -> Result<Vec<AccessionSpan>, AccessionError> {
        let tree = region
            .intervaltree(conn, workspace)
            .map_err(|err| AccessionError::RegionProjection(err.to_string()))?;
        match ranges {
            Some(ranges) => Self::from_intervaltree_ranges(&tree, ranges),
            None => {
                let range = region.start..region.end;
                Self::from_intervaltree_ranges(&tree, std::slice::from_ref(&range))
            }
        }
    }
}

fn intervaltree_length(tree: &IntervalTree<i64, NodeIntervalBlock>) -> Result<i64, AccessionError> {
    if let Some(end_block) = tree
        .query_point(i64::MAX - 2)
        .map(|entry| &entry.value)
        .find(|block| block.node_id == PATH_END_NODE_ID)
    {
        return Ok(end_block.start);
    }

    tree.iter_sorted()
        .map(|entry| &entry.value.end)
        .last()
        .copied()
        .ok_or(AccessionError::MissingIntervalTreeLength)
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
            1 => {
                if let Some(accession) = matches.into_iter().next() {
                    Ok(accession)
                } else {
                    Err(RegionResolutionError::NotFound(region.name.clone()))
                }
            }
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

impl AccessionNode {
    pub fn query_accessions(
        conn: &GraphConnection,
        accession_ids: &[HashId],
    ) -> Result<HashMap<HashId, Vec<AccessionNode>>, QueryError> {
        if accession_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let accession_values = accession_ids
            .iter()
            .copied()
            .map(SQLValue::from)
            .collect::<Vec<_>>();
        let nodes = AccessionNode::try_query(
            conn,
            "select * from accession_nodes where accession_id in rarray(?1) order by accession_id, index_in_path;",
            params![Rc::new(accession_values)],
        )?;
        let mut nodes_by_accession = HashMap::new();
        for node in nodes {
            nodes_by_accession
                .entry(node.accession_id)
                .or_insert_with(Vec::new)
                .push(node);
        }
        Ok(nodes_by_accession)
    }

    pub fn create(
        conn: &GraphConnection,
        node: AccessionNodeData,
    ) -> Result<AccessionNode, AccessionNodeError> {
        let hash = node.id_hash();
        let insert_statement = "INSERT INTO accession_nodes (id, accession_id, node_id, sequence_start, sequence_end, strand, index_in_path) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);";
        let mut stmt = conn.prepare(insert_statement)?;
        match stmt.execute(params![
            hash,
            node.accession_id,
            node.node_id,
            node.sequence_start,
            node.sequence_end,
            node.strand,
            node.index_in_path
        ]) {
            Ok(_) => {}
            Err(rusqlite::Error::SqliteFailure(err, _details)) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {}
            }
            Err(err) => return Err(AccessionNodeError::DatabaseError(err)),
        };
        Ok(AccessionNode::from(node))
    }

    pub fn bulk_create(
        conn: &GraphConnection,
        nodes: &[AccessionNodeData],
    ) -> Result<Vec<HashId>, AccessionNodeError> {
        let node_ids = nodes
            .iter()
            .map(AccessionNodeData::id_hash)
            .collect::<Vec<_>>();
        let batch_size = max_rows_per_batch(conn, 7);

        for chunk in nodes.chunks(batch_size) {
            let mut rows = vec![];
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            for node in chunk {
                params.push(Box::new(node.id_hash()));
                params.push(Box::new(node.accession_id));
                params.push(Box::new(node.node_id));
                params.push(Box::new(node.sequence_start));
                params.push(Box::new(node.sequence_end));
                params.push(Box::new(node.strand));
                params.push(Box::new(node.index_in_path));
                rows.push("(?, ?, ?, ?, ?, ?, ?)");
            }
            let sql = format!(
                "INSERT OR IGNORE INTO accession_nodes (id, accession_id, node_id, sequence_start, sequence_end, strand, index_in_path) VALUES {};",
                rows.join(",")
            );
            conn.execute(&sql, rusqlite::params_from_iter(params))?;
        }
        Ok(node_ids)
    }

    pub fn bulk_delete(conn: &GraphConnection, nodes: &[AccessionNodeData]) {
        let ids = nodes
            .iter()
            .map(AccessionNodeData::id_hash)
            .collect::<Vec<_>>();
        AccessionNode::delete_by_ids(conn, &ids);
    }
}

impl Query for AccessionNode {
    type Model = AccessionNode;

    const TABLE_NAME: &'static str = "accession_nodes";

    fn process_row(row: &Row) -> Self::Model {
        AccessionNode {
            id: row.get(0).unwrap(),
            accession_id: row.get(1).unwrap(),
            node_id: row.get(2).unwrap(),
            sequence_start: row.get(3).unwrap(),
            sequence_end: row.get(4).unwrap(),
            strand: row.get(5).unwrap(),
            index_in_path: row.get(6).unwrap(),
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
        block_group_edge::{AugmentedEdgeData, BlockGroupEdgeData},
        edge::EdgeData,
        path::Path,
        test_helpers::{create_bg, get_connection, interval_tree_verify, setup_block_group},
    };

    mod region_resolver {
        use super::*;

        #[test]
        fn test_resolves_accession_by_name_case_insensitively() {
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
        fn test_returns_not_found_for_missing_accession() {
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
        fn test_returns_ambiguous_for_multiple_matching_accessions() {
            let conn = &get_connection(None).unwrap();
            let (_bg, path) = setup_block_group(conn);
            let mut path_cache = PathCache::new(conn);
            let _ = BlockGroup::add_accession(conn, &path, "mreB", 5, 15, &mut path_cache).unwrap();

            let other_block_group = create_bg(conn, "test", "test", "other");
            let edge_ids = Path::edges_for_path(conn, &path.id, None)
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
            id: HashId::pad_str(200),
            name: "test_accession".to_string(),
            block_group_id: HashId::pad_str(150),
            parent_accession_id: Some(HashId::pad_str(100)),
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
            id: HashId::pad_str(201),
            name: "test_accession_2".to_string(),
            block_group_id: HashId::pad_str(151),
            parent_accession_id: None,
        };

        let mut message = TypedBuilder::<accession::Owned>::new_default();
        let mut root = message.init_root();
        accession.write_capnp(&mut root);

        let deserialized = Accession::read_capnp(root.into_reader());
        assert_eq!(accession, deserialized);
    }

    #[test]
    fn test_accession_node_capnp_serialization() {
        let accession_node = AccessionNode {
            id: HashId::pad_str(300),
            accession_id: HashId::convert_str("accession"),
            node_id: HashId::convert_str("20"),
            sequence_start: 2,
            sequence_end: 4,
            strand: Strand::Reverse,
            index_in_path: 1,
        };

        let mut message = TypedBuilder::<accession_node::Owned>::new_default();
        let mut root = message.init_root();
        accession_node.write_capnp(&mut root);

        let deserialized = AccessionNode::read_capnp(root.into_reader());
        assert_eq!(accession_node, deserialized);
    }

    #[test]
    fn test_accession_create_query() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let accession = Accession::create(
            conn,
            &NewAccession {
                name: "test".to_string(),
                block_group_id,
                parent_accession_id: None,
                spans: vec![AccessionSpan {
                    node_id: HashId::convert_str("test-a-node"),
                    range: Range { start: 0, end: 1 },
                    strand: Strand::Forward,
                }],
            },
        )
        .unwrap();
        let _accession_2 = Accession::create(
            conn,
            &NewAccession {
                name: "test2".to_string(),
                block_group_id,
                parent_accession_id: None,
                spans: vec![AccessionSpan {
                    node_id: HashId::convert_str("test-a-node"),
                    range: Range { start: 1, end: 2 },
                    strand: Strand::Forward,
                }],
            },
        )
        .unwrap();
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
    fn test_create_from_new_accession_inserts_ordered_spans() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let new_accession = NewAccession {
            name: "test".to_string(),
            block_group_id,
            parent_accession_id: None,
            spans: vec![
                AccessionSpan {
                    node_id: HashId::convert_str("test-a-node"),
                    range: Range { start: 2, end: 4 },
                    strand: Strand::Forward,
                },
                AccessionSpan {
                    node_id: HashId::convert_str("test-t-node"),
                    range: Range { start: 0, end: 2 },
                    strand: Strand::Forward,
                },
            ],
        };

        let accession = Accession::create(conn, &new_accession).unwrap();

        assert_eq!(accession.name, "test");
        let nodes = Accession::get_nodes_by_id(conn, &accession.id, None);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].node_id, HashId::convert_str("test-a-node"));
        assert_eq!(nodes[0].index_in_path, 0);
        assert_eq!(nodes[1].node_id, HashId::convert_str("test-t-node"));
        assert_eq!(nodes[1].index_in_path, 1);
    }

    #[test]
    fn test_get_or_create_returns_existing_accession_for_duplicate() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let new_accession = NewAccession {
            name: "test".to_string(),
            block_group_id,
            parent_accession_id: None,
            spans: vec![AccessionSpan {
                node_id: HashId::convert_str("test-a-node"),
                range: Range { start: 2, end: 4 },
                strand: Strand::Forward,
            }],
        };

        let first = Accession::create(conn, &new_accession).unwrap();
        let second = Accession::get_or_create(conn, &new_accession).unwrap();

        assert_eq!(second.id, first.id);
        assert_eq!(Accession::get_nodes_by_id(conn, &second.id, None).len(), 1);
    }

    #[test]
    fn get_or_create_rejects_duplicate_name_with_different_spans() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let new_accession = NewAccession {
            name: "test".to_string(),
            block_group_id,
            parent_accession_id: None,
            spans: vec![AccessionSpan {
                node_id: HashId::convert_str("test-a-node"),
                range: Range { start: 2, end: 4 },
                strand: Strand::Forward,
            }],
        };
        let different_spans = NewAccession {
            spans: vec![AccessionSpan {
                node_id: HashId::convert_str("test-a-node"),
                range: Range { start: 3, end: 5 },
                strand: Strand::Forward,
            }],
            ..new_accession.clone()
        };

        Accession::create(conn, &new_accession).unwrap();
        let err = Accession::get_or_create(conn, &different_spans).unwrap_err();

        assert!(matches!(err, AccessionError::Duplicate(_)));
    }

    #[test]
    fn test_spans_from_intervaltree_ranges_preserve_range_order_and_clip_blocks() {
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, path) = setup_block_group(conn);
        let tree = path.intervaltree(conn).unwrap();

        let spans = AccessionSpan::from_intervaltree_ranges(&tree, &[12..16, 2..4]).unwrap();

        assert_eq!(
            spans,
            vec![
                AccessionSpan {
                    node_id: HashId::convert_str("test-t-node"),
                    range: Range { start: 2, end: 6 },
                    strand: Strand::Forward,
                },
                AccessionSpan {
                    node_id: HashId::convert_str("test-a-node"),
                    range: Range { start: 2, end: 4 },
                    strand: Strand::Forward,
                },
            ]
        );
    }

    #[test]
    fn test_spans_from_intervaltree_ranges_errors_on_wraparound_range() {
        let conn = &get_connection(None).unwrap();
        let (_block_group_id, path) = setup_block_group(conn);
        let tree = path.intervaltree(conn).unwrap();

        let range = StdRange { start: 35, end: 3 };
        let err = AccessionSpan::from_intervaltree_ranges(&tree, std::slice::from_ref(&range))
            .unwrap_err();

        assert!(matches!(
            err,
            AccessionError::InvalidRange { start: 35, end: 3 }
        ));
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
    fn test_accession_node_to_accession_blocks_conversion() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let accession = Accession::create(
            conn,
            &NewAccession {
                name: "test".to_string(),
                block_group_id,
                parent_accession_id: None,
                spans: vec![
                    AccessionSpan {
                        node_id: HashId::convert_str("test-a-node"),
                        range: Range { start: 2, end: 4 },
                        strand: Strand::Forward,
                    },
                    AccessionSpan {
                        node_id: HashId::convert_str("test-t-node"),
                        range: Range { start: 0, end: 2 },
                        strand: Strand::Forward,
                    },
                ],
            },
        )
        .unwrap();

        assert_eq!(
            accession.blocks(conn).unwrap(),
            vec![
                NodeIntervalBlock {
                    node_id: PATH_START_NODE_ID,
                    start: i64::MIN + 1,
                    end: 0,
                    sequence_start: 0,
                    sequence_end: 0,
                    strand: Strand::Forward,
                },
                NodeIntervalBlock {
                    node_id: HashId::convert_str("test-a-node"),
                    start: 0,
                    end: 2,
                    sequence_start: 2,
                    sequence_end: 4,
                    strand: Strand::Forward,
                },
                NodeIntervalBlock {
                    node_id: HashId::convert_str("test-t-node"),
                    start: 2,
                    end: 4,
                    sequence_start: 0,
                    sequence_end: 2,
                    strand: Strand::Forward,
                },
                NodeIntervalBlock {
                    node_id: PATH_END_NODE_ID,
                    start: 4,
                    end: i64::MAX - 1,
                    sequence_start: 0,
                    sequence_end: 0,
                    strand: Strand::Forward,
                },
            ],
        );
    }

    #[test]
    fn test_query_accessions() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let accession_1 = Accession::create(
            conn,
            &NewAccession {
                name: "test-1".to_string(),
                block_group_id,
                parent_accession_id: None,
                spans: vec![
                    AccessionSpan {
                        node_id: HashId::convert_str("test-a-node"),
                        range: Range { start: 2, end: 4 },
                        strand: Strand::Forward,
                    },
                    AccessionSpan {
                        node_id: HashId::convert_str("test-t-node"),
                        range: Range { start: 0, end: 2 },
                        strand: Strand::Reverse,
                    },
                ],
            },
        )
        .unwrap();
        let accession_2 = Accession::create(
            conn,
            &NewAccession {
                name: "test-2".to_string(),
                block_group_id,
                parent_accession_id: None,
                spans: vec![AccessionSpan {
                    node_id: HashId::convert_str("test-c-node"),
                    range: Range { start: 1, end: 3 },
                    strand: Strand::Forward,
                }],
            },
        )
        .unwrap();
        let accession_1_nodes = vec![
            AccessionNodeData {
                accession_id: accession_1.id,
                node_id: HashId::convert_str("test-a-node"),
                sequence_start: 2,
                sequence_end: 4,
                strand: Strand::Forward,
                index_in_path: 0,
            },
            AccessionNodeData {
                accession_id: accession_1.id,
                node_id: HashId::convert_str("test-t-node"),
                sequence_start: 0,
                sequence_end: 2,
                strand: Strand::Reverse,
                index_in_path: 1,
            },
        ];
        let accession_2_nodes = vec![AccessionNodeData {
            accession_id: accession_2.id,
            node_id: HashId::convert_str("test-c-node"),
            sequence_start: 1,
            sequence_end: 3,
            strand: Strand::Forward,
            index_in_path: 0,
        }];

        let grouped =
            AccessionNode::query_accessions(conn, &[accession_2.id, accession_1.id]).unwrap();

        assert_eq!(
            grouped.get(&accession_1.id).unwrap(),
            &accession_1_nodes
                .into_iter()
                .map(AccessionNode::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            grouped.get(&accession_2.id).unwrap(),
            &accession_2_nodes
                .into_iter()
                .map(AccessionNode::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_create_from_edges() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let path_edges = vec![
            AugmentedEdgeData {
                edge_data: EdgeData {
                    source_node_id: PATH_START_NODE_ID,
                    source_coordinate: -1,
                    source_strand: Strand::Forward,
                    target_node_id: HashId::convert_str("test-a-node"),
                    target_coordinate: 2,
                    target_strand: Strand::Forward,
                },
                chromosome_index: 0,
                phased: 0,
            },
            AugmentedEdgeData {
                edge_data: EdgeData {
                    source_node_id: HashId::convert_str("test-a-node"),
                    source_coordinate: 4,
                    source_strand: Strand::Forward,
                    target_node_id: HashId::convert_str("test-t-node"),
                    target_coordinate: 0,
                    target_strand: Strand::Reverse,
                },
                chromosome_index: 0,
                phased: 0,
            },
            AugmentedEdgeData {
                edge_data: EdgeData {
                    source_node_id: HashId::convert_str("test-t-node"),
                    source_coordinate: 2,
                    source_strand: Strand::Reverse,
                    target_node_id: PATH_END_NODE_ID,
                    target_coordinate: -1,
                    target_strand: Strand::Forward,
                },
                chromosome_index: 0,
                phased: 0,
            },
        ];

        let accession =
            Accession::create_from_edges(conn, "test", &block_group_id, None, &path_edges).unwrap();
        let expected_nodes = vec![
            AccessionNodeData {
                accession_id: accession.id,
                node_id: HashId::convert_str("test-a-node"),
                sequence_start: 2,
                sequence_end: 4,
                strand: Strand::Forward,
                index_in_path: 0,
            },
            AccessionNodeData {
                accession_id: accession.id,
                node_id: HashId::convert_str("test-t-node"),
                sequence_start: 0,
                sequence_end: 2,
                strand: Strand::Reverse,
                index_in_path: 1,
            },
        ];

        assert_eq!(
            Accession::get_nodes_by_id(conn, &accession.id, None),
            expected_nodes
                .into_iter()
                .map(AccessionNode::from)
                .collect::<Vec<_>>(),
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
