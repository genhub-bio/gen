use std::{collections::HashMap, rc::Rc};

use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, calculate_hash,
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

    pub fn create_from_edges(
        conn: &GraphConnection,
        name: &str,
        block_group_id: &HashId,
        parent_accession_id: Option<&HashId>,
        edges: &[AugmentedEdgeData],
    ) -> Result<Accession, AccessionError> {
        let accession = Self::create(conn, name, block_group_id, parent_accession_id)?;
        let accession_nodes = edges
            .windows(2)
            .enumerate()
            .map(|(index, edge_pair)| {
                let into = &edge_pair[0].edge_data;
                let out_of = &edge_pair[1].edge_data;
                AccessionNodeData {
                    accession_id: accession.id,
                    node_id: into.target_node_id,
                    sequence_start: into.target_coordinate,
                    sequence_end: out_of.source_coordinate,
                    strand: into.target_strand,
                    index_in_path: index as i64,
                }
            })
            .collect::<Vec<_>>();
        AccessionNode::bulk_create(conn, &accession_nodes)?;
        Ok(accession)
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

    pub fn get_nodes_by_id(conn: &GraphConnection, accession_id: &HashId) -> Vec<AccessionNode> {
        AccessionNode::query(
            conn,
            "select * from accession_nodes where accession_id = ?1 order by index_in_path;",
            params![accession_id],
        )
    }

    pub fn blocks(&self, conn: &GraphConnection) -> Result<Vec<NodeIntervalBlock>, AccessionError> {
        let nodes = Self::get_nodes_by_id(conn, &self.id);
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
        let nodes = Self::get_nodes_by_id(conn, &self.id);
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
    fn test_accession_node_capnp_serialization() {
        let accession_node = AccessionNode {
            id: "0000000000000000000000000000030000000000000000000000000000000000"
                .try_into()
                .unwrap(),
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
    fn test_accession_node_to_accession_blocks_conversion() {
        let conn = &get_connection(None).unwrap();
        let (block_group_id, _path) = setup_block_group(conn);
        let accession = Accession::create(conn, "test", &block_group_id, None).unwrap();

        AccessionNode::bulk_create(
            conn,
            &[
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
                    strand: Strand::Forward,
                    index_in_path: 1,
                },
            ],
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
        let accession_1 = Accession::create(conn, "test-1", &block_group_id, None).unwrap();
        let accession_2 = Accession::create(conn, "test-2", &block_group_id, None).unwrap();
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
        AccessionNode::bulk_create(conn, &accession_1_nodes).unwrap();
        AccessionNode::bulk_create(conn, &accession_2_nodes).unwrap();

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
            Accession::get_nodes_by_id(conn, &accession.id),
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
