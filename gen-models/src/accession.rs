use std::collections::HashSet;

use gen_core::{
    HashId, Strand, calculate_hash,
    region::{Region, RegionResolutionError, RegionResolver},
    traits::Capnp,
};
use itertools::Itertools;
use rusqlite::{Row, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    block_group_edge::AugmentedEdgeData,
    db::GraphConnection,
    gen_models_capnp::{accession, accession_edge, accession_path},
    traits::*,
};

#[derive(Deserialize, Serialize, Debug, Eq, PartialEq)]
pub struct Accession {
    pub id: HashId,
    pub name: String,
    pub path_id: HashId,
    pub parent_accession_id: Option<HashId>,
}

#[derive(Debug, Error, PartialEq)]
pub enum AccessionError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    #[error("Duplicate entry with uuid: {0}")]
    Duplicate(String),
}

impl<'a> Capnp<'a> for Accession {
    type Builder = accession::Builder<'a>;
    type Reader = accession::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_id(&self.id.0).unwrap();
        builder.set_name(&self.name);
        builder.set_path_id(&self.path_id.0).unwrap();
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
        let path_id = reader
            .get_path_id()
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
            path_id,
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
    fn id_hash(path_id: &HashId, parent_accession_id: Option<&HashId>, name: &str) -> HashId {
        HashId(calculate_hash(&format!(
            "{path_id}:{parent_accession_id:?}:{name}"
        )))
    }

    pub fn create(
        conn: &GraphConnection,
        name: &str,
        path_id: &HashId,
        parent_accession_id: Option<&HashId>,
    ) -> Result<Accession, AccessionError> {
        let query = "INSERT INTO accessions (id, name, path_id, parent_accession_id) VALUES (?1, ?2, ?3, ?4);";
        let mut stmt = match conn.prepare(query) {
            Ok(s) => s,
            Err(e) => return Err(AccessionError::DatabaseError(e)),
        };

        let hash = Accession::id_hash(path_id, parent_accession_id, name);
        match stmt.execute((hash, name, path_id, parent_accession_id)) {
            Ok(_) => Ok(Accession {
                id: hash,
                name: name.to_string(),
                path_id: *path_id,
                parent_accession_id: parent_accession_id.copied(),
            }),
            Err(rusqlite::Error::SqliteFailure(err, _details))
                if err.code == rusqlite::ErrorCode::ConstraintViolation =>
            {
                Err(AccessionError::Duplicate(format!(
                    "An accession with the same name, path_id, and parent_accession_id already exists. name: {name}, path_id: {path_id}, parent_accession_id: {parent_accession_id:?}"
                )))
            }
            Err(e) => Err(AccessionError::DatabaseError(e)),
        }
    }

    pub fn get_or_create(
        conn: &GraphConnection,
        name: &str,
        path_id: &HashId,
        parent_accession_id: Option<&HashId>,
    ) -> Result<Accession, AccessionError> {
        match Accession::create(conn, name, path_id, parent_accession_id) {
            Ok(accession) => Ok(accession),
            Err(AccessionError::Duplicate(_)) => {
                let hash = Accession::id_hash(path_id, parent_accession_id, name);
                Ok(Accession {
                    id: hash,
                    name: name.to_string(),
                    path_id: *path_id,
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
             JOIN paths p ON a.path_id = p.id \
             JOIN block_groups bg ON p.block_group_id = bg.id \
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
            path_id: row.get(2).unwrap(),
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

    use super::*;
    use crate::test_helpers::{get_connection, setup_block_group};

    #[test]
    fn test_accession_capnp_serialization() {
        let accession = Accession {
            id: "0000000000000000000000000000000000000000000000000000000000000200"
                .try_into()
                .unwrap(),
            name: "test_accession".to_string(),
            path_id: "0000000000000000000000000000000000000000000000000000000000000150"
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
            path_id: "0000000000000000000000000000000000000000000000000000000000000151"
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
        let (_bg, path) = setup_block_group(conn);
        let accession = Accession::create(conn, "test", &path.id, None).unwrap();
        let _accession_2 = Accession::create(conn, "test2", &path.id, None).unwrap();
        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                params!["test"],
            ),
            vec![Accession {
                id: accession.id,
                name: "test".to_string(),
                path_id: path.id,
                parent_accession_id: None,
            }]
        );
    }
}
