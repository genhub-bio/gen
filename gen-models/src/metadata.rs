use gen_core::traits::Capnp;
use rusqlite::Row;

use crate::{db::GraphConnection, gen_models_capnp::metadata, traits::*};

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub db_uuid: String,
}

impl<'a> Capnp<'a> for Metadata {
    type Builder = metadata::Builder<'a>;
    type Reader = metadata::Reader<'a>;

    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.set_db_uuid(&self.db_uuid);
    }

    fn read_capnp(reader: Self::Reader) -> Self {
        let db_uuid = reader.get_db_uuid().unwrap().to_string().unwrap();
        Metadata { db_uuid }
    }
}

impl Query for Metadata {
    type Model = Metadata;

    const TABLE_NAME: &'static str = "gen_metadata";

    fn process_row(row: &Row) -> Self::Model {
        Metadata {
            db_uuid: row.get(0).unwrap(),
        }
    }
}

impl Metadata {
    pub fn get_db_uuid(conn: &GraphConnection) -> String {
        let metadata = Metadata::get(conn, "SELECT db_uuid FROM gen_metadata LIMIT 1", [])
            .expect("Failed to get database UUID from metadata");
        metadata.db_uuid
    }

    pub fn get_all(conn: &GraphConnection) -> Vec<Metadata> {
        Metadata::query(conn, "SELECT db_uuid FROM gen_metadata", [])
    }
}

// Keep the old function for backwards compatibility
pub fn get_db_uuid(conn: &GraphConnection) -> String {
    Metadata::get_db_uuid(conn)
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use capnp::message::TypedBuilder;

    use super::*;
    use crate::test_helpers::{get_connection, setup_gen};

    #[test]
    fn test_metadata_capnp_serialization() {
        let metadata = Metadata {
            db_uuid: "test-uuid-12345".to_string(),
        };

        let mut message = TypedBuilder::<metadata::Owned>::new_default();
        let mut root = message.init_root();
        metadata.write_capnp(&mut root);

        let deserialized = Metadata::read_capnp(root.into_reader());
        assert_eq!(metadata, deserialized);
    }

    #[test]
    fn test_sets_uuid() {
        let conn = get_connection(None).unwrap();
        assert!(!get_db_uuid(&conn).is_empty());
    }

    #[test]
    fn test_sets_uuid_with_db_context() {
        let ctx = setup_gen();
        let conn = ctx.graph().conn();

        assert!(!get_db_uuid(conn).is_empty());
    }

    #[test]
    fn test_table_name_constants() {
        use crate::{
            accession::{Accession, AccessionEdge, AccessionPath},
            block_group::BlockGroup,
            block_group_edge::BlockGroupEdge,
            collection::Collection,
            edge::Edge,
            files::GenDatabase,
            metadata::Metadata,
            node::Node,
            operations::{Branch, FileAddition, Operation, OperationSummary},
            path::Path,
            path_edge::PathEdge,
            sample::Sample,
        };

        // Test core models
        assert_eq!(Node::table_name(), "nodes");
        assert_eq!(Collection::table_name(), "collections");
        assert_eq!(Sample::table_name(), "samples");

        // Test graph models
        assert_eq!(Edge::table_name(), "edges");
        assert_eq!(Path::table_name(), "paths");
        assert_eq!(PathEdge::table_name(), "path_edges");
        assert_eq!(BlockGroup::table_name(), "block_groups");
        assert_eq!(BlockGroupEdge::table_name(), "block_group_edges");

        // Test accession models
        assert_eq!(Accession::table_name(), "accessions");
        assert_eq!(AccessionEdge::table_name(), "accession_edges");
        assert_eq!(AccessionPath::table_name(), "accession_paths");

        // Test operation models
        assert_eq!(Operation::table_name(), "operations");
        assert_eq!(FileAddition::table_name(), "file_additions");
        assert_eq!(OperationSummary::table_name(), "operation_summaries");
        assert_eq!(Branch::table_name(), "branches");

        // Test remaining models
        assert_eq!(GenDatabase::table_name(), "gen_databases");
        assert_eq!(Metadata::table_name(), "gen_metadata");
    }
}
