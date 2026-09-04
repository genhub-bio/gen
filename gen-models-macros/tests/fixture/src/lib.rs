//! Fake persistence models used to exercise the generated selector API.

use gen_models::{
    select::{Connection, Row},
    traits::Query,
};
use gen_models_macros::ModelSelect;

pub const IDENTIFIER_INJECTED_BRANCH: &str = "selector_identifier_injection";

#[derive(Debug, ModelSelect, PartialEq)]
pub struct FixtureSample {
    #[model_select(primary_key)]
    pub name: String,
    pub is_reference: bool,
}

impl Query for FixtureSample {
    type Model = Self;

    const TABLE_NAME: &'static str = "fixture_samples";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            name: row.get(0).expect("should read fixture sample name"),
            is_reference: row.get(1).expect("should read fixture reference flag"),
        }
    }
}

#[derive(Debug, ModelSelect, PartialEq)]
pub struct FixtureGroup {
    pub id: i64,
    pub sample_name: String,
    pub name: String,
    pub collection_name: String,
}

impl Query for FixtureGroup {
    type Model = Self;

    const TABLE_NAME: &'static str = "fixture_groups";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            id: row.get(0).expect("should read fixture group id"),
            sample_name: row.get(1).expect("should read fixture group sample name"),
            name: row.get(2).expect("should read fixture group name"),
            collection_name: row.get(3).expect("should read fixture collection name"),
        }
    }
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(alias = "selector alias\" --")]
pub struct QuotedIdentifierModel {
    #[model_select(column = "value\" || dolt_branch('selector_identifier_injection') || \"")]
    pub value: String,
    #[model_select(column = "optional value")]
    pub optional_value: Option<String>,
}

impl Query for QuotedIdentifierModel {
    type Model = Self;

    const TABLE_NAME: &'static str = "selector table\" --";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            value: row.get(0).expect("should read quoted value"),
            optional_value: row.get(1).expect("should read optional quoted value"),
        }
    }
}

fn missing_table_source(_history_ref: Option<&str>) -> String {
    "missing_fixture_models AS missing_fixture_models".to_string()
}

#[derive(Debug, ModelSelect)]
#[model_select(source = missing_table_source)]
pub struct MissingTableModel {
    pub value: i64,
}

impl Query for MissingTableModel {
    type Model = Self;

    const TABLE_NAME: &'static str = "missing_fixture_models";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            value: row.get(0).expect("should read missing model value"),
        }
    }
}

fn custom_source(_history_ref: Option<&str>) -> String {
    "(SELECT 'custom' AS value) AS custom_rows".to_string()
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(
    alias = "custom_rows",
    source = custom_source,
    select = "\"custom_rows\".\"value\""
)]
pub struct CustomSourceModel {
    pub value: String,
}

impl Query for CustomSourceModel {
    type Model = Self;

    const TABLE_NAME: &'static str = "custom_source_models";

    fn process_row(row: &Row) -> Self::Model {
        Self {
            value: row.get(0).expect("should read custom source value"),
        }
    }
}

pub fn connection() -> Connection {
    let conn = Connection::open_in_memory().expect("should open fixture database");
    rusqlite::vtab::array::load_module(&conn).expect("should load the fixture rarray module");
    conn.execute_batch(
        r#"
        CREATE TABLE fixture_samples (
            name TEXT PRIMARY KEY,
            is_reference BOOLEAN NOT NULL
        );
        CREATE TABLE fixture_groups (
            id INTEGER PRIMARY KEY,
            sample_name TEXT NOT NULL,
            name TEXT NOT NULL,
            collection_name TEXT NOT NULL
        );
        CREATE TABLE "selector table"" --" (
            "value"" || dolt_branch('selector_identifier_injection') || """ TEXT NOT NULL,
            "optional value" TEXT
        );
        INSERT INTO "selector table"" --" (
            "value"" || dolt_branch('selector_identifier_injection') || """,
            "optional value"
        ) VALUES ('safe', NULL);
        "#,
    )
    .expect("should create fixture tables");
    conn
}

pub fn insert_sample(conn: &Connection, name: &str, is_reference: bool) {
    conn.execute(
        "INSERT INTO fixture_samples (name, is_reference) VALUES (?1, ?2)",
        (name, is_reference),
    )
    .expect("should insert fixture sample");
}

pub fn insert_group(
    conn: &Connection,
    id: i64,
    sample_name: &str,
    name: &str,
    collection_name: &str,
) {
    conn.execute(
        "INSERT INTO fixture_groups (id, sample_name, name, collection_name) VALUES (?1, ?2, ?3, ?4)",
        (id, sample_name, name, collection_name),
    )
    .expect("should insert fixture group");
}
