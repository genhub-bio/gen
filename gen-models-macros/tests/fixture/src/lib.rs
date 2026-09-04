//! Fake persistence models used to exercise the generated selector API.

use gen_models::select::{Connection, Row};
use gen_models_macros::ModelSelect;

pub const IDENTIFIER_INJECTED_BRANCH: &str = "selector_identifier_injection";

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(table = "fixture_samples")]
pub struct FixtureSample {
    #[model_select(primary_key)]
    pub name: String,
    pub is_reference: bool,
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(table = "fixture_groups")]
pub struct FixtureGroup {
    pub id: i64,
    pub sample_name: String,
    pub name: String,
    pub collection_name: String,
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(table = "selector table\" --", alias = "selector alias\" --")]
pub struct QuotedIdentifierModel {
    #[model_select(column = "value\" || dolt_branch('selector_identifier_injection') || \"")]
    pub value: String,
    #[model_select(column = "optional value")]
    pub optional_value: Option<String>,
}

fn missing_table_source(_history_ref: Option<&str>) -> String {
    "missing_fixture_models AS missing_fixture_models".to_string()
}

#[derive(Debug, ModelSelect)]
#[model_select(table = "missing_fixture_models", source = missing_table_source)]
pub struct MissingTableModel {
    pub value: i64,
}

fn custom_source(_history_ref: Option<&str>) -> String {
    "(SELECT 'custom' AS value) AS custom_rows".to_string()
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(
    table = "custom_source_models",
    alias = "custom_rows",
    source = custom_source,
    select = "\"custom_rows\".\"value\""
)]
pub struct CustomSourceModel {
    pub value: String,
}

fn derived_model_from_row(row: &Row) -> DerivedModel {
    let value: String = row.get("value").expect("should read derived model value");
    DerivedModel {
        uppercase: value.to_uppercase(),
        value,
    }
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(table = "derived_models", history = false, from_row = derived_model_from_row)]
pub struct DerivedModel {
    pub value: String,
    #[model_select(skip)]
    pub uppercase: String,
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
        CREATE TABLE derived_models (
            value TEXT NOT NULL
        );
        INSERT INTO derived_models (value) VALUES ('derived');
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
