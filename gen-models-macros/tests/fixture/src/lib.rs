//! Fake persistence models used to exercise the generated selector API.

use std::cell::Cell;

use gen_models::select::{Connection, Row};
use gen_models_macros::ModelSelect;
use rusqlite::{
    ToSql,
    types::{FromSql, FromSqlResult, ToSqlOutput, ValueRef},
};

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
#[model_select(table = "fixture_composite_keys")]
pub struct FixtureCompositeKey {
    #[model_select(primary_key, default_sort = "desc")]
    pub namespace: String,
    #[model_select(primary_key, default_sort = "asc")]
    pub name: String,
    pub position: i64,
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(table = "selector table\" --", alias = "selector alias\" --")]
pub struct QuotedIdentifierModel {
    #[model_select(
        primary_key,
        column = "value\" || dolt_branch('selector_identifier_injection') || \""
    )]
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

fn derived_model_from_row(row: &Row) -> rusqlite::Result<DerivedModel> {
    let value: String = row.get("value")?;
    Ok(DerivedModel {
        uppercase: value.to_uppercase(),
        value,
    })
}

#[derive(Debug, ModelSelect, PartialEq)]
#[model_select(table = "derived_models", history = false, from_row = derived_model_from_row)]
pub struct DerivedModel {
    pub value: String,
    #[model_select(skip)]
    pub uppercase: String,
}

#[derive(Debug, ModelSelect)]
#[model_select(table = "invalid_rows")]
pub struct InvalidRow {
    pub value: String,
}

#[derive(Debug)]
pub struct FailingSqlValue(pub i64);

impl FromSql for FailingSqlValue {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        value.as_i64().map(Self)
    }
}

impl ToSql for FailingSqlValue {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Err(rusqlite::Error::ToSqlConversionFailure(Box::new(
            std::io::Error::other("fixture selector conversion failure"),
        )))
    }
}

#[derive(Debug, ModelSelect)]
#[model_select(table = "failing_sql_values")]
pub struct FailingSqlValueModel {
    pub value: FailingSqlValue,
}

thread_local! {
    static PROCESSED_ROW_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn counted_model_from_row(row: &Row) -> rusqlite::Result<CountedModel> {
    PROCESSED_ROW_COUNT.set(PROCESSED_ROW_COUNT.get() + 1);
    Ok(CountedModel { id: row.get("id")? })
}

#[derive(Debug, ModelSelect)]
#[model_select(table = "counted_models", from_row = counted_model_from_row)]
pub struct CountedModel {
    pub id: i64,
}

pub fn reset_processed_row_count() {
    PROCESSED_ROW_COUNT.set(0);
}

pub fn processed_row_count() -> usize {
    PROCESSED_ROW_COUNT.get()
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
        CREATE TABLE fixture_composite_keys (
            namespace TEXT NOT NULL,
            name TEXT NOT NULL,
            position INTEGER NOT NULL,
            PRIMARY KEY (namespace, name)
        );
        CREATE TABLE derived_models (
            value TEXT NOT NULL
        );
        INSERT INTO derived_models (value) VALUES ('derived');
        CREATE TABLE invalid_rows (
            value INTEGER NOT NULL
        );
        INSERT INTO invalid_rows (value) VALUES (42);
        CREATE TABLE failing_sql_values (
            value INTEGER NOT NULL
        );
        CREATE TABLE counted_models (
            id INTEGER PRIMARY KEY
        );
        INSERT INTO counted_models (id) VALUES (1), (2), (3), (4);
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

pub fn insert_composite_key(conn: &Connection, namespace: &str, name: &str, position: i64) {
    conn.execute(
        "INSERT INTO fixture_composite_keys (namespace, name, position) VALUES (?1, ?2, ?3)",
        (namespace, name, position),
    )
    .expect("should insert fixture composite key");
}
