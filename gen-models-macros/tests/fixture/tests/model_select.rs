use gen_models::{Direction, ModelSelectError, select::Connection};
use gen_models_macros_fixture::{
    CustomSourceModel, CustomSourceModelSelect, FixtureGroup, FixtureGroupSelect, FixtureSample,
    FixtureSampleSelect, IDENTIFIER_INJECTED_BRANCH, MissingTableModel, MissingTableModelSelect,
    QuotedIdentifierModel, QuotedIdentifierModelSelect, connection, insert_group, insert_sample,
};

fn branch_exists(conn: &Connection, branch_name: &str) -> bool {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM dolt_branches WHERE name = ?1)",
        [branch_name],
        |row| row.get(0),
    )
    .expect("should inspect fixture branches")
}

#[test]
fn test_name_contains_treats_like_wildcards_as_literals() {
    let conn = connection();
    for name in ["foo%", "foo_", "fooX"] {
        insert_sample(&conn, name, false);
    }

    let percent_matches = FixtureSample::select(&conn)
        .name_contains("%")
        .load()
        .expect("should match a literal percent sign");
    let underscore_matches = FixtureSample::select(&conn)
        .name_contains("_")
        .load()
        .expect("should match a literal underscore");
    let exact_percent = FixtureSample::select(&conn)
        .name("foo%")
        .load()
        .expect("should exactly match a percent sign");
    let exact_underscore = FixtureSample::select(&conn)
        .name("foo_")
        .load()
        .expect("should exactly match an underscore");
    let sql_looking_input = FixtureSample::select(&conn)
        .name("' OR dolt_branch('selector_value_injection') IS NOT NULL --")
        .load()
        .expect("should treat SQL-looking input as a value");

    assert_eq!(percent_matches[0].name, "foo%");
    assert_eq!(underscore_matches[0].name, "foo_");
    assert_eq!(exact_percent[0].name, "foo%");
    assert_eq!(exact_underscore[0].name, "foo_");
    assert!(sql_looking_input.is_empty());
    assert!(!branch_exists(&conn, "selector_value_injection"));
}

#[test]
fn test_generated_filters_order_and_paginate() {
    let conn = connection();
    for (name, is_reference) in [
        ("alpha", false),
        ("BarFooBaz", true),
        ("foo", true),
        ("QuxFood", true),
        ("zzz", false),
    ] {
        insert_sample(&conn, name, is_reference);
    }

    let matches = FixtureSample::select(&conn)
        .name_contains("o")
        .is_reference(true)
        .order_by(FixtureSampleSelect::Name, Direction::CaseInsensitiveAsc)
        .limit(2)
        .offset(1)
        .load()
        .expect("should load paginated fixture matches")
        .into_iter()
        .map(|sample| sample.name)
        .collect::<Vec<_>>();

    assert_eq!(matches, vec!["foo", "QuxFood"]);
}

#[test]
fn test_generated_field_projections_are_typed() {
    let conn = connection();
    insert_sample(&conn, "alpha", false);
    insert_sample(&conn, "beta", true);

    let names: Vec<String> = FixtureSample::select(&conn)
        .order_by(FixtureSampleSelect::Name, Direction::Asc)
        .only(FixtureSampleSelect::Name)
        .load()
        .expect("should load selected fixture names");
    let rows: Vec<(String, bool)> = FixtureSample::select(&conn)
        .order_by(FixtureSampleSelect::Name, Direction::Asc)
        .only((FixtureSampleSelect::Name, FixtureSampleSelect::IsReference))
        .load()
        .expect("should load selected fixture fields");
    let sixteen_names = FixtureSample::select(&conn)
        .name("alpha")
        .only((
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
            FixtureSampleSelect::Name,
        ))
        .load()
        .expect("should load a 16-field fixture projection");

    assert_eq!(names, vec!["alpha", "beta"]);
    assert_eq!(
        rows,
        vec![("alpha".to_string(), false), ("beta".to_string(), true)]
    );
    assert_eq!(sixteen_names[0].0, "alpha");
    assert_eq!(sixteen_names[0].15, "alpha");
}

#[test]
fn test_generated_explicit_joins_and_joined_projections() {
    let conn = connection();
    insert_sample(&conn, "sample-alpha", true);
    insert_sample(&conn, "sample-beta", false);
    insert_group(&conn, 1, "sample-alpha", "target-group", "join-test");
    insert_group(&conn, 2, "sample-beta", "other-group", "join-test");

    let samples = FixtureSample::select(&conn)
        .name_contains("sample")
        .join_filtered_on(
            FixtureSampleSelect::Name,
            FixtureGroupSelect::SampleName,
            FixtureGroup::select(&conn).name("target-group"),
        )
        .join_filtered_on(
            FixtureSampleSelect::Name,
            FixtureGroupSelect::SampleName,
            FixtureGroup::select(&conn).collection_name("join-test"),
        )
        .load()
        .expect("should load joined fixture samples");
    let groups = FixtureGroup::select(&conn)
        .name_contains("group")
        .join_filtered_on(
            FixtureGroupSelect::SampleName,
            FixtureSampleSelect::Name,
            FixtureSample::select(&conn).is_reference(true),
        )
        .load()
        .expect("should load reverse joined fixture groups");
    let projected_rows: Vec<(String, String)> = FixtureSample::select(&conn)
        .name("sample-alpha")
        .join_on(FixtureSampleSelect::Name, FixtureGroupSelect::SampleName)
        .only((FixtureSampleSelect::Name, FixtureGroupSelect::Name))
        .load()
        .expect("should load fields from joined fixture models");
    let model_rows: Vec<(FixtureSample, FixtureGroup)> = FixtureSample::select(&conn)
        .join_filtered_on(
            FixtureSampleSelect::Name,
            FixtureGroupSelect::SampleName,
            FixtureGroup::select(&conn).name("target-group"),
        )
        .models::<(FixtureSample, FixtureGroup)>()
        .load()
        .expect("should load both joined fixture models");

    assert_eq!(samples[0].name, "sample-alpha");
    assert_eq!(groups[0].name, "target-group");
    assert_eq!(
        projected_rows,
        vec![("sample-alpha".to_string(), "target-group".to_string())]
    );
    assert_eq!(model_rows[0].0.name, "sample-alpha");
    assert_eq!(model_rows[0].1.name, "target-group");
}

#[test]
fn test_generated_projections_reject_unjoined_sources() {
    let conn = connection();

    let field_error = FixtureSample::select(&conn)
        .only((FixtureSampleSelect::Name, FixtureGroupSelect::Name))
        .load()
        .expect_err("should reject an unjoined fixture field");
    let model_error = FixtureSample::select(&conn)
        .models::<(FixtureSample, FixtureGroup)>()
        .load()
        .expect_err("should reject an unjoined fixture model");
    let expected = ModelSelectError::ProjectionSourceNotSelected {
        table_name: "fixture_groups".to_string(),
        alias: "fixture_groups".to_string(),
    };

    assert_eq!(field_error, expected);
    assert_eq!(model_error, expected);
}

#[test]
fn test_generated_history_ref_applies_to_every_source() {
    let conn = connection();
    insert_sample(&conn, "historical-sample", false);
    insert_group(
        &conn,
        1,
        "historical-sample",
        "matching-group",
        "history-test",
    );
    let historical_ref: String = conn
        .query_row(
            "SELECT dolt_commit('-A', '-m', 'add historical fixture rows')",
            [],
            |row| row.get(0),
        )
        .expect("should commit historical fixture rows");
    insert_sample(&conn, "current-sample", false);
    insert_group(&conn, 2, "current-sample", "matching-group", "history-test");

    let samples = FixtureSample::select(&conn)
        .join_filtered_on(
            FixtureSampleSelect::Name,
            FixtureGroupSelect::SampleName,
            FixtureGroup::select(&conn).name("matching-group"),
        )
        .with_ref(historical_ref)
        .load()
        .expect("should load historical joined fixture samples");

    assert_eq!(samples[0].name, "historical-sample");
}

#[test]
fn test_generated_queries_quote_every_structured_identifier() {
    let conn = connection();
    assert!(!branch_exists(&conn, IDENTIFIER_INJECTED_BRANCH));

    let models = QuotedIdentifierModel::select(&conn)
        .value_contains("AF")
        .optional_value_is_null()
        .order_by(QuotedIdentifierModelSelect::Value, Direction::Asc)
        .load()
        .expect("should load a model using quoted identifiers");
    let values = QuotedIdentifierModel::select(&conn)
        .only(QuotedIdentifierModelSelect::Value)
        .load()
        .expect("should project a quoted identifier");
    let model_tuples = QuotedIdentifierModel::select(&conn)
        .models::<(QuotedIdentifierModel,)>()
        .load()
        .expect("should project a model using quoted identifiers");

    assert_eq!(models[0].value, "safe");
    assert_eq!(values, vec!["safe"]);
    assert_eq!(model_tuples[0].0.value, "safe");
    assert!(!branch_exists(&conn, IDENTIFIER_INJECTED_BRANCH));
}

#[test]
fn test_generated_loads_return_errors_and_custom_sql_remains_supported() {
    let conn = connection();

    let load_error = MissingTableModel::select(&conn)
        .load()
        .expect_err("should reject a missing selector source");
    let projection_error = MissingTableModel::select(&conn)
        .only(MissingTableModelSelect::Value)
        .load()
        .expect_err("should reject a missing projected selector source");
    let custom_models = CustomSourceModel::select(&conn)
        .value("custom")
        .load()
        .expect("should load from a custom SQL source and select list");
    let custom_values = CustomSourceModel::select(&conn)
        .only(CustomSourceModelSelect::Value)
        .load()
        .expect("should project from a custom SQL source");

    assert!(matches!(load_error, ModelSelectError::DatabaseError(_)));
    assert!(matches!(
        projection_error,
        ModelSelectError::DatabaseError(_)
    ));
    assert_eq!(custom_models[0].value, "custom");
    assert_eq!(custom_values, vec!["custom"]);
}
