use gen_models::{ModelSelectError, select::Connection};
use gen_models_macros_fixture::{
    FixtureCompositeKey, FixtureGroup, connection, insert_composite_key, insert_group,
};
use rusqlite::limits::Limit;

fn lower_parameter_limit(conn: &Connection) {
    conn.set_limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER, 3)
        .expect("should lower the fixture parameter limit");
}

#[test]
fn test_query_by_ids_shares_the_parameter_budget_with_scalar_filters() {
    let conn = connection();
    insert_group(&conn, 1, "alpha", "selected", "collection");
    insert_group(&conn, 3, "gamma", "selected", "collection");
    lower_parameter_limit(&conn);

    let groups = FixtureGroup::select(&conn)
        .name("selected")
        .query_by_ids(1..=10)
        .expect("should bind a long ID list and scalar filter within the parameter budget");

    assert_eq!(
        groups.into_iter().map(|group| group.id).collect::<Vec<_>>(),
        vec![1, 3]
    );
}

#[test]
fn test_multiple_in_filters_share_the_connection_parameter_budget() {
    let conn = connection();
    insert_group(&conn, 1, "alpha", "first", "a");
    insert_group(&conn, 2, "beta", "second", "b");
    lower_parameter_limit(&conn);

    let groups = FixtureGroup::select(&conn)
        .name_in(["second", "missing", "first", "also-missing"])
        .collection_name_in(["b", "missing", "a", "also-missing"])
        .load()
        .expect("should bind each IN filter as one array parameter");

    assert_eq!(
        groups.into_iter().map(|group| group.id).collect::<Vec<_>>(),
        vec![2, 1]
    );
}

#[test]
fn test_pagination_shares_the_parameter_budget_with_query_by_ids() {
    let conn = connection();
    for id in 1..=3 {
        insert_group(&conn, id, "sample", "selected", "collection");
    }
    lower_parameter_limit(&conn);

    let groups = FixtureGroup::select(&conn)
        .limit(10)
        .offset(1)
        .query_by_ids(1..=10)
        .expect("should bind an ID list, limit, and offset within the parameter budget");

    assert_eq!(
        groups.into_iter().map(|group| group.id).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn test_composite_query_by_ids_shares_the_parameter_budget_with_pagination() {
    let conn = connection();
    insert_composite_key(&conn, "alpha", "one", 1);
    insert_composite_key(&conn, "beta", "two", 2);
    lower_parameter_limit(&conn);

    let keys = FixtureCompositeKey::select(&conn)
        .limit(10)
        .query_by_ids([("beta", "two"), ("alpha", "one")])
        .expect("should bind composite IDs and the limit within the parameter budget");

    assert_eq!(
        keys.into_iter()
            .map(|key| (key.namespace, key.name))
            .collect::<Vec<_>>(),
        vec![
            ("beta".to_string(), "two".to_string()),
            ("alpha".to_string(), "one".to_string()),
        ]
    );
}

#[test]
fn test_history_ref_shares_the_parameter_budget_with_query_by_ids() {
    let conn = connection();
    insert_group(&conn, 1, "sample", "selected", "collection");
    let historical_ref: String = conn
        .query_row(
            "SELECT dolt_commit('-A', '-m', 'add historical fixture row')",
            [],
            |row| row.get(0),
        )
        .expect("should commit the fixture row");
    lower_parameter_limit(&conn);

    let groups = FixtureGroup::select(&conn)
        .with_ref(historical_ref.as_str())
        .query_by_ids(1..=10)
        .expect("should bind an ID list and historical reference within the parameter budget");

    assert_eq!(
        groups.into_iter().map(|group| group.id).collect::<Vec<_>>(),
        vec![1]
    );
}

#[test]
fn test_selector_rejects_queries_that_exceed_the_parameter_budget() {
    let conn = connection();
    conn.set_limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER, 2)
        .expect("should lower the fixture parameter limit");

    let error = FixtureGroup::select(&conn)
        .name("selected")
        .name_in(["selected"])
        .collection_name_in(["collection"])
        .load()
        .expect_err(
            "should reject selectors that require more parameters than the connection allows",
        );

    assert!(
        matches!(error, ModelSelectError::InvalidSelector(message) if message.contains("SQL parameters"))
    );
}
