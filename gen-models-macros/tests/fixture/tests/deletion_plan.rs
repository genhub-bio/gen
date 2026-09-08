use std::{cell::RefCell, rc::Rc};

use gen_models_macros_fixture::{
    FixtureCompositeKey, FixtureSample, connection, insert_composite_key, insert_sample,
};
use rusqlite::{
    trace::{TraceEvent, TraceEventCodes},
    types::Value,
};

thread_local! {
    static TRACED_STATEMENTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

fn capture_statement(event: TraceEvent<'_>) {
    if let TraceEvent::Stmt(statement, _) = event {
        TRACED_STATEMENTS.with(|statements| statements.borrow_mut().push(statement.sql().into()));
    }
}

fn traced_delete_statement() -> String {
    TRACED_STATEMENTS
        .with(|statements| {
            statements
                .borrow()
                .iter()
                .find(|statement| statement.starts_with("DELETE FROM"))
                .cloned()
        })
        .expect("should trace the generated deletion statement")
}

#[test]
fn test_delete_by_ids_uses_the_primary_key_index() {
    let conn = connection();
    insert_sample(&conn, "alpha", false);

    TRACED_STATEMENTS.with(|statements| statements.borrow_mut().clear());
    conn.trace_v2(TraceEventCodes::SQLITE_TRACE_STMT, Some(capture_statement));
    FixtureSample::select(&conn)
        .delete_by_ids(["alpha"])
        .expect("should delete the fixture sample");
    conn.trace_v2(TraceEventCodes::empty(), None);

    let query = traced_delete_statement();
    let explain = format!("EXPLAIN QUERY PLAN {query}");
    let mut statement = conn
        .prepare(&explain)
        .expect("should prepare the deletion query plan");
    let plan = statement
        .query_map([Rc::new(vec![Value::from("alpha".to_string())])], |row| {
            row.get::<_, String>(3)
        })
        .expect("should inspect the deletion query plan")
        .collect::<rusqlite::Result<Vec<_>>>()
        .expect("should read the deletion query plan");

    assert!(
        plan.iter()
            .any(|detail| detail.contains("SEARCH fixture_samples")),
        "expected an indexed primary-key lookup, got: {plan:?}",
    );
    assert!(
        !plan
            .iter()
            .any(|detail| detail.contains("SCAN fixture_samples")),
        "deletion must not scan the complete table: {plan:?}",
    );
}

#[test]
fn test_composite_delete_by_ids_uses_the_primary_key_index() {
    let conn = connection();
    insert_composite_key(&conn, "alpha", "one", 1);
    insert_composite_key(&conn, "alpha", "two", 2);
    insert_composite_key(&conn, "beta", "one", 3);
    insert_composite_key(&conn, "beta", "two", 4);

    TRACED_STATEMENTS.with(|statements| statements.borrow_mut().clear());
    conn.trace_v2(TraceEventCodes::SQLITE_TRACE_STMT, Some(capture_statement));
    let deleted = FixtureCompositeKey::select(&conn)
        .delete_by_ids([("alpha", "one"), ("beta", "two")])
        .expect("should delete the composite fixture key");
    conn.trace_v2(TraceEventCodes::empty(), None);

    let explain = format!("EXPLAIN QUERY PLAN {}", traced_delete_statement());
    let mut statement = conn
        .prepare(&explain)
        .expect("should prepare the composite deletion query plan");
    let plan = statement
        .query_map(
            [
                Rc::new(vec![
                    Value::from("alpha".to_string()),
                    Value::from("beta".to_string()),
                ]),
                Rc::new(vec![
                    Value::from("one".to_string()),
                    Value::from("two".to_string()),
                ]),
            ],
            |row| row.get::<_, String>(3),
        )
        .expect("should inspect the composite deletion query plan")
        .collect::<rusqlite::Result<Vec<_>>>()
        .expect("should read the composite deletion query plan");

    assert!(
        plan.iter().any(|detail| {
            detail.contains("SEARCH fixture_composite_keys")
                && detail.contains("namespace=? AND name=?")
        }),
        "expected an indexed composite primary-key lookup, got: {plan:?}",
    );
    assert!(
        !plan
            .iter()
            .any(|detail| detail.contains("SCAN fixture_composite_keys")),
        "composite deletion must not scan the complete table: {plan:?}",
    );
    assert_eq!(deleted, 2);
    assert_eq!(
        FixtureCompositeKey::select(&conn)
            .load()
            .expect("should load the surviving composite fixture keys")
            .into_iter()
            .map(|key| (key.namespace, key.name))
            .collect::<Vec<_>>(),
        vec![
            ("beta".to_string(), "one".to_string()),
            ("alpha".to_string(), "two".to_string()),
        ],
        "a composite deletion must preserve keys assembled from different input rows",
    );
}
