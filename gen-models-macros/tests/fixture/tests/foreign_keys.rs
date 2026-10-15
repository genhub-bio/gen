use std::sync::Arc;

use gen_models::ModelSelectError;
use gen_models_macros_fixture::{
    FixtureForeignTarget, FixtureNullableRelation, FixtureRequiredRelation, connection,
};

#[test]
fn test_required_foreign_key_cache_and_setter_synchronize_id() {
    let conn = connection();
    let cached = Arc::new(FixtureForeignTarget {
        id: 42,
        name: "runtime-target".to_string(),
        history_ref: None,
    });
    let mut relation = FixtureRequiredRelation {
        id: 9,
        target_id: 1,
        cached_target: Default::default(),
        history_ref: None,
    };

    relation.set_target(Arc::clone(&cached));

    assert_eq!(relation.target_id, 42);
    conn.execute("DROP TABLE fixture_foreign_targets", [])
        .expect("should drop target table to prove cache-only lookup");
    let resolved = relation
        .get_target(&conn)
        .expect("should use the attached target without SQL");
    assert!(Arc::ptr_eq(&resolved, &cached));
}

#[test]
fn test_required_foreign_key_getter_loads_rows_and_reports_missing_target() {
    let conn = connection();
    let relation = FixtureRequiredRelation::select(&conn)
        .id(1)
        .get()
        .expect("should load required relation")
        .expect("should find required relation");
    let target = relation
        .get_target(&conn)
        .expect("should load target by foreign key");
    assert_eq!(target.id, 1);
    assert_eq!(target.name, "target-one");

    let missing_relation = FixtureRequiredRelation::select(&conn)
        .id(2)
        .get()
        .expect("should load dangling relation")
        .expect("should find dangling relation");
    assert!(matches!(
        missing_relation.get_target(&conn),
        Err(ModelSelectError::DatabaseError(
            rusqlite::Error::QueryReturnedNoRows
        ))
    ));
}

#[test]
fn test_required_foreign_key_getter_caches_database_load() {
    let conn = connection();
    let relation = FixtureRequiredRelation::select(&conn)
        .id(1)
        .get()
        .expect("should load required relation")
        .expect("should find required relation");
    let loaded = relation
        .get_target(&conn)
        .expect("should load target by foreign key");
    let cached = relation
        .cached_target
        .snapshot()
        .expect("should retain successful target load");
    assert!(Arc::ptr_eq(&loaded, &cached));

    conn.execute("DROP TABLE fixture_foreign_targets", [])
        .expect("should drop target table after loading target");
    let loaded_again = relation
        .get_target(&conn)
        .expect("should return cached target without querying dropped table");
    assert!(Arc::ptr_eq(&loaded, &loaded_again));
}

#[test]
fn test_failed_foreign_key_load_is_not_cached_and_can_recover() {
    let conn = connection();
    let relation = FixtureRequiredRelation::select(&conn)
        .id(2)
        .get()
        .expect("should load dangling relation")
        .expect("should find dangling relation");

    assert!(matches!(
        relation.get_target(&conn),
        Err(ModelSelectError::DatabaseError(
            rusqlite::Error::QueryReturnedNoRows
        ))
    ));
    assert!(relation.cached_target.snapshot().is_none());

    conn.execute(
        "INSERT INTO fixture_foreign_targets (id, name) VALUES (99, 'recovered-target')",
        [],
    )
    .expect("should insert the previously missing target");
    let recovered = relation
        .get_target(&conn)
        .expect("should retry a previously failed target load");
    assert_eq!(recovered.name, "recovered-target");
    assert!(Arc::ptr_eq(
        &recovered,
        &relation
            .cached_target
            .snapshot()
            .expect("should cache recovered target")
    ));
}

#[test]
fn test_foreign_key_getter_ignores_a_stale_cached_relation() {
    let conn = connection();
    let mut relation = FixtureRequiredRelation::select(&conn)
        .id(1)
        .get()
        .expect("should load relation")
        .expect("should find relation");
    let stale = Arc::new(
        FixtureForeignTarget::select(&conn)
            .get_by_id(1)
            .expect("should query old target")
            .expect("should find old target"),
    );
    relation.set_target(Arc::clone(&stale));
    let mut cloned_relation = relation.clone();
    assert!(Arc::ptr_eq(
        &stale,
        &cloned_relation
            .cached_target
            .snapshot()
            .expect("should clone the existing target Arc")
    ));
    cloned_relation.target_id = 2;

    let current = cloned_relation
        .get_target(&conn)
        .expect("should load target matching the cloned foreign key");
    assert_eq!(current.id, 2);
    assert!(!Arc::ptr_eq(&current, &stale));
    assert!(Arc::ptr_eq(
        &stale,
        &relation
            .cached_target
            .snapshot()
            .expect("should leave the original cache independent")
    ));

    relation.target_id = 2;

    let original_current = relation
        .get_target(&conn)
        .expect("should load target matching the current foreign key");
    assert_eq!(original_current.id, 2);
    assert_eq!(original_current.name, "target-two");
    assert!(!Arc::ptr_eq(&original_current, &stale));
    assert!(Arc::ptr_eq(
        &original_current,
        &relation
            .cached_target
            .snapshot()
            .expect("should retain the replacement target")
    ));
    assert!(Arc::ptr_eq(
        &original_current,
        &relation
            .get_target(&conn)
            .expect("should reuse the replacement target")
    ));
}

#[test]
fn test_nullable_foreign_key_supports_absent_and_present_relations() {
    let conn = connection();
    conn.execute(
        "INSERT INTO fixture_nullable_relations (id, optional_target_id) VALUES (3, 1)",
        [],
    )
    .expect("should insert nullable relation with a target");
    let mut relation = FixtureNullableRelation::select(&conn)
        .id(1)
        .get()
        .expect("should load nullable relation")
        .expect("should find nullable relation");
    assert_eq!(
        relation
            .get_optional_target(&conn)
            .expect("should read absent optional relation"),
        None
    );

    let uncached_relation = FixtureNullableRelation::select(&conn)
        .id(3)
        .get()
        .expect("should load nullable relation with target")
        .expect("should find nullable relation with target");
    let loaded_target = uncached_relation
        .get_optional_target(&conn)
        .expect("should load optional target from database")
        .expect("should have optional target");
    assert_eq!(loaded_target.id, 1);
    assert_eq!(loaded_target.name, "target-one");
    assert!(Arc::ptr_eq(
        &loaded_target,
        &uncached_relation
            .cached_target
            .snapshot()
            .expect("should cache loaded optional target")
    ));
    conn.execute("DROP TABLE fixture_foreign_targets", [])
        .expect("should drop target table after optional target load");
    assert!(Arc::ptr_eq(
        &loaded_target,
        &uncached_relation
            .get_optional_target(&conn)
            .expect("should reuse cached optional target")
            .expect("should retain optional target")
    ));

    let target = loaded_target;
    relation.set_optional_target(Some(Arc::clone(&target)));
    assert_eq!(relation.optional_target_id, Some(1));
    assert!(Arc::ptr_eq(
        &relation
            .get_optional_target(&conn)
            .expect("should return attached optional target")
            .expect("should have optional target"),
        &target
    ));

    relation.set_optional_target(None);
    assert_eq!(relation.optional_target_id, None);
    assert!(relation.cached_target.snapshot().is_none());
    assert!(
        relation
            .get_optional_target(&conn)
            .expect("should clear optional relation")
            .is_none()
    );
}

#[test]
fn test_nullable_foreign_key_none_skips_database_lookup() {
    let conn = connection();
    let relation = FixtureNullableRelation::select(&conn)
        .id(1)
        .get()
        .expect("should load nullable relation")
        .expect("should find nullable relation");
    conn.execute("DROP TABLE fixture_foreign_targets", [])
        .expect("should drop target table before reading null relation");

    assert!(
        relation
            .get_optional_target(&conn)
            .expect("should return None without querying absent target table")
            .is_none()
    );
    assert!(relation.cached_target.snapshot().is_none());
}

#[test]
fn test_nullable_foreign_key_reports_missing_target_for_non_null_id() {
    let conn = connection();
    let relation = FixtureNullableRelation::select(&conn)
        .id(2)
        .get()
        .expect("should load dangling nullable relation")
        .expect("should find dangling nullable relation");

    assert!(matches!(
        relation.get_optional_target(&conn),
        Err(ModelSelectError::DatabaseError(
            rusqlite::Error::QueryReturnedNoRows
        ))
    ));
}

#[test]
fn test_required_foreign_key_uses_the_owning_models_history_ref() {
    let conn = connection();
    let historical_ref: String = conn
        .query_row(
            "SELECT dolt_commit('-A', '-m', 'save historical relation targets')",
            [],
            |row| row.get(0),
        )
        .expect("should commit the historical relation fixture");

    conn.execute(
        "UPDATE fixture_foreign_targets SET name = 'current-target' WHERE id = 1",
        [],
    )
    .expect("should update the target in the current view");
    conn.execute(
        "INSERT INTO fixture_foreign_targets (id, name) VALUES (99, 'current-only-target')",
        [],
    )
    .expect("should add a target that is absent from the historical view");

    let relation = FixtureRequiredRelation::select(&conn)
        .id(1)
        .with_ref(historical_ref.as_str())
        .get()
        .expect("should load the historical relation")
        .expect("should find the historical relation");
    assert_eq!(relation.history_ref(), Some(historical_ref.as_str()));
    assert_eq!(
        relation.clone().history_ref(),
        Some(historical_ref.as_str())
    );

    let target = relation
        .get_target(&conn)
        .expect("should load the related target from the same historical view");
    assert_eq!(target.name, "target-one");
    assert_eq!(target.history_ref(), Some(historical_ref.as_str()));

    let dangling_relation = FixtureRequiredRelation::select(&conn)
        .id(2)
        .with_ref(historical_ref.as_str())
        .get()
        .expect("should load the historical relation with a missing target")
        .expect("should find the historical dangling relation");
    assert!(matches!(
        dangling_relation.get_target(&conn),
        Err(ModelSelectError::DatabaseError(
            rusqlite::Error::QueryReturnedNoRows
        ))
    ));
    assert!(dangling_relation.cached_target.snapshot().is_none());
}

#[test]
fn test_nullable_foreign_key_uses_the_owning_models_history_ref() {
    let conn = connection();
    conn.execute(
        "UPDATE fixture_nullable_relations SET optional_target_id = 1 WHERE id = 1",
        [],
    )
    .expect("should point the nullable relation at a target before committing");
    let historical_ref: String = conn
        .query_row(
            "SELECT dolt_commit('-A', '-m', 'save historical nullable relation target')",
            [],
            |row| row.get(0),
        )
        .expect("should commit the historical nullable relation fixture");

    conn.execute(
        "UPDATE fixture_foreign_targets SET name = 'current-target' WHERE id = 1",
        [],
    )
    .expect("should update the target in the current view");
    conn.execute(
        "INSERT INTO fixture_foreign_targets (id, name) VALUES (99, 'current-only-target')",
        [],
    )
    .expect("should add a target that is absent from the historical view");

    let relation = FixtureNullableRelation::select(&conn)
        .id(1)
        .with_ref(historical_ref.as_str())
        .get()
        .expect("should load the historical nullable relation")
        .expect("should find the historical nullable relation");
    let target = relation
        .get_optional_target(&conn)
        .expect("should load the related target from the same historical view")
        .expect("should find the historical target");
    assert_eq!(target.name, "target-one");
    assert_eq!(target.history_ref(), Some(historical_ref.as_str()));

    let null_relation = FixtureNullableRelation::select(&conn)
        .id(1)
        .with_ref(None::<&str>)
        .get()
        .expect("should load the current nullable relation")
        .expect("should find the current nullable relation");
    assert_eq!(null_relation.history_ref(), None);
}
