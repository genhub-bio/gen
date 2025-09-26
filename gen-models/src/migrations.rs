use include_dir::{Dir, include_dir};
use rusqlite::Connection;
use rusqlite_migration::Migrations;

static MIGRATION_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../migrations/core");
static OPERATIONS_MIGRATION_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../migrations/operations");

pub fn run_migrations(conn: &mut Connection) {
    let migrations = Migrations::from_directory(&MIGRATION_DIR).unwrap();

    // Apply some PRAGMA, often better to do it outside of migrations
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_| Ok(()))
        .unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    conn.execute("PRAGMA cache_size=50000;", []).unwrap();
    // synchronous = NORMAL should be fine with WAL mode, and helps with performance
    // https://developer.android.com/topic/performance/sqlite-performance-best-practices
    conn.execute("PRAGMA synchronous = NORMAL;", []).unwrap();

    // 2️⃣ Update the database schema, atomically
    let r = migrations.to_latest(conn);
    r.unwrap()
}

pub fn run_operation_migrations(conn: &mut Connection) {
    let migrations = Migrations::from_directory(&OPERATIONS_MIGRATION_DIR).unwrap();

    // Apply some PRAGMA, often better to do it outside of migrations
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_| Ok(()))
        .unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    conn.execute("PRAGMA cache_size=50000;", []).unwrap();

    // 2️⃣ Update the database schema, atomically
    let r = migrations.to_latest(conn);
    r.unwrap()
}
