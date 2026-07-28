use include_dir::{Dir, include_dir};
use rusqlite::Connection;
use rusqlite_migration::Migrations;
use thiserror::Error;

static MIGRATION_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/migrations/core");
static CONFIG_MIGRATION_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/migrations/config");

const MIGRATION_COMMIT_MESSAGE: &str = "Apply Gen schema migrations";

#[derive(Debug, Error)]
enum MigrationError {
    #[error(transparent)]
    Migration(#[from] rusqlite_migration::Error),
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
    #[error(
        "cannot apply Gen schema migrations while the Dolt working set has uncommitted changes"
    )]
    DirtyWorkingSet,
}

pub fn run_migrations(conn: &mut Connection) {
    apply_graph_migrations(conn).unwrap();
}

fn apply_graph_migrations(conn: &mut Connection) -> Result<bool, MigrationError> {
    let migrations = Migrations::from_directory(&MIGRATION_DIR)?;

    // Apply some PRAGMA, often better to do it outside of migrations
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_| Ok(()))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.execute("PRAGMA cache_size=50000;", [])?;
    // synchronous = NORMAL should be fine with WAL mode, and helps with performance
    // https://developer.android.com/topic/performance/sqlite-performance-best-practices
    conn.execute("PRAGMA synchronous = NORMAL;", [])?;

    let pending_migrations = migrations.pending_migrations(conn)?;
    if pending_migrations > 0 {
        let changed_tables: i64 =
            conn.query_row("SELECT COUNT(*) FROM dolt_status", [], |row| row.get(0))?;
        if changed_tables != 0 {
            return Err(MigrationError::DirtyWorkingSet);
        }
    }

    // 2️⃣ Update the database schema, atomically
    migrations.to_latest(conn)?;

    if pending_migrations > 0 {
        conn.query_row(
            "SELECT dolt_commit('-A', '-m', ?1)",
            [MIGRATION_COMMIT_MESSAGE],
            |row| row.get_ref(0).map(|_| ()),
        )?;
    }

    Ok(pending_migrations > 0)
}

pub fn run_config_migrations(conn: &mut Connection) {
    let migrations = Migrations::from_directory(&CONFIG_MIGRATION_DIR).unwrap();

    // Apply some PRAGMA, often better to do it outside of migrations
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |_| Ok(()))
        .unwrap();
    conn.pragma_update(None, "foreign_keys", "ON").unwrap();
    conn.execute("PRAGMA cache_size=50000;", []).unwrap();

    // 2️⃣ Update the database schema, atomically
    let r = migrations.to_latest(conn);
    r.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn migration_commit_count(conn: &Connection) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM dolt_log WHERE message = ?1",
            [MIGRATION_COMMIT_MESSAGE],
            |row| row.get(0),
        )
        .expect("should count migration commits")
    }

    #[test]
    fn test_graph_migrations_are_committed_and_leave_a_clean_working_set() {
        let mut conn = Connection::open_in_memory().expect("should open database");

        assert!(apply_graph_migrations(&mut conn).expect("should apply migrations"));

        let changed_tables: i64 = conn
            .query_row("SELECT COUNT(*) FROM dolt_status", [], |row| row.get(0))
            .expect("should read Dolt status");
        assert_eq!(changed_tables, 0);
        assert_eq!(migration_commit_count(&conn), 1);
    }

    #[test]
    fn test_current_schema_does_not_create_another_migration_commit() {
        let mut conn = Connection::open_in_memory().expect("should open database");
        assert!(apply_graph_migrations(&mut conn).expect("should apply migrations"));

        assert!(!apply_graph_migrations(&mut conn).expect("schema should be current"));

        assert_eq!(migration_commit_count(&conn), 1);
    }

    #[test]
    fn test_graph_migrations_do_not_commit_preexisting_work() {
        let mut conn = Connection::open_in_memory().expect("should open database");
        conn.execute("CREATE TABLE user_work (id INTEGER PRIMARY KEY)", [])
            .expect("should create user table");

        let error = apply_graph_migrations(&mut conn).expect_err("dirty migration should fail");

        assert!(matches!(error, MigrationError::DirtyWorkingSet));
        assert_eq!(migration_commit_count(&conn), 0);
        let user_version: i64 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .expect("should read schema version");
        assert_eq!(user_version, 0);
    }
}
