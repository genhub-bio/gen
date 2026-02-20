use std::{
    fs::File,
    io,
    io::BufRead,
    path::{Path, PathBuf},
    str,
};
pub mod base16;
pub mod commands;
pub mod config;
pub mod diffs;
pub mod errors;
pub mod exports;
pub mod fasta;
pub mod genbank;
#[allow(clippy::all)]
pub mod generated;
pub use generated::gen_schema_capnp;
pub mod gfa;
pub mod gfa_reader;
pub mod graph_operators;
pub mod imports;

pub mod operation_management;
pub mod patch;
mod progress_bar;
#[cfg(any(test, debug_assertions))]
pub mod test_helpers;
pub mod updates;
pub mod views;

// reexports for public api, put behind features as needed
pub use gen_annotations as annotations;
pub use gen_core as core;
use gen_core::config::Workspace;
#[cfg(feature = "diff")]
pub use gen_diff as diff;
pub use gen_graph as graph;
#[cfg(feature = "models")]
pub use gen_models as models;
use gen_models::{
    db::{GraphConnection, OperationsConnection},
    files::GenDatabase,
    migrations::{run_migrations, run_operation_migrations},
};
use noodles::vcf::variant::record::samples::series::value::genotype::Phasing;
use rusqlite::{Connection, OptionalExtension};

pub fn get_connection(
    db_path: impl Into<PathBuf>,
) -> Result<GraphConnection, core::errors::ConnectionError> {
    let db_path = db_path.into();
    let mut conn = Connection::open(&db_path)?;
    rusqlite::vtab::array::load_module(&conn).unwrap();
    run_migrations(&mut conn);

    Ok(GraphConnection(conn))
}

pub fn track_database(
    conn: &GraphConnection,
    op_conn: &OperationsConnection,
) -> Result<(), core::errors::ConnectionError> {
    // This records the database in the main gen database file. It prevents things like a user copying a database to a new file
    // or overwriting databases without informing Gen about the changes. Not doing this would lead to situations like 2 databases
    // with the same uuid having changes being made in parallel but not having the same content.
    let db_uuid = models::metadata::Metadata::get_db_uuid(conn);
    if let Some(db_path) = conn.path() {
        if db_path.is_empty() {
            GenDatabase::create(op_conn, &db_uuid, "memory", ":memory:").map_err(|e| {
                core::errors::ConnectionError::DatabaseTracking(format!(
                    "Failed to create database tracking entry: {e}"
                ))
            })?;
        } else {
            let path = PathBuf::from(db_path);
            let mut rel_path = vec![];
            for component in path.ancestors() {
                let component_name = component.file_name().unwrap();
                if component.join(".gen").exists() {
                    break;
                }
                rel_path.push(component_name);
            }
            let relative_path = rel_path.iter().rev().collect::<PathBuf>();
            let relative_path_str = relative_path.to_str().unwrap();

            let exist_by_uuid = GenDatabase::get_by_uuid(op_conn, &db_uuid).optional()?;
            let exist_by_path = GenDatabase::get_by_path(op_conn, relative_path_str).optional()?;
            if let Some(path_db) = exist_by_path {
                if let Some(uuid_db) = exist_by_uuid {
                    if path_db == uuid_db {
                        return Ok(());
                    }
                } else {
                    return Err(core::errors::ConnectionError::DatabaseTracking(format!(
                        "Database conflict: Database '{}' (UUID: {}) is registered at path '{}', which does not match the database found at {}",
                        path_db.name, path_db.db_uuid, path_db.path, db_path
                    )));
                }
            } else if let Some(uuid_db) = exist_by_uuid {
                return Err(core::errors::ConnectionError::DatabaseTracking(format!(
                    "Database conflict: Database '{}' (UUID: {}) is registered at path '{}', but was accessed from {}. Was this file moved?",
                    uuid_db.name, uuid_db.db_uuid, uuid_db.path, db_path
                )));
            }

            let db_name = relative_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            GenDatabase::create(op_conn, &db_uuid, &db_name, relative_path_str).map_err(|e| {
                core::errors::ConnectionError::DatabaseTracking(format!(
                    "Failed to create database tracking entry: {e}"
                ))
            })?;
        }
    }
    Ok(())
}

pub fn get_operation_connection(
    db_path: impl Into<Option<PathBuf>>,
) -> Result<OperationsConnection, core::errors::ConnectionError> {
    let db_path = db_path.into();
    let path = if let Some(s) = db_path {
        s
    } else {
        Workspace::from_current_dir().gen_db_path()?
    };
    let mut conn = Connection::open(&path)?;
    rusqlite::vtab::array::load_module(&conn).unwrap();
    run_operation_migrations(&mut conn);
    Ok(OperationsConnection(conn))
}

pub fn run_query(conn: &Connection, query: &str) {
    let mut stmt = conn.prepare(query).unwrap();
    for entry in stmt.query_map([], |_| Ok(())).unwrap() {
        println!("{entry:?}");
    }
}

pub struct Genotype {
    pub allele: i64,
    pub phasing: Phasing,
}

pub fn parse_genotype(gt: &str) -> Vec<Option<Genotype>> {
    let mut genotypes = vec![];
    let mut phase = match gt.contains('/') {
        true => Phasing::Unphased,
        false => Phasing::Phased,
    };
    for entry in gt.split_inclusive(['|', '/']) {
        let allele;
        let mut phasing = Phasing::Unphased;
        if entry.ends_with(['/', '|']) {
            let (allele_str, phasing_str) = entry.split_at(entry.len() - 1);
            allele = allele_str;
            phasing = match phasing_str == "|" {
                true => Phasing::Phased,
                false => Phasing::Unphased,
            }
        } else {
            allele = entry;
        }
        if allele == "." {
            genotypes.push(None);
        } else {
            genotypes.push(Some(Genotype {
                allele: allele.parse::<i64>().unwrap(),
                phasing: phase,
            }));
        }
        // we're always 1 behind on phase, e.g. 0|1, the | is the phase of the next allele
        phase = phasing;
    }
    genotypes
}

pub fn get_overlap(a: i64, b: i64, x: i64, y: i64) -> (bool, bool, bool) {
    let contains_start = a <= x && x < b;
    let contains_end = a <= y && y < b;
    let overlap = a < y && x < b;
    (contains_start, contains_end, overlap)
}

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn normalize_string(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use gen_models::{db::DbContext, files::GenDatabase, metadata::get_db_uuid, traits::Query};

    use super::*;
    use crate::test_helpers::{get_connection, get_operation_connection, setup_gen};

    #[cfg(test)]
    mod test_normalize_string {
        use super::*;

        #[test]
        fn test_removes_whitespace() {
            assert_eq!(normalize_string(" this has a space "), "thishasaspace")
        }

        #[test]
        fn test_removes_newlines() {
            assert_eq!(
                normalize_string("\nthis\nhas\n\nnew\nlines"),
                "thishasnewlines"
            )
        }
    }

    #[test]
    fn it_queries() {
        let conn = get_connection(None).unwrap();
        let sequence_count: i64 = conn
            .query_row(
                "SELECT count(*) from sequences where hash = 'foo'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(sequence_count, 0);
    }

    #[test]
    fn parses_genotype() {
        let genotypes = parse_genotype("1");
        let genotype_1 = genotypes[0].as_ref().unwrap();
        assert_eq!(genotype_1.allele, 1);
        assert_eq!(genotype_1.phasing, Phasing::Phased);
        let genotypes = parse_genotype("0|1");
        let genotype_1 = genotypes[0].as_ref().unwrap();
        let genotype_2 = genotypes[1].as_ref().unwrap();
        assert_eq!(genotype_1.allele, 0);
        assert_eq!(genotype_1.phasing, Phasing::Phased);
        assert_eq!(genotype_2.allele, 1);
        assert_eq!(genotype_2.phasing, Phasing::Phased);
        let genotypes = parse_genotype("0/1");
        let genotype_1 = genotypes[0].as_ref().unwrap();
        let genotype_2 = genotypes[1].as_ref().unwrap();
        assert_eq!(genotype_1.allele, 0);
        assert_eq!(genotype_1.phasing, Phasing::Unphased);
        assert_eq!(genotype_2.allele, 1);
        assert_eq!(genotype_2.phasing, Phasing::Unphased);
        let genotypes = parse_genotype("0/1|2");
        let genotype_1 = genotypes[0].as_ref().unwrap();
        let genotype_2 = genotypes[1].as_ref().unwrap();
        let genotype_3 = genotypes[2].as_ref().unwrap();
        assert_eq!(genotype_1.allele, 0);
        assert_eq!(genotype_1.phasing, Phasing::Unphased);
        assert_eq!(genotype_2.allele, 1);
        assert_eq!(genotype_2.phasing, Phasing::Unphased);
        assert_eq!(genotype_3.allele, 2);
        assert_eq!(genotype_3.phasing, Phasing::Phased);
        let genotypes = parse_genotype("2|1|2");
        let genotype_1 = genotypes[0].as_ref().unwrap();
        let genotype_2 = genotypes[1].as_ref().unwrap();
        let genotype_3 = genotypes[2].as_ref().unwrap();
        assert_eq!(genotype_1.allele, 2);
        assert_eq!(genotype_1.phasing, Phasing::Phased);
        assert_eq!(genotype_2.allele, 1);
        assert_eq!(genotype_2.phasing, Phasing::Phased);
        assert_eq!(genotype_3.allele, 2);
        assert_eq!(genotype_3.phasing, Phasing::Phased);
        let genotypes = parse_genotype("2|.|2");
        let genotype_1 = genotypes[0].as_ref().unwrap();
        let genotype_3 = genotypes[2].as_ref().unwrap();
        assert_eq!(genotype_1.allele, 2);
        assert_eq!(genotype_1.phasing, Phasing::Phased);
        assert_eq!(genotype_3.allele, 2);
        assert_eq!(genotype_3.phasing, Phasing::Phased);
        assert!(genotypes[1].is_none());
    }

    #[test]
    fn test_overlaps() {
        assert_eq!(get_overlap(0, 10, 10, 10), (false, false, false));
        assert_eq!(get_overlap(10, 20, 10, 20), (true, false, true));
        assert_eq!(get_overlap(10, 20, 5, 15), (false, true, true));
        assert_eq!(get_overlap(10, 20, 0, 10), (false, true, false));
    }

    #[test]
    fn test_database_tracking_integration() {
        // Assert that connecting to the database triggers database tracking
        let context = setup_gen();
        let db_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("test_tracking.db");
        let context = DbContext::new(
            context.workspace().clone(),
            get_connection(db_path.to_str()).unwrap(),
            get_operation_connection(None).unwrap(),
        );
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        let db_uuid: String = models::metadata::get_db_uuid(conn);

        let tracked_db = GenDatabase::get_by_uuid(op_conn, &db_uuid).unwrap();

        assert_eq!(tracked_db.db_uuid, db_uuid);
        assert_eq!(tracked_db.name, "test_tracking");
        assert_eq!(tracked_db.path, "test_tracking.db");

        // Connect again - should not create duplicate entry
        let _conn2 = crate::get_connection(db_path).unwrap();

        let all_entries: Vec<GenDatabase> = GenDatabase::query(
            op_conn,
            "SELECT * FROM gen_databases WHERE db_uuid = ?1",
            rusqlite::params![db_uuid],
        );

        assert_eq!(all_entries.len(), 1);
    }

    #[test]
    fn test_path_conflict_detection_different_path() {
        let context = setup_gen();
        let db_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("original_location.db");
        let context = DbContext::new(
            context.workspace().clone(),
            crate::get_connection(&db_path).unwrap(),
            get_operation_connection(None).unwrap(),
        );
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        // Force write of the WAL file into the db so we can copy
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();

        // Move database to different location within .gen directory
        let moved_db_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("moved_location.db");
        fs::copy(&db_path, &moved_db_path).unwrap();

        // Try to access from moved location - should detect path conflict
        let conn = &crate::get_connection(&moved_db_path).unwrap();
        let result = track_database(conn, op_conn);

        assert!(result.is_err());
        match result {
            Err(core::errors::ConnectionError::DatabaseTracking(_)) => {
                // Expected error type
            }
            _ => panic!("Expected DatabaseTracking error, got: {result:?}"),
        }
    }

    #[test]
    fn test_path_conflict_detection_uuid_mismatch() {
        let context = setup_gen();
        let db_path = context
            .workspace()
            .repo_root()
            .unwrap()
            .join("original_location.db");
        let context = DbContext::new(
            context.workspace().clone(),
            crate::get_connection(db_path.clone()).unwrap(),
            get_operation_connection(None).unwrap(),
        );
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();
        track_database(conn, op_conn).unwrap();

        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();
        let db_uuid1 = get_db_uuid(conn);

        let new_db_path = context.workspace().repo_root().unwrap().join("new.db");
        let conn = &crate::get_connection(new_db_path.clone()).unwrap();
        track_database(conn, op_conn).unwrap();
        conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")
            .unwrap();

        let db_uuid2 = get_db_uuid(conn);

        assert_ne!(db_uuid1, db_uuid2);

        // Replace
        fs::copy(&new_db_path, &db_path).unwrap();
        let conn = &crate::get_connection(db_path.clone()).unwrap();
        let result = track_database(conn, op_conn);

        assert!(result.is_err());
        match result {
            Err(core::errors::ConnectionError::DatabaseTracking(_)) => {
                // Expected error type
            }
            _ => panic!("Expected DatabaseTracking error, got: {result:?}"),
        }
    }
}
