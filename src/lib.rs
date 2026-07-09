use std::{
    fs::File,
    io,
    io::BufRead,
    path::{Path, PathBuf},
    str,
};
pub mod commands;
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
pub mod graphs;
pub mod imports;

pub mod patch;
#[cfg(feature = "profiling")]
pub mod profiling;
mod progress_bar;
#[cfg(any(test, debug_assertions))]
pub mod test_helpers;
pub mod theme;
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
    db::{ConfigConnection, GraphConnection},
    migrations::{run_migrations, run_operation_migrations},
};
use noodles::vcf::variant::record::samples::series::value::genotype::Phasing;
use rusqlite::{Connection, OpenFlags};

pub fn get_connection(
    db_path: impl Into<PathBuf>,
) -> Result<GraphConnection, core::errors::ConnectionError> {
    let db_path = db_path.into();
    let mut conn = Connection::open(&db_path)?;
    rusqlite::vtab::array::load_module(&conn).unwrap();
    run_migrations(&mut conn);
    Ok(GraphConnection(conn))
}

pub fn get_config_connection(
    db_path: impl Into<Option<PathBuf>>,
) -> Result<ConfigConnection, core::errors::ConnectionError> {
    let db_path = db_path.into();
    let path = if let Some(s) = db_path {
        s
    } else {
        Workspace::from_current_dir().gen_db_path()?
    };
    let mut conn = Connection::open(&path)?;
    rusqlite::vtab::array::load_module(&conn).unwrap();
    run_operation_migrations(&mut conn);
    Ok(ConfigConnection(conn))
}

pub fn get_history_connection(
    db_path: impl AsRef<Path>,
) -> Result<GraphConnection, core::errors::ConnectionError> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    Ok(GraphConnection(conn))
}

pub fn get_operation_connection(
    db_path: impl Into<Option<PathBuf>>,
) -> Result<ConfigConnection, core::errors::ConnectionError> {
    get_config_connection(db_path)
}

pub fn end_transaction_if_active(conn: &Connection) -> Result<(), rusqlite::Error> {
    if !conn.is_autocommit() {
        conn.execute("END TRANSACTION", [])?;
    }
    Ok(())
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
    use gen_models::{
        collection::Collection,
        history::{HistoryStore, dolt::DoltHistoryStore},
    };

    use super::*;
    use crate::test_helpers::{get_connection, get_operation_connection};

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
    fn test_reopening_graph_connection_on_disk_reuses_existing_schema() {
        let temp_dir = tempfile::tempdir().expect("should create temp directory");
        let workspace = Workspace::new(temp_dir.path());
        workspace.ensure_gen_dir();
        let graph_db_path = workspace
            .graph_db_path()
            .expect("should resolve graph db path");

        drop(
            get_connection(graph_db_path.to_str().expect("should encode graph db path"))
                .expect("should create graph database"),
        );

        let reopened = get_connection(graph_db_path.to_str().expect("should encode graph db path"));
        assert!(
            reopened.is_ok(),
            "reopening an initialized graph database should succeed: {reopened:?}"
        );
    }

    #[test]
    fn test_reopening_operation_connection_on_disk_reuses_existing_schema() {
        let temp_dir = tempfile::tempdir().expect("should create temp directory");
        let workspace = Workspace::new(temp_dir.path());
        workspace.ensure_gen_dir();
        let gen_db_path = workspace
            .gen_db_path()
            .expect("should resolve config db path");

        drop(
            get_operation_connection(gen_db_path.to_str().expect("should encode config db path"))
                .expect("should create config database"),
        );

        let reopened =
            get_operation_connection(gen_db_path.to_str().expect("should encode config db path"));
        assert!(
            reopened.is_ok(),
            "reopening an initialized config database should succeed: {reopened:?}"
        );
    }

    #[test]
    fn test_history_connection_reads_history_and_rejects_writes() {
        let temp_dir = tempfile::tempdir().expect("should create temp directory");
        let workspace = Workspace::new(temp_dir.path());
        workspace.ensure_gen_dir();
        let graph_db_path = workspace
            .graph_db_path()
            .expect("should resolve graph db path");
        let graph_db_path_str = graph_db_path.to_str().expect("should encode graph db path");
        let graph_connection =
            get_connection(graph_db_path_str).expect("should create graph database");
        Collection::get_or_create(&graph_connection, "test-collection")
            .expect("should insert collection row");
        gen_models::history::dolt::commit_all(&graph_connection, "initial collection commit")
            .expect("should commit collection row");
        drop(graph_connection);

        let history_connection =
            get_history_connection(&graph_db_path).expect("should reopen history connection");
        let history_store = DoltHistoryStore::new(&history_connection);
        let history_entries = history_store.log(None).expect("should read history");
        assert_eq!(history_entries.len(), 2);
        assert_eq!(history_entries[0].message, "initial collection commit");

        let write_result = history_connection.execute("CREATE TABLE should_fail (id INTEGER)", []);
        assert!(
            write_result.is_err(),
            "history connection should reject writes"
        );
    }
}
