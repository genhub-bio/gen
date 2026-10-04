//! Disposable Dolt HTTP fixture for the Pyodide browser regression.

use core::error::Error;
use std::io::{self, BufRead as _};

use r#gen::get_connection;
use rusqlite::RemoteServer;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let database = get_connection(directory.path().join("fixture.db"))?;
    database.execute_batch(
        "CREATE TABLE browser_fixture(value TEXT NOT NULL);
         INSERT INTO browser_fixture VALUES ('cloned through DoltLite');",
    )?;
    database.query_row(
        "SELECT dolt_commit('-A', '-m', 'Seed disposable browser fixture')",
        [],
        |_| Ok(()),
    )?;
    drop(database);
    let server = RemoteServer::start(directory.path())?;
    println!("{}", server.database_url("fixture.db"));
    // The browser test owns stdin; EOF tears down the server and its temporary data.
    for line in io::stdin().lock().lines() {
        line?;
    }
    Ok(())
}
