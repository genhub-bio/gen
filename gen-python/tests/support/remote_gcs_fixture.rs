use std::{
    env,
    error::Error,
    io::{self, BufRead as _, Write as _},
    path::PathBuf,
};

#[path = "../../../tests/support/fake_gcs.rs"]
mod fake_gcs;

fn main() -> Result<(), Box<dyn Error>> {
    let local_database = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other("expected a local graph database path"))?;
    let server = fake_gcs::FakeGcsServer::start();
    let database = server.create_database(&local_database);
    println!("{}", database.uri());
    io::stdout().flush()?;

    let mut shutdown_request = String::new();
    io::stdin().lock().read_line(&mut shutdown_request)?;
    Ok(())
}
