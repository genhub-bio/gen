use clap::Args;

use crate::{
    commands::{cli_context::CliContext, get_db_for_command, get_default_collection},
    get_connection, get_operation_connection,
    updates::gfa::update_with_gfa,
};

/// Update with a GFA file
#[derive(Debug, Args)]
pub struct Command {
    /// GFA file path
    #[clap(index = 1)]
    pub path: String,
    /// The name of the collection to update
    #[arg(short, long)]
    name: Option<String>,
    /// The name of the sample to update
    #[arg(short, long)]
    sample: Option<String>,
    /// A new sample name to associate with the update
    #[arg(long)]
    new_sample: String,
}

pub fn execute(cli_context: &CliContext, cmd: Command) {
    println!("Update with GFA called");

    let operation_conn = get_operation_connection(None).unwrap();
    let db = get_db_for_command(cli_context.db.clone(), &operation_conn);
    let conn = get_connection(&db).unwrap();

    // initialize the selected database if needed.

    conn.execute("BEGIN TRANSACTION", []).unwrap();
    operation_conn.execute("BEGIN TRANSACTION", []).unwrap();

    let name = &cmd
        .name
        .clone()
        .unwrap_or_else(|| get_default_collection(&operation_conn));

    match update_with_gfa(
        &conn,
        &operation_conn,
        name,
        cmd.sample.clone().as_deref(),
        &cmd.new_sample,
        &cmd.path,
    ) {
        Ok(_) => {
            conn.execute("END TRANSACTION;", []).unwrap();
            operation_conn.execute("END TRANSACTION;", []).unwrap();
        }
        Err(e) => {
            conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            operation_conn.execute("ROLLBACK TRANSACTION;", []).unwrap();
            panic!("Failed to update. Error is: {e}");
        }
    }
}
