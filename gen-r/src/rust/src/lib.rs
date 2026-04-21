use std::path::PathBuf;

use extendr_api::prelude::*;
use gen_core::config::Workspace;
use gen_models::{db::DbContext as GenDbContext, errors::OperationError, operations::Defaults};

fn nullable_string_to_option(value: Nullable<String>) -> Option<String> {
    match value {
        Nullable::NotNull(value) => Some(value),
        Nullable::Null => None,
    }
}

fn open_db_context(
    workspace_path: Option<String>,
    db_path: Option<String>,
) -> std::result::Result<(GenDbContext, String, String), String> {
    let workspace = match workspace_path {
        Some(path) => Workspace::new(path),
        None => Workspace::from_current_dir(),
    };

    let resolved_workspace_path = workspace.base_dir().to_string_lossy().into_owned();
    let gen_dir = workspace.ensure_gen_dir();
    let operations_path = gen_dir.join("gen.db");
    let operations_conn = r#gen::get_operation_connection(Some(operations_path))
        .map_err(|err| format!("Failed to open operations database: {err}"))?;

    let resolved_db_path = match db_path {
        Some(path) => path,
        None => {
            let mut stmt = operations_conn
                .prepare("select db_name from defaults where id = 1;")
                .map_err(|err| format!("Failed to load defaults: {err}"))?;
            let row: Option<String> = stmt.query_row([], |row| row.get(0)).ok();

            row.unwrap_or_else(|| gen_dir.join("default.db").to_string_lossy().into_owned())
        }
    };

    let graph_conn = r#gen::get_connection(PathBuf::from(&resolved_db_path))
        .map_err(|err| format!("Failed to open database '{resolved_db_path}': {err}"))?;

    Ok((
        GenDbContext::new(workspace, graph_conn, operations_conn),
        resolved_workspace_path,
        resolved_db_path,
    ))
}

fn resolve_collection_name(
    operations_conn: &gen_models::db::OperationsConnection,
    collection_name: Option<String>,
) -> std::result::Result<String, String> {
    match collection_name {
        Some(name) => Ok(name),
        None => Defaults::get(operations_conn)
            .map_err(|err| format!("Failed to load defaults: {err}"))?
            .collection_name
            .ok_or_else(|| "No default collection is set. Pass `name` explicitly.".to_string()),
    }
}

fn begin_transactions(context: &GenDbContext) -> std::result::Result<(), String> {
    let operations_conn = context.operations().conn();
    let graph_conn = context.graph().conn();

    r#gen::track_database(graph_conn, operations_conn)
        .map_err(|err| format!("Failed to track database: {err}"))?;
    graph_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|err| format!("Failed to begin graph transaction: {err}"))?;
    operations_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|err| format!("Failed to begin operations transaction: {err}"))?;
    Ok(())
}

fn end_transactions(context: &GenDbContext) -> std::result::Result<(), String> {
    let operations_conn = context.operations().conn();
    let graph_conn = context.graph().conn();

    graph_conn
        .execute("END TRANSACTION;", [])
        .map_err(|err| format!("Failed to commit graph transaction: {err}"))?;
    operations_conn
        .execute("END TRANSACTION;", [])
        .map_err(|err| format!("Failed to commit operations transaction: {err}"))?;
    Ok(())
}

fn rollback_transactions(context: &GenDbContext) {
    let operations_conn = context.operations().conn();
    let graph_conn = context.graph().conn();
    let _ = graph_conn.execute("ROLLBACK TRANSACTION;", []);
    let _ = operations_conn.execute("ROLLBACK TRANSACTION;", []);
}

/// Initialise a Gen workspace in the current directory.
/// @export
#[extendr]
fn init() -> std::result::Result<String, Error> {
    Workspace::from_current_dir().ensure_gen_dir();
    Ok("Gen repository initialized.".to_string())
}

/// Return the path to the current workspace's .gen directory.
/// @export
#[extendr]
fn get_gen_dir() -> std::result::Result<String, Error> {
    Ok(Workspace::from_current_dir()
        .ensure_gen_dir()
        .to_string_lossy()
        .into_owned())
}

/// Open a Gen database context.
/// @param workspace_path Optional path to the workspace root.
/// @param db_path Optional path to a specific SQLite database.
/// @export
#[extendr]
fn db_context(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
) -> std::result::Result<List, Error> {
    let (_db_context, resolved_workspace_path, resolved_db_path) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;

    Ok(list!(
        workspace_path = resolved_workspace_path,
        db_path = resolved_db_path
    ))
}

/// Import a FASTA file into a Gen collection.
/// @param workspace_path Optional path to the workspace root.
/// @param db_path Optional path to a specific SQLite database.
/// @param filename Path to the FASTA file to import.
/// @param name Collection name. Required if no default collection is set.
/// @param sample Sample name to import into.
/// @param shallow Whether to store sequence data by reference instead of inline.
/// @export
#[extendr]
fn import_fasta(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
    shallow: bool,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;

    begin_transactions(&context).map_err(Error::Other)?;

    match r#gen::imports::fasta::import_fasta(
        &context,
        &filename,
        &collection_name,
        &sample,
        shallow,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Fasta imported.".to_string())
        }
        Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
            rollback_transactions(&context);
            Err(Error::Other("Fasta contents already exist.".to_string()))
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Import failed: {err}")))
        }
    }
}

/// Export a Gen collection to FASTA.
/// @param workspace_path Optional path to the workspace root.
/// @param db_path Optional path to a specific SQLite database.
/// @param filename Output FASTA path.
/// @param name Collection name. Required if no default collection is set.
/// @param sample Optional sample name to export.
/// @export
#[extendr]
fn export_fasta(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: Nullable<String>,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;

    r#gen::track_database(context.graph().conn(), context.operations().conn())
        .map_err(|err| Error::Other(format!("Failed to track database: {err}")))?;

    r#gen::exports::fasta::export_fasta(
        context.graph().conn(),
        &collection_name,
        nullable_string_to_option(sample).as_deref(),
        &PathBuf::from(&filename),
    )
    .map_err(|err| Error::Other(format!("FASTA export failed: {err}")))?;

    Ok(filename)
}

extendr_module! {
    mod genr;
    fn init;
    fn get_gen_dir;
    fn db_context;
    fn import_fasta;
    fn export_fasta;
}
