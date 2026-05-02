use std::{
    fs::File,
    io::{BufWriter, Read},
    path::{Path, PathBuf},
};

use extendr_api::prelude::*;
use flate2::read::GzDecoder;
use r#gen::{
    get_connection,
    graphs::combinatorial_library::{SequencePart, parse_library},
    views::gen_graph_widget::{GenGraphNodeRenderer, GenGraphNodeSizer},
};
use gen_core::{HashId, config::Workspace};
use gen_graph::GenGraph;
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup},
    db::{DbContext as GenDbContext, GraphConnection},
    errors::OperationError,
    node::Node,
    operations::{Defaults, OperationFile, OperationInfo},
    traits::Query,
};
use gen_tui::{graph_controller::GraphController, graph_widget::GraphWidget, layout::VisualDetail};
use ratatui::{buffer::Buffer, layout::Rect, style::Modifier, widgets::StatefulWidget};
use rusqlite::{Connection, types::ValueRef};
use serde::Serialize;

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
            .ok_or_else(|| "No defaults row is set. Pass `name` explicitly.".to_string())?
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

fn open_repo_gen_dir(path: Option<String>) -> PathBuf {
    match path {
        Some(path_str) => PathBuf::from(path_str),
        None => Workspace::from_current_dir().ensure_gen_dir(),
    }
}

fn open_repo_connection(db_path: &str) -> std::result::Result<GraphConnection, String> {
    get_connection(db_path).map_err(|err| format!("Failed to open database '{db_path}': {err}"))
}

fn hash_id_from_string(value: &str) -> std::result::Result<HashId, String> {
    HashId::try_from(value.to_string()).map_err(|err| format!("Invalid hash id '{value}': {err}"))
}

fn block_group_record(block_group: BlockGroup, db_path: Option<&str>) -> List {
    list!(
        id = block_group.id.to_string(),
        collection_name = block_group.collection_name,
        sample_name = block_group.sample_name,
        name = block_group.name,
        db_path = db_path.map(|path| path.to_string())
    )
}

fn node_key_record(node_id: HashId, sequence_start: i64, sequence_end: i64) -> List {
    list!(
        node_id = node_id.to_string(),
        sequence_start = sequence_start,
        sequence_end = sequence_end
    )
}

fn sqlite_value_to_robj(value: ValueRef<'_>) -> Robj {
    match value {
        ValueRef::Null => NULL.into(),
        ValueRef::Integer(i) => r!(i),
        ValueRef::Real(f) => r!(f),
        ValueRef::Text(s) => r!(String::from_utf8_lossy(s).to_string()),
        ValueRef::Blob(b) => r!(b.to_vec()),
    }
}

fn query_rows(conn: &Connection, query: &str) -> std::result::Result<List, String> {
    let mut stmt = conn
        .prepare(query)
        .map_err(|err| format!("Failed to prepare query: {err}"))?;
    let column_count = stmt.column_count();
    let mut row_iter = stmt
        .query([])
        .map_err(|err| format!("Failed to execute query: {err}"))?;
    let mut rows = vec![];

    while let Some(row) = row_iter
        .next()
        .map_err(|err| format!("Failed to fetch row: {err}"))?
    {
        let mut values = vec![];
        for index in 0..column_count {
            values.push(
                row.get_ref(index)
                    .map(sqlite_value_to_robj)
                    .map_err(|err| format!("Failed to read row value: {err}"))?,
            );
        }
        rows.push(List::from_values(values));
    }

    Ok(List::from_values(rows))
}

fn parse_sequence_part(part: &Robj) -> std::result::Result<SequencePart, String> {
    if !part.is_list() {
        return Err("Each sequence part must be a list.".to_string());
    }

    let name = part
        .dollar("name")
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .or_else(|| {
            part.index(1)
                .ok()
                .and_then(|value| value.as_str().map(ToOwned::to_owned))
        })
        .ok_or_else(|| "Sequence part is missing a 'name' field.".to_string())?;
    let sequence = part
        .dollar("sequence")
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .or_else(|| {
            part.index(2)
                .ok()
                .and_then(|value| value.as_str().map(ToOwned::to_owned))
        })
        .ok_or_else(|| "Sequence part is missing a 'sequence' field.".to_string())?;

    Ok(SequencePart {
        name,
        sequence_length: sequence.len() as i64,
        sequence,
    })
}

fn parse_parts_list(parts_list: Robj) -> std::result::Result<Vec<Vec<SequencePart>>, String> {
    let outer = parts_list
        .as_list()
        .ok_or_else(|| "parts_list must be a list of columns.".to_string())?;
    outer
        .values()
        .map(|column| {
            let column_list = column
                .as_list()
                .ok_or_else(|| "Each library column must be a list of parts.".to_string())?;
            column_list
                .values()
                .map(|part| parse_sequence_part(&part))
                .collect()
        })
        .collect()
}

fn read_genbank_reader(filename: &str) -> std::result::Result<Box<dyn Read>, String> {
    let file = File::open(filename)
        .map_err(|err| format!("Failed to open GenBank file '{filename}': {err}"))?;
    if filename.ends_with(".gz") {
        Ok(Box::new(GzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

fn color_to_hex(color: Option<ratatui::style::Color>, default_hex: &str) -> String {
    use ratatui::style::Color;
    match color {
        None | Some(Color::Reset) => default_hex.to_string(),
        Some(Color::Rgb(r, g, b)) => format!("#{r:02x}{g:02x}{b:02x}"),
        Some(Color::Black) => "#000000".to_string(),
        Some(Color::Red) => "#cc0000".to_string(),
        Some(Color::Green) => "#00cc00".to_string(),
        Some(Color::Yellow) => "#cccc00".to_string(),
        Some(Color::Blue) => "#0000cc".to_string(),
        Some(Color::Magenta) => "#cc00cc".to_string(),
        Some(Color::Cyan) => "#00cccc".to_string(),
        Some(Color::Gray) => "#888888".to_string(),
        Some(Color::DarkGray) => "#444444".to_string(),
        Some(Color::LightRed) => "#ff5555".to_string(),
        Some(Color::LightGreen) => "#55ff55".to_string(),
        Some(Color::LightYellow) => "#ffff55".to_string(),
        Some(Color::LightBlue) => "#5555ff".to_string(),
        Some(Color::LightMagenta) => "#ff55ff".to_string(),
        Some(Color::LightCyan) => "#55ffff".to_string(),
        Some(Color::White) => "#ffffff".to_string(),
        Some(Color::Indexed(i)) => indexed_to_hex(i),
    }
}

fn indexed_to_hex(i: u8) -> String {
    const STANDARD: [&str; 16] = [
        "#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#c0c0c0",
        "#808080", "#ff0000", "#00ff00", "#ffff00", "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
    ];
    if (i as usize) < STANDARD.len() {
        return STANDARD[i as usize].to_string();
    }
    if i >= 232 {
        let v = 8u8.saturating_add((i - 232) * 10);
        return format!("#{v:02x}{v:02x}{v:02x}");
    }
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let n = i - 16;
    let r = LEVELS[(n / 36) as usize];
    let g = LEVELS[((n / 6) % 6) as usize];
    let b = LEVELS[(n % 6) as usize];
    format!("#{r:02x}{g:02x}{b:02x}")
}

#[derive(Serialize)]
struct RenderedCell {
    text: String,
    fg: String,
    bg: String,
    bold: bool,
    italic: bool,
    underline: bool,
}

#[derive(Serialize)]
struct RenderedFrame {
    cols: u16,
    rows: u16,
    cells: Vec<RenderedCell>,
}

fn serialize_buffer(buf: &Buffer, cols: u16, rows: u16) -> RenderedFrame {
    let mut cells = Vec::with_capacity(cols as usize * rows as usize);
    for row in 0..rows {
        for col in 0..cols {
            let cell = buf.cell((col, row)).expect("cell index in bounds");
            let style = cell.style();
            cells.push(RenderedCell {
                text: cell.symbol().to_string(),
                fg: color_to_hex(style.fg, "#cdd6f4"),
                bg: color_to_hex(style.bg, "#1e1e2e"),
                bold: style.add_modifier.contains(Modifier::BOLD),
                italic: style.add_modifier.contains(Modifier::ITALIC),
                underline: style.add_modifier.contains(Modifier::UNDERLINED),
            });
        }
    }
    RenderedFrame { cols, rows, cells }
}

fn visual_detail(detail: &str) -> std::result::Result<VisualDetail, String> {
    match detail {
        "normal" => Ok(VisualDetail::Truncated),
        "full" => Ok(VisualDetail::Full),
        "minimal" => Ok(VisualDetail::Minimal),
        other => Err(format!(
            "detail must be \"normal\", \"full\", or \"minimal\"; got {other:?}"
        )),
    }
}

fn block_group_graph(
    db_path: &str,
    block_group_id: &str,
) -> std::result::Result<(GraphConnection, GenGraph), String> {
    let conn = open_repo_connection(db_path)?;
    let bg_id = hash_id_from_string(block_group_id)?;
    let graph = BlockGroup::get_graph(&conn, &bg_id);
    Ok((conn, graph))
}

fn apply_graph_ops(
    controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    ops: &str,
) -> std::result::Result<(), String> {
    for op in ops.split(';').filter(|segment| !segment.is_empty()) {
        let parts = op.split(',').collect::<Vec<_>>();
        match parts.first().copied() {
            Some("zi") => controller.zoom_in(),
            Some("zo") => controller.zoom_out(),
            Some("m") => {
                if parts.len() != 3 {
                    return Err(format!("Invalid move op: {op}"));
                }
                let dx = parts[1]
                    .parse::<i16>()
                    .map_err(|err| format!("Invalid move dx in '{op}': {err}"))?;
                let dy = parts[2]
                    .parse::<i16>()
                    .map_err(|err| format!("Invalid move dy in '{op}': {err}"))?;
                controller.move_by_terminal(dx, dy);
                controller.sync_cursor_to_closest_node();
            }
            Some("c") => {
                if parts.len() != 3 {
                    return Err(format!("Invalid click op: {op}"));
                }
                let col = parts[1]
                    .parse::<u16>()
                    .map_err(|err| format!("Invalid click col in '{op}': {err}"))?;
                let row = parts[2]
                    .parse::<u16>()
                    .map_err(|err| format!("Invalid click row in '{op}': {err}"))?;
                let _ = controller.handle_click(col, row);
            }
            Some(other) => return Err(format!("Unknown graph op prefix '{other}'.")),
            None => {}
        }
    }
    Ok(())
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

#[extendr]
fn import_gfa(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
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
    match r#gen::imports::gfa::import_gfa(&context, Path::new(&filename), &collection_name, &sample)
    {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("GFA imported.".to_string())
        }
        Err(r#gen::imports::gfa::GFAImportError::OperationError(OperationError::NoChanges)) => {
            rollback_transactions(&context);
            Err(Error::Other("GFA already exists.".to_string()))
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Import failed: {err}")))
        }
    }
}

#[extendr]
fn import_genbank(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name_opt = nullable_string_to_option(name);
    let mut reader = read_genbank_reader(&filename).map_err(Error::Other)?;

    begin_transactions(&context).map_err(Error::Other)?;

    match r#gen::imports::genbank::import_genbank(
        &context,
        &mut reader,
        collection_name_opt.as_deref(),
        &sample,
        OperationInfo {
            files: vec![OperationFile {
                file_path: filename.clone(),
                file_type: gen_models::file_types::FileTypes::GenBank,
            }],
            description: "GenBank Import".to_string(),
        },
        r#gen::imports::genbank::GenBankImportOptions::default()
            .annotation_name_from_path(&filename),
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("GenBank imported.".to_string())
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Import failed: {err}")))
        }
    }
}

#[extendr]
fn import_library_files(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    library_name: String,
    parts: String,
    library: String,
    name: Nullable<String>,
    sample: String,
) -> std::result::Result<String, Error> {
    let parts_list = parse_library(&parts, &library)
        .map_err(|err| Error::Other(format!("Problem parsing library files: {err}")))?;
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;

    begin_transactions(&context).map_err(Error::Other)?;
    match r#gen::imports::library::import_library(
        &context,
        &collection_name,
        &sample,
        &library_name,
        parts_list,
        Some(&parts),
        Some(&library),
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Library imported.".to_string())
        }
        Err(r#gen::imports::library::LibraryImportError::OperationError(
            OperationError::NoChanges,
        )) => {
            rollback_transactions(&context);
            Err(Error::Other("Library already exists.".to_string()))
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Library import failed: {err}")))
        }
    }
}

#[extendr]
fn import_library(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    library_name: String,
    parts_list: Robj,
    name: Nullable<String>,
    sample: Nullable<String>,
) -> std::result::Result<String, Error> {
    let rust_parts_list = parse_parts_list(parts_list).map_err(Error::Other)?;
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;
    let sample_name = nullable_string_to_option(sample)
        .unwrap_or_else(|| gen_models::sample::Sample::DEFAULT_NAME.to_string());

    begin_transactions(&context).map_err(Error::Other)?;
    match r#gen::imports::library::import_library(
        &context,
        &collection_name,
        &sample_name,
        &library_name,
        rust_parts_list,
        None,
        None,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Library imported.".to_string())
        }
        Err(r#gen::imports::library::LibraryImportError::OperationError(
            OperationError::NoChanges,
        )) => {
            rollback_transactions(&context);
            Err(Error::Other("Library already exists.".to_string()))
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Library import failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_fasta(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
    new_sample: String,
    region_name: String,
    start: i64,
    end: i64,
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
    match r#gen::updates::fasta::update_with_fasta(
        &context,
        &collection_name,
        &sample,
        &new_sample,
        &region_name,
        start,
        end,
        &filename,
        false,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with fasta.".to_string())
        }
        Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
            rollback_transactions(&context);
            Err(Error::Other("Fasta contents already exist.".to_string()))
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_gfa(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
    new_sample: String,
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
    match r#gen::updates::gfa::update_with_gfa(
        &context,
        &collection_name,
        &sample,
        &new_sample,
        &filename,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with GFA.".to_string())
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_gaf(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    csv: String,
    name: Nullable<String>,
    sample: String,
    parent_sample: Nullable<String>,
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
    match r#gen::updates::gaf::update_with_gaf(
        &context,
        &filename,
        &csv,
        &collection_name,
        &sample,
        nullable_string_to_option(parent_sample).as_deref(),
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with GAF.".to_string())
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_vcf(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    genotype: Nullable<String>,
    sample: Nullable<String>,
    parent_samples: Vec<String>,
    in_place: bool,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;
    let genotype = nullable_string_to_option(genotype).unwrap_or_default();
    let sample = nullable_string_to_option(sample);

    begin_transactions(&context).map_err(Error::Other)?;
    match r#gen::updates::vcf::update_with_vcf(
        &context,
        &filename,
        &collection_name,
        genotype,
        sample.as_deref(),
        parent_samples,
        in_place,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with VCF.".to_string())
        }
        Err(r#gen::updates::vcf::VcfError::OperationError(OperationError::NoChanges)) => {
            rollback_transactions(&context);
            Err(Error::Other(
                "No changes made. Provide sample and genotype if missing from VCF.".to_string(),
            ))
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_genbank(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
    create_missing: bool,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name_opt = nullable_string_to_option(name);
    let mut reader = read_genbank_reader(&filename).map_err(Error::Other)?;

    begin_transactions(&context).map_err(Error::Other)?;
    match r#gen::updates::genbank::update_with_genbank(
        &context,
        &mut reader,
        collection_name_opt.as_deref(),
        &sample,
        create_missing,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: filename.clone(),
                file_type: gen_models::file_types::FileTypes::GenBank,
            }],
            description: "Update from GenBank".to_string(),
        },
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with GenBank.".to_string())
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_library_files(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    name: Nullable<String>,
    sample: String,
    new_sample: String,
    path_name: String,
    start: i64,
    end: i64,
    library: String,
    parts: String,
) -> std::result::Result<String, Error> {
    let parts_list = parse_library(&parts, &library)
        .map_err(|err| Error::Other(format!("Couldn't parse library files: {err}")))?;
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;

    begin_transactions(&context).map_err(Error::Other)?;
    match r#gen::updates::library::update_with_library(
        &context,
        &collection_name,
        &sample,
        &new_sample,
        &path_name,
        start,
        end,
        parts_list,
        Some(&parts),
        Some(&library),
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with library.".to_string())
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_library(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    name: Nullable<String>,
    sample: Nullable<String>,
    new_sample_name: String,
    path_name: String,
    start: i64,
    end: i64,
    parts_list: Robj,
) -> std::result::Result<String, Error> {
    let rust_parts_list = parse_parts_list(parts_list).map_err(Error::Other)?;
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;
    let sample_name = nullable_string_to_option(sample)
        .unwrap_or_else(|| gen_models::sample::Sample::DEFAULT_NAME.to_string());

    begin_transactions(&context).map_err(Error::Other)?;
    match r#gen::updates::library::update_with_library(
        &context,
        &collection_name,
        &sample_name,
        &new_sample_name,
        &path_name,
        start,
        end,
        rust_parts_list,
        None,
        None,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with library.".to_string())
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

#[extendr]
fn update_with_sequence(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    sequence: String,
    name: Nullable<String>,
    sample: String,
    new_sample: String,
    region_name: String,
    start: i64,
    end: i64,
    no_reference_path_update: bool,
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
    match r#gen::updates::sequence::update_with_sequence(
        &context,
        &collection_name,
        &sample,
        &new_sample,
        &region_name,
        start,
        end,
        &sequence,
        no_reference_path_update,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Updated with sequence.".to_string())
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Update failed: {err}")))
        }
    }
}

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

#[extendr]
fn export_gfa(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
    node_max: Nullable<i64>,
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

    r#gen::exports::gfa::export_gfa(
        context.graph().conn(),
        &collection_name,
        &PathBuf::from(&filename),
        &sample,
        match node_max {
            Nullable::NotNull(value) => Some(value),
            Nullable::Null => None,
        },
    )
    .map_err(|err| Error::Other(format!("GFA export failed: {err}")))?;

    Ok(filename)
}

#[extendr]
fn export_genbank(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    filename: String,
    name: Nullable<String>,
    sample: String,
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

    let writer = BufWriter::new(File::create(&filename).map_err(|err| {
        Error::Other(format!("Failed to create GenBank file '{filename}': {err}"))
    })?);

    r#gen::exports::genbank::export_genbank(
        context.graph().conn(),
        &collection_name,
        &sample,
        writer,
    )
    .map_err(|err| Error::Other(format!("GenBank export failed: {err}")))?;

    Ok(filename)
}

#[extendr]
fn derive_chunks(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    name: Nullable<String>,
    sample: String,
    new_sample: String,
    region: String,
    backbone: Nullable<String>,
    breakpoints: Nullable<String>,
    chunk_size: Nullable<i64>,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;

    r#gen::commands::graph_operations::derive_chunks::derive_chunks_operation(
        &context,
        nullable_string_to_option(name),
        sample,
        new_sample,
        region,
        nullable_string_to_option(backbone),
        nullable_string_to_option(breakpoints),
        match chunk_size {
            Nullable::NotNull(value) => Some(value),
            Nullable::Null => None,
        },
    )
    .map_err(|err| Error::Other(format!("Error deriving chunks: {err}")))?;

    Ok("Derived chunks.".to_string())
}

#[extendr]
fn derive_subgraph(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    name: Nullable<String>,
    sample: String,
    new_sample: String,
    region: String,
    backbone: Nullable<String>,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;

    r#gen::commands::graph_operations::derive_subgraph::derive_subgraph_operation(
        &context,
        nullable_string_to_option(name),
        sample,
        new_sample,
        region,
        nullable_string_to_option(backbone),
    )
    .map_err(|err| Error::Other(format!("Error deriving subgraph: {err}")))?;

    Ok("Derived subgraph.".to_string())
}

#[extendr]
fn make_stitch(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    name: Nullable<String>,
    sample: String,
    new_sample: String,
    regions: String,
    new_region: String,
) -> std::result::Result<String, Error> {
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;

    r#gen::commands::graph_operations::make_stitch::make_stitch_operation(
        &context,
        nullable_string_to_option(name),
        sample,
        new_sample,
        regions,
        new_region,
    )
    .map_err(|err| Error::Other(format!("Error making stitch: {err}")))?;

    Ok("Made stitch.".to_string())
}

#[extendr]
fn repo_get_gen_dir(path: Nullable<String>) -> String {
    open_repo_gen_dir(nullable_string_to_option(path))
        .to_string_lossy()
        .into_owned()
}

#[extendr]
fn repo_get_db_path(path: Nullable<String>) -> String {
    open_repo_gen_dir(nullable_string_to_option(path))
        .join("default.db")
        .to_string_lossy()
        .into_owned()
}

#[extendr]
fn repo_execute(db_path: String, query: String) -> std::result::Result<(), Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    conn.execute(&query, [])
        .map_err(|err| Error::Other(format!("SQLite error: {err}")))?;
    Ok(())
}

#[extendr]
fn repo_query(db_path: String, query: String) -> std::result::Result<List, Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    query_rows(&conn.0, &query).map_err(Error::Other)
}

#[extendr]
fn repo_get_block_group_by_id(db_path: String, id: String) -> std::result::Result<List, Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let block_group =
        BlockGroup::get_by_id(&conn, &hash_id_from_string(&id).map_err(Error::Other)?)
            .map_err(|err| Error::Other(err.to_string()))?;
    Ok(block_group_record(block_group, Some(&db_path)))
}

#[extendr]
fn repo_get_block_groups(db_path: String) -> std::result::Result<List, Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let values = BlockGroup::all(&conn)
        .into_iter()
        .map(|bg| block_group_record(bg, Some(&db_path)))
        .collect::<Vec<_>>();
    Ok(List::from_values(values))
}

#[extendr]
fn repo_get_block_groups_by_collection(
    db_path: String,
    collection_name: String,
) -> std::result::Result<List, Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let values = BlockGroup::query(
        &conn,
        "SELECT * FROM block_groups WHERE collection_name = ?1",
        rusqlite::params![collection_name],
    )
    .into_iter()
    .map(|bg| block_group_record(bg, Some(&db_path)))
    .collect::<Vec<_>>();
    Ok(List::from_values(values))
}

#[extendr]
fn repo_create_block_group(
    db_path: String,
    name: String,
    collection_name: String,
    sample_name: String,
) -> std::result::Result<List, Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let block_group = BlockGroup::create(
        &conn,
        NewBlockGroup {
            collection_name: &collection_name,
            sample_name: &sample_name,
            name: &name,
            ..Default::default()
        },
    )
    .map_err(|err| Error::Other(err.to_string()))?;
    Ok(block_group_record(block_group, Some(&db_path)))
}

#[extendr]
fn repo_block_group_to_dict(
    db_path: String,
    block_group_id: String,
) -> std::result::Result<List, Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let bg_id = hash_id_from_string(&block_group_id).map_err(Error::Other)?;
    let graph = BlockGroup::get_graph(&conn, &bg_id);

    let nodes = graph
        .nodes()
        .map(|node| {
            list!(
                key = node_key_record(node.node_id, node.sequence_start, node.sequence_end),
                node_id = node.node_id.to_string(),
                sequence_start = node.sequence_start,
                sequence_end = node.sequence_end
            )
        })
        .collect::<Vec<_>>();

    let edges = graph
        .all_edges()
        .map(|(src, dst, edge_weights)| {
            let weights = edge_weights
                .iter()
                .map(|weight| {
                    list!(
                        edge_id = weight.edge_id.to_string(),
                        source_strand = weight.source_strand.to_string(),
                        target_strand = weight.target_strand.to_string(),
                        chromosome_index = weight.chromosome_index,
                        phased = weight.phased
                    )
                })
                .collect::<Vec<_>>();
            list!(
                source = node_key_record(src.node_id, src.sequence_start, src.sequence_end),
                target = node_key_record(dst.node_id, dst.sequence_start, dst.sequence_end),
                weights = List::from_values(weights)
            )
        })
        .collect::<Vec<_>>();

    Ok(list!(
        nodes = List::from_values(nodes),
        edges = List::from_values(edges)
    ))
}

#[extendr]
fn repo_get_block_sequence(
    db_path: String,
    node_id: String,
    sequence_start: i64,
    sequence_end: i64,
) -> std::result::Result<String, Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let node_id = hash_id_from_string(&node_id).map_err(Error::Other)?;
    let sequences = Node::get_sequences_by_node_ids(&conn, &[node_id]);
    let sequence = sequences
        .get(&node_id)
        .ok_or_else(|| Error::Other(format!("Node with id {node_id} not found")))?;
    Ok(sequence
        .get_sequence(sequence_start, sequence_end)
        .map_err(|err| Error::Other(err.to_string()))?)
}

#[extendr]
fn graph_render_frame(
    db_path: String,
    block_group_id: String,
    detail: String,
    cols: i32,
    rows: i32,
    ops: String,
) -> std::result::Result<String, Error> {
    let (conn, graph) = block_group_graph(&db_path, &block_group_id).map_err(Error::Other)?;
    let node_sizer = GenGraphNodeSizer;
    let mut controller = GraphController::new(graph, node_sizer);
    controller.set_detail_level(visual_detail(&detail).map_err(Error::Other)?);
    controller.hide_cursor();
    apply_graph_ops(&mut controller, &ops).map_err(Error::Other)?;

    let area = Rect::new(0, 0, cols as u16, rows as u16);
    let mut buf = Buffer::empty(area);
    let renderer = GenGraphNodeRenderer::new(&conn);
    GraphWidget::with_renderer(renderer).render(area, &mut buf, &mut controller);

    serde_json::to_string(&serialize_buffer(&buf, cols as u16, rows as u16))
        .map_err(|err| Error::Other(err.to_string()))
}

#[extendr]
fn graph_handle_click(
    db_path: String,
    block_group_id: String,
    detail: String,
    ops: String,
    col: i32,
    row: i32,
) -> std::result::Result<bool, Error> {
    let (_conn, graph) = block_group_graph(&db_path, &block_group_id).map_err(Error::Other)?;
    let node_sizer = GenGraphNodeSizer;
    let mut controller = GraphController::new(graph, node_sizer);
    controller.set_detail_level(visual_detail(&detail).map_err(Error::Other)?);
    controller.hide_cursor();
    apply_graph_ops(&mut controller, &ops).map_err(Error::Other)?;
    Ok(controller.handle_click(col as u16, row as u16))
}

extendr_module! {
    mod genr;
    fn init;
    fn get_gen_dir;
    fn db_context;
    fn import_fasta;
    fn import_gfa;
    fn import_genbank;
    fn import_library_files;
    fn import_library;
    fn update_with_fasta;
    fn update_with_gfa;
    fn update_with_gaf;
    fn update_with_vcf;
    fn update_with_genbank;
    fn update_with_library_files;
    fn update_with_library;
    fn update_with_sequence;
    fn export_fasta;
    fn export_gfa;
    fn export_genbank;
    fn derive_chunks;
    fn derive_subgraph;
    fn make_stitch;
    fn repo_get_gen_dir;
    fn repo_get_db_path;
    fn repo_execute;
    fn repo_query;
    fn repo_get_block_group_by_id;
    fn repo_get_block_groups;
    fn repo_get_block_groups_by_collection;
    fn repo_create_block_group;
    fn repo_block_group_to_dict;
    fn repo_get_block_sequence;
    fn graph_render_frame;
    fn graph_handle_click;
}
