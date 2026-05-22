use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Cursor, Read},
    path::{Path, PathBuf},
};

use extendr_api::prelude::*;
use flate2::read::GzDecoder;
use r#gen::{
    commands::graph_operations::{
        derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
        make_stitch::make_stitch_operation,
    },
    exports::{
        fasta::export_fasta as fasta_export, genbank::export_genbank as genbank_export,
        gfa::export_gfa as gfa_export,
    },
    get_connection,
    graphs::{
        combinatorial_library::{SequencePart, parse_library},
        graph_search::{GenGraphMatcher, GraphLocus, SeedIndex, SequenceKind},
    },
    views::{
        annotation_groups::{AnnotationGroupEntry, AnnotationGroupOrigin},
        annotation_track::AnnotationTrack,
        annotations::{
            AnnotationGroupTrackRequest, load_annotations_for_group, parse_translated_bed,
            parse_translated_bed_file, parse_translated_gff, parse_translated_gff_file,
        },
        gen_graph_widget::{
            GenGraphNodeRenderer, GenGraphNodeSizer, center_on_node_offset, highlight_match_range,
        },
    },
};
use gen_annotations::translate::{bed::translate_bed, gff::translate_gff};
use gen_core::{HashId, Strand, config::Workspace, is_end_node, is_start_node};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    block_group::BlockGroup,
    db::{DbContext as GenDbContext, GraphConnection},
    errors::OperationError,
    node::Node,
    operations::{Defaults, OperationFile, OperationInfo},
    traits::Query,
};
use gen_tui::{
    LineStyle, graph_controller::GraphController, graph_widget::GraphWidget, layout::VisualDetail,
    plotter::PathStyle,
};
use ratatui::{buffer::Buffer, layout::Rect, style::Modifier, widgets::StatefulWidget};
use rusqlite::{Connection, types::ValueRef};
use serde::{Deserialize, Serialize};

fn nullable_string_to_option(value: Nullable<String>) -> Option<String> {
    match value {
        Nullable::NotNull(value) => Some(value),
        Nullable::Null => None,
    }
}

fn nullable_i64_to_option(value: Nullable<i64>) -> Option<i64> {
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
    x: u16,
    y: u16,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bg: Option<String>,
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
    let neutral_fg = "#cdd6f4";
    let neutral_bg = "#1e1e2e";
    let mut cells = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let cell = buf.cell((col, row)).expect("cell index in bounds");
            let text = cell.symbol().to_string();
            let style = cell.style();
            let fg_str = color_to_hex(style.fg, neutral_fg);
            let bg_str = color_to_hex(style.bg, neutral_bg);

            if (text == " " || text.is_empty()) && fg_str == neutral_fg && bg_str == neutral_bg {
                continue;
            }

            cells.push(RenderedCell {
                x: col,
                y: row,
                text,
                fg: if fg_str == neutral_fg {
                    None
                } else {
                    Some(fg_str)
                },
                bg: if bg_str == neutral_bg {
                    None
                } else {
                    Some(bg_str)
                },
                bold: style.add_modifier.contains(Modifier::BOLD),
                italic: style.add_modifier.contains(Modifier::ITALIC),
                underline: style.add_modifier.contains(Modifier::UNDERLINED),
            });
        }
    }
    RenderedFrame { cols, rows, cells }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum TrackSpec {
    Group {
        name: String,
    },
    File {
        path: String,
        name: Option<String>,
        sample: Option<String>,
    },
}

fn load_tracks_from_specs(
    conn: &GraphConnection,
    controller: &GraphController<GenGraph, GenGraphNodeSizer>,
    block_group_id: &HashId,
    tracks_json: &str,
) -> Vec<AnnotationTrack> {
    let specs: Vec<TrackSpec> = match serde_json::from_str(tracks_json) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    if specs.is_empty() {
        return Vec::new();
    }

    let node_ranges: HashMap<HashId, Vec<(i64, i64)>> = controller
        .graph()
        .nodes()
        .filter(|n| !is_start_node(n.node_id) && !is_end_node(n.node_id))
        .map(|n| (n.node_id, vec![(n.sequence_start, n.sequence_end)]))
        .collect();
    let node_filter: HashSet<HashId> = node_ranges.keys().copied().collect();

    let mut tracks = Vec::new();
    for spec in specs {
        match spec {
            TrackSpec::Group { name } => {
                if let Some(bg) = BlockGroup::get_by_id(conn, block_group_id).ok() {
                    let entry = AnnotationGroupEntry {
                        id: name.clone(),
                        name: name.clone(),
                        sample_name: bg.sample_name.clone(),
                        source_block_group_id: *block_group_id,
                        origin: AnnotationGroupOrigin::CurrentSample,
                    };
                    let request = AnnotationGroupTrackRequest {
                        conn,
                        current_block_group: &bg,
                        entry: &entry,
                        visible_ranges_by_node: &node_ranges,
                    };
                    if let Ok(spans) = load_annotations_for_group(&request) {
                        tracks.push(AnnotationTrack::new(name, spans));
                    }
                }
            }
            TrackSpec::File { path, name, sample } => {
                let display_name = name.as_deref().unwrap_or(&path);
                tracks.push(load_annotation_file_as_track(
                    conn,
                    block_group_id,
                    &path,
                    display_name,
                    sample.as_deref(),
                    &node_filter,
                ));
            }
        }
    }
    tracks
}

fn load_annotation_file_as_track(
    conn: &GraphConnection,
    block_group_id: &HashId,
    file_path: &str,
    display_name: &str,
    sample: Option<&str>,
    node_filter: &HashSet<HashId>,
) -> AnnotationTrack {
    let path = Path::new(file_path);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if let (Some(sample_name), Ok(bg)) = (sample, BlockGroup::get_by_id(conn, block_group_id)) {
        let mut buffer: Vec<u8> = Vec::new();
        let translated = match ext.as_str() {
            "gff" | "gff3" => File::open(file_path)
                .ok()
                .and_then(|f| {
                    translate_gff(
                        conn,
                        &bg.collection_name,
                        sample_name,
                        BufReader::new(f),
                        &mut buffer,
                    )
                    .ok()
                })
                .is_some(),
            "bed" => File::open(file_path)
                .ok()
                .and_then(|f| {
                    translate_bed(conn, &bg.collection_name, sample_name, f, &mut buffer).ok()
                })
                .is_some(),
            _ => false,
        };
        if translated && !buffer.is_empty() {
            let spans = match ext.as_str() {
                "gff" | "gff3" => parse_translated_gff(
                    Cursor::new(buffer),
                    node_filter,
                    display_name,
                    HashMap::new(),
                ),
                _ => parse_translated_bed(
                    Cursor::new(buffer),
                    node_filter,
                    display_name,
                    HashMap::new(),
                ),
            };
            return AnnotationTrack::new(display_name.to_string(), spans);
        }
    }

    let spans = match ext.as_str() {
        "gff" | "gff3" => {
            parse_translated_gff_file(path, node_filter, display_name).unwrap_or_default()
        }
        "bed" => parse_translated_bed_file(path, node_filter, display_name).unwrap_or_default(),
        _ => Vec::new(),
    };
    AnnotationTrack::new(display_name.to_string(), spans)
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

fn parse_op_color(s: &str) -> std::result::Result<ratatui::style::Color, String> {
    use ratatui::style::Color;
    match s {
        "red" => Ok(Color::Red),
        "green" => Ok(Color::Green),
        "yellow" => Ok(Color::Yellow),
        "blue" => Ok(Color::Blue),
        "magenta" => Ok(Color::Magenta),
        "cyan" => Ok(Color::Cyan),
        "white" => Ok(Color::White),
        hex if hex.starts_with('#') && hex.len() == 7 => {
            let r = u8::from_str_radix(&hex[1..3], 16)
                .map_err(|_| format!("Bad red component in color '{hex}'"))?;
            let g = u8::from_str_radix(&hex[3..5], 16)
                .map_err(|_| format!("Bad green component in color '{hex}'"))?;
            let b = u8::from_str_radix(&hex[5..7], 16)
                .map_err(|_| format!("Bad blue component in color '{hex}'"))?;
            Ok(Color::Rgb(r, g, b))
        }
        other => Err(format!("Unknown color '{other}'")),
    }
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
            Some("goto") => {
                if parts.len() != 5 {
                    return Err(format!("Invalid goto op: {op}"));
                }
                let node_id = hash_id_from_string(parts[1])
                    .map_err(|err| format!("Invalid goto node_id in '{op}': {err}"))?;
                let seq_start = parts[2]
                    .parse::<i64>()
                    .map_err(|err| format!("Invalid goto seq_start in '{op}': {err}"))?;
                let seq_end = parts[3]
                    .parse::<i64>()
                    .map_err(|err| format!("Invalid goto seq_end in '{op}': {err}"))?;
                let frac_x = parts[4]
                    .parse::<f64>()
                    .map_err(|err| format!("Invalid goto frac_x in '{op}': {err}"))?;
                let node = GraphNode {
                    node_id,
                    sequence_start: seq_start,
                    sequence_end: seq_end,
                };
                controller.set_detail_level(VisualDetail::Full);
                center_on_node_offset(controller, node, (frac_x, 0.5));
                controller.hide_cursor();
            }
            Some("hl") => {
                // hl,{color},{start_offset},{end_offset},{strand},{n},{node_id},{seq_start},{seq_end},...
                if parts.len() < 6 {
                    return Err(format!("Invalid hl op (too short): {op}"));
                }
                let color = parse_op_color(parts[1])?;
                let start_offset = parts[2]
                    .parse::<usize>()
                    .map_err(|err| format!("Invalid hl start_offset in '{op}': {err}"))?;
                let end_offset = parts[3]
                    .parse::<usize>()
                    .map_err(|err| format!("Invalid hl end_offset in '{op}': {err}"))?;
                let strand = match parts[4] {
                    "f" => Strand::Forward,
                    "r" => Strand::Reverse,
                    _ => Strand::Unknown,
                };
                let n = parts[5]
                    .parse::<usize>()
                    .map_err(|err| format!("Invalid hl block count in '{op}': {err}"))?;
                if parts.len() != 6 + n * 3 {
                    return Err(format!(
                        "Invalid hl op: expected {} parts for {n} blocks, got {}",
                        6 + n * 3,
                        parts.len()
                    ));
                }
                let mut blocks = Vec::with_capacity(n);
                for i in 0..n {
                    let base = 6 + i * 3;
                    let node_id = hash_id_from_string(parts[base])
                        .map_err(|err| format!("Invalid hl node_id[{i}] in '{op}': {err}"))?;
                    let seq_start = parts[base + 1]
                        .parse::<i64>()
                        .map_err(|err| format!("Invalid hl seq_start[{i}] in '{op}': {err}"))?;
                    let seq_end = parts[base + 2]
                        .parse::<i64>()
                        .map_err(|err| format!("Invalid hl seq_end[{i}] in '{op}': {err}"))?;
                    blocks.push(GraphNode {
                        node_id,
                        sequence_start: seq_start,
                        sequence_end: seq_end,
                    });
                }
                let locus = GraphLocus {
                    start_offset,
                    end_offset,
                    blocks,
                    strand,
                };
                let style = PathStyle::new(color)
                    .with_line_style(LineStyle::Bold)
                    .with_merge_glyphs(true);
                highlight_match_range(controller, &locus, style);
            }
            Some("clrhl") => controller.clear_all_highlights(),
            Some(other) => return Err(format!("Unknown graph op prefix '{other}'.")),
            None => {}
        }
    }
    Ok(())
}

fn parse_sequence_kind_r(s: &str) -> std::result::Result<SequenceKind, String> {
    match s {
        "exact" => Ok(SequenceKind::Exact),
        "dna" => Ok(SequenceKind::Dna),
        "ssdna" => Ok(SequenceKind::SsDna),
        "protein" => Ok(SequenceKind::Protein),
        _ => Err(format!(
            "Unknown sequence_kind '{s}'; use 'exact', 'dna', 'ssdna', or 'protein'"
        )),
    }
}

fn graph_locus_record(locus: &GraphLocus) -> List {
    let start_block = locus.blocks[0];
    let end_block = *locus.blocks.last().unwrap();

    let start = list!(
        block = node_key_record(
            start_block.node_id,
            start_block.sequence_start,
            start_block.sequence_end
        ),
        offset = locus.start_offset as i64
    );
    let end = list!(
        block = node_key_record(
            end_block.node_id,
            end_block.sequence_start,
            end_block.sequence_end
        ),
        offset = locus.end_offset as i64
    );
    let blocks = locus
        .blocks
        .iter()
        .map(|n| node_key_record(n.node_id, n.sequence_start, n.sequence_end))
        .collect::<Vec<_>>();
    let strand = match locus.strand {
        Strand::Forward => "forward",
        Strand::Reverse => "reverse",
        _ => "unknown",
    };
    list!(
        start = start,
        end = end,
        blocks = List::from_values(blocks),
        strand = strand
    )
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
    sample: String,
    shallow: bool,
    name: Nullable<String>,
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
fn import_sequences(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    names: Vec<String>,
    sequences: Vec<String>,
    sample: String,
    name: Nullable<String>,
) -> std::result::Result<String, Error> {
    if names.len() != sequences.len() {
        return Err(Error::Other(
            "names and sequences must have the same length".to_string(),
        ));
    }
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;

    begin_transactions(&context).map_err(Error::Other)?;

    let entries: Vec<(String, String)> = names.into_iter().zip(sequences).collect();

    match r#gen::imports::sequences::import_sequences(&context, &entries, &collection_name, &sample)
    {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Sequences imported.".to_string())
        }
        Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
            rollback_transactions(&context);
            Err(Error::Other("Sequence contents already exist.".to_string()))
        }
        Err(err) => {
            rollback_transactions(&context);
            Err(Error::Other(format!("Import failed: {err}")))
        }
    }
}

#[extendr]
fn import_genomic_regions(
    workspace_path: Nullable<String>,
    db_path: Nullable<String>,
    seq_names: Vec<String>,
    seq_sequences: Vec<String>,
    region_names: Vec<String>,
    region_seq_names: Vec<String>,
    region_starts: Vec<f64>,
    region_ends: Vec<f64>,
    sample: String,
    name: Nullable<String>,
) -> std::result::Result<String, Error> {
    if seq_names.len() != seq_sequences.len() {
        return Err(Error::Other(
            "seq_names and seq_sequences must have the same length".to_string(),
        ));
    }
    let n = region_names.len();
    if region_seq_names.len() != n || region_starts.len() != n || region_ends.len() != n {
        return Err(Error::Other(
            "region_names, region_seq_names, region_starts, and region_ends must all have the same length".to_string(),
        ));
    }
    let (context, _, _) = open_db_context(
        nullable_string_to_option(workspace_path),
        nullable_string_to_option(db_path),
    )
    .map_err(Error::Other)?;
    let collection_name =
        resolve_collection_name(context.operations().conn(), nullable_string_to_option(name))
            .map_err(Error::Other)?;

    begin_transactions(&context).map_err(Error::Other)?;

    let reference_sequences: Vec<(String, String)> =
        seq_names.into_iter().zip(seq_sequences).collect();
    let regions: Vec<(String, String, i64, i64)> = region_names
        .into_iter()
        .zip(region_seq_names)
        .zip(region_starts)
        .zip(region_ends)
        .map(|(((rn, rsn), rs), re)| (rn, rsn, rs as i64, re as i64))
        .collect();

    match r#gen::imports::sequences::import_genomic_regions(
        &context,
        &reference_sequences,
        &regions,
        &collection_name,
        &sample,
    ) {
        Ok(_) => {
            end_transactions(&context).map_err(Error::Other)?;
            Ok("Genomic regions imported.".to_string())
        }
        Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
            rollback_transactions(&context);
            Err(Error::Other(
                "Genomic region contents already exist.".to_string(),
            ))
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
    sample: String,
    name: Nullable<String>,
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
    sample: String,
    name: Nullable<String>,
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
            files: vec![
                OperationFile::new(filename.clone())
                    .set_file_type(gen_models::file_types::FileTypes::GenBank),
            ],
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
    sample: String,
    name: Nullable<String>,
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
    sample: Nullable<String>,
    name: Nullable<String>,
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
    sample: String,
    new_sample: String,
    region_name: String,
    name: Nullable<String>,
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
    sample: String,
    new_sample: String,
    name: Nullable<String>,
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
    sample: String,
    parent_sample: Nullable<String>,
    name: Nullable<String>,
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
    genotype: Nullable<String>,
    sample: Nullable<String>,
    parent_samples: Vec<String>,
    in_place: bool,
    name: Nullable<String>,
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
    sample: String,
    create_missing: bool,
    name: Nullable<String>,
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
            files: vec![
                OperationFile::new(filename.clone())
                    .set_file_type(gen_models::file_types::FileTypes::GenBank),
            ],
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
    sample: String,
    new_sample: String,
    path_name: String,
    library: String,
    parts: String,
    name: Nullable<String>,
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
    sample: Nullable<String>,
    new_sample_name: String,
    path_name: String,
    parts_list: Robj,
    name: Nullable<String>,
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
    sample: String,
    new_sample: String,
    region_name: String,
    no_reference_path_update: bool,
    name: Nullable<String>,
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
    sample: Nullable<String>,
    name: Nullable<String>,
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
    sample: String,
    node_max: Nullable<i64>,
    name: Nullable<String>,
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
    sample: String,
    name: Nullable<String>,
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
    sample: String,
    new_sample: String,
    region: String,
    backbone: Nullable<String>,
    breakpoints: Vec<i32>,
    chunk_size: Nullable<i64>,
    name: Nullable<String>,
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
        if breakpoints.is_empty() {
            None
        } else {
            Some(breakpoints.into_iter().map(i64::from).collect())
        },
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
    sample: String,
    new_sample: String,
    region: String,
    backbone: Nullable<String>,
    name: Nullable<String>,
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
    sample: String,
    new_sample: String,
    regions: String,
    new_region: String,
    name: Nullable<String>,
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
    tracks_json: String,
) -> std::result::Result<String, Error> {
    let (conn, graph) = block_group_graph(&db_path, &block_group_id).map_err(Error::Other)?;
    let bg_id = hash_id_from_string(&block_group_id).map_err(Error::Other)?;
    let node_sizer = GenGraphNodeSizer;
    let mut controller = GraphController::new(graph, node_sizer);
    controller.set_detail_level(visual_detail(&detail).map_err(Error::Other)?);
    controller.hide_cursor();
    apply_graph_ops(&mut controller, &ops).map_err(Error::Other)?;

    let area = Rect::new(0, 0, cols as u16, rows as u16);
    let mut buf = Buffer::empty(area);
    let renderer = GenGraphNodeRenderer::new(&conn);
    GraphWidget::with_renderer(renderer).render(area, &mut buf, &mut controller);

    let tracks = load_tracks_from_specs(&conn, &controller, &bg_id, &tracks_json);
    let mut remaining = area;
    for track in tracks.iter().rev() {
        let height = track.draw(&mut buf, remaining, &controller);
        if height == 0 {
            break;
        }
        remaining.height = remaining.height.saturating_sub(height);
    }

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

#[extendr]
fn repo_stitch(
    workspace_path: String,
    db_path: String,
    collection_name: String,
    sample_name: String,
    new_sample: String,
    new_region: String,
    regions: String,
) -> std::result::Result<List, Error> {
    let (context, _, _) =
        open_db_context(Some(workspace_path), Some(db_path.clone())).map_err(Error::Other)?;

    make_stitch_operation(
        &context,
        Some(collection_name.clone()),
        sample_name,
        new_sample.clone(),
        regions,
        new_region.clone(),
    )
    .map_err(|err| Error::Other(format!("Error stitching block groups: {err}")))?;

    let conn = context.graph().conn();
    BlockGroup::query(
        conn,
        "SELECT * FROM block_groups WHERE collection_name = ?1 AND sample_name = ?2 AND name = ?3",
        rusqlite::params![collection_name, new_sample, new_region],
    )
    .into_iter()
    .next()
    .map(|bg| block_group_record(bg, Some(&db_path)))
    .ok_or_else(|| Error::Other("Stitched block group not found after creation".to_string()))
}

#[extendr]
fn repo_build_index(
    db_path: String,
    gen_dir: String,
    block_group_ids: Vec<String>,
    sequence_kind: String,
    k: i32,
) -> std::result::Result<(), Error> {
    let kind = parse_sequence_kind_r(&sequence_kind).map_err(Error::Other)?;
    let normalized = kind != SequenceKind::Exact;
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let index_dir = PathBuf::from(&gen_dir).join("search_index");
    fs::create_dir_all(&index_dir)
        .map_err(|e| Error::Other(format!("Failed to create index dir: {e}")))?;

    let bgs: Vec<_> = if block_group_ids.is_empty() {
        BlockGroup::all(&conn)
    } else {
        block_group_ids
            .iter()
            .filter_map(|id| hash_id_from_string(id).ok())
            .filter_map(|id| BlockGroup::get_by_id(&conn, &id).ok())
            .collect()
    };

    for bg in bgs {
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let matcher = GenGraphMatcher::new_with_sequence_kind(&conn, graph, kind);
        let index = SeedIndex::build(&matcher, k as usize, normalized);
        let path = index_dir.join(format!("{}.bin", bg.id));
        let bytes = index
            .to_bytes_with_header()
            .map_err(|e| Error::Other(format!("Failed to serialize index: {e}")))?;
        fs::write(&path, bytes).map_err(|e| Error::Other(format!("Failed to write index: {e}")))?;
    }
    Ok(())
}

#[extendr]
fn repo_search(
    db_path: String,
    gen_dir: String,
    query: String,
    block_group_ids: Vec<String>,
    sequence_kind: String,
) -> std::result::Result<List, Error> {
    let kind = parse_sequence_kind_r(&sequence_kind).map_err(Error::Other)?;
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;

    let bgs: Vec<_> = if block_group_ids.is_empty() {
        BlockGroup::all(&conn)
    } else {
        block_group_ids
            .iter()
            .filter_map(|id| hash_id_from_string(id).ok())
            .filter_map(|id| BlockGroup::get_by_id(&conn, &id).ok())
            .collect()
    };

    let query_bytes = query.as_bytes();
    let index_dir = PathBuf::from(&gen_dir).join("search_index");
    let mut results = Vec::new();

    for bg in bgs {
        let graph = BlockGroup::get_graph(&conn, &bg.id);
        let matcher = GenGraphMatcher::new_with_sequence_kind(&conn, graph, kind);

        let index_path = index_dir.join(format!("{}.bin", bg.id));
        let index = fs::read(&index_path)
            .ok()
            .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16).ok());

        let matches = match index {
            Some(idx) => matcher
                .find_all_with_seed_index(&idx, query_bytes)
                .unwrap_or_else(|_| matcher.find_all(query_bytes)),
            None => matcher.find_all(query_bytes),
        };

        if !matches.is_empty() {
            let locus_records = matches.iter().map(graph_locus_record).collect::<Vec<_>>();
            results.push(list!(
                block_group = block_group_record(bg, Some(&db_path)),
                matches = List::from_values(locus_records)
            ));
        }
    }

    Ok(List::from_values(results))
}

#[extendr]
fn repo_clear_index(
    gen_dir: String,
    block_group_ids: Vec<String>,
) -> std::result::Result<(), Error> {
    let index_dir = PathBuf::from(&gen_dir).join("search_index");
    if !index_dir.exists() {
        return Ok(());
    }

    if block_group_ids.is_empty() {
        for entry in fs::read_dir(&index_dir)
            .map_err(|e| Error::Other(format!("Failed to read index dir: {e}")))?
        {
            let entry = entry.map_err(|e| Error::Other(format!("Failed to read entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                let _ = fs::remove_file(&path);
            }
        }
    } else {
        for id in &block_group_ids {
            let path = index_dir.join(format!("{id}.bin"));
            if path.exists() {
                fs::remove_file(&path)
                    .map_err(|e| Error::Other(format!("Failed to delete index: {e}")))?;
            }
        }
    }
    Ok(())
}

#[extendr]
fn repo_bg_subgraph(
    workspace_path: String,
    db_path: String,
    collection_name: String,
    sample_name: String,
    bg_name: String,
    new_sample: String,
    start: i64,
    end: i64,
    backbone: Nullable<String>,
) -> std::result::Result<List, Error> {
    let (context, _, _) =
        open_db_context(Some(workspace_path), Some(db_path.clone())).map_err(Error::Other)?;

    let region = format!("{bg_name}:{start}-{end}");
    derive_subgraph_operation(
        &context,
        Some(collection_name.clone()),
        sample_name,
        new_sample.clone(),
        region,
        nullable_string_to_option(backbone),
    )
    .map_err(|e| Error::Other(format!("Error deriving subgraph: {e}")))?;

    let conn = context.graph().conn();
    BlockGroup::query(
        conn,
        "SELECT * FROM block_groups WHERE collection_name = ?1 AND sample_name = ?2 AND name = ?3",
        rusqlite::params![collection_name, new_sample, bg_name],
    )
    .into_iter()
    .next()
    .map(|bg| block_group_record(bg, Some(&db_path)))
    .ok_or_else(|| Error::Other("Derived subgraph not found after creation".to_string()))
}

#[extendr]
fn repo_bg_chunks(
    workspace_path: String,
    db_path: String,
    collection_name: String,
    sample_name: String,
    bg_name: String,
    new_sample: String,
    breakpoints: Vec<i32>,
    chunk_size: Nullable<i64>,
    backbone: Nullable<String>,
) -> std::result::Result<List, Error> {
    let (context, _, _) =
        open_db_context(Some(workspace_path), Some(db_path.clone())).map_err(Error::Other)?;

    derive_chunks_operation(
        &context,
        Some(collection_name.clone()),
        sample_name,
        new_sample.clone(),
        bg_name.clone(),
        nullable_string_to_option(backbone),
        if breakpoints.is_empty() {
            None
        } else {
            Some(breakpoints.into_iter().map(i64::from).collect())
        },
        match chunk_size {
            Nullable::NotNull(v) => Some(v),
            Nullable::Null => None,
        },
    )
    .map_err(|e| Error::Other(format!("Error deriving chunks: {e}")))?;

    let conn = context.graph().conn();
    let prefix = format!("{bg_name}.");
    let matching = BlockGroup::query(
        conn,
        "SELECT * FROM block_groups WHERE collection_name = ?1 AND sample_name = ?2",
        rusqlite::params![collection_name, new_sample],
    )
    .into_iter()
    .filter(|bg| bg.name == bg_name || bg.name.starts_with(&prefix))
    .map(|bg| block_group_record(bg, Some(&db_path)))
    .collect::<Vec<_>>();

    Ok(List::from_values(matching))
}

#[extendr]
fn repo_bg_export_fasta(
    db_path: String,
    collection_name: String,
    sample_name: String,
    filename: String,
) -> std::result::Result<(), Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    fasta_export(
        &conn,
        &collection_name,
        Some(&sample_name),
        &PathBuf::from(&filename),
    )
    .map_err(|e| Error::Other(format!("FASTA export failed: {e}")))
}

#[extendr]
fn repo_bg_export_gfa(
    db_path: String,
    collection_name: String,
    sample_name: String,
    filename: String,
    node_max: Nullable<i64>,
) -> std::result::Result<(), Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    gfa_export(
        &conn,
        &collection_name,
        &PathBuf::from(&filename),
        &sample_name,
        match node_max {
            Nullable::NotNull(v) => Some(v),
            Nullable::Null => None,
        },
    )
    .map_err(|e| Error::Other(format!("GFA export failed: {e}")))
}

#[extendr]
fn repo_bg_export_genbank(
    db_path: String,
    collection_name: String,
    sample_name: String,
    filename: String,
) -> std::result::Result<(), Error> {
    let conn = open_repo_connection(&db_path).map_err(Error::Other)?;
    let writer = BufWriter::new(
        File::create(&filename)
            .map_err(|e| Error::Other(format!("Failed to create file '{filename}': {e}")))?,
    );
    genbank_export(&conn, &collection_name, &sample_name, writer)
        .map_err(|e| Error::Other(format!("GenBank export failed: {e}")))
}

extendr_module! {
    mod genr;
    fn init;
    fn get_gen_dir;
    fn db_context;
    fn import_fasta;
    fn import_sequences;
    fn import_genomic_regions;
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
    fn repo_block_group_to_dict;
    fn repo_get_block_sequence;
    fn repo_stitch;
    fn repo_build_index;
    fn repo_search;
    fn repo_clear_index;
    fn repo_bg_subgraph;
    fn repo_bg_chunks;
    fn repo_bg_export_fasta;
    fn repo_bg_export_gfa;
    fn repo_bg_export_genbank;
    fn graph_render_frame;
    fn graph_handle_click;
}
