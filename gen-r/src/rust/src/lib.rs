use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Cursor, Read},
    path::{Path, PathBuf},
};

use extendr_api::prelude::*;
use flate2::read::GzDecoder;
use r#gen::{
    commands::{
        commit_operation,
        graph_operations::{
            derive_chunks::derive_chunks_operation, derive_subgraph::derive_subgraph_operation,
            make_stitch::make_stitch_operation,
        },
    },
    exports::{
        fasta::export_fasta as fasta_export, genbank::export_genbank as genbank_export,
        gfa::export_gfa as gfa_export,
    },
    get_config_connection, get_connection,
    graphs::{
        combinatorial_library::{SequencePart, parse_library},
        graph_search::{GenGraphMatcher, SeedIndex, SequenceKind},
        translation::{
            CodonTable, TranslationParams, translate_annotation, translate_block_group,
            translate_from_path,
        },
    },
    views::{
        annotation_groups::{AnnotationGroupEntry, AnnotationGroupOrigin, annotation_group_names},
        annotation_track::{
            AnnotationSegment as ViewAnnotationSegment, AnnotationSpan, AnnotationTrack,
            graph_locus_from_annotation_span,
        },
        annotations::{
            AnnotationGroupTrackRequest, load_annotations_for_group, parse_translated_bed,
            parse_translated_bed_file, parse_translated_gff, parse_translated_gff_file,
        },
        gen_graph_widget::{
            GenGraphNodeRenderer, GenGraphNodeSizer, draw_annotation_labels, highlight_locus,
            locus_midpoint, reapply_overlays,
        },
        graph_overlay::{AnnotationColorCache, GraphOverlay, OverlayContent, OverlaySource},
    },
};
use gen_annotations::{
    projection::annotation_segments,
    translate::{bed::translate_bed, gff::translate_gff},
};
use gen_core::{HashId, Strand, config::Workspace, is_end_node, is_start_node, region::Region};
use gen_graph::{GenGraph, GraphNode, GraphNodeSlice};
use gen_models::{
    annotations::Annotation,
    block_group::BlockGroup,
    db::{DbContext as GenDbContext, GraphConnection},
    errors::OperationError,
    locus::GraphLocus,
    node::Node,
    operations::{
        Defaults, OperationFile, OperationInfo, OperationSummary, commit_operation_summary,
    },
    sample::{NewSample, Sample},
};
use gen_tui::{
    LineStyle, graph_controller::GraphController, graph_widget::GraphWidget, layout::VisualDetail,
    plotter::PathStyle, theme::current_theme,
};
use petgraph::{graph::NodeIndex, visit::NodeIndexable};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    widgets::StatefulWidget,
};
use rusqlite::{Connection, types::ValueRef};
use serde::{Deserialize, Serialize};
use serde_json::from_str as json_from_str;

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

fn resolve_collection_name(
    config_conn: &gen_models::db::ConfigConnection,
    collection_name: Option<String>,
) -> std::result::Result<String, String> {
    match collection_name {
        Some(name) => Ok(name),
        None => Ok(Defaults::get(config_conn)
            .and_then(|d| d.collection_name)
            .unwrap_or_else(|| "default".to_string())),
    }
}

fn begin_transactions(context: &GenDbContext) -> std::result::Result<(), String> {
    let config_conn = context.config().conn();
    let graph_conn = context.graph().conn();

    graph_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|err| format!("Failed to begin graph transaction: {err}"))?;
    config_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|err| format!("Failed to begin config transaction: {err}"))?;
    Ok(())
}

fn end_transactions(
    context: &GenDbContext,
    operation_summary: &OperationSummary,
) -> std::result::Result<(), String> {
    let config_conn = context.config().conn();
    let graph_conn = context.graph().conn();

    graph_conn
        .execute("END TRANSACTION", [])
        .map_err(|err| format!("Failed to commit graph transaction: {err}"))?;
    config_conn
        .execute("END TRANSACTION", [])
        .map_err(|err| format!("Failed to commit config transaction: {err}"))?;
    commit_operation(context, operation_summary)
        .map_err(|err| format!("Failed to commit graph operation: {err}"))?;
    Ok(())
}

fn rollback_transactions(context: &GenDbContext) {
    let config_conn = context.config().conn();
    let graph_conn = context.graph().conn();
    let _ = graph_conn.execute("ROLLBACK TRANSACTION;", []);
    let _ = config_conn.execute("ROLLBACK TRANSACTION;", []);
}

fn hash_id_from_string(value: &str) -> std::result::Result<HashId, String> {
    HashId::try_from(value.to_string()).map_err(|err| format!("Invalid hash id '{value}': {err}"))
}

fn node_record(node_id: HashId, sequence_start: i64, sequence_end: i64) -> Robj {
    let mut obj = r!(list!(
        node_id = node_id.to_string(),
        sequence_start = sequence_start,
        sequence_end = sequence_end
    ));
    obj.set_class(&["gen_node"]).unwrap();
    obj
}

/// The sequence graphs produced by a single import/update/derive call, all
/// within one sample. Index with `[[` or iterate over `$block_groups`.
fn sample_record(collection_name: &str, sample_name: &str, block_groups: Vec<Robj>) -> Robj {
    let mut obj = r!(list!(
        collection_name = collection_name,
        sample_name = sample_name,
        block_groups = List::from_values(block_groups)
    ));
    obj.set_class(&["gen_sample"]).unwrap();
    obj
}

fn strand_str(strand: Strand) -> &'static str {
    match strand {
        Strand::Forward => "+",
        Strand::Reverse => "-",
        _ => ".",
    }
}

/// Read a length-1 character value out of an R object, if it is one.
fn robj_scalar_string(obj: &Robj) -> Option<String> {
    obj.as_str()
        .map(String::from)
        .or_else(|| obj.as_string_vector().and_then(|v| v.into_iter().next()))
}

/// If `obj` is a `gen_annotation` record (a named list with an `id` field),
/// return the parsed annotation hash id.
fn gen_annotation_record_id(obj: &Robj) -> std::result::Result<Option<HashId>, Error> {
    let Some(list) = obj.as_list() else {
        return Ok(None);
    };
    let Some(names) = call!("names", obj).ok().and_then(|r| r.as_string_vector()) else {
        return Ok(None);
    };
    let id_value = list
        .values()
        .zip(names)
        .find(|(_, name)| name == "id")
        .map(|(value, _)| value);
    match id_value.and_then(|v| robj_scalar_string(&v)) {
        Some(id) => Ok(Some(hash_id_from_string(&id).map_err(Error::Other)?)),
        None => Ok(None),
    }
}

/// Build a `gen_annotation` R record (id, name, group, kind, segments, length, locus).
fn annotation_record(conn: &GraphConnection, annotation: &Annotation, graph: &GenGraph) -> Robj {
    let segments = annotation_segments(conn, annotation, None);
    let span = AnnotationSpan {
        id: annotation.id,
        name: annotation.name.clone(),
        segments: segments
            .iter()
            .map(|s| ViewAnnotationSegment {
                node_id: s.node_id,
                start: s.range.start,
                end: s.range.end,
                strand: s.strand,
            })
            .collect(),
    };
    let locus_robj: Robj = match graph_locus_from_annotation_span(&span, graph) {
        Some(locus) => graph_locus_record(&locus),
        None => r!(NULL),
    };
    let length: i64 = segments.iter().map(|s| s.range.end - s.range.start).sum();
    let segment_records: Vec<Robj> = segments
        .iter()
        .map(|s| {
            let mut seg = r!(list!(
                node_id = s.node_id.to_string(),
                start = s.range.start,
                end = s.range.end,
                strand = strand_str(s.strand)
            ));
            seg.set_class(&["gen_annotation_segment"]).unwrap();
            seg
        })
        .collect();
    let kind = annotation
        .extra
        .as_ref()
        .and_then(|e| e.genbank.as_ref())
        .map(|g| g.kind.clone());
    let kind_robj: Robj = match kind {
        Some(k) => r!(k),
        None => r!(NULL),
    };
    let mut obj = r!(list!(
        id = annotation.id.to_string(),
        name = annotation.name.as_str(),
        group = annotation.group.as_str(),
        kind = kind_robj,
        segments = List::from_values(segment_records),
        length = length,
        locus = locus_robj
    ));
    obj.set_class(&["gen_annotation"]).unwrap();
    obj
}

/// Rich `gen_annotation` records for a block group, following the sample
/// lineage. Single listing route shared by `SequenceGraph` and `Repository`.
fn list_annotation_records(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: &HashId,
    collection_name: &str,
    sample_name: &str,
    name: &str,
) -> std::result::Result<List, Error> {
    let graph = BlockGroup::get_graph(conn, workspace, block_group_id, None)
        .map_err(|e| Error::Other(e.to_string()))?;
    let annotations = Annotation::query_with_lineage(conn, collection_name, sample_name, name)
        .map_err(|e| Error::Other(e.to_string()))?;
    let records: Vec<Robj> = annotations
        .iter()
        .map(|a| annotation_record(conn, a, &graph))
        .collect();
    Ok(List::from_values(records))
}

fn run_translation_operation<F>(
    context: &GenDbContext,
    label: &str,
    f: F,
) -> std::result::Result<BlockGroup, Error>
where
    F: FnOnce() -> std::result::Result<BlockGroup, r#gen::graphs::translation::TranslationError>,
{
    let graph_conn = context.graph().conn();
    graph_conn
        .execute("BEGIN TRANSACTION", [])
        .map_err(|e| Error::Other(e.to_string()))?;

    let protein_block_group = match f() {
        Ok(block_group) => block_group,
        Err(error) => {
            let _ = graph_conn.execute("ROLLBACK", []);
            return Err(Error::Other(error.to_string()));
        }
    };

    graph_conn
        .execute("END TRANSACTION", [])
        .map_err(|e| Error::Other(e.to_string()))?;

    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![],
            description: "translate annotation".to_string(),
        },
        format!(
            " {}: protein sequence graph derived from {label}",
            protein_block_group.name
        ),
    );
    commit_operation_summary(context, &operation_summary)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(protein_block_group)
}

fn node_slice_record(
    node_id: HashId,
    sequence_start: i64,
    sequence_end: i64,
    relative_start: usize,
    relative_end: usize,
    strand: Strand,
) -> Robj {
    let mut obj = r!(list!(
        node = node_record(node_id, sequence_start, sequence_end),
        start = relative_start as i64,
        end = relative_end as i64,
        strand = strand_str(strand)
    ));
    obj.set_class(&["gen_node_slice"]).unwrap();
    obj
}

fn position_record(
    node_id: HashId,
    sequence_start: i64,
    sequence_end: i64,
    offset: usize,
    strand: Strand,
) -> Robj {
    let mut obj = r!(list!(
        node = node_record(node_id, sequence_start, sequence_end),
        offset = offset as i64,
        strand = strand_str(strand)
    ));
    obj.set_class(&["gen_position"]).unwrap();
    obj
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

fn is_xstringset(obj: &Robj) -> bool {
    call!("is", obj, "XStringSet")
        .ok()
        .and_then(|r| r.as_logical_vector())
        .and_then(|v| v.into_iter().next())
        .map(|b| b == Rbool::from(true))
        .unwrap_or(false)
}

fn parse_column(column: &Robj) -> std::result::Result<Vec<SequencePart>, String> {
    let sequences = if is_xstringset(column) {
        call!("as.character", column)
            .ok()
            .and_then(|r| r.as_string_vector())
            .ok_or_else(|| "Failed to convert XStringSet to character strings.".to_string())?
    } else {
        column.as_string_vector().ok_or_else(|| {
            "Each library column must be a named character vector or XStringSet.".to_string()
        })?
    };
    let names = call!("names", column)
        .ok()
        .and_then(|r| r.as_string_vector())
        .ok_or_else(|| "Library column must have a name for every element.".to_string())?;
    if names.len() != sequences.len() {
        return Err("Library column names and sequences have different lengths.".to_string());
    }
    Ok(names
        .into_iter()
        .zip(sequences)
        .map(|(name, sequence)| SequencePart {
            sequence_length: sequence.len() as i64,
            name,
            sequence,
        })
        .collect())
}

fn parse_parts_list(parts_list: Robj) -> std::result::Result<Vec<Vec<SequencePart>>, String> {
    let outer = parts_list
        .as_list()
        .ok_or_else(|| "parts_list must be a list of columns.".to_string())?;
    outer.values().map(|column| parse_column(&column)).collect()
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
    workspace: &Workspace,
    controller: &GraphController<GenGraph, GenGraphNodeSizer>,
    sequence_graph_id: &HashId,
    tracks_json: &str,
) -> Vec<AnnotationTrack> {
    let specs: Vec<TrackSpec> = match serde_json::from_str(tracks_json) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    if specs.is_empty() {
        return Vec::new();
    }

    let node_filter: HashSet<HashId> = controller
        .graph()
        .nodes()
        .filter(|n| !is_start_node(n.node_id) && !is_end_node(n.node_id))
        .map(|n| n.node_id)
        .collect();

    let mut tracks = Vec::new();
    for spec in specs {
        match spec {
            TrackSpec::Group { name } => {
                if let Ok(bg) = BlockGroup::get_by_id(conn, sequence_graph_id, None) {
                    let entry = AnnotationGroupEntry {
                        id: name.clone(),
                        name: name.clone(),
                        sample_name: bg.sample_name.clone(),
                        source_block_group_id: *sequence_graph_id,
                        origin: AnnotationGroupOrigin::CurrentSample,
                    };
                    let request = AnnotationGroupTrackRequest {
                        conn,
                        workspace,
                        history_ref: None,
                        current_block_group: &bg,
                        entry: &entry,
                        node_ids: &node_filter,
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
                    workspace,
                    sequence_graph_id,
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
    workspace: &Workspace,
    sequence_graph_id: &HashId,
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

    if let (Some(sample_name), Ok(bg)) =
        (sample, BlockGroup::get_by_id(conn, sequence_graph_id, None))
    {
        let mut buffer: Vec<u8> = Vec::new();
        let translated = match ext.as_str() {
            "gff" | "gff3" => File::open(file_path)
                .ok()
                .and_then(|f| {
                    translate_gff(
                        conn,
                        workspace,
                        &bg.collection_name,
                        sample_name,
                        None,
                        BufReader::new(f),
                        &mut buffer,
                    )
                    .ok()
                })
                .is_some(),
            "bed" => File::open(file_path)
                .ok()
                .and_then(|f| {
                    translate_bed(
                        conn,
                        workspace,
                        &bg.collection_name,
                        sample_name,
                        None,
                        f,
                        &mut buffer,
                    )
                    .ok()
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

fn parse_hl_op(op: &str) -> std::result::Result<(GraphLocus, ratatui::style::Color), String> {
    let parts = op.split(',').collect::<Vec<_>>();
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
    let mut slices = Vec::with_capacity(n);
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
        let block = GraphNode {
            node_id,
            sequence_start: seq_start,
            sequence_end: seq_end,
        };
        let slice_start = if i == 0 { start_offset } else { 0 };
        let slice_end = if i == n - 1 {
            end_offset
        } else {
            block.length() as usize
        };
        slices.push(GraphNodeSlice {
            block,
            start: slice_start,
            end: slice_end,
            strand,
        });
    }
    Ok((GraphLocus { slices }, color))
}

fn apply_graph_ops(
    controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    ops: &str,
) -> std::result::Result<(), String> {
    let mut deferred_hl: Vec<&str> = Vec::new();

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
            Some("goto") | Some("gotoc") => {
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
                if !controller.graph().contains_node(node) {
                    continue;
                }
                controller.set_detail_level(VisualDetail::Full);
                if let Ok((partition_idx, _)) = controller
                    .partition_controller
                    .partition_table
                    .find_node(&node)
                {
                    let _ = controller.ensure_partition_loaded(partition_idx);
                    let _ = controller.set_anchor_partition(partition_idx);
                }
                let domain_idx = NodeIndex::new(NodeIndexable::to_index(controller.graph(), node));
                controller.go_to_node(domain_idx, (frac_x, 0.5));
                if parts[0] == "goto" {
                    controller.queue_snap_left();
                }
                controller.hide_cursor();
                // Resolve the pending goto's camera move now rather than leaving it
                // deferred to the final widget render, so later ops in this batch
                // (e.g. "m" pan) apply on top of it instead of being overwritten by it.
                controller
                    .rebuild_viewport_graph()
                    .map_err(|err| format!("Failed to resolve goto in '{op}': {err}"))?;
            }
            Some("hl") => deferred_hl.push(op),
            Some("clrhl") => {
                controller.clear_all_highlights();
                deferred_hl.clear();
            }
            Some(other) => return Err(format!("Unknown graph op prefix '{other}'.")),
            None => {}
        }
    }

    for op in deferred_hl {
        let (locus, color) = parse_hl_op(op)?;
        let style = PathStyle::new(color)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        highlight_locus(controller, &locus, style);
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

fn graph_locus_record(locus: &GraphLocus) -> Robj {
    let first = &locus.slices[0];
    let last = locus.slices.last().unwrap();
    let start = position_record(
        first.block.node_id,
        first.block.sequence_start,
        first.block.sequence_end,
        first.start,
        first.strand,
    );
    let end = position_record(
        last.block.node_id,
        last.block.sequence_start,
        last.block.sequence_end,
        last.end,
        last.strand,
    );
    let slices = locus
        .slices
        .iter()
        .map(|s| {
            node_slice_record(
                s.block.node_id,
                s.block.sequence_start,
                s.block.sequence_end,
                s.start,
                s.end,
                s.strand,
            )
        })
        .collect::<Vec<_>>();
    let overall_strand = {
        let mut iter = locus.slices.iter().map(|s| s.strand);
        match iter.next() {
            None => ".",
            Some(first) => {
                if iter.all(|s| s == first) {
                    strand_str(first)
                } else {
                    "mixed"
                }
            }
        }
    };
    // center = TRUE targets the span's true midpoint (computed here) rather than its start,
    // so the match sits balanced in the viewport instead of against the edge.
    let midpoint = locus_midpoint(locus)
        .map(|(slice, offset)| {
            position_record(
                slice.block.node_id,
                slice.block.sequence_start,
                slice.block.sequence_end,
                offset,
                slice.strand,
            )
        })
        .unwrap_or_else(|| NULL.into());
    let mut obj = r!(list!(
        start = start,
        end = end,
        midpoint = midpoint,
        slices = List::from_values(slices),
        strand = overall_strand
    ));
    obj.set_class(&["gen_locus"]).unwrap();
    obj
}

#[extendr]
#[derive(Clone)]
struct Repository {
    context: GenDbContext,
}

// DbContext uses Rc internally; R is single-threaded so this is safe.
unsafe impl Send for Repository {}
unsafe impl Sync for Repository {}

#[extendr]
impl Repository {
    fn new(path: Nullable<String>) -> std::result::Result<Repository, Error> {
        let workspace = match nullable_string_to_option(path) {
            Some(p) => Workspace::new(p),
            None => Workspace::from_current_dir(),
        };
        let gen_dir = workspace.ensure_gen_dir();
        let config_path = gen_dir.join("gen.db");
        let config_conn = get_config_connection(Some(config_path))
            .map_err(|e| Error::Other(format!("Failed to open config database: {e}")))?;
        let db_path = gen_dir.join("default.db");
        let graph_conn = get_connection(db_path.clone()).map_err(|e| {
            Error::Other(format!(
                "Failed to open database '{}': {e}",
                db_path.display()
            ))
        })?;
        Ok(Repository {
            context: GenDbContext::new(workspace, graph_conn, config_conn)
                .map_err(|err| Error::Other(err.to_string()))?,
        })
    }

    fn gen_dir(&self) -> String {
        self.context
            .workspace()
            .ensure_gen_dir()
            .to_string_lossy()
            .into_owned()
    }

    fn db_path(&self) -> String {
        self.context
            .graph()
            .conn()
            .path()
            .map(|p| p.to_string())
            .unwrap_or_else(|| {
                self.context
                    .workspace()
                    .ensure_gen_dir()
                    .join("default.db")
                    .to_string_lossy()
                    .into_owned()
            })
    }

    fn execute(&self, query: String) -> std::result::Result<(), Error> {
        self.context
            .graph()
            .conn()
            .execute(&query, [])
            .map(|_| ())
            .map_err(|e| Error::Other(format!("SQLite error: {e}")))
    }

    fn query(&self, query: String) -> std::result::Result<List, Error> {
        query_rows(self.context.graph().conn(), &query).map_err(Error::Other)
    }

    fn get_sequence_graph_by_id(&self, id: String) -> std::result::Result<SequenceGraph, Error> {
        let conn = self.context.graph().conn();
        let bg_id = hash_id_from_string(&id).map_err(Error::Other)?;
        let bg =
            BlockGroup::get_by_id(conn, &bg_id, None).map_err(|e| Error::Other(e.to_string()))?;
        Ok(self.to_sequence_graph(bg))
    }

    fn get_sequence_graphs(&self) -> std::result::Result<List, Error> {
        let conn = self.context.graph().conn();
        let values = BlockGroup::select(conn)
            .load()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .map(|bg| r!(self.to_sequence_graph(bg)))
            .collect::<Vec<_>>();
        Ok(List::from_values(values))
    }

    fn get_sequence_graphs_by_collection(
        &self,
        collection_name: String,
    ) -> std::result::Result<List, Error> {
        let conn = self.context.graph().conn();
        let values = BlockGroup::select(conn)
            .collection_name(collection_name)
            .load()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .map(|bg| r!(self.to_sequence_graph(bg)))
            .collect::<Vec<_>>();
        Ok(List::from_values(values))
    }

    /// All samples in the repository, each holding its sequence graphs.
    fn get_samples(&self) -> std::result::Result<List, Error> {
        let conn = self.context.graph().conn();
        let mut samples: Vec<(String, String, Vec<Robj>)> = Vec::new();
        for bg in BlockGroup::select(conn)
            .load()
            .map_err(|error| Error::Other(error.to_string()))?
        {
            let collection_name = bg.collection_name.clone();
            let sample_name = bg.sample_name.clone();
            let sg = r!(self.to_sequence_graph(bg));
            match samples
                .iter_mut()
                .find(|(c, s, _)| *c == collection_name && *s == sample_name)
            {
                Some((_, _, block_groups)) => block_groups.push(sg),
                None => samples.push((collection_name, sample_name, vec![sg])),
            }
        }
        let values = samples
            .into_iter()
            .map(|(c, s, block_groups)| sample_record(&c, &s, block_groups))
            .collect::<Vec<_>>();
        Ok(List::from_values(values))
    }

    fn get_node_sequence(
        &self,
        node_id: String,
        sequence_start: i64,
        sequence_end: i64,
    ) -> std::result::Result<String, Error> {
        let conn = self.context.graph().conn();
        let nid = hash_id_from_string(&node_id).map_err(Error::Other)?;
        let sequences =
            Node::get_sequences_by_node_ids(conn, self.context.workspace(), &[nid], None);
        let seq = sequences
            .get(&nid)
            .ok_or_else(|| Error::Other(format!("Node with id {nid} not found")))?;
        seq.get_sequence(sequence_start, sequence_end)
            .map_err(|e| Error::Other(e.to_string()))
    }

    fn import_fasta(
        &self,
        filename: String,
        sample: String,
        shallow: bool,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::imports::fasta::import_fasta(
            &self.context,
            &filename,
            &collection_name,
            &sample,
            shallow,
            &[],
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &sample))
            }
            Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
                rollback_transactions(&self.context);
                Err(Error::Other("Fasta contents already exist.".to_string()))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Import failed: {e}")))
            }
        }
    }

    fn import_reference_fasta(
        &self,
        filename: String,
        reference: String,
        shallow: bool,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        if let Err(e) = Sample::get_or_create(
            self.context.graph().conn(),
            NewSample {
                name: &reference,
                is_reference: true,
            },
        ) {
            rollback_transactions(&self.context);
            return Err(Error::Other(format!(
                "Failed to create reference sample: {e}"
            )));
        }
        match r#gen::imports::fasta::import_fasta(
            &self.context,
            &filename,
            &collection_name,
            &reference,
            shallow,
            &[],
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &reference))
            }
            Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
                rollback_transactions(&self.context);
                Err(Error::Other("Fasta contents already exist.".to_string()))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Import failed: {e}")))
            }
        }
    }

    fn import_sequences(
        &self,
        names: Vec<String>,
        sequences: Vec<String>,
        sample: String,
        collection: Nullable<String>,
    ) -> std::result::Result<String, Error> {
        if names.len() != sequences.len() {
            return Err(Error::Other(
                "names and sequences must have the same length".to_string(),
            ));
        }
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        let entries: Vec<(String, String)> = names.into_iter().zip(sequences).collect();
        match r#gen::imports::sequences::import_sequences(
            &self.context,
            &entries,
            &collection_name,
            &sample,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok("Sequences imported.".to_string())
            }
            Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
                rollback_transactions(&self.context);
                Err(Error::Other("Sequence contents already exist.".to_string()))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Import failed: {e}")))
            }
        }
    }

    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn import_genomic_regions(
        &self,
        seq_names: Vec<String>,
        seq_sequences: Vec<String>,
        region_names: Vec<String>,
        region_seq_names: Vec<String>,
        region_starts: Vec<f64>,
        region_ends: Vec<f64>,
        sample: String,
        collection: Nullable<String>,
    ) -> std::result::Result<String, Error> {
        if seq_names.len() != seq_sequences.len() {
            return Err(Error::Other(
                "seq_names and seq_sequences must have the same length".to_string(),
            ));
        }
        let n = region_names.len();
        if region_seq_names.len() != n || region_starts.len() != n || region_ends.len() != n {
            return Err(Error::Other("region_names, region_seq_names, region_starts, and region_ends must all have the same length".to_string()));
        }
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let reference_sequences: Vec<(String, String)> =
            seq_names.into_iter().zip(seq_sequences).collect();
        let regions: Vec<(String, String, i64, i64)> = region_names
            .into_iter()
            .zip(region_seq_names)
            .zip(region_starts)
            .zip(region_ends)
            .map(|(((rn, rsn), rs), re)| (rn, rsn, rs as i64, re as i64))
            .collect();
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::imports::sequences::import_genomic_regions(
            &self.context,
            &reference_sequences,
            &regions,
            &collection_name,
            &sample,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok("Genomic regions imported.".to_string())
            }
            Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
                rollback_transactions(&self.context);
                Err(Error::Other(
                    "Genomic region contents already exist.".to_string(),
                ))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Import failed: {e}")))
            }
        }
    }

    fn import_gfa(
        &self,
        filename: String,
        sample: String,
        collection: Nullable<String>,
    ) -> std::result::Result<SequenceGraph, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::imports::gfa::import_gfa(
            &self.context,
            Path::new(&filename),
            &collection_name,
            &sample,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                self.get_block_group(&collection_name, &sample, "")
            }
            Err(r#gen::imports::gfa::GFAImportError::OperationError(OperationError::NoChanges)) => {
                rollback_transactions(&self.context);
                Err(Error::Other("GFA already exists.".to_string()))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Import failed: {e}")))
            }
        }
    }

    fn import_genbank(
        &self,
        filename: String,
        sample: String,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let mut reader = read_genbank_reader(&filename).map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::imports::genbank::import_genbank(
            &self.context,
            &mut reader,
            &*collection_name,
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
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &sample))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Import failed: {e}")))
            }
        }
    }

    fn import_library_files(
        &self,
        library_name: String,
        parts: String,
        library: String,
        sample: String,
        collection: Nullable<String>,
    ) -> std::result::Result<SequenceGraph, Error> {
        let parts_list = parse_library(&parts, &library)
            .map_err(|e| Error::Other(format!("Problem parsing library files: {e}")))?;
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::imports::library::import_library(
            &self.context,
            &collection_name,
            &sample,
            &library_name,
            parts_list,
            Some(&parts),
            Some(&library),
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                self.get_block_group(&collection_name, &sample, &library_name)
            }
            Err(r#gen::imports::library::LibraryImportError::OperationError(
                OperationError::NoChanges,
            )) => {
                rollback_transactions(&self.context);
                Err(Error::Other("Library already exists.".to_string()))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Library import failed: {e}")))
            }
        }
    }

    fn import_library(
        &self,
        library_name: String,
        parts_list: Robj,
        sample: Nullable<String>,
        collection: Nullable<String>,
    ) -> std::result::Result<SequenceGraph, Error> {
        let rust_parts_list = parse_parts_list(parts_list).map_err(Error::Other)?;
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let sample_name = nullable_string_to_option(sample)
            .unwrap_or_else(|| gen_models::sample::Sample::DEFAULT_NAME.to_string());
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::imports::library::import_library(
            &self.context,
            &collection_name,
            &sample_name,
            &library_name,
            rust_parts_list,
            None,
            None,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                self.get_block_group(&collection_name, &sample_name, &library_name)
            }
            Err(r#gen::imports::library::LibraryImportError::OperationError(
                OperationError::NoChanges,
            )) => {
                rollback_transactions(&self.context);
                Err(Error::Other("Library already exists.".to_string()))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Library import failed: {e}")))
            }
        }
    }

    fn update_with_fasta(
        &self,
        filename: String,
        sample: String,
        new_sample: String,
        region_name: String,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::fasta::update_with_fasta(
            &self.context,
            &collection_name,
            &sample,
            &new_sample,
            &region_name,
            &filename,
            false,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &new_sample))
            }
            Err(r#gen::fasta::FastaError::OperationError(OperationError::NoChanges)) => {
                rollback_transactions(&self.context);
                Err(Error::Other("Fasta contents already exist.".to_string()))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn update_with_gfa(
        &self,
        filename: String,
        sample: String,
        new_sample: String,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::gfa::update_with_gfa(
            &self.context,
            &collection_name,
            &sample,
            &new_sample,
            &filename,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &new_sample))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn update_with_gaf(
        &self,
        filename: String,
        csv: String,
        sample: String,
        parent_sample: Nullable<String>,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::gaf::update_with_gaf(
            &self.context,
            &filename,
            &csv,
            &collection_name,
            &sample,
            nullable_string_to_option(parent_sample).as_deref(),
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &sample))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn update_with_vcf(
        &self,
        filename: String,
        genotype: Nullable<String>,
        sample: Nullable<String>,
        reference: Vec<String>,
        in_place: bool,
        collection: Nullable<String>,
    ) -> std::result::Result<List, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let genotype = nullable_string_to_option(genotype).unwrap_or_default();
        let sample = nullable_string_to_option(sample);
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::vcf::update_with_vcf(
            &self.context,
            &filename,
            &collection_name,
            genotype,
            sample.as_deref(),
            reference,
            in_place,
        ) {
            Ok((operation_summary, output_samples)) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                let samples = output_samples
                    .into_iter()
                    .map(|sample_name| self.block_groups_in_sample(&collection_name, &sample_name))
                    .collect::<Vec<_>>();
                Ok(List::from_values(samples))
            }
            Err(r#gen::updates::vcf::VcfError::OperationError(OperationError::NoChanges)) => {
                rollback_transactions(&self.context);
                Err(Error::Other(
                    "No changes made. Provide sample and genotype if missing from VCF.".to_string(),
                ))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn update_with_genbank(
        &self,
        filename: String,
        sample: String,
        create_missing: bool,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let mut reader = read_genbank_reader(&filename).map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::genbank::update_with_genbank(
            &self.context,
            &mut reader,
            Some(collection_name.as_str()),
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
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &sample))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn update_with_library_files(
        &self,
        sample: String,
        new_sample: String,
        path_name: String,
        library: String,
        parts: String,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let parts_list = parse_library(&parts, &library)
            .map_err(|e| Error::Other(format!("Couldn't parse library files: {e}")))?;
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::library::update_with_library(
            &self.context,
            &collection_name,
            &sample,
            &new_sample,
            &path_name,
            parts_list,
            Some(&parts),
            Some(&library),
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &new_sample))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn update_with_library(
        &self,
        sample: Nullable<String>,
        new_sample_name: String,
        path_name: String,
        parts_list: Robj,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let rust_parts_list = parse_parts_list(parts_list).map_err(Error::Other)?;
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let sample_name = nullable_string_to_option(sample)
            .unwrap_or_else(|| gen_models::sample::Sample::DEFAULT_NAME.to_string());
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::library::update_with_library(
            &self.context,
            &collection_name,
            &sample_name,
            &new_sample_name,
            &path_name,
            rust_parts_list,
            None,
            None,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &new_sample_name))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn update_with_sequence(
        &self,
        sequence: String,
        sample: String,
        new_sample: String,
        region_name: String,
        no_reference_path_update: bool,
        collection: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::updates::sequence::update_with_sequence(
            &self.context,
            &collection_name,
            &sample,
            &new_sample,
            &region_name,
            &sequence,
            no_reference_path_update,
        ) {
            Ok(operation_summary) => {
                end_transactions(&self.context, &operation_summary).map_err(Error::Other)?;
                Ok(self.block_groups_in_sample(&collection_name, &new_sample))
            }
            Err(e) => {
                rollback_transactions(&self.context);
                Err(Error::Other(format!("Update failed: {e}")))
            }
        }
    }

    fn export_fasta(
        &self,
        filename: String,
        sample: Nullable<String>,
        collection: Nullable<String>,
    ) -> std::result::Result<String, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        r#gen::exports::fasta::export_fasta(
            self.context.graph().conn(),
            self.context.workspace(),
            &collection_name,
            nullable_string_to_option(sample).as_deref(),
            &PathBuf::from(&filename),
            None,
        )
        .map_err(|e| Error::Other(format!("FASTA export failed: {e}")))?;
        Ok(filename)
    }

    fn export_gfa(
        &self,
        filename: String,
        sample: String,
        node_max: Nullable<i64>,
        collection: Nullable<String>,
    ) -> std::result::Result<String, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        r#gen::exports::gfa::export_gfa(
            self.context.graph().conn(),
            self.context.workspace(),
            &collection_name,
            &PathBuf::from(&filename),
            &sample,
            nullable_i64_to_option(node_max),
            None,
        )
        .map_err(|e| Error::Other(format!("GFA export failed: {e}")))?;
        Ok(filename)
    }

    fn export_genbank(
        &self,
        filename: String,
        sample: String,
        collection: Nullable<String>,
    ) -> std::result::Result<String, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let writer = BufWriter::new(File::create(&filename).map_err(|e| {
            Error::Other(format!("Failed to create GenBank file '{filename}': {e}"))
        })?);
        r#gen::exports::genbank::export_genbank(
            self.context.graph().conn(),
            self.context.workspace(),
            &collection_name,
            &sample,
            writer,
            None,
        )
        .map_err(|e| Error::Other(format!("GenBank export failed: {e}")))?;
        Ok(filename)
    }

    fn stitch(
        &self,
        collection_name: String,
        sample_name: String,
        new_sample: String,
        new_region: String,
        regions: String,
    ) -> std::result::Result<SequenceGraph, Error> {
        make_stitch_operation(
            &self.context,
            Some(collection_name.clone()),
            sample_name,
            new_sample.clone(),
            regions,
            new_region.clone(),
        )
        .map_err(|e| Error::Other(format!("Error stitching block groups: {e}")))?;
        let conn = self.context.graph().conn();
        BlockGroup::select(conn)
            .collection_name(collection_name)
            .sample_name(new_sample)
            .name(new_region)
            .load()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .next()
            .map(|bg| self.to_sequence_graph(bg))
            .ok_or_else(|| {
                Error::Other("Stitched block group not found after creation".to_string())
            })
    }

    fn build_index(
        &self,
        sequence_graph_ids: Vec<String>,
        sequence_kind: String,
        k: i32,
    ) -> std::result::Result<(), Error> {
        let kind = parse_sequence_kind_r(&sequence_kind).map_err(Error::Other)?;
        let normalized = kind != SequenceKind::Exact;
        let conn = self.context.graph().conn();
        let index_dir = self
            .context
            .workspace()
            .ensure_gen_dir()
            .join("search_index");
        fs::create_dir_all(&index_dir)
            .map_err(|e| Error::Other(format!("Failed to create index dir: {e}")))?;
        let bgs: Vec<_> = if sequence_graph_ids.is_empty() {
            BlockGroup::select(conn)
                .load()
                .map_err(|error| Error::Other(error.to_string()))?
        } else {
            sequence_graph_ids
                .iter()
                .filter_map(|id| hash_id_from_string(id).ok())
                .filter_map(|id| BlockGroup::get_by_id(conn, &id, None).ok())
                .collect()
        };
        for bg in bgs {
            let graph = BlockGroup::get_graph(conn, self.context.workspace(), &bg.id, None)
                .map_err(|e| Error::Other(e.to_string()))?;
            let matcher = GenGraphMatcher::new_with_sequence_kind(
                conn,
                self.context.workspace(),
                graph,
                kind,
            );
            let index = SeedIndex::build(&matcher, k as usize, normalized);
            let path = index_dir.join(format!("{}.bin", bg.id));
            let bytes = index
                .to_bytes_with_header()
                .map_err(|e| Error::Other(format!("Failed to serialize index: {e}")))?;
            fs::write(&path, bytes)
                .map_err(|e| Error::Other(format!("Failed to write index: {e}")))?;
        }
        Ok(())
    }

    fn search(
        &self,
        query: String,
        sequence_graph_ids: Vec<String>,
        sequence_kind: String,
    ) -> std::result::Result<List, Error> {
        let kind = parse_sequence_kind_r(&sequence_kind).map_err(Error::Other)?;
        let conn = self.context.graph().conn();
        let bgs: Vec<_> = if sequence_graph_ids.is_empty() {
            BlockGroup::select(conn)
                .load()
                .map_err(|error| Error::Other(error.to_string()))?
        } else {
            sequence_graph_ids
                .iter()
                .filter_map(|id| hash_id_from_string(id).ok())
                .filter_map(|id| BlockGroup::get_by_id(conn, &id, None).ok())
                .collect()
        };
        let query_bytes = query.as_bytes();
        let index_dir = self
            .context
            .workspace()
            .ensure_gen_dir()
            .join("search_index");
        let mut results = Vec::new();
        for bg in bgs {
            let graph = BlockGroup::get_graph(conn, self.context.workspace(), &bg.id, None)
                .map_err(|e| Error::Other(e.to_string()))?;
            let matcher = GenGraphMatcher::new_with_sequence_kind(
                conn,
                self.context.workspace(),
                graph,
                kind,
            );
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
                let gen_bg = self.to_sequence_graph(bg);
                results.push(list!(
                    sequence_graph = r!(gen_bg),
                    matches = List::from_values(locus_records)
                ));
            }
        }
        Ok(List::from_values(results))
    }

    fn clear_index(&self, sequence_graph_ids: Vec<String>) -> std::result::Result<(), Error> {
        let index_dir = self
            .context
            .workspace()
            .ensure_gen_dir()
            .join("search_index");
        if !index_dir.exists() {
            return Ok(());
        }
        if sequence_graph_ids.is_empty() {
            for entry in fs::read_dir(&index_dir)
                .map_err(|e| Error::Other(format!("Failed to read index dir: {e}")))?
            {
                let entry =
                    entry.map_err(|e| Error::Other(format!("Failed to read entry: {e}")))?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    let _ = fs::remove_file(&path);
                }
            }
        } else {
            for id in &sequence_graph_ids {
                let path = index_dir.join(format!("{id}.bin"));
                if path.exists() {
                    fs::remove_file(&path)
                        .map_err(|e| Error::Other(format!("Failed to delete index: {e}")))?;
                }
            }
        }
        Ok(())
    }

    fn derive_subgraph(
        &self,
        collection: Nullable<String>,
        sample: String,
        new_sample: String,
        region: String,
        backbone: Nullable<String>,
    ) -> std::result::Result<SequenceGraph, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let parsed_region = Region::parse(&region)
            .map_err(|e| Error::Other(format!("Failed to parse region '{region}': {e}")))?;
        derive_subgraph_operation(
            &self.context,
            Some(collection_name.clone()),
            sample,
            new_sample.clone(),
            region,
            nullable_string_to_option(backbone),
        )
        .map_err(|e| Error::Other(format!("Error deriving subgraph: {e}")))?;
        self.get_block_group(
            &collection_name,
            &new_sample,
            &parsed_region.name.to_string(),
        )
    }

    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn derive_chunks(
        &self,
        collection: Nullable<String>,
        sample: String,
        new_sample: String,
        region: String,
        backbone: Nullable<String>,
        breakpoints: Vec<i32>,
        chunk_size: Nullable<i64>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        derive_chunks_operation(
            &self.context,
            Some(collection_name.clone()),
            sample,
            new_sample.clone(),
            region,
            nullable_string_to_option(backbone),
            if breakpoints.is_empty() {
                None
            } else {
                Some(breakpoints.into_iter().map(i64::from).collect())
            },
            nullable_i64_to_option(chunk_size),
        )
        .map_err(|e| Error::Other(format!("Error deriving chunks: {e}")))?;
        Ok(self.block_groups_in_sample(&collection_name, &new_sample))
    }

    fn auto_load_annotation_groups(
        &self,
        sequence_graph_id: String,
    ) -> std::result::Result<Vec<String>, Error> {
        let conn = self.context.graph().conn();
        let bg_id = hash_id_from_string(&sequence_graph_id).map_err(Error::Other)?;
        let bg = BlockGroup::get_by_id(conn, &bg_id, None)
            .map_err(|e| Error::Other(format!("Block group not found: {e}")))?;
        Ok(annotation_group_names(conn, &bg, None))
    }

    fn list_annotations(&self, sequence_graph_id: String) -> std::result::Result<List, Error> {
        let conn = self.context.graph().conn();
        let bg_id = hash_id_from_string(&sequence_graph_id).map_err(Error::Other)?;
        let bg = BlockGroup::get_by_id(conn, &bg_id, None)
            .map_err(|e| Error::Other(format!("Block group not found: {e}")))?;
        list_annotation_records(
            conn,
            self.context.workspace(),
            &bg.id,
            &bg.collection_name,
            &bg.sample_name,
            &bg.name,
        )
    }

    fn render_frame(
        &self,
        sequence_graph_id: String,
        detail: String,
        cols: i32,
        rows: i32,
        ops: String,
        tracks_json: String,
        annotation_colors_json: String,
    ) -> std::result::Result<String, Error> {
        let conn = self.context.graph().conn();
        let bg_id = hash_id_from_string(&sequence_graph_id).map_err(Error::Other)?;
        let graph = BlockGroup::get_graph(conn, self.context.workspace(), &bg_id, None)
            .map_err(|e| Error::Other(e.to_string()))?;
        let node_sizer = GenGraphNodeSizer;
        let mut controller = GraphController::new(graph, node_sizer);
        controller.set_detail_level(visual_detail(&detail).map_err(Error::Other)?);
        controller.hide_cursor();

        let area = Rect::new(0, 0, cols as u16, rows as u16);
        let mut buf = Buffer::empty(area);

        let tracks = load_tracks_from_specs(
            conn,
            self.context.workspace(),
            &controller,
            &bg_id,
            &tracks_json,
        );

        // Parse the caller-supplied color map: id_hex → Some(color) to use that
        // color, None to hide the annotation entirely. Empty map means use the
        // default theme cycle for every annotation.
        let color_overrides: HashMap<String, Option<String>> =
            json_from_str(&annotation_colors_json).unwrap_or_default();

        // Build one overlay per visible annotation span. A caller-supplied override color
        // wins; spans with no override fall back to the theme rotation. `reapply_overlays`
        // then repaints every overlay, greedily reassigning colors to keep overlapping
        // spans distinguishable.
        let mut overlays: Vec<GraphOverlay> = Vec::new();
        let mut annotation_colors = AnnotationColorCache::new();
        let theme = current_theme();
        let mut theme_idx = 0usize;
        for track in &tracks {
            for span in &track.annotations {
                let id_hex = span.id.to_string();
                let color = match color_overrides.get(&id_hex) {
                    Some(None) => continue,
                    Some(Some(hex)) => parse_op_color(hex).unwrap_or_else(|_| {
                        let c = theme[0x08 + (theme_idx % 8)];
                        theme_idx += 1;
                        c
                    }),
                    None => {
                        let c = theme[0x08 + (theme_idx % 8)];
                        theme_idx += 1;
                        c
                    }
                };
                overlays.push(GraphOverlay {
                    content: OverlayContent::Span(span.clone()),
                    source: OverlaySource::Track(track.name.clone()),
                    style: PathStyle::new(color),
                });
            }
        }
        reapply_overlays(&mut controller, &mut overlays, &mut annotation_colors);

        // Apply match highlights after annotations so they render on top.
        apply_graph_ops(&mut controller, &ops).map_err(Error::Other)?;

        // Render graph with highlights applied.
        let renderer = GenGraphNodeRenderer::new(conn, self.context.workspace());
        GraphWidget::with_renderer(renderer).render(area, &mut buf, &mut controller);

        // Draw floating labels after the graph, then a single hint if any were hidden.
        let any_hidden = draw_annotation_labels(&mut buf, area, &controller, &overlays);
        if any_hidden {
            let detail_level = controller.get_detail_level();
            let note = if detail_level == VisualDetail::Full {
                " some annotations hidden due to space constraints "
            } else {
                " some annotations hidden in truncated view "
            };
            let note_style = Style::default()
                .fg(current_theme()[0x09])
                .bg(current_theme()[0x00]);
            buf.set_string(area.x, area.bottom().saturating_sub(1), note, note_style);
        }

        serde_json::to_string(&serialize_buffer(&buf, cols as u16, rows as u16))
            .map_err(|err| Error::Other(err.to_string()))
    }

    fn handle_click(
        &self,
        sequence_graph_id: String,
        detail: String,
        ops: String,
        col: i32,
        row: i32,
    ) -> std::result::Result<bool, Error> {
        let conn = self.context.graph().conn();
        let bg_id = hash_id_from_string(&sequence_graph_id).map_err(Error::Other)?;
        let graph = BlockGroup::get_graph(conn, self.context.workspace(), &bg_id, None)
            .map_err(|e| Error::Other(e.to_string()))?;
        let node_sizer = GenGraphNodeSizer;
        let mut controller = GraphController::new(graph, node_sizer);
        controller.set_detail_level(visual_detail(&detail).map_err(Error::Other)?);
        controller.hide_cursor();
        apply_graph_ops(&mut controller, &ops).map_err(Error::Other)?;
        Ok(controller.handle_click(col as u16, row as u16))
    }
}

impl Repository {
    fn to_sequence_graph(&self, bg: BlockGroup) -> SequenceGraph {
        SequenceGraph {
            context: self.context.clone(),
            id: bg.id,
            collection_name: bg.collection_name,
            sample_name: bg.sample_name,
            name: bg.name,
        }
    }

    /// All block groups currently in `(collection, sample)`, as a `gen_sample` record.
    fn block_groups_in_sample(&self, collection_name: &str, sample_name: &str) -> Robj {
        let conn = self.context.graph().conn();
        let block_groups = Sample::get_block_groups(conn, collection_name, sample_name, None)
            .into_iter()
            .map(|bg| r!(self.to_sequence_graph(bg)))
            .collect();
        sample_record(collection_name, sample_name, block_groups)
    }

    /// Look up a single block group by its deterministic (collection, sample, name) id.
    fn get_block_group(
        &self,
        collection_name: &str,
        sample_name: &str,
        name: &str,
    ) -> std::result::Result<SequenceGraph, Error> {
        let conn = self.context.graph().conn();
        Sample::get_block_groups(conn, collection_name, sample_name, None)
            .into_iter()
            .find(|bg| bg.name == name)
            .map(|bg| self.to_sequence_graph(bg))
            .ok_or_else(|| {
                Error::Other(format!(
                    "Block group '{name}' not found in sample '{sample_name}'"
                ))
            })
    }
}

// --- SequenceGraph ---

#[extendr]
#[derive(Clone)]
struct SequenceGraph {
    context: GenDbContext,
    id: HashId,
    collection_name: String,
    sample_name: String,
    name: String,
}

// DbContext uses Rc internally; R is single-threaded so this is safe.
unsafe impl Send for SequenceGraph {}
unsafe impl Sync for SequenceGraph {}

#[extendr]
impl SequenceGraph {
    fn id(&self) -> String {
        self.id.to_string()
    }

    fn collection(&self) -> &str {
        &self.collection_name
    }

    fn sample_name(&self) -> &str {
        &self.sample_name
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn db_path(&self) -> String {
        self.context
            .graph()
            .conn()
            .path()
            .map(|p| p.to_string())
            .unwrap_or_else(|| {
                self.context
                    .workspace()
                    .ensure_gen_dir()
                    .join("default.db")
                    .to_string_lossy()
                    .into_owned()
            })
    }

    fn gen_dir(&self) -> String {
        self.context
            .workspace()
            .ensure_gen_dir()
            .to_string_lossy()
            .into_owned()
    }

    fn export_fasta(&self, filename: String) -> std::result::Result<(), Error> {
        let conn = self.context.graph().conn();
        fasta_export(
            conn,
            self.context.workspace(),
            &self.collection_name,
            Some(&self.sample_name),
            &PathBuf::from(&filename),
            None,
        )
        .map_err(|e| Error::Other(format!("FASTA export failed: {e}")))
    }

    fn export_gfa(
        &self,
        filename: String,
        node_max: Nullable<i64>,
    ) -> std::result::Result<(), Error> {
        let conn = self.context.graph().conn();
        gfa_export(
            conn,
            self.context.workspace(),
            &self.collection_name,
            &PathBuf::from(&filename),
            &self.sample_name,
            nullable_i64_to_option(node_max),
            None,
        )
        .map_err(|e| Error::Other(format!("GFA export failed: {e}")))
    }

    fn export_genbank(&self, filename: String) -> std::result::Result<(), Error> {
        let conn = self.context.graph().conn();
        let writer = BufWriter::new(
            File::create(&filename)
                .map_err(|e| Error::Other(format!("Failed to create file '{filename}': {e}")))?,
        );
        genbank_export(
            conn,
            self.context.workspace(),
            &self.collection_name,
            &self.sample_name,
            writer,
            None,
        )
        .map_err(|e| Error::Other(format!("GenBank export failed: {e}")))
    }

    fn build_index(&self, sequence_kind: String, k: i32) -> std::result::Result<(), Error> {
        let kind = parse_sequence_kind_r(&sequence_kind).map_err(Error::Other)?;
        let normalized = kind != SequenceKind::Exact;
        let conn = self.context.graph().conn();
        let index_dir = self
            .context
            .workspace()
            .ensure_gen_dir()
            .join("search_index");
        fs::create_dir_all(&index_dir)
            .map_err(|e| Error::Other(format!("Failed to create index dir: {e}")))?;
        let graph = BlockGroup::get_graph(conn, self.context.workspace(), &self.id, None)
            .map_err(|e| Error::Other(e.to_string()))?;
        let matcher =
            GenGraphMatcher::new_with_sequence_kind(conn, self.context.workspace(), graph, kind);
        let index = SeedIndex::build(&matcher, k as usize, normalized);
        let path = index_dir.join(format!("{}.bin", self.id));
        let bytes = index
            .to_bytes_with_header()
            .map_err(|e| Error::Other(format!("Failed to serialize index: {e}")))?;
        fs::write(&path, bytes).map_err(|e| Error::Other(format!("Failed to write index: {e}")))
    }

    fn search(&self, query: String, sequence_kind: String) -> std::result::Result<List, Error> {
        let kind = parse_sequence_kind_r(&sequence_kind).map_err(Error::Other)?;
        let conn = self.context.graph().conn();
        let graph = BlockGroup::get_graph(conn, self.context.workspace(), &self.id, None)
            .map_err(|e| Error::Other(e.to_string()))?;
        let matcher =
            GenGraphMatcher::new_with_sequence_kind(conn, self.context.workspace(), graph, kind);
        let index_dir = self
            .context
            .workspace()
            .ensure_gen_dir()
            .join("search_index");
        let index_path = index_dir.join(format!("{}.bin", self.id));
        let index = fs::read(&index_path)
            .ok()
            .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16).ok());
        let query_bytes = query.as_bytes();
        let matches = match index {
            Some(idx) => matcher
                .find_all_with_seed_index(&idx, query_bytes)
                .unwrap_or_else(|_| matcher.find_all(query_bytes)),
            None => matcher.find_all(query_bytes),
        };
        let locus_records = matches.iter().map(graph_locus_record).collect::<Vec<_>>();
        Ok(List::from_values(locus_records))
    }

    fn clear_index(&self) -> std::result::Result<(), Error> {
        let path = self
            .context
            .workspace()
            .ensure_gen_dir()
            .join("search_index")
            .join(format!("{}.bin", self.id));
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| Error::Other(format!("Failed to delete index: {e}")))?;
        }
        Ok(())
    }

    fn get_node_sequence(
        &self,
        node_id: String,
        sequence_start: i64,
        sequence_end: i64,
    ) -> std::result::Result<String, Error> {
        let conn = self.context.graph().conn();
        let nid = hash_id_from_string(&node_id).map_err(Error::Other)?;
        let sequences =
            Node::get_sequences_by_node_ids(conn, self.context.workspace(), &[nid], None);
        let seq = sequences
            .get(&nid)
            .ok_or_else(|| Error::Other(format!("Node with id {nid} not found")))?;
        seq.get_sequence(sequence_start, sequence_end)
            .map_err(|e| Error::Other(e.to_string()))
    }

    fn subgraph(
        &self,
        new_sample: String,
        start: i64,
        end: i64,
        backbone: Nullable<String>,
    ) -> std::result::Result<SequenceGraph, Error> {
        let region = format!("{}:{start}-{end}", self.name);
        derive_subgraph_operation(
            &self.context,
            Some(self.collection_name.clone()),
            self.sample_name.clone(),
            new_sample.clone(),
            region,
            nullable_string_to_option(backbone),
        )
        .map_err(|e| Error::Other(format!("Error deriving subgraph: {e}")))?;
        let conn = self.context.graph().conn();
        BlockGroup::select(conn)
            .collection_name(&self.collection_name)
            .sample_name(new_sample)
            .name(&self.name)
            .load()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .next()
            .map(|bg| SequenceGraph {
                context: self.context.clone(),
                id: bg.id,
                collection_name: bg.collection_name,
                sample_name: bg.sample_name,
                name: bg.name,
            })
            .ok_or_else(|| Error::Other("Derived subgraph not found after creation".to_string()))
    }

    fn chunks(
        &self,
        new_sample: String,
        breakpoints: Vec<i32>,
        chunk_size: Nullable<i64>,
        backbone: Nullable<String>,
    ) -> std::result::Result<List, Error> {
        derive_chunks_operation(
            &self.context,
            Some(self.collection_name.clone()),
            self.sample_name.clone(),
            new_sample.clone(),
            self.name.clone(),
            nullable_string_to_option(backbone),
            if breakpoints.is_empty() {
                None
            } else {
                Some(breakpoints.into_iter().map(i64::from).collect())
            },
            nullable_i64_to_option(chunk_size),
        )
        .map_err(|e| Error::Other(format!("Error deriving chunks: {e}")))?;
        let conn = self.context.graph().conn();
        let prefix = format!("{}.", self.name);
        let gen_bgs: Vec<Robj> = BlockGroup::select(conn)
            .collection_name(&self.collection_name)
            .sample_name(new_sample)
            .load()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .filter(|bg| bg.name == self.name || bg.name.starts_with(&prefix))
            .map(|bg| {
                r!(SequenceGraph {
                    context: self.context.clone(),
                    id: bg.id,
                    collection_name: bg.collection_name,
                    sample_name: bg.sample_name,
                    name: bg.name,
                })
            })
            .collect();
        Ok(List::from_values(gen_bgs))
    }

    fn to_dict(&self) -> std::result::Result<List, Error> {
        let conn = self.context.graph().conn();
        let graph = BlockGroup::get_graph(conn, self.context.workspace(), &self.id, None)
            .map_err(|e| Error::Other(e.to_string()))?;

        let nodes = graph
            .nodes()
            .map(|node| {
                list!(
                    key = node_record(node.node_id, node.sequence_start, node.sequence_end),
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
                    source = node_record(src.node_id, src.sequence_start, src.sequence_end),
                    target = node_record(dst.node_id, dst.sequence_start, dst.sequence_end),
                    weights = List::from_values(weights)
                )
            })
            .collect::<Vec<_>>();

        Ok(list!(
            nodes = List::from_values(nodes),
            edges = List::from_values(edges)
        ))
    }

    /// List the gene annotations associated with this sequence graph.
    ///
    /// Reads persisted annotations from the database (including those inherited
    /// from ancestor samples), so it does not depend on the viewer/widget.
    ///
    /// @return A list of `gen_annotation` records, each with `id`, `name`,
    ///   `group`, `kind`, `segments`, `length`, and `locus` fields.
    fn list_annotations(&self) -> std::result::Result<List, Error> {
        let conn = self.context.graph().conn();
        list_annotation_records(
            conn,
            self.context.workspace(),
            &self.id,
            &self.collection_name,
            &self.sample_name,
            &self.name,
        )
    }

    /// Translate a sequence graph or annotation into a protein SequenceGraph.
    ///
    /// When `region` is a character string it is resolved against this sequence
    /// graph only, in priority order: a named path within this graph first,
    /// then an annotation in this graph's lineage. No other sequence graphs
    /// are searched.
    ///
    /// @param region One of: `NULL` to translate the entire sequence graph; a
    ///   path name or annotation name scoped to this sequence graph (path names
    ///   take priority); or a `gen_annotation` record from `list_annotations()`
    ///   (matched by database id, so unambiguous).
    /// @param start 0-based path-space coordinate to translate from. Defaults to
    ///   0 (the start of the path) when NULL, and is ignored when `region`
    ///   names an annotation (the annotation's own entry point is used
    ///   instead). Translation reads forward from this coordinate to its own
    ///   first in-frame stop codon; it is not bounded by any end coordinate.
    ///   Default: NULL.
    /// @param output_collection Collection for the protein sequence graph.
    ///   Defaults to this graph's collection.
    /// @param name Name for the protein sequence graph. Defaults to
    ///   "{region} (protein)".
    /// @param strand `"forward"` or `"reverse"`. NULL infers from the annotation.
    /// @param frame Initial reading frame offset: 0, 1, or 2.
    /// @param codon_table NCBI codon table ID (default: 1 = Standard).
    /// @return A new SequenceGraph containing the protein sequence, in this
    ///   graph's sample.
    #[expect(clippy::too_many_arguments, reason = "mirrors underlying API")]
    fn translate_annotation(
        &self,
        region: Robj,
        start: Nullable<i64>,
        output_collection: Nullable<String>,
        name: Nullable<String>,
        strand: Nullable<String>,
        frame: i32,
        codon_table: i32,
    ) -> std::result::Result<SequenceGraph, Error> {
        let start = nullable_i64_to_option(start);

        let conn = self.context.graph().conn();

        let resolved_strand = match nullable_string_to_option(strand).as_deref() {
            None => None,
            Some("forward") => Some(Strand::Forward),
            Some("reverse") => Some(Strand::Reverse),
            Some(s) => {
                return Err(Error::Other(format!(
                    "Unknown strand '{s}'; use 'forward' or 'reverse'"
                )));
            }
        };

        let table_id = codon_table as u8;
        let table = CodonTable::ncbi(table_id)
            .ok_or_else(|| Error::Other(format!("Unknown NCBI codon table id {table_id}")))?;

        let out_collection = nullable_string_to_option(output_collection)
            .unwrap_or_else(|| self.collection_name.clone());

        let mut tr_params = TranslationParams::new(&out_collection)
            .initial_frame(frame as u8)
            .map_err(|e| Error::Other(e.to_string()))?
            .codon_table(table);
        if let Some(s) = resolved_strand {
            tr_params = tr_params.strand(s);
        }
        let name = nullable_string_to_option(name);
        if let Some(n) = name.as_deref() {
            tr_params = tr_params.name(n);
        }

        let bg_id = self.id;
        let protein_bg = if region.is_null() {
            let label = self.name.clone();
            if let Some(start) = start {
                run_translation_operation(&self.context, &label, || {
                    translate_from_path(conn, self.context.workspace(), &bg_id, start, tr_params)
                })?
            } else {
                run_translation_operation(&self.context, &label, || {
                    translate_block_group(conn, self.context.workspace(), &bg_id, tr_params)
                })?
            }
        } else if let Some(name) = robj_scalar_string(&region) {
            // Resolution scoped to self: named path first, then annotation in lineage.
            let path = BlockGroup::get_path_by_name(conn, &bg_id, &name)
                .map_err(|e| Error::Other(e.to_string()))?;

            if path.is_some() {
                let coordinate = start.unwrap_or(0);
                run_translation_operation(&self.context, &name, || {
                    translate_from_path(
                        conn,
                        self.context.workspace(),
                        &bg_id,
                        coordinate,
                        tr_params,
                    )
                })?
            } else {
                let annotation = Annotation::query_with_lineage(
                    conn,
                    &self.collection_name,
                    &self.sample_name,
                    &self.name,
                )
                .map_err(|e| Error::Other(e.to_string()))?
                .into_iter()
                .find(|a| a.name.eq_ignore_ascii_case(&name))
                .ok_or_else(|| {
                    Error::Other(format!(
                        "no path or annotation named '{name}' in sequence graph '{}'",
                        self.name
                    ))
                })?;

                run_translation_operation(&self.context, &name, || {
                    translate_annotation(
                        conn,
                        self.context.workspace(),
                        &annotation,
                        Some(&bg_id),
                        tr_params,
                    )
                })?
            }
        } else if let Some(id) = gen_annotation_record_id(&region)? {
            let annotation = Annotation::get_by_id(conn, &id, None)
                .ok_or_else(|| Error::Other(format!("Annotation with id '{id}' not found")))?;
            let label = annotation.name.clone();
            run_translation_operation(&self.context, &label, || {
                translate_annotation(
                    conn,
                    self.context.workspace(),
                    &annotation,
                    Some(&bg_id),
                    tr_params,
                )
            })?
        } else {
            return Err(Error::Other(
                "region must be NULL, an annotation name, or a gen_annotation record".to_string(),
            ));
        };

        Ok(SequenceGraph {
            context: self.context.clone(),
            id: protein_bg.id,
            collection_name: protein_bg.collection_name,
            sample_name: protein_bg.sample_name,
            name: protein_bg.name,
        })
    }
}

extendr_module! {
    mod genr;
    impl Repository;
    impl SequenceGraph;
}
