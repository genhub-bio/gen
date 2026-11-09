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
        annotation_groups::annotation_group_names,
        annotation_track::{
            AnnotationSpan, AnnotationTrack, LoadedNodeSlices, annotation_span_from_graph_locus,
            graph_locus_from_annotation_span,
        },
        annotations::{
            parse_translated_bed, parse_translated_bed_file, parse_translated_gff,
            parse_translated_gff_file,
        },
        graph_database::GraphDatabase,
        graph_session::{
            GraphPage, GraphSession, GraphSessionError, PlotOptions, validated_png_scale,
            workspace_for_connection,
        },
    },
};
use gen_annotations::{
    projection::annotation_segments,
    translate::{bed::translate_bed, gff::translate_gff},
};
use gen_core::{HashId, Strand, config::Workspace, region::Region};
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
use rusqlite::{Connection, types::ValueRef};
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
        segments: segments.clone(),
    };
    let locus_robj: Robj =
        match graph_locus_from_annotation_span(&span, &LoadedNodeSlices::new(graph)) {
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
    let midpoint = locus
        .midpoint()
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
        collection: Nullable<String>,
        fai: Nullable<String>,
        gzi: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let fai = nullable_string_to_option(fai);
        let gzi = nullable_string_to_option(gzi);
        begin_transactions(&self.context).map_err(Error::Other)?;
        match r#gen::imports::fasta::import_fasta(
            &self.context,
            &filename,
            &collection_name,
            &sample,
            fai.as_deref(),
            gzi.as_deref(),
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
        collection: Nullable<String>,
        fai: Nullable<String>,
        gzi: Nullable<String>,
    ) -> std::result::Result<Robj, Error> {
        let collection_name = resolve_collection_name(
            self.context.config().conn(),
            nullable_string_to_option(collection),
        )
        .map_err(Error::Other)?;
        let fai = nullable_string_to_option(fai);
        let gzi = nullable_string_to_option(gzi);
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
            fai.as_deref(),
            gzi.as_deref(),
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
                .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16).ok())
                .filter(|index| index.is_valid_for(&matcher));
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

    /// Builds a persistent, live `RGraphController` for `sequence_graph_id`. The returned
    /// controller is mutated in place by its own methods and renders its current state to
    /// PNG bytes on request.
    fn create_graph_controller(
        &self,
        sequence_graph_id: String,
        detail: String,
    ) -> std::result::Result<RGraphController, Error> {
        let conn = self.context.graph().conn();
        let bg_id = hash_id_from_string(&sequence_graph_id).map_err(Error::Other)?;
        let block_group = BlockGroup::get_by_id(conn, &bg_id, None)
            .map_err(|e| Error::Other(format!("Block group not found: {e}")))?;
        let database = GraphDatabase::for_connection(conn, self.context.workspace())
            .map_err(|e| Error::Other(e.to_string()))?;
        let mut page = GraphPage::new(block_group.name, database, bg_id, PlotOptions::default())
            .map_err(session_error)?;
        page.set_detail(&detail).map_err(session_error)?;
        let mut session = GraphSession::single(page);
        session.set_cursor_hidden(true).map_err(session_error)?;
        Ok(RGraphController { session })
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
            .and_then(|bytes| SeedIndex::from_bytes_with_header(&bytes, 16).ok())
            .filter(|index| index.is_valid_for(&matcher));
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
            let annotation = Annotation::select(conn)
                .get_by_id(id)
                .map_err(|error| Error::Other(error.to_string()))?
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

fn session_error(error: GraphSessionError) -> Error {
    Error::Other(error.to_string())
}

/// A persistent, mutable-in-place graph controller for the R viewers. Each method mutates the
/// shared graph session directly and rendering draws its current state to PNG bytes, so the
/// R widget shows the same pixels as the Python widget and holds no operation history.
#[extendr]
#[derive(Clone)]
struct RGraphController {
    session: GraphSession,
}

#[extendr]
impl RGraphController {
    fn set_detail(&mut self, detail: String) -> std::result::Result<(), Error> {
        self.session
            .active()
            .and_then(|page| page.set_detail(&detail))
            .map_err(session_error)
    }

    fn zoom_in(&mut self) -> std::result::Result<(), Error> {
        self.session.active().map_err(session_error)?.zoom_in();
        Ok(())
    }

    fn zoom_out(&mut self) -> std::result::Result<(), Error> {
        self.session.active().map_err(session_error)?.zoom_out();
        Ok(())
    }

    fn move_by(&mut self, dx: i32, dy: i32) -> std::result::Result<(), Error> {
        let dx = i16::try_from(dx).map_err(|_| Error::Other(format!("dx {dx} out of range")))?;
        let dy = i16::try_from(dy).map_err(|_| Error::Other(format!("dy {dy} out of range")))?;
        self.session
            .active()
            .map_err(session_error)?
            .move_by(dx, dy);
        Ok(())
    }

    fn handle_click(&mut self, col: i32, row: i32) -> std::result::Result<bool, Error> {
        let col =
            u16::try_from(col).map_err(|_| Error::Other(format!("col {col} out of range")))?;
        let row =
            u16::try_from(row).map_err(|_| Error::Other(format!("row {row} out of range")))?;
        Ok(self
            .session
            .active()
            .map_err(session_error)?
            .handle_click(col, row))
    }

    /// Move the viewport to the base `frac_x` of the way along the block `node_id`
    /// `seq_start..seq_end`, snapping it left unless `center` is set. The zoom level stays as
    /// the user set it.
    fn go_to_match(
        &mut self,
        node_id: String,
        seq_start: i64,
        seq_end: i64,
        frac_x: f64,
        center: bool,
    ) -> std::result::Result<(), Error> {
        let node_id = hash_id_from_string(&node_id).map_err(Error::Other)?;
        let width = (seq_end - seq_start).max(1);
        let offset = ((width as f64) * frac_x.clamp(0.0, 1.0)) as i64;
        let coordinate = seq_start + offset.min(width - 1);
        self.session
            .active()
            .map_err(session_error)?
            .go_to_coordinate(node_id, coordinate, center)
            .map_err(|error| Error::Other(format!("cannot go to position: {error}")))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the arguments of the R highlight_match wrapper"
    )]
    fn highlight_match(
        &mut self,
        color: String,
        start_offset: i64,
        end_offset: i64,
        strand: String,
        node_ids: Vec<String>,
        seq_starts: Vec<i32>,
        seq_ends: Vec<i32>,
    ) -> std::result::Result<(), Error> {
        if node_ids.len() != seq_starts.len() || seq_starts.len() != seq_ends.len() {
            return Err(Error::Other(
                "node_ids, seq_starts, and seq_ends must have the same length".to_string(),
            ));
        }
        if node_ids.is_empty() {
            return Err(Error::Other(
                "highlight_match requires at least one block".to_string(),
            ));
        }
        if start_offset < 0 || end_offset < 0 {
            return Err(Error::Other(
                "start_offset and end_offset must be non-negative".to_string(),
            ));
        }
        let strand = match strand.as_str() {
            "f" => Strand::Forward,
            "r" => Strand::Reverse,
            _ => Strand::Unknown,
        };
        let block_count = node_ids.len();
        let mut slices = Vec::with_capacity(block_count);
        for index in 0..block_count {
            let node_id = hash_id_from_string(&node_ids[index]).map_err(Error::Other)?;
            let block = GraphNode {
                node_id,
                sequence_start: i64::from(seq_starts[index]),
                sequence_end: i64::from(seq_ends[index]),
            };
            let slice_start = if index == 0 { start_offset as usize } else { 0 };
            let slice_end = if index == block_count - 1 {
                end_offset as usize
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
        let span = annotation_span_from_graph_locus(&GraphLocus { slices }, "");
        self.session
            .active()
            .and_then(|page| page.push_adhoc_highlight(span, Some(&color)))
            .map_err(session_error)
    }

    /// Remove match highlights, keeping named tracks and the selected path.
    fn clear_highlights(&mut self) -> std::result::Result<(), Error> {
        self.session
            .active()
            .map_err(session_error)?
            .clear_highlights();
        Ok(())
    }

    fn add_track_group(&mut self, group: String) -> std::result::Result<(), Error> {
        self.session
            .active()
            .and_then(|page| page.add_track_group(&group))
            .map_err(session_error)
    }

    fn add_track_file(
        &mut self,
        path: String,
        name: Nullable<String>,
        sample: Nullable<String>,
    ) -> std::result::Result<(), Error> {
        let name = nullable_string_to_option(name);
        let sample = nullable_string_to_option(sample);
        let display_name = name.unwrap_or_else(|| path.clone());
        let page = self.session.active().map_err(session_error)?;
        let block_group_id = page
            .block_group_id()
            .ok_or_else(|| Error::Other("controller has no block group".to_string()))?;
        let node_filter = page.controller.loaded_node_ids();
        let mut database = page.controller.database_mut().clone();
        let conn = database
            .connection()
            .map_err(|e| Error::Other(e.to_string()))?;
        let workspace = workspace_for_connection(conn).map_err(session_error)?;
        let track = load_annotation_file_as_track(
            conn,
            &workspace,
            &block_group_id,
            &path,
            &display_name,
            sample.as_deref(),
            &node_filter,
        );
        page.push_track(&display_name, track.annotations);
        Ok(())
    }

    /// Show every annotation group in the automatic theme palette.
    fn trigger_auto_load(&mut self) -> std::result::Result<(), Error> {
        self.session
            .active()
            .and_then(|page| page.trigger_auto_load())
            .map_err(session_error)
    }

    /// Show every annotation group, coloring annotations from a JSON object mapping annotation
    /// id hex strings to a `#rrggbb` color or `null` to hide that annotation.
    fn load_annotation_groups_with_colors_json(
        &mut self,
        color_map_json: String,
    ) -> std::result::Result<(), Error> {
        let color_overrides: HashMap<String, Option<String>> =
            json_from_str(&color_map_json).unwrap_or_default();
        self.session
            .active()
            .and_then(|page| page.load_annotation_groups_with_colors(&color_overrides))
            .map_err(session_error)
    }

    /// Render the graph alone to PNG bytes, cropped to its content, with `scale` image pixels
    /// per logical pixel.
    fn render_png(&mut self, cols: i32, rows: i32, scale: i32) -> std::result::Result<Raw, Error> {
        self.render_png_bytes(cols, rows, scale, false)
    }

    /// Render the full interactive viewport to PNG bytes, keeping the header and every cell at
    /// its position so pointer coordinates map straight to cells.
    fn render_png_viewport(
        &mut self,
        cols: i32,
        rows: i32,
        scale: i32,
    ) -> std::result::Result<Raw, Error> {
        self.render_png_bytes(cols, rows, scale, true)
    }

    /// The pixel size of one terminal cell in a PNG rendered at `scale`, as `c(width, height)`.
    fn png_cell_size(&self, scale: i32) -> std::result::Result<Vec<i32>, Error> {
        let scale = Self::scale(scale)?;
        let (width, height) = GraphSession::png_cell_size();
        Ok(vec![(width * scale) as i32, (height * scale) as i32])
    }

    /// The canvas color of the PNG renderer, as `#rrggbb`.
    fn background_color(&self) -> String {
        GraphSession::png_background_color()
    }

    /// The clickable cells of the last rendered viewport, as a JSON array of
    /// `{"x", "y", "kind"}` with `kind` either `"node"` or `"wormhole"`.
    fn hit_regions(&mut self, cols: i32, rows: i32) -> std::result::Result<String, Error> {
        let (cols, rows) = Self::viewport(cols, rows)?;
        self.session
            .hit_regions_json(cols, rows)
            .map_err(session_error)
    }

    fn clone_controller(&self) -> Self {
        self.clone()
    }
}

impl RGraphController {
    fn scale(scale: i32) -> std::result::Result<usize, Error> {
        let scale = usize::try_from(scale)
            .map_err(|_| Error::Other(format!("scale {scale} out of range")))?;
        validated_png_scale(scale).map_err(session_error)
    }

    fn viewport(cols: i32, rows: i32) -> std::result::Result<(u16, u16), Error> {
        let cols =
            u16::try_from(cols).map_err(|_| Error::Other(format!("cols {cols} out of range")))?;
        let rows =
            u16::try_from(rows).map_err(|_| Error::Other(format!("rows {rows} out of range")))?;
        Ok((cols, rows))
    }

    /// The PNG bytes of the current view, kept apart from the R vector so tests need no R runtime.
    fn png_bytes(
        &mut self,
        cols: i32,
        rows: i32,
        scale: i32,
        full_viewport: bool,
    ) -> std::result::Result<Vec<u8>, Error> {
        let scale = Self::scale(scale)?;
        let size = Self::viewport(cols, rows)?;
        self.session
            .render_png(size, full_viewport, scale)
            .map_err(session_error)
    }

    fn render_png_bytes(
        &mut self,
        cols: i32,
        rows: i32,
        scale: i32,
        full_viewport: bool,
    ) -> std::result::Result<Raw, Error> {
        Ok(Raw::from_bytes(&self.png_bytes(
            cols,
            rows,
            scale,
            full_viewport,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader, path::PathBuf};

    use r#gen::{
        imports::genbank::{GenBankImportOptions, import_genbank},
        test_helpers::{TestContext, setup_block_group, setup_gen_on_disk},
        views::{
            annotation_groups::{AnnotationGroupEntry, AnnotationGroupOrigin},
            annotations::{AnnotationGroupTrackRequest, load_annotations_for_group},
        },
    };
    use gen_core::{is_end_node, is_start_node};
    use gen_models::annotations::AnnotationGroup;

    use super::*;

    /// The fixture context owns the temporary workspace, so it must outlive the repository.
    fn linear_repo() -> (TestContext, Repository, String) {
        let context = setup_gen_on_disk();
        let (bg_id, _path) = setup_block_group(context.graph().conn());
        let repo = Repository {
            context: (*context).clone(),
        };
        (context, repo, bg_id.to_string())
    }

    /// Sets up puc19 with real GenBank annotations imported, and returns the
    /// repository, its block group id, the annotation group name, the resolved
    /// spans for that group (for color-override tests, keyed by `span.id`), and one
    /// real data `GraphNode` from the controller's own graph (for `highlight_match`
    /// calls -- an `AnnotationSegment`'s node_id/start/end is not guaranteed to be an
    /// exact `GraphNode` in the render graph's `GraphMap`, so building a match
    /// highlight from span coordinates directly panics in `apply_cell_highlight`'s
    /// `to_index` lookup).
    fn puc19_repo() -> (
        TestContext,
        Repository,
        String,
        String,
        Vec<AnnotationSpan>,
        GraphNode,
    ) {
        let context = setup_gen_on_disk();
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures/puc19.gb");
        {
            let file = File::open(&path).unwrap();
            import_genbank(
                &context,
                BufReader::new(file),
                Some("fixtures"),
                "puc19-sample",
                OperationInfo {
                    files: vec![
                        OperationFile::new(path.to_str().unwrap().to_string())
                            .set_file_type(gen_models::file_types::FileTypes::GenBank),
                    ],
                    description: "test".to_string(),
                },
                GenBankImportOptions::default().annotation_name_from_path(&path),
            )
            .unwrap();
        }
        let conn = context.graph().conn();
        let block_groups = Sample::get_block_groups(conn, "fixtures", "puc19-sample", None);
        let block_group = &block_groups[0];
        let bg_id = block_group.id;
        let group_name = AnnotationGroup::query_by_sample(conn, "puc19-sample", None)
            .into_iter()
            .find(|group| group.name != "source")
            .expect("puc19 fixture should import at least one non-source annotation group")
            .name;

        let graph = BlockGroup::get_graph(conn, context.workspace(), &bg_id, None).unwrap();
        let node_ids: HashSet<HashId> = graph
            .nodes()
            .filter(|n| !is_start_node(n.node_id) && !is_end_node(n.node_id))
            .map(|n| n.node_id)
            .collect();
        let entry = AnnotationGroupEntry {
            id: group_name.clone(),
            name: group_name.clone(),
            sample_name: "puc19-sample".to_string(),
            source_block_group_id: bg_id,
            origin: AnnotationGroupOrigin::CurrentSample,
        };
        let spans = load_annotations_for_group(&AnnotationGroupTrackRequest {
            conn,
            history_ref: None,
            entry: &entry,
            projection_graph: &graph,
            node_ids: &node_ids,
        })
        .unwrap();
        assert!(
            !spans.is_empty(),
            "puc19 fixture's non-source group should resolve at least one span"
        );
        let highlight_node = graph
            .nodes()
            .find(|n| !is_start_node(n.node_id) && !is_end_node(n.node_id))
            .expect("puc19 graph should contain at least one data node");

        let repo = Repository {
            context: (*context).clone(),
        };
        (
            context,
            repo,
            bg_id.to_string(),
            group_name,
            spans,
            highlight_node,
        )
    }

    const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

    fn png_width(bytes: &[u8]) -> u32 {
        assert_eq!(&bytes[..8], SIGNATURE, "should be a PNG");
        u32::from_be_bytes(bytes[16..20].try_into().expect("should read the PNG width"))
    }

    fn viewport(controller: &mut RGraphController) -> Vec<u8> {
        controller
            .png_bytes(60, 10, 1, true)
            .expect("should render the viewport")
    }

    // `RGraphController` characterization: pan/zoom/click/detail baselines, the
    // ordering-sensitive scenarios where a mutating call has no render in between,
    // and the two behavior fixes the persistent design makes over the old ops-replay
    // path (clear_highlights keeps tracks; color overrides stick).

    #[test]
    fn test_rgraph_controller_zoom_then_move_with_no_render_between() {
        let (_context, repo, id) = linear_repo();
        let mut controller = repo
            .create_graph_controller(id, "minimal".to_string())
            .unwrap();
        controller.zoom_in().unwrap();
        controller.move_by(3, 0).unwrap();
        assert!(png_width(&viewport(&mut controller)) > 0);
    }

    #[test]
    fn test_rgraph_controller_zoom_then_click_with_no_render_between() {
        let (_context, repo, id) = linear_repo();
        let mut controller = repo
            .create_graph_controller(id, "minimal".to_string())
            .unwrap();
        controller.zoom_in().unwrap();
        let _ = controller.handle_click(5, 5).unwrap();
        assert!(png_width(&viewport(&mut controller)) > 0);
    }

    #[test]
    fn test_rgraph_controller_baseline_pan_zoom_detail() {
        let (_context, repo, id) = linear_repo();

        let mut minimal = repo
            .create_graph_controller(id.clone(), "minimal".to_string())
            .unwrap();
        let minimal_baseline = viewport(&mut minimal);
        minimal.zoom_in().unwrap();
        assert_ne!(
            minimal_baseline,
            viewport(&mut minimal),
            "zoom_in should change the rendered view"
        );

        let mut normal = repo
            .create_graph_controller(id.clone(), "normal".to_string())
            .unwrap();
        let baseline = viewport(&mut normal);
        normal.move_by(3, 0).unwrap();
        assert_ne!(
            baseline,
            viewport(&mut normal),
            "pan should change the rendered view"
        );

        let mut full = repo
            .create_graph_controller(id, "full".to_string())
            .unwrap();
        assert_ne!(
            minimal_baseline,
            viewport(&mut full),
            "detail change should change the rendered view"
        );
    }

    #[test]
    fn test_rgraph_controller_scale_doubles_the_png() {
        let (_context, repo, id) = linear_repo();
        let mut controller = repo
            .create_graph_controller(id, "normal".to_string())
            .unwrap();

        let single = controller.png_bytes(60, 10, 1, true).unwrap();
        let doubled = controller.png_bytes(60, 10, 2, true).unwrap();

        assert_eq!(png_width(&doubled), 2 * png_width(&single));
        let logical = controller.png_cell_size(1).unwrap();
        assert_eq!(
            controller.png_cell_size(2).unwrap(),
            vec![2 * logical[0], 2 * logical[1]]
        );
        assert!(controller.png_bytes(60, 10, 0, true).is_err());
    }

    #[test]
    fn test_rgraph_controller_hit_regions_find_nodes() {
        let (_context, repo, id) = linear_repo();
        let mut controller = repo
            .create_graph_controller(id, "normal".to_string())
            .unwrap();
        viewport(&mut controller);

        let regions = controller.hit_regions(60, 10).unwrap();

        assert!(regions.contains("\"kind\":\"node\""));
    }

    #[test]
    fn test_rgraph_controller_clear_highlights_preserves_tracks() {
        let (_context, repo, id, group_name, _spans, highlight_node) = puc19_repo();
        let mut controller = repo
            .create_graph_controller(id, "full".to_string())
            .unwrap();
        controller.add_track_group(group_name).unwrap();
        let render =
            |controller: &mut RGraphController| controller.png_bytes(200, 20, 1, true).unwrap();
        let track_only = render(&mut controller);

        controller
            .highlight_match(
                "red".to_string(),
                0,
                highlight_node.length(),
                "f".to_string(),
                vec![highlight_node.node_id.to_string()],
                vec![highlight_node.sequence_start as i32],
                vec![highlight_node.sequence_end as i32],
            )
            .unwrap();
        assert_ne!(
            track_only,
            render(&mut controller),
            "a match highlight should change the view"
        );

        controller.clear_highlights().unwrap();
        assert_eq!(
            track_only,
            render(&mut controller),
            "clear_highlights should remove the match highlight but keep tracks intact"
        );
    }

    #[test]
    fn test_rgraph_controller_color_override_takes_effect() {
        let (_context, repo, id, _group_name, spans, _highlight_node) = puc19_repo();
        let mut default_controller = repo
            .create_graph_controller(id.clone(), "full".to_string())
            .unwrap();
        default_controller.trigger_auto_load().unwrap();
        let default_colors = default_controller.png_bytes(200, 20, 1, true).unwrap();

        let mut overridden_controller = repo
            .create_graph_controller(id, "full".to_string())
            .unwrap();
        let color_map_json = format!(
            "{{{}}}",
            spans
                .iter()
                .map(|span| format!("\"{}\":\"#00ff00\"", span.id))
                .collect::<Vec<_>>()
                .join(",")
        );
        overridden_controller
            .load_annotation_groups_with_colors_json(color_map_json)
            .unwrap();
        let overridden_colors = overridden_controller.png_bytes(200, 20, 1, true).unwrap();

        assert_ne!(
            default_colors, overridden_colors,
            "an id -> color override should change the rendered annotation color"
        );
    }

    #[test]
    fn test_rgraph_controller_clone_is_independent() {
        let (_context, repo, id) = linear_repo();
        let mut original = repo
            .create_graph_controller(id, "normal".to_string())
            .unwrap();
        let original_baseline = viewport(&mut original);

        let mut clone = original.clone_controller();
        clone.move_by(3, 0).unwrap();
        assert_ne!(
            original_baseline,
            viewport(&mut clone),
            "the clone should reflect its own move"
        );

        assert_eq!(
            original_baseline,
            viewport(&mut original),
            "mutating a clone must not affect the original controller"
        );
    }
}

extendr_module! {
    mod genr;
    impl Repository;
    impl SequenceGraph;
    impl RGraphController;
}
