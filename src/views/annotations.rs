//! Annotation track loading resolves local files first and materializes remote indexes and data
//! only when the selected workflow requires them. A remote tabix index is downloaded in full,
//! cached under `.gen/cache/annotation-indexes`, and parsed before the annotation is queried. With
//! an index and viewport, the annotation is read through OpenDAL range requests; if ranged access
//! fails, the complete file is cached and the query is retried locally. Remote annotations without
//! a usable index are cached in full before parsing. Cached compressed bytes remain compressed and
//! are decoded only at the parser boundary, while downloads are streamed through checksum
//! verification and atomically published for later reuse.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::{self, File},
    io::{self, BufRead, BufReader, Cursor, Read, Seek, Write},
    path::{Path as FsPath, PathBuf},
};

use flate2::read::MultiGzDecoder;
use gen_annotations::{
    projection as annotation_projection,
    translate::{bed::translate_bed, gff::translate_gff},
};
use gen_core::{HashId, Strand, Workspace, is_terminal};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    accession::Accession,
    annotations::{Annotation, AnnotationError},
    assets::{AssetUri, ChecksummedReader, LocalAssetUri},
    block_group::BlockGroup,
    db::GraphConnection,
    file_types::FileTypes,
    reference_alias::ReferenceAlias,
    traits::Query,
};
use noodles::{bed, core::Region, csi, gff, tabix};
use petgraph::Direction;
use tempfile::NamedTempFile;

use crate::views::{
    annotation_files::{AnnotationAssetEntry, AnnotationFileEntry},
    annotation_groups::AnnotationGroupEntry,
    annotation_track::{AnnotationSegment, AnnotationSpan, AnnotationTrack},
};

pub struct AnnotationGroupTrackRequest<'a> {
    pub conn: &'a GraphConnection,
    pub history_ref: Option<&'a str>,
    pub current_block_group: &'a BlockGroup,
    pub entry: &'a AnnotationGroupEntry,
    /// Node IDs to restrict results to — typically the viewport's visible nodes.
    pub node_ids: &'a HashSet<HashId>,
}

pub fn load_annotations_for_group(
    request: &AnnotationGroupTrackRequest<'_>,
) -> Result<Vec<AnnotationSpan>, AnnotationError> {
    load_group_annotations(
        request.conn,
        request.current_block_group,
        request.entry,
        request.node_ids,
        request.history_ref,
    )
}

/// Clip `accession_segments` (stored in per-node absolute sequence coordinates, from
/// whenever the annotation was created) onto every node currently present in the block
/// group's full graph — every branch, not just one arbitrarily selected path. Unlike
/// `gen_annotations::projection::project_annotation_segments`, coordinates stay in
/// per-node absolute space rather than being shifted into path space, matching what
/// `AnnotationSegment`/`graph_locus_from_annotation_span` expect elsewhere in the TUI and
/// Jupyter viewers.
///
/// A `node_id` is stable across later edits, so scanning every node for one whose
/// `node_id` overlaps a stored segment already produces one clipped segment per
/// surviving fragment, with no segment at all over whatever unrelated sequence (e.g. a
/// library insertion) was spliced into the gap. Scanning the whole graph rather than a
/// single selected path additionally means an annotation still shows up on every branch
/// that carries a fragment of it — e.g. every candidate in a combinatorial library, not
/// only whichever one happens to be "the" current path.
fn clip_segments_to_graph(
    accession_segments: &[annotation_projection::AnnotationSegment],
    graph: &GenGraph,
) -> Vec<AnnotationSegment> {
    let mut clipped = Vec::new();
    for node in graph.nodes() {
        for segment in accession_segments {
            if segment.node_id != node.node_id {
                continue;
            }
            let overlap_start = segment.range.start.max(node.sequence_start);
            let overlap_end = segment.range.end.min(node.sequence_end);
            if overlap_end <= overlap_start {
                continue;
            }
            clipped.push(AnnotationSegment {
                node_id: node.node_id,
                start: overlap_start,
                end: overlap_end,
                strand: segment.strand,
            });
        }
    }
    clipped
}

/// True if `segments` cover the block group graph end to end. Root/leaf are resolved past
/// any zero-length PATH_START_NODE_ID/PATH_END_NODE_ID sentinel to the real content node.
fn spans_whole_block_group(segments: &[AnnotationSegment], graph: &GenGraph) -> bool {
    let mut root = None;
    let mut leaf = None;
    for node in graph.nodes() {
        if graph
            .neighbors_directed(node, Direction::Incoming)
            .next()
            .is_none()
        {
            root = Some(node);
        }
        if graph
            .neighbors_directed(node, Direction::Outgoing)
            .next()
            .is_none()
        {
            leaf = Some(node);
        }
    }

    let boundary_nodes = |node: GraphNode, direction: Direction| -> Vec<GraphNode> {
        if is_terminal(node.node_id) {
            graph.neighbors_directed(node, direction).collect()
        } else {
            vec![node]
        }
    };

    let starts_at_root = root.is_some_and(|root| {
        boundary_nodes(root, Direction::Outgoing)
            .into_iter()
            .any(|boundary| {
                segments.iter().any(|segment| {
                    segment.node_id == boundary.node_id && segment.start <= boundary.sequence_start
                })
            })
    });
    let ends_at_leaf = leaf.is_some_and(|leaf| {
        boundary_nodes(leaf, Direction::Incoming)
            .into_iter()
            .any(|boundary| {
                segments.iter().any(|segment| {
                    segment.node_id == boundary.node_id && segment.end >= boundary.sequence_end
                })
            })
    });
    starts_at_root && ends_at_leaf
}

fn load_group_annotations(
    conn: &GraphConnection,
    current_block_group: &BlockGroup,
    entry: &AnnotationGroupEntry,
    node_ids: &HashSet<HashId>,
    history_ref: Option<&str>,
) -> Result<Vec<AnnotationSpan>, AnnotationError> {
    // The entry identifies the block group that owns the annotations. The current
    // selection below is only the graph onto which those annotations are projected.
    let annotations = Annotation::query_by_group_and_block_group(
        conn,
        &entry.name,
        &entry.source_block_group_id,
        history_ref,
    )?;

    // Annotations are stored against the node(s) they were created on, which may since
    // have been split by later edits (a library insertion, a sequence update, ...), or may
    // live on a branch other than whichever one is "the" current path (e.g. one option in
    // a combinatorial library). Clip them onto the block group's full graph so an
    // annotation still covers every surviving fragment of its original range, on every
    // branch, with a gap wherever an edit spliced in unrelated sequence.
    let graph = BlockGroup::get_graph(conn, &current_block_group.id, history_ref)
        .unwrap_or_else(|_| GenGraph::new());
    Ok(annotations
        .into_iter()
        .filter_map(|annotation| {
            let _ = Accession::get_by_id(conn, &annotation.accession_id, history_ref)?;
            let accession_segments =
                annotation_projection::annotation_segments(conn, &annotation, history_ref);
            let clipped_segments = clip_segments_to_graph(&accession_segments, &graph);
            let full_segments = if clipped_segments.is_empty() {
                accession_segments
                    .iter()
                    .map(|segment| AnnotationSegment {
                        node_id: segment.node_id,
                        start: segment.range.start,
                        end: segment.range.end,
                        strand: segment.strand,
                    })
                    .collect()
            } else {
                clipped_segments
            };
            if spans_whole_block_group(&full_segments, &graph) {
                return None;
            }
            let segments = full_segments
                .into_iter()
                .filter(|segment| node_ids.contains(&segment.node_id))
                .collect::<Vec<_>>();
            if segments.is_empty() {
                None
            } else {
                Some(AnnotationSpan {
                    id: annotation.id,
                    name: annotation.name,
                    segments,
                })
            }
        })
        .collect())
}

fn gff_attribute_value_to_string(
    attrs: &gff::feature::record_buf::Attributes,
    key: &str,
) -> Option<String> {
    let key_bytes = key.as_bytes();
    attrs.as_ref().iter().find_map(|(tag, value)| {
        let tag_bytes: &[u8] = tag.as_ref();
        if !tag_bytes.eq_ignore_ascii_case(key_bytes) {
            return None;
        }
        if let Some(value) = value.as_string() {
            Some(String::from_utf8_lossy(value.as_ref()).to_string())
        } else {
            value
                .iter()
                .next()
                .map(|item| String::from_utf8_lossy(item.as_ref()).to_string())
        }
    })
}

fn build_annotation_spans(
    track_label: &str,
    segments_by_name: HashMap<String, Vec<AnnotationSegment>>,
) -> Vec<AnnotationSpan> {
    segments_by_name
        .into_iter()
        .map(|(name, segments)| AnnotationSpan {
            id: HashId::convert_str(&format!("{track_label}:{name}")),
            name,
            segments,
        })
        .collect()
}

/// Parse a translated GFF3 file from a filesystem path.
pub fn parse_translated_gff_file(
    path: &std::path::Path,
    node_filter: &HashSet<HashId>,
    track_label: &str,
) -> Result<Vec<AnnotationSpan>, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    Ok(parse_translated_gff(
        reader,
        node_filter,
        track_label,
        HashMap::new(),
    ))
}

/// Parse a translated BED file from a filesystem path.
pub fn parse_translated_bed_file(
    path: &std::path::Path,
    node_filter: &HashSet<HashId>,
    track_label: &str,
) -> Result<Vec<AnnotationSpan>, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    Ok(parse_translated_bed(
        reader,
        node_filter,
        track_label,
        HashMap::new(),
    ))
}

pub fn parse_translated_gff<R: BufRead>(
    reader: R,
    node_filter: &HashSet<HashId>,
    track_label: &str,
    references_by_alias: HashMap<String, String>,
) -> Vec<AnnotationSpan> {
    let mut segments_by_name: HashMap<String, Vec<AnnotationSegment>> = HashMap::new();
    let mut reader = gff::io::Reader::new(reader);
    for result in reader.record_bufs() {
        let record = match result {
            Ok(record) => record,
            Err(_) => continue,
        };
        let ref_name = record.reference_sequence_name();
        let ref_name = references_by_alias
            .get(&ref_name.to_string())
            .unwrap_or(&ref_name.to_string())
            .to_string();
        let node_id = match HashId::try_from(ref_name) {
            Ok(id) => id,
            Err(_) => continue,
        };
        if !node_filter.contains(&node_id) {
            continue;
        }
        let start = record.start().get() as i64;
        let end = record.end().get() as i64;
        if end <= 0 {
            continue;
        }
        let start = start.saturating_sub(1);
        let (seg_start, seg_end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let attrs = record.attributes();
        let name = gff_attribute_value_to_string(attrs, "Name")
            .or_else(|| gff_attribute_value_to_string(attrs, "ID"))
            .or_else(|| gff_attribute_value_to_string(attrs, "gene"))
            .or_else(|| gff_attribute_value_to_string(attrs, "db_xref"))
            .unwrap_or_else(|| record.ty().to_string());
        segments_by_name
            .entry(name)
            .or_default()
            .push(AnnotationSegment {
                node_id,
                start: seg_start,
                end: seg_end,
                strand: record.strand().into(),
            });
    }
    build_annotation_spans(track_label, segments_by_name)
}

pub fn parse_translated_bed<R: BufRead>(
    reader: R,
    node_filter: &HashSet<HashId>,
    track_label: &str,
    references_by_alias: HashMap<String, String>,
) -> Vec<AnnotationSpan> {
    let mut segments_by_name: HashMap<String, Vec<AnnotationSegment>> = HashMap::new();
    let mut bed_reader = bed::io::reader::Builder::<6>.build_from_reader(reader);
    let mut record = bed::Record::<6>::default();
    while let Ok(read) = bed_reader.read_record(&mut record) {
        if read == 0 {
            break;
        }
        let ref_name = String::from_utf8_lossy(record.reference_sequence_name().as_ref());
        let ref_name = references_by_alias
            .get(&ref_name.to_string())
            .unwrap_or(&ref_name.to_string())
            .to_string();
        let node_id = match HashId::try_from(ref_name.to_string()) {
            Ok(id) => id,
            Err(_) => continue,
        };
        if !node_filter.contains(&node_id) {
            continue;
        }
        let start = match record.feature_start() {
            Ok(pos) => pos.get() as i64,
            Err(_) => continue,
        };
        let end = match record.feature_end() {
            Some(Ok(pos)) => pos.get() as i64,
            _ => continue,
        };
        if end <= 0 {
            continue;
        }
        let start = start.saturating_sub(1);
        let (seg_start, seg_end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let name = record
            .name()
            .and_then(|value| std::str::from_utf8(value.as_ref()).ok())
            .filter(|value| !value.is_empty())
            .unwrap_or("feature")
            .to_string();
        let strand = match record.strand() {
            Ok(Some(bed::feature::record::Strand::Forward)) => Strand::Forward,
            Ok(Some(bed::feature::record::Strand::Reverse)) => Strand::Reverse,
            Ok(None) | Err(_) => Strand::Unknown,
        };
        segments_by_name
            .entry(name)
            .or_default()
            .push(AnnotationSegment {
                node_id,
                start: seg_start,
                end: seg_end,
                strand,
            });
    }
    build_annotation_spans(track_label, segments_by_name)
}

/// Finds an annotation or explicit index sidecar that should already exist in the workspace.
///
/// Track loading calls this first for the annotation itself, and index loading reuses it for a
/// configured local index. Remote URIs deliberately return `None` so the caller can choose between
/// ranged access and the remote cache instead of treating the URI as a filesystem path.
fn resolve_local_annotation_file_path(
    workspace: &Workspace,
    file_addition: &AnnotationAssetEntry,
) -> Option<PathBuf> {
    if !LocalAssetUri::is_local_path_or_file_uri(&file_addition.asset_uri) {
        return None;
    }
    if let Ok(repo_root) = workspace.repo_root() {
        let repo_path = repo_root.join(file_addition.file_path());
        if repo_path.exists() {
            return Some(repo_path);
        }
    }
    let hashed_filename = file_addition.hashed_filename()?;
    let asset_path = workspace.asset_dir().ok()?.join(hashed_filename);
    if asset_path.exists() {
        return Some(asset_path);
    }
    None
}

/// Builds the stable filename used when a remote annotation or index is cached locally.
///
/// Cache path construction calls this for both data and index files. A known content checksum is
/// the preferred identity; checksumless assets use the URI for stable reuse, and retaining the
/// suffix lets the parser recognize compressed annotation files after they are cached.
fn remote_cache_filename(file_addition: &AnnotationAssetEntry) -> String {
    let cache_key = file_addition
        .checksum
        .map(|checksum| checksum.to_string())
        .unwrap_or_else(|| format!("uri-{}", HashId::convert_str(&file_addition.asset_uri)));
    let suffix = <dyn AssetUri>::from_uri(&file_addition.asset_uri)
        .suffix()
        .unwrap_or_default();
    if suffix.is_empty() {
        cache_key
    } else {
        format!("{cache_key}.{suffix}")
    }
}

/// Resolves the cache destination before reading or downloading a remote annotation asset.
///
/// Index loading uses the `annotation-indexes` subdirectory, while annotation loading uses
/// `annotations`. Keeping those namespaces separate lets both files retain their original suffix
/// and prevents an index from being mistaken for annotation data.
fn remote_annotation_cache_path(
    workspace: &Workspace,
    file_addition: &AnnotationAssetEntry,
    cache_subdirectory: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let cache_dir = workspace.ensure_cache_dir()?.join(cache_subdirectory);
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir.join(remote_cache_filename(file_addition)))
}

/// Materializes an entire remote annotation or index when the workflow needs a local file.
///
/// This is called unconditionally for remote indexes, for unindexed remote annotations, and as the
/// fallback when an indexed remote annotation cannot satisfy ranged reads. Existing cache entries
/// are reused; new downloads are checksummed while streaming and published atomically only after
/// the complete file has arrived.
fn cache_remote_annotation_asset(
    workspace: &Workspace,
    file_addition: &AnnotationAssetEntry,
    cache_subdirectory: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    if LocalAssetUri::is_local_path_or_file_uri(&file_addition.asset_uri) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "local annotation assets must not be copied into the remote cache",
        )
        .into());
    }

    let cache_path = remote_annotation_cache_path(workspace, file_addition, cache_subdirectory)?;
    if cache_path.exists() {
        return Ok(cache_path);
    }
    let cache_dir = cache_path
        .parent()
        .expect("should have cache parent directory");

    let asset_uri = <dyn AssetUri>::from_uri(&file_addition.asset_uri);
    let mut reader = ChecksummedReader::new(asset_uri.reader(workspace)?);
    let checksum_handle = reader.checksum_handle();
    let mut cache_file = NamedTempFile::new_in(cache_dir)?;
    io::copy(&mut reader, &mut cache_file)?;
    cache_file.flush()?;
    if let Some(expected_checksum) = file_addition.checksum
        && checksum_handle.checksum() != Some(expected_checksum)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "remote annotation asset checksum mismatch: {}",
                file_addition.asset_uri
            ),
        )
        .into());
    }

    match cache_file.persist_noclobber(&cache_path) {
        Ok(_) => Ok(cache_path),
        Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => Ok(cache_path),
        Err(error) => Err(error.error.into()),
    }
}

/// Finds the conventional sibling index for a local annotation without an explicit index asset.
///
/// Index loading calls this only after the primary annotation has resolved to a local path. An
/// explicit index entry, including a remote index URI, takes the other branch of that workflow.
fn tabix_index_path(file_path: &FsPath) -> PathBuf {
    let mut index_path = file_path.to_path_buf();
    index_path.set_extension(format!(
        "{}.tbi",
        file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
    ));
    if index_path.exists() {
        return index_path;
    }
    PathBuf::from(format!("{}.tbi", file_path.display()))
}

/// Reports whether track loading should use the explicitly recorded index as a tabix index.
///
/// Other index file types do not enter the tabix query workflow and leave local sibling discovery
/// to `load_annotation_index`.
fn annotation_index_is_tabix(entry: &AnnotationFileEntry) -> bool {
    entry
        .index_file_addition
        .as_ref()
        .is_some_and(|index_file_addition| index_file_addition.file_type == FileTypes::Tabix)
}

/// Loads the tabix index before track loading decides how to access the annotation data.
///
/// A configured remote index is downloaded in full and cached, while a configured local index is
/// resolved from the workspace. Without an explicit tabix asset, local annotations are checked for
/// a conventional sibling index. The resulting BGZF-compressed index is parsed into memory before
/// any local or remote annotation query begins.
fn load_annotation_index(
    workspace: &Workspace,
    entry: &AnnotationFileEntry,
    annotation_path: Option<&FsPath>,
) -> Result<Option<tabix::Index>, Box<dyn Error>> {
    let index_path = if annotation_index_is_tabix(entry) {
        let index_file_addition = entry
            .index_file_addition
            .as_ref()
            .expect("should have index file addition when index type is tabix");
        if LocalAssetUri::is_local_path_or_file_uri(&index_file_addition.asset_uri) {
            resolve_local_annotation_file_path(workspace, index_file_addition)
                .ok_or("Annotation index file not found in repo or assets")?
        } else {
            cache_remote_annotation_asset(workspace, index_file_addition, "annotation-indexes")?
        }
    } else {
        let Some(annotation_path) = annotation_path else {
            return Ok(None);
        };
        let index_path = tabix_index_path(annotation_path);
        if !index_path.exists() {
            return Ok(None);
        }
        index_path
    };
    Ok(Some(tabix::fs::read(index_path)?))
}

/// Executes a tabix query after the workflow has selected a seekable annotation source.
///
/// Local files, cached files, and OpenDAL remote readers all enter here as raw BGZF streams. The
/// noodles indexed reader applies the parsed index's virtual offsets and decompresses only the
/// blocks needed to return records intersecting the requested window.
fn load_tabix_region_bytes<R>(
    reader: R,
    index: tabix::Index,
    reference_name: &str,
    window: (i64, i64),
) -> Result<Vec<u8>, Box<dyn Error>>
where
    R: Read + Seek,
{
    let start = (window.0 + 1).max(1);
    let end = window.1.max(start);
    let region = format!("{reference_name}:{start}-{end}").parse::<Region>()?;

    let mut reader = csi::io::IndexedReader::new(reader, index);
    let query = reader.query(&region)?;

    let mut bytes = Vec::new();
    for result in query {
        let record = result?;
        bytes.extend_from_slice(record.as_ref().as_bytes());
        bytes.push(b'\n');
    }

    Ok(bytes)
}

/// Chooses how to obtain annotation records once an index and query window are available.
///
/// Local and previously cached files are queried directly. A remote uncached annotation is first
/// queried through its seekable OpenDAL reader so range-capable storage transfers only the needed
/// blocks; if that query fails, the complete file is cached and the same query is retried locally.
fn load_indexed_annotation_bytes(
    workspace: &Workspace,
    file_addition: &AnnotationAssetEntry,
    local_file_path: Option<&FsPath>,
    index: tabix::Index,
    reference_name: &str,
    window: (i64, i64),
) -> Result<Vec<u8>, Box<dyn Error>> {
    if let Some(path) = local_file_path {
        return load_tabix_region_bytes(File::open(path)?, index, reference_name, window);
    }
    if LocalAssetUri::is_local_path_or_file_uri(&file_addition.asset_uri) {
        return Err("Annotation file not found in repo or assets".into());
    }

    let cache_path = remote_annotation_cache_path(workspace, file_addition, "annotations")?;
    if cache_path.exists() {
        return load_tabix_region_bytes(File::open(cache_path)?, index, reference_name, window);
    }

    // The indexed query itself is the range-support check. This tests the actual read behavior
    // instead of trusting an optional Accept-Ranges header.
    let asset_uri = <dyn AssetUri>::from_uri(&file_addition.asset_uri);
    let ranged_result = asset_uri
        .reader(workspace)
        .map_err(|error| Box::new(error) as Box<dyn Error>)
        .and_then(|reader| load_tabix_region_bytes(reader, index.clone(), reference_name, window));
    match ranged_result {
        Ok(bytes) => Ok(bytes),
        Err(_) => {
            let cache_path =
                cache_remote_annotation_asset(workspace, file_addition, "annotations")?;
            load_tabix_region_bytes(File::open(cache_path)?, index, reference_name, window)
        }
    }
}

/// Adapts the selected annotation representation for translation and parsing.
///
/// Track loading calls this after indexed access has produced in-memory record bytes or after a
/// complete local/cached file has been selected. Indexed bytes are already decompressed text;
/// complete `.gz` and `.bgz` files are decompressed here at the parser boundary.
fn annotation_reader<'a>(
    file_path: Option<&FsPath>,
    indexed_bytes: Option<&'a [u8]>,
) -> Result<Box<dyn BufRead + 'a>, Box<dyn Error>> {
    if let Some(bytes) = indexed_bytes {
        return Ok(Box::new(BufReader::new(Cursor::new(bytes))));
    }
    let file_path = file_path.ok_or("Annotation file not found in repo or assets")?;
    let file = File::open(file_path)?;
    match file_path
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("gz" | "bgz") => Ok(Box::new(BufReader::new(MultiGzDecoder::new(file)))),
        _ => Ok(Box::new(BufReader::new(file))),
    }
}

pub struct AnnotationFileTrackLoadResult {
    pub track: AnnotationTrack,
    pub index_available: bool,
    pub loaded_window: Option<(i64, i64)>,
}

pub struct AnnotationFileTrackRequest<'a> {
    pub conn: &'a GraphConnection,
    pub history_ref: Option<&'a str>,
    pub workspace: &'a Workspace,
    pub collection_name: &'a str,
    pub sample_name: &'a str,
    // Used as the tabix reference/contig name. translate_gff/translate_bed re-project the file's
    // reference-relative coordinates onto each sample's own current path
    // (BlockGroup::get_current_path), so edits/indels within a lineage don't break the mapping.
    // Matching purely by name is only ambiguous for unrelated samples that happen to share a
    // contig name with no common ancestry (e.g. two independently-imported genomes both naming a
    // contig "chr1"). File-based annotations are linear-space by nature, so the robust fix is to
    // anchor a file to the node_id of its original reference sequence rather than to a name:
    // node identity already survives the whole derivation lineage for free (copy_contents_from
    // rebinds existing edges/nodes rather than duplicating them), so "is this node present in the
    // current block group" answers applicability correctly without needing a sample_lineage walk
    // or relying on lineage having been explicitly registered.
    pub block_group_name: Option<&'a str>,
    pub query_window: Option<(i64, i64)>,
    pub node_filter: &'a HashSet<HashId>,
    pub entry: &'a AnnotationFileEntry,
}

/// Loads an annotation track from the cheapest available local or remote representation.
///
/// The workflow first resolves local data and loads any tabix index, caching a remote index when
/// necessary. An indexed track without a reference or viewport stops after index discovery. With
/// a complete query window, the loader attempts an indexed local, cached, or ranged remote read;
/// without an index it materializes remote data before parsing. Both paths finish by adapting the
/// selected bytes or file into the same translation and graph-projection pipeline.
pub fn load_annotation_file_track(
    request: &AnnotationFileTrackRequest<'_>,
) -> Result<AnnotationFileTrackLoadResult, Box<dyn Error>> {
    let mut file_path =
        resolve_local_annotation_file_path(request.workspace, &request.entry.file_addition);
    let index = load_annotation_index(request.workspace, request.entry, file_path.as_deref())
        .ok()
        .flatten();
    let index_available = index.is_some();

    if index_available && (request.block_group_name.is_none() || request.query_window.is_none()) {
        return Ok(AnnotationFileTrackLoadResult {
            track: AnnotationTrack::new(request.entry.display_name.clone(), Vec::new()),
            index_available,
            loaded_window: None,
        });
    }

    let (indexed_source_bytes, loaded_window) = if let Some(index) = index {
        let reference_name = request
            .block_group_name
            .expect("indexed annotation should have a reference name");
        let window = request
            .query_window
            .expect("indexed annotation should have a query window");
        let bytes = load_indexed_annotation_bytes(
            request.workspace,
            &request.entry.file_addition,
            file_path.as_deref(),
            index,
            reference_name,
            window,
        )?;
        (Some(bytes), Some(window))
    } else {
        if file_path.is_none() {
            if LocalAssetUri::is_local_path_or_file_uri(&request.entry.file_addition.asset_uri) {
                return Err("Annotation file not found in repo or assets".into());
            }
            file_path = Some(cache_remote_annotation_asset(
                request.workspace,
                &request.entry.file_addition,
                "annotations",
            )?);
        }
        (None, None)
    };

    let mut buffer = Vec::new();
    let reader = annotation_reader(file_path.as_deref(), indexed_source_bytes.as_deref())?;
    match request.entry.file_addition.file_type {
        FileTypes::Gff3 => translate_gff(
            request.conn,
            request.collection_name,
            request.sample_name,
            request.history_ref,
            reader,
            &mut buffer,
        )?,
        FileTypes::Bed => translate_bed(
            request.conn,
            request.collection_name,
            request.sample_name,
            request.history_ref,
            reader,
            &mut buffer,
        )?,
        other => {
            return Err(format!("Unsupported annotation file type: {other:?}").into());
        }
    }

    let references_by_alias = ReferenceAlias::get_references_by_alias(
        request.conn,
        vec![request.block_group_name.unwrap_or_default().to_string()],
        request.history_ref,
    )?;
    let spans = match request.entry.file_addition.file_type {
        FileTypes::Gff3 => {
            if buffer.is_empty() {
                let reader =
                    annotation_reader(file_path.as_deref(), indexed_source_bytes.as_deref())?;
                parse_translated_gff(
                    reader,
                    request.node_filter,
                    &request.entry.display_name,
                    references_by_alias,
                )
            } else {
                parse_translated_gff(
                    Cursor::new(buffer),
                    request.node_filter,
                    &request.entry.display_name,
                    references_by_alias,
                )
            }
        }
        FileTypes::Bed => {
            if buffer.is_empty() {
                let reader =
                    annotation_reader(file_path.as_deref(), indexed_source_bytes.as_deref())?;
                parse_translated_bed(
                    reader,
                    request.node_filter,
                    &request.entry.display_name,
                    references_by_alias,
                )
            } else {
                parse_translated_bed(
                    Cursor::new(buffer),
                    request.node_filter,
                    &request.entry.display_name,
                    references_by_alias,
                )
            }
        }
        _ => Vec::new(),
    };
    Ok(AnnotationFileTrackLoadResult {
        track: AnnotationTrack::new(request.entry.display_name.clone(), spans),
        index_available,
        loaded_window,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs,
        io::{Cursor, Read as _, Write as _},
        net::TcpListener,
        path::PathBuf,
        thread,
        time::{Duration, Instant},
    };

    use flate2::{Compression, write::GzEncoder};
    use gen_core::{HashId, Sha256Hash, Strand};
    use gen_graph::{GenGraph, GraphNode};
    use gen_models::{
        annotations::add_annotation, block_group::BlockGroup, file_types::FileTypes, sample::Sample,
    };
    use noodles::tabix;

    use super::{
        AnnotationFileTrackRequest, AnnotationGroupTrackRequest, AnnotationSegment,
        annotation_index_is_tabix, load_annotation_file_track, load_annotations_for_group,
        load_indexed_annotation_bytes, parse_translated_bed, remote_annotation_cache_path,
        spans_whole_block_group,
    };
    use crate::{
        graphs::combinatorial_library::parse_library,
        imports::fasta::import_fasta,
        test_helpers::setup_gen,
        updates::{library::update_with_library, sequence::update_with_sequence},
        views::{
            annotation_files::{AnnotationAssetEntry, AnnotationFileEntry},
            annotation_groups::load_annotation_group_entries,
        },
    };

    fn serve_http_content(
        contents: &[u8],
        expected_get_count: usize,
    ) -> (String, thread::JoinHandle<()>) {
        let contents = contents.to_vec();
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("should bind test HTTP listener");
        listener
            .set_nonblocking(true)
            .expect("should configure test HTTP listener");
        let address = listener.local_addr().expect("should read listener address");
        let handle = thread::spawn(move || {
            let started = Instant::now();
            let mut served_get_count = 0;
            while served_get_count < expected_get_count
                && started.elapsed() < Duration::from_secs(5)
            {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                };
                stream
                    .set_nonblocking(false)
                    .expect("should configure test HTTP stream");
                let mut request = [0; 1024];
                let length = stream.read(&mut request).expect("should read request");
                let request = String::from_utf8_lossy(&request[..length]);
                if request.starts_with("GET ") {
                    served_get_count += 1;
                    let response = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        contents.len()
                    )
                    .and_then(|()| stream.write_all(&contents));
                    if let Err(error) = response {
                        assert!(
                            matches!(
                                error.kind(),
                                std::io::ErrorKind::BrokenPipe
                                    | std::io::ErrorKind::ConnectionReset
                            ),
                            "should only fail when the range probe disconnects: {error}"
                        );
                    }
                } else {
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        contents.len()
                    )
                    .expect("should write metadata response");
                }
            }
            assert_eq!(served_get_count, expected_get_count);
        });
        (format!("http://{address}/asset"), handle)
    }

    #[test]
    fn parse_translated_bed_preserves_strand() {
        let node_id = HashId::convert_str("node-1");
        let bed = format!("{node_id}\t5\t8\tfeature-a\t0\t-\n");
        let spans = parse_translated_bed(
            Cursor::new(bed),
            &HashSet::from([node_id]),
            "bed",
            HashMap::new(),
        );

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "feature-a");
        assert_eq!(spans[0].segments.len(), 1);
        assert_eq!(spans[0].segments[0].node_id, node_id);
        assert_eq!(spans[0].segments[0].start, 5);
        assert_eq!(spans[0].segments[0].end, 8);
        assert_eq!(spans[0].segments[0].strand, Strand::Reverse);
    }

    /// A node that gets split by a later edit (e.g. a library insertion in the middle)
    /// should keep every surviving fragment of an annotation that used to cover the
    /// whole thing, with a gap over whatever unrelated sequence was spliced in.
    #[test]
    fn clip_segments_to_graph_leaves_a_gap_over_an_inserted_block() {
        use gen_graph::{GenGraph, GraphNode};

        use super::{annotation_projection, clip_segments_to_graph};

        let original_node_id = HashId::convert_str("original");
        let inserted_node_id = HashId::convert_str("inserted");

        // The annotation was created before the split, covering the whole original node.
        let accession_segments = vec![annotation_projection::AnnotationSegment {
            node_id: original_node_id,
            range: gen_core::range::Range {
                start: 0,
                end: 2686,
            },
            strand: Strand::Forward,
        }];

        // The current graph has since been split: left fragment of the original node,
        // an unrelated inserted block, then the right fragment of the original node.
        let mut graph = GenGraph::new();
        graph.add_node(GraphNode {
            node_id: original_node_id,
            sequence_start: 0,
            sequence_end: 395,
        });
        graph.add_node(GraphNode {
            node_id: inserted_node_id,
            sequence_start: 0,
            sequence_end: 50,
        });
        graph.add_node(GraphNode {
            node_id: original_node_id,
            sequence_start: 483,
            sequence_end: 2686,
        });

        let mut segments = clip_segments_to_graph(&accession_segments, &graph);
        segments.sort_by_key(|s| s.start);

        assert_eq!(segments.len(), 2, "no segment over the inserted block");
        assert_eq!(segments[0].node_id, original_node_id);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 395);
        assert_eq!(segments[1].node_id, original_node_id);
        assert_eq!(segments[1].start, 483);
        assert_eq!(segments[1].end, 2686);
    }

    /// pUC19's `source 1..2686` feature spans the whole plasmid and should be hidden.
    #[test]
    fn load_annotations_for_group_hides_puc19_whole_plasmid_source_annotation() {
        use std::{fs::File, io::BufReader, path::PathBuf};

        use gen_models::{
            block_group::BlockGroup,
            file_types::FileTypes,
            operations::{OperationFile, OperationInfo},
            sample::Sample,
        };

        use super::{AnnotationGroupTrackRequest, load_annotations_for_group};
        use crate::{
            imports::genbank::{GenBankImportOptions, import_genbank},
            test_helpers::setup_gen,
            views::annotation_groups::{AnnotationGroupEntry, AnnotationGroupOrigin},
        };

        let context = setup_gen();
        let conn = context.graph().conn();

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/puc19.gb");
        let file = File::open(&path).unwrap();
        import_genbank(
            &context,
            BufReader::new(file),
            Some("fixtures"),
            "puc19-sample",
            OperationInfo {
                files: vec![
                    OperationFile::new(path.to_str().unwrap().to_string())
                        .set_file_type(FileTypes::GenBank),
                ],
                description: "test".to_string(),
            },
            GenBankImportOptions::default().annotation_name_from_path(&path),
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, "fixtures", "puc19-sample", None);
        let block_group = &block_groups[0];
        let graph = BlockGroup::get_graph(conn, &block_group.id, None).unwrap();
        let node_ids: HashSet<HashId> = graph.nodes().map(|n| n.node_id).collect();

        let groups =
            gen_models::annotations::AnnotationGroup::query_by_sample(conn, "puc19-sample", None);
        let entry = AnnotationGroupEntry {
            id: groups[0].name.clone(),
            name: groups[0].name.clone(),
            sample_name: "puc19-sample".to_string(),
            source_block_group_id: block_group.id,
            origin: AnnotationGroupOrigin::CurrentSample,
        };

        let spans = load_annotations_for_group(&AnnotationGroupTrackRequest {
            conn,
            history_ref: None,
            current_block_group: block_group,
            entry: &entry,
            node_ids: &node_ids,
        })
        .unwrap();

        let names: Vec<_> = spans.iter().map(|s| s.name.as_str()).collect();
        assert!(
            !names.contains(&"source"),
            "expected the whole-plasmid `source` feature to be hidden, got {names:?}"
        );
        assert!(
            names.contains(&"AmpR"),
            "real features should still resolve, got {names:?}"
        );
    }

    /// Each option in a combinatorial library gets its own annotation on its own graph
    /// branch. `load_annotations_for_group` must resolve every one of them — not just
    /// whichever candidate happens to end up on the block group's current path — since
    /// every candidate is a real, visible branch in the graph.
    #[test]
    fn load_annotations_for_group_finds_every_combinatorial_branch() {
        use std::path::PathBuf;

        use gen_models::{block_group::BlockGroup, sample::Sample};

        use super::{AnnotationGroupTrackRequest, load_annotations_for_group};
        use crate::{
            graphs::combinatorial_library::parse_library,
            imports::library::import_library,
            test_helpers::setup_gen,
            views::annotation_groups::{AnnotationGroupEntry, AnnotationGroupOrigin},
        };

        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test";
        let library_name = "m123";

        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");
        let parts_list =
            parse_library(parts_path.to_str().unwrap(), library_path.to_str().unwrap()).unwrap();

        import_library(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            library_name,
            parts_list,
            Some(parts_path.to_str().unwrap()),
            Some(library_path.to_str().unwrap()),
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME, None);
        let block_group = &block_groups[0];

        let graph = BlockGroup::get_graph(conn, &block_group.id, None).unwrap();
        let node_ids: HashSet<HashId> = graph.nodes().map(|n| n.node_id).collect();

        let entry = AnnotationGroupEntry {
            id: library_name.to_string(),
            name: library_name.to_string(),
            sample_name: Sample::DEFAULT_NAME.to_string(),
            source_block_group_id: block_group.id,
            origin: AnnotationGroupOrigin::CurrentSample,
        };

        let spans = load_annotations_for_group(&AnnotationGroupTrackRequest {
            conn,
            history_ref: None,
            current_block_group: block_group,
            entry: &entry,
            node_ids: &node_ids,
        })
        .unwrap();

        let mut names: Vec<_> = spans.iter().map(|s| s.name.as_str()).collect();
        names.sort();
        assert_eq!(
            names,
            ["cds1", "cds2", "cds3", "p1", "p2", "p3"],
            "every combinatorial candidate's annotation should resolve, not just the one \
             on the current path"
        );
    }

    #[test]
    fn test_load_annotations_for_group_uses_entry_source_block_group_id() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let collection = "test".to_string();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let parts_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let library_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");
        let fasta_path = fasta_path.to_str().unwrap().to_string();
        let parts_path = parts_path.to_str().unwrap().to_string();
        let library_path = library_path.to_str().unwrap().to_string();

        import_fasta(
            &context,
            &fasta_path,
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();
        add_annotation(
            &context,
            &collection,
            "SITE",
            None,
            Sample::DEFAULT_NAME,
            "m123:7-20",
        )
        .unwrap();
        update_with_library(
            &context,
            &collection,
            Sample::DEFAULT_NAME,
            "design",
            "SITE",
            parse_library(&parts_path, &library_path).unwrap(),
            Some(&parts_path),
            Some(&library_path),
        )
        .unwrap();
        update_with_sequence(
            &context,
            &collection,
            "design",
            "deleted",
            "cds1:0-1",
            "",
            false,
        )
        .unwrap();

        let selected_block_group = Sample::get_block_groups(conn, &collection, "deleted", None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should contain the deleted m123 block group");
        let entry = load_annotation_group_entries(conn, &selected_block_group, None)
            .into_iter()
            .find(|entry| entry.name == "design" && entry.sample_name == "design")
            .expect("should list the design annotation group entry");
        assert_ne!(
            entry.source_block_group_id, selected_block_group.id,
            "the entry source should differ from the currently selected block group"
        );
        let selected_graph = BlockGroup::get_graph(conn, &selected_block_group.id, None).unwrap();
        let node_ids = selected_graph
            .nodes()
            .map(|node| node.node_id)
            .collect::<HashSet<_>>();

        let spans = load_annotations_for_group(&AnnotationGroupTrackRequest {
            conn,
            history_ref: None,
            current_block_group: &selected_block_group,
            entry: &entry,
            node_ids: &node_ids,
        })
        .unwrap();
        let mut names = spans
            .iter()
            .map(|span| span.name.as_str())
            .collect::<Vec<_>>();
        names.sort_unstable();

        assert_eq!(
            names,
            ["cds1", "cds2", "cds3", "p1", "p2", "p3"],
            "annotations should be queried with the entry source ID and projected onto the selected graph"
        );
    }

    /// A full-length annotation between sentinel-wrapped root/leaf should be hidden.
    #[test]
    fn spans_whole_block_group_hides_full_length_annotation_on_single_node_graph() {
        use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID};

        let node_id = HashId::convert_str("only-node");
        let start = GraphNode {
            node_id: PATH_START_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let content = GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 100,
        };
        let end = GraphNode {
            node_id: PATH_END_NODE_ID,
            sequence_start: 0,
            sequence_end: 0,
        };
        let mut graph = GenGraph::new();
        graph.add_edge(start, content, Vec::new());
        graph.add_edge(content, end, Vec::new());

        let segments = vec![AnnotationSegment {
            node_id,
            start: 0,
            end: 100,
            strand: Strand::Forward,
        }];
        assert!(spans_whole_block_group(&segments, &graph));
    }

    #[test]
    fn test_annotation_index_is_tabix_only_for_tabix_sidecars() {
        let annotation_entry = AnnotationFileEntry {
            file_addition: AnnotationAssetEntry {
                id: HashId::convert_str("annotation"),
                asset_uri: "file:///tmp/annotation.gff3".to_string(),
                file_type: FileTypes::Gff3,
                checksum: Some(Sha256Hash::convert_str("annotation-checksum")),
            },
            index_file_addition: Some(AnnotationAssetEntry {
                id: HashId::convert_str("index"),
                asset_uri: "file:///tmp/annotation.csi".to_string(),
                file_type: FileTypes::None,
                checksum: Some(Sha256Hash::convert_str("index-checksum")),
            }),
            name: None,
            display_name: "annotation".to_string(),
        };
        assert!(!annotation_index_is_tabix(&annotation_entry));

        let tabix_entry = AnnotationFileEntry {
            index_file_addition: Some(AnnotationAssetEntry {
                file_type: FileTypes::Tabix,
                ..annotation_entry
                    .index_file_addition
                    .clone()
                    .expect("should have index sidecar")
            }),
            ..annotation_entry
        };
        assert!(annotation_index_is_tabix(&tabix_entry));
    }

    #[test]
    fn test_indexed_remote_annotation_downloads_index_without_query_window() {
        let context = setup_gen();
        let index_contents = include_bytes!("../../fixtures/chr22_100k_no_samples.vcf.gz.tbi");
        let (index_uri, server) = serve_http_content(index_contents, 1);
        let index_file_addition = AnnotationAssetEntry {
            id: HashId::convert_str("remote-index"),
            asset_uri: index_uri,
            file_type: FileTypes::Tabix,
            checksum: None,
        };
        let entry = AnnotationFileEntry {
            file_addition: AnnotationAssetEntry {
                id: HashId::convert_str("remote-annotation"),
                asset_uri: "http://127.0.0.1:1/annotation.gff3.gz".to_string(),
                file_type: FileTypes::Gff3,
                checksum: None,
            },
            index_file_addition: Some(index_file_addition.clone()),
            name: Some("remote-track".to_string()),
            display_name: "remote-track".to_string(),
        };
        let node_filter = HashSet::new();
        let request = AnnotationFileTrackRequest {
            conn: context.graph().conn(),
            history_ref: None,
            workspace: context.workspace(),
            collection_name: "collection",
            sample_name: "sample",
            block_group_name: None,
            query_window: None,
            node_filter: &node_filter,
            entry: &entry,
        };

        let result = load_annotation_file_track(&request)
            .expect("metadata-only indexed track load should cache the index");
        server.join().expect("should finish HTTP server");
        let cached_result =
            load_annotation_file_track(&request).expect("should reuse the cached index");

        assert!(result.index_available);
        assert!(cached_result.index_available);
        assert!(result.track.annotations.is_empty());
        assert_eq!(result.loaded_window, None);
        let cache_path = remote_annotation_cache_path(
            context.workspace(),
            &index_file_addition,
            "annotation-indexes",
        )
        .expect("should resolve annotation index cache path");
        assert_eq!(
            fs::read(cache_path).expect("should read cached annotation index"),
            index_contents
        );
    }

    #[test]
    fn test_unindexed_remote_annotation_is_cached_and_reused() {
        let context = setup_gen();
        let contents = b"##gff-version 3\n";
        let (asset_uri, server) = serve_http_content(contents, 1);
        let entry = AnnotationFileEntry {
            file_addition: AnnotationAssetEntry {
                id: HashId::convert_str("unindexed-remote-annotation"),
                asset_uri,
                file_type: FileTypes::Gff3,
                checksum: None,
            },
            index_file_addition: None,
            name: None,
            display_name: "remote-track".to_string(),
        };
        let node_filter = HashSet::new();
        let request = AnnotationFileTrackRequest {
            conn: context.graph().conn(),
            history_ref: None,
            workspace: context.workspace(),
            collection_name: "collection",
            sample_name: "sample",
            block_group_name: None,
            query_window: None,
            node_filter: &node_filter,
            entry: &entry,
        };

        let first_result =
            load_annotation_file_track(&request).expect("should load remote annotation");
        server.join().expect("should finish HTTP server");
        let second_result =
            load_annotation_file_track(&request).expect("should reuse cached remote annotation");

        assert!(!first_result.index_available);
        assert!(first_result.track.annotations.is_empty());
        assert!(second_result.track.annotations.is_empty());
        let cache_dir = context
            .workspace()
            .ensure_cache_dir()
            .expect("should resolve cache directory")
            .join("annotations");
        let cached_paths = fs::read_dir(&cache_dir)
            .expect("should read annotation cache")
            .map(|entry| entry.expect("should read cached entry").path())
            .collect::<Vec<_>>();
        assert_eq!(cached_paths.len(), 1);
        assert_eq!(
            fs::read(&cached_paths[0]).expect("should read cached annotation"),
            contents
        );
    }

    #[test]
    fn test_unindexed_remote_gzip_annotation_is_decompressed_from_cache() {
        let context = setup_gen();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(b"##gff-version 3\n")
            .expect("should compress annotation");
        let compressed = encoder
            .finish()
            .expect("should finish compressed annotation");
        let (asset_uri, server) = serve_http_content(&compressed, 1);
        let file_addition = AnnotationAssetEntry {
            id: HashId::convert_str("unindexed-remote-gzip-annotation"),
            asset_uri: format!("{asset_uri}.gff3.gz"),
            file_type: FileTypes::Gff3,
            checksum: None,
        };
        let entry = AnnotationFileEntry {
            file_addition: file_addition.clone(),
            index_file_addition: None,
            name: None,
            display_name: "remote-gzip-track".to_string(),
        };
        let node_filter = HashSet::new();
        let request = AnnotationFileTrackRequest {
            conn: context.graph().conn(),
            history_ref: None,
            workspace: context.workspace(),
            collection_name: "collection",
            sample_name: "sample",
            block_group_name: None,
            query_window: None,
            node_filter: &node_filter,
            entry: &entry,
        };

        let result =
            load_annotation_file_track(&request).expect("should decode remote gzip annotation");
        server.join().expect("should finish HTTP server");
        let cached_result =
            load_annotation_file_track(&request).expect("should decode cached gzip annotation");

        assert!(result.track.annotations.is_empty());
        assert!(cached_result.track.annotations.is_empty());
        let cache_path =
            remote_annotation_cache_path(context.workspace(), &file_addition, "annotations")
                .expect("should resolve annotation cache path");
        assert_eq!(
            fs::read(cache_path).expect("should read compressed annotation cache"),
            compressed
        );
    }

    #[test]
    fn test_indexed_remote_annotation_without_range_support_uses_cache_fallback() {
        let context = setup_gen();
        let contents = include_bytes!("../../fixtures/chr22_100k_no_samples.vcf.gz");
        let (asset_uri, server) = serve_http_content(contents, 3);
        let file_addition = AnnotationAssetEntry {
            id: HashId::convert_str("remote-indexed-annotation"),
            asset_uri,
            file_type: FileTypes::Gff3,
            checksum: None,
        };
        let index_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("chr22_100k_no_samples.vcf.gz.tbi");

        let bytes = load_indexed_annotation_bytes(
            context.workspace(),
            &file_addition,
            None,
            tabix::fs::read(&index_path).expect("should read tabix index"),
            "chr22",
            (16_050_074, 16_050_120),
        )
        .expect("should retry the indexed query from a cached file");
        server.join().expect("should finish HTTP server");

        let records = String::from_utf8(bytes).expect("should return text records");
        assert!(records.contains("chr22\t16050075"));
        assert!(records.contains("chr22\t16050115"));

        let cache_path =
            remote_annotation_cache_path(context.workspace(), &file_addition, "annotations")
                .expect("should resolve annotation cache path");
        assert_eq!(
            fs::read(&cache_path).expect("should read cached annotation"),
            contents
        );

        let cached_bytes = load_indexed_annotation_bytes(
            context.workspace(),
            &file_addition,
            None,
            tabix::fs::read(index_path).expect("should reread tabix index"),
            "chr22",
            (16_050_074, 16_050_120),
        )
        .expect("should reuse the cached file");
        assert!(!cached_bytes.is_empty());
    }
}
