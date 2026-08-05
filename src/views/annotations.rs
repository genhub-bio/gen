use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::{Path as FsPath, PathBuf},
};

use gen_annotations::{
    projection as annotation_projection,
    translate::{bed::translate_bed, gff::translate_gff},
};
use gen_core::{HashId, Strand, Workspace, is_terminal};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{
    accession::Accession,
    annotations::{Annotation, AnnotationError},
    block_group::BlockGroup,
    db::GraphConnection,
    file_types::FileTypes,
    reference_alias::ReferenceAlias,
    traits::Query,
};
use noodles::{bed, core::Region, gff, tabix};
use petgraph::Direction;

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

fn resolve_annotation_file_path(
    workspace: &Workspace,
    file_addition: &AnnotationAssetEntry,
) -> Option<PathBuf> {
    if let Ok(repo_root) = workspace.repo_root() {
        let repo_path = repo_root.join(file_addition.file_path());
        if repo_path.exists() {
            return Some(repo_path);
        }
    }
    let asset_path = workspace
        .asset_dir()
        .ok()?
        .join(file_addition.hashed_filename()?);
    if asset_path.exists() {
        return Some(asset_path);
    }
    None
}

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

fn annotation_index_is_tabix(entry: &AnnotationFileEntry) -> bool {
    entry
        .index_file_addition
        .as_ref()
        .is_some_and(|index_file_addition| index_file_addition.file_type == FileTypes::Tabix)
}

fn resolve_annotation_index_file_path(
    workspace: &Workspace,
    entry: &AnnotationFileEntry,
    file_path: &FsPath,
) -> Option<PathBuf> {
    if annotation_index_is_tabix(entry) {
        let index_file_addition = entry
            .index_file_addition
            .as_ref()
            .expect("should have index file addition when index type is tabix");
        return resolve_annotation_file_path(workspace, index_file_addition);
    }
    let index_path = tabix_index_path(file_path);
    if index_path.exists() {
        Some(index_path)
    } else {
        None
    }
}

fn load_tabix_region_bytes(
    file_path: &FsPath,
    index_path: Option<&FsPath>,
    reference_name: &str,
    window: (i64, i64),
) -> Result<Vec<u8>, Box<dyn Error>> {
    let start = (window.0 + 1).max(1);
    let end = window.1.max(start);
    let region = format!("{reference_name}:{start}-{end}").parse::<Region>()?;

    let mut builder = tabix::io::indexed_reader::Builder::default();
    if let Some(index_path) = index_path {
        builder = builder.set_index(tabix::fs::read(index_path)?);
    }
    let mut reader = builder.build_from_path(file_path)?;
    let query = reader.query(&region)?;

    let mut bytes = Vec::new();
    for result in query {
        let record = result?;
        bytes.extend_from_slice(record.as_ref().as_bytes());
        bytes.push(b'\n');
    }

    Ok(bytes)
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

pub fn load_annotation_file_track(
    request: &AnnotationFileTrackRequest<'_>,
) -> Result<AnnotationFileTrackLoadResult, Box<dyn Error>> {
    let file_path = resolve_annotation_file_path(request.workspace, &request.entry.file_addition)
        .ok_or("Annotation file not found in repo or assets")?;
    let index_path =
        resolve_annotation_index_file_path(request.workspace, request.entry, &file_path);
    let index_available = index_path.is_some();
    let mut indexed_source_bytes = None;
    let mut loaded_window = None;

    if index_available {
        if let (Some(reference_name), Some(window)) =
            (request.block_group_name, request.query_window)
        {
            indexed_source_bytes = Some(load_tabix_region_bytes(
                &file_path,
                index_path.as_deref(),
                reference_name,
                window,
            )?);
            loaded_window = Some(window);
        } else {
            return Ok(AnnotationFileTrackLoadResult {
                track: AnnotationTrack::new(request.entry.display_name.clone(), Vec::new()),
                index_available,
                loaded_window: None,
            });
        }
    }

    let mut buffer = Vec::new();
    match request.entry.file_addition.file_type {
        FileTypes::Gff3 => {
            if let Some(bytes) = indexed_source_bytes.as_deref() {
                translate_gff(
                    request.conn,
                    request.collection_name,
                    request.sample_name,
                    request.history_ref,
                    BufReader::new(Cursor::new(bytes)),
                    &mut buffer,
                )?;
            } else {
                translate_gff(
                    request.conn,
                    request.collection_name,
                    request.sample_name,
                    request.history_ref,
                    BufReader::new(File::open(&file_path)?),
                    &mut buffer,
                )?;
            }
        }
        FileTypes::Bed => {
            if let Some(bytes) = indexed_source_bytes.as_deref() {
                translate_bed(
                    request.conn,
                    request.collection_name,
                    request.sample_name,
                    request.history_ref,
                    Cursor::new(bytes),
                    &mut buffer,
                )?;
            } else {
                translate_bed(
                    request.conn,
                    request.collection_name,
                    request.sample_name,
                    request.history_ref,
                    File::open(&file_path)?,
                    &mut buffer,
                )?;
            }
        }
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
                let reader: Box<dyn BufRead> = if let Some(bytes) = indexed_source_bytes.as_deref()
                {
                    Box::new(BufReader::new(Cursor::new(bytes)))
                } else {
                    Box::new(BufReader::new(File::open(&file_path)?))
                };
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
                let reader: Box<dyn BufRead> = if let Some(bytes) = indexed_source_bytes.as_deref()
                {
                    Box::new(BufReader::new(Cursor::new(bytes)))
                } else {
                    Box::new(BufReader::new(File::open(&file_path)?))
                };
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
        io::Cursor,
        path::PathBuf,
    };

    use gen_core::{HashId, Strand};
    use gen_graph::{GenGraph, GraphNode};
    use gen_models::{
        annotations::add_annotation, block_group::BlockGroup, file_types::FileTypes, sample::Sample,
    };

    use super::{
        AnnotationGroupTrackRequest, AnnotationSegment, annotation_index_is_tabix,
        load_annotations_for_group, parse_translated_bed, spans_whole_block_group,
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
                checksum: Some(gen_core::Sha256Hash::convert_str("annotation-checksum")),
            },
            index_file_addition: Some(AnnotationAssetEntry {
                id: HashId::convert_str("index"),
                asset_uri: "file:///tmp/annotation.csi".to_string(),
                file_type: FileTypes::None,
                checksum: Some(gen_core::Sha256Hash::convert_str("index-checksum")),
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
}
