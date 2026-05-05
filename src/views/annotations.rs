use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::{Path as FsPath, PathBuf},
};

use gen_annotations::translate::{bed::translate_bed, gff::translate_gff};
use gen_core::{HashId, Strand, Workspace, is_end_node, is_start_node};
use gen_models::{
    accession::{Accession, AccessionEdge},
    annotations::{Annotation, AnnotationError},
    block_group::BlockGroup,
    db::GraphConnection,
    file_types::FileTypes,
    operations::FileAddition,
    reference_alias::ReferenceAlias,
    traits::Query,
};
use noodles::{bed, core::Region, gff, tabix};

use crate::views::{
    annotation_files::AnnotationFileEntry,
    annotation_groups::{AnnotationGroupEntry, AnnotationGroupOrigin},
    annotation_track::{AnnotationSegment, AnnotationSpan, AnnotationTrack},
};

fn accession_edges_to_segments(edges: &[AccessionEdge]) -> Vec<AnnotationSegment> {
    let mut segments = Vec::new();
    let mut current_node: Option<HashId> = None;
    let mut current_start: Option<i64> = None;
    let mut current_strand: Option<Strand> = None;

    for edge in edges {
        if is_start_node(edge.source_node_id) {
            current_node = Some(edge.target_node_id);
            current_start = Some(edge.target_coordinate);
            current_strand = Some(edge.target_strand);
            continue;
        }

        if is_end_node(edge.target_node_id) {
            if let (Some(node_id), Some(start), Some(strand)) =
                (current_node, current_start, current_strand)
            {
                let (segment_start, segment_end) = if start <= edge.source_coordinate {
                    (start, edge.source_coordinate)
                } else {
                    (edge.source_coordinate, start)
                };
                segments.push(AnnotationSegment {
                    node_id,
                    start: segment_start,
                    end: segment_end,
                    strand,
                });
            }
            break;
        }

        if let (Some(node_id), Some(start), Some(strand)) =
            (current_node, current_start, current_strand)
        {
            let (segment_start, segment_end) = if start <= edge.source_coordinate {
                (start, edge.source_coordinate)
            } else {
                (edge.source_coordinate, start)
            };
            segments.push(AnnotationSegment {
                node_id,
                start: segment_start,
                end: segment_end,
                strand,
            });
        }

        current_node = Some(edge.target_node_id);
        current_start = Some(edge.target_coordinate);
        current_strand = Some(edge.target_strand);
    }

    segments
}

pub struct AnnotationGroupTrackRequest<'a> {
    pub conn: &'a GraphConnection,
    pub current_block_group: &'a BlockGroup,
    pub entry: &'a AnnotationGroupEntry,
    pub visible_ranges_by_node: &'a HashMap<HashId, Vec<(i64, i64)>>,
}

pub fn load_annotations_for_group(
    request: &AnnotationGroupTrackRequest<'_>,
) -> Result<Vec<AnnotationSpan>, AnnotationError> {
    let _ = request.current_block_group;
    load_group_annotations(request.conn, request.entry, request.visible_ranges_by_node)
}

fn load_group_annotations(
    conn: &GraphConnection,
    entry: &AnnotationGroupEntry,
    visible_ranges_by_node: &HashMap<HashId, Vec<(i64, i64)>>,
) -> Result<Vec<AnnotationSpan>, AnnotationError> {
    let annotations = Annotation::query_by_group(conn, &entry.name)?;
    let inherited = entry.origin != AnnotationGroupOrigin::CurrentSample;

    Ok(annotations
        .into_iter()
        .filter_map(|annotation| {
            let _ = Accession::get_by_id(conn, &annotation.accession_id)?;
            let edges = Accession::get_edges_by_id(conn, &annotation.accession_id);
            let segments = accession_edges_to_segments(&edges)
                .into_iter()
                .filter(|segment| {
                    visible_ranges_by_node
                        .get(&segment.node_id)
                        .is_some_and(|ranges| {
                            ranges
                                .iter()
                                .any(|(start, end)| segment.start < *end && *start < segment.end)
                        })
                })
                .collect::<Vec<_>>();
            if segments.is_empty() {
                None
            } else {
                Some(AnnotationSpan {
                    id: annotation.id,
                    name: if inherited {
                        format!("{} [{}]", annotation.name, entry.sample_name)
                    } else {
                        annotation.name
                    },
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

fn parse_translated_gff<R: BufRead>(
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

fn parse_translated_bed<R: BufRead>(
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
    file_addition: &FileAddition,
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
        .join(file_addition.clone().hashed_filename());
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

fn resolve_annotation_index_file_path(
    workspace: &Workspace,
    entry: &AnnotationFileEntry,
    file_path: &FsPath,
) -> Option<PathBuf> {
    if let Some(index_file_addition) = entry.index_file_addition.as_ref() {
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
    pub workspace: &'a Workspace,
    pub collection_name: &'a str,
    pub sample_name: &'a str,
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
                    BufReader::new(Cursor::new(bytes)),
                    &mut buffer,
                )?;
            } else {
                translate_gff(
                    request.conn,
                    request.collection_name,
                    request.sample_name,
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
                    Cursor::new(bytes),
                    &mut buffer,
                )?;
            } else {
                translate_bed(
                    request.conn,
                    request.collection_name,
                    request.sample_name,
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
    };

    use gen_core::{HashId, Strand};

    use super::parse_translated_bed;

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
}
