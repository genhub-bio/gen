use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
    thread,
};

use gai::{IndexedSource, MatchMode};
use gen_annotations::{
    AnnotationTranslationContext, translate_bed_annotation, translate_gff_annotation,
};
use gen_core::{HashId, NodeIntervalBlock, Sha256Hash, Strand, Workspace, region::Region};
use gen_graph::GenGraph;
#[cfg(test)]
use gen_models::{annotations::add_annotation, sample::Sample};
use gen_models::{
    annotations::{Annotation, AnnotationExtra, GffAttribute},
    block_group::BlockGroup,
    db::GraphConnection,
    file_types::FileTypes,
    interval_tree::IntervalTreeSource as _,
    locus::GraphLocus,
    reference_alias::ReferenceAlias,
    region::{GenRegionError, ResolvedGenRegion, ResolvedRegionKind},
};
use gen_tui::{
    LineStyle, geometry::WorldPos, graph_controller::GraphController, layout::VisualDetail,
    plotter::PathStyle,
};
use intervaltree::IntervalTree;
use petgraph::{graph::NodeIndex, visit::NodeIndexable};
use ratatui::style::Color;

#[cfg(test)]
use crate::test_helpers::TestContext;
use crate::views::{
    annotation_files::{AnnotationAssetEntry, AnnotationFileEntry},
    annotation_track::{
        AnnotationSegment, AnnotationSpan, annotation_span_from_resolved_region,
        graph_locus_from_annotation_span,
    },
    annotations::resolve_annotation_search_paths,
    gen_graph_widget::{GenGraphNodeSizer, locus_midpoint},
    graph_overlay::{GraphOverlay, OverlayContent, OverlaySource},
    helpers::open_worker_connection,
};
#[cfg(test)]
use crate::{imports::fasta::import_fasta, test_helpers::setup_gen_on_disk};

#[derive(Clone, Debug)]
pub(super) struct RegionSearchMatch {
    pub(super) label: String,
    target: RegionSearchTarget,
}

#[derive(Clone, Debug)]
enum RegionSearchTarget {
    Resolved(Box<ResolvedGenRegion>),
    FileAnnotation {
        region: Region,
        records: Box<FileAnnotationRecords>,
    },
}

pub(super) struct RegionSearchRequest<'a> {
    pub(super) conn: &'a GraphConnection,
    pub(super) workspace: &'a Workspace,
    pub(super) collection_name: &'a str,
    pub(super) sample_name: &'a str,
    pub(super) block_group: &'a BlockGroup,
    pub(super) history_ref: Option<&'a str>,
    pub(super) annotation_files: &'a [AnnotationFileEntry],
    pub(super) active_annotation_files: &'a HashSet<HashId>,
    pub(super) index_cache: &'a RefCell<RegionSearchIndexCache>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct IndexedAssetIdentity {
    id: HashId,
    checksum: Option<Sha256Hash>,
    asset_uri: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct RegionSearchIndexIdentity {
    source: IndexedAssetIdentity,
    coordinate_index: IndexedAssetIdentity,
    attribute_index: IndexedAssetIdentity,
}

pub(super) enum CachedIndexedSource {
    Open(Box<IndexedSource>),
    Failed(String),
}

pub(super) type RegionSearchIndexCache = HashMap<RegionSearchIndexIdentity, CachedIndexedSource>;

#[derive(Clone, Debug)]
struct FileAnnotationRecords {
    entry: AnnotationFileEntry,
    annotation_name: String,
    raw_records: Vec<String>,
}

#[derive(Clone)]
pub(super) struct RegionSearchContext {
    workspace: Workspace,
    collection_name: String,
    sample_name: String,
    block_group: BlockGroup,
    history_ref: Option<String>,
    annotation_files: Vec<AnnotationFileEntry>,
    active_annotation_files: HashSet<HashId>,
}

impl RegionSearchContext {
    pub(super) fn new(
        workspace: &Workspace,
        collection_name: &str,
        sample_name: &str,
        block_group: &BlockGroup,
        history_ref: Option<&str>,
        annotation_files: &[AnnotationFileEntry],
        active_annotation_files: &HashSet<HashId>,
    ) -> Self {
        Self {
            workspace: workspace.clone(),
            collection_name: collection_name.to_string(),
            sample_name: sample_name.to_string(),
            block_group: block_group.clone(),
            history_ref: history_ref.map(str::to_string),
            annotation_files: annotation_files.to_vec(),
            active_annotation_files: active_annotation_files.clone(),
        }
    }

    fn request<'a>(
        &'a self,
        conn: &'a GraphConnection,
        index_cache: &'a RefCell<RegionSearchIndexCache>,
    ) -> RegionSearchRequest<'a> {
        RegionSearchRequest {
            conn,
            workspace: &self.workspace,
            collection_name: &self.collection_name,
            sample_name: &self.sample_name,
            block_group: &self.block_group,
            history_ref: self.history_ref.as_deref(),
            annotation_files: &self.annotation_files,
            active_annotation_files: &self.active_annotation_files,
            index_cache,
        }
    }
}

pub(super) struct RegionSearchTask {
    generation: u64,
    context: RegionSearchContext,
    operation: RegionSearchOperation,
}

enum RegionSearchOperation {
    Query(String),
    ResolveSelection(Box<RegionSearchMatch>),
}

impl RegionSearchTask {
    pub(super) fn query(generation: u64, context: RegionSearchContext, query: String) -> Self {
        Self {
            generation,
            context,
            operation: RegionSearchOperation::Query(query),
        }
    }

    pub(super) fn resolve_selection(
        generation: u64,
        context: RegionSearchContext,
        search_match: RegionSearchMatch,
    ) -> Self {
        Self {
            generation,
            context,
            operation: RegionSearchOperation::ResolveSelection(Box::new(search_match)),
        }
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }
}

pub(super) struct RegionSearchWorker {
    task_sender: Sender<RegionSearchTask>,
    response_receiver: Receiver<RegionSearchWorkerResponse>,
}

pub(super) enum RegionSearchWorkerPoll {
    Response(Box<RegionSearchWorkerResponse>),
    Empty,
    Disconnected,
}

pub(super) struct RegionSearchWorkerResponse {
    pub(super) generation: u64,
    pub(super) result: RegionSearchWorkerResult,
}

pub(super) enum RegionSearchWorkerResult {
    Matches(Result<Vec<RegionSearchMatch>, String>),
    Selection(Result<Box<RegionSearchMatch>, String>),
}

impl RegionSearchWorker {
    pub(super) fn new(
        database_path: PathBuf,
        branch: Option<String>,
        history_ref: Option<String>,
    ) -> Result<Self, String> {
        let (task_sender, task_receiver) = mpsc::channel::<RegionSearchTask>();
        let (response_sender, response_receiver) = mpsc::channel();
        thread::Builder::new()
            .name("region-search".to_string())
            .spawn(move || {
                let connection = open_worker_connection(
                    &database_path,
                    branch.as_deref(),
                    history_ref.as_deref(),
                );
                let index_cache = RefCell::new(RegionSearchIndexCache::default());
                while let Ok(task) = task_receiver.recv() {
                    let generation = task.generation;
                    let result = match task.operation {
                        RegionSearchOperation::Query(query) => match &connection {
                            Ok(conn) => {
                                let request = task.context.request(conn, &index_cache);
                                RegionSearchWorkerResult::Matches(resolve_region_search_matches(
                                    &request, &query,
                                ))
                            }
                            Err(error) => RegionSearchWorkerResult::Matches(Err(error.clone())),
                        },
                        RegionSearchOperation::ResolveSelection(search_match) => {
                            match &connection {
                                Ok(conn) => {
                                    let request = task.context.request(conn, &index_cache);
                                    RegionSearchWorkerResult::Selection(
                                        resolve_selected_search_match(&request, *search_match)
                                            .map(Box::new),
                                    )
                                }
                                Err(error) => {
                                    RegionSearchWorkerResult::Selection(Err(error.clone()))
                                }
                            }
                        }
                    };
                    if response_sender
                        .send(RegionSearchWorkerResponse { generation, result })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|error| format!("failed to start region search worker: {error}"))?;
        Ok(Self {
            task_sender,
            response_receiver,
        })
    }

    pub(super) fn submit(&self, task: RegionSearchTask) -> Result<(), String> {
        self.task_sender
            .send(task)
            .map_err(|_| "region search worker is unavailable".to_string())
    }

    pub(super) fn try_receive(&self) -> RegionSearchWorkerPoll {
        match self.response_receiver.try_recv() {
            Ok(response) => RegionSearchWorkerPoll::Response(Box::new(response)),
            Err(mpsc::TryRecvError::Empty) => RegionSearchWorkerPoll::Empty,
            Err(mpsc::TryRecvError::Disconnected) => RegionSearchWorkerPoll::Disconnected,
        }
    }
}

fn resolved_match_label(query: &str, kind: ResolvedRegionKind) -> String {
    let source = match kind {
        ResolvedRegionKind::Path => "path",
        ResolvedRegionKind::Annotation => "model annotation",
        ResolvedRegionKind::Accession => "accession",
        ResolvedRegionKind::BlockGroup => "block group",
    };
    format!("{query} ({source})")
}

fn translate_user_region(region: &Region) -> Region {
    Region {
        name: region.name.clone(),
        start: region.start.map(|coordinate| {
            if coordinate > 0 {
                coordinate - 1
            } else {
                coordinate
            }
        }),
        end: region.end,
    }
}

pub(super) fn resolve_region_search_matches(
    request: &RegionSearchRequest<'_>,
    query: &str,
) -> Result<Vec<RegionSearchMatch>, String> {
    let reference_names = searchable_reference_names(request)?;
    let mut index_cache = request.index_cache.borrow_mut();
    let mut file_search_errors = Vec::new();
    let literal_file_records = query_active_annotation_files(
        request,
        query,
        &reference_names,
        &mut index_cache,
        &mut file_search_errors,
    );

    // GFF aliases such as `GeneID:123` are literal attribute values for file lookup. Keep the
    // parsed model/path region separate so the query can also return coordinate matches.
    let literal_colon_alias = query.contains(':')
        && literal_file_records
            .iter()
            .any(|records| records.annotation_name.contains(':'));
    let user_region = match Region::parse(query) {
        Ok(region) => region,
        Err(_error) if literal_colon_alias => Region {
            name: query.to_string(),
            start: None,
            end: None,
        },
        Err(error) => {
            return Err(file_search_errors.first().map_or_else(
                || error.to_string(),
                |file_error| format!("{error}; {file_error}"),
            ));
        }
    };
    let region = translate_user_region(&user_region);
    let file_region = if literal_colon_alias {
        Region {
            name: query.to_string(),
            start: None,
            end: None,
        }
    } else {
        region.clone()
    };

    let file_records = if file_region.start.is_some() || file_region.end.is_some() {
        query_active_annotation_files(
            request,
            &file_region.name,
            &reference_names,
            &mut index_cache,
            &mut file_search_errors,
        )
    } else {
        literal_file_records
    };
    let resolved_regions = match gen_models::region::resolve_all(
        &region,
        request.conn,
        request.collection_name,
        request.sample_name,
    ) {
        Ok(resolved_regions) => resolved_regions,
        Err(GenRegionError::NotFound(_)) => Vec::new(),
        Err(error) => {
            file_search_errors.push(error.to_string());
            Vec::new()
        }
    };
    let mut matches = resolved_regions
        .into_iter()
        .map(|region| RegionSearchMatch {
            label: resolved_match_label(query, region.kind),
            target: RegionSearchTarget::Resolved(Box::new(region)),
        })
        .collect::<Vec<_>>();

    for file_records in file_records {
        matches.push(RegionSearchMatch {
            label: format!(
                "{} (file annotation: {})",
                file_records.annotation_name, file_records.entry.display_name
            ),
            target: RegionSearchTarget::FileAnnotation {
                region: file_region.clone(),
                records: Box::new(file_records),
            },
        });
    }

    if matches.is_empty() {
        if let Some(error) = file_search_errors.first() {
            return Err(error.clone());
        }
        return Err(format!("no region matched {query}"));
    }

    Ok(matches)
}

fn searchable_reference_names(
    request: &RegionSearchRequest<'_>,
) -> Result<HashSet<String>, String> {
    let block_group_name = request.block_group.name.clone();
    let references_by_alias = ReferenceAlias::get_references_by_alias(
        request.conn,
        vec![block_group_name.clone()],
        request.history_ref,
    )
    .map_err(|error| error.to_string())?;
    let mut reference_names = HashSet::from([block_group_name.clone()]);
    reference_names.extend(
        references_by_alias
            .into_iter()
            .filter_map(|(alias, reference)| (reference == block_group_name).then_some(alias)),
    );
    Ok(reference_names)
}

fn indexed_asset_identity(asset: &AnnotationAssetEntry) -> IndexedAssetIdentity {
    IndexedAssetIdentity {
        id: asset.id,
        checksum: asset.checksum,
        asset_uri: asset.asset_uri.clone(),
    }
}

fn indexed_source_identity(entry: &AnnotationFileEntry) -> Option<RegionSearchIndexIdentity> {
    Some(RegionSearchIndexIdentity {
        source: indexed_asset_identity(&entry.file_addition),
        coordinate_index: indexed_asset_identity(entry.index_file_addition.as_ref()?),
        attribute_index: indexed_asset_identity(entry.attribute_index_file_addition.as_ref()?),
    })
}

fn cached_indexed_source<'a>(
    workspace: &Workspace,
    entry: &AnnotationFileEntry,
    cache: &'a mut RegionSearchIndexCache,
) -> Result<&'a mut IndexedSource, String> {
    let identity = indexed_source_identity(entry).ok_or_else(|| {
        format!(
            "{} has no recorded coordinate and attribute indexes",
            entry.display_name
        )
    })?;
    let cached = cache.entry(identity).or_insert_with(|| {
        let result = resolve_annotation_search_paths(workspace, entry)
            .map_err(|error| error.to_string())
            .and_then(|paths| {
                paths
                    .ok_or_else(|| "recorded annotation search indexes are unavailable".to_string())
            })
            .and_then(|paths| {
                IndexedSource::open(
                    paths.source_path,
                    paths.coordinate_index_path,
                    paths.attribute_index_path,
                )
                .map_err(|error| error.to_string())
            });
        match result {
            Ok(indexed_source) => CachedIndexedSource::Open(Box::new(indexed_source)),
            Err(error) => CachedIndexedSource::Failed(error),
        }
    });
    match cached {
        CachedIndexedSource::Open(indexed_source) => Ok(indexed_source.as_mut()),
        CachedIndexedSource::Failed(error) => Err(format!("{}: {error}", entry.display_name)),
    }
}

fn query_active_annotation_files(
    request: &RegionSearchRequest<'_>,
    query: &str,
    reference_names: &HashSet<String>,
    cache: &mut RegionSearchIndexCache,
    errors: &mut Vec<String>,
) -> Vec<FileAnnotationRecords> {
    let mut candidates = Vec::new();
    for entry in request.annotation_files {
        if !request
            .active_annotation_files
            .contains(&entry.file_addition.id)
        {
            continue;
        }
        if entry.index_file_addition.is_none() || entry.attribute_index_file_addition.is_none() {
            continue;
        }
        if !matches!(
            entry.file_addition.file_type,
            FileTypes::Gff3 | FileTypes::Bed
        ) {
            continue;
        }
        match query_annotation_file(request.workspace, entry, query, reference_names, cache) {
            Ok(file_candidates) => candidates.extend(file_candidates),
            Err(error) => errors.push(error),
        }
    }
    candidates
}

fn query_annotation_file(
    workspace: &Workspace,
    entry: &AnnotationFileEntry,
    query: &str,
    reference_names: &HashSet<String>,
    cache: &mut RegionSearchIndexCache,
) -> Result<Vec<FileAnnotationRecords>, String> {
    let indexed_source = cached_indexed_source(workspace, entry, cache)?;
    let configured_attributes = indexed_source.metadata().attributes.clone();
    let indexed_references = reference_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let records = indexed_source
        .query_on_contigs_with_mode(query, &indexed_references, MatchMode::Contains)
        .map_err(|error| format!("{}: {error}", entry.display_name))?;
    let query = query.to_lowercase();
    let configured_attributes = configured_attributes
        .into_iter()
        .map(|attribute| attribute.to_lowercase())
        .collect::<HashSet<_>>();
    let mut grouped_records = BTreeMap::<String, (String, Vec<String>, HashSet<String>)>::new();
    for record in records {
        let annotation_names = record
            .attributes
            .iter()
            .filter(|(attribute, _)| {
                let attribute = attribute.to_lowercase();
                attribute != "id" && configured_attributes.contains(&attribute)
            })
            .flat_map(|(_, values)| values)
            .filter(|value| value.to_lowercase().contains(&query))
            .filter(|value| !value.trim().is_empty())
            .cloned()
            .collect::<HashSet<_>>();
        if annotation_names.is_empty() {
            continue;
        }

        let raw_record = match entry.file_addition.file_type {
            FileTypes::Bed => pad_bed_record(&record.raw_line),
            FileTypes::Gff3 => record.raw_line,
            _ => continue,
        };
        for annotation_name in annotation_names {
            let key = annotation_name.to_lowercase();
            let (_, raw_records, distinct_records) = grouped_records
                .entry(key)
                .or_insert_with(|| (annotation_name.clone(), Vec::new(), HashSet::new()));
            if distinct_records.insert(raw_record.clone()) {
                raw_records.push(raw_record.clone());
            }
        }
    }

    Ok(grouped_records
        .into_values()
        .map(|(annotation_name, raw_records, _)| FileAnnotationRecords {
            entry: entry.clone(),
            annotation_name,
            raw_records,
        })
        .collect())
}

fn pad_bed_record(raw_record: &str) -> String {
    let mut fields = raw_record.split('\t').collect::<Vec<_>>();
    while fields.len() < 6 {
        fields.push(match fields.len() {
            4 => "0",
            5 => ".",
            _ => "",
        });
    }
    fields.join("\t")
}

fn selected_annotation_records(raw_records: &[String]) -> Vec<u8> {
    let mut input = Vec::new();
    for raw_record in raw_records {
        input.extend_from_slice(raw_record.as_bytes());
        input.push(b'\n');
    }
    input
}

fn combine_file_annotation_records(
    name: &str,
    file_type: FileTypes,
    mut records: Vec<Annotation>,
) -> Result<Annotation, String> {
    if records.len() == 1 {
        let mut annotation = records
            .pop()
            .ok_or_else(|| "annotation has no translated records".to_string())?;
        annotation.name = name.to_string();
        return Ok(annotation);
    }

    let mut annotation = records
        .first()
        .cloned()
        .ok_or_else(|| "annotation has no translated records".to_string())?;
    let accession = annotation
        .cached_accession
        .snapshot()
        .ok_or_else(|| "translated annotation has no cached accession".to_string())?;
    let feature_strand = annotation
        .cached_interval_tree
        .as_ref()
        .and_then(|tree| tree.iter_sorted().next())
        .map(|element| element.value.strand)
        .unwrap_or(Strand::Unknown);

    if file_type == FileTypes::Gff3 {
        let attributes = records
            .iter()
            .filter_map(|record| record.extra.as_ref())
            .filter_map(|extra| extra.gff.as_ref())
            .flat_map(|extra| extra.attributes.iter().cloned())
            .collect::<Vec<GffAttribute>>();
        let gff_extra = annotation
            .extra
            .get_or_insert_with(AnnotationExtra::default)
            .gff
            .get_or_insert_with(Default::default);
        gff_extra.attributes = attributes;
    }

    if feature_strand == Strand::Reverse {
        records.reverse();
    }

    let mut position = 0_i64;
    let mut intervals = Vec::new();
    for record in &records {
        let interval_tree = record
            .cached_interval_tree
            .as_ref()
            .ok_or_else(|| "translated annotation has no cached intervals".to_string())?;
        let record_length = interval_tree
            .iter_sorted()
            .map(|element| element.range.end)
            .max()
            .unwrap_or_default();
        for element in interval_tree.iter_sorted() {
            let start = position.saturating_add(element.range.start);
            let end = position.saturating_add(element.range.end);
            let mut block = element.value;
            block.start = start;
            block.end = end;
            intervals.push((start..end, block));
        }
        position = position.saturating_add(record_length);
    }

    if intervals.is_empty() {
        return Err("annotation has no translated intervals".to_string());
    }

    let interval_tree = Arc::new(
        intervals
            .into_iter()
            .collect::<IntervalTree<i64, NodeIntervalBlock>>(),
    );
    let identifier = HashId::convert_str(name);
    annotation.id = identifier;
    annotation.name = name.to_string();
    annotation.accession_id = identifier;
    annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));

    let mut accession = accession.as_ref().clone();
    accession.id = identifier;
    accession.name = name.to_string();
    accession.set_cached_interval_tree(Some(interval_tree));
    annotation.set_accession(Arc::new(accession));
    Ok(annotation)
}

fn resolve_file_annotation_match(
    request: &RegionSearchRequest<'_>,
    region: &Region,
    file_records: FileAnnotationRecords,
) -> Result<RegionSearchMatch, String> {
    let mut translation_context = AnnotationTranslationContext::new(
        request.conn,
        request.workspace,
        request.collection_name,
        request.sample_name,
        request.history_ref,
    )
    .map_err(|error| error.to_string())?;
    let input = selected_annotation_records(&file_records.raw_records);
    let translated_records = match file_records.entry.file_addition.file_type {
        FileTypes::Gff3 => translate_gff_annotation(&mut translation_context, Cursor::new(input))
            .collect::<Result<Vec<_>, _>>(),
        FileTypes::Bed => translate_bed_annotation(&mut translation_context, Cursor::new(input))
            .collect::<Result<Vec<_>, _>>(),
        file_type => return Err(format!("unsupported annotation file type: {file_type:?}")),
    }
    .map_err(|error| error.to_string())?;
    let annotation = combine_file_annotation_records(
        &file_records.annotation_name,
        file_records.entry.file_addition.file_type,
        translated_records,
    )?;
    let annotation_length = annotation
        .length(request.conn)
        .map_err(|error| error.to_string())?;
    let (start, end) = match (region.start, region.end) {
        (None, None) => (0, annotation_length),
        (Some(start), None) => (start, annotation_length),
        (Some(start), Some(end)) => (start, end),
        (None, Some(_)) => return Err("invalid relative annotation range".to_string()),
    };
    let mut resolved_region =
        ResolvedGenRegion::from_annotation(request.conn, &annotation, start, end)
            .map_err(|error| error.to_string())?;
    resolved_region.block_group = request.block_group.clone();
    Ok(RegionSearchMatch {
        label: format!(
            "{} (file annotation: {})",
            file_records.annotation_name, file_records.entry.display_name
        ),
        target: RegionSearchTarget::Resolved(Box::new(resolved_region)),
    })
}

fn resolve_selected_search_match(
    request: &RegionSearchRequest<'_>,
    search_match: RegionSearchMatch,
) -> Result<RegionSearchMatch, String> {
    let RegionSearchMatch { label, target } = search_match;
    match target {
        RegionSearchTarget::Resolved(region) => Ok(RegionSearchMatch {
            label,
            target: RegionSearchTarget::Resolved(region),
        }),
        RegionSearchTarget::FileAnnotation { region, records } => {
            resolve_file_annotation_match(request, &region, *records)
        }
    }
}

impl RegionSearchMatch {
    pub(super) fn needs_selection_resolution(&self) -> bool {
        matches!(&self.target, RegionSearchTarget::FileAnnotation { .. })
    }

    fn resolved_region(&self) -> Result<&ResolvedGenRegion, String> {
        match &self.target {
            RegionSearchTarget::Resolved(region) => Ok(region.as_ref()),
            RegionSearchTarget::FileAnnotation { .. } => {
                Err("file annotation must be resolved before activation".to_string())
            }
        }
    }
}

pub(super) fn replace_search_overlay(overlays: &mut Vec<GraphOverlay>, span: AnnotationSpan) {
    remove_search_overlay(overlays);
    overlays.push(GraphOverlay {
        content: OverlayContent::Span(span),
        source: OverlaySource::Search,
        style: PathStyle::new(Color::LightYellow)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true),
    });
}

pub(super) fn remove_search_overlay(overlays: &mut Vec<GraphOverlay>) {
    overlays.retain(|overlay| !matches!(&overlay.source, OverlaySource::Search));
}

fn fallback_locus_for_empty_span(
    region: &ResolvedGenRegion,
    conn: &GraphConnection,
    workspace: &Workspace,
    graph: &GenGraph,
) -> Result<GraphLocus, String> {
    let midpoint = region
        .start
        .saturating_add(region.end.saturating_sub(region.start) / 2);
    let offset = midpoint.saturating_sub(region.start);
    let positioned = region
        .find_graph_positions(conn, workspace, offset, 0)
        .map_err(|error| format!("failed to locate region on graph: {error}"))?;
    let position = positioned
        .start_anchors
        .and_then(|positions| positions.into_iter().next())
        .ok_or_else(|| "region did not map to a graph position".to_string())?;
    let coordinate = position.coordinate();

    // A zero-width point has no interval to project. Represent it as a one-base span at the
    // resolved backing-node coordinate so the normal graph projection handles displayed slices.
    for (start, end) in [
        (coordinate, coordinate.saturating_add(1)),
        (coordinate.saturating_sub(1), coordinate),
    ] {
        if start >= end {
            continue;
        }
        let point_span = AnnotationSpan {
            id: HashId::convert_str("region-search-point"),
            name: String::new(),
            segments: vec![AnnotationSegment {
                node_id: position.graph_node.node_id,
                start,
                end,
                strand: Strand::Forward,
            }],
        };
        if let Some(locus) = graph_locus_from_annotation_span(&point_span, graph) {
            return Ok(locus);
        }
    }

    Err("region did not map to a graph position".to_string())
}

pub(super) fn activate_search_match(
    graph_controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    overlays: &mut Vec<GraphOverlay>,
    search_match: &RegionSearchMatch,
    conn: &GraphConnection,
    workspace: &Workspace,
) -> Result<(), String> {
    let region = search_match.resolved_region()?;
    let span = annotation_span_from_resolved_region(conn, workspace, region)?;
    let locus = if span.segments.is_empty() {
        fallback_locus_for_empty_span(region, conn, workspace, graph_controller.graph())?
    } else {
        graph_locus_from_annotation_span(&span, graph_controller.graph())
            .ok_or_else(|| "region did not map to a graph position".to_string())?
    };
    navigate_to_search_locus(graph_controller, &locus)?;
    replace_search_overlay(overlays, span);
    Ok(())
}

fn navigate_to_search_locus(
    graph_controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    locus: &GraphLocus,
) -> Result<(), String> {
    let (midpoint_slice, midpoint_offset) = locus_midpoint(locus)
        .ok_or_else(|| "region did not map to a graph position".to_string())?;
    let was_base_resolution = graph_controller.get_detail_level() == VisualDetail::Full;
    let region_is_visible = was_base_resolution && search_locus_is_visible(graph_controller, locus);
    graph_controller.set_detail_level(VisualDetail::Full);
    if region_is_visible {
        return Ok(());
    }

    let locus_length = locus
        .slices
        .iter()
        .map(|slice| slice.end.saturating_sub(slice.start))
        .sum::<usize>();
    let viewport_width = graph_controller.viewport_state.viewport_bounds.width as usize;
    let leading_slice = locus
        .slices
        .first()
        .ok_or_else(|| "region did not map to a graph position".to_string())?;
    let target = if locus_length > viewport_width {
        (leading_slice, leading_slice.start, true)
    } else {
        (&midpoint_slice, midpoint_offset, false)
    };
    let node_index = NodeIndex::new(<GenGraph as NodeIndexable>::to_index(
        graph_controller.graph(),
        target.0.block,
    ));
    let node_length = target.0.block.length();
    let fraction = if node_length == 0 {
        0.5
    } else {
        (target.1 as f64 / node_length as f64).clamp(0.0, 1.0)
    };
    graph_controller.go_to_node(node_index, (fraction, 0.5));
    if target.2 {
        graph_controller.queue_snap_left();
    }
    graph_controller.rebuild_viewport_graph()?;
    if !target.2 && !search_locus_is_visible(graph_controller, locus) {
        let node_index = NodeIndex::new(<GenGraph as NodeIndexable>::to_index(
            graph_controller.graph(),
            leading_slice.block,
        ));
        let node_length = leading_slice.block.length();
        let fraction = if node_length == 0 {
            0.5
        } else {
            (leading_slice.start as f64 / node_length as f64).clamp(0.0, 1.0)
        };
        graph_controller.go_to_node(node_index, (fraction, 0.5));
        graph_controller.queue_snap_left();
        graph_controller.rebuild_viewport_graph()?;
    }
    Ok(())
}

fn search_locus_is_visible(
    graph_controller: &GraphController<GenGraph, GenGraphNodeSizer>,
    locus: &GraphLocus,
) -> bool {
    let camera_rect = graph_controller.viewport_state.camera_rect();
    let viewport_graph = graph_controller.get_viewport_graph();
    locus.slices.iter().all(|slice| {
        let node_index = NodeIndex::new(<GenGraph as NodeIndexable>::to_index(
            graph_controller.graph(),
            slice.block,
        ));
        let Some(node_center) = viewport_graph.node_positions.get(&node_index) else {
            return false;
        };
        let node_width = slice.block.length();
        if node_width <= 0 {
            return false;
        }
        let node_start = node_center.x - (node_width.saturating_sub(1) / 2);
        let slice_start = i64::try_from(slice.start).unwrap_or(i64::MAX);
        let slice_end = i64::try_from(slice.end.saturating_sub(1)).unwrap_or(i64::MAX);
        let first_base = WorldPos::new(node_start + slice_start, node_center.y);
        let last_base = WorldPos::new(node_start + slice_end, node_center.y);
        camera_rect.contains(first_base) && camera_rect.contains(last_base)
    })
}

#[cfg(test)]
pub(super) struct RegionSearchFixture {
    pub(super) context: TestContext,
    pub(super) graph: GenGraph,
    pub(super) block_group: BlockGroup,
    pub(super) annotation_files: Vec<AnnotationFileEntry>,
    pub(super) active_annotation_files: HashSet<HashId>,
    pub(super) index_cache: RefCell<RegionSearchIndexCache>,
}

#[cfg(test)]
impl RegionSearchFixture {
    pub(super) fn request(&self) -> RegionSearchRequest<'_> {
        RegionSearchRequest {
            conn: self.context.graph().conn(),
            workspace: self.context.workspace(),
            collection_name: "test",
            sample_name: self.block_group.sample_name.as_str(),
            block_group: &self.block_group,
            history_ref: None,
            annotation_files: &self.annotation_files,
            active_annotation_files: &self.active_annotation_files,
            index_cache: &self.index_cache,
        }
    }

    pub(super) fn workspace(&self) -> &Workspace {
        self.context.workspace()
    }
}

#[cfg(test)]
pub(super) fn search_request_fixture() -> RegionSearchFixture {
    let context = setup_gen_on_disk();
    let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
    let fasta_path = fasta_path
        .to_str()
        .expect("should encode FASTA fixture path")
        .to_string();
    import_fasta(&context, &fasta_path, "test", "simple", None, None)
        .expect("should import search fixture FASTA");
    add_annotation(&context, "test", "model-gene", None, "simple", "m123:5-20")
        .expect("should create model annotation");
    let conn = context.graph().conn();
    let duplicate_block_group = BlockGroup::get_by_name(conn, "test", "simple", "m123", None)
        .expect("should find duplicate annotation block group");
    let duplicate_path = BlockGroup::get_current_path(conn, &duplicate_block_group.id, None)
        .expect("should find duplicate annotation path");
    let path_edge_ids = gen_models::path::Path::edge_ids_for_path(conn, &duplicate_path.id, None);
    gen_models::path::Path::create(conn, "chr1", &duplicate_block_group.id, &path_edge_ids)
        .expect("should create a distinct path search fixture");
    for (accession_name, start, end) in [
        ("duplicate-accession-a", 0, 5),
        ("duplicate-accession-b", 20, 25),
    ] {
        let accession =
            BlockGroup::add_accession(conn, &duplicate_path, accession_name, start, end)
                .expect("should create duplicate annotation accession");
        Annotation::get_or_create(conn, "duplicate-gene", "default", &accession.id, None)
            .expect("should create duplicate model annotation");
    }

    let conn = context.graph().conn();
    let block_group = Sample::get_block_groups(conn, "test", "simple", None)
        .into_iter()
        .find(|block_group| block_group.name == "m123")
        .expect("should find imported block group");
    let graph = BlockGroup::get_graph(conn, context.workspace(), &block_group.id, None)
        .expect("should load search graph");
    let annotation_files =
        crate::views::annotation_files::load_annotation_file_entries(context.graph().conn(), None);
    RegionSearchFixture {
        context,
        graph,
        block_group,
        annotation_files,
        active_annotation_files: HashSet::new(),
        index_cache: RefCell::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs, path::PathBuf};

    use gen_core::{HashId, Strand, region::Region};
    use gen_graph::GenGraph;
    use gen_models::{
        annotations::AnnotationFileChecksumOverrides,
        locus::{GraphLocus, GraphNodeSlice},
    };
    use gen_tui::{geometry::WorldPos, layout::VisualDetail, plotter::PathStyle};
    use petgraph::visit::NodeIndexable;
    use ratatui::{layout::Rect, style::Color};
    use tempfile::tempdir;

    use super::{
        RegionSearchFixture, RegionSearchMatch, activate_search_match, navigate_to_search_locus,
        query_annotation_file, remove_search_overlay, resolve_region_search_matches,
        resolve_selected_search_match, search_locus_is_visible, search_request_fixture,
        translate_user_region,
    };
    use crate::views::{
        annotation_track::{
            AnnotationSpan, annotation_span_from_resolved_region, graph_locus_from_annotation_span,
        },
        gen_graph_widget::create_gen_graph_controller,
        graph_overlay::{GraphOverlay, OverlayContent, OverlaySource},
    };

    fn match_for_query(fixture: &RegionSearchFixture, query: &str) -> RegionSearchMatch {
        let request = fixture.request();
        let search_match = resolve_region_search_matches(&request, query)
            .expect("should resolve query")
            .into_iter()
            .next()
            .expect("query should have a match");
        resolve_selected_search_match(&request, search_match)
            .expect("should resolve selected search match")
    }

    fn span_for_match(
        fixture: &RegionSearchFixture,
        search_match: &RegionSearchMatch,
    ) -> AnnotationSpan {
        let request = fixture.request();
        let selected_match = resolve_selected_search_match(&request, search_match.clone())
            .expect("should resolve selected file annotations before projecting their spans");
        annotation_span_from_resolved_region(
            fixture.context.graph().conn(),
            fixture.workspace(),
            selected_match
                .resolved_region()
                .expect("should resolve search match"),
        )
        .expect("should map resolved region to an annotation span")
    }

    fn add_indexed_annotation_file(
        fixture: &mut RegionSearchFixture,
        filename: &str,
        file_type: &str,
        contents: &str,
        activate: bool,
    ) -> HashId {
        let directory = tempdir().expect("should create annotation input directory");
        let annotation_path = directory.path().join(filename);
        fs::write(&annotation_path, contents).expect("should write annotation search fixture");
        let annotation_path = annotation_path
            .to_str()
            .expect("should encode annotation search fixture path")
            .to_string();
        gen_models::annotations::add_annotation_file(
            &fixture.context,
            &annotation_path,
            Some(file_type),
            None,
            Some(filename),
            Some("add TUI annotation search fixture"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should add and index TUI annotation search fixture");
        fixture.annotation_files = crate::views::annotation_files::load_annotation_file_entries(
            fixture.context.graph().conn(),
            None,
        );
        let entry = fixture
            .annotation_files
            .iter()
            .find(|entry| entry.display_name == filename)
            .expect("should load added annotation file entry");
        let id = entry.file_addition.id;
        if activate {
            fixture.active_annotation_files.insert(id);
        }
        id
    }

    fn split_fixture_graph(fixture: &mut RegionSearchFixture) {
        crate::updates::sequence::update_with_sequence(
            &fixture.context,
            "test",
            "simple",
            "tui-search-split",
            "m123:8-10",
            "AAA",
            false,
        )
        .expect("should create a split graph around the file annotation");
        let conn = fixture.context.graph().conn();
        fixture.block_group =
            gen_models::sample::Sample::get_block_groups(conn, "test", "tui-search-split", None)
                .into_iter()
                .find(|block_group| block_group.name == "m123")
                .expect("should find the edited block group");
        fixture.graph = gen_models::block_group::BlockGroup::get_graph(
            conn,
            fixture.context.workspace(),
            &fixture.block_group.id,
            None,
        )
        .expect("should load the edited search graph");
    }

    #[test]
    fn test_region_search_locates_block_groups_paths_and_model_annotations() {
        let request = search_request_fixture();

        let path_match = match_for_query(&request, "chr1:20-30");
        assert_eq!(path_match.label, "chr1:20-30 (path)");
        let path_span = span_for_match(&request, &path_match);
        assert_eq!(path_span.segments.len(), 1);
        assert_eq!(path_span.segments[0].start, 19);
        assert_eq!(path_span.segments[0].end, 30);

        let mut controller = create_gen_graph_controller(request.graph.clone());
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, 80, 24);
        controller.set_detail_level(VisualDetail::Truncated);
        controller
            .rebuild_viewport_graph()
            .expect("should render the starting locus at truncated detail");
        let mut overlays = Vec::new();
        activate_search_match(
            &mut controller,
            &mut overlays,
            &path_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("should select and center the path match");
        assert_eq!(controller.get_detail_level(), VisualDetail::Full);
        controller
            .rebuild_viewport_graph()
            .expect("should rebuild the full-detail search viewport");
        assert!(search_locus_is_visible(
            &controller,
            &graph_locus_from_annotation_span(&path_span, &request.graph)
                .expect("should map path search span to a graph locus"),
        ));
        controller.viewport_state.camera_current.x += 3;
        controller.viewport_state.camera_target = controller.viewport_state.camera_current;
        controller
            .rebuild_viewport_graph()
            .expect("should rebuild after a small pan");
        let panned_camera = controller.viewport_state.camera_current;
        activate_search_match(
            &mut controller,
            &mut overlays,
            &path_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("should keep an already-visible search locus in place");
        controller
            .rebuild_viewport_graph()
            .expect("should rebuild the already-visible locus without recentering");
        assert_eq!(controller.viewport_state.camera_current, panned_camera);

        controller.set_detail_level(VisualDetail::Truncated);
        controller
            .rebuild_viewport_graph()
            .expect("should render the selected locus at truncated detail");
        activate_search_match(
            &mut controller,
            &mut overlays,
            &path_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("should return a selected region to base detail");
        assert_eq!(controller.get_detail_level(), VisualDetail::Full);
        controller
            .rebuild_viewport_graph()
            .expect("should rebuild the base-detail search viewport");
        assert!(search_locus_is_visible(
            &controller,
            &graph_locus_from_annotation_span(&path_span, &request.graph)
                .expect("should map the selected path to a graph locus"),
        ));

        let selected_index = controller
            .cursor
            .node_idx()
            .expect("should select the resolved graph node");
        let selected_node =
            <&GenGraph as NodeIndexable>::from_index(&controller.graph(), selected_index.index());
        assert_eq!(selected_node.sequence_start, 0);
        let (selected_fraction, _) = controller.cursor.fractional_pos();
        assert!((selected_fraction - (24.0 / 34.0)).abs() < f64::EPSILON);
        assert_eq!(overlays.len(), 1);
        assert!(matches!(overlays[0].source, OverlaySource::Search));

        let path_slice_match = match_for_query(&request, "chr1:5-10");
        let path_slice_span = span_for_match(&request, &path_slice_match);
        assert_eq!(path_slice_span.segments.len(), 1);
        assert_eq!(path_slice_span.segments[0].start, 4);
        assert_eq!(path_slice_span.segments[0].end, 10);

        let single_base_match = match_for_query(&request, "chr1:5-5");
        let single_base_span = span_for_match(&request, &single_base_match);
        assert_eq!(single_base_span.segments.len(), 1);
        assert_eq!(single_base_span.segments[0].start, 4);
        assert_eq!(single_base_span.segments[0].end, 5);

        let model_match = match_for_query(&request, "model-gene:5-10");
        assert_eq!(model_match.label, "model-gene:5-10 (model annotation)");
        let model_span = span_for_match(&request, &model_match);
        assert_eq!(model_span.segments.len(), 1);
        assert_eq!(model_span.segments[0].start, 9);
        assert_eq!(model_span.segments[0].end, 15);

        let model_slice_match = match_for_query(&request, "model-gene:1-4");
        let model_slice_span = span_for_match(&request, &model_slice_match);
        assert_eq!(model_slice_span.segments.len(), 1);
        assert_eq!(model_slice_span.segments[0].start, 5);
        assert_eq!(model_slice_span.segments[0].end, 9);

        let model_single_base_match = match_for_query(&request, "model-gene:5-5");
        let model_single_base_span = span_for_match(&request, &model_single_base_match);
        assert_eq!(model_single_base_span.segments[0].start, 9);
        assert_eq!(model_single_base_span.segments[0].end, 10);

        let block_group_match = match_for_query(&request, "m123:20-30");
        assert_eq!(block_group_match.label, "m123:20-30 (block group)");
        let block_group_span = span_for_match(&request, &block_group_match);
        assert_eq!(block_group_span.segments.len(), 1);
        assert_eq!(block_group_span.segments[0].start, 19);
        assert_eq!(block_group_span.segments[0].end, 30);

        let zero_match = match_for_query(&request, "chr1:0-1");
        let zero_span = span_for_match(&request, &zero_match);
        assert_eq!(zero_span.segments[0].start, 0);
        assert_eq!(zero_span.segments[0].end, 1);
    }

    #[test]
    fn test_reverse_locus_wider_than_viewport_starts_at_visual_lower_bound() {
        let request = search_request_fixture();
        let path_match = match_for_query(&request, "chr1:20-30");
        let path_span = span_for_match(&request, &path_match);
        let graph_locus = graph_locus_from_annotation_span(&path_span, &request.graph)
            .expect("should map path search span to a graph locus");
        let block = graph_locus
            .slices
            .first()
            .expect("should find one graph slice for the path search")
            .block;
        let reverse_locus = GraphLocus {
            slices: vec![GraphNodeSlice {
                block,
                start: 0,
                end: block.length() as usize,
                strand: Strand::Reverse,
            }],
        };
        let viewport_width = 12;
        let mut controller = create_gen_graph_controller(request.graph.clone());
        controller.viewport_state.viewport_bounds = Rect::new(0, 0, viewport_width, 24);

        navigate_to_search_locus(&mut controller, &reverse_locus)
            .expect("should navigate to a wide reverse locus");

        assert_eq!(controller.get_detail_level(), VisualDetail::Full);
        let node_index = petgraph::graph::NodeIndex::new(<GenGraph as NodeIndexable>::to_index(
            controller.graph(),
            block,
        ));
        let node_center = controller
            .get_viewport_graph()
            .node_positions
            .get(&node_index)
            .expect("the selected reverse locus should be in the viewport graph");
        let node_start = node_center.x - (block.length().saturating_sub(1) / 2);
        let lower_bound = WorldPos::new(node_start, node_center.y);
        let camera_rect = controller.viewport_state.camera_rect();
        assert!(
            camera_rect.contains(lower_bound),
            "the visual lower bound should remain visible at Full detail"
        );
        let visible_bases = camera_rect
            .right()
            .min(node_start + block.length() - 1)
            .saturating_sub(camera_rect.left().max(node_start))
            + 1;
        assert!(
            visible_bases >= i64::from(viewport_width / 2),
            "the camera should show a substantial part of the selected span"
        );
    }

    #[test]
    fn test_region_search_activates_zero_and_negative_points() {
        let request = search_request_fixture();
        let zero_match = match_for_query(&request, "chr1:0");
        let mut zero_controller = create_gen_graph_controller(request.graph.clone());
        let mut zero_overlays = Vec::new();

        activate_search_match(
            &mut zero_controller,
            &mut zero_overlays,
            &zero_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("zero coordinate should navigate to the graph boundary");
        assert!(zero_controller.cursor.node_idx().is_some());
        assert_eq!(zero_overlays.len(), 1);
        assert!(matches!(
            &zero_overlays[0].content,
            OverlayContent::Span(span) if span.segments.is_empty()
        ));

        let negative_match = match_for_query(&request, "model-gene:-3");
        assert_eq!(
            (
                negative_match
                    .resolved_region()
                    .expect("should resolve negative search match")
                    .start,
                negative_match
                    .resolved_region()
                    .expect("should resolve negative search match")
                    .end,
            ),
            (-3, -3)
        );
        let mut negative_controller = create_gen_graph_controller(request.graph.clone());
        let mut negative_overlays = Vec::new();
        activate_search_match(
            &mut negative_controller,
            &mut negative_overlays,
            &negative_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("negative model annotation coordinate should navigate upstream");
        assert!(negative_controller.cursor.node_idx().is_some());
        assert_eq!(negative_overlays.len(), 1);
        assert!(matches!(
            &negative_overlays[0].content,
            OverlayContent::Span(span) if span.segments.is_empty()
        ));
    }

    #[test]
    fn test_user_region_coordinates_translate_to_zero_based_half_open() {
        assert_eq!(
            translate_user_region(&Region::parse("foo:5-10").expect("should parse range")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: Some(10),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:5-5").expect("should parse single base")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: Some(5),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:5").expect("should parse point")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: Some(5),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:5..").expect("should parse open range")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: None,
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo").expect("should parse name")),
            Region {
                name: "foo".to_string(),
                start: None,
                end: None,
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:0").expect("should parse zero")),
            Region {
                name: "foo".to_string(),
                start: Some(0),
                end: Some(0),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:0-5").expect("should parse mixed range")),
            Region {
                name: "foo".to_string(),
                start: Some(0),
                end: Some(5),
            }
        );
        assert_eq!(
            translate_user_region(
                &Region::parse("foo:-5--1").expect("should parse negative range")
            ),
            Region {
                name: "foo".to_string(),
                start: Some(-5),
                end: Some(-1),
            }
        );
    }

    #[test]
    fn test_region_search_resolves_ambiguous_matches() {
        let request = search_request_fixture();
        let same_name_matches = resolve_region_search_matches(&request.request(), "m123")
            .expect("should return block group and path matches with the same name");
        assert_eq!(same_name_matches.len(), 2);
        assert!(
            same_name_matches
                .iter()
                .any(|search_match| search_match.label.contains("block group"))
        );
        assert!(
            same_name_matches
                .iter()
                .any(|search_match| search_match.label.contains("path"))
        );

        let duplicate_matches = resolve_region_search_matches(&request.request(), "duplicate-gene")
            .expect("should return ambiguous model annotation matches");
        assert_eq!(duplicate_matches.len(), 2);
    }

    #[test]
    fn test_indexed_file_annotation_search_translates_and_activates_matches() {
        let mut fixture = search_request_fixture();
        add_indexed_annotation_file(
            &mut fixture,
            "genes.gff3",
            "gff3",
            concat!(
                "##gff-version 3\n",
                "m123\tgen-test\tgene\t5\t20\t.\t+\t.\tID=private-id-a;Name=gene-a0001;gene=GENE_ONE;db_xref=GeneID:123\n",
                "other\tgen-test\tgene\t5\t20\t.\t+\t.\tID=private-id-b;Name=gene-a0001;gene=GENE_ONE;db_xref=GeneID:123\n",
                "m123\tgen-test\tgene\t10\t12\t.\t+\t.\tID=private-id-c;Name=chr1:5;gene=coordinate_alias\n",
                "m123\tgen-test\tgene\t25\t30\t.\t+\t.\tID=private-id-gene-name;gene_name=BRCA1\n",
            ),
            true,
        );
        split_fixture_graph(&mut fixture);

        let name_matches = resolve_region_search_matches(&fixture.request(), "a0001")
            .expect("should search indexed GFF Name values");
        assert_eq!(name_matches.len(), 1);
        assert_eq!(
            name_matches[0].label,
            "gene-a0001 (file annotation: genes.gff3)"
        );
        assert!(name_matches[0].needs_selection_resolution());
        let selected_name_match =
            resolve_selected_search_match(&fixture.request(), name_matches[0].clone())
                .expect("should resolve a selected indexed GFF annotation");
        let full_span = span_for_match(&fixture, &selected_name_match);
        assert!(full_span.segments.len() > 1);
        assert_eq!(
            full_span
                .segments
                .iter()
                .map(|segment| segment.end - segment.start)
                .sum::<i64>(),
            16
        );

        let mut controller = create_gen_graph_controller(fixture.graph.clone());
        let mut overlays = Vec::new();
        activate_search_match(
            &mut controller,
            &mut overlays,
            &selected_name_match,
            fixture.context.graph().conn(),
            fixture.workspace(),
        )
        .expect("should activate selected file annotation on the current graph");
        assert!(controller.cursor.node_idx().is_some());
        assert!(matches!(
            &overlays[0].content,
            OverlayContent::Span(span)
                if span.segments.len() > 1
                    && span
                        .segments
                        .iter()
                        .map(|segment| segment.end - segment.start)
                        .sum::<i64>() == 16
        ));

        let gene_alias_matches = resolve_region_search_matches(&fixture.request(), "GENE_ONE")
            .expect("should search indexed GFF gene values case-insensitively");
        assert!(
            gene_alias_matches
                .iter()
                .any(|search_match| search_match.label.starts_with("GENE_ONE (file annotation:"))
        );

        let gene_name_matches = resolve_region_search_matches(&fixture.request(), "brca1")
            .expect("should search gene_name values case-insensitively");
        assert_eq!(gene_name_matches.len(), 1);
        assert_eq!(
            gene_name_matches[0].label,
            "BRCA1 (file annotation: genes.gff3)"
        );
        let selected_gene_name_match =
            resolve_selected_search_match(&fixture.request(), gene_name_matches[0].clone())
                .expect("should resolve the selected gene_name-only annotation");
        assert_eq!(
            (
                selected_gene_name_match
                    .resolved_region()
                    .expect("should resolve gene_name match")
                    .start,
                selected_gene_name_match
                    .resolved_region()
                    .expect("should resolve gene_name match")
                    .end,
            ),
            (0, 6),
            "gene_name-only search should preserve the six-base feature span"
        );
        let gene_name_span = span_for_match(&fixture, &selected_gene_name_match);
        assert_eq!(
            gene_name_span
                .segments
                .iter()
                .map(|segment| segment.end - segment.start)
                .sum::<i64>(),
            6
        );
        let mut gene_name_controller = create_gen_graph_controller(fixture.graph.clone());
        let mut gene_name_overlays = Vec::new();
        activate_search_match(
            &mut gene_name_controller,
            &mut gene_name_overlays,
            &selected_gene_name_match,
            fixture.context.graph().conn(),
            fixture.workspace(),
        )
        .expect("should activate selected gene_name-only file annotation");
        assert!(gene_name_controller.cursor.node_idx().is_some());
        assert!(matches!(
            &gene_name_overlays[0].content,
            OverlayContent::Span(span)
                if span
                    .segments
                    .iter()
                    .map(|segment| segment.end - segment.start)
                    .sum::<i64>() == 6
        ));
        assert!(resolve_region_search_matches(&fixture.request(), "private-id-gene-name").is_err());

        let db_xref_matches = resolve_region_search_matches(&fixture.request(), "GeneID:123")
            .expect("should search db_xref values containing colons literally");
        assert!(db_xref_matches.iter().any(|search_match| {
            search_match
                .label
                .starts_with("GeneID:123 (file annotation:")
        }));

        let sliced_matches = resolve_region_search_matches(&fixture.request(), "gene-a0001:3-5")
            .expect("should resolve coordinates relative to a file annotation");
        assert_eq!(sliced_matches.len(), 1);
        let sliced_span = span_for_match(&fixture, &sliced_matches[0]);
        assert!(sliced_span.segments.len() > 1);
        assert_eq!(
            sliced_span
                .segments
                .iter()
                .map(|segment| segment.end - segment.start)
                .sum::<i64>(),
            3
        );
        assert_eq!(
            sliced_span
                .segments
                .iter()
                .map(|segment| (segment.start, segment.end))
                .collect::<Vec<_>>(),
            vec![(6, 8), (0, 1)]
        );

        let ambiguous_colon_matches = resolve_region_search_matches(&fixture.request(), "chr1:5")
            .expect("should retain coordinate and literal file matches for a colon alias");
        assert!(
            ambiguous_colon_matches
                .iter()
                .any(|search_match| search_match.label == "chr1:5 (path)")
        );
        assert!(ambiguous_colon_matches.iter().any(|search_match| {
            search_match
                .label
                .starts_with("chr1:5 (file annotation: genes.gff3)")
        }));

        assert!(resolve_region_search_matches(&fixture.request(), "private-id-a").is_err());
        assert_eq!(fixture.index_cache.borrow().len(), 1);
    }

    #[test]
    fn test_multi_record_file_annotation_preserves_record_and_slice_order() {
        let mut fixture = search_request_fixture();
        add_indexed_annotation_file(
            &mut fixture,
            "multipart.gff3",
            "gff3",
            concat!(
                "##gff-version 3\n",
                "m123\tgen-test\tgene\t5\t12\t.\t+\t.\tID=forward-a;Name=forward-multipart\n",
                "m123\tgen-test\tgene\t5\t12\t.\t-\t.\tID=reverse-a;Name=reverse-multipart\n",
                "m123\tgen-test\texon\t20\t22\t.\t+\t.\tID=forward-b;Name=forward-multipart\n",
                "m123\tgen-test\texon\t20\t22\t.\t-\t.\tID=reverse-b;Name=reverse-multipart\n",
            ),
            true,
        );

        for (name, slice_query, slice_bounds, expected_full_segments, expected_slice_segments) in [
            (
                "forward-multipart",
                "forward-multipart:8-9",
                (7, 9),
                vec![(4, 12, Strand::Forward), (19, 22, Strand::Forward)],
                vec![(11, 12, Strand::Forward), (19, 20, Strand::Forward)],
            ),
            (
                "reverse-multipart",
                "reverse-multipart:3-4",
                (2, 4),
                vec![(19, 22, Strand::Reverse), (4, 12, Strand::Reverse)],
                vec![(19, 20, Strand::Reverse), (11, 12, Strand::Reverse)],
            ),
        ] {
            let request = fixture.request();
            let matches = resolve_region_search_matches(&request, name)
                .expect("should find the indexed multipart annotation");
            let file_match = matches
                .into_iter()
                .find(|search_match| search_match.needs_selection_resolution())
                .expect("should return a selectable file annotation");
            let selected_match = resolve_selected_search_match(&request, file_match)
                .expect("should compose the selected file records");
            let resolved_region = selected_match
                .resolved_region()
                .expect("should resolve the multipart annotation");
            assert_eq!((resolved_region.start, resolved_region.end), (0, 11));
            let full_span = span_for_match(&fixture, &selected_match);
            assert_eq!(
                full_span
                    .segments
                    .iter()
                    .map(|segment| (segment.start, segment.end, segment.strand))
                    .collect::<Vec<_>>(),
                expected_full_segments
            );

            let slice_match = resolve_region_search_matches(&request, slice_query)
                .expect("should resolve an annotation-relative slice")
                .into_iter()
                .find(|search_match| search_match.needs_selection_resolution())
                .expect("should return the sliced file annotation");
            let sliced_match = resolve_selected_search_match(&request, slice_match)
                .expect("should resolve the annotation-relative slice");
            let sliced_region = sliced_match
                .resolved_region()
                .expect("should resolve sliced annotation match");
            assert_eq!((sliced_region.start, sliced_region.end), slice_bounds);
            let sliced_span = span_for_match(&fixture, &sliced_match);
            assert_eq!(
                sliced_span
                    .segments
                    .iter()
                    .map(|segment| (segment.start, segment.end, segment.strand))
                    .collect::<Vec<_>>(),
                expected_slice_segments,
                "the annotation-relative slice should cross the boundary between the two source records"
            );
        }
    }

    #[test]
    fn test_simple_gff_lowercase_name_search_activates_file_annotation() {
        let mut fixture = search_request_fixture();
        let annotation_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");
        gen_models::annotations::add_annotation_file(
            &fixture.context,
            annotation_path
                .to_str()
                .expect("should encode simple GFF fixture path"),
            Some("gff3"),
            None,
            Some("simple.gff"),
            Some("add simple GFF viewer search fixture"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should add and index the simple GFF fixture");
        fixture.annotation_files = crate::views::annotation_files::load_annotation_file_entries(
            fixture.context.graph().conn(),
            None,
        );
        let annotation_file = fixture
            .annotation_files
            .iter()
            .find(|entry| entry.display_name == "simple.gff")
            .expect("should load simple GFF annotation file entry");
        fixture
            .active_annotation_files
            .insert(annotation_file.file_addition.id);

        let matches = resolve_region_search_matches(&fixture.request(), "FoObAr")
            .expect("should find a lowercase GFF name attribute case-insensitively");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].label, "foobar (file annotation: simple.gff)");
        assert!(matches[0].needs_selection_resolution());
        let selected_match = resolve_selected_search_match(&fixture.request(), matches[0].clone())
            .expect("should resolve the selected lowercase-name annotation");
        let span = span_for_match(&fixture, &selected_match);
        assert_eq!(
            span.segments
                .iter()
                .map(|segment| segment.end - segment.start)
                .sum::<i64>(),
            16
        );

        let mut controller = create_gen_graph_controller(fixture.graph.clone());
        let mut overlays = Vec::new();
        activate_search_match(
            &mut controller,
            &mut overlays,
            &selected_match,
            fixture.context.graph().conn(),
            fixture.workspace(),
        )
        .expect("should activate selected simple GFF file annotation");
        assert!(controller.cursor.node_idx().is_some());
        assert!(matches!(
            &overlays[0].content,
            OverlayContent::Span(span)
                if span
                    .segments
                    .iter()
                    .map(|segment| segment.end - segment.start)
                    .sum::<i64>() == 16
        ));
        assert!(resolve_region_search_matches(&fixture.request(), "gene-a0001").is_err());
    }

    #[test]
    fn test_region_search_uses_only_active_files_and_keeps_model_matches() {
        let mut fixture = search_request_fixture();
        let active_id = add_indexed_annotation_file(
            &mut fixture,
            "active.gff3",
            "gff3",
            "m123\tgen-test\tgene\t5\t20\t.\t+\t.\tID=active-private-id;Name=active-marker\n",
            true,
        );
        add_indexed_annotation_file(
            &mut fixture,
            "inactive.gff3",
            "gff3",
            "m123\tgen-test\tgene\t5\t20\t.\t+\t.\tID=inactive-private-id;Name=inactive-marker\n",
            false,
        );

        assert!(resolve_region_search_matches(&fixture.request(), "inactive-marker").is_err());
        let active_entry = fixture
            .annotation_files
            .iter()
            .find(|entry| entry.file_addition.id == active_id)
            .expect("should find active annotation file");
        let active_identity = super::indexed_source_identity(active_entry)
            .expect("active file should have recorded indexes");
        let inactive_entry = fixture
            .annotation_files
            .iter()
            .find(|entry| entry.display_name == "inactive.gff3")
            .expect("should find inactive annotation file");
        let inactive_identity = super::indexed_source_identity(inactive_entry)
            .expect("inactive file should have recorded indexes");
        assert!(fixture.index_cache.borrow().contains_key(&active_identity));
        assert!(
            !fixture
                .index_cache
                .borrow()
                .contains_key(&inactive_identity)
        );
        let active_matches = resolve_region_search_matches(&fixture.request(), "active-marker")
            .expect("should search the active annotation file");
        assert_eq!(active_matches.len(), 1);

        let broken_index_directory = tempdir().expect("should create missing index directory");
        let missing_index_path = broken_index_directory.path().join("missing.gai");
        let active_entry = fixture
            .annotation_files
            .iter_mut()
            .find(|entry| entry.file_addition.id == active_id)
            .expect("should find active annotation file");
        let attribute_index = active_entry
            .attribute_index_file_addition
            .as_mut()
            .expect("active file should have recorded attribute index");
        attribute_index.asset_uri = format!("file://{}", missing_index_path.display());
        attribute_index.checksum = None;
        fixture.index_cache.borrow_mut().clear();
        let model_matches = resolve_region_search_matches(&fixture.request(), "model-gene")
            .expect("a broken selected file index should not hide model results");
        assert!(
            model_matches
                .iter()
                .any(|search_match| search_match.label == "model-gene (model annotation)")
        );
        assert!(
            fixture
                .index_cache
                .borrow()
                .values()
                .any(|source| matches!(source, super::CachedIndexedSource::Failed(_)))
        );

        let active_entry = fixture
            .annotation_files
            .iter_mut()
            .find(|entry| entry.file_addition.id == active_id)
            .expect("should find active annotation file");
        active_entry.attribute_index_file_addition = None;
        fixture.index_cache.borrow_mut().clear();
        let model_matches = resolve_region_search_matches(&fixture.request(), "model-gene")
            .expect("a selected file without recorded indexes should not hide model results");
        assert!(
            model_matches
                .iter()
                .any(|search_match| search_match.label == "model-gene (model annotation)")
        );
    }

    #[test]
    fn test_indexed_contig_lookup_uses_case_insensitive_gai_match() {
        let mut fixture = search_request_fixture();
        add_indexed_annotation_file(
            &mut fixture,
            "uppercase-contig.gff3",
            "gff3",
            "M123\tgen-test\tgene\t5\t20\t.\t+\t.\tID=private-id;Name=case-contig-marker\n",
            true,
        );
        let entry = fixture
            .annotation_files
            .iter()
            .find(|entry| entry.display_name == "uppercase-contig.gff3")
            .expect("should find uppercase-contig annotation file");
        let reference_names = HashSet::from(["m123".to_string()]);
        let mut cache = fixture.index_cache.borrow_mut();

        let candidates = query_annotation_file(
            fixture.workspace(),
            entry,
            "case-contig-marker",
            &reference_names,
            &mut cache,
        )
        .expect("should query an indexed annotation on a case-insensitive contig match");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].annotation_name, "case-contig-marker");
        assert!(candidates[0].raw_records[0].starts_with("M123\t"));
    }

    #[test]
    fn test_bed4_name_search_pads_selected_records_for_annotation_reader() {
        let mut fixture = search_request_fixture();
        add_indexed_annotation_file(
            &mut fixture,
            "genes.bed",
            "bed",
            "m123\t4\t20\tbed-feature\n",
            true,
        );

        let matches = resolve_region_search_matches(&fixture.request(), "bed-feature")
            .expect("should search BED name field and translate a BED4 hit");
        assert_eq!(matches.len(), 1);
        assert!(
            matches[0]
                .label
                .starts_with("bed-feature (file annotation:")
        );
        assert!(matches[0].needs_selection_resolution());
        let span = span_for_match(&fixture, &matches[0]);
        assert_eq!(span.segments.len(), 1);
        assert_eq!((span.segments[0].start, span.segments[0].end), (4, 20));
    }

    #[test]
    fn test_remove_search_overlay_preserves_path_and_track_overlays() {
        let span = AnnotationSpan {
            id: gen_core::HashId::convert_str("overlay-span"),
            name: "overlay".to_string(),
            segments: vec![],
        };
        let mut overlays = vec![
            GraphOverlay {
                content: OverlayContent::Span(span.clone()),
                source: OverlaySource::Track("gff".to_string()),
                style: PathStyle::new(Color::Cyan),
            },
            GraphOverlay {
                content: OverlayContent::Path(vec![]),
                source: OverlaySource::Path,
                style: PathStyle::new(Color::Blue),
            },
            GraphOverlay {
                content: OverlayContent::Span(span),
                source: OverlaySource::Adhoc,
                style: PathStyle::new(Color::Yellow),
            },
            GraphOverlay {
                content: OverlayContent::Span(AnnotationSpan {
                    id: gen_core::HashId::convert_str("search-span"),
                    name: "search".to_string(),
                    segments: vec![],
                }),
                source: OverlaySource::Search,
                style: PathStyle::new(Color::LightYellow),
            },
        ];

        remove_search_overlay(&mut overlays);

        assert_eq!(overlays.len(), 3);
        assert!(
            overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Track(_)))
        );
        assert!(
            overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Path))
        );
        assert!(
            overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Adhoc))
        );
        assert!(
            !overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Search))
        );
    }
}
