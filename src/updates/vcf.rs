use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    fmt::Debug,
    io, str,
};

use gen_core::{HashId, NodeIntervalBlock, PathBlock, Strand, is_terminal};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange, BlockGroupData, PathCache},
    block_group_edge::BlockGroupEdge,
    db::{DbContext, GraphConnection},
    edge::EdgeData,
    errors::{BlockGroupError, NodeError, OperationError, PathError, SampleError, SequenceError},
    file_types::FileTypes,
    node::Node,
    operations::{OperationFile, OperationInfo, OperationSummary},
    path::Path,
    reference_alias::ReferenceAlias,
    region::{Region, ResolvedGenRegion, ResolvedRegionKind, resolve_path},
    sample::Sample,
    sequence::Sequence,
};
use intervaltree::IntervalTree;
use noodles::{
    vcf,
    vcf::variant::{
        Record,
        record::{
            AlternateBases,
            info::field::Value as InfoValue,
            samples::{
                Sample as NoodlesSample,
                series::{Value, value::genotype::Phasing},
            },
        },
    },
};
use regex::{self, Regex};
use thiserror::Error;

use crate::{
    parse_genotype,
    progress_bar::{add_saving_operation_bar, get_handler, get_progress_bar},
};

const VCF_CHANGE_APPLY_CHUNK_SIZE: usize = 5_000;

#[derive(Debug)]
struct BlockGroupCache<'a> {
    pub cache: HashMap<BlockGroupData<'a>, Vec<HashId>>,
    pub conn: &'a GraphConnection,
}

impl<'a> BlockGroupCache<'_> {
    pub fn new(conn: &GraphConnection) -> BlockGroupCache<'_> {
        BlockGroupCache {
            cache: HashMap::<BlockGroupData, Vec<HashId>>::new(),
            conn,
        }
    }

    pub fn lookup(
        block_group_cache: &mut BlockGroupCache<'a>,
        collection_name: &'a str,
        sample_name: &'a str,
        name: String,
        parent_samples: &[String],
    ) -> Result<Vec<HashId>, BlockGroupError> {
        let block_group_key = BlockGroupData {
            collection_name,
            sample_name,
            name: name.clone(),
        };
        let block_group_lookup = block_group_cache.cache.get(&block_group_key);
        if let Some(block_group_id) = block_group_lookup {
            Ok(block_group_id.clone())
        } else {
            let result = BlockGroup::get_or_create_sample_block_groups(
                block_group_cache.conn,
                collection_name,
                sample_name,
                &name,
                parent_samples.to_vec(),
            )?;

            let block_group_ids = result
                .iter()
                .map(|block_group| block_group.id)
                .collect::<Vec<_>>();
            block_group_cache
                .cache
                .insert(block_group_key, block_group_ids.clone());
            Ok(block_group_ids)
        }
    }
}

/// Identify incomplete, conflicting, or unsupported calls that prevent path inference.
fn ambiguous_call(alleles: &[Option<usize>], alternates: &[&str], cnv: &Regex) -> bool {
    let Some(Some(allele)) = alleles.first() else {
        return true;
    };
    alleles.iter().any(|value| value != &Some(*allele))
        || (*allele != 0
            && alternates.get(*allele - 1).is_none_or(|sequence| {
                *sequence == "*"
                    || sequence.contains(['[', ']'])
                    || (sequence.starts_with('<') && !cnv.is_match(sequence))
            }))
}

fn append_parent_range(blocks: &[PathBlock], start: i64, end: i64, output: &mut Vec<PathBlock>) {
    let first = blocks.partition_point(|block| block.path_end <= start);
    // Adjacent edits return to and leave the same parent coordinate. Keep that
    // zero-length slice so their existing edges meet without a new shortcut edge.
    // At the sequence boundaries the variant edges already connect to terminals.
    if start == end && start > 0 {
        if let Some(block) = blocks.get(first) {
            let mut junction = block.clone();
            junction.sequence_start += start - block.path_start;
            junction.sequence_end = junction.sequence_start;
            junction.path_start = start;
            junction.path_end = end;
            output.push(junction);
        }
        return;
    }
    for block in blocks[first..]
        .iter()
        .take_while(|block| block.path_start < end)
    {
        let mut slice = block.clone();
        slice.sequence_start += start.max(block.path_start) - block.path_start;
        slice.sequence_end -= block.path_end - end.min(block.path_end);
        if slice.sequence_start <= slice.sequence_end {
            output.push(slice);
        }
    }
}

// Terminal coordinates do not describe sequence. Existing edges may use different
// sentinel coordinates for the same start or end connection.
fn path_edge_key(mut edge: EdgeData) -> EdgeData {
    if is_terminal(edge.source_node_id) {
        edge.source_coordinate = 0;
    }
    if is_terminal(edge.target_node_id) {
        edge.target_coordinate = 0;
    }
    edge
}

/// Assemble a single traversal in parent coordinates before persisting any path changes.
///
/// Path inference deliberately skips multiple parent block groups, multiple parent
/// paths, updates to existing block groups, --inplace updates, and parent paths
/// containing any non-forward strand. The caller excludes all but the strand case
/// before invoking this function. It also requires a parent path and rejects missing
/// or multi-allele calls, unsupported alleles, and conflicting VCF records.
///
/// After deduplicating identical edits, this function returns Ok(()) without changing
/// the inherited path if edits overlap or share a start coordinate, a parent block
/// is not forward-stranded, or an edit starts before the assembled position (including
/// before coordinate zero) or ends beyond the parent path. These are conservative
/// skips: graph updates remain valid even when we cannot confidently infer a path.
/// It also skips if any required connection is absent from the child's existing
/// block-group edges or those edges fail path validation. Adjacent edits retain
/// zero-length parent junctions so their existing variant edges can form a path.
/// This method only writes the path; it never creates edges or their associations.
/// Otherwise, Ok(()) means the inherited path was replaced under the same name with
/// the assembled sample traversal.
fn record_sample_path(
    conn: &GraphConnection,
    path: &Path,
    changes: &[BlockGroupChange],
) -> Result<(), VcfError> {
    let mut edits = changes.iter().collect::<Vec<_>>();
    edits.sort_by_key(|change| (change.region.start, change.region.end, change.block.node_id));
    edits.dedup_by_key(|change| (change.region.start, change.region.end, change.block.node_id));
    if edits.windows(2).any(|pair| {
        pair[1].region.start < pair[0].region.end || pair[1].region.start == pair[0].region.start
    }) {
        return Ok(());
    }
    let parent_blocks = path.coordinate_blocks(conn, None);
    if parent_blocks
        .iter()
        .any(|block| block.strand != Strand::Forward)
    {
        return Ok(());
    }
    let length = parent_blocks
        .last()
        .expect("should have an end block")
        .path_start;
    let mut blocks = vec![parent_blocks[0].clone()];
    let mut position = 0;
    for change in edits {
        if change.region.start < position || change.region.end > length {
            return Ok(());
        }
        append_parent_range(
            &parent_blocks[1..parent_blocks.len() - 1],
            position,
            change.region.start,
            &mut blocks,
        );
        if change.block.sequence_start < change.block.sequence_end {
            blocks.push(change.block.clone());
        }
        position = change.region.end;
    }
    append_parent_range(
        &parent_blocks[1..parent_blocks.len() - 1],
        position,
        length,
        &mut blocks,
    );
    blocks.push(
        parent_blocks
            .last()
            .expect("should have an end block")
            .clone(),
    );
    let mut existing_edges = HashMap::new();
    for augmented_edge in BlockGroupEdge::edges_for_block_group(conn, &path.block_group_id, None) {
        let edge = augmented_edge.edge;
        existing_edges
            .entry(path_edge_key(EdgeData::from(&edge)))
            // Genotype copies and equivalent terminal connections can share a key.
            .and_modify(|edge_id: &mut HashId| *edge_id = (*edge_id).min(edge.id))
            .or_insert(edge.id);
    }
    let mut edge_ids = Vec::new();
    for pair in blocks.windows(2) {
        let source = &pair[0];
        let target = &pair[1];
        let key = path_edge_key(EdgeData {
            source_node_id: source.node_id,
            source_coordinate: source.sequence_end,
            source_strand: source.strand,
            target_node_id: target.node_id,
            target_coordinate: target.sequence_start,
            target_strand: target.strand,
        });
        let Some(edge_id) = existing_edges.get(&key) else {
            return Ok(());
        };
        edge_ids.push(*edge_id);
    }
    match Path::validate_edges(conn, &edge_ids, &path.block_group_id) {
        Ok(()) => {}
        Err(PathError::Invalid(_)) => return Ok(()),
        Err(error) => return Err(error.into()),
    }
    Path::delete(conn, &path.name, &path.block_group_id);
    Path::create(conn, &path.name, &path.block_group_id, &edge_ids)?;
    Ok(())
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SequenceKey<'a> {
    sequence_type: &'a str,
    sequence: String,
}

#[derive(Debug)]
pub struct SequenceCache<'a> {
    pub cache: HashMap<SequenceKey<'a>, Sequence>,
    pub conn: &'a GraphConnection,
}

impl<'a> SequenceCache<'_> {
    pub fn new(conn: &GraphConnection) -> SequenceCache<'_> {
        SequenceCache {
            cache: HashMap::<SequenceKey, Sequence>::new(),
            conn,
        }
    }

    pub fn lookup(
        sequence_cache: &mut SequenceCache<'a>,
        sequence_type: &'a str,
        sequence: String,
    ) -> Result<Sequence, SequenceError> {
        let sequence_key = SequenceKey {
            sequence_type,
            sequence: sequence.clone(),
        };
        let sequence_lookup = sequence_cache.cache.get(&sequence_key);
        if let Some(found_sequence) = sequence_lookup {
            Ok(found_sequence.clone())
        } else {
            let new_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(&sequence)
                .save(sequence_cache.conn)?;

            sequence_cache
                .cache
                .insert(sequence_key, new_sequence.clone());
            Ok(new_sequence)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_change(
    mut region: ResolvedGenRegion,
    ids: Option<String>,
    ref_start: i64,
    ref_end: i64,
    chromosome_index: i64,
    phased: i64,
    block_sequence: String,
    sequence_length: i64,
    node_id: HashId,
    preserve_edge: bool,
) -> Result<BlockGroupChange, BlockGroupError> {
    let (start, end) = region
        .offset_range(ref_start, ref_end)
        .map_err(|err| BlockGroupError::ChangeOutOfBounds(err.to_string()))?;
    region.start = start;
    region.end = end;
    let new_block = PathBlock {
        node_id,
        block_sequence,
        sequence_start: 0,
        sequence_end: sequence_length,
        path_start: ref_start,
        path_end: ref_end,
        strand: Strand::Forward,
    };
    Ok(BlockGroupChange {
        region,
        path_accession: ids,
        block: new_block,
        chromosome_index,
        phased,
        preserve_edge,
    })
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip(
        path_region_cache,
        parent_bg_cache,
        sequence_cache,
        conn,
        collection_name,
        sample_name,
        sample_bg_id,
        seq_name,
        ids,
        alt_seq
    ))
)]
#[allow(clippy::too_many_arguments)]
fn prepare_vcf_entry(
    path_region_cache: &mut HashMap<(HashId, String), ResolvedGenRegion>,
    parent_bg_cache: &mut HashMap<HashId, Option<HashId>>,
    sequence_cache: &mut SequenceCache<'_>,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    sample_bg_id: &HashId,
    seq_name: &str,
    ids: Option<String>,
    ref_start: i64,
    ref_end: i64,
    chromosome_index: i64,
    phased: i64,
    alt_seq: String,
    has_ref: bool,
) -> Result<VcfEntry, VcfError> {
    let path_region = lookup_path_region(
        path_region_cache,
        conn,
        collection_name,
        sample_name,
        sample_bg_id,
        seq_name,
    )?;
    let source_bg_id = lookup_parent_bg_id(parent_bg_cache, conn, sample_bg_id)?;
    let source_region = lookup_path_region(
        path_region_cache,
        conn,
        collection_name,
        sample_name,
        source_bg_id.as_ref().unwrap_or(sample_bg_id),
        seq_name,
    )?;
    let sequence = SequenceCache::lookup(sequence_cache, "DNA", alt_seq)?;
    let sequence_string = sequence.get_sequence(None, None)?;
    let source_path_id = source_region.path.as_ref().unwrap().id;
    let node_id = Node::create(
        conn,
        &sequence.hash,
        &HashId::convert_str(&format!(
            "{path_id}:{ref_start}-{ref_end}->{sequence_hash}",
            path_id = source_path_id,
            sequence_hash = sequence.hash
        )),
    )?;
    let change = prepare_change(
        path_region,
        ids,
        ref_start,
        ref_end,
        chromosome_index,
        phased,
        sequence_string.clone(),
        sequence_string.len() as i64,
        node_id,
        has_ref,
    )?;
    Ok(VcfEntry {
        sample_name: sample_name.to_string(),
        change,
    })
}

fn lookup_parent_bg_id(
    parent_bg_cache: &mut HashMap<HashId, Option<HashId>>,
    conn: &GraphConnection,
    sample_bg_id: &HashId,
) -> Result<Option<HashId>, VcfError> {
    if let Some(parent_bg_id) = parent_bg_cache.get(sample_bg_id) {
        return Ok(*parent_bg_id);
    }

    let parent_bg_id = BlockGroup::get_by_id(conn, sample_bg_id, None)?.parent_block_group_id;
    parent_bg_cache.insert(*sample_bg_id, parent_bg_id);
    Ok(parent_bg_id)
}

fn lookup_path_region(
    path_region_cache: &mut HashMap<(HashId, String), ResolvedGenRegion>,
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    block_group_id: &HashId,
    seq_name: &str,
) -> Result<ResolvedGenRegion, BlockGroupError> {
    let cache_key = (*block_group_id, seq_name.to_string());
    Ok(match path_region_cache.entry(cache_key) {
        Entry::Occupied(entry) => entry.get().clone(),
        Entry::Vacant(entry) => entry
            .insert(
                resolve_path(
                    &Region {
                        name: seq_name.to_string(),
                        start: None,
                        end: None,
                    },
                    conn,
                    collection_name,
                    sample_name,
                )
                .map_err(|err| BlockGroupError::ChangeOutOfBounds(err.to_string()))?,
            )
            .clone(),
    })
}

#[derive(Debug)]
struct VcfEntry {
    sample_name: String,
    change: BlockGroupChange,
}

#[derive(Error, Debug, PartialEq)]
pub enum VcfError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Sample Error: {0}")]
    SampleError(#[from] SampleError),
    #[error("Invalid Record: {0}")]
    InvalidRecord(String),
    #[error("Node creation error: {0}")]
    NodeError(#[from] NodeError),
    #[error("Block group creation error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("Sequence save error: {0}")]
    SequenceError(#[from] SequenceError),
    #[error("Path error: {0}")]
    PathError(#[from] PathError),
}

fn resolve_parent_samples(
    conn: &GraphConnection,
    sample_name: &str,
    explicit_parent_samples: &[String],
    resolved_parent_samples: &mut HashMap<String, Vec<String>>,
) -> Vec<String> {
    resolved_parent_samples
        .entry(sample_name.to_string())
        .or_insert_with(|| {
            if explicit_parent_samples.is_empty() {
                Sample::get_parent_names(conn, sample_name, None)
            } else {
                explicit_parent_samples.to_vec()
            }
        })
        .clone()
}

#[cfg_attr(
    feature = "profiling",
    tracing::instrument(skip(context, vcf_path, parent_samples))
)]
pub fn update_with_vcf(
    context: &DbContext,
    vcf_path: &String,
    collection_name: &str,
    fixed_genotype: String,
    fixed_sample: Option<&str>,
    parent_samples: Vec<String>,
    in_place: bool,
) -> Result<(OperationSummary, Vec<String>), VcfError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let cnv_re = Regex::new(r"(?x)<CN(?P<count>\d+)>").unwrap();

    let mut reader = vcf::io::reader::Builder::default()
        .build_from_path(vcf_path)
        .expect("Unable to parse");
    let header = reader.read_header().unwrap();
    let sample_names = header.sample_names();
    let mut genotype = vec![];
    if !fixed_genotype.is_empty() {
        genotype = parse_genotype(&fixed_genotype);
    }

    // Cache a bunch of data ahead of making changes
    let mut block_group_cache = BlockGroupCache::new(conn);
    let mut path_cache = PathCache::new(conn);
    let mut sequence_cache = SequenceCache::new(conn);
    let mut accession_cache = HashMap::new();
    let mut path_region_cache: HashMap<(HashId, String), ResolvedGenRegion> = HashMap::new();
    let mut parent_bg_cache: HashMap<HashId, Option<HashId>> = HashMap::new();

    let existing_block_groups = BlockGroup::select(conn)
        .collection_name(collection_name)
        .load()
        .map_err(BlockGroupError::from)?
        .into_iter()
        .map(|block_group| block_group.id)
        .collect::<HashSet<_>>();
    let mut ambiguous_samples = HashSet::<(String, String)>::new();
    let mut called_regions = HashMap::<(String, String), Vec<(i64, i64, String)>>::new();
    let mut paths_to_record = Vec::new();
    let mut changes: HashMap<(Path, String), Vec<BlockGroupChange>> = HashMap::new();

    let mut resolved_parent_samples: HashMap<String, Vec<String>> = HashMap::new();
    let mut created_samples: HashSet<&str> = HashSet::new();

    let mut block_group_names = vec![];
    for parent_sample in &parent_samples {
        let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample, None);
        block_group_names.extend(block_groups.iter().map(|bg| bg.name.clone()));
    }
    let references_by_alias =
        ReferenceAlias::get_references_by_alias(conn, block_group_names, None).unwrap();

    let _ = progress_bar.println("Parsing VCF for changes.");

    let bar = progress_bar.add(get_progress_bar(None));

    bar.set_message("Records Parsed");
    for result in reader.records() {
        let record = result.unwrap();
        let seq_name: String = record.reference_sequence_name().to_string();
        let seq_name = references_by_alias
            .get(&seq_name)
            .unwrap_or(&seq_name)
            .to_string();
        let ref_seq = record.reference_bases();
        // this converts the coordinates to be zero based, start inclusive, end exclusive
        let ref_end = record.variant_end(&header).unwrap().get() as i64;
        let alt_bases = record.alternate_bases();
        let alt_alleles: Vec<_> = alt_bases.iter().collect::<io::Result<_>>().unwrap();
        let mut vcf_entries = vec![];
        let accession_name: Option<String> = match record.info().get(&header, "GAN") {
            Some(v) => match v.unwrap().unwrap() {
                InfoValue::String(v) => Some(v.to_string()),
                _ => None,
            },
            _ => None,
        };
        let accession_allele: i32 = match record.info().get(&header, "GAA") {
            Some(v) => match v.unwrap().unwrap() {
                InfoValue::Integer(v) => v,
                _ => 0,
            },
            _ => 0,
        };

        if let Some(fixed_sample) = fixed_sample.filter(|_| !genotype.is_empty()) {
            let sample_parent_samples = resolve_parent_samples(
                conn,
                fixed_sample,
                &parent_samples,
                &mut resolved_parent_samples,
            );
            if !created_samples.contains(fixed_sample) {
                Sample::get_or_create_child(
                    conn,
                    collection_name,
                    fixed_sample,
                    sample_parent_samples.clone(),
                )?;
                created_samples.insert(fixed_sample);
            }
            let sample_bg_ids = BlockGroupCache::lookup(
                &mut block_group_cache,
                collection_name,
                fixed_sample,
                seq_name.clone(),
                &sample_parent_samples,
            )
            .expect("can't find sample bg....check this out more");
            let alleles = genotype
                .iter()
                .map(|call| {
                    call.as_ref()
                        .and_then(|call| usize::try_from(call.allele).ok())
                })
                .collect::<Vec<_>>();
            if ambiguous_call(&alleles, &alt_alleles, &cnv_re) {
                ambiguous_samples.insert((fixed_sample.to_string(), seq_name.clone()));
            } else {
                let allele = alleles[0].expect("should have a called allele");
                let sequence = if allele == 0 {
                    ref_seq
                } else {
                    alt_alleles[allele - 1]
                };
                called_regions
                    .entry((fixed_sample.to_string(), seq_name.clone()))
                    .or_default()
                    .push((
                        (record.variant_start().unwrap().unwrap().get() - 1) as i64,
                        ref_end,
                        sequence.to_string(),
                    ));
            }
            let has_ref = genotype.iter().any(|gt| {
                if let Some(gt) = gt {
                    gt.allele == 0
                } else {
                    false
                }
            });
            for (chromosome_index, genotype) in genotype.iter().enumerate() {
                if let Some(gt) = genotype {
                    let allele_accession = accession_name
                        .clone()
                        .filter(|_| gt.allele as i32 == accession_allele);
                    let mut ref_start = (record.variant_start().unwrap().unwrap().get() - 1) as i64;
                    if gt.allele != 0 {
                        let allele_index = usize::try_from(gt.allele - 1)
                            .expect("alternate genotype allele should be positive");
                        let mut alt_seq = alt_alleles[allele_index].to_string();
                        let mut is_cnv = false;
                        if alt_seq.starts_with("<") {
                            if let Some(cap) = cnv_re.captures(&alt_seq) {
                                let count: usize =
                                    cap["count"].parse().expect("Invalid CN specification");
                                alt_seq = ref_seq.to_string().repeat(count);
                                is_cnv = true;
                            } else {
                                continue;
                            };
                        }
                        // If the alt sequence is a deletion or insertion, we want to remove the base in common in the VCF spec.
                        // So if VCF says ATC -> A or A -> ATTC, we don't want to include the `A` in the alt_seq.
                        if !alt_seq.is_empty() && alt_seq != "*" && alt_seq.len() != ref_seq.len() {
                            if is_cnv {
                                // move past the common regions
                                // ref_start += ref_seq.len() as i64;
                                // alt_seq = alt_seq[ref_seq.len()..].to_string();
                            } else {
                                ref_start += 1;
                                alt_seq = alt_seq[1..].to_string();
                            }
                        }
                        let phased = match gt.phasing {
                            Phasing::Phased => 1,
                            Phasing::Unphased => 0,
                        };
                        if alt_seq == "*" {
                            continue;
                        }
                        for sample_bg_id in &sample_bg_ids {
                            let entry = prepare_vcf_entry(
                                &mut path_region_cache,
                                &mut parent_bg_cache,
                                &mut sequence_cache,
                                conn,
                                collection_name,
                                fixed_sample,
                                sample_bg_id,
                                &seq_name,
                                allele_accession.clone(),
                                ref_start,
                                ref_end,
                                chromosome_index as i64,
                                phased,
                                alt_seq.clone(),
                                has_ref,
                            )?;
                            vcf_entries.push(entry);
                        }
                    } else if let Some(ref_accession) = allele_accession {
                        for sample_bg_id in &sample_bg_ids {
                            let sample_path =
                                PathCache::lookup(&mut path_cache, sample_bg_id, seq_name.clone())?;

                            let key = (sample_path, ref_accession.clone());

                            accession_cache.entry(key).or_insert_with(|| {
                                (ref_start, ref_start + record.reference_bases().len() as i64)
                            });
                        }
                    }
                }
            }
        } else {
            for (sample_index, sample) in record.samples().iter().enumerate() {
                let sample_name: &str = sample_names[sample_index].as_ref();
                let sample_parent_samples = resolve_parent_samples(
                    conn,
                    sample_name,
                    &parent_samples,
                    &mut resolved_parent_samples,
                );
                if !created_samples.contains(sample_name) {
                    Sample::get_or_create_child(
                        conn,
                        collection_name,
                        sample_name,
                        sample_parent_samples.clone(),
                    )?;
                    created_samples.insert(sample_name);
                }
                let sample_bg_ids = BlockGroupCache::lookup(
                    &mut block_group_cache,
                    collection_name,
                    sample_name,
                    seq_name.clone(),
                    &sample_parent_samples,
                )
                .expect("can't find sample bg....check this out more");
                let genotype = sample.get(&header, "GT");
                let alleles = match &genotype {
                    Some(Ok(Some(Value::Genotype(calls)))) => calls
                        .iter()
                        .map(|call| call.ok().and_then(|(allele, _)| allele))
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };
                if ambiguous_call(&alleles, &alt_alleles, &cnv_re) {
                    ambiguous_samples.insert((sample_name.to_string(), seq_name.clone()));
                } else {
                    let allele = alleles[0].expect("should have a called allele");
                    let sequence = if allele == 0 {
                        ref_seq
                    } else {
                        alt_alleles[allele - 1]
                    };
                    called_regions
                        .entry((sample_name.to_string(), seq_name.clone()))
                        .or_default()
                        .push((
                            (record.variant_start().unwrap().unwrap().get() - 1) as i64,
                            ref_end,
                            sequence.to_string(),
                        ));
                }
                if let Some(Ok(Some(Value::Genotype(genotypes)))) = genotype {
                    // what needs to be done is when it is a cnv, we need to check that ref_start is the same for all variants
                    // and calculate is_ref for each variant at the same ref_start. So we need to have 2 end list of of variants
                    // here that is passed to vcf_entry that knows about ref_start.
                    let has_ref = genotypes.iter().any(|gt| matches!(gt, Ok((Some(0), _))));
                    for (chromosome_index, gt) in genotypes.iter().enumerate() {
                        if let Ok((allele, phasing)) = gt {
                            let phased = match phasing {
                                Phasing::Phased => 1,
                                Phasing::Unphased => 0,
                            };
                            let mut ref_start =
                                (record.variant_start().unwrap().unwrap().get() - 1) as i64;
                            if let Some(allele) = allele {
                                let allele_accession = accession_name
                                    .clone()
                                    .filter(|_| allele as i32 == accession_allele);
                                if allele != 0 {
                                    let mut alt_seq = alt_alleles[allele - 1].to_string();
                                    let mut is_cnv = false;
                                    if alt_seq.starts_with("<") {
                                        if let Some(cap) = cnv_re.captures(&alt_seq) {
                                            let count: usize = cap["count"]
                                                .parse()
                                                .expect("Invalid CN specification");
                                            is_cnv = true;
                                            // our ref sequence will be something like "ATC" and our new alt
                                            // sequence will be (ATC)*count. The position provided will be
                                            // the left most base, so the A here.
                                            alt_seq = ref_seq.to_string().repeat(count);
                                        } else {
                                            continue;
                                        }
                                    }
                                    if !alt_seq.is_empty()
                                        && alt_seq != "*"
                                        && alt_seq.len() != ref_seq.len()
                                    {
                                        if is_cnv {
                                            // ref_start += ref_seq.len() as i64;
                                            // alt_seq = alt_seq[ref_seq.len()..].to_string();
                                        } else {
                                            ref_start += 1;
                                            alt_seq = alt_seq[1..].to_string();
                                        }
                                    }
                                    if alt_seq == "*" {
                                        continue;
                                    }
                                    for sample_bg_id in &sample_bg_ids {
                                        let entry = prepare_vcf_entry(
                                            &mut path_region_cache,
                                            &mut parent_bg_cache,
                                            &mut sequence_cache,
                                            conn,
                                            collection_name,
                                            sample_name,
                                            sample_bg_id,
                                            &seq_name,
                                            allele_accession.clone(),
                                            ref_start,
                                            ref_end,
                                            chromosome_index as i64,
                                            phased,
                                            alt_seq.clone(),
                                            has_ref,
                                        )?;
                                        vcf_entries.push(entry);
                                    }
                                } else if let Some(ref_accession) = allele_accession {
                                    for sample_bg_id in &sample_bg_ids {
                                        let sample_path = PathCache::lookup(
                                            &mut path_cache,
                                            sample_bg_id,
                                            seq_name.clone(),
                                        )?;

                                        let key = (sample_path, ref_accession.clone());

                                        accession_cache.entry(key).or_insert_with(|| {
                                            (
                                                ref_start,
                                                ref_start + record.reference_bases().len() as i64,
                                            )
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for vcf_entry in vcf_entries {
            changes
                .entry((
                    vcf_entry.change.region.path.clone().unwrap(),
                    vcf_entry.sample_name,
                ))
                .or_default()
                .push(vcf_entry.change);
        }
        bar.inc(1);
    }
    bar.finish();

    // Include reference calls so a second record cannot silently contradict an edit.
    for (key, mut regions) in called_regions {
        regions.sort_unstable();
        regions.dedup();
        if regions.windows(2).any(|pair| pair[1].0 < pair[0].1) {
            ambiguous_samples.insert(key);
        }
    }

    let bar = progress_bar.add(get_progress_bar(
        changes.values().map(|c| c.len() as u64).sum::<u64>(),
    ));
    bar.set_message("Changes applied");
    let mut summary: HashMap<String, HashMap<String, i64>> = HashMap::new();
    let mut tree_map: HashMap<(HashId, ResolvedRegionKind), IntervalTree<i64, NodeIntervalBlock>> =
        HashMap::new();
    for ((path, sample_name), path_changes) in changes {
        for chunk in path_changes.chunks(VCF_CHANGE_APPLY_CHUNK_SIZE) {
            if in_place {
                let in_place_changes = chunk
                    .iter()
                    .map(|change| {
                        let mut region = change.region.clone();
                        region.kind = ResolvedRegionKind::BlockGroup;
                        region.remove_ambiguous_positions = true;
                        BlockGroupChange {
                            region,
                            path_accession: change.path_accession.clone(),
                            block: change.block.clone(),
                            chromosome_index: change.chromosome_index,
                            phased: change.phased,
                            preserve_edge: change.preserve_edge,
                        }
                    })
                    .collect::<Vec<_>>();
                BlockGroup::insert_changes(
                    conn,
                    context.workspace(),
                    &in_place_changes,
                    Some(&mut tree_map),
                )
                .unwrap();
            } else {
                BlockGroup::insert_changes(conn, context.workspace(), chunk, Some(&mut tree_map))
                    .unwrap();
            }
            bar.inc(chunk.len() as u64);
        }
        let change_count = path_changes.len() as i64;
        // Apply the conservative path inference policies documented on record_sample_path.
        let siblings = block_group_cache.cache.get(&BlockGroupData {
            collection_name,
            sample_name: &sample_name,
            name: path.name.clone(),
        });
        if !in_place
            && !existing_block_groups.contains(&path.block_group_id)
            && !ambiguous_samples.contains(&(sample_name.clone(), path.name.clone()))
            && siblings.is_some_and(|ids| ids.len() == 1)
        {
            let block_group = BlockGroup::get_by_id(conn, &path.block_group_id, None)?;
            if let Some(parent_id) = block_group.parent_block_group_id {
                let parent_paths = Path::select(conn)
                    .block_group_id(parent_id)
                    .load()
                    .map_err(BlockGroupError::from)?;
                if parent_paths.len() == 1 {
                    paths_to_record.push((path.clone(), path_changes));
                }
            }
        }
        summary
            .entry(sample_name)
            .or_default()
            .entry(path.name)
            .or_insert(change_count);
    }
    bar.finish();
    for ((path, accession_name), (acc_start, acc_end)) in accession_cache.iter() {
        BlockGroup::add_accession(
            conn,
            path,
            accession_name,
            *acc_start,
            *acc_end,
            &mut path_cache,
        )?;
    }
    for (path, path_changes) in paths_to_record {
        record_sample_path(conn, &path, &path_changes)?;
    }
    let mut summary_str = "".to_string();
    for (sample_name, sample_changes) in summary.iter() {
        summary_str.push_str(&format!("Sample {sample_name}\n"));
        for (path_name, change_count) in sample_changes.iter() {
            summary_str.push_str(&format!(" {path_name}: {change_count} changes.\n"));
        }
    }

    let bar = add_saving_operation_bar(&progress_bar);
    bar.set_message("Saving operation");
    let operation_summary = OperationSummary::new(
        OperationInfo {
            files: vec![OperationFile::new(vcf_path.to_string()).set_file_type(FileTypes::VCF)],
            description: "vcf_addition".to_string(),
        },
        summary_str,
    );
    bar.finish();
    let output_samples: Vec<String> = created_samples.into_iter().map(String::from).collect();
    Ok((operation_summary, output_samples))
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    #[allow(unused_imports)]
    use std::time;
    use std::{collections::HashSet, path::PathBuf};

    use gen_models::{
        accession::Accession, edge::Edge, node::Node, sample::Sample, sample_lineage::SampleLineage,
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_gen},
    };
    #[test]
    fn test_record_sample_path_only_uses_existing_block_group_edges() {
        for alternate in ["C", ""] {
            let context = setup_gen();
            let conn = context.graph().conn();
            let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
            import_fasta(
                &context,
                &fasta_path.to_string_lossy().into_owned(),
                "test",
                Sample::DEFAULT_NAME,
                false,
                &[],
            )
            .unwrap();
            Sample::get_or_create_child(
                conn,
                "test",
                "child",
                vec![Sample::DEFAULT_NAME.to_string()],
            )
            .unwrap();
            let block_groups = BlockGroup::get_or_create_sample_block_groups(
                conn,
                "test",
                "child",
                "m123",
                vec![Sample::DEFAULT_NAME.to_string()],
            )
            .unwrap();
            let path = Path::select(conn)
                .block_group_id(block_groups[0].id)
                .load()
                .unwrap()
                .remove(0);
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(alternate)
                .save(conn)
                .unwrap();
            let node_id = Node::create(
                conn,
                &sequence.hash,
                &HashId::convert_str("path test variant"),
            )
            .unwrap();
            let change = BlockGroupChange {
                region: ResolvedGenRegion::from_path(conn, path.block_group_id, &path, 1, 2)
                    .unwrap(),
                path_accession: None,
                block: PathBlock {
                    node_id,
                    block_sequence: alternate.to_string(),
                    sequence_start: 0,
                    sequence_end: alternate.len() as i64,
                    path_start: 1,
                    path_end: 2,
                    strand: Strand::Forward,
                },
                chromosome_index: 0,
                phased: 0,
                preserve_edge: false,
            };
            let mut adjacent_change = change.clone();
            adjacent_change.region.start = 2;
            adjacent_change.region.end = 3;
            adjacent_change.block.path_start = 2;
            adjacent_change.block.path_end = 3;
            let changes = [change, adjacent_change];
            let original_edge_ids = Path::edge_ids_for_path(conn, &path.id, None);
            for stage in 0..3 {
                if stage == 1 {
                    // Edges in the database alone must not qualify as child connections.
                    let planned = changes
                        .iter()
                        .flat_map(|change| {
                            BlockGroup::set_up_new_edges(change, &path.intervaltree(conn).unwrap())
                                .unwrap()
                        })
                        .collect::<Vec<_>>();
                    Edge::bulk_create(
                        conn,
                        &planned
                            .iter()
                            .map(|edge| edge.edge_data)
                            .collect::<Vec<_>>(),
                    );
                } else if stage == 2 {
                    BlockGroup::insert_changes(conn, context.workspace(), &changes, None).unwrap();
                }
                let edges_before = Edge::select(conn)
                    .load()
                    .unwrap()
                    .into_iter()
                    .collect::<HashSet<_>>();
                let associations_before = BlockGroupEdge::select(conn)
                    .load()
                    .unwrap()
                    .into_iter()
                    .collect::<HashSet<_>>();
                record_sample_path(conn, &path, &changes).unwrap();
                assert_eq!(
                    Edge::select(conn)
                        .load()
                        .unwrap()
                        .into_iter()
                        .collect::<HashSet<_>>(),
                    edges_before
                );
                assert_eq!(
                    BlockGroupEdge::select(conn)
                        .load()
                        .unwrap()
                        .into_iter()
                        .collect::<HashSet<_>>(),
                    associations_before
                );
                if stage < 2 {
                    assert_eq!(
                        Path::edge_ids_for_path(conn, &path.id, None),
                        original_edge_ids
                    );
                } else {
                    let blocks = path.coordinate_blocks(conn, None);
                    assert!(
                        blocks.iter().any(|block| !is_terminal(block.node_id)
                            && block.path_start == block.path_end)
                    );
                    assert_eq!(
                        path.length(conn, None).unwrap(),
                        32 + 2 * alternate.len() as i64
                    );
                    let tree = path.intervaltree(conn).unwrap();
                    assert_eq!(tree.query_point(1).count(), 1);
                    assert_eq!(
                        path.sequence(conn, context.workspace(), None).unwrap(),
                        if alternate.is_empty() {
                            "AGATCGATCGATCGATCGGGAACACACAGAGA"
                        } else {
                            "ACCGATCGATCGATCGATCGGGAACACACAGAGA"
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn test_vcf_path_connects_adjacent_variants_through_parent_junction() {
        assert_sample_path(
            "m123\t2\t.\tT\tC\t.\t.\t.\tGT\t1/1\nm123\t3\t.\tC\tG\t.\t.\t.\tGT\t1/1\n",
            "",
            "ACGGATCGATCGATCGATCGGGAACACACAGAGA",
        );
    }

    #[test]
    fn test_vcf_sample_paths() {
        let cases = [
            ("1/1", "1/1", "1/1", "ACCGGGAGATCGATCGATCGGGAACACACAGAGA"),
            ("1", "1", "1", "ACCGGGAGATCGATCGATCGGGAACACACAGAGA"),
            ("0/0", "0/0", "0/0", "ATCGATCGATCGATCGATCGGGAACACACAGAGA"),
            ("0/1", "1/1", "1/1", "ATCGATCGATCGATCGATCGGGAACACACAGAGA"),
            ("1/2", "1/1", "1/1", "ATCGATCGATCGATCGATCGGGAACACACAGAGA"),
            ("./1", "1/1", "1/1", "ATCGATCGATCGATCGATCGGGAACACACAGAGA"),
            (".", "1/1", "1/1", "ATCGATCGATCGATCGATCGGGAACACACAGAGA"),
        ];
        for (first, second, third, expected) in cases {
            let records = format!(
                "m123\t2\t.\tT\tC,G\t.\t.\t.\tGT\t{first}\nm123\t4\t.\tG\tGGG\t.\t.\t.\tGT\t{second}\nm123\t5\t.\tATC\tA\t.\t.\t.\tGT\t{third}\n"
            );
            assert_sample_path(&records, "", expected);
        }
    }

    fn assert_sample_path(records: &str, fixed_genotype: &str, expected: &str) {
        let context = setup_gen();
        let conn = context.graph().conn();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            &context,
            &fasta_path.to_string_lossy().into_owned(),
            "test",
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let vcf_path = directory.path().join("paths.vcf");
        std::fs::write(&vcf_path, format!(
            "##fileformat=VCFv4.3\n##contig=<ID=m123,length=34>\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n{records}"
        )).unwrap();
        update_with_vcf(
            &context,
            &vcf_path.to_string_lossy().into_owned(),
            "test",
            fixed_genotype.to_string(),
            Some("sample"),
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        let paths = Path::query_for_collection_and_sample(conn, "test", "sample");
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0].sequence(conn, context.workspace(), None).unwrap(),
            expected,
            "records: {records}, fixed genotype: {fixed_genotype}"
        );
        let parent_paths =
            Path::query_for_collection_and_sample(conn, "test", Sample::DEFAULT_NAME);
        assert_eq!(
            parent_paths[0]
                .sequence(conn, context.workspace(), None)
                .unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
        );
    }

    #[test]
    fn test_vcf_path_boundaries_and_duplicate_calls() {
        let records = "m123\t1\t.\tA\tC\t.\t.\t.\tGT\t1/1\nm123\t1\t.\tA\tC\t.\t.\t.\tGT\t1/1\nm123\t34\t.\tA\tAT\t.\t.\t.\tGT\t1/1\n";
        assert_sample_path(records, "", "CTCGATCGATCGATCGATCGGGAACACACAGAGAT");
        assert_sample_path(records, "1/1", "CTCGATCGATCGATCGATCGGGAACACACAGAGAT");
        assert_sample_path(records, "0/1", "ATCGATCGATCGATCGATCGGGAACACACAGAGA");
    }

    #[test]
    fn test_vcf_path_conflicting_and_unsupported_calls() {
        let original = "ATCGATCGATCGATCGATCGGGAACACACAGAGA";
        for record in [
            "m123\t2\t.\tT\tG\t.\t.\t.\tGT\t1/1\n",
            "m123\t2\t.\tT\tC\t.\t.\t.\tGT\t0/0\n",
            "m123\t1\t.\tATC\tA\t.\t.\t.\tGT\t1/1\n",
            "m123\t5\t.\tA\t<DEL>\t.\t.\t.\tGT\t1/1\n",
            "m123\t5\t.\tA\t*\t.\t.\t.\tGT\t1/1\n",
        ] {
            let records = format!("m123\t2\t.\tT\tC\t.\t.\t.\tGT\t1/1\n{record}");
            assert_sample_path(&records, "", original);
        }
    }

    #[test]
    fn test_update_fasta_with_vcf() -> Result<(), VcfError> {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let _operation = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )?;
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                false,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `G1` genotype has no changes
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "G1").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `foo` is homozygous for the first variant and does not contain the second
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "foo").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCATCGATCGATCGATCGGGAACACACAGAGA".to_string(),])
        );

        Ok(())
    }

    #[test]
    fn test_update_fasta_with_complex_vcf() {
        let context = setup_gen();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vcfs/complex.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let _operation = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                false,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `bar` sample has the refrence + a deletion of the C
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "bar").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATGATCGATCGGGAACACACAGAGA".to_string()
            ])
        );
        // `baz` sample has a deletion of CG and an insertion of A
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "baz").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(vec![
                "ATCGATCGATATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCAGATCGATCGGGAACACACAGAGA".to_string(),
            ])
        );
    }

    #[test]
    fn test_update_fasta_with_vcf_custom_genotype() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/general.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let _operation = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0/1".to_string(),
            Some("sample 1"),
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                false,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "sample 1").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(
                [
                    "ATCGATCGATAGAGATCGATCGGGAACACACAGAGA",
                    "ATCATCGATAGAGATCGATCGGGAACACACAGAGA",
                    "ATCGATCGATCGATCGATCGGGAACACACAGAGA",
                    "ATCATCGATCGATCGATCGGGAACACACAGAGA"
                ]
                .iter()
                .map(|v| v.to_string())
            )
        );
    }

    #[test]
    fn test_update_fasta_with_vcf_homozygous_custom_genotype() {
        let context = setup_gen();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/general.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "1/1".to_string(),
            Some("sample 1"),
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "sample 1").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCATCGATAGAGATCGATCGGGAACACACAGAGA".to_string()])
        );
    }

    #[test]
    fn test_error_when_vcf_has_changes_out_of_bounds() {
        let context = setup_gen();
        let _conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vcfs/out_of_bounds.vcf");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let res = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0/1".to_string(),
            Some("sample 1"),
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        );
        assert!(matches!(
            res,
            Err(VcfError::BlockGroupError(
                BlockGroupError::ChangeOutOfBounds(_)
            ))
        ));
    }

    #[test]
    fn test_handles_missing_allele() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple_missing_allele.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        let _operation = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "unknown").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(
                ["ATCGATCGATAGACGATCGATCGGGAACACACAGAGA",]
                    .iter()
                    .map(|v| v.to_string())
            )
        );
    }

    #[test]
    fn test_handles_overlap_allele() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple_overlap.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "foo").id,
                false
            )
            .unwrap(),
            HashSet::from_iter(
                ["ATCATCGATCGATCGATCGGGAACACACAGAGA",]
                    .iter()
                    .map(|v| v.to_string())
            )
        );
    }

    #[test]
    fn test_parses_cnvs() {
        let context = setup_gen();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple_cnv.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "foo").id,
                true
            )
            .unwrap(),
            HashSet::from_iter(vec![
                "ATCGATCGATCGGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCGATCATCATCGATCGGGAACACACAGAGA".to_string()
            ])
        );
    }

    #[test]
    fn test_deduplicates_nodes() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        let nodes = Node::select(conn).load().expect("should load nodes");
        assert_eq!(nodes.len(), 5);

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        assert_eq!(
            Node::select(conn).load().expect("should load nodes").len(),
            5
        );
    }

    #[test]
    fn test_deduplicates_nodes_multiple_paths() {
        let context = setup_gen();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/multiseq.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/multiseq.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        assert_eq!(
            Node::select(conn).load().expect("should load nodes").len(),
            5
        );

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        let nodes = Node::select(conn).load().expect("should load nodes");
        assert_eq!(nodes.len(), 8);

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        assert_eq!(
            Node::select(conn).load().expect("should load nodes").len(),
            8
        );
    }

    #[test]
    #[cfg(feature = "benchmark")]
    #[ignore = "manual benchmark; large fixture is not stable in the all-features test suite"]
    fn test_vcf_import_benchmark() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/chr22_100k_no_samples.vcf.gz");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/chr22.fa.gz");
        let _conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        let s = time::Instant::now();
        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0|1".to_string(),
            Some("test"),
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        let elapsed = s.elapsed().as_secs();
        let max_elapsed = if cfg!(debug_assertions) { 120 } else { 20 };
        assert!(
            elapsed < max_elapsed,
            "VCF import benchmark failed: Elapsed time is {elapsed}."
        );
    }

    #[test]
    fn test_creates_accession_nodes() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/accession.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();
        assert_eq!(
            Accession::select(conn)
                .name("del1")
                .load()
                .expect("should load the named accession")
                .len(),
            1
        );

        assert_eq!(
            Accession::select(conn)
                .name("lp1")
                .load()
                .expect("should load the named accession")
                .len(),
            1
        );
    }

    #[test]
    fn test_disallows_creating_accession_nodes_that_exist() {
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/accession.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        let _operation = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap();

        assert_eq!(
            Accession::select(conn)
                .name("lp1")
                .load()
                .expect("should load the named accession")
                .len(),
            1
        );

        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // This is invalid because lp1 already exists from accession.vcf
        vcf_path.push("fixtures/accession_2_invalid.vcf");

        let err = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            VcfError::BlockGroupError(BlockGroupError::AccessionError(
                gen_models::accession::AccessionError::Duplicate(_)
            ))
        ));
    }

    #[test]
    fn test_changes_in_child_samples() {
        let f0_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_iterative_engineering_1.vcf");
        let f1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_iterative_engineering_2.vcf");
        let f2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_iterative_engineering_3.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        update_with_vcf(
            &context,
            &f0_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            true,
        )
        .unwrap();

        update_with_vcf(
            &context,
            &f1_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec!["f1".to_string()],
            true,
        )
        .unwrap();

        update_with_vcf(
            &context,
            &f2_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec!["f2".to_string()],
            true,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                true,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "f1").id,
                true
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCTCGATCGATCGCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "f2").id,
                true
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCTGGATCGATCGCGGAATCAGAACACACAGGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &get_sample_bg(conn, &collection, "f3").id,
                true
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGGGATCGATCGCTCAGAACACACAGGA".to_string()])
        );
    }

    #[test]
    fn test_update_vcf_uses_lineage_parent_when_parent_sample_is_omitted() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test".to_string();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "reference",
            false,
            &[],
        )
        .unwrap();

        Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "child",
                ..Default::default()
            },
        )
        .unwrap();
        SampleLineage::create(conn, "reference", "child").unwrap();

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0/1".to_string(),
            Some("child"),
            vec![],
            false,
        )
        .unwrap();

        let child_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &get_sample_bg(conn, &collection, "child").id,
            true,
        )
        .unwrap();
        assert_eq!(
            child_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATAGACGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCATCGATAGACGATCGATCGGGAACACACAGAGA".to_string()
            ])
        );
    }

    #[test]
    fn test_update_vcf_with_parents_having_same_reference_names() {
        // Ensure if we have a child sample with multiple parents with the same contig names it works
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test".to_string();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "parent-a",
            false,
            &[],
        )
        .unwrap();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "parent-b",
            false,
            &[],
        )
        .unwrap();

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0/1".to_string(),
            Some("child"),
            vec!["parent-a".to_string(), "parent-b".to_string()],
            false,
        )
        .unwrap();

        let child_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &get_sample_bg(conn, &collection, "child").id,
            true,
        )
        .unwrap();
        assert_eq!(
            child_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATAGACGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCATCGATAGACGATCGATCGGGAACACACAGAGA".to_string()
            ])
        );
    }

    #[test]
    fn test_update_vcf_does_not_overwrite_parent_lineage_for_existing_sample() {
        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test".to_string();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "reference",
            false,
            &[],
        )
        .unwrap();

        Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "child",
                ..Default::default()
            },
        )
        .unwrap();

        update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0/1".to_string(),
            Some("child"),
            vec!["reference".to_string()],
            false,
        )
        .unwrap();

        assert!(SampleLineage::get_parents(conn, "child", None).is_empty());
    }
}
