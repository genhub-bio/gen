use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    fmt::Debug,
    io, str,
};

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange, BlockGroupData, EditBatch, PathCache},
    db::{DbContext, GraphConnection},
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
    // A deletion's node is made when the change is planned, identified by the bases it removes.
    let node_id = if sequence_string.is_empty() {
        HashId::convert_str("")
    } else {
        Node::create(
            conn,
            &sequence.hash,
            &HashId::convert_str(&format!(
                "{path_id}:{ref_start}-{ref_end}->{sequence_hash}",
                path_id = source_path_id,
                sequence_hash = sequence.hash
            )),
        )?
    };
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

    let bar = progress_bar.add(get_progress_bar(
        changes.values().map(|c| c.len() as u64).sum::<u64>(),
    ));
    bar.set_message("Changes applied");
    let mut summary: HashMap<String, HashMap<String, i64>> = HashMap::new();
    // One batch for the whole file: every chunk is planned against the graph as it was before
    // the file, and the variants that meet are joined once all of them are written.
    let mut batch = EditBatch::default();
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
                    &mut batch,
                )
                .unwrap();
            } else {
                BlockGroup::insert_changes(conn, context.workspace(), chunk, &mut batch).unwrap();
            }
            bar.inc(chunk.len() as u64);
        }
        summary
            .entry(sample_name)
            .or_default()
            .entry(path.name)
            .or_insert_with(|| path_changes.len() as i64);
    }
    BlockGroup::combine_batch(conn, batch)?;
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

    use gen_core::is_terminal;
    use gen_models::{
        accession::Accession, block_group_edge::BlockGroupEdge, node::Node, sample::Sample,
        sample_lineage::SampleLineage,
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_gen},
    };
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

    /// Two deletions on one haplotype that meet at a coordinate, applied from the same VCF,
    /// combine into a route that skips both: bases 10-11 and 12-13 of `m123` are both gone.
    #[test]
    fn test_adjacent_deletions_in_one_vcf_combine() {
        let context = setup_gen();
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let conn = context.graph().conn();
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fixtures.join("simple.fa").to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();
        update_with_vcf(
            &context,
            &fixtures
                .join("simple_adjacent_deletions.vcf")
                .to_str()
                .unwrap()
                .to_string(),
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
                &get_sample_bg(conn, &collection, "adjacent").id,
                false
            )
            .unwrap(),
            HashSet::from(["ATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
    }

    /// `m123` of `simple.fa`.
    const SIMPLE_REFERENCE: &str = "ATCGATCGATCGATCGATCGGGAACACACAGAGA";

    /// Applies a VCF of `records` on `m123` of `simple.fa`, one genotype column per sample in
    /// `samples`. Each record is its fields from `POS` on, tab-separated.
    fn apply_simple_vcf(samples: &[&str], records: &[&str]) -> DbContext {
        use std::io::Write;

        let directory = tempfile::tempdir().unwrap();
        let vcf_path = directory.path().join("variants.vcf");
        let mut vcf = std::fs::File::create(&vcf_path).unwrap();
        writeln!(
            vcf,
            "##fileformat=VCFv4.1\n##contig=<ID=m123,length=34>\n\
             ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}",
            samples.join("\t")
        )
        .unwrap();
        for record in records {
            writeln!(vcf, "m123\t{record}").unwrap();
        }
        drop(vcf);
        let context = setup_gen();
        let collection = "test".to_string();
        let fasta = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        import_fasta(
            &context,
            &fasta.to_str().unwrap().to_string(),
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
        context
    }

    /// The sequences `sample` spells.
    fn sample_sequences(context: &DbContext, sample: &str) -> HashSet<String> {
        let conn = context.graph().conn();
        BlockGroup::get_all_sequences(
            conn,
            context.workspace(),
            &get_sample_bg(conn, "test", sample).id,
            false,
        )
        .unwrap()
    }

    /// The deletion nodes in `sample`'s graph: the nodes that spell no sequence.
    fn deletion_nodes(context: &DbContext, sample: &str) -> HashSet<HashId> {
        let conn = context.graph().conn();
        let graph = BlockGroup::get_graph(
            conn,
            context.workspace(),
            &get_sample_bg(conn, "test", sample).id,
            None,
        )
        .unwrap();
        graph
            .nodes()
            .filter(|node| !is_terminal(node.node_id) && node.sequence_start == node.sequence_end)
            .map(|node| node.node_id)
            .collect()
    }

    /// `m123` with the bases in `deleted` (half-open, sorted, disjoint) removed.
    fn without(deleted: &[(usize, usize)]) -> String {
        let mut spelled = String::new();
        let mut position = 0;
        for &(start, end) in deleted {
            spelled.push_str(&SIMPLE_REFERENCE[position..start]);
            position = end;
        }
        spelled.push_str(&SIMPLE_REFERENCE[position..]);
        spelled
    }

    /// A VCF record deleting `m123`'s bases `start..end`, anchored on the base before them.
    fn deletion_record(start: usize, end: usize, genotypes: &str) -> String {
        format!(
            "{start}\t.\t{}\t{}\t60\t.\t.\tGT\t{genotypes}",
            &SIMPLE_REFERENCE[start - 1..end],
            &SIMPLE_REFERENCE[start - 1..start]
        )
    }

    /// However many deletions meet end to end, the route through all of them is spelled, and
    /// each is its own deletion node joined only to its neighbours, with no route skipping
    /// several at once.
    #[test]
    fn test_chain_of_adjacent_deletions_in_one_vcf() {
        for count in 1..=10 {
            let deleted = (0..count)
                .map(|index| (1 + 2 * index, 3 + 2 * index))
                .collect::<Vec<_>>();
            let records = deleted
                .iter()
                .map(|&(start, end)| deletion_record(start, end, "1"))
                .collect::<Vec<_>>();
            let context = apply_simple_vcf(
                &["s"],
                &records.iter().map(String::as_str).collect::<Vec<_>>(),
            );
            let conn = context.graph().conn();

            assert_eq!(
                sample_sequences(&context, "s"),
                HashSet::from([without(&[(1, 1 + 2 * count)])]),
                "{count} deletions"
            );
            assert_eq!(
                deletion_nodes(&context, "s").len(),
                count,
                "{count} deletions"
            );
            let skips = BlockGroupEdge::edges_for_block_group(
                conn,
                &get_sample_bg(conn, "test", "s").id,
                None,
            )
            .into_iter()
            .filter(|augmented_edge| {
                let edge = &augmented_edge.edge;
                edge.source_node_id == edge.target_node_id
                    && edge.source_coordinate < edge.target_coordinate
            })
            .count();
            assert_eq!(
                skips, 0,
                "{count} deletions should write no edge skipping bases"
            );
        }
    }

    /// The order of the records does not change which deletions combine.
    #[test]
    fn test_adjacent_deletions_combine_in_either_record_order() {
        let context = apply_simple_vcf(
            &["s"],
            &[&deletion_record(11, 13, "1"), &deletion_record(9, 11, "1")],
        );
        assert_eq!(
            sample_sequences(&context, "s"),
            HashSet::from([without(&[(9, 13)])])
        );
    }

    /// A deletion next to a substitution, on either side, combines with it, as two adjacent
    /// substitutions do.
    #[test]
    fn test_adjacent_mixed_variants_in_one_vcf_combine() {
        let cases = [
            (
                vec![
                    deletion_record(9, 11, "1"),
                    "12\t.\tG\tT\t60\t.\t.\tGT\t1".to_string(),
                ],
                format!("{}T{}", &SIMPLE_REFERENCE[..9], &SIMPLE_REFERENCE[12..]),
            ),
            (
                vec![
                    "10\t.\tT\tG\t60\t.\t.\tGT\t1".to_string(),
                    deletion_record(10, 12, "1"),
                ],
                format!("{}G{}", &SIMPLE_REFERENCE[..9], &SIMPLE_REFERENCE[12..]),
            ),
            (
                vec![
                    "10\t.\tT\tG\t60\t.\t.\tGT\t1".to_string(),
                    "11\t.\tC\tA\t60\t.\t.\tGT\t1".to_string(),
                ],
                format!("{}GA{}", &SIMPLE_REFERENCE[..9], &SIMPLE_REFERENCE[11..]),
            ),
        ];
        for (records, expected) in cases {
            let context = apply_simple_vcf(
                &["s"],
                &records.iter().map(String::as_str).collect::<Vec<_>>(),
            );
            assert!(
                sample_sequences(&context, "s").contains(&expected),
                "{records:?} should spell {expected}"
            );
        }
    }

    /// Deletions phased onto different haplotypes never occur together, so no route combines
    /// them even though they meet.
    #[test]
    fn test_phased_adjacent_deletions_on_different_haplotypes_do_not_combine() {
        let context = apply_simple_vcf(
            &["s"],
            &[
                &deletion_record(9, 11, "1|0"),
                &deletion_record(11, 13, "0|1"),
            ],
        );
        let sequences = sample_sequences(&context, "s");
        assert!(sequences.contains(&without(&[(9, 11)])), "{sequences:?}");
        assert!(!sequences.contains(&without(&[(9, 13)])), "{sequences:?}");
    }

    /// A deletion is identified by the bases it removes: two samples deleting the same bases
    /// share its node, while deleting the same bases as two adjacent deletions or as one are
    /// different alleles even though they spell the same sequence.
    #[test]
    fn test_deletion_nodes_are_identified_by_the_bases_they_remove() {
        let context = apply_simple_vcf(
            &["two", "first", "one"],
            &[
                &deletion_record(9, 11, "1\t1\t0"),
                &deletion_record(11, 13, "1\t0\t0"),
                &deletion_record(9, 13, "0\t0\t1"),
            ],
        );
        assert_eq!(
            sample_sequences(&context, "two"),
            sample_sequences(&context, "one")
        );
        let two = deletion_nodes(&context, "two");
        let first = deletion_nodes(&context, "first");
        let one = deletion_nodes(&context, "one");
        assert_eq!((two.len(), first.len(), one.len()), (2, 1, 1));
        assert!(first.is_subset(&two));
        assert!(one.is_disjoint(&two));
    }

    /// Variants that meet across the boundary between two chunks of changes combine as they do
    /// within one chunk.
    #[test]
    fn test_adjacent_deletions_across_a_chunk_boundary_combine() {
        use std::io::Write;

        // A 12 kb reference with a substitution every other base up to the chunk size, then two
        // deletions meeting at 11,000 as the last change of one chunk and the first of the next.
        let substitutions = VCF_CHANGE_APPLY_CHUNK_SIZE - 1;
        let reference = "AC".repeat(6_000);
        let directory = tempfile::tempdir().unwrap();
        let fasta_path = directory.path().join("reference.fa");
        std::fs::write(&fasta_path, format!(">chr1\n{reference}\n")).unwrap();
        let vcf_path = directory.path().join("variants.vcf");
        let mut vcf = std::fs::File::create(&vcf_path).unwrap();
        writeln!(
            vcf,
            "##fileformat=VCFv4.1\n##contig=<ID=chr1,length=12000>\n\
             ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts"
        )
        .unwrap();
        for index in 0..substitutions {
            writeln!(vcf, "chr1\t{}\t.\tA\tT\t60\t.\t.\tGT\t1", 1 + 2 * index).unwrap();
        }
        writeln!(vcf, "chr1\t10998\t.\tCAC\tC\t60\t.\t.\tGT\t1").unwrap();
        writeln!(vcf, "chr1\t11000\t.\tCAC\tC\t60\t.\t.\tGT\t1").unwrap();
        drop(vcf);

        let context = setup_gen();
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

        let mut expected = "TC".repeat(substitutions) + &reference[2 * substitutions..10_998];
        expected.push_str(&reference[11_002..]);
        assert_eq!(sample_sequences(&context, "s"), HashSet::from([expected]));
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

    /// Applying a VCF grows linearly with its variants, including once it spans more than one
    /// chunk of changes: 8,000 variants on a synthetic 2 Mb reference take at most ten times as
    /// long as 2,000. Each edit is planned against the routes cached before the file, not against
    /// every edge the earlier chunks stored.
    #[test]
    #[cfg(feature = "benchmark")]
    #[ignore = "manual benchmark; timing depends on the machine and its load"]
    fn test_vcf_update_scales_linearly_with_variant_count() {
        use std::io::Write;

        let reference_length = 2_000_000usize;
        let mut state = 12345u64;
        let mut next_random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let bases = *b"ACGT";
        let reference: Vec<u8> = (0..reference_length)
            .map(|_| bases[(next_random() % 4) as usize])
            .collect();
        let mut seconds = vec![];
        for variant_count in [2_000usize, 8_000] {
            let directory = tempfile::tempdir().unwrap();
            let fasta_path = directory.path().join("reference.fa");
            let mut fasta = std::fs::File::create(&fasta_path).unwrap();
            writeln!(fasta, ">chr1").unwrap();
            fasta.write_all(&reference).unwrap();
            writeln!(fasta).unwrap();
            let vcf_path = directory.path().join("variants.vcf");
            let mut vcf = std::fs::File::create(&vcf_path).unwrap();
            writeln!(
                vcf,
                "##fileformat=VCFv4.1\n##contig=<ID=chr1,length={reference_length}>\n\
                 ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
                 #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample"
            )
            .unwrap();
            // A SNP, a one-base deletion and a two-base insertion in turn, 100 bases apart.
            for index in 0..variant_count {
                let position = 100 + index * 100;
                let base = reference[position - 1] as char;
                let (reference_allele, alternative_allele) = match index % 3 {
                    0 => (
                        base.to_string(),
                        if base == 'A' { "C" } else { "A" }.to_string(),
                    ),
                    1 => (
                        format!("{base}{}", reference[position] as char),
                        base.to_string(),
                    ),
                    _ => (base.to_string(), format!("{base}GG")),
                };
                writeln!(
                    vcf,
                    "chr1\t{position}\t.\t{reference_allele}\t{alternative_allele}\t60\t.\t.\tGT\t1"
                )
                .unwrap();
            }
            drop(vcf);

            let context = setup_gen();
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
            let started = time::Instant::now();
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
            seconds.push(started.elapsed().as_secs_f64());
        }
        println!(
            "2,000 variants: {:.2} s; 8,000 variants: {:.2} s",
            seconds[0], seconds[1]
        );
        assert!(
            seconds[1] <= seconds[0] * 10.0,
            "8,000 variants took {:.2} s against {:.2} s for 2,000",
            seconds[1],
            seconds[0]
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
