use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    fmt::Debug,
    io, str,
};

use gen_core::{HashId, NodeIntervalBlock, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, BlockGroupChange, BlockGroupData, PathCache},
    db::{DbContext, GraphConnection},
    errors::{BlockGroupError, NodeError, OperationError, PathError, SampleError, SequenceError},
    file_types::FileTypes,
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    reference_alias::ReferenceAlias,
    region::{ResolvedGenRegion, ResolvedRegionKind},
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
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
    block_group: BlockGroup,
    sample_path: &Path,
    ids: Option<String>,
    ref_start: i64,
    ref_end: i64,
    path_length: i64,
    chromosome_index: i64,
    phased: i64,
    block_sequence: String,
    sequence_length: i64,
    node_id: HashId,
    preserve_edge: bool,
) -> Result<BlockGroupChange, BlockGroupError> {
    let new_block = PathBlock {
        node_id,
        block_sequence,
        sequence_start: 0,
        sequence_end: sequence_length,
        path_start: ref_start,
        path_end: ref_end,
        strand: Strand::Forward,
    };
    let region = ResolvedGenRegion {
        block_group,
        path: Some(sample_path.clone()),
        accession: None,
        annotation: None,
        kind: ResolvedRegionKind::Path,
        anchor_start: 0,
        anchor_end: path_length,
        feature_length: path_length,
        start: ref_start,
        end: ref_end,
        start_anchors: None,
        end_anchors: None,
        remove_ambiguous_positions: false,
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

#[derive(Debug)]
struct VcfEntry {
    block_group_id: HashId,
    sample_name: String,
    path: Path,
    ids: Option<String>,
    ref_start: i64,
    path_length: i64,
    alt_seq: String,
    chromosome_index: i64,
    phased: i64,
    preserve_reference: bool,
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
                Sample::get_parent_names(conn, sample_name)
            } else {
                explicit_parent_samples.to_vec()
            }
        })
        .clone()
}

pub fn update_with_vcf(
    context: &DbContext,
    vcf_path: &String,
    collection_name: &str,
    fixed_genotype: String,
    fixed_sample: Option<&str>,
    parent_samples: Vec<String>,
    in_place: bool,
) -> Result<Operation, VcfError> {
    let conn = context.graph().conn();
    let progress_bar = get_handler();
    let cnv_re = Regex::new(r"(?x)<CN(?P<count>\d+)>").unwrap();

    let mut session = start_operation(conn);

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

    let mut changes: HashMap<(Path, String), Vec<BlockGroupChange>> = HashMap::new();

    let mut node_source_paths: HashMap<HashId, HashId> = HashMap::new();
    let mut resolved_parent_samples: HashMap<String, Vec<String>> = HashMap::new();
    let mut created_samples: HashSet<&str> = HashSet::new();

    let mut path_lengths: HashMap<HashId, i64> = HashMap::new();
    let mut block_groups_by_id: HashMap<HashId, BlockGroup> = HashMap::new();

    let mut block_group_names = vec![];
    for parent_sample in &parent_samples {
        let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample);
        block_group_names.extend(block_groups.iter().map(|bg| bg.name.clone()));
    }
    let references_by_alias =
        ReferenceAlias::get_references_by_alias(conn, block_group_names).unwrap();

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
                        let mut alt_seq = alt_alleles[chromosome_index - 1].to_string();
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
                        for sample_bg_id in &sample_bg_ids {
                            let sample_path =
                                PathCache::lookup(&mut path_cache, sample_bg_id, seq_name.clone())?;
                            let path_length = match path_lengths.entry(sample_path.id) {
                                Entry::Occupied(entry) => *entry.get(),
                                Entry::Vacant(entry) => *entry.insert(sample_path.length(conn)?),
                            };

                            if ref_start > path_length {
                                return Err(VcfError::InvalidRecord(format!(
                                    "Invalid position found. Path {0} has length of {path_length}, change is in position {ref_start}.",
                                    sample_path.name
                                )));
                            }
                            vcf_entries.push(VcfEntry {
                                ids: allele_accession.clone(),
                                ref_start,
                                block_group_id: *sample_bg_id,
                                path: sample_path.clone(),
                                sample_name: fixed_sample.to_string(),
                                path_length,
                                alt_seq: alt_seq.clone(),
                                chromosome_index: chromosome_index as i64,
                                phased,
                                preserve_reference: has_ref,
                            });
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
                                    for sample_bg_id in &sample_bg_ids {
                                        let sample_path = PathCache::lookup(
                                            &mut path_cache,
                                            sample_bg_id,
                                            seq_name.clone(),
                                        )?;
                                        let path_length = match path_lengths.entry(sample_path.id) {
                                            Entry::Occupied(entry) => *entry.get(),
                                            Entry::Vacant(entry) => {
                                                *entry.insert(sample_path.length(conn)?)
                                            }
                                        };

                                        if ref_start > path_length {
                                            return Err(VcfError::InvalidRecord(format!(
                                                "Invalid position found. Path {0} has length of {path_length}, change is in position {ref_start}.",
                                                sample_path.name
                                            )));
                                        }

                                        vcf_entries.push(VcfEntry {
                                            ids: allele_accession.clone(),
                                            block_group_id: *sample_bg_id,
                                            ref_start,
                                            path: sample_path.clone(),
                                            sample_name: sample_name.to_string(),
                                            path_length,
                                            alt_seq: alt_seq.clone(),
                                            chromosome_index: chromosome_index as i64,
                                            phased,
                                            preserve_reference: has_ref,
                                        });
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
            // * indicates this allele is removed by another deletion in the sample
            if vcf_entry.alt_seq == "*" {
                continue;
            }
            let ref_start = vcf_entry.ref_start;
            let sequence =
                SequenceCache::lookup(&mut sequence_cache, "DNA", vcf_entry.alt_seq.to_string())?;
            let sequence_string = sequence.get_sequence(None, None)?;

            let source_path_id =
                if let Some(source_path_id) = node_source_paths.get(&vcf_entry.path.id) {
                    *source_path_id
                } else {
                    let block_group = BlockGroup::get_by_id(conn, &vcf_entry.block_group_id)?;
                    if let Some(parent_block_group_id) = block_group.parent_block_group_id {
                        let parent_path = PathCache::lookup(
                            &mut path_cache,
                            &parent_block_group_id,
                            vcf_entry.path.name.clone(),
                        )?;
                        node_source_paths.insert(vcf_entry.path.id, parent_path.id);
                        parent_path.id
                    } else {
                        node_source_paths.insert(vcf_entry.path.id, vcf_entry.path.id);
                        vcf_entry.path.id
                    }
                };

            let node_id = Node::create(
                conn,
                &sequence.hash,
                &HashId::convert_str(&format!(
                    "{path_id}:{ref_start}-{ref_end}->{sequence_hash}",
                    path_id = source_path_id,
                    sequence_hash = sequence.hash
                )),
            )?;
            let block_group = match block_groups_by_id.entry(vcf_entry.block_group_id) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => entry
                    .insert(BlockGroup::get_by_id(conn, &vcf_entry.block_group_id)?)
                    .clone(),
            };
            let change = prepare_change(
                block_group,
                &vcf_entry.path,
                vcf_entry.ids,
                ref_start,
                ref_end,
                vcf_entry.path_length,
                vcf_entry.chromosome_index,
                vcf_entry.phased,
                sequence_string.clone(),
                sequence_string.len() as i64,
                node_id,
                vcf_entry.preserve_reference,
            )?;
            changes
                .entry((vcf_entry.path, vcf_entry.sample_name))
                .or_default()
                .push(change);
        }
        bar.inc(1);
    }
    bar.finish();

    let bar = progress_bar.add(get_progress_bar(
        changes.values().map(|c| c.len() as u64).sum::<u64>(),
    ));
    bar.set_message("Changes applied");
    let mut summary: HashMap<String, HashMap<String, i64>> = HashMap::new();
    let mut tree_map: HashMap<(HashId, ResolvedRegionKind), IntervalTree<i64, NodeIntervalBlock>> =
        HashMap::new();
    for ((path, sample_name), path_changes) in changes {
        for chunk in path_changes.chunks(1000) {
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
                BlockGroup::insert_changes(conn, &in_place_changes, Some(&mut tree_map)).unwrap();
            } else {
                BlockGroup::insert_changes(conn, chunk, Some(&mut tree_map)).unwrap();
            }
            bar.inc(chunk.len() as u64);
        }
        summary
            .entry(sample_name)
            .or_default()
            .entry(path.name)
            .or_insert_with(|| path_changes.len() as i64);
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
    let mut summary_str = "".to_string();
    for (sample_name, sample_changes) in summary.iter() {
        summary_str.push_str(&format!("Sample {sample_name}\n"));
        for (path_name, change_count) in sample_changes.iter() {
            summary_str.push_str(&format!(" {path_name}: {change_count} changes.\n"));
        }
    }

    let bar = add_saving_operation_bar(&progress_bar);
    bar.set_message("Saving operation");
    let op = end_operation(
        context,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile::new(vcf_path.to_string()).set_file_type(FileTypes::VCF)],
            description: "vcf_addition".to_string(),
        },
        &summary_str,
        None,
    )
    .map_err(VcfError::OperationError);
    bar.finish();
    op
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    #[allow(unused_imports)]
    use std::time;
    use std::{collections::HashSet, path::PathBuf};

    use gen_models::{
        accession::Accession, node::Node, sample::Sample, sample_lineage::SampleLineage,
        traits::Query,
    };
    use rusqlite::params;

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_sample_bg, setup_gen},
        track_database,
    };
    #[test]
    fn test_update_fasta_with_vcf() -> Result<(), VcfError> {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                false,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `G1` genotype has no changes
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "G1").id, false)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `foo` is homozygous for the first variant and does not contain the second
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "foo").id, false)
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                false,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `bar` sample has the refrence + a deletion of the C
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "bar").id, false)
                .unwrap(),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATGATCGATCGGGAACACACAGAGA".to_string()
            ])
        );
        // `baz` sample has a deletion of CG and an insertion of A
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "baz").id, false)
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                false,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
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
    fn test_error_when_vcf_has_changes_out_of_bounds() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

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
        assert!(matches!(res, Err(VcfError::InvalidRecord(_))));
    }

    #[test]
    fn test_handles_missing_allele() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple_missing_allele.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "foo").id, false)
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "foo").id, true)
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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

        let nodes = Node::query(conn, "select * from nodes;", rusqlite::params!());
        assert_eq!(nodes.len(), 5);

        let second_update = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        );
        assert!(matches!(
            second_update,
            Err(VcfError::OperationError(OperationError::NoChanges))
        ));
        assert_eq!(
            Node::query(conn, "select * from nodes;", rusqlite::params!()).len(),
            5
        );
    }

    #[test]
    fn test_deduplicates_nodes_multiple_paths() {
        let context = setup_gen();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/multiseq.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/multiseq.fa");
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        assert_eq!(
            Node::query(conn, "select * from nodes;", rusqlite::params!()).len(),
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

        let nodes = Node::query(conn, "select * from nodes;", rusqlite::params!());
        assert_eq!(nodes.len(), 8);

        let second_update = update_with_vcf(
            &context,
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            None,
            vec![Sample::DEFAULT_NAME.to_string()],
            false,
        );
        assert!(matches!(
            second_update,
            Err(VcfError::OperationError(OperationError::NoChanges))
        ));
        assert_eq!(
            Node::query(conn, "select * from nodes;", rusqlite::params!()).len(),
            8
        );
    }

    #[test]
    #[cfg(feature = "benchmark")]
    fn test_vcf_import_benchmark() {
        let context = setup_gen();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/chr22_100k_no_samples.vcf.gz");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/chr22.fa.gz");
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
            Accession::query(
                conn,
                "select * from accessions where name = ?1;",
                params!["del1"],
            )
            .len(),
            1
        );

        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1;",
                params!["lp1"],
            )
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                params!["lp1"]
            )
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
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
                &get_sample_bg(conn, &collection, Sample::DEFAULT_NAME).id,
                true,
            )
            .unwrap(),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "f1").id, true)
                .unwrap(),
            HashSet::from_iter(vec!["ATCTCGATCGATCGCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "f2").id, true)
                .unwrap(),
            HashSet::from_iter(vec!["ATCTGGATCGATCGCGGAATCAGAACACACAGGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "f3").id, true)
                .unwrap(),
            HashSet::from_iter(vec!["ATCGGGATCGATCGCTCAGAACACACAGGA".to_string()])
        );
    }

    #[test]
    fn test_update_vcf_uses_lineage_parent_when_parent_sample_is_omitted() {
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "reference",
            false,
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "parent-a",
            false,
        )
        .unwrap();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "parent-b",
            false,
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
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "reference",
            false,
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

        assert!(SampleLineage::get_parents(conn, "child").is_empty());
    }
}
