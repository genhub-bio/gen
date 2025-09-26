use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    io, str,
};

use gen_core::{HashId, PathBlock, Strand};
use gen_models::{
    block_group::{BlockGroup, BlockGroupData, PathCache, PathChange},
    errors::{OperationError, QueryError},
    file_types::FileTypes,
    node::Node,
    operations::{Operation, OperationFile, OperationInfo},
    path::Path,
    sample::Sample,
    sequence::Sequence,
    session_operations::{end_operation, start_operation},
    traits::*,
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
use rusqlite::{self, Connection, types::Value as SQLValue};
use thiserror::Error;

use crate::{
    parse_genotype,
    progress_bar::{add_saving_operation_bar, get_handler, get_progress_bar},
};

#[derive(Debug)]
struct BlockGroupCache<'a> {
    pub cache: HashMap<BlockGroupData<'a>, HashId>,
    pub conn: &'a Connection,
}

impl<'a> BlockGroupCache<'_> {
    pub fn new(conn: &Connection) -> BlockGroupCache<'_> {
        BlockGroupCache {
            cache: HashMap::<BlockGroupData, HashId>::new(),
            conn,
        }
    }

    pub fn lookup(
        block_group_cache: &mut BlockGroupCache<'a>,
        collection_name: &'a str,
        sample_name: &'a str,
        name: String,
        parent_sample: Option<&'a str>,
    ) -> Result<HashId, QueryError> {
        let block_group_key = BlockGroupData {
            collection_name,
            sample_name: Some(sample_name),
            name: name.clone(),
        };
        let block_group_lookup = block_group_cache.cache.get(&block_group_key);
        if let Some(block_group_id) = block_group_lookup {
            Ok(*block_group_id)
        } else {
            let new_block_group_id = BlockGroup::get_or_create_sample_block_group(
                block_group_cache.conn,
                collection_name,
                sample_name,
                &name,
                parent_sample,
            );

            let result = new_block_group_id?;
            block_group_cache.cache.insert(block_group_key, result);
            Ok(result)
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
    pub conn: &'a Connection,
}

impl<'a> SequenceCache<'_> {
    pub fn new(conn: &Connection) -> SequenceCache<'_> {
        SequenceCache {
            cache: HashMap::<SequenceKey, Sequence>::new(),
            conn,
        }
    }

    pub fn lookup(
        sequence_cache: &mut SequenceCache<'a>,
        sequence_type: &'a str,
        sequence: String,
    ) -> Sequence {
        let sequence_key = SequenceKey {
            sequence_type,
            sequence: sequence.clone(),
        };
        let sequence_lookup = sequence_cache.cache.get(&sequence_key);
        if let Some(found_sequence) = sequence_lookup {
            found_sequence.clone()
        } else {
            let new_sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(&sequence)
                .save(sequence_cache.conn);

            sequence_cache
                .cache
                .insert(sequence_key, new_sequence.clone());
            new_sequence
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_change(
    sample_bg_id: &HashId,
    sample_path: &Path,
    ids: Option<String>,
    ref_start: i64,
    ref_end: i64,
    chromosome_index: i64,
    phased: i64,
    block_sequence: String,
    sequence_length: i64,
    node_id: HashId,
    preserve_edge: bool,
) -> PathChange {
    // TODO: new sequence may not be real and be <DEL> or some sort. Handle these.
    let new_block = PathBlock {
        id: 0,
        node_id,
        block_sequence,
        sequence_start: 0,
        sequence_end: sequence_length,
        path_start: ref_start,
        path_end: ref_end,
        strand: Strand::Forward,
    };
    PathChange {
        block_group_id: *sample_bg_id,
        path: sample_path.clone(),
        path_accession: ids,
        start: ref_start,
        end: ref_end,
        block: new_block,
        chromosome_index,
        phased,
        preserve_edge,
    }
}

#[derive(Debug)]
struct VcfEntry {
    block_group_id: HashId,
    sample_name: String,
    path: Path,
    ids: Option<String>,
    ref_start: i64,
    alt_seq: String,
    chromosome_index: i64,
    phased: i64,
    preserve_reference: bool,
}

#[derive(Error, Debug, PartialEq)]
pub enum VcfError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Invalid Record: {0}")]
    InvalidRecord(String),
}

pub fn update_with_vcf<'a>(
    vcf_path: &String,
    collection_name: &'a str,
    fixed_genotype: String,
    fixed_sample: String,
    conn: &Connection,
    operation_conn: &Connection,
    coordinate_frame: impl Into<Option<&'a str>>,
) -> Result<Operation, VcfError> {
    let progress_bar = get_handler();
    let coordinate_frame = coordinate_frame.into();
    let cnv_re = Regex::new(r"(?x)<CN(?P<count>\d+)>").unwrap();

    let mut session = start_operation(conn);

    let mut reader = vcf::io::reader::Builder::default()
        .build_from_path(vcf_path)
        .expect("Unable to parse");
    let header = reader.read_header().unwrap();
    let sample_names = header.sample_names();
    for name in sample_names {
        Sample::get_or_create(conn, name);
    }
    if !fixed_sample.is_empty() {
        Sample::get_or_create(conn, &fixed_sample);
    }
    let mut genotype = vec![];
    if !fixed_genotype.is_empty() {
        genotype = parse_genotype(&fixed_genotype);
    }

    // Cache a bunch of data ahead of making changes
    let mut block_group_cache = BlockGroupCache::new(conn);
    let mut path_cache = PathCache::new(conn);
    let mut sequence_cache = SequenceCache::new(conn);
    let mut accession_cache = HashMap::new();

    let mut changes: HashMap<(Path, String), Vec<PathChange>> = HashMap::new();

    let mut parent_block_groups: HashMap<(&str, HashId), HashId> = HashMap::new();
    let mut created_samples = HashSet::new();

    let mut path_lengths: HashMap<HashId, i64> = HashMap::new();

    let _ = progress_bar.println("Parsing VCF for changes.");

    let bar = progress_bar.add(get_progress_bar(None));

    bar.set_message("Records Parsed");
    for result in reader.records() {
        let record = result.unwrap();
        let seq_name: String = record.reference_sequence_name().to_string();
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

        if !fixed_sample.is_empty() && !genotype.is_empty() {
            if !created_samples.contains(&fixed_sample) {
                Sample::get_or_create_child(conn, collection_name, &fixed_sample, coordinate_frame);
                created_samples.insert(&fixed_sample);
            }
            let sample_bg_id = BlockGroupCache::lookup(
                &mut block_group_cache,
                collection_name,
                &fixed_sample,
                seq_name.clone(),
                coordinate_frame,
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
                        let sample_path =
                            PathCache::lookup(&mut path_cache, &sample_bg_id, seq_name.clone());
                        let path_length = path_lengths
                            .entry(sample_path.id)
                            .or_insert(sample_path.length(conn));

                        if ref_start > *path_length {
                            return Err(VcfError::InvalidRecord(format!(
                                "Invalid position found. Path {0} has length of {path_length}, change is in position {ref_start}.",
                                sample_path.name
                            )));
                        }
                        vcf_entries.push(VcfEntry {
                            ids: allele_accession,
                            ref_start,
                            block_group_id: sample_bg_id,
                            path: sample_path.clone(),
                            sample_name: fixed_sample.clone(),
                            alt_seq,
                            chromosome_index: chromosome_index as i64,
                            phased,
                            preserve_reference: has_ref,
                        });
                    } else if let Some(ref_accession) = allele_accession {
                        let sample_path =
                            PathCache::lookup(&mut path_cache, &sample_bg_id, seq_name.clone());

                        let key = (sample_path, ref_accession.clone());

                        accession_cache.entry(key).or_insert_with(|| {
                            (ref_start, ref_start + record.reference_bases().len() as i64)
                        });
                    }
                }
            }
        } else {
            for (sample_index, sample) in record.samples().iter().enumerate() {
                let sample_name = &sample_names[sample_index];
                if !created_samples.contains(sample_name) {
                    Sample::get_or_create_child(
                        conn,
                        collection_name,
                        sample_name,
                        coordinate_frame,
                    );
                    created_samples.insert(sample_name);
                }
                let sample_bg_id = BlockGroupCache::lookup(
                    &mut block_group_cache,
                    collection_name,
                    sample_name,
                    seq_name.clone(),
                    coordinate_frame,
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
                                    let sample_path = PathCache::lookup(
                                        &mut path_cache,
                                        &sample_bg_id,
                                        seq_name.clone(),
                                    );
                                    let path_length =
                                        if let Some(l) = path_lengths.get(&sample_path.id) {
                                            l
                                        } else {
                                            let l = sample_path.sequence(conn).len();
                                            path_lengths.insert(sample_path.id, l as i64);
                                            &path_lengths[&sample_path.id]
                                        };

                                    if ref_start > *path_length {
                                        return Err(VcfError::InvalidRecord(format!(
                                            "Invalid position found. Path {0} has length of {path_length}, change is in position {ref_start}.",
                                            sample_path.name
                                        )));
                                    }

                                    vcf_entries.push(VcfEntry {
                                        ids: allele_accession,
                                        block_group_id: sample_bg_id,
                                        ref_start,
                                        path: sample_path.clone(),
                                        sample_name: sample_name.clone(),
                                        alt_seq,
                                        chromosome_index: chromosome_index as i64,
                                        phased,
                                        preserve_reference: has_ref,
                                    });
                                } else if let Some(ref_accession) = allele_accession {
                                    let sample_path = PathCache::lookup(
                                        &mut path_cache,
                                        &sample_bg_id,
                                        seq_name.clone(),
                                    );

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

        for vcf_entry in vcf_entries {
            // * indicates this allele is removed by another deletion in the sample
            if vcf_entry.alt_seq == "*" {
                continue;
            }
            let ref_start = vcf_entry.ref_start;
            let sequence =
                SequenceCache::lookup(&mut sequence_cache, "DNA", vcf_entry.alt_seq.to_string());
            let sequence_string = sequence.get_sequence(None, None);

            let parent_path_id = parent_block_groups.entry((collection_name, vcf_entry.path.id)).or_insert_with(|| {
                let parent_bg = BlockGroup::query(conn, "select * from block_groups where collection_name = ?1 AND sample_name is null and name = ?2", rusqlite::params!(SQLValue::from(collection_name.to_string()), SQLValue::from(vcf_entry.path.name.clone())));
                if parent_bg.is_empty() {
                    vcf_entry.path.id
                } else {
                    let parent_path =
                        PathCache::lookup(&mut path_cache, &parent_bg.first().unwrap().id, vcf_entry.path.name.clone());
                    parent_path.id
                }
            });

            let node_id = Node::create(
                conn,
                &sequence.hash,
                &HashId::convert_str(&format!(
                    "{path_id}:{ref_start}-{ref_end}->{sequence_hash}",
                    path_id = parent_path_id,
                    sequence_hash = sequence.hash
                )),
            );
            let change = prepare_change(
                &vcf_entry.block_group_id,
                &vcf_entry.path,
                vcf_entry.ids,
                ref_start,
                ref_end,
                vcf_entry.chromosome_index,
                vcf_entry.phased,
                sequence_string.clone(),
                sequence_string.len() as i64,
                node_id,
                vcf_entry.preserve_reference,
            );
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
    for ((path, sample_name), path_changes) in changes {
        BlockGroup::insert_changes(
            conn,
            &path_changes,
            &mut path_cache,
            coordinate_frame.is_some(),
        )
        .unwrap();
        bar.inc(path_changes.len() as u64);
        summary
            .entry(sample_name)
            .or_default()
            .entry(path.name)
            .or_insert(path_changes.len() as i64);
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
        );
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
        conn,
        operation_conn,
        &mut session,
        &OperationInfo {
            files: vec![OperationFile {
                file_path: vcf_path.to_string(),
                file_type: FileTypes::VCF,
            }],
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

    use gen_models::{accession::Accession, node::Node, operations::setup_db};

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{get_connection, get_operation_connection, get_sample_bg, setup_gen_dir},
        track_database,
    };

    #[test]
    fn test_update_fasta_with_vcf() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, None).id, false),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `G1` genotype has no changes
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "G1").id, false),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `foo` is homozygous for the first variant and does not contain the second
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "foo").id, false),
            HashSet::from_iter(vec!["ATCATCGATCGATCGATCGGGAACACACAGAGA".to_string(),])
        );
    }

    #[test]
    fn test_update_fasta_with_complex_vcf() {
        setup_gen_dir();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vcfs/complex.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, None).id, false),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        // `bar` sample has the refrence + a deletion of the C
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "bar").id, false),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATGATCGATCGGGAACACACAGAGA".to_string()
            ])
        );
        // `baz` sample has a deletion of CG and an insertion of A
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "baz").id, false),
            HashSet::from_iter(vec![
                "ATCGATCGATATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCAGATCGATCGGGAACACACAGAGA".to_string(),
            ])
        );
    }

    #[test]
    fn test_update_fasta_with_vcf_custom_genotype() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/general.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0/1".to_string(),
            "sample 1".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, None).id, false),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                &get_sample_bg(conn, &collection, "sample 1").id,
                false
            ),
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
        setup_gen_dir();
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let vcf_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vcfs/out_of_bounds.vcf");
        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        let res = update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0/1".to_string(),
            "sample 1".to_string(),
            conn,
            op_conn,
            None,
        );
        assert!(matches!(res, Err(VcfError::InvalidRecord(_))));
    }

    #[test]
    fn test_handles_missing_allele() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple_missing_allele.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                &get_sample_bg(conn, &collection, "unknown").id,
                false
            ),
            HashSet::from_iter(
                ["ATCGATCGATAGACGATCGATCGGGAACACACAGAGA",]
                    .iter()
                    .map(|v| v.to_string())
            )
        );
    }

    #[test]
    fn test_handles_overlap_allele() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple_overlap.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();
        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();
        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "foo").id, false),
            HashSet::from_iter(
                ["ATCATCGATCGATCGATCGGGAACACACAGAGA",]
                    .iter()
                    .map(|v| v.to_string())
            )
        );
    }

    #[test]
    fn test_parses_cnvs() {
        setup_gen_dir();
        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple_cnv.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();

        // TODO: Fix this once pruning works correctly. The issue currently is we prune away the boundary edge
        // and only see a single path through the graph.
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "foo").id, true),
            HashSet::from_iter(vec![
                "ATCGATCGATCGGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCGATCATCATCGATCGGGAACACACAGAGA".to_string()
            ])
        );
    }

    #[test]
    fn test_deduplicates_nodes() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/simple.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();

        let nodes = Node::query(conn, "select * from nodes;", rusqlite::params!());
        assert_eq!(nodes.len(), 5);

        assert_eq!(
            update_with_vcf(
                &vcf_path.to_str().unwrap().to_string(),
                &collection,
                "".to_string(),
                "".to_string(),
                conn,
                op_conn,
                None,
            ),
            Err(VcfError::OperationError(OperationError::NoChanges))
        )
    }

    #[test]
    fn test_deduplicates_nodes_multiple_paths() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/multiseq.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/multiseq.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        assert_eq!(
            Node::query(conn, "select * from nodes;", rusqlite::params!()).len(),
            5
        );

        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();

        let nodes = Node::query(conn, "select * from nodes;", rusqlite::params!());
        assert_eq!(nodes.len(), 8);

        assert_eq!(
            update_with_vcf(
                &vcf_path.to_str().unwrap().to_string(),
                &collection,
                "".to_string(),
                "".to_string(),
                conn,
                op_conn,
                None,
            ),
            Err(VcfError::OperationError(OperationError::NoChanges))
        )
    }

    #[test]
    #[cfg(feature = "benchmark")]
    fn test_vcf_import_benchmark() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/chr22_100k_no_samples.vcf.gz");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/chr22.fa.gz");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        let s = time::Instant::now();
        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "0|1".to_string(),
            "test".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();
        let elapsed = s.elapsed().as_secs();
        assert!(
            elapsed < 20,
            "VCF import benchmark failed: Elapsed time is {elapsed}."
        );
    }

    #[test]
    fn test_creates_accession_paths() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/accession.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();
        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1;",
                rusqlite::params!(SQLValue::from("del1".to_string())),
            )
            .len(),
            1
        );

        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1;",
                rusqlite::params!(SQLValue::from("lp1".to_string())),
            )
            .len(),
            1
        );
    }

    #[test]
    #[should_panic(expected = "Unable to create accession")]
    fn test_disallows_creating_accession_paths_that_exist() {
        setup_gen_dir();
        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        vcf_path.push("fixtures/accession.vcf");
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();

        assert_eq!(
            Accession::query(
                conn,
                "select * from accessions where name = ?1",
                rusqlite::params!(SQLValue::from("lp1".to_string()))
            )
            .len(),
            1
        );

        let mut vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // This is invalid because lp1 already exists from accession.vcf
        vcf_path.push("fixtures/accession_2_invalid.vcf");

        update_with_vcf(
            &vcf_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();
    }

    #[test]
    fn test_changes_in_child_samples() {
        setup_gen_dir();
        let f0_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_iterative_engineering_1.vcf");
        let f1_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_iterative_engineering_2.vcf");
        let f2_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/simple_iterative_engineering_3.vcf");
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let conn = &get_connection(None).unwrap();
        let op_conn = &get_operation_connection(None).unwrap();
        setup_db(op_conn);
        track_database(conn, op_conn).unwrap();

        let collection = "test".to_string();

        import_fasta(
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            None,
            false,
            conn,
            op_conn,
        )
        .unwrap();

        update_with_vcf(
            &f0_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            None,
        )
        .unwrap();

        update_with_vcf(
            &f1_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            "f1",
        )
        .unwrap();

        update_with_vcf(
            &f2_path.to_str().unwrap().to_string(),
            &collection,
            "".to_string(),
            "".to_string(),
            conn,
            op_conn,
            "f2",
        )
        .unwrap();

        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, None).id, true),
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "f1").id, true),
            HashSet::from_iter(vec!["ATCTCGATCGATCGCGGGAACACACAGAGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "f2").id, true),
            HashSet::from_iter(vec!["ATCTGGATCGATCGCGGAATCAGAACACACAGGA".to_string()])
        );
        assert_eq!(
            BlockGroup::get_all_sequences(conn, &get_sample_bg(conn, &collection, "f3").id, true),
            HashSet::from_iter(vec!["ATCGGGATCGATCGCTCAGAACACACAGGA".to_string()])
        );
    }
}
