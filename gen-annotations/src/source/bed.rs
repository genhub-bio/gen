//! BED parsing and translation adapters.
//!
//! This serves as a general mechanism for parsing bed records from any source implementing the
//! Read trait and converting them into Annotation objects. This lets us provide BED files from
//! a variety of sources.

use std::{collections::HashMap, io::Read, sync::Arc};

use gen_core::{HashId, NodeIntervalBlock, Strand, is_terminal};
use gen_models::{
    ForeignKeyCache,
    accession::Accession,
    annotations::{Annotation, AnnotationExtra, BedExtra},
    block_group::BlockGroup,
    interval_tree::IntervalTreeSource as _,
    reference_alias::ReferenceAlias,
    region::ResolvedGenRegion,
    sample::Sample,
};
use intervaltree::IntervalTree;
use noodles::bed;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::translate::compose_strand;

/// Parse one BED name's matching records into the existing annotation model.
pub fn parse_bed_annotation<R>(
    identifier: impl Into<String>,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
where
    R: Read,
{
    let identifier = identifier.into();
    let mut bed_reader = bed::io::reader::Builder::<6>.build_from_reader(reader);
    let mut record = bed::Record::<6>::default();
    let mut records = Vec::new();
    while bed_reader.read_record(&mut record)? != 0 {
        if matching_bed_record(&record, &identifier) {
            records.push(record.clone());
        }
    }
    parse_bed_annotation_records(identifier, records)
}

/// Build a BED annotation from records already matched by an upstream lookup provider.
pub fn parse_bed_annotation_records<I>(
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = bed::Record<6>>,
{
    let identifier = identifier.into();
    let records: Vec<_> = records.into_iter().collect();
    let first = records.first().ok_or(FileAnnotationError::Empty)?;
    let fields = bed_other_fields(first);
    let bed = BedExtra {
        score: Some(first.score()?.to_string()),
        thick_start: fields.first().and_then(|value| value.parse().ok()),
        thick_end: fields.get(1).and_then(|value| value.parse().ok()),
        item_rgb: fields.get(2).cloned(),
        block_count: fields.get(3).and_then(|value| value.parse().ok()),
        block_sizes: fields
            .get(4)
            .map(|value| parse_bed_list(value))
            .filter(|values| !values.is_empty()),
        block_starts: fields
            .get(5)
            .map(|value| parse_bed_list(value))
            .filter(|values| !values.is_empty()),
        other_fields: fields.get(6..).unwrap_or_default().to_vec(),
    };
    let id = HashId::convert_str(&identifier);
    Ok(Annotation {
        id,
        name: identifier,
        group: "bed".to_string(),
        accession_id: id,
        cached_accession: ForeignKeyCache::default(),
        cached_interval_tree: None,
        extra: Some(AnnotationExtra {
            bed: Some(bed),
            ..AnnotationExtra::default()
        }),
    })
}

/// Translate matching BED records and return an annotation with cached graph intervals and accession.
pub fn translate_bed_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &Annotation,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
where
    R: Read,
{
    let mut bed_reader = bed::io::reader::Builder::<6>.build_from_reader(reader);
    let mut record = bed::Record::<6>::default();
    let mut records = Vec::new();
    while bed_reader.read_record(&mut record)? != 0 {
        if matching_bed_record(&record, &annotation.name) {
            records.push(record.clone());
        }
    }
    translate_bed_annotation_records(context, annotation, records)
}

/// Translate selected BED records in supplied order, orienting each record's reverse-strand
/// fragments from its 5′ end, and return an annotation with cached graph intervals and accession.
pub fn translate_bed_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &Annotation,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = bed::Record<6>>,
{
    let mut translated_annotation = annotation.clone();
    let mut position = 0;
    let mut blocks = Vec::new();
    let sample_block_groups = Sample::get_block_groups(
        context.conn,
        context.collection_name,
        context.sample_name,
        context.history_ref,
    )
    .into_iter()
    .map(|block_group| (block_group.name, block_group.id))
    .collect::<HashMap<_, _>>();
    let references = sample_block_groups.keys().cloned().collect();
    let references_by_alias =
        ReferenceAlias::get_references_by_alias(context.conn, references, context.history_ref)
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
    let mut regions_by_reference = HashMap::<String, ResolvedGenRegion>::new();
    for record in records {
        let reference_name = record.reference_sequence_name().to_string();
        let start = record
            .feature_start()
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            .get() as i64
            - 1;
        let end = record
            .feature_end()
            .ok_or_else(|| FileAnnotationError::Translation("BED record has no end".to_string()))?
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            .get() as i64;
        let feature_strand = match record.strand() {
            Ok(Some(bed::feature::record::Strand::Forward)) => Strand::Forward,
            Ok(Some(bed::feature::record::Strand::Reverse)) => Strand::Reverse,
            Ok(None) | Err(_) => Strand::Unknown,
        };
        let resolved_reference = references_by_alias
            .get(&reference_name)
            .map_or(reference_name.as_str(), String::as_str);
        let Some(block_group_id) = sample_block_groups.get(resolved_reference) else {
            continue;
        };
        let mut region = match regions_by_reference.get(resolved_reference) {
            Some(region) => region.clone(),
            None => {
                let path =
                    BlockGroup::get_current_path(context.conn, block_group_id, context.history_ref)
                        .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
                let region = ResolvedGenRegion::from_path(context.conn, &path, 0, 0)
                    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
                regions_by_reference.insert(resolved_reference.to_string(), region.clone());
                region
            }
        };
        region.start = start;
        region.end = end;
        let mut node_slices = region
            .graph_locus(context.conn, context.workspace)
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            .slices;
        if feature_strand == Strand::Reverse {
            node_slices.reverse();
        }
        for node_slice in node_slices {
            if is_terminal(node_slice.block.node_id) {
                continue;
            }
            let sequence_start = node_slice.block.sequence_start + node_slice.start as i64;
            let sequence_end = node_slice.block.sequence_start + node_slice.end as i64;
            if sequence_end <= sequence_start {
                continue;
            }
            let block = NodeIntervalBlock {
                node_id: node_slice.block.node_id,
                start: position,
                end: position + sequence_end - sequence_start,
                sequence_start,
                sequence_end,
                strand: compose_strand(feature_strand, node_slice.strand),
            };
            position = block.end;
            blocks.push(block);
        }
    }
    if blocks.is_empty() {
        return Err(FileAnnotationError::Empty);
    }
    let interval_tree = Arc::new(
        blocks
            .into_iter()
            .map(|block| (block.start..block.end, block))
            .collect::<IntervalTree<_, _>>(),
    );
    translated_annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));
    let mut accession = Accession {
        id: translated_annotation.accession_id,
        name: translated_annotation.name.clone(),
        block_group_id: context.block_group_id,
        parent_accession_id: None,
        cached_interval_tree: None,
    };
    accession.set_cached_interval_tree(Some(interval_tree));
    translated_annotation.set_accession(Arc::new(accession));
    Ok(translated_annotation)
}

fn matching_bed_record(record: &bed::Record<6>, identifier: &str) -> bool {
    record
        .name()
        .and_then(|name| std::str::from_utf8(name.as_ref()).ok())
        .is_some_and(|name| name.eq_ignore_ascii_case(identifier))
}

fn bed_other_fields(record: &bed::Record<6>) -> Vec<String> {
    record
        .other_fields()
        .iter()
        .map(|value| String::from_utf8_lossy(value.as_ref()).into_owned())
        .collect()
}

fn parse_bed_list(value: &str) -> Vec<i64> {
    value
        .split(',')
        .filter(|item| !item.is_empty())
        .filter_map(|item| item.parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor, sync::Arc};

    use gen_core::{HashId, Strand};
    use gen_models::{
        block_group::BlockGroup,
        history::dolt::{active_branch, commit_all},
        interval_tree::IntervalTreeSource as _,
        region::ResolvedGenRegion,
        sample::Sample,
    };

    use super::{AnnotationTranslationContext, parse_bed_annotation, translate_bed_annotation};
    use crate::{
        test_helpers::{create_forward_test_block_group, create_reverse_test_block_group},
        translate::bed::translate_bed,
    };

    #[test]
    fn test_bed_annotation_matches_identifier_and_preserves_metadata() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed");
        let annotation = parse_bed_annotation(
            "abc123.1",
            File::open(path).expect("should open simple BED fixture"),
        )
        .expect("should parse the matching BED record");
        assert_eq!(annotation.name, "abc123.1");
        assert_eq!(annotation.id, HashId::convert_str("abc123.1"));
        assert_eq!(annotation.accession_id, annotation.id);
        assert_eq!(annotation.group, "bed");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.bed.as_ref())
            .expect("should preserve BED metadata");
        assert_eq!(metadata.score.as_deref(), Some("0"));
        assert_eq!(metadata.block_count, Some(3));
        assert_eq!(
            metadata.block_sizes.as_deref(),
            Some([102, 188, 129].as_slice())
        );
        assert_eq!(
            metadata.block_starts.as_deref(),
            Some([0, 3508, 4691].as_slice())
        );
        let translated_annotation = translate_bed_annotation(
            &context,
            &annotation,
            File::open(path).expect("should reopen simple BED fixture"),
        )
        .expect("should translate the matching BED record");
        assert!(annotation.cached_interval_tree().is_none());
        assert!(annotation.cached_accession.snapshot().is_none());
        let accession = translated_annotation
            .get_accession(&conn)
            .expect("should return translated BED accession");
        assert_eq!(accession.id, translated_annotation.accession_id);
        assert_eq!(
            translated_annotation
                .intervaltree(&conn)
                .expect("should get translated BED tree")
                .iter()
                .count(),
            1
        );
    }

    #[test]
    fn test_bed_annotation_preserves_negative_strand_coordinates() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed");
        let annotation = parse_bed_annotation(
            "xyz.1",
            File::open(path).expect("should open simple BED fixture"),
        )
        .expect("should parse the negative-strand BED record");
        assert_eq!(annotation.name, "xyz.1");

        let translated_annotation = translate_bed_annotation(
            &context,
            &annotation,
            File::open(path).expect("should reopen simple BED fixture"),
        )
        .expect("should translate the negative-strand BED record");
        let accession = translated_annotation
            .get_accession(&conn)
            .expect("should return translated BED accession");
        assert_eq!(accession.name, "xyz.1");
        let annotation_tree = translated_annotation
            .intervaltree(&conn)
            .expect("should get the cached BED annotation tree");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should get the cached BED accession tree");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert!(Arc::ptr_eq(
            &annotation_tree,
            translated_annotation
                .cached_interval_tree()
                .expect("should find the cached BED annotation tree")
        ));
        assert!(Arc::ptr_eq(
            &accession_tree,
            accession
                .cached_interval_tree()
                .expect("should find the cached BED accession tree")
        ));
        assert_eq!(accession.length(&conn).expect("should get BED length"), 3);
        let cloned_annotation = translated_annotation.clone();
        assert!(Arc::ptr_eq(
            &annotation_tree,
            cloned_annotation
                .cached_interval_tree()
                .expect("should clone the cached BED annotation tree")
        ));
        let cloned_accession = accession.clone();
        assert!(Arc::ptr_eq(
            &accession_tree,
            cloned_accession
                .cached_interval_tree()
                .expect("should clone the cached BED accession tree")
        ));

        let intervals: Vec<_> = annotation_tree.iter().collect();
        assert_eq!(intervals.len(), 1, "should emit one node interval");

        let interval = intervals[0];
        assert_eq!(interval.range, 0..3);
        assert_eq!(
            interval.value.node_id,
            HashId::try_from("b59698a422128d20462c44537b2d23ef")
                .expect("should parse the translated node id")
        );
        assert_eq!(interval.value.start, 0);
        assert_eq!(interval.value.end, 3);
        assert_eq!(interval.value.sequence_start, 5);
        assert_eq!(interval.value.sequence_end, 8);
        assert_eq!(interval.value.strand, Strand::Reverse);

        // Stored path node and reverse-oriented BED feature, with one ruler column per base:
        //   index tens:  00000000001111111
        //   index ones:  01234567890123456
        //   node DNA:    ATCGATCGATCGATCGA
        //   edge strand: +++++++++++++++++
        //   node span:   ^----------------^ [0,17)
        //   feature -:        ^--^ BED [5,8) selects raw TCG.
        //
        //   feature offset: 012
        //   feature DNA:    CGA
        //   interior [1,3):  ^-^ = GA; it maps to raw [5,7) TC.
        let resolved = ResolvedGenRegion::from_annotation(&conn, &translated_annotation, 0, 3)
            .expect("should resolve the negative-strand BED feature");
        let graph_locus = resolved
            .graph_locus(&conn, crate::test_helpers::test_workspace())
            .expect("should build the negative-strand BED graph locus");
        assert_eq!(
            graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
            b"CGA",
            "negative-strand BED retrieval should reverse-complement raw path TCG"
        );

        let resolved = ResolvedGenRegion::from_annotation(&conn, &translated_annotation, 1, 3)
            .expect("should resolve the negative-strand BED interior");
        let graph_locus = resolved
            .graph_locus(&conn, crate::test_helpers::test_workspace())
            .expect("should build the trimmed negative-strand BED graph locus");
        assert_eq!(
            graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
            b"GA",
            "negative-strand BED interior [1,3) should map to raw [5,7) and fetch GA"
        );
    }

    #[test]
    fn test_translates_reverse_split_bed_to_exact_node_local_spans() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        // Path edges traverse B then A; edge endpoint strands at B/A are -. Feature strand is +.
        //   path index:  01234567890123456
        //   path nodes:  BBBBBBBBAAAAAAAAA
        //   path DNA:    TGCAACGGTAACCGGTT
        //   feature:            ^^ [7,9) = GT
        //
        // Stored node B:
        //   index:     012345678
        //   DNA:       CCGTTGCAT
        //   span:      ^-------^ [0,8)
        //   selected:  ^ [0,1) = C
        //
        // Stored node A:
        //   index:     012345678
        //   DNA:       AACCGGTTA
        //   span:      ^--------^ [0,9)
        //   selected:          ^ [8,9) = A
        // Edge strand - maps stored C to G and A to T; feature strand + emits fragments B then A.
        // BED [7,9) maps to path [7,9), producing B [0,1) (-) then A [8,9) (-).
        // This explains orientation; the assertion checks emitted records, not fetched DNA.
        let input = format!("{}\t7\t9\tcross-boundary\t0\t+\n", block_group.name);
        let mut translated = Vec::new();
        translate_bed(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate BED input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 BED output");
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\t0\t1\tcross-boundary\t0\t-\n{node_a_id}\t8\t9\tcross-boundary\t0\t-\n"
            ),
            "positive-coordinate BED fragments should clip to each reverse-oriented node in path order"
        );
    }

    #[test]
    fn test_resolves_minus_strand_bed_from_forward_and_reverse_path_dna() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);

        // The same minus-strand feature is fetched through both fixture edge traversals.
        for ((block_group, node_a_id, node_b_id), name, expected_sequence) in [
            (forward, "forward", b"GGTTTGCAA".as_slice()),
            (reverse, "reverse", b"GTTACCGTT".as_slice()),
        ] {
            let context = AnnotationTranslationContext {
                conn: &conn,
                workspace: crate::test_helpers::test_workspace(),
                collection_name: "test",
                sample_name: Sample::DEFAULT_NAME,
                block_group_id: block_group.id,
                history_ref: None,
            };
            let identifier = format!("minus-{name}");
            let input = format!("{name}\t3\t12\t{identifier}\t0\t-\n");
            let annotation =
                parse_bed_annotation(identifier.as_str(), Cursor::new(input.as_bytes()))
                    .expect("should parse minus-strand BED feature");
            let annotation =
                translate_bed_annotation(&context, &annotation, Cursor::new(input.as_bytes()))
                    .expect("should translate minus-strand BED feature");
            let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 9)
                .expect("should resolve minus-strand BED feature");
            let graph_locus = resolved
                .graph_locus(&conn, crate::test_helpers::test_workspace())
                .expect("should build minus-strand BED graph locus");
            assert_eq!(
                graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                expected_sequence,
                "minus-strand BED sequence should be fetched in feature orientation on the {name} path"
            );
            assert_eq!(
                graph_locus
                    .slices
                    .iter()
                    .map(|slice| slice.block.node_id)
                    .collect::<Vec<_>>(),
                [node_a_id, node_b_id],
                "minus-strand BED slices should follow feature order on the {name} path"
            );

            for (start, end, expected_base) in [
                (0, 1, &expected_sequence[..1]),
                (8, 9, &expected_sequence[8..9]),
            ] {
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, start, end)
                    .expect("should resolve one minus-strand BED base");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build one-base minus-strand BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_base,
                    "one-base BED slices should retain feature orientation on the {name} path"
                );
            }

            // These are two separate path inputs; both features have strand -.
            // Case 1: edge endpoints at B/A +.
            //   feature nodes:   AAAABBBBB
            //   feature offset:  012345678
            //   feature DNA:     GGTTTGCAA
            //   query [1,3):      ^-^ = GT
            //   query [2,6):       ^---^ = TTTG
            // Case 2: edge endpoints at B/A -.
            //   feature nodes:   AAAABBBBB
            //   feature offset:  012345678
            //   feature DNA:     GTTACCGTT
            //   query [1,3):      ^-^ = TT
            //   query [2,6):       ^---^ = TACC
            // [1,3) checks an interior slice of A; [2,6) trims across A/B. These independent
            // queries overlap at offset 2 (T), included in both results.
            for (start, end, expected_forward, expected_reverse) in [
                (1, 3, b"GT".as_slice(), b"TT".as_slice()),
                (2, 6, b"TTTG".as_slice(), b"TACC".as_slice()),
            ] {
                let expected_subsequence = if name == "forward" {
                    expected_forward
                } else {
                    expected_reverse
                };
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, start, end)
                    .expect("should resolve a minus-strand BED subrange");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build the trimmed minus-strand BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_subsequence,
                    "minus-strand BED subrange {start}..{end} should trim in feature orientation on the {name} path"
                );
                if (start, end) == (2, 6) {
                    let actual_slices = graph_locus
                        .slices
                        .iter()
                        .map(|slice| {
                            (
                                slice.block.node_id,
                                slice.block.sequence_start,
                                slice.block.sequence_end,
                                slice.start,
                                slice.end,
                                slice.strand,
                            )
                        })
                        .collect::<Vec<_>>();
                    let expected_slices = if name == "forward" {
                        vec![
                            (node_a_id, 0, 4, 0, 2, Strand::Reverse),
                            (node_b_id, 3, 8, 3, 5, Strand::Reverse),
                        ]
                    } else {
                        vec![
                            (node_a_id, 5, 9, 2, 4, Strand::Forward),
                            (node_b_id, 0, 5, 0, 2, Strand::Forward),
                        ]
                    };
                    assert_eq!(
                        actual_slices, expected_slices,
                        "boundary-crossing BED subrange should trim node-local slices in feature order"
                    );
                }
            }
        }
    }

    #[test]
    fn test_multipart_minus_bed_cache_preserves_supplied_record_order() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        // The two records are supplied low then high on the reverse-traversed path.
        //   path index: 01234567890123456
        //   path nodes: BBBBBBBBAAAAAAAAA
        //   path DNA:   TGCAACGGTAACCGGTT
        //   BED [2,4) - selects CA and yields feature DNA TG.
        //   BED [10,12) - selects AC and yields feature DNA GT.
        // The adapter assembles those records in supplied order: TG + GT = TGGT.
        let input = format!(
            "{}\t2\t4\tminus-multipart\t0\t-\n{}\t10\t12\tminus-multipart\t0\t-\n",
            block_group.name, block_group.name
        );
        let mut translated = Vec::new();
        translate_bed(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate multipart BED records");
        assert_eq!(
            String::from_utf8(translated).expect("should write UTF-8 BED output"),
            format!(
                "{node_b_id}\t4\t6\tminus-multipart\t0\t+\n{node_a_id}\t5\t7\tminus-multipart\t0\t+\n"
            ),
            "translated BED fragments should stay in positive path order"
        );

        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let annotation = parse_bed_annotation("minus-multipart", Cursor::new(input.as_bytes()))
            .expect("should parse multipart minus-strand BED records");
        let annotation =
            translate_bed_annotation(&context, &annotation, Cursor::new(input.as_bytes()))
                .expect("should translate multipart minus-strand BED records");
        let tree = annotation
            .intervaltree(&conn)
            .expect("should read multipart BED cache");
        let mut spans = tree
            .iter()
            .map(|entry| (entry.range.clone(), entry.value))
            .collect::<Vec<_>>();
        spans.sort_by_key(|(range, _)| range.start);
        assert_eq!(
            spans
                .iter()
                .map(|(range, block)| (
                    range.clone(),
                    block.node_id,
                    block.sequence_start,
                    block.sequence_end,
                    block.strand,
                ))
                .collect::<Vec<_>>(),
            [
                (0..2, node_b_id, 4, 6, Strand::Forward),
                (2..4, node_a_id, 5, 7, Strand::Forward),
            ],
            "minus-strand fragments should preserve supplied record order with positive stored coordinates"
        );
        let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 4)
            .expect("should resolve multipart minus-strand BED annotation");
        let graph_locus = resolved
            .graph_locus(&conn, crate::test_helpers::test_workspace())
            .expect("should build multipart minus-strand BED graph locus");
        assert_eq!(
            graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
            b"TGGT",
            "multipart minus-strand BED cache should assemble records in supplied order"
        );
    }

    #[test]
    fn test_multipart_bed_cache_preserves_input_record_order() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);

        // Each input has three records in ascending or descending coordinate order. The adapter
        // preserves that order and reverses fragments only within an individual minus record.
        // Stored node B:
        //   index:   012345678
        //   DNA:     CCGTTGCAT
        //   used:    ^-------^ [0,8)
        // Stored node A:
        //   index:   012345678
        //   DNA:     AACCGGTTA
        //   used:    ^--------^ [0,9)
        // Edge endpoint strands at B/A +:
        //   offset:  01234567890123456
        //   nodes:   BBBBBBBBAAAAAAAAA
        //   path:    CCGTTGCAAACCGGTTA
        //   parts:   ..LL.MM...HH.....
        //   BED:     L [2,4)=GT, M [5,7)=GC, H [10,12)=CC
        //   feature +: L+M+H = GTGCCC; H+M+L = CCGCGT.
        //   feature -: L+M+H = ACGCGG; H+M+L = GGGCAC.
        // Edge endpoint strands at B/A -:
        //   offset:  01234567890123456
        //   nodes:   BBBBBBBBAAAAAAAAA
        //   path:    TGCAACGGTAACCGGTT
        //   parts:   ..LL.MM...HH.....
        //   BED:     L [2,4)=CA, M [5,7)=CG, H [10,12)=AC
        //   feature +: L+M+H = CACGAC; H+M+L = ACCGCA.
        //   feature -: L+M+H = TGCGGT; H+M+L = GTCGTG.
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);
        for (name, feature_strand, ascending_sequence, descending_sequence, fixture) in [
            (
                "forward-plus",
                "+",
                b"GTGCCC".as_slice(),
                b"CCGCGT".as_slice(),
                &forward,
            ),
            (
                "forward-minus",
                "-",
                b"ACGCGG".as_slice(),
                b"GGGCAC".as_slice(),
                &forward,
            ),
            (
                "reverse-plus",
                "+",
                b"CACGAC".as_slice(),
                b"ACCGCA".as_slice(),
                &reverse,
            ),
            (
                "reverse-minus",
                "-",
                b"TGCGGT".as_slice(),
                b"GTCGTG".as_slice(),
                &reverse,
            ),
        ] {
            let (block_group, node_a_id, node_b_id) = fixture;
            let identifier = format!("multipart-{name}");
            let parts = [(2, 4), (5, 7), (10, 12)];

            for (record_order, part_indices) in
                [("ascending", [0, 1, 2]), ("descending", [2, 1, 0])]
            {
                let expected_sequence = if record_order == "ascending" {
                    ascending_sequence
                } else {
                    descending_sequence
                };
                let input = part_indices
                    .into_iter()
                    .map(|index| {
                        let (start, end) = parts[index];
                        format!(
                            "{}\t{start}\t{end}\t{identifier}\t0\t{feature_strand}\n",
                            block_group.name
                        )
                    })
                    .collect::<String>();
                let context = AnnotationTranslationContext {
                    conn: &conn,
                    workspace: crate::test_helpers::test_workspace(),
                    collection_name: "test",
                    sample_name: Sample::DEFAULT_NAME,
                    block_group_id: block_group.id,
                    history_ref: None,
                };
                let annotation =
                    parse_bed_annotation(identifier.as_str(), Cursor::new(input.as_bytes()))
                        .expect("should parse multipart BED feature");
                let annotation =
                    translate_bed_annotation(&context, &annotation, Cursor::new(input.as_bytes()))
                        .expect("should translate multipart BED feature");
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 6)
                    .expect("should resolve multipart BED feature");
                assert_eq!(
                    resolved.feature_length, 6,
                    "should measure all three BED parts"
                );
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build multipart BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_sequence,
                    "{record_order} BED records should preserve feature DNA for {name}"
                );
                let expected_node_order = if record_order == "ascending" {
                    [*node_b_id, *node_b_id, *node_a_id]
                } else {
                    [*node_a_id, *node_b_id, *node_b_id]
                };
                assert_eq!(
                    graph_locus
                        .slices
                        .iter()
                        .map(|slice| slice.block.node_id)
                        .collect::<Vec<_>>(),
                    expected_node_order,
                    "{record_order} BED records should keep the parts in feature order for {name}"
                );

                // This independent query trims the first and last bases across the three parts.
                //   offset   012345
                //   [1,5)    .^^^^.
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 1, 5)
                    .expect("should resolve trimmed multipart BED feature");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build trimmed multipart BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    &expected_sequence[1..5],
                    "{record_order} BED records should trim across feature parts for {name}"
                );
            }
        }
    }

    #[test]
    fn test_mixed_strand_bed_cache_preserves_record_order_and_local_orientation() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);

        // Stored node B:
        //   index: 012345678
        //   DNA:   CCGTTGCAT
        //   slice: ^-------^ [0,8)
        // Stored node A:
        //   index: 012345678
        //   DNA:   AACCGGTTA
        //   slice: ^--------^ [0,9)
        // Edge endpoints at B/A +:
        //   path index: 01234567890123456
        //   path nodes: BBBBBBBBAAAAAAAAA
        //   path DNA:   CCGTTGCAAACCGGTTA
        //   BED [3,12): TTGCAAACC; BED [14,16): TT.
        //   parts:      ...CCCCCCCCC..TT.
        //   case -,+:   ...---------..++.
        //   case +,-:   ...+++++++++..--.
        //   [3,12) - then [14,16) + => GGTTTGCAATT; reverse record order => TTGGTTTGCAA.
        //   [3,12) + then [14,16) - => TTGCAAACCAA; reverse record order => AATTGCAAACC.
        //   [0,2)=CC, [3,12)=TTGCAAACC, [14,16)=TT; the middle record crosses B/A.
        //   parts:      HH.MMMMMMMMM..TT.
        //   case +,-,+: ++.---------..++.
        //   case -,+,-: --.+++++++++..--.
        //   +,-,+ => CCGGTTTGCAATT; -,+,- => GGTTGCAAACCAA.
        // Edge endpoints at B/A -:
        //   path index: 01234567890123456
        //   path nodes: BBBBBBBBAAAAAAAAA
        //   path DNA:   TGCAACGGTAACCGGTT
        //   BED [3,12): AACGGTAAC; BED [14,16): GT.
        //   parts:      ...CCCCCCCCC..TT.
        //   case -,+:   ...---------..++.
        //   case +,-:   ...+++++++++..--.
        //   [3,12) - then [14,16) + => GTTACCGTTGT; reverse record order => GTGTTACCGTT.
        //   [3,12) + then [14,16) - => AACGGTAACAC; reverse record order => ACAACGGTAAC.
        //   [0,2)=TG, [3,12)=AACGGTAAC, [14,16)=GT; the middle record crosses B/A.
        //   parts:      HH.MMMMMMMMM..TT.
        //   case +,-,+: ++.---------..++.
        //   case -,+,-: --.+++++++++..--.
        //   +,-,+ => TGGTTACCGTTGT; -,+,- => CAAACGGTAACAC.
        // These are independent source records. Each record's strand orders only its own
        // projected fragments; the two-record query [1,10) and three-record query [1,12)
        // below independently trim the assembled feature.
        type ExpectedBlock = (HashId, i64, i64, Strand);
        type SourceRecord = (&'static str, i64, i64);
        type MixedCase = (
            &'static str,
            Vec<SourceRecord>,
            &'static [u8],
            Vec<ExpectedBlock>,
        );
        for ((block_group, node_a_id, node_b_id), path_name) in
            [(forward, "forward"), (reverse, "reverse")]
        {
            let cases: Vec<MixedCase> = if path_name == "forward" {
                vec![
                    (
                        "cross-minus-then-tail-plus",
                        vec![("-", 3, 12), ("+", 14, 16)],
                        b"GGTTTGCAATT",
                        vec![
                            (node_a_id, 0, 4, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Reverse),
                            (node_a_id, 6, 8, Strand::Forward),
                        ],
                    ),
                    (
                        "cross-plus-then-tail-minus",
                        vec![("+", 3, 12), ("-", 14, 16)],
                        b"TTGCAAACCAA",
                        vec![
                            (node_b_id, 3, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Forward),
                            (node_a_id, 6, 8, Strand::Reverse),
                        ],
                    ),
                    (
                        "tail-plus-then-cross-minus",
                        vec![("+", 14, 16), ("-", 3, 12)],
                        b"TTGGTTTGCAA",
                        vec![
                            (node_a_id, 6, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Reverse),
                        ],
                    ),
                    (
                        "tail-minus-then-cross-plus",
                        vec![("-", 14, 16), ("+", 3, 12)],
                        b"AATTGCAAACC",
                        vec![
                            (node_a_id, 6, 8, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Forward),
                        ],
                    ),
                    (
                        "alternating-plus-minus-plus",
                        vec![("+", 0, 2), ("-", 3, 12), ("+", 14, 16)],
                        b"CCGGTTTGCAATT",
                        vec![
                            (node_b_id, 0, 2, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Reverse),
                            (node_a_id, 6, 8, Strand::Forward),
                        ],
                    ),
                    (
                        "alternating-minus-plus-minus",
                        vec![("-", 0, 2), ("+", 3, 12), ("-", 14, 16)],
                        b"GGTTGCAAACCAA",
                        vec![
                            (node_b_id, 0, 2, Strand::Reverse),
                            (node_b_id, 3, 8, Strand::Forward),
                            (node_a_id, 0, 4, Strand::Forward),
                            (node_a_id, 6, 8, Strand::Reverse),
                        ],
                    ),
                ]
            } else {
                vec![
                    (
                        "cross-minus-then-tail-plus",
                        vec![("-", 3, 12), ("+", 14, 16)],
                        b"GTTACCGTTGT",
                        vec![
                            (node_a_id, 5, 9, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Forward),
                            (node_a_id, 1, 3, Strand::Reverse),
                        ],
                    ),
                    (
                        "cross-plus-then-tail-minus",
                        vec![("+", 3, 12), ("-", 14, 16)],
                        b"AACGGTAACAC",
                        vec![
                            (node_b_id, 0, 5, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Reverse),
                            (node_a_id, 1, 3, Strand::Forward),
                        ],
                    ),
                    (
                        "tail-plus-then-cross-minus",
                        vec![("+", 14, 16), ("-", 3, 12)],
                        b"GTGTTACCGTT",
                        vec![
                            (node_a_id, 1, 3, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Forward),
                        ],
                    ),
                    (
                        "tail-minus-then-cross-plus",
                        vec![("-", 14, 16), ("+", 3, 12)],
                        b"ACAACGGTAAC",
                        vec![
                            (node_a_id, 1, 3, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Reverse),
                        ],
                    ),
                    (
                        "alternating-plus-minus-plus",
                        vec![("+", 0, 2), ("-", 3, 12), ("+", 14, 16)],
                        b"TGGTTACCGTTGT",
                        vec![
                            (node_b_id, 6, 8, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Forward),
                            (node_a_id, 1, 3, Strand::Reverse),
                        ],
                    ),
                    (
                        "alternating-minus-plus-minus",
                        vec![("-", 0, 2), ("+", 3, 12), ("-", 14, 16)],
                        b"CAAACGGTAACAC",
                        vec![
                            (node_b_id, 6, 8, Strand::Forward),
                            (node_b_id, 0, 5, Strand::Reverse),
                            (node_a_id, 5, 9, Strand::Reverse),
                            (node_a_id, 1, 3, Strand::Forward),
                        ],
                    ),
                ]
            };

            for (case_name, records, expected_sequence, expected_blocks) in cases {
                let identifier = format!("mixed-{path_name}-{case_name}");
                let input = records
                    .into_iter()
                    .map(|(strand, start, end)| {
                        format!(
                            "{}\t{start}\t{end}\t{identifier}\t0\t{strand}\n",
                            block_group.name
                        )
                    })
                    .collect::<String>();
                let context = AnnotationTranslationContext {
                    conn: &conn,
                    workspace: crate::test_helpers::test_workspace(),
                    collection_name: "test",
                    sample_name: Sample::DEFAULT_NAME,
                    block_group_id: block_group.id,
                    history_ref: None,
                };
                let annotation =
                    parse_bed_annotation(identifier.as_str(), Cursor::new(input.as_bytes()))
                        .expect("should parse mixed-strand BED records");
                let annotation =
                    translate_bed_annotation(&context, &annotation, Cursor::new(input.as_bytes()))
                        .expect("should translate mixed-strand BED records");
                let tree = annotation
                    .intervaltree(&conn)
                    .expect("should read mixed-strand BED intervals");
                let mut actual_blocks = tree
                    .iter()
                    .map(|entry| {
                        (
                            entry.range.start,
                            entry.range.end,
                            entry.value.node_id,
                            entry.value.sequence_start,
                            entry.value.sequence_end,
                            entry.value.strand,
                        )
                    })
                    .collect::<Vec<_>>();
                actual_blocks.sort_by_key(|block| block.0);
                let mut feature_start = 0;
                let expected_blocks = expected_blocks
                    .into_iter()
                    .map(|(node_id, sequence_start, sequence_end, strand)| {
                        let feature_end = feature_start + sequence_end - sequence_start;
                        let block = (
                            feature_start,
                            feature_end,
                            node_id,
                            sequence_start,
                            sequence_end,
                            strand,
                        );
                        feature_start = feature_end;
                        block
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual_blocks, expected_blocks,
                    "{case_name} BED records should retain exact node coordinates and strands"
                );

                let feature_end = expected_sequence.len() as i64;
                let resolved =
                    ResolvedGenRegion::from_annotation(&conn, &annotation, 0, feature_end)
                        .expect("should resolve mixed-strand BED feature");
                assert_eq!(
                    resolved.feature_length, feature_end,
                    "should measure all supplied BED records"
                );
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build mixed-strand BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    expected_sequence,
                    "{case_name} BED records should retrieve their supplied feature sequence"
                );
                let trim_end = feature_end - 1;
                let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 1, trim_end)
                    .expect("should resolve an interior mixed-strand BED range");
                let graph_locus = resolved
                    .graph_locus(&conn, crate::test_helpers::test_workspace())
                    .expect("should build the trimmed mixed-strand BED graph locus");
                assert_eq!(
                    graph_locus.sequence(&conn, crate::test_helpers::test_workspace()),
                    &expected_sequence[1..trim_end as usize],
                    "{case_name} BED range should trim across the source-record and node boundaries"
                );
            }
        }
    }

    #[test]
    fn test_bed_annotation_uses_historical_path_after_current_path_deletion() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find the test block group");
        let initial_branch = active_branch(&conn).expect("should read the active branch");
        let history_ref = commit_all(&conn, "store BED annotation path")
            .expect("should commit the BED annotation path")
            .to_string();

        // BED interval [5,8) is the negative-strand feature on the first of two path nodes.
        //   tens:  00000000001111111
        //   ones:  01234567890123456
        //   node1: ATCGATCGATCGATCGA [0,17)
        //   edge:  +++++++++++++++++
        //   BED:   .....^^^......... [5,8), raw TCG fetches as CGA.
        // The following node covers path [17,34).
        // The historical path must supply this interval after the current row is deleted.
        let input = "m123\t5\t8\thistory-feature\t0\t-\n";
        let annotation = parse_bed_annotation("history-feature", Cursor::new(input.as_bytes()))
            .expect("should parse the negative-strand BED feature");

        conn.execute(
            "DELETE FROM paths WHERE block_group_id = ?1",
            [block_group.id],
        )
        .expect("should delete the current path row");
        assert!(
            BlockGroup::get_current_path(&conn, &block_group.id, None).is_err(),
            "should not find a current path after deletion"
        );
        let historical_block_group =
            BlockGroup::get_by_id(&conn, &block_group.id, Some(&history_ref))
                .expect("should load the block group from history");
        assert_eq!(historical_block_group, block_group);

        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: historical_block_group.id,
            history_ref: Some(&history_ref),
        };
        let translated =
            translate_bed_annotation(&context, &annotation, Cursor::new(input.as_bytes()))
                .expect("should translate the BED feature from its historical path");
        let tree = translated
            .intervaltree(&conn)
            .expect("should read the translated BED intervals");
        let intervals: Vec<_> = tree.iter().collect();
        assert_eq!(intervals.len(), 1, "should emit one historical interval");
        assert_eq!(intervals[0].range, 0..3);
        assert_eq!(
            intervals[0].value.node_id,
            HashId::try_from("b59698a422128d20462c44537b2d23ef")
                .expect("should parse the historical node id")
        );
        assert_eq!(intervals[0].value.sequence_start, 5);
        assert_eq!(intervals[0].value.sequence_end, 8);
        assert_eq!(
            intervals[0].value.strand,
            Strand::Reverse,
            "should preserve the feature's reverse strand"
        );

        let resolved = ResolvedGenRegion::from_annotation(&conn, &translated, 0, 3)
            .expect("should resolve the historical BED feature");
        let locus = resolved
            .graph_locus(&conn, crate::test_helpers::test_workspace())
            .expect("should build the historical BED graph locus");
        assert_eq!(
            locus.sequence(&conn, crate::test_helpers::test_workspace()),
            b"CGA",
            "should fetch the complete negative-strand BED feature"
        );
        assert!(
            BlockGroup::get_current_path(&conn, &block_group.id, None).is_err(),
            "historical translation should leave the current path deleted"
        );
        assert_eq!(
            active_branch(&conn).expect("should read the active branch"),
            initial_branch,
            "historical translation should not switch the active branch"
        );
    }
}
