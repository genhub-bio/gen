//! BED parsing and translation adapters.
//!
//! This serves as a general mechanism for parsing bed records from any source implementing the
//! Read trait and converting them into Annotation objects. This lets us provide BED files from
//! a variety of sources.

use std::{
    io::{BufRead, BufReader, Read},
    sync::Arc,
};

use gen_core::{HashId, Strand};
use gen_models::{
    ForeignKeyCache,
    accession::Accession,
    annotations::{Annotation, AnnotationExtra, BedExtra},
    interval_tree::IntervalTreeSource as _,
};
use noodles::bed;

use super::{AnnotationTranslationContext, FileAnnotationError};

/// Lazily translates each BED record into a graph-resolved annotation.
///
/// Each yielded annotation contains metadata and mapped intervals from one input record.
pub fn translate_bed_annotation<'context, 'graph, R>(
    context: &'context mut AnnotationTranslationContext<'graph>,
    reader: R,
) -> impl Iterator<Item = Result<Annotation, FileAnnotationError>> + 'context
where
    R: Read + 'context,
{
    let rows = BufReader::new(reader).lines();

    rows.fuse().map(move |row_result| {
        let mut row = row_result?;
        let raw_row_len = row.len();
        // BED6 needs an EOL to validate missing required fields on unterminated input lines.
        row.push('\n');
        let mut bed_reader = bed::io::Reader::<6, _>::new(row.as_bytes());
        let mut record = bed::Record::<6>::default();
        bed_reader.read_record(&mut record)?;

        let mut annotation = annotation_from_record(&record, &row[..raw_row_len])?;
        let start = i64::try_from(
            record
                .feature_start()
                .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
                .get(),
        )
        .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            - 1;
        let end = i64::try_from(
            record
                .feature_end()
                .ok_or_else(|| {
                    FileAnnotationError::Translation("BED record has no end".to_string())
                })?
                .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
                .get(),
        )
        .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
        if end <= start {
            return Err(FileAnnotationError::Empty);
        }

        let strand = match record.strand() {
            Ok(Some(bed::feature::record::Strand::Forward)) => Strand::Forward,
            Ok(Some(bed::feature::record::Strand::Reverse)) => Strand::Reverse,
            Ok(None) | Err(_) => Strand::Unknown,
        };
        let reference_name = record.reference_sequence_name().to_string();
        let (block_group_id, interval_tree) =
            context.intervaltree(&reference_name, start, end, strand)?;

        annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));
        let accession = Accession {
            id: annotation.accession_id,
            name: annotation.name.clone(),
            block_group_id,
            parent_accession_id: None,
            cached_interval_tree: Some(interval_tree),
            history_ref: None,
        };
        annotation.set_accession(Arc::new(accession));
        Ok(annotation)
    })
}

fn annotation_from_record(
    record: &bed::Record<6>,
    raw_row: &str,
) -> Result<Annotation, FileAnnotationError> {
    let name = record
        .name()
        .filter(|name| !name.is_empty())
        .map(|name| String::from_utf8_lossy(name.as_ref()).into_owned())
        .ok_or(FileAnnotationError::MissingBedName)?;
    let fields = record.other_fields();
    let metadata = BedExtra {
        score: Some(
            record
                .score()
                .map_err(FileAnnotationError::InvalidBedScore)?
                .to_string(),
        ),
        thick_start: fields
            .get(0)
            .and_then(|value| std::str::from_utf8(value).ok()?.parse().ok()),
        thick_end: fields
            .get(1)
            .and_then(|value| std::str::from_utf8(value).ok()?.parse().ok()),
        item_rgb: fields
            .get(2)
            .map(|value| String::from_utf8_lossy(value).into_owned()),
        block_count: fields
            .get(3)
            .and_then(|value| std::str::from_utf8(value).ok()?.parse().ok()),
        block_sizes: fields
            .get(4)
            .map(|value| parse_bed_integer_list(&String::from_utf8_lossy(value)))
            .filter(|values| !values.is_empty()),
        block_starts: fields
            .get(5)
            .map(|value| parse_bed_integer_list(&String::from_utf8_lossy(value)))
            .filter(|values| !values.is_empty()),
        other_fields: fields.iter().skip(6).map(ToString::to_string).collect(),
    };
    let id = HashId::convert_str(raw_row);

    Ok(Annotation {
        id,
        name,
        group: "bed".to_string(),
        accession_id: id,
        cached_accession: ForeignKeyCache::default(),
        cached_interval_tree: None,
        history_ref: None,
        extra: Some(AnnotationExtra {
            bed: Some(metadata),
            ..AnnotationExtra::default()
        }),
    })
}

fn parse_bed_integer_list(value: &str) -> Vec<i64> {
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
    use gen_models::{region::ResolvedGenRegion, sample::Sample};
    use noodles::bed;

    use super::{FileAnnotationError, annotation_from_record, translate_bed_annotation};
    use crate::{
        source::AnnotationTranslationContext,
        test_helpers::{
            create_reverse_test_block_group, get_connection, setup_test_data, test_workspace,
        },
        translate::bed::translate_bed,
    };

    fn annotation_context(
        conn: &gen_models::db::GraphConnection,
    ) -> AnnotationTranslationContext<'_> {
        AnnotationTranslationContext::new(
            conn,
            test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
        )
        .expect("should create annotation translation context")
    }

    fn bed_record(input: &str) -> bed::Record<6> {
        let mut reader = bed::io::reader::Builder::<6>.build_from_reader(Cursor::new(input));
        let mut record = bed::Record::<6>::default();
        reader
            .read_record(&mut record)
            .expect("should parse one BED record");
        record
    }

    #[test]
    fn test_bed_record_conversion_rejects_missing_name_and_invalid_score() {
        let unnamed_record = bed_record("m123\t0\t1\t\t0\t+\n");
        assert!(matches!(
            annotation_from_record(&unnamed_record, "unnamed row"),
            Err(FileAnnotationError::MissingBedName)
        ));

        let invalid_score_record = bed_record("m123\t0\t1\tfeature\tbad\t+\n");
        assert!(matches!(
            annotation_from_record(&invalid_score_record, "invalid score row"),
            Err(FileAnnotationError::InvalidBedScore(_))
        ));
    }

    #[test]
    fn test_bed_annotation_matches_identifier_and_preserves_metadata() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let mut context = annotation_context(&conn);
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed");
        let mut annotations = translate_bed_annotation(
            &mut context,
            File::open(path).expect("should open simple BED fixture"),
        );
        let annotation = annotations
            .next()
            .expect("should yield the first BED record")
            .expect("should translate the BED record in one pass");

        assert_eq!(annotation.name, "abc123.1");
        assert_eq!(
            annotation.id,
            HashId::convert_str(
                "m123\t1\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,"
            )
        );
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

        let accession = annotation
            .get_accession(&conn)
            .expect("should return the attached accession without a database row");
        assert_eq!(accession.block_group_id, block_group.id);
        let annotation_tree = annotation
            .intervaltree(&conn)
            .expect("should return the attached annotation tree");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should return the attached accession tree");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));

        // Ensure that we can get a non-empty set of slices from the annotation used as a region
        let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, 0, 1)
            .expect("should resolve a runtime annotation");
        assert!(
            !resolved
                .graph_locus(&conn, test_workspace())
                .expect("should resolve annotation graph slices")
                .slices
                .is_empty()
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
        let mut context = annotation_context(&conn);
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed");

        let mut annotations = translate_bed_annotation(
            &mut context,
            File::open(path).expect("should open simple BED fixture"),
        );
        annotations
            .next()
            .expect("should yield the first BED record")
            .expect("should translate the first BED record");
        let translated_annotation = annotations
            .next()
            .expect("should yield the negative-strand BED record")
            .expect("should translate the negative-strand BED record");
        assert_eq!(translated_annotation.name, "xyz.1");
        let accession = translated_annotation
            .get_accession(&conn)
            .expect("should return translated BED accession");
        assert_eq!(accession.name, "xyz.1");
        assert_eq!(accession.block_group_id, block_group.id);
        let annotation_tree = translated_annotation
            .intervaltree(&conn)
            .expect("should get the cached BED annotation tree");
        assert_eq!(accession.length(&conn).expect("should get BED length"), 3);

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
    }

    #[test]
    fn test_bed_reader_preserves_record() {
        let conn = get_connection();
        setup_test_data(&conn);
        let mut context = annotation_context(&conn);
        let input = concat!(
            "m123\t4\t8\tordered-feature\t10\t+\n",
            "m123\t0\t1\tordered-feature\t20\t-\n",
            "m123\t2\t3\tordered-feature\t30\t.\n",
            "m123\t0\t1\tordered-feature\t40\t+\r\n",
            "m123\t0\t1\tordered-feature\t41\t+\n",
            "m123\t0\t1\tordered-feature\t40\t+",
        );
        let annotations = translate_bed_annotation(&mut context, input.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("should translate all BED records");

        assert_eq!(annotations.len(), 6);
        assert!(
            annotations
                .iter()
                .all(|annotation| annotation.name == "ordered-feature")
        );
        assert_eq!(
            annotations[0].id,
            HashId::convert_str("m123\t4\t8\tordered-feature\t10\t+")
        );
        assert_eq!(
            annotations[1].id,
            HashId::convert_str("m123\t0\t1\tordered-feature\t20\t-")
        );
        assert_eq!(
            annotations[2].id,
            HashId::convert_str("m123\t2\t3\tordered-feature\t30\t.")
        );
        assert_eq!(
            annotations[3].id,
            HashId::convert_str("m123\t0\t1\tordered-feature\t40\t+")
        );
        assert_eq!(
            annotations[4].id,
            HashId::convert_str("m123\t0\t1\tordered-feature\t41\t+")
        );
        assert_eq!(annotations[3].id, annotations[5].id);
        assert_ne!(annotations[3].id, annotations[4].id);
        assert_eq!(annotations[3].accession_id, annotations[3].id);
        assert_eq!(annotations[4].accession_id, annotations[4].id);
        for annotation in [&annotations[3], &annotations[4], &annotations[5]] {
            let accession = annotation
                .get_accession(&conn)
                .expect("should get the attached BED accession");
            assert_eq!(accession.id, annotation.id);
        }

        for (annotation, expected_end, expected_strand, expected_sequence_range, expected_score) in [
            (&annotations[0], 4, Strand::Forward, 4..8, "10"),
            (&annotations[1], 1, Strand::Reverse, 0..1, "20"),
            (&annotations[2], 1, Strand::Unknown, 2..3, "30"),
            (&annotations[3], 1, Strand::Forward, 0..1, "40"),
            (&annotations[4], 1, Strand::Forward, 0..1, "41"),
            (&annotations[5], 1, Strand::Forward, 0..1, "40"),
        ] {
            let metadata = annotation
                .extra
                .as_ref()
                .and_then(|extra| extra.bed.as_ref())
                .expect("should preserve BED record metadata");
            assert_eq!(metadata.score.as_deref(), Some(expected_score));
            let tree = annotation
                .intervaltree(&conn)
                .expect("should get the BED record interval tree");
            let intervals: Vec<_> = tree.iter().collect();
            assert_eq!(intervals.len(), 1);
            assert_eq!(intervals[0].value.start, 0);
            assert_eq!(intervals[0].value.end, expected_end);
            assert_eq!(intervals[0].value.strand, expected_strand);
            assert_eq!(
                intervals[0].value.sequence_start..intervals[0].value.sequence_end,
                expected_sequence_range
            );
        }
    }

    #[test]
    fn test_bed_reader_yields_read_errors_and_continues() {
        let conn = get_connection();
        setup_test_data(&conn);
        let mut context = annotation_context(&conn);
        let input = b"m123\t0\t1\tfirst-feature\t0\t+\nm123\t1\nm123\t2\t3\tthird-feature\t0\t+\n";
        let mut annotations = translate_bed_annotation(&mut context, &input[..]);

        assert_eq!(
            annotations
                .next()
                .expect("should yield the valid first BED record")
                .expect("should translate the valid first BED record")
                .name,
            "first-feature"
        );
        assert!(matches!(
            annotations
                .next()
                .expect("should yield the malformed second BED record"),
            Err(super::FileAnnotationError::Io(_))
        ));
        assert_eq!(
            annotations
                .next()
                .expect("should yield the valid third BED record")
                .expect("should translate the valid third BED record")
                .name,
            "third-feature"
        );
        assert!(annotations.next().is_none());
        assert!(annotations.next().is_none());
    }

    #[test]
    fn test_translates_reverse_split_bed_to_exact_node_local_spans() {
        // This test has 2 nodes connected by a negative edge strand while the feature's annotated strand is +.
        // So this is a translation with a graph of the negative strand with a feature annotated on the + strand.
        // Path edges traverse B then A
        //   path index:  01234567890123456
        //   path nodes:  BBBBBBBBAAAAAAAAA
        //   path DNA:    TGCAACGGTAACCGGTT
        //   feature:            ^^ [7,9) = GT
        //
        // Stored node B:
        //   index:     012345678
        //   DNA:       CCGTTGCAT
        //   span:      ^-------^ [0,8)
        //   feature:   ^ [0,1) = C
        //
        // Stored node A:
        //   index:     012345678
        //   DNA:       AACCGGTTA
        //   span:      ^--------^ [0,9)
        //   feature:           ^ [8,9) = A

        // BED [7,9) maps to path [7,9), producing B [0,1) then A [8,9) with feature strand +.
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
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
                "{node_b_id}\t0\t1\tcross-boundary\t0\t+\n{node_a_id}\t8\t9\tcross-boundary\t0\t+\n"
            ),
            "BED fragments should map to stored-node spans in path order and preserve the input feature strand"
        );
    }
}
