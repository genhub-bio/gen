//! GFF parsing and translation adapters.
//!
//! This serves as a general mechanism for parsing GFF records from any source implementing the
//! Read trait and converting them into Annotation objects. This lets us provide GFF files from
//! a variety of sources.

use std::{
    io::{self, BufReader, Read},
    sync::Arc,
};

use gen_core::{HashId, Strand};
use gen_models::{
    ForeignKeyCache,
    accession::{Accession, AccessionSpan, NewAccession},
    annotations::{
        Annotation, AnnotationExtra, AnnotationGroupSample, GffAttribute, GffExtra, NewAnnotation,
    },
    interval_tree::IntervalTreeSource as _,
};
use noodles::gff;

use super::{AnnotationTranslationContext, FileAnnotationError};

/// Imports translated GFF features as database annotations for a selected sample.
pub fn import_gff_annotations<R: Read>(
    context: &mut AnnotationTranslationContext<'_>,
    reader: R,
    group: &str,
    sample_name: &str,
) -> Result<(), FileAnnotationError> {
    let translated = translate_gff_annotation(context, reader)
        .filter_map(|result| match result {
            Ok(annotation) => Some(Ok(annotation)),
            Err(FileAnnotationError::Empty) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut persisted_annotations = Vec::new();
    for annotation in translated {
        let interval_tree = annotation
            .intervaltree(context.conn)
            .map_err(|error| FileAnnotationError::Persistence(error.to_string()))?;
        let spans = interval_tree
            .iter()
            .map(|entry| AccessionSpan {
                node_id: entry.value.node_id,
                range: gen_core::range::Range {
                    start: entry.value.sequence_start,
                    end: entry.value.sequence_end,
                },
                strand: entry.value.strand,
            })
            .collect();
        let accession = Accession::get_or_create(
            context.conn,
            &NewAccession {
                name: annotation.id.to_string(),
                block_group_id: annotation
                    .get_accession(context.conn)
                    .map_err(|error| FileAnnotationError::Persistence(error.to_string()))?
                    .block_group_id,
                parent_accession_id: None,
                spans,
            },
        )
        .map_err(|error| FileAnnotationError::Persistence(error.to_string()))?;
        persisted_annotations.push((annotation.name, accession.id, annotation.extra));
    }
    let new_annotations = persisted_annotations
        .iter()
        .map(|(name, accession_id, extra)| NewAnnotation {
            name,
            accession_id: *accession_id,
            extra: extra.as_ref(),
        })
        .collect::<Vec<_>>();
    Annotation::bulk_create(context.conn, group, &new_annotations)
        .map_err(|error| FileAnnotationError::Persistence(error.to_string()))?;
    AnnotationGroupSample::create(context.conn, group, sample_name)
        .map_err(|error| FileAnnotationError::Persistence(error.to_string()))
}

/// Lazily translates each GFF record into a graph-resolved annotation.
///
/// Each yielded annotation contains metadata and mapped intervals from one input record.
pub fn translate_gff_annotation<'context, 'graph, R>(
    context: &'context mut AnnotationTranslationContext<'graph>,
    reader: R,
) -> impl Iterator<Item = Result<Annotation, FileAnnotationError>> + 'context
where
    R: Read + 'context,
{
    let mut gff_reader = gff::io::Reader::new(BufReader::new(reader));

    std::iter::from_fn(move || {
        loop {
            let line = match gff_reader.lines().next()? {
                Ok(line) => line,
                Err(error) => return Some(Err(FileAnnotationError::Io(error))),
            };

            if let Some(gff::directive_buf::key::FASTA) = line
                .as_directive()
                .map(|directive| directive.key().as_ref())
            {
                return None;
            }

            let Some(record_result) = line.as_record() else {
                continue;
            };
            let record = match record_result {
                Ok(record) => record,
                Err(error) => return Some(Err(FileAnnotationError::Io(error))),
            };
            let record = match gff::feature::RecordBuf::try_from_feature_record(&record) {
                Ok(record) => record,
                Err(error) => return Some(Err(FileAnnotationError::Io(error))),
            };

            return Some(Ok((record, line)));
        }
    })
    .fuse()
    .map(move |record_result| {
        let (record, line) = record_result?;
        let raw_row: &[u8] = line.as_ref().as_ref();
        let mut annotation = annotation_from_record(&record, raw_row)?;
        let start = i64::try_from(record.start().get())
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?
            - 1;
        let end = i64::try_from(record.end().get())
            .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
        if end <= start {
            return Err(FileAnnotationError::Empty);
        }

        let reference_name = record.reference_sequence_name().to_string();
        let strand = Strand::from(record.strand());
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
    record: &gff::feature::RecordBuf,
    raw_row: &[u8],
) -> Result<Annotation, FileAnnotationError> {
    let attributes = record.attributes().as_ref();
    let name = ["Name", "ID", "gene", "gene_name", "db_xref"]
        .iter()
        .find_map(|key| {
            attributes
                .iter()
                .find(|(tag, _)| {
                    let tag_bytes: &[u8] = tag.as_ref();
                    tag_bytes.eq_ignore_ascii_case(key.as_bytes())
                })
                .and_then(|(_, value)| value.iter().next())
                .map(|value| String::from_utf8_lossy(value.as_ref()).into_owned())
                .filter(|name| !name.is_empty())
        })
        .ok_or(FileAnnotationError::MissingGffName)?;
    let source_id = attributes
        .iter()
        .filter(|(tag, _)| {
            let tag_bytes: &[u8] = tag.as_ref();
            tag_bytes.eq_ignore_ascii_case(b"ID")
        })
        .flat_map(|(_, value)| value.iter())
        .map(|value| String::from_utf8_lossy(value.as_ref()).into_owned())
        .find(|id| !id.is_empty());
    let id = if let Some(source_id) = source_id {
        HashId::convert_str(&source_id)
    } else {
        let raw_row = std::str::from_utf8(raw_row).map_err(|error| {
            FileAnnotationError::Io(io::Error::new(io::ErrorKind::InvalidData, error))
        })?;
        HashId::convert_str(raw_row)
    };
    let metadata = GffExtra {
        source: Some(record.source().to_string()),
        ty: record.ty().to_string(),
        score: record.score().map(|score| score.to_string()),
        phase: record.phase().map(|phase| match phase {
            gff::feature::record::Phase::Zero => "0".to_string(),
            gff::feature::record::Phase::One => "1".to_string(),
            gff::feature::record::Phase::Two => "2".to_string(),
        }),
        attributes: record
            .attributes()
            .as_ref()
            .iter()
            .map(|(tag, value)| GffAttribute {
                key: String::from_utf8_lossy(tag.as_ref()).into_owned(),
                values: value
                    .iter()
                    .map(|item| String::from_utf8_lossy(item.as_ref()).into_owned())
                    .collect(),
            })
            .collect(),
    };

    Ok(Annotation {
        id,
        name,
        group: "gff".to_string(),
        accession_id: id,
        cached_accession: ForeignKeyCache::default(),
        cached_interval_tree: None,
        history_ref: None,
        extra: Some(AnnotationExtra {
            gff: Some(metadata),
            ..AnnotationExtra::default()
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor, sync::Arc};

    use gen_core::{HashId, Strand};
    use gen_models::{
        accession::Accession, annotations::Annotation, interval_tree::IntervalTreeSource as _,
        region::ResolvedGenRegion, sample::Sample,
    };
    use noodles::gff;

    use super::{FileAnnotationError, annotation_from_record, translate_gff_annotation};
    use crate::{
        source::AnnotationTranslationContext,
        test_helpers::{
            create_reverse_test_block_group, get_connection, setup_test_data, test_workspace,
        },
        translate::gff::translate_gff,
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

    fn gff_record(input: &str) -> gff::feature::RecordBuf {
        gff::io::Reader::new(Cursor::new(input))
            .record_bufs()
            .next()
            .expect("should contain one GFF record")
            .expect("should parse one GFF record")
    }

    #[test]
    fn test_gff_record_conversion_rejects_missing_name_attribute() {
        let record = gff_record("m123\tsource\tgene\t1\t2\t.\t+\t.\t.\n");
        assert!(matches!(
            annotation_from_record(&record, b"missing-name-row"),
            Err(FileAnnotationError::MissingGffName)
        ));
    }

    #[test]
    fn test_gff_record_conversion_uses_later_nonempty_name_attribute() {
        let record = gff_record("m123\tsource\tgene\t1\t2\t.\t+\t.\tName=;ID=usable-id\n");
        let annotation = annotation_from_record(
            &record,
            b"m123\tsource\tgene\t1\t2\t.\t+\t.\tName=;ID=usable-id",
        )
        .expect("should use the next available GFF name attribute");

        assert_eq!(annotation.name, "usable-id");
    }

    #[test]
    fn test_gff_record_conversion_uses_id_attribute_for_identity() {
        let first_record =
            gff_record("m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;iD=stable-id\n");
        let first_annotation = annotation_from_record(
            &first_record,
            b"m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;iD=stable-id",
        )
        .expect("should convert GFF metadata without a graph or database");
        assert_eq!(first_annotation.name, "display-name");
        assert_eq!(first_annotation.id, HashId::convert_str("stable-id"));
        assert_eq!(first_annotation.accession_id, first_annotation.id);

        let second_record = gff_record(
            "m123\tsource\tgene\t3\t4\t.\t+\t.\tID=another-stable-id;Name=display-name\n",
        );
        let second_annotation = annotation_from_record(
            &second_record,
            b"m123\tsource\tgene\t3\t4\t.\t+\t.\tID=another-stable-id;Name=display-name",
        )
        .expect("should convert a second GFF record with the same display name");
        assert_eq!(second_annotation.name, "display-name");
        assert_eq!(
            second_annotation.id,
            HashId::convert_str("another-stable-id")
        );
        assert_ne!(first_annotation.id, second_annotation.id);
    }

    // Missing IDs hash the full row while the selected display name remains readable.
    #[test]
    fn test_gff_record_conversion_falls_back_to_row_hash_when_id_is_missing_or_empty() {
        let missing_id_row = "m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name";
        let missing_id_record = gff_record(&format!("{missing_id_row}\n"));
        let missing_id_annotation =
            annotation_from_record(&missing_id_record, missing_id_row.as_bytes())
                .expect("should convert a GFF record without an ID attribute");
        assert_eq!(missing_id_annotation.name, "display-name");
        assert_eq!(
            missing_id_annotation.id,
            HashId::convert_str(missing_id_row)
        );
        assert_eq!(missing_id_annotation.accession_id, missing_id_annotation.id);

        let empty_id_row = "m123\tsource\tgene\t3\t4\t.\t+\t.\tID=;gene_name=BRCA1";
        let empty_id_record = gff_record(&format!("{empty_id_row}\n"));
        let empty_id_annotation = annotation_from_record(&empty_id_record, empty_id_row.as_bytes())
            .expect("should use gene_name when the ID attribute is empty");
        assert_eq!(empty_id_annotation.name, "BRCA1");
        assert_eq!(empty_id_annotation.id, HashId::convert_str(empty_id_row));
        assert_eq!(empty_id_annotation.accession_id, empty_id_annotation.id);
    }

    #[test]
    fn test_gff_annotation_matches_metadata_and_resolves_slices() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let mut context = annotation_context(&conn);
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.gff");
        let mut annotations = translate_gff_annotation(
            &mut context,
            File::open(path).expect("should open simple GFF fixture"),
        );
        annotations
            .next()
            .expect("should yield the region record")
            .expect("should translate the region record");
        let annotation = annotations
            .next()
            .expect("should yield the gene record")
            .expect("should translate the gene record in one pass");

        assert_eq!(annotation.name, "gene-a0001");
        assert_eq!(annotation.id, HashId::convert_str("gene-a0001"));
        assert_eq!(annotation.accession_id, annotation.id);
        assert_eq!(annotation.group, "gff");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.gff.as_ref())
            .expect("should preserve GFF metadata");
        assert_eq!(metadata.ty, "Gene");
        assert!(
            metadata
                .attributes
                .iter()
                .any(|attribute| { attribute.key == "ID" && attribute.values == ["gene-a0001"] })
        );
        let first_attribute = metadata
            .attributes
            .first()
            .expect("should preserve the GFF ID attribute");
        assert_eq!(first_attribute.key, "ID");
        assert_eq!(first_attribute.values, vec!["gene-a0001".to_string()]);

        let accession = annotation
            .get_accession(&conn)
            .expect("should return translated GFF accession");
        assert_eq!(accession.block_group_id, block_group.id);
        assert_eq!(accession.id, annotation.accession_id);
        let annotation_tree = annotation
            .intervaltree(&conn)
            .expect("should get the cached GFF annotation tree");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should get the cached GFF accession tree");
        assert_eq!(annotation_tree.iter().count(), 2);
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert!(
            Annotation::select(&conn)
                .id(annotation.id)
                .load()
                .expect("should query absent file annotation")
                .is_empty()
        );
        assert!(
            Accession::select(&conn)
                .id(accession.id)
                .load()
                .expect("should query absent file accession")
                .is_empty()
        );
        assert_eq!(annotation.length(&conn).expect("should get GFF length"), 16);
        let mut accession_without_cache = accession.as_ref().clone();
        accession_without_cache.set_cached_interval_tree(None);
        let mut annotation_with_uncached_accession = annotation.clone();
        annotation_with_uncached_accession.set_accession(Arc::new(accession_without_cache));
        let resolved =
            ResolvedGenRegion::from_annotation(&conn, &annotation_with_uncached_accession, 12, 15)
                .expect("should resolve a cached GFF annotation without accession nodes")
                .find_graph_positions(&conn, test_workspace(), 0, 0)
                .expect("should resolve graph positions across translated slices");
        assert_eq!(resolved.anchor_end, 16);
        assert_eq!(resolved.feature_length, 16);
        assert_eq!(
            resolved
                .start_anchors
                .as_ref()
                .expect("should have start anchors")[0]
                .graph_node
                .node_id,
            HashId::try_from("b59698a422128d20462c44537b2d23ef")
                .expect("should parse the first translated node id")
        );
        assert_eq!(
            resolved
                .end_anchors
                .as_ref()
                .expect("should have end anchors")[0]
                .graph_node
                .node_id,
            HashId::try_from("6b460b727030cae3bae7cf389074d4ba")
                .expect("should parse the second translated node id")
        );
    }

    #[test]
    fn test_gff_reader_uses_each_record_reference_block_group() {
        let conn = get_connection();
        setup_test_data(&conn);
        let m123_block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find test block group");
        let (reverse_block_group, _, _) = create_reverse_test_block_group(&conn);
        let mut context = annotation_context(&conn);
        let input = format!(
            "m123\tsource\tgene\t1\t1\t.\t+\t.\tID=m123-feature\n{}\tsource\tgene\t1\t1\t.\t-\t.\tID=reverse-feature\n",
            reverse_block_group.name
        );
        let annotations = translate_gff_annotation(&mut context, input.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("should translate both GFF records");

        assert_eq!(annotations.len(), 2);
        let first_accession = annotations[0]
            .get_accession(&conn)
            .expect("should get the m123 accession");
        let second_accession = annotations[1]
            .get_accession(&conn)
            .expect("should get the reverse accession");
        assert_eq!(first_accession.block_group_id, m123_block_group.id);
        assert_eq!(second_accession.block_group_id, reverse_block_group.id);
        assert_ne!(annotations[0].accession_id, annotations[1].accession_id);
        assert_eq!(
            annotations[0]
                .intervaltree(&conn)
                .expect("should get the m123 tree")
                .iter()
                .count(),
            1
        );
        assert_eq!(
            annotations[1]
                .intervaltree(&conn)
                .expect("should get the reverse tree")
                .iter()
                .count(),
            1
        );
    }

    #[test]
    fn test_gff_reader_preserves_id_identity_and_falls_back_when_id_is_empty() {
        let conn = get_connection();
        setup_test_data(&conn);
        let mut context = annotation_context(&conn);
        let input = concat!(
            "m123\tsource\tgene\t1\t2\t.\t+\t.\tID=stable-id;Name=display-name\n",
            "m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;ID=another-id\n",
            "m123\tsource\tgene\t1\t2\t.\t+\t.\tID=;gene_name=BRCA1\n",
            "m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;Note=first\r\n",
            "m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;Note=second\n",
            "m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;Note=first",
        );
        let annotations = translate_gff_annotation(&mut context, input.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("should translate the GFF records");

        assert_eq!(annotations.len(), 6);
        assert_eq!(annotations[0].name, "display-name");
        assert_eq!(annotations[1].name, "display-name");
        assert_eq!(annotations[2].name, "BRCA1");
        assert_eq!(annotations[3].name, "display-name");
        assert_eq!(annotations[4].name, "display-name");
        assert_eq!(annotations[5].name, "display-name");
        assert_eq!(annotations[0].id, HashId::convert_str("stable-id"));
        assert_eq!(annotations[1].id, HashId::convert_str("another-id"));
        assert_eq!(
            annotations[2].id,
            HashId::convert_str("m123\tsource\tgene\t1\t2\t.\t+\t.\tID=;gene_name=BRCA1")
        );
        assert_eq!(
            annotations[3].id,
            HashId::convert_str("m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;Note=first")
        );
        assert_eq!(
            annotations[4].id,
            HashId::convert_str("m123\tsource\tgene\t1\t2\t.\t+\t.\tName=display-name;Note=second")
        );
        assert_eq!(annotations[3].id, annotations[5].id);
        assert_ne!(annotations[3].id, annotations[4].id);
        assert_ne!(annotations[0].id, annotations[1].id);

        for annotation in &annotations {
            assert_eq!(annotation.accession_id, annotation.id);
            let accession = annotation
                .get_accession(&conn)
                .expect("should get the attached accession");
            assert_eq!(accession.id, annotation.id);
        }
    }

    #[test]
    fn test_gff_reader_yields_errors_lazily_and_stops_at_fasta() {
        let conn = get_connection();
        setup_test_data(&conn);
        let mut context = annotation_context(&conn);
        let valid_then_invalid_then_valid = concat!(
            "m123\tsource\tgene\t1\t1\t.\t+\t.\tID=first-feature\n",
            "not a valid GFF record\n",
            "m123\tsource\tgene\t2\t2\t.\t+\t.\tID=after-error\n",
        );
        let mut annotations =
            translate_gff_annotation(&mut context, valid_then_invalid_then_valid.as_bytes());
        let first_annotation = annotations
            .next()
            .expect("should yield the valid first GFF record")
            .expect("should translate the valid first GFF record");
        assert_eq!(first_annotation.name, "first-feature");
        assert!(matches!(
            annotations
                .next()
                .expect("should yield the malformed second GFF record"),
            Err(super::FileAnnotationError::Io(_))
        ));
        assert_eq!(
            annotations
                .next()
                .expect("should yield the valid third GFF record")
                .expect("should translate the valid third GFF record")
                .name,
            "after-error"
        );
        assert!(annotations.next().is_none());
        assert!(annotations.next().is_none());

        let fasta = concat!(
            "m123\tsource\tgene\t1\t1\t.\t+\t.\tID=before-fasta\n",
            "##FASTA\n",
            ">m123\nACGT\n",
        );
        drop(annotations);
        let mut annotations = translate_gff_annotation(&mut context, fasta.as_bytes());
        assert_eq!(
            annotations
                .next()
                .expect("should yield the record before FASTA")
                .expect("should translate the record before FASTA")
                .name,
            "before-fasta"
        );
        assert!(annotations.next().is_none());
        assert!(annotations.next().is_none());
    }

    #[test]
    fn test_translates_reverse_split_gff_to_exact_node_local_spans() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        // Path edges traverse B then A with negative edge strands; input feature strand is +.
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
        let input = format!(
            "##gff-version 3\n{}\tsource\tgene\t8\t9\t.\t+\t.\tID=cross-boundary;Name=cross-boundary\n",
            block_group.name
        );
        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 GFF output");
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\tsource\tgene\t1\t1\t.\t+\t.\tID=cross-boundary;Name=cross-boundary\n{node_a_id}\tsource\tgene\t9\t9\t.\t+\t.\tID=cross-boundary;Name=cross-boundary\n"
            ),
            "GFF fragments should map to stored-node spans in path order and preserve the input feature strand"
        );
    }

    #[test]
    fn test_translates_single_base_gff_before_node_boundary() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, _, node_b_id) = create_reverse_test_block_group(&conn);
        // The reverse edge maps path position 7 to stored B [0,1); feature strand stays +.
        let input = format!(
            "{}\tsource\tgene\t8\t8\t.\t+\t.\tID=before-boundary;Name=before-boundary\n",
            block_group.name
        );
        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 GFF output");
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\tsource\tgene\t1\t1\t.\t+\t.\tID=before-boundary;Name=before-boundary\n"
            ),
            "GFF coordinate 8 should map to the final base of the first path node"
        );
    }

    #[test]
    fn test_translates_single_base_gff_after_node_boundary() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, _) = create_reverse_test_block_group(&conn);
        // The reverse edge maps path position 8 to stored A [8,9); feature strand stays +.
        let input = format!(
            "{}\tsource\tgene\t9\t9\t.\t+\t.\tID=after-boundary;Name=after-boundary\n",
            block_group.name
        );
        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        let translated = String::from_utf8(translated).expect("should write UTF-8 GFF output");
        assert_eq!(
            translated,
            format!(
                "{node_a_id}\tsource\tgene\t9\t9\t.\t+\t.\tID=after-boundary;Name=after-boundary\n"
            ),
            "GFF coordinate 9 should map to the first base of the second path node"
        );
    }
}
