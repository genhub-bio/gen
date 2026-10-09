//! GFF parsing and translation adapters.
//!
//! The source of the GFF file comes from the caller, this file works off a readable input.
//! The readers here carry out parsing that input, matching identifiers, and returning an
//! Annotation object for use downstream.

use std::{
    io::{BufRead, Cursor, Read},
    sync::Arc,
};

use gen_core::{HashId, NodeIntervalBlock};
use gen_models::{
    accession::Accession,
    annotations::{Annotation, AnnotationExtra, GffAttribute, GffExtra},
    interval_tree::IntervalTreeSource as _,
};
use intervaltree::IntervalTree;
use noodles::gff;

use super::{AnnotationTranslationContext, FileAnnotationError};
use crate::{gff::gff_attribute_value_to_string, translate::gff::translate_gff};

/// Parse one identifier's matching GFF records into the existing annotation model.
pub fn parse_gff_annotation<R>(
    identifier: impl Into<String>,
    reader: R,
) -> Result<Annotation, FileAnnotationError>
where
    R: Read + BufRead,
{
    let identifier = identifier.into();
    let records = gff::io::Reader::new(reader)
        .record_bufs()
        .filter_map(|result| match result {
            Ok(record) => matching_gff_record(&record, &identifier).then_some(Ok(record)),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    parse_gff_annotation_records(identifier, records)
}

/// Build an annotation from records already matched by an upstream lookup provider.
pub fn parse_gff_annotation_records<I>(
    identifier: impl Into<String>,
    records: I,
) -> Result<Annotation, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let identifier = identifier.into();
    let records: Vec<_> = records.into_iter().collect();
    let first = records.first().ok_or(FileAnnotationError::Empty)?;
    let gff = GffExtra {
        source: Some(first.source().to_string()),
        ty: first.ty().to_string(),
        score: first.score().map(|score| score.to_string()),
        phase: first.phase().map(|phase| match phase {
            gff::feature::record::Phase::Zero => "0".to_string(),
            gff::feature::record::Phase::One => "1".to_string(),
            gff::feature::record::Phase::Two => "2".to_string(),
        }),
        attributes: records
            .iter()
            .flat_map(|record| {
                record
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
            })
            .collect(),
    };
    let id = HashId::convert_str(&identifier);
    Ok(Annotation {
        id,
        name: identifier,
        group: "gff".to_string(),
        accession_id: id,
        cached_interval_tree: None,
        extra: Some(AnnotationExtra {
            gff: Some(gff),
            ..AnnotationExtra::default()
        }),
    })
}

/// Translate matching GFF records and cache their graph intervals on the annotation.
///
/// Returns the associated accession.
pub fn translate_gff_annotation<R>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &mut Annotation,
    reader: R,
) -> Result<Accession, FileAnnotationError>
where
    R: Read + BufRead,
{
    let records = gff::io::Reader::new(reader)
        .record_bufs()
        .filter_map(|result| match result {
            Ok(record) => matching_gff_record(&record, &annotation.name).then_some(Ok(record)),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>, _>>()?;
    translate_gff_annotation_records(context, annotation, records)
}

/// Translate selected GFF records and cache their graph intervals on the annotation.
///
/// Returns the associated accession.
pub fn translate_gff_annotation_records<I>(
    context: &AnnotationTranslationContext<'_>,
    annotation: &mut Annotation,
    records: I,
) -> Result<Accession, FileAnnotationError>
where
    I: IntoIterator<Item = gff::feature::RecordBuf>,
{
    let mut input = Vec::new();
    {
        let mut writer = gff::io::Writer::new(&mut input);
        for record in records {
            writer.write_record(&record)?;
        }
    }
    let mut translated = Vec::new();
    translate_gff(
        context.conn,
        context.workspace,
        context.collection_name,
        context.sample_name,
        context.history_ref,
        Cursor::new(input),
        &mut translated,
    )
    .map_err(|error| FileAnnotationError::Translation(error.to_string()))?;
    let interval_tree = Arc::new(translated_gff_interval_tree(&translated)?);
    annotation.set_cached_interval_tree(Some(Arc::clone(&interval_tree)));
    let mut accession = Accession {
        id: annotation.accession_id,
        name: annotation.name.clone(),
        block_group_id: context.block_group_id,
        parent_accession_id: None,
        cached_interval_tree: None,
    };
    accession.set_cached_interval_tree(Some(interval_tree));
    Ok(accession)
}

fn matching_gff_record(record: &gff::feature::RecordBuf, identifier: &str) -> bool {
    ["Name", "ID", "gene", "gene_name", "db_xref"]
        .iter()
        .any(|key| {
            gff_attribute_value_to_string(record.attributes(), key)
                .is_some_and(|value| value.eq_ignore_ascii_case(identifier))
        })
}

fn translated_gff_interval_tree(
    translated: &[u8],
) -> Result<IntervalTree<i64, NodeIntervalBlock>, FileAnnotationError> {
    let mut reader = gff::io::Reader::new(Cursor::new(translated));
    let mut position = 0;
    let mut blocks = Vec::new();
    for result in reader.record_bufs() {
        let record = result?;
        let node_name = record.reference_sequence_name().to_string();
        let node_id = HashId::try_from(node_name.as_str())
            .map_err(|_| FileAnnotationError::InvalidNode(node_name.clone()))?;
        let start = record.start().get() as i64 - 1;
        let end = record.end().get() as i64;
        if end <= start {
            continue;
        }
        let block = NodeIntervalBlock {
            node_id,
            start: position,
            end: position + end - start,
            sequence_start: start,
            sequence_end: end,
            strand: record.strand().into(),
        };
        position = block.end;
        blocks.push(block);
    }
    if blocks.is_empty() {
        return Err(FileAnnotationError::Empty);
    }
    Ok(blocks
        .into_iter()
        .map(|block| (block.start..block.end, block))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Cursor},
        sync::Arc,
    };

    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        accession::Accession,
        block_group::{BlockGroup, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        edge::Edge,
        interval_tree::IntervalTreeSource as _,
        node::Node,
        path::Path,
        region::ResolvedGenRegion,
        sample::Sample,
        sequence::Sequence,
    };

    use super::{AnnotationTranslationContext, parse_gff_annotation, translate_gff_annotation};
    use crate::{
        parse_bed_annotation,
        translate::{bed::translate_bed, gff::translate_gff},
        translate_bed_annotation,
    };

    fn create_two_node_test_block_group(
        conn: &gen_models::db::GraphConnection,
        name: &str,
        strand: Strand,
        sequence_a_text: &str,
        sequence_b_text: &str,
    ) -> (BlockGroup, HashId, HashId) {
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: Sample::DEFAULT_NAME,
                name,
                parent_block_group_id: None,
                is_default: false,
            },
        )
        .expect("should create test block group");
        let sequence_a = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence_a_text)
            .save(conn)
            .expect("should save first path sequence");
        let node_a_id = HashId::convert_str(&format!("{name}-path-node-a"));
        Node::create(conn, &sequence_a.hash, &node_a_id).expect("should create first path node");
        let sequence_b = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence_b_text)
            .save(conn)
            .expect("should save second path sequence");
        let node_b_id = HashId::convert_str(&format!("{name}-path-node-b"));
        Node::create(conn, &sequence_b.hash, &node_b_id).expect("should create second path node");

        let edge_start = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_b_id,
            0,
            strand,
        )
        .expect("should create path start edge");
        let edge_between = Edge::create(conn, node_b_id, 8, strand, node_a_id, 0, strand)
            .expect("should connect path nodes");
        let edge_end = Edge::create(
            conn,
            node_a_id,
            9,
            strand,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .expect("should create path end edge");
        let edge_ids = [edge_start.id, edge_between.id, edge_end.id];
        BlockGroupEdge::bulk_create(
            conn,
            &edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: 0,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        let path =
            Path::create(conn, name, &block_group.id, &edge_ids).expect("should create test path");
        let path_sequence = path
            .sequence(conn, crate::test_helpers::test_workspace(), None)
            .expect("should load test path sequence");
        assert_eq!(
            &path_sequence[7..9],
            "GT",
            "the split-node feature fixture should spell GT on the path"
        );

        (block_group, node_a_id, node_b_id)
    }

    fn create_reverse_test_block_group(
        conn: &gen_models::db::GraphConnection,
    ) -> (BlockGroup, HashId, HashId) {
        create_two_node_test_block_group(conn, "reverse", Strand::Reverse, "AACCGGTTA", "CCGTTGCAT")
    }

    fn create_forward_test_block_group(
        conn: &gen_models::db::GraphConnection,
    ) -> (BlockGroup, HashId, HashId) {
        create_two_node_test_block_group(conn, "forward", Strand::Forward, "TAACCGGTT", "ACGTTGCGT")
    }

    #[test]
    fn test_gff_annotation_matches_metadata_and_resolves_slices() {
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
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.gff");
        let mut annotation = parse_gff_annotation(
            "gene-a0001",
            BufReader::new(File::open(path).expect("should open simple GFF fixture")),
        )
        .expect("should parse the matching GFF record");
        assert_eq!(annotation.name, "gene-a0001");
        assert_eq!(annotation.id, HashId::convert_str("gene-a0001"));
        assert_eq!(annotation.accession_id, annotation.id);
        assert_eq!(annotation.group, "gff");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.gff.as_ref())
            .expect("should preserve GFF metadata");
        let first_attribute = metadata
            .attributes
            .first()
            .expect("should preserve the GFF ID attribute");
        assert_eq!(first_attribute.key, "ID");
        assert_eq!(first_attribute.values, vec!["gene-a0001".to_string()]);
        let accession = translate_gff_annotation(
            &context,
            &mut annotation,
            BufReader::new(File::open(path).expect("should reopen simple GFF fixture")),
        )
        .expect("should translate the matching GFF record");
        assert_eq!(accession.id, annotation.accession_id);
        let annotation_tree = annotation
            .intervaltree(&conn)
            .expect("should get the cached GFF annotation tree");
        let accession_tree = accession
            .intervaltree(&conn)
            .expect("should get the cached GFF accession tree");
        assert!(Arc::ptr_eq(&annotation_tree, &accession_tree));
        assert!(Arc::ptr_eq(
            &annotation_tree,
            annotation
                .cached_interval_tree()
                .expect("should find the cached GFF annotation tree")
        ));
        assert!(Arc::ptr_eq(
            &accession_tree,
            accession
                .cached_interval_tree()
                .expect("should find the cached GFF accession tree")
        ));
        assert_eq!(annotation_tree.iter().count(), 2);
        assert_eq!(annotation.length(&conn).expect("should get GFF length"), 16);
        assert!(
            Accession::select(&conn)
                .id(accession.id)
                .load()
                .expect("should query absent file accession")
                .is_empty()
        );

        let mut accession_without_cache = accession.clone();
        accession_without_cache.set_cached_interval_tree(None);
        let resolved = ResolvedGenRegion::from_annotation(
            &conn,
            &annotation,
            &accession_without_cache,
            12,
            15,
        )
        .expect("should resolve a cached GFF annotation without accession nodes")
        .find_graph_positions(&conn, crate::test_helpers::test_workspace(), 0, 0)
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

        let mut intervals = annotation_tree
            .iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>();
        intervals.sort_by_key(|interval| interval.start);
        assert_eq!(
            intervals
                .iter()
                .map(|interval| (
                    interval.node_id,
                    interval.sequence_start,
                    interval.sequence_end,
                    interval.strand,
                ))
                .collect::<Vec<_>>(),
            [
                (
                    HashId::try_from("b59698a422128d20462c44537b2d23ef").unwrap(),
                    4,
                    17,
                    Strand::Forward,
                ),
                (
                    HashId::try_from("6b460b727030cae3bae7cf389074d4ba").unwrap(),
                    0,
                    3,
                    Strand::Forward,
                ),
            ],
            "split feature overlaps should use node-local coordinates"
        );
    }

    #[test]
    fn test_gene_name_only_gff_record_parses_and_translates_case_insensitively() {
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
        let input = "m123\tsource\tgene\t5\t6\t.\t+\t.\tID=private-gene-name-id;gene_name=BRCA1\n";
        let mut annotation =
            parse_gff_annotation("brca1", BufReader::new(Cursor::new(input.as_bytes())))
                .expect("should match a gene_name-only GFF record ignoring case");
        assert_eq!(annotation.name, "brca1");
        let metadata = annotation
            .extra
            .as_ref()
            .and_then(|extra| extra.gff.as_ref())
            .expect("should preserve the gene_name-only GFF metadata");
        assert!(
            metadata
                .attributes
                .iter()
                .any(|attribute| { attribute.key == "gene_name" && attribute.values == ["BRCA1"] })
        );

        let accession = translate_gff_annotation(
            &context,
            &mut annotation,
            BufReader::new(Cursor::new(input.as_bytes())),
        )
        .expect("should translate the gene_name-only GFF record");
        assert_eq!(
            annotation.length(&conn).expect("should measure annotation"),
            2
        );
        assert_eq!(
            accession.length(&conn).expect("should measure accession"),
            2
        );
    }

    #[test]
    fn test_translates_reverse_split_gff_to_exact_node_local_spans() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: Sample::DEFAULT_NAME,
            block_group_id: block_group.id,
            history_ref: None,
        };
        let input = concat!(
            "##gff-version 3\n",
            "reverse\tsource\tgene\t8\t9\t.\t+\t.\tID=cross-boundary;Name=cross-boundary\n",
            "reverse\tsource\tgene\t8\t8\t.\t+\t.\tID=before-boundary;Name=before-boundary\n",
            "reverse\tsource\tgene\t9\t9\t.\t+\t.\tID=after-boundary;Name=after-boundary\n",
        );
        let mut annotation =
            parse_gff_annotation("cross-boundary", BufReader::new(Cursor::new(input)))
                .expect("should parse feature crossing reverse path nodes");
        let accession = translate_gff_annotation(
            &context,
            &mut annotation,
            BufReader::new(Cursor::new(input)),
        )
        .expect("should translate feature onto reverse path nodes");
        let intervals = annotation
            .intervaltree(&conn)
            .expect("should obtain translated cached intervals")
            .iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>();
        assert_eq!(
            intervals
                .iter()
                .map(|interval| (
                    interval.node_id,
                    interval.sequence_start,
                    interval.sequence_end,
                    interval.strand,
                ))
                .collect::<Vec<_>>(),
            [
                (node_b_id, 0, 1, Strand::Reverse),
                (node_a_id, 8, 9, Strand::Reverse),
            ],
            "the feature should preserve its path order with exact reverse node slices"
        );
        assert_eq!(
            accession
                .length(&conn)
                .expect("should measure split feature"),
            2
        );

        let mut translated = Vec::new();
        translate_gff(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            BufReader::new(Cursor::new(input)),
            &mut translated,
        )
        .expect("should translate one-base records on both sides of the path boundary");
        let translated = String::from_utf8(translated).expect("should write UTF-8 GFF output");
        assert!(translated.contains(&format!(
            "{}\tsource\tgene\t1\t1\t.\t-\t.\tID=before-boundary;Name=before-boundary\n",
            node_b_id
        )));
        assert!(translated.contains(&format!(
            "{}\tsource\tgene\t9\t9\t.\t-\t.\tID=after-boundary;Name=after-boundary\n",
            node_a_id
        )));
    }

    #[test]
    fn test_translates_reverse_split_bed_to_exact_node_local_spans() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let (block_group, node_a_id, node_b_id) = create_reverse_test_block_group(&conn);
        let input = b"reverse\t7\t9\tcross-boundary\t0\t+\n";
        let mut translated = Vec::new();
        translate_bed(
            &conn,
            crate::test_helpers::test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input),
            &mut translated,
        )
        .expect("should translate reverse BED feature across path nodes");

        let translated = String::from_utf8(translated).expect("should write UTF-8 BED output");
        assert_eq!(
            translated,
            format!(
                "{node_b_id}\t0\t1\tcross-boundary\t0\t-\n{node_a_id}\t8\t9\tcross-boundary\t0\t-\n"
            ),
            "BED overlaps should preserve path order and project exact reverse node-local spans"
        );
        assert_eq!(block_group.name, "reverse");
    }

    #[test]
    fn test_resolves_minus_strand_gff_as_reverse_complement_across_split_nodes() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);

        for (block_group, name) in [(forward.0, "forward"), (reverse.0, "reverse")] {
            let context = AnnotationTranslationContext {
                conn: &conn,
                workspace: crate::test_helpers::test_workspace(),
                collection_name: "test",
                sample_name: Sample::DEFAULT_NAME,
                block_group_id: block_group.id,
                history_ref: None,
            };
            let identifier = format!("minus-{name}");
            let input =
                format!("{name}\tsource\tgene\t8\t9\t.\t-\t.\tID={identifier};Name={identifier}\n");
            let mut annotation = parse_gff_annotation(
                identifier.as_str(),
                BufReader::new(Cursor::new(input.as_bytes())),
            )
            .expect("should parse minus-strand GFF feature");
            let accession = translate_gff_annotation(
                &context,
                &mut annotation,
                BufReader::new(Cursor::new(input.as_bytes())),
            )
            .expect("should translate minus-strand GFF feature");
            let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, 0, 2)
                .expect("should resolve minus-strand GFF feature")
                .graph_locus(&conn, crate::test_helpers::test_workspace())
                .expect("should build minus-strand GFF graph locus");
            assert_eq!(
                resolved.sequence(&conn, crate::test_helpers::test_workspace()),
                b"AC",
                "minus-strand GFF sequence should reverse-complement path bases GT on {name} path"
            );
        }
    }

    #[test]
    fn test_resolves_minus_strand_bed_as_reverse_complement_across_split_nodes() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let forward = create_forward_test_block_group(&conn);
        let reverse = create_reverse_test_block_group(&conn);

        for (block_group, name) in [(forward.0, "forward"), (reverse.0, "reverse")] {
            let context = AnnotationTranslationContext {
                conn: &conn,
                workspace: crate::test_helpers::test_workspace(),
                collection_name: "test",
                sample_name: Sample::DEFAULT_NAME,
                block_group_id: block_group.id,
                history_ref: None,
            };
            let identifier = format!("minus-{name}");
            let input = format!("{name}\t7\t9\t{identifier}\t0\t-\n");
            let mut annotation =
                parse_bed_annotation(identifier.as_str(), Cursor::new(input.as_bytes()))
                    .expect("should parse minus-strand BED feature");
            let accession =
                translate_bed_annotation(&context, &mut annotation, Cursor::new(input.as_bytes()))
                    .expect("should translate minus-strand BED feature");
            let resolved = ResolvedGenRegion::from_annotation(&conn, &annotation, &accession, 0, 2)
                .expect("should resolve minus-strand BED feature")
                .graph_locus(&conn, crate::test_helpers::test_workspace())
                .expect("should build minus-strand BED graph locus");
            assert_eq!(
                resolved.sequence(&conn, crate::test_helpers::test_workspace()),
                b"AC",
                "minus-strand BED sequence should reverse-complement path bases GT on {name} path"
            );
        }
    }
}
