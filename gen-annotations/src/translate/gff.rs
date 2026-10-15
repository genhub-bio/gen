use std::{
    collections::HashMap,
    io::{BufRead, Read, Write},
};

use gen_core::{Strand, Workspace};
use gen_models::{
    block_group::BlockGroup,
    db::GraphConnection,
    errors::{BlockGroupError, PathError},
    reference_alias::{ReferenceAlias, ReferenceAliasError},
    region::{GenRegionError, ResolvedGenRegion},
    sample::Sample,
};
use noodles::{core::Position, gff};
use thiserror::Error;

use super::compose_strand;

#[derive(Debug, Error)]
pub enum GffError {
    #[error("Couldn't translate GFF entry: {0}")]
    Io(#[from] std::io::Error),
    #[error("Error loading reference aliases: {0}")]
    ReferenceAliasError(#[from] ReferenceAliasError),
    #[error("Error loading path blocks: {0}")]
    PathError(#[from] PathError),
    #[error("Error loading block group graph: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("Error resolving path region: {0}")]
    RegionError(#[from] GenRegionError),
}

pub fn translate_gff<R, W>(
    conn: &GraphConnection,
    workspace: &Workspace,
    collection: &str,
    sample: &str,
    history_ref: Option<&str>,
    reader: R,
    writer: &mut W,
) -> Result<(), GffError>
where
    R: Read + BufRead,
    W: Write,
{
    let mut gff_reader = gff::io::Reader::new(reader);
    let mut gff_writer = gff::io::Writer::new(writer);
    let sample_block_groups = Sample::get_block_groups(conn, collection, sample, history_ref)
        .into_iter()
        .map(|block_group| (block_group.name, block_group.id))
        .collect::<HashMap<_, _>>();
    let references = sample_block_groups.keys().cloned().collect();
    let references_by_alias =
        ReferenceAlias::get_references_by_alias(conn, references, history_ref)?;
    let mut regions_by_reference = HashMap::<String, ResolvedGenRegion>::new();

    for result in gff_reader.record_bufs() {
        let record = result?;
        let ref_name = record.reference_sequence_name().to_string();
        let start = record.start().get() as i64 - 1;
        let end = record.end().get() as i64;
        let feature_strand: Strand = record.strand().into();
        let resolved_reference = references_by_alias
            .get(&ref_name)
            .map_or(ref_name.as_str(), String::as_str);
        let Some(block_group_id) = sample_block_groups.get(resolved_reference) else {
            continue;
        };
        let mut region = match regions_by_reference.get(resolved_reference) {
            Some(region) => region.clone(),
            None => {
                let path = BlockGroup::get_current_path(conn, block_group_id, history_ref)?;
                let region = ResolvedGenRegion::from_path(conn, &path, 0, 0)?;
                regions_by_reference.insert(resolved_reference.to_string(), region.clone());
                region
            }
        };
        region.start = start;
        region.end = end;
        let slices = region.graph_locus(conn, workspace)?.slices;
        for slice in slices {
            let mut updated_record_builder = gff::feature::RecordBuf::builder()
                .set_reference_sequence_name(format!("{nid}", nid = slice.block.node_id))
                .set_source(record.source().to_string())
                .set_type(record.ty().to_string())
                .set_start(
                    Position::try_from(slice.block.sequence_start as usize + slice.start + 1)
                        .expect("should convert the stored-node start to a GFF position"),
                )
                .set_end(
                    Position::try_from(slice.block.sequence_start as usize + slice.end)
                        .expect("should convert the stored-node end to a GFF position"),
                )
                .set_strand(compose_strand(feature_strand, slice.strand).into())
                .set_attributes(record.attributes().clone());
            if let Some(phase) = record.phase() {
                updated_record_builder = updated_record_builder.set_phase(phase);
            }
            gff_writer.write_record(&updated_record_builder.build())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Cursor},
        path::PathBuf,
    };

    use gen_core::Strand;
    use gen_models::{
        block_group::BlockGroup, history::dolt::commit_all, reference_alias::ReferenceAlias,
        sample::Sample,
    };

    use super::translate_gff;
    use crate::test_helpers::{
        create_sliced_test_block_group, get_connection, setup_test_data, test_workspace,
    };

    fn translate_gff_input(conn: &gen_models::db::GraphConnection, input: &str) -> String {
        let mut translated = Vec::new();
        translate_gff(
            conn,
            test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate GFF input");
        String::from_utf8(translated).expect("should write UTF-8 GFF output")
    }

    #[test]
    fn translates_coordinates_to_nodes() {
        let conn = get_connection();
        setup_test_data(&conn);

        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("./fixtures/complex.gff3");
        let collection = "test".to_string();

        let mut buffer = Vec::new();
        translate_gff(
            &conn,
            test_workspace(),
            &collection,
            "foo",
            None,
            BufReader::new(File::open(gff_path.clone()).expect("should open fixture gff")),
            &mut buffer,
        )
        .expect("should translate gff for sample foo");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        // This selected Path stores one interval for the first node; the feature bounds clip that
        // interval without splitting it at graph edges.
        assert_eq!(
            results,
            concat!(
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\texon\t1\t2\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\tgene\t3\t15\t.\t-\t.\tID=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\ttranscript\t3\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\texon\t3\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
            )
        );

        let mut buffer = Vec::new();
        translate_gff(
            &conn,
            test_workspace(),
            &collection,
            Sample::DEFAULT_NAME,
            None,
            BufReader::new(File::open(gff_path).expect("should open fixture gff")),
            &mut buffer,
        )
        .expect("should translate gff for reference sample");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\texon\t1\t2\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\tgene\t3\t15\t.\t-\t.\tID=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\ttranscript\t3\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\texon\t3\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
            )
        );
    }

    #[test]
    fn translates_gff_using_reference_aliases() {
        let conn = get_connection();
        setup_test_data(&conn);

        // Add a reference alias for the block group used in the fixture gff
        ReferenceAlias::create(
            &conn,
            "alternate contig",
            Some("m456".to_string()),
            Some("m123".to_string()),
            Some("m789".to_string()),
            Some("chr1".to_string()),
            None,
            None,
        )
        .expect("should create reference alias");

        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("./fixtures/complex-with-reference-alias.gff3");
        let collection = "test".to_string();

        let mut buffer = Vec::new();
        translate_gff(
            &conn,
            test_workspace(),
            &collection,
            Sample::DEFAULT_NAME,
            None,
            BufReader::new(File::open(gff_path.clone()).expect("should open fixture gff")),
            &mut buffer,
        )
        .expect("should translate gff for default sample");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\texon\t1\t2\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\tgene\t3\t15\t.\t-\t.\tID=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\ttranscript\t3\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\texon\t3\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
            )
        );
    }

    #[test]
    fn test_translates_sliced_graph_nodes_to_stored_sequence_coordinates() {
        let conn = get_connection();
        setup_test_data(&conn);

        for (name, traversal_strand, output_strand) in [
            ("sliced-forward", Strand::Forward, "+"),
            ("sliced-reverse", Strand::Reverse, "-"),
        ] {
            let (block_group, node_id) =
                create_sliced_test_block_group(&conn, name, traversal_strand);
            let input = format!(
                "{}\tsource\tgene\t2\t3\t.\t+\t.\tID=sliced-feature\n",
                block_group.name
            );
            assert_eq!(
                translate_gff_input(&conn, &input),
                format!(
                    "{node_id}\tsource\tgene\t5\t6\t.\t{output_strand}\t.\tID=sliced-feature\n"
                ),
                "GFF overlaps should map to the backing sequence span on {name}"
            );
        }
    }

    #[test]
    fn test_translates_off_center_sliced_graph_nodes_to_stored_coordinates() {
        let conn = get_connection();
        setup_test_data(&conn);

        // Stored sequence and path slice:
        // stored index: 0123456789
        // stored DNA:   AACCGGTTAA
        // stored slice:    [3,7) CGGT
        // Edge + path:     [0,4) CGGT; query [0,2) CG maps to stored [3,5).
        // Edge - path:     [0,4) ACCG; query [0,2) AC maps to stored [5,7).
        // With feature strand +, the emitted edge strands are + and - respectively.
        for (name, traversal_strand, start, end, output_strand) in [
            ("sliced-forward", Strand::Forward, 4, 5, "+"),
            ("sliced-reverse", Strand::Reverse, 6, 7, "-"),
        ] {
            let (block_group, node_id) =
                create_sliced_test_block_group(&conn, name, traversal_strand);
            let input = format!(
                "{}\tsource\tgene\t1\t2\t.\t+\t.\tID=sliced-feature\n",
                block_group.name
            );
            assert_eq!(
                translate_gff_input(&conn, &input),
                format!(
                    "{node_id}\tsource\tgene\t{start}\t{end}\t.\t{output_strand}\t.\tID=sliced-feature\n"
                ),
                "GFF query [0,2) should map to the off-center stored bases on {name}"
            );
        }
    }

    #[test]
    fn test_translates_gff_from_historical_path_after_current_path_deletion() {
        let conn = get_connection();
        setup_test_data(&conn);
        let block_group = Sample::get_block_groups(&conn, "test", Sample::DEFAULT_NAME, None)
            .into_iter()
            .find(|block_group| block_group.name == "m123")
            .expect("should find the test block group");
        let history_ref = commit_all(&conn, "store GFF translator path")
            .expect("should commit the GFF translator path")
            .to_string();

        // The path has two forward-traversed nodes; this feature selects the first node only.
        //   Node 1 stored index: 01234567890123456
        //   Node 1 stored DNA:   ATCGATCGATCGATCGA [path 0..17)
        //   Edge strand:        +
        //   GFF 6..8:           .....^^^......... raw TCG, feature strand -.
        //   Node 2 stored DNA:  TCGGGAACACACAGAGA [path 17..34), not selected.
        let input = "m123\tsource\tgene\t6\t8\t.\t-\t.\tID=history-feature\n";
        conn.execute(
            "DELETE FROM paths WHERE block_group_id = ?1",
            [block_group.id],
        )
        .expect("should delete the current path row");
        assert!(
            BlockGroup::get_current_path(&conn, &block_group.id, None).is_err(),
            "should not find a current path after deletion"
        );

        let mut output = Vec::new();
        translate_gff(
            &conn,
            test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            Some(&history_ref),
            BufReader::new(Cursor::new(input.as_bytes())),
            &mut output,
        )
        .expect("should translate GFF coordinates from the historical path");
        assert_eq!(
            String::from_utf8(output).expect("should encode GFF output as UTF-8"),
            "b59698a422128d20462c44537b2d23ef\tsource\tgene\t6\t8\t.\t-\t.\tID=history-feature\n",
            "should retain the exact stored-node span and reverse strand"
        );
        assert!(
            BlockGroup::get_current_path(&conn, &block_group.id, None).is_err(),
            "historical translation should leave the current path deleted"
        );
    }
}
