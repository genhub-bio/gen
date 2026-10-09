use std::{
    cmp::{max, min},
    collections::{HashMap, hash_map::Entry},
    io::{BufRead, Read, Write},
};

use gen_core::{HashId, Strand, Workspace, is_terminal};
use gen_graph::{GraphNode, project_path};
use gen_models::{
    block_group::BlockGroup,
    db::GraphConnection,
    errors::{BlockGroupError, PathError},
    reference_alias::{ReferenceAlias, ReferenceAliasError},
    sample::Sample,
};
use interavl::IntervalTree;
use noodles::{core::Position, gff};
use thiserror::Error;

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

    let bgs = Sample::get_block_groups(conn, collection, sample, history_ref);
    let sample_bgs: HashMap<String, &BlockGroup> = HashMap::from_iter(
        bgs.iter()
            .map(|bg| (bg.name.clone(), bg))
            .collect::<Vec<(String, &BlockGroup)>>(),
    );

    // Load all reference aliases, to accommodate alternate reference names in the GFF file
    let references = sample_bgs.keys().cloned().collect::<Vec<String>>();
    let references_by_alias =
        ReferenceAlias::get_references_by_alias(conn, references, history_ref)?;

    let mut paths: HashMap<HashId, IntervalTree<i64, (GraphNode, Strand, i64)>> = HashMap::new();

    for result in gff_reader.record_bufs() {
        let record = result?;
        let ref_name = record.reference_sequence_name().to_string();
        let ref_name = references_by_alias
            .get(&ref_name)
            .unwrap_or(&ref_name)
            .to_string();
        let start = record.start().get() as i64 - 1;
        let end = record.end().get() as i64;
        if let Some(bg) = sample_bgs.get(&ref_name) {
            let projection = match paths.entry(bg.id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let path = BlockGroup::get_current_path(conn, &bg.id, history_ref)?;
                    let graph = BlockGroup::get_graph(conn, workspace, &bg.id, history_ref)?;
                    let mut tree = IntervalTree::default();
                    let mut position: i64 = 0;
                    for (node, strand) in
                        project_path(&graph, &path.coordinate_blocks(conn, history_ref))
                    {
                        if !is_terminal(node.node_id) {
                            let end_position = position + node.length();
                            tree.insert(position..end_position, (node, strand, position));
                            position = end_position;
                        }
                    }
                    entry.insert(tree)
                }
            };
            let range = start..end;
            let mut overlaps = projection.iter_overlaps(&range).collect::<Vec<_>>();
            if record.strand() == gff::feature::record::Strand::Reverse {
                overlaps.reverse();
            }
            for (overlap, (node, overlap_strand, path_start)) in overlaps {
                let overlap_start = max(start, overlap.start) - path_start;
                let overlap_end = min(end, overlap.end) - path_start;
                // The source uses path coordinates, but emitted node records need local node coordinates.
                let (node_start, node_end) = if *overlap_strand == Strand::Reverse {
                    (
                        node.sequence_end - overlap_end,
                        node.sequence_end - overlap_start,
                    )
                } else {
                    (
                        node.sequence_start + overlap_start,
                        node.sequence_start + overlap_end,
                    )
                };
                let strand = match (record.strand(), overlap_strand) {
                    (gff::feature::record::Strand::Forward, Strand::Reverse) => {
                        gff::feature::record::Strand::Reverse
                    }
                    (gff::feature::record::Strand::Reverse, Strand::Reverse) => {
                        gff::feature::record::Strand::Forward
                    }
                    (strand, _) => strand,
                };

                let mut updated_record_builder =
                    gff::feature::RecordBuf::builder()
                        .set_reference_sequence_name(format!("{nid}", nid = node.node_id))
                        .set_source(record.source().to_string())
                        .set_type(record.ty().to_string())
                        .set_start(Position::try_from((node_start + 1) as usize).expect(
                            "Could not convert start ({overlap_start}) to usize for propagation",
                        ))
                        .set_end(Position::try_from(node_end as usize).expect(
                            "Could not convert end ({overlap_end}) to usize for propagation",
                        ))
                        .set_strand(strand)
                        .set_attributes(record.attributes().clone());
                if let Some(phase) = record.phase() {
                    updated_record_builder = updated_record_builder.set_phase(phase);
                }
                gff_writer.write_record(&updated_record_builder.build())?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader, path::PathBuf};

    use gen_models::{reference_alias::ReferenceAlias, sample::Sample};

    use super::translate_gff;
    use crate::test_helpers::{get_connection, setup_test_data, test_workspace};

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
        assert_eq!(
            results,
            concat!(
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t5\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t4\t4\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t5\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t5\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t4\t4\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\texon\t1\t2\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\tgene\t5\t15\t.\t-\t.\tID=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\tgene\t4\t4\t.\t-\t.\tID=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\tgene\t3\t3\t.\t-\t.\tID=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\ttranscript\t5\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\ttranscript\t3\t3\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\texon\t5\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\texon\t4\t4\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\texon\t3\t3\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
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
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\texon\t1\t2\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
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
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "6b460b727030cae3bae7cf389074d4ba\tHAVANA\texon\t1\t2\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\tgene\t3\t15\t.\t-\t.\tID=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\ttranscript\t3\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "b59698a422128d20462c44537b2d23ef\tENSEMBL\texon\t3\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
            )
        );
    }
}
