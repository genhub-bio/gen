use crate::graph::{connect_all_boundary_edges, project_path, GraphNode};
use crate::models::block_group::BlockGroup;
use crate::models::node::Node;
use crate::models::sample::Sample;
use crate::models::strand::Strand;
use interavl::IntervalTree;
use noodles::core::Position;
use noodles::gff;
use rusqlite::Connection;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::io::{BufRead, Error, Read, Write};

pub fn translate_gff<'a, R, W>(
    conn: &Connection,
    collection: &str,
    sample: impl Into<Option<&'a str>>,
    reader: R,
    writer: &mut W,
) -> Result<(), Error>
where
    R: Read + BufRead,
    W: Write,
{
    let sample = sample.into();
    let mut gff_reader = gff::io::Reader::new(reader);
    let mut gff_writer = gff::io::Writer::new(writer);

    let bgs = Sample::get_block_groups(conn, collection, sample);
    let sample_bgs: HashMap<String, &BlockGroup> = HashMap::from_iter(
        bgs.iter()
            .map(|bg| (bg.name.clone(), bg))
            .collect::<Vec<(String, &BlockGroup)>>(),
    );
    let mut paths: HashMap<i64, IntervalTree<i64, (GraphNode, Strand)>> = HashMap::new();

    for result in gff_reader.record_bufs() {
        let record = result?;
        let ref_name = record.reference_sequence_name();
        let start = record.start().get() as i64;
        let end = record.end().get() as i64;
        if let Some(bg) = sample_bgs.get(ref_name) {
            let projection = paths.entry(bg.id).or_insert_with(|| {
                let path = BlockGroup::get_current_path(conn, bg.id);
                let mut graph = BlockGroup::get_graph(conn, bg.id);
                connect_all_boundary_edges(&mut graph);
                let mut tree = IntervalTree::default();
                let mut position: i64 = 0;
                for (node, strand) in project_path(&graph, &path.blocks(conn)) {
                    if !Node::is_terminal(node.node_id) {
                        // GFF indexing is one based, inclusive, so we add 1 to the start.
                        // Take a sequence that is 1-4 in our coordinates, this converts to:
                        // 0123456
                        // ATCGATC
                        // 1234567
                        // 1-4 in our zero-based half open interval would be 2-4 in GFF coordinates
                        let end_position = position + node.length();
                        tree.insert(position + 1..end_position, (node, strand));
                        position = end_position;
                    }
                }
                tree
            });
            let range = start..end;
            for (overlap, (node, _overlap_strand)) in projection.iter_overlaps(&range) {
                let overlap_start = max(start, overlap.start) as usize;
                let overlap_end = min(end, overlap.end) as usize;

                let updated_record_builder =
                    gff::RecordBuf::builder()
                        .set_reference_sequence_name(format!("{nid}", nid = node.node_id))
                        .set_source(record.source().to_string())
                        .set_type(record.ty().to_string())
                        .set_start(Position::try_from(overlap_start).expect(
                            "Could not convert start ({overlap_start}) to usize for propagation",
                        ))
                        .set_end(Position::try_from(overlap_end).expect(
                            "Could not convert end ({overlap_end}) to usize for propagation",
                        ))
                        .set_strand(record.strand())
                        .set_attributes(record.attributes().clone());
                gff_writer.write_record(&updated_record_builder.build())?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::metadata;
    use crate::models::operations::setup_db;
    use crate::test_helpers::{get_connection, get_operation_connection, setup_gen_dir};
    use crate::translate::test_helpers::get_simple_sequence;
    use crate::updates::vcf::update_with_vcf;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;

    #[test]
    fn translates_coordinates_to_nodes() {
        setup_gen_dir();
        let conn = &get_connection(None);
        let db_uuid = metadata::get_db_uuid(conn);
        let op_conn = &get_operation_connection(None);
        setup_db(op_conn, &db_uuid);

        let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gffs/simple.gff3");
        let collection = "test".to_string();

        get_simple_sequence(conn);
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
        let mut buffer = Vec::new();
        // "foo" is a sample from simple.vcf
        translate_gff(
            conn,
            &collection,
            "foo",
            BufReader::new(File::open(gff_path.clone()).unwrap()),
            &mut buffer,
        )
        .unwrap();
        let results = String::from_utf8(buffer).unwrap();
        assert_eq!(
            results,
            "\
            3\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n\
            3\tHAVANA\tgene\t4\t4\t.\t-\t.\tID=ENSG00000294541.1\n\
            3\tHAVANA\tgene\t5\t17\t.\t-\t.\tID=ENSG00000294541.1\n\
            4\tHAVANA\tgene\t18\t20\t.\t-\t.\tID=ENSG00000294541.1\n\
            3\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
            3\tHAVANA\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
            3\tHAVANA\ttranscript\t5\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
            4\tHAVANA\ttranscript\t18\t20\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
            3\tHAVANA\texon\t5\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n\
            3\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n\
            3\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
            4\tHAVANA\texon\t18\t19\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
            3\tENSEMBL\tgene\t4\t4\t.\t-\t.\tID=ENSG00000277248.1\n\
            3\tENSEMBL\tgene\t5\t15\t.\t-\t.\tID=ENSG00000277248.1\n\
            3\tENSEMBL\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n\
            3\tENSEMBL\ttranscript\t5\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n\
            3\tENSEMBL\texon\t4\t4\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n\
            3\tENSEMBL\texon\t5\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n\
            "
        );

        // The None sample has no variants, so should be a simple mapping with the single split
        let mut buffer = Vec::new();
        translate_gff(
            conn,
            &collection,
            None,
            BufReader::new(File::open(gff_path.clone()).unwrap()),
            &mut buffer,
        )
        .unwrap();
        let results = String::from_utf8(buffer).unwrap();
        assert_eq!(
            results,
            "\
            3\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n\
            4\tHAVANA\tgene\t18\t20\t.\t-\t.\tID=ENSG00000294541.1\n\
            3\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
            4\tHAVANA\ttranscript\t18\t20\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
            3\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n\
            3\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n\
            3\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
            4\tHAVANA\texon\t18\t19\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
            3\tENSEMBL\tgene\t3\t15\t.\t-\t.\tID=ENSG00000277248.1\n\
            3\tENSEMBL\ttranscript\t3\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n\
            3\tENSEMBL\texon\t3\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n\
            "
        );
    }
}
