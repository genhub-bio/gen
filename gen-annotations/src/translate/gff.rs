use std::{
    cmp::{max, min},
    collections::HashMap,
    io::{BufRead, Error, Read, Write},
};

use gen_core::{HashId, Strand, is_terminal};
use gen_graph::{GraphNode, project_path};
use gen_models::{block_group::BlockGroup, db::GraphConnection, sample::Sample};
use interavl::IntervalTree;
use noodles::{core::Position, gff};

pub fn translate_gff<R, W>(
    conn: &GraphConnection,
    collection: &str,
    sample: &str,
    reader: R,
    writer: &mut W,
) -> Result<(), Error>
where
    R: Read + BufRead,
    W: Write,
{
    let mut gff_reader = gff::io::Reader::new(reader);
    let mut gff_writer = gff::io::Writer::new(writer);

    let bgs = Sample::get_block_groups(conn, collection, sample);
    let sample_bgs: HashMap<String, &BlockGroup> = HashMap::from_iter(
        bgs.iter()
            .map(|bg| (bg.name.clone(), bg))
            .collect::<Vec<(String, &BlockGroup)>>(),
    );
    let mut paths: HashMap<HashId, IntervalTree<i64, (GraphNode, Strand)>> = HashMap::new();

    for result in gff_reader.record_bufs() {
        let record = result?;
        let ref_name = record.reference_sequence_name().to_string();
        let start = record.start().get() as i64;
        let end = record.end().get() as i64;
        if let Some(bg) = sample_bgs.get(&ref_name) {
            let projection = paths.entry(bg.id).or_insert_with(|| {
                let path = BlockGroup::get_current_path(conn, &bg.id);
                let graph = BlockGroup::get_graph(conn, &bg.id);
                let mut tree = IntervalTree::default();
                let mut position: i64 = 0;
                for (node, strand) in project_path(&graph, &path.blocks(conn)) {
                    if !is_terminal(node.node_id) {
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

                let mut updated_record_builder =
                    gff::feature::RecordBuf::builder()
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

    use super::translate_gff;
    use crate::test_helpers::{get_connection, setup_test_data};

    #[test]
    fn translates_coordinates_to_nodes() {
        let conn = get_connection();
        setup_test_data(&conn);

        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("./fixtures/complex.gff3");
        let collection = "test".to_string();

        let mut buffer = Vec::new();
        translate_gff(
            &conn,
            &collection,
            "foo",
            BufReader::new(File::open(gff_path.clone()).expect("should open fixture gff")),
            &mut buffer,
        )
        .expect("should translate gff for sample foo");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t4\t4\t.\t-\t.\tID=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t5\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\tgene\t18\t20\t.\t-\t.\tID=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t5\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\ttranscript\t18\t20\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t5\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\texon\t18\t19\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\tgene\t4\t4\t.\t-\t.\tID=ENSG00000277248.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\tgene\t5\t15\t.\t-\t.\tID=ENSG00000277248.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\ttranscript\t5\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\texon\t4\t4\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\texon\t5\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
            )
        );

        let mut buffer = Vec::new();
        translate_gff(
            &conn,
            &collection,
            Sample::DEFAULT_NAME,
            BufReader::new(File::open(gff_path).expect("should open fixture gff")),
            &mut buffer,
        )
        .expect("should translate gff for reference sample");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\tgene\t18\t20\t.\t-\t.\tID=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\ttranscript\t18\t20\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\texon\t18\t19\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\tgene\t3\t15\t.\t-\t.\tID=ENSG00000277248.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\ttranscript\t3\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\texon\t3\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n",
            )
        );
    }
}
