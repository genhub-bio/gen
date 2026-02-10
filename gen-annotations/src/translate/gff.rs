use std::{
    cmp::{max, min},
    collections::HashMap,
    io::{BufRead, Error, Read, Write},
};

use gen_core::{HashId, Strand, is_terminal};
use gen_graph::{GraphNode, connect_all_boundary_edges, project_path};
use gen_models::{block_group::BlockGroup, db::GraphConnection, sample::Sample};
use interavl::IntervalTree;
use noodles::{core::Position, gff};

pub fn translate_gff<'a, R, W>(
    conn: &GraphConnection,
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
    let mut paths: HashMap<HashId, IntervalTree<i64, (GraphNode, Strand)>> = HashMap::new();

    for result in gff_reader.record_bufs() {
        let record = result?;
        let ref_name = record.reference_sequence_name().to_string();
        let start = record.start().get() as i64;
        let end = record.end().get() as i64;
        if let Some(bg) = sample_bgs.get(&ref_name) {
            let projection = paths.entry(bg.id).or_insert_with(|| {
                let path = BlockGroup::get_current_path(conn, &bg.id);
                let mut graph = BlockGroup::get_graph(conn, &bg.id);
                connect_all_boundary_edges(&mut graph);
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
