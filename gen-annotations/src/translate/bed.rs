use std::{
    cmp::{max, min},
    collections::HashMap,
    io::{Error, Read, Write},
};

use gen_core::{HashId, Strand, is_terminal};
use gen_graph::{GraphNode, connect_all_boundary_edges, project_path};
use gen_models::{block_group::BlockGroup, db::GraphConnection, sample::Sample};
use interavl::IntervalTree;
use noodles::{
    bed,
    bed::feature::record_buf::{OtherFields, other_fields::Value},
    core::Position,
};

pub fn translate_bed<'a, R, W>(
    conn: &GraphConnection,
    collection: &str,
    sample: impl Into<Option<&'a str>>,
    reader: R,
    writer: &mut W,
) -> Result<(), Error>
where
    R: Read,
    W: Write,
{
    let sample = sample.into();
    let mut record = bed::Record::default();
    let mut bed_reader = bed::io::reader::Builder::<3>.build_from_reader(reader);
    let mut bed_writer = bed::io::Writer::<3, _>::new(writer);

    let bgs = Sample::get_block_groups(conn, collection, sample);
    let sample_bgs: HashMap<String, &BlockGroup> = HashMap::from_iter(
        bgs.iter()
            .map(|bg| (bg.name.clone(), bg))
            .collect::<Vec<(String, &BlockGroup)>>(),
    );
    let mut paths: HashMap<HashId, IntervalTree<i64, (GraphNode, Strand)>> = HashMap::new();

    while bed_reader.read_record(&mut record)? != 0 {
        let ref_name = record.reference_sequence_name().to_string();
        // noodles converts to 1 index, keep it 0.
        let start = record.feature_start().unwrap().get() as i64 - 1;
        let end = record.feature_end().unwrap().unwrap().get() as i64;
        if let Some(bg) = sample_bgs.get(&ref_name) {
            let projection = paths.entry(bg.id).or_insert_with(|| {
                let path = BlockGroup::get_current_path(conn, &bg.id);
                let mut graph = BlockGroup::get_graph(conn, &bg.id);
                connect_all_boundary_edges(&mut graph);
                let mut tree = IntervalTree::default();
                let mut position: i64 = 0;
                for (node, strand) in project_path(&graph, &path.blocks(conn)) {
                    if !is_terminal(node.node_id) {
                        let end_position = position + node.length();
                        tree.insert(position..end_position, (node, strand));
                        position = end_position;
                    }
                }
                tree
            });
            let range = start..end;
            let values: Vec<_> = record.other_fields().iter().map(Value::from).collect();
            let other_fields = OtherFields::from(values);
            for (overlap, (node, _strand)) in projection.iter_overlaps(&range) {
                let overlap_start = max(start, overlap.start) as usize;
                let overlap_end = min(end, overlap.end) as usize;
                let out_record = bed::feature::RecordBuf::<3>::builder()
                    .set_reference_sequence_name(format!("{nid}", nid = node.node_id))
                    .set_feature_start(Position::try_from(overlap_start + 1).unwrap())
                    .set_feature_end(Position::try_from(overlap_end).unwrap())
                    .set_other_fields(other_fields.clone())
                    .build();
                bed_writer.write_feature_record(&out_record)?;
            }
        }
    }
    Ok(())
}
