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

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use gen_core::{HashId, PathBlock, Strand};
    use gen_models::{
        block_group::{BlockGroup, PathChange},
        db::GraphConnection,
        node::Node,
        sample::Sample,
        sequence::Sequence,
    };

    use super::translate_bed;
    use crate::test_helpers::{get_connection, get_simple_sequence};

    fn apply_foo_variant_from_simple_vcf(conn: &GraphConnection) {
        Sample::get_or_create(conn, "foo");
        let _ = Sample::get_or_create_child(conn, "test", "foo", None);

        let sample_bg_id =
            BlockGroup::get_or_create_sample_block_group(conn, "test", "foo", "m123", None)
                .expect("should create child block group");
        let sample_path = BlockGroup::get_current_path(conn, &sample_bg_id);
        let tree = sample_path.intervaltree(conn);

        let alt_seq = "C";

        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(alt_seq)
            .save(conn);
        let node_id = Node::create(
            conn,
            &sequence.hash,
            &HashId::convert_str(&format!(
                "{path_id}:3-4->{sequence_hash}",
                path_id = sample_path.id,
                sequence_hash = sequence.hash,
            )),
        );
        let change = PathChange {
            block_group_id: sample_bg_id,
            path: sample_path,
            path_accession: None,
            start: 3,
            end: 4,
            block: PathBlock {
                id: 0,
                node_id,
                block_sequence: alt_seq.to_string(),
                sequence_start: 0,
                sequence_end: alt_seq.len() as i64,
                path_start: 3,
                path_end: 4,
                strand: Strand::Forward,
            },
            chromosome_index: 0,
            phased: 0,
            preserve_edge: false,
        };

        BlockGroup::insert_change(conn, &change, &tree)
            .expect("should apply variant change from simple.vcf");
    }

    #[test]
    fn translates_coordinates_to_nodes() {
        let conn = get_connection();
        let _ = get_simple_sequence(&conn);
        apply_foo_variant_from_simple_vcf(&conn);

        let bed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("./fixtures/simple.bed");
        let collection = "test".to_string();

        let mut buffer = Vec::new();
        translate_bed(
            &conn,
            &collection,
            Some("foo"),
            File::open(bed_path.clone()).expect("should open fixture bed"),
            &mut buffer,
        )
        .expect("should translate bed for sample foo");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t1\t3\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t3\t4\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t4\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\t17\t23\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
            )
        );

        let mut buffer = Vec::new();
        translate_bed(
            &conn,
            &collection,
            None,
            File::open(bed_path).expect("should open fixture bed"),
            &mut buffer,
        )
        .expect("should translate bed for default sample");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t1\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n",
                "0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
                "086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\t17\t23\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
            )
        );
    }
}
