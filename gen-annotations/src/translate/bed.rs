use std::{
    cmp::{max, min},
    collections::{HashMap, hash_map::Entry},
    io::{Read, Write},
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
use noodles::{
    bed,
    bed::feature::record_buf::{OtherFields, other_fields::Value},
    core::Position,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BedError {
    #[error("Couldn't translate BED entry: {0}")]
    Io(#[from] std::io::Error),
    #[error("Error loading reference aliases: {0}")]
    ReferenceAliasError(#[from] ReferenceAliasError),
    #[error("Error loading path blocks: {0}")]
    PathError(#[from] PathError),
    #[error("Error loading block group graph: {0}")]
    BlockGroupError(#[from] BlockGroupError),
}

pub fn translate_bed<R, W>(
    conn: &GraphConnection,
    workspace: &Workspace,
    collection: &str,
    sample: &str,
    history_ref: Option<&str>,
    reader: R,
    writer: &mut W,
) -> Result<(), BedError>
where
    R: Read,
    W: Write,
{
    let mut record = bed::Record::<6>::default();
    let mut bed_reader = bed::io::reader::Builder::<6>.build_from_reader(reader);
    let mut bed_writer = bed::io::Writer::<6, _>::new(writer);

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

    let mut paths: HashMap<HashId, IntervalTree<i64, (GraphNode, Strand)>> = HashMap::new();

    while bed_reader.read_record(&mut record)? != 0 {
        let ref_name = record.reference_sequence_name().to_string();
        let ref_name = references_by_alias
            .get(&ref_name)
            .unwrap_or(&ref_name)
            .to_string();
        // noodles converts to 1 index, keep it 0.
        let start = record.feature_start().unwrap().get() as i64 - 1;
        let end = record.feature_end().unwrap().unwrap().get() as i64;
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
                            tree.insert(position..end_position, (node, strand));
                            position = end_position;
                        }
                    }
                    entry.insert(tree)
                }
            };
            let range = start..end;
            let values: Vec<_> = record.other_fields().iter().map(Value::from).collect();
            let other_fields = OtherFields::from(values);
            let name = record
                .name()
                .map(|name| String::from_utf8_lossy(name.as_ref()).to_string());
            let score = record.score().ok();
            let strand = record.strand().ok().flatten();
            for (overlap, (node, _strand)) in projection.iter_overlaps(&range) {
                let overlap_start = max(start, overlap.start) as usize;
                let overlap_end = min(end, overlap.end) as usize;
                let mut out_record = bed::feature::RecordBuf::<6>::builder()
                    .set_reference_sequence_name(format!("{nid}", nid = node.node_id))
                    .set_feature_start(Position::try_from(overlap_start + 1).unwrap())
                    .set_feature_end(Position::try_from(overlap_end).unwrap())
                    .set_other_fields(other_fields.clone());
                if let Some(name) = &name {
                    out_record = out_record.set_name(name.clone());
                }
                if let Some(score) = score {
                    out_record = out_record.set_score(score);
                }
                if let Some(strand) = strand {
                    out_record = out_record.set_strand(strand);
                }
                let out_record = out_record.build();
                bed_writer.write_feature_record(&out_record)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use gen_models::{reference_alias::ReferenceAlias, sample::Sample};

    use super::translate_bed;
    use crate::test_helpers::{get_connection, setup_test_data, test_workspace};

    #[test]
    fn translates_coordinates_to_nodes() {
        let conn = get_connection();
        setup_test_data(&conn);

        let bed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("./fixtures/simple.bed");
        let collection = "test".to_string();

        let mut buffer = Vec::new();
        translate_bed(
            &conn,
            test_workspace(),
            &collection,
            "foo",
            None,
            File::open(bed_path.clone()).expect("should open fixture bed"),
            &mut buffer,
        )
        .expect("should translate bed for sample foo");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "b59698a422128d20462c44537b2d23ef\t1\t3\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "b59698a422128d20462c44537b2d23ef\t3\t4\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "b59698a422128d20462c44537b2d23ef\t4\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "b59698a422128d20462c44537b2d23ef\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n",
                "b59698a422128d20462c44537b2d23ef\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n",
                "b59698a422128d20462c44537b2d23ef\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
                "6b460b727030cae3bae7cf389074d4ba\t17\t23\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
            )
        );

        let mut buffer = Vec::new();
        translate_bed(
            &conn,
            test_workspace(),
            &collection,
            Sample::DEFAULT_NAME,
            None,
            File::open(bed_path).expect("should open fixture bed"),
            &mut buffer,
        )
        .expect("should translate bed for reference sample");
        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "b59698a422128d20462c44537b2d23ef\t1\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "b59698a422128d20462c44537b2d23ef\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n",
                "b59698a422128d20462c44537b2d23ef\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n",
                "b59698a422128d20462c44537b2d23ef\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
                "6b460b727030cae3bae7cf389074d4ba\t17\t23\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
            )
        );
    }

    #[test]
    fn translates_bed_using_reference_aliases() {
        let conn = get_connection();
        setup_test_data(&conn);

        let bed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("./fixtures/simple-with-reference-alias.bed");
        let collection = "test".to_string();

        // Add a reference alias for the block group used in the fixture bed
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

        let mut buffer = Vec::new();
        translate_bed(
            &conn,
            test_workspace(),
            &collection,
            Sample::DEFAULT_NAME,
            None,
            File::open(bed_path).expect("should open fixture bed"),
            &mut buffer,
        )
        .expect("should translate bed for reference sample");

        let results = String::from_utf8(buffer).expect("translated output should be valid UTF-8");
        assert_eq!(
            results,
            concat!(
                "b59698a422128d20462c44537b2d23ef\t1\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "b59698a422128d20462c44537b2d23ef\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n",
                "b59698a422128d20462c44537b2d23ef\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n",
                "b59698a422128d20462c44537b2d23ef\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
                "6b460b727030cae3bae7cf389074d4ba\t17\t23\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
            )
        );
    }
}
