use std::{
    collections::HashMap,
    io::{Read, Write},
};

use gen_core::Workspace;
use gen_models::{
    block_group::BlockGroup,
    db::GraphConnection,
    errors::{BlockGroupError, PathError},
    reference_alias::{ReferenceAlias, ReferenceAliasError},
    region::{GenRegionError, ResolvedGenRegion},
    sample::Sample,
};
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
    #[error("Error resolving path region: {0}")]
    RegionError(#[from] GenRegionError),
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
    let sample_block_groups = Sample::get_block_groups(conn, collection, sample, history_ref)
        .into_iter()
        .map(|block_group| (block_group.name, block_group.id))
        .collect::<HashMap<_, _>>();
    let references = sample_block_groups.keys().cloned().collect();
    let references_by_alias =
        ReferenceAlias::get_references_by_alias(conn, references, history_ref)?;
    let mut regions_by_reference = HashMap::<String, ResolvedGenRegion>::new();

    while bed_reader.read_record(&mut record)? != 0 {
        let ref_name = record.reference_sequence_name().to_string();
        // noodles converts to 1 index, keep it 0.
        let start = record.feature_start().unwrap().get() as i64 - 1;
        let end = record.feature_end().unwrap().unwrap().get() as i64;
        let values: Vec<_> = record.other_fields().iter().map(Value::from).collect();
        let other_fields = OtherFields::from(values);
        let name = record
            .name()
            .map(|name| String::from_utf8_lossy(name.as_ref()).to_string());
        let score = record.score().ok();
        let feature_strand = record.strand().ok().flatten();
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
            let mut out_record = bed::feature::RecordBuf::<6>::builder()
                .set_reference_sequence_name(format!("{nid}", nid = slice.block.node_id))
                .set_feature_start(
                    Position::try_from((slice.block.sequence_start as usize + slice.start) + 1)
                        .expect("should convert the stored-node start to a BED position"),
                )
                .set_feature_end(
                    Position::try_from(slice.block.sequence_start as usize + slice.end)
                        .expect("should convert the stored-node end to a BED position"),
                )
                .set_other_fields(other_fields.clone());
            if let Some(name) = &name {
                out_record = out_record.set_name(name.clone());
            }
            if let Some(score) = score {
                out_record = out_record.set_score(score);
            }
            if let Some(output_strand) = feature_strand {
                out_record = out_record.set_strand(output_strand);
            }
            let out_record = out_record.build();
            bed_writer.write_feature_record(&out_record)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor, path::PathBuf};

    use gen_core::Strand;
    use gen_models::{
        block_group::BlockGroup, history::dolt::commit_all, reference_alias::ReferenceAlias,
        sample::Sample,
    };

    use super::translate_bed;
    use crate::test_helpers::{
        create_sliced_test_block_group, get_connection, setup_test_data, test_workspace,
    };

    fn translate_bed_input(conn: &gen_models::db::GraphConnection, input: &str) -> String {
        let mut translated = Vec::new();
        translate_bed(
            conn,
            test_workspace(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            Cursor::new(input.as_bytes()),
            &mut translated,
        )
        .expect("should translate BED input");
        String::from_utf8(translated).expect("should write UTF-8 BED output")
    }

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
        // This selected Path stores one interval for the first node; the feature bounds clip that
        // interval without splitting it at graph edges.
        assert_eq!(
            results,
            concat!(
                "b59698a422128d20462c44537b2d23ef\t1\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n",
                "b59698a422128d20462c44537b2d23ef\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n",
                "b59698a422128d20462c44537b2d23ef\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n",
                "b59698a422128d20462c44537b2d23ef\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
                "6b460b727030cae3bae7cf389074d4ba\t0\t6\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
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
                "6b460b727030cae3bae7cf389074d4ba\t0\t6\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
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
                "6b460b727030cae3bae7cf389074d4ba\t0\t6\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n",
            )
        );
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
        // Feature strand stays + while edge direction determines the mapped stored coordinates.
        for (name, traversal_strand, start, end) in [
            ("sliced-forward", Strand::Forward, 3, 5),
            ("sliced-reverse", Strand::Reverse, 5, 7),
        ] {
            let (block_group, node_id) =
                create_sliced_test_block_group(&conn, name, traversal_strand);
            let input = format!("{}\t0\t2\tsliced-feature\t0\t+\n", block_group.name);
            assert_eq!(
                translate_bed_input(&conn, &input),
                format!("{node_id}\t{start}\t{end}\tsliced-feature\t0\t+\n"),
                "BED query [0,2) should map to the off-center stored bases on {name}"
            );
        }
    }
}
