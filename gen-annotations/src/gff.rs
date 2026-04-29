use std::{collections::HashMap, fs::File, io, io::BufReader};

use gen_models::{
    block_group::BlockGroup,
    db::GraphConnection,
    path::{Annotation, Path},
    sample::Sample,
};
use noodles::{core::Position, gff};

pub fn gff_attribute_value_to_string(
    attrs: &gff::feature::record_buf::Attributes,
    key: &str,
) -> Option<String> {
    let key_bytes = key.as_bytes();
    attrs.as_ref().iter().find_map(|(tag, value)| {
        let tag_bytes: &[u8] = tag.as_ref();
        if !tag_bytes.eq_ignore_ascii_case(key_bytes) {
            return None;
        }
        if let Some(value) = value.as_string() {
            Some(String::from_utf8_lossy(value.as_ref()).to_string())
        } else {
            value
                .iter()
                .next()
                .map(|item| String::from_utf8_lossy(item.as_ref()).to_string())
        }
    })
}

pub fn propagate_gff(
    conn: &GraphConnection,
    collection_name: &str,
    from_sample_name: &str,
    to_sample_name: &str,
    gff_input_filename: &str,
    gff_output_filename: &str,
) -> io::Result<()> {
    let mut reader = File::open(gff_input_filename)
        .map(BufReader::new)
        .map(gff::io::Reader::new)?;

    let output_file = File::create(gff_output_filename).unwrap();
    let mut writer = gff::io::Writer::new(output_file);

    let source_block_groups = Sample::get_block_groups(conn, collection_name, from_sample_name);
    let target_block_groups = Sample::get_block_groups(conn, collection_name, to_sample_name);
    let source_paths_by_bg_name = source_block_groups
        .iter()
        .map(|bg| (bg.name.clone(), BlockGroup::get_current_path(conn, &bg.id)))
        .collect::<HashMap<String, Path>>();
    let target_paths_by_bg_name = target_block_groups
        .iter()
        .map(|bg| (bg.name.clone(), BlockGroup::get_current_path(conn, &bg.id)))
        .collect::<HashMap<String, Path>>();

    let mut path_mappings_by_bg_name = HashMap::new();
    for (name, target_path) in &target_paths_by_bg_name {
        let source_path = source_paths_by_bg_name.get(name).unwrap();
        let mapping = source_path
            .get_mapping_tree(conn, target_path)
            .map_err(io::Error::other)?;
        path_mappings_by_bg_name.insert(name, mapping);
    }

    let sequence_lengths_by_path_name = target_paths_by_bg_name
        .iter()
        .map(|(name, path)| {
            path.sequence(conn)
                .map(|sequence| (name.clone(), sequence.len() as i64))
        })
        .collect::<Result<HashMap<String, i64>, gen_models::errors::SequenceError>>()
        .map_err(io::Error::other)?;

    for result in reader.record_bufs() {
        let record = result?;
        let path_name = record.reference_sequence_name().to_string();
        let annotation = Annotation {
            name: "".to_string(),
            start: record.start().get() as i64,
            end: record.end().get() as i64,
        };
        let mapping_tree = path_mappings_by_bg_name.get(&path_name).unwrap();
        let sequence_length = sequence_lengths_by_path_name.get(&path_name).unwrap();
        let propagated_annotation =
            Path::propagate_annotation(annotation, mapping_tree, *sequence_length).unwrap();

        let score = record.score();
        let phase = record.phase();
        let mut updated_record_builder = gff::feature::RecordBuf::builder()
            .set_reference_sequence_name(path_name)
            .set_source(record.source().to_string())
            .set_type(record.ty().to_string())
            .set_start(
                Position::new(propagated_annotation.start.try_into().unwrap())
                    .expect("Could not convert start ({start}) to usize for propagation"),
            )
            .set_end(
                Position::new(propagated_annotation.end.try_into().unwrap())
                    .expect("Could not convert end ({end}) to usize for propagation"),
            )
            .set_strand(record.strand())
            .set_attributes(record.attributes().clone());

        if let Some(score) = score {
            updated_record_builder = updated_record_builder.set_score(score);
        }
        if let Some(phase) = phase {
            updated_record_builder = updated_record_builder.set_phase(phase);
        }

        writer.write_record(&updated_record_builder.build())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader, path::PathBuf};

    use gen_core::{
        HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID, PathBlock, Strand,
    };
    use gen_models::{
        block_group::{BlockGroup, NewBlockGroup, PathChange},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        node::Node,
        path::Path,
        sample::Sample,
        sequence::Sequence,
        traits::Query,
    };
    use noodles::gff;
    use tempfile::tempdir;

    use super::propagate_gff;
    use crate::test_helpers::get_connection;

    fn create_block_group(conn: &GraphConnection) {
        let collection = Collection::create(conn, "test");
        Sample::get_or_create(conn, Sample::DEFAULT_NAME);
        let sequence = "ATCGATCGATCGATCGATCGGGAACACACAGAGA";
        let reference_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn);
        let node_id = Node::create(
            conn,
            &reference_sequence.hash,
            &HashId::convert_str(&format!(
                "{collection}.m123:{hash}",
                collection = collection.name,
                hash = reference_sequence.hash
            )),
        );
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &collection.name,
                sample_name: Sample::DEFAULT_NAME,
                name: "m123",
                parent_block_group_id: None,
                is_default: false,
            },
        );

        let edge_into = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        );
        let edge_out_of = Edge::create(
            conn,
            node_id,
            reference_sequence.length,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        );

        let new_block_group_edges = vec![
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge_into.id,
                chromosome_index: 0,
                phased: 0,
            },
            BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: edge_out_of.id,
                chromosome_index: 0,
                phased: 0,
            },
        ];

        BlockGroupEdge::bulk_create(conn, &new_block_group_edges);
        Path::create(
            conn,
            "m123",
            &block_group.id,
            &[edge_into.id, edge_out_of.id],
        );
    }

    fn apply_child_sample_update_from_aa_fasta(conn: &GraphConnection) {
        Sample::get_or_create(conn, "child sample");
        let _ = Sample::get_or_create_child(
            conn,
            "test",
            "child sample",
            vec![Sample::DEFAULT_NAME.to_string()],
        );

        let sample_bg_id = BlockGroup::get_or_create_sample_block_groups(
            conn,
            "test",
            "child sample",
            "m123",
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .expect("should create child block group")[0]
            .id;
        let sample_path = BlockGroup::get_current_path(conn, &sample_bg_id);
        let tree = sample_path.intervaltree(conn).unwrap();
        let replacement_sequence = "AA";

        let replacement = Sequence::new()
            .sequence_type("DNA")
            .sequence(replacement_sequence)
            .save(conn);
        let node_id = Node::create(
            conn,
            &replacement.hash,
            &HashId::convert_str(&format!(
                "{path_id}:15-25->{sequence_hash}",
                path_id = sample_path.id,
                sequence_hash = replacement.hash,
            )),
        );
        let change = PathChange {
            block_group_id: sample_bg_id,
            path: sample_path.clone(),
            path_accession: None,
            start: 15,
            end: 25,
            block: PathBlock {
                node_id,
                block_sequence: replacement_sequence.to_string(),
                sequence_start: 0,
                sequence_end: replacement_sequence.len() as i64,
                path_start: 15,
                path_end: 25,
                strand: Strand::Forward,
            },
            chromosome_index: NO_CHROMOSOME_INDEX,
            phased: 0,
            preserve_edge: true,
        };

        BlockGroup::insert_change(conn, &change, &tree)
            .expect("should apply AA update to child sample");

        let edge_to_insert = Edge::query(
            conn,
            "select * from edges where target_node_id = ?1",
            rusqlite::params![node_id],
        )[0]
        .clone();
        let edge_from_insert = Edge::query(
            conn,
            "select * from edges where source_node_id = ?1",
            rusqlite::params![node_id],
        )[0]
        .clone();
        let _ = sample_path
            .new_path_with(conn, 15, 25, &edge_to_insert, &edge_from_insert)
            .unwrap();
    }

    #[test]
    fn simple_propagate() {
        let conn = get_connection();
        create_block_group(&conn);
        apply_child_sample_update_from_aa_fasta(&conn);

        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");
        let temp_dir = tempdir().expect("should create temp directory");
        let output_path = temp_dir.path().join("output.gff");

        propagate_gff(
            &conn,
            "test",
            Sample::DEFAULT_NAME,
            "child sample",
            gff_path.to_str().expect("should convert gff path to UTF-8"),
            output_path
                .to_str()
                .expect("should convert output path to UTF-8"),
        )
        .expect("should propagate gff to child sample");

        let mut reader = File::open(output_path)
            .map(BufReader::new)
            .map(gff::io::Reader::new)
            .expect("should read output file");

        for (index, result) in reader.record_bufs().enumerate() {
            let record = result.expect("should parse output gff record");
            assert_eq!(record.reference_sequence_name(), "m123");
            if index == 0 {
                assert_eq!(record.source(), "gen-test");
                assert_eq!(record.ty(), "Region");
                assert_eq!(record.start().get(), 1);
                assert_eq!(record.end().get(), 26);
            } else {
                assert_eq!(record.source(), "gen-test");
                assert_eq!(record.ty(), "Gene");
                assert_eq!(record.start().get(), 5);
                assert_eq!(record.end().get(), 15);
            }
        }
    }
}
