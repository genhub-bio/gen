use std::{collections::HashMap, fs::File, io, io::BufReader};

use gen_core::HashId;
use gen_models::{
    block_group::{BlockGroup, BlockGroupError},
    db::GraphConnection,
    errors::PathError,
    path::{Annotation, Path},
    sample::Sample,
};
use noodles::{core::Position, gff};

#[derive(Debug, thiserror::Error)]
pub enum PropagateGffError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Path error: {0}")]
    Path(#[from] PathError),
    #[error("Block group error: {0}")]
    BlockGroup(#[from] BlockGroupError),
}

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
) -> Result<(), PropagateGffError> {
    let mut reader = File::open(gff_input_filename)
        .map(BufReader::new)
        .map(gff::io::Reader::new)?;

    let output_file = File::create(gff_output_filename).unwrap();
    let mut writer = gff::io::Writer::new(output_file);

    let source_block_groups_by_name: HashMap<String, HashId> =
        Sample::get_block_groups(conn, collection_name, from_sample_name)
            .into_iter()
            .map(|bg| (bg.name, bg.id))
            .collect();
    let target_block_groups_by_name: HashMap<String, HashId> =
        Sample::get_block_groups(conn, collection_name, to_sample_name)
            .into_iter()
            .map(|bg| (bg.name, bg.id))
            .collect();

    // Only the block groups the GFF actually references need a current path resolved.
    // A sample can also contain block groups with no path at all (e.g. a protein
    // sequence graph created by `derive translation`, which lives in the same
    // sample as the DNA graph it was translated from) -- resolving paths for every
    // block group up front would fail on those even though they're never used here.
    let mut path_mappings_by_name = HashMap::new();
    let mut sequence_lengths_by_name: HashMap<String, i64> = HashMap::new();

    for result in reader.record_bufs() {
        let record = result?;
        let path_name = record.reference_sequence_name().to_string();
        let annotation = Annotation {
            name: "".to_string(),
            start: record.start().get() as i64,
            end: record.end().get() as i64,
        };

        if !path_mappings_by_name.contains_key(&path_name) {
            let source_block_group_id = source_block_groups_by_name.get(&path_name).unwrap();
            let target_block_group_id = target_block_groups_by_name.get(&path_name).unwrap();
            let source_path = BlockGroup::get_current_path(conn, source_block_group_id)?;
            let target_path = BlockGroup::get_current_path(conn, target_block_group_id)?;
            let mapping = source_path.get_mapping_tree(conn, &target_path)?;
            sequence_lengths_by_name
                .insert(path_name.clone(), target_path.sequence(conn)?.len() as i64);
            path_mappings_by_name.insert(path_name.clone(), mapping);
        }
        let mapping_tree = path_mappings_by_name.get(&path_name).unwrap();
        let sequence_length = sequence_lengths_by_name.get(&path_name).unwrap();
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
        block_group::{BlockGroup, BlockGroupChange, NewBlockGroup},
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        db::GraphConnection,
        edge::Edge,
        node::Node,
        path::Path,
        region::ResolvedGenRegion,
        sample::Sample,
        sequence::Sequence,
        traits::Query,
    };
    use noodles::gff;
    use tempfile::tempdir;

    use super::propagate_gff;
    use crate::test_helpers::get_connection;

    fn create_block_group(conn: &GraphConnection) {
        let collection = Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: Sample::DEFAULT_NAME,
                ..Default::default()
            },
        )
        .unwrap();
        let sequence = "ATCGATCGATCGATCGATCGGGAACACACAGAGA";
        let reference_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence(sequence)
            .save(conn)
            .unwrap();
        let node_id = Node::create(
            conn,
            &reference_sequence.hash,
            &HashId::convert_str(&format!(
                "{collection}.m123:{hash}",
                collection = collection.name,
                hash = reference_sequence.hash
            )),
        )
        .unwrap();
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: &collection.name,
                sample_name: Sample::DEFAULT_NAME,
                name: "m123",
                parent_block_group_id: None,
                is_default: false,
            },
        )
        .unwrap();

        let edge_into = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge_out_of = Edge::create(
            conn,
            node_id,
            reference_sequence.length,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

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
        )
        .unwrap();
    }

    fn apply_child_sample_update_from_aa_fasta(conn: &GraphConnection) {
        Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "child sample",
                ..Default::default()
            },
        )
        .unwrap();
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
        let sample_path = BlockGroup::get_current_path(conn, &sample_bg_id).unwrap();
        let replacement_sequence = "AA";

        let replacement = Sequence::new()
            .sequence_type("DNA")
            .sequence(replacement_sequence)
            .save(conn)
            .unwrap();
        let node_id = Node::create(
            conn,
            &replacement.hash,
            &HashId::convert_str(&format!(
                "{path_id}:15-25->{sequence_hash}",
                path_id = sample_path.id,
                sequence_hash = replacement.hash,
            )),
        )
        .unwrap();
        let region =
            ResolvedGenRegion::from_path(conn, sample_bg_id, &sample_path, 15, 25).unwrap();
        let change = BlockGroupChange {
            region,
            path_accession: None,
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

        BlockGroup::insert_change(conn, &change).expect("should apply AA update to child sample");

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
        sample_path
            .new_path_with(conn, 15, 25, &edge_to_insert, &edge_from_insert)
            .unwrap();
    }

    fn add_pathless_block_group(conn: &GraphConnection, collection_name: &str, sample_name: &str) {
        // Mimics a protein sequence graph created by `derive translation`, which is
        // created in the same sample as the DNA graph it came from but never gets a
        // `Path` row -- propagate_gff must not require a current path for it.
        BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name,
                sample_name,
                name: "protein",
                parent_block_group_id: None,
                is_default: false,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_propagate_ignores_pathless_block_group_in_sample() {
        let conn = get_connection();
        create_block_group(&conn);
        apply_child_sample_update_from_aa_fasta(&conn);
        add_pathless_block_group(&conn, "test", Sample::DEFAULT_NAME);

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
        .expect("should propagate gff even though the source sample has a pathless block group");
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
