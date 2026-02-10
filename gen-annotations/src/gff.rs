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
    from_sample_name: Option<&str>,
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
    let target_block_groups = Sample::get_block_groups(conn, collection_name, Some(to_sample_name));
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
        let mapping = source_path.get_mapping_tree(conn, target_path);
        path_mappings_by_bg_name.insert(name, mapping);
    }

    let sequence_lengths_by_path_name = target_paths_by_bg_name
        .iter()
        .map(|(name, path)| (name.clone(), path.sequence(conn).len() as i64))
        .collect::<HashMap<String, i64>>();

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
