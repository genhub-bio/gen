use std::{fs::File, path::PathBuf};

use r#gen::{
    annotations::translate::bed::translate_bed, test_helpers::setup_gen, track_database,
    updates::vcf::update_with_vcf,
};

mod test_helpers;

use test_helpers::get_simple_sequence;

#[test]
fn translates_coordinates_to_nodes() {
    let context = setup_gen();
    let conn = context.graph().conn();
    let op_conn = context.operations().conn();

    track_database(conn, op_conn).unwrap();

    let vcf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.vcf");
    let bed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/beds/simple.bed");
    let collection = "test".to_string();

    get_simple_sequence(conn);

    update_with_vcf(
        &context,
        &vcf_path.to_str().unwrap().to_string(),
        &collection,
        "".to_string(),
        "".to_string(),
        None,
    )
    .unwrap();
    let mut buffer = Vec::new();
    // "foo" is a sample from simple.vcf
    translate_bed(
        conn,
        &collection,
        "foo",
        File::open(bed_path.clone()).unwrap(),
        &mut buffer,
    )
    .unwrap();
    let results = String::from_utf8(buffer).unwrap();
    assert_eq!(
        results,
        "\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t1\t3\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t3\t4\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t4\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n\
    086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\t17\t23\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n"
    );

    // The None sample has no variants, so should be a simple mapping and covers the split node
    let mut buffer = Vec::new();
    translate_bed(
        conn,
        &collection,
        None,
        File::open(bed_path).unwrap(),
        &mut buffer,
    )
    .unwrap();
    let results = String::from_utf8(buffer).unwrap();
    assert_eq!(
        results,
        "\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t1\t10\tabc123.1\t0\t-\t1\t10\t0,0,0\t3\t102,188,129,\t0,3508,4691,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t5\t8\txyz.1\t0\t-\t5\t8\t0,0,0\t1\t113,\t0,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t10\t16\txyz.2\t0\t+\t10\t16\t0,0,0\t2\t142,326,\t0,10710,\n\
    0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\t14\t17\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n\
    086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\t17\t23\tfoo.1\t0\t+\t14\t23\t0,0,0\t2\t142,326,\t0,10710,\n"
    );
}
