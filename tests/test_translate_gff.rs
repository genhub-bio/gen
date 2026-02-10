use std::{fs::File, io::BufReader, path::PathBuf};

use r#gen::{
    annotations::translate::gff::translate_gff, test_helpers::setup_gen, track_database,
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
    let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/gffs/simple.gff3");
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
    translate_gff(
        conn,
        &collection,
        "foo",
        BufReader::new(File::open(gff_path.clone()).unwrap()),
        &mut buffer,
    )
    .unwrap();
    let results = String::from_utf8(buffer).unwrap();
    assert_eq!(
        results,
        "\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t1\t3\t.\t-\t.\tID=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t4\t4\t.\t-\t.\tID=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t5\t17\t.\t-\t.\tID=ENSG00000294541.1\n\
        086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\tgene\t18\t20\t.\t-\t.\tID=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t1\t3\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t5\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
        086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\ttranscript\t18\t20\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t5\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
        086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\texon\t18\t19\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\tgene\t4\t4\t.\t-\t.\tID=ENSG00000277248.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\tgene\t5\t15\t.\t-\t.\tID=ENSG00000277248.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\ttranscript\t4\t4\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\ttranscript\t5\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\texon\t4\t4\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\texon\t5\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n\
        "
    );

    // The None sample has no variants, so should be a simple mapping with the single split
    let mut buffer = Vec::new();
    translate_gff(
        conn,
        &collection,
        None,
        BufReader::new(File::open(gff_path.clone()).unwrap()),
        &mut buffer,
    )
    .unwrap();
    let results = String::from_utf8(buffer).unwrap();
    assert_eq!(
        results,
        "\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\tgene\t1\t17\t.\t-\t.\tID=ENSG00000294541.1\n\
        086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\tgene\t18\t20\t.\t-\t.\tID=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\ttranscript\t1\t17\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
        086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\ttranscript\t18\t20\t.\t-\t.\tID=ENST00000724296.1;Parent=ENSG00000294541.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t4\t8\t.\t-\t.\tID=exon:ENST00000724296.1:1;Parent=ENST00000724296.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t10\t14\t.\t-\t.\tID=exon:ENST00000724296.1:2;Parent=ENST00000724296.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tHAVANA\texon\t16\t17\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
        086ae30894dda8efdc19d4dfadd5e6e24af8066e9ee63e56abe897993bebd112\tHAVANA\texon\t18\t19\t.\t-\t.\tID=exon:ENST00000724296.1:3;Parent=ENST00000724296.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\tgene\t3\t15\t.\t-\t.\tID=ENSG00000277248.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\ttranscript\t3\t15\t.\t-\t.\tID=ENST00000615943.1;Parent=ENSG00000277248.1\n\
        0cbb0b7e8171228e0ea97287f369c0099f0ccabc6a9950320650bc27bd974c8a\tENSEMBL\texon\t3\t15\t.\t-\t.\tID=exon:ENST00000615943.1:1;Parent=ENST00000615943.1\n\
        "
    );
}
