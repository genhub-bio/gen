use gen_models::{block_group::BlockGroup, collection::Collection};

use super::*;
use crate::test_helpers::{setup_block_group, setup_gen};

fn test_matcher() -> GenGraphMatcher {
    let ctx = setup_gen();
    let conn = ctx.graph().conn();
    Collection::create(conn, "test");
    let (block_group_id, _path) = setup_block_group(conn);
    let graph = BlockGroup::get_graph(conn, &block_group_id);
    GenGraphMatcher::new(conn, graph)
}

fn test_protein_matcher() -> GenGraphMatcher {
    let ctx = setup_gen();
    let conn = ctx.graph().conn();
    Collection::create(conn, "test");
    let (block_group_id, _path) = setup_block_group(conn);
    let graph = BlockGroup::get_graph(conn, &block_group_id);
    GenGraphMatcher::new_protein(conn, graph)
}

#[test]
fn reverse_complement_basic() {
    assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
    assert_eq!(reverse_complement(b"AAAA"), b"TTTT");
    assert_eq!(reverse_complement(b"TTTT"), b"AAAA");
    assert_eq!(reverse_complement(b"AACGT"), b"ACGTT");
    assert_eq!(reverse_complement(b""), b"");
}

#[test]
fn reverse_complement_lowercase() {
    assert_eq!(reverse_complement(b"acgt"), b"ACGT");
    assert_eq!(reverse_complement(b"aatt"), b"AATT");
}

#[test]
fn reverse_complement_degenerate_codes() {
    assert_eq!(reverse_complement(b"RYSWKMBVDH"), b"DHBVKMWSRY");
    assert_eq!(reverse_complement(b"ry"), b"RY");
    assert_eq!(reverse_complement(b"ACGR"), b"YCGT");
}

#[test]
fn reverse_complement_uracil() {
    assert_eq!(reverse_complement(b"U"), b"A");
    assert_eq!(reverse_complement(b"AU"), b"AT");
}

#[test]
fn iupac_matches_standard_bases() {
    assert!(iupac_matches(b'A', b'A'));
    assert!(iupac_matches(b'C', b'C'));
    assert!(iupac_matches(b'G', b'G'));
    assert!(iupac_matches(b'T', b'T'));

    assert!(!iupac_matches(b'A', b'C'));
    assert!(!iupac_matches(b'C', b'G'));
    assert!(!iupac_matches(b'G', b'T'));
    assert!(!iupac_matches(b'T', b'A'));
}

#[test]
fn iupac_matches_degenerate_codes() {
    assert!(iupac_matches(b'N', b'A'));
    assert!(iupac_matches(b'N', b'C'));
    assert!(iupac_matches(b'N', b'G'));
    assert!(iupac_matches(b'N', b'T'));

    assert!(iupac_matches(b'R', b'A'));
    assert!(iupac_matches(b'R', b'G'));
    assert!(!iupac_matches(b'R', b'C'));
    assert!(!iupac_matches(b'R', b'T'));

    assert!(iupac_matches(b'Y', b'C'));
    assert!(iupac_matches(b'Y', b'T'));
    assert!(!iupac_matches(b'Y', b'A'));
    assert!(!iupac_matches(b'Y', b'G'));

    assert!(iupac_matches(b'B', b'C'));
    assert!(iupac_matches(b'B', b'G'));
    assert!(iupac_matches(b'B', b'T'));
    assert!(!iupac_matches(b'B', b'A'));
}

#[test]
fn iupac_matches_unknown_falls_back_to_exact() {
    assert!(iupac_matches(b'X', b'X'));
    assert!(iupac_matches(b'!', b'!'));
    assert!(!iupac_matches(b'X', b'A'));
    assert!(!iupac_matches(b'!', b'A'));
}

#[test]
fn query_contains_degenerate_iupac_detects_only_degenerate_codes() {
    assert!(!query_contains_degenerate_iupac(b"ACGT"));
    assert!(!query_contains_degenerate_iupac(b"acgt"));
    assert!(query_contains_degenerate_iupac(b"ACGN"));
    assert!(query_contains_degenerate_iupac(b"ry"));
}

#[test]
fn contains_exact_within_single_node() {
    let matcher = test_matcher();

    assert!(matcher.contains(b"AAAA"));
    assert!(matcher.contains(b"TTTT"));
    assert!(matcher.contains(b"CCCC"));
    assert!(matcher.contains(b"GGGG"));
}

#[test]
fn contains_spanning_node_boundary() {
    let matcher = test_matcher();

    assert!(matcher.contains(b"AAAATTTT"));
    assert!(matcher.contains(b"TTTTCCCC"));
}

#[test]
fn contains_absent_exact_sequence() {
    let matcher = test_matcher();

    assert!(!matcher.contains(b"ACGT"));
}

#[test]
fn contains_empty_query_always_true() {
    let matcher = test_matcher();

    assert!(matcher.contains(b""));
}

#[test]
fn contains_reverse_complement_for_dna_matcher() {
    let matcher = test_matcher();

    assert!(matcher.contains(b"CCCCCCCCCC"));
}

#[test]
fn protein_matcher_does_not_search_reverse_complement() {
    let matcher = test_protein_matcher();

    // AAAAACCCCC is absent. Its reverse complement, GGGGGTTTTT, is present
    // across the G-node and T-node, but protein mode does not search reverse
    // complements.
    assert!(!matcher.contains(b"AAAAACCCCC"));
}

#[test]
fn contains_iupac_n_matches_any_base() {
    let matcher = test_matcher();

    assert!(matcher.contains(b"NNNN"));
}

#[test]
fn contains_iupac_r_matches_purine() {
    let matcher = test_matcher();

    assert!(matcher.contains(b"RRRR"));
}

#[test]
fn contains_iupac_y_matches_pyrimidine() {
    let matcher = test_matcher();

    assert!(matcher.contains(b"YYYY"));
}

#[test]
fn protein_matcher_does_not_use_iupac_matching() {
    let matcher = test_protein_matcher();

    assert!(!matcher.contains(b"NNNN"));
    assert!(!matcher.contains(b"RRRR"));
    assert!(!matcher.contains(b"YYYY"));
}

#[test]
fn find_all_single_node_match() {
    let matcher = test_matcher();

    let hits = matcher.find_all(b"AAAAAAAAAA");
    assert_eq!(hits.len(), 2);

    assert!(
        hits.iter()
            .any(|hit| { hit.start_offset == 0 && hit.end_offset == 10 && hit.blocks.len() == 1 })
    );
}

#[test]
fn find_all_spanning_match() {
    let matcher = test_matcher();

    let hits = matcher.find_all(b"AAAAATTTTT");
    assert_eq!(hits.len(), 2);

    assert!(
        hits.iter()
            .any(|hit| { hit.blocks.len() == 2 && hit.start_offset == 5 && hit.end_offset == 5 })
    );
}

#[test]
fn find_all_no_match_returns_empty() {
    let matcher = test_matcher();

    assert!(matcher.find_all(b"ACGT").is_empty());
}

#[test]
fn find_all_empty_query_returns_empty() {
    let matcher = test_matcher();

    assert!(matcher.find_all(b"").is_empty());
}

#[test]
fn find_all_iupac_n_matches_each_node() {
    let matcher = test_matcher();

    let hits = matcher.find_all(b"NNNNNNNNNN");
    assert_eq!(hits.len(), 62);
}

#[test]
fn find_all_exact_still_works_without_iupac_codes() {
    let matcher = test_matcher();

    let hits = matcher.find_all(b"GGGGGGGGGG");
    assert_eq!(hits.len(), 2);
}

#[test]
fn seed_index_build_and_find() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    assert_eq!(index.k, 4);
    assert!(index.table.contains_key(b"AAAA".as_ref()));
    assert!(index.table.contains_key(b"TTTT".as_ref()));
    assert!(index.table.contains_key(b"CCCC".as_ref()));
    assert!(index.table.contains_key(b"GGGG".as_ref()));
    assert!(index.table.contains_key(b"ATTT".as_ref()));
}

#[test]
fn seed_index_find_all_with_index_matches_exact_forward_search() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    let query = b"AAAAATTTTT";
    let via_index = matcher.find_all_with_seed_index(&index, query).unwrap();
    assert_eq!(via_index.len(), 1);
}

#[test]
fn seed_index_find_all_absent_query_empty() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    let hits = matcher.find_all_with_seed_index(&index, b"ACGT").unwrap();
    assert!(hits.is_empty());
}

#[test]
fn seed_index_short_query_falls_back_to_exact_forward_find_all() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    let via_index = matcher.find_all_with_seed_index(&index, b"AA").unwrap();
    assert_eq!(via_index.len(), 9);
}

#[test]
fn seed_index_rejects_degenerate_query() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    let err = matcher
        .find_all_with_seed_index(&index, b"ATGN")
        .unwrap_err();
    assert_eq!(err, SeedIndexSearchError::UnsupportedQuery);
}

#[test]
fn seed_index_roundtrip_bytes() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    let bytes = index.to_bytes_with_header(true).unwrap();
    let loaded = SeedIndex::from_bytes_with_header(&bytes, 4, true).unwrap();
    assert_eq!(loaded.k, index.k);
    assert_eq!(loaded.table.len(), index.table.len());
}

#[test]
fn seed_index_error_cases() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    let bytes = index.to_bytes_with_header(true).unwrap();

    assert!(matches!(
        SeedIndex::from_bytes_with_header(&bytes[..2], 4, true),
        Err(SeedIndexIoError::Truncated)
    ));
    assert!(matches!(
        SeedIndex::from_bytes_with_header(&bytes, 8, true),
        Err(SeedIndexIoError::KMismatch { .. })
    ));
    assert!(matches!(
        SeedIndex::from_bytes_with_header(&bytes, 4, false),
        Err(SeedIndexIoError::CaseSensitiveMismatch { .. })
    ));
}

#[test]
fn seed_index_save_and_load_path() {
    let matcher = test_matcher();

    let index = SeedIndex::build(&matcher, 4);
    let tmp = tempfile::NamedTempFile::new().unwrap();
    index.save_to_path(tmp.path(), true).unwrap();
    let loaded = SeedIndex::load_from_path(tmp.path(), true).unwrap();
    assert_eq!(loaded.k, index.k);
    assert_eq!(loaded.table.len(), index.table.len());
}
