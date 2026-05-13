use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use gen_core::{NodeIntervalBlock, range::Range};
use gen_models::{
    block_group::BlockGroup, db::GraphConnection, errors::PathError, path::Path, sample::Sample,
};
use itertools::Itertools;
use thiserror::Error;

use crate::gfa::{Link, Path as GFAPath, Segment, path_line, write_links, write_segments};

#[derive(Debug, Error)]
pub enum GfaDiffError {
    #[error("I/O error while writing GFA diff: {0}")]
    Io(#[from] std::io::Error),
    #[error("Path error while generating GFA diff: {0}")]
    Path(#[from] PathError),
}

pub fn gfa_sample_diff(
    conn: &GraphConnection,
    collection_name: &str,
    filename: &PathBuf,
    from_sample_name: &str,
    to_sample_name: &str,
) -> Result<(), GfaDiffError> {
    /*
    Generate a GFA file that represents the differences between two samples in a collection.

    General approach: For each pair of shared block groups between the samples, get the current path
    for each and call find_block_mappings on the pair of paths to get mappings between shared
    regions on the paths.  Each shared region may cover multiple nodes.  We assume the mappings will
    be in order from upstream to downstream on the sequences.  We iterate over them to produce a
    list of ranges on each path's sequence.  For each mapping, and for each path, if there is an
    unshared region before the common region, we append the range for that unshared region to the
    path's list of ranges, and then the range for the common region.  Obviously there may be an
    unshared region on each path at the very end, and if so, we append the range for that region to
    the appropriate path's list.

    We then convert the list of ranges for a path to a list of segments, with each range being
    converted to a segment with the range's start coordinate, and the subsequence of the path for
    that range.  We also create links, with one link per pair of adjacent segments in the path.
    Each shared segment will have two links going in (one for each path) and two links going out.
    Each unshared segment will have one link in and one link out.

    We also create a GFA path for each path, which is just a list of the segments generated for that
    path.
    */
    let source_block_groups = Sample::get_block_groups(conn, collection_name, from_sample_name);
    let target_block_groups = Sample::get_block_groups(conn, collection_name, to_sample_name);

    let source_paths_by_name = source_block_groups
        .iter()
        .map(|bg| (bg.name.clone(), BlockGroup::get_current_path(conn, &bg.id)))
        .collect::<HashMap<String, Path>>();
    let target_paths_by_name = target_block_groups
        .iter()
        .map(|bg| (bg.name.clone(), BlockGroup::get_current_path(conn, &bg.id)))
        .collect::<HashMap<String, Path>>();

    let mut segments = HashSet::new();
    let mut links = HashSet::new();
    let mut paths = vec![];

    let target_path_names = target_paths_by_name
        .keys()
        .cloned()
        .collect::<HashSet<String>>();
    let source_path_names = source_paths_by_name
        .keys()
        .cloned()
        .collect::<HashSet<String>>();
    let path_names = source_path_names
        .union(&target_path_names)
        .cloned()
        .collect::<Vec<String>>();

    for path_name in &path_names {
        let source_path_result = source_paths_by_name.get(path_name);
        let target_path_result = target_paths_by_name.get(path_name);

        let mappings = match (source_path_result, target_path_result) {
            (Some(source_path), Some(target_path)) => {
                source_path.find_block_mappings(conn, target_path)?
            }
            _ => vec![],
        };

        let mut source_ranges = vec![];
        let mut target_ranges = vec![];

        let mut last_source_position = 0;
        let mut last_target_position = 0;
        for mapping in &mappings {
            // Iterate over the shared regions between the source and target path.  If there is an
            // unshared region before the shared region, append the range for the unshared region.
            // Then append the range for the shared region.
            if mapping.source_range.start > last_source_position {
                source_ranges.push(Range {
                    start: last_source_position,
                    end: mapping.source_range.start,
                });
            }
            source_ranges.push(mapping.source_range.clone());
            last_source_position = mapping.source_range.end;
            if mapping.target_range.start > last_target_position {
                target_ranges.push(Range {
                    start: last_target_position,
                    end: mapping.target_range.start,
                });
            }
            target_ranges.push(mapping.target_range.clone());
            last_target_position = mapping.target_range.end;
        }

        if let Some(source_path) = source_path_result {
            let source_sequence = source_path.sequence(conn)?;

            let source_len = source_sequence.len() as i64;
            if last_source_position < source_len {
                source_ranges.push(Range {
                    start: last_source_position,
                    end: source_len,
                });
            }

            let source_node_blocks = source_path.node_block_partition(conn, source_ranges)?;
            let source_segments = segments_from_blocks(&source_node_blocks, &source_sequence);
            segments.extend(source_segments.iter().cloned());

            let source_links = links_from_blocks(&source_node_blocks);
            links.extend(source_links.iter().cloned());

            let source_gfa_path =
                path_from_segments(from_sample_name, source_path, &source_segments);
            paths.push(source_gfa_path);
        }

        if let Some(target_path) = target_path_result {
            let target_sequence = target_path.sequence(conn)?;

            let target_len = target_sequence.len() as i64;
            if last_target_position < target_len {
                target_ranges.push(Range {
                    start: last_target_position,
                    end: target_len,
                });
            }

            let target_node_blocks = target_path.node_block_partition(conn, target_ranges)?;
            let target_segments = segments_from_blocks(&target_node_blocks, &target_sequence);
            segments.extend(target_segments.iter().cloned());

            let target_links = links_from_blocks(&target_node_blocks);
            links.extend(target_links.iter().cloned());

            let target_gfa_path = path_from_segments(to_sample_name, target_path, &target_segments);
            paths.push(target_gfa_path);
        }
    }

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    write_segments(&mut writer, &segments.iter().collect::<Vec<&Segment>>())?;
    write_links(&mut writer, &links.iter().collect::<Vec<&Link>>())?;

    for path in paths {
        writer.write_all(&path_line(&path).into_bytes())?;
    }

    Ok(())
}

fn segments_from_blocks(node_blocks: &Vec<NodeIntervalBlock>, sequence: &str) -> Vec<Segment> {
    let mut segments = vec![];
    for block in node_blocks {
        let start = block.start as usize;
        let end = block.end as usize;
        let segment = Segment {
            sequence: sequence[start..end].to_string(),
            node_id: block.node_id,
            sequence_start: block.sequence_start,
            sequence_end: block.sequence_end,
            strand: block.strand,
        };
        segments.push(segment.clone());
    }
    segments
}

fn links_from_blocks(node_blocks: &[NodeIntervalBlock]) -> Vec<Link> {
    let mut links = vec![];

    for (block1, block2) in node_blocks.iter().tuple_windows() {
        let source_segment = Segment {
            sequence: "".to_string(),
            node_id: block1.node_id,
            sequence_start: block1.sequence_start,
            sequence_end: block1.sequence_end,
            strand: block1.strand,
        };
        let target_segment = Segment {
            sequence: "".to_string(),
            node_id: block2.node_id,
            sequence_start: block2.sequence_start,
            sequence_end: block2.sequence_end,
            strand: block2.strand,
        };

        // General note about how we encode segment IDs.  The node ID and the start coordinate in
        // the sequence are all that's needed, because the end coordinate can be inferred from the
        // length of the segment's sequence.  So the segment ID is of the form <node ID>.<start
        // coordinate>
        let link = Link {
            source_segment_id: source_segment.segment_id(),
            source_strand: block1.strand,
            target_segment_id: target_segment.segment_id(),
            target_strand: block2.strand,
        };
        links.push(link);
    }

    links
}

fn path_from_segments(sample_name: &str, path: &Path, segments: &[Segment]) -> GFAPath {
    let path_name = format!("{sample_name}.{}", path.name);
    GFAPath {
        name: path_name.clone(),
        segment_ids: segments.iter().map(|s| s.segment_id()).collect(),
        node_strands: segments.iter().map(|s| s.strand).collect(),
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use gen_core::{HashId, NO_CHROMOSOME_INDEX, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        block_group::BlockGroup,
        block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
        collection::Collection,
        edge::Edge,
        node::Node,
        sequence::Sequence,
    };
    use tempfile::tempdir;

    use super::*;
    use crate::{
        imports::gfa::import_gfa,
        test_helpers::{create_bg, setup_gen},
        track_database,
    };

    #[test]
    fn test_gfa_diff() {
        // Sets up a basic graph and then exports it to a GFA file
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection_name = "test collection";
        Collection::create(conn, collection_name).unwrap();
        let block_group = create_bg(
            conn,
            collection_name,
            Sample::DEFAULT_NAME,
            "test block group",
        );
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();

        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path1 = Path::create(conn, "parent", &block_group.id, &edge_ids);

        // Set up child
        let _child_sample = Sample::get_or_create_child(
            conn,
            collection_name,
            "child",
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .unwrap();
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge4 = Edge::create(
            conn,
            node1_id,
            2,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node3_id,
            4,
            Strand::Forward,
            node1_id,
            6,
            Strand::Forward,
        )
        .unwrap();

        let child_block_groups = Sample::get_block_groups(conn, collection_name, "child");
        let child_block_group = child_block_groups.first().unwrap();
        let child_edge_ids = [edge4.id, edge5.id];
        let child_block_group_edges = child_edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: child_block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &child_block_group_edges);
        let original_child_path = BlockGroup::get_current_path(conn, &child_block_group.id);
        let _child_path = original_child_path.new_path_with(conn, 2, 6, &edge4, &edge5);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("parent-child-diff.gfa");
        gfa_sample_diff(
            conn,
            collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            "child",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let new_child_block_group = Collection::get_block_groups(conn, "test collection 2")
            .pop()
            .unwrap();
        let all_child_sequences =
            BlockGroup::get_all_sequences(conn, &new_child_block_group.id, false);

        // We've replaced the middle AAAA with CCCC, so expect that as the child sequence
        assert_eq!(
            all_child_sequences,
            ["AAAAAAAATTTTTTTT", "AACCCCAATTTTTTTT"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );

        // Set up grandchild
        let _grandchild_sample = Sample::get_or_create_child(
            conn,
            collection_name,
            "grandchild",
            vec!["child".to_string()],
        )
        .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGG")
            .save(conn)
            .unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();
        let edge6 = Edge::create(
            conn,
            node2_id,
            2,
            Strand::Forward,
            node4_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge7 = Edge::create(
            conn,
            node4_id,
            4,
            Strand::Forward,
            node2_id,
            6,
            Strand::Forward,
        )
        .unwrap();

        let grandchild_block_groups = Sample::get_block_groups(conn, collection_name, "grandchild");
        let grandchild_block_group = grandchild_block_groups.first().unwrap();
        let grandchild_edge_ids = [edge6.id, edge7.id];
        let grandchild_block_group_edges = grandchild_edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: grandchild_block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &grandchild_block_group_edges);
        let original_grandchild_path =
            BlockGroup::get_current_path(conn, &grandchild_block_group.id);
        let _grandchild_path = original_grandchild_path.new_path_with(conn, 10, 14, &edge6, &edge7);

        let gfa_path = temp_dir.path().join("parent-grandchild-diff.gfa");
        gfa_sample_diff(
            conn,
            collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            "grandchild",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 3",
            Sample::DEFAULT_NAME,
        );
        let new_grandchild_block_group = Collection::get_block_groups(conn, "test collection 3")
            .pop()
            .unwrap();
        let all_grandchild_sequences =
            BlockGroup::get_all_sequences(conn, &new_grandchild_block_group.id, false);

        // We've replaced the middle AAAA with CCCC and the middle TTTT with GGGG, so four possible sequences
        assert_eq!(
            all_grandchild_sequences,
            [
                "AAAAAAAATTTTTTTT",
                "AACCCCAATTTTTTTT",
                "AACCCCAATTGGGGTT",
                "AAAAAAAATTGGGGTT"
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<String>>()
        );

        let gfa_path = temp_dir.path().join("child-grandchild-diff.gfa");
        gfa_sample_diff(conn, collection_name, &gfa_path, "child", "grandchild").unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 4",
            Sample::DEFAULT_NAME,
        );

        let new_grandchild_block_group = Collection::get_block_groups(conn, "test collection 4")
            .pop()
            .unwrap();
        let all_grandchild_sequences =
            BlockGroup::get_all_sequences(conn, &new_grandchild_block_group.id, false);

        assert_eq!(
            all_grandchild_sequences,
            ["AACCCCAATTTTTTTT", "AACCCCAATTGGGGTT"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );
    }

    #[test]
    fn test_gfa_diff_against_nothing() {
        // Confirm diff of a sample against nothing is just the sample
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection_name = "test collection";
        Collection::create(conn, collection_name).unwrap();
        let _sample = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "test sample",
                ..Default::default()
            },
        );
        let block_group = create_bg(conn, collection_name, "test sample", "test block group");
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();

        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path1 = Path::create(conn, "test path", &block_group.id, &edge_ids);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("diff-against-nothing.gfa");
        gfa_sample_diff(
            conn,
            collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            "test sample",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let new_block_group = Collection::get_block_groups(conn, "test collection 2")
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_block_group.id, false);

        assert_eq!(
            all_sequences,
            ["AAAAAAAATTTTTTTT"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );
    }

    #[test]
    fn test_self_diff() {
        // Confirm diff of a sample to itself just results in a graph that's a single path
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection_name = "test collection";
        Collection::create(conn, collection_name).unwrap();
        let _sample = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "test sample",
                ..Default::default()
            },
        );
        let block_group = create_bg(conn, collection_name, "test sample", "test block group");
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();

        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path1 = Path::create(conn, "test path", &block_group.id, &edge_ids);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("self-diff.gfa");
        gfa_sample_diff(
            conn,
            collection_name,
            &gfa_path,
            "test sample",
            "test sample",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let new_block_group = Collection::get_block_groups(conn, "test collection 2")
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_block_group.id, false);

        assert_eq!(
            all_sequences,
            ["AAAAAAAATTTTTTTT"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );
    }

    #[test]
    fn test_gfa_diff_unrelated_paths() {
        // Confirm diff of a sample to totally unrelated sample produces two separate paths
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection_name = "test collection";
        Collection::create(conn, collection_name).unwrap();
        let _sample1 = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "sample1",
                ..Default::default()
            },
        );
        let block_group = create_bg(conn, collection_name, "sample1", "test block group");
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();

        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path1 = Path::create(conn, "parent", &block_group.id, &edge_ids);

        let _sample2 = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "sample2",
                ..Default::default()
            },
        );
        let block_group2 = create_bg(conn, collection_name, "sample2", "test block group 2");
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGGGGG")
            .save(conn)
            .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();

        let edge4 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node4_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge6 = Edge::create(
            conn,
            node4_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id, edge6.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group2.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path2 = Path::create(conn, "parent", &block_group2.id, &edge_ids);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("unrelated-diff.gfa");
        gfa_sample_diff(conn, collection_name, &gfa_path, "sample1", "sample2").unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 3",
            Sample::DEFAULT_NAME,
        );

        let new_block_group = Collection::get_block_groups(conn, "test collection 3")
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_block_group.id, false);

        assert_eq!(
            all_sequences,
            ["AAAAAAAATTTTTTTT", "GGGGGGGGCCCCCCCC"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );
    }

    #[test]
    fn test_gfa_diff_unrelated_paths_matching_block_group_names() {
        // Confirm diff of two paths that are in the same block group but don't share any nodes
        // results in two disjoint sequences
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection_name = "test collection";
        Collection::create(conn, collection_name).unwrap();
        let _sample1 = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "sample1",
                ..Default::default()
            },
        );
        let block_group = create_bg(conn, collection_name, "sample1", "test block group");
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTT")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();

        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            8,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge3 = Edge::create(
            conn,
            node2_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id, edge3.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path1 = Path::create(conn, "parent", &block_group.id, &edge_ids);

        let _sample2 = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "sample2",
                ..Default::default()
            },
        );
        let block_group2 = create_bg(conn, collection_name, "sample2", "test block group");
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGGGGGG")
            .save(conn)
            .unwrap();
        let sequence4 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCCCCCC")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let node4_id = Node::create(conn, &sequence4.hash, &HashId::convert_str("4")).unwrap();

        let edge4 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge5 = Edge::create(
            conn,
            node3_id,
            8,
            Strand::Forward,
            node4_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge6 = Edge::create(
            conn,
            node4_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge4.id, edge5.id, edge6.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group2.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path2 = Path::create(conn, "parent", &block_group2.id, &edge_ids);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("unrelated-diff.gfa");
        gfa_sample_diff(conn, collection_name, &gfa_path, "sample1", "sample2").unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 3",
            Sample::DEFAULT_NAME,
        );

        let new_block_group = Collection::get_block_groups(conn, "test collection 3")
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(conn, &new_block_group.id, false);

        assert_eq!(
            all_sequences,
            ["AAAAAAAATTTTTTTT", "GGGGGGGGCCCCCCCC"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );
    }

    #[test]
    fn test_gfa_diff_overlapping_replacements() {
        // Set up a child with a replacement, then a grandchild with a replacement on the child that
        // partially overlaps the child's replacement, and confirm diffs between all pairs from
        // (original, child, grandchild)
        let context = setup_gen();
        let conn = context.graph().conn();
        let op_conn = context.operations().conn();

        track_database(conn, op_conn).unwrap();

        let collection_name = "test collection";
        Collection::create(conn, collection_name).unwrap();
        let block_group = create_bg(
            conn,
            collection_name,
            Sample::DEFAULT_NAME,
            "test block group",
        );
        let sequence1 = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAAAAAAAAAA")
            .save(conn)
            .unwrap();
        let node1_id = Node::create(conn, &sequence1.hash, &HashId::convert_str("1")).unwrap();

        let edge1 = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node1_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge2 = Edge::create(
            conn,
            node1_id,
            16,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [edge1.id, edge2.id];
        let block_group_edges = edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let _path1 = Path::create(conn, "parent", &block_group.id, &[edge1.id, edge2.id]);

        // Set up child
        let _child_sample = Sample::get_or_create_child(
            conn,
            collection_name,
            "child",
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .unwrap();
        let sequence2 = Sequence::new()
            .sequence_type("DNA")
            .sequence("CCCC")
            .save(conn)
            .unwrap();
        let node2_id = Node::create(conn, &sequence2.hash, &HashId::convert_str("2")).unwrap();
        let edge3 = Edge::create(
            conn,
            node1_id,
            2,
            Strand::Forward,
            node2_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge4 = Edge::create(
            conn,
            node2_id,
            4,
            Strand::Forward,
            node1_id,
            6,
            Strand::Forward,
        )
        .unwrap();

        let child_block_groups = Sample::get_block_groups(conn, collection_name, "child");
        let child_block_group = child_block_groups.first().unwrap();
        let child_edge_ids = [edge3.id, edge4.id];
        let child_block_group_edges = child_edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: child_block_group.id,
                edge_id,
                chromosome_index: NO_CHROMOSOME_INDEX,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &child_block_group_edges);
        let original_child_path = BlockGroup::get_current_path(conn, &child_block_group.id);
        let _child_path = original_child_path.new_path_with(conn, 2, 6, &edge3, &edge4);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("parent-child-diff.gfa");
        gfa_sample_diff(
            conn,
            collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            "child",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 2",
            Sample::DEFAULT_NAME,
        );

        let new_child_block_group = Collection::get_block_groups(conn, "test collection 2")
            .pop()
            .unwrap();
        let all_child_sequences =
            BlockGroup::get_all_sequences(conn, &new_child_block_group.id, false);

        // We've replaced [2, 6) of AAAA with CCCC
        assert_eq!(
            all_child_sequences,
            ["AAAAAAAAAAAAAAAA", "AACCCCAAAAAAAAAA"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );

        // Set up grandchild
        let _grandchild_sample = Sample::get_or_create_child(
            conn,
            collection_name,
            "grandchild",
            vec!["child".to_string()],
        )
        .unwrap();
        let sequence3 = Sequence::new()
            .sequence_type("DNA")
            .sequence("GGGG")
            .save(conn)
            .unwrap();
        let node3_id = Node::create(conn, &sequence3.hash, &HashId::convert_str("3")).unwrap();
        let edge5 = Edge::create(
            conn,
            node2_id,
            2,
            Strand::Forward,
            node3_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge6 = Edge::create(
            conn,
            node3_id,
            4,
            Strand::Forward,
            node1_id,
            10,
            Strand::Forward,
        )
        .unwrap();

        let grandchild_block_groups = Sample::get_block_groups(conn, collection_name, "grandchild");
        let grandchild_block_group = grandchild_block_groups.first().unwrap();
        let grandchild_edge_ids = [edge5.id, edge6.id];
        let grandchild_block_group_edges = grandchild_edge_ids
            .iter()
            .map(|&edge_id| BlockGroupEdgeData {
                block_group_id: grandchild_block_group.id,
                edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &grandchild_block_group_edges);
        let original_grandchild_path =
            BlockGroup::get_current_path(conn, &grandchild_block_group.id);
        let _grandchild_path = original_grandchild_path.new_path_with(conn, 4, 10, &edge5, &edge6);

        let gfa_path = temp_dir.path().join("parent-grandchild-diff.gfa");
        gfa_sample_diff(
            conn,
            collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            "grandchild",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 3",
            Sample::DEFAULT_NAME,
        );

        let new_grandchild_block_group = Collection::get_block_groups(conn, "test collection 3")
            .pop()
            .unwrap();
        let all_grandchild_sequences =
            BlockGroup::get_all_sequences(conn, &new_grandchild_block_group.id, false);

        // Original is AAAAAAAAAAAAAAAA
        // Grandchild is AACCGGGGAAAAAA
        // Because the grandchild change overlaps with the child change, there are no other possibiiities
        assert_eq!(
            all_grandchild_sequences,
            ["AAAAAAAAAAAAAAAA", "AACCGGGGAAAAAA"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );

        let gfa_path = temp_dir.path().join("child-grandchild-diff.gfa");
        gfa_sample_diff(conn, collection_name, &gfa_path, "child", "grandchild").unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 4",
            Sample::DEFAULT_NAME,
        );

        let new_grandchild_block_group = Collection::get_block_groups(conn, "test collection 4")
            .pop()
            .unwrap();
        let all_grandchild_sequences =
            BlockGroup::get_all_sequences(conn, &new_grandchild_block_group.id, false);

        // Child is      AACCCCAAAAAAAAAA
        // Grandchild is AACCGGGGAAAAAA
        assert_eq!(
            all_grandchild_sequences,
            ["AACCCCAAAAAAAAAA", "AACCGGGGAAAAAA"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );
    }
}
