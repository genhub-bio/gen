use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use gen_core::{Workspace, is_end_node, is_start_node};
use gen_diff::{graph::DiffGenGraph, sample::build_sample_block_group_diff};
use gen_models::{
    block_group::{BlockGroup, BlockGroupError},
    db::GraphConnection,
    errors::PathError,
    node::Node,
    path::Path,
    sample::Sample,
};
use petgraph::Direction;
use thiserror::Error;

use crate::gfa::{Link, Path as GFAPath, Segment, path_line, write_links, write_segments};

#[derive(Debug, Error)]
pub enum GfaDiffError {
    #[error("I/O error while writing GFA diff: {0}")]
    Io(#[from] std::io::Error),
    #[error("Path error while generating GFA diff: {0}")]
    Path(#[from] PathError),
    #[error("Block group error while generating GFA diff: {0}")]
    BlockGroup(#[from] BlockGroupError),
}

/// Exports the shared sample diff graph as GFA segments and links.
///
/// Current sample paths are mapped onto the graph's normalized slices to keep
/// their sequence and traversal order in the exported path records.
pub fn gfa_sample_diff(
    conn: &GraphConnection,
    workspace: &Workspace,
    collection_name: &str,
    filename: &PathBuf,
    base_name: &str,
    query_name: &str,
) -> Result<(), GfaDiffError> {
    let query_block_groups = Sample::get_block_groups(conn, collection_name, query_name, None);
    let base_block_groups = Sample::get_block_groups(conn, collection_name, base_name, None);

    let query_paths_by_name = query_block_groups
        .iter()
        .map(|block_group| {
            Ok((
                block_group.name.clone(),
                BlockGroup::get_current_path(conn, &block_group.id, None)?,
            ))
        })
        .collect::<Result<HashMap<String, Path>, BlockGroupError>>()?;
    let base_paths_by_name = base_block_groups
        .iter()
        .map(|block_group| {
            Ok((
                block_group.name.clone(),
                BlockGroup::get_current_path(conn, &block_group.id, None)?,
            ))
        })
        .collect::<Result<HashMap<String, Path>, BlockGroupError>>()?;
    let query_block_groups_by_name = query_block_groups
        .iter()
        .map(|block_group| (block_group.name.as_str(), block_group))
        .collect::<HashMap<_, _>>();
    let base_block_groups_by_name = base_block_groups
        .iter()
        .map(|block_group| (block_group.name.as_str(), block_group))
        .collect::<HashMap<_, _>>();

    let mut segments = HashSet::new();
    let mut links = vec![];
    let mut paths = vec![];

    let path_names = query_paths_by_name
        .keys()
        .chain(base_paths_by_name.keys())
        .cloned()
        .collect::<BTreeSet<_>>();

    for path_name in path_names {
        let query_block_group = query_block_groups_by_name.get(path_name.as_str()).copied();
        let base_block_group = base_block_groups_by_name.get(path_name.as_str()).copied();
        let graph = build_sample_block_group_diff(conn, base_block_group, query_block_group, None);
        let graph = terminal_walk_graph(&graph);
        let graph_segments = segments_from_graph(conn, workspace, &graph)?;
        segments.extend(graph_segments.iter().cloned());
        links.extend(links_from_graph(&graph));

        if let Some(path) = query_paths_by_name.get(&path_name) {
            paths.push(path_from_graph(conn, query_name, path, &graph_segments)?);
        }
        if let Some(path) = base_paths_by_name.get(&path_name) {
            paths.push(path_from_graph(conn, base_name, path, &graph_segments)?);
        }
    }

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    write_segments(&mut writer, &segments.iter().collect::<Vec<&Segment>>())?;
    write_links(&mut writer, &links.iter().collect::<Vec<&Link>>())?;

    for path in paths {
        writer.write_all(&path_line(&path).into_bytes())?;
    }
    writer.flush()?;

    Ok(())
}

/// Retains only normalized topology that belongs to a complete stored walk.
///
/// The shared graph remains lossless for general diff consumers. GFA has no
/// terminal segments, so exporting an orphan fragment would cause importers to
/// infer a new root path. Restricting only this export projection to nodes both
/// reachable from a start terminal and able to reach an end terminal avoids
/// that invented path while preserving all complete alternatives.
fn terminal_walk_graph(graph: &DiffGenGraph) -> DiffGenGraph {
    let mut graph = graph.clone();
    let start = graph.nodes().find(|node| is_start_node(node.node.node_id));
    let end = graph.nodes().find(|node| is_end_node(node.node.node_id));
    let (Some(start), Some(end)) = (start, end) else {
        return graph;
    };
    let from_start = reachable_nodes(&graph, start, Direction::Outgoing);
    let to_end = reachable_nodes(&graph, end, Direction::Incoming);
    let orphaned_nodes = graph
        .nodes()
        .filter(|node| !from_start.contains(node) || !to_end.contains(node))
        .collect::<Vec<_>>();
    for node in orphaned_nodes {
        graph.remove_node(node);
    }
    graph
}

fn reachable_nodes(
    graph: &DiffGenGraph,
    start: gen_diff::graph::DiffGraphNode,
    direction: Direction,
) -> HashSet<gen_diff::graph::DiffGraphNode> {
    let mut reachable = HashSet::from([start]);
    let mut pending = vec![start];
    while let Some(node) = pending.pop() {
        for neighbor in graph.neighbors_directed(node, direction) {
            if reachable.insert(neighbor) {
                pending.push(neighbor);
            }
        }
    }
    reachable
}

fn segments_from_graph(
    conn: &GraphConnection,
    workspace: &Workspace,
    graph: &DiffGenGraph,
) -> Result<Vec<Segment>, GfaDiffError> {
    let nodes = graph
        .nodes()
        .map(|node| node.node)
        .filter(|node| !is_start_node(node.node_id) && !is_end_node(node.node_id))
        .collect::<Vec<_>>();
    let node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let sequences = Node::get_sequences_by_node_ids(conn, workspace, &node_ids, None);
    nodes
        .into_iter()
        .map(|node| {
            let sequence = sequences
                .get(&node.node_id)
                .expect("should load every normalized graph node sequence")
                .get_sequence(node.sequence_start, node.sequence_end)?;
            Ok(Segment {
                sequence,
                node_id: node.node_id,
                sequence_start: node.sequence_start,
                sequence_end: node.sequence_end,
                strand: gen_core::Strand::Forward,
            })
        })
        .collect::<Result<Vec<_>, gen_models::sequence::SequenceError>>()
        .map_err(PathError::from)
        .map_err(GfaDiffError::from)
}

fn links_from_graph(graph: &DiffGenGraph) -> Vec<Link> {
    graph
        .all_edges()
        .flat_map(|(source, target, edges)| {
            if is_start_node(source.node.node_id)
                || is_end_node(source.node.node_id)
                || is_start_node(target.node.node_id)
                || is_end_node(target.node.node_id)
            {
                return Vec::new();
            }
            edges
                .iter()
                .map(|edge| Link {
                    source_segment_id: segment_id(source.node),
                    source_strand: edge.edge.source_strand,
                    target_segment_id: segment_id(target.node),
                    target_strand: edge.edge.target_strand,
                })
                .collect()
        })
        .collect()
}

fn path_from_graph(
    conn: &GraphConnection,
    sample_name: &str,
    path: &Path,
    segments: &[Segment],
) -> Result<GFAPath, GfaDiffError> {
    let mut segments_by_node_id = HashMap::new();
    for segment in segments {
        segments_by_node_id
            .entry(segment.node_id)
            .or_insert_with(Vec::new)
            .push(segment);
    }
    let mut segment_ids = vec![];
    let mut node_strands = vec![];
    for block in path.coordinate_blocks(conn, None) {
        if is_start_node(block.node_id) || is_end_node(block.node_id) {
            continue;
        }
        let mut matching_segments = segments_by_node_id
            .get(&block.node_id)
            .into_iter()
            .flatten()
            .filter(|segment| {
                segment.sequence_start >= block.sequence_start
                    && segment.sequence_end <= block.sequence_end
            })
            .collect::<Vec<_>>();
        if matching_segments.is_empty() {
            return Err(PathError::Missing(format!(
                "No normalized graph segments cover path {} block {}[{}..{}]",
                path.id, block.node_id, block.sequence_start, block.sequence_end
            ))
            .into());
        }
        matching_segments.sort_by_key(|segment| (segment.sequence_start, segment.sequence_end));
        let mut expected_start = block.sequence_start;
        for segment in &matching_segments {
            if segment.sequence_start != expected_start {
                return Err(PathError::Missing(format!(
                    "Normalized graph segments do not cover path {} block {}[{}..{}]",
                    path.id, block.node_id, block.sequence_start, block.sequence_end
                ))
                .into());
            }
            expected_start = segment.sequence_end;
        }
        if expected_start != block.sequence_end {
            return Err(PathError::Missing(format!(
                "Normalized graph segments do not cover path {} block {}[{}..{}]",
                path.id, block.node_id, block.sequence_start, block.sequence_end
            ))
            .into());
        }
        if block.strand == gen_core::Strand::Reverse {
            matching_segments.reverse();
        }
        segment_ids.extend(matching_segments.iter().map(|segment| segment.segment_id()));
        node_strands.extend(std::iter::repeat_n(block.strand, matching_segments.len()));
    }
    Ok(GFAPath {
        name: format!("{sample_name}.{}", path.name),
        segment_ids,
        node_strands,
    })
}

fn segment_id(node: gen_graph::GraphNode) -> String {
    format!(
        "{}.{}.{}",
        node.node_id, node.sequence_start, node.sequence_end
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

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
    };

    #[test]
    fn test_gfa_diff_reference_frame() {
        // Assert that the comparison is in the right direction. Normally GFA doesn't annotate
        // removed/added, so the order of inputs doesn't matter as the graph has no indication
        // of what segment something belongs to. This comes out in paths, so we assert that the
        // paths are correct based on the input they should belong to.
        // Only the child's path replaces the middle AA with CC:
        //   base:  A -> AA -> A
        //   child: A -> CC -> A
        let context = setup_gen();
        let conn = context.graph().conn();
        let collection_name = "argument order";
        Collection::create(conn, collection_name).unwrap();
        let block_group = create_bg(conn, collection_name, Sample::DEFAULT_NAME, "sequence");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAA")
            .save(conn)
            .unwrap();
        let node_id =
            Node::create(conn, &sequence.hash, &HashId::convert_str("argument-order")).unwrap();
        let entry = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let exit = Edge::create(
            conn,
            node_id,
            4,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge_ids = [entry.id, exit.id];
        BlockGroupEdge::bulk_create(
            conn,
            &edge_ids
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        Path::create(conn, "sequence", &block_group.id, &edge_ids).unwrap();
        Sample::get_or_create_child(
            conn,
            collection_name,
            "child",
            vec![Sample::DEFAULT_NAME.to_string()],
        )
        .unwrap();
        let child_block_group =
            BlockGroup::get_by_name(conn, collection_name, "child", "sequence", None).unwrap();
        let child_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("CC")
            .save(conn)
            .unwrap();
        let child_node = Node::create(
            conn,
            &child_sequence.hash,
            &HashId::convert_str("child-only-edit"),
        )
        .unwrap();
        let edit_entry = Edge::create(
            conn,
            node_id,
            1,
            Strand::Forward,
            child_node,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edit_exit = Edge::create(
            conn,
            child_node,
            2,
            Strand::Forward,
            node_id,
            3,
            Strand::Forward,
        )
        .unwrap();
        BlockGroupEdge::bulk_create(
            conn,
            &[edit_entry.id, edit_exit.id]
                .iter()
                .map(|edge_id| BlockGroupEdgeData {
                    block_group_id: child_block_group.id,
                    edge_id: *edge_id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        let child_path = BlockGroup::get_current_path(conn, &child_block_group.id, None).unwrap();
        child_path
            .new_path_with(conn, 1, 3, &edit_entry, &edit_exit)
            .unwrap();
        let directory = tempdir().unwrap();
        let filename = directory.path().join("diff.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &filename,
            Sample::DEFAULT_NAME,
            "child",
        )
        .unwrap();
        let output = fs::read_to_string(&filename).unwrap();
        let segments = output
            .lines()
            .filter(|line| line.starts_with("S\t"))
            .map(|line| {
                let fields = line.split('\t').collect::<Vec<_>>();
                (fields[1], fields[2])
            })
            .collect::<HashMap<_, _>>();
        let path_sequences = output
            .lines()
            .filter(|line| line.starts_with("P\t"))
            .map(|line| {
                let fields = line.split('\t').collect::<Vec<_>>();
                let sequence = fields[2]
                    .split(',')
                    .map(|segment| {
                        let segment_id = segment
                            .strip_suffix('+')
                            .expect("should traverse forward in this fixture");
                        segments[segment_id]
                    })
                    .collect::<String>();
                (fields[1], sequence)
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(path_sequences.len(), 2);
        assert_eq!(path_sequences["Reference.sequence"], "AAAA");
        // Path updates give the child path a new name containing the edit site.
        let (_, child_sequence) = path_sequences
            .iter()
            .find(|(name, _)| name.starts_with("Child."))
            .expect("should export the child's updated path");
        assert_eq!(child_sequence, "ACCA");
    }

    #[test]
    fn test_gfa_diff() {
        // Sets up a basic graph and then exports it to a GFA file
        let context = setup_gen();
        let conn = context.graph().conn();

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

        let child_block_groups = Sample::get_block_groups(conn, collection_name, "child", None);
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
        let original_child_path =
            BlockGroup::get_current_path(conn, &child_block_group.id, None).unwrap();
        let _child_path = original_child_path.new_path_with(conn, 2, 6, &edge4, &edge5);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("parent-child-diff.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
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

        let new_child_block_group = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_child_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_child_block_group.id,
            false,
        )
        .unwrap();

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

        let grandchild_block_groups =
            Sample::get_block_groups(conn, collection_name, "grandchild", None);
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
            BlockGroup::get_current_path(conn, &grandchild_block_group.id, None).unwrap();
        let _grandchild_path = original_grandchild_path.new_path_with(conn, 10, 14, &edge6, &edge7);

        let gfa_path = temp_dir.path().join("parent-grandchild-diff.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
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
        let new_grandchild_block_group =
            Collection::get_block_groups(conn, "test collection 3", None)
                .pop()
                .unwrap();
        let all_grandchild_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_grandchild_block_group.id,
            false,
        )
        .unwrap();

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
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &gfa_path,
            "child",
            "grandchild",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 4",
            Sample::DEFAULT_NAME,
        );

        let new_grandchild_block_group =
            Collection::get_block_groups(conn, "test collection 4", None)
                .pop()
                .unwrap();
        let all_grandchild_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_grandchild_block_group.id,
            false,
        )
        .unwrap();

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
        // Confirm diff of a sample against nothing retains its complete graph.
        let context = setup_gen();
        let conn = context.graph().conn();

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
            context.workspace(),
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

        let new_block_group = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_block_group.id,
            false,
        )
        .unwrap();

        assert_eq!(
            all_sequences,
            ["AAAAAAAATTTTTTTT"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<String>>()
        );

        let reverse_gfa_path = temp_dir.path().join("diff-from-nothing.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &reverse_gfa_path,
            "test sample",
            Sample::DEFAULT_NAME,
        )
        .unwrap();
        let _ = import_gfa(
            &context,
            &reverse_gfa_path,
            "test collection 3",
            Sample::DEFAULT_NAME,
        );
        let reverse_block_group = Collection::get_block_groups(conn, "test collection 3", None)
            .pop()
            .unwrap();
        assert_eq!(
            BlockGroup::get_all_sequences(
                conn,
                crate::test_helpers::test_workspace(),
                &reverse_block_group.id,
                false,
            )
            .unwrap(),
            ["AAAAAAAATTTTTTTT"]
                .iter()
                .map(|sequence| sequence.to_string())
                .collect::<HashSet<String>>()
        );
    }

    #[test]
    fn test_gfa_diff_retains_alternative_outside_current_path() {
        // The named path uses A -> T, but the graph also allows A -> C -> T:
        //   start -> A ---------> T -> end
        //             \-> C ->/
        // Export C and its links even though neither P record mentions it.
        let context = setup_gen();
        let conn = context.graph().conn();
        let collection_name = "alternative collection";
        Collection::create(conn, collection_name).unwrap();
        let block_group = create_bg(conn, collection_name, Sample::DEFAULT_NAME, "alternatives");
        let mut node_ids = Vec::new();
        for bases in ["AAAA", "CCCC", "TTTT"] {
            let sequence = Sequence::new()
                .sequence_type("DNA")
                .sequence(bases)
                .save(conn)
                .unwrap();
            node_ids.push(Node::create(conn, &sequence.hash, &HashId::convert_str(bases)).unwrap());
        }
        let [node_a, node_c, node_t] = node_ids[..] else {
            panic!("should create three nodes")
        };
        let mut edges = Vec::new();
        for (source, coordinate, target) in [
            (PATH_START_NODE_ID, 0, node_a),
            (node_a, 4, node_t),
            (node_t, 4, PATH_END_NODE_ID),
            (node_a, 4, node_c),
            (node_c, 4, node_t),
        ] {
            edges.push(
                Edge::create(
                    conn,
                    source,
                    coordinate,
                    Strand::Forward,
                    target,
                    0,
                    Strand::Forward,
                )
                .unwrap(),
            );
        }
        BlockGroupEdge::bulk_create(
            conn,
            &edges
                .iter()
                .map(|edge| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id: edge.id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<_>>(),
        );
        Path::create(
            conn,
            "direct",
            &block_group.id,
            &edges[..3].iter().map(|edge| edge.id).collect::<Vec<_>>(),
        )
        .unwrap();
        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("alternative.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &gfa_path,
            Sample::DEFAULT_NAME,
            Sample::DEFAULT_NAME,
        )
        .unwrap();
        let gfa = fs::read_to_string(gfa_path).unwrap();
        assert!(
            gfa.lines()
                .any(|line| line == format!("S\t{node_c}.0.4\tCCCC"))
        );
        for (source, target) in [(node_a, node_c), (node_c, node_t)] {
            assert!(
                gfa.lines()
                    .any(|line| line == format!("L\t{source}.0.4\t+\t{target}.0.4\t+\t0M"))
            );
        }
        let paths = gfa
            .lines()
            .filter(|line| line.starts_with("P\t"))
            .collect::<Vec<_>>();
        assert!(!paths.is_empty());
        for path in paths {
            assert!(!path.contains(&node_c.to_string()));
            assert!(path.contains(&format!("{node_a}.0.4+,{node_t}.0.4+")));
        }
    }

    #[test]
    fn test_gfa_diff_reverse_path_uses_normalized_reverse_links() {
        // Shared normalization splits the query at coordinate 3. The base
        // traverses the same backing node in reverse, so its GFA walk must use
        // the high slice first and the reverse continuation back to the low slice.
        //
        //     query: [0..3]+ -> [3..8]+
        //     base:  [3..8]- -> [0..3]-
        let context = setup_gen();
        let conn = context.graph().conn();
        let collection_name = "reverse split collection";
        Collection::create(conn, collection_name).unwrap();
        let _base_sample = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "base",
                ..Default::default()
            },
        );
        let _query_sample = Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "query",
                ..Default::default()
            },
        );
        let base_block_group = create_bg(conn, collection_name, "base", "reverse split");
        let query_block_group = create_bg(conn, collection_name, "query", "reverse split");
        let sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AACCGGTA")
            .save(conn)
            .unwrap();
        let node_id =
            Node::create(conn, &sequence.hash, &HashId::convert_str("reverse node")).unwrap();

        let base_start = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Reverse,
        )
        .unwrap();
        let base_end = Edge::create(
            conn,
            node_id,
            8,
            Strand::Reverse,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();
        let query_start = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let query_split = Edge::create(
            conn,
            node_id,
            3,
            Strand::Forward,
            node_id,
            3,
            Strand::Forward,
        )
        .unwrap();
        let query_end = Edge::create(
            conn,
            node_id,
            8,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();
        let base_edge_ids = [base_start.id, base_end.id];
        let query_edge_ids = [query_start.id, query_split.id, query_end.id];
        for (block_group, edge_ids) in [
            (base_block_group, &base_edge_ids[..]),
            (query_block_group, &query_edge_ids[..]),
        ] {
            let memberships = edge_ids
                .iter()
                .map(|&edge_id| BlockGroupEdgeData {
                    block_group_id: block_group.id,
                    edge_id,
                    chromosome_index: NO_CHROMOSOME_INDEX,
                    phased: 0,
                })
                .collect::<Vec<_>>();
            BlockGroupEdge::bulk_create(conn, &memberships);
            Path::create(conn, "reverse split", &block_group.id, edge_ids).unwrap();
        }

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("reverse-split.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &gfa_path,
            "base",
            "query",
        )
        .unwrap();
        let gfa = fs::read_to_string(&gfa_path).unwrap();
        let base_path = gfa
            .lines()
            .find(|line| line.starts_with("P\tBase.reverse"))
            .unwrap_or_else(|| panic!("should export the base reverse path:\n{gfa}"));
        let fields = base_path.split('\t').collect::<Vec<_>>();
        let reverse_segments = fields[2].split(',').collect::<Vec<_>>();
        assert_eq!(reverse_segments.len(), 2);
        assert!(reverse_segments[0].ends_with(".3.8-"));
        assert!(reverse_segments[1].ends_with(".0.3-"));
        let high = reverse_segments[0].trim_end_matches('-');
        let low = reverse_segments[1].trim_end_matches('-');
        assert!(
            gfa.lines()
                .any(|line| line == format!("L\t{high}\t-\t{low}\t-\t0M")),
            "the reverse path must retain its descending continuation link"
        );
        assert!(
            gfa.lines()
                .any(|line| line == format!("L\t{low}\t+\t{high}\t+\t0M")),
            "the query path must retain its ascending continuation link"
        );
        let segments_by_id = gfa
            .lines()
            .filter_map(|line| {
                let fields = line.split('\t').collect::<Vec<_>>();
                (fields.first() == Some(&"S")).then(|| (fields[1], fields[2]))
            })
            .collect::<HashMap<_, _>>();
        let reconstructed_base = reverse_segments
            .iter()
            .map(|segment| {
                segments_by_id[segment.trim_end_matches('-')]
                    .chars()
                    .rev()
                    .map(|base| match base {
                        'A' => 'T',
                        'C' => 'G',
                        'G' => 'C',
                        'T' => 'A',
                        _ => base,
                    })
                    .collect::<String>()
            })
            .collect::<String>();
        assert_eq!(reconstructed_base, "TACCGGTT");
    }

    #[test]
    fn test_self_diff() {
        // Confirm diff of a sample to itself just results in a graph that's a single path
        let context = setup_gen();
        let conn = context.graph().conn();

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
            context.workspace(),
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

        let new_block_group = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_block_group.id,
            false,
        )
        .unwrap();

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
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &gfa_path,
            "sample1",
            "sample2",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 3",
            Sample::DEFAULT_NAME,
        );

        let new_block_group = Collection::get_block_groups(conn, "test collection 3", None)
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_block_group.id,
            false,
        )
        .unwrap();

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
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &gfa_path,
            "sample1",
            "sample2",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 3",
            Sample::DEFAULT_NAME,
        );

        let new_block_group = Collection::get_block_groups(conn, "test collection 3", None)
            .pop()
            .unwrap();
        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_block_group.id,
            false,
        )
        .unwrap();

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

        let child_block_groups = Sample::get_block_groups(conn, collection_name, "child", None);
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
        let original_child_path =
            BlockGroup::get_current_path(conn, &child_block_group.id, None).unwrap();
        let _child_path = original_child_path.new_path_with(conn, 2, 6, &edge3, &edge4);

        let temp_dir = tempdir().unwrap();
        let gfa_path = temp_dir.path().join("parent-child-diff.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
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

        let new_child_block_group = Collection::get_block_groups(conn, "test collection 2", None)
            .pop()
            .unwrap();
        let all_child_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_child_block_group.id,
            false,
        )
        .unwrap();

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

        let grandchild_block_groups =
            Sample::get_block_groups(conn, collection_name, "grandchild", None);
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
            BlockGroup::get_current_path(conn, &grandchild_block_group.id, None).unwrap();
        let _grandchild_path = original_grandchild_path.new_path_with(conn, 4, 10, &edge5, &edge6);

        let gfa_path = temp_dir.path().join("parent-grandchild-diff.gfa");
        gfa_sample_diff(
            conn,
            context.workspace(),
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

        let new_grandchild_block_group =
            Collection::get_block_groups(conn, "test collection 3", None)
                .pop()
                .unwrap();
        let all_grandchild_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_grandchild_block_group.id,
            false,
        )
        .unwrap();

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
        gfa_sample_diff(
            conn,
            context.workspace(),
            collection_name,
            &gfa_path,
            "child",
            "grandchild",
        )
        .unwrap();

        let _ = import_gfa(
            &context,
            &gfa_path,
            "test collection 4",
            Sample::DEFAULT_NAME,
        );

        let new_grandchild_block_group =
            Collection::get_block_groups(conn, "test collection 4", None)
                .pop()
                .unwrap();
        let all_grandchild_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &new_grandchild_block_group.id,
            false,
        )
        .unwrap();

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
