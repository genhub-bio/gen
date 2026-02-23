use gen_core::{HashId, INDETERMINATE_CHROMOSOME_INDEX, Strand};
use gen_models::{
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::GraphConnection,
    edge::{Edge, EdgeData},
    path_edge::PathEdge,
    traits::Query,
};
use thiserror::Error;

pub mod combinatorial_library;
pub mod operators;

// A NodePoint is grouping of the fields that an edge has as a source or a
// target.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NodePoint {
    pub id: HashId,
    pub coordinate: i64,
    pub strand: Strand,
}

// A BlockGroupChunk is basically an in-memory BlockGroup but with the start and
// end nodes deleted.  This makes it possible to stitch two BlockGroupChunks
// together fairly easily.  The entry node points are what edges from the start
// node pointed to, and the exit node points are the sources of edges that point
// to the end node.  For paths it's a bit more complicated.  If the block group
// has a path, the path is defined by a start node point (instead of an edge
// from the start node to a node point), followed by a set of "internal" edges
// that form the path, and then an end node point instead of an edge to the end
// node.  The path start point must be in the entry node points, and the path
// end point must be in the exit node points.
//
// So to stitch chunk A (upstream) to chunk B (downstream), we create edges
// between all the exit node points of A and all the entry node points of B.
// The new path is defined by the path start point of A, then all the path edges
// of A, then the new edge from the path end point of A to the path start point
// of B, then the path edges of B, then the path end point of B.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockGroupChunk {
    pub entry_node_points: Vec<NodePoint>,
    pub exit_node_points: Vec<NodePoint>,
    pub path_edges: Vec<Edge>,
    pub path_start_point: Option<NodePoint>,
    pub path_end_point: Option<NodePoint>,
}

#[derive(Debug, Error, PartialEq)]
pub enum GraphError {
    #[error("Edge error: {0}")]
    StitchEdgeNotFound(String),
}

pub fn load_block_group_chunk(conn: &GraphConnection, block_group_id: HashId) -> BlockGroupChunk {
    let edges = BlockGroupEdge::edges_for_block_group(conn, &block_group_id);

    let start_edges = edges.iter().filter(|edge| edge.edge.is_start_edge());

    let entry_node_points = start_edges
        .map(|start_edge| NodePoint {
            id: start_edge.edge.target_node_id,
            coordinate: start_edge.edge.target_coordinate,
            strand: start_edge.edge.target_strand,
        })
        .collect();

    let end_edges = edges.iter().filter(|edge| edge.edge.is_end_edge());

    let exit_node_points = end_edges
        .map(|edge| NodePoint {
            id: edge.edge.source_node_id,
            coordinate: edge.edge.source_coordinate,
            strand: edge.edge.source_strand,
        })
        .collect();

    let path = BlockGroup::get_current_path(conn, &block_group_id);
    let path_edges = PathEdge::edges_for_path(conn, &path.id);

    let start_edge = path_edges[0].clone();
    let path_start_point = Some(NodePoint {
        id: start_edge.target_node_id,
        coordinate: start_edge.target_coordinate,
        strand: start_edge.target_strand,
    });

    let end_edge = path_edges[path_edges.len() - 1].clone();
    let path_end_point = Some(NodePoint {
        id: end_edge.source_node_id,
        coordinate: end_edge.source_coordinate,
        strand: end_edge.source_strand,
    });

    let path_edges = path_edges[1..path_edges.len() - 1].to_vec();

    BlockGroupChunk {
        entry_node_points,
        exit_node_points,
        path_edges,
        path_start_point,
        path_end_point,
    }
}

/// Stitches two block group chunks together.  Creates edges between exit node
/// points from one chunk and entry node points of the target chunk.  For those
/// edges, also creates block group edges for the given block group ID.
/// Finally, if both chunks have paths, stitches those together.  Returns the
/// resulting block group chunk.
pub fn stitch(
    conn: &GraphConnection,
    source_block_group_chunk: &BlockGroupChunk,
    target_block_group_chunk: &BlockGroupChunk,
    block_group_id: HashId,
) -> Result<BlockGroupChunk, GraphError> {
    let mut edges = vec![];

    let source_node_points = &source_block_group_chunk.exit_node_points;
    let target_node_points = &target_block_group_chunk.entry_node_points;

    // Create edges between source nodes and target nodes
    for source_point in source_node_points {
        for target_point in target_node_points {
            edges.push(EdgeData {
                source_node_id: source_point.id,
                source_coordinate: source_point.coordinate,
                source_strand: source_point.strand,
                target_node_id: target_point.id,
                target_coordinate: target_point.coordinate,
                target_strand: target_point.strand,
            });
        }
    }

    let edge_ids: Vec<HashId> = Edge::bulk_create(conn, &edges);

    let block_group_edges = edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id,
            edge_id: *edge_id,
            chromosome_index: INDETERMINATE_CHROMOSOME_INDEX,
            phased: 0,
        })
        .collect::<Vec<BlockGroupEdgeData>>();

    BlockGroupEdge::bulk_create(conn, &block_group_edges);

    let source_path_edges = source_block_group_chunk.path_edges.clone();
    let target_path_edges = target_block_group_chunk.path_edges.clone();

    let mut path_edges = vec![];
    if let Some(edge_start_point) = &source_block_group_chunk.path_end_point
        && let Some(edge_end_point) = &target_block_group_chunk.path_start_point
    {
        path_edges.extend(source_path_edges.clone());

        let created_edges = Edge::query_by_ids(conn, &edge_ids);
        let stitch_edge = created_edges.iter().find(|edge| {
            edge.source_node_id == edge_start_point.id
                && edge.source_coordinate == edge_start_point.coordinate
                && edge.source_strand == edge_start_point.strand
                && edge.target_node_id == edge_end_point.id
                && edge.target_coordinate == edge_end_point.coordinate
                && edge.target_strand == edge_end_point.strand
        });
        if let Some(stitch_edge) = stitch_edge {
            path_edges.push(stitch_edge.clone());
        } else {
            return Err(GraphError::StitchEdgeNotFound(
                "Couldn't find stitch edge in edges that were just created!".to_string(),
            ));
        }

        path_edges.extend(target_path_edges);
    }

    Ok(BlockGroupChunk {
        entry_node_points: source_block_group_chunk.entry_node_points.clone(),
        exit_node_points: target_block_group_chunk.exit_node_points.clone(),
        path_edges,
        path_start_point: source_block_group_chunk.path_start_point.clone(),
        path_end_point: target_block_group_chunk.path_end_point.clone(),
    })
}
