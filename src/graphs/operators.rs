use core::ops::Range;
use std::collections::{HashMap, HashSet};

use gen_core::{
    HashId, NodeIntervalBlock, PATH_END_NODE_ID, PATH_START_NODE_ID, Strand, Workspace,
    is_end_node, is_start_node,
};
use gen_graph::GraphNode;
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup, SubgraphBoundary},
    block_group_edge::{AugmentedEdge, BlockGroupEdge, BlockGroupEdgeData},
    db::{DbContext, GraphConnection},
    edge::{Edge, EdgeError},
    errors::{BlockGroupError, OperationError, PathError},
    path::Path,
    region::{Region, resolve},
    sample::Sample,
};
use petgraph::algo::{is_cyclic_directed, kosaraju_scc};
use thiserror::Error;

use crate::graphs::{BlockGroupChunk, GraphError, NodePoint, load_block_group_chunk, stitch};

#[derive(Debug, Error, PartialEq)]
pub enum GraphOperationError {
    #[error("Operation Error: {0}")]
    OperationError(#[from] OperationError),
    #[error("Invalid coordinate(s): {0}")]
    InvalidCoordinate(String),
    #[error("Region not found: {0}")]
    RegionNotFound(String),
    #[error("Path not found: {0}")]
    PathNotFound(String),
    #[error("Graph error: {0}")]
    GraphError(#[from] GraphError),
    #[error("Path creation error: {0}")]
    PathError(#[from] PathError),
    #[error("Block group creation error: {0}")]
    BlockGroupError(#[from] BlockGroupError),
    #[error("Invalid stitch input: {0}")]
    InvalidStitchInput(String),
    #[error("Stitched block group contains a cycle: {0}")]
    StitchedGraphCycle(String),
    #[error("Edge creation error: {0}")]
    EdgeError(#[from] EdgeError),
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

pub fn get_path(
    conn: &GraphConnection,
    collection_name: &str,
    sample_name: &str,
    region_name: &str,
    backbone: Option<&str>,
) -> Result<Path, GraphOperationError> {
    let resolved_region = Region::parse(region_name)
        .map_err(|err| GraphOperationError::RegionNotFound(err.to_string()))?;
    let resolved_region = resolve(&resolved_region, conn, collection_name, sample_name)
        .map_err(|err| GraphOperationError::RegionNotFound(err.to_string()))?;
    let block_group_id = resolved_region.block_group.id;

    if let Some(backbone) = backbone {
        let path = BlockGroup::get_path_by_name(conn, &block_group_id, backbone)?;
        if path.is_none() {
            return Err(GraphOperationError::PathNotFound(format!(
                "No path found with name {backbone}"
            )));
        }
        Ok(path.unwrap())
    } else {
        match resolved_region.path {
            Some(path) => Ok(path),
            None => Ok(BlockGroup::get_current_path(conn, &block_group_id, None)?),
        }
    }
}

/// Given a path (default or specified by backbone) and a sample, splits the default block group of that sample into
/// multiple chunks specified by chunk_ranges occurring along the path.
///
/// We currently assume each chunk boundary creates a partition in the graph.  To put
/// it another way, we assume each boundary is on an edge that is the only one connecting the upstream part of the graph
/// to the downstream part.  TODO: Add guardrails that confirm this assumption.
///
/// The resulting new "chunk" block groups are created in the new sample.
#[allow(clippy::too_many_arguments)]
pub fn derive_chunks(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_name: &str,
    backbone: Option<&str>,
    chunk_ranges: Vec<Range<i64>>,
    child_block_group_id: Option<HashId>,
    create_block_group: bool,
) -> Result<Vec<BlockGroupChunk>, GraphOperationError> {
    let conn = context.graph().conn();
    let _new_sample = Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: new_sample_name,
            ..Default::default()
        },
    );

    let parent_block_group_id =
        get_block_group_id(conn, collection_name, parent_sample_name, region_name)?;
    let current_path = get_path(
        conn,
        collection_name,
        parent_sample_name,
        region_name,
        backbone,
    )?;

    let current_path_length = current_path.length(conn, None)?;

    let current_intervaltree = current_path.intervaltree(conn)?;
    let current_path_edges = Path::edges_for_path(conn, &current_path.id, None);

    let chunk_ranges_length = chunk_ranges.len();

    let mut block_group_chunks = vec![];

    for (i, chunk_range) in chunk_ranges.clone().into_iter().enumerate() {
        let child_block_group_id = if let Some(child_block_group_id) = child_block_group_id {
            child_block_group_id
        } else {
            let child_block_group_name = if chunk_ranges_length > 1 {
                format!("{}.{}", region_name, i + 1)
            } else {
                region_name.to_string()
            };

            let child_block_group = BlockGroup::create(
                conn,
                NewBlockGroup {
                    collection_name,
                    sample_name: new_sample_name,
                    name: child_block_group_name.as_str(),
                    parent_block_group_id: Some(&parent_block_group_id),
                    ..Default::default()
                },
            )?;
            child_block_group.id
        };

        let start_coordinate = chunk_range.start;
        let end_coordinate = chunk_range.end;
        if (start_coordinate < 0 || start_coordinate > current_path_length)
            || (end_coordinate < 0 || end_coordinate > current_path_length)
        {
            return Err(GraphOperationError::InvalidCoordinate(format!(
                "Start and/or end coordinates ({start_coordinate}, {end_coordinate}) are out of range for the current path."
            )));
        }

        let mut blocks = current_intervaltree
            .query(Range {
                start: start_coordinate,
                end: end_coordinate,
            })
            .map(|x| x.value)
            .collect::<Vec<_>>();
        blocks.sort_by_key(|a| a.start);
        let start_block = blocks[0];
        let start_node_coordinate =
            start_coordinate - start_block.start + start_block.sequence_start;
        let end_block = blocks[blocks.len() - 1];
        let end_node_coordinate = end_coordinate - end_block.start + end_block.sequence_start;

        BlockGroup::derive_subgraph(
            conn,
            context.workspace(),
            &parent_block_group_id,
            SubgraphBoundary {
                block: &start_block,
                sequence_coordinate: start_node_coordinate,
            },
            SubgraphBoundary {
                block: &end_block,
                sequence_coordinate: end_node_coordinate,
            },
            &child_block_group_id,
            create_block_group,
        )?;

        let child_block_group_edges =
            BlockGroupEdge::edges_for_block_group(conn, &child_block_group_id, None);

        let child_edge_ids_by_key = child_block_group_edges
            .iter()
            .map(|augmented_edge| {
                let edge = &augmented_edge.edge;
                (
                    (
                        edge.source_node_id,
                        edge.source_coordinate,
                        edge.source_strand,
                        edge.target_node_id,
                        edge.target_coordinate,
                        edge.target_strand,
                    ),
                    edge.id,
                )
            })
            .collect::<HashMap<_, _>>();

        let mut new_path_edge_ids = vec![];

        let start_node_point = NodePoint {
            id: start_block.node_id,
            coordinate: start_node_coordinate,
            strand: Strand::Forward,
        };

        if create_block_group {
            let new_start_edge = child_block_group_edges
                .iter()
                .find(|e| {
                    is_start_node(e.edge.source_node_id)
                        && e.edge.target_node_id == start_block.node_id
                        && e.edge.target_coordinate == start_node_coordinate
                })
                .unwrap();
            new_path_edge_ids.push(new_start_edge.edge.id);
        }

        for edge in &current_path_edges {
            if is_start_node(edge.source_node_id) || is_end_node(edge.target_node_id) {
                continue;
            }

            let key = &(
                edge.source_node_id,
                edge.source_coordinate,
                edge.source_strand,
                edge.target_node_id,
                edge.target_coordinate,
                edge.target_strand,
            );
            let child_edge_id = child_edge_ids_by_key.get(key);
            if let Some(child_edge_id) = child_edge_id {
                new_path_edge_ids.push(*child_edge_id);
            }
        }

        let end_node_point = NodePoint {
            id: end_block.node_id,
            coordinate: end_node_coordinate,
            strand: Strand::Forward,
        };

        if create_block_group {
            let new_end_edge = child_block_group_edges
                .iter()
                .find(|e| {
                    is_end_node(e.edge.target_node_id)
                        && e.edge.source_node_id == end_block.node_id
                        && e.edge.source_coordinate == end_node_coordinate
                })
                .unwrap();
            new_path_edge_ids.push(new_end_edge.edge.id);

            let _path = Path::create(
                conn,
                &current_path.name,
                &child_block_group_id,
                &new_path_edge_ids,
            )?;
        }

        let path_edges = Edge::select(conn)
            .query_by_ids(new_path_edge_ids)
            .expect("should load path edges by id");

        block_group_chunks.push(BlockGroupChunk {
            entry_node_points: vec![start_node_point.clone()],
            exit_node_points: vec![end_node_point.clone()],
            path_edges,
            path_start_point: Some(start_node_point.clone()),
            path_end_point: Some(end_node_point.clone()),
        });
    }

    Ok(block_group_chunks)
}

fn get_block_group_id(
    conn: &GraphConnection,
    collection_name: &str,
    parent_sample_name: &str,
    region_name: &str,
) -> Result<HashId, GraphOperationError> {
    let resolved_region = Region::parse(region_name)
        .map_err(|err| GraphOperationError::RegionNotFound(err.to_string()))?;
    resolve(&resolved_region, conn, collection_name, parent_sample_name)
        .map(|resolved| resolved.block_group.id)
        .map_err(|err| GraphOperationError::RegionNotFound(err.to_string()))
}

/// Given a sample and one or more region (block group) names, creates a new graph where all the end nodes of one block
/// group are connected to all the start nodes of the next block group.  Saves the result as a block group with
/// new_region_name in a new sample with the specified name.
pub fn make_stitch(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_names: &Vec<&str>,
    new_region_name: &str,
) -> Result<(), GraphOperationError> {
    let conn = context.graph().conn();

    let stitch_inputs = stitch_inputs(conn, collection_name, parent_sample_name, region_names)?;
    validate_stitch_inputs(&stitch_inputs)?;
    let block_group_chunks = stitch_inputs
        .iter()
        .map(|input| input.chunk.clone())
        .collect::<Vec<_>>();

    conn.execute_batch("SAVEPOINT make_stitch")?;
    let result = create_stitched_block_group(
        context,
        collection_name,
        new_sample_name,
        new_region_name,
        &stitch_inputs,
        &block_group_chunks,
    );
    match result {
        Ok(()) => {
            conn.execute_batch("RELEASE make_stitch")?;
            Ok(())
        }
        Err(err) => {
            conn.execute_batch("ROLLBACK TO make_stitch")?;
            conn.execute_batch("RELEASE make_stitch")?;
            Err(err)
        }
    }
}

struct StitchInput<'a> {
    region_name: &'a str,
    block_group_id: HashId,
    chunk: BlockGroupChunk,
    nonterminal_edges: Vec<AugmentedEdge>,
}

fn stitch_inputs<'a>(
    conn: &GraphConnection,
    collection_name: &str,
    parent_sample_name: &str,
    region_names: &'a Vec<&'a str>,
) -> Result<Vec<StitchInput<'a>>, GraphOperationError> {
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name, None);

    let mut block_groups_by_name = HashMap::new();
    for block_group in &block_groups {
        let block_group_name = block_group.name.as_str();
        if region_names.contains(&block_group_name) {
            block_groups_by_name.insert(block_group_name, block_group.clone());
        }
    }

    let mut stitch_inputs = vec![];
    for region_name in region_names {
        if let Some(block_group) = block_groups_by_name.get(region_name) {
            let chunk = load_block_group_chunk(conn, block_group.id);
            let edges = BlockGroupEdge::edges_for_block_group(conn, &block_group.id, None);
            let nonterminal_edges = edges
                .into_iter()
                .filter(|edge| !edge.edge.is_start_edge() && !edge.edge.is_end_edge())
                .collect();
            stitch_inputs.push(StitchInput {
                region_name,
                block_group_id: block_group.id,
                chunk,
                nonterminal_edges,
            });
        } else {
            return Err(GraphOperationError::RegionNotFound(format!(
                "No region found with name: {region_name}"
            )));
        }
    }

    Ok(stitch_inputs)
}

/// Given a list of block groups to stitch together, confirms there are no
/// duplicate block groups (which would generate a cycle in the graph) or edges
/// that are shared between block groups
fn validate_stitch_inputs(stitch_inputs: &[StitchInput<'_>]) -> Result<(), GraphOperationError> {
    let mut seen_block_group_ids = HashMap::<HashId, &str>::new();
    let mut seen_edge_ids = HashMap::<HashId, &str>::new();
    for input in stitch_inputs {
        if let Some(previous_region_name) =
            seen_block_group_ids.insert(input.block_group_id, input.region_name)
        {
            return Err(GraphOperationError::InvalidStitchInput(format!(
                "Regions {previous_region_name} and {} refer to the same block group",
                input.region_name
            )));
        }

        for edge in &input.nonterminal_edges {
            if let Some(previous_region_name) =
                seen_edge_ids.insert(edge.edge.id, input.region_name)
            {
                return Err(GraphOperationError::InvalidStitchInput(format!(
                    "Regions {previous_region_name} and {} share edge {}",
                    input.region_name, edge.edge.id
                )));
            }
        }
    }

    Ok(())
}

/// A point on a node, in the node's own sequence coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubgraphPoint {
    pub node_id: HashId,
    pub coordinate: i64,
}

/// Creates a block group named like `source_block_group_id` in `new_sample_name` that holds every
/// route of the source between `start` and `end`, which need no linear coordinates.
///
/// The new block group gets a current path when the source's own current path runs from `start`
/// to `end`; otherwise it has none, since no single route is the obvious one.
pub fn derive_subgraph_between(
    context: &DbContext,
    source_block_group_id: &HashId,
    new_sample_name: &str,
    start: SubgraphPoint,
    end: SubgraphPoint,
) -> Result<BlockGroup, GraphOperationError> {
    let conn = context.graph().conn();
    conn.execute_batch("SAVEPOINT derive_subgraph_between")?;
    let result =
        create_subgraph_between(context, source_block_group_id, new_sample_name, start, end);
    match result {
        Ok(block_group) => {
            conn.execute_batch("RELEASE derive_subgraph_between")?;
            Ok(block_group)
        }
        Err(err) => {
            conn.execute_batch("ROLLBACK TO derive_subgraph_between")?;
            conn.execute_batch("RELEASE derive_subgraph_between")?;
            Err(err)
        }
    }
}

fn create_subgraph_between(
    context: &DbContext,
    source_block_group_id: &HashId,
    new_sample_name: &str,
    start: SubgraphPoint,
    end: SubgraphPoint,
) -> Result<BlockGroup, GraphOperationError> {
    let conn = context.graph().conn();
    let source = BlockGroup::get_by_id(conn, source_block_group_id, None)?;
    let _new_sample = Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: new_sample_name,
            ..Default::default()
        },
    );
    let child = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name: &source.collection_name,
            sample_name: new_sample_name,
            name: &source.name,
            parent_block_group_id: Some(source_block_group_id),
            ..Default::default()
        },
    )?;

    let boundary = |point: &SubgraphPoint| NodeIntervalBlock {
        node_id: point.node_id,
        start: 0,
        end: 0,
        sequence_start: point.coordinate,
        sequence_end: point.coordinate,
        strand: Strand::Forward,
    };
    let start_block = boundary(&start);
    let end_block = boundary(&end);
    BlockGroup::derive_subgraph(
        conn,
        context.workspace(),
        source_block_group_id,
        SubgraphBoundary {
            block: &start_block,
            sequence_coordinate: start.coordinate,
        },
        SubgraphBoundary {
            block: &end_block,
            sequence_coordinate: end.coordinate,
        },
        &child.id,
        true,
    )?;

    if let Ok(current_path) = BlockGroup::get_current_path(conn, source_block_group_id, None) {
        let child_edges = BlockGroupEdge::edges_for_block_group(conn, &child.id, None);
        let child_edge_ids = child_edges
            .iter()
            .map(|edge| edge.edge.id)
            .collect::<HashSet<_>>();
        let internal = Path::edges_for_path(conn, &current_path.id, None)
            .into_iter()
            .filter(|edge| {
                !is_start_node(edge.source_node_id)
                    && !is_end_node(edge.target_node_id)
                    && child_edge_ids.contains(&edge.id)
            })
            .collect::<Vec<_>>();
        let chained = internal
            .windows(2)
            .all(|pair| pair[0].target_node_id == pair[1].source_node_id);
        let begins_at_start = internal
            .first()
            .map_or(start.node_id == end.node_id, |edge| {
                edge.source_node_id == start.node_id && edge.source_coordinate >= start.coordinate
            });
        let ends_at_end = internal
            .last()
            .map_or(start.node_id == end.node_id, |edge| {
                edge.target_node_id == end.node_id && edge.target_coordinate <= end.coordinate
            });
        let start_edge = child_edges.iter().find(|edge| {
            is_start_node(edge.edge.source_node_id)
                && edge.edge.target_node_id == start.node_id
                && edge.edge.target_coordinate == start.coordinate
        });
        let end_edge = child_edges.iter().find(|edge| {
            is_end_node(edge.edge.target_node_id)
                && edge.edge.source_node_id == end.node_id
                && edge.edge.source_coordinate == end.coordinate
        });
        if chained
            && begins_at_start
            && ends_at_end
            && let (Some(start_edge), Some(end_edge)) = (start_edge, end_edge)
        {
            let edge_ids = core::iter::once(start_edge.edge.id)
                .chain(internal.iter().map(|edge| edge.id))
                .chain(core::iter::once(end_edge.edge.id))
                .collect::<Vec<_>>();
            Path::create(conn, &current_path.name, &child.id, &edge_ids)?;
        }
    }
    Ok(child)
}

/// A node-coordinate span of a forward-strand locus, as read in order along the locus.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StitchRange {
    pub node_id: HashId,
    pub start: i64,
    pub end: i64,
}

/// One piece of a stitch: a whole block group, or the span of a locus within one.
///
/// A locus is linear, but stitching it takes every variant route between its first and last
/// positions, so the result keeps the variation of that part of the graph without first
/// deriving a subgraph for it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StitchSource {
    BlockGroup(HashId),
    Locus {
        block_group_id: HashId,
        ranges: Vec<StitchRange>,
    },
}

/// Creates a new block group in `new_sample_name` by concatenating `sources` in order: the end of
/// each piece is connected to the start of the next. The result has a current path when every
/// piece does (a locus contributes the route it spells).
pub fn stitch_sources(
    context: &DbContext,
    collection_name: &str,
    new_sample_name: &str,
    new_region_name: &str,
    sources: &[StitchSource],
) -> Result<BlockGroup, GraphOperationError> {
    let conn = context.graph().conn();
    conn.execute_batch("SAVEPOINT stitch_sources")?;
    let result = create_stitched_from_sources(
        context,
        collection_name,
        new_sample_name,
        new_region_name,
        sources,
    );
    match result {
        Ok(block_group) => {
            conn.execute_batch("RELEASE stitch_sources")?;
            Ok(block_group)
        }
        Err(err) => {
            conn.execute_batch("ROLLBACK TO stitch_sources")?;
            conn.execute_batch("RELEASE stitch_sources")?;
            Err(err)
        }
    }
}

fn create_stitched_from_sources(
    context: &DbContext,
    collection_name: &str,
    new_sample_name: &str,
    new_region_name: &str,
    sources: &[StitchSource],
) -> Result<BlockGroup, GraphOperationError> {
    let conn = context.graph().conn();
    let mut seen_block_group_ids = HashSet::new();
    let mut seen_edge_ids = HashSet::new();
    for source in sources {
        if let StitchSource::BlockGroup(block_group_id) = source {
            if !seen_block_group_ids.insert(*block_group_id) {
                return Err(GraphOperationError::InvalidStitchInput(format!(
                    "sequence graph {block_group_id} appears more than once"
                )));
            }
            // Stitching reads each graph's current path to build the new one.
            if BlockGroup::get_current_path(conn, block_group_id, None).is_err() {
                return Err(GraphOperationError::InvalidStitchInput(format!(
                    "sequence graph {block_group_id} has no current path to stitch"
                )));
            }
            for edge in BlockGroupEdge::edges_for_block_group(conn, block_group_id, None) {
                if !edge.edge.is_start_edge()
                    && !edge.edge.is_end_edge()
                    && !seen_edge_ids.insert(edge.edge.id)
                {
                    return Err(GraphOperationError::InvalidStitchInput(format!(
                        "sequence graphs share edge {}",
                        edge.edge.id
                    )));
                }
            }
        }
    }

    let _new_sample = Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: new_sample_name,
            ..Default::default()
        },
    );
    let child_block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name: new_sample_name,
            name: new_region_name,
            ..Default::default()
        },
    )?;

    let mut chunks = Vec::with_capacity(sources.len());
    for source in sources {
        match source {
            StitchSource::BlockGroup(block_group_id) => {
                let bg_edges = BlockGroupEdge::edges_for_block_group(conn, block_group_id, None)
                    .into_iter()
                    .filter(|edge| !edge.edge.is_start_edge() && !edge.edge.is_end_edge())
                    .map(|edge| BlockGroupEdgeData {
                        block_group_id: child_block_group.id,
                        edge_id: edge.edge.id,
                        chromosome_index: edge.chromosome_index,
                        phased: edge.phased,
                    })
                    .collect::<Vec<_>>();
                BlockGroupEdge::bulk_create(conn, &bg_edges);
                chunks.push(load_block_group_chunk(conn, *block_group_id));
            }
            StitchSource::Locus {
                block_group_id,
                ranges,
            } => chunks.push(add_locus_chunk(
                context,
                block_group_id,
                ranges,
                &child_block_group.id,
            )?),
        }
    }

    make_stitch_from_block_groups(context, &chunks, child_block_group.id, new_region_name)?;
    validate_stitched_block_group_is_acyclic(conn, context.workspace(), &child_block_group.id)?;
    Ok(child_block_group)
}

/// Copies the subgraph spanned by a locus into `target_block_group_id` and returns the chunk that
/// stitches it, whose path is the route the locus itself spells.
fn add_locus_chunk(
    context: &DbContext,
    block_group_id: &HashId,
    ranges: &[StitchRange],
    target_block_group_id: &HashId,
) -> Result<BlockGroupChunk, GraphOperationError> {
    let conn = context.graph().conn();
    let (Some(first), Some(last)) = (ranges.first(), ranges.last()) else {
        return Err(GraphOperationError::InvalidStitchInput(
            "a locus to stitch must cover at least one base".to_string(),
        ));
    };
    let graph = BlockGroup::get_graph(conn, context.workspace(), block_group_id, None)?;
    let holds = |node_id: HashId, coordinate: i64| {
        graph.nodes().any(|node| {
            node.node_id == node_id
                && node.sequence_start <= coordinate
                && node.sequence_end >= coordinate
        })
    };
    if !holds(first.node_id, first.start) || !holds(last.node_id, last.end) {
        return Err(GraphOperationError::InvalidStitchInput(
            "a locus to stitch is not part of its sequence graph".to_string(),
        ));
    }

    let boundary_block = |range: &StitchRange| NodeIntervalBlock {
        node_id: range.node_id,
        start: 0,
        end: 0,
        sequence_start: range.start,
        sequence_end: range.end,
        strand: Strand::Forward,
    };
    let start_block = boundary_block(first);
    let end_block = boundary_block(last);
    BlockGroup::derive_subgraph(
        conn,
        context.workspace(),
        block_group_id,
        SubgraphBoundary {
            block: &start_block,
            sequence_coordinate: first.start,
        },
        SubgraphBoundary {
            block: &end_block,
            sequence_coordinate: last.end,
        },
        target_block_group_id,
        false,
    )?;

    let mut path_edges = Vec::with_capacity(ranges.len().saturating_sub(1));
    for pair in ranges.windows(2) {
        path_edges.push(Edge::create(
            conn,
            pair[0].node_id,
            pair[0].end,
            Strand::Forward,
            pair[1].node_id,
            pair[1].start,
            Strand::Forward,
        )?);
    }
    let start_point = NodePoint {
        id: first.node_id,
        coordinate: first.start,
        strand: Strand::Forward,
    };
    let end_point = NodePoint {
        id: last.node_id,
        coordinate: last.end,
        strand: Strand::Forward,
    };
    Ok(BlockGroupChunk {
        entry_node_points: vec![start_point.clone()],
        exit_node_points: vec![end_point.clone()],
        path_edges,
        path_start_point: Some(start_point),
        path_end_point: Some(end_point),
    })
}

fn create_stitched_block_group(
    context: &DbContext,
    collection_name: &str,
    new_sample_name: &str,
    new_region_name: &str,
    stitch_inputs: &[StitchInput<'_>],
    block_group_chunks: &[BlockGroupChunk],
) -> Result<(), GraphOperationError> {
    let conn = context.graph().conn();

    let _new_sample = Sample::get_or_create(
        conn,
        gen_models::sample::NewSample {
            name: new_sample_name,
            ..Default::default()
        },
    );

    let child_block_group = BlockGroup::create(
        conn,
        NewBlockGroup {
            collection_name,
            sample_name: new_sample_name,
            name: new_region_name,
            ..Default::default()
        },
    )?;

    for input in stitch_inputs {
        let bg_edges = input
            .nonterminal_edges
            .iter()
            .map(|edge| BlockGroupEdgeData {
                block_group_id: child_block_group.id,
                edge_id: edge.edge.id,
                chromosome_index: edge.chromosome_index,
                phased: edge.phased,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &bg_edges);
    }

    make_stitch_from_block_groups(
        context,
        block_group_chunks,
        child_block_group.id,
        new_region_name,
    )?;

    validate_stitched_block_group_is_acyclic(conn, context.workspace(), &child_block_group.id)?;

    Ok(())
}

fn validate_stitched_block_group_is_acyclic(
    conn: &GraphConnection,
    workspace: &Workspace,
    block_group_id: &HashId,
) -> Result<(), GraphOperationError> {
    let mut graph = BlockGroup::get_graph(conn, workspace, block_group_id, None)?;
    // An insertion inside a node meets its own routing block again, a loop that reads no bases
    // twice. Only cycles through sequence blocks come from stitching.
    BlockGroup::contract_zero_width_blocks(&mut graph);
    if is_cyclic_directed(&graph) {
        let describe = |block: &GraphNode| {
            format!(
                "{}:{}-{}",
                block.node_id, block.sequence_start, block.sequence_end
            )
        };
        let cycle = kosaraju_scc(&graph)
            .into_iter()
            .find(|component| component.len() > 1)
            .map(|component| {
                graph
                    .all_edges()
                    .filter(|(source, target, _)| {
                        component.contains(source) && component.contains(target)
                    })
                    .map(|(source, target, _)| {
                        format!("{} -> {}", describe(&source), describe(&target))
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .unwrap_or_default();
        return Err(GraphOperationError::StitchedGraphCycle(format!(
            "block group {block_group_id} is cyclic through {cycle}"
        )));
    }

    Ok(())
}

pub fn make_stitch_from_block_groups(
    context: &DbContext,
    block_group_chunks: &[BlockGroupChunk],
    child_block_group_id: HashId,
    new_region_name: &str,
) -> Result<(), GraphOperationError> {
    let conn = context.graph().conn();

    let start_node_point = NodePoint {
        id: PATH_START_NODE_ID,
        coordinate: 0,
        strand: Strand::Forward,
    };
    let mut result_block_group_chunk = BlockGroupChunk {
        entry_node_points: vec![start_node_point.clone()],
        exit_node_points: vec![start_node_point.clone()],
        path_edges: vec![],
        path_start_point: Some(start_node_point.clone()),
        path_end_point: Some(start_node_point.clone()),
    };

    for chunk in block_group_chunks {
        result_block_group_chunk =
            stitch(conn, &result_block_group_chunk, chunk, child_block_group_id)?;
    }

    let end_node_point = NodePoint {
        id: PATH_END_NODE_ID,
        coordinate: 0,
        strand: Strand::Forward,
    };
    let end_chunk = BlockGroupChunk {
        entry_node_points: vec![end_node_point.clone()],
        exit_node_points: vec![end_node_point.clone()],
        path_edges: vec![],
        path_start_point: Some(end_node_point.clone()),
        path_end_point: Some(end_node_point.clone()),
    };

    result_block_group_chunk = stitch(
        conn,
        &result_block_group_chunk,
        &end_chunk,
        child_block_group_id,
    )?;

    if !result_block_group_chunk.path_edges.is_empty() {
        let new_path_edge_ids = result_block_group_chunk
            .path_edges
            .iter()
            .map(|edge| edge.id)
            .collect::<Vec<HashId>>();

        Path::create(
            conn,
            new_region_name,
            &child_block_group_id,
            &new_path_edge_ids,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use gen_core::{PATH_END_NODE_ID, PATH_START_NODE_ID, Strand};
    use gen_models::{
        block_group::NewBlockGroup, block_group_edge::BlockGroupEdgeData, collection::Collection,
        edge::Edge, node::Node, path::Path, sample::Sample, sequence::Sequence,
    };

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{setup_block_group, setup_gen},
        updates::fasta::update_with_fasta,
    };

    #[test]
    fn test_derive_chunks_one_insertion() {
        /*
        AAAAAAAAAA -> TTTTTTTTTT -> CCCCCCCCCC -> GGGGGGGGGG
                          \-> AAAAAAAA ->/
        Subgraph range:  |-----------------|
        Sequences of the subgraph are TAAAAAAAAC, TTTTTCCCCC
         */
        let context = setup_gen();
        let conn = context.graph().conn();

        Collection::create(conn, "test").unwrap();
        let (block_group1_id, original_path) = setup_block_group(conn);

        let intervaltree = original_path.intervaltree(conn).unwrap();
        let insert_start_node_id = intervaltree.query_point(16).next().unwrap().value.node_id;
        let insert_end_node_id = intervaltree.query_point(24).next().unwrap().value.node_id;

        let insert_sequence = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAA")
            .save(conn)
            .unwrap();
        let insert_node_id = Node::create(
            conn,
            &insert_sequence.hash,
            &HashId::convert_str(&format!("test-insert-a.{}", insert_sequence.hash)),
        )
        .unwrap();
        let edge_into_insert = Edge::create(
            conn,
            insert_start_node_id,
            6,
            Strand::Forward,
            insert_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let edge_out_of_insert = Edge::create(
            conn,
            insert_node_id,
            8,
            Strand::Forward,
            insert_end_node_id,
            4,
            Strand::Forward,
        )
        .unwrap();
        let ref_heal_1 = Edge::create(
            conn,
            insert_start_node_id,
            6,
            Strand::Forward,
            insert_start_node_id,
            6,
            Strand::Forward,
        )
        .unwrap();
        let ref_heal_2 = Edge::create(
            conn,
            insert_end_node_id,
            4,
            Strand::Forward,
            insert_end_node_id,
            4,
            Strand::Forward,
        )
        .unwrap();

        let edge_ids = [
            edge_into_insert.id,
            edge_out_of_insert.id,
            ref_heal_1.id,
            ref_heal_2.id,
        ];
        let block_group_edges = edge_ids
            .iter()
            .enumerate()
            .map(|(i, edge_id)| BlockGroupEdgeData {
                block_group_id: block_group1_id,
                edge_id: *edge_id,
                chromosome_index: if i < 2 { 1 } else { 0 },
                phased: 0,
            })
            .collect::<Vec<BlockGroupEdgeData>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);

        let insert_path = original_path
            .new_path_with(conn, 16, 24, &edge_into_insert, &edge_out_of_insert)
            .unwrap();
        assert_eq!(
            insert_path
                .sequence(conn, context.workspace(), None)
                .unwrap(),
            "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG"
        );

        let all_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group1_id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "AAAAAAAAAATTTTTTTTTTCCCCCCCCCCGGGGGGGGGG".to_string(),
                "AAAAAAAAAATTTTTTAAAAAAAACCCCCCGGGGGGGGGG".to_string(),
            ])
        );

        derive_chunks(
            &context,
            "test",
            "test",
            Sample::DEFAULT_NAME,
            "chr1",
            None,
            vec![Range { start: 15, end: 25 }],
            None,
            true,
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, "test", Sample::DEFAULT_NAME, None);
        let block_group2 = block_groups.iter().find(|x| x.name == "chr1").unwrap();

        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences2,
            HashSet::from_iter(vec!["TTTTTCCCCC".to_string(), "TAAAAAAAAC".to_string(),])
        );

        let new_path = BlockGroup::get_current_path(conn, &block_group2.id, None).unwrap();
        assert_eq!(
            new_path.sequence(conn, context.workspace(), None).unwrap(),
            "TAAAAAAAAC"
        );
    }

    #[test]
    fn test_derive_chunks_two_inserts() {
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aa.fa");

        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test";

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        let _ = update_with_fasta(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "test1",
            "m123:3-5",
            fasta_update_path.to_str().unwrap(),
            false,
        )
        .unwrap();

        let _ = update_with_fasta(
            &context,
            collection,
            "test1",
            "test2",
            "m123:15-20",
            fasta_update_path.to_str().unwrap(),
            false,
        )
        .unwrap();

        let original_block_groups =
            Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME, None);
        let original_block_group_id = &original_block_groups[0].id;
        let all_original_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            original_block_group_id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_original_sequences,
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),])
        );

        let grandchild_block_groups = Sample::get_block_groups(conn, collection, "test2", None);
        let grandchild_block_group_id = &grandchild_block_groups[0].id;
        let all_grandchild_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            grandchild_block_group_id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_grandchild_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCGATCAAGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCAAGGAACACACAGAGA".to_string(),
            ])
        );

        derive_chunks(
            &context,
            collection,
            "test2",
            "test3",
            "m123",
            None,
            vec![
                Range { start: 0, end: 1 },
                Range { start: 1, end: 8 },
                Range { start: 8, end: 25 },
                Range { start: 25, end: 31 },
            ],
            None,
            true,
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, "test3", None);
        let block_group2 = block_groups.iter().find(|x| x.name == "m123.2").unwrap();

        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences2,
            HashSet::from_iter(vec!["TCAATCG".to_string(), "TCGATCG".to_string(),])
        );

        let path2 = BlockGroup::get_current_path(conn, &block_group2.id, None).unwrap();
        assert_eq!(
            path2.sequence(conn, context.workspace(), None).unwrap(),
            "TCAATCG"
        );

        let block_group3 = block_groups.iter().find(|x| x.name == "m123.3").unwrap();
        let all_sequences3 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group3.id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences3,
            HashSet::from_iter(vec![
                "ATCGATCAAGGAACACA".to_string(),
                "ATCGATCGATCGGGAACACA".to_string(),
            ])
        );

        let path3 = BlockGroup::get_current_path(conn, &block_group3.id, None).unwrap();
        assert_eq!(
            path3.sequence(conn, context.workspace(), None).unwrap(),
            "ATCGATCAAGGAACACA"
        );
    }

    #[test]
    fn test_derive_chunks_two_inserts_then_stitch() {
        let mut fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_path.push("fixtures/simple.fa");
        let mut fasta_update_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fasta_update_path.push("fixtures/aa.fa");

        let context = setup_gen();
        let conn = context.graph().conn();

        let collection = "test";

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            collection,
            Sample::DEFAULT_NAME,
            false,
            &[],
        )
        .unwrap();

        let _ = update_with_fasta(
            &context,
            collection,
            Sample::DEFAULT_NAME,
            "test1",
            "m123:3-5",
            fasta_update_path.to_str().unwrap(),
            false,
        )
        .unwrap();

        let _ = update_with_fasta(
            &context,
            collection,
            "test1",
            "test2",
            "m123:15-20",
            fasta_update_path.to_str().unwrap(),
            false,
        )
        .unwrap();

        let original_block_groups =
            Sample::get_block_groups(conn, collection, Sample::DEFAULT_NAME, None);
        let original_block_group_id = &original_block_groups[0].id;
        let all_original_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            original_block_group_id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_original_sequences,
            HashSet::from_iter(vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),])
        );

        let grandchild_block_groups = Sample::get_block_groups(conn, collection, "test2", None);
        let grandchild_block_group_id = &grandchild_block_groups[0].id;
        let all_grandchild_sequences = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            grandchild_block_group_id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_grandchild_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCGATCGATCAAGGAACACACAGAGA".to_string(),
                "ATCAATCGATCGATCAAGGAACACACAGAGA".to_string(),
            ])
        );

        derive_chunks(
            &context,
            collection,
            "test2",
            "test3",
            "m123",
            None,
            vec![
                Range { start: 0, end: 1 },
                Range { start: 1, end: 8 },
                Range { start: 8, end: 25 },
                Range { start: 25, end: 31 },
            ],
            None,
            true,
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, "test3", None);
        let block_group2 = block_groups.iter().find(|x| x.name == "m123.2").unwrap();

        let all_sequences2 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group2.id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences2,
            HashSet::from_iter(vec!["TCAATCG".to_string(), "TCGATCG".to_string(),])
        );

        let path2 = BlockGroup::get_current_path(conn, &block_group2.id, None).unwrap();
        assert_eq!(
            path2.sequence(conn, context.workspace(), None).unwrap(),
            "TCAATCG"
        );

        let block_group3 = block_groups.iter().find(|x| x.name == "m123.3").unwrap();
        let all_sequences3 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group3.id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences3,
            HashSet::from_iter(vec![
                "ATCGATCAAGGAACACA".to_string(),
                "ATCGATCGATCGGGAACACA".to_string(),
            ])
        );

        let path3 = BlockGroup::get_current_path(conn, &block_group3.id, None).unwrap();
        assert_eq!(
            path3.sequence(conn, context.workspace(), None).unwrap(),
            "ATCGATCAAGGAACACA"
        );

        // Stitch the two main chunks back together in same order
        make_stitch(
            &context,
            collection,
            "test3",
            "test4",
            &vec!["m123.2", "m123.3"],
            "m123.stitched",
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, "test4", None);
        let block_group4 = block_groups
            .iter()
            .find(|x| x.name == "m123.stitched")
            .unwrap();

        let all_sequences4 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group4.id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences4,
            HashSet::from_iter(vec![
                "TCAATCGATCGATCAAGGAACACA".to_string(),
                "TCAATCGATCGATCGATCGGGAACACA".to_string(),
                "TCGATCGATCGATCAAGGAACACA".to_string(),
                "TCGATCGATCGATCGATCGGGAACACA".to_string(),
            ])
        );

        let path4 = BlockGroup::get_current_path(conn, &block_group4.id, None).unwrap();
        // path2 + path3 concatenated
        assert_eq!(
            path4.sequence(conn, context.workspace(), None).unwrap(),
            "TCAATCGATCGATCAAGGAACACA"
        );

        // Stitch the two main chunks together but in reverse order
        make_stitch(
            &context,
            collection,
            "test3",
            "test5",
            &vec!["m123.3", "m123.2"],
            "m123.reverse-stitched",
        )
        .unwrap();

        let block_groups = Sample::get_block_groups(conn, collection, "test5", None);
        let block_group5 = block_groups
            .iter()
            .find(|x| x.name == "m123.reverse-stitched")
            .unwrap();

        let all_sequences5 = BlockGroup::get_all_sequences(
            conn,
            crate::test_helpers::test_workspace(),
            &block_group5.id,
            false,
        )
        .unwrap();
        assert_eq!(
            all_sequences5,
            HashSet::from_iter(vec![
                "ATCGATCAAGGAACACATCAATCG".to_string(),
                "ATCGATCAAGGAACACATCGATCG".to_string(),
                "ATCGATCGATCGGGAACACATCAATCG".to_string(),
                "ATCGATCGATCGGGAACACATCGATCG".to_string(),
            ])
        );

        let path5 = BlockGroup::get_current_path(conn, &block_group5.id, None).unwrap();
        // path3 + path2 concatenated
        assert_eq!(
            path5.sequence(conn, context.workspace(), None).unwrap(),
            "ATCGATCAAGGAACACATCAATCG"
        );
    }

    #[test]
    fn test_make_stitch_rejects_duplicate_region_input() {
        let context = setup_gen();
        let conn = context.graph().conn();
        setup_block_group(conn);

        let result = make_stitch(
            &context,
            "test",
            "test",
            "stitched",
            &vec!["chr1", "chr1"],
            "chr1.stitched",
        );

        assert!(matches!(
            result,
            Err(GraphOperationError::InvalidStitchInput(_))
        ));
        let block_groups = Sample::get_block_groups(conn, "test", "stitched", None);
        assert!(block_groups.is_empty());
    }

    #[test]
    fn test_make_stitch_rolls_back_cyclic_output() {
        let context = setup_gen();
        let conn = context.graph().conn();
        Collection::create(conn, "test").unwrap();
        Sample::get_or_create(
            conn,
            gen_models::sample::NewSample {
                name: "parent",
                ..Default::default()
            },
        )
        .unwrap();

        let a_seq = Sequence::new()
            .sequence_type("DNA")
            .sequence("AAAAAAAAAA")
            .save(conn)
            .unwrap();
        let a_node_id = Node::create(
            conn,
            &a_seq.hash,
            &HashId::convert_str(&format!("cycle-a.{}", a_seq.hash)),
        )
        .unwrap();
        let t_seq = Sequence::new()
            .sequence_type("DNA")
            .sequence("TTTTTTTTTT")
            .save(conn)
            .unwrap();
        let t_node_id = Node::create(
            conn,
            &t_seq.hash,
            &HashId::convert_str(&format!("cycle-t.{}", t_seq.hash)),
        )
        .unwrap();

        create_two_node_block_group(conn, "forward", a_node_id, t_node_id);
        create_two_node_block_group(conn, "reverse", t_node_id, a_node_id);

        let result = make_stitch(
            &context,
            "test",
            "parent",
            "stitched",
            &vec!["forward", "reverse"],
            "cycle",
        );

        assert!(matches!(
            result,
            Err(GraphOperationError::StitchedGraphCycle(_))
        ));
        let block_groups = Sample::get_block_groups(conn, "test", "stitched", None);
        assert!(block_groups.is_empty());
    }

    fn create_two_node_block_group(
        conn: &GraphConnection,
        name: &str,
        source_node_id: HashId,
        target_node_id: HashId,
    ) {
        let block_group = BlockGroup::create(
            conn,
            NewBlockGroup {
                collection_name: "test",
                sample_name: "parent",
                name,
                ..Default::default()
            },
        )
        .unwrap();
        let start_edge = Edge::create(
            conn,
            PATH_START_NODE_ID,
            0,
            Strand::Forward,
            source_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let internal_edge = Edge::create(
            conn,
            source_node_id,
            10,
            Strand::Forward,
            target_node_id,
            0,
            Strand::Forward,
        )
        .unwrap();
        let end_edge = Edge::create(
            conn,
            target_node_id,
            10,
            Strand::Forward,
            PATH_END_NODE_ID,
            0,
            Strand::Forward,
        )
        .unwrap();
        let block_group_edges = [start_edge.id, internal_edge.id, end_edge.id]
            .iter()
            .map(|edge_id| BlockGroupEdgeData {
                block_group_id: block_group.id,
                edge_id: *edge_id,
                chromosome_index: 0,
                phased: 0,
            })
            .collect::<Vec<_>>();
        BlockGroupEdge::bulk_create(conn, &block_group_edges);
        Path::create(
            conn,
            name,
            &block_group.id,
            &[start_edge.id, internal_edge.id, end_edge.id],
        )
        .unwrap();
    }
}
