use std::collections::HashSet;

use r#gen::graphs::{NodePoint, operators::GraphOperationError};
use gen_core::{HashId, NodeIntervalBlock, is_end_node, is_start_node};
use gen_models::{
    block_group::{BlockGroup, NewBlockGroup, SubgraphBoundary},
    block_group_edge::BlockGroupEdge,
    db::DbContext,
    path::Path,
    sample::{NewSample, Sample},
};

/// Creates a block group named like `source_block_group_id` in `new_sample_name` that holds every
/// route of the source between `start` and `end`, which need no linear coordinates.
///
/// The new block group gets a current path when the source's own current path runs from `start`
/// to `end`; otherwise it has none, since no single route is the obvious one.
pub(crate) fn derive_subgraph_between(
    context: &DbContext,
    source_block_group_id: &HashId,
    new_sample_name: &str,
    start: NodePoint,
    end: NodePoint,
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
        Err(error) => {
            conn.execute_batch("ROLLBACK TO derive_subgraph_between")?;
            conn.execute_batch("RELEASE derive_subgraph_between")?;
            Err(error)
        }
    }
}

fn create_subgraph_between(
    context: &DbContext,
    source_block_group_id: &HashId,
    new_sample_name: &str,
    start: NodePoint,
    end: NodePoint,
) -> Result<BlockGroup, GraphOperationError> {
    let conn = context.graph().conn();
    let source = BlockGroup::get_by_id(conn, source_block_group_id, None)?;
    let _new_sample = Sample::get_or_create(
        conn,
        NewSample {
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

    let boundary = |point: &NodePoint| NodeIntervalBlock {
        node_id: point.id,
        start: 0,
        end: 0,
        sequence_start: point.coordinate,
        sequence_end: point.coordinate,
        strand: point.strand,
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
        let begins_at_start = internal.first().map_or(start.id == end.id, |edge| {
            edge.source_node_id == start.id && edge.source_coordinate >= start.coordinate
        });
        let ends_at_end = internal.last().map_or(start.id == end.id, |edge| {
            edge.target_node_id == end.id && edge.target_coordinate <= end.coordinate
        });
        let start_edge = child_edges.iter().find(|edge| {
            is_start_node(edge.edge.source_node_id)
                && edge.edge.target_node_id == start.id
                && edge.edge.target_coordinate == start.coordinate
        });
        let end_edge = child_edges.iter().find(|edge| {
            is_end_node(edge.edge.target_node_id)
                && edge.edge.source_node_id == end.id
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
