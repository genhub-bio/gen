//! Adapter functions to convert between LayoutNode/LayoutEdge and NodeData/LayoutEdge
//! for the Rust edge router implementation.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::{
    edge_router::{LayoutError, NodeData},
    layout::{LayoutEdge, LayoutNode, NodeRole},
};

/// Convert a LayoutNode to NodeData for the Rust edge router
pub fn layout_node_to_node_data(
    layout_node: &LayoutNode,
    node_index: NodeIndex<u32>,
) -> Result<NodeData, LayoutError> {
    // Extract node ID from the node index
    let node_id = node_index.index() as u64;

    // Convert position from LocalPos to (i64, i64)
    let position = (layout_node.pos.x, layout_node.pos.y);

    // Convert size from (u64, u64) to (i64, i64)
    let size = (layout_node.size.0 as i64, layout_node.size.1 as i64);

    // Determine node type and extract original domain data based on role
    let (node_type, glyph_index, original_node_id, layer, partition_index) = match &layout_node.role
    {
        NodeRole::Data(original_id) => (
            Some("Graph".to_string()),
            None,
            Some(original_id.index() as u64), // Preserve original domain node ID
            layout_node.layer,                // Preserve layer info
            Some(layout_node.pos.partition_idx as u64), // Preserve partition info
        ),
        NodeRole::Routing => (
            Some("Routing".to_string()),
            None, // Glyph computed on-the-fly, not stored
            None, // No original domain ID for routing nodes
            None, // No layer info for routing nodes
            None, // No partition info for routing nodes
        ),
        NodeRole::Stitch(_) => (
            Some("Stitching".to_string()),
            None,
            None, // No original domain ID for stitch nodes
            None, // No layer info for stitch nodes
            None, // No partition info for stitch nodes
        ),
    };

    Ok(NodeData {
        node_id,
        position,
        node_type,
        ports: None, // Will be computed by the routing algorithm
        glyph_index,
        size,
        original_node_id,
        layer,
        partition_index,
    })
}

/// Convert NodeData back to LayoutNode after routing
pub fn node_data_to_layout_node(node_data: &NodeData) -> Result<LayoutNode, LayoutError> {
    // Convert position back to LocalPos
    let pos = crate::geometry::LocalPos::new_xy(
        0, // partition_idx - preserve original or default to 0
        node_data.position.0,
        node_data.position.1,
    );

    // Convert size back to (u64, u64)
    let size = (
        node_data.size.0.max(0) as u64,
        node_data.size.1.max(0) as u64,
    );

    // Reconstruct role based on preserved original domain data
    let role = if let Some(original_domain_id) = node_data.original_node_id {
        // This is a data node - use the preserved original domain node ID
        NodeRole::Data(NodeIndex::new(original_domain_id as usize))
    } else if let Some(ref node_type) = node_data.node_type {
        match node_type.as_str() {
            "Routing" => NodeRole::Routing,
            "Stitching" => {
                // Would need to determine stitch side, but for now use Left as default
                NodeRole::Stitch(crate::partition::StitchSide::Left)
            }
            _ => NodeRole::Routing, // Default fallback
        }
    } else {
        NodeRole::Routing // Default fallback
    };

    // Update partition index in position if preserved
    let final_pos = if let Some(partition_idx) = node_data.partition_index {
        crate::geometry::LocalPos::new(
            partition_idx as crate::geometry::PartitionIndex,
            (node_data.position.0, node_data.position.1).into(),
        )
    } else {
        pos
    };

    Ok(LayoutNode {
        role,
        pos: final_pos,
        size,
        layer: node_data.layer,
    })
}

/// Validate that a LayoutNode has all required fields for routing
pub fn validate_layout_node(layout_node: &LayoutNode) -> Result<(), LayoutError> {
    // Check that position is reasonable
    if layout_node.pos.x.abs() > 10000 || layout_node.pos.y.abs() > 10000 {
        return Err(LayoutError::InvalidPosition(
            layout_node.pos.x as i32,
            layout_node.pos.y as i32,
        ));
    }

    // Check that size is positive
    if layout_node.size.0 == 0 || layout_node.size.1 == 0 {
        return Err(LayoutError::InvalidSize(
            layout_node.size.0,
            layout_node.size.1,
        ));
    }

    Ok(())
}

/// Validate that a LayoutEdge has valid bundle data
pub fn validate_layout_edge(layout_edge: &LayoutEdge) -> Result<(), LayoutError> {
    if layout_edge.bundle.is_empty() {
        return Err(LayoutError::EmptyEdgeBundle);
    }

    Ok(())
}

/// Create a mapping of node indices to LayoutNodes for preservation during conversion
pub fn create_node_mapping(
    graph: &petgraph::stable_graph::StableGraph<LayoutNode, LayoutEdge, petgraph::Undirected, u32>,
) -> HashMap<u64, LayoutNode> {
    use petgraph::visit::IntoNodeReferences;

    graph
        .node_references()
        .map(|(node_idx, layout_node)| (node_idx.index() as u64, layout_node.clone()))
        .collect()
}

/// Create a mapping of edge pairs to LayoutEdges for preservation during conversion
pub fn create_edge_mapping(
    graph: &petgraph::stable_graph::StableGraph<LayoutNode, LayoutEdge, petgraph::Undirected, u32>,
) -> HashMap<(NodeIndex<u32>, NodeIndex<u32>), LayoutEdge> {
    use petgraph::visit::{EdgeRef, IntoEdgeReferences};

    graph
        .edge_references()
        .map(|edge_ref| {
            let source = edge_ref.source();
            let target = edge_ref.target();
            let edge_weight = edge_ref.weight();
            ((source, target), edge_weight.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use petgraph::graph::NodeIndex;

    use super::*;
    use crate::geometry::LocalPos;

    #[test]
    fn test_layout_node_to_node_data_conversion() {
        let layout_node = LayoutNode::data(
            NodeIndex::new(42),
            LocalPos::new_xy(0, 100, 200),
            (5, 7),
            Some(1),
        );

        let node_data = layout_node_to_node_data(&layout_node, NodeIndex::new(0)).unwrap();

        assert_eq!(node_data.node_id, 0);
        assert_eq!(node_data.position, (100, 200));
        assert_eq!(node_data.size, (5, 7));
        assert_eq!(node_data.node_type, Some("Graph".to_string()));
        assert_eq!(node_data.glyph_index, None);
        assert_eq!(node_data.original_node_id, Some(42)); // Preserved from original domain node
        assert_eq!(node_data.layer, Some(1));
        assert_eq!(node_data.partition_index, Some(0));
    }

    #[test]
    fn test_routing_node_conversion() {
        let layout_node = LayoutNode::routing(LocalPos::new_xy(0, 50, 75), (1, 1));

        let node_data = layout_node_to_node_data(&layout_node, NodeIndex::new(1)).unwrap();

        assert_eq!(node_data.node_id, 1);
        assert_eq!(node_data.position, (50, 75));
        assert_eq!(node_data.size, (1, 1));
        assert_eq!(node_data.node_type, Some("Routing".to_string()));
        assert_eq!(node_data.glyph_index, None); // Glyph computed on-the-fly, not stored
        assert_eq!(node_data.original_node_id, None); // No original domain ID for routing nodes
        assert_eq!(node_data.layer, None);
        assert_eq!(node_data.partition_index, None);
    }

    #[test]
    fn test_validation_functions() {
        let valid_node =
            LayoutNode::data(NodeIndex::new(0), LocalPos::new_xy(0, 10, 20), (2, 3), None);
        assert!(validate_layout_node(&valid_node).is_ok());

        let valid_edge = LayoutEdge::new(NodeIndex::new(0), NodeIndex::new(1));
        assert!(validate_layout_edge(&valid_edge).is_ok());

        let empty_edge = LayoutEdge::empty();
        assert!(validate_layout_edge(&empty_edge).is_err());
    }
}
