use std::collections::HashMap;

use gen_core::is_terminal;
use gen_graph::{GenGraph, GraphEdge, GraphNode, find_articulation_points};
use gen_models::node::Node;
use log::{debug, info, warn};
use petgraph::{
    Directed, algo::toposort, graph::NodeIndex, graphmap::GraphMap, stable_graph::StableDiGraph,
};
use rust_sugiyama::configure::Config;

use crate::views::block_group_viewer::PlotParameters;

/// Parameters for the partition-based layout algorithm
/// - `MIN_CHUNK_SIZE`: the minimum size to start a new partition at an articulation point
/// - `MAX_CHUNK_SIZE`: the size at which we forcibly close out a partition
const MIN_CHUNK_SIZE: usize = 1e4 as usize;
const MAX_CHUNK_SIZE: usize = 1e5 as usize;

/// A graph that is compatible with the sugiyama crate.
pub type SugiyamaGraph = StableDiGraph<GraphNode, Vec<GraphEdge>, u32>;

/// Raw layout data in the format returned by the rust_sugiyama crate
pub type RawLayout = Vec<(NodeIndex, (f64, f64))>;

/// Type alias for inter-partition edges
pub type PartitionEdge = (GraphNode, GraphNode, Vec<GraphEdge>);

/// Type alias for partial layout data
pub type PartialLayout = (Vec<(GraphNode, (f64, f64))>, f64, f64);

/// The result of partitioning a graph into mutually exclusive subgraphs, intended for use in layout algorithms.
/// - `parts` is vector of StableDiGraphs that make up the partition.
///     - Node weights are GraphNode, edge weights are Vec<GraphEdge>.
///     - Indices are u32 for compatibility with the sugiyama crate.
///     - The edges are defined by the index of the nodes within the partition.
/// - `inter_part_edges` is a hashmap of edges that cross the partition boundaries
///     - Keyed to a tuple of the source and target subgraph indices.
///     - Edges are defined by the NodeIndex of the nodes within their respective subgraphs.
#[derive(Debug)]
pub struct Partition {
    pub parts: Vec<SugiyamaGraph>,
    pub inter_part_edges: HashMap<(usize, usize), Vec<PartitionEdge>>,
}

/// Partition a GenGraph into subgraphs, preferably at articulation points
/// - Subgraph sizes are controlled by min_size and max_size
/// - Algorithm:
///     - Perform a topological sort of the graph
///     - Visit nodes in topological order, accumulating nodes in a subgraph.
///     - Keep track of the number of nodes seen so far.
///     - If an articulation point is encountered, and the minimum part size has been reached,
///        - add the current subgraph to the list of parts
///        - start a new subgraph
///     - If the maximum part size is reached, forcibly close out the current subgraph.
impl Partition {
    pub fn new(graph: &GenGraph, min_size: usize, max_size: usize) -> Self {
        let mut subgraphs: Vec<SugiyamaGraph> = Vec::new();
        let mut current_subgraph: SugiyamaGraph = SugiyamaGraph::new();
        let mut current_subgraph_index = 0;

        // Mapping from GraphNode to (part index, node index)
        let mut node_to_nx: HashMap<GraphNode, (usize, NodeIndex<u32>)> = HashMap::new();

        // Find articulation points
        let articulation_points = find_articulation_points(graph);

        // Create a topological sort of the graph to ensure that each time we visit an articulation point,
        // we have already visited all the nodes that can possibly come before it.
        let sorted_nodes = toposort(&graph, None)
            .unwrap_or_else(|_| {
                panic!("Graph is not a DAG");
            })
            .into_iter()
            .collect::<Vec<_>>();

        // Create the partition subgraphs
        for node in sorted_nodes {
            // Once we have enough nodes, try to close out at an articulation point.
            // Forcibly close out if we don't find an articulation point in time.
            if (current_subgraph.node_count() >= min_size && articulation_points.contains(&node))
                || (current_subgraph.node_count() >= max_size)
            {
                subgraphs.push(current_subgraph);
                current_subgraph = StableDiGraph::new();
                current_subgraph_index += 1;
            }

            // Add the node to the current subgraph
            let node_index: NodeIndex<u32> = current_subgraph.add_node(node);
            node_to_nx.insert(node, (current_subgraph_index, node_index));
        }

        // Add the last subgraph to the list of parts
        if current_subgraph.node_count() > 0 {
            subgraphs.push(current_subgraph);
        }

        // Now that every node ended up in a StableDiGraph, we can add edges by looking up the part index and node index for each node.
        // We keep track of edges that cross part boundaries in a separate hashmap keyed to a tuple of the source and target GraphNodes.
        #[allow(clippy::type_complexity)]
        let mut partition_edges: HashMap<(usize, usize), Vec<PartitionEdge>> = HashMap::new();

        for (source, target, edges) in graph.all_edges() {
            let (source_part_index, source_node_index) = node_to_nx.get(&source).unwrap();
            let (target_part_index, target_node_index) = node_to_nx.get(&target).unwrap();
            if source_part_index == target_part_index {
                subgraphs[*source_part_index].add_edge(
                    *source_node_index,
                    *target_node_index,
                    edges.clone(),
                );
            } else {
                partition_edges
                    .entry((*source_part_index, *target_part_index))
                    .or_default()
                    .push((source, target, edges.clone()));
            }
        }

        Partition {
            parts: subgraphs,
            inter_part_edges: partition_edges,
        }
    }
}

/// Partitions a graph into subgraphs, computes layouts for each subgraph, and grows the layout as needed.
/// The sugiyama crate uses recursion internally, and is susceptible to stack overflows if we don't constrain the size.
/// Hence we render the layout as subgraphs from a partition, based on which area the user is looking at.
/// To account for nodes close to the boundaries, we compute layouts for (at least) 3 partition subgraphs:
/// - The one currently scrolled into view, the one after, and the one before.
/// - The base_layout holds the boundaries as left_idx and right_idx.
/// - The functions expand_left and expand_right make the graph grow.
///
/// Public fields:
/// - `layout_graph` = a subgraph in DiGraphMap format that grows as we traverse the partition
/// - `node_positions` = growing map of nodes to their computed positions HashMap<GraphNode, (f64, f64)>
/// - `size` = layout size in number of nodes (number of layers, maximum number of nodes in any layer)
/// - `partition` = the partition we are working with (graph terminology: set of non-overlapping subgraphs)
/// - `left_idx` = the partition index of the leftmost subgraph in the current layout
/// - `right_idx` = the partition index of the rightmost subgraph in the current layout
///
/// Private fields:
/// - `_vertex_size` = closure that specifies the size of each node for the layout algorithm
/// - `_sugiyama_config` = configuration for the layout algorithm
/// - `_partial_layouts` = hashmap of partition index to individiual subgraph layouts
#[derive(Debug)]
pub struct BaseLayout {
    pub layout_graph: GenGraph,
    pub node_positions: HashMap<GraphNode, (f64, f64)>,
    pub size: (f64, f64),
    pub partition: Partition,
    pub left_idx: usize,
    pub right_idx: usize,
    _vertex_size: fn(_id: NodeIndex<u32>, _v: &GraphNode) -> (f64, f64),
    _sugiyama_config: rust_sugiyama::configure::Config,
    _partial_layouts: HashMap<usize, PartialLayout>,
}

impl BaseLayout {
    /// Create a new BaseLayout from a block graph.
    /// - `block_graph`: the graph to layout
    /// - Returns a new BaseLayout
    pub fn new(block_graph: &GenGraph) -> Self {
        // Check if the block_graph has a starting node
        let start_node_id = Node::get_start_node().id;
        let origin = block_graph
            .nodes()
            .find(|gn| gn.node_id == start_node_id)
            .map(|gn| (Node::get_start_node(), gn.sequence_start))
            .unwrap_or_else(|| {
                panic!("Could not find a starting node in the graph");
            });
        Self::with_origin(block_graph, origin)
    }

    /// Create a new BaseLayout with a specified origin (Node object, sequence position).
    /// - `block_graph`: the graph to layout
    /// - `origin`: the origin as (node, sequence position)
    /// - Returns a new BaseLayout
    pub fn with_origin(block_graph: &GenGraph, origin: (Node, i64)) -> Self {
        Self::with_origin_and_chunksize(block_graph, origin, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
    }

    /// Create a new BaseLayout with a specified chunk size.
    /// - `block_graph`: the graph to layout
    /// - `min_chunk_size`: the minimum size of a partition subgraph
    /// - `max_chunk_size`: the maximum size of a partition subgraph
    /// - Returns a new BaseLayout
    pub fn with_chunksize(
        block_graph: &GenGraph,
        min_chunk_size: usize,
        max_chunk_size: usize,
    ) -> Self {
        Self::with_origin_and_chunksize(
            block_graph,
            (Node::get_start_node(), 0),
            min_chunk_size,
            max_chunk_size,
        )
    }

    /// Create a new BaseLayout with a specified origin block and chunk size.
    /// - `block_graph`: the graph to layout
    /// - `origin`: the origin block coordinates (node, sequence position)
    /// - `min_chunk_size`: the minimum size of a partition subgraph
    /// - `max_chunk_size`: the maximum size of a partition subgraph
    /// - Returns a new BaseLayout
    pub fn with_origin_and_chunksize(
        block_graph: &GenGraph,
        origin: (Node, i64),
        min_chunk_size: usize,
        max_chunk_size: usize,
    ) -> Self {
        // Partition the graph at articulation points
        info!("Partitioning graph...");
        let partition = Partition::new(block_graph, min_chunk_size, max_chunk_size);

        // Find where in the partition the origin node is
        let origin_idx = partition
            .parts
            .iter()
            .position(|part| {
                part.node_indices().any(|idx| {
                    part.node_weight(idx)
                        .map(|gn| {
                            gn.node_id == origin.0.id
                                && gn.sequence_start <= origin.1
                                && gn.sequence_end >= origin.1
                        })
                        .unwrap_or(false)
                })
            })
            .unwrap_or_else(|| {
                panic!(
                    "Could not find the origin position {}:{} in the partition",
                    origin.0.id, origin.1
                );
            });

        // Make a GenGraph based on the first partition subgraph (StableGraph) we're asked to process
        // - we can't convert a StableGraph to a DiGraphMap directly, so we convert to a Graph first
        // - this will grow (and shrink) as we traverse the partition
        let temp_graph: GraphMap<GraphNode, Vec<GraphEdge>, Directed> =
            GraphMap::from_graph(partition.parts[origin_idx].clone().into());
        let mut subgraph = GenGraph::new();
        subgraph.extend(temp_graph.all_edges().map(|(a, b, c)| (a, b, c.clone())));

        // Set up the config for the layout algorithm
        // We set the vertex size to 1.0, 1.0 so that the layout algorithm does not take individual node size into account
        // since we do our own stretching/scaling later.
        let _vertex_size = |_id: NodeIndex<u32>, _v: &GraphNode| (1.0, 1.0);
        let _sugiyama_config = Config {
            vertex_spacing: 1.0,
            ..Default::default()
        };

        let mut base_layout = Self {
            layout_graph: subgraph,
            node_positions: HashMap::new(),
            size: (0.0, 0.0),
            partition,
            left_idx: origin_idx,
            right_idx: origin_idx,
            _vertex_size,
            _sugiyama_config,
            _partial_layouts: HashMap::new(),
        };

        // Compute the local layout
        // - In new(), you want it to block, but for later updates we should look into async
        // - Places the resulting layout in the _partial_layouts hashmap (for caching and future async updates)
        base_layout.run_sugiyama(origin_idx);

        // Make sure it succeeded
        assert!(
            base_layout._partial_layouts.contains_key(&origin_idx),
            "Failed to compute layout for partition index {origin_idx}"
        );

        // Because this is the first run, we can just move it over to the main layout, without modifications.
        // - note that the partial layouts include the dummy nodes to connect to the next partition subgraph,
        //   but no edges (for actual topology we have the layout_graph)
        let (sub_layout, width, height) = base_layout._partial_layouts.get(&origin_idx).unwrap();
        for (node, pos) in sub_layout {
            base_layout.node_positions.insert(*node, *pos);
        }

        // Update the size of the layout
        base_layout.size = (*width, *height);

        // Expand one unit left and right (bounds are handled in the expand functions)
        base_layout.expand_left();
        base_layout.expand_right();

        base_layout
    }

    /// Run the layout algorithm on a partition subgraph.
    /// Writes to the _sublayouts hashmap with future async rendering in mind.
    /// - `partition_idx`: which subgraph we want to compute a layout for
    /// - this is a separate function that writes to a `_partial_layouts` hashmap,
    ///   with future async updates in mind
    fn run_sugiyama(&mut self, partition_index: usize) {
        // We partitioned at articulation points so that it's easier to stitch layouts
        // together. However, partitioning does not duplicate the articulation nodes,
        // they end up on the right side of the boundary. We add dummy nodes to solve this.

        assert!(
            partition_index < self.partition.parts.len(),
            "Invalid partition index"
        );

        info!("Computing layout for partition index {partition_index}");

        // Make a mutable clone of the subgraph we want to layout
        // - this does not include the dummy nodes and edges to connect to the next subgraph
        let mut subgraph = self.partition.parts[partition_index].clone();

        // Add dummy nodes to tie into the next partition, unless we're at one of the boundaries
        if partition_index < self.partition.parts.len() - 1 {
            // Adds edges and nodes for each partition edge that departs here
            for (source, target, edges) in self
                .partition
                .inter_part_edges
                .get(&(partition_index, partition_index + 1))
                .unwrap()
            {
                // We have to use the stablegraph format, which means we need to
                // add the edges by node index, not by node value.
                let source_idx = subgraph
                    .node_indices()
                    .find(|&idx| subgraph.node_weight(idx).unwrap() == source)
                    .unwrap();

                // Check if dummy node already exists, add it if not
                let dummy_idx = subgraph
                    .node_indices()
                    .find(|&idx| subgraph.node_weight(idx).unwrap() == target)
                    .unwrap_or_else(|| subgraph.add_node(*target));

                subgraph.add_edge(source_idx, dummy_idx, edges.clone());
            }
        }

        // Run the layout algorithm
        let layouts =
            rust_sugiyama::from_graph(&subgraph, &self._vertex_size, &self._sugiyama_config);

        // Confirm that there is only one layout, which means that the graph is connected
        assert!(
            layouts.len() <= 1,
            "Disconnected graphs are not supported in the viewer currently."
        );
        assert_eq!(
            layouts.len(),
            1,
            "Could not compute layout for the selected partition."
        );

        let (idx_positions, width, height) = &layouts[0];

        // Transpose x and y (converts top-to-bottom to left-to-right) and remap to GraphNodes
        let node_positions: Vec<(GraphNode, (f64, f64))> = idx_positions
            .iter()
            .map(|(idx, (x, y))| (*subgraph.node_weight(*idx).unwrap(), (*y, *x)))
            .collect();

        // Store the layout along with its width and height transposed
        self._partial_layouts
            .insert(partition_index, (node_positions.clone(), *height, *width));
    }

    /// Expand the layout to the right by adding a new partition subgraph
    pub fn expand_right(&mut self) {
        self.expand(true);
    }

    /// Expand the layout to the left by adding a new partition subgraph
    pub fn expand_left(&mut self) {
        self.expand(false);
    }

    /// Generalized expand function that can expand to the left or right
    fn expand(&mut self, rightwards: bool) {
        // Get the next partition index
        let next_idx = if rightwards && self.right_idx < (self.partition.parts.len() - 1) {
            self.right_idx + 1
        } else if !rightwards && self.left_idx > 0 {
            self.left_idx - 1
        } else {
            return;
        };

        // Compute the layout for the new partition subgraph (with dummy nodes)
        // - dummy nodes are added in run_sugiyama
        // - this will update the _partial_layouts hashmap
        // - currently this blocks the main thread, we should make this async
        self.run_sugiyama(next_idx);

        assert!(
            self._partial_layouts.contains_key(&next_idx),
            "Failed to compute layout for partition index {next_idx}"
        );

        // Get the partial layout of the new partition subgraph
        // (this *does* include dummy nodes, but no edges)
        let (new_node_positions, width, height) = self._partial_layouts.get(&next_idx).unwrap();

        // The partition itself do not include dummy nodes,
        // but *does* include edges to them in the inter_part_edges hashmap
        let dummy_nodes = if rightwards {
            // Find the dummy nodes of the partition subgraph to the left
            let mut nodes = self
                .partition
                .inter_part_edges
                .get(&(self.right_idx, next_idx))
                .unwrap()
                .iter()
                .map(|(_, target, _)| *target)
                .collect::<Vec<_>>();
            nodes.sort();
            nodes.dedup();
            nodes
        } else {
            // Find the dummy nodes of the partition subgraph to the right
            let mut nodes = self
                .partition
                .inter_part_edges
                .get(&(next_idx, self.left_idx))
                .unwrap()
                .iter()
                .map(|(_, target, _)| *target)
                .collect::<Vec<_>>();
            nodes.sort();
            nodes.dedup();
            nodes
        };

        // Ideally we have just one, the articulation point, but we'll have more if we had to split early
        if dummy_nodes.len() > 1 {
            warn!(
                "Partition index {} and {} could not be joined together cleanly. Dummy nodes: {:?}",
                if rightwards { self.right_idx } else { next_idx },
                if rightwards { next_idx } else { self.left_idx },
                dummy_nodes
            );
        };

        // The first dummy node is used as the point of reference around which we'll move the new layout
        // We can get it from the already computed layout since it's a hashmap keyed to GraphNode,
        // and we cloned GraphNodes to make the dummy nodes
        let stitch_dummy = *self.node_positions.get(&dummy_nodes[0]).unwrap();
        let stitch_real = new_node_positions
            .iter()
            .find(|(node, _)| node == &dummy_nodes[0])
            .unwrap()
            .1;
        // stitch_real + offset = stitch_dummy
        let offset = (
            stitch_dummy.0 - stitch_real.0,
            stitch_dummy.1 - stitch_real.1,
        );

        // Shift the new layout so that its origin is the same as the reference node,
        // and add it to the main layout.
        // THERE'S A BUG HERE:
        if rightwards {
            for (node, pos) in new_node_positions {
                let new_pos = (pos.0 + offset.0, pos.1 + offset.1);
                self.node_positions.insert(*node, new_pos);
            }
        } else {
            for (node, pos) in new_node_positions {
                let new_pos = (pos.0 - offset.0, pos.1 - offset.1);
                self.node_positions.insert(*node, new_pos);
            }
        }

        // Grow the layout graph by including:
        // - edges from the new partition subgraph
        // - inter-partition edges
        // This does require converting from the stablegraph format to graphmap format
        let subgraph = self.partition.parts[next_idx].clone();
        let temp_graph: GraphMap<GraphNode, Vec<GraphEdge>, Directed> =
            GraphMap::from_graph(subgraph.into());
        let mut subgraph_map = GenGraph::new();
        subgraph_map.extend(temp_graph.all_edges().map(|(a, b, c)| (a, b, c.clone())));
        self.layout_graph.extend(subgraph_map.all_edges());

        // The inter-partition edges are not included in the subgraph, so we add them separately
        if rightwards {
            for (source, target, edges) in self
                .partition
                .inter_part_edges
                .get(&(self.right_idx, next_idx))
                .unwrap()
            {
                self.layout_graph.add_edge(*source, *target, edges.clone());
                self.right_idx = next_idx;
            }
        } else {
            for (source, target, edges) in self
                .partition
                .inter_part_edges
                .get(&(next_idx, self.left_idx))
                .unwrap()
            {
                self.layout_graph.add_edge(*source, *target, edges.clone());
                self.left_idx = next_idx;
            }
        }

        // Update the size of the main layout (note that these are the number of nodes, not the width and height in cell units)
        self.size.0 += *width - 1.0; // -1.0 because we added a dummy node to stitch the layouts together
        self.size.1 = f64::max(self.size.1, *height);
    }
}

/// Holds processed and scaled layout data, but not the actual sequences.
/// - `lines` = pairs of coordinates for each edge.
/// - `labels` = starting and ending coordinates for each label.
///
/// The raw layout from the Sugiyama algorithm is processed as follow:
/// - The coordinates are rounded to the nearest integer and transposed to go from top-to-bottom to left-to-right.
/// - Each block is assigned a layer (or rank) based on its x-coordinate (transposed y-coordinate).
/// - The width of each layer is determined by the widest label in that layer.
/// - The distance between layers is scaled horizontally and vertically
#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub struct ScaledLayout {
    pub lines: HashMap<(GraphNode, GraphNode), ((f64, f64), (f64, f64))>, // Edge -> (start_coord, end_coord)
    pub labels: HashMap<GraphNode, ((f64, f64), (f64, f64))>, // Node -> (start_coord, end_coord)
}

impl ScaledLayout {
    pub fn from_base_layout(base_layout: &BaseLayout, parameters: &PlotParameters) -> Self {
        let mut layout = ScaledLayout {
            lines: HashMap::new(),
            labels: HashMap::new(),
        };
        layout.refresh(base_layout, parameters);
        layout
    }

    pub fn refresh(&mut self, base_layout: &BaseLayout, parameters: &PlotParameters) {
        // Scale the overall layout and round to integer coordinates
        let scale_x = parameters.scale as f64;
        let scale_y = parameters.scale as f64 * parameters.aspect_ratio as f64;

        let mut working_layout: Vec<(GraphNode, (i64, i64))> = base_layout
            .node_positions
            .iter()
            .map(|(node, (x, y))| {
                (
                    *node,
                    ((x * scale_x).round() as i64, (y * scale_y).round() as i64),
                )
            })
            .collect();

        // We don't need to stretch the layout if the requested label width is too small
        if parameters.label_width < 5 {
            warn!(
                "Requested label width is too small to shrink the labels, falling back to view without sequences."
            );
            self.labels = working_layout
                .iter()
                .map(|(node, (x, y))| {
                    (
                        *node,
                        ((*x as f64, *y as f64), (*x as f64 + 1.0, *y as f64)),
                    )
                })
                .collect();
        } else {
            // To stretch, we first sort the layout by x-coordinate so that we can group the blocks by rank
            working_layout.sort_by(|a, b| a.1.0.cmp(&b.1.0));

            // Loop over the sorted layout and group the blocks by rank by keeping track of the x-coordinate
            let mut current_x = working_layout[0].1.0;
            let mut current_layer: Vec<(GraphNode, i64, i64)> = Vec::new(); // node, label_width, y-coordinate

            // Initial values:
            let mut layer_width = 1;
            let mut cumulative_offset = 0;

            for (node, (x, y)) in working_layout.iter() {
                let label_width = std::cmp::min(
                    node.sequence_end - node.sequence_start,
                    parameters.label_width as i64,
                );

                if *x == current_x {
                    // This means we are still in the same layer
                    // Keep a tally of the maximum label width
                    layer_width = std::cmp::max(layer_width, label_width);

                    // Add the block to the current layer vector
                    current_layer.push((*node, label_width, *y));
                } else {
                    // We switched to a new layer
                    // Loop over the current layer (now previous) and:
                    // - increment the x-coordinate by the cumulative offset so far
                    // - horizontally center the block in its layer
                    for (node, label_width, y) in current_layer {
                        let centering_offset =
                            ((layer_width - label_width) as f64 / 2.0).round() as i64;
                        let x = current_x + centering_offset + cumulative_offset;
                        self.labels.insert(
                            node,
                            ((x as f64, y as f64), ((x + label_width) as f64, y as f64)),
                        );
                    }
                    // Increment the cumulative offset for the next layer by the width of the current layer
                    cumulative_offset += layer_width;

                    // Reset the layer width and the current layer
                    layer_width = label_width;
                    current_layer = vec![(*node, label_width, *y)];
                    current_x = *x;
                }
            }
            // Loop over the last layer (wasn't processed yet)
            for (node, label_width, y) in current_layer {
                let centering_offset = ((layer_width - label_width) as f64 / 2.0).round() as i64;
                let x = current_x + centering_offset + cumulative_offset;
                self.labels.insert(
                    node,
                    ((x as f64, y as f64), ((x + label_width) as f64, y as f64)),
                );
            }
        }

        // Recalculate all the edges so they meet labels on the sides instead of the center
        self.lines = base_layout
            .layout_graph
            .all_edges()
            .filter(|(source, target, _)| {
                // Filter out edges that connect to start or end nodes
                !is_terminal(source.node_id) && !is_terminal(target.node_id)
            })
            .map(|(source, target, _)| {
                let source_coord = self
                    .labels
                    .get(&source)
                    .map(|(_, (x2, y2))| (*x2, *y2))
                    .unwrap();
                let target_coord = self
                    .labels
                    .get(&target)
                    .map(|((x1, y1), _)| (*x1 - 1.0, *y1))
                    .unwrap();
                ((source, target), (source_coord, target_coord))
            })
            .collect();

        // Pretty print the lines
        for ((source, target), (source_coord, target_coord)) in self.lines.iter() {
            debug!(
                "Edge {}-{} from ({:.1},{:.1}) to ({:.1},{:.1})",
                source.block_id,
                target.block_id,
                source_coord.0,
                source_coord.1,
                target_coord.0,
                target_coord.1
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, PATH_END_NODE_ID, PATH_START_NODE_ID, strand::Strand};
    use gen_graph::{GraphEdge, GraphNode};
    use itertools::Itertools;
    use petgraph::graphmap::DiGraphMap;

    use super::*;

    fn get_id(i: i32) -> HashId {
        if i == 1 {
            PATH_START_NODE_ID
        } else if i == 2 {
            PATH_END_NODE_ID
        } else {
            HashId::pad_str(i)
        }
    }

    fn make_test_graph(edges: Vec<(i32, i32)>, nodes: Option<Vec<GraphNode>>) -> GenGraph {
        // Create default nodes if none provided
        let nodes = nodes.unwrap_or_else(|| {
            edges
                .iter()
                .flat_map(|(s, t)| vec![*s, *t])
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .map(|id| GraphNode {
                    block_id: id as i64,
                    node_id: get_id(id),
                    sequence_start: 0,
                    sequence_end: 10,
                })
                .collect()
        });

        let temp_graph = DiGraphMap::from_edges(edges.iter().map(|(s, t)| {
            (
                *nodes.iter().find(|gn| gn.block_id == *s as i64).unwrap(),
                *nodes.iter().find(|gn| gn.block_id == *t as i64).unwrap(),
                vec![GraphEdge {
                    edge_id: HashId::pad_str(0),
                    source_strand: Strand::Forward,
                    target_strand: Strand::Forward,
                    chromosome_index: 0,
                    phased: 0,
                    created_on: 0,
                }],
            )
        }));
        let mut wrapped_graph = GenGraph::new();
        wrapped_graph.extend(
            temp_graph
                .all_edges()
                .map(|(src, dest, weight)| (src, dest, weight.clone())),
        );
        wrapped_graph
    }

    #[test]
    fn test_make_test_graph() {
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        let graph = make_test_graph(edges, None);
        assert_eq!(graph.node_count(), 6);
        assert_eq!(graph.edge_count(), 5);
    }

    #[test]
    fn test_find_articulation_points_simple() {
        // 1 -> 2 -> 3
        //      |
        //      v
        //      4 -> 5
        let edges = vec![(1, 2), (2, 3), (2, 4), (4, 5)];
        let graph = make_test_graph(edges, None);
        let ap = find_articulation_points(&graph)
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>();
        let expected = vec![get_id(2), get_id(4)];
        assert_eq!(ap, expected);
    }

    #[test]
    fn test_find_articulation_points_multiple_ap() {
        // 1 -> 2 -> 3
        //      |    |
        //      v    v
        //      4 -> 5 -> 6
        let edges = vec![(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)];
        let graph = make_test_graph(edges, None);
        let ap = find_articulation_points(&graph)
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>();
        let expected = vec![get_id(2), get_id(5)];
        assert_eq!(ap, expected);
    }

    #[test]
    fn test_find_articulation_points_single_node() {
        let graph = make_test_graph(vec![(1, 1)], None);

        let ap = find_articulation_points(&graph);
        let expected = vec![];
        assert_eq!(ap, expected);
    }

    #[test]
    fn test_find_articulation_points_disconnected_graph() {
        // 1 -> 2
        // 3 -> 4
        let graph = make_test_graph(vec![(1, 2), (3, 4)], None);

        let ap = find_articulation_points(&graph);
        let expected = vec![];
        assert_eq!(ap, expected);
    }

    #[test]
    fn test_partition_new() {
        // 1 -> 2 -> 3
        //      |    |
        //      v    v
        //      4 -> 5 -> 6
        let edges = vec![(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)];
        let graph = make_test_graph(edges, None);
        let partition = Partition::new(&graph, 3, 10);
        assert_eq!(partition.parts.len(), 2);
        assert_eq!(partition.inter_part_edges.len(), 1);
    }

    #[test]
    fn test_partition_deletion() {
        // 1 -> 2 ---|
        //      |    |
        //      v    v
        //      4 -> 5 -> 6
        let edges = vec![(1, 2), (2, 5), (2, 4), (4, 5), (5, 6)];
        let graph = make_test_graph(edges, None);
        let partition = Partition::new(&graph, 2, 10);
        assert_eq!(partition.parts.len(), 2);
        assert_eq!(
            partition.parts[0].node_count(),
            3,
            "Partition 0 has nodes: {:?}",
            partition.parts[0]
                .node_weights()
                .map(|n| n.block_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            partition.parts[1].node_count(),
            2,
            "Partition 1 has nodes: {:?}",
            partition.parts[1]
                .node_weights()
                .map(|n| n.block_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(partition.inter_part_edges.len(), 1);
    }

    #[test]
    fn test_partition_new_small() {
        // 1 -> 2 -> 3
        //      |    |
        //      v    v
        //      4 -> 5 -> 6
        let edges = vec![(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)];
        let graph = make_test_graph(edges, None);
        let partition = Partition::new(&graph, 1, 10);
        assert_eq!(partition.parts.len(), 3);
        assert_eq!(partition.inter_part_edges.len(), 2);
    }

    #[test]
    fn test_partition_new_force_close() {
        // 1 -> 2 -> 3
        //      |    |
        //      v    v
        //      4 -> 5 -> 6
        let edges = vec![(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)];
        let graph = make_test_graph(edges, None);
        let partition = Partition::new(&graph, 1, 2);
        assert_eq!(partition.parts.len(), 4);
        assert_eq!(partition.inter_part_edges.len(), 4);
    }

    #[test]
    fn test_base_layout_new() {
        // 0 -> 1 -> 2
        //      |    |
        //      v    v
        //      3 -> 4 -> 5
        let edges = vec![(0, 1), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5)];
        let graph = make_test_graph(edges, None);
        let base_layout = BaseLayout::new(&graph);
        assert_eq!(base_layout.node_positions.len(), 6);
        assert_eq!(base_layout.size, (5.0, 2.0)); // (number of layers, largest layer width)
    }

    #[test]
    fn test_base_layout_with_chunksize() {
        // 0 -> 1 -> 2
        //      |    |
        //      v    v
        //      3 -> 4 -> 5
        let edges = vec![(0, 1), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5)];
        let graph = make_test_graph(edges, None);
        let base_layout = BaseLayout::with_chunksize(&graph, 2, 10);
        assert_eq!(base_layout.partition.parts.len(), 2);
        assert_eq!(base_layout.node_positions.len(), 6);
        assert_eq!(base_layout.size, (5.0, 2.0));
    }

    #[test]
    fn test_scaled_layout_new_no_zoom() {
        // Create a test subgraph manually
        let n0 = GraphNode {
            block_id: 0,
            node_id: get_id(0),
            sequence_start: 0,
            sequence_end: 10,
        };
        let n1 = GraphNode {
            block_id: 1,
            node_id: get_id(1),
            sequence_start: 0,
            sequence_end: 20,
        };
        let n2 = GraphNode {
            block_id: 2,
            node_id: get_id(2),
            sequence_start: 0,
            sequence_end: 40,
        };

        let graph = make_test_graph(vec![(0, 1), (1, 2)], Some(vec![n0, n1, n2]));
        let base_layout = BaseLayout::new(&graph);
        let parameters = PlotParameters {
            label_width: 5,
            scale: 1,
            aspect_ratio: 1.0,
            ..Default::default()
        };
        let scaled_layout = ScaledLayout::from_base_layout(&base_layout, &parameters);

        let sorted_layout = scaled_layout
            .labels
            .iter()
            .sorted_by_key(|(node, _)| node.block_id)
            .map(|(node, (start, end))| (node.block_id, (*start, *end)))
            .collect::<Vec<_>>();

        let expected_layout = vec![
            (0, ((0.0, 0.0), (5.0, 0.0))),
            (1, ((7.0, 0.0), (12.0, 0.0))),
            (2, ((14.0, 0.0), (19.0, 0.0))),
        ];
        assert_eq!(sorted_layout, expected_layout);
    }

    #[test]
    fn test_scaled_layout_new_unlabeled() {
        // 0 -> 1 -> 2
        //      |    |
        //      v    v
        //      3 -> 4 -> 5
        let edges = vec![(0, 1), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5)];
        // Generate 6 nodes with sequence length of 10
        let mut nodes = (0..6)
            .map(|i| GraphNode {
                block_id: i,
                node_id: get_id(i as i32),
                sequence_start: 0,
                sequence_end: 10,
            })
            .collect::<Vec<_>>();
        // Set the first node to the start node
        nodes[0].node_id = Node::get_start_node().id;

        let graph = make_test_graph(edges, Some(nodes));
        let base_layout = BaseLayout::with_chunksize(&graph, 2, 10);
        let parameters = PlotParameters {
            label_width: 1,
            scale: 10,
            aspect_ratio: 1.0,
            ..Default::default()
        };
        let scaled_layout = ScaledLayout::from_base_layout(&base_layout, &parameters);

        let sorted_layout = scaled_layout
            .labels
            .iter()
            .sorted_by_key(|(node, _)| node.block_id)
            .map(|(node, (start, end))| (node.block_id, (*start, *end)))
            .collect::<Vec<_>>();

        let expected_layout = [
            (0, ((0.0, 10.0), (1.0, 10.0))),
            (1, ((20.0, 10.0), (21.0, 10.0))),
            (2, ((40.0, 20.0), (41.0, 20.0))),
            (3, ((40.0, 0.0), (41.0, 0.0))),
            (4, ((60.0, 10.0), (61.0, 10.0))),
            (5, ((80.0, 10.0), (81.0, 10.0))),
        ];
        assert_eq!(sorted_layout, expected_layout);
    }

    #[test]
    fn test_scaled_layout_new_truncations() {
        // 0 -> 1 -> 2
        //      |    |
        //      v    v
        //      3 -> 4 -> 5
        let edges = vec![(0, 1), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5)];
        // Generate 6 nodes with sequence length of 10, 20, 30, ...
        let nodes = (0..6)
            .map(|i| GraphNode {
                block_id: i,
                node_id: get_id(i as i32),
                sequence_start: 0,
                sequence_end: (i + 1) * 10,
            })
            .collect::<Vec<_>>();

        let graph = make_test_graph(edges, Some(nodes));
        let base_layout = BaseLayout::with_chunksize(&graph, 2, 10);
        let parameters = PlotParameters {
            label_width: 15,
            scale: 1,
            aspect_ratio: 1.0,
            ..Default::default()
        };
        let scaled_layout = ScaledLayout::from_base_layout(&base_layout, &parameters);

        let sorted_layout = scaled_layout
            .labels
            .iter()
            .sorted_by_key(|(node, _)| node.block_id)
            .map(|(node, (start, end))| (node.block_id, (*start, *end)))
            .collect::<Vec<_>>();

        let expected_layout = vec![
            (0, ((0.0, 1.0), (10.0, 1.0))),
            (1, ((12.0, 1.0), (27.0, 1.0))),
            (2, ((29.0, 2.0), (44.0, 2.0))),
            (3, ((29.0, 0.0), (44.0, 0.0))),
            (4, ((46.0, 1.0), (61.0, 1.0))),
            (5, ((63.0, 1.0), (78.0, 1.0))),
        ];
        assert_eq!(sorted_layout, expected_layout);
    }

    #[test]
    fn test_base_layout_expand_right() {
        // 0 -> 1 -> 2 -> 3 -> ... -> 100
        let edges = (0..100).map(|i| (i, i + 1)).collect::<Vec<_>>();
        let graph = make_test_graph(edges, None);
        let ref_layout = BaseLayout::with_chunksize(&graph, usize::MAX, usize::MAX); // no chunking
        assert_eq!(ref_layout.partition.parts.len(), 1);

        // Chunked layout
        let mut base_layout = BaseLayout::with_chunksize(&graph, 50, usize::MAX);
        assert_eq!(base_layout.partition.parts.len(), 2);

        // Expand the layout to the right
        base_layout.expand_right();

        // Check that the topology is the same as the original graph
        assert_eq!(
            base_layout.layout_graph.node_count(),
            graph.node_count(),
            "Node count mismatch"
        );
        assert_eq!(
            base_layout.layout_graph.edge_count(),
            graph.edge_count(),
            "Edge count mismatch"
        );

        // Check that the coordinates are the same between the reference and the chunked layout
        for node in graph.nodes() {
            assert_eq!(
                ref_layout.node_positions[&node],
                base_layout.node_positions[&node]
            );
        }
    }

    #[test]
    fn test_base_layout_expand_right_complex() {
        // 0 -> 1 -> 10 -> 11 -> 20 -> 21 -> 30 -> 31 -> 40 ...
        //   -> 2 ->       12          22          32       ...
        //   -> 3 ->       13          23          33       ...
        //   -> 4 ->       14          24          34       ...
        //   -> 5 ->       15          25          35       ...
        //   -> 6 ->       16          26          36       ...
        //   -> 7 ->       17          27          37       ...
        //   -> 8 ->       18          28          38       ...
        //   -> 9 ->       19          29          39       ...

        let num_buckets = 10; // A bucket is a variable locus here
        let mut edges = Vec::new();
        for j in 0..num_buckets {
            edges.append(
                &mut ((1 + j * 10)..(10 + j * 10))
                    .map(|i| (j * 10, i))
                    .collect::<Vec<_>>(),
            );
            edges.append(
                &mut ((1 + j * 10)..(10 + j * 10))
                    .map(|i| (i, 10 + j * 10))
                    .collect::<Vec<_>>(),
            );
        }

        let graph = make_test_graph(edges, None);
        let ref_layout = BaseLayout::with_chunksize(&graph, usize::MAX, usize::MAX);
        assert_eq!(ref_layout.partition.parts.len(), 1);

        let mut base_layout = BaseLayout::with_chunksize(&graph, 30, usize::MAX);
        //assert_eq!(base_layout.partition.parts.len(), num_cols as usize);

        // Expand the layout all the way to the right
        for _ in 0..base_layout.partition.parts.len() {
            base_layout.expand_right();
        }

        // Check that the topology is the same as the original graph
        assert_eq!(
            base_layout.layout_graph.node_count(),
            graph.node_count(),
            "Node count mismatch"
        );
        assert_eq!(
            base_layout.layout_graph.edge_count(),
            graph.edge_count(),
            "Edge count mismatch"
        );

        // Check that the coordinates are the same between the reference and the chunked layout
        for node in graph.nodes() {
            assert_eq!(
                ref_layout.node_positions[&node],
                base_layout.node_positions[&node]
            );
        }
    }
}
