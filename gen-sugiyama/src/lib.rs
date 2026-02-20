//! The implementation roughly follows sugiyamas algorithm for creating
//! a layered graph layout.
//!
//! Usually Sugiyamas algorithm consists of 4 Phases:
//! 1. Remove Cycles
//! 2. Assign each vertex to a rank/layer
//! 3. Reorder vertices in each rank to reduce crossings
//! 4. Calculate the final coordinates.
//!
//! Currently, phase 2 to 4 are implemented, Cycle removal might be added at
//! a later time.
//!
//! The whole algorithm roughly follows the 1993 paper "A technique for drawing
//! directed graphs" by Gansner et al. It can be found
//! [here](https://ieeexplore.ieee.org/document/221135).
//!
//! See the submodules for each phase for more details on the implementation
//! and references used.
use std::collections::{BTreeMap, HashMap};

use log::info;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

// Internal module dependencies for self-contained algorithm
mod config;
pub mod types;
mod util;

// Re-export types for external use
pub use config::{Config, CrossingMinimization, RankingType, VERTEX_SPACING_DEFAULT};
// Algorithm phases
#[allow(unused_imports)]
use p0_cycle_removal as p0;
use p1_layering as p1;
use p2_reduce_crossings as p2;
use p3_calculate_coordinates as p3;

//use self::p3_calculate_coordinates::VDir;

mod p0_cycle_removal;
mod p1_layering;
mod p2_reduce_crossings;
mod p3_calculate_coordinates;

// Changes made to the original implementation:
// - Changed id (usize) to original_id:Option<NodeIndex> that refers to
// the original node in the graph and None for dummy nodes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub input_node_idx: Option<NodeIndex>,
    /// Optional stable ordering preference used only as a *tiebreaker* during
    /// crossing reduction when multiple orders yield equal heuristic scores.
    ///
    /// Default is 0. Positive/negative values allow callers to bias tied
    /// vertices to one side or the other.
    sort_bias: i32,
    size: (f64, f64),
    rank: i32,
    pos: usize,
    low: u32,
    lim: u32,
    parent: Option<NodeIndex>,
    is_tree_vertex: bool,
    is_dummy: bool,
    root: NodeIndex,
    align: NodeIndex,
    shift: f64,
    sink: NodeIndex,
    block_max_vertex_width: f64,
}

impl Vertex {
    pub fn new(original_id: NodeIndex) -> Self {
        Self {
            input_node_idx: Some(original_id),
            ..Default::default()
        }
    }

    pub fn new_dummy() -> Self {
        Self {
            input_node_idx: None,
            is_dummy: true,
            ..Default::default()
        }
    }

    pub fn with_sort_bias(mut self, sort_bias: i32) -> Self {
        self.sort_bias = sort_bias;
        self
    }

    pub fn set_sort_bias(&mut self, sort_bias: i32) {
        self.sort_bias = sort_bias;
    }

    pub fn sort_bias(&self) -> i32 {
        self.sort_bias
    }

    pub fn set_size(&mut self, (width, height): (u64, u64), vertex_spacing: f64) {
        // Vertex spacing isn't actually taken into account by itself,
        // it's just padding to the size of the vertex. So if we change the
        // size of the vertex, we need include the padding as well.
        // TODO: don't hardcode the transposition here either
        self.size = (
            (height.max(1) as f64 + vertex_spacing),
            (width.max(1) as f64 + vertex_spacing),
        );
    }

    pub fn get_size(&self, vertex_spacing: f64) -> (u64, u64) {
        // Inverse operation of set_size
        let (width, height) = self.size;
        // TODO: don't hardcode the transposition here either
        (
            (height - vertex_spacing).round() as u64,
            (width - vertex_spacing).round() as u64,
        )
    }

    pub fn is_dummy(&self) -> bool {
        self.is_dummy
    }

    pub fn get_rank(&self) -> i32 {
        self.rank
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            input_node_idx: None,
            sort_bias: 0,
            size: (1.0, 1.0),
            rank: 0,
            pos: 0,
            low: 0,
            lim: 0,
            parent: None,
            is_tree_vertex: false,
            is_dummy: false,
            root: 0.into(),
            align: 0.into(),
            shift: f64::INFINITY,
            sink: 0.into(),
            block_max_vertex_width: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub weight: i32,
    cut_value: Option<i32>,
    is_tree_edge: bool,
    has_type_1_conflict: bool,
    pub input_node_idx_pair: Option<(NodeIndex, NodeIndex)>,
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            weight: 1,
            cut_value: None,
            is_tree_edge: false,
            has_type_1_conflict: false,
            input_node_idx_pair: None,
        }
    }
}

impl Edge {
    pub fn with_label(self, (tail, head): (NodeIndex, NodeIndex)) -> Self {
        Self {
            input_node_idx_pair: Some((tail, head)),
            ..self
        }
    }
}

/// Runs the Sugiyama algorithm and returns the layers and vertex graph.
/// This is the computationally expensive part that only needs to be run once.
pub fn run_sugiyama_algorithm(
    vertex_graph: &mut StableDiGraph<Vertex, Edge>,
    config: &Config,
) -> Vec<Vec<NodeIndex>> {
    info!(target: "layouting", "Start building layout");
    info!(target: "layouting", "Configuration is: {:?}", config);

    // Phase 0: Cycle Removal
    info!(target: "layouting", "Skipping phase 0: Cycle Removal");
    //info!(target: "layouting", "Executing phase 0: Cycle Removal");
    //let _reversed_edges = p0::remove_cycles(&mut vertex_graph);

    // Phase 1: Layering/Ranking
    info!(target: "layouting", "Executing phase 1: Ranking");
    p1::rank(
        vertex_graph,
        config.minimum_length as i32,
        config.ranking_type,
    );

    // Phase 2: Reorder vertices in ranks to reduce crossings.
    info!(target: "layouting", "Executing phase 2: Crossing Reduction");
    info!(target: "layouting",
        "dummy vertex size: {:?}, heuristic for crossing minimization: {:?}, using transpose: {}",
        config.dummy_size,
        config.c_minimization,
        config.transpose
    );

    let _dropped_edges = p2::insert_dummy_vertices(
        vertex_graph,
        config.minimum_length as i32,
        config.dummy_size,
    );

    let mut layers = p2::ordering(vertex_graph, config.c_minimization, config.transpose);
    if !config.dummy_vertices {
        p2::remove_dummy_vertices(vertex_graph, &mut layers);
    }

    layers
}

/// Assigns final coordinates to vertices using the computed algorithm state and a label dimensions map.
/// This can be called multiple times with different size maps to create different layout variants.
pub fn assign_coordinates(
    layers: &mut [Vec<NodeIndex>],
    vertex_graph: &mut StableDiGraph<Vertex, Edge>,
) -> Vec<(NodeIndex, (i64, i64))> {
    // Phase 3: Coordinate Assignment
    info!(target: "layouting", "Executing phase 3: Coordinate Calculation on a vertex graph with {} nodes and {} edges", vertex_graph.node_count(), vertex_graph.edge_count());
    let mut layouts = p3::create_layouts(vertex_graph, layers);
    p3::align_to_smallest_width_layout(&mut layouts);

    // Calculate and normalize X coordinates
    let mut x_coordinates = p3::calculate_relative_coords(layouts);
    let min = x_coordinates
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .1;
    for (_, c) in &mut x_coordinates {
        *c -= min;
    }

    // Calculate Y coordinates
    let mut rank_to_max_height = BTreeMap::<i32, f64>::new();
    for vertex in vertex_graph.node_weights() {
        let max = rank_to_max_height.entry(vertex.rank).or_default();
        *max = max.max(vertex.size.1);
    }

    let mut rank_to_y_offset = HashMap::new();
    let mut current_rank_top_offset = *rank_to_max_height.iter().next().unwrap().1 * -0.5;
    for (rank, max_height) in rank_to_max_height {
        rank_to_y_offset.insert(rank, current_rank_top_offset + max_height * 0.5);
        current_rank_top_offset += max_height;
    }

    // Compute provisional coordinates, transposed and rounded to the nearest integer.
    // TODO: don't hardcode the horizontal vs vertical orientation, use a config option

    x_coordinates
        .into_iter()
        .map(|(v, x)| {
            (
                v,
                (x, *rank_to_y_offset.get(&vertex_graph[v].rank).unwrap()),
            )
        })
        // if we round() instead of floor() the layout will be off by 1 cell
        .map(|(i, (x, y))| (i, (y.floor() as i64, x.floor() as i64)))
        .collect::<Vec<(NodeIndex, (i64, i64))>>()
}

fn slack(graph: &StableDiGraph<Vertex, Edge>, edge: EdgeIndex, minimum_length: i32) -> i32 {
    let (tail, head) = graph.edge_endpoints(edge).unwrap();
    graph[head].rank - graph[tail].rank - minimum_length
}
