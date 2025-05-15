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

// Suppress clippy::type_complexity warnings, as this is a graph layout algorithm
// that inherently deals with complex types and data structures.
#![allow(clippy::type_complexity)]

use std::collections::{BTreeMap, HashMap};

use log::{debug, info};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

use super::configure::{Config, CrossingMinimization, RankingType};
use super::util::weakly_connected_components;

pub mod p0_cycle_removal;
pub mod p1_layering;
pub mod p2_reduce_crossings;
pub mod p3_calculate_coordinates;

// Re-export the functions we need
pub use p0_cycle_removal::remove_cycles;
pub use p1_layering::rank;
pub use p2_reduce_crossings::{insert_dummy_vertices, ordering, remove_dummy_vertices};
pub use p3_calculate_coordinates::{
    align_to_smallest_width_layout, calculate_relative_coords, create_layouts, VDir,
};

type LayoutResult = (Vec<(usize, (f64, f64))>, f64, f64);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub id: usize,
    pub(crate) size: (f64, f64),
    pub(crate) rank: i32,
    pub(crate) pos: usize,
    pub(crate) low: u32,
    pub(crate) lim: u32,
    pub(crate) parent: Option<NodeIndex>,
    pub(crate) is_tree_vertex: bool,
    pub(crate) is_dummy: bool,
    pub(crate) root: NodeIndex,
    pub(crate) align: NodeIndex,
    pub(crate) shift: f64,
    pub(crate) sink: NodeIndex,
    pub(crate) block_max_vertex_width: f64,
    pub(crate) x: i32,
    pub(crate) y: i32,
}

impl Vertex {
    pub fn new(id: usize, size: (f64, f64)) -> Self {
        Self {
            id,
            size,
            ..Default::default()
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            id: 0,
            size: (0.0, 0.0),
            rank: 0,
            x: 0,
            y: 0,
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
    pub(crate) weight: i32,
    pub(crate) cut_value: Option<i32>,
    pub(crate) is_tree_edge: bool,
    pub(crate) has_type_1_conflict: bool,
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            weight: 1,
            cut_value: None,
            is_tree_edge: false,
            has_type_1_conflict: false,
        }
    }
}

pub fn start(
    mut graph: StableDiGraph<Vertex, Edge>,
    config: &Config,
) -> Vec<(Vec<(usize, (f64, f64))>, f64, f64)> {
    init_graph(&mut graph);
    weakly_connected_components(graph)
        .into_iter()
        .map(|g| build_layout(g, config))
        .collect()
}

fn init_graph(graph: &mut StableDiGraph<Vertex, Edge>) {
    info!("Initializing graphs vertex weights");
    for id in graph.node_indices().collect::<Vec<_>>() {
        graph[id].id = id.index();
        graph[id].root = id;
        graph[id].align = id;
        graph[id].sink = id;
    }
}

fn build_layout(
    mut graph: StableDiGraph<Vertex, Edge>,
    config: &Config,
) -> LayoutResult {
    info!(target: "layouting", "Start building layout");
    info!(target: "layouting", "Configuration is: {:?}", config);

    // Treat the vertex spacing as just additional padding in each node. Each node will then take
    // 50% of the "responsibility" of the vertex spacing. This does however mean that dummy vertices
    // will have a gap of 50% of the vertex spacing between them and the next and previous vertex.
    for vertex in graph.node_weights_mut() {
        vertex.size.0 += config.vertex_spacing;
        vertex.size.1 += config.vertex_spacing;
    }

    // we don't remember the edges that where reversed for now, since they are
    // currently not needed
    let _ = execute_phase_0(&mut graph);

    execute_phase_1(
        &mut graph,
        config.minimum_length as i32,
        config.ranking_type,
    );

    let layers = execute_phase_2(
        &mut graph,
        config.minimum_length as i32,
        config.dummy_vertices.then_some(config.dummy_size),
        config.c_minimization,
        config.transpose,
    );

    let layout = execute_phase_3(&mut graph, layers);
    debug!(target: "layouting", "Node coordinates: {:?}\nwidth: {}, height:{}", 
        layout.0,
        layout.1,
        layout.2
    );
    layout
}

fn execute_phase_0(graph: &mut StableDiGraph<Vertex, Edge>) -> Vec<EdgeIndex> {
    info!(target: "layouting", "Executing phase 0: Cycle Removal");
    remove_cycles(graph)
}

/// Assign each vertex a rank
fn execute_phase_1(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    ranking_type: RankingType,
) {
    info!(target: "layouting", "Executing phase 1: Ranking");
    rank(graph, minimum_length, ranking_type);
}

/// Reorder vertices in ranks to reduce crossings. If `dummy_size` is [Some],
/// dummies will be passed along to the next phase.
fn execute_phase_2(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    dummy_size: Option<f64>,
    crossing_minimization: CrossingMinimization,
    transpose: bool,
) -> Vec<Vec<NodeIndex>> {
    info!(target: "layouting", "Executing phase 2: Crossing Reduction");
    info!(target: "layouting",
        "dummy vertex size: {:?}, heuristic for crossing minimization: {:?}, using transpose: {}",
        dummy_size,
        crossing_minimization,
        transpose
    );

    insert_dummy_vertices(graph, minimum_length, dummy_size.unwrap_or(0.0));
    let mut order = ordering(graph, crossing_minimization, transpose);
    if dummy_size.is_none() {
        remove_dummy_vertices(graph, &mut order);
    }
    order
}

/// calculate the final coordinates for each vertex, after the graph was layered and crossings where minimized.
fn execute_phase_3(
    graph: &mut StableDiGraph<Vertex, Edge>,
    mut layers: Vec<Vec<NodeIndex>>,
) -> LayoutResult {
    info!(target: "layouting", "Executing phase 3: Coordinate Calculation");
    for n in graph.node_indices().collect::<Vec<_>>() {
        if graph[n].is_dummy {
            graph[n].id = n.index();
        }
    }
    let width = layers.iter().map(|l| l.len()).max().unwrap_or(0) as f64;
    let height = layers.len() as f64;
    let mut layouts = create_layouts(graph, &mut layers);

    align_to_smallest_width_layout(&mut layouts);
    let mut x_coordinates = calculate_relative_coords(layouts);
    // determine the smallest x-coordinate
    let min = x_coordinates
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .1;

    // shift all coordinates so the minimum coordinate is 0
    for (_, c) in &mut x_coordinates {
        *c -= min;
    }

    // Find max y size in each rank. Use a BTreeMap so iteration through the map
    // is ordered.
    let mut rank_to_max_height = BTreeMap::<i32, f64>::new();
    for vertex in graph.node_weights() {
        let max = rank_to_max_height.entry(vertex.rank).or_default();
        *max = max.max(vertex.size.1);
    }

    // Stack up each rank to assign it an offset. The gap between each rank and the next is half the
    // height of the current rank, plus half the height of the next rank.
    let mut rank_to_y_offset = HashMap::new();
    let mut current_rank_top_offset = *rank_to_max_height.iter().next().unwrap().1 * -0.5;
    for (rank, max_height) in rank_to_max_height {
        // The center of the rank is the middle of the max height plus the top of the rank.
        rank_to_y_offset.insert(rank, current_rank_top_offset + max_height * 0.5);
        // Shift by the height of the rank. The height of a rank already includes the vertex
        // spacing.
        current_rank_top_offset += max_height;
    }

    let mut v = x_coordinates.iter().collect::<Vec<_>>();
    v.sort_by(|a, b| a.0.index().cmp(&b.0.index()));
    // format to NodeIndex: (x, y), width, height
    (
        x_coordinates
            .into_iter()
            .filter(|(v, _)| !graph[*v].is_dummy)
            // calculate y coordinate
            .map(|(v, x)| {
                (
                    graph[v].id,
                    (x, *rank_to_y_offset.get(&graph[v].rank).unwrap()),
                )
            })
            .collect::<Vec<_>>(),
        width,
        height,
    )
    
}

pub fn slack(graph: &StableDiGraph<Vertex, Edge>, edge: EdgeIndex, minimum_length: i32) -> i32 {
    let (tail, head) = graph.edge_endpoints(edge).unwrap();
    graph[head].rank - graph[tail].rank - minimum_length
}

/// Node data for the final layout graph.
#[derive(Debug, Clone, Default)]
pub struct LayoutNode {
    pub x: i32,
    pub y: i32,
    pub original_id: Option<usize>,
}

/// Builds a new graph containing original and dummy nodes with their final layout attributes.
///
/// This function performs the Sugiyama layout process, including dummy node insertion,
/// and returns a new `StableDiGraph` where nodes contain their calculated `x`, `y` 
/// coordinates and `original_id` for non-dummy nodes.
/// Edges from the layout process (including those involving dummy nodes) are included.
pub fn build_layout_graph(
    mut graph: StableDiGraph<Vertex, Edge>,
    config: &Config,
) -> StableDiGraph<LayoutNode, ()> { 
    info!(target: "layouting", "Start building layout graph (integer coordinates, with dummies)");
    info!(target: "layouting", "Configuration is: {:?}", config);

    init_graph(&mut graph);

    // Phase 1: Ranking
    execute_phase_1(
        &mut graph,
        config.minimum_length as i32,
        config.ranking_type,
    );

    // Phase 2: Reduce crossings 
    let mut order = execute_phase_2(
        &mut graph,
        config.minimum_length as i32,
        Some(config.dummy_size), 
        config.c_minimization,
        config.transpose,
    );

    // Phase 3: Calculate initial coordinates for all nodes, including dummies
    // (The original implementation filters out dummies at this stage)
    // This is a top-to-bottom layout, so:
    // x = within-layer position 
    // y = layer index (rank)
    let mut layouts = create_layouts(&mut graph, &mut order); 
    align_to_smallest_width_layout(&mut layouts);
    let x_coords_float = calculate_relative_coords(layouts);
    let min_x_float = x_coords_float
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .1;

    let mut x_coords_int = HashMap::new();
    for (node_idx, x_f) in x_coords_float {
        let normalized_x_f = x_f - min_x_float;
        // Double to create space for integer rounding and ensure minimum separation
        x_coords_int.insert(node_idx, (normalized_x_f * 2.0).round() as i32);
    }
    
    // Construct a new graph 
    let layout_graph = graph.map(
        |old_node_idx, old_node_data| {
            // Node mapping: Create LayoutNode with calculated coordinates
            let x = *x_coords_int.get(&old_node_idx).unwrap_or(&0); // Assuming coords exist
            let y = old_node_data.rank;
            LayoutNode {
                x,
                y,
                original_id: if old_node_data.is_dummy { None } else { Some(old_node_data.id) },
            }
        },
        |_old_edge_idx, _old_edge_data| {
            // Edge mapping: New edges have no weight
            ()
        },
    );

    info!(target: "layouting", "Finished building layout graph with {} nodes and {} edges", 
          layout_graph.node_count(), layout_graph.edge_count());

    layout_graph
}


// The rectilinear routing algorithm was prototyped in Python,
// we call it here via a Python bridge until we have a Rust implementation.
#[cfg(feature = "python-bindings")]
mod temporary_python_bridge;

#[cfg(feature = "python-bindings")]
pub use temporary_python_bridge::call_python_with_layout_graph;


