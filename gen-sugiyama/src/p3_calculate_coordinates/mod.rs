#[cfg(test)]
mod tests;

use std::collections::HashMap;

use log::info;
use petgraph::{
    Direction::Incoming,
    stable_graph::{NodeIndex, StableDiGraph},
    visit::EdgeRef,
};

use super::{Edge, Vertex, slack};

pub(super) fn create_layouts(
    graph: &mut StableDiGraph<Vertex, Edge>,
    layers: &mut [Vec<NodeIndex>],
) -> Vec<HashMap<NodeIndex, f64>> {
    info!(target: "coordinate_calculation", "Creating individual layouts for coordinate calculation");
    let mut layouts = Vec::new();
    mark_type_1_conflicts(graph, layers);
    // calculate the coordinates for each direction
    for _v_dir in [VDir::Down, VDir::Up] {
        for h_dir in [HDir::Right, HDir::Left] {
            // reset root, align and sink values
            info!(target: "coordinate_calculation",
                "creating layouts for vertical direction: {:?}, horizontal direction {:?}", 
                _v_dir, 
                h_dir);

            reset_alignment(graph, layers);
            create_vertical_alignments(graph, layers);
            let mut layout = do_horizontal_compaction(graph, layers);
            // flip x_coordinates if we went from right to left
            if let HDir::Left = h_dir {
                layout.values_mut().for_each(|x| *x = -*x);
            }
            // print_to_console(v_dir, graph, &orig_layers, layout.clone(), vertex_spacing);
            layouts.push(layout);

            // rotate the graph
            for row in layers.iter_mut() {
                row.reverse();
            }
        }
        // rotate the graph
        graph.reverse();
        layers.reverse();
    }
    // do this one last time, so ranks are in original order
    reset_alignment(graph, layers);
    layouts
}

pub(crate) fn align_to_smallest_width_layout(aligned_layouts: &mut [HashMap<NodeIndex, f64>]) {
    info!(target: "coordinate_calculation", "Aligning all layouts to the one with the smallest width");
    // determine minimum and maximum coordinate of each layout, plus the width
    let min_max: Vec<(f64, f64, f64)> = aligned_layouts
        .iter()
        .map(|c| {
            let min = *c.values().min_by(|a, b| a.total_cmp(b)).unwrap();
            let max = *c.values().max_by(|a, b| a.total_cmp(b)).unwrap();
            (min, max, max - min)
        })
        .collect();

    // determine the layout with the minimum width
    let min_width = min_max
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.2.total_cmp(&b.1.2))
        .unwrap()
        .0;

    // align all other layouts to the lowest coordinate of the layout with the minimum width,
    for (i, layout) in aligned_layouts.iter_mut().enumerate() {
        // if i % 2 == 0, then horizontal direction was left
        let shift = if i % 2 == 0 {
            min_max[i].0 - min_max[min_width].0
        } else {
            min_max[min_width].1 - min_max[i].1
        };
        for v in layout.values_mut() {
            let new = *v + shift;
            *v = new;
        }
    }
}

pub(crate) fn calculate_relative_coords(
    aligned_layouts: Vec<HashMap<NodeIndex, f64>>,
) -> Vec<(NodeIndex, f64)> {
    info!(target: "coordinate_calculation", 
        "Calculate relative coordinates, by taking average between two medians of absolute x-coordinates for each layout direction");
    // sort all 4 coordinates per vertex in ascending order
    for l in &aligned_layouts {
        let mut v = l.iter().collect::<Vec<_>>();
        v.sort_by_key(|a| a.0.index());
        // format to NodeIndex: (x, y), width, height
        // println!("{v:?}\n");
    }
    let mut sorted_layouts = HashMap::new();
    for k in aligned_layouts.first().unwrap().keys() {
        let mut vertex_coordinates = [
            *aligned_layouts.first().unwrap().get(k).unwrap(),
            *aligned_layouts.get(1).unwrap().get(k).unwrap(),
            *aligned_layouts.get(2).unwrap().get(k).unwrap(),
            *aligned_layouts.get(3).unwrap().get(k).unwrap(),
        ];
        vertex_coordinates.sort_by(|a, b| a.total_cmp(b));
        sorted_layouts.insert(k, vertex_coordinates);
    }

    // create final layout, by averaging the two median values
    // try to use something like mean
    sorted_layouts
        .into_iter()
        .map(|(k, v)| (*k, (v[0] + v[1] + v[2] + v[3]) / 4.0))
        .collect::<Vec<_>>()
}

fn is_incident_to_inner_segment(graph: &StableDiGraph<Vertex, Edge>, id: NodeIndex) -> bool {
    graph[id].is_dummy
        && graph
            .neighbors_directed(id, Incoming)
            .any(|n| graph[n].is_dummy)
}

/// Assumes id is incident to inner segment
fn get_inner_segment_upper_neighbor(
    graph: &StableDiGraph<Vertex, Edge>,
    id: NodeIndex,
) -> Option<NodeIndex> {
    if is_incident_to_inner_segment(graph, id) {
        graph.neighbors_directed(id, Incoming).next()
    } else {
        None
    }
}

fn mark_type_1_conflicts(graph: &mut StableDiGraph<Vertex, Edge>, layers: &[Vec<NodeIndex>]) {
    info!(target: "coordinate_calculation", 
        "Marking type one conflicts (edge crossings between dummy vertices and non dummy vertices)");

    for (level, next_level) in layers[..layers.len() - 1].iter().zip(layers[1..].iter()) {
        let mut left_dummy_index = 0;
        let mut l = 0;
        for (l_1, dummy_candidate) in next_level.iter().enumerate() {
            let right_dummy_index = match get_inner_segment_upper_neighbor(graph, *dummy_candidate)
            {
                Some(id) => graph[id].pos,
                None => {
                    if l_1 == next_level.len() - 1 {
                        level.len()
                    } else {
                        continue;
                    }
                }
            };
            while l < l_1 {
                let vertex = next_level[l];
                let mut upper_neighbors = graph
                    .neighbors_directed(vertex, Incoming)
                    .collect::<Vec<_>>();
                upper_neighbors.sort_by(|a, b| graph[*a].pos.cmp(&graph[*b].pos));
                for upper_neighbor in upper_neighbors {
                    let vertex_index = graph[upper_neighbor].pos;
                    if vertex_index < left_dummy_index || vertex_index > right_dummy_index {
                        let edge = graph.find_edge(upper_neighbor, vertex).unwrap();
                        graph[edge].has_type_1_conflict = true;
                    }
                }
                l += 1;
            }
            left_dummy_index = right_dummy_index;
        }
    }
}

pub(super) fn reset_alignment(graph: &mut StableDiGraph<Vertex, Edge>, layers: &[Vec<NodeIndex>]) {
    for (rank, row) in layers.iter().enumerate() {
        for (pos, v) in row.iter().enumerate() {
            let weight: &mut Vertex = &mut graph[*v];
            weight.rank = rank as i32;
            weight.pos = pos;
            weight.shift = f64::INFINITY;
            weight.align = *v;
            weight.root = *v;
            weight.sink = *v;
        }
    }
}

// TODO: Change this so the graph gets rotated outside of the function
/// Aligns the graph in so called blocks, which are used in the next step
/// to determine the x-coordinate of a vertex.
fn create_vertical_alignments(
    graph: &mut StableDiGraph<Vertex, Edge>,
    layers: &mut [Vec<NodeIndex>],
) {
    info!(target: "coordinate_calculation", "Creating vertical alignments");
    for layer in layers {
        let mut r = -1;

        for v in layer.iter().copied() {
            let mut edges = graph
                .edges_directed(v, Incoming)
                .filter(|e| slack(graph, e.id(), 1) == 0)
                .map(|e| (e.id(), e.source()))
                .collect::<Vec<_>>();

            if edges.is_empty() {
                continue;
            }

            edges.sort_by(|e1, e2| graph[e1.1].pos.cmp(&graph[e2.1].pos));

            let d = (edges.len() as f64 + 1.) / 2. - 1.; // need to subtract one because indices are zero based
            let lower_upper_median = [d.floor() as usize, d.ceil() as usize];

            for m in lower_upper_median {
                if graph[v].align == v {
                    let edge_id = edges[m].0;
                    let median_neighbor = edges[m].1;

                    // TODO: do not force alignment through nodes that have a degree > 2
                    // (currently label dimensions can leak between layers)
                    if !graph[edge_id].has_type_1_conflict
                        && r < graph[median_neighbor].pos as isize
                    {
                        graph[median_neighbor].align = v;
                        graph[v].root = graph[median_neighbor].root;
                        graph[v].align = graph[v].root;
                        r = graph[median_neighbor].pos as isize;
                    }
                }
            }
        }
    }
}

fn do_horizontal_compaction(
    graph: &mut StableDiGraph<Vertex, Edge>,
    layers: &[Vec<NodeIndex>],
) -> HashMap<NodeIndex, f64> {
    info!(target: "coordinate_calculation", "calculating coordinates for layout.");
    compute_block_max_vertex_widths(graph);

    let mut x_coordinates = place_blocks(graph, layers);
    // calculate class shifts
    info!(target: "coordinate_calculation", "move blocks as close together as possible");
    for i in 0..layers.len() {
        let mut v = layers[i][0];
        if graph[v].sink == v {
            if graph[graph[v].sink].shift == f64::INFINITY {
                let v_sink = graph[v].sink;
                graph[v_sink].shift = 0.0;
            }
            let mut j = i; // level index
            let mut k = 0; // vertex in level index
            loop {
                v = layers[j][k];

                // traverse one block
                while graph[v].align != graph[v].root {
                    v = graph[v].align;
                    j += 1;

                    if graph[v].pos > 1 {
                        let u = pred(graph[v], layers);
                        let gap = (graph[v].block_max_vertex_width
                            + graph[u].block_max_vertex_width)
                            * 0.5;
                        let distance_v_u = *x_coordinates.get(&v).unwrap()
                            - (*x_coordinates.get(&u).unwrap() + gap);
                        let u_sink = graph[u].sink;
                        graph[u_sink].shift = graph[u_sink]
                            .shift
                            .min(graph[graph[v].sink].shift + distance_v_u);
                    }
                }
                k = graph[v].pos + 1;

                if k == layers[j].len() || graph[v].sink != graph[layers[j][k]].sink {
                    break;
                }
            }
        }
    }

    // calculate absolute x-coordinates
    for v in graph.node_indices() {
        x_coordinates.insert(
            v,
            *x_coordinates.get(&v).unwrap() + graph[graph[v].sink].shift,
        );
    }
    x_coordinates
}

/// Computes the maximum width of the vertices in each block and assigns the
/// width to [Vertex::block_max_vertex_width] of each vertex in the block.
fn compute_block_max_vertex_widths(graph: &mut StableDiGraph<Vertex, Edge>) {
    for root in graph
        .node_indices()
        .filter(|v| graph[*v].root == *v)
        // Collect so we can mutate nodes while iterating.
        .collect::<Vec<_>>()
    {
        let root_vertex = &mut graph[root];

        let mut max_vertex_width = root_vertex.size.0;
        let mut current = root_vertex.align;
        while current != root {
            let current_vertex = &graph[current];
            max_vertex_width = max_vertex_width.max(current_vertex.size.0);
            current = current_vertex.align;
        }

        let root_vertex = &mut graph[root];
        root_vertex.block_max_vertex_width = max_vertex_width;

        current = root_vertex.align;
        while current != root {
            let current_vertex = &mut graph[current];
            current_vertex.block_max_vertex_width = max_vertex_width;
            current = current_vertex.align;
        }
    }
}

fn place_blocks(
    graph: &mut StableDiGraph<Vertex, Edge>,
    layers: &[Vec<NodeIndex>],
) -> HashMap<NodeIndex, f64> {
    info!(target: "coordinate_calculation", "Placing vertices in blocks.");
    let mut x_coordinates = HashMap::new();
    // place blocks
    for root in graph
        .node_indices()
        .filter(|v| graph[*v].root == *v)
        .collect::<Vec<_>>()
    {
        place_block(graph, layers, root, &mut x_coordinates);
    }
    x_coordinates
}
fn place_block(
    graph: &mut StableDiGraph<Vertex, Edge>,
    layers: &[Vec<NodeIndex>],
    root: NodeIndex,
    x_coordinates: &mut HashMap<NodeIndex, f64>,
) {
    if x_coordinates.get(&root).is_some() {
        return;
    }
    x_coordinates.insert(root, 0.0);
    let mut w = root;
    loop {
        if graph[w].pos > 0 {
            let u = graph[pred(graph[w], layers)].root;
            place_block(graph, layers, u, x_coordinates);
            // initialize sink of current node to have the same sink as the root
            if graph[root].sink == root {
                graph[root].sink = graph[u].sink;
            }
            if graph[root].sink == graph[u].sink {
                let gap =
                    (graph[root].block_max_vertex_width + graph[u].block_max_vertex_width) * 0.5;
                x_coordinates.insert(
                    root,
                    x_coordinates
                        .get(&root)
                        .unwrap()
                        .max(x_coordinates.get(&u).unwrap() + gap),
                );
            }
        }
        w = graph[w].align;
        if w == root {
            break;
        }
    }
    // align all other vertices in this block to the x-coordinate of the root
    while graph[w].align != root {
        w = graph[w].align;
        x_coordinates.insert(w, *x_coordinates.get(&root).unwrap());
        graph[w].sink = graph[root].sink;
    }
}

fn pred(vertex: Vertex, layers: &[Vec<NodeIndex>]) -> NodeIndex {
    layers[vertex.rank as usize][vertex.pos - 1]
}
/// Represents a layered graph whose vertices have been aligned in blocks.
/// A root is the highest node in a block, depending on the direction.
///
/// It is used to determine classes of a block, calculate the x-coordinates of a block
/// in regard to its class and shift classes together as close as possible.
///
/// Represents the horizontal direction in which the algorithm is run
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum HDir {
    Left,
    Right,
}

/// Represents the vertical direction in which the algorithm is run
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum VDir {
    Up,
    Down,
}
