use std::collections::HashSet;

use log::{debug, info};
use petgraph::stable_graph::{NodeIndex, StableDiGraph};

#[allow(dead_code)]
/// Split a graph into its weakly connected components
pub fn weakly_connected_components<V: Copy, E: Copy>(
    graph: StableDiGraph<V, E>,
) -> Vec<StableDiGraph<V, E>> {
    info!(target: "connected_components", "Splitting graph into its connected components");
    let mut components = Vec::new();
    let mut visited = HashSet::new();

    for node in graph.node_indices() {
        if visited.contains(&node) {
            continue;
        }

        let component_nodes = component_dfs(node, &graph);
        let component = graph.filter_map(
            |n, w| {
                if component_nodes.contains(&n) {
                    Some(*w)
                } else {
                    None
                }
            },
            |_, w| Some(*w),
        );

        component_nodes.into_iter().for_each(|n| {
            visited.insert(n);
        });
        components.push(component);
    }
    debug!(target: "connected_components", "Found {} components", components.len());

    components
}

fn component_dfs<V: Copy, E: Copy>(
    start: NodeIndex,
    graph: &StableDiGraph<V, E>,
) -> HashSet<NodeIndex> {
    let mut queue = vec![start];
    let mut visited = HashSet::new();

    visited.insert(start);

    while let Some(cur) = queue.pop() {
        for neighbor in graph.neighbors_undirected(cur) {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);
            queue.push(neighbor);
        }
    }

    visited
}

/// Direction for iteration
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum IterDir {
    Forward,
    Backward,
}

/// Create an iterator that produces indices in the specified direction
pub fn iterate(dir: IterDir, length: usize) -> impl Iterator<Item = usize> {
    let (mut start, step) = match dir {
        IterDir::Forward => (usize::MAX, 1), // up corresponds to left to right
        IterDir::Backward => (length, usize::MAX),
    };
    std::iter::repeat_with(move || {
        start = start.wrapping_add(step);
        start
    })
    .take(length)
}

/// Sort a vector of usize values using radix sort
pub fn radix_sort(mut input: Vec<usize>, key_length: usize) -> Vec<usize> {
    let mut output = vec![0; input.len()];

    let mut key = 1;
    for _ in 0..key_length {
        counting_sort(&mut input, key, &mut output);
        key *= 10;
    }
    input
}

#[inline(always)]
fn counting_sort(input: &mut [usize], key: usize, output: &mut [usize]) {
    let mut count = [0; 10];
    // insert initial counts
    for i in input.iter().map(|n| self::key(key, *n)) {
        count[i] += 1;
    }

    // built accumulative sum
    for i in 1..10 {
        count[i] += count[i - 1];
    }

    for i in (0..input.len()).rev() {
        let k = self::key(key, input[i]);
        count[k] -= 1;
        output[count[k]] = input[i];
    }
    input.copy_from_slice(&output[..input.len()]);
}

#[inline(always)]
fn key(key: usize, n: usize) -> usize {
    (n / key) % 10
}
