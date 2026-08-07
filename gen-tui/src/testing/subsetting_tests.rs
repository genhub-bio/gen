//! Crawl and full-render checks across representative topologies, budgets, and anchor positions.

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::NodeIndex;

use crate::{
    crawl::neighborhood,
    distribute_nodes::GapSizes,
    graph_widget::build_window_geometry,
    layout::NodeRole,
    layout_engine::LayoutEngine,
    testing::{
        mocks::{MockDomainGraph, TestGraphs},
        validate_layout_graph,
    },
};

/// Return every domain-level window invariant violated by one crawl.
fn crawl_invariant_errors(
    graph: &MockDomainGraph,
    anchor: NodeIndex,
    node_budget: usize,
    case: &str,
) -> Vec<String> {
    let subgraph = match neighborhood(
        anchor,
        node_budget,
        graph,
        None,
        &HashMap::new(),
        &HashMap::new(),
    ) {
        Ok(subgraph) => subgraph,
        Err(error) => return vec![format!("{case}: neighborhood failed: {error}")],
    };

    let mut errors = Vec::new();

    if !subgraph.nodes.contains(&anchor) {
        errors.push(format!("{case}: anchor must be part of its own window"));
    }
    if subgraph.nodes.len() > node_budget {
        errors.push(format!(
            "{case}: crawl produced {} nodes, exceeding its budget of {node_budget}",
            subgraph.nodes.len()
        ));
    }

    let windowed: HashSet<NodeIndex> = subgraph.nodes.iter().copied().collect();
    let mut adjacency: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for &(source, target) in &subgraph.edges {
        adjacency.entry(source).or_default().push(target);
        adjacency.entry(target).or_default().push(source);
    }
    let mut visited = HashSet::from([anchor]);
    let mut queue = VecDeque::from([anchor]);
    while let Some(current) = queue.pop_front() {
        for &neighbor in adjacency.get(&current).into_iter().flatten() {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    if visited != windowed {
        errors.push(format!(
            "{case}: crawled window is not fully connected via its own edges"
        ));
    }

    for edge in &subgraph.external_edges {
        let boundary = edge.boundary;
        let target = edge.target;
        let bundle = &edge.all_collapsed;
        if !windowed.contains(&boundary) {
            errors.push(format!(
                "{case}: external door's boundary {boundary:?} is not actually in the window"
            ));
        }
        if windowed.contains(&target) {
            errors.push(format!(
                "{case}: external door's navigation target {target:?} should be outside the window"
            ));
        }
        if bundle.is_empty() {
            errors.push(format!(
                "{case}: a wormhole door's collapsed bundle must not be empty"
            ));
        }
        for &collapsed in bundle {
            if windowed.contains(&collapsed) {
                errors.push(format!(
                    "{case}: bundle entry {collapsed:?} should be outside the window"
                ));
            }
        }
    }

    errors
}

/// Return every layout invariant violated by a full render pipeline run.
fn full_render_errors(
    graph: MockDomainGraph,
    anchor: NodeIndex,
    node_budget: usize,
    case: &str,
) -> Vec<String> {
    let mut engine = LayoutEngine::new(graph);
    let assembled = match engine.window_for(anchor, node_budget) {
        Ok(assembled) => assembled,
        Err(error) => return vec![format!("{case}: window_for failed: {error}")],
    };
    let geometry = build_window_geometry(assembled, &GapSizes::default(), |role| match role {
        NodeRole::Data(_) => (5, 3),
        _ => (1, 1),
    });
    let validation = validate_layout_graph(&geometry.graph);
    if validation.is_valid() {
        Vec::new()
    } else {
        vec![format!("{case}: {}", validation.summary())]
    }
}

/// Runs both the crawl-invariant and full-render checks for one topology/anchor/budget
/// combination, appending every failure found to `errors` instead of stopping at the first.
/// When `node_budget` is large enough to cover the whole graph, also checks there is no window
/// boundary left at all (every node present, no wormhole doors).
fn run_case(
    graph: &MockDomainGraph,
    total_nodes: usize,
    anchor: NodeIndex,
    node_budget: usize,
    case: &str,
    errors: &mut Vec<String>,
) {
    errors.extend(crawl_invariant_errors(graph, anchor, node_budget, case));

    if node_budget >= total_nodes {
        match neighborhood(
            anchor,
            node_budget,
            graph,
            None,
            &HashMap::new(),
            &HashMap::new(),
        ) {
            Ok(subgraph) => {
                if subgraph.nodes.len() != total_nodes {
                    errors.push(format!(
                        "{case}: an unlimited budget captured {} of {total_nodes} nodes",
                        subgraph.nodes.len()
                    ));
                }
                if !subgraph.external_edges.is_empty() {
                    errors.push(format!(
                        "{case}: an unlimited budget left {} window boundary door(s)",
                        subgraph.external_edges.len()
                    ));
                }
            }
            Err(error) => errors.push(format!("{case}: neighborhood failed: {error}")),
        }
    }

    errors.extend(full_render_errors(graph.clone(), anchor, node_budget, case));
}

/// Fails with every accumulated case error, one per line, or does nothing if `errors` is empty.
fn assert_no_errors(errors: Vec<String>) {
    assert!(
        errors.is_empty(),
        "{} case(s) failed:\n{}",
        errors.len(),
        errors.join("\n")
    );
}

/// A trailing wormhole must remain connected after channel routing.
#[test]
fn test_trailing_wormhole_door_keeps_layout_connected() {
    let graph = TestGraphs::domain_long_chain(40);
    let anchor = NodeIndex::new(0);
    let errors = full_render_errors(
        graph,
        anchor,
        12,
        "trailing_wormhole_door_keeps_layout_connected",
    );
    assert_no_errors(errors);
}

const PARTIAL_BUDGET: usize = 12;

#[test]
fn subsetting_linear_chain() {
    const LENGTH: usize = 40;
    let graph = TestGraphs::domain_long_chain(LENGTH);
    let first = NodeIndex::new(0);
    let middle = NodeIndex::new(LENGTH / 2);
    let last = NodeIndex::new(LENGTH - 1);

    let mut errors = Vec::new();
    for (label, anchor) in [
        ("side (first)", first),
        ("middle", middle),
        ("side (last)", last),
    ] {
        run_case(
            &graph,
            LENGTH,
            anchor,
            PARTIAL_BUDGET,
            &format!("linear_chain/{label}/partial_budget"),
            &mut errors,
        );
        run_case(
            &graph,
            LENGTH,
            anchor,
            LENGTH,
            &format!("linear_chain/{label}/unlimited_budget"),
            &mut errors,
        );
    }
    assert_no_errors(errors);
}

#[test]
fn subsetting_diamond_chain() {
    const DIAMONDS: usize = 10;
    let graph = TestGraphs::domain_diamond_chain(DIAMONDS);
    let total_nodes = 1 + 3 * DIAMONDS;

    let first = NodeIndex::new(0);
    // The join node of the middle diamond: node index 3 * (diamond index + 1).
    let middle = NodeIndex::new(3 * (DIAMONDS / 2 + 1));
    // The join node of the last diamond, the highest node index in the graph.
    let last = NodeIndex::new(3 * DIAMONDS);

    let mut errors = Vec::new();
    for (label, anchor) in [
        ("side (first)", first),
        ("middle", middle),
        ("side (last)", last),
    ] {
        run_case(
            &graph,
            total_nodes,
            anchor,
            PARTIAL_BUDGET,
            &format!("diamond_chain/{label}/partial_budget"),
            &mut errors,
        );
        run_case(
            &graph,
            total_nodes,
            anchor,
            total_nodes,
            &format!("diamond_chain/{label}/unlimited_budget"),
            &mut errors,
        );
    }
    assert_no_errors(errors);
}

#[test]
fn subsetting_strut() {
    const LAYERS: usize = 15;
    const WIDTH: usize = 2;
    let graph = TestGraphs::domain_strut(LAYERS, WIDTH);
    let total_nodes = LAYERS * WIDTH;

    let first = NodeIndex::new(0);
    let middle = NodeIndex::new((LAYERS / 2) * WIDTH);
    let last = NodeIndex::new((LAYERS - 1) * WIDTH);

    let mut errors = Vec::new();
    for (label, anchor) in [
        ("side (first)", first),
        ("middle", middle),
        ("side (last)", last),
    ] {
        run_case(
            &graph,
            total_nodes,
            anchor,
            PARTIAL_BUDGET,
            &format!("strut/{label}/partial_budget"),
            &mut errors,
        );
        run_case(
            &graph,
            total_nodes,
            anchor,
            total_nodes,
            &format!("strut/{label}/unlimited_budget"),
            &mut errors,
        );
    }
    assert_no_errors(errors);
}

#[test]
fn subsetting_grid() {
    const ROWS: usize = 6;
    const COLS: usize = 6;
    let (graph, node_at) = TestGraphs::domain_grid(ROWS, COLS);
    let total_nodes = ROWS * COLS;

    let corner = node_at[0][0];
    let center = node_at[ROWS / 2][COLS / 2];
    let opposite_corner = node_at[ROWS - 1][COLS - 1];

    let mut errors = Vec::new();
    for (label, anchor) in [
        ("side (corner)", corner),
        ("middle (center)", center),
        ("side (opposite corner)", opposite_corner),
    ] {
        run_case(
            &graph,
            total_nodes,
            anchor,
            PARTIAL_BUDGET,
            &format!("grid/{label}/partial_budget"),
            &mut errors,
        );
        run_case(
            &graph,
            total_nodes,
            anchor,
            total_nodes,
            &format!("grid/{label}/unlimited_budget"),
            &mut errors,
        );
    }
    assert_no_errors(errors);
}
