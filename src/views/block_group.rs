use std::{
    error::Error,
    time::{Duration, Instant},
};

use crossterm::event::{self, KeyCode, KeyEventKind, MouseButton, MouseEventKind};
use gen_core::PATH_START_NODE_ID;
use gen_graph::{GenGraph, GraphNode};
use gen_models::{block_group::BlockGroup, db::GraphConnection, traits::Query};
use gen_tui::{
    LineStyle,
    graph_view::{GraphView, GraphViewState},
    layout_engine::LayoutEngine,
    plotter::PathStyle,
    theme::current_theme,
};
use log::{info, warn};
use ratatui::{
    layout::{Constraint, Direction, HorizontalAlignment, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Padding, Paragraph, Wrap},
};
use rusqlite::params;

use crate::{
    progress_bar::{get_handler, get_time_elapsed_bar},
    views::{
        collection::{CollectionExplorer, CollectionExplorerState, FocusZone},
        gen_graph_widget::{self, create_gen_graph_engine},
        panels::{render_status_bar, render_with_optional_clear},
        tui_runtime::TuiSession,
    },
};

// Frequency by which we check for external updates to the db
const REFRESH_INTERVAL: u64 = 3; // seconds
const MESSAGE_BUFFER_LIMIT: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelMode {
    Details,
    Messages,
}

fn get_empty_graph() -> GenGraph {
    let mut g = GenGraph::new();
    g.add_node(GraphNode {
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 0,
    });
    g
}

/// Get the most recent path for a block group and map it to GraphNodes in the current graph
fn get_block_group_path_nodes(
    conn: &GraphConnection,
    block_group_id: &gen_core::HashId,
    graph: &GenGraph,
) -> Result<Vec<gen_graph::GraphNode>, String> {
    use gen_models::path::Path;

    // Query the database for the most recent path for this block group
    let path = Path::get(
        conn,
        "SELECT * FROM paths WHERE block_group_id = ?1 ORDER BY created_on DESC LIMIT 1",
        rusqlite::params![block_group_id],
    )
    .map_err(|e| format!("Failed to query path: {}", e))?;

    crate::views::helpers::project_path_nodes(conn, &path, graph)
}

/// Toggle path highlighting for a block group
fn toggle_path_highlight(
    conn: &GraphConnection,
    engine: &LayoutEngine<GenGraph>,
    view_state: &mut GraphViewState<GraphNode>,
    block_group_id: &gen_core::HashId,
    color: ratatui::style::Color,
) -> Result<bool, String> {
    let style = PathStyle::new(color)
        .with_line_style(LineStyle::Bold)
        .with_merge_glyphs(true);
    // Check if highlighting is already active for this style
    if view_state.has_highlight(&style) {
        view_state.clear_highlight(&style);
        Ok(false)
    } else {
        // Get the path nodes for this block group
        let path_nodes = get_block_group_path_nodes(conn, block_group_id, engine.graph())?;

        // Set the path highlight using GraphNodes directly
        view_state.set_path_highlight(style, path_nodes);
        Ok(true)
    }
}

/// Handle a click on a wormhole (`NodeRole::Wormhole`) stub: `boundary` is the node inside the
/// current window the stub is attached to, `target` is the off-screen domain node it leads to
/// (see `GraphViewState::wormhole_hit`).
///
/// If `target` belongs to a known world (`LayoutEngine::wormhole_world_for`), reactivate that
/// exact world and frame the entry node. An evicted world is rebuilt from the same `WorldKey`.
///
/// Otherwise this is new territory: build fresh, anchored on `target`, with `boundary`
/// force-included so it is guaranteed visible as the new world's return boundary.
///
/// Either way, direction (which edge of the new window to enter from) is decided by whether
/// `target` is a successor or predecessor of `boundary` (`LayoutEngine::is_successor`): exiting
/// toward a successor enters the new window from the left, exiting toward a predecessor enters
/// from the right - the same direction you'd naturally keep moving in.
fn teleport_through_wormhole(
    graph_engine: &mut LayoutEngine<GenGraph>,
    graph_view_state: &mut GraphViewState<GraphNode>,
    boundary: GraphNode,
    target: GraphNode,
