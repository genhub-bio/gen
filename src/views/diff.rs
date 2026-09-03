use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    time::Instant,
};

use crossterm::event::{self, Event, KeyCode};
use gen_diff::{
    graph::DiffGenGraph,
    operations::{BlockGroupChangeKind, BlockGroupDiff, OperationDiff},
};
use gen_models::db::GraphConnection;
use gen_tui::theme::current_theme;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{List, ListItem},
};

use crate::views::{
    diff_graph::{
        DiffGraphComponent, apply_diff_highlights, block_group_label, build_diff_graph_component,
        change_label_for_block_group,
    },
    gen_graph_widget::{create_gen_graph_controller, create_gen_graph_widget},
    panels::{PanelFocus, PanelStyles, panel_block, render_status_bar},
    tui_runtime::TuiSession,
};

struct DiffComponent {
    render: DiffGraphComponent,
    block_group: String,
    change_label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SampleStatus {
    Added,
    Removed,
    Modified,
}

impl SampleStatus {
    fn title(self) -> &'static str {
        match self {
            SampleStatus::Added => "Added Samples",
            SampleStatus::Removed => "Removed Samples",
            SampleStatus::Modified => "Modified Samples",
        }
    }
}

struct SampleComponent {
    collection: String,
    sample: String,
    status: SampleStatus,
    block_groups: Vec<DiffComponent>,
}

struct ExplorerEntry {
    label: String,
    sample_index: Option<usize>,
    block_group_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffPanel {
    List,
    Graph,
}

pub fn view_diff(
    conn: &GraphConnection,
    workspace: &gen_core::Workspace,
    diff: &OperationDiff,
) -> Result<(), io::Error> {
    let samples = collect_samples(&diff.diff_graph);

    if samples.is_empty() {
        println!("No differences to display.");
        return Ok(());
    }

    let mut expanded_samples = BTreeSet::new();
    expanded_samples.insert(0usize);
    let mut entries = build_explorer_entries(&samples, &expanded_samples);
    let mut selected_entry = first_selectable_entry(&entries).unwrap_or(0);

    let mut session = TuiSession::enter()?;
    let terminal = session.terminal_mut();

    let mut current_component = resolve_current_component(&samples, &entries, selected_entry)
        .unwrap_or(&samples[0].block_groups[0]);

    let mut panel_focus = PanelFocus::new(DiffPanel::List);
    panel_focus.include_panel(DiffPanel::Graph);
    let panel_styles = PanelStyles::default();

    let mut graph_controller = create_gen_graph_controller(current_component.render.graph.clone());
    apply_diff_highlights(&mut graph_controller, &current_component.render);

    let mut last_frame_time = Instant::now();

    loop {
        entries = build_explorer_entries(&samples, &expanded_samples);
        if let Some(selected_component) =
            resolve_current_component(&samples, &entries, selected_entry)
            && selected_component.render.title != current_component.render.title
        {
            current_component = selected_component;
            graph_controller = create_gen_graph_controller(current_component.render.graph.clone());
            apply_diff_highlights(&mut graph_controller, &current_component.render);
        }

        let now = Instant::now();
        let frame_delta = now.duration_since(last_frame_time);
        last_frame_time = now;

        terminal.draw(|f| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(1)])
                .split(f.area());

            let main = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(60), Constraint::Min(1)])
                .split(outer[0]);

            let list_items: Vec<ListItem> = entries
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    let style = if index == selected_entry {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else if entry.sample_index.is_none() {
                        Style::default().add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    ListItem::new(entry.label.clone()).style(style)
                })
                .collect();

            let list = List::new(list_items).block(panel_block(
                "Samples",
                &panel_focus,
                DiffPanel::List,
                panel_styles,
            ));
            f.render_widget(list, main[0]);

            let graph_title = current_component.render.title.clone();
            let graph_block =
                panel_block(graph_title, &panel_focus, DiffPanel::Graph, panel_styles);
            let inner_canvas = graph_block.inner(main[1]);

            graph_controller.viewport_state.focus();
            graph_controller.viewport_state.viewport_bounds = inner_canvas;
            graph_controller.update_animations(frame_delta);

            f.render_widget(graph_block, main[1]);

            let canvas_style = Style::default().bg(current_theme()[0x00]);
            let widget = create_gen_graph_widget(conn, workspace)
                .detail_level(graph_controller.get_detail_level())
                .style(canvas_style)
                .cursor();
            f.render_stateful_widget(widget, inner_canvas, &mut graph_controller);

            let panel_messages = if panel_focus.is_navigation() {
                "*tab* toggle focus | *enter* activate | *q* quit"
            } else if panel_focus.current() == DiffPanel::List {
                "*↑↓* select | *enter/right* expand | *left* collapse | *tab* graph | *esc* leave | *q* quit"
            } else {
                "*←→↑↓* pan | *+/-* zoom | *tab* list | *esc* leave | *q* quit"
            };
            render_status_bar(f, outer[1], panel_messages);
        })?;

        if event::poll(std::time::Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            if panel_focus.is_navigation() {
                match key.code {
                    KeyCode::Tab | KeyCode::Left | KeyCode::Right => {
                        panel_focus.cycle_next();
                    }
                    KeyCode::Enter => {
                        panel_focus.activate();
                    }
                    KeyCode::Esc | KeyCode::Char('q') => break,
                    _ => {}
                }
            } else if key.code == KeyCode::Esc {
                panel_focus.deactivate();
            } else if key.code == KeyCode::Tab {
                panel_focus.cycle_next();
            } else if key.code == KeyCode::Char('q') {
                break;
            } else if panel_focus.current() == DiffPanel::List {
                match key.code {
                    KeyCode::Up => {
                        if let Some(previous_entry) =
                            previous_selectable_entry(&entries, selected_entry)
                        {
                            selected_entry = previous_entry;
                        }
                    }
                    KeyCode::Down => {
                        if let Some(next_entry) = next_selectable_entry(&entries, selected_entry) {
                            selected_entry = next_entry;
                        }
                    }
                    KeyCode::Enter | KeyCode::Right => {
                        if let Some(sample_index) = entries[selected_entry].sample_index
                            && entries[selected_entry].block_group_index.is_none()
                        {
                            expanded_samples.insert(sample_index);
                            entries = build_explorer_entries(&samples, &expanded_samples);
                            selected_entry =
                                sample_row_index(&entries, sample_index).unwrap_or(selected_entry);
                        }
                    }
                    KeyCode::Left => {
                        if let Some(sample_index) = entries[selected_entry].sample_index
                            && (entries[selected_entry].block_group_index.is_some()
                                || expanded_samples.contains(&sample_index))
                        {
                            expanded_samples.remove(&sample_index);
                            entries = build_explorer_entries(&samples, &expanded_samples);
                            selected_entry =
                                sample_row_index(&entries, sample_index).unwrap_or(selected_entry);
                        }
                    }
                    _ => {}
                }
            } else {
                let _ = graph_controller.handle_key_event(key);
            }
        }
    }

    Ok(())
}

/// Display one annotated graph diff without the operation/sample explorer.
pub fn view_diff_graph(
    conn: &GraphConnection,
    workspace: &gen_core::Workspace,
    diff_graph: &DiffGenGraph,
    title: String,
) -> Result<(), io::Error> {
    let component = build_diff_graph_component(diff_graph, title);
    let mut session = TuiSession::enter()?;
    let terminal = session.terminal_mut();
    let mut graph_controller = create_gen_graph_controller(component.graph.clone());
    apply_diff_highlights(&mut graph_controller, &component);
    let mut last_frame_time = Instant::now();

    loop {
        let now = Instant::now();
        let frame_delta = now.duration_since(last_frame_time);
        last_frame_time = now;

        terminal.draw(|frame| {
            let areas = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(1)])
                .split(frame.area());
            let graph_block = ratatui::widgets::Block::bordered().title(component.title.clone());
            let canvas_area = graph_block.inner(areas[0]);
            graph_controller.viewport_state.focus();
            graph_controller.viewport_state.viewport_bounds = canvas_area;
            graph_controller.update_animations(frame_delta);

            frame.render_widget(graph_block, areas[0]);
            let widget = create_gen_graph_widget(conn, workspace)
                .detail_level(graph_controller.get_detail_level())
                .style(ratatui::style::Style::default().bg(current_theme()[0x00]))
                .cursor();
            frame.render_stateful_widget(widget, canvas_area, &mut graph_controller);
            render_status_bar(frame, areas[1], "*←→↑↓* pan | *+/-* zoom | *q/esc* quit");
        })?;

        if event::poll(std::time::Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('q')) {
                break;
            }
            let _ = graph_controller.handle_key_event(key);
        }
    }

    Ok(())
}

fn collect_samples(graphs: &[BlockGroupDiff]) -> Vec<SampleComponent> {
    let mut grouped = BTreeMap::<(SampleStatus, String, String), Vec<DiffComponent>>::new();
    for graph_diff in graphs {
        let change_label = change_label_for_block_group(graph_diff);
        let (collection, sample, block_group) = if let Some(block_group) = graph_diff
            .target_block_group
            .as_ref()
            .or(graph_diff.source_block_group.as_ref())
        {
            (
                block_group.collection_name.clone(),
                block_group.sample_name.clone(),
                block_group.name.clone(),
            )
        } else {
            (
                String::from("Unknown"),
                String::from("Unknown"),
                String::from("Unknown"),
            )
        };
        let status = sample_status_for_block_group(graph_diff);
        grouped
            .entry((status, collection.clone(), sample.clone()))
            .or_default()
            .push(build_component(
                &graph_diff.graph,
                change_label,
                &block_group_label(graph_diff),
                block_group,
            ));
    }

    grouped
        .into_iter()
        .map(|((status, collection, sample), mut block_groups)| {
            block_groups.sort_by(|left, right| left.block_group.cmp(&right.block_group));
            SampleComponent {
                collection,
                sample,
                status,
                block_groups,
            }
        })
        .collect()
}

fn build_component(
    diff_graph: &DiffGenGraph,
    change_label: &'static str,
    title: &str,
    block_group: String,
) -> DiffComponent {
    DiffComponent {
        render: build_diff_graph_component(diff_graph, format!("{change_label} {title}")),
        block_group,
        change_label,
    }
}

fn build_explorer_entries(
    samples: &[SampleComponent],
    expanded_samples: &BTreeSet<usize>,
) -> Vec<ExplorerEntry> {
    let mut entries = Vec::new();
    let mut current_status = None;
    for (sample_index, sample) in samples.iter().enumerate() {
        if current_status != Some(sample.status) {
            current_status = Some(sample.status);
            entries.push(ExplorerEntry {
                label: sample.status.title().to_string(),
                sample_index: None,
                block_group_index: None,
            });
        }

        let expanded = expanded_samples.contains(&sample_index);
        let marker = if expanded { "v" } else { ">" };
        entries.push(ExplorerEntry {
            label: format!("{marker} {}", sample_label(sample)),
            sample_index: Some(sample_index),
            block_group_index: None,
        });

        if expanded {
            for (block_group_index, block_group) in sample.block_groups.iter().enumerate() {
                entries.push(ExplorerEntry {
                    label: format!(
                        "  {} | {}",
                        block_group.change_label, block_group.block_group
                    ),
                    sample_index: Some(sample_index),
                    block_group_index: Some(block_group_index),
                });
            }
        }
    }
    entries
}

fn sample_label(sample: &SampleComponent) -> String {
    if sample.collection == "Default" || sample.collection.is_empty() {
        sample.sample.clone()
    } else {
        format!("{} | {}", sample.collection, sample.sample)
    }
}

fn resolve_current_component<'a>(
    samples: &'a [SampleComponent],
    entries: &[ExplorerEntry],
    selected_entry: usize,
) -> Option<&'a DiffComponent> {
    let entry = entries.get(selected_entry)?;
    let sample = samples.get(entry.sample_index?)?;
    let block_group_index = entry.block_group_index.unwrap_or(0);
    sample.block_groups.get(block_group_index)
}

fn first_selectable_entry(entries: &[ExplorerEntry]) -> Option<usize> {
    entries
        .iter()
        .position(|entry| entry.sample_index.is_some())
}

fn previous_selectable_entry(entries: &[ExplorerEntry], selected_entry: usize) -> Option<usize> {
    entries[..selected_entry]
        .iter()
        .rposition(|entry| entry.sample_index.is_some())
}

fn next_selectable_entry(entries: &[ExplorerEntry], selected_entry: usize) -> Option<usize> {
    entries
        .iter()
        .enumerate()
        .skip(selected_entry + 1)
        .find_map(|(index, entry)| entry.sample_index.map(|_| index))
}

fn sample_row_index(entries: &[ExplorerEntry], sample_index: usize) -> Option<usize> {
    entries.iter().position(|entry| {
        entry.sample_index == Some(sample_index) && entry.block_group_index.is_none()
    })
}

fn sample_status_for_block_group(diff: &BlockGroupDiff) -> SampleStatus {
    match diff.change_kind() {
        Some(BlockGroupChangeKind::Added) => SampleStatus::Added,
        Some(BlockGroupChangeKind::Removed) => SampleStatus::Removed,
        Some(BlockGroupChangeKind::Modified) => SampleStatus::Modified,
        None => unreachable!("block group diff has neither source nor target"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use gen_core::{DoltHashId, HashId};
    use gen_diff::graph::{DiffChange, DiffChangeKind, DiffGenGraph, DiffGraphEdge, DiffGraphNode};
    use gen_graph::{GraphEdge, GraphNode};
    use gen_models::block_group::BlockGroup;

    use super::{
        SampleStatus, build_explorer_entries, collect_samples, resolve_current_component,
        sample_label, sample_status_for_block_group,
    };
    use crate::views::diff_graph::{change_label_for_block_group, change_label_for_graph};

    fn graph_node(id: i64, start: i64, end: i64, is_new: bool) -> DiffGraphNode {
        DiffGraphNode {
            node: GraphNode {
                node_id: HashId::pad_str(id),
                sequence_start: start,
                sequence_end: end,
            },
            change: if is_new {
                DiffChange::new(DiffChangeKind::Added, Some(DoltHashId([100; 20])))
            } else {
                DiffChange::unchanged()
            },
        }
    }

    fn graph_edge(id: i64, is_new: bool) -> Vec<DiffGraphEdge> {
        vec![DiffGraphEdge {
            edge: GraphEdge {
                edge_id: HashId::pad_str(id),
                source_strand: gen_core::Strand::Forward,
                target_strand: gen_core::Strand::Forward,
                chromosome_index: 0,
                phased: 0,
                created_on: 0,
            },
            change: if is_new {
                DiffChange::new(DiffChangeKind::Added, Some(DoltHashId([100; 20])))
            } else {
                DiffChange::unchanged()
            },
        }]
    }

    fn block_group_diff_with_two_components() -> gen_diff::operations::BlockGroupDiff {
        let left_start = graph_node(1, 0, 3, true);
        let left_end = graph_node(2, 3, 6, true);
        let right_start = graph_node(3, 0, 2, true);
        let right_end = graph_node(4, 2, 4, true);

        let mut graph = DiffGenGraph::new();
        graph.add_edge(left_start, left_end, graph_edge(10, true));
        graph.add_edge(right_start, right_end, graph_edge(11, true));

        gen_diff::operations::BlockGroupDiff {
            id: HashId::pad_str(99),
            source_block_group: None,
            target_block_group: Some(BlockGroup {
                id: HashId::pad_str(99),
                collection_name: "Default".to_string(),
                sample_name: "sample".to_string(),
                name: "block-group".to_string(),
                created_on: 0,
                parent_block_group_id: None,
                is_default: false,
            }),
            graph,
        }
    }

    #[test]
    fn test_collect_samples_groups_new_sample_under_added() {
        let block_group_diff = block_group_diff_with_two_components();

        let samples = collect_samples(&[block_group_diff]);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].status, SampleStatus::Added);
        assert_eq!(samples[0].collection, "Default");
        assert_eq!(samples[0].sample, "sample");
        assert_eq!(samples[0].block_groups.len(), 1);
        assert_eq!(
            samples[0].block_groups[0].render.title,
            "Add Default sample block-group"
        );
    }

    #[test]
    fn test_build_explorer_entries_uses_inline_expansion() {
        let block_group_diff = block_group_diff_with_two_components();
        let samples = collect_samples(&[block_group_diff]);
        let mut expanded_samples = BTreeSet::new();
        expanded_samples.insert(0);

        let entries = build_explorer_entries(&samples, &expanded_samples);

        assert_eq!(entries[0].label, "Added Samples");
        assert_eq!(entries[1].label, "v sample");
        assert_eq!(entries[2].label, "  Add | block-group");
    }

    #[test]
    fn test_sample_label_omits_default_collection() {
        let block_group_diff = block_group_diff_with_two_components();
        let samples = collect_samples(&[block_group_diff]);

        assert_eq!(sample_label(&samples[0]), "sample");
    }

    #[test]
    fn test_sample_row_selects_first_block_group_graph() {
        let block_group_diff = block_group_diff_with_two_components();
        let samples = collect_samples(&[block_group_diff]);
        let mut expanded_samples = BTreeSet::new();
        expanded_samples.insert(0);
        let entries = build_explorer_entries(&samples, &expanded_samples);

        let component = resolve_current_component(&samples, &entries, 1)
            .expect("should resolve first block group from sample row");

        assert_eq!(component.block_group, "block-group");
    }

    #[test]
    fn test_change_label_mapping_stays_stable() {
        let mut added_graph = DiffGenGraph::new();
        let added_start = graph_node(10, 0, 3, true);
        let added_end = graph_node(11, 3, 6, true);
        added_graph.add_edge(added_start, added_end, graph_edge(12, true));
        assert_eq!(change_label_for_graph(&added_graph), "Add");

        let mut removed_graph = DiffGenGraph::new();
        let removed_start = DiffGraphNode {
            change: DiffChange::new(DiffChangeKind::Removed, Some(DoltHashId([100; 20]))),
            ..graph_node(20, 0, 3, false)
        };
        let removed_end = DiffGraphNode {
            change: DiffChange::new(DiffChangeKind::Removed, Some(DoltHashId([100; 20]))),
            ..graph_node(21, 3, 6, false)
        };
        let removed_edge = vec![DiffGraphEdge {
            change: DiffChange::new(DiffChangeKind::Removed, Some(DoltHashId([100; 20]))),
            ..graph_edge(22, false)[0]
        }];
        removed_graph.add_edge(removed_start, removed_end, removed_edge);
        assert_eq!(change_label_for_graph(&removed_graph), "Remove");

        let mut modified_graph = DiffGenGraph::new();
        let modified_start = DiffGraphNode {
            change: DiffChange::new(DiffChangeKind::Modified, Some(DoltHashId([100; 20]))),
            ..graph_node(30, 0, 3, false)
        };
        let modified_end = DiffGraphNode {
            change: DiffChange::new(DiffChangeKind::Modified, Some(DoltHashId([100; 20]))),
            ..graph_node(31, 3, 6, false)
        };
        let modified_edge = vec![DiffGraphEdge {
            change: DiffChange::new(DiffChangeKind::Modified, Some(DoltHashId([100; 20]))),
            ..graph_edge(32, false)[0]
        }];
        modified_graph.add_edge(modified_start, modified_end, modified_edge);
        assert_eq!(change_label_for_graph(&modified_graph), "Modify");

        let created_diff = gen_diff::operations::BlockGroupDiff {
            id: HashId::pad_str(200),
            source_block_group: None,
            target_block_group: Some(BlockGroup {
                id: HashId::pad_str(200),
                collection_name: "collection".to_string(),
                sample_name: "sample".to_string(),
                name: "block-group".to_string(),
                created_on: 0,
                parent_block_group_id: None,
                is_default: false,
            }),
            graph: DiffGenGraph::new(),
        };
        assert_eq!(change_label_for_block_group(&created_diff), "Created");
        assert_eq!(
            sample_status_for_block_group(&created_diff),
            SampleStatus::Added
        );
    }
}
