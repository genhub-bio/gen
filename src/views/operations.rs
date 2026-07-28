use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    time::Instant,
};

use crossterm::event::{self, KeyCode};
use gen_core::PATH_START_NODE_ID;
use gen_diff::operations::{
    BlockGroupChangeKind, BlockGroupDiff, DiffRange, collect_operation_diff,
};
use gen_graph::{GenGraph, GraphNode};
use gen_models::{db::DbContext, history::HistoryEntry};
use gen_tui::{graph_controller::GraphController, theme::current_theme};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    prelude::Style,
    style::Modifier,
    widgets::{Block, List, ListItem, Row, Table},
};

use crate::views::{
    diff_graph::{
        DiffGraphComponent, apply_diff_highlights, block_group_label, build_diff_graph_component,
        change_label_for_block_group,
    },
    gen_graph_widget::{GenGraphNodeSizer, create_gen_graph_controller, create_gen_graph_widget},
    panels::{PanelFocus, PanelStyles, panel_block, render_status_bar},
    tui_runtime::TuiSession,
};

fn clip_text(text: &str, limit: usize) -> String {
    let normalized = text.replace('\n', " ");
    if normalized.len() > limit.saturating_sub(3) {
        format!("{}...", &normalized[..limit - 3])
    } else {
        normalized
    }
}

struct OperationDiffComponent {
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

struct OperationSampleComponent {
    collection: String,
    sample: String,
    status: SampleStatus,
    block_groups: Vec<OperationDiffComponent>,
}

struct ExplorerEntry {
    label: String,
    sample_index: Option<usize>,
    block_group_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationPanel {
    Operations,
    GraphView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphViewFocus {
    List,
    GraphCanvas,
}

fn graph_subpanel_border_style(
    panel_focus: &PanelFocus<OperationPanel>,
    active_subpanel: GraphViewFocus,
    subpanel: GraphViewFocus,
    panel_styles: PanelStyles,
) -> Style {
    if panel_focus.is_active_panel(OperationPanel::GraphView) {
        if active_subpanel == subpanel {
            panel_styles.focused
        } else {
            panel_styles.unfocused
        }
    } else if panel_focus.is_navigation_selected(OperationPanel::GraphView) {
        panel_styles.selected
    } else {
        panel_styles.unfocused
    }
}

fn collect_diff_samples(graph_diffs: &[BlockGroupDiff]) -> Vec<OperationSampleComponent> {
    let mut grouped =
        BTreeMap::<(SampleStatus, String, String), Vec<OperationDiffComponent>>::new();

    for graph_diff in graph_diffs {
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
            .push(OperationDiffComponent {
                render: build_diff_graph_component(
                    &graph_diff.graph,
                    format!("{change_label} {}", block_group_label(graph_diff)),
                ),
                block_group,
                change_label,
            });
    }

    grouped
        .into_iter()
        .map(|((status, collection, sample), mut block_groups)| {
            block_groups.sort_by(|left, right| left.block_group.cmp(&right.block_group));
            OperationSampleComponent {
                collection,
                sample,
                status,
                block_groups,
            }
        })
        .collect()
}

fn load_diff_samples_for_entry(
    context: &DbContext,
    history_entry: &HistoryEntry,
) -> Vec<OperationSampleComponent> {
    let commit_hash = history_entry.commit_hash;
    let diffs = match collect_operation_diff(
        context.graph().conn(),
        history_entry.parent_hash,
        commit_hash,
        DiffRange::TwoDot,
    ) {
        Ok(diffs) => diffs,
        Err(_) => return Vec::new(),
    };
    collect_diff_samples(&diffs.diff_graph)
}

fn build_graph_controller(
    samples: &[OperationSampleComponent],
    entries: &[ExplorerEntry],
    selected_entry: usize,
    empty_graph: &GenGraph,
) -> GraphController<GenGraph, GenGraphNodeSizer> {
    if let Some(component) = resolve_current_component(samples, entries, selected_entry) {
        let mut controller = create_gen_graph_controller(component.render.graph.clone());
        apply_diff_highlights(&mut controller, &component.render);
        controller
    } else {
        create_gen_graph_controller(empty_graph.clone())
    }
}

fn build_explorer_entries(
    samples: &[OperationSampleComponent],
    expanded_samples: &BTreeSet<usize>,
) -> Vec<ExplorerEntry> {
    let mut entries: Vec<ExplorerEntry> = Vec::new();
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

fn sample_label(sample: &OperationSampleComponent) -> String {
    if sample.collection == "Default" || sample.collection.is_empty() {
        sample.sample.clone()
    } else {
        format!("{} | {}", sample.collection, sample.sample)
    }
}

fn resolve_current_component<'a>(
    samples: &'a [OperationSampleComponent],
    entries: &[ExplorerEntry],
    selected_entry: usize,
) -> Option<&'a OperationDiffComponent> {
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

pub fn view_operations(
    context: &DbContext,
    history_entries: &[HistoryEntry],
) -> Result<(), io::Error> {
    let conn = context.graph().conn();

    let mut session = TuiSession::enter()?;
    let terminal = session.terminal_mut();

    let mut empty_graph = GenGraph::new();
    empty_graph.add_node(GraphNode {
        node_id: PATH_START_NODE_ID,
        sequence_start: 0,
        sequence_end: 1,
    });

    let mut diff_samples: Vec<OperationSampleComponent> = Vec::new();
    let mut expanded_samples = BTreeSet::new();
    let mut entries: Vec<ExplorerEntry> = Vec::new();
    let mut selected_entry = 0usize;
    let mut graph_controller = create_gen_graph_controller(empty_graph.clone());

    let mut view_graph = false;
    let mut graph_view_focus = GraphViewFocus::List;
    let mut panel_focus = PanelFocus::new(OperationPanel::Operations);
    let panel_styles = PanelStyles::default();
    let status_bar_height: u16 = 1;

    let mut selected = 0usize;
    let mut last_frame_time = Instant::now();

    loop {
        let now = Instant::now();
        let frame_delta = now.duration_since(last_frame_time);
        last_frame_time = now;

        terminal.draw(|frame| {
            let rows: Vec<Row> = history_entries
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    let style = if index == selected {
                        Style::default().add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };

                    let short_hash = entry
                        .commit_hash
                        .to_string()
                        .chars()
                        .take(12)
                        .collect::<String>();
                    Row::new(vec![
                        short_hash,
                        clip_text(&entry.date, 24),
                        clip_text(&entry.message, 72),
                    ])
                    .style(style)
                })
                .collect();

            let table = Table::new(
                rows,
                [
                    Constraint::Length(14),
                    Constraint::Length(24),
                    Constraint::Min(24),
                ],
            )
            .header(
                Row::new(vec!["Commit", "Date", "Summary"])
                    .style(Style::default().add_modifier(Modifier::UNDERLINED)),
            )
            .block(panel_block(
                "Operations",
                &panel_focus,
                OperationPanel::Operations,
                panel_styles,
            ));

            let outer_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![
                    Constraint::Min(1),
                    Constraint::Length(status_bar_height),
                ])
                .split(frame.area());

            let main_area = outer_layout[0];
            let status_bar_area = outer_layout[1];

            let panel_messages = if panel_focus.is_navigation() {
                let mut message = "*tab/arrows* navigate | *enter* activate panel".to_string();
                if panel_focus.current() == OperationPanel::GraphView {
                    message.push_str(" | *x* close panel");
                }
                message.push_str(" | *q* quit");
                message
            } else {
                match panel_focus.current() {
                    OperationPanel::Operations => {
                        "*↑↓* select | *v* view graph | *esc* leave panel".to_string()
                    }
                    OperationPanel::GraphView => {
                        if graph_view_focus == GraphViewFocus::List {
                            "*↑↓* select | *enter/right* expand | *left* collapse | *tab* graph | *esc* leave panel"
                                .to_string()
                        } else {
                            "*←→↑↓* pan | *+/-* zoom | *tab* list | *esc* leave panel"
                                .to_string()
                        }
                    }
                }
            };

            let canvas_area = if view_graph {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(main_area);
                frame.render_widget(table, chunks[0]);
                Some(chunks[1])
            } else {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(100)].as_ref())
                    .split(main_area);
                frame.render_widget(table, chunks[0]);
                None
            };

            if let Some(canvas_area) = canvas_area {
                let graph_panel = panel_block(
                    "Operation Diff",
                    &panel_focus,
                    OperationPanel::GraphView,
                    panel_styles,
                );
                let panel_inner = graph_panel.inner(canvas_area);
                frame.render_widget(graph_panel, canvas_area);

                let graph_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Length(60), Constraint::Min(1)])
                    .split(panel_inner);

                let list_items: Vec<ListItem> = entries
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| {
                        let style = if index == selected_entry {
                            Style::default()
                                .fg(ratatui::style::Color::Cyan)
                                .add_modifier(Modifier::BOLD)
                        } else if entry.sample_index.is_none() {
                            Style::default().add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        };
                        ListItem::new(entry.label.clone()).style(style)
                    })
                    .collect();

                let list_border_style = graph_subpanel_border_style(
                    &panel_focus,
                    graph_view_focus,
                    GraphViewFocus::List,
                    panel_styles,
                );
                let list = List::new(list_items).block(
                    Block::default()
                        .title("Samples")
                        .borders(ratatui::widgets::Borders::ALL)
                        .border_style(list_border_style),
                );
                frame.render_widget(list, graph_chunks[0]);

                let graph_title = if let Some(component) =
                    resolve_current_component(&diff_samples, &entries, selected_entry)
                {
                    component.render.title.clone()
                } else {
                    "No Diff Graph".to_string()
                };
                let graph_border_style = graph_subpanel_border_style(
                    &panel_focus,
                    graph_view_focus,
                    GraphViewFocus::GraphCanvas,
                    panel_styles,
                );
                let graph_block = Block::default()
                    .title(graph_title)
                    .borders(ratatui::widgets::Borders::ALL)
                    .border_style(graph_border_style);
                let inner_canvas = graph_block.inner(graph_chunks[1]);
                frame.render_widget(graph_block, graph_chunks[1]);

                graph_controller.viewport_state.focus();
                graph_controller.viewport_state.viewport_bounds = inner_canvas;
                graph_controller.update_animations(frame_delta);

                let canvas_style = Style::default().bg(current_theme()[0x00]);
                let widget = create_gen_graph_widget(conn)
                    .detail_level(graph_controller.get_detail_level())
                    .style(canvas_style)
                    .cursor();
                frame.render_stateful_widget(widget, inner_canvas, &mut graph_controller);
            }

            render_status_bar(frame, status_bar_area, &panel_messages);
        })?;

        if event::poll(std::time::Duration::from_millis(100))?
            && let event::Event::Key(key) = event::read()?
        {
            if panel_focus.is_navigation() {
                match key.code {
                    KeyCode::Tab => {
                        panel_focus.cycle_next();
                    }
                    KeyCode::BackTab => {
                        panel_focus.cycle_prev();
                    }
                    KeyCode::Up => {
                        panel_focus.focus(OperationPanel::Operations);
                    }
                    KeyCode::Down => {
                        if panel_focus.current() == OperationPanel::Operations && view_graph {
                            panel_focus.focus(OperationPanel::GraphView);
                        }
                    }
                    KeyCode::Enter => {
                        panel_focus.activate();
                    }
                    KeyCode::Char('x') => {
                        if panel_focus.current() == OperationPanel::GraphView {
                            view_graph = false;
                            panel_focus.remove_panel(OperationPanel::GraphView);
                        }
                    }
                    KeyCode::Char('q') => {
                        break;
                    }
                    _ => {}
                }
            } else if key.code == KeyCode::Esc {
                panel_focus.deactivate();
            } else {
                match panel_focus.current() {
                    OperationPanel::Operations => match key.code {
                        KeyCode::Up => {
                            if selected > 0 {
                                selected = selected.saturating_sub(1);
                            }
                            if view_graph {
                                diff_samples = load_diff_samples_for_entry(
                                    context,
                                    &history_entries[selected],
                                );
                                expanded_samples.clear();
                                if !diff_samples.is_empty() {
                                    expanded_samples.insert(0);
                                }
                                entries = build_explorer_entries(&diff_samples, &expanded_samples);
                                selected_entry = first_selectable_entry(&entries).unwrap_or(0);
                                graph_controller = build_graph_controller(
                                    &diff_samples,
                                    &entries,
                                    selected_entry,
                                    &empty_graph,
                                );
                            }
                        }
                        KeyCode::Down => {
                            if selected + 1 < history_entries.len() {
                                selected += 1;
                            }
                            if view_graph {
                                diff_samples = load_diff_samples_for_entry(
                                    context,
                                    &history_entries[selected],
                                );
                                expanded_samples.clear();
                                if !diff_samples.is_empty() {
                                    expanded_samples.insert(0);
                                }
                                entries = build_explorer_entries(&diff_samples, &expanded_samples);
                                selected_entry = first_selectable_entry(&entries).unwrap_or(0);
                                graph_controller = build_graph_controller(
                                    &diff_samples,
                                    &entries,
                                    selected_entry,
                                    &empty_graph,
                                );
                            }
                        }
                        KeyCode::Char('v') => {
                            view_graph = true;
                            graph_view_focus = GraphViewFocus::List;
                            panel_focus.include_panel(OperationPanel::GraphView);
                            panel_focus.focus(OperationPanel::GraphView);
                            panel_focus.activate();

                            diff_samples =
                                load_diff_samples_for_entry(context, &history_entries[selected]);
                            expanded_samples.clear();
                            if !diff_samples.is_empty() {
                                expanded_samples.insert(0);
                            }
                            entries = build_explorer_entries(&diff_samples, &expanded_samples);
                            selected_entry = first_selectable_entry(&entries).unwrap_or(0);
                            graph_controller = build_graph_controller(
                                &diff_samples,
                                &entries,
                                selected_entry,
                                &empty_graph,
                            );
                        }
                        _ => {}
                    },
                    OperationPanel::GraphView => {
                        if key.code == KeyCode::Tab || key.code == KeyCode::BackTab {
                            graph_view_focus = if graph_view_focus == GraphViewFocus::List {
                                GraphViewFocus::GraphCanvas
                            } else {
                                GraphViewFocus::List
                            };
                        } else if graph_view_focus == GraphViewFocus::List {
                            match key.code {
                                KeyCode::Up => {
                                    if let Some(previous_entry) =
                                        previous_selectable_entry(&entries, selected_entry)
                                    {
                                        selected_entry = previous_entry;
                                        graph_controller = build_graph_controller(
                                            &diff_samples,
                                            &entries,
                                            selected_entry,
                                            &empty_graph,
                                        );
                                    }
                                }
                                KeyCode::Down => {
                                    if let Some(next_entry) =
                                        next_selectable_entry(&entries, selected_entry)
                                    {
                                        selected_entry = next_entry;
                                        graph_controller = build_graph_controller(
                                            &diff_samples,
                                            &entries,
                                            selected_entry,
                                            &empty_graph,
                                        );
                                    }
                                }
                                KeyCode::Enter | KeyCode::Right => {
                                    if let Some(sample_index) = entries[selected_entry].sample_index
                                        && entries[selected_entry].block_group_index.is_none()
                                    {
                                        expanded_samples.insert(sample_index);
                                        entries = build_explorer_entries(
                                            &diff_samples,
                                            &expanded_samples,
                                        );
                                        selected_entry = sample_row_index(&entries, sample_index)
                                            .unwrap_or(selected_entry);
                                        graph_controller = build_graph_controller(
                                            &diff_samples,
                                            &entries,
                                            selected_entry,
                                            &empty_graph,
                                        );
                                    }
                                }
                                KeyCode::Left => {
                                    if let Some(sample_index) = entries[selected_entry].sample_index
                                        && (entries[selected_entry].block_group_index.is_some()
                                            || expanded_samples.contains(&sample_index))
                                    {
                                        expanded_samples.remove(&sample_index);
                                        entries = build_explorer_entries(
                                            &diff_samples,
                                            &expanded_samples,
                                        );
                                        selected_entry = sample_row_index(&entries, sample_index)
                                            .unwrap_or(selected_entry);
                                        graph_controller = build_graph_controller(
                                            &diff_samples,
                                            &entries,
                                            selected_entry,
                                            &empty_graph,
                                        );
                                    }
                                }
                                _ => {}
                            }
                        } else {
                            let _ = graph_controller.handle_key_event(key);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use gen_core::{BranchName, CommitRef, DoltHashId, HashId};
    use gen_diff::{
        graph::{DiffChange, DiffChangeKind, DiffGenGraph, DiffGraphEdge, DiffGraphNode},
        operations::BlockGroupDiff,
    };
    use gen_graph::{GraphEdge, GraphNode};
    use gen_models::{
        block_group::BlockGroup,
        collection::Collection,
        history::{HistoryStore, dolt::DoltHistoryStore},
    };

    use super::{
        SampleStatus, build_explorer_entries, collect_diff_samples, load_diff_samples_for_entry,
        sample_label, sample_status_for_block_group,
    };
    use crate::test_helpers::{create_bg, setup_gen};

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

    fn block_group_diff_with_two_components() -> BlockGroupDiff {
        let left_start = graph_node(1, 0, 3, true);
        let left_end = graph_node(2, 3, 6, true);
        let right_start = graph_node(3, 0, 2, true);
        let right_end = graph_node(4, 2, 4, true);

        let mut graph = DiffGenGraph::new();
        graph.add_edge(left_start, left_end, graph_edge(10, true));
        graph.add_edge(right_start, right_end, graph_edge(11, true));

        BlockGroupDiff {
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
    fn test_collect_diff_samples_keeps_block_group_unified() {
        let block_group_diff = block_group_diff_with_two_components();

        let samples = collect_diff_samples(&[block_group_diff]);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].status, SampleStatus::Added);
        assert_eq!(samples[0].block_groups.len(), 1);
        assert_eq!(
            samples[0].block_groups[0].render.title,
            "Add Default sample block-group"
        );
    }

    #[test]
    fn test_sample_label_omits_default_collection() {
        let block_group_diff = block_group_diff_with_two_components();
        let samples = collect_diff_samples(&[block_group_diff]);

        assert_eq!(sample_label(&samples[0]), "sample");
    }

    #[test]
    fn test_build_explorer_entries_expands_sample_inline() {
        let block_group_diff = block_group_diff_with_two_components();
        let samples = collect_diff_samples(&[block_group_diff]);
        let mut expanded_samples = BTreeSet::new();
        expanded_samples.insert(0);

        let entries = build_explorer_entries(&samples, &expanded_samples);

        assert_eq!(entries[0].label, "Added Samples");
        assert_eq!(entries[1].label, "v sample");
        assert_eq!(entries[2].label, "  Add | block-group");
    }

    #[test]
    fn test_sample_status_for_created_block_group_is_added() {
        let block_group_diff = block_group_diff_with_two_components();

        assert_eq!(
            sample_status_for_block_group(&block_group_diff),
            SampleStatus::Added
        );
    }

    // The operation list includes both sides of a merge, so adjacent entries
    // need not be parent and child. The graph must use the parent carried by
    // the selected history entry or it can display changes from the wrong side.
    #[test]
    fn test_merge_operation_diff_uses_history_entry_parent() {
        let context = setup_gen();
        let graph = context.graph().conn();
        let history = DoltHistoryStore::new(graph);

        Collection::create(graph, "merge-view").expect("should create collection");
        create_bg(graph, "merge-view", "base-sample", "base-region");
        history
            .commit_all("base")
            .expect("should commit base state");
        history
            .create_branch(&BranchName("feature".to_string()), None)
            .expect("should create feature branch");
        history
            .checkout_branch(&BranchName("feature".to_string()))
            .expect("should checkout feature branch");
        create_bg(graph, "merge-view", "feature-sample", "feature-region");
        history
            .commit_all("feature")
            .expect("should commit feature state");

        history
            .checkout_branch(&BranchName("main".to_string()))
            .expect("should checkout main branch");
        create_bg(graph, "merge-view", "main-sample", "main-region");
        let main_commit = history
            .commit_all("main")
            .expect("should commit main state");
        history
            .merge(&CommitRef("feature".to_string()))
            .expect("should merge feature branch");
        let merge_commit = history
            .current_head()
            .expect("should query merged head")
            .expect("should create merge commit");
        let history_entries = history.log(None).expect("should load merged history");
        let merge_entry = history_entries
            .iter()
            .find(|entry| entry.commit_hash == merge_commit)
            .expect("should include merge commit");

        assert_eq!(
            merge_entry.parent_hash,
            Some(main_commit),
            "merge history entry should record the first parent"
        );
        let samples = load_diff_samples_for_entry(&context, merge_entry);
        assert!(
            samples
                .iter()
                .any(|sample| sample.sample == "feature-sample"),
            "merge diff should show feature state added relative to main"
        );
        assert!(
            samples.iter().all(|sample| sample.sample != "main-sample"),
            "merge diff should not treat main state as newly added"
        );
    }
}
