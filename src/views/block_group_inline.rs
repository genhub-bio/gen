use std::{io, panic, time::Duration};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use gen_core::{HashId, Workspace};
use gen_models::{db::GraphConnection, path::Path};
use gen_tui::{layout::VisualDetail, theme::current_theme};
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    prelude::*,
    widgets::{Block, Borders},
};

use crate::views::{
    block_group::discard_pending_input,
    gen_graph_controller::{AnnotationDisplay, GenGraphController, GraphKeyOutcome},
    graph_overlay::has_path_overlay,
};

#[derive(Debug)]
pub enum AppEvent {
    KeyPress(KeyEvent),
    Resize(u16, u16),
}

pub trait EventSource {
    fn poll_next(&mut self, timeout: Duration) -> Option<AppEvent>;

    /// Drop every event already waiting, returning whether one of them was a resize.
    fn discard_pending(&mut self) -> bool;
}

/// Reads key/resize events straight from the terminal. Rendering is event-driven (no
/// animation to advance), so this blocks on `poll` rather than waking at a fixed rate.
pub struct CrosstermEventSource;

impl EventSource for CrosstermEventSource {
    fn poll_next(&mut self, timeout: Duration) -> Option<AppEvent> {
        if event::poll(timeout).unwrap_or(false) {
            match event::read().unwrap() {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    return Some(AppEvent::KeyPress(k));
                }
                Event::Resize(w, h) => {
                    return Some(AppEvent::Resize(w, h));
                }
                _ => {}
            }
        }
        None
    }

    fn discard_pending(&mut self) -> bool {
        discard_pending_input().unwrap_or(false)
    }
}

/// How the inline widget was left.
pub enum InlineOutcome<'a> {
    /// The user closed the widget.
    Closed,
    /// The user asked for the full-screen viewer, which keeps drawing this controller.
    OpenFullViewer(Box<GenGraphController<'a>>),
}

/// What a key press asks of the event loop.
#[derive(Debug, PartialEq)]
enum KeyOutcome {
    Exit {
        upgrade_requested: bool,
    },
    Redraw,
    /// Redraw, then discard the input queued while the door's batch loaded.
    EnteredDoor,
    Ignore,
}

/// Apply `key` to `controller`: the inline widget's own exit keys, then the graph keys every
/// viewer shares.
fn handle_key(controller: &mut GenGraphController, key: KeyEvent) -> KeyOutcome {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => KeyOutcome::Exit {
            upgrade_requested: false,
        },
        KeyCode::Char('f') => KeyOutcome::Exit {
            upgrade_requested: true,
        },
        _ => match controller.handle_key(key) {
            GraphKeyOutcome::Redraw => KeyOutcome::Redraw,
            GraphKeyOutcome::EnteredDoor => KeyOutcome::EnteredDoor,
            GraphKeyOutcome::Ignore => KeyOutcome::Ignore,
        },
    }
}

/// Draw one frame, then once more if drawing claimed a batch whose dimming, annotation
/// groups, or path highlight differ from what was just drawn.
fn draw_synced<B: Backend>(
    terminal: &mut Terminal<B>,
    controller: &mut GenGraphController,
) -> Result<(), B::Error> {
    // A door handled since the last draw may already have grown the graph.
    controller.sync_active_world();
    terminal.draw(|frame| render_inline(frame, controller))?;
    if controller.sync_active_world().changed {
        terminal.draw(|frame| render_inline(frame, controller))?;
    }
    Ok(())
}

/// Run the widget until the user exits, returning whether they asked for the full viewer.
fn run_inline_event_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    controller: &mut GenGraphController,
    events: &mut impl EventSource,
) -> Result<bool, B::Error> {
    draw_synced(terminal, controller)?;
    loop {
        // Rendering is event-driven (no animation to advance): block indefinitely until the
        // next input event wakes us.
        let Some(event) = events.poll_next(Duration::from_secs(3600)) else {
            continue;
        };
        let entered_door = match event {
            AppEvent::KeyPress(key) => match handle_key(controller, key) {
                KeyOutcome::Exit { upgrade_requested } => return Ok(upgrade_requested),
                KeyOutcome::Redraw => false,
                KeyOutcome::EnteredDoor => true,
                KeyOutcome::Ignore => continue,
            },
            // Width is picked up automatically from `frame.area()` on the next render; the
            // inline viewport's height stays fixed.
            AppEvent::Resize(_, _) => false,
        };
        draw_synced(terminal, controller)?;
        // Keys pressed while the new batch loaded were aimed at the old screen.
        if entered_door && events.discard_pending() {
            draw_synced(terminal, controller)?;
        }
    }
}

/// Display an inline widget for a `BlockGroup`'s graph, with annotations loaded.
///
/// The widget appears inline in the terminal without taking over the entire screen. Like the
/// full viewer, it loads the graph lazily one batch at a time and steps between batches
/// through their doors.
///
/// # Controls
/// * Arrow keys: Navigate cursor between nodes and pan the view
/// * +/-: Zoom in/out (Minimal → Truncated → Full)
/// * p: Toggle the highlight of the last of `paths`
/// * f: Switch to the full-screen viewer
/// * q/Enter/Esc: Exit the widget
///
/// # Returns
/// * `Ok(InlineOutcome::OpenFullViewer(_))` if the user requested the full-screen view,
///   carrying the controller for it to keep drawing
/// * `Ok(InlineOutcome::Closed)` if completed successfully and exited
pub fn show_inline_block_group_widget<'a>(
    conn: &'a GraphConnection,
    workspace: &'a Workspace,
    block_group_id: HashId,
    paths: Vec<Path>,
    height: u16,
    history_ref: Option<&'a str>,
) -> io::Result<InlineOutcome<'a>> {
    let mut controller =
        GenGraphController::for_block_group(conn, workspace, &block_group_id, history_ref)
            .map_err(|error| io::Error::other(error.to_string()))?;
    // The inline widget follows the keyboard cursor from the start.
    controller.view_state_mut().show_cursor();
    for path in paths {
        controller.add_path(&path);
    }

    let terminal_result = panic::catch_unwind(|| {
        ratatui::init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })
    });

    match terminal_result {
        Ok(mut terminal) => {
            let upgrade_requested =
                run_inline_event_loop(&mut terminal, &mut controller, &mut CrosstermEventSource)?;

            // Final render without border -> capture the viewport area
            let viewport_area = terminal.get_frame().area();

            terminal.draw(|frame| render_final(frame, &mut controller))?;

            // For inline viewports, we need to manually restore terminal state
            // (ratatui::restore() loses the cursor which resets cursor position incorrectly.

            // Position cursor at the end of the viewport BEFORE restoring terminal mode
            let target_line = viewport_area.y + viewport_area.height;
            let _ =
                crossterm::execute!(std::io::stdout(), crossterm::cursor::MoveTo(0, target_line));

            // Now restore terminal modes manually (show cursor, disable raw mode)
            let _ = crossterm::execute!(std::io::stdout(), crossterm::cursor::Show);
            let _ = crossterm::terminal::disable_raw_mode();

            std::io::Write::flush(&mut std::io::stdout()).ok();

            if upgrade_requested {
                Ok(InlineOutcome::OpenFullViewer(Box::new(controller)))
            } else {
                Ok(InlineOutcome::Closed)
            }
        }
        Err(_) => {
            eprintln!("Interactive terminal not available, omitting visualization.");
            Ok(InlineOutcome::Closed)
        }
    }
}

/// Draw the inline widget with a border and controls help
fn render_inline(frame: &mut Frame, controller: &mut GenGraphController) {
    let area = frame.area();

    // Ratatui layout (not graph layout) - split main area for graph box and controls
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    let block = Block::default().borders(Borders::ALL);
    let inner_area = block.inner(main_layout[0]);

    // Render the border and content
    frame.render_widget(block, main_layout[0]);

    let any_hidden = controller.render(
        frame,
        inner_area,
        AnnotationDisplay::FloatingLabels,
        Style::default(),
    );
    let hidden_legend = any_hidden.then(|| {
        if controller.detail_level() == VisualDetail::Full {
            "* some annotations hidden due to space constraints"
        } else {
            "* zoom in for more features"
        }
    });
    draw_controls_help(frame, main_layout[1], controller, hidden_legend);
}

/// Draw the final plot after the widget is done
fn render_final(frame: &mut Frame, controller: &mut GenGraphController) {
    let area = frame.area().offset(ratatui::layout::Offset { x: 0, y: -1 });
    // The final render omits the cursor overlay.
    controller.render_plain(frame, area);
}

/// Draw the bottom controls line. When `hidden_legend` is set, it's right-aligned on the
/// same line and the path-visibility shortcut is dropped to make room for it.
fn draw_controls_help(
    frame: &mut Frame,
    area: Rect,
    controller: &GenGraphController,
    hidden_legend: Option<&str>,
) {
    let help_text = if hidden_legend.is_some() {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | q: Exit".to_string()
    } else if has_path_overlay(controller.overlays()) {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | p: Hide Path | q: Exit".to_string()
    } else {
        "←→↑↓: Nav | +/-: Zoom | f: Full window | p: Show Path | q: Exit".to_string()
    };

    let buf = frame.buffer_mut();
    buf.set_string(
        area.x,
        area.y,
        &help_text,
        Style::default().fg(Color::Yellow),
    );

    if let Some(legend) = hidden_legend {
        let help_width = help_text.chars().count() as u16;
        let legend_width = legend.chars().count() as u16;
        let legend_x = area.right().saturating_sub(legend_width);
        if legend_x > area.x + help_width {
            buf.set_string(
                legend_x,
                area.y,
                legend,
                Style::default().fg(current_theme()[0x09]),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crossterm::event::KeyModifiers;
    use gen_models::db::get_connection;
    use petgraph::Direction::Incoming;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::views::{
        gen_graph_widget::DEFAULT_ZOOM_LEVEL,
        lazy_graph_source::tests::{setup_circular_block_group, setup_labelled_chain_block_group},
    };

    /// Replays a fixed list of key presses, then exits. The whole remaining script counts as
    /// already queued, so discarding pending input drops all of it.
    struct ScriptedEvents(VecDeque<KeyCode>);

    impl EventSource for ScriptedEvents {
        fn poll_next(&mut self, _timeout: Duration) -> Option<AppEvent> {
            let code = self.0.pop_front().unwrap_or(KeyCode::Char('q'));
            Some(AppEvent::KeyPress(KeyEvent::new(code, KeyModifiers::NONE)))
        }

        fn discard_pending(&mut self) -> bool {
            self.0.clear();
            false
        }
    }

    fn chain_labels(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("n{index}")).collect()
    }

    fn run_script(controller: &mut GenGraphController, width: u16, keys: Vec<KeyCode>) -> bool {
        let mut terminal =
            Terminal::new(TestBackend::new(width, 12)).expect("should create a test terminal");
        run_inline_event_loop(&mut terminal, controller, &mut ScriptedEvents(keys.into()))
            .expect("should run the inline event loop")
    }

    #[test]
    fn test_inline_state_loads_only_the_first_batch_of_a_long_chain() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels = chain_labels(80);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();

        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
                .expect("should load the block group");
        assert_eq!(controller.view_state().zoom_index, DEFAULT_ZOOM_LEVEL);
        assert!(controller.engine().graph().node_count() <= 2);

        assert!(!run_script(&mut controller, 12, Vec::new()));
        let loaded = controller.engine().graph().node_count();
        assert!(
            loaded > 2 && loaded < labels.len(),
            "only the first batch should be loaded, got {loaded} nodes"
        );
    }

    #[test]
    fn test_inline_keyboard_door_leaves_the_first_batch() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels = chain_labels(80);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
                .expect("should load the block group");

        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        draw_synced(&mut terminal, &mut controller).expect("should draw the first batch");
        let first_batch = controller.engine().active_batch();
        assert!(first_batch.is_some());

        let right = KeyEvent::new(KeyCode::Right, KeyModifiers::NONE);
        for _ in 0..2000 {
            if handle_key(&mut controller, right) != KeyOutcome::Ignore {
                draw_synced(&mut terminal, &mut controller).expect("should draw");
            }
            if controller.engine().active_batch() != first_batch {
                break;
            }
        }
        let active_batch = controller.engine().active_batch();
        assert!(active_batch.is_some());
        assert_ne!(active_batch, first_batch);
    }

    #[test]
    fn test_inline_door_discards_keys_queued_while_loading() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let labels = chain_labels(80);
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &label_refs);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
                .expect("should load the block group");
        controller.view_state_mut().show_cursor();

        let mut events = ScriptedEvents(vec![KeyCode::Right; 2000].into());
        let mut terminal =
            Terminal::new(TestBackend::new(12, 12)).expect("should create a test terminal");
        run_inline_event_loop(&mut terminal, &mut controller, &mut events)
            .expect("should run the inline event loop");

        assert!(
            events.0.is_empty(),
            "the presses left after the door should be discarded"
        );
        // Nothing past the door moved the cursor: it is still on the door's target, whose
        // predecessor lies in the batch the door led out of.
        let engine = controller.engine();
        let cursor = controller
            .view_state()
            .cursor
            .node
            .expect("should keep a cursor");
        let cursor_batch = engine.batch_of(cursor);
        assert!(cursor_batch.is_some());
        let predecessor = engine
            .graph()
            .neighbors_directed(cursor, Incoming)
            .next()
            .expect("should have a predecessor");
        assert_ne!(engine.batch_of(predecessor), cursor_batch);
    }

    #[test]
    fn test_inline_exit_keys_leave_the_widget() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, _) = setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
                .expect("should load the block group");

        let press = |code| KeyEvent::new(code, KeyModifiers::NONE);
        assert_eq!(
            handle_key(&mut controller, press(KeyCode::Char('f'))),
            KeyOutcome::Exit {
                upgrade_requested: true
            }
        );
        assert_eq!(
            handle_key(&mut controller, press(KeyCode::Esc)),
            KeyOutcome::Exit {
                upgrade_requested: false
            }
        );
        assert_eq!(
            handle_key(&mut controller, press(KeyCode::Char('x'))),
            KeyOutcome::Ignore
        );
    }

    #[test]
    fn test_inline_path_toggle_highlights_a_lazily_loaded_graph() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, edge_ids) =
            setup_labelled_chain_block_group(&db_path, &["x", "y", "z"]);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "chain", &block_group_id, &edge_ids).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
                .expect("should load the block group");
        controller.add_path(&path);

        run_script(&mut controller, 80, vec![KeyCode::Char('p')]);

        assert!(has_path_overlay(controller.overlays()));
        assert!(!controller.view_state().highlights.styles.is_empty());
    }

    #[test]
    fn test_inline_path_toggle_on_a_circular_block_group_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");
        let (block_group_id, edge_ids) = setup_circular_block_group(&db_path);
        let conn = get_connection(&db_path).unwrap();
        let path = Path::create(&conn, "circle", &block_group_id, &edge_ids).unwrap();
        let workspace = Workspace::from_current_dir();
        let mut controller =
            GenGraphController::for_block_group(&conn, &workspace, &block_group_id, None)
                .expect("should load the block group");
        controller.add_path(&path);

        run_script(
            &mut controller,
            80,
            vec![KeyCode::Char('p'), KeyCode::Char('p')],
        );

        assert!(!has_path_overlay(controller.overlays()));
    }
}
