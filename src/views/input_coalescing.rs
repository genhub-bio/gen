use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};

/// The most identical movement presses kept from one unbroken run. A slow frame lets key
/// repeats pile up; applying all of them makes the camera jump on the next frame, so a
/// backlog is trimmed to a few steps per key and the viewer keeps up with the keyboard
/// instead of replaying it.
pub const MAX_MOVEMENT_REPEATS: usize = 3;

/// The events to apply this frame, in their original order, and how many were dropped.
#[derive(Debug, PartialEq)]
pub struct CoalescedEvents {
    pub events: Vec<Event>,
    pub dropped: usize,
}

/// Whether a key press moves or zooms the camera by one fixed step per event, so that a run
/// of them can be shortened without changing what any single press does.
fn is_continuous_movement(code: KeyCode) -> bool {
    matches!(
        code,
        KeyCode::Left
            | KeyCode::Right
            | KeyCode::Up
            | KeyCode::Down
            | KeyCode::Char('h' | 'j' | 'k' | 'l' | 'w' | 'b' | 'e' | '+' | '=' | '-')
    )
}

/// Whether an event can move keyboard focus, or leave the viewer, so that events after it
/// may be meant for a different widget than the canvas.
fn can_change_focus(event: &Event) -> bool {
    match event {
        Event::Key(key) => matches!(
            key.code,
            KeyCode::Tab
                | KeyCode::BackTab
                | KeyCode::Esc
                | KeyCode::Enter
                | KeyCode::Char('m' | 'g' | 'q')
        ),
        Event::Mouse(mouse) => matches!(mouse.kind, MouseEventKind::Down(_)),
        _ => false,
    }
}

/// Shorten runs of repeated movement key presses that queued up while a frame was slow, so
/// one frame applies at most [`MAX_MOVEMENT_REPEATS`] steps per run.
///
/// A run is consecutive presses of the same key with the same modifiers, so any event but
/// pointer motion between two presses starts a new run and nothing is ever reordered. Every other event, such as
/// layout toggles, quitting, search, mouse clicks, resizes and explorer or text input, passes
/// through untouched. Only the canvas turns key presses into steps, so nothing is dropped
/// unless `canvas_focused`, and an event that can move focus ends coalescing for the rest of
/// the batch because later keys may belong to another widget.
pub fn coalesce_movement_events(events: Vec<Event>, canvas_focused: bool) -> CoalescedEvents {
    let mut kept = Vec::with_capacity(events.len());
    let mut dropped = 0;
    let mut coalescing = canvas_focused;
    let mut run: Option<(KeyCode, KeyModifiers, usize)> = None;
    for event in events {
        let movement = match &event {
            Event::Key(key)
                if coalescing
                    && key.kind == KeyEventKind::Press
                    && is_continuous_movement(key.code) =>
            {
                Some((key.code, key.modifiers))
            }
            _ => None,
        };
        let Some((code, modifiers)) = movement else {
            // Pointer motion never reaches the camera, so it does not break up a run.
            if !matches!(&event, Event::Mouse(mouse) if mouse.kind == MouseEventKind::Moved) {
                run = None;
            }
            if can_change_focus(&event) {
                coalescing = false;
            }
            kept.push(event);
            continue;
        };
        let length = match run {
            Some((run_code, run_modifiers, length))
                if run_code == code && run_modifiers == modifiers =>
            {
                length + 1
            }
            _ => 1,
        };
        run = Some((code, modifiers, length));
        if length > MAX_MOVEMENT_REPEATS {
            dropped += 1;
        } else {
            kept.push(event);
        }
    }
    CoalescedEvents {
        events: kept,
        dropped,
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{
        Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, MouseEvent,
        MouseEventKind,
    };

    use super::{MAX_MOVEMENT_REPEATS, coalesce_movement_events};

    fn press(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn press_with(code: KeyCode, modifiers: KeyModifiers) -> Event {
        Event::Key(KeyEvent::new(code, modifiers))
    }

    #[test]
    fn test_run_of_identical_pan_keys_collapses_to_the_cap() {
        let events = vec![press(KeyCode::Right); 20];

        let result = coalesce_movement_events(events, true);

        assert_eq!(
            result.events,
            vec![press(KeyCode::Right); MAX_MOVEMENT_REPEATS]
        );
        assert_eq!(result.dropped, 20 - MAX_MOVEMENT_REPEATS);
    }

    #[test]
    fn test_short_runs_are_left_alone() {
        let events = vec![press(KeyCode::Left); MAX_MOVEMENT_REPEATS];

        let result = coalesce_movement_events(events.clone(), true);

        assert_eq!(result.events, events);
        assert_eq!(result.dropped, 0);
    }

    #[test]
    fn test_zoom_keys_collapse_like_pan_keys() {
        let events = vec![press(KeyCode::Char('+')); 10];

        let result = coalesce_movement_events(events, true);

        assert_eq!(result.events.len(), MAX_MOVEMENT_REPEATS);
    }

    #[test]
    fn test_discrete_key_splits_runs_without_reordering() {
        let mut events = vec![press(KeyCode::Right); 5];
        events.push(press(KeyCode::Char('s')));
        events.extend(vec![press(KeyCode::Right); 5]);

        let result = coalesce_movement_events(events, true);

        let mut expected = vec![press(KeyCode::Right); MAX_MOVEMENT_REPEATS];
        expected.push(press(KeyCode::Char('s')));
        expected.extend(vec![press(KeyCode::Right); MAX_MOVEMENT_REPEATS]);
        assert_eq!(result.events, expected);
        assert_eq!(result.dropped, 2 * (5 - MAX_MOVEMENT_REPEATS));
    }

    #[test]
    fn test_alternating_keys_are_never_merged() {
        let events: Vec<Event> = (0..12)
            .map(|index| {
                press(if index % 2 == 0 {
                    KeyCode::Left
                } else {
                    KeyCode::Right
                })
            })
            .collect();

        let result = coalesce_movement_events(events.clone(), true);

        assert_eq!(result.events, events);
        assert_eq!(result.dropped, 0);
    }

    #[test]
    fn test_modifiers_distinguish_runs() {
        let mut events = vec![press(KeyCode::Right); 4];
        events.extend(vec![press_with(KeyCode::Right, KeyModifiers::SHIFT); 4]);

        let result = coalesce_movement_events(events, true);

        let mut expected = vec![press(KeyCode::Right); MAX_MOVEMENT_REPEATS];
        expected.extend(vec![
            press_with(KeyCode::Right, KeyModifiers::SHIFT);
            MAX_MOVEMENT_REPEATS
        ]);
        assert_eq!(result.events, expected);
    }

    #[test]
    fn test_non_key_events_pass_through_untouched() {
        let resize = Event::Resize(80, 24);
        let moved = Event::Mouse(MouseEvent {
            kind: MouseEventKind::Moved,
            column: 3,
            row: 4,
            modifiers: KeyModifiers::NONE,
        });
        let release = Event::Key(KeyEvent {
            code: KeyCode::Right,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        });
        let events = vec![
            resize.clone(),
            release.clone(),
            release.clone(),
            release.clone(),
            release.clone(),
            moved.clone(),
        ];

        let result = coalesce_movement_events(events.clone(), true);

        assert_eq!(result.events, events);
        assert_eq!(result.dropped, 0);
    }

    #[test]
    fn test_typed_search_text_is_never_coalesced() {
        let events = vec![
            press(KeyCode::Char('a')),
            press(KeyCode::Char('a')),
            press(KeyCode::Char('a')),
        ];

        let canvas = coalesce_movement_events(events.clone(), true);
        let search_box = coalesce_movement_events(events.clone(), false);

        assert_eq!(canvas.events, events);
        assert_eq!(search_box.events, events);
    }

    #[test]
    fn test_movement_letters_typed_into_a_focused_search_box_are_kept() {
        let events = vec![press(KeyCode::Char('l')); 10];

        let result = coalesce_movement_events(events.clone(), false);

        assert_eq!(result.events, events);
        assert_eq!(result.dropped, 0);
    }

    #[test]
    fn test_focus_changing_key_ends_coalescing_for_the_rest_of_the_batch() {
        let mut events = vec![press(KeyCode::Down); 5];
        events.push(press(KeyCode::Tab));
        events.extend(vec![press(KeyCode::Down); 5]);

        let result = coalesce_movement_events(events, true);

        let mut expected = vec![press(KeyCode::Down); MAX_MOVEMENT_REPEATS];
        expected.push(press(KeyCode::Tab));
        expected.extend(vec![press(KeyCode::Down); 5]);
        assert_eq!(result.events, expected);
        assert_eq!(result.dropped, 5 - MAX_MOVEMENT_REPEATS);
    }
}
