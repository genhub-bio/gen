//! Keyboard/mouse input for `gen view` under `target_os = "emscripten"`.
//!
//! Crossterm's `event` module (and the `mio`-based polling it's built on) is unavailable on this
//! target, so this reimplements just enough of it: a bounded-wait readiness check plus a small
//! ANSI/SGR parser, producing the same `gen_tui::key_event` types crossterm would.
//!
//! The bounded wait uses `libc::select()`, not `libc::poll()`: emscripten's own `poll(2)` syscall
//! shim (`__syscall_poll` in Emscripten's `libsyscall.js`) hardcodes an infinite timeout
//! regardless of what's requested, while `select(2)`'s shim correctly threads the real timeout
//! through to cockle's `Atomics.wait`-based stdin poll. Verified empirically: idle `select()`
//! calls correctly report not-ready, and the next call after a keypress correctly reports ready.

#[cfg(target_os = "emscripten")]
use std::{
    cell::RefCell,
    collections::VecDeque,
    io::{Read, Write},
    os::fd::RawFd,
    time::Duration,
};

use gen_tui::key_event::{
    Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};

#[cfg(target_os = "emscripten")]
const STDIN_FD: RawFd = 0;

/// Returns true if stdin has bytes available to read within `timeout`.
#[cfg(target_os = "emscripten")]
pub(crate) fn stdin_ready(timeout: Duration) -> bool {
    let mut readfds: libc::fd_set = unsafe { std::mem::zeroed() };
    unsafe {
        libc::FD_ZERO(&mut readfds);
        libc::FD_SET(STDIN_FD, &mut readfds);
    }
    let mut tv = libc::timeval {
        tv_sec: timeout.as_secs() as libc::time_t,
        tv_usec: timeout.subsec_micros() as libc::suseconds_t,
    };
    let ret = unsafe {
        libc::select(
            STDIN_FD + 1,
            &mut readfds,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut tv,
        )
    };
    ret > 0 && unsafe { libc::FD_ISSET(STDIN_FD, &readfds) }
}

/// Reads whatever bytes are currently available. Only called once `stdin_ready` has reported
/// readiness, so this should not block for long: a full escape sequence arrives as one buffered
/// burst (confirmed empirically), so one read gets the whole sequence, not a partial one.
#[cfg(target_os = "emscripten")]
pub(crate) fn read_available() -> Vec<u8> {
    let mut buf = [0u8; 64];
    match std::io::stdin().read(&mut buf) {
        Ok(n) => buf[..n].to_vec(),
        Err(_) => Vec::new(),
    }
}

/// Queries the terminal for the cursor's current position via the DSR (`ESC [ 6 n`) escape
/// sequence, blocking until the terminal's `ESC [ row ; col R` reply arrives. Mirrors
/// `crossterm::cursor::position`, which is unavailable here since it's implemented on top of the
/// `mio`-gated `event` module's internal poll/read.
///
/// Only ever called once, before `poll_next`'s event loop starts, so there is no risk of
/// consuming bytes that belong to a real keystroke.
#[cfg(target_os = "emscripten")]
pub fn read_cursor_position() -> std::io::Result<(u16, u16)> {
    std::io::stdout().write_all(b"\x1b[6n")?;
    std::io::stdout().flush()?;

    let mut buffer = Vec::new();
    loop {
        if let Some(pos) = parse_cursor_position_report(&buffer) {
            return Ok(pos);
        }
        if !stdin_ready(Duration::from_secs(2)) {
            return Err(std::io::Error::other(
                "timed out waiting for cursor position report",
            ));
        }
        buffer.extend(read_available());
    }
}

/// Parses a `ESC [ row ; col R` cursor position report, returning `(col - 1, row - 1)` to match
/// crossterm's 0-indexed `(x, y)` convention.
#[cfg(target_os = "emscripten")]
fn parse_cursor_position_report(bytes: &[u8]) -> Option<(u16, u16)> {
    let start = bytes.windows(2).position(|pair| pair == b"\x1b[")?;
    let rest = &bytes[start + 2..];
    let terminator = rest.iter().position(|&b| b == b'R')?;
    let body = std::str::from_utf8(&rest[..terminator]).ok()?;
    let mut parts = body.split(';');
    let row = parts.next()?.parse::<u16>().ok()?;
    let col = parts.next()?.parse::<u16>().ok()?;
    Some((col.saturating_sub(1), row.saturating_sub(1)))
}

/// Parses a burst of raw terminal input bytes into zero or more events.
///
/// Only called by `poll_next` (`target_os = "emscripten"`) outside of tests, so a native,
/// non-test build sees no caller and would otherwise warn this whole parse chain as dead code.
#[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
fn parse_bytes(bytes: &[u8]) -> Vec<Event> {
    let mut events = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte == 0x1b {
            let (event, consumed) = parse_escape(&bytes[i..]);
            if let Some(event) = event {
                events.push(event);
            }
            i += consumed;
            continue;
        }

        if let Some(event) = parse_single_byte(byte) {
            events.push(event);
        }
        i += 1;
    }
    events
}

#[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
fn parse_single_byte(byte: u8) -> Option<Event> {
    let code = match byte {
        b'\r' => KeyCode::Enter,
        0x7f => KeyCode::Backspace,
        b'\t' => KeyCode::Tab,
        0x20..=0x7e => KeyCode::Char(byte as char),
        _ => return None,
    };
    Some(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)))
}

/// Parses an escape sequence starting at `bytes[0] == 0x1b`. Returns the event (if any) and how
/// many bytes were consumed.
#[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
fn parse_escape(bytes: &[u8]) -> (Option<Event>, usize) {
    if bytes.len() < 2 {
        // Lone ESC with nothing following in this burst.
        return (
            Some(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))),
            1,
        );
    }

    if bytes[1] != b'[' {
        // Not a CSI sequence (e.g. Alt+key sends ESC followed by the key byte). Treat the ESC
        // alone as Esc and let the next byte be parsed as its own event on the next loop turn.
        return (
            Some(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))),
            1,
        );
    }

    if bytes.len() >= 3 && bytes[2] == b'<' {
        return parse_sgr_mouse(bytes);
    }

    if bytes.len() >= 3 {
        let code = match bytes[2] {
            b'A' => Some(KeyCode::Up),
            b'B' => Some(KeyCode::Down),
            b'C' => Some(KeyCode::Right),
            b'D' => Some(KeyCode::Left),
            b'Z' => Some(KeyCode::BackTab),
            _ => None,
        };
        if let Some(code) = code {
            return (Some(Event::Key(KeyEvent::new(code, KeyModifiers::NONE))), 3);
        }
    }

    // Unrecognized CSI sequence: consume just the ESC and re-parse the rest, rather than
    // dropping potentially-meaningful bytes.
    (
        Some(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))),
        1,
    )
}

/// Parses an SGR mouse sequence `ESC [ < Cb ; Cx ; Cy M` (press/drag) or `...m` (release), per
/// xterm.js's `CoreMouseService` SGR encoding: low 2 bits of Cb select the button, bit 5 (32) set
/// means this is a drag/motion report.
#[cfg_attr(not(target_os = "emscripten"), allow(dead_code))]
fn parse_sgr_mouse(bytes: &[u8]) -> (Option<Event>, usize) {
    // bytes[0..=2] == ESC [ <
    let rest = &bytes[3..];
    let Some(terminator_pos) = rest.iter().position(|b| *b == b'M' || *b == b'm') else {
        // Incomplete sequence in this burst; drop the whole thing rather than misparse.
        return (None, bytes.len());
    };
    let is_release = rest[terminator_pos] == b'm';
    let params_str = match std::str::from_utf8(&rest[..terminator_pos]) {
        Ok(s) => s,
        Err(_) => return (None, 3 + terminator_pos + 1),
    };
    let mut parts = params_str.split(';');
    let (Some(cb), Some(cx), Some(cy)) = (
        parts.next().and_then(|p| p.parse::<u32>().ok()),
        parts.next().and_then(|p| p.parse::<u16>().ok()),
        parts.next().and_then(|p| p.parse::<u16>().ok()),
    ) else {
        return (None, 3 + terminator_pos + 1);
    };

    let button = match cb & 0x3 {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        _ => MouseButton::Right,
    };
    let is_drag = cb & 32 != 0;
    let kind = if is_release {
        MouseEventKind::Up(button)
    } else if is_drag {
        MouseEventKind::Drag(button)
    } else {
        MouseEventKind::Down(button)
    };

    let event = Event::Mouse(MouseEvent {
        kind,
        // SGR coordinates are 1-based.
        column: cx.saturating_sub(1),
        row: cy.saturating_sub(1),
        modifiers: KeyModifiers::NONE,
    });
    (Some(event), 3 + terminator_pos + 1)
}

#[cfg(target_os = "emscripten")]
thread_local! {
    static PENDING_EVENTS: RefCell<VecDeque<Event>> = RefCell::new(VecDeque::new());
}

/// Waits up to `timeout` for the next input event (keyboard or mouse), returning `None` on
/// timeout. Mirrors `crossterm::event::poll` + `crossterm::event::read` combined into one call.
#[cfg(target_os = "emscripten")]
pub fn poll_next(timeout: Duration) -> Option<Event> {
    if let Some(event) = PENDING_EVENTS.with(|events| events.borrow_mut().pop_front()) {
        return Some(event);
    }

    if !stdin_ready(timeout) {
        return None;
    }

    let bytes = read_available();
    let mut events: VecDeque<Event> = parse_bytes(&bytes).into();
    let first = events.pop_front();
    PENDING_EVENTS.with(|pending| pending.borrow_mut().extend(events));
    first
}

/// Returns true if an event is ready without consuming any stdin bytes: either an already-parsed
/// event is sitting in the pending queue from a previous burst, or fresh bytes are available.
/// Mirrors `crossterm::event::poll`'s non-destructive readiness check.
#[cfg(target_os = "emscripten")]
pub fn wait_ready(timeout: Duration) -> bool {
    if PENDING_EVENTS.with(|events| !events.borrow().is_empty()) {
        return true;
    }
    stdin_ready(timeout)
}

/// Enables SGR mouse tracking (modes 1000/1002/1006), matching what
/// `crossterm::event::EnableMouseCapture` would send — unavailable here since it's part of the
/// `mio`-gated `event` module. xterm.js implements these modes/encoding natively (confirmed by
/// reading its `CoreMouseService`/`InputHandler` source).
#[cfg(target_os = "emscripten")]
pub fn enable_mouse_capture() -> std::io::Result<()> {
    write!(std::io::stdout(), "\x1b[?1000h\x1b[?1002h\x1b[?1006h")
}

/// Disables SGR mouse tracking, matching `crossterm::event::DisableMouseCapture`.
#[cfg(target_os = "emscripten")]
pub fn disable_mouse_capture() -> std::io::Result<()> {
    write!(std::io::stdout(), "\x1b[?1006l\x1b[?1002l\x1b[?1000l")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_char() {
        assert_eq!(
            parse_bytes(b"a"),
            vec![Event::Key(KeyEvent::new(
                KeyCode::Char('a'),
                KeyModifiers::NONE
            ))]
        );
    }

    #[test]
    fn parses_enter_backspace_tab() {
        assert_eq!(
            parse_bytes(b"\r\x7f\t"),
            vec![
                Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
                Event::Key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE)),
                Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
            ]
        );
    }

    #[test]
    fn parses_arrow_keys() {
        assert_eq!(
            parse_bytes(b"\x1b[A\x1b[B\x1b[C\x1b[D"),
            vec![
                Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)),
                Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
                Event::Key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE)),
                Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)),
            ]
        );
    }

    #[test]
    fn parses_backtab() {
        assert_eq!(
            parse_bytes(b"\x1b[Z"),
            vec![Event::Key(KeyEvent::new(
                KeyCode::BackTab,
                KeyModifiers::NONE
            ))]
        );
    }

    #[test]
    fn parses_bare_escape() {
        assert_eq!(
            parse_bytes(b"\x1b"),
            vec![Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))]
        );
    }

    #[test]
    fn parses_sgr_mouse_down_and_up() {
        assert_eq!(
            parse_bytes(b"\x1b[<0;10;20M\x1b[<0;10;20m"),
            vec![
                Event::Mouse(MouseEvent {
                    kind: MouseEventKind::Down(MouseButton::Left),
                    column: 9,
                    row: 19,
                    modifiers: KeyModifiers::NONE,
                }),
                Event::Mouse(MouseEvent {
                    kind: MouseEventKind::Up(MouseButton::Left),
                    column: 9,
                    row: 19,
                    modifiers: KeyModifiers::NONE,
                }),
            ]
        );
    }

    #[test]
    fn parses_sgr_mouse_drag() {
        assert_eq!(
            parse_bytes(b"\x1b[<32;5;6M"),
            vec![Event::Mouse(MouseEvent {
                kind: MouseEventKind::Drag(MouseButton::Left),
                column: 4,
                row: 5,
                modifiers: KeyModifiers::NONE,
            })]
        );
    }
}
