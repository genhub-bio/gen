#[cfg(not(target_os = "emscripten"))]
use std::io;

#[cfg(not(target_os = "emscripten"))]
use crossterm::{
    cursor::Show,
    event::DisableMouseCapture,
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use gen_tui::key_event::Event;
#[cfg(not(target_os = "emscripten"))]
use ratatui::{Terminal, backend::CrosstermBackend};

#[cfg(not(target_os = "emscripten"))]
pub type CrosstermTerminal = Terminal<CrosstermBackend<std::io::Stdout>>;

/// Restore terminal state to normal mode.
#[cfg(not(target_os = "emscripten"))]
pub fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(
        std::io::stdout(),
        DisableMouseCapture,
        LeaveAlternateScreen,
        Show
    )?;
    Ok(())
}

/// Install a global panic hook that always restores terminal state before
/// printing the crash message.
#[cfg(not(target_os = "emscripten"))]
pub fn install_global_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let _ = restore_terminal();
        eprintln!("❗ The application has encountered an unexpected error and must exit.");
        eprintln!("Message: {}", info);
        eprintln!();
        eprintln!("👉 Please file an issue at: https://github.com/genhub-bio/gen/issues");
        eprintln!("   Include the full output above, what you were doing, and system info.");
    }));
}

/// Install a global panic hook. This target has no alternate-screen/raw-mode
/// terminal state to restore, so this just prints the crash message.
#[cfg(target_os = "emscripten")]
pub fn install_global_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("❗ The application has encountered an unexpected error and must exit.");
        eprintln!("Message: {}", info);
        eprintln!();
        eprintln!("👉 Please file an issue at: https://github.com/genhub-bio/gen/issues");
        eprintln!("   Include the full output above, what you were doing, and system info.");
    }));
}

/// RAII guard for full-screen TUI sessions.
///
/// Enters alternate screen + raw mode on creation and restores terminal state
/// on drop, even when returning early from the view function.
#[cfg(not(target_os = "emscripten"))]
pub struct TuiSession {
    terminal: CrosstermTerminal,
    restored: bool,
}

#[cfg(not(target_os = "emscripten"))]
impl TuiSession {
    pub fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(err);
        }

        match Terminal::new(CrosstermBackend::new(stdout)) {
            Ok(terminal) => Ok(Self {
                terminal,
                restored: false,
            }),
            Err(err) => {
                let _ = restore_terminal();
                Err(err)
            }
        }
    }

    pub fn terminal_mut(&mut self) -> &mut CrosstermTerminal {
        &mut self.terminal
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }

        self.restored = true;
        self.terminal.show_cursor().ok();
        disable_raw_mode()?;
        execute!(
            self.terminal.backend_mut(),
            DisableMouseCapture,
            LeaveAlternateScreen,
            Show
        )?;
        Ok(())
    }
}

#[cfg(not(target_os = "emscripten"))]
impl Drop for TuiSession {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(target_os = "emscripten")]
pub type EmscriptenTerminal =
    ratatui::Terminal<crate::views::emscripten_backend::EmscriptenBackend<std::io::Stdout>>;

/// RAII guard for full-screen TUI sessions on `target_os = "emscripten"`. Mirrors the native
/// `TuiSession` above, but builds the terminal on `EmscriptenBackend` instead of
/// `ratatui-crossterm`'s `CrosstermBackend` (see `emscripten_backend.rs` for why that crate is
/// unusable on this target) and drives mouse capture through `emscripten_input`'s hand-written
/// SGR escape sequences instead of `crossterm::event::{Enable,Disable}MouseCapture`.
#[cfg(target_os = "emscripten")]
pub struct TuiSession {
    terminal: EmscriptenTerminal,
    restored: bool,
}

#[cfg(target_os = "emscripten")]
impl TuiSession {
    pub fn enter() -> std::io::Result<Self> {
        crossterm::terminal::enable_raw_mode()?;
        if let Err(err) =
            crossterm::execute!(std::io::stdout(), crossterm::terminal::EnterAlternateScreen)
        {
            let _ = crossterm::terminal::disable_raw_mode();
            return Err(err);
        }

        match ratatui::Terminal::new(crate::views::emscripten_backend::EmscriptenBackend::new(
            std::io::stdout(),
        )) {
            Ok(terminal) => Ok(Self {
                terminal,
                restored: false,
            }),
            Err(err) => {
                let _ = Self::restore_terminal_state();
                Err(err)
            }
        }
    }

    pub fn terminal_mut(&mut self) -> &mut EmscriptenTerminal {
        &mut self.terminal
    }

    fn restore_terminal_state() -> std::io::Result<()> {
        let _ = crate::views::emscripten_input::disable_mouse_capture();
        crossterm::terminal::disable_raw_mode()?;
        crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::cursor::Show
        )
    }

    pub fn restore(&mut self) -> std::io::Result<()> {
        if self.restored {
            return Ok(());
        }

        self.restored = true;
        self.terminal.show_cursor().ok();
        Self::restore_terminal_state()
    }
}

#[cfg(target_os = "emscripten")]
impl Drop for TuiSession {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Waits up to `timeout` for an input event to become ready, without consuming it. Used to
/// sleep between animation ticks; the next loop iteration drains whatever arrived via
/// [`poll_immediate_event`].
#[cfg(not(target_os = "emscripten"))]
pub fn wait_for_event(timeout: std::time::Duration) {
    let _ = crossterm::event::poll(timeout);
}

#[cfg(target_os = "emscripten")]
pub fn wait_for_event(timeout: std::time::Duration) {
    let _ = crate::views::emscripten_input::wait_ready(timeout);
}

/// Returns the next already-buffered input event, if any, without blocking.
#[cfg(not(target_os = "emscripten"))]
pub fn poll_immediate_event() -> Option<Event> {
    if crossterm::event::poll(std::time::Duration::from_millis(0)).unwrap_or(false) {
        crossterm::event::read().ok()
    } else {
        None
    }
}

#[cfg(target_os = "emscripten")]
pub fn poll_immediate_event() -> Option<Event> {
    crate::views::emscripten_input::poll_next(std::time::Duration::from_millis(0))
}
