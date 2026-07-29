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
use ratatui::{TerminalOptions, Viewport, layout::Rect};

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

    pub fn enable_mouse_capture(&mut self) -> io::Result<()> {
        execute!(
            self.terminal.backend_mut(),
            crossterm::event::EnableMouseCapture
        )
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

    pub fn enable_mouse_capture(&mut self) -> std::io::Result<()> {
        crate::views::emscripten_input::enable_mouse_capture()
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

/// Restores terminal state after an inline (non-alternate-screen) TUI session: positions the
/// cursor just past the bottom of the rendered viewport before disabling raw mode, so the shell
/// prompt reappears below the last frame instead of wherever ratatui's generic `restore()` (which
/// assumes a full-screen alternate-screen session) would leave it. Shared by both `InlineTuiSession`
/// variants below since it only touches `crossterm::cursor`/`crossterm::terminal`, neither of which
/// need `mio` and so work identically on native and `target_os = "emscripten"`.
fn restore_inline_terminal(viewport_area: Rect) -> std::io::Result<()> {
    let target_line = viewport_area.y + viewport_area.height;
    crossterm::execute!(std::io::stdout(), crossterm::cursor::MoveTo(0, target_line))?;
    crossterm::execute!(std::io::stdout(), crossterm::cursor::Show)?;
    crossterm::terminal::disable_raw_mode()?;
    std::io::Write::flush(&mut std::io::stdout())
}

/// RAII guard for inline (non-alternate-screen) TUI sessions, e.g. `gen view`'s inline widget.
/// Mirrors `TuiSession`, but uses `Viewport::Inline` and restores via
/// [`restore_inline_terminal`] instead of leaving an alternate screen.
#[cfg(not(target_os = "emscripten"))]
pub struct InlineTuiSession {
    terminal: CrosstermTerminal,
    restored: bool,
}

#[cfg(not(target_os = "emscripten"))]
impl InlineTuiSession {
    pub fn enter(height: u16) -> io::Result<Self> {
        let terminal = ratatui::try_init_with_options(TerminalOptions {
            viewport: Viewport::Inline(height),
        })?;
        Ok(Self {
            terminal,
            restored: false,
        })
    }

    pub fn terminal_mut(&mut self) -> &mut CrosstermTerminal {
        &mut self.terminal
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }

        self.restored = true;
        restore_inline_terminal(self.terminal.get_frame().area())
    }
}

#[cfg(not(target_os = "emscripten"))]
impl Drop for InlineTuiSession {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// RAII guard for inline TUI sessions on `target_os = "emscripten"`. `ratatui::try_init_with_options`
/// requires ratatui's `crossterm` cargo feature, which pulls in `mio` via `ratatui-crossterm`'s
/// default-featured `crossterm` dependency (no emscripten backend), so the terminal is built
/// manually on `EmscriptenBackend` instead, same as `TuiSession` does for the full-screen case.
#[cfg(target_os = "emscripten")]
pub struct InlineTuiSession {
    terminal: EmscriptenTerminal,
    restored: bool,
}

#[cfg(target_os = "emscripten")]
impl InlineTuiSession {
    pub fn enter(height: u16) -> std::io::Result<Self> {
        crossterm::terminal::enable_raw_mode()?;
        let terminal = ratatui::Terminal::with_options(
            crate::views::emscripten_backend::EmscriptenBackend::new(std::io::stdout()),
            TerminalOptions {
                viewport: Viewport::Inline(height),
            },
        )?;
        Ok(Self {
            terminal,
            restored: false,
        })
    }

    pub fn terminal_mut(&mut self) -> &mut EmscriptenTerminal {
        &mut self.terminal
    }

    pub fn restore(&mut self) -> std::io::Result<()> {
        if self.restored {
            return Ok(());
        }

        self.restored = true;
        restore_inline_terminal(self.terminal.get_frame().area())
    }
}

#[cfg(target_os = "emscripten")]
impl Drop for InlineTuiSession {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Waits up to `timeout` for an input event to become ready, without consuming it. Used to
/// sleep between animation ticks; the next loop iteration drains whatever arrived via
/// [`poll_immediate_event`].
#[cfg(not(target_os = "emscripten"))]
pub fn wait_for_event(timeout: std::time::Duration) -> std::io::Result<()> {
    crossterm::event::poll(timeout).map(|_| ())
}

#[cfg(target_os = "emscripten")]
pub fn wait_for_event(timeout: std::time::Duration) -> std::io::Result<()> {
    let _ = crate::views::emscripten_input::wait_ready(timeout);
    Ok(())
}

/// Returns the next already-buffered input event, if any, without blocking.
#[cfg(not(target_os = "emscripten"))]
pub fn poll_immediate_event() -> std::io::Result<Option<Event>> {
    if crossterm::event::poll(std::time::Duration::from_millis(0))? {
        crossterm::event::read().map(Some)
    } else {
        Ok(None)
    }
}

#[cfg(target_os = "emscripten")]
pub fn poll_immediate_event() -> std::io::Result<Option<Event>> {
    Ok(crate::views::emscripten_input::poll_next(
        std::time::Duration::from_millis(0),
    ))
}
