#[cfg(not(target_os = "emscripten"))]
use std::io;

#[cfg(not(target_os = "emscripten"))]
use crossterm::{
    cursor::Show,
    event::DisableMouseCapture,
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
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
