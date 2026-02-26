use std::io;

use crossterm::{
    cursor::Show,
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

pub type CrosstermTerminal = Terminal<CrosstermBackend<std::io::Stdout>>;

/// Restore terminal state to normal mode.
pub fn restore_terminal() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(std::io::stdout(), LeaveAlternateScreen, Show)?;
    Ok(())
}

/// Install a global panic hook that always restores terminal state before
/// printing the crash message.
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

/// RAII guard for full-screen TUI sessions.
///
/// Enters alternate screen + raw mode on creation and restores terminal state
/// on drop, even when returning early from the view function.
pub struct TuiSession {
    terminal: CrosstermTerminal,
    restored: bool,
}

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
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen, Show)?;
        Ok(())
    }
}

impl Drop for TuiSession {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}
