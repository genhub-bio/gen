use std::backtrace::Backtrace;

use crossterm::{
    execute,
    terminal::{LeaveAlternateScreen, disable_raw_mode},
};
use ratatui::{
    style::Style,
    text::{Line, Span},
};

/// Parses a string with markdown-like asterisk syntax for highlighting.
/// Segments surrounded by '*' are styled with `highlight_style`.
/// Other segments are styled with `default_style`.
pub fn style_text(text: &str, default_style: Style, highlight_style: Style) -> Line<'_> {
    let mut spans = Vec::new();
    let mut is_highlighted = false;
    for part in text.split('*') {
        if !part.is_empty() {
            spans.push(Span::styled(
                part,
                if is_highlighted {
                    highlight_style
                } else {
                    default_style
                },
            ));
        }
        is_highlighted = !is_highlighted;
    }
    Line::from(spans)
}

/// Restore the terminal to its normal state after a full-screen TUI session.
///
/// Disables raw mode and leaves the alternate screen. Errors are silently
/// ignored because this is typically called from panic handlers where there
/// is nothing useful to do with them.
pub fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
}

/// Install a panic hook that restores the terminal before printing the crash
/// report. Without this, a panic inside a full-screen TUI leaves the terminal
/// in raw/alternate-screen mode and the error message is invisible.
pub fn install_tui_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        restore_terminal();
        eprintln!("Application crashed: {info}");
        let backtrace = Backtrace::capture();
        eprintln!("Stack trace:\n{backtrace}");
    }));
}
