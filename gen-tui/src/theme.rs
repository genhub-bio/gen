use std::{ops::Index, sync::LazyLock, sync::RwLock};

use ratatui::style::Color;

/// Compact Base16-style palette of 16 colors addressed by slot index.
///
/// Slot assignments:
///   0x00 – canvas bg, edge bg, node fg, cursor fg
///   0x05 – node bg, edge fg
///   0x07 – cursor bg
///   0x08–0x0E – highlight / accent colors
#[derive(Debug, Clone, Copy)]
pub struct Theme(pub [Color; 16]);

impl Index<usize> for Theme {
    type Output = Color;

    fn index(&self, i: usize) -> &Self::Output {
        debug_assert!(i < 16, "theme index {i} out of range");
        &self.0[i]
    }
}

impl Default for Theme {
    /// Catppuccin Mocha defaults.
    fn default() -> Self {
        use Color::Rgb;
        Theme([
            Rgb(0x1e, 0x1e, 0x2e), // base00 – darkest bg
            Rgb(0x18, 0x18, 0x25), // base01
            Rgb(0x31, 0x32, 0x44), // base02
            Rgb(0x45, 0x47, 0x5a), // base03
            Rgb(0x58, 0x5b, 0x70), // base04
            Rgb(0xcd, 0xd6, 0xf4), // base05 – main text
            Rgb(0xf5, 0xe0, 0xdc), // base06
            Rgb(0xb4, 0xbe, 0xfe), // base07 – periwinkle
            Rgb(0xf3, 0x8b, 0xa8), // base08 – red
            Rgb(0xfa, 0xb3, 0x87), // base09 – peach
            Rgb(0xf9, 0xe2, 0xaf), // base0a – yellow
            Rgb(0xa6, 0xe3, 0xa1), // base0b – green
            Rgb(0x94, 0xe2, 0xd5), // base0c – teal
            Rgb(0x89, 0xb4, 0xfa), // base0d – blue
            Rgb(0xcb, 0xa6, 0xf7), // base0e – mauve
            Rgb(0xf2, 0xcd, 0xcd), // base0f – flamingo
        ])
    }
}

static THEME: LazyLock<RwLock<Theme>> = LazyLock::new(|| RwLock::new(Theme::default()));

/// Returns a snapshot copy of the current global theme.
pub fn current_theme() -> Theme {
    *THEME.read().expect("theme lock poisoned")
}

/// Replace the global theme.
pub fn set_theme(theme: Theme) {
    *THEME.write().expect("theme lock poisoned") = theme;
}
