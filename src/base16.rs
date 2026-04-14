use std::str::FromStr;

use gen_tui::theme::{Theme, set_theme};
use ratatui::style::Color;
use serde::Deserialize;

#[derive(Deserialize)]
struct RawPalette {
    base00: String,
    base01: String,
    base02: String,
    base03: String,
    base04: String,
    base05: String,
    base06: String,
    base07: String,
    base08: String,
    base09: String,
    base0a: String,
    base0b: String,
    base0c: String,
    base0d: String,
    base0e: String,
    base0f: String,
}

impl RawPalette {
    fn to_theme(&self) -> Theme {
        fn c(s: &str) -> Color {
            Color::from_str(s).unwrap_or_else(|_| {
                Color::from_str(&format!("#{s}")).expect("valid hex color in palette")
            })
        }
        Theme([
            c(&self.base00),
            c(&self.base01),
            c(&self.base02),
            c(&self.base03),
            c(&self.base04),
            c(&self.base05),
            c(&self.base06),
            c(&self.base07),
            c(&self.base08),
            c(&self.base09),
            c(&self.base0a),
            c(&self.base0b),
            c(&self.base0c),
            c(&self.base0d),
            c(&self.base0e),
            c(&self.base0f),
        ])
    }
}

/// Initialize the global theme from bundled JSON palettes.
///
/// Reads `GEN_THEME` env var: `"light"` → Catppuccin Latte, anything else → Catppuccin Mocha.
pub fn init_theme() {
    let json = match std::env::var("GEN_THEME").ok().as_deref() {
        Some("light") => include_str!("../themes/latte.json"),
        _ => include_str!("../themes/mocha.json"),
    };
    let raw: RawPalette = serde_json::from_str(json).expect("bundled palette JSON is valid");
    set_theme(raw.to_theme());
}
