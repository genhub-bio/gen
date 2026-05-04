use std::str::FromStr;

use gen_tui::theme::{Theme, set_theme};
use ratatui::style::Color;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
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
    #[serde(alias = "base0A")]
    base0a: String,
    #[serde(alias = "base0B")]
    base0b: String,
    #[serde(alias = "base0C")]
    base0c: String,
    #[serde(alias = "base0D")]
    base0d: String,
    #[serde(alias = "base0E")]
    base0e: String,
    #[serde(alias = "base0F")]
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

pub fn init_theme() {
    let json = match std::env::var("GEN_THEME").ok().as_deref() {
        Some("light") => include_str!("../../theme/latte.json"),
        _ => include_str!("../../theme/mocha.json"),
    };
    let raw: RawPalette = serde_json::from_str(json).expect("bundled palette JSON is valid");
    set_theme(raw.to_theme());
}

#[cfg(test)]
mod tests {
    use super::RawPalette;

    #[test]
    fn bundled_palettes_parse() {
        let latte = include_str!("../../theme/latte.json");
        let mocha = include_str!("../../theme/mocha.json");
        serde_json::from_str::<RawPalette>(latte).expect("latte.json is valid");
        serde_json::from_str::<RawPalette>(mocha).expect("mocha.json is valid");
    }
}
