use std::{env, path::PathBuf, string::ToString, sync::RwLock};

use itertools::iproduct;
use once_cell::sync::Lazy;
use ratatui::style::Color;

use crate::base16::Base16Palette;

// Color theme support
fn get_theme_path() -> &'static str {
    match env::var("GEN_THEME").ok().as_deref() {
        Some("dark") => "config/mocha.yaml",
        Some("light") => "config/latte.yaml",
        _ => "config/mocha.yaml",
    }
}

/// Converts HTML color code (hex) to closest indexed color
pub fn html_to_ansi_color(html_code: &str) -> Color {
    // Index from 1 because the first character is the #
    let r = u8::from_str_radix(&html_code[1..3], 16).unwrap();
    let g = u8::from_str_radix(&html_code[3..5], 16).unwrap();
    let b = u8::from_str_radix(&html_code[5..7], 16).unwrap();
    let target = (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);

    // Create color palette (just the 6x6x6 cube, no grayscale, no first 16 colors)
    let steps = [0x00, 0x5f, 0x87, 0xaf, 0xd7, 0xff];
    let colorcube = iproduct!(&steps, &steps, &steps)
        .map(|(&r, &g, &b)| (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0))
        .collect::<Vec<_>>();

    // Find closest color
    let mut min_distance = f32::INFINITY;
    let mut closest_index = 0;

    for (i, color) in colorcube.iter().enumerate() {
        let distance = (target.0 - color.0).powi(2)
            + (target.1 - color.1).powi(2)
            + (target.2 - color.2).powi(2);

        if distance < min_distance {
            min_distance = distance;
            closest_index = i;
        }
    }

    // Add 16 to account for the standard ANSI colors (0-15)
    Color::Indexed(16 + closest_index as u8)
}

pub static PALETTE: Lazy<RwLock<Base16Palette>> = Lazy::new(|| {
    let palette_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(get_theme_path());
    let mut palette = Base16Palette::from_yaml(palette_path).expect("Failed to load theme");

    // If the terminal does not support truecolor, convert to ansi colors
    // (The mac os terminal degrades poorly when presented with truecolor)
    if !std::env::var("COLORTERM").is_ok_and(|v| v == "truecolor" || v == "24bit") {
        palette.base00 = html_to_ansi_color(&palette.base00.to_string());
        palette.base01 = html_to_ansi_color(&palette.base01.to_string());
        palette.base02 = html_to_ansi_color(&palette.base02.to_string());
        palette.base03 = html_to_ansi_color(&palette.base03.to_string());
        palette.base04 = html_to_ansi_color(&palette.base04.to_string());
        palette.base05 = html_to_ansi_color(&palette.base05.to_string());
        palette.base06 = html_to_ansi_color(&palette.base06.to_string());
        palette.base07 = html_to_ansi_color(&palette.base07.to_string());
        palette.base08 = html_to_ansi_color(&palette.base08.to_string());
        palette.base09 = html_to_ansi_color(&palette.base09.to_string());
        palette.base0a = html_to_ansi_color(&palette.base0a.to_string());
        palette.base0b = html_to_ansi_color(&palette.base0b.to_string());
        palette.base0c = html_to_ansi_color(&palette.base0c.to_string());
        palette.base0d = html_to_ansi_color(&palette.base0d.to_string());
        palette.base0e = html_to_ansi_color(&palette.base0e.to_string());
        palette.base0f = html_to_ansi_color(&palette.base0f.to_string());
    }
    RwLock::new(palette)
});

pub fn get_theme_color(color_name: &str) -> Result<Color, String> {
    let palette = PALETTE
        .read()
        .map_err(|e| format!("Failed to read palette: {e}"))?;
    match color_name {
        "sidebar" | "panel" | "statusbar" | "base00" => Ok(palette.base00), // Main background color
        "canvas" | "base01" => Ok(palette.base01), // Secondary (darker) background
        "separator" | "base02" => Ok(palette.base02), // Muted background
        "edge" | "node" | "highlight_muted" | "base03" => Ok(palette.base03), // Edge lines and muted highlights
        "text_muted" | "cursor_fg" | "base04" => Ok(palette.base04), // Secondary text, icons, cursor foreground
        "text" | "base05" => Ok(palette.base05),                     // Primary text color
        "text_bright" | "base06" => Ok(palette.base06),              // Bright text (rarely used)
        "highlight" | "cursor" | "cursor_bg" | "base07" => Ok(palette.base07), // Selection highlight, cursor background
        "error" | "base08" => Ok(palette.base08), // Errors, path highlighting
        "warning" | "base09" => Ok(palette.base09), // Warnings
        "success" | "base0a" | "base0A" => Ok(palette.base0a), // Success indicators
        "base0b" | "base0B" => Ok(palette.base0b), // Accent color 1
        "base0c" | "base0C" => Ok(palette.base0c), // Accent color 2
        "base0d" | "base0D" => Ok(palette.base0d), // Accent color 3
        "base0e" | "base0E" => Ok(palette.base0e), // Accent color 4
        "base0f" | "base0F" => Ok(palette.base0f), // Accent color 5
        _ => Err(format!("Color '{color_name}' not found in palette")),
    }
}

/// Returns a map of all theme colors for use in the main application UI
pub fn get_theme_map() -> std::collections::HashMap<String, Color> {
    let mut map = std::collections::HashMap::new();
    let keys = [
        "canvas",
        "panel",
        "statusbar",
        "sidebar",
        "separator",
        "edge",
        "node",
        "highlight_muted",
        "text_muted",
        "cursor_fg",
        "text",
        "text_bright",
        "highlight",
        "cursor",
        "cursor_bg",
        "error",
        "warning",
        "success",
        "base00",
        "base01",
        "base02",
        "base03",
        "base04",
        "base05",
        "base06",
        "base07",
        "base08",
        "base09",
        "base0a",
        "base0b",
        "base0c",
        "base0d",
        "base0e",
        "base0f",
    ];

    for key in keys {
        if let Ok(color) = get_theme_color(key) {
            map.insert(key.to_string(), color);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_theme_color() {
        let color = get_theme_color("canvas");
        assert!(color.is_ok());

        let color = get_theme_color("invalid_color");
        assert!(color.is_err());
    }
}
