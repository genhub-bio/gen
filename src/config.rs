use crate::migrations::run_operation_migrations;
use crate::models::operations::Operation;
use itertools::iproduct;
use ratatui::style::Color;
use ratatui_base16::Base16Palette;
use rusqlite::Connection;
use std::string::ToString;
use std::sync::RwLock;
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::LazyLock,
};

thread_local! {
pub static BASE_DIR: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(env::current_dir().unwrap()));
}

pub fn get_operation_connection(db_path: impl Into<Option<PathBuf>>) -> Connection {
    let db_path = db_path.into();
    let path = if let Some(s) = db_path {
        s
    } else {
        get_gen_db_path()
    };
    let mut conn =
        Connection::open(&path).unwrap_or_else(|_| panic!("Error connecting to {:?}", &path));
    rusqlite::vtab::array::load_module(&conn).unwrap();
    run_operation_migrations(&mut conn);
    conn
}

fn ensure_dir(path: &PathBuf) {
    if !path.is_dir() {
        fs::create_dir_all(path).unwrap();
    }
}

/// Looks for the .gen directory in the current directory, or in a temporary directory if setup_gen_dir()
/// was called first.  If it doesn't exist, it will be created.
/// Returns the path to the .gen directory.
pub fn get_or_create_gen_dir() -> PathBuf {
    let start_dir = BASE_DIR.with(|v| v.read().unwrap().clone());
    let cur_dir = start_dir.as_path();
    let gen_path = cur_dir.join(".gen");
    ensure_dir(&gen_path);
    gen_path
}

// TODO: maybe just store all these things in a sqlite file too in .gen
/// Searches for the .gen directory in the current directory and all parent directories,
/// or in a temporary directory if setup_gen_dir() was called first.
/// Returns the path to the .gen directory if found, otherwise returns None.
pub fn get_gen_dir() -> Option<String> {
    let start_dir = BASE_DIR.with(|v| v.read().unwrap().clone());
    let mut cur_dir = start_dir.as_path();
    let mut gen_path = cur_dir.join(".gen");
    while !gen_path.is_dir() {
        match cur_dir.parent() {
            Some(v) => {
                cur_dir = v;
            }
            None => {
                // TODO: make gen init
                return None;
            }
        };
        gen_path = cur_dir.join(".gen");
    }
    Some(gen_path.to_str().unwrap().to_string())
}

/// Returns the path to the gen.db file in the .gen directory.
/// If the .gen directory is not found, it will panic.
pub fn get_gen_db_path() -> PathBuf {
    match get_gen_dir() {
        Some(dir) => Path::new(&dir).join("gen.db"),
        None => {
            panic!("No .gen directory found. Please run 'gen init' first.")
        }
    }
}

pub fn get_changeset_path(operation: &Operation) -> PathBuf {
    let gen_dir = get_gen_dir()
        .unwrap_or_else(|| panic!("No .gen directory found. Please run 'gen init' first."));
    let path = Path::new(&gen_dir)
        .join(operation.db_uuid.clone())
        .join("changeset");
    ensure_dir(&path);
    path
}

// Theme support (TODO: make this configurable)
const THEME_PATH: &str = "config/mocha.yaml"; // dark
                                              // const THEME_PATH: &str = "config/latte.yaml"; // light

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

//

thread_local! {
pub static PALETTE: LazyLock<RwLock<Base16Palette>> =
        LazyLock::new(|| {
            let palette_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(THEME_PATH);
            let mut palette = Base16Palette::from_yaml(palette_path).expect("Failed to load theme");
            //
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
}

pub fn col(color_name: &str) -> Result<Color, String> {
    PALETTE.with(|palette_lock| {
        let palette = palette_lock
            .read()
            .map_err(|e| format!("Failed to read palette: {}", e))?;
        match color_name {
            "canvas" | "panel" | "statusbar" | "base00" => Ok(palette.base00), // Main background color
            "sidebar" | "base01" => Ok(palette.base01), // Secondary background
            "separator" | "base02" => Ok(palette.base02), // Muted background
            "edge" | "node" | "highlight_muted" | "base03" => Ok(palette.base03), // Edge lines and muted highlights
            "text_muted" | "base04" => Ok(palette.base04), // Secondary text, icons
            "text" | "base05" => Ok(palette.base05),       // Primary text color
            "text_bright" | "base06" => Ok(palette.base06), // Bright text (rarely used)
            "highlight" | "base07" => Ok(palette.base07),  // Selection highlight
            "error" | "base08" => Ok(palette.base08),      // Errors, path highlighting
            "warning" | "base09" => Ok(palette.base09),    // Warnings
            "success" | "base0a" | "base0A" => Ok(palette.base0a), // Success indicators
            "base0b" | "base0B" => Ok(palette.base0b),     // Accent color 1
            "base0c" | "base0C" => Ok(palette.base0c),     // Accent color 2
            "base0d" | "base0D" => Ok(palette.base0d),     // Accent color 3
            "base0e" | "base0E" => Ok(palette.base0e),     // Accent color 4
            "base0f" | "base0F" => Ok(palette.base0f),     // Accent color 5
            _ => Err(format!("Color '{}' not found in palette", color_name)),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::setup_gen_dir;

    #[test]
    fn test_finds_gen_dir() {
        setup_gen_dir();
        assert!(get_gen_dir().is_some());
    }

    #[test]
    fn test_get_theme_color() {
        let color = col("canvas");
        assert!(color.is_ok());

        let color = col("invalid_color");
        assert!(color.is_err());
    }
}
