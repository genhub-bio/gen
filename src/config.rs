use crate::migrations::run_operation_migrations;
use crate::models::operations::Operation;
use rusqlite::Connection;
use std::string::ToString;
use std::sync::RwLock;
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::LazyLock,
};
use ratatui_base16::Base16Palette;
use ratatui::style::Color;

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

// Color palette:
thread_local! {
    pub static PALETTE: LazyLock<RwLock<Base16Palette>> = 
            LazyLock::new(|| {
                let palette_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("config/mocha.yaml");
                RwLock::new(Base16Palette::from_yaml(palette_path).expect("Failed to load theme"))
            });
    }

pub fn col(color_name: &str) -> Result<Color, String> {
    PALETTE.with(|palette_lock| {
        let palette = palette_lock.read().map_err(|e| format!("Failed to read palette: {}", e))?;
        match color_name {
            "base00" => Ok(palette.base00),
            "base01" => Ok(palette.base01),
            "base02" => Ok(palette.base02),
            "base03" => Ok(palette.base03),
            "base04" => Ok(palette.base04),
            "base05" => Ok(palette.base05),
            "base06" => Ok(palette.base06),
            "base07" => Ok(palette.base07),
            "base08" => Ok(palette.base08),
            "base09" => Ok(palette.base09),
            "base0a" => Ok(palette.base0a),
            "base0b" => Ok(palette.base0b),
            "base0c" => Ok(palette.base0c),
            "base0d" => Ok(palette.base0d),
            "base0e" => Ok(palette.base0e),
            "base0f" => Ok(palette.base0f),
            "base0A" => Ok(palette.base0a),
            "base0B" => Ok(palette.base0b),
            "base0C" => Ok(palette.base0c),
            "base0D" => Ok(palette.base0d),
            "base0E" => Ok(palette.base0e),
            "base0F" => Ok(palette.base0f),
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
        let color = col("base00");
        assert!(color.is_ok());
        
        let color = col("invalid_color");
        assert!(color.is_err());
    }
}
