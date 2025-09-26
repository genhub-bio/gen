use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{LazyLock, RwLock},
};

use crate::HashId;

thread_local! {
pub static BASE_DIR: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(env::current_dir().unwrap()));
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

pub fn get_changeset_path(hash: &HashId) -> PathBuf {
    let gen_dir = get_gen_dir()
        .unwrap_or_else(|| panic!("No .gen directory found. Please run 'gen init' first."));
    let path = Path::new(&gen_dir)
        .join("changeset")
        .join(format!("{hash}"));
    ensure_dir(&path);
    path
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_finds_gen_dir() {
        let tmp_dir = tempdir().unwrap().keep();
        {
            BASE_DIR.with(|v| {
                let mut writer = v.write().unwrap();
                *writer = tmp_dir;
            });
        }
        get_or_create_gen_dir();
        assert!(get_gen_dir().is_some());
    }
}
