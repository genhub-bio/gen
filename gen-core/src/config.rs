use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{LazyLock, RwLock},
};

use crate::{HashId, errors::ConfigError};

thread_local! {
pub static BASE_DIR: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(env::current_dir().unwrap()));
}

pub fn ensure_dir(path: &PathBuf) {
    if !path.is_dir() {
        fs::create_dir_all(path).unwrap();
    }
}

pub fn set_base_dir(d: &Path) {
    BASE_DIR
        .try_with(|v| {
            let mut w = v.write().unwrap();
            *w = d.to_path_buf()
        })
        .unwrap();
}

pub fn get_base_dir() -> PathBuf {
    BASE_DIR.with(|v| v.read().unwrap().clone())
}

/// Looks for the .gen directory in the current directory, or in a temporary directory if setup_gen_dir()
/// was called first.  If it doesn't exist, it will be created.
/// Returns the path to the .gen directory.
pub fn get_or_create_gen_dir() -> PathBuf {
    let start_dir = get_base_dir();
    let cur_dir = start_dir.as_path();
    let gen_path = cur_dir.join(".gen");
    ensure_dir(&gen_path);
    let asset_path = gen_path.join("assets");
    ensure_dir(&asset_path);
    gen_path
}

// TODO: maybe just store all these things in a sqlite file too in .gen
/// Searches for the .gen directory in the current directory and all parent directories,
/// or in a temporary directory if setup_gen_dir() was called first.
/// Returns the path to the .gen directory if found, otherwise returns None.
pub fn get_gen_dir() -> Option<String> {
    let start_dir = get_base_dir();
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

pub fn get_repo_root_path() -> Result<PathBuf, ConfigError> {
    let gen_dir = get_gen_dir().ok_or(ConfigError::GenDirectoryNotFound)?;
    PathBuf::from(gen_dir)
        .parent()
        .map(Path::to_path_buf)
        .ok_or(ConfigError::RepoRootNotFound)
}

pub fn get_gen_db_path() -> Result<PathBuf, ConfigError> {
    match get_gen_dir() {
        Some(dir) => Ok(Path::new(&dir).join("gen.db")),
        None => Err(ConfigError::GenDirectoryNotFound),
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
        set_base_dir(&tmp_dir);
        get_or_create_gen_dir();
        assert!(get_gen_dir().is_some());
    }

    #[test]
    fn test_set_base_dir() {
        let old_dir = get_base_dir();
        let tmp_dir = tempdir().unwrap().keep();
        set_base_dir(&tmp_dir);
        assert_eq!(tmp_dir, get_base_dir());
        assert_ne!(get_base_dir(), old_dir);
    }

    #[test]
    fn test_get_repo_root_path() {
        let gen_dir = setup_gen_environment();
        let expected_root = gen_dir.parent().unwrap().to_path_buf();
        assert_eq!(get_repo_root_path().unwrap(), expected_root);
    }

    #[test]
    fn test_get_repo_root_path_missing_gen_dir() {
        let tmp_dir = tempdir().unwrap().keep();
        set_base_dir(&tmp_dir);
        assert_eq!(Err(ConfigError::GenDirectoryNotFound), get_repo_root_path());
    }

    fn setup_gen_environment() -> PathBuf {
        let tmp_dir = tempdir().unwrap().keep();
        set_base_dir(&tmp_dir);
        get_or_create_gen_dir()
    }
}
