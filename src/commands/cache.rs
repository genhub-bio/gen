use std::{fs, io};

use gen_core::Workspace;

/// Removes downloaded remote data without touching tracked assets.
///
/// `cache-clear` uses the non-creating cache lookup so running it against an empty workspace does
/// not leave a new `.gen/cache` behind.
pub fn clear(workspace: &Workspace) -> io::Result<bool> {
    let Some(cache_dir) = workspace.find_cache_dir() else {
        return Ok(false);
    };
    if !cache_dir.exists() {
        return Ok(false);
    }

    fs::remove_dir_all(cache_dir)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use gen_core::Workspace;
    use tempfile::tempdir;

    use super::clear;

    #[test]
    fn test_clear_removes_only_the_cache_directory() {
        let temp_dir = tempdir().expect("should create temporary workspace");
        let workspace = Workspace::new(temp_dir.path());
        workspace.ensure_gen_dir();
        let cache_dir = workspace
            .ensure_cache_dir()
            .expect("should create cache directory");
        fs::write(cache_dir.join("remote-index.tbi"), "cached").expect("should write cached index");
        let asset_path = workspace
            .asset_dir()
            .expect("should resolve asset directory")
            .join("asset.gff3");
        fs::write(&asset_path, "asset").expect("should write local asset");

        assert!(clear(&workspace).expect("should clear cache"));
        assert!(!cache_dir.exists());
        assert!(asset_path.exists());
        assert!(!clear(&workspace).expect("should tolerate an empty cache"));
    }
}
