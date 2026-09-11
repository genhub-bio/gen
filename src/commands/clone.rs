use std::{fs, io, path::Path};

use gen_core::config::Workspace;
use gen_models::operations::{Defaults, Remote, RemoteBranch};

use crate::{
    commands::remote::operations::{
        canonical_remote_url, clone_destination_path, clone_into_workspace,
    },
    get_config_connection,
};

pub fn execute(url: &str, parent: &Workspace) -> Result<(), Box<dyn std::error::Error>> {
    let destination = clone_destination_path(parent, url)?;
    clone_to_workspace(url, &Workspace::new(&destination))?;
    println!(
        "Cloned {} into {}.",
        canonical_remote_url(url)?,
        destination.display()
    );
    Ok(())
}

/// Clones a remote repository into a new or empty destination workspace.
pub fn clone_to_workspace(
    url: &str,
    workspace: &Workspace,
) -> Result<(), Box<dyn std::error::Error>> {
    let destination = workspace.base_dir();
    let destination_exists = destination.exists();
    if destination_exists && (!destination.is_dir() || fs::read_dir(destination)?.next().is_some())
    {
        return Err(format!(
            "Clone destination already exists and is not an empty directory: {}",
            destination.display()
        )
        .into());
    }
    if !destination_exists {
        fs::create_dir(destination)?;
    }
    let result = (|| {
        let workspace = Workspace::new(destination);
        workspace.ensure_gen_dir();
        let config = get_config_connection(Some(workspace.gen_db_path()?))?;
        let canonical_url = canonical_remote_url(url)?;
        let remote = Remote::create(&config, "origin", &canonical_url)?;
        Defaults::set_default_remote(&config, Some("origin"))?;
        let branch = clone_into_workspace(&config, &remote, &workspace)?;
        RemoteBranch::set_remote_validated(&config, &branch, Some("origin"))?;
        Defaults::set_current_branch(&config, Some(&branch))?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })();
    if result.is_err()
        && let Err(error) = remove_incomplete_clone(destination, destination_exists)
    {
        eprintln!(
            "Warning: failed to remove incomplete clone at {}: {error}",
            destination.display()
        );
    }
    result
}

fn remove_incomplete_clone(destination: &Path, preserve_directory: bool) -> io::Result<()> {
    if !preserve_directory {
        return fs::remove_dir_all(destination);
    }
    // The clone only removes what it created. A directory supplied by the caller may be a mount
    // point, a symlink target, another process's working directory, or carry permissions the clone
    // cannot reconstruct, so delete its contents and leave the directory itself in place.
    for entry in fs::read_dir(destination)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            fs::remove_dir_all(entry.path())?;
        } else {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}
