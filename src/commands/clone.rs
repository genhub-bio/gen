use std::fs;

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

/// Clones a remote repository into an explicit destination workspace.
pub fn clone_to_workspace(
    url: &str,
    workspace: &Workspace,
) -> Result<(), Box<dyn std::error::Error>> {
    let destination = workspace.base_dir();
    if destination.exists() {
        return Err(format!(
            "Clone destination already exists: {}",
            destination.display()
        )
        .into());
    }
    fs::create_dir(destination)?;
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
        && let Err(error) = fs::remove_dir_all(destination)
    {
        eprintln!(
            "Warning: failed to remove incomplete clone at {}: {error}",
            destination.display()
        );
    }
    result
}
