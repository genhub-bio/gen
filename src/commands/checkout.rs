use gen_core::{BranchName, CommitRef};
use gen_models::{
    db::{ConfigConnection, GraphConnection},
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, branch_exists, checkout},
    },
    operations::Defaults,
};

use crate::history::ensure_clean_working_set;

pub fn execute(
    graph: &GraphConnection,
    config: &ConfigConnection,
    branch: Option<&str>,
    hash: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let history_store = DoltHistoryStore::new(graph);
    ensure_clean_working_set(&history_store, "checkout")?;
    if let Some(name) = branch {
        if !branch_exists(graph, name)? {
            history_store.create_branch(&BranchName(name.to_string()), None)?;
            println!("Created branch {name}");
        }
        println!("Checking out branch {name}");
        checkout(graph, name)
            .map_err(|error| format!("Failed to check out branch '{name}': {error}"))?;
        Defaults::set_current_branch(config, Some(name))
            .map_err(|error| format!("Failed to save current branch '{name}': {error}"))?;
    } else if let Some(hash_name) = hash {
        if branch_exists(graph, hash_name)? {
            println!("Checking out branch {hash_name}");
            checkout(graph, hash_name)
                .map_err(|error| format!("Failed to check out branch '{hash_name}': {error}"))?;
            Defaults::set_current_branch(config, Some(hash_name))
                .map_err(|error| format!("Failed to save current branch '{hash_name}': {error}"))?;
        } else {
            let commit_hash =
                history_store.resolve_operation_hash(&CommitRef(hash_name.to_string()))?;
            return Err(format!(
                "Detached HEAD checkouts are not supported for ref '{hash_name}' (resolved to {commit_hash}). Use --ref with read-only commands such as export, view, list-samples, list-graphs, or get-sequence."
            )
            .into());
        }
    } else {
        println!("No branch or hash to checkout provided.");
    }
    Ok(())
}
