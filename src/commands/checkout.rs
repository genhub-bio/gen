use std::{collections::HashMap, error::Error};

use gen_core::{BranchName, CommitRef, DoltHashId, HashId, config::Workspace};
use gen_models::{
    assets::{AssetRef, materialization_destination_path},
    db::{ConfigConnection, GraphConnection},
    history::{
        HistoryStore,
        dolt::{DoltHistoryStore, branch_exists, checkout, connect_branch, hash_of},
    },
    operations::Defaults,
};

use crate::{
    commands::remote::operations::{
        DownloadAssetOutcome, materialize_versioned_asset, warn_asset_conflict,
    },
    history::ensure_clean_working_set,
};

/// Restores assets present at the requested commit from `.gen/assets` into the workspace
///
/// Previously stored versioned files are safely replaced by requested versions.
/// Unknown local contents are preserved and receive the requested version marked as a conflict.
fn materialize_checked_out_assets(
    graph: &GraphConnection,
    workspace: &Workspace,
    commit_hash: &DoltHashId,
    previous_assets: &HashMap<HashId, AssetRef>,
) -> Result<(), Box<dyn Error>> {
    for asset in AssetRef::get_materialized_assets_at(graph, None, Some(commit_hash))? {
        let destination_logical_path = asset.logical_path.as_deref();
        let versioned_path =
            materialization_destination_path(workspace, &asset.uri, asset.checksum.as_ref(), None)?;
        if let DownloadAssetOutcome::Conflict(conflict_path) = materialize_versioned_asset(
            workspace,
            &asset,
            previous_assets,
            destination_logical_path,
            &versioned_path,
            false,
        )? {
            warn_asset_conflict(workspace, &asset, destination_logical_path, &conflict_path)?;
        }
    }
    Ok(())
}

pub fn execute(
    graph: &GraphConnection,
    config: &ConfigConnection,
    workspace: &Workspace,
    branch: Option<&str>,
    hash: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // We want to connect to the current branch here instead of the branch we are checking out. This lets us
    // track and record the current state of assets prior to the checkout, so we can distinguish whether
    // a file should be replaced or if it needs to be preserved as a conflict.
    if let Some(current_branch) = Defaults::get_current_branch(config)
        && branch_exists(graph, &current_branch)?
    {
        connect_branch(graph, &current_branch)?;
    }
    let history_store = DoltHistoryStore::new(graph);
    ensure_clean_working_set(&history_store, "checkout")?;
    let previous_assets = AssetRef::get_cumulative_assets_at(graph, None, None)?
        .into_iter()
        .map(|asset| (asset.id, asset))
        .collect();
    if let Some(name) = branch {
        let branch_already_exists = branch_exists(graph, name)?;
        if branch_already_exists && hash.is_some() {
            return Err(format!(
                "Branch '{name}' already exists; cannot start it at a given operation. Choose a new branch name."
            )
            .into());
        }
        if !branch_already_exists {
            let start_ref = hash.map(|hash_name| CommitRef(hash_name.to_string()));
            history_store.create_branch(&BranchName(name.to_string()), start_ref.as_ref())?;
            println!("Created branch {name}");
        }
        println!("Checking out branch {name}");
        checkout(graph, name)
            .map_err(|error| format!("Failed to check out branch '{name}': {error}"))?;
        Defaults::set_current_branch(config, Some(name))
            .map_err(|error| format!("Failed to save current branch '{name}': {error}"))?;
        let commit_hash = hash_of(graph, name)?;
        materialize_checked_out_assets(graph, workspace, &commit_hash, &previous_assets)?;
    } else if let Some(hash_name) = hash {
        if branch_exists(graph, hash_name)? {
            println!("Checking out branch {hash_name}");
            checkout(graph, hash_name)
                .map_err(|error| format!("Failed to check out branch '{hash_name}': {error}"))?;
            Defaults::set_current_branch(config, Some(hash_name))
                .map_err(|error| format!("Failed to save current branch '{hash_name}': {error}"))?;
            let commit_hash = hash_of(graph, hash_name)?;
            materialize_checked_out_assets(graph, workspace, &commit_hash, &previous_assets)?;
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

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        io::{Cursor, Write as _},
    };

    use gen_core::{HashId, Sha256Hash, config::Workspace};
    use gen_models::{
        assets::{AssetRef, AssetRole, LocalAssetUri, materialization_destination_path},
        collection::Collection,
        file_types::FileTypes,
        history::dolt::commit_all,
        operations::{FileAddition, calculate_reader_checksum},
    };
    use noodles::bgzf;
    use tempfile::tempdir;

    use super::materialize_checked_out_assets;
    use crate::get_connection;

    fn asset(
        uri: &str,
        logical_path: &str,
        archived_contents: &[u8],
        materialized_checksum: Sha256Hash,
        created_on: i64,
    ) -> AssetRef {
        let checksum = calculate_reader_checksum(Cursor::new(archived_contents))
            .expect("should checksum archived bytes");
        let role = AssetRole::Input;
        let uri = LocalAssetUri::asset_uri(uri);
        let file_addition = FileAddition {
            id: HashId::convert_str("checkout-test-asset"),
            asset_uri: uri.clone(),
            file_type: FileTypes::Fasta,
            checksum: Some(checksum),
            materialized_checksum: Some(materialized_checksum),
        };
        AssetRef {
            id: AssetRef::id_hash(
                &file_addition,
                &role,
                Some(logical_path),
                Some("reference.fa.bgz"),
                None,
            ),
            uri,
            file_type: "fasta".to_string(),
            checksum: Some(checksum),
            materialized_checksum: Some(materialized_checksum),
            size: Some(
                i64::try_from(archived_contents.len())
                    .expect("should fit archived input size in i64"),
            ),
            role,
            logical_path: Some(logical_path.to_string()),
            name: Some("reference.fa.bgz".to_string()),
            created_on,
            upstream_asset_ref_id: None,
        }
    }

    fn archived_fasta(contents: &[u8]) -> Vec<u8> {
        let mut archived_contents = Vec::new();
        let mut writer = bgzf::io::Writer::new(&mut archived_contents);
        writer
            .write_all(contents)
            .expect("should write FASTA bytes to BGZF");
        writer.finish().expect("should finish BGZF stream");
        archived_contents
    }

    #[test]
    fn test_checkout_materializes_bgzf_over_known_previous_plain_file() {
        let temp = tempdir().expect("should create workspace");
        let workspace = Workspace::new(temp.path());
        workspace.ensure_gen_dir();
        let graph = get_connection(workspace.graph_db_path().unwrap())
            .expect("should create graph database");
        Collection::create(&graph, "checkout-fixture").expect("should create collection");

        let previous_contents = b">chr1\nAACCGG\n";
        let previous_archive = archived_fasta(previous_contents);
        let previous_asset = asset(
            ".gen/assets/previous.fa.bgz",
            "reference.fa",
            &previous_archive,
            calculate_reader_checksum(Cursor::new(previous_contents))
                .expect("should checksum previous FASTA"),
            1,
        );
        fs::write(temp.path().join("reference.fa"), previous_contents)
            .expect("should write previous plain FASTA");

        let current_contents = b">chr1\nTTGGCC\n";
        let current_archive = archived_fasta(current_contents);
        let current_asset = asset(
            ".gen/assets/current.fa.bgz",
            "reference.fa",
            &current_archive,
            calculate_reader_checksum(Cursor::new(current_contents))
                .expect("should checksum current FASTA"),
            2,
        );
        let current_versioned_path = materialization_destination_path(
            &workspace,
            &current_asset.uri,
            current_asset.checksum.as_ref(),
            None,
        )
        .expect("should resolve current archive path");
        fs::create_dir_all(
            current_versioned_path
                .parent()
                .expect("should have archive path parent"),
        )
        .expect("should create asset directory");
        fs::write(&current_versioned_path, &current_archive)
            .expect("should write current BGZF archive");
        AssetRef::create(&graph, &current_asset).expect("should add current asset ref");
        let commit_hash =
            commit_all(&graph, "add archived FASTA").expect("should commit current asset ref");
        let previous_assets =
            HashMap::<HashId, AssetRef>::from([(previous_asset.id, previous_asset)]);

        materialize_checked_out_assets(&graph, &workspace, &commit_hash, &previous_assets)
            .expect("should materialize checked-out FASTA");

        assert_eq!(
            fs::read(temp.path().join("reference.fa")).expect("should read restored FASTA"),
            current_contents,
            "checkout should replace a recognized previous plain file with decoded bytes"
        );
        assert_eq!(
            fs::read(current_versioned_path).expect("should read retained BGZF archive"),
            current_archive,
            "checkout should leave archived BGZF bytes intact"
        );
    }
}
