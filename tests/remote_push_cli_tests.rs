use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use r#gen::{core::Workspace, get_connection, get_raw_connection};
use gen_models::{
    collection::Collection,
    db::get_connection as open_graph,
    history::dolt::{branch_hash, clone_remote, commit_all},
};
use tempfile::tempdir;
use url::Url;

fn gen_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_gen"))
}

fn run_gen(repo_root: &Path, args: &[&str]) -> Output {
    Command::new(gen_binary())
        .current_dir(repo_root)
        .args(args)
        .output()
        .expect("should run gen command")
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context}: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn initialize_repository(root: &Path) {
    fs::create_dir_all(root).expect("should create Gen repository directory");
    let output = run_gen(root, &["init"]);
    assert_success(&output, "Gen repository should initialize");
}

fn create_collection_commit(graph_path: &Path, collection_name: &str, message: &str) {
    let graph = open_graph(graph_path).expect("should open graph database");
    Collection::create(&graph, collection_name).expect("should create collection");
    commit_all(&graph, message).expect("should commit collection");
}

#[test]
fn test_cli_reports_non_fast_forward_and_force_overwrites_remote() {
    let temporary_directory = tempdir().expect("should create temporary test directory");
    let local_root = temporary_directory.path().join("local");
    let remote_root = temporary_directory.path().join("remote");
    initialize_repository(&local_root);
    initialize_repository(&remote_root);

    let local_workspace = Workspace::new(&local_root);
    let remote_workspace = Workspace::new(&remote_root);
    let local_graph_path = local_workspace
        .graph_db_path()
        .expect("should resolve local graph database path");
    let remote_graph_path = remote_workspace
        .graph_db_path()
        .expect("should resolve remote graph database path");
    create_collection_commit(&local_graph_path, "base", "common base");

    let remote_graph =
        get_raw_connection(&remote_graph_path).expect("should create empty remote graph database");
    let local_graph_url = Url::from_file_path(&local_graph_path)
        .expect("should create local graph database URL")
        .to_string();
    clone_remote(&remote_graph, &local_graph_url).expect("should clone common history to remote");
    drop(remote_graph);

    create_collection_commit(&remote_graph_path, "remote-only", "remote divergence");
    create_collection_commit(&local_graph_path, "local-only", "local divergence");

    let remote_url = Url::from_directory_path(&remote_root)
        .expect("should create remote workspace URL")
        .to_string();
    let add_remote = run_gen(&local_root, &["remote", "add", "origin", &remote_url]);
    assert_success(&add_remote, "file remote should be configured");

    let remote_head_before_rejection = {
        let graph = open_graph(&remote_graph_path).expect("should inspect remote before push");
        branch_hash(&graph, "main").expect("should read remote head before push")
    };
    let rejected = run_gen(&local_root, &["push", "--remote", "origin"]);
    assert!(!rejected.status.success(), "divergent push should fail");
    let remote_head_after_rejection = {
        let graph = open_graph(&remote_graph_path).expect("should inspect remote after rejection");
        branch_hash(&graph, "main").expect("should read remote head after rejection")
    };
    assert_eq!(
        remote_head_after_rejection, remote_head_before_rejection,
        "rejected push should leave remote main unchanged"
    );
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains(
            "The branch is not a fast-forward of the remote. Use --force to overwrite the remote."
        ),
        "CLI should explain how to overwrite a divergent remote: {stderr}"
    );
    assert!(!stderr.contains("Graph transfer failed"), "{stderr}");
    assert!(!stderr.contains("Graph upload attempt"), "{stderr}");
    assert!(
        !stderr.contains("No completed GCS block counters"),
        "{stderr}"
    );

    let forced = run_gen(&local_root, &["push", "--remote", "origin", "--force"]);
    assert_success(&forced, "forced push should overwrite divergent branch");
    assert!(
        String::from_utf8_lossy(&forced.stdout).contains("Pushed branch 'main' to 'origin'."),
        "successful forced push should be reported"
    );

    let local_graph =
        get_connection(&local_graph_path).expect("should reopen local graph database");
    let remote_graph = open_graph(&remote_graph_path).expect("should reopen remote graph database");
    assert_eq!(
        branch_hash(&local_graph, "main").expect("should read local main head"),
        branch_hash(&remote_graph, "main").expect("should read remote main head"),
        "forced push should replace the divergent remote head with local main"
    );
}
