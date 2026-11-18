#[cfg(target_os = "emscripten")]
use std::path::PathBuf;
use std::{
    ffi::{CStr, CString, c_char},
    ptr,
};

#[cfg(any(target_os = "emscripten", test))]
use gen_core::{BranchName, CommitRef, config::Workspace};
#[cfg(target_os = "emscripten")]
use gen_models::operations::{Defaults, Remote, RemoteBranch};
#[cfg(any(target_os = "emscripten", test))]
use gen_models::{
    db::ConfigConnection,
    history::{HistoryStore, dolt::DoltHistoryStore},
};
use serde::{Deserialize, Serialize};
#[cfg(target_os = "emscripten")]
use sha2::{Digest, Sha256};

#[cfg(any(target_os = "emscripten", test))]
const DEFAULT_BRANCH: &str = "main";
#[cfg(target_os = "emscripten")]
const REMOTE_NAME: &str = "origin";

#[derive(Debug, Deserialize)]
#[serde(
    tag = "command",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
enum RepositoryAction {
    InitializeRepository {
        server_url: String,
        namespace: String,
        slug: String,
        branch: Option<String>,
        committer: String,
        email: String,
    },
    CreateRepositoryBranch {
        server_url: String,
        namespace: String,
        slug: String,
        source_branch: String,
        branch: String,
        committer: String,
        email: String,
    },
    MergeRepositoryBranches {
        server_url: String,
        namespace: String,
        slug: String,
        source_branch: String,
        target_branch: String,
        committer: String,
        email: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ActionResult {
    branch: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    commit_hash: Option<String>,
}

#[derive(Serialize)]
struct ActionResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<ActionResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl ActionResponse {
    fn success(result: ActionResult) -> Self {
        Self {
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    fn failure(error: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: None,
            error: Some(error.into()),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn genhub_repository_command(request: *const c_char) -> *mut c_char {
    if request.is_null() {
        return response_pointer(ActionResponse::failure("Invalid repository action request"));
    }
    let request = unsafe { CStr::from_ptr(request) };
    let response = match request.to_str() {
        Ok(request) => match serde_json::from_str::<RepositoryAction>(request) {
            Ok(action) => execute_action(action)
                .map(ActionResponse::success)
                .unwrap_or_else(ActionResponse::failure),
            Err(_) => ActionResponse::failure("Invalid repository action request"),
        },
        Err(_) => ActionResponse::failure("Invalid repository action request"),
    };
    response_pointer(response)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn genhub_free_string(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

fn response_pointer(response: ActionResponse) -> *mut c_char {
    let response = serde_json::to_string(&response)
        .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"Repository action failed\"}".to_string());
    CString::new(response)
        .map(CString::into_raw)
        .unwrap_or(ptr::null_mut())
}

fn execute_action(action: RepositoryAction) -> Result<ActionResult, String> {
    #[cfg(target_os = "emscripten")]
    {
        match action {
            RepositoryAction::InitializeRepository {
                server_url,
                namespace,
                slug,
                branch,
                committer,
                email,
            } => initialize_repository(
                &server_url,
                &namespace,
                &slug,
                branch.as_deref(),
                &committer,
                &email,
            ),
            RepositoryAction::CreateRepositoryBranch {
                server_url,
                namespace,
                slug,
                source_branch,
                branch,
                committer,
                email,
            } => create_repository_branch(
                &server_url,
                &namespace,
                &slug,
                &source_branch,
                &branch,
                &committer,
                &email,
            ),
            RepositoryAction::MergeRepositoryBranches {
                server_url,
                namespace,
                slug,
                source_branch,
                target_branch,
                committer,
                email,
            } => merge_repository_branches(
                &server_url,
                &namespace,
                &slug,
                &source_branch,
                &target_branch,
                &committer,
                &email,
            ),
        }
    }
    #[cfg(not(target_os = "emscripten"))]
    {
        match action {
            RepositoryAction::InitializeRepository {
                server_url,
                namespace,
                slug,
                branch,
                committer,
                email,
            } => {
                let _ = (server_url, namespace, slug, branch, committer, email);
            }
            RepositoryAction::CreateRepositoryBranch {
                server_url,
                namespace,
                slug,
                source_branch,
                branch,
                committer,
                email,
            } => {
                let _ = (
                    server_url,
                    namespace,
                    slug,
                    source_branch,
                    branch,
                    committer,
                    email,
                );
            }
            RepositoryAction::MergeRepositoryBranches {
                server_url,
                namespace,
                slug,
                source_branch,
                target_branch,
                committer,
                email,
            } => {
                let _ = (
                    server_url,
                    namespace,
                    slug,
                    source_branch,
                    target_branch,
                    committer,
                    email,
                );
            }
        }
        Err("Gen browser repository actions require the Emscripten build".to_string())
    }
}

#[cfg(target_os = "emscripten")]
fn initialize_repository(
    server_url: &str,
    namespace: &str,
    slug: &str,
    requested_branch: Option<&str>,
    committer: &str,
    email: &str,
) -> Result<ActionResult, String> {
    let remote_url = repository_url(server_url, namespace, slug)?;
    let branch = checked_branch(requested_branch.unwrap_or(DEFAULT_BRANCH))?;
    let workspace = action_workspace(&remote_url, namespace, slug, email)?;
    workspace.ensure_gen_dir();
    let config =
        r#gen::get_config_connection(Some(workspace.gen_db_path().map_err(safe_workspace_error)?))
            .map_err(|_| "Could not initialize the Gen workspace".to_string())?;
    let graph = r#gen::get_raw_connection(workspace.graph_db_path().map_err(safe_workspace_error)?)
        .map_err(|_| "Could not initialize the Gen graph database".to_string())?;
    configure_author(&graph, committer, email)?;
    let mut migrations_connection = graph.0;
    gen_models::migrations::run_migrations(&mut migrations_connection);
    let graph = gen_models::db::GraphConnection(migrations_connection);
    select_initial_branch(&graph, &branch)?;
    configure_remote(&config, &remote_url, &branch)?;
    let context = gen_models::db::DbContext::new_raw(workspace.clone(), graph, config);
    let commit_hash = commit_repository_readme(&context, &remote_url, &branch)?;
    drop(context);
    push_branch(&workspace, &branch)?;
    Ok(ActionResult {
        branch,
        commit_hash: Some(commit_hash.to_string()),
    })
}

#[cfg(target_os = "emscripten")]
fn create_repository_branch(
    server_url: &str,
    namespace: &str,
    slug: &str,
    source_branch: &str,
    branch: &str,
    committer: &str,
    email: &str,
) -> Result<ActionResult, String> {
    let remote_url = repository_url(server_url, namespace, slug)?;
    let source_branch = checked_branch(source_branch)?;
    let branch = checked_branch(branch)?;
    if source_branch == branch {
        return Err("The new branch must differ from its source".to_string());
    }
    let workspace = action_workspace(&remote_url, namespace, slug, email)?;
    let opened =
        r#gen::commands::remote::operations::open_browser_graph(&remote_url, Some(&source_branch))?;
    let workspace = workspace
        .with_graph_db_path_override(opened.graph_path.clone())
        .with_asset_reader_source(std::sync::Arc::clone(&opened.asset_reader_source));
    workspace.ensure_gen_dir();
    let config =
        r#gen::get_config_connection(Some(workspace.gen_db_path().map_err(safe_workspace_error)?))
            .map_err(|_| "Could not open the GenHub workspace".to_string())?;
    let graph = &opened.graph;
    configure_remote(&config, &remote_url, &source_branch)?;
    configure_author(graph, committer, email)?;
    let commit_hash = create_branch_from(graph, &source_branch, &branch)?;
    Defaults::set_current_branch(&config, Some(&branch))
        .map_err(|_| "Could not save the selected branch".to_string())?;
    RemoteBranch::set_remote_validated(&config, &branch, Some(REMOTE_NAME))
        .map_err(|_| "Could not configure the branch remote".to_string())?;
    push_branch_with_published_assets(&workspace, &branch)?;
    drop(opened);
    Ok(ActionResult {
        branch,
        commit_hash: Some(commit_hash),
    })
}

#[cfg(target_os = "emscripten")]
fn merge_repository_branches(
    server_url: &str,
    namespace: &str,
    slug: &str,
    source_branch: &str,
    target_branch: &str,
    committer: &str,
    email: &str,
) -> Result<ActionResult, String> {
    let remote_url = repository_url(server_url, namespace, slug)?;
    let source_branch = checked_branch(source_branch)?;
    let target_branch = checked_branch(target_branch)?;
    if source_branch == target_branch {
        return Err("Source and target branches must be different".to_string());
    }
    let workspace = action_workspace(&remote_url, namespace, slug, email)?;
    let opened =
        r#gen::commands::remote::operations::open_browser_graph(&remote_url, Some(&target_branch))?;
    let workspace = workspace
        .with_graph_db_path_override(opened.graph_path.clone())
        .with_asset_reader_source(std::sync::Arc::clone(&opened.asset_reader_source));
    workspace.ensure_gen_dir();
    let config =
        r#gen::get_config_connection(Some(workspace.gen_db_path().map_err(safe_workspace_error)?))
            .map_err(|_| "Could not open the GenHub workspace".to_string())?;
    configure_remote(&config, &remote_url, &target_branch)?;
    configure_author(&opened.graph, committer, email)?;
    let previous_target_hash = gen_models::history::dolt::hash_of(&opened.graph, &target_branch)
        .map_err(|_| "Could not read the target branch head".to_string())?;
    let commit_hash = merge_branch_into(
        &opened.graph,
        &workspace,
        &config,
        &source_branch,
        committer,
        email,
    )?;
    Defaults::set_current_branch(&config, Some(&target_branch))
        .map_err(|_| "Could not save the target branch".to_string())?;
    RemoteBranch::set_remote_validated(&config, &target_branch, Some(REMOTE_NAME))
        .map_err(|_| "Could not configure the target branch remote".to_string())?;
    if previous_target_hash.to_string() != commit_hash {
        push_branch_with_published_assets(&workspace, &target_branch)?;
    }
    drop(opened);
    Ok(ActionResult {
        branch: target_branch,
        commit_hash: Some(commit_hash),
    })
}

#[cfg(any(target_os = "emscripten", test))]
fn create_branch_from(
    graph: &gen_models::db::GraphConnection,
    source_branch: &str,
    branch: &str,
) -> Result<String, String> {
    let source_branch = checked_branch(source_branch)?;
    let branch = checked_branch(branch)?;
    if !gen_models::history::dolt::branch_exists(graph, &source_branch)
        .map_err(|_| "Could not check the source branch".to_string())?
    {
        return Err("The source branch does not exist".to_string());
    }
    if gen_models::history::dolt::branch_exists(graph, &branch)
        .map_err(|_| "Could not check whether the branch already exists".to_string())?
    {
        return Err("A branch with that name already exists".to_string());
    }
    DoltHistoryStore::new(graph)
        .create_branch(&BranchName(branch.clone()), Some(&CommitRef(source_branch)))
        .map_err(|_| "Could not create the branch".to_string())?;
    gen_models::history::dolt::connect_branch(graph, &branch)
        .map_err(|_| "Could not select the new branch".to_string())?;
    gen_models::history::dolt::hash_of(graph, &branch)
        .map(|hash| hash.to_string())
        .map_err(|_| "Could not read the new branch head".to_string())
}

#[cfg(any(target_os = "emscripten", test))]
fn select_initial_branch(
    graph: &gen_models::db::GraphConnection,
    branch: &str,
) -> Result<(), String> {
    let branch = checked_branch(branch)?;
    let current = gen_models::history::dolt::active_branch(graph)
        .map_err(|_| "Could not read the initialized Gen branch".to_string())?;
    if current == branch {
        return Ok(());
    }
    if current != DEFAULT_BRANCH {
        return Err(
            "The local initialization branch does not match the repository default".to_string(),
        );
    }
    graph
        .query_row(
            "SELECT dolt_branch('-m', ?1, ?2)",
            [DEFAULT_BRANCH, branch.as_str()],
            |row| row.get_ref(0).map(|_| ()),
        )
        .map_err(|_| "Could not select the repository's default branch".to_string())?;
    if gen_models::history::dolt::active_branch(graph)
        .ok()
        .as_deref()
        != Some(branch.as_str())
    {
        gen_models::history::dolt::connect_branch(graph, &branch)
            .map_err(|_| "Could not select the repository's default branch".to_string())?;
    }
    Ok(())
}

#[cfg(any(target_os = "emscripten", test))]
fn merge_branch_into(
    graph: &gen_models::db::GraphConnection,
    workspace: &Workspace,
    config: &ConfigConnection,
    source_branch: &str,
    committer: &str,
    email: &str,
) -> Result<String, String> {
    let source_branch = checked_branch(source_branch)?;
    if !gen_models::history::dolt::branch_exists(graph, &source_branch)
        .map_err(|_| "Could not check the source branch".to_string())?
    {
        return Err("The source branch does not exist".to_string());
    }
    configure_author(graph, committer, email)?;
    graph
        .with_transaction_behavior(
            gen_models::db::graph_write_transaction_behavior(workspace),
            || DoltHistoryStore::new_with_config(graph, config).merge(&CommitRef(source_branch)),
        )
        .map(|hash| hash.to_string())
        .map_err(|_| "The merge could not be completed; check for conflicts and retry".to_string())
}

#[cfg(any(target_os = "emscripten", test))]
fn configure_author(
    graph: &gen_models::db::GraphConnection,
    committer: &str,
    email: &str,
) -> Result<(), String> {
    if committer.trim().is_empty() || email.trim().is_empty() {
        return Err("A signed-in user identity is required".to_string());
    }
    gen_models::history::dolt::set_commit_author_name(graph, committer)
        .and_then(|()| gen_models::history::dolt::set_commit_author_email(graph, email))
        .map_err(|_| "Could not configure the repository author".to_string())
}

#[cfg(any(target_os = "emscripten", test))]
fn commit_repository_readme(
    context: &gen_models::db::DbContext,
    remote_url: &str,
    branch: &str,
) -> Result<String, String> {
    let readme_path = context.workspace().base_dir().join("README.md");
    std::fs::write(&readme_path, repository_readme_contents(remote_url))
        .map_err(|_| "Could not create the repository README".to_string())?;
    let readme_file =
        gen_models::operations::OperationFile::new(readme_path.to_string_lossy().into_owned());
    let operation = match gen_models::operations::add_files_operation(
        context,
        &[readme_file],
        Some("Add repository README"),
    ) {
        Ok(operation) => operation,
        Err(gen_models::errors::AddFilesOperationError::OperationError(
            gen_models::errors::OperationError::NoChanges,
        )) => {
            return gen_models::history::dolt::hash_of(context.graph().conn(), branch)
                .map(|hash| hash.to_string())
                .map_err(|_| "Could not read the initialized branch head".to_string());
        }
        Err(_) => return Err("Could not track the repository README".to_string()),
    };
    DoltHistoryStore::new_with_config(context.graph().conn(), context.config().conn())
        .commit_all(&operation.summary)
        .map(|hash| hash.to_string())
        .map_err(|_| "Could not commit the repository README".to_string())
}

#[cfg(any(target_os = "emscripten", test))]
fn repository_readme_contents(remote_url: &str) -> String {
    format!(
        "# GenHub repository\n\nRun this command to clone the repository with Gen.\n\n```sh\ngen clone {remote_url}\n```\n\nIf prompted, sign in to GenHub to authorize the clone.\n"
    )
}

#[cfg(target_os = "emscripten")]
fn configure_remote(
    config: &ConfigConnection,
    remote_url: &str,
    branch: &str,
) -> Result<(), String> {
    match Remote::get_by_name_optional(config, REMOTE_NAME) {
        Some(remote) if remote.url != remote_url => {
            return Err("The GenHub workspace is configured for another repository".to_string());
        }
        Some(_) => {}
        None => {
            Remote::create(config, REMOTE_NAME, remote_url)
                .map_err(|_| "Could not configure the GenHub remote".to_string())?;
        }
    }
    Defaults::set_default_remote(config, Some(REMOTE_NAME))
        .map_err(|_| "Could not select the GenHub remote".to_string())?;
    Defaults::set_current_branch(config, Some(branch))
        .map_err(|_| "Could not save the selected branch".to_string())?;
    RemoteBranch::set_remote_validated(config, branch, Some(REMOTE_NAME))
        .map_err(|_| "Could not configure the branch remote".to_string())
}

#[cfg(target_os = "emscripten")]
fn push_branch(workspace: &Workspace, branch: &str) -> Result<(), String> {
    r#gen::commands::remote::operations::execute_push(
        workspace,
        Some(REMOTE_NAME),
        Some(branch),
        false,
    )
    .map_err(|_| {
        "GenHub could not publish the branch; refresh the repository and retry".to_string()
    })
}

#[cfg(target_os = "emscripten")]
fn push_branch_with_published_assets(workspace: &Workspace, branch: &str) -> Result<(), String> {
    r#gen::commands::remote::operations::execute_push_with_published_assets(
        workspace,
        Some(REMOTE_NAME),
        Some(branch),
    )
    .map_err(|_| {
        "GenHub could not publish the branch; refresh the repository and retry".to_string()
    })
}

#[cfg(target_os = "emscripten")]
fn repository_url(server_url: &str, namespace: &str, slug: &str) -> Result<String, String> {
    checked_path_segment(namespace)?;
    checked_path_segment(slug)?;
    let mut server_url =
        url::Url::parse(server_url).map_err(|_| "Invalid GenHub application URL".to_string())?;
    if !matches!(server_url.scheme(), "http" | "https")
        || server_url.host_str().is_none()
        || !server_url.username().is_empty()
        || server_url.password().is_some()
        || server_url.query().is_some()
        || server_url.fragment().is_some()
    {
        return Err("Invalid GenHub application URL".to_string());
    }
    server_url.set_path(&format!("/api/repos/{namespace}/{slug}"));
    Ok(server_url.to_string().trim_end_matches('/').to_string())
}

#[cfg(any(target_os = "emscripten", test))]
fn checked_path_segment(value: &str) -> Result<(), String> {
    if value.is_empty()
        || matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err("Invalid repository identity".to_string());
    }
    Ok(())
}

#[cfg(any(target_os = "emscripten", test))]
fn checked_branch(value: &str) -> Result<String, String> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err("Branch name is invalid".to_string());
    }
    Ok(value.to_string())
}

#[cfg(target_os = "emscripten")]
fn action_workspace(
    remote_url: &str,
    namespace: &str,
    slug: &str,
    email: &str,
) -> Result<Workspace, String> {
    checked_path_segment(namespace)?;
    checked_path_segment(slug)?;
    let mut hasher = Sha256::new();
    hasher.update(remote_url.as_bytes());
    hasher.update([0]);
    hasher.update(email.as_bytes());
    let digest = hasher.finalize();
    let workspace_key = digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(Workspace::new(PathBuf::from(format!(
        "/genhub-actions/{workspace_key}/{namespace}/{slug}"
    ))))
}

#[cfg(target_os = "emscripten")]
fn safe_workspace_error(error: impl std::fmt::Display) -> String {
    let _ = error;
    "Could not prepare the GenHub workspace".to_string()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_core::config::Workspace;
    use gen_models::{
        assets::AssetRef,
        db::{DbContext, get_config_connection, get_connection},
        history::dolt::{
            active_branch, branch_rows, commit_all, connect_branch, hash_of, status_rows,
        },
    };

    use super::{
        RepositoryAction, checked_branch, checked_path_segment, commit_repository_readme,
        create_branch_from, merge_branch_into, select_initial_branch,
    };

    #[test]
    fn test_checked_branch_rejects_empty_and_control_names() {
        assert!(checked_branch("").is_err());
        assert!(checked_branch("feature\nmain").is_err());
        assert_eq!(
            checked_branch("feature/graph").as_deref(),
            Ok("feature/graph")
        );
    }

    #[test]
    fn test_checked_path_segment_rejects_path_injection() {
        assert!(checked_path_segment("alice/api").is_err());
        assert!(checked_path_segment("..").is_err());
        assert!(checked_path_segment("alice-genhub").is_ok());
    }

    #[test]
    fn test_action_deserialization_uses_camel_case_fields() {
        let action: RepositoryAction = serde_json::from_str(
            r#"{"command":"mergeRepositoryBranches","serverUrl":"https://genhub.example","namespace":"alice","slug":"sample","sourceBranch":"feature","targetBranch":"main","committer":"Alice","email":"alice@users.noreply.genhub.bio"}"#,
        )
        .expect("should parse the public JavaScript action shape");
        assert!(matches!(
            action,
            RepositoryAction::MergeRepositoryBranches {
                source_branch,
                target_branch,
                committer,
                ..
            } if source_branch == "feature" && target_branch == "main" && committer == "Alice"
        ));
    }

    #[test]
    fn test_create_branch_starts_at_requested_head_and_refuses_existing_branch() {
        let directory = tempfile::tempdir().expect("should create a temporary workspace");
        let graph = get_connection(directory.path().join("default.db"))
            .expect("should initialize a Gen graph database");
        let source_hash = hash_of(&graph, "main").expect("should read the initial branch head");
        let branch_hash = create_branch_from(&graph, "main", "ui-branch")
            .expect("should create a branch from main");
        assert_eq!(branch_hash, source_hash.to_string());
        assert_eq!(
            active_branch(&graph).expect("should read the active branch"),
            "ui-branch"
        );
        assert!(create_branch_from(&graph, "main", "ui-branch").is_err());
    }

    #[test]
    fn test_initialization_renames_main_to_the_configured_default_branch() {
        let directory = tempfile::tempdir().expect("should create a temporary workspace");
        let graph = get_connection(directory.path().join("default.db"))
            .expect("should initialize a Gen graph database");
        select_initial_branch(&graph, "release")
            .expect("should select the configured repository default branch");
        let branches = branch_rows(&graph).expect("should read initialized repository branches");
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].name, "release");
        assert_eq!(
            active_branch(&graph).expect("should read active branch"),
            "release"
        );
    }

    #[test]
    fn test_initial_repository_readme_is_tracked_and_committed_on_selected_branch() {
        let directory = tempfile::tempdir().expect("should create a temporary workspace");
        let workspace = Workspace::new(directory.path());
        workspace.ensure_gen_dir();
        let config = get_config_connection(
            workspace
                .gen_db_path()
                .expect("should locate config database"),
        )
        .expect("should initialize a Gen config database");
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should locate graph database"),
        )
        .expect("should initialize a Gen graph database");
        super::configure_author(&graph, "Alice", "alice@users.noreply.genhub.bio")
            .expect("should configure the repository author");
        select_initial_branch(&graph, "release")
            .expect("should select the configured repository default branch");
        let context = DbContext::new_raw(workspace.clone(), graph, config);
        let remote_url = "https://genhub.example/api/repos/alice/sample";

        let commit_hash = commit_repository_readme(&context, remote_url, "release")
            .expect("should commit the README to the configured branch");
        let retry_hash = commit_repository_readme(&context, remote_url, "release")
            .expect("should preserve a committed README on retry");
        let head = hash_of(context.graph().conn(), "release")
            .expect("should read the initialized branch head");
        let readme =
            AssetRef::get_materialized_assets_at(context.graph().conn(), None, Some(&head))
                .expect("should read files committed on the initial branch")
                .into_iter()
                .find(|asset| asset.name.as_deref() == Some("README.md"))
                .expect("should find the README in the initial branch history");
        let readme_contents = std::fs::read_to_string(workspace.base_dir().join("README.md"))
            .expect("should read the generated README");

        assert_eq!(commit_hash, head.to_string());
        assert_eq!(retry_hash, commit_hash);
        assert_eq!(
            active_branch(context.graph().conn()).expect("should read the active branch"),
            "release"
        );
        assert_eq!(readme.logical_path.as_deref(), Some("README.md"));
        assert_eq!(readme.file_type, "none");
        let clone_command = format!("```sh\ngen clone {remote_url}\n```");
        assert!(readme_contents.contains(&clone_command));
        assert!(readme_contents.contains("sign in to GenHub"));
        assert!(
            status_rows(context.graph().conn())
                .expect("should read repository status")
                .is_empty(),
            "the README commit should leave the selected branch clean"
        );
    }

    #[test]
    fn test_merge_branch_into_commits_to_active_target_branch() {
        let directory = tempfile::tempdir().expect("should create a temporary workspace");
        let workspace = Workspace::new(PathBuf::from(directory.path()));
        workspace.ensure_gen_dir();
        let config = get_config_connection(
            workspace
                .gen_db_path()
                .expect("should locate config database"),
        )
        .expect("should initialize a Gen config database");
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should locate graph database"),
        )
        .expect("should initialize a Gen graph database");
        graph
            .execute_batch(
                "CREATE TABLE action_rows (id INTEGER PRIMARY KEY, value TEXT NOT NULL);",
            )
            .expect("should create an isolated merge fixture table");
        commit_all(&graph, "Create action fixture table")
            .expect("should commit the fixture schema");
        create_branch_from(&graph, "main", "feature").expect("should create a feature branch");
        graph
            .execute(
                "INSERT INTO action_rows (id, value) VALUES (1, 'feature')",
                [],
            )
            .expect("should update the feature branch");
        commit_all(&graph, "Feature change").expect("should commit the feature branch");
        connect_branch(&graph, "main").expect("should return to the main target branch");
        let merge_hash = merge_branch_into(
            &graph,
            &workspace,
            &config,
            "feature",
            "Alice",
            "alice@users.noreply.genhub.bio",
        )
        .expect("should merge the feature branch through Gen history");
        let value: String = graph
            .query_row("SELECT value FROM action_rows WHERE id = 1", [], |row| {
                row.get(0)
            })
            .expect("should read the merged feature row");
        let target_hash = hash_of(&graph, "main").expect("should read the merged target head");
        assert_eq!(value, "feature");
        assert_eq!(merge_hash, target_hash.to_string());
        assert_eq!(
            active_branch(&graph).expect("should read active target"),
            "main"
        );
    }

    #[test]
    fn test_merge_conflict_keeps_target_head_and_working_set_unchanged() {
        let directory = tempfile::tempdir().expect("should create a temporary workspace");
        let workspace = Workspace::new(PathBuf::from(directory.path()));
        workspace.ensure_gen_dir();
        let config = get_config_connection(
            workspace
                .gen_db_path()
                .expect("should locate config database"),
        )
        .expect("should initialize a Gen config database");
        let graph = get_connection(
            workspace
                .graph_db_path()
                .expect("should locate graph database"),
        )
        .expect("should initialize a Gen graph database");
        graph
            .execute_batch(
                "CREATE TABLE action_rows (id INTEGER PRIMARY KEY, value TEXT NOT NULL);",
            )
            .expect("should create an isolated conflict fixture table");
        commit_all(&graph, "Create conflict fixture table")
            .expect("should commit the fixture schema");
        create_branch_from(&graph, "main", "feature").expect("should create a feature branch");
        graph
            .execute(
                "INSERT INTO action_rows (id, value) VALUES (1, 'feature')",
                [],
            )
            .expect("should update the feature branch");
        commit_all(&graph, "Feature conflict").expect("should commit the feature branch");
        connect_branch(&graph, "main").expect("should return to the main target branch");
        graph
            .execute(
                "INSERT INTO action_rows (id, value) VALUES (1, 'target')",
                [],
            )
            .expect("should update the target branch");
        commit_all(&graph, "Target conflict").expect("should commit the target branch");
        let target_hash = hash_of(&graph, "main").expect("should read target head");
        let error = merge_branch_into(
            &graph,
            &workspace,
            &config,
            "feature",
            "Alice",
            "alice@users.noreply.genhub.bio",
        )
        .expect_err("should reject a conflicting merge");
        let value: String = graph
            .query_row("SELECT value FROM action_rows WHERE id = 1", [], |row| {
                row.get(0)
            })
            .expect("should read the unchanged target row");
        assert!(error.contains("conflicts"));
        assert_eq!(
            hash_of(&graph, "main").expect("should reread target head"),
            target_hash
        );
        assert_eq!(value, "target");
        assert!(
            status_rows(&graph)
                .expect("should read working-set status")
                .is_empty()
        );
        assert_eq!(
            active_branch(&graph).expect("should read active target"),
            "main"
        );
    }
}
