use std::{fs, path::PathBuf};

use gen_core::config::Workspace;
use gen_models::db::DbContext;

use crate::{
    commands::remote::{add_remote, login_remote, set_default_remote},
    get_connection, get_operation_connection,
    operation_management::pull,
    track_database,
};

const ORIGIN: &str = "origin";

pub fn execute(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let repo_name = infer_repo_name(url)?;
    let repo_path = PathBuf::from(&repo_name);

    fs::create_dir(&repo_path)?;

    let workspace = Workspace::new(&repo_path);
    workspace.ensure_gen_dir();
    println!("Gen repository initialized.");

    let operation_conn = get_operation_connection(Some(workspace.gen_db_path()?))?;
    add_remote(&operation_conn, ORIGIN, url)?;
    println!("Remote '{ORIGIN}' added successfully");

    set_default_remote(&operation_conn, ORIGIN)?;
    println!("Default remote set to '{ORIGIN}'");

    login_remote(&operation_conn, Some(ORIGIN))?;

    let graph_conn = get_connection(workspace.ensure_gen_dir().join("default.db"))?;
    let context = DbContext::new(workspace, graph_conn, operation_conn);
    track_database(context.graph().conn(), context.operations().conn())?;
    pull(&context, None)?;

    Ok(())
}

fn infer_repo_name(repo_url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let parsed = url::Url::parse(repo_url)?;
    let name = parsed
        .path_segments()
        .and_then(|segments| segments.rev().find(|segment| !segment.is_empty()))
        .ok_or("Unable to infer repository name from URL.")?;

    if name == "." || name == ".." || name.contains(std::path::MAIN_SEPARATOR) {
        return Err(format!("Invalid repository name inferred from URL: {name}").into());
    }

    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_repo_name_from_genhub_url() {
        assert_eq!(
            infer_repo_name("https://www.genhub.bio/api/repos/foo/bar").unwrap(),
            "bar"
        );
    }

    #[test]
    fn infer_repo_name_ignores_trailing_slash() {
        assert_eq!(
            infer_repo_name("https://www.genhub.bio/api/repos/foo/bar/").unwrap(),
            "bar"
        );
    }
}
