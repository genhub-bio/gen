use gen_core::CommitRef;
use gen_models::history::{HistoryEntry, HistoryStore};

pub fn operations_history_entries(
    history_store: &impl HistoryStore,
    branch_name: Option<&str>,
) -> Result<(String, Vec<HistoryEntry>), Box<dyn std::error::Error>> {
    let selected_branch_name = match branch_name {
        Some(branch_name) => branch_name.to_string(),
        None => history_store
            .current_branch()?
            .map(|branch| branch.0)
            .unwrap_or_else(|| "HEAD".to_string()),
    };

    let history_entries = match branch_name {
        Some(branch_name) => {
            history_store.log_for_ref(&CommitRef(branch_name.to_string()), None)?
        }
        None => history_store.log(None)?,
    };

    Ok((selected_branch_name, history_entries))
}

pub fn ensure_clean_working_set(
    history_store: &impl HistoryStore,
    action: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let status_rows = history_store.status()?;
    if status_rows.is_empty() {
        return Ok(());
    }

    Err(format!(
        "Cannot {action}: the working set has uncommitted changes. Commit or reset them first."
    )
    .into())
}

pub fn history_action_error(
    action: &str,
    error: &dyn std::error::Error,
) -> Box<dyn std::error::Error> {
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("conflict") {
        return format!(
            "{action} failed with Dolt conflicts. Gen does not yet provide conflict resolution commands. Original error: {message}"
        )
        .into();
    }

    format!("{action} failed: {message}").into()
}
