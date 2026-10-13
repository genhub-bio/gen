use std::path::Path;

use gen_models::{db::GraphConnection, history::dolt::connect_branch};
use ratatui::{
    style::Style,
    text::{Line, Span},
};

use crate::{get_raw_connection, get_ro_connection};

/// Opens the database connection a background viewer worker uses for its selected history.
pub(super) fn open_worker_connection(
    db_path: &Path,
    branch: Option<&str>,
    history_ref: Option<&str>,
) -> Result<GraphConnection, String> {
    if history_ref.is_some() {
        return get_ro_connection(db_path).map_err(|error| error.to_string());
    }

    let connection = get_raw_connection(db_path).map_err(|error| error.to_string())?;
    if let Some(branch) = branch {
        connect_branch(&connection, branch).map_err(|error| error.to_string())?;
    }
    Ok(connection)
}

/// Parses a string with markdown-like asterisk syntax for highlighting.
/// Segments surrounded by '*' are styled with `highlight_style`.
/// Other segments are styled with `default_style`.
pub fn style_text(text: &str, default_style: Style, highlight_style: Style) -> Line<'_> {
    let mut spans = Vec::new();
    let mut is_highlighted = false;
    for part in text.split('*') {
        if !part.is_empty() {
            spans.push(Span::styled(
                part,
                if is_highlighted {
                    highlight_style
                } else {
                    default_style
                },
            ));
        }
        is_highlighted = !is_highlighted;
    }
    Line::from(spans)
}
