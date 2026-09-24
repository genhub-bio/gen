use ratatui::{
    style::Style,
    text::{Line, Span},
};

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
