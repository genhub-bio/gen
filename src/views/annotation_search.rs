use gen_tui::{LineStyle, plotter::PathStyle, theme::current_theme};

pub fn match_style() -> PathStyle {
    PathStyle::new(current_theme()[0x09])
        .with_line_style(LineStyle::Normal)
        .with_merge_glyphs(true)
}

pub fn selected_match_style() -> PathStyle {
    PathStyle::new(current_theme()[0x0B])
        .with_line_style(LineStyle::Bold)
        .with_merge_glyphs(true)
}

#[derive(Default)]
pub struct AnnotationSearchState {
    pub results: Vec<usize>,
    pub cursor: usize,
}

impl AnnotationSearchState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Case-insensitive substring search over `names` (parallel to caller's annotation slice).
    /// Fills `results` with matching indices; resets `cursor` to 0. Returns match count.
    pub fn search<'a>(&mut self, query: &str, names: impl Iterator<Item = &'a str>) -> usize {
        let q = query.to_lowercase();
        self.results = names
            .enumerate()
            .filter(|(_, name)| name.to_lowercase().contains(&q))
            .map(|(i, _)| i)
            .collect();
        self.cursor = 0;
        self.results.len()
    }

    /// Index into the caller's annotation slice for the current cursor position.
    pub fn current_annotation_idx(&self) -> Option<usize> {
        self.results.get(self.cursor).copied()
    }

    /// Advance cursor forward (wrapping). Returns false if no results.
    pub fn advance(&mut self) -> bool {
        if self.results.is_empty() {
            return false;
        }
        self.cursor = (self.cursor + 1) % self.results.len();
        true
    }

    /// Retreat cursor backward (wrapping). Returns false if no results.
    pub fn retreat(&mut self) -> bool {
        if self.results.is_empty() {
            return false;
        }
        self.cursor = self.cursor.checked_sub(1).unwrap_or(self.results.len() - 1);
        true
    }

    pub fn clear(&mut self) {
        self.results.clear();
        self.cursor = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}
