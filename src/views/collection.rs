use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crossterm::event::{KeyCode, KeyEvent};
use gen_core::HashId;
use gen_models::{
    block_group::BlockGroup,
    collection::Collection,
    db::{GraphConnection, OperationsConnection},
    file_types::FileTypes,
    sample::Sample,
    traits::Query,
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, StatefulWidget, Wrap},
};
use rusqlite::params;
use tui_widget_list::{ListBuilder, ListState, ListView};

use crate::{
    config::get_theme_color,
    views::{
        annotation_files::{AnnotationFileEntry, load_annotation_file_entries},
        annotation_groups::{AnnotationGroupEntry, load_annotation_group_entries},
    },
};

/// Represents the different focus zones in the UI
/// TODO: implement a proper cycler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusZone {
    Canvas,
    Panel,
    Sidebar,
}
// For debugging
impl fmt::Display for FocusZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FocusZone::Canvas => write!(f, "canvas"),
            FocusZone::Panel => write!(f, "panel"),
            FocusZone::Sidebar => write!(f, "sidebar"),
        }
    }
}

/// Normalize a hierarchical collection name by removing trailing delimiters
/// (except if the entire collection name is "/"). For example:
/// "/foo/bar///" -> "/foo/bar", but "/" stays "/".
fn normalize_collection_name(mut full_collection: &str) -> &str {
    if full_collection == "/" {
        return "/";
    }
    full_collection = full_collection.trim_end_matches('/');
    if full_collection.is_empty() {
        // If it was all delimiters (e.g. "////"), treat it as "/"
        "/"
    } else {
        full_collection
    }
}

/// Return the final segment of a hierarchical collection name. For example,
/// given "/foo/bar", the final segment is "bar". Special case: "/" is root.
fn collection_basename(full_collection: &str) -> &str {
    let normalized = normalize_collection_name(full_collection);
    if normalized == "/" {
        return "/";
    }
    if let Some(idx) = normalized.rfind('/') {
        &normalized[idx + 1..]
    } else {
        normalized
    }
}

/// Return the parent portion of a hierarchical collection name. For example:
///   parent_collection("/foo/bar")   -> "/foo"
///   parent_collection("/foo/bar/")  -> "/foo"
///   parent_collection("/foo")       -> "/"
///   parent_collection("/")          -> "/"
///   parent_collection("bar")        -> "."
///
/// Note: If there's no slash in `full_collection`, we return "." to indicate
/// the "current directory" (matching typical Unix `dirname` behavior).
fn parent_collection(full_collection: &str) -> String {
    let normalized = normalize_collection_name(full_collection);
    if normalized == "/" {
        // Root has no parent
        return "/".to_string();
    }
    if let Some(idx) = normalized.rfind('/') {
        if idx == 0 {
            // "/foo"; parent is "/"
            "/".to_string()
        } else {
            normalized[..idx].to_string()
        }
    } else {
        // If there's no slash, treat it as a single component => parent is "."
        ".".to_string()
    }
}

#[derive(Debug)]
pub struct CollectionExplorerData {
    /// The final segment of the current collection name. For example,
    /// if the full collection is "/foo/bar", this would be "bar".
    pub current_collection: String,
    /// The block groups in the *entire* collection that have sample_name = NULL
    pub reference_block_groups: Vec<(HashId, String)>,
    /// The samples in the entire collection
    pub collection_samples: Vec<String>,
    /// The block groups for each sample
    pub sample_block_groups: HashMap<String, Vec<(HashId, String)>>,
    /// Immediate sub-collections ("direct children") one level deeper
    pub nested_collections: Vec<String>,
    /// Annotation files available in the operations database
    pub annotation_files: Vec<AnnotationFileEntry>,
    /// Annotation groups associated with the selected sample (if any)
    pub annotation_groups: Vec<AnnotationGroupEntry>,
}

/// Gathers information about a hierarchical collection, enumerating reference (null-sample)
/// block groups, sample block groups, and immediate sub-collections.
pub fn gather_collection_explorer_data(
    conn: &GraphConnection,
    op_conn: &OperationsConnection,
    sample_name: Option<&str>,
    full_collection_name: &str,
) -> CollectionExplorerData {
    let current_collection = collection_basename(full_collection_name).to_string();
    let _parent = parent_collection(full_collection_name);

    // 2) Query block groups that have sample_name = NULL for the entire collection
    let base_bgs = BlockGroup::query(
        conn,
        "SELECT * FROM block_groups
         WHERE collection_name = ?1
           AND sample_name IS NULL",
        params![full_collection_name],
    );
    let reference_block_groups: Vec<(gen_core::HashId, String)> =
        base_bgs.iter().map(|bg| (bg.id, bg.name.clone())).collect();

    // 3) Gather all samples associated with the entire collection
    let all_blocks = Collection::get_block_groups(conn, full_collection_name);
    let mut sample_names: HashSet<String> = all_blocks
        .iter()
        .filter_map(|bg| bg.sample_name.clone())
        .collect();
    let mut collection_samples: Vec<String> = sample_names.drain().collect();
    collection_samples.sort();

    // 4) For each sample, retrieve block groups
    let mut sample_block_groups = HashMap::new();
    for sample in &collection_samples {
        let bgs = Sample::get_block_groups(conn, full_collection_name, Some(sample));
        let pairs = bgs
            .iter()
            .map(|bg| (bg.id, bg.name.clone()))
            .collect::<Vec<(gen_core::HashId, String)>>();
        sample_block_groups.insert(sample.clone(), pairs);
    }

    // 5) Direct "nested" collections: must start with "full_collection_name + /" but no further delimiter
    let direct_prefix = format!("{}{}", full_collection_name, "/");

    let sibling_candidates = Collection::query(
        conn,
        "SELECT * FROM collections
         WHERE name GLOB ?1",
        params![format!("{}*", direct_prefix)],
    );

    let mut nested_collections = Vec::new();
    for child in sibling_candidates {
        // The portion *after* "/foo/bar/"
        let remainder = &child.name[direct_prefix.len()..];
        // If there's no further slash, it's a direct child
        if !remainder.is_empty() && !remainder.contains('/') {
            nested_collections.push(remainder.to_string());
        }
    }

    let annotation_files = load_annotation_file_entries(op_conn);
    let annotation_groups = load_annotation_group_entries(conn, sample_name);

    CollectionExplorerData {
        current_collection,
        reference_block_groups,
        collection_samples,
        sample_block_groups,
        nested_collections,
        annotation_files,
        annotation_groups,
    }
}

#[derive(Debug)]
pub enum ExplorerItem {
    Collection {
        name: String,
        /// Whether this is the current collection (listed at the top), or a link to another collection
        is_current: bool,
    },
    BlockGroup {
        id: HashId,
        name: String,
    },
    Sample {
        name: String,
        expanded: bool,
    },
    Header {
        text: String,
    },
    AnnotationFile {
        id: HashId,
        display_name: String,
        file_type: FileTypes,
        active: bool,
    },
    AnnotationGroup {
        name: String,
        active: bool,
    },
}

impl ExplorerItem {
    /// Skip over headers and the top-level collection name
    pub fn is_selectable(&self) -> bool {
        match self {
            ExplorerItem::Collection { is_current, .. } => !is_current,
            ExplorerItem::BlockGroup { .. } => true,
            ExplorerItem::Sample { .. } => true,
            ExplorerItem::Header { .. } => false,
            ExplorerItem::AnnotationFile { .. } => true,
            ExplorerItem::AnnotationGroup { .. } => true,
        }
    }
}

#[derive(Debug, Default)]
pub struct CollectionExplorerState {
    pub list_state: ListState,
    pub total_items: usize,
    pub has_focus: bool,
    /// The currently selected block group
    pub selected_block_group_id: Option<HashId>,
    pub active_annotation_files: HashSet<HashId>,
    pub active_annotation_groups: HashSet<String>,
    /// Tracks which samples are expanded/collapsed
    expanded_samples: HashSet<String>,
    /// Indicates which focus zone should receive focus (if any)
    pub focus_change_requested: Option<FocusZone>,
    pub annotation_file_toggle_requested: Option<HashId>,
    pub annotation_group_toggle_requested: Option<String>,
}

impl CollectionExplorerState {
    pub fn new() -> Self {
        Self::with_selected_block_group(None)
    }

    pub fn with_selected_block_group(block_group_id: Option<HashId>) -> Self {
        Self {
            list_state: ListState::default(),
            total_items: 0,
            has_focus: false,
            selected_block_group_id: block_group_id,
            active_annotation_files: HashSet::new(),
            active_annotation_groups: HashSet::new(),
            expanded_samples: HashSet::new(),
            focus_change_requested: None,
            annotation_file_toggle_requested: None,
            annotation_group_toggle_requested: None,
        }
    }

    /// Toggle expansion state of a sample
    pub fn toggle_sample(&mut self, sample_name: &str) {
        if self.expanded_samples.contains(sample_name) {
            self.expanded_samples.remove(sample_name);
        } else {
            self.expanded_samples.insert(sample_name.to_string());
        }
    }

    /// Check if a sample is expanded
    pub fn is_sample_expanded(&self, sample_name: &str) -> bool {
        self.expanded_samples.contains(sample_name)
    }

    pub fn toggle_annotation_file(&mut self, id: HashId) {
        if self.active_annotation_files.contains(&id) {
            self.active_annotation_files.remove(&id);
        } else {
            self.active_annotation_files.insert(id);
        }
    }

    pub fn deactivate_annotation_file(&mut self, id: &HashId) {
        self.active_annotation_files.remove(id);
    }

    pub fn is_annotation_file_active(&self, id: &HashId) -> bool {
        self.active_annotation_files.contains(id)
    }

    pub fn retain_annotation_files(&mut self, entries: &[AnnotationFileEntry]) {
        let valid_ids: HashSet<HashId> =
            entries.iter().map(|entry| entry.file_addition.id).collect();
        self.active_annotation_files
            .retain(|id| valid_ids.contains(id));
    }

    pub fn toggle_annotation_group(&mut self, name: &str) {
        if self.active_annotation_groups.contains(name) {
            self.active_annotation_groups.remove(name);
        } else {
            self.active_annotation_groups.insert(name.to_string());
        }
    }

    pub fn deactivate_annotation_group(&mut self, name: &str) {
        self.active_annotation_groups.remove(name);
    }

    pub fn is_annotation_group_active(&self, name: &str) -> bool {
        self.active_annotation_groups.contains(name)
    }

    pub fn retain_annotation_groups(&mut self, entries: &[AnnotationGroupEntry]) {
        let valid: HashSet<String> = entries.iter().map(|entry| entry.name.clone()).collect();
        self.active_annotation_groups
            .retain(|name| valid.contains(name));
    }
}

#[derive(Debug)]
pub struct CollectionExplorer {
    pub data: CollectionExplorerData,
}

impl CollectionExplorer {
    pub fn new(
        conn: &GraphConnection,
        op_conn: &OperationsConnection,
        sample_name: Option<&str>,
        full_collection_name: &str,
    ) -> Self {
        let data =
            gather_collection_explorer_data(conn, op_conn, sample_name, full_collection_name);
        Self { data }
    }

    /// Refresh the explorer data from the database and return true if data changed
    pub fn refresh(
        &mut self,
        conn: &GraphConnection,
        op_conn: &OperationsConnection,
        sample_name: Option<&str>,
        full_collection_name: &str,
    ) -> bool {
        let new_data =
            gather_collection_explorer_data(conn, op_conn, sample_name, full_collection_name);
        let changed = self.data.reference_block_groups.len()
            != new_data.reference_block_groups.len()
            || self.data.sample_block_groups != new_data.sample_block_groups
            || self.data.annotation_files != new_data.annotation_files
            || self.data.annotation_groups != new_data.annotation_groups;
        self.data = new_data;
        changed
    }

    pub fn annotation_file_entry(&self, id: &HashId) -> Option<&AnnotationFileEntry> {
        self.data
            .annotation_files
            .iter()
            .find(|entry| entry.file_addition.id == *id)
    }

    /// Force the widget to reload by resetting its state
    pub fn force_reload(&self, state: &mut CollectionExplorerState) {
        state.list_state = ListState::default();
        // Find first selectable item to maintain a valid selection
        state.list_state.selected = self.find_next_selectable(state, 0);
    }

    /// Find the next selectable item after the given index, wrapping around to the start if needed
    fn find_next_selectable(
        &self,
        state: &CollectionExplorerState,
        from_idx: usize,
    ) -> Option<usize> {
        let items = self.get_display_items(state);
        // First try after the current index
        items
            .iter()
            .enumerate()
            .skip(from_idx)
            .find(|(_, item)| item.is_selectable())
            .map(|(i, _)| i)
            // If nothing found after current index, wrap around to start
            .or_else(|| {
                items
                    .iter()
                    .enumerate()
                    .take(from_idx)
                    .find(|(_, item)| item.is_selectable())
                    .map(|(i, _)| i)
            })
    }

    /// Find the previous selectable item before the given index, wrapping around to the end if needed
    fn find_prev_selectable(
        &self,
        state: &CollectionExplorerState,
        from_idx: usize,
    ) -> Option<usize> {
        let items = self.get_display_items(state);
        // First try before the current index
        items
            .iter()
            .enumerate()
            .take(from_idx)
            .rev()
            .find(|(_, item)| item.is_selectable())
            .map(|(i, _)| i)
            // If nothing found before current index, wrap around to end
            .or_else(|| {
                items
                    .iter()
                    .enumerate()
                    .skip(from_idx)
                    .rev()
                    .find(|(_, item)| item.is_selectable())
                    .map(|(i, _)| i)
            })
    }

    pub fn next(&self, state: &mut CollectionExplorerState) {
        let items = self.get_display_items(state);
        if items.is_empty() {
            return;
        }

        let current_idx = state.list_state.selected.unwrap_or(0);
        state.list_state.selected = self.find_next_selectable(state, current_idx + 1);
    }

    pub fn previous(&self, state: &mut CollectionExplorerState) {
        let items = self.get_display_items(state);
        if items.is_empty() {
            return;
        }

        let current_idx = state.list_state.selected.unwrap_or(0);
        state.list_state.selected = self.find_prev_selectable(state, current_idx);
    }

    pub fn handle_input(&self, state: &mut CollectionExplorerState, key: KeyEvent) {
        match key.code {
            KeyCode::Up => self.previous(state),
            KeyCode::Down => self.next(state),
            KeyCode::Enter | KeyCode::Char(' ') => {
                if let Some(selected_idx) = state.list_state.selected {
                    let items = self.get_display_items(state);
                    match &items[selected_idx] {
                        ExplorerItem::BlockGroup { id, .. } => {
                            state.selected_block_group_id = Some(*id);
                            state.focus_change_requested = Some(FocusZone::Canvas);
                        }
                        ExplorerItem::Sample { .. } => {
                            self.toggle_sample_expansion(state);
                        }
                        ExplorerItem::AnnotationFile { id, .. } => {
                            state.toggle_annotation_file(*id);
                            state.annotation_file_toggle_requested = Some(*id);
                        }
                        ExplorerItem::AnnotationGroup { name, .. } => {
                            state.toggle_annotation_group(name);
                            state.annotation_group_toggle_requested = Some(name.clone());
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    pub fn get_status_line() -> String {
        "*▼ ▲* navigate | *return* select/toggle".to_string()
    }

    /// Get all items to display, taking into account the current state
    fn get_display_items(&self, state: &CollectionExplorerState) -> Vec<ExplorerItem> {
        let mut items = Vec::new();

        // Current collection name
        items.push(ExplorerItem::Collection {
            name: self.data.current_collection.clone(),
            is_current: true,
        });

        // Blank line
        items.push(ExplorerItem::Header {
            text: String::new(),
        });

        // Reference graphs section
        items.push(ExplorerItem::Header {
            text: "Reference graphs:".to_string(),
        });

        // Reference block groups
        for (id, name) in &self.data.reference_block_groups {
            items.push(ExplorerItem::BlockGroup {
                id: *id,
                name: name.clone(),
            });
        }

        // Blank line
        items.push(ExplorerItem::Header {
            text: String::new(),
        });

        // Samples section
        items.push(ExplorerItem::Header {
            text: "Sample graphs:".to_string(),
        });

        // Samples and their block groups
        for sample in &self.data.collection_samples {
            items.push(ExplorerItem::Sample {
                name: sample.clone(),
                expanded: state.is_sample_expanded(sample),
            });

            if state.is_sample_expanded(sample)
                && let Some(block_groups) = self.data.sample_block_groups.get(sample)
            {
                for (id, name) in block_groups {
                    items.push(ExplorerItem::BlockGroup {
                        id: *id,
                        name: name.clone(),
                    });
                }
            }
        }

        // Blank line
        items.push(ExplorerItem::Header {
            text: String::new(),
        });

        // Nested collections section
        items.push(ExplorerItem::Header {
            text: "Nested Collections:".to_string(),
        });

        // Nested collections
        for collection in &self.data.nested_collections {
            items.push(ExplorerItem::Collection {
                name: collection.clone(),
                is_current: false,
            });
        }

        // Blank line
        items.push(ExplorerItem::Header {
            text: String::new(),
        });

        // Annotation files section
        items.push(ExplorerItem::Header {
            text: "Annotation files:".to_string(),
        });

        for entry in &self.data.annotation_files {
            items.push(ExplorerItem::AnnotationFile {
                id: entry.file_addition.id,
                display_name: entry.display_name.clone(),
                file_type: entry.file_addition.file_type,
                active: state.is_annotation_file_active(&entry.file_addition.id),
            });
        }

        // Blank line
        items.push(ExplorerItem::Header {
            text: String::new(),
        });

        // Annotation groups section
        items.push(ExplorerItem::Header {
            text: "Annotation groups:".to_string(),
        });

        for entry in &self.data.annotation_groups {
            items.push(ExplorerItem::AnnotationGroup {
                name: entry.name.clone(),
                active: state.is_annotation_group_active(&entry.name),
            });
        }

        items
    }

    pub fn toggle_sample_expansion(&self, state: &mut CollectionExplorerState) {
        if let Some(selected_idx) = state.list_state.selected {
            let items = self.get_display_items(state);
            if let Some(ExplorerItem::Sample { name, .. }) = items.get(selected_idx) {
                state.toggle_sample(name);
            }
        }
    }
}

impl StatefulWidget for &CollectionExplorer {
    type State = CollectionExplorerState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let items = self.get_display_items(state);
        let mut display_items = Vec::new();

        // Convert ExplorerItems to display items
        for item in &items {
            let paragraph = match item {
                ExplorerItem::Collection { name, is_current } => {
                    if *is_current {
                        // This is the current collection header
                        Paragraph::new(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(
                                "Collection:",
                                Style::default().add_modifier(Modifier::UNDERLINED),
                            ),
                            Span::raw(format!(" {}", name)),
                        ]))
                        .wrap(Wrap { trim: false })
                    } else {
                        // This is a link to another collection
                        Paragraph::new(Line::from(vec![Span::raw(format!("  • {}", name))]))
                            .wrap(Wrap { trim: false })
                    }
                }
                ExplorerItem::BlockGroup { id, name, .. } => {
                    // Check if this block group is one of the sample_name = NULL reference block groups
                    // This influences the indentation
                    let is_reference = self
                        .data
                        .reference_block_groups
                        .iter()
                        .any(|(ref_id, _)| ref_id == id);

                    if is_reference {
                        Paragraph::new(Line::from(vec![Span::raw(format!("   • {}", name))]))
                            .wrap(Wrap { trim: false })
                    } else {
                        Paragraph::new(Line::from(vec![Span::raw(format!("     • {}", name))]))
                            .wrap(Wrap { trim: false })
                    }
                }
                ExplorerItem::Sample { name, expanded } => Paragraph::new(Line::from(vec![
                    Span::raw(if *expanded { "   ▼ " } else { "   ▶ " }),
                    Span::styled(name, Style::default()),
                ]))
                .wrap(Wrap { trim: false }),
                ExplorerItem::Header { text } => Paragraph::new(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(text, Style::default().add_modifier(Modifier::UNDERLINED)),
                ]))
                .wrap(Wrap { trim: false }),
                ExplorerItem::AnnotationFile {
                    display_name,
                    file_type,
                    active,
                    ..
                } => {
                    let marker = if *active { "[x]" } else { "[ ]" };
                    let suffix = FileTypes::suffix(*file_type);
                    Paragraph::new(Line::from(vec![
                        Span::raw(format!("   {marker} {display_name}")),
                        Span::styled(
                            format!(" ({suffix})"),
                            Style::default().fg(get_theme_color("text_muted").unwrap()),
                        ),
                    ]))
                    .wrap(Wrap { trim: false })
                }
                ExplorerItem::AnnotationGroup { name, active } => {
                    let marker = if *active { "[x]" } else { "[ ]" };
                    Paragraph::new(Line::from(vec![Span::raw(format!("   {marker} {name}"))]))
                        .wrap(Wrap { trim: false })
                }
            };

            display_items.push(paragraph);
        }

        // Store total items
        let total_items = display_items.len();
        let has_focus = state.has_focus;

        // Create and render the list
        let builder = ListBuilder::new(move |context| {
            let item = display_items[context.index].clone();
            let available_width = context.cross_axis_size;
            let item_height = item.line_count(available_width) as u16;

            if context.is_selected {
                let style = if has_focus {
                    Style::default()
                        .fg(get_theme_color("text_muted").unwrap())
                        .bg(get_theme_color("highlight").unwrap())
                } else {
                    Style::default()
                        .fg(get_theme_color("text").unwrap())
                        .bg(get_theme_color("highlight_muted").unwrap())
                };
                (item.style(style), item_height)
            } else {
                (item, item_height)
            }
        });

        let list = ListView::new(builder, total_items).block(Block::default());

        state.total_items = total_items;

        // Ensure selection is valid for the current items
        if state.list_state.selected.is_none() || state.list_state.selected.unwrap() >= total_items
        {
            // Selection is invalid or missing - try to find a valid one
            state.list_state.selected = if let Some(ref block_group_id) =
                state.selected_block_group_id
            {
                // Try to find the selected block group in the current items
                self.get_display_items(state).iter()
                    .enumerate()
                    .find(|(_, item)| matches!(item, ExplorerItem::BlockGroup { id, .. } if id == block_group_id))
                    .map(|(i, _)| i)
                    .or_else(|| self.find_next_selectable(state, 0))
            } else {
                // No block group selected, just find the next selectable item
                self.find_next_selectable(state, 0)
            };
        }

        list.render(area, buf, &mut state.list_state);
    }
}

#[cfg(test)]
mod tests {
    use gen_models::{block_group::BlockGroup, sample::Sample};

    use super::*;
    use crate::test_helpers::setup_gen;

    /// For these tests we create an in-memory database, run minimal schema
    /// creation, and insert data to test gather_collection_explorer_data.
    #[test]
    fn test_gather_collection_explorer_data() {
        let context = setup_gen();
        let conn = context.graph().conn();

        // Create collections with hierarchical paths
        Collection::create(conn, "/foo/bar");
        Collection::create(conn, "/foo/bar/a");
        Collection::create(conn, "/foo/bar/a/b");
        Collection::create(conn, "/foo/bar2");
        Collection::create(conn, "/foo/baz");

        // Create samples
        let sample_alpha = Sample::get_or_create(conn, "SampleAlpha");
        let sample_beta = Sample::get_or_create(conn, "SampleBeta");

        // Create block groups: some with sample = null (reference), some with a sample
        BlockGroup::create(conn, "/foo/bar", None, "BG_ReferenceA");
        BlockGroup::create(conn, "/foo/bar", None, "BG_ReferenceB");
        BlockGroup::create(conn, "/foo/bar", Some(&sample_alpha.name), "BG_Alpha1");
        BlockGroup::create(conn, "/foo/bar", Some(&sample_beta.name), "BG_Beta1");

        // Call the function under test—notice we pass the full path
        let op_conn = context.operations().conn();
        let explorer_data = gather_collection_explorer_data(conn, op_conn, None, "/foo/bar");

        // Verify results
        // (A) The final path component is "bar"
        assert_eq!(explorer_data.current_collection, "bar");

        // (B) Reference block groups (sample_name IS NULL)
        let base_names: Vec<_> = explorer_data
            .reference_block_groups
            .iter()
            .map(|(_, name)| name.clone())
            .collect();
        assert_eq!(base_names.len(), 2);
        assert!(base_names.contains(&"BG_ReferenceA".to_string()));
        assert!(base_names.contains(&"BG_ReferenceB".to_string()));

        // (C) Collection samples
        // We expect SampleAlpha and SampleBeta
        assert_eq!(explorer_data.collection_samples.len(), 2);
        assert!(
            explorer_data
                .collection_samples
                .contains(&"SampleAlpha".to_string())
        );
        assert!(
            explorer_data
                .collection_samples
                .contains(&"SampleBeta".to_string())
        );

        // (D) Sample block groups
        // "SampleAlpha"
        let alpha_bg = explorer_data
            .sample_block_groups
            .get("SampleAlpha")
            .unwrap();
        let alpha_bg_names: Vec<_> = alpha_bg.iter().map(|(_, n)| n.clone()).collect();
        assert_eq!(alpha_bg_names, vec!["BG_Alpha1".to_string()]);
        // "SampleBeta"
        let beta_bg = explorer_data.sample_block_groups.get("SampleBeta").unwrap();
        let beta_bg_names: Vec<_> = beta_bg.iter().map(|(_, n)| n.clone()).collect();
        assert_eq!(beta_bg_names, vec!["BG_Beta1".to_string()]);

        // (E) Nested collections: we only want the direct child after "/foo/bar/"
        // e.g. "/foo/bar/a" => child is "a"
        // "/foo/bar/a/b" is not a direct child, it's an extra level
        // "/foo/bar2" doesn't match the prefix "/foo/bar/"
        // ... So only "a" is a direct nested collection
        assert_eq!(explorer_data.nested_collections, vec!["a".to_string()]);
    }

    #[test]
    fn test_trailing_delimiter_behavior() {
        // This verifies how we handle trailing hierarchical delimiters
        assert_eq!(normalize_collection_name("/foo/bar/"), "/foo/bar");
        assert_eq!(normalize_collection_name("////"), "/");
        assert_eq!(normalize_collection_name("/"), "/");

        assert_eq!(collection_basename("/foo/bar/"), "bar");
        assert_eq!(collection_basename("////"), "/");
        assert_eq!(collection_basename("/"), "/");

        assert_eq!(parent_collection("/foo/bar/"), "/foo");
        // parent of /foo => /
        assert_eq!(parent_collection("/foo/"), "/");
        // parent of / => /
        assert_eq!(parent_collection("////"), "/");
        // parent of a single "segment" => "."
        assert_eq!(parent_collection("bar"), ".");
    }
}
