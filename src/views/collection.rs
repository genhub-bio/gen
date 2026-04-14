use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crossterm::event::{KeyCode, KeyEvent};
use gen_core::HashId;
use gen_models::{
    collection::Collection,
    db::{GraphConnection, OperationsConnection},
    file_types::FileTypes,
    sample::Sample,
    sample_lineage::SampleLineage,
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
        samples::{SampleTree, SampleTreeEntry},
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
    /// The block groups in the *entire* collection that have is_reference set
    pub reference_block_groups: Vec<(gen_core::HashId, String)>,
    /// The samples in the entire collection
    pub collection_samples: Vec<String>,
    /// Root samples for the lineage tree.
    pub sample_roots: Vec<String>,
    /// Direct child samples for each sample.
    pub sample_children: HashMap<String, Vec<String>>,
    /// Direct parent samples for each sample.
    pub sample_parents: HashMap<String, Vec<String>>,
    /// The block groups for each sample
    pub sample_block_groups: HashMap<String, Vec<(gen_core::HashId, String)>>,
    /// Immediate sub-collections ("direct children") one level deeper
    pub nested_collections: Vec<String>,
    /// Annotation files available in the operations database
    pub annotation_files: Vec<AnnotationFileEntry>,
    /// Annotation groups associated with the selected sample (if any)
    pub annotation_groups: Vec<AnnotationGroupEntry>,
}

/// Gathers information about a hierarchical collection, enumerating reference
/// block groups, sample block groups, and immediate sub-collections.
pub fn gather_collection_explorer_data(
    conn: &GraphConnection,
    op_conn: &OperationsConnection,
    sample_name: Option<&str>,
    full_collection_name: &str,
) -> CollectionExplorerData {
    let current_collection = collection_basename(full_collection_name).to_string();
    let _parent = parent_collection(full_collection_name);

    let reference_block_groups: Vec<(HashId, String)> = vec![];

    // 3) Gather all samples associated with the entire collection
    let all_blocks = Collection::get_block_groups(conn, full_collection_name);
    let mut sample_names: HashSet<String> =
        all_blocks.iter().map(|bg| bg.sample_name.clone()).collect();
    let mut collection_samples: Vec<String> = sample_names.drain().collect();
    collection_samples.sort();

    let collection_sample_set: HashSet<String> = collection_samples.iter().cloned().collect();
    let mut sample_children = HashMap::new();
    let mut sample_parents = HashMap::new();
    let mut sample_roots = Vec::new();
    for sample in &collection_samples {
        let mut parents = SampleLineage::get_parents(conn, sample)
            .into_iter()
            .filter(|parent| collection_sample_set.contains(parent))
            .collect::<Vec<_>>();
        parents.sort();
        parents.dedup();

        let mut children = SampleLineage::get_children(conn, sample)
            .into_iter()
            .filter(|child| collection_sample_set.contains(child))
            .collect::<Vec<_>>();
        children.sort();
        children.dedup();

        if parents.is_empty() {
            sample_roots.push(sample.clone());
        }

        sample_parents.insert(sample.clone(), parents);
        sample_children.insert(sample.clone(), children);
    }
    sample_roots.sort();

    // 4) For each sample, retrieve block groups
    let mut sample_block_groups = HashMap::new();
    for sample in &collection_samples {
        let bgs = Sample::get_block_groups(conn, full_collection_name, sample);
        let pairs = bgs
            .iter()
            .map(|bg| (bg.id, bg.name.clone()))
            .collect::<Vec<(HashId, String)>>();
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
    let annotation_groups = sample_name
        .map(|sample_name| load_annotation_group_entries(conn, sample_name))
        .unwrap_or_default();

    CollectionExplorerData {
        current_collection,
        reference_block_groups,
        collection_samples,
        sample_roots,
        sample_children,
        sample_parents,
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
        depth: usize,
    },
    Sample {
        name: String,
        expanded: bool,
        depth: usize,
        has_children: bool,
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
    /// Explicit sample expansion overrides, separate from the default auto-open tree.
    sample_expansion_overrides: HashMap<String, bool>,
    /// Indicates which focus zone should receive focus (if any)
    pub focus_change_requested: Option<FocusZone>,
    /// Active annotation files
    pub active_annotation_files: HashSet<HashId>,
    /// Active annotation groups
    pub active_annotation_groups: HashSet<String>,
    /// Pending annotation file toggle request
    pub annotation_file_toggle_requested: Option<HashId>,
    /// Pending annotation group toggle request
    pub annotation_group_toggle_requested: Option<String>,
    /// Horizontal scroll offset for the sample lineage tree.
    pub sample_tree_scroll: u16,
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
            sample_expansion_overrides: HashMap::new(),
            focus_change_requested: None,
            active_annotation_files: HashSet::new(),
            active_annotation_groups: HashSet::new(),
            annotation_file_toggle_requested: None,
            annotation_group_toggle_requested: None,
            sample_tree_scroll: 0,
        }
    }

    /// Toggle expansion state of a sample
    pub fn toggle_sample(&mut self, sample_name: &str, currently_expanded: bool) {
        let next = !currently_expanded;
        self.sample_expansion_overrides
            .insert(sample_name.to_string(), next);
    }

    /// Check if a sample is expanded
    pub fn is_sample_expanded(&self, sample_name: &str, default_expanded: bool) -> bool {
        self.sample_expansion_overrides
            .get(sample_name)
            .copied()
            .unwrap_or(default_expanded)
    }

    /// Force a sample to a specific expansion state.
    pub fn set_sample_expanded(&mut self, sample_name: &str, expanded: bool) {
        self.sample_expansion_overrides
            .insert(sample_name.to_string(), expanded);
    }

    /// Toggle an annotation file on/off
    pub fn toggle_annotation_file(&mut self, id: HashId) {
        if self.active_annotation_files.contains(&id) {
            self.active_annotation_files.remove(&id);
        } else {
            self.active_annotation_files.insert(id);
        }
    }

    /// Deactivate an annotation file
    pub fn deactivate_annotation_file(&mut self, id: &HashId) {
        self.active_annotation_files.remove(id);
    }

    /// Check if an annotation file is active
    pub fn is_annotation_file_active(&self, id: &HashId) -> bool {
        self.active_annotation_files.contains(id)
    }

    /// Retain only annotation files that exist in the provided list
    pub fn retain_annotation_files(
        &mut self,
        entries: &[crate::views::annotation_files::AnnotationFileEntry],
    ) {
        let valid_ids: HashSet<HashId> =
            entries.iter().map(|entry| entry.file_addition.id).collect();
        self.active_annotation_files
            .retain(|id| valid_ids.contains(id));
    }

    /// Toggle an annotation group on/off
    pub fn toggle_annotation_group(&mut self, name: &str) {
        if self.active_annotation_groups.contains(name) {
            self.active_annotation_groups.remove(name);
        } else {
            self.active_annotation_groups.insert(name.to_string());
        }
    }

    /// Deactivate an annotation group
    pub fn deactivate_annotation_group(&mut self, name: &str) {
        self.active_annotation_groups.remove(name);
    }

    /// Check if an annotation group is active
    pub fn is_annotation_group_active(&self, name: &str) -> bool {
        self.active_annotation_groups.contains(name)
    }

    /// Retain only annotation groups that exist in the provided list
    pub fn retain_annotation_groups(
        &mut self,
        entries: &[crate::views::annotation_groups::AnnotationGroupEntry],
    ) {
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
        op_conn: &gen_models::db::OperationsConnection,
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
        op_conn: &gen_models::db::OperationsConnection,
        sample_name: Option<&str>,
        full_collection_name: &str,
    ) -> bool {
        let new_data =
            gather_collection_explorer_data(conn, op_conn, sample_name, full_collection_name);
        let changed = self.data.reference_block_groups.len()
            != new_data.reference_block_groups.len()
            || self.data.sample_block_groups != new_data.sample_block_groups
            || self.data.sample_roots != new_data.sample_roots
            || self.data.sample_children != new_data.sample_children
            || self.data.sample_parents != new_data.sample_parents;
        self.data = new_data;
        changed
    }

    /// Get annotation file entry by ID
    pub fn annotation_file_entry(
        &self,
        id: &HashId,
    ) -> Option<&crate::views::annotation_files::AnnotationFileEntry> {
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
            KeyCode::Left => {
                if let Some(selected_idx) = state.list_state.selected {
                    let items = self.get_display_items(state);
                    if let Some(ExplorerItem::Sample { name, expanded, .. }) =
                        items.get(selected_idx)
                    {
                        if *expanded {
                            state.toggle_sample(name, *expanded);
                        } else if let Some(parent_name) = self
                            .data
                            .sample_parents
                            .get(name)
                            .and_then(|parents| parents.first())
                        {
                            state.list_state.selected =
                                items.iter().enumerate().find_map(|(idx, item)| match item {
                                    ExplorerItem::Sample {
                                        name: item_name, ..
                                    } if item_name == parent_name => Some(idx),
                                    _ => None,
                                });
                        }
                    }
                }
            }
            KeyCode::Right => {
                if let Some(selected_idx) = state.list_state.selected {
                    let items = self.get_display_items(state);
                    if let Some(ExplorerItem::Sample {
                        name,
                        expanded,
                        has_children,
                        ..
                    }) = items.get(selected_idx)
                        && *has_children
                        && !*expanded
                    {
                        state.toggle_sample(name, *expanded);
                    }
                }
            }
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
        "*▼ ▲* navigate | *return/space* toggle | *←/→* tree".to_string()
    }

    /// Get all items to display, taking into account the current state
    fn get_display_items(&self, state: &CollectionExplorerState) -> Vec<ExplorerItem> {
        let mut items = Vec::new();
        let sample_tree = SampleTree::new(&self.data);
        let sample_entries = sample_tree.build_entries(state);

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
                depth: 0,
            });
        }

        // Blank line
        items.push(ExplorerItem::Header {
            text: String::new(),
        });

        // Samples section
        items.push(ExplorerItem::Header {
            text: "Sample lineages:".to_string(),
        });

        for entry in sample_entries {
            match entry {
                SampleTreeEntry::Sample {
                    name,
                    expanded,
                    depth,
                    has_children,
                } => items.push(ExplorerItem::Sample {
                    name,
                    expanded,
                    depth,
                    has_children,
                }),
                SampleTreeEntry::BlockGroup { id, name, depth } => {
                    items.push(ExplorerItem::BlockGroup { id, name, depth })
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
            text: "Annotation Files:".to_string(),
        });

        // Annotation files
        for entry in &self.data.annotation_files {
            // Extract just the filename from the path
            let display_name = std::path::Path::new(&entry.file_addition.file_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&entry.file_addition.file_path)
                .to_string();

            items.push(ExplorerItem::AnnotationFile {
                id: entry.file_addition.id,
                display_name,
                file_type: entry.file_addition.file_type,
                active: state.is_annotation_file_active(&entry.file_addition.id),
            });
        }

        // Annotation groups section (if there are any)
        if !self.data.annotation_groups.is_empty() {
            // Blank line
            items.push(ExplorerItem::Header {
                text: String::new(),
            });

            items.push(ExplorerItem::Header {
                text: "Annotation Groups:".to_string(),
            });

            // Annotation groups
            for entry in &self.data.annotation_groups {
                items.push(ExplorerItem::AnnotationGroup {
                    name: entry.name.clone(),
                    active: state.is_annotation_group_active(&entry.name),
                });
            }
        }

        items
    }

    pub fn toggle_sample_expansion(&self, state: &mut CollectionExplorerState) {
        if let Some(selected_idx) = state.list_state.selected {
            let items = self.get_display_items(state);
            if let Some(ExplorerItem::Sample { name, expanded, .. }) = items.get(selected_idx) {
                state.toggle_sample(name, *expanded);
            }
        }
    }
}

impl StatefulWidget for &CollectionExplorer {
    type State = CollectionExplorerState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let items = self.get_display_items(state);
        state.sample_tree_scroll = state
            .list_state
            .selected
            .and_then(|idx| items.get(idx))
            .map(|item| match item {
                ExplorerItem::Sample { depth, .. } | ExplorerItem::BlockGroup { depth, .. } => {
                    (*depth as u16).saturating_mul(2)
                }
                _ => 0,
            })
            .unwrap_or(0);
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
                ExplorerItem::BlockGroup { id, name, depth } => {
                    // Check if this block group is one of the sample_name = NULL reference block groups
                    // This influences the indentation
                    let is_reference = self
                        .data
                        .reference_block_groups
                        .iter()
                        .any(|(ref_id, _)| ref_id == id);
                    let scroll = if *depth > 0 {
                        state.sample_tree_scroll
                    } else {
                        0
                    };

                    if is_reference {
                        Paragraph::new(Line::from(vec![Span::raw(format!("  • {}", name))]))
                            .wrap(Wrap { trim: false })
                    } else {
                        Paragraph::new(Line::from(vec![Span::raw(format!(
                            "  {}• {}",
                            "  ".repeat(*depth),
                            name
                        ))]))
                        .scroll((0, scroll))
                        .wrap(Wrap { trim: false })
                    }
                }
                ExplorerItem::Sample {
                    name,
                    expanded,
                    depth,
                    has_children,
                } => {
                    let marker = if *has_children {
                        if *expanded { "▼" } else { "▶" }
                    } else {
                        "•"
                    };
                    Paragraph::new(Line::from(vec![Span::raw(format!(
                        "  {}{} {}",
                        "  ".repeat(*depth),
                        marker,
                        name
                    ))]))
                    .scroll((0, state.sample_tree_scroll))
                    .wrap(Wrap { trim: false })
                }
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
                    let checkbox = if *active { "[✓]" } else { "[ ]" };
                    let type_str = match file_type {
                        FileTypes::Gff3 => "gff3",
                        FileTypes::Bed => "bed",
                        _ => "other",
                    };
                    Paragraph::new(Line::from(vec![
                        Span::raw(format!("     {} ", checkbox)),
                        Span::styled(display_name, Style::default()),
                        Span::raw(format!(" ({})", type_str)),
                    ]))
                    .wrap(Wrap { trim: false })
                }
                ExplorerItem::AnnotationGroup { name, active } => {
                    let checkbox = if *active { "[✓]" } else { "[ ]" };
                    Paragraph::new(Line::from(vec![
                        Span::raw(format!("     {} ", checkbox)),
                        Span::styled(name, Style::default()),
                    ]))
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
        let sample_reference = Sample::get_or_create(conn, Sample::DEFAULT_NAME);
        let sample_alpha = Sample::get_or_create(conn, "SampleAlpha");
        let sample_beta = Sample::get_or_create(conn, "SampleBeta");

        // Create block groups for three explicit samples
        BlockGroup::create(
            conn,
            gen_models::block_group::NewBlockGroup {
                collection_name: "/foo/bar",
                sample_name: &sample_reference.name,
                name: "BG_ReferenceA",
                ..Default::default()
            },
        );
        BlockGroup::create(
            conn,
            gen_models::block_group::NewBlockGroup {
                collection_name: "/foo/bar",
                sample_name: &sample_reference.name,
                name: "BG_ReferenceB",
                ..Default::default()
            },
        );
        BlockGroup::create(
            conn,
            gen_models::block_group::NewBlockGroup {
                collection_name: "/foo/bar",
                sample_name: &sample_alpha.name,
                name: "BG_Alpha1",
                ..Default::default()
            },
        );
        BlockGroup::create(
            conn,
            gen_models::block_group::NewBlockGroup {
                collection_name: "/foo/bar",
                sample_name: &sample_beta.name,
                name: "BG_Beta1",
                ..Default::default()
            },
        );

        // Call the function under test—notice we pass the full path
        let op_conn = context.operations().conn();
        let explorer_data = gather_collection_explorer_data(conn, op_conn, None, "/foo/bar");

        // Verify results
        // (A) The final path component is "bar"
        assert_eq!(explorer_data.current_collection, "bar");

        // (B) There are no special reference block groups now
        assert!(explorer_data.reference_block_groups.is_empty());

        // (C) Collection samples
        // We expect reference, SampleAlpha, and SampleBeta
        assert_eq!(explorer_data.collection_samples.len(), 3);
        assert!(
            explorer_data
                .collection_samples
                .contains(&Sample::DEFAULT_NAME.to_string())
        );
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
        let reference_bg = explorer_data
            .sample_block_groups
            .get(Sample::DEFAULT_NAME)
            .unwrap();
        let reference_bg_names: Vec<_> = reference_bg.iter().map(|(_, n)| n.clone()).collect();
        assert_eq!(
            reference_bg_names,
            vec!["BG_ReferenceA".to_string(), "BG_ReferenceB".to_string()]
        );

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
