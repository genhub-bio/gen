//! The pages, overlays and PNG rendering behind the notebook and R widgets, free of any
//! language-binding types. A [`GraphSession`] holds one or more [`GraphPage`]s, each a
//! [`GenGraphController`] plus the annotation tracks and highlights a plot layers on it. The
//! Python and R bindings wrap a session, so both draw the same pixels and both reach the
//! controller through the same methods.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use gen_core::{HashId, Workspace, is_terminal};
use gen_models::{block_group::BlockGroup, db::GraphConnection, port_crawl::LocateError};
use gen_tui::{
    LayoutMode, LineStyle, RasterConfig, highlight_style::PathStyle, layout::VisualDetail,
    theme::current_theme,
};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Clear, Widget},
};
use serde::Serialize;
use thiserror::Error;

use crate::views::{
    annotation_files::{AnnotationFileEntry, load_annotation_file_entries},
    annotation_track::AnnotationSpan,
    annotations::{AnnotationFileTrackRequest, load_annotation_file_track},
    gen_graph_controller::{ClickOutcome, GenGraphController},
    graph_database::GraphDatabase,
    graph_overlay::{
        GraphOverlay, OverlayContent, OverlaySource, PathMembership, remove_path_overlay,
        set_path_overlay,
    },
    path_scale::ScalePlacement,
    png_export::{MAX_SCALE, write_cropped_png, write_uncropped_png},
};

/// Why a session operation failed, split the way bindings report it: a bad argument is a
/// value error, anything else a runtime error.
#[derive(Debug, Error)]
pub enum GraphSessionError {
    #[error("{0}")]
    InvalidArgument(String),
    #[error("{0}")]
    Operation(String),
}

impl GraphSessionError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    pub fn operation(message: impl Into<String>) -> Self {
        Self::Operation(message.into())
    }
}

pub fn operation_error(error: impl ToString) -> GraphSessionError {
    GraphSessionError::operation(error.to_string())
}

const HIGHLIGHT_ACCENT_SLOT: usize = 0x0A;
const FRAME_HEADER_HEIGHT: u16 = 1;

pub fn frame_areas(cols: u16, rows: u16) -> (Rect, Rect) {
    (
        Rect::new(0, 0, cols, FRAME_HEADER_HEIGHT.min(rows)),
        Rect::new(
            0,
            FRAME_HEADER_HEIGHT,
            cols,
            rows.saturating_sub(FRAME_HEADER_HEIGHT),
        ),
    )
}

pub fn workspace_for_connection(conn: &GraphConnection) -> Result<Workspace, GraphSessionError> {
    let database_path = conn
        .path()
        .map(PathBuf::from)
        .ok_or_else(|| GraphSessionError::operation("graph DB has no file path"))?;
    let base_dir = database_path
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| GraphSessionError::operation("graph DB path has no workspace parent"))?;
    Ok(Workspace::new(base_dir))
}

/// Parse a CSS hex colour string like `"#rrggbb"` into a ratatui `Color`.
pub fn parse_hex_color(hex: &str) -> Result<ratatui::style::Color, GraphSessionError> {
    use ratatui::style::Color;
    if hex.starts_with('#') && hex.len() == 7 {
        let r = u8::from_str_radix(&hex[1..3], 16)
            .map_err(|_| GraphSessionError::invalid("bad color"))?;
        let g = u8::from_str_radix(&hex[3..5], 16)
            .map_err(|_| GraphSessionError::invalid("bad color"))?;
        let b = u8::from_str_radix(&hex[5..7], 16)
            .map_err(|_| GraphSessionError::invalid("bad color"))?;
        Ok(Color::Rgb(r, g, b))
    } else {
        Err(GraphSessionError::invalid(format!(
            "invalid colour {hex:?}; expected a CSS hex string like \"#ff4444\""
        )))
    }
}

/// Sort key ordering annotation spans longest-first, so shorter (inner) spans paint on top.
pub fn sort_key_longest_first(span: &AnnotationSpan) -> i64 {
    -span
        .segments
        .iter()
        .map(|segment| segment.range.end - segment.range.start)
        .sum::<i64>()
}

/// One sequence graph "page" of a `PyGraphController`. A plain `GraphWidget` has exactly one
/// page; a `Sample`-backed widget pages through several.
///
/// The graph, its view, dimming, overlays and annotation groups all live in a
/// [`GenGraphController`], the same one the terminal viewers draw, so a notebook widget loads
/// lazily, steps through doors and reloads annotation groups per batch exactly as they do. The
/// page adds only what is particular to notebooks: the Python-side colors, file tracks and
/// annotation listing.
///
/// # Thread safety
///
/// ipykernel 6+ runs cell code in a thread-pool executor while anywidget comm/observe callbacks
/// fire on the asyncio ioloop, so a page is created on one thread and used from another. The
/// controller owns its database handle rather than borrowing a connection, which keeps it
/// `Send + Sync`.
#[derive(Clone)]
pub struct GraphPage {
    pub name: String,
    pub controller: GenGraphController,
    /// Set once Python has set up the annotation groups (with or without colors). Survives
    /// cloning so that cell-display clones do not set them up again.
    pub annotation_groups_loaded: bool,
    pub annotation_files: Vec<FileTrack>,
}

#[derive(Clone)]
pub struct FileTrack {
    entry: AnnotationFileEntry,
    shown: bool,
    index_available: bool,
    loaded_window: Option<(i64, i64)>,
}

/// How `plot()` loads and lays out a page.
#[derive(Clone, Copy, Debug)]
pub struct PlotOptions {
    /// Keep pruned/retired edit-site edges in the graph, dimmed, instead of removing them.
    pub show_history: bool,
    /// Place the block group's current path on one straight row.
    pub align_path: bool,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            show_history: false,
            align_path: true,
        }
    }
}

/// The information needed to lazily build a `GraphPage` on first visit.
#[derive(Clone)]
pub struct PageRef {
    name: String,
    database: GraphDatabase,
    block_group_id: HashId,
    /// Mirrors `plot(show_history=..., align_path=...)` - see `GraphPage::new`.
    options: PlotOptions,
}

/// One page of a `PyGraphController`: either already loaded, or pending lazy
/// construction the first time it becomes the active page.
#[derive(Clone)]
pub enum Page {
    Loaded(Box<GraphPage>),
    Pending(Box<PageRef>),
}

impl Page {
    pub fn name(&self) -> &str {
        match self {
            Page::Loaded(page) => &page.name,
            Page::Pending(page_ref) => &page_ref.name,
        }
    }
}

impl GraphPage {
    /// Open `block_group_id` in a controller, its graph seeded and grown lazily as renders
    /// crawl it. `show_history=False` leaves pruned/retired edit-site edges out of the crawl
    /// entirely; `show_history=True` loads them and dims them instead. `align_path=True`
    /// aligns the current path to y = 0, with variants placed around that axis.
    pub fn new(
        name: String,
        database: GraphDatabase,
        block_group_id: HashId,
        options: PlotOptions,
    ) -> Result<Self, GraphSessionError> {
        let mut controller = GenGraphController::new(database, None)
            .with_pruned_history(!options.show_history)
            .with_aligned_current_path(options.align_path);
        controller
            .open_block_group(&block_group_id)
            .map_err(operation_error)?;
        Ok(Self {
            name,
            controller,
            annotation_groups_loaded: false,
            annotation_files: Vec::new(),
        })
    }

    pub fn block_group_id(&self) -> Option<HashId> {
        self.controller
            .block_group()
            .map(|block_group| block_group.id)
    }

    /// Draw the graph, its annotation flags into `graph_area`.
    pub fn render_into(&mut self, buf: &mut Buffer, graph_area: Rect) {
        self.draw_graph(buf, graph_area);
        // Annotation files are read for the nodes already loaded, so the first load has to
        // follow a first draw; a file loaded against an empty graph would find nothing.
        let mut redraw = false;
        if self.annotation_files.is_empty() {
            let mut database = self.controller.database_mut().clone();
            if let Ok(conn) = database.connection() {
                self.show_all_annotation_files(conn);
            }
            redraw = self.annotation_files.iter().any(|file| file.shown);
        }
        redraw |= self.refresh_annotation_files_for_viewport();
        if redraw {
            Clear.render(graph_area, buf);
            self.draw_graph(buf, graph_area);
        }
    }

    pub fn draw_graph(&mut self, buf: &mut Buffer, graph_area: Rect) {
        self.controller
            .render_settled(buf, graph_area, Style::default());
    }

    pub fn resolve_color(&mut self, color: Option<&str>) -> Result<Color, GraphSessionError> {
        match color {
            None => Ok(self.controller.view_state_mut().next_accent_color()),
            Some(s) => match s {
                "red" => Ok(Color::Red),
                "green" => Ok(Color::Green),
                "yellow" => Ok(Color::Yellow),
                "blue" => Ok(Color::Blue),
                "magenta" => Ok(Color::Magenta),
                "cyan" => Ok(Color::Cyan),
                "white" => Ok(Color::White),
                hex if hex.starts_with('#') => parse_hex_color(hex),
                other => Err(GraphSessionError::invalid(format!(
                    "unknown color {other:?}"
                ))),
            },
        }
    }

    /// Add `span` as a nameless ad hoc highlight. It gets its own id so an explicit `color`
    /// pins only this highlight. Without one it takes the theme's highlight accent, the same
    /// on every call: ad hoc highlights mark what the reader just asked to see, so they
    /// aren't colored apart from each other or from the annotation tracks.
    pub fn push_adhoc_highlight(
        &mut self,
        mut span: AnnotationSpan,
        color: Option<&str>,
    ) -> Result<(), GraphSessionError> {
        span.id = HashId::random_str();
        let color = match color {
            Some(color) => self.resolve_color(Some(color))?,
            None => current_theme()[HIGHLIGHT_ACCENT_SLOT],
        };
        self.controller.pin_annotation_color(span.id, color);
        self.controller.overlays_mut().push(GraphOverlay {
            content: OverlayContent::Span(span),
            source: OverlaySource::Adhoc,
            style: annotation_style(color),
        });
        Ok(())
    }

    /// Add `spans` as the file or ad hoc track `name`; `reapply_overlays` colors them.
    pub fn push_track(&mut self, name: &str, mut spans: Vec<AnnotationSpan>) {
        spans.sort_by_key(sort_key_longest_first);
        let overlays = self.controller.overlays_mut();
        overlays.extend(spans.into_iter().map(|span| GraphOverlay {
            content: OverlayContent::Span(span),
            source: OverlaySource::Track(name.to_string()),
            style: annotation_style(Color::Reset),
        }));
    }

    /// The annotation group named `name`, as the entry id its overlays are keyed by.
    pub fn annotation_group_id(&self, name: &str) -> Option<String> {
        self.controller
            .annotation_group_entries()
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.id.clone())
    }
}

pub fn expand_query_window(window: (i64, i64)) -> (i64, i64) {
    let span = (window.1 - window.0).max(1);
    (window.0.saturating_sub(span), window.1.saturating_add(span))
}

pub fn annotation_style(color: Color) -> PathStyle {
    PathStyle::new(color)
        .with_line_style(LineStyle::Bold)
        .with_merge_glyphs(true)
}

impl GraphPage {
    /// Whether Python has already set up this page's annotation groups.
    ///
    /// Returns ``true`` after either [`Self::trigger_auto_load`] or
    /// [`Self::load_annotation_groups_with_colors`] has been called. Cloned
    /// pages inherit this flag, preventing double set-up on cell re-display.
    pub fn annotations_loaded(&self) -> bool {
        self.annotation_groups_loaded
    }

    /// Show every annotation group in the automatic theme palette. The groups load for each
    /// batch as renders reach it.
    pub fn trigger_auto_load(&mut self) -> Result<(), GraphSessionError> {
        self.annotation_groups_loaded = true;
        Ok(())
    }

    /// Show every annotation group, coloring annotations from a Python-resolved map.
    ///
    /// `color_map` maps annotation ID hex strings to a CSS hex colour string
    /// (e.g. ``"#ff4444"``) or ``None`` to hide that annotation entirely.
    /// Annotations absent from the map fall back to the auto theme palette. No-op if
    /// annotations have already been set up.
    pub fn load_annotation_groups_with_colors(
        &mut self,
        color_map: &HashMap<String, Option<String>>,
    ) -> Result<(), GraphSessionError> {
        if self.annotation_groups_loaded {
            return Ok(());
        }
        let overrides = color_map
            .iter()
            .filter_map(|(id, color)| {
                let id = HashId::try_from(id.as_str()).ok()?;
                match color {
                    None => Some((id, None)),
                    // An unreadable color falls back to the automatic palette.
                    Some(hex) => parse_hex_color(hex).ok().map(|color| (id, Some(color))),
                }
            })
            .collect();
        self.controller.set_annotation_color_overrides(overrides);
        self.annotation_groups_loaded = true;
        Ok(())
    }

    /// Set the level of node detail, matching the four zoom steps of the CLI viewer.
    ///
    /// Parameters
    /// detail : {"minimal", "normal", "labels", "full"}
    ///     ``"minimal"`` shows the smallest representation; ``"normal"`` shows truncated
    ///     nodes with annotation names hidden; ``"labels"`` shows truncated nodes with
    ///     annotation names; ``"full"`` shows complete sequences.
    pub fn set_detail(&mut self, detail: &str) -> Result<(), GraphSessionError> {
        let (level, labels) = match detail {
            "minimal" => (VisualDetail::Minimal, None),
            "normal" => (VisualDetail::Truncated, Some(false)),
            "labels" => (VisualDetail::Truncated, Some(true)),
            "full" => (VisualDetail::Full, None),
            other => {
                return Err(GraphSessionError::invalid(format!(
                    "detail must be \"minimal\", \"normal\", \"labels\", or \"full\"; got {other:?}"
                )));
            }
        };
        self.controller.set_detail_level(level);
        if let Some(shown) = labels {
            self.controller.show_annotation_names(shown);
        }
        Ok(())
    }

    pub fn truncate_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Truncated);
    }

    pub fn full_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Full);
    }

    pub fn minimize_sequences(&mut self) {
        self.controller.set_detail_level(VisualDetail::Minimal);
    }

    /// Show `coordinate` of node `node_id` at the current detail level, centred when `center`
    /// is set and otherwise against the left edge. Fails when the block group lacks that
    /// position or edits cut it off from the start of the pruned view.
    pub fn go_to_coordinate(
        &mut self,
        node_id: HashId,
        coordinate: i64,
        center: bool,
    ) -> Result<(), LocateError> {
        self.controller
            .go_to_coordinate(node_id, coordinate, center)
    }

    pub fn zoom_in(&mut self) {
        self.controller.zoom_in();
    }

    pub fn set_labels_shown(&mut self, shown: bool) {
        self.controller.show_annotation_names(shown);
    }

    pub fn labels_shown(&self) -> bool {
        self.controller.annotation_names_shown()
    }

    pub fn zoom_out(&mut self) {
        self.controller.zoom_out();
    }

    /// Apply a click; a door takes the view into the batch behind it. Returns whether
    /// anything changed.
    pub fn handle_click(&mut self, col: u16, row: u16) -> bool {
        self.controller.click(col, row) != ClickOutcome::Missed
    }

    pub fn move_by(&mut self, dx: i16, dy: i16) {
        self.controller.pan(dx, dy);
    }

    /// Remove ephemeral highlights while keeping named tracks and the selected path.
    pub fn clear_highlights(&mut self) {
        self.controller.overlays_mut().retain(|overlay| {
            matches!(
                overlay.source,
                OverlaySource::Track(_) | OverlaySource::Path
            )
        });
    }

    /// Highlight the most recent path associated with this sequence graph.
    ///
    /// Parameters
    /// color : str, optional
    ///     Colour for the highlight.  Accepts named colours
    ///     (``"yellow"``, ``"cyan"``, ``"red"``, …) or a CSS hex string
    ///     (``"#ff4444"``).  When omitted the next unused theme accent
    ///     colour is chosen automatically.
    ///
    /// Raises
    /// RuntimeError
    ///     If no sequence graph is associated with this widget, or if no path
    ///     exists for the sequence graph.
    /// ValueError
    ///     If ``color`` is not a recognised colour name or CSS hex string.
    pub fn show_path(&mut self, color: Option<&str>) -> Result<(), GraphSessionError> {
        let block_group_id = self.block_group_id().ok_or_else(|| {
            GraphSessionError::operation(
                "show_path() requires a sequence graph; obtain the widget via SequenceGraph.plot()",
            )
        })?;
        let highlight_color = self.resolve_color(color)?;
        let conn = self
            .controller
            .database_mut()
            .connection()
            .map_err(operation_error)?;
        let path =
            BlockGroup::get_current_path(conn, &block_group_id, None).map_err(operation_error)?;
        // Only the path's edge ids are fetched; nothing is added to the crawled graph. Each
        // reapply highlights whichever of those edges are loaded, so batches crawled later
        // extend the highlight without touching the database again.
        let membership = PathMembership::load(conn, &path.id, None);
        if membership.is_empty() {
            return Err(GraphSessionError::operation("Path has no edges"));
        }
        set_path_overlay(
            self.controller.overlays_mut(),
            annotation_style(highlight_color),
            membership,
        );
        Ok(())
    }

    /// Clear path highlighting previously applied by `show_path`.
    pub fn clear_path(&mut self) {
        remove_path_overlay(self.controller.overlays_mut());
    }

    /// Show the annotation group `group` again after `remove_track` or
    /// `clear_all_annotations` hid it.
    pub fn add_track_group(&mut self, group: &str) -> Result<(), GraphSessionError> {
        if self.annotation_group_id(group).is_none() {
            let mut database = self.controller.database_mut().clone();
            let conn = database.connection().map_err(operation_error)?;
            if self.show_annotation_file(conn, group) {
                return Ok(());
            }
        }
        let group_id = self.annotation_group_id(group).ok_or_else(|| {
            GraphSessionError::operation(format!("no annotation group named {group:?}"))
        })?;
        self.controller
            .set_annotation_group_enabled(&group_id, true);
        Ok(())
    }

    /// Remove the track `name`: an annotation group stays hidden for later batches too, until
    /// `add_track_group` shows it again.
    pub fn remove_track(&mut self, name: &str) {
        for file in &mut self.annotation_files {
            if file.entry.display_name == name {
                file.shown = false;
            }
        }
        if let Some(group_id) = self.annotation_group_id(name) {
            self.controller
                .set_annotation_group_enabled(&group_id, false);
        }
        self.controller
            .overlays_mut()
            .retain(|overlay| !matches!(&overlay.source, OverlaySource::Track(key) if key == name));
    }

    /// Clear all annotations from the graph, hiding every annotation group.
    pub fn clear_all_annotations(&mut self) {
        for file in &mut self.annotation_files {
            file.shown = false;
        }
        let group_ids: Vec<String> = self
            .controller
            .annotation_group_entries()
            .iter()
            .map(|entry| entry.id.clone())
            .collect();
        for group_id in &group_ids {
            self.controller
                .set_annotation_group_enabled(group_id, false);
        }
        // Keep ad hoc highlights (e.g. search matches) and the path.
        self.controller
            .overlays_mut()
            .retain(|overlay| matches!(overlay.source, OverlaySource::Adhoc | OverlaySource::Path));
    }

    /// Remove all overlays whose annotation name matches `name`, regardless of
    /// which track (if any) they belong to. If the same name was added more
    /// than once, every copy is removed.
    pub fn remove_annotation(&mut self, name: &str) {
        self.controller
            .overlays_mut()
            .retain(|overlay| overlay.span().is_none_or(|span| span.name != name));
    }
}

impl GraphPage {
    pub fn visible_coordinate_window(&self) -> Option<(i64, i64)> {
        let world = self.controller.engine().active_world()?;
        let mut start = i64::MAX;
        let mut end = i64::MIN;
        for node in world.members() {
            if !is_terminal(node.node_id) {
                start = start.min(node.sequence_start);
                end = end.max(node.sequence_end);
            }
        }
        (start <= end).then_some((start, end))
    }

    /// Read the repository's annotation files, keeping the display state of ones already known.
    pub fn sync_annotation_files(&mut self, conn: &GraphConnection) {
        let entries = load_annotation_file_entries(conn, None);
        let mut previous = std::mem::take(&mut self.annotation_files);
        self.annotation_files = entries
            .into_iter()
            .map(|entry| {
                let known = previous
                    .iter()
                    .position(|file| file.entry.file_addition.id == entry.file_addition.id);
                match known {
                    Some(index) => FileTrack {
                        entry,
                        ..previous.swap_remove(index)
                    },
                    None => FileTrack {
                        entry,
                        shown: false,
                        index_available: false,
                        loaded_window: None,
                    },
                }
            })
            .collect();
    }

    /// Load one file for `window`, replacing whatever it showed before.
    pub fn load_annotation_file(
        &mut self,
        conn: &GraphConnection,
        index: usize,
        window: Option<(i64, i64)>,
    ) -> bool {
        let Some(block_group_id) = self.block_group_id() else {
            return false;
        };
        let Ok(block_group) = BlockGroup::get_by_id(conn, &block_group_id, None) else {
            return false;
        };
        let node_filter = self.controller.loaded_node_ids();
        let entry = self.annotation_files[index].entry.clone();
        let Ok(workspace) = workspace_for_connection(conn) else {
            return false;
        };
        let loaded = load_annotation_file_track(&AnnotationFileTrackRequest {
            conn,
            history_ref: None,
            workspace: &workspace,
            collection_name: &block_group.collection_name,
            sample_name: &block_group.sample_name,
            block_group_name: Some(&block_group.name),
            query_window: window,
            node_filter: &node_filter,
            entry: &entry,
        });
        let Ok(loaded) = loaded else {
            return false;
        };
        let name = entry.display_name.clone();
        self.controller.overlays_mut().retain(
            |overlay| !matches!(&overlay.source, OverlaySource::Track(track) if *track == name),
        );
        self.push_track(&loaded.track.name, loaded.track.annotations);
        let file = &mut self.annotation_files[index];
        file.shown = true;
        file.index_available = loaded.index_available;
        file.loaded_window = loaded.loaded_window;
        true
    }

    /// The window of sequence coordinates to read an indexed file for: the viewport and as much
    /// again on each side, so small camera moves don't reload.
    pub fn annotation_query_window(&self) -> Option<(i64, i64)> {
        self.visible_coordinate_window().map(expand_query_window)
    }

    /// Show every annotation file, as database groups are all shown when a graph is plotted.
    /// A file that can't be read is skipped so it doesn't hide the others.
    pub fn show_all_annotation_files(&mut self, conn: &GraphConnection) {
        self.sync_annotation_files(conn);
        let window = self.annotation_query_window();
        for index in 0..self.annotation_files.len() {
            self.load_annotation_file(conn, index, window);
        }
    }

    /// Show the file recorded under `name`, returning whether there is one.
    pub fn show_annotation_file(&mut self, conn: &GraphConnection, name: &str) -> bool {
        self.sync_annotation_files(conn);
        let Some(index) = self
            .annotation_files
            .iter()
            .position(|file| file.entry.display_name == name)
        else {
            return false;
        };
        let window = self.annotation_query_window();
        self.load_annotation_file(conn, index, window)
    }

    /// Reload shown, indexed files whose loaded window no longer covers the viewport.
    pub fn refresh_annotation_files_for_viewport(&mut self) -> bool {
        let Some(visible) = self.visible_coordinate_window() else {
            return false;
        };
        let window = expand_query_window(visible);
        let stale: Vec<usize> = self
            .annotation_files
            .iter()
            .enumerate()
            .filter(|(_, file)| file.shown && file.index_available)
            .filter(|(_, file)| match file.loaded_window {
                Some((start, end)) => visible.0 < start || visible.1 > end,
                None => true,
            })
            .map(|(index, _)| index)
            .collect();
        if stale.is_empty() {
            return false;
        }
        let mut database = self.controller.database_mut().clone();
        let Ok(conn) = database.connection() else {
            return false;
        };
        let mut reloaded = false;
        for index in stale {
            reloaded |= self.load_annotation_file(conn, index, Some(window));
        }
        reloaded
    }
}

/// Draw a one-line header into `area`: the sequence graph name, centred. The
/// page index/count are not drawn into the grid; they are exposed as widget
/// metadata instead, so the frontend can render its own `<index/count>`
/// pager indicator outside the canvas.
pub fn draw_header(buf: &mut Buffer, area: Rect, name: &str) {
    let theme = current_theme();
    let style = Style::default().fg(theme[0x07]);
    let name_x = area.x + area.width.saturating_sub(name.len() as u16) / 2;
    buf.set_string(name_x, area.y, name, style);
}

/// Reject a pixel scale the exporter should not be asked for.
pub fn validated_png_scale(scale: usize) -> Result<usize, GraphSessionError> {
    if (1..=MAX_SCALE).contains(&scale) {
        Ok(scale)
    } else {
        Err(GraphSessionError::invalid(format!(
            "scale must be between 1 and {MAX_SCALE}, got {scale}"
        )))
    }
}

/// One clickable cell of the last rendered viewport, for the browser's hover and click mapping.
#[derive(Serialize)]
struct HitRegion {
    x: u16,
    y: u16,
    kind: &'static str,
}

/// One or more pages of sequence graphs, one of them active.
///
/// A plain single-graph plot is the common case of one page. A sample pages through several,
/// building each page's graph on first visit since most sessions never page through them all.
#[derive(Clone)]
pub struct GraphSession {
    pages: Vec<Page>,
    current_index: usize,
    /// Set by `set_cursor_hidden`: keeps the cursor out of every render, including after a
    /// later go-to or click would otherwise bring it back.
    cursor_hidden: bool,
}

impl GraphSession {
    /// A session holding one already built page.
    pub fn single(page: GraphPage) -> Self {
        Self {
            pages: vec![Page::Loaded(Box::new(page))],
            current_index: 0,
            cursor_hidden: false,
        }
    }

    /// A session paging through `page_refs`, each page built when first visited.
    pub fn paged(page_refs: Vec<PageRef>) -> Result<Self, GraphSessionError> {
        if page_refs.is_empty() {
            return Err(GraphSessionError::operation(
                "Sample has no sequence graphs to plot",
            ));
        }
        Ok(Self {
            pages: page_refs
                .into_iter()
                .map(|page_ref| Page::Pending(Box::new(page_ref)))
                .collect(),
            current_index: 0,
            cursor_hidden: false,
        })
    }

    /// The active page, built now if it has not been visited yet.
    pub fn active(&mut self) -> Result<&mut GraphPage, GraphSessionError> {
        let page = &mut self.pages[self.current_index];
        if let Page::Pending(page_ref) = page {
            let loaded = GraphPage::new(
                page_ref.name.clone(),
                page_ref.database.clone(),
                page_ref.block_group_id,
                page_ref.options,
            )?;
            *page = Page::Loaded(Box::new(loaded));
        }
        match page {
            Page::Loaded(page) => Ok(page.as_mut()),
            Page::Pending(_) => unreachable!("just loaded above"),
        }
    }

    pub fn active_name(&self) -> &str {
        self.pages[self.current_index].name()
    }

    /// Switch to the next page, wrapping around. A no-op when there is only one page.
    pub fn next_page(&mut self) {
        self.current_index = (self.current_index + 1) % self.pages.len();
    }

    /// Switch to the previous page, wrapping around. A no-op when there is only one page.
    pub fn previous_page(&mut self) {
        self.current_index = (self.current_index + self.pages.len() - 1) % self.pages.len();
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn page_index(&self) -> usize {
        self.current_index
    }

    /// Keep the cursor out of every later render (`true`), or let navigation bring it back.
    pub fn set_cursor_hidden(&mut self, hidden: bool) -> Result<(), GraphSessionError> {
        self.cursor_hidden = hidden;
        if !hidden {
            self.active()?.controller.view_state_mut().show_cursor();
        }
        Ok(())
    }

    /// Draw just the active page's graph into a fresh `cols` x `rows` buffer, without the header.
    pub fn render_graph_buffer(
        &mut self,
        cols: u16,
        rows: u16,
    ) -> Result<Buffer, GraphSessionError> {
        let mut buf = Buffer::empty(Rect::new(0, 0, cols, rows));
        self.render_active(&mut buf, Rect::new(0, 0, cols, rows))?;
        Ok(buf)
    }

    /// Draw the header and the active page into a fresh `cols` x `rows` buffer.
    pub fn render_frame_buffer(
        &mut self,
        cols: u16,
        rows: u16,
    ) -> Result<Buffer, GraphSessionError> {
        let mut buf = Buffer::empty(Rect::new(0, 0, cols, rows));
        let (header_area, graph_area) = frame_areas(cols, rows);
        draw_header(&mut buf, header_area, self.active_name());
        self.render_active(&mut buf, graph_area)?;
        Ok(buf)
    }

    /// Draw the active page into `graph_area`, honouring `set_cursor_hidden`.
    pub fn render_active(
        &mut self,
        buf: &mut Buffer,
        graph_area: Rect,
    ) -> Result<(), GraphSessionError> {
        let cursor_hidden = self.cursor_hidden;
        let page = self.active()?;
        if cursor_hidden {
            page.controller.view_state_mut().hide_cursor();
        }
        page.render_into(buf, graph_area);
        Ok(())
    }

    /// Rasterize the active page to PNG bytes with Bezier edges. With `full_viewport` the
    /// header and every cell keep their positions, as in the interactive widget; otherwise
    /// the image is the graph alone, cropped to its content.
    pub fn render_png(
        &mut self,
        (cols, rows): (u16, u16),
        full_viewport: bool,
        scale: usize,
    ) -> Result<Vec<u8>, GraphSessionError> {
        let scale = validated_png_scale(scale)?;
        let controller = &mut self.active()?.controller;
        let previous_layout_mode = controller.view_state().layout_mode;
        let previous_placement = controller.scale_placement();
        controller.view_state_mut().layout_mode = LayoutMode::Curved;
        controller.set_scale_placement(if full_viewport {
            ScalePlacement::Bottom
        } else {
            ScalePlacement::BelowContent
        });
        let buffer_result = if full_viewport {
            self.render_frame_buffer(cols, rows)
        } else {
            self.render_graph_buffer(cols, rows)
        };
        let controller = &mut self.active()?.controller;
        controller.view_state_mut().layout_mode = previous_layout_mode;
        controller.set_scale_placement(previous_placement);
        let buffer = buffer_result?;
        let mut bytes = Vec::new();
        let write_result = if full_viewport {
            write_uncropped_png(&buffer, controller.overlay_primitives(), scale, &mut bytes)
        } else {
            write_cropped_png(&buffer, controller.overlay_primitives(), scale, &mut bytes)
        };
        write_result.map_err(operation_error)?;
        Ok(bytes)
    }

    /// The clickable cells of the last rendered `cols` x `rows` viewport, as JSON
    /// `[{"x", "y", "kind"}]` with `kind` either `"node"` or `"wormhole"`.
    pub fn hit_regions_json(&mut self, cols: u16, rows: u16) -> Result<String, GraphSessionError> {
        let view_state = self.active()?.controller.view_state();
        let mut hit_regions = Vec::new();
        for y in 0..rows {
            for x in 0..cols {
                let kind = if view_state.wormhole_hit(x, y).is_some() {
                    Some("wormhole")
                } else if view_state.node_hit(x, y).is_some() {
                    Some("node")
                } else {
                    None
                };
                if let Some(kind) = kind {
                    hit_regions.push(HitRegion { x, y, kind });
                }
            }
        }
        serde_json::to_string(&hit_regions).map_err(operation_error)
    }

    /// The logical pixel dimensions of a PNG terminal cell. A PNG rendered at scale `s` has
    /// cells `s` times this size in image pixels.
    pub fn png_cell_size() -> (usize, usize) {
        RasterConfig::default().logical_cell_size()
    }

    /// The exact canvas color of the native PNG renderer, as `#rrggbb`.
    pub fn png_background_color() -> String {
        let [red, green, blue] = RasterConfig::default().background;
        format!("#{red:02x}{green:02x}{blue:02x}")
    }

    /// Every annotation-group and annotation-file name visible from the active page, the full
    /// menu `add_track_group` accepts.
    pub fn track_names(&mut self) -> Result<Vec<String>, GraphSessionError> {
        let page = self.active()?;
        let mut database = page.controller.database_mut().clone();
        let conn = database.connection().map_err(operation_error)?;
        page.sync_annotation_files(conn);
        Ok(page
            .controller
            .annotation_group_entries()
            .iter()
            .map(|entry| entry.name.clone())
            .chain(
                page.annotation_files
                    .iter()
                    .map(|file| file.entry.display_name.clone()),
            )
            .collect())
    }
}

impl PageRef {
    pub fn new(
        name: String,
        database: GraphDatabase,
        block_group_id: HashId,
        options: PlotOptions,
    ) -> Self {
        Self {
            name,
            database,
            block_group_id,
            options,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GraphPage, GraphSession, GraphSessionError, MAX_SCALE, PlotOptions, validated_png_scale,
        workspace_for_connection,
    };
    use crate::{
        test_helpers::{setup_block_group, setup_gen_on_disk},
        views::graph_database::GraphDatabase,
    };

    fn session_with_one_page() -> (crate::test_helpers::TestContext, GraphSession) {
        let context = setup_gen_on_disk();
        let conn = context.graph().conn();
        let (block_group_id, _) = setup_block_group(conn);
        let workspace = workspace_for_connection(conn).expect("should find the workspace");
        let database = GraphDatabase::for_connection(conn, &workspace)
            .expect("should open the graph database");
        let page = GraphPage::new(
            "test".to_string(),
            database,
            block_group_id,
            PlotOptions::default(),
        )
        .expect("should open the block group");
        (context, GraphSession::single(page))
    }

    fn png_width(bytes: &[u8]) -> u32 {
        assert_eq!(
            &bytes[..8],
            b"\x89PNG\r\n\x1a\n",
            "should start with the PNG signature"
        );
        u32::from_be_bytes(
            bytes[16..20]
                .try_into()
                .expect("should read the IHDR width"),
        )
    }

    #[test]
    fn test_validated_png_scale_rejects_zero_and_oversized_scales() {
        assert!(matches!(
            validated_png_scale(0),
            Err(GraphSessionError::InvalidArgument(_))
        ));
        assert!(matches!(
            validated_png_scale(MAX_SCALE + 1),
            Err(GraphSessionError::InvalidArgument(_))
        ));
        assert_eq!(validated_png_scale(2).expect("should accept 2"), 2);
    }

    #[test]
    fn test_paged_session_needs_at_least_one_page() {
        assert!(matches!(
            GraphSession::paged(Vec::new()),
            Err(GraphSessionError::Operation(_))
        ));
    }

    #[test]
    fn test_render_png_doubles_at_scale_two() {
        let (_context, mut session) = session_with_one_page();

        let single = session
            .render_png((40, 12), true, 1)
            .expect("should render at scale 1");
        let doubled = session
            .render_png((40, 12), true, 2)
            .expect("should render at scale 2");

        assert_eq!(png_width(&doubled), 2 * png_width(&single));
    }

    #[test]
    fn test_hit_regions_cover_the_rendered_nodes() {
        let (_context, mut session) = session_with_one_page();
        session
            .render_png((40, 12), true, 1)
            .expect("should render the first batch");

        let regions = session
            .hit_regions_json(40, 12)
            .expect("should scan the viewport");

        assert!(regions.contains("\"kind\":\"node\""));
    }

    #[test]
    fn test_png_cell_size_and_background_describe_the_exporter_defaults() {
        let (width, height) = GraphSession::png_cell_size();

        assert!(width > 0 && height > 0);
        assert!(GraphSession::png_background_color().starts_with('#'));
    }
}
