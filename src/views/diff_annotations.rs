//! Annotations on a sample diff, drawn the way the block-group viewers draw them.
//!
//! A diff graph is built in memory from two samples' block groups, so there is no lazy crawl
//! that loads annotations batch by batch. Instead every annotation that applies to either
//! sample is loaded once against the whole diff graph: the annotations stored in the database
//! always, and the imported annotation files the user named on the command line.

use std::collections::HashSet;

use gen_core::{HashId, Workspace, is_end_node, is_start_node};
use gen_diff::graph::{DiffGenGraph, DiffGenGraphRef};
use gen_graph::{GenGraph, GraphNode, PackedGenGraph};
use gen_models::{block_group::BlockGroup, db::GraphConnection};
use gen_tui::{
    graph_view::GraphViewState, layout::VisualDetail, layout_compaction::GapPolicy,
    layout_engine::LayoutEngine, plotter::NodeRenderer,
};
use log::warn;

use crate::views::{
    annotation_files::AnnotationFileEntry,
    annotation_groups::load_annotation_group_entries,
    annotations::{
        AnnotationFileTrackRequest, AnnotationGroupTrackRequest, load_annotation_file_track,
        load_annotations_for_group,
    },
    block_group::expand_query_window,
    diff_graph::{DiffGraphComponent, apply_diff_highlights},
    gen_graph_widget::{
        NodeAnnotationLayer, SequenceSource, ZoomLevels, create_annotated_gen_graph_engine,
        create_gen_graph_engine, reapply_overlays, update_node_annotations,
    },
    graph_overlay::{
        AnnotationColorCache, GraphOverlay, file_track_overlays, group_track_key,
        replace_track_overlays,
    },
};

/// Where a diff's annotations come from.
pub struct DiffAnnotationSources<'a> {
    pub conn: &'a GraphConnection,
    pub workspace: &'a Workspace,
    pub history_ref: Option<&'a str>,
    pub query_block_group: &'a BlockGroup,
    pub base_block_group: &'a BlockGroup,
    /// The imported annotation files to draw in addition to the stored annotations.
    pub files: &'a [AnnotationFileEntry],
}

/// The annotations of one diff graph and the state that colors and lays them out.
pub struct DiffAnnotations {
    layer: NodeAnnotationLayer,
    overlays: Vec<GraphOverlay>,
    colors: AnnotationColorCache,
    applied_zoom_index: Option<usize>,
}

impl DiffAnnotations {
    /// Load the annotations of both samples onto `graph`. An annotation both samples carry,
    /// such as one inherited from a common ancestor, is drawn once. A group or file that
    /// cannot be read is skipped with a warning so it does not hide the others.
    pub fn load(sources: &DiffAnnotationSources<'_>, diff_graph: &DiffGenGraph) -> Self {
        let graph: GenGraph = DiffGenGraphRef(diff_graph).into();
        let graph = &graph;
        let node_ids: HashSet<HashId> = graph.nodes().map(|node| node.node_id).collect();
        let query_window = coordinate_window(graph).map(expand_query_window);
        let block_groups = [sources.query_block_group, sources.base_block_group];
        let mut drawn: HashSet<HashId> = HashSet::new();
        let mut overlays: Vec<GraphOverlay> = Vec::new();

        let mut loaded_groups: HashSet<String> = HashSet::new();
        for block_group in block_groups {
            let entries =
                load_annotation_group_entries(sources.conn, block_group, sources.history_ref);
            for entry in entries {
                if !loaded_groups.insert(entry.id.clone()) {
                    continue;
                }
                let spans = match load_annotations_for_group(&AnnotationGroupTrackRequest {
                    conn: sources.conn,
                    history_ref: sources.history_ref,
                    entry: &entry,
                    projection_graph: graph,
                    node_ids: &node_ids,
                }) {
                    Ok(spans) => spans,
                    Err(error) => {
                        warn!("Failed to load annotations for group {}: {error}", entry.id);
                        continue;
                    }
                };
                let spans = spans
                    .into_iter()
                    .filter(|span| drawn.insert(span.id))
                    .collect();
                replace_track_overlays(&mut overlays, &group_track_key(&entry.id), spans);
            }
        }

        for entry in sources.files {
            for block_group in block_groups {
                let loaded = match load_annotation_file_track(&AnnotationFileTrackRequest {
                    conn: sources.conn,
                    history_ref: sources.history_ref,
                    workspace: sources.workspace,
                    collection_name: &block_group.collection_name,
                    sample_name: &block_group.sample_name,
                    block_group_name: Some(&block_group.name),
                    query_window,
                    node_filter: &node_ids,
                    entry,
                }) {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        warn!(
                            "Failed to load annotation file {}: {error}",
                            entry.display_name
                        );
                        continue;
                    }
                };
                overlays.extend(
                    file_track_overlays(loaded.track)
                        .into_iter()
                        .filter(|overlay| overlay.span().is_some_and(|span| drawn.insert(span.id))),
                );
            }
        }

        Self {
            layer: NodeAnnotationLayer::default(),
            overlays,
            colors: AnnotationColorCache::new(),
            applied_zoom_index: None,
        }
    }

    /// No annotations, for a diff whose samples are not stored block groups.
    pub fn none() -> Self {
        Self {
            layer: NodeAnnotationLayer::default(),
            overlays: Vec::new(),
            colors: AnnotationColorCache::new(),
            applied_zoom_index: None,
        }
    }

    /// The layout engine, zoom levels and view state for `graph`. Without annotations the
    /// plain renderers are used, so an unannotated diff looks exactly as it always has.
    pub fn create_engine<'a, S: SequenceSource + Clone + 'a>(
        &self,
        graph: GenGraph,
        source: S,
    ) -> (
        LayoutEngine<PackedGenGraph>,
        ZoomLevels<'a>,
        GraphViewState<GraphNode>,
    ) {
        if self.overlays.is_empty() {
            create_gen_graph_engine(graph, source)
        } else {
            create_annotated_gen_graph_engine(graph, source, self.layer.clone())
        }
    }

    /// Color the annotations and refill the flag layer for the current zoom level. The diff
    /// graph never changes, so only a zoom change needs this; call it before every draw.
    /// Coloring the annotations rebuilds the view's highlights, so the diff's own added and
    /// removed marks from `component` are put back afterwards.
    pub fn prepare<R>(
        &mut self,
        engine: &LayoutEngine<PackedGenGraph>,
        view_state: &mut GraphViewState<GraphNode>,
        zoom_levels: &[(VisualDetail, R, GapPolicy)],
        component: &DiffGraphComponent,
    ) where
        R: NodeRenderer<PackedGenGraph>,
    {
        if self.applied_zoom_index == Some(view_state.zoom_index) {
            return;
        }
        reapply_overlays(
            engine,
            view_state,
            zoom_levels,
            &mut self.overlays,
            &mut self.colors,
        );
        update_node_annotations(&self.layer, engine, &self.overlays);
        apply_diff_highlights(view_state, component);
        self.applied_zoom_index = Some(view_state.zoom_index);
    }
}

/// The span of stored sequence coordinates the diff graph covers, which bounds how much of an
/// indexed annotation file is read.
fn coordinate_window(graph: &GenGraph) -> Option<(i64, i64)> {
    let mut start = i64::MAX;
    let mut end = i64::MIN;
    for node in graph.nodes() {
        if is_start_node(node.node_id) || is_end_node(node.node_id) {
            continue;
        }
        start = start.min(node.sequence_start);
        end = end.max(node.sequence_end);
    }
    (start <= end).then_some((start, end))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gen_diff::sample::build_sample_diff;
    use gen_models::{
        annotations::{AnnotationFileChecksumOverrides, add_annotation, add_annotation_file},
        operations::commit_operation_summary,
        sample::Sample,
    };
    use gen_tui::{LineStyle, graph_view::GraphView, plotter::PathStyle};
    use ratatui::{buffer::Buffer, layout::Rect, style::Color, widgets::StatefulWidget};

    use super::*;
    use crate::{
        imports::fasta::import_fasta,
        test_helpers::{TestContext, setup_gen_on_disk},
        views::{
            annotation_files::load_annotation_file_entries, diff_graph::build_diff_graph_component,
        },
    };

    /// A repository with one sequence, one annotation stored in the database and one imported
    /// GFF file, and the diff of that sequence's sample against itself.
    fn setup_annotated_diff() -> (TestContext, gen_diff::sample::SampleDiff) {
        let context = setup_gen_on_disk();
        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let import = import_fasta(
            &context,
            &fasta_path
                .to_str()
                .expect("should encode the fixture path")
                .to_string(),
            "test",
            Sample::DEFAULT_NAME,
            None,
            None,
        )
        .expect("should import the fixture FASTA");
        commit_operation_summary(&context, &import).expect("should commit the fixture FASTA");
        add_annotation(
            &context,
            "test",
            "stored-gene",
            Some("genes"),
            Sample::DEFAULT_NAME,
            "m123:5-20",
        )
        .expect("should store an annotation");
        let gff_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.gff");
        add_annotation_file(
            &context,
            gff_path.to_str().expect("should encode the GFF path"),
            Some("gff3"),
            None,
            Some("feature-annotations"),
            Some("add feature annotations"),
            AnnotationFileChecksumOverrides::default(),
        )
        .expect("should add the annotation file");
        let diff = build_sample_diff(
            context.graph().conn(),
            "test",
            "m123",
            Sample::DEFAULT_NAME,
            Sample::DEFAULT_NAME,
            None,
        )
        .expect("should build the sample diff");
        (context, diff)
    }

    fn load_annotations(
        context: &TestContext,
        diff: &gen_diff::sample::SampleDiff,
        files: &[AnnotationFileEntry],
    ) -> DiffAnnotations {
        DiffAnnotations::load(
            &DiffAnnotationSources {
                conn: context.graph().conn(),
                workspace: context.workspace(),
                history_ref: None,
                query_block_group: &diff.query_block_group,
                base_block_group: &diff.base_block_group,
                files,
            },
            &diff.graph,
        )
    }

    fn span_names(annotations: &DiffAnnotations) -> Vec<String> {
        annotations
            .overlays
            .iter()
            .filter_map(|overlay| overlay.span().map(|span| span.name.clone()))
            .collect()
    }

    #[test]
    fn test_diff_draws_stored_annotations_without_naming_a_file() {
        let (context, diff) = setup_annotated_diff();

        let annotations = load_annotations(&context, &diff, &[]);

        assert_eq!(span_names(&annotations), ["stored-gene"]);
    }

    #[test]
    fn test_diff_draws_each_annotation_once_when_both_samples_carry_it() {
        let (context, diff) = setup_annotated_diff();
        let files = load_annotation_file_entries(context.graph().conn(), None);

        let annotations = load_annotations(&context, &diff, &files);

        let names = span_names(&annotations);
        let unique: HashSet<&String> = names.iter().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "should repeat no annotation: {names:?}"
        );
    }

    #[test]
    fn test_diff_adds_the_features_of_a_named_file() {
        let (context, diff) = setup_annotated_diff();
        let files = load_annotation_file_entries(context.graph().conn(), None);

        let annotations = load_annotations(&context, &diff, &files);

        let names = span_names(&annotations);
        assert!(names.contains(&"stored-gene".to_string()), "{names:?}");
        assert!(names.contains(&"foobar".to_string()), "{names:?}");
    }

    #[test]
    fn test_diff_without_annotations_uses_the_plain_renderers() {
        let (context, diff) = setup_annotated_diff();
        let plain = DiffAnnotations::none();
        let component = build_diff_graph_component(&diff.graph, String::new());

        let (_, zoom_levels, view_state) = plain.create_engine(
            component.graph.clone(),
            (context.graph().conn(), context.workspace()),
        );
        let (_, expected_levels, expected_state) = create_gen_graph_engine(
            component.graph,
            (context.graph().conn(), context.workspace()),
        );

        assert_eq!(zoom_levels.len(), expected_levels.len());
        assert_eq!(view_state.zoom_index, expected_state.zoom_index);
    }

    #[test]
    fn test_diff_draws_annotation_names_under_the_nodes() {
        let (context, diff) = setup_annotated_diff();
        let mut annotations = load_annotations(&context, &diff, &[]);
        let component = build_diff_graph_component(&diff.graph, String::new());
        let (mut engine, zoom_levels, mut view_state) = annotations.create_engine(
            component.graph.clone(),
            (context.graph().conn(), context.workspace()),
        );
        view_state.zoom_index = zoom_levels.len() - 1;
        annotations.prepare(&engine, &mut view_state, &zoom_levels, &component);
        let area = Rect::new(0, 0, 120, 20);
        let mut buffer = Buffer::empty(area);

        let view = GraphView::new(&mut engine, &zoom_levels[view_state.zoom_index].1);
        view.render(area, &mut buffer, &mut view_state);

        let drawn: String = (0..area.height)
            .flat_map(|row| (0..area.width).map(move |column| (column, row)))
            .map(|position| buffer[position].symbol().to_string())
            .collect();
        assert!(drawn.contains("stored-gene"), "{drawn}");
    }

    #[test]
    fn test_preparing_annotations_keeps_the_diff_highlights() {
        let (context, diff) = setup_annotated_diff();
        let mut annotations = load_annotations(&context, &diff, &[]);
        let mut component = build_diff_graph_component(&diff.graph, String::new());
        let marked = component
            .graph
            .nodes()
            .next()
            .expect("should have a node to mark");
        component.highlighted_nodes.push((marked, Color::Green));
        let (engine, zoom_levels, mut view_state) = annotations.create_engine(
            component.graph.clone(),
            (context.graph().conn(), context.workspace()),
        );

        annotations.prepare(&engine, &mut view_state, &zoom_levels, &component);

        let added = PathStyle::new(Color::Green)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true);
        assert!(view_state.has_highlight(&added));
    }
}
