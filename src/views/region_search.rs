#[cfg(test)]
use std::path::PathBuf;

use gen_core::{HashId, Strand, Workspace, region::Region};
use gen_graph::{GenGraph, GraphNode};
#[cfg(test)]
use gen_models::db::DbContext;
#[cfg(test)]
use gen_models::{
    annotations::{Annotation, add_annotation},
    block_group::{BlockGroup, PathCache},
    path::Path,
    sample::Sample,
};
use gen_models::{
    db::GraphConnection,
    locus::GraphLocus,
    region::{GenRegionError, ResolvedGenRegion, ResolvedRegionKind},
};
use gen_tui::{LineStyle, graph_view::GraphViewState, plotter::PathStyle};
use ratatui::style::Color;

use crate::views::{
    annotation_track::{
        AnnotationSegment, AnnotationSpan, LoadedNodeSlices, annotation_span_from_resolved_region,
        graph_locus_from_annotation_span,
    },
    gen_graph_widget::locus_midpoint,
    graph_overlay::{GraphOverlay, OverlayContent, OverlaySource},
};
#[cfg(test)]
use crate::{imports::fasta::import_fasta, test_helpers::setup_gen_on_disk};

#[derive(Clone, Debug)]
pub(super) struct RegionSearchMatch {
    pub(super) label: String,
    region: ResolvedGenRegion,
}

pub(super) struct RegionSearchRequest<'a> {
    pub(super) conn: &'a GraphConnection,
    pub(super) collection_name: &'a str,
    pub(super) sample_name: &'a str,
}

fn resolved_match_label(query: &str, kind: ResolvedRegionKind) -> String {
    let source = match kind {
        ResolvedRegionKind::Path => "path",
        ResolvedRegionKind::Annotation => "model annotation",
        ResolvedRegionKind::Accession => "accession",
        ResolvedRegionKind::BlockGroup => "block group",
    };
    format!("{query} ({source})")
}

fn translate_user_region(region: &Region) -> Region {
    Region {
        name: region.name.clone(),
        start: region.start.map(|coordinate| {
            if coordinate > 0 {
                coordinate - 1
            } else {
                coordinate
            }
        }),
        end: region.end,
    }
}

pub(super) fn resolve_region_search_matches(
    request: &RegionSearchRequest<'_>,
    query: &str,
) -> Result<Vec<RegionSearchMatch>, String> {
    let user_region = Region::parse(query).map_err(|error| error.to_string())?;
    let region = translate_user_region(&user_region);
    let resolved_regions = match gen_models::region::resolve_all(
        &region,
        request.conn,
        request.collection_name,
        request.sample_name,
    ) {
        Ok(resolved_regions) => resolved_regions,
        Err(GenRegionError::NotFound(_)) => {
            return Err(format!("no region matched {query}"));
        }
        Err(error) => return Err(error.to_string()),
    };
    Ok(resolved_regions
        .into_iter()
        .map(|region| RegionSearchMatch {
            label: resolved_match_label(query, region.kind),
            region,
        })
        .collect())
}

pub(super) fn replace_search_overlay(overlays: &mut Vec<GraphOverlay>, span: AnnotationSpan) {
    remove_search_overlay(overlays);
    overlays.push(GraphOverlay {
        content: OverlayContent::Span(span),
        source: OverlaySource::Search,
        style: PathStyle::new(Color::LightYellow)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true),
    });
}

pub(super) fn remove_search_overlay(overlays: &mut Vec<GraphOverlay>) {
    overlays.retain(|overlay| !matches!(&overlay.source, OverlaySource::Search));
}

fn fallback_locus_for_empty_span(
    region: &ResolvedGenRegion,
    conn: &GraphConnection,
    workspace: &Workspace,
    loaded: &LoadedNodeSlices,
) -> Result<GraphLocus, String> {
    let midpoint = region
        .start
        .saturating_add(region.end.saturating_sub(region.start) / 2);
    let offset = midpoint.saturating_sub(region.start);
    let positioned = region
        .find_graph_positions(conn, workspace, offset, 0)
        .map_err(|error| format!("failed to locate region on graph: {error}"))?;
    let position = positioned
        .start_anchors
        .and_then(|positions| positions.into_iter().next())
        .ok_or_else(|| "region did not map to a graph position".to_string())?;
    let coordinate = position.coordinate();

    // A zero-width point has no interval to project. Represent it as a one-base span at the
    // resolved backing-node coordinate so the normal graph projection handles displayed slices.
    for (start, end) in [
        (coordinate, coordinate.saturating_add(1)),
        (coordinate.saturating_sub(1), coordinate),
    ] {
        if start >= end {
            continue;
        }
        let point_span = AnnotationSpan {
            id: HashId::convert_str("region-search-point"),
            name: String::new(),
            segments: vec![AnnotationSegment {
                node_id: position.graph_node.node_id,
                start,
                end,
                strand: Strand::Forward,
            }],
        };
        if let Some(locus) = graph_locus_from_annotation_span(&point_span, loaded) {
            return Ok(locus);
        }
    }

    Err("region did not map to a graph position".to_string())
}

pub(super) fn activate_search_match(
    view_state: &mut GraphViewState<GraphNode>,
    graph: &GenGraph,
    overlays: &mut Vec<GraphOverlay>,
    search_match: &RegionSearchMatch,
    conn: &GraphConnection,
    workspace: &Workspace,
) -> Result<(), String> {
    let span = annotation_span_from_resolved_region(conn, workspace, &search_match.region)?;
    let loaded = LoadedNodeSlices::new(graph);
    let locus = if span.segments.is_empty() {
        fallback_locus_for_empty_span(&search_match.region, conn, workspace, &loaded)?
    } else {
        graph_locus_from_annotation_span(&span, &loaded)
            .ok_or_else(|| "region did not map to a graph position".to_string())?
    };
    let (midpoint_slice, midpoint_offset) = locus_midpoint(&locus)
        .ok_or_else(|| "region did not map to a graph position".to_string())?;
    let node_length = midpoint_slice.block.length();
    let fraction = if node_length == 0 {
        0.5
    } else {
        (midpoint_offset as f64 / node_length as f64).clamp(0.0, 1.0)
    };
    view_state.go_to_node(midpoint_slice.block, (fraction, 0.5));
    replace_search_overlay(overlays, span);
    Ok(())
}

#[cfg(test)]
pub(super) struct RegionSearchFixture {
    pub(super) context: DbContext,
    pub(super) graph: GenGraph,
}

#[cfg(test)]
impl RegionSearchFixture {
    pub(super) fn request(&self) -> RegionSearchRequest<'_> {
        RegionSearchRequest {
            conn: self.context.graph().conn(),
            collection_name: "test",
            sample_name: "simple",
        }
    }

    pub(super) fn workspace(&self) -> &Workspace {
        self.context.workspace()
    }
}

#[cfg(test)]
pub(super) fn search_request_fixture() -> RegionSearchFixture {
    let context = setup_gen_on_disk();
    let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
    let fasta_path = fasta_path
        .to_str()
        .expect("should encode FASTA fixture path")
        .to_string();
    import_fasta(&context, &fasta_path, "test", "simple", false, &[])
        .expect("should import search fixture FASTA");
    add_annotation(&context, "test", "model-gene", None, "simple", "m123:5-20")
        .expect("should create model annotation");
    let conn = context.graph().conn();
    let duplicate_block_group = BlockGroup::get_by_name(conn, "test", "simple", "m123", None)
        .expect("should find duplicate annotation block group");
    let duplicate_path = BlockGroup::get_current_path(conn, &duplicate_block_group.id, None)
        .expect("should find duplicate annotation path");
    let path_edge_ids = Path::edge_ids_for_path(conn, &duplicate_path.id, None);
    Path::create(conn, "chr1", &duplicate_block_group.id, &path_edge_ids)
        .expect("should create a distinct path search fixture");
    let mut path_cache = PathCache::new(conn);
    for (accession_name, start, end) in [
        ("duplicate-accession-a", 0, 5),
        ("duplicate-accession-b", 20, 25),
    ] {
        let accession = BlockGroup::add_accession(
            conn,
            &duplicate_path,
            accession_name,
            start,
            end,
            &mut path_cache,
        )
        .expect("should create duplicate annotation accession");
        Annotation::get_or_create(conn, "duplicate-gene", "default", &accession.id, None)
            .expect("should create duplicate model annotation");
    }

    let conn = context.graph().conn();
    let block_group = Sample::get_block_groups(conn, "test", "simple", None)
        .into_iter()
        .find(|block_group| block_group.name == "m123")
        .expect("should find imported block group");
    let graph = BlockGroup::get_graph(conn, context.workspace(), &block_group.id, None)
        .expect("should load search graph");
    RegionSearchFixture { context, graph }
}

#[cfg(test)]
mod tests {
    use gen_core::region::Region;
    use gen_tui::{graph_view::GraphViewState, plotter::PathStyle};
    use ratatui::style::Color;

    use super::{
        RegionSearchFixture, RegionSearchMatch, activate_search_match, remove_search_overlay,
        resolve_region_search_matches, search_request_fixture, translate_user_region,
    };
    use crate::views::{
        annotation_track::{AnnotationSpan, annotation_span_from_resolved_region},
        graph_overlay::{GraphOverlay, OverlayContent, OverlaySource, PathMembership},
    };

    fn match_for_query(fixture: &RegionSearchFixture, query: &str) -> RegionSearchMatch {
        resolve_region_search_matches(&fixture.request(), query)
            .expect("should resolve query")
            .into_iter()
            .next()
            .expect("query should have a match")
    }

    fn span_for_match(
        fixture: &RegionSearchFixture,
        search_match: &RegionSearchMatch,
    ) -> AnnotationSpan {
        annotation_span_from_resolved_region(
            fixture.context.graph().conn(),
            fixture.workspace(),
            &search_match.region,
        )
        .expect("should map resolved region to an annotation span")
    }

    #[test]
    fn test_region_search_locates_block_groups_paths_and_model_annotations() {
        let request = search_request_fixture();

        let path_match = match_for_query(&request, "chr1:20-30");
        assert_eq!(path_match.label, "chr1:20-30 (path)");
        let path_span = span_for_match(&request, &path_match);
        assert_eq!(path_span.segments.len(), 1);
        assert_eq!(path_span.segments[0].start, 19);
        assert_eq!(path_span.segments[0].end, 30);

        let mut view_state = GraphViewState::default();
        let mut overlays = Vec::new();
        activate_search_match(
            &mut view_state,
            &request.graph,
            &mut overlays,
            &path_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("should select and center the path match");
        let selected_node = view_state
            .cursor
            .node
            .expect("should select the resolved graph node");
        assert_eq!(selected_node.sequence_start, 0);
        let selected_fraction = view_state.cursor.fractional.0;
        assert!((selected_fraction - (24.0 / 34.0)).abs() < f64::EPSILON);
        assert_eq!(overlays.len(), 1);
        assert!(matches!(overlays[0].source, OverlaySource::Search));

        let path_slice_match = match_for_query(&request, "chr1:5-10");
        let path_slice_span = span_for_match(&request, &path_slice_match);
        assert_eq!(path_slice_span.segments.len(), 1);
        assert_eq!(path_slice_span.segments[0].start, 4);
        assert_eq!(path_slice_span.segments[0].end, 10);

        let single_base_match = match_for_query(&request, "chr1:5-5");
        let single_base_span = span_for_match(&request, &single_base_match);
        assert_eq!(single_base_span.segments.len(), 1);
        assert_eq!(single_base_span.segments[0].start, 4);
        assert_eq!(single_base_span.segments[0].end, 5);

        let model_match = match_for_query(&request, "model-gene:5-10");
        assert_eq!(model_match.label, "model-gene:5-10 (model annotation)");
        let model_span = span_for_match(&request, &model_match);
        assert_eq!(model_span.segments.len(), 1);
        assert_eq!(model_span.segments[0].start, 9);
        assert_eq!(model_span.segments[0].end, 15);

        let model_slice_match = match_for_query(&request, "model-gene:1-4");
        let model_slice_span = span_for_match(&request, &model_slice_match);
        assert_eq!(model_slice_span.segments.len(), 1);
        assert_eq!(model_slice_span.segments[0].start, 5);
        assert_eq!(model_slice_span.segments[0].end, 9);

        let model_single_base_match = match_for_query(&request, "model-gene:5-5");
        let model_single_base_span = span_for_match(&request, &model_single_base_match);
        assert_eq!(model_single_base_span.segments[0].start, 9);
        assert_eq!(model_single_base_span.segments[0].end, 10);

        let block_group_match = match_for_query(&request, "m123:20-30");
        assert_eq!(block_group_match.label, "m123:20-30 (block group)");
        let block_group_span = span_for_match(&request, &block_group_match);
        assert_eq!(block_group_span.segments.len(), 1);
        assert_eq!(block_group_span.segments[0].start, 19);
        assert_eq!(block_group_span.segments[0].end, 30);

        let zero_match = match_for_query(&request, "chr1:0-1");
        let zero_span = span_for_match(&request, &zero_match);
        assert_eq!(zero_span.segments[0].start, 0);
        assert_eq!(zero_span.segments[0].end, 1);
    }

    #[test]
    fn test_region_search_activates_zero_and_negative_points() {
        let request = search_request_fixture();
        let zero_match = match_for_query(&request, "chr1:0");
        let mut zero_view_state = GraphViewState::default();
        let mut zero_overlays = Vec::new();

        activate_search_match(
            &mut zero_view_state,
            &request.graph,
            &mut zero_overlays,
            &zero_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("zero coordinate should navigate to the graph boundary");
        assert!(zero_view_state.cursor.node.is_some());
        assert_eq!(zero_overlays.len(), 1);
        assert!(matches!(
            &zero_overlays[0].content,
            OverlayContent::Span(span) if span.segments.is_empty()
        ));

        let negative_match = match_for_query(&request, "model-gene:-3");
        let mut negative_view_state = GraphViewState::default();
        let mut negative_overlays = Vec::new();
        activate_search_match(
            &mut negative_view_state,
            &request.graph,
            &mut negative_overlays,
            &negative_match,
            request.context.graph().conn(),
            request.workspace(),
        )
        .expect("negative model annotation coordinate should navigate upstream");
        assert!(negative_view_state.cursor.node.is_some());
        assert_eq!(negative_overlays.len(), 1);
        assert!(matches!(
            &negative_overlays[0].content,
            OverlayContent::Span(span) if span.segments.is_empty()
        ));
    }

    #[test]
    fn test_user_region_coordinates_translate_to_zero_based_half_open() {
        assert_eq!(
            translate_user_region(&Region::parse("foo:5-10").expect("should parse range")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: Some(10),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:5-5").expect("should parse single base")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: Some(5),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:5").expect("should parse point")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: Some(5),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:5..").expect("should parse open range")),
            Region {
                name: "foo".to_string(),
                start: Some(4),
                end: None,
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo").expect("should parse name")),
            Region {
                name: "foo".to_string(),
                start: None,
                end: None,
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:0").expect("should parse zero")),
            Region {
                name: "foo".to_string(),
                start: Some(0),
                end: Some(0),
            }
        );
        assert_eq!(
            translate_user_region(&Region::parse("foo:0-5").expect("should parse mixed range")),
            Region {
                name: "foo".to_string(),
                start: Some(0),
                end: Some(5),
            }
        );
        assert_eq!(
            translate_user_region(
                &Region::parse("foo:-5--1").expect("should parse negative range")
            ),
            Region {
                name: "foo".to_string(),
                start: Some(-5),
                end: Some(-1),
            }
        );
    }

    #[test]
    fn test_region_search_resolves_ambiguous_matches() {
        let request = search_request_fixture();
        let same_name_matches = resolve_region_search_matches(&request.request(), "m123")
            .expect("should return block group and path matches with the same name");
        assert_eq!(same_name_matches.len(), 2);
        assert!(
            same_name_matches
                .iter()
                .any(|search_match| search_match.label.contains("block group"))
        );
        assert!(
            same_name_matches
                .iter()
                .any(|search_match| search_match.label.contains("path"))
        );

        let duplicate_matches = resolve_region_search_matches(&request.request(), "duplicate-gene")
            .expect("should return ambiguous model annotation matches");
        assert_eq!(duplicate_matches.len(), 2);
    }

    #[test]
    fn test_remove_search_overlay_preserves_path_and_track_overlays() {
        let span = AnnotationSpan {
            id: gen_core::HashId::convert_str("overlay-span"),
            name: "overlay".to_string(),
            segments: vec![],
        };
        let mut overlays = vec![
            GraphOverlay {
                content: OverlayContent::Span(span.clone()),
                source: OverlaySource::Track("gff".to_string()),
                style: PathStyle::new(Color::Cyan),
            },
            GraphOverlay {
                content: OverlayContent::Path(PathMembership::default()),
                source: OverlaySource::Path,
                style: PathStyle::new(Color::Blue),
            },
            GraphOverlay {
                content: OverlayContent::Span(span),
                source: OverlaySource::Adhoc,
                style: PathStyle::new(Color::Yellow),
            },
            GraphOverlay {
                content: OverlayContent::Span(AnnotationSpan {
                    id: gen_core::HashId::convert_str("search-span"),
                    name: "search".to_string(),
                    segments: vec![],
                }),
                source: OverlaySource::Search,
                style: PathStyle::new(Color::LightYellow),
            },
        ];

        remove_search_overlay(&mut overlays);

        assert_eq!(overlays.len(), 3);
        assert!(
            overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Track(_)))
        );
        assert!(
            overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Path))
        );
        assert!(
            overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Adhoc))
        );
        assert!(
            !overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Search))
        );
    }
}
