#[cfg(test)]
use std::path::PathBuf;

use gen_core::{Workspace, region::Region};
use gen_graph::{GenGraph, GraphNodePosition};
use gen_models::{
    accession::Accession,
    annotations::Annotation,
    block_group::BlockGroup,
    db::GraphConnection,
    path::Path,
    region::{GenRegionError, ResolvedGenRegion, ResolvedRegionKind},
};
#[cfg(test)]
use gen_models::{annotations::add_annotation, block_group::PathCache, sample::Sample};
use gen_tui::{LineStyle, graph_controller::GraphController, plotter::PathStyle};
use petgraph::{graph::NodeIndex, visit::NodeIndexable};
use ratatui::style::Color;

use crate::views::{
    annotation_track::{AnnotationSpan, annotation_span_from_resolved_region},
    gen_graph_widget::GenGraphNodeSizer,
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
    pub(super) current_block_group: &'a BlockGroup,
}

fn linear_region_bounds(region: &Region, feature_length: i64) -> Result<(i64, i64), String> {
    let bounds = match (region.start, region.end) {
        (None, None) => (0, feature_length),
        (Some(start), None) => (start, feature_length),
        (Some(start), Some(end)) => (start, end),
        (None, Some(_)) => return Err(format!("invalid region syntax: {region}")),
    };
    if bounds.0 < 0 || bounds.0 > bounds.1 || bounds.1 > feature_length {
        return Err(format!(
            "region {region} is outside the feature bounds (0-{feature_length})"
        ));
    }
    Ok(bounds)
}

fn annotation_region_bounds(region: &Region, feature_length: i64) -> Result<(i64, i64), String> {
    match (region.start, region.end) {
        (None, None) => Ok((0, feature_length)),
        (Some(start), None) => Ok((start, feature_length)),
        (Some(start), Some(end)) => Ok((start, end)),
        (None, Some(_)) => Err(format!("invalid region syntax: {region}")),
    }
}

fn path_region_bounds(
    conn: &GraphConnection,
    path: &Path,
    region: &Region,
) -> Result<(i64, i64), String> {
    let feature_length = path
        .length(conn, None)
        .map_err(|error| format!("failed to load path length: {error}"))?;
    linear_region_bounds(region, feature_length)
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

fn append_database_matches(
    request: &RegionSearchRequest<'_>,
    region: &Region,
    display_query: &str,
    matches: &mut Vec<RegionSearchMatch>,
) -> Result<(), String> {
    let block_groups = BlockGroup::select(request.conn)
        .collection_name(request.collection_name)
        .sample_name(request.sample_name)
        .name_case_insensitive(&region.name)
        .load()
        .map_err(|error| format!("failed to search block groups: {error}"))?;
    for block_group in block_groups {
        let path = BlockGroup::get_current_path(request.conn, &block_group.id, None)
            .map_err(|error| format!("failed to load block group path: {error}"))?;
        let (start, end) = path_region_bounds(request.conn, &path, region)?;
        let resolved = ResolvedGenRegion::from_block_group(request.conn, &block_group, start, end)
            .map_err(|error| format!("failed to resolve block group: {error}"))?;
        matches.push(RegionSearchMatch {
            label: resolved_match_label(display_query, resolved.kind),
            region: resolved,
        });
    }

    let paths = Path::select(request.conn)
        .name_case_insensitive(&region.name)
        .load()
        .map_err(|error| format!("failed to search paths: {error}"))?;
    for path in paths {
        let Ok(block_group) = BlockGroup::get_by_id(request.conn, &path.block_group_id, None)
        else {
            continue;
        };
        if block_group.collection_name != request.collection_name
            || block_group.sample_name != request.sample_name
        {
            continue;
        }
        let (start, end) = path_region_bounds(request.conn, &path, region)?;
        let resolved =
            ResolvedGenRegion::from_path(request.conn, block_group.id, &path, start, end)
                .map_err(|error| format!("failed to resolve path: {error}"))?;
        matches.push(RegionSearchMatch {
            label: resolved_match_label(display_query, resolved.kind),
            region: resolved,
        });
    }

    let annotations = Annotation::query_with_lineage(
        request.conn,
        request.collection_name,
        request.sample_name,
        &request.current_block_group.name,
    )
    .map_err(|error| format!("failed to search model annotations: {error}"))?;
    for annotation in annotations
        .into_iter()
        .filter(|annotation| annotation.name.eq_ignore_ascii_case(&region.name))
    {
        let Ok(accession) = Accession::select(request.conn)
            .id(annotation.accession_id)
            .load()
            .map(|mut accessions| accessions.pop())
        else {
            continue;
        };
        let Some(accession) = accession else {
            continue;
        };
        let feature_length = accession
            .length(request.conn)
            .map_err(|error| format!("failed to load annotation length: {error}"))?;
        let (start, end) = annotation_region_bounds(region, feature_length)?;
        let resolved =
            ResolvedGenRegion::from_annotation(request.conn, &annotation, &accession, start, end)
                .map_err(|error| format!("failed to resolve annotation: {error}"))?;
        matches.push(RegionSearchMatch {
            label: resolved_match_label(display_query, resolved.kind),
            region: resolved,
        });
    }
    Ok(())
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
    let mut matches = Vec::new();
    match gen_models::region::resolve(
        &region,
        request.conn,
        request.collection_name,
        request.sample_name,
    ) {
        Ok(resolved) => {
            append_database_matches(request, &region, query, &mut matches)?;
            if matches.is_empty() {
                matches.push(RegionSearchMatch {
                    label: resolved_match_label(query, resolved.kind),
                    region: resolved,
                });
            }
        }
        Err(GenRegionError::Ambiguous(_)) => {
            append_database_matches(request, &region, query, &mut matches)?
        }
        Err(GenRegionError::NotFound(_)) => {}
        Err(error) => return Err(error.to_string()),
    }
    if matches.is_empty() {
        return Err(format!("no region matched {query}"));
    }
    Ok(matches)
}

pub(super) fn search_destination_span(
    search_match: &RegionSearchMatch,
    conn: &GraphConnection,
    workspace: &Workspace,
) -> Result<AnnotationSpan, String> {
    annotation_span_from_resolved_region(conn, workspace, &search_match.region)
}

pub(super) fn replace_search_overlay(overlays: &mut Vec<GraphOverlay>, span: AnnotationSpan) {
    remove_search_overlay(overlays);
    overlays.push(GraphOverlay {
        content: OverlayContent::Span(span),
        source: OverlaySource::Adhoc,
        style: PathStyle::new(Color::LightYellow)
            .with_line_style(LineStyle::Bold)
            .with_merge_glyphs(true),
    });
}

pub(super) fn remove_search_overlay(overlays: &mut Vec<GraphOverlay>) {
    overlays.retain(|overlay| !matches!(&overlay.source, OverlaySource::Adhoc));
}

fn search_destination_position(
    search_match: &RegionSearchMatch,
    conn: &GraphConnection,
    workspace: &Workspace,
) -> Result<GraphNodePosition, String> {
    let region = &search_match.region;
    let midpoint = region
        .start
        .saturating_add(region.end.saturating_sub(region.start) / 2);
    let offset = midpoint.saturating_sub(region.start);
    let positioned = region
        .find_graph_positions(conn, workspace, offset, 0)
        .map_err(|error| format!("failed to locate region on graph: {error}"))?;
    positioned
        .start_anchors
        .and_then(|positions| positions.into_iter().next())
        .ok_or_else(|| "region did not map to a graph position".to_string())
}

fn remap_search_position(
    position: GraphNodePosition,
    graph: &GenGraph,
) -> Result<GraphNodePosition, String> {
    if graph.contains_node(position.graph_node) {
        return Ok(position);
    }

    let sequence_coordinate = position.coordinate();
    let candidates = graph
        .nodes()
        .filter(|node| node.node_id == position.graph_node.node_id)
        .collect::<Vec<_>>();
    let mapped_node = candidates
        .iter()
        .filter(|node| {
            node.sequence_start <= sequence_coordinate && sequence_coordinate < node.sequence_end
        })
        .min_by_key(|node| (node.sequence_end - node.sequence_start, node.sequence_start))
        .or_else(|| {
            candidates
                .iter()
                .filter(|node| node.sequence_end == sequence_coordinate)
                .min_by_key(|node| (node.sequence_end - node.sequence_start, node.sequence_start))
        })
        .or_else(|| {
            candidates
                .iter()
                .filter(|node| node.sequence_start == sequence_coordinate)
                .min_by_key(|node| (node.sequence_end - node.sequence_start, node.sequence_start))
        })
        .copied()
        .ok_or_else(|| format!("graph node {} was not found", position.graph_node.node_id))?;
    Ok(GraphNodePosition {
        graph_node: mapped_node,
        offset: (sequence_coordinate - mapped_node.sequence_start).clamp(0, mapped_node.length()),
    })
}

pub(super) fn go_to_search_match(
    graph_controller: &mut GraphController<GenGraph, GenGraphNodeSizer>,
    search_match: &RegionSearchMatch,
    conn: &GraphConnection,
    workspace: &Workspace,
) -> Result<(), String> {
    let position = search_destination_position(search_match, conn, workspace)?;
    let position = remap_search_position(position, graph_controller.graph())?;
    let node_index = NodeIndex::new(<GenGraph as NodeIndexable>::to_index(
        graph_controller.graph(),
        position.graph_node,
    ));
    let node_length = position.graph_node.length();
    let fraction = if node_length == 0 {
        0.5
    } else {
        (position.offset as f64 / node_length as f64).clamp(0.0, 1.0)
    };
    graph_controller.go_to_node(node_index, (fraction, 0.5));
    Ok(())
}

#[cfg(test)]
pub(super) struct RegionSearchFixture {
    pub(super) request: RegionSearchRequest<'static>,
    pub(super) workspace: &'static Workspace,
    pub(super) graph: &'static GenGraph,
}

#[cfg(test)]
pub(super) fn search_request_fixture() -> RegionSearchFixture {
    let context = Box::leak(Box::new(setup_gen_on_disk()));
    let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
    let fasta_path = fasta_path
        .to_str()
        .expect("should encode FASTA fixture path")
        .to_string();
    import_fasta(context, &fasta_path, "test", "simple", false, &[])
        .expect("should import search fixture FASTA");
    add_annotation(context, "test", "model-gene", None, "simple", "m123:5-20")
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
    let block_group = Box::leak(Box::new(block_group));
    let graph = Box::leak(Box::new(
        BlockGroup::get_graph(conn, context.workspace(), &block_group.id, None)
            .expect("should load search graph"),
    ));
    RegionSearchFixture {
        request: RegionSearchRequest {
            conn,
            collection_name: "test",
            sample_name: "simple",
            current_block_group: block_group,
        },
        workspace: context.workspace(),
        graph,
    }
}

#[cfg(test)]
mod tests {
    use gen_core::region::Region;
    use gen_graph::{GenGraph, GraphNodePosition};
    use gen_tui::plotter::PathStyle;
    use petgraph::visit::NodeIndexable;
    use ratatui::style::Color;

    use super::{
        RegionSearchFixture, RegionSearchMatch, go_to_search_match, remove_search_overlay,
        resolve_region_search_matches, search_destination_position, search_request_fixture,
        translate_user_region,
    };
    use crate::views::{
        annotation_track::{AnnotationSpan, annotation_span_from_resolved_region},
        gen_graph_widget::create_gen_graph_controller,
        graph_overlay::{GraphOverlay, OverlayContent, OverlaySource},
    };

    fn position_for_query(
        fixture: &RegionSearchFixture,
        query: &str,
    ) -> (Vec<RegionSearchMatch>, GraphNodePosition) {
        let matches =
            resolve_region_search_matches(&fixture.request, query).expect("should resolve query");
        let position =
            search_destination_position(&matches[0], fixture.request.conn, fixture.workspace)
                .expect("should map query to graph position");
        (matches, position)
    }

    fn span_for_query(fixture: &RegionSearchFixture, query: &str) -> AnnotationSpan {
        let matches =
            resolve_region_search_matches(&fixture.request, query).expect("should resolve query");
        annotation_span_from_resolved_region(
            fixture.request.conn,
            fixture.workspace,
            &matches[0].region,
        )
        .expect("should map resolved region to an annotation span")
    }

    #[test]
    fn test_region_search_locates_block_groups_paths_and_model_annotations() {
        let request = search_request_fixture();

        let (path_matches, path_position) = position_for_query(&request, "chr1:20-30");
        assert_eq!(
            path_matches.len(),
            1,
            "path query should have one resolver match"
        );
        assert_eq!(path_position.graph_node.sequence_start, 0);
        assert_eq!(path_matches[0].label, "chr1:20-30 (path)");
        assert_eq!(path_position.offset, 24);
        let path_span = span_for_query(&request, "chr1:20-30");
        assert_eq!(path_span.segments.len(), 1);
        assert_eq!(path_span.segments[0].start, 19);
        assert_eq!(path_span.segments[0].end, 30);

        let mut controller = create_gen_graph_controller(request.graph.clone());
        go_to_search_match(
            &mut controller,
            &path_matches[0],
            request.request.conn,
            request.workspace,
        )
        .expect("should select and center the path match");
        let selected_index = controller
            .cursor
            .node_idx()
            .expect("should select the resolved graph node");
        let selected_node =
            <&GenGraph as NodeIndexable>::from_index(&controller.graph(), selected_index.index());
        assert_eq!(selected_node, path_position.graph_node);
        let (selected_fraction, _) = controller.cursor.fractional_pos();
        assert!((selected_fraction - (24.0 / 34.0)).abs() < f64::EPSILON);

        let (_path_slice_matches, path_slice_position) = position_for_query(&request, "chr1:5-10");
        assert_eq!(path_slice_position.offset, 7);
        let path_slice_span = span_for_query(&request, "chr1:5-10");
        assert_eq!(path_slice_span.segments.len(), 1);
        assert_eq!(
            path_slice_span.segments[0].node_id,
            path_slice_position.graph_node.node_id
        );
        assert_eq!(path_slice_span.segments[0].start, 4);
        assert_eq!(path_slice_span.segments[0].end, 10);

        let (_single_base_matches, single_base_position) = position_for_query(&request, "chr1:5-5");
        assert_eq!(single_base_position.offset, 4);
        let single_base_span = span_for_query(&request, "chr1:5-5");
        assert_eq!(single_base_span.segments.len(), 1);
        assert_eq!(single_base_span.segments[0].start, 4);
        assert_eq!(single_base_span.segments[0].end, 5);

        let (model_matches, model_position) = position_for_query(&request, "model-gene:5-10");
        assert_eq!(
            model_matches.len(),
            1,
            "model annotation should resolve uniquely"
        );
        assert_eq!(model_position.graph_node.sequence_start, 5);
        assert_eq!(model_position.offset, 7);
        assert_eq!(model_matches[0].label, "model-gene:5-10 (model annotation)");
        let model_span = span_for_query(&request, "model-gene:5-10");
        assert_eq!(model_span.segments.len(), 1);
        assert_eq!(model_span.segments[0].start, 9);
        assert_eq!(model_span.segments[0].end, 15);

        let (_model_slice_matches, model_slice_position) =
            position_for_query(&request, "model-gene:1-4");
        assert_eq!(model_slice_position.graph_node.sequence_start, 5);
        assert_eq!(model_slice_position.offset, 2);
        let model_slice_span = span_for_query(&request, "model-gene:1-4");
        assert_eq!(model_slice_span.segments.len(), 1);
        assert_eq!(
            model_slice_span.segments[0].node_id,
            model_slice_position.graph_node.node_id
        );
        assert_eq!(model_slice_span.segments[0].start, 5);
        assert_eq!(model_slice_span.segments[0].end, 9);

        let (_model_single_base_matches, model_single_base_position) =
            position_for_query(&request, "model-gene:5-5");
        assert_eq!(model_single_base_position.offset, 4);
        let model_single_base_span = span_for_query(&request, "model-gene:5-5");
        assert_eq!(model_single_base_span.segments[0].start, 9);
        assert_eq!(model_single_base_span.segments[0].end, 10);

        let (block_group_matches, block_group_position) =
            position_for_query(&request, "m123:20-30");
        assert_eq!(block_group_matches[0].label, "m123:20-30 (block group)");
        assert_eq!(block_group_position.offset, 24);
        let block_group_span = span_for_query(&request, "m123:20-30");
        assert_eq!(block_group_span.segments.len(), 1);
        assert_eq!(block_group_span.segments[0].start, 19);
        assert_eq!(block_group_span.segments[0].end, 30);

        let (zero_matches, zero_position) = position_for_query(&request, "chr1:0-1");
        assert_eq!(zero_matches.len(), 1);
        assert_eq!(zero_position.offset, 0);
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
        let same_name_matches = resolve_region_search_matches(&request.request, "m123")
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

        let duplicate_matches = resolve_region_search_matches(&request.request, "duplicate-gene")
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
                content: OverlayContent::Path(vec![]),
                source: OverlaySource::Path,
                style: PathStyle::new(Color::Blue),
            },
            GraphOverlay {
                content: OverlayContent::Span(span),
                source: OverlaySource::Adhoc,
                style: PathStyle::new(Color::Yellow),
            },
        ];

        remove_search_overlay(&mut overlays);

        assert_eq!(overlays.len(), 2);
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
            !overlays
                .iter()
                .any(|overlay| matches!(overlay.source, OverlaySource::Adhoc))
        );
    }

    #[test]
    fn test_search_position_remaps_resolver_node_to_displayed_slice() {
        let node_id = gen_core::HashId::convert_str("m123-node");
        let resolver_node = gen_graph::GraphNode {
            node_id,
            sequence_start: 0,
            sequence_end: 34,
        };
        let displayed_node = gen_graph::GraphNode {
            node_id,
            sequence_start: 2,
            sequence_end: 10,
        };
        let mut graph = GenGraph::new();
        graph.add_node(displayed_node);

        let mapped = super::remap_search_position(
            GraphNodePosition {
                graph_node: resolver_node,
                offset: 3,
            },
            &graph,
        )
        .expect("should map a resolver position onto a displayed node slice");
        assert_eq!(mapped.graph_node, displayed_node);
        assert_eq!(mapped.offset, 1);

        let mut unrelated_graph = GenGraph::new();
        unrelated_graph.add_node(gen_graph::GraphNode {
            node_id: gen_core::HashId::convert_str("other-node"),
            sequence_start: 2,
            sequence_end: 10,
        });
        assert!(
            super::remap_search_position(
                GraphNodePosition {
                    graph_node: resolver_node,
                    offset: 3,
                },
                &unrelated_graph,
            )
            .is_err(),
            "a resolver position must not map onto an unrelated node"
        );
    }
}
