//! Shared relative-coordinate resolution after annotation lookup.
//!
//! Applications choose the source and perform identifier lookup. The source then supplies a
//! translated, accession-style interval tree through [`AnnotationRegionSource`]. Persisted,
//! GFF, and BED annotations consequently use one graph-position implementation; this module
//! does not know how a file was selected or opened.

use gen_core::{HashId, NodeIntervalBlock, Strand, Workspace, region::Region};
use gen_graph::{GenGraph, GraphError, GraphNodePosition};
use gen_models::{
    db::GraphConnection,
    region::{
        AnnotationGraphPositionRequest, compute_annotation_graph_positions,
        compute_annotation_graph_positions_with_expansion,
    },
};
use intervaltree::IntervalTree;
use thiserror::Error;

/// Source-neutral translated annotation data.
///
/// The interval tree is zero-based, half-open, and cumulative from `0` to the annotation's
/// feature length, matching persisted accession interval trees. The values retain node sequence
/// coordinates and strand metadata for graph anchor resolution. For reverse-strand annotations,
/// relative offset zero is resolved from the rightmost interval in this source/path order.
#[derive(Clone, Debug)]
pub struct AnnotationRegionData {
    /// Translated node-space coordinates concatenated from zero to feature length.
    pub interval_tree: IntervalTree<i64, NodeIntervalBlock>,
    /// Block group providing graph topology for the annotation.
    pub block_group_id: HashId,
}

/// A lookup source that can provide one translated annotation after the application has selected
/// its source. The associated context lets database-backed annotations use a connection while
/// file-backed annotations can use the unit context after their selected records were translated.
pub trait AnnotationRegionSource {
    type Context;
    type Error: std::error::Error + 'static;

    fn annotation_region(
        &self,
        context: &Self::Context,
    ) -> Result<AnnotationRegionData, Self::Error>;
}

/// Resolved annotation-relative offsets and graph anchors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedAnnotationRegion {
    /// Uniform source annotation strand.
    pub strand: Strand,
    /// Normalized annotation-relative start offset.
    pub start_offset: i64,
    /// Normalized annotation-relative end offset.
    pub end_offset: i64,
    /// All graph positions resolved at the start boundary.
    pub start_anchors: Vec<GraphNodePosition>,
    /// All graph positions resolved at the end boundary.
    pub end_anchors: Vec<GraphNodePosition>,
}

/// Errors produced while resolving a translated annotation source.
#[derive(Debug, Error)]
pub enum AnnotationRegionError<E: std::error::Error + 'static> {
    #[error("annotation source error: {0}")]
    Source(#[source] E),
    #[error("annotation has no translated intervals")]
    EmptyAnnotation,
    #[error("annotation intervals have mixed strands")]
    MixedStrands,
    #[error("annotation strand is not directional")]
    NonDirectionalStrand,
    #[error("annotation region start {start} is greater than end {end}")]
    InvalidRange { start: i64, end: i64 },
    #[error(transparent)]
    Parse(#[from] gen_core::region::RegionParseError),
    #[error(transparent)]
    Graph(#[from] GraphError),
}

/// Resolve an already-normalized region from a source-backed annotation.
///
/// Raw user-facing coordinates must first pass through
/// [`gen_core::region::normalize_user_search_region`]. This function intentionally accepts only
/// the normalized form so positive coordinates cannot be converted twice.
pub fn resolve_annotation_region<S>(
    normalized_region: &Region,
    source: &S,
    source_context: &S::Context,
    conn: &GraphConnection,
    workspace: &Workspace,
    graph: &mut GenGraph,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError<S::Error>>
where
    S: AnnotationRegionSource,
{
    let data = source
        .annotation_region(source_context)
        .map_err(AnnotationRegionError::Source)?;
    resolve_annotation_region_data(
        normalized_region,
        data,
        conn,
        workspace,
        graph,
        compute_annotation_graph_positions,
    )
}

/// Resolve an already-normalized region while delegating graph expansion to the caller.
///
/// Raw user-facing coordinates must first pass through
/// [`gen_core::region::normalize_user_search_region`]. Source lookup, file access, and expansion
/// remain owned by the caller; the callback is passed through to the shared model graph walker.
pub fn resolve_annotation_region_with_expansion<S, F>(
    normalized_region: &Region,
    source: &S,
    source_context: &S::Context,
    conn: &GraphConnection,
    workspace: &Workspace,
    graph: &mut GenGraph,
    mut expand: F,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError<S::Error>>
where
    S: AnnotationRegionSource,
    F: FnMut(&mut GenGraph, HashId) -> bool,
{
    let data = source
        .annotation_region(source_context)
        .map_err(AnnotationRegionError::Source)?;
    resolve_annotation_region_data(
        normalized_region,
        data,
        conn,
        workspace,
        graph,
        |request, conn, workspace| {
            compute_annotation_graph_positions_with_expansion(request, conn, workspace, &mut expand)
        },
    )
}

fn resolve_annotation_region_data<E, F>(
    normalized_region: &Region,
    data: AnnotationRegionData,
    conn: &GraphConnection,
    workspace: &Workspace,
    graph: &mut GenGraph,
    resolve_positions: F,
) -> Result<ResolvedAnnotationRegion, AnnotationRegionError<E>>
where
    E: std::error::Error + 'static,
    F: FnOnce(
        AnnotationGraphPositionRequest<'_>,
        &GraphConnection,
        &Workspace,
    ) -> Result<(Vec<GraphNodePosition>, Vec<GraphNodePosition>), GraphError>,
{
    let (strand, feature_length) =
        annotation_shape(&data.interval_tree).map_err(|error| match error {
            AnnotationShapeError::Empty => AnnotationRegionError::EmptyAnnotation,
            AnnotationShapeError::MixedStrands => AnnotationRegionError::MixedStrands,
            AnnotationShapeError::NonDirectional => AnnotationRegionError::NonDirectionalStrand,
        })?;
    let (start_offset, end_offset) =
        normalized_region.resolve_relative_bounds(0, feature_length)?;
    if start_offset > end_offset {
        return Err(AnnotationRegionError::InvalidRange {
            start: start_offset,
            end: end_offset,
        });
    }

    let (start_anchors, end_anchors) = resolve_positions(
        AnnotationGraphPositionRequest {
            graph,
            interval_tree: &data.interval_tree,
            block_group_id: data.block_group_id,
            strand,
            feature_length,
            start_offset,
            end_offset,
        },
        conn,
        workspace,
    )?;

    Ok(ResolvedAnnotationRegion {
        strand,
        start_offset,
        end_offset,
        start_anchors,
        end_anchors,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AnnotationShapeError {
    Empty,
    MixedStrands,
    NonDirectional,
}

fn annotation_shape(
    interval_tree: &IntervalTree<i64, NodeIntervalBlock>,
) -> Result<(Strand, i64), AnnotationShapeError> {
    let mut strand = None;
    let mut feature_length = 0;
    for item in interval_tree.iter() {
        if gen_core::is_terminal(item.value.node_id) {
            continue;
        }
        if Strand::is_ambiguous(item.value.strand) {
            return Err(AnnotationShapeError::NonDirectional);
        }
        match strand {
            Some(previous) if previous != item.value.strand => {
                return Err(AnnotationShapeError::MixedStrands);
            }
            None => strand = Some(item.value.strand),
            Some(_) => {}
        }
        feature_length = feature_length.max(item.value.end);
    }
    let strand = strand.ok_or(AnnotationShapeError::Empty)?;
    Ok((strand, feature_length))
}

#[cfg(test)]
mod tests {
    use gen_core::{HashId, NodeIntervalBlock, Strand, region::normalize_user_search_region};
    use gen_graph::{GraphEdge, GraphNode};

    use super::{
        AnnotationRegionData, AnnotationRegionSource, resolve_annotation_region,
        resolve_annotation_region_with_expansion,
    };
    use crate::source::{AnnotationTranslationContext, BedAnnotation, GffAnnotation};

    struct TestSource {
        data: AnnotationRegionData,
    }

    impl AnnotationRegionSource for TestSource {
        type Context = ();
        type Error = std::convert::Infallible;

        fn annotation_region(
            &self,
            _context: &Self::Context,
        ) -> Result<AnnotationRegionData, Self::Error> {
            Ok(self.data.clone())
        }
    }

    fn source(
        node_name: &str,
        sequence_start: i64,
        sequence_end: i64,
        strand: Strand,
    ) -> TestSource {
        let node_id = HashId::convert_str(node_name);
        let block = NodeIntervalBlock {
            node_id,
            start: 0,
            end: sequence_end - sequence_start,
            sequence_start,
            sequence_end,
            strand,
        };
        TestSource {
            data: AnnotationRegionData {
                interval_tree: vec![(block.start..block.end, block)].into_iter().collect(),
                block_group_id: HashId::convert_str("block-group"),
            },
        }
    }

    fn graph(node_name: &str, sequence_start: i64, sequence_end: i64) -> gen_graph::GenGraph {
        let graph_node = GraphNode {
            node_id: HashId::convert_str(node_name),
            sequence_start,
            sequence_end,
        };
        let mut graph = gen_graph::GenGraph::new();
        graph.add_node(graph_node);
        graph
    }

    fn normalized(region: &str) -> gen_core::region::Region {
        normalize_user_search_region(&gen_core::region::Region::parse(region).unwrap())
    }

    fn connection() -> gen_models::db::GraphConnection {
        crate::test_helpers::get_connection()
    }

    #[test]
    fn test_resolves_zero_and_positive_offsets_through_source_tree() {
        let source = source("node", 4, 20, Strand::Forward);
        let context = ();
        let conn = connection();
        let workspace = crate::test_helpers::test_workspace();
        let mut graph = graph("node", 0, 34);

        let zero = resolve_annotation_region(
            &normalized("gene:0"),
            &source,
            &context,
            &conn,
            workspace,
            &mut graph,
        )
        .unwrap();
        assert_eq!((zero.start_offset, zero.end_offset), (0, 0));
        assert_eq!(zero.start_anchors, zero.end_anchors);
        assert_eq!(zero.start_anchors[0].coordinate(), 4);

        let positive = resolve_annotation_region(
            &normalized("gene:5-8"),
            &source,
            &context,
            &conn,
            workspace,
            &mut graph,
        )
        .unwrap();
        assert_eq!((positive.start_offset, positive.end_offset), (4, 8));
        assert_eq!(positive.start_anchors[0].coordinate(), 8);
        assert_eq!(positive.end_anchors[0].coordinate(), 12);
    }

    #[test]
    fn test_resolves_forward_and_reverse_offsets_outside_feature() {
        let conn = connection();
        let workspace = crate::test_helpers::test_workspace();
        let context = ();

        let forward = source("node", 4, 20, Strand::Forward);
        let mut forward_graph = graph("node", 0, 34);
        let forward_region = resolve_annotation_region(
            &normalized("gene:-3"),
            &forward,
            &context,
            &conn,
            workspace,
            &mut forward_graph,
        )
        .unwrap();
        assert_eq!(forward_region.start_anchors[0].coordinate(), 1);

        let reverse = source("node", 4, 20, Strand::Reverse);
        let mut reverse_graph = graph("node", 0, 34);
        let reverse_region = resolve_annotation_region(
            &normalized("gene:-3"),
            &reverse,
            &context,
            &conn,
            workspace,
            &mut reverse_graph,
        )
        .unwrap();
        assert_eq!(reverse_region.start_anchors[0].coordinate(), 23);

        let reverse_after = resolve_annotation_region(
            &normalized("gene:17"),
            &reverse,
            &context,
            &conn,
            workspace,
            &mut reverse_graph,
        )
        .unwrap();
        assert_eq!(reverse_after.end_anchors[0].coordinate(), 3);
    }

    #[test]
    fn test_resolves_branch_boundaries_and_expansion_callback() {
        let source = source("first", 0, 4, Strand::Forward);
        let context = ();
        let conn = connection();
        let workspace = crate::test_helpers::test_workspace();
        let first = GraphNode {
            node_id: HashId::convert_str("first"),
            sequence_start: 0,
            sequence_end: 4,
        };
        let left = GraphNode {
            node_id: HashId::convert_str("left"),
            sequence_start: 0,
            sequence_end: 6,
        };
        let right = GraphNode {
            node_id: HashId::convert_str("right"),
            sequence_start: 0,
            sequence_end: 6,
        };
        let mut graph = gen_graph::GenGraph::new();
        graph.add_node(first);
        let mut expanded = false;
        let resolved = resolve_annotation_region_with_expansion(
            &normalized("gene:5-8"),
            &source,
            &context,
            &conn,
            workspace,
            &mut graph,
            |graph, node_id| {
                if !expanded && node_id == first.node_id {
                    graph.add_edge(first, left, Vec::<GraphEdge>::new());
                    graph.add_edge(first, right, Vec::<GraphEdge>::new());
                    expanded = true;
                    true
                } else {
                    false
                }
            },
        )
        .unwrap();
        assert!(expanded);
        assert_eq!(resolved.end_anchors.len(), 2);
        assert!(resolved.end_anchors.iter().any(|position| {
            position.graph_node.node_id == left.node_id && position.offset == 4
        }));
        assert!(resolved.end_anchors.iter().any(|position| {
            position.graph_node.node_id == right.node_id && position.offset == 4
        }));
    }

    #[test]
    fn test_resolves_preloaded_branch_from_annotation_boundary_node() {
        let source = source("first", 0, 4, Strand::Forward);
        let context = ();
        let conn = connection();
        let workspace = crate::test_helpers::test_workspace();
        let first = GraphNode {
            node_id: HashId::convert_str("first"),
            sequence_start: 0,
            sequence_end: 4,
        };
        let left = GraphNode {
            node_id: HashId::convert_str("left"),
            sequence_start: 0,
            sequence_end: 6,
        };
        let right = GraphNode {
            node_id: HashId::convert_str("right"),
            sequence_start: 0,
            sequence_end: 6,
        };
        let mut graph = gen_graph::GenGraph::new();
        graph.add_edge(first, left, Vec::<GraphEdge>::new());
        graph.add_edge(first, right, Vec::<GraphEdge>::new());

        let resolved = resolve_annotation_region(
            &normalized("gene:5-8"),
            &source,
            &context,
            &conn,
            workspace,
            &mut graph,
        )
        .unwrap();
        assert_eq!(resolved.end_anchors.len(), 2);
        assert!(resolved.end_anchors.iter().any(|position| {
            position.graph_node.node_id == left.node_id && position.offset == 4
        }));
        assert!(resolved.end_anchors.iter().any(|position| {
            position.graph_node.node_id == right.node_id && position.offset == 4
        }));
    }

    #[test]
    fn test_resolves_ranges_across_discontinuous_node_slices() {
        let first = NodeIntervalBlock {
            node_id: HashId::convert_str("first-slice"),
            start: 0,
            end: 3,
            sequence_start: 0,
            sequence_end: 3,
            strand: Strand::Forward,
        };
        let second = NodeIntervalBlock {
            node_id: HashId::convert_str("second-slice"),
            start: 3,
            end: 6,
            sequence_start: 10,
            sequence_end: 13,
            strand: Strand::Forward,
        };
        let source = TestSource {
            data: AnnotationRegionData {
                interval_tree: vec![
                    (first.start..first.end, first),
                    (second.start..second.end, second),
                ]
                .into_iter()
                .collect(),
                block_group_id: HashId::convert_str("block-group"),
            },
        };
        let mut graph = gen_graph::GenGraph::new();
        let first_node = GraphNode {
            node_id: first.node_id,
            sequence_start: 0,
            sequence_end: 3,
        };
        let second_node = GraphNode {
            node_id: second.node_id,
            sequence_start: 10,
            sequence_end: 13,
        };
        graph.add_edge(first_node, second_node, Vec::<GraphEdge>::new());
        let conn = connection();
        let workspace = crate::test_helpers::test_workspace();
        let context = ();
        let resolved = resolve_annotation_region(
            &normalized("gene:2-5"),
            &source,
            &context,
            &conn,
            workspace,
            &mut graph,
        )
        .unwrap();
        assert_eq!(resolved.start_anchors[0].coordinate(), 1);
        assert_eq!(resolved.end_anchors[0].coordinate(), 12);

        let point = resolve_annotation_region(
            &gen_core::region::Region::parse("gene:3").unwrap(),
            &source,
            &context,
            &conn,
            workspace,
            &mut graph,
        )
        .unwrap();
        assert_eq!(point.start_anchors, point.end_anchors);
        assert_eq!(point.start_anchors[0].coordinate(), 10);
    }

    #[test]
    fn test_gff_and_bed_annotations_match_identifiers_before_translation() {
        let conn = crate::test_helpers::get_connection();
        crate::test_helpers::setup_test_data(&conn);
        let block_group = gen_models::sample::Sample::get_block_groups(
            &conn,
            "test",
            gen_models::sample::Sample::DEFAULT_NAME,
            None,
        )
        .into_iter()
        .find(|block_group| block_group.name == "m123")
        .unwrap();
        let context = AnnotationTranslationContext {
            conn: &conn,
            workspace: crate::test_helpers::test_workspace(),
            collection_name: "test",
            sample_name: gen_models::sample::Sample::DEFAULT_NAME,
            history_ref: None,
            block_group_id: block_group.id,
        };
        let gff = GffAnnotation::from_reader(
            &context,
            "gene-a0001",
            std::io::BufReader::new(
                std::fs::File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.gff"))
                    .unwrap(),
            ),
        )
        .unwrap();
        let bed = BedAnnotation::from_reader(
            &context,
            "abc123.1",
            std::fs::File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/simple.bed"))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(gff.identifier(), "gene-a0001");
        assert_eq!(bed.identifier(), "abc123.1");
        let gff_data = gff.annotation_region(&()).unwrap();
        let bed_data = bed.annotation_region(&()).unwrap();
        assert_eq!(gff_data.interval_tree.iter().count(), 2);
        assert_eq!(bed_data.interval_tree.iter().count(), 1);

        let mut bed_graph = gen_graph::graph_from_interval_tree(&bed_data.interval_tree);
        let resolved_bed = resolve_annotation_region(
            &normalized("abc123.1:2-3"),
            &bed,
            &(),
            &conn,
            context.workspace,
            &mut bed_graph,
        )
        .unwrap();
        assert_eq!(resolved_bed.strand, Strand::Reverse);
        assert_eq!((resolved_bed.start_offset, resolved_bed.end_offset), (1, 3));
        assert!(!resolved_bed.start_anchors.is_empty());
        assert!(!resolved_bed.end_anchors.is_empty());
    }
}
