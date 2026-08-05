use core::ops::Range;

use gen_core::Strand;
use gen_models::{
    accession::{Accession, AccessionSpan, NewAccession},
    block_group::BlockGroup,
    block_group_edge::{BlockGroupEdge, BlockGroupEdgeData},
    db::DbContext,
    edge::Edge,
    errors::{BlockGroupError, OperationError, PathError, SampleError},
    file_types::FileTypes,
    operations::{OperationFile, OperationInfo, OperationSummary},
    region::{GenRegionError, Region, ResolvedGenRegion, ResolvedRegionKind},
    sample::Sample,
};
use thiserror::Error;

use crate::{
    graphs::{
        BlockGroupChunk, NodePoint,
        combinatorial_library::{
            CombinatorialLibraryCreationError, CombinatorialLibraryParseError, SequencePart,
            create_library, create_part_annotations,
        },
        operators::{GraphOperationError, derive_chunks, make_stitch_from_block_groups},
        stitch,
    },
    updates::resolve_update_region,
};

/// Creates a top-level accession anchored at a single resolved graph
/// position where a library update was spliced in. Each part placed by that
/// update becomes a child accession (see `create_part_annotations`), so
/// reusing a part name/sequence across separate update calls on the same
/// block group never collides: siblings only need to be unique relative to
/// this location accession, not block-group-wide.
///
/// This is anchored at a resolved graph position rather than built via
/// `AccessionSpan::from_resolved_region`, because region bounds for
/// Annotation/Accession-kind regions can extend past the region's own
/// intervaltree (e.g. `foobar:-3-5`, 3bp upstream of annotation `foobar`),
/// which only `find_graph_positions`' graph walk resolves correctly.
///
/// `find_graph_positions` can return more than one position when a splice
/// site sits past a branch point, but at the zero offsets used by both call
/// sites below it always resolves through `ResolvedGraph::resolve_anchor`,
/// which already collapses same-coordinate branches down to a single
/// arbitrary match before any walk happens — so taking `positions[0]` at
/// those call sites doesn't lose anything beyond what's already lost
/// upstream. Fixing that collapse would mean changing `resolve_anchor`
/// itself (pre-existing, merged, shared by every region resolution in the
/// system), which is out of scope here.
fn create_location_accession(
    conn: &gen_models::db::GraphConnection,
    block_group_id: gen_core::HashId,
    position: &gen_graph::GraphNodePosition,
) -> Result<Accession, BlockGroupError> {
    let coordinate = position.coordinate();
    Ok(Accession::get_or_create(
        conn,
        &NewAccession {
            name: format!("{}:{coordinate}", position.graph_node.node_id),
            block_group_id,
            parent_accession_id: None,
            spans: vec![AccessionSpan {
                node_id: position.graph_node.node_id,
                range: gen_core::range::Range {
                    start: coordinate,
                    end: coordinate,
                },
                strand: Strand::Forward,
            }],
        },
    )?)
}

#[derive(Error, Debug)]
pub enum UpdateWithLibraryError {
    #[error("Failed to find block group")]
    BlockGroupLookupFailed(String),
    #[error("Failed to create block group")]
    BlockGroupCreationFailed(#[from] BlockGroupError),
    #[error("Failed to create output graph(s)")]
    GraphOperation(GraphOperationError),
    #[error("Graph error: {0}")]
    Graph(#[from] crate::graphs::GraphError),
    #[error("Graph resolution error: {0}")]
    GraphResolution(#[from] gen_graph::GraphError),
    #[error("Sample error: {0}")]
    Sample(#[from] SampleError),
    #[error("Failed to read path")]
    Path(#[from] PathError),
    #[error("Failed to parse library files")]
    FileParse(CombinatorialLibraryParseError),
    #[error("Failed to create library")]
    LibraryCreation(CombinatorialLibraryCreationError),
    #[error("Failed to resolve region")]
    Region(GenRegionError),
    #[error("Missing coordinates for region '{0}'. Use region syntax like 'name:start-end'.")]
    MissingCoordinates(String),
    #[error("Unsupported region type for library update: {0}")]
    UnsupportedRegionType(String),
    #[error("Operation error: {0}")]
    Operation(#[from] OperationError),
}

impl From<CombinatorialLibraryParseError> for UpdateWithLibraryError {
    fn from(err: CombinatorialLibraryParseError) -> Self {
        UpdateWithLibraryError::FileParse(err)
    }
}

impl From<GraphOperationError> for UpdateWithLibraryError {
    fn from(err: GraphOperationError) -> Self {
        UpdateWithLibraryError::GraphOperation(err)
    }
}

impl From<CombinatorialLibraryCreationError> for UpdateWithLibraryError {
    fn from(err: CombinatorialLibraryCreationError) -> Self {
        UpdateWithLibraryError::LibraryCreation(err)
    }
}

impl From<GenRegionError> for UpdateWithLibraryError {
    fn from(err: GenRegionError) -> Self {
        UpdateWithLibraryError::Region(err)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    all(debug_assertions, feature = "profiling"),
    tracing::instrument(skip(context, parts_list, library_file_path, parts_file_path))
)]
pub fn update_with_library(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    region_name: &str,
    parts_list: Vec<Vec<SequencePart>>,
    library_file_path: Option<&str>,
    parts_file_path: Option<&str>,
) -> Result<OperationSummary, UpdateWithLibraryError> {
    let conn = context.graph().conn();
    let parsed_region = Region::parse(region_name).map_err(GenRegionError::from)?;
    let resolved_region =
        resolve_update_region(&parsed_region, conn, collection_name, parent_sample_name)?;

    update_graph_with_library(
        context,
        collection_name,
        parent_sample_name,
        new_sample_name,
        &resolved_region,
        parts_list,
    )?;

    let files = operation_files(library_file_path, parts_file_path);
    let summary_str = format!("{region_name} created.\n");
    Ok(OperationSummary::new(
        OperationInfo {
            files,
            description: "library_csv_update".to_string(),
        },
        summary_str,
    ))
}

fn update_graph_with_library(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    resolved_region: &ResolvedGenRegion,
    parts_list: Vec<Vec<SequencePart>>,
) -> Result<(), UpdateWithLibraryError> {
    let conn = context.graph().conn();
    let target_block_groups = target_library_block_groups(
        conn,
        collection_name,
        parent_sample_name,
        new_sample_name,
        resolved_region,
    )?;

    for target_block_group in target_block_groups {
        let mut target_region = resolved_region.clone();
        target_region.block_group = target_block_group.clone();

        match target_region.kind {
            ResolvedRegionKind::Path => update_path_library(
                context,
                collection_name,
                parent_sample_name,
                new_sample_name,
                &target_region,
                &target_block_group,
                parts_list.clone(),
            )?,
            ResolvedRegionKind::Annotation | ResolvedRegionKind::Accession => {
                update_graph_native_library(
                    conn,
                    new_sample_name,
                    &target_region,
                    &target_block_group,
                    parts_list.clone(),
                )?;
            }
            kind => {
                return Err(UpdateWithLibraryError::UnsupportedRegionType(format!(
                    "{kind:?}"
                )));
            }
        }
    }

    Ok(())
}

fn target_library_block_groups(
    conn: &gen_models::db::GraphConnection,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    resolved_region: &ResolvedGenRegion,
) -> Result<Vec<BlockGroup>, UpdateWithLibraryError> {
    let _new_sample = Sample::get_or_create_child(
        conn,
        collection_name,
        new_sample_name,
        vec![parent_sample_name.to_string()],
    )?;
    let block_groups = Sample::get_block_groups(conn, collection_name, parent_sample_name, None);

    let mut target_block_groups = vec![];
    for block_group in block_groups {
        let new_block_groups = BlockGroup::get_or_create_sample_block_groups(
            conn,
            collection_name,
            new_sample_name,
            &block_group.name,
            vec![parent_sample_name.to_string()],
        )?;

        if block_group.name == resolved_region.block_group.name {
            target_block_groups = new_block_groups;
        }
    }

    if target_block_groups.is_empty() {
        return Err(UpdateWithLibraryError::Region(GenRegionError::NotFound(
            resolved_region.block_group.name.clone(),
        )));
    }

    Ok(target_block_groups)
}

fn update_path_library(
    context: &DbContext,
    collection_name: &str,
    parent_sample_name: &str,
    new_sample_name: &str,
    resolved_region: &ResolvedGenRegion,
    target_block_group: &BlockGroup,
    parts_list: Vec<Vec<SequencePart>>,
) -> Result<(), UpdateWithLibraryError> {
    let conn = context.graph().conn();
    let parent_path = resolved_region.path.clone().unwrap();
    let parent_path_length = parent_path.length(conn, None)?;
    let start_coordinate = resolved_region.start;
    let end_coordinate = resolved_region.end;
    let mut chunk_ranges = vec![];
    if start_coordinate > 0 {
        chunk_ranges.push(Range {
            start: 0,
            end: start_coordinate,
        });
    }
    chunk_ranges.push(Range {
        start: start_coordinate,
        end: end_coordinate,
    });
    if end_coordinate < parent_path_length {
        chunk_ranges.push(Range {
            start: end_coordinate,
            end: parent_path_length,
        });
    }

    let derived_block_group_chunks = derive_chunks(
        context,
        collection_name,
        parent_sample_name,
        new_sample_name,
        &resolved_region.block_group.name,
        None,
        chunk_ranges,
        Some(target_block_group.id),
        false,
    )?;

    let (library_block_group_chunk, part_nodes) = create_library(
        conn,
        target_block_group.id,
        new_sample_name,
        parts_list,
        false,
    )?;

    let resolved_with_positions =
        gen_graph::models::find_region_graph_positions(resolved_region, conn, 0, 0)
            .map_err(UpdateWithLibraryError::from)?;
    let splice_point = &resolved_with_positions.start_anchors.unwrap()[0];
    let location_accession = create_location_accession(conn, target_block_group.id, splice_point)?;
    create_part_annotations(
        conn,
        target_block_group.id,
        Some(location_accession.id),
        new_sample_name,
        new_sample_name,
        &part_nodes,
    )?;

    let mut block_group_chunks = vec![];
    let mut reference_block_group_chunks = vec![];
    let mut chunk_index = 0;

    if start_coordinate > 0 {
        let start_chunk = derived_block_group_chunks[0].clone();
        reference_block_group_chunks.push(start_chunk.clone());
        let pathless_start_chunk = BlockGroupChunk {
            entry_node_points: start_chunk.entry_node_points.clone(),
            exit_node_points: start_chunk.exit_node_points.clone(),
            path_edges: vec![],
            path_start_point: None,
            path_end_point: None,
        };
        block_group_chunks.push(pathless_start_chunk);

        chunk_index += 1;
    }

    reference_block_group_chunks.push(derived_block_group_chunks[chunk_index].clone());
    block_group_chunks.push(library_block_group_chunk);

    chunk_index += 1;

    if end_coordinate < parent_path_length {
        let end_chunk = derived_block_group_chunks[chunk_index].clone();
        reference_block_group_chunks.push(end_chunk.clone());
        let pathless_end_chunk = BlockGroupChunk {
            entry_node_points: end_chunk.entry_node_points.clone(),
            exit_node_points: end_chunk.exit_node_points.clone(),
            path_edges: vec![],
            path_start_point: None,
            path_end_point: None,
        };
        block_group_chunks.push(pathless_end_chunk);
    }

    // Create (re-create) the reference sequence/path out of the derived chunks,
    // in the child block group
    make_stitch_from_block_groups(
        context,
        &reference_block_group_chunks,
        target_block_group.id,
        new_sample_name,
    )?;

    // Stitch the library in between the first and last reference chunks
    make_stitch_from_block_groups(
        context,
        &block_group_chunks,
        target_block_group.id,
        new_sample_name,
    )?;

    Ok(())
}

fn update_graph_native_library(
    conn: &gen_models::db::GraphConnection,
    new_sample_name: &str,
    resolved_region: &ResolvedGenRegion,
    target_block_group: &BlockGroup,
    parts_list: Vec<Vec<SequencePart>>,
) -> Result<(), UpdateWithLibraryError> {
    let resolved = gen_graph::models::find_region_graph_positions(resolved_region, conn, 0, 0)
        .map_err(UpdateWithLibraryError::from)?;
    let start_positions = resolved.start_anchors.as_ref().unwrap();
    let end_positions = resolved.end_anchors.as_ref().unwrap();
    let (library_chunk, part_nodes) = create_library(
        conn,
        target_block_group.id,
        new_sample_name,
        parts_list,
        false,
    )?;

    let location_accession =
        create_location_accession(conn, target_block_group.id, &start_positions[0])?;
    create_part_annotations(
        conn,
        target_block_group.id,
        Some(location_accession.id),
        new_sample_name,
        new_sample_name,
        &part_nodes,
    )?;

    let start_chunk = BlockGroupChunk {
        entry_node_points: graph_node_positions_to_points(start_positions),
        exit_node_points: graph_node_positions_to_points(start_positions),
        path_edges: vec![],
        path_start_point: None,
        path_end_point: None,
    };
    let end_chunk = BlockGroupChunk {
        entry_node_points: graph_node_positions_to_points(end_positions),
        exit_node_points: graph_node_positions_to_points(end_positions),
        path_edges: vec![],
        path_start_point: None,
        path_end_point: None,
    };

    let stitched_start = stitch(conn, &start_chunk, &library_chunk, target_block_group.id)?;
    let _stitched_end = stitch(conn, &stitched_start, &end_chunk, target_block_group.id)?;
    preserve_library_boundary_points(conn, target_block_group.id, start_positions)?;
    preserve_library_boundary_points(conn, target_block_group.id, end_positions)?;

    Ok(())
}

fn operation_files(
    library_file_path: Option<&str>,
    parts_file_path: Option<&str>,
) -> Vec<OperationFile> {
    let mut files = vec![];
    if let Some(library_file_path) = library_file_path {
        files.push(OperationFile::new(library_file_path.to_string()).set_file_type(FileTypes::CSV));
    }
    if let Some(parts_file_path) = parts_file_path {
        files.push(OperationFile::new(parts_file_path.to_string()).set_file_type(FileTypes::Fasta));
    }
    files
}

fn graph_node_positions_to_points(positions: &[gen_graph::GraphNodePosition]) -> Vec<NodePoint> {
    positions
        .iter()
        .map(|position| NodePoint {
            id: position.graph_node.node_id,
            coordinate: position.coordinate(),
            strand: gen_core::Strand::Forward,
        })
        .collect()
}

/// Creates an edge at a given set of positions. Used for splitting the graph at start/end boundaries.
fn preserve_library_boundary_points(
    conn: &gen_models::db::GraphConnection,
    block_group_id: gen_core::HashId,
    positions: &[gen_graph::GraphNodePosition],
) -> Result<(), UpdateWithLibraryError> {
    let edge_data = positions
        .iter()
        .map(|position| {
            let coordinate = position.coordinate();
            gen_models::edge::EdgeData {
                source_node_id: position.graph_node.node_id,
                source_coordinate: coordinate,
                source_strand: gen_core::Strand::Forward,
                target_node_id: position.graph_node.node_id,
                target_coordinate: coordinate,
                target_strand: gen_core::Strand::Forward,
            }
        })
        .collect::<Vec<_>>();
    let edge_ids = Edge::bulk_create(conn, &edge_data);
    let block_group_edges = edge_ids
        .iter()
        .map(|edge_id| BlockGroupEdgeData {
            block_group_id,
            edge_id: *edge_id,
            chromosome_index: 0,
            phased: 0,
        })
        .collect::<Vec<_>>();
    BlockGroupEdge::bulk_create(conn, &block_group_edges);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use anyhow::Result;
    use gen_models::{
        annotations::Annotation, block_group::BlockGroup, path::Path, sample_lineage::SampleLineage,
    };

    use super::*;
    use crate::{
        graphs::combinatorial_library::parse_library, imports::fasta::import_fasta,
        test_helpers::setup_gen,
    };

    #[test]
    fn makes_a_pool() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();

        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/combinatorial_design.csv");
        let library_path = binding.to_str().unwrap();

        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123:7-20",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample", None);
        let block_group = &block_groups[0];

        let all_sequences =
            gen_graph::models::get_all_sequences_with_pruning(conn, &block_group.id, false)
                .unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAATGATAAGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAATGTTAAGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAATGCTAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATATGATAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATATGTTAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATATGCTAAGGAACACACAGAGA".to_string(),
                "ATCGATCCAACATGATAAGGAACACACAGAGA".to_string(),
                "ATCGATCCAACATGTTAAGGAACACACAGAGA".to_string(),
                "ATCGATCCAACATGCTAAGGAACACACAGAGA".to_string(),
            ])
        );

        Ok(())
    }

    #[test]
    fn one_column_of_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123:7-20",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample", None);
        let block_group = &block_groups[0];

        let all_sequences =
            gen_graph::models::get_all_sequences_with_pruning(conn, &block_group.id, false)
                .unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGATCAAAAGGAACACACAGAGA".to_string(),
                "ATCGATCTAATGGAACACACAGAGA".to_string(),
                "ATCGATCCAACGGAACACACAGAGA".to_string(),
            ])
        );

        let path = BlockGroup::get_current_path(conn, &block_group.id, None).unwrap();
        assert_eq!(
            path.sequence(conn, None).unwrap(),
            "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()
        );
        assert_eq!(
            SampleLineage::get_parents(conn, "new sample", None),
            vec![Sample::DEFAULT_NAME.to_string()]
        );

        Ok(())
    }

    #[test]
    fn annotation_update_does_not_require_target_path() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "simple",
            false,
        )
        .unwrap();
        gen_graph::models::add_annotation(
            &context,
            &collection,
            "foobar",
            None,
            "simple",
            "m123:5-20",
        )
        .unwrap();

        Sample::get_or_create_child(conn, &collection, "derived", vec!["simple".to_string()])
            .unwrap();
        let derived_block_group = crate::test_helpers::get_sample_bg(conn, &collection, "derived");
        Path::delete(conn, "m123", &derived_block_group.id);

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        update_with_library(
            &context,
            &collection,
            "simple",
            "derived",
            "foobar:-3-5",
            parts_list,
            Some(parts_path),
            Some(library_path),
        )?;

        let block_group = crate::test_helpers::get_sample_bg(conn, &collection, "derived");
        assert_eq!(
            gen_graph::models::get_all_sequences_with_pruning(conn, &block_group.id, false)
                .unwrap(),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATAAAACGATCGATCGGGAACACACAGAGA".to_string(),
                "ATTAATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCAACCGATCGATCGGGAACACACAGAGA".to_string(),
            ])
        );

        Ok(())
    }

    #[test]
    fn annotation_update_can_use_resolved_bounds() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            "simple",
            false,
        )
        .unwrap();
        gen_graph::models::add_annotation(
            &context,
            &collection,
            "foobar",
            None,
            "simple",
            "m123:5-20",
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        update_with_library(
            &context,
            &collection,
            "simple",
            "derived",
            "foobar",
            parts_list,
            Some(parts_path),
            Some(library_path),
        )?;

        let block_group = crate::test_helpers::get_sample_bg(conn, &collection, "derived");
        assert_eq!(
            gen_graph::models::get_all_sequences_with_pruning(conn, &block_group.id, false)
                .unwrap(),
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "ATCGAAAAAGGAACACACAGAGA".to_string(),
                "ATCGATAATGGAACACACAGAGA".to_string(),
                "ATCGACAACGGAACACACAGAGA".to_string(),
            ])
        );

        Ok(())
    }

    #[test]
    fn two_columns_of_same_parts() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123:7-20",
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample", None);
        let block_group = &block_groups[0];

        let mut expected_sequences = vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()];
        for part1 in ["AAAA", "TAAT", "CAAC"].iter() {
            for part2 in ["AAAA", "TAAT", "CAAC"].iter() {
                let seq = "ATCGATC".to_owned() + part1 + part2 + "GGAACACACAGAGA";
                expected_sequences.push(seq);
            }
        }
        let all_sequences =
            gen_graph::models::get_all_sequences_with_pruning(conn, &block_group.id, false)
                .unwrap();
        assert_eq!(
            all_sequences,
            expected_sequences
                .into_iter()
                .map(|x| x.to_string())
                .collect()
        );

        Ok(())
    }

    #[test]
    fn one_column_of_parts_full_replacement() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/single_column_design.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123:0-34", // Full sequence replacement
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample", None);
        let block_group = &block_groups[0];

        let all_sequences =
            gen_graph::models::get_all_sequences_with_pruning(conn, &block_group.id, false)
                .unwrap();
        assert_eq!(
            all_sequences,
            HashSet::from_iter(vec![
                "ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string(),
                "AAAA".to_string(),
                "TAAT".to_string(),
                "CAAC".to_string(),
            ])
        );

        Ok(())
    }

    #[test]
    fn two_columns_of_same_parts_full_replacement() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parts.fa");
        let parts_path = binding.to_str().unwrap();
        let binding =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/design_reusing_parts.csv");
        let library_path = binding.to_str().unwrap();
        let parts_list = parse_library(parts_path, library_path)?;

        let _ = update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "new sample",
            "m123:0-34", // Full sequence replacement
            parts_list,
            Some(parts_path),
            Some(library_path),
        );

        let block_groups = Sample::get_block_groups(conn, "test", "new sample", None);
        let block_group = &block_groups[0];

        let mut expected_sequences = vec!["ATCGATCGATCGATCGATCGGGAACACACAGAGA".to_string()];
        for part1 in ["AAAA", "TAAT", "CAAC"].iter() {
            for part2 in ["AAAA", "TAAT", "CAAC"].iter() {
                let seq = part1.to_owned().to_owned() + part2;
                expected_sequences.push(seq);
            }
        }
        let all_sequences =
            gen_graph::models::get_all_sequences_with_pruning(conn, &block_group.id, false)
                .unwrap();
        assert_eq!(
            all_sequences,
            expected_sequences
                .into_iter()
                .map(|x| x.to_string())
                .collect()
        );

        Ok(())
    }

    #[test]
    fn reusing_a_part_name_at_a_different_locus_does_not_collide() -> Result<()> {
        let context = setup_gen();
        let conn = context.graph().conn();

        let fasta_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/simple.fa");
        let collection = "test".to_string();

        import_fasta(
            &context,
            &fasta_path.to_str().unwrap().to_string(),
            &collection,
            Sample::DEFAULT_NAME,
            false,
        )
        .unwrap();

        let part = SequencePart {
            name: "p1".to_string(),
            sequence: "AAAA".to_string(),
            sequence_length: 4,
        };

        update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "lib_sample",
            "m123:5-10",
            vec![vec![part.clone()]],
            None,
            None,
        )?;

        // Same sample, a different locus, the same reused part name.
        update_with_library(
            &context,
            "test",
            Sample::DEFAULT_NAME,
            "lib_sample",
            "m123:20-25",
            vec![vec![part]],
            None,
            None,
        )?;

        let annotations = Annotation::query_by_group(conn, "lib_sample", None).unwrap();
        let names: Vec<_> = annotations.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["p1", "p1"]);

        Ok(())
    }
}
